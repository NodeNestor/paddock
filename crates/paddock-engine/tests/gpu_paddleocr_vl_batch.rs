//! PaddleOCR-VL batched serving lane gate: the paged-KV lane
//! against the same decoder oracle the serial spine gates on - batched text
//! prefill, batched multimodal slot prefill through the engine tower, the
//! radix resume, and the chunked encoder-budget admission path.
//!
//! Numeric classes: the batched lane runs `bf16_gemm` where the serial spine
//! ran `bf16_gemv`, and `rmsnorm_batch` elects its reduction width by row
//! count - separate last-ulp classes on the same f32 math. Taps are judged at
//! the serial gate's class bounds (loosened where the pass shape changes the
//! op order); greedy chains must match the oracle exactly, with the
//! manifest's top-2 margins printed on failure for near-tie judgement.

mod common;

use std::sync::Arc;

use paddock_engine::generator::Generator;
use paddock_engine::gpu_model::paddleocr_vl::GpuPaddleOcrVl;
use paddock_engine::gpu_model::paddleocr_vl::preprocess;
use paddock_engine::service::MmChunk;
use paddock_models::mapped::MappedGguf;

const DECODER: &str = "PaddleOCR-VL-1.6-GGUF/PaddleOCR-VL-1.6-GGUF.gguf";
const MMPROJ: &str = "PaddleOCR-VL-1.6-GGUF/PaddleOCR-VL-1.6-GGUF-mmproj.gguf";
const IMAGE_TOKEN: u32 = 100295;
const EOS: u32 = 2;

fn find_model(rel: &str, env: &str) -> Option<std::path::PathBuf> {
    if let Ok(p) = std::env::var(env) {
        let p = std::path::PathBuf::from(p);
        return p.exists().then_some(p);
    }
    common::model_roots().iter().find_map(|r| {
        let p = r.join(rel);
        p.exists().then_some(p)
    })
}

fn oracle_dir() -> Option<std::path::PathBuf> {
    common::model_roots()
        .iter()
        .map(|r| r.join("ocr-battery").join("paddle-oracle"))
        .find(|p| p.join("manifest_dec.json").exists())
}

fn manifest(dir: &std::path::Path) -> serde_json::Value {
    let text = std::fs::read_to_string(dir.join("manifest_dec.json")).expect("manifest_dec");
    serde_json::from_str(&text).expect("manifest_dec json")
}

fn read_f32(path: &std::path::Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn read_i32(path: &std::path::Path) -> Vec<i32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    bytes
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn rel_l2(got: &[f32], want: &[f32]) -> f64 {
    assert_eq!(got.len(), want.len(), "tensor size mismatch");
    let (mut d2, mut w2) = (0f64, 0f64);
    for (&g, &w) in got.iter().zip(want) {
        let d = g as f64 - w as f64;
        d2 += d * d;
        w2 += (w as f64) * (w as f64);
    }
    (d2 / w2.max(1e-30)).sqrt()
}

fn argmax(v: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best as u32
}

struct Probe {
    ids: Vec<u32>,
    greedy: Vec<u32>,
    margins: serde_json::Value,
}

fn load_probe(dir: &std::path::Path, m: &serde_json::Value, tag: &str) -> Probe {
    let ids: Vec<u32> = read_i32(&dir.join(format!("dec_ids_{tag}.bin")))
        .iter()
        .map(|&v| v as u32)
        .collect();
    let p = &m["probes"][tag];
    assert_eq!(ids.len(), p["seq"].as_u64().unwrap() as usize);
    Probe {
        ids,
        greedy: p["greedy_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect(),
        margins: p["greedy_margins"].clone(),
    }
}

/// Greedy continuation through the serial surface routed onto slot 0 of the
/// batch lane (Generator::forward - this is exactly what exercises the
/// per-slot M-RoPE delta on decode).
fn greedy_slot0(model: &mut GpuPaddleOcrVl, last_logits: &[f32], steps: usize) -> Vec<u32> {
    let mut out = Vec::with_capacity(steps);
    let mut logits = last_logits.to_vec();
    for _ in 0..steps {
        let tok = argmax(&logits);
        out.push(tok);
        if tok == EOS {
            break;
        }
        logits = Generator::forward(model, tok).expect("slot0 decode");
    }
    out
}

/// The m_a700 probe as serving-shaped chunks: leading template text, the raw
/// 700×500 hash-pixel image, the trailing task text. The plan must expand the
/// image to exactly the oracle's placeholder run.
fn m_a700_chunks(ids: &[u32]) -> Vec<MmChunk> {
    let start = ids
        .iter()
        .position(|&t| t == IMAGE_TOKEN)
        .expect("image run");
    let n = ids[start..]
        .iter()
        .take_while(|&&t| t == IMAGE_TOKEN)
        .count();
    assert!(
        !ids[start + n..].contains(&IMAGE_TOKEN),
        "probe has one image run by construction"
    );
    vec![
        MmChunk::Text(ids[..start].to_vec()),
        MmChunk::Image {
            rgb: preprocess::hash_pixels(700, 500, 1),
            w: 700,
            h: 500,
        },
        MmChunk::Text(ids[start + n..].to_vec()),
    ]
}

fn load_batched(
    exec: &Arc<paddock_engine::gpu::GpuExecutor>,
    dec: &std::path::Path,
    mm: Option<&std::path::Path>,
    slots: usize,
) -> GpuPaddleOcrVl {
    let map = MappedGguf::open(dec).expect("open decoder gguf");
    let mut model = GpuPaddleOcrVl::load(Arc::clone(exec), &map, 4096).expect("load");
    if let Some(mm) = mm {
        let mm_map = MappedGguf::open(mm).expect("open mmproj gguf");
        model.attach_vision(&mm_map).expect("attach vision");
    }
    let got = model
        .enable_batch(slots)
        .expect("enable_batch (paged-KV pack required)");
    assert_eq!(got, slots);
    model
}

/// Batched TEXT prefill on a non-zero slot + batched decode: logits at the
/// gemm class against the oracle, greedy exact via batch_step_slots.
#[test]
fn batched_text_prefill_matches_the_oracle() {
    let Some(dir) = oracle_dir() else {
        common::missing("no PaddleOCR-VL decoder oracle");
        return;
    };
    let Some(path) = find_model(DECODER, "PADDLEOCR_VL_GGUF") else {
        common::missing("no PaddleOCR-VL decoder GGUF");
        return;
    };
    let Some(exec) = common::gpu_arc() else {
        return;
    };

    let m = manifest(&dir);
    let p = load_probe(&dir, &m, "t_sv");
    let mut model = load_batched(&exec, &path, None, 4);

    // slot 2 deliberately - a non-zero slot exercises the block-table striding
    let logits = model.forward_prefill(2, &p.ids).expect("batched prefill");
    let want = read_f32(&dir.join("dec_logits_last_t_sv.bin"));
    let lr = rel_l2(&logits, &want);
    eprintln!("batched t_sv logits relL2 {lr:.3e}");
    // 8e-3, not the old 2e-3: the prefill band rides the bf16 tensor-core arm
    // which casts the f32 activations to bf16 in its smem stage -
    // the same class llama.cpp's batched BF16 path computes (cublasGemmEx
    // bf16xbf16). 18 layers of that cast measure 4.1e-3 against this f32
    // oracle dump (the f32-FMA tile measured under 2e-3). The token-level
    // contract is the greedy-exact assert below, not this norm.
    assert!(lr < 8e-3, "batched t_sv logits relL2 {lr:.3e}");

    // greedy through the batched decode step at the slot (text delta = 0)
    let mut greedy = Vec::new();
    let mut logits = logits;
    let mut pos = p.ids.len() as u32;
    for _ in 0..p.greedy.len() {
        let tok = argmax(&logits);
        greedy.push(tok);
        if tok == EOS {
            break;
        }
        model
            .batch_step_slots(&[tok], &[pos], &[2])
            .expect("decode step");
        logits = model.read_batch_logits(1).expect("logits");
        pos += 1;
    }
    assert_eq!(
        greedy, p.greedy,
        "batched t_sv greedy diverged; margins: {}",
        p.margins
    );
}

/// Batched MULTIMODAL slot prefill through the engine tower: row count,
/// full-chain logits class, greedy exact through the slot-0 serial surface
/// (which is what proves the decode-side M-RoPE delta).
#[test]
fn batched_multimodal_slot_matches_the_oracle() {
    let Some(dir) = oracle_dir() else {
        common::missing("no PaddleOCR-VL decoder oracle");
        return;
    };
    let Some(dec) = find_model(DECODER, "PADDLEOCR_VL_GGUF") else {
        common::missing("no PaddleOCR-VL decoder GGUF");
        return;
    };
    let Some(mm) = find_model(MMPROJ, "PADDLEOCR_VL_MMPROJ") else {
        common::missing("no PaddleOCR-VL mmproj GGUF");
        return;
    };
    let Some(exec) = common::gpu_arc() else {
        return;
    };

    let m = manifest(&dir);
    let p = load_probe(&dir, &m, "m_a700");
    let mut model = load_batched(&exec, &dec, Some(&mm), 2);

    let chunks = m_a700_chunks(&p.ids);
    let (logits, rows) =
        Generator::forward_prefill_multimodal(&mut model, 0, &chunks).expect("mm slot prefill");
    assert_eq!(
        rows,
        p.ids.len(),
        "row count = the whole interleaved stream"
    );

    let want = read_f32(&dir.join("dec_logits_last_m_a700.bin"));
    let lr = rel_l2(&logits, &want);
    eprintln!("batched m_a700 logits relL2 {lr:.3e}");
    assert!(lr < 2e-2, "batched m_a700 logits relL2 {lr:.3e}");

    let greedy = greedy_slot0(&mut model, &logits, p.greedy.len());
    assert_eq!(
        greedy, p.greedy,
        "batched m_a700 greedy diverged; margins: {}",
        p.margins
    );
}

/// Same image prompt twice: the second pass must resume off the radix
/// (reused rows reported) and reproduce the cold pass's greedy chain - the
/// "same page must parse identically with and without the cache" contract.
#[test]
fn radix_resume_reproduces_the_cold_pass() {
    let Some(dir) = oracle_dir() else {
        common::missing("no PaddleOCR-VL decoder oracle");
        return;
    };
    let Some(dec) = find_model(DECODER, "PADDLEOCR_VL_GGUF") else {
        common::missing("no PaddleOCR-VL decoder GGUF");
        return;
    };
    let Some(mm) = find_model(MMPROJ, "PADDLEOCR_VL_MMPROJ") else {
        common::missing("no PaddleOCR-VL mmproj GGUF");
        return;
    };
    let Some(exec) = common::gpu_arc() else {
        return;
    };

    let m = manifest(&dir);
    let p = load_probe(&dir, &m, "m_a700");
    let mut model = load_batched(&exec, &dec, Some(&mm), 2);
    let chunks = m_a700_chunks(&p.ids);

    let (cold_logits, _) =
        Generator::forward_prefill_multimodal(&mut model, 0, &chunks).expect("cold prefill");
    assert_eq!(
        model.take_prefill_reused(0),
        0,
        "cold pass must not report reuse"
    );
    let cold = greedy_slot0(&mut model, &cold_logits, p.greedy.len());
    assert_eq!(
        cold, p.greedy,
        "cold greedy diverged; margins: {}",
        p.margins
    );

    let (warm_logits, _) =
        Generator::forward_prefill_multimodal(&mut model, 1, &chunks).expect("warm prefill");
    let reused = model.take_prefill_reused(1);
    eprintln!("radix resume reused {reused} of {} rows", p.ids.len());
    assert!(
        reused >= 16,
        "second pass should resume off the radix (got {reused})"
    );
    // decode the warm slot through the batched step directly
    let mut warm = Vec::new();
    let mut logits = warm_logits;
    let mut pos = p.ids.len() as u32;
    for _ in 0..p.greedy.len() {
        let tok = argmax(&logits);
        warm.push(tok);
        if tok == EOS {
            break;
        }
        model
            .batch_step_slots(&[tok], &[pos], &[1])
            .expect("decode step");
        logits = model.read_batch_logits(1).expect("logits");
        pos += 1;
    }
    assert_eq!(warm, cold, "resumed pass diverged from its own cold pass");
}

/// The chunked encoder-budget admission: Encoding -> encode_step ticks ->
/// Queued -> mixed ticks drain the row plan -> the finisher matches the oracle
/// and decode continues at the delta positions.
#[test]
fn chunked_mm_admission_completes_and_matches() {
    let Some(dir) = oracle_dir() else {
        common::missing("no PaddleOCR-VL decoder oracle");
        return;
    };
    let Some(dec) = find_model(DECODER, "PADDLEOCR_VL_GGUF") else {
        common::missing("no PaddleOCR-VL decoder GGUF");
        return;
    };
    let Some(mm) = find_model(MMPROJ, "PADDLEOCR_VL_MMPROJ") else {
        common::missing("no PaddleOCR-VL mmproj GGUF");
        return;
    };
    let Some(exec) = common::gpu_arc() else {
        return;
    };

    let m = manifest(&dir);
    let p = load_probe(&dir, &m, "m_a700");
    let mut model = load_batched(&exec, &dec, Some(&mm), 2);
    assert!(Generator::supports_chunked_multimodal(&model));

    let chunks = m_a700_chunks(&p.ids);
    let verdicts = Generator::prefill_begin_multimodal(&mut model, vec![(0, chunks)]);
    assert_eq!(verdicts.len(), 1);
    let mut queued = matches!(verdicts[0].1, paddock_engine::generator::MmAdmit::Queued);
    if let paddock_engine::generator::MmAdmit::Failed(e) = &verdicts[0].1 {
        panic!("admission failed: {e}");
    }
    // spend encoder budgets until the entry queues (one image -> few ticks).
    // Real scheduler ticks are ms apart; pace the loop so the prep worker's
    // Wait verdicts get the wall time they'd get in a serve.
    let mut ticks = 0;
    while !queued {
        assert!(
            Generator::encoding_pending(&model),
            "entry vanished without a report"
        );
        for (slot, v) in Generator::encode_step(&mut model) {
            assert_eq!(slot, 0);
            match v {
                paddock_engine::generator::MmAdmit::Queued => queued = true,
                paddock_engine::generator::MmAdmit::Failed(e) => panic!("encode failed: {e}"),
                _ => {}
            }
        }
        ticks += 1;
        assert!(ticks < 400, "encoder budget never finished");
        if !queued {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
    }
    assert!(!Generator::encoding_pending(&model));

    // drain the queued row plan through mixed ticks (no decode rows)
    let mut finished: Option<(Vec<f32>, usize)> = None;
    let mut passes = 0;
    while finished.is_none() {
        let (_dec, fin) = Generator::forward_mixed(&mut model, &[], 1024).expect("mixed tick");
        for (slot, logits, rows) in fin {
            assert_eq!(slot, 0);
            finished = Some((logits, rows));
        }
        passes += 1;
        assert!(passes < 64, "chunked prefill never finished");
    }
    let (logits, rows) = finished.expect("finisher");
    assert_eq!(rows, p.ids.len());
    let want = read_f32(&dir.join("dec_logits_last_m_a700.bin"));
    let lr = rel_l2(&logits, &want);
    eprintln!("chunked m_a700 logits relL2 {lr:.3e} ({ticks} encode ticks, {passes} mixed passes)");
    assert!(lr < 2e-2, "chunked m_a700 logits relL2 {lr:.3e}");

    let greedy = greedy_slot0(&mut model, &logits, p.greedy.len());
    assert_eq!(
        greedy, p.greedy,
        "chunked m_a700 greedy diverged; margins: {}",
        p.margins
    );
}
