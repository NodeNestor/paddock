//! PaddleOCR-VL ERNIE decoder oracle gate: loader ledger, the
//! get_rope_index port integer-exact, and the serial forward against the
//! checkpoint's own modeling_paddleocr_vl.py (f32, eager, CUDA) on two
//! probes - a text-only chat prompt and a real image prompt through the
//! reference processor.
//!
//! Oracle artifacts come from an out-of-tree dump tool
//! into `<models>/ocr-battery/paddle-oracle/` (dec_* files). Skips cleanly
//! when absent; fails under `PADDOCK_STRICT_GATES=1`.
//!
//! Numeric classes:
//! * positions and input ids are exact (integers);
//! * the input-embedding tap is effectively exact - the engine's bf16 planes
//!   hold literally the oracle's weights (bf16 widens to f32 losslessly and
//!   the oracle model was loaded f32 from the same bf16 checkpoint), so the
//!   gather is the same widen and the image rows are the oracle's own
//!   projector output fed back;
//! * everything through the stack is CLASS tolerance - f32 activation math
//!   in a different op order, 18 layers deep. A real graph bug (wrong eps,
//!   swapped gate/up, wrong section walk, decoupled-head_dim mixups) moves
//!   relL2 by orders of magnitude, not percent.
//! * greedy ids must match exactly; the manifest carries per-step top-2
//!   margins so a near-tie flip can be judged as the model's coin toss
//!   (near-tie-margin discipline) rather than silently absorbed.
// Test code: a failed assumption stops the test where it happened.
#![allow(clippy::unwrap_used)]

mod common;

use std::sync::Arc;

use paddock_engine::gpu_model::paddleocr_vl::GpuPaddleOcrVl;
use paddock_engine::gpu_model::paddleocr_vl::forward::{MmGrid, build_positions};
use paddock_engine::gpu_model::paddleocr_vl::preprocess;
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
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

fn read_i32(path: &std::path::Path) -> Vec<i32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| i32::from_le_bytes(*c))
        .collect()
}

fn read_i64(path: &std::path::Path) -> Vec<i64> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    bytes
        .as_chunks::<8>()
        .0
        .iter()
        .map(|c| i64::from_le_bytes(*c))
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

struct ProbeData {
    ids: Vec<u32>,
    pos: Vec<i64>,
    grids: Vec<MmGrid>,
    greedy: Vec<u32>,
    margins: serde_json::Value,
}

fn load_probe(dir: &std::path::Path, m: &serde_json::Value, tag: &str) -> ProbeData {
    let ids: Vec<u32> = read_i32(&dir.join(format!("dec_ids_{tag}.bin")))
        .iter()
        .map(|&v| v as u32)
        .collect();
    let p = &m["probes"][tag];
    assert_eq!(ids.len(), p["seq"].as_u64().unwrap() as usize);
    let grids = p["image_grid_thw"]
        .as_array()
        .map(|imgs| {
            imgs.iter()
                .map(|g| {
                    let (t, gh, gw) = (
                        g[0].as_u64().unwrap(),
                        g[1].as_u64().unwrap() as usize,
                        g[2].as_u64().unwrap() as usize,
                    );
                    assert_eq!(t, 1, "images only");
                    // patch grid -> merged decoder-token grid (2x2)
                    MmGrid {
                        ny: gh / 2,
                        nx: gw / 2,
                    }
                })
                .collect()
        })
        .unwrap_or_default();
    ProbeData {
        ids,
        pos: read_i64(&dir.join(format!("dec_pos_{tag}.bin"))),
        grids,
        greedy: p["greedy_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect(),
        margins: p["greedy_margins"].clone(),
    }
}

/// The get_rope_index port must reproduce the reference positions
/// integer-for-integer on both probes - pure host math, no GPU needed.
#[test]
fn rope_index_port_matches_the_reference() {
    let Some(dir) = oracle_dir() else {
        common::missing("no PaddleOCR-VL decoder oracle");
        return;
    };
    let m = manifest(&dir);
    for tag in ["t_sv", "m_a700"] {
        let p = load_probe(&dir, &m, tag);
        let pos = build_positions(&p.ids, IMAGE_TOKEN, &p.grids).expect("positions");
        let seq = p.ids.len();
        assert_eq!(p.pos.len(), 3 * seq, "{tag}: pos dump shape");
        for i in 0..seq {
            assert_eq!(pos.t[i] as i64, p.pos[i], "{tag}: t axis at {i}");
            assert_eq!(pos.h[i] as i64, p.pos[seq + i], "{tag}: h axis at {i}");
            assert_eq!(pos.w[i] as i64, p.pos[2 * seq + i], "{tag}: w axis at {i}");
        }
        let pos_max = m["probes"][tag]["pos_max"].as_i64().unwrap();
        assert_eq!(pos.next as i64, pos_max + 1, "{tag}: decode continuation");
    }
}

#[test]
fn loads_the_decoder_and_geometry_matches_the_design() {
    let Some(path) = find_model(DECODER, "PADDLEOCR_VL_GGUF") else {
        common::missing("no PaddleOCR-VL decoder GGUF (set PADDLEOCR_VL_GGUF)");
        return;
    };
    let Some(exec) = common::gpu_arc() else {
        return;
    };

    let map = MappedGguf::open(&path).expect("open gguf");
    let m = GpuPaddleOcrVl::load(Arc::clone(&exec), &map, 8192).expect("load decoder");

    assert_eq!(m.hp.n_layer, 18);
    assert_eq!(m.hp.n_embd, 1024);
    assert_eq!(m.hp.n_head, 16);
    assert_eq!(m.hp.n_kv_heads, 2);
    assert_eq!(
        m.hp.head_dim, 128,
        "decoupled - from attention.key_length, never n_embd/n_head"
    );
    assert_eq!(m.hp.n_ff, 3072);
    assert_eq!(
        m.hp.n_vocab, 103_424,
        "measured off token_embd, the header has no vocab key"
    );
    assert_eq!(m.hp.sections, [16, 24, 24, 0]);
    assert_eq!(m.hp.n_rot, 128, "full-head rotation");
    assert!((m.hp.eps - 1e-5).abs() < 1e-9);
    assert!((m.hp.rope_base - 500_000.0).abs() < 1.0);
    assert_eq!(m.hp.n_ctx_train, 131_072);

    // BF16 planes resident verbatim: the ledger must land at ~file size
    // (the file is weights + ~2 MB of tokenizer metadata, no repack overhead)
    let file_gb = std::fs::metadata(&path).expect("stat").len() as f64 / 1e9;
    let resident_gb = m.weights_bytes as f64 / 1e9;
    assert!(
        resident_gb > 0.8 * file_gb && resident_gb < 1.2 * file_gb,
        "decoder resident {resident_gb:.2} GB vs file {file_gb:.2} GB"
    );
    eprintln!("decoder resident {resident_gb:.2} GB (file {file_gb:.2} GB)");
}

/// Text probe: taps at every dumped depth, last-row logits, greedy chain.
#[test]
fn text_probe_matches_the_reference() {
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
    let map = MappedGguf::open(&path).expect("open gguf");
    let mut model = GpuPaddleOcrVl::load(Arc::clone(&exec), &map, 4096).expect("load");

    let taps = model
        .prefill_taps(&p.ids, IMAGE_TOKEN, None, &[], &[0, 3, 9, 17])
        .expect("prefill");
    check_probe(&dir, "t_sv", &taps, &mut model, &p);
}

/// Multimodal probe with the ORACLE's projector rows spliced in - isolates
/// the decoder: any failure here is decoder graph, not tower.
#[test]
fn multimodal_probe_matches_the_reference() {
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
    let p = load_probe(&dir, &m, "m_a700");
    let n_img = m["probes"]["m_a700"]["n_image_tokens"].as_u64().unwrap() as usize;

    // probe a's projector output from the VISION oracle - same image
    let proj = read_f32(&dir.join("proj_a_700x500.bin"));
    assert_eq!(proj.len(), n_img * 1024, "projector plane shape");

    let map = MappedGguf::open(&path).expect("open gguf");
    let mut model = GpuPaddleOcrVl::load(Arc::clone(&exec), &map, 4096).expect("load");
    let d_proj = exec.stream.clone_htod(&proj).expect("upload proj");

    let taps = model
        .prefill_taps(&p.ids, IMAGE_TOKEN, Some(&d_proj), &p.grids, &[0, 3, 9, 17])
        .expect("prefill");
    check_probe(&dir, "m_a700", &taps, &mut model, &p);
}

/// Full chain: our tower encodes probe a, its output feeds the decoder.
/// Slightly looser logits class (the tower's own f16-plane distance rides
/// on top), greedy must still agree.
#[test]
fn engine_tower_feeds_the_decoder() {
    let Some(dir) = oracle_dir() else {
        common::missing("no PaddleOCR-VL decoder oracle");
        return;
    };
    let Some(dec_path) = find_model(DECODER, "PADDLEOCR_VL_GGUF") else {
        common::missing("no PaddleOCR-VL decoder GGUF");
        return;
    };
    let Some(mm_path) = find_model(MMPROJ, "PADDLEOCR_VL_MMPROJ") else {
        common::missing("no PaddleOCR-VL mmproj GGUF");
        return;
    };
    let Some(exec) = common::gpu_arc() else {
        return;
    };

    let m = manifest(&dir);
    let p = load_probe(&dir, &m, "m_a700");

    let map = MappedGguf::open(&dec_path).expect("open decoder gguf");
    let mut model = GpuPaddleOcrVl::load(Arc::clone(&exec), &map, 4096).expect("load");
    let mm_map = MappedGguf::open(&mm_path).expect("open mmproj gguf");
    model.attach_vision(&mm_map).expect("attach vision");

    // probe a end to end: hash pixels -> bit-exact preprocess -> our tower
    let rgb = preprocess::hash_pixels(700, 500, 1);
    let (planar, tw, th) =
        preprocess::preprocess_rgb(&rgb, 700, 500, preprocess::PixelBudget::DEFAULT)
            .expect("preprocess");
    let vout = model
        .vision
        .as_mut()
        .unwrap()
        .encode(&planar, tw, th)
        .expect("encode");
    assert_eq!(
        (vout.ny, vout.nx),
        (p.grids[0].ny, p.grids[0].nx),
        "tower grid"
    );

    let taps = model
        .prefill_taps(&p.ids, IMAGE_TOKEN, Some(&vout.embd), &p.grids, &[])
        .expect("prefill");

    let want_logits = read_f32(&dir.join("dec_logits_last_m_a700.bin"));
    let lr = rel_l2(&taps.last_logits, &want_logits);
    eprintln!("full-chain logits relL2 {lr:.3e}");
    assert!(lr < 2e-2, "full-chain logits relL2 {lr:.3e}");

    let greedy = model
        .greedy(&taps.last_logits, p.greedy.len(), EOS)
        .expect("greedy");
    assert_eq!(
        greedy, p.greedy,
        "full-chain greedy diverged; margins: {}",
        p.margins
    );
}

/// Shared oracle comparison: embd near-exact, stack taps + logits at class,
/// greedy exact.
fn check_probe(
    dir: &std::path::Path,
    tag: &str,
    taps: &paddock_engine::gpu_model::paddleocr_vl::forward::DecTaps,
    model: &mut GpuPaddleOcrVl,
    p: &ProbeData,
) {
    // the input rows are the same bf16 widen / the same spliced plane -
    // this failing at all means gather or splice order, not numerics
    let want = read_f32(&dir.join(format!("dec_embd_{tag}.bin")));
    let r = rel_l2(&taps.embd, &want);
    assert!(r < 1e-6, "{tag}: embd relL2 {r:.3e}");

    for li in [0usize, 3, 9, 17] {
        let want = read_f32(&dir.join(format!("dec_layer{li}_{tag}.bin")));
        let r = rel_l2(&taps.layers[&li], &want);
        eprintln!("{tag}: layer{li} relL2 {r:.3e}");
        assert!(r < 5e-4, "{tag}: layer{li} relL2 {r:.3e}");
    }
    // the final-norm tap runs a shade above the layer class on SHORT probes:
    // RMSNorm rescales per row, so low-magnitude rows (BOS, template
    // punctuation) carry amplified relative error that a 19-row probe can't
    // average away (measured 5.7e-4 on t_sv vs 3.4e-4 on the 463-row
    // m_a700, layers at 0.6-1.4e-4 in the same runs). The logits row below
    // stays at the tight class - that's the one that decides tokens.
    let want = read_f32(&dir.join(format!("dec_norm_{tag}.bin")));
    let r = rel_l2(&taps.norm, &want);
    eprintln!("{tag}: norm relL2 {r:.3e}");
    assert!(r < 1.2e-3, "{tag}: norm relL2 {r:.3e}");

    let want = read_f32(&dir.join(format!("dec_logits_last_{tag}.bin")));
    let r = rel_l2(&taps.last_logits, &want);
    eprintln!("{tag}: logits relL2 {r:.3e}");
    assert!(r < 5e-4, "{tag}: logits relL2 {r:.3e}");

    let greedy = model
        .greedy(&taps.last_logits, p.greedy.len(), EOS)
        .expect("greedy");
    assert_eq!(
        greedy, p.greedy,
        "{tag}: greedy diverged; oracle margins: {}",
        p.margins
    );
}
