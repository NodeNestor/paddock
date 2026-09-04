//! Decoder parity gate: our Q8_0 serial forward vs the reference
//! oracle (our OCR oracle tool - the checkpoint's own
//! modeling code, f32, ring armed), through all three R-SWA phases.
//!
//! TEACHER-FORCED replay: the oracle's greedy tokens are fed regardless of our
//! own argmax, so every step's comparison stays on the oracle's trajectory and
//! one near-tie flip cannot cascade into 190 meaningless mismatches. The ring
//! region (steps ≥ 128, where steady-state overwrites run) is gated
//! separately from the warmup so a broken ring cannot hide in the average.
//!
//! Numeric class: Q8_0 weights + our kernel ladder vs an f32 oracle. The gate
//! values are measured class behavior with headroom; a graph bug - wrong
//! router denominator, a rope off-by-one, a ring slot writing the wrong row -
//! moves these by orders, not points.
//!
//! Arrays ride flat .bin sidecars; only scalars are read from the JSON, with
//! the same hand scan `asr_mel_oracle` uses - no serde in the engine's tests.
// Test code: a failed assumption stops the test where it happened.
#![allow(clippy::unwrap_used)]

mod common;

use std::sync::Arc;

use paddock_engine::gpu_model::deepseek_ocr::GpuDeepseekOcr;
use paddock_models::mapped::MappedGguf;

fn oracle_dir() -> Option<std::path::PathBuf> {
    common::model_roots()
        .iter()
        .map(|r| r.join("ocr-battery").join("oracle"))
        .find(|p| p.join("decoder.json").exists())
}

fn model_path() -> Option<std::path::PathBuf> {
    if let Ok(p) = std::env::var("UNLIMITED_OCR_GGUF") {
        let p = std::path::PathBuf::from(p);
        return p.exists().then_some(p);
    }
    common::model_roots().iter().find_map(|r| {
        let p = r.join("Unlimited-OCR-GGUF").join("Unlimited-OCR-Q8_0.gguf");
        p.exists().then_some(p)
    })
}

fn read_u32s(p: &std::path::Path) -> Vec<u32> {
    std::fs::read(p)
        .unwrap_or_else(|e| panic!("{}: {e}", p.display()))
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect()
}

fn read_f32s(p: &std::path::Path) -> Vec<f32> {
    std::fs::read(p)
        .unwrap_or_else(|e| panic!("{}: {e}", p.display()))
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

/// Scalar out of the oracle JSON - the `asr_mel_oracle` scan.
fn json_usize(txt: &str, key: &str) -> usize {
    let s = &txt[txt.find(&format!("\"{key}\":")).unwrap() + key.len() + 3..];
    s[..s.find([',', '}', '\n']).unwrap()]
        .trim()
        .parse()
        .unwrap()
}

fn cos_rel(got: &[f32], want: &[f32]) -> (f64, f64) {
    let (mut d2, mut w2, mut g2, mut dot) = (0f64, 0f64, 0f64, 0f64);
    for (&g, &w) in got.iter().zip(want) {
        let d = (g - w) as f64;
        d2 += d * d;
        w2 += (w as f64) * (w as f64);
        g2 += (g as f64) * (g as f64);
        dot += (g as f64) * (w as f64);
    }
    (dot / (w2.sqrt() * g2.sqrt()), (d2 / w2).sqrt())
}

fn argmax(v: &[f32]) -> u32 {
    let mut bi = 0usize;
    for (i, &x) in v.iter().enumerate() {
        if x > v[bi] {
            bi = i;
        }
    }
    bi as u32
}

#[test]
fn decoder_matches_the_oracle_through_the_ring() {
    let Some(dir) = oracle_dir() else {
        common::missing("no decoder oracle");
        return;
    };
    let Some(path) = model_path() else {
        common::missing("no Unlimited-OCR GGUF (set UNLIMITED_OCR_GGUF)");
        return;
    };
    let Some(exec) = common::gpu_arc() else {
        return;
    };

    let meta = std::fs::read_to_string(dir.join("decoder.json")).expect("decoder.json");
    let ring_window = json_usize(&meta, "ring_window");
    assert_eq!(
        json_usize(&meta, "final_cache_len"),
        json_usize(&meta, "expect_cache_len"),
        "the ORACLE's own ring did not engage - regenerate it"
    );
    let prompt = read_u32s(&dir.join("decoder_prompt_ids.bin"));
    let greedy = read_u32s(&dir.join("decoder_greedy_ids.bin"));
    let t5_ids = read_u32s(&dir.join("decoder_top5_ids.bin"));
    let want_logits = read_f32s(&dir.join("decoder_prefill_logits.bin"));
    let steps = greedy.len();
    assert_eq!(t5_ids.len(), steps * 5);

    let map = MappedGguf::open(&path).expect("open gguf");
    let mut m = GpuDeepseekOcr::load(Arc::clone(&exec), &map, 32_768).expect("load decoder");
    assert_eq!(m.hp.rswa_window, Some(ring_window));

    // --- prefill: token-by-token, logits at the last prompt position.
    let mut logits = Vec::new();
    for &t in &prompt {
        logits = m.forward_one(t).expect("prefill step");
    }
    assert_eq!(logits.len(), want_logits.len());
    // triage aid: PADDOCK_OCR_DEC_DUMP=1 prints our accumulated per-layer
    // sums, directly comparable to the oracle's "layer-i" prefill stages
    if std::env::var_os("PADDOCK_OCR_DEC_DUMP").is_some()
        && let Some(sums) = m.layer_sums()
    {
        for (i, s) in sums.iter().enumerate() {
            eprintln!("ours layer-{i}: {s:.4}");
        }
    }
    let (cos, rel) = cos_rel(&logits, &want_logits);
    // The gate anchors on the HEAD of the distribution, not the full vector:
    // 129k logits on a garbage LCG prompt are mostly near-zero noise where
    // Q8-vs-f32 relative error is meaningless, so full-vector cos sits at
    // ~0.982 while the top-5 ids AND values agree within 0.3. What decodes is
    // the head; that is what must match.
    let mut widx: Vec<usize> = (0..want_logits.len()).collect();
    widx.sort_by(|&a, &b| want_logits[b].partial_cmp(&want_logits[a]).unwrap());
    let head_dmax = widx[..50]
        .iter()
        .map(|&i| (logits[i] - want_logits[i]).abs())
        .fold(0f32, f32::max);
    eprintln!("prefill logits: cos {cos:.6} relL2 {rel:.5} head50 max|Δ| {head_dmax:.4}");
    assert!(
        cos > 0.97,
        "prefill logits cosine {cos} (class-measured 0.982)"
    );
    assert!(head_dmax < 0.5, "top-50 logit drift {head_dmax}");
    assert_eq!(
        argmax(&logits),
        greedy[0],
        "the very first greedy token disagrees - a graph bug, not quant noise"
    );

    m.note_prefill_end();

    // --- teacher-forced decode through warmup and the ring.
    let ring_from = ring_window; // steps >= W run steady-state overwrites
    let (mut agree_warm, mut agree_ring) = (0usize, 0usize);
    let (mut t5_warm, mut t5_ring) = (0usize, 0usize);
    let mut first_miss = None;
    for s in 0..steps {
        let l = m.forward_one(greedy[s]).expect("decode step");
        let oracle5 = &t5_ids[s * 5..s * 5 + 5];
        let ours = argmax(&l);
        let top1_ok = ours == oracle5[0];
        // top-5 containment is the near-tie-tolerant signal (margin lesson:
        // a hot runner-up is the model's coin flip, not a defect)
        let in_t5 = oracle5.contains(&ours);
        if s < ring_from {
            agree_warm += top1_ok as usize;
            t5_warm += in_t5 as usize;
        } else {
            agree_ring += top1_ok as usize;
            t5_ring += in_t5 as usize;
        }
        if !top1_ok && first_miss.is_none() {
            first_miss = Some((s, ours, oracle5.to_vec()));
        }
    }
    let ring_steps = steps - ring_from;
    eprintln!(
        "warmup: top1 {agree_warm}/{ring_from}  top5 {t5_warm}/{ring_from}   \
         ring: top1 {agree_ring}/{ring_steps}  top5 {t5_ring}/{ring_steps}"
    );
    if let Some((s, ours, ids)) = first_miss {
        eprintln!("first top-1 miss at step {s}: ours {ours}, oracle top5 {ids:?}");
    }
    // Class gates. The ring region gates separately at the same bar: a ring
    // writing wrong rows collapses agreement there outright, while Q8-vs-f32
    // near-tie flips cost isolated steps on both sides of the boundary alike.
    let warm_frac = agree_warm as f64 / ring_from as f64;
    let ring_frac = agree_ring as f64 / ring_steps as f64;
    assert!(warm_frac >= 0.95, "warmup top-1 agreement {warm_frac:.3}");
    assert!(ring_frac >= 0.95, "ring top-1 agreement {ring_frac:.3}");
    assert!(
        t5_warm as f64 / ring_from as f64 >= 0.99,
        "warmup top-5 containment"
    );
    assert!(
        t5_ring as f64 / ring_steps as f64 >= 0.99,
        "ring top-5 containment"
    );

    // The family's point, checked through the public API: output length does
    // not move the KV row count once the ring engages.
    assert_eq!(m.kv_rows(prompt.len(), steps), prompt.len() + ring_window);
}
