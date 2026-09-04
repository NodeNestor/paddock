//! Batched-lane gate: the paged pool, the R-SWA ring on the pool,
//! and the radix prefix cache, all against the same reference oracle the
//! serial gate uses - plus the two claims no oracle can state:
//!
//!  * the POOL FOOTPRINT PINS: after 200 generated tokens a slot owns exactly
//!    ⌈(prefill + W)/16⌉ blocks, not ⌈(prefill + 200)/16⌉ - the family's
//!    whole point, asserted on the block table itself;
//!  * a SECOND slot prefilling the same prompt resumes off the radix and
//!    produces the same head logits - the prefix cache staying on is the
//!    competitive opening, so it is gated, not assumed.
//!
//! The final leg decodes both slots in one r=2 tick with one deep in ring
//! steady state and the other in warmup - the mixed-phase case a per-slot
//! ring must not cross-contaminate.
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

fn json_usize(txt: &str, key: &str) -> usize {
    let s = &txt[txt.find(&format!("\"{key}\":")).unwrap() + key.len() + 3..];
    s[..s.find([',', '}', '\n']).unwrap()]
        .trim()
        .parse()
        .unwrap()
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
fn pool_ring_and_prefix_match_the_oracle() {
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
    let w = json_usize(&meta, "ring_window");
    let prompt = read_u32s(&dir.join("decoder_prompt_ids.bin"));
    let greedy = read_u32s(&dir.join("decoder_greedy_ids.bin"));
    let t5_ids = read_u32s(&dir.join("decoder_top5_ids.bin"));
    let steps = greedy.len();
    let pf = prompt.len();

    let map = MappedGguf::open(&path).expect("open gguf");
    let mut m = GpuDeepseekOcr::load(Arc::clone(&exec), &map, 32_768).expect("load decoder");
    let slots = m.enable_batch(2).expect("enable_batch");
    assert_eq!(slots, 2, "pool must enable at the requested width");

    // ── leg 1: pool prefill (chunked GEMM path) + teacher-forced ring decode.
    let logits = m.forward_prefill(0, &prompt).expect("pool prefill");
    assert_eq!(
        argmax(&logits),
        greedy[0],
        "pool-prefill argmax disagrees with the oracle - the chunked GEMM \
         ladder or the prefill attention is wrong, not quant noise"
    );

    let (mut top1, mut top1_ring) = (0usize, 0usize);
    for s in 0..steps {
        m.batch_step_slots(&[greedy[s]], &[(pf + s) as u32], &[0])
            .expect("decode step");
        let l = m.read_batch_logits(1).expect("logits");
        let ok = argmax(&l) == t5_ids[s * 5];
        top1 += ok as usize;
        if s >= w {
            top1_ring += ok as usize;
        }
    }
    let ring_steps = steps - w;
    eprintln!("pool decode: top1 {top1}/{steps} (ring {top1_ring}/{ring_steps})");
    assert!(top1 as f64 / steps as f64 >= 0.95, "pool top-1 agreement");
    assert!(
        top1_ring as f64 / ring_steps as f64 >= 0.95,
        "pool ring agreement"
    );

    // ── leg 2: the footprint PINS. 200 generated tokens, blocks for pf + W.
    let blocks = m.pool_slot_blocks(0).expect("slot blocks");
    assert_eq!(
        blocks,
        (pf + w).div_ceil(16),
        "slot 0 owns {blocks} blocks after {steps} generated tokens - the ring \
         did not pin the pool footprint"
    );

    // ── leg 3: the same prompt into slot 1 resumes off the radix (published
    // at slot 0's prefill completion). The adopted blocks hold identical KV
    // BYTES; the recomputed tail is the engine's sanctioned near-tie class -
    // the norm's reduction width elects at the 64-row boundary
    // (pd_rmsnorm_batch's own comment), so a short tail norms 1024-wide where
    // the cold chunk normed 256-wide, and last-ulp differences can flip a
    // near-tie router expert. Gate = argmax + head tolerance, the oracle
    // gate's bar. Cold-vs-cold was verified bitwise while building this
    // (max|Δ| exactly 0.0 with the cache disabled).
    let logits1 = m.forward_prefill(1, &prompt).expect("resumed prefill");
    assert_eq!(argmax(&logits1), greedy[0], "resumed prefill argmax");
    let mut widx: Vec<usize> = (0..logits.len()).collect();
    widx.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
    let head_dmax = widx[..50]
        .iter()
        .map(|&i| (logits1[i] - logits[i]).abs())
        .fold(0f32, f32::max);
    eprintln!("resumed prefill: head50 max|Δ| {head_dmax:.4}");
    assert!(head_dmax < 0.5, "resumed prefill head drift {head_dmax}");

    // ── leg 4: mixed ring phases in one r=2 tick. Slot 0 is deep in steady
    // state (positions past pf + W + steps), slot 1 in warmup - 32 joint
    // steps, slot 1 teacher-forced against the oracle. A per-slot ring that
    // leaks state across rows collapses slot 1's agreement here.
    let mut agree1 = 0usize;
    for s in 0..32 {
        let toks = [greedy[(steps + s) % steps], greedy[s]];
        let pos = [(pf + steps + s) as u32, (pf + s) as u32];
        m.batch_step_slots(&toks, &pos, &[0, 1]).expect("r=2 step");
        let l = m.read_batch_logits(2).expect("logits r2");
        let row1 = &l[m.hp.n_vocab..];
        agree1 += (argmax(row1) == t5_ids[s * 5]) as usize;
    }
    eprintln!("mixed-phase r=2: slot1 top1 {agree1}/32");
    assert!(
        agree1 >= 30,
        "slot 1 agreement {agree1}/32 under a mixed-phase tick"
    );

    // slot 0 still pinned after 232 generated tokens
    assert_eq!(m.pool_slot_blocks(0).unwrap(), (pf + w).div_ceil(16));
}
