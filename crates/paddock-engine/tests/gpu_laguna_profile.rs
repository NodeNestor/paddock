//! Phase-split timing probe for the laguna batched decode tick: where do the
//! milliseconds of an r==1 tick go (projections / attention / MoE / shexp /
//! head)? A bench harness, not a correctness gate -
//! gated on PADDOCK_HEAVY_TESTS + the elected XS-2.1 file + the sm86 pack.

mod common;

use std::sync::Arc;

use paddock_engine::generator::Generator;
use paddock_engine::gpu::{GpuExecutor, KvDtype};
use paddock_engine::gpu_model::laguna::GpuLaguna;
use paddock_models::mapped::MappedGguf;

#[test]
fn laguna_tick_phase_split() {
    if !common::heavy() {
        return;
    }
    let Some(path) = common::model("LAGUNA_GGUF", common::LAGUNA_XS_Q4) else {
        return;
    };
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    let map = MappedGguf::open(&path).expect("open gguf");
    let mut gpu = GpuLaguna::load(exec, &map, 2048).expect("load");
    let slots = gpu.enable_batch(2).expect("enable_batch");
    assert!(slots > 1, "batch lane required for the probe (got {slots})");
    // prefix length sets the attention depth the decode tick sees (SWA
    // layers cap at window 512; the 10 full layers walk the whole prefix).
    // LAGUNA_PROBE_CTX overrides - default 1900 (near the 2048 alloc), the
    // regime where the FlashDecoding split matters.
    let ctx: u32 = std::env::var("LAGUNA_PROBE_CTX")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1900);
    let prompt: Vec<u32> = (0..ctx).map(|i| 100 + (i % 50)).collect();
    gpu.forward_prefill(0, &prompt).expect("prefill");
    gpu.profile_batch_tick(prompt.len() as u32, 32)
        .expect("profile");
}

fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i as u32)
        .expect("nonempty logits")
}

/// Greedy decode loop over the serial (batch-1) lane - laguna has no
/// `generate_greedy` convenience method (unlike qwen35/gpt_oss).
fn generate_greedy_serial(m: &mut GpuLaguna, prompt: &[u32], steps: usize) -> Vec<u32> {
    m.reset();
    let mut logits = m.forward_prefill_stream(prompt).expect("prefill");
    let mut out = Vec::with_capacity(steps);
    for _ in 0..steps {
        let tok = argmax(&logits);
        out.push(tok);
        logits = m.forward(tok).expect("decode step");
    }
    out
}

/// Greedy decode loop over the BATCHED (paged) lane - `forward_prefill` +
/// `forward_batch` with explicit, externally-tracked positions, exactly how
/// the real service driver calls it - unlike `forward`, which stays on the
/// serial lane even after `enable_batch` succeeds. Caller must have already
/// called `enable_batch`.
fn generate_greedy_batched(m: &mut GpuLaguna, prompt: &[u32], steps: usize) -> Vec<u32> {
    let mut logits = m.forward_prefill(0, prompt).expect("prefill");
    let mut out = Vec::with_capacity(steps);
    for pos in (prompt.len() as u32..).take(steps) {
        let tok = argmax(&logits);
        out.push(tok);
        logits = m.forward_batch(&[tok], &[pos]).expect("decode step");
    }
    out
}

/// fp8 KV cache drift gate - the laguna sibling of the qwen35/gpt-oss/granite
/// gate: decode the same prompt greedily with f16 then fp8 KV and report the
/// divergence point. fp8 is a lossy opt-in class, so the assertion is only
/// completion + valid ids (drift is EXPECTED); the printed match count is the
/// honest signal. Laguna is the hybrid case: 36 of 48 layers are SWA-512
/// (window-capped, cheap to hold in fp8) and 12 are full attention.
#[test]
fn kv_fp8_decodes_and_reports_drift() {
    if !common::heavy() {
        return;
    }
    let Some(path) = common::model("LAGUNA_GGUF", common::LAGUNA_XS_Q4) else {
        return;
    };
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    let map = MappedGguf::open(&path).expect("open gguf");
    let mut m = GpuLaguna::load(exec, &map, 2048).expect("load laguna");
    let vocab = m.vocab();

    let prompt: Vec<u32> = vec![760, 6511, 314, 9338, 369]; // robust counting-style
    let steps = 64usize;
    let fp16 = generate_greedy_serial(&mut m, &prompt, steps);
    m.set_kv_dtype(KvDtype::Fp8E4m3);
    let fp8 = generate_greedy_serial(&mut m, &prompt, steps);

    let matched = fp16.iter().zip(&fp8).take_while(|(a, b)| a == b).count();
    let same = fp16.iter().zip(&fp8).filter(|(a, b)| a == b).count();
    eprintln!(
        "fp8 KV drift: {same}/{steps} greedy tokens match fp16 ({matched} before first divergence)"
    );
    assert_eq!(fp8.len(), steps, "fp8 decode did not run to completion");
    assert!(
        fp8.iter().all(|&t| (t as usize) < vocab),
        "fp8 produced invalid token ids"
    );
}

/// Serial (batch-1) vs batched (paged) lane parity, at both KV dtypes - the
/// specific regression this refactor could introduce silently: laguna has
/// two independent code paths that both read `self.kv_dtype`
/// (`forward_one`/`ensure_decode` vs `layer_walk`/`enable_batch_impl`), plus
/// `prefix.rs`'s SWA-checkpoint byte-budget math - a missed call site in any
/// of them would only surface here.
#[test]
fn serial_and_batched_lanes_agree_at_both_kv_dtypes() {
    if !common::heavy() {
        return;
    }
    // keeps the pack PATH: one executor per KV dtype below
    let Some(pack) = common::pack() else {
        return;
    };
    let Some(path) = common::model("LAGUNA_GGUF", common::LAGUNA_XS_Q4) else {
        return;
    };
    let prompt: Vec<u32> = vec![760, 6511, 314, 9338, 369];
    let steps = 16usize;

    for dtype in [KvDtype::Fp16, KvDtype::Fp8E4m3] {
        // One model instance at a time - S-2.1's 68 GiB resident weights
        // don't fit two-up on a single card. `serial` is fully dropped
        // (freeing its VRAM back to the pool) before `batched` loads.
        let serial_out = {
            let exec = match GpuExecutor::new(0, &pack) {
                Ok(e) => Arc::new(e),
                Err(e) => {
                    eprintln!("no CUDA ({e}) - skipping");
                    return;
                }
            };
            let map = MappedGguf::open(&path).expect("open gguf");
            let mut serial = GpuLaguna::load(exec, &map, 2048).expect("load laguna");
            serial.set_kv_dtype(dtype);
            generate_greedy_serial(&mut serial, &prompt, steps)
        };

        let exec = match GpuExecutor::new(0, &pack) {
            Ok(e) => Arc::new(e),
            Err(e) => {
                eprintln!("no CUDA ({e}) - skipping");
                return;
            }
        };
        let map = MappedGguf::open(&path).expect("open gguf");
        let mut batched = GpuLaguna::load(exec, &map, 2048).expect("load laguna");
        batched.set_kv_dtype(dtype);
        let slots = batched.enable_batch(4).expect("enable_batch");
        assert!(
            slots > 1,
            "batch lane required for this check (got {slots})"
        );
        let batched_out = generate_greedy_batched(&mut batched, &prompt, steps);

        assert_eq!(
            serial_out, batched_out,
            "serial vs batched lane diverged at {dtype:?} - a kv_dtype call site was likely missed"
        );
    }
}
