//! Loader milestone for granite: open a real GGUF, upload every tensor, and
//! run the fp8 KV drift gate (the granite sibling of the qwen35/gpt-oss one).
//!
//! Heavy (uploads several GB to the device) - gated on the model file, a
//! built pack, a CUDA device, and PADDOCK_HEAVY_TESTS for the decode tests.

mod common;

use std::sync::Arc;

use paddock_engine::generator::Generator;
use paddock_engine::gpu::{GpuExecutor, KvDtype};
use paddock_engine::gpu_model::granite::GpuGranite;
use paddock_models::mapped::MappedGguf;

fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i as u32)
        .expect("nonempty logits")
}

/// Greedy decode loop over the serial (batch-1) lane - `forward_one` via the
/// plain `Generator::forward`, granite has no `generate_greedy` convenience
/// method (unlike qwen35/gpt_oss).
fn generate_greedy_serial(m: &mut GpuGranite, prompt: &[u32], steps: usize) -> Vec<u32> {
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
/// the real service driver calls it (see `service.rs`'s scheduler loop) -
/// unlike `forward`/`forward_one`, which stays on the serial lane even after
/// `enable_batch` succeeds. Caller must have already called `enable_batch`.
fn generate_greedy_batched(m: &mut GpuGranite, prompt: &[u32], steps: usize) -> Vec<u32> {
    let mut logits = m.forward_prefill(0, prompt).expect("prefill");
    let mut out = Vec::with_capacity(steps);
    for pos in (prompt.len() as u32..).take(steps) {
        let tok = argmax(&logits);
        out.push(tok);
        logits = m.forward_batch(&[tok], &[pos]).expect("decode step");
    }
    out
}

#[test]
fn loads_granite_and_geometry_matches() {
    let Some(path) = common::model("GRANITE_GGUF", common::GRANITE_8B_Q8) else {
        return;
    };
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    let map = MappedGguf::open(&path).expect("open gguf");
    assert_eq!(map.gguf().architecture(), Some("granite"), "arch string");

    let model = GpuGranite::load(exec, &map, 4096).expect("load granite");
    assert!(model.vocab() > 100_000, "vocab {}", model.vocab());
}

/// fp8 KV cache drift gate - the granite sibling of the qwen35/gpt-oss gate:
/// decode the same prompt greedily with f16 then fp8 KV and report the
/// divergence point. fp8 is a lossy opt-in class, so the assertion is only
/// completion + valid ids (drift is EXPECTED); the printed match count is the
/// honest signal. Granite is full attention on every layer (no SWA at all),
/// so this is the least forgiving shape for KV precision loss in this repo.
#[test]
fn kv_fp8_decodes_and_reports_drift() {
    if !common::heavy() {
        return;
    }
    let Some(path) = common::model("GRANITE_GGUF", common::GRANITE_8B_Q8) else {
        return;
    };
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    let map = MappedGguf::open(&path).expect("open gguf");
    let mut m = GpuGranite::load(exec, &map, 2048).expect("load granite");
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
/// specific regression this refactor could introduce silently: granite has
/// two independent code paths that both read `self.kv_dtype`
/// (`forward_one`/`ensure_decode` vs `layer_walk`/`enable_batch_impl`), and a
/// missed call site in one of them would only surface here.
#[test]
fn serial_and_batched_lanes_agree_at_both_kv_dtypes() {
    if !common::heavy() {
        return;
    }
    // keeps the path: this gate builds and drops an executor per KV dtype
    let Some(pack) = common::pack() else {
        return;
    };
    let Some(path) = common::model("GRANITE_GGUF", common::GRANITE_8B_Q8) else {
        return;
    };
    let prompt: Vec<u32> = vec![760, 6511, 314, 9338, 369];
    let steps = 16usize;

    for dtype in [KvDtype::Fp16, KvDtype::Fp8E4m3] {
        // One model instance at a time - the 30b variant's resident weights
        // may not fit two-up on a single card. `serial` is fully dropped
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
            let mut serial = GpuGranite::load(exec, &map, 2048).expect("load granite");
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
        let mut batched = GpuGranite::load(exec, &map, 2048).expect("load granite");
        batched.set_kv_dtype(dtype);
        let slots = batched.enable_batch(4).expect("enable_batch");
        assert!(
            slots > 1,
            "batch lane required for this check (got {slots})"
        );
        let batched_out = generate_greedy_batched(&mut batched, &prompt, steps);

        assert_eq!(
            serial_out, batched_out,
            "serial vs batched lane diverged at {dtype:?} - a KV_DTYPE call site was likely missed"
        );
    }
}
