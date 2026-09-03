//! Perf-only bench for per-slot MTP spec decode on Qwen3.6-27B: base batch
//! decode vs `generate_greedy_spec_batch` aggregate tok/s, no reference
//! generation (correctness is gated in gpu_qwen36_spec_batch.rs).
//!
//! Knobs: PADDOCK_SPEC_B (default 4), PADDOCK_SPEC_K (default 4),
//! PADDOCK_SPEC_NEW (default 64), PADDOCK_SPEC_PHASE_TIME=1 for the per-round
//! phase breakdown. Heavy (~27 GB load): gated on PADDOCK_HEAVY_TESTS,
//! --test-threads=1.

mod common;

use std::time::Instant;

use paddock_engine::gpu_model::qwen35::GpuQwen35;
use paddock_models::mapped::MappedGguf;

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best as u32
}

#[test]
fn spec_batch_bench() {
    if !common::heavy() {
        return;
    }
    let Some(path) = common::model("QWEN36_27B_GGUF", common::QWEN36_27B_Q8) else {
        return;
    };
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    let map = MappedGguf::open(&path).expect("open gguf");
    let mut m = GpuQwen35::load(exec, &map, 4096).expect("load 27B");

    let b = env_usize("PADDOCK_SPEC_B", 4);
    let n_new = env_usize("PADDOCK_SPEC_NEW", 64);
    let n_draft = env_usize("PADDOCK_SPEC_K", 4);

    // synthetic prompts with different lengths (5..5+B tokens from a fixed pool)
    let pool = [760u32, 6511, 314, 9338, 369, 1207, 4552, 88];
    let prompts: Vec<Vec<u32>> = (0..b)
        .map(|s| (0..5 + s).map(|i| pool[(s + i) % pool.len()]).collect())
        .collect();

    m.enable_batch(b).expect("enable_batch");
    m.enable_spec_batch(b, n_draft).expect("enable_spec_batch");

    // --- base batch decode ----------------------------------------------------
    if std::env::var_os("PADDOCK_SPEC_SKIP_BASE").is_none() {
        let mut outs: Vec<Vec<u32>> = Vec::with_capacity(b);
        let mut vocab = 0usize;
        for (slot, p) in prompts.iter().enumerate() {
            let logits = m.forward_prefill_slot(slot, p).expect("prefill slot");
            vocab = logits.len();
            outs.push(vec![argmax(&logits)]);
        }
        let t0 = Instant::now();
        let mut positions: Vec<u32> = prompts.iter().map(|p| p.len() as u32).collect();
        while outs[0].len() < n_new {
            let toks: Vec<u32> = outs.iter().map(|o| *o.last().unwrap()).collect();
            let logits = m.forward_batch(&toks, &positions).expect("batch step");
            for slot in 0..b {
                outs[slot].push(argmax(&logits[slot * vocab..(slot + 1) * vocab]));
                positions[slot] += 1;
            }
        }
        let dt = t0.elapsed().as_secs_f64();
        eprintln!(
            "base batch B={b}: {:.1} tok/s aggregate ({:.2}ms/step)",
            (b * (n_new - 1)) as f64 / dt,
            dt * 1e3 / (n_new - 1) as f64
        );
    }

    // --- per-slot spec decode (includes its own prefill+warm; prompts are tiny)
    // alternate graph/eager IN-PROCESS: cold/warm clock variance swamps
    // cross-process comparisons
    let runs = env_usize("PADDOCK_SPEC_RUNS", 4);
    for round in 0..runs {
        let graphed = round % 2 == 0;
        if graphed {
            unsafe { std::env::remove_var("PADDOCK_SPEC_NOGRAPH") };
        } else {
            unsafe { std::env::set_var("PADDOCK_SPEC_NOGRAPH", "1") };
        }
        let t1 = Instant::now();
        let spec = m
            .generate_greedy_spec_batch(&prompts, n_new, n_draft)
            .expect("spec batch");
        let spec_dt = t1.elapsed().as_secs_f64();
        assert!(spec.iter().all(|o| o.len() >= n_new));
        eprintln!(
            "spec batch B={b} k={n_draft} {}: {:.1} tok/s aggregate (incl prefill+warm)",
            if graphed { "GRAPH" } else { "EAGER" },
            (b * n_new) as f64 / spec_dt
        );
    }
}
