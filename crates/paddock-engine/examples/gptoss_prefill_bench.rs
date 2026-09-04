//! G1 prefill benchmark: COLD prompt-processing throughput on gpt-oss-20b.
//! Every run uses a distinct prompt (varied from token 0) so the radix prefix
//! cache never matches - this measures real prefill, not cache-hit TTFT.
//! The comparable reference shape is llama.cpp's
//! `llama-bench -m <gguf> -ngl 99 -n 0 -p 128,512` (pp128/pp512), run in the
//! same thermal window.
//!
//! Usage: gptoss_prefill_bench [len,len,...]   (default 128,512)
// A development probe: it runs on a box its author is looking at, and a
// failure should stop it where it happened rather than be reported.
#![allow(clippy::unwrap_used)]

use std::sync::Arc;

use paddock_engine::gpu::GpuExecutor;
use paddock_engine::gpu_model::gpt_oss::GpuGptOss;
use paddock_models::mapped::MappedGguf;

/// Deterministic token stream, distinct per (seed, position), well inside the
/// vocab and away from the special-token tail.
fn prompt(seed: u64, len: usize) -> Vec<u32> {
    (0..len)
        .map(|i| {
            let h = (seed.wrapping_add(i as u64).wrapping_mul(0x9E3779B97F4A7C15)) >> 33;
            (h % 100_000) as u32
        })
        .collect()
}

fn main() {
    let lens: Vec<usize> = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "128,512".into())
        .split(',')
        .map(|s| s.trim().parse().expect("length"))
        .collect();
    let model_path = std::env::var_os("PADDOCK_MODEL")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            let home = std::env::var("USERPROFILE")
                .or_else(|_| std::env::var("HOME"))
                .expect("USERPROFILE or HOME");
            std::path::PathBuf::from(home).join("paddock/models/gpt-oss-20b-mxfp4.gguf")
        });
    let pack = std::env::var_os("PADDOCK_PACK")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from("packs/cuda/build/pd-cuda-sm86.dll"));
    let exec = Arc::new(GpuExecutor::new(0, &pack).expect("executor"));
    let map = MappedGguf::open(&model_path).expect("open gguf");
    let mut m = GpuGptOss::load(exec, &map, 2048).expect("load");
    m.enable_batch(1).expect("batch");
    let mut seed = 1u64;
    for &len in &lens {
        // warm (allocations, module loads); distinct prompt like every run
        m.forward_prefill(0, &prompt(seed, len)).expect("warm");
        seed += 1;
        let mut best = 0.0f64;
        for run in 0..4 {
            let toks = prompt(seed, len);
            seed += 1;
            let t0 = std::time::Instant::now();
            let out = m.forward_prefill(0, &toks).expect("prefill");
            let dt = t0.elapsed().as_secs_f64();
            assert_eq!(out.len(), 201_088, "vocab logits");
            let tps = len as f64 / dt;
            println!("pp{len} run {run}: {dt:.4}s = {tps:.1} tok/s");
            best = best.max(tps);
        }
        println!("pp{len} best: {best:.1} tok/s");
    }
}
