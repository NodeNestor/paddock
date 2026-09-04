//! Standalone decode driver for Nsight profiling. Loads Qwen3.5-9B, warms the
//! graph, then runs a fixed number of O(1)/token decode steps so `nsys`/`ncu` can
//! wrap a clean single process. The model load (dequant/repack kernels) is a
//! one-time blip; the decode kernels (gemv_repacked, attn_decode_batch, the
//! DeltaNet ops, norms, ...) dominate the kernel summary.
//!
//!   nsys profile --stats=true -o qwen35 target/release/examples/qwen35_profile.exe
//!   ncu -k pd_q8_0_gemv_repacked_kernel --launch-skip 5000 --launch-count 20 ...
//!
//! Args: [steps] [depth] (default 200, 0). A nonzero depth prefills a
//! `depth`-token prompt first so the STABLE decode probe runs at that context
//! depth - the agentic-serving operating point, where the attention geometry
//! matters. Env QWEN35_GGUF overrides the model path.
// A development probe: it runs on a box its author is looking at, and a
// failure should stop it where it happened rather than be reported.
#![allow(clippy::unwrap_used)]

use std::sync::Arc;
use std::time::Instant;

use paddock_engine::gpu::GpuExecutor;
use paddock_engine::gpu_model::qwen35::GpuQwen35;
use paddock_models::mapped::MappedGguf;

fn main() {
    let steps: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(200);
    let depth: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let pack = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../packs/cuda/build/pd-cuda-sm86.dll");
    let model = std::env::var("QWEN35_GGUF")
        .unwrap_or_else(|_| "C:/dev/models/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q8_0.gguf".to_string());

    let exec = Arc::new(GpuExecutor::new(0, &pack).expect("cuda executor"));
    let map = MappedGguf::open(std::path::Path::new(&model)).expect("open gguf");
    // KV sized for the deepest probe: depth prefix + bench decode + slack,
    // and never below the largest prefill sweep size.
    let max_ctx = (depth + steps + 1024)
        .next_multiple_of(4096)
        .max(8192 + 4096);
    let mut m = GpuQwen35::load(exec, &map, max_ctx).expect("load qwen35");
    // PADDOCK_KV_FP8=1 opts into the lossy fp8 KV cache (halved KV traffic)
    if std::env::var_os("PADDOCK_KV_FP8").is_some() {
        m.set_kv_dtype(paddock_engine::gpu::KvDtype::Fp8E4m3);
        eprintln!("KV dtype: fp8 e4m3 (lossy opt-in)");
    }

    // "The capital of France is"
    let prompt: Vec<u32> = vec![760, 6511, 314, 9338, 369];
    // warm: capture the graph + settle clocks
    let _ = m.generate_greedy(&prompt, 8, None).expect("warmup");

    let t0 = Instant::now();
    let out = m.generate_greedy(&prompt, steps, None).expect("decode");
    let dt = t0.elapsed();
    eprintln!(
        "decoded {} tokens in {:?} = {:.1} tok/s (first out id {})",
        out.len(),
        dt,
        steps as f64 / dt.as_secs_f64(),
        out.first().copied().unwrap_or(0),
    );

    // Low-variance kernel-tuning probe: min ms/token over steady-clock batches.
    // At depth > 0 the probe decodes from a depth-token prefix, so attention
    // (and everything else O(ctx)) is measured at that context depth.
    let bench_prompt: Vec<u32> = if depth > 0 {
        prompt.iter().copied().cycle().take(depth).collect()
    } else {
        prompt.clone()
    };
    let ms = m.bench_decode_ms(&bench_prompt, 300, 400).expect("bench");
    eprintln!(
        "STABLE decode @depth {}: {ms:.4} ms/token = {:.2} tok/s (min-of-batches @ boost)",
        bench_prompt.len(),
        1000.0 / ms
    );

    // Prefill throughput (pp): batched pass over a longer prompt, llama-bench style.
    for pp in [128usize, 512, 2048, 8192] {
        let long: Vec<u32> = prompt.iter().copied().cycle().take(pp).collect();
        m.reset();
        let _ = m.prefill(&long).expect("prefill warm"); // warm/alloc at this size
        let mut best = f64::INFINITY;
        for _ in 0..3 {
            m.reset();
            let t0 = Instant::now();
            let _ = m.prefill(&long).expect("prefill");
            best = best.min(t0.elapsed().as_secs_f64());
        }
        eprintln!(
            "prefill pp{pp}: {:.1} ms = {:.0} tok/s",
            best * 1e3,
            pp as f64 / best
        );
    }
}
