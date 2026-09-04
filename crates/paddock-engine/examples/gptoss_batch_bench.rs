//! G2 serving-batch benchmark: aggregate decode throughput at batch B on
//! gpt-oss-20b. The comparable reference shape is llama.cpp's
//! `llama-batched-bench -m <gguf> -ngl 99 -npp 16 -ntg 64 -npl <B>`, run in
//! the same thermal window. Args: `<B list, e.g. 8,32>` `<steps>` (defaults
//! 32, 64). A single-B invocation is the cleanest shape to profile for
//! per-kernel attribution.
// A development probe: it runs on a box its author is looking at, and a
// failure should stop it where it happened rather than be reported.
#![allow(clippy::unwrap_used)]

use std::sync::Arc;

use paddock_engine::gpu::GpuExecutor;
use paddock_engine::gpu_model::gpt_oss::GpuGptOss;
use paddock_models::mapped::MappedGguf;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let bs: Vec<usize> = args
        .get(1)
        .map(|s| s.split(',').map(|x| x.parse().expect("B")).collect())
        .unwrap_or_else(|| vec![32]);
    let steps: usize = args.get(2).map(|s| s.parse().expect("steps")).unwrap_or(64);
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
    let max_b = bs.iter().copied().max().unwrap_or(32);
    // optional arg 3: per-row KV depth before timing (serving decodes at
    // depth ~1k-4k, where the attention walk - not the MoE - can dominate;
    // the default 8 measures the near-empty-cache floor)
    let depth: usize = std::env::args()
        .nth(3)
        .map(|s| s.parse().expect("depth"))
        .unwrap_or(8);
    let max_ctx = (depth + steps + 16).next_multiple_of(64).max(1024);
    let mut m = GpuGptOss::load(exec, &map, max_ctx).expect("load");
    m.enable_batch(max_b).expect("enable_batch");
    for &b in &bs {
        // distinct tokens per row so expert routing is not degenerate (identical
        // rows all pick the same 4 experts and flatter the grouped path).
        let toks: Vec<u32> = (0..b as u32).map(|i| 100 + i * 37).collect();
        if depth > 8 {
            // fill each row's cache to `depth` with a real (varied) prefill
            for slot in 0..b {
                let prompt: Vec<u32> = (0..depth as u32)
                    .map(|i| 100 + (i * 13 + slot as u32 * 101) % 5000)
                    .collect();
                m.forward_prefill(slot, &prompt).expect("depth prefill");
            }
        } else {
            for p in 0..8u32 {
                let pos: Vec<u32> = vec![p; b];
                m.forward_batch(&toks, &pos).expect("warm");
            }
        }
        // warm the timed shape itself (dispatch/graph caches)
        for w in 0..4u32 {
            let pos: Vec<u32> = vec![depth as u32 + w; b];
            m.forward_batch(&toks, &pos).expect("warm2");
        }
        let t0 = std::time::Instant::now();
        for s in 0..steps {
            let pos: Vec<u32> = vec![(depth + 4 + s) as u32; b];
            m.forward_batch(&toks, &pos).expect("fwd");
        }
        let dt = t0.elapsed().as_secs_f64();
        println!(
            "B={b:>2} depth={depth}: {:6.2} ms/step | aggregate {:7.1} tok/s | per-seq {:.1} tok/s",
            dt * 1e3 / steps as f64,
            (b * steps) as f64 / dt,
            steps as f64 / dt
        );
    }
}
