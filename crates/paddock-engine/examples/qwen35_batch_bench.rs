//! Serving-batch benchmark for qwen3.5/3.6: aggregate decode throughput at
//! batch B, the qwen sibling of gptoss_batch_bench (same methodology: warm 8
//! positions, then time `steps` batched decode steps from there). The
//! comparable reference shape is llama.cpp's `llama-batched-bench -m <gguf>
//! -ngl 99 -fa 1 -npp 16 -ntg 64 -npl <B,...>` S_TG, same thermal window.
//! Args: `<B list, e.g. 1,8,32>` `<steps>` (defaults 1,2,4,8,16,32,64 / 64).
//! Env: QWEN35_GGUF picks the model; PADDOCK_KV_FP8=1 opts into fp8 KV.

use std::sync::Arc;

use paddock_engine::gpu::{GpuExecutor, KvDtype};
use paddock_engine::gpu_model::qwen35::GpuQwen35;
use paddock_models::mapped::MappedGguf;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let bs: Vec<usize> = args
        .get(1)
        .map(|s| s.split(',').map(|x| x.parse().expect("B")).collect())
        .unwrap_or_else(|| vec![1, 2, 4, 8, 16, 32, 64]);
    let steps: usize = args.get(2).map(|s| s.parse().expect("steps")).unwrap_or(64);
    let model = std::env::var("QWEN35_GGUF")
        .unwrap_or_else(|_| "C:/dev/models/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q8_0.gguf".to_string());
    let pack = std::env::var_os("PADDOCK_PACK")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../packs/cuda/build/pd-cuda-sm86.dll")
        });
    let exec = Arc::new(GpuExecutor::new(0, &pack).expect("executor"));
    let map = MappedGguf::open(std::path::Path::new(&model)).expect("open gguf");
    let max_b = bs.iter().copied().max().unwrap_or(64);
    let mut m = GpuQwen35::load(exec, &map, 1024).expect("load qwen35");
    if std::env::var_os("PADDOCK_KV_FP8").is_some() {
        m.set_kv_dtype(KvDtype::Fp8E4m3);
        eprintln!("KV dtype: fp8 e4m3 (lossy opt-in)");
    }
    m.enable_batch(max_b).expect("enable_batch");
    for &b in &bs {
        // distinct tokens per row so nothing degenerates to identical rows
        let toks: Vec<u32> = (0..b as u32).map(|i| 100 + i * 37).collect();
        for p in 0..8u32 {
            let pos: Vec<u32> = vec![p; b];
            m.forward_batch(&toks, &pos).expect("warm");
        }
        let t0 = std::time::Instant::now();
        for s in 0..steps {
            let pos: Vec<u32> = vec![8 + s as u32; b];
            m.forward_batch(&toks, &pos).expect("fwd");
        }
        let dt = t0.elapsed().as_secs_f64();
        println!(
            "B={b:>2}: {:6.2} ms/step | aggregate {:7.1} tok/s | per-seq {:.1} tok/s",
            dt * 1e3 / steps as f64,
            (b * steps) as f64 / dt,
            steps as f64 / dt
        );
    }
}
