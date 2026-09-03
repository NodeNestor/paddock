//! G0 decode benchmark: graph-resident B=1 greedy throughput on gpt-oss-20b.
//! The comparable reference shape is llama.cpp's
//! `llama-bench -m <gguf> -ngl 99 -n 128` (tg128), run in the same thermal
//! window.

use std::sync::Arc;

use paddock_engine::gpu::GpuExecutor;
use paddock_engine::gpu_model::gpt_oss::GpuGptOss;
use paddock_models::mapped::MappedGguf;
use paddock_tokenizer::GgufTokenizer;

fn main() {
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
    let tok = GgufTokenizer::from_gguf(map.gguf()).expect("tokenizer");
    let prompt = tok.encode("Once upon a time").expect("encode");
    let mut m = GpuGptOss::load(exec, &map, 2048).expect("load");
    m.generate_greedy(&prompt, 16).expect("warm"); // captures the gen graph
    let n = 128;
    let mut best = 0.0f64;
    for run in 0..4 {
        let t0 = std::time::Instant::now();
        let out = m.generate_greedy(&prompt, n).expect("gen");
        let dt = t0.elapsed().as_secs_f64();
        let tps = out.len() as f64 / dt;
        println!(
            "run {run}: {} tokens in {dt:.3}s = {tps:.1} tok/s",
            out.len()
        );
        best = best.max(tps);
    }
    println!("best: {best:.1} tok/s");
}
