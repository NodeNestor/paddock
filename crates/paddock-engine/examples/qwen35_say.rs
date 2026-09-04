//! Greedy text sanity for qwen: load a Qwen3.5/3.6 GGUF, decode N tokens, print
//! the text. Same bring-up coherence gate as gptoss_say - a broken kernel writes
//! word salad, not English. Works for both 9B and 27B (QWEN35_GGUF picks the file).
//! Usage: qwen35_say [prompt] [n_tokens]   (QWEN35_GGUF/PADDOCK_PACK override paths)
// A development probe: it runs on a box its author is looking at, and a
// failure should stop it where it happened rather than be reported.
#![allow(clippy::unwrap_used)]

use std::sync::Arc;

use paddock_engine::gpu::GpuExecutor;
use paddock_engine::gpu_model::qwen35::GpuQwen35;
use paddock_models::mapped::MappedGguf;
use paddock_tokenizer::GgufTokenizer;

fn main() {
    let model = std::env::var("QWEN35_GGUF")
        .unwrap_or_else(|_| "C:/dev/models/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q8_0.gguf".to_string());
    let pack = std::env::var_os("PADDOCK_PACK")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../packs/cuda/build/pd-cuda-sm86.dll")
        });
    let mut args = std::env::args().skip(1);
    let prompt = args
        .next()
        .unwrap_or_else(|| "The capital of France is".to_string());
    let n: usize = args
        .next()
        .map(|s| s.parse().expect("n_tokens"))
        .unwrap_or(64);
    let exec = Arc::new(GpuExecutor::new(0, &pack).expect("executor"));
    let map = MappedGguf::open(std::path::Path::new(&model)).expect("open gguf");
    let tok = GgufTokenizer::from_gguf(map.gguf()).expect("tokenizer");
    let ids = tok.encode(&prompt).expect("encode");
    let mut m = GpuQwen35::load(exec, &map, 4096).expect("load qwen35");
    let out = m.generate_greedy(&ids, n, None).expect("gen");
    println!("prompt: {prompt:?}");
    println!("tokens: {out:?}");
    println!("text:   {}", tok.decode(&out, true).expect("decode"));
}
