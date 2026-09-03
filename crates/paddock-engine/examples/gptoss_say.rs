//! Greedy text sanity: load a gpt-oss GGUF, decode N tokens, print the text.
//! The bring-up coherence gate - a broken kernel writes word salad, not English.
//! Usage: gptoss_say [prompt] [n_tokens]   (PADDOCK_MODEL/PADDOCK_PACK override paths)

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
    let mut args = std::env::args().skip(1);
    let prompt = args
        .next()
        .unwrap_or_else(|| "Once upon a time".to_string());
    let n: usize = args
        .next()
        .map(|s| s.parse().expect("n_tokens"))
        .unwrap_or(64);
    let exec = Arc::new(GpuExecutor::new(0, &pack).expect("executor"));
    let map = MappedGguf::open(&model_path).expect("open gguf");
    let tok = GgufTokenizer::from_gguf(map.gguf()).expect("tokenizer");
    let ids = tok.encode(&prompt).expect("encode");
    let mut m = GpuGptOss::load(exec, &map, 2048).expect("load");
    let out = m.generate_greedy(&ids, n).expect("gen");
    println!("prompt: {prompt:?}");
    println!("tokens: {out:?}");
    println!("text:   {}", tok.decode(&out, true).expect("decode"));
}
