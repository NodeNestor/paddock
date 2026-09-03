//! Laguna loader-milestone probe: parse geometry + upload every weight of a
//! Laguna GGUF, print the audit. First rung of the bring-up ladder -
//! forward/greedy comes next.
//!
//! Usage: LAGUNA_GGUF=<path>\Laguna-XS-2.1-Q4_K_M.gguf
//!        PADDOCK_PACK=packs\cuda\build\pd-cuda-sm86.dll
//!        cargo run --release --example laguna_load

use std::sync::Arc;

use paddock_engine::gpu::GpuExecutor;
use paddock_engine::gpu_model::laguna::GpuLaguna;
use paddock_models::mapped::MappedGguf;
use paddock_tokenizer::GgufTokenizer;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();
    let model = std::env::var("LAGUNA_GGUF").expect("set LAGUNA_GGUF");
    let pack = std::env::var("PADDOCK_PACK").expect("set PADDOCK_PACK");
    let max_ctx: usize = std::env::var("MAX_CTX")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4096);

    let map = MappedGguf::open(model.as_ref()).expect("open gguf");

    // Tokenizer sanity ride-along: the GGUF ships a gpt2-class BPE with the
    // "laguna" pre-tokenizer - from_gguf failing here tells us the tokenizer
    // seam needs work before serving does.
    match GgufTokenizer::from_gguf(map.gguf()) {
        Ok(tok) => {
            let ids = tok.encode("Reply with exactly: ok").expect("encode");
            eprintln!("tokenizer OK, vocab {}, sample ids {ids:?}", tok.vocab_size);
        }
        Err(e) => eprintln!("tokenizer NOT ready yet: {e}"),
    }

    let t0 = std::time::Instant::now();
    let exec = Arc::new(GpuExecutor::new(0, pack.as_ref()).expect("executor"));
    let m = GpuLaguna::load(exec, &map, max_ctx).expect("load");
    eprintln!(
        "laguna loaded in {:.1}s - audit above",
        t0.elapsed().as_secs_f32()
    );
    drop(m);
    eprintln!("released cleanly");
}
