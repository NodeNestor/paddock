//! Qwen3-Embedding throughput bench: encode a corpus of realistic passages in
//! batches and report texts/sec + tokens/sec. The batch-amortized prefill is
//! the RAG-ingestion throughput lever. Compare against llama.cpp same-window
//! (llama-server --embeddings on the identical GGUF).
//!
//! Usage: qwen3_embed_bench [n_texts] [batch]   (env QWEN3_EMBED_GGUF overrides)

use std::sync::Arc;
use std::time::Instant;

use paddock_engine::gpu::GpuExecutor;
use paddock_engine::gpu_model::qwen3::GpuQwen3;
use paddock_models::mapped::MappedGguf;
use paddock_tokenizer::GgufTokenizer;

/// Deterministic ~40-60 token passages, varied per index.
fn corpus(n: usize) -> Vec<String> {
    let subjects = [
        "The distributed database",
        "A convolutional network",
        "The migratory bird",
        "This financial model",
        "The volcanic eruption",
        "Our new scheduler",
        "The ancient manuscript",
        "A superconducting magnet",
        "The coral reef",
        "The quarterly report",
    ];
    let bodies = [
        "processes incoming requests through a pipeline of validation, routing and replication stages, each of which can fail independently and must be retried with exponential backoff to preserve consistency across the cluster.",
        "learns hierarchical features from raw pixels, progressively composing edges into textures and textures into objects, and its accuracy depends heavily on the diversity of the training distribution it was exposed to.",
        "navigates thousands of kilometres using a combination of magnetic sensing, star patterns and remembered landmarks, returning each season to the same nesting grounds with remarkable precision despite changing weather.",
        "estimates future cash flows under several macroeconomic scenarios, discounting them to present value and stress-testing the result against interest-rate shocks and correlated defaults in the underlying portfolio.",
    ];
    (0..n)
        .map(|i| {
            format!(
                "{} {}",
                subjects[i % subjects.len()],
                bodies[(i / subjects.len()) % bodies.len()]
            )
        })
        .collect()
}

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(256);
    let batch: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);
    let pack = std::env::var_os("PADDOCK_PACK")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../packs/cuda/build/pd-cuda-sm120.so")
        });
    let model = std::env::var("QWEN3_EMBED_GGUF").unwrap_or_else(|_| {
        "C:/dev/models/Qwen3-Embedding-0.6B-GGUF/Qwen3-Embedding-0.6B-Q8_0.gguf".into()
    });
    let exec = Arc::new(GpuExecutor::new(0, &pack).expect("cuda executor"));
    let map = MappedGguf::open(std::path::Path::new(&model)).expect("open gguf");
    let tok = GgufTokenizer::from_gguf(map.gguf()).expect("tokenizer");
    let mut m = GpuQwen3::load(exec, &map, 4096).expect("load qwen3");

    let eos = 151643u32;
    let texts = corpus(n);
    let seqs: Vec<Vec<u32>> = texts
        .iter()
        .map(|t| {
            let mut e = tok.encode(t).expect("encode");
            e.push(eos);
            e
        })
        .collect();
    let total_toks: usize = seqs.iter().map(|s| s.len()).sum();
    eprintln!(
        "corpus: {n} texts, {total_toks} tokens ({:.1} avg), batch={batch}",
        total_toks as f64 / n as f64
    );

    // warm (alloc scratch/KV at this batch shape)
    let _ = m.embed(&seqs[..batch.min(n)]).expect("warm");

    let mut best = 0.0f64;
    for run in 0..3 {
        let t0 = Instant::now();
        for chunk in seqs.chunks(batch) {
            let _ = m.embed(chunk).expect("embed");
        }
        let dt = t0.elapsed().as_secs_f64();
        let tps = n as f64 / dt;
        let tok_s = total_toks as f64 / dt;
        eprintln!("run {run}: {n} texts in {dt:.3}s = {tps:.1} texts/s ({tok_s:.0} tok/s)");
        best = best.max(tps);
    }
    eprintln!("QWEN3 EMBED BENCH: best {best:.1} texts/s (batch {batch})");
}
