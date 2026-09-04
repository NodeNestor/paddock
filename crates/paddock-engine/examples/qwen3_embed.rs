//! Qwen3-Embedding bring-up smoke: load the dense encoder, embed a few texts
//! through the batched prefill, and verify the vectors are meaningful - a
//! semantically similar pair must score higher cosine than a dissimilar pair.
//! Embeddings are L2-normalized, so cosine == dot.
//!
//! Usage: qwen3_embed [gguf]   (env QWEN3_EMBED_GGUF overrides)
// A development probe: it runs on a box its author is looking at, and a
// failure should stop it where it happened rather than be reported.
#![allow(clippy::unwrap_used)]

use std::sync::Arc;

use paddock_engine::gpu::GpuExecutor;
use paddock_engine::gpu_model::qwen3::GpuQwen3;
use paddock_models::mapped::MappedGguf;
use paddock_tokenizer::GgufTokenizer;

fn cos(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn main() {
    let pack = std::env::var_os("PADDOCK_PACK")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../packs/cuda/build/pd-cuda-sm120.so")
        });
    let model = std::env::args()
        .nth(1)
        .or_else(|| std::env::var("QWEN3_EMBED_GGUF").ok())
        .unwrap_or_else(|| {
            "C:/dev/models/Qwen3-Embedding-0.6B-GGUF/Qwen3-Embedding-0.6B-Q8_0.gguf".into()
        });
    let exec = Arc::new(GpuExecutor::new(0, &pack).expect("cuda executor"));
    let map = MappedGguf::open(std::path::Path::new(&model)).expect("open gguf");
    let tok = GgufTokenizer::from_gguf(map.gguf()).expect("tokenizer");
    let mut m = GpuQwen3::load(exec, &map, 4096).expect("load qwen3");

    let texts = [
        "The cat sat on the mat.",
        "A feline rested on the rug.",
        "Quantum chromodynamics describes the strong nuclear force.",
    ];
    // add_eos_token=true: last-token pooling reads the appended EOS position
    let eos = 151643u32;
    let seqs: Vec<Vec<u32>> = texts
        .iter()
        .map(|t| {
            let mut e = tok.encode(t).expect("encode");
            e.push(eos);
            e
        })
        .collect();

    let embs = m.embed(&seqs).expect("embed");
    for (i, e) in embs.iter().enumerate() {
        let norm = e.iter().map(|x| x * x).sum::<f32>().sqrt();
        eprintln!(
            "emb[{i}] dim={} norm={norm:.4} text={:?}",
            e.len(),
            texts[i]
        );
    }
    let sim_close = cos(&embs[0], &embs[1]);
    let sim_far_a = cos(&embs[0], &embs[2]);
    let sim_far_b = cos(&embs[1], &embs[2]);
    eprintln!("cos(cat, feline)   = {sim_close:.4}   <- similar, expect HIGH");
    eprintln!("cos(cat, quantum)  = {sim_far_a:.4}   <- different, expect LOW");
    eprintln!("cos(feline, quantum) = {sim_far_b:.4}");
    assert!(
        sim_close > sim_far_a && sim_close > sim_far_b,
        "similar pair must out-score dissimilar pairs"
    );
    eprintln!(
        "QWEN3 EMBED OK: similar pair {sim_close:.3} > dissimilar {sim_far_a:.3}/{sim_far_b:.3}"
    );
}
