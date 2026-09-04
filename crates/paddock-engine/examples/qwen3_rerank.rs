//! Qwen3-Reranker bring-up smoke: score query-document relevance and verify
//! the ordering - the relevant document must out-score the irrelevant ones.
//! Uses the official Qwen3-Reranker prompt (a generative yes/no relevance
//! judge; score = P(yes) at the final position).
//!
//! Usage: qwen3_rerank [gguf]   (env QWEN3_RERANK_GGUF overrides)
// A development probe: it runs on a box its author is looking at, and a
// failure should stop it where it happened rather than be reported.
#![allow(clippy::unwrap_used)]

use std::sync::Arc;

use paddock_engine::gpu::GpuExecutor;
use paddock_engine::gpu_model::qwen3::GpuQwen3;
use paddock_models::mapped::MappedGguf;
use paddock_tokenizer::GgufTokenizer;

const INSTRUCT: &str = "Given a web search query, retrieve relevant passages that answer the query";

fn prompt(instruction: &str, query: &str, doc: &str) -> String {
    format!(
        "<|im_start|>system\nJudge whether the Document meets the requirements \
         based on the Query and the Instruct provided. Note that the answer can \
         only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n<Instruct>: \
         {instruction}\n<Query>: {query}\n<Document>: {doc}<|im_end|>\n\
         <|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
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
        .or_else(|| std::env::var("QWEN3_RERANK_GGUF").ok())
        .unwrap_or_else(|| {
            "C:/dev/models/Qwen3-Reranker-0.6B-GGUF/Qwen3-Reranker-0.6B.Q8_0.gguf".into()
        });
    let exec = Arc::new(GpuExecutor::new(0, &pack).expect("cuda executor"));
    let map = MappedGguf::open(std::path::Path::new(&model)).expect("open gguf");
    let tok = GgufTokenizer::from_gguf(map.gguf()).expect("tokenizer");
    let mut m = GpuQwen3::load(exec, &map, 4096).expect("load qwen3 reranker");

    let yes = tok.token_to_id("yes").expect("'yes' token");
    let no = tok.token_to_id("no").expect("'no' token");
    eprintln!("yes_id={yes} no_id={no}");

    let query = "What is the capital of France?";
    let docs = [
        "Paris is the capital and most populous city of France.",
        "The Great Wall of China is a series of fortifications in northern China.",
        "France is a country in Western Europe with several overseas territories.",
        "Photosynthesis converts light energy into chemical energy in plants.",
    ];
    let seqs: Vec<Vec<u32>> = docs
        .iter()
        .map(|d| tok.encode(&prompt(INSTRUCT, query, d)).expect("encode"))
        .collect();

    // batched (all docs in one pass) vs per-doc (batch=1) - a divergence
    // isolates a cross-slot attention / pooling bug in the ragged batch path
    let scores = m.rerank(&seqs, yes, no).expect("rerank");
    let solo: Vec<f32> = seqs
        .iter()
        .map(|s| {
            m.rerank(std::slice::from_ref(s), yes, no)
                .expect("rerank solo")[0]
        })
        .collect();
    eprintln!("batched: {scores:?}");
    eprintln!("solo   : {solo:?}");
    let mut ranked: Vec<(usize, f32)> = solo.iter().copied().enumerate().collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
    eprintln!("ranking (query: {query:?}):");
    for (i, s) in &ranked {
        eprintln!("  {s:.4}  {:?}", docs[*i]);
    }
    // self-consistency gate: the ragged batched pass must reproduce the
    // per-document (batch=1) scores - proves per-sequence attention isolation
    // in the batch (a cross-slot leak mushes every score toward ~1).
    for (i, (b, s)) in scores.iter().zip(&solo).enumerate() {
        assert!(
            (b - s).abs() < 5e-3,
            "doc {i}: batched {b} != solo {s} (batch contamination)"
        );
    }
    // the Paris passage (doc 0) must rank first; the off-topic ones (1,3) last
    assert_eq!(ranked[0].0, 0, "the relevant passage must rank #1");
    assert!(
        solo[0] > solo[1] && solo[0] > solo[3],
        "relevant must beat irrelevant"
    );
    eprintln!(
        "QWEN3 RERANK OK: relevant {:.3} > irrelevant {:.3}/{:.3}; batched==solo",
        solo[0], solo[1], solo[3]
    );
}
