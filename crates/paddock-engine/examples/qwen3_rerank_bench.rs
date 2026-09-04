//! Qwen3-Reranker throughput bench: score one query against N documents
//! (the RAG second-stage workload) and report docs/sec. The shared-prefix
//! cache should make the per-doc cost ~the suffix tokens only.
//!
//! Usage: qwen3_rerank_bench [n_docs]   (env QWEN3_RERANK_GGUF overrides)
// A development probe: it runs on a box its author is looking at, and a
// failure should stop it where it happened rather than be reported.
#![allow(clippy::unwrap_used)]

use std::sync::Arc;
use std::time::Instant;

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
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(128);
    let pack = std::env::var_os("PADDOCK_PACK")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../packs/cuda/build/pd-cuda-sm120.so")
        });
    let model = std::env::var("QWEN3_RERANK_GGUF").unwrap_or_else(|_| {
        "C:/dev/models/Qwen3-Reranker-0.6B-GGUF/Qwen3-Reranker-0.6B.Q8_0.gguf".into()
    });
    let exec = Arc::new(GpuExecutor::new(0, &pack).expect("cuda executor"));
    let map = MappedGguf::open(std::path::Path::new(&model)).expect("open gguf");
    let tok = GgufTokenizer::from_gguf(map.gguf()).expect("tokenizer");
    let mut m = GpuQwen3::load(exec, &map, 4096).expect("load qwen3 reranker");
    let yes = tok.token_to_id("yes").expect("'yes' token");
    let no = tok.token_to_id("no").expect("'no' token");

    let query = "What is the capital of France?";
    let docs: Vec<String> = (0..n)
        .map(|i| {
            if i == n / 3 {
                "Paris is the capital and most populous city of France, situated on the \
                 Seine river in the north of the country."
                    .to_owned()
            } else {
                format!(
                    "Report {i}: the distributed scheduler coordinates seasonal workload \
                     spikes across regions through a pipeline of validation, routing and \
                     replication stages, each of which can fail independently and must be \
                     retried with exponential backoff to preserve consistency across \
                     cluster {}.",
                    i % 7
                )
            }
        })
        .collect();
    let seqs: Vec<Vec<u32>> = docs
        .iter()
        .map(|d| tok.encode(&prompt(INSTRUCT, query, d)).expect("encode"))
        .collect();
    let tokens: usize = seqs.iter().map(Vec::len).sum();
    println!(
        "{n} docs, {tokens} tokens ({:.1} avg)",
        tokens as f64 / n as f64
    );

    let mut best = f64::INFINITY;
    for run in 0..4 {
        let t0 = Instant::now();
        let scores = m.rerank(&seqs, yes, no).expect("rerank");
        let dt = t0.elapsed().as_secs_f64();
        let top = scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i)
            .unwrap();
        if run == 0 {
            assert_eq!(top, n / 3, "relevant doc must rank first");
            continue; // warmup
        }
        best = best.min(dt);
        println!(
            "run {run}: {n} docs in {dt:.3}s = {:.1} docs/s",
            n as f64 / dt
        );
    }
    println!("QWEN3 RERANK BENCH: best {:.1} docs/s", n as f64 / best);
}
