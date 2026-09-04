//! G3 speculative-decoding benchmark: n-gram (prompt-lookup) drafted greedy
//! vs the plain graph-resident greedy loop, on gpt-oss-20b. Spec wins scale
//! with output repetition - report repetitive/agentic AND honest prose
//! numbers. Args: `<max_new>` (default 192) `<n_draft>` (default 7).
//! Set PADDOCK_SPEC_DEBUG=1 for per-run round/acceptance stats.
// A development probe: it runs on a box its author is looking at, and a
// failure should stop it where it happened rather than be reported.
#![allow(clippy::unwrap_used)]

use std::sync::Arc;

use paddock_engine::gpu::GpuExecutor;
use paddock_engine::gpu_model::gpt_oss::GpuGptOss;
use paddock_models::mapped::MappedGguf;
use paddock_tokenizer::GgufTokenizer;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let max_new: usize = args
        .get(1)
        .map(|s| s.parse().expect("max_new"))
        .unwrap_or(192);
    let n_draft: usize = args
        .get(2)
        .map(|s| s.parse().expect("n_draft"))
        .unwrap_or(7);
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
    let mut m = GpuGptOss::load(exec, &map, 4096).expect("load");

    let cases: &[(&str, &str)] = &[
        (
            "repeat",
            "Repeat the following paragraph exactly, word for word, over and \
             over without stopping: The quick brown fox jumps over the lazy dog \
             while the cat watches from the window and the birds sing in the \
             garden by the old stone wall.\n\nThe quick brown fox jumps over \
             the lazy dog while the cat watches from the window and the birds \
             sing in the garden by the old stone wall. The quick brown fox",
        ),
        (
            "counting",
            "Let me count slowly from one to two hundred without skipping any \
             number: one, two, three, four, five, six, seven, eight, nine, ten, \
             eleven, twelve, thirteen",
        ),
        (
            "agentic",
            "Convert each item to a JSON object with fields name, id and price, \
             one per line:\napple 1 3.50\nbanana 2 1.25\ncherry 3 8.00\ndamson 4 \
             2.75\nelderberry 5 9.10\nfig 6 4.20\ngrape 7 2.30\n\n{\"name\": \
             \"apple\", \"id\": 1, \"price\": 3.50}\n",
        ),
        (
            "prose",
            "Once upon a time, in a quiet village nestled between rolling hills, \
             there lived an old clockmaker who",
        ),
    ];

    for (name, text) in cases {
        let prompt = tok.encode(text).expect("encode");
        // warm both paths (captures graphs, loads prefix cache)
        m.generate_greedy(&prompt, 16).expect("warm plain");
        m.generate_greedy_spec(&prompt, 16, n_draft)
            .expect("warm spec");

        // GENERATION-ONLY rate: subtract a short run of the same path so the
        // prompt ingestion cost (per-token forward_one for plain, cached
        // prefill for spec) drops out of the comparison.
        let head = 8usize;
        let mut best_plain = 0.0f64;
        let mut plain_out = Vec::new();
        for _ in 0..3 {
            let t0 = std::time::Instant::now();
            m.generate_greedy(&prompt, head).expect("plain head");
            let t_head = t0.elapsed().as_secs_f64();
            let t1 = std::time::Instant::now();
            plain_out = m.generate_greedy(&prompt, max_new).expect("plain");
            let t_full = t1.elapsed().as_secs_f64();
            best_plain = best_plain.max((max_new - head) as f64 / (t_full - t_head));
        }
        let mut best_spec = 0.0f64;
        let mut spec_out = Vec::new();
        for _ in 0..3 {
            let t0 = std::time::Instant::now();
            m.generate_greedy_spec(&prompt, head, n_draft)
                .expect("spec head");
            let t_head = t0.elapsed().as_secs_f64();
            let t1 = std::time::Instant::now();
            spec_out = m
                .generate_greedy_spec(&prompt, max_new, n_draft)
                .expect("spec");
            let t_full = t1.elapsed().as_secs_f64();
            best_spec = best_spec.max((max_new - head) as f64 / (t_full - t_head));
        }
        let same = plain_out == spec_out;
        println!(
            "{name:>9}: plain {best_plain:6.1} tok/s | spec(k={n_draft}) {best_spec:6.1} tok/s | {:.2}x | streams {}",
            best_spec / best_plain,
            if same {
                "IDENTICAL"
            } else {
                "differ (class near-tie)"
            }
        );
    }
}
