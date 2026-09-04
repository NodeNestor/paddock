//! Laguna greedy parity probe - the bring-up acceptance harness.
//!
//! Loads the GGUF on the GPU model, prefills PROMPT token-by-token (the
//! milestone path), then greedy-decodes N tokens and prints them one per
//! line as `id<TAB>text`, plus the assembled continuation. The parity gate
//! diffs these ids/text against the newest llama.cpp release binary running
//! the identical GGUF greedy.
//!
//! Usage: LAGUNA_GGUF=... PADDOCK_PACK=... [LAGUNA_PROMPT=...] [N_GEN=32]
//!        [MAX_CTX=4096] laguna_greedy
//!
//! (LAGUNA_PROMPT, not PROMPT: on Windows cmd.exe exports PROMPT="$P$G" -
//! its prompt-format string - and the inherited env silently replaced the
//! default prompt on first bring-up. An hour of "why is it doing geometry".)
// A development probe: it runs on a box its author is looking at, and a
// failure should stop it where it happened rather than be reported.
#![allow(clippy::unwrap_used)]

use std::sync::Arc;

use paddock_engine::generator::Generator;
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
    let prompt =
        std::env::var("LAGUNA_PROMPT").unwrap_or_else(|_| "The capital of France is".to_owned());
    let n_gen: usize = std::env::var("N_GEN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(32);
    let max_ctx: usize = std::env::var("MAX_CTX")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4096);

    let map = MappedGguf::open(model.as_ref()).expect("open gguf");
    let tok = GgufTokenizer::from_gguf(map.gguf()).expect("tokenizer");
    let mut ids = Vec::new();
    if tok.add_bos {
        ids.push(tok.bos_id.expect("laguna has BOS"));
    }
    ids.extend(tok.encode(&prompt).expect("encode"));
    eprintln!("prompt ids: {ids:?}");

    let t0 = std::time::Instant::now();
    let exec = Arc::new(GpuExecutor::new(0, pack.as_ref()).expect("executor"));
    let mut m = GpuLaguna::load(exec, &map, max_ctx).expect("load");
    eprintln!("loaded in {:.1}s", t0.elapsed().as_secs_f32());

    let t1 = std::time::Instant::now();
    let mut logits = Vec::new();
    for &t in &ids {
        logits = m.forward(t).expect("prefill");
    }
    eprintln!(
        "prefilled {} tokens in {:.2}s",
        ids.len(),
        t1.elapsed().as_secs_f32()
    );

    // LAGUNA_TOP5=1: print the top-5 (id, logit) per step - the near-tie
    // classifier for parity divergences (diff the top-2 gap vs llama.cpp's
    // n_probs at the flip point)
    let top5 = std::env::var_os("LAGUNA_TOP5").is_some();

    let t2 = std::time::Instant::now();
    let mut out = Vec::with_capacity(n_gen);
    for _ in 0..n_gen {
        let mut best = 0usize;
        for (i, v) in logits.iter().enumerate() {
            if *v > logits[best] {
                best = i;
            }
        }
        if top5 {
            let mut idx: Vec<usize> = (0..logits.len()).collect();
            idx.sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]));
            let line: Vec<String> = idx[..5]
                .iter()
                .map(|&i| {
                    format!(
                        "{}={:.4}{:?}",
                        i,
                        logits[i],
                        tok.decode(&[i as u32], false).unwrap_or_default()
                    )
                })
                .collect();
            eprintln!("top5: {}", line.join(" "));
        }
        let id = best as u32;
        out.push(id);
        println!(
            "{id}\t{}",
            tok.decode(&[id], false).unwrap_or_default().escape_debug()
        );
        logits = m.forward(id).expect("decode");
    }
    eprintln!(
        "generated {n_gen} tokens in {:.2}s ({:.1} tok/s)",
        t2.elapsed().as_secs_f32(),
        n_gen as f32 / t2.elapsed().as_secs_f32()
    );
    println!("---\n{}", tok.decode(&out, false).unwrap_or_default());
}
