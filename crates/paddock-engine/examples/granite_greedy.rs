//! Granite greedy parity probe - the bring-up acceptance harness.
//!
//! Loads the GGUF on the GPU model, prefills PROMPT token-by-token (the
//! milestone path), then greedy-decodes N tokens and prints them one per
//! line as `id<TAB>text`, plus the assembled continuation. The parity gate
//! diffs these ids/text against the newest llama.cpp release binary running
//! the identical GGUF greedy.
//!
//! Granite's two ways to be silently wrong, both of which show up here as a
//! divergence rather than an error:
//!   - a missed scalar multiplier (embedding ×12 / residual ×0.22 / logits
//!     ÷16 / KQ scale 0.0078125) - fluent but wrong continuation;
//!   - the rope convention: granite is llama.cpp ROPE_TYPE_NORM (interleaved
//!     pairs) while every other family here is NEOX. NEOX rope on granite
//!     degrades gradually with position, so a short probe can look fine -
//!     run N_GEN high enough to leave the first few positions behind.
//!
//! Usage: GRANITE_GGUF=... PADDOCK_PACK=... [GRANITE_PROMPT=...] [N_GEN=32]
//!        [MAX_CTX=4096] granite_greedy
//!
//! (GRANITE_PROMPT, not PROMPT: cmd.exe exports PROMPT="$P$G" and the
//! inherited env silently replaces the default - the laguna bring-up lost an
//! hour to exactly that.)
// A development probe: it runs on a box its author is looking at, and a
// failure should stop it where it happened rather than be reported.
#![allow(clippy::unwrap_used)]

use std::sync::Arc;

use paddock_engine::generator::Generator;
use paddock_engine::gpu::GpuExecutor;
use paddock_engine::gpu_model::granite::GpuGranite;
use paddock_models::mapped::MappedGguf;
use paddock_tokenizer::GgufTokenizer;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();
    let model = std::env::var("GRANITE_GGUF").expect("set GRANITE_GGUF");
    let pack = std::env::var("PADDOCK_PACK").expect("set PADDOCK_PACK");
    let prompt =
        std::env::var("GRANITE_PROMPT").unwrap_or_else(|_| "The capital of France is".to_owned());
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
    // granite 4.1 stamps add_bos_token = false; honour the file rather than
    // assuming, since a stray BOS shifts every position and breaks parity.
    if tok.add_bos
        && let Some(b) = tok.bos_id
    {
        ids.push(b);
    }
    ids.extend(tok.encode(&prompt).expect("encode"));
    eprintln!("prompt ids: {ids:?}");

    let t0 = std::time::Instant::now();
    let exec = Arc::new(GpuExecutor::new(0, pack.as_ref()).expect("executor"));
    let mut m = GpuGranite::load(exec, &map, max_ctx).expect("load");
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

    // GRANITE_TOP5=1: print the top-5 (id, logit) per step - the near-tie
    // classifier for parity divergences (diff the top-2 gap against
    // llama.cpp's n_probs at the flip point).
    let top5 = std::env::var_os("GRANITE_TOP5").is_some();

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
