//! Gemma 4 greedy parity probe - the bring-up acceptance harness.
//!
//! Loads the GGUF on the GPU model, prefills PROMPT token-by-token (the
//! milestone path), then greedy-decodes N tokens and prints them one per
//! line as `id<TAB>text`, plus the assembled continuation. The parity gate
//! diffs these ids/text against the llama.cpp oracle running the
//! identical GGUF greedy.
//!
//! Usage: GEMMA4_GGUF=... PADDOCK_PACK=... [PROMPT=...] [N_GEN=32]
//!        [MAX_CTX=4096] gemma4_greedy

use std::sync::Arc;

use paddock_engine::generator::Generator;
use paddock_engine::gpu::GpuExecutor;
use paddock_engine::gpu_model::gemma4::GpuGemma4;
use paddock_models::mapped::MappedGguf;
use paddock_tokenizer::GgufTokenizer;

fn main() {
    let model = std::env::var("GEMMA4_GGUF").expect("set GEMMA4_GGUF");
    let pack = std::env::var("PADDOCK_PACK").expect("set PADDOCK_PACK");
    let prompt = std::env::var("PROMPT").unwrap_or_else(|_| "The capital of France is".to_owned());
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
        ids.push(tok.bos_id.expect("gemma4 has BOS"));
    }
    ids.extend(tok.encode(&prompt).expect("encode"));
    eprintln!("prompt ids: {ids:?}");

    let t0 = std::time::Instant::now();
    let exec = Arc::new(GpuExecutor::new(0, pack.as_ref()).expect("executor"));
    let mut m = GpuGemma4::load(exec, &map, max_ctx).expect("load");
    eprintln!("loaded in {:.1}s", t0.elapsed().as_secs_f32());

    // PF_MODE=serial forces the token-by-token trait default (the batched
    // path's own A/B reference); default = the batched prefill lane
    let t1 = std::time::Instant::now();
    let logits_res = if std::env::var("PF_MODE").as_deref() == Ok("serial") {
        let mut l = Vec::new();
        for &t in &ids {
            l = m.forward(t).expect("prefill");
        }
        l
    } else {
        m.forward_prefill_stream(&ids).expect("prefill")
    };
    let mut logits = logits_res;
    eprintln!(
        "prefill {} tokens in {:.2}s",
        ids.len(),
        t1.elapsed().as_secs_f32()
    );

    let argmax = |l: &[f32]| -> u32 {
        let mut best = 0usize;
        for i in 1..l.len() {
            if l[i] > l[best] {
                best = i;
            }
        }
        best as u32
    };

    // top-5 of the first decode step - the near-tie diagnostic when a greedy
    // sequence diverges from the oracle at token 0
    let mut top: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    top.sort_by(|a, b| b.1.total_cmp(&a.1));
    eprintln!(
        "first-step top5: {:?}",
        top[..5]
            .iter()
            .map(|(i, l)| (
                *i as u32,
                *l,
                tok.id_to_token(*i as u32).unwrap_or_default()
            ))
            .collect::<Vec<_>>()
    );

    let t2 = std::time::Instant::now();
    let mut out = Vec::new();
    for _ in 0..n_gen {
        let next = argmax(&logits);
        out.push(next);
        println!("{next}\t{:?}", tok.id_to_token(next).unwrap_or_default());
        logits = m.forward(next).expect("decode");
    }
    eprintln!(
        "decode {} tokens in {:.2}s ({:.1} tok/s)",
        n_gen,
        t2.elapsed().as_secs_f32(),
        n_gen as f32 / t2.elapsed().as_secs_f32()
    );
    // continuation as one debug-escaped line - multi-line text stays diffable
    // by line-oriented parity harnesses
    let text = tok.decode(&out, false).expect("decode text");
    println!("==={text:?}");
    // raw bytes for byte-exact `cmp` against the oracle's continuation
    if let Ok(path) = std::env::var("OUT_FILE") {
        std::fs::write(path, &text).expect("write OUT_FILE");
    }
}
