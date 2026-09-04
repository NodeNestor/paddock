//! MTP speculative-decode benchmark for qwen35 - drives the
//! `generate_greedy_spec_batch` machinery (draft ring + verify chunk
//! + ragged commit) on the 35B MoE. Compares the
//!   plain greedy serving loop (forward_prefill_slot + forward_batch ticks)
//!   against the spec loop at the same batch, and checks BIT-PARITY: greedy
//!   spec decode must reproduce the plain greedy stream exactly (the built-in
//!   correctness oracle - acceptance only ever commits trunk argmaxes).
//!   Args: `<B>` (default 1) `<tokens per seq>` (default 256) `<K drafts>` (2).
//!   Env: QWEN35_GGUF / PADDOCK_PACK as usual; PADDOCK_SPEC_PHASE_TIME=1 for
//!   the per-phase wall breakdown.
// A development probe: it runs on a box its author is looking at, and a
// failure should stop it where it happened rather than be reported.
#![allow(clippy::unwrap_used)]

use std::sync::Arc;

use paddock_engine::gpu::GpuExecutor;
use paddock_engine::gpu_model::qwen35::GpuQwen35;
use paddock_models::mapped::MappedGguf;
use paddock_tokenizer::GgufTokenizer;

const PROSE: &str = "Write a detailed, flowing essay about the history of \
    container shipping (variant %), its economics, key inventions, and \
    global impact. Begin:";

fn argmax(l: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, v) in l.iter().enumerate() {
        if *v > l[best] {
            best = i;
        }
    }
    best as u32
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let b: usize = args.get(1).map(|s| s.parse().expect("B")).unwrap_or(1);
    let n_tok: usize = args
        .get(2)
        .map(|s| s.parse().expect("tokens"))
        .unwrap_or(256);
    let k: usize = args.get(3).map(|s| s.parse().expect("K")).unwrap_or(2);
    let model = std::env::var("QWEN35_GGUF")
        .unwrap_or_else(|_| "/llms/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-Q8_0.gguf".to_string());
    let pack = std::env::var_os("PADDOCK_PACK")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../packs/cuda/build/pd-cuda-sm120.so")
        });
    let exec = Arc::new(GpuExecutor::new(0, &pack).expect("executor"));
    let map = MappedGguf::open(std::path::Path::new(&model)).expect("open gguf");
    let tok = GgufTokenizer::from_gguf(map.gguf()).expect("tokenizer");
    let mut m = GpuQwen35::load(exec, &map, 8192).expect("load qwen35");
    m.enable_batch(b).expect("enable_batch");
    let vocab = m.vocab;

    let prompts: Vec<Vec<u32>> = (0..b)
        .map(|s| {
            tok.encode(&PROSE.replace('%', &s.to_string()))
                .expect("encode")
        })
        .collect();

    // ---- plain greedy serving loop (prefill + one forward_batch per tick)
    let t_plain0 = std::time::Instant::now();
    let mut pending = vec![0u32; b];
    let mut pos = vec![0u32; b];
    for (s, p) in prompts.iter().enumerate() {
        let logits = m.forward_prefill_slot(s, p).expect("prefill");
        pending[s] = argmax(&logits);
        pos[s] = p.len() as u32;
    }
    let t_dec0 = std::time::Instant::now();
    let mut plain: Vec<Vec<u32>> = (0..b).map(|s| vec![pending[s]]).collect();
    while plain.iter().any(|st| st.len() < n_tok) {
        let logits = m.forward_batch(&pending, &pos).expect("fwd");
        for s in 0..b {
            pos[s] += 1;
            pending[s] = argmax(&logits[s * vocab..(s + 1) * vocab]);
            if plain[s].len() < n_tok {
                plain[s].push(pending[s]);
            }
        }
    }
    let dt_dec = t_dec0.elapsed().as_secs_f64();
    let dt_plain = t_plain0.elapsed().as_secs_f64();
    let plain_dec_rate = (b * (n_tok - 1)) as f64 / dt_dec;

    // ---- MTP spec loop. Warmup call absorbs MTP warm + graph captures (and
    // seeds the prefix cache so the timed call's internal re-prefill is ~free);
    // the timed call is then round-dominated and timed directly.
    let _ = dt_plain;
    m.generate_greedy_spec_batch(&prompts, 8, k)
        .expect("spec warmup");
    let t_spec0 = std::time::Instant::now();
    let spec = m
        .generate_greedy_spec_batch(&prompts, n_tok, k)
        .expect("spec");
    let dt_spec_dec = t_spec0.elapsed().as_secs_f64();
    let spec_dec_rate = (b * (n_tok - 1)) as f64 / dt_spec_dec;

    let mut same = true;
    for s in 0..b {
        let n = n_tok.min(plain[s].len()).min(spec[s].len());
        if plain[s][..n] != spec[s][..n] {
            same = false;
            let d = (0..n).find(|&i| plain[s][i] != spec[s][i]).unwrap();
            eprintln!(
                "slot {s}: DIVERGES at token {d} (plain {} vs spec {})",
                plain[s][d], spec[s][d]
            );
        }
    }
    println!(
        "B={b} K={k} n={n_tok}: plain {plain_dec_rate:7.1} tok/s | spec {spec_dec_rate:7.1} tok/s | {:.2}x | parity {}",
        spec_dec_rate / plain_dec_rate,
        if same { "IDENTICAL" } else { "DIVERGED" }
    );
    if std::env::var_os("SPEC_TEXT").is_some() {
        let n = n_tok.min(plain[0].len()).min(spec[0].len());
        println!(
            "--- plain[0] ---\n{}",
            tok.decode(&plain[0][..n], true).unwrap_or_default()
        );
        println!(
            "--- spec[0] ---\n{}",
            tok.decode(&spec[0][..n], true).unwrap_or_default()
        );
    }
}
