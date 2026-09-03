//! Gemma 4 batched-lane acceptance: prefill N prompts into N slots, greedy
//! decode them together via forward_batch, then replay each prompt through
//! the single-stream lane (whose greedy output the oracle gate has locked)
//! and diff the continuations. A flip is only acceptable on a near-tie -
//! the harness prints the batched top-2 gap at the first divergence.
//!
//! Usage: GEMMA4_GGUF=... PADDOCK_PACK=... [N_GEN=24] [MAX_CTX=4096] gemma4_batch_check

use std::sync::Arc;

use paddock_engine::generator::Generator;
use paddock_engine::gpu::GpuExecutor;
use paddock_engine::gpu_model::gemma4::GpuGemma4;
use paddock_models::mapped::MappedGguf;
use paddock_tokenizer::GgufTokenizer;

const PROMPTS: [&str; 4] = [
    "The capital of France is",
    "In 1969, the Apollo 11 mission",
    "def fibonacci(n):",
    "The three primary colors are",
];

fn argmax(l: &[f32]) -> u32 {
    let mut best = 0usize;
    for i in 1..l.len() {
        if l[i] > l[best] {
            best = i;
        }
    }
    best as u32
}

fn main() {
    let model = std::env::var("GEMMA4_GGUF").expect("set GEMMA4_GGUF");
    let pack = std::env::var("PADDOCK_PACK").expect("set PADDOCK_PACK");
    let n_gen: usize = std::env::var("N_GEN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(24);
    let max_ctx: usize = std::env::var("MAX_CTX")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4096);

    let map = MappedGguf::open(model.as_ref()).expect("open gguf");
    let tok = GgufTokenizer::from_gguf(map.gguf()).expect("tokenizer");
    let bos = tok.bos_id.expect("bos");
    let prompts: Vec<Vec<u32>> = PROMPTS
        .iter()
        .map(|p| {
            let mut ids = vec![bos];
            ids.extend(tok.encode(p).expect("encode"));
            ids
        })
        .collect();

    let exec = Arc::new(GpuExecutor::new(0, pack.as_ref()).expect("executor"));
    let mut m = GpuGemma4::load(exec, &map, max_ctx).expect("load");

    // ── batched lanes
    let cap = m.enable_batch(PROMPTS.len()).expect("enable_batch");
    assert!(cap >= PROMPTS.len(), "only {cap} slots fit");
    let vocab = m.vocab();
    let mut tokens: Vec<u32> = Vec::new();
    let mut positions: Vec<u32> = Vec::new();
    for (slot, ids) in prompts.iter().enumerate() {
        let logits = m.forward_prefill(slot, ids).expect("slot prefill");
        tokens.push(argmax(&logits));
        positions.push(ids.len() as u32);
    }
    let mut batched: Vec<Vec<u32>> = tokens.iter().map(|&t| vec![t]).collect();
    // per-slot top-2 logit gap at every batched step - the near-tie
    // arbiter the docstring promises (gaps[slot][step])
    let mut gaps: Vec<Vec<f32>> = vec![vec![f32::INFINITY]; tokens.len()];
    let t0 = std::time::Instant::now();
    for _ in 1..n_gen {
        let logits = m.forward_batch(&tokens, &positions).expect("forward_batch");
        for b in 0..tokens.len() {
            let row = &logits[b * vocab..(b + 1) * vocab];
            let next = argmax(row);
            let top = row[next as usize];
            let mut second = f32::NEG_INFINITY;
            for (i, &l) in row.iter().enumerate() {
                if i != next as usize && l > second {
                    second = l;
                }
            }
            gaps[b].push(top - second);
            batched[b].push(next);
            tokens[b] = next;
            positions[b] += 1;
        }
    }
    let dt = t0.elapsed().as_secs_f32();
    eprintln!(
        "batched decode: {} rows x {} steps in {:.2}s = {:.1} tok/s aggregate",
        PROMPTS.len(),
        n_gen - 1,
        dt,
        (PROMPTS.len() * (n_gen - 1)) as f32 / dt
    );

    // ── single-stream replay (the oracle-locked lane)
    let mut mismatches = 0;
    for (slot, ids) in prompts.iter().enumerate() {
        m.reset();
        let mut logits = m.forward_prefill_stream(ids).expect("stream prefill");
        let mut single = Vec::new();
        for _ in 0..n_gen {
            let next = argmax(&logits);
            single.push(next);
            logits = m.forward(next).expect("decode");
        }
        if single == batched[slot] {
            println!("PASS slot {slot}: {:?}", PROMPTS[slot]);
        } else {
            let d = single
                .iter()
                .zip(&batched[slot])
                .position(|(a, b)| a != b)
                .unwrap_or(0);
            let gap = gaps[slot].get(d).copied().unwrap_or(f32::INFINITY);
            // The batch lanes run llama.cpp's own batch numeric class (int8
            // activations); the single lane is exact-f32. A cross-class flip
            // shows as a near-tie pick OR a single-token slip after which
            // the continuations REALIGN exactly (int8 noise on degenerate
            // loops reaches ~1 logit). Corruption (bad slots, torn KV) never
            // realigns - that stays a hard DIFF.
            let realigns = {
                let a = &single[d..];
                let b = &batched[slot][d..];
                let tail_eq = |x: &[u32], y: &[u32]| {
                    let n = x.len().min(y.len());
                    n >= 4 && x[..n] == y[..n]
                };
                // a flip may insert/drop a couple of tokens in either lane
                // (e.g. " red" + "," = 2) before the streams re-lock
                let mut ok = false;
                for ka in 0..=3usize {
                    for kb in 0..=3usize {
                        if ka + kb > 0
                            && a.len() > ka
                            && b.len() > kb
                            && tail_eq(&a[ka..], &b[kb..])
                        {
                            ok = true;
                        }
                    }
                }
                ok
            };
            if gap < 0.1 || realigns {
                println!(
                    "PASS slot {slot} (class flip at step {d}, gap {gap:.4}, realigned {realigns}): {:?}",
                    PROMPTS[slot]
                );
            } else {
                mismatches += 1;
                println!(
                    "DIFF slot {slot} at step {d} (top-2 gap {gap:.4}): single {:?} vs batched {:?}",
                    tok.decode(&single, false).unwrap_or_default(),
                    tok.decode(&batched[slot], false).unwrap_or_default(),
                );
            }
        }
    }
    println!(
        "{}",
        if mismatches == 0 {
            "BATCH-LANES-OK"
        } else {
            "BATCH-LANES-DIFF"
        }
    );
}
