//! Teacher-forced perplexity for Qwen3.8-Flash-Next - the quality gate for the
//! chartered 8-bit DENSE class election.
//!
//! Scores a fixed corpus token by token: prefill the first `PPL_PREFIX` tokens,
//! then feed each true token through the decode tick and read the next
//! position's logits, accumulating the negative log-likelihood of the actual
//! next token. `PPL = exp(mean NLL)`.
//!
//! The point is to run it twice on the identical corpus with one env apart -
//! `PADDOCK_Q38FN_DENSE=bf16` (the parity class, checkpoint bytes as shipped)
//! against an 8-bit class - and compare both axes plus the per-position CSV.
//!
//! No vacuous-pass trap here (the one the b200 f8t gate hit): the dense class
//! is a LOAD-TIME election, so prefill and every decode tick run it. What this
//! harness does not reach: batch > 1 (no lane scores batched yet, repo-wide),
//! and the KV class, which is fixed at f16 in this lane.
//!
//! Usage:
//!   QWEN4EXP_DIR=... PADDOCK_PACK=... PPL_CORPUS=wiki.test.raw \
//!   [PPL_MAX_TOK=4096] [PPL_PREFIX=1] [PPL_OUT=perpos.csv] q38fn_ppl
//! Prints: token count, mean NLL (nats), perplexity, self-top-1.
//! Optional CSV: pos,true_id,nll,argmax_id - the cross-leg comparison
//! (top-1 agreement, positions-worse, median abs delta) is computed from two.
// A development probe: it runs on a box its author is looking at, and a
// failure should stop it where it happened rather than be reported.
#![allow(clippy::unwrap_used)]

use std::io::Write;
use std::sync::Arc;

use paddock_engine::gpu::GpuExecutor;
use paddock_engine::gpu_model::qwen4exp::{Qwen4ExpGpu, dense_class_from_env};
use paddock_tokenizer::GgufTokenizer;

/// log-softmax value for `tid` plus the argmax index, in f64 - a 248320-wide
/// vocab loses bits to a naive f32 logsumexp.
fn logprob_and_argmax(logits: &[f32], tid: u32) -> (f64, u32) {
    let mut max = f32::NEG_INFINITY;
    let mut amax = 0u32;
    for (i, &v) in logits.iter().enumerate() {
        if v > max {
            max = v;
            amax = i as u32;
        }
    }
    let mut sum = 0.0f64;
    for &v in logits {
        sum += ((v - max) as f64).exp();
    }
    let lse = (max as f64) + sum.ln();
    ((logits[tid as usize] as f64) - lse, amax)
}

fn main() {
    let dir = std::path::PathBuf::from(
        std::env::var("QWEN4EXP_DIR")
            .unwrap_or_else(|_| "/models/Qwen3.8-Flash-Next-NVFP4".to_string()),
    );
    let pack = std::env::var_os("PADDOCK_PACK")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../packs/cuda/build/pd-cuda-sm120.so")
        });
    let corpus = match std::env::var("PPL_CORPUS") {
        Ok(p) => std::fs::read_to_string(&p).expect("read PPL_CORPUS"),
        Err(_) => panic!("PPL_CORPUS required (wikitext-2-raw wiki.test.raw)"),
    };
    let max_tok: usize = std::env::var("PPL_MAX_TOK")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4096);
    let prefix: usize = std::env::var("PPL_PREFIX")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1)
        .max(1);

    let exec = Arc::new(GpuExecutor::new(0, &pack).expect("executor"));
    let tok = GgufTokenizer::from_hf_dir(&dir).expect("tokenizer from HF dir");
    let mut ids = tok.encode(&corpus).expect("encode");
    ids.truncate(max_tok);
    assert!(ids.len() > 2, "corpus too short");
    assert!(prefix < ids.len(), "PPL_PREFIX >= corpus tokens");

    let t0 = std::time::Instant::now();
    // +8 of slack so the cursor never runs into the cap on the last token
    let mut m = Qwen4ExpGpu::load(&exec, &dir, ids.len() + 8).expect("load qwen4exp");
    eprintln!(
        "loaded in {:.1}s - dense class {}",
        t0.elapsed().as_secs_f32(),
        m.dense_class()
    );

    let mut out = std::env::var_os("PPL_OUT").map(|p| {
        let mut f = std::fs::File::create(p).expect("PPL_OUT");
        writeln!(f, "pos,true_id,nll,argmax_id").unwrap();
        f
    });

    let mut logits = m.forward_prompt(&ids[0..prefix]).expect("prefill");
    let (mut nll_sum, mut n, mut top1_hits) = (0.0f64, 0usize, 0usize);
    for i in prefix..ids.len() {
        let (lp, amax) = logprob_and_argmax(&logits, ids[i]);
        nll_sum += -lp;
        n += 1;
        if amax == ids[i] {
            top1_hits += 1;
        }
        if let Some(f) = out.as_mut() {
            writeln!(f, "{},{},{:.6},{}", i, ids[i], -lp, amax).unwrap();
        }
        if i + 1 < ids.len() {
            logits = m.decode_step(ids[i]).expect("decode step");
        }
    }
    let mean_nll = nll_sum / n as f64;
    println!(
        "PPL[dense-{}] prefix={prefix} tokens={n} mean_nll={mean_nll:.5} ppl={:.5} \
         self_top1={:.4}",
        dense_class_from_env().label(),
        mean_nll.exp(),
        top1_hits as f64 / n as f64
    );
}
