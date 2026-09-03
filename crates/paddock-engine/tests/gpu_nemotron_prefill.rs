//! Bulk-vs-serial prefill gate for the nemotron chunked prefill.
//! The serial token-by-token walk is the parity-pinned reference
//! (it carries the oracle gate vs the vLLM reference dump); the bulk path must
//! reproduce it through a NEAR-EXACT band:
//!
//! Exact equality is not expected - the bulk lane runs the mamba in/out
//! projections as W8A8 f8row GEMM (dynamic per-token e4m3 activations, the
//! checkpoint's own W8A8 class) where decode runs the W8A16 f8r GEMV, and
//! the attention/tile + gemm_f32 summation orders differ. The gates are:
//! same top-1 at the prompt boundary, an identical greedy continuation, and
//! percent-band logit closeness - the same class of gate the arbiter oracle
//! uses. Prompt length 700 crosses the 512 chunk boundary deliberately (conv
//! window + scan state + KV position carry across chunks is the risk).

mod common;

use paddock_engine::generator::Generator;
use paddock_engine::gpu_model::nemotron::GpuNemotron;

const CKPT_ENV: &str = "NEMOTRON_NVFP4_DIR";
const CKPT_DIR: &str = "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4";
const ORACLE: &str = "/models/nemotron-battery/oracle/decoder-oracle.json";
const PROMPT_LEN: usize = 700;
const GREEDY_STEPS: usize = 24;

fn argmax(l: &[f32]) -> u32 {
    let mut bi = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &v) in l.iter().enumerate() {
        if v > bv {
            bv = v;
            bi = i;
        }
    }
    bi as u32
}

/// Serve-geometry smoke: NEMOTRON_LONG="<n_tokens>,<max_ctx>" runs a bulk
/// prefill at that exact shape and one decode step - the localization
/// harness for launch failures that only show at serving lengths.
#[test]
fn bulk_prefill_long_smoke() {
    let Ok(spec) = std::env::var("NEMOTRON_LONG") else {
        return;
    };
    let (n_tok, max_ctx) = spec.split_once(',').expect("NEMOTRON_LONG=n,ctx");
    let (n_tok, max_ctx): (usize, usize) =
        (n_tok.parse().expect("n"), max_ctx.parse().expect("ctx"));
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    if !exec.has_nemotron_prefill_f8() {
        common::missing("pack lacks the bulk-prefill kernel set");
        return;
    }
    let Some(dir) = common::model_dir(CKPT_ENV, &[CKPT_DIR]) else {
        return;
    };
    let mut model = GpuNemotron::load_dir(exec, &dir, max_ctx).expect("load");
    let prompt: Vec<u32> = (0..n_tok).map(|i| 1000 + (i % 5000) as u32).collect();
    // replicate the serve's request order: a short serial request (prefill
    // under the bulk threshold + decode) before the long bulk one
    model.reset();
    for &t in &prompt[..6] {
        model.forward(t).expect("serial pre-request");
    }
    for _ in 0..8 {
        model.forward(1000).expect("serial pre-decode");
    }
    model.reset();
    let logits = model.forward_prefill_stream(&prompt).expect("bulk prefill");
    let tok = argmax(&logits);
    model.forward(tok).expect("decode step");
    println!("long smoke ok: {n_tok} tokens @ max_ctx {max_ctx}, top-1 {tok}");
}

#[test]
fn bulk_prefill_matches_serial() {
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    if !exec.has_mamba2() || !exec.has_nvf4_moe() || !exec.has_nvf4_ckpt() {
        common::missing("pack lacks nemotron kernels (cc != 12.0?)");
        return;
    }
    if !exec.has_nemotron_prefill_f8() {
        common::missing("pack lacks the bulk-prefill kernel set");
        return;
    }
    let Some(dir) = common::model_dir(CKPT_ENV, &[CKPT_DIR]) else {
        return;
    };

    // prompt: real token ids cycled from the oracle prompt (deterministic,
    // valid ids; content coherence doesn't matter for a bulk-vs-serial diff)
    let oracle_path = std::env::var("NEMOTRON_ORACLE").unwrap_or_else(|_| ORACLE.into());
    let Ok(raw) = std::fs::read(&oracle_path) else {
        common::missing(&format!("no oracle dump at {oracle_path}"));
        return;
    };
    let oracle: serde_json::Value = serde_json::from_slice(&raw).expect("oracle json");
    let seed_ids: Vec<u32> = oracle["prompt_ids"]
        .as_array()
        .expect("prompt_ids")
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    let prompt: Vec<u32> = (0..PROMPT_LEN)
        .map(|i| seed_ids[i % seed_ids.len()])
        .collect();

    let mut model = GpuNemotron::load_dir(exec, &dir, 4096).expect("load");

    // ---- serial reference walk + greedy continuation ----------------------
    model.reset();
    let mut logits_s = Vec::new();
    for &t in &prompt {
        logits_s = model.forward(t).expect("serial forward");
    }
    let mut ids_s = Vec::with_capacity(GREEDY_STEPS);
    let mut l = logits_s.clone();
    for _ in 0..GREEDY_STEPS {
        let tok = argmax(&l);
        ids_s.push(tok);
        l = model.forward(tok).expect("serial decode");
    }

    // ---- bulk prefill + the same greedy continuation ----------------------
    model.reset();
    let logits_b = model.forward_prefill_stream(&prompt).expect("bulk prefill");
    let mut ids_b = Vec::with_capacity(GREEDY_STEPS);
    let mut l = logits_b.clone();
    for _ in 0..GREEDY_STEPS {
        let tok = argmax(&l);
        ids_b.push(tok);
        l = model.forward(tok).expect("bulk-side decode");
    }

    // logit closeness at the prompt boundary. Exactness is impossible by
    // construction (W8A8 dynamic-act projections + different summation
    // orders across 52 layers); the drift must look like smooth numerics
    // noise, not structure: gate the mean |delta| tightly, report the tail.
    let rms = (logits_s
        .iter()
        .map(|v| (*v as f64) * (*v as f64))
        .sum::<f64>()
        / logits_s.len() as f64)
        .sqrt();
    let mut deltas: Vec<f64> = logits_s
        .iter()
        .zip(logits_b.iter())
        .map(|(a, b)| (*a as f64 - *b as f64).abs())
        .collect();
    let mean_abs = deltas.iter().sum::<f64>() / deltas.len() as f64;
    deltas.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pct = |p: f64| deltas[((deltas.len() as f64 * p) as usize).min(deltas.len() - 1)];
    let max_abs = *deltas.last().unwrap();
    println!(
        "boundary logits: mean |d| {mean_abs:.5}  p99 {:.4}  p99.9 {:.4}  max {max_abs:.4}  (row rms {rms:.3})",
        pct(0.99),
        pct(0.999)
    );
    let mut top: Vec<(usize, f32)> = logits_s.iter().cloned().enumerate().collect();
    top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    for (id, ls) in top.iter().take(5) {
        println!("  top5 id {id}: serial {ls:.4} bulk {:.4}", logits_b[*id]);
    }
    println!("greedy serial: {ids_s:?}\ngreedy bulk:   {ids_b:?}");

    assert_eq!(
        argmax(&logits_s),
        argmax(&logits_b),
        "top-1 flipped at the prompt boundary"
    );
    // Measured class-change drift: mean |d| 0.18 / rms 2.84 =
    // 6.5%, smooth across the row (p99 0.64, max 1.17), top-5 order and
    // margins intact - the W8A16 -> W8A8-dynamic activation quantization on
    // 46 projections, not chunking structure. A carry bug (conv window, scan
    // state, KV positions) lands O(rms) and kills the greedy match below.
    assert!(
        mean_abs / rms.max(1e-3) < 0.10,
        "boundary logits drifted structurally: mean |delta| {mean_abs:.4} vs rms {rms:.3}"
    );
    assert_eq!(ids_s, ids_b, "greedy continuation diverged");

    // ---- short-prompt path (single chunk, T near the conv window) ---------
    let short: Vec<u32> = prompt[..9].to_vec();
    model.reset();
    let mut ls = Vec::new();
    for &t in &short {
        ls = model.forward(t).expect("serial short");
    }
    model.reset();
    let lb = model.forward_prefill_stream(&short).expect("bulk short");
    assert_eq!(
        argmax(&ls),
        argmax(&lb),
        "top-1 flipped on the short prompt"
    );
    println!("short prompt (9 tokens): top-1 agrees ({})", argmax(&ls));
}
