//! Teacher-forced perplexity for nemotron - the quality gate for the f16
//! SSM-state class election. Scores a fixed corpus token-by-token:
//! prefill the first PPL_PREFIX tokens into slot 0, then feed each true token
//! through the batched decode tick and read the next-position logits,
//! accumulating the negative log-likelihood of the actual next token.
//! PPL = exp(mean NLL).
//!
//! The whole point is to run it twice on the identical corpus - once at the
//! elected f16 default (no env) and once with PADDOCK_SSM_DTYPE=f32 (the
//! checkpoint's own `mamba_ssm_cache_dtype: float32` declaration, the
//! reference class) - and compare. There is deliberately no serial-path
//! mode: `ssm_dtype` is consumed only by the batch lane's slot arenas
//! (batch.rs / spec.rs), so the single-sequence `forward_one` walk would
//! measure nothing and pass the gate vacuously.
//! Prefill AND decode both read the slot arena, so a long teacher-forced
//! continuation accumulates the f16 state error the way a long serving
//! session does.
//!
//! Usage: [NEMOTRON_DIR=...] [PADDOCK_PACK=...] PPL_CORPUS=wiki.test.raw \
//!        PADDOCK_KV_FP8=1 [PPL_MAX_TOK=4096] [PPL_PREFIX=1] \
//!        [PPL_OUT=perpos.csv] nemotron_ppl
//! Prints: token count, mean NLL (nats), perplexity. Optional CSV:
//! pos,true_id,nll,argmax_id - the cross-leg per-position comparison
//! (top-1 agreement, positions-worse) is computed from two CSVs.
//!
//! PADDOCK_KV_FP8 selects the fp8-e4m3 attention KV - the serving class every
//! board leg runs - so the measured delta is the marginal effect of the SSM
//! class on top of the real serving config, not on a config nobody serves.
// A development probe: it runs on a box its author is looking at, and a
// failure should stop it where it happened rather than be reported.
#![allow(clippy::unwrap_used)]

use std::io::Write;
use std::sync::Arc;

use paddock_engine::generator::Generator;
use paddock_engine::gpu::GpuExecutor;
use paddock_engine::gpu_model::nemotron::GpuNemotron;
use paddock_tokenizer::GgufTokenizer;

/// log-softmax value for `tid` plus the argmax index, computed in f64 for a
/// stable logsumexp (the vocab is wide enough that a naive f32 sum loses bits).
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
        std::env::var("NEMOTRON_DIR")
            .unwrap_or_else(|_| "/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4".to_string()),
    );
    let pack = std::env::var_os("PADDOCK_PACK")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../packs/cuda/build/pd-cuda-sm120.so")
        });
    let corpus = match std::env::var("PPL_CORPUS") {
        Ok(p) => std::fs::read_to_string(&p).expect("read PPL_CORPUS"),
        Err(_) => "The quick brown fox jumps over the lazy dog. \
                   In 1687 Isaac Newton published the Principia, laying out the laws \
                   of motion and universal gravitation that would anchor physics for \
                   two centuries. Photosynthesis converts carbon dioxide and water \
                   into glucose and oxygen using energy captured from sunlight."
            .to_string(),
    };
    let max_tok: usize = std::env::var("PPL_MAX_TOK")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4096);

    let exec = Arc::new(GpuExecutor::new(0, &pack).expect("executor"));
    let tok = GgufTokenizer::from_hf_dir(&dir).expect("tokenizer from HF dir");
    let mut ids = tok.encode(&corpus).expect("encode");
    ids.truncate(max_tok);
    assert!(ids.len() > 2, "corpus too short");
    let max_ctx = ids.len().max(4096);
    let mut m = GpuNemotron::load_dir(exec, &dir, max_ctx).expect("load nemotron");
    // fp8-e4m3 KV = the kv8 serving class; set before enable_batch so the
    // pool is sized for the elected element from the start.
    if std::env::var_os("PADDOCK_KV_FP8").is_some() {
        use paddock_engine::gpu::KvDtype;
        m.set_kv_dtype(KvDtype::Fp8E4m3);
        eprintln!("KV dtype: fp8 e4m3 (kv8 serving class)");
    }
    // 4 slots matches the standing batch gates; only slot 0 is used. This is
    // what builds the mamba slot arena in the elected SSM dtype.
    let slots = m.enable_batch(4).expect("enable_batch");
    assert!(slots >= 1, "batch lane gave zero slots");

    let mut out = std::env::var_os("PPL_OUT").map(|p| {
        let mut f = std::fs::File::create(p).expect("PPL_OUT");
        writeln!(f, "pos,true_id,nll,argmax_id").unwrap();
        f
    });

    // Score index PREFIX..N: prefill ids[0..PREFIX] into slot 0 (boundary
    // logits predict ids[PREFIX]), then teacher-force each true token through
    // the r=1 decode graph - the exact path the c32 cell's ticks run.
    let prefix: usize = std::env::var("PPL_PREFIX")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1)
        .max(1);
    assert!(prefix < ids.len(), "PPL_PREFIX >= corpus tokens");
    let mut logits = m.forward_prefill(0, &ids[0..prefix]).expect("slot prefill");
    let mut nll_sum = 0.0f64;
    let mut n = 0usize;
    let mut top1_hits = 0usize;
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
            logits = m
                .forward_batch(&[ids[i]], &[i as u32])
                .expect("forward_batch");
        }
    }
    let mean_nll = nll_sum / n as f64;
    let ppl = mean_nll.exp();
    // Mirrors ssm_arena::ssm_dtype_from_env - f16 is the elected default,
    // f32 the explicit reference deviation.
    let ssm = match std::env::var("PADDOCK_SSM_DTYPE").as_deref() {
        Ok("f32") | Ok("float32") | Ok("fp32") => "f32",
        _ => "f16",
    };
    let kv = if std::env::var_os("PADDOCK_KV_FP8").is_some() {
        "fp8"
    } else {
        "f16"
    };
    println!(
        "PPL[ssm-{ssm} kv-{kv}] prefix={prefix} tokens={n} mean_nll={mean_nll:.5} \
         ppl={ppl:.5} self_top1={:.4}",
        top1_hits as f64 / n as f64
    );
}
