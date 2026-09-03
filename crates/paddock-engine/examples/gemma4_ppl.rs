//! Teacher-forced perplexity for gemma4.
//!
//! STATUS: USABLE as A RELATIVE GATE on the stream lane. Do not read the
//! absolute number as a model-quality score.
//!
//! Two things make it trustworthy as a relative instrument, even though the
//! absolute number looks implausible on raw wikitext:
//!
//! 1. ABSOLUTE PPL is not OURS to EXPLAIN. llama.cpp's `llama-perplexity`, on
//!    the same gguf and the same wikitext slice, scores this model far worse
//!    than we do. So a high absolute number is a property of the model plus
//!    raw un-templated wikitext, not evidence of a broken forward pass here.
//!    (`llama-perplexity` refuses corpora under 2*n_ctx tokens.)
//!
//!    Also ruled out along the way: a uniformly mis-scaled logit vector. The
//!    `TEMPS` sweep below rescores every position at nine temperatures; ppl
//!    bottoms out at the TOP of the sweep and even there is far from healthy,
//!    so the distribution is confidently wrong rather than merely flattened or
//!    sharpened. A pure scale bug would show a deep, interior minimum. It
//!    does not.
//!
//! 2. Use the STREAM LANE AND the NONDETERMINISM is MOOT. `PPL_STREAM=1` is
//!    bit-exact run to run, so a gate built on it has zero measurement noise and
//!    needs no repeat legs to average. The batch lane's nondeterminism is real
//!    and still worth chasing (it puts a numerics floor under every serve A/B),
//!    but it does not block this gate.
//!
//! The gate is also SENSITIVE, which is the property that actually matters - a
//! reproducible instrument that cannot see the change would be useless. It
//! resolved the question it was built for: `PADDOCK_NORM_WIDE_NTH=auto` moves
//! mean NLL clearly upward (worse) on four INDEPENDENT wikitext slices, and
//! costs self-top-1 too. Both arms are bit-exact, so there is no within-arm
//! noise this could be hiding in: the wide-branch thread count is a systematic
//! quality regression, not a coin-flip in rounding order. It stays OPT-IN, and
//! this is the reason.
//!
//! A width sweep isolates it, and shows the env path is clean - a fixed 256
//! reproduces the unset default exactly, so nothing else rides along. NOTE the
//! NON-MONOTONICITY: 1024 scores better than 512. A width-dependent defect
//! (undersized staging, a hardcoded warp count) would get monotonically worse,
//! and it does not - the cross-warp reduction is correctly sized as
//! `(nth+31)>>5` for any nth <= 1024. This is real float regrouping in which 256
//! simply happens to be the most accurate arrangement for these shapes, so do
//! not go looking for a bug here.
//!
//! !!! GATE LIMITATION - READ before TRUSTING this FOR ATTENTION work !!!
//! PADDOCK_KV_FP8=1 produces a BIT-IDENTICAL score to kv16 on the stream lane
//! (9.96530 both). A quantized KV cache cannot be bit-identical to f16, so the
//! flag is not reaching this path - `set_kv_dtype` presumably does not affect
//! `forward_prefill_stream`/`forward`. Consequences: (a) this gate currently
//! measures the kv16 class, not the kv8 class we serve in, and (b) it cannot
//! be used to size any attention/KV numerics change until that
//! is fixed. It remains valid for norm/quant/GEMM work, which is what it was
//! built for.
//!
//! Original intent (still the reason to finish it):
//! Mirrors `qwen35_ppl.rs` on `GpuGemma4`. The wide-branch norm/quant
//! thread-count change (`pd_norm_wide_nth`, `PADDOCK_NORM_WIDE_NTH=auto`) is
//! a throughput win at matched acceptance but
//! regroups a float reduction, which `pd_norm_decode_nth`'s comment calls the
//! sanctioned near-tie class - a DEFAULT flip needs a perplexity gate, and no
//! gemma harness existed (`qwen35_ppl.rs` is hardcoded to `GpuQwen35`).
//!
//! **PPL_PREFIX must be large (>= 64, in practice several hundred).** That is
//! the whole point here: the change only touches launches with `rows >= 64`,
//! so the batched PREFILL is the only path that reaches it - a token-by-token
//! prefill (what `gemma4_greedy.rs` does, at rows=1) would pass the gate
//! vacuously without executing a single changed launch. PPL_SLOT has the same
//! trap.
//!
//! So the measurement is: prefill ids[0..PREFIX] as one wide batch (the arm's
//! kernels run here), then teacher-force the continuation on the batch=1
//! decode path (which takes the narrow branch either way). The continuation's
//! NLL therefore measures exactly what the change perturbs - the KV cache and
//! residual stream built through the wide-branch norms.
//!
//! Run twice on the identical corpus, reference then arm:
//!   GEMMA4_GGUF=... PADDOCK_PACK=... PPL_CORPUS=wiki.test.raw PPL_PREFIX=512 \
//!     cargo run --release --example gemma4_ppl
//!   ... PADDOCK_NORM_WIDE_NTH=auto cargo run --release --example gemma4_ppl
//!
//! Prints token count, mean NLL (nats), perplexity and self-top-1.

use std::io::Write;
use std::sync::Arc;

use paddock_engine::generator::Generator;
use paddock_engine::gpu::GpuExecutor;
use paddock_engine::gpu_model::gemma4::GpuGemma4;
use paddock_models::mapped::MappedGguf;
use paddock_tokenizer::GgufTokenizer;

/// log-softmax value for `tid` plus the argmax index, in f64 for a stable
/// logsumexp (gemma's vocab is ~256k wide, so a naive f32 sum loses bits).
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

/// NLL of `tid` under logits divided by `t`. Used to test whether the logits
/// are uniformly MIS-SCALED: a constant factor leaves the argmax ranking intact
/// (so self_top1 looks sane) while wrecking the probabilities (so NLL blows up).
/// If PPL bottoms out hard at some t != 1, the forward pass is fine and the
/// final logit scale is wrong - a very different bug from a broken model.
fn nll_at_temp(logits: &[f32], tid: u32, t: f32) -> f64 {
    let mut max = f32::NEG_INFINITY;
    for &v in logits {
        let v = v / t;
        if v > max {
            max = v;
        }
    }
    let mut sum = 0.0f64;
    for &v in logits {
        sum += ((v / t - max) as f64).exp();
    }
    ((max as f64) + sum.ln()) - ((logits[tid as usize] / t) as f64)
}

const TEMPS: [f32; 9] = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0, 2.0];

fn main() {
    let model = std::env::var("GEMMA4_GGUF").expect("set GEMMA4_GGUF");
    let pack = std::env::var("PADDOCK_PACK").expect("set PADDOCK_PACK");
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
        .unwrap_or(1024);
    let max_ctx: usize = std::env::var("MAX_CTX")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8192);
    let prefix: usize = std::env::var("PPL_PREFIX")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(512)
        .max(1);

    let map = MappedGguf::open(model.as_ref()).expect("open gguf");
    let tok = GgufTokenizer::from_gguf(map.gguf()).expect("tokenizer");
    let mut ids = Vec::new();
    if tok.add_bos
        && let Some(b) = tok.bos_id
    {
        ids.push(b);
    }
    ids.extend(tok.encode(&corpus).expect("encode"));
    ids.truncate(max_tok);
    assert!(ids.len() > prefix + 1, "corpus shorter than PPL_PREFIX+1");
    if prefix < 64 {
        eprintln!(
            "WARNING: PPL_PREFIX={prefix} < 64 - the wide-branch launches this gate \
             exists to test never fire below 64 rows; the run would pass vacuously."
        );
    }

    let exec = Arc::new(GpuExecutor::new(0, pack.as_ref()).expect("executor"));
    let mut m = GpuGemma4::load(exec, &map, max_ctx).expect("load gemma4");
    // Score in the kv8 serving class so the prefill exercises the same fp8
    // attention tiles a real serve runs through.
    if std::env::var_os("PADDOCK_KV_FP8").is_some() {
        m.set_kv_dtype(paddock_engine::gpu::KvDtype::Fp8E4m3);
        eprintln!("KV dtype: fp8 e4m3 (kv8 serving class)");
    }

    let mut out = std::env::var_os("PPL_OUT").map(|p| {
        let mut f = std::fs::File::create(p).expect("PPL_OUT");
        writeln!(f, "pos,true_id,nll,argmax_id").unwrap();
        f
    });

    // PPL_STREAM=1 scores through the SINGLE-STREAM lane
    // (forward_prefill_stream + forward) that gemma4_prefix_check.rs treats as
    // the oracle, instead of the slot/batch lane. Diagnostic for the
    // nondeterminism: if stream mode is sane and deterministic while batch mode
    // is not, the defect is in the batch path, not in this harness.
    let stream_mode = std::env::var_os("PPL_STREAM").is_some();
    m.reset();
    if !stream_mode {
        m.enable_batch(1).expect("enable_batch");
    }
    // gemma's logits buffer is WIDER than the vocab (the lm_head plane is
    // padded to a tile multiple), so every consumer must slice to vocab first -
    // scoring the padding puts garbage in both the argmax and the logsumexp.
    // gemma4_prefix_check.rs does the same (`&l[..vocab]`).
    let vocab = m.vocab();
    let mut logits = if stream_mode {
        m.forward_prefill_stream(&ids[0..prefix])
            .expect("stream prefill")
    } else {
        m.forward_prefill(0, &ids[0..prefix]).expect("slot prefill")
    };
    let mut nll_sum = 0.0f64;
    let mut n = 0usize;
    let mut top1 = 0usize;
    let mut tsum = [0.0f64; TEMPS.len()];
    for i in prefix..ids.len() {
        let (lp, amax) = logprob_and_argmax(&logits[..vocab], ids[i]);
        nll_sum += -lp;
        n += 1;
        if amax == ids[i] {
            top1 += 1;
        }
        for (j, &t) in TEMPS.iter().enumerate() {
            tsum[j] += nll_at_temp(&logits[..vocab], ids[i], t);
        }
        if let Some(f) = out.as_mut() {
            writeln!(f, "{},{},{:.6},{}", i, ids[i], -lp, amax).unwrap();
        }
        if i + 1 < ids.len() {
            logits = if stream_mode {
                m.forward(ids[i]).expect("forward")
            } else {
                m.forward_batch(&[ids[i]], &[i as u32])
                    .expect("forward_batch")
            };
        }
    }
    let mean_nll = nll_sum / n as f64;
    let lane = if stream_mode { "stream" } else { "batch" };
    let label = if std::env::var_os("PADDOCK_NORM_WIDE_NTH").is_some() {
        "widenth"
    } else {
        "ref"
    };
    println!("--- logit-scale sweep (is the forward pass fine but the scale wrong?) ---");
    let mut best = (f64::INFINITY, 1.0f32);
    for (j, &t) in TEMPS.iter().enumerate() {
        let mn = tsum[j] / n as f64;
        println!("  T={t:<5} mean_nll={mn:8.5} ppl={:12.3}", mn.exp());
        if mn < best.0 {
            best = (mn, t);
        }
    }
    println!(
        "  best T={} ppl={:.3}  (T=1 ppl={:.3})",
        best.1,
        best.0.exp(),
        (tsum[7] / n as f64).exp()
    );
    println!(
        "PPL[{label}/{lane}] prefix={prefix} tokens={n} mean_nll={mean_nll:.5} ppl={:.5} self_top1={:.4}",
        mean_nll.exp(),
        top1 as f64 / n as f64
    );
}
