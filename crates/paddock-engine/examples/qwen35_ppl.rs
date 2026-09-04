//! Teacher-forced perplexity for qwen35 - the quality gate for the fp4 W4A8 MoE.
//! Scores a fixed corpus token-by-token: prefill the first token, then feed each
//! true token with forward_one and read the next-position logits, accumulating the
//! negative log-likelihood of the actual next token. PPL = exp(mean NLL).
//!
//! The whole point is to run it twice on the identical corpus - once on the Q8_0
//! reference (no env) and once with the fp4 MoE forced on at decode
//! (PADDOCK_QWEN35_MOE_FP4=1 PADDOCK_QWEN35_MOE_FP4_MIN=1 PADDOCK_QMOE_SORTED_MIN=1)
//! - and compare. The step() decode path bakes its MoE dispatch into the captured
//!   graph from these env vars, so the fp4 kernel is what actually scores. Per-token
//!   fp4 numerics are batch-independent, so batch=1 scoring is faithful to the
//!   batch=32 serving decode.
//!
//! Usage: QWEN35_GGUF=... PADDOCK_PACK=... PPL_CORPUS=corpus.txt \
//!        [PPL_OUT=perpos.csv] [PPL_MAX_TOK=1024] [PPL_PREFIX=1] qwen35_ppl
//! Prints: token count, mean NLL (nats), perplexity. Optional CSV: pos,true_id,nll,argmax_id.
//!
//! PPL_PREFIX (default 1) is the fp8-per-32 PREFILL gate mode: prefill the
//! first P tokens as one batch - above `w8_min` (512) that engages the W8
//! projections (PADDOCK_QWEN35_W8=1 [+PADDOCK_F8W8_TMA=1]) exactly as serving
//! prefill does - then teacher-force the continuation on the exact-Q8 decode
//! path (batch=1 never triggers W8). The NLL of the continuation thus measures
//! precisely what W8 perturbs: the KV cache, DeltaNet states and conv windows
//! built through the fp8 projections. Run twice (env off/on), same corpus.
// A development probe: it runs on a box its author is looking at, and a
// failure should stop it where it happened rather than be reported.
#![allow(clippy::unwrap_used)]

use std::io::Write;
use std::sync::Arc;

use paddock_engine::gpu::GpuExecutor;
use paddock_engine::gpu_model::qwen35::GpuQwen35;
use paddock_models::mapped::MappedGguf;
use paddock_tokenizer::GgufTokenizer;

/// log-softmax value for `tid` plus the argmax index, computed in f64 for a stable
/// logsumexp (the vocab is ~150k wide so a naive sum loses bits).
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
    let model = std::env::var("QWEN35_GGUF")
        .unwrap_or_else(|_| "/llms/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-Q8_0.gguf".to_string());
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
        .unwrap_or(1024);

    let exec = Arc::new(GpuExecutor::new(0, &pack).expect("executor"));
    let map = MappedGguf::open(std::path::Path::new(&model)).expect("open gguf");
    let tok = GgufTokenizer::from_gguf(map.gguf()).expect("tokenizer");
    let mut ids = tok.encode(&corpus).expect("encode");
    ids.truncate(max_tok);
    assert!(ids.len() > 2, "corpus too short");
    // QWEN35_FP8_NATIVE=<snapshot dir>: the runner's --fp8-native, so the
    // fp8-native / NVFP4 lanes (checkpoint planes sourced from safetensors)
    // score the same weight classes they serve - a GGUF-only load would
    // silently gate the Q8-derived class instead.
    let fp8_native = std::env::var_os("QWEN35_FP8_NATIVE").map(std::path::PathBuf::from);
    if let Some(d) = &fp8_native {
        eprintln!("fp8-native dir: {}", d.display());
    }
    let mut m = GpuQwen35::load_with(exec, &map, 8192, fp8_native.as_deref()).expect("load qwen35");
    // PADDOCK_KV_FP8: score in the kv8 serving class (fp8-e4m3 KV) so the
    // prefill leg exercises the fp8 attention tiles (pf7/v4-PIPE) exactly as
    // the serve does - the same opt-in the batch_bench example carries.
    if std::env::var_os("PADDOCK_KV_FP8").is_some() {
        use paddock_engine::gpu::KvDtype;
        m.set_kv_dtype(KvDtype::Fp8E4m3);
        eprintln!("KV dtype: fp8 e4m3 (kv8 serving class)");
    }

    let mut out = std::env::var_os("PPL_OUT").map(|p| {
        let mut f = std::fs::File::create(p).expect("PPL_OUT");
        writeln!(f, "pos,true_id,nll,argmax_id").unwrap();
        f
    });

    // Score index PREFIX..N: prefill ids[0..PREFIX], then feed each true token.
    // PPL_SLOT=1 routes the prefill through the SERVING slot path
    // (`forward_prefill_slot` -> prefill_slot_chunk - the path where the W8/fp8
    // projections are actually wired; the example-only `prefill()` graph path
    // is not) and the continuation through `forward_batch` (batch=1, exact Q8).
    let prefix: usize = std::env::var("PPL_PREFIX")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1)
        .max(1);
    assert!(prefix < ids.len(), "PPL_PREFIX >= corpus tokens");
    let slot_mode = std::env::var_os("PPL_SLOT").is_some();
    let mut logits = if slot_mode {
        m.enable_batch(1).expect("enable_batch");
        m.forward_prefill_slot(0, &ids[0..prefix])
            .expect("slot prefill")
    } else {
        m.reset();
        m.prefill(&ids[0..prefix]).expect("prefill") // predicts ids[prefix]
    };
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
            logits = if slot_mode {
                m.forward_batch(&[ids[i]], &[i as u32])
                    .expect("forward_batch")
            } else {
                m.forward_one(ids[i]).expect("forward_one")
            };
        }
    }
    let mean_nll = nll_sum / n as f64;
    let ppl = mean_nll.exp();
    let label = if std::env::var_os("PADDOCK_QWEN35_W8").is_some() {
        "w8"
    } else if std::env::var_os("PADDOCK_QWEN35_MOE_FP4").is_some() {
        "fp4"
    } else {
        "q8"
    };
    println!(
        "PPL[{label}] prefix={prefix} tokens={n} mean_nll={mean_nll:.5} ppl={ppl:.5} \
         self_top1={:.4}",
        top1_hits as f64 / n as f64
    );
}
