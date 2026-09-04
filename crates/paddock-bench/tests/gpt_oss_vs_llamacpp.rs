//! gpt-oss acceptance gate: same-weights greedy parity vs a prebuilt
//! llama.cpp binary on the identical MXFP4 GGUF - the oracle for the
//! dp4a B=1 fast path (whose numeric class is llama.cpp's own mmvq q8_1/dp4a,
//! so exact greedy match on a short clear-margin prompt is the bar, per the
//! qwen35 gate methodology). Also prints the fast-vs-exact internal comparison
//! (different classes - informational, marginal flips are legitimate there).
//!
//! Heavy + CUDA-gated: launches llama-server (~11 GB) then Paddock (~11 GB).
//! Run with `--test-threads=1`.
// Test code: a failed assumption stops the test where it happened.
#![allow(clippy::unwrap_used)]

use std::io::Read;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::time::{Duration, Instant};

use paddock_engine::gpu::GpuExecutor;
use paddock_engine::gpu_model::gpt_oss::GpuGptOss;
use paddock_models::mapped::MappedGguf;
use serde_json::{Value, json};

const PORT: u16 = 8138;
const N: usize = 24;

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap()
}
fn model_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("GPTOSS_GGUF") {
        return Some(PathBuf::from(p));
    }
    std::env::var_os("USERPROFILE")
        .map(|h| PathBuf::from(h).join("paddock/models/gpt-oss-20b-mxfp4.gguf"))
}

/// Kills the llama-server child on scope exit.
struct ServerGuard(Child);
impl Drop for ServerGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn post(url: &str, body: &Value) -> Result<Value, String> {
    let resp = ureq::post(url)
        .header("content-type", "application/json")
        .send(&body.to_string())
        .map_err(|e| format!("POST {url}: {e}"))?;
    let mut s = String::new();
    resp.into_body()
        .into_reader()
        .read_to_string(&mut s)
        .map_err(|e| e.to_string())?;
    serde_json::from_str(&s).map_err(|e| format!("bad json from {url}: {e}; body={s}"))
}

fn health_ok(url: &str) -> bool {
    ureq::get(url)
        .call()
        .map(|r| r.status().as_u16() == 200)
        .unwrap_or(false)
}

fn ids_of(v: &Value) -> Vec<u32> {
    v.as_array()
        .map(|a| {
            a.iter()
                .filter_map(|x| x.as_u64().map(|u| u as u32))
                .collect()
        })
        .unwrap_or_default()
}

#[test]
fn gpt_oss_greedy_matches_llamacpp() {
    if std::env::var_os("PADDOCK_HEAVY_TESTS").is_none() {
        eprintln!("set PADDOCK_HEAVY_TESTS=1 to run the gpt-oss vs llama gate");
        return;
    }
    let server = repo().join("vendor/llamacpp/llama-server.exe");
    let pack = repo().join("packs/cuda/build/pd-cuda-sm86.dll");
    let Some(model) = model_path() else {
        eprintln!("no USERPROFILE - skipping");
        return;
    };
    for (what, p) in [
        ("llama-server", &server),
        ("pack", &pack),
        ("model", &model),
    ] {
        if !p.exists() {
            eprintln!("{what} {p:?} missing - skipping");
            return;
        }
    }
    let exec = match GpuExecutor::new(0, &pack) {
        Ok(e) => Arc::new(e),
        Err(e) => {
            eprintln!("no CUDA ({e}) - skipping");
            return;
        }
    };
    // this oracle pins Paddock's int8 (dp4a/mmq) classes against llama.cpp's
    // own int8 mmvq/mmq - token-for-token. The sm_120a block-scale prefill is
    // a different (fp8-activation) numeric class and is gated by the internal
    // parity suite instead (greedy + rel tolerance vs the int8 single-stream).
    paddock_engine::gpu_model::gpt_oss::set_moe_bs(false);

    // ---- launch the llama.cpp server on the same GGUF ----
    let child = Command::new(&server)
        .args([
            "--model",
            model.to_str().unwrap(),
            "-ngl",
            "99",
            "--host",
            "127.0.0.1",
            "--port",
            &PORT.to_string(),
            "-c",
            "4096",
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn llama-server");
    let _guard = ServerGuard(child);

    let base = format!("http://127.0.0.1:{PORT}");
    let t0 = Instant::now();
    while !health_ok(&format!("{base}/health")) {
        if t0.elapsed() > Duration::from_secs(180) {
            panic!("llama-server never became healthy");
        }
        std::thread::sleep(Duration::from_millis(500));
    }
    eprintln!("llama-server ready in {:?}", t0.elapsed());

    // ---- single source of input ids: llama.cpp's own tokenization ----
    // NOTE: "The capital of France is" goes marginal immediately after " Paris."
    // on this chat-tuned model (even the exact-f32 path flips vs llama at token
    // 15); story openers hold clear margins much longer.
    let prompt =
        std::env::var("PADDOCK_GATE_PROMPT").unwrap_or_else(|_| "Once upon a time".to_owned());
    let prompt = prompt.as_str();
    let tok = post(
        &format!("{base}/tokenize"),
        &json!({"content": prompt, "add_special": true}),
    )
    .expect("tokenize");
    let ids = ids_of(&tok["tokens"]);
    assert!(!ids.is_empty(), "empty tokenization: {tok}");
    eprintln!("prompt -> {} tokens", ids.len());

    // ---- llama.cpp greedy continuation (exact N tokens, ignore EOS) ----
    let comp = post(
        &format!("{base}/completion"),
        &json!({
            "prompt": ids,
            "n_predict": N,
            "temperature": 0.0,
            "top_k": 1,
            "seed": 0,
            "ignore_eos": true,
            "cache_prompt": false,
            "return_tokens": true,
        }),
    )
    .expect("completion");
    let llama_ids = ids_of(&comp["tokens"]);
    assert_eq!(
        llama_ids.len(),
        N,
        "llama returned {} tokens, want {N}: keys {:?}",
        llama_ids.len(),
        comp.as_object().map(|o| o.keys().collect::<Vec<_>>())
    );

    // ---- Paddock greedy continuation on the identical ids ----
    let map = MappedGguf::open(&model).expect("open gguf");
    let mut pad = GpuGptOss::load(exec, &map, 4096).expect("load gpt-oss");
    let pad_ids = pad.generate_greedy(&ids, N).expect("paddock generate");
    // (no second "exact f32" leg - llama.cpp itself is the reference class here)

    // ---- report + compare ----
    let detok = |v: &[u32]| {
        post(&format!("{base}/detokenize"), &json!({ "tokens": v }))
            .ok()
            .and_then(|r| r["content"].as_str().map(str::to_owned))
            .unwrap_or_else(|| "<detok failed>".into())
    };
    eprintln!(
        "llama       : {llama_ids:?}\n            = {:?}",
        detok(&llama_ids)
    );
    eprintln!(
        "paddock fast: {pad_ids:?}\n            = {:?}",
        detok(&pad_ids)
    );

    let first_div = (0..N).find(|&i| llama_ids.get(i) != pad_ids.get(i));
    match first_div {
        None => eprintln!("EXACT MATCH over {N} tokens"),
        Some(i) => eprintln!(
            "FIRST DIVERGENCE at token {i}: llama={:?} paddock={:?}",
            llama_ids.get(i),
            pad_ids.get(i)
        ),
    }
    assert_eq!(
        pad_ids, llama_ids,
        "greedy token streams diverge (see first-divergence above)"
    );

    // ---- G1 prefill-path gate: the same-weights bar for the int8 batch stack
    // (mma dense + sorted mmq MoE + f16 WMMA attention). A ~46-token COUNTING
    // prompt: > 24 rows engages the f16 prefill attention, > 32 the mma dense
    // tile, and pattern completion has greedy margins orders of magnitude
    // above class-level noise. Free-form story prompts are knife-edged at
    // this length: G2's router kernel (1.7e-7 vs cuBLAS, pure f32 sum-order
    // noise) deterministically flipped the old clockmaker prompt at token 9
    // (1232 vs 290 - the same binary near-tie G1's atomic scatter flipped)
    // and a second story prompt at token 3. One ulp of router noise can flip
    // a top-4-of-32 expert pick among ~900 routing decisions, so cross-engine
    // exact on open-ended prose ≥ ~35 tokens re-rolls on every numeric-class
    // change; constrained continuations don't.
    let prompt2 = std::env::var("PADDOCK_GATE_PREFILL_PROMPT").unwrap_or_else(|_| {
        "Let me count slowly from one to fifty without skipping any number: \
         one, two, three, four, five, six, seven, eight, nine, ten, eleven, \
         twelve, thirteen, fourteen, fifteen, sixteen, seventeen"
            .to_owned()
    });
    let tok2 = post(
        &format!("{base}/tokenize"),
        &json!({"content": prompt2.as_str(), "add_special": true}),
    )
    .expect("tokenize prefill prompt");
    let ids2 = ids_of(&tok2["tokens"]);
    eprintln!("prefill prompt -> {} tokens", ids2.len());
    assert!(
        ids2.len() > 32,
        "prefill gate prompt too short to engage the batch paths"
    );
    let comp2 = post(
        &format!("{base}/completion"),
        &json!({
            "prompt": ids2,
            "n_predict": N,
            "temperature": 0.0,
            "top_k": 1,
            "seed": 0,
            "ignore_eos": true,
            "cache_prompt": false,
            "return_tokens": true,
        }),
    )
    .expect("completion prefill prompt");
    let llama2 = ids_of(&comp2["tokens"]);
    assert_eq!(llama2.len(), N);

    let argmax = |v: &[f32]| -> u32 {
        let mut bi = 0usize;
        let mut bv = f32::NEG_INFINITY;
        for (i, &x) in v.iter().enumerate() {
            if x > bv {
                bv = x;
                bi = i;
            }
        }
        bi as u32
    };
    pad.reset();
    pad.enable_batch(1).expect("enable_batch"); // fresh KV/state for slot 0
    let vocab = pad.vocab;
    let mut logits = pad.forward_prefill(0, &ids2).expect("prefill");
    let mut pad2 = Vec::with_capacity(N);
    let mut next = argmax(&logits[..vocab]);
    pad2.push(next);
    for i in 1..N {
        logits = pad
            .forward_batch(&[next], &[(ids2.len() + i - 1) as u32])
            .expect("decode after prefill");
        next = argmax(&logits[..vocab]);
        pad2.push(next);
    }
    eprintln!(
        "llama   (prefill case): {llama2:?}\n            = {:?}",
        detok(&llama2)
    );
    eprintln!(
        "paddock (prefill case): {pad2:?}\n            = {:?}",
        detok(&pad2)
    );
    match (0..N).find(|&i| llama2.get(i) != pad2.get(i)) {
        None => eprintln!("PREFILL EXACT MATCH over {N} tokens"),
        Some(i) => eprintln!(
            "PREFILL FIRST DIVERGENCE at token {i}: llama={:?} paddock={:?}",
            llama2.get(i),
            pad2.get(i)
        ),
    }
    assert_eq!(pad2, llama2, "prefill-path greedy diverges from llama");
}
