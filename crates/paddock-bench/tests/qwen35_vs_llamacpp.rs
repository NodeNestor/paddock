//! Qwen3.5 acceptance gate: same-weights greedy parity vs a prebuilt
//! llama.cpp binary. Both engines consume the identical Q8_0 GGUF, so an exact
//! greedy-token match is the bar (no bf16/quant noise to explain away). We feed
//! both the same prompt token-ids (llama.cpp's own /tokenize output) so the
//! tokenizer is not a variable - any divergence is purely Paddock's forward math.
//!
//! Heavy + CUDA-gated: launches llama-server (loads ~9 GB) alongside Paddock's
//! own load (~9 GB). Run with `--test-threads=1`.
// Test code: a failed assumption stops the test where it happened.
#![allow(clippy::unwrap_used)]

use std::io::Read;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::time::{Duration, Instant};

use paddock_engine::gpu::GpuExecutor;
use paddock_engine::gpu_model::qwen35::GpuQwen35;
use paddock_models::mapped::MappedGguf;
use serde_json::{Value, json};

const PORT: u16 = 8137;
const N: usize = 24;

fn repo() -> PathBuf {
    // crates/paddock-bench -> repo root
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap()
}
fn model_path() -> PathBuf {
    std::env::var("QWEN35_GGUF")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("C:/dev/models/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q8_0.gguf"))
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

/// A prompt long enough to push every DeltaNet layer through the chunked-scan
/// prefill (dispatched at 128+ tokens; ~450 tokens here = full + partial
/// chunks) - the short-prompt gate never leaves the exact v2 recurrence.
const LONG_PROMPT: &str = "The history of numerical computation is a history of \
trading exactness for speed and then spending decades winning the exactness \
back. Early floating-point hardware differed between vendors in rounding \
behavior, exception handling, and even the width of intermediate registers, \
so the same program could produce three different answers on three machines. \
The IEEE 754 standard settled the semantics of the basic operations, but it \
did not settle the order in which a compiler or a parallel machine may combine \
them, and floating-point addition is famously not associative. A sum taken \
left to right, a sum taken in a tree, and a sum taken in whatever order a \
thousand threads happen to finish will in general disagree in their last few \
bits. For most of numerical practice this is harmless noise, absorbed by \
condition numbers far larger than the rounding terms. But there are settings \
where the bits matter: reproducible research pipelines, consensus systems \
that must agree bit-for-bit across replicas, regression gates that compare a \
new implementation against a reference, and debugging sessions where the only \
way to localize a defect is to hold everything else exactly constant. In \
those settings engineers pin the reduction order, fix the fused-multiply-add \
contraction behavior, and document every place where a reformulation changes \
the rounding profile even when it preserves the mathematics. The discipline \
resembles proof more than measurement: each transformation is either exact, \
in which case it may be applied freely, or it perturbs the result, in which \
case its perturbation must be bounded and gated. Modern accelerators raise \
the stakes because their performance comes precisely from reordering: tensor \
cores contract in blocks, warps reduce in trees, and asynchronous pipelines \
overlap operations whose sequential order the programmer once took for \
granted. The craft is knowing which reorderings are safe for the task at \
hand, and building the harness that proves it. Given all of this, the single \
most important practical rule for an engineer validating a rewritten kernel \
against a sequential reference is";

#[test]
fn qwen35_greedy_matches_llamacpp() {
    greedy_gate("The capital of France is", 0, false);
}

/// Long-prompt gate. Cross-engine greedy exact is not attainable at this
/// length - both engines run f16-class prefill attention with different
/// accumulation orders (~1e-4 apart) and deep-context logit margins fall
/// below that, so marginal tokens flip sporadically REGARDLESS of Paddock's
/// internal classes (measured: prompt lengths 64/128/382 flip after 9-20
/// exact tokens while 80/96/192/256/320 match exactly, invariant under
/// pinning attention to f32/decode and DeltaNet to v2). The stable, meaningful
/// bar here is INTERNAL: the chunked-scan DeltaNet prefill (dispatched at
/// 128+ tokens) must reproduce the exact v2 recurrence's greedy stream
/// token-for-token; the llama comparison is printed for the record.
#[test]
fn qwen35_long_prompt_chunked_matches_v2() {
    greedy_gate(LONG_PROMPT, 300, true);
}

fn greedy_gate(prompt: &str, min_tokens: usize, internal_ab: bool) {
    let server = repo().join("vendor/llamacpp/llama-server.exe");
    let pack = repo().join("packs/cuda/build/pd-cuda-sm86.dll");
    let model = model_path();
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
    // Paddock's own device - bail cleanly if there's no CUDA.
    let exec = match GpuExecutor::new(0, &pack) {
        Ok(e) => Arc::new(e),
        Err(e) => {
            eprintln!("no CUDA ({e}) - skipping");
            return;
        }
    };

    // ---- launch the pinned b9895 server on the same GGUF ----
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
    let tok = post(
        &format!("{base}/tokenize"),
        &json!({"content": prompt, "add_special": true}),
    )
    .expect("tokenize");
    let mut ids = ids_of(&tok["tokens"]);
    assert!(!ids.is_empty(), "empty tokenization: {tok}");
    // PADDOCK_GATE_PROMPT_TOKENS=n truncates the shared ids (length bisection;
    // suspends the length floor)
    let mut min_tokens = min_tokens;
    if let Ok(n) = std::env::var("PADDOCK_GATE_PROMPT_TOKENS")
        && let Ok(n) = n.parse::<usize>()
    {
        ids.truncate(n);
        min_tokens = 0;
    }
    eprintln!("prompt -> {} tokens", ids.len());
    assert!(
        ids.len() >= min_tokens,
        "prompt tokenized to {} tokens, gate needs >= {min_tokens} to exercise the intended path",
        ids.len()
    );

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
    let mut pad = GpuQwen35::load(exec, &map, 4096).expect("load qwen35");
    let pad_ids = pad
        .generate_greedy(&ids, N, None)
        .expect("paddock generate");

    // ---- report + compare ----
    let detok = |v: &[u32]| {
        post(&format!("{base}/detokenize"), &json!({ "tokens": v }))
            .ok()
            .and_then(|r| r["content"].as_str().map(str::to_owned))
            .unwrap_or_else(|| "<detok failed>".into())
    };
    eprintln!("llama  : {llama_ids:?}\n       = {:?}", detok(&llama_ids));
    eprintln!("paddock: {pad_ids:?}\n       = {:?}", detok(&pad_ids));

    let first_div = (0..N).find(|&i| llama_ids.get(i) != pad_ids.get(i));
    match first_div {
        None => eprintln!("EXACT MATCH over {N} tokens"),
        Some(i) => eprintln!(
            "FIRST DIVERGENCE at token {i}: llama={:?} paddock={:?}",
            llama_ids.get(i),
            pad_ids.get(i)
        ),
    }
    if internal_ab {
        // SAFETY: single-threaded test; the engine reads this env per prefill
        unsafe { std::env::set_var("PADDOCK_NO_CHUNKED_DN", "1") };
        let pad_v2 = pad
            .generate_greedy(&ids, N, None)
            .expect("paddock v2 generate");
        unsafe { std::env::remove_var("PADDOCK_NO_CHUNKED_DN") };
        eprintln!("paddock v2 recurrence: {pad_v2:?}");
        if pad_ids == pad_v2 {
            eprintln!("INTERNAL MATCH: chunked-scan == v2 over {N} tokens");
        }
        assert_eq!(
            pad_ids, pad_v2,
            "chunked-scan and v2 DeltaNet prefill greedy streams diverge"
        );
    } else {
        assert_eq!(
            pad_ids, llama_ids,
            "greedy token streams diverge (see first-divergence above)"
        );
    }
}
