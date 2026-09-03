//! Qwen3.6-27B (MTP) acceptance gate: same-weights greedy parity vs a prebuilt
//! llama.cpp binary on the identical Q8_0 GGUF - the 27B analog of the 9B gate,
//! and the base-correctness bar the MTP speculative decoder builds on.
//!
//! Sequential by necessity: the 27B weighs ~27 GB, so llama-server and Paddock
//! cannot both be resident on the 48 GB card. Flow: launch llama-server ->
//! tokenize + greedy completion + detokenize -> kill the server -> load Paddock ->
//! generate on the identical ids -> compare token streams.
//!
//! Very heavy (two ~27 GB loads): gated on PADDOCK_HEAVY_TESTS, --test-threads=1.

use std::io::Read;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::time::{Duration, Instant};

use paddock_engine::gpu::GpuExecutor;
use paddock_engine::gpu_model::qwen35::GpuQwen35;
use paddock_models::mapped::MappedGguf;
use serde_json::{Value, json};

const PORT: u16 = 8139;
const N: usize = 24;

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap()
}

fn model_path() -> PathBuf {
    std::env::var("QWEN36_27B_GGUF").map(PathBuf::from).unwrap_or_else(|_| {
        PathBuf::from(std::env::var("USERPROFILE").expect("USERPROFILE")).join(
            ".cache/huggingface/hub/models--unsloth--Qwen3.6-27B-MTP-GGUF/snapshots/5cb35eb3dcbf52dbce5f87dbc64df6aaffadcace/Qwen3.6-27B-Q8_0.gguf",
        )
    })
}

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

/// Long enough that every DeltaNet layer prefills through the chunked scan
/// (dispatched at 128+ tokens); the short prompt stays on the exact v2
/// recurrence and never exercises it. Mirrors the 9B gate's long prompt.
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
fn qwen36_27b_greedy_matches_llamacpp() {
    greedy_gate("The capital of France is", 0, false);
}

/// The chunked-scan DeltaNet gate at 27B scale: chunked must reproduce the v2
/// recurrence's greedy stream exactly; the llama comparison is informational
/// (cross-engine exact is not stable at this prompt length - see the 9B gate's
/// doc comment for the measurement).
#[test]
fn qwen36_27b_long_prompt_chunked_matches_v2() {
    greedy_gate(LONG_PROMPT, 300, true);
}

fn greedy_gate(prompt: &str, min_tokens: usize, internal_ab: bool) {
    if std::env::var_os("PADDOCK_HEAVY_TESTS").is_none() {
        eprintln!("set PADDOCK_HEAVY_TESTS=1 to run the 27B gate (two ~27 GB loads)");
        return;
    }
    let server = repo().join("vendor/llamacpp/llama-server.exe");
    let pack = repo().join("packs/cuda/build/pd-cuda-sm86.dll");
    let model = model_path();
    for (what, p) in [
        ("llama-server", &server),
        ("pack", &pack),
        ("27B model", &model),
    ] {
        if !p.exists() {
            eprintln!("{what} {p:?} missing - skipping");
            return;
        }
    }

    // ---- phase 1: llama.cpp alone on the GPU ----
    let (ids, llama_ids, llama_text) = {
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
            if t0.elapsed() > Duration::from_secs(600) {
                panic!("llama-server never became healthy");
            }
            std::thread::sleep(Duration::from_millis(500));
        }
        eprintln!("llama-server ready in {:?}", t0.elapsed());

        let tok = post(
            &format!("{base}/tokenize"),
            &json!({"content": prompt, "add_special": true}),
        )
        .expect("tokenize");
        let ids = ids_of(&tok["tokens"]);
        assert!(!ids.is_empty(), "empty tokenization: {tok}");
        eprintln!("prompt -> {} tokens", ids.len());
        assert!(
            ids.len() >= min_tokens,
            "prompt tokenized to {} tokens, gate needs >= {min_tokens}",
            ids.len()
        );

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
            "llama returned {} tokens, want {N}",
            llama_ids.len()
        );
        let text = post(
            &format!("{base}/detokenize"),
            &json!({ "tokens": llama_ids }),
        )
        .ok()
        .and_then(|r| r["content"].as_str().map(str::to_owned))
        .unwrap_or_else(|| "<detok failed>".into());
        (ids, llama_ids, text)
        // _guard drops here -> server killed -> VRAM freed for Paddock
    };
    // give the driver a moment to release the server's allocations
    std::thread::sleep(Duration::from_secs(3));

    // ---- phase 2: Paddock alone on the GPU, identical ids ----
    let exec = Arc::new(GpuExecutor::new(0, &pack).expect("cuda executor"));
    let map = MappedGguf::open(&model).expect("open gguf");
    let t0 = Instant::now();
    let mut pad = GpuQwen35::load(exec, &map, 4096).expect("load qwen36-27b");
    eprintln!(
        "paddock loaded 27B in {:?}: {}",
        t0.elapsed(),
        pad.geometry()
    );
    let pad_ids = pad
        .generate_greedy(&ids, N, None)
        .expect("paddock generate");

    eprintln!("llama  : {llama_ids:?}\n       = {llama_text:?}");
    eprintln!("paddock: {pad_ids:?}");

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
            llama_ids, pad_ids,
            "27B greedy token stream must match b9895 exactly"
        );
    }
}
