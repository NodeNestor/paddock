//! SDK conformance gates. Boot the real paddock server on a loopback
//! TCP port and drive it with the official Python SDKs, which arbitrate the
//! wire format the way llama.cpp arbitrates numerics:
//! - `openai` (tests/python/openai_conformance.py): chat non-stream/stream,
//!   the tool-calling agent loop, `.parse()` structured outputs on chat AND
//!   responses, n/logprobs/usage, legacy completions, the Responses API's
//!   typed event stream + truthful incomplete status, and vision.
//! - `anthropic` (tests/python/anthropic_conformance.py): /v1/messages
//!   content blocks, thinking, tool_use round trips, the strict streaming
//!   event protocol via the SDK accumulator, stop_sequences, count_tokens,
//!   and vision - the surface Claude Code speaks.
//!   Heavy + gated (PADDOCK_HEAVY_TESTS=1, model, pack, GPU, python + SDKs;
//!   run --release --test-threads=1).
//!
//! `python` must be the PINNED SDKs, not just any interpreter: the version
//! currency check hard-fails on drift, and a shared venv carries whatever
//! else was installed into it. A dedicated one is the way in - its bin
//! dir goes on PATH so `python` resolves to it, because a host with python3
//! only and no `python` makes every gate decline:
//!
//!   python3 -m venv ~/spec-gate-venv
//!   ~/spec-gate-venv/bin/pip install openai==<pin> anthropic==<pin> \
//!       httpx pydantic websockets   # the pins live in tests/spec/coverage.json
//!   PATH=~/spec-gate-venv/bin:$PATH PADDOCK_HEAVY_TESTS=1 \
//!       cargo test --release -p paddock-runner --test sdk_conformance \
//!       -- --test-threads=1 --nocapture
//!
//! `websockets` is not optional here - the audio gates open a realtime
//! session, and `openai` only pulls it in under the `[realtime]` extra.
#![allow(clippy::unwrap_used)]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use paddock_runner::routes::{AppState, router};
use paddock_runner::serving::{self, AsrModel, ServingModel};

fn heavy() -> bool {
    if std::env::var_os("PADDOCK_HEAVY_TESTS").is_none() {
        eprintln!("set PADDOCK_HEAVY_TESTS=1 to run the SDK conformance gate");
        return false;
    }
    true
}

/// Every gate below can decline to run: no model present, no kernel pack,
/// no SDK installed. That is deliberate - the file has to compile and be
/// checkable on a machine that cannot serve. It is also exactly how a
/// conformance suite lies to you: this file once printed ten greens against a
/// model path that had gone stale, and nothing had executed; the
/// same shape hid a stale SDK for months until the pin check landed.
///
/// So a decline is loud, and under `PADDOCK_CONFORMANCE_STRICT=1` it is a
/// FAILURE. That is the mode a release build runs: green then means the whole
/// surface actually executed, not that the machine was too empty to try. Left off
/// by default so a laptop can still run the subset it can serve.
fn decline(what: &str) -> bool {
    let strict = std::env::var_os("PADDOCK_CONFORMANCE_STRICT").is_some();
    assert!(
        !strict,
        "PADDOCK_CONFORMANCE_STRICT=1 and this gate cannot run: {what}. \
         A conformance gate that skips is not a conformance gate - install \
         what is missing or drop the strict flag."
    );
    eprintln!("SKIPPING: {what}");
    false
}

/// The kernel pack every heavy gate needs. `PADDOCK_PACK` (a full path) wins,
/// same as the engine examples; otherwise the platform's own build artifact -
/// a hardcoded `.dll` made every gate below decline on Linux, and a declining
/// gate reports "ok".
fn pack_path() -> PathBuf {
    if let Some(p) = std::env::var_os("PADDOCK_PACK") {
        return PathBuf::from(p);
    }
    let build = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../packs/cuda/build");
    if cfg!(windows) {
        build.join("pd-cuda-sm86.dll")
    } else {
        build.join("pd-cuda-sm120.so")
    }
}

/// The gate needs the SDK package; decline loudly if the box lacks it.
fn python_ready(package: &str) -> bool {
    let ok = std::process::Command::new("python")
        .args(["-c", &format!("import {package}, pydantic")])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    if !ok {
        return decline(&format!(
            "python or the `{package}` package is missing (pip install {package})"
        ));
    }
    ok
}

/// Serve the model on an ephemeral loopback port; returns the port.
async fn serve(model: ServingModel) -> (u16, tokio::task::JoinHandle<()>) {
    serve_state(AppState::for_tests(Some(model))).await
}

async fn serve_state(state: AppState) -> (u16, tokio::task::JoinHandle<()>) {
    let app = router(Arc::new(state));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let handle = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    (port, handle)
}

/// Run one conformance script against the served port. The openai SDK takes
/// the /v1 base; the anthropic SDK appends /v1/messages to the server root.
async fn run_gate(
    script: &str,
    base_url: String,
    model: &str,
    dialect: &str,
    vision: bool,
    extra: &[&str],
) {
    let script = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/python")
        .join(script);
    let mut cmd = tokio::process::Command::new("python");
    cmd.arg(&script)
        .arg("--base-url")
        .arg(base_url)
        .arg("--model")
        .arg(model)
        .arg("--dialect")
        .arg(dialect)
        // model output may contain non-ASCII; keep piped stdout UTF-8
        .env("PYTHONUTF8", "1");
    if vision {
        cmd.arg("--vision");
    }
    cmd.args(extra);
    let out = cmd.output().await.unwrap();
    eprintln!("{}", String::from_utf8_lossy(&out.stdout));
    let stderr = String::from_utf8_lossy(&out.stderr);
    if !stderr.is_empty() {
        eprintln!("--- python stderr ---\n{stderr}");
    }
    assert!(out.status.success(), "SDK conformance script failed");
}

async fn run_openai(port: u16, model: &str, dialect: &str, vision: bool) {
    run_gate(
        "openai_conformance.py",
        format!("http://127.0.0.1:{port}/v1"),
        model,
        dialect,
        vision,
        &[],
    )
    .await;
}

async fn run_anthropic(port: u16, model: &str, dialect: &str, vision: bool) {
    run_gate(
        "anthropic_conformance.py",
        format!("http://127.0.0.1:{port}"),
        model,
        dialect,
        vision,
        &[],
    )
    .await;
}

/// Where the elected checkpoints live. `PADDOCK_MODELS_DIR` wins; otherwise
/// `models/` under the workspace root, which is gitignored so a symlink to
/// your own store works. The per-model env vars below still override outright.
fn models_root() -> PathBuf {
    if let Some(d) = std::env::var_os("PADDOCK_MODELS_DIR") {
        return PathBuf::from(d);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models")
}

fn load_qwen35() -> Option<ServingModel> {
    let model_path = std::env::var("QWEN35_GGUF")
        .map(PathBuf::from)
        .unwrap_or_else(|_| models_root().join("Qwen3.5-9B-MTP-GGUF/Qwen3.5-9B-Q8_0.gguf"));
    let pack = pack_path();
    if !model_path.exists() || !pack.exists() {
        decline("qwen35-9b GGUF or the kernel pack is missing");
        return None;
    }
    serving::load(
        "qwen35-9b".into(),
        &model_path,
        "cuda",
        0,
        Some(&pack),
        // the gates' own probes size this: sec_context_management
        // builds an ~8k-token prompt deliberately to trigger compaction,
        // and 4096 turned that section into a context_length_exceeded
        16384,
        8,
        None,
        None,
        None,
        None,
    )
    .map_err(|e| panic!("the model is on the box but would not load: {e}"))
    .ok()
}

fn load_gpt_oss() -> Option<ServingModel> {
    let model_path = std::env::var("GPT_OSS_GGUF")
        .map(PathBuf::from)
        .unwrap_or_else(|_| models_root().join("gpt-oss-20b-GGUF/gpt-oss-20b-mxfp4.gguf"));
    let pack = pack_path();
    if !model_path.exists() || !pack.exists() {
        decline("gpt-oss-20b GGUF or the kernel pack is missing");
        return None;
    }
    serving::load(
        "gpt-oss-20b".into(),
        &model_path,
        "cuda",
        0,
        Some(&pack),
        // the gates' own probes size this: sec_context_management
        // builds an ~8k-token prompt deliberately to trigger compaction,
        // and 4096 turned that section into a context_length_exceeded
        16384,
        8,
        None,
        None,
        None,
        None,
    )
    .map_err(|e| panic!("the model is on the box but would not load: {e}"))
    .ok()
}

fn load_qwen36_vision() -> Option<ServingModel> {
    // These paths went stale once already (an HF-cache snapshot dir that was
    // later deleted, expecting the F32-era mmproj name) and the gates SKIPPED
    // silently for it. The elected line lives under the models root, and a
    // skip like this is now a FAILURE under PADDOCK_CONFORMANCE_STRICT=1.
    let dir = std::env::var("QWEN36_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| models_root().join("Qwen3.6-27B-MTP-GGUF"));
    let model_path = dir.join("Qwen3.6-27B-Q8_0.gguf");
    let mmproj = dir.join("mmproj-BF16.gguf");
    let pack = pack_path();
    if !model_path.exists() || !mmproj.exists() || !pack.exists() {
        decline("qwen36 vision model, mmproj or kernel pack missing");
        return None;
    }
    serving::load(
        "qwen36-27b".into(),
        &model_path,
        "cuda",
        0,
        Some(&pack),
        // the gates' own probes size this: sec_context_management
        // builds an ~8k-token prompt deliberately to trigger compaction,
        // and 4096 turned that section into a context_length_exceeded
        16384,
        8,
        Some(&mmproj),
        None,
        None,
        None,
    )
    .map_err(|e| panic!("the model is on the box but would not load: {e}"))
    .ok()
}

/// granite-speech-4.1-2b-PLUS, the one generative lane that times words: ask
/// it to and it writes `[T:N]` tags into its transcript. Its base
/// sibling deliberately cannot, which is why the gate names the -plus file
/// rather than either granite-speech that happens to be on the box.
fn load_granite_speech_plus() -> Option<ServingModel> {
    let dir = std::env::var("GRANITE_SPEECH_PLUS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| models_root().join("granite-speech-4.1-2b-plus-GGUF"));
    let model_path = dir.join("granite-speech-4.1-2b-plus-Q8_0.gguf");
    let mmproj = dir.join("mmproj-model-f16.gguf");
    let pack = pack_path();
    if !model_path.exists() || !mmproj.exists() || !pack.exists() {
        decline("granite-speech-plus model, mmproj or kernel pack missing");
        return None;
    }
    serving::load(
        "granite-speech-4.1-2b-plus".into(),
        &model_path,
        "cuda",
        0,
        Some(&pack),
        4096,
        8,
        Some(&mmproj),
        None,
        None,
        None,
    )
    .map_err(|e| panic!("the model is on the box but would not load: {e}"))
    .ok()
}

fn load_whisper() -> Option<AsrModel> {
    let model_path = std::env::var("WHISPER_GGUF")
        .map(PathBuf::from)
        .unwrap_or_else(|_| models_root().join("kb-whisper-large/kb-whisper-large-f16.gguf"));
    let pack = pack_path();
    if !model_path.exists() || !pack.exists() {
        decline("whisper model or kernel pack missing");
        return None;
    }
    serving::load_asr(
        "kb-whisper-large".into(),
        &model_path,
        "cuda",
        0,
        Some(&pack),
        4096,
        8,
        None,
    )
    .map_err(|e| panic!("the model is on the box but would not load: {e}"))
    .ok()
}

#[tokio::test]
async fn openai_sdk_gate_qwen35() {
    if !heavy() || !python_ready("openai") {
        return;
    }
    let Some(model) = load_qwen35() else { return };
    let (port, server) = serve(model).await;
    run_openai(port, "qwen35-9b", "qwen", false).await;
    server.abort();
}

#[tokio::test]
async fn openai_sdk_gate_gpt_oss() {
    if !heavy() || !python_ready("openai") {
        return;
    }
    let Some(model) = load_gpt_oss() else { return };
    let (port, server) = serve(model).await;
    run_openai(port, "gpt-oss-20b", "harmony", false).await;
    server.abort();
}

#[tokio::test]
async fn openai_sdk_gate_vision_qwen36() {
    if !heavy() || !python_ready("openai") {
        return;
    }
    let Some(model) = load_qwen36_vision() else {
        return;
    };
    let (port, server) = serve(model).await;
    run_openai(port, "qwen36-27b", "qwen", true).await;
    server.abort();
}

/// S7: the spec-driven gate - request-surface audit against the pinned SDKs'
/// request TypedDicts + strict pydantic validation of responses and stream
/// events (tests/python/spec_conformance.py + tests/spec/coverage.json).
async fn run_spec(port: u16, model: &str, dialect: &str) {
    run_gate(
        "spec_conformance.py",
        format!("http://127.0.0.1:{port}"),
        model,
        dialect,
        false,
        &[],
    )
    .await;
}

/// The audio half of the same gate. It runs alone because the server it needs
/// is an ASR-only one: a whisper checkpoint has no chat, completions,
/// responses or messages surface for the other sections to audit.
async fn run_spec_audio(port: u16, model: &str) {
    run_gate(
        "spec_conformance.py",
        format!("http://127.0.0.1:{port}"),
        model,
        "qwen",
        false,
        &["--audio-only"],
    )
    .await;
}

#[tokio::test]
async fn spec_gate_qwen35() {
    if !heavy() || !python_ready("openai") || !python_ready("anthropic") {
        return;
    }
    let Some(model) = load_qwen35() else { return };
    let (port, server) = serve(model).await;
    run_spec(port, "qwen35-9b", "qwen").await;
    server.abort();
}

/// The audio surface of the spec gate: `/v1/audio/transcriptions`
/// probed against a real whisper server, with the fixture clip in
/// tests/fixtures/audio. kb-whisper is the pick because it is the checkpoint
/// the timestamp work was measured on, but any whisper GGUF serves - the gate
/// asserts wire SHAPE, never accuracy.
#[tokio::test]
async fn spec_gate_whisper() {
    if !heavy() || !python_ready("openai") || !python_ready("anthropic") {
        return;
    }
    let Some(asr) = load_whisper() else { return };
    let (port, server) = serve_state(AppState::for_tests_asr(asr)).await;
    run_spec_audio(port, "kb-whisper-large").await;
    server.abort();
}

/// The same audio surface against the other kind of lane: a
/// generative ASR model that answers `word` and refuses `segment`.
///
/// Worth its own gate rather than a note in the whisper one, because every
/// interesting thing here differs - the times come back as TEXT the runner has
/// to parse out of the model's own answer, the transcript that carries them is
/// a different decode from the plain one, and there are no segments to hang
/// anything on. The python side reads the served granularity list and asserts
/// accordingly, so this leg costs one `#[tokio::test]` and no forked
/// expectations.
#[tokio::test]
async fn spec_gate_granite_speech_plus() {
    if !heavy() || !python_ready("openai") || !python_ready("anthropic") {
        return;
    }
    let Some(model) = load_granite_speech_plus() else {
        return;
    };
    let (port, server) = serve(model).await;
    run_spec_audio(port, "granite-speech-4.1-2b-plus").await;
    server.abort();
}

#[tokio::test]
async fn spec_gate_gpt_oss() {
    if !heavy() || !python_ready("openai") || !python_ready("anthropic") {
        return;
    }
    let Some(model) = load_gpt_oss() else { return };
    let (port, server) = serve(model).await;
    run_spec(port, "gpt-oss-20b", "harmony").await;
    server.abort();
}

#[tokio::test]
async fn anthropic_sdk_gate_qwen35() {
    if !heavy() || !python_ready("anthropic") {
        return;
    }
    let Some(model) = load_qwen35() else { return };
    let (port, server) = serve(model).await;
    run_anthropic(port, "qwen35-9b", "qwen", false).await;
    server.abort();
}

#[tokio::test]
async fn anthropic_sdk_gate_gpt_oss() {
    if !heavy() || !python_ready("anthropic") {
        return;
    }
    let Some(model) = load_gpt_oss() else { return };
    let (port, server) = serve(model).await;
    run_anthropic(port, "gpt-oss-20b", "harmony", false).await;
    server.abort();
}

#[tokio::test]
async fn anthropic_sdk_gate_vision_qwen36() {
    if !heavy() || !python_ready("anthropic") {
        return;
    }
    let Some(model) = load_qwen36_vision() else {
        return;
    };
    let (port, server) = serve(model).await;
    run_anthropic(port, "qwen36-27b", "qwen", true).await;
    server.abort();
}
