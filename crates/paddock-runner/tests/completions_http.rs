//! End-to-end HTTP test of /v1/completions through the real engine service,
//! serving gpt-oss-20b on the A6000. Proves the full product path: request ->
//! tokenizer -> engine thread -> GPU forward -> sampler -> SSE/JSON -> decode.
//!
//! Heavy + gated (model + pack + GPU + PADDOCK_HEAVY_TESTS=1; run --release).
//! The tiny-llama fixture can't drive this yet - its SPM tokenizer isn't in
//! paddock-tokenizer (gpt2-BPE only); that family lands with SPM support.
#![allow(clippy::unwrap_used)] // unwrap is idiomatic in test assertions

use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use paddock_runner::routes::{AppState, router};
use paddock_runner::serving;
use tower::ServiceExt;

fn app() -> Option<axum::Router> {
    if std::env::var_os("PADDOCK_HEAVY_TESTS").is_none() {
        eprintln!("set PADDOCK_HEAVY_TESTS=1 to run the gpt-oss HTTP serving test");
        return None;
    }
    let home = std::env::var_os("USERPROFILE")?;
    let model_path = std::path::PathBuf::from(&home).join("paddock/models/gpt-oss-20b-mxfp4.gguf");
    let pack = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../packs/cuda/build/pd-cuda-sm86.dll");
    if !model_path.exists() || !pack.exists() {
        eprintln!("model or pack missing - skipping");
        return None;
    }
    let model = match serving::load(
        "gpt-oss-20b".into(),
        &model_path,
        "cuda",
        0,
        Some(&pack),
        256,
        4,
        None,
        None,
        None,
        None,
    ) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("serving load failed ({e}) - skipping");
            return None;
        }
    };
    let state = Arc::new(AppState::for_tests(Some(model)));
    Some(router(state))
}

#[tokio::test]
async fn gpt_oss_completion_over_http_matches_known_greedy() {
    let Some(app) = app() else { return };
    // greedy -> the same continuation the parity tests pin
    let body = serde_json::json!({
        "model": "gpt-oss-20b",
        "prompt": "Once upon a time",
        "max_tokens": 12,
        "temperature": 0.0
    });
    let res = app
        .oneshot(
            Request::post("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let text = json["choices"][0]["text"].as_str().unwrap();
    eprintln!("http completion: {text:?}");
    assert!(text.contains("small town"), "got {text:?}");
}

#[tokio::test]
async fn gpt_oss_streaming_over_http() {
    let Some(app) = app() else { return };
    let body = serde_json::json!({
        "model": "gpt-oss-20b",
        "prompt": "Once upon a time",
        "max_tokens": 12,
        "temperature": 0.0,
        "stream": true
    });
    let res = app
        .oneshot(
            Request::post("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let ct = res
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert!(ct.contains("text/event-stream"), "content-type {ct:?}");

    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let text = String::from_utf8_lossy(&bytes);
    let mut assembled = String::new();
    let mut saw_done = false;
    for line in text.lines() {
        let Some(data) = line.strip_prefix("data: ") else {
            continue;
        };
        if data == "[DONE]" {
            saw_done = true;
            continue;
        }
        let chunk: serde_json::Value = serde_json::from_str(data).unwrap();
        if let Some(d) = chunk["choices"][0]["text"].as_str() {
            assembled.push_str(d);
        }
    }
    eprintln!("streamed: {assembled:?}");
    assert!(saw_done, "must end with [DONE]");
    assert!(assembled.contains("small town"), "got {assembled:?}");
}
