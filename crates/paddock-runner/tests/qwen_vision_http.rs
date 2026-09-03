//! End-to-end HTTP vision test: /v1/chat/completions with an image data: URI
//! on Qwen3.6-27B + mmproj. Also proves the scheduler survives the exclusive
//! multimodal round trip: vision -> text-only -> vision again on one engine.
//! Very heavy (~28 GB residency); gated on PADDOCK_HEAVY_TESTS, model + mmproj
//! in the HF cache, pack, GPU. Run --release --test-threads=1.
#![allow(clippy::unwrap_used)]

use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use base64::Engine as _;
use http_body_util::BodyExt;
use paddock_runner::routes::{AppState, router};
use paddock_runner::serving;
use tower::ServiceExt;

fn model_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(std::env::var("USERPROFILE").expect("USERPROFILE")).join(
        ".cache/huggingface/hub/models--unsloth--Qwen3.6-27B-MTP-GGUF/snapshots/5cb35eb3dcbf52dbce5f87dbc64df6aaffadcace",
    )
}

fn app() -> Option<axum::Router> {
    if std::env::var_os("PADDOCK_HEAVY_TESTS").is_none() {
        eprintln!("set PADDOCK_HEAVY_TESTS=1 to run the vision http test");
        return None;
    }
    let dir = model_dir();
    let model_path = dir.join("Qwen3.6-27B-Q8_0.gguf");
    let mmproj = dir.join("mmproj-F16.gguf");
    let pack = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../packs/cuda/build/pd-cuda-sm86.dll");
    if !model_path.exists() || !mmproj.exists() || !pack.exists() {
        eprintln!("model/mmproj/pack missing - skipping");
        return None;
    }
    let model = serving::load(
        "qwen36-27b".into(),
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
    .map_err(|e| eprintln!("load: {e}"))
    .ok()?;
    Some(router(Arc::new(AppState::for_tests(Some(model)))))
}

/// A 256x160 BMP: solid red left half, solid blue right half.
fn red_blue_bmp() -> Vec<u8> {
    let (w, h) = (256usize, 160usize);
    let img_size = w * h * 3;
    let mut out = Vec::with_capacity(54 + img_size);
    out.extend_from_slice(b"BM");
    out.extend_from_slice(&(54u32 + img_size as u32).to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(&54u32.to_le_bytes());
    out.extend_from_slice(&40u32.to_le_bytes());
    out.extend_from_slice(&(w as i32).to_le_bytes());
    out.extend_from_slice(&(h as i32).to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&24u16.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(&(img_size as u32).to_le_bytes());
    out.extend_from_slice(&2835u32.to_le_bytes());
    out.extend_from_slice(&2835u32.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    for _y in 0..h {
        for x in 0..w {
            // BGR: red left, blue right
            if x < w / 2 {
                out.extend_from_slice(&[0, 0, 255]);
            } else {
                out.extend_from_slice(&[255, 0, 0]);
            }
        }
    }
    out
}

async fn post_raw(app: axum::Router, body: serde_json::Value) -> (StatusCode, serde_json::Value) {
    let res = app
        .oneshot(
            Request::post("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = res.status();
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let json = serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null);
    (status, json)
}

fn vision_request(uri: &str) -> serde_json::Value {
    serde_json::json!({
        "model": "qwen36-27b",
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "What two colors is this image? Answer briefly."},
            {"type": "image_url", "image_url": {"url": uri}}
        ]}],
        // Qwen3.6-27B thinks spontaneously - leave room to close the think
        // block and answer (reasoning_content absorbs the preamble)
        "max_tokens": 500, "temperature": 0.0
    })
}

#[tokio::test]
async fn vision_then_text_then_vision_on_one_engine() {
    let Some(app) = app() else { return };
    let uri = format!(
        "data:image/bmp;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(red_blue_bmp())
    );

    // 1) vision request through the exclusive serial path
    let (status, json) = post_raw(app.clone(), vision_request(&uri)).await;
    assert_eq!(status, StatusCode::OK, "body: {json}");
    let content = json["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("")
        .to_lowercase();
    eprintln!("vision 1: {content:?}");
    assert!(content.contains("red"), "missing red: {content:?}");
    assert!(content.contains("blue"), "missing blue: {content:?}");

    // 2) a plain text request must revive the batched path untouched. This
    //    model's template THINKS by DEFAULT (inverse of the 9B) - disable via
    //    the kwarg so 30 tokens suffice, which also exercises enable_thinking
    //    on an inverse-default template.
    let (status, json) = post_raw(
        app.clone(),
        serde_json::json!({
            "model": "qwen36-27b",
            "messages": [{"role":"user","content":"What is the capital of France? One word."}],
            "max_tokens": 30, "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": false}
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {json}");
    let content = json["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("");
    eprintln!("text    : {content:?}");
    assert!(
        content.contains("Paris"),
        "text path broken after mm: {content:?}"
    );

    // 3) vision again (fresh exclusive round on a used engine)
    let (status, json) = post_raw(app.clone(), vision_request(&uri)).await;
    assert_eq!(status, StatusCode::OK, "body: {json}");
    let content = json["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("")
        .to_lowercase();
    eprintln!("vision 2: {content:?}");
    assert!(
        content.contains("red") && content.contains("blue"),
        "{content:?}"
    );

    // 4) two images in one request: honest 400 (S2 limit)
    let mut two = vision_request(&uri);
    two["messages"][0]["content"]
        .as_array_mut()
        .unwrap()
        .push(serde_json::json!({"type": "image_url", "image_url": {"url": uri}}));
    let (status, json) = post_raw(app, two).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "body: {json}");
    assert_eq!(json["error"]["type"], "invalid_request_error");
}

/// S8: text and vision decode CONCURRENTLY - the image rides a batch slot
/// instead of draining the batch, and a re-sent image is served from the
/// embedding cache.
#[tokio::test]
async fn concurrent_text_and_vision() {
    let Some(app) = app() else { return };
    let uri = format!(
        "data:image/bmp;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(red_blue_bmp())
    );

    let vision = post_raw(app.clone(), vision_request(&uri));
    let text = post_raw(
        app.clone(),
        serde_json::json!({
            "model": "qwen36-27b",
            "messages": [{"role":"user","content":"What is the capital of France? One word."}],
            "max_tokens": 30, "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": false}
        }),
    );
    let ((vs, vj), (ts, tj)) = tokio::join!(vision, text);
    assert_eq!(vs, StatusCode::OK, "vision: {vj}");
    assert_eq!(ts, StatusCode::OK, "text: {tj}");
    let vc = vj["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("")
        .to_lowercase();
    let tc = tj["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("");
    eprintln!("concurrent vision: {vc:?} | text: {tc:?}");
    assert!(vc.contains("red") && vc.contains("blue"), "{vc:?}");
    assert!(tc.contains("Paris"), "{tc:?}");

    // the same image re-sent: embedding-cache-served, same answer
    let (s2, j2) = post_raw(app, vision_request(&uri)).await;
    assert_eq!(s2, StatusCode::OK, "body: {j2}");
    let c2 = j2["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("")
        .to_lowercase();
    eprintln!("resent image: {c2:?}");
    assert!(c2.contains("red") && c2.contains("blue"), "{c2:?}");
}
