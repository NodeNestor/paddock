//! End-to-end HTTP tests of /v1/chat/completions with gpt-oss-20b on GPU:
//! Harmony rendering + parsing, content, reasoning, and tool calls.
//! Heavy + gated (PADDOCK_HEAVY_TESTS=1, model, pack, GPU; run --release).
#![allow(clippy::unwrap_used)]

use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use paddock_runner::routes::{AppState, router};
use paddock_runner::serving;
use tower::ServiceExt;

fn app() -> Option<axum::Router> {
    if std::env::var_os("PADDOCK_HEAVY_TESTS").is_none() {
        eprintln!("set PADDOCK_HEAVY_TESTS=1 to run the gpt-oss chat test");
        return None;
    }
    let home = std::env::var_os("USERPROFILE")?;
    let model_path = std::path::PathBuf::from(&home).join("paddock/models/gpt-oss-20b-mxfp4.gguf");
    let pack = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../packs/cuda/build/pd-cuda-sm86.dll");
    if !model_path.exists() || !pack.exists() {
        return None;
    }
    let model = serving::load(
        "gpt-oss-20b".into(),
        &model_path,
        "cuda",
        0,
        Some(&pack),
        512,
        4,
        None,
        None,
        None,
        None,
    )
    .ok()?;
    Some(router(Arc::new(AppState::for_tests(Some(model)))))
}

async fn post(app: axum::Router, body: serde_json::Value) -> serde_json::Value {
    let res = app
        .oneshot(
            Request::post("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    serde_json::from_slice(&bytes).unwrap()
}

async fn post_to(
    app: axum::Router,
    path: &str,
    body: serde_json::Value,
) -> (StatusCode, serde_json::Value) {
    let res = app
        .oneshot(
            Request::post(path)
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = res.status();
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    (
        status,
        serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null),
    )
}

/// gpt-oss is a text-only architecture: an image on any surface that accepts
/// image inputs must be the honest capability 400 (fired before decode, so
/// even a bogus payload names the real problem), never a silent drop or an
/// engine-level failure.
#[tokio::test]
async fn image_requests_get_the_honest_vision_400() {
    let Some(app) = app() else { return };
    let uri = "data:image/bmp;base64,AAAA"; // capability check precedes decode

    // chat: image_url content part
    let (status, json) = post_to(
        app.clone(),
        "/v1/chat/completions",
        serde_json::json!({
            "model": "gpt-oss-20b",
            "messages": [{"role":"user","content":[
                {"type":"text","text":"What is this?"},
                {"type":"image_url","image_url":{"url": uri}}
            ]}],
            "max_tokens": 20
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "chat: {json}");
    let msg = json["error"]["message"].as_str().unwrap_or("");
    assert!(
        msg.contains("vision"),
        "chat message should name the gap: {msg:?}"
    );

    // responses: input_image item
    let (status, json) = post_to(
        app.clone(),
        "/v1/responses",
        serde_json::json!({
            "model": "gpt-oss-20b",
            "input": [{"role":"user","content":[
                {"type":"input_text","text":"What is this?"},
                {"type":"input_image","image_url": uri}
            ]}],
            "max_output_tokens": 16
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "responses: {json}");
    let msg = json["error"]["message"].as_str().unwrap_or("");
    assert!(
        msg.contains("vision"),
        "responses message should name the gap: {msg:?}"
    );

    // anthropic messages: base64 image source block
    let (status, json) = post_to(
        app,
        "/v1/messages",
        serde_json::json!({
            "model": "gpt-oss-20b",
            "max_tokens": 16,
            "messages": [{"role":"user","content":[
                {"type":"text","text":"What is this?"},
                {"type":"image","source":{"type":"base64","media_type":"image/bmp","data":"AAAA"}}
            ]}]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "anthropic: {json}");
    assert_eq!(json["type"], "error", "anthropic envelope: {json}");
    let msg = json["error"]["message"].as_str().unwrap_or("");
    assert!(
        msg.contains("vision"),
        "anthropic message should name the gap: {msg:?}"
    );
}

#[tokio::test]
async fn chat_answers_a_question_with_content_and_reasoning() {
    let Some(app) = app() else { return };
    let json = post(
        app,
        serde_json::json!({
            "model": "gpt-oss-20b",
            "messages": [{"role":"user","content":"What is the capital of France? One word."}],
            "max_tokens": 80, "temperature": 0.0
        }),
    )
    .await;
    assert_eq!(json["object"], "chat.completion");
    let msg = &json["choices"][0]["message"];
    let content = msg["content"].as_str().unwrap_or("");
    eprintln!(
        "content={content:?} reasoning={:?}",
        msg["reasoning_content"]
    );
    assert!(content.contains("Paris"));
    assert_eq!(json["choices"][0]["finish_reason"], "stop");
    assert!(
        msg["reasoning_content"].is_string(),
        "analysis channel captured"
    );
}

/// S6: turn 2 re-sends the rendered history, whose prefix the radix KV cache
/// serves page-granularly - visible as usage cached_tokens.
#[tokio::test]
async fn multi_turn_reuses_prefix_cache() {
    let Some(app) = app() else { return };
    let system = "You are a concise reference assistant for a European travel agency. \
                  Answer factual questions with a single word or a very short phrase.";
    let q1 = "What is the capital of France?";
    let first = post(
        app.clone(),
        serde_json::json!({
            "model": "gpt-oss-20b",
            "messages": [{"role":"system","content":system},{"role":"user","content":q1}],
            "max_tokens": 200, "temperature": 0.0
        }),
    )
    .await;
    let a1 = first["choices"][0]["message"]["content"]
        .as_str()
        .unwrap()
        .to_owned();
    assert!(a1.contains("Paris"), "{a1:?}");
    assert_eq!(
        first["usage"]["prompt_tokens_details"]["cached_tokens"].as_u64(),
        Some(0),
        "fresh engine cannot have cache hits"
    );

    let second = post(
        app,
        serde_json::json!({
            "model": "gpt-oss-20b",
            "messages": [
                {"role":"system","content":system},
                {"role":"user","content":q1},
                {"role":"assistant","content":a1},
                {"role":"user","content":"And of Italy?"}
            ],
            "max_tokens": 300, "temperature": 0.0
        }),
    )
    .await;
    let content = second["choices"][0]["message"]["content"].as_str().unwrap();
    let cached = second["usage"]["prompt_tokens_details"]["cached_tokens"]
        .as_u64()
        .unwrap();
    let prompt2 = second["usage"]["prompt_tokens"].as_u64().unwrap();
    eprintln!("multi-turn: turn2 cached {cached}/{prompt2}, answer {content:?}");
    assert!(content.contains("Rome"), "{content:?}");
    assert!(
        cached >= 32,
        "turn 2 must reuse the shared prefix (got {cached})"
    );
    assert!(cached < prompt2, "reuse cannot cover the whole prompt");
}

#[tokio::test]
async fn chat_emits_a_tool_call() {
    let Some(app) = app() else { return };
    let json = post(
        app,
        serde_json::json!({
            "model": "gpt-oss-20b",
            "messages": [{"role":"user","content":"What's the weather in Paris? Use the tool."}],
            "tools": [{"type":"function","function":{
                "name":"get_weather",
                "description":"Get current weather for a city",
                "parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}
            }}],
            "max_tokens": 200, "temperature": 0.0
        }),
    )
    .await;
    let choice = &json["choices"][0];
    assert_eq!(choice["finish_reason"], "tool_calls");
    let calls = choice["message"]["tool_calls"].as_array().unwrap();
    assert!(!calls.is_empty());
    assert_eq!(calls[0]["function"]["name"], "get_weather");
    let args: serde_json::Value =
        serde_json::from_str(calls[0]["function"]["arguments"].as_str().unwrap()).unwrap();
    eprintln!("tool call args: {args}");
    assert!(
        args["city"]
            .as_str()
            .unwrap()
            .to_lowercase()
            .contains("paris")
    );
}

#[tokio::test]
async fn chat_response_format_json_via_final_channel() {
    let Some(app) = app() else { return };
    let json = post(
        app,
        serde_json::json!({
            "model": "gpt-oss-20b",
            "messages": [{"role":"user","content":"Give a JSON object with the capital and population of France."}],
            "response_format": {"type":"json_object"},
            "max_tokens": 250, "temperature": 0.0
        }),
    )
    .await;
    let msg = &json["choices"][0]["message"];
    let content = msg["content"].as_str().unwrap_or("");
    eprintln!(
        "gpt-oss json: content={content:?} reasoning={:?}",
        msg["reasoning_content"]
    );
    // analysis stays free; the final channel is grammar-locked to JSON
    let parsed: serde_json::Value = serde_json::from_str(content).expect("valid JSON forced");
    assert!(parsed.is_object(), "expected an object: {content}");
}

#[tokio::test]
async fn chat_streams_reasoning_and_content_deltas() {
    let Some(app) = app() else { return };
    let res = app
        .oneshot(
            Request::post("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::json!({
                        "model": "gpt-oss-20b",
                        "messages": [{"role":"user","content":"What is the capital of France? One word."}],
                        "max_tokens": 120, "temperature": 0.0, "stream": true
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let text = String::from_utf8_lossy(&bytes);

    let mut reasoning = String::new();
    let mut content = String::new();
    let mut done = false;
    for line in text.lines() {
        let Some(data) = line.strip_prefix("data: ") else {
            continue;
        };
        if data.trim() == "[DONE]" {
            done = true;
            continue;
        }
        let v: serde_json::Value = serde_json::from_str(data).unwrap();
        if let Some(d) = v["choices"][0]["delta"]["reasoning_content"].as_str() {
            reasoning.push_str(d);
        }
        if let Some(d) = v["choices"][0]["delta"]["content"].as_str() {
            content.push_str(d);
        }
    }
    eprintln!("streamed reasoning={reasoning:?}\nstreamed content={content:?}");
    assert!(done, "[DONE] terminal missing");
    assert!(!reasoning.is_empty(), "analysis channel did not stream");
    assert!(
        content.contains("Paris"),
        "content deltas wrong: {content:?}"
    );
}
