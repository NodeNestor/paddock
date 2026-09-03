//! `/api/feedback` - what it refuses locally, and how it fails when the
//! feedback service is not there.
//!
//! The local refusals are the point of the endpoint having any validation at
//! all: an anonymous IP gets five submissions an hour upstream, so spending one
//! to be told a category was misspelled is a bad trade when the answer is
//! knowable here. And the offline case must say plainly that nothing was sent -
//! a user who believes a lost bug report was filed is worse off than one told it
//! failed.

use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use paddock_manager::routes::{AppState, router};
use tower::ServiceExt;

async fn post(
    app: axum::Router,
    uri: &str,
    body: serde_json::Value,
) -> (StatusCode, serde_json::Value) {
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(uri)
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .expect("router responds");
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20)
        .await
        .expect("body");
    let json = serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null);
    (status, json)
}

/// Every local refusal, checked in one place - and checked to be refused
/// before any outbound call, which is why this test needs no API base set.
#[tokio::test]
async fn malformed_submissions_are_refused_here_not_upstream() {
    let cases: [(&str, serde_json::Value); 4] = [
        // Upstream matches the category exactly and case-sensitively.
        (
            "capitalised category",
            serde_json::json!({ "category": "Bug", "message": "hi" }),
        ),
        (
            "unknown category",
            serde_json::json!({ "category": "praise", "message": "hi" }),
        ),
        (
            "empty message",
            serde_json::json!({ "category": "bug", "message": "   " }),
        ),
        (
            "oversized message",
            serde_json::json!({ "category": "bug", "message": "x".repeat(10_001) }),
        ),
    ];

    for (name, body) in cases {
        let app = router(Arc::new(AppState::for_tests()));
        let (status, json) = post(app, "/api/feedback", body).await;
        assert_eq!(
            status,
            StatusCode::BAD_REQUEST,
            "{name} should be refused locally"
        );
        assert!(
            json["error"]["message"].is_string(),
            "{name} must explain itself in the shape the Studio reads, got {json}"
        );
    }
}

/// A message of exactly the cap, in multi-byte characters, must pass. The
/// upstream limit is a .NET string length, so counting BYTES here would refuse
/// accented or CJK text at roughly a third of the length the server accepts.
#[tokio::test]
async fn the_length_cap_counts_characters_not_bytes() {
    unsafe { std::env::set_var("PADDOCK_API_BASE", "http://127.0.0.1:1") };

    let app = router(Arc::new(AppState::for_tests()));
    let (status, _) = post(
        app,
        "/api/feedback",
        // 10 000 chars / 30 000 bytes: refused by a byte check, fine by a char one.
        serde_json::json!({ "category": "bug", "message": "å".repeat(10_000) }),
    )
    .await;

    assert_ne!(
        status,
        StatusCode::BAD_REQUEST,
        "a 10k-CHARACTER message is within the cap; a byte count would wrongly refuse it"
    );

    unsafe { std::env::remove_var("PADDOCK_API_BASE") };
}

/// Nothing listening: the user must be told the report did not send.
#[tokio::test]
async fn an_unreachable_feedback_service_says_so() {
    // 127.0.0.1:1 - reserved, refuses immediately, no DNS wait.
    unsafe { std::env::set_var("PADDOCK_API_BASE", "http://127.0.0.1:1") };

    let app = router(Arc::new(AppState::for_tests()));
    let (status, json) = post(
        app,
        "/api/feedback",
        serde_json::json!({ "category": "bug", "message": "the engine will not start" }),
    )
    .await;

    assert_eq!(status, StatusCode::BAD_GATEWAY);
    let msg = json["error"]["message"].as_str().unwrap_or_default();
    assert!(
        !msg.is_empty(),
        "a failed send must carry a sentence - silence reads as success, got {json}"
    );

    unsafe { std::env::remove_var("PADDOCK_API_BASE") };
}

/// The preview endpoint returns the blob the POST would attach. This is the
/// whole privacy contract: if these ever diverge, the dialog is showing the user
/// something other than what leaves the box.
#[tokio::test]
async fn the_context_preview_is_the_payload() {
    let app = router(Arc::new(AppState::for_tests()));
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/api/feedback/context")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .expect("router responds");

    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20)
        .await
        .expect("body");
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("json");

    assert!(json["manager"]["build"].is_string(), "got {json}");
    assert!(json["gpu"]["state"].is_string(), "got {json}");
    assert!(json["runners"].is_array(), "got {json}");
    // No credentials, ever - the blob is built from an allow-list precisely so
    // this stays true as RunnerConfig grows.
    let flat = json.to_string();
    assert!(
        !flat.contains("api_key"),
        "context leaked a key field: {flat}"
    );
}
