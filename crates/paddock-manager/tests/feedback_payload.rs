//! What actually goes on the wire to the truespar API.
//!
//! Its own test BINARY, not another case in `feedback_api.rs`: both files drive
//! `PADDOCK_API_BASE`, and cargo runs the tests inside one binary on parallel
//! threads - so cases setting the same process-wide env var race. A separate
//! file buys separation from `feedback_api.rs`; [`ENV`] below buys it between
//! the two cases here, which is the half this file originally got wrong (each
//! test's submission landed on the other's stub, and the orphaned receiver hung
//! the run forever rather than failing it).
//!
//! This is the contract in `truespar-core/`.
//! If a field name here changes, that document is wrong and somebody is about to
//! implement against it.

use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::routing::post;
use paddock_manager::routes::{AppState, router};
use tokio::sync::oneshot;
use tower::ServiceExt;

/// One test at a time may own `PADDOCK_API_BASE`.
///
/// `tokio::sync::Mutex`, not `std::sync::Mutex`: the guard is deliberately held
/// across `.await` (that is the whole point - it must cover the submission), and
/// this is the lock built for that.
static ENV: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

/// Belt and braces on the hang: even with the lock, a bug that stops the
/// submission from arriving must fail rather than block the suite forever. An
/// awaited oneshot with no sender has no timeout of its own.
async fn captured(rx: oneshot::Receiver<serde_json::Value>) -> serde_json::Value {
    tokio::time::timeout(std::time::Duration::from_secs(10), rx)
        .await
        .expect("the manager never POSTed to the stub - it did not forward")
        .expect("capture channel dropped")
}

/// Stand up a one-shot listener that captures the first POSTed body, and answer
/// the way the real API does so the manager's success path runs to completion.
async fn capture_one() -> (String, oneshot::Receiver<serde_json::Value>) {
    let (tx, rx) = oneshot::channel();
    let tx = Arc::new(std::sync::Mutex::new(Some(tx)));

    let app = axum::Router::new().route(
        "/api/feedback",
        post(move |axum::Json(body): axum::Json<serde_json::Value>| {
            let tx = tx.clone();
            async move {
                if let Some(tx) = tx.lock().expect("capture lock").take() {
                    let _ = tx.send(body);
                }
                axum::Json(serde_json::json!({
                    "ok": true,
                    "id": "11111111-2222-3333-4444-555555555555"
                }))
            }
        }),
    );

    // Port 0 - the OS picks a free one, so parallel runs never collide.
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let addr = listener.local_addr().expect("addr");
    tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });

    (format!("http://{addr}"), rx)
}

#[tokio::test]
async fn the_wire_payload_matches_the_documented_contract() {
    let _env = ENV.lock().await;
    let (base, rx) = capture_one().await;
    unsafe { std::env::set_var("PADDOCK_API_BASE", &base) };

    let app = router(Arc::new(AppState::for_tests()));
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/feedback")
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::json!({
                        "category": "bug",
                        "message": "  the engine will not start  ",
                        "email": "  someone@example.com  ",
                        "include_context": true
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .expect("router responds");

    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20)
        .await
        .expect("body");
    let reply: serde_json::Value = serde_json::from_slice(&bytes).expect("json");
    assert_eq!(
        reply["ok"], true,
        "the upstream body is relayed, not rewritten"
    );
    assert!(
        reply["id"].is_string(),
        "the id comes back for the user's reference"
    );

    let sent = captured(rx).await;

    // The identity fields - what the handoff doc asks the API to start binding.
    assert_eq!(sent["appIdentifier"], "paddock");
    assert_eq!(sent["category"], "bug");
    // Trimmed on the way out: leading whitespace in a report body is noise, and
    // a padded email address is not a deliverable one.
    assert_eq!(sent["message"], "the engine will not start");
    assert_eq!(sent["email"], "someone@example.com");

    // appVersion is the LONG stamp, not the bare SemVer - the whole reason a bug
    // report is worth more than a version number.
    assert_eq!(sent["appVersion"], paddock_admin::version::LONG);

    // platform is {os}-{arch}, not a bare consts::OS. traverse shipped the bare
    // form and it silently never matched; the mapping is shared with the update
    // path precisely so that cannot happen twice.
    let platform = sent["platform"].as_str().expect("platform string");
    assert!(
        platform.contains('-') && !platform.is_empty(),
        "platform must be {{os}}-{{arch}}, got {platform}"
    );

    // Context is present because it was asked for, and carries the three blocks.
    assert!(
        sent["context"]["manager"]["build"].is_string(),
        "got {sent}"
    );
    assert!(sent["context"]["gpu"]["state"].is_string(), "got {sent}");
    assert!(sent["context"]["runners"].is_array(), "got {sent}");

    unsafe { std::env::remove_var("PADDOCK_API_BASE") };
}

/// The real API, for real. `#[ignore]` because it POSTs to production:
///
///   cargo test -p paddock-manager --test feedback_payload -- --ignored
///
/// Run it when the feedback API changes on the truespar side, and sparingly -
/// an anonymous IP gets FIVE submissions an hour and paddock has no licence key
/// to skip the limiter, so a loop of these locks the box out for an hour and the
/// failure will look like a paddock bug.
///
/// Deliberately goes through the manager's own router rather than curling a
/// hand-written body: what needs proving is that the bytes PADDOCK sends are
/// accepted, not that a payload somebody retyped from the docs is.
#[tokio::test]
#[ignore = "posts to the production feedback API"]
async fn a_real_submission_reaches_the_live_api() {
    let _env = ENV.lock().await;
    // No override: fall through to the real DEFAULT_API_BASE.
    unsafe { std::env::remove_var("PADDOCK_API_BASE") };

    let app = router(Arc::new(AppState::for_tests()));
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/feedback")
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::json!({
                        "category": "feedback",
                        "message": "Smoke test from paddock's feedback path - \
                                    ignore. Confirms appIdentifier and context \
                                    are bound.",
                        "include_context": true
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .expect("router responds");

    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20)
        .await
        .expect("body");
    let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null);

    assert_eq!(status, StatusCode::OK, "the live API refused us: {body}");
    assert_eq!(body["ok"], true, "got {body}");
    assert!(
        body["id"].is_string(),
        "a stored row must come back with its id, got {body}"
    );
    println!("live feedback accepted, id = {}", body["id"]);
}

/// The privacy default, checked on the wire rather than in the UI. A dialog can
/// be rewritten; this is the line that actually decides whether diagnostics
/// leave the machine.
#[tokio::test]
async fn context_is_absent_unless_it_was_asked_for() {
    let _env = ENV.lock().await;
    let (base, rx) = capture_one().await;
    unsafe { std::env::set_var("PADDOCK_API_BASE", &base) };

    let app = router(Arc::new(AppState::for_tests()));
    let _ = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/feedback")
                .header("content-type", "application/json")
                // No `include_context` key at all - a client that forgets the
                // field must get the private behaviour.
                .body(Body::from(
                    serde_json::json!({ "category": "feedback", "message": "nice work" })
                        .to_string(),
                ))
                .unwrap(),
        )
        .await
        .expect("router responds");

    let sent = captured(rx).await;
    assert!(
        sent.get("context").is_none(),
        "diagnostics must not ride along by default, got {sent}"
    );
    // An omitted email is omitted, not sent as an empty string.
    assert!(sent.get("email").is_none(), "got {sent}");

    unsafe { std::env::remove_var("PADDOCK_API_BASE") };
}
