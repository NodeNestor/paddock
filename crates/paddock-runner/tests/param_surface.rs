//! What does the runner actually do with every request parameter?
//!
//! `tests/spec/coverage.json` records a disposition per param and
//! `spec_conformance.py` enforces it - but that gate needs a GPU, a model and
//! the Python SDKs, so in practice it runs rarely, and its `rejected` label
//! carries two very different meanings that nobody can tell apart from the
//! matrix alone:
//!
//!   - the field is not in the request struct, so `deny_unknown_fields`
//!     refuses it (genuinely unimplemented), or
//!   - the field is accepted and only the probe VALUE is refused (implemented
//!     for a subset - `partial`, not `rejected`).
//!
//! This file separates them, with no GPU and no model: request validation runs
//! before any dispatch, so a server built on `AppState::for_tests(None)`
//! answers 400 for a refused param and something else for an accepted one. It
//! runs anywhere `cargo test` runs, which is the point - the expensive gate is
//! the one that rots.
#![allow(clippy::unwrap_used)]

use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use paddock_runner::routes::{AppState, router};
use serde_json::{Value, json};
use tower::ServiceExt;

/// A server with no model. Every handler still parses and validates first.
fn app() -> axum::Router {
    router(Arc::new(AppState::for_tests(None)))
}

async fn post(path: &str, body: Value) -> (StatusCode, String) {
    let res = app()
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
    (status, String::from_utf8_lossy(&bytes).into_owned())
}

/// True when the body reads as an unknown-field refusal rather than a
/// deliberate check of ours. That is the line between "we never heard of this
/// param" and "we know it and refused this value".
///
/// The runner rewrites serde's message into OpenAI's own wording
/// ("Unrecognized request argument supplied: X"), which is the right thing for
/// conformance and the reason a naive `contains("unknown field")` classifies
/// every one of them wrong. Both spellings count, plus the deserialize-failed
/// shape serde emits when a field is known but the VALUE has the wrong type.
fn unknown_field(body: &str) -> bool {
    body.contains("unknown field") || body.contains("Unrecognized request argument")
}

fn chat_base() -> Value {
    json!({"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_completion_tokens": 4})
}
fn comp_base() -> Value {
    json!({"model": "m", "prompt": "hi", "max_tokens": 4})
}
fn resp_base() -> Value {
    json!({"model": "m", "input": "hi", "max_output_tokens": 8})
}
fn anth_base() -> Value {
    json!({"model": "m", "max_tokens": 4, "messages": [{"role": "user", "content": "hi"}]})
}

/// Post `base` with `k: v` merged in and classify the outcome.
async fn probe(path: &str, base: Value, k: &str, v: Value) -> Outcome {
    let mut b = base;
    b.as_object_mut().unwrap().insert(k.to_owned(), v);
    let (status, body) = post(path, b).await;
    if status == StatusCode::BAD_REQUEST {
        if unknown_field(&body) {
            Outcome::UnknownField
        } else {
            Outcome::RefusedValue(body)
        }
    } else if status == StatusCode::SERVICE_UNAVAILABLE {
        // No model is loaded, so the request got past PARSING and reached
        // dispatch. That is not the same as "accepted": several checks
        // (`include` values, `top_logprobs` range) sit after the
        // model-availability check and this probe can never reach them.
        // Saying so beats guessing -- the model-bearing gate settles these.
        Outcome::PastParsing
    } else {
        Outcome::Accepted(status)
    }
}

#[derive(Debug)]
enum Outcome {
    /// serde refused it: the field is not in our request struct at all.
    UnknownField,
    /// We parsed it and a deliberate check refused this VALUE.
    RefusedValue(String),
    /// Parsed, and reached dispatch on a model-less server. Whatever
    /// validation lives beyond that point is untested here.
    PastParsing,
    /// Parsed and answered. On a model-less server this should not happen;
    /// if it does the param is genuinely accepted with no check at all.
    Accepted(StatusCode),
}

impl Outcome {
    fn label(&self) -> &'static str {
        match self {
            Outcome::UnknownField => "not-in-struct",
            Outcome::RefusedValue(_) => "value-refused",
            Outcome::PastParsing => "past-parsing*",
            Outcome::Accepted(_) => "ACCEPTED",
        }
    }
}

/// The probe values `spec_conformance.py` uses, so the two agree on what is
/// being asked. Kept verbatim rather than "improved" - a divergence here would
/// make this file describe a surface the gate never tests.
fn probes() -> Vec<(&'static str, Value, &'static str, Value)> {
    vec![
        (
            "/v1/chat/completions",
            chat_base(),
            "audio",
            json!({"voice": "alloy", "format": "wav"}),
        ),
        (
            "/v1/chat/completions",
            chat_base(),
            "function_call",
            json!("auto"),
        ),
        (
            "/v1/chat/completions",
            chat_base(),
            "functions",
            json!([{"name": "f"}]),
        ),
        (
            "/v1/chat/completions",
            chat_base(),
            "moderation",
            json!("auto"),
        ),
        (
            "/v1/chat/completions",
            chat_base(),
            "prediction",
            json!({"type": "content", "content": "x"}),
        ),
        (
            "/v1/chat/completions",
            chat_base(),
            "prompt_cache_options",
            json!({}),
        ),
        (
            "/v1/chat/completions",
            chat_base(),
            "prompt_cache_retention",
            json!("24h"),
        ),
        (
            "/v1/chat/completions",
            chat_base(),
            "web_search_options",
            json!({}),
        ),
        ("/v1/completions", comp_base(), "suffix", json!("!!")),
        ("/v1/responses", resp_base(), "background", json!(true)),
        (
            "/v1/responses",
            resp_base(),
            "context_management",
            json!({}),
        ),
        (
            "/v1/responses",
            resp_base(),
            "conversation",
            json!("conv_1"),
        ),
        (
            "/v1/responses",
            resp_base(),
            "include",
            json!(["reasoning.encrypted_content"]),
        ),
        ("/v1/responses", resp_base(), "moderation", json!("auto")),
        ("/v1/responses", resp_base(), "prompt", json!({"id": "p1"})),
        (
            "/v1/responses",
            resp_base(),
            "prompt_cache_options",
            json!({}),
        ),
        (
            "/v1/responses",
            resp_base(),
            "prompt_cache_retention",
            json!("24h"),
        ),
        (
            "/v1/responses",
            resp_base(),
            "stream_options",
            json!({"include_obfuscation": true}),
        ),
        ("/v1/responses", resp_base(), "top_logprobs", json!(3)),
        ("/v1/messages", anth_base(), "container", json!("c1")),
        ("/v1/messages", anth_base(), "output_config", json!({})),
    ]
}

/// Print the real disposition of every param the matrix calls `rejected`.
/// Not an assertion - a census. The assertion is the next test.
#[tokio::test]
async fn census_of_every_rejected_param() {
    println!("\n{:<24} {:<22} outcome", "param", "endpoint");
    println!("{}", "-".repeat(78));
    for (path, base, k, v) in probes() {
        let out = probe(path, base, k, v).await;
        let extra = match &out {
            Outcome::RefusedValue(b) => {
                let b = b.replace('\n', " ");
                format!("  {}", &b[..b.len().min(90)])
            }
            Outcome::PastParsing => "  parses; any later check needs a model".to_owned(),
            Outcome::Accepted(s) => format!("  ({s})"),
            Outcome::UnknownField => String::new(),
        };
        println!("{k:<24} {path:<22} {}{extra}", out.label());
    }
    println!(
        "
* past-parsing: the field is in the request struct; whether the
              VALUE is refused is decided after model dispatch, so only the
              model-bearing spec gate can settle it."
    );
}

/// A param the matrix calls `rejected` must actually be refused - either shape
/// is fine, but a 200-shaped acceptance means the matrix is lying and the
/// conformance claim with it.
#[tokio::test]
async fn every_rejected_param_is_really_refused() {
    let matrix: Value = serde_json::from_str(
        &std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/spec/coverage.json"),
        )
        .unwrap(),
    )
    .unwrap();
    let fam_of = |path: &str| match path {
        "/v1/chat/completions" => "chat",
        "/v1/completions" => "completions",
        "/v1/responses" => "responses",
        _ => "anthropic_messages",
    };
    let mut lies = Vec::new();
    for (path, base, k, v) in probes() {
        let want = matrix[fam_of(path)][k].as_str().unwrap_or("(absent)");
        let out = probe(path, base, k, v).await;
        if want == "rejected" && matches!(out, Outcome::Accepted(_)) {
            lies.push(format!(
                "{k} on {path}: matrix says rejected, server parsed it"
            ));
        }
    }
    assert!(
        lies.is_empty(),
        "matrix over-states our refusals:\n  {}",
        lies.join("\n  ")
    );
}

// ── legacy functions / function_call  ─────────────────────────────
//
// The pre-2023-11 tool protocol. These probe the TRANSLATION, which is all
// pre-dispatch, so a model-less server exercises every branch: a well-formed
// legacy request must get PAST parsing (503 here, a real answer with a model),
// and every malformed one must take a named 400 rather than serde's.

async fn chat_probe(extra: Vec<(&str, Value)>) -> (StatusCode, String) {
    let mut b = chat_base();
    for (k, v) in extra {
        b.as_object_mut().unwrap().insert(k.to_owned(), v);
    }
    post("/v1/chat/completions", b).await
}

#[tokio::test]
async fn legacy_functions_are_accepted() {
    let (s, body) = chat_probe(vec![(
        "functions",
        json!([{"name": "get_weather", "parameters": {"type": "object"}}]),
    )])
    .await;
    assert_eq!(
        s,
        StatusCode::SERVICE_UNAVAILABLE,
        "should reach dispatch, got: {body}"
    );
}

#[tokio::test]
async fn legacy_function_call_forms_are_accepted() {
    for fc in [json!("none"), json!("auto"), json!({"name": "get_weather"})] {
        let (s, body) = chat_probe(vec![
            ("functions", json!([{"name": "get_weather"}])),
            ("function_call", fc.clone()),
        ])
        .await;
        assert_eq!(
            s,
            StatusCode::SERVICE_UNAVAILABLE,
            "function_call {fc} rejected: {body}"
        );
    }
}

#[tokio::test]
async fn mixing_the_two_tool_generations_is_a_named_400() {
    // Which shape should the ANSWER wear? Unanswerable, so it is refused.
    let (s, body) = chat_probe(vec![
        ("functions", json!([{"name": "f"}])),
        (
            "tools",
            json!([{"type": "function", "function": {"name": "f"}}]),
        ),
    ])
    .await;
    assert_eq!(s, StatusCode::BAD_REQUEST);
    assert!(body.contains("not both"), "unhelpful error: {body}");

    let (s, body) = chat_probe(vec![
        ("functions", json!([{"name": "f"}])),
        ("function_call", json!("auto")),
        ("tool_choice", json!("auto")),
    ])
    .await;
    assert_eq!(s, StatusCode::BAD_REQUEST);
    assert!(body.contains("not both"), "unhelpful error: {body}");
}

#[tokio::test]
async fn malformed_legacy_shapes_are_refused_by_name() {
    let cases: Vec<(Vec<(&str, Value)>, &str)> = vec![
        (vec![("functions", json!([]))], "must not be empty"),
        (
            vec![("functions", json!([{"description": "no name"}]))],
            "string `name`",
        ),
        (
            vec![("function_call", json!("auto"))],
            "requires `functions`",
        ),
        (
            vec![
                ("functions", json!([{"name": "f"}])),
                ("function_call", json!("sometimes")),
            ],
            "invalid function_call",
        ),
    ];
    for (extra, want) in cases {
        let (s, body) = chat_probe(extra).await;
        assert_eq!(s, StatusCode::BAD_REQUEST, "should be refused: {body}");
        assert!(body.contains(want), "expected {want:?} in: {body}");
    }
}

#[tokio::test]
async fn the_matrix_agrees_that_both_are_implemented() {
    // The disposition and the behaviour move together or the gate is fiction.
    let matrix: Value = serde_json::from_str(
        &std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/spec/coverage.json"),
        )
        .unwrap(),
    )
    .unwrap();
    for k in ["functions", "function_call"] {
        assert_eq!(
            matrix["chat"][k], "implemented",
            "coverage.json still refuses {k}"
        );
    }
}

// ── anthropic output_config ───────────────────────────────────────
//
// `{effort, format}`: Anthropic's own graded reasoning ladder and their own
// structured output. Both used to be `rejected` on reasoning the
// spec has since overtaken - the code said "Anthropic's schema has no effort
// concept" and "it has no `response_format`", and both are now false.

async fn anth_probe(extra: Vec<(&str, Value)>) -> (StatusCode, String) {
    let mut b = anth_base();
    for (k, v) in extra {
        b.as_object_mut().unwrap().insert(k.to_owned(), v);
    }
    post("/v1/messages", b).await
}

#[tokio::test]
async fn output_config_is_accepted_on_both_halves() {
    let cases = vec![
        json!({"effort": "high"}),
        json!({"format": {"type": "json_schema", "schema": {"type": "object"}}}),
        json!({"effort": "low", "format": {"type": "json_schema", "schema": {"type": "object"}}}),
        json!({}),
    ];
    for cfg in cases {
        let (s, body) = anth_probe(vec![("output_config", cfg.clone())]).await;
        assert_eq!(
            s,
            StatusCode::SERVICE_UNAVAILABLE,
            "output_config {cfg} rejected: {body}"
        );
    }
}

#[tokio::test]
async fn count_tokens_takes_output_config_too() {
    // The SDK's MessageCountTokensParams lists it, and on a graded model the
    // rung is a template kwarg that can change the rendered prompt - so a
    // counting client must not be refused for sending what it will send.
    let (s, body) = post(
        "/v1/messages/count_tokens",
        json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "output_config": {"effort": "high"}
        }),
    )
    .await;
    assert_ne!(
        s,
        StatusCode::BAD_REQUEST,
        "count_tokens refused output_config: {body}"
    );
}

#[tokio::test]
async fn malformed_output_config_is_refused_by_name() {
    let cases: Vec<(Value, &str)> = vec![
        (json!("high"), "must be an object"),
        (json!({"effrot": "high"}), "unsupported output_config field"),
        (json!({"effort": 3}), "must be a string"),
        (
            json!({"format": {"type": "regex", "pattern": "x"}}),
            "unsupported output_config.format",
        ),
        (
            json!({"format": {"schema": {"type": "object"}}}),
            "needs a `type`",
        ),
        (
            json!({"format": {"type": "json_schema"}}),
            "schema is required",
        ),
    ];
    for (cfg, want) in cases {
        let (s, body) = anth_probe(vec![("output_config", cfg.clone())]).await;
        assert_eq!(
            s,
            StatusCode::BAD_REQUEST,
            "{cfg} should be refused, got {s}: {body}"
        );
        assert!(
            body.contains(want),
            "expected {want:?} for {cfg} in: {body}"
        );
    }
}

#[tokio::test]
async fn the_matrix_agrees_output_config_is_implemented() {
    let matrix: Value = serde_json::from_str(
        &std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/spec/coverage.json"),
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(matrix["anthropic_messages"]["output_config"], "implemented");
}

// ── completions n / best_of / echo, responses stream_options  ─────

async fn comp_probe(extra: Vec<(&str, Value)>) -> (StatusCode, String) {
    let mut b = comp_base();
    for (k, v) in extra {
        b.as_object_mut().unwrap().insert(k.to_owned(), v);
    }
    post("/v1/completions", b).await
}

#[tokio::test]
async fn completions_n_best_of_and_echo_are_accepted() {
    let cases: Vec<Vec<(&str, Value)>> = vec![
        vec![("echo", json!(true))],
        vec![("n", json!(4))],
        vec![("n", json!(2)), ("best_of", json!(5))],
        vec![("best_of", json!(3))],
        vec![
            ("echo", json!(true)),
            ("n", json!(2)),
            ("logprobs", json!(3)),
        ],
    ];
    for extra in cases {
        let names: Vec<&str> = extra.iter().map(|(k, _)| *k).collect();
        let (s, body) = comp_probe(extra).await;
        assert_eq!(
            s,
            StatusCode::SERVICE_UNAVAILABLE,
            "{names:?} rejected: {body}"
        );
    }
}

#[tokio::test]
async fn completions_n_and_best_of_bounds_are_named() {
    let cases: Vec<(Vec<(&str, Value)>, &str)> = vec![
        (vec![("n", json!(0))], "n must be 1..=8"),
        (vec![("n", json!(9))], "n must be 1..=8"),
        // best_of below n cannot satisfy the request: there is nothing to rank
        (vec![("n", json!(4)), ("best_of", json!(2))], "must be >= n"),
        (vec![("best_of", json!(20))], "best_of must be 1..=8"),
        // ranking only resolves once every candidate has finished
        (
            vec![
                ("n", json!(1)),
                ("best_of", json!(3)),
                ("stream", json!(true)),
            ],
            "cannot be combined with stream",
        ),
    ];
    for (extra, want) in cases {
        let (s, body) = comp_probe(extra).await;
        assert_eq!(s, StatusCode::BAD_REQUEST, "should be refused: {body}");
        assert!(body.contains(want), "expected {want:?} in: {body}");
    }
}

#[tokio::test]
async fn best_of_equal_to_n_streams_fine() {
    // Only best_of > n forces the barrier; best_of == n is just n, and a
    // blanket "best_of + stream is an error" would refuse a legal request.
    let (s, body) = comp_probe(vec![
        ("n", json!(2)),
        ("best_of", json!(2)),
        ("stream", json!(true)),
    ])
    .await;
    assert_eq!(
        s,
        StatusCode::SERVICE_UNAVAILABLE,
        "wrongly refused: {body}"
    );
}

#[tokio::test]
async fn responses_stream_options_takes_only_what_we_can_honour() {
    // We emit no obfuscation padding, so `false` describes this server exactly
    // and `true` asks for bytes that never arrive - refused rather than
    // silently ignored.
    let mut ok = resp_base();
    ok.as_object_mut()
        .unwrap()
        .insert("stream".into(), json!(true));
    ok.as_object_mut().unwrap().insert(
        "stream_options".into(),
        json!({"include_obfuscation": false}),
    );
    let (s, body) = post("/v1/responses", ok).await;
    assert_eq!(
        s,
        StatusCode::SERVICE_UNAVAILABLE,
        "false should pass: {body}"
    );

    for (opts, stream, want) in [
        (
            json!({"include_obfuscation": true}),
            true,
            "emits no obfuscation padding",
        ),
        (
            json!({"include_obfuscation": false}),
            false,
            "requires stream: true",
        ),
        (
            json!({"include_usage": true}),
            true,
            "unsupported stream_options field",
        ),
        (
            json!({"include_obfuscation": "yes"}),
            true,
            "must be a boolean",
        ),
    ] {
        let mut b = resp_base();
        b.as_object_mut()
            .unwrap()
            .insert("stream".into(), json!(stream));
        b.as_object_mut()
            .unwrap()
            .insert("stream_options".into(), opts.clone());
        let (s, body) = post("/v1/responses", b).await;
        assert_eq!(
            s,
            StatusCode::BAD_REQUEST,
            "{opts} should be refused: {body}"
        );
        assert!(
            body.contains(want),
            "expected {want:?} for {opts} in: {body}"
        );
    }
}

#[tokio::test]
async fn the_matrix_agrees_with_this_batch() {
    let matrix: Value = serde_json::from_str(
        &std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/spec/coverage.json"),
        )
        .unwrap(),
    )
    .unwrap();
    for k in ["n", "best_of", "echo"] {
        assert_eq!(matrix["completions"][k], "implemented", "completions.{k}");
    }
    // partial, not implemented: one value of one field is honoured
    assert_eq!(matrix["responses"]["stream_options"], "partial");
    // and `suffix` stays refused deliberately - FIM is a per-model token
    // vocabulary, so it is a model capability rather than a request parameter
    assert_eq!(matrix["completions"]["suffix"], "rejected");
}

// ── the last three  ───────────────────────────────────────────────
//
// Two of these were on the "implementable, we have the subsystem" list and came
// off it once the code was read. Having the subsystem is necessary and not
// sufficient: each needs a REQUEST-SCOPED path into it that does not exist.

#[tokio::test]
async fn prediction_and_web_search_options_are_refused_by_name() {
    // Not by serde's generic unknown-field message: a caller should learn the
    // reason and, where there is one, the surface that does serve it.
    let (s, body) = chat_probe(vec![(
        "prediction",
        json!({"type": "content", "content": "the expected answer"}),
    )])
    .await;
    assert_eq!(s, StatusCode::BAD_REQUEST);
    assert!(
        body.contains("drafts from the model itself"),
        "unhelpful: {body}"
    );

    let (s, body) = chat_probe(vec![("web_search_options", json!({}))]).await;
    assert_eq!(s, StatusCode::BAD_REQUEST);
    assert!(
        body.contains("/v1/responses"),
        "should point at the surface that serves it: {body}"
    );
}

#[tokio::test]
async fn the_matrix_still_refuses_the_two_that_need_engine_work() {
    let matrix: Value = serde_json::from_str(
        &std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/spec/coverage.json"),
        )
        .unwrap(),
    )
    .unwrap();
    // A better error message does not make a param implemented.
    assert_eq!(matrix["chat"]["prediction"], "rejected");
    assert_eq!(matrix["chat"]["web_search_options"], "rejected");
    // chunking_strategy is served, for a subset: auto / server_vad(+threshold)
    assert_eq!(
        matrix["audio_transcriptions"]["chunking_strategy"],
        "partial"
    );
}
