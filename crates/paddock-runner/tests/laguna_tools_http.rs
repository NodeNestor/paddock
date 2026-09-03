//! Forced tool calls on laguna, over all three API surfaces.
//!
//! Laguna's call syntax is GLM-shaped and unpadded -
//! `<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value>...</tool_call>` -
//! with two properties that make it worth its own end-to-end gate rather than
//! unit tests alone:
//!
//! 1. The function name rides BARE after the opener, so the grammar's candidate
//!    literals have to run into the following tag to stay prefix-free.
//! 2. Values are TYPED by the template (`v | tojson if v is not string else v`),
//!    so a declared integer must come back as a number and a declared string
//!    bare. Getting that wrong is silent: `coerce` would just hand the caller a
//!    string where the tool declared a number.
//!
//! Heavy (~20 GB): PADDOCK_HEAVY_TESTS=1, the model on disk, pack, GPU.
//! Run --release --test-threads=1.
#![allow(clippy::unwrap_used)]

use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use paddock_runner::routes::{AppState, router};
use paddock_runner::serving;
use tower::ServiceExt;

const MODEL: &str = "laguna-xs";

/// `LAGUNA_XS_GGUF`, else `PADDOCK_MODELS_DIR`, else `models/` under the
/// workspace root (gitignored, so a symlink to your own store works).
fn model_path() -> std::path::PathBuf {
    if let Ok(p) = std::env::var("LAGUNA_XS_GGUF") {
        return p.into();
    }
    std::env::var_os("PADDOCK_MODELS_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models")
        })
        .join("Laguna-XS-2.1-GGUF/Laguna-XS-2.1-Q4_K_M.gguf")
}

/// The grammar's load-bearing assumption about laguna's vocabulary, asserted
/// rather than remembered: `GatedConstraint::allows` rejects every SPECIAL
/// token while a grammar is active, so if `<tool_call>` were one the forced
/// call would deadlock on its very first token with nothing legal. Measured:
/// it is a single atomic token (id 25) and it is not special, while
/// `<arg_key>`/`<arg_value>` are not tokens at all (four ordinary BPE pieces
/// each). `</assistant>` is special - but that is the turn terminator, which
/// the engine handles as a stop token and the grammar never emits.
///
/// Cheap: mmap + tokenizer, no GPU. If a future GGUF re-types these markers,
/// this fails with the reason instead of a mysterious empty generation.
#[test]
fn laguna_tool_markers_are_not_special_tokens() {
    if std::env::var_os("PADDOCK_HEAVY_TESTS").is_none() {
        return;
    }
    let path = model_path();
    if !path.exists() {
        eprintln!("model missing - skipping");
        return;
    }
    let map = paddock_models::mapped::MappedGguf::open(&path).expect("open gguf");
    let tok = paddock_tokenizer::GgufTokenizer::from_gguf(map.gguf()).expect("tokenizer");
    for m in [
        "<tool_call>",
        "</tool_call>",
        "<arg_key>",
        "</arg_key>",
        "<arg_value>",
        "</arg_value>",
    ] {
        let Some(id) = tok.token_to_id(m) else {
            // spelled from ordinary pieces: no specialness question at all
            continue;
        };
        let visible = tok.decode(&[id], true).unwrap_or_default();
        assert!(
            !visible.is_empty(),
            "{m} (id {id}) is a SPECIAL token - the forced-tool grammar cannot emit it, \
             see the deadlock note in this test"
        );
    }
}

fn app() -> Option<axum::Router> {
    if std::env::var_os("PADDOCK_HEAVY_TESTS").is_none() {
        eprintln!("set PADDOCK_HEAVY_TESTS=1 to run the laguna tool gates");
        return None;
    }
    let path = model_path();
    let pack = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../packs/cuda/build/pd-cuda-sm86.dll");
    if !path.exists() || !pack.exists() {
        eprintln!("model or pack missing - skipping");
        return None;
    }
    let model = serving::load(
        MODEL.into(),
        &path,
        "cuda",
        0,
        Some(&pack),
        4096,
        4,
        None,
        None,
        None,
        None,
    )
    .map_err(|e| eprintln!("load: {e}"))
    .ok()?;
    Some(router(Arc::new(AppState::for_tests(Some(model)))))
}

async fn post(
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

/// `city` is a string (written bare), `days` an integer (tojson'd), `units` a
/// string enum - one parameter per value mode the grammar has.
fn schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "days": {"type": "integer"},
            "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}
        },
        "required": ["city", "days"]
    })
}

const ASK: &str = "Say hello.";

/// A forced call names the tool and its arguments carry the DECLARED types,
/// not strings that merely look like them. `days` being a JSON number is the
/// sharp assertion: laguna's template writes non-strings through `tojson`, so
/// a grammar that treated every value as free text would still round-trip
/// through the parser - as `"3"`.
fn assert_weather_call(name: &str, args: &serde_json::Value, where_: &str) {
    assert_eq!(name, "get_weather", "{where_}: wrong tool");
    let obj = args
        .as_object()
        .unwrap_or_else(|| panic!("{where_}: args not an object: {args}"));
    assert!(
        obj["city"].is_string(),
        "{where_}: city must be a string: {args}"
    );
    assert!(
        obj["days"].is_number(),
        "{where_}: days must be a NUMBER, not text that looks like one: {args}"
    );
    if let Some(u) = obj.get("units") {
        let u = u.as_str().unwrap_or_default();
        assert!(
            ["celsius", "fahrenheit"].contains(&u),
            "{where_}: enum escaped: {u:?}"
        );
    }
    assert!(
        obj.keys()
            .all(|k| ["city", "days", "units"].contains(&k.as_str())),
        "{where_}: undeclared argument: {args}"
    );
}

#[tokio::test]
async fn a_forced_tool_call_is_typed_on_every_surface() {
    let Some(app) = app() else { return };

    // Anthropic: flat tools with `input_schema`, forced with {"type":"any"}
    let (s, j) = post(
        app.clone(),
        "/v1/messages",
        serde_json::json!({
            "model": MODEL, "max_tokens": 400, "temperature": 0.0,
            "messages": [{"role": "user", "content": ASK}],
            "tools": [{"name": "get_weather", "description": "look up a forecast",
                       "input_schema": schema()}],
            "tool_choice": {"type": "any"}
        }),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "messages: {j}");
    eprintln!("messages: {}", j["content"]);
    assert_eq!(j["stop_reason"], "tool_use", "{j}");
    let block = j["content"]
        .as_array()
        .and_then(|bs| bs.iter().find(|b| b["type"] == "tool_use"))
        .unwrap_or_else(|| panic!("no tool_use block: {j}"));
    assert_weather_call(
        block["name"].as_str().unwrap_or_default(),
        &block["input"],
        "messages",
    );

    // chat completions: nested tools, `tool_choice: "required"`
    let (s, j) = post(
        app.clone(),
        "/v1/chat/completions",
        serde_json::json!({
            "model": MODEL, "max_tokens": 400, "temperature": 0.0,
            "messages": [{"role": "user", "content": ASK}],
            "tools": [{"type": "function", "function": {
                "name": "get_weather", "parameters": schema()}}],
            "tool_choice": "required"
        }),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "chat: {j}");
    let call = &j["choices"][0]["message"]["tool_calls"][0];
    eprintln!("chat: {call}");
    assert_eq!(j["choices"][0]["finish_reason"], "tool_calls", "{j}");
    let args: serde_json::Value =
        serde_json::from_str(call["function"]["arguments"].as_str().unwrap_or_default())
            .unwrap_or_else(|e| panic!("chat: arguments not JSON ({e}): {call}"));
    assert_weather_call(
        call["function"]["name"].as_str().unwrap_or_default(),
        &args,
        "chat",
    );

    // responses: flat tools, a function_call output item
    let (s, j) = post(
        app.clone(),
        "/v1/responses",
        serde_json::json!({
            "model": MODEL, "max_output_tokens": 400, "temperature": 0.0,
            "input": [{"type": "message", "role": "user",
                       "content": [{"type": "input_text", "text": ASK}]}],
            "tools": [{"type": "function", "name": "get_weather", "parameters": schema()}],
            "tool_choice": "required"
        }),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "responses: {j}");
    let item = j["output"]
        .as_array()
        .and_then(|is| is.iter().find(|i| i["type"] == "function_call"))
        .unwrap_or_else(|| panic!("no function_call item: {j}"));
    eprintln!("responses: {item}");
    let args: serde_json::Value =
        serde_json::from_str(item["arguments"].as_str().unwrap_or_default())
            .unwrap_or_else(|e| panic!("responses: arguments not JSON ({e}): {item}"));
    assert_weather_call(
        item["name"].as_str().unwrap_or_default(),
        &args,
        "responses",
    );

    // the named form picks one tool out of several whose names SHARE A PREFIX -
    // the case laguna's bare-name opener would get wrong
    let (s, j) = post(
        app,
        "/v1/messages",
        serde_json::json!({
            "model": MODEL, "max_tokens": 400, "temperature": 0.0,
            "messages": [{"role": "user", "content": ASK}],
            "tools": [
                {"name": "get", "input_schema": {"type": "object", "properties": {}}},
                {"name": "get_weather", "input_schema": schema()},
            ],
            "tool_choice": {"type": "tool", "name": "get_weather"}
        }),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "messages named: {j}");
    let block = j["content"]
        .as_array()
        .and_then(|bs| bs.iter().find(|b| b["type"] == "tool_use"))
        .unwrap_or_else(|| panic!("no tool_use block: {j}"));
    eprintln!("messages named: {block}");
    assert_weather_call(
        block["name"].as_str().unwrap_or_default(),
        &block["input"],
        "messages named",
    );
}
