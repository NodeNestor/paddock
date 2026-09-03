//! granite-vision over all three API surfaces, end to end.
//!
//! `granite_vision_template.rs` renders the template; this drives the whole
//! request path - surface conversion, image extraction, the tower, the
//! sampler - for `/v1/chat/completions`, `/v1/messages` and `/v1/responses`.
//!
//! Why granite in particular: its template tests the content-part type as a
//! STRING, so it is the one that notices what a surface leaves behind. Both
//! historical bugs here were that class and both were silent - extraction
//! ordered after normalization meant `/v1/messages` could not accept a picture
//! at all (`source` is Anthropic's only inline image shape), and Responses'
//! `input_text` parts were skipped, so an image sent with a question arrived
//! with no question.
//!
//! The assertions are cross-surface EQUIVALENCE rather than answer quality:
//! same picture, same words, temperature 0 - the three surfaces must put the
//! same prompt in front of the model and get the same tokens back. That holds
//! whatever granite makes of a synthetic chart, so the test measures the
//! plumbing and nothing else.
//!
//! Heavy (~5 GB residency): PADDOCK_HEAVY_TESTS=1, the model + mmproj on disk,
//! pack, GPU. Run --release --test-threads=1.
#![allow(clippy::unwrap_used)]

use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use base64::Engine as _;
use http_body_util::BodyExt;
use paddock_runner::routes::{AppState, router};
use paddock_runner::serving;
use tower::ServiceExt;

const MODEL: &str = "granite-vision-4.1-4b";

/// `GRANITE_VISION_DIR`, else `PADDOCK_MODELS_DIR`, else `models/` under the
/// workspace root (gitignored, so a symlink to your own store works).
fn model_dir() -> std::path::PathBuf {
    if let Ok(d) = std::env::var("GRANITE_VISION_DIR") {
        return d.into();
    }
    std::env::var_os("PADDOCK_MODELS_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models")
        })
        .join("granite-vision-4.1-4b-GGUF")
}

fn app() -> Option<axum::Router> {
    if std::env::var_os("PADDOCK_HEAVY_TESTS").is_none() {
        eprintln!("set PADDOCK_HEAVY_TESTS=1 to run the granite-vision http gates");
        return None;
    }
    let dir = model_dir();
    let model_path = dir.join("granite-vision-4.1-4b-Q8_0.gguf");
    let mmproj = dir.join("mmproj-model-f16.gguf");
    let pack = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../packs/cuda/build/pd-cuda-sm86.dll");
    if !model_path.exists() || !mmproj.exists() || !pack.exists() {
        eprintln!("model/mmproj/pack missing - skipping");
        return None;
    }
    let model = serving::load(
        MODEL.into(),
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

/// A 320x200 bar chart as a 24-bit BMP: white ground, four dark bars of
/// different heights on a baseline. Granite is a document model, so give it
/// something document-shaped - but note that no assertion here depends on what
/// it reads, only on the three surfaces agreeing.
fn bar_chart_bmp() -> Vec<u8> {
    let (w, h) = (320usize, 200usize);
    let img = w * h * 3;
    let mut out = Vec::with_capacity(54 + img);
    out.extend_from_slice(b"BM");
    out.extend_from_slice(&(54u32 + img as u32).to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(&54u32.to_le_bytes());
    out.extend_from_slice(&40u32.to_le_bytes());
    out.extend_from_slice(&(w as i32).to_le_bytes());
    out.extend_from_slice(&(h as i32).to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&24u16.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(&(img as u32).to_le_bytes());
    out.extend_from_slice(&2835u32.to_le_bytes());
    out.extend_from_slice(&2835u32.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    // four bars, left to right, heights in chart units; BMP rows run bottom-up
    // so y here is already "height above the baseline"
    const BARS: [(usize, usize, usize); 4] = [
        (30, 70, 60),
        (100, 140, 120),
        (170, 210, 90),
        (240, 280, 160),
    ];
    for y in 0..h {
        for x in 0..w {
            let axis = x == 20 || y == 20; // L-shaped axes
            let bar = y > 20
                && BARS
                    .iter()
                    .any(|&(x0, x1, top)| x >= x0 && x < x1 && y - 20 < top);
            if axis || bar {
                out.extend_from_slice(&[40, 40, 40]);
            } else {
                out.extend_from_slice(&[255, 255, 255]);
            }
        }
    }
    out
}

fn data_uri() -> String {
    format!(
        "data:image/bmp;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(bar_chart_bmp())
    )
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

/// The same picture and the same words in each surface's own image shape:
/// OpenAI `image_url`, Anthropic `{"type":"image","source":{base64,...}}`,
/// Responses `input_image` + `input_text`.
fn chat_body(uri: &str, text: &str) -> serde_json::Value {
    serde_json::json!({
        "model": MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": uri}},
            {"type": "text", "text": text}
        ]}],
        "max_tokens": 120, "temperature": 0.0
    })
}

fn messages_body(uri: &str, text: &str) -> serde_json::Value {
    let b64 = uri
        .strip_prefix("data:image/bmp;base64,")
        .expect("data uri");
    serde_json::json!({
        "model": MODEL,
        "max_tokens": 120, "temperature": 0.0,
        "messages": [{"role": "user", "content": [
            {"type": "image", "source": {
                "type": "base64", "media_type": "image/bmp", "data": b64}},
            {"type": "text", "text": text}
        ]}]
    })
}

fn responses_body(uri: &str, text: &str) -> serde_json::Value {
    serde_json::json!({
        "model": MODEL,
        "max_output_tokens": 120, "temperature": 0.0,
        "input": [{"type": "message", "role": "user", "content": [
            {"type": "input_image", "image_url": uri},
            {"type": "input_text", "text": text}
        ]}]
    })
}

/// The assistant text out of each surface's own response shape.
fn chat_text(j: &serde_json::Value) -> String {
    j["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or_default()
        .to_owned()
}

fn messages_text(j: &serde_json::Value) -> String {
    j["content"]
        .as_array()
        .map(|blocks| {
            blocks
                .iter()
                .filter(|b| b["type"] == "text")
                .filter_map(|b| b["text"].as_str())
                .collect::<String>()
        })
        .unwrap_or_default()
}

fn responses_text(j: &serde_json::Value) -> String {
    j["output"]
        .as_array()
        .map(|items| {
            items
                .iter()
                .filter(|i| i["type"] == "message")
                .filter_map(|i| i["content"].as_array())
                .flatten()
                .filter_map(|c| c["text"].as_str())
                .collect::<String>()
        })
        .unwrap_or_default()
}

/// Run one prompt through all three surfaces and return (text, prompt_tokens)
/// for each. Serial deliberately - a multimodal round is exclusive anyway, and
/// the point is the comparison, not the throughput.
async fn three_ways(app: &axum::Router, text: &str) -> [(String, u64); 3] {
    let uri = data_uri();

    let (s, j) = post(app.clone(), "/v1/chat/completions", chat_body(&uri, text)).await;
    assert_eq!(s, StatusCode::OK, "chat: {j}");
    let chat = (
        chat_text(&j),
        j["usage"]["prompt_tokens"].as_u64().unwrap_or_default(),
    );

    let (s, j) = post(app.clone(), "/v1/messages", messages_body(&uri, text)).await;
    assert_eq!(s, StatusCode::OK, "messages: {j}");
    let msgs = (
        messages_text(&j),
        j["usage"]["input_tokens"].as_u64().unwrap_or_default(),
    );

    let (s, j) = post(app.clone(), "/v1/responses", responses_body(&uri, text)).await;
    assert_eq!(s, StatusCode::OK, "responses: {j}");
    let resp = (
        responses_text(&j),
        j["usage"]["input_tokens"].as_u64().unwrap_or_default(),
    );

    [chat, msgs, resp]
}

/// A plain question about an image, and then a TASK TAG - the model's real
/// interface, which travels as ordinary message text and so is exactly what a
/// surface that mishandles text parts silently disables.
///
/// Equal prompt-token counts are the sharper half of this: they say the three
/// surfaces built the same prompt, not merely that each built a usable one.
#[tokio::test]
async fn all_three_surfaces_send_the_same_prompt_and_get_the_same_answer() {
    let Some(app) = app() else { return };

    for prompt in ["What is in this image? Answer briefly.", "<chart2csv>"] {
        let [chat, msgs, resp] = three_ways(&app, prompt).await;
        eprintln!("--- {prompt:?}");
        eprintln!("chat      ({} prompt tokens): {:?}", chat.1, chat.0);
        eprintln!("messages  ({} prompt tokens): {:?}", msgs.1, msgs.0);
        eprintln!("responses ({} prompt tokens): {:?}", resp.1, resp.0);

        assert!(!chat.0.trim().is_empty(), "chat returned nothing");
        // image rows are counted  - a text-only prompt this
        // short could never reach three digits, so this pins that the tower
        // actually ran on every surface
        assert!(
            chat.1 > 100,
            "prompt_tokens {} omits the image rows",
            chat.1
        );
        assert_eq!(chat.1, msgs.1, "chat vs anthropic built different prompts");
        assert_eq!(chat.1, resp.1, "chat vs responses built different prompts");
        assert_eq!(chat.0, msgs.0, "chat vs anthropic answered differently");
        assert_eq!(chat.0, resp.0, "chat vs responses answered differently");
    }
}

/// Structured output on AN IMAGE REQUEST. `response_format` compiles to a
/// `ConstraintSpec` in `prepare()` while the multimodal path only rewrites
/// `engine_prompt`, so the two compose by construction - but "by construction"
/// is an argument, and this is the documented answer to "how do I get JSON out
/// of granite-vision", so it should rest on a test.
///
/// The grammar is enforced at the token level (illegal tokens are masked, and
/// a stop token is legal only when the machine may end), so a run that ENDS
/// must parse and match the schema whatever granite makes of the picture. The
/// finish-reason assertions carry that "ends" - a length-truncated constrained
/// generation is a legitimately unparseable prefix, so a test that just called
/// `from_str` would be measuring the token budget, not the grammar.
///
/// The schema is deliberately SMALL AND CLOSED. The first cut asked for
/// `{"title": string, "bars": array of number}` alongside `<tables_json>`, and
/// the two fought: the tag's own instruction demands "a list of dictionaries",
/// so under a conflicting schema the model went off-distribution and filled an
/// unbounded array with zeros until it hit the cap. Unbounded arrays are a
/// budget trap in a grammar test - nothing can force the model to close one.
#[tokio::test]
async fn structured_output_composes_with_an_image() {
    let Some(app) = app() else { return };
    let uri = data_uri();
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "chart_type": {"type": "string"},
            "bar_count": {"type": "integer"}
        },
        "required": ["chart_type", "bar_count"]
    });
    const ASK: &str = "Describe this chart.";

    /// What JSON Schema's `integer` actually promises: a number with no
    /// fractional part. Not "fits in i64" - the compiled-schema subset has no
    /// `minimum`/`maximum` (those are a 400 at compile, by design), so nothing
    /// bounds the magnitude and granite happily wrote 5e23 for a bar count it
    /// could not read. `is_i64()` would fail on that for the wrong reason.
    fn is_integral(v: &serde_json::Value) -> bool {
        v.as_i64().is_some() || v.as_f64().is_some_and(|n| n.fract() == 0.0)
    }

    // chat completions: response_format.json_schema.schema (nested)
    let mut body = chat_body(&uri, ASK);
    body["response_format"] = serde_json::json!({"type": "json_schema", "json_schema": {"name": "chart", "schema": schema}});
    body["max_tokens"] = serde_json::json!(200);
    let (s, j) = post(app.clone(), "/v1/chat/completions", body).await;
    assert_eq!(s, StatusCode::OK, "chat: {j}");
    let text = chat_text(&j);
    eprintln!(
        "chat json_schema: {text:?} (finish {})",
        j["choices"][0]["finish_reason"]
    );
    assert_ne!(
        j["choices"][0]["finish_reason"], "length",
        "ran out of budget: {text:?}"
    );
    let v: serde_json::Value =
        serde_json::from_str(&text).unwrap_or_else(|e| panic!("not JSON ({e}): {text:?}"));
    assert!(
        v["chart_type"].is_string(),
        "chart_type missing/wrong type: {v}"
    );
    assert!(
        is_integral(&v["bar_count"]),
        "bar_count missing/wrong type: {v}"
    );

    // responses: text.format, schema FLAT under format. A length-truncated
    // response reports status "incomplete" here rather than a fake "completed".
    let mut body = responses_body(&uri, ASK);
    body["text"] = serde_json::json!({"format": {"type": "json_schema", "schema": schema}});
    body["max_output_tokens"] = serde_json::json!(200);
    let (s, j) = post(app.clone(), "/v1/responses", body).await;
    assert_eq!(s, StatusCode::OK, "responses: {j}");
    let text = responses_text(&j);
    eprintln!("responses text.format: {text:?} (status {})", j["status"]);
    assert_eq!(j["status"], "completed", "ran out of budget: {j}");
    let v: serde_json::Value =
        serde_json::from_str(&text).unwrap_or_else(|e| panic!("not JSON ({e}): {text:?}"));
    assert!(
        v["chart_type"].is_string() && is_integral(&v["bar_count"]),
        "schema not honored: {v}"
    );

    // json_object: valid JSON, no shape imposed.
    //
    // Not paired with `<tables_json>`, and that is a finding worth keeping:
    // the tag's instruction asks for a full cell dump ({row, col, colspan,
    // rowspan, type, content} per cell) and `json_object` imposes no bound, so
    // on a chart the model invents a table and runs past any budget - measured
    // at finish_reason "length" with 500 tokens, mid-cell. `json_object` + an
    // open-ended extraction tag is an unbounded pairing; if you want the tag,
    // give it a `json_schema` that closes the shape, or read the plain text.
    let mut body = chat_body(&uri, ASK);
    body["response_format"] = serde_json::json!({"type": "json_object"});
    body["max_tokens"] = serde_json::json!(500);
    let (s, j) = post(app, "/v1/chat/completions", body).await;
    assert_eq!(s, StatusCode::OK, "chat json_object: {j}");
    let text = chat_text(&j);
    eprintln!(
        "chat json_object: {text:?} (finish {})",
        j["choices"][0]["finish_reason"]
    );
    assert_ne!(
        j["choices"][0]["finish_reason"], "length",
        "ran out of budget: {text:?}"
    );
    serde_json::from_str::<serde_json::Value>(&text)
        .unwrap_or_else(|e| panic!("not JSON ({e}): {text:?}"));
    // Granite is Dialect::JsonToolCall, so under an ANY-JSON grammar it drifts
    // into its own tool-call syntax - this leg has come back as
    // `[{"name": "dummy", "arguments": {...}}]`. That must stay CONTENT: the
    //  gate (no tools declared => no tool_calls) is the only thing
    // keeping the parser off it, and if that regressed the user's JSON would
    // silently move to `tool_calls` with content null.
    assert!(
        j["choices"][0]["message"]["tool_calls"].is_null(),
        "json_object output was parsed as a tool call: {j}"
    );
}

/// The `arguments` schema every forced-call leg below shares. `bar_count` is
/// an integer and `chart_type` an enum, so the assertions can check that the
/// grammar constrained VALUES and not merely the call's shape.
fn chart_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "chart_type": {"type": "string", "enum": ["bar", "line", "pie"]},
            "bar_count": {"type": "integer"},
            "title": {"type": "string"}
        },
        "required": ["chart_type", "bar_count"]
    })
}

/// A forced tool call parses, names the tool, and honors its schema. Shared by
/// the three surfaces since only the wire shape differs.
fn assert_chart_call(name: &str, arguments: &serde_json::Value, where_: &str) {
    assert_eq!(name, "chart", "{where_}: wrong tool");
    let obj = arguments
        .as_object()
        .unwrap_or_else(|| panic!("{where_}: args not an object: {arguments}"));
    let kind = obj["chart_type"]
        .as_str()
        .unwrap_or_else(|| panic!("{where_}: no chart_type: {arguments}"));
    assert!(
        ["bar", "line", "pie"].contains(&kind),
        "{where_}: enum escaped: {kind:?}"
    );
    let n = &obj["bar_count"];
    assert!(
        n.as_i64().is_some() || n.as_f64().is_some_and(|f| f.fract() == 0.0),
        "{where_}: bar_count is not integral: {arguments}"
    );
    // `title` is optional, so it may be absent - but if present it is a string
    assert!(
        obj.get("title").is_none_or(serde_json::Value::is_string),
        "{where_}: title present but not a string: {arguments}"
    );
    assert!(
        obj.keys()
            .all(|k| ["chart_type", "bar_count", "title"].contains(&k.as_str())),
        "{where_}: undeclared argument: {arguments}"
    );
}

/// The HOLE this TASK closed: a forced tool call is the only
/// structured-output mechanism the Anthropic API defines - it has no
/// `response_format` - so while `tool_choice` was gated on `Dialect::QwenXml`,
/// `/v1/messages` had no way at all to get schema-shaped output out of granite.
/// That landed hardest on the one model whose whole purpose is structured
/// extraction.
///
/// The JSON tool-call grammar closes it, and because granite's arguments are
/// real JSON (not qwen's free-text parameter values) the argument object runs
/// through the schema machine - so this is genuine constrained decoding, not
/// just a forced opener. All three surfaces, one picture, one tool.
#[tokio::test]
async fn a_forced_tool_call_is_schema_shaped_on_every_surface() {
    let Some(app) = app() else { return };
    let uri = data_uri();
    const ASK: &str = "Record this chart.";

    // Anthropic: tools are FLAT with `input_schema`, and the forced call comes
    // back as a tool_use block with `stop_reason: "tool_use"`
    let mut body = messages_body(&uri, ASK);
    body["tools"] = serde_json::json!([{
        "name": "chart", "description": "record the chart", "input_schema": chart_schema()
    }]);
    body["tool_choice"] = serde_json::json!({"type": "any"});
    let (s, j) = post(app.clone(), "/v1/messages", body).await;
    assert_eq!(s, StatusCode::OK, "messages: {j}");
    eprintln!("messages tool_use: {}", j["content"]);
    assert_eq!(j["stop_reason"], "tool_use", "{j}");
    let block = j["content"]
        .as_array()
        .and_then(|bs| bs.iter().find(|b| b["type"] == "tool_use"))
        .unwrap_or_else(|| panic!("no tool_use block: {j}"));
    assert_chart_call(
        block["name"].as_str().unwrap_or_default(),
        &block["input"],
        "messages",
    );

    // chat completions: nested tool shape, `tool_choice: "required"`, and the
    // arguments arrive as a JSON-encoded STRING per the OpenAI wire shape
    let mut body = chat_body(&uri, ASK);
    body["tools"] = serde_json::json!([{
        "type": "function",
        "function": {"name": "chart", "parameters": chart_schema()}
    }]);
    body["tool_choice"] = serde_json::json!("required");
    let (s, j) = post(app.clone(), "/v1/chat/completions", body).await;
    assert_eq!(s, StatusCode::OK, "chat: {j}");
    let call = &j["choices"][0]["message"]["tool_calls"][0];
    eprintln!("chat tool_calls: {call}");
    assert_eq!(j["choices"][0]["finish_reason"], "tool_calls", "{j}");
    let args: serde_json::Value =
        serde_json::from_str(call["function"]["arguments"].as_str().unwrap_or_default())
            .unwrap_or_else(|e| panic!("chat: arguments not JSON ({e}): {call}"));
    assert_chart_call(
        call["function"]["name"].as_str().unwrap_or_default(),
        &args,
        "chat",
    );

    // responses: FLAT tools, and a function_call output item
    let mut body = responses_body(&uri, ASK);
    body["tools"] = serde_json::json!([{
        "type": "function", "name": "chart", "parameters": chart_schema()
    }]);
    body["tool_choice"] = serde_json::json!("required");
    let (s, j) = post(app.clone(), "/v1/responses", body).await;
    assert_eq!(s, StatusCode::OK, "responses: {j}");
    let item = j["output"]
        .as_array()
        .and_then(|is| is.iter().find(|i| i["type"] == "function_call"))
        .unwrap_or_else(|| panic!("no function_call item: {j}"));
    eprintln!("responses function_call: {item}");
    let args: serde_json::Value =
        serde_json::from_str(item["arguments"].as_str().unwrap_or_default())
            .unwrap_or_else(|e| panic!("responses: arguments not JSON ({e}): {item}"));
    assert_chart_call(
        item["name"].as_str().unwrap_or_default(),
        &args,
        "responses",
    );

    // and the NAMED form picks that tool out of several - the grammar's
    // candidate set is filtered at compile, so `other` is unreachable
    let mut body = messages_body(&uri, ASK);
    body["tools"] = serde_json::json!([
        {"name": "other", "input_schema": {"type": "object", "properties": {}}},
        {"name": "chart", "description": "record the chart", "input_schema": chart_schema()},
    ]);
    body["tool_choice"] = serde_json::json!({"type": "tool", "name": "chart"});
    let (s, j) = post(app, "/v1/messages", body).await;
    assert_eq!(s, StatusCode::OK, "messages named: {j}");
    let block = j["content"]
        .as_array()
        .and_then(|bs| bs.iter().find(|b| b["type"] == "tool_use"))
        .unwrap_or_else(|| panic!("no tool_use block: {j}"));
    assert_chart_call(
        block["name"].as_str().unwrap_or_default(),
        &block["input"],
        "messages named",
    );
}
