//! End-to-end HTTP tests of /v1/chat/completions with Qwen3.5-9B on GPU:
//! ChatML rendering, <think> reasoning (opt-in via chat_template_kwargs),
//! XML tool-call parsing, tool round trips, stop strings, streaming deltas.
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
        eprintln!("set PADDOCK_HEAVY_TESTS=1 to run the qwen chat test");
        return None;
    }
    let model_path = std::env::var("QWEN35_GGUF")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            std::path::PathBuf::from("C:/dev/models/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q8_0.gguf")
        });
    let pack = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../packs/cuda/build/pd-cuda-sm86.dll");
    if !model_path.exists() || !pack.exists() {
        eprintln!("model or pack missing - skipping");
        return None;
    }
    let model = serving::load(
        "qwen35-9b".into(),
        &model_path,
        "cuda",
        0,
        Some(&pack),
        2048,
        4,
        None,
        None,
        None,
        None,
    )
    .ok()?;
    Some(router(Arc::new(AppState::for_tests(Some(model)))))
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

async fn post_ok(app: axum::Router, body: serde_json::Value) -> serde_json::Value {
    let (status, json) = post_raw(app, body).await;
    assert_eq!(status, StatusCode::OK, "body: {json}");
    json
}

fn weather_tools() -> serde_json::Value {
    serde_json::json!([{"type":"function","function":{
        "name":"get_weather",
        "description":"Get the current weather for a city",
        "parameters":{"type":"object","properties":{
            "city":{"type":"string","description":"City name"}
        },"required":["city"]}
    }}])
}

#[tokio::test]
async fn plain_thinking_and_stop_strings() {
    let Some(app) = app() else { return };

    // default = non-thinking: content only
    let json = post_ok(
        app.clone(),
        serde_json::json!({
            "model": "qwen35-9b",
            "messages": [{"role":"user","content":"What is the capital of France? Answer with one word."}],
            "max_tokens": 60, "temperature": 0.0
        }),
    )
    .await;
    let msg = &json["choices"][0]["message"];
    eprintln!(
        "plain: content={:?} reasoning={:?}",
        msg["content"], msg["reasoning_content"]
    );
    assert!(msg["content"].as_str().unwrap().contains("Paris"));
    assert!(
        msg["reasoning_content"].is_null(),
        "non-thinking mode must not reason"
    );
    assert_eq!(json["choices"][0]["finish_reason"], "stop");

    // enable_thinking: reasoning_content captured, separate from content
    let json = post_ok(
        app.clone(),
        serde_json::json!({
            "model": "qwen35-9b",
            "messages": [{"role":"user","content":"What is 17 + 25? Answer with just the number."}],
            "max_tokens": 500, "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": true}
        }),
    )
    .await;
    let msg = &json["choices"][0]["message"];
    eprintln!(
        "thinking: content={:?} reasoning={:?}",
        msg["content"], msg["reasoning_content"]
    );
    assert!(msg["reasoning_content"].is_string(), "think block captured");
    assert!(msg["content"].as_str().unwrap().contains("42"));
    assert!(
        !msg["content"].as_str().unwrap().contains("</think>"),
        "think markers must not leak into content"
    );

    // stop strings honored in chat
    let json = post_ok(
        app,
        serde_json::json!({
            "model": "qwen35-9b",
            "messages": [{"role":"user","content":"Count from one to ten in English words, lowercase, separated by spaces. Nothing else."}],
            "max_tokens": 80, "temperature": 0.0,
            "stop": ["five"]
        }),
    )
    .await;
    let content = json["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("");
    eprintln!(
        "stop: content={content:?} finish={:?}",
        json["choices"][0]["finish_reason"]
    );
    assert!(!content.contains("five"), "stop string leaked: {content:?}");
    assert_eq!(json["choices"][0]["finish_reason"], "stop");
}

/// S6: a second turn re-sends the rendered history verbatim, so its prefill
/// must resume from the prefix cache's DeltaNet checkpoint - visible as
/// usage.prompt_tokens_details.cached_tokens - and still answer correctly.
#[tokio::test]
async fn multi_turn_reuses_prefix_cache() {
    let Some(app) = app() else { return };
    let system = "You are a concise reference assistant for a European travel agency. \
                  Answer factual questions with a single word or a very short phrase, \
                  never a full sentence, and never add commentary or caveats.";
    let q1 = "What is the capital of France?";

    let first = post_ok(
        app.clone(),
        serde_json::json!({
            "model": "qwen35-9b",
            "messages": [{"role":"system","content":system},{"role":"user","content":q1}],
            "max_tokens": 60, "temperature": 0.0
        }),
    )
    .await;
    let a1 = first["choices"][0]["message"]["content"]
        .as_str()
        .unwrap()
        .to_owned();
    assert!(a1.contains("Paris"), "{a1:?}");
    let cached1 = first["usage"]["prompt_tokens_details"]["cached_tokens"]
        .as_u64()
        .unwrap();
    assert_eq!(cached1, 0, "fresh engine cannot have cache hits");

    let second = post_ok(
        app,
        serde_json::json!({
            "model": "qwen35-9b",
            "messages": [
                {"role":"system","content":system},
                {"role":"user","content":q1},
                {"role":"assistant","content":a1},
                {"role":"user","content":"And of Italy?"}
            ],
            "max_tokens": 60, "temperature": 0.0
        }),
    )
    .await;
    let content = second["choices"][0]["message"]["content"].as_str().unwrap();
    let cached2 = second["usage"]["prompt_tokens_details"]["cached_tokens"]
        .as_u64()
        .unwrap();
    let prompt2 = second["usage"]["prompt_tokens"].as_u64().unwrap();
    eprintln!("multi-turn: turn2 cached {cached2}/{prompt2}, answer {content:?}");
    assert!(content.contains("Rome"), "{content:?}");
    assert!(
        cached2 >= 32,
        "turn 2 must reuse the shared history prefix (got {cached2})"
    );
    assert!(cached2 < prompt2, "reuse cannot cover the whole prompt");
}

#[tokio::test]
async fn tool_call_round_trip_and_tool_choice() {
    let Some(app) = app() else { return };

    // 1) the model calls the tool
    let json = post_ok(
        app.clone(),
        serde_json::json!({
            "model": "qwen35-9b",
            "messages": [{"role":"user","content":"What's the weather in Paris right now? Use the tool."}],
            "tools": weather_tools(),
            "max_tokens": 300, "temperature": 0.0
        }),
    )
    .await;
    let choice = &json["choices"][0];
    eprintln!(
        "tool call: {}",
        serde_json::to_string_pretty(choice).unwrap()
    );
    assert_eq!(choice["finish_reason"], "tool_calls");
    let calls = choice["message"]["tool_calls"].as_array().unwrap();
    assert!(!calls.is_empty());
    assert_eq!(calls[0]["function"]["name"], "get_weather");
    let args: serde_json::Value =
        serde_json::from_str(calls[0]["function"]["arguments"].as_str().unwrap()).unwrap();
    assert_eq!(args["city"].as_str().unwrap(), "Paris");

    // 2) feed the result back (OpenAI wire shape: arguments as a STRING -
    //    exercises normalize_messages) and get a final answer
    let json = post_ok(
        app.clone(),
        serde_json::json!({
            "model": "qwen35-9b",
            "messages": [
                {"role":"user","content":"What's the weather in Paris right now? Use the tool."},
                {"role":"assistant","content":null,"tool_calls":[
                    {"id":"call_1","type":"function","function":{
                        "name":"get_weather","arguments":"{\"city\":\"Paris\"}"}}]},
                {"role":"tool","tool_call_id":"call_1","content":"18C and sunny"}
            ],
            "tools": weather_tools(),
            "max_tokens": 120, "temperature": 0.0
        }),
    )
    .await;
    let msg = &json["choices"][0]["message"];
    eprintln!("round trip: content={:?}", msg["content"]);
    let content = msg["content"].as_str().unwrap();
    assert!(content.contains("18"), "tool result not used: {content:?}");
    assert_eq!(json["choices"][0]["finish_reason"], "stop");

    // 3) tool_choice "none": tools hidden, no call possible (the model may
    //    ramble to the token cap explaining it has no live data - the
    //    invariant is that no tool call comes back, not how it finishes)
    let json = post_ok(
        app.clone(),
        serde_json::json!({
            "model": "qwen35-9b",
            "messages": [{"role":"user","content":"What's the weather in Paris? Use the tool."}],
            "tools": weather_tools(),
            "tool_choice": "none",
            "max_tokens": 100, "temperature": 0.0
        }),
    )
    .await;
    eprintln!(
        "tool_choice none: finish={:?}",
        json["choices"][0]["finish_reason"]
    );
    assert_ne!(json["choices"][0]["finish_reason"], "tool_calls");
    assert!(
        json["choices"][0]["message"]["tool_calls"]
            .as_array()
            .is_none_or(|c| c.is_empty())
    );

    // 4) tool_choice "required": the grammar FORCES a call even when the
    //    prompt would never trigger one naturally
    let json = post_ok(
        app.clone(),
        serde_json::json!({
            "model": "qwen35-9b",
            "messages": [{"role":"user","content":"Say hello."}],
            "tools": weather_tools(),
            "tool_choice": "required",
            "max_tokens": 120, "temperature": 0.0
        }),
    )
    .await;
    let choice = &json["choices"][0];
    eprintln!("forced: {}", serde_json::to_string(choice).unwrap());
    assert_eq!(choice["finish_reason"], "tool_calls");
    let calls = choice["message"]["tool_calls"].as_array().unwrap();
    assert_eq!(calls[0]["function"]["name"], "get_weather");
    let args: serde_json::Value =
        serde_json::from_str(calls[0]["function"]["arguments"].as_str().unwrap()).unwrap();
    assert!(args.get("city").is_some(), "required param forced: {args}");

    // 5) named function forcing
    let json = post_ok(
        app.clone(),
        serde_json::json!({
            "model": "qwen35-9b",
            "messages": [{"role":"user","content":"What's the weather in Oslo?"}],
            "tools": weather_tools(),
            "tool_choice": {"type":"function","function":{"name":"get_weather"}},
            "max_tokens": 120, "temperature": 0.0
        }),
    )
    .await;
    let calls = json["choices"][0]["message"]["tool_calls"]
        .as_array()
        .unwrap();
    assert_eq!(calls[0]["function"]["name"], "get_weather");

    // 6) image on a server with no mmproj: honest 400, no silent text-only
    let (status, json) = post_raw(
        app,
        serde_json::json!({
            "model": "qwen35-9b",
            "messages": [{"role":"user","content":[
                {"type":"text","text":"what is this?"},
                {"type":"image_url","image_url":{"url":"data:image/bmp;base64,AAAA"}}
            ]}],
            "max_tokens": 20
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "body: {json}");
    assert!(
        json["error"]["message"]
            .as_str()
            .unwrap_or("")
            .contains("mmproj"),
        "unhelpful error: {json}"
    );
}

fn city_schema() -> serde_json::Value {
    serde_json::json!({"type":"json_schema","json_schema":{"name":"city","schema":{
        "type":"object","additionalProperties":false,
        "properties":{
            "city":{"type":"string"},
            "country":{"type":"string"},
            "population_millions":{"type":"number"}
        },
        "required":["city","country","population_millions"]
    }}})
}

#[tokio::test]
async fn response_format_constrains_output() {
    let Some(app) = app() else { return };

    // json_object: any valid JSON, non-thinking. (Like OpenAI's json mode,
    // content may be partial when generation dies at max_tokens - so give it
    // room and require a clean stop.)
    let json = post_ok(
        app.clone(),
        serde_json::json!({
            "model": "qwen35-9b",
            "messages": [{"role":"user","content":"Reply with a small JSON object with exactly two fields: name and country of one famous city."}],
            "response_format": {"type":"json_object"},
            "max_tokens": 500, "temperature": 0.0
        }),
    )
    .await;
    assert_eq!(json["choices"][0]["finish_reason"], "stop");
    let content = json["choices"][0]["message"]["content"].as_str().unwrap();
    eprintln!("json_object: {content}");
    let parsed: serde_json::Value = serde_json::from_str(content).expect("valid JSON forced");
    assert!(parsed.is_object(), "expected an object: {content}");

    // json_schema: typed fields in required order
    let json = post_ok(
        app.clone(),
        serde_json::json!({
            "model": "qwen35-9b",
            "messages": [{"role":"user","content":"Describe Paris, France."}],
            "response_format": city_schema(),
            "max_tokens": 200, "temperature": 0.0
        }),
    )
    .await;
    let content = json["choices"][0]["message"]["content"].as_str().unwrap();
    eprintln!("json_schema: {content}");
    let parsed: serde_json::Value = serde_json::from_str(content).expect("schema JSON");
    assert!(parsed["city"].is_string());
    assert!(parsed["country"].is_string());
    assert!(parsed["population_millions"].is_number());

    // thinking mode: reasoning stays free, the CONTENT is schema-constrained
    // (the free phase is genuinely unconstrained, so budget for a full think)
    let json = post_ok(
        app.clone(),
        serde_json::json!({
            "model": "qwen35-9b",
            "messages": [{"role":"user","content":"Describe Paris, France. Think briefly."}],
            "response_format": city_schema(),
            "chat_template_kwargs": {"enable_thinking": true},
            "max_tokens": 1500, "temperature": 0.0
        }),
    )
    .await;
    let msg = &json["choices"][0]["message"];
    eprintln!("thinking+schema: content={:?}", msg["content"]);
    assert!(
        msg["reasoning_content"].is_string(),
        "reasoning must stay free"
    );
    let parsed: serde_json::Value =
        serde_json::from_str(msg["content"].as_str().unwrap()).expect("schema JSON after think");
    assert!(parsed["city"].is_string());

    // sampling (temperature > 0) still cannot escape the grammar
    let json = post_ok(
        app.clone(),
        serde_json::json!({
            "model": "qwen35-9b",
            "messages": [{"role":"user","content":"Describe Berlin, Germany."}],
            "response_format": city_schema(),
            "max_tokens": 200, "temperature": 0.9, "seed": 7
        }),
    )
    .await;
    let content = json["choices"][0]["message"]["content"].as_str().unwrap();
    eprintln!("sampled schema: {content}");
    let parsed: serde_json::Value = serde_json::from_str(content).expect("sampled schema JSON");
    assert!(parsed["city"].is_string());

    // unsupported schema keyword: honest 400 naming it
    let (status, json) = post_raw(
        app.clone(),
        serde_json::json!({
            "model": "qwen35-9b",
            "messages": [{"role":"user","content":"x"}],
            "response_format": {"type":"json_schema","json_schema":{"name":"bad","schema":{
                "type":"string","pattern":"^x+$"
            }}},
            "max_tokens": 20
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(
        json["error"]["message"]
            .as_str()
            .unwrap_or("")
            .contains("pattern"),
        "{json}"
    );

    // response_format + forced tool_choice is contradictory: 400
    let (status, _) = post_raw(
        app,
        serde_json::json!({
            "model": "qwen35-9b",
            "messages": [{"role":"user","content":"x"}],
            "tools": weather_tools(),
            "tool_choice": "required",
            "response_format": {"type":"json_object"},
            "max_tokens": 20
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

async fn sse_lines(app: axum::Router, body: serde_json::Value) -> Vec<serde_json::Value> {
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
    String::from_utf8_lossy(&bytes)
        .lines()
        .filter_map(|l| l.strip_prefix("data: ").map(str::to_owned))
        .filter(|d| d.trim() != "[DONE]")
        .map(|d| serde_json::from_str(&d).expect("chunk json"))
        .collect()
}

#[tokio::test]
async fn s4_choices_logprobs_usage_and_penalties() {
    let Some(app) = app() else { return };

    // n=2 sampled choices: independent seeds, both present and indexed
    let json = post_ok(
        app.clone(),
        serde_json::json!({
            "model": "qwen35-9b",
            "messages": [{"role":"user","content":"Name any one animal, one word."}],
            "n": 2, "max_tokens": 30, "temperature": 0.8, "seed": 11
        }),
    )
    .await;
    let choices = json["choices"].as_array().unwrap();
    assert_eq!(choices.len(), 2);
    assert_eq!(choices[0]["index"], 0);
    assert_eq!(choices[1]["index"], 1);
    for c in choices {
        assert!(
            c["message"]["content"]
                .as_str()
                .is_some_and(|s| !s.is_empty())
        );
    }

    // logprobs + top_logprobs shape
    let json = post_ok(
        app.clone(),
        serde_json::json!({
            "model": "qwen35-9b",
            "messages": [{"role":"user","content":"Say the word hello."}],
            "logprobs": true, "top_logprobs": 3,
            "max_tokens": 20, "temperature": 0.0
        }),
    )
    .await;
    let entries = json["choices"][0]["logprobs"]["content"]
        .as_array()
        .unwrap();
    assert!(!entries.is_empty(), "logprob entries expected");
    for e in entries {
        assert!(e["logprob"].as_f64().unwrap() <= 0.0);
        assert!(e["token"].is_string());
        assert_eq!(e["top_logprobs"].as_array().unwrap().len(), 3);
    }
    eprintln!(
        "logprobs: {} entries, first token {:?}",
        entries.len(),
        entries[0]["token"]
    );

    // penalties accepted and generation completes
    let json = post_ok(
        app.clone(),
        serde_json::json!({
            "model": "qwen35-9b",
            "messages": [{"role":"user","content":"Count from one to five in words."}],
            "presence_penalty": 0.5, "frequency_penalty": 0.3,
            "max_tokens": 60, "temperature": 0.0
        }),
    )
    .await;
    assert!(json["choices"][0]["message"]["content"].as_str().is_some());

    // validation 400s
    for bad in [
        serde_json::json!({"model":"m","messages":[{"role":"user","content":"x"}],"n":0}),
        serde_json::json!({"model":"m","messages":[{"role":"user","content":"x"}],"top_logprobs":3}),
        serde_json::json!({"model":"m","messages":[{"role":"user","content":"x"}],
                            "stream_options":{"include_usage":true}}),
    ] {
        let (status, _) = post_raw(app.clone(), bad).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }

    // n=2 streaming with include_usage: per-index deltas + terminal usage chunk
    let chunks = sse_lines(
        app.clone(),
        serde_json::json!({
            "model": "qwen35-9b",
            "messages": [{"role":"user","content":"Name any one color, one word."}],
            "n": 2, "max_tokens": 20, "temperature": 0.8, "seed": 3,
            "stream": true, "stream_options": {"include_usage": true}
        }),
    )
    .await;
    let mut content = [String::new(), String::new()];
    let mut finishes = 0;
    let mut usage_seen = false;
    for ch in &chunks {
        if let Some(u) = ch.get("usage").filter(|u| !u.is_null()) {
            assert!(ch["choices"].as_array().unwrap().is_empty());
            assert!(u["completion_tokens"].as_u64().unwrap() > 0);
            usage_seen = true;
            continue;
        }
        let c = &ch["choices"][0];
        let i = c["index"].as_u64().unwrap() as usize;
        if let Some(d) = c["delta"]["content"].as_str() {
            content[i].push_str(d);
        }
        if c["finish_reason"].as_str().is_some() {
            finishes += 1;
        }
    }
    eprintln!("n=2 stream: {:?} / {:?}", content[0], content[1]);
    assert_eq!(finishes, 2, "one finish chunk per choice");
    assert!(usage_seen, "include_usage terminal chunk missing");
    assert!(!content[0].is_empty() && !content[1].is_empty());

    // streaming tool call arrives as a delta chunk (per-call atomic), then
    // the finish chunk says tool_calls
    let chunks = sse_lines(
        app,
        serde_json::json!({
            "model": "qwen35-9b",
            "messages": [{"role":"user","content":"What's the weather in Paris? Use the tool."}],
            "tools": weather_tools(),
            "max_tokens": 200, "temperature": 0.0, "stream": true
        }),
    )
    .await;
    let call_chunk = chunks
        .iter()
        .find(|c| !c["choices"][0]["delta"]["tool_calls"].is_null());
    let call = &call_chunk.expect("tool_calls delta chunk")["choices"][0]["delta"]["tool_calls"][0];
    assert_eq!(call["function"]["name"], "get_weather");
    let args: serde_json::Value =
        serde_json::from_str(call["function"]["arguments"].as_str().unwrap()).unwrap();
    assert_eq!(args["city"], "Paris");
    let finish = chunks
        .iter()
        .filter_map(|c| c["choices"][0]["finish_reason"].as_str())
        .next_back();
    assert_eq!(finish, Some("tool_calls"));
}

#[tokio::test]
async fn streaming_reasoning_and_content_deltas() {
    let Some(app) = app() else { return };

    let res = app
        .oneshot(
            Request::post("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::json!({
                        "model": "qwen35-9b",
                        "messages": [{"role":"user","content":"What is 17 + 25? Answer with just the number."}],
                        "max_tokens": 500, "temperature": 0.0, "stream": true,
                        "chat_template_kwargs": {"enable_thinking": true}
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
    let mut finish = None;
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
        let choice = &v["choices"][0];
        if let Some(d) = choice["delta"]["reasoning_content"].as_str() {
            reasoning.push_str(d);
        }
        if let Some(d) = choice["delta"]["content"].as_str() {
            content.push_str(d);
        }
        if let Some(f) = choice["finish_reason"].as_str() {
            finish = Some(f.to_owned());
        }
    }
    eprintln!("streamed reasoning={reasoning:?}\nstreamed content={content:?} finish={finish:?}");
    assert!(done, "[DONE] terminal missing");
    assert!(!reasoning.is_empty(), "no reasoning deltas streamed");
    assert!(content.contains("42"), "content deltas wrong: {content:?}");
    assert!(
        !content.contains("</think>"),
        "think marker leaked into content stream"
    );
    assert_eq!(finish.as_deref(), Some("stop"));
}
