//! Render the real Laguna chat template (byte-exact fixture from the XS-2.1
//! GGUF) through our minijinja pipeline. The template needs three things
//! beyond what qwen exercised: the `{%- generation -%}` loss-mask tags (no
//! render semantics - neutralized before parse), `tojson(ensure_ascii=False)`
//! on history tool-call values, and `messages[1:]` slicing. This is the
//! go/no-go gate for laguna chat serving, independent of any GPU.
// Test code: a failed assumption stops the test where it happened.
#![allow(clippy::unwrap_used)]

use paddock_runner::chat_template;
use serde_json::json;

fn template() -> &'static str {
    include_str!("fixtures/laguna_chat_template.jinja")
}

fn render_kw(
    messages: serde_json::Value,
    tools: Option<serde_json::Value>,
    kwargs: Option<serde_json::Value>,
) -> String {
    let msgs: Vec<serde_json::Value> = messages.as_array().unwrap().clone();
    let msgs = chat_template::normalize_messages(&msgs);
    let tools: Option<Vec<serde_json::Value>> = tools.map(|t| t.as_array().unwrap().clone());
    chat_template::render(template(), &msgs, tools.as_deref(), kwargs.as_ref()).expect("render")
}

fn render(messages: serde_json::Value, tools: Option<serde_json::Value>) -> String {
    render_kw(messages, tools, None)
}

#[test]
fn plain_chat_defaults_to_thinking() {
    let out = render(
        json!([
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": "Hello"}
        ]),
        None,
    );
    // template emits its BOS marker as text; encode() maps it to id 2 and the
    // serving BOS guard sees it already leading
    assert!(
        out.starts_with("〈|EOS|〉"),
        "head: {:?}",
        &out[..30.min(out.len())]
    );
    assert!(out.contains("<system>You are terse.</system>\n"));
    assert!(out.contains("<user>Hello</user>\n"));
    // The render PIPELINE injects `enable_thinking=true` when the request does
    // not set it (the thinking-model default, same contract as qwen/gemma4);
    // the template's own `default(false)` only applies when UNDEFINED. So a
    // plain chat ends inside an open bare `<think>` - the laguna
    // `thinking_open` signal.
    assert!(
        out.ends_with("<assistant><think>"),
        "tail: {:?}",
        &out[out.len().saturating_sub(40)..]
    );
}

#[test]
fn enable_thinking_false_pre_closes_the_block() {
    let out = render_kw(
        json!([{"role": "user", "content": "Hello"}]),
        None,
        Some(json!({"enable_thinking": false})),
    );
    // non-thinking: the template pre-emits `</think>` - content starts
    // immediately, and thinking_open must read false on this suffix
    assert!(
        out.ends_with("<assistant></think>"),
        "tail: {:?}",
        &out[out.len().saturating_sub(40)..]
    );
}

#[test]
fn empty_system_message_opts_out_of_the_default() {
    // a caller-supplied empty system message suppresses the Poolside default
    // block entirely (the template's documented opt-out) - but only with
    // thinking off; enable_thinking forces the <system> wrapper
    let out = render_kw(
        json!([
            {"role": "system", "content": ""},
            {"role": "user", "content": "Hi"}
        ]),
        None,
        Some(json!({"enable_thinking": false})),
    );
    assert!(!out.contains("<system>"), "out: {out:?}");
    // no system message given at all -> the Poolside default renders
    let out = render_kw(
        json!([{"role": "user", "content": "Hi"}]),
        None,
        Some(json!({"enable_thinking": false})),
    );
    assert!(out.contains("<system>You are a helpful"));
}

#[test]
fn tools_render_in_the_system_block() {
    let out = render(
        json!([{"role": "user", "content": "weather?"}]),
        Some(json!([{"type": "function", "function": {
            "name": "get_weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
        }}])),
    );
    assert!(out.contains("### Tools"));
    assert!(out.contains("<available_tools>\n"));
    assert!(out.contains("\"get_weather\""));
    assert!(out.contains("</available_tools>"));
}

#[test]
fn assistant_history_rerenders_tool_calls() {
    // the agentic round-trip: an assistant tool-call turn from OpenAI wire
    // history (arguments as a JSON STRING - normalize_messages objectifies)
    // re-renders in laguna syntax, including tojson(ensure_ascii=False) on
    // the non-string value
    let out = render_kw(
        json!([
            {"role": "user", "content": "weather in Paris and depth 2"},
            {"role": "assistant", "content": null, "tool_calls": [{
                "id": "c1", "type": "function",
                "function": {"name": "get_weather",
                             "arguments": "{\"city\": \"Paris\", \"days\": 2}"}
            }]},
            {"role": "tool", "content": "sunny"}
        ]),
        None,
        Some(json!({"enable_thinking": false})),
    );
    assert!(out.contains("<assistant></think><tool_call>get_weather"));
    assert!(out.contains("<arg_key>city</arg_key><arg_value>Paris</arg_value>"));
    // non-string value goes through tojson -> stays a bare JSON number
    assert!(out.contains("<arg_key>days</arg_key><arg_value>2</arg_value>"));
    assert!(out.contains("</tool_call></assistant>\n"));
    assert!(out.contains("<tool_response>sunny</tool_response>\n"));
}

#[test]
fn assistant_history_shows_reasoning_when_thinking() {
    // enable_thinking renders prior reasoning back into history (the vLLM
    // `reasoning` field name and `reasoning_content` both work)
    let out = render(
        json!([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "Hello!", "reasoning_content": "greet back"},
            {"role": "user", "content": "bye"}
        ]),
        None,
    );
    assert!(out.contains("<assistant><think>greet back</think>Hello!</assistant>\n"));
}

#[test]
fn generation_tags_leave_no_residue() {
    // the {%- generation -%} wrappers must vanish without disturbing
    // whitespace: turns butt up against each other exactly as HF renders them
    let out = render_kw(
        json!([
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"}
        ]),
        None,
        Some(json!({"enable_thinking": false})),
    );
    assert!(out.contains("<user>a</user>\n<assistant></think>b</assistant>\n<user>c</user>\n"));
    assert!(!out.contains("generation"), "tag residue in: {out:?}");
}
