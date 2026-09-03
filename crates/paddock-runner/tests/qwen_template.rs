//! Render the real Qwen3.5 chat template (byte-exact fixture from the 9B GGUF)
//! through our minijinja pipeline. The template leans on jinja2 features beyond
//! pycompat method shims - `messages[::-1]` slice-step, `loop.previtem` /
//! `loop.nextitem`, `namespace()` - so this is the go/no-go gate for qwen chat
//! serving, independent of any GPU.

use paddock_runner::chat_template;
use serde_json::json;

fn template() -> &'static str {
    include_str!("fixtures/qwen35_chat_template.jinja")
}

fn render(messages: serde_json::Value, tools: Option<serde_json::Value>) -> String {
    render_kw(messages, tools, None)
}

fn render_kw(
    messages: serde_json::Value,
    tools: Option<serde_json::Value>,
    kwargs: Option<serde_json::Value>,
) -> String {
    let msgs: Vec<serde_json::Value> = messages.as_array().unwrap().clone();
    let tools: Option<Vec<serde_json::Value>> = tools.map(|t| t.as_array().unwrap().clone());
    chat_template::render(template(), &msgs, tools.as_deref(), kwargs.as_ref()).expect("render")
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
    assert!(out.contains("<|im_start|>system\nYou are terse.<|im_end|>\n"));
    assert!(out.contains("<|im_start|>user\nHello<|im_end|>\n"));
    // The render PIPELINE injects `enable_thinking=true` when the request
    // does not set it (llama.cpp/minja parity - thinking models
    // think by default; official Qwen3-line templates agree), so a plain
    // chat ends inside an open think block. The fixture alone would default
    // non-thinking (`is defined and ... true`), but the pipeline default is
    // the served contract, and this test pins THAT. It once went stale the
    // other way - pinned to the pre-gemma4 non-thinking default - and stayed
    // red for days, so keep it aimed at the pipeline, not the fixture.
    assert!(
        out.ends_with("<|im_start|>assistant\n<think>\n"),
        "tail: {:?}",
        &out[out.len().saturating_sub(60)..]
    );
}

#[test]
fn enable_thinking_false_pre_closes_the_block() {
    let out = render_kw(
        json!([{"role": "user", "content": "Hello"}]),
        None,
        Some(json!({"enable_thinking": false})),
    );
    // the non-thinking escape hatch (chat_template_kwargs override): an
    // empty pre-closed think block, no reasoning tokens generated
    assert!(
        out.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"),
        "tail: {:?}",
        &out[out.len().saturating_sub(60)..]
    );
}

#[test]
fn enable_thinking_kwarg_opens_a_think_block() {
    let out = render_kw(
        json!([{"role": "user", "content": "Hello"}]),
        None,
        Some(json!({"enable_thinking": true})),
    );
    // thinking mode: the prompt ends inside an open think block - this is the
    // `thinking_open` signal chat.rs keys reasoning parsing on
    assert!(
        out.ends_with("<|im_start|>assistant\n<think>\n"),
        "tail: {:?}",
        &out[out.len().saturating_sub(60)..]
    );
}

#[test]
fn tools_render_in_system_block() {
    let out = render(
        json!([{"role": "user", "content": "weather in Paris?"}]),
        Some(json!([{"type": "function", "function": {
            "name": "get_weather",
            "description": "Get current weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
        }}])),
    );
    assert!(out.contains("# Tools"));
    assert!(out.contains("<tools>"));
    assert!(out.contains("\"get_weather\""));
    // the template's own calling-convention instructions
    assert!(out.contains("<function=example_function_name>"));
}

#[test]
fn tool_call_history_round_trips() {
    // assistant tool_calls with arguments as a MAPPING (chat.rs normalizes the
    // OpenAI arguments-string to this before rendering) + a tool result. This
    // path exercises loop.previtem / loop.nextitem and messages[::-1].
    let out = render(
        json!([
            {"role": "user", "content": "weather in Paris?"},
            {"role": "assistant", "content": null, "tool_calls": [
                {"id": "call_1", "type": "function", "function": {
                    "name": "get_weather", "arguments": {"city": "Paris"}
                }}
            ]},
            {"role": "tool", "tool_call_id": "call_1", "content": "18C, sunny"},
            {"role": "user", "content": "and in Berlin?"}
        ]),
        Some(json!([{"type": "function", "function": {
            "name": "get_weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
        }}])),
    );
    assert!(
        out.contains(
            "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>"
        ),
        "tool call block missing in:\n{out}"
    );
    assert!(out.contains("<tool_response>\n18C, sunny\n</tool_response>"));
}

#[test]
fn arguments_as_string_would_be_dropped() {
    // Documents why chat.rs must normalize: OpenAI sends arguments as a JSON
    // string, and the template's `is mapping` guard silently drops them.
    let out = render(
        json!([
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": null, "tool_calls": [
                {"id": "c", "type": "function", "function": {
                    "name": "get_weather", "arguments": "{\"city\":\"Paris\"}"
                }}
            ]},
            {"role": "tool", "tool_call_id": "c", "content": "18C"},
            {"role": "user", "content": "thanks"}
        ]),
        None,
    );
    assert!(out.contains("<function=get_weather>"));
    assert!(
        !out.contains("<parameter=city>"),
        "string arguments unexpectedly rendered - normalization no longer needed?"
    );
}
