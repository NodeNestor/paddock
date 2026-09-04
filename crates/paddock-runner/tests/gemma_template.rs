//! Render the real gemma-4 QAT chat template (byte-exact fixture from the
//! 26B-A4B-it-qat GGUF - a NEWER revision than gemma4_chat_template.jinja:
//! null-argument branch, strip_thinking macro) through our minijinja pipeline,
//! pinned byte-for-byte to what llama-server b10486 renders for the same
//! messages (captured from its `/apply-template`). Greedy parity needs the
//! two engines to tokenize the same prompt, so this is the template half of
//! that guarantee; the other half is the preopen kill in greedy-parity.py
//! (paddock's serving deliberately appends `<|channel>thought\n` after this
//! render - a TTFT optimization, not part of the template's output).
//!
//! Note neither engine renders `<bos>` here: the template's `{{ bos_token }}`
//! resolves empty on both sides, and each engine's TOKENIZER supplies the
//! leading BOS id (chat.rs inserts model.bos; llama.cpp tokenizes with
//! add_special).
// Test code: a failed assumption stops the test where it happened.
#![allow(clippy::unwrap_used)]

use paddock_runner::chat_template;
use serde_json::json;

fn template() -> &'static str {
    include_str!("fixtures/gemma4_qat_chat_template.jinja")
}

fn render_kw(messages: serde_json::Value, kwargs: Option<serde_json::Value>) -> String {
    let msgs: Vec<serde_json::Value> = messages.as_array().unwrap().clone();
    chat_template::render(template(), &msgs, None, kwargs.as_ref()).expect("render")
}

/// The greedy-parity probe shape: one user message, thinking at the serving
/// default (the pipeline injects `enable_thinking=true`). The template opens
/// a synthesized system turn carrying only the `<|think|>` marker.
#[test]
fn user_only_matches_llamacpp_reference() {
    let out = render_kw(json!([{"role": "user", "content": "What is 2+2?"}]), None);
    assert_eq!(
        out,
        "<|turn>system\n<|think|>\n<turn|>\n<|turn>user\nWhat is 2+2?<turn|>\n<|turn>model\n"
    );
}

/// `enable_thinking=false` (request chat_template_kwargs): with no system
/// message and no tools the system block disappears entirely, and the
/// TEMPLATE pre-closes an empty thought channel after the generation prompt
/// (its own no-think convention - chat_template.rs's enable_thinking comment
/// describes exactly this shape).
#[test]
fn thinking_off_drops_the_system_turn() {
    let out = render_kw(
        json!([{"role": "user", "content": "What is 2+2?"}]),
        Some(json!({"enable_thinking": false})),
    );
    assert_eq!(
        out,
        "<|turn>user\nWhat is 2+2?<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
    );
}

/// A real system message rides in the same turn as the thinking marker,
/// trimmed, before the turn closes.
#[test]
fn system_message_joins_the_think_turn() {
    let out = render_kw(
        json!([
            {"role": "system", "content": "Be terse."},
            {"role": "user", "content": "What is 2+2?"}
        ]),
        None,
    );
    assert_eq!(
        out,
        "<|turn>system\n<|think|>\nBe terse.<turn|>\n<|turn>user\nWhat is 2+2?<turn|>\n<|turn>model\n"
    );
}
