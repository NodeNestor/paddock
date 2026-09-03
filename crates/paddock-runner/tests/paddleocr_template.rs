//! Render PaddleOCR-VL's real chat template (the string its official GGUF
//! carries in `tokenizer.chat_template` - the serving default, no override)
//! through our minijinja pipeline, and compare byte-for-byte against the
//! checkpoint processor's own `apply_chat_template` output for the six task
//! prompts plus text-only and multi-turn shapes
//! (our OCR oracle tool).
//!
//! The template is ERNIE-shaped: `<|begin_of_sentence|>` emitted by the
//! template itself (add_bos stays false), `User: `/`Assistant:\n` role
//! envelopes, `</s>` closing assistant turns, and per image part the exact
//! `<|IMAGE_START|><|IMAGE_PLACEHOLDER|><|IMAGE_END|>` slot the engine
//! splices at. Images render before the text of the same message regardless
//! of part order - the reference template iterates image parts first.
//!
//! Skips cleanly when the model or fixtures are absent.

use paddock_runner::chat_template;
use serde_json::json;

fn fixtures() -> Option<serde_json::Value> {
    let path = std::env::var("PADDLEOCR_VL_FIXTURES")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            std::path::PathBuf::from("/models/ocr-battery/paddle-oracle/template_fixtures.json")
        });
    let text = std::fs::read_to_string(path).ok()?;
    Some(serde_json::from_str(&text).expect("fixtures json"))
}

fn render(template: &str, messages: serde_json::Value) -> String {
    let msgs: Vec<serde_json::Value> = messages.as_array().unwrap().clone();
    let msgs = chat_template::normalize_messages(&msgs);
    chat_template::render(template, &msgs, None, None).expect("render")
}

/// The six task prompts, sent the way an OpenAI client (and the official
/// paddleocr client) actually sends them: an `image_url` data-URI part plus
/// the task text.
#[test]
fn task_prompts_render_byte_identical_to_the_processor() {
    let template = chat_template::PADDLEOCR_VL_TEMPLATE;
    let Some(fix) = fixtures() else {
        eprintln!("fixtures missing - skipping");
        return;
    };
    for (task, f) in fix["tasks"].as_object().unwrap() {
        let messages = json!([{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}},
                {"type": "text", "text": task}
            ]
        }]);
        let got = render(template, messages);
        assert_eq!(
            got,
            f["rendered"].as_str().unwrap(),
            "render diverged for task {task:?}"
        );
    }
}

/// Text-only and multi-turn shapes: our all-text flatten routes through the
/// GGUF template's string-content branches, which must emit the same bytes
/// the reference's list-content walk does.
#[test]
fn text_shapes_render_byte_identical_to_the_processor() {
    let template = chat_template::PADDLEOCR_VL_TEMPLATE;
    let Some(fix) = fixtures() else {
        eprintln!("fixtures missing - skipping");
        return;
    };
    let got = render(template, json!([{ "role": "user", "content": "Hej!" }]));
    assert_eq!(
        got,
        fix["text_only"]["rendered"].as_str().unwrap(),
        "text-only diverged"
    );

    let got = render(
        template,
        json!([
            {"role": "system", "content": "Svara kort."},
            {"role": "user", "content": "Hej!"},
            {"role": "assistant", "content": "Hej på dig!"},
            {"role": "user", "content": "Vad heter du?"}
        ]),
    );
    assert_eq!(
        got,
        fix["multi_turn"]["rendered"].as_str().unwrap(),
        "multi-turn diverged"
    );
}
