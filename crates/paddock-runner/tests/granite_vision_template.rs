//! Render the real granite-vision-4.1-4b chat template through our minijinja
//! pipeline. The fixture is byte-identical to the string
//! `granite-vision-4.1-4b-Q8_0.gguf` carries in `tokenizer.chat_template`
//! (verified by searching the GGUF header for the file verbatim).
//!
//! This template is the model's INTERFACE, not decoration. It carries six task
//! tags - `<chart2csv>`, `<chart2code>`, `<chart2summary>`, `<tables_json>`,
//! `<tables_html>`, `<tables_otsl>` - that expand into long, exact instruction
//! prompts, and the model was tuned on those exact strings. A tag that reaches
//! the model unexpanded, or expanded with a byte out of place, is a silently
//! worse answer rather than an error, so the expansions are asserted verbatim
//! here.
//!
//! It also needs two things no template we serve has needed before:
//!
//! - `text.index("<image>")` - Python's `str.index`, used to decide whether the
//!   image goes before or after the expanded instruction. minijinja only has it
//!   through `minijinja_contrib::pycompat`, which the render pipeline installs.
//! - `<image>` as the placeholder, emitted by `render_content` for content
//!   parts of type `image`. OpenAI clients send `image_url`, so
//!   `normalize_messages` has to rewrite the part first - the same seam gemma4
//!   uses - or the slot count comes out zero and the pixels are dropped.
// Test code: a failed assumption stops the test where it happened.
#![allow(clippy::unwrap_used)]

use paddock_runner::chat_template;
use serde_json::json;

fn template() -> &'static str {
    include_str!("fixtures/granite_vision_chat_template.jinja")
}

fn render(messages: serde_json::Value, tools: Option<serde_json::Value>) -> String {
    let msgs: Vec<serde_json::Value> = messages.as_array().unwrap().clone();
    let msgs = chat_template::normalize_messages(&msgs);
    let tools: Option<Vec<serde_json::Value>> = tools.map(|t| t.as_array().unwrap().clone());
    chat_template::render(template(), &msgs, tools.as_deref(), None).expect("render")
}

/// Content in the shape an OpenAI client actually sends: an `image_url` part
/// with an inline data URI, plus text.
fn image_then(text: &str) -> serde_json::Value {
    json!([{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}},
            {"type": "text", "text": text}
        ]
    }])
}

#[test]
fn plain_chat_renders_with_granite_role_markers() {
    let out = render(json!([{"role": "user", "content": "Hello"}]), None);
    assert!(
        out.starts_with("<|start_of_role|>system<|end_of_role|>You are a helpful assistant."),
        "head: {:?}",
        &out[..80.min(out.len())]
    );
    assert!(out.contains("<|start_of_role|>user<|end_of_role|>"));
    assert!(
        out.ends_with("<|start_of_role|>assistant<|end_of_role|>"),
        "tail: {:?}",
        &out[out.len().saturating_sub(50)..]
    );
}

#[test]
fn an_explicit_system_message_replaces_the_default() {
    let out = render(
        json!([
            {"role": "system", "content": "Answer in CSV only."},
            {"role": "user", "content": "Hi"}
        ]),
        None,
    );
    assert!(
        out.contains("<|start_of_role|>system<|end_of_role|>Answer in CSV only.<|end_of_text|>")
    );
    assert!(!out.contains("You are a helpful assistant. Please ensure"));
}

/// The whole point of the `normalize_messages` seam: an OpenAI `image_url`
/// part must produce exactly one `<image>` placeholder, because
/// `build_mm_chunks` splices the encoded rows at those slots and errors when
/// the counts disagree.
#[test]
fn an_openai_image_part_renders_exactly_one_placeholder() {
    let out = render(image_then("What is this?"), None);
    assert_eq!(out.matches("<image>").count(), 1, "rendered:\n{out}");
    assert!(out.contains("<image>\nWhat is this?"), "rendered:\n{out}");
}

/// The same request over /v1/responses. This surface does not convert its
/// content parts at all - `prepare` hands the array to `normalize_messages`
/// verbatim - so the Responses spellings (`input_image`, `input_text`) are
/// what actually reach this template, and they are a different payload from
/// the chat one. That has already broken here once: granite's `render_content`
/// tests the type STRING, so every `input_text` part was skipped and an image
/// sent with a question arrived with no question, silently. Templates that key
/// off `'text' in item` (qwen) never noticed, which is exactly why granite
/// needs its own case.
///
/// The assertion is equivalence with the chat shape, not just "renders": the
/// two surfaces must put the same prompt in front of the model.
#[test]
fn a_responses_image_item_renders_the_same_prompt_as_the_chat_shape() {
    let responses = json!([{
        "role": "user",
        "content": [
            {"type": "input_image", "image_url": "data:image/png;base64,iVBORw0KGgo="},
            {"type": "input_text", "text": "What is this?"}
        ]
    }]);
    let out = render(responses, None);
    assert_eq!(out.matches("<image>").count(), 1, "rendered:\n{out}");
    assert!(
        out.contains("<image>\nWhat is this?"),
        "question dropped; rendered:\n{out}"
    );
    assert_eq!(
        out,
        render(image_then("What is this?"), None),
        "surfaces diverged"
    );
}

/// Task tags are the model's real interface and they travel as ORDINARY
/// MESSAGE TEXT, so a surface that mishandles text parts does not error - it
/// quietly turns a tuned extraction prompt into a bare picture. Assert the
/// expansion survives the Responses spelling too.
#[test]
fn a_task_tag_still_expands_through_the_responses_spelling() {
    let responses = json!([{
        "role": "user",
        "content": [
            {"type": "input_image", "image_url": "data:image/png;base64,iVBORw0KGgo="},
            {"type": "input_text", "text": "<chart2csv>"}
        ]
    }]);
    let out = render(responses, None);
    assert!(
        out.contains("extract the data into a CSV table"),
        "tag did not expand:\n{out}"
    );
    assert!(
        !out.contains("<chart2csv>"),
        "tag survived into the prompt:\n{out}"
    );
    assert_eq!(
        out,
        render(image_then("<chart2csv>"), None),
        "surfaces diverged"
    );
}

#[test]
fn two_images_render_two_placeholders() {
    let msgs = json!([{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,BB=="}},
            {"type": "text", "text": "Compare these."}
        ]
    }]);
    let out = render(msgs, None);
    assert_eq!(out.matches("<image>").count(), 2, "rendered:\n{out}");
}

/// Every task tag expands to its exact instruction and the tag itself
/// disappears. The expansions are transcribed from the template's own
/// constants; if upstream ever edits one, this test is what catches it.
#[test]
fn every_task_tag_expands_verbatim() {
    const CASES: &[(&str, &str)] = &[
        (
            "<chart2code>",
            "Generate code that recreates the chart as best as possible.",
        ),
        ("<chart2summary>", "Can you describe this chart image?"),
        (
            "<chart2csv>",
            "Please examine this chart image. Consider you are a data visualization expert, \
             and extract the data into a CSV table.\n\nYour CSV should:\n- Include a header row \
             with clear column names\n- Represent all data series/categories shown in the chart\n\
             - Use numeric values that match the chart as closely as possible\n\nOutput only the \
             CSV data, nothing else.",
        ),
    ];
    for &(tag, want) in CASES {
        let out = render(image_then(tag), None);
        assert!(
            out.contains(want),
            "{tag} did not expand verbatim; got:\n{out}"
        );
        assert!(
            !out.contains(tag),
            "{tag} survived into the prompt; got:\n{out}"
        );
    }
    // the three table tags all share a prefix and differ in their output spec
    for (tag, marker) in [
        (
            "<tables_json>",
            "The output must be a valid JSON object containing a list of dictionaries",
        ),
        (
            "<tables_html>",
            "The output must be a list of valid HTML tables",
        ),
        ("<tables_otsl>", "<fcel> - a cell with content in it"),
    ] {
        let out = render(image_then(tag), None);
        assert!(
            out.contains("Identify and extract the table schema"),
            "{tag} did not expand; got:\n{out}"
        );
        assert!(
            out.contains(marker),
            "{tag} expanded to the wrong spec; got:\n{out}"
        );
        assert!(!out.contains(tag), "{tag} survived into the prompt");
    }
}

/// The tags a client is TOLD about, read back out of this very template. All
/// six, in template order, and `<image>` - which the dispatcher tests for the
/// same way but renders straight through - must not be among them.
///
/// This is what puts the six extraction tasks in the Studio's composer, so a
/// tag going missing here is a feature quietly disappearing from the UI.
#[test]
fn all_six_task_tags_are_discoverable_from_the_template() {
    let tags = chat_template::task_tags(template());
    let names: Vec<&str> = tags.iter().map(|t| t.tag.as_str()).collect();
    assert_eq!(
        names,
        [
            "<chart2code>",
            "<chart2csv>",
            "<chart2summary>",
            "<tables_json>",
            "<tables_html>",
            "<tables_otsl>",
        ],
        "discovered: {names:?}"
    );
    // the advertised prompt is the real expansion, byte for byte
    let by_tag = |t: &str| tags.iter().find(|x| x.tag == t).expect(t).prompt.clone();
    assert_eq!(
        by_tag("<chart2code>"),
        "Generate code that recreates the chart as best as possible."
    );
    assert_eq!(
        by_tag("<chart2summary>"),
        "Can you describe this chart image?"
    );
    assert!(by_tag("<chart2csv>").starts_with("Please examine this chart image."));
    assert!(by_tag("<tables_otsl>").contains("<fcel> - a cell with content in it"));
}

/// The `str.index` path: the image goes before the expanded instruction when
/// the `<image>` placeholder came first, and after when the tag did. Without
/// pycompat's `index` this render throws rather than mis-ordering, but assert
/// the ordering itself so a silent fallback would still be caught.
#[test]
fn image_position_follows_the_tag_position() {
    const SUMMARY: &str = "Can you describe this chart image?";

    // image part first => "<image>\n" prefix
    let out = render(image_then("<chart2summary>"), None);
    let (img, tag) = (out.find("<image>").unwrap(), out.find(SUMMARY).unwrap());
    assert!(img < tag, "image should lead here; got:\n{out}");

    // tag first, image second => suffix
    let msgs = json!([{
        "role": "user",
        "content": [
            {"type": "text", "text": "<chart2summary>"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}
        ]
    }]);
    let out = render(msgs, None);
    let (img, tag) = (out.find("<image>").unwrap(), out.find(SUMMARY).unwrap());
    assert!(tag < img, "instruction should lead here; got:\n{out}");
}

/// A tag-free message passes through untouched - `expand_tags` must not eat
/// ordinary prompts, and must not inject an instruction nobody asked for.
#[test]
fn untagged_text_passes_through_unchanged() {
    let out = render(image_then("How many bars are in the chart?"), None);
    assert!(
        out.contains("<image>\nHow many bars are in the chart?"),
        "rendered:\n{out}"
    );
    assert!(!out.contains("data visualization expert"));
    assert!(!out.contains("Identify and extract the table schema"));
}

/// A tag with no image still expands: the model can be asked for the CSV
/// instruction over a document supplied as text.
#[test]
fn a_tag_without_an_image_still_expands() {
    let out = render(
        json!([{"role": "user", "content": "<chart2summary>"}]),
        None,
    );
    assert!(
        out.contains("Can you describe this chart image?"),
        "rendered:\n{out}"
    );
    assert!(
        !out.contains("<image>"),
        "no image was sent; rendered:\n{out}"
    );
}

#[test]
fn tools_land_in_the_system_message_with_the_tool_call_contract() {
    let tools = json!([{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Look up weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
        }
    }]);
    let out = render(
        json!([{"role": "user", "content": "Weather in Malmo?"}]),
        Some(tools),
    );
    assert!(out.contains("<tools>"), "rendered:\n{out}");
    assert!(out.contains("get_weather"));
    assert!(out.contains("<tool_call>"));
}

/// Assistant tool calls re-render from history in the template's own shape -
/// the round-trip a multi-turn agent depends on.
#[test]
fn assistant_tool_calls_round_trip_from_history() {
    let out = render(
        json!([
            {"role": "user", "content": "Weather?"},
            {"role": "assistant", "content": "", "tool_calls": [{
                "id": "c1", "type": "function",
                "function": {"name": "get_weather", "arguments": "{\"city\": \"Malmo\"}"}
            }]},
            {"role": "tool", "content": "12C"}
        ]),
        None,
    );
    assert!(
        out.contains("<tool_call>\n{\"name\": \"get_weather\", \"arguments\": "),
        "rendered:\n{out}"
    );
    // the leading whitespace is the template's own `render_content` macro
    // indentation, which jinja2 emits identically - matched, not trimmed
    assert!(out.contains("<tool_response>"), "rendered:\n{out}");
    assert!(out.contains("12C\n</tool_response>"), "rendered:\n{out}");
}
