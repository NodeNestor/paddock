//! Parse gpt-oss Harmony assistant output into clean content, reasoning, and
//! tool calls. Input is the generated text decoded with special tokens visible
//! (the engine stops on `<|return|>`/`<|call|>`, so those terminals are absent
//! - we parse what precedes them).
//!
//! Structure (after the prompt's `<|start|>assistant`):
//!   <|channel|>analysis<|message|>REASONING<|end|>
//!   <|start|>assistant<|channel|>final<|message|>ANSWER
//! Tool call form:
//!   <|channel|>commentary to=functions.NAME <|constrain|>json<|message|>{ARGS}
//! Multiple commentary messages => parallel tool calls.

pub use crate::parsers::{Parsed, ToolCallRaw};

const CHANNEL: &str = "<|channel|>";
const MESSAGE: &str = "<|message|>";
const END: &str = "<|end|>";
const START: &str = "<|start|>";
const CALL: &str = "<|call|>";
const RETURN: &str = "<|return|>";

/// Parse a full Harmony assistant turn. `tools`: the request declared tools -
/// when false, a `commentary to=functions.X` message is user-visible preamble
/// like any untargeted commentary (a model can't call tools never offered).
pub fn parse(text: &str, tools: bool) -> Parsed {
    let mut out = Parsed::default();
    let mut content_parts: Vec<String> = Vec::new();

    // no channel markers at all -> treat the whole thing as content (robust
    // fallback for models/templates that don't emit Harmony)
    if !text.contains(CHANNEL) {
        let t = strip_terminals(text).trim();
        if !t.is_empty() {
            out.content = Some(t.to_owned());
        }
        return out;
    }

    let mut rest = text;
    while let Some(ch_at) = rest.find(CHANNEL) {
        let after_ch = &rest[ch_at + CHANNEL.len()..];
        let Some(msg_at) = after_ch.find(MESSAGE) else {
            break;
        };
        let header = after_ch[..msg_at].trim();
        let body_start = &after_ch[msg_at + MESSAGE.len()..];

        // body ends at the earliest of the message/turn delimiters
        let end = [END, CALL, RETURN, START, CHANNEL]
            .iter()
            .filter_map(|d| body_start.find(d))
            .min()
            .unwrap_or(body_start.len());
        let body = &body_start[..end];

        // a tool call is complete once its body hit a real delimiter (a
        // mid-generation parse leaves the trailing call in progress)
        let terminated = end < body_start.len();
        let calls_before = out.tool_calls.len();
        classify(header, body, tools, &mut out, &mut content_parts);
        if terminated && out.tool_calls.len() > calls_before {
            out.complete_calls = out.tool_calls.len();
        }

        rest = &body_start[end..];
    }

    if !content_parts.is_empty() {
        out.content = Some(content_parts.join(""));
    }
    out
}

fn classify(header: &str, body: &str, tools: bool, out: &mut Parsed, content: &mut Vec<String>) {
    if header.starts_with("final") {
        content.push(body.to_owned());
    } else if header.starts_with("analysis") {
        let r = body.trim();
        if !r.is_empty() {
            // multiple analysis blocks concatenate
            let acc = out.reasoning.take().unwrap_or_default();
            out.reasoning = Some(if acc.is_empty() {
                r.to_owned()
            } else {
                format!("{acc}\n{r}")
            });
        }
    } else if header.starts_with("commentary") {
        match tool_name(header) {
            Some(name) if tools => out.tool_calls.push(ToolCallRaw {
                name,
                arguments: body.trim().to_owned(),
            }),
            // commentary without a tool target - or with no tools declared in
            // the request - = user-visible preamble
            _ => content.push(body.to_owned()),
        }
    }
    // unknown channels are dropped (forward-compat)
}

/// Extract `NAME` from a `commentary to=functions.NAME ...` header.
fn tool_name(header: &str) -> Option<String> {
    let to = header.find("to=")?;
    let after = &header[to + 3..];
    // strip an optional "functions." namespace prefix
    let target = after.split([' ', '<']).next().unwrap_or(after).trim();
    let name = target.strip_prefix("functions.").unwrap_or(target);
    (!name.is_empty()).then(|| name.to_owned())
}

fn strip_terminals(text: &str) -> &str {
    for t in [RETURN, CALL, END] {
        if let Some(i) = text.find(t) {
            return &text[..i];
        }
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_final_answer() {
        let t = "<|channel|>final<|message|>Hello there!";
        let p = parse(t, true);
        assert_eq!(p.content.as_deref(), Some("Hello there!"));
        assert_eq!(p.finish_reason(), "stop");
    }

    #[test]
    fn analysis_then_final() {
        let t = "<|channel|>analysis<|message|>Let me think...<|end|>\
                 <|start|>assistant<|channel|>final<|message|>The answer is 42.";
        let p = parse(t, true);
        assert_eq!(p.reasoning.as_deref(), Some("Let me think..."));
        assert_eq!(p.content.as_deref(), Some("The answer is 42."));
    }

    #[test]
    fn single_tool_call() {
        let t = "<|channel|>analysis<|message|>need weather<|end|>\
                 <|start|>assistant<|channel|>commentary to=functions.get_weather \
                 <|constrain|>json<|message|>{\"location\":\"Paris\"}";
        let p = parse(t, true);
        assert_eq!(p.tool_calls.len(), 1);
        assert_eq!(p.tool_calls[0].name, "get_weather");
        assert_eq!(p.tool_calls[0].arguments, "{\"location\":\"Paris\"}");
        assert_eq!(p.finish_reason(), "tool_calls");
        assert!(p.content.is_none());
    }

    #[test]
    fn parallel_tool_calls() {
        let t = "<|channel|>commentary to=functions.a <|constrain|>json<|message|>{\"x\":1}<|end|>\
                 <|start|>assistant<|channel|>commentary to=functions.b <|constrain|>json<|message|>{\"y\":2}";
        let p = parse(t, true);
        assert_eq!(p.tool_calls.len(), 2);
        assert_eq!(p.tool_calls[0].name, "a");
        assert_eq!(p.tool_calls[1].name, "b");
        assert_eq!(p.tool_calls[1].arguments, "{\"y\":2}");
    }

    #[test]
    fn no_markers_is_content() {
        let p = parse("just plain text<|return|>", true);
        assert_eq!(p.content.as_deref(), Some("just plain text"));
    }
}
