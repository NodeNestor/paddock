//! Parse Muse Glimmer assistant output into content, reasoning and tool calls.
//!
//! The turn is a SEQUENCE of MESSAGES, each addressed to a recipient - the
//! shape is Harmony-ish but the channel is spelled as an address, not a name:
//!
//! ```text
//! <|start|>assistant to=self<|message|>THOUGHT<|eom|>
//! <|start|>assistant to=user<|message|>ANSWER<|eot|>
//! ```
//!
//! `self` is the reasoning channel, `user` (or no recipient at all - the
//! template makes ` to=user` optional) is the answer, and any other recipient
//! is a tool call, whose body is Anthropic-style ATEM markup:
//!
//! ```text
//! <|start|>assistant to=weather.get<|message|><atem:function_calls>
//! <atem:invoke name="weather.get">
//! <atem:parameter name="city">Paris</atem:parameter>
//! </atem:invoke>
//! </atem:function_calls>
//! ```
//!
//! Parallel calls are consecutive messages joined by `<|eom|>`.
//!
//! The generation prompt is exactly `<|start|>assistant`, so the first header
//! arrives without its `<|start|>` - the model types ` to=self<|message|>`
//! straight out. Every later message carries its own. That asymmetry is the
//! one thing this parser has to remember.
//!
//! Verified against the shipped `tokenizer.chat_template` in the official
//! GGUF and against llama.cpp's `common_chat_params_init_muse_glimmer`
//! (`common/chat.cpp`), which reads the same grammar with a PEG.

use serde_json::Value;

pub use crate::parsers::{Parsed, ToolCallRaw};
use crate::parsers::{ToolHints, coerce};

/// The four turn-structure markers. All are single special tokens in the
/// vocab (`<|start|>` 200022, `<|message|>` 200023, `<|eom|>` 200007,
/// `<|eot|>` 200008), so they decode atomically and cannot split across
/// streaming deltas - the header TEXT between them can, which is what
/// `partial_header` is for.
pub(crate) const START: &str = "<|start|>";
pub(crate) const MESSAGE: &str = "<|message|>";
pub(crate) const EOM: &str = "<|eom|>";
pub(crate) const EOT: &str = "<|eot|>";

/// PRE-OPEN the reasoning message in the generation prompt (the g4_preopen
/// pattern). The template's `render_reasoning()` is unconditional - the model
/// always opens ` to=self<|message|>` - so these are 3 deterministic tokens
/// (` to`, `=self`, `<|message|>`) that produce no visible delta. Left to the
/// model they pace at decode/mixed-tick cadence: at 32-way concurrency under
/// admission waves that is ~95 ms per TOKEN - measured as the entire 182 ms
/// prefilled->first-token band of the TTFT chain. Forced into the prompt, the
/// token sampled from the PREFILL logits is already visible reasoning text.
/// Kill: PADDOCK_MUSE_NO_PREOPEN=1 (the model generates the opener again).
pub(crate) const PREOPEN: &str = " to=self<|message|>";

pub(crate) fn preopen() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_MUSE_NO_PREOPEN").is_none())
}

/// ATEM tool-call markup, from the template's `render_atem` macro. Note there
/// is no padding around a parameter value - the template writes
/// `'<atem:parameter name="' + k + '">'`, the value, then
/// `'</atem:parameter>\n'` - and the model's own tool instructions say so out
/// loud ("spaces for string values are not stripped"), so values are taken
/// verbatim.
/// The longest run a message header may take before its `<|message|>`:
/// `assistant to=` plus a namespaced function name, with margin. Also the
/// classification horizon for the not-a-header rule in `parse` - it must be
/// a fixed byte bound (not a smarter shape test) so the verdict is stable
/// across streaming re-parses.
const HEADER_MAX: usize = 96;

const INVOKE_OPEN: &str = "<atem:invoke name=\"";
const INVOKE_CLOSE: &str = "</atem:invoke>";
const PARAM_OPEN: &str = "<atem:parameter name=\"";
const PARAM_CLOSE: &str = "</atem:parameter>";

/// Parse one assistant turn. `thinking_open`: the prompt pre-opened
/// ` to=self<|message|>` (see `PREOPEN`), so the turn begins inside a
/// reasoning body and no header will arrive for it - later messages carry
/// their own headers as usual. `hints: None` = the request declared no tools ->
/// tool extraction is off and a tool-addressed message stays visible text
/// (see `parsers::tool_hints` for why that gate exists).
pub fn parse(text: &str, thinking_open: bool, hints: Option<&ToolHints>) -> Parsed {
    let mut out = Parsed::default();
    let mut content = String::new();
    let mut reasoning = String::new();

    let mut cur = text;
    let mut first = true;
    if thinking_open {
        let end = [EOM, EOT, START]
            .iter()
            .filter_map(|d| cur.find(d))
            .min()
            .unwrap_or(cur.len());
        push(&mut reasoning, cur[..end].trim());
        // leave the terminator in place: the loop below consumes an `<|eom|>`
        // only via the next message's `<|start|>` search, same as any body
        cur = &cur[end..];
        first = false;
    }
    loop {
        if !first {
            // every message after the first introduces itself
            let Some(s) = cur.find(START) else { break };
            cur = &cur[s + START.len()..];
        }
        // Headers are a handful of tokens (`assistant to=namespace.function`),
        // so a real one meets its `<|message|>` within HEADER_MAX bytes. Text
        // that runs past that without one is the model having left the channel
        // syntax - typically a length-cut turn that opened `<|start|>` and kept
        // talking (reachable under `<|eot|>` suppression, measured at ~8% of
        // ignore_eos bench requests, up to ~114 dropped tokens). The dialect
        // never drops what the model wrote: the run stays visible as content
        // and the scan resumes at the next structural marker. The verdict
        // reads only the first HEADER_MAX+1 bytes after `<|start|>`, which
        // never change once written - that is what keeps streaming re-parses
        // prefix-stable (a later `<|message|>` cannot retract the text).
        let msg = cur.find(MESSAGE).filter(|&m| m <= HEADER_MAX);
        if !first && msg.is_none() && cur.len() > HEADER_MAX {
            let end = [EOM, EOT, START]
                .iter()
                .filter_map(|d| cur.find(d))
                .min()
                .unwrap_or(cur.len());
            push(&mut content, cur[..end].trim());
            cur = &cur[end..];
            if let Some(r) = cur.strip_prefix(EOM) {
                cur = r;
            } else if let Some(r) = cur.strip_prefix(EOT) {
                cur = r;
            }
            if cur.is_empty() {
                break;
            }
            continue;
        }
        let Some(m) = msg else {
            // No header terminator yet. Past a `<|start|>` we are certainly
            // inside a header, so there is nothing to classify. At the very
            // start of the turn the model could instead have skipped the
            // channel syntax altogether - then the text is simply its answer,
            // unless it still reads as a header being typed out one token at
            // a time (emitting that as content could never be retracted).
            if first && !partial_header(cur) {
                push(&mut content, cur);
            }
            break;
        };
        let Some(recipient) = split_header(&cur[..m]) else {
            // Not an assistant header. Before any message that means the model
            // never opened a channel and the whole thing is its answer; after
            // one it means a hallucinated next turn (`<|start|>user...`), which
            // is only reachable when `<|eot|>` was suppressed - the turn ended
            // at the boundary either way.
            if first {
                push(&mut content, cur);
            }
            break;
        };
        first = false;

        let body_start = &cur[m + MESSAGE.len()..];
        // the body ends at the earliest structural marker; a missing
        // terminator (still generating, or max_tokens) just runs to the end
        let end = [EOM, EOT, START]
            .iter()
            .filter_map(|d| body_start.find(d))
            .min()
            .unwrap_or(body_start.len());
        let body = &body_start[..end];

        match recipient {
            Some("self") => push(&mut reasoning, body.trim()),
            None | Some("user") => push(&mut content, body.trim()),
            // Any other recipient is a tool. A body that carries no parseable
            // invoke - or a request that declared no tools - keeps its text
            // as content: this dialect never drops what the model wrote.
            Some(_) => {
                if !hints.is_some_and(|h| scan_atem(body, h, &mut out)) {
                    push(&mut content, body.trim());
                }
            }
        }

        cur = &body_start[end..];
        if let Some(r) = cur.strip_prefix(EOM) {
            cur = r;
        } else if let Some(r) = cur.strip_prefix(EOT) {
            cur = r;
        }
        // a `<|start|>` terminator is left in place for the next lap to find
        if cur.is_empty() {
            break;
        }
    }

    if !reasoning.is_empty() {
        out.reasoning = Some(reasoning);
    }
    if !content.is_empty() {
        out.content = Some(content);
    }
    out
}

/// Append one message body to a channel. Separate messages get a newline
/// between them rather than being run together; empty ones add nothing, which
/// is what keeps the accumulated string a growing PREFIX across streaming
/// re-parses (the delta path emits `[emitted..safe]` of it every tick).
fn push(dst: &mut String, s: &str) {
    if s.is_empty() {
        return;
    }
    if !dst.is_empty() {
        dst.push('\n');
    }
    dst.push_str(s);
}

/// Read an ASSISTANT message header, returning its recipient (`None` = the
/// optional ` to=user` was left out). The role word is present only on
/// messages the model introduced itself (`assistant to=user`); the first
/// header of a turn has none, because the generation prompt already spelled
/// `<|start|>assistant`.
///
/// The outer `None` means "not an assistant header": either a different role
/// (a hallucinated `<|start|>user`) or prose that happens to sit in front of a
/// `<|message|>`. Recipients are single words (`self`, `user`,
/// `namespace.function`), so whitespace inside the address disqualifies it.
fn split_header(h: &str) -> Option<Option<&str>> {
    let h = h.trim();
    let (role, recipient) = match h.find("to=") {
        Some(i) => {
            let recipient = h[i + "to=".len()..].trim();
            if recipient.is_empty() || recipient.split_whitespace().count() != 1 {
                return None;
            }
            (h[..i].trim(), Some(recipient))
        }
        // a bare role word, or nothing at all (` to=user` is optional)
        None if h.split_whitespace().count() <= 1 => (h, None),
        None => return None,
    };
    // Case-insensitive: past an `<|eot|>` (reachable only when it was
    // suppressed) the model re-introduces itself without template guidance
    // and drifts to `Assistant to=user` - a real answer-shaped message that
    // an exact match dropped whole (8-10 requests/leg, 22 tokens
    // each, the entire 126.75 -> 126.05 OSL regression). Foreign roles
    // (`user`, `system`) stay rejected.
    (role.is_empty() || role.eq_ignore_ascii_case("assistant")).then_some(recipient)
}

/// Is this text a header still being typed? Mid-stream the recipient arrives
/// as ordinary text tokens (` to`, `=self`, ...) long before the `<|message|>`
/// that would let us classify the message, so it must not be mistaken for
/// content the model started answering with.
fn partial_header(t: &str) -> bool {
    let t = t.trim_start();
    if t.is_empty() || "to=".starts_with(t) {
        return true;
    }
    t.strip_prefix("to=")
        .is_some_and(|r| !r.contains(char::is_whitespace))
}

/// Pull every `<atem:invoke>` out of one tool-addressed message body.
///
/// The template renders exactly one invoke per call, but the ATEM format
/// itself allows several inside one `<atem:function_calls>` block, so this
/// scans for all of them. The call NAME comes from the invoke tag, not from
/// the message's `to=` recipient - same choice llama.cpp's PEG makes, and the
/// two are written from the same `tc.function.name` anyway.
///
/// Returns false when nothing parsed, so the caller can keep the text visible.
fn scan_atem(body: &str, hints: &ToolHints, out: &mut Parsed) -> bool {
    let mut found = false;
    let mut cur = body;
    while let Some(i) = cur.find(INVOKE_OPEN) {
        let after = &cur[i + INVOKE_OPEN.len()..];
        let Some(q) = after.find("\">") else { break };
        let name = after[..q].trim();
        if name.is_empty() {
            break;
        }
        let rest = &after[q + "\">".len()..];
        // an unterminated final invoke (max_tokens mid-call, or still
        // generating) still parses best-effort - only closed ones count
        // complete, which is what gates streaming a call to the client
        let (block, next, closed) = match rest.find(INVOKE_CLOSE) {
            Some(e) => (&rest[..e], &rest[e + INVOKE_CLOSE.len()..], true),
            None => (rest, "", false),
        };

        let param_hints = hints.get(name);
        let mut args = serde_json::Map::new();
        let mut p = block;
        while let Some(k) = p.find(PARAM_OPEN) {
            let after_k = &p[k + PARAM_OPEN.len()..];
            let Some(kq) = after_k.find("\">") else { break };
            let key = after_k[..kq].trim().to_owned();
            let vstart = &after_k[kq + "\">".len()..];
            // a half-emitted value is dropped rather than guessed at - the
            // same rule laguna's parser follows, for the same reason
            let Some(ve) = vstart.find(PARAM_CLOSE) else {
                break;
            };
            let declared_string = param_hints.and_then(|h| h.get(&key)).copied();
            args.insert(key, coerce(&vstart[..ve], declared_string));
            p = &vstart[ve + PARAM_CLOSE.len()..];
        }

        out.tool_calls.push(ToolCallRaw {
            name: name.to_owned(),
            arguments: Value::Object(args).to_string(),
        });
        if closed {
            out.complete_calls = out.tool_calls.len();
        }
        found = true;
        cur = next;
    }
    found
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parsers::tool_hints;

    fn hints_weather() -> Option<ToolHints> {
        tool_hints(Some(&[serde_json::json!({"type":"function","function":{
            "name":"weather.get",
            "parameters":{"type":"object","properties":{
                "city":{"type":"string"},
                "days":{"type":"integer"},
                "units":{"type":"string"}
            }}
        }})]))
    }

    #[test]
    fn thought_then_answer() {
        // the shape every ordinary turn has: the prompt's `<|start|>assistant`
        // is not in the generated text, so the first header rides bare
        let t = " to=self<|message|>the user wants the capital<|eom|>\
                  <|start|>assistant to=user<|message|>Paris.<|eot|>";
        let p = parse(t, false, None);
        assert_eq!(p.reasoning.as_deref(), Some("the user wants the capital"));
        assert_eq!(p.content.as_deref(), Some("Paris."));
        assert_eq!(p.finish_reason(), "stop");
    }

    #[test]
    fn the_marker_never_leaks_into_content() {
        // this whole turn used to come back as one `content`
        // string with `to=self<|message|>` still in it and no reasoning at all
        let t = " to=self<|message|>hmm<|eom|><|start|>assistant to=user<|message|>Red.<|eot|>";
        let p = parse(t, false, None);
        assert!(!p.content.as_deref().expect("content").contains("to=self"));
        assert!(!p.content.as_deref().expect("content").contains(MESSAGE));
        assert_eq!(p.content.as_deref(), Some("Red."));
    }

    #[test]
    fn mid_thought_is_all_reasoning() {
        let p = parse(" to=self<|message|>let me consider", false, None);
        assert_eq!(p.reasoning.as_deref(), Some("let me consider"));
        assert!(p.content.is_none());
    }

    #[test]
    fn header_being_typed_classifies_nothing() {
        // mid-stream the recipient arrives as ordinary text tokens, long
        // before the `<|message|>` that says which channel this is
        for prefix in [
            "", " ", " t", " to", " to=", " to=s", " to=self", " to=user",
        ] {
            let p = parse(prefix, false, None);
            assert_eq!(p.content, None, "{prefix:?} leaked as content");
            assert_eq!(p.reasoning, None, "{prefix:?} leaked as reasoning");
        }
        // ...and the same after `<|eom|>`, where the role word comes first
        for tail in ["", "assistant", "assistant to=us"] {
            let t = format!(" to=self<|message|>hmm<|eom|><|start|>{tail}");
            let p = parse(&t, false, None);
            assert_eq!(p.reasoning.as_deref(), Some("hmm"));
            assert_eq!(p.content, None, "{tail:?} leaked as content");
        }
    }

    #[test]
    fn optional_recipient_is_content() {
        // the template makes ` to=user` optional - a bare `<|message|>` after
        // the header is the answer channel
        let p = parse("<|message|>Paris.<|eot|>", false, None);
        assert_eq!(p.content.as_deref(), Some("Paris."));
        assert!(p.reasoning.is_none());
    }

    #[test]
    fn no_channel_syntax_at_all_is_content() {
        // robustness fallback: a model that never opened a channel still gets
        // its text to the client rather than having it silently dropped
        let p = parse("Hello there.", false, None);
        assert_eq!(p.content.as_deref(), Some("Hello there."));
        assert!(p.reasoning.is_none());
    }

    #[test]
    fn multiple_thoughts_join() {
        let t = " to=self<|message|>first<|eom|><|start|>assistant to=self<|message|>second<|eom|>\
                  <|start|>assistant to=user<|message|>Done.<|eot|>";
        let p = parse(t, false, None);
        assert_eq!(p.reasoning.as_deref(), Some("first\nsecond"));
        assert_eq!(p.content.as_deref(), Some("Done."));
    }

    #[test]
    fn tool_call_arrives_typed() {
        let t = " to=self<|message|>need weather<|eom|>\
                  <|start|>assistant to=weather.get<|message|><atem:function_calls>\n\
                  <atem:invoke name=\"weather.get\">\n\
                  <atem:parameter name=\"city\">Paris</atem:parameter>\n\
                  <atem:parameter name=\"days\">3</atem:parameter>\n\
                  </atem:invoke>\n</atem:function_calls><|eot|>";
        let p = parse(t, false, hints_weather().as_ref());
        assert_eq!(p.reasoning.as_deref(), Some("need weather"));
        assert_eq!(p.tool_calls.len(), 1);
        assert_eq!(p.tool_calls[0].name, "weather.get");
        let args: Value = serde_json::from_str(&p.tool_calls[0].arguments).expect("args parse");
        assert_eq!(args["city"], "Paris");
        assert_eq!(args["days"], 3);
        assert_eq!((p.complete_calls, p.finish_reason()), (1, "tool_calls"));
        assert!(p.content.is_none());
    }

    #[test]
    fn string_param_never_coerced() {
        let t = " to=weather.get<|message|><atem:function_calls>\n\
                  <atem:invoke name=\"weather.get\">\n\
                  <atem:parameter name=\"units\">123</atem:parameter>\n\
                  </atem:invoke>\n</atem:function_calls>";
        let p = parse(t, false, hints_weather().as_ref());
        let args: Value = serde_json::from_str(&p.tool_calls[0].arguments).expect("args parse");
        assert_eq!(args["units"], "123");
    }

    #[test]
    fn multiline_value_keeps_its_newlines() {
        // the template pads nothing around a value, so inner newlines are data
        let t = " to=weather.get<|message|><atem:function_calls>\n\
                  <atem:invoke name=\"weather.get\">\n\
                  <atem:parameter name=\"city\">line one\nline two</atem:parameter>\n\
                  </atem:invoke>\n</atem:function_calls>";
        let p = parse(t, false, hints_weather().as_ref());
        let args: Value = serde_json::from_str(&p.tool_calls[0].arguments).expect("args parse");
        assert_eq!(args["city"], "line one\nline two");
    }

    #[test]
    fn parallel_calls_are_consecutive_messages() {
        let t = " to=weather.get<|message|><atem:function_calls>\n\
                  <atem:invoke name=\"weather.get\">\n\
                  <atem:parameter name=\"city\">Paris</atem:parameter>\n\
                  </atem:invoke>\n</atem:function_calls><|eom|>\
                  <|start|>assistant to=weather.get<|message|><atem:function_calls>\n\
                  <atem:invoke name=\"weather.get\">\n\
                  <atem:parameter name=\"city\">Berlin</atem:parameter>\n\
                  </atem:invoke>\n</atem:function_calls><|eot|>";
        let p = parse(t, false, hints_weather().as_ref());
        assert_eq!((p.tool_calls.len(), p.complete_calls), (2, 2));
        let b: Value = serde_json::from_str(&p.tool_calls[1].arguments).expect("args parse");
        assert_eq!(b["city"], "Berlin");
        assert!(p.content.is_none());
    }

    #[test]
    fn unterminated_call_keeps_complete_args_only() {
        // max_tokens mid-value: the finished arg survives, the half one drops,
        // and the call does not count complete so streaming holds it back
        let t = " to=weather.get<|message|><atem:function_calls>\n\
                  <atem:invoke name=\"weather.get\">\n\
                  <atem:parameter name=\"city\">Paris</atem:parameter>\n\
                  <atem:parameter name=\"days\">3";
        let p = parse(t, false, hints_weather().as_ref());
        assert_eq!((p.tool_calls.len(), p.complete_calls), (1, 0));
        let args: Value = serde_json::from_str(&p.tool_calls[0].arguments).expect("args parse");
        assert_eq!(args["city"], "Paris");
        assert!(args.get("days").is_none());
    }

    #[test]
    fn no_tools_declared_keeps_atem_markup_visible() {
        // the gate: a model can't call tools the request never offered, so
        // the markup is ordinary text the client must still see
        let t = " to=weather.get<|message|><atem:function_calls>\n\
                  <atem:invoke name=\"weather.get\">\n</atem:invoke>\n</atem:function_calls>";
        let p = parse(t, false, None);
        assert!(p.tool_calls.is_empty());
        assert_eq!(p.finish_reason(), "stop");
        assert!(
            p.content
                .as_deref()
                .expect("content")
                .contains("<atem:invoke")
        );
    }

    #[test]
    fn tool_recipient_without_a_call_stays_content() {
        // addressed to a tool but carrying prose: nothing parses, and nothing
        // may disappear either
        let t = " to=weather.get<|message|>I meant to call this.";
        let p = parse(t, false, hints_weather().as_ref());
        assert!(p.tool_calls.is_empty());
        assert_eq!(p.content.as_deref(), Some("I meant to call this."));
    }

    #[test]
    fn a_hallucinated_next_turn_ends_the_assistant_turn() {
        // only reachable with `<|eot|>` suppressed (it is a stop token), but
        // the parser must not hand a fabricated user turn back as content
        let t = " to=user<|message|>Paris.<|eot|><|start|>user<|message|>and Berlin?";
        let p = parse(t, false, None);
        assert_eq!(p.content.as_deref(), Some("Paris."));
        assert!(!p.content.as_deref().expect("content").contains("Berlin"));
    }

    #[test]
    fn a_start_that_never_opens_a_message_stays_visible() {
        // A length-cut turn that opened `<|start|>` and kept talking without
        // `<|message|>` (reachable under `<|eot|>` suppression, ~8% of
        // ignore_eos bench requests): past HEADER_MAX the run is not a header
        // and must stay visible - the old parse silently dropped up to ~114
        // tokens here.
        let ramble = "assistant went completely off the rails here and kept \
                      producing prose for a very long stretch without ever \
                      closing its channel header";
        assert!(ramble.len() > HEADER_MAX);
        let t = format!("thinking hard{EOM}{START}{ramble}");
        let p = parse(&t, true, None);
        assert_eq!(p.reasoning.as_deref(), Some("thinking hard"));
        assert_eq!(p.content.as_deref(), Some(ramble));
    }

    #[test]
    fn a_short_unterminated_header_is_still_held() {
        // within HEADER_MAX the text is genuinely ambiguous (a header being
        // typed at the moment of a length cut) - it must not flash as content
        let t = format!("thinking hard{EOM}{START}assistant to=we");
        let p = parse(&t, true, None);
        assert_eq!(p.reasoning.as_deref(), Some("thinking hard"));
        assert!(p.content.is_none());
    }

    #[test]
    fn a_capitalized_assistant_header_is_still_the_assistant() {
        // past an `<|eot|>` (only reachable when it was suppressed) the model
        // re-introduces itself without template guidance and drifts to
        // `Assistant to=user` - that is a real answer, not a foreign turn
        // (8-10 requests/leg lost 22 tokens each on the exact-case
        // tripwire). Foreign roles stay dropped.
        let t = format!(
            "thinking hard{EOT}{START}Assistant to=user{MESSAGE}Your posted fragment looks fine."
        );
        let p = parse(&t, true, None);
        assert_eq!(p.reasoning.as_deref(), Some("thinking hard"));
        assert_eq!(
            p.content.as_deref(),
            Some("Your posted fragment looks fine.")
        );
        let t = format!("thinking{EOT}{START}user{MESSAGE}pretend turn");
        let p = parse(&t, true, None);
        assert!(p.content.is_none(), "foreign roles must stay rejected");
    }

    #[test]
    fn disowned_run_grows_prefix_stable() {
        // the not-a-header verdict reads only the first HEADER_MAX+1 bytes,
        // so streaming re-parses may never retract text - even when a
        // `<|message|>` eventually arrives beyond the horizon
        let mut toks = vec!["thinking", EOM, START];
        toks.extend_from_slice(&[" blah"; 30]);
        toks.extend_from_slice(&[MESSAGE, " tail", EOT]);
        let (mut acc, mut c_prev) = (String::new(), String::new());
        for t in &toks {
            acc.push_str(t);
            let p = parse(&acc, true, None);
            let c = p.content.unwrap_or_default();
            assert!(
                c.starts_with(&c_prev),
                "content shrank at {t:?}: {c_prev:?} -> {c:?}"
            );
            c_prev = c;
        }
        // nothing dropped: the disowned run keeps its text through the end,
        // with the stray `<|message|>` visible as the raw prose it is
        assert!(c_prev.contains("blah blah"));
        assert!(c_prev.ends_with(" tail"));
    }

    #[test]
    fn streaming_reparse_only_ever_grows_the_prefix() {
        // The delta path re-parses the whole turn every token and emits
        // `[emitted..len]`, so a later parse may never contradict an earlier
        // one - that is the contract this dialect has to hold.
        //
        // Grown one TOKEN at a time, not one byte: the four markers are single
        // special ids in this vocab, so they arrive whole. (Byte-splitting a
        // marker would leave `<|eo` inside the body, which is a hazard the
        // holdback layer owns, not the parser.)
        let toks = [
            " to",
            "=self",
            MESSAGE,
            "thinking",
            " hard",
            EOM,
            START,
            "assistant",
            " to",
            "=user",
            MESSAGE,
            "The",
            " answer",
            " is",
            " 42",
            ".",
            EOT,
        ];
        stream_monotone(&toks, false);
        // pre-opened render (PREOPEN): the header never appears in the text
        let toks = [
            "thinking",
            " hard",
            EOM,
            START,
            "assistant",
            " to",
            "=user",
            MESSAGE,
            "The",
            " answer",
            " is",
            " 42",
            ".",
            EOT,
        ];
        stream_monotone(&toks, true);
    }

    fn stream_monotone(toks: &[&str], thinking_open: bool) {
        let (mut acc, mut r_prev, mut c_prev) = (String::new(), String::new(), String::new());
        for t in toks {
            acc.push_str(t);
            let p = parse(&acc, thinking_open, None);
            let r = p.reasoning.unwrap_or_default();
            let c = p.content.unwrap_or_default();
            assert!(
                r.starts_with(&r_prev),
                "reasoning shrank at {t:?}: {r_prev:?} -> {r:?}"
            );
            assert!(
                c.starts_with(&c_prev),
                "content shrank at {t:?}: {c_prev:?} -> {c:?}"
            );
            r_prev = r;
            c_prev = c;
        }
        assert_eq!(r_prev, "thinking hard");
        assert_eq!(c_prev, "The answer is 42.");
    }
}
