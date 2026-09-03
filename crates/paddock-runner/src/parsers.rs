//! Model-family output dialects: one enum, one parse entry point. The routes
//! never call a family parser directly - they ask the served model's `Dialect`
//! to turn raw generated text (specials visible) into content / reasoning /
//! tool calls.

use std::collections::HashMap;

use serde_json::Value;

/// Parsed assistant output, shared across dialects.
#[derive(Debug, Default, PartialEq)]
pub struct Parsed {
    pub content: Option<String>,
    pub reasoning: Option<String>,
    pub tool_calls: Vec<ToolCallRaw>,
    /// how many leading `tool_calls` are DEFINITIVELY terminated (their block
    /// closed) - a mid-generation parse may carry one trailing in-progress
    /// call beyond this count. Streaming emits calls as they complete.
    pub complete_calls: usize,
}

#[derive(Debug, PartialEq)]
pub struct ToolCallRaw {
    pub name: String,
    /// JSON-encoded arguments object (OpenAI wire shape).
    pub arguments: String,
}

impl Parsed {
    pub fn finish_reason(&self) -> &'static str {
        if !self.tool_calls.is_empty() {
            "tool_calls"
        } else {
            "stop"
        }
    }
}

/// Which assistant-output syntax the served model emits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dialect {
    /// gpt-oss Harmony channels (`<|channel|>analysis/final/commentary`).
    Harmony,
    /// Qwen3.5 `<think>` blocks + XML-ish tool calls
    /// (`<tool_call><function=NAME><parameter=KEY>...`).
    QwenXml,
    /// Gemma 4 channels: `<|channel>thought\n...<channel|>` reasoning, then
    /// content. `<|channel>`/`<channel|>` are single special tokens (ids
    /// 100/101), so they decode atomically like Harmony's markers. Tool-call
    /// channels parse with the gemma4 tools milestone - until then non-thought
    /// channels pass through as content (nothing silently disappears).
    GemmaChannel,
    /// Laguna (poolside) GLM-shaped XML: `<think>` blocks like qwen, but tool
    /// calls are `<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value>...
    /// </tool_call>` - the name rides bare after the opener, no `<function=>`
    /// wrapper, and values are tojson'd unless the schema says string. The
    /// turn ends on `</assistant>` (a single control token, id 24 - wired as a
    /// stop token, so it never appears in parsed text).
    Laguna,
    /// Muse Glimmer's recipient channels: the turn is a run of messages, each
    /// `<|start|>assistant to=RECIPIENT<|message|>...<|eom|>/<|eot|>`, where
    /// `self` is reasoning, `user` (or no recipient) is the answer, and any
    /// other recipient is a tool call carrying Anthropic-style
    /// `<atem:function_calls>` markup. Harmony-SHAPED but not Harmony: the
    /// channel is an address, and the tool body is XML rather than JSON - see
    /// `crate::muse`.
    MuseChannel,
    /// The Hermes-style convention: a whole JSON object inside `<tool_call>`
    /// tags - `<tool_call>\n{"name": ..., "arguments": {...}}\n</tool_call>` - and
    /// no reasoning region at all. Named for the shape, not a vendor, because
    /// several families share it; IBM Granite 4.1 is the first one we serve.
    /// The arguments arrive already typed (real JSON), so unlike the two XML
    /// dialects there is nothing to coerce against the request's schema.
    JsonToolCall,
    /// No known structure: the whole text is content.
    Plain,
}

impl Dialect {
    pub fn for_arch(arch: &str) -> Dialect {
        match arch {
            "gpt-oss" => Dialect::Harmony,
            // nemotron_h_moe's template speaks the same XML tool dialect and
            // `<think>` region as qwen3.5 (the card's own parser election is
            // vLLM's `qwen3_coder`)
            "qwen35" | "qwen35moe" | "nemotron" | "nemotron_h_moe" => Dialect::QwenXml,
            "gemma4" => Dialect::GemmaChannel,
            "laguna" => Dialect::Laguna,
            "muse-glimmer" => Dialect::MuseChannel,
            "granite" => Dialect::JsonToolCall,
            _ => Dialect::Plain,
        }
    }

    /// Same, but allowed to read the checkpoint's own chat template where the
    /// arch string cannot decide the answer. Prefer this at every load site.
    ///
    /// Granite is what forces it: `general.architecture` is `granite` for both
    /// 4.1 and 4.2, but 4.2 changed the two things a dialect names.
    ///
    /// | | 4.1 | 4.2 |
    /// |---|---|---|
    /// | tool calls | `<tool_call>\n{json}\n</tool_call>` | `<tool_call>\n<function=NAME>\n<parameter=K>` |
    /// | thinking | none | prompt pre-opens `<think>\n`, model closes `</think>` |
    ///
    /// Both of those are exactly `QwenXml`. Keyed on arch alone, 4.2 serves its
    /// entire reasoning block as user-visible content (with a stray `</think>`
    /// in the middle) and every tool call as prose - measured on
    /// `granite-4.2-8b-Q8_0` before this existed.
    ///
    /// The probe is the marker pair, not a version string: `<function=` says
    /// the tool body is XML rather than JSON, and `<think>` says there is a
    /// reasoning region at all. A future granite that keeps 4.1's JSON body
    /// has neither and still lands on `JsonToolCall`. Verified against both
    /// templates - 4.1 has 0 `<function=` and 0 `<think>`; 4.2
    /// has both.
    pub fn for_arch_and_template(arch: &str, template: Option<&str>) -> Dialect {
        if arch == "granite"
            && template.is_some_and(|t| t.contains("<function=") && t.contains("<think>"))
        {
            return Dialect::QwenXml;
        }
        Dialect::for_arch(arch)
    }

    /// Did the rendered generation prompt leave the model inside an open
    /// thinking region? Dialect-shaped, one definition for all three routes
    /// (chat, messages, responses): qwen3.5 pre-opens `<think>\n`; laguna
    /// pre-opens a bare `<think>` (thinking on) or pre-closes `</think>`
    /// (off - which never suffix-matches `<think>`, so the probe is safe);
    /// gemma4 pre-closes an empty thought channel when thinking is off, so
    /// thinking is on exactly when that suffix is absent.
    pub fn thinking_open(self, prompt: &str) -> bool {
        match self {
            Dialect::GemmaChannel => !prompt.ends_with("<channel|>"),
            Dialect::Laguna => prompt.ends_with("<think>"),
            // granite 4.1's template has no thinking region to open - say so
            // rather than leaning on a `<think>\n` probe that can't match
            Dialect::JsonToolCall => false,
            // muse-glimmer's generation prompt is a bare `<|start|>assistant`
            // and the model types its own ` to=self<|message|>` - Unless the
            // render pre-opened it (crate::muse::PREOPEN, the g4_preopen
            // pattern), in which case the turn starts inside the reasoning body
            Dialect::MuseChannel => prompt.ends_with(crate::muse::PREOPEN),
            _ => prompt.ends_with("<think>\n"),
        }
    }

    /// Markers that can open a non-content region mid-stream. Content deltas
    /// hold back a partial-tag tail so these never leak to the client.
    /// Harmony's channel markers are single special tokens (decode atomically),
    /// so it needs no holdback.
    pub fn content_markers(self) -> &'static [&'static str] {
        match self {
            // These three all spell `<tool_call>`/`<think>` as single atomic,
            // NON-special tokens - measured on the served GGUFs
            // (laguna ids 25/26/18/19, qwen 248058/248059/248068/248069,
            // granite 100270/100271/100274/100275; all decode visibly, which is
            // also what lets the forced-tool grammar emit them at all). So they
            // cannot split across deltas, and the holdback is here for the
            // other case: a model spelling a marker out of ordinary text.
            Dialect::QwenXml | Dialect::Laguna => &["<tool_call>", "<think>", "</think>"],
            // the bare opener rides along so a turn that is a call cannot leak
            // its first bytes as content before the parse can classify it
            // (`bare_lead`); it withholds at most those few bytes, and a turn
            // that merely starts with a `{` resolves on the next delta
            Dialect::JsonToolCall => &["<tool_call>", JSON_BARE_OPEN],
            // atomic special tokens can't split across deltas, but a marker
            // in the running text must still gate what streams as content
            Dialect::GemmaChannel => &["<|channel>"],
            // muse-glimmer's four structure markers are single special ids
            // (200022/200023/200007/200008), so they arrive whole. The part
            // that does split is the ` to=RECIPIENT` header text, and holding
            // bytes back cannot help there - the recipient decides which
            // channel the message is, so nothing may be classified until its
            // `<|message|>` lands. `muse::partial_header` is that guard.
            _ => &[],
        }
    }

    /// The syntax a FORCED tool call has to be generated in on this family, or
    /// `None` when we have no grammar for it yet (`tool_choice: "required"` /
    /// `"any"` / a named tool is then a 400 - see `no_forced_tool_grammar`).
    ///
    /// A dialect belongs here only once its parser reads that syntax back:
    /// forcing a call whose text never becomes `tool_calls` would hand the
    /// client raw markup and call it content. That is why gemma4 is absent -
    /// its tool channels aren't parsed yet.
    pub fn tool_syntax(self) -> Option<crate::constrained::ToolSyntax> {
        use crate::constrained::ToolSyntax;
        match self {
            Dialect::QwenXml => Some(ToolSyntax::QwenXml),
            Dialect::Laguna => Some(ToolSyntax::LagunaXml),
            Dialect::JsonToolCall => Some(ToolSyntax::Json),
            Dialect::MuseChannel => Some(ToolSyntax::AtemXml),
            _ => None,
        }
    }

    /// The CITED rung names for a family whose template splices the effort
    /// value straight into the prompt instead of validating it.
    ///
    /// Two families do that, two spellings, two ladders: gpt-oss reads
    /// `reasoning_effort` at low/medium/high, and muse-glimmer reads
    /// `reasoning_strength` at low/medium/high/**xhigh** (the model card's own
    /// list; its template renders `Reasoning strength: <value>.` verbatim).
    /// Because any string renders on those two, no probe can discover the real
    /// vocabulary - the card is the only source, so it lives here with its
    /// citation.
    ///
    /// This is not the general answer to "does this model grade effort", and
    /// must not be used as one: `crate::reasoning::probe` measures that from
    /// the served template and consults this only for the interpolating case.
    /// Qwen3.8 is the worked example - same dialect as Qwen3.6, three rungs to
    /// 3.6's none, and its template declares them itself.
    pub fn effort_kwarg(self) -> Option<(&'static str, &'static [&'static str])> {
        match self {
            Dialect::Harmony => Some(("reasoning_effort", &["low", "medium", "high"])),
            Dialect::MuseChannel => {
                Some(("reasoning_strength", &["low", "medium", "high", "xhigh"]))
            }
            _ => None,
        }
    }

    /// Markers that end the reasoning region (for reasoning-delta holdback).
    /// QwenXml also holds `<tool_call>` back: blocks extract as calls even
    /// inside the think region, so the syntax must never flash in the fold.
    pub fn reasoning_markers(self) -> &'static [&'static str] {
        match self {
            Dialect::QwenXml => &["</think>", "<tool_call>"],
            Dialect::Laguna => &["</think>"],
            Dialect::GemmaChannel => &["<channel|>"],
            // muse-glimmer closes a thought with `<|eom|>`, a single special
            // id - same reasoning as `content_markers`
            _ => &[],
        }
    }

    /// Special tokens this family's tool-call GRAMMAR is allowed to spell.
    ///
    /// The constrained decoder refuses control tokens inside a grammar region
    /// by default - a model must not be able to inject `<|im_end|>` into a
    /// forced tool call. muse-glimmer is the family where that blanket rule is
    /// wrong: its message envelope is made of special tokens, so a forced call
    /// has to write `<|message|>` and an analysis block has to close with
    /// `<|eom|>`. This is the same idea as llama.cpp's `preserved_tokens`
    /// list, which its muse handler populates with exactly these.
    ///
    /// `<|eot|>` is deliberately not here: it is a stop token, and stop tokens
    /// never reach the grammar - the engine gates them on `may_stop()`, which
    /// is what makes "a forced call cannot be truncated" true.
    pub fn grammar_specials(self) -> &'static [&'static str] {
        match self {
            Dialect::MuseChannel => &[crate::muse::START, crate::muse::MESSAGE, crate::muse::EOM],
            _ => &[],
        }
    }
}

/// `tools[].function.name -> (parameter name -> declared type is string)`,
/// from the request's tool definitions. Qwen XML parameter values arrive as
/// raw text; the schema decides whether `123` stays a string or becomes JSON.
pub type ToolHints = HashMap<String, HashMap<String, bool>>;

/// `None` when the request declares no tools - and that `None` is a GATE, not
/// just missing type info: a model can't call tools the request never offered
/// (OpenAI semantics; llama.cpp behaves the same), so with no tools declared
/// the dialect's tool-call syntax is ordinary text. This matters beyond
/// conformance: synthetic benchmark prompts (coding corpora) contain fake
/// tool markup the model imitates, and parsing it into `tool_calls` on a
/// tools-free request silently ate ~1/3 of the visible stream.
pub fn tool_hints(tools: Option<&[Value]>) -> Option<ToolHints> {
    let ts = tools?;
    if ts.is_empty() {
        return None;
    }
    let mut hints = ToolHints::new();
    for t in ts {
        let f = t.get("function").unwrap_or(t);
        let Some(name) = f.get("name").and_then(Value::as_str) else {
            continue;
        };
        let mut params = HashMap::new();
        if let Some(props) = f
            .get("parameters")
            .and_then(|p| p.get("properties"))
            .and_then(Value::as_object)
        {
            for (k, schema) in props {
                let is_string = schema.get("type").and_then(Value::as_str) == Some("string");
                params.insert(k.clone(), is_string);
            }
        }
        hints.insert(name.to_owned(), params);
    }
    Some(hints)
}

/// Parse one assistant turn. `thinking_open`: the rendered prompt ended inside
/// an open thinking region (qwen pre-opens `<think>\n` in its template;
/// gemma4 gets `<|channel>thought\n` forced at render - see `g4_preopen`),
/// so text is reasoning until the dialect's close marker shows up.
/// `hints: None` = the request declared no tools -> tool extraction is off and
/// tool-shaped syntax stays in content/reasoning verbatim (see `tool_hints`).
pub fn parse(
    dialect: Dialect,
    text: &str,
    thinking_open: bool,
    hints: Option<&ToolHints>,
) -> Parsed {
    match dialect {
        Dialect::Harmony => crate::harmony::parse(text, hints.is_some()),
        Dialect::QwenXml => qwen_parse(text, thinking_open, hints),
        Dialect::GemmaChannel => gemma_parse(text, thinking_open),
        Dialect::Laguna => laguna_parse(text, thinking_open, hints),
        Dialect::MuseChannel => crate::muse::parse(text, thinking_open, hints),
        Dialect::JsonToolCall => json_tool_parse(text, hints),
        Dialect::Plain => {
            let t = text.trim();
            Parsed {
                content: (!t.is_empty()).then(|| t.to_owned()),
                ..Parsed::default()
            }
        }
    }
}

/// Bytes to hold back from a streamed delta: the longest suffix of `s` that is
/// a proper prefix of any marker, so a tag arriving across chunks never leaks.
/// Byte-wise (markers are ASCII; a multi-byte tail can't prefix-match them
/// beyond whole bytes) - callers must still round down to a char boundary.
pub fn holdback(s: &str, markers: &[&str]) -> usize {
    let b = s.as_bytes();
    let max = markers
        .iter()
        .map(|m| m.len().saturating_sub(1))
        .max()
        .unwrap_or(0)
        .min(b.len());
    for take in (1..=max).rev() {
        let tail = &b[b.len() - take..];
        if markers
            .iter()
            .any(|m| m.len() > take && &m.as_bytes()[..take] == tail)
        {
            return take;
        }
    }
    0
}

pub(crate) const G_THOUGHT: &str = "<|channel>thought\n";
pub(crate) const G_CLOSE: &str = "<channel|>";

/// gemma4 thinking: PRE-OPEN the thought channel in the generation prompt
/// (the qwen pattern, where the template itself ends `<think>\n`). Left to
/// the model, the opener costs ~3 sampled tokens that produce no visible
/// delta - paced at decode/mixed-tick cadence they were the entire
/// first-token residual of the admission path (+56 ms at
/// idle, ~+600 ms under admission bursts). Forced into the prompt, the
/// token sampled from the PREFILL logits is already visible reasoning text.
/// Kill: PADDOCK_G4_NO_PREOPEN=1 (the model generates the opener again).
pub(crate) fn g4_preopen() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_G4_NO_PREOPEN").is_none())
}

/// Gemma 4 assistant output (thinking on - the serving default): the turn
/// opens with a thought channel, content follows its close. Mid-stream, an
/// unclosed thought is all reasoning. Without a thought channel the whole
/// text is content. Non-thought channels (tool_call) are content until the
/// gemma4 tools milestone parses them. `thinking_open`: the prompt already
/// pre-opened the thought channel, so the text starts inside it (a
/// model-generated opener is still stripped - the kill-env path keeps it).
fn gemma_parse(text: &str, thinking_open: bool) -> Parsed {
    let trimmed = text.trim_start();
    // a proper prefix of the thought marker ("<|chan", "<|channel>thought")
    // is still AMBIGUOUS mid-stream - emit nothing rather than misfile the
    // opening tag as content it can never retract
    if !trimmed.is_empty() && trimmed.len() < G_THOUGHT.len() && G_THOUGHT.starts_with(trimmed) {
        return Parsed::default();
    }
    let after = match (trimmed.strip_prefix(G_THOUGHT), thinking_open) {
        (Some(a), _) => Some(a),
        (None, true) => Some(trimmed),
        (None, false) => None,
    };
    if let Some(after) = after {
        match after.find(G_CLOSE) {
            Some(end) => {
                let reasoning = after[..end].trim();
                let content = after[end + G_CLOSE.len()..].trim();
                Parsed {
                    reasoning: (!reasoning.is_empty()).then(|| reasoning.to_owned()),
                    content: (!content.is_empty()).then(|| content.to_owned()),
                    ..Parsed::default()
                }
            }
            None => Parsed {
                reasoning: {
                    let r = after.trim();
                    (!r.is_empty()).then(|| r.to_owned())
                },
                ..Parsed::default()
            },
        }
    } else {
        let t = trimmed.trim_end();
        Parsed {
            content: (!t.is_empty()).then(|| t.to_owned()),
            ..Parsed::default()
        }
    }
}

const THINK: &str = "<think>";
const THINK_END: &str = "</think>";
const TOOL_CALL: &str = "<tool_call>";
const TOOL_CALL_END: &str = "</tool_call>";

/// Qwen3.5 assistant output. Structure (specials visible, engine already
/// stopped before `<|im_end|>`):
///
///   [reasoning `</think>`]        - thinking mode: the PROMPT opened `<think>`,
///                                   so the opening tag is usually absent
///   preamble text                 - optional, template allows it before calls
///   <tool_call>\n<function=NAME>\n<parameter=KEY>\nVALUE\n</parameter>...
///   </function>\n</tool_call>     - repeated for parallel calls
fn qwen_parse(text: &str, thinking_open: bool, hints: Option<&ToolHints>) -> Parsed {
    let mut out = Parsed::default();

    // Blocks are calls wherever they appear, including inside the reasoning
    // region: qwen sometimes writes its tool call inside a still-open <think>
    // and ends the turn without ever closing it (seen live - a
    // well-formed call, stop right after it, round dead with zero calls and
    // zero content: "the model stopped responding"). vLLM's hermes parser
    // tolerates think-embedded calls the same way. No tools declared -> no
    // extraction anywhere: the syntax is just text the model produced.
    let scan = |region: &str, out: &mut Parsed| -> String {
        match hints {
            Some(h) => scan_qwen_blocks(region, h, out),
            None => region.to_owned(),
        }
    };

    // 1) split off the reasoning block
    let rest = if let Some(i) = text.find(THINK_END) {
        let r = text[..i].trim();
        let r = r.strip_prefix(THINK).unwrap_or(r);
        let r = scan(r, &mut out);
        let r = r.trim();
        if !r.is_empty() {
            out.reasoning = Some(r.to_owned());
        }
        &text[i + THINK_END.len()..]
    } else if thinking_open || text.trim_start().starts_with(THINK) {
        // still inside the think block (streaming, or max_tokens mid-thought)
        let r = text.trim();
        let r = r.strip_prefix(THINK).unwrap_or(r);
        let r = scan(r, &mut out);
        let r = r.trim();
        if !r.is_empty() {
            out.reasoning = Some(r.to_owned());
        }
        return out;
    } else {
        text
    };

    // 2) tool_call blocks; content is everything outside them
    let content = scan(rest, &mut out);
    let content = content.trim();
    if !content.is_empty() {
        out.content = Some(content.to_owned());
    }
    out
}

/// Scan a region for `<tool_call>...</tool_call>` blocks: parsed calls land in
/// `out`, and the return is the region's text with the blocks removed. An
/// unterminated final block (max_tokens mid-call, or still generating) still
/// parses best-effort - but only closed blocks count complete.
fn scan_qwen_blocks(region: &str, hints: &ToolHints, out: &mut Parsed) -> String {
    let mut kept = String::new();
    let mut cur = region;
    while let Some(s) = cur.find(TOOL_CALL) {
        kept.push_str(&cur[..s]);
        let after = &cur[s + TOOL_CALL.len()..];
        let (block, next, closed) = match after.find(TOOL_CALL_END) {
            Some(e) => (&after[..e], &after[e + TOOL_CALL_END.len()..], true),
            None => (after, "", false),
        };
        if let Some(tc) = parse_function_block(block, hints) {
            out.tool_calls.push(tc);
            if closed {
                out.complete_calls = out.tool_calls.len();
            }
        }
        cur = next;
    }
    kept.push_str(cur);
    kept
}

fn parse_function_block(block: &str, hints: &ToolHints) -> Option<ToolCallRaw> {
    let f = block.find("<function=")?;
    let after = &block[f + "<function=".len()..];
    let name_end = after.find('>')?;
    let name = after[..name_end].trim();
    if name.is_empty() {
        return None;
    }
    let mut body = &after[name_end + 1..];
    if let Some(e) = body.find("</function>") {
        body = &body[..e];
    }

    let param_hints = hints.get(name);
    let mut args = serde_json::Map::new();
    let mut cur = body;
    while let Some(p) = cur.find("<parameter=") {
        let after_p = &cur[p + "<parameter=".len()..];
        let Some(k_end) = after_p.find('>') else {
            break;
        };
        let key = after_p[..k_end].trim().to_owned();
        let vstart = &after_p[k_end + 1..];
        let (raw, next) = match vstart.find("</parameter>") {
            Some(e) => (&vstart[..e], &vstart[e + "</parameter>".len()..]),
            None => (vstart, ""),
        };
        // the template wraps values in single newlines; inner newlines are data
        let val = raw.strip_prefix('\n').unwrap_or(raw);
        let val = val.strip_suffix('\n').unwrap_or(val);
        let declared_string = param_hints.and_then(|h| h.get(&key)).copied();
        args.insert(key, coerce(val, declared_string));
        cur = next;
    }

    Some(ToolCallRaw {
        name: name.to_owned(),
        arguments: Value::Object(args).to_string(),
    })
}

/// Laguna assistant output (specials visible, engine already stopped before
/// `</assistant>`). Same `<think>` choreography as qwen - the template
/// pre-opens `<think>` (thinking on) or pre-closes `</think>` (off) - but
/// GLM-shaped tool calls:
///
///   <tool_call>NAME<arg_key>KEY</arg_key><arg_value>VALUE</arg_value>...</tool_call>
///
/// repeated for parallel calls. No newline padding around values (unlike the
/// qwen template), so string values are taken verbatim.
fn laguna_parse(text: &str, thinking_open: bool, hints: Option<&ToolHints>) -> Parsed {
    let mut out = Parsed::default();

    // 1) split off the reasoning block (identical structure to qwen_parse)
    let rest = if let Some(i) = text.find(THINK_END) {
        let r = text[..i].trim();
        let r = r.strip_prefix(THINK).unwrap_or(r).trim();
        if !r.is_empty() {
            out.reasoning = Some(r.to_owned());
        }
        &text[i + THINK_END.len()..]
    } else if thinking_open || text.trim_start().starts_with(THINK) {
        // still inside the think block (streaming, or max_tokens mid-thought)
        let r = text.trim();
        let r = r.strip_prefix(THINK).unwrap_or(r).trim();
        if !r.is_empty() {
            out.reasoning = Some(r.to_owned());
        }
        return out;
    } else {
        text
    };

    // 2) tool_call blocks; content is everything outside them. No tools
    // declared -> no extraction: the syntax is just text the model produced.
    let Some(hints) = hints else {
        let content = rest.trim();
        if !content.is_empty() {
            out.content = Some(content.to_owned());
        }
        return out;
    };
    let mut content = String::new();
    let mut cur = rest;
    while let Some(s) = cur.find(TOOL_CALL) {
        content.push_str(&cur[..s]);
        let after = &cur[s + TOOL_CALL.len()..];
        // an unterminated block (max_tokens mid-call, or still generating)
        // still parses best-effort - but only closed blocks count complete
        let (block, next, closed) = match after.find(TOOL_CALL_END) {
            Some(e) => (&after[..e], &after[e + TOOL_CALL_END.len()..], true),
            None => (after, "", false),
        };
        if let Some(tc) = parse_laguna_call(block, hints) {
            out.tool_calls.push(tc);
            if closed {
                out.complete_calls = out.tool_calls.len();
            }
        }
        cur = next;
    }
    content.push_str(cur);
    let content = content.trim();
    if !content.is_empty() {
        out.content = Some(content.to_owned());
    }
    out
}

/// One laguna tool-call block (the text between `<tool_call>` and
/// `</tool_call>`): the function name runs bare from the opener to the first
/// `<arg_key>` (or the block's end for a zero-arg call).
fn parse_laguna_call(block: &str, hints: &ToolHints) -> Option<ToolCallRaw> {
    let name_end = block.find("<arg_key>").unwrap_or(block.len());
    let name = block[..name_end].trim();
    if name.is_empty() {
        return None;
    }

    let param_hints = hints.get(name);
    let mut args = serde_json::Map::new();
    let mut cur = &block[name_end..];
    while let Some(k) = cur.find("<arg_key>") {
        let after_k = &cur[k + "<arg_key>".len()..];
        let Some(k_end) = after_k.find("</arg_key>") else {
            break;
        };
        let key = after_k[..k_end].trim().to_owned();
        let after_key = &after_k[k_end + "</arg_key>".len()..];
        let Some(v) = after_key.find("<arg_value>") else {
            break;
        };
        let vstart = &after_key[v + "<arg_value>".len()..];
        // an unterminated value (max_tokens mid-call) is dropped rather than
        // guessed at - a half-emitted argument is worse than a missing one
        let Some(v_end) = vstart.find("</arg_value>") else {
            break;
        };
        let declared_string = param_hints.and_then(|h| h.get(&key)).copied();
        args.insert(key, coerce(&vstart[..v_end], declared_string));
        cur = &vstart[v_end + "</arg_value>".len()..];
    }

    Some(ToolCallRaw {
        name: name.to_owned(),
        arguments: Value::Object(args).to_string(),
    })
}

/// Hermes-style JSON tool calls (IBM Granite 4.1's template, verified against
/// the shipped `tokenizer.chat_template` in the official GGUF):
///
/// ```text
/// preamble text                                   - optional
/// <tool_call>
/// {"name": "NAME", "arguments": {...}}
/// </tool_call>                                    - repeated for parallel calls
/// ```
///
/// No reasoning region exists in this dialect, so there is no `thinking_open`
/// choreography and nothing is ever filed as reasoning. Arguments arrive as
/// real JSON, so `hints` is purely the tools-declared GATE here  -
/// there is no per-parameter type to coerce against.
fn json_tool_parse(text: &str, hints: Option<&ToolHints>) -> Parsed {
    let mut out = Parsed::default();

    // No tools declared -> no extraction: the syntax is just text the model
    // produced (a bench corpus full of fake tool markup must stay visible).
    let Some(hints) = hints else {
        let t = text.trim();
        out.content = (!t.is_empty()).then(|| t.to_owned());
        return out;
    };

    let mut content = String::new();
    let mut cur = text;

    // The UNWRAPPED opener, read back. `ToolSyntax::bare_trigger` arms the same
    // grammar on a bare `{"name": "` at turn start, so a call can legitimately
    // arrive with no `<tool_call>` around it and this is the only place that
    // becomes a `tool_calls` entry rather than a JSON blob in the chat.
    //
    // Turn start on both sides: the grammar's `fresh` guard and this leading
    // position are the same rule, so the two cannot disagree about which
    // objects are calls.
    match bare_lead(cur, hints) {
        BareLead::Call(tc, rest) => {
            out.tool_calls.push(tc);
            out.complete_calls = 1;
            cur = rest;
        }
        // Mid-generation, or truncated by max_tokens. Withhold it exactly as an
        // unterminated `<tool_call>` block below is withheld - showing a half
        // call as content would stream the blob we are here to prevent, and
        // under the grammar it cannot end this way anyway (`may_stop` is false
        // inside a call).
        BareLead::Partial => return out,
        BareLead::No => {}
    }
    while let Some(s) = cur.find(TOOL_CALL) {
        content.push_str(&cur[..s]);
        let after = &cur[s + TOOL_CALL.len()..];
        // an unterminated block (max_tokens mid-call, or still generating)
        // still parses best-effort - but only closed blocks count complete
        let (block, next, closed) = match after.find(TOOL_CALL_END) {
            Some(e) => (&after[..e], &after[e + TOOL_CALL_END.len()..], true),
            None => (after, "", false),
        };
        if let Some(tc) = parse_json_call(block) {
            out.tool_calls.push(tc);
            if closed {
                out.complete_calls = out.tool_calls.len();
            }
        }
        cur = next;
    }
    content.push_str(cur);
    let content = content.trim();
    if !content.is_empty() {
        out.content = Some(content.to_owned());
    }
    out
}

/// The JSON syntax's unwrapped opener - kept byte-identical to the grammar's
/// `constrained::JSON_BARE_OPEN`, which is what the model is constrained to
/// emit once it arms.
const JSON_BARE_OPEN: &str = "{\"name\": \"";

/// What sits at the very start of the turn.
enum BareLead<'a> {
    /// a complete unwrapped call, and whatever followed it
    Call(ToolCallRaw, &'a str),
    /// the opening of one, not finished yet
    Partial,
    /// ordinary content
    No,
}

/// Read a leading unwrapped call. Deliberately strict, because arming on this
/// shape is a read of intent rather than of markup: the text has to be the
/// opener (or a prefix of it), and the name has to be one the request actually
/// declared - the same alternation the grammar enforces, so a model quoting a
/// tool it does not have keeps its content.
fn bare_lead<'a>(text: &'a str, hints: &ToolHints) -> BareLead<'a> {
    let lead = text.trim_start();
    if lead.is_empty() {
        return BareLead::No;
    }
    // still typing the opener itself
    if JSON_BARE_OPEN.starts_with(lead) {
        return BareLead::Partial;
    }
    let Some(after) = lead.strip_prefix(JSON_BARE_OPEN) else {
        return BareLead::No;
    };
    // the name so far, up to its closing quote (all of `after` while unclosed)
    let typed = after.split('"').next().unwrap_or(after);
    if !hints.keys().any(|n| n.starts_with(typed)) {
        return BareLead::No;
    }
    // One JSON value off the front; `byte_offset` says where it ended, so a
    // call followed by prose keeps the prose.
    let mut stream = serde_json::Deserializer::from_str(lead).into_iter::<Value>();
    let Some(Ok(v)) = stream.next() else {
        return BareLead::Partial;
    };
    let rest = &lead[stream.byte_offset()..];
    let Some(name) = v.get("name").and_then(Value::as_str) else {
        return BareLead::No;
    };
    // `arguments` is required here, unlike the wrapped form: the grammar always
    // spells it, and a lone `{"name": ...}` is far likelier to be prose
    let Some(args) = v.get("arguments") else {
        return BareLead::No;
    };
    if !hints.contains_key(name) {
        return BareLead::No;
    }
    let arguments = match args {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    };
    BareLead::Call(
        ToolCallRaw {
            name: name.to_owned(),
            arguments,
        },
        rest,
    )
}

/// One `{"name": ..., "arguments": ...}` object. A half-emitted block is dropped
/// rather than guessed at - serde won't parse truncated JSON anyway, and a
/// partial argument object is worse than a missing call.
fn parse_json_call(block: &str) -> Option<ToolCallRaw> {
    let v: Value = serde_json::from_str(block.trim()).ok()?;
    let name = v.get("name")?.as_str()?.trim();
    if name.is_empty() {
        return None;
    }
    // the template emits `arguments` as an object, but tolerate the
    // JSON-encoded-string spelling too (it renders that when a caller's own
    // history carried arguments as a string)
    let arguments = match v.get("arguments") {
        Some(Value::String(s)) => s.clone(),
        Some(other) => other.to_string(),
        None => "{}".to_owned(),
    };
    Some(ToolCallRaw {
        name: name.to_owned(),
        arguments,
    })
}

/// Schema-typed reassembly: a declared-string parameter stays text verbatim;
/// anything else tries JSON (numbers, bools, objects, arrays - the template
/// renders those with `tojson`) and falls back to text.
///
/// Shared with the muse dialect, whose `render_atem` macro splits values the
/// same way (`v | tojson` for mappings/iterables, bare text otherwise).
pub(crate) fn coerce(val: &str, declared_string: Option<bool>) -> Value {
    if declared_string == Some(true) {
        return Value::String(val.to_owned());
    }
    serde_json::from_str::<Value>(val.trim()).unwrap_or_else(|_| Value::String(val.to_owned()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One arch string, two dialects. The marker pair is the whole test:
    /// granite 4.1 carries neither `<function=` nor `<think>`, granite 4.2
    /// carries both, and picking by `arch` alone sent 4.2's reasoning to the
    /// user as content.
    #[test]
    fn granite_dialect_comes_from_the_template() {
        // 4.1: Hermes-style JSON body, no reasoning region.
        let g41 = r#"{{- '<tool_call>\n' }}{{- tool | tojson }}{{- '\n</tool_call>' }}"#;
        assert_eq!(
            Dialect::for_arch_and_template("granite", Some(g41)),
            Dialect::JsonToolCall
        );
        // 4.2: XML body + a pre-opened think region.
        let g42 = r#"{{- '<tool_call>\n<function=' }}{{- '<parameter=' }}
                    {%- if add_generation_prompt %}{{- '<think>\n' }}{%- endif %}"#;
        assert_eq!(
            Dialect::for_arch_and_template("granite", Some(g42)),
            Dialect::QwenXml
        );
        // No template at all must not silently upgrade the dialect.
        assert_eq!(
            Dialect::for_arch_and_template("granite", None),
            Dialect::JsonToolCall
        );
        // Half a match is not a match: XML body without a think region, and a
        // think region without an XML body, both stay on the arch default.
        assert_eq!(
            Dialect::for_arch_and_template("granite", Some("<function=only")),
            Dialect::JsonToolCall
        );
        assert_eq!(
            Dialect::for_arch_and_template("granite", Some("<think>only")),
            Dialect::JsonToolCall
        );
        // Other families keep arch-keyed dispatch even with those markers.
        assert_eq!(
            Dialect::for_arch_and_template("gpt-oss", Some("<function=<think>")),
            Dialect::Harmony
        );
    }

    /// granite 4.2's prompt ends inside an open think region, so `thinking_open`
    /// must say so on the dialect the template selects - the QwenXml arm.
    #[test]
    fn granite_42_prompt_opens_thinking() {
        let d = Dialect::QwenXml;
        assert!(d.thinking_open("<|im_start|>assistant\n<think>\n"));
        // thinking off renders a pre-closed empty block and must read as closed
        assert!(!d.thinking_open("<|im_start|>assistant\n<think></think>"));
    }

    fn hints_weather() -> Option<ToolHints> {
        tool_hints(Some(&[serde_json::json!({"type":"function","function":{
            "name":"get_weather",
            "parameters":{"type":"object","properties":{
                "city":{"type":"string"},
                "days":{"type":"integer"},
                "units":{"type":"string"}
            }}
        }})]))
    }

    #[test]
    fn qwen_thinking_then_content() {
        // thinking mode: prompt opened <think>, so no opening tag in output
        let p = parse(
            Dialect::QwenXml,
            "the user wants brevity\n</think>\n\nParis.",
            true,
            None,
        );
        assert_eq!(p.reasoning.as_deref(), Some("the user wants brevity"));
        assert_eq!(p.content.as_deref(), Some("Paris."));
        assert_eq!(p.finish_reason(), "stop");
    }

    #[test]
    fn qwen_mid_thought_is_all_reasoning() {
        let p = parse(Dialect::QwenXml, "let me consider", true, None);
        assert_eq!(p.reasoning.as_deref(), Some("let me consider"));
        assert!(p.content.is_none());
    }

    #[test]
    fn qwen_non_thinking_is_all_content() {
        let p = parse(Dialect::QwenXml, "Paris.", false, None);
        assert_eq!(p.content.as_deref(), Some("Paris."));
        assert!(p.reasoning.is_none());
    }

    #[test]
    fn qwen_self_opened_think_block() {
        let p = parse(
            Dialect::QwenXml,
            "<think>\nhmm\n</think>\n\nDone.",
            false,
            None,
        );
        assert_eq!(p.reasoning.as_deref(), Some("hmm"));
        assert_eq!(p.content.as_deref(), Some("Done."));
    }

    #[test]
    fn qwen_single_tool_call_with_typed_args() {
        let t = "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n\
                 <parameter=days>\n3\n</parameter>\n</function>\n</tool_call>";
        let p = parse(Dialect::QwenXml, t, false, hints_weather().as_ref());
        assert_eq!(p.tool_calls.len(), 1);
        assert_eq!(p.tool_calls[0].name, "get_weather");
        let args: Value = serde_json::from_str(&p.tool_calls[0].arguments).unwrap();
        assert_eq!(args["city"], "Paris");
        assert_eq!(args["days"], 3);
        assert_eq!(p.finish_reason(), "tool_calls");
        assert!(p.content.is_none());
    }

    #[test]
    fn qwen_tool_call_inside_unclosed_think_is_a_call() {
        // The live dead-end this guards: the model wrote a well-formed call
        // inside a still-open <think> and stopped without </think>. The block
        // must extract as a CALL; the reasoning keeps only the prose.
        let t = "I should look this up.\n\n<tool_call>\n<function=get_weather>\n\
                 <parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>";
        let p = parse(Dialect::QwenXml, t, true, hints_weather().as_ref());
        assert_eq!(p.tool_calls.len(), 1);
        assert_eq!(p.tool_calls[0].name, "get_weather");
        assert_eq!(p.complete_calls, 1);
        assert_eq!(p.reasoning.as_deref(), Some("I should look this up."));
        assert!(p.content.is_none());
        assert_eq!(p.finish_reason(), "tool_calls");
    }

    #[test]
    fn qwen_tool_call_inside_closed_think_is_a_call() {
        // Same slip but the model then closes </think> and answers: the
        // in-think block still extracts, the content survives, and nothing
        // raw leaks into the reasoning text.
        let t = "Deciding.\n<tool_call>\n<function=get_weather>\n<parameter=city>\n\
                 Paris\n</parameter>\n</function>\n</tool_call>\n</think>\n\nHere you go.";
        let p = parse(Dialect::QwenXml, t, true, hints_weather().as_ref());
        assert_eq!(p.tool_calls.len(), 1);
        assert_eq!(p.reasoning.as_deref(), Some("Deciding."));
        assert_eq!(p.content.as_deref(), Some("Here you go."));
    }

    #[test]
    fn qwen_think_syntax_stays_text_without_tools() {
        // No tools declared: the syntax inside think is just text the model
        // produced (same rule as the content region).
        let t = "Hmm <tool_call>\n<function=x>\n</function>\n</tool_call> done";
        let p = parse(Dialect::QwenXml, t, true, None);
        assert!(p.tool_calls.is_empty());
        assert!(p.reasoning.as_deref().unwrap().contains("<tool_call>"));
    }

    #[test]
    fn qwen_string_param_never_coerced() {
        // "123" with a string-typed schema must stay a string
        let t = "<tool_call>\n<function=get_weather>\n<parameter=units>\n123\n</parameter>\n\
                 </function>\n</tool_call>";
        let p = parse(Dialect::QwenXml, t, false, hints_weather().as_ref());
        let args: Value = serde_json::from_str(&p.tool_calls[0].arguments).unwrap();
        assert_eq!(args["units"], "123");
    }

    #[test]
    fn qwen_multiline_param_value() {
        let t = "<tool_call>\n<function=get_weather>\n<parameter=city>\nline one\nline two\n\
                 </parameter>\n</function>\n</tool_call>";
        let p = parse(Dialect::QwenXml, t, false, hints_weather().as_ref());
        let args: Value = serde_json::from_str(&p.tool_calls[0].arguments).unwrap();
        assert_eq!(args["city"], "line one\nline two");
    }

    #[test]
    fn qwen_parallel_calls_with_preamble() {
        let t = "I'll check both cities.\n\n\
                 <tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>\n\
                 <tool_call>\n<function=get_weather>\n<parameter=city>\nBerlin\n</parameter>\n</function>\n</tool_call>";
        let p = parse(Dialect::QwenXml, t, false, hints_weather().as_ref());
        assert_eq!(p.tool_calls.len(), 2);
        assert_eq!(p.content.as_deref(), Some("I'll check both cities."));
        let b: Value = serde_json::from_str(&p.tool_calls[1].arguments).unwrap();
        assert_eq!(b["city"], "Berlin");
    }

    #[test]
    fn qwen_thinking_then_tool_call() {
        let t = "need the weather tool\n</think>\n\n\
                 <tool_call>\n<function=get_weather>\n<parameter=city>\nOslo\n</parameter>\n</function>\n</tool_call>";
        let p = parse(Dialect::QwenXml, t, true, hints_weather().as_ref());
        assert_eq!(p.reasoning.as_deref(), Some("need the weather tool"));
        assert_eq!(p.tool_calls.len(), 1);
        assert!(p.content.is_none());
    }

    #[test]
    fn complete_calls_counts_only_closed_blocks() {
        let closed = "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>";
        let open = "<tool_call>\n<function=get_weather>\n<parameter=city>\nBer";
        let p = parse(Dialect::QwenXml, closed, false, hints_weather().as_ref());
        assert_eq!((p.tool_calls.len(), p.complete_calls), (1, 1));
        let p = parse(
            Dialect::QwenXml,
            &format!("{closed}\n{open}"),
            false,
            hints_weather().as_ref(),
        );
        assert_eq!((p.tool_calls.len(), p.complete_calls), (2, 1));
        // harmony: a delimiter-terminated commentary call is complete, a
        // still-generating one is not
        let t = "<|channel|>commentary to=functions.a <|constrain|>json<|message|>{\"x\":1}<|end|>\
                 <|start|>assistant<|channel|>commentary to=functions.b <|constrain|>json<|message|>{\"y\":";
        let p = parse(Dialect::Harmony, t, false, Some(&ToolHints::new()));
        assert_eq!((p.tool_calls.len(), p.complete_calls), (2, 1));
    }

    #[test]
    fn qwen_unterminated_call_parses_complete_params() {
        // max_tokens mid-call: complete parameters still come through
        let t = "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n<parameter=da";
        let p = parse(Dialect::QwenXml, t, false, hints_weather().as_ref());
        assert_eq!(p.tool_calls.len(), 1);
        let args: Value = serde_json::from_str(&p.tool_calls[0].arguments).unwrap();
        assert_eq!(args["city"], "Paris");
    }

    #[test]
    fn plain_dialect_passthrough() {
        let p = parse(Dialect::Plain, "  hello  ", false, None);
        assert_eq!(p.content.as_deref(), Some("hello"));
    }

    #[test]
    fn no_tools_declared_keeps_tool_syntax_as_content() {
        // the benchmark regression this pins: a tools-free request whose model
        // imitates tool markup (synthetic coding corpora are full of it) must
        // stream that text verbatim - never parse calls the client can't get
        let t = "I'll check.\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n\
                 </parameter>\n</function>\n</tool_call>";
        let p = parse(Dialect::QwenXml, t, false, None);
        assert!(p.tool_calls.is_empty());
        assert_eq!(p.finish_reason(), "stop");
        assert!(p.content.as_deref().unwrap().contains("<tool_call>"));

        let t = "<tool_call>read<arg_key>file_path</arg_key><arg_value>src/x.rs</arg_value></tool_call>";
        let p = parse(Dialect::Laguna, t, false, None);
        assert!(p.tool_calls.is_empty());
        assert_eq!(p.content.as_deref(), Some(t));

        // harmony: an untooled commentary-with-target is user-visible preamble
        let t = "<|channel|>commentary to=functions.get_weather \
                 <|constrain|>json<|message|>{\"location\":\"Paris\"}";
        let p = parse(Dialect::Harmony, t, false, None);
        assert!(p.tool_calls.is_empty());
        assert_eq!(p.content.as_deref(), Some("{\"location\":\"Paris\"}"));
    }

    #[test]
    fn gemma_thought_then_content() {
        let p = parse(
            Dialect::GemmaChannel,
            "<|channel>thought\nlet me think\n<channel|>Stockholm",
            true,
            None,
        );
        assert_eq!(p.reasoning.as_deref(), Some("let me think"));
        assert_eq!(p.content.as_deref(), Some("Stockholm"));
    }

    #[test]
    fn gemma_unclosed_thought_is_all_reasoning() {
        let p = parse(
            Dialect::GemmaChannel,
            "<|channel>thought\nstill going",
            true,
            None,
        );
        assert_eq!(p.reasoning.as_deref(), Some("still going"));
        assert_eq!(p.content, None);
    }

    #[test]
    fn gemma_marker_prefix_is_ambiguous_not_content() {
        // mid-stream: the opening tag hasn't fully arrived - nothing may be
        // classified yet (a content emission here could never be retracted)
        for prefix in ["<", "<|chan", "<|channel>", "<|channel>thought"] {
            let p = parse(Dialect::GemmaChannel, prefix, true, None);
            assert_eq!(p.content, None, "{prefix:?} leaked as content");
            assert_eq!(p.reasoning, None);
        }
    }

    #[test]
    fn gemma_no_thought_is_content() {
        let p = parse(Dialect::GemmaChannel, "Hello there.", false, None);
        assert_eq!(p.content.as_deref(), Some("Hello there."));
        assert_eq!(p.reasoning, None);
    }

    #[test]
    fn gemma_preopened_thought_streams_as_reasoning() {
        // the prompt pre-opened the thought channel (g4_preopen): the very
        // first sampled tokens are reasoning even without a generated opener
        let p = parse(Dialect::GemmaChannel, "let me think", true, None);
        assert_eq!(p.reasoning.as_deref(), Some("let me think"));
        assert_eq!(p.content, None);
    }

    #[test]
    fn gemma_preopened_thought_close_flips_to_content() {
        let p = parse(
            Dialect::GemmaChannel,
            "let me think\n<channel|>Stockholm",
            true,
            None,
        );
        assert_eq!(p.reasoning.as_deref(), Some("let me think"));
        assert_eq!(p.content.as_deref(), Some("Stockholm"));
    }

    #[test]
    fn laguna_thinking_then_content() {
        // thinking mode: the prompt pre-opened <think>, so no opening tag
        let p = parse(
            Dialect::Laguna,
            "the user wants the capital</think>Paris.",
            true,
            None,
        );
        assert_eq!(p.reasoning.as_deref(), Some("the user wants the capital"));
        assert_eq!(p.content.as_deref(), Some("Paris."));
    }

    #[test]
    fn laguna_non_thinking_is_all_content() {
        // thinking off: the prompt pre-closed </think>; text is plain content
        let p = parse(Dialect::Laguna, "Paris.", false, None);
        assert_eq!(p.content.as_deref(), Some("Paris."));
        assert!(p.reasoning.is_none());
    }

    #[test]
    fn laguna_mid_thought_is_all_reasoning() {
        let p = parse(Dialect::Laguna, "let me consider", true, None);
        assert_eq!(p.reasoning.as_deref(), Some("let me consider"));
        assert!(p.content.is_none());
    }

    #[test]
    fn laguna_tool_call_with_typed_args() {
        let t = "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value>\
                 <arg_key>days</arg_key><arg_value>3</arg_value></tool_call>";
        let p = parse(Dialect::Laguna, t, false, hints_weather().as_ref());
        assert_eq!(p.tool_calls.len(), 1);
        assert_eq!(p.tool_calls[0].name, "get_weather");
        let args: Value = serde_json::from_str(&p.tool_calls[0].arguments).unwrap();
        assert_eq!(args["city"], "Paris");
        assert_eq!(args["days"], 3);
        assert_eq!(p.finish_reason(), "tool_calls");
        assert!(p.content.is_none());
        assert_eq!(p.complete_calls, 1);
    }

    #[test]
    fn laguna_string_param_never_coerced() {
        let t =
            "<tool_call>get_weather<arg_key>units</arg_key><arg_value>123</arg_value></tool_call>";
        let p = parse(Dialect::Laguna, t, false, hints_weather().as_ref());
        let args: Value = serde_json::from_str(&p.tool_calls[0].arguments).unwrap();
        assert_eq!(args["units"], "123");
    }

    #[test]
    fn laguna_thinking_then_parallel_calls_with_preamble() {
        let t = "need both cities</think>Checking both.\
                 <tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>\
                 <tool_call>get_weather<arg_key>city</arg_key><arg_value>Berlin</arg_value></tool_call>";
        let p = parse(Dialect::Laguna, t, true, hints_weather().as_ref());
        assert_eq!(p.reasoning.as_deref(), Some("need both cities"));
        assert_eq!(p.content.as_deref(), Some("Checking both."));
        assert_eq!(p.tool_calls.len(), 2);
        assert_eq!(p.complete_calls, 2);
        let b: Value = serde_json::from_str(&p.tool_calls[1].arguments).unwrap();
        assert_eq!(b["city"], "Berlin");
    }

    #[test]
    fn laguna_unterminated_call_keeps_complete_args_only() {
        // max_tokens mid-value: the finished arg survives, the half one drops
        let t = "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value>\
                 <arg_key>days</arg_key><arg_value>3";
        let p = parse(Dialect::Laguna, t, false, hints_weather().as_ref());
        assert_eq!((p.tool_calls.len(), p.complete_calls), (1, 0));
        let args: Value = serde_json::from_str(&p.tool_calls[0].arguments).unwrap();
        assert_eq!(args["city"], "Paris");
        assert!(args.get("days").is_none());
    }

    #[test]
    fn laguna_json_object_arg_survives_tojson() {
        // non-string values render through tojson - they must come back typed
        let t = "<tool_call>configure<arg_key>opts</arg_key>\
                 <arg_value>{\"depth\": 2, \"verbose\": true}</arg_value></tool_call>";
        let p = parse(Dialect::Laguna, t, false, Some(&ToolHints::new()));
        let args: Value = serde_json::from_str(&p.tool_calls[0].arguments).unwrap();
        assert_eq!(args["opts"]["depth"], 2);
        assert_eq!(args["opts"]["verbose"], true);
    }

    #[test]
    fn laguna_thinking_open_probe() {
        // laguna pre-opens a bare <think>; the pre-closed </think> (thinking
        // off) must not read as open even though it contains "think>"
        assert!(Dialect::Laguna.thinking_open("<system>...</system>\n<assistant><think>"));
        assert!(!Dialect::Laguna.thinking_open("<system>...</system>\n<assistant></think>"));
        // qwen still wants the newline form; gemma is suffix-absence
        assert!(Dialect::QwenXml.thinking_open("...<think>\n"));
        assert!(!Dialect::GemmaChannel.thinking_open("...<channel|>"));
    }

    // ---- granite 4.1 / the shared Hermes-style JSON tool-call dialect ----

    #[test]
    fn json_tool_plain_answer_is_content() {
        let p = parse(
            Dialect::JsonToolCall,
            "The capital of France is Paris.",
            false,
            None,
        );
        assert_eq!(
            p.content.as_deref(),
            Some("The capital of France is Paris.")
        );
        assert!(p.reasoning.is_none());
        assert_eq!(p.finish_reason(), "stop");
    }

    // ------------------------------ the unwrapped call, read back --

    /// The measured granite-vision output: a whole, correct call with no
    /// `<tool_call>` around it. The grammar's second opener makes this shape
    /// well-formed by construction; this is the read-back half of that pair.
    #[test]
    fn bare_json_call_at_turn_start_is_a_call() {
        let t = "{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}";
        let p = parse(Dialect::JsonToolCall, t, false, hints_weather().as_ref());
        assert_eq!(p.tool_calls.len(), 1);
        assert_eq!(p.tool_calls[0].name, "get_weather");
        assert_eq!(p.complete_calls, 1);
        assert_eq!(p.finish_reason(), "tool_calls");
        assert!(p.content.is_none(), "the blob must not also be content");
    }

    /// The dispatch grammar RELEASES after a call, so prose can follow one.
    #[test]
    fn bare_json_call_keeps_what_follows_it() {
        let t = "{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\nChecking now.";
        let p = parse(Dialect::JsonToolCall, t, false, hints_weather().as_ref());
        assert_eq!(p.tool_calls.len(), 1);
        assert_eq!(p.content.as_deref(), Some("Checking now."));
    }

    /// Mid-stream. Every prefix of a bare call has to withhold rather than
    /// stream - this is the whole point of doing it at generation time, and a
    /// leak here would put the blob back on the wire delta by delta.
    #[test]
    fn a_bare_call_in_flight_shows_nothing() {
        let full = "{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}";
        for cut in 1..full.len() {
            let p = parse(
                Dialect::JsonToolCall,
                &full[..cut],
                false,
                hints_weather().as_ref(),
            );
            assert!(
                p.content.is_none(),
                "prefix {:?} leaked as content",
                &full[..cut]
            );
            assert_eq!(p.complete_calls, 0, "nothing is complete yet");
        }
    }

    /// The name gate, which is what keeps the read of intent honest: an object
    /// naming a tool this request never declared is just an object.
    #[test]
    fn bare_json_object_naming_an_unknown_tool_is_content() {
        let t = "{\"name\": \"launch_missiles\", \"arguments\": {\"city\": \"Paris\"}}";
        let p = parse(Dialect::JsonToolCall, t, false, hints_weather().as_ref());
        assert!(p.tool_calls.is_empty());
        assert_eq!(p.content.as_deref(), Some(t));
    }

    /// Mirrors the grammar's `fresh` guard: past turn start the same bytes are
    /// a model writing about a tool, and both halves have to agree about that.
    #[test]
    fn a_bare_object_after_prose_is_content() {
        let t = "You would send {\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}";
        let p = parse(Dialect::JsonToolCall, t, false, hints_weather().as_ref());
        assert!(p.tool_calls.is_empty());
        assert_eq!(p.content.as_deref(), Some(t));
    }

    /// A JSON answer that is not call-shaped keeps being an answer.
    #[test]
    fn a_leading_json_object_that_is_not_a_call_is_content() {
        let t = "{\"city\": \"Paris\", \"temp\": 14}";
        let p = parse(Dialect::JsonToolCall, t, false, hints_weather().as_ref());
        assert!(p.tool_calls.is_empty());
        assert_eq!(p.content.as_deref(), Some(t));
    }

    /// `name` alone is not enough - the grammar always spells `arguments`, so
    /// the read-back requires it too.
    #[test]
    fn a_bare_object_without_arguments_is_content() {
        let t = "{\"name\": \"get_weather\"}";
        let p = parse(Dialect::JsonToolCall, t, false, hints_weather().as_ref());
        assert!(p.tool_calls.is_empty());
        assert_eq!(p.content.as_deref(), Some(t));
    }

    /// The tools-declared gate still rules everything: with no tools, the same
    /// bytes are text the model produced (bench-corpus rule).
    #[test]
    fn a_bare_call_without_tools_declared_stays_text() {
        let t = "{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}";
        let p = parse(Dialect::JsonToolCall, t, false, None);
        assert!(p.tool_calls.is_empty());
        assert_eq!(p.content.as_deref(), Some(t));
    }

    /// Both openers in one turn: the bare one at the start, then a wrapped one
    /// the model writes properly afterwards.
    #[test]
    fn bare_then_wrapped_both_land() {
        let t = "{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n\
                 and Berlin:\n\
                 <tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Berlin\"}}\n</tool_call>";
        let p = parse(Dialect::JsonToolCall, t, false, hints_weather().as_ref());
        assert_eq!(p.tool_calls.len(), 2);
        assert_eq!(p.complete_calls, 2);
        assert_eq!(p.content.as_deref(), Some("and Berlin:"));
    }

    #[test]
    fn json_tool_call_arrives_typed() {
        // granite's template emits the whole call as one JSON object
        let t = "<tool_call>\n{\"name\": \"get_weather\", \
                 \"arguments\": {\"city\": \"Paris\", \"days\": 3}}\n</tool_call>";
        let p = parse(Dialect::JsonToolCall, t, false, hints_weather().as_ref());
        assert_eq!(p.tool_calls.len(), 1);
        assert_eq!(p.tool_calls[0].name, "get_weather");
        let args: Value = serde_json::from_str(&p.tool_calls[0].arguments).unwrap();
        // types come straight from the model's JSON - no schema coercion needed
        assert_eq!(args["city"], "Paris");
        assert_eq!(args["days"], 3);
        assert_eq!(p.complete_calls, 1);
        assert_eq!(p.finish_reason(), "tool_calls");
        assert!(p.content.is_none());
    }

    #[test]
    fn json_tool_parallel_calls_with_preamble() {
        let t = "Checking both.\n\
                 <tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>\n\
                 <tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Berlin\"}}\n</tool_call>";
        let p = parse(Dialect::JsonToolCall, t, false, hints_weather().as_ref());
        assert_eq!(p.content.as_deref(), Some("Checking both."));
        assert_eq!((p.tool_calls.len(), p.complete_calls), (2, 2));
        let b: Value = serde_json::from_str(&p.tool_calls[1].arguments).unwrap();
        assert_eq!(b["city"], "Berlin");
    }

    #[test]
    fn json_tool_truncated_call_is_dropped_not_guessed() {
        // max_tokens mid-object: serde can't parse it, and half an argument
        // set is worse than no call at all
        let t = "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Par";
        let p = parse(Dialect::JsonToolCall, t, false, hints_weather().as_ref());
        assert_eq!((p.tool_calls.len(), p.complete_calls), (0, 0));
    }

    #[test]
    fn json_tool_no_tools_declared_keeps_markup_visible() {
        // the gate: a request with no tools must not eat tool-shaped text
        let t = "<tool_call>\n{\"name\": \"x\", \"arguments\": {}}\n</tool_call>";
        let p = parse(Dialect::JsonToolCall, t, false, None);
        assert!(p.tool_calls.is_empty());
        assert_eq!(p.content.as_deref(), Some(t));
    }

    #[test]
    fn json_tool_string_arguments_pass_through() {
        // some histories carry arguments already JSON-encoded as a string
        let t = "<tool_call>\n{\"name\": \"f\", \"arguments\": \"{\\\"a\\\": 1}\"}\n</tool_call>";
        let p = parse(Dialect::JsonToolCall, t, false, Some(&ToolHints::new()));
        assert_eq!(p.tool_calls[0].arguments, "{\"a\": 1}");
    }

    #[test]
    fn json_tool_has_no_thinking_region() {
        // granite's template opens none - and a `<think>`-looking prompt tail
        // must not accidentally flip it on
        assert!(!Dialect::JsonToolCall.thinking_open("...<think>\n"));
        let p = parse(Dialect::JsonToolCall, "plain</think>text", false, None);
        assert!(p.reasoning.is_none());
        assert_eq!(p.content.as_deref(), Some("plain</think>text"));
    }

    #[test]
    fn holdback_partial_markers() {
        let m = Dialect::QwenXml.content_markers();
        assert_eq!(holdback("text <tool_c", m), 7);
        assert_eq!(holdback("text </thi", m), 5);
        assert_eq!(holdback("text <", m), 1);
        assert_eq!(holdback("no tag here", m), 0);
        // a COMPLETE marker is not a partial (parser already handled it)
        assert_eq!(holdback("text <tool_call>", m), 0);
        assert_eq!(holdback("", m), 0);
        assert_eq!(holdback("x", &[]), 0);
    }
}
