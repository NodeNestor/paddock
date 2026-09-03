//! Server-side context management (design:
//!
//! The Anthropic `context_management.edits` contract, applied to the raw
//! Anthropic-shaped message array before template render. Pure: this module
//! never touches the engine or the template - callers hand it the messages
//! and a `count` closure (render + tokenize) and get back the edited
//! messages plus the `applied_edits` report the response carries. All of it
//! is opt-in: no `context_management` in the request, nothing here runs, and
//! an overflowing prompt stays the loud error it always was.
//!
//! Strategies (beta `context-management-2025-06-27`, accepted with or
//! without the header - real-API clients always send it, and requiring it
//! adds a failure mode without adding conformance):
//! - `clear_thinking_20251015`: strip thinking blocks from all but the last
//!   N assistant thinking-turns. No trigger - applies when configured.
//! - `clear_tool_uses_20250919`: past a trigger, replace the OLDEST tool
//!   results (and optionally tool inputs) with placeholders, keeping the
//!   most recent `keep` pairs and everything in `exclude_tools`.
//!   `clear_at_least` skips the whole edit when it would not reclaim enough
//!   - pointless prompt-cache invalidation is the thing that knob exists to
//!     prevent (our radix cache has the same economics as Anthropic's).
//! - `compact_20260112` (beta `compact-2026-01-12`): past an input_tokens
//!   trigger, a summarization generation replaces the "compact span" with a
//!   `compaction` content block. This module owns the pure parts - trigger
//!   math, the span/tail split, the resend rewrite, the post-compaction
//!   message build; the two-iteration orchestration lives in the serving
//!   layer (messages.rs), which is the only place that can run a generation.
//!
//! Spec points pinned here because the public docs leave them open (wire
//! shapes are pinned against anthropic-sdk-python 0.120.2 - see
//! - `clear_at_least` unmet => the edit applies nothing this request (rather
//!   than clearing into `keep`); revisit against the SDK if it ever differs.
//! - a cleared tool result is REPLACED (content -> one text block of
//!   placeholder), never removed - templates keep their structure and the
//!   model is told something was there.
//! - re-clearing an already-cleared pair reclaims ~0 tokens, so repeated
//!   requests converge to no-ops via `clear_at_least` naturally.
//! - the compact SPAN is everything before the pending turn: the tail (kept
//!   raw) starts at the last user message holding no tool_result blocks -
//!   the user's current request plus any tool round-trips under it. This
//!   keeps tool_use/tool_result pairs intact and makes the in-request
//!   continuation and the resend render the identical token stream, which is
//!   what turns every post-compaction turn into a radix-cache hit.
//! - compaction produces no applied_edits entry (the SDK's AppliedEdit union
//!   has only the two clear strategies); its report is the compaction block
//!   itself plus usage.iterations.
//! - a compaction block with null content is a FAILED compaction and
//!   round-trips as a no-op (SDK docstring) - we strip the block and keep
//!   the conversation uncompacted.

use serde_json::{Value, json};

/// What a cleared tool result reads as. The model is told the content was
/// removed; the exact wording is ours (Anthropic does not publish theirs).
pub const CLEARED_RESULT: &str = "[Tool result cleared by context management]";

/// Lead-in the model sees before a compaction summary at render time. The
/// summary rides user-voiced inside the pending turn's first user message -
/// the one shape every template family accepts (an assistant-first or
/// doubled-user opening breaks strict-alternation templates like gemma's).
pub const COMPACTION_FRAME: &str =
    "Summary of the earlier conversation (compacted to save context):\n\n";

/// Default summarization instructions (callers replace them verbatim via the
/// edit's `instructions` field). Anthropic's server-side prompt is
/// unpublished; this is modeled on the structure of their SDK's client-side
/// compaction prompt. The pending user turn survives raw in the tail, so the
/// summary only has to carry the PAST.
pub const DEFAULT_COMPACT_INSTRUCTIONS: &str = "\
The conversation above is about to be compacted: everything before the \
current request will be replaced by the summary you write now. Write a \
structured, self-contained summary that lets the conversation resume \
seamlessly: the user's goals and constraints, what has been done or decided \
so far and why, key facts and results worth carrying forward (including \
important tool results), errors already resolved, and what remains open. Be \
concise but complete - err toward keeping anything that would otherwise \
have to be rediscovered. Reply with the summary only.";

/// Whole-conversation variant (the Responses `compaction_trigger` item and
/// the standalone `/v1/responses/compact` endpoint): nothing survives raw,
/// so the summary must carry the pending thread too.
pub const DEFAULT_COMPACT_ALL_INSTRUCTIONS: &str = "\
The conversation above is about to be archived: it will be replaced entirely \
by the summary you write now. Write a structured, self-contained summary that \
lets the conversation resume seamlessly: the user's goals and constraints, \
what has been done or decided so far and why, key facts and results worth \
carrying forward (including important tool results), errors already resolved, \
and what remains open. Be concise but complete - err toward keeping anything \
that would otherwise have to be rediscovered. Reply with the summary only.";

#[derive(Debug, Clone, PartialEq)]
pub enum Threshold {
    InputTokens(u64),
    ToolUses(u64),
}

fn parse_threshold(v: &Value, field: &str) -> Result<Threshold, String> {
    let ty = v
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or("input_tokens");
    let val = v
        .get("value")
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("context_management: {field}.value must be a positive integer"))?;
    match ty {
        "input_tokens" => Ok(Threshold::InputTokens(val)),
        "tool_uses" => Ok(Threshold::ToolUses(val)),
        other => Err(format!(
            "context_management: invalid {field}.type {other:?}"
        )),
    }
}

#[derive(Debug, Clone)]
pub enum Edit {
    ClearThinking {
        /// None = keep all (no-op edit); Some(n) = keep last n thinking turns.
        keep_turns: Option<u64>,
    },
    ClearToolUses {
        trigger: Threshold,
        /// Most recent tool_use/result pairs preserved.
        keep: u64,
        clear_at_least: Option<Threshold>,
        exclude_tools: Vec<String>,
        clear_tool_inputs: bool,
    },
    Compact {
        /// input_tokens only (SDK: BetaInputTokensTriggerParam). Their default
        /// is 150000 with a documented 50k floor; we accept smaller values -
        /// a 32k-window local server needs compaction at ~24k (documented
        /// deviation, a superset the SDKs never notice).
        trigger: u64,
        /// Replaces DEFAULT_COMPACT_INSTRUCTIONS verbatim when set.
        instructions: Option<String>,
        /// Stop with stop_reason "compaction" right after the summary instead
        /// of continuing into the real generation.
        pause: bool,
    },
}

#[derive(Debug, Clone)]
pub struct Config {
    pub edits: Vec<Edit>,
}

/// Parse the request's `context_management` value. Unknown strategies and
/// malformed configs are loud 400s - a control the caller set is never
/// silently ignored.
pub fn parse(v: &Value) -> Result<Config, String> {
    let edits_v = v
        .get("edits")
        .and_then(Value::as_array)
        .ok_or("context_management.edits must be an array")?;
    let mut edits = Vec::new();
    for (i, e) in edits_v.iter().enumerate() {
        let ty = e
            .get("type")
            .and_then(Value::as_str)
            .ok_or("context_management edit needs a type")?;
        match ty {
            "clear_thinking_20251015" => {
                // documented combination rule: clear_thinking must come first
                if i != 0 {
                    return Err(
                        "context_management: clear_thinking_20251015 must be listed first in edits"
                            .into(),
                    );
                }
                let keep_turns = match e.get("keep") {
                    // local default: keep all. The Anthropic per-model default
                    // table is about their models; keep-all is the
                    // cache-friendly default here and matches 4.6+/5 behavior.
                    None => None,
                    Some(Value::String(s)) if s == "all" => None,
                    Some(k) => {
                        let ty = k.get("type").and_then(Value::as_str);
                        if ty != Some("thinking_turns") {
                            return Err(
                                "context_management: clear_thinking keep must be \"all\" or \
                                 {type: \"thinking_turns\", value: N}"
                                    .into(),
                            );
                        }
                        let n = k
                            .get("value")
                            .and_then(Value::as_u64)
                            .filter(|&n| n > 0)
                            .ok_or("context_management: clear_thinking keep.value must be > 0")?;
                        Some(n)
                    }
                };
                edits.push(Edit::ClearThinking { keep_turns });
            }
            "clear_tool_uses_20250919" => {
                let trigger = match e.get("trigger") {
                    Some(t) => parse_threshold(t, "trigger")?,
                    None => Threshold::InputTokens(100_000), // documented default
                };
                let keep = match e.get("keep") {
                    Some(k) => {
                        if k.get("type").and_then(Value::as_str) != Some("tool_uses") {
                            return Err(
                                "context_management: clear_tool_uses keep.type must be \"tool_uses\""
                                    .into(),
                            );
                        }
                        k.get("value")
                            .and_then(Value::as_u64)
                            .ok_or("context_management: keep.value must be an integer")?
                    }
                    None => 3, // documented default
                };
                let clear_at_least = e
                    .get("clear_at_least")
                    .map(|c| parse_threshold(c, "clear_at_least"))
                    .transpose()?;
                let exclude_tools = e
                    .get("exclude_tools")
                    .and_then(Value::as_array)
                    .map(|a| {
                        a.iter()
                            .filter_map(Value::as_str)
                            .map(str::to_owned)
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default();
                let clear_tool_inputs = e
                    .get("clear_tool_inputs")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                edits.push(Edit::ClearToolUses {
                    trigger,
                    keep,
                    clear_at_least,
                    exclude_tools,
                    clear_tool_inputs,
                });
            }
            "compact_20260112" => {
                if edits.iter().any(|e| matches!(e, Edit::Compact { .. })) {
                    return Err(
                        "context_management: only one compact_20260112 edit is allowed".into(),
                    );
                }
                let trigger = match e.get("trigger") {
                    None | Some(Value::Null) => 150_000, // documented default
                    Some(t) => match parse_threshold(t, "trigger")? {
                        Threshold::InputTokens(v) => v,
                        Threshold::ToolUses(_) => {
                            return Err("context_management: compact_20260112 trigger must be \
                                        {type: \"input_tokens\", value: N}"
                                .into());
                        }
                    },
                };
                let instructions = match e.get("instructions") {
                    None | Some(Value::Null) => None,
                    Some(Value::String(s)) => Some(s.clone()),
                    Some(_) => {
                        return Err(
                            "context_management: compact_20260112 instructions must be a string"
                                .into(),
                        );
                    }
                };
                let pause = e
                    .get("pause_after_compaction")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                edits.push(Edit::Compact {
                    trigger,
                    instructions,
                    pause,
                });
            }
            other => return Err(format!("context_management: unknown edit type {other:?}")),
        }
    }
    Ok(Config { edits })
}

/// A fired compact_20260112 edit, for the serving layer to orchestrate (the
/// pure core cannot run the summarization generation).
pub struct CompactFire {
    pub instructions: Option<String>,
    pub pause: bool,
}

/// One applied edit for the response report + the running token count after it.
pub struct Applied {
    /// `applied_edits` entries, Anthropic shape. Compaction never appears
    /// here (SDK pin: the AppliedEdit union has only the clear strategies).
    pub edits: Vec<Value>,
    /// Token count of the final (possibly edited) prompt.
    pub final_tokens: usize,
    /// Set when a compact_20260112 edit's trigger fired over a non-empty
    /// span; the caller runs the summarization pass.
    pub compact: Option<CompactFire>,
}

/// The compact span/tail split: the tail (kept raw) starts at the last user
/// message holding no tool_result blocks - the pending request plus any tool
/// round-trips under it. Everything before is the span a compaction
/// summarizes away. 0 = no span, nothing to compact.
pub fn compact_tail_start(messages: &[Value]) -> usize {
    messages
        .iter()
        .rposition(|m| m.get("role").and_then(Value::as_str) == Some("user") && !has_tool_result(m))
        .unwrap_or(0)
}

fn has_tool_result(m: &Value) -> bool {
    m.get("content")
        .and_then(Value::as_array)
        .is_some_and(|blocks| {
            blocks
                .iter()
                .any(|b| b.get("type").and_then(Value::as_str) == Some("tool_result"))
        })
}

/// Prepend the framed summary into the conversation's first message. The
/// summary rides user-voiced inside the tail's first user message so every
/// template family renders it (see COMPACTION_FRAME); a tail that does not
/// open with a plain user message (hand-crafted resends) gets a fresh user
/// message instead.
fn prepend_summary(out: &mut Vec<Value>, summary: &str) {
    let text = json!({"type": "text", "text": format!("{COMPACTION_FRAME}{summary}")});
    let user_first = out
        .first()
        .and_then(|m| m.get("role"))
        .and_then(Value::as_str)
        == Some("user");
    if !user_first {
        out.insert(0, json!({"role": "user", "content": [text]}));
        return;
    }
    match out[0].get_mut("content") {
        Some(Value::Array(blocks)) => blocks.insert(0, text),
        Some(v) => {
            // string content (the common plain-message shape): wrap both
            // into text blocks so nothing about the original is lost
            let orig = v.take();
            *v = match orig {
                Value::String(s) => json!([text, {"type": "text", "text": s}]),
                _ => json!([text]),
            };
        }
        None => out[0]["content"] = json!([text]),
    }
}

/// The post-compaction conversation the real generation runs over:
/// [tail with the framed summary prepended]. A later resend of the response
/// (compaction block + generated blocks appended by the client) rewrites to
/// the identical prefix via `resend_rewrite`, so every subsequent turn is a
/// radix-cache hit over this exact token stream.
pub fn compacted_messages(messages: &[Value], summary: &str) -> Vec<Value> {
    let tail_start = compact_tail_start(messages);
    let mut out = messages[tail_start..].to_vec();
    prepend_summary(&mut out, summary);
    out
}

/// Render-time semantics of a round-tripped `compaction` block (active on
/// every request, config or not - a request carrying our own response's
/// content must always be valid). Everything before the last compaction
/// block collapses into its summary: drop the messages before the pending
/// turn that preceded it, drop blocks before the block inside its message
/// (spec: "resend drops blocks before compaction"), and prepend the framed
/// summary exactly as `compacted_messages` did in-request. A null-content
/// block is a FAILED compaction and is a no-op: stripped, nothing dropped.
/// Returns None when there is no compaction block (the common case, borrow
/// the original).
pub fn resend_rewrite(messages: &[Value]) -> Option<Vec<Value>> {
    let mut pos: Option<(usize, usize, Option<String>)> = None;
    for (mi, m) in messages.iter().enumerate() {
        if let Some(blocks) = m.get("content").and_then(Value::as_array) {
            for (bi, b) in blocks.iter().enumerate() {
                if b.get("type").and_then(Value::as_str) == Some("compaction") {
                    pos = Some((
                        mi,
                        bi,
                        b.get("content").and_then(Value::as_str).map(str::to_owned),
                    ));
                }
            }
        }
    }
    let (k, j, summary) = pos?;
    let mut out: Vec<Value>;
    match summary.filter(|s| !s.trim().is_empty()) {
        Some(s) => {
            let tail_start = compact_tail_start(&messages[..k]);
            out = messages[tail_start..].to_vec();
            let k_local = k - tail_start;
            if let Some(blocks) = out[k_local]
                .get_mut("content")
                .and_then(Value::as_array_mut)
            {
                blocks.drain(..=j);
            }
            strip_compaction_blocks(&mut out);
            prepend_summary(&mut out, &s);
        }
        None => {
            out = messages.to_vec();
            strip_compaction_blocks(&mut out);
        }
    }
    Some(out)
}

/// Remove every compaction block (and any message emptied by that). After a
/// rewrite the surviving ones are stale no-ops, and the message converter
/// rejects unknown block types loudly.
fn strip_compaction_blocks(out: &mut Vec<Value>) {
    for m in out.iter_mut() {
        if let Some(blocks) = m.get_mut("content").and_then(Value::as_array_mut) {
            blocks.retain(|b| b.get("type").and_then(Value::as_str) != Some("compaction"));
        }
    }
    out.retain(|m| {
        m.get("content")
            .and_then(Value::as_array)
            .is_none_or(|blocks| !blocks.is_empty())
    });
}

// ── OpenAI Responses dialect (item-level) ───────────────────────────────────
//
// Same algebra as the Anthropic half, applied to Responses INPUT ITEMS before
// they are converted to chat messages. Wire shapes pinned against openai
// 2.53.0:
// - config: `context_management: [{"type": "compaction", "compact_threshold"}]`
//   (compaction is the only entry type; no instructions/pause knobs).
// - the compaction item is `{"id", "type": "compaction", "encrypted_content"}`
//   with encrypted_content required - ours carries the plaintext summary in
//   that field so unmodified SDK types round-trip it (no encryption theater
//   on a local box; documented deviation).
// - `{"type": "compaction_trigger"}` must be the FINAL input item and forces
//   a compact-now (the whole conversation, no tail split - the caller is
//   archiving, not asking).
// - an empty encrypted_content is the failed-compaction no-op (the twin of
//   the Anthropic null-content pin): stripped, nothing dropped.

/// Absent/null `compact_threshold` default. OpenAI does not publish theirs;
/// mirroring the Anthropic-dialect default keeps the two dialects symmetric,
/// and (same documented deviation) any positive value is accepted.
pub const OA_DEFAULT_THRESHOLD: u64 = 150_000;

/// Parse the Responses `context_management` array. Returns the compaction
/// threshold when a compaction entry is configured.
pub fn oa_parse(entries: &[Value]) -> Result<Option<u64>, String> {
    let mut threshold = None;
    for e in entries {
        match e.get("type").and_then(Value::as_str) {
            Some("compaction") => {
                if threshold.is_some() {
                    return Err("context_management: only one compaction entry is allowed".into());
                }
                threshold = Some(match e.get("compact_threshold") {
                    None | Some(Value::Null) => OA_DEFAULT_THRESHOLD,
                    Some(v) => v.as_u64().filter(|&n| n > 0).ok_or(
                        "context_management: compact_threshold must be a positive integer",
                    )?,
                });
            }
            other => {
                return Err(format!(
                    "context_management: unknown entry type {:?} (only \"compaction\" is served)",
                    other.unwrap_or("missing")
                ));
            }
        }
    }
    Ok(threshold)
}

/// A plain user message item (`type` defaults to "message" on the wire).
/// pub(crate): the standalone compact endpoint echoes exactly these items.
pub(crate) fn oa_user_message(it: &Value) -> bool {
    matches!(
        it.get("type").and_then(Value::as_str),
        None | Some("message")
    ) && it.get("role").and_then(Value::as_str) == Some("user")
}

/// Item-level twin of `compact_tail_start`: the tail starts at the last user
/// message item - the pending request plus its function_call/output
/// round-trips (tool results are separate items in this dialect, so every
/// user message qualifies). 0 = no span, nothing to compact.
pub fn oa_tail_start(items: &[Value]) -> usize {
    items.iter().rposition(oa_user_message).unwrap_or(0)
}

/// Prepend the framed summary into the first item, user-voiced (same
/// template-safety reasoning as `prepend_summary`). The trailing blank line
/// matters: the Responses converter flattens content parts with no
/// separator, so without it the summary would glue onto the user's text.
fn oa_prepend_summary(out: &mut Vec<Value>, summary: &str) {
    let text = json!({"type": "input_text", "text": format!("{COMPACTION_FRAME}{summary}\n\n")});
    if !out.first().is_some_and(oa_user_message) {
        out.insert(
            0,
            json!({"type": "message", "role": "user", "content": [text]}),
        );
        return;
    }
    match out[0].get_mut("content") {
        Some(Value::Array(parts)) => parts.insert(0, text),
        Some(v) => {
            let orig = v.take();
            *v = match orig {
                Value::String(s) => json!([text, {"type": "input_text", "text": s}]),
                _ => json!([text]),
            };
        }
        None => out[0]["content"] = json!([text]),
    }
}

/// The post-compaction item list iteration 2 runs over: tail + framed
/// summary. The resend rewrite reproduces this exact list, so every
/// post-compaction turn is a radix hit (the same invariant as the
/// Anthropic-dialect pair, unit-asserted below).
pub fn oa_compacted_items(items: &[Value], summary: &str) -> Vec<Value> {
    let mut out = items[oa_tail_start(items)..].to_vec();
    oa_prepend_summary(&mut out, summary);
    out
}

/// Render-time semantics of a round-tripped compaction ITEM (active on every
/// request, config or not). Everything before the last compaction item's
/// tail collapses into its summary; the item itself (and stale earlier ones)
/// is consumed. An empty/missing encrypted_content is a failed compaction:
/// strip-only no-op. None = no compaction item (borrow the original).
pub fn oa_resend_rewrite(items: &[Value]) -> Option<Vec<Value>> {
    let k = items
        .iter()
        .rposition(|it| it.get("type").and_then(Value::as_str) == Some("compaction"))?;
    let summary = items[k]
        .get("encrypted_content")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_owned);
    let mut out: Vec<Value> = match &summary {
        Some(_) => {
            let tail_start = oa_tail_start(&items[..k]);
            items[tail_start..].to_vec()
        }
        None => items.to_vec(),
    };
    out.retain(|it| it.get("type").and_then(Value::as_str) != Some("compaction"));
    if let Some(s) = &summary {
        oa_prepend_summary(&mut out, s);
    }
    Some(out)
}

/// Extract a `compaction_trigger` input item. The SDK pins it as the final
/// input item; anywhere else is a loud error. Returns true (with the item
/// removed) when the caller should run a compact-now.
pub fn oa_take_trigger(items: &mut Vec<Value>) -> Result<bool, String> {
    let Some(pos) = items
        .iter()
        .position(|it| it.get("type").and_then(Value::as_str) == Some("compaction_trigger"))
    else {
        return Ok(false);
    };
    if pos != items.len() - 1 {
        return Err("a compaction_trigger must be the final input item".into());
    }
    items.pop();
    Ok(true)
}

/// A tool_use block's location + identity, in conversation order.
struct ToolUse {
    msg: usize,
    block: usize,
    id: String,
    name: String,
}

/// Every tool_use block in assistant messages, oldest first.
fn tool_uses(messages: &[Value]) -> Vec<ToolUse> {
    let mut out = Vec::new();
    for (mi, m) in messages.iter().enumerate() {
        if m.get("role").and_then(Value::as_str) != Some("assistant") {
            continue;
        }
        let Some(blocks) = m.get("content").and_then(Value::as_array) else {
            continue;
        };
        for (bi, b) in blocks.iter().enumerate() {
            if b.get("type").and_then(Value::as_str) == Some("tool_use") {
                out.push(ToolUse {
                    msg: mi,
                    block: bi,
                    id: b.get("id").and_then(Value::as_str).unwrap_or("").to_owned(),
                    name: b
                        .get("name")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_owned(),
                });
            }
        }
    }
    out
}

/// Replace the matching tool_result blocks' content with the placeholder.
/// Returns true when something actually changed (an already-cleared pair
/// changes nothing, which is what lets repeated requests converge).
fn clear_result(messages: &mut [Value], tool_use_id: &str) -> bool {
    let mut changed = false;
    for m in messages.iter_mut() {
        if m.get("role").and_then(Value::as_str) != Some("user") {
            continue;
        }
        let Some(blocks) = m.get_mut("content").and_then(Value::as_array_mut) else {
            continue;
        };
        for b in blocks.iter_mut() {
            if b.get("type").and_then(Value::as_str) == Some("tool_result")
                && b.get("tool_use_id").and_then(Value::as_str) == Some(tool_use_id)
            {
                let placeholder = json!([{"type": "text", "text": CLEARED_RESULT}]);
                if b.get("content") != Some(&placeholder) {
                    b["content"] = placeholder;
                    changed = true;
                }
            }
        }
    }
    changed
}

/// Apply the configured edits. `original_tokens` is the rendered token count
/// of the unedited prompt; `count` re-renders candidate message arrays so
/// triggers and reports use exact tokenizer counts, not estimates. Returns
/// the (possibly edited) messages and the report.
pub fn apply(
    cfg: &Config,
    messages: &[Value],
    original_tokens: usize,
    mut count: impl FnMut(&[Value]) -> Result<usize, String>,
) -> Result<(Vec<Value>, Applied), String> {
    let mut msgs: Vec<Value> = messages.to_vec();
    let mut edits_out: Vec<Value> = Vec::new();
    let mut tokens_now = original_tokens;
    let mut compact: Option<CompactFire> = None;

    for edit in &cfg.edits {
        match edit {
            Edit::Compact {
                trigger,
                instructions,
                pause,
            } => {
                // evaluated at its list position, so clears listed before it
                // get their chance to bring the prompt back under the trigger
                // (the cheap strategies first, the generation only as needed)
                if (tokens_now as u64) < *trigger {
                    continue;
                }
                // an empty span means the whole prompt is the pending turn -
                // nothing to summarize away; clear_tool_uses is the tool there
                if compact_tail_start(&msgs) == 0 {
                    continue;
                }
                compact = Some(CompactFire {
                    instructions: instructions.clone(),
                    pause: *pause,
                });
            }
            Edit::ClearThinking { keep_turns } => {
                let Some(keep) = keep_turns else { continue }; // keep all = no-op
                // a "thinking turn" is an assistant message holding >= 1
                // thinking block; keep the last `keep` of those intact
                let turn_idx: Vec<usize> = msgs
                    .iter()
                    .enumerate()
                    .filter(|(_, m)| {
                        m.get("role").and_then(Value::as_str) == Some("assistant")
                            && m.get("content")
                                .and_then(Value::as_array)
                                .is_some_and(|blocks| {
                                    blocks.iter().any(|b| {
                                        matches!(
                                            b.get("type").and_then(Value::as_str),
                                            Some("thinking") | Some("redacted_thinking")
                                        )
                                    })
                                })
                    })
                    .map(|(i, _)| i)
                    .collect();
                if turn_idx.len() <= *keep as usize {
                    continue;
                }
                let clear_until = turn_idx[turn_idx.len() - *keep as usize];
                let mut cleared_turns = 0u64;
                for (mi, m) in msgs.iter_mut().enumerate() {
                    if mi >= clear_until {
                        break;
                    }
                    let Some(blocks) = m.get_mut("content").and_then(Value::as_array_mut) else {
                        continue;
                    };
                    let before = blocks.len();
                    blocks.retain(|b| {
                        !matches!(
                            b.get("type").and_then(Value::as_str),
                            Some("thinking") | Some("redacted_thinking")
                        )
                    });
                    if blocks.len() < before {
                        cleared_turns += 1;
                    }
                }
                if cleared_turns == 0 {
                    continue;
                }
                let after = count(&msgs)?;
                edits_out.push(json!({
                    "type": "clear_thinking_20251015",
                    "cleared_thinking_turns": cleared_turns,
                    "cleared_input_tokens": tokens_now.saturating_sub(after),
                }));
                tokens_now = after;
            }
            Edit::ClearToolUses {
                trigger,
                keep,
                clear_at_least,
                exclude_tools,
                clear_tool_inputs,
            } => {
                let uses = tool_uses(&msgs);
                let fired = match trigger {
                    Threshold::InputTokens(v) => tokens_now as u64 >= *v,
                    Threshold::ToolUses(v) => uses.len() as u64 >= *v,
                };
                if !fired {
                    continue;
                }
                // candidates: everything except the most recent `keep` pairs
                // and excluded tools - oldest first, which maximizes the
                // surviving cacheable prefix
                let keep_from = uses.len().saturating_sub(*keep as usize);
                let candidates: Vec<&ToolUse> = uses[..keep_from]
                    .iter()
                    .filter(|u| !exclude_tools.iter().any(|x| x == &u.name))
                    .collect();
                if candidates.is_empty() {
                    continue;
                }
                let mut edited = msgs.clone();
                let mut cleared = 0u64;
                for u in &candidates {
                    let mut changed = clear_result(&mut edited, &u.id);
                    if *clear_tool_inputs {
                        let b = &mut edited[u.msg]["content"][u.block];
                        if b.get("input") != Some(&json!({})) {
                            b["input"] = json!({});
                            changed = true;
                        }
                    }
                    if changed {
                        cleared += 1;
                    }
                }
                if cleared == 0 {
                    continue; // everything clearable was already placeholders
                }
                let after = count(&edited)?;
                let reclaimed = tokens_now.saturating_sub(after);
                // clear_at_least: an edit that cannot reclaim enough applies
                // nothing - a small clear still invalidates the whole prompt
                // cache after the first edited token, which is exactly the
                // waste this knob exists to refuse (pinned semantics; see
                // module docs)
                if let Some(min) = clear_at_least {
                    let enough = match min {
                        Threshold::InputTokens(v) => reclaimed as u64 >= *v,
                        Threshold::ToolUses(v) => cleared >= *v,
                    };
                    if !enough {
                        continue;
                    }
                }
                msgs = edited;
                edits_out.push(json!({
                    "type": "clear_tool_uses_20250919",
                    "cleared_tool_uses": cleared,
                    "cleared_input_tokens": reclaimed,
                }));
                tokens_now = after;
            }
        }
    }
    Ok((
        msgs,
        Applied {
            edits: edits_out,
            final_tokens: tokens_now,
            compact,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic fake "tokenizer": chars/4 over the serialized array -
    /// the core only compares counts, so any monotone measure exercises it.
    fn fake_count(msgs: &[Value]) -> Result<usize, String> {
        Ok(serde_json::to_string(msgs).unwrap().len() / 4)
    }

    fn tool_turn(id: &str, name: &str, result: &str) -> [Value; 2] {
        [
            json!({"role": "assistant", "content": [
                {"type": "tool_use", "id": id, "name": name, "input": {"q": "x"}}]}),
            json!({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": id, "content": result}]}),
        ]
    }

    fn cfg(v: Value) -> Config {
        parse(&v).expect("valid config")
    }

    #[test]
    fn trigger_unmet_is_a_noop() {
        let msgs: Vec<Value> = tool_turn("t1", "search", "big result").into();
        let c = cfg(json!({"edits": [{"type": "clear_tool_uses_20250919",
            "trigger": {"type": "input_tokens", "value": 1_000_000}}]}));
        let (out, applied) = apply(&c, &msgs, 500, fake_count).unwrap();
        assert!(applied.edits.is_empty());
        assert_eq!(out, msgs);
    }

    #[test]
    fn clears_oldest_keeps_recent_and_excluded() {
        let mut msgs: Vec<Value> = Vec::new();
        msgs.extend(tool_turn("t1", "lookup", &"x".repeat(400)));
        msgs.extend(tool_turn("t2", "web_search", &"y".repeat(400)));
        msgs.extend(tool_turn("t3", "lookup", &"z".repeat(400)));
        let c = cfg(json!({"edits": [{"type": "clear_tool_uses_20250919",
            "trigger": {"type": "tool_uses", "value": 2},
            "keep": {"type": "tool_uses", "value": 1},
            "exclude_tools": ["web_search"]}]}));
        let orig = fake_count(&msgs).unwrap();
        let (out, applied) = apply(&c, &msgs, orig, fake_count).unwrap();
        assert_eq!(applied.edits.len(), 1);
        assert_eq!(
            applied.edits[0]["cleared_tool_uses"], 1,
            "t1 only: t3 kept, t2 excluded"
        );
        let r1 = out[1]["content"][0]["content"][0]["text"].as_str().unwrap();
        assert_eq!(r1, CLEARED_RESULT);
        assert!(
            out[3]["content"][0]["content"]
                .as_str()
                .unwrap()
                .contains('y'),
            "excluded intact"
        );
        assert!(
            out[5]["content"][0]["content"]
                .as_str()
                .unwrap()
                .contains('z'),
            "kept intact"
        );
        // inputs stay by default
        assert_eq!(out[0]["content"][0]["input"]["q"], "x");
        assert!(applied.edits[0]["cleared_input_tokens"].as_u64().unwrap() > 0);
    }

    #[test]
    fn clear_tool_inputs_also_empties_the_call() {
        let msgs: Vec<Value> = [
            tool_turn("t1", "a", &"x".repeat(200)).to_vec(),
            tool_turn("t2", "a", "tail").to_vec(),
        ]
        .concat();
        let c = cfg(json!({"edits": [{"type": "clear_tool_uses_20250919",
            "trigger": {"type": "tool_uses", "value": 1},
            "keep": {"type": "tool_uses", "value": 1},
            "clear_tool_inputs": true}]}));
        let (out, applied) = apply(&c, &msgs, 9999, fake_count).unwrap();
        assert_eq!(applied.edits[0]["cleared_tool_uses"], 1);
        assert_eq!(out[0]["content"][0]["input"], json!({}));
    }

    #[test]
    fn clear_at_least_refuses_a_pointless_clear() {
        // one tiny clearable result - reclaiming it cannot meet the floor
        let msgs: Vec<Value> = [
            tool_turn("t1", "a", "tiny").to_vec(),
            tool_turn("t2", "a", "tail").to_vec(),
        ]
        .concat();
        let c = cfg(json!({"edits": [{"type": "clear_tool_uses_20250919",
            "trigger": {"type": "tool_uses", "value": 1},
            "keep": {"type": "tool_uses", "value": 1},
            "clear_at_least": {"type": "input_tokens", "value": 100_000}}]}));
        let (out, applied) = apply(&c, &msgs, 9999, fake_count).unwrap();
        assert!(
            applied.edits.is_empty(),
            "small reclaim + cache invalidation = refused"
        );
        assert_eq!(out, msgs);
    }

    #[test]
    fn already_cleared_pairs_converge_to_noop() {
        let mut msgs: Vec<Value> = [
            tool_turn("t1", "a", &"x".repeat(200)).to_vec(),
            tool_turn("t2", "a", "tail").to_vec(),
        ]
        .concat();
        let c = cfg(json!({"edits": [{"type": "clear_tool_uses_20250919",
            "trigger": {"type": "tool_uses", "value": 1},
            "keep": {"type": "tool_uses", "value": 1}}]}));
        let (once, applied) = apply(&c, &msgs, 9999, fake_count).unwrap();
        assert_eq!(applied.edits.len(), 1);
        msgs = once;
        let (again, applied2) = apply(&c, &msgs, 9999, fake_count).unwrap();
        assert!(
            applied2.edits.is_empty(),
            "re-clearing placeholders reports nothing"
        );
        assert_eq!(again, msgs);
    }

    #[test]
    fn clear_thinking_keeps_last_n_turns() {
        let think = |id: u32| {
            json!({"role": "assistant", "content": [
                {"type": "thinking", "thinking": format!("thought {id} {}", "t".repeat(100)), "signature": ""},
                {"type": "text", "text": format!("answer {id}")}]})
        };
        let user = json!({"role": "user", "content": "q"});
        let msgs = vec![
            user.clone(),
            think(1),
            user.clone(),
            think(2),
            user.clone(),
            think(3),
        ];
        let c = cfg(json!({"edits": [{"type": "clear_thinking_20251015",
            "keep": {"type": "thinking_turns", "value": 1}}]}));
        let orig = fake_count(&msgs).unwrap();
        let (out, applied) = apply(&c, &msgs, orig, fake_count).unwrap();
        assert_eq!(applied.edits.len(), 1);
        assert_eq!(applied.edits[0]["cleared_thinking_turns"], 2);
        // turns 1+2 lost their thinking blocks, answers intact, turn 3 whole
        assert_eq!(out[1]["content"].as_array().unwrap().len(), 1);
        assert_eq!(out[1]["content"][0]["type"], "text");
        assert_eq!(out[3]["content"].as_array().unwrap().len(), 1);
        assert_eq!(out[5]["content"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn keep_all_thinking_is_a_noop_and_order_rule_enforced() {
        let c = cfg(json!({"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]}));
        let msgs = vec![json!({"role": "user", "content": "q"})];
        let (_, applied) = apply(&c, &msgs, 10, fake_count).unwrap();
        assert!(applied.edits.is_empty());
        // clear_thinking anywhere but first is the documented error
        assert!(
            parse(&json!({"edits": [
                {"type": "clear_tool_uses_20250919"},
                {"type": "clear_thinking_20251015"}]}))
            .is_err()
        );
        // unknown strategies are refused
        assert!(parse(&json!({"edits": [{"type": "clear_everything_9000"}]})).is_err());
    }

    #[test]
    fn compact_parse_defaults_and_deviations() {
        // bare edit: default trigger, no instructions, no pause
        let c = cfg(json!({"edits": [{"type": "compact_20260112"}]}));
        assert!(matches!(
            &c.edits[0],
            Edit::Compact {
                trigger: 150_000,
                instructions: None,
                pause: false
            }
        ));
        // sub-50k triggers are ACCEPTED (documented deviation: a 32k-window
        // local server needs compaction well under Anthropic's floor)
        let c = cfg(json!({"edits": [{"type": "compact_20260112",
            "trigger": {"type": "input_tokens", "value": 500},
            "instructions": "keep the numbers", "pause_after_compaction": true}]}));
        assert!(matches!(&c.edits[0],
            Edit::Compact { trigger: 500, instructions: Some(i), pause: true } if i == "keep the numbers"));
        // trigger is input_tokens only (SDK: BetaInputTokensTriggerParam)
        assert!(
            parse(&json!({"edits": [{"type": "compact_20260112",
                "trigger": {"type": "tool_uses", "value": 3}}]}))
            .is_err()
        );
        // one compact edit max
        assert!(
            parse(&json!({"edits": [
                {"type": "compact_20260112"}, {"type": "compact_20260112"}]}))
            .is_err()
        );
    }

    /// A grown conversation: two finished exchanges (one with a tool round),
    /// then the pending turn with a tool round under it.
    fn grown_conversation() -> Vec<Value> {
        let mut msgs = vec![
            json!({"role": "user", "content": "First question?"}),
            json!({"role": "assistant", "content": "First answer."}),
            json!({"role": "user", "content": "Second question?"}),
        ];
        msgs.extend(tool_turn("t1", "lookup", &"x".repeat(200)));
        msgs.push(json!({"role": "assistant", "content": "Second answer."}));
        msgs.push(
            json!({"role": "user", "content": [{"type": "text", "text": "Pending question?"}]}),
        );
        msgs.extend(tool_turn("t2", "lookup", "pending tool data"));
        msgs
    }

    #[test]
    fn compact_span_is_everything_before_the_pending_turn() {
        let msgs = grown_conversation();
        // the tail starts at the pending user question (index 6), not at the
        // t2 tool_result user message after it
        assert_eq!(compact_tail_start(&msgs), 6);
        // all-tool tails (no plain user message at all) have no split
        let only_tools: Vec<Value> = tool_turn("t9", "a", "r").into();
        assert_eq!(compact_tail_start(&only_tools), 0);
    }

    #[test]
    fn compact_fires_without_an_applied_edits_entry() {
        let msgs = grown_conversation();
        let c = cfg(json!({"edits": [{"type": "compact_20260112",
            "trigger": {"type": "input_tokens", "value": 100}}]}));
        let (out, applied) = apply(&c, &msgs, 5000, fake_count).unwrap();
        assert!(applied.compact.is_some(), "trigger met, span non-empty");
        assert!(
            applied.edits.is_empty(),
            "SDK pin: compaction never in applied_edits"
        );
        assert_eq!(
            out, msgs,
            "the pure core does not touch messages for compact"
        );
        // unmet trigger
        let (_, applied) = apply(&c, &msgs, 50, fake_count).unwrap();
        assert!(applied.compact.is_none());
        // empty span: the whole prompt is the pending turn
        let pending_only = vec![json!({"role": "user", "content": "hi"})];
        let (_, applied) = apply(&c, &pending_only, 5000, fake_count).unwrap();
        assert!(applied.compact.is_none());
    }

    #[test]
    fn clears_listed_first_can_defuse_the_compact_trigger() {
        let mut msgs: Vec<Value> = Vec::new();
        msgs.extend(tool_turn("t1", "a", &"x".repeat(4000)));
        msgs.extend(tool_turn("t2", "a", "small"));
        msgs.push(json!({"role": "user", "content": "Pending?"}));
        let orig = fake_count(&msgs).unwrap();
        // clear reclaims ~1000 fake-tokens; trigger sits between post- and
        // pre-clear counts, so the compact edit sees the DEFUSED count
        let c = cfg(json!({"edits": [
            {"type": "clear_tool_uses_20250919",
             "trigger": {"type": "tool_uses", "value": 1},
             "keep": {"type": "tool_uses", "value": 1}},
            {"type": "compact_20260112",
             "trigger": {"type": "input_tokens", "value": (orig - 100) as u64}}]}));
        let (_, applied) = apply(&c, &msgs, orig, fake_count).unwrap();
        assert_eq!(applied.edits.len(), 1, "the clear applied");
        assert!(
            applied.compact.is_none(),
            "post-clear count is under the compact trigger"
        );
    }

    #[test]
    fn compacted_and_resend_render_the_identical_conversation() {
        // The cache-stability invariant: iteration 2's conversation must be a
        // prefix of what the next turn's resend rewrites to, or every
        // post-compaction turn is a cold prefill.
        let msgs = grown_conversation();
        let summary = "The user asked two questions; both were answered.";
        let in_request = compacted_messages(&msgs, summary);
        // tail = pending question + t2 round; summary framed into the front
        assert_eq!(in_request.len(), 3);
        let first_text = in_request[0]["content"][0]["text"].as_str().unwrap();
        assert!(first_text.starts_with(COMPACTION_FRAME));
        assert!(first_text.ends_with(summary));
        assert_eq!(in_request[0]["content"][1]["text"], "Pending question?");

        // the client appends our response (compaction block + answer) and a
        // new user turn, then resends the whole history
        let mut resend = msgs.clone();
        resend.push(json!({"role": "assistant", "content": [
            {"type": "compaction", "content": summary},
            {"type": "text", "text": "Final answer."}]}));
        resend.push(json!({"role": "user", "content": "Follow-up?"}));
        let rewritten = resend_rewrite(&resend).expect("compaction block present");
        assert_eq!(
            &rewritten[..in_request.len()],
            &in_request[..],
            "identical prefix"
        );
        assert_eq!(
            rewritten[in_request.len()],
            json!({"role": "assistant", "content": [{"type": "text", "text": "Final answer."}]}),
            "the compaction block is consumed, the answer survives"
        );
        assert_eq!(
            rewritten[in_request.len() + 1],
            json!({"role": "user", "content": "Follow-up?"})
        );
    }

    /// A grown Responses conversation: finished exchange with a tool round,
    /// then the pending user item with a tool round under it.
    fn oa_items() -> Vec<Value> {
        vec![
            json!({"role": "user", "content": "First question?"}),
            json!({"type": "message", "role": "assistant",
                   "content": [{"type": "output_text", "text": "First answer."}]}),
            json!({"type": "function_call", "call_id": "c1", "name": "lookup",
                   "arguments": "{}"}),
            json!({"type": "function_call_output", "call_id": "c1", "output": "old data"}),
            json!({"type": "message", "role": "assistant",
                   "content": [{"type": "output_text", "text": "Second answer."}]}),
            json!({"role": "user", "content": [{"type": "input_text", "text": "Pending?"}]}),
            json!({"type": "function_call", "call_id": "c2", "name": "lookup",
                   "arguments": "{}"}),
            json!({"type": "function_call_output", "call_id": "c2", "output": "fresh data"}),
        ]
    }

    #[test]
    fn oa_parse_validates_and_defaults() {
        assert_eq!(oa_parse(&[]).unwrap(), None);
        assert_eq!(
            oa_parse(&[json!({"type": "compaction"})]).unwrap(),
            Some(OA_DEFAULT_THRESHOLD)
        );
        assert_eq!(
            oa_parse(&[json!({"type": "compaction", "compact_threshold": null})]).unwrap(),
            Some(OA_DEFAULT_THRESHOLD)
        );
        // sub-50k accepted, same deviation as the Anthropic dialect
        assert_eq!(
            oa_parse(&[json!({"type": "compaction", "compact_threshold": 800})]).unwrap(),
            Some(800)
        );
        assert!(oa_parse(&[json!({"type": "compaction", "compact_threshold": 0})]).is_err());
        assert!(oa_parse(&[json!({"type": "truncation"})]).is_err());
        assert!(oa_parse(&[json!({"type": "compaction"}), json!({"type": "compaction"})]).is_err());
    }

    #[test]
    fn oa_tail_is_the_pending_user_item() {
        let items = oa_items();
        // the pending user item at 5, not the function_call_output after it
        assert_eq!(oa_tail_start(&items), 5);
        // no user item at all: no split
        assert_eq!(oa_tail_start(&items[2..4]), 0);
    }

    #[test]
    fn oa_trigger_must_be_final() {
        let mut items = oa_items();
        assert!(!oa_take_trigger(&mut items).unwrap());
        items.push(json!({"type": "compaction_trigger"}));
        assert!(oa_take_trigger(&mut items).unwrap());
        assert_eq!(items.len(), oa_items().len(), "trigger consumed");
        // anywhere else is the documented error
        let mut bad = oa_items();
        bad.insert(0, json!({"type": "compaction_trigger"}));
        assert!(oa_take_trigger(&mut bad).is_err());
    }

    #[test]
    fn oa_compacted_and_resend_render_the_identical_items() {
        // the same cache-stability invariant as the Anthropic pair, at item
        // level: iteration 2's list must be a prefix of the resend rewrite
        let items = oa_items();
        let summary = "One exchange happened; a lookup returned old data.";
        let in_request = oa_compacted_items(&items, summary);
        assert_eq!(in_request.len(), 3, "pending user + c2 round");
        let first = in_request[0]["content"][0]["text"].as_str().unwrap();
        assert!(first.starts_with(COMPACTION_FRAME) && first.contains(summary));
        assert_eq!(in_request[0]["content"][1]["text"], "Pending?");

        // client appends our output (compaction item first, then the answer)
        // and a new turn, then resends the whole item list
        let mut resend = items.clone();
        resend.push(json!({"id": "cp_1", "type": "compaction", "encrypted_content": summary}));
        resend.push(json!({"type": "message", "role": "assistant",
                           "content": [{"type": "output_text", "text": "Answer."}]}));
        resend.push(json!({"role": "user", "content": "Follow-up?"}));
        let rewritten = oa_resend_rewrite(&resend).expect("compaction item present");
        assert_eq!(
            &rewritten[..in_request.len()],
            &in_request[..],
            "identical prefix"
        );
        assert_eq!(
            rewritten.len(),
            in_request.len() + 2,
            "answer + follow-up survive"
        );

        // string-content pending turn wraps into parts without losing text
        let short = vec![
            json!({"role": "user", "content": "Old."}),
            json!({"type": "message", "role": "assistant",
                   "content": [{"type": "output_text", "text": "A."}]}),
            json!({"role": "user", "content": "Now?"}),
        ];
        let c = oa_compacted_items(&short, "s");
        assert_eq!(c[0]["content"][1]["text"], "Now?");
    }

    #[test]
    fn oa_failed_compaction_item_is_a_noop_on_resend() {
        let mut items = oa_items();
        items.push(json!({"type": "compaction", "encrypted_content": ""}));
        items.push(json!({"role": "user", "content": "Continue."}));
        let rewritten = oa_resend_rewrite(&items).expect("item present");
        assert_eq!(rewritten.len(), items.len() - 1, "only the item stripped");
        assert_eq!(rewritten[0], items[0]);
        // no compaction item: nothing to rewrite
        assert!(oa_resend_rewrite(&oa_items()).is_none());
    }

    #[test]
    fn failed_compaction_block_is_a_noop_on_resend() {
        // SDK pin: null content = failed compaction, round-trips as a no-op
        let mut msgs = grown_conversation();
        msgs.push(json!({"role": "assistant", "content": [
            {"type": "compaction", "content": null}]}));
        msgs.push(json!({"role": "user", "content": "Continue."}));
        let rewritten = resend_rewrite(&msgs).expect("block present");
        // nothing dropped, the block (and its emptied message) stripped
        assert_eq!(rewritten.len(), msgs.len() - 1);
        assert_eq!(rewritten[0], msgs[0]);
        assert_eq!(rewritten.last().unwrap(), &msgs[msgs.len() - 1]);
        // no compaction block anywhere: nothing to rewrite
        assert!(resend_rewrite(&grown_conversation()).is_none());
    }
}
