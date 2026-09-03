//! Anthropic Messages API (`/v1/messages`) request types.
//!
//! Content blocks, tools and tool_choice are verbatim JSON - the server
//! converts them to chat-template shapes. Response objects and streaming
//! events are built as JSON in the server to track the evolving spec.

use serde::Deserialize;

/// Unknown fields are REJECTED, matching the real API ("Extra inputs are not
/// permitted") - unimplemented spec parameters (container) 400 loudly, tracked
/// in tests/spec/coverage.json. seed/min_p are local extensions.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MessagesRequest {
    pub model: String,
    /// Required by the Anthropic API (no default).
    pub max_tokens: usize,
    /// [{role: user|assistant, content: string | [blocks]}]
    pub messages: Vec<serde_json::Value>,
    /// Anthropic MCP connector (beta `mcp-client-2025-04-04`): remote MCP servers
    /// Paddock connects to and executes tool calls against, emitting
    /// `mcp_tool_use` / `mcp_tool_result` content blocks. Each entry is
    /// `{type:"url", url, name, authorization_token?, tool_configuration?}`.
    #[serde(default)]
    pub mcp_servers: Option<Vec<serde_json::Value>>,
    /// A string, or an array of text blocks.
    #[serde(default)]
    pub system: Option<serde_json::Value>,
    /// [{name, description?, input_schema}]
    #[serde(default)]
    pub tools: Option<Vec<serde_json::Value>>,
    /// {type: auto|any|tool|none, name?, disable_parallel_tool_use?}
    #[serde(default)]
    pub tool_choice: Option<serde_json::Value>,
    #[serde(default)]
    pub stop_sequences: Vec<String>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub min_p: Option<f32>,
    #[serde(default)]
    pub seed: Option<u64>,
    /// {type: enabled|disabled|adaptive, budget_tokens?, display?} - budget is
    /// accepted but not enforced (thinking length is model-driven); adaptive
    /// maps to enabled; display "omitted" withholds thinking text from the
    /// response (see the server's messages module).
    #[serde(default)]
    pub thinking: Option<serde_json::Value>,
    #[serde(default)]
    pub stream: bool,
    /// {user_id} - accepted, unused.
    #[serde(default)]
    pub metadata: Option<serde_json::Value>,
    /// Routing hint: accepted with its spec semantics (no tiers here).
    #[serde(default)]
    pub service_tier: Option<String>,
    /// Local extension (deepseek2-ocr family only, same as chat/responses):
    /// the OCR request object - `{mode, grounding, crop,
    /// no_repeat_ngram_size, ngram_window}`. This surface has no
    /// chat_template_kwargs channel, so the top-level field is the only form.
    #[serde(default)]
    pub ocr: Option<serde_json::Value>,
    /// vLLM-compat (paddleocr family, same as chat): per-request multimodal
    /// processor kwargs - `{min_pixels, max_pixels}` smart-resize budget.
    #[serde(default)]
    pub mm_processor_kwargs: Option<serde_json::Value>,
    /// Local extension (same as chat/responses): document metadata injected
    /// with extracted file content - "full" (the default) or "off".
    #[serde(default)]
    pub file_metadata: Option<String>,
    /// Local extension (same as chat/responses): cap the pages taken from
    /// every multi-page attachment.
    #[serde(default)]
    pub max_pages: Option<u32>,
    /// Local extension (same as chat/responses): "render" | "text" PDF route.
    #[serde(default)]
    pub pdf_mode: Option<String>,
    /// Local extension (same as chat/responses): run the forensics pass this
    /// turn - "on" | "off". Absent = the endpoint's `[forensics] auto` default.
    /// `/v1/messages` stays injection-only (spec-strict), like `/v1/chat`.
    #[serde(default)]
    pub forensics: Option<String>,
    /// Caching/routing/telemetry hints: accepted - Paddock's prefix caching
    /// is implicit and subsumes explicit cache_control; the rest have no
    /// server-side effect to have.
    #[serde(default)]
    pub cache_control: Option<serde_json::Value>,
    #[serde(default)]
    pub inference_geo: Option<serde_json::Value>,
    #[serde(default)]
    pub user_profile_id: Option<serde_json::Value>,
    /// Server-side context editing (beta context-management-2025-06-27):
    /// {edits: [{type: clear_tool_uses_20250919 | clear_thinking_20251015,
    /// ...}]}. Applied before render; the response reports `applied_edits`.
    #[serde(default)]
    pub context_management: Option<serde_json::Value>,
    /// Anthropic's `{effort, format}`. `effort` is their own graded reasoning
    /// ladder (low|medium|high|xhigh|max); `format` is their structured output,
    /// `{type: "json_schema", schema: {...}}`. Both land on machinery this
    /// server already has -- see `messages::apply_output_config`.
    #[serde(default)]
    pub output_config: Option<serde_json::Value>,
}

/// `/v1/messages/count_tokens` - same conversion pipeline, no generation.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CountTokensRequest {
    pub model: String,
    pub messages: Vec<serde_json::Value>,
    #[serde(default)]
    pub system: Option<serde_json::Value>,
    #[serde(default)]
    pub tools: Option<Vec<serde_json::Value>>,
    #[serde(default)]
    pub thinking: Option<serde_json::Value>,
    /// Local extension, accepted so a caller can mirror their generate request
    /// verbatim - and honored, so the count matches what generation injects.
    #[serde(default)]
    pub file_metadata: Option<String>,
    /// Local extension, honored like `file_metadata` for the same reason.
    #[serde(default)]
    pub max_pages: Option<u32>,
    /// Local extension, honored like `file_metadata` for the same reason.
    #[serde(default)]
    pub pdf_mode: Option<String>,
    /// Local extension, honored like `file_metadata` for the same reason -
    /// forensics injects a directive block, so the count must include it.
    #[serde(default)]
    pub forensics: Option<String>,
    /// Local extension, honored like `file_metadata` for the same reason -
    /// an OCR mode changes the injected task text, so the count must too.
    #[serde(default)]
    pub ocr: Option<serde_json::Value>,
    /// Same contract as MessagesRequest: applied to the count, and the
    /// response carries context_management.original_input_tokens.
    #[serde(default)]
    pub context_management: Option<serde_json::Value>,
    /// Same field as the create call. Accepted so a counting client is not
    /// refused for sending what it will send for real; it changes no count
    /// here, because neither the effort kwarg nor the output schema is part
    /// of the rendered prompt.
    #[serde(default)]
    pub output_config: Option<serde_json::Value>,
}
