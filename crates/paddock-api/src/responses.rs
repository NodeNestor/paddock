//! OpenAI Responses API (`/v1/responses`) request type.
//!
//! `input` is string-or-items; `tools` are verbatim JSON (function + mcp
//! shapes). Response objects and streaming events are built as JSON in the
//! server to track the evolving spec precisely.

use serde::Deserialize;

/// Unknown fields are REJECTED, matching the real API (see chat.rs) - spec
/// parameters we don't implement (background, include, prompt templates,
/// conversation state) 400 loudly, tracked in tests/spec/coverage.json.
/// top_k/min_p/seed are local extensions.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResponsesRequest {
    pub model: String,
    /// A string, or an array of input items.
    pub input: serde_json::Value,
    /// System-level instructions (prepended as a system message).
    #[serde(default)]
    pub instructions: Option<String>,
    #[serde(default)]
    pub tools: Option<Vec<serde_json::Value>>,
    #[serde(default)]
    pub tool_choice: Option<serde_json::Value>,
    #[serde(default)]
    pub parallel_tool_calls: Option<bool>,
    /// Server-side conversation chaining. Not yet supported - we reject rather
    /// than silently fake state (no-silent-failures).
    #[serde(default)]
    pub previous_response_id: Option<String>,
    #[serde(default)]
    pub store: Option<bool>,
    /// {"format": {"type": "text" | "json_object" | "json_schema", ...}} -
    /// the Responses-API structured-output knob (flat schema under `format`).
    #[serde(default)]
    pub text: Option<serde_json::Value>,
    /// Spec `include`: extra payloads the response should carry. Implemented:
    /// `"message.output_text.logprobs"` (per-token logprobs on the text
    /// deltas/done events and the final content part). Any other documented
    /// value 400s loudly - accepting one and not delivering it would be a
    /// silent failure.
    #[serde(default)]
    pub include: Option<Vec<String>>,
    /// 0..=20 top alternatives per token, spec semantics: it shapes what the
    /// logprobs include DELIVERS, and does nothing without that include.
    #[serde(default)]
    pub top_logprobs: Option<u8>,
    #[serde(default = "default_max")]
    pub max_output_tokens: usize,
    /// How many BUILT-IN tool calls this response may run - the MCP tools, the
    /// catalog search and web search, i.e. the ones the server executes. The
    /// caller's own function tools are not built-in and are not counted (their
    /// calls end the turn anyway).
    ///
    /// Reaching it is not an error: the agent loop takes its answer round, so
    /// the turn still ends with an answer built from what did come back
    /// (`paddock_mcp::loop_budget`). `0` means no tool may run at all.
    #[serde(default)]
    pub max_tool_calls: Option<usize>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub min_p: Option<f32>,
    /// Local extensions, mirroring /v1/chat/completions: the official
    /// Responses wire has no penalty knobs, but the Studio speaks Responses
    /// and its dials must reach the sampler somewhere.
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub repeat_penalty: Option<f32>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub stream: bool,
    /// OpenAI's `{include_obfuscation}`. Their hosted stream pads events to
    /// defeat a token-length side channel over TLS; we emit no padding, so
    /// `false` describes this server exactly and `true` asks for bytes it does
    /// not produce. Validated in `responses::check_stream_options`.
    #[serde(default)]
    pub stream_options: Option<serde_json::Value>,
    /// `{"effort": ...}` - the same seven-value vocabulary as chat completions'
    /// `reasoning_effort`, landing on the served template's own rungs
    /// (`/v1/models` -> `capabilities.reasoning_effort`).
    #[serde(default)]
    pub reasoning: Option<serde_json::Value>,
    /// Extra chat-template context (vLLM-style, a local extension like top_k):
    /// qwen3.5 `{"enable_thinking": true}` opens the `<think>` block so the
    /// model reasons; gpt-oss `{"reasoning_effort": "high"}`. `reasoning.effort`
    /// overlays this for Harmony models.
    #[serde(default)]
    pub chat_template_kwargs: Option<serde_json::Value>,
    /// "disabled" (default): an over-window prompt is a loud 400.
    /// "auto": drop items from the BEGINNING of the conversation until it
    /// fits (openai 2.53.0 docstring semantics), reported honestly via a
    /// response extension field.
    #[serde(default)]
    pub truncation: Option<String>,
    /// `[{"type": "compaction", "compact_threshold": N}]` - server-side
    /// compaction past a rendered-input-tokens threshold (openai 2.53.0
    /// shape; entries validated in the server).
    #[serde(default)]
    pub context_management: Option<Vec<serde_json::Value>>,
    /// Telemetry / routing hints: accepted with their spec semantics.
    #[serde(default)]
    pub user: Option<String>,
    #[serde(default)]
    pub safety_identifier: Option<String>,
    #[serde(default)]
    pub prompt_cache_key: Option<String>,
    #[serde(default)]
    pub service_tier: Option<String>,
    /// Non-empty metadata requires persistence, which this server does not
    /// have - honest 400 (same policy as store:true).
    #[serde(default)]
    pub metadata: Option<serde_json::Value>,
    /// Local extension (deepseek2-ocr family only, same as chat/messages):
    /// the OCR request object - `{mode, grounding, crop,
    /// no_repeat_ngram_size, ngram_window}`. Also reachable as
    /// `chat_template_kwargs.ocr`; this form is the documented one.
    #[serde(default)]
    pub ocr: Option<serde_json::Value>,
    /// vLLM-compat (paddleocr family, same as chat): per-request multimodal
    /// processor kwargs - `{min_pixels, max_pixels}` smart-resize budget.
    #[serde(default)]
    pub mm_processor_kwargs: Option<serde_json::Value>,
    /// Local extension (same as chat/messages): document metadata injected
    /// with extracted file content - "full" (the default) or "off".
    #[serde(default)]
    pub file_metadata: Option<String>,
    /// Local extension (same as chat/messages): cap the pages taken from
    /// every multi-page attachment.
    #[serde(default)]
    pub max_pages: Option<u32>,
    /// Local extension (same as chat/messages): "render" | "text" PDF route.
    #[serde(default)]
    pub pdf_mode: Option<String>,
    /// Local extension (same as chat/messages): run the forensics pass this
    /// turn - "on" | "off". Absent = the endpoint's `[forensics] auto` default.
    /// On `/v1/responses` an analyzed attachment additionally surfaces as a
    /// `{type:"forensics"}` output item (the persistable report).
    #[serde(default)]
    pub forensics: Option<String>,
}

/// `POST /v1/responses/compact` (openai 2.53.0 `ResponseCompactParams`).
/// The prompt-cache knobs are accepted as no-ops: the radix cache is always
/// on locally, so the intent they express is already served. Unknown fields
/// are rejected like every other surface.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompactRequest {
    pub model: Option<String>,
    /// A string, or an array of input items - the conversation to compact.
    pub input: serde_json::Value,
    /// System message for the summarization pass (the spec's standard
    /// `instructions` semantics; the summarization prompt itself stays ours).
    #[serde(default)]
    pub instructions: Option<String>,
    /// Stored-state continuation - rejected, same policy as create.
    #[serde(default)]
    pub previous_response_id: Option<String>,
    #[serde(default)]
    pub prompt_cache_key: Option<String>,
    #[serde(default)]
    pub prompt_cache_options: Option<serde_json::Value>,
    #[serde(default)]
    pub prompt_cache_retention: Option<String>,
    #[serde(default)]
    pub service_tier: Option<String>,
}

fn default_max() -> usize {
    // Server default; overridable via PADDOCK_MAX_OUTPUT_TOKENS, which the
    // server sets from `--max-output-tokens` / config at startup.
    std::env::var("PADDOCK_MAX_OUTPUT_TOKENS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(1024)
}
