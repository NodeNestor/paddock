//! Render a model's embedded Jinja chat template (minijinja). Generalizes to
//! any model that ships `tokenizer.chat_template` - the same approach llama.cpp
//! takes with minja. gpt-oss's template emits Harmony format; qwen3.5's emits
//! ChatML with XML-ish tool calls and (opt-in) `<think>` blocks.

use minijinja::{Environment, Error, ErrorKind, Value};

/// Render `messages`/`tools` through `template`, ready for tokenization.
/// `kwargs` (request `chat_template_kwargs`, vLLM-style) overlays the default
/// context - `enable_thinking` for qwen3.5, `reasoning_effort` for gpt-oss.
pub fn render(
    template: &str,
    messages: &[serde_json::Value],
    tools: Option<&[serde_json::Value]>,
    kwargs: Option<&serde_json::Value>,
) -> Result<String, String> {
    let mut env = Environment::new();

    // Chat templates are AUTHORED against transformers' environment, which is
    // `ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)`.
    // minijinja defaults both to false, so a template that relies on them
    // rather than on explicit `{%-`/`-%}` markers renders with stray
    // whitespace here that the model never saw in training.
    //
    // Found on granite 4.2: its template opens with a macro and a
    // run of `{%- set ... %}` lines whose closing tags carry no `-`, so without
    // trim_blocks paddock emitted a LEADING NEWLINE before `<|im_start|>`, and
    // every prompt tokenized one token longer than the same conversation on
    // llama.cpp (29 vs 28) - which is the reference makes us match.
    //
    // Blast radius was measured before flipping it, not assumed: all 23 chat
    // templates we serve were rendered both ways and diffed against jinja2
    // with these exact settings. The 18 that already matched are templates that
    // mark their whitespace explicitly, so the flags are a no-op for them; they
    // stay byte-identical. Only granite 4.2 moves, and it moves onto the
    // reference.
    env.set_trim_blocks(true);
    env.set_lstrip_blocks(true);

    // HF chat templates lean on Python-isms (.items(), .get(), .startswith(),
    // string methods...). pycompat implements them, same as llama.cpp's minja -
    // with `str.index`/`str.rindex` filled in below.
    env.set_unknown_method_callback(|state, value, method, args| {
        if let Some(s) = value.as_str()
            && matches!(method, "index" | "rindex")
        {
            return py_str_index(s, method == "rindex", args);
        }
        minijinja_contrib::pycompat::unknown_method_callback(state, value, method, args)
    });

    // helpers HF chat templates commonly call
    env.add_function("strftime_now", |fmt: String| {
        Ok(Value::from(chrono::Local::now().format(&fmt).to_string()))
    });
    env.add_function("raise_exception", |msg: String| -> Result<Value, Error> {
        Err(Error::new(ErrorKind::InvalidOperation, msg))
    });

    // HF templates may wrap assistant turns in `{% generation %}` /
    // `{% endgeneration %}` - training-time loss-mask markers with no render
    // semantics (laguna's template does). minijinja has no such tag, so they
    // must be neutralized before parse; whitespace-control dashes survive.
    let template = neutralize_generation_tags(template);
    // `tojson(ensure_ascii=False)` (laguna renders history tool-call values
    // with it) asks for raw UTF-8 instead of \uXXXX escapes - which is what
    // minijinja's plain `tojson` already emits, but minijinja rejects the
    // unknown kwarg. Dropping it is behavior-preserving. `ensure_ascii=True`
    // would be a real divergence (we don't escape non-ASCII), so that form is
    // left alone to fail loudly rather than silently mis-render.
    let template: std::borrow::Cow<'_, str> = if template.contains("tojson(ensure_ascii=False)") {
        template
            .replace("tojson(ensure_ascii=False)", "tojson")
            .into()
    } else {
        template
    };
    // ternaries inside call arguments - a minijinja PARSER gap, so this has to
    // happen before add_template or nothing renders at all
    let template: std::borrow::Cow<'_, str> = match parenthesize_call_ternaries(&template) {
        std::borrow::Cow::Borrowed(_) => template,
        std::borrow::Cow::Owned(s) => s.into(),
    };
    env.add_template("chat", &template)
        .map_err(|e| format!("chat template parse error: {e}"))?;
    let tmpl = env
        .get_template("chat")
        .map_err(|e| format!("chat template lookup: {e}"))?;

    let mut ctx = serde_json::Map::new();
    ctx.insert(
        "messages".into(),
        serde_json::Value::Array(messages.to_vec()),
    );
    ctx.insert(
        "tools".into(),
        // No tools renders as an empty ARRAY, not null: minijinja says
        // `none is iterable` is true and then dies on `none | length`
        // (nemotron's template guards with exactly that pair), while [] is
        // falsy for `{% if tools %}`, passes `is iterable` with length 0,
        // and iterates zero times - the "no tools" reading in every
        // template shape we serve.
        tools.map_or(serde_json::Value::Array(Vec::new()), |t| {
            // `description` is optional in both vendors' tool schemas, but
            // gpt-oss's Harmony template concatenates it unguarded - a tool
            // declared without one was a 400 ("+ operator on string and
            // none/undefined"). Fill the empty
            // string here at the one seam every surface renders through.
            let mut tools = t.to_vec();
            for tool in &mut tools {
                if let Some(f) = tool.get_mut("function").and_then(|f| f.as_object_mut())
                    && !f.get("description").is_some_and(|d| d.is_string())
                {
                    f.insert(
                        "description".into(),
                        serde_json::Value::String(String::new()),
                    );
                }
            }
            serde_json::Value::Array(tools)
        }),
    );
    ctx.insert("add_generation_prompt".into(), true.into());
    // No `reasoning_effort` default here, deliberately. This used
    // to be pinned to "medium" for gpt-oss's benefit, which gpt-oss never
    // needed - its own template already self-defaults to medium when the
    // variable is undefined. What the house value did do was override every
    // other template's published default: Qwen3.8-27B defaults to `xhigh` and
    // was being served at `medium` by this line alone, silently, on a prompt
    // llama.cpp renders differently. Leaving the variable unset is what makes
    // a checkpoint's own default reach the wire - the same principle as the
    // elected sampling profiles, and what `reasoning::probe` measures.
    //
    // thinking-model default, matching llama.cpp/minja: gemma4's template
    // DISABLES thinking when this is undefined (it pre-closes an empty
    // thought channel), while qwen3.5's treats defined-true as its own
    // default - so `true` here turns gemma4 thinking on and is a no-op for
    // qwen. Request `chat_template_kwargs.enable_thinking=false` overrides.
    ctx.insert("enable_thinking".into(), true.into());
    if let Some(kw) = kwargs {
        let obj = kw
            .as_object()
            .ok_or("chat_template_kwargs must be a JSON object")?;
        for (k, v) in obj {
            ctx.insert(k.clone(), v.clone());
        }
    }

    tmpl.render(Value::from_serialize(&ctx))
        .map_err(|e| format!("chat template render error: {e}"))
}

/// Rewrite `{% generation %}` / `{% endgeneration %}` statement tags into
/// Jinja comments, PRESERVING their whitespace-control dashes - the trim
/// behavior must survive or every rendered assistant turn grows the stray
/// indentation the original tags were eating. `{%- generation -%}` becomes
/// `{#- generation -#}`, which minijinja parses and drops with identical
/// whitespace handling. Templates without the tags pass through unchanged.
fn neutralize_generation_tags(template: &str) -> std::borrow::Cow<'_, str> {
    if !template.contains("generation") {
        return template.into();
    }
    let mut out = String::with_capacity(template.len());
    let mut rest = template;
    let mut changed = false;
    while let Some(start) = rest.find("{%") {
        let Some(end) = rest[start..].find("%}") else {
            break;
        };
        let inner = &rest[start + 2..start + end];
        // strip whitespace-control dashes to inspect the statement name
        let body = inner.strip_prefix('-').unwrap_or(inner);
        let body = body.strip_suffix('-').unwrap_or(body);
        if matches!(body.trim(), "generation" | "endgeneration") {
            out.push_str(&rest[..start]);
            out.push_str("{#");
            out.push_str(inner);
            out.push_str("#}");
            changed = true;
        } else {
            out.push_str(&rest[..start + end + 2]);
        }
        rest = &rest[start + end + 2..];
    }
    if !changed {
        return template.into();
    }
    out.push_str(rest);
    out.into()
}

/// minijinja (through 2.23) cannot parse a CONDITIONAL EXPRESSION as a call
/// argument: `namespace(name=tcid if tcid else '')` - ordinary Jinja2, and
/// exactly what muse-glimmer's template writes - fails to PARSE with
/// "unexpected identifier, expected `,`". Its argument grammar stops below the
/// ternary, so the parser reads `tcid`, then wants a `,` and finds `if`.
///
/// Wrapping the argument in parentheses is a pure syntax change - `f((a if b
/// else c))` and `f(k=(a if b else c))` evaluate identically in every Jinja -
/// and it is the whole fix. This rewrites the general case rather than the one
/// expression muse ships, because the gap is minijinja's and the next family
/// to hit it should not need another patch.
///
/// Scope discipline: only inside `{{ ... }}` / `{% ... %}` (template TEXT may
/// contain anything), only at a paren's own depth, and never inside a string
/// literal. Parens that group rather than call get the same treatment, which
/// is harmless - `((a if b else c))` is still the same expression.
fn parenthesize_call_ternaries(template: &str) -> std::borrow::Cow<'_, str> {
    // Cheap reject: no ternary anywhere means nothing to do.
    if !template.contains(" if ") {
        return template.into();
    }
    let b = template.as_bytes();
    // (open_paren_index, arg_start, saw_top_level_if) per open paren
    let mut stack: Vec<(usize, usize, bool)> = Vec::new();
    // byte offsets to insert "(" and ")" at, collected then applied in order
    let mut inserts: Vec<(usize, char)> = Vec::new();
    let mut i = 0usize;
    let mut in_tag = false;

    // Close the argument that ends at `end`: if it held a top-level ternary,
    // parenthesize it - after an `ident=` prefix when there is one.
    let close_arg = |start: usize, end: usize, saw_if: bool, ins: &mut Vec<(usize, char)>| {
        if !saw_if {
            return;
        }
        let arg = &template[start..end];
        let lead = arg.len() - arg.trim_start().len();
        let mut open = start + lead;
        // `ident =` (a keyword argument, not `==`/`!=`/`<=`/`>=`)
        let after_ident = arg[lead..]
            .find(|c: char| !(c.is_alphanumeric() || c == '_'))
            .map(|n| lead + n);
        if let Some(n) = after_ident {
            let rest = arg[n..].trim_start();
            if n > lead && rest.starts_with('=') && !rest.starts_with("==") {
                open = start + arg[n..].find('=').map(|k| n + k).unwrap_or(n) + 1;
            }
        }
        ins.push((open, '('));
        ins.push((end, ')'));
    };

    while i < b.len() {
        if !in_tag {
            if b[i] == b'{' && i + 1 < b.len() && (b[i + 1] == b'{' || b[i + 1] == b'%') {
                in_tag = true;
                stack.clear();
                i += 2;
                continue;
            }
            i += 1;
            continue;
        }
        match b[i] {
            b'}' | b'%' if i + 1 < b.len() && b[i + 1] == b'}' => {
                in_tag = false;
                i += 2;
            }
            q @ (b'\'' | b'"') => {
                i += 1;
                while i < b.len() && b[i] != q {
                    i += if b[i] == b'\\' { 2 } else { 1 };
                }
                i += 1;
            }
            b'(' | b'[' | b'{' => {
                stack.push((i, i + 1, false));
                i += 1;
            }
            b')' | b']' | b'}' => {
                if let Some((open, arg_start, saw_if)) = stack.pop()
                    && b[open] == b'('
                {
                    close_arg(arg_start, i, saw_if, &mut inserts);
                }
                i += 1;
            }
            b',' => {
                if let Some(&mut (open, ref mut arg_start, ref mut saw_if)) = stack.last_mut()
                    && b[open] == b'('
                {
                    let (s, f) = (*arg_start, *saw_if);
                    *arg_start = i + 1;
                    *saw_if = false;
                    close_arg(s, i, f, &mut inserts);
                }
                i += 1;
            }
            _ => {
                // a bare `if` token at this paren's own depth
                if template[i..].starts_with("if")
                    && (i == 0 || !is_word(b[i - 1]))
                    && !template[i + 2..].starts_with(|c: char| c.is_alphanumeric() || c == '_')
                    && let Some(top) = stack.last_mut()
                {
                    top.2 = true;
                }
                i += 1;
            }
        }
    }

    if inserts.is_empty() {
        return template.into();
    }
    inserts.sort_by_key(|&(at, _)| at);
    let mut out = String::with_capacity(template.len() + inserts.len());
    let mut last = 0usize;
    for (at, ch) in inserts {
        out.push_str(&template[last..at]);
        out.push(ch);
        last = at;
    }
    out.push_str(&template[last..]);
    out.into()
}

fn is_word(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'_'
}

/// Python's `str.index` / `str.rindex` - `find`/`rfind` but raising instead of
/// returning -1. `minijinja_contrib`'s pycompat stops at find/rfind, so without
/// this granite-vision's template cannot render at all once an image is
/// present: it calls `text.index("<image>")` to decide whether the picture goes
/// before or after the expanded task instruction, and an unknown method is a
/// hard render error, not a fallback.
///
/// Returns a CHARACTER offset, which is what Python means and what minijinja's
/// own string slicing consumes. Note pycompat's neighbouring `find`/`rfind`
/// return BYTE offsets, so the two disagree on non-ASCII text - an upstream
/// quirk worth knowing about before mixing them in one expression.
///
/// Only the one-argument form is implemented. Python also takes `start`/`end`;
/// no chat template we serve uses them, and a silently-ignored slice bound
/// would be worse than a loud refusal.
fn py_str_index(s: &str, from_end: bool, args: &[Value]) -> Result<Value, Error> {
    let [needle] = args else {
        return Err(Error::new(
            ErrorKind::InvalidOperation,
            "str.index() takes exactly one argument here (start/end bounds are not implemented)",
        ));
    };
    let needle = needle.as_str().ok_or_else(|| {
        Error::new(
            ErrorKind::InvalidOperation,
            "str.index() needs a string argument",
        )
    })?;
    let at = if from_end {
        s.rfind(needle)
    } else {
        s.find(needle)
    };
    // Python raises ValueError; an InvalidOperation is the same contract here -
    // it aborts the render rather than yielding a bogus position.
    let at = at.ok_or_else(|| {
        Error::new(
            ErrorKind::InvalidOperation,
            format!("substring {needle:?} not found"),
        )
    })?;
    Ok(Value::from(s[..at].chars().count()))
}

/// OpenAI sends assistant `tool_calls[].function.arguments` as a JSON STRING;
/// HF chat templates expect an object (qwen3.5's `is mapping` guard silently
/// drops string arguments from re-rendered history). Parse them back to
/// objects where they parse; anything else passes through untouched.
///
/// Also drops explicit `"content": null` keys: the OpenAI spec allows null
/// content on assistant tool-call messages, but templates guard with
/// `"content" in message` and then substring-test it (gpt-oss line 264), which
/// throws on null - with the key absent the guard skips instead.
///
/// **PRECONDITION: images must already be extracted.** This is DESTRUCTIVE to
/// image parts - every one is rewritten down to a bare `{"type":"image"}`
/// marker, because that is all the template needs once the pixels are out.
/// Call `find_image_urls` first or the payload is gone: /v1/responses and
/// /v1/messages both had the order inverted and consequently rejected every
/// image with "image content part has no url" - on the Anthropic surface that
/// meant it could not accept a picture at all, since `source` is its only
/// inline image shape.
/// Content-part types the OpenAI surfaces accept, across both spellings.
/// `image`/`text` are what an Anthropic block or a Responses part normalizes
/// to; the rest are the wire names.
const KNOWN_PARTS: &[&str] = &[
    "text",
    "image_url",
    "file", // chat completions
    "input_text",
    "output_text",
    "input_image",
    "input_file", // responses
    "image",      // post-normalization / Anthropic-converted
    // audio: OpenAI's part type, vLLM's url extension, and the
    // post-normalization marker the ASR chat template keys on. Capability is
    // checked by the handler (chat.rs's find_audio + supports_audio), the
    // same split the image parts use - validation here only answers "is this
    // a part type the server understands at all".
    "input_audio",
    "audio_url",
    "audio",
];
/// Parts the spec defines that we deliberately do not serve. Named separately
/// so the refusal says why rather than "unknown".
const UNSERVED_PARTS: &[(&str, &str)] = &[(
    "refusal",
    "`refusal` parts belong to assistant history the server generated, not to a request",
)];

/// The official Qwen3-ASR chat template, verbatim from the checkpoint's
/// `chat_template.json` (Qwen/Qwen3-ASR-1.7B - the Omni-style file
/// transformers' AutoProcessor reads; `tokenizer_config.json` has none, so
/// GGUF conversions embed the generic ChatML fallback instead, which has no
/// audio branch at all). Served as the family override on an audio-serving
/// qwen3_asr endpoint: it is what vLLM renders, so chat
/// prompts match byte-for-byte. Its semantics are deliberately degenerate -
/// system text is collected across turns, the user turn carries only the
/// audio slots (`<|audio_start|><|audio_pad|><|audio_end|>` each), and user
/// TEXT is discarded - this model is a transcriber, not a chat model, and
/// that is upstream's contract, not our simplification.
pub const QWEN3_ASR_AUDIO_TEMPLATE: &str = "{%- set ns = namespace(system_text=\"\") -%}
{%- for m in messages -%}
  {%- if m.role == 'system' -%}
    {%- if m.content is string -%}
      {%- set ns.system_text = ns.system_text + m.content -%}
    {%- else -%}
      {%- for c in m.content -%}
        {%- if c.type == 'text' and (c.text is defined) -%}
          {%- set ns.system_text = ns.system_text + c.text -%}
        {%- endif -%}
      {%- endfor -%}
    {%- endif -%}
  {%- endif -%}
{%- endfor -%}

{%- set ns2 = namespace(audio_tokens=\"\") -%}
{%- for m in messages -%}
  {%- if m.content is not string -%}
    {%- for c in m.content -%}
      {%- if c.type == 'audio' or ('audio' in c) or ('audio_url' in c) -%}
        {%- set ns2.audio_tokens = ns2.audio_tokens + \"<|audio_start|><|audio_pad|><|audio_end|>\" -%}
      {%- endif -%}
    {%- endfor -%}
  {%- endif -%}
{%- endfor -%}

{{- '<|im_start|>system\\n' + (ns.system_text if ns.system_text is string else '') + '<|im_end|>\\n' -}}
{{- '<|im_start|>user\\n' + ns2.audio_tokens + '<|im_end|>\\n' -}}
{%- if add_generation_prompt -%}
{{- '<|im_start|>assistant\\n' -}}
{%- endif -%}";

/// The DeepSeek-OCR family's prompt shape (`DeepSeek-OCR`, `DeepSeek-OCR-2`,
/// `baidu/Unlimited-OCR` - arch `deepseek2-ocr`). Design + evidence:
///
/// These checkpoints ship no chat template at all - not in
/// `tokenizer_config.json`, not in `processor_config.json`, not as a
/// `chat_template.json`. The reference builds its prompt with
/// `format_messages(..., sft_format='plain')`, and `plain` is
/// `roles=("","")` / `sep=""` / `sep2=""`, so `get_prompt()` is a bare
/// concatenation of message content: no role tags, no separators, no system
/// prompt. (`<|User|>`/`<|Assistant|>` are in the vocab, but the plain style
/// discards the role names.) llama.cpp's converter, having nothing to copy,
/// writes the placeholder `{% for m in messages %}{{m['content']}}{% endfor %}`
/// into the GGUF - close to right, but with no image marker, no system
/// handling and no BOS. Hence a family override rather than the embedded one.
///
/// Matching the other implementations matters here because they are the parity
/// target: vLLM's `template_deepseek_ocr.jinja` emits `bos_token + system_message`
/// then concatenates raw content, and does not insert `<image>` (the caller
/// types it, and vLLM's processor expands the marker). SGLang instead sets
/// `image_token_at_prefix=True` and - uniquely for this family - suppresses
/// the `"\n"` it appends after the image token for every other model. We do
/// both: markers first, no newline, and the caller's own `<image>` wins.
///
/// So the two layers this template implements (the other two live in Rust):
///   1. PASS-THROUGH - if the message text already contains `<image>`, it is
///      used verbatim and we inject nothing. That is byte-identical to what
///      someone following the official README or the vLLM recipe gets, and it
///      is the layer the parity gate measures. If they wrote one marker but
///      sent two images, `build_mm_chunks` fails loudly on the count mismatch
///      rather than quietly mis-splicing.
///   3. INJECT - otherwise one `<image>` per image part, at the PREFIX, then
///      the message's text.
///
/// BOS is deliberately absent: `chat.rs` prepends `model.bos` when the
/// tokenized prompt does not already start with it, same as the other
/// BOS-leading families, so emitting it here would double it.
///
/// One `<image>` per image part is correct even though the reference writes a
/// single marker for a whole multi-page batch. Its `infer_multi` splits the
/// prompt on that one marker and appends every page's token run at that one
/// position, each run being `(num_queries+1) * num_queries` image-id tokens
/// plus one more as the inter-page separator. N adjacent markers, each
/// expanded to one page's run, yields the identical token stream - and it is
/// what `build_mm_chunks` is built for (one pad per media item). The invariant
/// that makes them equal is that nothing may sit between the markers, which is
/// why they are emitted as one unbroken prefix run.
/// PaddleOCR-VL's family template - the checkpoint's own ERNIE envelope
/// (`<|begin_of_sentence|>`, `User: `/`Assistant:\n`, `</s>` closes assistant
/// turns, `<|IMAGE_START|><|IMAGE_PLACEHOLDER|><|IMAGE_END|>` per image part,
/// images before same-message text).
///
/// This is the official GGUF's `tokenizer.chat_template` with one branch
/// aligned: the GGUF adds `content is string` arms the checkpoint template
/// lacks (which our all-text flatten routes through), but its system-string
/// arm drops the trailing newline the checkpoint's list walk emits
/// (`content["text"] + "\n"`). The arbiter (vLLM, checkpoint template, list
/// parts) emits that newline, so the string arm here appends it too - same
/// bytes for the same conversation on either wire shape. Verified
/// byte-identical to the reference processor's `apply_chat_template` on all
/// six task prompts + text-only + multi-turn (tests/paddleocr_template.rs).
pub const PADDLEOCR_VL_TEMPLATE: &str = r#"{%- if not add_generation_prompt is defined -%}
    {%- set add_generation_prompt = true -%}
{%- endif -%}
{%- if not cls_token is defined -%}
    {%- set cls_token = "<|begin_of_sentence|>" -%}
{%- endif -%}
{%- if not eos_token is defined -%}
    {%- set eos_token = "</s>" -%}
{%- endif -%}
{{- cls_token -}}
{%- for message in messages -%}
    {%- if message["role"] == "user" -%}
        {{- "User: " -}}
      {%- if message["content"] is string -%}
        {{- message["content"] }}
      {%- else -%}
        {%- for content in message["content"] -%}
            {%- if content["type"] == "image" -%}
                {{ "<|IMAGE_START|><|IMAGE_PLACEHOLDER|><|IMAGE_END|>" }}
            {%- endif -%}
        {%- endfor -%}
        {%- for content in message["content"] -%}
            {%- if content["type"] == "text" -%}
                {{ content["text"] }}
            {%- endif -%}
        {%- endfor -%}
      {%- endif -%}
        {{ "\n" -}}
    {%- elif message["role"] == "assistant" -%}
        {{- "Assistant:\n" -}}
      {%- if message["content"] is string -%}
        {{- message["content"] }}
      {%- else -%}
        {%- for content in message["content"] -%}
            {%- if content["type"] == "text" -%}
                {{ content["text"] }}
            {%- endif -%}
        {%- endfor -%}
      {%- endif -%}
        {{ eos_token -}}
    {%- elif message["role"] == "system" -%}
      {%- if message["content"] is string -%}
        {{- message["content"] + "\n" }}
      {%- else -%}
        {%- for content in message["content"] -%}
            {%- if content["type"] == "text" -%}
                {{ content["text"] + "\n" }}
            {%- endif -%}
        {%- endfor -%}
      {%- endif -%}
    {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{- "Assistant:\n" -}}
{%- endif -%}"#;

pub const DEEPSEEK_OCR_TEMPLATE: &str = "{%- set ns = namespace(sys=\"\", body=\"\") -%}
{%- for m in messages -%}
  {%- if m.role == 'system' -%}
    {%- if m.content is string -%}
      {%- set ns.sys = ns.sys + m.content -%}
    {%- else -%}
      {%- for c in m.content -%}
        {%- if c.type == 'text' and (c.text is defined) -%}
          {%- set ns.sys = ns.sys + c.text -%}
        {%- endif -%}
      {%- endfor -%}
    {%- endif -%}
  {%- elif m.content is string -%}
    {%- set ns.body = ns.body + m.content -%}
  {%- else -%}
    {%- set part = namespace(imgs=\"\", text=\"\") -%}
    {%- for c in m.content -%}
      {%- if c.type == 'image' or c.type == 'image_url' or c.type == 'input_image' or ('image' in c) or ('image_url' in c) -%}
        {%- set part.imgs = part.imgs + '<image>' -%}
      {%- elif c.type == 'text' and (c.text is defined) -%}
        {%- set part.text = part.text + c.text -%}
      {%- endif -%}
    {%- endfor -%}
    {%- if '<image>' in part.text -%}
      {%- set ns.body = ns.body + part.text -%}
    {%- else -%}
      {%- set ns.body = ns.body + part.imgs + part.text -%}
    {%- endif -%}
  {%- endif -%}
{%- endfor -%}
{{- ns.sys + ns.body -}}";

/// Collapse each message's content PARTS into the single string the
/// granite-speech family's own chat templates are written against, writing
/// `marker` (`<|audio|>`) once per audio part ahead of the message's text.
///
/// Why this rather than a family template override: granite-speech is
/// prompt-driven, and its two variants ship different envelopes - the base's
/// is `USER: {{ message['content'] }}\n ASSISTANT:`, while `-plus` ships the
/// full granite-4 envelope (role tags, tools, documents, and a default system
/// message). Both are written for a STRING content the caller has already
/// written `<|audio|>` into, which is exactly the model card's usage; neither
/// can take the OpenAI wire's parts LIST (the base renders a Jinja repr of it,
/// the plus's part walk only knows `type == 'text'` and drops the clip). So
/// the fix belongs on the MESSAGE, not the template: flatten here, then render
/// whatever envelope the checkpoint actually ships.
///
/// This replaced a hardcoded copy of the BASE envelope. That
/// copy served `-plus` a prompt 23 tokens short of its own template - the
/// whole system block - and the two variants' transcripts diverged in
/// punctuation and casing because of it. A hardcoded envelope is a label
/// under the label law: it must be derived from the file, not from the
/// sibling checkpoint that happened to arrive first.
///
/// Order matters and is the card's: audio marker first, instruction after
/// (`"<|audio|>transcribe the speech with proper punctuation and
/// capitalization."`), regardless of the order the parts arrived in. The
/// user's text is kept, unlike the Qwen3-ASR template above which discards it
/// - on granite-speech that text is the task selector (raw vs punctuated
///   transcript, keyword biasing, translation), so dropping it would silently
///   collapse four capabilities into one.
pub fn inline_audio_content(messages: &mut [serde_json::Value], marker: &str) {
    for msg in messages {
        let Some(parts) = msg.get("content").and_then(|c| c.as_array()) else {
            continue; // already a string - the model card's own shape
        };
        let (mut audio, mut text) = (String::new(), String::new());
        for part in parts {
            let ty = part
                .get("type")
                .and_then(|t| t.as_str())
                .unwrap_or_default();
            if ty == "audio" || part.get("audio").is_some() || part.get("audio_url").is_some() {
                audio.push_str(marker);
            } else if let Some(t) = part.get("text").and_then(|t| t.as_str()) {
                text.push_str(t);
            }
        }
        msg["content"] = serde_json::Value::String(audio + &text);
    }
}

/// Reject any content part we would otherwise drop on the floor.
///
/// Without this a part type we don't know reaches the template, where the
/// outcome depends entirely on how that template was written: qwen's tests
/// `'text' in item` and skips it silently, granite's tests the type string and
/// errors with an internal Jinja message. Both are wrong answers to "the
/// client sent something we don't serve" - the first loses content without
/// telling anyone, the second blames the template. Name the type instead.
///
/// (The Anthropic surface has always done this - `unsupported content block
/// type ...` - which is where the wording comes from.)
pub fn validate_content_parts(messages: &[serde_json::Value]) -> Result<(), String> {
    for msg in messages {
        let Some(parts) = msg.get("content").and_then(|c| c.as_array()) else {
            continue;
        };
        for part in parts {
            let ty = part
                .get("type")
                .and_then(|t| t.as_str())
                .unwrap_or_default();
            if KNOWN_PARTS.contains(&ty) {
                continue;
            }
            if let Some((_, why)) = UNSERVED_PARTS.iter().find(|(n, _)| *n == ty) {
                return Err((*why).to_owned());
            }
            return Err(format!(
                "unsupported content part type {ty:?} (this server accepts text, images, \
                 audio and PDF files)"
            ));
        }
    }
    Ok(())
}

/// Message roles the OpenAI chat schema defines. `function` is deprecated but
/// still in the spec, so it is accepted; anything else is a typo or a client
/// bug, and rendering it as if it were a user turn would be a silent one.
const KNOWN_ROLES: &[&str] = &[
    "system",
    "developer",
    "user",
    "assistant",
    "tool",
    "function",
];

pub fn validate_roles(messages: &[serde_json::Value]) -> Result<(), String> {
    for msg in messages {
        let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or_default();
        if !KNOWN_ROLES.contains(&role) {
            return Err(format!(
                "unsupported message role {role:?} (expected one of {})",
                KNOWN_ROLES.join(", ")
            ));
        }
    }
    Ok(())
}

pub fn normalize_messages(messages: &[serde_json::Value]) -> Vec<serde_json::Value> {
    let mut out = messages.to_vec();
    for msg in &mut out {
        // OpenAI image parts arrive as {"type":"image_url","image_url":{...}};
        // gemma4's template only renders an image slot for type == "image"
        // (qwen's accepts both). The pixels were already extracted - the
        // template only needs the type marker.
        if let Some(parts) = msg.get_mut("content").and_then(|c| c.as_array_mut()) {
            for part in parts.iter_mut() {
                let is_img = part.get("image_url").is_some()
                    || part.get("image").is_some()
                    || part.get("type").and_then(|t| t.as_str()) == Some("image_url");
                if is_img {
                    *part = serde_json::json!({"type": "image"});
                    continue;
                }
                // Audio parts: the payload was already extracted
                // by find_audio; the template only needs the type marker.
                // OpenAI spells it `input_audio`, vLLM adds `audio_url` - the
                // ASR template's condition matches `type == 'audio'`, which is
                // also exactly what vLLM canonicalizes both spellings to
                // before rendering, so the rendered bytes agree.
                let ty = part
                    .get("type")
                    .and_then(|t| t.as_str())
                    .unwrap_or_default();
                if part.get("input_audio").is_some()
                    || part.get("audio_url").is_some()
                    || matches!(ty, "input_audio" | "audio_url")
                {
                    *part = serde_json::json!({"type": "audio"});
                    continue;
                }
                // /v1/responses passes image-bearing content arrays through
                // VERBATIM, so text parts still wear their Responses type
                // (`input_text`). Templates that key off `'text' in item`
                // (qwen) never noticed; templates that test the type STRING
                // (granite) skipped every one of them - an image sent with a
                // question over Responses arrived with no question, silently.
                // Rewrite to the chat-completions spelling, which both shapes
                // of template match.
                let ty = part
                    .get("type")
                    .and_then(|t| t.as_str())
                    .unwrap_or_default();
                if matches!(ty, "input_text" | "output_text") {
                    let text = part
                        .get("text")
                        .and_then(|t| t.as_str())
                        .unwrap_or_default();
                    *part = serde_json::json!({"type": "text", "text": text});
                }
            }
            // A content array that is all text flattens to the plain string -
            // the spec defines the array form as exactly that concatenation.
            // This is what lets string-templated models take the multi-part
            // spelling at all: gpt-oss's Harmony template concatenates
            // `content` with `+` and blew up on a sequence ("tried to use +
            // operator on unsupported types string and sequence") - the
            // standard OpenAI part shape was a 400 on gpt-oss.
            // Arrays carrying image parts stay arrays;
            // the vision templates key on them.
            let all_text = parts
                .iter()
                .all(|p| p.get("type").and_then(|t| t.as_str()) == Some("text"));
            if all_text {
                let joined: String = parts
                    .iter()
                    .filter_map(|p| p.get("text").and_then(|t| t.as_str()))
                    .collect();
                msg["content"] = serde_json::Value::String(joined);
            }
        }
        if let Some(obj) = msg.as_object_mut()
            && obj.get("content").is_some_and(serde_json::Value::is_null)
        {
            obj.remove("content");
        }
        // An assistant tool-call turn has null content on the wire (that is
        // the OpenAI shape, and the Anthropic conversion produces the same).
        // Give it an empty STRING instead of nothing: gpt-oss's Harmony
        // template concatenates `message.content` unguarded, so the canonical
        // agent history - the second turn of every tool loop - was a 400
        // ("+ operator on string and undefined").
        // "" is falsy in every template that branches on content, so the
        // families that guard render identically.
        if let Some(obj) = msg.as_object_mut()
            && obj
                .get("tool_calls")
                .and_then(|c| c.as_array())
                .is_some_and(|c| !c.is_empty())
            && obj.get("content").is_none()
        {
            obj.insert("content".into(), serde_json::Value::String(String::new()));
        }
        let Some(calls) = msg.get_mut("tool_calls").and_then(|c| c.as_array_mut()) else {
            continue;
        };
        for call in calls {
            let Some(args) = call
                .get_mut("function")
                .and_then(|f| f.get_mut("arguments"))
            else {
                continue;
            };
            if let Some(s) = args.as_str()
                && let Ok(parsed) = serde_json::from_str::<serde_json::Value>(s)
                && parsed.is_object()
            {
                *args = parsed;
            }
        }
    }
    out
}

/// One task tag a template expands, with the instruction it expands to.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct TaskTag {
    /// the literal a user puts in their message, angle brackets included
    pub tag: String,
    /// exactly what the model receives instead of that message
    pub prompt: String,
}

/// Task tags this template EXPANDS - literals like `<chart2csv>` that it swaps
/// for a long canned instruction the model was fine-tuned on.
///
/// granite-vision is the first model we serve with these, and they are its real
/// interface: `<chart2csv>` becomes IBM's exact 300-character CSV-extraction
/// prompt, and asking the same thing in your own words lands on the open-ended
/// path IBM's own card warns may not generalize. So a client that cannot see
/// the tag list cannot reach most of what the model does.
///
/// Read from the TEMPLATE rather than declared in the catalog, because the
/// template is what actually expands them - a declared list is a second place
/// to keep in step, and the failure mode is silent (a tag reaching the model
/// unexpanded is a worse answer, not an error).
///
/// Two stages, because neither alone is trustworthy:
///
/// 1. **Scan** for `"<name>"` string literals (quotes adjacent, lowercase
///    identifier inside). That's the shape of the dispatcher's membership
///    tests, and the adjacent quotes keep the OTSL cell tokens the tables_otsl
///    instruction *describes* (`<fcel>`, `<nl>`, ...) out of the candidates -
///    they live mid-string, not as literals of their own.
/// 2. **Confirm by rendering.** A candidate is a task tag only if the template
///    actually replaces it with something else. `<image>` is the reason this
///    stage exists: it passes the scan (the dispatcher tests for it the same
///    way) but renders straight through, so it drops out here.
///
/// Returns them in template order. Empty for every model we serve but granite -
/// no template scanned so far even has candidates past `<image>`.
pub fn task_tags(template: &str) -> Vec<TaskTag> {
    // A pathological template shouldn't turn model load into hundreds of
    // renders; nothing real is near this.
    const MAX_CANDIDATES: usize = 32;
    // Rendered in place of a user message to learn what the template wraps
    // content in. Plain ASCII with no markup so nothing can escape or split it.
    const PROBE: &str = "PADDOCKTAGPROBE";

    let candidates = scan_tag_literals(template, MAX_CANDIDATES);
    if candidates.is_empty() {
        return Vec::new();
    }
    // The wrapper, learned once: everything before and after the user's own
    // text. Trimmed at the seam because a template's content macro may pad the
    // text with whitespace it does not emit around an expansion (granite's
    // does), so an untrimmed prefix wouldn't match the expanded render.
    let probe = render_user(template, PROBE).unwrap_or_default();
    let mut split = probe.split(PROBE);
    let (Some(head), Some(tail)) = (split.next(), split.next()) else {
        return Vec::new(); // template ignored the message entirely
    };
    if split.next().is_some() {
        return Vec::new(); // rendered twice - can't tell which one expanded
    }
    let (head, tail) = (head.trim_end(), tail.trim_start());

    let mut out = Vec::new();
    for tag in candidates {
        let Some(rendered) = render_user(template, &tag) else {
            continue;
        };
        // Same wrapper, so whatever sits between it is what the tag became.
        let Some(mid) = rendered
            .strip_prefix(head)
            .and_then(|r| r.strip_suffix(tail))
        else {
            continue;
        };
        let prompt = mid.trim();
        // Unexpanded (`<image>`), or swallowed outright: not a task tag.
        if prompt.is_empty() || prompt.contains(&tag) {
            continue;
        }
        out.push(TaskTag {
            tag,
            prompt: prompt.to_owned(),
        });
    }
    out
}

/// Render one user message through `template`, or None if it won't render.
fn render_user(template: &str, text: &str) -> Option<String> {
    let msgs = [serde_json::json!({"role": "user", "content": text})];
    render(template, &msgs, None, None).ok()
}

/// Every `"<name>"` / `'<name>'` literal in the template source, in order,
/// deduplicated. `name` must be a lowercase identifier - that excludes role
/// markers like `'<|start_of_role|>'` and anything with punctuation or spaces.
fn scan_tag_literals(template: &str, limit: usize) -> Vec<String> {
    let bytes = template.as_bytes();
    let mut out: Vec<String> = Vec::new();
    let mut i = 0;
    while i + 3 < bytes.len() {
        let quote = bytes[i];
        if (quote != b'"' && quote != b'\'') || bytes[i + 1] != b'<' {
            i += 1;
            continue;
        }
        // scan the identifier, then require '>' immediately followed by the
        // same quote - "<image>\n" is a different literal and not a tag
        let start = i + 2;
        let mut j = start;
        while j < bytes.len()
            && (bytes[j].is_ascii_lowercase() || bytes[j].is_ascii_digit() || bytes[j] == b'_')
        {
            j += 1;
        }
        if j > start && j + 1 < bytes.len() && bytes[j] == b'>' && bytes[j + 1] == quote {
            let tag = format!("<{}>", &template[start..j]);
            if !out.contains(&tag) {
                out.push(tag);
                if out.len() >= limit {
                    return out;
                }
            }
            i = j + 2;
            continue;
        }
        i += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{normalize_messages, parenthesize_call_ternaries, render, task_tags};
    use serde_json::json;

    /// Chat templates are authored against transformers' jinja2 environment,
    /// which sets `trim_blocks=True, lstrip_blocks=True`. minijinja defaults
    /// both off, so a template that leans on them instead of writing `{%-`
    /// everywhere renders with whitespace the model never saw.
    ///
    /// This is granite 4.2's opening shape reduced to its essentials: a run of
    /// `{%- set %}` lines whose CLOSING tags carry no `-`. Without trim_blocks
    /// each of those newlines survives and the prompt picks up a leading blank
    /// line - measured as one extra prompt token against llama.cpp on the
    /// identical GGUF (29 vs 28), on every single conversation.
    #[test]
    fn block_tags_do_not_leak_their_trailing_newline() {
        let tpl = "{%- set a = 1 %}\n{%- set b = 2 %}\n<|im_start|>user\n\
                   {{ messages[0]['content'] }}<|im_end|>\n";
        let msgs = [json!({"role": "user", "content": "hi"})];
        let out = render(tpl, &msgs, None, None).expect("renders");
        assert!(
            !out.starts_with('\n'),
            "template rendered a leading newline - trim_blocks is off: {out:?}"
        );
        // The template's own final newline is dropped, which is jinja2's
        // default too (`keep_trailing_newline=False`) - so this matches the
        // reference rather than diverging from it.
        assert_eq!(out, "<|im_start|>user\nhi<|im_end|>");
    }

    /// lstrip_blocks is the other half: indentation in front of a block tag is
    /// layout in the template, not content for the model.
    #[test]
    fn indentation_before_a_block_tag_is_not_content() {
        let tpl = "    {% if true %}\n<|im_start|>user\n    {% endif %}done";
        let msgs = [json!({"role": "user", "content": "hi"})];
        let out = render(tpl, &msgs, None, None).expect("renders");
        assert!(
            !out.starts_with(' '),
            "leading indentation leaked into the prompt: {out:?}"
        );
    }

    /// The official ASR template end-to-end through our normalize + render
    /// pipeline: OpenAI `input_audio` parts normalize
    /// to `{"type":"audio"}` markers, and the render emits exactly one
    /// `<|audio_start|><|audio_pad|><|audio_end|>` triple per clip with the
    /// system text kept and the user text DISCARDED (upstream's contract -
    /// this is what vLLM renders, byte for byte).
    #[test]
    fn asr_template_renders_one_pad_triple_per_audio_part() {
        let msgs = [
            json!({"role": "system", "content": "Bias: paddock, laguna."}),
            json!({"role": "user", "content": [
                {"type": "text", "text": "this text is discarded by the template"},
                {"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}},
                {"type": "input_audio", "input_audio": {"data": "BBBB", "format": "wav"}},
            ]}),
        ];
        let normalized = normalize_messages(&msgs);
        // payload gone, marker present - find_audio extracted the bytes first
        assert_eq!(normalized[1]["content"][1], json!({"type": "audio"}));
        let out = super::render(super::QWEN3_ASR_AUDIO_TEMPLATE, &normalized, None, None)
            .expect("render");
        assert_eq!(
            out,
            "<|im_start|>system\nBias: paddock, laguna.<|im_end|>\n\
             <|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|>\
             <|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n\
             <|im_start|>assistant\n"
        );
    }

    /// The family's own shape: no role tags, no separators, no BOS (chat.rs
    /// prepends it), image markers at the PREFIX with no trailing newline.
    #[test]
    fn ocr_template_prefixes_one_marker_per_image() {
        let msgs = [json!({"role": "user", "content": [
            {"type": "text", "text": "document parsing."},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ]})];
        let out = super::render(super::DEEPSEEK_OCR_TEMPLATE, &msgs, None, None).expect("render");
        // marker first even though the text part arrived first, and nothing
        // between it and the text - no "\n", no role tag, no BOS.
        assert_eq!(out, "<image>document parsing.");
    }

    /// Multi-page: N adjacent markers, nothing between them. That adjacency is
    /// what makes this token-identical to the reference's single marker with
    /// every page's run concatenated at it.
    #[test]
    fn ocr_template_emits_one_marker_per_page_unbroken() {
        let msgs = [json!({"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,BBBB"}},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,CCCC"}},
            {"type": "text", "text": "Multi page parsing."},
        ]})];
        let out = super::render(super::DEEPSEEK_OCR_TEMPLATE, &msgs, None, None).expect("render");
        assert_eq!(out, "<image><image><image>Multi page parsing.");
    }

    /// Layer 1: a caller who wrote the family vocabulary themselves - the
    /// official README's and vLLM's usage - gets it back untouched. Injecting
    /// a second marker here would silently double the image slots.
    #[test]
    fn ocr_template_passes_through_a_caller_written_marker() {
        let msgs = [json!({"role": "user", "content": [
            {"type": "text", "text": "<image>\n<|grounding|>Given the layout of the image."},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ]})];
        let out = super::render(super::DEEPSEEK_OCR_TEMPLATE, &msgs, None, None).expect("render");
        assert_eq!(out, "<image>\n<|grounding|>Given the layout of the image.");
        assert_eq!(
            out.matches("<image>").count(),
            1,
            "must not inject a second marker"
        );
    }

    /// A string content is the model card's own shape and passes straight
    /// through; a system message is prepended (vLLM's template does this) but
    /// we never invent one - the reference passes system_prompt=''.
    #[test]
    fn ocr_template_string_content_and_system() {
        let bare = [json!({"role": "user", "content": "<image>Free OCR."})];
        assert_eq!(
            super::render(super::DEEPSEEK_OCR_TEMPLATE, &bare, None, None).expect("render"),
            "<image>Free OCR."
        );
        let with_sys = [
            json!({"role": "system", "content": "Answer in Swedish."}),
            json!({"role": "user", "content": "<image>document parsing."}),
        ];
        assert_eq!(
            super::render(super::DEEPSEEK_OCR_TEMPLATE, &with_sys, None, None).expect("render"),
            "Answer in Swedish.<image>document parsing."
        );
    }

    /// granite-speech-4.1-2b's own template, verbatim from its GGUF. The
    /// flattened content is what it is written against, so what comes out is
    /// the model card's prompt.
    fn granite_base_template() -> &'static str {
        "{% for message in messages %}{% if message['role'] == 'user' %}USER: \
         {{ message['content'] }}\n ASSISTANT:{% elif message['role'] == 'assistant' %}\
         {{ message['content'] }}{% endif %}{% endfor %}"
    }

    /// Flattening puts `<|audio|>` before the instruction (the model card's
    /// order) and keeps the user's text - unlike the Qwen3-ASR template, which
    /// discards it; on this family that text is the task selector.
    /// The exact expression muse-glimmer's template ships - plain Jinja2 that
    /// minijinja's parser rejects outright, so without the rewrite the model
    /// cannot answer a single chat request.
    #[test]
    fn call_kwarg_ternary_parses() {
        let t = "{%- set ns = namespace(name=tcid if tcid else 'fallback') -%}{{- ns.name -}}";
        let msgs = [serde_json::json!({"role": "user", "content": "x"})];
        assert_eq!(render(t, &msgs, None, None).unwrap(), "fallback");
        assert_eq!(
            render(t, &msgs, None, Some(&serde_json::json!({"tcid": "call_7"}))).unwrap(),
            "call_7"
        );
    }

    /// Positional arguments hit the same parser gap, and a ternary that is
    /// already parenthesized must not be double-wrapped into something else.
    #[test]
    fn positional_ternary_and_idempotence() {
        let msgs = [serde_json::json!({"role": "user", "content": "x"})];
        let t = "{{- [1, 2, 3] | join('a' if messages else 'b') -}}";
        assert_eq!(render(t, &msgs, None, None).unwrap(), "1a2a3");
        let already = "{{- [1, 2] | join(('a' if messages else 'b')) -}}";
        assert_eq!(render(already, &msgs, None, None).unwrap(), "1a2");
    }

    /// The rewrite must not touch template TEXT, string literals, or `==`.
    #[test]
    fn rewrite_leaves_text_and_comparisons_alone() {
        let untouched = [
            "plain text with (parens) and the word if in it",
            "{{- 'a literal ( if b else ) stays' -}}",
            "{%- if messages == none -%}x{%- endif -%}",
        ];
        for t in untouched {
            assert!(
                matches!(
                    parenthesize_call_ternaries(t),
                    std::borrow::Cow::Borrowed(_)
                ),
                "rewrote {t:?} - it has no call-argument ternary"
            );
        }
        // and a real one is rewritten, at the value rather than the whole arg
        let got = parenthesize_call_ternaries("{{- f(k=a if b else c) -}}");
        assert_eq!(got, "{{- f(k=(a if b else c)) -}}");
    }

    #[test]
    fn granite_speech_flatten_renders_the_card_envelope() {
        let msgs = [json!({"role": "user", "content": [
            {"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}},
            {"type": "text", "text": "transcribe the speech with proper punctuation and capitalization."},
        ]})];
        let mut normalized = normalize_messages(&msgs);
        super::inline_audio_content(&mut normalized, "<|audio|>");
        assert_eq!(
            normalized[0]["content"],
            json!("<|audio|>transcribe the speech with proper punctuation and capitalization.")
        );
        let out = super::render(granite_base_template(), &normalized, None, None).expect("render");
        assert_eq!(
            out,
            "USER: <|audio|>transcribe the speech with proper punctuation and \
             capitalization.\n ASSISTANT:"
        );
    }

    /// Text-before-audio in the parts list still flattens audio-first: the
    /// marker's position is the model's contract, not the caller's ordering.
    /// And a plain STRING content passes through untouched, so a caller
    /// following the model card verbatim gets byte-identical bytes.
    #[test]
    fn granite_speech_flatten_is_order_stable_and_string_compatible() {
        let parts = [json!({"role": "user", "content": [
            {"type": "text", "text": "translate the speech to German."},
            {"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}},
        ]})];
        let mut msgs = normalize_messages(&parts);
        super::inline_audio_content(&mut msgs, "<|audio|>");
        let out = super::render(granite_base_template(), &msgs, None, None).expect("render");
        assert_eq!(
            out,
            "USER: <|audio|>translate the speech to German.\n ASSISTANT:"
        );

        let mut string =
            normalize_messages(&[json!({"role": "user", "content": "<|audio|>transcribe."})]);
        super::inline_audio_content(&mut string, "<|audio|>");
        assert_eq!(string[0]["content"], json!("<|audio|>transcribe."));
        let out = super::render(granite_base_template(), &string, None, None).expect("render");
        assert_eq!(out, "USER: <|audio|>transcribe.\n ASSISTANT:");
    }

    /// The `-plus` variant ships the full granite-4 envelope - role tags and a
    /// default system message - and its own part walk only knows
    /// `type == 'text'`, so an un-flattened clip would vanish and the prompt
    /// would be 23 tokens short of what the checkpoint asks for. Flattening
    /// first is what lets one code path serve both variants' own templates.
    #[test]
    fn granite_speech_plus_envelope_keeps_the_clip() {
        let plus = "{%- if messages[0].role != 'system' -%}\
            {{- '<|start_of_role|>system<|end_of_role|>be safe<|end_of_text|>\n' -}}{%- endif -%}\
            {%- for m in messages -%}\
            {%- set c = namespace(val='') -%}\
            {%- if m.content is string -%}{%- set c.val = m.content -%}\
            {%- else -%}{%- for e in m.content -%}{%- if e.type == 'text' -%}\
            {%- set c.val = c.val + e.text -%}{%- endif -%}{%- endfor -%}{%- endif -%}\
            {{- '<|start_of_role|>' + m.role + '<|end_of_role|>' + c.val + '<|end_of_text|>\n' -}}\
            {%- endfor -%}{{- '<|start_of_role|>assistant<|end_of_role|>' -}}";
        let msgs = [json!({"role": "user", "content": [
            {"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}},
            {"type": "text", "text": "Transcribe this audio."},
        ]})];

        // un-flattened: the clip is dropped on the floor
        let bare = super::render(plus, &normalize_messages(&msgs), None, None).expect("render");
        assert!(
            !bare.contains("<|audio|>"),
            "control: the plus walk drops audio parts"
        );

        let mut flat = normalize_messages(&msgs);
        super::inline_audio_content(&mut flat, "<|audio|>");
        let out = super::render(plus, &flat, None, None).expect("render");
        assert!(out.contains("<|start_of_role|>system<|end_of_role|>be safe"));
        assert!(out.contains("<|audio|>Transcribe this audio."));
    }

    /// vLLM's `audio_url` spelling normalizes to the same marker.
    #[test]
    fn audio_url_parts_normalize_to_the_audio_marker() {
        let msgs = [json!({"role": "user", "content": [
            {"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64,AAAA"}},
        ]})];
        let normalized = normalize_messages(&msgs);
        assert_eq!(normalized[0]["content"][0], json!({"type": "audio"}));
    }

    /// The scan must not mistake a role marker, a padded literal, or the cell
    /// tokens an instruction merely DESCRIBES for a tag of its own.
    #[test]
    fn only_bare_lowercase_literals_are_candidates() {
        let tpl = r#"
            {%- set a = "<chart2csv>" in text -%}
            {%- set b = '<tables_json>' in text -%}
            {%- set pad = "<image>\n" -%}
            {%- set role = '<|start_of_role|>' -%}
            {%- set desc = "use <fcel> for a cell and <nl> for a new line" -%}
            {%- set upper = "<NotATag>" -%}
        "#;
        assert_eq!(
            super::scan_tag_literals(tpl, 32),
            ["<chart2csv>", "<tables_json>"]
        );
    }

    /// The confirm stage, which is the whole reason the extractor renders at
    /// all: `<image>` is tested for exactly like a task tag but passes through,
    /// so it must not be advertised as one.
    #[test]
    fn a_tag_the_template_does_not_expand_is_not_advertised() {
        let tpl = "{% for m in messages %}\
            {%- if '<image>' in m.content -%}{{ m.content }}\
            {%- elif '<summarize>' in m.content -%}Summarize this.\
            {%- else -%}{{ m.content }}{%- endif -%}{% endfor %}";
        let tags = task_tags(tpl);
        assert_eq!(tags.len(), 1, "{tags:?}");
        assert_eq!(tags[0].tag, "<summarize>");
        assert_eq!(tags[0].prompt, "Summarize this.");
    }

    /// An ordinary template has none, and must not pay for a scan that finds
    /// nothing interesting.
    #[test]
    fn a_template_without_tags_reports_none() {
        let tpl = "{% for m in messages %}<|{{ m.role }}|>{{ m.content }}{% endfor %}";
        assert!(task_tags(tpl).is_empty());
    }

    /// `str.index` is what granite-vision's template uses to order the image
    /// against the expanded task instruction. Offsets are CHARACTERS (Python's
    /// unit), so a multi-byte prefix must not inflate them.
    #[test]
    fn str_index_returns_character_offsets_and_raises_when_absent() {
        let render1 =
            |tpl: &str| super::render(tpl, &[json!({"role": "user", "content": "x"})], None, None);
        assert_eq!(render1("{{ 'abcXYZ'.index('XYZ') }}").unwrap(), "3");
        assert_eq!(render1("{{ 'aXbXc'.rindex('X') }}").unwrap(), "3");
        // 'é' is two bytes; a byte offset would say 4 here
        assert_eq!(render1("{{ 'ééé<img>'.index('<img>') }}").unwrap(), "3");
        let err = render1("{{ 'abc'.index('zzz') }}").unwrap_err();
        assert!(err.contains("not found"), "{err}");
        // start/end bounds are refused rather than silently ignored
        let err = render1("{{ 'abcabc'.index('a', 1) }}").unwrap_err();
        assert!(err.contains("exactly one argument"), "{err}");
    }

    /// The new callback must not shadow anything pycompat already provides.
    #[test]
    fn pycompat_methods_still_resolve_through_the_wrapper() {
        let render1 = |tpl: &str| {
            super::render(tpl, &[json!({"role": "user", "content": "x"})], None, None).unwrap()
        };
        assert_eq!(render1("{{ 'abcXYZ'.find('XYZ') }}"), "3");
        assert_eq!(render1("{{ 'abc'.find('zzz') }}"), "-1");
        assert_eq!(render1("{{ 'a,b'.split(',') | join('-') }}"), "a-b");
        assert_eq!(render1("{{ 'Ab'.upper() }}"), "AB");
        // Booleans render Python-style, "True"/"False" - which is what
        // Jinja2 and llama.cpp's minja produce, so this is the FAITHFUL
        // spelling, not a quirk. minijinja rendered "true" through 2.21 and
        // corrected it in 2.23; the served templates are unaffected either way
        // (none of the six emits a bare boolean - gemma4 and gpt-oss build the
        // strings themselves, and `| tojson` still emits lowercase JSON).
        assert_eq!(render1("{{ 'abc'.startswith('ab') }}"), "True");
        assert_eq!(render1("{{ {'a': true} | tojson }}"), "{\"a\":true}");
    }

    /// The precondition, as a test: this function is DESTRUCTIVE to image
    /// parts. /v1/responses and /v1/messages both called it before extracting,
    /// so every image request on those surfaces died with "image content part
    /// has no url" - the Anthropic one could not take a picture at all. If a
    /// future refactor moves normalization ahead of extraction again, this is
    /// the test that says why it must not.
    #[test]
    fn normalize_strips_image_payloads_so_extraction_must_run_first() {
        let msgs = [json!({"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            {"type": "text", "text": "what is this?"},
        ]})];
        let out = normalize_messages(&msgs);
        let parts = out[0]["content"].as_array().expect("array");
        assert_eq!(
            parts[0],
            json!({"type": "image"}),
            "payload is gone by design"
        );
        assert_eq!(parts[1]["text"], "what is this?", "text parts survive");
        // and the extractor can no longer find anything to decode
        assert!(crate::chat::find_images(&out).is_err());
    }

    /// The Responses surface hands us `input_text` alongside the image. A
    /// template that tests `chunk['type'] == 'text'` (granite's does) skips
    /// that verbatim, which cost the user their entire question with no error
    /// anywhere - so normalization has to spell text parts the one way every
    /// template we serve recognizes.
    #[test]
    fn responses_text_parts_are_respelled_for_the_template() {
        let msgs = [json!({"role": "user", "content": [
            {"type": "input_text", "text": "what is this?"},
            {"type": "input_image", "image_url": "data:image/png;base64,AAAA"},
        ]})];
        let out = normalize_messages(&msgs);
        let parts = out[0]["content"].as_array().expect("array");
        assert_eq!(parts[0], json!({"type": "text", "text": "what is this?"}));
        assert_eq!(parts[1], json!({"type": "image"}));
        // an ALL-text array flattens to the plain string - the spec's own
        // semantics for the array form, and what lets string-concatenating
        // templates (gpt-oss's Harmony) take the multi-part spelling at all
        let msgs = [json!({"role": "user", "content": [
            {"type": "text", "text": "hi "},
            {"type": "input_text", "text": "there"},
        ]})];
        let out = normalize_messages(&msgs);
        assert_eq!(out[0]["content"], json!("hi there"));
    }

    #[test]
    fn null_content_normalizes_per_shape() {
        // a null-content message without tool calls just loses the key
        let out = normalize_messages(&[json!({"role": "assistant", "content": null})]);
        assert!(out[0].get("content").is_none(), "{:?}", out[0]);
        let msgs = [json!({
            "role": "assistant",
            "content": null,
            "tool_calls": [{"id": "c1", "type": "function",
                            "function": {"name": "f", "arguments": "{\"x\": 1}"}}]
        })];
        let out = normalize_messages(&msgs);
        // a tool-call turn gets empty content, not a missing key - Harmony's
        // template concatenates content unguarded
        assert_eq!(out[0]["content"], json!(""), "{:?}", out[0]);
        // arguments-string still parsed back to an object
        assert!(out[0]["tool_calls"][0]["function"]["arguments"].is_object());
        // non-null content untouched
        let out = normalize_messages(&[json!({"role": "user", "content": "hi"})]);
        assert_eq!(out[0]["content"], "hi");
    }
}
