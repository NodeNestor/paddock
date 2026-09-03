//! Progressive MCP tool disclosure - the "search, don't dump" path.
//!
//! When a chat's enabled MCP servers expose more tools than a model should hold
//! in context, injecting every schema is both wasteful and degrades the model's
//! selection accuracy (and, at small context windows, overflows the prompt). So
//! above a threshold we inject just two synthetic tools -
//!   * `mcp_search_tools(query, limit)` - returns matching tools *with their
//!     input schema inline*, so the model can call one immediately, and
//!   * `mcp_call_tool(name, arguments_json)` - a generic executor for a tool the
//!     model discovered but doesn't have listed.
//!     and rank the catalog here.
//!
//! Ranking is a self-contained **Okapi BM25** over each tool's name (weighted),
//! description, and parameter names/descriptions - with snake_case/camelCase
//! tokenization so `search_lei` matches "lei". Deliberately no embedding/reranker
//! model: tool routing must never require the user to load a second model. For a
//! local host's tool counts (tens-low hundreds) lexical retrieval is both
//! sufficient and instant. This mirrors the single-turn discover->use design used
//! by production MCP gateways (schema returned inline; no mid-turn tool
//! registration, which SDKs don't surface).

use std::collections::{HashMap, HashSet};

use serde_json::{Value, json};

/// The reserved synthetic tool names paddock intercepts in the agent loop; they
/// are never routed to an MCP server.
pub const SEARCH_TOOL: &str = "mcp_search_tools";
pub const CALL_TOOL: &str = "mcp_call_tool";

/// Switch a request from direct tool injection to search-mode disclosure once the
/// scoped catalog exceeds this many tools. (Also triggered by a token-budget
/// check at the call site, so verbose schemas can't overflow a small context.)
pub const SEARCH_DISCLOSURE_THRESHOLD: usize = 16;

/// Told to the model when the schemas are HIDDEN (progressive disclosure on).
///
/// The two meta-tools describe themselves, but nothing said the real tools are
/// behind them - so a model with no matching tool in sight can simply conclude
/// it has none and answer without searching. Same failure the artifact tool hit:
/// a capability it must know exists before it can look for it.
pub const SEARCH_MODE_INSTRUCTIONS: &str = "Only two tool-discovery tools are listed, not the full set: the connected servers \
     expose many more. Before answering that you lack a capability, call \
     `mcp_search_tools` with keywords for what you need, then invoke what it returns \
     with `mcp_call_tool`.";

/// Told to the model when the schemas are visible.
///
/// The pair is declared in every mode now, so the mention has to be too -
/// leaving it conditional meant the model always had the search tools and was
/// only sometimes told they existed. Short, because here it is a fallback
/// rather than the main route.
pub const SEARCH_AVAILABLE_INSTRUCTIONS: &str = "Your tool list may not show every tool the connected servers expose. If the one you \
     need is missing, call `mcp_search_tools` with keywords to find it and \
     `mcp_call_tool` to run it.";

/// Told to the model when some servers kept their schemas and others did not.
///
/// Naming the hidden servers is the cheap part that earns its tokens: a model
/// that can see `tic` is missing knows there is a company/registry capability
/// worth searching for. Without the names it has to first guess that anything
/// is missing at all - the same blind spot the all-hidden text exists to fix.
/// A template, not a `format!`, so the Studio's system-prompt panel can render
/// the same sentence from `/api/server` instead of keeping a second copy of it.
pub const SEARCH_PARTIAL_TEMPLATE: &str = "Your tool list is partial: {tools} more tools are available from these servers but are \
     not listed - {servers}. Call the tools you CAN see directly, by name. For anything else, \
     call `mcp_search_tools` with keywords for what you need and run what it returns with \
     `mcp_call_tool`. Never answer that you lack a capability without searching first.";

pub fn partial_mode_instructions(hidden_labels: &[String], hidden_tools: usize) -> String {
    SEARCH_PARTIAL_TEMPLATE
        .replace("{tools}", &hidden_tools.to_string())
        .replace("{servers}", &hidden_labels.join(", "))
}

/// One server's weight in the disclosure decision: how many tools it declares
/// and how many characters their schemas cost.
pub struct ServerWeight {
    pub label: String,
    pub tools: usize,
    pub chars: usize,
}

/// Pick the servers that keep their real schemas; the rest go behind search.
///
/// Disclosure used to be one global switch, and that was a correctness bug and
/// not just a latency one. Past the threshold every schema vanished at once, so
/// every call travelled as `mcp_call_tool(name, arguments_json)` - where the
/// real arguments are a *string* the constrained-decoding grammar cannot see.
/// The guarantee we rely on ("a malformed tool call is unrepresentable") silently
/// did not hold for any MCP tool. With one busy ~40-tool connector attached that
/// is every request, and a 5-tool server was paying a 40-tool server's price.
///
/// So the budget is spent smallest-first: sort by tool count and admit servers
/// while both the count and the token estimate allow it. A big server is hidden
/// whole - its tool COUNT is what costs, so half of it is no cheaper to reason
/// about - and a small one stays visible with real schemas, which is what puts
/// the grammar back in front of it. Nothing here is a first-party privilege;
/// it is "hide what is actually big", and artifacts benefits only because it is.
pub fn disclose_servers(servers: &[ServerWeight], max_ctx: usize) -> HashSet<String> {
    let mut order: Vec<&ServerWeight> = servers.iter().collect();
    order.sort_by(|a, b| {
        a.tools
            .cmp(&b.tools)
            .then(a.chars.cmp(&b.chars))
            .then_with(|| a.label.cmp(&b.label))
    });
    // ~4 chars/token - the same estimate the call sites used for the old
    // global check, kept so the fork does not move on a rebuild.
    let token_budget = if max_ctx > 0 { max_ctx / 3 } else { usize::MAX };
    let (mut tools, mut tokens) = (0usize, 0usize);
    let mut keep = HashSet::new();
    for s in order {
        let (t, k) = (tools + s.tools, tokens + s.chars / 4);
        // A server that does not fit is SKIPPED, not a stop: sorting is by tool
        // count first, so a later server with the same count but slimmer
        // schemas can still make it.
        if t > SEARCH_DISCLOSURE_THRESHOLD || k > token_budget {
            continue;
        }
        tools = t;
        tokens = k;
        keep.insert(s.label.clone());
    }
    keep
}

/// Strip a client-side namespace prefix from a tool name.
///
/// OpenAI-lineage models emit `functions.mcp_call_tool` (seen live on gpt-5.6)
/// - the `functions.` namespace from the older function-calling
///   dialect, leaking into the name itself. Unprefixed it matches nothing and the
///   call dies. MCP tool names are `[A-Za-z0-9_-]` and ours namespace with `__`,
///   so a dot is never part of a real name and the tail is always the intent.
pub fn strip_client_prefix(name: &str) -> &str {
    name.rsplit_once('.').map_or(name, |(_, tail)| tail)
}

/// What a `mcp_call_tool` wrapper turned out to contain.
pub enum Unwrapped {
    /// The wrapper was well formed: the tool to run and its arguments as text.
    Call { name: String, arguments: String },
    /// The wrapper itself was wrong. The message is written FOR the MODEL - it
    /// says what was missing and how to send it, because the alternative is a
    /// dispatch to the empty tool name and a dead round trip.
    Bad(String),
}

/// Unwrap `mcp_call_tool({name, arguments_json})`.
///
/// This lived in three places (Responses, Anthropic, the manager's cloud loop)
/// and had already drifted: two tolerated an object-valued `arguments_json`,
/// one did not, and all three turned a MISSING `name` into a dispatch of `""`.
/// That is how a model putting the target tool's own arguments at the top
/// level - the single most common mistake, seen live on gpt-5.6 - produced an
/// unexplained error instead of a correction it could act on.
pub fn unwrap_call_tool(arguments: &str) -> Unwrapped {
    let v: serde_json::Value = match serde_json::from_str(arguments) {
        Ok(v) => v,
        Err(e) => {
            return Unwrapped::Bad(format!("mcp_call_tool arguments are not valid JSON: {e}"));
        }
    };
    let name = v
        .get("name")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default()
        .trim();
    if name.is_empty() {
        // Naming the keys that were sent is what makes this correctable: they
        // are almost always the target tool's own arguments.
        let saw = v
            .as_object()
            .map(|o| o.keys().cloned().collect::<Vec<_>>().join(", "))
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "nothing".into());
        return Unwrapped::Bad(format!(
            "mcp_call_tool needs `name` (the tool to run) and `arguments_json` (that \
             tool's own arguments). This call had neither - only: {saw}. Call \
             mcp_call_tool again shaped as \
             {{\"name\": \"<tool>\", \"arguments_json\": \"{{...}}\"}}, with those \
             keys INSIDE arguments_json. Use mcp_search_tools if you need the name."
        ));
    }
    // `arguments_json` is spec'd as a JSON string; an object says the same
    // thing and refusing it would cost a round trip to no one's benefit.
    let arguments = match v.get("arguments_json") {
        Some(serde_json::Value::String(s)) => s.clone(),
        Some(other) => other.to_string(),
        None => "{}".to_string(),
    };
    Unwrapped::Call {
        name: name.to_string(),
        arguments,
    }
}

/// One searchable tool: the namespaced name the model invokes (`label__tool`),
/// its description, and its JSON-Schema (returned inline in search results).
#[derive(Clone)]
pub struct CatalogTool {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
}

/// The result of checking one call's arguments against its tool's schema.
pub enum ArgCheck {
    /// Usable. The text is the arguments, possibly REPAIRED (see `coerce`).
    Ok(String),
    /// Not usable. The message is written for the model and names the field.
    Bad(String),
}

/// How much of a property's own description to quote back. Long enough to
/// disambiguate a field, short enough that a ten-field schema stays readable.
const DESC_CAP: usize = 120;

/// The `type` keyword, which JSON Schema allows to be a string or a list.
fn type_names(spec: &Value) -> Vec<&str> {
    match spec.get("type") {
        Some(Value::String(s)) => vec![s.as_str()],
        Some(Value::Array(a)) => a.iter().filter_map(Value::as_str).collect(),
        _ => Vec::new(),
    }
}

fn matches_type(v: &Value, ty: &str) -> bool {
    match ty {
        "string" => v.is_string(),
        "number" => v.is_number(),
        "integer" => v.is_i64() || v.is_u64() || v.as_f64().is_some_and(|f| f.fract() == 0.0),
        "boolean" => v.is_boolean(),
        "array" => v.is_array(),
        "object" => v.is_object(),
        "null" => v.is_null(),
        // A keyword we do not know is not ours to enforce - a validator that
        // rejects what it merely fails to understand is worse than none.
        _ => true,
    }
}

/// Repair a value that is the right THING in the wrong JSON type.
///
/// Models emit numbers, booleans, arrays and objects as strings constantly, and
/// the server would have coerced most of it anyway. Refusing costs a whole
/// round trip to say something we can just fix - so fix it, and spend the
/// refusals on what is genuinely missing.
fn coerce(v: &Value, types: &[&str]) -> Option<Value> {
    // The one non-string direction worth taking: a scalar where a string is
    // wanted. 5 -> "5" cannot change what the tool does.
    if types.contains(&"string") && (v.is_number() || v.is_boolean()) {
        return Some(json!(v.to_string()));
    }
    let t = v.as_str()?.trim();
    for ty in types {
        match *ty {
            "integer" => {
                if let Ok(i) = t.parse::<i64>() {
                    return Some(json!(i));
                }
            }
            "number" => {
                if let Ok(f) = t.parse::<f64>()
                    && let Some(n) = serde_json::Number::from_f64(f)
                {
                    return Some(Value::Number(n));
                }
            }
            "boolean" => match t {
                "true" => return Some(json!(true)),
                "false" => return Some(json!(false)),
                _ => {}
            },
            // A JSON array/object that arrived quoted - one parse away from right.
            "array" | "object" => {
                if let Ok(p) = serde_json::from_str::<Value>(t)
                    && matches_type(&p, ty)
                {
                    return Some(p);
                }
            }
            _ => {}
        }
    }
    None
}

/// Whitespace-collapsed, capped copy of a schema description.
fn one_line(s: &str) -> String {
    let flat = s.split_whitespace().collect::<Vec<_>>().join(" ");
    if flat.chars().count() > DESC_CAP {
        flat.chars().take(DESC_CAP).collect::<String>() + "..."
    } else {
        flat
    }
}

/// "`content` (string - the document body)" - what the model needs to fill a
/// field in without going back to the schema.
fn describe(name: &str, spec: Option<&Value>) -> String {
    let mut s = format!("`{name}`");
    let Some(spec) = spec else { return s };
    let ty = type_names(spec).join(" or ");
    let desc = spec
        .get("description")
        .and_then(Value::as_str)
        .map(one_line)
        .unwrap_or_default();
    match (ty.is_empty(), desc.is_empty()) {
        (true, true) => {}
        (false, true) => s.push_str(&format!(" ({ty})")),
        (true, false) => s.push_str(&format!(" ({desc})")),
        (false, false) => s.push_str(&format!(" ({ty} - {desc})")),
    }
    s
}

fn kind_of(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "a boolean",
        Value::Number(_) => "a number",
        Value::String(_) => "a string",
        Value::Array(_) => "an array",
        Value::Object(_) => "an object",
    }
}

/// Check (and where possible repair) a call's arguments against its schema,
/// before anything is dispatched.
///
/// Local models get this for free from the grammar - but only on tools
/// whose schema was declared, and only locally: a cloud provider constrains
/// nothing, and anything routed through the `mcp_call_tool` envelope hides its
/// arguments in a string the grammar never sees. So the check runs on every
/// call in every dialect; where the grammar already did its job this is a
/// no-op, and where it could not, this is the whole guarantee.
///
/// It enforces only what a schema says unambiguously - required, type, enum,
/// and unknown keys when `additionalProperties: false`. Unknown keywords are
/// left alone: the server is a better judge of its own schema than we are, and
/// a false refusal here breaks a working tool for no gain.
pub fn check_arguments(tool: &str, schema: &Value, arguments: &str) -> ArgCheck {
    let text = arguments.trim();
    let mut v: Value = if text.is_empty() {
        json!({})
    } else {
        match serde_json::from_str(text) {
            Ok(v) => v,
            Err(e) => {
                return ArgCheck::Bad(format!(
                    "{tool} was NOT called: its arguments are not valid JSON ({e}). Send them as \
                     one JSON object and call it again."
                ));
            }
        }
    };
    // Double-encoded: the whole object arrived as a JSON string. Through the
    // mcp_call_tool envelope that is a string inside a string, which models get
    // wrong often enough that refusing it would be pedantry.
    if let Some(s) = v.as_str()
        && let Ok(inner @ Value::Object(_)) = serde_json::from_str::<Value>(s)
    {
        v = inner;
    }
    // Nothing to check against (no properties, or not an object schema at all).
    let Some(props) = schema.get("properties").and_then(Value::as_object) else {
        return ArgCheck::Ok(v.to_string());
    };
    let required: Vec<&str> = schema
        .get("required")
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    let Some(obj) = v.as_object() else {
        return ArgCheck::Bad(format!(
            "{tool} was NOT called: its arguments must be a JSON object, got {}. It takes: {}.",
            kind_of(&v),
            props.keys().cloned().collect::<Vec<_>>().join(", ")
        ));
    };

    let mut out = serde_json::Map::new();
    let mut problems: Vec<String> = Vec::new();
    let mut unknown: Vec<String> = Vec::new();
    for (k, val) in obj {
        let Some(spec) = props.get(k) else {
            unknown.push(k.clone());
            out.insert(k.clone(), val.clone());
            continue;
        };
        let types = type_names(spec);
        // An explicit null for an optional field means "not set". Dropping it
        // is what the server would infer anyway and keeps a nuisance type
        // error out of the model's way - Unless the schema accepts null, where
        // an explicit null can mean "clear this" and is ours to pass through.
        if val.is_null() && !required.contains(&k.as_str()) && !types.contains(&"null") {
            continue;
        }
        let val = if types.is_empty() || types.iter().any(|t| matches_type(val, t)) {
            val.clone()
        } else if let Some(fixed) = coerce(val, &types) {
            fixed
        } else {
            problems.push(format!(
                "`{k}` must be {} (got {})",
                types.join(" or "),
                kind_of(val)
            ));
            val.clone()
        };
        // The server enumerated what it accepts, so an unlisted value is a
        // guaranteed failure - quote the list instead of letting it try.
        if let Some(allowed) = spec.get("enum").and_then(Value::as_array)
            && !allowed.contains(&val)
        {
            problems.push(format!(
                "`{k}` must be one of {} (got {val})",
                allowed
                    .iter()
                    .map(Value::to_string)
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
        out.insert(k.clone(), val);
    }
    let missing: Vec<&str> = required
        .iter()
        .copied()
        .filter(|r| !out.contains_key(*r))
        .collect();
    // Only when the server itself said so - plenty of schemas omit the keyword
    // and tolerate extras, and rejecting those would be us inventing a rule.
    // And only when the root is a plain object schema: under composition the
    // accepted keys can come from a branch we are not reading, so "this tool
    // takes: ..." would be a confident lie that breaks a working call.
    let composed = ["allOf", "anyOf", "oneOf", "$ref", "not"]
        .iter()
        .any(|k| schema.get(*k).is_some());
    let strict = !composed && schema.get("additionalProperties") == Some(&Value::Bool(false));

    if missing.is_empty() && problems.is_empty() && (!strict || unknown.is_empty()) {
        return ArgCheck::Ok(Value::Object(out).to_string());
    }
    let mut lines: Vec<String> = Vec::new();
    if !missing.is_empty() {
        lines.push(format!(
            "missing required {}",
            missing
                .iter()
                .map(|m| describe(m, props.get(*m)))
                .collect::<Vec<_>>()
                .join("; ")
        ));
    }
    lines.extend(problems);
    if strict && !unknown.is_empty() {
        lines.push(format!(
            "`{}` {} not accepted (this tool takes: {})",
            unknown.join("`, `"),
            if unknown.len() == 1 { "is" } else { "are" },
            props.keys().cloned().collect::<Vec<_>>().join(", ")
        ));
    }
    ArgCheck::Bad(format!(
        "{tool} was NOT called - the arguments did not match its schema: {}. Nothing ran and \
         nothing changed; call {tool} again with that corrected.",
        lines.join("; ")
    ))
}

/// One model tool call resolved to something dispatchable - or to a refusal.
pub enum Resolved {
    Call {
        name: String,
        arguments: String,
    },
    /// Do not dispatch. `message` goes back as this call's (failed) result.
    Refuse {
        name: String,
        message: String,
    },
}

/// The one seam every dialect goes through: unwrap the `mcp_call_tool` envelope
/// when that is what was called, then validate the arguments against the target
/// tool's schema before anything runs.
///
/// One function deliberately. The unwrapping alone lived in three copies and had
/// already drifted; adding validation next to each of them would have been the
/// same mistake with more surface. An unknown `name` still falls through to the
/// caller's own unknown-tool path - that message names the search tool, which
/// is the right advice for a name we cannot place.
pub fn resolve_call(name: &str, arguments: &str, catalog: &[CatalogTool]) -> Resolved {
    let (name, arguments) = if name == CALL_TOOL {
        match unwrap_call_tool(arguments) {
            Unwrapped::Bad(message) => {
                return Resolved::Refuse {
                    name: CALL_TOOL.to_string(),
                    message,
                };
            }
            Unwrapped::Call { name, arguments } => (name, arguments),
        }
    } else {
        (name.to_string(), arguments.to_string())
    };
    let Some(t) = catalog.iter().find(|t| t.name == name) else {
        return Resolved::Call { name, arguments };
    };
    match check_arguments(&name, &t.input_schema, &arguments) {
        ArgCheck::Ok(arguments) => Resolved::Call { name, arguments },
        ArgCheck::Bad(message) => Resolved::Refuse { name, message },
    }
}

const SEARCH_DESC: &str = "Search the available tool catalog for a capability that is not in your current tool \
list. Returns matching tools with their name, description, and input schema, plus `all_tool_names` - the complete \
catalog - so if the ranked results miss what you need, pick the right name from that list and search for it by its \
exact name to get its schema. After finding the right tool, invoke it with mcp_call_tool. Whenever the user asks for \
something you don't have a listed tool for, search first - never assume a capability is unavailable without searching. \
A result with `count` 0 carries a `no_match` verdict: the catalog holds nothing for that query, so answer without a \
tool rather than searching again.";

const CALL_DESC: &str = "Invoke a tool discovered with mcp_search_tools. Pass the tool's exact `name` and its \
arguments in `arguments_json` as a JSON object encoded as a string.";

fn search_params() -> Value {
    json!({
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Keywords for the capability you need, e.g. 'search company by name' or 'LEI lookup'."
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of tools to return (default 5, max 25)."
            }
        },
        "required": ["query"],
        "additionalProperties": false
    })
}

fn call_params() -> Value {
    json!({
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "The exact tool name returned by mcp_search_tools (e.g. 'tic__search_lei')."
            },
            "arguments_json": {
                "type": "string",
                "description": "The tool's arguments as a JSON object encoded as a string, e.g. '{\"lei\": \"...\"}'."
            }
        },
        "required": ["name", "arguments_json"],
        "additionalProperties": false
    })
}

/// `mcp_search_tools` in OpenAI Chat/Responses function shape.
pub fn search_tool_def() -> Value {
    json!({ "type": "function", "function": { "name": SEARCH_TOOL, "description": SEARCH_DESC, "parameters": search_params() } })
}

/// `mcp_call_tool` in OpenAI Chat/Responses function shape.
pub fn call_tool_def() -> Value {
    json!({ "type": "function", "function": { "name": CALL_TOOL, "description": CALL_DESC, "parameters": call_params() } })
}

/// `mcp_search_tools` in Anthropic `{name, description, input_schema}` shape.
pub fn search_tool_def_anthropic() -> Value {
    json!({ "name": SEARCH_TOOL, "description": SEARCH_DESC, "input_schema": search_params() })
}

/// `mcp_call_tool` in Anthropic `{name, description, input_schema}` shape.
pub fn call_tool_def_anthropic() -> Value {
    json!({ "name": CALL_TOOL, "description": CALL_DESC, "input_schema": call_params() })
}

/// Lowercase alphanumeric terms, splitting snake_case and camelCase so both
/// `search_lei` and `searchLei` yield `["search","lei"]`.
fn tokenize(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut prev_alnum_lower = false;
    for ch in s.chars() {
        if ch.is_alphanumeric() {
            // camelCase boundary: a lower/digit run followed by an uppercase.
            if ch.is_uppercase() && prev_alnum_lower && !cur.is_empty() {
                out.push(std::mem::take(&mut cur));
            }
            for lc in ch.to_lowercase() {
                cur.push(lc);
            }
            prev_alnum_lower = ch.is_lowercase() || ch.is_numeric();
        } else {
            if !cur.is_empty() {
                out.push(std::mem::take(&mut cur));
            }
            prev_alnum_lower = false;
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// Tiny suffix stripper - just the inflections that actually cost us matches
/// against real MCP catalogs (companies/company, registered/register,
/// vehicles/vehicle, listings/listing). Not a Porter stemmer: tool routing
/// needs predictable, not linguistic - both query and docs go through the
/// same rules, so even an ugly stem unifies. Observed live:
/// "company lookup Sweden business registry" missed tic__get_company_by_name
/// partly because nothing unified registered/registry-class inflections.
fn stem(t: &str) -> String {
    if let Some(base) = t.strip_suffix("ies").filter(|b| b.len() >= 3) {
        return format!("{base}y");
    }
    // "es" only after a sibilant (searches/boxes) - otherwise the plain "s"
    // rule below keeps the e (vehicles -> vehicle, not vehicl)
    if let Some(base) = t.strip_suffix("es")
        && base.len() >= 3
        && (base.ends_with('s')
            || base.ends_with('x')
            || base.ends_with('z')
            || base.ends_with("ch")
            || base.ends_with("sh"))
    {
        return base.to_string();
    }
    for suf in ["ing", "ed"] {
        if let Some(base) = t.strip_suffix(suf)
            && base.len() >= 4
        {
            return base.to_string();
        }
    }
    // plain plural; never mangle an "ss" word (business stays business)
    if let Some(base) = t.strip_suffix('s')
        && base.len() >= 4
        && !base.ends_with('s')
    {
        return base.to_string();
    }
    t.to_string()
}

/// tokenize + stem: the term stream both documents and queries index by.
fn terms(s: &str) -> Vec<String> {
    tokenize(s).iter().map(|t| stem(t)).collect()
}

/// Indexable text pulled from a tool's JSON-Schema: property names + their
/// descriptions (so "invoice number" can find a tool whose param is `invoice_no`).
fn schema_terms(schema: &Value) -> String {
    let mut s = String::new();
    if let Some(props) = schema.get("properties").and_then(Value::as_object) {
        for (k, v) in props {
            s.push_str(k);
            s.push(' ');
            if let Some(d) = v.get("description").and_then(Value::as_str) {
                s.push_str(d);
                s.push(' ');
            }
        }
    }
    s
}

/// The weighted token bag for one tool: name ×3 (label + tool name carry the
/// most signal), then description and parameter text ×1.
fn doc_tokens(t: &CatalogTool) -> Vec<String> {
    let mut out = Vec::new();
    let name_toks = terms(&t.name);
    for _ in 0..3 {
        out.extend(name_toks.iter().cloned());
    }
    out.extend(terms(&t.description));
    out.extend(terms(&schema_terms(&t.input_schema)));
    out
}

/// BM25-rank `catalog` against `query`, best first, at most `limit` hits with a
/// positive score. Okapi BM25 (k1=1.5, b=0.75) over the weighted token bags.
pub fn search<'a>(catalog: &'a [CatalogTool], query: &str, limit: usize) -> Vec<&'a CatalogTool> {
    if catalog.is_empty() {
        return Vec::new();
    }
    let q_terms = terms(query);
    if q_terms.is_empty() {
        return Vec::new();
    }
    let docs: Vec<Vec<String>> = catalog.iter().map(doc_tokens).collect();
    let n = docs.len() as f64;
    let total_len: usize = docs.iter().map(Vec::len).sum();
    let avgdl = (total_len as f64 / n).max(1.0);

    // document frequency per term
    let mut df: HashMap<&str, usize> = HashMap::new();
    for d in &docs {
        let uniq: HashSet<&str> = d.iter().map(String::as_str).collect();
        for term in uniq {
            *df.entry(term).or_insert(0) += 1;
        }
    }

    const K1: f64 = 1.5;
    const B: f64 = 0.75;
    let mut scored: Vec<(f64, usize)> = Vec::new();
    for (i, d) in docs.iter().enumerate() {
        let dl = d.len() as f64;
        let mut tf: HashMap<&str, usize> = HashMap::new();
        for term in d {
            *tf.entry(term.as_str()).or_insert(0) += 1;
        }
        let mut score = 0.0;
        for qt in &q_terms {
            let f = *tf.get(qt.as_str()).unwrap_or(&0) as f64;
            if f == 0.0 {
                continue;
            }
            let nq = *df.get(qt.as_str()).unwrap_or(&0) as f64;
            let idf = ((n - nq + 0.5) / (nq + 0.5) + 1.0).ln();
            score += idf * (f * (K1 + 1.0)) / (f + K1 * (1.0 - B + B * dl / avgdl));
        }
        if score > 0.0 {
            scored.push((score, i));
        }
    }
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut hits: Vec<usize> = scored.into_iter().take(limit).map(|(_, i)| i).collect();
    // BM25 is a LEXICAL match: a term the catalog does not literally contain
    // scores zero however close it is. That is fine for "company lookup" and
    // useless for a typo, a near-miss name half-remembered from an earlier
    // round, or a word the tool spells differently ("bankrupcy", "org number",
    // "vat_no"). An empty result costs a whole round and teaches the model
    // nothing - and a round is the scarcest thing a tool turn has.
    //
    // So top up with character-trigram similarity, which degrades gracefully
    // where token matching falls off a cliff. Deliberately a FALLBACK and not a
    // blended score: where BM25 has an opinion it is the better one, and mixing
    // would let a fuzzy near-miss outrank an exact term match.
    if hits.len() < limit {
        let taken: HashSet<usize> = hits.iter().copied().collect();
        let mut fuzzy: Vec<(f64, usize)> = catalog
            .iter()
            .enumerate()
            .filter(|(i, _)| !taken.contains(i))
            .filter_map(|(i, t)| {
                let s = trigram_similarity(query, &format!("{} {}", t.name, t.description));
                (s >= FUZZY_FLOOR).then_some((s, i))
            })
            .collect();
        fuzzy.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        hits.extend(fuzzy.into_iter().take(limit - hits.len()).map(|(_, i)| i));
    }
    hits.into_iter().map(|i| &catalog[i]).collect()
}

/// How alike a fuzzy hit must be before it is worth a line in the results.
///
/// Tuned to admit a misspelling or a synonym-adjacent name and reject the rest:
/// too low and every search returns the whole catalog ranked by noise, which is
/// worse than the empty result it replaces. Overlap coefficient, so a short
/// query against a long description is not punished for the length difference.
const FUZZY_FLOOR: f64 = 0.34;

/// Overlap coefficient over character trigrams: |A∩B| / min(|A|,|B|).
///
/// Not Jaccard, deliberately - a three-word query against a forty-word tool
/// description has a tiny union, so Jaccard would score every real match near
/// zero. Overlap asks the question we actually mean: how much of the SHORTER
/// side is present in the longer one.
fn trigram_similarity(a: &str, b: &str) -> f64 {
    let (ta, tb) = (trigrams(a), trigrams(b));
    if ta.is_empty() || tb.is_empty() {
        return 0.0;
    }
    let shared = ta.intersection(&tb).count() as f64;
    shared / (ta.len().min(tb.len()) as f64)
}

/// Character trigrams of the lowercased alphanumeric stream, word boundaries
/// collapsed to single spaces so `get_company` and `getCompany` and
/// `get company` produce the same set.
fn trigrams(s: &str) -> HashSet<[char; 3]> {
    let flat: Vec<char> = tokenize(s).join(" ").chars().collect();
    flat.windows(3).map(|w| [w[0], w[1], w[2]]).collect()
}

/// The JSON payload a `mcp_search_tools` call returns to the model: each hit's
/// name, description, and full input schema (so it can call one straight away),
/// plus `all_tool_names` - the complete catalog index. Names are cheap
/// (~200 tokens for a 29-tool server) and rescue every ranking miss: the
/// model spots the right name in the same round instead of guessing new
/// keywords (observed live: two blind searches for what
/// tic__get_company_by_name does, because BM25 kept ranking it out).
pub fn search_result(query: &str, hits: &[&CatalogTool], catalog: &[CatalogTool]) -> String {
    let results: Vec<Value> = hits
        .iter()
        .map(|t| {
            json!({
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            })
        })
        .collect();
    // safety cap for pathological catalogs - the index stops being cheap
    // past a few hundred names, and truncation is announced, never silent
    const INDEX_CAP: usize = 300;
    let mut names: Vec<&str> = catalog.iter().map(|t| t.name.as_str()).collect();
    names.sort_unstable();
    let mut index: Vec<Value> = names.iter().take(INDEX_CAP).map(|n| json!(n)).collect();
    if names.len() > INDEX_CAP {
        index.push(json!(format!(
            "...and {} more (search to find them)",
            names.len() - INDEX_CAP
        )));
    }
    // Zero HITS needs A VERDICT, not just an empty array beside a name list.
    // a 9B asked "what is this?" about a photo of a church
    // in Arezzo, searched "Arezzo Italy church piazza square GPS 43.466012",
    // and got count 0 next to 34 Swedish company-registry and property tool
    // names. Nothing there could ever answer it, but an empty result beside a
    // catalog reads as "keep looking" - the model has no way to tell "ranked
    // badly" from "does not exist", and those want opposite next moves.
    //
    // The names STAY even at zero: BM25 scoring nothing is not proof that
    // nothing fits, and the whole reason the index exists is that BM25 ranked
    // out the right tool on a live turn. What changes is that the payload
    // now says out loud which kind of nothing this is.
    let verdict = (results.is_empty()).then_some(
        "No tool in this catalog matches that query. The names below are the COMPLETE list of \
         what exists here - if none of them does the job, the capability is not available: say \
         so, or answer from your own knowledge. Do not search again for the same thing.",
    );
    json!({ "query": query, "count": results.len(), "results": results,
            "no_match": verdict, "all_tool_names": index })
    .to_string()
}

#[cfg(test)]
mod tests {
    /// Injected prompt text is prose the model reads; a run of spaces from a
    /// mangled line continuation is invisible in review and shows up in the
    /// user-facing panel. Guard both constants (spotted it
    /// in the Studio before any of us read the literal).
    #[test]
    fn injected_prompt_text_has_no_stray_whitespace() {
        for (name, text) in [
            ("SEARCH_MODE_INSTRUCTIONS", super::SEARCH_MODE_INSTRUCTIONS),
            (
                "SEARCH_AVAILABLE_INSTRUCTIONS",
                super::SEARCH_AVAILABLE_INSTRUCTIONS,
            ),
            ("SEARCH_PARTIAL_TEMPLATE", super::SEARCH_PARTIAL_TEMPLATE),
        ] {
            assert!(!text.contains("  "), "{name} has a double space: {text:?}");
            assert!(
                !text.contains(char::from(10)),
                "{name} has a newline: {text:?}"
            );
            assert_eq!(text.trim(), text, "{name} has edge whitespace");
        }
    }

    use super::*;

    fn tool(name: &str, desc: &str) -> CatalogTool {
        CatalogTool {
            name: name.into(),
            description: desc.into(),
            input_schema: json!({}),
        }
    }

    #[test]
    fn tokenize_splits_snake_and_camel() {
        assert_eq!(tokenize("search_lei"), vec!["search", "lei"]);
        assert_eq!(
            tokenize("getCompanyByName"),
            vec!["get", "company", "by", "name"]
        );
        assert_eq!(tokenize("tic__search_lei"), vec!["tic", "search", "lei"]);
    }

    #[test]
    fn bm25_ranks_the_relevant_tool_first() {
        let catalog = vec![
            tool(
                "tic__search_lei",
                "Search the global LEI (Legal Entity Identifier) register worldwide",
            ),
            tool("tic__get_credit_score", "Get credit score by companyId"),
            tool(
                "tic__get_company_vehicles",
                "Get vehicles owned or used by companyId",
            ),
        ];
        let hits = search(&catalog, "LEI legal entity identifier lookup", 5);
        assert!(!hits.is_empty());
        assert_eq!(hits[0].name, "tic__search_lei");
    }

    #[test]
    fn search_finds_by_parameter_and_returns_schema_inline() {
        let catalog = vec![CatalogTool {
            name: "vat__lookup".into(),
            description: "Company tax records".into(),
            input_schema: json!({"properties": {"vat_number": {"description": "the VAT registration id"}}}),
        }];
        let hits = search(&catalog, "vat number", 5);
        assert_eq!(hits.len(), 1);
        let payload = search_result("vat number", &hits, &catalog);
        assert!(payload.contains("input_schema") && payload.contains("vat_number"));
    }

    /// The fuzzy floor has to actually hold: a query about nothing in the
    /// catalog must still return nothing, or the fallback has just replaced an
    /// honest empty result with a page of noise.
    #[test]
    fn no_match_returns_empty() {
        let catalog = vec![tool("fs__read_file", "read a file")];
        assert!(search(&catalog, "quantum chromodynamics", 5).is_empty());
    }

    /// The reason this is called FUZZY search. BM25 scores a misspelling zero
    /// however close it is, and a zero-result search costs a whole round of a
    /// budget that has ~16 of them. Trigram overlap degrades where token
    /// matching falls off a cliff.
    #[test]
    fn a_misspelled_query_still_finds_its_tool() {
        let catalog = vec![
            tool(
                "tic__get_company_bankruptcies",
                "Bankruptcy filings for a company",
            ),
            tool("fs__read_file", "read a file from disk"),
        ];
        // "bankrupcy" shares no STEMMED term with "bankruptcies"/"bankruptcy",
        // so BM25 alone returns nothing here.
        let hits = search(&catalog, "bankrupcy", 5);
        assert_eq!(hits.len(), 1, "the fallback should rescue exactly one tool");
        assert_eq!(hits[0].name, "tic__get_company_bankruptcies");
    }

    /// The fallback fills the tail, it never reorders the head: where BM25 has
    /// an opinion it is the better one, and a blended score would let a fuzzy
    /// near-miss outrank an exact term match.
    #[test]
    fn a_lexical_hit_outranks_a_fuzzy_one() {
        let catalog = vec![
            tool(
                "tic__get_company_bankrupt_status",
                "is the company bankrupt",
            ),
            tool("tic__get_company_bankruptcies", "bankruptcy filings"),
        ];
        let hits = search(&catalog, "bankrupt", 5);
        assert_eq!(
            hits[0].name, "tic__get_company_bankrupt_status",
            "exact term wins the top"
        );
        assert_eq!(hits.len(), 2, "...and the near-miss still gets a line");
    }

    #[test]
    fn stemming_unifies_real_inflections() {
        // the live miss class: query and doc inflect the same word differently
        assert_eq!(stem("companies"), "company");
        assert_eq!(stem("registered"), "register");
        assert_eq!(stem("vehicles"), "vehicle");
        assert_eq!(stem("listings"), "listing");
        // short/awkward words stay put instead of degrading
        assert_eq!(stem("lei"), "lei");
        assert_eq!(stem("name"), "name");
        let catalog = vec![
            tool(
                "tic__get_company_vehicles",
                "Get vehicles owned or used by companyId",
            ),
            tool("tic__get_credit_score", "Get credit score by companyId"),
        ];
        // singular query finds the plural-named tool
        let hits = search(&catalog, "company vehicle", 5);
        assert_eq!(hits[0].name, "tic__get_company_vehicles");
    }

    #[test]
    fn every_result_carries_the_full_name_index() {
        // ranking missed the right tool (live: "company lookup
        // Sweden business registry" never surfaced get_company_by_name) -
        // the index puts every name in front of the model anyway
        let catalog = vec![
            tool("tic__get_company_by_name", "Get a company by name"),
            tool(
                "tic__get_company_business_mortgages",
                "Registered business mortgages",
            ),
        ];
        let hits = search(&catalog, "mortgages", 5);
        let payload = search_result("mortgages", &hits, &catalog);
        assert!(payload.contains("all_tool_names"));
        assert!(
            payload.contains("tic__get_company_by_name"),
            "unranked tools are still indexed"
        );
        // A hit means there is nothing to rule out.
        assert!(payload.contains("\"no_match\":null"), "{payload}");
    }

    /// Zero hits and a name list, with nothing to tell them apart, reads as
    /// "keep looking". : a 9B asked what a photo of a church
    /// in Arezzo showed, searched for it, and got count 0 beside 34 Swedish
    /// company-registry tool names. The names stay - BM25 scoring nothing is
    /// not proof that nothing fits - but the verdict has to be said.
    #[test]
    fn nothing_matched_says_so_instead_of_just_listing_names() {
        let catalog = vec![
            tool("tic__get_company_by_name", "Get a company by name"),
            tool("tic__get_vehicle_by_vin", "Look up a vehicle by VIN"),
        ];
        let q = "Arezzo Italy church piazza square";
        let hits = search(&catalog, q, 5);
        assert!(
            hits.is_empty(),
            "the fixture must genuinely miss, got {}",
            hits.len()
        );
        let payload = search_result(q, &hits, &catalog);
        assert!(
            payload.contains("No tool in this catalog matches"),
            "{payload}"
        );
        assert!(payload.contains("Do not search again"), "{payload}");
        // ...and the catalog is still there to rescue a ranking miss.
        assert!(payload.contains("tic__get_company_by_name"), "{payload}");
    }

    #[test]
    fn a_client_namespace_prefix_is_not_part_of_the_name() {
        assert_eq!(strip_client_prefix("functions.mcp_call_tool"), CALL_TOOL);
        assert_eq!(strip_client_prefix("mcp_call_tool"), CALL_TOOL);
        assert_eq!(
            strip_client_prefix("artifacts__artifact_create"),
            "artifacts__artifact_create"
        );
    }

    #[test]
    fn a_well_formed_wrapper_unwraps_either_spelling() {
        let string_form =
            r#"{"name":"artifacts__artifact_create","arguments_json":"{\"kind\":\"html\"}"}"#;
        let Unwrapped::Call { name, arguments } = unwrap_call_tool(string_form) else {
            panic!("string form must unwrap")
        };
        assert_eq!(name, "artifacts__artifact_create");
        assert_eq!(arguments, r#"{"kind":"html"}"#);

        // An object where the spec wants a string says the same thing.
        let object_form = r#"{"name":"t","arguments_json":{"kind":"html"}}"#;
        let Unwrapped::Call { arguments, .. } = unwrap_call_tool(object_form) else {
            panic!("object form must unwrap")
        };
        assert_eq!(arguments, r#"{"kind":"html"}"#);
    }

    /// The live failure: the target tool's own arguments at the top level, no
    /// `name` anywhere. It must come back as advice, not as a dispatch of "".
    #[test]
    fn a_wrapper_with_no_name_names_what_it_did_get() {
        let flat = r#"{"kind":"html","title":"Hero","content":"<h1>x</h1>"}"#;
        let Unwrapped::Bad(msg) = unwrap_call_tool(flat) else {
            panic!("a missing name must not dispatch")
        };
        // Every key it did send is named (serde orders them, so do not assert
        // on the order - only that nothing is left out).
        for key in ["kind", "title", "content"] {
            assert!(msg.contains(key), "{key} missing from: {msg}");
        }
        assert!(msg.contains("arguments_json"), "{msg}");

        let Unwrapped::Bad(msg) = unwrap_call_tool("not json") else {
            panic!("must refuse")
        };
        assert!(msg.contains("not valid JSON"), "{msg}");
    }

    fn weight(label: &str, tools: usize) -> ServerWeight {
        ServerWeight {
            label: label.into(),
            tools,
            chars: tools * 200,
        }
    }

    /// The live shape: artifacts(5) next to tic(~40). All-or-
    /// nothing hid both, so a five-tool server's calls travelled through the
    /// wrapper for no reason of its own.
    #[test]
    fn a_small_server_survives_a_big_one() {
        let keep = disclose_servers(&[weight("tic", 40), weight("artifacts", 5)], 0);
        assert!(
            keep.contains("artifacts"),
            "the small server keeps its schemas"
        );
        assert!(
            !keep.contains("tic"),
            "the big one is what goes behind search"
        );
    }

    #[test]
    fn disclosure_matches_the_old_global_switch_on_one_server() {
        // under the threshold: everything, exactly as before
        assert_eq!(disclose_servers(&[weight("solo", 10)], 0).len(), 1);
        // over it: nothing - a big server is hidden whole, not trimmed to fit
        assert!(disclose_servers(&[weight("solo", 20)], 0).is_empty());
    }

    #[test]
    fn a_tight_context_shrinks_disclosure_before_the_count_does() {
        // 8 tools is under the count threshold, but 8*200 chars ~= 400 tokens
        // and a 600-token context affords 200.
        let one = [ServerWeight {
            label: "big".into(),
            tools: 8,
            chars: 1600,
        }];
        assert!(disclose_servers(&one, 600).is_empty());
        assert_eq!(disclose_servers(&one, 100_000).len(), 1);
    }

    #[test]
    fn the_partial_notice_names_what_is_hidden() {
        let m = partial_mode_instructions(&["tic".into(), "github".into()], 47);
        assert!(m.contains("tic, github") && m.contains("47"), "{m}");
        assert!(!m.contains("  ") && !m.contains(char::from(10)), "{m}");
    }

    fn create_schema() -> Value {
        json!({
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": ["html", "csv"]},
                "title": {"type": "string"},
                "content": {"type": "string", "description": "The full document text."},
                "seq": {"type": "integer"}
            },
            "required": ["kind", "title", "content"],
            "additionalProperties": false
        })
    }

    /// The 75-second failure: artifact_create called with everything but the
    /// content. It must never reach the server, and the refusal must name the
    /// field so the retry is one round, not four.
    #[test]
    fn a_missing_required_field_is_refused_by_name() {
        let args = r#"{"kind":"html","title":"Hero"}"#;
        let ArgCheck::Bad(m) = check_arguments("artifact_create", &create_schema(), args) else {
            panic!("a missing required field must not dispatch")
        };
        assert!(m.contains("`content`"), "{m}");
        assert!(
            m.contains("The full document text."),
            "the description guides the retry: {m}"
        );
        assert!(
            m.contains("NOT called"),
            "the model must know nothing ran: {m}"
        );
    }

    #[test]
    fn wrong_types_are_repaired_rather_than_refused() {
        // every one of these is a real model habit: numbers and objects quoted
        let args = r#"{"kind":"html","title":"Hero","content":"<h1>x</h1>","seq":"3"}"#;
        let ArgCheck::Ok(fixed) = check_arguments("t", &create_schema(), args) else {
            panic!("a coercible type must not refuse")
        };
        let v: Value = serde_json::from_str(&fixed).expect("repaired arguments are still JSON");
        assert_eq!(v["seq"], json!(3), "the string became the integer it meant");

        // the whole object double-encoded through the envelope
        let doubled = r#""{\"kind\":\"html\",\"title\":\"H\",\"content\":\"x\"}""#;
        assert!(matches!(
            check_arguments("t", &create_schema(), doubled),
            ArgCheck::Ok(_)
        ));
    }

    #[test]
    fn an_enum_and_an_unknown_key_are_both_named() {
        let args = r#"{"kind":"pdf","title":"H","content":"x","colour":"red"}"#;
        let ArgCheck::Bad(m) = check_arguments("t", &create_schema(), args) else {
            panic!("must refuse")
        };
        assert!(
            m.contains("\"html\"") && m.contains("\"csv\""),
            "the accepted set: {m}"
        );
        assert!(
            m.contains("`colour`"),
            "additionalProperties:false was declared: {m}"
        );
    }

    /// A validator that rejects what it merely fails to understand is worse
    /// than none - these all have to pass straight through.
    #[test]
    fn schemas_we_cannot_judge_are_left_alone() {
        for schema in [
            json!({}),
            json!({"$ref": "#/definitions/Thing"}),
            json!({"type": "object"}),
            json!({"anyOf": [{"type": "object"}]}),
        ] {
            let r = check_arguments("t", &schema, r#"{"whatever":1}"#);
            assert!(
                matches!(r, ArgCheck::Ok(_)),
                "schema {schema} must not refuse"
            );
        }
        // extras are fine unless the server said otherwise
        let lax = json!({"type":"object","properties":{"a":{"type":"string"}}});
        assert!(matches!(
            check_arguments("t", &lax, r#"{"a":"x","b":2}"#),
            ArgCheck::Ok(_)
        ));

        // ...and even when it did, a COMPOSED root can accept keys from a
        // branch we are not reading, so "this tool takes: a" would be a lie.
        let composed = json!({"type":"object","properties":{"a":{"type":"string"}},
            "additionalProperties": false, "allOf": [{"properties":{"b":{"type":"integer"}}}]});
        assert!(matches!(
            check_arguments("t", &composed, r#"{"a":"x","b":2}"#),
            ArgCheck::Ok(_)
        ));
    }

    #[test]
    fn an_explicit_null_is_dropped_only_where_it_cannot_be_meant() {
        let schema = json!({"type":"object","properties":{
            "note": {"type": "string"},
            "clear": {"type": ["string", "null"]}
        }});
        let ArgCheck::Ok(fixed) = check_arguments("t", &schema, r#"{"note":null,"clear":null}"#)
        else {
            panic!("neither is required, so neither is an error")
        };
        let v: Value = serde_json::from_str(&fixed).expect("still JSON");
        assert!(
            v.get("note").is_none(),
            "a null where only string is allowed means 'not set'"
        );
        assert_eq!(
            v["clear"],
            Value::Null,
            "a schema that accepts null keeps it"
        );
    }

    #[test]
    fn resolve_validates_through_the_wrapper_and_direct_alike() {
        let catalog = vec![CatalogTool {
            name: "artifacts__artifact_create".into(),
            description: "make one".into(),
            input_schema: create_schema(),
        }];
        // through mcp_call_tool: the arguments the grammar never saw
        let wrapped = r#"{"name":"artifacts__artifact_create","arguments_json":"{\"kind\":\"html\",\"title\":\"H\"}"}"#;
        let Resolved::Refuse { name, message } = resolve_call(CALL_TOOL, wrapped, &catalog) else {
            panic!("the envelope must not smuggle a bad call past the check")
        };
        assert_eq!(name, "artifacts__artifact_create");
        assert!(message.contains("`content`"), "{message}");

        // called directly, same verdict - a cloud provider constrains nothing
        let direct = r#"{"kind":"html","title":"H"}"#;
        assert!(matches!(
            resolve_call("artifacts__artifact_create", direct, &catalog),
            Resolved::Refuse { .. }
        ));

        // a name we cannot place still falls through to the unknown-tool path
        assert!(matches!(
            resolve_call("nope", "{}", &catalog),
            Resolved::Call { .. }
        ));
    }

    /// Regression guard for a bug seen in a sibling codebase; this code was
    /// already correct when it was audited, and this test is what keeps it so.
    ///
    /// There, the envelope parsed `arguments_json` with
    /// `from_str(s).ok().unwrap_or_default()`, so malformed JSON became empty
    /// arguments and the schema check then reported a phantom "missing field
    /// <first required field>" - naming a field the model had actually sent.
    /// Live cost: gpt-5.4 emitted a `runs` array missing its `]`, was told
    /// `document_id` was missing (it wasn't), gave up on the tool entirely and
    /// hand-rolled a PDF. The model can only fix what it is told the truth
    /// about, so the parse error must survive to the message.
    #[test]
    fn malformed_inner_json_says_so_instead_of_inventing_a_missing_field() {
        let catalog = vec![CatalogTool {
            name: "artifacts__artifact_create".into(),
            description: "make one".into(),
            input_schema: create_schema(),
        }];
        // every required field present, but the array is missing its `]`
        let broken = r#"{"name":"artifacts__artifact_create","arguments_json":"{\"kind\":\"html\",\"title\":\"H\",\"content\":[1,2}"}"#;
        let Resolved::Refuse { message, .. } = resolve_call(CALL_TOOL, broken, &catalog) else {
            panic!("malformed arguments must be refused, not dispatched as empty")
        };
        assert!(
            message.contains("not valid JSON"),
            "the real reason has to survive: {message}"
        );
        assert!(
            !message.contains("`kind`") && !message.contains("`title`"),
            "must not name a field the model DID send: {message}"
        );

        // ...and the outer envelope gets the same treatment
        let Resolved::Refuse { message, .. } = resolve_call(CALL_TOOL, "{not json", &catalog)
        else {
            panic!("a malformed envelope must be refused too")
        };
        assert!(message.contains("not valid JSON"), "{message}");
    }
}
