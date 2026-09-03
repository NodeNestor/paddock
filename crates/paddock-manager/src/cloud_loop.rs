//! The cloud MCP agent loop: the manager is the MCP host
//! for cloud lanes. Connectors ride the Studio request as inline `mcp` tools;
//! we list their tools (paddock-mcp - credentials never leave this machine),
//! declare them to the provider as plain FUNCTION tools over the Responses
//! wire (OpenAI + OpenRouter - their own cookbook prescribes exactly this
//! client-side pattern), intercept function calls, execute, feed results
//! back, and re-round. The Studio sees the runner's exact item vocabulary
//! (mcp_list_tools / mcp_call / mcp_approval_request), so the cards and the
//! approval gate work unchanged. The Anthropic + generic-compat adapters are
//! not finished; those lanes keep stripping tools until they are.

use std::collections::HashMap;
use std::sync::LazyLock;
use std::sync::Mutex as StdMutex;

use paddock_mcp::clock::{self, ClockSpec};
use paddock_mcp::{loop_budget, tool_search};
use serde_json::{Value, json};
use tokio::sync::oneshot;

const APPROVAL_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(180);

/// Fold the servers' guidance into the request's system field, OURS first and
/// the caller's last - identical to the runner's merge_instructions, and for
/// the same measured reason: a short caller instruction stops being obeyed
/// when a few hundred words of tool procedure trail it. `field` is
/// `instructions` on the Responses wire and `system` on Anthropic's.
fn prepend_system(body: &mut Value, field: &str, blocks: &[String]) {
    if blocks.is_empty() {
        return;
    }
    let ours = blocks.join(
        "

",
    );
    let merged = match body.get(field) {
        Some(Value::String(s)) if !s.trim().is_empty() => format!(
            "{ours}

{s}"
        ),
        // Anthropic also accepts a block array; flatten it behind ours.
        Some(Value::Array(parts)) => {
            let tail: Vec<String> = parts
                .iter()
                .filter_map(|p| p.get("text").and_then(Value::as_str).map(str::to_owned))
                .collect();
            if tail.is_empty() {
                ours
            } else {
                format!(
                    "{ours}

{}",
                    tail.join(
                        "

"
                    )
                )
            }
        }
        _ => ours,
    };
    body[field] = Value::String(merged);
}

/// One connector extracted from the request's inline `mcp` tools.
pub struct Spec {
    pub label: String,
    pub url: String,
    pub headers: HashMap<String, String>,
    pub allowed: Option<Vec<String>>,
    pub needs_approval: bool,
}

/// Pull the inline mcp specs out of a Studio Responses body. Bare labels
/// (a runner's registry reference) have no meaning on a cloud lane and are
/// skipped; the runner tier owns those.
pub fn extract_specs(body: &Value) -> Vec<Spec> {
    let Some(tools) = body.get("tools").and_then(Value::as_array) else {
        return Vec::new();
    };
    tools
        .iter()
        .filter(|t| t.get("type").and_then(Value::as_str) == Some("mcp"))
        .filter_map(|t| {
            let url = t.get("server_url").and_then(Value::as_str)?;
            Some(Spec {
                label: t
                    .get("server_label")
                    .and_then(Value::as_str)
                    .unwrap_or("mcp")
                    .to_owned(),
                url: url.to_owned(),
                headers: t
                    .get("headers")
                    .and_then(Value::as_object)
                    .map(|h| {
                        h.iter()
                            .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_owned())))
                            .collect()
                    })
                    .unwrap_or_default(),
                allowed: t.get("allowed_tools").and_then(Value::as_array).map(|a| {
                    a.iter()
                        .filter_map(Value::as_str)
                        .map(str::to_owned)
                        .collect()
                }),
                needs_approval: t.get("require_approval").and_then(Value::as_str) != Some("never"),
            })
        })
        .collect()
}

/// Pull a `{"type":"current_time"}` tool out of a Studio Responses body: the
/// builtin clock, ours to serve on the cloud lane exactly as the runner serves
/// it locally (a lane's tool behaviour must not depend on where the model
/// runs). A junk timezone is a 400 before anything reaches the provider.
pub fn extract_clock(body: &Value) -> Result<Option<ClockSpec>, String> {
    let Some(tools) = body.get("tools").and_then(Value::as_array) else {
        return Ok(None);
    };
    match tools
        .iter()
        .find(|t| t.get("type").and_then(Value::as_str) == Some("current_time"))
    {
        Some(t) => clock::parse_spec(t).map(Some),
        None => Ok(None),
    }
}

// The approval registry: card id -> the parked round's waker. Same pattern
// as the runner's, module-static like oauth::FLOWS.
static APPROVALS: LazyLock<StdMutex<HashMap<String, oneshot::Sender<bool>>>> =
    LazyLock::new(StdMutex::default);

pub fn resolve_approval(id: &str, approve: bool) -> bool {
    match APPROVALS.lock().unwrap().remove(id) {
        Some(tx) => {
            let _ = tx.send(approve);
            true
        }
        None => false,
    }
}

/// Cumulative output budget across rounds - each round re-spends the
/// per-round cap, so the loop's total is bounded at 4x it (floored so short
/// caps don't strangle a legitimate multi-tool turn). The rule now lives in
/// `loop_budget` with the rest of the turn budget; this reads it off the wire.
fn out_token_cap(body: &Value, field: &str) -> u64 {
    loop_budget::turn_output_cap(body[field].as_u64().unwrap_or(4096) as usize) as u64
}

/// Put the tool-round ceiling on the request, or take it back off for the
/// answer round. Returns true when the round was actually held below what the
/// caller asked for - which is how a `max_tokens` finish is told apart from
/// the caller's own truncation.
///
/// The ceiling is what the turn's tool budget has left, so a cloud round can
/// no longer overshoot that budget by its whole cap; it applies even when the
/// caller named no cap of their own, because a cloud round spends real money.
/// The field goes back to the caller's number (or away entirely) for the
/// answer round, which is deliberately outside the budget. The thinking floor
/// keeps Anthropic's `max_tokens > thinking.budget_tokens` rule intact: a
/// thinking turn is never capped below the budget it already promised.
fn round_ceiling(
    body: &mut Value,
    field: &str,
    answering: bool,
    original: Option<u64>,
    remaining: u64,
) -> bool {
    if answering {
        if let Some(obj) = body.as_object_mut() {
            match original {
                Some(v) => obj.insert(field.to_owned(), json!(v)),
                None => obj.remove(field),
            };
        }
        return false;
    }
    let floor = body["thinking"]["budget_tokens"]
        .as_u64()
        .map_or(0, |b| b + 1024);
    // `MIN_ROUND_TOKENS` for the same reason the runner takes it: the last
    // sliver of a budget must not become a `max_tokens` a provider rejects.
    let cap = remaining
        .max(floor)
        .max(loop_budget::MIN_ROUND_TOKENS as u64);
    let cap = original.map_or(cap, |v| cap.min(v));
    body[field] = json!(cap);
    // A caller who named no cap of their own would have had the provider's
    // default, so whatever we wrote is ours by definition - otherwise a round
    // cut by our ceiling would read as the caller's own truncation and end the
    // turn on an empty answer.
    original.is_none_or(|v| cap < v)
}

fn sse(v: &Value) -> String {
    format!(
        "event: {}\ndata: {}\n\n",
        v["type"].as_str().unwrap_or("message"),
        v
    )
}

fn mcp_call_item(
    id: &str,
    label: &str,
    name: &str,
    args: &str,
    output: Option<&str>,
    error: Option<&str>,
    status: &str,
) -> Value {
    json!({"type":"mcp_call","id":id,"server_label":label,"name":name,"arguments":args,
           "output":output,"error":error,"status":status,"approval_request_id":null})
}

/// Shared listing step: connect + list every connector, emit its
/// mcp_list_tools item, and return (function defs in `shape`, routing).
async fn gather(
    specs: &[Spec],
    shape: fn(&str, &str, &Value) -> Value,
    send: &impl Fn(String),
) -> (
    HashMap<String, paddock_mcp::McpClient>,
    // Defs kept per SERVER (label, defs): disclosure is decided one server at
    // a time, so they cannot be flattened until that decision is made.
    Vec<(String, Vec<Value>)>,
    HashMap<String, (String, String, bool)>,
    // Each server's handshake `instructions`, with the names we declared for
    // it. The runner folds these into the prompt; this
    // loop did not, so a cloud model got the artifact TOOLS with none of the
    // guidance that makes a model reach for them - Sonnet answered inline
    // while the identical local model used them. A lane's tool behaviour must
    // not depend on where the model runs.
    Vec<String>,
) {
    let mut clients = HashMap::new();
    let mut instructions: Vec<String> = Vec::new();
    let mut per_server: Vec<(String, Vec<Value>)> = Vec::new();
    let mut routing = HashMap::new();
    for spec in specs {
        let cfg = paddock_mcp::ServerConfig {
            id: format!("cloud:{}", spec.url),
            label: spec.label.clone(),
            transport: paddock_mcp::Transport::Http {
                url: spec.url.clone(),
                headers: spec.headers.clone(),
            },
        };
        // 60s guard: generous enough for a cold-starting server (IIS app
        // pools take 15-30s - the maintainers' tic), but a host that accepts and never
        // answers becomes a failed listing card instead of a hung chat turn
        // (the client itself carries no timeout).
        let listed = match tokio::time::timeout(std::time::Duration::from_secs(60), async {
            let c = paddock_mcp::McpClient::connect(&cfg).await?;
            let tools = c.list_tools().await?;
            Ok::<_, paddock_mcp::McpError>((c, tools))
        })
        .await
        {
            Ok(Ok((c, tools))) => {
                clients.insert(spec.label.clone(), c);
                Ok(tools)
            }
            Ok(Err(e)) => Err(e.to_string()),
            Err(_) => Err("timed out".to_owned()),
        };
        match listed {
            Ok(tools) => {
                let kept: Vec<_> = tools
                    .into_iter()
                    .filter(|t| {
                        spec.allowed
                            .as_ref()
                            .is_none_or(|a| a.iter().any(|n| n == &t.name))
                    })
                    .collect();
                send(sse(&json!({"type":"response.output_item.added","item":{
                    "type":"mcp_list_tools","id":format!("mcpl_{}", uuid::Uuid::new_v4().simple()),
                    "server_label":spec.label,
                    "tools":kept.iter().map(|t| json!({"name":t.name,"description":t.description,"input_schema":t.input_schema})).collect::<Vec<_>>(),
                }})));
                let mut declared: Vec<String> = Vec::new();
                let mut server_defs: Vec<Value> = Vec::new();
                for t in kept {
                    let ns = format!("{}__{}", spec.label, t.name);
                    server_defs.push(shape(
                        &ns,
                        t.description.as_deref().unwrap_or(""),
                        &t.input_schema,
                    ));
                    declared.push(ns.clone());
                    routing.insert(ns, (spec.label.clone(), t.name, spec.needs_approval));
                }
                match per_server.iter_mut().find(|(l, _)| l == &spec.label) {
                    Some((_, d)) => d.extend(server_defs),
                    None => per_server.push((spec.label.clone(), server_defs)),
                }
                // Same treatment the runner gives: the server's own guidance,
                // plus the names we actually declared (it writes them
                // unprefixed and cannot know our `label__tool` namespacing).
                if let Some(instr) = clients.get(&spec.label).and_then(|c| c.instructions())
                    && !declared.is_empty()
                {
                    instructions.push(format!(
                        "{instr}

Call this server's tools by these exact names: {}.",
                        declared.join(", ")
                    ));
                }
            }
            Err(e) => {
                send(sse(&json!({"type":"response.output_item.added","item":{
                    "type":"mcp_list_tools","id":format!("mcpl_{}", uuid::Uuid::new_v4().simple()),
                    "server_label":spec.label,"tools":[],"error":e}})));
            }
        }
    }
    (clients, per_server, routing, instructions)
}

/// Split the gathered defs into what the provider gets declared and what goes
/// behind search, plus the notice that matches.
///
/// Same policy and the same shared decision the runner makes - a lane's tool
/// behaviour must not depend on where the model runs, and the split confused
/// real use the last time the two drifted. `max_ctx` is 0 here: a cloud
/// context window is never the binding constraint, the tool COUNT is.
fn disclose(per_server: Vec<(String, Vec<Value>)>) -> (Vec<Value>, Option<String>) {
    let weights: Vec<tool_search::ServerWeight> = per_server
        .iter()
        .map(|(label, defs)| tool_search::ServerWeight {
            label: label.clone(),
            tools: defs.len(),
            chars: defs.iter().map(|d| d.to_string().chars().count()).sum(),
        })
        .collect();
    let total: usize = weights.iter().map(|w| w.tools).sum();
    let shown = tool_search::disclose_servers(&weights, 0);
    let mut declared: Vec<Value> = Vec::new();
    let mut hidden_labels: Vec<String> = Vec::new();
    let mut hidden_tools = 0usize;
    for (label, defs) in per_server {
        if shown.contains(&label) {
            declared.extend(defs);
        } else {
            hidden_tools += defs.len();
            hidden_labels.push(label);
        }
    }
    let notice = if total == 0 {
        None
    } else if declared.is_empty() {
        Some(tool_search::SEARCH_MODE_INSTRUCTIONS.to_owned())
    } else if hidden_labels.is_empty() {
        Some(tool_search::SEARCH_AVAILABLE_INSTRUCTIONS.to_owned())
    } else {
        Some(tool_search::partial_mode_instructions(
            &hidden_labels,
            hidden_tools,
        ))
    };
    (declared, notice)
}

/// One gated + executed call; returns the feedback text for the model.
async fn execute(
    call_id: &str,
    ns: &str,
    args_text: &str,
    args: Result<Value, String>,
    routing: &HashMap<String, (String, String, bool)>,
    clients: &HashMap<String, paddock_mcp::McpClient>,
    send: &impl Fn(String),
) -> (String, bool) {
    let Some((label, real, gated)) = routing.get(ns).cloned() else {
        let m = format!("unknown tool {ns:?}");
        send(sse(&json!({"type":"response.output_item.done","item":
            mcp_call_item(call_id, "mcp", ns, args_text, None, Some(&m), "failed")})));
        return (m, true);
    };
    if gated {
        let approval_id = format!("appr_{}", uuid::Uuid::new_v4().simple());
        let (atx, arx) = oneshot::channel();
        APPROVALS.lock().unwrap().insert(approval_id.clone(), atx);
        send(sse(&json!({"type":"response.output_item.added","item":{
            "type":"mcp_approval_request","id":approval_id,"call_id":call_id,
            "server_label":label,"name":real,"arguments":args_text}})));
        let approved = matches!(
            tokio::time::timeout(APPROVAL_TIMEOUT, arx).await,
            Ok(Ok(true))
        );
        APPROVALS.lock().unwrap().remove(&approval_id);
        send(sse(&json!({"type":"response.output_item.done","item":{
            "type":"mcp_approval_request","id":approval_id,"call_id":call_id,
            "server_label":label,"name":real,"arguments":args_text,
            "status": if approved {"approved"} else {"denied"}}})));
        if !approved {
            return ("the user denied this tool call".to_owned(), true);
        }
    }
    send(sse(&json!({"type":"response.output_item.added","item":
        mcp_call_item(call_id, &label, &real, args_text, None, None, "in_progress")})));
    let (output, error) = match args {
        Err(e) => (None, Some(e)),
        Ok(a) => match clients.get(&label) {
            None => (None, Some("connector is not connected".to_owned())),
            Some(c) => match c.call_tool(&real, a).await {
                Ok(r) => {
                    let text = serde_json::to_string(&r.content).unwrap_or_default();
                    if r.is_error {
                        (None, Some(text))
                    } else {
                        (Some(text), None)
                    }
                }
                Err(e) => (None, Some(e.to_string())),
            },
        },
    };
    let status = if error.is_none() {
        "completed"
    } else {
        "failed"
    };
    send(sse(&json!({"type":"response.output_item.done","item":
        mcp_call_item(call_id, &label, &real, args_text, output.as_deref(), error.as_deref(), status)})));
    let is_err = error.is_some();
    (output.or(error).unwrap_or_default(), is_err)
}

/// The Anthropic lane: native tool_use/tool_result blocks over /v1/messages.
/// Intermediate rounds run NON-streaming (a single JSON body carries the
/// content blocks - thinking rides back verbatim, as their docs require);
/// the final round's text is emitted to the Studio as synthetic deltas.
/// True token streaming for tool turns is the documented upgrade point.
pub async fn run_anthropic(
    specs: Vec<Spec>,
    clock_spec: Option<ClockSpec>,
    mut body: Value,
    // The CALLER's `max_tool_calls`, read off the Responses request they sent
    // US - Anthropic's own wire has no such field, and this loop is the thing
    // that would spend it, so it never travels upstream either way.
    max_tool_calls: Option<usize>,
    post: impl Fn(Value) -> reqwest::RequestBuilder,
    tx: futures::channel::mpsc::UnboundedSender<String>,
) {
    let send = |s: String| {
        let _ = tx.unbounded_send(s);
    };
    let (clients, per_server, routing, mut instructions) = gather(
        &specs,
        |ns, desc, schema| json!({"name":ns,"description":desc,"input_schema":schema}),
        &send,
    )
    .await;
    let catalog: Vec<tool_search::CatalogTool> = per_server
        .iter()
        .flat_map(|(_, defs)| defs.iter())
        .map(|d| tool_search::CatalogTool {
            name: d["name"].as_str().unwrap_or_default().to_owned(),
            description: d["description"].as_str().unwrap_or_default().to_owned(),
            input_schema: d["input_schema"].clone(),
        })
        .collect();
    let (mut tools, notice) = disclose(per_server);
    // The search pair rides along in every mode now, matching the runner:
    // searchability is not a mode, only the direct schemas are.
    if !catalog.is_empty() {
        tools.push(tool_search::search_tool_def_anthropic());
        tools.push(tool_search::call_tool_def_anthropic());
    }
    if clock_spec.is_some() {
        tools.push(clock::anthropic_tool_def());
    }
    body["tools"] = json!(tools);
    instructions.extend(notice);
    prepend_system(&mut body, "system", &instructions);
    // every round streams: thinking/text deltas forward live, and the
    // assistant's content blocks (signatures included) are reassembled
    // verbatim for the history the next round passes back
    body["stream"] = json!(true);
    let out_cap = out_token_cap(&body, "max_tokens");
    let asked_max = body["max_tokens"].as_u64();
    // every round bills separately - the completed event carries the SUM
    // (Anthropic reports no cost or reasoning split; tokens are the story)
    let (mut total_out, mut total_in) = (0u64, 0u64);
    // The turn budget: repeat ledger, per-round ceiling, and one last
    // round with the tools switched off that answers with what came back.
    let mut ledger = loop_budget::CallLedger::with_limit(max_tool_calls);
    // Their number sets the ROUND ceiling as well - see `rounds_cap`.
    let rounds = loop_budget::rounds_cap(max_tool_calls);
    let mut stop: Option<loop_budget::Stop> = None;
    let mut announced = false;
    for round in 0..=rounds {
        // The caller's ceiling, before the round: `max_tool_calls: 0` has to
        // stop the turn before it calls anything.
        if stop.is_none() {
            stop = ledger.limit_reached();
        }
        if stop.is_none() && round == rounds {
            stop = Some(loop_budget::Stop::Rounds(rounds));
        }
        let answering = stop.is_some();
        if answering && !announced {
            announced = true;
            // tool_choice "none" rather than dropping `tools`: the history
            // holds tool_use blocks, and Anthropic reads a request whose
            // history uses tools but declares none as malformed. "none" is
            // their documented way to say this.
            body["tool_choice"] = json!({"type": "none"});
            send(sse(&json!({"type":"response.output_text.delta",
                "delta":format!("\n\n{}\n\n", stop.expect("answering means stopped").notice())})));
            if let Some(msgs) = body["messages"].as_array_mut() {
                append_user_text(msgs, loop_budget::ANSWER_ONLY_NUDGE);
            }
        }
        let ours = round_ceiling(
            &mut body,
            "max_tokens",
            answering,
            asked_max,
            out_cap.saturating_sub(total_out),
        );
        let res = match post(body.clone()).send().await {
            Ok(r) if r.status().is_success() => r,
            Ok(r) => {
                let detail = r.text().await.unwrap_or_default();
                send(sse(
                    &json!({"type":"response.failed","response":{"status":"failed",
                    "error":{"message":detail.chars().take(400).collect::<String>()}}}),
                ));
                return;
            }
            Err(e) => {
                send(sse(
                    &json!({"type":"response.failed","response":{"status":"failed",
                    "error":{"message":format!("provider not answering: {e}")}}}),
                ));
                return;
            }
        };
        let mut content: Vec<Value> = Vec::new();
        let mut json_buf: HashMap<usize, String> = HashMap::new();
        let (mut stop_reason, mut in_tok, mut out_tok) = (String::new(), 0u64, 0u64);
        let mut buf = String::new();
        let mut stream = res.bytes_stream();
        use futures::StreamExt;
        while let Some(chunk) = stream.next().await {
            let Ok(chunk) = chunk else { break };
            buf.push_str(&String::from_utf8_lossy(&chunk));
            while let Some(i) = buf.find("\n\n") {
                let frame: String = buf.drain(..i + 2).collect();
                let data: String = frame
                    .lines()
                    .filter_map(|l| l.strip_prefix("data: "))
                    .collect::<Vec<_>>()
                    .join("");
                let Ok(v) = serde_json::from_str::<Value>(&data) else {
                    continue;
                };
                match v["type"].as_str().unwrap_or("") {
                    "message_start" => {
                        in_tok = v["message"]["usage"]["input_tokens"].as_u64().unwrap_or(0);
                    }
                    "content_block_start" => {
                        let idx = v["index"].as_u64().unwrap_or(0) as usize;
                        while content.len() <= idx {
                            content.push(json!({}));
                        }
                        content[idx] = v["content_block"].clone();
                    }
                    "content_block_delta" => {
                        let idx = v["index"].as_u64().unwrap_or(0) as usize;
                        let d = &v["delta"];
                        match d["type"].as_str().unwrap_or("") {
                            "thinking_delta" => {
                                let t = d["thinking"].as_str().unwrap_or("");
                                send(sse(
                                    &json!({"type":"response.reasoning_text.delta","delta":t}),
                                ));
                                if let Some(b) = content.get_mut(idx) {
                                    let cur = b["thinking"].as_str().unwrap_or("").to_owned();
                                    b["thinking"] = json!(cur + t);
                                }
                            }
                            "text_delta" => {
                                let t = d["text"].as_str().unwrap_or("");
                                send(sse(&json!({"type":"response.output_text.delta","delta":t})));
                                if let Some(b) = content.get_mut(idx) {
                                    let cur = b["text"].as_str().unwrap_or("").to_owned();
                                    b["text"] = json!(cur + t);
                                }
                            }
                            "input_json_delta" => {
                                json_buf
                                    .entry(idx)
                                    .or_default()
                                    .push_str(d["partial_json"].as_str().unwrap_or(""));
                            }
                            "signature_delta" => {
                                if let Some(b) = content.get_mut(idx) {
                                    let cur = b["signature"].as_str().unwrap_or("").to_owned();
                                    b["signature"] =
                                        json!(cur + d["signature"].as_str().unwrap_or(""));
                                }
                            }
                            _ => {}
                        }
                    }
                    "content_block_stop" => {
                        let idx = v["index"].as_u64().unwrap_or(0) as usize;
                        if let Some(j) = json_buf.remove(&idx)
                            && let Some(b) = content.get_mut(idx)
                            && b["type"] == "tool_use"
                        {
                            b["input"] = serde_json::from_str(&j).unwrap_or(json!({}));
                        }
                    }
                    "message_delta" => {
                        if let Some(s) = v["delta"]["stop_reason"].as_str() {
                            stop_reason = s.to_owned();
                        }
                        if let Some(o) = v["usage"]["output_tokens"].as_u64() {
                            out_tok = o;
                        }
                    }
                    "error" => {
                        send(sse(
                            &json!({"type":"response.failed","response":{"status":"failed",
                            "error":{"message":v["error"]["message"].clone()}}}),
                        ));
                        return;
                    }
                    _ => {}
                }
            }
        }
        total_out += out_tok;
        total_in += in_tok;
        let tool_uses: Vec<&Value> = content.iter().filter(|b| b["type"] == "tool_use").collect();
        // A round the tool BUDGET cut (rather than the caller's own
        // max_tokens) is checked before the terminal below: it is exactly the
        // case where the round produced nothing usable, so returning here
        // would hand back an empty turn.
        if !answering && ours && stop_reason == "max_tokens" {
            stop = Some(loop_budget::Stop::Output);
            continue;
        }
        // The answer round ends the turn whatever it said, and nothing it may
        // have emitted as a tool runs.
        if answering || stop_reason != "tool_use" || tool_uses.is_empty() {
            send(sse(
                &json!({"type":"response.completed","response":{"status":"completed",
                "usage":{"input_tokens":total_in,"output_tokens":total_out}}}),
            ));
            return;
        }
        // the assistant turn goes back VERBATIM - thinking blocks included
        body["messages"]
            .as_array_mut()
            .expect("messages is an array")
            .push(json!({"role":"assistant","content":content}));
        // PLAN sequentially (the repeat ledger has to see the round in order),
        // then execute CONCURRENTLY; tool_results append in call order.
        let plans: Vec<RoundPlan> = tool_uses
            .iter()
            .map(|tu| {
                plan_call(
                    tu["id"].as_str().unwrap_or_default(),
                    tu["name"].as_str().unwrap_or_default(),
                    &tu["input"].to_string(),
                    clock_spec,
                    &catalog,
                    &routing,
                    &mut ledger,
                )
            })
            .collect();
        let futs = plans.iter().map(|p| {
            let (catalog, routing, clients, send) = (&catalog, &routing, &clients, &send);
            async move { run_planned(p, clock_spec, catalog, routing, clients, send).await }
        });
        let outcomes = futures::future::join_all(futs).await;
        let mut results: Vec<Value> = Vec::with_capacity(outcomes.len());
        for (p, (feedback, is_err)) in plans.iter().zip(outcomes) {
            if let Some(sig) = &p.sig {
                ledger.record(sig, !is_err, &feedback);
            }
            results.push(json!({"type":"tool_result","tool_use_id":p.call_id,
                "content":feedback,"is_error":is_err}));
        }
        body["messages"]
            .as_array_mut()
            .unwrap()
            .push(json!({"role":"user","content":results}));
        if total_out >= out_cap {
            stop = Some(loop_budget::Stop::Output);
        }
    }
}

/// Append instruction text to the conversation for the answer round, joining
/// the last user turn when it is a block array (after a tool round it always
/// is, holding the tool_results) - Anthropic rejects two user turns in a row,
/// and text after tool_result in the same turn is what their docs show.
fn append_user_text(messages: &mut Vec<Value>, text: &str) {
    if let Some(last) = messages.last_mut()
        && last["role"] == "user"
        && let Some(blocks) = last["content"].as_array_mut()
    {
        blocks.push(json!({"type": "text", "text": text}));
        return;
    }
    messages.push(json!({"role": "user", "content": text}));
}

/// Run the loop and stream Studio-dialect SSE into `tx`. `send` posts one
/// provider round and returns the response; the caller owns auth/base/model.
pub async fn run(
    specs: Vec<Spec>,
    clock_spec: Option<ClockSpec>,
    mut body: Value,
    // The caller's `max_tool_calls` - see run_anthropic. It is stripped from
    // the body below rather than forwarded: we execute these tools, so a
    // provider counting its own built-ins against the same number would bound
    // a budget that has already been spent here.
    max_tool_calls: Option<usize>,
    post: impl Fn(Value) -> reqwest::RequestBuilder,
    tx: futures::channel::mpsc::UnboundedSender<String>,
) {
    let send = |s: String| {
        let _ = tx.unbounded_send(s);
    };
    // 1) connect + list every connector (shared gather: a dead one becomes a
    // failed listing item and the rest carry on; a cold-starting one gets the
    // 60s budget) - defs land in the Responses flat function shape.
    let (clients, per_server, routing, mut instructions) = gather(
        &specs,
        |ns, desc, schema| {
            json!({"type":"function","name":ns,"description":desc,"parameters":schema})
        },
        &send,
    )
    .await;
    // The disclosure fork, same decision as the runner (a lane's tool behavior
    // must not depend on where the model runs - the split confused real use):
    // big servers go behind the two meta-tools, searches resolve locally
    // (BM25) with inline schemas; small servers keep their real schemas.
    let catalog: Vec<tool_search::CatalogTool> = per_server
        .iter()
        .flat_map(|(_, defs)| defs.iter())
        .map(|d| tool_search::CatalogTool {
            name: d["name"].as_str().unwrap_or_default().to_owned(),
            description: d["description"].as_str().unwrap_or_default().to_owned(),
            input_schema: d["parameters"].clone(),
        })
        .collect();
    let (mut tools, notice) = disclose(per_server);
    if !catalog.is_empty() {
        // flatten the chat-completions-nested defs to the Responses shape
        let flat = |d: Value| {
            json!({"type":"function","name":d["function"]["name"],
                "description":d["function"]["description"],
                "parameters":d["function"]["parameters"]})
        };
        tools.push(flat(tool_search::search_tool_def()));
        tools.push(flat(tool_search::call_tool_def()));
    }
    // The builtin clock joins the declared set directly - one tiny schema,
    // never behind the search pair.
    if clock_spec.is_some() {
        tools.push(clock::responses_tool_def());
    }
    body["tools"] = json!(tools);
    instructions.extend(notice);
    prepend_system(&mut body, "instructions", &instructions);
    // the Responses `input` may be a bare string; rounds append items, so
    // normalize to the items form first
    if let Some(s) = body["input"].as_str() {
        let s = s.to_owned();
        body["input"] = json!([{"type":"message","role":"user","content": s}]);
    }

    // 2) rounds. Provider events stream through verbatim EXCEPT the terminal
    // response.completed, which we swallow between rounds (it would end the
    // Studio's read) and forward only from the final round.
    let out_cap = out_token_cap(&body, "max_output_tokens");
    let asked_max = body["max_output_tokens"].as_u64();
    // Ours to enforce, not the provider's (see the parameter).
    if let Some(obj) = body.as_object_mut() {
        obj.remove("max_tool_calls");
    }
    // whole-turn usage: every round bills separately, so the terminal we
    // forward must carry the SUM - the Studio stat line and the manager's
    // usage ledger both read that one event (a 5-round tool turn otherwise
    // reports only its final round's tokens and cost)
    let mut total_out = 0u64;
    let (mut total_in, mut total_reason, mut total_cost) = (0u64, 0u64, None::<f64>);
    // The turn budget - the same three levers as every other loop.
    let mut ledger = loop_budget::CallLedger::with_limit(max_tool_calls);
    // Their number sets the ROUND ceiling as well - see `rounds_cap`.
    let rounds = loop_budget::rounds_cap(max_tool_calls);
    let mut stop: Option<loop_budget::Stop> = None;
    let mut announced = false;
    for round in 0..=rounds {
        // The caller's ceiling, before the round: `max_tool_calls: 0` has to
        // stop the turn before it calls anything.
        if stop.is_none() {
            stop = ledger.limit_reached();
        }
        if stop.is_none() && round == rounds {
            stop = Some(loop_budget::Stop::Rounds(rounds));
        }
        let answering = stop.is_some();
        if answering && !announced {
            announced = true;
            // tool_choice "none" rather than dropping `tools`: the input items
            // already carry function_call/function_call_output pairs, and a
            // request whose history uses tools but declares none is the shape
            // providers reject. "none" is the documented way to say it.
            body["tool_choice"] = json!("none");
            // A text DELTA, not a bare message item: a reader accumulating
            // output_text never looks inside an item, so the old item-only
            // "[tool loop stopped]" notice was invisible in the Studio.
            send(sse(&json!({"type":"response.output_text.delta",
                "delta":format!("{}\n\n", stop.expect("answering means stopped").notice())})));
            if let Some(items) = body["input"].as_array_mut() {
                items.push(json!({"type":"message","role":"user",
                    "content": loop_budget::ANSWER_ONLY_NUDGE}));
            }
        }
        let ours = round_ceiling(
            &mut body,
            "max_output_tokens",
            answering,
            asked_max,
            out_cap.saturating_sub(total_out),
        );
        let res = match post(body.clone()).send().await {
            Ok(r) if r.status().is_success() => r,
            Ok(r) => {
                let detail = r.text().await.unwrap_or_default();
                send(sse(
                    &json!({"type":"response.failed","response":{"status":"failed",
                    "error":{"message":format!("provider error: {}", detail.chars().take(400).collect::<String>())}}}),
                ));
                return;
            }
            Err(e) => {
                send(sse(
                    &json!({"type":"response.failed","response":{"status":"failed",
                    "error":{"message":format!("provider not answering: {e}")}}}),
                ));
                return;
            }
        };
        let mut calls: Vec<(String, String, String)> = Vec::new(); // (call_id, fn name, args)
        let mut terminal: Option<Value> = None;
        let mut buf = String::new();
        let mut stream = res.bytes_stream();
        use futures::StreamExt;
        while let Some(chunk) = stream.next().await {
            let Ok(chunk) = chunk else { break };
            buf.push_str(&String::from_utf8_lossy(&chunk));
            while let Some(i) = buf.find("\n\n") {
                let frame: String = buf.drain(..i + 2).collect();
                let data: String = frame
                    .lines()
                    .filter_map(|l| l.strip_prefix("data: "))
                    .collect::<Vec<_>>()
                    .join("");
                if data.is_empty() {
                    continue;
                }
                let Ok(v) = serde_json::from_str::<Value>(&data) else {
                    continue;
                };
                match v["type"].as_str().unwrap_or("") {
                    "response.completed" | "response.incomplete" => {
                        // harvest this round's function calls from the final
                        // output (authoritative over delta reassembly)
                        if let Some(out) = v["response"]["output"].as_array() {
                            for it in out {
                                if it["type"] == "function_call" {
                                    calls.push((
                                        it["call_id"].as_str().unwrap_or_default().to_owned(),
                                        it["name"].as_str().unwrap_or_default().to_owned(),
                                        it["arguments"].as_str().unwrap_or("{}").to_owned(),
                                    ));
                                }
                            }
                        }
                        terminal = Some(v);
                    }
                    "response.failed" => {
                        send(sse(&v));
                        return;
                    }
                    _ => send(sse(&v)),
                }
            }
        }
        let Some(mut terminal) = terminal else {
            send(sse(
                &json!({"type":"response.failed","response":{"status":"failed",
                "error":{"message":"provider stream ended without a terminal event"}}}),
            ));
            return;
        };
        {
            let u = &terminal["response"]["usage"];
            total_out += u["output_tokens"].as_u64().unwrap_or(0);
            total_in += u["input_tokens"].as_u64().unwrap_or(0);
            total_reason += u["output_tokens_details"]["reasoning_tokens"]
                .as_u64()
                .unwrap_or(0);
            if let Some(c) = u["cost"].as_f64() {
                total_cost = Some(total_cost.unwrap_or(0.0) + c);
            }
        }
        // A round the tool BUDGET cut mid-flight, checked before the terminal:
        // it is exactly the case where the round produced no usable call, so
        // returning here would hand back an empty turn. Its calls are dropped
        // and the answer round follows with the caller's full cap.
        if !answering
            && ours
            && terminal["response"]["incomplete_details"]["reason"] == "max_output_tokens"
        {
            stop = Some(loop_budget::Stop::Output);
            continue;
        }
        // The answer round ends the turn whatever it said, and nothing it may
        // have emitted as a tool runs.
        if answering || calls.is_empty() {
            let u = &mut terminal["response"]["usage"];
            if !u.is_null() {
                u["input_tokens"] = json!(total_in);
                u["output_tokens"] = json!(total_out);
                u["output_tokens_details"]["reasoning_tokens"] = json!(total_reason);
                if let Some(c) = total_cost {
                    u["cost"] = json!(c);
                }
            }
            send(sse(&terminal));
            return;
        }
        // 3) execute this round's calls: local meta-tools (search mode)
        // resolve here; real tools go through approval + execution
        let items = body["input"].as_array_mut().expect("input is an array");
        // the round's full output rides back verbatim - reasoning items
        // (signatures included, which Anthropic models require on tool
        // turns) and any text interleaved before the calls, not just the
        // function_call items (a thinking tool round
        // returns [reasoning, message, function_call] and OpenRouter
        // accepts the verbatim replay). Then the outputs append in call
        // order - execution itself is CONCURRENT, three slow remote MCPs
        // in one round dial together.
        if let Some(out) = terminal["response"]["output"].as_array() {
            for it in out {
                items.push(it.clone());
            }
        }
        // PLAN sequentially (the repeat ledger has to see the round in the
        // order the model wrote it), then execute CONCURRENTLY.
        let plans: Vec<RoundPlan> = calls
            .iter()
            .map(|(call_id, ns, args)| {
                plan_call(
                    call_id,
                    ns,
                    args,
                    clock_spec,
                    &catalog,
                    &routing,
                    &mut ledger,
                )
            })
            .collect();
        let futs = plans.iter().map(|p| {
            let (catalog, routing, clients, send) = (&catalog, &routing, &clients, &send);
            async move { run_planned(p, clock_spec, catalog, routing, clients, send).await }
        });
        let outcomes = futures::future::join_all(futs).await;
        let items = body["input"].as_array_mut().expect("input is an array");
        for (p, (feedback, is_err)) in plans.iter().zip(outcomes) {
            if let Some(sig) = &p.sig {
                ledger.record(sig, !is_err, &feedback);
            }
            items
                .push(json!({"type":"function_call_output","call_id":p.call_id,"output":feedback}));
        }
        if total_out >= out_cap {
            stop = Some(loop_budget::Stop::Output);
        }
    }
}

/// What one call of a round turned out to be, decided before anything runs.
enum Planned {
    /// The local catalog search - no server, no gate.
    Search { query: String, limit: usize },
    /// A real tool: approval (if gated) then execution.
    Invoke { ns: String, args: String },
    /// The builtin clock - answered here, no server at all.
    Clock,
    /// Already run this turn with these exact arguments.
    Replay { name: String, output: String },
    /// Never dispatched: the schema check or the repeat ledger said no.
    Refuse { name: String, message: String },
}

/// One planned call, its card id, and its slot in the turn's repeat ledger.
struct RoundPlan {
    call_id: String,
    /// The model's own arguments text, which is what the card shows.
    raw_args: String,
    planned: Planned,
    sig: Option<loop_budget::Signature>,
}

/// Decide what a call is, without running it.
///
/// Planning is sequential and execution is concurrent, which is the only way
/// the repeat ledger can be right: it has to see a round's calls in the order
/// the model wrote them, and two identical calls in one round have to resolve
/// against each other rather than race.
fn plan_call(
    call_id: &str,
    ns: &str,
    args: &str,
    clock_spec: Option<ClockSpec>,
    catalog: &[tool_search::CatalogTool],
    routing: &HashMap<String, (String, String, bool)>,
    ledger: &mut loop_budget::CallLedger,
) -> RoundPlan {
    let plan = |planned, sig| RoundPlan {
        call_id: call_id.to_owned(),
        raw_args: args.to_owned(),
        planned,
        sig,
    };
    let verdict = |ledger: &mut loop_budget::CallLedger,
                   name: &str,
                   ident: &str,
                   fresh: Planned| {
        match ledger.check(name, ident) {
            (sig, loop_budget::Verdict::Fresh) => (fresh, Some(sig)),
            (_, loop_budget::Verdict::Replay(output)) => (
                Planned::Replay {
                    name: name.to_owned(),
                    output,
                },
                None,
            ),
            (_, loop_budget::Verdict::Refuse(message)) => (
                Planned::Refuse {
                    name: name.to_owned(),
                    message,
                },
                None,
            ),
        }
    };
    // `functions.mcp_call_tool` is the same tool as `mcp_call_tool`; the prefix
    // is the caller's dialect leaking into the name. Matching before the strip
    // sent it down the "unknown tool" path.
    let ns = if routing.contains_key(ns) {
        ns
    } else {
        tool_search::strip_client_prefix(ns)
    };
    // The builtin clock: ours, not the catalog's. Raw args as identity, same
    // as the runner - a repeated identical call replays (seconds-stale inside
    // one turn, within the tool's minute resolution) instead of fueling a
    // loop.
    if clock_spec.is_some() && ns == clock::TOOL_NAME {
        let (planned, sig) = verdict(ledger, clock::TOOL_NAME, args, Planned::Clock);
        return plan(planned, sig);
    }
    if ns == tool_search::SEARCH_TOOL {
        let a: Value = serde_json::from_str(args).unwrap_or(json!({}));
        let query = a["query"].as_str().unwrap_or("").to_owned();
        let limit = a["limit"].as_u64().unwrap_or(5).min(25) as usize;
        // Discovery has a budget of its own. A search that is past it
        // never runs, so - like every other refusal here - it stays out of the
        // ledger and spends none of the caller's `max_tool_calls`.
        if let Some(message) = ledger.search_budget_spent() {
            return plan(
                Planned::Refuse {
                    name: tool_search::SEARCH_TOOL.to_owned(),
                    message,
                },
                None,
            );
        }
        let ident = json!({"query": query, "limit": limit}).to_string();
        let (planned, sig) = verdict(
            ledger,
            tool_search::SEARCH_TOOL,
            &ident,
            Planned::Search { query, limit },
        );
        return plan(planned, sig);
    }
    // mcp_call_tool wraps the real call: {name, arguments_json}. Unwrapping AND
    // the schema check are shared with both runner dialects
    // (paddock_mcp::tool_search) - the unwrapping alone had lived in three
    // copies and already drifted, and the one here was the strictest: it took
    // only a STRING arguments_json and turned a missing `name` into a dispatch
    // of "". That is what killed four calls in a row on gpt-5.6 before it gave
    // up and found artifact_create by itself. No provider
    // constrains its own tool arguments, so on this lane the check is the only
    // thing standing between a malformed call and a wasted round.
    let (real_ns, args_text) = match tool_search::resolve_call(ns, args, catalog) {
        tool_search::Resolved::Refuse { name, message } => {
            // A refusal never ran, so it stays out of the ledger.
            return plan(Planned::Refuse { name, message }, None);
        }
        tool_search::Resolved::Call { name, arguments } => (name, arguments),
    };
    let (planned, sig) = verdict(
        ledger,
        &real_ns,
        &args_text,
        Planned::Invoke {
            ns: real_ns.clone(),
            args: args_text.clone(),
        },
    );
    plan(planned, sig)
}

/// Run one planned call and return the feedback text the model sees.
async fn run_planned(
    plan: &RoundPlan,
    clock_spec: Option<ClockSpec>,
    catalog: &[tool_search::CatalogTool],
    routing: &HashMap<String, (String, String, bool)>,
    clients: &HashMap<String, paddock_mcp::McpClient>,
    send: &impl Fn(String),
) -> (String, bool) {
    let (call_id, raw) = (plan.call_id.as_str(), plan.raw_args.as_str());
    match &plan.planned {
        Planned::Search { query, limit } => {
            let hits = tool_search::search(catalog, query, *limit);
            let result = tool_search::search_result(query, &hits, catalog);
            send(sse(&json!({"type":"response.output_item.done","item":
                mcp_call_item(call_id, "mcp", tool_search::SEARCH_TOOL, raw, Some(&result), None, "completed")})));
            (result, false)
        }
        Planned::Clock => {
            let spec = clock_spec.expect("clock spec");
            let (content, output, error, status) = clock::run(spec, raw);
            send(sse(&json!({"type":"response.output_item.done","item":
                mcp_call_item(call_id, "time", clock::TOOL_NAME, raw, output.as_deref(), error.as_deref(), status)})));
            let is_err = error.is_some();
            (content, is_err)
        }
        Planned::Refuse { name, message } => {
            send(sse(&json!({"type":"response.output_item.done","item":
                mcp_call_item(call_id, "mcp", name, raw, None, Some(message), "failed")})));
            (message.clone(), true)
        }
        // Nothing was touched; the card carries the server a live call would
        // have shown, so the transcript reads as what happened.
        Planned::Replay { name, output } => {
            let fallback = if name == clock::TOOL_NAME {
                "time"
            } else {
                "mcp"
            };
            let label = routing.get(name).map_or(fallback, |(l, _, _)| l.as_str());
            let real = routing
                .get(name)
                .map_or(name.as_str(), |(_, r, _)| r.as_str());
            send(sse(&json!({"type":"response.output_item.done","item":
                mcp_call_item(call_id, label, real, raw, Some(output), None, "completed")})));
            (output.clone(), false)
        }
        Planned::Invoke { ns, args } => {
            let parsed = serde_json::from_str::<Value>(args)
                .map_err(|e| format!("arguments are not valid JSON: {e}"));
            execute(call_id, ns, args, parsed, routing, clients, send).await
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn one_tool() -> (
        Vec<tool_search::CatalogTool>,
        HashMap<String, (String, String, bool)>,
    ) {
        (
            vec![tool_search::CatalogTool {
                name: "artifacts__artifact_read".into(),
                description: "read one".into(),
                input_schema: json!({"type":"object",
                    "properties":{"artifact_id":{"type":"string"}},
                    "required":["artifact_id"]}),
            }],
            HashMap::from([(
                "artifacts__artifact_read".to_string(),
                ("artifacts".to_string(), "artifact_read".to_string(), false),
            )]),
        )
    }

    /// Lever 1 on the cloud lanes, where nothing else stands between a
    /// looping model and the provider's bill.
    #[test]
    fn a_repeated_cloud_call_replays_then_is_refused() {
        let (catalog, routing) = one_tool();
        let mut ledger = loop_budget::CallLedger::new();
        let args = r#"{"artifact_id":"a1"}"#;

        let p = plan_call(
            "c1",
            "artifacts__artifact_read",
            args,
            None,
            &catalog,
            &routing,
            &mut ledger,
        );
        assert!(matches!(p.planned, Planned::Invoke { .. }));
        ledger.record(&p.sig.expect("a call that runs is filed"), true, "the page");

        // Same call, wrapper spelling, and the OpenAI-dialect name prefix on
        // top of it: still one call.
        let wrapped =
            r#"{"name":"artifacts__artifact_read","arguments_json":"{\"artifact_id\":\"a1\"}"}"#;
        let p = plan_call(
            "c2",
            "functions.mcp_call_tool",
            wrapped,
            None,
            &catalog,
            &routing,
            &mut ledger,
        );
        match p.planned {
            Planned::Replay { output, .. } => assert!(output.ends_with("the page"), "{output}"),
            _ => panic!("the wrapper spelling must hit the ledger"),
        }

        let p = plan_call(
            "c3",
            "artifacts__artifact_read",
            args,
            None,
            &catalog,
            &routing,
            &mut ledger,
        );
        match p.planned {
            Planned::Refuse { message, .. } => assert!(message.contains("twice"), "{message}"),
            _ => panic!("the third emission is a loop"),
        }
    }

    /// Two identical calls in one round cannot replay a result that does not
    /// exist yet - the second is refused and pointed at its twin.
    #[test]
    fn a_duplicate_within_one_round_is_refused() {
        let (catalog, routing) = one_tool();
        let mut ledger = loop_budget::CallLedger::new();
        let args = r#"{"artifact_id":"a1"}"#;
        let a = plan_call(
            "c1",
            "artifacts__artifact_read",
            args,
            None,
            &catalog,
            &routing,
            &mut ledger,
        );
        let b = plan_call(
            "c2",
            "artifacts__artifact_read",
            args,
            None,
            &catalog,
            &routing,
            &mut ledger,
        );
        assert!(matches!(a.planned, Planned::Invoke { .. }));
        match b.planned {
            Planned::Refuse { message, .. } => assert!(message.contains("same round"), "{message}"),
            _ => panic!("the round's second identical call must not run too"),
        }
    }

    /// On the cloud lanes: the caller's `max_tool_calls` bounds the calls
    /// we execute. It is the one bound that costs the caller money per call,
    /// so it must bite before dispatch, not after the round.
    #[test]
    fn the_callers_tool_call_limit_bounds_the_cloud_lane() {
        let (catalog, routing) = one_tool();
        let mut ledger = loop_budget::CallLedger::with_limit(Some(1));
        let p = plan_call(
            "c1",
            "artifacts__artifact_read",
            r#"{"artifact_id":"a1"}"#,
            None,
            &catalog,
            &routing,
            &mut ledger,
        );
        assert!(matches!(p.planned, Planned::Invoke { .. }));
        ledger.record(&p.sig.expect("filed"), true, "the page");
        assert_eq!(
            ledger.limit_reached(),
            Some(loop_budget::Stop::ToolCalls(1))
        );

        let p = plan_call(
            "c2",
            "artifacts__artifact_read",
            r#"{"artifact_id":"a2"}"#,
            None,
            &catalog,
            &routing,
            &mut ledger,
        );
        match p.planned {
            Planned::Refuse { message, .. } => {
                assert!(message.contains("max_tool_calls: 1"), "{message}");
            }
            _ => panic!("past the caller's limit nothing may be dispatched"),
        }
    }

    #[test]
    fn the_builtin_clock_plans_ours_only_when_declared() {
        let (catalog, routing) = one_tool();
        let mut ledger = loop_budget::CallLedger::new();
        let spec =
            Some(clock::parse_spec(&json!({"type": "current_time"})).expect("bare spec is valid"));
        let p = plan_call(
            "c1",
            clock::TOOL_NAME,
            "{}",
            spec,
            &catalog,
            &routing,
            &mut ledger,
        );
        assert!(matches!(p.planned, Planned::Clock));
        assert!(
            p.sig.is_some(),
            "a fresh clock call files in the repeat ledger"
        );
        // Without the declaration the name is just an unknown tool - a
        // provider model calling it un-declared must not reach our clock.
        let p = plan_call(
            "c2",
            clock::TOOL_NAME,
            "{}",
            None,
            &catalog,
            &routing,
            &mut ledger,
        );
        assert!(!matches!(p.planned, Planned::Clock));
    }

    #[test]
    fn a_schema_refusal_stays_out_of_the_ledger() {
        let (catalog, routing) = one_tool();
        let mut ledger = loop_budget::CallLedger::new();
        let p = plan_call(
            "c1",
            "artifacts__artifact_read",
            "{}",
            None,
            &catalog,
            &routing,
            &mut ledger,
        );
        assert!(p.sig.is_none(), "nothing ran, so nothing is filed");
        // The corrected call still runs rather than reading as a repeat.
        let p = plan_call(
            "c2",
            "artifacts__artifact_read",
            r#"{"artifact_id":"a1"}"#,
            None,
            &catalog,
            &routing,
            &mut ledger,
        );
        assert!(matches!(p.planned, Planned::Invoke { .. }));
    }

    /// Lever 2 on the wire: a tool round is held to what the turn's budget has
    /// left, and the answer round gets the caller's number back - or the field
    /// removed, if they named none.
    #[test]
    fn the_round_ceiling_goes_on_and_comes_back_off() {
        let mut body = json!({"max_output_tokens": 64_000});
        // Plenty left: the caller's own number stands, and nothing was held back.
        assert!(!round_ceiling(
            &mut body,
            "max_output_tokens",
            false,
            Some(64_000),
            256_000
        ));
        assert_eq!(body["max_output_tokens"], 64_000);
        // Nearly spent: the round may only have the remainder, and says so.
        assert!(round_ceiling(
            &mut body,
            "max_output_tokens",
            false,
            Some(64_000),
            900
        ));
        assert_eq!(body["max_output_tokens"], 900);
        // The answer round is outside the budget: the caller's number, in full.
        assert!(!round_ceiling(
            &mut body,
            "max_output_tokens",
            true,
            Some(64_000),
            0
        ));
        assert_eq!(body["max_output_tokens"], 64_000);

        // Nothing asked for: a tool round is still bounded (cloud rounds cost
        // real money) and that bound is OURS, so a round it cuts routes to the
        // answer round instead of reading as the caller's own truncation. The
        // answer round is handed back unbounded.
        let mut body = json!({});
        assert!(round_ceiling(
            &mut body,
            "max_output_tokens",
            false,
            None,
            16_384
        ));
        assert_eq!(body["max_output_tokens"], 16_384);
        round_ceiling(&mut body, "max_output_tokens", true, None, 0);
        assert!(body.get("max_output_tokens").is_none());
    }

    /// Anthropic rejects `max_tokens <= thinking.budget_tokens`, so a thinking
    /// turn's ceiling can never dip below the budget it already promised.
    #[test]
    fn the_round_ceiling_never_undercuts_a_thinking_budget() {
        let mut body = json!({"max_tokens": 32_000, "thinking": {"budget_tokens": 10_000}});
        round_ceiling(&mut body, "max_tokens", false, Some(32_000), 500);
        assert_eq!(body["max_tokens"], 11_024);
        // ...and it still never exceeds what the caller asked for.
        let mut body = json!({"max_tokens": 6_000, "thinking": {"budget_tokens": 10_000}});
        round_ceiling(&mut body, "max_tokens", false, Some(6_000), 500);
        assert_eq!(body["max_tokens"], 6_000);
    }

    #[test]
    fn the_answer_nudge_joins_the_tool_result_turn() {
        let mut msgs = vec![json!({"role":"user","content":[{"type":"tool_result"}]})];
        append_user_text(&mut msgs, "answer now");
        assert_eq!(
            msgs.len(),
            1,
            "two user turns in a row is a shape Anthropic rejects"
        );
        assert_eq!(msgs[0]["content"][1]["text"], "answer now");
    }
}
