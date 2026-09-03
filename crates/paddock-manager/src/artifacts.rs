//! Artifacts - the manager's own first-party MCP server.
//!
//! An artifact is substantial standalone content the model wrote and the user
//! keeps: a page, a chart, a document. It lives in the Studio's side panel,
//! not in the message flow, and it is VERSIONED - "make the header blue" three
//! turns later appends v2 rather than starting over.
//!
//! ## Why this is an MCP server and not a runner built-in
//!
//! The store is the manager's SQLite (the only DB on the box) and runners are
//! stateless, so the tools have to execute where the data is. Hosting them as
//! an MCP server means both lanes reach them through machinery that already
//! exists: a local model goes runner -> `paddock-mcp` -> here, and a cloud model
//! goes `cloud_loop` -> here. No new tool-execution path anywhere, and the
//! Studio renders the calls with the `mcp_call` cards it already has.
//!
//! The runner needs zero changes for this: `paddock-mcp` already speaks
//! Streamable HTTP with custom headers, so we are just another URL.
//!
//! ## Why the Studio attaches it per request instead of configuring it
//!
//! These tools ride as an INLINE `mcp` tool on each request (the shape
//! `resolve_mcp_server` already parses), never as an entry in
//! `servers/<port>.toml`. If a runner advertised `artifact_create` to every
//! caller, a Claude Code session or a curl client would have the model call
//! it, the content would land here, and the caller would get back
//! `art_7f3a...` - a reference to something it has no panel to render. That is
//! a silent failure, and it would also make a standalone runner depend on the
//! manager being up. Inline-per-request keeps the capability exactly where the
//! surface to show it exists.
//!
//! ## The context rule
//!
//! The conversation carries the OPERATIONS; the body stays here. `update`
//! answers "replaced 4 lines with 6", never the new content - echoing it would
//! put the whole artifact back in the prompt and defeat the entire point.
//! `artifact_read` exists so the model can pull a body back deliberately when
//! it actually needs to see one again.
//!
//! ## Editing semantics
//!
//! `update` takes an exact string that must match exactly ONCE. That is the
//! convergent industry design (Claude's artifacts); OpenAI's canvas shipped
//! regex patching and then instructed the model to always full-rewrite code
//! documents, which is the falsified branch. `rewrite` is the escape hatch for
//! changes too broad to express as a replacement.

use std::sync::Arc;

use axum::http::request::Parts;
use rmcp::ErrorData;
use rmcp::handler::server::common::Extension;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{CallToolResult, ContentBlock, ServerCapabilities, ServerInfo};
use rmcp::{ServerHandler, tool, tool_handler, tool_router};
use schemars::JsonSchema;
use serde::Deserialize;

use crate::store::Store;

/// Header the Studio sets on the inline spec so a call lands in the right
/// conversation. It is not a security boundary - the manager's bearer key
/// already gates the endpoint - it is the routing key, kept out of the tool
/// arguments so the model can neither see nor forge another chat's id.
pub const CONVERSATION_HEADER: &str = "x-paddock-conversation";

/// Which model is writing. A compare turn runs several models against one
/// conversation, so without this their artifacts are indistinguishable in the
/// panel. Set by the Studio per LANE, alongside the conversation header.
pub const MODEL_HEADER: &str = "x-paddock-model";

/// The server's MCP `instructions`: folded into the system prompt by the
/// runner, and shown read-only in the Studio's system-prompt panel. Injected
/// text the user cannot see is a silent behaviour change - an empty prompt box
/// must not mean "nothing is being said on your behalf".
///
/// This TEXT LEADS the SYSTEM PROMPT, so an over-broad rule here becomes the
/// model's default posture for the whole conversation. Two live failures on
/// Qwen3.5-9B  forced the qualifiers above, and both came
/// from "whenever you are asked to produce ... a document":
///
///   - "what is this?" about a photo produced a 38-line HTML artifact of
///     analysis and a two-sentence reply. Nobody asked for a document; the
///     model read "answer at length" as "produce a document".
///   - "show me a map", with a real map surface offered in the same system
///     prompt, produced a hand-drawn SVG of coloured ellipses labelled AREZZO.
///     It never called the surface, because "put it in an artifact rather than
///     a fenced code block in your reply" is word-for-word the opposite of
///     what the map capability asks for, and the general rule beat the
///     specific request.
///
/// So the trigger is now the user's request rather than the answer's size, and
/// app-rendered fences are named as off-limits. Any future surface that asks
/// the model for a fenced block needs the same protection - this is a general
/// hazard of leading the prompt with a rule about where content goes.
pub const INSTRUCTIONS: &str = "Artifacts are pieces of content shown in a panel beside the conversation, where \
                 the user can view and iterate on them. When the user ASKS YOU TO PRODUCE an \
                 HTML page, chart, diagram, SVG, document or program, put it in an artifact \
                 rather than a fenced code block in your reply - anything over roughly 15 lines \
                 belongs there. \"Simple\" or \"small\" describes the content, not where it goes. \
                 Answering a question is not producing a document: a description, an explanation \
                 or an analysis goes in the reply, however long it runs. And never draw your own \
                 picture of something this app already renders - a fenced block the app turns \
                 into a view, a `map` block for one, is part of the reply and must not be moved \
                 into an artifact or redrawn as one. \
                 Do not repeat an artifact's content in your message; say briefly what you made. \
                 An artifact renders with NO network access, so a remote image, font or script \
                 will not appear - draw with CSS/SVG, or embed the bytes as a data: URI. Never \
                 link a placeholder image service; several are dead, and the page renders wrong \
                 rather than empty. \
                 Content is not echoed back into the chat, so read it back when you need the \
                 exact text. Earlier tool calls are NOT in the transcript, so when the user asks \
                 to change something, find its id first rather than creating a second copy. \
                 The tools, with their arguments: \
                 artifact_create(kind, title, content, language?) - kind is one of html, \
                 markdown, svg, mermaid, code, csv (rendered as a table), graph, text. A written \
                 document is markdown; html is for a page you actually wrote tags for. \
                 A GRAPH OF DATA - entities and their relationships (people, cities, \
                 companies, anything the user may want to query or explore) - is ALWAYS \
                 kind graph: an openCypher script of CREATE statements the app runs in a \
                 live, queryable graph view. This holds however casually it is asked \
                 (\"make a graph of this\"). mermaid is ONLY for diagrams of processes \
                 and flows, never for data. Put real property values in graph artifacts, \
                 not placeholders, and separate CREATE clauses with newlines - never \
                 semicolons - so node variables stay shared for the relationship clauses. \
                 Relationships connect NODE VARIABLES on both ends - (a)-[:REL]->(b) where \
                 b was CREATEd as a node - never a bare string: make the thing a node \
                 (CREATE a Portfolio node with a name property) and point at it. \
                 Every relationship is DIRECTED - always write the arrow, \
                 (a)-[:REL]->(b) or (a)<-[:REL]-(b); an arrowless -[:REL]- fails the \
                 whole import. Do not add summary or legend nodes - the graph itself \
                 is the summary. A property map matches ONE literal value per key - OR \
                 inside braces is not Cypher; to wire several nodes at once, MATCH with \
                 a WHERE filter (m.role IN ['a', 'b']) and CREATE from those variables, \
                 which also works across CREATE and MATCH clauses in the same script. \
                 One relationship per clause: (a, b, c)-[:R]->(x) is not Cypher, and every \
                 clause must start with a keyword (CREATE or MATCH) - a bare pattern line \
                 fails the whole import. Declare each node variable ONCE with its braces; \
                 every later mention is BARE - (a)-[:R]->(germany), never \
                 (a)-[:R]->(germany with braces again) - re-CREATEing a bound variable \
                 with properties fails the import. Every line copies this shape EXACTLY \
                 (variable, colon, Label, properties - the colon is required): \
                 CREATE (ana:Person {name: 'Ana', role: 'CEO'}) \
                 CREATE (acme:Company {name: 'Acme'}) \
                 CREATE (ana)-[:WORKS_AT {since: 2020}]->(acme) \
                 - a fresh variable per node, one CREATE per line, arrow on every \
                 relationship; \
                 artifact_list() - the id, title and version of everything in this conversation; \
                 artifact_read(artifact_id, version?) - the current text, or an older version; \
                 artifact_update(artifact_id, old_string, new_string) - old_string must appear \
                 exactly once; \
                 artifact_rewrite(artifact_id, content) - replace the whole body. \
                 These are listed here in full because your visible tool list may not carry \
                 their schemas: if a name is missing from it, call mcp_search_tools with \
                 \"artifact\" and invoke through mcp_call_tool.";

/// Renderable artifact kinds. Closed deliberately: the panel has to know how to
/// display the thing, and an unrecognized kind would render as nothing at all.
const KINDS: &[&str] = &[
    "html", "markdown", "svg", "mermaid", "code", "csv", "graph", "text",
];

/// Does this text contain anything a parser would read as a tag?
fn has_tag(s: &str) -> bool {
    let b = s.as_bytes();
    b.iter().enumerate().any(|(i, &c)| {
        c == b'<'
            && b.get(i + 1)
                .is_some_and(|n| n.is_ascii_alphabetic() || *n == b'/' || *n == b'!')
    })
}

/// Markdown's block markers, at the start of a line, plus bold anywhere.
fn looks_markdown(s: &str) -> bool {
    s.contains("**")
        || s.lines().any(|l| {
            let t = l.trim_start();
            ["# ", "## ", "### ", "#### ", "- ", "* ", "> ", "```", "| "]
                .iter()
                .any(|m| t.starts_with(m))
        })
}

/// The kind this content actually is, when the declared one is impossible.
///
/// asked "what is this?" about a photo, a 9B called
/// artifact_create with kind "html" and a body of pure markdown - `# Street
/// Scene in Arezzo`, `**Capture Date:**`, bullet lists. The panel did as it
/// was told and rendered it in the HTML frame, so the reader met the hashes
/// and asterisks raw. The model was not confused about what it wrote; it was
/// confused about the label, and "html" is where a model reaches by default.
///
/// Only an IMPOSSIBLE declaration is corrected: html or svg for a body with no
/// tag in it anywhere. That is not a judgement about style - an HTML document
/// containing zero tags is not an HTML document, and there is no reading under
/// which the frame renders it correctly. Anything with markup, however broken,
/// is left exactly as declared.
///
/// Corrected rather than refused, and the correction is stated in the tool
/// result. A refusal here costs a whole extra generation to fix a label while
/// the content was already right (measured one such retry at 4096
/// tokens and 75 s), and the model is told what happened, so it is not silent.
fn settle_kind(kind: &str, content: &str) -> Option<&'static str> {
    if (kind != "html" && kind != "svg") || has_tag(content) {
        return None;
    }
    Some(if looks_markdown(content) {
        "markdown"
    } else {
        "text"
    })
}

#[derive(Clone)]
pub struct Artifacts {
    db: Arc<Store>,
}

/// Every field is Option even where it is required, and the requirement is
/// checked in the handler instead - but the SCHEMA still declares it, via
/// `extend`.
///
/// rmcp deserializes arguments before the tool runs, so a missing field came
/// back as a JSON-RPC -32602 "failed to deserialize parameters: missing field
/// `content`" and none of the guidance below ever reached the model. It then
/// spent 4096 tokens flailing. A protocol error is the
/// wrong channel for "you forgot an argument": the model can act on a tool
/// result, so that is what it gets.
///
/// The Option-everywhere trick fixed the channel and quietly broke the
/// contract: schemars omits `required` for an Option, so the published schema
/// said every argument was optional. Nothing upstream could disagree - the
/// The grammar had no constraint to compile and the pre-dispatch check
/// had no rule to check, so `artifact_create(kind, title)` was a *legal* call
/// all the way to the handler. So: Option in Rust for the deserializer, `required` in the schema
/// for everyone who reads it, and the handler check as the last line.
#[derive(Debug, Deserialize, JsonSchema)]
#[schemars(extend("required" = ["kind", "title", "content"]))]
pub struct CreateArgs {
    /// One of: html, markdown, svg, mermaid, code, csv, graph, text.
    #[serde(default)]
    pub kind: Option<String>,
    /// Short human title shown on the panel tab.
    #[serde(default)]
    pub title: Option<String>,
    /// The full content. Required in the same call - an artifact cannot be
    /// created empty and filled in later.
    #[serde(default)]
    pub content: Option<String>,
    /// Source language when kind is "code" (e.g. "python", "rust").
    #[serde(default)]
    pub language: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[schemars(extend("required" = ["artifact_id", "old_string", "new_string"]))]
pub struct UpdateArgs {
    /// The artifact id returned by artifact_create.
    #[serde(default)]
    pub artifact_id: Option<String>,
    /// Exact text to replace. Must appear exactly once in the current version -
    /// include surrounding lines if the snippet would otherwise be ambiguous.
    #[serde(default)]
    pub old_string: Option<String>,
    /// Replacement text.
    #[serde(default)]
    pub new_string: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[schemars(extend("required" = ["artifact_id", "content"]))]
pub struct RewriteArgs {
    #[serde(default)]
    pub artifact_id: Option<String>,
    /// The complete new content, replacing everything.
    #[serde(default)]
    pub content: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ListArgs {}

#[derive(Debug, Deserialize, JsonSchema)]
#[schemars(extend("required" = ["artifact_id"]))]
pub struct ReadArgs {
    #[serde(default)]
    pub artifact_id: Option<String>,
    /// Version to read; omit for the current one.
    #[serde(default)]
    pub version: Option<i64>,
}

/// Tool-visible failure: comes back as tool CONTENT with is_error set, not a
/// JSON-RPC error, so the model reads it and can correct itself on the next
/// round instead of the call vanishing into a protocol fault.
fn refuse(msg: impl Into<String>) -> CallToolResult {
    CallToolResult::error(vec![ContentBlock::text(msg.into())])
}

fn ok(msg: impl Into<String>) -> CallToolResult {
    CallToolResult::success(vec![ContentBlock::text(msg.into())])
}

/// Missing-argument refusal, phrased as an instruction rather than a
/// complaint: the model has to know what to send on the retry.
fn need(tool: &str, field: &str, all: &str) -> CallToolResult {
    refuse(format!(
        "{tool} needs `{field}`, which was not in the call. Call {tool} again with {all}          together in ONE call - the arguments cannot be sent separately."
    ))
}

fn lines(s: &str) -> usize {
    if s.is_empty() {
        0
    } else {
        s.split('\n').count()
    }
}

/// A routing header off the injected HTTP parts.
fn header(parts: &Parts, name: &str) -> Option<String> {
    parts
        .headers
        .get(name)
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned)
        .filter(|s| !s.is_empty())
}

/// The conversation this call belongs to, read from the injected HTTP parts.
fn conversation(parts: &Parts) -> Option<String> {
    parts
        .headers
        .get(CONVERSATION_HEADER)
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned)
        .filter(|s| !s.is_empty())
}

#[tool_router]
impl Artifacts {
    pub fn new(db: Arc<Store>) -> Self {
        Self { db }
    }

    /// Create an artifact - the panel beside the chat - for a page, chart,
    /// diagram, document or program.
    ///
    /// The trigger is a LINE COUNT, not a judgement call. The first cut led
    /// with "substantial standalone content" and warned off "short snippets",
    /// and that fails in practice: asked for a
    /// "nicely designed simple html page", the model read "simple", matched
    /// the warning, and printed a fenced block into the chat instead. Models
    /// follow a concrete threshold far better than an adjective, so the rule
    /// is stated as one.
    #[tool(
        description = "Create an artifact - the panel beside the chat - for an HTML page, \
        chart, diagram, SVG, document or program you were asked to produce. RULE: if your reply \
        would otherwise contain a fenced code block longer than about 15 lines, make it an \
        artifact instead. Words like \"simple\", \"small\" or \"quick\" describe the CONTENT, not \
        where it goes - a simple page is still an artifact. Keep it in your reply only for a \
        brief snippet, a single command, or a fragment you are explaining line by line. Returns \
        an id for later edits."
    )]
    async fn artifact_create(
        &self,
        Parameters(p): Parameters<CreateArgs>,
        Extension(parts): Extension<Parts>,
    ) -> Result<CallToolResult, ErrorData> {
        let Some(conv) = conversation(&parts) else {
            return Ok(refuse("artifacts are unavailable in this context"));
        };
        let Some(content) = p.content.filter(|c| !c.trim().is_empty()) else {
            return Ok(need(
                "artifact_create",
                "content",
                "kind, title AND content",
            ));
        };
        let Some(kind_raw) = p.kind else {
            return Ok(need("artifact_create", "kind", "kind, title AND content"));
        };
        let title = p.title.unwrap_or_default();
        let kind = kind_raw.trim().to_ascii_lowercase();
        if !KINDS.contains(&kind.as_str()) {
            return Ok(refuse(format!(
                "unknown kind {kind_raw:?}; use one of: {}",
                KINDS.join(", ")
            )));
        }
        let language = p.language.unwrap_or_default();
        let model = header(&parts, MODEL_HEADER).unwrap_or_default();
        // Stated, never silent: the model asked for one kind and got another,
        // and it needs to know before it calls artifact_update on it.
        let (kind, corrected) = match settle_kind(&kind, &content) {
            Some(k) => (
                k.to_string(),
                format!(
                    " Your content has no markup in it, so it is stored and rendered as {k} \
                     rather than {kind}."
                ),
            ),
            None => (kind, String::new()),
        };
        match self
            .db
            .create_artifact(&conv, &kind, title.trim(), &language, &model, &content)
        {
            Ok(id) => Ok(ok(format!(
                "Created {id} ({kind}, {} lines) and opened it in the side panel.{corrected} \
                 Edit it with artifact_update; do not repeat the content in your reply.",
                lines(&content)
            ))),
            Err(e) => Ok(refuse(format!("could not store the artifact: {e}"))),
        }
    }

    /// Replace an exact string in an artifact.
    #[tool(
        description = "Replace an exact string in an existing artifact. old_string must match \
        EXACTLY ONCE - include surrounding lines to disambiguate. Prefer this over rewriting: it \
        keeps the conversation short. Use artifact_rewrite when the change is too broad."
    )]
    async fn artifact_update(
        &self,
        Parameters(p): Parameters<UpdateArgs>,
        Extension(parts): Extension<Parts>,
    ) -> Result<CallToolResult, ErrorData> {
        let Some(conv) = conversation(&parts) else {
            return Ok(refuse("artifacts are unavailable in this context"));
        };
        let (Some(artifact_id), Some(old_string), Some(new_string)) =
            (p.artifact_id, p.old_string, p.new_string)
        else {
            return Ok(need(
                "artifact_update",
                "artifact_id, old_string and new_string",
                "all three",
            ));
        };
        if let Err(why) = self.owns(&conv, &artifact_id) {
            return Ok(refuse(why));
        }
        let Ok(Some((_, current, seq))) = self.db.artifact_content(&artifact_id, None) else {
            return Ok(refuse(format!("no artifact {artifact_id}")));
        };
        if old_string.is_empty() {
            return Ok(refuse(
                "old_string is empty; use artifact_rewrite to replace everything",
            ));
        }
        let hits = current.matches(old_string.as_str()).count();
        match hits {
            0 => {
                return Ok(refuse(format!(
                    "old_string does not appear in {} (v{seq}). Call artifact_read to see the \
                     current contents before editing.",
                    artifact_id
                )));
            }
            1 => {}
            n => {
                return Ok(refuse(format!(
                    "old_string matches {n} places in {} (v{seq}); include more surrounding text \
                     so it identifies exactly one.",
                    artifact_id
                )));
            }
        }
        let next = current.replacen(old_string.as_str(), &new_string, 1);
        match self
            .db
            .append_artifact_version(&artifact_id, "update", &next)
        {
            Ok(v) if v == seq => Ok(ok(format!("{} is unchanged (v{seq}).", artifact_id))),
            Ok(v) => Ok(ok(format!(
                "Updated {} to v{v}: replaced {} lines with {}. The panel is showing it - do not \
                 repeat the content in your reply.",
                artifact_id,
                lines(&old_string),
                lines(&new_string),
            ))),
            Err(e) => Ok(refuse(format!("could not save the edit: {e}"))),
        }
    }

    /// Replace an artifact's whole body.
    #[tool(
        description = "Replace an artifact's entire content. Use only when the change is too \
        broad for artifact_update - a full rewrite costs the whole file in output tokens."
    )]
    async fn artifact_rewrite(
        &self,
        Parameters(p): Parameters<RewriteArgs>,
        Extension(parts): Extension<Parts>,
    ) -> Result<CallToolResult, ErrorData> {
        let Some(conv) = conversation(&parts) else {
            return Ok(refuse("artifacts are unavailable in this context"));
        };
        let (Some(artifact_id), Some(content)) = (p.artifact_id, p.content) else {
            return Ok(need("artifact_rewrite", "artifact_id and content", "both"));
        };
        if let Err(why) = self.owns(&conv, &artifact_id) {
            return Ok(refuse(why));
        }
        if content.trim().is_empty() {
            return Ok(refuse("content is empty - an artifact must have a body"));
        }
        match self
            .db
            .append_artifact_version(&artifact_id, "rewrite", &content)
        {
            Ok(v) => Ok(ok(format!(
                "Rewrote {} to v{v} ({} lines). The panel is showing it - do not repeat the \
                 content in your reply.",
                artifact_id,
                lines(&content)
            ))),
            Err(e) => Ok(refuse(format!("could not save the rewrite: {e}"))),
        }
    }

    /// What this conversation already has.
    ///
    /// Load-bearing, not a convenience: the client replays only MESSAGE items
    /// into the next turn, so an earlier `artifact_create` and the id it
    /// returned are gone by the time the user says "make the header blue".
    /// Without a way to ask, the model can only guess an id or create a
    /// duplicate - which is what it does in practice.
    #[tool(
        description = "List the artifacts already in this conversation (id, title, kind, \
        version count). ALWAYS call this first when the user asks to change, fix, update or \
        extend something - an earlier artifact's id is not in the transcript, and creating a \
        second copy instead of editing the first is wrong."
    )]
    async fn artifact_list(
        &self,
        Parameters(_): Parameters<ListArgs>,
        Extension(parts): Extension<Parts>,
    ) -> Result<CallToolResult, ErrorData> {
        let Some(conv) = conversation(&parts) else {
            return Ok(refuse("artifacts are unavailable in this context"));
        };
        match self.db.list_artifacts(&conv) {
            Ok(rows) if rows.is_empty() => Ok(ok(
                "No artifacts in this conversation yet - artifact_create makes the first.",
            )),
            Ok(rows) => {
                let lines: Vec<String> = rows
                    .iter()
                    .map(|a| {
                        format!(
                            "{} - {:?} ({}, v{})",
                            a["id"].as_str().unwrap_or(""),
                            a["title"].as_str().unwrap_or(""),
                            a["kind"].as_str().unwrap_or(""),
                            a["versions"].as_i64().unwrap_or(1),
                        )
                    })
                    .collect();
                Ok(ok(lines.join("\n")))
            }
            Err(e) => Ok(refuse(format!("could not list artifacts: {e}"))),
        }
    }

    /// Read an artifact back.
    #[tool(
        description = "Read an artifact's current content back into the conversation. Use this \
        when you need to see the exact text before editing - the body is not kept in the chat."
    )]
    async fn artifact_read(
        &self,
        Parameters(p): Parameters<ReadArgs>,
        Extension(parts): Extension<Parts>,
    ) -> Result<CallToolResult, ErrorData> {
        let Some(conv) = conversation(&parts) else {
            return Ok(refuse("artifacts are unavailable in this context"));
        };
        let Some(artifact_id) = p.artifact_id else {
            return Ok(need(
                "artifact_read",
                "artifact_id",
                "the id from artifact_list",
            ));
        };
        if let Err(why) = self.owns(&conv, &artifact_id) {
            return Ok(refuse(why));
        }
        match self.db.artifact_content(&artifact_id, p.version) {
            Ok(Some((_, content, seq))) => Ok(ok(format!("{} v{seq}:\n\n{content}", artifact_id))),
            Ok(None) => Ok(refuse(format!(
                "no artifact {} at that version",
                artifact_id
            ))),
            Err(e) => Ok(refuse(format!("could not read the artifact: {e}"))),
        }
    }

    /// An artifact belongs to exactly one conversation. Without this check a
    /// chat could edit another chat's artifact just by guessing an id.
    fn owns(&self, conversation_id: &str, artifact_id: &str) -> Result<(), String> {
        match self.db.artifact_conversation(artifact_id) {
            Ok(Some(c)) if c == conversation_id => Ok(()),
            // Same answer for "belongs to another chat" and "does not exist":
            // a differing message would confirm the id is real.
            _ => Err(format!("no artifact {artifact_id} in this conversation")),
        }
    }
}

// ── HTTP surface: the MCP endpoint, the Studio's REST view, the frame shell ──

/// The MCP endpoint, mounted under `/api/` so the manager's bearer auth covers
/// it. Stateless mode: every call is one request/response, so there is no
/// session to keep alive and nothing to leak between conversations.
pub fn mcp_service(
    db: Arc<Store>,
    allowed_hosts: Vec<String>,
) -> rmcp::transport::streamable_http_server::StreamableHttpService<
    Artifacts,
    rmcp::transport::streamable_http_server::session::local::LocalSessionManager,
> {
    use rmcp::transport::streamable_http_server::{
        StreamableHttpService, session::local::LocalSessionManager,
    };
    // rmcp 3.x: `stateful_mode` became `legacy_session_mode`, and it only
    // applies below MCP 2026-07-28 - that revision removed sessions outright
    // (SEP-2567), so a modern client is served statelessly either way. false
    // keeps the old behaviour for the clients still on the older revision.
    //
    // allowed_hosts is new and defaults to loopback. That check earns its keep
    // here: auth_mw exempts loopback PEERS, and a browser pointed at us by a
    // rebinding attack is one - the Host header is the only thing that tells
    // the two apart. So it is set to what this manager actually answers on,
    // never disabled.
    let mut config = rmcp::transport::streamable_http_server::StreamableHttpServerConfig::default()
        .with_allowed_hosts(allowed_hosts);
    config.legacy_session_mode = false;
    StreamableHttpService::new(
        move || Ok(Artifacts::new(db.clone())),
        Arc::new(LocalSessionManager::default()),
        config,
    )
}

/// What the composer's tool picker shows for the artifacts group. Read off the
/// live router rather than a second hand-written list, so adding a tool cannot
/// leave the picker stale - and answered in-process, since dialing our own
/// HTTP endpoint to ask ourselves what we serve would be daft.
pub fn tool_list() -> Vec<serde_json::Value> {
    Artifacts::tool_router()
        .list_all()
        .into_iter()
        .map(|t| serde_json::json!({ "name": t.name, "description": t.description }))
        .collect()
}

/// Ceiling on a hand-edited body. Artifacts are documents, not uploads, and
/// axum's 2 MiB default would silently truncate a long one into a 413.
const EDIT_MAX: usize = 4 * 1024 * 1024;

pub fn routes() -> axum::Router<Arc<crate::routes::AppState>> {
    use axum::routing::get;
    axum::Router::new()
        .route("/api/conversations/{id}/artifacts", get(list))
        .route("/api/artifacts/{id}", get(meta))
        .route(
            "/api/artifacts/{id}/content",
            get(content)
                .put(put_content)
                .layer(axum::extract::DefaultBodyLimit::max(EDIT_MAX)),
        )
        // Deliberately outside /api: an iframe cannot send an Authorization
        // header, and this shell serves no data - the body arrives by
        // postMessage from the parent, which did authenticate.
        .route("/artifact-frame", get(frame))
}

fn err500(e: impl std::fmt::Display) -> axum::response::Response {
    use axum::response::IntoResponse;
    (
        axum::http::StatusCode::INTERNAL_SERVER_ERROR,
        axum::response::Json(paddock_api::ErrorBody::new("internal_error", e.to_string())),
    )
        .into_response()
}

fn err404(msg: &str) -> axum::response::Response {
    use axum::response::IntoResponse;
    (
        axum::http::StatusCode::NOT_FOUND,
        axum::response::Json(paddock_api::ErrorBody::new("not_found", msg)),
    )
        .into_response()
}

async fn list(
    axum::extract::State(s): axum::extract::State<Arc<crate::routes::AppState>>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    match s.db.list_artifacts(&id) {
        Ok(v) => axum::response::Json(serde_json::json!({ "artifacts": v })).into_response(),
        Err(e) => err500(e),
    }
}

async fn meta(
    axum::extract::State(s): axum::extract::State<Arc<crate::routes::AppState>>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    let Ok(Some(conv)) = s.db.artifact_conversation(&id) else {
        return err404("no such artifact");
    };
    match (s.db.list_artifacts(&conv), s.db.artifact_versions(&id)) {
        (Ok(all), Ok(versions)) => {
            let Some(row) = all
                .into_iter()
                .find(|a| a.get("id").and_then(|v| v.as_str()) == Some(id.as_str()))
            else {
                return err404("no such artifact");
            };
            axum::response::Json(serde_json::json!({ "artifact": row, "versions": versions }))
                .into_response()
        }
        (Err(e), _) | (_, Err(e)) => err500(e),
    }
}

#[derive(Deserialize)]
pub struct VersionQuery {
    version: Option<i64>,
}

/// One version's raw body. Served as text/plain so a browser never renders it
/// as a document on this origin - the panel decides how to display it, and the
/// scripting path is the sandboxed frame, never here.
async fn content(
    axum::extract::State(s): axum::extract::State<Arc<crate::routes::AppState>>,
    axum::extract::Path(id): axum::extract::Path<String>,
    axum::extract::Query(q): axum::extract::Query<VersionQuery>,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    match s.db.artifact_content(&id, q.version) {
        Ok(Some((kind, body, seq))) => {
            let mut h = axum::http::HeaderMap::new();
            h.insert(
                axum::http::header::CONTENT_TYPE,
                axum::http::HeaderValue::from_static("text/plain; charset=utf-8"),
            );
            h.insert(
                axum::http::header::CACHE_CONTROL,
                axum::http::HeaderValue::from_static("no-store"),
            );
            // Kind and version travel as headers so the panel can render a
            // fetched body without a second round trip for its metadata.
            if let Ok(v) = axum::http::HeaderValue::from_str(&kind) {
                h.insert(axum::http::HeaderName::from_static("x-artifact-kind"), v);
            }
            h.insert(
                axum::http::HeaderName::from_static("x-artifact-version"),
                axum::http::HeaderValue::from(seq),
            );
            (h, body).into_response()
        }
        Ok(None) => err404("no such artifact or version"),
        Err(e) => err500(e),
    }
}

/// A person editing the artifact by hand. It appends a version like any tool
/// call does, under its own op so the history says who changed what - and the
/// model picks the edit up for free, because `artifact_read` and every `update`
/// anchor work against the latest version, which this now is.
async fn put_content(
    axum::extract::State(s): axum::extract::State<Arc<crate::routes::AppState>>,
    axum::extract::Path(id): axum::extract::Path<String>,
    body: String,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    match s.db.append_artifact_version(&id, "edit", &body) {
        Ok(seq) => axum::response::Json(serde_json::json!({ "seq": seq })).into_response(),
        // The store reports an unknown id this way; everything else is ours.
        Err(crate::store::StoreError::Bad(m)) => err404(&m),
        Err(e) => err500(e),
    }
}

/// Strict policy: the canvas may compute and draw, and may not reach the
/// network at all - `connect-src 'none'` also stops an `<img src>` beacon.
/// `frame-ancestors` and `sandbox` are HEADER-ONLY directives (a `<meta>` CSP
/// ignores both), which is the whole reason this shell is served over HTTP
/// instead of dropped into a `srcdoc`. Matches the MCP Apps host requirements.
const FRAME_CSP: &str = "default-src 'none'; \
     script-src 'unsafe-inline'; \
     style-src 'unsafe-inline'; \
     img-src data: blob:; \
     font-src data:; \
     media-src data: blob:; \
     connect-src 'none'; \
     object-src 'none'; \
     base-uri 'none'; \
     form-action 'none'; \
     frame-ancestors 'self'; \
     sandbox allow-scripts";

/// The same policy with remote PICTURES allowed, served only when the person
/// looking at the preview asks for it (`/artifact-frame?img=1`).
///
/// Models reach for placeholder image services constantly, and under the strict
/// policy that renders as a page which is simply wrong - the case that forced
/// this drew white hero text over a background that never loaded.
/// Refusing by default is still right: an `<img>` URL is an
/// exfiltration channel, and the artifact was written by a model that may have
/// read untrusted input. So this is a per-preview opt-in, and it widens exactly
/// one directive - `connect-src` stays `'none'`, so scripts still cannot talk to
/// anything, and the sandbox is unchanged.
const FRAME_CSP_IMG: &str = "default-src 'none'; \
     script-src 'unsafe-inline'; \
     style-src 'unsafe-inline'; \
     img-src data: blob: https:; \
     font-src data:; \
     media-src data: blob:; \
     connect-src 'none'; \
     object-src 'none'; \
     base-uri 'none'; \
     form-action 'none'; \
     frame-ancestors 'self'; \
     sandbox allow-scripts";

/// The shell the artifact panel points its iframe at. It holds no content: the
/// panel posts the body in after load. Sandboxed with `allow-scripts` and no
/// `allow-same-origin`, so the document runs at an opaque origin and can touch
/// nothing of ours.
const FRAME_HTML: &str = r#"<!doctype html>
<html><head><meta charset="utf-8"></head><body><script>
(() => {
  // An opaque origin makes Web Storage THROW on access, so ordinary canvas
  // code dies on its first localStorage read. Shim both with a memory map.
  const memory = () => {
    const d = new Map();
    return {
      get length() { return d.size },
      key: (i) => Array.from(d.keys())[i] ?? null,
      getItem: (k) => d.has(String(k)) ? d.get(String(k)) : null,
      setItem: (k, v) => d.set(String(k), String(v)),
      removeItem: (k) => d.delete(String(k)),
      clear: () => d.clear(),
    };
  };
  for (const name of ["localStorage", "sessionStorage"]) {
    try { void window[name]; continue } catch {}
    try { Object.defineProperty(window, name, { value: memory(), configurable: true }) } catch {}
  }
  const resize = '<scr' + 'ipt>(()=>{const p=()=>parent.postMessage(' +
    '{paddockArtifactHeight:document.documentElement.scrollHeight},"*");' +
    'new ResizeObserver(p).observe(document.documentElement);' +
    'addEventListener("load",p);p();})();</scr' + 'ipt>';
  // The Studio's own thin, tinted scrollbars, applied to a document we do not
  // otherwise style. Both properties INHERIT, so setting them on <html> reaches
  // every scroller inside the artifact, and going through CSSOM rather than
  // writing a <style> means a bad value is rejected instead of injected. The
  // panel sends the colour because only it knows the current theme.
  const skin = (thumb) => {
    const de = document.documentElement;
    if (!de || typeof thumb !== "string") return;
    de.style.setProperty("scrollbar-width", "thin");
    de.style.setProperty("scrollbar-color", thumb + " transparent");
  };
  // No silent failures, even in here. A picture that never arrives renders as
  // a page that is simply wrong - the case that forced this drew white hero
  // text over a background that never loaded, which reads as blank. Report
  // every one, and say which kind of nothing it was: refused by our policy
  // (the panel can offer to allow pictures) or fetched and failed (the address
  // is dead, as via.placeholder.com now is - and no policy fixes that).
  //
  // A CSS background announces itself through neither channel: no load event,
  // no error event, and once pictures are allowed not even a violation. So the
  // shell collects every URL the finished document references and test-loads
  // each. Listeners go on the WINDOW - document.open() below drops everything
  // registered on the document.
  const missing = { blocked: [], failed: [] };
  const seen = new Set();
  const report = () => parent.postMessage({ paddockArtifactMissing: missing }, "*");
  addEventListener("securitypolicyviolation", (ev) => {
    const u = String(ev.blockedURI || "").slice(0, 300);
    if (!u || u === "inline" || seen.has(u)) return;
    seen.add(u); missing.blocked.push(u); report();
  }, true);
  const probe = () => {
    const urls = new Set();
    for (const el of Array.from(document.querySelectorAll("*")).slice(0, 600)) {
      let bg = "";
      try { bg = getComputedStyle(el).backgroundImage || "" } catch {}
      for (const m of bg.matchAll(/url\((['"]?)([^'")]+)\1\)/g)) urls.add(m[2]);
      const src = el.currentSrc || el.src || "";
      if (src) urls.add(String(src));
    }
    for (const u of urls) {
      if (!/^https?:/i.test(u) || seen.has(u)) continue;
      const im = new Image();
      // A refused fetch fires both a violation and an error; let the violation
      // land first so it reads as refused rather than as broken.
      im.onerror = () => setTimeout(() => {
        if (seen.has(u)) return;
        seen.add(u); missing.failed.push(String(u).slice(0, 300)); report();
      }, 0);
      im.src = u;
    }
  };
  addEventListener("message", (e) => {
    const d = e.data;
    if (!d || d.type !== "paddock:artifact" || typeof d.html !== "string") return;
    // document.write() on PURPOSE, and Chrome will log a [Violation] for it.
    // That advisory is about write() during the INITIAL PARSE, where it blocks
    // the parser and stalls load; this runs from a message handler long after
    // load, so the harm it warns about does not apply.
    //
    // Do not "fix" it. The alternatives cannot do this job: innerHTML and
    // DOMParser do not EXECUTE scripts, and both the appended `resize` snippet
    // and the artifact's own scripts (mermaid and friends) have to run. srcdoc
    // is out for a different reason - see ArtifactPane.vue: the frame is served
    // from a real URL precisely so frame-ancestors and sandbox apply to it.
    document.open();
    document.write(d.html + resize);
    document.close();
    // After close(): document.write replaced the element we just styled.
    skin(d.scrollbar);
    // Let styles apply before asking what the page references.
    setTimeout(probe, 50);
  });
})();
</script></body></html>"#;

#[derive(Deserialize)]
pub struct FrameQuery {
    /// `1` = the viewer asked to let remote pictures through, this once.
    img: Option<String>,
}

async fn frame(
    axum::extract::Query(q): axum::extract::Query<FrameQuery>,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    let csp = if q.img.as_deref() == Some("1") {
        FRAME_CSP_IMG
    } else {
        FRAME_CSP
    };
    (
        [
            (axum::http::header::CONTENT_TYPE, "text/html; charset=utf-8"),
            (axum::http::header::CACHE_CONTROL, "no-store"),
            (axum::http::header::CONTENT_SECURITY_POLICY, csp),
            (axum::http::header::REFERRER_POLICY, "no-referrer"),
            (axum::http::header::X_CONTENT_TYPE_OPTIONS, "nosniff"),
        ],
        FRAME_HTML,
    )
        .into_response()
}

#[tool_handler]
impl ServerHandler for Artifacts {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_instructions(INSTRUCTIONS)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::routes::{AppState, router};
    use http_body_util::BodyExt;
    use serde_json::{Value, json};
    use tower::ServiceExt;

    /// One JSON-RPC call against the mounted MCP endpoint. Streamable HTTP
    /// wants both mime types in Accept, and the conversation rides the header
    /// the Studio sets on the inline spec.
    ///
    /// The Host header is not decoration: rmcp 3.x validates it against an
    /// allow-list to stop DNS rebinding, and a request carrying neither Host
    /// nor an HTTP/2 `:authority` is a 400. `oneshot` with a path-only URI
    /// produces exactly that, where a real client never would - so the test
    /// harness has to supply what the wire always carries.
    async fn mcp(state: &Arc<AppState>, conversation: &str, body: Value) -> Value {
        let res = router(state.clone())
            .oneshot(
                axum::http::Request::post("/api/mcp/artifacts")
                    .header("content-type", "application/json")
                    .header("accept", "application/json, text/event-stream")
                    .header("host", "127.0.0.1")
                    .header(CONVERSATION_HEADER, conversation)
                    .body(axum::body::Body::from(body.to_string()))
                    .expect("request"),
            )
            .await
            .expect("response");
        assert!(
            res.status().is_success(),
            "mcp call failed: {:?}",
            res.status()
        );
        let bytes = res.into_body().collect().await.expect("body").to_bytes();
        let text = String::from_utf8_lossy(&bytes).to_string();
        // Stateless mode may answer as a one-shot SSE frame; take the payload.
        let payload = text
            .lines()
            .find_map(|l| l.strip_prefix("data: "))
            .unwrap_or(text.trim())
            .to_string();
        serde_json::from_str(&payload).unwrap_or_else(|e| panic!("not json ({e}): {text}"))
    }

    async fn call(state: &Arc<AppState>, conversation: &str, name: &str, args: Value) -> String {
        let out = mcp(
            state,
            conversation,
            json!({"jsonrpc":"2.0","id":1,"method":"tools/call",
                   "params":{"name":name,"arguments":args}}),
        )
        .await;
        let content = &out["result"]["content"][0]["text"];
        content
            .as_str()
            .unwrap_or_else(|| panic!("no text content in {out}"))
            .to_string()
    }

    fn artifact_id(reply: &str) -> String {
        reply
            .split_whitespace()
            .find(|w| w.starts_with("art_"))
            .expect("id in reply")
            .to_string()
    }

    #[tokio::test]
    async fn tools_list_advertises_the_four_tools() {
        let state = Arc::new(AppState::for_tests());
        let out = mcp(
            &state,
            "c1",
            json!({"jsonrpc":"2.0","id":1,"method":"tools/list"}),
        )
        .await;
        let names: Vec<&str> = out["result"]["tools"]
            .as_array()
            .expect("tools array")
            .iter()
            .filter_map(|t| t["name"].as_str())
            .collect();
        for want in [
            "artifact_create",
            "artifact_update",
            "artifact_rewrite",
            "artifact_read",
            "artifact_list",
        ] {
            assert!(names.contains(&want), "{want} missing from {names:?}");
        }
    }

    /// The picker's listing and the served tools are one source, so adding a
    /// tool can never leave the composer showing a stale set.
    #[tokio::test]
    async fn picker_listing_matches_what_is_served() {
        let state = Arc::new(AppState::for_tests());
        let served: Vec<String> = mcp(
            &state,
            "c1",
            json!({"jsonrpc":"2.0","id":1,"method":"tools/list"}),
        )
        .await["result"]["tools"]
            .as_array()
            .expect("tools")
            .iter()
            .filter_map(|t| t["name"].as_str().map(str::to_owned))
            .collect();

        let res = router(state)
            .oneshot(
                axum::http::Request::post("/api/mcp/tools")
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from(
                        json!({"builtin":"artifacts"}).to_string(),
                    ))
                    .expect("request"),
            )
            .await
            .expect("response");
        let bytes = res.into_body().collect().await.expect("body").to_bytes();
        let doc: Value = serde_json::from_slice(&bytes).expect("json");
        let picker: Vec<String> = doc["tools"]
            .as_array()
            .expect("tools")
            .iter()
            .filter_map(|t| t["name"].as_str().map(str::to_owned))
            .collect();
        assert_eq!(picker, served);
        assert!(
            doc["tools"][0]["description"].is_string(),
            "the picker needs descriptions"
        );
    }

    /// The flail this guards: the model calls artifact_create with kind+title
    /// and no content. rmcp rejected it at deserialization (-32602), so none of
    /// the guidance reached the model and it burned 4096 tokens guessing. A
    /// missing argument must arrive as a tool RESULT it can act on.
    #[tokio::test]
    async fn a_missing_required_argument_is_a_readable_tool_error() {
        let state = Arc::new(AppState::for_tests());
        let out = mcp(
            &state,
            "c1",
            json!({"jsonrpc":"2.0","id":1,"method":"tools/call",
                   "params":{"name":"artifact_create",
                             "arguments":{"kind":"html","title":"Simple Hero Section Example"}}}),
        )
        .await;
        assert!(
            out.get("error").is_none(),
            "must NOT be a protocol error: {out}"
        );
        let res = &out["result"];
        assert_eq!(res["isError"], true, "should be a tool error: {res}");
        let text = res["content"][0]["text"].as_str().expect("text");
        assert!(
            text.contains("content"),
            "must name the missing field: {text}"
        );
        assert!(
            text.contains("ONE call"),
            "must say to resend everything together: {text}"
        );
    }

    #[tokio::test]
    async fn missing_arguments_on_the_edit_tools_are_readable_too() {
        let state = Arc::new(AppState::for_tests());
        for (tool, args) in [
            ("artifact_update", json!({"artifact_id":"art_x"})),
            ("artifact_rewrite", json!({"artifact_id":"art_x"})),
            ("artifact_read", json!({})),
        ] {
            let out = mcp(
                &state,
                "c1",
                json!({"jsonrpc":"2.0","id":1,"method":"tools/call",
                       "params":{"name":tool,"arguments":args}}),
            )
            .await;
            assert!(
                out.get("error").is_none(),
                "{tool} became a protocol error: {out}"
            );
            assert_eq!(out["result"]["isError"], true, "{tool}: {out}");
        }
    }

    #[tokio::test]
    async fn create_then_update_appends_a_version() {
        let state = Arc::new(AppState::for_tests());
        let id = artifact_id(
            &call(
                &state,
                "c1",
                "artifact_create",
                json!({"kind":"html","title":"Chart","content":"<h1>Old</h1>\n<p>body</p>"}),
            )
            .await,
        );
        let reply = call(
            &state,
            "c1",
            "artifact_update",
            json!({"artifact_id": id, "old_string":"<h1>Old</h1>", "new_string":"<h1>New</h1>"}),
        )
        .await;
        assert!(reply.contains("v2"), "{reply}");
        let (_, body, seq) = state
            .db
            .artifact_content(&id, None)
            .expect("read")
            .expect("exists");
        assert_eq!(seq, 2);
        assert!(body.contains("<h1>New</h1>"));
        assert!(
            body.contains("<p>body</p>"),
            "the untouched part must survive"
        );
        // v1 is still there - that is what the version selector shows.
        let (_, v1, _) = state
            .db
            .artifact_content(&id, Some(1))
            .expect("read")
            .expect("exists");
        assert!(v1.contains("<h1>Old</h1>"));
    }

    /// The whole point of the design: an edit must not put the artifact body
    /// back into the conversation. If this ever regresses, context blows up
    /// silently and the feature stops paying for itself.
    #[tokio::test]
    async fn update_never_echoes_the_content() {
        let state = Arc::new(AppState::for_tests());
        let id = artifact_id(
            &call(
                &state,
                "c1",
                "artifact_create",
                json!({"kind":"html","title":"T","content":"<h1>SECRET_MARKER</h1>"}),
            )
            .await,
        );
        let reply = call(
            &state,
            "c1",
            "artifact_update",
            json!({"artifact_id": id, "old_string":"SECRET_MARKER", "new_string":"OTHER_MARKER"}),
        )
        .await;
        assert!(
            !reply.contains("OTHER_MARKER"),
            "update echoed the new body: {reply}"
        );
        assert!(
            !reply.contains("SECRET_MARKER"),
            "update echoed the old body: {reply}"
        );
    }

    #[tokio::test]
    async fn update_refuses_zero_and_multiple_matches() {
        let state = Arc::new(AppState::for_tests());
        let id = artifact_id(
            &call(
                &state,
                "c1",
                "artifact_create",
                json!({"kind":"text","title":"T","content":"a\nrow\nrow\n"}),
            )
            .await,
        );
        let none = call(
            &state,
            "c1",
            "artifact_update",
            json!({"artifact_id": id, "old_string":"absent", "new_string":"x"}),
        )
        .await;
        assert!(none.contains("does not appear"), "{none}");
        assert!(
            none.contains("artifact_read"),
            "the refusal should say how to recover: {none}"
        );
        let many = call(
            &state,
            "c1",
            "artifact_update",
            json!({"artifact_id": id, "old_string":"row", "new_string":"x"}),
        )
        .await;
        assert!(many.contains("matches 2"), "{many}");
        // Neither refusal may have written anything.
        let (_, _, seq) = state
            .db
            .artifact_content(&id, None)
            .expect("read")
            .expect("exists");
        assert_eq!(seq, 1);
    }

    #[tokio::test]
    async fn another_conversation_cannot_touch_it() {
        let state = Arc::new(AppState::for_tests());
        let id = artifact_id(
            &call(
                &state,
                "c1",
                "artifact_create",
                json!({"kind":"text","title":"T","content":"mine"}),
            )
            .await,
        );
        for tool in ["artifact_read", "artifact_update", "artifact_rewrite"] {
            let args = match tool {
                "artifact_read" => json!({"artifact_id": id}),
                "artifact_update" => {
                    json!({"artifact_id": id, "old_string":"mine", "new_string":"theirs"})
                }
                _ => json!({"artifact_id": id, "content":"theirs"}),
            };
            let reply = call(&state, "c2", tool, args).await;
            assert!(
                reply.contains("no artifact"),
                "{tool} leaked across chats: {reply}"
            );
        }
        let (_, body, _) = state
            .db
            .artifact_content(&id, None)
            .expect("read")
            .expect("exists");
        assert_eq!(body, "mine");
    }

    #[tokio::test]
    async fn identical_rewrite_does_not_manufacture_a_version() {
        let state = Arc::new(AppState::for_tests());
        let id = artifact_id(
            &call(
                &state,
                "c1",
                "artifact_create",
                json!({"kind":"text","title":"T","content":"same"}),
            )
            .await,
        );
        call(
            &state,
            "c1",
            "artifact_rewrite",
            json!({"artifact_id": id, "content":"same"}),
        )
        .await;
        let (_, _, seq) = state
            .db
            .artifact_content(&id, None)
            .expect("read")
            .expect("exists");
        assert_eq!(seq, 1);
    }

    /// A kind the content cannot possibly be is corrected, and the model is
    /// told. The case that forced this: kind "html", body of pure markdown,
    /// rendered in the HTML frame so the reader met `#` and `**` raw.
    #[tokio::test]
    async fn markdown_declared_as_html_is_stored_as_markdown() {
        let state = Arc::new(AppState::for_tests());
        let md = "# Street Scene in Arezzo\n**Capture Date: October 22, 2008**\n\n- Coordinates\n";
        let reply = call(
            &state,
            "c1",
            "artifact_create",
            json!({"kind":"html","title":"T","content":md}),
        )
        .await;
        assert!(reply.contains("(markdown,"), "stored as markdown: {reply}");
        assert!(
            reply.contains("rendered as markdown rather than html"),
            "and says so: {reply}"
        );

        // Markup - however broken - is left exactly as declared. The model's
        // own judgement about its HTML is not ours to second-guess; only an
        // impossible label is.
        let reply = call(
            &state,
            "c1",
            "artifact_create",
            json!({"kind":"html","title":"T","content":"<!-- x\n--><svg><circle/></svg>"}),
        )
        .await;
        assert!(reply.contains("(html,"), "{reply}");
        assert!(
            !reply.contains("rather than"),
            "no correction volunteered: {reply}"
        );

        // No markup and no markdown markers either: plain text, not markdown.
        let reply = call(
            &state,
            "c1",
            "artifact_create",
            json!({"kind":"html","title":"T","content":"just one sentence"}),
        )
        .await;
        assert!(reply.contains("(text,"), "{reply}");
    }

    #[tokio::test]
    async fn unknown_kind_is_refused_with_the_list() {
        let state = Arc::new(AppState::for_tests());
        let reply = call(
            &state,
            "c1",
            "artifact_create",
            json!({"kind":"powerpoint","title":"T","content":"x"}),
        )
        .await;
        assert!(reply.contains("unknown kind"), "{reply}");
        assert!(
            reply.contains("html"),
            "the refusal should list what IS allowed: {reply}"
        );
    }

    /// Everything above drives the router in-process, which never exercises
    /// the HTTP transport or our own MCP client - the exact seam a runner uses.
    /// This one serves the real router on a real socket and drives it with
    /// `paddock-mcp`, the same client `resolve_mcp_server` builds.
    #[tokio::test]
    async fn our_own_mcp_client_can_drive_it_over_real_http() {
        let state = Arc::new(AppState::for_tests());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind");
        let port = listener.local_addr().expect("addr").port();
        let app = router(state.clone());
        let server = tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });

        let cfg = paddock_mcp::ServerConfig {
            id: "test:artifacts".into(),
            label: "artifacts".into(),
            transport: paddock_mcp::Transport::Http {
                url: format!("http://127.0.0.1:{port}/api/mcp/artifacts"),
                headers: std::collections::HashMap::from([(
                    CONVERSATION_HEADER.to_string(),
                    "conv-http".to_string(),
                )]),
            },
        };
        let client = paddock_mcp::McpClient::connect(&cfg)
            .await
            .expect("connect");

        let tools = client.list_tools().await.expect("list");
        let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
        assert!(names.contains(&"artifact_create"), "{names:?}");

        let out = client
            .call_tool(
                "artifact_create",
                serde_json::json!({"kind":"html","title":"Live","content":"<h1>one</h1>"}),
            )
            .await
            .expect("create");
        assert!(!out.is_error, "{out:?}");
        let text = out.content[0]["text"].as_str().expect("text").to_string();
        let id = artifact_id(&text);

        let out = client
            .call_tool(
                "artifact_update",
                serde_json::json!({"artifact_id": id, "old_string":"one", "new_string":"two"}),
            )
            .await
            .expect("update");
        assert!(!out.is_error, "{out:?}");

        // A refusal must arrive as an error RESULT the model can read, not as a
        // transport failure that would blow up the round.
        let out = client
            .call_tool(
                "artifact_update",
                serde_json::json!({"artifact_id": id, "old_string":"nope", "new_string":"x"}),
            )
            .await
            .expect("refusal is still a successful call");
        assert!(
            out.is_error,
            "a missed match must be flagged is_error: {out:?}"
        );

        let (_, body, seq) = state
            .db
            .artifact_content(&id, None)
            .expect("read")
            .expect("exists");
        assert_eq!(seq, 2);
        assert_eq!(body, "<h1>two</h1>");

        // The 75-second failure, checked against the schema this server really
        // publishes: artifact_create through the mcp_call_tool
        // envelope with no `content`. The agent loops run every call through
        // resolve_call, so this must be refused before dispatch, by name - the
        // assertion is as much about `content` still being required here as
        // about the validator.
        use paddock_mcp::tool_search::{CALL_TOOL, CatalogTool, Resolved, resolve_call};
        let catalog: Vec<CatalogTool> = tools
            .iter()
            .map(|t| CatalogTool {
                name: format!("artifacts__{}", t.name),
                description: t.description.clone().unwrap_or_default(),
                input_schema: t.input_schema.clone(),
            })
            .collect();
        let wrapped = serde_json::json!({
            "name": "artifacts__artifact_create",
            "arguments_json": r#"{"kind":"html","title":"Hero"}"#,
        })
        .to_string();
        let Resolved::Refuse { message, .. } = resolve_call(CALL_TOOL, &wrapped, &catalog) else {
            panic!("a create with no content must not reach the store")
        };
        assert!(message.contains("`content`"), "{message}");

        // ...and the same envelope with content goes straight through.
        let ok = serde_json::json!({
            "name": "artifacts__artifact_create",
            "arguments_json": r#"{"kind":"html","title":"Hero","content":"<h1>x</h1>"}"#,
        })
        .to_string();
        assert!(matches!(
            resolve_call(CALL_TOOL, &ok, &catalog),
            Resolved::Call { .. }
        ));
        server.abort();
    }

    /// The frame is reachable without a bearer key (an iframe cannot send one)
    /// and carries the header-only directives a <meta> CSP cannot express.
    #[tokio::test]
    async fn frame_is_unauthenticated_and_locked_down() {
        let mut state = AppState::for_tests();
        state.auth_key = Some("secret".into());
        let res = router(Arc::new(state))
            .oneshot(
                axum::http::Request::get("/artifact-frame")
                    .body(axum::body::Body::empty())
                    .expect("request"),
            )
            .await
            .expect("response");
        assert_eq!(res.status(), axum::http::StatusCode::OK);
        let csp = res
            .headers()
            .get(axum::http::header::CONTENT_SECURITY_POLICY)
            .and_then(|v| v.to_str().ok())
            .expect("a CSP header, not a meta tag")
            .to_string();
        assert!(csp.contains("frame-ancestors 'self'"), "{csp}");
        assert!(csp.contains("sandbox allow-scripts"), "{csp}");
        assert!(
            csp.contains("connect-src 'none'"),
            "the canvas must not reach the network: {csp}"
        );
        assert!(
            csp.contains("img-src data: blob:;"),
            "remote pictures are opt-in: {csp}"
        );
        let bytes = res.into_body().collect().await.expect("body").to_bytes();
        let html = String::from_utf8_lossy(&bytes);
        assert!(
            html.contains("paddock:artifact"),
            "shell must listen for the payload"
        );
        // The CSP means a remote image simply never arrives, which renders as a
        // wrong page rather than an error. The shell has to report the refusal
        // or the panel has nothing to tell the user.
        assert!(
            html.contains("securitypolicyviolation") && html.contains("paddockArtifactMissing"),
            "shell must report missing resources to the panel"
        );
    }

    /// A hand edit from the panel is a version like any other, under its own
    /// op - and because it becomes the LATEST, the model reads it back without
    /// anyone telling it that a person was here.
    #[tokio::test]
    async fn a_person_can_edit_the_body_and_the_model_reads_it_back() {
        let state = Arc::new(AppState::for_tests());
        let id = state
            .db
            .create_artifact("conv-edit", "html", "T", "html", "m", "<h1>one</h1>")
            .expect("create");

        let res = router(state.clone())
            .oneshot(
                axum::http::Request::put(format!("/api/artifacts/{id}/content"))
                    .header("content-type", "text/plain; charset=utf-8")
                    .body(axum::body::Body::from("<h1>edited</h1>"))
                    .expect("request"),
            )
            .await
            .expect("response");
        assert_eq!(res.status(), axum::http::StatusCode::OK);

        let (_, body, seq) = state
            .db
            .artifact_content(&id, None)
            .expect("read")
            .expect("exists");
        assert_eq!(seq, 2, "the edit is a new version, not an overwrite");
        assert_eq!(body, "<h1>edited</h1>");
        let versions = state.db.artifact_versions(&id).expect("versions");
        assert_eq!(
            versions[1]["op"], "edit",
            "history must say a person did this"
        );
        // v1 is still readable, which is what makes the version picker honest.
        let (_, first, _) = state
            .db
            .artifact_content(&id, Some(1))
            .expect("read")
            .expect("v1");
        assert_eq!(first, "<h1>one</h1>");

        // An unknown id is a 404, not a 500 - the store reports it as Bad.
        let res = router(state)
            .oneshot(
                axum::http::Request::put("/api/artifacts/art_nope/content")
                    .body(axum::body::Body::from("x"))
                    .expect("request"),
            )
            .await
            .expect("response");
        assert_eq!(res.status(), axum::http::StatusCode::NOT_FOUND);
    }

    /// Asking for pictures widens one directive and nothing else - in
    /// particular the frame still cannot talk to anything.
    #[tokio::test]
    async fn opting_into_pictures_widens_only_img_src() {
        let res = router(Arc::new(AppState::for_tests()))
            .oneshot(
                axum::http::Request::get("/artifact-frame?img=1")
                    .body(axum::body::Body::empty())
                    .expect("request"),
            )
            .await
            .expect("response");
        let csp = res
            .headers()
            .get(axum::http::header::CONTENT_SECURITY_POLICY)
            .and_then(|v| v.to_str().ok())
            .expect("a CSP header")
            .to_string();
        assert!(csp.contains("img-src data: blob: https:;"), "{csp}");
        assert!(
            csp.contains("connect-src 'none'"),
            "scripts still get no network: {csp}"
        );
        assert!(csp.contains("sandbox allow-scripts"), "{csp}");
        assert!(csp.contains("frame-ancestors 'self'"), "{csp}");
        // Anything other than the exact opt-in stays strict - a stray query
        // string must not be a way in.
        let res = router(Arc::new(AppState::for_tests()))
            .oneshot(
                axum::http::Request::get("/artifact-frame?img=yes")
                    .body(axum::body::Body::empty())
                    .expect("request"),
            )
            .await
            .expect("response");
        let csp = res
            .headers()
            .get(axum::http::header::CONTENT_SECURITY_POLICY)
            .and_then(|v| v.to_str().ok())
            .expect("a CSP header")
            .to_string();
        assert!(csp.contains("img-src data: blob:;"), "{csp}");
    }

    /// A Host we do not answer to is refused outright (rmcp 3.x DNS-rebinding
    /// guard). This is the case that matters: `auth_mw` exempts loopback PEERS,
    /// so a browser driven at 127.0.0.1 by a malicious page passes the auth
    /// check, and the Host header it is forced to send is the only thing that
    /// gives it away.
    #[tokio::test]
    async fn a_foreign_host_header_is_refused() {
        let state = Arc::new(AppState::for_tests());
        let res = router(state)
            .oneshot(
                axum::http::Request::post("/api/mcp/artifacts")
                    .header("content-type", "application/json")
                    .header("accept", "application/json, text/event-stream")
                    .header("host", "attacker.example.com")
                    .header(CONVERSATION_HEADER, "c1")
                    .body(axum::body::Body::from(
                        json!({"jsonrpc":"2.0","id":1,"method":"tools/list"}).to_string(),
                    ))
                    .expect("request"),
            )
            .await
            .expect("response");
        assert_eq!(res.status(), axum::http::StatusCode::FORBIDDEN);
    }

    /// The MCP endpoint is under /api, so the bearer key gates it.
    #[tokio::test]
    async fn mcp_endpoint_requires_the_key() {
        let mut state = AppState::for_tests();
        state.auth_key = Some("secret".into());
        let res = router(Arc::new(state))
            .oneshot(
                axum::http::Request::post("/api/mcp/artifacts")
                    .header("content-type", "application/json")
                    .header("accept", "application/json, text/event-stream")
                    // otherwise a missing-Host 400 would satisfy the assertion
                    // and this would pass with the key check ripped out
                    .header("host", "127.0.0.1")
                    .body(axum::body::Body::from(
                        json!({"jsonrpc":"2.0","id":1,"method":"tools/list"}).to_string(),
                    ))
                    .expect("request"),
            )
            .await
            .expect("response");
        assert_eq!(res.status(), axum::http::StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn rest_lists_and_serves_a_version() {
        let state = Arc::new(AppState::for_tests());
        let id = artifact_id(
            &call(
                &state,
                "c1",
                "artifact_create",
                json!({"kind":"html","title":"Chart","content":"<h1>v1</h1>"}),
            )
            .await,
        );
        call(
            &state,
            "c1",
            "artifact_update",
            json!({"artifact_id": id, "old_string":"v1", "new_string":"v2"}),
        )
        .await;

        let res = router(state.clone())
            .oneshot(
                axum::http::Request::get("/api/conversations/c1/artifacts")
                    .body(axum::body::Body::empty())
                    .expect("request"),
            )
            .await
            .expect("response");
        let bytes = res.into_body().collect().await.expect("body").to_bytes();
        let listing: Value = serde_json::from_slice(&bytes).expect("json");
        assert_eq!(listing["artifacts"][0]["id"], id.as_str());
        assert_eq!(listing["artifacts"][0]["versions"], 2);

        // An explicit version still serves the older body.
        let res = router(state.clone())
            .oneshot(
                axum::http::Request::get(format!("/api/artifacts/{id}/content?version=1"))
                    .body(axum::body::Body::empty())
                    .expect("request"),
            )
            .await
            .expect("response");
        assert_eq!(res.headers()["x-artifact-kind"], "html");
        assert_eq!(res.headers()["x-artifact-version"], "1");
        let bytes = res.into_body().collect().await.expect("body").to_bytes();
        assert_eq!(String::from_utf8_lossy(&bytes), "<h1>v1</h1>");
    }
}
