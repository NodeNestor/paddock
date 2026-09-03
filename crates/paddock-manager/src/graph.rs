//! Graph bridge - `graph_query` against a conversation's attached graph
//! (phase 2).
//!
//! ## Why the manager only FORWARDS here (plan D2/D4)
//!
//! The only Traverse engine anywhere in the product is the WASM build running
//! in the Studio tab's Web Worker. This module holds no engine and parses no
//! Cypher: a tool call arrives from the runner (inline MCP, same lane as
//! artifacts), gets forwarded over the conversation's WebSocket to the tab,
//! and whatever text the tab answers is the tool result. The tab owns the
//! read-only gate too (it classifies via EXPLAIN before executing - EXPLAIN
//! returns the executor's query_type without running the
//! statement), because policy belongs where the engine is.
//!
//! Consequence, stated rather than hidden: the capability exists only while
//! the Studio has the conversation open. That is the same scoping the
//! artifacts server chose (inline per request - "keep the capability exactly
//! where the surface to show it exists"), and both failure modes name
//! themselves: no session registered, and a session that stopped answering.
//!
//! ## Wire shape
//!
//! Tab -> manager: connect `GET /api/graph/bridge?conversation=<id>` (bearer
//! auth rides the /api cookie/loopback path like every Studio socket), then
//! answer each `{"id":n,"cypher":"..."}` frame with `{"id":n,"body":"..."}` -
//! `body` is the finished tool-result text (compact JSON for results, a plain
//! sentence for refusals). One tab per conversation: a reconnect REPLACES the
//! old registration, and the old socket task ends on its next poll.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::{Query, State};
use axum::http::request::Parts;
use axum::response::Response;
use rmcp::ErrorData;
use rmcp::handler::server::common::Extension;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{CallToolResult, ContentBlock, ServerCapabilities, ServerInfo};
use rmcp::{ServerHandler, tool, tool_handler, tool_router};
use schemars::JsonSchema;
use serde::Deserialize;

use crate::artifacts::{CONVERSATION_HEADER, MODEL_HEADER};

/// Folded into the system prompt by the runner when the Studio attaches this
/// server (only for conversations that have a graph - the Studio decides).
/// The dynamic half - the graph's actual schema and counts - travels in the
/// request `instructions`, built tab-side where the schema lives.
pub const INSTRUCTIONS: &str = "A graph database is attached to this conversation. \
    graph_query(cypher) runs ONE openCypher statement against it and returns columns, \
    the first rows, and total_rows. The graph is read-only for you: MATCH, RETURN and \
    aggregations work; CREATE, MERGE, SET, DELETE and schema statements are refused. \
    When a query matches many rows, narrow with WHERE or aggregate (count, collect) \
    rather than paging through entities. Pattern syntax, exactly: \
    MATCH (a:Label)-[:REL_TYPE]->(b:Label) RETURN a.prop, count(b) - one dash on each \
    side of the brackets. If a query fails twice, stop retrying: report the error to \
    the user and answer with what you have. When a query identifies specific nodes \
    or relationships, RETURN the elements themselves (RETURN a, t, b): their cells \
    include their properties, and returned elements are highlighted on the user's \
    graph view. ORDER BY and WHERE work on element properties (ORDER BY t.amount).     The schema in your instructions lists every node and relationship property - use     it instead of exploring, and keep any exploratory LIMIT at 5. Answer graph     questions in your reply - do not create artifacts to restate query results.";

/// How long a forwarded query may wait for the tab before the tool call gives
/// up. Covers the tab's own 15 s execution deadline plus transit; a closed
/// laptop lid should fail the CALL, not wedge the runner's tool loop.
const ANSWER_TIMEOUT: Duration = Duration::from_secs(20);

/// One tool call in flight to the tab.
struct Job {
    cypher: String,
    /// Which model asked - a compare turn runs several lanes against one
    /// conversation and one bridge, and without this their queries land in
    /// the pane's history indistinguishable ('' when the header is absent).
    model: String,
    reply: tokio::sync::oneshot::Sender<String>,
}

/// Conversation -> the live tab session's job queue.
///
/// Registration is a plain map swap: the last tab to connect for a
/// conversation wins (a refresh must take over from its dead predecessor,
/// which may never have sent a Close frame). The replaced task notices only
/// when it next polls its queue and finds it closed.
pub struct Bridge {
    sessions: Mutex<HashMap<String, tokio::sync::mpsc::Sender<Job>>>,
}

impl Default for Bridge {
    fn default() -> Self {
        Self::new()
    }
}

impl Bridge {
    pub fn new() -> Self {
        Self {
            sessions: Mutex::new(HashMap::new()),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<String, tokio::sync::mpsc::Sender<Job>>> {
        self.sessions.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Register a tab for `conversation`, replacing any prior registration.
    /// Returns the queue the socket task serves.
    fn attach(&self, conversation: &str) -> tokio::sync::mpsc::Receiver<Job> {
        // Small buffer deliberately: queries are serialized by the worker
        // anyway, and a runner retry storm should backpressure here rather
        // than pile up frames for a tab that is not answering.
        let (tx, rx) = tokio::sync::mpsc::channel(4);
        self.lock().insert(conversation.to_string(), tx);
        rx
    }

    /// Remove a registration, but only if it is still OURS - a replacement
    /// must not be torn down by the task it replaced.
    fn detach(&self, conversation: &str, mine: &tokio::sync::mpsc::Sender<Job>) {
        let mut map = self.lock();
        if map
            .get(conversation)
            .is_some_and(|cur| cur.same_channel(mine))
        {
            map.remove(conversation);
        }
    }

    fn sender(&self, conversation: &str) -> Option<tokio::sync::mpsc::Sender<Job>> {
        self.lock().get(conversation).cloned()
    }

    /// Forward one query to the conversation's tab and wait for its answer.
    /// Every error is a sentence the model can act on.
    pub async fn query(
        &self,
        conversation: &str,
        cypher: String,
        model: String,
    ) -> Result<String, String> {
        let Some(tx) = self.sender(conversation) else {
            return Err(
                "No graph is available: the Studio tab holding this conversation's \
                        graph is not open. Tell the user to open the graph panel."
                    .into(),
            );
        };
        let (reply, rx) = tokio::sync::oneshot::channel();
        // The send must be deadlined too: with the 4-slot buffer full (a tab
        // whose worker wedged with queries pending), an un-deadlined send
        // blocks FOREVER - which stalls the runner's tool loop and shows the
        // user a thinking spinner that never ends. Every hop in this chain
        // has a clock; this was the one that didn't.
        match tokio::time::timeout(
            ANSWER_TIMEOUT,
            tx.send(Job {
                cypher,
                model,
                reply,
            }),
        )
        .await
        {
            Err(_) => {
                return Err(
                    "The graph session is not accepting queries - the Studio tab \
                            may be suspended. Ask the user to bring it back."
                        .into(),
                );
            }
            Ok(Err(_)) => {
                return Err(
                    "The graph session just disconnected. Ask the user to reopen \
                            the conversation's graph panel and try again."
                        .into(),
                );
            }
            Ok(Ok(())) => {}
        }
        match tokio::time::timeout(ANSWER_TIMEOUT, rx).await {
            Ok(Ok(body)) => Ok(body),
            // Sender dropped: socket died with the query in flight.
            Ok(Err(_)) => Err(
                "The graph session disconnected while the query was running. \
                               Ask the user to reopen the graph panel and try again."
                    .into(),
            ),
            Err(_) => Err(format!(
                "The graph session did not answer within {} s - the Studio tab may be \
                 closed or suspended. Ask the user to bring it back.",
                ANSWER_TIMEOUT.as_secs()
            )),
        }
    }
}

// ── the WebSocket the Studio tab holds open ─────────────────────────────

#[derive(Deserialize)]
struct BridgeQuery {
    conversation: String,
}

async fn bridge(
    State(state): State<Arc<crate::routes::AppState>>,
    Query(q): Query<BridgeQuery>,
    ws: WebSocketUpgrade,
) -> Response {
    let graphs = state.graphs.clone();
    ws.on_upgrade(move |socket| bridge_ws(socket, graphs, q.conversation))
}

async fn bridge_ws(mut socket: WebSocket, graphs: Arc<Bridge>, conversation: String) {
    let mut rx = graphs.attach(&conversation);
    // A clone of our own sender, kept only so detach can prove identity.
    let mine = match graphs.sender(&conversation) {
        Some(tx) => tx,
        None => return,
    };
    // Replies waiting on the tab, keyed by frame id. Dropping this map on any
    // exit path is what turns a dead socket into per-call "disconnected"
    // errors rather than 20 s of silence each.
    let mut pending: HashMap<u64, tokio::sync::oneshot::Sender<String>> = HashMap::new();
    let mut next_id: u64 = 0;

    loop {
        tokio::select! {
            job = rx.recv() => {
                let Some(job) = job else { break }; // replaced by a newer tab
                next_id += 1;
                pending.insert(next_id, job.reply);
                let frame =
                    serde_json::json!({ "id": next_id, "cypher": job.cypher, "model": job.model });
                if socket.send(Message::Text(frame.to_string().into())).await.is_err() {
                    break;
                }
            }
            msg = socket.recv() => {
                match msg {
                    Some(Ok(Message::Text(text))) => {
                        #[derive(Deserialize)]
                        struct Answer { id: u64, body: String }
                        if let Ok(a) = serde_json::from_str::<Answer>(&text)
                            && let Some(reply) = pending.remove(&a.id) {
                                let _ = reply.send(a.body);
                            }
                    }
                    Some(Ok(Message::Close(_))) | Some(Err(_)) | None => break,
                    // Ping/pong are answered by axum; other frames carry nothing.
                    Some(Ok(_)) => {}
                }
            }
        }
    }
    graphs.detach(&conversation, &mine);
}

// ── the MCP server the runner calls ─────────────────────────────────────

fn refuse(msg: impl Into<String>) -> CallToolResult {
    CallToolResult::error(vec![ContentBlock::text(msg.into())])
}

fn ok(msg: impl Into<String>) -> CallToolResult {
    CallToolResult::success(vec![ContentBlock::text(msg.into())])
}

/// Same Option-everywhere shape as the artifacts args (see the CreateArgs doc
/// there): the schema still says `required` via `extend`, and the handler
/// answers a missing field with an instruction instead of a -32602.
#[derive(Debug, Deserialize, JsonSchema)]
#[schemars(extend("required" = ["cypher"]))]
pub struct QueryArgs {
    /// One openCypher statement, e.g. "MATCH (p:Person) RETURN p.name LIMIT 10".
    #[serde(default)]
    pub cypher: Option<String>,
}

#[derive(Clone)]
pub struct GraphTools {
    bridge: Arc<Bridge>,
}

#[tool_router]
impl GraphTools {
    pub fn new(bridge: Arc<Bridge>) -> Self {
        Self { bridge }
    }

    #[tool(
        description = "Run one openCypher statement against the graph database attached to \
        this conversation. Returns columns, the first rows, and total_rows as JSON. \
        Read-only: write and schema statements are refused. Narrow large results with \
        WHERE or aggregation instead of paging."
    )]
    async fn graph_query(
        &self,
        Parameters(p): Parameters<QueryArgs>,
        Extension(parts): Extension<Parts>,
    ) -> Result<CallToolResult, ErrorData> {
        let conv = parts
            .headers
            .get(CONVERSATION_HEADER)
            .and_then(|v| v.to_str().ok())
            .filter(|s| !s.is_empty());
        let Some(conv) = conv else {
            return Ok(refuse("graph_query is unavailable in this context"));
        };
        let Some(cypher) = p.cypher.filter(|c| !c.trim().is_empty()) else {
            return Ok(refuse(
                "graph_query needs `cypher` - one openCypher statement as a string.",
            ));
        };
        let model = parts
            .headers
            .get(MODEL_HEADER)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();
        Ok(match self.bridge.query(conv, cypher, model).await {
            Ok(body) => ok(body),
            Err(e) => refuse(e),
        })
    }
}

#[tool_handler]
impl ServerHandler for GraphTools {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_instructions(INSTRUCTIONS)
    }
}

/// Streamable-HTTP MCP service, mounted at /api/mcp/graph (same stateless
/// config as artifacts: one request per call, headers reach tools via
/// Extension<Parts>).
pub fn mcp_service(
    bridge: Arc<Bridge>,
    allowed_hosts: Vec<String>,
) -> rmcp::transport::streamable_http_server::StreamableHttpService<
    GraphTools,
    rmcp::transport::streamable_http_server::session::local::LocalSessionManager,
> {
    use rmcp::transport::streamable_http_server::{
        StreamableHttpServerConfig, StreamableHttpService, session::local::LocalSessionManager,
    };
    // see artifacts::mcp_service for why the Host allow-list is set rather
    // than disabled, and why legacy_session_mode replaces stateful_mode
    let mut config = StreamableHttpServerConfig::default().with_allowed_hosts(allowed_hosts);
    config.legacy_session_mode = false;
    StreamableHttpService::new(
        move || Ok(GraphTools::new(bridge.clone())),
        Arc::new(LocalSessionManager::default()),
        config,
    )
}

pub fn routes() -> axum::Router<Arc<crate::routes::AppState>> {
    axum::Router::new().route("/api/graph/bridge", axum::routing::get(bridge))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fake tab: serves its job queue, answering every query by echoing.
    fn fake_tab(mut rx: tokio::sync::mpsc::Receiver<Job>) {
        tokio::spawn(async move {
            while let Some(job) = rx.recv().await {
                let _ = job.reply.send(format!("echo:{}", job.cypher));
            }
        });
    }

    #[tokio::test]
    async fn query_round_trips_through_a_session() {
        let b = Bridge::new();
        fake_tab(b.attach("c1"));
        let out = b
            .query("c1", "MATCH (n) RETURN n".into(), String::new())
            .await
            .unwrap();
        assert_eq!(out, "echo:MATCH (n) RETURN n");
    }

    #[tokio::test]
    async fn no_session_names_the_fix() {
        let b = Bridge::new();
        let err = b
            .query("c1", "MATCH (n) RETURN n".into(), String::new())
            .await
            .unwrap_err();
        assert!(err.contains("not open"), "{err}");
    }

    #[tokio::test]
    async fn a_dropped_session_fails_the_call_not_the_clock() {
        let b = Bridge::new();
        let rx = b.attach("c1");
        drop(rx); // tab vanished without a Close
        let err = b
            .query("c1", "MATCH (n) RETURN n".into(), String::new())
            .await
            .unwrap_err();
        assert!(err.contains("disconnected"), "{err}");
    }

    #[tokio::test]
    async fn a_reconnect_replaces_the_old_tab() {
        let b = Bridge::new();
        let old = b.attach("c1"); // first tab, never answers
        fake_tab(b.attach("c1")); // refresh takes over
        drop(old);
        let out = b
            .query("c1", "RETURN 1".into(), String::new())
            .await
            .unwrap();
        assert_eq!(out, "echo:RETURN 1");
    }

    /// The freeze-vector test: a tab that stopped serving its queue must fail
    /// tool calls on the clock, never hang them. Paused time makes the
    /// timeouts fire instantly.
    #[tokio::test(start_paused = true)]
    async fn a_wedged_tab_fails_calls_instead_of_hanging_them() {
        let b = Bridge::new();
        let _rx = b.attach("c1"); // held open, never served: queue fills
        for _ in 0..4 {
            // fill the buffer; these queries expire on the reply clock
            let _ = b.query("c1", "RETURN 1".into(), String::new()).await;
        }
        let err = b
            .query("c1", "RETURN 2".into(), String::new())
            .await
            .unwrap_err();
        assert!(
            err.contains("not accepting") || err.contains("did not answer"),
            "{err}"
        );
    }

    #[tokio::test]
    async fn detach_only_removes_its_own_registration() {
        let b = Bridge::new();
        let _old_rx = b.attach("c1");
        let old_tx = b.sender("c1").unwrap();
        fake_tab(b.attach("c1")); // replacement
        b.detach("c1", &old_tx); // old task cleaning up must be a no-op
        assert!(
            b.sender("c1").is_some(),
            "replacement was torn down by its predecessor"
        );
    }
}
