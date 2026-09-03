//! The Studio's personal MCP connector library - CRUD over the `connectors`
//! table. A connector is a hosted MCP server the user tries per chat: it rides
//! per REQUEST as the OpenAI inline `mcp` tool (server_url + headers), so it
//! never configures a runner and stays invisible to external API clients. The
//! endpoint-contract tier (a server's own tools, every client sees them) stays
//! in servers/<port>.toml.
//!
//! Headers are returned to the Studio (it builds the requests), which is a
//! looser posture than cloud keys; the store schema comment carries the
//! rationale and the hardening path.

use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};
use serde_json::{Value, json};

use crate::routes::AppState;

/// Public shape: the internal `oauth` blob never leaves the manager - the
/// row exposes `connected` and, when signed in, the bearer merged into
/// `headers` (the Studio rides headers inline; same v1 posture as pasted
/// keys). Stale tokens are refreshed on the way out.
async fn public_row(state: &Arc<AppState>, row: Value) -> Value {
    let mut row = crate::oauth::ensure_fresh(state, row).await;
    let oauth = row["oauth"].take();
    let connected = oauth.get("access_token").and_then(Value::as_str).is_some();
    row["connected"] = json!(connected);
    if let Some(tok) = oauth.get("access_token").and_then(Value::as_str)
        && let Some(h) = row["headers"].as_object_mut()
        && !h.contains_key("Authorization")
    {
        h.insert("Authorization".into(), json!(format!("Bearer {tok}")));
    }
    row
}

/// `GET /api/connectors` - the full library, headers included.
pub async fn list(State(state): State<Arc<AppState>>) -> Response {
    match state.db.list_connectors() {
        Ok(rows) => {
            let mut out = Vec::with_capacity(rows.len());
            for r in rows {
                out.push(public_row(&state, r).await);
            }
            Json(Value::Array(out)).into_response()
        }
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    }
}

/// Re-run a connector's materialization from its stored scope - the seam the
/// OAuth flow uses after a token lands, refreshes or is dropped, so the
/// TOML-carried bearer always matches the stored one (runners re-read live).
pub fn rematerialize(state: &Arc<AppState>, id: &str) {
    if let Ok(Some(row)) = state.db.get_connector(id) {
        let (all, ports) = row_scope(&row);
        if (all || !ports.is_empty())
            && let Err(e) = materialize(
                &state.supervisor.servers_dir(),
                id,
                Some(&entry_from_row(id, &row)),
                all,
                &ports,
            )
        {
            tracing::warn!(error = %e, "connector re-materialization failed");
        }
    }
}

/// `POST /api/connectors` - create `{label, url, headers?, registryKey?}`.
pub async fn create(State(state): State<Arc<AppState>>, Json(doc): Json<Value>) -> Response {
    match state.db.create_connector(&doc) {
        Ok(row) => Json(row).into_response(),
        Err(crate::store::StoreError::Bad(m)) => err(StatusCode::BAD_REQUEST, m),
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    }
}

/// `PUT /api/connectors/{id}` - full-row update `{label, url, headers?}`. A
/// system connector's materialized entries are rewritten in the same breath,
/// so the TOMLs never drift from the library row they mirror.
pub async fn update(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    Json(doc): Json<Value>,
) -> Response {
    match state.db.update_connector(&id, &doc) {
        Ok(()) => {
            if let Ok(Some(row)) = state.db.get_connector(&id) {
                let (all, ports) = row_scope(&row);
                if (all || !ports.is_empty())
                    && let Err(e) = materialize(
                        &state.supervisor.servers_dir(),
                        &id,
                        Some(&entry_from_row(&id, &row)),
                        all,
                        &ports,
                    )
                {
                    return err(StatusCode::INTERNAL_SERVER_ERROR, e);
                }
            }
            Json(json!({"ok": true})).into_response()
        }
        Err(crate::store::StoreError::Bad(m)) => err(StatusCode::BAD_REQUEST, m),
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    }
}

/// `DELETE /api/connectors/{id}`. A system connector is dematerialized from
/// every server config first - deleting the library row must never leave
/// ghost tool entries behind in the TOMLs.
pub async fn remove(State(state): State<Arc<AppState>>, Path(id): Path<String>) -> Response {
    if let Ok(Some(row)) = state.db.get_connector(&id) {
        let (all, ports) = row_scope(&row);
        if (all || !ports.is_empty())
            && let Err(e) = materialize(&state.supervisor.servers_dir(), &id, None, false, &[])
        {
            return err(StatusCode::INTERNAL_SERVER_ERROR, e);
        }
    }
    match state.db.delete_connector(&id) {
        Ok(()) => Json(json!({"ok": true})).into_response(),
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    }
}

// ── the "available on every server" tier ────────────────────────────────────
//
// Checked = the connector is MATERIALIZED into every servers/<port>.toml
// `mcp_servers` array, so any API client of any endpoint can call it by
// label. The per-port TOML stays the whole config (spawn-verbatim, portable,
// standalone runners keep working): this endpoint only
// rewrites that one array, via toml_edit so a hand-formatted file keeps its
// comments and layout. Every entry the manager writes carries
// `connector_id = "<uuid>"` - the ownership marker that lets uncheck/delete/
// edit strip or replace exactly its entries and never a tool someone added
// to an endpoint by hand.
//
// A running model picks this up without a restart: `mcp_servers` is one of the
// runner's LiveConfig keys (paddock-runner routes.rs), re-read on the next
// request whenever the file's mtime moves. Do not tell users it needs a
// restart - that was true before live config and is not any more.

/// `POST /api/connectors/{id}/scope` - body `{"all": bool, "ports": [u16]}`.
/// `all` = every model incl. future ones; `ports` = exactly those endpoints;
/// both empty = per-chat only. Returns the ports whose configs changed.
pub async fn scope(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    Json(body): Json<Value>,
) -> Response {
    let all = body.get("all").and_then(Value::as_bool).unwrap_or(false);
    let ports: Vec<u16> = body
        .get("ports")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_u64().map(|n| n as u16))
                .collect()
        })
        .unwrap_or_default();
    let row = match state.db.get_connector(&id) {
        Ok(Some(r)) => r,
        Ok(None) => {
            return err(
                StatusCode::BAD_REQUEST,
                format!("no connector with id {id}"),
            );
        }
        Err(e) => return err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    };
    let entry = entry_from_row(&id, &row);
    match materialize(
        &state.supervisor.servers_dir(),
        &id,
        Some(&entry),
        all,
        &ports,
    ) {
        Ok(changed) => match state.db.set_connector_scope(&id, all, &ports) {
            Ok(()) => Json(json!({"ok": true, "ports": changed})).into_response(),
            Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
        },
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, e),
    }
}

fn row_scope(row: &Value) -> (bool, Vec<u16>) {
    let all = row["system"].as_bool() == Some(true);
    let ports = row["ports"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_u64().map(|n| n as u16))
                .collect()
        })
        .unwrap_or_default();
    (all, ports)
}

/// The TOML entry a connector materializes as - the same shape the runner's
/// launch registry resolves (server_label/server_url/headers), plus the
/// ownership marker. A signed-in connector's bearer joins the headers, so
/// scoped entries authenticate exactly like the inline tier.
pub(crate) fn entry_from_row(id: &str, row: &Value) -> Value {
    let mut e = json!({
        "server_label": row["label"],
        "server_url": row["url"],
        "connector_id": id,
    });
    let mut headers = row["headers"].as_object().cloned().unwrap_or_default();
    if let Some(tok) = row["oauth"].get("access_token").and_then(Value::as_str)
        && !headers.contains_key("Authorization")
    {
        headers.insert("Authorization".into(), json!(format!("Bearer {tok}")));
    }
    if !headers.is_empty() {
        e["headers"] = Value::Object(headers);
    }
    e
}

/// Rewrite every `servers/*.toml`: strip entries owned by `id`, then append
/// `entry` where the scope says so (`all`, or the port is in `ports`; an
/// entry of None strips everywhere - deletion). Returns changed ports.
fn materialize(
    servers_dir: &std::path::Path,
    id: &str,
    entry: Option<&Value>,
    all: bool,
    ports: &[u16],
) -> Result<Vec<u16>, String> {
    use toml_edit::{DocumentMut, Item};
    let mut changed = Vec::new();
    let listing = match std::fs::read_dir(servers_dir) {
        Ok(l) => l,
        Err(_) => return Ok(changed), // no servers dir yet = nothing to do
    };
    for f in listing.flatten() {
        let path = f.path();
        let Some(port) = path
            .file_stem()
            .and_then(|s| s.to_str())
            .and_then(|s| s.parse::<u16>().ok())
        else {
            continue;
        };
        if path.extension().and_then(|e| e.to_str()) != Some("toml") {
            continue;
        }
        let text =
            std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut doc: DocumentMut = text
            .parse()
            .map_err(|e| format!("{} is not valid TOML: {e}", path.display()))?;
        let mut touched = false;
        // The generator writes mcp_servers as [[mcp_servers]] tables; the
        // Advanced editor may hold an inline array - handle both.
        match doc.get_mut("mcp_servers") {
            Some(Item::ArrayOfTables(arr)) => {
                let before = arr.len();
                arr.retain(|t| t.get("connector_id").and_then(|v| v.as_str()) != Some(id));
                touched = arr.len() != before;
            }
            Some(Item::Value(v)) => {
                if let Some(arr) = v.as_array_mut() {
                    let before = arr.len();
                    arr.retain(|e| {
                        e.as_inline_table()
                            .and_then(|t| t.get("connector_id"))
                            .and_then(|v| v.as_str())
                            != Some(id)
                    });
                    touched = arr.len() != before;
                }
            }
            _ => {}
        }
        let wanted_here = entry.is_some() && (all || ports.contains(&port));
        if let Some(e) = entry.filter(|_| wanted_here) {
            let toml_text =
                toml::to_string(&json!({ "mcp_servers": [e] })).map_err(|x| x.to_string())?;
            let parsed: DocumentMut = toml_text
                .parse()
                .map_err(|x: toml_edit::TomlError| x.to_string())?;
            let new_tbl = parsed["mcp_servers"]
                .as_array_of_tables()
                .and_then(|a| a.get(0))
                .cloned();
            let Some(new_tbl) = new_tbl else {
                return Err("connector entry not TOML-expressible".into());
            };
            match doc.get_mut("mcp_servers") {
                Some(Item::ArrayOfTables(arr)) => arr.push(new_tbl),
                Some(Item::Value(v)) => {
                    if let Some(arr) = v.as_array_mut() {
                        let mut inline = toml_edit::InlineTable::new();
                        for (k, val) in new_tbl.iter() {
                            if let Item::Value(val) = val {
                                inline.insert(k, val.clone());
                            }
                        }
                        arr.push(inline);
                    }
                }
                _ => {
                    let mut arr = toml_edit::ArrayOfTables::new();
                    arr.push(new_tbl);
                    doc.insert("mcp_servers", Item::ArrayOfTables(arr));
                }
            }
            touched = true;
        } else if doc.get("mcp_servers").is_some_and(|i| match i {
            Item::ArrayOfTables(a) => a.is_empty(),
            Item::Value(v) => v.as_array().is_some_and(|a| a.is_empty()),
            _ => false,
        }) {
            // an emptied array is noise in the file
            doc.remove("mcp_servers");
        }
        if touched {
            std::fs::write(&path, doc.to_string())
                .map_err(|e| format!("{}: {e}", path.display()))?;
            changed.push(port);
        }
    }
    changed.sort_unstable();
    Ok(changed)
}

/// The entries every new server config gets at creation: all system-checked
/// connectors (skipping any the spec already carries by label).
pub fn system_entries(db: &crate::store::Store, existing: &[Value]) -> Vec<Value> {
    let have: std::collections::HashSet<&str> = existing
        .iter()
        .filter_map(|e| e.get("server_label").and_then(Value::as_str))
        .collect();
    db.list_connectors()
        .unwrap_or_default()
        .into_iter()
        .filter(|c| c["system"].as_bool() == Some(true))
        .filter(|c| !have.contains(c["label"].as_str().unwrap_or("")))
        .map(|c| entry_from_row(c["id"].as_str().unwrap_or(""), &c))
        .collect()
}

/// `POST /api/mcp/tools` - what does this server actually expose? The
/// composer's tool picker needs names + descriptions to fuzzy-search over.
/// `{connector_id}` resolves a library row (token refreshed, bearer merged);
/// `{port, label}` resolves an entry in that endpoint's own config file -
/// the same registry the runner reads, bearer already materialized. One-shot:
/// connect, list, drop. Failures come back soft (`ok:false`) - the picker
/// still offers the whole server when its listing is unreachable.
pub async fn tools(State(state): State<Arc<AppState>>, Json(doc): Json<Value>) -> Response {
    // The manager's own MCP server (artifacts) is answered in-process: it is
    // not a remote to dial, and reading its live router keeps the picker from
    // drifting when a tool is added.
    if doc.get("builtin").and_then(Value::as_str) == Some("artifacts") {
        return Json(json!({
            "ok": true,
            "tools": crate::artifacts::tool_list(),
            "instructions": crate::artifacts::INSTRUCTIONS,
        }))
        .into_response();
    }
    let (url, headers) = if let Some(id) = doc.get("connector_id").and_then(Value::as_str) {
        let Ok(Some(row)) = state.db.get_connector(id) else {
            return err(StatusCode::NOT_FOUND, "no such connector".into());
        };
        let row = crate::oauth::ensure_fresh(&state, row).await;
        let mut headers = row["headers"].as_object().cloned().unwrap_or_default();
        if let Some(tok) = row["oauth"].get("access_token").and_then(Value::as_str)
            && !headers.contains_key("Authorization")
        {
            headers.insert("Authorization".into(), json!(format!("Bearer {tok}")));
        }
        (row["url"].as_str().unwrap_or_default().to_string(), headers)
    } else if let (Some(port), Some(label)) = (
        doc.get("port").and_then(Value::as_u64),
        doc.get("label").and_then(Value::as_str),
    ) {
        let path = state.supervisor.server_config_path(port as u16);
        let Ok(raw) = std::fs::read_to_string(&path) else {
            return err(
                StatusCode::NOT_FOUND,
                format!("no config file for port {port}"),
            );
        };
        let Ok(cfg) = toml::from_str::<toml::Value>(&raw) else {
            return err(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("{} is not valid TOML", path.display()),
            );
        };
        let Some(entry) = cfg
            .get("mcp_servers")
            .and_then(toml::Value::as_array)
            .and_then(|a| {
                a.iter()
                    .find(|e| e.get("server_label").and_then(toml::Value::as_str) == Some(label))
            })
        else {
            return err(
                StatusCode::NOT_FOUND,
                format!("no server \"{label}\" on port {port}"),
            );
        };
        let Some(url) = entry.get("server_url").and_then(toml::Value::as_str) else {
            return err(
                StatusCode::NOT_FOUND,
                format!("\"{label}\" has no server_url (stdio servers list on demand only)"),
            );
        };
        let mut headers = serde_json::Map::new();
        if let Some(h) = entry.get("headers").and_then(toml::Value::as_table) {
            for (k, v) in h {
                if let Some(s) = v.as_str() {
                    headers.insert(k.clone(), json!(s));
                }
            }
        }
        (url.to_string(), headers)
    } else {
        return err(
            StatusCode::BAD_REQUEST,
            "need connector_id, or port + label".into(),
        );
    };
    if !(url.starts_with("http://") || url.starts_with("https://")) {
        return err(
            StatusCode::BAD_REQUEST,
            "url must start with http:// or https://".into(),
        );
    }
    let cfg = paddock_mcp::ServerConfig {
        id: format!("picker:{url}"),
        label: "picker".into(),
        transport: paddock_mcp::Transport::Http {
            url,
            headers: headers
                .iter()
                .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
                .collect(),
        },
    };
    // 45s, not a snappy 15: real MCP servers cold-start (an IIS app pool
    // takes 15-30s to spin up). A dead host still
    // fails fast on connect-refused; only a genuinely warming server waits.
    let started = std::time::Instant::now();
    // Keep the client: its handshake `instructions` are what the runner will
    // fold into the system prompt, and the picker has to be able to show that.
    let listing = tokio::time::timeout(std::time::Duration::from_secs(45), async {
        let client = paddock_mcp::McpClient::connect(&cfg).await?;
        let tools = client.list_tools().await?;
        Ok::<_, paddock_mcp::McpError>((tools, client.instructions()))
    })
    .await;
    if !matches!(listing, Ok(Ok(_))) {
        tracing::warn!(url = %cfg.id, elapsed_ms = started.elapsed().as_millis() as u64,
            "picker tool listing failed");
    }
    let verdict = match listing {
        Err(_) => json!({"ok": false, "error": "timed out"}),
        Ok(Err(e)) => json!({"ok": false, "error": e.to_string()}),
        Ok(Ok((tools, instructions))) => json!({
            "ok": true,
            // What the runner will fold into the system prompt for this
            // server. The Studio shows it read-only so injected text is never
            // invisible to the user.
            "instructions": instructions,
            "tools": tools
                .iter()
                .map(|t| json!({"name": t.name, "description": t.description}))
                .collect::<Vec<_>>(),
        }),
    };
    Json(verdict).into_response()
}

/// `POST /api/connectors/check` - `{url, headers?}` -> the handshake probe's
/// verdict, so the form can test a server before saving it.
pub async fn check(Json(doc): Json<Value>) -> Response {
    let url = doc.get("url").and_then(Value::as_str).unwrap_or("");
    if !(url.starts_with("http://") || url.starts_with("https://")) {
        return err(
            StatusCode::BAD_REQUEST,
            "url must start with http:// or https://".into(),
        );
    }
    let headers = doc
        .get("headers")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    Json(crate::oauth::probe(url, &headers).await).into_response()
}

fn err(status: StatusCode, msg: String) -> Response {
    (
        status,
        Json(json!({"error": {"type": "connector_error", "message": msg}})),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    #[test]
    fn materialize_marks_strips_and_spares_hand_entries() {
        let dir = std::env::temp_dir().join(format!("pk-mat-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        // one config with a HAND-ADDED tool (no marker), one bare, one non-config file
        std::fs::write(
            dir.join("11540.toml"),
            "# hand-formatted comment survives\nmodel = 'a.gguf'\n\n[[mcp_servers]]\nserver_label = \"mine\"\nserver_url = \"https://hand.example\"\n",
        )
        .unwrap();
        std::fs::write(dir.join("11541.toml"), "model = 'b.gguf'\n").unwrap();
        std::fs::write(dir.join("notes.txt"), "not a config").unwrap();

        let row =
            json!({"label": "registry", "url": "https://r.example/mcp", "headers": {"X-K": "v"}});
        let entry = super::entry_from_row("cid-1", &row);
        let ports = super::materialize(&dir, "cid-1", Some(&entry), true, &[]).unwrap();
        assert_eq!(ports, vec![11540, 11541]);
        let a = std::fs::read_to_string(dir.join("11540.toml")).unwrap();
        assert!(
            a.contains("# hand-formatted comment survives"),
            "toml_edit keeps layout"
        );
        assert!(
            a.contains("hand.example") && a.contains("r.example"),
            "both entries present"
        );
        assert!(
            a.contains("connector_id"),
            "materialized entry carries the marker"
        );

        // updated row re-materializes: strip + re-add with the new url
        let row2 = json!({"label": "registry", "url": "https://r2.example/mcp", "headers": {}});
        super::materialize(
            &dir,
            "cid-1",
            Some(&super::entry_from_row("cid-1", &row2)),
            true,
            &[],
        )
        .unwrap();
        let a = std::fs::read_to_string(dir.join("11540.toml")).unwrap();
        assert!(a.contains("r2.example") && !a.contains("r.example/mcp"));

        // narrow the scope to one port: the other loses its entry
        super::materialize(
            &dir,
            "cid-1",
            Some(&super::entry_from_row("cid-1", &row2)),
            false,
            &[11540],
        )
        .unwrap();
        assert!(
            std::fs::read_to_string(dir.join("11540.toml"))
                .unwrap()
                .contains("r2.example")
        );
        assert!(
            !std::fs::read_to_string(dir.join("11541.toml"))
                .unwrap()
                .contains("r2.example")
        );

        // uncheck: exactly the marked entries go, the hand one stays
        let ports = super::materialize(&dir, "cid-1", None, false, &[]).unwrap();
        assert_eq!(ports, vec![11540]);
        let a = std::fs::read_to_string(dir.join("11540.toml")).unwrap();
        assert!(a.contains("hand.example") && !a.contains("r2.example"));
        let b = std::fs::read_to_string(dir.join("11541.toml")).unwrap();
        assert!(
            !b.contains("mcp_servers"),
            "emptied array is removed, not left as noise"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn connector_crud_round_trips() {
        let dir = std::env::temp_dir().join(format!("pk-conn-{}", uuid::Uuid::new_v4()));
        let db = crate::store::Store::open(&dir.join("t.db")).unwrap();
        let row = db
            .create_connector(&json!({
                "label": "registry", "url": "https://registry.truespar.com/mcp",
                "headers": {"Authorization": "Bearer x"}, "registryKey": "repo:x/y",
            }))
            .unwrap();
        let id = row["id"].as_str().unwrap().to_string();
        assert_eq!(row["headers"]["Authorization"], "Bearer x");
        // duplicate label refused, bad label refused, bad url refused
        assert!(
            db.create_connector(&json!({"label": "registry", "url": "https://x"}))
                .is_err()
        );
        assert!(
            db.create_connector(&json!({"label": "no spaces", "url": "https://x"}))
                .is_err()
        );
        assert!(
            db.create_connector(&json!({"label": "ok", "url": "ftp://x"}))
                .is_err()
        );
        db.update_connector(
            &id,
            &json!({"label": "reg2", "url": "https://r.example/mcp"}),
        )
        .unwrap();
        let all = db.list_connectors().unwrap();
        assert_eq!(all.len(), 1);
        assert_eq!(all[0]["label"], "reg2");
        assert_eq!(
            all[0]["headers"],
            json!({}),
            "unsent headers reset to empty on full update"
        );
        db.delete_connector(&id).unwrap();
        assert!(db.list_connectors().unwrap().is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
