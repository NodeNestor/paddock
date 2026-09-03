//! Server-push for the Studio shell (`GET /api/events`, SSE): the manager
//! tells every open tab when fleet state CHANGES instead of each tab asking
//! twice every few seconds. Polling was the accretion, not the design - the
//! runners already keep event rings and the manager already watches VRAM;
//! this is the missing fan-out to the browser.
//!
//! Shape decisions:
//! - SSE, not a second WebSocket: this is one-directional STATE push, and
//!   `EventSource` brings auto-reconnect for free with zero upgrade ceremony
//!   through the TLS layer. (The graph bridge stays WS - it is bidirectional
//!   RPC.)
//! - Events are STATE SNAPSHOTS on change, not a log: a fresh or reconnected
//!   client is synced by the initial burst, so there is no replay cursor to
//!   get wrong. `Last-Event-ID` is deliberately unused.
//! - One subscriber-gated watcher: the manager sweeps its own runners (the
//!   same `admin` pipes the HTTP route asks) only WHILE at least one tab
//!   listens, every 2s - replacing N tabs x two overlapping poll loops.
//!   Change detection is a signature over the things the shell renders, so a
//!   quiet fleet emits nothing at all.
//! - The browser keeps its poll code as the FALLBACK: an older manager 404s
//!   this endpoint and the stores keep their timers. Degrade, never blank.

use std::sync::Arc;
use std::time::Duration;

use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use futures_util::stream::Stream;
use serde_json::Value;
use tokio::sync::broadcast;

use crate::routes::AppState;

/// One pushed event: `kind` becomes the SSE event name, `data` the payload.
#[derive(Clone, Debug)]
pub struct Push {
    pub kind: &'static str,
    pub data: Arc<Value>,
}

/// The fan-out hub. Broadcast is the right primitive: every subscriber sees
/// every event, laggards drop oldest (state events are snapshots, so a drop
/// only means the next snapshot re-syncs them).
pub struct Hub {
    tx: broadcast::Sender<Push>,
}

impl Hub {
    pub fn new() -> Self {
        Self {
            tx: broadcast::channel(64).0,
        }
    }

    pub fn publish(&self, kind: &'static str, data: Value) {
        // no subscribers = no clone, no send - publishing stays free when
        // nothing listens
        if self.tx.receiver_count() > 0 {
            let _ = self.tx.send(Push {
                kind,
                data: Arc::new(data),
            });
        }
    }

    pub fn subscribe(&self) -> broadcast::Receiver<Push> {
        self.tx.subscribe()
    }

    pub fn has_listeners(&self) -> bool {
        self.tx.receiver_count() > 0
    }
}

impl Default for Hub {
    fn default() -> Self {
        Self::new()
    }
}

/// The watcher: while any tab listens, sweep fleet state every 2s and publish
/// on CHANGE. This is the one poller the whole box needs - each browser tab
/// used to run two (3s fleet + 5s models), each sweep walking every runner's
/// admin pipe.
pub fn spawn_watcher(state: Arc<AppState>) {
    tokio::spawn(async move {
        let mut last_fleet_sig = String::new();
        let mut last_update_sig = String::new();
        loop {
            tokio::time::sleep(Duration::from_secs(2)).await;
            if !state.push.has_listeners() {
                // idle box: nothing sweeps, nothing burns
                last_fleet_sig.clear();
                continue;
            }
            let rows = fleet_rows(&state).await;
            // The signature covers what the shell RENDERS: identity, status,
            // spec badge, vram. Uptime/in-flight churn on every sweep and are
            // deliberately excluded - the slow reconcile poll carries them.
            let sig: String = rows
                .iter()
                .map(|r| {
                    format!(
                        "{}:{}:{}:{}:{};",
                        r.get("port").and_then(Value::as_u64).unwrap_or(0),
                        r.get("pid").and_then(Value::as_u64).unwrap_or(0),
                        r.get("status").and_then(Value::as_str).unwrap_or(""),
                        r.get("spec").and_then(Value::as_str).unwrap_or(""),
                        r.get("model").and_then(Value::as_str).unwrap_or(""),
                    )
                })
                .collect();
            if sig != last_fleet_sig {
                last_fleet_sig = sig;
                state.push.publish("fleet", Value::Array(rows));
            }
            // update state: progress while a download runs, else the cached
            // (hourly) check - tiny payload, only pushed on change
            let info = update_info(&state).await;
            let s = info.to_string();
            if s != last_update_sig {
                last_update_sig = s;
                state.push.publish("update", info);
            }
        }
    });
}

/// The same rows `GET /api/runners` serves - one builder, two consumers, so
/// the pushed state can never drift from the polled state.
pub async fn fleet_rows(state: &AppState) -> Vec<Value> {
    let views = state.supervisor.list().await;
    let recon = state.recon.borrow().clone();
    views
        .into_iter()
        .map(|v| {
            let mut row = serde_json::to_value(&v).unwrap_or_else(|_| serde_json::json!({}));
            if let Some(r) = recon.as_ref()
                && let Some(rv) = r
                    .runners
                    .iter()
                    .find(|r| r.port == v.port && r.pid == v.pid)
            {
                row["vram"] = serde_json::to_value(rv).unwrap_or(Value::Null);
            }
            row
        })
        .collect()
}

/// `GET /api/events` - the SSE stream. Initial burst: the current fleet and
/// update state, so a fresh tab renders without waiting for a change.
pub async fn events_stream(
    State(state): State<Arc<AppState>>,
) -> Sse<impl Stream<Item = Result<Event, std::convert::Infallible>>> {
    let rx = state.push.subscribe();
    let fleet_now = fleet_rows(&state).await;
    let update_now = update_info(&state).await;
    let stream = async_stream::stream! {
        yield Ok(Event::default().event("fleet").data(Value::Array(fleet_now).to_string()));
        yield Ok(Event::default().event("update").data(update_now.to_string()));
        let mut rx = rx;
        loop {
            match rx.recv().await {
                Ok(p) => {
                    yield Ok(Event::default().event(p.kind).data(p.data.to_string()));
                }
                Err(broadcast::error::RecvError::Lagged(_)) => continue, // next snapshot re-syncs
                Err(broadcast::error::RecvError::Closed) => break,
            }
        }
    };
    Sse::new(stream).keep_alive(KeepAlive::new().interval(Duration::from_secs(25)))
}

/// The `/api/updates` body - check result plus any in-flight download - as
/// one builder for the HTTP route and the push watcher alike.
pub async fn update_info(state: &AppState) -> Value {
    let checked = state.updates.get_or_check(crate::updates::http()).await;
    let dl = state
        .update_dl
        .lock()
        .expect("update dl mutex")
        .clone()
        .map(|d| d.status());
    let mut body = serde_json::to_value(&checked).unwrap_or_else(|_| serde_json::json!({}));
    if let Some(obj) = body.as_object_mut() {
        obj.insert("download".into(), dl.unwrap_or(Value::Null));
    }
    body
}
