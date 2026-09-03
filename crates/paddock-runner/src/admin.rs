//! The runner's admin router - served only over the paddock-admin local
//! transport (named pipe / UDS), never on the TCP port. No Bearer auth here:
//! the transport's OS-identity ACL is the authentication (doc §5.1).
//!
//! v1-frozen core: identify, health, drain, shutdown. Rich (capability-gated
//! via identify): stats, events (§8.1 - resumable long-poll over the ring).

use std::sync::Arc;
use std::time::Duration;

use axum::Json;
use axum::extract::{Query, State};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use paddock_admin::types::{
    DrainRequest, DrainState, Health, Identify, ShutdownAck, ShutdownRequest, SpecInfo,
    WIRE_VERSION,
};

/// What the admin surface needs from the runner. Deliberately its own struct
/// (not `AppState`) so the admin router states exactly what it touches.
pub struct AdminState {
    pub app: Arc<crate::routes::AppState>,
    pub port: u16,
    pub started_at_unix: u64,
    pub started: std::time::Instant,
}

pub fn router(state: Arc<AdminState>) -> axum::Router {
    axum::Router::new()
        .route("/v1/identify", get(identify))
        .route("/v1/health", get(health))
        .route("/v1/drain", post(drain))
        .route("/v1/shutdown", post(shutdown))
        .route("/v1/stats", get(stats))
        .route("/v1/events", get(events))
        .route("/v1/metrics", get(metrics))
        .route("/v1/metrics_snapshots", get(metrics_snapshots))
        .fallback(crate::routes::not_found)
        .with_state(state)
}

async fn identify(State(s): State<Arc<AdminState>>) -> Response {
    Json(Identify {
        wire: WIRE_VERSION,
        role: "runner".into(),
        version: env!("CARGO_PKG_VERSION").into(),
        pid: std::process::id(),
        port: s.port,
        model: s.app.serving.as_ref().map(|m| m.id.clone()),
        embedder: s.app.embedder.as_ref().map(|e| e.id.clone()),
        asr: s.app.asr.as_ref().map(|a| a.id.clone()),
        aligner: s.app.aligner.as_ref().map(|a| a.id.clone()),
        started_at_unix: s.started_at_unix,
        instance_id: s.app.instance_id.clone(),
        // the load's own record of what it wired - catalog predictions defer
        spec: s.app.serving.as_ref().map(|m| SpecInfo {
            heads: m.spec.heads,
            drafter: m.spec.drafter.clone(),
            off: s.app.spec_off,
        }),
        capabilities: {
            let mut caps = vec!["stats".to_owned()];
            if s.app.events.enabled() {
                caps.push("events".to_owned());
            }
            if s.app.metrics.enabled() {
                caps.push("metrics".to_owned());
                // The snapshot ring rides the metrics gate - no counters, no
                // snapshots of them.
                caps.push("metrics-snapshots".to_owned());
            }
            caps
        },
    })
    .into_response()
}

async fn health(State(s): State<Arc<AdminState>>) -> Response {
    Json(Health {
        status: if s.app.drain.is_draining() {
            "draining"
        } else {
            "ok"
        }
        .into(),
        in_flight: s.app.drain.in_flight() as u64,
        uptime_s: s.started.elapsed().as_secs(),
    })
    .into_response()
}

const DEFAULT_DRAIN_TIMEOUT_MS: u64 = 30_000;

/// Begin draining and wait (bounded) for in-flight work to finish. Returns
/// the resulting state either way - a timeout is reported, never hidden.
async fn drain(State(s): State<Arc<AdminState>>, body: Option<Json<DrainRequest>>) -> Response {
    let timeout = body
        .and_then(|b| b.timeout_ms)
        .unwrap_or(DEFAULT_DRAIN_TIMEOUT_MS);
    s.app.drain.begin();
    let drained = s
        .app
        .drain
        .wait_drained(Duration::from_millis(timeout))
        .await;
    Json(DrainState {
        draining: true,
        in_flight: s.app.drain.in_flight() as u64,
        drained,
        timed_out: !drained,
    })
    .into_response()
}

/// Drain then EXIT the process. Acks immediately; the caller then waits on
/// the process handle (`Child::wait`), not the pipe. Exit is the design, not
/// a shortcut: process teardown is the one guaranteed VRAM release (doc §4),
/// and the runner is stateless on disk so there is nothing to flush.
async fn shutdown(
    State(s): State<Arc<AdminState>>,
    body: Option<Json<ShutdownRequest>>,
) -> Response {
    let timeout = body
        .and_then(|b| b.timeout_ms)
        .unwrap_or(DEFAULT_DRAIN_TIMEOUT_MS);
    s.app.drain.begin();
    let ctl = s.app.clone();
    tokio::spawn(async move {
        let drained = ctl.drain.wait_drained(Duration::from_millis(timeout)).await;
        if drained {
            tracing::info!("drained; exiting");
        } else {
            tracing::warn!(
                in_flight = ctl.drain.in_flight(),
                "drain timed out; exiting with requests still in flight"
            );
        }
        std::process::exit(0);
    });
    (
        axum::http::StatusCode::ACCEPTED,
        Json(ShutdownAck {
            status: "draining-then-exit".into(),
        }),
    )
        .into_response()
}

/// Rich surface: the engine self-report (same payload as `/api/stats` on the
/// network port - offered here too so a manager needs only the pipe).
async fn stats(State(s): State<Arc<AdminState>>) -> Response {
    Json(&*s.app.stats.latest()).into_response()
}

/// Rich surface: the same Prometheus exposition `/metrics` serves
/// on the inference port, over the pipe - so the manager's scrape depends on
/// neither the inference port's bind nor its auth. No key check here: the
/// transport's OS-identity ACL is the authentication, like everything else on
/// this router.
async fn metrics(State(s): State<Arc<AdminState>>, headers: axum::http::HeaderMap) -> Response {
    if !s.app.metrics.enabled() {
        return (
            axum::http::StatusCode::NOT_IMPLEMENTED,
            Json(serde_json::json!({
                "error": "metrics are disabled on this runner (--no-metrics)"
            })),
        )
            .into_response();
    }
    crate::metrics::render_response(&s.app, &headers)
}

#[derive(serde::Deserialize)]
struct SnapshotsQuery {
    /// Resume cursor: return snapshots with sequence ≥ this. Default 0.
    since: Option<u64>,
    /// Page cap. Default 512, hard cap = the ring's full depth.
    max: Option<usize>,
}

/// Rich surface: the 1-minute counter self-snapshot
/// ring, resumable by sequence like /v1/events. A manager reattaching after
/// an outage replays consecutive pairs to reconstruct its blind window at
/// full resolution; a reader that fell off the tail gets `dropped`, never a
/// silent hole. No long-poll - this is pulled on attach, not subscribed to.
async fn metrics_snapshots(
    State(s): State<Arc<AdminState>>,
    Query(q): Query<SnapshotsQuery>,
) -> Response {
    if !s.app.metrics.enabled() {
        return (
            axum::http::StatusCode::NOT_IMPLEMENTED,
            Json(serde_json::json!({
                "error": "metrics are disabled on this runner (--no-metrics), so there are no snapshots"
            })),
        )
            .into_response();
    }
    let since = q.since.unwrap_or(0);
    let max = q.max.unwrap_or(512).min(1440);
    let (dropped, next, snapshots) = s.app.snapshots.since(since, max);
    Json(serde_json::json!({ "next": next, "dropped": dropped, "snapshots": snapshots }))
        .into_response()
}

#[derive(serde::Deserialize)]
struct EventsQuery {
    /// Resume cursor: return records with sequence ≥ this. Default 0.
    since: Option<u64>,
    /// Page cap. Default 512, hard cap 2048.
    max: Option<usize>,
    /// Long-poll: wait up to this for at least one new record (cap 60 s).
    wait_ms: Option<u64>,
}

/// Rich surface (§8.1): the event ring, resumable by sequence number. A reader
/// that fell off the tail gets `dropped` - the "K events dropped" contract,
/// never a silent gap. Long-polling via `wait_ms` fits the one-connection-per-
/// call admin client; the manager's collector resumes from its last cursor.
async fn events(State(s): State<Arc<AdminState>>, Query(q): Query<EventsQuery>) -> Response {
    if !s.app.events.enabled() {
        return (
            axum::http::StatusCode::NOT_IMPLEMENTED,
            Json(serde_json::json!({
                "error": "the event ring is disabled on this runner (--no-events)"
            })),
        )
            .into_response();
    }
    let since = q.since.unwrap_or(0);
    let max = q.max.unwrap_or(512).min(2048);
    let wait = Duration::from_millis(q.wait_ms.unwrap_or(0).min(60_000));
    let (dropped, next, events) = s.app.events.wait_since(since, max, wait).await;
    Json(serde_json::json!({ "next": next, "dropped": dropped, "events": events })).into_response()
}
