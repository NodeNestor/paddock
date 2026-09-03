//! Per-request event records + the bounded in-memory ring (doc §8.1).
//!
//! Every inference request produces exactly one record, built at the HTTP
//! edge (correlation ids, timing, status) and enriched by the handler (usage,
//! finish reasons, cached-prefix depth) through an [`EventScope`] riding the
//! request extensions. The record is pushed when the RESPONSE BODY completes -
//! a streaming completion's record carries its true duration and edge TTFT,
//! and a client disconnect still yields a record with whatever was known.
//!
//! Invariant (doc §8.7): observability never blocks serving. The hot-path cost
//! is one short mutex push into a fixed-size ring (~4K records ≈ 2 MB); there
//! are no per-token events and nothing ever touches disk. Subscribers read by
//! sequence number over the ADMIN surface; a reader that fell off the tail is
//! told "K events dropped" - never a silent gap.
//!
//! Field names follow the OTel GenAI semantic conventions where one exists
//! (`gen_ai.request.model`, `gen_ai.usage.*`, `gen_ai.response.finish_reasons`)
//! so the future OTLP export (§8.1) is a 1:1 mapping. Engine attributes with
//! no semconv equivalent are `paddock.*`: the per-request phase split
//! (tokenize at the edge; queue/prefill/decode measured inside the engine and
//! carried on `TokenEvent::Done` as `RunStats`), spec-decode drafted/accepted,
//! and the KV page footprint. The edge-measured `ttft_ms` stays alongside as
//! the client-visible number the engine phases decompose.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::Serialize;

/// One request, one record (~0.5 KB serialized).
#[derive(Debug, Clone, Serialize)]
pub struct EventRecord {
    /// Ring sequence number - the subscriber's resume cursor.
    pub seq: u64,
    /// Unix millis at request arrival.
    pub ts_ms: u64,
    /// Request path (`/v1/chat/completions`, ...).
    pub endpoint: String,
    pub status: u16,
    /// Arrival -> response body complete (streaming included).
    pub duration_ms: u64,
    /// Whether the response went out as an event stream.
    pub stream: bool,
    /// Client `X-Request-ID`, honored and echoed; generated otherwise.
    pub request_id: String,
    /// Client `traceparent`, recorded verbatim so this request can join the
    /// caller's own distributed trace. Never propagated by us.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub traceparent: Option<String>,
    /// First matching session header (default `X-Session-ID`,
    /// `X-Litellm-Session-Id`) - explicit session identity for grouping.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    /// The OpenAI `user` / Anthropic `metadata.user_id` grouping key.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    /// Served model id (the truthful one, not the request's alias).
    #[serde(
        rename = "gen_ai.request.model",
        skip_serializing_if = "Option::is_none"
    )]
    pub model: Option<String>,
    #[serde(
        rename = "gen_ai.usage.input_tokens",
        skip_serializing_if = "Option::is_none"
    )]
    pub input_tokens: Option<u64>,
    #[serde(
        rename = "gen_ai.usage.output_tokens",
        skip_serializing_if = "Option::is_none"
    )]
    pub output_tokens: Option<u64>,
    #[serde(
        rename = "gen_ai.response.finish_reasons",
        skip_serializing_if = "Vec::is_empty"
    )]
    pub finish_reasons: Vec<String>,
    /// Prompt tokens served from the prefix cache = the engine's resume
    /// position for this request. The conversation-threading signal (§8.6).
    #[serde(
        rename = "paddock.prefix_resume_pos",
        skip_serializing_if = "Option::is_none"
    )]
    pub prefix_resume_pos: Option<u64>,
    /// Edge-measured prompt build: template render + tokenization. Multi-round
    /// requests (agent loops, n>1) sum their rounds - as do all phases below.
    #[serde(
        rename = "paddock.tokenize_ms",
        skip_serializing_if = "Option::is_none"
    )]
    pub tokenize_ms: Option<u64>,
    /// Engine-measured scheduler wait (submit -> slot admission).
    #[serde(rename = "paddock.queue_ms", skip_serializing_if = "Option::is_none")]
    pub queue_ms: Option<u64>,
    /// Engine-measured prefill wall clock (admission -> prompt prefilled).
    #[serde(rename = "paddock.prefill_ms", skip_serializing_if = "Option::is_none")]
    pub prefill_ms: Option<u64>,
    /// Engine-measured decode wall clock (prefill done -> finish).
    #[serde(rename = "paddock.decode_ms", skip_serializing_if = "Option::is_none")]
    pub decode_ms: Option<u64>,
    /// Speculative tokens drafted / accepted across the request's rounds.
    /// Absent = the request never rode spec decode.
    #[serde(
        rename = "paddock.spec_drafted",
        skip_serializing_if = "Option::is_none"
    )]
    pub spec_drafted: Option<u64>,
    #[serde(
        rename = "paddock.spec_accepted",
        skip_serializing_if = "Option::is_none"
    )]
    pub spec_accepted: Option<u64>,
    /// Page-granular KV footprint at completion (max across rounds); absent
    /// on the serial (non-paged) engine.
    #[serde(rename = "paddock.kv_pages", skip_serializing_if = "Option::is_none")]
    pub kv_pages: Option<u64>,
    /// Edge-measured time to first response-body byte. For streaming this is
    /// the client-visible TTFT (queue + tokenize + prefill + first decode).
    #[serde(rename = "paddock.ttft_ms", skip_serializing_if = "Option::is_none")]
    pub ttft_ms: Option<u64>,
    /// Output tokens over the post-first-byte window - the edge view of decode
    /// rate. Streaming responses only (non-streamed compute is not separable
    /// at the edge).
    #[serde(
        rename = "paddock.decode_tok_s",
        skip_serializing_if = "Option::is_none"
    )]
    pub decode_tok_s: Option<f64>,
    /// FNV-1a of the presented bearer key - a stable grouping key, not a
    /// secret store (the key itself never lands in a record).
    #[serde(
        rename = "paddock.api_key_hash",
        skip_serializing_if = "Option::is_none"
    )]
    pub api_key_hash: Option<String>,
    /// Who sent it (`x-paddock-origin`: batch | studio); absent = live
    /// traffic. The forecaster's own-exhaust guard.
    #[serde(rename = "paddock.origin", skip_serializing_if = "Option::is_none")]
    pub origin: Option<&'static str>,
    /// The response body was dropped before it ran to end - the client
    /// vanished mid-stream. Without this bit an abandoned stream reads as a
    /// clean 200 with a short output count.
    #[serde(
        rename = "paddock.client_disconnected",
        skip_serializing_if = "is_false"
    )]
    pub client_disconnected: bool,
}

fn is_false(b: &bool) -> bool {
    !*b
}

/// Handler-filled slots for the in-flight request. Cheap Mutex - a handful of
/// short writes per request, never on the token loop.
#[derive(Debug, Default)]
struct Cells {
    model: Option<String>,
    user: Option<String>,
    input_tokens: Option<u64>,
    output_tokens: Option<u64>,
    cached_tokens: Option<u64>,
    finish_reasons: Vec<String>,
    /// Accumulated engine RunStats (agent loops / n>1 sum their generations;
    /// kv_pages keeps the max - a footprint, not a flow).
    tokenize_ms: Option<u64>,
    queue_ms: Option<u64>,
    prefill_ms: Option<u64>,
    decode_ms: Option<u64>,
    spec_drafted: Option<u64>,
    spec_accepted: Option<u64>,
    kv_pages: Option<u64>,
}

/// Cloneable handle the middleware plants in the request extensions and the
/// handlers write through. Default (no middleware - unit tests, disabled
/// ring) is a no-op, so handlers call unconditionally.
#[derive(Debug, Clone, Default)]
pub struct EventScope(Option<Arc<Mutex<Cells>>>);

impl EventScope {
    fn live() -> Self {
        EventScope(Some(Arc::new(Mutex::new(Cells::default()))))
    }

    fn with(&self, f: impl FnOnce(&mut Cells)) {
        if let Some(c) = &self.0
            && let Ok(mut cells) = c.lock()
        {
            f(&mut cells);
        }
    }

    pub fn model(&self, id: &str) {
        self.with(|c| c.model = Some(id.to_owned()));
    }

    pub fn user(&self, u: Option<&str>) {
        if let Some(u) = u {
            self.with(|c| c.user = Some(u.to_owned()));
        }
    }

    /// Final token accounting (last write wins - agent loops report totals).
    pub fn usage(&self, input: usize, output: usize) {
        self.with(|c| {
            c.input_tokens = Some(input as u64);
            c.output_tokens = Some(output as u64);
        });
    }

    /// Prompt tokens served from the prefix cache (the resume position).
    pub fn cached(&self, n: usize) {
        self.with(|c| c.cached_tokens = Some(n as u64));
    }

    /// Edge prompt-build time: template render + tokenization. Agent loops
    /// re-prepare every round - rounds sum, like the engine phases.
    pub fn tokenized(&self, d: Duration) {
        self.with(|c| {
            c.tokenize_ms = Some(c.tokenize_ms.unwrap_or(0) + d.as_millis() as u64);
        });
    }

    /// Append one choice's finish reason.
    pub fn finish(&self, reason: &str) {
        self.with(|c| c.finish_reasons.push(reason.to_owned()));
    }

    /// Fold one generation's engine-measured RunStats into the record.
    /// Durations and spec counters SUM across a request's generations (agent
    /// rounds, n>1 choices) - total engine time attributable to the request;
    /// kv_pages keeps the MAX (a footprint, not a flow). Spec fields stay
    /// absent while drafted is 0, so "never rode spec" is distinguishable
    /// from "rode spec, nothing accepted".
    pub fn phases(&self, s: &paddock_engine::service::RunStats) {
        self.with(|c| {
            let add = |slot: &mut Option<u64>, v: u64| *slot = Some(slot.unwrap_or(0) + v);
            add(&mut c.queue_ms, s.queued_ms as u64);
            add(&mut c.prefill_ms, s.prefill_ms as u64);
            add(&mut c.decode_ms, s.decode_ms as u64);
            if s.spec_drafted > 0 {
                add(&mut c.spec_drafted, s.spec_drafted as u64);
                add(&mut c.spec_accepted, s.spec_accepted as u64);
            }
            if s.kv_pages > 0 {
                c.kv_pages = Some(c.kv_pages.unwrap_or(0).max(s.kv_pages as u64));
            }
        });
    }

    fn take(&self) -> Cells {
        match &self.0 {
            Some(c) => c
                .lock()
                .map(|mut c| std::mem::take(&mut *c))
                .unwrap_or_default(),
            None => Cells::default(),
        }
    }
}

const RING_CAP: usize = 4096;

struct Inner {
    buf: VecDeque<Arc<EventRecord>>,
    /// Sequence the next record will get; oldest held = next_seq - buf.len().
    next_seq: u64,
}

/// The bounded in-memory ring. Fixed RAM, zero disk, overwrite-oldest.
pub struct EventRing {
    enabled: bool,
    inner: Mutex<Inner>,
    notify: tokio::sync::Notify,
    /// Cumulative records that fell off the tail (for the ops counter; a
    /// subscriber's per-read `dropped` is computed from sequences).
    overwritten: AtomicU64,
}

impl EventRing {
    pub fn new() -> Arc<Self> {
        Arc::new(EventRing {
            enabled: true,
            inner: Mutex::new(Inner {
                buf: VecDeque::with_capacity(RING_CAP),
                next_seq: 0,
            }),
            notify: tokio::sync::Notify::new(),
            overwritten: AtomicU64::new(0),
        })
    }

    /// `--no-events`: pushes are dropped, the admin endpoint reports disabled,
    /// identify omits the "events" capability.
    pub fn disabled() -> Arc<Self> {
        Arc::new(EventRing {
            enabled: false,
            inner: Mutex::new(Inner {
                buf: VecDeque::new(),
                next_seq: 0,
            }),
            notify: tokio::sync::Notify::new(),
            overwritten: AtomicU64::new(0),
        })
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }

    fn push(&self, mut record: EventRecord) {
        if !self.enabled {
            return;
        }
        {
            let Ok(mut inner) = self.inner.lock() else {
                return;
            };
            record.seq = inner.next_seq;
            inner.next_seq += 1;
            inner.buf.push_back(Arc::new(record));
            if inner.buf.len() > RING_CAP {
                inner.buf.pop_front();
                self.overwritten.fetch_add(1, Relaxed);
            }
        }
        self.notify.notify_waiters();
    }

    /// Records at sequence ≥ `from`, up to `max`. Returns
    /// (dropped-before-from, next-cursor, records) - `dropped > 0` means the
    /// reader fell off the ring's tail and lost exactly that many records.
    pub fn since(&self, from: u64, max: usize) -> (u64, u64, Vec<Arc<EventRecord>>) {
        let Ok(inner) = self.inner.lock() else {
            return (0, from, Vec::new());
        };
        let oldest = inner.next_seq - inner.buf.len() as u64;
        let dropped = oldest.saturating_sub(from);
        let start = from.max(oldest);
        let skip = (start - oldest) as usize;
        let out: Vec<Arc<EventRecord>> = inner.buf.iter().skip(skip).take(max).cloned().collect();
        let next = start + out.len() as u64;
        (dropped, next, out)
    }

    /// Long-poll form of [`since`]: waits up to `wait` for at least one record
    /// past `from` before returning (immediately when records already exist).
    pub async fn wait_since(
        &self,
        from: u64,
        max: usize,
        wait: Duration,
    ) -> (u64, u64, Vec<Arc<EventRecord>>) {
        let deadline = tokio::time::Instant::now() + wait;
        loop {
            // Arm the notification before checking, so a push between the check
            // and the await can't be missed.
            let notified = self.notify.notified();
            let (dropped, next, events) = self.since(from, max);
            if !events.is_empty() || dropped > 0 {
                return (dropped, next, events);
            }
            let now = tokio::time::Instant::now();
            if now >= deadline {
                return (dropped, next, events);
            }
            tokio::select! {
                _ = notified => {}
                _ = tokio::time::sleep_until(deadline) => {}
            }
        }
    }
}

/// FNV-1a 64 - stable across restarts (unlike SipHash's random keys), good
/// enough for a grouping key, deliberately not cryptographic.
fn fnv1a64(s: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    h
}

/// Default session headers (llama-swap's default list - the LiteLLM/agent
/// ecosystem already sends these).
pub fn default_session_headers() -> Vec<String> {
    vec!["x-session-id".to_owned(), "x-litellm-session-id".to_owned()]
}

/// Everything the finalizer needs, carried by the response body.
struct Pending {
    ring: Arc<EventRing>,
    /// The second sink off this middleware. Independently gated:
    /// `--no-events` must not kill `/metrics`, nor `--no-metrics` the ring -
    /// both handles no-op internally when disabled.
    metrics: Arc<crate::metrics::Metrics>,
    scope: EventScope,
    endpoint: String,
    origin: crate::metrics::Origin,
    request_id: String,
    traceparent: Option<String>,
    session_id: Option<String>,
    api_key_hash: Option<String>,
    ts_ms: u64,
    t0: Instant,
    status: u16,
    stream: bool,
}

impl Pending {
    fn finalize(self, ttft: Option<Duration>, completed: bool) {
        let duration = self.t0.elapsed();
        let cells = self.scope.take();
        let ttft_ms = ttft.map(|d| d.as_millis() as u64);
        // Edge decode rate: tokens over the post-first-byte window. Streaming
        // only - a non-streamed response computes everything before byte one.
        let decode_tok_s = match (self.stream, cells.output_tokens, ttft) {
            (true, Some(out), Some(t)) if out > 0 && duration > t => {
                Some(out as f64 / (duration - t).as_secs_f64())
            }
            _ => None,
        };
        self.metrics.observe(&crate::metrics::Observation {
            path: &self.endpoint,
            origin: self.origin,
            status: self.status,
            disconnected: !completed,
            model: cells.model.as_deref(),
            duration,
            ttft,
            decode_ms: cells.decode_ms,
            input_tokens: cells.input_tokens,
            output_tokens: cells.output_tokens,
            cached_tokens: cells.cached_tokens,
            spec_drafted: cells.spec_drafted,
            spec_accepted: cells.spec_accepted,
            trace: self
                .traceparent
                .as_deref()
                .and_then(crate::metrics::parse_traceparent),
        });
        self.ring.push(EventRecord {
            seq: 0, // assigned by push
            ts_ms: self.ts_ms,
            endpoint: self.endpoint,
            status: self.status,
            duration_ms: duration.as_millis() as u64,
            stream: self.stream,
            request_id: self.request_id,
            traceparent: self.traceparent,
            session_id: self.session_id,
            user: cells.user,
            model: cells.model,
            input_tokens: cells.input_tokens,
            output_tokens: cells.output_tokens,
            finish_reasons: cells.finish_reasons,
            prefix_resume_pos: cells.cached_tokens,
            tokenize_ms: cells.tokenize_ms,
            queue_ms: cells.queue_ms,
            prefill_ms: cells.prefill_ms,
            decode_ms: cells.decode_ms,
            spec_drafted: cells.spec_drafted,
            spec_accepted: cells.spec_accepted,
            kv_pages: cells.kv_pages,
            ttft_ms,
            decode_tok_s,
            api_key_hash: self.api_key_hash,
            origin: match self.origin {
                crate::metrics::Origin::Live => None,
                o => Some(match o {
                    crate::metrics::Origin::Batch => "batch",
                    _ => "studio",
                }),
            },
            client_disconnected: !completed,
        });
    }
}

/// Response body that timestamps its first data frame (edge TTFT) and emits
/// the event record when dropped - i.e. when the body finished streaming, the
/// client disconnected, or the response was discarded.
struct RecordingBody {
    inner: axum::body::Body,
    pending: Option<Pending>,
    first_frame: Option<Duration>,
    /// The stream was polled to its natural end (`Ready(None)`). A body
    /// dropped without reaching it - and not already at end-of-stream - is a
    /// client that vanished mid-response: the `disconnect` accounting bit.
    completed: bool,
}

impl http_body::Body for RecordingBody {
    type Data = axum::body::Bytes;
    type Error = axum::Error;

    fn poll_frame(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Result<http_body::Frame<Self::Data>, Self::Error>>> {
        // axum's Body is Unpin (boxed inner stream) - safe projection.
        let this = self.get_mut();
        let poll = std::pin::Pin::new(&mut this.inner).poll_frame(cx);
        match &poll {
            std::task::Poll::Ready(None) => this.completed = true,
            std::task::Poll::Ready(Some(Ok(f))) => {
                if this.first_frame.is_none()
                    && f.is_data()
                    && let Some(p) = &this.pending
                {
                    this.first_frame = Some(p.t0.elapsed());
                }
            }
            _ => {}
        }
        poll
    }

    fn is_end_stream(&self) -> bool {
        self.inner.is_end_stream()
    }

    fn size_hint(&self) -> http_body::SizeHint {
        self.inner.size_hint()
    }
}

impl Drop for RecordingBody {
    fn drop(&mut self) {
        if let Some(p) = self.pending.take() {
            // A fixed-size body (plain JSON) may be finished off is_end_stream
            // without a final Ready(None) poll - that is a completion too;
            // only a drop with bytes still owed is a disconnect.
            let completed = self.completed || http_body::Body::is_end_stream(&self.inner);
            p.finalize(self.first_frame, completed);
        }
    }
}

/// A client-supplied request id we'll echo: visible ASCII, sane length.
fn valid_request_id(v: &axum::http::HeaderValue) -> Option<String> {
    let s = v.to_str().ok()?;
    (!s.is_empty() && s.len() <= 120 && s.bytes().all(|b| (0x21..=0x7e).contains(&b)))
        .then(|| s.to_owned())
}

/// Middleware: correlation + timing + the record push for inference paths.
/// Sits outside drain/auth so refused requests (503/401/429) get records too.
pub async fn events_mw(
    axum::extract::State(state): axum::extract::State<Arc<crate::routes::AppState>>,
    mut req: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    // Two sinks, two independent gates: `--no-events` must not silently kill
    // `/metrics`, nor `--no-metrics` the ring. Skip only when both are off.
    if (!state.events.enabled() && !state.metrics.enabled())
        || !crate::drain::is_inference_path(req.uri().path())
    {
        return next.run(req).await;
    }
    let t0 = Instant::now();
    let ts_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);
    let headers = req.headers();
    let request_id = headers
        .get("x-request-id")
        .and_then(valid_request_id)
        .unwrap_or_else(|| format!("req_{}", uuid::Uuid::new_v4().simple()));
    let traceparent = headers
        .get("traceparent")
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned);
    let session_id = state.session_headers.iter().find_map(|h| {
        headers
            .get(h.as_str())
            .and_then(|v| v.to_str().ok())
            .map(str::to_owned)
    });
    let api_key_hash = headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.strip_prefix("Bearer "))
        .map(|k| format!("{:016x}", fnv1a64(k)));
    let origin = crate::metrics::Origin::from_header(
        headers
            .get("x-paddock-origin")
            .and_then(|v| v.to_str().ok()),
    );
    let endpoint = req.uri().path().to_owned();

    let scope = EventScope::live();
    req.extensions_mut().insert(scope.clone());

    let mut res = next.run(req).await;

    if let Ok(v) = axum::http::HeaderValue::from_str(&request_id) {
        res.headers_mut().insert("x-request-id", v);
    }
    let stream = res
        .headers()
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|ct| ct.starts_with("text/event-stream"));
    let pending = Pending {
        ring: state.events.clone(),
        metrics: state.metrics.clone(),
        scope,
        endpoint,
        origin,
        request_id,
        traceparent,
        session_id,
        api_key_hash,
        ts_ms,
        t0,
        status: res.status().as_u16(),
        stream,
    };
    let (parts, body) = res.into_parts();
    axum::response::Response::from_parts(
        parts,
        axum::body::Body::new(RecordingBody {
            inner: body,
            pending: Some(pending),
            first_frame: None,
            completed: false,
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(endpoint: &str) -> EventRecord {
        EventRecord {
            seq: 0,
            ts_ms: 1,
            endpoint: endpoint.to_owned(),
            status: 200,
            duration_ms: 5,
            stream: false,
            request_id: "req_x".into(),
            traceparent: None,
            session_id: None,
            user: None,
            model: None,
            input_tokens: None,
            output_tokens: None,
            finish_reasons: Vec::new(),
            prefix_resume_pos: None,
            tokenize_ms: None,
            queue_ms: None,
            prefill_ms: None,
            decode_ms: None,
            spec_drafted: None,
            spec_accepted: None,
            kv_pages: None,
            ttft_ms: None,
            decode_tok_s: None,
            api_key_hash: None,
            origin: None,
            client_disconnected: false,
        }
    }

    #[test]
    fn ring_assigns_sequences_and_reports_drops() {
        let ring = EventRing::new();
        for i in 0..(RING_CAP + 10) {
            ring.push(record(&format!("/v1/x{i}")));
        }
        // Reader starting at 0 fell off the tail: exactly 10 dropped.
        let (dropped, next, events) = ring.since(0, 100);
        assert_eq!(dropped, 10);
        assert_eq!(events.first().unwrap().seq, 10);
        assert_eq!(next, 110);
        // Resume from the cursor: contiguous, no drops.
        let (dropped, next2, events) = ring.since(next, 100_000);
        assert_eq!(dropped, 0);
        assert_eq!(events.len(), RING_CAP - 100);
        assert_eq!(next2, (RING_CAP + 10) as u64);
        // At the head: nothing new.
        let (dropped, _, events) = ring.since(next2, 10);
        assert_eq!((dropped, events.len()), (0, 0));
    }

    #[test]
    fn disabled_ring_drops_everything() {
        let ring = EventRing::disabled();
        ring.push(record("/v1/chat/completions"));
        let (dropped, next, events) = ring.since(0, 10);
        assert_eq!((dropped, next, events.len()), (0, 0, 0));
    }

    #[tokio::test]
    async fn wait_since_returns_on_push() {
        let ring = EventRing::new();
        let r2 = ring.clone();
        let waiter =
            tokio::spawn(async move { r2.wait_since(0, 10, Duration::from_secs(5)).await });
        tokio::time::sleep(Duration::from_millis(20)).await;
        ring.push(record("/v1/messages"));
        let (dropped, next, events) = waiter.await.unwrap();
        assert_eq!((dropped, next, events.len()), (0, 1, 1));
        assert_eq!(events[0].endpoint, "/v1/messages");
    }

    #[test]
    fn semconv_names_on_the_wire() {
        let mut r = record("/v1/chat/completions");
        r.model = Some("qwen3.5-9b".into());
        r.input_tokens = Some(100);
        r.finish_reasons.push("stop".into());
        let v = serde_json::to_value(&r).unwrap();
        assert_eq!(v["gen_ai.request.model"], "qwen3.5-9b");
        assert_eq!(v["gen_ai.usage.input_tokens"], 100);
        assert_eq!(v["gen_ai.response.finish_reasons"][0], "stop");
        // Unset options stay off the wire (records stay ~0.5 KB).
        assert!(v.get("paddock.ttft_ms").is_none());
    }

    #[test]
    fn phases_sum_rounds_and_keep_kv_max() {
        let ring = EventRing::new();
        let scope = EventScope::live();
        // two agent rounds: durations/spec sum, kv_pages keeps the max
        scope.phases(&paddock_engine::service::RunStats {
            queued_ms: 5,
            prefill_ms: 100,
            decode_ms: 900,
            spec_drafted: 40,
            spec_accepted: 30,
            kv_pages: 12,
        });
        scope.phases(&paddock_engine::service::RunStats {
            queued_ms: 1,
            prefill_ms: 20,
            decode_ms: 400,
            spec_drafted: 10,
            spec_accepted: 8,
            kv_pages: 9,
        });
        let p = Pending {
            ring: ring.clone(),
            metrics: crate::metrics::Metrics::disabled(),
            scope,
            origin: Default::default(),
            endpoint: "/v1/responses".into(),
            request_id: "req_p".into(),
            traceparent: None,
            session_id: None,
            api_key_hash: None,
            ts_ms: 0,
            t0: Instant::now(),
            status: 200,
            stream: false,
        };
        p.finalize(None, true);
        let (_, _, events) = ring.since(0, 10);
        let r = &events[0];
        assert_eq!(
            (r.queue_ms, r.prefill_ms, r.decode_ms),
            (Some(6), Some(120), Some(1300))
        );
        assert_eq!((r.spec_drafted, r.spec_accepted), (Some(50), Some(38)));
        assert_eq!(r.kv_pages, Some(12));
        let v = serde_json::to_value(r).unwrap();
        assert_eq!(v["paddock.prefill_ms"], 120);
        assert_eq!(v["paddock.spec_accepted"], 38);
        assert_eq!(v["paddock.kv_pages"], 12);
    }

    #[test]
    fn no_spec_stays_off_the_wire() {
        let ring = EventRing::new();
        let scope = EventScope::live();
        // a request that never rode spec on the serial (non-paged) engine
        scope.phases(&paddock_engine::service::RunStats {
            queued_ms: 0,
            prefill_ms: 10,
            decode_ms: 50,
            spec_drafted: 0,
            spec_accepted: 0,
            kv_pages: 0,
        });
        let p = Pending {
            ring: ring.clone(),
            metrics: crate::metrics::Metrics::disabled(),
            scope,
            origin: Default::default(),
            endpoint: "/v1/completions".into(),
            request_id: "req_q".into(),
            traceparent: None,
            session_id: None,
            api_key_hash: None,
            ts_ms: 0,
            t0: Instant::now(),
            status: 200,
            stream: false,
        };
        p.finalize(None, true);
        let (_, _, events) = ring.since(0, 10);
        let v = serde_json::to_value(&events[0]).unwrap();
        // "never rode spec" / "no paged pool" must be absent, not zero
        assert!(v.get("paddock.spec_drafted").is_none());
        assert!(v.get("paddock.kv_pages").is_none());
        assert_eq!(v["paddock.decode_ms"], 50);
    }

    #[test]
    fn scope_fills_flow_into_the_record() {
        let ring = EventRing::new();
        let scope = EventScope::live();
        scope.model("m");
        scope.usage(10, 5);
        scope.cached(8);
        scope.finish("stop");
        let p = Pending {
            ring: ring.clone(),
            metrics: crate::metrics::Metrics::disabled(),
            scope,
            origin: Default::default(),
            endpoint: "/v1/chat/completions".into(),
            request_id: "req_1".into(),
            traceparent: None,
            session_id: Some("s1".into()),
            api_key_hash: None,
            ts_ms: 0,
            t0: Instant::now(),
            status: 200,
            stream: false,
        };
        p.finalize(None, true);
        let (_, _, events) = ring.since(0, 10);
        let r = &events[0];
        assert_eq!(r.model.as_deref(), Some("m"));
        assert_eq!((r.input_tokens, r.output_tokens), (Some(10), Some(5)));
        assert_eq!(r.prefix_resume_pos, Some(8));
        assert_eq!(r.session_id.as_deref(), Some("s1"));
        assert_eq!(r.finish_reasons, ["stop"]);
    }
}
