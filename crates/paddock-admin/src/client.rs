//! Admin-surface client - what the manager (and `paddock` CLI verbs) use to
//! talk to a runner's pipe/socket. One connection per call: admin traffic is
//! rare and tiny, so simplicity beats pooling; the events subscription is a
//! resumable long-poll (`since` cursor), so it fits the same model.

use http_body_util::{BodyExt, Full};
use hyper::body::Bytes;
use hyper::{Method, Request, StatusCode};
use hyper_util::rt::TokioIo;
use serde_json::Value;

use crate::types::{
    DrainRequest, DrainState, EventsPage, Health, Identify, ShutdownAck, ShutdownRequest,
    SnapshotsPage,
};

#[derive(Debug, thiserror::Error)]
pub enum AdminError {
    /// Nothing answered on the endpoint - no runner (or it just died).
    #[error("no admin endpoint for port {port}: {source}")]
    Connect { port: u16, source: std::io::Error },
    #[error("admin transport error: {0}")]
    Transport(String),
    /// The endpoint answered but refused/errored the call.
    #[error("admin call failed: {status} {body}")]
    Status { status: StatusCode, body: String },
    #[error("admin response was not the expected shape: {0}")]
    Decode(String),
}

/// Client for one runner's admin surface, addressed by inference port.
#[derive(Debug, Clone, Copy)]
pub struct AdminClient {
    port: u16,
}

impl AdminClient {
    pub fn new(port: u16) -> Self {
        Self { port }
    }

    pub async fn identify(&self) -> Result<Identify, AdminError> {
        self.get_json("/v1/identify").await
    }

    pub async fn health(&self) -> Result<Health, AdminError> {
        self.get_json("/v1/health").await
    }

    pub async fn drain(&self, timeout_ms: Option<u64>) -> Result<DrainState, AdminError> {
        self.post_json("/v1/drain", &DrainRequest { timeout_ms })
            .await
    }

    pub async fn shutdown(&self, timeout_ms: Option<u64>) -> Result<ShutdownAck, AdminError> {
        self.post_json("/v1/shutdown", &ShutdownRequest { timeout_ms })
            .await
    }

    /// Rich surface (capability "stats"): the runner's engine self-report.
    pub async fn stats(&self) -> Result<Value, AdminError> {
        self.get_json("/v1/stats").await
    }

    /// Rich surface (capability "events"): records at sequence ≥ `since`.
    /// `wait_ms > 0` long-polls until at least one new record exists. Resume
    /// by passing the returned `next` back as `since`; a non-zero `dropped`
    /// means the reader fell off the ring's tail and lost that many records.
    pub async fn events(
        &self,
        since: u64,
        max: usize,
        wait_ms: u64,
    ) -> Result<EventsPage, AdminError> {
        self.get_json(&format!(
            "/v1/events?since={since}&max={max}&wait_ms={wait_ms}"
        ))
        .await
    }

    /// Rich surface (capability "metrics-snapshots"): one page of the
    /// runner's 1-minute counter self-snapshot ring. Same resume
    /// contract as `events`: pass the returned `next` back as `since`.
    pub async fn metrics_snapshots(
        &self,
        since: u64,
        max: usize,
    ) -> Result<SnapshotsPage, AdminError> {
        self.get_json(&format!("/v1/metrics_snapshots?since={since}&max={max}"))
            .await
    }

    /// Rich surface (capability "metrics"): the Prometheus exposition,
    /// classic text format - Not JSON, returned verbatim. The manager's
    /// usage scrape parses the families it knows out of it.
    pub async fn metrics(&self) -> Result<String, AdminError> {
        let bytes = self.call(Method::GET, "/v1/metrics", None).await?;
        String::from_utf8(bytes.to_vec()).map_err(|e| AdminError::Decode(e.to_string()))
    }

    async fn get_json<T: serde::de::DeserializeOwned>(&self, path: &str) -> Result<T, AdminError> {
        let bytes = self.call(Method::GET, path, None).await?;
        serde_json::from_slice(&bytes).map_err(|e| AdminError::Decode(e.to_string()))
    }

    async fn post_json<B: serde::Serialize, T: serde::de::DeserializeOwned>(
        &self,
        path: &str,
        body: &B,
    ) -> Result<T, AdminError> {
        let payload = serde_json::to_vec(body).map_err(|e| AdminError::Decode(e.to_string()))?;
        let bytes = self.call(Method::POST, path, Some(payload)).await?;
        serde_json::from_slice(&bytes).map_err(|e| AdminError::Decode(e.to_string()))
    }

    async fn call(
        &self,
        method: Method,
        path: &str,
        body: Option<Vec<u8>>,
    ) -> Result<Bytes, AdminError> {
        let stream = connect(self.port).await.map_err(|e| AdminError::Connect {
            port: self.port,
            source: e,
        })?;
        let (mut sender, conn) = hyper::client::conn::http1::handshake(TokioIo::new(stream))
            .await
            .map_err(|e| AdminError::Transport(e.to_string()))?;
        // Drive the connection; it ends when the request/response completes.
        tokio::spawn(async move {
            let _ = conn.await;
        });
        let req = Request::builder()
            .method(method)
            .uri(path)
            // hyper requires a Host header on HTTP/1.1; the value is meaningless
            // on a local pipe.
            .header(hyper::header::HOST, "paddock-admin")
            .header(hyper::header::CONTENT_TYPE, "application/json")
            .body(Full::new(Bytes::from(body.unwrap_or_default())))
            .map_err(|e| AdminError::Transport(e.to_string()))?;
        let res = sender
            .send_request(req)
            .await
            .map_err(|e| AdminError::Transport(e.to_string()))?;
        let status = res.status();
        let bytes = res
            .into_body()
            .collect()
            .await
            .map_err(|e| AdminError::Transport(e.to_string()))?
            .to_bytes();
        if !status.is_success() {
            return Err(AdminError::Status {
                status,
                body: String::from_utf8_lossy(&bytes).into_owned(),
            });
        }
        Ok(bytes)
    }
}

#[cfg(windows)]
async fn connect(port: u16) -> std::io::Result<tokio::net::windows::named_pipe::NamedPipeClient> {
    use tokio::net::windows::named_pipe::ClientOptions;
    let name = crate::pipe_name(port);
    // ERROR_PIPE_BUSY (231): every instance is mid-accept - the server stands
    // a fresh instance up immediately after each connect, so this window is
    // tiny; retry briefly instead of failing.
    const ERROR_PIPE_BUSY: i32 = 231;
    for _ in 0..40 {
        match ClientOptions::new().open(&name) {
            Ok(c) => return Ok(c),
            Err(e) if e.raw_os_error() == Some(ERROR_PIPE_BUSY) => {
                tokio::time::sleep(std::time::Duration::from_millis(25)).await;
            }
            Err(e) => return Err(e),
        }
    }
    Err(std::io::Error::new(
        std::io::ErrorKind::TimedOut,
        "admin pipe stayed busy",
    ))
}

#[cfg(unix)]
async fn connect(port: u16) -> std::io::Result<tokio::net::UnixStream> {
    tokio::net::UnixStream::connect(crate::socket_path(port)).await
}
