//! Admin-surface server: accept local connections (named pipe / UDS) and serve
//! an axum `Router` over HTTP/1 on each. The router is supplied by the runner
//! - this module owns only transport + security.

use std::io;

use hyper_util::rt::TokioIo;
use hyper_util::service::TowerToHyperService;

/// Serve `app` on the admin endpoint for `port`, forever. Errors out only if
/// the endpoint can't be created (name squatted, no runtime dir) - per-
/// connection failures are logged and survived. Callers spawn this; a failure
/// must not take serving down (the runner is a complete product without a
/// manager attached).
pub async fn serve(port: u16, app: axum::Router) -> io::Result<()> {
    #[cfg(windows)]
    {
        serve_windows(port, app).await
    }
    #[cfg(unix)]
    {
        serve_unix(port, app).await
    }
}

#[cfg(windows)]
async fn serve_windows(port: u16, app: axum::Router) -> io::Result<()> {
    use tokio::net::windows::named_pipe::ServerOptions;

    let name = crate::pipe_name(port);
    let security = crate::winsec::PipeSecurity::user_only()?;
    // first_pipe_instance: if something already owns this name (a squatter, or
    // a runner that failed to die), creation fails loudly instead of silently
    // queueing behind it.
    let mut server = unsafe {
        ServerOptions::new()
            .first_pipe_instance(true)
            .reject_remote_clients(true)
            .create_with_security_attributes_raw(&name, security.as_ptr())
    }?;
    tracing::info!(pipe = %name, "admin surface listening (named pipe, user-only DACL)");
    loop {
        server.connect().await?;
        // Hand the connected instance to a task; stand up the next instance
        // first so a second client never sees "pipe busy" longer than needed.
        let connected = server;
        server = unsafe {
            ServerOptions::new()
                .reject_remote_clients(true)
                .create_with_security_attributes_raw(&name, security.as_ptr())
        }?;
        let svc = TowerToHyperService::new(app.clone());
        tokio::spawn(async move {
            if let Err(e) = hyper::server::conn::http1::Builder::new()
                .serve_connection(TokioIo::new(connected), svc)
                .await
            {
                // A client hanging up between keep-alive requests is the
                // NORMAL end of every manager poll (collector, reconciler,
                // fleet health) - trace, or it floods the log many times a
                // minute. Anything else is a real transport oddity.
                if e.is_incomplete_message() {
                    tracing::trace!(error = %e, "admin client disconnected");
                } else {
                    tracing::debug!(error = %e, "admin connection ended with error");
                }
            }
        });
    }
}

#[cfg(unix)]
async fn serve_unix(port: u16, app: axum::Router) -> io::Result<()> {
    use std::os::unix::fs::{DirBuilderExt, PermissionsExt};
    use tokio::net::UnixListener;

    let dir = crate::runtime_dir();
    if !dir.exists() {
        std::fs::DirBuilder::new()
            .recursive(true)
            .mode(0o700)
            .create(&dir)?;
    }
    // Belt-and-braces if the dir pre-existed with wider permissions.
    std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700))?;
    let path = crate::socket_path(port);
    // A leftover socket from a dead runner would EADDRINUSE. Removing it is
    // safe: a live runner on this port would have failed our TCP bind first,
    // so a present-but-stale socket can only be a corpse.
    let _ = std::fs::remove_file(&path);
    let listener = UnixListener::bind(&path)?;
    tracing::info!(socket = %path.display(), "admin surface listening (unix socket, 0700 dir)");
    loop {
        let (stream, _addr) = listener.accept().await?;
        let svc = TowerToHyperService::new(app.clone());
        tokio::spawn(async move {
            if let Err(e) = hyper::server::conn::http1::Builder::new()
                .serve_connection(TokioIo::new(stream), svc)
                .await
            {
                // see the windows branch: keep-alive hang-ups are the normal
                // end of every manager poll - trace, not log flood
                if e.is_incomplete_message() {
                    tracing::trace!(error = %e, "admin client disconnected");
                } else {
                    tracing::debug!(error = %e, "admin connection ended with error");
                }
            }
        });
    }
}
