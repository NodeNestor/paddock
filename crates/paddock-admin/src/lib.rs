//! The runner **admin surface** - the local control channel between a runner
//! and the manager.
//!
//! Transport is HTTP/1 over a **local-only** endpoint (the Docker precedent):
//! a Windows named pipe (`\\.\pipe\paddock-runner-<port>`, DACL = owning user
//! + SYSTEM only, `PIPE_REJECT_REMOTE_CLIENTS`) or a Unix domain socket
//!   (`$XDG_RUNTIME_DIR/paddock/runner-<port>.sock`, dir mode 0700). The OS *is*
//!   the authentication: same user => full admin, anyone else => can't connect.
//!   **This surface never binds TCP**, and inference API keys never grant admin
//!   ops - the separation is by transport, not policy.
//!
//! The wire contract is versioned (`WIRE_VERSION`) with a **v1-frozen core**:
//! identify + health + drain/shutdown never change shape, so any future
//! manager can recognize and cleanly stop any runner ever shipped. Everything
//! richer (stats, events) is capability-discovered via `identify`.

pub mod client;
pub mod codec;
pub mod server;
pub mod types;
pub mod version;
#[cfg(windows)]
mod winsec;

use std::path::PathBuf;

/// The data root + which rung of the ladder chose it. One resolver for the
/// three distribution modes; manager and runner both link this crate, so the
/// two halves can never disagree about where data lives:
///
///   1. `PADDOCK_DATA` env - explicit override, always wins
///   2. `~/paddock` for a DEV build - an exe under a cargo `target/` keeps the
///      checkout's own models rather than growing a data root inside `target/`
///      that `cargo clean` would eat
///   3. `data\` beside the exe - PORTABLE, and CREATED if absent: a copy of the
///      folder is the whole world, wherever it lands
///   4. the machine root an installer created - `%ProgramData%\Paddock` on
///      Windows, `/var/lib/paddock` elsewhere - the per-box appliance mode
///   5. `~/paddock`, then the cwd - last resorts for an exe somewhere it may
///      not write
///
/// Rung 3 used to require `data\` to already exist, and nothing ever created
/// it but the packaging script. So a portable folder copied without its data
/// subtree - a drag-and-drop that skipped the multi-GB part, a zip tool that
/// dropped an empty dir, a copy taken while SQLite held the -wal open - landed
/// and quietly adopted `%USERPROFILE%\paddock`: on a machine with an
/// existing install, another install's models, servers and database. Exactly
/// what portable mode exists to prevent (found by copying a portable folder
/// somewhere else and reading the startup banner).
///
/// Portable now means what it says: unless you point somewhere else, the
/// program's own folder is where its data lives. Writability is settled by
/// TRYING to create the directory rather than by sniffing the path - a
/// read-only Program Files or a mounted image fails and falls through on its
/// own, with no list of special locations to keep current.
///
/// The source string feeds the startup banner: where data lives must be
/// STATED, not divined from behavior.
pub fn data_root_resolved() -> (PathBuf, &'static str) {
    if let Some(p) = std::env::var_os("PADDOCK_DATA").filter(|p| !p.is_empty()) {
        return (PathBuf::from(p), "PADDOCK_DATA");
    }
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|e| e.parent().map(PathBuf::from));
    if let Some(dir) = exe_dir.as_ref().filter(|d| !in_cargo_target(d)) {
        let portable = dir.join("data");
        // is_dir first so the common case costs one stat, then create for the
        // first run of a fresh copy. `create_dir_all` is idempotent, so the
        // race with a second process starting beside us is a non-event.
        if portable.is_dir() || std::fs::create_dir_all(&portable).is_ok() {
            return (portable, "portable");
        }
    }
    let machine = if cfg!(windows) {
        std::env::var_os("ProgramData").map(|p| PathBuf::from(p).join("Paddock"))
    } else {
        Some(PathBuf::from("/var/lib/paddock"))
    };
    if let Some(m) = machine
        && m.is_dir()
    {
        return (m, "installed");
    }
    match std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME")) {
        Some(home) => (PathBuf::from(home).join("paddock"), "home"),
        None => (PathBuf::from("."), "cwd"),
    }
}

/// Is this exe a cargo build artifact - `target/{debug,release}/paddock.exe`?
///
/// Asked so a `cargo run` does not become "portable" and start a fresh, empty
/// data root inside `target/`, orphaning the checkout's models and losing the
/// lot to the next `cargo clean`.
///
/// The test is cargo's own marker, not a path-name guess: cargo writes a
/// `CACHEDIR.TAG` at the root of every target dir (the freedesktop
/// cache-directory convention, so backup tools skip it). A user directory that
/// happens to be called "release" has no such file; a target dir renamed by
/// `CARGO_TARGET_DIR` still does.
fn in_cargo_target(exe_dir: &std::path::Path) -> bool {
    exe_dir
        .parent()
        .is_some_and(|p| p.join("CACHEDIR.TAG").is_file())
}

/// [`data_root_resolved`] without the provenance tag.
pub fn data_root() -> PathBuf {
    data_root_resolved().0
}

/// The pipe name (Windows) for a runner's admin surface, keyed by its
/// inference port - one runner per port, one pipe per runner.
#[cfg(windows)]
pub fn pipe_name(port: u16) -> String {
    format!(r"\\.\pipe\paddock-runner-{port}")
}

/// Per-boot runtime dir for admin sockets (Unix): `$XDG_RUNTIME_DIR/paddock/`,
/// falling back to `~/paddock/runtime/` (created 0700).
#[cfg(unix)]
pub fn runtime_dir() -> PathBuf {
    if let Some(x) = std::env::var_os("XDG_RUNTIME_DIR") {
        return PathBuf::from(x).join("paddock");
    }
    match std::env::var_os("HOME") {
        Some(h) => PathBuf::from(h).join("paddock").join("runtime"),
        None => PathBuf::from("./paddock-runtime"),
    }
}

/// The socket path (Unix) for a runner's admin surface.
#[cfg(unix)]
pub fn socket_path(port: u16) -> PathBuf {
    runtime_dir().join(format!("runner-{port}.sock"))
}

/// Enumerate ports with a present admin endpoint on this host - the manager's
/// startup reconciliation input. Presence ≠ liveness: a Unix socket can
/// be stale (its runner died); `identify` is the liveness check. Windows pipes
/// disappear with their process, so presence there is close to liveness.
pub fn enumerate() -> Vec<u16> {
    let mut ports = Vec::new();
    #[cfg(windows)]
    {
        // The pipe namespace is enumerable as a directory listing.
        if let Ok(entries) = std::fs::read_dir(r"\\.\pipe\") {
            for e in entries.flatten() {
                let name = e.file_name().to_string_lossy().into_owned();
                if let Some(p) = name.strip_prefix("paddock-runner-")
                    && let Ok(port) = p.parse::<u16>()
                {
                    ports.push(port);
                }
            }
        }
    }
    #[cfg(unix)]
    {
        if let Ok(entries) = std::fs::read_dir(runtime_dir()) {
            for e in entries.flatten() {
                let name = e.file_name().to_string_lossy().into_owned();
                if let Some(rest) = name.strip_prefix("runner-")
                    && let Some(p) = rest.strip_suffix(".sock")
                    && let Ok(port) = p.parse::<u16>()
                {
                    ports.push(port);
                }
            }
        }
    }
    ports.sort_unstable();
    ports
}

/// Default `RUST_LOG` when the environment does not set one.
///
/// Ours at `debug`, the plumbing at `warn`. The plumbing part is the point:
/// `rmcp` logs at INFO what an SDK reasonably logs - service lifecycle, task
/// cancellation, and the full JSON-RPC response body of every call. Embedded in
/// a host that talks to several MCP servers, that buries our own lines under
/// hundreds of characters of protocol per tool call, and a log nobody can read
/// is a log nobody reads. Their default is fine for a standalone client; it is
/// wrong for us, and the level is ours to choose.
///
/// `hyper`/`h2`/`tower_http` get the same treatment for the same reason -
/// per-connection and per-frame chatter that says nothing an operator acts on.
/// Anything genuinely wrong still arrives, because they are capped at warn, not
/// silenced.
///
/// One constant because there were three copies of this string (manager main,
/// runner startup, runner service) and they had already begun to matter
/// separately. `RUST_LOG` still overrides everything.
pub mod logging;

pub const DEFAULT_LOG_FILTER: &str =
    "info,paddock=debug,rmcp=warn,hyper=warn,h2=warn,tower_http=warn";

#[cfg(test)]
mod data_root_tests {
    use super::in_cargo_target;

    /// The dev carve-out keys on cargo's own CACHEDIR.TAG, so a directory that
    /// merely LOOKS like a build output is not one. This is the difference
    /// between "portable" and "a data root cargo clean will delete".
    #[test]
    fn cargo_target_is_recognised_by_its_marker_not_its_name() {
        let tmp = std::env::temp_dir().join(format!("pd-dr-{}", std::process::id()));
        let target = tmp.join("target");
        let release = target.join("release");
        std::fs::create_dir_all(&release).unwrap();

        // A folder called target/release with no marker is just a folder: a
        // user is entitled to unzip paddock into one and get portable mode.
        assert!(!in_cargo_target(&release));

        std::fs::write(target.join("CACHEDIR.TAG"), b"Signature: 8a477f597d28d172").unwrap();
        assert!(in_cargo_target(&release));

        // And the marker only counts one level up - the exe's own directory
        // holding one would mean something else entirely.
        let stray = tmp.join("elsewhere");
        std::fs::create_dir_all(&stray).unwrap();
        std::fs::write(stray.join("CACHEDIR.TAG"), b"Signature: 8a477f597d28d172").unwrap();
        assert!(!in_cargo_target(&stray));

        std::fs::remove_dir_all(&tmp).ok();
    }
}
