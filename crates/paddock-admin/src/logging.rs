//! One place where every paddock process decides what it logs and where it goes.
//!
//! There were four copies of the subscriber setup - manager main, runner
//! startup, and both arms of runner service - and they had already drifted:
//! the systemd arm still carried a hand-written `"info,paddock=debug"` and so
//! lost the plumbing caps that [`crate::DEFAULT_LOG_FILTER`] exists to apply,
//! meaning a service-run runner logged the MCP protocol chatter that the same
//! binary run by hand does not. Consolidating the filter string alone did not
//! prevent that, because the setup around it was still copied. This is the
//! whole setup, once.
//!
//! ## Terminal vs file
//!
//! Terminal gets ANSI, the file never does - colour escapes in a log are
//! garbage to `grep`, to a text editor, and to anything a user pastes into a
//! bug report. Both carry the same events at the same level: a file that
//! disagrees with what the operator watched scroll past is worse than no file.
//!
//! That promise held for the tee and was broken for the RUNNER, because a
//! runner's stdout is a file: the supervisor opens `runner-<port>.log` and
//! passes the handle at spawn (§11.3). tracing-subscriber's fmt layer does not
//! tty-detect - ANSI is on unless you say otherwise - so the "terminal" copy
//! wrote colour into the log, and every runner line then began with `\x1b[2m`.
//! The Studio's log viewer parses `<time> <LEVEL> <module>:` from the start of
//! the line, so not one runner line parsed: no clock, no level chip, and the
//! level filter had nothing to select on. So ask stdout what
//! it is - the same question the startup banners already ask before colouring.
//!
//! CLI verbs (`paddock ps`, `paddock ls`) deliberately do not pass a tee. Their
//! output is `println!` and belongs to the invoking terminal; a `ps` must never
//! append to the running service's log.

use std::io::IsTerminal;
use std::path::Path;

/// Rotate at startup when the existing log is already this big.
///
/// Size, not per-run: the RUNNER logs rotate per serving generation because a
/// port gets reused by a different model and mixing their lifetimes is
/// genuinely confusing. The manager has no such identity to confuse, and its
/// restarts are frequent on a dev box - rotating per start would throw away
/// yesterday's evidence to save nothing. Bounding it bounds it; 8 MiB of
/// manager log is months of normal operation.
///
/// Two files, never more. A log directory that grows without limit is the
/// thing being fixed; a numbered series just fixes it more slowly.
const ROTATE_AT: u64 = 8 * 1024 * 1024;

/// Install the process-wide subscriber.
///
/// `tee` is where the file copy goes, or `None` for terminal only. Infallible
/// by construction: logging must never be the reason a process fails to start,
/// so a log we cannot open degrades to terminal-only with a line saying so
/// rather than taking the server down with it.
pub fn init(tee: Option<&Path>) {
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;

    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| crate::DEFAULT_LOG_FILTER.into());

    let file_layer = tee.and_then(|path| {
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        rotate_if_large(path);
        match std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
        {
            Ok(f) => Some(
                tracing_subscriber::fmt::layer()
                    .with_ansi(false)
                    .with_writer(std::sync::Mutex::new(f)),
            ),
            Err(e) => {
                eprintln!(
                    "warning: cannot open {} ({e}) - logging to the terminal only",
                    path.display()
                );
                None
            }
        }
    });

    // Colour only when somebody is there to see it. A redirected stdout is a
    // FILE (the manager hands each runner one), and NO_COLOR is the standard
    // way to say "plain text" even on a real console.
    let color = std::io::stdout().is_terminal() && std::env::var_os("NO_COLOR").is_none();

    // `try_init`, not `init`: a second call is a bug in our wiring, and the
    // place it would bite hardest is a Windows service, where a panic has no
    // console to print to and the process just vanishes. Say it and carry on.
    if tracing_subscriber::registry()
        .with(filter)
        .with(tracing_subscriber::fmt::layer().with_ansi(color))
        .with(file_layer)
        .try_init()
        .is_err()
    {
        eprintln!("warning: logging was already initialised - this call did nothing");
    }
}

/// Move an oversized log aside so the new run starts against a bounded file.
///
/// Best-effort throughout. On Windows a live `tail` holds the file open and the
/// rename fails; appending past the cap is a far better outcome than refusing to
/// start, so every error here is swallowed deliberately - same call the runner's
/// per-generation rotation already makes.
fn rotate_if_large(path: &Path) {
    let Ok(meta) = std::fs::metadata(path) else {
        return; // no file yet: nothing to rotate
    };
    if meta.len() < ROTATE_AT {
        return;
    }
    let prev = path.with_extension("prev.log");
    let _ = std::fs::remove_file(&prev);
    let _ = std::fs::rename(path, &prev);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_small_log_is_left_alone() {
        let dir = std::env::temp_dir().join("pd-log-small");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("t.log");
        std::fs::write(&path, b"recent evidence").expect("write");

        rotate_if_large(&path);

        assert_eq!(
            std::fs::read(&path).expect("still there"),
            b"recent evidence",
            "rotating a small log throws away history for nothing"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_oversized_log_moves_aside_and_keeps_exactly_one_generation() {
        let dir = std::env::temp_dir().join("pd-log-big");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("t.log");
        let prev = path.with_extension("prev.log");
        std::fs::write(&prev, b"older still").expect("seed prev");
        std::fs::write(&path, vec![b'x'; (ROTATE_AT + 1) as usize]).expect("write big");

        rotate_if_large(&path);

        assert!(
            !path.exists(),
            "the oversized log should have been moved aside"
        );
        assert_eq!(
            std::fs::metadata(&prev).expect("prev exists").len(),
            ROTATE_AT + 1,
            "prev must hold the log we just rotated, not the older one it replaced"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Missing file is the first-run case and must be silent, not an error.
    #[test]
    fn rotating_a_log_that_does_not_exist_is_a_no_op() {
        let path = std::env::temp_dir().join("pd-log-absent").join("nope.log");
        rotate_if_large(&path);
        assert!(!path.exists());
    }
}
