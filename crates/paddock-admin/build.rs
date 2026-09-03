// Stamp the commit into the binary.
//
// SemVer alone cannot answer "which build is this?" during 0.x, where two
// people can hand you a `paddock 0.1.0` built a week apart. So every binary
// carries `0.1.0 (g58056f8a)` - SemVer for ordering and compatibility, the
// short SHA for exactness.
//
// Deliberately not a build counter. llama.cpp uses `git rev-list --count` and
// ships `0.0.${BUILD_NUMBER}`, which is honest for a project that promises no
// compatibility; here it would be actively wrong, because the count is not
// monotonic across branches and shifts under rebase - with more than one
// machine pushing to the repo that is a live hazard, not a theoretical one.
//
// This lives in paddock-admin because that is already the crate both binaries
// link for box-level facts (`data_root_resolved`, `DEFAULT_LOG_FILTER`), so
// the manager and the runner can never disagree about what build they are.

use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR");
    let semver = std::env::var("CARGO_PKG_VERSION").expect("cargo sets CARGO_PKG_VERSION");

    let sha = git_sha(&manifest);
    match &sha {
        Some(sha) => {
            println!("cargo:rustc-env=PADDOCK_GIT_SHA={sha}");
            println!("cargo:rustc-env=PADDOCK_VERSION_LONG={semver} (g{sha})");
        }
        // A source tarball, a vendored build, a box without git on PATH. The
        // version is still true; only the commit is unknown, and saying
        // nothing beats inventing "unknown" and printing it at every startup.
        None => {
            println!("cargo:rustc-env=PADDOCK_GIT_SHA=");
            println!("cargo:rustc-env=PADDOCK_VERSION_LONG={semver}");
        }
    }

    watch_head(&manifest);
}

fn git_sha(manifest: &str) -> Option<String> {
    let out = Command::new("git")
        .args(["-C", manifest, "rev-parse", "--short=8", "HEAD"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let sha = String::from_utf8(out.stdout).ok()?.trim().to_string();
    // A shallow or freshly-`git init`ed tree answers with nothing useful.
    if sha.is_empty() { None } else { Some(sha) }
}

/// Re-run when HEAD moves, and only then.
///
/// Without this the stamp is written once and then cached forever: cargo has
/// no reason to re-run a build script whose crate did not change, so a binary
/// rebuilt after ten commits would still name the first one. Watching HEAD and
/// the ref it points at covers commit, checkout and branch switch.
///
/// NOTE what is deliberately absent: a clean/dirty marker. Dirtiness is a
/// property of the whole worktree, and no rerun-if-changed can express "any
/// tracked file". A cached script would therefore print `clean` over a modified
/// tree - a stamp that lies in the dangerous direction, in the exact artifact
/// (a bug report) where it is trusted. The release script refuses a dirty tree
/// instead, which is the check that actually holds.
fn watch_head(manifest: &str) {
    let Some(git_dir) = git_dir(manifest) else {
        return;
    };

    let head = git_dir.join("HEAD");
    if !head.exists() {
        return;
    }
    println!("cargo:rerun-if-changed={}", head.display());

    // `ref: refs/heads/main` - watch the branch tip too, so a commit (which
    // leaves HEAD's own bytes untouched) still triggers. Detached HEAD holds
    // the SHA directly and needs nothing further.
    if let Ok(text) = std::fs::read_to_string(&head)
        && let Some(r) = text.trim().strip_prefix("ref:")
    {
        let path = git_dir.join(r.trim());
        if path.exists() {
            println!("cargo:rerun-if-changed={}", path.display());
        } else {
            // Packed refs: the loose file does not exist and the tip lives in
            // one shared file instead.
            let packed = git_dir.join("packed-refs");
            if packed.exists() {
                println!("cargo:rerun-if-changed={}", packed.display());
            }
        }
    }
}

/// Resolve the real git dir - asking git rather than assuming `../../.git`,
/// which is wrong in a worktree (`.git` is a file pointing elsewhere) and
/// wrong again if the crate ever moves.
fn git_dir(manifest: &str) -> Option<PathBuf> {
    let out = Command::new("git")
        .args(["-C", manifest, "rev-parse", "--absolute-git-dir"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let dir = String::from_utf8(out.stdout).ok()?.trim().to_string();
    let dir = Path::new(&dir);
    dir.is_dir().then(|| dir.to_path_buf())
}
