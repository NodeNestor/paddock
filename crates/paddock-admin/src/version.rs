//! What this build *is* - the one place either binary answers that.
//!
//! Three strings, and the difference between them matters:
//!
//! - [`SEMVER`] is the CONTRACT. Update checks, the runner-version comparison
//!   and anything that decides "is this newer" use it and nothing else,
//!   because a build stamp would make every dev build look like a stranger.
//! - [`GIT_SHA`] is the EVIDENCE - which commit produced these bytes.
//! - [`LONG`] is what a HUMAN reads: `--version`, the startup banner, a bug
//!   report. Never parse it.
//!
//! See `build.rs` next door for how the stamp is taken, and why there is no
//! dirty marker.

/// The product version. Every crate in the workspace shares it, so this is
/// paddock's version, not this crate's.
pub const SEMVER: &str = env!("CARGO_PKG_VERSION");

/// Short commit hash, or `None` when built outside a git checkout (a source
/// tarball, a vendored tree, a box with no git).
pub const GIT_SHA: Option<&str> = {
    let s = env!("PADDOCK_GIT_SHA");
    if s.is_empty() { None } else { Some(s) }
};

/// `0.1.0 (g58056f8a)`, or bare `0.1.0` with no commit to name. For display.
pub const LONG: &str = env!("PADDOCK_VERSION_LONG");

#[cfg(test)]
mod tests {
    use super::*;

    /// The two must not drift apart: LONG is built from SEMVER in build.rs, and
    /// a refactor that stamped, say, the admin crate's own version instead
    /// would be invisible until someone read a release note.
    #[test]
    fn long_starts_with_the_semver() {
        assert!(
            LONG.starts_with(SEMVER),
            "LONG {LONG:?} does not lead with SEMVER {SEMVER:?}"
        );
    }

    /// Whichever branch build.rs took, the two outputs have to agree - a sha
    /// with no parenthetical, or a parenthetical with no sha, means the stamp
    /// was written by two code paths that disagree.
    #[test]
    fn the_sha_and_the_long_form_agree() {
        match GIT_SHA {
            Some(sha) => {
                assert_eq!(LONG, format!("{SEMVER} (g{sha})"));
                assert_eq!(sha.len(), 8, "short sha is --short=8: {sha:?}");
                assert!(
                    sha.chars().all(|c| c.is_ascii_hexdigit()),
                    "not a hash: {sha:?}"
                );
            }
            None => assert_eq!(LONG, SEMVER, "no sha, so LONG must be the bare version"),
        }
    }

    /// This test suite runs inside the checkout, so the stamp must have been
    /// taken. If it were None here, build.rs failed silently and releases
    /// would ship unstamped without anyone noticing.
    #[test]
    fn a_build_from_the_repo_names_its_commit() {
        assert!(
            GIT_SHA.is_some(),
            "built inside a git checkout but no commit was stamped"
        );
    }
}
