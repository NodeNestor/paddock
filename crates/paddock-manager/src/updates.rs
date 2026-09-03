//! Is there a newer paddock, and can we trust the copy we fetched.
//!
//! ## Why the truespar API and not our own origin
//!
//! Models, the CUDA runtime and kernel packs come from R2 because they are big,
//! hardware-matched, and hash-pinned at build time. The APP is none of those
//! things, and the release API already carries two things a static manifest
//! cannot: RELEASE NOTES (the "what's new" the UI wants) and publish/unpublish
//! (a bad release can be pulled centrally). Its read path is public - verified
//! in truespar-core: `get_latest` and `download_latest` take no AuthPrincipal
//! while create/upload/publish do - so the account-free promise holds.
//!
//! ## The hash is optional deliberately
//!
//! `sha256` was added to the API for us (truespar-core migration 0011, computed
//! server-side while streaming the upload, so it is the hash of what the API
//! actually stored rather than what a publisher claimed). Rows uploaded before
//! that column existed return null, and will forever.
//!
//! So it is `Option<String>` and the rule is: VERIFY when PRESENT, SAY so when
//! ABSENT. Never silently trust. An update is the one download that replaces the
//! program itself, and TLS protects the transport, not a bad upload or a
//! compromised bucket - `cuda_setup` already refuses to half-install on a hash
//! mismatch and this path should not be the softer one.
//!
//! ## Checking is not installing
//!
//! This module only ANSWERS the question. It deliberately does not swap
//! anything: on Windows a running exe cannot be overwritten, and per the
//! "the maintainer starts services" rule a manager must never restart itself behind the
//! user's back. Download and apply is a separate, opt-in step.

use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, SystemTime};

use futures::StreamExt;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Where releases live. Overridable so a staging API can be pointed at without
/// a rebuild - same idiom as traverse's `TRAVERSE_API_BASE`.
const DEFAULT_API_BASE: &str = "https://api.truespar.com";

/// One app id. The manager and the runner ship in the same package, so there is
/// exactly one version stream; a second id would only earn its keep the day a
/// runner ships out of band. `app_identifier` is free text in the API with no
/// registry table, so publishing is what creates the stream - there is nothing
/// to register.
const APP_ID: &str = "paddock";

/// Don't hammer it. A release is a human-scale event; once an hour is plenty and
/// keeps a long-running manager from making a nuisance of itself.
pub const CHECK_INTERVAL: Duration = Duration::from_secs(3600);

pub fn api_base() -> String {
    std::env::var("PADDOCK_API_BASE").unwrap_or_else(|_| DEFAULT_API_BASE.to_string())
}

/// One client for the whole process - same idiom as `cloud.rs`. Building a
/// `reqwest::Client` per request throws away the connection pool, which for a
/// once-an-hour check is merely wasteful and for a 112 MB download is worse.
pub fn http() -> &'static reqwest::Client {
    static HTTP: std::sync::LazyLock<reqwest::Client> = std::sync::LazyLock::new(|| {
        reqwest::Client::builder()
            .user_agent(concat!("paddock/", env!("CARGO_PKG_VERSION")))
            .build()
            .expect("update http client")
    });
    &HTTP
}

/// The API's platform identifier - `{os}-{arch}` with `x86_64` spelled `x64`.
///
/// Not `std::env::consts::OS` on its own. traverse shipped exactly that bug and
/// had to fix it in 0.8.2, whose release notes read: "the version check
/// previously queried with the wrong platform identifier and never found new
/// releases". It fails silently - the endpoint 404s and the app concludes it is
/// up to date forever - so it is worth the two lines to get right.
pub fn release_platform() -> String {
    let os = std::env::consts::OS;
    let arch = match std::env::consts::ARCH {
        "x86_64" => "x64",
        "aarch64" => "arm64",
        other => other,
    };
    format!("{os}-{arch}")
}

/// `GET /api/versions/{app}/latest`. Field names are the API's (camelCase).
///
/// `file_size` and `sha256` are `Option` because rows published before
/// truespar-core 0011 have neither, and the API returns them as explicit nulls.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LatestVersion {
    pub version: String,
    pub release_notes: Option<String>,
    pub published_at: Option<String>,
    pub download_available: bool,
    pub file_size: Option<i64>,
    pub sha256: Option<String>,
}

/// What we can honestly say about updates right now.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "state", rename_all = "kebab-case")]
pub enum State {
    /// Running the newest published build.
    Current { version: String },
    /// A newer one exists. `verifiable` is false when the published row has no
    /// sha256 - surfaced rather than hidden, so "we could not check the bytes"
    /// is never something the user finds out about only from a log file.
    Available {
        current: String,
        latest: String,
        notes: Option<String>,
        published_at: Option<String>,
        size: Option<i64>,
        downloadable: bool,
        verifiable: bool,
    },
    /// We could not find out. Offline, DNS, a 500 - all the same to the user,
    /// and none of them are worth an alarm. `why` is for the log, not the UI.
    Unknown { current: String, why: String },
}

impl State {
    /// True only when there is something to offer. The UI shows nothing at all
    /// otherwise - an "up to date" badge is noise on a screen about models.
    pub fn is_available(&self) -> bool {
        matches!(self, State::Available { .. })
    }
}

/// Our own version, single-sourced from the workspace `version` field.
pub fn current_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Parse `MAJOR.MINOR.PATCH`, ignoring any pre-release or build metadata.
///
/// Deliberately not a semver dependency: we control the format we publish, this
/// is the whole of it, and a release check must never fail to parse its way into
/// telling somebody they are up to date when they are not.
fn triple(v: &str) -> Option<(u64, u64, u64)> {
    // `0.5.2+g58056f8a` and `0.5.2-rc.1` both reduce to their core.
    let core = v.trim().trim_start_matches('v');
    let core = core.split(['+', '-']).next()?;
    let mut parts = core.split('.');
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next().unwrap_or("0").parse().ok()?;
    let patch = parts.next().unwrap_or("0").parse().ok()?;
    Some((major, minor, patch))
}

/// Is `candidate` newer than `running`?
///
/// Unparseable on either side means "no". A malformed version is not a reason to
/// nag somebody into a download; it is a reason for us to notice a bug.
pub fn is_newer(candidate: &str, running: &str) -> bool {
    match (triple(candidate), triple(running)) {
        (Some(a), Some(b)) => a > b,
        _ => false,
    }
}

/// Ask the API what the newest published build is.
///
/// Never returns `Err`: from the user's seat "we could not reach the release
/// server" is information, not a failure, and a manager whose UI breaks because
/// a laptop is on a train is worse than one that quietly says it does not know.
/// The newest published release for this platform.
///
/// `Ok(None)` is a 404, which is the normal answer before anything has been
/// published for a platform - deliberately not an error, because treating
/// "nothing released yet" as a failure would put a scary state in the UI on
/// day one. `Err` means we genuinely could not find out.
pub async fn latest(client: &reqwest::Client) -> Result<Option<LatestVersion>, String> {
    let url = format!(
        "{}/api/versions/{APP_ID}/latest?platform={}",
        api_base().trim_end_matches('/'),
        release_platform(),
    );
    let resp = client
        .get(&url)
        .timeout(Duration::from_secs(15))
        .send()
        .await
        .map_err(|e| format!("request failed: {e}"))?;

    if resp.status() == reqwest::StatusCode::NOT_FOUND {
        return Ok(None);
    }
    if !resp.status().is_success() {
        return Err(format!("HTTP {}", resp.status()));
    }
    resp.json::<LatestVersion>()
        .await
        .map(Some)
        .map_err(|e| format!("bad response: {e}"))
}

pub async fn check(client: &reqwest::Client) -> State {
    let current = current_version().to_string();

    let latest = match latest(client).await {
        Ok(Some(v)) => v,
        Ok(None) => return State::Current { version: current },
        Err(why) => return State::Unknown { current, why },
    };

    if !is_newer(&latest.version, &current) {
        return State::Current { version: current };
    }

    if latest.sha256.is_none() {
        // Loud in the log, and surfaced to the UI as `verifiable: false`. This
        // is the pre-0011 case and it should get rarer, not become normal.
        tracing::warn!(
            version = %latest.version,
            "release has no sha256 - the download cannot be verified beyond TLS"
        );
    }

    State::Available {
        current,
        latest: latest.version,
        notes: latest.release_notes,
        published_at: latest.published_at,
        size: latest.file_size,
        downloadable: latest.download_available,
        verifiable: latest.sha256.is_some(),
    }
}

/// The last answer, so every UI poll does not become an outbound request.
#[derive(Debug)]
pub struct Cache {
    inner: std::sync::Mutex<Option<(SystemTime, State)>>,
}

impl Default for Cache {
    fn default() -> Self {
        Self {
            inner: std::sync::Mutex::new(None),
        }
    }
}

impl Cache {
    /// Cached answer if it is fresher than [`CHECK_INTERVAL`], else `None`.
    pub fn fresh(&self) -> Option<State> {
        let guard = self.inner.lock().expect("update cache mutex");
        let (at, state) = guard.as_ref()?;
        (at.elapsed().ok()? < CHECK_INTERVAL).then(|| state.clone())
    }

    pub fn put(&self, state: State) {
        *self.inner.lock().expect("update cache mutex") = Some((SystemTime::now(), state));
    }

    /// Cached if fresh, otherwise ask and remember.
    pub async fn get_or_check(&self, client: &reqwest::Client) -> State {
        if let Some(hit) = self.fresh() {
            return hit;
        }
        let state = check(client).await;
        self.put(state.clone());
        state
    }
}

// ---------------------------------------------------------------- download

/// Where a fetched update lands. Beside the data root, never over the running
/// install: this step FETCHES and VERIFIES, it does not apply.
pub fn download_dir() -> PathBuf {
    paddock_admin::data_root().join("updates")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Phase {
    Idle,
    Running,
    /// Downloaded, hash checked (or honestly recorded as uncheckable), and
    /// sitting on disk. `Download::path` says where.
    Ready,
    /// Stopped without finishing; `error` says why. A retry starts clean.
    Failed,
}

/// A download in flight. One per process - a second request joins the running
/// one rather than racing it onto the same file, same rule as `cuda_setup`.
#[derive(Debug)]
pub struct Download {
    pub version: String,
    pub phase: std::sync::Mutex<Phase>,
    pub received: Arc<AtomicU64>,
    /// From the API's `fileSize`; 0 when the row predates that column, in which
    /// case the UI shows bytes rather than a percentage instead of inventing one.
    pub total: u64,
    pub path: std::sync::Mutex<Option<PathBuf>>,
    pub error: std::sync::Mutex<Option<String>>,
    pub cancel: Arc<AtomicBool>,
}

impl Download {
    fn fail(&self, why: impl std::fmt::Display) {
        *self.error.lock().expect("dl error") = Some(why.to_string());
        *self.phase.lock().expect("dl phase") = Phase::Failed;
    }

    /// Snapshot for the API. Cheap enough to poll.
    pub fn status(&self) -> serde_json::Value {
        serde_json::json!({
            "version": self.version,
            "phase": *self.phase.lock().expect("dl phase"),
            "received": self.received.load(Ordering::Relaxed),
            "total": self.total,
            "path": self.path.lock().expect("dl path").as_ref().map(|p| p.display().to_string()),
            "error": *self.error.lock().expect("dl error"),
        })
    }
}

/// Fetch the newest published package for this platform.
///
/// Streams to a `.part` file and hashes as it GOES, then renames on success.
/// Three reasons it is done that way rather than the obvious `read_to_vec`:
///
/// 1. The package is ~112 MB. Holding it resident to hash it afterwards is a
///    pointless spike on a box whose whole job is fitting models in memory.
/// 2. A `.part` that is never renamed cannot be mistaken for a usable download
///    by anything that later goes looking - an interrupted transfer leaves
///    obvious debris rather than a plausible-looking truncated zip.
/// 3. The hash is over exactly the bytes that landed, computed once.
///
/// A mismatch DELETES the file. A download we cannot vouch for is not something
/// to leave lying around for someone to run.
pub async fn download(latest: &LatestVersion, dl: Arc<Download>) {
    *dl.phase.lock().expect("dl phase") = Phase::Running;

    let dir = download_dir();
    if let Err(e) = std::fs::create_dir_all(&dir) {
        return dl.fail(format!("cannot create {}: {e}", dir.display()));
    }

    // Name it after what it IS. The Content-Disposition filename is the
    // publisher's, and two releases could share it; version + platform cannot
    // collide with another release of ours.
    let name = format!("paddock-{}-{}.zip", latest.version, release_platform());
    let final_path = dir.join(&name);
    let part_path = dir.join(format!("{name}.part"));

    let url = format!(
        "{}/api/versions/{APP_ID}/latest/download?platform={}",
        api_base().trim_end_matches('/'),
        release_platform(),
    );

    let resp = match http().get(&url).send().await {
        Ok(r) if r.status().is_success() => r,
        Ok(r) => return dl.fail(format!("HTTP {}", r.status())),
        Err(e) => return dl.fail(format!("request failed: {e}")),
    };

    let mut file = match std::fs::File::create(&part_path) {
        Ok(f) => f,
        Err(e) => return dl.fail(format!("cannot write {}: {e}", part_path.display())),
    };

    let mut hasher = Sha256::new();
    let mut stream = resp.bytes_stream();
    while let Some(chunk) = stream.next().await {
        if dl.cancel.load(Ordering::Relaxed) {
            drop(file);
            let _ = std::fs::remove_file(&part_path);
            return dl.fail("cancelled");
        }
        let chunk = match chunk {
            Ok(c) => c,
            Err(e) => {
                drop(file);
                let _ = std::fs::remove_file(&part_path);
                return dl.fail(format!("transfer failed: {e}"));
            }
        };
        hasher.update(&chunk);
        if let Err(e) = file.write_all(&chunk) {
            drop(file);
            let _ = std::fs::remove_file(&part_path);
            return dl.fail(format!("write failed: {e}"));
        }
        dl.received.fetch_add(chunk.len() as u64, Ordering::Relaxed);
    }
    if let Err(e) = file.flush() {
        return dl.fail(format!("flush failed: {e}"));
    }
    drop(file);

    let got = crate::registry::hex(&hasher.finalize());
    match latest.sha256.as_deref() {
        Some(want) if !want.eq_ignore_ascii_case(&got) => {
            // Do not keep it. An unverifiable binary left on disk is a trap.
            let _ = std::fs::remove_file(&part_path);
            tracing::error!(expected = %want, actual = %got, "update hash mismatch - deleted");
            return dl.fail(format!("hash mismatch: expected {want}, got {got}"));
        }
        Some(_) => tracing::info!(version = %latest.version, sha256 = %got, "update verified"),
        None => tracing::warn!(
            version = %latest.version, sha256 = %got,
            "release published without a sha256 - downloaded but NOT verified"
        ),
    }

    if let Err(e) = std::fs::rename(&part_path, &final_path) {
        return dl.fail(format!("cannot finalise {}: {e}", final_path.display()));
    }
    *dl.path.lock().expect("dl path") = Some(final_path);
    *dl.phase.lock().expect("dl phase") = Phase::Ready;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn newer_is_ordered_by_component_not_lexically() {
        assert!(
            is_newer("0.10.0", "0.9.0"),
            "0.10 > 0.9 numerically; lexically it is not"
        );
        assert!(is_newer("1.0.0", "0.99.99"));
        assert!(is_newer("0.5.3", "0.5.2"));
        assert!(!is_newer("0.5.2", "0.5.2"));
        assert!(!is_newer("0.5.1", "0.5.2"));
    }

    #[test]
    fn build_metadata_and_prerelease_do_not_affect_the_comparison() {
        // VERSION.txt carries `0.5.2 (g58056f8a)` and a tag may read `v0.5.2`;
        // neither should read as a different release from `0.5.2`.
        assert!(!is_newer("0.5.2+g58056f8a", "0.5.2"));
        assert!(!is_newer("v0.5.2", "0.5.2"));
        assert!(is_newer("0.6.0-rc.1", "0.5.2"));
    }

    #[test]
    fn a_version_we_cannot_parse_never_prompts_an_update() {
        // A malformed version is our bug to find, not a reason to nag a user
        // toward a download.
        assert!(!is_newer("garbage", "0.5.2"));
        assert!(!is_newer("0.5.3", "garbage"));
        assert!(!is_newer("", "0.5.2"));
    }

    #[test]
    fn short_versions_fill_missing_components_with_zero() {
        assert!(is_newer("1", "0.9.9"));
        assert!(!is_newer("1", "1.0.0"));
        assert!(is_newer("1.1", "1.0.9"));
    }

    #[test]
    fn platform_matches_what_the_api_keys_on() {
        // The exact string traverse had to fix in 0.8.2 - a wrong one 404s and
        // the app then believes it is up to date forever.
        let p = release_platform();
        assert!(!p.contains("x86_64"), "arch must be spelled x64, got {p}");
        assert!(p.contains('-'), "expected {{os}}-{{arch}}, got {p}");
        #[cfg(all(target_os = "windows", target_arch = "x86_64"))]
        assert_eq!(p, "windows-x64");
        #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
        assert_eq!(p, "linux-x64");
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        assert_eq!(p, "macos-arm64");
    }

    #[test]
    fn a_pre_0011_release_row_parses_with_both_new_fields_null() {
        // Exactly what the live API returns for traverse today, so the shape is
        // not hypothetical: explicit nulls, not absent keys.
        // r###"..."### and not r#"..."#: real release notes open with a markdown
        // heading, so the JSON contains the literal `"## ` - which is the
        // terminator for r#"..."# AND for r##"..."##. Needs one more hash than the
        // longest `"#...` run inside. The live traverse notes really do start
        // `## 0.8.2`, so an honest fixture forces this.
        let body = r###"{
            "version": "0.8.2",
            "releaseNotes": "## 0.8.2",
            "publishedAt": "2026-07-10T11:30:57.238621Z",
            "downloadAvailable": true,
            "fileSize": 38548954,
            "sha256": null
        }"###;
        let v: LatestVersion = serde_json::from_str(body).expect("parses");
        assert_eq!(v.file_size, Some(38_548_954));
        assert!(
            v.sha256.is_none(),
            "null sha256 must parse as None, not fail"
        );
        assert!(v.download_available);
    }

    #[test]
    fn a_row_predating_both_columns_still_parses() {
        // Absent keys, not just null ones - an older API build would omit them.
        let body = r#"{"version":"0.1.0","releaseNotes":null,
                       "publishedAt":null,"downloadAvailable":false}"#;
        let v: LatestVersion = serde_json::from_str(body).expect("parses without the new fields");
        assert!(v.file_size.is_none() && v.sha256.is_none());
    }

    #[test]
    fn a_stale_cache_entry_is_not_served() {
        let c = Cache::default();
        assert!(c.fresh().is_none(), "empty cache has nothing fresh");
        c.put(State::Current {
            version: "0.1.0".into(),
        });
        assert!(c.fresh().is_some(), "just-written entry is fresh");
        // Backdate past the interval and it must stop being served.
        *c.inner.lock().unwrap() = Some((
            SystemTime::now() - CHECK_INTERVAL - Duration::from_secs(1),
            State::Current {
                version: "0.1.0".into(),
            },
        ));
        assert!(
            c.fresh().is_none(),
            "an entry older than the interval is not fresh"
        );
    }

    /// Hits the real API. Ignored so CI and offline boxes stay green:
    ///
    ///   cargo test -p paddock-manager --lib updates -- --ignored --nocapture
    ///
    /// Worth having despite the network, because it is the only check that the
    /// URL and platform string are right end to end - and getting those wrong
    /// fails SILENTLY. traverse shipped exactly that: its 0.8.2 notes read "the
    /// version check previously queried with the wrong platform identifier and
    /// never found new releases", i.e. every user was told they were current,
    /// forever, and no error was ever raised. A fixture cannot catch it.
    #[tokio::test]
    #[ignore = "network"]
    async fn the_live_api_answers_the_url_we_build() {
        let client = reqwest::Client::new();
        let state = check(&client).await;
        println!("live check -> {state:?}");
        // Before the first `release.py --app paddock` publish the endpoint 404s,
        // which we read as "nothing newer" rather than an error. After it, this
        // becomes Current or Available. Unknown means we could not talk to it at
        // all - a wrong URL, a wrong platform, or a broken deploy.
        assert!(
            !matches!(state, State::Unknown { .. }),
            "could not reach or parse the release API: {state:?}"
        );
    }

    #[test]
    fn only_available_counts_as_available() {
        assert!(
            !State::Current {
                version: "0.1.0".into()
            }
            .is_available()
        );
        assert!(
            !State::Unknown {
                current: "0.1.0".into(),
                why: "offline".into()
            }
            .is_available()
        );
        assert!(
            State::Available {
                current: "0.1.0".into(),
                latest: "0.2.0".into(),
                notes: None,
                published_at: None,
                size: None,
                downloadable: true,
                verifiable: false,
            }
            .is_available()
        );
    }
}
