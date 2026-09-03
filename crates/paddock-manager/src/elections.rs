//! Election persistence (doc §11.2): `~/paddock/managed.toml`, the
//! manager-written desired state - which models on which ports. Deliberately
//! not named paddock.toml (that name is the operator's, settings-and-reload).
//!
//! Semantics are desired-state, compose-style: a successful spawn records an
//! election, a stop removes it, a same-port takeover replaces it. On boot the
//! manager reads the file and respawns every elected runner whose port is not
//! already serving (§11.4 flow 5 - the service posture). The file is plain
//! TOML in the user's data dir: readable, hand-editable, no lock-in.
//!
//! The runner API key is persisted as-is when the spawn carried one. This is
//! deliberate, not an oversight: the §5.1 trust model is a single-user box
//! (manager and runners share one user), and the key is already visible to
//! that user on the runner's command line - a file in the same user's profile
//! dir grants nothing new.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use serde::{Deserialize, Serialize};

/// One elected runner. Deliberately SMALL since the config-file split: the
/// endpoint's actual configuration - envelope, key,
/// GPU pin, fp8 planes, server tools, everything - lives in its own
/// `servers/<port>.toml` (the file is the truth; runnable standalone).
/// managed.toml only says which files start on boot, plus the two launch
/// facts a file can't carry (catalog identity for the editor's re-resolution,
/// runner-version pin) and policy.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Election {
    /// Catalog identity (model id + artifact) - the Start/Edit page and a
    /// takeover re-resolve weights through the registry with these.
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub artifact: Option<String>,
    pub port: u16,
    /// The configuration: this endpoint's config file. Boot respawns launch
    /// it verbatim (`paddock-runner --config <file>`), never a re-render.
    pub config: PathBuf,
    /// Pinned runner artifact version (§11.5: rollback = re-elect the
    /// previous version). None = newest installed at each respawn.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub runner_version: Option<String>,
    /// §10.1 policy pin: never auto-stopped to make room; excluded from the
    /// estimator's reclaimable VRAM. Omitted from the file when false.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub pinned: bool,
}

/// On-disk shape. `version` guards future migrations; unknown future fields
/// in old managers fail the parse loudly instead of being silently dropped
/// on the next rewrite.
#[derive(Debug, Default, Serialize, Deserialize)]
struct ManagedFile {
    version: u32,
    #[serde(default, rename = "runner")]
    runners: Vec<Election>,
}

// v2: the config-file split - serving fields moved out of the
// election into servers/<port>.toml. v3: env + gpu moved into
// the FILE too (`[env]` never shipped; gpu/fp8_native are named file fields)
// - the spawn command is purely `paddock-runner --config <file>`. Old files
// are parked as .invalid, loudly (no silent migration; a v2 env map dropped
// quietly would be a silent failure). Start the models again to re-elect.
const FILE_VERSION: u32 = 3;

/// The election set, mirrored to `managed.toml` on every mutation.
pub struct Elections {
    path: PathBuf,
    inner: Mutex<Vec<Election>>,
}

impl Elections {
    /// Load the file (missing file = empty set). A file that exists but does
    /// not parse is moved aside to `managed.toml.invalid` - preserved for the
    /// operator, never overwritten by our next write - and logged loudly.
    pub fn load(path: PathBuf) -> Self {
        let runners = match std::fs::read_to_string(&path) {
            Ok(s) => match toml::from_str::<ManagedFile>(&s) {
                Ok(f) if f.version == FILE_VERSION => f.runners,
                Ok(f) if f.version < FILE_VERSION => {
                    tracing::error!(
                        path = %path.display(),
                        version = f.version,
                        "managed.toml is an OLDER format (fields since moved into servers/<port>.toml) - parked as .invalid; start your models again to re-elect"
                    );
                    // No silent migration: an old file may carry fields (v2's
                    // env map) whose meaning moved into the config files.
                    Self::park_invalid(&path);
                    Vec::new()
                }
                Ok(f) => {
                    tracing::error!(
                        path = %path.display(),
                        version = f.version,
                        "managed.toml written by a NEWER manager - elections ignored (not rewritten); upgrade or delete the file"
                    );
                    // Refuse to mutate a newer file: park it so our writes
                    // can't destroy information we don't understand.
                    Self::park_invalid(&path);
                    Vec::new()
                }
                Err(e) => {
                    tracing::error!(
                        path = %path.display(),
                        %e,
                        "managed.toml does not parse - moved aside to .invalid, starting with no elections"
                    );
                    Self::park_invalid(&path);
                    Vec::new()
                }
            },
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Vec::new(),
            Err(e) => {
                tracing::error!(path = %path.display(), %e, "managed.toml unreadable - starting with no elections");
                Vec::new()
            }
        };
        Self {
            path,
            inner: Mutex::new(runners),
        }
    }

    fn park_invalid(path: &Path) {
        let parked = path.with_extension("toml.invalid");
        if let Err(e) = std::fs::rename(path, &parked) {
            tracing::error!(%e, "could not move the bad file aside - refusing would clobber it; elections stay read-only this session");
        }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn list(&self) -> Vec<Election> {
        self.lock().clone()
    }

    /// Poison-tolerant lock: every mutation leaves the Vec in a valid state,
    /// so a panicked writer's data is still the best desired-state we have.
    fn lock(&self) -> std::sync::MutexGuard<'_, Vec<Election>> {
        self.inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Record (or replace, keyed by port) an election and rewrite the file.
    pub fn record(&self, e: Election) {
        let mut inner = self.lock();
        inner.retain(|x| x.port != e.port);
        inner.push(e);
        inner.sort_by_key(|x| x.port);
        self.write(&inner);
    }

    /// Flip the §10.1 pin on an elected port and rewrite the file. Returns
    /// whether an election existed (a bench spawn has none - the pin then
    /// lives only in the supervisor's live record).
    pub fn set_pinned(&self, port: u16, pinned: bool) -> bool {
        let mut inner = self.lock();
        let Some(e) = inner.iter_mut().find(|e| e.port == port) else {
            return false;
        };
        if e.pinned != pinned {
            e.pinned = pinned;
            self.write(&inner);
        }
        true
    }

    /// Drop the election on `port` (a stop is a desired-state change) and
    /// rewrite the file. No-op if the port was never elected.
    pub fn remove(&self, port: u16) {
        let mut inner = self.lock();
        let before = inner.len();
        inner.retain(|x| x.port != port);
        if inner.len() != before {
            self.write(&inner);
        }
    }

    /// Atomic rewrite: temp file + rename (std's rename replaces on Windows
    /// too). A torn write must never eat the election set.
    fn write(&self, runners: &[Election]) {
        let file = ManagedFile {
            version: FILE_VERSION,
            runners: runners.to_vec(),
        };
        let body = match toml::to_string_pretty(&file) {
            Ok(b) => format!(
                "# managed.toml - written by paddock-manager (doc 11.2). Desired state:\n\
                 # each [[runner]] is respawned on manager boot if its port is not serving.\n\
                 # Hand-edits are honored on the next boot but rewritten by the next\n\
                 # spawn/stop through the manager.\n{b}"
            ),
            Err(e) => {
                tracing::error!(%e, "election set does not serialize - file left untouched");
                return;
            }
        };
        if let Some(dir) = self.path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        let tmp = self.path.with_extension("toml.tmp");
        let res = std::fs::write(&tmp, &body).and_then(|()| std::fs::rename(&tmp, &self.path));
        if let Err(e) = res {
            tracing::error!(path = %self.path.display(), %e, "failed to persist managed.toml");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmpdir() -> PathBuf {
        let d = std::env::temp_dir().join(format!(
            "paddock-elections-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn record_remove_roundtrip_through_the_file() {
        let dir = tmpdir();
        let path = dir.join("managed.toml");
        let el = Elections::load(path.clone());
        el.record(Election {
            model: "qwen3.5-9b".into(),
            artifact: Some("q8".into()),
            port: 11540,
            config: dir.join("servers/11540.toml"),
            runner_version: None,
            pinned: false,
        });
        el.record(Election {
            model: "gpt-oss-20b".into(),
            artifact: None,
            port: 11541,
            config: dir.join("servers/11541.toml"),
            runner_version: Some("1.4.0".into()),
            pinned: true,
        });
        // replace on the same port, not append
        el.record(Election {
            model: "qwen3.6-27b".into(),
            artifact: None,
            port: 11540,
            config: dir.join("servers/11540.toml"),
            runner_version: None,
            pinned: false,
        });

        let reloaded = Elections::load(path.clone());
        let list = reloaded.list();
        assert_eq!(list.len(), 2);
        assert_eq!(list[0].model, "qwen3.6-27b");
        assert_eq!(list[1].config, dir.join("servers/11541.toml"));
        assert_eq!(list[1].runner_version.as_deref(), Some("1.4.0"));
        assert!(list[1].pinned, "pin survives the file roundtrip");

        // pin toggle persists through the file; unknown port reports false
        assert!(reloaded.set_pinned(11540, true));
        assert!(!reloaded.set_pinned(65000, true));
        assert!(Elections::load(path.clone()).list()[0].pinned);

        reloaded.remove(11540);
        assert_eq!(Elections::load(path).list().len(), 1);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn older_version_is_parked_not_migrated() {
        let dir = tmpdir();
        let path = dir.join("managed.toml");
        // a v2 file may carry an env map whose meaning has moved into the
        // endpoint's config file - loading it quietly would drop that
        std::fs::write(
            &path,
            "version = 2\n\n[[runner]]\nmodel = \"m\"\nport = 11540\nconfig = \"c.toml\"\n\n[runner.env]\nPADDOCK_X = \"1\"\n",
        )
        .unwrap();
        let el = Elections::load(path.clone());
        assert!(
            el.list().is_empty(),
            "an older-format file must not load silently"
        );
        assert!(
            path.with_extension("toml.invalid").exists(),
            "parked, not deleted"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn bad_file_is_parked_not_clobbered() {
        let dir = tmpdir();
        let path = dir.join("managed.toml");
        std::fs::write(&path, "this is not toml [[[").unwrap();
        let el = Elections::load(path.clone());
        assert!(el.list().is_empty());
        // the broken content survives, parked
        let parked = std::fs::read_to_string(path.with_extension("toml.invalid")).unwrap();
        assert!(parked.contains("not toml"));
        let _ = std::fs::remove_dir_all(dir);
    }
}
