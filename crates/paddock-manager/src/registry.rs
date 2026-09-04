//! Model registry + download engine. The set of models a paddock build can pull
//! is a **compiled-in manifest** (`models.toml`, embedded via `include_str!`), so
//! a release only ever offers models it was built to load - compatibility is by
//! construction: if a model is in this release's manifest, it works with this
//! release. There is no remote catalog to fetch or keep in sync; the origin
//! (Cloudflare R2) is a dumb file host, and each manifest entry carries the file's
//! stable URL, sha256 and size.
//!
//! The puller itself is fast: parallel HTTP-Range segments, resumable (a sidecar
//! segment map), SHA-256-verified, atomic temp->rename. Origin-agnostic - any host
//! with Range support works; a non-Range origin falls back to a single stream. The
//! browser never downloads - the server pulls to disk, where models load from.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::sync::{Mutex, mpsc};

// ─── the embedded manifest (models.toml, compiled into the binary) ──────────

/// The blessed-models manifest this release ships with. Parsed once from the
/// compiled-in `models.toml`; author-controlled, so a parse failure is a build
/// bug, never a runtime condition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Catalog {
    pub schema: u32,
    /// One entry per model. TOML spells this `[[model]]`; JSON emits `models`.
    #[serde(rename(serialize = "models", deserialize = "model"), default)]
    pub models: Vec<CatalogModel>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogModel {
    pub id: String,
    pub display: String,
    /// Maker (e.g. "OpenAI", "Alibaba") - shown as its own column in the Studio.
    #[serde(default)]
    pub vendor: Option<String>,
    #[serde(default)]
    pub family: Option<String>,
    pub capability: Vec<String>,
    /// The model's speculative heads live in the weights file (qwen nextn,
    /// nemotron MTP): speculation needs no companion, and an attached
    /// drafter forms a HYBRID with them rather than replacing them. False =
    /// a drafter is the only mechanism (muse, gemma4, laguna).
    #[serde(default)]
    pub mtp_in_file: bool,
    #[serde(default)]
    pub revision: Option<String>,
    #[serde(default)]
    pub license: Option<String>,
    /// KV cache precision this family serves at when nothing overrides it:
    /// "f16" or "fp8_e4m3". Absent = f16.
    ///
    /// MIRRORS the ENGINE and must be kept in step with it - gemma4 pools its
    /// KV at fp8-e4m3 (gpu_model/gemma4/batch.rs alloc_kv), every other family
    /// is f16. It lives here so the Studio can PRESELECT the real value
    /// instead of offering a vague "auto" that hides which one you get.
    #[serde(default)]
    pub kv_default: Option<String>,
    /// Vendor-sourced spec sheet, shown when a row is expanded in the Studio.
    /// Grouped in one struct (with Default) so adding a field never breaks the
    /// construction sites; TOML spells it `[model.specs]`.
    #[serde(default)]
    pub specs: ModelSpecs,
    /// The model's PIECES (schema 3): weight alternatives + optional
    /// companions, each independently downloadable. TOML spells this
    /// `[[model.artifact]]`; JSON emits `artifacts`.
    #[serde(rename(serialize = "artifacts", deserialize = "artifact"), default)]
    pub artifacts: Vec<CatalogArtifact>,
}

/// What role a piece plays in a serving composition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ArtifactKind {
    /// Alternatives - a server picks exactly one (the quality/format choice).
    Weights,
    /// mmproj vision tower (optional companion to GGUF weights).
    Vision,
    /// mmproj AUDIO tower - the speech encoder a generative ASR model
    /// (Qwen3-ASR) needs to hear anything. Deliberately its own kind rather
    /// than a Vision reuse: the two imply different capabilities and the
    /// picker must not offer image input for a model that only takes audio.
    /// Both are charged identically, because both are resident from startup.
    Audio,
    /// MTP/speculative drafter sideload (in-file MTP exports need none).
    Drafter,
    /// Official FP8/bf16 safetensors checkpoint dir. Today a native-plane
    /// source over a GGUF base (PADDOCK_FP8_NATIVE); becomes a weights
    /// alternative when the engine serves it directly.
    Fp8Snapshot,
}

impl ArtifactKind {
    /// Does this ride the runner's `--mmproj` flag? Vision and Audio are
    /// separate KINDS because they imply different input capabilities, but
    /// they are the same KIND of FILE and the runner takes them through one
    /// flag - it reads the tower out of the GGUF and reports `vision` or
    /// `audio` accordingly. Anywhere the composition is assembled must ask
    /// this rather than name Vision, or a speech model resolves with no
    /// companion and refuses to start (Qwen3-ASR is the case that showed it).
    pub fn is_mmproj(self) -> bool {
        matches!(self, ArtifactKind::Vision | ArtifactKind::Audio)
    }
}

/// One independently downloadable piece of a model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogArtifact {
    /// Unique within the model, e.g. "q8", "q4", "vision", "fp8".
    pub id: String,
    pub kind: ArtifactKind,
    /// "gguf" | "safetensors" - what the bytes are, for honest labeling.
    pub format: String,
    /// Human label, e.g. "Full quality", "Vision (mmproj BF16)".
    pub label: String,
    /// The honest quant tag for weights ("Q8_0", "UD-Q4_K_XL", "MXFP4").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quant: Option<String>,
    /// Part of the row-level Download bundle (default weights + every
    /// default companion).
    #[serde(default)]
    pub default: bool,
    /// This companion is the model's point - a vision artifact marked
    /// required gets no on/off switch in the Studio (granite-vision without
    /// its tower is a plain text model with the purpose gone).
    #[serde(default)]
    pub required: bool,
    /// Minimum compute capability this artifact can be SERVED on, as
    /// `[major, minor]` - absent means every GPU the engine supports.
    ///
    /// Not every weight format runs everywhere. NVFP4's W4A16 consumers are
    /// compiled for sm_120a only (consumer Blackwell), and off that target the
    /// engine falls back to the base build - correct, but a fallback nobody
    /// warned about is a download that changed nothing. The Studio greys the
    /// choice out with the requirement named, instead of letting someone pull
    /// 22 GB to get the answers they already had.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_cc: Option<[u32; 2]>,
    /// Device bytes the loaded artifact holds BEYOND its file bytes - the
    /// persistent workspace the engine pins for it from load to shutdown.
    /// Most artifacts hold nothing worth naming; the two kinds that do:
    /// towers (deepseek-ocr's DeepEncoder pins ~950 MiB of encode slabs,
    /// more than its weight file) and MoE weights (gemma-4-26B-A4B's expert
    /// serving scratch self-reports 5.79 GiB at the default 32-slot width -
    /// double the estimator's whole graph margin). A fit estimate that skips
    /// either says "fits" about a start that isn't. Measured per release,
    /// recorded here as data next to the equally build-measured sha256/size,
    /// not as a per-family constant in estimate.rs (which the tower-pricing
    /// comment there rightly bans).
    #[serde(default)]
    pub workspace: Option<u64>,
    /// The artifact's SHAPE - everything will-it-fit needs that is intrinsic to
    /// the file. Published so the estimate is the same estimate
    /// before and after download.
    ///
    /// Before this, the picker had two answers: a real one for installed files
    /// (probe the header, run the estimator) and, for everything else,
    /// `file_bytes * 1.05 + tower + 1.5 GiB` - a fudge in the same family as the
    /// `total_size * 1.2 + 1 GB` guess estimate.rs records drifting 2.8x. The
    /// split was never measured-vs-predicted; both are estimates, and only one
    /// was built out of the model's real geometry.
    ///
    /// Wanted on every weights artifact - that is what "always" buys: no row can
    /// be shown to a user unpriced, and no artifact can be published without
    /// someone having established its cost. `source` says whether the numbers
    /// were probed or measured, because the GGUF probe cannot read every format
    /// we ship (nemotron's NVFP4 arm is safetensors).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shape: Option<paddock_estimator::PublishedShape>,
    /// TOML spells this `[[model.artifact.file]]`; JSON emits `files`.
    #[serde(rename(serialize = "files", deserialize = "file"), default)]
    pub files: Vec<CatalogFile>,
}

impl CatalogArtifact {
    pub fn total_size(&self) -> u64 {
        self.files.iter().map(|f| f.size).sum()
    }

    /// Whether this artifact can run on a GPU of compute-capability `cc`: no
    /// `min_cc` floor, or `cc` meets it. `cc = None` (no card / unknown) fails
    /// any floor - a Blackwell-only lane must never be handed to a card we
    /// cannot confirm is Blackwell, so the picker falls back to a floorless one.
    pub fn fits_cc(&self, cc: Option<[u32; 2]>) -> bool {
        match self.min_cc {
            None => true,
            Some(f) => cc.is_some_and(|c| c[0] > f[0] || (c[0] == f[0] && c[1] >= f[1])),
        }
    }
}

/// Does this artifact contain a file whose dest matches `name` (an already
/// lower-cased last path component)? Name, stem, or parent-DIR - see
/// `Registry::identify_weights` for why all three. Shared so the forward and
/// reverse lookups cannot drift apart on what "the same file" means.
fn artifact_holds(a: &CatalogArtifact, name: &str) -> bool {
    a.files.iter().any(|f| {
        let d = Path::new(&f.dest);
        d.file_name()
            .is_some_and(|x| x.to_string_lossy().to_lowercase() == name)
            || d.file_stem()
                .is_some_and(|x| x.to_string_lossy().to_lowercase() == name)
            || d.parent()
                .and_then(|p| p.file_name())
                .is_some_and(|x| x.to_string_lossy().to_lowercase() == name)
    })
}

impl CatalogModel {
    pub fn weights(&self) -> impl Iterator<Item = &CatalogArtifact> {
        self.artifacts
            .iter()
            .filter(|a| a.kind == ArtifactKind::Weights)
    }

    /// The NOMINAL default weights: the one marked `default`, else the first
    /// listed. Compute-capability-agnostic - for display and tests. The live
    /// serve/download paths use `default_weights_for` so a Blackwell-gated
    /// default falls back on a card that cannot run it.
    pub fn default_weights(&self) -> Option<&CatalogArtifact> {
        self.weights()
            .find(|a| a.default)
            .or_else(|| self.weights().next())
    }

    /// Hardware-aware default: the marked `default` if it runs on a GPU of
    /// compute-capability `cc`, else the first weights that does, else the
    /// marked default (readiness then refuses with a card-specific message
    /// rather than this returning None). This is what makes NVFP4 the default
    /// on Blackwell while Q8_0 stays the default on everything else, with no
    /// per-model `default` juggling.
    pub fn default_weights_for(&self, cc: Option<[u32; 2]>) -> Option<&CatalogArtifact> {
        let marked = self.default_weights();
        match marked {
            Some(m) if m.fits_cc(cc) => Some(m),
            _ => self.weights().find(|a| a.fits_cc(cc)).or(marked),
        }
    }

    pub fn artifact(&self, id: &str) -> Option<&CatalogArtifact> {
        self.artifacts.iter().find(|a| a.id == id)
    }

    /// The row-level Download bundle for a GPU of compute-capability `cc`: the
    /// hardware-aware default weights + every default companion - what "just
    /// download it" means for this model on this machine.
    pub fn default_bundle_for(&self, cc: Option<[u32; 2]>) -> Vec<&CatalogArtifact> {
        let mut out: Vec<&CatalogArtifact> = Vec::new();
        if let Some(w) = self.default_weights_for(cc) {
            out.push(w);
        }
        out.extend(
            self.artifacts
                .iter()
                .filter(|a| a.kind != ArtifactKind::Weights && a.default),
        );
        out
    }

    /// cc-agnostic bundle (display/tests) - the NOMINAL marked default weights
    /// + default companions.
    pub fn default_bundle(&self) -> Vec<&CatalogArtifact> {
        let mut out: Vec<&CatalogArtifact> = Vec::new();
        if let Some(w) = self.default_weights() {
            out.push(w);
        }
        out.extend(
            self.artifacts
                .iter()
                .filter(|a| a.kind != ArtifactKind::Weights && a.default),
        );
        out
    }
}

/// Exact specs taken from the vendor's model card. Every field is optional - the
/// Studio renders whichever are present. Add fields freely: `Default` keeps the
/// struct-literal test sites compiling.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelSpecs {
    /// Parameter count, e.g. "20.9B total · 3.6B active (MoE)".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub params: Option<String>,
    /// Native max context length, e.g. "128K" - the value shown in the column.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context: Option<String>,
    /// Extended context beyond native (e.g. via YaRN), e.g. "up to 1M with YaRN".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_max: Option<String>,
    /// Embedding dimension (embedding models), e.g. "1024 (MRL)".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dims: Option<String>,
    /// Canonical vendor model-card URL - the authoritative spec source.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub homepage: Option<String>,
    /// One-line description in the vendor's own words.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub about: Option<String>,
    /// Short "good at" bullets for the deploy comparison card. Factual and
    /// sourced (vendor card / our measurements) - honest-naming applies to
    /// prose too, so no marketing adjectives.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub strengths: Vec<String>,
    /// The other side of the same card - what picking this model costs you.
    /// A catalog that only lists strengths is an ad, not a comparison.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tradeoffs: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogFile {
    /// Stable, absolute download URL - never changes once published.
    pub url: String,
    /// Where the file lands, relative to the models dir. Same layout a manually
    /// placed model uses, so a pull de-dups against an already-present file.
    pub dest: String,
    pub sha256: String,
    pub size: u64,
}

// ─── download engine ────────────────────────────────────────────────────────

const SEGMENT: u64 = 16 * 1024 * 1024; // 16 MiB range segments
const WORKERS: usize = 8; // concurrent range connections

#[derive(Debug, thiserror::Error)]
pub enum DlError {
    #[error("http error: {0}")]
    Http(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("checksum mismatch for {name}: expected {expected}, got {got}")]
    Checksum {
        name: String,
        expected: String,
        got: String,
    },
    #[error("origin returned status {0}")]
    Status(u16),
    #[error(
        "model file is no longer available at the origin (removed, moved, or access revoked): {url}"
    )]
    NotFound { url: String },
    #[error("size mismatch: manifest says {expected}, origin served {got}")]
    Size { expected: u64, got: u64 },
    #[error("not enough disk space at {dir}: need {need} bytes, {free} free")]
    Disk { need: u64, free: u64, dir: String },
    #[error("download cancelled")]
    Cancelled,
}

/// Map a non-success HTTP status to the right error: a *definitively gone* file
/// (404 deleted / 410 gone / 403 access revoked - e.g. someone force-deleted it
/// from R2) is a distinct `NotFound` so callers can report it clearly and never
/// confuse it with a transient origin hiccup (502/503) that's worth retrying.
fn classify_status(status: reqwest::StatusCode, url: &str) -> DlError {
    use reqwest::StatusCode;
    match status {
        StatusCode::NOT_FOUND | StatusCode::GONE | StatusCode::FORBIDDEN => DlError::NotFound {
            url: url.to_owned(),
        },
        s => DlError::Status(s.as_u16()),
    }
}

/// Free (available-to-the-user) bytes on the volume holding `path`; `None` if
/// the platform query fails.
pub fn disk_free(path: &Path) -> Option<u64> {
    fs4::available_space(path).ok()
}
/// Total bytes on the volume holding `path`.
pub fn disk_total(path: &Path) -> Option<u64> {
    fs4::total_space(path).ok()
}

/// Lowercase hex of a digest. Shared because `sha2` 0.11 moved its output to
/// `hybrid_array::Array`, which - unlike the old `GenericArray` - does not
/// implement `LowerHex`, so `format!("{:x}", h.finalize())` no longer compiles.
/// Encoding the bytes ourselves is a few lines and keeps a hex crate out of the
/// graph.
pub(crate) fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

/// Cross-platform positioned write. Each worker holds its own file handle and
/// writes a disjoint region, so there's no shared-cursor race.
fn write_at(f: &std::fs::File, buf: &[u8], offset: u64) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        f.write_all_at(buf, offset)
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::FileExt;
        // seek_write is a short-write API, so we loop - shadowed here rather
        // than `mut` on the args, which would warn on the unix arm.
        let (mut buf, mut offset) = (buf, offset);
        while !buf.is_empty() {
            let n = f.seek_write(buf, offset)?;
            if n == 0 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::WriteZero,
                    "seek_write wrote 0",
                ));
            }
            buf = &buf[n..];
            offset += n as u64;
        }
        Ok(())
    }
}

fn part_path(dest: &Path) -> PathBuf {
    dest.with_extension("part")
}
fn state_path(dest: &Path) -> PathBuf {
    dest.with_extension("part.state")
}

fn seg_len(idx: usize, size: u64) -> u64 {
    let start = idx as u64 * SEGMENT;
    (start + SEGMENT).min(size) - start
}

/// Load the resume bitmap (`1` byte per completed segment). Returns all-false
/// when absent or stale (wrong length).
fn load_state(path: &Path, n_seg: usize) -> Vec<bool> {
    match std::fs::read(path) {
        Ok(bytes) if bytes.len() == n_seg => bytes.iter().map(|&b| b == 1).collect(),
        _ => vec![false; n_seg],
    }
}

async fn fetch_range(
    client: &reqwest::Client,
    url: &str,
    start: u64,
    end_inclusive: u64,
) -> Result<Vec<u8>, DlError> {
    let resp = client
        .get(url)
        .header(
            reqwest::header::RANGE,
            format!("bytes={start}-{end_inclusive}"),
        )
        .send()
        .await
        .map_err(|e| DlError::Http(e.to_string()))?;
    if !resp.status().is_success() {
        return Err(classify_status(resp.status(), url));
    }
    let bytes = resp
        .bytes()
        .await
        .map_err(|e| DlError::Http(e.to_string()))?;
    Ok(bytes.to_vec())
}

/// Probe the origin: `Ok(true)` = honours Range (206), `Ok(false)` = serves the
/// whole file (2xx, no Range). A definitively-gone file (404/410/403) errors
/// here as `NotFound` - fail fast, before any `.part` file is created on disk.
async fn supports_range(client: &reqwest::Client, url: &str) -> Result<bool, DlError> {
    let resp = client
        .get(url)
        .header(reqwest::header::RANGE, "bytes=0-0")
        .send()
        .await
        .map_err(|e| DlError::Http(e.to_string()))?;
    let status = resp.status();
    if status == reqwest::StatusCode::PARTIAL_CONTENT {
        return Ok(true);
    }
    if status.is_success() {
        return Ok(false);
    }
    Err(classify_status(status, url))
}

async fn sha256_file(path: PathBuf) -> Result<String, DlError> {
    tokio::task::spawn_blocking(move || {
        use std::io::Read;
        let mut f = std::fs::File::open(&path)?;
        let mut hasher = Sha256::new();
        let mut buf = vec![0u8; 8 * 1024 * 1024];
        loop {
            let n = f.read(&mut buf)?;
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
        }
        Ok::<String, DlError>(hex(&hasher.finalize()))
    })
    .await
    .map_err(|e| DlError::Http(format!("hash task: {e}")))?
}

/// Download `url` -> `dest`, verifying `sha256` (+ `size`). Parallel range
/// segments (single-stream fallback for non-Range origins); resumes from a
/// `<dest>.part` + `.part.state` sidecar; atomic rename on success. `downloaded`
/// is the live byte counter the caller reads for progress. `cancel` (optional)
/// is checked between segments/chunks: on cancel the partial bytes and the
/// sidecar STAY on disk so a later call resumes. **CPU/IO + network heavy** -
/// call from a task, not a latency-sensitive path.
pub async fn download_file(
    client: &reqwest::Client,
    url: &str,
    dest: &Path,
    sha256: &str,
    size: u64,
    downloaded: Arc<AtomicU64>,
    cancel: Option<Arc<std::sync::atomic::AtomicBool>>,
) -> Result<(), DlError> {
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let part = part_path(dest);
    let want_sha = sha256.to_lowercase();

    if supports_range(client, url).await? {
        download_ranged(client, url, &part, dest, size, &downloaded, cancel.as_ref()).await?;
    } else {
        download_stream(client, url, &part, &downloaded, cancel.as_ref()).await?;
    }

    // verify then publish atomically
    let got = sha256_file(part.clone()).await?;
    if got != want_sha {
        return Err(DlError::Checksum {
            name: dest
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("?")
                .to_owned(),
            expected: want_sha,
            got,
        });
    }
    std::fs::rename(&part, dest)?;
    let _ = std::fs::remove_file(state_path(dest));
    Ok(())
}

async fn download_ranged(
    client: &reqwest::Client,
    url: &str,
    part: &Path,
    dest: &Path,
    size: u64,
    downloaded: &Arc<AtomicU64>,
    cancel: Option<&Arc<std::sync::atomic::AtomicBool>>,
) -> Result<(), DlError> {
    // preallocate the part file to the final size (idempotent on resume)
    {
        let f = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(false)
            .open(part)?;
        f.set_len(size)?;
    }
    let n_seg = size.div_ceil(SEGMENT) as usize;
    let statep = state_path(dest);
    let done = load_state(&statep, n_seg);
    // already-done bytes count toward progress immediately
    for (i, &d) in done.iter().enumerate() {
        if d {
            downloaded.fetch_add(seg_len(i, size), Ordering::Relaxed);
        }
    }
    let pending: Vec<usize> = (0..n_seg).filter(|&i| !done[i]).collect();
    if pending.is_empty() {
        return Ok(());
    }
    let queue = Arc::new(Mutex::new(pending.into_iter()));

    // one persister task: single-byte positioned writes to the state sidecar as
    // segments complete, so a crash resumes without re-downloading them.
    let (state_tx, mut state_rx) = mpsc::unbounded_channel::<usize>();
    let statep2 = statep.clone();
    let n_seg_u = n_seg as u64;
    let persister = tokio::spawn(async move {
        let sf = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(false)
            .open(&statep2);
        if let Ok(sf) = sf {
            // full length up front so an out-of-order completion never leaves the
            // sidecar short of n_seg (which load_state requires to resume).
            let _ = sf.set_len(n_seg_u);
            while let Some(idx) = state_rx.recv().await {
                let _ = write_at(&sf, &[1u8], idx as u64);
            }
        }
    });

    let mut tasks = Vec::new();
    for _ in 0..WORKERS.min(n_seg) {
        let client = client.clone();
        let url = url.to_owned();
        let part = part.to_path_buf();
        let queue = queue.clone();
        let downloaded = downloaded.clone();
        let state_tx = state_tx.clone();
        let cancel = cancel.cloned();
        tasks.push(tokio::spawn(async move {
            let fh = std::fs::OpenOptions::new().write(true).open(&part)?;
            loop {
                // cooperative cancel between segments: completed segments are
                // already persisted in the sidecar, so nothing is lost
                if cancel.as_ref().is_some_and(|c| c.load(Ordering::Relaxed)) {
                    return Err(DlError::Cancelled);
                }
                let idx = {
                    let mut q = queue.lock().await;
                    q.next()
                };
                let Some(idx) = idx else { break };
                let start = idx as u64 * SEGMENT;
                let end = (start + SEGMENT).min(size) - 1;
                let bytes = fetch_range(&client, &url, start, end).await?;
                if bytes.len() as u64 != end - start + 1 {
                    return Err(DlError::Size {
                        expected: end - start + 1,
                        got: bytes.len() as u64,
                    });
                }
                write_at(&fh, &bytes, start)?;
                downloaded.fetch_add(bytes.len() as u64, Ordering::Relaxed);
                let _ = state_tx.send(idx);
            }
            Ok::<(), DlError>(())
        }));
    }
    drop(state_tx); // so the persister ends when all workers finish
    let mut outcome: Result<(), DlError> = Ok(());
    for t in tasks {
        let r = t.await.map_err(|e| DlError::Http(format!("worker: {e}")))?;
        // join every worker before reporting (a cancel hits all of them);
        // keep the first real error, with Cancelled winning only over Ok
        match (&outcome, r) {
            (Ok(()), Err(e)) => outcome = Err(e),
            (Err(DlError::Cancelled), Err(e)) if !matches!(e, DlError::Cancelled) => {
                outcome = Err(e);
            }
            _ => {}
        }
    }
    let _ = persister.await;
    outcome
}

/// Non-Range origin: one stream, written sequentially.
async fn download_stream(
    client: &reqwest::Client,
    url: &str,
    part: &Path,
    downloaded: &Arc<AtomicU64>,
    cancel: Option<&Arc<std::sync::atomic::AtomicBool>>,
) -> Result<(), DlError> {
    use futures::StreamExt;
    use tokio::io::AsyncWriteExt;
    let resp = client
        .get(url)
        .send()
        .await
        .map_err(|e| DlError::Http(e.to_string()))?;
    if !resp.status().is_success() {
        return Err(classify_status(resp.status(), url));
    }
    let mut file = tokio::fs::File::create(part).await?;
    let mut stream = resp.bytes_stream();
    while let Some(chunk) = stream.next().await {
        if cancel.is_some_and(|c| c.load(Ordering::Relaxed)) {
            return Err(DlError::Cancelled);
        }
        let chunk = chunk.map_err(|e| DlError::Http(e.to_string()))?;
        file.write_all(&chunk).await?;
        downloaded.fetch_add(chunk.len() as u64, Ordering::Relaxed);
    }
    file.flush().await?;
    Ok(())
}

// ─── pull manager (stateful, on AppState) ───────────────────────────────────

/// Status of a pull job, JSON-tagged for the Studio to poll.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "state", rename_all = "lowercase")]
pub enum PullStatus {
    Running,
    Done,
    /// User-stopped. Partial bytes + the segment sidecar stay on disk, so a
    /// resume (a fresh job over the same selection) continues where this one
    /// stopped instead of starting over.
    Cancelled,
    Error {
        message: String,
    },
}

/// One in-flight (or finished) model pull. `downloaded`/`total` drive the bar.
pub struct PullJob {
    pub id: String,
    pub model_id: String,
    /// The catalog's human name at start time - so every progress surface can
    /// say "Qwen 3.5 9B" without a catalog lookup.
    pub display: String,
    /// The artifact selection this job was started with (None = default
    /// bundle) - retained so a resume re-pulls exactly the same pieces.
    pub artifacts: Option<Vec<String>>,
    pub downloaded: Arc<AtomicU64>,
    pub total: u64,
    /// Unix millis at creation - orders the jobs list.
    pub created_ms: u64,
    pub status: std::sync::Mutex<PullStatus>,
    /// Cooperative cancel: download workers check it between range segments.
    pub cancel: Arc<std::sync::atomic::AtomicBool>,
    /// A queued follow-up ("start this spawn spec when the bytes land") and
    /// its live state. The ROUTES layer owns the meaning and the orchestration;
    /// they ride the job so one snapshot tells the whole story and a resume
    /// keeps the plan.
    pub follow: std::sync::Mutex<Option<serde_json::Value>>,
    pub follow_state: std::sync::Mutex<Option<serde_json::Value>>,
}

impl PullJob {
    pub fn snapshot(&self) -> serde_json::Value {
        let mut v = serde_json::json!({
            "id": self.id,
            "model": self.model_id,
            "display": self.display,
            "artifacts": self.artifacts,
            "downloaded": self.downloaded.load(Ordering::Relaxed),
            "total": self.total,
            "created_ms": self.created_ms,
            "status": &*self.status.lock().unwrap_or_else(std::sync::PoisonError::into_inner),
        });
        if let Some(f) = &*self
            .follow
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
        {
            v["start"] = serde_json::json!({
                "port": f.get("spec").and_then(|s| s.get("port")),
                "action": f.get("action"),
                "state": &*self.follow_state.lock().unwrap_or_else(std::sync::PoisonError::into_inner),
            });
        }
        v
    }
}

fn unix_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Model registry: the compiled-in manifest of pullable models plus the pull
/// jobs into the models dir. One per server (on `AppState`).
pub struct Registry {
    client: reqwest::Client,
    catalog: Catalog,
    models_dir: PathBuf,
    jobs: std::sync::Mutex<std::collections::HashMap<String, Arc<PullJob>>>,
    /// This box's GPU compute-capability, from `readiness::probe` (None = no
    /// card / not wired). The live serve + download paths resolve default
    /// weights against it so a Blackwell-only lane (NVFP4) is the default here
    /// and Q8_0 the default on a card that cannot run it - see
    /// `CatalogModel::default_weights_for`.
    cc: Option<[u32; 2]>,
}

/// The release's blessed-models manifest, compiled into the binary.
const MANIFEST_TOML: &str = include_str!("../models.toml");

/// A model resolved to a concrete serving composition on disk.
#[derive(Debug)]
pub struct Resolved {
    /// The weights to load (the selected weights artifact's main file).
    pub weights: PathBuf,
    /// The vision tower (mmproj), when a vision artifact is installed.
    pub mmproj: Option<PathBuf>,
    /// The drafter to wire in without being asked: installed AND marked
    /// default in the catalog - e.g. gemma-4-31b's 470M assistant.
    pub mtp: Option<PathBuf>,
    /// Any installed drafter, default or not. This is what an explicit
    /// `spec = "auto"` reaches for: opting into speculation is exactly the
    /// permission a non-default drafter was waiting for.
    pub drafter_any: Option<PathBuf>,
    /// The catalog declares a drafter for this model (installed or not). Lets
    /// a spawn that was ASKED for speculation fail with "download the drafter"
    /// instead of quietly serving without it.
    pub drafter_declared: bool,
    /// Which drafter artifact `mtp`/`drafter_any` came from - `(id, label)`,
    /// e.g. `("drafter2", "Speed drafter (DFlash2)")`. None when the model
    /// speculates from an IN-FILE MTP (no drafter artifact) or declares none.
    /// Surfaced so "Speculation: On" can say which drafter it wired: with two
    /// DFlash versions catalogued, "on" alone is no longer self-explanatory.
    pub drafter_pick: Option<(String, String)>,
    /// The catalog claims the `speculative` capability: this engine implements
    /// speculative decode for this model, either from in-file MTP (`nextn`) or
    /// a drafter artifact. A model without it must REFUSE a spec request -
    /// accepting a setting that does nothing is the silent failure the
    /// principles forbid.
    pub speculative: bool,
    /// An installed FP8/bf16 snapshot DIRECTORY (native-plane source for the
    /// engine's PADDOCK_FP8_NATIVE ingestion). Present ≠ used: the spawn only
    /// wires it when explicitly asked.
    pub fp8_snapshot: Option<PathBuf>,
}

impl Registry {
    /// Where a catalog model's pieces live - or will live once pulled
    /// (models_dir + each file's dest). None for unknown ids. The config
    /// PREVIEW uses this so a not-yet-downloaded model still shows its real
    /// future paths instead of refusing to render.
    pub fn planned_paths(
        &self,
        id: &str,
        artifact: Option<&str>,
    ) -> Option<(PathBuf, Option<PathBuf>, Option<PathBuf>)> {
        let m = self.catalog.models.iter().find(|m| m.id == id)?;
        let w = match artifact {
            Some(a) => m
                .artifacts
                .iter()
                .find(|x| x.id == a && x.kind == ArtifactKind::Weights)?,
            None => m.default_weights_for(self.cc)?,
        };
        let dest = |a: &CatalogArtifact| a.files.first().map(|f| self.models_dir.join(&f.dest));
        let weights = dest(w)?;
        // Vision OR Audio: both are mmproj companions riding the same
        // `--mmproj` flag, and a model has one or the other (a speech tower
        // and an image tower are different files for different senses). Only
        // matching Vision here is what left Qwen3-ASR unservable - the
        // manager pulled its speech encoder and then never passed it, so the
        // runner refused with "pass its audio mmproj" for a file already on
        // disk.
        let mmproj = m
            .artifacts
            .iter()
            .find(|a| a.kind.is_mmproj() && a.default)
            .and_then(dest);
        let mtp = m
            .artifacts
            .iter()
            .find(|a| a.kind == ArtifactKind::Drafter && a.default)
            .and_then(dest);
        Some((weights, mmproj, mtp))
    }

    /// The mmproj companion this model cannot be served without, if it
    /// declares one: `Some(Ok(path))` when it is on disk, `Some(Err(label))`
    /// when the catalog declares it but the bytes are missing, `None` when the
    /// model needs no companion (or is not ours).
    ///
    /// `required` is the discriminator, and it is what makes acting on this
    /// safe: a VISION tower is a default-but-optional companion an operator
    /// may deliberately drop to get its VRAM back, while a required one means
    /// "the engine refuses to serve this architecture without it" - serving
    /// without it is not a choice, it is a crash.
    pub fn required_companion(&self, model_id: &str) -> Option<Result<PathBuf, String>> {
        let m = self.catalog.models.iter().find(|m| m.id == model_id)?;
        let a = m
            .artifacts
            .iter()
            .find(|a| a.kind.is_mmproj() && a.required)?;
        let f = a.files.first()?;
        Some(if self.is_artifact_installed(a) {
            Ok(self.models_dir.join(&f.dest))
        } else {
            Err(a.label.clone())
        })
    }

    /// Parse the embedded manifest. It ships with the binary and is author-
    /// controlled, so a parse failure is a build bug, not a runtime condition.
    pub fn new(models_dir: PathBuf) -> Self {
        let catalog: Catalog =
            toml::from_str(MANIFEST_TOML).expect("embedded models.toml is malformed");
        Self::from_catalog(catalog, models_dir)
    }

    /// Build a registry over an explicit catalog - used by tests to point the
    /// puller at a local origin instead of the baked-in R2 URLs.
    pub fn from_catalog(catalog: Catalog, models_dir: PathBuf) -> Self {
        Self {
            client: reqwest::Client::new(),
            catalog,
            models_dir,
            jobs: std::sync::Mutex::new(std::collections::HashMap::new()),
            cc: None,
        }
    }

    /// Wire the local GPU's compute-capability (from `readiness::probe`) so the
    /// live default-weights resolution is hardware-aware. Builder form: the
    /// registry is built once at startup, then wrapped in an Arc.
    pub fn with_cc(mut self, cc: Option<[u32; 2]>) -> Self {
        self.cc = cc;
        self
    }

    /// The manifest is always compiled in; `false` only if it were empty.
    pub fn enabled(&self) -> bool {
        !self.catalog.models.is_empty()
    }

    pub fn models_dir(&self) -> &Path {
        &self.models_dir
    }

    pub fn catalog(&self) -> &Catalog {
        &self.catalog
    }

    /// The manifest with install state annotated per ARTIFACT (and rolled up
    /// per model), so the Studio shows piece-level state without stat'ing from
    /// the browser. `installed` on the model = at least one weights artifact
    /// present (it is servable); `total_size` = the default bundle's bytes
    /// (what the row-level Download button means).
    pub fn catalog_annotated(&self) -> serde_json::Value {
        let models: Vec<serde_json::Value> = self
            .catalog
            .models
            .iter()
            .map(|m| {
                let mut v = serde_json::to_value(m).unwrap_or_default();
                v["installed"] = serde_json::json!(self.is_installed(m));
                v["total_size"] = serde_json::json!(
                    m.default_bundle_for(self.cc)
                        .iter()
                        .map(|a| a.total_size())
                        .sum::<u64>()
                );
                if let Some(arts) = v.get_mut("artifacts").and_then(|a| a.as_array_mut()) {
                    for (av, a) in arts.iter_mut().zip(&m.artifacts) {
                        av["installed"] = serde_json::json!(self.is_artifact_installed(a));
                        av["total_size"] = serde_json::json!(a.total_size());
                    }
                }
                v
            })
            .collect();
        serde_json::json!({ "schema": self.catalog.schema, "models": models })
    }

    /// Is every file of this artifact present locally at the right size?
    pub fn is_artifact_installed(&self, a: &CatalogArtifact) -> bool {
        !a.files.is_empty()
            && a.files.iter().all(|f| {
                std::fs::metadata(self.models_dir.join(&f.dest))
                    .map(|md| md.len())
                    .ok()
                    == Some(f.size)
            })
    }

    /// Servable now: at least one weights artifact fully present.
    pub fn is_installed(&self, m: &CatalogModel) -> bool {
        m.weights().any(|a| self.is_artifact_installed(a))
    }

    /// Reverse resolution: which catalog `(model id, weights-artifact id)`
    /// does this weights file belong to? Matched by file NAME or STEM
    /// (case-insensitive) against every weights artifact's dest - the same
    /// rule the Studio's edit page uses. The stem match matters because a
    /// runner's file-derived model id drops the `.gguf` ("Qwen3.5-9B-Q8_0"),
    /// and the parent-DIR match because directory-shaped serving (safetensors
    /// checkpoints: the forced aligner, HF-dir lanes) reports the directory
    /// as its id ("Qwen3-ForcedAligner-0.6B-hf") - without it those runners
    /// showed no name and no vendor in the fleet.
    ///
    /// GUESSWORK, and known to be: it reads a NAME. A renamed file, a copy
    /// outside the models dir, or an imported-in-place GGUF returns None even
    /// though the endpoint is perfectly well configured. That is why config
    /// files carry their own `[catalog]` block now and why this is
    /// the FALLBACK inside `identity_for`, not the answer.
    pub fn identify_weights(&self, path: &Path) -> Option<(String, String)> {
        let name = path.file_name()?.to_string_lossy().to_lowercase();
        for m in &self.catalog.models {
            for a in m.weights() {
                if artifact_holds(a, &name) {
                    return Some((m.id.clone(), a.id.clone()));
                }
            }
        }
        None
    }

    /// The catalog identity of an endpoint, from what its config file DECLARES
    /// (`[catalog]`) reconciled against the weights path it actually serves.
    /// One rule, one place - the two surfaces that used to answer this
    /// separately (`/api/servers` and `heal_spec_identity`) disagreed, which is
    /// how the edit page ended up showing a model's name in the row and
    /// "select" in the dropdown beside it.
    ///
    /// `declared` is `(model id, artifact id)` straight off the file. The
    /// reconciliation, in full:
    ///
    /// - **They agree** -> the declaration names the model, the FILE names the
    ///   artifact. Point `model` at the same model's other quant and the block
    ///   does not need editing to stay honest.
    /// - **They name different models** -> the file wins, loudly. Somebody
    ///   repointed `model` and left the block behind; serving gemma while
    ///   claiming qwen is the one outcome worth a warning.
    /// - **The catalog does not recognize the file** -> the declaration stands.
    ///   This is the case `identify_weights` can never serve - a renamed file,
    ///   a copy, a path outside the models dir - and the whole reason the block
    ///   exists. An endpoint does not lose its identity because someone
    ///   reorganized their disk.
    /// - **No block** -> `identify_weights` alone, i.e. exactly the earlier
    ///   behaviour, so every config file already on disk keeps working and
    ///   nothing needs migrating.
    ///
    /// A declared id the catalog has never heard of is discarded rather than
    /// passed through: every consumer here assumes `model` is a catalog id or a
    /// path, and a third kind ("an id nobody can look up") would show the user
    /// a selection they cannot select.
    pub fn identity_for(
        &self,
        declared: Option<(&str, Option<&str>)>,
        weights: &Path,
    ) -> Option<(String, Option<String>)> {
        let by_file = self.identify_weights(weights);
        let declared = declared.filter(|(id, _)| self.catalog.models.iter().any(|m| &m.id == id));
        match (declared, by_file) {
            (Some((id, _)), Some((fid, fart))) if id == fid => Some((fid, Some(fart))),
            (Some((id, art)), Some((fid, fart))) => {
                tracing::warn!(
                    declared = %id, declared_artifact = ?art, found = %fid,
                    weights = %weights.display(),
                    "config file's [catalog] block does not match the weights it points at - trusting the file"
                );
                Some((fid, Some(fart)))
            }
            (Some((id, art)), None) => Some((id.to_string(), art.map(str::to_string))),
            (None, Some((fid, fart))) => Some((fid, Some(fart))),
            (None, None) => None,
        }
    }

    /// The catalog entry behind a runner's advertised model - matched by
    /// catalog id first, then by weights file name for path-shaped names.
    /// None when this catalog doesn't know the model (a hand-typed GGUF, a
    /// foreign runner).
    pub fn catalog_of(&self, name: &str) -> Option<&CatalogModel> {
        match self.catalog.models.iter().find(|m| m.id == name) {
            Some(m) => Some(m),
            None => {
                let (id, _) = self.identify_weights(Path::new(name))?;
                self.catalog.models.iter().find(|m| m.id == id)
            }
        }
    }

    /// Human labels for a runner's advertised model: `(display, vendor)`.
    /// None when the catalog doesn't know it: callers fall back to the raw
    /// name rather than invent a pretty one.
    pub fn display_of(&self, name: &str) -> Option<(String, Option<String>)> {
        let m = self.catalog_of(name)?;
        Some((m.display.clone(), m.vendor.clone()))
    }

    /// What the catalog says this model can do ("chat", "vision",
    /// "transcription", ...). The only capability answer available for an
    /// endpoint that is not running: a live runner advertises its own, but a
    /// stopped one has nothing to ask, and the Studio still has to know
    /// whether starting it would get you a speech model.
    pub fn capability_of(&self, name: &str) -> Option<Vec<String>> {
        Some(self.catalog_of(name)?.capability.clone())
    }

    /// Resolve a model id to a serving composition. `weights` selects the
    /// weights artifact (None = the default choice, preferring an INSTALLED
    /// one). `pull = false` - the deploy contract - never downloads: a
    /// missing selection is an honest error naming the fix; only installed
    /// companions join the composition. `pull = true` (the CLI convenience)
    /// fetches the selected weights + default companions, blocking until
    /// done. `Ok(None)` when `name` is not a manifest id, so the caller
    /// treats it as a filesystem path instead.
    /// `drafter` names a drafter artifact id when the endpoint pinned one
    /// (muse catalogues DFlash1 and DFlash2); None takes the catalog default.
    pub async fn resolve(
        &self,
        name: &str,
        weights: Option<&str>,
        pull: bool,
        drafter: Option<&str>,
    ) -> Result<Option<Resolved>, DlError> {
        let Some(model) = self.catalog.models.iter().find(|m| m.id == name).cloned() else {
            return Ok(None); // not a known id -> the caller loads it as a path
        };

        // elect the weights artifact
        let chosen = match weights {
            Some(id) => {
                let a = model
                    .artifact(id)
                    .ok_or_else(|| DlError::Http(format!("model {name} has no artifact {id:?}")))?;
                if a.kind != ArtifactKind::Weights {
                    return Err(DlError::Http(format!(
                        "artifact {id:?} of {name} is not a weights artifact ({:?})",
                        a.kind
                    )));
                }
                a.clone()
            }
            None => model
                .weights()
                .find(|a| self.is_artifact_installed(a))
                .or_else(|| model.default_weights_for(self.cc))
                .ok_or_else(|| DlError::Http(format!("model {name} has no weights artifact")))?
                .clone(),
        };

        // the pieces this composition wants: the chosen weights + the default
        // companions (pull mode fetches them; no-pull mode uses what's there)
        let mut wanted: Vec<CatalogArtifact> = vec![chosen.clone()];
        wanted.extend(
            model
                .artifacts
                .iter()
                .filter(|a| a.kind != ArtifactKind::Weights && a.default)
                .cloned(),
        );

        if pull {
            let missing: Vec<&CatalogFile> = wanted
                .iter()
                .flat_map(|a| a.files.iter())
                .filter(|f| {
                    std::fs::metadata(self.models_dir.join(&f.dest))
                        .map(|m| m.len())
                        .ok()
                        != Some(f.size)
                })
                .collect();
            // disk guard for the not-yet-present bytes (keep ~1 GiB headroom)
            let need: u64 = missing.iter().map(|f| f.size).sum();
            if let Some(free) = disk_free(&self.models_dir)
                && need > free.saturating_sub(1 << 30)
            {
                return Err(DlError::Disk {
                    need,
                    free,
                    dir: self.models_dir.display().to_string(),
                });
            }
            for f in missing {
                tracing::info!(model = %name, file = %f.dest, size = f.size, "pulling missing model file");
                download_file(
                    &self.client,
                    &f.url,
                    &self.models_dir.join(&f.dest),
                    &f.sha256,
                    f.size,
                    Arc::new(AtomicU64::new(0)),
                    None,
                )
                .await?;
            }
        } else if !self.is_artifact_installed(&chosen) {
            return Err(DlError::Http(format!(
                "model {name} ({}) is not downloaded - get it on the Models page (or `paddock pull {name}`)",
                chosen.label
            )));
        }

        // assemble the composition from what is actually on disk
        let weights_path = chosen
            .files
            .first()
            .map(|f| self.models_dir.join(&f.dest))
            .ok_or_else(|| {
                DlError::Http(format!("artifact {} of {name} has no files", chosen.id))
            })?;
        let installed_path = |kind: ArtifactKind| -> Option<PathBuf> {
            model
                .artifacts
                .iter()
                .filter(|a| a.kind == kind)
                .find(|a| self.is_artifact_installed(a))
                .and_then(|a| a.files.first())
                .map(|f| self.models_dir.join(&f.dest))
        };
        // The mmproj companion, whichever SENSE it serves - see planned_paths
        // for why this is not a Vision-only lookup.
        let installed_mmproj = || -> Option<PathBuf> {
            model
                .artifacts
                .iter()
                .filter(|a| a.kind.is_mmproj())
                .find(|a| self.is_artifact_installed(a))
                .and_then(|a| a.files.first())
                .map(|f| self.models_dir.join(&f.dest))
        };
        // Which drafter, when a model catalogues more than one (muse ships
        // DFlash1 and DFlash2). Three rungs, and every rung requires the bytes
        // to be on DISK: the id the endpoint ASKED for, else the catalog
        // default, else any installed sibling. The election used to be able to
        // return an artifact that was not installed (the pin arm skipped the
        // check), and the three consumers below each patched around that with
        // their own installed-filter and their own fallback - so in the
        // pin-and-default-both-missing corner they DISAGREED: `drafter_any`
        // wired the installed sibling while `drafter_pick` said nothing was
        // wired, silencing "which drafter did On get me" exactly where the
        // answer is least guessable.
        let ds = || {
            model
                .artifacts
                .iter()
                .filter(|a| a.kind == ArtifactKind::Drafter)
        };
        // The pin's rung is consent for that artifact and nothing else: a pin
        // whose bytes are missing must not read as consent for a non-default
        // sibling in the default lane below (spec/MTP is a user toggle, never
        // default-on - `installed` alone would silently re-enable
        // spec for anyone who once downloaded a drafter).
        let pin_hit =
            drafter.and_then(|w| ds().find(|a| a.id == w && self.is_artifact_installed(a)));
        let default_hit = ds()
            .filter(|a| a.default)
            .find(|a| self.is_artifact_installed(a));
        let picked = pin_hit
            .or(default_hit)
            .or_else(|| ds().find(|a| self.is_artifact_installed(a)));
        let picked_path = |a: &CatalogArtifact| -> Option<PathBuf> {
            a.files.first().map(|f| self.models_dir.join(&f.dest))
        };
        // the snapshot is a DIRECTORY (config.json + shards) - resolve to it
        let fp8_snapshot = installed_path(ArtifactKind::Fp8Snapshot)
            .and_then(|p| p.parent().map(Path::to_path_buf));
        Ok(Some(Resolved {
            weights: weights_path,
            mmproj: installed_mmproj(),
            // What wires without being asked: the pin (an explicit choice is
            // the same consent that marking it default expresses), else the
            // installed default - never a non-default sibling.
            mtp: pin_hit.or(default_hit).and_then(picked_path),
            // What an explicit "on" wires: the same election with the
            // installed-sibling rung included.
            drafter_any: picked.and_then(picked_path),
            // One elected artifact feeds all three fields, so what is WIRED
            // and what is NAMED cannot part ways again.
            drafter_pick: picked.map(|a| (a.id.clone(), a.label.clone())),
            drafter_declared: model
                .artifacts
                .iter()
                .any(|a| a.kind == ArtifactKind::Drafter),
            speculative: model.capability.iter().any(|c| c == "speculative"),
            fp8_snapshot,
        }))
    }

    pub fn job(&self, id: &str) -> Option<Arc<PullJob>> {
        self.jobs
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(id)
            .cloned()
    }

    /// Every pull job this manager has run since boot, oldest first - the
    /// downloads surface (the Studio's header indicator + list).
    pub fn jobs(&self) -> Vec<Arc<PullJob>> {
        let mut v: Vec<Arc<PullJob>> = self
            .jobs
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .values()
            .cloned()
            .collect();
        v.sort_by_key(|j| j.created_ms);
        v
    }

    /// Ask a running job to stop. Cooperative: workers notice between range
    /// segments and the job settles to Cancelled shortly after. False when
    /// the job is unknown or already finished.
    pub fn cancel_pull(&self, id: &str) -> bool {
        let Some(job) = self.job(id) else {
            return false;
        };
        let running = matches!(
            &*job
                .status
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
            PullStatus::Running
        );
        if running {
            job.cancel.store(true, Ordering::Relaxed);
        }
        running
    }

    /// Resume a cancelled/failed pull: a FRESH job over the same selection.
    /// Already-complete files are skipped and partial files continue from
    /// their segment sidecars, so only the missing bytes move. The old job's
    /// queued follow-up (start-after-download) carries over; the caller
    /// re-arms its watcher.
    pub fn resume_pull(&self, id: &str) -> Result<String, DlError> {
        let old = self
            .job(id)
            .ok_or_else(|| DlError::Http(format!("unknown pull job {id}")))?;
        if matches!(
            &*old
                .status
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
            PullStatus::Running | PullStatus::Done
        ) {
            return Err(DlError::Http("job is still running or already done".into()));
        }
        let new_id = self.start_pull(&old.model_id, old.artifacts.as_deref())?;
        if let Some(new) = self.job(&new_id) {
            *new.follow
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = old
                .follow
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .clone();
        }
        Ok(new_id)
    }

    /// Start pulling pieces of `model_id` into the models dir. `artifacts` =
    /// specific artifact ids, or None for the default bundle (default weights
    /// + default companions). Returns the job id; the downloads run on a
    ///   background task, tracked in `jobs` so the Studio can poll
    ///   `job(id).snapshot()`.
    pub fn start_pull(
        &self,
        model_id: &str,
        artifacts: Option<&[String]>,
    ) -> Result<String, DlError> {
        let model = self
            .catalog
            .models
            .iter()
            .find(|m| m.id == model_id)
            .ok_or_else(|| DlError::Http(format!("unknown model {model_id}")))?
            .clone();
        let selected: Vec<&CatalogArtifact> = match artifacts {
            Some(ids) => {
                let mut out = Vec::with_capacity(ids.len());
                for id in ids {
                    out.push(model.artifact(id).ok_or_else(|| {
                        DlError::Http(format!("model {model_id} has no artifact {id:?}"))
                    })?);
                }
                out
            }
            None => model.default_bundle_for(self.cc),
        };
        // de-dup by dest (defensive - pieces should not share files anymore)
        let mut seen = std::collections::HashSet::new();
        let files: Vec<CatalogFile> = selected
            .iter()
            .flat_map(|a| a.files.iter())
            .filter(|f| seen.insert(f.dest.clone()))
            .cloned()
            .collect();
        let total: u64 = files.iter().map(|f| f.size).sum();

        // disk guard: refuse the pull up front if the not-yet-present bytes
        // wouldn't fit (keep ~1 GiB headroom), so the UI can warn instead of
        // filling the drive mid-download.
        let need: u64 = files
            .iter()
            .filter(|f| {
                std::fs::metadata(self.models_dir.join(&f.dest))
                    .map(|m| m.len())
                    .ok()
                    != Some(f.size)
            })
            .map(|f| f.size)
            .sum();
        if let Some(free) = disk_free(&self.models_dir)
            && need > free.saturating_sub(1 << 30)
        {
            return Err(DlError::Disk {
                need,
                free,
                dir: self.models_dir.display().to_string(),
            });
        }

        let job = Arc::new(PullJob {
            id: uuid::Uuid::new_v4().simple().to_string(),
            model_id: model_id.to_owned(),
            display: model.display.clone(),
            artifacts: artifacts.map(<[String]>::to_vec),
            downloaded: Arc::new(AtomicU64::new(0)),
            total,
            created_ms: unix_ms(),
            status: std::sync::Mutex::new(PullStatus::Running),
            cancel: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            follow: std::sync::Mutex::new(None),
            follow_state: std::sync::Mutex::new(None),
        });
        self.jobs
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(job.id.clone(), job.clone());

        let client = self.client.clone();
        let models_dir = self.models_dir.clone();
        let job2 = job.clone();
        tokio::spawn(async move {
            let mut outcome: Result<(), DlError> = Ok(());
            for f in &files {
                if job2.cancel.load(Ordering::Relaxed) {
                    outcome = Err(DlError::Cancelled);
                    break;
                }
                let dest = models_dir.join(&f.dest);
                // already present at the right size -> skip the download (de-dup)
                if std::fs::metadata(&dest).map(|m| m.len()).ok() == Some(f.size) {
                    job2.downloaded.fetch_add(f.size, Ordering::Relaxed);
                    continue;
                }
                if let Err(e) = download_file(
                    &client,
                    &f.url,
                    &dest,
                    &f.sha256,
                    f.size,
                    job2.downloaded.clone(),
                    Some(job2.cancel.clone()),
                )
                .await
                {
                    outcome = Err(e);
                    break;
                }
            }
            *job2
                .status
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = match outcome {
                Ok(()) => PullStatus::Done,
                Err(DlError::Cancelled) => PullStatus::Cancelled,
                Err(e) => PullStatus::Error {
                    message: e.to_string(),
                },
            };
        });
        Ok(job.id.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nvfp4_is_the_default_only_where_it_runs() {
        let reg = Registry::new(std::path::PathBuf::from("./models"));
        for id in ["granite-4.2-8b", "granite-4.2-30b"] {
            let m = reg.catalog().models.iter().find(|m| m.id == id).expect(id);
            // Blackwell (sm_120): the NVFP4 lane is the default.
            assert_eq!(
                m.default_weights_for(Some([12, 0]))
                    .and_then(|a| a.quant.as_deref()),
                Some("NVFP4"),
                "{id}: NVFP4 is the default on Blackwell",
            );
            // Ampere (sm_86): NVFP4's min_cc is unmet, so the floorless Q8_0 wins.
            assert_eq!(
                m.default_weights_for(Some([8, 6]))
                    .and_then(|a| a.quant.as_deref()),
                Some("Q8_0"),
                "{id}: falls back to Q8_0 off Blackwell",
            );
            // No card / unknown cc: never hand out a gated default.
            assert_eq!(
                m.default_weights_for(None).and_then(|a| a.quant.as_deref()),
                Some("Q8_0"),
                "{id}: falls back to Q8_0 when cc is unknown",
            );
            // The nominal (cc-agnostic) default is still the marked one.
            assert_eq!(
                m.default_weights().and_then(|a| a.quant.as_deref()),
                Some("NVFP4"),
                "{id}: nominal default is the marked NVFP4",
            );
        }
    }
    use axum::extract::State;
    use axum::http::{HeaderMap, StatusCode};
    use axum::response::IntoResponse;

    /// Over the real catalog: a model that declares a default mmproj
    /// companion must get one back in its composition - whichever sense it
    /// serves. Written after Qwen3-ASR shipped unservable: the manager
    /// downloaded its speech encoder, resolved Vision only, passed no
    /// `--mmproj`, and the runner refused with "pass its audio mmproj" for a
    /// file already on disk. Asserting the CLASS rather than that one model
    /// is the point - the next sense (video, whatever) fails here first.
    #[test]
    fn every_default_mmproj_companion_reaches_the_composition() {
        let reg = Registry::new(std::path::PathBuf::from("./this-dir-does-not-exist"));
        let mut checked = 0;
        for m in &reg.catalog().models {
            let Some((_, mmproj, _)) = reg.planned_paths(&m.id, None) else {
                continue;
            };
            let declares = m.artifacts.iter().any(|a| a.kind.is_mmproj() && a.default);
            assert_eq!(
                declares,
                mmproj.is_some(),
                "{}: declares a default mmproj companion = {declares}, composition carries one = {}",
                m.id,
                mmproj.is_some()
            );
            checked += 1;
        }
        assert!(
            checked > 0,
            "the embedded catalog resolved no models at all"
        );
    }

    // A tiny origin that serves a fixed buffer and honours `Range: bytes=a-b`
    // (206 + Content-Range), so we exercise the parallel path.
    async fn serve(State(data): State<Arc<Vec<u8>>>, headers: HeaderMap) -> impl IntoResponse {
        let total = data.len() as u64;
        if let Some(r) = headers.get(axum::http::header::RANGE) {
            let spec = r.to_str().unwrap_or("").trim_start_matches("bytes=");
            let (a, b) = spec.split_once('-').unwrap_or(("0", ""));
            let start: u64 = a.parse().unwrap_or(0);
            let end: u64 = if b.is_empty() {
                total - 1
            } else {
                b.parse::<u64>().unwrap_or(total - 1).min(total - 1)
            };
            let slice = data[start as usize..=end as usize].to_vec();
            let mut h = HeaderMap::new();
            h.insert(
                axum::http::header::CONTENT_RANGE,
                format!("bytes {start}-{end}/{total}").parse().unwrap(),
            );
            h.insert(axum::http::header::ACCEPT_RANGES, "bytes".parse().unwrap());
            (StatusCode::PARTIAL_CONTENT, h, slice).into_response()
        } else {
            (StatusCode::OK, data.to_vec()).into_response()
        }
    }

    async fn spawn_origin(data: Vec<u8>) -> String {
        let app = axum::Router::new()
            .route("/f", axum::routing::get(serve))
            .with_state(Arc::new(data));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        format!("http://{addr}/f")
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn parallel_range_download_verifies_and_publishes() {
        // ~40 MiB of a deterministic pattern -> several segments across workers
        let data: Vec<u8> = (0..40 * 1024 * 1024u32)
            .map(|i| (i.wrapping_mul(2654435761) >> 13) as u8)
            .collect();
        let sha = hex(&Sha256::digest(&data));
        let size = data.len() as u64;
        let url = spawn_origin(data.clone()).await;

        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("model.gguf");
        let client = reqwest::Client::new();
        let progress = Arc::new(AtomicU64::new(0));

        download_file(&client, &url, &dest, &sha, size, progress.clone(), None)
            .await
            .expect("download");

        assert!(dest.exists(), "final file published");
        assert!(!part_path(&dest).exists(), "part file cleaned up");
        assert_eq!(
            progress.load(Ordering::Relaxed),
            size,
            "progress reached total"
        );
        assert_eq!(std::fs::read(&dest).unwrap(), data, "bytes match exactly");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn wrong_checksum_is_rejected() {
        let data: Vec<u8> = vec![7u8; 3 * 1024 * 1024];
        let size = data.len() as u64;
        let url = spawn_origin(data).await;
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("bad.gguf");
        let client = reqwest::Client::new();
        let bad_sha = "0".repeat(64);
        let err = download_file(
            &client,
            &url,
            &dest,
            &bad_sha,
            size,
            Arc::new(AtomicU64::new(0)),
            None,
        )
        .await
        .unwrap_err();
        assert!(matches!(err, DlError::Checksum { .. }), "got {err:?}");
        assert!(!dest.exists(), "a bad download is never published");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn missing_origin_file_is_a_clean_not_found() {
        // origin serves only /f; any other path 404s - as R2 would for a file
        // that was force-deleted while still listed in the manifest.
        let base = spawn_origin(vec![1u8; 4096]).await;
        let gone = base.replace("/f", "/deleted-from-r2.gguf");
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("x.gguf");
        let err = download_file(
            &reqwest::Client::new(),
            &gone,
            &dest,
            &"0".repeat(64),
            4096,
            Arc::new(AtomicU64::new(0)),
            None,
        )
        .await
        .unwrap_err();
        assert!(
            matches!(err, DlError::NotFound { .. }),
            "gone file -> NotFound, got {err:?}"
        );
        assert!(!dest.exists(), "nothing published");
        assert!(
            !part_path(&dest).exists(),
            "no .part left behind (failed fast before disk write)"
        );
    }

    /// Two models, so "the file names a different model" is expressible.
    fn two_model_registry() -> Registry {
        let art = |id: &str, kind: ArtifactKind, dest: &str| CatalogArtifact {
            id: id.into(),
            kind,
            format: "gguf".into(),
            label: "Full quality".into(),
            quant: None,
            default: id == "q8",
            required: false,
            min_cc: None,
            workspace: None,
            shape: None,
            files: vec![CatalogFile {
                url: String::new(),
                dest: dest.into(),
                sha256: String::new(),
                size: 1,
            }],
        };
        let model = |id: &str, artifacts: Vec<CatalogArtifact>| CatalogModel {
            id: id.into(),
            display: id.into(),
            vendor: None,
            family: None,
            mtp_in_file: false,
            capability: vec!["chat".into()],
            revision: None,
            license: None,
            kv_default: None,
            specs: Default::default(),
            artifacts,
        };
        let catalog = Catalog {
            schema: 3,
            models: vec![
                model(
                    "tiny",
                    vec![
                        art("q8", ArtifactKind::Weights, "tiny-GGUF/Tiny-Q8_0.gguf"),
                        art("q4", ArtifactKind::Weights, "tiny-GGUF/Tiny-Q4_K_M.gguf"),
                        // a vision companion must not identify as weights
                        art("vision", ArtifactKind::Vision, "tiny-GGUF/mmproj-BF16.gguf"),
                    ],
                ),
                model(
                    "other",
                    vec![art("q4", ArtifactKind::Weights, "other-GGUF/Other-Q4.gguf")],
                ),
            ],
        };
        Registry::from_catalog(catalog, PathBuf::from("unused"))
    }

    #[test]
    fn identify_weights_maps_a_path_back_to_catalog_identity() {
        let reg = two_model_registry();
        // case-insensitive, matched by file name regardless of the dir it
        // actually lives in (an election path uses the real install root)
        assert_eq!(
            reg.identify_weights(Path::new(r"E:\models\tiny-GGUF/tiny-q8_0.gguf")),
            Some(("tiny".into(), "q8".into()))
        );
        // a runner's file-derived id has no extension - the stem still matches
        assert_eq!(
            reg.identify_weights(Path::new("Tiny-Q8_0")),
            Some(("tiny".into(), "q8".into()))
        );
        // directory-shaped serving (a safetensors checkpoint dir like the
        // forced aligner) reports the DIRECTORY name as its id - the dest's
        // parent matches it back to the model
        assert_eq!(
            reg.identify_weights(Path::new("tiny-gguf")),
            Some(("tiny".into(), "q8".into()))
        );
        assert_eq!(
            reg.identify_weights(Path::new(r"E:\models\tiny-GGUF/mmproj-BF16.gguf")),
            None
        );
        assert_eq!(reg.identify_weights(Path::new("something-else.gguf")), None);
    }

    /// The four ways a config file's `[catalog]` block and its `model` path can
    /// stand to each other. The fourth is the one that motivates
    /// the block at all: a file the catalog cannot recognize by name.
    #[test]
    fn identity_for_reconciles_the_declaration_with_the_weights() {
        let reg = two_model_registry();
        // Mixed separators deliberately, exactly as the sibling test above does:
        // `\` is not a path separator on unix, so an all-backslash literal makes
        // `file_name()` return the whole string and `identify_weights` answer
        // None for a path that is perfectly recognizable on Windows. That is
        // not a harmless platform quirk here - with `by_file` forced to None,
        // most assertions below stop testing reconciliation at all and pass
        // through the "declaration stands" arm instead. Only the `retired`
        // case, where the declaration is discarded, ever noticed.
        let q8 = Path::new(r"E:\models\tiny-GGUF/Tiny-Q8_0.gguf");

        // agree -> the declaration names the model, the FILE names the artifact
        assert_eq!(
            reg.identity_for(Some(("tiny", Some("q8"))), q8),
            Some(("tiny".into(), Some("q8".into())))
        );
        // same model, other quant: repointing `model` does not require editing
        // the block, and the artifact follows the bytes
        assert_eq!(
            reg.identity_for(Some(("tiny", Some("q8"))), Path::new("Tiny-Q4_K_M.gguf")),
            Some(("tiny".into(), Some("q4".into())))
        );
        // the file is a different model - somebody repointed `model` and left
        // the block behind. Serving `other` while claiming `tiny` is the one
        // outcome worth overruling the declaration for.
        assert_eq!(
            reg.identity_for(Some(("tiny", Some("q8"))), Path::new("Other-Q4.gguf")),
            Some(("other".into(), Some("q4".into())))
        );
        // The CASE identify_weights can never serve: renamed, copied, imported
        // in place. The declaration stands - an endpoint does not lose its
        // identity because someone reorganized their disk.
        assert_eq!(
            reg.identity_for(
                Some(("tiny", Some("q4"))),
                Path::new(r"D:\keep\my-copy.gguf")
            ),
            Some(("tiny".into(), Some("q4".into())))
        );
        // an id the catalog has never heard of is discarded rather than passed
        // through: every consumer assumes catalog id or path, and a third kind
        // would offer the user a selection they cannot select
        assert_eq!(
            reg.identity_for(Some(("retired", None)), Path::new("my-copy.gguf")),
            None
        );
        assert_eq!(
            reg.identity_for(Some(("retired", None)), q8),
            Some(("tiny".into(), Some("q8".into())))
        );
        // no block at all = every config file written before then, and the
        // answer is exactly what those files got then
        assert_eq!(
            reg.identity_for(None, q8),
            Some(("tiny".into(), Some("q8".into())))
        );
        assert_eq!(reg.identity_for(None, Path::new("my-copy.gguf")), None);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn registry_pull_from_manifest_downloads_to_dest() {
        let data: Vec<u8> = (0..5 * 1024 * 1024u32).map(|i| (i >> 3) as u8).collect();
        let sha = hex(&Sha256::digest(&data));
        let size = data.len() as u64;
        let url = spawn_origin(data.clone()).await; // serves the bytes at /f, honours Range

        // a one-model manifest pointing at the local origin (as models.toml would
        // at real R2), with an explicit dest - no remote catalog is ever fetched.
        let catalog = Catalog {
            schema: 3,
            models: vec![CatalogModel {
                id: "tiny".into(),
                display: "Tiny".into(),
                vendor: None,
                family: None,
                mtp_in_file: false,
                capability: vec!["chat".into()],
                revision: None,
                license: Some("apache-2.0".into()),
                kv_default: None,
                specs: Default::default(),
                artifacts: vec![CatalogArtifact {
                    id: "q8".into(),
                    kind: ArtifactKind::Weights,
                    format: "gguf".into(),
                    label: "Full quality".into(),
                    quant: Some("Q8_0".into()),
                    default: true,
                    required: false,
                    min_cc: None,
                    workspace: None,
                    shape: None,
                    files: vec![CatalogFile {
                        url: url.clone(),
                        dest: "tiny-GGUF/tiny.gguf".into(),
                        sha256: sha.clone(),
                        size,
                    }],
                }],
            }],
        };
        let dir = tempfile::tempdir().unwrap();
        let reg = Registry::from_catalog(catalog, dir.path().to_path_buf());
        assert_eq!(reg.catalog().models[0].id, "tiny");
        assert!(
            !reg.is_installed(&reg.catalog().models[0]),
            "not installed before pull"
        );

        let jid = reg.start_pull("tiny", None).expect("start pull");
        // poll to completion
        let mut done = false;
        for _ in 0..300 {
            let st = reg
                .job(&jid)
                .unwrap()
                .status
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .clone();
            match st {
                PullStatus::Done => {
                    done = true;
                    break;
                }
                PullStatus::Error { message } => panic!("pull failed: {message}"),
                PullStatus::Cancelled => panic!("nothing cancelled this pull"),
                PullStatus::Running => {}
            }
            tokio::time::sleep(std::time::Duration::from_millis(15)).await;
        }
        assert!(done, "pull reached Done");
        let job = reg.job(&jid).unwrap();
        assert_eq!(
            job.downloaded.load(Ordering::Relaxed),
            size,
            "progress = total"
        );
        let landed = dir.path().join("tiny-GGUF").join("tiny.gguf");
        assert!(landed.exists(), "model file landed at its dest");
        assert_eq!(std::fs::read(&landed).unwrap(), data, "bytes verified");
        assert!(
            reg.is_installed(&reg.catalog().models[0]),
            "installed after pull"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn resolve_pulls_missing_files_and_splits_weights_from_mmproj() {
        let data: Vec<u8> = (0..2 * 1024 * 1024u32).map(|i| (i >> 2) as u8).collect();
        let sha = hex(&Sha256::digest(&data));
        let size = data.len() as u64;
        let url = spawn_origin(data.clone()).await;

        // a weights + vision model (schema 3 artifacts) - both files point at
        // the local origin.
        let catalog = Catalog {
            schema: 3,
            models: vec![CatalogModel {
                id: "vis".into(),
                display: "Vis".into(),
                vendor: None,
                family: None,
                mtp_in_file: false,
                capability: vec!["chat".into(), "vision".into()],
                revision: None,
                license: None,
                kv_default: None,
                specs: Default::default(),
                artifacts: vec![
                    CatalogArtifact {
                        id: "q8".into(),
                        kind: ArtifactKind::Weights,
                        format: "gguf".into(),
                        label: "Full quality".into(),
                        quant: Some("Q8_0".into()),
                        default: true,
                        required: false,
                        min_cc: None,
                        workspace: None,
                        shape: None,
                        files: vec![CatalogFile {
                            url: url.clone(),
                            dest: "Vis-GGUF/vis-Q8_0.gguf".into(),
                            sha256: sha.clone(),
                            size,
                        }],
                    },
                    CatalogArtifact {
                        id: "vision".into(),
                        kind: ArtifactKind::Vision,
                        format: "gguf".into(),
                        label: "Vision".into(),
                        quant: None,
                        default: true,
                        required: false,
                        min_cc: None,
                        workspace: None,
                        shape: None,
                        files: vec![CatalogFile {
                            url: url.clone(),
                            dest: "Vis-GGUF/vis-mmproj-F16.gguf".into(),
                            sha256: sha.clone(),
                            size,
                        }],
                    },
                ],
            }],
        };
        let dir = tempfile::tempdir().unwrap();
        let reg = Registry::from_catalog(catalog, dir.path().to_path_buf());

        // an unknown name resolves to None -> caller treats it as a path
        assert!(
            reg.resolve("not-a-model", None, true, None)
                .await
                .unwrap()
                .is_none()
        );

        // the deploy contract: pull=false on a not-installed model is an
        // honest error naming the fix, never a silent download
        let err = reg.resolve("vis", None, false, None).await.unwrap_err();
        assert!(
            err.to_string().contains("not downloaded"),
            "honest no-pull error: {err}"
        );

        // pull=true fetches the composition and returns split paths
        let r = reg
            .resolve("vis", None, true, None)
            .await
            .unwrap()
            .expect("known id resolves");
        assert!(r.weights.exists(), "weights pulled to disk");
        assert!(
            !r.weights.to_string_lossy().contains("mmproj"),
            "weights is the weights artifact"
        );
        let mm = r.mmproj.expect("vision companion detected");
        assert!(mm.exists(), "mmproj pulled to disk");
        assert!(
            mm.to_string_lossy().contains("mmproj"),
            "mmproj is the vision artifact"
        );

        // now installed: the no-pull path resolves the same composition
        let r2 = reg
            .resolve("vis", Some("q8"), false, None)
            .await
            .unwrap()
            .expect("resolves installed");
        assert_eq!(r2.weights, r.weights);
    }

    /// Two catalogued drafters (muse ships DFlash1 and DFlash2) must elect
    /// deliberately, not by declaration order. Plain first-match made the
    /// choice invisible and the ordering load-bearing.
    #[tokio::test]
    async fn drafter_election_prefers_the_pin_then_the_default() {
        let dir = tempfile::tempdir().unwrap();
        let drafter = |id: &str, dest: &str, default: bool| CatalogArtifact {
            id: id.into(),
            kind: ArtifactKind::Drafter,
            format: "gguf".into(),
            label: format!("Speed drafter ({id})"),
            quant: None,
            default,
            required: false,
            min_cc: None,
            workspace: None,
            shape: None,
            files: vec![CatalogFile {
                url: "http://invalid.invalid/x".into(),
                dest: dest.into(),
                sha256: "0".repeat(64),
                size: 3,
            }],
        };
        let catalog = Catalog {
            schema: 3,
            models: vec![CatalogModel {
                id: "m".into(),
                display: "M".into(),
                vendor: None,
                family: None,
                mtp_in_file: false,
                capability: vec!["chat".into(), "speculative".into()],
                revision: None,
                license: None,
                kv_default: None,
                specs: Default::default(),
                artifacts: vec![
                    CatalogArtifact {
                        id: "q8".into(),
                        kind: ArtifactKind::Weights,
                        format: "gguf".into(),
                        label: "Full quality".into(),
                        quant: Some("Q8_0".into()),
                        default: true,
                        required: false,
                        min_cc: None,
                        workspace: None,
                        shape: None,
                        files: vec![CatalogFile {
                            url: "http://invalid.invalid/w".into(),
                            dest: "M/w.gguf".into(),
                            sha256: "0".repeat(64),
                            size: 3,
                        }],
                    },
                    // v2 declared first and default; v1 second. Order must not
                    // be what decides.
                    drafter("d2", "M/d2.gguf", true),
                    drafter("d1", "M/d1.gguf", false),
                ],
            }],
        };
        let reg = Registry::from_catalog(catalog, dir.path().to_path_buf());
        let put = |rel: &str| {
            let p = dir.path().join(rel);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(&p, b"abc").unwrap();
        };
        put("M/w.gguf");

        // Only v1 on disk: the default is unavailable, so the installed one is
        // wired rather than nothing - an endpoint should not lose speculation
        // because a newer drafter exists that was never downloaded.
        put("M/d1.gguf");
        let r = reg
            .resolve("m", Some("q8"), false, None)
            .await
            .unwrap()
            .expect("resolves");
        assert_eq!(r.drafter_pick.as_ref().map(|(i, _)| i.as_str()), Some("d1"));

        // both on disk: the DEFAULT wins, not the first declared or the first
        // installed
        put("M/d2.gguf");
        let r = reg
            .resolve("m", Some("q8"), false, None)
            .await
            .unwrap()
            .expect("resolves");
        assert_eq!(r.drafter_pick.as_ref().map(|(i, _)| i.as_str()), Some("d2"));

        // an explicit pin beats the default, and is wired without being
        // default: asking for it is the same consent `default` expresses
        let r = reg
            .resolve("m", Some("q8"), false, Some("d1"))
            .await
            .unwrap()
            .expect("resolves");
        assert_eq!(r.drafter_pick.as_ref().map(|(i, _)| i.as_str()), Some("d1"));
        assert!(
            r.mtp
                .expect("pin wires without asking")
                .ends_with("d1.gguf")
        );

        // a pin naming an artifact this model does not have falls back rather
        // than serving nothing
        let r = reg
            .resolve("m", Some("q8"), false, Some("nope"))
            .await
            .unwrap()
            .expect("resolves");
        assert_eq!(r.drafter_pick.as_ref().map(|(i, _)| i.as_str()), Some("d2"));
    }

    /// The corner that used to go silent (a follow-up): a pin naming
    /// a real catalogued artifact whose bytes are not downloaded. The election
    /// skipped the installed check on the pin arm, and the three consumers each
    /// patched around it separately - so `drafter_any` wired the installed
    /// sibling while `drafter_pick` reported nothing, and the "which drafter
    /// did On get me" surface said nothing in the one case where the answer is
    /// least guessable. One election feeds all three fields now.
    #[tokio::test]
    async fn a_dead_pin_falls_back_and_the_fallback_is_named() {
        let dir = tempfile::tempdir().unwrap();
        let drafter = |id: &str, dest: &str, default: bool| CatalogArtifact {
            id: id.into(),
            kind: ArtifactKind::Drafter,
            format: "gguf".into(),
            label: format!("Speed drafter ({id})"),
            quant: None,
            default,
            required: false,
            min_cc: None,
            workspace: None,
            shape: None,
            files: vec![CatalogFile {
                url: "http://invalid.invalid/x".into(),
                dest: dest.into(),
                sha256: "0".repeat(64),
                size: 3,
            }],
        };
        let catalog = Catalog {
            schema: 3,
            models: vec![CatalogModel {
                id: "m".into(),
                display: "M".into(),
                vendor: None,
                family: None,
                mtp_in_file: false,
                capability: vec!["chat".into(), "speculative".into()],
                revision: None,
                license: None,
                kv_default: None,
                specs: Default::default(),
                artifacts: vec![
                    CatalogArtifact {
                        id: "q8".into(),
                        kind: ArtifactKind::Weights,
                        format: "gguf".into(),
                        label: "Full quality".into(),
                        quant: Some("Q8_0".into()),
                        default: true,
                        required: false,
                        min_cc: None,
                        workspace: None,
                        shape: None,
                        files: vec![CatalogFile {
                            url: "http://invalid.invalid/w".into(),
                            dest: "M/w.gguf".into(),
                            sha256: "0".repeat(64),
                            size: 3,
                        }],
                    },
                    drafter("d2", "M/d2.gguf", true),
                    drafter("d1", "M/d1.gguf", false),
                ],
            }],
        };
        let reg = Registry::from_catalog(catalog, dir.path().to_path_buf());
        let put = |rel: &str| {
            let p = dir.path().join(rel);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(&p, b"abc").unwrap();
        };
        put("M/w.gguf");
        put("M/d1.gguf"); // Only the non-default sibling is on disk

        // Pin the default (d2) while its bytes are missing: an explicit "on"
        // wires the installed sibling, and NAMES it - wired and named must be
        // the same artifact.
        let r = reg
            .resolve("m", Some("q8"), false, Some("d2"))
            .await
            .unwrap()
            .expect("resolves");
        assert_eq!(r.drafter_pick.as_ref().map(|(i, _)| i.as_str()), Some("d1"));
        assert!(
            r.drafter_any
                .as_ref()
                .expect("explicit on wires the sibling")
                .ends_with("d1.gguf"),
            "drafter_any must wire what drafter_pick names"
        );
        // ...but the DEFAULT lane stays empty: a dead pin was consent for d2,
        // not for silently enabling the non-default d1 (never default-on).
        assert!(
            r.mtp.is_none(),
            "a dead pin must not default-enable a non-default sibling"
        );
    }

    // The blessed-models manifest ships in the binary - it must always parse and
    // every entry must be complete, so a hand-edited models.toml can't ship broken.
    // /api/servers answers the capability of a STOPPED endpoint from here, and
    // the composer's mic decides from that whether starting it would give you
    // a transcriber. The lookup has to survive the shape a config
    // file actually stores: `model` is normally the resolved WEIGHTS PATH, not
    // the catalog id, so a by-id-only match would report no capability for
    // every configured endpoint and the mic would offer nothing.
    #[test]
    fn capability_resolves_by_id_and_by_weights_path() {
        let reg = Registry::new(std::path::PathBuf::from("./models"));
        let speech = reg
            .catalog()
            .models
            .iter()
            .find(|m| m.capability.iter().any(|c| c == "transcription"))
            .expect("catalog ships at least one speech model");

        let by_id = reg
            .capability_of(&speech.id)
            .expect("resolves by catalog id");
        assert!(
            by_id.iter().any(|c| c == "transcription"),
            "{}: speech by id",
            speech.id
        );

        // ...and by the weights filename, which is what servers/<port>.toml holds.
        let file = speech
            .default_weights()
            .and_then(|a| a.files.first())
            .map(|f| f.dest.clone())
            .expect("speech model has a default weights file");
        let path = format!("/some/models/dir/{file}");
        let by_path = reg
            .capability_of(&path)
            .unwrap_or_else(|| panic!("{path}: resolves by weights path"));
        assert!(
            by_path.iter().any(|c| c == "transcription"),
            "{path}: speech by path"
        );

        // A model the catalog has never heard of stays unknown rather than
        // being guessed at - the mic then leaves it out instead of offering a
        // "speech model" that would not work once started.
        assert!(reg.capability_of("/models/somebody-elses.gguf").is_none());
    }

    /// Every GGUF weights artifact publishes a shape.
    ///
    /// "Always publish the shape" is only a rule if something enforces it, and
    /// this is the half that can be enforced with no GPU and no models on disk:
    /// the block is PRESENT. `the shapes generator --check` is the other half -
    /// it re-probes installed files and catches a block that drifted from the
    /// bytes it describes.
    ///
    /// The consequence of a missing block is not cosmetic: the picker has no
    /// second path any more (`approxResident` is gone), so an unpriced artifact
    /// shows a dash where a fit verdict belongs.
    #[test]
    fn every_gguf_weights_artifact_publishes_a_shape() {
        let reg = Registry::new(std::path::PathBuf::from("./models"));
        let mut naked = Vec::new();
        for m in &reg.catalog().models {
            for a in m.weights() {
                // safetensors is exempt and NAMED, not silently skipped:
                // `probe_path` reads GGUF only, so the generator has no
                // geometry source for those and refuses to invent one. The
                // exemption disappears the day the runner can report its own
                // shape after a load.
                if a.format != "gguf" {
                    continue;
                }
                if a.shape.is_none() {
                    naked.push(format!("{}/{}", m.id, a.id));
                }
            }
        }
        assert!(
            naked.is_empty(),
            "these GGUF weights artifacts publish no shape, so will-it-fit cannot price them: \
             {naked:?}\nregenerate with the shapes generator"
        );
    }

    #[test]
    fn a_published_shape_round_trips_through_the_estimator() {
        let reg = Registry::new(std::path::PathBuf::from("./models"));
        let a = reg
            .catalog()
            .models
            .iter()
            .flat_map(|m| m.weights())
            .find(|a| a.shape.is_some())
            .expect("at least one artifact publishes a shape");
        let s = a.shape.clone().unwrap();
        let weight_bytes = s.weight_bytes;
        let kv_runs: u64 = s.kv_layers.iter().map(|r| r.count).sum();
        let shape = s.into_model_shape(1234, 5678);
        assert_eq!(
            shape.weight_bytes, weight_bytes,
            "weights survive the completion"
        );
        assert_eq!(shape.tower_bytes, 1234, "tower comes from the caller");
        assert_eq!(
            shape.workspace_bytes, 5678,
            "workspace comes from the caller"
        );
        // The published form collapses identical consecutive blocks into runs;
        // the estimator still prices block by block, so the expansion has to
        // give every one of them back.
        assert_eq!(
            shape.kv_layers.len() as u64,
            kv_runs,
            "every KV block in the runs is expanded"
        );
    }

    #[test]
    fn embedded_manifest_parses_and_is_well_formed() {
        let reg = Registry::new(std::path::PathBuf::from("./models"));
        assert!(!reg.catalog().models.is_empty(), "manifest lists models");
        for m in &reg.catalog().models {
            assert!(!m.capability.is_empty(), "{}: has a capability", m.id);
            assert!(
                m.weights().next().is_some(),
                "{}: has a weights artifact",
                m.id
            );
            assert!(
                m.default_weights().is_some(),
                "{}: has a default weights choice",
                m.id
            );
            for a in &m.artifacts {
                assert!(!a.files.is_empty(), "{}/{}: artifact has files", m.id, a.id);
                for f in &a.files {
                    assert!(f.url.starts_with("http"), "{}: absolute url", m.id);
                    assert_eq!(f.sha256.len(), 64, "{}: sha256 present", m.id);
                    assert!(f.size > 0, "{}: nonzero size", m.id);
                    assert!(
                        !f.dest.is_empty() && !f.dest.starts_with('/'),
                        "{}: relative dest",
                        m.id
                    );
                }
            }
        }
        // A weights artifact can span several files, for two different
        // reasons, and only one of them is sharding.
        //
        // The spawn path hands the runner `files.first()` and the engine takes
        // it from there - for a gguf-split family the loader walks the
        // remaining shards from that first shard's own metadata. So whatever
        // else the artifact carries, file[0] has to be the thing the engine
        // can open: shard 1 of a split, or the single .gguf otherwise.
        // Listing a middle shard first would fail at load, and it is a
        // hand-editing mistake nothing else here would catch - the sizes and
        // hashes would all be correct.
        //
        // The other reason is a NOTICE riding with the weights: Røst's licence
        // is use-restricted and its text has to be undownloadable-without-the-
        // model, so LICENSE.txt sits in the same artifact rather than in an
        // optional companion someone could decline. That is not a shard and
        // must not be read as one.
        for m in &reg.catalog().models {
            for a in m.weights().filter(|a| a.files.len() > 1) {
                let first = &a.files[0].dest;
                let sharded = a.files.iter().any(|f| f.dest.contains("-of-"));
                if sharded {
                    assert!(
                        first.contains("-00001-of-"),
                        "{}/{}: a sharded weights artifact must list shard 1 first, not {first}",
                        m.id,
                        a.id
                    );
                } else {
                    assert!(
                        first.ends_with(".gguf") || first.ends_with(".safetensors"),
                        "{}/{}: file[0] is what the runner is handed - it must be the \
                         loadable weights, not {first}",
                        m.id,
                        a.id
                    );
                }
            }
        }
        // the annotated view the Studio consumes is valid JSON with the fields it needs
        let v = reg.catalog_annotated();
        assert!(
            v["models"]
                .as_array()
                .map(|a| !a.is_empty())
                .unwrap_or(false)
        );
        assert!(v["models"][0]["installed"].is_boolean());
        assert!(v["models"][0]["total_size"].is_number());
        assert!(
            v["models"][0]["artifacts"].is_array(),
            "serialized as `artifacts`, not `artifact`"
        );
        assert!(
            v["models"][0]["artifacts"][0]["installed"].is_boolean(),
            "piece-level install state"
        );
    }
}
