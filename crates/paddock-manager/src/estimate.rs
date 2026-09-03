//! `/api/models/estimate` - honest will-it-fit for the whole catalog at once.
//!
//! The models page needs a VRAM figure per row, and the only figure worth
//! showing is one derived from the actual file. So this endpoint answers for
//! every catalog model in one request, at a caller-supplied (or
//! server-default) concurrency, and says plainly which rows it could not
//! measure rather than inventing a number for them.
//!
//! Context is an OUTPUT, not a parameter: each model's trained window and the
//! cache the card can afford decide it together, so the response carries both
//! the value and a whole `curve` of it against concurrency.
//!
//! Geometry comes from `paddock_models::probe` (bounded header read, never the
//! weights) and the arithmetic from `paddock_estimator`. Nothing here does math
//! of its own - a second copy of the KV formula is exactly how the old
//! `total_size * 1.2 + 1 GB` guess drifted 2.8× away from reality.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use axum::Json;
use axum::extract::{Query, State};
use axum::response::{IntoResponse, Response};
use paddock_estimator::{Device, Envelope, KvDtype, ModelKind, ModelShape};
use paddock_models::probe::{ModelReport, probe_path};
use serde::Deserialize;

/// Probing reads up to a 256 MB header prefix per file, so a naive per-request
/// sweep of a 14-model catalog would be gigabytes of I/O every time the user
/// nudges the context slider. Keyed by path + mtime + len so a re-pulled or
/// swapped file re-probes instead of serving a stale shape.
#[derive(Default)]
pub struct ProbeCache(Mutex<HashMap<(PathBuf, u64, u64), Arc<ModelReport>>>);

impl ProbeCache {
    pub fn get(&self, path: &Path) -> Option<Arc<ModelReport>> {
        let md = std::fs::metadata(path).ok()?;
        let stamp = md
            .modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map_or(0, |d| d.as_secs());
        let key = (path.to_path_buf(), stamp, md.len());
        if let Some(hit) = self.0.lock().ok()?.get(&key) {
            return Some(hit.clone());
        }
        let report = Arc::new(probe_path(path).ok()?);
        self.0.lock().ok()?.insert(key, report.clone());
        Some(report)
    }
}

#[derive(Debug, Deserialize)]
pub struct EstimateQuery {
    /// Concurrent sequences to price. Defaults to the configured `max_batch`.
    /// There is deliberately no `ctx` parameter - context is derived from what
    /// the card can back at this concurrency, capped by the model's own
    /// trained window. Asking callers to pick it from a list was both
    /// arbitrary and wrong at the edges.
    batch: Option<u64>,
    /// `f16` (default, and exact) or `fp8_e4m3`.
    kv: Option<String>,
    /// Price speculative decode: the drafter's resident bytes plus the wider
    /// verify logits plane. Off by default. Per-MODEL, since the drafter (and
    /// whether there is a separate one at all) differs per row.
    spec: Option<bool>,
    /// Which GPU to price against (NVML index, default 0). Reclaimable VRAM
    /// counts only the runners attributed to this device.
    gpu: Option<u32>,
    /// Price the vision/audio tower. Defaults to true, which is both the old
    /// behaviour and the safe direction: a caller that says nothing gets the
    /// heavier answer. `false` mirrors the start form's vision switch, which
    /// the supervisor really honours (`supervisor.rs`: `spec.vision ==
    /// Some(false)` drops the mmproj), so leaving it out of the estimate
    /// over-charged every vision model by its whole tower.
    vision: Option<bool>,
    /// This device's compute capability as `"major.minor"` (e.g. `"8.6"`), so
    /// the estimate can price the KV width the RUNNER will actually serve
    /// rather than the one that was asked for. Optional: an older Studio
    /// sends nothing and gets the request honoured verbatim, which is the
    /// earlier behaviour rather than a new guess.
    cc: Option<String>,
    /// Prefix-cache offload budget in GiB - the form's "In memory" field,
    /// which becomes `[kv_offload] ram_gb`. Absent or 0 = no tier.
    ///
    /// It has to reach the estimate for the same reason `spec` does: arming
    /// the tier reserves device staging out of the VRAM the pool is sized
    /// from, so an estimate that ignored it would draw a context the runner
    /// then seats smaller. The GiB figure itself is not VRAM and never enters
    /// a device total - it rides through to `host_ram` so the form can show
    /// what the feature actually costs the machine.
    offload_ram_gb: Option<f64>,
    /// Ceiling on what this endpoint may hold, in MiB - the form's "how much
    /// of the card" choice, which becomes the config file's `vram_budget`.
    ///
    /// Without it the estimate priced against all free VRAM while the spawn
    /// obeyed the ceiling, so a 20 GB limit drew a 37 GB endpoint. The budget
    /// is exactly "act as if the card had only this much free", so that is
    /// what it does here.
    budget: Option<u64>,
}

/// `"8.6"` -> `(8, 6)`. Anything unparseable is None, and None never gates -
/// same fail-open stance as the runner's own device singleton: refusing to
/// price because a string was malformed helps nobody.
fn parse_cc(s: &str) -> Option<(u32, u32)> {
    let (a, b) = s.split_once('.')?;
    Some((a.trim().parse().ok()?, b.trim().parse().ok()?))
}

/// Concurrency steps the curve is sampled at, so the UI can show the
/// context/concurrency trade-off without a request per point.
const CURVE: [u64; 6] = [1, 2, 4, 8, 16, 32];

/// Bytes of weights a runner's model would need reloaded after eviction -
/// the computed restore cost. Resolution mirrors spawn's: catalog id (the
/// installed - else default - weights artifact) -> installed model name ->
/// filesystem path. None when the model string can't be resolved (adopted
/// runner with a foreign path, deleted file).
fn resolve_weight_bytes(state: &crate::routes::AppState, model: &str) -> Option<u64> {
    resolve_weight_bytes_for(state, model, None)
}

/// Like `resolve_weight_bytes`, honoring an explicit weights-artifact choice
/// (the admission guard prices the artifact a spawn actually selects).
pub(crate) fn resolve_weight_bytes_for(
    state: &crate::routes::AppState,
    model: &str,
    artifact: Option<&str>,
) -> Option<u64> {
    resolve_weights_for(state, model, artifact).map(|(_, bytes, _, _)| bytes)
}

/// Full weights resolution for admission: the on-disk GGUF path (probeable),
/// the artifact's byte size, how paddock would serve it (encoders hold no
/// decode cache - pricing them with one is the 124-GB-embedding-model bug),
/// and the artifact's declared serving workspace (0 when it declares none, or
/// when the model isn't a catalog row at all - a raw path carries no such
/// manifest data, and under-charging there is the pre-existing behaviour).
/// Same resolution order as spawn: catalog id -> installed model name -> raw
/// path. The path may not exist yet (pre-download admission) - callers probe
/// and fall back to weights-only arithmetic when it doesn't.
pub(crate) fn resolve_weights_for(
    state: &crate::routes::AppState,
    model: &str,
    artifact: Option<&str>,
) -> Option<(PathBuf, u64, ModelKind, u64)> {
    if let Some(m) = state
        .registry
        .catalog()
        .models
        .iter()
        .find(|m| m.id == model)
    {
        let a = match artifact {
            Some(id) => m.artifact(id),
            None => None,
        }
        .or_else(|| {
            m.weights()
                .find(|a| state.registry.is_artifact_installed(a))
        })
        .or_else(|| m.default_weights())?;
        let kind = if m
            .capability
            .iter()
            .any(|c| c == "embeddings" || c == "rerank")
        {
            ModelKind::Encoder
        } else {
            ModelKind::Generative
        };
        let path = state.registry.models_dir().join(&a.files.first()?.dest);
        return Some((path, a.total_size(), kind, a.workspace.unwrap_or(0)));
    }
    let store = paddock_models::ModelStore::new(state.supervisor.models_dirs().to_vec());
    if let Ok(models) = store.list()
        && let Some(m) = models.into_iter().find(|m| m.id == model)
    {
        let bytes = std::fs::metadata(&m.path).ok()?.len();
        return Some((m.path, bytes, ModelKind::Generative, 0));
    }
    let p = Path::new(model);
    if p.is_file() {
        let bytes = std::fs::metadata(p).ok()?.len();
        return Some((p.to_path_buf(), bytes, ModelKind::Generative, 0));
    }
    None
}

/// Resident bytes of the vision tower this model would serve with, 0 for a
/// text-only one. Same precedence spawn uses (`Registry::resolve` wires the
/// first INSTALLED `Vision` artifact, and the row-level download bundle pulls
/// the DEFAULT one), so the estimate prices the mmproj that will actually
/// load rather than the largest one on offer.
///
/// Charged when the caller asks for vision, exactly like the drafter is
/// charged when it asks for spec - the start form has a vision switch and the
/// supervisor honours it (`supervisor.rs`: `spec.vision == Some(false)` drops
/// the mmproj before spawn), so a vision-off server genuinely does not pay
/// this. This used to be unconditional on the belief that no toggle existed,
/// which over-charged every vision-off estimate by the whole tower - 0.9 GB on
/// qwen3.8-27b. The CALLER decides; this function still answers
/// "what would the tower cost", which is a question with one answer.
///
/// Pricing a not-yet-downloaded tower over-states by exactly its file size,
/// which is the safe direction for a fit check and is reported as its own line
/// either way.
///
/// The file's bytes are the resident bytes, for every family: all three vision
/// loaders keep their weight planes at 16 bits and accumulate in f32 (granite
/// gemma4 + qwen3.5/3.6). This used to be a floor - those
/// two widened every plane to f32 and held about twice the file - and the fix
/// went into the loaders rather than a per-family multiplier here, which would
/// have been a magic constant in the wrong crate that went stale the moment
/// they were fixed.
pub(crate) fn tower_bytes(
    m: &crate::registry::CatalogModel,
    reg: &crate::registry::Registry,
) -> u64 {
    use crate::registry::ArtifactKind;
    // Vision and Audio towers are the same KIND of cost - an mmproj companion
    // held from startup to shutdown - so they are charged by one rule. Only
    // the capability they imply differs, and that is the catalog's business,
    // not this function's.
    let tower = || {
        m.artifacts
            .iter()
            .filter(|a| matches!(a.kind, ArtifactKind::Vision | ArtifactKind::Audio))
    };
    tower()
        .find(|a| reg.is_artifact_installed(a))
        .or_else(|| tower().find(|a| a.default))
        .or_else(|| tower().next())
        // file bytes + the persistent workspace the tower pins at attach
        // (catalog data, measured per release - deepseek-ocr's encode slabs
        // are ~950 MiB, more than its weight file; see CatalogArtifact::
        // workspace for why this is a manifest field and not a constant here)
        .map_or(0, |a| a.total_size() + a.workspace.unwrap_or(0))
}

pub async fn handle(
    State(state): State<Arc<crate::routes::AppState>>,
    Query(q): Query<EstimateQuery>,
) -> Response {
    let asked_kv = match q.kv.as_deref() {
        Some("fp8_e4m3" | "fp8") => KvDtype::Fp8E4m3,
        _ => KvDtype::F16,
    };
    // Price what the RUNNER will serve, not what was asked. On a card with no
    // FP8 tensor cores `serving.rs::apply_kv_dtype` downgrades fp8 to f16 and
    // says so in its log; an estimate that kept the fp8 rate would report half
    // the KV pool the server then allocates - the panel saying "fits" about a
    // configuration twice the size it drew. One shared predicate,
    // so this cannot drift from the runner's own gate.
    let cc = q.cc.as_deref().and_then(parse_cc);
    let kv_blocked = cc.and_then(paddock_models::gpu_support::fp8_kv_blocked);
    let kv_downgraded = asked_kv == KvDtype::Fp8E4m3 && kv_blocked.is_some();
    let env = Envelope {
        concurrency: q.batch.unwrap_or(state.max_batch as u64).max(1),
        kv_dtype: if kv_downgraded {
            KvDtype::F16
        } else {
            asked_kv
        },
        // filled per model below - the drafter is a per-model artifact
        spec: None,
        offload: q
            .offload_ram_gb
            .filter(|g| *g > 0.0)
            .map(|g| paddock_estimator::OffloadCost::armed((g * (1u64 << 30) as f64) as u64)),
    };
    let want_spec = q.spec.unwrap_or(false);
    // Absent = charge it, so a caller that never heard of this parameter keeps
    // the old (heavier, safer) answer.
    let want_vision = q.vision.unwrap_or(true);

    // Free VRAM, not total: the engine sizes against what is actually
    // available, and so must anything claiming to predict it. Multi-GPU:
    // the caller picks the device (`gpu` = NVML index, default 0) and both
    // the free figure and the reclaimable fleet VRAM are per-THAT-device.
    let snap = state.gpu.latest();
    let sel = q.gpu.unwrap_or(0);
    let gpu = snap.gpus.iter().find(|g| g.index == sel);
    // The running fleet's allocator self-reports (model_mem = weights +
    // KV/state pools), via the reconciler's join over the admin pipes. That
    // VRAM comes back when runners are stopped/switched, so the estimate adds
    // it to what a swapped-in model could have. Only runners attributed to
    // the selected device count; on a single-GPU box a runner NVML can't
    // attribute (WDDM blind spot) can only be here, so it counts too - on a
    // multi-GPU box an unattributable runner honestly counts for none.
    //
    // §10.1 policy rides on top of the fit math: PINNED runners (the resident
    // embedder, the prod endpoint) are never auto-stopped to make room, so
    // their VRAM is not reclaimable-by-swap and they never appear as eviction
    // candidates. Who yields is policy; the estimator only answers fit.
    let single_gpu = snap.gpus.len() <= 1;
    let fleet = state.supervisor.fleet_meta().await;
    let pinned_ports: std::collections::HashSet<u16> = fleet
        .iter()
        .filter(|(_, _, pinned)| *pinned)
        .map(|(p, _, _)| *p)
        .collect();
    let recon = state.recon.borrow().clone();
    let on_device: Vec<&crate::telemetry::RunnerVram> = match &*recon {
        Some(r) => r
            .runners
            .iter()
            .filter(|rv| rv.gpu == Some(sel) || (single_gpu && rv.gpu.is_none()))
            .collect(),
        None => Vec::new(),
    };
    let reclaimable: u64 = on_device
        .iter()
        .filter(|rv| !pinned_ports.contains(&rv.port))
        .filter_map(|rv| rv.self_mem)
        .sum();
    // Eviction order (llama-swap's `evict_costs` lesson, computed instead of
    // hand-declared): unpinned device runners, cheapest-to-restore first -
    // restore cost ≈ bytes of weights to reload, so a small fast-loading model
    // yields before a 30 GB one. Falls back to the runner's resident VRAM when
    // the weights can't be resolved (labeled, never silent).
    let mut eviction: Vec<serde_json::Value> = on_device
        .iter()
        .filter(|rv| !pinned_ports.contains(&rv.port))
        .map(|rv| {
            let model = fleet
                .iter()
                .find(|(p, _, _)| *p == rv.port)
                .and_then(|(_, m, _)| m.clone());
            let weights = model
                .as_deref()
                .and_then(|m| resolve_weight_bytes(&state, m));
            let vram = rv.self_mem.or(rv.nvml_mem).unwrap_or(0);
            let (cost, basis) = match weights {
                Some(w) => (w, "weights"),
                None => (vram, "vram"),
            };
            serde_json::json!({
                "port": rv.port,
                "model": model,
                "vram": vram,
                "evict_cost": cost,
                "cost_basis": basis,
            })
        })
        .collect();
    eviction.sort_by_key(|e| e["evict_cost"].as_u64().unwrap_or(u64::MAX));
    let pinned_vram: u64 = on_device
        .iter()
        .filter(|rv| pinned_ports.contains(&rv.port))
        .filter_map(|rv| rv.self_mem)
        .sum();
    // `model_mem` covers weights + KV/state pools but not the CUDA context,
    // cuBLAS workspaces and allocator slack the process also holds (~2.5 GB
    // here). Those are exactly what the estimate's own graph margin budgets
    // for, so leaving them in "used by others" charges them twice and
    // under-reports what a swapped-in model could have.
    let others_raw = gpu
        .and_then(|g| g.mem_used)
        .unwrap_or(0)
        .saturating_sub(reclaimable);
    let in_use_by_others = if reclaimable > 0 {
        others_raw.saturating_sub(paddock_estimator::GRAPH_MARGIN)
    } else {
        others_raw
    };
    let budget_bytes = q.budget.map(|mib| mib << 20);
    let device = gpu.and_then(|g| g.mem_total).map(|t| Device {
        // The ceiling caps what is on offer; it never invents room that is not
        // there, so it is a min() against real free VRAM rather than a
        // replacement for it.
        free_bytes: {
            let free = t.saturating_sub(in_use_by_others);
            budget_bytes.map_or(free, |b| free.min(b))
        },
        total_bytes: t,
    });

    let models_dir = state.registry.models_dir().to_path_buf();
    let mut rows = serde_json::Map::new();
    for m in &state.registry.catalog().models {
        // Embedding and rerank models are served by the encoder path - one
        // forward pass per input, nothing cached between calls. They must not
        // be priced with a decode cache.
        let kind = if m
            .capability
            .iter()
            .any(|c| c == "embeddings" || c == "rerank")
        {
            ModelKind::Encoder
        } else {
            ModelKind::Generative
        };
        // Speculation, priced per model. A separate drafter artifact is
        // resident weights; in-file MTP (qwen3.5/3.6 `nextn`) contributes 0
        // because those tensors already sit inside the weights file we are
        // counting. Either way the verify plane widens, which is the term the
        // Default carries. Only offered for models the engine can speculate
        // for, so an unsupported row is never priced for a thing it cannot do.
        let spec = (want_spec && m.capability.iter().any(|c| c == "speculative")).then(|| {
            let drafter_bytes = m
                .artifacts
                .iter()
                .find(|a| a.kind == crate::registry::ArtifactKind::Drafter)
                .map_or(0, |a| a.total_size());
            paddock_estimator::SpecCost {
                drafter_bytes,
                ..Default::default()
            }
        });
        let env = Envelope { spec, ..env };
        // The vision tower is shared across the weights alternatives - one
        // mmproj serves the Q8 and the Q4 alike - so its bytes belong in every
        // artifact row, not in one of them.
        let tower = if want_vision {
            tower_bytes(m, &state.registry)
        } else {
            0
        };
        // One row per WEIGHTS ARTIFACT (schema 3): Q8 and Q4 are different
        // footprints of one model, and the picker's fit verdicts need both.
        let mut art_rows = serde_json::Map::new();
        // The checkpoint's own architecture, learned from whichever artifact we
        // could probe. Needed for the elected sampling profile below, which is
        // keyed on arch - and deliberately only known for a DOWNLOADED model,
        // because the arch is read from the file rather than declared.
        let mut arch: Option<String> = None;
        for a in m.weights() {
            let weights = a.total_size();
            let published = a.shape.clone();
            // Only an installed file can be probed. Rather than guess geometry
            // for the rest, say so: "download to measure" is a true answer,
            // and the disk size we do know is still shown.
            //
            // LAZY, because a probe reads up to a 256 MB header prefix and the
            // cache is cold on the first request after a restart. An artifact
            // that publishes a shape needs nothing from the file (26 of 33
            // and the one thing the probe still supplies -
            // `arch`, for the elected-sampling row - is per MODEL, so the
            // first artifact that yields it ends the probing for this model.
            // Probing every installed artifact and discarding most of the
            // results was seconds of cold I/O on the first paint of the
            // Start/Edit page, all of it for numbers the published block
            // already carried.
            let probed = (published.is_none() || arch.is_none())
                .then(|| {
                    a.files
                        .first()
                        .map(|f| models_dir.join(&f.dest))
                        .filter(|p| p.exists())
                        .and_then(|p| state.probes.get(&p))
                })
                .flatten();

            if let Some(r) = probed.as_ref()
                && arch.is_none()
            {
                arch.clone_from(&r.architecture);
            }

            // One shape, whether or not the file is here. The
            // published block wins over a local probe rather than being its
            // fallback, and that ordering is the point: it carries RESIDENT
            // weight bytes, which the probe cannot produce - `total_size()` is
            // the file, and the loader repacks on the way to the GPU (a Q4_K
            // costs ~13.7% more in VRAM than on disk). Preferring the
            // probe when installed would keep exactly the two-answer split this
            // was built to remove, with the worse number winning after the
            // download.
            //
            // Probe geometry still fills in for an artifact published before
            // this existed, and for a format the generator cannot read.
            let shape_source = published.as_ref().map(|s| s.source);
            let shape = published
                .map(|s| s.into_model_shape(tower, a.workspace.unwrap_or(0)))
                .or_else(|| {
                    probed.as_ref().map(|r| ModelShape {
                        tower_bytes: tower,
                        // The artifact's declared workspace (MoE serving
                        // scratch, measured per release) is resident from load,
                        // same as the tower - see CatalogArtifact::workspace.
                        workspace_bytes: a.workspace.unwrap_or(0),
                        ..ModelShape::from_report(r, weights, kind)
                    })
                });
            // What the row SAYS the weights cost: resident where we know it,
            // the file size otherwise - never a scaled guess. And the same
            // weights term the estimate itself used, which means subtracting
            // in-file nextn when speculation is off, because the engine does
            // not load those blocks then. FitChart derives the scratch band as
            // `resident - weights - tower - workspace`, so a `weights` bigger
            // than the one inside `resident` eats the whole band and clamps it
            // to zero - on nemotron that is 1.42 GiB of chart.
            let shown = shape.as_ref().map_or(weights, |s| {
                if env.spec.is_some() {
                    s.weight_bytes
                } else {
                    s.weight_bytes.saturating_sub(s.nextn_bytes)
                }
            });
            let source = match shape_source {
                Some(paddock_estimator::ShapeSource::Measured) => "measured",
                Some(paddock_estimator::ShapeSource::Probed) => "probed",
                None => "file",
            };

            let row = match (shape, device) {
                (Some(shape), Some(dev)) => {
                    let est = paddock_estimator::estimate(&shape, &env, &dev);
                    // the whole trade-off in one payload: how the window shrinks as
                    // sessions are added, so the UI never has to guess or re-ask
                    let curve: Vec<_> = paddock_estimator::ctx_curve(&shape, &dev, &env, &CURVE)
                        .into_iter()
                        .map(|(n, ctx)| serde_json::json!({ "at": n, "ctx": ctx }))
                        .collect();
                    serde_json::json!({
                        "known": true, "kind": kind, "weights": shown,
                        "tower": tower, "weights_source": source,
                        "estimate": est, "curve": curve,
                    })
                }
                // A shape but no GPU telemetry: the footprint is still real, there
                // is just nothing to compare it against.
                (Some(shape), None) => serde_json::json!({
                    "known": true,
                    "kind": kind,
                    "weights": shown,
                    "tower": tower,
                    "weights_source": source,
                    "kv_bytes_per_token": shape.kv_per_sequence(1, env.kv_dtype),
                    "reason": "no GPU telemetry - cannot judge fit",
                }),
                // No published shape and nothing to probe. Today this is the
                // safetensors lane only (nemotron's NVFP4 arm): probe_path is
                // GGUF-only, so the shapes generator cannot generate a block for
                // it. Still the honest answer rather than a scaled file size.
                (None, _) => serde_json::json!({
                    "known": false,
                    "kind": kind,
                    "weights": weights,
                    "tower": tower,
                    "weights_source": source,
                    "reason": "VRAM for this format is measured from a load, not guessed",
                }),
            };
            art_rows.insert(a.id.clone(), row);
        }
        // The decoding parameters this checkpoint's own authors published
        // The runner resolves pin -> election -> wire, so a form
        // that shows a blank field has to be able to say what blank RESOLVES
        // to; without this it could only claim "the model's default" and hope.
        //
        // `instruct` rides along when the card publishes a second row for
        // thinking-off (the qwen family does), because "the default" is then
        // genuinely two values and picking one to display would be a guess.
        let knobs = |k: &paddock_models::sampling::Knobs| {
            serde_json::json!({
                "temperature": paddock_models::sampling::as_written(k.temperature),
                "top_k": k.top_k,
                "top_p": paddock_models::sampling::as_written(k.top_p),
                "min_p": paddock_models::sampling::as_written(k.min_p),
            })
        };
        let sampling = arch
            .as_deref()
            .and_then(paddock_models::sampling::elected)
            .map(|e| {
                let mut o = knobs(&e.thinking).as_object().cloned().unwrap_or_default();
                o.insert("source".into(), serde_json::json!(e.source));
                if let Some(i) = e.instruct {
                    o.insert("instruct".into(), knobs(&i));
                }
                serde_json::Value::Object(o)
            });
        rows.insert(
            m.id.clone(),
            serde_json::json!({ "kind": kind, "artifacts": art_rows, "sampling": sampling }),
        );
    }

    Json(serde_json::json!({
        "envelope": {
            "batch": env.concurrency,
            // The width these numbers were PRICED at, which is what the runner
            // will serve - not necessarily what was asked for. A UI labelling
            // its KV row from its own form control instead of from here can
            // say "8-bit" over bytes counted at 16.
            "kv_dtype": env.kv_dtype,
            "kv_asked": asked_kv,
            // Set when the two differ, with the hardware reason, so the panel
            // can say why rather than silently showing a bigger number than
            // the control implies.
            "kv_downgraded": kv_downgraded.then_some(kv_blocked).flatten(),
            // The ceiling these numbers were priced under, so a panel can say
            // the fit is against the LIMIT rather than against the card.
            "budget": budget_bytes,
            // Whether the vision/audio tower is in these numbers. Mirrors the
            // start form's switch and the supervisor's `spec.vision`.
            "vision": want_vision,
            // The server's own --max-ctx caps what it will actually serve,
            // independently of what the card could back. A model may report a
            // 262144 window while this server is configured for 32768; the UI
            // has to be able to say so rather than promise the larger number.
            "server_ctx": state.max_ctx,
            "server_batch": state.max_batch,
            // What the prefix cache was priced with, so a panel can say the
            // context it draws is the one an ARMED tier leaves.
            "offload_ram_gb": q.offload_ram_gb.filter(|g| *g > 0.0),
        },
        // Host memory: the resource prefix-cache offload actually spends, and
        // the one this manager never used to price. `total` is None on a
        // platform we cannot ask (see hostmem) - the panel then shows the
        // commitment without a denominator rather than inventing one.
        "host": {
            "total": crate::hostmem::total_bytes(),
            // ceilings other endpoints have already promised their caches;
            // every one is reachable at once, so it is what to subtract
            "committed": crate::hostmem::committed_bytes(
                state
                    .supervisor
                    .configured_offload_ram_gb()
                    .into_iter()
                    .map(Some),
            ),
            "requested": q
                .offload_ram_gb
                .filter(|g| *g > 0.0)
                .map(|g| (g * (1u64 << 30) as f64) as u64),
            // What the folder field defaults to. The store appends its own
            // `kv-cache` segment inside this, so showing the ROOT means the
            // placeholder is something a user could paste back verbatim
            // without landing in kv-cache/kv-cache.
            "cache_dir": paddock_admin::data_root().display().to_string(),
        },
        "device": device.map(|d| serde_json::json!({
            // which GPU this estimate priced (NVML index) - label it
            "index": sel,
            // Two different numbers, and conflating them reads as a bug: a card
            // showing "38 of 48 GB used" cannot also be "43 GB free". `free` is
            // what a model would GET (the loaded one is released first);
            // `free_now` is what is unallocated at this instant. The UI must
            // label which it is showing.
            "free": d.free_bytes,
            "free_now": d.total_bytes.saturating_sub(gpu.and_then(|g| g.mem_used).unwrap_or(0)),
            "total": d.total_bytes,
            "name": gpu.map(|g| g.name.clone()),
            "held_by_loaded_model": reclaimable,
            "used_by_others": in_use_by_others,
            // §10.1 pinned runners: resident by policy, so their VRAM is part
            // of used_by_others for the fit math - this labels the subset so
            // the UI can say "of which pinned: X" instead of "other apps".
            "held_by_pinned": pinned_vram,
            // Who yields when room must be made (unpinned device runners,
            // cheapest-to-restore first). Policy input for serve-also/compare;
            // the estimator itself never stops anything.
            "eviction": eviction,
            // paddock's CUDA context / cuBLAS workspaces / allocator slack:
            // real, resident, and not inside `model_mem`. Reported as its own
            // line so total - others - runtime - model == free_now actually
            // balances; folding it into "other apps" left a ~2 GB hole in a
            // tooltip whose entire job is to reconcile against nvidia-smi.
            "paddock_runtime": if reclaimable > 0 { paddock_estimator::GRAPH_MARGIN } else { 0 },
        })),
        "models": rows,
    }))
    .into_response()
}

#[cfg(test)]
mod tests {
    use crate::registry::{ArtifactKind, Registry};

    // NOTE: the float-noise invariant is tested where the helper lives
    // (`paddock_models::sampling::as_written`), against the whole elected
    // table. Both this endpoint and the runner's capability surface publish
    // the same numbers through it, so one test covers both.

    /// The query field is deserialized by NAME, so a rename or a typo on
    /// either side degrades to "absent" - which is silently the old, wrong
    /// answer rather than an error. This is the one plumbing failure the
    /// arithmetic tests in `paddock-estimator` cannot see, so it is checked
    /// where the name actually crosses the wire.
    #[test]
    fn the_offload_budget_survives_the_query_string() {
        // through the real extractor, so this is the path a request takes
        let parse = |qs: &str| -> super::EstimateQuery {
            let uri: axum::http::Uri = format!("/api/models/estimate?{qs}").parse().unwrap();
            axum::extract::Query::try_from_uri(&uri)
                .map(|axum::extract::Query(q)| q)
                .expect("parse")
        };
        assert_eq!(
            parse("batch=4&kv=f16&offload_ram_gb=24").offload_ram_gb,
            Some(24.0)
        );
        // absent stays absent - no tier, and no accidental default that would
        // charge staging to an endpoint that never armed one
        assert_eq!(parse("batch=4").offload_ram_gb, None);
        // and 0 is treated as "no tier" downstream, not as an armed one
        let zero = parse("offload_ram_gb=0");
        assert_eq!(zero.offload_ram_gb, Some(0.0));
        assert!(zero.offload_ram_gb.filter(|g| *g > 0.0).is_none());
    }

    /// The estimate has to price the width the RUNNER will serve. Before this,
    /// the manager had no compute-capability input at all, so on an A6000 it
    /// counted fp8 bytes for a cache the runner then allocated at f16 - half
    /// the pool it drew.
    #[test]
    fn the_kv_width_follows_the_card_not_the_request() {
        use paddock_models::gpu_support::fp8_kv;
        assert_eq!(super::parse_cc("8.6"), Some((8, 6)));
        assert_eq!(super::parse_cc("12.0"), Some((12, 0)));
        // a malformed string must not gate - absent means "honour the request",
        // the same fail-open stance the runner's own device singleton takes
        assert_eq!(super::parse_cc("ampere"), None);
        assert_eq!(super::parse_cc("8"), None);

        // the property, stated against the two cards this actually decides
        // the estimator must price the width the runner will actually serve -
        // and that is fp8 on Ampere too (gpu_support::fp8_kv)
        assert!(
            fp8_kv(super::parse_cc("8.6").unwrap()),
            "A6000 is no longer downgraded"
        );
        assert!(
            fp8_kv(super::parse_cc("12.0").unwrap()),
            "Blackwell keeps fp8"
        );
    }

    /// The bug this closes was invisible by construction: the estimate looped
    /// over `m.weights()`, so a companion could never appear in it no matter
    /// how large. Assert the property over the whole catalog rather than one
    /// model, so adding a vision model without pricing its tower fails here.
    #[test]
    fn every_tower_model_prices_its_tower_and_no_one_else_does() {
        // a models dir that cannot exist, so nothing reads as installed and
        // the precedence falls through to the artifact the bundle would pull
        let reg = Registry::new(std::path::PathBuf::from("./this-dir-does-not-exist"));
        let mut priced = 0;
        for m in &reg.catalog().models {
            let tower = super::tower_bytes(m, &reg);
            let has_vision = m.artifacts.iter().any(|a| a.kind == ArtifactKind::Vision);
            let has_audio = m.artifacts.iter().any(|a| a.kind == ArtifactKind::Audio);
            let has_tower = has_vision || has_audio;
            assert_eq!(
                has_tower,
                tower > 0,
                "{}: an mmproj artifact must be charged",
                m.id
            );
            // An AUDIO tower and the `transcription` capability are the same
            // claim said twice - the picker must not offer speech input for a
            // model priced as text-only, nor price a tower it never labels.
            // Whisper is the deliberate exception on one side: it transcribes
            // with no mmproj at all, because its audio encoder ships inside
            // the weights file rather than as a companion. So the implication
            // runs one way only: audio tower => transcription.
            let claims_audio = m.capability.iter().any(|c| c == "transcription");
            assert!(
                !has_audio || claims_audio,
                "{}: an audio mmproj must come with the `transcription` capability",
                m.id
            );
            // the catalog's own two claims about images have to agree, or the
            // picker offers image input for a model priced as text-only.
            //
            // "Takes images" is not one capability string: `vision` is general
            // image chat, `documents` is granite-vision's structured extraction
            // (IBM's card says it may not generalize past that, so it must not
            // wear the general chip - models.toml explains the split). What
            // must hold is that a tower comes with exactly one of them. A new
            // image capability has to be added here deliberately; forgetting
            // fails this assertion rather than quietly shipping a model the
            // picker prices but never labels.
            const IMAGE_CAPS: [&str; 2] = ["vision", "documents"];
            let claims_images = m
                .capability
                .iter()
                .filter(|c| IMAGE_CAPS.contains(&c.as_str()))
                .count();
            assert_eq!(
                claims_images,
                usize::from(has_vision),
                "{}: exactly one image capability ({IMAGE_CAPS:?}) goes with a vision artifact",
                m.id
            );
            // ...and an image capability must never ride on an audio tower.
            assert!(
                !has_audio || claims_images == 0,
                "{}: an audio mmproj must not claim image input",
                m.id
            );
            // A tower's `workspace` bytes ride the same charge as its file
            // bytes - deepseek-ocr's encode slabs are BIGGER than its mmproj
            // file, and a fit that skips them is off by a gigabyte.
            if let Some(a) = m
                .artifacts
                .iter()
                .find(|a| a.kind.is_mmproj() && a.workspace.is_some())
            {
                assert_eq!(
                    tower,
                    a.total_size() + a.workspace.unwrap(),
                    "{}: the tower's workspace must be charged on top of its file",
                    m.id
                );
            }
            priced += u32::from(has_tower);
        }
        assert!(priced > 0, "the catalog has tower models to price");
        // the field is live in the shipped catalog: unlimited-ocr declares it
        let reg = Registry::new(std::path::PathBuf::from("./this-dir-does-not-exist"));
        let ocr = reg
            .catalog()
            .models
            .iter()
            .find(|m| m.id == "unlimited-ocr");
        let ocr = ocr.expect("unlimited-ocr row in models.toml");
        assert!(
            ocr.artifacts
                .iter()
                .any(|a| a.workspace.unwrap_or(0) > 900 << 20),
            "unlimited-ocr's mmproj must declare its ~950 MiB workspace"
        );
    }
}
