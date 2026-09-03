//! The manager-side collector: one discovery loop that keeps, per live runner
//! INSTANCE, an event-ring subscriber (§8.1, `activity` rows) and a metrics
//! scraper (`usage_*` rows) - plus the lifecycle
//! bookkeeping both tiers hang off: `service_generation` bands opened when a
//! generation is first seen and closed when it vanishes, and `usage_gap` rows
//! wherever the record has a hole (a blind manager window, an unobserved
//! restart, an overrun ring).
//!
//! Instances are keyed by their per-process-start UUID from identify
//! - the load-bearing identity; pid and started_at only corroborate.
//!   Keying on (port, started_at) collided when two generations started inside
//!   one second, and the second generation's records were silently discarded.
//!
//! Retention is a stated knob (`activity_retention_days`, purged hourly);
//! the 5-minute usage grain keeps a fixed ~90 days (hourly grain is forever,
//! ~13 MB/yr). Persistence has its own switch (`activity` mode): `full` runs
//! both tiers, `aggregates` runs only the metrics tier (counts without
//! content), `off` runs neither - not recording is a first-class
//! configuration.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use paddock_admin::client::{AdminClient, AdminError};

use crate::config::ActivityMode;
use crate::store::Store;
use crate::usage::{self, SeriesKey, Snapshot};

/// One collector's identity key: a runner INSTANCE, not just a port.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct InstanceKey {
    port: u16,
    pid: u32,
    started_at: u64,
    instance_id: String,
}

/// What identify said about the instance beyond its key - the generation
/// band's content, read per GENERATION and never cached per port (§5.0.1: a
/// port-keyed model cache would relabel a successor's traffic).
#[derive(Debug, Clone)]
struct InstanceMeta {
    version: String,
    model: Option<String>,
    embedder: Option<String>,
    asr: Option<String>,
    aligner: Option<String>,
    has_events: bool,
    has_metrics: bool,
    has_snapshots: bool,
}

struct Attached {
    events: Option<tokio::task::JoinHandle<()>>,
    metrics: Option<tokio::task::JoinHandle<()>>,
}

impl Attached {
    fn abort(&self) {
        if let Some(h) = &self.events {
            h.abort();
        }
        if let Some(h) = &self.metrics {
            h.abort();
        }
    }
    /// Every task that was spawned has returned (runner likely died between
    /// discovery ticks) - with none running, the entry can drop so the next
    /// tick re-attaches; the band is only touched when identify agrees.
    fn all_done(&self) -> bool {
        let spawned = self.events.is_some() || self.metrics.is_some();
        spawned
            && self
                .events
                .as_ref()
                .is_none_or(tokio::task::JoinHandle::is_finished)
            && self
                .metrics
                .as_ref()
                .is_none_or(tokio::task::JoinHandle::is_finished)
    }
}

const SCRAPE_INTERVAL: Duration = Duration::from_secs(15);
/// A scrape loop that wakes to find this much time missing was not sleeping
/// between scrapes - the machine (or the manager) was. The delta's shape is
/// unknowable, so it becomes a gap row, never a false spike (§5.3).
const BLIND_WINDOW_MS: i64 = 120_000;
const USAGE_5M_RETENTION_MS: i64 = 90 * 86_400_000;

fn now_ms() -> i64 {
    chrono::Utc::now().timestamp_millis()
}

/// Spawn the discovery loop. `retention_days = 0` disables the age-based
/// activity purge (keep forever until an explicit user purge).
pub fn start(db: Arc<Store>, retention_days: u32, mode: ActivityMode) {
    tokio::spawn(async move {
        let mut tasks: HashMap<InstanceKey, Attached> = HashMap::new();
        let mut last_purge = tokio::time::Instant::now();
        loop {
            // Discover live runner instances (every runner with an admin
            // surface gets a lifecycle band; tiers attach per capability).
            let mut live: Vec<(InstanceKey, InstanceMeta)> = Vec::new();
            for port in paddock_admin::enumerate() {
                let c = AdminClient::new(port);
                if let Ok(Ok(id)) = tokio::time::timeout(Duration::from_secs(2), c.identify()).await
                {
                    // An earlier runner reports no instance_id: synthesize the
                    // same id the store migration stamped on old rows, so its
                    // cursor lines up (accepting the old second-resolution
                    // semantics for old binaries only).
                    let instance_id = if id.instance_id.is_empty() {
                        format!("legacy-{}-{}", port, id.started_at_unix)
                    } else {
                        id.instance_id.clone()
                    };
                    live.push((
                        InstanceKey {
                            port,
                            pid: id.pid,
                            started_at: id.started_at_unix,
                            instance_id,
                        },
                        InstanceMeta {
                            version: id.version.clone(),
                            model: id.model.clone(),
                            embedder: id.embedder.clone(),
                            asr: id.asr.clone(),
                            aligner: id.aligner.clone(),
                            has_events: id.capabilities.iter().any(|c| c == "events"),
                            has_metrics: id.capabilities.iter().any(|c| c == "metrics"),
                            has_snapshots: id.capabilities.iter().any(|c| c == "metrics-snapshots"),
                        },
                    ));
                }
            }

            // Reap collectors whose instance vanished, closing its band. A
            // live instance on the same port at the moment of death is the
            // takeover signature; otherwise the cause is honestly unknown
            // (the stop route stamps 'stopped' when the manager did it -
            // COALESCE in close_generation keeps that stamp).
            tasks.retain(|key, t| {
                let alive = live.iter().any(|(k, _)| k == key);
                if !alive {
                    t.abort();
                    let takeover = live.iter().any(|(k, _)| k.port == key.port);
                    let cause = if takeover { "takeover" } else { "unknown" };
                    if let Err(e) = db.close_generation(&key.instance_id, now_ms(), cause) {
                        tracing::warn!(port = key.port, %e, "closing generation band failed");
                    }
                    tracing::info!(
                        port = key.port,
                        instance = %key.instance_id,
                        cause,
                        "runner generation ended"
                    );
                    return false;
                }
                !t.all_done()
            });

            // Attach to new instances.
            for (key, meta) in live {
                if tasks.contains_key(&key) {
                    continue;
                }
                let started_ms = key.started_at as i64 * 1000;
                // An open band on this port that is not this instance died
                // while nobody watched (an observed death is closed at reap).
                // Close it at the successor's start and record the hole - the
                // predecessor's counters died with it, so the lost totals are
                // genuinely unknown (§5.3, the one case only a journal covers).
                if let Ok(Some(old)) = db.open_predecessor_on_port(key.port, &key.instance_id) {
                    let _ = db.close_generation(&old, started_ms, "unknown");
                    if let Ok(Some(last)) = db.last_scrape_of(&old) {
                        let _ = db.insert_usage_gap(
                            key.port,
                            &old,
                            last,
                            started_ms,
                            "runner-restart-unobserved",
                            None,
                            None,
                        );
                    }
                }
                match db.open_generation(
                    &key.instance_id,
                    key.port,
                    key.pid,
                    &meta.version,
                    meta.model.as_deref(),
                    meta.embedder.as_deref(),
                    meta.asr.as_deref(),
                    meta.aligner.as_deref(),
                    started_ms,
                ) {
                    Ok(true) => tracing::info!(
                        port = key.port,
                        pid = key.pid,
                        instance = %key.instance_id,
                        "runner generation opened"
                    ),
                    Ok(false) => {} // re-attach after a manager restart
                    Err(e) => tracing::warn!(port = key.port, %e, "opening generation band failed"),
                }

                let events = (mode == ActivityMode::Full && meta.has_events).then(|| {
                    tracing::info!(port = key.port, "collector attached to runner event ring");
                    tokio::spawn(collect_events(db.clone(), key.clone()))
                });
                let metrics = meta.has_metrics.then(|| {
                    tracing::info!(port = key.port, "collector attached to runner metrics");
                    tokio::spawn(collect_metrics(
                        db.clone(),
                        key.clone(),
                        meta.model.clone(),
                        meta.has_snapshots,
                    ))
                });
                tasks.insert(key, Attached { events, metrics });
            }

            // Hourly retention pass: activity by the knob, the 5-minute usage
            // grain by the fixed ~90-day policy (hourly grain is kept).
            if last_purge.elapsed() > Duration::from_secs(3600) {
                last_purge = tokio::time::Instant::now();
                if retention_days > 0 {
                    let cutoff = now_ms() - i64::from(retention_days) * 86_400_000;
                    match db.purge_activity_before(cutoff) {
                        Ok(0) => {}
                        Ok(n) => {
                            tracing::info!(
                                rows = n,
                                days = retention_days,
                                "activity retention purge"
                            );
                        }
                        Err(e) => tracing::warn!(%e, "activity retention purge failed"),
                    }
                }
                match db.purge_usage_5m_before(now_ms() - USAGE_5M_RETENTION_MS) {
                    Ok(0) | Err(_) => {}
                    Ok(n) => tracing::info!(rows = n, "5-minute usage grain purge"),
                }
            }

            tokio::time::sleep(Duration::from_secs(3)).await;
        }
    });
}

/// Collect one runner instance's ring until it goes away. Long-poll keeps the
/// steady state at one held request per runner; batches insert in one
/// transaction.
async fn collect_events(db: Arc<Store>, key: InstanceKey) {
    let client = AdminClient::new(key.port);
    // Resume across manager restarts: the table is the cursor.
    let mut cursor = match db.activity_cursor(&key.instance_id) {
        Ok(c) => c,
        Err(e) => {
            tracing::warn!(port = key.port, %e, "activity cursor read failed; starting at 0");
            0
        }
    };
    loop {
        match client.events(cursor, 512, 20_000).await {
            Ok(page) => {
                if page.dropped > 0 {
                    // The ring's honesty contract, surfaced manager-side -
                    // logged AND recorded, so the chart can say "the inspector
                    // is missing K requests here" instead of under-reporting.
                    tracing::warn!(
                        port = key.port,
                        dropped = page.dropped,
                        "runner event ring overran this collector: events dropped"
                    );
                    let first_kept = page.next - page.events.len() as u64;
                    let ts = page
                        .events
                        .first()
                        .and_then(|e| e.get("ts_ms").and_then(serde_json::Value::as_i64))
                        .unwrap_or_else(now_ms);
                    let _ = db.insert_usage_gap(
                        key.port,
                        &key.instance_id,
                        ts,
                        ts,
                        "ring-overrun",
                        Some((cursor as i64, first_kept as i64)),
                        None,
                    );
                }
                if !page.events.is_empty() {
                    match db.insert_activity(
                        &key.instance_id,
                        key.port,
                        key.started_at,
                        &page.events,
                    ) {
                        // TRACE, not debug: this fires on every non-empty poll
                        // and says only that the collector did its job. It was
                        // 451 of the 1650 lines in a 5-day manager log - 27% of
                        // the file, none of it actionable, burying the lines
                        // that are. The failures either side of it (a ring
                        // overrun above, an insert error below) stay loud.
                        Ok(n) => tracing::trace!(port = key.port, rows = n, "activity collected"),
                        Err(e) => {
                            tracing::warn!(port = key.port, %e, "activity insert failed; retrying page");
                            tokio::time::sleep(Duration::from_secs(1)).await;
                            continue; // do not advance the cursor past a failed write
                        }
                    }
                }
                cursor = page.next;
            }
            Err(AdminError::Connect { .. }) => {
                // Runner gone - end this collector; discovery re-attaches to a
                // successor instance (with its own key and fresh cursor).
                tracing::debug!(
                    port = key.port,
                    "runner admin endpoint gone; collector ends"
                );
                return;
            }
            Err(e) => {
                // A keep-alive/long-poll connection the runner closed between
                // polls is ROUTINE - the runner logs the same hang-up at trace
                // - mirroring that here keeps DEBUG readable. The
                // short sleep guards against a hot loop if a runner ever
                // closes every request instantly. Real faults keep DEBUG and
                // the full backoff.
                let s = e.to_string();
                if s.contains("connection closed") {
                    tracing::trace!(port = key.port, %e, "event poll connection closed; re-polling");
                    tokio::time::sleep(Duration::from_millis(100)).await;
                } else {
                    tracing::debug!(port = key.port, %e, "event poll failed; backing off");
                    tokio::time::sleep(Duration::from_secs(1)).await;
                }
            }
        }
    }
}

/// Scrape one runner instance's `/v1/metrics` over the pipe every 15 s and
/// fold counter deltas into the usage tables. The first scrape is
/// the ATTACH: anything the counters moved during a blind window is first
/// reconstructed from the runner's 1-minute snapshot ring (§5.4) -
/// full shape, minute resolution; only what the ring cannot cover becomes a
/// gap row with exact totals (§5.3), because one delta covering hours cannot
/// be honestly spread.
async fn collect_metrics(
    db: Arc<Store>,
    key: InstanceKey,
    serving_model: Option<String>,
    has_snapshots: bool,
) {
    let client = AdminClient::new(key.port);
    let started_ms = key.started_at as i64 * 1000;
    let mut ids: HashMap<SeriesKey, i64> = HashMap::new();
    let mut prev: Option<Snapshot> = None;
    let mut prev_at: i64 = 0;
    loop {
        let text = match client.metrics().await {
            Ok(t) => t,
            Err(AdminError::Connect { .. }) => {
                tracing::debug!(port = key.port, "runner admin endpoint gone; scraper ends");
                return;
            }
            Err(e) => {
                tracing::debug!(port = key.port, %e, "metrics scrape failed; backing off");
                tokio::time::sleep(Duration::from_secs(2)).await;
                continue;
            }
        };
        let now = now_ms();
        let snap = usage::parse(&text);

        // May this scrape fold as a normal delta? A blind window in the way -
        // a manager restart (attach against persisted totals), a late arrival
        // (never-scraped generation), or the loop itself losing time (suspend,
        // debugger, wedged runtime) - is recovered from the snapshot ring
        // first; recovery seeds `prev` at the last replayed snapshot so the
        // window's tail folds normally on this very scrape.
        let fold = 'fold: {
            let (from, baseline) = match &prev {
                Some(p) => {
                    if now - prev_at <= BLIND_WINDOW_MS {
                        break 'fold true;
                    }
                    (prev_at, Baseline::Full(p.clone()))
                }
                None => match attach_plan(&db, &key, started_ms, now) {
                    AttachPlan::Fold => break 'fold true,
                    // No prev_at update needed on either seed: the loop's
                    // bottom stamps it `now` after this scrape's delta folds.
                    AttachPlan::Seed(state) => {
                        prev = Some(state);
                        break 'fold true;
                    }
                    AttachPlan::Skip => break 'fold false,
                    AttachPlan::Blind { from, baseline } => (from, baseline),
                },
            };
            if has_snapshots
                && let Some((seed, _)) = recover_window(
                    &db,
                    &client,
                    &key,
                    &mut ids,
                    serving_model.as_deref(),
                    from,
                    now,
                    &baseline,
                )
                .await
            {
                prev = Some(seed);
                break 'fold true;
            }
            window_gap(&db, &key, &baseline, &snap, from, now);
            false
        };

        // Build the scrape's writes as one atomic step: per-series folds,
        // totals absolutes, and the full-state attach baseline commit
        // together, so a crash between them can never make the next attach
        // double-fold or lose an interval (the state is the resume cursor).
        let zero = usage::SeriesCounters::default();
        let mut items: Vec<usage::UsageFoldItem> = Vec::new();
        for (series, cur) in &snap.series {
            let Some(id) = intern(&db, &mut ids, &key, series) else {
                continue;
            };
            // A series born since the last scrape deltas from zero: its
            // whole content is this window's traffic.
            let delta = fold
                .then(|| {
                    usage::series_delta(
                        prev.as_ref()
                            .and_then(|p| p.series.get(series))
                            .unwrap_or(&zero),
                        cur,
                    )
                })
                .flatten();
            items.push(usage::UsageFoldItem {
                series_id: id,
                delta,
                requests: cur.requests,
                input_tokens: cur.input_tokens,
                output_tokens: cur.output_tokens,
                cached_tokens: cur.cached_tokens,
                spec_drafted: 0,
                spec_accepted: 0,
            });
        }

        // Engine-scoped numbers (spec decode, KV occupancy) live on the
        // generation's 'engine' pseudo-series - they have no operation or
        // origin of their own, and one runner serves one model. On a first
        // fold of a young generation there is no prev: the delta comes from
        // zero, the same rule the per-series loop applies.
        let engine_active =
            snap.spec_drafted > 0 || snap.spec_accepted > 0 || snap.kv_pages_used > 0;
        if engine_active {
            let engine_key = SeriesKey {
                operation: "engine".into(),
                origin: "live".into(),
                model: serving_model.clone().unwrap_or_default(),
            };
            if let Some(id) = intern(&db, &mut ids, &key, &engine_key) {
                let delta = fold
                    .then(|| {
                        usage::engine_delta(prev.as_ref().unwrap_or(&Snapshot::default()), &snap)
                    })
                    .flatten();
                items.push(usage::UsageFoldItem {
                    series_id: id,
                    delta,
                    requests: 0,
                    input_tokens: 0,
                    output_tokens: 0,
                    cached_tokens: 0,
                    spec_drafted: snap.spec_drafted,
                    spec_accepted: snap.spec_accepted,
                });
            }
        }

        // Web-search spend: whole-runner counters labelled by provider, so
        // they never mint a series. The absolutes ride on every scrape even
        // when the delta is skipped - that is what keeps lifetime spend exact
        // across a blind window whose shape we refuse to invent.
        let web: Vec<usage::WebFoldItem> = {
            let base = prev.clone().unwrap_or_default();
            let moved: HashMap<String, usage::WebSpend> = if fold {
                usage::web_deltas(&base, &snap).into_iter().collect()
            } else {
                HashMap::new()
            };
            snap.web
                .iter()
                .map(|(provider, abs)| usage::WebFoldItem {
                    provider: provider.clone(),
                    delta: moved.get(provider).copied(),
                    absolute: *abs,
                })
                .collect()
        };

        match serde_json::to_string(&usage::snapshot_to_wire(&snap, now)) {
            Ok(state) => {
                if let Err(e) = db.fold_usage_step(
                    &key.instance_id,
                    started_ms,
                    now,
                    &items,
                    &web,
                    key.port,
                    &state,
                ) {
                    tracing::warn!(port = key.port, %e, "usage step write failed; interval dropped");
                }
            }
            Err(e) => tracing::warn!(port = key.port, %e, "usage state serialize failed"),
        }

        // An idle scrape still proves the manager was watching - that is
        // what keeps the next gap window's edges honest.
        let _ = db.touch_usage_totals(&key.instance_id, now);

        prev = Some(snap);
        prev_at = now;
        tokio::time::sleep(SCRAPE_INTERVAL).await;
    }
}

fn intern(
    db: &Store,
    cache: &mut HashMap<SeriesKey, i64>,
    key: &InstanceKey,
    series: &SeriesKey,
) -> Option<i64> {
    if let Some(id) = cache.get(series) {
        return Some(*id);
    }
    match db.intern_usage_series(key.port, &series.model, &series.operation, &series.origin) {
        Ok(id) => {
            cache.insert(series.clone(), id);
            Some(id)
        }
        Err(e) => {
            tracing::warn!(port = key.port, %e, "usage series intern failed");
            None
        }
    }
}

/// A blind window's baseline: what was known about the counters at its start.
enum Baseline {
    /// The in-memory previous snapshot - the loop itself lost time (suspend,
    /// debugger, wedged runtime). Complete: every column recovers.
    Full(Snapshot),
    /// The persisted totals - a manager restart. Requests + token counters
    /// recover exactly; columns the totals never tracked delta to zero for
    /// the sub-minute head interval and recover fully from the first
    /// snapshot pair on.
    Totals(Vec<usage::TotalRow>),
    /// Nothing - a never-scraped generation the manager arrived late to.
    /// Everything recovers, from zero.
    Zero,
}

enum AttachPlan {
    /// Young never-scraped generation: its whole history honestly fits the
    /// open bucket, fold this scrape normally.
    Fold,
    /// A fresh persisted full state: seed the loop with it and fold this
    /// scrape as a normal delta - a quick manager restart loses NOTHING.
    Seed(Snapshot),
    /// Totals read failed: fold nothing, re-attach on the next scrape.
    Skip,
    /// A blind window `[from, now)` to reconstruct - or record as a gap.
    Blind { from: i64, baseline: Baseline },
}

/// What the attach scrape is looking at. The persisted full state (written
/// atomically with every fold) is the preferred baseline - it makes the
/// window's head recover every column, and a short manager restart fold
/// seamlessly. The four-column totals serve databases from before the state
/// tier existed.
fn attach_plan(db: &Store, key: &InstanceKey, started_ms: i64, now: i64) -> AttachPlan {
    if let Ok(Some((ts, json))) = db.usage_state_of(&key.instance_id)
        && let Ok(wire) = serde_json::from_str::<paddock_admin::types::MetricsSnapshot>(&json)
    {
        let state = usage::snapshot_from_wire(&wire);
        if now - ts <= BLIND_WINDOW_MS {
            return AttachPlan::Seed(state);
        }
        return AttachPlan::Blind {
            from: ts,
            baseline: Baseline::Full(state),
        };
    }
    let stored = match db.usage_totals_for_instance(&key.instance_id) {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!(port = key.port, %e, "usage totals read failed on attach");
            return AttachPlan::Skip;
        }
    };
    if stored.is_empty() {
        if now - started_ms <= BLIND_WINDOW_MS {
            return AttachPlan::Fold;
        }
        return AttachPlan::Blind {
            from: started_ms,
            baseline: Baseline::Zero,
        };
    }
    let from = stored.iter().map(|r| r.last_scrape_ms).max().unwrap_or(now);
    AttachPlan::Blind {
        from,
        baseline: Baseline::Totals(stored),
    }
}

/// Pull every page of the runner's snapshot ring, oldest first, as the
/// manager's own `Snapshot` shape. None on any transport failure - recovery
/// is best-effort and the gap fallback stays honest without it.
async fn pull_snapshots(client: &AdminClient) -> Option<Vec<(i64, Snapshot)>> {
    let mut out = Vec::new();
    let mut since = 0u64;
    loop {
        let page = client.metrics_snapshots(since, 512).await.ok()?;
        if page.snapshots.is_empty() {
            break;
        }
        since = page.next;
        out.extend(
            page.snapshots
                .iter()
                .map(|s| (s.ts_ms as i64, usage::snapshot_from_wire(s))),
        );
    }
    Some(out)
}

/// Reconstruct a blind window `[from, now)` from the runner's snapshot ring
/// replay consecutive snapshot pairs as if they had been live
/// scrapes, each interval folding into the bucket at its snapshot's time.
/// Steps commit atomically with their totals advance, so a crash mid-recovery
/// resumes from the last committed step instead of double-counting. Returns
/// the seed for the live loop - the last replayed snapshot and its time - so
/// the window's tail folds as a normal delta on the very scrape that
/// triggered recovery. None = the ring had nothing for this window.
#[allow(clippy::too_many_arguments)]
async fn recover_window(
    db: &Store,
    client: &AdminClient,
    key: &InstanceKey,
    ids: &mut HashMap<SeriesKey, i64>,
    serving_model: Option<&str>,
    from: i64,
    now: i64,
    baseline: &Baseline,
) -> Option<(Snapshot, i64)> {
    let snaps = pull_snapshots(client).await?;
    let kept: Vec<(i64, Snapshot)> = snaps
        .into_iter()
        .filter(|(ts, _)| *ts > from && *ts <= now)
        .collect();
    if kept.is_empty() {
        return None;
    }
    let base = match baseline {
        Baseline::Full(p) => p.clone(),
        Baseline::Zero => Snapshot::default(),
        Baseline::Totals(rows) => usage::totals_pseudo_baseline(rows, &kept[0].1),
    };
    let plan = usage::plan_recovery(&base, from, &kept);
    // The stretch the ring no longer reaches stays a §5.3 gap - written first
    // so a crash during the folds can only duplicate the hole's record, never
    // silently lose it.
    if let Some((gfrom, gto, lost)) = plan.head_gap {
        tracing::info!(
            port = key.port,
            requests = lost.0,
            gap_ms = gto - gfrom,
            "snapshot ring does not reach the blind window's start: head recorded as a gap"
        );
        let _ = db.insert_usage_gap(
            key.port,
            &key.instance_id,
            gfrom,
            gto,
            "manager-down",
            None,
            Some(lost),
        );
    }
    let started_ms = key.started_at as i64 * 1000;
    let engine_key = SeriesKey {
        operation: "engine".into(),
        origin: "live".into(),
        model: serving_model.unwrap_or_default().to_owned(),
    };
    for step in &plan.folds {
        let cur = &kept[step.idx].1;
        let mut items: Vec<usage::UsageFoldItem> = Vec::new();
        for (series, delta) in &step.series {
            let Some(id) = intern(db, ids, key, series) else {
                continue;
            };
            let abs = cur.series.get(series).cloned().unwrap_or_default();
            items.push(usage::UsageFoldItem {
                series_id: id,
                delta: Some(delta.clone()),
                requests: abs.requests,
                input_tokens: abs.input_tokens,
                output_tokens: abs.output_tokens,
                cached_tokens: abs.cached_tokens,
                spec_drafted: 0,
                spec_accepted: 0,
            });
        }
        if let Some(d) = &step.engine
            && let Some(id) = intern(db, ids, key, &engine_key)
        {
            items.push(usage::UsageFoldItem {
                series_id: id,
                delta: Some(d.clone()),
                requests: 0,
                input_tokens: 0,
                output_tokens: 0,
                cached_tokens: 0,
                spec_drafted: cur.spec_drafted,
                spec_accepted: cur.spec_accepted,
            });
        }
        let web: Vec<usage::WebFoldItem> = step
            .web
            .iter()
            .map(|(provider, d)| usage::WebFoldItem {
                provider: provider.clone(),
                delta: Some(*d),
                absolute: cur.web.get(provider).copied().unwrap_or_default(),
            })
            .collect();
        if items.is_empty() && web.is_empty() {
            continue;
        }
        // Each replayed interval advances the persisted state to its
        // snapshot: crash mid-recovery and the next attach resumes from
        // exactly the last committed interval.
        let Ok(state) = serde_json::to_string(&usage::snapshot_to_wire(cur, step.ts)) else {
            continue;
        };
        if let Err(e) = db.fold_usage_step(
            &key.instance_id,
            started_ms,
            step.ts,
            &items,
            &web,
            key.port,
            &state,
        ) {
            // Same posture as a live fold failure: warn and drop this
            // interval's delta. The state not advancing means nothing
            // downstream ever double-counts.
            tracing::warn!(port = key.port, ts = step.ts, %e, "recovered fold failed; interval dropped");
        }
    }
    let (seed_at, seed) = {
        let last = kept.last().expect("kept is non-empty");
        (last.0, last.1.clone())
    };
    tracing::info!(
        port = key.port,
        window_ms = now - from,
        snapshots = kept.len(),
        folds = plan.folds.len(),
        "blind window reconstructed from the runner's snapshot ring"
    );
    Some((seed, seed_at))
}

/// The §5.3 fallback when no snapshots cover a blind window: exact lost
/// totals into one gap row, folded nowhere - one delta covering the window
/// cannot be honestly spread, and a false spike is worse than a hole.
fn window_gap(
    db: &Store,
    key: &InstanceKey,
    baseline: &Baseline,
    cur: &Snapshot,
    from: i64,
    now: i64,
) {
    let lost = match baseline {
        Baseline::Full(p) => usage::lost_between(p, cur),
        Baseline::Zero => usage::lost_between(&Snapshot::default(), cur),
        Baseline::Totals(rows) => {
            usage::lost_between(&usage::totals_pseudo_baseline(rows, cur), cur)
        }
    };
    if lost != (0, 0, 0) {
        tracing::info!(
            port = key.port,
            requests = lost.0,
            gap_ms = now - from,
            "blind window: totals recovered into a gap row, shape unknown"
        );
        let _ = db.insert_usage_gap(
            key.port,
            &key.instance_id,
            from,
            now,
            "manager-down",
            None,
            Some(lost),
        );
    }
}
