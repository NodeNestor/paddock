//! Usage-metrics ingest: parse a runner's Prometheus
//! exposition into per-series counter snapshots, and turn two snapshots into
//! the non-cumulative deltas the `usage_bucket` tier folds in.
//!
//! Deliberately a parser for our families only - the manager scrapes its own
//! runners over the admin pipe, so the format is the classic text exposition
//! `/metrics` already serves external scrapers (promtool-validated).
//! Unknown families and labels pass through unharmed; an unknown `le`
//! is skipped rather than misfiled, because a ladder change is a collector
//! upgrade, not a guess.

use std::collections::HashMap;

/// The semconv bucket ladder, as the runner PRINTS it (fixed strings, so a
/// float-formatting drift can never mis-index a bucket). Mirrors
/// `paddock-runner`'s `BOUND_LABELS`; `+Inf` rides as index 14.
pub const LADDER: [&str; 14] = [
    "0.01", "0.02", "0.04", "0.08", "0.16", "0.32", "0.64", "1.28", "2.56", "5.12", "10.24",
    "20.48", "40.96", "81.92",
];

/// One series' dimensions as scraped. `model` may be empty - a refused
/// request on a model-less runner has no model label at all.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SeriesKey {
    pub operation: String,
    pub origin: String,
    pub model: String,
}

/// Cumulative counter values for one series, as of one scrape. Histogram
/// arrays stay CUMULATIVE here (exactly as exposed); de-cumulation happens in
/// the delta step. Index 14 is `+Inf` (== the observation count).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SeriesCounters {
    pub requests: u64,
    pub duration_seconds_sum: f64,
    pub errors_4xx: u64,
    pub errors_5xx: u64,
    pub disconnects: u64,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cached_tokens: u64,
    pub e2e: [u64; 15],
    pub ttft: [u64; 15],
}

/// Cumulative web-search spend for one provider, as of one scrape.
///
/// Its own tier rather than a series, because its one dimension is the
/// provider - not (operation, origin, model) - and because two of its three
/// counters are currencies no other row has. Folding it into `usage_series`
/// would have meant writing a provider name into the `model` column and
/// answering "which model is exa" in every model breakdown from then on.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct WebSpend {
    /// Searches executed. The only counter every provider reports, which
    /// makes it the one that must never be inferred from the other two.
    pub requests: u64,
    pub credits: u64,
    pub microdollars: u64,
}

impl WebSpend {
    pub fn is_zero(&self) -> bool {
        self.requests == 0 && self.credits == 0 && self.microdollars == 0
    }
}

/// Everything one scrape says. Spec-decode and KV numbers are engine-scoped
/// (one model per runner), not per-series - they fold into the generation's
/// `operation = "engine"` pseudo-series.
#[derive(Debug, Clone, Default)]
pub struct Snapshot {
    pub start_unix: u64,
    pub series: HashMap<SeriesKey, SeriesCounters>,
    pub spec_drafted: u64,
    pub spec_accepted: u64,
    pub kv_pages_used: u64,
    /// Cumulative spend per provider name (`exa`, `tavily`, ...).
    pub web: HashMap<String, WebSpend>,
}

/// Non-cumulative additions for one bucket row. Histograms are per-bucket
/// increments (h[i] = observations that landed in bucket i alone); the +Inf
/// overflow is not stored - it is derivable (`requests - Σ e2e_h*`, and for
/// TTFT `successes - Σ ttft_h*` where successes = requests - errors -
/// disconnects, since TTFT is observed only on success).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct BucketDelta {
    pub requests: i64,
    pub disconnects: i64,
    pub errors_4xx: i64,
    pub errors_5xx: i64,
    pub input_tokens: i64,
    pub output_tokens: i64,
    pub cached_tokens: i64,
    pub duration_ms_sum: i64,
    pub ttft: [i64; 14],
    pub e2e: [i64; 14],
    pub spec_drafted: i64,
    pub spec_accepted: i64,
    /// A gauge high-water, not an increment: MAX-merged into the row.
    pub kv_pages_max: i64,
}

/// One stored `usage_total` row with its interned dimensions - the attach
/// baseline the collector diffs a fresh scrape against.
#[derive(Debug, Clone)]
pub struct TotalRow {
    pub series_id: i64,
    pub key: SeriesKey,
    pub requests: u64,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cached_tokens: u64,
    pub spec_drafted: u64,
    pub spec_accepted: u64,
    pub last_scrape_ms: i64,
}

// ── read path: what the timeline API serves ─────────────────────

/// One timeline slot: a port's traffic in one (possibly server-side regrouped)
/// bucket. Series dimensions are summed away - the chart's unit is the PORT
/// (port = service identity; a port can be re-pointed, the generations say
/// what it ran). The engine pseudo-series folds in naturally: its request and
/// token columns are zero, so it only contributes spec_drafted/accepted.
#[derive(Debug, Clone, serde::Serialize)]
pub struct UsageSlot {
    pub t: i64,
    pub port: u16,
    pub requests: i64,
    pub errors_4xx: i64,
    pub errors_5xx: i64,
    pub disconnects: i64,
    pub input_tokens: i64,
    pub output_tokens: i64,
    pub cached_tokens: i64,
    pub duration_ms_sum: i64,
    pub spec_drafted: i64,
    pub spec_accepted: i64,
    /// Page-granular KV high-water inside the slot (MAX across the grouped
    /// buckets, not a sum - it is a level, not a flow).
    pub kv_pages_max: i64,
    /// Per-slot increments on the runner's 14-step semconv ladder
    /// (metrics.rs SECONDS_BUCKETS, 0.01..81.92 s) - the duration/TTFT
    /// percentile panels interpolate on these. +Inf overflow is derivable:
    /// requests - Σe2e_h, and successes - Σttft_h (TTFT observes successes
    /// only).
    pub e2e_h: [i64; 14],
    pub ttft_h: [i64; 14],
}

/// One timeline slot of web-search spend: a port's searches on one provider
/// in one (possibly regrouped) bucket. Provider survives the aggregation that
/// series dimensions do not, because it is the whole point - three providers
/// costing three different amounts is the picture, and a summed row would say
/// "128 searches cost $0.89 and 340 credits" without saying who charged what.
#[derive(Debug, Clone, serde::Serialize)]
pub struct WebSlot {
    pub t: i64,
    pub port: u16,
    pub provider: String,
    pub requests: i64,
    pub credits: i64,
    pub microdollars: i64,
}

/// A hole in observation, verbatim from `usage_gap`. The chart draws these as
/// hatched bands - never lets a hole read as "quiet".
#[derive(Debug, Clone, serde::Serialize)]
pub struct UsageGapRow {
    pub id: i64,
    pub port: u16,
    pub from_ts_ms: i64,
    pub to_ts_ms: i64,
    pub cause: String,
    pub lost_requests: Option<i64>,
    pub lost_input_tokens: Option<i64>,
    pub lost_output_tokens: Option<i64>,
    pub from_seq: Option<i64>,
    pub to_seq: Option<i64>,
}

/// One lifecycle band from `service_generation` - what a port was running
/// and why it started/stopped, causes NULL where nobody observed them.
#[derive(Debug, Clone, serde::Serialize)]
pub struct GenerationRow {
    pub instance_id: String,
    pub port: u16,
    pub runner_version: String,
    pub model: Option<String>,
    pub embedder: Option<String>,
    pub asr: Option<String>,
    /// Served forced-alignment model id. The fourth serving role, carried for
    /// the same reason `asr` is: an aligner runner has only this, so a reader
    /// keyed on the other three names it nothing.
    pub aligner: Option<String>,
    pub started_ms: i64,
    pub ended_ms: Option<i64>,
    pub start_cause: Option<String>,
    pub end_cause: Option<String>,
}

impl BucketDelta {
    /// The additive `usage_bucket` columns, in the one canonical order both
    /// the fold SQL and `add_values` are built from. `kv_pages_max` is
    /// deliberately absent - it MAX-merges instead of adding.
    pub fn columns() -> Vec<String> {
        let mut cols: Vec<String> = [
            "requests",
            "disconnects",
            "errors_4xx",
            "errors_5xx",
            "input_tokens",
            "output_tokens",
            "cached_tokens",
            "duration_ms_sum",
        ]
        .into_iter()
        .map(String::from)
        .collect();
        cols.extend((0..14).map(|i| format!("ttft_h{i}")));
        cols.extend((0..14).map(|i| format!("e2e_h{i}")));
        cols.push("spec_drafted".into());
        cols.push("spec_accepted".into());
        cols
    }

    /// Values in exactly `columns()` order.
    pub fn add_values(&self) -> Vec<i64> {
        let mut v = vec![
            self.requests,
            self.disconnects,
            self.errors_4xx,
            self.errors_5xx,
            self.input_tokens,
            self.output_tokens,
            self.cached_tokens,
            self.duration_ms_sum,
        ];
        v.extend_from_slice(&self.ttft);
        v.extend_from_slice(&self.e2e);
        v.push(self.spec_drafted);
        v.push(self.spec_accepted);
        v
    }

    pub fn is_zero(&self) -> bool {
        self.requests == 0
            && self.disconnects == 0
            && self.errors_4xx == 0
            && self.errors_5xx == 0
            && self.input_tokens == 0
            && self.output_tokens == 0
            && self.cached_tokens == 0
            && self.duration_ms_sum == 0
            && self.ttft.iter().all(|v| *v == 0)
            && self.e2e.iter().all(|v| *v == 0)
            && self.spec_drafted == 0
            && self.spec_accepted == 0
            && self.kv_pages_max == 0
    }
}

// ── snapshot recovery: a blind window's shape ────────

/// How far past the blind-window start the oldest available snapshot may sit
/// and still count as covering it: 1.5 snapshot periods. Any further and the
/// ring genuinely does not reach back to the window - the uncovered head
/// becomes an honest gap row instead.
pub const SNAPSHOT_HEAD_SLACK_MS: i64 = 90_000;

/// A runner self-snapshot off the wire, as the manager's own `Snapshot`
/// shape - from here on the recovery path and the live scrape path share
/// every delta rule, so the two cannot disagree about semantics.
pub fn snapshot_from_wire(w: &paddock_admin::types::MetricsSnapshot) -> Snapshot {
    let mut snap = Snapshot {
        start_unix: 0,
        series: HashMap::new(),
        spec_drafted: w.spec_drafted,
        spec_accepted: w.spec_accepted,
        kv_pages_used: w.kv_pages_used,
        web: w
            .web
            .iter()
            .map(|s| {
                (
                    s.provider.clone(),
                    WebSpend {
                        requests: s.requests,
                        credits: s.credits,
                        microdollars: s.microdollars,
                    },
                )
            })
            .collect(),
    };
    for s in &w.series {
        snap.series.insert(
            SeriesKey {
                operation: s.operation.clone(),
                origin: s.origin.clone(),
                model: s.model.clone(),
            },
            SeriesCounters {
                requests: s.requests,
                duration_seconds_sum: s.duration_seconds_sum,
                errors_4xx: s.errors_4xx,
                errors_5xx: s.errors_5xx,
                disconnects: s.disconnects,
                input_tokens: s.input_tokens,
                output_tokens: s.output_tokens,
                cached_tokens: s.cached_tokens,
                e2e: s.e2e,
                ttft: s.ttft,
            },
        );
    }
    snap
}

/// The inverse of [`snapshot_from_wire`]: the manager's own parsed state in
/// the ring's wire shape. This is what the collector PERSISTS as its full
/// attach baseline (`usage_state`) - one serialization for both directions
/// means the persisted baseline and a runner snapshot can never disagree
/// about a column.
pub fn snapshot_to_wire(s: &Snapshot, ts_ms: i64) -> paddock_admin::types::MetricsSnapshot {
    paddock_admin::types::MetricsSnapshot {
        seq: 0,
        ts_ms: ts_ms.max(0) as u64,
        series: s
            .series
            .iter()
            .map(|(k, c)| paddock_admin::types::SnapshotSeries {
                operation: k.operation.clone(),
                origin: k.origin.clone(),
                model: k.model.clone(),
                requests: c.requests,
                errors_4xx: c.errors_4xx,
                errors_5xx: c.errors_5xx,
                disconnects: c.disconnects,
                input_tokens: c.input_tokens,
                output_tokens: c.output_tokens,
                cached_tokens: c.cached_tokens,
                duration_seconds_sum: c.duration_seconds_sum,
                e2e: c.e2e,
                ttft: c.ttft,
            })
            .collect(),
        spec_drafted: s.spec_drafted,
        spec_accepted: s.spec_accepted,
        kv_pages_used: s.kv_pages_used,
        web: s
            .web
            .iter()
            .map(|(provider, w)| paddock_admin::types::WebSpendSeries {
                provider: provider.clone(),
                requests: w.requests,
                credits: w.credits,
                microdollars: w.microdollars,
            })
            .collect(),
    }
}

/// Per-provider web spend between two counter states of one generation.
/// Providers absent from `prev` were born inside the window, so their whole
/// content is the delta; a FALLEN counter folds nothing, same rule as
/// [`series_delta`] - a counter only falls when the two states are from
/// different processes, and folding that pair would invent a huge charge.
pub fn web_deltas(prev: &Snapshot, cur: &Snapshot) -> Vec<(String, WebSpend)> {
    let mut out = Vec::new();
    for (provider, c) in &cur.web {
        let p = prev.web.get(provider).copied().unwrap_or_default();
        if c.requests < p.requests {
            continue;
        }
        let d = WebSpend {
            requests: c.requests - p.requests,
            credits: c.credits.saturating_sub(p.credits),
            microdollars: c.microdollars.saturating_sub(p.microdollars),
        };
        if !d.is_zero() {
            out.push((provider.clone(), d));
        }
    }
    out
}

/// Per-series deltas between two counter states of one generation. A series
/// absent from `prev` was born inside the window, so its whole content is
/// the delta; unchanged and fallen series fold nothing (`series_delta`'s
/// rules apply verbatim).
pub fn deltas_between(prev: &Snapshot, cur: &Snapshot) -> Vec<(SeriesKey, BucketDelta)> {
    let zero = SeriesCounters::default();
    let mut out = Vec::new();
    for (k, c) in &cur.series {
        if let Some(d) = series_delta(prev.series.get(k).unwrap_or(&zero), c) {
            out.push((k.clone(), d));
        }
    }
    out
}

/// Engine-scoped delta between two counter states: spec counters plus the
/// KV high-water gauge. None when nothing moved - the same gate the live
/// path's `engine_active` check applies.
pub fn engine_delta(prev: &Snapshot, cur: &Snapshot) -> Option<BucketDelta> {
    let d = BucketDelta {
        spec_drafted: cur.spec_drafted.saturating_sub(prev.spec_drafted) as i64,
        spec_accepted: cur.spec_accepted.saturating_sub(prev.spec_accepted) as i64,
        kv_pages_max: cur.kv_pages_used as i64,
        ..Default::default()
    };
    if d.is_zero() { None } else { Some(d) }
}

/// What a window moved in (requests, input, output) - the gap row's "lost"
/// triple, computed the same way for every baseline kind.
pub fn lost_between(prev: &Snapshot, cur: &Snapshot) -> (i64, i64, i64) {
    let mut lost = (0i64, 0i64, 0i64);
    for (k, c) in &cur.series {
        let p = prev.series.get(k);
        lost.0 += c.requests.saturating_sub(p.map_or(0, |b| b.requests)) as i64;
        lost.1 += c
            .input_tokens
            .saturating_sub(p.map_or(0, |b| b.input_tokens)) as i64;
        lost.2 += c
            .output_tokens
            .saturating_sub(p.map_or(0, |b| b.output_tokens)) as i64;
    }
    lost
}

/// The manager-restart baseline: persisted totals promoted to a `Snapshot`
/// so the recovery head can fold. Totals track only requests + the three
/// token counters (plus engine spec), so every other column starts from the
/// anchor's value - deltaing to exactly zero, never a fabricated number.
/// A series the totals never saw was born inside the blind window: its
/// baseline is fully zero so its whole content folds. The `anchor` is the
/// first snapshot the head will fold TO.
pub fn totals_pseudo_baseline(rows: &[TotalRow], anchor: &Snapshot) -> Snapshot {
    let mut base = anchor.clone();
    base.spec_drafted = 0;
    base.spec_accepted = 0;
    for (k, s) in base.series.iter_mut() {
        match rows.iter().find(|r| &r.key == k) {
            Some(row) => {
                s.requests = row.requests;
                s.input_tokens = row.input_tokens;
                s.output_tokens = row.output_tokens;
                s.cached_tokens = row.cached_tokens;
            }
            None => *s = SeriesCounters::default(),
        }
    }
    // The 'engine' pseudo-series is manager-synthesized - it exists in the
    // totals but never in a wire snapshot's series map.
    for row in rows {
        if row.key.operation == "engine" {
            base.spec_drafted = row.spec_drafted;
            base.spec_accepted = row.spec_accepted;
        }
    }
    base
}

/// One recovered fold: everything that lands in the bucket at `ts` - the
/// deltas of the interval that ENDED at snapshot `idx` (same convention as
/// the live path, which folds at scrape time).
#[derive(Debug)]
pub struct FoldStep {
    /// Index into the caller's kept-snapshot slice; the step's totals move
    /// to that snapshot's absolutes.
    pub idx: usize,
    pub ts: i64,
    pub series: Vec<(SeriesKey, BucketDelta)>,
    pub engine: Option<BucketDelta>,
    /// Web-search spend that moved in this interval, per provider. Recovered
    /// on the same terms as everything else: money the box spent while nobody
    /// was watching is exactly the number a user must not have to guess at.
    pub web: Vec<(String, WebSpend)>,
}

#[derive(Debug)]
pub struct RecoveryPlan {
    /// The stretch of the blind window the ring did not reach back to, with
    /// its exact lost totals: (from, to, (requests, input, output)). Honest
    /// fallback for the head only - everything after it folds for real.
    pub head_gap: Option<(i64, i64, (i64, i64, i64))>,
    pub folds: Vec<FoldStep>,
}

/// Turn a blind window plus the runner's snapshots into bucket folds.
/// `kept` must be the snapshots inside `(from, now]`, ascending by time.
/// When the oldest kept snapshot sits close enough to `from`, the head
/// interval folds off `baseline`; otherwise the head becomes a gap row and
/// folding starts at the first snapshot pair. Empty steps are dropped (the
/// sparse-tier rule: never write zeros).
pub fn plan_recovery(baseline: &Snapshot, from: i64, kept: &[(i64, Snapshot)]) -> RecoveryPlan {
    let mut plan = RecoveryPlan {
        head_gap: None,
        folds: Vec::new(),
    };
    let Some(first) = kept.first() else {
        return plan;
    };
    let contiguous = first.0 - from <= SNAPSHOT_HEAD_SLACK_MS;
    let mut prev = baseline;
    let mut start = 0;
    if !contiguous {
        let lost = lost_between(baseline, &first.1);
        if lost != (0, 0, 0) {
            plan.head_gap = Some((from, first.0, lost));
        }
        prev = &first.1;
        start = 1;
    }
    for (i, (ts, cur)) in kept.iter().enumerate().skip(start) {
        let series = deltas_between(prev, cur);
        let engine = engine_delta(prev, cur);
        let web = web_deltas(prev, cur);
        if !series.is_empty() || engine.is_some() || !web.is_empty() {
            plan.folds.push(FoldStep {
                idx: i,
                ts: *ts,
                series,
                engine,
                web,
            });
        }
        prev = cur;
    }
    plan
}

/// One series' write inside an atomic usage step - a live scrape and a
/// recovered interval use the same shape. `delta` is what folds into the
/// bucket tier (None = totals-only advance: the series exists but nothing
/// moved, and the sparse tier never writes zeros); the absolutes are what
/// the totals row moves to. A step's items commit in one transaction with
/// the full-state baseline, so the persisted state can never disagree with
/// the folds it follows - that is what makes a crash at any point
/// re-runnable without double-counting.
#[derive(Debug)]
pub struct UsageFoldItem {
    pub series_id: i64,
    pub delta: Option<BucketDelta>,
    pub requests: u64,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cached_tokens: u64,
    pub spec_drafted: u64,
    pub spec_accepted: u64,
}

/// One provider's write inside an atomic usage step, mirroring
/// [`UsageFoldItem`]: `delta` folds into the bucket tier (None = the totals
/// advance alone, which is what keeps LIFETIME spend exact across a blind
/// window even when its shape is unknowable), `absolute` is where the totals
/// row moves to.
#[derive(Debug, Clone)]
pub struct WebFoldItem {
    pub provider: String,
    pub delta: Option<WebSpend>,
    pub absolute: WebSpend,
}

/// prev -> cur for one series. Returns `None` when nothing moved (sparse tier:
/// never write zeros) or when a counter FELL - within one generation counters
/// are monotonic, so a fall means the snapshots are not from the same process
/// and folding the pair would fabricate a huge delta.
pub fn series_delta(prev: &SeriesCounters, cur: &SeriesCounters) -> Option<BucketDelta> {
    if cur.requests < prev.requests || cur.input_tokens < prev.input_tokens {
        return None;
    }
    let mut d = BucketDelta {
        requests: (cur.requests - prev.requests) as i64,
        disconnects: cur.disconnects.saturating_sub(prev.disconnects) as i64,
        errors_4xx: cur.errors_4xx.saturating_sub(prev.errors_4xx) as i64,
        errors_5xx: cur.errors_5xx.saturating_sub(prev.errors_5xx) as i64,
        input_tokens: (cur.input_tokens - prev.input_tokens) as i64,
        output_tokens: cur.output_tokens.saturating_sub(prev.output_tokens) as i64,
        cached_tokens: cur.cached_tokens.saturating_sub(prev.cached_tokens) as i64,
        duration_ms_sum: ((cur.duration_seconds_sum - prev.duration_seconds_sum) * 1000.0).round()
            as i64,
        ..Default::default()
    };
    // Cumulative-bucket deltas, then de-cumulate: h[i] holds only the
    // observations that landed in bucket i itself. The runner emits every
    // `le` line, so the scraped arrays are complete and non-decreasing; the
    // clamp only guards a torn parse from writing negatives into SQL.
    for (out, (p, c)) in [
        (&mut d.e2e, (&prev.e2e, &cur.e2e)),
        (&mut d.ttft, (&prev.ttft, &cur.ttft)),
    ] {
        let mut below = 0i64;
        for i in 0..14 {
            let cum = c[i].saturating_sub(p[i]) as i64;
            out[i] = (cum - below).max(0);
            below = cum;
        }
    }
    if d.is_zero() { None } else { Some(d) }
}

/// Parse the classic text exposition. Families we do not know are ignored -
/// this is a consumer of a stable contract, not a validator of it.
pub fn parse(text: &str) -> Snapshot {
    let mut snap = Snapshot::default();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (name, labels, value) = match split_sample(line) {
            Some(t) => t,
            None => continue,
        };
        let count = value.round() as u64; // counters print as integral floats
        match name {
            "process_start_time_seconds" => snap.start_unix = count,
            "paddock_spec_decode_draft_tokens_total" => snap.spec_drafted = count,
            "paddock_spec_decode_accepted_tokens_total" => snap.spec_accepted = count,
            "paddock_kv_pages_used" => snap.kv_pages_used = count,
            "gen_ai_server_request_duration_seconds_bucket" => {
                if let (Some(k), Some(i)) = (series_key(&labels), le_index(&labels)) {
                    snap.series.entry(k).or_default().e2e[i] += count;
                }
            }
            // The duration count/sum are summed across error_type splits into
            // one per-(op, origin, model) series - the bucket tier does not
            // keep the error dimension (the failure counters carry it).
            "gen_ai_server_request_duration_seconds_count" => {
                if let Some(k) = series_key(&labels) {
                    snap.series.entry(k).or_default().requests += count;
                }
            }
            "gen_ai_server_request_duration_seconds_sum" => {
                if let Some(k) = series_key(&labels) {
                    snap.series.entry(k).or_default().duration_seconds_sum += value;
                }
            }
            "gen_ai_server_time_to_first_token_seconds_bucket" => {
                if let (Some(k), Some(i)) = (series_key(&labels), le_index(&labels)) {
                    snap.series.entry(k).or_default().ttft[i] += count;
                }
            }
            "paddock_prompt_tokens_total" => {
                if let Some(k) = series_key(&labels) {
                    snap.series.entry(k).or_default().input_tokens += count;
                }
            }
            "paddock_prompt_cached_tokens_total" => {
                if let Some(k) = series_key(&labels) {
                    snap.series.entry(k).or_default().cached_tokens += count;
                }
            }
            "paddock_generation_tokens_total" => {
                if let Some(k) = series_key(&labels) {
                    snap.series.entry(k).or_default().output_tokens += count;
                }
            }
            // Web-search spend: whole-runner counters whose
            // one label is the provider. A search has no model or route of
            // its own, so these never touch `series`. Family names mirror
            // paddock-runner's `render_web_spend` - same arrangement as the
            // bucket LADDER above, and the same rule: change one side and the
            // spend panel goes quietly blank, so change both.
            "paddock_web_search_requests_total"
            | "paddock_web_search_credits_total"
            | "paddock_web_search_cost_microdollars_total" => {
                if let Some(p) = labels.get("provider") {
                    let e = snap.web.entry(p.clone()).or_default();
                    match name {
                        "paddock_web_search_requests_total" => e.requests = count,
                        "paddock_web_search_credits_total" => e.credits = count,
                        _ => e.microdollars = count,
                    }
                }
            }
            "paddock_request_failure_total" => {
                if let Some(k) = series_key(&labels) {
                    let e = snap.series.entry(k).or_default();
                    match labels.get("class").map(String::as_str) {
                        Some("4xx") => e.errors_4xx += count,
                        Some("5xx") => e.errors_5xx += count,
                        Some("disconnect") => e.disconnects += count,
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }
    snap
}

fn series_key(labels: &HashMap<String, String>) -> Option<SeriesKey> {
    Some(SeriesKey {
        operation: labels.get("gen_ai_operation_name")?.clone(),
        origin: labels
            .get("origin")
            .cloned()
            .unwrap_or_else(|| "live".into()),
        model: labels
            .get("gen_ai_request_model")
            .cloned()
            .unwrap_or_default(),
    })
}

fn le_index(labels: &HashMap<String, String>) -> Option<usize> {
    let le = labels.get("le")?;
    if le == "+Inf" {
        return Some(14);
    }
    LADDER.iter().position(|b| b == le)
}

/// One sample line -> (name, labels, value). Handles the classic escapes in
/// label values (`\\`, `\"`, `\n`); returns None on anything malformed rather
/// than guessing.
fn split_sample(line: &str) -> Option<(&str, HashMap<String, String>, f64)> {
    let (name_part, rest) = match line.find('{') {
        Some(brace) => {
            let close = find_closing_brace(line, brace)?;
            (
                &line[..brace],
                Some((&line[brace + 1..close], &line[close + 1..])),
            )
        }
        None => {
            let sp = line.find(' ')?;
            (&line[..sp], None)
        }
    };
    let mut labels = HashMap::new();
    let value_str = match rest {
        Some((label_str, tail)) => {
            parse_labels(label_str, &mut labels)?;
            tail.trim()
        }
        None => line[name_part.len()..].trim(),
    };
    // A value may carry a timestamp after it; the first token is the value.
    let value: f64 = value_str.split_whitespace().next()?.parse().ok()?;
    if value.is_nan() {
        return None;
    }
    Some((name_part, labels, value))
}

/// The label block may contain `}` inside quoted values, so scan with quote
/// state instead of a naive rfind.
fn find_closing_brace(line: &str, open: usize) -> Option<usize> {
    let bytes = line.as_bytes();
    let mut in_quotes = false;
    let mut escaped = false;
    for (i, &b) in bytes.iter().enumerate().skip(open + 1) {
        if escaped {
            escaped = false;
        } else if in_quotes && b == b'\\' {
            escaped = true;
        } else if b == b'"' {
            in_quotes = !in_quotes;
        } else if !in_quotes && b == b'}' {
            return Some(i);
        }
    }
    None
}

fn parse_labels(s: &str, out: &mut HashMap<String, String>) -> Option<()> {
    let mut rest = s.trim();
    while !rest.is_empty() {
        let eq = rest.find('=')?;
        let key = rest[..eq].trim().to_owned();
        let after = rest[eq + 1..].trim_start();
        let mut chars = after.char_indices();
        if chars.next()?.1 != '"' {
            return None;
        }
        let mut val = String::new();
        let mut end = None;
        let mut escaped = false;
        for (i, c) in chars {
            if escaped {
                val.push(match c {
                    'n' => '\n',
                    other => other, // covers \" and \\
                });
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                end = Some(i);
                break;
            } else {
                val.push(c);
            }
        }
        let end = end?;
        out.insert(key, val);
        rest = after[end + 1..]
            .trim_start()
            .trim_start_matches(',')
            .trim_start();
    }
    Some(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"
# HELP gen_ai_server_request_duration_seconds End-to-end request duration
# TYPE gen_ai_server_request_duration_seconds histogram
gen_ai_server_request_duration_seconds_bucket{gen_ai_operation_name="chat",gen_ai_provider_name="paddock",gen_ai_request_model="m1",origin="live",le="0.01"} 1
gen_ai_server_request_duration_seconds_bucket{gen_ai_operation_name="chat",gen_ai_provider_name="paddock",gen_ai_request_model="m1",origin="live",le="0.02"} 3
gen_ai_server_request_duration_seconds_bucket{gen_ai_operation_name="chat",gen_ai_provider_name="paddock",gen_ai_request_model="m1",origin="live",le="+Inf"} 4
gen_ai_server_request_duration_seconds_sum{gen_ai_operation_name="chat",gen_ai_provider_name="paddock",gen_ai_request_model="m1",origin="live"} 0.75
gen_ai_server_request_duration_seconds_count{gen_ai_operation_name="chat",gen_ai_provider_name="paddock",gen_ai_request_model="m1",origin="live"} 4
gen_ai_server_request_duration_seconds_bucket{gen_ai_operation_name="chat",gen_ai_provider_name="paddock",gen_ai_request_model="m1",origin="live",error_type="503",le="+Inf"} 2
gen_ai_server_request_duration_seconds_count{gen_ai_operation_name="chat",gen_ai_provider_name="paddock",gen_ai_request_model="m1",origin="live",error_type="503"} 2
gen_ai_server_time_to_first_token_seconds_bucket{gen_ai_operation_name="chat",gen_ai_provider_name="paddock",gen_ai_request_model="m1",origin="live",le="0.16"} 4
gen_ai_server_time_to_first_token_seconds_bucket{gen_ai_operation_name="chat",gen_ai_provider_name="paddock",gen_ai_request_model="m1",origin="live",le="+Inf"} 4
paddock_prompt_tokens_total{gen_ai_operation_name="chat",gen_ai_provider_name="paddock",gen_ai_request_model="m1",origin="live"} 1200
paddock_generation_tokens_total{gen_ai_operation_name="chat",gen_ai_provider_name="paddock",gen_ai_request_model="m1",origin="live"} 340
paddock_prompt_cached_tokens_total{gen_ai_operation_name="chat",gen_ai_provider_name="paddock",gen_ai_request_model="m1",origin="live"} 800
paddock_request_failure_total{gen_ai_operation_name="chat",gen_ai_provider_name="paddock",gen_ai_request_model="m1",origin="live",class="5xx"} 2
paddock_request_failure_total{gen_ai_operation_name="chat",gen_ai_provider_name="paddock",gen_ai_request_model="m1",origin="live",class="disconnect"} 1
paddock_spec_decode_draft_tokens_total 50
paddock_spec_decode_accepted_tokens_total 35
paddock_kv_pages_used 17
paddock_web_search_requests_total{provider="exa"} 12
paddock_web_search_cost_microdollars_total{provider="exa"} 84000
paddock_web_search_requests_total{provider="firecrawl"} 3
paddock_web_search_credits_total{provider="firecrawl"} 48
paddock_web_search_requests_total{provider="brave"} 5
process_start_time_seconds 1755000000
"#;

    #[test]
    fn parses_series_globals_and_merges_error_split() {
        let s = parse(SAMPLE);
        assert_eq!(s.start_unix, 1_755_000_000);
        assert_eq!(
            (s.spec_drafted, s.spec_accepted, s.kv_pages_used),
            (50, 35, 17)
        );
        assert_eq!(s.series.len(), 1);
        let k = SeriesKey {
            operation: "chat".into(),
            origin: "live".into(),
            model: "m1".into(),
        };
        let c = &s.series[&k];
        // error_type="503" split merged into the one series
        assert_eq!(c.requests, 6);
        assert_eq!(c.e2e[14], 6);
        assert_eq!(c.e2e[0], 1);
        assert_eq!(c.e2e[1], 3);
        assert_eq!(c.ttft[4], 4);
        assert_eq!(
            (c.input_tokens, c.output_tokens, c.cached_tokens),
            (1200, 340, 800)
        );
        assert_eq!((c.errors_5xx, c.disconnects, c.errors_4xx), (2, 1, 0));
        assert!((c.duration_seconds_sum - 0.75).abs() < 1e-9);
    }

    /// The three spend families land on the PROVIDER, never on a series, and
    /// a provider that priced nothing still counts its searches - that row is
    /// the difference between "we cannot tell you what it cost" and "nothing
    /// happened".
    #[test]
    fn web_spend_parses_per_provider_and_never_becomes_a_series() {
        let s = parse(SAMPLE);
        assert_eq!(s.series.len(), 1, "spend must not mint a series");
        assert_eq!(s.web.len(), 3);
        assert_eq!(
            s.web["exa"],
            WebSpend {
                requests: 12,
                credits: 0,
                microdollars: 84_000
            }
        );
        assert_eq!(
            s.web["firecrawl"],
            WebSpend {
                requests: 3,
                credits: 48,
                microdollars: 0
            }
        );
        assert_eq!(
            s.web["brave"],
            WebSpend {
                requests: 5,
                credits: 0,
                microdollars: 0
            }
        );
    }

    /// The wire keys the Studio's `WebSlot` interface reads by name. Nothing
    /// on the TypeScript side can fail a Rust rename, so the field set is
    /// pinned here - a dropped key would render as `undefined` and quietly
    /// draw a zero.
    #[test]
    fn web_slot_serializes_the_keys_the_studio_reads() {
        let v = serde_json::to_value(WebSlot {
            t: 1_000,
            port: 11540,
            provider: "exa".into(),
            requests: 12,
            credits: 0,
            microdollars: 84_000,
        })
        .expect("serialize");
        let mut keys: Vec<&str> = v
            .as_object()
            .expect("object")
            .keys()
            .map(String::as_str)
            .collect();
        keys.sort_unstable();
        assert_eq!(
            keys,
            [
                "credits",
                "microdollars",
                "port",
                "provider",
                "requests",
                "t"
            ]
        );
        assert_eq!(v["microdollars"], 84_000);
        assert_eq!(v["provider"], "exa");
    }

    #[test]
    fn web_deltas_skip_the_flat_and_refuse_the_fallen() {
        let snap = |entries: &[(&str, WebSpend)]| Snapshot {
            web: entries.iter().map(|(p, w)| ((*p).to_owned(), *w)).collect(),
            ..Default::default()
        };
        let prev = snap(&[
            (
                "exa",
                WebSpend {
                    requests: 10,
                    credits: 0,
                    microdollars: 70_000,
                },
            ),
            (
                "tavily",
                WebSpend {
                    requests: 4,
                    credits: 4,
                    microdollars: 0,
                },
            ),
        ]);
        let cur = snap(&[
            (
                "exa",
                WebSpend {
                    requests: 12,
                    credits: 0,
                    microdollars: 84_000,
                },
            ),
            (
                "tavily",
                WebSpend {
                    requests: 4,
                    credits: 4,
                    microdollars: 0,
                },
            ), // flat
            (
                "brave",
                WebSpend {
                    requests: 5,
                    credits: 0,
                    microdollars: 0,
                },
            ), // born
        ]);
        let mut d = web_deltas(&prev, &cur);
        d.sort_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(d.len(), 2, "a flat provider folds nothing: {d:?}");
        assert_eq!(d[0].0, "brave");
        assert_eq!(
            d[0].1.requests, 5,
            "a provider born in the window folds in full"
        );
        assert_eq!(d[1].0, "exa");
        assert_eq!(
            d[1].1,
            WebSpend {
                requests: 2,
                credits: 0,
                microdollars: 14_000
            }
        );

        // A fallen counter is a different process - never a negative charge,
        // and never the huge positive one a saturating subtraction would make.
        assert!(web_deltas(&cur, &prev).iter().all(|(p, _)| p != "exa"));
    }

    /// Round-tripping through the wire shape must preserve spend exactly: that
    /// serialization is the persisted attach baseline, so anything it drops
    /// comes back as a fabricated spike on the next scrape.
    #[test]
    fn web_spend_survives_the_wire_round_trip() {
        let mut s = Snapshot::default();
        s.web.insert(
            "exa".into(),
            WebSpend {
                requests: 12,
                credits: 0,
                microdollars: 84_000,
            },
        );
        s.web.insert(
            "firecrawl".into(),
            WebSpend {
                requests: 3,
                credits: 48,
                microdollars: 0,
            },
        );
        let back = snapshot_from_wire(&snapshot_to_wire(&s, 1_000));
        assert_eq!(back.web, s.web);
        assert!(
            web_deltas(&back, &s).is_empty(),
            "a round trip must move nothing"
        );
    }

    #[test]
    fn label_escapes_and_braces_inside_values_survive() {
        let s = parse(
            "paddock_prompt_tokens_total{gen_ai_operation_name=\"chat\",origin=\"live\",gen_ai_request_model=\"a\\\"b}\\\\c\\nd\"} 7\n",
        );
        let key = s.series.keys().next().expect("one series");
        assert_eq!(key.model, "a\"b}\\c\nd");
        assert_eq!(s.series[key].input_tokens, 7);
    }

    fn chat_key() -> SeriesKey {
        SeriesKey {
            operation: "chat".into(),
            origin: "live".into(),
            model: "m1".into(),
        }
    }

    fn counters(requests: u64, input: u64, output: u64) -> SeriesCounters {
        let mut c = SeriesCounters {
            requests,
            input_tokens: input,
            output_tokens: output,
            ..Default::default()
        };
        c.e2e[14] = requests; // +Inf == count keeps the arrays plausible
        c
    }

    fn snap_of(entries: &[(SeriesKey, SeriesCounters)], spec: (u64, u64), kv: u64) -> Snapshot {
        Snapshot {
            start_unix: 0,
            series: entries.iter().cloned().collect(),
            spec_drafted: spec.0,
            spec_accepted: spec.1,
            kv_pages_used: kv,
            web: HashMap::new(),
        }
    }

    #[test]
    fn snapshot_from_wire_maps_every_column() {
        let wire = paddock_admin::types::MetricsSnapshot {
            seq: 7,
            ts_ms: 123_000,
            series: vec![paddock_admin::types::SnapshotSeries {
                operation: "chat".into(),
                origin: "live".into(),
                model: "m1".into(),
                requests: 6,
                errors_4xx: 1,
                errors_5xx: 2,
                disconnects: 1,
                input_tokens: 1200,
                output_tokens: 340,
                cached_tokens: 800,
                duration_seconds_sum: 0.75,
                e2e: [1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 6],
                ttft: [0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            }],
            spec_drafted: 50,
            spec_accepted: 35,
            kv_pages_used: 17,
            web: vec![paddock_admin::types::WebSpendSeries {
                provider: "exa".into(),
                requests: 12,
                credits: 0,
                microdollars: 84_000,
            }],
        };
        let s = snapshot_from_wire(&wire);
        assert_eq!(
            s.web["exa"],
            WebSpend {
                requests: 12,
                credits: 0,
                microdollars: 84_000
            }
        );
        assert_eq!(
            (s.spec_drafted, s.spec_accepted, s.kv_pages_used),
            (50, 35, 17)
        );
        let c = &s.series[&chat_key()];
        assert_eq!(c.requests, 6);
        assert_eq!((c.errors_4xx, c.errors_5xx, c.disconnects), (1, 2, 1));
        assert_eq!(
            (c.input_tokens, c.output_tokens, c.cached_tokens),
            (1200, 340, 800)
        );
        assert!((c.duration_seconds_sum - 0.75).abs() < 1e-9);
        assert_eq!(c.e2e[0], 1);
        assert_eq!(c.e2e[14], 6);
        assert_eq!(c.ttft[1], 2);
    }

    /// Contiguous ring: the head folds off the baseline, unchanged intervals
    /// drop, and the engine pseudo-series moves only where spec moved.
    #[test]
    fn plan_recovery_folds_head_and_pairs() {
        let k = chat_key();
        let baseline = snap_of(&[(k.clone(), counters(10, 1000, 100))], (5, 4), 0);
        let kept = vec![
            (
                1_000i64,
                snap_of(&[(k.clone(), counters(12, 1200, 120))], (5, 4), 0),
            ),
            (
                61_000,
                snap_of(&[(k.clone(), counters(12, 1200, 120))], (5, 4), 0),
            ),
            (
                121_000,
                snap_of(&[(k.clone(), counters(15, 1500, 150))], (9, 7), 3),
            ),
        ];
        let plan = plan_recovery(&baseline, 0, &kept);
        assert!(
            plan.head_gap.is_none(),
            "kept[0] sits within the head slack"
        );
        assert_eq!(
            plan.folds.len(),
            2,
            "the flat middle interval folds nothing"
        );

        let head = &plan.folds[0];
        assert_eq!((head.idx, head.ts), (0, 1_000));
        assert_eq!(head.series.len(), 1);
        assert_eq!(head.series[0].1.requests, 2);
        assert_eq!(head.series[0].1.input_tokens, 200);
        assert!(head.engine.is_none(), "spec flat, kv zero: no engine fold");

        let tail = &plan.folds[1];
        assert_eq!((tail.idx, tail.ts), (2, 121_000));
        assert_eq!(tail.series[0].1.requests, 3);
        let engine = tail.engine.as_ref().expect("spec moved");
        assert_eq!(
            (
                engine.spec_drafted,
                engine.spec_accepted,
                engine.kv_pages_max
            ),
            (4, 3, 3)
        );
    }

    /// A ring that no longer reaches the window's start: the uncovered head
    /// becomes a gap with exact lost totals; folding starts at the first
    /// snapshot PAIR (never a baseline->far-snapshot lump).
    #[test]
    fn plan_recovery_head_gap_when_ring_falls_short() {
        let k = chat_key();
        let kept = vec![
            (
                200_000i64,
                snap_of(&[(k.clone(), counters(40, 4000, 400))], (0, 0), 0),
            ),
            (
                260_000,
                snap_of(&[(k.clone(), counters(45, 4500, 450))], (0, 0), 0),
            ),
        ];
        let plan = plan_recovery(&Snapshot::default(), 0, &kept);
        let (gfrom, gto, lost) = plan.head_gap.expect("200s past the window start");
        assert_eq!((gfrom, gto), (0, 200_000));
        assert_eq!(lost, (40, 4000, 400));
        assert_eq!(plan.folds.len(), 1);
        assert_eq!((plan.folds[0].idx, plan.folds[0].ts), (1, 260_000));
        assert_eq!(plan.folds[0].series[0].1.requests, 5);
    }

    /// The manager-restart baseline: persisted totals recover their four
    /// columns exactly; columns the totals never tracked delta to zero (never
    /// fabricated); a series born inside the window folds in full.
    #[test]
    fn totals_pseudo_baseline_recovers_tracked_columns_only() {
        let chat = chat_key();
        let born = SeriesKey {
            operation: "embeddings".into(),
            origin: "live".into(),
            model: "m1".into(),
        };
        let mut anchor_chat = counters(8, 800, 80);
        anchor_chat.errors_5xx = 2;
        anchor_chat.cached_tokens = 300;
        let anchor = snap_of(
            &[
                (chat.clone(), anchor_chat),
                (born.clone(), counters(4, 400, 40)),
            ],
            (7, 3),
            0,
        );
        let rows = vec![
            TotalRow {
                series_id: 1,
                key: chat.clone(),
                requests: 5,
                input_tokens: 500,
                output_tokens: 50,
                cached_tokens: 200,
                spec_drafted: 0,
                spec_accepted: 0,
                last_scrape_ms: 0,
            },
            TotalRow {
                series_id: 2,
                key: SeriesKey {
                    operation: "engine".into(),
                    origin: "live".into(),
                    model: "m1".into(),
                },
                requests: 0,
                input_tokens: 0,
                output_tokens: 0,
                cached_tokens: 0,
                spec_drafted: 6,
                spec_accepted: 2,
                last_scrape_ms: 0,
            },
        ];
        let base = totals_pseudo_baseline(&rows, &anchor);
        assert_eq!(
            (base.spec_drafted, base.spec_accepted),
            (6, 2),
            "engine row restores spec"
        );

        let mut deltas = deltas_between(&base, &anchor);
        deltas.sort_by(|a, b| a.0.operation.cmp(&b.0.operation));
        assert_eq!(deltas.len(), 2);
        let (bk, bd) = &deltas[0];
        assert_eq!(bk.operation, "chat");
        assert_eq!(
            (bd.requests, bd.input_tokens, bd.cached_tokens),
            (3, 300, 100)
        );
        assert_eq!(
            bd.errors_5xx, 0,
            "untracked columns must delta to zero, never fabricate"
        );
        let (ek, ed) = &deltas[1];
        assert_eq!(ek.operation, "embeddings");
        assert_eq!(ed.requests, 4, "a series born in the window folds in full");

        let engine = engine_delta(&base, &anchor).expect("spec moved 6->7");
        assert_eq!((engine.spec_drafted, engine.spec_accepted), (1, 1));

        assert_eq!(lost_between(&base, &anchor), (3 + 4, 300 + 400, 30 + 40));
    }

    #[test]
    fn delta_decumulates_and_refuses_a_fallen_counter() {
        // Cumulative arrays as the exposition really prints them: every `le`
        // carries forward, +Inf (index 14) is the count.
        let prev = SeriesCounters {
            requests: 4,
            input_tokens: 100,
            e2e: [1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4],
            ..Default::default()
        };
        let mut cur = prev.clone();
        cur.requests = 10;
        cur.input_tokens = 400;
        cur.duration_seconds_sum = 1.5;
        // +1 in bucket 0, +3 in bucket 1 alone, the rest of the ladder flat
        cur.e2e = [2, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 10];
        let d = series_delta(&prev, &cur).expect("moved");
        assert_eq!(d.requests, 6);
        assert_eq!(d.input_tokens, 300);
        assert_eq!(d.duration_ms_sum, 1500);
        assert_eq!(d.e2e[0], 1);
        assert_eq!(d.e2e[1], 3);
        assert_eq!(d.e2e[2], 0);

        // identical snapshots -> sparse skip
        assert!(series_delta(&cur, &cur).is_none());
        // a fallen counter is a different process, never a negative delta
        assert!(series_delta(&cur, &prev).is_none());
    }
}
