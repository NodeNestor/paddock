//! Per-tick host-segment attribution.
//!
//! A busy/idle ledger can show a cell is idle-bound - GPU utilization well
//! under the roof while the engine does less total GPU work than it should -
//! and the question is then where the serve loop spends its host time. This
//! module splits a decode tick's wall into:
//!
//!   fwd - the forward call (kernel launches + GPU wait + the readback sync)
//!   rb  - the [rows, vocab] logits dtoh inside fwd (counted separately,
//!         with bytes, by the family's read_batch_logits/head_row)
//!   smp - host sampling + commit (pick_next, ngram scan, channel pushes)
//!   gap - everything between one tick's host work ending and the next
//!         forward starting (scheduler, admissions, encode planning)
//!
//! PADDOCK_TICK_SEG=1 arms it; totals print via tracing every ~5 s and
//! reset. Diagnostics only - default-off, no serving behavior changes, and
//! the accumulators are one relaxed atomic add per segment when armed.
//! Attribution is observational within one run: read the steady-state
//! window's dumps, never subtract across runs.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

static FWD_NS: AtomicU64 = AtomicU64::new(0);
static FWD_N: AtomicU64 = AtomicU64::new(0);
static RB_NS: AtomicU64 = AtomicU64::new(0);
static RB_N: AtomicU64 = AtomicU64::new(0);
static RB_BYTES: AtomicU64 = AtomicU64::new(0);
static SMP_NS: AtomicU64 = AtomicU64::new(0);
static SMP_N: AtomicU64 = AtomicU64::new(0);
static GAP_NS: AtomicU64 = AtomicU64::new(0);
static GAP_N: AtomicU64 = AtomicU64::new(0);
// gap sub-segments (both run at the loop top, i.e. Inside gap): the
// encoder-budget advance's host span (its tower kernels queue async and
// drain inside the next tick's readback sync) and the mm admission wave
static ENC_NS: AtomicU64 = AtomicU64::new(0);
static ENC_N: AtomicU64 = AtomicU64::new(0);
static ADM_NS: AtomicU64 = AtomicU64::new(0);
static ADM_N: AtomicU64 = AtomicU64::new(0);

pub fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_TICK_SEG").is_some())
}

pub fn fwd(d: Duration) {
    FWD_NS.fetch_add(d.as_nanos() as u64, Relaxed);
    FWD_N.fetch_add(1, Relaxed);
}

pub fn rb(d: Duration, bytes: usize) {
    RB_NS.fetch_add(d.as_nanos() as u64, Relaxed);
    RB_N.fetch_add(1, Relaxed);
    RB_BYTES.fetch_add(bytes as u64, Relaxed);
}

pub fn smp(d: Duration) {
    SMP_NS.fetch_add(d.as_nanos() as u64, Relaxed);
    SMP_N.fetch_add(1, Relaxed);
}

pub fn gap(d: Duration) {
    GAP_NS.fetch_add(d.as_nanos() as u64, Relaxed);
    GAP_N.fetch_add(1, Relaxed);
}

pub fn enc(d: Duration) {
    ENC_NS.fetch_add(d.as_nanos() as u64, Relaxed);
    ENC_N.fetch_add(1, Relaxed);
}

pub fn adm(d: Duration) {
    ADM_NS.fetch_add(d.as_nanos() as u64, Relaxed);
    ADM_N.fetch_add(1, Relaxed);
}

/// Print-and-reset, self-throttled to ~5 s. Call from the serve loop top;
/// cheap no-op when unarmed.
pub fn maybe_dump() {
    if !on() {
        return;
    }
    static LAST: OnceLock<Mutex<Instant>> = OnceLock::new();
    let last = LAST.get_or_init(|| Mutex::new(Instant::now()));
    {
        let mut g = last.lock().expect("tickseg clock");
        if g.elapsed().as_secs() < 5 {
            return;
        }
        *g = Instant::now();
    }
    let (fn_, fns) = (FWD_N.swap(0, Relaxed), FWD_NS.swap(0, Relaxed));
    let (rn, rns, rb) = (
        RB_N.swap(0, Relaxed),
        RB_NS.swap(0, Relaxed),
        RB_BYTES.swap(0, Relaxed),
    );
    let (sn, sns) = (SMP_N.swap(0, Relaxed), SMP_NS.swap(0, Relaxed));
    let (gn, gns) = (GAP_N.swap(0, Relaxed), GAP_NS.swap(0, Relaxed));
    let (en, ens) = (ENC_N.swap(0, Relaxed), ENC_NS.swap(0, Relaxed));
    let (an, ans) = (ADM_N.swap(0, Relaxed), ADM_NS.swap(0, Relaxed));
    if fn_ + rn + sn + gn == 0 {
        return;
    }
    let per = |ns: u64, n: u64| {
        if n == 0 {
            0.0
        } else {
            ns as f64 / n as f64 / 1e3
        }
    };
    tracing::info!(
        "tickseg: fwd {}x tot {:.3}s avg {:.0}us | rb {}x tot {:.3}s avg {:.0}us {:.1}MB \
         | smp {}x tot {:.3}s avg {:.0}us | gap {}x tot {:.3}s avg {:.0}us \
         | enc {}x tot {:.3}s avg {:.0}us | adm {}x tot {:.3}s avg {:.0}us",
        fn_,
        fns as f64 / 1e9,
        per(fns, fn_),
        rn,
        rns as f64 / 1e9,
        per(rns, rn),
        rb as f64 / 1e6,
        sn,
        sns as f64 / 1e9,
        per(sns, sn),
        gn,
        gns as f64 / 1e9,
        per(gns, gn),
        en,
        ens as f64 / 1e9,
        per(ens, en),
        an,
        ans as f64 / 1e9,
        per(ans, an),
    );
}
