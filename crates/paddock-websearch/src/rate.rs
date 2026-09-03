//! Per-key request pacing, learned from the providers' own rate-limit headers.
//!
//! This matters more here than it would in a single-user tool. The tier-1
//! workload is N concurrent agent sessions on one box, and they all share the
//! one provider key configured on that endpoint. Eight sessions that each
//! decide to search will burst straight through a per-second limit, and
//! without pacing the only outcome is that some of them fail while nothing in
//! the logs explains why. Rate limiting is scheduling, which is the part of
//! this project that is supposed to be good.
//!
//! Nothing here is configured. Providers advertise their limits on every
//! response and we learn from those: Brave sends
//! `x-ratelimit-policy: 50;w=1, 0;w=2678400` - fifty a second, plus a monthly
//! window. Until a provider tells us something, requests are unpaced; the cost
//! of guessing a limit that isn't there is throttling a user for no reason.

use crate::Provider;
use reqwest::header::HeaderMap;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

/// The longest we will ever sit on a 429 before giving the caller an honest
/// failure. A per-second limit clears in about a second; a spent monthly quota
/// does not clear at all, and blocking an agent's turn on it would be a worse
/// answer than saying so.
pub(crate) const MAX_WAIT: Duration = Duration::from_secs(5);

#[derive(Clone, Copy)]
struct Slot {
    /// smallest gap between requests this key sustains, from the tightest
    /// window the provider advertises
    gap: Option<Duration>,
    /// earliest instant the next request may leave
    next: Option<Instant>,
}

fn table() -> &'static Mutex<HashMap<&'static str, Slot>> {
    static T: OnceLock<Mutex<HashMap<&'static str, Slot>>> = OnceLock::new();
    T.get_or_init(|| Mutex::new(HashMap::new()))
}

fn with<R>(f: impl FnOnce(&mut HashMap<&'static str, Slot>) -> R) -> R {
    let mut t = table().lock().unwrap_or_else(|e| e.into_inner());
    f(&mut t)
}

/// Wait until this key may take another request, and reserve that slot.
///
/// The reservation happens under the lock and the sleep happens outside it, so
/// concurrent searches queue up behind each other in order instead of all
/// waking at the same instant and re-colliding.
pub(crate) async fn acquire(p: Provider) {
    let wait = with(|t| {
        let slot = t.entry(p.as_str()).or_insert(Slot {
            gap: None,
            next: None,
        });
        let now = Instant::now();
        let at = slot.next.filter(|n| *n > now).unwrap_or(now);
        slot.next = Some(at + slot.gap.unwrap_or(Duration::ZERO));
        at.saturating_duration_since(now)
    });
    if !wait.is_zero() {
        tokio::time::sleep(wait).await;
    }
}

/// Push this key's next slot out by `wait` - used after a 429, so that every
/// other in-flight search backs off too rather than each discovering the wall
/// on its own.
pub(crate) fn back_off(p: Provider, wait: Duration) {
    with(|t| {
        let slot = t.entry(p.as_str()).or_insert(Slot {
            gap: None,
            next: None,
        });
        let at = Instant::now() + wait;
        if slot.next.is_none_or(|n| n < at) {
            slot.next = Some(at);
        }
    });
}

/// Learn this key's budget from a response.
pub(crate) fn observe(p: Provider, h: &HeaderMap) {
    let Some(gap) = tightest_gap(h) else { return };
    with(|t| {
        t.entry(p.as_str())
            .or_insert(Slot {
                gap: None,
                next: None,
            })
            .gap = Some(gap);
    });
}

/// A header carrying one entry per rate-limit window, comma separated.
fn entries<'a>(h: &'a HeaderMap, name: &str) -> Vec<&'a str> {
    h.get(name)
        .and_then(|v| v.to_str().ok())
        .map(|v| v.split(',').map(str::trim).collect())
        .unwrap_or_default()
}

/// `x-ratelimit-policy: 50;w=1, 0;w=2678400` -> the shortest window's minimum
/// gap between requests. The tightest window is the one a burst actually
/// trips; a monthly quota is not something pacing can help with.
fn tightest_gap(h: &HeaderMap) -> Option<Duration> {
    entries(h, "x-ratelimit-policy")
        .into_iter()
        .filter_map(|e| {
            let (limit, rest) = e.split_once(';')?;
            let limit: u64 = limit.trim().parse().ok()?;
            let window: u64 = rest.trim().strip_prefix("w=")?.trim().parse().ok()?;
            // a zero limit is "no budget declared for this window", not "no
            // requests allowed" - pacing off it would wedge the provider
            (limit > 0 && window > 0).then_some((window, limit))
        })
        .min_by_key(|(window, _)| *window)
        .map(|(window, limit)| Duration::from_secs(window) / limit as u32)
}

/// How long the provider wants us to wait, from `retry-after` (seconds form)
/// or failing that the soonest `x-ratelimit-reset`. `None` means it didn't say.
pub(crate) fn asked_wait(h: &HeaderMap) -> Option<Duration> {
    let retry = h
        .get(reqwest::header::RETRY_AFTER)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.trim().parse::<u64>().ok());
    // an HTTP-date Retry-After is legal but nobody here sends one; falling
    // through to the reset header is better than parsing dates for nothing
    let reset = entries(h, "x-ratelimit-reset")
        .into_iter()
        .filter_map(|e| e.parse::<u64>().ok())
        .filter(|s| *s > 0)
        .min();
    retry.or(reset).map(Duration::from_secs)
}

/// What the headers say about why we were refused, in words a user can act on.
/// Empty when the provider told us nothing.
pub(crate) fn limit_detail(h: &HeaderMap) -> String {
    let remaining = entries(h, "x-ratelimit-remaining");
    let limit = entries(h, "x-ratelimit-limit");
    let reset = entries(h, "x-ratelimit-reset");
    // the last window is the long one (Brave orders per-second then per-month),
    // and a spent long window is the difference between "slow down" and "this
    // key is done for the month"
    let spent_long = remaining.last().is_some_and(|r| r.trim() == "0")
        && reset
            .last()
            .and_then(|s| s.parse::<u64>().ok())
            .is_some_and(|s| s > 3_600);
    if spent_long {
        let secs = reset
            .last()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(0);
        return format!(
            " - this key's quota is spent, and resets in {}",
            human(secs)
        );
    }
    match (remaining.first(), limit.first()) {
        (Some(r), Some(l)) => format!(" - {r} of {l} requests left in this window"),
        _ => String::new(),
    }
}

fn human(secs: u64) -> String {
    match secs {
        s if s >= 86_400 => format!("{} day(s)", s / 86_400),
        s if s >= 3_600 => format!("{} hour(s)", s / 3_600),
        s if s >= 60 => format!("{} minute(s)", s / 60),
        s => format!("{s} second(s)"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn headers(pairs: &[(&str, &str)]) -> HeaderMap {
        let mut h = HeaderMap::new();
        for (k, v) in pairs {
            h.insert(
                reqwest::header::HeaderName::from_bytes(k.as_bytes()).expect("header name"),
                v.parse().expect("header value"),
            );
        }
        h
    }

    #[test]
    fn the_tightest_window_sets_the_pace() {
        // Brave's real header as measured: 50 a second plus a month
        let h = headers(&[("x-ratelimit-policy", "50;w=1, 0;w=2678400")]);
        assert_eq!(tightest_gap(&h), Some(Duration::from_millis(20)));

        // the monthly window must not become the pace even when it parses
        let h = headers(&[("x-ratelimit-policy", "20;w=1, 1000;w=2678400")]);
        assert_eq!(tightest_gap(&h), Some(Duration::from_millis(50)));

        // nothing advertised = no pacing invented
        assert_eq!(tightest_gap(&HeaderMap::new()), None);
        assert_eq!(
            tightest_gap(&headers(&[("x-ratelimit-policy", "nonsense")])),
            None
        );
        // a zero limit is "not declared", not "never allowed"
        assert_eq!(
            tightest_gap(&headers(&[("x-ratelimit-policy", "0;w=60")])),
            None
        );
    }

    #[test]
    fn the_wait_comes_from_the_provider_not_from_us() {
        assert_eq!(
            asked_wait(&headers(&[("retry-after", "3")])),
            Some(Duration::from_secs(3))
        );
        // no Retry-After: fall back to the soonest reset
        assert_eq!(
            asked_wait(&headers(&[("x-ratelimit-reset", "1, 1508427")])),
            Some(Duration::from_secs(1))
        );
        // Retry-After wins when both are present
        assert_eq!(
            asked_wait(&headers(&[
                ("retry-after", "2"),
                ("x-ratelimit-reset", "9")
            ])),
            Some(Duration::from_secs(2))
        );
        assert_eq!(asked_wait(&HeaderMap::new()), None);
    }

    #[test]
    fn a_spent_monthly_quota_reads_differently_from_a_busy_second() {
        let month = headers(&[
            ("x-ratelimit-limit", "50, 2000"),
            ("x-ratelimit-remaining", "49, 0"),
            ("x-ratelimit-reset", "1, 1508427"),
        ]);
        let d = limit_detail(&month);
        assert!(d.contains("quota is spent"), "{d}");
        assert!(d.contains("day(s)"), "{d}");

        let second = headers(&[
            ("x-ratelimit-limit", "50, 2000"),
            ("x-ratelimit-remaining", "0, 1999"),
            ("x-ratelimit-reset", "1, 1508427"),
        ]);
        let d = limit_detail(&second);
        assert!(d.contains("0 of 50"), "{d}");
        assert!(!d.contains("quota is spent"), "{d}");

        assert_eq!(limit_detail(&HeaderMap::new()), "");
    }

    #[tokio::test]
    async fn pacing_queues_requests_instead_of_letting_them_collide() {
        // a deliberately coarse gap so the timing assertion cannot flake
        let p = Provider::Brave;
        with(|t| {
            t.insert(
                p.as_str(),
                Slot {
                    gap: Some(Duration::from_millis(40)),
                    next: None,
                },
            );
        });
        let start = Instant::now();
        for _ in 0..3 {
            acquire(p).await;
        }
        // first goes immediately, the next two wait one gap each
        assert!(
            start.elapsed() >= Duration::from_millis(70),
            "{:?}",
            start.elapsed()
        );
        with(|t| t.remove(p.as_str()));
    }
}
