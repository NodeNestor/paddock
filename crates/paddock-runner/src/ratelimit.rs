//! In-memory, per-client abuse limiting for exposed (public/demo) deployments.
//!
//! No external store: a single Paddock process holds the counters itself, which
//! is sufficient for one instance and resets on restart - acceptable for abuse
//! control (it is not billing). A future distributed deployment can back the
//! same [`Limits`] interface with a shared store.
//!
//! Every limit is opt-in (`None` = unlimited), so a default Paddock is entirely
//! unthrottled; only an intentionally-exposed instance sets these. Pair with the
//! per-request output-token clamp (see `AppState::max_output_ceiling`) so each
//! admitted request is also bounded in cost, not just in frequency.

use std::collections::HashMap;
use std::net::{IpAddr, SocketAddr};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use axum::http::HeaderMap;

const MINUTE: Duration = Duration::from_secs(60);
const DAY: Duration = Duration::from_secs(86_400);

/// Per-client request limits. All optional; `is_enabled` is false when none are
/// set, letting callers skip the work entirely.
#[derive(Debug, Clone, Default)]
pub struct Limits {
    /// Max generation requests per client per minute.
    pub per_minute: Option<u32>,
    /// Max generation requests per client per day.
    pub per_day: Option<u32>,
}

impl Limits {
    pub fn is_enabled(&self) -> bool {
        self.per_minute.is_some() || self.per_day.is_some()
    }
}

/// Which limit refused a request (maps to a `Retry-After` + message upstream).
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Reject {
    PerMinute,
    PerDay,
}

/// Fixed-window counters. A rolling window would need a ring buffer; fixed
/// windows are simpler and fine for abuse control - the worst case is ~2x the
/// limit across a boundary, which does not matter here.
#[derive(Debug)]
struct Client {
    minute_start: Instant,
    minute_count: u32,
    day_start: Instant,
    day_count: u32,
    last_seen: Instant,
}

pub struct RateLimiter {
    limits: Limits,
    /// `trusted_proxy` = derive the client from the `X-Real-IP` our reverse
    /// proxy sets (it overwrites any client value); otherwise use the socket
    /// peer. Never trust `X-Forwarded-For` (a client can forge/prepend it).
    trusted_proxy: bool,
    clients: Mutex<HashMap<IpAddr, Client>>,
}

impl RateLimiter {
    pub fn new(limits: Limits, trusted_proxy: bool) -> Self {
        Self {
            limits,
            trusted_proxy,
            clients: Mutex::new(HashMap::new()),
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.limits.is_enabled()
    }

    /// Resolve the client key. With `trusted_proxy`, prefer `X-Real-IP` (set by
    /// our proxy to the true peer); else fall back to the socket peer. Returns
    /// `None` only when neither is available (then the caller allows the
    /// request - failing open on an unkeyable request beats blocking real
    /// traffic on a proxy misconfig).
    pub fn client_key(&self, headers: &HeaderMap, peer: Option<SocketAddr>) -> Option<IpAddr> {
        if self.trusted_proxy
            && let Some(ip) = headers
                .get("x-real-ip")
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.trim().parse::<IpAddr>().ok())
        {
            return Some(ip);
        }
        peer.map(|p| p.ip())
    }

    /// Count one request against the client and report whether it is allowed.
    /// `now` is injected for testability. Prunes idle clients opportunistically.
    pub fn check(&self, ip: IpAddr, now: Instant) -> Result<(), Reject> {
        if !self.limits.is_enabled() {
            return Ok(());
        }
        // Loopback is exempt: the box's own health probes and ops tooling
        // must not burn the public budget (144 probe requests/day would trip
        // a 40/day cap and turn the health check into a restart loop). Real
        // clients behind the proxy key via X-Real-IP, never loopback; a
        // local proxy that fails to set the header degrades to fail-open,
        // which is this limiter's stated philosophy anyway.
        if ip.is_loopback() {
            return Ok(());
        }
        let mut map = self.clients.lock().expect("ratelimit mutex");

        // Opportunistic prune: keep the map bounded without a background task.
        if map.len() > 4096 {
            map.retain(|_, c| now.duration_since(c.last_seen) < DAY);
        }

        let c = map.entry(ip).or_insert_with(|| Client {
            minute_start: now,
            minute_count: 0,
            day_start: now,
            day_count: 0,
            last_seen: now,
        });
        c.last_seen = now;

        // Roll the windows first so the counts we test are current.
        if now.duration_since(c.minute_start) >= MINUTE {
            c.minute_start = now;
            c.minute_count = 0;
        }
        if now.duration_since(c.day_start) >= DAY {
            c.day_start = now;
            c.day_count = 0;
        }

        if let Some(limit) = self.limits.per_minute
            && c.minute_count >= limit
        {
            return Err(Reject::PerMinute);
        }
        if let Some(limit) = self.limits.per_day
            && c.day_count >= limit
        {
            return Err(Reject::PerDay);
        }

        c.minute_count += 1;
        c.day_count += 1;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ip() -> IpAddr {
        "203.0.113.5".parse().unwrap()
    }

    #[test]
    fn disabled_allows_everything() {
        let rl = RateLimiter::new(Limits::default(), true);
        let t = Instant::now();
        for _ in 0..1000 {
            assert!(rl.check(ip(), t).is_ok());
        }
    }

    #[test]
    fn per_minute_blocks_after_limit_then_resets() {
        let rl = RateLimiter::new(
            Limits {
                per_minute: Some(3),
                per_day: None,
            },
            true,
        );
        let t0 = Instant::now();
        assert!(rl.check(ip(), t0).is_ok());
        assert!(rl.check(ip(), t0).is_ok());
        assert!(rl.check(ip(), t0).is_ok());
        assert_eq!(rl.check(ip(), t0), Err(Reject::PerMinute));
        // A minute later the window rolls and it's allowed again.
        let t1 = t0 + Duration::from_secs(61);
        assert!(rl.check(ip(), t1).is_ok());
    }

    #[test]
    fn per_day_blocks_across_minute_windows() {
        let rl = RateLimiter::new(
            Limits {
                per_minute: Some(100),
                per_day: Some(5),
            },
            true,
        );
        let mut t = Instant::now();
        for _ in 0..5 {
            assert!(rl.check(ip(), t).is_ok());
            t += Duration::from_secs(61); // new minute each time, same day
        }
        assert_eq!(rl.check(ip(), t), Err(Reject::PerDay));
    }

    #[test]
    fn different_clients_have_independent_counters() {
        let rl = RateLimiter::new(
            Limits {
                per_minute: Some(1),
                per_day: None,
            },
            true,
        );
        let t = Instant::now();
        let a: IpAddr = "203.0.113.1".parse().unwrap();
        let b: IpAddr = "203.0.113.2".parse().unwrap();
        assert!(rl.check(a, t).is_ok());
        assert_eq!(rl.check(a, t), Err(Reject::PerMinute));
        assert!(rl.check(b, t).is_ok()); // b unaffected by a
    }

    #[test]
    fn client_key_prefers_x_real_ip_when_trusted() {
        let rl = RateLimiter::new(Limits::default(), true);
        let mut h = HeaderMap::new();
        h.insert("x-forwarded-for", "1.2.3.4".parse().unwrap()); // must be ignored
        h.insert("x-real-ip", "198.51.100.9".parse().unwrap());
        let peer: SocketAddr = "10.0.0.1:5000".parse().unwrap();
        assert_eq!(
            rl.client_key(&h, Some(peer)),
            Some("198.51.100.9".parse().unwrap())
        );
    }

    #[test]
    fn client_key_uses_peer_when_untrusted() {
        let rl = RateLimiter::new(Limits::default(), false);
        let mut h = HeaderMap::new();
        h.insert("x-real-ip", "198.51.100.9".parse().unwrap()); // ignored when untrusted
        let peer: SocketAddr = "10.0.0.1:5000".parse().unwrap();
        assert_eq!(
            rl.client_key(&h, Some(peer)),
            Some("10.0.0.1".parse().unwrap())
        );
    }
}
