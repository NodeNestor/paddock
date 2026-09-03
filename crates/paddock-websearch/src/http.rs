//! Shared plumbing for the provider clients: one HTTP client, one error
//! reader, and the post-processing every search gets no matter which engine
//! answered it.

use crate::{Hit, Provider, SearchError, SearchOpts, rate};
use serde_json::Value;
use std::sync::OnceLock;
use std::time::Duration;

/// Whole-request deadline for one provider call. This is a hard ceiling on how
/// long a model's turn can stall on a search, so providers that scrape pages
/// (Firecrawl) are given a shorter server-side budget of their own rather than
/// being allowed to run up to this one.
const TIMEOUT: Duration = Duration::from_secs(20);

pub(crate) fn client() -> &'static reqwest::Client {
    static CLIENT: OnceLock<reqwest::Client> = OnceLock::new();
    CLIENT.get_or_init(|| {
        reqwest::Client::builder()
            .timeout(TIMEOUT)
            .build()
            .expect("reqwest client")
    })
}

/// Where the wire tests point every provider. Set once, in-process, so the
/// suite never has to touch a process-global env var (which is `unsafe` on
/// edition 2024 and races across parallel tests anyway).
#[cfg(test)]
pub(crate) static TEST_URL: OnceLock<String> = OnceLock::new();

/// Test override: point the configured provider at a mock endpoint instead of
/// the real API. Only one provider is live at a time, so one variable is
/// enough.
///
/// The env spelling is compiled out of a shipped binary. It was a
/// development hook that happened to be live in production, and what it does
/// there is redirect a provider call - carrying the user's API key - to any
/// host whoever set the variable chose.
pub(crate) fn endpoint(real: &str) -> String {
    #[cfg(test)]
    if let Some(u) = TEST_URL.get() {
        return u.clone();
    }
    #[cfg(not(feature = "hardened"))]
    if let Ok(u) = std::env::var("PADDOCK_SEARCH_URL") {
        return u;
    }
    real.to_string()
}

/// Transport failure (DNS, TLS, timeout) named with the provider that was
/// being called, so "who is broken" is never a guess.
pub(crate) fn transport(p: Provider) -> impl Fn(reqwest::Error) -> SearchError {
    move |e| SearchError {
        status: None,
        msg: format!("{}: {e}", p.label()),
    }
}

/// Send one prepared request on this provider's terms.
///
/// Three things happen around the round trip that no provider module should
/// have to remember: the request is PACED against what this key can sustain,
/// the response's rate-limit headers are LEARNED from, and a 429 is waited out
/// once on the provider's own `Retry-After` rather than surrendered to. A rate
/// limit is a queue right up until the wait stops being affordable - and then
/// it is a failure that says which limit and when it clears.
pub(crate) async fn send(p: Provider, req: reqwest::RequestBuilder) -> Result<Value, SearchError> {
    // a JSON body or query string clones fine; if one ever didn't, the only
    // cost is that this request doesn't get its retry
    let again = req.try_clone();
    rate::acquire(p).await;
    let res = req.send().await.map_err(transport(p))?;
    rate::observe(p, res.headers());
    if res.status() != reqwest::StatusCode::TOO_MANY_REQUESTS {
        return read_json(p.label(), res).await;
    }

    let detail = rate::limit_detail(res.headers());
    let wait = rate::asked_wait(res.headers());
    match (again, wait) {
        // the provider named a wait we can afford: hold every other search on
        // this key back too, sit it out, then try once more
        (Some(again), Some(wait)) if wait <= rate::MAX_WAIT => {
            rate::back_off(p, wait);
            tokio::time::sleep(wait).await;
            rate::acquire(p).await;
            let res = again.send().await.map_err(transport(p))?;
            rate::observe(p, res.headers());
            if res.status() != reqwest::StatusCode::TOO_MANY_REQUESTS {
                return read_json(p.label(), res).await;
            }
            Err(rate_limited(p, &rate::limit_detail(res.headers())))
        }
        _ => {
            // too long to wait on, or nothing to wait on. Stalling an agent's
            // turn behind a spent monthly quota helps nobody; saying so does.
            if let Some(w) = wait {
                rate::back_off(p, w.min(rate::MAX_WAIT));
            }
            Err(rate_limited(p, &detail))
        }
    }
}

fn rate_limited(p: Provider, detail: &str) -> SearchError {
    SearchError {
        status: Some(429),
        msg: format!("{} is rate limiting this key{detail}", p.label()),
    }
}

/// Read a provider response as JSON, mapping HTTP errors to a short
/// user-facing message.
///
/// Every provider spells its error differently - Exa puts it in `message`,
/// Tavily in `detail.error`, Firecrawl in `error`, Brave in `error.detail`,
/// Perplexity in `error.message` - so all the known shapes are probed in this
/// one place. Anything unrecognized falls back to the raw body: the provider's
/// own words beat a message we invented.
pub(crate) async fn read_json(
    provider: &str,
    res: reqwest::Response,
) -> Result<Value, SearchError> {
    let status = res.status();
    let text = res.text().await.map_err(|e| SearchError {
        status: None,
        msg: format!("{provider}: {e}"),
    })?;
    let v: Value = serde_json::from_str(&text).unwrap_or(Value::Null);
    if !status.is_success() {
        const PATHS: [&str; 7] = [
            "/message",
            "/error/message",
            "/error/detail",
            "/detail/error",
            "/error",
            "/detail",
            "/details",
        ];
        let detail = PATHS
            .iter()
            .filter_map(|p| v.pointer(p).and_then(Value::as_str))
            .map(str::trim)
            .find(|s| !s.is_empty())
            .map(str::to_string)
            .unwrap_or_else(|| text.chars().take(200).collect());
        return Err(SearchError {
            status: Some(status.as_u16()),
            msg: format!("{provider} returned {}: {detail}", status.as_u16()),
        });
    }
    Ok(v)
}

/// Is this failure worth one retry with the optional localization dropped?
///
/// The geo hint is the fragile part of every request: Tavily's `country` is an
/// enum of English names whose spellings drift ("czech republic", not
/// "czechia"), and providers tighten validation over time. A rejected *hint*
/// must not cost the user the whole search, so a 4xx that isn't about
/// credentials (401/403) or rate limits (429) - retrying those changes
/// nothing - earns exactly one plain retry. 5xx is the provider being down,
/// not us being wrong, so it is not retried either.
pub(crate) fn retryable(e: &SearchError) -> bool {
    matches!(e.status, Some(s) if (400..500).contains(&s) && !matches!(s, 401 | 403 | 429))
}

/// The one place the "we dropped your location" decision is announced. It is a
/// warn, not a silent degrade: the search still answers, but it answered a
/// slightly different question than the caller asked for.
pub(crate) fn note_retry(provider: &str, e: &SearchError) {
    tracing::warn!(
        provider,
        error = %e.msg,
        "provider rejected the request; retrying once without the location hint"
    );
}

/// String field or "" - providers omit fields freely and a missing title is
/// not worth failing a whole search over.
pub(crate) fn s(v: &Value, key: &str) -> String {
    v.get(key).and_then(Value::as_str).unwrap_or("").to_string()
}

/// Join a provider's array-of-strings field (Exa highlights, Brave
/// extra_snippets) into one block of text.
pub(crate) fn joined(v: &Value, key: &str, sep: &str) -> String {
    v.get(key)
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(Value::as_str)
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .collect::<Vec<_>>()
                .join(sep)
        })
        .unwrap_or_default()
}

/// A domain as the caller wrote it, reduced to something comparable: no
/// scheme, no leading `www.`, no trailing slash, lower-case.
fn normalize(d: &str) -> String {
    let d = d.trim().to_ascii_lowercase();
    let d = d
        .strip_prefix("https://")
        .or_else(|| d.strip_prefix("http://"))
        .unwrap_or(&d);
    d.strip_prefix("www.")
        .unwrap_or(d)
        .trim_end_matches('/')
        .to_string()
}

/// Does this result's host (and path) satisfy one declared entry? Subdomains
/// count - both APIs define `example.com` as covering `docs.example.com` -
/// and an entry may carry a path (`nature.com/articles`), which Exa and
/// Perplexity both accept and we honour for every provider.
fn matches(host: &str, path: &str, decl: &str) -> bool {
    let (dom, sub) = match decl.split_once('/') {
        Some((d, p)) => (d, p),
        None => (decl, ""),
    };
    if !(host == dom || host.ends_with(&format!(".{dom}"))) {
        return false;
    }
    sub.is_empty() || path.trim_start_matches('/').starts_with(sub)
}

/// Enforce the caller's `allowed_domains` / `blocked_domains` on the results.
///
/// Providers apply their own domain filters for RECALL - they search the right
/// corner of the web, which no amount of post-filtering can do. This pass is
/// for CORRECTNESS: the lists come from the tool declaration and are a
/// contract, and provider support ranges from native parameters (Exa, Tavily,
/// Firecrawl) through query operators (Brave) to nothing at all. A result we
/// cannot even parse a host from is dropped when an allowlist is in force and
/// kept when it isn't - an unreadable URL is not evidence of belonging.
pub(crate) fn enforce_domains(hits: &mut Vec<Hit>, opts: &SearchOpts) {
    if !opts.has_domain_filter() {
        return;
    }
    hits.retain(|h| match reqwest::Url::parse(&h.url) {
        Ok(url) => url_allowed(&url, &opts.allowed_domains, &opts.blocked_domains),
        // an unreadable URL is not evidence of belonging
        Err(_) => opts.allowed_domains.is_empty(),
    });
}

/// Does one URL satisfy the caller's declared lists? Shared with `web_fetch`,
/// which has to answer the same question about a single URL before it spends
/// a request on it.
pub(crate) fn url_allowed(url: &reqwest::Url, allow: &[String], block: &[String]) -> bool {
    let Some(host) = url.host_str() else {
        return allow.is_empty();
    };
    let host = host.trim_start_matches("www.").to_ascii_lowercase();
    let path = url.path();
    let prep = |v: &[String]| -> Vec<String> {
        v.iter()
            .map(|d| normalize(d))
            .filter(|d| !d.is_empty())
            .collect()
    };
    let allow = prep(allow);
    let block = prep(block);
    let hit = |d: &String| matches(&host, path, d);
    if !allow.is_empty() && !allow.iter().any(hit) {
        return false;
    }
    !block.iter().any(hit)
}

/// Everything that must hold whichever provider answered: the domain contract,
/// one page-text budget, and no more results than were asked for.
pub(crate) fn finish(hits: &mut Vec<Hit>, opts: &SearchOpts) {
    enforce_domains(hits, opts);
    let cap = opts.depth.content_cap();
    for h in hits.iter_mut() {
        // trim on a char boundary; a provider may ignore the budget we asked
        // it for, and a mid-codepoint cut would corrupt the tool turn
        if h.content.chars().count() > cap {
            h.content = h.content.chars().take(cap).collect();
        }
        h.content = h.content.trim().to_string();
    }
    hits.truncate(opts.results());
}
