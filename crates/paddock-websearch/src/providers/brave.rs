//! Brave - the only provider here backed by an INDEPENDENT web index rather
//! than a reseller of someone else's. That is the reason to offer it: when a
//! second opinion matters, Brave is genuinely a second opinion.
//!
//! What we use: `count` (its ceiling is 20), `country`, `extra_snippets` at
//! medium and high depth - up to five extra excerpts per result, which is
//! Brave's answer to "page content" - and the `site:` / `-site:` query
//! operators.
//!
//! Domain filtering is the interesting part. Brave has no domain parameters,
//! only query operators, and while `-site:x` is safe to AND onto any query
//! there is no documented `OR`, so a multi-domain allowlist cannot be
//! expressed as an operator at all. Rather than ship an unverified `site:a OR
//! site:b` and quietly return junk, this asks for the operator only when it is
//! certain (a single allowlisted domain, and every blocked one), over-fetches
//! to Brave's maximum whenever a filter is in force, and lets `http::finish`
//! enforce the actual contract on the results. Fewer correct results beat more
//! wrong ones.
//!
//! Deliberately not sent: `search_lang` (a country is not a language, and
//! guessing one narrows results on a mistake), `freshness` and `safesearch`
//! (no caller signal - neither dialect's tool declaration has a recency or
//! safety knob), and the `X-Loc-*` location headers, which are not in the
//! documentation this was written against; unverified headers are how a
//! working search turns into a 422.

use crate::{
    Depth, Found, Hit, Provider, SearchConfig, SearchError, SearchOpts, SearchUsage, http,
};
use serde_json::Value;

const ME: Provider = Provider::Brave;
const URL: &str = "https://api.search.brave.com/res/v1/web/search";
/// Brave's documented ceiling for `count`.
const MAX_RESULTS: usize = 20;

pub(crate) async fn search(
    cfg: &SearchConfig,
    opts: &SearchOpts,
    query: &str,
) -> Result<Found, SearchError> {
    let geo = opts.location.code();
    let v = match call(cfg, opts, query, geo.as_deref()).await {
        Err(e) if geo.is_some() && http::retryable(&e) => {
            http::note_retry(ME.label(), &e);
            call(cfg, opts, query, None).await?
        }
        r => r?,
    };
    // Brave prices nothing in the response body - its budget lives entirely in
    // the rate-limit headers, which `rate` reads on the way past. An empty
    // usage is the honest answer, not a zero.
    Ok(Found {
        hits: v
            .pointer("/web/results")
            .and_then(Value::as_array)
            .map(|arr| arr.iter().map(hit).collect())
            .unwrap_or_default(),
        usage: SearchUsage::default(),
    })
}

/// The query with whatever domain operators can be expressed without guessing.
/// Everything else is left to the result-side contract in `http::finish`.
fn operators(opts: &SearchOpts, query: &str) -> String {
    let mut q = query.to_string();
    // one allowed domain is `site:x`; several would need an OR that Brave does
    // not document, so they are enforced on the results instead
    if let [only] = opts.allowed_domains.as_slice() {
        q.push_str(&format!(" site:{}", only.trim()));
    }
    for d in &opts.blocked_domains {
        // `-site:` ANDs cleanly, so every blocked domain can be expressed
        q.push_str(&format!(" -site:{}", d.trim()));
    }
    q
}

async fn call(
    cfg: &SearchConfig,
    opts: &SearchOpts,
    query: &str,
    country: Option<&str>,
) -> Result<Value, SearchError> {
    // With a filter in force some results will be dropped on the way out, so
    // ask for Brave's maximum and let the trim in http::finish do the cutting.
    let count = if opts.has_domain_filter() {
        MAX_RESULTS
    } else {
        opts.results_capped(MAX_RESULTS)
    };
    let mut params: Vec<(&str, String)> =
        vec![("q", operators(opts, query)), ("count", count.to_string())];
    if opts.depth != Depth::Low {
        // Brave returns no page text, so the extra excerpts are the content
        params.push(("extra_snippets", "true".into()));
    }
    if let Some(c) = country {
        params.push(("country", c.to_ascii_uppercase()));
    }
    http::send(
        ME,
        http::client()
            .get(http::endpoint(URL))
            .query(&params)
            .header("X-Subscription-Token", &cfg.api_key)
            .header("Accept", "application/json"),
    )
    .await
}

fn hit(r: &Value) -> Hit {
    // description is the blurb; extra_snippets are further passages from the
    // page. Together they are as close to page content as Brave offers.
    let mut content = http::s(r, "description");
    let extra = http::joined(r, "extra_snippets", "\n");
    if !extra.is_empty() {
        if !content.is_empty() {
            content.push('\n');
        }
        content.push_str(&extra);
    }
    Hit {
        title: http::s(r, "title"),
        url: http::s(r, "url"),
        content,
        // Brave's `page_age` is an ISO timestamp when the page declared one
        published: r
            .get("page_age")
            .and_then(Value::as_str)
            .map(|d| d.chars().take(10).collect()),
    }
}
