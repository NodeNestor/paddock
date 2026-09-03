//! Perplexity - the search index behind its answer engine, exposed raw. It
//! ranks for "what would answer this question", which is a different and often
//! better ordering than link popularity for the kind of query a model makes.
//!
//! What we use: `max_results`, `max_tokens_per_page` (its content dial, tied
//! to our depth budget so one knob governs both), `search_domain_filter` and
//! `country`.
//!
//! `search_domain_filter` is allowlist OR denylist, never both: bare entries
//! include, a `-` prefix excludes, and the cap is 20. When a caller declares
//! both lists the allowlist is sent - it is the stronger constraint - and the
//! blocklist is enforced on the results by `http::finish`.
//!
//! Deliberately not sent: `search_recency_filter` (no caller signal), and the
//! multi-query form of `query`, which would fan one tool call out into several
//! billed searches the model never asked for.

use crate::{Found, Hit, Provider, SearchConfig, SearchError, SearchOpts, SearchUsage, http};
use serde_json::{Value, json};

const ME: Provider = Provider::Perplexity;
const URL: &str = "https://api.perplexity.ai/search";
/// Documented ceilings: 20 results, 20 domain-filter entries.
const MAX_RESULTS: usize = 20;
const MAX_FILTER: usize = 20;

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
    // Perplexity reports no cost at all, only an id. Recording the id anyway
    // means a line in the ledger can still be matched against their dashboard.
    Ok(Found {
        hits: v
            .get("results")
            .and_then(Value::as_array)
            .map(|arr| arr.iter().map(hit).collect())
            .unwrap_or_default(),
        usage: SearchUsage {
            request_id: v.get("id").and_then(Value::as_str).map(str::to_string),
            ..Default::default()
        },
    })
}

/// Allowlist if there is one, otherwise the denylist with its `-` prefixes.
/// Mixing the two is not expressible, so the half that isn't sent is enforced
/// on the results instead.
fn domain_filter(opts: &SearchOpts) -> Option<Vec<String>> {
    let list: Vec<String> = if !opts.allowed_domains.is_empty() {
        opts.allowed_domains
            .iter()
            .map(|d| d.trim().to_string())
            .collect()
    } else {
        opts.blocked_domains
            .iter()
            .map(|d| format!("-{}", d.trim()))
            .collect()
    };
    let list: Vec<String> = list
        .into_iter()
        .filter(|d| d != "-")
        .take(MAX_FILTER)
        .collect();
    (!list.is_empty()).then_some(list)
}

async fn call(
    cfg: &SearchConfig,
    opts: &SearchOpts,
    query: &str,
    country: Option<&str>,
) -> Result<Value, SearchError> {
    let mut body = json!({
        "query": query,
        "max_results": opts.results_capped(MAX_RESULTS),
        // Perplexity meters extracted content in tokens; our budget is in
        // characters, and ~4 chars a token is the usual English ratio.
        "max_tokens_per_page": opts.depth.content_cap() / 4,
    });
    if let Some(f) = domain_filter(opts) {
        body["search_domain_filter"] = json!(f);
    }
    if let Some(c) = country {
        body["country"] = json!(c.to_ascii_uppercase());
    }
    http::send(
        ME,
        http::client()
            .post(http::endpoint(URL))
            .bearer_auth(&cfg.api_key)
            .json(&body),
    )
    .await
}

fn hit(r: &Value) -> Hit {
    Hit {
        title: http::s(r, "title"),
        url: http::s(r, "url"),
        content: http::s(r, "snippet"),
        // `date` is publication, `last_updated` the crawl - publication is the
        // one a reader means by "when was this written"
        published: ["date", "last_updated"]
            .iter()
            .filter_map(|k| r.get(k).and_then(Value::as_str))
            .map(str::trim)
            .find(|d| !d.is_empty())
            .map(|d| d.chars().take(10).collect()),
    }
}
