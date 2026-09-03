//! Firecrawl - a scraper with a search front door. Alone among the five it
//! will go and FETCH each result and hand back the page as markdown, so the
//! model reads the article instead of a search blurb.
//!
//! What we use: `scrapeOptions.formats: [{type: markdown}]` at medium and high
//! depth (the whole reason to pick Firecrawl), native `includeDomains`/
//! `excludeDomains`, `country`, the free-form `location` place string, and its
//! own server-side `timeout`.
//!
//! Two details that matter:
//!
//! - At low depth we send no `scrapeOptions`. Scraping is what makes Firecrawl
//!   slow and expensive, and "low" is the caller saying they want a cheap
//!   look; descriptions alone are the honest reading of that.
//! - `timeout` is set below our own client deadline deliberately. Firecrawl
//!   defaults to 60 s, which our 20 s client would guillotine into a bare
//!   transport error with nothing to show. Told to finish sooner, Firecrawl
//!   returns the results it has with `markdown: null` on the pages it couldn't
//!   reach, and those degrade to their description instead of to nothing.
//!
//! `includeDomains` and `excludeDomains` are mutually exclusive here, so when
//! a caller declares both the allowlist goes to Firecrawl (it is the stronger
//! constraint and the one that shapes retrieval) and the blocklist is applied
//! by `http::finish` on the way out.
//!
//! Deliberately not sent: `categories` (github/research/pdf - no caller
//! signal), `sources` (web is the default and the only one this tool wants),
//! and the `json`/`summary` scrape formats, which would have Firecrawl's own
//! model paraphrase the page before ours ever sees it.

use crate::{
    Depth, Found, Hit, Provider, SearchConfig, SearchError, SearchOpts, SearchUsage, http,
};
use serde_json::{Value, json};

const ME: Provider = Provider::Firecrawl;
const URL: &str = "https://api.firecrawl.dev/v2/search";
/// Server-side deadline, comfortably inside the client's own (ms).
const BUDGET_MS: u64 = 15_000;

pub(crate) async fn search(
    cfg: &SearchConfig,
    opts: &SearchOpts,
    query: &str,
) -> Result<Found, SearchError> {
    let geo = opts.location.code().is_some() || opts.location.place().is_some();
    let v = match call(cfg, opts, query, true).await {
        Err(e) if geo && http::retryable(&e) => {
            http::note_retry(ME.label(), &e);
            call(cfg, opts, query, false).await?
        }
        r => r?,
    };
    // v2 keys results by source: `data.web`, `data.images`, `data.news`.
    Ok(Found {
        hits: v
            .pointer("/data/web")
            .and_then(Value::as_array)
            .map(|arr| arr.iter().map(hit).collect())
            .unwrap_or_default(),
        usage: usage(&v),
    })
}

/// Firecrawl bills in credits, and not one per search: a single-result search
/// with no scraping came back at 2. Nothing here tries to derive a rate from
/// that - the number it reports is the number we record.
fn usage(v: &Value) -> SearchUsage {
    SearchUsage {
        dollars: None,
        credits: v.get("creditsUsed").and_then(Value::as_u64),
        request_id: v.get("id").and_then(Value::as_str).map(str::to_string),
        detail: v.get("warning").and_then(Value::as_str).map(str::to_string),
    }
}

async fn call(
    cfg: &SearchConfig,
    opts: &SearchOpts,
    query: &str,
    localized: bool,
) -> Result<Value, SearchError> {
    let mut body = json!({
        "query": query,
        "limit": opts.results_capped(100),
        "timeout": BUDGET_MS,
    });
    if opts.depth != Depth::Low {
        body["scrapeOptions"] = json!({ "formats": [{ "type": "markdown" }] });
    }
    // mutually exclusive: the allowlist wins, the blocklist is enforced on the
    // results by http::finish
    if !opts.allowed_domains.is_empty() {
        body["includeDomains"] = json!(opts.allowed_domains);
    } else if !opts.blocked_domains.is_empty() {
        body["excludeDomains"] = json!(opts.blocked_domains);
    }
    if localized {
        if let Some(c) = opts.location.code() {
            body["country"] = json!(c.to_ascii_uppercase());
        }
        if let Some(p) = opts.location.place() {
            body["location"] = json!(p);
        }
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
    // `markdown` is the scraped page; it is null for anything Firecrawl could
    // not reach inside its budget, and those fall back to the description
    // rather than coming back empty.
    let md = http::s(r, "markdown");
    let content = if md.trim().is_empty() {
        http::s(r, "description")
    } else {
        md
    };
    Hit {
        title: http::s(r, "title"),
        url: http::s(r, "url"),
        content,
        // search results carry no publish date; scrape metadata may, when the
        // page declared one
        published: r
            .pointer("/metadata/publishedTime")
            .and_then(Value::as_str)
            .map(|d| d.chars().take(10).collect()),
    }
}
