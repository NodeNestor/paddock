//! Tavily - built for RAG rather than for humans: it returns query-relevant
//! *chunks* of each page instead of a search-results-page blurb, and it will
//! tune its own retrieval settings from the query.
//!
//! What we use: `search_depth` across its four rungs mapped onto our depth
//! dial, `chunks_per_source` so the excerpt count rises with depth,
//! `auto_parameters` (Tavily reads the query and picks topic/time range
//! itself - its distinguishing feature), `include_raw_content: "markdown"` at
//! high depth for whole-page text, native `include_domains`/`exclude_domains`,
//! and `country`.
//!
//! Note on `auto_parameters`: it can promote `search_depth` to `advanced` and
//! bill two credits instead of one, so we set `search_depth` explicitly
//! alongside it. Auto then does the useful half (topic, recency) while the
//! cost dial stays the caller's.
//!
//! `country` is the odd one, twice over. Tavily takes an English NAME, not the
//! ISO code both API dialects hand us, so it goes through
//! `providers::country_name` - sending the raw code is a hard 400 ("Invalid
//! country. Must be a valid country name...", confirmed live). And `country` is
//! refused outright on the `fast`/`ultra-fast` rungs, which is why the depth
//! mapping below is a function of the location too, not of depth alone.
//!
//! Worth knowing when adding anything here: Tavily IGNORES unknown fields
//! rather than rejecting them (a made-up key returns 200). A request that
//! succeeds is therefore no evidence that a field name is right - new knobs
//! have to be checked by their effect on the response, the way
//! `chunks_per_source` and `include_raw_content` were.
//!
//! Deliberately not sent: `topic`/`time_range` by hand (auto_parameters does
//! it better than a guess), `include_answer` (we want sources for the model to
//! read, not Tavily's own summary standing in for them), and the image fields.

use crate::{
    Depth, Found, Hit, Provider, SearchConfig, SearchError, SearchOpts, SearchUsage, http,
    providers,
};
use serde_json::{Value, json};

const ME: Provider = Provider::Tavily;
const URL: &str = "https://api.tavily.com/search";

pub(crate) async fn search(
    cfg: &SearchConfig,
    opts: &SearchOpts,
    query: &str,
) -> Result<Found, SearchError> {
    let geo = opts
        .location
        .code()
        .and_then(|c| providers::country_name(&c));
    let v = match call(cfg, opts, query, geo).await {
        Err(e) if geo.is_some() && http::retryable(&e) => {
            http::note_retry(ME.label(), &e);
            call(cfg, opts, query, None).await?
        }
        r => r?,
    };
    Ok(Found {
        hits: v
            .get("results")
            .and_then(Value::as_array)
            .map(|arr| arr.iter().map(hit).collect())
            .unwrap_or_default(),
        usage: usage(&v),
    })
}

/// Credits, but only because we asked: `include_usage` is off by default, so
/// before it was sent Tavily's spend was invisible even though it was there.
fn usage(v: &Value) -> SearchUsage {
    SearchUsage {
        dollars: None,
        credits: v.pointer("/usage/credits").and_then(Value::as_u64),
        request_id: v
            .get("request_id")
            .and_then(Value::as_str)
            .map(str::to_string),
        detail: None,
    }
}

async fn call(
    cfg: &SearchConfig,
    opts: &SearchOpts,
    query: &str,
    country: Option<&str>,
) -> Result<Value, SearchError> {
    // Tavily REFUSES `country` on its fast rungs - "Country parameter is not
    // supported for fast or ultra-fast search_depth", a hard 400 (verified
    // live). A caller who asked for low context AND told us where
    // they are wants both, so the depth steps up to `basic` instead of the
    // location being quietly dropped: same one credit, and silently answering
    // with another region's results is exactly the failure this product
    // refuses. On the no-location retry below it falls back to `fast`.
    let depth = match (opts.depth, country.is_some()) {
        (Depth::Low, false) => "fast",
        (Depth::Low, true) | (Depth::Medium, _) => "basic",
        (Depth::High, _) => "advanced",
    };
    let mut body = json!({
        "query": query,
        "search_depth": depth,
        "max_results": opts.results_capped(20),
        // relevant excerpts per source: one is enough to judge a hit, three is
        // enough to answer from it
        "chunks_per_source": match opts.depth {
            Depth::Low => 1,
            Depth::Medium => 2,
            Depth::High => 3,
        },
        // let Tavily infer topic and recency from the query itself - the depth
        // above stays pinned so this cannot quietly double the credit cost
        "auto_parameters": opts.depth != Depth::Low,
        // off by default, and the only way Tavily will tell us what a search
        // cost - see `usage` below
        "include_usage": true,
    });
    if opts.depth == Depth::High {
        // whole page as markdown, not just the matched chunks
        body["include_raw_content"] = json!("markdown");
    }
    if !opts.allowed_domains.is_empty() {
        body["include_domains"] = json!(opts.allowed_domains);
    }
    if !opts.blocked_domains.is_empty() {
        body["exclude_domains"] = json!(opts.blocked_domains);
    }
    if let Some(c) = country {
        body["country"] = json!(c);
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
    // `content` is Tavily's relevance-ranked chunks; `raw_content` is the whole
    // page and only present at high depth. Prefer the fuller text when it came.
    let raw = http::s(r, "raw_content");
    let content = if raw.trim().is_empty() {
        http::s(r, "content")
    } else {
        raw
    };
    Hit {
        title: http::s(r, "title"),
        url: http::s(r, "url"),
        content,
        // only the `news` topic carries a date, so this is usually absent
        published: r
            .get("published_date")
            .and_then(Value::as_str)
            .map(str::to_string),
    }
}
