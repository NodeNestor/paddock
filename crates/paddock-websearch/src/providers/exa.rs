//! Exa - neural/semantic retrieval over its own index, and the only provider
//! here that will read a *description* of what you want rather than keywords.
//!
//! What we use: `type` to pick retrieval mode per depth (`fast` for a cheap
//! look, `auto` to let Exa choose neural vs keyword per query - the thing it
//! is actually good at), page `contents.text` at the depth's character budget,
//! query-relevant `contents.highlights`, native `includeDomains`/
//! `excludeDomains`, and `userLocation`.
//!
//! Deliberately not sent: `category` (company/news/paper/...) and the published
//! date windows, because neither API dialect gives us a signal to set them
//! from and a guessed category silently narrows the whole search; `subpages`,
//! which multiplies cost for a crawl the caller didn't ask for; and the
//! `deep`/`deep-reasoning` types, which routinely run past our 20 s deadline -
//! an agent turn stalling a minute on one search is a worse product than a
//! shallower answer.

use crate::{
    Depth, Found, Hit, Provider, SearchConfig, SearchError, SearchOpts, SearchUsage, http,
};
use serde_json::{Value, json};

const ME: Provider = Provider::Exa;
const URL: &str = "https://api.exa.ai/search";

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
    Ok(Found {
        hits: v
            .get("results")
            .and_then(Value::as_array)
            .map(|arr| arr.iter().map(hit).collect())
            .unwrap_or_default(),
        usage: usage(&v),
    })
}

/// Exa is the only one that prices in real money, and it also reports which
/// retrieval `auto` actually resolved to - the one piece of feedback we get on
/// what the depth dial bought.
fn usage(v: &Value) -> SearchUsage {
    SearchUsage {
        dollars: v.pointer("/costDollars/total").and_then(Value::as_f64),
        credits: None,
        request_id: v
            .get("requestId")
            .and_then(Value::as_str)
            .map(str::to_string),
        detail: v
            .get("resolvedSearchType")
            .and_then(Value::as_str)
            .map(|t| format!("resolved to {t}")),
    }
}

async fn call(
    cfg: &SearchConfig,
    opts: &SearchOpts,
    query: &str,
    country: Option<&str>,
) -> Result<Value, SearchError> {
    let cap = opts.depth.content_cap();
    let mut contents = json!({ "text": { "maxCharacters": cap } });
    if opts.depth != Depth::Low {
        // boolean form: Exa picks the sentences most relevant to the query.
        // They lead the content below, so the useful part survives trimming.
        contents["highlights"] = json!(true);
    }
    let mut body = json!({
        "query": query,
        // `fast` is Exa's low-latency retrieval; `auto` lets it choose neural
        // vs keyword per query, which is the whole reason to use Exa.
        "type": if opts.depth == Depth::Low { "fast" } else { "auto" },
        "numResults": opts.results_capped(100),
        "contents": contents,
    });
    if !opts.allowed_domains.is_empty() {
        body["includeDomains"] = json!(opts.allowed_domains);
    }
    if !opts.blocked_domains.is_empty() {
        body["excludeDomains"] = json!(opts.blocked_domains);
    }
    if let Some(c) = country {
        // Exa wants the bare two-letter code, upper-cased in its own examples
        body["userLocation"] = json!(c.to_ascii_uppercase());
    }
    http::send(
        ME,
        http::client()
            .post(http::endpoint(URL))
            .header("x-api-key", &cfg.api_key)
            .json(&body),
    )
    .await
}

fn hit(r: &Value) -> Hit {
    // Highlights first, then the page text: whichever sentences Exa judged
    // most relevant then survive the shared content cap, so a long page can't
    // push the answer off the end of the budget.
    let highlights = http::joined(r, "highlights", "\n");
    let text = http::s(r, "text");
    let content = match (highlights.is_empty(), text.is_empty()) {
        (false, false) => format!("{highlights}\n\n{text}"),
        (false, true) => highlights,
        _ => text,
    };
    Hit {
        title: http::s(r, "title"),
        url: http::s(r, "url"),
        content,
        // full ISO datetime -> keep the date part
        published: r
            .get("publishedDate")
            .and_then(Value::as_str)
            .map(|d| d.chars().take(10).collect()),
    }
}
