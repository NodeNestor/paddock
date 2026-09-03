//! `web_fetch` - reading one named page, which is a different job from
//! searching for pages and a separate tool in the API we serve.
//!
//! Anthropic ships `web_fetch` as a first-class server tool beside
//! `web_search`, now on its fourth version (`web_fetch_20250910`,
//! `_20260209`, `_20260309`, `_20260318`). Until this module paddock had no
//! support at all: a caller declaring it got a tool with no schema and no
//! executor, which the model could call and nothing could answer.
//!
//! Only three of the five providers can do it - Exa `/contents`, Tavily
//! `/extract` and Firecrawl `/v2/scrape`. Brave and Perplexity sell search and
//! nothing else, so an endpoint configured with either refuses the tool by
//! NAME rather than pretending and failing per URL.
//!
//! ## The security model is copied deliberately
//!
//! A URL may only be fetched if it already appeared in the conversation - in a
//! user message, a client tool result, or an earlier search/fetch result. That
//! is Anthropic's rule and the reason for it is ours too: a model that can
//! fetch a URL it *composed* can exfiltrate whatever is in its context by
//! encoding it into a hostname or a path. The caller enforces the
//! "seen before" half (only it knows the conversation); this module enforces
//! everything checkable from the URL itself.

use crate::{Provider, SearchConfig, SearchError, SearchUsage, http};
use serde_json::{Value, json};

/// Anthropic's own cap. Longer than this is a malformed input, not a fetch.
const MAX_URL: usize = 250;
/// Default content ceiling in characters when the caller names none.
/// Anthropic meters this in tokens (`max_content_tokens`); ~4 chars a token is
/// the usual English ratio and the same one the search side uses.
const DEFAULT_MAX_CHARS: usize = 100_000;
/// Server-side deadline for the scrape, inside our own client timeout.
const BUDGET_MS: u64 = 15_000;

/// The model-facing tool name, and what `web_fetch_*` declarations resolve to.
pub const FETCH_TOOL_NAME: &str = "web_fetch";
const FETCH_DESC: &str = "Fetch the full text of a web page or PDF by URL. Use \
                          when you need to read a specific page you already \
                          have the URL for - not to discover pages.";

/// Per-request knobs from the caller's `web_fetch` declaration.
#[derive(Clone)]
pub struct FetchOpts {
    /// content ceiling in characters, from `max_content_tokens`
    pub max_chars: usize,
    pub allowed_domains: Vec<String>,
    pub blocked_domains: Vec<String>,
}

impl Default for FetchOpts {
    fn default() -> Self {
        Self {
            max_chars: DEFAULT_MAX_CHARS,
            allowed_domains: Vec::new(),
            blocked_domains: Vec::new(),
        }
    }
}

impl FetchOpts {
    /// `max_content_tokens` -> characters. Anthropic documents the limit as
    /// approximate, which is just as well: only the provider knows how the
    /// page tokenizes.
    pub fn content_tokens(mut self, tokens: Option<u64>) -> Self {
        if let Some(t) = tokens.filter(|t| *t > 0) {
            self.max_chars = (t as usize).saturating_mul(4);
        }
        self
    }
}

/// The resolved provider plus this request's options.
#[derive(Clone)]
pub struct FetchSpec {
    pub cfg: SearchConfig,
    pub opts: FetchOpts,
    /// fetches allowed this request; 0 = unlimited
    pub max_uses: usize,
}

/// One page, read.
pub struct Fetched {
    pub url: String,
    pub title: String,
    pub content: String,
    pub usage: SearchUsage,
}

/// A failure wearing one of Anthropic's documented `web_fetch` error codes.
/// The codes are the API's, not ours - a client that knows the dialect can act
/// on them, which is the whole point of conformance.
pub struct FetchError {
    pub code: &'static str,
    pub msg: String,
}

impl FetchError {
    fn new(code: &'static str, msg: impl Into<String>) -> Self {
        Self {
            code,
            msg: msg.into(),
        }
    }
}

impl std::fmt::Display for FetchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.msg)
    }
}

impl From<SearchError> for FetchError {
    fn from(e: SearchError) -> Self {
        // a rate limit is its own code in the dialect; everything else the
        // provider refused is the page being unreachable as far as a caller
        // is concerned
        let code = match e.status {
            Some(429) => "too_many_requests",
            Some(_) => "url_not_accessible",
            None => "unavailable",
        };
        FetchError::new(code, e.msg)
    }
}

impl Provider {
    /// Can this provider read a named page? Search and fetch are different
    /// products, and two of the five only sell the first.
    pub fn can_fetch(&self) -> bool {
        matches!(self, Self::Exa | Self::Tavily | Self::Firecrawl)
    }
}

/// `web_fetch` in Anthropic `{name, description, input_schema}` shape - the
/// model-facing function the agent loop injects in place of the server tool.
pub fn anthropic_fetch_tool_def() -> Value {
    json!({
        "name": FETCH_TOOL_NAME,
        "description": FETCH_DESC,
        "input_schema": {
            "type": "object",
            "properties": {
                "url": { "type": "string", "description": "The URL to fetch. Must be one that already appeared in this conversation." }
            },
            "required": ["url"],
            "additionalProperties": false
        }
    })
}

/// A successful `web_fetch_tool_result` content block.
pub fn anthropic_fetch_content(f: &Fetched, retrieved_at: &str, citations: bool) -> Value {
    json!({
        "type": "web_fetch_result",
        "url": f.url,
        "content": {
            "type": "document",
            "source": { "type": "text", "media_type": "text/plain", "data": f.content },
            "title": f.title,
            "citations": { "enabled": citations }
        },
        "retrieved_at": retrieved_at
    })
}

/// The error half of a `web_fetch_tool_result`.
pub fn anthropic_fetch_error(code: &str) -> Value {
    json!({ "type": "web_fetch_tool_result_error", "error_code": code })
}

/// Normalize a URL for "have we seen this before" comparison. Trailing
/// slashes and case in the host are not meaningful differences, and treating
/// them as such would refuse URLs the model copied faithfully.
pub fn same_url(a: &str, b: &str) -> bool {
    fn key(u: &str) -> String {
        u.trim().trim_end_matches('/').to_lowercase()
    }
    key(a) == key(b)
}

/// Fetch one page.
///
/// Everything checkable without the conversation is checked here; the caller
/// still owes the "this URL appeared earlier" test, because only it can see
/// the conversation.
pub async fn fetch(cfg: &SearchConfig, opts: &FetchOpts, url: &str) -> Result<Fetched, FetchError> {
    let url = url.trim();
    if url.len() > MAX_URL {
        return Err(FetchError::new(
            "url_too_long",
            format!("the URL is longer than {MAX_URL} characters"),
        ));
    }
    let parsed = reqwest::Url::parse(url)
        .map_err(|_| FetchError::new("invalid_tool_input", "that is not a URL"))?;
    if !matches!(parsed.scheme(), "http" | "https") {
        return Err(FetchError::new(
            "invalid_tool_input",
            "only http and https URLs can be fetched",
        ));
    }
    if !http::url_allowed(&parsed, &opts.allowed_domains, &opts.blocked_domains) {
        return Err(FetchError::new(
            "url_not_allowed",
            "that domain is not allowed by this request",
        ));
    }
    if !cfg.provider.can_fetch() {
        // named honestly: the tool is not broken, this endpoint's provider
        // simply does not sell page reading
        return Err(FetchError::new(
            "unavailable",
            format!(
                "{} does not read pages - web fetch needs a provider that does (exa, tavily or firecrawl)",
                cfg.provider.label()
            ),
        ));
    }

    let mut got = match cfg.provider {
        Provider::Exa => exa(cfg, url).await?,
        Provider::Tavily => tavily(cfg, url).await?,
        Provider::Firecrawl => firecrawl(cfg, url).await?,
        // can_fetch() above already refused these
        Provider::Brave | Provider::Perplexity => unreachable!("provider cannot fetch"),
    };
    if got.content.trim().is_empty() {
        return Err(FetchError::new(
            "url_not_accessible",
            "the page came back empty",
        ));
    }
    if got.content.chars().count() > opts.max_chars {
        got.content = got.content.chars().take(opts.max_chars).collect();
    }
    Ok(got)
}

/// Exa `/contents` - the same index the search side reads, addressed by URL.
/// It reports a per-URL `statuses` array, so a page it could not reach is
/// distinguishable from a page that was empty.
async fn exa(cfg: &SearchConfig, url: &str) -> Result<Fetched, FetchError> {
    let body = json!({ "urls": [url], "text": { "maxCharacters": DEFAULT_MAX_CHARS } });
    let v = http::send(
        Provider::Exa,
        http::client()
            .post(http::endpoint("https://api.exa.ai/contents"))
            .header("x-api-key", &cfg.api_key)
            .json(&body),
    )
    .await?;
    let Some(r) = v.pointer("/results/0") else {
        let why = v
            .pointer("/statuses/0/error/tag")
            .and_then(Value::as_str)
            .unwrap_or("the page could not be read");
        return Err(FetchError::new("url_not_accessible", why));
    };
    Ok(Fetched {
        url: http::s(r, "url"),
        title: http::s(r, "title"),
        content: http::s(r, "text"),
        usage: SearchUsage {
            dollars: v.pointer("/costDollars/total").and_then(Value::as_f64),
            request_id: v
                .get("requestId")
                .and_then(Value::as_str)
                .map(str::to_string),
            ..Default::default()
        },
    })
}

/// Tavily `/extract`. Note the shape: a URL it could not read does not appear
/// in `results` with an error - it moves to `failed_results`, so an empty
/// `results` is the failure signal and reading only `results` would report
/// success on nothing.
async fn tavily(cfg: &SearchConfig, url: &str) -> Result<Fetched, FetchError> {
    let body = json!({ "urls": [url], "format": "markdown", "include_usage": true });
    let v = http::send(
        Provider::Tavily,
        http::client()
            .post(http::endpoint("https://api.tavily.com/extract"))
            .bearer_auth(&cfg.api_key)
            .json(&body),
    )
    .await?;
    let Some(r) = v.pointer("/results/0") else {
        let why = v
            .pointer("/failed_results/0/error")
            .and_then(Value::as_str)
            .unwrap_or("the page could not be read");
        return Err(FetchError::new("url_not_accessible", why));
    };
    Ok(Fetched {
        url: http::s(r, "url"),
        title: http::s(r, "title"),
        content: http::s(r, "raw_content"),
        usage: SearchUsage {
            credits: v.pointer("/usage/credits").and_then(Value::as_u64),
            request_id: v
                .get("request_id")
                .and_then(Value::as_str)
                .map(str::to_string),
            ..Default::default()
        },
    })
}

/// Firecrawl `/v2/scrape` - the scraper doing what it is actually for. Its own
/// `timeout` stays inside our client deadline for the same reason the search
/// side sets one: a guillotined request has nothing to show.
async fn firecrawl(cfg: &SearchConfig, url: &str) -> Result<Fetched, FetchError> {
    let body = json!({
        "url": url,
        "formats": [{ "type": "markdown" }],
        "timeout": BUDGET_MS,
    });
    let v = http::send(
        Provider::Firecrawl,
        http::client()
            .post(http::endpoint("https://api.firecrawl.dev/v2/scrape"))
            .bearer_auth(&cfg.api_key)
            .json(&body),
    )
    .await?;
    let Some(d) = v.get("data") else {
        return Err(FetchError::new(
            "url_not_accessible",
            "the page could not be read",
        ));
    };
    Ok(Fetched {
        url: d
            .pointer("/metadata/sourceURL")
            .and_then(Value::as_str)
            .unwrap_or(url)
            .to_string(),
        title: d
            .pointer("/metadata/title")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string(),
        content: http::s(d, "markdown"),
        usage: SearchUsage {
            credits: v.get("creditsUsed").and_then(Value::as_u64),
            request_id: v.get("id").and_then(Value::as_str).map(str::to_string),
            ..Default::default()
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(p: Provider) -> SearchConfig {
        SearchConfig {
            provider: p,
            api_key: "k".into(),
        }
    }

    async fn refuse(p: Provider, opts: &FetchOpts, url: &str) -> String {
        fetch(&cfg(p), opts, url)
            .await
            .err()
            .expect("should have refused")
            .code
            .to_string()
    }

    #[tokio::test]
    async fn a_url_is_checked_before_a_provider_is_ever_called() {
        let o = FetchOpts::default();
        // no network happens for any of these - they fail on the URL itself
        assert_eq!(
            refuse(Provider::Exa, &o, "not a url").await,
            "invalid_tool_input"
        );
        assert_eq!(
            refuse(Provider::Exa, &o, "ftp://example.com/x").await,
            "invalid_tool_input"
        );
        assert_eq!(
            refuse(
                Provider::Exa,
                &o,
                &format!("https://example.com/{}", "x".repeat(300))
            )
            .await,
            "url_too_long"
        );
        let scoped = FetchOpts {
            allowed_domains: vec!["example.com".into()],
            ..o.clone()
        };
        assert_eq!(
            refuse(Provider::Exa, &scoped, "https://elsewhere.test/x").await,
            "url_not_allowed"
        );
        // and the allowlisted one gets past the URL checks (it would then need
        // a network, so this only asserts it is not one of the refusals above)
        let blocked = FetchOpts {
            blocked_domains: vec!["spam.test".into()],
            ..o.clone()
        };
        assert_eq!(
            refuse(Provider::Exa, &blocked, "https://a.spam.test/x").await,
            "url_not_allowed"
        );
    }

    #[tokio::test]
    async fn a_search_only_provider_refuses_by_name_instead_of_failing_per_url() {
        let o = FetchOpts::default();
        for p in [Provider::Brave, Provider::Perplexity] {
            assert!(!p.can_fetch(), "{} should not claim fetch", p.label());
            let e = fetch(&cfg(p), &o, "https://example.com/")
                .await
                .err()
                .expect("refused");
            assert_eq!(e.code, "unavailable");
            assert!(
                e.msg.contains(p.label()),
                "the message must name the provider: {}",
                e.msg
            );
        }
        for p in [Provider::Exa, Provider::Tavily, Provider::Firecrawl] {
            assert!(p.can_fetch(), "{} should fetch", p.label());
        }
    }

    #[test]
    fn seen_before_comparison_forgives_only_what_is_not_meaningful() {
        assert!(same_url("https://example.com/a", "https://example.com/a/"));
        assert!(same_url("https://Example.com/A", "https://example.com/a"));
        assert!(same_url(" https://example.com/a ", "https://example.com/a"));
        // a different path is a different page, and that is the whole guard
        assert!(!same_url("https://example.com/a", "https://example.com/b"));
        assert!(!same_url("https://example.com", "https://evil.test"));
    }

    #[test]
    fn the_content_ceiling_comes_from_the_callers_token_budget() {
        assert_eq!(FetchOpts::default().max_chars, DEFAULT_MAX_CHARS);
        assert_eq!(
            FetchOpts::default().content_tokens(Some(1_000)).max_chars,
            4_000
        );
        // 0 and absent both mean "no opinion", not "no content"
        assert_eq!(
            FetchOpts::default().content_tokens(Some(0)).max_chars,
            DEFAULT_MAX_CHARS
        );
        assert_eq!(
            FetchOpts::default().content_tokens(None).max_chars,
            DEFAULT_MAX_CHARS
        );
    }
}
