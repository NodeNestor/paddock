//! Web-search provider client, shared across the split:
//!
//! - the **runner** executes the `web_search` tool (API conformance: OpenAI
//!   `web_search`, Anthropic `web_search_20250305`) with provider config
//!   declared at launch (`--web-search-provider`/`--web-search-api-key`) -
//!   the runner is stateless and has no settings table;
//! - the **manager** stores the user's provider choice in its SQLite, passes
//!   it to runners at spawn, and uses this client for the Studio's
//!   "test my key" button.
//!
//! Five providers, and each one is driven with its own best features rather
//! than a lowest-common-denominator query string - see `providers/` for what
//! each engine is actually good at and which of its knobs we reach. What the
//! callers hand us (both dialects) is normalized into [`SearchOpts`]: a
//! result count, a [`Depth`] dial, domain allow/block lists and a
//! [`Location`]. Everything a provider cannot express is either emulated here
//! (domain filtering is enforced on the results no matter what) or documented
//! as deliberately unsent in that provider's module.

mod fetch;
mod http;
mod providers;
mod rate;
mod wire;

pub use fetch::{
    FETCH_TOOL_NAME, FetchError, FetchOpts, FetchSpec, Fetched, anthropic_fetch_content,
    anthropic_fetch_error, anthropic_fetch_tool_def, fetch, same_url,
};
use serde_json::{Value, json};

/// The function name the model sees (and `classify_call` intercepts).
pub const TOOL_NAME: &str = "web_search";
const DESC: &str = "Search the web. Returns page titles, URLs, and text \
                    content. Use for current events, recent facts, or anything \
                    that may have changed since your training data.";

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Provider {
    Exa,
    Tavily,
    Firecrawl,
    Brave,
    Perplexity,
}

impl Provider {
    /// Every provider, in the order the Studio offers them. The one true list:
    /// `parse`, the "unknown provider" warning and the Studio's pills all read
    /// from here, so adding a provider is one variant plus one module.
    pub const ALL: [Provider; 5] = [
        Self::Exa,
        Self::Tavily,
        Self::Firecrawl,
        Self::Brave,
        Self::Perplexity,
    ];

    /// Config-file spelling. Case-insensitive on the way in - a hand-edited
    /// `web_search_provider = "Brave"` is obviously meant and shouldn't
    /// silently disable web search.
    pub fn parse(s: &str) -> Option<Self> {
        let s = s.trim();
        Self::ALL
            .into_iter()
            .find(|p| p.as_str().eq_ignore_ascii_case(s))
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Exa => "exa",
            Self::Tavily => "tavily",
            Self::Firecrawl => "firecrawl",
            Self::Brave => "brave",
            Self::Perplexity => "perplexity",
        }
    }

    /// Brand spelling, for error messages the user reads.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Exa => "Exa",
            Self::Tavily => "Tavily",
            Self::Firecrawl => "Firecrawl",
            Self::Brave => "Brave",
            Self::Perplexity => "Perplexity",
        }
    }

    /// Where a key comes from, so "needs an API key" is a next step and not a
    /// dead end. The Studio shows this next to the key field.
    pub fn key_url(&self) -> &'static str {
        match self {
            Self::Exa => "https://dashboard.exa.ai",
            Self::Tavily => "https://app.tavily.com",
            Self::Firecrawl => "https://firecrawl.dev/app/api-keys",
            Self::Brave => "https://api-dashboard.search.brave.com",
            Self::Perplexity => "https://www.perplexity.ai/account/api/keys",
        }
    }
}

/// "exa, tavily, firecrawl, brave or perplexity" - generated, so a new
/// provider can never be missing from the message that lists them.
fn provider_names() -> String {
    let all: Vec<&str> = Provider::ALL.iter().map(Provider::as_str).collect();
    match all.split_last() {
        Some((last, rest)) if !rest.is_empty() => format!("{} or {last}", rest.join(", ")),
        _ => all.join(""),
    }
}

#[derive(Clone)]
pub struct SearchConfig {
    pub provider: Provider,
    pub api_key: String,
}

impl SearchConfig {
    /// Derive a config from raw provider/key fields (the config-file pair) -
    /// one honest place for the "declared but unusable" warnings, shared by
    /// startup parsing and the runner's live config re-reads.
    pub fn from_fields(provider: Option<&str>, api_key: Option<&str>) -> Option<Self> {
        let name = provider?.trim();
        if name.is_empty() {
            return None;
        }
        let Some(provider) = Provider::parse(name) else {
            tracing::warn!(
                provider = name,
                expected = %provider_names(),
                "unknown web-search provider - web search disabled"
            );
            return None;
        };
        let key = api_key.unwrap_or("").trim();
        if key.is_empty() {
            tracing::warn!(
                provider = name,
                "web-search provider declared without an API key - web search disabled"
            );
            return None;
        }
        Some(Self {
            provider,
            api_key: key.to_string(),
        })
    }
}

/// OpenAI's `search_context_size`: the caller's one cost/quality dial. It is
/// worth more than a result count, so it drives three things per provider -
/// how many results come back, how hard the engine is asked to work (Exa's
/// retrieval mode, Tavily's search depth, whether Firecrawl scrapes the pages
/// at all), and how much page text we hand the model. Medium is the spec's
/// default and ours.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Depth {
    Low,
    #[default]
    Medium,
    High,
}

impl Depth {
    fn from_size(size: Option<&str>) -> Self {
        match size {
            Some("low") => Self::Low,
            Some("high") => Self::High,
            _ => Self::Medium,
        }
    }

    /// Results to ask for when the caller named no count of its own.
    pub fn results(self) -> usize {
        match self {
            Self::Low => 3,
            Self::Medium => 5,
            Self::High => 8,
        }
    }

    /// Per-result page text handed to the model, in characters (~4 chars a
    /// token). This is both what we ask providers for and the ceiling every
    /// hit is trimmed to, so one dial governs the whole context cost.
    pub fn content_cap(self) -> usize {
        match self {
            Self::Low => 1_000,
            Self::Medium => 2_000,
            Self::High => 4_000,
        }
    }
}

/// The caller's `user_location`. Both dialects send an "approximate" location
/// with up to four fields; we keep all of them because providers want
/// different shapes - Exa and Perplexity take a country code, Firecrawl takes
/// a free-form "city,region,country" place.
#[derive(Clone, Default)]
pub struct Location {
    /// two-letter ISO-3166 country code, as the APIs define it
    pub country: Option<String>,
    pub city: Option<String>,
    pub region: Option<String>,
    pub timezone: Option<String>,
}

impl Location {
    /// Both dialects spell the location the same way - an "approximate"
    /// object with `city` / `region` / `country` / `timezone` - so both parse
    /// it here. Blank strings are dropped: a provider given `city: ""` is
    /// being told something false about where the user is.
    pub fn from_json(v: Option<&Value>) -> Self {
        let get = |k: &str| {
            v.and_then(|l| l.get(k))
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_string)
        };
        Self {
            country: get("country"),
            city: get("city"),
            region: get("region"),
            timezone: get("timezone"),
        }
    }

    /// Lower-cased two-letter code, or None if it isn't one. Guarding the
    /// shape here keeps a malformed `user_location` from 400-ing a provider
    /// that validates the field.
    pub fn code(&self) -> Option<String> {
        let c = self.country.as_deref()?.trim();
        (c.len() == 2 && c.chars().all(|ch| ch.is_ascii_alphabetic()))
            .then(|| c.to_ascii_lowercase())
    }

    /// "San Francisco,California,United States" - Firecrawl's free-form
    /// `location`, from whichever parts the caller sent. The country is
    /// title-cased on the way out: the table stores lower-case because
    /// Tavily's enum is spelled that way, while Firecrawl's own example is
    /// capitalized. A free-form field is probably case-blind, but matching the
    /// documented example costs nothing and guessing wrong might not.
    pub fn place(&self) -> Option<String> {
        let parts: Vec<String> = [self.city.as_deref(), self.region.as_deref()]
            .into_iter()
            .flatten()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .chain(
                self.code()
                    .as_deref()
                    .and_then(providers::country_name)
                    .map(title_case),
            )
            .collect();
        (!parts.is_empty()).then(|| parts.join(","))
    }
}

/// "united states" -> "United States". Word-wise and ASCII-first-letter only,
/// which is all these names need - anything cleverer would mangle "cote
/// d'ivoire" for no gain.
fn title_case(s: &str) -> String {
    s.split(' ')
        .map(|w| match w.chars().next() {
            Some(c) => c
                .to_uppercase()
                .chain(w.chars().skip(1))
                .collect::<String>(),
            None => String::new(),
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Per-request knobs from the caller's tool declaration (OpenAI `filters` /
/// `search_context_size` / `user_location`, Anthropic `allowed_domains` /
/// `blocked_domains` / `user_location`).
#[derive(Clone, Default)]
pub struct SearchOpts {
    /// results per search; 0 = whatever `depth` implies.
    pub count: usize,
    pub depth: Depth,
    pub allowed_domains: Vec<String>,
    pub blocked_domains: Vec<String>,
    pub location: Location,
}

impl SearchOpts {
    /// OpenAI `search_context_size` -> the depth dial.
    pub fn context_size(mut self, size: Option<&str>) -> Self {
        self.depth = Depth::from_size(size);
        self
    }

    /// The result count to actually ask for.
    fn results(&self) -> usize {
        if self.count > 0 {
            self.count
        } else {
            self.depth.results()
        }
    }

    /// Same, clamped to a provider's own documented maximum (Brave and
    /// Perplexity both stop at 20, Tavily at 20, Exa at 100).
    fn results_capped(&self, max: usize) -> usize {
        self.results().min(max).max(1)
    }

    fn has_domain_filter(&self) -> bool {
        !self.allowed_domains.is_empty() || !self.blocked_domains.is_empty()
    }
}

/// The resolved provider + the request's options: everything one search needs.
#[derive(Clone)]
pub struct WebSpec {
    pub cfg: SearchConfig,
    pub opts: SearchOpts,
}

/// A provider failure with enough shape to map onto the Anthropic error codes.
#[derive(Debug)]
pub struct SearchError {
    /// HTTP status from the provider, when the request got that far.
    pub status: Option<u16>,
    pub msg: String,
}

impl std::fmt::Display for SearchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.msg)
    }
}

impl SearchError {
    /// Anthropic `WebSearchToolResultError.error_code`.
    pub fn error_code(&self) -> &'static str {
        match self.status {
            Some(429) => "too_many_requests",
            Some(401) | Some(403) => "unavailable",
            _ => "unavailable",
        }
    }
}

fn params() -> Value {
    json!({
        "type": "object",
        "properties": {
            "query": { "type": "string", "description": "The search query." }
        },
        "required": ["query"],
        "additionalProperties": false
    })
}

/// `web_search` in OpenAI Chat/Responses function shape (what `prepare` merges
/// into the model's toolset).
pub fn tool_def() -> Value {
    json!({
        "type": "function",
        "function": { "name": TOOL_NAME, "description": DESC, "parameters": params() }
    })
}

/// `web_search` in Anthropic `{name, description, input_schema}` shape (what the
/// /v1/messages agent loop injects).
pub fn anthropic_tool_def() -> Value {
    json!({ "name": TOOL_NAME, "description": DESC, "input_schema": params() })
}

/// One normalized result, whichever provider produced it.
pub struct Hit {
    pub title: String,
    pub url: String,
    pub content: String,
    pub published: Option<String>,
}

/// What one search cost, exactly as the provider reported it - never inferred.
///
/// Every search bills the user's own key, and until this existed paddock spent
/// that money in silence. Four of the five providers do say, in three
/// different currencies: Exa prices in dollars (`costDollars`), Tavily and
/// Firecrawl in their own credits, Perplexity reports nothing at all. The
/// shapes are kept apart deliberately - Firecrawl charged 2 credits for a
/// one-result search with no scraping, so credits are not searches and
/// dollars are not credits, and a number we invented to make the columns
/// line up would be worse than an honest gap.
#[derive(Clone, Default, Debug, PartialEq)]
pub struct SearchUsage {
    /// real money, when the provider prices in it
    pub dollars: Option<f64>,
    /// provider-internal credits, which only mean something on their pricing page
    pub credits: Option<u64>,
    /// the provider's own request id, so a line here reconciles against their
    /// dashboard when a bill looks wrong
    pub request_id: Option<String>,
    /// anything else worth keeping. Exa reports which retrieval `auto`
    /// resolved to, the only feedback we get on what the depth dial bought.
    pub detail: Option<String>,
}

impl SearchUsage {
    /// Did the provider price this at all?
    pub fn is_empty(&self) -> bool {
        self.dollars.is_none() && self.credits.is_none()
    }

    /// One line for a log or a metric. `None` when the provider said nothing,
    /// so a caller can never print "cost: unknown" as though it were a number.
    pub fn summary(&self) -> Option<String> {
        let mut parts = Vec::new();
        if let Some(d) = self.dollars {
            parts.push(format!("${d:.4}"));
        }
        if let Some(c) = self.credits {
            parts.push(format!("{c} credit(s)"));
        }
        (!parts.is_empty()).then(|| parts.join(", "))
    }
}

/// One search's results and what it cost.
pub struct Found {
    pub hits: Vec<Hit>,
    pub usage: SearchUsage,
}

/// A spec `web_search_call` output item (openai `ResponseFunctionWebSearch`):
/// `status` ∈ in_progress/searching/completed/failed, `action` carries the
/// query. `action.sources` (url+title) is what the Studio renders as the
/// sources list, and `error` rides along on failure - both tolerated extras on
/// the SDK models.
pub fn call_item(
    id: &str,
    provider: Provider,
    status: &str,
    query: &str,
    hits: &[Hit],
    error: Option<&str>,
) -> Value {
    let sources: Vec<Value> = hits
        .iter()
        .map(|h| json!({"type": "url", "url": h.url, "title": h.title}))
        .collect();
    let mut o = serde_json::Map::new();
    o.insert("id".into(), json!(id));
    o.insert("type".into(), json!("web_search_call"));
    o.insert("status".into(), json!(status));
    // Who searched. Not in OpenAI's item shape - theirs describes one search
    // engine, ours can be any of five, and a reader who cannot tell which one
    // answered cannot judge the results or the bill. Additive and namespaced
    // so no SDK confuses it for spec: unknown fields are carried through by
    // both official SDKs, and a client that ignores it loses nothing.
    o.insert("paddock_provider".into(), json!(provider.as_str()));
    o.insert(
        "action".into(),
        json!({"type": "search", "query": query, "sources": sources}),
    );
    if let Some(e) = error {
        o.insert("error".into(), json!(e));
    }
    Value::Object(o)
}

/// Everything one executed search yields: the tool turn the model sees, the
/// hits behind it, the failure if there was one, and the bill.
pub struct Executed {
    pub feedback: String,
    pub hits: Vec<Hit>,
    pub error: Option<String>,
    /// completed | failed - the spec's `web_search_call.status`
    pub status: &'static str,
    pub usage: SearchUsage,
}

/// Execute one web search, mirroring `execute_mcp_call`.
pub async fn execute(spec: &WebSpec, query: &str) -> Executed {
    fn failed(msg: String) -> Executed {
        Executed {
            feedback: format!("web search failed: {msg}"),
            hits: Vec::new(),
            error: Some(msg),
            status: "failed",
            usage: SearchUsage::default(),
        }
    }
    if query.trim().is_empty() {
        return failed("the search query was empty".to_string());
    }
    match search(&spec.cfg, &spec.opts, query).await {
        Ok(found) => {
            // the one place the spend is announced. The provider's own request
            // id rides along, so a bill that looks wrong can be traced back to
            // the exact call rather than argued about.
            if let Some(cost) = found.usage.summary() {
                tracing::info!(
                    provider = spec.cfg.provider.as_str(),
                    cost,
                    request_id = found.usage.request_id.as_deref().unwrap_or(""),
                    detail = found.usage.detail.as_deref().unwrap_or(""),
                    results = found.hits.len(),
                    "web search billed"
                );
            }
            Executed {
                feedback: result_feedback(query, &found.hits),
                hits: found.hits,
                error: None,
                status: "completed",
                usage: found.usage,
            }
        }
        Err(e) => failed(e.msg),
    }
}

/// Run one query against the configured provider, then apply the guarantees
/// that must hold whichever engine answered:
///
/// 1. the caller's domain lists are a CONTRACT, so results are filtered here
///    too - providers honour them to wildly different degrees and Brave has
///    no domain parameter at all (see `providers::brave`);
/// 2. page text is trimmed to one budget, so `search_context_size` really is
///    the single dial that governs what a search costs in context;
/// 3. no more hits come back than were asked for, since providers that need
///    over-fetching to survive step 1 ask for extra.
pub async fn search(
    cfg: &SearchConfig,
    opts: &SearchOpts,
    query: &str,
) -> Result<Found, SearchError> {
    let mut found = match cfg.provider {
        Provider::Exa => providers::exa::search(cfg, opts, query).await?,
        Provider::Tavily => providers::tavily::search(cfg, opts, query).await?,
        Provider::Firecrawl => providers::firecrawl::search(cfg, opts, query).await?,
        Provider::Brave => providers::brave::search(cfg, opts, query).await?,
        Provider::Perplexity => providers::perplexity::search(cfg, opts, query).await?,
    };
    http::finish(&mut found.hits, opts);
    Ok(found)
}

/// Anthropic `web_search_tool_result` content on success: a list of
/// `web_search_result` blocks. (`encrypted_content` carries the page text -
/// opaque to clients, fed back on multi-turn just like Anthropic's.)
pub fn anthropic_result_content(hits: &[Hit]) -> Value {
    Value::Array(
        hits.iter()
            .map(|h| {
                let mut o = serde_json::Map::new();
                o.insert("type".into(), json!("web_search_result"));
                o.insert("url".into(), json!(h.url));
                o.insert("title".into(), json!(h.title));
                o.insert("encrypted_content".into(), json!(h.content));
                o.insert(
                    "page_age".into(),
                    h.published
                        .as_deref()
                        .map(|d| json!(d))
                        .unwrap_or(Value::Null),
                );
                Value::Object(o)
            })
            .collect(),
    )
}

/// The tool turn the model sees: numbered results with title, URL, date, and
/// page text. Content arrives already trimmed to the depth budget by
/// [`search`], so this formats and does not truncate.
pub fn result_feedback(query: &str, hits: &[Hit]) -> String {
    if hits.is_empty() {
        return format!("No web results for {query:?}.");
    }
    let mut s = format!("Web search results for {query:?}:\n");
    for (i, h) in hits.iter().enumerate() {
        s.push_str(&format!("\n{}. {}\n   {}\n", i + 1, h.title, h.url));
        if let Some(d) = &h.published {
            s.push_str(&format!("   Published: {d}\n"));
        }
        let text = h.content.trim();
        if !text.is_empty() {
            s.push_str(&format!("   {text}\n"));
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hit(url: &str) -> Hit {
        Hit {
            title: "t".into(),
            url: url.into(),
            content: String::new(),
            published: None,
        }
    }

    fn filtered(opts: &SearchOpts, urls: &[&str]) -> Vec<String> {
        // a count high enough that `finish`'s truncation never masks a
        // filtering bug - this is testing the domain contract, not the cap
        let opts = SearchOpts {
            count: 100,
            ..opts.clone()
        };
        let mut hits: Vec<Hit> = urls.iter().map(|u| hit(u)).collect();
        http::finish(&mut hits, &opts);
        hits.into_iter().map(|h| h.url).collect()
    }

    /// The item keeps the spec's own shape AND names the engine that answered.
    /// The provider is what lets a stored turn still say who searched after
    /// the endpoint has been re-pointed at a different provider - and it is
    /// the id, not the label, so the Studio can look up a mark by it.
    #[test]
    fn the_search_item_says_who_answered() {
        let it = call_item(
            "ws_1",
            Provider::Firecrawl,
            "completed",
            "rust async",
            &[hit("https://example.com/a")],
            None,
        );
        assert_eq!(it["type"], "web_search_call");
        assert_eq!(it["status"], "completed");
        assert_eq!(it["paddock_provider"], "firecrawl");
        assert_eq!(it["action"]["query"], "rust async");
        assert_eq!(it["action"]["sources"][0]["url"], "https://example.com/a");
        // a spec field must never be displaced by the extension
        assert_eq!(it["id"], "ws_1");
        assert!(it.get("error").is_none(), "no error on a completed search");

        // A failure still names the provider - "which of my five keys just
        // broke" is the whole question at that moment.
        let bad = call_item("ws_2", Provider::Brave, "failed", "q", &[], Some("401"));
        assert_eq!(bad["paddock_provider"], "brave");
        assert_eq!(bad["error"], "401");
    }

    #[test]
    fn every_provider_round_trips_its_config_spelling() {
        for p in Provider::ALL {
            assert_eq!(
                Provider::parse(p.as_str()),
                Some(p),
                "{} does not parse",
                p.as_str()
            );
            // hand-edited config files are not required to be lower-case
            assert_eq!(
                Provider::parse(p.label()),
                Some(p),
                "{} label does not parse",
                p.label()
            );
            assert!(
                p.key_url().starts_with("https://"),
                "{} has no key url",
                p.as_str()
            );
        }
        assert_eq!(Provider::parse("bing"), None);
        // the "expected ..." warning is generated, so it can never go stale
        let names = provider_names();
        for p in Provider::ALL {
            assert!(names.contains(p.as_str()), "{names} omits {}", p.as_str());
        }
    }

    #[test]
    fn an_allowlist_keeps_subdomains_and_drops_lookalikes() {
        let opts = SearchOpts {
            allowed_domains: vec!["example.com".into(), "https://www.rust-lang.org/".into()],
            ..Default::default()
        };
        assert_eq!(
            filtered(
                &opts,
                &[
                    "https://example.com/a",
                    "https://docs.example.com/b",
                    "https://www.example.com/c",
                    "https://rust-lang.org/d",
                    // the whole point: a lookalike host must not pass
                    "https://notexample.com/e",
                    "https://example.com.evil.net/f",
                    "https://other.org/g",
                ]
            ),
            vec![
                "https://example.com/a",
                "https://docs.example.com/b",
                "https://www.example.com/c",
                "https://rust-lang.org/d",
            ]
        );
    }

    #[test]
    fn a_blocklist_drops_the_domain_and_its_subdomains() {
        let opts = SearchOpts {
            blocked_domains: vec!["spam.example".into()],
            ..Default::default()
        };
        assert_eq!(
            filtered(
                &opts,
                &[
                    "https://a.spam.example/x",
                    "https://spam.example/y",
                    "https://ok.net/z"
                ]
            ),
            vec!["https://ok.net/z"]
        );
    }

    #[test]
    fn a_declared_path_narrows_to_that_section() {
        let opts = SearchOpts {
            allowed_domains: vec!["nature.com/articles".into()],
            ..Default::default()
        };
        assert_eq!(
            filtered(
                &opts,
                &[
                    "https://www.nature.com/articles/x",
                    "https://www.nature.com/news/y"
                ]
            ),
            vec!["https://www.nature.com/articles/x"]
        );
    }

    #[test]
    fn an_unparseable_url_survives_only_when_nothing_was_required() {
        let allow = SearchOpts {
            allowed_domains: vec!["example.com".into()],
            ..Default::default()
        };
        assert!(filtered(&allow, &["not a url"]).is_empty());
        let block = SearchOpts {
            blocked_domains: vec!["spam.example".into()],
            ..Default::default()
        };
        assert_eq!(filtered(&block, &["not a url"]), vec!["not a url"]);
    }

    #[test]
    fn depth_drives_count_and_context_budget_together() {
        assert_eq!(Depth::from_size(Some("low")), Depth::Low);
        assert_eq!(Depth::from_size(Some("high")), Depth::High);
        assert_eq!(Depth::from_size(None), Depth::Medium);
        assert_eq!(Depth::from_size(Some("nonsense")), Depth::Medium);
        let mut last = 0;
        for d in [Depth::Low, Depth::Medium, Depth::High] {
            assert!(d.results() > last, "results must grow with depth");
            last = d.results();
        }
        assert!(Depth::Low.content_cap() < Depth::High.content_cap());
        // an explicit count wins over the depth default, but a provider's own
        // ceiling still wins over the caller
        let opts = SearchOpts {
            count: 50,
            ..Default::default()
        };
        assert_eq!(opts.results(), 50);
        assert_eq!(opts.results_capped(20), 20);
        assert_eq!(SearchOpts::default().results(), Depth::Medium.results());
    }

    #[test]
    fn a_location_only_travels_when_it_is_actually_a_country_code() {
        let l = Location {
            country: Some("US".into()),
            ..Default::default()
        };
        assert_eq!(l.code().as_deref(), Some("us"));
        for bad in ["", "USA", "u", "1s"] {
            let l = Location {
                country: Some(bad.into()),
                ..Default::default()
            };
            assert_eq!(l.code(), None, "{bad:?} is not a country code");
        }
        // Firecrawl's free-form place uses whatever parts the caller sent, and
        // spells the country the way search engines expect to read it
        let full = Location {
            country: Some("us".into()),
            city: Some("San Francisco".into()),
            region: Some("California".into()),
            timezone: None,
        };
        assert_eq!(
            full.place().as_deref(),
            Some("San Francisco,California,United States")
        );
        assert_eq!(Location::default().place(), None);
    }

    #[test]
    fn hits_are_trimmed_to_one_budget_however_chatty_the_provider() {
        // providers are ASKED for the budget; nothing stops one ignoring it.
        // The ceiling is enforced on the way out rather than trusted on the
        // way in, so a chatty engine can't quietly cost 10x the context.
        let opts = SearchOpts {
            depth: Depth::Low,
            count: 2,
            ..Default::default()
        };
        let mut hits: Vec<Hit> = (0..5)
            .map(|i| {
                let mut h = hit(&format!("https://example.com/{i}"));
                // multi-byte deliberately: a naive byte slice would panic here
                h.content = "påddock".repeat(2_000);
                h
            })
            .collect();
        http::finish(&mut hits, &opts);
        assert_eq!(hits.len(), 2, "more results than asked for");
        for h in &hits {
            assert_eq!(h.content.chars().count(), Depth::Low.content_cap());
        }
    }
}
