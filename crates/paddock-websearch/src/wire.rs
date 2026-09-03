//! What each provider actually puts on the wire, and what it reads back.
//!
//! Five APIs, five spellings, and no way to hold a key for any of them in CI -
//! so a mistyped field name would otherwise surface as a live 400 in front of
//! a user. These tests stand a mock HTTP server in front of the clients and
//! assert both directions: the request carries the auth header, path and the
//! knobs that make each provider worth choosing, and the response is read out
//! of the shape that provider returns.
//!
//! The canned response deliberately carries every provider's field names at
//! once (`text` and `content` and `snippet`, nested under `results` and
//! `data.web` and `web.results`). That is what makes "Exa reads `text`,
//! Perplexity reads `snippet`, Tavily prefers `raw_content`" a real assertion
//! rather than a coincidence: a client reading the wrong field would come back
//! with the wrong provider's words, not with nothing.

#![cfg(test)]

use crate::{Depth, Location, Provider, SearchConfig, SearchOpts, http, search};
use serde_json::{Value, json};
use std::sync::{Mutex, OnceLock};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

/// Every request the mock server has taken, in order.
static SEEN: Mutex<Vec<Request>> = Mutex::new(Vec::new());

#[derive(Clone)]
struct Request {
    /// "POST /v2/search?x=1" - method, path and query as sent
    line: String,
    headers: Vec<(String, String)>,
    body: Value,
}

impl Request {
    fn header(&self, name: &str) -> String {
        self.headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.clone())
            .unwrap_or_default()
    }
    /// A query parameter from the request line (Brave is the GET one). The
    /// line is "GET <target> HTTP/1.1", so the version has to come off before
    /// the query is split or the last parameter reads back with it attached.
    fn param(&self, name: &str) -> Option<String> {
        let target = self.line.split_whitespace().nth(1)?;
        let q = target.split('?').nth(1)?;
        q.split('&').find_map(|kv| {
            let (k, v) = kv.split_once('=')?;
            (k == name).then(|| percent_decode(v))
        })
    }
}

/// Enough of percent-decoding for a test assertion: `+` and `%XX`.
fn percent_decode(s: &str) -> String {
    let b = s.replace('+', " ");
    let b = b.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%'
            && i + 2 < b.len()
            && let Ok(v) =
                u8::from_str_radix(std::str::from_utf8(&b[i + 1..i + 3]).unwrap_or(""), 16)
        {
            out.push(v);
            i += 3;
            continue;
        }
        out.push(b[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// One result object wearing every provider's field names at once, so each
/// client has to pick its own out of the crowd.
fn canned_result() -> Value {
    json!({
        "title": "Paddock",
        "url": "https://example.com/a",
        // exa
        "text": "exa-text",
        "highlights": ["exa-highlight"],
        "publishedDate": "2026-08-14T09:00:00.000Z",
        // tavily (content) and firecrawl/brave (description)
        "content": "tavily-content",
        "raw_content": "tavily-raw",
        "description": "brave-description",
        "extra_snippets": ["brave-extra"],
        "published_date": "2026-08-13",
        // firecrawl
        "markdown": "firecrawl-markdown",
        "metadata": { "publishedTime": "2026-08-12T00:00:00Z" },
        // perplexity
        "snippet": "perplexity-snippet",
        "date": "2026-08-11",
    })
}

fn canned_body() -> String {
    let r = canned_result();
    // one body carrying all three envelope shapes: `results` (exa, tavily,
    // perplexity), `data.web` (firecrawl), `web.results` (brave)
    json!({
        "results": [r],
        "data": { "web": [r] },
        "web": { "results": [r] },
    })
    .to_string()
}

/// One test at a time: the mock records into a single `SEEN`, and
/// `#[tokio::test]` would otherwise run these concurrently and let one test
/// read another's request. Async-aware, because the guard is deliberately held
/// across the search itself.
static ONE_AT_A_TIME: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

async fn serialize() -> tokio::sync::MutexGuard<'static, ()> {
    ONE_AT_A_TIME.lock().await
}

/// Poison is stepped over rather than propagated, so one failing assertion
/// doesn't cascade into six unrelated failures.
fn seen() -> std::sync::MutexGuard<'static, Vec<Request>> {
    SEEN.lock().unwrap_or_else(|e| e.into_inner())
}

/// Start the mock once for the whole suite and point every provider at it.
///
/// It gets its own thread and runtime deliberately: `#[tokio::test]` builds a
/// runtime per test and drops it at the end, which would take a spawned accept
/// loop down with it and leave every later test connecting to nothing.
fn mock_url() -> String {
    static URL: OnceLock<String> = OnceLock::new();
    URL.get_or_init(|| {
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("mock runtime");
            rt.block_on(serve(tx));
        });
        let addr = rx.recv().expect("mock server never bound");
        format!("http://{addr}/mock")
    })
    .clone()
}

async fn serve(tx: std::sync::mpsc::Sender<std::net::SocketAddr>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind mock");
    tx.send(listener.local_addr().expect("mock addr"))
        .expect("report mock addr");
    loop {
        let Ok((mut sock, _)) = listener.accept().await else {
            continue;
        };
        tokio::spawn(async move {
            let mut buf = Vec::new();
            let mut chunk = [0u8; 4096];
            // read until the headers are complete, then until the declared
            // body has arrived - enough HTTP for a client we control
            loop {
                let Ok(n) = sock.read(&mut chunk).await else {
                    return;
                };
                if n == 0 {
                    break;
                }
                buf.extend_from_slice(&chunk[..n]);
                let text = String::from_utf8_lossy(&buf).into_owned();
                let Some(head_end) = text.find("\r\n\r\n") else {
                    continue;
                };
                let head = &text[..head_end];
                let body = &text[head_end + 4..];
                let len: usize = head
                    .lines()
                    .find_map(|l| {
                        let (k, v) = l.split_once(':')?;
                        k.eq_ignore_ascii_case("content-length")
                            .then(|| v.trim().parse().ok())
                            .flatten()
                    })
                    .unwrap_or(0);
                if body.len() < len {
                    continue;
                }
                let mut lines = head.lines();
                let line = lines.next().unwrap_or("").trim().to_string();
                let headers = lines
                    .filter_map(|l| l.split_once(':'))
                    .map(|(k, v)| (k.trim().to_string(), v.trim().to_string()))
                    .collect();
                let body = serde_json::from_str(body).unwrap_or(Value::Null);
                seen().push(Request {
                    line,
                    headers,
                    body,
                });
                break;
            }
            let payload = canned_body();
            let res = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{payload}",
                payload.len()
            );
            let _ = sock.write_all(res.as_bytes()).await;
            let _ = sock.shutdown().await;
        });
    }
}

/// Run one search against the mock and hand back (the hits, the request the
/// provider sent).
async fn run(provider: Provider, opts: SearchOpts) -> (Vec<crate::Hit>, Request) {
    let _ = http::TEST_URL.set(mock_url());
    seen().clear();
    let cfg = SearchConfig {
        provider,
        api_key: "test-key".into(),
    };
    let found = search(&cfg, &opts, "who makes paddock")
        .await
        .expect("mock search");
    let req = seen().last().cloned().expect("no request reached the mock");
    (found.hits, req)
}

fn deep() -> SearchOpts {
    SearchOpts {
        depth: Depth::High,
        location: Location {
            country: Some("SE".into()),
            city: Some("Stockholm".into()),
            region: Some("Stockholm County".into()),
            timezone: None,
        },
        ..Default::default()
    }
}

#[tokio::test]
async fn exa_sends_its_retrieval_mode_and_reads_text_plus_highlights() {
    let _one = serialize().await;
    let (hits, req) = run(Provider::Exa, deep()).await;
    assert_eq!(
        req.header("x-api-key"),
        "test-key",
        "Exa authenticates on x-api-key"
    );
    assert_eq!(req.body["type"], "auto");
    assert_eq!(
        req.body["contents"]["text"]["maxCharacters"],
        Depth::High.content_cap()
    );
    assert_eq!(req.body["contents"]["highlights"], true);
    assert_eq!(req.body["userLocation"], "SE");
    // highlights lead so they survive the content cap, then the page text
    assert_eq!(hits[0].content, "exa-highlight\n\nexa-text");
    assert_eq!(hits[0].published.as_deref(), Some("2026-08-14"));

    // low depth is a different question: cheap retrieval, no highlights
    let (_, req) = run(
        Provider::Exa,
        SearchOpts {
            depth: Depth::Low,
            ..Default::default()
        },
    )
    .await;
    assert_eq!(req.body["type"], "fast");
    assert!(req.body["contents"].get("highlights").is_none());
}

#[tokio::test]
async fn tavily_maps_depth_onto_its_own_rungs_and_names_the_country() {
    let _one = serialize().await;
    let (hits, req) = run(Provider::Tavily, deep()).await;
    assert_eq!(req.header("authorization"), "Bearer test-key");
    assert_eq!(req.body["search_depth"], "advanced");
    assert_eq!(req.body["chunks_per_source"], 3);
    assert_eq!(req.body["auto_parameters"], true);
    assert_eq!(req.body["include_raw_content"], "markdown");
    // the whole point of the name table: Tavily's enum is words, not codes
    assert_eq!(req.body["country"], "sweden");
    // whole-page markdown beats the chunk summary when it came
    assert_eq!(hits[0].content, "tavily-raw");

    let (_, req) = run(
        Provider::Tavily,
        SearchOpts {
            depth: Depth::Low,
            ..Default::default()
        },
    )
    .await;
    assert_eq!(req.body["search_depth"], "fast");
    assert_eq!(req.body["auto_parameters"], false);
    assert!(req.body.get("include_raw_content").is_none());

    // Tavily rejects `country` outright on its fast rungs, so a low-depth
    // search that does know where the user is steps up to basic rather than
    // dropping the location. Get this wrong and every cheap localized search
    // is a 400 that only the retry net rescues, one wasted round trip at a
    // time.
    let low_here = SearchOpts {
        depth: Depth::Low,
        location: Location {
            country: Some("SE".into()),
            ..Default::default()
        },
        ..Default::default()
    };
    let (_, req) = run(Provider::Tavily, low_here).await;
    assert_eq!(
        req.body["search_depth"], "basic",
        "country cannot ride on fast"
    );
    assert_eq!(req.body["country"], "sweden");
}

#[tokio::test]
async fn firecrawl_scrapes_only_when_asked_and_bounds_its_own_deadline() {
    let _one = serialize().await;
    let (hits, req) = run(Provider::Firecrawl, deep()).await;
    assert_eq!(req.header("authorization"), "Bearer test-key");
    assert_eq!(req.body["scrapeOptions"]["formats"][0]["type"], "markdown");
    assert_eq!(req.body["country"], "SE");
    assert_eq!(req.body["location"], "Stockholm,Stockholm County,Sweden");
    // must finish inside our own client deadline, or we get nothing at all
    assert!(req.body["timeout"].as_u64().is_some_and(|t| t < 20_000));
    // read out of data.web, and the scraped page beats the blurb
    assert_eq!(hits[0].content, "firecrawl-markdown");

    // low depth means "cheap look": no scraping at all
    let (_, req) = run(
        Provider::Firecrawl,
        SearchOpts {
            depth: Depth::Low,
            ..Default::default()
        },
    )
    .await;
    assert!(req.body.get("scrapeOptions").is_none());
}

#[tokio::test]
async fn brave_is_a_get_with_its_own_token_header_and_snippet_excerpts() {
    let _one = serialize().await;
    let (hits, req) = run(Provider::Brave, deep()).await;
    assert!(req.line.starts_with("GET "), "Brave is a GET: {}", req.line);
    assert_eq!(req.header("x-subscription-token"), "test-key");
    assert_eq!(req.param("extra_snippets").as_deref(), Some("true"));
    assert_eq!(req.param("country").as_deref(), Some("SE"));
    // description plus the extra excerpts is as close to page text as Brave gets
    assert_eq!(hits[0].content, "brave-description\nbrave-extra");
}

#[tokio::test]
async fn brave_expresses_the_domain_filters_it_can_and_enforces_the_rest() {
    let _one = serialize().await;
    // one allowed domain is expressible as an operator
    let one = SearchOpts {
        allowed_domains: vec!["example.com".into()],
        ..Default::default()
    };
    let (_, req) = run(Provider::Brave, one).await;
    let q = req.param("q").unwrap_or_default();
    assert!(
        q.contains("site:example.com"),
        "single allowlist should ride as an operator: {q}"
    );
    assert_eq!(
        req.param("count").as_deref(),
        Some("20"),
        "over-fetch when filtering"
    );

    // two are not (Brave documents no OR), so no operator is guessed - and the
    // contract still holds because the results are filtered on the way out
    let two = SearchOpts {
        allowed_domains: vec!["nowhere.test".into(), "elsewhere.test".into()],
        ..Default::default()
    };
    let (hits, req) = run(Provider::Brave, two).await;
    let q = req.param("q").unwrap_or_default();
    assert!(
        !q.contains("site:"),
        "no unverified OR operator may be sent: {q}"
    );
    assert!(
        hits.is_empty(),
        "example.com is on neither allowlist, so it must not survive"
    );

    let blocked = SearchOpts {
        blocked_domains: vec!["spam.test".into()],
        ..Default::default()
    };
    let (_, req) = run(Provider::Brave, blocked).await;
    assert!(
        req.param("q")
            .unwrap_or_default()
            .contains("-site:spam.test")
    );
}

#[tokio::test]
async fn perplexity_ties_its_token_budget_to_the_depth_dial() {
    let _one = serialize().await;
    let (hits, req) = run(Provider::Perplexity, deep()).await;
    assert_eq!(req.header("authorization"), "Bearer test-key");
    assert_eq!(
        req.body["max_tokens_per_page"],
        Depth::High.content_cap() / 4
    );
    assert_eq!(req.body["country"], "SE");
    assert_eq!(hits[0].content, "perplexity-snippet");
    assert_eq!(hits[0].published.as_deref(), Some("2026-08-11"));

    // allowlist and denylist are not expressible together: the allowlist wins
    let both = SearchOpts {
        allowed_domains: vec!["example.com".into()],
        blocked_domains: vec!["spam.test".into()],
        ..Default::default()
    };
    let (_, req) = run(Provider::Perplexity, both).await;
    assert_eq!(req.body["search_domain_filter"], json!(["example.com"]));
}

#[tokio::test]
async fn every_provider_honours_the_result_count_and_the_content_budget() {
    let _one = serialize().await;
    for provider in Provider::ALL {
        let opts = SearchOpts {
            count: 1,
            depth: Depth::Low,
            ..Default::default()
        };
        let (hits, _) = run(provider, opts).await;
        assert_eq!(
            hits.len(),
            1,
            "{} returned the wrong count",
            provider.as_str()
        );
        let h = &hits[0];
        assert!(!h.title.is_empty(), "{} lost the title", provider.as_str());
        assert_eq!(
            h.url,
            "https://example.com/a",
            "{} lost the url",
            provider.as_str()
        );
        assert!(
            !h.content.is_empty() && h.content.chars().count() <= Depth::Low.content_cap(),
            "{} returned no usable content",
            provider.as_str()
        );
    }
}
