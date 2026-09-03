//! Live smoke tests against the real provider APIs.
//!
//! The unit tests in `src/wire.rs` pin what we put on the wire against a mock;
//! they cannot tell you whether a provider actually accepts it. This file is
//! the other half - it spends real credits, so every test is `#[ignore]`d and
//! runs only when asked:
//!
//! ```text
//! cargo test -p paddock-websearch --test live -- --ignored --nocapture
//! ```
//!
//! Keys come from the environment and never from the repo:
//!
//! ```text
//! PADDOCK_LIVE_EXA_KEY  PADDOCK_LIVE_TAVILY_KEY  PADDOCK_LIVE_FIRECRAWL_KEY
//! PADDOCK_LIVE_BRAVE_KEY  PADDOCK_LIVE_PERPLEXITY_KEY
//! ```
//!
//! A provider whose key is absent SAYS so and passes - a missing key is not a
//! failure, but it must not read as a green tick for a lane nobody ran.
//!
//! This is an integration test deliberately: it links the crate as a normal
//! dependency, so `src/wire.rs`'s `#[cfg(test)]` endpoint override doesn't
//! exist here and every request goes to the real host.

use paddock_websearch::{
    Depth, FetchOpts, Location, Provider, SearchConfig, SearchOpts, fetch, search,
};

fn key(provider: Provider) -> Option<String> {
    let var = format!("PADDOCK_LIVE_{}_KEY", provider.as_str().to_uppercase());
    std::env::var(var)
        .ok()
        .map(|k| k.trim().to_string())
        .filter(|k| !k.is_empty())
}

fn opts(depth: Depth) -> SearchOpts {
    // small deliberately: these are billed requests, and three results prove the
    // shape as well as ten do
    SearchOpts {
        count: 3,
        depth,
        ..Default::default()
    }
}

fn show(label: &str, found: &paddock_websearch::Found) {
    let cost = found
        .usage
        .summary()
        .map(|c| format!(" [{c}]"))
        .unwrap_or_else(|| " [no cost reported]".into());
    println!("  {label}: {} hit(s){cost}", found.hits.len());
    for h in &found.hits {
        let title = if h.title.is_empty() {
            "(no title)"
        } else {
            &h.title
        };
        println!(
            "    - {title} | {} | {} chars{}",
            h.url,
            h.content.chars().count(),
            h.published
                .as_deref()
                .map(|d| format!(" | {d}"))
                .unwrap_or_default(),
        );
    }
}

/// Three search probes plus a fetch, chosen to exercise what is most likely to
/// break against a real API: the plain path, the domain filter (a native
/// parameter for some, a query operator or result-side enforcement for
/// others), the localization our name/code mapping has to get right, and -
/// for the three that can - reading one named page.
async fn probe(provider: Provider) {
    let Some(api_key) = key(provider) else {
        println!(
            "
{} - SKIPPED, no key in the environment",
            provider.label()
        );
        return;
    };
    let cfg = SearchConfig { provider, api_key };
    println!(
        "
{} - live",
        provider.label()
    );

    let found = search(&cfg, &opts(Depth::Medium), "what is speculative decoding")
        .await
        .unwrap_or_else(|e| panic!("{} plain search failed: {e}", provider.label()));
    show("plain", &found);
    assert!(
        !found.hits.is_empty(),
        "{} returned nothing at all",
        provider.label()
    );
    for h in &found.hits {
        assert!(
            h.url.starts_with("http"),
            "{} returned a junk url: {:?}",
            provider.label(),
            h.url
        );
    }
    assert!(
        found.hits.iter().any(|h| !h.content.trim().is_empty()),
        "{} returned no usable content on any hit",
        provider.label()
    );

    let scoped = SearchOpts {
        allowed_domains: vec!["arxiv.org".into()],
        ..opts(Depth::Medium)
    };
    let found = search(&cfg, &scoped, "speculative decoding")
        .await
        .unwrap_or_else(|e| panic!("{} domain-scoped search failed: {e}", provider.label()));
    show("allowlist arxiv.org", &found);
    for h in &found.hits {
        assert!(
            h.url.contains("arxiv.org"),
            "{} broke the allowlist contract: {}",
            provider.label(),
            h.url
        );
    }

    // the risky one: Tavily wants a country NAME (and refuses it outright on
    // its fast rungs), everyone else a code
    let local = SearchOpts {
        location: Location {
            country: Some("SE".into()),
            city: Some("Stockholm".into()),
            region: Some("Stockholm County".into()),
            timezone: None,
        },
        ..opts(Depth::Low)
    };
    let found = search(&cfg, &local, "local news today")
        .await
        .unwrap_or_else(|e| panic!("{} localized search failed: {e}", provider.label()));
    show("located in SE", &found);

    probe_fetch(&cfg).await;
}

/// Reading one named page - the `web_fetch` half. Two of the five providers
/// sell search only, and those must refuse by NAME rather than failing per URL.
async fn probe_fetch(cfg: &SearchConfig) {
    let url = "https://example.com/";
    let o = FetchOpts::default().content_tokens(Some(500));
    match fetch(cfg, &o, url).await {
        Ok(f) => {
            let cost = f
                .usage
                .summary()
                .map(|c| format!(" [{c}]"))
                .unwrap_or_else(|| " [no cost reported]".into());
            println!(
                "  fetch: {} | {:?} | {} chars{cost}",
                f.url,
                f.title,
                f.content.chars().count()
            );
            assert!(
                !f.content.trim().is_empty(),
                "{} fetched an empty page",
                cfg.provider.label()
            );
            assert!(cfg.provider.can_fetch());
        }
        Err(e) => {
            // a search-only provider is expected here, and must say so clearly
            assert!(
                !cfg.provider.can_fetch(),
                "{} can fetch but failed: {} ({})",
                cfg.provider.label(),
                e.msg,
                e.code
            );
            assert_eq!(e.code, "unavailable");
            println!("  fetch: not offered - {}", e.msg);
        }
    }
}

#[tokio::test]
#[ignore = "spends real credits; needs PADDOCK_LIVE_EXA_KEY"]
async fn exa_live() {
    probe(Provider::Exa).await;
}

#[tokio::test]
#[ignore = "spends real credits; needs PADDOCK_LIVE_TAVILY_KEY"]
async fn tavily_live() {
    probe(Provider::Tavily).await;
}

#[tokio::test]
#[ignore = "spends real credits; needs PADDOCK_LIVE_FIRECRAWL_KEY"]
async fn firecrawl_live() {
    probe(Provider::Firecrawl).await;
}

#[tokio::test]
#[ignore = "spends real credits; needs PADDOCK_LIVE_BRAVE_KEY"]
async fn brave_live() {
    probe(Provider::Brave).await;
}

#[tokio::test]
#[ignore = "spends real credits; needs PADDOCK_LIVE_PERPLEXITY_KEY"]
async fn perplexity_live() {
    probe(Provider::Perplexity).await;
}
