//! Benchmark any OpenAI-compatible server over HTTP by timing its streaming
//! response. Used identically for Paddock's own server, mistral.rs (candle),
//! llama.cpp, vLLM - competitors stay external black boxes (our principle),
//! and one client measures everyone the same way.
//!
//! Metrics: TTFT (submit -> first token event) and decode tok/s (tokens after
//! the first / elapsed after the first). Prefill tok/s isn't reliably
//! client-observable, so it's left blank in HTTP mode.

use std::io::{BufRead, BufReader};
use std::time::{Duration, Instant};

use crate::timings::Timings;

/// `base_url` like http://127.0.0.1:1234/v1 ; hits {base}/completions with a
/// streaming request and times the SSE token events.
pub fn run(
    name: &str,
    base_url: &str,
    model: &str,
    prompt: &str,
    decode_tokens: usize,
) -> Result<Timings, String> {
    let url = format!("{}/completions", base_url.trim_end_matches('/'));
    let body = serde_json::json!({
        "model": model,
        "prompt": prompt,
        "max_tokens": decode_tokens,
        "temperature": 0.0,
        "stream": true,
    })
    .to_string();

    let t0 = Instant::now();
    let resp = ureq::post(&url)
        .header("content-type", "application/json")
        .send(&body)
        .map_err(|e| format!("request failed: {e}"))?;

    let reader = BufReader::new(resp.into_body().into_reader());
    let mut first: Option<Instant> = None;
    let mut last = t0;
    let mut tokens = 0usize;

    for line in reader.lines() {
        let line = line.map_err(|e| format!("read error: {e}"))?;
        let Some(data) = line.strip_prefix("data: ") else {
            continue;
        };
        if data == "[DONE]" {
            break;
        }
        // count any chunk carrying a non-empty text/content delta as one token
        let has_text = serde_json::from_str::<serde_json::Value>(data)
            .ok()
            .and_then(|v| {
                let c = &v["choices"][0];
                c["text"]
                    .as_str()
                    .or_else(|| c["delta"]["content"].as_str())
                    .map(|s| !s.is_empty())
            })
            .unwrap_or(false);
        if has_text {
            let now = Instant::now();
            if first.is_none() {
                first = Some(now);
            }
            last = now;
            tokens += 1;
        }
    }

    let ttft = first.map(|f| f - t0).unwrap_or_else(|| t0.elapsed());
    // decode window excludes the first token (that's TTFT territory)
    let decode = first
        .map(|f| last.saturating_duration_since(f))
        .unwrap_or(Duration::ZERO);
    let decode_tokens = tokens.saturating_sub(1);

    Ok(Timings {
        runner: name.to_owned(),
        load: Duration::ZERO, // server was already up; load not measured here
        prefill_tokens: 0,    // not client-observable
        prefill: Duration::ZERO,
        ttft,
        decode_tokens,
        decode,
    })
}
