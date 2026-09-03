//! The concurrent / agentic-load benchmark. Fires N streaming sessions at
//! an OpenAI-compatible server *simultaneously*, each sharing a long common
//! prefix (the agentic shape: big system prompt + short unique turn), and
//! reports the numbers that separate a serial engine from a batched one:
//!
//!  - aggregate tok/s (total output tokens / wall clock) - a batched engine
//!    amortizes weight reads across the batch so this climbs with concurrency;
//!    a serial engine stays pinned at its single-stream rate.
//!  - TTFT p50/p99 - a serial engine queues later requests, wrecking the tail.
//!  - inter-token latency p50/p99 - smoothness under load.
//!
//! Runs identically against Paddock and llama.cpp/vLLM (external black boxes).
//! Blocking `ureq` on one thread per session; a start barrier makes them fire
//! together so the load is genuinely concurrent, not a rolling start.

use std::io::{BufRead, BufReader};
use std::sync::{Arc, Barrier};
use std::time::{Duration, Instant};

/// Per-session outcome.
struct SessionResult {
    submit: Instant,
    first: Option<Instant>,
    done: Instant,
    tokens: usize,
    /// gaps between consecutive token arrivals (inter-token latency samples)
    gaps_ms: Vec<f64>,
    error: Option<String>,
}

pub struct ConcReport {
    pub name: String,
    pub concurrency: usize,
    pub prefix_words: usize,
    pub total_tokens: usize,
    pub wall: Duration,
    pub aggregate_tok_s: f64,
    pub per_session_tok_s_mean: f64,
    pub ttft_p50_ms: f64,
    pub ttft_p99_ms: f64,
    pub itl_p50_ms: f64,
    pub itl_p99_ms: f64,
    pub errors: usize,
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let rank = (p / 100.0 * (sorted.len() as f64 - 1.0)).round() as usize;
    sorted[rank.min(sorted.len() - 1)]
}

/// A deterministic long shared prefix (~`words` words). Stands in for a big
/// shared system prompt / conversation history - the thing prefix caching wins.
fn shared_prefix(words: usize) -> String {
    const LEX: [&str; 16] = [
        "the",
        "function",
        "returns",
        "a",
        "value",
        "when",
        "the",
        "input",
        "buffer",
        "is",
        "valid",
        "otherwise",
        "it",
        "reports",
        "an",
        "error",
    ];
    let mut s = String::from(
        "You are a meticulous coding assistant. Follow the project conventions exactly. ",
    );
    for i in 0..words {
        s.push_str(LEX[i % LEX.len()]);
        s.push(' ');
    }
    s
}

/// One streaming session against {base}/completions; records token arrival times.
/// `delay_ms` staggers the fire time after the shared barrier (0 = fire together).
fn run_session(
    base_url: &str,
    model: &str,
    prompt: String,
    max_tokens: usize,
    barrier: &Barrier,
    delay_ms: u64,
) -> SessionResult {
    let url = format!("{}/completions", base_url.trim_end_matches('/'));
    let body = serde_json::json!({
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": true,
    })
    .to_string();

    // all sessions release together -> genuinely concurrent load; an optional
    // per-session delay staggers arrivals (so later sessions join mid-decode).
    barrier.wait();
    if delay_ms > 0 {
        std::thread::sleep(Duration::from_millis(delay_ms));
    }
    let submit = Instant::now();

    let resp = match ureq::post(&url)
        .header("content-type", "application/json")
        .send(&body)
    {
        Ok(r) => r,
        Err(e) => {
            return SessionResult {
                submit,
                first: None,
                done: Instant::now(),
                tokens: 0,
                gaps_ms: Vec::new(),
                error: Some(format!("request failed: {e}")),
            };
        }
    };

    let reader = BufReader::new(resp.into_body().into_reader());
    let mut first: Option<Instant> = None;
    let mut last = submit;
    let mut tokens = 0usize;
    let mut gaps_ms = Vec::new();
    let mut error = None;

    for line in reader.lines() {
        let line = match line {
            Ok(l) => l,
            Err(e) => {
                error = Some(format!("read error: {e}"));
                break;
            }
        };
        let Some(data) = line.strip_prefix("data: ") else {
            continue;
        };
        if data == "[DONE]" {
            break;
        }
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
            } else {
                gaps_ms.push((now - last).as_secs_f64() * 1e3);
            }
            last = now;
            tokens += 1;
        }
    }

    SessionResult {
        submit,
        first,
        done: Instant::now(),
        tokens,
        gaps_ms,
        error,
    }
}

#[allow(clippy::too_many_arguments)]
pub fn run(
    name: &str,
    base_url: &str,
    model: &str,
    concurrency: usize,
    prefix_words: usize,
    max_tokens: usize,
    stagger_ms: u64,
    unique: bool,
) -> Result<ConcReport, String> {
    let n = concurrency.max(1);
    let prefix = shared_prefix(prefix_words);
    let barrier = Arc::new(Barrier::new(n));

    let results: Vec<SessionResult> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..n)
            .map(|i| {
                let (base_url, model, prefix, barrier) =
                    (base_url, model, prefix.clone(), Arc::clone(&barrier));
                // `unique` gives each session a DISTINCT prefix (cold prefill, no
                // cache sharing) - stresses mid-decode prefill; otherwise the prefix
                // is shared (prefix-cache hits after the first).
                let prompt = if unique {
                    format!("Session {i} unique context. {prefix}\nTask {i}.\n")
                } else {
                    format!("{prefix}\nTask {i}: explain step {i} in one sentence.\n")
                };
                let delay = stagger_ms * i as u64;
                scope.spawn(move || {
                    run_session(base_url, model, prompt, max_tokens, &barrier, delay)
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| {
                h.join().unwrap_or_else(|_| SessionResult {
                    submit: Instant::now(),
                    first: None,
                    done: Instant::now(),
                    tokens: 0,
                    gaps_ms: Vec::new(),
                    error: Some("session thread panicked".into()),
                })
            })
            .collect()
    });

    let errors = results.iter().filter(|r| r.error.is_some()).count();
    let total_tokens: usize = results.iter().map(|r| r.tokens).sum();
    let start = results
        .iter()
        .map(|r| r.submit)
        .min()
        .unwrap_or_else(Instant::now);
    let end = results
        .iter()
        .map(|r| r.done)
        .max()
        .unwrap_or_else(Instant::now);
    let wall = end - start;
    let aggregate_tok_s = total_tokens as f64 / wall.as_secs_f64().max(1e-9);

    let per_session: Vec<f64> = results
        .iter()
        .filter_map(|r| {
            let f = r.first?;
            (r.tokens > 1).then(|| (r.tokens - 1) as f64 / (r.done - f).as_secs_f64().max(1e-9))
        })
        .collect();
    let per_session_tok_s_mean = if per_session.is_empty() {
        0.0
    } else {
        per_session.iter().sum::<f64>() / per_session.len() as f64
    };

    let mut ttfts: Vec<f64> = results
        .iter()
        .filter_map(|r| r.first.map(|f| (f - r.submit).as_secs_f64() * 1e3))
        .collect();
    ttfts.sort_by(f64::total_cmp);

    let mut itls: Vec<f64> = results
        .iter()
        .flat_map(|r| r.gaps_ms.iter().copied())
        .collect();
    itls.sort_by(f64::total_cmp);

    Ok(ConcReport {
        name: name.to_owned(),
        concurrency: n,
        prefix_words,
        total_tokens,
        wall,
        aggregate_tok_s,
        per_session_tok_s_mean,
        ttft_p50_ms: percentile(&ttfts, 50.0),
        ttft_p99_ms: percentile(&ttfts, 99.0),
        itl_p50_ms: percentile(&itls, 50.0),
        itl_p99_ms: percentile(&itls, 99.0),
        errors,
    })
}

impl ConcReport {
    pub fn print(&self) {
        println!(
            "\n=== concurrency benchmark: {} (N={}, shared prefix ~{} words) ===",
            self.name, self.concurrency, self.prefix_words
        );
        if self.errors > 0 {
            println!(
                "  errors: {}/{} sessions failed",
                self.errors, self.concurrency
            );
        }
        println!(
            "  aggregate throughput : {:.1} tok/s  ({} tokens in {:.2}s)",
            self.aggregate_tok_s,
            self.total_tokens,
            self.wall.as_secs_f64()
        );
        println!(
            "  per-session decode   : {:.1} tok/s (mean)",
            self.per_session_tok_s_mean
        );
        println!(
            "  TTFT                 : p50 {:.0} ms | p99 {:.0} ms",
            self.ttft_p50_ms, self.ttft_p99_ms
        );
        println!(
            "  inter-token latency  : p50 {:.1} ms | p99 {:.1} ms",
            self.itl_p50_ms, self.itl_p99_ms
        );
    }
}
