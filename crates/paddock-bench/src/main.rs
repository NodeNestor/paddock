//! `paddock-bench` - throughput/TTFT benchmark.
//!
//! Two modes:
//!  - in-process: benchmark our engine directly (needs a GGUF + kernel pack).
//!  - HTTP (--endpoint URL): benchmark any OpenAI-compatible server the same
//!    way - point it at Paddock's own server AND at mistral.rs / llama.cpp /
//!    vLLM for a like-for-like comparison. Competitors stay external.

mod concurrency_runner;
mod http_runner;
mod paddock_runner;
mod timings;

use clap::Parser;
use timings::Timings;

#[derive(Parser)]
#[command(name = "paddock-bench", about = "Throughput/TTFT benchmark")]
struct Args {
    /// In-process: GGUF model to benchmark with our engine.
    #[arg(long)]
    model: Option<std::path::PathBuf>,
    /// Kernel pack (in-process cuda).
    #[arg(long)]
    pack: Option<std::path::PathBuf>,
    /// Compute device (in-process); only "cuda" exists.
    #[arg(long, default_value = "cuda")]
    device: String,

    /// HTTP: base URL of an OpenAI-compatible server, e.g. http://127.0.0.1:1234/v1.
    /// Repeatable - pass several to compare servers in one run.
    #[arg(long)]
    endpoint: Vec<String>,
    /// Label(s) for the endpoint(s), aligned by position (default: the URL).
    #[arg(long)]
    name: Vec<String>,
    /// Model id sent to the endpoint(s).
    #[arg(long, default_value = "default")]
    endpoint_model: String,
    /// Prompt text for HTTP mode.
    #[arg(long, default_value = "The three laws of robotics are")]
    prompt: String,

    /// Prompt tokens (in-process synthetic prompt).
    #[arg(long, default_value_t = 128)]
    prompt_tokens: usize,
    /// Tokens to decode (timed).
    #[arg(long, default_value_t = 64)]
    decode_tokens: usize,
    /// Warmup decode tokens discarded (in-process).
    #[arg(long, default_value_t = 8)]
    warmup: usize,

    /// Concurrency: fire this many streaming sessions at once at each
    /// --endpoint. 0 = the normal single-shot HTTP/in-process modes.
    #[arg(long, default_value_t = 0)]
    concurrency: usize,
    /// Length of the shared prompt prefix (words) for the concurrency benchmark.
    #[arg(long, default_value_t = 400)]
    shared_prefix_words: usize,
    /// Stagger session arrivals by this many ms each (0 = fire together). Makes
    /// later sessions join while earlier ones decode - the chunked-prefill stress.
    #[arg(long, default_value_t = 0)]
    stagger_ms: u64,
    /// Give each session a DISTINCT prefix (cold prefill, no cache sharing).
    #[arg(long, default_value_t = false)]
    unique_prefix: bool,
}

fn main() -> std::process::ExitCode {
    // engine diagnostics ride tracing - without a subscriber
    // they vanish; default to warn+ so product output stays clean
    // (RUST_LOG=info shows the engine's operational lines)
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "warn".into()),
        )
        .try_init();
    let args = Args::parse();
    let mut results: Vec<Timings> = Vec::new();

    if let Some(model) = &args.model {
        let prompt: Vec<u32> = vec![0u32; args.prompt_tokens.max(1)];
        match paddock_runner::run(
            model,
            &args.device,
            args.pack.as_deref(),
            &prompt,
            args.decode_tokens,
            args.warmup,
        ) {
            Ok(t) => results.push(t),
            Err(e) => eprintln!("in-process runner failed: {e}"),
        }
    }

    // Concurrency mode: fire N sessions at each endpoint and report
    // aggregate throughput + latency distributions instead of the single-shot row.
    if args.concurrency > 0 {
        for (i, url) in args.endpoint.iter().enumerate() {
            let name = args.name.get(i).cloned().unwrap_or_else(|| url.clone());
            match concurrency_runner::run(
                &name,
                url,
                &args.endpoint_model,
                args.concurrency,
                args.shared_prefix_words,
                args.decode_tokens,
                args.stagger_ms,
                args.unique_prefix,
            ) {
                Ok(report) => report.print(),
                Err(e) => eprintln!("endpoint {url} concurrency run failed: {e}"),
            }
        }
        return std::process::ExitCode::SUCCESS;
    }

    for (i, url) in args.endpoint.iter().enumerate() {
        let name = args.name.get(i).cloned().unwrap_or_else(|| url.clone());
        match http_runner::run(
            &name,
            url,
            &args.endpoint_model,
            &args.prompt,
            args.decode_tokens,
        ) {
            Ok(t) => results.push(t),
            Err(e) => eprintln!("endpoint {url} failed: {e}"),
        }
    }

    if results.is_empty() {
        eprintln!("nothing to benchmark: pass --model <gguf> and/or --endpoint <url>");
        return std::process::ExitCode::FAILURE;
    }

    println!("\n{}", Timings::header());
    for t in &results {
        println!("{}", t.row());
    }
    if results.len() > 1 {
        let base = &results[0];
        for t in &results[1..] {
            let ratio = base.decode_tok_s() / t.decode_tok_s().max(1e-9);
            println!("\n{} decode vs {}: {:.2}x", base.runner, t.runner, ratio);
        }
    }
    std::process::ExitCode::SUCCESS
}
