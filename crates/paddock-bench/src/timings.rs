//! Benchmark measurement: prefill/decode throughput + TTFT, reported the same
//! way for every runner so cross-engine numbers are comparable.

use std::time::Duration;

#[derive(Debug, Clone)]
pub struct Timings {
    pub runner: String,
    pub load: Duration,
    pub prefill_tokens: usize,
    pub prefill: Duration,
    /// time from submit to the first generated token (prefill + one decode).
    pub ttft: Duration,
    pub decode_tokens: usize,
    pub decode: Duration,
}

impl Timings {
    pub fn prefill_tok_s(&self) -> f64 {
        self.prefill_tokens as f64 / self.prefill.as_secs_f64().max(1e-9)
    }
    pub fn decode_tok_s(&self) -> f64 {
        self.decode_tokens as f64 / self.decode.as_secs_f64().max(1e-9)
    }

    pub fn header() -> String {
        format!(
            "{:<22} {:>8} {:>12} {:>10} {:>12}",
            "runner", "load(s)", "prefill tk/s", "ttft(ms)", "decode tk/s"
        )
    }

    pub fn row(&self) -> String {
        // prefill isn't client-observable over HTTP; show '-' then
        let prefill = if self.prefill_tokens == 0 {
            format!("{:>12}", "-")
        } else {
            format!("{:>12.1}", self.prefill_tok_s())
        };
        let load = if self.load.is_zero() {
            format!("{:>8}", "-")
        } else {
            format!("{:>8.1}", self.load.as_secs_f64())
        };
        format!(
            "{:<22} {} {} {:>10.1} {:>12.2}",
            self.runner,
            load,
            prefill,
            self.ttft.as_secs_f64() * 1000.0,
            self.decode_tok_s(),
        )
    }
}
