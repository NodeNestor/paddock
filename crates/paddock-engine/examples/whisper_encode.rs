//! Whisper encoder probe: decode a WAV, run the mel frontend and
//! the 32-layer encoder, and write the `[1500, 1280]` states as raw f32 -
//! the input side of the oracle gate.
//!
//! Write our states, then diff them against the same clip run through HF
//! `transformers` on the same checkpoint:
//!   cargo run --release --example whisper_encode -- <model.gguf> <clip.wav> ours.f32
//!
//! Clips longer than 30 s are truncated to the first window here - long-form
//! windowing is the serving lane's job.
// A development probe: it runs on a box its author is looking at, and a
// failure should stop it where it happened rather than be reported.
#![allow(clippy::unwrap_used)]

use std::sync::Arc;

use paddock_engine::audio::{self, wav};
use paddock_engine::gpu::GpuExecutor;
use paddock_engine::gpu_model::whisper::GpuWhisper;
use paddock_models::mapped::MappedGguf;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [gguf, wav_path, out] = args.as_slice() else {
        eprintln!("usage: whisper_encode <model.gguf> <clip.wav> <out.f32>");
        std::process::exit(2);
    };
    let pack = std::env::var("PADDOCK_PACK").expect("set PADDOCK_PACK");

    let clip = wav::decode_wav(&std::fs::read(wav_path).expect("read wav")).expect("decode wav");
    assert_eq!(clip.sample_rate, 16000, "feed 16 kHz mono");
    let mel = audio::whisper_features(&clip.samples).expect("mel");

    let exec = Arc::new(GpuExecutor::new(0, pack.as_ref()).expect("cuda executor"));
    let map = MappedGguf::open(gguf.as_ref()).expect("open gguf");
    let mut m = GpuWhisper::load(exec, &map, 448).expect("load whisper");

    let t0 = std::time::Instant::now();
    let enc = m.encode(&mel).expect("encode");
    let host = m.states_to_host(&enc).expect("readback");
    let ms = t0.elapsed().as_secs_f64() * 1e3;

    let (frames, d) = m.encoder_shape();
    let bytes: Vec<u8> = host.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(out, &bytes).expect("write states");
    let mean = host.iter().map(|v| *v as f64).sum::<f64>() / host.len() as f64;
    let absmax = host.iter().fold(0f32, |a, v| a.max(v.abs()));
    println!(
        "encoded {} samples ({:.2}s) -> [{frames}, {d}] in {ms:.1} ms; mean {mean:+.5} absmax {absmax:.3} -> {out}",
        clip.samples.len(),
        clip.samples.len() as f64 / 16000.0,
    );
}
