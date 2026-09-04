//! Granite Speech tower probe: decode a WAV, run the host mel
//! frontend, and push it through the conformer encoder + Q-Former projector,
//! writing the `[n_tokens, 2048]` LLM-space audio embeddings as raw f32 -
//! the input side of the oracle gate.
//!
//! Only the mmproj is loaded; the decoder half is the stock granite family
//! and has nothing to do with this measurement.
//!
//!   cargo run --release --example granite_speech_encode -- \
//!       <mmproj.gguf> <clip.wav> ours.f32
//!   python our ASR oracle tool \
//!       --model /models/granite-speech-4.1-2b --wav <clip.wav> \
//!       --ours ours.f32
// A development probe: it runs on a box its author is looking at, and a
// failure should stop it where it happened rather than be reported.
#![allow(clippy::unwrap_used)]

use std::sync::Arc;

use paddock_engine::audio::{granite as mel, wav};
use paddock_engine::gpu::GpuExecutor;
use paddock_engine::gpu_model::granite::audio::SpeechTower;
use paddock_models::mapped::MappedGguf;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [mmproj, wav_path, out] = args.as_slice() else {
        eprintln!("usage: granite_speech_encode <mmproj.gguf> <clip.wav> <out.f32>");
        std::process::exit(2);
    };
    let pack = std::env::var("PADDOCK_PACK").expect("set PADDOCK_PACK");

    let clip = wav::decode_wav(&std::fs::read(wav_path).expect("read wav")).expect("decode wav");
    assert_eq!(clip.sample_rate, 16000, "feed 16 kHz mono");
    let feats = mel::speech_features(&clip.samples).expect("mel");

    let exec = Arc::new(GpuExecutor::new(0, pack.as_ref()).expect("cuda executor"));
    let map = MappedGguf::open(mmproj.as_ref()).expect("open mmproj");
    let mut tower = SpeechTower::load(exec.clone(), &map).expect("load tower");

    // one warm pass so the number below is steady state, not first-touch
    let _ = tower.encode(&feats).expect("warm encode");
    exec.synchronize().expect("sync");
    let t0 = std::time::Instant::now();
    let enc = tower.encode(&feats).expect("encode");
    exec.synchronize().expect("sync");
    let ms = t0.elapsed().as_secs_f64() * 1e3;

    let host = exec
        .to_host_len(&enc.embd, enc.n_tokens * tower.out_dim)
        .expect("readback");
    let bytes: Vec<u8> = host.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(out, &bytes).expect("write embeddings");
    let mean = host.iter().map(|v| *v as f64).sum::<f64>() / host.len() as f64;
    let absmax = host.iter().fold(0f32, |a, v| a.max(v.abs()));
    println!(
        "encoded {} samples ({:.2}s) -> {} frames -> [{}, {}] in {ms:.1} ms; \
         mean {mean:+.5} absmax {absmax:.3} -> {out}",
        clip.samples.len(),
        clip.samples.len() as f64 / 16000.0,
        feats.n_frames,
        enc.n_tokens,
        tower.out_dim,
    );
}
