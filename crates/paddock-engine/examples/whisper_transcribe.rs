//! Whisper end-to-end transcription probe: WAV, transcript
//! out, over the serial bring-up lane - mel -> encoder -> cross-attention
//! precompute -> greedy decode with the checkpoint's own prompt contract.
//!
//!   cargo run --release --example whisper_transcribe -- <model.gguf> <clip.wav> [lang]
//!
//! `lang` is a bare code ("sv"); omit it to let whisper detect the language
//! itself (one step from `<|startoftranscript|>`, argmax over the language
//! tokens - what vLLM's detection path does). Clips longer than 30 s decode
//! as sequential windows.

use std::sync::Arc;

use paddock_engine::audio::wav;
use paddock_engine::gpu::GpuExecutor;
use paddock_engine::gpu_model::whisper::GpuWhisper;
use paddock_models::mapped::MappedGguf;
use paddock_tokenizer::GgufTokenizer;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .init();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (gguf, wav_path, lang) = match args.as_slice() {
        [g, w] => (g, w, None),
        [g, w, l] => (g, w, Some(l.as_str())),
        _ => {
            eprintln!("usage: whisper_transcribe <model.gguf> <clip.wav> [lang]");
            std::process::exit(2);
        }
    };
    let pack = std::env::var("PADDOCK_PACK").expect("set PADDOCK_PACK");

    let clip = wav::decode_wav(&std::fs::read(wav_path).expect("read wav")).expect("decode wav");
    assert_eq!(clip.sample_rate, 16000, "feed 16 kHz mono");

    let map = MappedGguf::open(gguf.as_ref()).expect("open gguf");
    let tok = GgufTokenizer::from_gguf(map.gguf()).expect("tokenizer");
    let exec = Arc::new(GpuExecutor::new(0, pack.as_ref()).expect("cuda executor"));
    let mut m = GpuWhisper::load(exec, &map, 448).expect("load whisper");

    let t0 = std::time::Instant::now();
    let (lang_code, windows) = m.transcribe(&clip.samples, lang, 448).expect("transcribe");
    let wall = t0.elapsed().as_secs_f64();

    let parts: Vec<String> = windows
        .iter()
        .map(|ids| tok.decode(ids, true).expect("decode ids"))
        .collect();
    let text = paddock_engine::gpu_model::whisper::join_windows(&parts, &lang_code);
    let n_tok: usize = windows.iter().map(Vec::len).sum();
    let audio_s = clip.samples.len() as f64 / 16000.0;
    println!(
        "\nlang={lang_code} windows={} tokens={n_tok}",
        windows.len()
    );
    println!(
        "{wall:.2}s wall for {audio_s:.2}s audio (RTFx {:.1}x, {:.1} tok/s)\n",
        audio_s / wall,
        n_tok as f64 / wall
    );
    if parts.len() > 1 {
        // quoted, per window: the join seam is where long-form transcripts
        // actually differ between engines (whisper emits each window with a
        // leading space and the boundary is invisible in flat output)
        for (i, p) in parts.iter().enumerate() {
            println!("  window {i}: {p:?}");
        }
        println!();
    }
    println!("{text}");
}
