//! Word-level timing from cross-attention DTW, on the real model.
//!
//!   cargo run --release --example whisper_align -- <model.gguf> <clip.wav> [lang]
//!
//! Why AN EXAMPLE AND not A UNIT TEST. `align`'s maths is covered by hand-worked
//! cases that need no GPU; what those cannot reach is whether the CAPTURE is
//! wired to the right thing - right heads, right layer, right query, right row.
//! Every one of those mistakes produces a plausible-looking monotonic path, so
//! the only honest check is a real clip whose words you can hear.
//!
//! What it asserts (the failures that would ship silently otherwise):
//!   - one more boundary than tokens, which is the row off-by-one holding
//!   - boundaries non-decreasing - DTW guarantees it, so a violation means the
//!     rows were transposed or the frames were
//!   - nothing past the end of the audio, which is what clipping the padded
//!     tail is for
//!   - the span actually MOVES: an all-zero or all-final matrix would pass the
//!     three above and be useless
//!
//! What it cannot assert is whether the times are right - read the printed
//! table against the clip for that. Known bias, measured, see
//! whisper's BPE keeps the leading
//! space inside the word token, so starts lean early into the preceding pause.
// A development probe: it runs on a box its author is looking at, and a
// failure should stop it where it happened rather than be reported.
#![allow(clippy::unwrap_used)]

use std::sync::Arc;

use paddock_engine::audio::{PAD_SAMPLES, wav, whisper_features};
use paddock_engine::gpu::{GpuExecutor, KvDtype};
use paddock_engine::gpu_model::whisper::GpuWhisper;
use paddock_models::mapped::MappedGguf;
use paddock_tokenizer::GgufTokenizer;

/// Group tokens into words the way whisper's BPE lays them out: a token whose
/// text starts with a space opens a new word, everything else continues the
/// current one. Deliberately the crude version - the runner has the real
/// grouping (shared with per-word confidence so the two cannot disagree), and
/// this only has to be good enough to read the table against the audio.
fn group_words(pieces: &[String]) -> Vec<(usize, usize)> {
    let mut out: Vec<(usize, usize)> = Vec::new();
    for (i, p) in pieces.iter().enumerate() {
        if out.is_empty() || p.starts_with(' ') {
            out.push((i, i + 1));
        } else {
            out.last_mut().expect("pushed above").1 = i + 1;
        }
    }
    out
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (gguf, wav_path, lang) = match args.as_slice() {
        [g, w] => (g, w, None),
        [g, w, l] => (g, w, Some(l.as_str())),
        _ => {
            eprintln!("usage: whisper_align <model.gguf> <clip.wav> [lang]");
            std::process::exit(2);
        }
    };
    let pack = std::env::var("PADDOCK_PACK").expect("set PADDOCK_PACK");
    // RUST_LOG=debug splits the pass into capture vs post-processing, which is
    // the only way to tell a slow decode from a slow median filter
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let clip = wav::decode_wav(&std::fs::read(wav_path).expect("read wav")).expect("decode wav");
    assert_eq!(clip.sample_rate, 16000, "feed 16 kHz mono");
    // One window. The alignment pass re-decodes into the slot whose cross
    // planes hold the window it is timing, so a multi-window clip would have to
    // re-encode per window - which the serving path will do in phase 4 and this
    // probe has no reason to.
    let samples = &clip.samples[..clip.samples.len().min(PAD_SAMPLES)];
    let dur = samples.len() as f32 / 16000.0;

    let map = MappedGguf::open(gguf.as_ref()).expect("open gguf");
    let tk = GgufTokenizer::from_gguf(map.gguf()).expect("tokenizer");
    let exec = Arc::new(GpuExecutor::new(0, pack.as_ref()).expect("cuda executor"));
    let mut m = GpuWhisper::load(exec, &map, 448).expect("load whisper");
    if let Ok(s) = std::env::var("PADDOCK_KV_CACHE_DTYPE") {
        let d = match s.as_str() {
            "f16" | "fp16" => KvDtype::Fp16,
            "fp8_e4m3" | "fp8" => KvDtype::Fp8E4m3,
            other => panic!("unknown PADDOCK_KV_CACHE_DTYPE {other:?}"),
        };
        m.set_kv_dtype(d);
    }
    m.prepare_batch(1).expect("pool");

    let mel = whisper_features(samples).expect("mel");
    m.encode_into(0, &mel).expect("encode");

    // Plain greedy decode, no timestamp grammar: word timing replaces the
    // `<|0.00|>` segment grid rather than reading it, so this is the ordinary
    // text lane and the tokens are the ordinary tokens.
    let (sot, eot) = m.contract_tokens();
    let (transcribe, no_ts) = m.prompt_tail();
    let ctx = m.text_ctx();
    let slots = [0u32];
    let mut pos = 0u32;
    let step = |m: &mut GpuWhisper, t: u32, p: &mut u32| {
        let out = m.step_batch(&slots, &[t], &[*p], None).expect("step");
        *p += 1;
        out.next[0]
    };

    let mut next = step(&mut m, sot, &mut pos);
    let lang_tok = match lang {
        Some(code) => m.lang_token(code).expect("language not in this checkpoint"),
        None => {
            let row = m.logits_row(0).expect("logits");
            m.detect_language(&row).expect("detect").1
        }
    };
    for t in [lang_tok, transcribe, no_ts] {
        next = step(&mut m, t, &mut pos);
    }
    let mut tokens: Vec<u32> = Vec::new();
    // timed because the honest way to price word timing is against the decode
    // it repeats - this loop is graph-replayed, the alignment pass is not
    let t_dec = std::time::Instant::now();
    while next != eot && pos as usize + 1 < ctx {
        tokens.push(next);
        next = step(&mut m, next, &mut pos);
    }
    let dec_ms = t_dec.elapsed().as_secs_f64() * 1e3;
    let text = tk.decode(&tokens, true).unwrap_or_default();
    println!("clip {dur:.2}s | {} tokens | {text:?}", tokens.len());
    println!(
        "greedy decode: {dec_ms:.1} ms ({:.2} ms/token)",
        dec_ms / tokens.len().max(1) as f64
    );
    if tokens.is_empty() {
        println!("nothing decoded - no timing to recover");
        return;
    }

    let t0 = std::time::Instant::now();
    let b = m
        .token_boundaries(0, lang_tok, &tokens, samples.len())
        .expect("token boundaries");
    let ms = t0.elapsed().as_secs_f64() * 1e3;

    assert_eq!(
        b.len(),
        tokens.len() + 1,
        "n tokens must give n+1 boundaries - the sot/eot row slice is off"
    );
    for w in b.windows(2) {
        assert!(
            w[0] <= w[1],
            "boundaries went backwards: {:?} then {:?}",
            w[0],
            w[1]
        );
    }
    // one frame of slack: the last boundary lands on the final frame index
    assert!(
        *b.last().expect("non-empty") <= dur + 0.02,
        "timing ran past the audio: {:.3}s of a {dur:.3}s clip",
        b.last().expect("non-empty")
    );
    assert!(
        b.last().expect("non-empty") - b[0] > 0.0,
        "every token got the same time - the capture is not reaching the DTW"
    );

    let pieces: Vec<String> = tokens
        .iter()
        .map(|&t| tk.decode(&[t], true).unwrap_or_default())
        .collect();
    println!("\n-- tokens --");
    for (i, p) in pieces.iter().enumerate() {
        println!("  {:>7.2} .. {:>7.2}  {p:?}", b[i], b[i + 1]);
    }
    println!("\n-- words --");
    for (s, e) in group_words(&pieces) {
        println!(
            "  {:>7.2} .. {:>7.2}  {:?}",
            b[s],
            b[e],
            pieces[s..e].concat().trim()
        );
    }
    println!("\nalignment pass: {ms:.1} ms for {} tokens", tokens.len());
}
