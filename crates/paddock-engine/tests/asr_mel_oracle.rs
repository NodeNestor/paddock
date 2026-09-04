//! Numeric gate for the ASR mel frontend: the Rust pipeline
//! must reproduce transformers' Qwen3ASRFeatureExtractor - the checkpoint's
//! declared extractor and the TRUE-correctness oracle - on the committed
//! battery in tests/data/asr-mel (regenerate with
//! Our ASR oracle tool; see that script for the pinned
//! semantics). Host-only, no GPU.
//!
//! Tolerance: the oracle runs torch.stft in f32 while our pipeline
//! accumulates in f64, so observed deltas are the oracle's own rounding.
//! The gate is set ~10x above the observed max delta; a structural mistake
//! (wrong window, wrong filterbank, off-by-one frame) moves values by
//! orders of magnitude more.
// Test code: a failed assumption stops the test where it happened.
#![allow(clippy::unwrap_used)]

use paddock_engine::audio::{self, wav};

const BATTERY: [&str; 8] = [
    "tone-2s",
    "chirp",
    "noise-1s",
    "am-12.5s",
    "tiny-0.3s",
    "exact-30s",
    "over-30s",
    "silence-1s",
];

struct Meta {
    samples: usize,
    real_frames: usize,
    stored_frames: usize,
    global_max: f32,
    audio_tokens: usize,
}

fn meta(dir: &std::path::Path, name: &str) -> Meta {
    let txt = std::fs::read_to_string(dir.join(format!("{name}.json"))).unwrap();
    let g = |k: &str| -> f64 {
        let pat = format!("\"{k}\":");
        let s = &txt[txt.find(&pat).unwrap() + pat.len()..];
        s[..s.find([',', '}']).unwrap()].trim().parse().unwrap()
    };
    Meta {
        samples: g("samples") as usize,
        real_frames: g("real_frames") as usize,
        stored_frames: g("stored_frames") as usize,
        global_max: g("global_max") as f32,
        audio_tokens: g("audio_tokens") as usize,
    }
}

#[test]
fn asr_mel_matches_oracle() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/asr-mel");
    for name in BATTERY {
        let m = meta(&dir, name);
        let w = wav::decode_wav(&std::fs::read(dir.join(format!("{name}.wav"))).unwrap()).unwrap();
        assert_eq!(w.sample_rate, 16000, "{name}");
        assert_eq!(w.samples.len(), m.samples, "{name}: sample count");

        assert_eq!(
            audio::real_frames(w.samples.len()),
            m.real_frames,
            "{name}: frame rule"
        );
        assert_eq!(
            audio::audio_tokens_for_samples(w.samples.len()),
            m.audio_tokens,
            "{name}: token count"
        );

        let feat = audio::qwen3_asr_features(&w.samples).unwrap();
        assert_eq!(feat.n_frames, m.real_frames, "{name}: extracted frames");
        assert!(
            (feat.global_max - m.global_max).abs() < 2e-4,
            "{name}: global max {} vs oracle {}",
            feat.global_max,
            m.global_max
        );

        // oracle file is frame-major [stored_frames][128], stored through the
        // 100-frame chunk boundary the conv stem consumes - the pad region
        // (silence log-mel, boundary bleed frames, >30s feature zeros) is
        // gated too, since the encoder reads it (a zero-fill there was a real bug)
        let oracle = std::fs::read(dir.join(format!("{name}.mel.f32"))).unwrap();
        assert_eq!(
            oracle.len(),
            m.stored_frames * 128 * 4,
            "{name}: oracle size"
        );
        let ov: Vec<f32> = oracle
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        let our_frames = feat.data.len() / 128;
        assert_eq!(
            our_frames,
            m.real_frames.div_ceil(100) * 100,
            "{name}: chunk-aligned frame count"
        );
        let n = our_frames.min(m.stored_frames) * 128;
        let (mut max_d, mut sum_d) = (0f32, 0f64);
        let mut arg = 0usize;
        for (i, (&ours, &theirs)) in feat.data[..n].iter().zip(&ov[..n]).enumerate() {
            let d = (ours - theirs).abs();
            if d > max_d {
                max_d = d;
                arg = i;
            }
            sum_d += d as f64;
        }
        let mean_d = sum_d / n as f64;
        eprintln!(
            "{name}: max|d|={max_d:.3e} @ frame {} mel {}, mean|d|={mean_d:.3e}",
            arg / 128,
            arg % 128
        );
        assert!(
            max_d < 5e-4,
            "{name}: max delta {max_d:.3e} at frame {} mel {} (ours {} oracle {})",
            arg / 128,
            arg % 128,
            feat.data[arg],
            ov[arg]
        );
        assert!(mean_d < 5e-5, "{name}: mean delta {mean_d:.3e}");
    }
}

/// The Granite Speech arm, gated against transformers'
/// `GraniteSpeechFeatureExtractor` (regenerate with
/// our ASR oracle tool).
///
/// This one shares no framing with the other two - 512-point transform, a
/// 400-sample window centered inside it, 80 HTK mels with no area norm, and
/// frame-pair stacking - so it is a genuinely independent check of the shared
/// `dsp` layer rather than a rerun of the same path. There is no padding
/// region and no fixed window, so the whole plane is the encoder's input and
/// the whole plane is compared.
#[test]
fn granite_mel_matches_oracle() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/asr-mel");
    for name in BATTERY {
        let txt = std::fs::read_to_string(dir.join(format!("{name}.granite.json"))).unwrap();
        let g = |k: &str| -> f64 {
            let pat = format!("\"{k}\":");
            let s = &txt[txt.find(&pat).unwrap() + pat.len()..];
            s[..s.find([',', '}']).unwrap()].trim().parse().unwrap()
        };
        let (frames, tokens) = (g("frames") as usize, g("audio_tokens") as usize);
        let gmax = g("global_max") as f32;

        let w = wav::decode_wav(&std::fs::read(dir.join(format!("{name}.wav"))).unwrap()).unwrap();
        assert_eq!(
            audio::granite::encoder_frames(w.samples.len()),
            frames,
            "{name}: frame rule"
        );
        assert_eq!(
            audio::granite::audio_tokens_for_samples(w.samples.len()),
            tokens,
            "{name}: token count"
        );

        let feat = audio::granite::speech_features(&w.samples).unwrap();
        assert_eq!(feat.n_frames, frames, "{name}: extracted frames");
        assert_eq!(
            feat.data.len(),
            frames * audio::granite::INPUT_DIM,
            "{name}: plane size"
        );
        assert!(
            (feat.global_max - gmax).abs() < 2e-4,
            "{name}: global max {} vs oracle {gmax}",
            feat.global_max
        );

        let oracle = std::fs::read(dir.join(format!("{name}.granite.f32"))).unwrap();
        assert_eq!(oracle.len(), feat.data.len() * 4, "{name}: oracle size");
        let (mut max_d, mut sum_d, mut arg) = (0f32, 0f64, 0usize);
        for (i, c) in oracle.as_chunks::<4>().0.iter().enumerate() {
            let d = (feat.data[i] - f32::from_le_bytes(*c)).abs();
            if d > max_d {
                max_d = d;
                arg = i;
            }
            sum_d += d as f64;
        }
        let mean_d = sum_d / feat.data.len() as f64;
        eprintln!(
            "{name}: granite {frames} frames max|d|={max_d:.3e} @ frame {} mel {}, mean|d|={mean_d:.3e}",
            arg / audio::granite::INPUT_DIM,
            arg % audio::granite::INPUT_DIM
        );
        assert!(
            max_d < 5e-4,
            "{name}: max delta {max_d:.3e} at index {arg} (ours {})",
            feat.data[arg]
        );
        assert!(mean_d < 5e-5, "{name}: mean delta {mean_d:.3e}");
    }
}

/// The whisper arm of the same frontend: the two extractors
/// share every DSP step and differ only in framing, so this gates that the
/// shared code really does reproduce transformers' `WhisperFeatureExtractor`
/// - the fixed 30 s window, the truncate/zero-pad rule, and (the seam that
///   bit us on Qwen3-ASR) the silence-constant tail every short clip carries.
///
/// The committed oracle stores the frames that can differ plus the tail
/// constant, which the dump script verified holds for the whole tail; the
/// test checks both halves.
#[test]
fn whisper_mel_matches_oracle() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/asr-mel");
    for name in BATTERY {
        let txt = std::fs::read_to_string(dir.join(format!("{name}.whisper.json"))).unwrap();
        let g = |k: &str| -> f64 {
            let pat = format!("\"{k}\":");
            let s = &txt[txt.find(&pat).unwrap() + pat.len()..];
            s[..s.find([',', '}']).unwrap()].trim().parse().unwrap()
        };
        let (frames, head) = (g("frames") as usize, g("head_frames") as usize);
        let (gmax, tail_value) = (g("global_max") as f32, g("tail_value") as f32);

        let w = wav::decode_wav(&std::fs::read(dir.join(format!("{name}.wav"))).unwrap()).unwrap();
        let feat = audio::whisper_features(&w.samples).unwrap();
        // the encoder is fixed-size: every window is 3000 frames, all of
        // them consumed (whisper has no audio-side attention mask)
        assert_eq!(feat.n_frames, frames, "{name}: window frames");
        assert_eq!(feat.data.len(), frames * 128, "{name}: plane size");
        assert!(
            (feat.global_max - gmax).abs() < 2e-4,
            "{name}: global max {} vs oracle {gmax}",
            feat.global_max
        );

        let oracle = std::fs::read(dir.join(format!("{name}.whisper.f32"))).unwrap();
        assert_eq!(oracle.len(), head * 128 * 4, "{name}: oracle size");
        let ov: Vec<f32> = oracle
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        let (mut max_d, mut sum_d, mut arg) = (0f32, 0f64, 0usize);
        for (i, &theirs) in ov.iter().enumerate() {
            let d = (feat.data[i] - theirs).abs();
            if d > max_d {
                max_d = d;
                arg = i;
            }
            sum_d += d as f64;
        }
        let mean_d = sum_d / ov.len() as f64;
        eprintln!(
            "{name}: whisper head {head}/{frames} max|d|={max_d:.3e} @ frame {} mel {}, mean|d|={mean_d:.3e}",
            arg / 128,
            arg % 128
        );
        assert!(
            max_d < 5e-4,
            "{name}: max delta {max_d:.3e} at frame {} mel {} (ours {} oracle {})",
            arg / 128,
            arg % 128,
            feat.data[arg],
            ov[arg]
        );
        assert!(mean_d < 5e-5, "{name}: mean delta {mean_d:.3e}");

        // the silence tail is real encoder input, not masked padding - a
        // zero-fill here is exactly the class of bug this oracle caught before
        for (i, &v) in feat.data.iter().enumerate().skip(head * 128) {
            assert!(
                (v - tail_value).abs() < 5e-4,
                "{name}: tail frame {} mel {} is {v}, want the silence constant {tail_value}",
                i / 128,
                i % 128
            );
        }
    }
}
