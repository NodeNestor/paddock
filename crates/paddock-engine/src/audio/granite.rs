//! Granite Speech host frontend: 16 kHz PCM -> the conformer encoder's
//! 160-wide stacked log-mel frames. Pure host preprocessing (no device
//! buffers), directly testable, same role `preprocess.rs` plays for images.
//!
//! This is a different extractor contract from the whisper/Qwen3-ASR one in
//! the parent module - it shares the FFT and the frame loop and nothing else,
//! which is why it lives in its own file rather than as another `MelPolicy`
//! arm. The contract is transformers' `GraniteSpeechFeatureExtractor`, which
//! is a thin wrapper over `torchaudio.transforms.MelSpectrogram`, so every
//! default below is torchaudio's, not librosa's:
//!   - n_fft 512 with a **400-sample** periodic Hann window, zero-padded and
//!     CENTERED inside the 512-point transform (torch.stft's own rule);
//!   - `center=True` + `pad_mode="reflect"`: the clip is reflected n_fft/2 =
//!     256 samples at each end, so frame t is centered on sample t*160 and
//!     there are exactly `len/160 + 1` of them;
//!   - **HTK** mel scale, **no** Slaney area normalization, fmax = sr/2 =
//!     8000, power (not magnitude) spectrum over 257 bins;
//!   - clamp 1e-10, log10, GLOBAL max over the whole plane, floor at max-8,
//!     then (x+4)/4 - the same normalization the other families use;
//!   - drop a trailing ODD frame, then stack frame pairs along the feature
//!     axis: row t is `[mel(2t) | mel(2t+1)]`, 160 wide at 20 ms/frame.
//!
//! No 30 s window anywhere: granite-speech consumes a whole clip in one
//! encoder pass (its attention is blockwise over 200-frame chunks), and the
//! encoder frame rate is 50/s, which the projector's 15-frame windows cut to
//! **10 audio tokens per second of speech**.
//!
//! llama.cpp b10330's `mtmd_audio_preprocessor_granite_speech` implements the
//! same contract (checked step by step); it differs only in clamping the
//! reflection index for clips shorter than the pad, where torch would raise.
//! We clamp too - a 100 ms clip should transcribe, not 500.

use super::MelFeatures;
use super::dsp::{self, MelScale};

pub const SAMPLE_RATE: usize = 16000;
pub const N_FFT: usize = 512;
/// Analysis window; shorter than the transform, centered inside it.
pub const WIN_LENGTH: usize = 400;
pub const HOP: usize = 160;
pub const N_MEL: usize = 80;
/// Encoder input width: two consecutive mel frames stacked.
pub const INPUT_DIM: usize = N_MEL * 2;
/// Projector window in encoder frames, and how many queries it emits.
pub const WINDOW_SIZE: usize = 15;
pub const DOWNSAMPLE_RATE: usize = 5;
pub const NUM_QUERIES: usize = WINDOW_SIZE / DOWNSAMPLE_RATE;

/// Mel frames for `len` samples: `center=True` gives one frame per hop plus
/// the one centered on sample 0.
pub fn mel_frames(len: usize) -> usize {
    len / HOP + 1
}

/// Encoder frames for `len` samples - mel frames pairwise stacked, trailing
/// odd frame dropped. Matches upstream `_get_num_audio_features`.
pub fn encoder_frames(len: usize) -> usize {
    mel_frames(len) / 2
}

/// Audio tokens the projector emits for `frames` encoder frames: the plane is
/// padded up to whole 15-frame windows and each window becomes 3 queries.
pub fn audio_token_count(frames: usize) -> usize {
    frames.div_ceil(WINDOW_SIZE) * NUM_QUERIES
}

/// Tokens the prompt must reserve at the `<|audio|>` position for a clip of
/// `len` samples - 10 per second, rounded up to the 300 ms window.
pub fn audio_tokens_for_samples(len: usize) -> usize {
    audio_token_count(encoder_frames(len))
}

/// Extract the encoder's input features for one 16 kHz mono clip.
///
/// The plane rides the parent module's [`MelFeatures`] - the same container
/// whisper and Qwen3-ASR fill - so one `MmChunk::Audio` type carries every
/// family's frontend output to the engine. Only the geometry differs: rows
/// here are [`INPUT_DIM`] (160) wide at 50 frames/s, not `N_MEL` at 100/s,
/// and the tower checks that width at encode time.
pub fn speech_features(samples: &[f32]) -> Result<MelFeatures, String> {
    if samples.is_empty() {
        return Err("empty audio".into());
    }
    let len = samples.len();
    let n_mel_frames = mel_frames(len);
    let half = N_FFT / 2;

    // torch.stft pads a short window to n_fft symmetrically, so the analysis
    // window sits in the middle of the transform with zeros either side.
    let mut window = vec![0.0f64; N_FFT];
    let lead = (N_FFT - WIN_LENGTH) / 2;
    window[lead..lead + WIN_LENGTH].copy_from_slice(&dsp::hann_periodic(WIN_LENGTH));
    let fb = dsp::mel_filterbank(
        N_MEL,
        N_FFT,
        SAMPLE_RATE as f64,
        0.0,
        (SAMPLE_RATE / 2) as f64,
        MelScale::Htk,
        false,
    );
    let plan = dsp::FftPlan::new(N_FFT);

    // Reflection at both ends, folded repeatedly so clips shorter than the
    // 256-sample pad still resolve (torch raises there; llama.cpp clamps).
    let at = |i: i64| -> f64 {
        let n = len as i64;
        if n == 1 {
            return samples[0] as f64;
        }
        let mut i = i;
        loop {
            if i < 0 {
                i = -i;
            } else if i >= n {
                i = 2 * n - 2 - i;
            } else {
                return samples[i as usize] as f64;
            }
        }
    };

    // Frames are independent; fan out over host threads exactly as the
    // qwen3-asr/whisper loop does. Bit-identical to the serial version -
    // per-frame arithmetic is self-contained and the max is order-insensitive.
    let mut mel = vec![0.0f64; n_mel_frames * N_MEL];
    let threads = std::thread::available_parallelism()
        .map_or(1, std::num::NonZeroUsize::get)
        .min(n_mel_frames.div_ceil(64))
        .max(1);
    let chunk_t = n_mel_frames.div_ceil(threads);
    let gmax = std::thread::scope(|s| {
        let mut handles = Vec::new();
        for (ci, out) in mel.chunks_mut(chunk_t * N_MEL).enumerate() {
            let (window, fb, plan, at) = (&window, &fb, &plan, &at);
            handles.push(s.spawn(move || {
                let mut frame = vec![0.0f64; N_FFT];
                let mut local_max = f64::NEG_INFINITY;
                for (j, row) in out.chunks_mut(N_MEL).enumerate() {
                    let base = ((ci * chunk_t + j) * HOP) as i64 - half as i64;
                    for (i, f) in frame.iter_mut().enumerate() {
                        *f = at(base + i as i64);
                    }
                    dsp::log_mel_frame(&frame, window, fb, plan, row);
                    for &v in row.iter() {
                        if v > local_max {
                            local_max = v;
                        }
                    }
                }
                local_max
            }));
        }
        handles
            .into_iter()
            .map(|h| h.join().expect("mel frame worker"))
            .fold(f64::NEG_INFINITY, f64::max)
    });

    // The max is taken over the full plane, before the odd frame is dropped -
    // upstream normalizes first and slices second, and a loud final frame
    // would otherwise shift every value in the clip.
    let floor = gmax - 8.0;
    let n_frames = n_mel_frames / 2;
    let mut data = vec![0.0f32; n_frames * INPUT_DIM];
    for (t, row) in data.chunks_mut(INPUT_DIM).enumerate() {
        for (h, half_row) in row.chunks_mut(N_MEL).enumerate() {
            let src = &mel[(2 * t + h) * N_MEL..(2 * t + h + 1) * N_MEL];
            for (d, &v) in half_row.iter_mut().zip(src) {
                *d = ((v.max(floor) + 4.0) / 4.0) as f32;
            }
        }
    }
    Ok(MelFeatures {
        data,
        n_frames,
        n_samples: len,
        global_max: gmax as f32,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_and_token_rules_match_upstream() {
        // (samples, encoder frames, audio tokens) - the upstream formula is
        // mel = L/160 + 1, enc = mel/2, tokens = ceil(enc/15) * 3
        for (len, frames, tokens) in [
            (16000, 50, 12),     // 1 s -> 10 tokens/s rounded up to the window
            (32000, 100, 21),    // 2 s
            (52327, 164, 33),    // not hop-divisible
            (480000, 1500, 300), // 30 s -> exactly 100 windows
            (4800, 15, 3),       // 0.3 s -> exactly one window
            (160, 1, 3),         // one hop: a single frame still costs a window
        ] {
            assert_eq!(encoder_frames(len), frames, "len={len}");
            assert_eq!(audio_tokens_for_samples(len), tokens, "len={len}");
        }
    }

    #[test]
    fn short_clips_survive_the_reflection() {
        // torch would raise (pad 256 >= len); we fold instead, and the plane
        // must still come out the declared shape
        let f = speech_features(&[0.1f32; 100]).unwrap();
        assert_eq!(f.n_frames, encoder_frames(100));
        assert_eq!(f.data.len(), f.n_frames * INPUT_DIM);
    }

    #[test]
    fn stacking_interleaves_consecutive_frames() {
        // a 1 s ramp: row t's two halves must be the mel of frames 2t/2t+1,
        // which for a smooth signal means they differ but stay close
        let sig: Vec<f32> = (0..16000).map(|i| (i as f32 * 0.01).sin() * 0.5).collect();
        let f = speech_features(&sig).unwrap();
        assert_eq!(f.n_frames, 50);
        assert_eq!(f.data.len(), 50 * 160);
        for t in 0..50 {
            let row = &f.data[t * INPUT_DIM..(t + 1) * INPUT_DIM];
            assert_ne!(row[..N_MEL], row[N_MEL..], "row {t} halves are identical");
        }
    }
}
