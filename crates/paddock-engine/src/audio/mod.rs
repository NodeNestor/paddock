//! Host audio frontend (mel spectrograms + WAV ingestion) for the ASR
//! families. Pure host preprocessing - the same role granite's
//! `preprocess.rs` plays for images; no device buffers, directly testable.
//!
//! Two contracts live here. This file carries the Whisper-derived one that
//! Qwen3-ASR and whisper share (400-point transform, 128 Slaney mels, 30 s
//! framing); Granite Speech's torchaudio-derived one (512-point transform,
//! 80 HTK mels, frame stacking, no fixed window) is in `granite.rs`. They
//! share `dsp` and nothing else.
//!
//! The Qwen3-ASR pipeline implements the upstream extractor contract
//! (transformers 5.14.1 `Qwen3ASRFeatureExtractor`, which vLLM also serves):
//!   - 16 kHz mono f32 in; clips shorter than 0.5 s (8000 samples) are
//!     zero-extended to 0.5 s and the extension counts as real audio
//!     (upstream keeps the mask unadjusted deliberately);
//!   - STFT center=True semantics over the clip zero-padded to
//!     max(len, 30 s): reflection continuation at the start, zeros at the
//!     end (reflection only when the clip itself is >= 30 s), Hann 400
//!     periodic, hop 160, power spectrum over 201 bins;
//!   - Slaney-scale + Slaney-area-norm 128-bin filterbank (fmax 8000),
//!     f64 accumulation, floor 1e-10, log10;
//!   - GLOBAL max over every frame with signal energy, floor at max-8,
//!     then (v + 4) / 4;
//!   - real frame count R = min(ceil(len/160), padded_len/160) - the
//!     attention-mask downsampling rule, including its drop-last edge for
//!     >30 s clips whose length is not hop-divisible.
//!
//! NOTE: llama.cpp b10327 diverges from upstream here - its center padding
//! adds one extra frame and its window split zero-pads to a multiple of 100
//! frames with those frames encoded as real audio tokens (a 2 s clip
//! becomes 39 tokens where upstream produces 26; `n_len_org` is never
//! consumed downstream). paddock serves the upstream contract, so the
//! correctness gate against llama-server is transcript-level greedy parity,
//! while numeric parity is gated against the transformers oracle
//! (tests/data/asr-mel, regenerated out of tree).

pub mod decode;
pub mod dsp;
pub mod granite;
pub mod guards;
pub mod resample;
pub mod vad;
pub mod wav;
// A shim `decode` reaches for and nobody else should - see its own header.
mod webm_live;

/// Qwen3-ASR mel geometry (from the checkpoint's preprocessor_config.json).
pub const SAMPLE_RATE: usize = 16000;
pub const N_FFT: usize = 400;
pub const HOP: usize = 160;
pub const N_MEL: usize = 128;
/// Upstream zero-extends anything shorter than this many samples (0.5 s).
pub const MIN_SAMPLES: usize = 8000;
/// Raw audio is zero-padded to at least this many samples (30 s) before the
/// STFT - the padding only matters at the clip boundary (last ~2 frames).
pub const PAD_SAMPLES: usize = 480000;
/// Mel frames per encoder forward (8 s window; windows never attend across).
pub const WINDOW_FRAMES: usize = 800;
/// Conv chunk: 100 mel frames -> up to 13 audio tokens.
pub const CHUNK_FRAMES: usize = 100;
/// Whisper's encoder is fixed at 30 s: one window is always this many mel
/// frames (`PAD_SAMPLES / HOP`), which the conv stem halves to 1500 encoder
/// positions.
pub const WHISPER_FRAMES: usize = PAD_SAMPLES / HOP;

/// Which extractor contract to run.
///
/// The two ASR families share all of the DSP geometry - 16 kHz, n_fft 400,
/// hop 160, 128 Slaney-scale/Slaney-norm mels to 8 kHz, log10 with the
/// global max-8 floor and the (x+4)/4 normalization - because Qwen3-ASR's
/// extractor is itself Whisper-derived. They differ only in how a clip is
/// framed, which is what this picks; everything downstream is shared code.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MelPolicy {
    /// Qwen3-ASR (transformers `Qwen3ASRFeatureExtractor`): clips under
    /// 0.5 s zero-extend, the STFT runs over the clip padded to
    /// max(len, 30 s), every real frame is kept, and the emitted plane
    /// extends to the conv stem's 100-frame chunk boundary.
    Qwen3Asr,
    /// Whisper (transformers `WhisperFeatureExtractor`): the encoder is
    /// FIXED at 30 s, so one call is one window - the clip is truncated or
    /// zero-padded to exactly 480000 samples and always yields 3000 frames.
    /// Trailing silence is real encoder input here, not padding to be
    /// masked (whisper has no attention mask on the audio side at all), so
    /// `n_frames` is the full window; long form is split into windows by
    /// the caller.
    Whisper,
}

/// Normalized log-mel features for one clip, frame-major `[frames][128]`.
/// `data` holds `n_frames` real frames extended to the 100-frame chunk
/// boundary the conv stem consumes (`data.len() / N_MEL` frames total); the
/// extension carries the upstream pad semantics (silence log-mel within the
/// 30 s audio pad, literal zeros past the stored spectrogram). Token count
/// and attention masking derive from `n_frames` alone.
#[derive(Clone, Debug)]
pub struct MelFeatures {
    pub data: Vec<f32>,
    pub n_frames: usize,
    /// How many input samples this block actually covers.
    ///
    /// Not derivable from `n_frames` on the whisper contract: its encoder is
    /// fixed-size, so `n_frames` is always the full 30 s and a 4 s tail window
    /// looks exactly like a full one. Word timing needs the real length to know
    /// where the audio stops and the zero padding begins (`timing::used_frames`),
    /// so the count travels with the features rather than beside them - a
    /// parallel vector is one reorder away from timing the wrong window.
    pub n_samples: usize,
    /// Un-normalized global log-mel max (diagnostic; the norm already
    /// applied it).
    pub global_max: f32,
}

/// Number of real mel frames the model consumes for a clip of `len` samples
/// (after the upstream min-length extension).
pub fn real_frames(len: usize) -> usize {
    let l = len.max(MIN_SAMPLES);
    let padded = l.max(PAD_SAMPLES);
    l.div_ceil(HOP).min(padded / HOP)
}

/// Audio-token count for `frames` real mel frames: each full 100-frame chunk
/// yields 13 tokens; the remainder chunk passes through the three k3/s2/p1
/// convs (floor arithmetic ported from the upstream extractor; floor
/// division must round toward negative infinity for the remainder-0 case).
pub fn audio_token_count(frames: usize) -> usize {
    let fd = |a: i64, b: i64| a.div_euclid(b);
    let leave = (frames % CHUNK_FRAMES) as i64;
    let c1 = fd(leave - 1, 2) + 1;
    let c2 = fd(c1 - 1, 2) + 1;
    let c3 = fd(c2 - 1, 2) + 1;
    (frames / CHUNK_FRAMES) * 13 + c3.max(0) as usize
}

/// Token count for a clip of `len` samples - what the prompt scaffolding
/// must reserve at `<|audio_token|>` positions.
pub fn audio_tokens_for_samples(len: usize) -> usize {
    audio_token_count(real_frames(len))
}

/// Split `n_frames` real frames into encoder windows: `(start, len)` pairs,
/// each at most [`WINDOW_FRAMES`]. The tower convolves each window in
/// 100-frame chunks with the last chunk short (upstream conv semantics -
/// no zero-frame inflation).
pub fn split_windows(n_frames: usize) -> Vec<(usize, usize)> {
    let mut v = Vec::new();
    let mut off = 0;
    while off < n_frames {
        let len = WINDOW_FRAMES.min(n_frames - off);
        v.push((off, len));
        off += len;
    }
    v
}

/// Extract the normalized log-mel features for one 16 kHz mono clip under
/// the Qwen3-ASR extractor contract.
pub fn qwen3_asr_features(samples: &[f32]) -> Result<MelFeatures, String> {
    mel_features(samples, MelPolicy::Qwen3Asr)
}

/// One Whisper encoder window (30 s, always [`WHISPER_FRAMES`] frames) of
/// normalized log-mel features. Pass at most 30 s of samples per call -
/// anything longer belongs to the next window and is truncated here.
pub fn whisper_features(samples: &[f32]) -> Result<MelFeatures, String> {
    mel_features(samples, MelPolicy::Whisper)
}

/// Extract the normalized log-mel features for one 16 kHz mono clip under
/// the given extractor contract (see [`MelPolicy`] - the DSP below is shared
/// verbatim between the families; only the framing differs).
pub fn mel_features(samples: &[f32], policy: MelPolicy) -> Result<MelFeatures, String> {
    if samples.is_empty() {
        return Err("empty audio".into());
    }
    let mut owned;
    let signal = match policy {
        MelPolicy::Qwen3Asr if samples.len() < MIN_SAMPLES => {
            owned = samples.to_vec();
            owned.resize(MIN_SAMPLES, 0.0);
            &owned[..]
        }
        // one call is one 30 s encoder pass: samples past the window belong
        // to the next one, and a short window is zero-padded by the STFT's
        // own boundary rule below
        MelPolicy::Whisper => &samples[..samples.len().min(PAD_SAMPLES)],
        MelPolicy::Qwen3Asr => samples,
    };
    let l = signal.len();
    let padded = l.max(PAD_SAMPLES);
    let (n_real, enc_frames) = match policy {
        MelPolicy::Qwen3Asr => {
            let r = real_frames(l);
            // the conv stem consumes whole 100-frame chunks
            (r, r.div_ceil(CHUNK_FRAMES) * CHUNK_FRAMES)
        }
        // fixed-size encoder: all 3000 frames are consumed, silence included
        MelPolicy::Whisper => (WHISPER_FRAMES, WHISPER_FRAMES),
    };
    // Frames beyond the signal are exactly log10(1e-10) = -10 and can never
    // exceed the clamp floor, so the oracle's global max over the whole
    // padded spectrogram equals the max over frames whose window still
    // touches signal: scan up to the last such frame.
    let half = N_FFT / 2;
    let n_energy = ((l + half).div_ceil(HOP) + 1).min(padded / HOP);

    let window = dsp::hann_periodic(N_FFT);
    let fb = dsp::mel_filterbank_slaney(N_MEL, N_FFT, SAMPLE_RATE as f64, 0.0, 8000.0);
    let plan = dsp::FftPlan::new(N_FFT);

    // Virtual sample access implementing the boundary semantics: reflection
    // at the start, zeros in the 30 s padding, reflection at the padded end
    // (reachable only when padded == len, i.e. clips >= 30 s).
    let at = |i: i64| -> f64 {
        let (l, p) = (l as i64, padded as i64);
        let i = if i < 0 {
            -i
        } else if i >= p {
            2 * p - 2 - i
        } else {
            i
        };
        if i < l {
            signal[i as usize] as f64
        } else {
            0.0
        }
    };

    // Frames are independent, so the loop fans out over host threads (the
    // service's sampling fan-out precedent): each thread owns a contiguous
    // t-range of the output plane. Bit-identical to the serial loop - every
    // frame's arithmetic is self-contained and gmax is an exact max-reduce,
    // insensitive to combination order.
    let mut mel = vec![0.0f64; n_energy * N_MEL];
    let threads = std::thread::available_parallelism()
        .map_or(1, std::num::NonZeroUsize::get)
        .min(n_energy.div_ceil(64))
        .max(1);
    let chunk_t = n_energy.div_ceil(threads);
    let gmax = std::thread::scope(|s| {
        let mut handles = Vec::new();
        for (ci, out) in mel.chunks_mut(chunk_t * N_MEL).enumerate() {
            let (window, fb, plan, at) = (&window, &fb, &plan, &at);
            handles.push(s.spawn(move || {
                let mut frame = vec![0.0f64; N_FFT];
                let mut local_max = f64::NEG_INFINITY;
                for (j, row) in out.chunks_mut(N_MEL).enumerate() {
                    let t = ci * chunk_t + j;
                    let base = (t * HOP) as i64 - half as i64;
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

    // Emit data through `enc_frames` with the upstream pad semantics: frames
    // past the clip but within the 30 s zero-padded AUDIO are the log-mel of
    // silence - the floored constant, with real spectral bleed in the first
    // frame(s) whose STFT window still straddles the boundary (all computed
    // above, the scan runs to n_energy) - while frames past the STORED
    // spectrogram (Qwen3-ASR clips over 30 s only) are the extractor's
    // feature-axis np.pad: literal zeros. Zero-filling the silence region
    // instead of using the constant corrupts the last chunk's trailing
    // tokens and, through the shared attention window, every token near them
    // (found with the embedding oracle: final-token cos 0.79 vs
    // upstream before this). Whisper hits the same seam every request - its
    // window is 30 s no matter how short the clip is.
    let stored = padded / HOP;
    let floor = gmax - 8.0;
    let silence = (((-10.0f64).max(floor) + 4.0) / 4.0) as f32;
    let mut data = vec![0.0f32; enc_frames * N_MEL];
    let computed = n_energy.min(enc_frames);
    for (d, &v) in data.iter_mut().zip(mel.iter().take(computed * N_MEL)) {
        *d = ((v.max(floor) + 4.0) / 4.0) as f32;
    }
    let constant_until = stored.min(enc_frames);
    data[computed * N_MEL..constant_until * N_MEL].fill(silence);
    // `l`, not `samples.len()`: the whisper policy has already cut anything past
    // the 30 s window (it belongs to the next one) and the Qwen3-ASR policy has
    // already zero-extended a sub-0.5 s clip, which upstream counts as real
    // audio. Both are the length this block genuinely covers.
    Ok(MelFeatures {
        data,
        n_frames: n_real,
        n_samples: l,
        global_max: gmax as f32,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_count_matches_upstream_formula() {
        // (frames, tokens) pairs cross-checked against the oracle battery
        // and vLLM's _get_feat_extract_output_lengths
        for (frames, want) in [
            (200, 26),
            (328, 43),
            (100, 13),
            (1251, 163),
            (50, 7),
            (3000, 390),
            (3050, 397),
            (201, 27),
            (1, 1),
        ] {
            assert_eq!(audio_token_count(frames), want, "frames={frames}");
        }
    }

    #[test]
    fn real_frame_rule_covers_the_edges() {
        assert_eq!(real_frames(32000), 200);
        assert_eq!(real_frames(52327), 328); // non-hop-divisible rounds up
        assert_eq!(real_frames(4800), 50); // min-length extension
        assert_eq!(real_frames(480000), 3000);
        assert_eq!(real_frames(488137), 3050); // >30s drop-last edge
    }

    #[test]
    fn windows_split_at_800_with_short_tail() {
        assert_eq!(split_windows(1251), vec![(0, 800), (800, 451)]);
        assert_eq!(split_windows(800), vec![(0, 800)]);
        assert_eq!(split_windows(26), vec![(0, 26)]);
    }
}
