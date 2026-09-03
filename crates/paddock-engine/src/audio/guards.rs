//! Decode robustness guards for the transcription lanes.
//!
//! An ASR decode has two failure modes that look nothing like an error and
//! answer 200 with garbage:
//!
//!   * **the loop** - the model gets stuck emitting the same few tokens and
//!     only the context bound stops it. Measured on granite-speech-plus
//!     7.8 s of Mandarin, ~4000 tokens of a single repeated
//!     character, 16.64 s of GPU. Whisper large-v3 does the same thing at 448.
//!   * **text over silence** - whisper writes a plausible sentence over a
//!     window that holds no speech at all. Its famous failure, and the one a
//!     clean read-speech battery (FLEURS) can never see, because every clip
//!     there decodes to `<|endoftext|>` long before anything can go wrong.
//!
//! whisper.cpp guards its loop with four thresholds (`src/whisper.cpp`
//! 5976-5979) and we had none. Three of the four are ported here with their
//! values unchanged; the fourth - `temperature_inc`, the retry ladder - is
//! deliberately not, and that changes what the other three are for:
//!
//! | guard | value | there | here |
//! |---|---|---|---|
//! | entropy | 2.4 | fail the window, retry hotter | CUT the decode |
//! | logprob | -1.0 | half of the retry test | half of the silence test |
//! | no_speech | 0.6 | half of the silence test | half of the silence test |
//! | temperature_inc | 0.2 | the ladder | not served (greedy lane) |
//!
//! Without a ladder there is nothing to fall back to, so a guard here either
//! stops work that is going nowhere or refuses a transcript that is not about
//! the audio. Both are reported: under the no-silent-failures principle a
//! decode that was cut has to SAY it was cut, which is why every guard here
//! returns a verdict the wire can carry rather than quietly editing a string.
//!
//! Two deviations from the reference are worth naming.
//!
//! whisper.cpp scores the entropy once, on the finished sequence's last 32
//! tokens, because it is deciding whether to retry. We check every step past
//! 32, because we are deciding whether to keep spending - a check that only
//! fires at the end saves nothing, and the whole point on the granite clip is
//! the 4000 tokens never decoded.
//!
//! And we added a second, stricter test beside it: **exact period**, the tail
//! being one short pattern repeated. It is what a degenerate ASR decode
//! actually looks like, it fires far earlier than a statistic can, and - the
//! reason it had to exist - it is the only one of the two that is safe on a
//! lane whose output has a GRAMMAR. See `Repetition`.
//!
//! Not a VAD. `audio::vad` answers "is anyone talking" from the samples; this
//! answers "did the decode stay about the audio" from the tokens. They pair
//! (a VAD-gated window never reaches a decoder to hallucinate in) but neither
//! substitutes for the other.
//!
//! ## this is not the STATE of the ART, and here is what is
//!
//! Stated because the convention says a knowingly-simpler-than-SOTA interim
//! documents its gap and its target, and because the survey that establishes
//! this was compiled after the code.
//!
//! Everything above except the period test is OpenAI's 2022 `decoding.py`,
//! copied forward unchanged by whisper.cpp and faster-whisper alike - which
//! makes the constants look validated when they are only widespread. They have
//! never been measured on a checkpoint we serve, and the one measurement that
//! exists is a failure (granite's timestamp mode scored 2.3754 against the 2.4
//! threshold). The field's own verdict on the family is that it "helps mitigate
//! but doesn't eliminate" (arXiv:2502.12414).
//!
//! Two upgrades need nothing trained, which matters here:
//!
//!   * cross-attention temporal monotonicity (NPUsper,
//!     arXiv:2607.01108). A hallucinated token's attention shifts backward
//!     along the audio timeline; a real transcript sweeps forward. No
//!     threshold to tune at all, and the tensor is already captured for word
//!     timing. It is the only signal here that asks whether the decoder
//!     is still listening rather than whether the output looks odd.
//!   * `is_no_speech` tests `avg_logprob`, i.e. raw softmax,
//!     which is the exact quantity NVIDIA's entropy-confidence work replaces
//!     (arXiv:2212.08703: 1.5-4x better at finding wrong words, non-trainable,
//!     same cost). A Renyi entropy is already computed in-kernel and read by
//!     nobody.
//!
//! Also open: VAD gating (every shipping ASR system does it), a measured
//! false-positive rate for anything in this file, and the compression ratio
//! we compute and ignore.

use std::collections::{HashMap, VecDeque};

/// Shannon entropy below this, over the tail window, is a repetition loop.
/// whisper.cpp's `entropy_thold`.
pub const ENTROPY_THOLD: f32 = 2.4;
/// Mean logprob below this is low-confidence output. whisper.cpp's
/// `logprob_thold`; on its own it means nothing here (there is no retry), and
/// it only acts as half of the silence rule.
pub const LOGPROB_THOLD: f32 = -1.0;
/// `<|nospeech|>` above this says the window holds no speech. whisper.cpp's
/// `no_speech_thold`.
pub const NO_SPEECH_THOLD: f32 = 0.6;
/// How many tokens the entropy window holds, and how many must be decoded
/// before it is allowed to judge anything. whisper.cpp's `n = 32`.
pub const ENTROPY_WINDOW: usize = 32;

/// Longest repeating pattern the period test looks for.
pub const PERIOD_MAX: usize = 12;
/// A repeated run must repeat at least this many times...
pub const PERIOD_REPEATS: usize = 4;
/// ...AND cover at least this many tokens. The second bound is what keeps
/// "no no no no" a sentence: four one-token repeats is emphasis, sixteen is a
/// decode that has stopped listening.
pub const PERIOD_MIN_TOKENS: usize = 16;

/// The deepest tail any test here reads.
const TAIL: usize = if PERIOD_MAX * PERIOD_REPEATS > ENTROPY_WINDOW {
    PERIOD_MAX * PERIOD_REPEATS
} else {
    ENTROPY_WINDOW
};

/// Two tests for "this decode has stopped being about the audio", over a
/// rolling tail of generated token ids.
///
/// **Exact period** - the tail is one short pattern repeated. This is what a
/// degenerate ASR decode actually looks like (one character, 4000 times),
/// it is the technique the modern repetition work uses (the DRY
/// sampler is exact-suffix matching, not a statistic), and it costs almost
/// nothing in false positives because natural language does not repeat a span
/// byte-for-byte sixteen tokens deep.
///
/// **Tail entropy** - Shannon entropy of the token-ID HISTOGRAM over the last
/// 32, whisper.cpp's `entropy_thold`. Catches degeneration that drifts instead
/// of cycling exactly. Not to be confused with the per-step distribution
/// entropy the sampler computes: that asks how unsure the model
/// was at one step, this asks how much the last 32 steps repeated themselves.
/// A decode cycling between three tokens scores ln 3 = 1.10 whatever its
/// confidence - and a model in a loop is usually very confident, which is
/// exactly why a logprob threshold cannot catch this and a histogram can.
///
/// **Entropy is only safe on PLAIN TEXT**, which is why it is optional and why
/// `structured()` exists. Measured by the conformance gate:
/// granite-speech-plus in timestamp mode emits `word [T:452] word [T:498]`,
/// where `[`, `T`, `:` and `]` recur every few tokens by construction - its
/// legitimate output scores **2.3754**, just under the 2.4 threshold, and the
/// guard cut a perfectly good transcript after six words. A format with a
/// grammar is low-entropy deliberately; the period test is not fooled by it
/// (the numbers differ, so there is no exact period) and is the only one that
/// lane runs.
pub struct Repetition {
    /// the last `TAIL` ids, oldest first
    tail: VecDeque<u32>,
    seen: usize,
    /// entropy threshold, or None where the output has a grammar
    entropy: Option<f32>,
}

impl Repetition {
    /// For a lane whose output is plain transcript text: both tests.
    pub fn text() -> Self {
        Self {
            tail: VecDeque::with_capacity(TAIL),
            seen: 0,
            entropy: Some(ENTROPY_THOLD),
        }
    }

    /// For a lane whose output has a GRAMMAR - tags, markers, anything that
    /// recurs by design. Period only; see the type doc for the measurement.
    pub fn structured() -> Self {
        Self {
            tail: VecDeque::with_capacity(TAIL),
            seen: 0,
            entropy: None,
        }
    }

    /// Feed one generated token. `true` means the tail has collapsed and the
    /// caller should stop decoding.
    pub fn push(&mut self, id: u32) -> bool {
        self.seen += 1;
        self.tail.push_back(id);
        while self.tail.len() > TAIL {
            self.tail.pop_front();
        }
        self.collapsed()
    }

    /// The entropy-window value in nats, or `f32::INFINITY` where there is not
    /// enough decoded to say (or the lane does not run the test) - infinity
    /// rather than zero because every reader compares it against a threshold,
    /// and a zero would read as "maximally repetitive" at exactly the moment
    /// nothing is known.
    pub fn value(&self) -> f32 {
        if self.entropy.is_none() || self.seen <= ENTROPY_WINDOW {
            return f32::INFINITY;
        }
        let mut counts: HashMap<u32, u32> = HashMap::new();
        let from = self.tail.len() - ENTROPY_WINDOW.min(self.tail.len());
        for i in from..self.tail.len() {
            *counts.entry(self.tail[i]).or_insert(0) += 1;
        }
        let n = (self.tail.len() - from) as f32;
        -counts
            .values()
            .map(|&c| {
                let p = c as f32 / n;
                p * p.ln()
            })
            .sum::<f32>()
    }

    /// Is the tail one short pattern repeated back to back?
    ///
    /// Shortest period first, so a single repeated token reports as period 1
    /// rather than as period 2 of a pair of them.
    pub fn periodic(&self) -> bool {
        for k in 1..=PERIOD_MAX {
            let reps = PERIOD_REPEATS.max(PERIOD_MIN_TOKENS.div_ceil(k));
            let need = k * reps;
            if self.tail.len() < need {
                continue;
            }
            let from = self.tail.len() - need;
            if (from + k..self.tail.len()).all(|i| self.tail[i] == self.tail[i - k]) {
                return true;
            }
        }
        false
    }

    pub fn collapsed(&self) -> bool {
        self.periodic() || self.entropy.is_some_and(|t| self.value() < t)
    }
}

/// Why a window's decode ended.
///
/// `Eot` is the model saying it is done and the only one that needs no
/// explaining to a caller; the other three all mean the decode was stopped
/// from outside, and each of them reaches the wire.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Stop {
    /// `<|endoftext|>` - the model finished
    #[default]
    Eot,
    /// the caller's `max_tokens`
    Budget,
    /// the decoder's served context ran out
    Context,
    /// the tail entropy collapsed: the decode was looping
    Repetition,
    /// the VAD found no speech in this span, so it was never decoded at all
    /// Not a failure and not a suppression - a decode that did
    /// not happen, which the caller still has to be told about: their audio
    /// produced nothing because we chose not to look at it.
    Vad,
    /// the model TYPED a no-speech marker as its whole transcript.
    ///
    /// Its own reason rather than folding into the silence rule, because the
    /// two know the same thing by different routes and the notice should not
    /// claim the wrong one: the rule reads two statistics, this reads the
    /// model saying so in words.
    Marker,
}

impl Stop {
    /// The wire name, which is also what the docs and the Studio use. `Eot`
    /// has none - a decode that simply finished is not a notice.
    pub fn wire(self) -> Option<&'static str> {
        match self {
            Stop::Eot => None,
            Stop::Budget => Some("length"),
            Stop::Context => Some("context"),
            Stop::Repetition => Some("repetition"),
            Stop::Vad => Some("vad"),
            Stop::Marker => Some("no_speech_marker"),
        }
    }
}

/// whisper.cpp's silence rule, verbatim (`whisper.cpp:7622`): a window is
/// non-speech when the no-speech head is confident AND the transcript it
/// produced anyway is weak.
///
/// The AND is the whole safety of it. `no_speech_prob` alone is noisy enough
/// that acting on it would delete real speech - which is a worse failure than
/// the one being fixed, and an invisible one. A window that genuinely holds
/// speech decodes with an average logprob around -0.1 to -0.3 and can never
/// trip this however the no-speech head feels about it.
pub fn is_no_speech(no_speech_prob: f32, avg_logprob: f32) -> bool {
    no_speech_prob > NO_SPEECH_THOLD && avg_logprob < LOGPROB_THOLD
}

/// Mean logprob over a decode, or 0.0 for an empty one - the neutral value,
/// since every threshold here is a "below this" test and an empty window has
/// nothing to be unconfident about.
pub fn avg_logprob(lps: &[f32]) -> f32 {
    if lps.is_empty() {
        return 0.0;
    }
    lps.iter().sum::<f32>() / lps.len() as f32
}

/// An honest ceiling on how many tokens a transcription of `duration_s`
/// seconds of audio can need.
///
/// The backstop for degeneration that is not a tight loop - drifting free
/// text, an answer to a question nobody asked - which the entropy guard
/// cannot see. Speech runs at ~3 words/s and tops out near 8; at ~2 tokens a
/// word that is under 20 tokens/s, so 50 is a factor of ~2.5 of headroom over
/// the fastest speech anyone produces, and it still leaves room for the
/// `[T:N]` tag stream granite-speech-plus writes in its timestamp mode. The
/// floor covers clips too short to have a sensible ceiling at all.
///
/// Rounded up, because the tail of a clip is a fraction of a second and a
/// transcript is not.
pub fn token_ceiling(duration_s: f64) -> usize {
    const PER_SECOND: f64 = 50.0;
    const FLOOR: usize = 128;
    (duration_s.max(0.0) * PER_SECOND).ceil() as usize + FLOOR
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feed_into(mut r: Repetition, ids: &[u32]) -> (Repetition, Option<usize>) {
        let mut at = None;
        for (i, &id) in ids.iter().enumerate() {
            if r.push(id) && at.is_none() {
                at = Some(i);
            }
        }
        (r, at)
    }

    fn feed(ids: &[u32]) -> (Repetition, Option<usize>) {
        feed_into(Repetition::text(), ids)
    }

    /// Neither test may judge a decode that has barely started: an unfilled
    /// entropy window is trivially low-entropy, and four tokens are not a
    /// pattern.
    #[test]
    fn a_short_decode_is_never_repetitive() {
        let (r, at) = feed(&[7, 8, 9, 10, 11, 12, 13, 14]);
        assert_eq!(at, None, "fired on eight tokens");
        assert_eq!(r.value(), f32::INFINITY);
    }

    /// The real-world failure, in miniature: one token, over and over. The period
    /// test catches it at 16 rather than waiting for the entropy window - the
    /// whole point is the 4000 that follow are never decoded.
    #[test]
    fn a_single_repeated_token_is_caught_at_sixteen() {
        let (_, at) = feed(&[7u32; 200]);
        assert_eq!(at, Some(PERIOD_MIN_TOKENS - 1), "index of the 16th token");
    }

    /// Four of the same word is emphasis, not a loop - the token floor is
    /// what separates them, and it is the whole reason the period test can
    /// run on plain speech at all.
    #[test]
    fn four_repeats_are_a_sentence_not_a_loop() {
        let mut ids = vec![100, 101, 102]; // "he said"
        ids.extend([55u32; 4]); // "no no no no"
        ids.extend([103, 104, 105, 106]);
        let (_, at) = feed(&ids);
        assert_eq!(at, None);
    }

    /// A cycle is the other shape a loop takes, and confidence says nothing
    /// about it - a model going "ha ha ha" is usually very sure.
    #[test]
    fn a_short_cycle_is_a_loop_too() {
        let ids: Vec<u32> = (0..200).map(|i| (i % 3) as u32).collect();
        let (_, at) = feed(&ids);
        assert!(at.is_some(), "3-token cycle not caught");
        // and the entropy arm agrees about it: ln 3 = 1.0986
        let (r, _) = feed_into(Repetition::text(), &ids[..64]);
        assert!((r.value() - 3f32.ln()).abs() < 1e-3, "{}", r.value());
    }

    /// And the false-positive side, which is what decides whether this can
    /// ship: ordinary text must survive. 11 distinct tokens uniformly spread
    /// is where the entropy threshold sits (ln 11 = 2.398), and real prose is
    /// far above it - a 32-token stretch of speech carries 25+ distinct
    /// pieces.
    #[test]
    fn ordinary_text_is_not_a_loop() {
        // a token stream with the repetition natural language actually has:
        // a handful of frequent function words plus a long tail
        let mut ids = Vec::new();
        for i in 0..200u32 {
            ids.push(if i % 4 == 0 { i % 3 } else { 100 + i });
        }
        let (r, at) = feed(&ids);
        assert_eq!(at, None, "entropy {} tripped on ordinary text", r.value());
    }

    /// The REGRESSION, measured by the conformance gate:
    /// granite-speech-plus's timestamp mode writes `word [T:452] word [T:498]`,
    /// so four of its six tokens per word recur by construction. Its
    /// legitimate output scored 2.3754 - under the 2.4 threshold - and the
    /// entropy guard cut a good transcript after six words. A format with a
    /// grammar gets the period test only, which its varying numbers defeat.
    #[test]
    fn a_tagged_transcript_is_not_a_loop() {
        // word, space, '[', 'T', ':', <number>, ']' - the number is the only
        // thing that moves, exactly as in the real stream
        let mut ids = Vec::new();
        for w in 0..40u32 {
            ids.extend([300 + w, 1, 2, 3, 4, 500 + w * 7, 5]);
        }
        let (_, at) = feed_into(Repetition::structured(), &ids);
        assert_eq!(at, None, "the tag scaffolding read as a loop");
        // and the entropy arm really would have fired - this is the number
        // the gate measured, reproduced
        let (t, text_at) = feed(&ids);
        assert!(
            t.value() < ENTROPY_THOLD,
            "entropy {} should be under 2.4",
            t.value()
        );
        assert!(
            text_at.is_some(),
            "the text-mode guard is what broke the granite lane"
        );
    }

    /// A structured lane is still guarded - the period test is what catches
    /// a real loop there, and it is not weakened by dropping entropy.
    #[test]
    fn a_structured_lane_still_catches_a_real_loop() {
        let (_, at) = feed_into(Repetition::structured(), &[7u32; 200]);
        assert_eq!(at, Some(PERIOD_MIN_TOKENS - 1));
    }

    /// A loop the model climbs out of still costs its cut, and that is the
    /// deliberate deviation from whisper.cpp (which scores the tail once, at
    /// the end, because it is deciding whether to retry). Pinned so the
    /// difference is a decision rather than a surprise.
    #[test]
    fn a_recovered_loop_is_still_cut() {
        let mut ids = vec![9u32; 40];
        ids.extend(200..260);
        let (_, at) = feed(&ids);
        assert_eq!(at, Some(PERIOD_MIN_TOKENS - 1));
    }

    /// The AND is the safety. A confident no-speech head over a transcript
    /// the model was sure of is not silence - that combination is a window of
    /// speech the head got wrong, and deleting it would be the worse bug.
    #[test]
    fn silence_needs_both_halves() {
        assert!(is_no_speech(0.9, -1.5));
        assert!(
            !is_no_speech(0.9, -0.2),
            "confident transcript is not silence"
        );
        assert!(
            !is_no_speech(0.2, -1.5),
            "weak transcript alone is not silence"
        );
        // and the thresholds are exclusive on both sides
        assert!(!is_no_speech(NO_SPEECH_THOLD, -2.0));
        assert!(!is_no_speech(0.9, LOGPROB_THOLD));
    }

    #[test]
    fn the_token_ceiling_is_generous_but_finite() {
        // the pathological clip: 7.8 s answered with ~4000 tokens
        let cap = token_ceiling(7.8);
        assert!(cap < 600 && cap > 200, "{cap}");
        // 30 s of the fastest speech anyone produces is ~240 words; the cap
        // has to clear that by a wide margin
        assert!(token_ceiling(30.0) > 1200);
        // a clip of nothing still gets room to say so
        assert!(token_ceiling(0.0) >= 128);
    }
}
