//! Whisper's timestamp tokens, parsed into segments.
//!
//! Whisper does not report times out of band - it EMITS them, as 1501 special
//! tokens `<|0.00|>`..`<|30.00|>` that the decoder can produce anywhere a text
//! token could go. A window transcribed with timestamps therefore looks like
//!
//! ```text
//! <|0.00|> Hej och välkommen <|2.48|><|2.48|> till sändningen <|5.02|>
//! ```
//!
//! and the whole job here is turning that stream into spans. This module is
//! pure host arithmetic over ids the decoder already produced; nothing about
//! it touches the GPU.
//!
//! SPLIT RULE - the reference's, deliberately. OpenAI's `transcribe.py` cuts a
//! window only where two timestamps sit next to each other (one closing a
//! segment, the next opening the following one), never at a lone timestamp in
//! the middle of running text. That is what keeps a stray `<|3.20|>` inside a
//! sentence from shattering it, and matching it exactly is what makes our
//! segment boundaries agree with every other whisper implementation on the
//! same audio.
//!
//! What we do DIFFERENTLY, and why:
//!
//! 1. No TOKEN is ever DROPPED. The reference re-seeks the audio after the
//!    last complete segment and re-decodes the remainder in the next window;
//!    our windows are a fixed 30 s cut made before the model ever ran, so a
//!    trailing incomplete segment has no second chance. It is emitted with
//!    the times we do know rather than discarded - `split_segments` guarantees
//!    every text token lands in exactly one segment, which is what lets the
//!    caller trust that the segments concatenate back to the transcript.
//!
//! 2. `avg_logprob` is per SEGMENT. In OpenAI's implementation every segment
//!    of a window carries the same average, because the value is computed once
//!    per `DecodingResult` and copied onto each. The API documents the field
//!    as "average logprob of the segment", per-segment is what that says, and
//!    a window-wide constant is useless for the one thing this exists to feed
//!    - colouring the transcript by how sure the model was.

use crate::gpu::asr::ts_flags;

use super::TimeScale;

/// Build one decode row's state for the device-side timestamp grammar
/// (`whisper_ts_rules`) from the tokens it has sampled so far.
///
/// This is the host half of `ApplyTimestampRules`: everything the filter needs
/// is three facts about the tail of the sequence plus the lowest timestamp
/// still allowed, and the scheduler has the sequence anyway. Keeping it here
/// rather than in the kernel means the kernel never walks a token history.
///
/// `enabled` false returns a state the kernel skips outright, which is what
/// lets one batched step carry timestamped and plain rows together.
pub fn ts_state(sampled: &[u32], ts: &TimeScale, enabled: bool) -> [u32; 2] {
    if !enabled {
        return [0, 0];
    }
    let mut flags = ts_flags::ON;
    if sampled.is_empty() {
        return [flags | ts_flags::BEGIN, 0];
    }
    let last_is_ts = ts.is_timestamp(sampled[sampled.len() - 1]);
    if last_is_ts {
        flags |= ts_flags::LAST;
    }
    // The reference counts a single sampled token as "the penultimate was a
    // timestamp", which is what makes a window's opening timestamp be followed
    // by text rather than by a second timestamp.
    if sampled.len() < 2 || ts.is_timestamp(sampled[sampled.len() - 2]) {
        flags |= ts_flags::PENULT;
    }
    let last_ts = sampled.iter().rev().find(|&&t| ts.is_timestamp(t)).copied();
    match last_ts {
        None => [flags, 0],
        Some(t) => {
            // Times never go backwards. A lone timestamp may REPEAT (it closes
            // one segment and opens the next at the same instant); a closing
            // one must advance.
            let floor = if last_is_ts && (flags & ts_flags::PENULT) == 0 {
                t
            } else {
                t + 1
            };
            [flags | ts_flags::HAVE, floor]
        }
    }
}

/// One timestamped span of a window's transcript.
#[derive(Clone, Debug, PartialEq)]
pub struct Segment {
    /// seconds from the START of the CLIP (the window offset is applied here)
    pub start: f32,
    pub end: f32,
    /// the segment's TEXT tokens - timestamps stripped, ready to detokenize
    pub tokens: Vec<u32>,
    /// parallel to `tokens`: what the model thought of each one. Kept per
    /// token rather than only averaged because a transcript coloured by
    /// confidence needs to point at the WORD the model was unsure of, and an
    /// average over a whole sentence cannot.
    pub logprobs: Vec<f32>,
    /// parallel to `tokens`: the runner-up at each pick, `(id, log p)`. It
    /// rides through the same timestamp filter as `logprobs` for the same
    /// reason - an index that means one token in one vector and another token
    /// in the next would attach an alternative to the wrong word.
    pub runners: Vec<Option<(u32, f32)>>,
    /// mean log-probability over `tokens`; 0.0 for an empty segment
    pub avg_logprob: f32,
    /// index of the 30 s window this came from
    pub window: usize,
}

impl Segment {
    /// Mean linear probability - what a confidence colour band reads.
    /// Kept next to the logprob rather than instead of it: the API reports
    /// the log, the UI wants the 0..1.
    pub fn confidence(&self) -> f32 {
        self.avg_logprob.exp()
    }
}

/// Split one window's emitted tokens into timestamped segments.
///
/// `tokens` is what the decoder produced for this window - no prompt, no
/// `<|endoftext|>` - and `logprobs` is parallel to it (empty is allowed, and
/// then every `avg_logprob` is 0). `runners` is parallel too and may be empty,
/// which is how a lane says it has no second candidate to report rather than
/// claiming there wasn't one. `window` is the window's index in the clip,
/// which is what turns a within-window timestamp into a clip time.
///
/// Returns an empty vec for an empty window. A window with no timestamp tokens
/// at all (which is what a `<|notimestamps|>` prompt produces) comes back as
/// one segment spanning the whole window - honest about what is known rather
/// than inventing boundaries.
pub fn split_segments(
    tokens: &[u32],
    logprobs: &[f32],
    runners: &[Option<(u32, f32)>],
    ts: &TimeScale,
    window: usize,
) -> Vec<Segment> {
    if tokens.is_empty() {
        return Vec::new();
    }
    let win0 = window as f32 * ts.window_s;
    let win1 = win0 + ts.window_s;
    let lp = |r: std::ops::Range<usize>| -> f32 {
        let vals: Vec<f32> = r
            .filter(|&i| i < logprobs.len() && !ts.is_timestamp(tokens[i]))
            .map(|i| logprobs[i])
            .collect();
        if vals.is_empty() {
            0.0
        } else {
            vals.iter().sum::<f32>() / vals.len() as f32
        }
    };

    // Cut points: the index just past the first of every adjacent timestamp
    // pair. A pair is "this segment closes here, the next opens with the same
    // (or a later) time" - the reference's `consecutive`.
    let mut cuts: Vec<usize> = (0..tokens.len().saturating_sub(1))
        .filter(|&i| ts.is_timestamp(tokens[i]) && ts.is_timestamp(tokens[i + 1]))
        .map(|i| i + 1)
        .collect();
    // A window that ENDS on a lone timestamp closed its last segment cleanly;
    // without this the tail would look incomplete and inherit the window end.
    if tokens.len() >= 2
        && ts.is_timestamp(tokens[tokens.len() - 1])
        && !ts.is_timestamp(tokens[tokens.len() - 2])
    {
        cuts.push(tokens.len());
    }
    if cuts.last() != Some(&tokens.len()) {
        // the leftover after the last cut (see note 1): emitted, never dropped
        cuts.push(tokens.len());
    }

    let mut out: Vec<Segment> = Vec::with_capacity(cuts.len());
    let mut last = 0usize;
    for cut in cuts {
        if cut <= last {
            continue;
        }
        let slice = &tokens[last..cut];
        // A slice normally opens and closes on a timestamp. When it does not -
        // the model skipped one, or this is the trailing remainder - fall back
        // to the previous segment's end and the window's end rather than
        // reading a text token as a time.
        let start = match slice.first() {
            Some(&t) if ts.is_timestamp(t) => ts.seconds(t, window),
            _ => out.last().map_or(win0, |s: &Segment| s.end),
        };
        let end = match slice.last() {
            Some(&t) if ts.is_timestamp(t) && slice.len() > 1 => ts.seconds(t, window),
            _ => win1,
        };
        // text tokens and their confidence, kept in step: the filter has to run
        // over all three together or a later index would read another token's
        // numbers
        let (text, (lps, runs)): (Vec<u32>, (Vec<f32>, Vec<Option<(u32, f32)>>)) = (last..cut)
            .filter(|&i| !ts.is_timestamp(tokens[i]))
            .map(|i| {
                (
                    tokens[i],
                    (
                        logprobs.get(i).copied().unwrap_or(0.0),
                        runners.get(i).copied().flatten(),
                    ),
                )
            })
            .unzip();
        if text.is_empty() {
            // a bare `<|t|><|t|>` pair carries no words; it is a boundary the
            // model emitted, not a segment, and an empty caption helps nobody
            last = cut;
            continue;
        }
        out.push(Segment {
            start,
            // a zero-length or backwards span is the model's own slip; clamp
            // so downstream players and SRT writers never see end < start
            end: end.max(start),
            avg_logprob: lp(last..cut),
            tokens: text,
            logprobs: lps,
            runners: runs,
            window,
        });
        last = cut;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// large-v3's real geometry: `<|0.00|>` at 50365, 0.02 s a step, 30 s
    /// windows.
    fn scale() -> TimeScale {
        TimeScale {
            begin: 50365,
            precision: 0.02,
            window_s: 30.0,
        }
    }

    /// `<|s|>` for a time in seconds.
    fn ts(sec: f32) -> u32 {
        50365 + (sec / 0.02).round() as u32
    }

    #[test]
    fn adjacent_timestamps_split_lone_ones_do_not() {
        let s = scale();
        // <|0.00|> a b <|2.00|><|2.00|> c <|4.00|>
        let toks = vec![ts(0.0), 1, 2, ts(2.0), ts(2.0), 3, ts(4.0)];
        let segs = split_segments(&toks, &[], &[], &s, 0);
        assert_eq!(segs.len(), 2);
        assert_eq!(segs[0].tokens, vec![1, 2]);
        assert!((segs[0].start - 0.0).abs() < 1e-4 && (segs[0].end - 2.0).abs() < 1e-4);
        assert_eq!(segs[1].tokens, vec![3]);
        assert!((segs[1].start - 2.0).abs() < 1e-4 && (segs[1].end - 4.0).abs() < 1e-4);

        // the same tokens with one timestamp in the middle stay one segment -
        // this is the case a naive "split on every timestamp" gets wrong
        let toks = vec![ts(0.0), 1, 2, ts(2.0), 3, ts(4.0)];
        let segs = split_segments(&toks, &[], &[], &s, 0);
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].tokens, vec![1, 2, 3]);
        assert!((segs[0].end - 4.0).abs() < 1e-4);
    }

    #[test]
    fn every_text_token_lands_in_exactly_one_segment() {
        let s = scale();
        // a deliberately ugly window: no opening timestamp, a pair mid-way,
        // and a trailing remainder the reference would have re-decoded
        let toks = vec![7, 8, ts(1.0), ts(1.0), 9, ts(3.0), ts(3.0), 10, 11];
        let segs = split_segments(&toks, &[], &[], &s, 0);
        let seen: Vec<u32> = segs.iter().flat_map(|g| g.tokens.clone()).collect();
        let want: Vec<u32> = toks
            .iter()
            .copied()
            .filter(|&t| !s.is_timestamp(t))
            .collect();
        assert_eq!(seen, want, "a token was dropped or reordered");
        // the tail with no closing timestamp runs to the end of the window
        assert!((segs.last().unwrap().end - 30.0).abs() < 1e-4);
    }

    #[test]
    fn window_offset_moves_times_into_clip_space() {
        let s = scale();
        let toks = vec![ts(1.0), 5, ts(2.0)];
        let segs = split_segments(&toks, &[], &[], &s, 3);
        assert_eq!(segs.len(), 1);
        // window 3 starts at 90 s
        assert!((segs[0].start - 91.0).abs() < 1e-3, "{}", segs[0].start);
        assert!((segs[0].end - 92.0).abs() < 1e-3, "{}", segs[0].end);
        assert_eq!(segs[0].window, 3);
    }

    #[test]
    fn a_window_without_timestamps_is_one_span() {
        // exactly what a `<|notimestamps|>` prompt produces - the times are
        // the window's own bounds, which is all that is actually known
        let s = scale();
        let segs = split_segments(&[1, 2, 3], &[], &[], &s, 1);
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].tokens, vec![1, 2, 3]);
        assert!((segs[0].start - 30.0).abs() < 1e-4);
        assert!((segs[0].end - 60.0).abs() < 1e-4);
    }

    #[test]
    fn avg_logprob_covers_the_segments_own_text_only() {
        let s = scale();
        let toks = vec![ts(0.0), 1, 2, ts(2.0), ts(2.0), 3, ts(4.0)];
        // timestamps get an absurd logprob so a leak into the mean is obvious
        let lps = vec![-9.0, -0.2, -0.4, -9.0, -9.0, -1.0, -9.0];
        let segs = split_segments(&toks, &lps, &[], &s, 0);
        assert!(
            (segs[0].avg_logprob - -0.3).abs() < 1e-5,
            "{}",
            segs[0].avg_logprob
        );
        assert!(
            (segs[1].avg_logprob - -1.0).abs() < 1e-5,
            "{}",
            segs[1].avg_logprob
        );
        // and the linear form the UI reads
        assert!((segs[0].confidence() - (-0.3f32).exp()).abs() < 1e-6);
    }

    #[test]
    fn per_token_logprobs_stay_aligned_to_their_tokens() {
        // The filter drops timestamps from all three lists or the survivors
        // shift onto the wrong tokens - which a UI would render as the wrong
        // words being uncertain, and the wrong alternatives offered for them,
        // silently.
        let s = scale();
        let toks = vec![ts(0.0), 11, ts(1.0), ts(1.0), 22, 33, ts(2.0)];
        let lps = vec![-9.0, -0.1, -9.0, -9.0, -0.2, -0.3, -9.0];
        // the runner-up ids echo their token so a shift is unmissable; the
        // timestamps carry None, which is also what the decoder reports for a
        // row the grammar left only one legal choice at
        let runs = vec![
            None,
            Some((110, -2.1)),
            None,
            None,
            Some((220, -2.2)),
            Some((330, -2.3)),
            None,
        ];
        let segs = split_segments(&toks, &lps, &runs, &s, 0);
        assert_eq!(segs[0].tokens, vec![11]);
        assert_eq!(segs[0].logprobs, vec![-0.1]);
        assert_eq!(segs[0].runners, vec![Some((110, -2.1))]);
        assert_eq!(segs[1].tokens, vec![22, 33]);
        assert_eq!(segs[1].logprobs, vec![-0.2, -0.3]);
        assert_eq!(segs[1].runners, vec![Some((220, -2.2)), Some((330, -2.3))]);
        for g in &segs {
            assert_eq!(
                g.tokens.len(),
                g.logprobs.len(),
                "lengths must never diverge"
            );
            assert_eq!(
                g.tokens.len(),
                g.runners.len(),
                "lengths must never diverge"
            );
        }
    }

    #[test]
    fn a_lane_with_no_runner_ups_still_splits() {
        // An empty `runners` is how a caller says "I have no second candidate
        // to report" - it must not become one None per token that reads as
        // "the model had no alternative", nor panic on the index.
        let s = scale();
        let toks = vec![ts(0.0), 11, 22, ts(2.0)];
        let segs = split_segments(&toks, &[-0.1, -0.2, -0.3, -0.4], &[], &s, 0);
        assert_eq!(segs[0].runners, vec![None, None]);
    }

    #[test]
    fn empty_and_boundary_only_windows_produce_nothing() {
        let s = scale();
        assert!(split_segments(&[], &[], &[], &s, 0).is_empty());
        // a pair of timestamps with no words between them is a boundary, not
        // a caption
        assert!(split_segments(&[ts(0.0), ts(0.0)], &[], &[], &s, 0).is_empty());
    }

    #[test]
    fn ts_state_tracks_the_reference_rules() {
        let s = scale();
        // disabled rows are invisible to the kernel
        assert_eq!(ts_state(&[1, 2], &s, false), [0, 0]);

        // nothing sampled: the window must open on a timestamp
        let st = ts_state(&[], &s, true);
        assert_eq!(st[0], ts_flags::ON | ts_flags::BEGIN);

        // one timestamp so far -> PENULT is set (len < 2), so text must follow
        let st = ts_state(&[ts(0.0)], &s, true);
        assert_eq!(st[0] & ts_flags::LAST, ts_flags::LAST);
        assert_eq!(st[0] & ts_flags::PENULT, ts_flags::PENULT);
        // and a repeat of the same instant stays legal (floor = t, not t+1)
        assert_eq!(st[1], ts(0.0) + 1);

        // mid-text: neither flag, but the floor still holds the last time
        let st = ts_state(&[ts(0.0), 5, 6], &s, true);
        assert_eq!(st[0] & (ts_flags::LAST | ts_flags::PENULT), 0);
        assert_eq!(st[0] & ts_flags::HAVE, ts_flags::HAVE);
        assert_eq!(st[1], ts(0.0) + 1);

        // a CLOSING timestamp (last is ts, penultimate is text): the next
        // timestamp may repeat it, so the floor is the value itself
        let st = ts_state(&[ts(0.0), 5, ts(2.0)], &s, true);
        assert_eq!(st[0] & ts_flags::LAST, ts_flags::LAST);
        assert_eq!(st[0] & ts_flags::PENULT, 0);
        assert_eq!(st[1], ts(2.0));

        // a completed PAIR: text must follow, and the next time must advance
        let st = ts_state(&[ts(0.0), 5, ts(2.0), ts(2.0)], &s, true);
        assert_eq!(st[0] & ts_flags::LAST, ts_flags::LAST);
        assert_eq!(st[0] & ts_flags::PENULT, ts_flags::PENULT);
        assert_eq!(st[1], ts(2.0) + 1);
    }

    #[test]
    fn a_backwards_span_is_clamped_not_emitted_reversed() {
        let s = scale();
        // the model closing earlier than it opened is its own slip; an SRT
        // writer downstream must never see end < start
        let toks = vec![ts(4.0), 1, ts(2.0)];
        let segs = split_segments(&toks, &[], &[], &s, 0);
        assert_eq!(segs.len(), 1);
        assert!(segs[0].end >= segs[0].start);
    }
}
