//! Word-level timing from the decoder's cross-attention.
//!
//! Whisper has no frame-level emission head, so nothing in the model says when
//! a word was spoken - its `<|0.00|>` tokens are a 20 ms grid of SEGMENT
//! boundaries and stop there. What it does have is cross-attention: when the
//! decoder emits a token, some of its heads look at the encoder frames for the
//! audio at that moment. The times are therefore RECOVERED, not computed.
//!
//! This module is the whole post-pass, and it is deliberately pure: given the
//! captured attention it returns a frame index per token, with no device and no
//! model in sight. That is what makes it testable against hand-worked cases.
//!
//! Pipeline (OpenAI `whisper/timing.py` is the reference; techniques studied
//! there and in whisper.cpp's `src/whisper.cpp`, implementation ours):
//!
//!   1. keep only the ALIGNMENT HEADS - see the table below
//!   2. normalise each head over the TOKEN axis
//!   3. median-filter width 7 over the TIME axis (reflect padding)
//!   4. mean over heads, negate -> a cost matrix
//!   5. DTW with a monotonic path; the frame where the path ENTERS a token is
//!      that token's start
//!
//! Step 5's monotonicity is the part that turns a noisy attention map into
//! something usable: tokens must go forward and time must go forward, so the
//! result is always a physically possible reading.
//!
//! KNOWN CEILING, measured:
//! whisper's BPE puts the leading space inside the word token (only ~13% of
//! spaces reach the explicit space token), so a token spans silence and speech
//! and its boundary is smeared. Word STARTS therefore bias early, into the
//! preceding pause. Fixing that needs a retrained tokenizer, which is out of
//! scope; reporting these as tighter than they are is not.

/// One cross-attention head, as `(decoder layer, head)`.
pub type Ahead = (usize, usize);

/// OpenAI's alignment heads per released model, decoded from the base85 masks
/// whisper ships. Taken from whisper.cpp's readable transcription of the same
/// data (`g_aheads_*`, MIT) - the values are OpenAI's, the selection code below
/// is ours.
///
/// These are the heads OpenAI found to be temporally aligned. Averaging all
/// heads instead gives mush: most are doing something other than tracking time.
const TINY_EN: &[Ahead] = &[
    (1, 0),
    (2, 0),
    (2, 5),
    (3, 0),
    (3, 1),
    (3, 2),
    (3, 3),
    (3, 4),
];
const TINY: &[Ahead] = &[(2, 2), (3, 0), (3, 2), (3, 3), (3, 4), (3, 5)];
const BASE_EN: &[Ahead] = &[(3, 3), (4, 7), (5, 1), (5, 5), (5, 7)];
const BASE: &[Ahead] = &[
    (3, 1),
    (4, 2),
    (4, 3),
    (4, 7),
    (5, 1),
    (5, 2),
    (5, 4),
    (5, 6),
];
const SMALL_EN: &[Ahead] = &[
    (6, 6),
    (7, 0),
    (7, 3),
    (7, 8),
    (8, 2),
    (8, 5),
    (8, 7),
    (9, 0),
    (9, 4),
    (9, 8),
    (9, 10),
    (10, 0),
    (10, 1),
    (10, 2),
    (10, 3),
    (10, 6),
    (10, 11),
    (11, 2),
    (11, 4),
];
const SMALL: &[Ahead] = &[
    (5, 3),
    (5, 9),
    (8, 0),
    (8, 4),
    (8, 7),
    (8, 8),
    (9, 0),
    (9, 7),
    (9, 9),
    (10, 5),
];
const MEDIUM_EN: &[Ahead] = &[
    (11, 4),
    (14, 1),
    (14, 12),
    (14, 14),
    (15, 4),
    (16, 0),
    (16, 4),
    (16, 9),
    (17, 12),
    (17, 14),
    (18, 7),
    (18, 10),
    (18, 15),
    (20, 0),
    (20, 3),
    (20, 9),
    (20, 14),
    (21, 12),
];
const MEDIUM: &[Ahead] = &[(13, 15), (15, 4), (15, 15), (16, 1), (20, 0), (23, 4)];
const LARGE_V2: &[Ahead] = &[
    (10, 12),
    (13, 17),
    (16, 11),
    (16, 12),
    (16, 13),
    (17, 15),
    (17, 16),
    (18, 4),
    (18, 11),
    (18, 19),
    (19, 11),
    (21, 2),
    (21, 3),
    (22, 3),
    (22, 9),
    (22, 12),
    (23, 5),
    (23, 7),
    (23, 13),
    (25, 5),
    (26, 1),
    (26, 12),
    (27, 15),
];
const LARGE_V3: &[Ahead] = &[
    (7, 0),
    (10, 17),
    (12, 18),
    (13, 12),
    (16, 1),
    (17, 14),
    (19, 11),
    (21, 4),
    (24, 1),
    (25, 6),
];
const LARGE_V3_TURBO: &[Ahead] = &[(2, 4), (2, 11), (3, 3), (3, 6), (3, 11), (3, 14)];

/// Which heads to align with, and where the choice came from.
#[derive(Clone, Debug, PartialEq)]
pub struct Heads {
    /// what this set is, for the log and the honest-provenance rule
    pub source: &'static str,
    pub heads: Vec<Ahead>,
}

/// The decoder geometry that identifies a released whisper.
///
/// A FINE-TUNE inherits its base model's shape, which is exactly why this keys
/// on geometry rather than on a name: KB-Whisper is large-v3-shaped and gets
/// large-v3's heads without knowing its own name.
///
/// `n_vocab` separates the English-only models (51864) from the multilingual
/// ones, and `n_mels` separates large-v3 (128) from v1/v2 (80) - the two large
/// generations are otherwise identical in shape.
pub fn heads_for(n_layer: usize, n_head: usize, n_mels: usize, n_vocab: usize) -> Option<Heads> {
    let english_only = n_vocab <= 51864;
    let pick = |name: &'static str, h: &'static [Ahead]| {
        Some(Heads {
            source: name,
            heads: h.to_vec(),
        })
    };
    match (n_layer, n_head, n_mels) {
        (4, 6, _) if english_only => pick("tiny.en", TINY_EN),
        (4, 6, _) => pick("tiny", TINY),
        (6, 8, _) if english_only => pick("base.en", BASE_EN),
        (6, 8, _) => pick("base", BASE),
        (12, 12, _) if english_only => pick("small.en", SMALL_EN),
        (12, 12, _) => pick("small", SMALL),
        (24, 16, _) if english_only => pick("medium.en", MEDIUM_EN),
        (24, 16, _) => pick("medium", MEDIUM),
        (4, 20, _) => pick("large-v3-turbo", LARGE_V3_TURBO),
        (32, 20, 128) => pick("large-v3", LARGE_V3),
        // v1 and v2 are the same geometry (32/20/80 mels, same vocab) and
        // cannot be told apart from the file's shape. v2 is what the world
        // actually runs, so it is the default and the ambiguity is stated
        // rather than hidden; a v1 file wanting its own heads overrides.
        (32, 20, _) => pick("large-v2 (v1 is indistinguishable by shape)", LARGE_V2),
        _ => None,
    }
}

/// The honest answer for a checkpoint whose shape matches nothing known: every
/// head of the top few layers.
///
/// Degraded deliberately, and a degradation worth having - the late decoder
/// layers are where temporal alignment concentrates, so this lands in the right
/// neighbourhood without claiming to know which heads. It costs more attention
/// to capture, which is the price of not knowing. Refusing instead would make
/// word timing simply unavailable for every fine-tune of an unusual size.
pub fn fallback_heads(n_layer: usize, n_head: usize) -> Heads {
    let top = (n_layer / 8).max(1);
    let heads = (n_layer - top..n_layer)
        .flat_map(|l| (0..n_head).map(move |h| (l, h)))
        .collect::<Vec<_>>();
    Heads {
        source: "top layers (no head table for this shape)",
        heads,
    }
}

/// Median filter along the last axis with reflect padding, in place per row.
///
/// Reflect, not edge-clamp: `[a,b,c]` widened reads `c,b,a,b,c,b,a`, which is
/// what the reference does and what keeps a spike at the boundary from being
/// held for `width/2` samples by a repeated edge value.
///
/// `width` must be odd - an even window has no middle element and the "median"
/// would silently become one of the two central values.
pub fn median_filter_rows(x: &mut [f32], rows: usize, cols: usize, width: usize) {
    debug_assert_eq!(width % 2, 1, "median filter width must be odd");
    if width <= 1 || cols == 0 || cols <= width / 2 {
        return;
    }
    let half = (width / 2) as isize;
    let n = cols as isize;
    let mut win: Vec<f32> = Vec::with_capacity(width);
    let mut out = vec![0.0f32; cols];
    for r in 0..rows {
        let row = &x[r * cols..(r + 1) * cols];
        for c in 0..cols {
            win.clear();
            for off in -half..=half {
                let mut i = c as isize + off;
                // reflect without repeating the edge sample
                if i < 0 {
                    i = -i;
                } else if i >= n {
                    i = 2 * (n - 1) - i;
                }
                win.push(row[i as usize]);
            }
            win.sort_by(f32::total_cmp);
            out[c] = win[win.len() / 2];
        }
        x[r * cols..(r + 1) * cols].copy_from_slice(&out);
    }
}

/// Normalise each head over the TOKEN axis: for every (head, frame), subtract
/// the mean over tokens and divide by the population standard deviation.
///
/// Over TOKENS, not over time - this is asking "for this frame, which tokens
/// stand out", which is the question the alignment is about. Population (biased)
/// variance, matching the reference's `unbiased=False`.
///
/// `w` is `[heads][tokens][frames]`.
pub fn normalize_over_tokens(w: &mut [f32], n_head: usize, n_tok: usize, n_frame: usize) {
    if n_tok == 0 || n_frame == 0 {
        return;
    }
    for h in 0..n_head {
        let base = h * n_tok * n_frame;
        for f in 0..n_frame {
            let mut mean = 0.0f64;
            for t in 0..n_tok {
                mean += w[base + t * n_frame + f] as f64;
            }
            mean /= n_tok as f64;
            let mut var = 0.0f64;
            for t in 0..n_tok {
                let d = w[base + t * n_frame + f] as f64 - mean;
                var += d * d;
            }
            var /= n_tok as f64;
            // a frame every token treats identically carries no alignment
            // signal; 1e-9 keeps it at zero instead of exploding it
            let inv = 1.0 / (var.sqrt() + 1e-9);
            for t in 0..n_tok {
                let v = &mut w[base + t * n_frame + f];
                *v = ((*v as f64 - mean) * inv) as f32;
            }
        }
    }
}

/// Mean over heads, negated - attention becomes a COST, so DTW's cheapest path
/// is the most-attended one. `[heads][tokens][frames]` -> `[tokens][frames]`.
pub fn mean_heads_negated(w: &[f32], n_head: usize, n_tok: usize, n_frame: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n_tok * n_frame];
    if n_head == 0 {
        return out;
    }
    for h in 0..n_head {
        let base = h * n_tok * n_frame;
        for i in 0..n_tok * n_frame {
            out[i] += w[base + i];
        }
    }
    let k = -1.0 / n_head as f32;
    for v in out.iter_mut() {
        *v *= k;
    }
    out
}

/// DTW over `cost[tokens][frames]`, returning the monotonic path as
/// `(token, frame)` pairs from the start of the audio to the end.
///
/// The tie rule is the reference's and is not a plain three-way minimum:
/// diagonal only when strictly below both others, then token-advance when
/// strictly below both, else frame-advance. Ties therefore fall to
/// frame-advance, which keeps a token stretched over silence rather than
/// racing ahead of it. Writing the obvious `min` here would drift from every
/// other implementation on exactly the flat regions that need a rule.
pub fn dtw_path(cost: &[f32], n_tok: usize, n_frame: usize) -> Vec<(usize, usize)> {
    if n_tok == 0 || n_frame == 0 {
        return Vec::new();
    }
    let w = n_frame + 1;
    let mut acc = vec![f32::INFINITY; (n_tok + 1) * w];
    // 0 = diagonal, 1 = token advances, 2 = frame advances
    let mut trace = vec![2u8; (n_tok + 1) * w];
    acc[0] = 0.0;
    for i in 0..=n_tok {
        trace[i * w] = 1;
    }
    for j in 0..=n_frame {
        trace[j] = 2;
    }
    for i in 1..=n_tok {
        for j in 1..=n_frame {
            let c0 = acc[(i - 1) * w + (j - 1)];
            let c1 = acc[(i - 1) * w + j];
            let c2 = acc[i * w + (j - 1)];
            let (c, t) = if c0 < c1 && c0 < c2 {
                (c0, 0u8)
            } else if c1 < c0 && c1 < c2 {
                (c1, 1u8)
            } else {
                (c2, 2u8)
            };
            acc[i * w + j] = cost[(i - 1) * n_frame + (j - 1)] + c;
            trace[i * w + j] = t;
        }
    }
    let mut path = Vec::with_capacity(n_tok + n_frame);
    let (mut i, mut j) = (n_tok, n_frame);
    while i > 0 || j > 0 {
        path.push((i - 1, j - 1));
        match trace[i * w + j] {
            0 => {
                i -= 1;
                j -= 1;
            }
            1 => i -= 1,
            _ => j -= 1,
        }
    }
    path.reverse();
    path
}

/// The frame each token STARTS at: where the path first enters it.
///
/// Start, not end. whisper.cpp assigns each token the frame where the path
/// LEAVES it (so token i carries token i+1's arrival and the last token gets
/// nothing); OpenAI takes the arrival frame, which is the token's own start and
/// is what word spans are built from. Followed OpenAI here.
///
/// A token the path skipped over entirely - possible when the audio is shorter
/// than the transcript claims - inherits the previous token's frame rather than
/// being dropped, so the result is always parallel to the token list.
pub fn token_start_frames(path: &[(usize, usize)], n_tok: usize) -> Vec<usize> {
    let mut out = vec![0usize; n_tok];
    let mut seen = vec![false; n_tok];
    for &(t, f) in path {
        if t < n_tok && !seen[t] {
            seen[t] = true;
            out[t] = f;
        }
    }
    let mut last = 0usize;
    for t in 0..n_tok {
        if seen[t] {
            last = out[t];
        } else {
            out[t] = last;
        }
    }
    out
}

/// Seconds per encoder frame: whisper's 10 ms hop, halved by the encoder's
/// stride-2 conv stack.
pub const SECONDS_PER_FRAME: f32 = 0.02;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_finetune_inherits_its_base_models_heads_by_shape() {
        // KB-Whisper large is a large-v3 fine-tune: same geometry, 128 mels,
        // multilingual vocab. It must resolve without knowing its own name -
        // that is the whole reason this keys on shape.
        let h = heads_for(32, 20, 128, 51866).expect("large-v3 shape");
        assert_eq!(h.source, "large-v3");
        assert_eq!(h.heads.len(), 10);
        assert_eq!(h.heads[0], (7, 0));

        // 80 mels at the same shape is the older generation
        assert!(
            heads_for(32, 20, 80, 51865)
                .unwrap()
                .source
                .starts_with("large-v2")
        );
        // the English-only models share their multilingual twin's geometry and
        // are told apart by vocabulary alone
        assert_eq!(heads_for(12, 12, 80, 51864).unwrap().source, "small.en");
        assert_eq!(heads_for(12, 12, 80, 51865).unwrap().source, "small");
        // and a shape nobody ships has no table
        assert!(heads_for(9, 7, 80, 51865).is_none());
    }

    #[test]
    fn the_fallback_takes_whole_layers_off_the_top() {
        let h = fallback_heads(32, 20);
        assert_eq!(h.heads.len(), 4 * 20);
        assert!(h.heads.iter().all(|&(l, _)| l >= 28));
        // never empty, however shallow the decoder
        assert_eq!(fallback_heads(4, 6).heads.len(), 6);
    }

    #[test]
    fn median_filter_reflects_rather_than_repeating_the_edge() {
        // a lone spike at the edge must not survive: with reflect padding the
        // window at column 0 is {c,b,a,b,c} of the real samples, so the spike
        // is outvoted. Edge-clamp padding would repeat it and keep it.
        let mut x = vec![9.0, 0.0, 0.0, 0.0, 0.0];
        median_filter_rows(&mut x, 1, 5, 3);
        assert_eq!(
            x,
            vec![0.0, 0.0, 0.0, 0.0, 0.0],
            "spike survived the filter"
        );

        // a plateau is left alone
        let mut flat = vec![2.0; 6];
        median_filter_rows(&mut flat, 1, 6, 5);
        assert_eq!(flat, vec![2.0; 6]);

        // rows are independent
        let mut two = vec![1.0, 1.0, 9.0, 7.0, 7.0, 7.0];
        median_filter_rows(&mut two, 2, 3, 3);
        assert_eq!(two, vec![1.0, 1.0, 1.0, 7.0, 7.0, 7.0]);
    }

    #[test]
    fn the_filter_erases_features_narrower_than_its_own_majority() {
        // Worth pinning because it bounds what the whole pass can resolve: a
        // width-7 median keeps a value only where it holds 4 of 7 samples, so
        // an attention band under 4 frames is removed outright. Real bands are
        // wider than that, but a synthetic test with narrow bands would
        // "fail" for this reason and look like a DTW bug.
        let mut narrow = vec![0.0f32; 12];
        narrow[4..7].copy_from_slice(&[1.0, 1.0, 1.0]); // 3 frames
        median_filter_rows(&mut narrow, 1, 12, 7);
        assert!(
            narrow.iter().all(|v| *v == 0.0),
            "a 3-frame band survived a width-7 median"
        );

        let mut wide = vec![0.0f32; 12];
        wide[3..8].copy_from_slice(&[1.0; 5]); // 5 frames
        median_filter_rows(&mut wide, 1, 12, 7);
        assert!(wide.contains(&1.0), "a 5-frame band was erased");
    }

    #[test]
    fn normalisation_runs_over_tokens_not_over_time() {
        // One head, 2 tokens, 2 frames. Token 0 attends frame 0, token 1
        // attends frame 1. Normalising over TOKENS makes each frame's column
        // zero-mean, so the attending token goes positive and the other
        // negative - which is the contrast DTW then follows.
        let mut w = vec![
            1.0, 0.0, // token 0
            0.0, 1.0, // token 1
        ];
        normalize_over_tokens(&mut w, 1, 2, 2);
        assert!(
            w[0] > 0.0 && w[2] < 0.0,
            "frame 0 column not centred: {w:?}"
        );
        assert!(
            w[3] > 0.0 && w[1] < 0.0,
            "frame 1 column not centred: {w:?}"
        );
        // symmetric input, symmetric output
        assert!((w[0] + w[2]).abs() < 1e-5);

        // a frame every token treats identically contributes nothing rather
        // than dividing by ~0 and exploding
        let mut flat = vec![0.5, 0.5];
        normalize_over_tokens(&mut flat, 1, 2, 1);
        assert!(
            flat.iter().all(|v| v.abs() < 1e-3),
            "constant column blew up: {flat:?}"
        );
    }

    #[test]
    fn dtw_follows_a_diagonal_of_attention() {
        // 3 tokens over 12 frames, each attending a 4-frame band.
        let n_tok = 3;
        let n_frame = 12;
        let band = 4;
        let mut cost = vec![0.0f32; n_tok * n_frame];
        for t in 0..n_tok {
            for f in t * band..(t + 1) * band {
                cost[t * n_frame + f] = -1.0;
            }
        }
        let path = dtw_path(&cost, n_tok, n_frame);
        assert!(!path.is_empty());
        // monotonic in both axes, which is the property that makes the answer
        // physically possible at all - a word cannot be spoken before the one
        // in front of it
        for pair in path.windows(2) {
            assert!(
                pair[1].0 >= pair[0].0 && pair[1].1 >= pair[0].1,
                "path went backwards"
            );
        }
        // it spans the whole matrix: every token placed, every frame accounted
        assert_eq!(path.first().unwrap(), &(0, 0));
        assert_eq!(path.last().unwrap(), &(n_tok - 1, n_frame - 1));

        // Each token starts within a frame of its planted band. Not exact, and
        // the slack is inherent rather than sloppy: outside the bands the cost
        // is flat, and the reference's tie rule resolves flat regions toward
        // frame-advance, which lets a token be entered one frame before its
        // own evidence begins. Every implementation of this shares the
        // behaviour, and it biases starts EARLY - the same direction as the
        // tokenizer defect in the module note.
        let starts = token_start_frames(&path, n_tok);
        for (t, &f) in starts.iter().enumerate() {
            let want = t * band;
            assert!(
                f + 1 >= want && f <= want + 1,
                "token {t} started at frame {f}, planted at {want}"
            );
        }
    }

    #[test]
    fn every_token_gets_a_frame_even_when_the_path_skips_it() {
        // A path that jumps 0 -> 2 leaves token 1 unvisited. It inherits the
        // previous frame instead of vanishing: the output has to stay parallel
        // to the token list or the caller reads the wrong word's time.
        let path = vec![(0usize, 0usize), (0, 1), (2, 2), (2, 3)];
        let starts = token_start_frames(&path, 3);
        assert_eq!(starts, vec![0, 0, 2]);
    }

    #[test]
    fn the_whole_pass_recovers_a_planted_alignment() {
        // End to end on synthetic attention: two heads, one of them carrying
        // nothing, four tokens each attending a distinct 8-frame band. The real
        // head has to carry the answer through normalisation, a width-7 median
        // and the averaging that the dead head is diluting. Bands are 8 wide
        // because the filter erases anything under 4 - see the test above.
        let (n_head, n_tok, band) = (2, 4, 8);
        let n_frame = n_tok * band;
        let mut w = vec![0.0f32; n_head * n_tok * n_frame];
        for t in 0..n_tok {
            for f in t * band..(t + 1) * band {
                w[t * n_frame + f] = 1.0; // head 0: the aligned one
            }
        }
        // head 1: a constant, i.e. no information at any frame
        for i in 0..n_tok * n_frame {
            w[n_tok * n_frame + i] = 0.3;
        }
        normalize_over_tokens(&mut w, n_head, n_tok, n_frame);
        median_filter_rows(&mut w, n_head * n_tok, n_frame, 7);
        let cost = mean_heads_negated(&w, n_head, n_tok, n_frame);
        let path = dtw_path(&cost, n_tok, n_frame);
        let starts = token_start_frames(&path, n_tok);
        for (t, &f) in starts.iter().enumerate() {
            let want = t * band;
            let off = (f as isize - want as isize).abs();
            assert!(
                off <= 1,
                "token {t} started at frame {f}, planted at {want}"
            );
        }
    }
}
