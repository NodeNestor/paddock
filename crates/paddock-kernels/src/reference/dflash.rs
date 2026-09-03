//! CPU reference for Muse Glimmer's DFlash2 drafter ops.
//!
//! Single source of truth for the GPU pack's `pd_dflash_conv` (slot 459).
//! Transcribed from the published DFlash2 forward, whose own unit test states
//! the contract elementwise:
//!
//! ```text
//! for position in range(block_size):
//!     for tap in range(min(taps, position + 1)):
//!         out[:, position] += (base[tap] + delta[:, position, tap, :, None])
//!                             * hidden[:, position - tap]
//! ```

/// One side of DFlash2's grouped dynamic convolution over `[r, embd]` rows.
///
/// A depthwise convolution along the TOKEN axis whose per-tap coefficient is a
/// per-channel static (`base`) plus a per-token, per-GROUP delta. `group_size`
/// adjacent channels share one delta, so the dynamic half is `embd/group_size`
/// numbers per token per tap.
///
/// Layouts:
/// - `h`, `out`: `[r, embd]` row-major.
/// - `base`: the whole `[2][taps][embd]` kernel; `side` selects a half.
/// - `delta`: `[r][2][taps][num_groups]` row-major - one projection row feeds
///   both wraps of a sublayer, which is why `side` indexes it too.
///
/// `rows_per_block` is the block the mask is taken modulo. Rows are packed one
/// block per sequence back to back, so this is what stops tap `t` at a block's
/// leading rows from reaching into the previous sequence's trailing rows: at
/// in-block position `p`, only taps `0..=p` contribute.
#[allow(clippy::too_many_arguments)]
pub fn grouped_conv(
    h: &[f32],
    out: &mut [f32],
    base: &[f32],
    delta: &[f32],
    side: usize,
    embd: usize,
    taps: usize,
    group_size: usize,
    rows_per_block: usize,
    r: usize,
) {
    assert_eq!(embd % group_size, 0, "groups must tile the channels");
    let ng = embd / group_size;
    let dstride = 2 * taps * ng;
    assert!(side < 2, "side is the before/after wrap");
    assert!(h.len() >= r * embd && out.len() >= r * embd);
    assert!(base.len() >= 2 * taps * embd);
    assert!(delta.len() >= r * dstride);
    for row in 0..r {
        let pos = row % rows_per_block;
        for c in 0..embd {
            let g = c / group_size;
            let mut acc = 0.0f32;
            for t in 0..taps.min(pos + 1) {
                let b = base[embd * (t + taps * side) + c];
                let d = delta[row * dstride + (side * taps + t) * ng + g];
                acc += (b + d) * h[(row - t) * embd + c];
            }
            out[row * embd + c] = acc;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tap 0 alone with base 1 / delta 0 is the identity, and the block mask
    /// is invisible there (position 0 already admits tap 0).
    #[test]
    fn single_tap_unit_kernel_is_identity() {
        let (embd, r) = (8usize, 4usize);
        let h: Vec<f32> = (0..r * embd).map(|i| i as f32).collect();
        let base = vec![1.0f32; 2 * embd];
        let delta = vec![0.0f32; (r * 2) * (embd / 4)];
        let mut out = vec![0.0f32; r * embd];
        grouped_conv(&h, &mut out, &base, &delta, 0, embd, 1, 4, r, r);
        assert_eq!(out, h);
    }

    /// The mask is the whole point: with a 2-tap kernel over blocks of 2, the
    /// leading row of every block must ignore the row physically before it,
    /// because that row belongs to another sequence.
    #[test]
    fn tap_never_crosses_a_block_boundary() {
        let (embd, gs, taps, rows, nblk) = (4usize, 4usize, 2usize, 2usize, 3usize);
        let r = rows * nblk;
        let h: Vec<f32> = (0..r * embd).map(|i| (i + 1) as f32).collect();
        // base = 0 for tap 0, 1 for tap 1 => output is PURELY the predecessor.
        let mut base = vec![0.0f32; 2 * taps * embd];
        for c in 0..embd {
            base[embd + c] = 1.0;
        }
        let delta = vec![0.0f32; r * 2 * taps * (embd / gs)];
        let mut out = vec![0.0f32; r * embd];
        grouped_conv(&h, &mut out, &base, &delta, 0, embd, taps, gs, rows, r);
        for b in 0..nblk {
            let lead = b * rows;
            // block-leading row: no in-block predecessor => zero, not the
            // previous block's trailing row.
            for c in 0..embd {
                assert_eq!(out[lead * embd + c], 0.0, "block {b} leaked a tap");
            }
            // second row of the block: exactly its in-block predecessor.
            for c in 0..embd {
                assert_eq!(out[(lead + 1) * embd + c], h[lead * embd + c]);
            }
        }
    }

    /// Deltas are per GROUP, so channels inside one group share a coefficient
    /// and channels in different groups do not.
    #[test]
    fn delta_is_shared_within_a_group_only() {
        let (embd, gs, taps, r) = (8usize, 4usize, 1usize, 1usize);
        let ng = embd / gs;
        let h = vec![1.0f32; embd];
        let base = vec![0.0f32; 2 * taps * embd];
        let mut delta = vec![0.0f32; r * 2 * taps * ng];
        delta[0] = 3.0; // group 0
        delta[1] = 5.0; // group 1
        let mut out = vec![0.0f32; embd];
        grouped_conv(&h, &mut out, &base, &delta, 0, embd, taps, gs, r, r);
        assert_eq!(&out[..gs], &[3.0; 4]);
        assert_eq!(&out[gs..], &[5.0; 4]);
    }

    /// `side` must select a different half of both the base kernel and the
    /// projection row - one projection feeds the before and after wraps.
    #[test]
    fn side_selects_both_base_and_delta_halves() {
        let (embd, gs, taps, r) = (4usize, 4usize, 1usize, 1usize);
        let h = vec![1.0f32; embd];
        let mut base = vec![0.0f32; 2 * taps * embd];
        base[0] = 1.0; // side 0, tap 0, channel 0
        base[taps * embd] = 7.0; // side 1, tap 0, channel 0
        let mut delta = vec![0.0f32; r * 2 * taps * (embd / gs)];
        delta[0] = 0.5; // side 0
        delta[1] = 0.25; // side 1
        let mut o0 = vec![0.0f32; embd];
        let mut o1 = vec![0.0f32; embd];
        grouped_conv(&h, &mut o0, &base, &delta, 0, embd, taps, gs, r, r);
        grouped_conv(&h, &mut o1, &base, &delta, 1, embd, taps, gs, r, r);
        assert_eq!(o0[0], 1.5);
        assert_eq!(o1[0], 7.25);
    }
}

/// DFlash2's candidate-selector walk - the CPU truth for `pd_dflash_select`.
///
/// v1 drafts each row by an independent argmax, so a block can be row-wise
/// plausible and jointly incoherent. The selector scores an EDGE from the
/// candidate actually taken one row back:
///
/// ```text
/// edge[c] = epilogue(logit[c]) + sum_r pred[prev][r] * hidden[r] * succ[c][r]
/// ```
///
/// and walks GREEDILY forward - argmax the edge row, emit that candidate,
/// carry its index. Not Viterbi: there is no backtrace and no max over paths,
/// which is why only one row of the KxK matrix is ever needed.
///
/// Row 0 of each block is the committed anchor (its logits are not a draft),
/// so the walk covers positions `1..rows` and the first predecessor is the
/// anchor's own codebook row, which callers park at `pred[(r*k + block)*rank]`.
///
/// The epilogue (`scale` then `cap`) is not optional here. Greedy per-row
/// drafting may skip it because both halves are monotone, but the unary is
/// ADDED to a bilinear term and addition does not commute with a softcap.
#[allow(clippy::too_many_arguments)]
pub fn select_walk(
    cand_ids: &[u32],
    cand_logits: &[f32],
    pred: &[f32],
    succ: &[f32],
    hs: &[f32],
    out: &mut [u32],
    scale: f32,
    cap: f32,
    rank: usize,
    k: usize,
    rows_per_block: usize,
    r: usize,
) {
    assert!(
        rows_per_block >= 2,
        "a block with no mask rows drafts nothing"
    );
    assert_eq!(r % rows_per_block, 0);
    let nk = r * k;
    for b in 0..r / rows_per_block {
        let mut previous = 0usize;
        for j in 1..rows_per_block {
            let row = b * rows_per_block + j;
            let pe = if j == 1 {
                (nk + b) * rank
            } else {
                ((row - 1) * k + previous) * rank
            };
            let mut best = 0usize;
            let mut best_v = f32::NEG_INFINITY;
            for c in 0..k {
                let se = (row * k + c) * rank;
                let mut acc = 0.0f32;
                for t in 0..rank {
                    acc += pred[pe + t] * hs[row * rank + t] * succ[se + t];
                }
                let mut u = cand_logits[row * k + c] * scale;
                if cap > 0.0 {
                    u = (u / cap).tanh() * cap;
                }
                let v = u + acc;
                if v > best_v {
                    best_v = v;
                    best = c;
                }
            }
            out[row] = cand_ids[row * k + best];
            previous = best;
        }
    }
}

#[cfg(test)]
mod select_tests {
    use super::*;

    /// With both codebooks zero the bilinear term vanishes and the walk must
    /// degenerate to exactly what v1 does: per-row argmax of the logits.
    #[test]
    fn zero_codebooks_degenerate_to_per_row_argmax() {
        let (rank, k, rows, r) = (32usize, 4usize, 4usize, 8usize);
        let ids: Vec<u32> = (0..r * k).map(|i| (i * 7 % 1000) as u32).collect();
        let logits: Vec<f32> = (0..r * k).map(|i| ((i * 13 % 17) as f32) - 8.0).collect();
        let pred = vec![0.0f32; (r * k + r / rows) * rank];
        let succ = vec![0.0f32; r * k * rank];
        let hs = vec![1.0f32; r * rank];
        let mut out = vec![0u32; r];
        select_walk(
            &ids, &logits, &pred, &succ, &hs, &mut out, 1.0, 0.0, rank, k, rows, r,
        );
        for b in 0..r / rows {
            for j in 1..rows {
                let row = b * rows + j;
                let (mut bi, mut bv) = (0usize, f32::NEG_INFINITY);
                for c in 0..k {
                    if logits[row * k + c] > bv {
                        bv = logits[row * k + c];
                        bi = c;
                    }
                }
                assert_eq!(out[row], ids[row * k + bi], "row {row}");
            }
        }
    }

    /// The carry is what makes this a PATH: a transition score that only one
    /// predecessor can offer must steer the following row's choice.
    #[test]
    fn the_chosen_index_steers_the_next_row() {
        let (rank, k, rows, r) = (32usize, 2usize, 3usize, 3usize);
        let ids: Vec<u32> = (0..r * k).map(|i| 100 + i as u32).collect();
        // Row 1 prefers candidate 1 on the unary alone.
        let mut logits = vec![0.0f32; r * k];
        logits[k + 1] = 5.0;
        let mut pred = vec![0.0f32; (r * k + 1) * rank];
        let mut succ = vec![0.0f32; r * k * rank];
        let hs = vec![1.0f32; r * rank];
        // Only predecessor index 1 at row 1 carries mass, and it favours
        // candidate 0 at row 2 strongly enough to beat that row's unary.
        pred[(k + 1) * rank] = 1.0;
        succ[(2 * k) * rank] = 9.0;
        logits[2 * k + 1] = 1.0;
        let mut out = vec![0u32; r];
        select_walk(
            &ids, &logits, &pred, &succ, &hs, &mut out, 1.0, 0.0, rank, k, rows, r,
        );
        assert_eq!(out[1], ids[k + 1], "row 1 takes the unary winner");
        assert_eq!(
            out[2],
            ids[2 * k],
            "row 2 must follow the EDGE, not its unary"
        );
    }

    /// The softcap does not commute with the addition, so it has to be applied
    /// to the unary before the bilinear term joins it.
    #[test]
    fn epilogue_is_applied_before_the_edge_sum() {
        let (rank, k, rows, r) = (32usize, 2usize, 2usize, 2usize);
        let ids = vec![7u32, 8, 9, 10];
        // Raw logits far apart, but the cap squashes them together so a small
        // bilinear term can flip the winner. Without the cap it cannot.
        let logits = vec![0.0f32, 0.0, 1000.0, 900.0];
        let mut pred = vec![0.0f32; (r * k + 1) * rank];
        let mut succ = vec![0.0f32; r * k * rank];
        let hs = vec![1.0f32; r * rank];
        pred[(r * k) * rank] = 1.0; // anchor row
        succ[(k + 1) * rank] = 5.0; // favours candidate 1 at row 1
        let mut capped = vec![0u32; r];
        select_walk(
            &ids,
            &logits,
            &pred,
            &succ,
            &hs,
            &mut capped,
            1.0,
            20.0,
            rank,
            k,
            rows,
            r,
        );
        let mut raw = vec![0u32; r];
        select_walk(
            &ids, &logits, &pred, &succ, &hs, &mut raw, 1.0, 0.0, rank, k, rows, r,
        );
        assert_eq!(capped[1], ids[k + 1], "capped unaries let the edge decide");
        assert_eq!(raw[1], ids[k], "uncapped, the raw logit gap dominates");
    }
}
