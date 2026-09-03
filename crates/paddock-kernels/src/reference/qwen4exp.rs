//! CPU reference for the Qwen3.8-Flash-Next (`qwen4_exp`) new-math ops.
//!
//! Ground truth for the GPU graph, source-verified against both upstream
//! implementations - HF transformers `modular_qwen4_exp.py` and the vLLM
//! PR#53899 nvidia
//! package (`hyperconnection.py`, `ops/hc.py`, `ple_layer.py`,
//! `qwen3_next._project_qkv_gate`) - which agree on every formula below.
//! f32 plain loops, auditability over speed.
//!
//! What is not here (reused instead): the GDN core recurrence
//! (`reference::delta_net::gated_delta_recurrent` - identical gated delta
//! rule, L2-norm + scale inside), gated attention
//! (`reference::qwen35_attn::gated_attention_core` - same per-head
//! q|gate interleave; pass `w+1` for the Gemma (1+w) norms since qwen4_exp
//! checkpoints store raw w), rms_norm/swiglu/rope (`reference::ops`).

/// Grouped Gemma RMSNorm, (1+w) affine: each of `groups` equal slices of `x`
/// is normalized by its own RMS, then the FULL-width weight applies as
/// `y = xn + xn*w` (the FMA form vLLM's kernel uses). In-place.
pub fn group_rms_norm_1p(x: &mut [f32], w: &[f32], groups: usize, eps: f32) {
    assert_eq!(x.len(), w.len());
    assert_eq!(x.len() % groups, 0);
    let gd = x.len() / groups;
    for g in 0..groups {
        let s = g * gd;
        let ms = x[s..s + gd].iter().map(|v| v * v).sum::<f32>() / gd as f32;
        let rrms = 1.0 / (ms + eps).sqrt();
        for i in s..s + gd {
            let xn = x[i] * rrms;
            x[i] = xn + xn * w[i];
        }
    }
}

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// softplus with the numerical guard HF uses (`log1p(exp(x))`, linear tail).
pub fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { x.exp().ln_1p() }
}

/// Row-major matvec: `w [rows, k]`, `x [k]` -> `[rows]`.
pub fn matvec(w: &[f32], x: &[f32], rows: usize, k: usize) -> Vec<f32> {
    assert_eq!(w.len(), rows * k);
    assert_eq!(x.len(), k);
    (0..rows)
        .map(|r| {
            let row = &w[r * k..(r + 1) * k];
            row.iter().zip(x).map(|(a, b)| a * b).sum()
        })
        .collect()
}

/// One hyper-connection mix for one token (doc §3.2 / vLLM `GatedResidual.mix`):
///
/// ```text
///   Hn        = group_rms_norm_1p(H)                    // [hc*hidden]
///   m         = silu(W_down·Hn / hc)                    // [lowrank]
///   gate      = W_up·m                                  // [hc*hidden]
///   x_in[d]   = Σ_s sigmoid(gate[s,d]) · Hn[s,d] / hc   // [hidden]
///   inj       = W_inj·Hn                                // [hc] (RAW logits -
///               the 2·sigmoid(inj/hc) happens in `hc_combine`)
/// ```
/// Returns `(block_input, inject_logits)`; pass `w_inj: None` for the final
/// mixer (no injection).
#[allow(clippy::too_many_arguments)]
pub fn hc_mix(
    h: &[f32],
    norm_w: &[f32],
    w_down: &[f32],
    w_up: &[f32],
    w_inj: Option<&[f32]>,
    hc: usize,
    lowrank: usize,
    eps: f32,
) -> (Vec<f32>, Option<Vec<f32>>) {
    let hw = h.len();
    let hidden = hw / hc;
    let mut xn = h.to_vec();
    group_rms_norm_1p(&mut xn, norm_w, hc, eps);
    let mut m = matvec(w_down, &xn, lowrank, hw);
    for v in m.iter_mut() {
        let s = *v / hc as f32;
        *v = s * sigmoid(s);
    }
    let gate = matvec(w_up, &m, hw, lowrank);
    let mut block_input = vec![0f32; hidden];
    for s in 0..hc {
        for d in 0..hidden {
            block_input[d] += sigmoid(gate[s * hidden + d]) * xn[s * hidden + d];
        }
    }
    for v in block_input.iter_mut() {
        *v /= hc as f32;
    }
    let inj = w_inj.map(|wi| matvec(wi, &xn, hc, hw));
    (block_input, inj)
}

/// Hyper-connection combine (vLLM `_hc_combine_kernel`):
/// `H[s,:] += block_out · 2·sigmoid(inj[s]/hc)`.
pub fn hc_combine(h: &mut [f32], block_out: &[f32], inj: &[f32], hc: usize) {
    let hidden = h.len() / hc;
    assert_eq!(block_out.len(), hidden);
    assert_eq!(inj.len(), hc);
    for s in 0..hc {
        let w = 2.0 * sigmoid(inj[s] / hc as f32);
        for d in 0..hidden {
            h[s * hidden + d] += block_out[d] * w;
        }
    }
}

/// Depthwise causal conv1d + silu over a token sequence, optional dilation.
/// `seq [n_tokens, dim]` in place; `w [dim, k]`; positions before the start
/// read zero (fresh state). GDN uses `dilation = 1`; PLE uses `dilation = 3`
/// (a 9-token receptive ring at k = 4).
pub fn conv1d_causal_silu(
    seq: &mut [f32],
    w: &[f32],
    n_tokens: usize,
    dim: usize,
    k: usize,
    dilation: usize,
) {
    assert_eq!(seq.len(), n_tokens * dim);
    assert_eq!(w.len(), dim * k);
    let src = seq.to_vec();
    for t in 0..n_tokens {
        for d in 0..dim {
            let mut acc = 0f32;
            for j in 0..k {
                let back = (k - 1 - j) * dilation;
                if t >= back {
                    acc += w[d * k + j] * src[(t - back) * dim + d];
                }
            }
            seq[t * dim + d] = acc * sigmoid(acc);
        }
    }
}

/// GDN decay/correction gates (doc §3.4): `g = -exp(A_log)·softplus(a + dt_bias)`,
/// `beta = sigmoid(b)`. `ax`/`bx`: `[n_tokens, heads]` projections.
pub fn gdn_gates(
    ax: &[f32],
    bx: &[f32],
    a_log: &[f32],
    dt_bias: &[f32],
    n_tokens: usize,
    heads: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut g = vec![0f32; n_tokens * heads];
    let mut beta = vec![0f32; n_tokens * heads];
    for t in 0..n_tokens {
        for h in 0..heads {
            let i = t * heads + h;
            g[i] = -a_log[h].exp() * softplus(ax[i] + dt_bias[h]);
            beta[i] = sigmoid(bx[i]);
        }
    }
    (g, beta)
}

/// Gated norm on the GDN output (doc §3.1 RMSNormGated): per `head_dim` group,
/// `y = w · rms_norm(x) · sigmoid(z)` - PLAIN w (not 1+w), sigmoid gate.
pub fn gdn_gated_norm(x: &mut [f32], z: &[f32], w: &[f32], head_dim: usize, eps: f32) {
    assert_eq!(x.len(), z.len());
    assert_eq!(w.len(), head_dim);
    for (xh, zh) in x.chunks_exact_mut(head_dim).zip(z.chunks_exact(head_dim)) {
        let ms = xh.iter().map(|v| v * v).sum::<f32>() / head_dim as f32;
        let inv = 1.0 / (ms + eps).sqrt();
        for d in 0..head_dim {
            xh[d] = w[d] * (xh[d] * inv) * sigmoid(zh[d]);
        }
    }
}

/// PLE n-gram ids for one token (doc §3.3 / vLLM `forward_impl`).
///
/// `window = [cur, prev1, prev2]` after EOS-segment substitution (use
/// [`ple_window`]); `mult [3]`, `sizes/offsets [16]` are the checkpoint's I64
/// buffers. Bigram heads 0..8 mix cur/prev1; trigram heads 8..16 add prev2.
/// All i64 arithmetic WRAPS (torch semantics); remainder is non-negative.
pub fn ple_ngram_ids(
    window: &[i64; 3],
    mult: &[i64],
    sizes: &[i64],
    offsets: &[i64],
    heads_per_ngram: usize,
) -> Vec<i64> {
    let mut ids = Vec::with_capacity(2 * heads_per_ngram);
    for ngram in 2..=3usize {
        let mut mixed = window[0].wrapping_mul(mult[0]);
        for i in 1..ngram {
            mixed ^= window[i].wrapping_mul(mult[i]);
        }
        let start = (ngram - 2) * heads_per_ngram;
        for hh in 0..heads_per_ngram {
            let m = sizes[start + hh];
            ids.push(mixed.rem_euclid(m) + offsets[start + hh]);
        }
    }
    ids
}

/// EOS-segment token window for position `i` of one request's token stream
/// (with the 2-token EOS priming already prepended by the caller - vLLM's
/// `ngram_context`). A token within `shift` positions of its segment start
/// (the position after the previous EOS) reads EOS instead of the real
/// previous token.
pub fn ple_window(tokens: &[i64], i: usize, eos: i64) -> [i64; 3] {
    // position of the previous EOS at-or-before i-1 (vLLM: cummax of eos
    // positions, exclusive of i)
    let mut prev_eos: i64 = -1;
    for (j, &t) in tokens[..i].iter().enumerate() {
        if t == eos {
            prev_eos = j as i64;
        }
    }
    let pos_in_seg = i as i64 - prev_eos - 1;
    let mut w = [tokens[i], eos, eos];
    for (shift, slot) in [(1usize, 1usize), (2, 2)] {
        if i >= shift && pos_in_seg >= shift as i64 {
            w[slot] = tokens[i - shift];
        }
    }
    w
}

/// PLE per-stream gate for one token (doc §3.3 / vLLM `Qwen4ExpPLELayer.forward`):
///
/// ```text
///   K  = group_rms_norm_1p(W_key·emb)      // [hc, hidden] (caller projects)
///   Q  = group_rms_norm_1p(H)
///   gate[s] = sigmoid( signed_sqrt( (K_s·Q_s) / sqrt(hidden) ) )
///   gv[s,:] = gate[s] · V                  // V = W_val·emb, [hidden]
/// ```
/// `key`/`h` are consumed raw (norms applied inside); returns `gv [hc*hidden]`.
/// `signed_sqrt(x) = sign(x)·sqrt(max(|x|, 1e-6))`.
#[allow(clippy::too_many_arguments)]
pub fn ple_gate(
    h: &[f32],
    key: &[f32],
    value: &[f32],
    norm_key_w: &[f32],
    norm_query_w: &[f32],
    hc: usize,
    eps: f32,
) -> Vec<f32> {
    let hw = h.len();
    let hidden = hw / hc;
    assert_eq!(key.len(), hw);
    assert_eq!(value.len(), hidden);
    let mut kn = key.to_vec();
    let mut qn = h.to_vec();
    group_rms_norm_1p(&mut kn, norm_key_w, hc, eps);
    group_rms_norm_1p(&mut qn, norm_query_w, hc, eps);
    let mut gv = vec![0f32; hw];
    for s in 0..hc {
        let dot: f32 = kn[s * hidden..(s + 1) * hidden]
            .iter()
            .zip(&qn[s * hidden..(s + 1) * hidden])
            .map(|(a, b)| a * b)
            .sum();
        let raw = dot / (hidden as f32).sqrt();
        let gate = sigmoid(raw.signum() * raw.abs().max(1e-6).sqrt());
        for d in 0..hidden {
            gv[s * hidden + d] = gate * value[d];
        }
    }
    gv
}

/// e4m3 byte -> f32 (the PLE table dtype; finite e4m3fn, no inf code).
pub fn e4m3_to_f32(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let exp = ((b >> 3) & 0x0F) as i32;
    let man = (b & 0x07) as f32;
    if exp == 0 {
        // subnormal: man/8 * 2^-6
        sign * (man / 8.0) * (0.5f32).powi(6)
    } else if exp == 15 && man == 7.0 {
        f32::NAN * sign
    } else {
        sign * (1.0 + man / 8.0) * 2f32.powi(exp - 7)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn group_norm_normalizes_groups_independently() {
        // two groups with very different scales: each must come out unit-RMS
        let mut x = vec![3.0, 4.0, 300.0, 400.0];
        let w = vec![0.0; 4]; // (1+0) affine = pure normalize
        group_rms_norm_1p(&mut x, &w, 2, 0.0);
        for g in 0..2 {
            let ms: f32 = x[g * 2..g * 2 + 2].iter().map(|v| v * v).sum::<f32>() / 2.0;
            assert!((ms - 1.0).abs() < 1e-5, "group {g} rms {ms}");
        }
        assert!((x[0] - 0.6 * 2f32.sqrt()).abs() < 1e-5);
    }

    #[test]
    fn hc_combine_matches_formula() {
        let mut h = vec![1.0f32; 8]; // hc=4, hidden=2
        let inj = vec![0.0f32; 4]; // sigmoid(0)=0.5 -> weight 1.0
        hc_combine(&mut h, &[2.0, 3.0], &inj, 4);
        assert_eq!(h, vec![3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0]);
    }

    #[test]
    fn ple_window_eos_segments() {
        let eos = 9i64;
        // stream: [eos, eos, 5, 6, eos, 7]  (2-token priming + real ids)
        let t = vec![9, 9, 5, 6, 9, 7];
        assert_eq!(ple_window(&t, 2, eos), [5, 9, 9]); // segment starts at 2
        assert_eq!(ple_window(&t, 3, eos), [6, 5, 9]); // one real prev
        assert_eq!(ple_window(&t, 5, eos), [7, 9, 9]); // reset after eos at 4
    }

    #[test]
    fn ngram_ids_hand_computed() {
        let mult = [3i64, 5, 7];
        let sizes = vec![11i64, 13];
        let offsets = vec![0i64, 11];
        let ids = ple_ngram_ids(&[2, 4, 6], &mult, &sizes, &offsets, 1);
        // bigram: 2*3 ^ 4*5 = 6 ^ 20 = 18 -> 18 % 11 = 7
        // trigram: 18 ^ 6*7 = 18 ^ 42 = 56 -> 56 % 13 = 4, + offset 11 = 15
        assert_eq!(ids, vec![7, 15]);
    }

    #[test]
    fn conv_dilated_reaches_back_nine() {
        // k=4, dilation=3: output at t depends on t, t-3, t-6, t-9
        let n = 10usize;
        let mut seq = vec![0f32; n];
        seq[0] = 1.0; // only t=0 nonzero
        let w = vec![1.0f32, 0.0, 0.0, 0.0]; // picks x[t-9] only
        conv1d_causal_silu(&mut seq, &w, n, 1, 4, 3);
        // silu(1.0) lands at t=9 (t-9==0); everything else silu(0)=0
        assert!((seq[9] - 1.0 * sigmoid(1.0)).abs() < 1e-6);
        assert_eq!(seq[8], 0.0);
    }

    #[test]
    fn e4m3_known_points() {
        assert_eq!(e4m3_to_f32(0x38), 1.0); // exp 7, man 0
        assert_eq!(e4m3_to_f32(0xB8), -1.0);
        assert_eq!(e4m3_to_f32(0x40), 2.0);
        assert_eq!(e4m3_to_f32(0x00), 0.0);
        assert_eq!(e4m3_to_f32(0x7E), 448.0); // max finite e4m3fn
    }
}
