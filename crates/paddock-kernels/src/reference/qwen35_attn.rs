//! CPU reference for the Qwen3.5 gated full-attention layer (every 4th layer).
//!
//! Ground truth for the GPU chain that assembles the full-attn mixer. The one
//! genuinely new op is **partial M-RoPE** (multimodal sectioned rotary); the rest
//! reuses ops the engine already has (per-head RMSNorm, GQA softmax attention).
//! Ported to agree with llama.cpp `src/models/qwen35.cpp::build_layer_attn` and
//! HF `modeling_qwen3_5.py::Qwen3_5Attention.forward`:
//!
//!   Q,gate = split(x @ Wq)   // one joint projection, query||gate per head
//!   Q = rmsnorm_head(Q, q_norm);  K = rmsnorm_head(K, k_norm)
//!   Q = mrope(Q);  K = mrope(K)                 // partial sectioned rotary
//!   ctx = softmax(Q·Kᵀ · scale, causal) · V     // GQA
//!   out = ctx * sigmoid(gate)                    // per-element output gate
//!
//! `out` is the pre-`Wo` `core_attn_out`; the surrounding Wq/Wk/Wv/Wo are plain
//! Q8_0 GEMMs. Built to carry vision unchanged: `positions` is the full 4-axis
//! M-RoPE layout, so text (all axes equal) and vision (distinct t/h/w) share one
//! path. f32, plain loops, auditability over speed.

use super::ops::YarnRope;

/// Apply partial sectioned M-RoPE in place to Q or K activations.
///
/// `x`: `[n_tokens, n_heads, head_dim]` row-major. `positions`: `[4, n_tokens]`
/// axis-major - row 0 temporal, 1 height, 2 width, 3 extra; for text every row
/// is the token index. `sections`: per-axis rotary-pair counts `[t,h,w,e]`
/// (their sum is the section period; `2*sum` = `rope.n_dims` = n_rot). Channels
/// past `n_rot` pass through untouched.
#[allow(clippy::too_many_arguments)]
pub fn mrope(
    x: &mut [f32],
    positions: &[f32],
    sections: &[u32; 4],
    n_tokens: usize,
    n_heads: usize,
    head_dim: usize,
    rope: &YarnRope,
) {
    for t in 0..n_tokens {
        let pos = [
            positions[t],
            positions[n_tokens + t],
            positions[2 * n_tokens + t],
            positions[3 * n_tokens + t],
        ];
        for h in 0..n_heads {
            let off = (t * n_heads + h) * head_dim;
            rope.apply_mrope(&mut x[off..off + head_dim], &pos, sections);
        }
    }
}

/// Sigmoid output gate, in place: `x[i] *= sigmoid(gate[i])`.
pub fn sigmoid_gate(x: &mut [f32], gate: &[f32]) {
    for (xi, gi) in x.iter_mut().zip(gate) {
        *xi *= 1.0 / (1.0 + (-*gi).exp());
    }
}

/// Per-head RMSNorm (no bias): each `head_dim`-vector normalized by its own RMS,
/// then scaled by the shared `weight` `[head_dim]`. `x`: `[n_rows, head_dim]`.
pub fn rmsnorm_head(x: &mut [f32], weight: &[f32], n_rows: usize, head_dim: usize, eps: f32) {
    for r in 0..n_rows {
        let row = &mut x[r * head_dim..(r + 1) * head_dim];
        let mean_sq = row.iter().map(|v| v * v).sum::<f32>() / head_dim as f32;
        let inv = 1.0 / (mean_sq + eps).sqrt();
        for (v, w) in row.iter_mut().zip(weight) {
            *v = *v * inv * *w;
        }
    }
}

/// Full gated full-attention core: from the joint QG projection + K/V projections
/// to the pre-`Wo` `core_attn_out`. `q_full`: `[n_tokens, n_heads, 2*head_dim]`
/// (query in `[0,head_dim)`, gate in `[head_dim,2*head_dim)` per head). `k`,`v`:
/// `[n_tokens, n_kv_heads, head_dim]`. `q_norm_w`,`k_norm_w`: `[head_dim]`.
/// `positions`: `[4, n_tokens]`. Causal GQA softmax, scale `1/sqrt(head_dim)`.
/// Writes `out`: `[n_tokens, n_heads, head_dim]`.
#[allow(clippy::too_many_arguments)]
pub fn gated_attention_core(
    q_full: &[f32],
    k: &[f32],
    v: &[f32],
    q_norm_w: &[f32],
    k_norm_w: &[f32],
    positions: &[f32],
    sections: &[u32; 4],
    out: &mut [f32],
    n_tokens: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    rope: &YarnRope,
    eps: f32,
) {
    // 1. split query||gate; gather the query rows and the gate rows.
    let mut q = vec![0f32; n_tokens * n_heads * head_dim];
    let mut gate = vec![0f32; n_tokens * n_heads * head_dim];
    for t in 0..n_tokens {
        for h in 0..n_heads {
            let src = (t * n_heads + h) * 2 * head_dim;
            let dst = (t * n_heads + h) * head_dim;
            q[dst..dst + head_dim].copy_from_slice(&q_full[src..src + head_dim]);
            gate[dst..dst + head_dim].copy_from_slice(&q_full[src + head_dim..src + 2 * head_dim]);
        }
    }
    let mut kk = k.to_vec();

    // 2. per-head QK-RMSNorm, then partial M-RoPE on Q and K.
    rmsnorm_head(&mut q, q_norm_w, n_tokens * n_heads, head_dim, eps);
    rmsnorm_head(&mut kk, k_norm_w, n_tokens * n_kv_heads, head_dim, eps);
    mrope(
        &mut q, positions, sections, n_tokens, n_heads, head_dim, rope,
    );
    mrope(
        &mut kk, positions, sections, n_tokens, n_kv_heads, head_dim, rope,
    );

    // 3. causal GQA softmax attention -> ctx.
    let scale = 1.0 / (head_dim as f32).sqrt();
    let rep = n_heads / n_kv_heads;
    let mut ctx = vec![0f32; n_tokens * n_heads * head_dim];
    let mut scores = vec![0f32; n_tokens];
    for h in 0..n_heads {
        let kvh = h / rep;
        for t in 0..n_tokens {
            let qoff = (t * n_heads + h) * head_dim;
            // scores over the causal window 0..=t
            let mut m = f32::NEG_INFINITY;
            for (j, s) in scores[..=t].iter_mut().enumerate() {
                let koff = (j * n_kv_heads + kvh) * head_dim;
                let mut dot = 0f32;
                for d in 0..head_dim {
                    dot += q[qoff + d] * kk[koff + d];
                }
                *s = dot * scale;
                m = m.max(*s);
            }
            let mut denom = 0f32;
            for s in scores.iter_mut().take(t + 1) {
                *s = (*s - m).exp();
                denom += *s;
            }
            let coff = (t * n_heads + h) * head_dim;
            for (j, &sj) in scores[..=t].iter().enumerate() {
                let w = sj / denom;
                let voff = (j * n_kv_heads + kvh) * head_dim;
                for d in 0..head_dim {
                    ctx[coff + d] += w * v[voff + d];
                }
            }
        }
    }

    // 4. sigmoid output gate.
    out.copy_from_slice(&ctx);
    sigmoid_gate(out, &gate);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mk(seed: u64, n: usize) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 33) as f32 / (1u64 << 31) as f32) - 0.5
            })
            .collect()
    }

    #[test]
    fn sigmoid_gate_hand() {
        let mut x = vec![2.0, -4.0, 1.0];
        let g = vec![0.0, 100.0, -100.0];
        sigmoid_gate(&mut x, &g);
        assert!((x[0] - 1.0).abs() < 1e-6); // sigmoid(0)=0.5, 2*0.5=1
        assert!((x[1] - -4.0).abs() < 1e-4); // sigmoid(+big)=1
        assert!(x[2].abs() < 1e-6); // sigmoid(-big)=0
    }

    // A single token attends only to itself: softmax over one score is 1, so
    // ctx == V (of its kv head), gated by sigmoid(gate). Independent of Q/K/rope.
    #[test]
    fn single_token_is_value_times_gate() {
        let (n_heads, n_kv, hd) = (4usize, 2usize, 8usize);
        let rope = YarnRope::new(4, 1e7, 1.0, 4096, 0.0, 1.0, 32.0, 1.0);
        let q_full = mk(1, n_heads * 2 * hd);
        let k = mk(2, n_kv * hd);
        let v = mk(3, n_kv * hd);
        let qn = mk(4, hd);
        let kn = mk(5, hd);
        let pos = [0.0f32; 4]; // one token, all axes = 0
        let mut out = vec![0f32; n_heads * hd];
        gated_attention_core(
            &q_full,
            &k,
            &v,
            &qn,
            &kn,
            &pos,
            &[2, 1, 1, 0],
            &mut out,
            1,
            n_heads,
            n_kv,
            hd,
            &rope,
            1e-6,
        );
        // out[h] == v[kvh] * sigmoid(gate[h])
        for h in 0..n_heads {
            let kvh = h / (n_heads / n_kv);
            for d in 0..hd {
                let gate = q_full[h * 2 * hd + hd + d];
                let want = v[kvh * hd + d] * (1.0 / (1.0 + (-gate).exp()));
                assert!((out[h * hd + d] - want).abs() < 1e-5, "h{h} d{d}");
            }
        }
    }

    #[test]
    fn gated_attention_runs_and_is_finite() {
        let (t, n_heads, n_kv, hd) = (6usize, 16usize, 4usize, 256usize);
        let rope = YarnRope::new(64, 1e7, 1.0, 4096, 0.0, 1.0, 32.0, 1.0);
        let q_full = mk(1, t * n_heads * 2 * hd);
        let k = mk(2, t * n_kv * hd);
        let v = mk(3, t * n_kv * hd);
        let qn = mk(4, hd);
        let kn = mk(5, hd);
        // text positions: every axis is the token index
        let mut pos = vec![0f32; 4 * t];
        for axis in 0..4 {
            for ti in 0..t {
                pos[axis * t + ti] = ti as f32;
            }
        }
        let mut out = vec![0f32; t * n_heads * hd];
        gated_attention_core(
            &q_full,
            &k,
            &v,
            &qn,
            &kn,
            &pos,
            &[11, 11, 10, 0],
            &mut out,
            t,
            n_heads,
            n_kv,
            hd,
            &rope,
            1e-6,
        );
        assert!(out.iter().all(|x| x.is_finite()));
        assert!(out.iter().any(|&x| x.abs() > 1e-6));
    }
}
