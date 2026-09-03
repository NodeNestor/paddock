//! CPU reference for the Qwen3.5 Gated DeltaNet linear-attention recurrence.
//!
//! Ground truth for the GPU delta-net kernels: both the single-token decode step
//! and the chunked prefill scan diff against this straightforward sequential
//! recurrence. Ported from the two agreeing upstream references - llama.cpp
//! `src/models/delta-net-base.cpp` (`build_delta_net_autoregressive`) and HF
//! transformers `modeling_qwen3_5.py` (`torch_recurrent_gated_delta_rule`),
//! which compute the identical gated delta rule. f32 throughout, plain loops for
//! auditability over speed.
//!
//! Per head the recurrent state `S` is a `[head_dim x head_dim]` matrix (key-dim
//! by value-dim). For each token, with `q`,`k` L2-normalized over the head dim
//! and `q` additionally scaled by `1/sqrt(head_dim)`:
//!
//! ```text
//!   S       = g_t · S              // data-dependent scalar decay, g_t = exp(gate)
//!   u[j]    = Σ_i S[i,j] · k[i]    // the state's current readout for this key
//!   d[j]    = β_t · (v[j] - u[j])  // delta correction
//!   S[i,j] += k[i] · d[j]          // rank-1 state update
//!   out[j]  = Σ_i S[i,j] · q[i]    // readout with the (scaled) query
//! ```
//!
//! `u` is read from the *decayed* state; `out` is read from the *updated* state -
//! matching both references exactly.

/// L2-norm epsilon, inside the rsqrt - matches the FLA/HF `l2norm` helper.
const L2_EPS: f32 = 1e-6;

/// `x / sqrt(Σ x² + eps)` over the whole slice (one head vector).
fn l2norm(x: &[f32]) -> Vec<f32> {
    let inv = 1.0 / (x.iter().map(|v| v * v).sum::<f32>() + L2_EPS).sqrt();
    x.iter().map(|v| v * inv).collect()
}

/// Sequential gated delta rule over `n_tokens`, continuing from `state`.
///
/// Shapes (row-major, contiguous):
/// - `q`, `k`, `v`: `[t][h][d]` - heads already GQA-repeated to `n_heads`.
/// - `g`, `beta`:   `[t][h]`    - `g` is the log-decay (negative); `beta` ∈ (0,1).
/// - `state`: `[h][d][d]` (`[dk][dv]`), read-modify-write; pass zeros to start
///   fresh. Advanced to the post-last-token state on return.
/// - `out`: `[t][h][d]`, written.
///
/// `q`/`k` are L2-normalized and `q` is scaled internally (the kernel boundary
/// matches HF's `use_qk_l2norm_in_kernel=True`): callers pass the post-conv,
/// pre-norm projections.
#[allow(clippy::too_many_arguments)]
pub fn gated_delta_recurrent(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: &[f32],
    beta: &[f32],
    state: &mut [f32],
    out: &mut [f32],
    n_tokens: usize,
    n_heads: usize,
    head_dim: usize,
) {
    let d = head_dim;
    let scale = 1.0 / (d as f32).sqrt();
    let mut u = vec![0f32; d];
    let mut delta = vec![0f32; d];
    let mut o = vec![0f32; d];

    for t in 0..n_tokens {
        for h in 0..n_heads {
            let off = (t * n_heads + h) * d;
            let qh = l2norm(&q[off..off + d]);
            let kh = l2norm(&k[off..off + d]);
            let vh = &v[off..off + d];
            let g_t = g[t * n_heads + h].exp();
            let beta_t = beta[t * n_heads + h];
            let s = &mut state[h * d * d..(h + 1) * d * d]; // [dk][dv], row i = key dim

            // decay the state, then read u = kᵀ·S from the decayed state
            u.iter_mut().for_each(|x| *x = 0.0);
            for i in 0..d {
                let ki = kh[i];
                let row = &mut s[i * d..(i + 1) * d];
                for (j, sij) in row.iter_mut().enumerate() {
                    *sij *= g_t;
                    u[j] += *sij * ki;
                }
            }
            // delta correction
            for j in 0..d {
                delta[j] = beta_t * (vh[j] - u[j]);
            }
            // rank-1 update S += k⊗delta, then read out = qᵀ·S from the updated state
            o.iter_mut().for_each(|x| *x = 0.0);
            for i in 0..d {
                let ki = kh[i];
                let qi = qh[i] * scale;
                let row = &mut s[i * d..(i + 1) * d];
                for (j, sij) in row.iter_mut().enumerate() {
                    *sij += ki * delta[j];
                    o[j] += *sij * qi;
                }
            }
            out[off..off + d].copy_from_slice(&o);
        }
    }
}

/// Chunked gated delta rule - same math as [`gated_delta_recurrent`], restructured
/// so only `n_tokens / chunk` state hops are sequential. This is the CPU oracle for
/// the chunked prefill GPU kernel and mirrors its op structure and numeric recipe
/// exactly: f32 FMA accumulation, cumulative log-decay carried in f64 with ratios
/// taken as `expf` of the f64 difference, and the solve split into two chunk-local
/// right-hand sides so the sequential pass is linear in the incoming state.
///
/// Derivation: unrolling the recurrence inside a chunk with local cumulative
/// log-decay `cg[i] = Σ_{r≤i} g_r` (every ratio `exp(cg[i]-cg[j])`, `j ≤ i`, is
/// bounded ≤ 1 because `g ≤ 0`):
///
/// ```text
///   (I + M)·Δ = diag(β)·(V - diag(exp cg)·K̂·S₀)      M[i,j] = β_i·e^{cg_i-cg_j}·(k̂_i·k̂_j), j<i
///   o_i = e^{cg_i}·(q̃_iᵀ·S₀) + Σ_{j≤i} e^{cg_i-cg_j}·(q̃_i·k̂_j)·δ_j
///   S  <- e^{cg_last}·S₀ + Σ_j e^{cg_last-cg_j}·k̂_j ⊗ δ_j
/// ```
///
/// `(I + M)` is unit lower triangular; with `T = (I+M)⁻¹` the solve is applied to
/// both RHS blocks by one forward substitution - `du = T·diag(β)·V` and
/// `dw = T·diag(β·e^{cg})·K̂` - and the state-dependent deltas resolve later as
/// `Δ = du - dw·S₀` (on the GPU, `du`/`dw` are the parallel per-chunk stage and
/// `Δ`/state hops are the slim sequential stage). The output sum includes the
/// diagonal (`o` reads the post-update state). Not bit-identical to the sequential
/// recurrence - different accumulation structure - so parity is tolerance-based;
/// see the tests for the drift bound vs an f64 ground truth.
#[allow(clippy::too_many_arguments)]
pub fn gated_delta_chunked(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: &[f32],
    beta: &[f32],
    state: &mut [f32],
    out: &mut [f32],
    n_tokens: usize,
    n_heads: usize,
    head_dim: usize,
    chunk: usize,
) {
    let d = head_dim;
    let c = chunk.max(1);
    let scale = 1.0 / (d as f32).sqrt();

    // chunk-local scratch, worst-case `c` rows
    let mut qn = vec![0f32; c * d]; // l2norm(q)·scale rows
    let mut kn = vec![0f32; c * d]; // l2norm(k) rows
    // cumulative log-decay in f64: |cg| grows to ~10s under strong decay, and f32
    // rounding there lands as ~1e-5 relative error after the exp - the dominant
    // drift term. O(c) sequential adds, cheap to mirror on the GPU. Ratios are
    // `expf` of the f64 difference (the subtraction is where f32 loses digits).
    let mut cg = vec![0f64; c];
    let mut akk = vec![0f32; c * c]; // k̂_i·k̂_j
    let mut aqk = vec![0f32; c * c]; // q̃_i·k̂_j
    let mut du = vec![0f32; c * d]; // T·diag(β)·V
    let mut dw = vec![0f32; c * d]; // T·diag(β·e^{cg})·K̂
    let mut delta = vec![0f32; c * d]; // Δ = du - dw·S₀

    for h in 0..n_heads {
        let s = &mut state[h * d * d..(h + 1) * d * d]; // [dk][dv]
        for c0 in (0..n_tokens).step_by(c) {
            let cl = c.min(n_tokens - c0);

            // gather normalized q/k and cumulative decay for the chunk
            let mut run = 0f64;
            for i in 0..cl {
                let off = ((c0 + i) * n_heads + h) * d;
                let qh = l2norm(&q[off..off + d]);
                let kh = l2norm(&k[off..off + d]);
                for a in 0..d {
                    qn[i * d + a] = qh[a] * scale;
                    kn[i * d + a] = kh[a];
                }
                run += g[(c0 + i) * n_heads + h] as f64;
                cg[i] = run;
            }

            // chunk-local dot-product matrices (j ≤ i is all we read); mul_add
            // throughout the chunked path = the GPU kernel's single-rounding FMA
            for i in 0..cl {
                for j in 0..=i {
                    let mut kk = 0f32;
                    let mut qk = 0f32;
                    for a in 0..d {
                        kk = kn[i * d + a].mul_add(kn[j * d + a], kk);
                        qk = qn[i * d + a].mul_add(kn[j * d + a], qk);
                    }
                    akk[i * c + j] = kk;
                    aqk[i * c + j] = qk;
                }
            }

            // forward-substitute (I+M)⁻¹ once over both RHS blocks (S₀-free):
            //   du_i = β_i·v_i - Σ_{j<i} m_ij·du_j,  dw_i = β_i·e^{cg_i}·k̂_i - Σ m_ij·dw_j
            for i in 0..cl {
                let off = ((c0 + i) * n_heads + h) * d;
                let b_i = beta[(c0 + i) * n_heads + h];
                let bg = b_i * (cg[i] as f32).exp();
                for dv in 0..d {
                    du[i * d + dv] = b_i * v[off + dv];
                    dw[i * d + dv] = bg * kn[i * d + dv];
                }
                for j in 0..i {
                    let m_ij = b_i * ((cg[i] - cg[j]) as f32).exp() * akk[i * c + j];
                    for dv in 0..d {
                        du[i * d + dv] = (-m_ij).mul_add(du[j * d + dv], du[i * d + dv]);
                        dw[i * d + dv] = (-m_ij).mul_add(dw[j * d + dv], dw[i * d + dv]);
                    }
                }
            }

            // the sequential-stage resolve: Δ = du - dw·S₀
            for i in 0..cl {
                for dv in 0..d {
                    let mut acc = du[i * d + dv];
                    for a in 0..d {
                        acc = (-dw[i * d + a]).mul_add(s[a * d + dv], acc);
                    }
                    delta[i * d + dv] = acc;
                }
            }

            // o_i = e^{cg_i}·(q̃_iᵀ·S₀) + Σ_{j≤i} e^{cg_i-cg_j}·(q̃_i·k̂_j)·δ_j
            for i in 0..cl {
                let off = ((c0 + i) * n_heads + h) * d;
                let gam = (cg[i] as f32).exp();
                for dv in 0..d {
                    let mut acc = 0f32;
                    for a in 0..d {
                        acc = qn[i * d + a].mul_add(s[a * d + dv], acc);
                    }
                    acc *= gam;
                    for j in 0..=i {
                        acc = (((cg[i] - cg[j]) as f32).exp() * aqk[i * c + j])
                            .mul_add(delta[j * d + dv], acc);
                    }
                    out[off + dv] = acc;
                }
            }

            // state hop: S <- e^{cg_last}·S₀ + Σ_j e^{cg_last-cg_j}·k̂_j ⊗ δ_j
            let g_all = (cg[cl - 1] as f32).exp();
            for a in 0..d {
                for dv in 0..d {
                    let mut acc = g_all * s[a * d + dv];
                    for j in 0..cl {
                        acc = (((cg[cl - 1] - cg[j]) as f32).exp() * kn[j * d + a])
                            .mul_add(delta[j * d + dv], acc);
                    }
                    s[a * d + dv] = acc;
                }
            }
        }
    }
}

fn silu(v: f32) -> f32 {
    v / (1.0 + (-v).exp())
}

/// Numerically-stable softplus: `log(1 + exp(x))` = `max(x,0) + log1p(exp(-|x|))`.
fn softplus(x: f32) -> f32 {
    x.max(0.0) + (-x.abs()).exp().ln_1p()
}

/// Depthwise causal conv1d (kernel `k`) + SiLU - the DeltaNet input conv. Matches
/// HF `F.conv1d(groups=conv_dim, padding=k-1)[:T]` then `silu`, and llama.cpp
/// `ggml_ssm_conv`. Zero left-padding (fresh state / start of sequence).
///
/// `x`: `[n_tokens][conv_dim]` row-major; `w`: `[conv_dim][k]` row-major
/// (`w[c*k + kk]`); `out`: `[n_tokens][conv_dim]`.
/// `out[t,c] = silu(Σ_kk w[c,kk]·x[t-(k-1)+kk, c])`, with `x[<0]=0`.
pub fn causal_conv1d_silu(
    x: &[f32],
    w: &[f32],
    out: &mut [f32],
    n_tokens: usize,
    conv_dim: usize,
    k: usize,
) {
    for t in 0..n_tokens {
        for c in 0..conv_dim {
            let mut acc = 0.0f32;
            for kk in 0..k {
                let ti = t as isize - (k as isize - 1) + kk as isize;
                if ti >= 0 {
                    acc += w[c * k + kk] * x[ti as usize * conv_dim + c];
                }
            }
            out[t * conv_dim + c] = silu(acc);
        }
    }
}

/// DeltaNet gate math, per (token, head):
///   `beta = sigmoid(b)`,  `g = ssm_a · softplus(a + dt_bias)`.
/// `ssm_a` is the GGUF value (already `-exp(A_log)`), `dt_bias` is per-head.
/// `a`,`b`: `[n_tokens][n_heads]`; `ssm_a`,`dt_bias`: `[n_heads]`; outputs
/// `g`,`beta`: `[n_tokens][n_heads]`.
#[allow(clippy::too_many_arguments)]
pub fn delta_gate(
    a: &[f32],
    b: &[f32],
    ssm_a: &[f32],
    dt_bias: &[f32],
    g: &mut [f32],
    beta: &mut [f32],
    n_tokens: usize,
    n_heads: usize,
) {
    for t in 0..n_tokens {
        for h in 0..n_heads {
            let idx = t * n_heads + h;
            beta[idx] = 1.0 / (1.0 + (-b[idx]).exp());
            g[idx] = ssm_a[h] * softplus(a[idx] + dt_bias[h]);
        }
    }
}

/// Gated RMSNorm over the head-value dim, per row (token×head):
///   `out = (x · rsqrt(mean(x²)+eps)) · weight · silu(z)`
/// matching HF `Qwen3_5RMSNormGated` (normalize, then weight, then silu gate).
/// `x`,`z`,`out`: `[n_rows][d]`; `weight`: `[d]`; `n_rows = n_tokens·n_heads`.
pub fn gated_rmsnorm(
    x: &[f32],
    z: &[f32],
    weight: &[f32],
    out: &mut [f32],
    n_rows: usize,
    d: usize,
    eps: f32,
) {
    for r in 0..n_rows {
        let row = &x[r * d..(r + 1) * d];
        let mean_sq = row.iter().map(|v| v * v).sum::<f32>() / d as f32;
        let inv = 1.0 / (mean_sq + eps).sqrt();
        for j in 0..d {
            out[r * d + j] = row[j] * inv * weight[j] * silu(z[r * d + j]);
        }
    }
}

/// Full Qwen3.5 DeltaNet mixer core: from the post-in_proj projections to the
/// pre-out_proj `core_attn_out`. Composes the verified sub-ops exactly as HF
/// `Qwen3_5GatedDeltaNet.forward`: conv+silu(mixed_qkv) -> split q,k,v ->
/// GQA-repeat q,k to n_v_heads -> gate(a,b) -> gated delta recurrence -> gated
/// RMSNorm(·, z). This is the piece with no existing analog in the engine; the
/// surrounding in_proj / out_proj are plain Q8_0 GEMMs.
///
/// Shapes (row-major): `mixed_qkv` [T, 2*key_dim+value_dim]; `z` [T, value_dim];
/// `a`,`b` [T, n_v_heads]; `conv_w` [conv_dim, k]; `ssm_a`,`dt_bias` [n_v_heads];
/// `ssm_norm_w` [s]; `core_out` [T, value_dim]. `s` = head_k_dim = head_v_dim;
/// `key_dim = s*n_k_heads`, `value_dim = s*n_v_heads`, `conv_dim = 2*key_dim+value_dim`.
#[allow(clippy::too_many_arguments)]
pub fn deltanet_mixer_core(
    mixed_qkv: &[f32],
    z: &[f32],
    a: &[f32],
    b: &[f32],
    conv_w: &[f32],
    ssm_a: &[f32],
    dt_bias: &[f32],
    ssm_norm_w: &[f32],
    core_out: &mut [f32],
    n_tokens: usize,
    n_k_heads: usize,
    n_v_heads: usize,
    s: usize,
    k: usize,
    eps: f32,
) {
    let key_dim = s * n_k_heads;
    let value_dim = s * n_v_heads;
    let conv_dim = 2 * key_dim + value_dim;

    // 1. depthwise causal conv + silu over all conv_dim channels
    let mut conv = vec![0f32; n_tokens * conv_dim];
    causal_conv1d_silu(mixed_qkv, conv_w, &mut conv, n_tokens, conv_dim, k);

    // 2. split channels [q(key_dim) | k(key_dim) | v(value_dim)]; GQA-repeat q,k
    //    to n_v_heads. Matches llama.cpp `ggml_repeat_4d` = TILING: output head hv
    //    reads key head (hv % n_k_heads), not hv/rep (repeat_interleave).
    let mut q = vec![0f32; n_tokens * n_v_heads * s];
    let mut kq = vec![0f32; n_tokens * n_v_heads * s];
    let mut v = vec![0f32; n_tokens * n_v_heads * s];
    for t in 0..n_tokens {
        let row = &conv[t * conv_dim..(t + 1) * conv_dim];
        for hv in 0..n_v_heads {
            let hk = hv % n_k_heads;
            let dst = (t * n_v_heads + hv) * s;
            q[dst..dst + s].copy_from_slice(&row[hk * s..(hk + 1) * s]);
            kq[dst..dst + s].copy_from_slice(&row[key_dim + hk * s..key_dim + (hk + 1) * s]);
            v[dst..dst + s].copy_from_slice(&row[2 * key_dim + hv * s..2 * key_dim + (hv + 1) * s]);
        }
    }

    // 3. gate math -> g, beta
    let mut g = vec![0f32; n_tokens * n_v_heads];
    let mut beta = vec![0f32; n_tokens * n_v_heads];
    delta_gate(a, b, ssm_a, dt_bias, &mut g, &mut beta, n_tokens, n_v_heads);

    // 4. gated delta recurrence (l2norm + scale happen inside)
    let mut state = vec![0f32; n_v_heads * s * s];
    let mut attn = vec![0f32; n_tokens * n_v_heads * s];
    gated_delta_recurrent(
        &q, &kq, &v, &g, &beta, &mut state, &mut attn, n_tokens, n_v_heads, s,
    );

    // 5. gated RMSNorm over s per (token,head) row, gated by silu(z)
    gated_rmsnorm(&attn, z, ssm_norm_w, core_out, n_tokens * n_v_heads, s, eps);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    // For a single token from a zero state the rule collapses to a closed form:
    //   out[j] = β · (q̂·scale · k̂) · v[j],   S[i,j] = k̂[i] · β · v[j]
    // with q̂,k̂ the L2-normalized q,k and scale = 1/sqrt(d).
    #[test]
    fn single_token_closed_form() {
        let d = 4;
        let q = vec![1.0, 0.0, 0.0, 0.0]; // l2 -> itself; scaled by 1/2
        let k = vec![1.0, 0.0, 0.0, 0.0]; // l2 -> itself
        let v = vec![2.0, 3.0, 4.0, 5.0];
        let g = vec![-0.7f32]; // any decay: state starts at 0 so it doesn't matter
        let beta = vec![0.5f32];
        let mut state = vec![0f32; d * d];
        let mut out = vec![0f32; d];
        gated_delta_recurrent(&q, &k, &v, &g, &beta, &mut state, &mut out, 1, 1, d);

        let scale = 1.0 / (d as f32).sqrt();
        let qk = dot(&q, &k) * scale; // 0.5
        for j in 0..d {
            assert!(
                (out[j] - beta[0] * qk * v[j]).abs() < 1e-5,
                "out[{j}]={}",
                out[j]
            );
        }
        // state row 0 (k[0]=1) holds β·v; the rest is zero
        for j in 0..d {
            assert!((state[j] - beta[0] * v[j]).abs() < 1e-5);
        }
        for i in 1..d {
            for j in 0..d {
                assert_eq!(state[i * d + j], 0.0);
            }
        }
    }

    // β = 0 makes every delta zero, so the state can never leave zero and the
    // output is identically zero regardless of q/k/v/g.
    #[test]
    fn beta_zero_stays_zero() {
        let (t, h, d) = (5, 3, 8);
        let n = t * h * d;
        let mk = |seed: u64| {
            let mut s = seed;
            (0..n)
                .map(|_| {
                    s = s
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    ((s >> 33) as f32 / (1u64 << 31) as f32) - 0.5
                })
                .collect::<Vec<_>>()
        };
        let q = mk(1);
        let k = mk(2);
        let v = mk(3);
        let g = vec![-0.3f32; t * h];
        let beta = vec![0.0f32; t * h];
        let mut state = vec![0f32; h * d * d];
        let mut out = vec![9f32; t * h * d];
        gated_delta_recurrent(&q, &k, &v, &g, &beta, &mut state, &mut out, t, h, d);
        assert!(out.iter().all(|&x| x == 0.0));
        assert!(state.iter().all(|&x| x == 0.0));
    }

    // With g -> -∞ the decay g_t = exp(g) = 0 wipes the state before each token,
    // so every step is independent and equals the single-token closed form.
    #[test]
    fn full_decay_decouples_tokens() {
        let (t, h, d) = (4, 2, 6);
        let mk = |seed: u64, n: usize| {
            let mut s = seed;
            (0..n)
                .map(|_| {
                    s = s
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    ((s >> 33) as f32 / (1u64 << 31) as f32) - 0.5
                })
                .collect::<Vec<_>>()
        };
        let q = mk(11, t * h * d);
        let k = mk(22, t * h * d);
        let v = mk(33, t * h * d);
        let g = vec![-1e30f32; t * h]; // exp -> 0
        let beta = mk(44, t * h).iter().map(|x| x + 0.5).collect::<Vec<_>>(); // ∈ (0,1)
        let mut state = vec![0f32; h * d * d];
        let mut out = vec![0f32; t * h * d];
        gated_delta_recurrent(&q, &k, &v, &g, &beta, &mut state, &mut out, t, h, d);

        let scale = 1.0 / (d as f32).sqrt();
        for ti in 0..t {
            for hi in 0..h {
                let off = (ti * h + hi) * d;
                let qn = l2norm(&q[off..off + d]);
                let kn = l2norm(&k[off..off + d]);
                let qk = dot(&qn, &kn) * scale;
                let b = beta[ti * h + hi];
                for j in 0..d {
                    let want = b * qk * v[off + j];
                    assert!((out[off + j] - want).abs() < 1e-5, "t{ti} h{hi} j{j}");
                }
            }
        }
    }

    #[test]
    fn conv1d_silu_causal_hand() {
        // conv_dim=1, K=4, x=[1,2,3]; causal taps w[0..3], w[3] is the current
        // token. y[0]=silu(w3·1); y[1]=silu(w2·1+w3·2); y[2]=silu(w1·1+w2·2+w3·3).
        let x = vec![1.0, 2.0, 3.0];
        let w = vec![0.1, 0.2, 0.3, 0.4];
        let mut out = vec![0.0; 3];
        causal_conv1d_silu(&x, &w, &mut out, 3, 1, 4);
        assert!((out[0] - silu(0.4)).abs() < 1e-6);
        assert!((out[1] - silu(0.3 + 0.4 * 2.0)).abs() < 1e-6);
        assert!((out[2] - silu(0.2 + 0.3 * 2.0 + 0.4 * 3.0)).abs() < 1e-6);
    }

    #[test]
    fn delta_gate_formula() {
        let a = vec![0.5, -1.0];
        let b = vec![0.0, 2.0];
        let ssm_a = vec![-1.5]; // one head; already -exp(A_log)
        let dt = vec![0.25];
        let mut g = vec![0.0; 2];
        let mut beta = vec![0.0; 2];
        delta_gate(&a, &b, &ssm_a, &dt, &mut g, &mut beta, 2, 1);
        let sig = |v: f32| 1.0 / (1.0 + (-v).exp());
        assert!((beta[0] - sig(0.0)).abs() < 1e-6);
        assert!((beta[1] - sig(2.0)).abs() < 1e-6);
        assert!((g[0] - (-1.5) * softplus(0.75)).abs() < 1e-5);
        assert!((g[1] - (-1.5) * softplus(-0.75)).abs() < 1e-5);
    }

    #[test]
    fn gated_rmsnorm_hand() {
        // d=2, one row; x=[3,4] -> mean_sq = (9+16)/2 = 12.5.
        let x = vec![3.0, 4.0];
        let z = vec![0.0, 1.0];
        let w = vec![1.0, 2.0];
        let mut out = vec![0.0; 2];
        gated_rmsnorm(&x, &z, &w, &mut out, 1, 2, 1e-6);
        let inv = 1.0 / (12.5f32 + 1e-6).sqrt();
        assert!((out[0] - 3.0 * inv * 1.0 * silu(0.0)).abs() < 1e-6);
        assert!((out[1] - 4.0 * inv * 2.0 * silu(1.0)).abs() < 1e-6);
    }

    fn mk(seed: u64, n: usize) -> Vec<f32> {
        let mut st = seed;
        (0..n)
            .map(|_| {
                st = st
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((st >> 33) as f32 / (1u64 << 31) as f32) - 0.5
            })
            .collect()
    }

    // ---- chunked-vs-sequential parity (the gate for the chunked GPU kernel) ----

    /// f64 twin of `gated_delta_recurrent` - drift ground truth.
    #[allow(clippy::too_many_arguments)]
    fn gated_delta_recurrent_f64(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        g: &[f32],
        beta: &[f32],
        state: &mut [f64],
        out: &mut [f64],
        n_tokens: usize,
        n_heads: usize,
        head_dim: usize,
    ) {
        let d = head_dim;
        let scale = 1.0 / (d as f64).sqrt();
        let l2 = |x: &[f32]| -> Vec<f64> {
            let inv =
                1.0 / (x.iter().map(|&v| v as f64 * v as f64).sum::<f64>() + L2_EPS as f64).sqrt();
            x.iter().map(|&v| v as f64 * inv).collect()
        };
        let mut u = vec![0f64; d];
        let mut delta = vec![0f64; d];
        for t in 0..n_tokens {
            for h in 0..n_heads {
                let off = (t * n_heads + h) * d;
                let qh = l2(&q[off..off + d]);
                let kh = l2(&k[off..off + d]);
                let g_t = (g[t * n_heads + h] as f64).exp();
                let b_t = beta[t * n_heads + h] as f64;
                let s = &mut state[h * d * d..(h + 1) * d * d];
                u.iter_mut().for_each(|x| *x = 0.0);
                for i in 0..d {
                    for j in 0..d {
                        s[i * d + j] *= g_t;
                        u[j] += s[i * d + j] * kh[i];
                    }
                }
                for j in 0..d {
                    delta[j] = b_t * (v[off + j] as f64 - u[j]);
                }
                for j in 0..d {
                    out[off + j] = 0.0;
                }
                for i in 0..d {
                    for j in 0..d {
                        s[i * d + j] += kh[i] * delta[j];
                        out[off + j] += s[i * d + j] * qh[i] * scale;
                    }
                }
            }
        }
    }

    /// max |a-b| normalized by rms(b): scale-free error across the whole tensor.
    fn nerr(a: &[f32], b: &[f32]) -> f64 {
        let rms = (b.iter().map(|&x| x as f64 * x as f64).sum::<f64>() / b.len() as f64).sqrt();
        let mx = a
            .iter()
            .zip(b)
            .map(|(&x, &y)| (x as f64 - y as f64).abs())
            .fold(0.0, f64::max);
        mx / rms.max(1e-30)
    }

    fn nerr64(a: &[f32], b: &[f64]) -> f64 {
        let rms = (b.iter().map(|&x| x * x).sum::<f64>() / b.len() as f64).sqrt();
        let mx = a
            .iter()
            .zip(b)
            .map(|(&x, &y)| (x as f64 - y).abs())
            .fold(0.0, f64::max);
        mx / rms.max(1e-30)
    }

    /// Realistic gates: g = ssm_a·softplus(x) with ssm_a ∈ [-5,-0.2] (per head),
    /// β = sigmoid(uniform·4). Mirrors `delta_gate` output ranges.
    fn realistic_gates(seed: u64, t: usize, h: usize) -> (Vec<f32>, Vec<f32>) {
        let ssm_a: Vec<f32> = mk(seed, h).iter().map(|x| -(x + 0.5) * 4.8 - 0.2).collect();
        let raw = mk(seed + 1, t * h);
        let braw = mk(seed + 2, t * h);
        let g = (0..t * h)
            .map(|i| ssm_a[i % h] * softplus(raw[i] * 4.0))
            .collect();
        let beta = braw
            .iter()
            .map(|x| 1.0 / (1.0 + (-x * 4.0).exp()))
            .collect();
        (g, beta)
    }

    fn run_both(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        g: &[f32],
        beta: &[f32],
        t: usize,
        h: usize,
        d: usize,
        chunk: usize,
    ) -> (f64, f64) {
        let mut st_s = vec![0f32; h * d * d];
        let mut out_s = vec![0f32; t * h * d];
        gated_delta_recurrent(q, k, v, g, beta, &mut st_s, &mut out_s, t, h, d);
        let mut st_c = vec![0f32; h * d * d];
        let mut out_c = vec![0f32; t * h * d];
        gated_delta_chunked(q, k, v, g, beta, &mut st_c, &mut out_c, t, h, d, chunk);
        (nerr(&out_c, &out_s), nerr(&st_c, &st_s))
    }

    // Chunk size 1 is the same per-token math in a marginally different op order:
    // it anchors the reformulation at near-f32-exact before any real chunking.
    #[test]
    fn chunked_c1_anchor() {
        let (t, h, d) = (64, 2, 32);
        let q = mk(101, t * h * d);
        let k = mk(102, t * h * d);
        let v = mk(103, t * h * d);
        let (g, beta) = realistic_gates(104, t, h);
        let (eo, es) = run_both(&q, &k, &v, &g, &beta, t, h, d, 1);
        assert!(
            eo < 1e-5 && es < 1e-5,
            "C=1 anchor: out={eo:.2e} state={es:.2e}"
        );
    }

    // Chunk sizes incl. odd (7), non-dividing (T=200, C=64 -> partial tail 8),
    // and larger-than-T (C=256): all must agree with the sequential recurrence.
    #[test]
    fn chunked_matches_sequential_across_chunk_sizes() {
        let (t, h, d) = (200, 3, 64);
        let q = mk(11, t * h * d);
        let k = mk(12, t * h * d);
        let v = mk(13, t * h * d);
        let (g, beta) = realistic_gates(14, t, h);
        for chunk in [7usize, 16, 32, 64, 128, 256] {
            let (eo, es) = run_both(&q, &k, &v, &g, &beta, t, h, d, chunk);
            println!("C={chunk:3}  out={eo:.3e}  state={es:.3e}");
            assert!(
                eo < 1e-4 && es < 1e-4,
                "C={chunk}: out={eo:.2e} state={es:.2e}"
            );
        }
    }

    // Model-shaped run: T=512, D=128, C=64 - the exact geometry the GPU kernel
    // will run for qwen35 pp512.
    #[test]
    fn chunked_model_shape_t512_d128() {
        let (t, h, d) = (512, 4, 128);
        let q = mk(21, t * h * d);
        let k = mk(22, t * h * d);
        let v = mk(23, t * h * d);
        let (g, beta) = realistic_gates(24, t, h);
        let (eo, es) = run_both(&q, &k, &v, &g, &beta, t, h, d, 64);
        println!("t512 d128 C=64  out={eo:.3e}  state={es:.3e}");
        assert!(eo < 1e-4 && es < 1e-4, "out={eo:.2e} state={es:.2e}");
    }

    // Continuation: the chunked scan must accept a nonzero incoming state (chat
    // continuation / multi-pass prefill) - sequential 48 tokens, then chunked for
    // the remaining 150 from that state, vs sequential over all 198.
    #[test]
    fn chunked_continues_from_state() {
        let (t0, t1, h, d) = (48usize, 150usize, 2, 64);
        let t = t0 + t1;
        let q = mk(31, t * h * d);
        let k = mk(32, t * h * d);
        let v = mk(33, t * h * d);
        let (g, beta) = realistic_gates(34, t, h);

        let mut st_s = vec![0f32; h * d * d];
        let mut out_s = vec![0f32; t * h * d];
        gated_delta_recurrent(&q, &k, &v, &g, &beta, &mut st_s, &mut out_s, t, h, d);

        let mut st_c = vec![0f32; h * d * d];
        let mut pre = vec![0f32; t0 * h * d];
        gated_delta_recurrent(&q, &k, &v, &g, &beta, &mut st_c, &mut pre, t0, h, d);
        let n0 = t0 * h * d;
        let mut out_c = vec![0f32; t1 * h * d];
        gated_delta_chunked(
            &q[n0..],
            &k[n0..],
            &v[n0..],
            &g[t0 * h..],
            &beta[t0 * h..],
            &mut st_c,
            &mut out_c,
            t1,
            h,
            d,
            64,
        );
        let eo = nerr(&out_c, &out_s[n0..]);
        let es = nerr(&st_c, &st_s);
        println!("continuation  out={eo:.3e}  state={es:.3e}");
        assert!(eo < 1e-4 && es < 1e-4, "out={eo:.2e} state={es:.2e}");
    }

    // Adversarial solve conditioning: near-identical keys, β -> 1, decay -> 1.
    // (I+M) is then close to all-ones-lower-triangular; the delta rule telescopes
    // (repeated writes to one key replace the value) so this must stay stable.
    #[test]
    fn chunked_adversarial_correlated_keys() {
        let (t, h, d) = (256, 2, 64);
        let base = mk(41, h * d);
        let noise = mk(42, t * h * d);
        let mut k = vec![0f32; t * h * d];
        for ti in 0..t {
            for hi in 0..h {
                for a in 0..d {
                    k[(ti * h + hi) * d + a] =
                        base[hi * d + a] + 0.01 * noise[(ti * h + hi) * d + a];
                }
            }
        }
        let q = mk(43, t * h * d);
        let v = mk(44, t * h * d);
        let g = vec![-1e-4f32; t * h]; // decay ≈ 1: worst accumulation case
        let beta = vec![0.999f32; t * h];
        let (eo, es) = run_both(&q, &k, &v, &g, &beta, t, h, d, 64);
        println!("adversarial  out={eo:.3e}  state={es:.3e}");
        assert!(eo < 1e-3 && es < 1e-3, "out={eo:.2e} state={es:.2e}");
    }

    // The decisive Phase-0 gate: both f32 implementations drift from the f64
    // ground truth; the chunked reformulation must not drift materially more than
    // the sequential recurrence itself does. Ratio ≤ 4 (plus an absolute floor for
    // when both are at f32 noise level).
    #[test]
    fn chunked_drift_vs_f64_comparable_to_sequential() {
        let (t, h, d) = (512, 4, 128);
        let q = mk(51, t * h * d);
        let k = mk(52, t * h * d);
        let v = mk(53, t * h * d);
        let (g, beta) = realistic_gates(54, t, h);

        let mut st64 = vec![0f64; h * d * d];
        let mut out64 = vec![0f64; t * h * d];
        gated_delta_recurrent_f64(&q, &k, &v, &g, &beta, &mut st64, &mut out64, t, h, d);

        let mut st_s = vec![0f32; h * d * d];
        let mut out_s = vec![0f32; t * h * d];
        gated_delta_recurrent(&q, &k, &v, &g, &beta, &mut st_s, &mut out_s, t, h, d);
        let mut st_c = vec![0f32; h * d * d];
        let mut out_c = vec![0f32; t * h * d];
        gated_delta_chunked(&q, &k, &v, &g, &beta, &mut st_c, &mut out_c, t, h, d, 64);

        let (seq_o, seq_s) = (nerr64(&out_s, &out64), nerr64(&st_s, &st64));
        let (chk_o, chk_s) = (nerr64(&out_c, &out64), nerr64(&st_c, &st64));
        println!("drift vs f64:  seq out={seq_o:.3e} state={seq_s:.3e}");
        println!("drift vs f64:  chk out={chk_o:.3e} state={chk_s:.3e}");
        assert!(
            chk_o < (seq_o * 4.0).max(1e-5),
            "chunked out drift {chk_o:.2e} vs sequential {seq_o:.2e}"
        );
        assert!(
            chk_s < (seq_s * 4.0).max(1e-5),
            "chunked state drift {chk_s:.2e} vs sequential {seq_s:.2e}"
        );
    }

    // Composition smoke test: real head geometry (GQA repeat 2×), runs end to end
    // and produces finite, non-trivial output. Full numeric correctness is gated
    // by the model-level HF parity in P2.
    #[test]
    fn deltanet_mixer_core_runs_and_is_finite() {
        let (t, n_k, n_v, s, k) = (4usize, 2usize, 4usize, 8usize, 4usize);
        let key_dim = s * n_k;
        let value_dim = s * n_v;
        let conv_dim = 2 * key_dim + value_dim;
        let mixed = mk(1, t * conv_dim);
        let z = mk(2, t * value_dim);
        let a = mk(3, t * n_v);
        let b = mk(4, t * n_v);
        let conv_w = mk(5, conv_dim * k);
        let ssm_a: Vec<f32> = mk(6, n_v).iter().map(|x| -x.abs() - 0.1).collect();
        let dt = mk(7, n_v);
        let norm_w = mk(8, s);
        let mut out = vec![0f32; t * value_dim];
        deltanet_mixer_core(
            &mixed, &z, &a, &b, &conv_w, &ssm_a, &dt, &norm_w, &mut out, t, n_k, n_v, s, k, 1e-6,
        );
        assert!(out.iter().all(|x| x.is_finite()), "mixer output not finite");
        assert!(
            out.iter().any(|&x| x.abs() > 1e-6),
            "mixer output all ~zero"
        );
    }
}
