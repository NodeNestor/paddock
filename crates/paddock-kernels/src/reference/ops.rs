//! CPU reference implementations of the per-op kernels (RMSNorm, YaRN rope,
//! softmax-with-sink, swiglu-oai, add). Single source of truth: the reference
//! models call these, and every GPU kernel is parity-gated against them.
//! Formulas source-verified against ggml/llama.cpp.

/// RMSNorm: x * w / sqrt(mean(x²) + eps).
pub fn rms_norm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv = 1.0 / (ms + eps).sqrt();
    x.iter().zip(weight).map(|(v, w)| v * inv * w).collect()
}

/// Softmax over `scores` with an extra sink logit that joins max and
/// denominator but keeps no slot (it absorbs probability mass).
pub fn softmax_with_sink(scores: &mut [f32], sink: f32) {
    let mut m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    m = m.max(sink);
    let mut denom = (sink - m).exp();
    for s in scores.iter_mut() {
        *s = (*s - m).exp();
        denom += *s;
    }
    for s in scores.iter_mut() {
        *s /= denom;
    }
}

/// The OpenAI swiglu variant (verified against ggml): gate is clamped from
/// above only, up symmetrically, and the up branch carries a +1.
pub fn swiglu_oai(gate: &mut [f32], up: &[f32], alpha: f32, limit: f32) {
    for (g, u) in gate.iter_mut().zip(up) {
        let x = g.min(limit);
        let y = u.clamp(-limit, limit);
        *g = (x / (1.0 + (-alpha * x).exp())) * (y + 1.0);
    }
}

/// Plain SwiGLU (the standard Llama/Qwen FFN activation, no clamps): in place on
/// `gate`, `gate[i] = silu(gate[i]) * up[i]` with `silu(x) = x·sigmoid(x)`.
/// Matches HF `act_fn(gate_proj(x)) * up_proj(x)` (act_fn = SiLU).
pub fn swiglu(gate: &mut [f32], up: &[f32]) {
    for (g, u) in gate.iter_mut().zip(up) {
        *g = (*g / (1.0 + (-*g).exp())) * *u;
    }
}

/// YaRN-scaled NEOX RoPE (pairs are (k, k + n_dims/2); mscale rides on
/// cos/sin). Ported from ggml's rope_yarn / rotate_pairs.
pub struct YarnRope {
    pub n_dims: usize,
    theta_scale: f32,
    freq_scale: f32,
    corr_low: f32,
    corr_high: f32,
    ext_factor: f32,
    mscale: f32,
}

impl YarnRope {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        n_dims: usize,
        base: f32,
        freq_scale: f32,
        n_ctx_orig: usize,
        ext_factor: f32,
        attn_factor: f32,
        beta_fast: f32,
        beta_slow: f32,
    ) -> Self {
        let corr_dim = |n_rot: f32| {
            (n_dims as f32) * (n_ctx_orig as f32 / (n_rot * 2.0 * std::f32::consts::PI)).ln()
                / (2.0 * base.ln())
        };
        let mscale = if ext_factor != 0.0 {
            attn_factor * (1.0 + 0.1 * (1.0 / freq_scale).ln())
        } else {
            attn_factor
        };
        Self {
            n_dims,
            theta_scale: base.powf(-2.0 / n_dims as f32),
            freq_scale,
            corr_low: corr_dim(beta_fast).floor().max(0.0),
            corr_high: corr_dim(beta_slow).ceil().min(n_dims as f32 - 1.0),
            ext_factor,
            mscale,
        }
    }

    /// Parameters in GPU-kernel argument order, so both sides consume the
    /// exact same numbers.
    pub fn kernel_params(&self) -> (f32, f32, f32, f32, f32, f32) {
        (
            self.theta_scale,
            self.freq_scale,
            self.corr_low,
            self.corr_high,
            self.ext_factor,
            self.mscale,
        )
    }

    pub fn apply(&self, head: &mut [f32], pos: usize) {
        let half = self.n_dims / 2;
        let mut theta = pos as f32;
        for k in 0..half {
            let y = (k as f32 - self.corr_low) / (self.corr_high - self.corr_low).max(0.001);
            let ramp = (1.0 - y.clamp(0.0, 1.0)) * self.ext_factor;
            let angle = (self.freq_scale * theta) * (1.0 - ramp) + theta * ramp;
            let (sin, cos) = angle.sin_cos();
            let (sin, cos) = (sin * self.mscale, cos * self.mscale);

            let a = head[k];
            let b = head[k + half];
            head[k] = a * cos - b * sin;
            head[k + half] = a * sin + b * cos;
            theta *= self.theta_scale;
        }
    }

    /// Partial sectioned M-RoPE over one head vector of length `head_dim`
    /// (`head.len()` may exceed `self.n_dims`). `self.n_dims` is the rotary width
    /// `n_rot`: rotates NEOX pairs `(p, p + n_rot/2)` for `p in [0, n_rot/2)` and
    /// leaves channels `[n_rot, head_dim)` untouched. Pair `p` reads its angle base
    /// from the position axis its section maps to - `positions = [t,h,w,e]`,
    /// `sections` the per-axis pair counts (their sum is the section period).
    /// Each axis advances its own theta chain every pair. For text all four
    /// positions are equal and this reduces to plain partial NEOX rope.
    /// Matches ggml `ggml_mrope_cache_init` + `rotate_pairs` (GGML_ROPE_TYPE_MROPE,
    /// non-interleaved) - the multimodal-ready form.
    pub fn apply_mrope(&self, head: &mut [f32], positions: &[f32; 4], sections: &[u32; 4]) {
        let half = self.n_dims / 2;
        let sect_dims = (sections[0] + sections[1] + sections[2] + sections[3]) as usize;
        let sec_h = sections[0] as usize;
        let sec_w = (sections[0] + sections[1]) as usize;
        let sec_e = sec_w + sections[2] as usize;
        let mut theta = *positions;
        for p in 0..half {
            let sector = p % sect_dims;
            let base = if sector < sec_h {
                theta[0]
            } else if sector < sec_w {
                theta[1]
            } else if sector < sec_e {
                theta[2]
            } else {
                theta[3]
            };
            let y = (p as f32 - self.corr_low) / (self.corr_high - self.corr_low).max(0.001);
            let ramp = (1.0 - y.clamp(0.0, 1.0)) * self.ext_factor;
            let angle = (self.freq_scale * base) * (1.0 - ramp) + base * ramp;
            let (sin, cos) = angle.sin_cos();
            let (sin, cos) = (sin * self.mscale, cos * self.mscale);

            let a = head[p];
            let b = head[p + half];
            head[p] = a * cos - b * sin;
            head[p + half] = a * sin + b * cos;
            for t in theta.iter_mut() {
                *t *= self.theta_scale;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_sink_conserves_less_than_one() {
        // the sink absorbs mass: probabilities must sum to < 1
        let mut s = vec![1.0f32, 2.0, 3.0];
        softmax_with_sink(&mut s, 2.5);
        let sum: f32 = s.iter().sum();
        assert!(sum < 1.0 && sum > 0.5, "sum {sum}");
    }

    #[test]
    fn swiglu_oai_clamps_and_shifts() {
        let mut g = vec![100.0f32]; // clamped to limit=7
        let up = vec![0.0f32]; // (0+1) => pure glu branch
        swiglu_oai(&mut g, &up, 1.702, 7.0);
        let expected = 7.0 / (1.0 + (-1.702f32 * 7.0).exp());
        assert!((g[0] - expected).abs() < 1e-6);
    }

    // For text (all four position axes equal) and a head as wide as the rotary
    // dims, M-RoPE with everything in one section must equal plain NEOX rope.
    #[test]
    fn mrope_text_collapses_to_plain_rope() {
        let r = YarnRope::new(64, 1e7, 1.0, 4096, 0.0, 1.0, 32.0, 1.0);
        let mut a: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin()).collect();
        let mut b = a.clone();
        r.apply(&mut a, 7);
        let pos = [7.0f32; 4];
        r.apply_mrope(&mut b, &pos, &[32, 0, 0, 0]);
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() < 1e-5, "{x} vs {y}");
        }
    }

    // Partial rotary: channels [n_rot, head_dim) must pass through untouched, and
    // distinct t/h/w positions (the vision case) must actually change the output.
    #[test]
    fn mrope_partial_passthrough_and_sections() {
        let (head_dim, n_rot) = (256usize, 64usize);
        let r = YarnRope::new(n_rot, 1e7, 1.0, 4096, 0.0, 1.0, 32.0, 1.0);
        let orig: Vec<f32> = (0..head_dim)
            .map(|i| (i as f32 * 0.03 + 1.0).cos())
            .collect();
        let mut text = orig.clone();
        let mut vision = orig.clone();
        r.apply_mrope(&mut text, &[5.0, 5.0, 5.0, 5.0], &[11, 11, 10, 0]);
        r.apply_mrope(&mut vision, &[5.0, 2.0, 9.0, 0.0], &[11, 11, 10, 0]);
        // dims past n_rot are copied verbatim on both
        for i in n_rot..head_dim {
            assert_eq!(text[i], orig[i]);
            assert_eq!(vision[i], orig[i]);
        }
        // rotated dims: text == its own token index everywhere; vision differs
        // wherever the height/width axes diverge from temporal.
        let diverged = (0..n_rot).any(|i| (text[i] - vision[i]).abs() > 1e-4);
        assert!(diverged, "distinct t/h/w positions left the head unchanged");
    }

    #[test]
    fn swiglu_plain_is_silu_times_up() {
        let mut g = vec![1.0f32, -2.0, 0.0];
        let up = vec![3.0f32, 5.0, 7.0];
        swiglu(&mut g, &up);
        let silu = |x: f32| x / (1.0 + (-x).exp());
        assert!((g[0] - silu(1.0) * 3.0).abs() < 1e-6);
        assert!((g[1] - silu(-2.0) * 5.0).abs() < 1e-6);
        assert!(g[2].abs() < 1e-6); // silu(0)=0
    }

    #[test]
    fn yarn_pos0_is_pure_mscale() {
        // at pos 0 every angle is 0: cos=mscale, sin=0 -> head scales by mscale
        let r = YarnRope::new(64, 150_000.0, 1.0 / 32.0, 4096, 1.0, 1.0, 32.0, 1.0);
        let mut head: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let orig = head.clone();
        r.apply(&mut head, 0);
        let mscale = 1.0 + 0.1 * 32f32.ln();
        for (h, o) in head.iter().zip(&orig) {
            assert!((h - o * mscale).abs() < 1e-4);
        }
    }
}
