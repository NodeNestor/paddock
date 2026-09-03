//! PRNU sensor-fingerprint analysis (Lukas/Fridrich/Goljan 2006, Chen et al.
//! 2008), ported verbatim from the CPU reference. CPU-only: the reference's
//! device path is a stub, and the Daubechies-4 wavelet denoise + BayesShrink is a
//! serial multi-level transform, so `gpu()` delegates to `cpu()`.
//!
//! Extracts the PRNU residual (image - wavelet-denoised image), zero-means it,
//! and cross-correlates per-block patterns: blocks below the Grubbs threshold ->
//! a different sensor (splice/AI); a near-zero global correlation -> no physical
//! sensor at all; low PCE -> weak fingerprint. Camera-specific -> skipped for
//! documents.

use ndarray::Array2;

use crate::analyzer::Analyzer;
use crate::context::ContentType;
use crate::{Context, Finding, Severity};

pub struct PrnuAnalyzer {
    block_size: usize,
    decomposition_levels: usize,
}

impl Default for PrnuAnalyzer {
    fn default() -> Self {
        Self {
            block_size: 64,
            decomposition_levels: 4,
        }
    }
}

/// Max dimension for PRNU analysis; larger images are downsampled (PRNU is a
/// statistical property preserved at reduced resolution).
const MAX_PRNU_DIM: usize = 800;

// Daubechies-4 wavelet filter coefficients.
const DB4_LO: [f64; 8] = [
    -0.010597401784997278,
    0.032883011666982945,
    0.030841381835986965,
    -0.18703481171888114,
    -0.02798376941698385,
    0.6308807679295904,
    0.7148465705525415,
    0.23037781330885523,
];

const DB4_HI: [f64; 8] = [
    -0.23037781330885523,
    0.7148465705525415,
    -0.6308807679295904,
    -0.02798376941698385,
    0.18703481171888114,
    0.030841381835986965,
    -0.032883011666982945,
    -0.010597401784997278,
];

impl Analyzer for PrnuAnalyzer {
    fn name(&self) -> &'static str {
        "prnu"
    }

    fn applies_to(&self, ctx: &Context) -> bool {
        !matches!(ctx.content_type, ContentType::Document)
    }

    fn cpu(&self, ctx: &Context) -> Vec<Finding> {
        let orig_w = ctx.width as usize;
        let orig_h = ctx.height as usize;

        if orig_w < self.block_size * 2 || orig_h < self.block_size * 2 {
            return vec![];
        }

        let scale = (MAX_PRNU_DIM as f64 / orig_w.max(orig_h) as f64).min(1.0);
        let align = 1_usize << self.decomposition_levels; // 16 for 4 levels
        let (w, h, gray_owned): (usize, usize, Option<Vec<u8>>) = if scale < 1.0 {
            let dw = ((orig_w as f64 * scale) as usize) / align * align;
            let dh = ((orig_h as f64 * scale) as usize) / align * align;
            if dw < self.block_size * 2 || dh < self.block_size * 2 {
                return vec![];
            }
            let buf = Self::downsample_gray(ctx.gray(), orig_w, orig_h, dw, dh);
            (dw, dh, Some(buf))
        } else {
            let dw = orig_w / align * align;
            let dh = orig_h / align * align;
            if dw < orig_w || dh < orig_h {
                let mut buf = vec![0u8; dw * dh];
                for y in 0..dh {
                    buf[y * dw..(y + 1) * dw]
                        .copy_from_slice(&ctx.gray()[y * orig_w..y * orig_w + dw]);
                }
                (dw, dh, Some(buf))
            } else {
                (dw, dh, None)
            }
        };
        let gray: &[u8] = match &gray_owned {
            Some(buf) => buf,
            None => ctx.gray(),
        };

        let prnu = self.extract_prnu_wavelet(gray, w, h);
        let prnu = Self::normalize_prnu(prnu);

        let blocks_x = w / self.block_size;
        let blocks_y = h / self.block_size;
        let num_blocks = blocks_x * blocks_y;

        if num_blocks < 4 {
            return vec![];
        }

        let block_patterns: Vec<Vec<f64>> = (0..num_blocks)
            .map(|i| {
                let bx = i % blocks_x;
                let by = i / blocks_x;
                self.extract_block_pattern(&prnu, bx * self.block_size, by * self.block_size)
            })
            .collect();

        let mut all_correlations = Vec::new();
        let mut block_avg_corr = vec![0.0_f64; num_blocks];
        let mut block_corr_count = vec![0_usize; num_blocks];

        for i in 0..num_blocks {
            for j in (i + 1)..num_blocks {
                let corr =
                    Self::normalized_cross_correlation(&block_patterns[i], &block_patterns[j]);
                all_correlations.push(corr);
                block_avg_corr[i] += corr;
                block_avg_corr[j] += corr;
                block_corr_count[i] += 1;
                block_corr_count[j] += 1;
            }
        }

        for i in 0..num_blocks {
            if block_corr_count[i] > 0 {
                block_avg_corr[i] /= block_corr_count[i] as f64;
            }
        }

        let mut findings = Vec::new();

        let global_avg: f64 = if !all_correlations.is_empty() {
            all_correlations.iter().sum::<f64>() / all_correlations.len() as f64
        } else {
            return findings;
        };

        let mean_corr: f64 = block_avg_corr.iter().sum::<f64>() / num_blocks as f64;
        let var_corr: f64 = block_avg_corr
            .iter()
            .map(|&c| (c - mean_corr) * (c - mean_corr))
            .sum::<f64>()
            / num_blocks as f64;
        let std_corr = var_corr.sqrt();

        let grubbs_threshold = mean_corr - 2.5 * std_corr;

        let anomalous: Vec<usize> = block_avg_corr
            .iter()
            .enumerate()
            .filter(|&(_, &c)| c < grubbs_threshold)
            .map(|(i, _)| i)
            .collect();
        let anomalous_ratio = anomalous.len() as f64 / num_blocks as f64;

        if anomalous_ratio > 0.05 && anomalous_ratio < 0.7 {
            findings.push(Finding::new(
                "prnu",
                "prnu_inconsistency",
                format!(
                    "PRNU sensor fingerprint inconsistency: {:.1}% of blocks show \
                     different sensor noise patterns (wavelet-based extraction, \
                     global correlation {:.4}, {} anomalous blocks below Grubbs threshold) - \
                     indicates content from different cameras or synthetic generation",
                    anomalous_ratio * 100.0,
                    global_avg,
                    anomalous.len()
                ),
                Severity::Critical,
                (0.6 + anomalous_ratio * 0.5).min(0.92),
            ));
        }

        if (0.0..0.05).contains(&global_avg) {
            findings.push(Finding::new(
                "prnu",
                "prnu_absent",
                format!(
                    "No detectable PRNU sensor fingerprint (mean correlation {:.4}, \
                     wavelet denoising with {} decomposition levels) - \
                     image was likely not captured by a physical camera sensor",
                    global_avg, self.decomposition_levels
                ),
                Severity::High,
                0.60,
            ));
        }

        let pce = self.compute_pce(&all_correlations);
        if pce < 30.0 && global_avg > 0.01 {
            findings.push(Finding::new(
                "prnu",
                "low_pce",
                format!(
                    "Low Peak-to-Correlation Energy ({pce:.1}) indicates weak sensor \
                     fingerprint - below threshold for confident camera attribution"
                ),
                Severity::Medium,
                0.50,
            ));
        }

        findings
    }

    #[cfg(feature = "cuda")]
    fn gpu(
        &self,
        _gpu: &crate::gpu::ForensicGpu,
        ctx: &Context,
    ) -> Result<Vec<Finding>, crate::gpu::GpuError> {
        Ok(self.cpu(ctx))
    }
}

impl PrnuAnalyzer {
    fn downsample_gray(src: &[u8], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<u8> {
        let mut dst = vec![0u8; dw * dh];
        let x_ratio = sw as f64 / dw as f64;
        let y_ratio = sh as f64 / dh as f64;
        for dy in 0..dh {
            let src_y = dy as f64 * y_ratio;
            let y0 = src_y as usize;
            let y1 = (y0 + 1).min(sh - 1);
            let fy = src_y - y0 as f64;
            let nfy = 1.0 - fy;
            for dx in 0..dw {
                let src_x = dx as f64 * x_ratio;
                let x0 = src_x as usize;
                let x1 = (x0 + 1).min(sw - 1);
                let fx = src_x - x0 as f64;
                let nfx = 1.0 - fx;
                let v = nfy * (nfx * src[y0 * sw + x0] as f64 + fx * src[y0 * sw + x1] as f64)
                    + fy * (nfx * src[y1 * sw + x0] as f64 + fx * src[y1 * sw + x1] as f64);
                dst[dy * dw + dx] = v as u8;
            }
        }
        dst
    }

    fn extract_prnu_wavelet(&self, gray: &[u8], orig_w: usize, orig_h: usize) -> Array2<f64> {
        let w = orig_w & !1;
        let h = orig_h & !1;

        let mut image_flat = vec![0.0_f64; h * w];
        for y in 0..h {
            for x in 0..w {
                image_flat[y * w + x] = gray[y * orig_w + x] as f64;
            }
        }

        let denoised_flat = self.wavelet_denoise_flat(&image_flat, w, h);

        let mut prnu = Array2::<f64>::zeros((h, w));
        {
            let prnu_slice = prnu.as_slice_mut().expect("Array2 must be contiguous");
            for i in 0..h * w {
                prnu_slice[i] = image_flat[i] - denoised_flat[i];
            }
        }

        prnu
    }

    fn wavelet_denoise_flat(&self, image: &[f64], w: usize, h: usize) -> Vec<f64> {
        let mut current: Vec<f64> = image.to_vec();
        let mut details: Vec<(Vec<f64>, Vec<f64>, Vec<f64>, usize, usize)> = Vec::new();

        let mut cw = w;
        let mut ch = h;

        let mut tmp_lo = Vec::<f64>::new();
        let mut tmp_hi = Vec::<f64>::new();

        for _level in 0..self.decomposition_levels {
            if cw < 16 || ch < 16 {
                break;
            }

            let hw = cw / 2;
            let hh = ch / 2;

            let mut lo_rows = vec![0.0_f64; ch * hw];
            let mut hi_rows = vec![0.0_f64; ch * hw];

            tmp_lo.resize(hw, 0.0);
            tmp_hi.resize(hw, 0.0);

            for y in 0..ch {
                let row = &current[y * cw..(y + 1) * cw];
                Self::dwt_1d_into(row, &mut tmp_lo, &mut tmp_hi);
                lo_rows[y * hw..(y + 1) * hw].copy_from_slice(&tmp_lo[..hw]);
                hi_rows[y * hw..(y + 1) * hw].copy_from_slice(&tmp_hi[..hw]);
            }

            let mut ll = vec![0.0_f64; hh * hw];
            let mut lh = vec![0.0_f64; hh * hw];
            let mut hl = vec![0.0_f64; hh * hw];
            let mut hh_buf = vec![0.0_f64; hh * hw];

            tmp_lo.resize(hh, 0.0);
            tmp_hi.resize(hh, 0.0);

            let mut col_lo = vec![0.0_f64; ch];
            let mut col_hi = vec![0.0_f64; ch];
            let mut col_lo_out = vec![0.0_f64; hh];
            let mut col_hi_out = vec![0.0_f64; hh];
            let mut col_ll_out = vec![0.0_f64; hh];
            let mut col_hh_out = vec![0.0_f64; hh];

            for x in 0..hw {
                for y in 0..ch {
                    col_lo[y] = lo_rows[y * hw + x];
                    col_hi[y] = hi_rows[y * hw + x];
                }
                Self::dwt_1d_into(&col_lo[..ch], &mut col_lo_out, &mut col_hi_out);
                Self::dwt_1d_into(&col_hi[..ch], &mut col_ll_out, &mut col_hh_out);
                for y in 0..hh {
                    ll[y * hw + x] = col_lo_out[y];
                    lh[y * hw + x] = col_hi_out[y];
                    hl[y * hw + x] = col_ll_out[y];
                    hh_buf[y * hw + x] = col_hh_out[y];
                }
            }

            let thresh_lh = Self::bayes_shrink_flat(&lh);
            let thresh_hl = Self::bayes_shrink_flat(&hl);
            let thresh_hh = Self::bayes_shrink_flat(&hh_buf);
            Self::soft_threshold_flat_inplace(&mut lh, thresh_lh);
            Self::soft_threshold_flat_inplace(&mut hl, thresh_hl);
            Self::soft_threshold_flat_inplace(&mut hh_buf, thresh_hh);

            details.push((lh, hl, hh_buf, hw, hh));

            cw = hw;
            ch = hh;
            current = ll;
        }

        for (detail_lh, detail_hl, detail_hh, hw, hh) in details.into_iter().rev() {
            let out_w = hw * 2;
            let out_h = hh * 2;

            let mut lo_rows = vec![0.0_f64; out_h * hw];
            let mut hi_rows = vec![0.0_f64; out_h * hw];

            let mut col_ll = vec![0.0_f64; hh];
            let mut col_lh = vec![0.0_f64; hh];
            let mut col_hl = vec![0.0_f64; hh];
            let mut col_hh_v = vec![0.0_f64; hh];
            let mut col_lo_out = vec![0.0_f64; out_h];
            let mut col_hi_out = vec![0.0_f64; out_h];

            for x in 0..hw {
                for y in 0..hh {
                    col_ll[y] = current[y * hw + x];
                    col_lh[y] = detail_lh[y * hw + x];
                    col_hl[y] = detail_hl[y * hw + x];
                    col_hh_v[y] = detail_hh[y * hw + x];
                }
                Self::idwt_1d_into(&col_ll, &col_lh, out_h, &mut col_lo_out);
                Self::idwt_1d_into(&col_hl, &col_hh_v, out_h, &mut col_hi_out);
                for y in 0..out_h {
                    lo_rows[y * hw + x] = col_lo_out[y];
                    hi_rows[y * hw + x] = col_hi_out[y];
                }
            }

            let mut output = vec![0.0_f64; out_h * out_w];
            let mut lo_row = vec![0.0_f64; hw];
            let mut hi_row = vec![0.0_f64; hw];
            let mut row_out = vec![0.0_f64; out_w];

            for y in 0..out_h {
                lo_row.copy_from_slice(&lo_rows[y * hw..(y + 1) * hw]);
                hi_row.copy_from_slice(&hi_rows[y * hw..(y + 1) * hw]);
                Self::idwt_1d_into(&lo_row, &hi_row, out_w, &mut row_out);
                output[y * out_w..(y + 1) * out_w].copy_from_slice(&row_out);
            }

            current = output;
        }

        current
    }

    fn dwt_1d_into(signal: &[f64], lo: &mut Vec<f64>, hi: &mut Vec<f64>) {
        let n = signal.len();
        let half = n / 2;
        lo.resize(half, 0.0);
        hi.resize(half, 0.0);

        let safe_end = if n >= 8 { (n - 8) / 2 + 1 } else { 0 };

        for i in 0..safe_end {
            let base = 2 * i;
            let mut lo_sum = 0.0_f64;
            let mut hi_sum = 0.0_f64;
            lo_sum += DB4_LO[0] * signal[base];
            lo_sum += DB4_LO[1] * signal[base + 1];
            lo_sum += DB4_LO[2] * signal[base + 2];
            lo_sum += DB4_LO[3] * signal[base + 3];
            lo_sum += DB4_LO[4] * signal[base + 4];
            lo_sum += DB4_LO[5] * signal[base + 5];
            lo_sum += DB4_LO[6] * signal[base + 6];
            lo_sum += DB4_LO[7] * signal[base + 7];
            hi_sum += DB4_HI[0] * signal[base];
            hi_sum += DB4_HI[1] * signal[base + 1];
            hi_sum += DB4_HI[2] * signal[base + 2];
            hi_sum += DB4_HI[3] * signal[base + 3];
            hi_sum += DB4_HI[4] * signal[base + 4];
            hi_sum += DB4_HI[5] * signal[base + 5];
            hi_sum += DB4_HI[6] * signal[base + 6];
            hi_sum += DB4_HI[7] * signal[base + 7];
            lo[i] = lo_sum;
            hi[i] = hi_sum;
        }

        for i in safe_end..half {
            let mut lo_sum = 0.0_f64;
            let mut hi_sum = 0.0_f64;
            for k in 0..8 {
                let idx = (2 * i + k) % n;
                lo_sum += DB4_LO[k] * signal[idx];
                hi_sum += DB4_HI[k] * signal[idx];
            }
            lo[i] = lo_sum;
            hi[i] = hi_sum;
        }
    }

    fn idwt_1d_into(lo: &[f64], hi: &[f64], output_len: usize, output: &mut Vec<f64>) {
        output.resize(output_len, 0.0);
        for v in output.iter_mut() {
            *v = 0.0;
        }
        let half = lo.len();
        let safe_end = if output_len >= 8 {
            (output_len - 8) / 2 + 1
        } else {
            0
        };

        for i in 0..safe_end.min(half) {
            let base = 2 * i;
            let lo_v = lo[i];
            let hi_v = hi[i];
            output[base] += DB4_LO[0] * lo_v + DB4_HI[0] * hi_v;
            output[base + 1] += DB4_LO[1] * lo_v + DB4_HI[1] * hi_v;
            output[base + 2] += DB4_LO[2] * lo_v + DB4_HI[2] * hi_v;
            output[base + 3] += DB4_LO[3] * lo_v + DB4_HI[3] * hi_v;
            output[base + 4] += DB4_LO[4] * lo_v + DB4_HI[4] * hi_v;
            output[base + 5] += DB4_LO[5] * lo_v + DB4_HI[5] * hi_v;
            output[base + 6] += DB4_LO[6] * lo_v + DB4_HI[6] * hi_v;
            output[base + 7] += DB4_LO[7] * lo_v + DB4_HI[7] * hi_v;
        }

        for i in safe_end.min(half)..half {
            let lo_v = lo[i];
            let hi_v = hi[i];
            for k in 0..8 {
                let idx = (2 * i + k) % output_len;
                output[idx] += DB4_LO[k] * lo_v + DB4_HI[k] * hi_v;
            }
        }
    }

    fn bayes_shrink_flat(detail: &[f64]) -> f64 {
        let n = detail.len();
        if n < 4 {
            return 0.0;
        }
        let nf = n as f64;

        let mut abs_values: Vec<f64> = detail.iter().map(|&v| v.abs()).collect();
        abs_values.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median_abs = abs_values[n / 2];
        let sigma_noise = median_abs / 0.6745;

        let mean: f64 = detail.iter().sum::<f64>() / nf;
        let total_var: f64 = detail.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / nf;

        let sigma_signal_sq = (total_var - sigma_noise * sigma_noise).max(0.0);

        if sigma_signal_sq < 1e-10 {
            return sigma_noise * (2.0 * nf.ln()).sqrt();
        }

        sigma_noise * sigma_noise / sigma_signal_sq.sqrt()
    }

    fn soft_threshold_flat_inplace(coeffs: &mut [f64], threshold: f64) {
        for v in coeffs.iter_mut() {
            let abs_v = v.abs();
            *v = if abs_v > threshold {
                v.signum() * (abs_v - threshold)
            } else {
                0.0
            };
        }
    }

    fn normalize_prnu(mut prnu: Array2<f64>) -> Array2<f64> {
        let (h, w) = (prnu.shape()[0], prnu.shape()[1]);
        {
            let s = prnu.as_slice_mut().expect("Array2 must be contiguous");

            for y in 0..h {
                let row = &mut s[y * w..(y + 1) * w];
                let row_mean: f64 = row.iter().sum::<f64>() / w as f64;
                for v in row.iter_mut() {
                    *v -= row_mean;
                }
            }

            let mut col_sums = vec![0.0_f64; w];
            for row_chunk in s.chunks_exact(w) {
                for (x, &v) in row_chunk.iter().enumerate() {
                    col_sums[x] += v;
                }
            }
            let inv_h = 1.0 / h as f64;
            let col_means: Vec<f64> = col_sums.iter().map(|&s| s * inv_h).collect();
            for row_chunk in s.chunks_exact_mut(w) {
                for (x, v) in row_chunk.iter_mut().enumerate() {
                    *v -= col_means[x];
                }
            }
        }
        prnu
    }

    fn extract_block_pattern(&self, prnu: &Array2<f64>, x0: usize, y0: usize) -> Vec<f64> {
        let prnu_h = prnu.shape()[0];
        let prnu_w = prnu.shape()[1];
        let bs = self.block_size;
        let mut pattern = vec![0.0_f64; bs * bs];

        let flat = prnu.as_slice().expect("Array2 must be contiguous");
        for dy in 0..bs {
            let y = y0 + dy;
            if y >= prnu_h {
                break;
            }
            let x_end = (x0 + bs).min(prnu_w);
            let src_len = x_end.saturating_sub(x0);
            let dst_off = dy * bs;
            pattern[dst_off..dst_off + src_len]
                .copy_from_slice(&flat[y * prnu_w + x0..y * prnu_w + x_end]);
        }

        pattern
    }

    fn normalized_cross_correlation(a: &[f64], b: &[f64]) -> f64 {
        let n = a.len() as f64;
        if n == 0.0 {
            return 0.0;
        }

        let mean_a: f64 = a.iter().sum::<f64>() / n;
        let mean_b: f64 = b.iter().sum::<f64>() / n;

        let mut cov = 0.0_f64;
        let mut var_a = 0.0_f64;
        let mut var_b = 0.0_f64;

        for i in 0..a.len() {
            let da = a[i] - mean_a;
            let db = b[i] - mean_b;
            cov += da * db;
            var_a += da * da;
            var_b += db * db;
        }

        let denom = (var_a * var_b).sqrt();
        if denom > 1e-10 { cov / denom } else { 0.0 }
    }

    fn compute_pce(&self, correlations: &[f64]) -> f64 {
        if correlations.is_empty() {
            return 0.0;
        }

        let mut sorted: Vec<f64> = correlations.iter().map(|&c| c.abs()).collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let peak = sorted.last().copied().unwrap_or(0.0);
        let peak_sq = peak * peak;

        let n = sorted.len();
        if n < 2 {
            return peak_sq;
        }

        let mean_sq: f64 = sorted[..n - 1].iter().map(|&c| c * c).sum::<f64>() / (n - 1) as f64;

        if mean_sq > 1e-15 {
            peak_sq / mean_sq
        } else {
            0.0
        }
    }
}
