//! CFA / demosaicing artifact analysis (Popescu & Farid 2005), ported verbatim
//! from the CPU reference. CPU-only: it uses a 2D `rustfft` - a GPU cuFFT
//! would not reproduce it bit-for-bit, and the reference's own CFA GPU path is a
//! CPU-fallback stub - so `gpu()` delegates to `cpu()`.
//!
//! Cameras sample one color per pixel (Bayer CFA) and interpolate the rest,
//! leaving periodic Nyquist peaks in the 2D FFT of the interpolation residual.
//! Absent peaks -> not camera-captured (AI / heavy processing); per-block
//! absence or a different best-fit Bayer pattern -> editing / splice from another
//! camera. Camera-specific -> skipped for documents.

use num_complex::Complex;
use rustfft::FftPlanner;

use crate::analyzer::Analyzer;
use crate::context::ContentType;
use crate::{Context, Finding, Severity};

pub struct CfaAnalyzer {
    block_size: usize,
    inconsistency_threshold: f64,
    peak_threshold: f64,
}

impl Default for CfaAnalyzer {
    fn default() -> Self {
        Self {
            block_size: 64,
            inconsistency_threshold: 0.4,
            peak_threshold: 3.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BayerPattern {
    Rggb,
    Bggr,
    Grbg,
    Gbrg,
}

impl BayerPattern {
    const ALL: [BayerPattern; 4] = [
        BayerPattern::Rggb,
        BayerPattern::Bggr,
        BayerPattern::Grbg,
        BayerPattern::Gbrg,
    ];

    fn channel_at(&self, row_mod2: usize, col_mod2: usize) -> usize {
        match self {
            BayerPattern::Rggb => match (row_mod2, col_mod2) {
                (0, 0) => 0,
                (0, 1) => 1,
                (1, 0) => 1,
                (1, 1) => 2,
                _ => unreachable!(),
            },
            BayerPattern::Bggr => match (row_mod2, col_mod2) {
                (0, 0) => 2,
                (0, 1) => 1,
                (1, 0) => 1,
                (1, 1) => 0,
                _ => unreachable!(),
            },
            BayerPattern::Grbg => match (row_mod2, col_mod2) {
                (0, 0) => 1,
                (0, 1) => 0,
                (1, 0) => 2,
                (1, 1) => 1,
                _ => unreachable!(),
            },
            BayerPattern::Gbrg => match (row_mod2, col_mod2) {
                (0, 0) => 1,
                (0, 1) => 2,
                (1, 0) => 0,
                (1, 1) => 1,
                _ => unreachable!(),
            },
        }
    }
}

struct BlockCfaResult {
    peak_strength: f64,
    best_pattern: BayerPattern,
}

impl Analyzer for CfaAnalyzer {
    fn name(&self) -> &'static str {
        "cfa"
    }

    fn applies_to(&self, ctx: &Context) -> bool {
        // Camera-specific -> skip documents (which includes PDFs).
        !matches!(ctx.content_type, ContentType::Document)
    }

    fn cpu(&self, ctx: &Context) -> Vec<Finding> {
        let w = ctx.width as usize;
        let h = ctx.height as usize;

        if w < self.block_size * 3 || h < self.block_size * 3 {
            return vec![];
        }

        // Downsample large images before FFT (CFA peak detection is
        // scale-invariant); rebuild the equivalent RgbImage from ctx.image.
        const MAX_PIXELS: usize = 500_000;
        let pixel_count = w * h;
        let (effective_w, effective_h, rgb_owned);

        let pixels: &[u8] = if pixel_count > MAX_PIXELS {
            let scale = (MAX_PIXELS as f64 / pixel_count as f64).sqrt();
            let new_w = ((w as f64 * scale) as u32).max(self.block_size as u32 * 3);
            let new_h = ((h as f64 * scale) as u32).max(self.block_size as u32 * 3);
            let rgb_src = ctx.image.to_rgb8();
            let resized = image::imageops::resize(
                &rgb_src,
                new_w,
                new_h,
                image::imageops::FilterType::Triangle,
            );
            effective_w = new_w as usize;
            effective_h = new_h as usize;
            rgb_owned = resized.into_raw();
            &rgb_owned
        } else {
            let rgb = ctx.image.to_rgb8();
            effective_w = w;
            effective_h = h;
            rgb_owned = rgb.into_raw();
            &rgb_owned
        };

        let global_result = self.analyze_global_cfa(pixels, effective_w, effective_h);

        let blocks_x = effective_w / self.block_size;
        let blocks_y = effective_h / self.block_size;
        let mut block_results: Vec<BlockCfaResult> = Vec::with_capacity(blocks_x * blocks_y);

        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                let result = self.analyze_block_cfa(
                    pixels,
                    effective_w,
                    bx * self.block_size,
                    by * self.block_size,
                );
                block_results.push(result);
            }
        }

        let mut findings = Vec::new();

        if block_results.is_empty() {
            return findings;
        }

        let mean_strength: f64 =
            block_results.iter().map(|r| r.peak_strength).sum::<f64>() / block_results.len() as f64;

        if mean_strength < self.peak_threshold * 0.3 {
            findings.push(Finding::new(
                "cfa",
                "no_cfa_artifacts",
                format!(
                    "No CFA demosaicing artifacts detected in Fourier domain \
                     (mean spectral peak ratio {mean_strength:.2}) - image was likely not \
                     captured by a camera sensor"
                ),
                Severity::High,
                0.65,
            ));
        }

        if mean_strength >= self.peak_threshold * 0.3 {
            let var_strength: f64 = block_results
                .iter()
                .map(|r| (r.peak_strength - mean_strength).powi(2))
                .sum::<f64>()
                / block_results.len() as f64;
            let cv = var_strength.sqrt() / mean_strength.max(1e-10);

            if cv > self.inconsistency_threshold {
                let std_dev = var_strength.sqrt();
                let threshold_low = (mean_strength - 2.0 * std_dev).max(0.0);

                let absent_blocks = block_results
                    .iter()
                    .filter(|r| r.peak_strength < threshold_low)
                    .count();
                let absent_ratio = absent_blocks as f64 / block_results.len() as f64;

                if absent_ratio > 0.05 {
                    findings.push(Finding::new(
                        "cfa",
                        "cfa_peak_absent",
                        format!(
                            "CFA spectral peak absent in {:.1}% of blocks (CV={cv:.2}) - \
                             these regions lack demosaicing artifacts, indicating editing \
                             or compositing",
                            absent_ratio * 100.0,
                        ),
                        Severity::High,
                        0.70,
                    ));
                }
            }

            if let Some(dominant) = &global_result {
                let mismatched = block_results
                    .iter()
                    .filter(|r| {
                        r.peak_strength >= self.peak_threshold * 0.5
                            && r.best_pattern != dominant.best_pattern
                    })
                    .count();
                let mismatch_ratio = mismatched as f64 / block_results.len() as f64;

                if mismatch_ratio > 0.05 {
                    findings.push(Finding::new(
                        "cfa",
                        "cfa_pattern_mismatch",
                        format!(
                            "CFA Bayer pattern mismatch: {:.1}% of blocks show a different \
                             demosaicing pattern than the dominant {:?} - regions may be \
                             spliced from a different camera",
                            mismatch_ratio * 100.0,
                            dominant.best_pattern
                        ),
                        Severity::High,
                        0.75,
                    ));
                }
            }
        }

        findings
    }

    #[cfg(feature = "cuda")]
    fn gpu(
        &self,
        _gpu: &crate::gpu::ForensicGpu,
        ctx: &Context,
    ) -> Result<Vec<Finding>, crate::gpu::GpuError> {
        // No bit-exact GPU FFT; the reference's CFA GPU path is a CPU stub too.
        Ok(self.cpu(ctx))
    }
}

impl CfaAnalyzer {
    fn analyze_global_cfa(
        &self,
        rgb_pixels: &[u8],
        img_width: usize,
        img_height: usize,
    ) -> Option<BlockCfaResult> {
        let size = 256.min(img_width).min(img_height);
        if size < 64 {
            return None;
        }
        let x0 = (img_width - size) / 2;
        let y0 = (img_height - size) / 2;

        let mut best_pattern = BayerPattern::Rggb;
        let mut best_peak = 0.0_f64;

        for &pattern in &BayerPattern::ALL {
            let residual = self
                .compute_interpolation_residual(rgb_pixels, img_width, x0, y0, size, size, pattern);
            let peak = self.compute_fourier_peak(&residual, size, size);
            if peak > best_peak {
                best_peak = peak;
                best_pattern = pattern;
            }
        }

        Some(BlockCfaResult {
            peak_strength: best_peak,
            best_pattern,
        })
    }

    fn analyze_block_cfa(
        &self,
        rgb_pixels: &[u8],
        img_width: usize,
        x0: usize,
        y0: usize,
    ) -> BlockCfaResult {
        let bs = self.block_size;
        let mut best_pattern = BayerPattern::Rggb;
        let mut best_peak = 0.0_f64;

        for &pattern in &BayerPattern::ALL {
            let residual =
                self.compute_interpolation_residual(rgb_pixels, img_width, x0, y0, bs, bs, pattern);
            let peak = self.compute_fourier_peak(&residual, bs, bs);
            if peak > best_peak {
                best_peak = peak;
                best_pattern = pattern;
            }
        }

        BlockCfaResult {
            peak_strength: best_peak,
            best_pattern,
        }
    }

    fn compute_interpolation_residual(
        &self,
        rgb_pixels: &[u8],
        img_width: usize,
        x0: usize,
        y0: usize,
        block_w: usize,
        block_h: usize,
        pattern: BayerPattern,
    ) -> Vec<f64> {
        let mut residual = vec![0.0_f64; block_w * block_h];

        for dy in 1..block_h.saturating_sub(1) {
            for dx in 1..block_w.saturating_sub(1) {
                let x = x0 + dx;
                let y = y0 + dy;

                let _sampled_channel = pattern.channel_at(y % 2, x % 2);
                let target_channel = 1_usize; // green

                let idx = (y * img_width + x) * 3 + target_channel;
                let center = rgb_pixels[idx] as f64;

                let north = rgb_pixels[((y - 1) * img_width + x) * 3 + target_channel] as f64;
                let south = rgb_pixels[((y + 1) * img_width + x) * 3 + target_channel] as f64;
                let west = rgb_pixels[(y * img_width + (x - 1)) * 3 + target_channel] as f64;
                let east = rgb_pixels[(y * img_width + (x + 1)) * 3 + target_channel] as f64;

                let predicted = (north + south + west + east) / 4.0;
                residual[dy * block_w + dx] = center - predicted;
            }
        }

        residual
    }

    fn compute_fourier_peak(&self, residual: &[f64], width: usize, height: usize) -> f64 {
        if width < 4 || height < 4 {
            return 0.0;
        }

        let n = width * height;

        let mut data: Vec<Complex<f64>> = residual.iter().map(|&v| Complex::new(v, 0.0)).collect();

        let mut planner = FftPlanner::new();

        let row_fft = planner.plan_fft_forward(width);
        for row in 0..height {
            let start = row * width;
            let end = start + width;
            row_fft.process(&mut data[start..end]);
        }

        let col_fft = planner.plan_fft_forward(height);
        let mut col_buf = vec![Complex::new(0.0, 0.0); height];
        for col in 0..width {
            for row in 0..height {
                col_buf[row] = data[row * width + col];
            }
            col_fft.process(&mut col_buf);
            for row in 0..height {
                data[row * width + col] = col_buf[row];
            }
        }

        let magnitudes: Vec<f64> = data.iter().map(|c| c.norm()).collect();

        let total_mag: f64 = magnitudes.iter().sum::<f64>() - magnitudes[0];
        let mean_mag = total_mag / (n as f64 - 1.0).max(1.0);

        if mean_mag < 1e-10 {
            return 0.0;
        }

        let peak_positions = [(height / 2, width / 2), (0, width / 2), (height / 2, 0)];

        let mut max_peak_ratio = 0.0_f64;

        for &(row, col) in &peak_positions {
            let mut peak_sum = 0.0_f64;
            let mut peak_count = 0;

            for dr in 0..=2_usize {
                for dc in 0..=2_usize {
                    let r = (row + dr).wrapping_sub(1) % height;
                    let c = (col + dc).wrapping_sub(1) % width;
                    peak_sum += magnitudes[r * width + c];
                    peak_count += 1;
                }
            }

            let peak_avg = peak_sum / peak_count as f64;
            let ratio = peak_avg / mean_mag;

            if ratio > max_peak_ratio {
                max_peak_ratio = ratio;
            }
        }

        max_peak_ratio
    }
}
