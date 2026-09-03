//! Resampling / interpolation-artifact detection (Popescu & Farid 2005),
//! ported verbatim from the CPU reference. CPU-only (per-line
//! second-derivative autocovariance; no GPU kernel), `gpu()` delegates.
//!
//! Resizing or rotating an image interpolates neighbouring pixels, which leaves
//! periodic correlations in the second derivative's autocovariance. Global
//! artifacts are merely informational (a resized photo); a *mix* of resampled
//! and non-resampled blocks points at spliced content. Photo-oriented: skipped
//! for PDFs.

use crate::analyzer::Analyzer;
use crate::{Context, Finding, Severity};

pub struct ResamplingDetector {
    block_size: usize,
    min_peak_ratio: f64,
}

impl Default for ResamplingDetector {
    fn default() -> Self {
        Self {
            block_size: 64,
            min_peak_ratio: 2.5,
        }
    }
}

impl Analyzer for ResamplingDetector {
    fn name(&self) -> &'static str {
        "resampling"
    }

    fn applies_to(&self, ctx: &Context) -> bool {
        !ctx.is_pdf()
    }

    fn cpu(&self, ctx: &Context) -> Vec<Finding> {
        let w = ctx.width as usize;
        let h = ctx.height as usize;

        if w < self.block_size * 3 || h < self.block_size * 3 {
            return vec![];
        }

        let gray = ctx.gray();

        let global_h = self.detect_resampling_1d(gray, w, h, true);
        let global_v = self.detect_resampling_1d(gray, w, h, false);

        let mut findings = Vec::new();

        // Whole-image resampling is only informational (a resized photo).
        if global_h.detected || global_v.detected {
            let direction = match (global_h.detected, global_v.detected) {
                (true, true) => "both horizontal and vertical",
                (true, false) => "horizontal",
                (false, true) => "vertical",
                _ => unreachable!(),
            };

            findings.push(Finding::new(
                "resampling",
                "global_resampling",
                format!(
                    "Image shows {} resampling artifacts (H peak ratio: {:.2}, \
                     V peak ratio: {:.2}) - image has been resized or rotated",
                    direction, global_h.peak_ratio, global_v.peak_ratio
                ),
                Severity::Low,
                0.7,
            ));
        }

        // Block-wise: inconsistent resampling across blocks = manipulation.
        let blocks_x = w / self.block_size;
        let blocks_y = h / self.block_size;
        let mut block_resampled = Vec::with_capacity(blocks_x * blocks_y);

        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                let x0 = bx * self.block_size;
                let y0 = by * self.block_size;

                let mut block = vec![0u8; self.block_size * self.block_size];
                for dy in 0..self.block_size {
                    for dx in 0..self.block_size {
                        block[dy * self.block_size + dx] = gray[(y0 + dy) * w + (x0 + dx)];
                    }
                }

                let h_result =
                    self.detect_resampling_1d(&block, self.block_size, self.block_size, true);
                let v_result =
                    self.detect_resampling_1d(&block, self.block_size, self.block_size, false);

                let max_ratio = h_result.peak_ratio.max(v_result.peak_ratio);
                block_resampled.push((h_result.detected || v_result.detected, max_ratio));
            }
        }

        let resampled_count = block_resampled.iter().filter(|(d, _)| *d).count();
        let resampled_ratio = resampled_count as f64 / block_resampled.len() as f64;

        if resampled_ratio > 0.1 && resampled_ratio < 0.8 && block_resampled.len() >= 9 {
            findings.push(Finding::new(
                "resampling",
                "resampling_inconsistency",
                format!(
                    "Resampling artifact inconsistency: {:.1}% of blocks show \
                     interpolation artifacts while {:.1}% do not - indicates \
                     spliced content from a resized/rotated source",
                    resampled_ratio * 100.0,
                    (1.0 - resampled_ratio) * 100.0
                ),
                Severity::High,
                (0.55 + resampled_ratio * 0.3).min(0.85),
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

struct ResamplingResult {
    detected: bool,
    peak_ratio: f64,
}

impl ResamplingDetector {
    /// One-direction detection: periodicity of the second derivative's
    /// autocovariance.
    fn detect_resampling_1d(
        &self,
        pixels: &[u8],
        w: usize,
        h: usize,
        horizontal: bool,
    ) -> ResamplingResult {
        let (num_lines, line_len) = if horizontal { (h, w) } else { (w, h) };

        if line_len < 16 {
            return ResamplingResult {
                detected: false,
                peak_ratio: 0.0,
            };
        }

        let max_lag = 16.min(line_len / 4);
        let mut autocov = vec![0.0_f64; max_lag];
        let mut lines_processed = 0;

        let sample_step = (num_lines / 100).max(1); // Subsample lines for speed.

        for line_idx in (0..num_lines).step_by(sample_step) {
            let line: Vec<f64> = (0..line_len)
                .map(|i| {
                    let (x, y) = if horizontal {
                        (i, line_idx)
                    } else {
                        (line_idx, i)
                    };
                    pixels[y * w + x] as f64
                })
                .collect();

            // Second derivative: d2[i] = line[i+1] - 2*line[i] + line[i-1].
            let d2: Vec<f64> = (1..line_len - 1)
                .map(|i| line[i + 1] - 2.0 * line[i] + line[i - 1])
                .collect();

            if d2.is_empty() {
                continue;
            }

            let d2_mean: f64 = d2.iter().sum::<f64>() / d2.len() as f64;

            for lag in 0..max_lag {
                let mut cov = 0.0_f64;
                let mut count = 0;
                for i in 0..d2.len() - lag {
                    cov += (d2[i] - d2_mean) * (d2[i + lag] - d2_mean);
                    count += 1;
                }
                if count > 0 {
                    autocov[lag] += cov / count as f64;
                }
            }

            lines_processed += 1;
        }

        if lines_processed == 0 {
            return ResamplingResult {
                detected: false,
                peak_ratio: 0.0,
            };
        }

        for v in &mut autocov {
            *v /= lines_processed as f64;
        }

        // Peaks in autocovariance (lag 0 is always the largest, so skip it).
        let ac0 = autocov[0].abs().max(1e-10);

        let mut max_peak = 0.0_f64;
        for lag in 2..max_lag {
            let normalized = autocov[lag].abs() / ac0;
            if normalized > max_peak {
                max_peak = normalized;
            }
        }

        // Secondary periodic peaks add strength.
        let mut periodic_strength = 0.0_f64;
        if max_lag > 4 {
            for lag in 2..max_lag - 1 {
                let val = autocov[lag].abs();
                if val > autocov[lag - 1].abs() && val > autocov[lag + 1].abs() {
                    periodic_strength += val / ac0;
                }
            }
        }

        let peak_ratio = max_peak * (1.0 + periodic_strength);

        ResamplingResult {
            detected: peak_ratio > (1.0 / self.min_peak_ratio),
            peak_ratio,
        }
    }
}
