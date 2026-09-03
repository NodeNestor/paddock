//! Illumination-direction estimation (Johnson&Farid 2005, Kee&Farid 2010),
//! ported verbatim from the CPU reference. CPU-only: the reference's device
//! path is a stub and the estimator is a small weighted-gradient + circular-stats
//! pass, so `gpu()` delegates to `cpu()`.
//!
//! Estimates the light direction per block and per intensity band (Lambertian
//! surfaces at different brightness should agree); disagreement across blocks
//! (Rayleigh test) or bands indicates compositing from different lighting.
//! Camera/scene-specific -> skipped for documents.

use crate::analyzer::Analyzer;
use crate::context::ContentType;
use crate::{Context, Finding, Severity};

pub struct IlluminationAnalyzer {
    block_size: usize,
    min_gradient_strength: f64,
    num_intensity_bands: usize,
    min_band_pixels: usize,
}

impl Default for IlluminationAnalyzer {
    fn default() -> Self {
        Self {
            block_size: 128,
            min_gradient_strength: 5.0,
            num_intensity_bands: 5,
            min_band_pixels: 500,
        }
    }
}

struct IlluminationEstimate {
    direction: f64,
    strength: f64,
    pixel_count: usize,
}

impl Analyzer for IlluminationAnalyzer {
    fn name(&self) -> &'static str {
        "illumination"
    }

    fn applies_to(&self, ctx: &Context) -> bool {
        !matches!(ctx.content_type, ContentType::Document)
    }

    fn cpu(&self, ctx: &Context) -> Vec<Finding> {
        let w = ctx.width as usize;
        let h = ctx.height as usize;

        if w < self.block_size * 2 || h < self.block_size * 2 {
            return vec![];
        }

        let gray = ctx.gray();
        let mut findings = Vec::new();

        let block_estimates = self.estimate_block_illumination(gray, w, h);
        if block_estimates.len() >= 4 {
            self.analyze_block_consistency(&block_estimates, &mut findings);
        }

        let band_estimates = self.estimate_per_band_illumination(gray, w, h);
        if band_estimates.len() >= 2 {
            self.analyze_band_consistency(&band_estimates, &mut findings);
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

impl IlluminationAnalyzer {
    fn estimate_block_illumination(
        &self,
        gray: &[u8],
        w: usize,
        h: usize,
    ) -> Vec<IlluminationEstimate> {
        let blocks_x = w / self.block_size;
        let blocks_y = h / self.block_size;
        let mut estimates = Vec::new();

        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                if let Some(est) = self.estimate_region_illumination(
                    gray,
                    w,
                    h,
                    bx * self.block_size,
                    by * self.block_size,
                    self.block_size,
                    self.block_size,
                    None,
                ) && est.strength > self.min_gradient_strength
                    && est.pixel_count > 100
                {
                    estimates.push(est);
                }
            }
        }

        estimates
    }

    fn estimate_per_band_illumination(
        &self,
        gray: &[u8],
        w: usize,
        h: usize,
    ) -> Vec<IlluminationEstimate> {
        let band_width = 256 / self.num_intensity_bands;
        let mut estimates = Vec::new();

        for band in 0..self.num_intensity_bands {
            let lo = (band * band_width) as u8;
            let hi = ((band + 1) * band_width - 1).min(255) as u8;

            if let Some(est) =
                self.estimate_region_illumination(gray, w, h, 1, 1, w - 2, h - 2, Some((lo, hi)))
                && est.pixel_count >= self.min_band_pixels
                && est.strength > self.min_gradient_strength
            {
                estimates.push(est);
            }
        }

        estimates
    }

    #[allow(clippy::too_many_arguments)]
    fn estimate_region_illumination(
        &self,
        gray: &[u8],
        w: usize,
        _h: usize,
        x0: usize,
        y0: usize,
        rw: usize,
        rh: usize,
        intensity_band: Option<(u8, u8)>,
    ) -> Option<IlluminationEstimate> {
        let mut gx_sum = 0.0_f64;
        let mut gy_sum = 0.0_f64;
        let mut magnitude_sum = 0.0_f64;
        let mut count = 0_usize;

        for dy in 1..rh.saturating_sub(1) {
            for dx in 1..rw.saturating_sub(1) {
                let x = x0 + dx;
                let y = y0 + dy;

                let pixel = gray[y * w + x];

                if let Some((lo, hi)) = intensity_band
                    && (pixel < lo || pixel > hi)
                {
                    continue;
                }

                let gx = -(gray[(y - 1) * w + (x - 1)] as f64)
                    + 1.0 * gray[(y - 1) * w + (x + 1)] as f64
                    + -2.0 * gray[y * w + (x - 1)] as f64
                    + 2.0 * gray[y * w + (x + 1)] as f64
                    + -(gray[(y + 1) * w + (x - 1)] as f64)
                    + 1.0 * gray[(y + 1) * w + (x + 1)] as f64;

                let gy = -(gray[(y - 1) * w + (x - 1)] as f64)
                    + -2.0 * gray[(y - 1) * w + x] as f64
                    + -(gray[(y - 1) * w + (x + 1)] as f64)
                    + 1.0 * gray[(y + 1) * w + (x - 1)] as f64
                    + 2.0 * gray[(y + 1) * w + x] as f64
                    + 1.0 * gray[(y + 1) * w + (x + 1)] as f64;

                let magnitude = (gx * gx + gy * gy).sqrt();

                if magnitude > self.min_gradient_strength {
                    let intensity = pixel as f64 / 255.0;
                    let weight = intensity * magnitude;

                    gx_sum += weight * gx;
                    gy_sum += weight * gy;
                    magnitude_sum += magnitude;
                    count += 1;
                }
            }
        }

        if count < 50 {
            return None;
        }

        Some(IlluminationEstimate {
            direction: gy_sum.atan2(gx_sum),
            strength: magnitude_sum / count as f64,
            pixel_count: count,
        })
    }

    fn analyze_block_consistency(
        &self,
        estimates: &[IlluminationEstimate],
        findings: &mut Vec<Finding>,
    ) {
        let n = estimates.len() as f64;

        let total_weight: f64 = estimates.iter().map(|e| e.strength).sum();
        let mut sin_sum = 0.0_f64;
        let mut cos_sum = 0.0_f64;

        for est in estimates {
            let w = est.strength / total_weight;
            sin_sum += w * est.direction.sin();
            cos_sum += w * est.direction.cos();
        }

        let mean_direction = sin_sum.atan2(cos_sum);
        let r_bar = (sin_sum * sin_sum + cos_sum * cos_sum).sqrt();
        let circular_variance = 1.0 - r_bar;

        let rayleigh_z = n * r_bar * r_bar;
        let rayleigh_p = (-rayleigh_z).exp();

        let outlier_threshold = std::f64::consts::PI / 4.0;
        let outliers = estimates
            .iter()
            .filter(|e| {
                let mut diff = e.direction - mean_direction;
                while diff > std::f64::consts::PI {
                    diff -= 2.0 * std::f64::consts::PI;
                }
                while diff < -std::f64::consts::PI {
                    diff += 2.0 * std::f64::consts::PI;
                }
                diff.abs() > outlier_threshold
            })
            .count();
        let outlier_ratio = outliers as f64 / n;

        if circular_variance > 0.3 && outlier_ratio > 0.1 {
            findings.push(Finding::new(
                "illumination",
                "illumination_inconsistency",
                format!(
                    "Illumination direction inconsistency: {:.1}% of blocks deviate >45° \
                     from dominant direction ({:.0}°), circular variance {:.3}, \
                     Rayleigh p={:.4} - indicates compositing from different lighting",
                    outlier_ratio * 100.0,
                    mean_direction.to_degrees(),
                    circular_variance,
                    rayleigh_p
                ),
                Severity::High,
                (0.45 + outlier_ratio * 0.4).min(0.80),
            ));
        }
    }

    fn analyze_band_consistency(
        &self,
        estimates: &[IlluminationEstimate],
        findings: &mut Vec<Finding>,
    ) {
        if estimates.len() < 2 {
            return;
        }

        let mut max_diff = 0.0_f64;
        let mut total_diff = 0.0_f64;
        let mut comparisons = 0;

        for i in 0..estimates.len() {
            for j in (i + 1)..estimates.len() {
                let mut diff = estimates[i].direction - estimates[j].direction;
                while diff > std::f64::consts::PI {
                    diff -= 2.0 * std::f64::consts::PI;
                }
                while diff < -std::f64::consts::PI {
                    diff += 2.0 * std::f64::consts::PI;
                }
                let abs_diff = diff.abs();

                if abs_diff > max_diff {
                    max_diff = abs_diff;
                }
                total_diff += abs_diff;
                comparisons += 1;
            }
        }

        if comparisons == 0 {
            return;
        }

        let mean_diff = total_diff / comparisons as f64;

        if mean_diff > std::f64::consts::PI / 6.0 {
            findings.push(Finding::new(
                "illumination",
                "illumination_band_inconsistency",
                format!(
                    "Illumination direction varies {:.0}° across intensity bands \
                     (max {:.0}°) - surfaces at different brightness levels show \
                     incompatible lighting, suggesting compositing",
                    mean_diff.to_degrees(),
                    max_diff.to_degrees()
                ),
                Severity::High,
                (0.40 + (mean_diff / std::f64::consts::PI) * 0.4).min(0.75),
            ));
        }
    }
}
