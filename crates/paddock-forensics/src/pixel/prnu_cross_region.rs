//! PRNU cross-region consistency analysis, ported verbatim from the CPU
//! reference. CPU-only (residual extraction + pairwise correlation; no GPU
//! kernel), `gpu()` delegates.
//!
//! An authentic image shares one sensor's PRNU pattern across every region, so
//! the noise residuals correlate. A region spliced from a different camera has
//! uncorrelated residuals. We divide the image into a grid, extract a simplified
//! residual (pixel - 3×3 neighbour mean), cross-correlate all region pairs, and
//! flag regions whose mean correlation is a robust (MAD) outlier below the floor.
//! Camera-specific: skipped for documents.

use crate::analyzer::Analyzer;
use crate::context::ContentType;
use crate::{Context, Finding, Severity};

pub struct PrnuCrossRegionAnalyzer {
    /// Horizontal divisions.
    grid_x: usize,
    /// Vertical divisions.
    grid_y: usize,
    /// Correlation floor below which a region is suspect.
    min_correlation: f64,
}

impl Default for PrnuCrossRegionAnalyzer {
    fn default() -> Self {
        Self {
            grid_x: 4,
            grid_y: 3,
            min_correlation: 0.15,
        }
    }
}

impl Analyzer for PrnuCrossRegionAnalyzer {
    fn name(&self) -> &'static str {
        "prnu_cross_region"
    }

    fn applies_to(&self, ctx: &Context) -> bool {
        // Camera-specific -> skip documents (which includes PDFs).
        !matches!(ctx.content_type, ContentType::Document)
    }

    fn cpu(&self, ctx: &Context) -> Vec<Finding> {
        let (width, height) = (ctx.width as usize, ctx.height as usize);
        let region_w = width / self.grid_x;
        let region_h = height / self.grid_y;

        if region_w < 64 || region_h < 64 {
            return Vec::new();
        }

        let gray = ctx.gray();

        let residual = self.extract_residual(gray, width, height);

        let num_regions = self.grid_x * self.grid_y;
        let mut correlations: Vec<Vec<f64>> = vec![vec![0.0; num_regions]; num_regions];

        for i in 0..num_regions {
            let (ix, iy) = (i % self.grid_x, i / self.grid_x);
            let i_x0 = ix * region_w;
            let i_y0 = iy * region_h;

            for j in (i + 1)..num_regions {
                let (jx, jy) = (j % self.grid_x, j / self.grid_x);
                let j_x0 = jx * region_w;
                let j_y0 = jy * region_h;

                let corr = self
                    .cross_correlate(&residual, width, i_x0, i_y0, j_x0, j_y0, region_w, region_h);

                correlations[i][j] = corr;
                correlations[j][i] = corr;
            }
        }

        self.analyze_correlations(&correlations, num_regions)
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

impl PrnuCrossRegionAnalyzer {
    /// Simplified PRNU residual: pixel - 3×3 neighbour mean (center excluded).
    fn extract_residual(&self, gray: &[u8], width: usize, height: usize) -> Vec<f64> {
        let mut residual = vec![0.0_f64; width * height];

        for y in 1..height - 1 {
            for x in 1..width - 1 {
                let center = gray[y * width + x] as f64;

                let mut sum = 0.0_f64;
                for ky in -1_i32..=1 {
                    for kx in -1_i32..=1 {
                        if kx == 0 && ky == 0 {
                            continue;
                        }
                        sum += gray[(y as i32 + ky) as usize * width + (x as i32 + kx) as usize]
                            as f64;
                    }
                }
                let local_mean = sum / 8.0;

                residual[y * width + x] = center - local_mean;
            }
        }

        residual
    }

    /// Normalized cross-correlation of residuals between two regions.
    #[allow(clippy::too_many_arguments)]
    fn cross_correlate(
        &self,
        residual: &[f64],
        stride: usize,
        x0_a: usize,
        y0_a: usize,
        x0_b: usize,
        y0_b: usize,
        region_w: usize,
        region_h: usize,
    ) -> f64 {
        let mut sum_ab = 0.0_f64;
        let mut sum_aa = 0.0_f64;
        let mut sum_bb = 0.0_f64;
        let mut count = 0_u64;

        let step = ((region_w * region_h) / 10000).max(1);
        let mut idx = 0;

        for dy in 0..region_h {
            for dx in 0..region_w {
                idx += 1;
                if idx % step != 0 {
                    continue;
                }

                let xa = x0_a + dx;
                let ya = y0_a + dy;
                let xb = x0_b + dx;
                let yb = y0_b + dy;

                let a = residual[ya * stride + xa];
                let b = residual[yb * stride + xb];

                sum_ab += a * b;
                sum_aa += a * a;
                sum_bb += b * b;
                count += 1;
            }
        }

        if count < 100 {
            return 0.0;
        }

        let denom = (sum_aa * sum_bb).max(1e-10).sqrt();
        sum_ab / denom
    }

    fn analyze_correlations(&self, correlations: &[Vec<f64>], num_regions: usize) -> Vec<Finding> {
        let mut findings = Vec::new();

        if num_regions < 4 {
            return findings;
        }

        // Each region's mean correlation with all others.
        let mut region_mean_corr: Vec<f64> = Vec::new();
        for i in 0..num_regions {
            let mut sum = 0.0_f64;
            let mut count = 0;
            for j in 0..num_regions {
                if i == j {
                    continue;
                }
                sum += correlations[i][j];
                count += 1;
            }
            region_mean_corr.push(if count > 0 { sum / count as f64 } else { 0.0 });
        }

        let mut sorted = region_mean_corr.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = sorted[sorted.len() / 2];
        let mad = {
            let mut devs: Vec<f64> = region_mean_corr
                .iter()
                .map(|c| (c - median).abs())
                .collect();
            devs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            devs[devs.len() / 2] * 1.4826
        };

        if mad < 1e-6 {
            return findings;
        }

        let mut low_corr_regions = 0;
        let mut min_corr = f64::MAX;
        let mut min_region = 0;

        for (i, &corr) in region_mean_corr.iter().enumerate() {
            let z = (median - corr) / mad;
            if z > 2.5 {
                low_corr_regions += 1;
                if corr < min_corr {
                    min_corr = corr;
                    min_region = i;
                }
            }
        }

        if low_corr_regions >= 1 && min_corr < self.min_correlation {
            let rx = min_region % self.grid_x;
            let ry = min_region / self.grid_x;

            findings.push(Finding::new(
                "prnu_cross_region",
                "prnu_cross_region_mismatch",
                format!(
                    "{low_corr_regions} of {num_regions} image regions show uncorrelated \
                     sensor noise residuals (min correlation {min_corr:.3} at region [{rx},{ry}]) - \
                     possible splice from a different camera sensor"
                ),
                Severity::High,
                (0.50 + (self.min_correlation - min_corr) * 2.0).min(0.85),
            ));
        }

        findings
    }
}
