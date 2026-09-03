//! Noise-consistency analysis via wavelet (Haar-HH) noise estimation, ported
//! from the CPU reference (Mahdian & Saic 2009 block-wise inconsistency;
//! Donoho & Johnstone 1994 MAD noise level).
//!
//! Canonical algorithm: per block, σ = median(|Haar HH|)/0.6745; then robust
//! cross-block statistics (IQR outliers, robust CV, excess kurtosis). The
//! reference's GPU path used a *different* (variance) estimator; paddock does not copy that
//! divergence - the GPU kernel computes the same per-block median(|HH|) (exact,
//! since HH values are multiples of 0.5), and the `/0.6745` + all cross-block
//! statistics run host-side on identical medians. Exact GPU==CPU parity.

use crate::analyzer::Analyzer;
use crate::{Context, Finding, Severity};

/// MAD->σ scaling for a normal distribution (Donoho-Johnstone).
const MAD_TO_SIGMA: f64 = 0.6745;

pub struct NoiseAnalyzer {
    block_size: usize,
    inconsistency_threshold: f64,
}

impl Default for NoiseAnalyzer {
    fn default() -> Self {
        Self {
            block_size: 32,
            inconsistency_threshold: 0.5,
        }
    }
}

impl Analyzer for NoiseAnalyzer {
    fn name(&self) -> &'static str {
        "noise"
    }

    /// Raster only - PDFs carry no single decoded image.
    fn applies_to(&self, ctx: &Context) -> bool {
        !ctx.is_pdf()
    }

    fn cpu(&self, ctx: &Context) -> Vec<Finding> {
        let (w, h) = (ctx.width as usize, ctx.height as usize);
        if w < self.block_size * 3 || h < self.block_size * 3 {
            return Vec::new();
        }
        let gray = ctx.gray();
        let (blocks_x, blocks_y) = (w / self.block_size, h / self.block_size);
        let mut sigmas = Vec::with_capacity(blocks_x * blocks_y);
        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                let median =
                    self.block_median_hh_cpu(gray, w, bx * self.block_size, by * self.block_size);
                sigmas.push(median / MAD_TO_SIGMA);
            }
        }
        self.emit_findings(&sigmas)
    }

    #[cfg(feature = "cuda")]
    fn gpu(
        &self,
        gpu: &crate::gpu::ForensicGpu,
        ctx: &Context,
    ) -> Result<Vec<Finding>, crate::gpu::GpuError> {
        use cudarc::driver::{LaunchConfig, PushKernelArg};

        let (w, h) = (ctx.width as usize, ctx.height as usize);
        if w < self.block_size * 3 || h < self.block_size * 3 {
            return Ok(Vec::new());
        }
        // The bitonic kernel is fixed at 256 lanes -> (block_size/2)^2 <= 256.
        if self.block_size == 0 || self.block_size > 32 || self.block_size % 2 != 0 {
            return Err(crate::gpu::GpuError::Other(format!(
                "noise GPU kernel supports even block_size <= 32, got {}",
                self.block_size
            )));
        }

        let gray = ctx.gray();
        let (blocks_x, blocks_y) = (w / self.block_size, h / self.block_size);
        let nblocks = blocks_x * blocks_y;
        let stream = gpu.stream();

        let d_gray = stream.clone_htod(gray)?;
        let mut d_out = stream.alloc_zeros::<f32>(nblocks)?;
        let (w_u, bs_u) = (w as u32, self.block_size as u32);
        let f = gpu.function("noise", "noise_block_median")?;
        let cfg = LaunchConfig {
            grid_dim: (blocks_x as u32, blocks_y as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(&f)
                .arg(&d_gray)
                .arg(&mut d_out)
                .arg(&w_u)
                .arg(&bs_u)
                .launch(cfg)?;
        }
        let medians: Vec<f32> = stream.clone_dtoh(&d_out)?;
        stream.synchronize()?;

        let sigmas: Vec<f64> = medians.iter().map(|&m| m as f64 / MAD_TO_SIGMA).collect();
        Ok(self.emit_findings(&sigmas))
    }
}

impl NoiseAnalyzer {
    /// Median of |Haar HH| coefficients over a block (CPU reference).
    fn block_median_hh_cpu(&self, gray: &[u8], img_w: usize, x0: usize, y0: usize) -> f64 {
        let bs = self.block_size;
        let half = bs / 2;
        if half < 2 {
            return 0.0;
        }
        let mut hh = Vec::with_capacity(half * half);
        for hy in 0..half {
            for hx in 0..half {
                let ay = y0 + 2 * hy;
                let ax = x0 + 2 * hx;
                let a = gray[ay * img_w + ax] as f64;
                let b = gray[ay * img_w + ax + 1] as f64;
                let c = gray[(ay + 1) * img_w + ax] as f64;
                let d = gray[(ay + 1) * img_w + ax + 1] as f64;
                hh.push(((a - b - c + d) / 2.0).abs());
            }
        }
        if hh.is_empty() {
            return 0.0;
        }
        hh.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        hh[hh.len() / 2]
    }

    /// Cross-block robust statistics over per-block σ values -> findings. Shared
    /// verbatim by the CPU and GPU paths (the only difference upstream is how
    /// each per-block median was computed).
    fn emit_findings(&self, block_noise: &[f64]) -> Vec<Finding> {
        let mut findings = Vec::new();
        if block_noise.is_empty() {
            return findings;
        }
        let total_blocks = block_noise.len() as f64;

        let mut sorted = block_noise.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median_noise = sorted[sorted.len() / 2];
        let q1 = sorted[sorted.len() / 4];
        let q3 = sorted[3 * sorted.len() / 4];
        let iqr = q3 - q1;
        let lower_fence = q1 - 1.5 * iqr;
        let upper_fence = q3 + 1.5 * iqr;

        let anomalous_low = block_noise
            .iter()
            .filter(|&&n| n < lower_fence && n < median_noise * 0.5)
            .count();
        let anomalous_high = block_noise
            .iter()
            .filter(|&&n| n > upper_fence && n > median_noise * 1.5)
            .count();
        let anomalous_ratio = (anomalous_low + anomalous_high) as f64 / total_blocks;

        let mad = {
            let mut abs_devs: Vec<f64> = block_noise
                .iter()
                .map(|&n| (n - median_noise).abs())
                .collect();
            abs_devs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            abs_devs[abs_devs.len() / 2]
        };
        let robust_cv = if median_noise > 0.0 {
            mad / median_noise
        } else {
            0.0
        };

        if robust_cv > self.inconsistency_threshold || anomalous_ratio > 0.08 {
            findings.push(Finding::new(
                "noise",
                "noise_inconsistency",
                format!(
                    "Noise level inconsistency detected via wavelet MAD estimation (median σ={median_noise:.2}, \
                     robust CV={robust_cv:.2}, {:.1}% anomalous blocks: {anomalous_low} low-noise, \
                     {anomalous_high} high-noise) - suggests splicing or AI generation",
                    anomalous_ratio * 100.0
                ),
                if anomalous_ratio > 0.15 { Severity::High } else { Severity::Medium },
                (0.5 + anomalous_ratio).min(0.85),
            ));
        }

        if median_noise < 0.8 {
            findings.push(Finding::new(
                "noise",
                "unnaturally_low_noise",
                format!(
                    "Extremely low sensor noise (median σ={median_noise:.3} via wavelet estimation) - \
                     image may be AI-generated or heavily denoised"
                ),
                Severity::Medium,
                0.6,
            ));
        }

        let mean_noise: f64 = block_noise.iter().sum::<f64>() / total_blocks;
        let var_noise: f64 = block_noise
            .iter()
            .map(|&n| (n - mean_noise).powi(2))
            .sum::<f64>()
            / total_blocks;
        let std_noise = var_noise.sqrt();
        if std_noise > 0.0 {
            let kurtosis: f64 = block_noise
                .iter()
                .map(|&n| ((n - mean_noise) / std_noise).powi(4))
                .sum::<f64>()
                / total_blocks
                - 3.0;
            if kurtosis > 3.0 {
                findings.push(Finding::new(
                    "noise",
                    "noise_distribution_anomaly",
                    format!(
                        "Noise level distribution has excess kurtosis {kurtosis:.2} (expected ~0 for \
                         genuine photos) - heavy tails suggest mixed content from multiple sources"
                    ),
                    Severity::Medium,
                    0.55,
                ));
            }
        }

        findings
    }
}
