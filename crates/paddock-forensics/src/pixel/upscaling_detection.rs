//! Upscaling / super-resolution detection, ported verbatim from the CPU
//! reference. CPU-only (autocorrelation + Haar high-frequency energy; no GPU
//! kernel), `gpu()` delegates.
//!
//! Detects an image upscaled from a lower resolution - a low-res web image
//! passed off as an original photo. Two signals: (1) integer-factor
//! interpolation periodicity in the second derivative's autocorrelation, and
//! (2) suppressed high-frequency energy relative to the stated resolution.
//! Photo-oriented: skipped for PDFs.

use crate::analyzer::Analyzer;
use crate::{Context, Finding, Severity};

pub struct UpscalingDetector {
    /// Block size for the high-frequency energy pass.
    block_size: usize,
}

impl Default for UpscalingDetector {
    fn default() -> Self {
        Self { block_size: 64 }
    }
}

impl Analyzer for UpscalingDetector {
    fn name(&self) -> &'static str {
        "upscaling_detection"
    }

    fn applies_to(&self, ctx: &Context) -> bool {
        !ctx.is_pdf()
    }

    fn cpu(&self, ctx: &Context) -> Vec<Finding> {
        let (width, height) = (ctx.width as usize, ctx.height as usize);
        if width < 200 || height < 200 {
            return Vec::new();
        }

        let gray = ctx.gray();
        let mut findings = Vec::new();

        self.detect_interpolation_periodicity(gray, width, height, &mut findings);
        self.detect_hf_suppression(gray, width, height, &mut findings);

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

impl UpscalingDetector {
    /// Periodicity in the second derivative at integer factors (2×/3×/4×).
    fn detect_interpolation_periodicity(
        &self,
        gray: &[u8],
        width: usize,
        height: usize,
        findings: &mut Vec<Finding>,
    ) {
        let mid_y = height / 2;
        let sample_length = width.min(1024);

        // Second derivative: d2[x] = gray[x+1] - 2*gray[x] + gray[x-1].
        let d2: Vec<f64> = (1..sample_length - 1)
            .map(|x| {
                let idx = mid_y * width + x;
                gray[idx + 1] as f64 - 2.0 * gray[idx] as f64 + gray[idx - 1] as f64
            })
            .collect();

        let n = d2.len();
        if n < 32 {
            return;
        }

        let mean: f64 = d2.iter().sum::<f64>() / n as f64;
        let var: f64 = d2.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n as f64;
        if var < 0.1 {
            return;
        }

        let mut detected_factor = 0;
        let mut max_acf = 0.0_f64;

        for factor in 2..=4 {
            let acf: f64 = (0..n - factor)
                .map(|i| (d2[i] - mean) * (d2[i + factor] - mean))
                .sum::<f64>()
                / (n as f64 * var);

            if acf > 0.4 && acf > max_acf {
                max_acf = acf;
                detected_factor = factor;
            }
        }

        if detected_factor > 0 {
            findings.push(Finding::new(
                "upscaling_detection",
                "upscaling_interpolation_periodicity",
                format!(
                    "Second derivative shows {detected_factor}× periodicity \
                     (autocorrelation {max_acf:.2}) - image appears to be upscaled \
                     by factor {detected_factor} from lower resolution"
                ),
                Severity::Medium,
                (0.45 + max_acf * 0.4).min(0.80),
            ));
        }
    }

    /// Suppressed high-frequency content relative to the resolution.
    fn detect_hf_suppression(
        &self,
        gray: &[u8],
        width: usize,
        height: usize,
        findings: &mut Vec<Finding>,
    ) {
        let bs = self.block_size;
        let blocks_x = width / bs;
        let blocks_y = height / bs;

        if blocks_x < 3 || blocks_y < 3 {
            return;
        }

        // Per-block Haar decomposition -> high-frequency energy ratio.
        let mut hf_ratios: Vec<f64> = Vec::new();

        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                let x0 = bx * bs;
                let y0 = by * bs;
                let half = bs / 2;

                let mut ll_energy = 0.0_f64;
                let mut hf_energy = 0.0_f64;

                for py in 0..half {
                    for px in 0..half {
                        let x = x0 + px * 2;
                        let y = y0 + py * 2;
                        if x + 1 >= width || y + 1 >= height {
                            continue;
                        }

                        let a = gray[y * width + x] as f64;
                        let b = gray[y * width + x + 1] as f64;
                        let c = gray[(y + 1) * width + x] as f64;
                        let d = gray[(y + 1) * width + x + 1] as f64;

                        let ll = (a + b + c + d) * 0.25;
                        let lh = (a + b - c - d) * 0.25;
                        let hl = (a - b + c - d) * 0.25;
                        let hh = (a - b - c + d) * 0.25;

                        ll_energy += ll * ll;
                        hf_energy += lh * lh + hl * hl + hh * hh;
                    }
                }

                let total = ll_energy + hf_energy;
                if total > 0.0 {
                    hf_ratios.push(hf_energy / total);
                }
            }
        }

        if hf_ratios.is_empty() {
            return;
        }

        let mean_hf: f64 = hf_ratios.iter().sum::<f64>() / hf_ratios.len() as f64;

        // Natural high-res images have HF ratio > ~0.05; upscaled << 0.02.
        if mean_hf < 0.015 {
            findings.push(Finding::new(
                "upscaling_detection",
                "upscaling_hf_suppression",
                format!(
                    "Image has very low high-frequency energy ratio ({mean_hf:.4}) for its \
                     resolution ({width}×{height}) - consistent with upscaling from \
                     lower resolution or AI super-resolution"
                ),
                Severity::Medium,
                (0.50 + (0.015 - mean_hf) * 20.0).min(0.75),
            ));
        }
    }
}
