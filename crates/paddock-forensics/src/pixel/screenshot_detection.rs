//! Screenshot / photo-of-photo detection, ported verbatim from the CPU
//! reference. CPU-only (color-quantization counting, gradient stats,
//! autocorrelation periodicity; no GPU kernel), `gpu()` delegates.
//!
//! Detects images that are screenshots or photos of a screen/print - a common
//! way to reuse an image in fraud. Three signals: reduced color depth (6-bit
//! display output), an unnaturally perfect pixel grid (no optical blur), and
//! moiré periodicity from a display sub-pixel grid. Photo-oriented: skipped for
//! PDFs.

use crate::analyzer::Analyzer;
use crate::{Context, Finding, Severity};

pub struct ScreenshotDetector {
    /// Minimum image dimension for analysis.
    min_dimension: usize,
}

impl Default for ScreenshotDetector {
    fn default() -> Self {
        Self { min_dimension: 200 }
    }
}

impl Analyzer for ScreenshotDetector {
    fn name(&self) -> &'static str {
        "screenshot_detection"
    }

    fn applies_to(&self, ctx: &Context) -> bool {
        !ctx.is_pdf()
    }

    fn cpu(&self, ctx: &Context) -> Vec<Finding> {
        let (width, height) = (ctx.width as usize, ctx.height as usize);
        if width < self.min_dimension || height < self.min_dimension {
            return Vec::new();
        }

        let gray = ctx.gray();
        let mut findings = Vec::new();

        self.detect_color_quantization(ctx, &mut findings);
        self.detect_pixel_grid(gray, width, height, &mut findings);
        self.detect_moire(gray, width, height, &mut findings);

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

impl ScreenshotDetector {
    /// Reduced color depth (6-bit display output / posterization).
    fn detect_color_quantization(&self, ctx: &Context, findings: &mut Vec<Finding>) {
        let rgb = ctx.image.to_rgb8();
        let pixels = rgb.as_raw();

        let mut r_used = [false; 256];
        let mut g_used = [false; 256];
        let mut b_used = [false; 256];

        let step = (pixels.len() / 3 / 50000).max(1);
        for i in (0..pixels.len() / 3).step_by(step) {
            r_used[pixels[i * 3] as usize] = true;
            g_used[pixels[i * 3 + 1] as usize] = true;
            b_used[pixels[i * 3 + 2] as usize] = true;
        }

        let r_unique = r_used.iter().filter(|&&v| v).count();
        let g_unique = g_used.iter().filter(|&&v| v).count();
        let b_unique = b_used.iter().filter(|&&v| v).count();

        // 6-bit quantization: values cluster at multiples of 4, gaps between.
        let check_quantized = |used: &[bool; 256]| -> bool {
            let at_multiples: usize = (0..64).filter(|&i| used[i * 4]).count();
            let between: usize = (0..256).filter(|&i| i % 4 != 0 && used[i]).count();
            at_multiples > 40 && between < at_multiples / 3
        };

        let channels_quantized = [
            check_quantized(&r_used),
            check_quantized(&g_used),
            check_quantized(&b_used),
        ]
        .iter()
        .filter(|&&v| v)
        .count();

        if channels_quantized >= 2 {
            findings.push(Finding::new(
                "screenshot_detection",
                "screenshot_color_quantization",
                format!(
                    "{channels_quantized} color channels show 6-bit quantization pattern \
                     (R:{r_unique}, G:{g_unique}, B:{b_unique} unique values) - \
                     image appears to be a screenshot or display capture"
                ),
                Severity::Medium,
                0.70,
            ));
        } else if r_unique < 100 && g_unique < 100 && b_unique < 100 {
            findings.push(Finding::new(
                "screenshot_detection",
                "screenshot_low_color_depth",
                format!(
                    "Very low color diversity (R:{r_unique}, G:{g_unique}, B:{b_unique} \
                     unique values) - possible screenshot, synthetic image, or heavy \
                     posterization"
                ),
                Severity::Low,
                0.50,
            ));
        }
    }

    /// Unnaturally perfect pixel grid (no lens blur).
    fn detect_pixel_grid(
        &self,
        gray: &[u8],
        width: usize,
        height: usize,
        findings: &mut Vec<Finding>,
    ) {
        let mut grad_sum = 0.0_f64;
        let mut grad_max = 0.0_f64;
        let mut count = 0_u64;

        let step = ((width * height) / 100000).max(1);
        let mut idx = 0;

        for y in 1..height - 1 {
            for x in 1..width - 1 {
                idx += 1;
                if idx % step != 0 {
                    continue;
                }

                let dx = (gray[y * width + x + 1] as f64 - gray[y * width + x] as f64).abs();
                let dy = (gray[(y + 1) * width + x] as f64 - gray[y * width + x] as f64).abs();
                let grad = dx.max(dy);

                grad_sum += grad;
                grad_max = grad_max.max(grad);
                count += 1;
            }
        }

        if count < 100 {
            return;
        }

        let grad_mean = grad_sum / count as f64;

        // Sharp boundaries + large flat areas + low mean gradient = screenshot.
        if grad_mean < 5.0 && grad_max > 100.0 {
            let ratio = grad_max / grad_mean.max(0.1);
            if ratio > 30.0 {
                findings.push(Finding::new(
                    "screenshot_detection",
                    "screenshot_pixel_grid",
                    format!(
                        "Pixel gradient distribution consistent with screenshot \
                         (mean {grad_mean:.1}, max {grad_max:.0}, ratio {ratio:.0}) - sharp \
                         boundaries with flat regions, no optical blur"
                    ),
                    Severity::Low,
                    0.45,
                ));
            }
        }
    }

    /// Moiré periodicity via 1D scan-line autocorrelation.
    fn detect_moire(&self, gray: &[u8], width: usize, height: usize, findings: &mut Vec<Finding>) {
        let mid_y = height / 2;
        let mid_x = width / 2;

        let h_periodic = self.check_periodicity(gray, width, mid_y, true, width);
        let v_periodic = self.check_periodicity(gray, width, mid_x, false, height);

        if h_periodic || v_periodic {
            let direction = if h_periodic && v_periodic {
                "horizontal and vertical"
            } else if h_periodic {
                "horizontal"
            } else {
                "vertical"
            };

            findings.push(Finding::new(
                "screenshot_detection",
                "screenshot_moire_pattern",
                format!(
                    "Periodic pattern detected in {direction} direction - \
                     possible moiré from photographing a screen or printed image"
                ),
                Severity::Medium,
                0.55,
            ));
        }
    }

    fn check_periodicity(
        &self,
        gray: &[u8],
        stride: usize,
        fixed_coord: usize,
        horizontal: bool,
        length: usize,
    ) -> bool {
        if length < 64 {
            return false;
        }

        let line: Vec<f64> = (0..length)
            .map(|i| {
                let (x, y) = if horizontal {
                    (i, fixed_coord)
                } else {
                    (fixed_coord, i)
                };
                gray[y * stride + x] as f64
            })
            .collect();

        let n = line.len();
        let mean: f64 = line.iter().sum::<f64>() / n as f64;
        let var: f64 = line.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n as f64;

        if var < 1.0 {
            return false;
        }

        let mut peak_count = 0;
        let mut prev_acf = 1.0_f64;

        for lag in 2..32.min(n / 4) {
            let acf: f64 = (0..n - lag)
                .map(|i| (line[i] - mean) * (line[i + lag] - mean))
                .sum::<f64>()
                / (n as f64 * var);

            if acf > 0.3 && acf > prev_acf {
                peak_count += 1;
            }
            prev_acf = acf;
        }

        peak_count >= 2 // Multiple periodic peaks = moiré.
    }
}
