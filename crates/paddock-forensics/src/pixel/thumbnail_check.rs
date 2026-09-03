//! EXIF-thumbnail consistency check, ported verbatim from the CPU
//! reference. CPU-only: it is a decode + MSE/SSIM compare against the embedded
//! thumbnail - there is no GPU kernel in the reference and no sensible CUDA
//! alternative, so `gpu()` delegates to `cpu()` (the double_jpeg policy).
//!
//! Cameras write the EXIF thumbnail at capture time. If the main image is edited
//! but the thumbnail is not regenerated, the two diverge - a common forgery
//! oversight. Never skipped by content type (a file with no thumbnail simply
//! returns nothing); a PDF carries no thumbnail bytes, so it no-ops there too.

use image::{DynamicImage, GenericImageView, ImageReader};

use crate::analyzer::Analyzer;
use crate::{Context, Finding, Severity};

pub struct ThumbnailChecker;

impl Analyzer for ThumbnailChecker {
    fn name(&self) -> &'static str {
        "thumbnail_check"
    }

    fn cpu(&self, ctx: &Context) -> Vec<Finding> {
        let thumb_bytes = match &ctx.thumbnail_bytes {
            Some(bytes) if !bytes.is_empty() => bytes,
            _ => return vec![],
        };

        let thumb_image = match ImageReader::new(std::io::Cursor::new(thumb_bytes))
            .with_guessed_format()
            .ok()
            .and_then(|r| r.decode().ok())
        {
            Some(img) => img,
            None => return vec![],
        };

        let mut findings = Vec::new();

        // Resize the main image to the thumbnail's dimensions for comparison.
        let (tw, th) = thumb_image.dimensions();
        if tw == 0 || th == 0 || tw > 1024 || th > 1024 {
            return findings;
        }

        let resized_main = ctx
            .image
            .resize_exact(tw, th, image::imageops::FilterType::Lanczos3);

        let mse = Self::compute_mse(&resized_main, &thumb_image);
        let ssim = Self::compute_ssim(&resized_main, &thumb_image);

        // High MSE or low SSIM means the thumbnail no longer matches the image.
        if mse > 500.0 || ssim < 0.85 {
            findings.push(Finding::new(
                "thumbnail_check",
                "thumbnail_mismatch",
                format!(
                    "EXIF thumbnail does not match main image (MSE={mse:.1}, SSIM={ssim:.3}) - \
                     main image was likely edited after capture without updating the thumbnail"
                ),
                Severity::High,
                0.85,
            ));
        } else if mse > 200.0 || ssim < 0.92 {
            findings.push(Finding::new(
                "thumbnail_check",
                "thumbnail_minor_mismatch",
                format!(
                    "EXIF thumbnail shows minor differences from main image \
                     (MSE={mse:.1}, SSIM={ssim:.3}) - possible post-processing"
                ),
                Severity::Medium,
                0.65,
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

impl ThumbnailChecker {
    fn compute_mse(a: &DynamicImage, b: &DynamicImage) -> f64 {
        let (w, h) = a.dimensions();
        let (bw, bh) = b.dimensions();
        if w != bw || h != bh || w == 0 || h == 0 {
            return f64::MAX;
        }

        let mut sum = 0.0_f64;
        let count = (w as f64) * (h as f64) * 3.0;

        for y in 0..h {
            for x in 0..w {
                let pa = a.get_pixel(x, y).0;
                let pb = b.get_pixel(x, y).0;
                for c in 0..3 {
                    let diff = pa[c] as f64 - pb[c] as f64;
                    sum += diff * diff;
                }
            }
        }

        sum / count
    }

    /// Simplified SSIM (global, single-window) - matches the reference.
    fn compute_ssim(a: &DynamicImage, b: &DynamicImage) -> f64 {
        let (w, h) = a.dimensions();
        let (bw, bh) = b.dimensions();
        if w != bw || h != bh || w == 0 || h == 0 {
            return 0.0;
        }

        let ga = a.to_luma8();
        let gb = b.to_luma8();
        let pa = ga.as_raw();
        let pb = gb.as_raw();
        let n = pa.len() as f64;

        let mean_a: f64 = pa.iter().map(|&v| v as f64).sum::<f64>() / n;
        let mean_b: f64 = pb.iter().map(|&v| v as f64).sum::<f64>() / n;

        let mut var_a = 0.0_f64;
        let mut var_b = 0.0_f64;
        let mut cov = 0.0_f64;

        for i in 0..pa.len() {
            let da = pa[i] as f64 - mean_a;
            let db = pb[i] as f64 - mean_b;
            var_a += da * da;
            var_b += db * db;
            cov += da * db;
        }

        var_a /= n;
        var_b /= n;
        cov /= n;

        // SSIM constants for 8-bit images.
        let c1 = 6.5025; // (0.01 * 255)^2
        let c2 = 58.5225; // (0.03 * 255)^2

        let numerator = (2.0 * mean_a * mean_b + c1) * (2.0 * cov + c2);
        let denominator = (mean_a * mean_a + mean_b * mean_b + c1) * (var_a + var_b + c2);

        if denominator > 0.0 {
            (numerator / denominator).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }
}
