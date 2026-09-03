//! Cross-validates EXIF claims against actual pixels (resolution, orientation,
//! color space, bit depth), ported verbatim from the reference. CPU-only.

use crate::analyzer::Analyzer;
use crate::{Context, Finding, Severity};

pub struct ExifPixelAnalyzer;

impl Analyzer for ExifPixelAnalyzer {
    fn name(&self) -> &'static str {
        "exif_pixel"
    }

    fn cpu(&self, ctx: &Context) -> Vec<Finding> {
        let mut findings = Vec::new();
        Self::check_resolution_consistency(ctx, &mut findings);
        Self::check_orientation_consistency(ctx, &mut findings);
        Self::check_color_space_consistency(ctx, &mut findings);
        Self::check_bits_per_sample(ctx, &mut findings);
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

impl ExifPixelAnalyzer {
    fn check_resolution_consistency(ctx: &Context, findings: &mut Vec<Finding>) {
        let exif_width = Self::get_tag_u32(&ctx.tags, "ImageWidth")
            .or_else(|| Self::get_tag_u32(&ctx.tags, "ExifImageWidth"));
        let exif_height = Self::get_tag_u32(&ctx.tags, "ImageHeight")
            .or_else(|| Self::get_tag_u32(&ctx.tags, "ExifImageHeight"));

        if let Some(ew) = exif_width
            && ew != ctx.width
            && ew != ctx.height
        {
            findings.push(Finding::new(
                "exif_pixel",
                "resolution_mismatch_width",
                format!(
                    "EXIF width ({}) does not match actual width ({}) or height ({})",
                    ew, ctx.width, ctx.height
                ),
                Severity::High,
                0.85,
            ));
        }

        if let Some(eh) = exif_height
            && eh != ctx.height
            && eh != ctx.width
        {
            findings.push(Finding::new(
                "exif_pixel",
                "resolution_mismatch_height",
                format!(
                    "EXIF height ({}) does not match actual height ({}) or width ({})",
                    eh, ctx.height, ctx.width
                ),
                Severity::High,
                0.85,
            ));
        }
    }

    fn check_orientation_consistency(ctx: &Context, findings: &mut Vec<Finding>) {
        let orientation = Self::get_tag_u32(&ctx.tags, "Orientation");

        if let Some(orient) = orientation {
            let is_rotated = matches!(orient, 5..=8);
            let pixel_landscape = ctx.width > ctx.height;

            let exif_width = Self::get_tag_u32(&ctx.tags, "ImageWidth");
            let exif_height = Self::get_tag_u32(&ctx.tags, "ImageHeight");

            if let (Some(ew), Some(eh)) = (exif_width, exif_height) {
                let exif_landscape = ew > eh;

                if is_rotated && exif_landscape == pixel_landscape && ew == ctx.width {
                    findings.push(Finding::new(
                        "exif_pixel",
                        "orientation_inconsistency",
                        format!(
                            "EXIF orientation tag ({orient}) indicates rotation but dimensions are \
                             already in display orientation - metadata may have been manipulated"
                        ),
                        Severity::Medium,
                        0.6,
                    ));
                }
            }
        }
    }

    fn check_color_space_consistency(ctx: &Context, findings: &mut Vec<Finding>) {
        let color_space = ctx
            .tags
            .iter()
            .find(|t| t.name == "ColorSpace")
            .map(|t| t.value.as_str());

        if let Some(cs) = color_space
            && (cs.contains("sRGB") || cs == "1")
        {
            let rgb = ctx.image.to_rgb8();
            let pixels = rgb.as_raw();
            let total = pixels.len() as f64;

            let clipped = pixels.iter().filter(|&&v| v == 0 || v == 255).count() as f64;
            let clip_ratio = clipped / total;

            if clip_ratio > 0.15 {
                findings.push(Finding::new(
                    "exif_pixel",
                    "color_space_clipping",
                    format!(
                        "Image claims sRGB but {:.1}% of channel values are clipped - \
                             possible color space mismatch or manipulation",
                        clip_ratio * 100.0
                    ),
                    Severity::Low,
                    0.45,
                ));
            }
        }
    }

    fn check_bits_per_sample(ctx: &Context, findings: &mut Vec<Finding>) {
        let bps = ctx
            .tags
            .iter()
            .find(|t| t.name == "BitsPerSample")
            .map(|t| t.value.clone());

        if let Some(bps_str) = bps
            && bps_str.contains("16")
        {
            let gray = ctx.gray();
            let unique_values: std::collections::HashSet<&u8> = gray.iter().collect();

            if unique_values.len() < 64 {
                findings.push(Finding::new(
                    "exif_pixel",
                    "bit_depth_inconsistency",
                    format!(
                        "EXIF claims {} bits per sample but only {} unique intensity values \
                             found - metadata may have been edited",
                        bps_str,
                        unique_values.len()
                    ),
                    Severity::Medium,
                    0.55,
                ));
            }
        }
    }

    fn get_tag_u32(tags: &[sift::Tag], name: &str) -> Option<u32> {
        tags.iter()
            .find(|t| t.name == name)
            .and_then(|t| t.value.trim().parse::<u32>().ok())
    }
}
