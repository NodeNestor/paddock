//! Extract embedded images from a PDF and run the pixel forensics (ELA on the
//! JPEG stream, noise on the decoded pixels) over each - a claimant's scanned
//! receipts/photos inside a PDF deserve the same scrutiny as standalone images.
//! Ported from the reference implementation.
//!
//! This is the one PDF analyzer with a GPU path: it reuses ELA/noise via
//! `run_analyzer`, so it inherits their GPU-first execution and CPU fallback.

use std::io::Cursor;

use image::{DynamicImage, GenericImageView, ImageReader};

use crate::analyzer::{Analyzer, run_analyzer};
use crate::pixel::ela::ErrorLevelAnalyzer;
use crate::pixel::noise::NoiseAnalyzer;
use crate::{Context, Finding, Severity};

/// Cap on embedded images analyzed per PDF (large PDFs can carry hundreds).
const MAX_IMAGES: usize = 20;

pub struct PdfImagePipeline;

impl Analyzer for PdfImagePipeline {
    fn name(&self) -> &'static str {
        "pdf_images"
    }

    fn applies_to(&self, ctx: &Context) -> bool {
        ctx.is_pdf()
    }

    fn cpu(&self, ctx: &Context) -> Vec<Finding> {
        self.run(ctx, None)
    }

    #[cfg(feature = "cuda")]
    fn gpu(
        &self,
        gpu: &crate::gpu::ForensicGpu,
        ctx: &Context,
    ) -> Result<Vec<Finding>, crate::gpu::GpuError> {
        Ok(self.run(ctx, Some(gpu)))
    }
}

impl PdfImagePipeline {
    /// `gpu = Some` runs the per-image ELA/noise on the GPU (CPU fallback per
    /// analyzer); `None` runs them on the CPU.
    fn run(&self, ctx: &Context, gpu: Option<&crate::gpu::ForensicGpu>) -> Vec<Finding> {
        let doc = match sift::read(&ctx.raw_bytes) {
            Ok(d) => d,
            Err(_) => return Vec::new(),
        };
        let images = match doc.images() {
            Ok(i) => i,
            Err(_) => return Vec::new(),
        };
        if images.is_empty() {
            return Vec::new();
        }

        let mut findings = vec![Finding::new(
            "pdf_images",
            "pdf_image_inventory",
            format!(
                "PDF contains {} embedded image(s) across pages",
                images.len()
            ),
            Severity::Info,
            1.0,
        )];
        if images.len() > MAX_IMAGES {
            findings.push(Finding::new(
                "pdf_images",
                "pdf_image_limit",
                format!(
                    "PDF contains {} images - analyzing first {MAX_IMAGES} only",
                    images.len()
                ),
                Severity::Info,
                1.0,
            ));
        }

        for (idx, image) in images.iter().take(MAX_IMAGES).enumerate() {
            findings.extend(self.analyze_embedded(image, idx, gpu));
        }
        findings
    }

    fn analyze_embedded(
        &self,
        image: &sift::Image,
        index: usize,
        gpu: Option<&crate::gpu::ForensicGpu>,
    ) -> Vec<Finding> {
        // Decode for the noise pass (and to gate on size).
        let decoded = match &image.data {
            sift::ImageData::Jpeg(b) => decode_bytes(b),
            sift::ImageData::Jpeg2000(b) => decode_bytes(b),
            sift::ImageData::Pixels(b) => {
                decode_raw(b, image.width, image.height, image.components)
            }
            _ => None,
        };
        let Some(decoded) = decoded else {
            return Vec::new();
        };
        if image.width < 100 || image.height < 100 {
            return Vec::new();
        }

        let mut findings = Vec::new();
        let tag = |f: &mut Finding| {
            f.description = format!(
                "[PDF image {index} (page {}, {}x{})] {}",
                image.page + 1,
                image.width,
                image.height,
                f.description
            );
        };

        // ELA on the original JPEG stream (keeps its quantization tables).
        if let sift::ImageData::Jpeg(jpeg) = &image.data
            && let Ok(sub) = Context::from_bytes(jpeg.clone())
        {
            let ela = ErrorLevelAnalyzer::default();
            let mut fs = run_analyzer(&ela, &sub, gpu);
            fs.iter_mut().for_each(&tag);
            findings.extend(fs);
        }

        // Noise on the decoded pixels (re-encode to PNG for a byte-exact,
        // lossless sub-context - no JPEG round-trip that would perturb noise).
        if let Some(png) = encode_png(&decoded)
            && let Ok(sub) = Context::from_bytes(png)
        {
            let noise = NoiseAnalyzer::default();
            let mut fs = run_analyzer(&noise, &sub, gpu);
            fs.iter_mut().for_each(&tag);
            findings.extend(fs);
        }

        findings
    }
}

fn decode_bytes(bytes: &[u8]) -> Option<DynamicImage> {
    ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .ok()?
        .decode()
        .ok()
}

fn decode_raw(bytes: &[u8], width: u32, height: u32, components: u8) -> Option<DynamicImage> {
    match components {
        1 => {
            image::GrayImage::from_raw(width, height, bytes.to_vec()).map(DynamicImage::ImageLuma8)
        }
        3 => image::RgbImage::from_raw(width, height, bytes.to_vec()).map(DynamicImage::ImageRgb8),
        _ => None,
    }
}

fn encode_png(img: &DynamicImage) -> Option<Vec<u8>> {
    let _ = img.dimensions();
    let mut buf = Cursor::new(Vec::new());
    img.write_to(&mut buf, image::ImageFormat::Png).ok()?;
    Some(buf.into_inner())
}
