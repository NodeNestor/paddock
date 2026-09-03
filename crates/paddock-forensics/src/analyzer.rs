//! The [`Analyzer`] trait, the GPU-first dispatch policy, and the top-level
//! [`run`] entry point.

use crate::gpu::ForensicGpu;
use crate::{Context, Finding, Severity};

/// A forensic analyzer. Implementors provide the *same* canonical algorithm on
/// both paths; the parity test enforces that `gpu` and `cpu` agree.
pub trait Analyzer: Send + Sync {
    /// Stable short name, also stamped into each [`Finding::analyzer`].
    fn name(&self) -> &'static str;

    /// Whether this analyzer is applicable to the given context (content-type
    /// gating). Default: always applicable.
    fn applies_to(&self, ctx: &Context) -> bool {
        let _ = ctx;
        true
    }

    /// CPU reference implementation - always available, the parity oracle.
    fn cpu(&self, ctx: &Context) -> Vec<Finding>;

    /// GPU implementation of the *same* algorithm. Returns `Err` on any GPU
    /// failure; the dispatcher then falls back to [`Analyzer::cpu`].
    #[cfg(feature = "cuda")]
    fn gpu(&self, gpu: &ForensicGpu, ctx: &Context) -> Result<Vec<Finding>, crate::gpu::GpuError>;
}

/// Run one analyzer with the GPU-first, CPU-fallback policy:
/// - not applicable -> no findings;
/// - GPU present and the `cuda` feature on -> try GPU, on error log and fall back;
/// - otherwise -> CPU.
pub fn run_analyzer(
    analyzer: &dyn Analyzer,
    ctx: &Context,
    gpu: Option<&ForensicGpu>,
) -> Vec<Finding> {
    if !analyzer.applies_to(ctx) {
        return Vec::new();
    }

    let name = analyzer.name();

    // Panic isolation (parity with the reference pipeline): one broken analyzer must
    // not abort the whole forensic run (or the serving request it rides on). A
    // panicking analyzer is logged and contributes no findings; the rest run.
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        #[cfg(feature = "cuda")]
        if let Some(gpu) = gpu {
            match analyzer.gpu(gpu, ctx) {
                Ok(findings) => return findings,
                Err(e) => {
                    tracing::warn!(
                        analyzer = name,
                        error = %e,
                        "forensic GPU path failed; falling back to CPU"
                    );
                }
            }
        }
        // Silence the unused warning when compiled without the `cuda` feature.
        let _ = gpu;

        analyzer.cpu(ctx)
    }));

    match result {
        Ok(findings) => findings,
        Err(_) => {
            tracing::error!(analyzer = name, "forensic analyzer panicked; skipping it");
            Vec::new()
        }
    }
}

/// The default analyzer set. Grows one wave at a time. Wave 0: ELA. Wave 1:
/// noise + jpeg_ghost (JPEG-domain pixel forensics) + the PDF document suite
/// (structure / overlay / embedded-image pipeline). Content-type gating
/// (`applies_to`) means pixel analyzers skip PDFs and PDF analyzers skip images,
/// so the whole set can be offered for any attachment.
pub fn default_analyzers() -> Vec<Box<dyn Analyzer>> {
    vec![
        // Wave 5: metadata (sift tags / raw bytes) - the reference's stage 1, CPU-only.
        Box::new(crate::metadata::analyzer::MetadataAnalyzer),
        Box::new(crate::metadata::exif_pixel::ExifPixelAnalyzer),
        Box::new(crate::metadata::c2pa::C2paChecker),
        Box::new(crate::pixel::ela::ErrorLevelAnalyzer::default()),
        Box::new(crate::pixel::noise::NoiseAnalyzer::default()),
        Box::new(crate::pixel::jpeg_ghost::JpegGhostDetector::default()),
        Box::new(crate::pixel::double_jpeg::DoubleJpegDetector),
        // Wave 2a: CPU pixel analyzers (verbatim reference ports).
        Box::new(crate::pixel::thumbnail_check::ThumbnailChecker),
        Box::new(crate::pixel::qtable_fingerprint::QtableFingerprintAnalyzer),
        Box::new(crate::pixel::histogram_analysis::HistogramGapAnalyzer::default()),
        Box::new(crate::pixel::resampling::ResamplingDetector::default()),
        Box::new(crate::pixel::upscaling_detection::UpscalingDetector::default()),
        Box::new(crate::pixel::screenshot_detection::ScreenshotDetector::default()),
        Box::new(crate::pixel::text_alignment::TextAlignmentAnalyzer::default()),
        // Wave 2b: CPU pixel analyzers (verbatim reference ports).
        Box::new(crate::pixel::font_consistency::FontConsistencyAnalyzer::default()),
        Box::new(crate::pixel::dof_consistency::DofConsistencyAnalyzer::default()),
        Box::new(crate::pixel::prnu_cross_region::PrnuCrossRegionAnalyzer::default()),
        Box::new(crate::pixel::jpeg_forensics::JpegForensicsAnalyzer::default()),
        Box::new(crate::pixel::vanishing_point::VanishingPointAnalyzer::default()),
        Box::new(crate::pixel::paste_rectangle::PasteRectangleDetector::default()),
        Box::new(crate::pixel::document_forensics::DocumentForensicsAnalyzer::default()),
        // Wave 3a: GPU pixel analyzers (canonical CPU + exact-parity CUDA kernel).
        Box::new(crate::pixel::edge_sharpness::EdgeSharpnessAnalyzer::default()),
        Box::new(crate::pixel::channel_correlation::ChannelCorrelationAnalyzer::default()),
        Box::new(crate::pixel::wavelet_consistency::WaveletConsistencyAnalyzer::default()),
        Box::new(crate::pixel::texture::TextureConsistencyAnalyzer::default()),
        Box::new(crate::pixel::color_consistency::ColorConsistencyAnalyzer::default()),
        Box::new(crate::pixel::anti_forensics::AntiForensicsDetector::default()),
        // Wave 3b: lighting has an exact-parity kernel; cfa/copy_move/
        // splice_boundary/shadow/geometric are CPU-only (the reference's GPU path is a stub, or
        // the canonical algorithm branches on a transcendental so no bit-exact
        // GPU is feasible - documented per-file).
        Box::new(crate::pixel::lighting_consistency::LightingConsistencyAnalyzer::default()),
        Box::new(crate::pixel::shadow_consistency::ShadowConsistencyAnalyzer::default()),
        Box::new(crate::pixel::geometric::GeometricConsistencyAnalyzer::default()),
        Box::new(crate::pixel::cfa::CfaAnalyzer::default()),
        Box::new(crate::pixel::copy_move::CopyMoveDetector::default()),
        Box::new(crate::pixel::splice_boundary::SpliceBoundaryDetector::default()),
        // Wave 4: ai_detect (all CPU-only - reference GPU stubs / rustfft, no
        // bit-exact GPU alternative). chromatic uses a fixed-seed LCG RANSAC
        // (deterministic).
        Box::new(crate::pixel::frequency::FrequencyAnalyzer),
        Box::new(crate::pixel::chromatic::ChromaticAberrationAnalyzer::default()),
        Box::new(crate::pixel::illumination::IlluminationAnalyzer::default()),
        Box::new(crate::pixel::prnu::PrnuAnalyzer::default()),
        Box::new(crate::pdf::structure::PdfStructureAnalyzer),
        Box::new(crate::pdf::overlay::OverlayDetector),
        Box::new(crate::pdf::image_pipeline::PdfImagePipeline),
    ]
}

/// The full forensic report for one image.
#[derive(Debug, Clone)]
pub struct Report {
    pub findings: Vec<Finding>,
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Boom;
    impl Analyzer for Boom {
        fn name(&self) -> &'static str {
            "boom"
        }
        fn cpu(&self, _ctx: &Context) -> Vec<Finding> {
            panic!("analyzer blew up");
        }
        #[cfg(feature = "cuda")]
        fn gpu(
            &self,
            _gpu: &ForensicGpu,
            ctx: &Context,
        ) -> Result<Vec<Finding>, crate::gpu::GpuError> {
            Ok(self.cpu(ctx))
        }
    }

    #[test]
    fn analyzer_panic_is_isolated() {
        // Any decodable ctx; the analyzer panics regardless.
        let ctx = Context::from_bytes(b"%PDF-1.4\ntrailer<<>>\n%%EOF".to_vec()).unwrap();
        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {})); // keep test output quiet
        let out = run_analyzer(&Boom, &ctx, None);
        std::panic::set_hook(prev);
        assert!(
            out.is_empty(),
            "a panicking analyzer must yield no findings, not abort"
        );
    }
}

impl Report {
    /// Score + dedup these findings into a [`crate::risk::RiskReport`]
    /// (risk_score, verdict, deduplicated key findings). See [`crate::risk`].
    pub fn risk(&self) -> crate::risk::RiskReport {
        crate::risk::score(&self.findings)
    }
}

/// Minimum megapixels for reliable document text-tampering detection. Below
/// this, splice boundaries / font anomalies / baseline evidence get missed or
/// imprecise, so we emit a `low_resolution_document` warning up front. Fixed
/// (no customer env knob - paddock ships elected settings; the reference made
/// this configurable, with the same default).
const DOCUMENT_QUALITY_MIN_MP: f64 = 1.0;

/// Run the default analyzer set over `ctx`, GPU-first when `gpu` is supplied.
pub fn run(ctx: &Context, gpu: Option<&ForensicGpu>) -> Report {
    let mut findings = Vec::new();

    // Pre-analysis check (parity with the reference pipeline): warn on low-resolution
    // document rasters. Gated `!is_pdf` - a paddock PDF context is a 1×1
    // placeholder, so the MP check only makes sense on real document images.
    if matches!(ctx.content_type, crate::context::ContentType::Document) && !ctx.is_pdf() {
        let mp = (ctx.width as u64 * ctx.height as u64) as f64 / 1_000_000.0;
        if mp < DOCUMENT_QUALITY_MIN_MP {
            findings.push(Finding::new(
                "metadata",
                "low_resolution_document",
                format!(
                    "Document is {}x{} ({:.2} MP), below the recommended {} MP for reliable \
                     text-tampering detection. Splice boundaries, font anomalies, and \
                     baseline evidence may be missed or imprecise. Submit a higher-resolution \
                     capture for best results.",
                    ctx.width, ctx.height, mp, DOCUMENT_QUALITY_MIN_MP
                ),
                Severity::Medium,
                0.95,
            ));
        }
    }

    for analyzer in default_analyzers() {
        findings.extend(run_analyzer(analyzer.as_ref(), ctx, gpu));
    }
    Report { findings }
}
