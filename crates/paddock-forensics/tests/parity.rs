//! Two-level parity harness for the forensic analyzers.
//!
//! Level 1 - **regression lock**: the CPU path is deterministic and its output
//! on a fixed synthetic input is pinned to a golden. (The golden was
//! cross-checked once against the CPU ELA reference.) This guards the port from
//! silent drift.
//!
//! Level 2 - **GPU == CPU**: under the `cuda` feature, the GPU path must produce
//! the same findings as the CPU path for the same input. "GPU-first, CPU
//! fallback" is only safe if the fallback is transparent, so we prove it rather
//! than assume it. Comparison is finding-level: exact code + severity, exact
//! region, confidence within a tolerance (f32 vs f64 reductions differ in the
//! last places), and - for findings whose text is integer-only - exact
//! description.
//!
//! The input is synthesized in-process (deterministic, no fixture blob): a
//! max-frequency checkerboard beside a smooth gradient. The checkerboard cannot
//! be reproduced by a Q92 re-save, so it yields a strong, uneven ELA signal that
//! exercises the full block-statistics + outlier-emission path.

use image::{ExtendedColorType, ImageEncoder};
use paddock_forensics::{Context, Finding, Severity, run};

/// Deterministic 512×512 image: left half max-frequency checkerboard (large,
/// uneven Q92 residual), right half smooth gradient (near-zero residual).
fn synth_checker_smooth_png() -> Vec<u8> {
    let (w, h) = (512u32, 512u32);
    let mut rgb = vec![0u8; (w * h * 3) as usize];
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) * 3) as usize;
            let v: u8 = if x < 256 {
                if (x + y) & 1 == 0 { 0 } else { 255 }
            } else {
                (y * 255 / h) as u8
            };
            rgb[i] = v;
            rgb[i + 1] = v;
            rgb[i + 2] = v;
        }
    }
    let mut png = std::io::Cursor::new(Vec::new());
    image::codecs::png::PngEncoder::new(&mut png)
        .write_image(&rgb, w, h, ExtendedColorType::Rgb8)
        .expect("encode synth png");
    png.into_inner()
}

/// 256×256 with two distinct edge-sharpness populations: the top 2 block-rows
/// carry a smooth high-amplitude sinusoid (gradual = LARGE edge width), the rest
/// carries 2px hard bars (HARD steps = small edge width). This gives the
/// edge-sharpness / texture / wavelet reductions a real spread across blocks so
/// findings actually emit - exercising the full flag path, not just "both empty".
#[cfg(feature = "cuda")]
fn synth_varied_sharpness_png() -> Vec<u8> {
    let (w, h) = (256u32, 256u32);
    let mut rgb = vec![0u8; (w * h * 3) as usize];
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) * 3) as usize;
            // Blurred is the MAJORITY (smooth sinusoid, gradual = large width);
            // the bottom two block-rows are the SHARP minority (2px hard bars =
            // small width). Sharp-as-minority keeps them as MAD outliers so
            // findings emit - a sharp majority would collapse the MAD to 0.
            let v: u8 = if y < 192 {
                (128.0 + 90.0 * ((x as f64) * 0.25).sin())
                    .round()
                    .clamp(0.0, 255.0) as u8
            } else {
                if (x / 2) % 2 == 0 { 50 } else { 205 }
            };
            rgb[i] = v;
            rgb[i + 1] = v;
            rgb[i + 2] = v;
        }
    }
    let mut png = std::io::Cursor::new(Vec::new());
    image::codecs::png::PngEncoder::new(&mut png)
        .write_image(&rgb, w, h, ExtendedColorType::Rgb8)
        .expect("encode varied-sharpness png");
    png.into_inner()
}

/// A flat gray image - an authentic, uniform source. ELA must stay silent.
fn flat_gray_png() -> Vec<u8> {
    let (w, h) = (256u32, 256u32);
    let rgb = vec![128u8; (w * h * 3) as usize];
    let mut png = std::io::Cursor::new(Vec::new());
    image::codecs::png::PngEncoder::new(&mut png)
        .write_image(&rgb, w, h, ExtendedColorType::Rgb8)
        .expect("encode flat png");
    png.into_inner()
}

/// The checkerboard|smooth image as a JPEG - a JPEG container so the JPEG-only
/// jpeg_ghost analyzer applies (alongside ela + noise).
fn synth_checker_smooth_jpeg() -> Vec<u8> {
    let (w, h) = (512u32, 512u32);
    let mut rgb = vec![0u8; (w * h * 3) as usize];
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) * 3) as usize;
            let v: u8 = if x < 256 {
                if (x + y) & 1 == 0 { 0 } else { 255 }
            } else {
                (y * 255 / h) as u8
            };
            rgb[i] = v;
            rgb[i + 1] = v;
            rgb[i + 2] = v;
        }
    }
    let mut jpg = std::io::Cursor::new(Vec::new());
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut jpg, 92)
        .write_image(&rgb, w, h, ExtendedColorType::Rgb8)
        .expect("encode jpeg");
    jpg.into_inner()
}

/// The pinned CPU result for the JPEG synth (ela is JPEG-only, per the
/// reference's should_skip). The ELA/noise algorithms were cross-checked against
/// the CPU reference; this pins their deterministic output on the JPEG input.
const GOLDEN_JSON: &str = r#"[
  {
    "analyzer": "ela",
    "code": "ela_block_outliers",
    "description": "7.8% of blocks are statistical ELA outliers (80 high, 0 low out of 1024 blocks) - complexity-normalized analysis indicates tampering",
    "severity": "medium",
    "confidence": 0.70625
  },
  {
    "analyzer": "noise",
    "code": "noise_inconsistency",
    "description": "Noise level inconsistency detected via wavelet MAD estimation (median σ=376.58, robust CV=1.00, 0.0% anomalous blocks: 0 low-noise, 0 high-noise) - suggests splicing or AI generation",
    "severity": "medium",
    "confidence": 0.5
  }
]"#;

fn golden() -> Vec<Finding> {
    serde_json::from_str(GOLDEN_JSON).expect("parse golden")
}

/// Findings minus the inherently non-deterministic analyzers, for the
/// exact-equality checks. `copy_move`'s RANSAC seeds from `SystemTime`, so two
/// runs (or the CPU vs GPU-delegating paths) can differ legitimately - it is
/// verbatim-faithful to the reference and stays in the default set, but it cannot
/// participate in a byte-for-byte equality assertion.
fn deterministic(findings: &[Finding]) -> Vec<Finding> {
    findings
        .iter()
        .filter(|f| f.analyzer != "copy_move")
        .cloned()
        .collect()
}

/// Finding-level equivalence used for the GPU==CPU check.
#[cfg(feature = "cuda")]
fn assert_findings_equivalent(a: &[Finding], b: &[Finding]) {
    assert_eq!(
        a.len(),
        b.len(),
        "finding count differs:\n a={a:#?}\n b={b:#?}"
    );
    for (x, y) in a.iter().zip(b.iter()) {
        assert_eq!(x.code, y.code, "code differs");
        assert_eq!(x.severity, y.severity, "severity differs for {}", x.code);
        assert_eq!(x.region, y.region, "region differs for {}", x.code);
        assert!(
            (x.confidence - y.confidence).abs() < 1e-4,
            "confidence differs for {}: {} vs {}",
            x.code,
            x.confidence,
            y.confidence
        );
    }
}

#[test]
fn level1_cpu_matches_golden() {
    // ELA is JPEG-only (reference should_skip parity), so the golden is pinned on
    // the JPEG synth where ela + noise both fire.
    let ctx = Context::from_bytes(synth_checker_smooth_jpeg()).expect("decode synth");
    let report = run(&ctx, None);
    let golden = golden();
    // The golden pins the Wave 0/1 analyzers (ela, noise). Later waves add more
    // analyzers to the default set, so scope the comparison to the analyzers the
    // golden actually covers - this stays a tight drift guard on ela/noise
    // without having to re-bless the golden every time a new analyzer lands.
    let covered: std::collections::HashSet<&str> =
        golden.iter().map(|f| f.analyzer.as_str()).collect();
    let scoped: Vec<Finding> = report
        .findings
        .iter()
        .filter(|f| covered.contains(f.analyzer.as_str()))
        .cloned()
        .collect();
    assert_eq!(
        scoped, golden,
        "CPU findings drifted from the golden:\n got={scoped:#?}\n want={golden:#?}"
    );
}

#[test]
fn level1_cpu_deterministic() {
    let ctx = Context::from_bytes(synth_checker_smooth_png()).expect("decode synth");
    let a = run(&ctx, None);
    let b = run(&ctx, None);
    assert_eq!(
        deterministic(&a.findings),
        deterministic(&b.findings),
        "CPU path is not deterministic"
    );
}

#[test]
fn flat_image_flags_unnaturally_low_noise() {
    // A perfectly flat image is not "clean" - zero sensor noise is itself a
    // signal (denoised / AI). ELA stays silent (a flat image re-saves exactly)
    // and jpeg_ghost is PNG-gated off. Wave 2a's screenshot detector also
    // legitimately fires here (a flat image has near-zero color diversity), so
    // assert on the noise finding specifically rather than a total count.
    let ctx = Context::from_bytes(flat_gray_png()).expect("decode flat");
    let report = run(&ctx, None);
    let noise: Vec<&Finding> = report
        .findings
        .iter()
        .filter(|f| f.analyzer == "noise")
        .collect();
    assert_eq!(noise.len(), 1, "got {:#?}", report.findings);
    assert_eq!(noise[0].code, "unnaturally_low_noise");
    // Severity ordering sanity (keeps the enum wiring honest).
    assert!(Severity::High > Severity::Low);
}

#[cfg(feature = "cuda")]
#[test]
fn level2_gpu_matches_cpu() {
    let gpu = match paddock_forensics::gpu::ForensicGpu::new(0) {
        Ok(g) => g,
        Err(e) => {
            // No usable GPU in this environment -> nothing to compare. The CPU
            // fallback is what runs in that case, and Level 1 already covers it.
            eprintln!("skipping GPU parity: {e}");
            return;
        }
    };

    // Firing case: the synthetic checkerboard.
    let ctx = Context::from_bytes(synth_checker_smooth_png()).expect("decode synth");
    let cpu = run(&ctx, None);
    let gpu_r = run(&ctx, Some(&gpu));
    assert!(
        !cpu.findings.is_empty(),
        "expected the synth to fire on CPU"
    );
    assert_findings_equivalent(
        &deterministic(&cpu.findings),
        &deterministic(&gpu_r.findings),
    );
    // For this input the description is integer-only, so it must match exactly
    // (excluding the SystemTime-seeded copy_move).
    assert_eq!(
        deterministic(&cpu.findings),
        deterministic(&gpu_r.findings),
        "GPU findings differ from CPU on the firing synth"
    );

    // noise: flat image -> unnaturally_low_noise, GPU == CPU.
    let ctx0 = Context::from_bytes(flat_gray_png()).expect("decode flat");
    let cpu0 = run(&ctx0, None);
    let gpu0 = run(&ctx0, Some(&gpu));
    assert!(
        cpu0.findings
            .iter()
            .any(|f| f.code == "unnaturally_low_noise")
    );
    assert_findings_equivalent(
        &deterministic(&cpu0.findings),
        &deterministic(&gpu0.findings),
    );

    // jpeg_ghost (JPEG-only) + ela + noise all run on a JPEG input.
    let ctxj = Context::from_bytes(synth_checker_smooth_jpeg()).expect("decode jpeg");
    let cpuj = run(&ctxj, None);
    let gpuj = run(&ctxj, Some(&gpu));
    assert!(
        cpuj.findings.iter().any(|f| f.analyzer == "jpeg_ghost"),
        "jpeg_ghost should run on a JPEG: {:#?}",
        cpuj.findings
    );
    assert_findings_equivalent(
        &deterministic(&cpuj.findings),
        &deterministic(&gpuj.findings),
    );
    assert_eq!(
        deterministic(&cpuj.findings),
        deterministic(&gpuj.findings),
        "GPU != CPU on the JPEG input"
    );
}

/// HEIC decode via the `heic` feature (libheif). Runs only when
/// `PADDOCK_FORENSICS_HEIC_SAMPLE` points at a real HEIC file - so it verifies
/// the libheif path against a genuine iPhone-style capture without committing a
/// binary fixture (no-ops in CI where the env is unset).
#[cfg(feature = "heic")]
#[test]
fn heic_decodes_when_sample_present() {
    let Ok(path) = std::env::var("PADDOCK_FORENSICS_HEIC_SAMPLE") else {
        eprintln!("skipping HEIC decode: set PADDOCK_FORENSICS_HEIC_SAMPLE to a .heic file");
        return;
    };
    let bytes = std::fs::read(&path).expect("read heic sample");
    let ctx = Context::from_bytes(bytes).expect("decode heic via libheif");
    assert!(
        ctx.width > 1 && ctx.height > 1,
        "decoded dims {}x{}",
        ctx.width,
        ctx.height
    );
    assert!(!ctx.gray().is_empty(), "gray plane populated");
    // The full analyzer set must run on a HEIC-sourced context without panicking.
    let report = run(&ctx, None);
    let _ = report.risk();
}

/// Direct per-analyzer kernel parity for the Wave 3+ GPU analyzers: call
/// `cpu()` and `gpu()` on the same context (bypassing `applies_to` content
/// gating, which would otherwise skip these on the synth's classification) and
/// require byte-identical findings. The checkerboard|smooth image has dense,
/// uneven edges, so it drives the per-block reductions hard. Each kernel is
/// f64 + `--fmad=false`, so equality is exact (no tolerance).
#[cfg(feature = "cuda")]
#[test]
fn wave3_gpu_kernels_match_cpu_exactly() {
    use paddock_forensics::Analyzer;
    use paddock_forensics::gpu::ForensicGpu;

    let gpu = match ForensicGpu::new(0) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("skipping GPU kernel parity: {e}");
            return;
        }
    };
    let ctx = Context::from_bytes(synth_varied_sharpness_png()).expect("decode synth");

    // Assert a GPU analyzer's kernel reproduces its CPU reference bit-for-bit,
    // and - for the edge-rich synth - that it actually fired (guards against a
    // vacuous "both empty" pass).
    fn check<A: Analyzer>(a: &A, ctx: &Context, gpu: &ForensicGpu, expect_fires: bool) {
        let cpu = a.cpu(ctx);
        let gpu_r = a.gpu(gpu, ctx).expect("gpu path");
        assert_eq!(cpu, gpu_r, "{}: GPU kernel diverged from CPU", a.name());
        if expect_fires {
            assert!(
                !cpu.is_empty(),
                "{}: expected findings on the synth",
                a.name()
            );
        }
    }

    use paddock_forensics::pixel::anti_forensics::AntiForensicsDetector;
    use paddock_forensics::pixel::channel_correlation::ChannelCorrelationAnalyzer;
    use paddock_forensics::pixel::color_consistency::ColorConsistencyAnalyzer;
    use paddock_forensics::pixel::edge_sharpness::EdgeSharpnessAnalyzer;
    use paddock_forensics::pixel::lighting_consistency::LightingConsistencyAnalyzer;
    use paddock_forensics::pixel::texture::TextureConsistencyAnalyzer;
    use paddock_forensics::pixel::wavelet_consistency::WaveletConsistencyAnalyzer;

    check(&EdgeSharpnessAnalyzer::default(), &ctx, &gpu, true);
    // The remaining Wave 3a kernels must match CPU bit-for-bit; whether they
    // fire depends on the synth's per-block spread, so parity (not firing) is
    // the hard assertion for these.
    check(&ChannelCorrelationAnalyzer::default(), &ctx, &gpu, false);
    check(&WaveletConsistencyAnalyzer::default(), &ctx, &gpu, false);
    check(&TextureConsistencyAnalyzer::default(), &ctx, &gpu, false);
    check(&ColorConsistencyAnalyzer::default(), &ctx, &gpu, false);
    check(&AntiForensicsDetector::default(), &ctx, &gpu, false);
    // lighting: kernel emits raw plane gradient (a,b); direction/magnitude and
    // all neighbour trig run host-side, so GPU == CPU exactly.
    check(&LightingConsistencyAnalyzer::default(), &ctx, &gpu, false);
}

/// A minimal PDF carrying fraud markers the raw-byte structure checks detect:
/// two %%EOF (an incremental save), /JavaScript and /OpenAction. Not a fully
/// valid document - it exercises the byte-level checks deterministically.
fn synth_fraud_pdf() -> Vec<u8> {
    let mut v = Vec::new();
    v.extend_from_slice(b"%PDF-1.4\n");
    v.extend_from_slice(b"1 0 obj<</Type/Catalog>>endobj\n");
    v.extend_from_slice(b"xref\n0 2\ntrailer<</Root 1 0 R>>\n%%EOF\n");
    // incremental save #2 with javascript + auto-run action
    v.extend_from_slice(b"2 0 obj<</S/JavaScript/JS(app.alert\\(1\\))>>endobj\n");
    v.extend_from_slice(b"<</OpenAction 2 0 R>>\n");
    v.extend_from_slice(b"xref\n0 3\ntrailer<</Root 1 0 R>>\n%%EOF\n");
    v
}

#[test]
fn pdf_structure_flags_fraud_markers() {
    let ctx = Context::from_bytes(synth_fraud_pdf()).expect("pdf ctx");
    assert!(ctx.is_pdf(), "recognized as PDF");
    let report = run(&ctx, None);
    let codes: Vec<&str> = report.findings.iter().map(|f| f.code.as_str()).collect();
    assert!(codes.contains(&"pdf_incremental_saves"), "got {codes:?}");
    assert!(codes.contains(&"pdf_contains_javascript"), "got {codes:?}");
    assert!(codes.contains(&"pdf_open_action"), "got {codes:?}");
    // pixel analyzers must not run on a PDF (applies_to gating)
    assert!(
        !report
            .findings
            .iter()
            .any(|f| matches!(f.analyzer.as_str(), "ela" | "noise" | "jpeg_ghost")),
        "pixel analyzers should skip PDFs: {:#?}",
        report.findings
    );
}

#[test]
fn pdf_analysis_deterministic() {
    let ctx = Context::from_bytes(synth_fraud_pdf()).expect("pdf ctx");
    assert_eq!(run(&ctx, None).findings, run(&ctx, None).findings);
}

// ── render_compare: needs pdfium (dev-dep), so it binds its own for the test. ──
#[cfg(feature = "cuda")]
mod render_compare_test {
    use super::*;
    use paddock_forensics::{PageRenderer, RenderCompareOpts, render_compare};
    use paddock_pdfium::Pdfium;
    use std::sync::{Mutex, MutexGuard, OnceLock};

    // Binds its own pdfium: production injects a PageRenderer from the runner,
    // which owns the process-wide one. paddock-pdfium does not serialise for
    // its caller, so the lock lives here too - same contract as pdf.rs.
    static PDFIUM: OnceLock<Mutex<Pdfium>> = OnceLock::new();
    fn pdfium() -> MutexGuard<'static, Pdfium> {
        PDFIUM
            .get_or_init(|| Mutex::new(Pdfium::new()))
            .lock()
            .unwrap_or_else(|e| e.into_inner())
    }

    struct PdfiumRenderer;
    impl PageRenderer for PdfiumRenderer {
        fn render_page(&self, bytes: &[u8], page: u32, dpi: f32) -> Option<(Vec<u8>, u32, u32)> {
            let pdfium = pdfium();
            let doc = pdfium.load(bytes).ok()?;
            let (pw, ph) = doc.page_size(page as usize)?;
            let bmp = doc
                .render(
                    page as usize,
                    (pw * dpi / 72.0).max(1.0) as u32,
                    (ph * dpi / 72.0).max(1.0) as u32,
                )
                .ok()?;
            Some((bmp.rgb, bmp.width, bmp.height))
        }
    }

    #[test]
    fn render_compare_flags_overlay_on_forged_pdf() {
        let path = "/testdata/samples/pdf-forged.pdf";
        let Ok(bytes) = std::fs::read(path) else {
            eprintln!("skip: fixture {path} not present");
            return;
        };
        let findings = render_compare(&bytes, &PdfiumRenderer, &RenderCompareOpts::default());
        eprintln!(
            "render_compare findings: {:?}",
            findings.iter().map(|f| f.code.as_str()).collect::<Vec<_>>()
        );
        assert!(
            findings
                .iter()
                .any(|f| f.code == "pdf_overlay_content_detected"),
            "forged PDF should show overlaid content: {findings:#?}"
        );
        // localized regions carry bounding boxes
        assert!(findings.iter().any(|f| f.region.is_some()));
    }
}

#[test]
fn double_jpeg_runs_and_is_deterministic() {
    use paddock_forensics::Analyzer;
    use paddock_forensics::pixel::double_jpeg::DoubleJpegDetector;
    let ctx = Context::from_bytes(synth_checker_smooth_jpeg()).expect("jpeg ctx");
    assert!(DoubleJpegDetector.applies_to(&ctx), "applies to JPEG");
    let a = DoubleJpegDetector.cpu(&ctx);
    let b = DoubleJpegDetector.cpu(&ctx);
    assert_eq!(a, b, "double_jpeg CPU path is not deterministic");
    // A JPEG always yields at least a quality estimate from its DQT.
    assert!(
        a.iter().any(|f| f.code == "jpeg_quality_estimate"),
        "got {a:#?}"
    );
    // PNG -> not a JPEG -> analyzer does not apply.
    let png = Context::from_bytes(synth_checker_smooth_png()).expect("png ctx");
    assert!(!DoubleJpegDetector.applies_to(&png), "must skip non-JPEG");
}
