//! PDF overlay / redaction-attack detector - the primary document-fraud
//! technique: scan a document, cover original text/digits with a white rectangle
//! or FreeText annotation, type new values on top. Ported from the reference
//! implementation. CPU-only (sift annotation/image/text model + raw-byte
//! content-stream and object scanning).

use crate::analyzer::Analyzer;
use crate::{Context, Finding, Severity};

pub struct OverlayDetector;

impl Analyzer for OverlayDetector {
    fn name(&self) -> &'static str {
        "pdf_overlay"
    }

    fn applies_to(&self, ctx: &Context) -> bool {
        ctx.is_pdf()
    }

    fn cpu(&self, ctx: &Context) -> Vec<Finding> {
        let mut findings = Vec::new();
        let doc = match sift::read(&ctx.raw_bytes) {
            Ok(d) => d,
            Err(_) => return findings,
        };
        self.check_suspicious_annotations(&doc, &mut findings);
        self.check_content_stream_overlays(&ctx.raw_bytes, &mut findings);
        self.check_mixed_content(&doc, &mut findings);
        self.check_font_inconsistency(&doc, &ctx.raw_bytes, &mut findings);
        self.check_page_image_sizes(&ctx.raw_bytes, &mut findings);
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

impl OverlayDetector {
    fn check_suspicious_annotations(
        &self,
        doc: &sift::SiftDocument<'_>,
        findings: &mut Vec<Finding>,
    ) {
        let annotations = match doc.all_annotations() {
            Ok(a) => a,
            Err(_) => return,
        };
        let mut freetext = 0;
        let mut redact = 0;
        let mut stamp = 0;
        let mut overlay_desc = Vec::new();
        for a in &annotations {
            match a.annot_type {
                sift::AnnotationType::FreeText => {
                    freetext += 1;
                    overlay_desc.push(format!(
                        "FreeText on page {} at [{:.0},{:.0},{:.0},{:.0}]",
                        a.page_index + 1,
                        a.rect[0],
                        a.rect[1],
                        a.rect[2],
                        a.rect[3]
                    ));
                }
                sift::AnnotationType::Redact => redact += 1,
                sift::AnnotationType::Stamp => stamp += 1,
                _ => {}
            }
        }
        if freetext > 0 {
            findings.push(Finding::new(
                "pdf_overlay",
                "pdf_freetext_overlay",
                format!(
                    "PDF contains {freetext} FreeText annotation(s) - text overlaid on document: {}. \
                     This is the primary technique for digit/text manipulation in scanned documents",
                    overlay_desc.join("; ")
                ),
                Severity::High,
                0.75,
            ));
        }
        if redact > 0 {
            findings.push(Finding::new(
                "pdf_overlay",
                "pdf_redaction_annotations",
                format!("PDF contains {redact} redaction annotation(s) - content has been deliberately hidden"),
                Severity::Medium,
                0.8,
            ));
        }
        if stamp > 0 {
            findings.push(Finding::new(
                "pdf_overlay",
                "pdf_stamp_overlay",
                format!("PDF contains {stamp} stamp annotation(s) - content overlaid on pages"),
                Severity::Low,
                0.5,
            ));
        }
    }

    fn check_content_stream_overlays(&self, raw: &[u8], findings: &mut Vec<Finding>) {
        let content = String::from_utf8_lossy(raw);
        let mut white_rect = 0;
        let mut fill_before_text = 0;
        let re_positions: Vec<usize> = content.match_indices(" re").map(|(p, _)| p).collect();
        for &re_pos in &re_positions {
            let after_re = &content[re_pos..content.len().min(re_pos + 100)];
            let has_fill = after_re.contains(" f\n")
                || after_re.contains(" f ")
                || after_re.contains(" f*")
                || after_re.contains(" F\n");
            if !has_fill {
                continue;
            }
            let before = &content[re_pos.saturating_sub(200)..re_pos];
            let white = before.contains("1 1 1 rg")
                || before.contains("1 1 1 RG")
                || before.contains("1.0 1.0 1.0 rg")
                || before.contains("1 g")
                || before.contains("1.0 g");
            if white {
                white_rect += 1;
            }
            let after_fill = &content[re_pos..content.len().min(re_pos + 500)];
            if after_fill.contains("BT") {
                fill_before_text += 1;
            }
        }
        if white_rect > 0 {
            findings.push(Finding::new(
                "pdf_overlay",
                "pdf_white_rectangle_overlay",
                format!(
                    "PDF contains {white_rect} white-filled rectangle(s) - classic overlay attack \
                     pattern used to cover original content before placing new text"
                ),
                Severity::Critical,
                0.85,
            ));
        }
        if fill_before_text > 0 && white_rect == 0 {
            findings.push(Finding::new(
                "pdf_overlay",
                "pdf_filled_rect_before_text",
                format!(
                    "{fill_before_text} instance(s) of filled rectangles followed by text operations - \
                     may indicate content overlay"
                ),
                Severity::Medium,
                0.6,
            ));
        }
    }

    fn check_mixed_content(&self, doc: &sift::SiftDocument<'_>, findings: &mut Vec<Finding>) {
        let images = match doc.images() {
            Ok(i) => i,
            Err(_) => return,
        };
        let text_pages = match doc.text_pages() {
            Ok(t) => t,
            Err(_) => return,
        };
        if images.is_empty() {
            return;
        }
        let mut mixed = Vec::new();
        for (page_idx, text) in text_pages.iter().enumerate() {
            let has_large_image = images
                .iter()
                .any(|img| img.page == page_idx as u32 && img.width > 500 && img.height > 500);
            let trimmed = text.trim();
            let has_text = !trimmed.is_empty() && trimmed.len() > 5;
            if has_large_image && has_text {
                mixed.push(page_idx + 1);
            }
        }
        if !mixed.is_empty() {
            findings.push(Finding::new(
                "pdf_overlay",
                "pdf_mixed_scan_and_text",
                format!(
                    "Pages {mixed:?} contain both scanned images AND vector text - a genuine scan \
                     should not have selectable/vector text overlaid. Text may have been added to \
                     modify scanned content"
                ),
                Severity::High,
                0.70,
            ));
        }
    }

    fn check_font_inconsistency(
        &self,
        doc: &sift::SiftDocument<'_>,
        raw: &[u8],
        findings: &mut Vec<Finding>,
    ) {
        let images = match doc.images() {
            Ok(i) => i,
            Err(_) => return,
        };
        let is_scan_like = images
            .iter()
            .any(|img| img.width > 1000 && img.height > 1000);
        if !is_scan_like {
            return;
        }
        let content = String::from_utf8_lossy(raw);
        let mut font_names = Vec::new();
        for (pos, _) in content.match_indices("/BaseFont") {
            let after = &content[pos + 10..content.len().min(pos + 80)];
            if let Some(end) = after.find(|c: char| c.is_whitespace() || c == '/' || c == '>') {
                let name = after[..end].trim().trim_start_matches('/');
                if !name.is_empty() {
                    font_names.push(name.to_string());
                }
            }
        }
        font_names.sort();
        font_names.dedup();
        if !font_names.is_empty() {
            findings.push(Finding::new(
                "pdf_overlay",
                "pdf_fonts_in_scan",
                format!(
                    "Scanned document contains {} embedded font(s): {} - fonts should not be present \
                     in a pure scan. Text was likely added or modified using a PDF editor",
                    font_names.len(),
                    font_names.join(", ")
                ),
                Severity::High,
                0.80,
            ));
        }
        if font_names.len() > 2 {
            findings.push(Finding::new(
                "pdf_overlay",
                "pdf_multiple_fonts_in_scan",
                format!(
                    "Scanned document uses {} different fonts - multiple font families in an \
                     allegedly scanned document is highly suspicious",
                    font_names.len()
                ),
                Severity::Critical,
                0.85,
            ));
        }
    }

    fn check_page_image_sizes(&self, raw: &[u8], findings: &mut Vec<Finding>) {
        // (obj_num, width, height, stream_length)
        let mut images: Vec<(u32, u32, u32, usize)> = Vec::new();
        let mut pos = 0;
        while pos < raw.len().saturating_sub(20) {
            let Some(obj_pos) = find_bytes(raw, b" 0 obj", pos) else {
                break;
            };
            let obj_start = raw[..obj_pos]
                .iter()
                .rposition(|&b| b == b'\n' || b == b'\r' || b == b' ')
                .map(|p| p + 1)
                .unwrap_or(0);
            let obj_num: u32 = std::str::from_utf8(&raw[obj_start..obj_pos])
                .ok()
                .and_then(|s| s.trim().parse().ok())
                .unwrap_or(0);
            let dict_end = (obj_pos + 1000).min(raw.len());
            let dict = &raw[obj_pos..dict_end];
            if dict.windows(6).any(|w| w == b"/Image") {
                let width = extract_int(dict, b"/Width ");
                let height = extract_int(dict, b"/Height ");
                let length = extract_int(dict, b"/Length ");
                if width > 0 && height > 0 && length > 0 {
                    images.push((obj_num, width, height, length as usize));
                }
            }
            pos = obj_pos + 6;
        }
        if images.is_empty() {
            return;
        }
        let full_page: Vec<_> = images.iter().filter(|i| i.2 > 500).collect();
        let strips: Vec<_> = images.iter().filter(|i| i.2 <= 100 && i.1 > 500).collect();

        for img in &full_page {
            let pixels = img.1 as u64 * img.2 as u64;
            let bpp = img.3 as f64 / pixels as f64;
            if bpp < 0.05 && pixels > 500_000 {
                findings.push(Finding::new(
                    "pdf_overlay",
                    "pdf_scan_replaced_with_placeholder",
                    format!(
                        "Page image (obj {}, {}x{}) has abnormally small data ({} bytes, {bpp:.4} \
                         bytes/pixel) - a genuine scan at this resolution would be 100-500KB. This \
                         indicates the original scan was replaced with a heavily compressed \
                         placeholder by a PDF editor, which is a strong sign of page-level manipulation",
                        img.0, img.1, img.2, img.3
                    ),
                    Severity::Critical,
                    0.90,
                ));
            }
        }

        if !strips.is_empty() && !full_page.is_empty() {
            findings.push(Finding::new(
                "pdf_overlay",
                "pdf_mixed_image_encoding",
                format!(
                    "PDF contains {} strip-encoded scan images and {} full-page images - genuine \
                     scanner output uses consistent encoding. Mixed encoding suggests some pages \
                     were re-processed by a PDF editor",
                    strips.len(),
                    full_page.len()
                ),
                Severity::High,
                0.80,
            ));
        }

        if images.len() >= 2 {
            for img in &images {
                if img.3 > 0 && img.2 > 500 {
                    let strip_total: usize = strips.iter().map(|s| s.3).sum();
                    if strip_total > 0 {
                        let ratio = strip_total as f64 / strips.len() as f64 / img.3 as f64;
                        if ratio > 10.0 {
                            findings.push(Finding::new(
                                "pdf_overlay",
                                "pdf_page_image_size_disparity",
                                format!(
                                    "Page image (obj {}, {}x{}, {} bytes) is {ratio:.0}x smaller than \
                                     average page scan ({:.0} bytes) - pages scanned together should \
                                     have similar image sizes. This page was likely replaced or \
                                     re-encoded by a PDF editor",
                                    img.0,
                                    img.1,
                                    img.2,
                                    img.3,
                                    strip_total as f64 / strips.len() as f64
                                ),
                                Severity::High,
                                0.85,
                            ));
                        }
                    }
                }
            }
        }
    }
}

/// Find a byte pattern from an offset.
fn find_bytes(haystack: &[u8], needle: &[u8], start: usize) -> Option<usize> {
    if start >= haystack.len() {
        return None;
    }
    haystack[start..]
        .windows(needle.len())
        .position(|w| w == needle)
        .map(|p| p + start)
}

/// Extract the integer following a PDF key (e.g. `/Width 1228`).
fn extract_int(dict: &[u8], key: &[u8]) -> u32 {
    if let Some(pos) = dict.windows(key.len()).position(|w| w == key) {
        let after = &dict[pos + key.len()..];
        let s: String = after
            .iter()
            .take_while(|&&b| b.is_ascii_digit())
            .map(|&b| b as char)
            .collect();
        s.parse().unwrap_or(0)
    } else {
        0
    }
}
