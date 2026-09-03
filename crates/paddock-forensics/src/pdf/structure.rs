//! PDF structural forensics: incremental saves, rebuilt xref, metadata
//! consistency, scanner->editor tool chain, JavaScript / OpenAction, editable
//! form fields, and known online-editor producers. Ported from the reference
//! implementation. CPU-only (byte scanning + sift metadata/form model).

use crate::analyzer::Analyzer;
use crate::{Context, Finding, Severity};

pub struct PdfStructureAnalyzer;

impl Analyzer for PdfStructureAnalyzer {
    fn name(&self) -> &'static str {
        "pdf_structure"
    }

    fn applies_to(&self, ctx: &Context) -> bool {
        ctx.is_pdf()
    }

    fn cpu(&self, ctx: &Context) -> Vec<Finding> {
        let mut findings = Vec::new();
        let tags = sift::read(&ctx.raw_bytes)
            .map(|d| d.tags())
            .unwrap_or_default();

        self.check_incremental_saves(&ctx.raw_bytes, &mut findings);
        self.check_metadata_consistency(&tags, &mut findings);
        self.check_javascript(&ctx.raw_bytes, &tags, &mut findings);
        self.check_form_fields(&ctx.raw_bytes, &mut findings);
        self.check_producer_tools(&tags, &mut findings);
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

impl PdfStructureAnalyzer {
    fn check_incremental_saves(&self, raw: &[u8], findings: &mut Vec<Finding>) {
        let content = String::from_utf8_lossy(raw);
        let eof_count = content.matches("%%EOF").count();
        let xref_count = content.matches("xref").count() + content.matches("/Type /XRef").count();

        if eof_count > 1 {
            findings.push(Finding::new(
                "pdf_structure",
                "pdf_incremental_saves",
                format!(
                    "PDF has {} incremental save(s) ({eof_count} %%EOF markers, {xref_count} xref \
                     sections) - document was edited and re-saved {} time(s) after initial creation",
                    eof_count - 1,
                    eof_count - 1
                ),
                if eof_count > 2 { Severity::High } else { Severity::Medium },
                0.80,
            ));
        }
        if xref_count > 1 && eof_count == 1 {
            findings.push(Finding::new(
                "pdf_structure",
                "pdf_rebuilt_xref",
                format!(
                    "PDF has {xref_count} xref sections but only 1 %%EOF - document structure was \
                     rebuilt, possibly to hide editing history"
                ),
                Severity::Medium,
                0.6,
            ));
        }
    }

    fn check_metadata_consistency(&self, tags: &[sift::Tag], findings: &mut Vec<Finding>) {
        let create = tags
            .iter()
            .find(|t| t.name == "CreateDate" || t.name == "CreationDate");
        let modify = tags
            .iter()
            .find(|t| t.name == "ModDate" || t.name == "ModifyDate");
        if let (Some(c), Some(m)) = (create, modify)
            && c.value != m.value
        {
            findings.push(Finding::new(
                "pdf_structure",
                "pdf_dates_differ",
                format!(
                    "PDF creation date ({}) differs from modification date ({}) - document was \
                         modified after initial creation",
                    c.value, m.value
                ),
                Severity::Low,
                0.7,
            ));
        }

        let creator = tags.iter().find(|t| t.name == "Creator");
        let producer = tags.iter().find(|t| t.name == "Producer");
        if let (Some(creator), Some(producer)) = (creator, producer) {
            let cv = creator.value.to_lowercase();
            let pv = producer.value.to_lowercase();
            let scanners = [
                "scan", "epson", "canon", "hp ", "xerox", "brother", "fujitsu",
            ];
            let editors = [
                "acrobat",
                "pdf-xchange",
                "foxit",
                "nitro",
                "smallpdf",
                "ilovepdf",
            ];
            if scanners.iter().any(|s| cv.contains(s)) && editors.iter().any(|s| pv.contains(s)) {
                findings.push(Finding::new(
                    "pdf_structure",
                    "pdf_scanner_then_editor",
                    format!(
                        "PDF created by scanner ('{}') but processed by editor ('{}') - scanned \
                         document was subsequently edited",
                        creator.value, producer.value
                    ),
                    Severity::High,
                    0.75,
                ));
            }
        }
    }

    fn check_javascript(&self, raw: &[u8], tags: &[sift::Tag], findings: &mut Vec<Finding>) {
        let content = String::from_utf8_lossy(raw);
        let has_js = content.contains("/JS ")
            || content.contains("/JavaScript")
            || content.contains("/Type /Action")
            || tags
                .iter()
                .any(|t| t.name == "JavaScript" && t.value == "Yes");
        if has_js {
            findings.push(Finding::new(
                "pdf_structure",
                "pdf_contains_javascript",
                "PDF contains JavaScript - can be used to dynamically alter displayed content, hide \
                 information, or execute code",
                Severity::High,
                0.9,
            ));
        }
        if content.contains("/OpenAction") {
            findings.push(Finding::new(
                "pdf_structure",
                "pdf_open_action",
                "PDF has an OpenAction - code executes automatically when opened",
                Severity::Medium,
                0.8,
            ));
        }
    }

    fn check_form_fields(&self, raw: &[u8], findings: &mut Vec<Finding>) {
        let doc = match sift::read(raw) {
            Ok(d) => d,
            Err(_) => return,
        };
        let form = match doc.acro_form() {
            Ok(Some(f)) => f,
            _ => return,
        };
        let filled: Vec<&sift::FormField> =
            form.fields.iter().filter(|f| f.value.is_some()).collect();
        if !filled.is_empty() {
            let names: Vec<String> = filled
                .iter()
                .take(5)
                .map(|f| format!("{}='{}'", f.full_name, f.value.as_deref().unwrap_or("")))
                .collect();
            findings.push(Finding::new(
                "pdf_structure",
                "pdf_filled_form_fields",
                format!(
                    "PDF contains {} filled form field(s): {} - form fields can be modified after \
                     document creation",
                    filled.len(),
                    names.join(", ")
                ),
                Severity::Low,
                0.5,
            ));
        }
        if form.need_appearances {
            findings.push(Finding::new(
                "pdf_structure",
                "pdf_editable_forms",
                "PDF form fields are not flattened (NeedAppearances=true) - field values can be \
                 modified by any PDF viewer",
                Severity::Medium,
                0.7,
            ));
        }
    }

    fn check_producer_tools(&self, tags: &[sift::Tag], findings: &mut Vec<Finding>) {
        let producer = tags
            .iter()
            .find(|t| t.name == "Producer")
            .map(|t| t.value.as_str());
        let creator = tags
            .iter()
            .find(|t| t.name == "Creator")
            .map(|t| t.value.as_str());
        let tool = producer.or(creator).unwrap_or("");
        let tl = tool.to_lowercase();
        let suspicious = [
            ("smallpdf", "online PDF editor"),
            ("ilovepdf", "online PDF editor"),
            ("sejda", "online PDF editor"),
            ("pdf2go", "online PDF editor"),
            ("pdfcandy", "online PDF editor"),
            ("pdfescape", "online PDF form editor"),
            ("pdf-xchange", "PDF editing software"),
            ("phantompdf", "PDF editing software"),
            ("master pdf", "PDF editing software"),
        ];
        for (pat, kind) in &suspicious {
            if tl.contains(pat) {
                findings.push(Finding::new(
                    "pdf_structure",
                    "pdf_editing_tool",
                    format!(
                        "PDF processed with {kind} ('{tool}') - document was edited using {kind}"
                    ),
                    Severity::Medium,
                    0.7,
                ));
                break;
            }
        }
    }
}
