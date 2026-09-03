//! Runner-side forensics gate.
//!
//! Builds a [`ForensicRuntime`] from the `[forensics]` config and runs
//! `paddock-forensics` over image attachments on their ORIGINAL bytes, feeding
//! the findings into the model's context (the sift injection lane in
//! [`crate::doc::inject_forensics`]). GPU-first with a parity-verified CPU
//! fallback; the GPU path exists only in a `forensics-cuda` build.

use std::sync::Arc;

use crate::config::{ForensicsAuto, ForensicsConfig};

/// Long-lived forensic runtime, built once at startup when `[forensics]
/// enabled = true` and stored in [`crate::routes::AppState`].
pub struct ForensicRuntime {
    /// When automatic preprocessing runs.
    pub auto: ForensicsAuto,
    /// Whether the on-demand tool surface is exposed (reserved; the tool lands
    /// in a later wave).
    pub tool: bool,
    /// The forensic GPU context (own cudarc context + side stream), if a
    /// `forensics-cuda` build initialized one. `None` -> CPU fallback.
    #[cfg(feature = "forensics-cuda")]
    gpu: Option<paddock_forensics::gpu::ForensicGpu>,
}

impl ForensicRuntime {
    /// Build the runtime, or `None` when forensics is disabled (zero request-path
    /// cost in that case). Never fails: a GPU that will not initialize degrades
    /// to the CPU path with a loud warning, consistent with the rest of the
    /// ingestion lane being CPU (sift, PDF text).
    pub fn build(cfg: &ForensicsConfig) -> Option<Arc<ForensicRuntime>> {
        if !cfg.enabled {
            return None;
        }

        #[cfg(feature = "forensics-cuda")]
        let gpu = {
            let device = cfg.device.unwrap_or(0);
            match paddock_forensics::gpu::ForensicGpu::new(device) {
                Ok(g) => {
                    tracing::info!(device, "forensics: GPU context initialized");
                    Some(g)
                }
                Err(e) => {
                    tracing::warn!(device, error = %e, "forensics: GPU init failed - CPU fallback");
                    None
                }
            }
        };

        tracing::info!(
            auto = ?cfg.auto,
            tool = cfg.tool,
            gpu = cfg!(feature = "forensics-cuda"),
            "forensics enabled"
        );

        Some(Arc::new(ForensicRuntime {
            auto: cfg.auto,
            tool: cfg.tool,
            #[cfg(feature = "forensics-cuda")]
            gpu,
        }))
    }

    /// Auto-run over image attachments (auto = images | all).
    pub fn auto_images(&self) -> bool {
        matches!(self.auto, ForensicsAuto::Images | ForensicsAuto::All)
    }

    /// Auto-run over PDF attachments (auto = all only - PDFs are documents).
    pub fn auto_pdfs(&self) -> bool {
        matches!(self.auto, ForensicsAuto::All)
    }

    /// The configured always-on scope as the wire word the Studio reads
    /// (`off` | `images` | `all`) - advertised on `/server` so the composer can
    /// show the forensics default without re-deriving it from two booleans.
    pub fn auto_word(&self) -> &'static str {
        match self.auto {
            ForensicsAuto::Off => "off",
            ForensicsAuto::Images => "images",
            ForensicsAuto::All => "all",
        }
    }

    /// Run the forensic analyzers over an attachment's ORIGINAL bytes - image OR
    /// PDF. `Context::from_bytes` + each analyzer's `applies_to` pick the right
    /// lane; for a PDF this additionally runs the render-vs-scan comparison,
    /// supplying the runner's process-wide pdfium as the page renderer. Blocking
    /// (JPEG/GPU, and for PDFs pdfium) - call under `spawn_blocking`.
    pub fn analyze(&self, bytes: &[u8]) -> (ForensicMeta, Vec<paddock_forensics::Finding>) {
        let ctx = match paddock_forensics::Context::from_bytes(bytes.to_vec()) {
            Ok(c) => c,
            Err(_) => {
                return (
                    ForensicMeta {
                        content_type: "unknown",
                        ..Default::default()
                    },
                    Vec::new(),
                );
            }
        };
        let is_pdf = ctx.is_pdf();
        let meta = ForensicMeta {
            content_type: content_type_word(&ctx.content_type),
            // A PDF has no single raster format or dimensions (the decoded image
            // is a 1×1 placeholder), so leave them empty / None.
            format: if is_pdf {
                String::new()
            } else {
                ctx.format
                    .map(|f| format!("{f:?}").to_lowercase())
                    .unwrap_or_default()
            },
            width: (!is_pdf).then_some(ctx.width),
            height: (!is_pdf).then_some(ctx.height),
        };
        #[cfg(feature = "forensics-cuda")]
        let mut findings = paddock_forensics::run(&ctx, self.gpu.as_ref()).findings;
        #[cfg(not(feature = "forensics-cuda"))]
        let mut findings = paddock_forensics::run(&ctx, None).findings;

        if is_pdf {
            findings.extend(paddock_forensics::render_compare(
                bytes,
                &PdfiumPageRenderer,
                &paddock_forensics::RenderCompareOpts::default(),
            ));
        }
        (meta, findings)
    }
}

/// Self-describing attachment metadata carried with a report so a persister
/// fills every column without re-decoding the bytes: content classification,
/// raster format, and pixel dimensions (format/dimensions absent for a PDF).
#[derive(Debug, Clone, Default)]
pub struct ForensicMeta {
    /// `photo | document | mixed | unknown`.
    pub content_type: &'static str,
    /// Decoded raster format (`"jpeg"`, `"png"`, ...), or `""` for a PDF.
    pub format: String,
    pub width: Option<u32>,
    pub height: Option<u32>,
}

fn content_type_word(ct: &paddock_forensics::ContentType) -> &'static str {
    use paddock_forensics::ContentType::*;
    match ct {
        Photo => "photo",
        Document => "document",
        Mixed => "mixed",
        Unknown => "unknown",
    }
}

/// Adapts the runner's process-wide pdfium (`crate::pdf`) to the forensics
/// `PageRenderer` trait, so render_compare can rasterize pages without the
/// forensics crate binding its own pdfium (only one binding is allowed per
/// process, and the runner owns it).
struct PdfiumPageRenderer;

impl paddock_forensics::PageRenderer for PdfiumPageRenderer {
    fn render_page(&self, pdf_bytes: &[u8], page: u32, dpi: f32) -> Option<(Vec<u8>, u32, u32)> {
        crate::pdf::render_page_rgb(pdf_bytes, page, dpi)
    }
}

/// Format findings as the injection text block prepended before an image/PDF
/// part. Returns `None` when there are no findings - a clean attachment should
/// not spend context on a "nothing found" note by default.
///
/// This ports the SUBSTANCE of the reference's VLM guidance, not its literal
/// service prompt. That reference is a dedicated forensics service that owns the
/// whole VLM turn and forces a JSON verdict; its findings block is bare data
/// because all the instruction weight lives in the surrounding prompt - verdict
/// anchoring, per-class false-positive caveats, and the receipt
/// arithmetic/VAT/date checks.
/// Paddock has no owning prompt - it ENRICHES a user's ordinary chat turn - so
/// bare findings would be ignored. We therefore carry that instructional
/// substance inline, adapted to guide rather than hijack: no JSON output, no
/// insurance framing, and the receipt block is content-type-gated so it only
/// appears on a document.
pub fn format_injection(
    meta: &ForensicMeta,
    findings: &[paddock_forensics::Finding],
) -> Option<String> {
    if findings.is_empty() {
        return None;
    }
    let ordered = severity_ordered(findings);
    let is_document = matches!(meta.content_type, "document" | "mixed");
    let subject = if is_document { "document" } else { "image" };

    // Group by analyzer, preserving the strongest-first order within each group.
    use std::collections::BTreeMap;
    let mut by: BTreeMap<&str, Vec<&paddock_forensics::Finding>> = BTreeMap::new();
    for f in &ordered {
        by.entry(f.analyzer.as_str()).or_default().push(f);
    }

    // Report-level scoring + dedup for the header summary - the calibrated
    // verdict the model should anchor to (the reference's "OVERALL FORENSIC
    // VERDICT is your ground truth").
    let risk = paddock_forensics::risk::score(findings);

    let mut s = String::new();
    s.push_str(&format!(
        "[Automated {subject} forensics - paddock-forensics - computed on the ORIGINAL uploaded \
         bytes. {} signal(s), strongest first. Risk {:.0}/100 ({}); {}]\n",
        findings.len(),
        risk.risk_score,
        severity_word(risk.risk_level),
        risk.verdict.summary,
    ));
    if !risk.key_findings.is_empty() {
        s.push_str("\nKey findings (deduplicated):\n");
        for k in &risk.key_findings {
            s.push_str(&format!(
                "  - [{}] (confidence {:.0}%) {} - {} [{}]\n",
                severity_word(k.severity).to_uppercase(),
                k.confidence * 100.0,
                k.title,
                k.description,
                k.sources.join(", "),
            ));
        }
        s.push_str("\nAll signals by analyzer:\n");
    }
    // Cap the raw per-analyzer list. A heavily-manipulated attachment can trip
    // hundreds of block-level signals from one analyzer (ELA outliers, noise
    // tiles), and listing every one balloons this ALWAYS-ON note to tens of
    // thousands of tokens - enough to crowd the model's window on its own. The
    // deduplicated `key_findings` above already carry what matters; findings
    // here are severity-ordered, so keeping the strongest few per analyzer keeps
    // the real evidence and sheds the long low-severity tail the guidance
    // already tells the model to discount. Nothing is lost: the COMPLETE
    // per-signal list stays available on demand via the forensics tool
    // (`tool_result_json`/`report_value` are uncapped - a tool result is asked
    // for, not force-fed into every prompt).
    const PER_ANALYZER_CAP: usize = 6;
    let mut omitted = 0usize;
    for (analyzer, fs) in &by {
        s.push_str(&format!("\n{}:\n", analyzer_label(analyzer)));
        for f in fs.iter().take(PER_ANALYZER_CAP) {
            s.push_str(&format!(
                "  - [{}] (confidence {:.0}%) {}{}  «{}»\n",
                severity_word(f.severity).to_uppercase(),
                f.confidence * 100.0,
                f.description,
                region_str(&f.region),
                f.code,
            ));
        }
        if fs.len() > PER_ANALYZER_CAP {
            let more = fs.len() - PER_ANALYZER_CAP;
            omitted += more;
            s.push_str(&format!(
                "  ... and {more} more lower-severity {} signal(s)\n",
                analyzer_label(analyzer),
            ));
        }
    }
    if omitted > 0 {
        s.push_str(&format!(
            "\n({omitted} lower-severity signal(s) omitted from this summary - the deduplicated \
             key findings above capture them; call the forensics tool for the complete \
             per-signal list.)\n",
        ));
    }

    // How to weigh the signals - the reference's evidence-weighting + verdict-anchoring
    // substance, shared with the on-demand tool result so both surfaces instruct
    // the model identically.
    s.push('\n');
    s.push_str(&weighing_guidance(is_document));

    Some(s)
}

/// The "how to weigh these signals" guidance shared by the always-on injection
/// ([`format_injection`]) and the on-demand tool result ([`tool_result_json`]),
/// so the model gets the same instruction whichever surface fired. Ports the
/// substance of the reference's synthesis prompt (evidence weighting +
/// contradiction) and, for documents, its arithmetic verification -
/// verbatim in meaning, reframed as guidance rather than a JSON-forcing task.
fn weighing_guidance(is_document: bool) -> String {
    let mut s = String::from(
        "How to weigh these: they are automated signal-level evidence, NOT a verdict. The risk \
         score already weighed them for false positives - anchor your judgment to it and do not \
         alarm-spiral on the raw list. Examine the flagged region(s) in the pixels yourself and say \
         whether they CONFIRM or CONTRADICT each signal:\n\
         - Strong, hard to explain away: arithmetic errors, copy-move with high spatial \
         consistency, and the SAME region flagged by several independent analyzers.\n\
         - Moderate, often benign: noise inconsistencies (JPEG compression causes them), splice \
         boundaries (text edges trip these), texture anomalies.\n\
         - Weak / likely false positive: anti-forensics on an otherwise normal image, histogram \
         gaps (normal for photos), and lone low-confidence signals.\n\
         The absence of signals is not proof of authenticity, and you must not invent findings \
         beyond those listed above.",
    );
    if is_document {
        // From the reference's synthesis prompt - independent arithmetic verification.
        s.push_str(
            "\n\nBecause this is a document: read every digit yourself from the pixels - do not \
             trust any embedded text layer, which regularly misreads digits (5↔6, 3↔8, 0↔O, 1↔I). \
             If it is a receipt, invoice, or other financial document, verify the arithmetic, which \
             is the single most reliable sign of document fraud: unit price × quantity = each line \
             total; the line totals sum to the subtotal; printed subtotal × the PRINTED tax rate ≈ \
             the printed tax amount; and subtotal + tax = the grand total. Do NOT flag a tax rate as \
             wrong merely because it differs from what you expect for the jurisdiction - VAT/sales \
             rates change over time and vary by product category (food, books, medicine, transport, \
             culture), and multiple VAT rows on one receipt (e.g. 6%, 12%, 25%) are normal, not a \
             fraud signal; only flag tax math that is internally inconsistent on the document \
             itself. A printed date is suspicious only if it is genuinely implausible (e.g. later \
             than when the file was created) - do not invent a \"current date\". On a multi-page \
             document, compare every field that repeats across pages (dates, reference numbers, \
             names, amounts); an inconsistency between pages is a strong signal. Note that on \
             documents, text-alignment and font findings are often false positives from \
             thermal-printer irregularity, and noise near text edges is usually a printing artifact \
             rather than manipulation.",
        );
    }
    s
}

// ---------------------------------------------------------------------------
// Tool surface (Surface B): the model-pull server tool. Off unless `[forensics]
// tool = true` AND the request declares `{"type":"forensics"}`, mirroring how
// web_search is both configured on the runner and requested per call.
// ---------------------------------------------------------------------------

/// The function name the model calls.
pub const TOOL_NAME: &str = "analyze_document_forensics";

const TOOL_DESC: &str = "Run forensic signal analysis (Error Level Analysis and \
    related pixel/structure checks) on an image or PDF document ALREADY PRESENT in this \
    conversation, reading its original uploaded bytes. Use it when authenticity or \
    tampering is in question. Returns detected manipulation signals with severity and \
    confidence; the ABSENCE of signals is evidence, not proof of authenticity.";

/// The tool schema disclosed to the model (chat/responses nested-function shape,
/// same as web_search).
pub fn tool_def() -> serde_json::Value {
    serde_json::json!({
        "type": "function",
        "function": {
            "name": TOOL_NAME,
            "description": TOOL_DESC,
            "parameters": {
                "type": "object",
                "properties": {
                    "image_index": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "0-based index of the attachment to analyze, counting images \
                                        and PDF documents together in order across the conversation. \
                                        Omit to analyze the most recent attachment."
                    }
                },
                "additionalProperties": false
            }
        }
    })
}

/// The Anthropic `/v1/messages` tool def (`{name, description, input_schema}`),
/// the shape that path's `tools` array carries before `convert_tools` nests it.
/// Same identity and schema as [`tool_def`] - only the envelope differs, exactly
/// as web search has an OpenAI and an Anthropic def.
pub fn anthropic_tool_def() -> serde_json::Value {
    let f = tool_def();
    let func = &f["function"];
    serde_json::json!({
        "name": func["name"],
        "description": func["description"],
        "input_schema": func["parameters"],
    })
}

/// Parse the `image_index` argument (JSON object), if any.
pub fn parse_image_index(arguments: &str) -> Option<usize> {
    serde_json::from_str::<serde_json::Value>(arguments)
        .ok()?
        .get("image_index")
        .and_then(serde_json::Value::as_u64)
        .map(|v| v as usize)
}

/// The structured result the model reads back (tool message content). Findings
/// are severity-ordered (strongest first) and carry their spatial region so the
/// model can localize and cross-check each signal.
pub fn tool_result_json(meta: &ForensicMeta, findings: &[paddock_forensics::Finding]) -> String {
    let mut v = report_value(meta, findings);
    // The tool result gets the directive that makes the model use the signals; the
    // always-on output item (same `report_value`) does not - it is data for a
    // caller, not a prompt for a model. Same weighing guidance as the injection
    // surface (content-type-aware), plus a one-line map of the JSON shape.
    let is_document = matches!(meta.content_type, "document" | "mixed");
    v["guidance"] = serde_json::json!(format!(
        "`key_findings` are the deduplicated, categorized summary; `findings` is the full signal \
         list; `risk_score`/`risk_level`/`verdict` are the calibrated report-level judgment. {}",
        weighing_guidance(is_document)
    ));
    v.to_string()
}

/// The full structured forensic report as JSON - the shared payload of both the
/// on-demand tool result and the always-on `/v1/responses` output item. It
/// carries every field the manager's `forensic_reports` + child tables persist
/// (the risk layer, key findings with region + collapse count, and the
/// per-category explanation with finding codes), so a persister loses nothing.
/// Surface-specific framing (the tool's `guidance`) is added by the caller.
pub fn report_value(
    meta: &ForensicMeta,
    findings: &[paddock_forensics::Finding],
) -> serde_json::Value {
    let round3 = |x: f64| (x * 1000.0).round() / 1000.0;
    let arr: Vec<_> = severity_ordered(findings)
        .into_iter()
        .map(|f| {
            let mut o = serde_json::json!({
                "analyzer": f.analyzer,
                "code": f.code,
                "severity": severity_word(f.severity),
                "confidence": round3(f.confidence),
                "description": f.description,
            });
            if let Some(r) = region_json(&f.region) {
                o["region"] = r;
            }
            o
        })
        .collect();
    // Report-level scoring + dedup: collapse the raw signals into a risk score,
    // verdict, and deduplicated key findings (so a consumer gets a summary, not
    // just a flat list to re-derive).
    let risk = paddock_forensics::risk::score(findings);
    let key: Vec<_> = risk
        .key_findings
        .iter()
        .map(|k| {
            let mut o = serde_json::json!({
                "title": k.title,
                "severity": severity_word(k.severity),
                "confidence": round3(k.confidence),
                "sources": k.sources,
                "description": k.description,
                "count": k.count,
            });
            if let Some(r) = region_json(&k.region) {
                o["region"] = r;
            }
            o
        })
        .collect();

    let mut explanation = serde_json::json!({
        "summary": risk.explanation.summary,
        "categories": risk.explanation.categories.iter().map(|c| serde_json::json!({
            "name": c.name,
            "severity": severity_word(c.max_severity),
            "finding_count": c.finding_count,
            "explanation": c.explanation,
            "finding_codes": c.finding_codes,
        })).collect::<Vec<_>>(),
    });
    if let Some(vr) = &risk.explanation.visual_review {
        explanation["visual_review"] = serde_json::json!(vr);
    }
    if let Some(cc) = &risk.explanation.cross_corroboration {
        explanation["cross_corroboration"] = serde_json::json!(cc);
    }
    if let Some(af) = &risk.explanation.anti_forensics_warning {
        explanation["anti_forensics_warning"] = serde_json::json!(af);
    }

    serde_json::json!({
        "count": findings.len(),
        "content_type": meta.content_type,
        "format": meta.format,
        "width": meta.width,
        "height": meta.height,
        "risk_score": risk.risk_score,
        "risk_level": severity_word(risk.risk_level),
        "verdict": risk.verdict.summary,
        "corroborating_families": risk.verdict.corroborating_stages,
        "key_findings": key,
        "explanation": explanation,
        "findings": arr,
    })
}

/// One analyzed attachment's structured result, produced by the always-on
/// preprocessing pass ([`crate::doc::inject_forensics`]) and surfaced as a
/// `/v1/responses` output item. `image_index` is the 0-based order of the
/// analyzed part among the turn's image/PDF attachments - matching the on-demand
/// tool's `image_index` - so a downstream persister can map it back to the
/// attachment it sent. An item is produced for a CLEAN attachment too (empty
/// `findings`): "analyzed, nothing found" is a real, persistable verdict.
pub struct ForensicItem {
    pub image_index: usize,
    /// `"image"` or `"pdf"`.
    pub kind: &'static str,
    pub meta: ForensicMeta,
    pub findings: Vec<paddock_forensics::Finding>,
}

impl ForensicItem {
    /// The `{ "type": "forensics", ... }` output item for the Responses API.
    /// A dedicated item type (not an `mcp_call`): this is server-produced
    /// always-on output, not something the model chose to call.
    pub fn output_item(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "forensics",
            "image_index": self.image_index,
            "kind": self.kind,
            "report": report_value(&self.meta, &self.findings),
        })
    }
}

/// Findings sorted strongest-first: severity descending, then confidence.
fn severity_ordered(findings: &[paddock_forensics::Finding]) -> Vec<&paddock_forensics::Finding> {
    let mut v: Vec<&paddock_forensics::Finding> = findings.iter().collect();
    v.sort_by(|a, b| {
        b.severity.cmp(&a.severity).then(
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal),
        )
    });
    v
}

/// Human-readable analyzer label for the grouped injection block.
fn analyzer_label(name: &str) -> &str {
    match name {
        "ela" => "Error Level Analysis (ELA)",
        other => other,
    }
}

/// Spatial region as a compact prompt string, or empty when unlocalized.
fn region_str(region: &Option<paddock_forensics::Region>) -> String {
    use paddock_forensics::Region::*;
    match region {
        Some(BoundingBox {
            x,
            y,
            width,
            height,
        }) => format!("  [region ({x},{y}) {width}×{height}px]"),
        Some(Points { points }) => format!("  [{} flagged point(s)]", points.len()),
        Some(Mask { .. }) => "  [per-pixel mask]".to_string(),
        None => String::new(),
    }
}

/// Spatial region as JSON for the structured tool result.
fn region_json(region: &Option<paddock_forensics::Region>) -> Option<serde_json::Value> {
    use paddock_forensics::Region::*;
    match region {
        Some(BoundingBox {
            x,
            y,
            width,
            height,
        }) => Some(serde_json::json!({
            "type": "bounding_box", "x": x, "y": y, "width": width, "height": height
        })),
        Some(Points { points }) => {
            Some(serde_json::json!({ "type": "points", "count": points.len() }))
        }
        Some(Mask { width, height, .. }) => {
            Some(serde_json::json!({ "type": "mask", "width": width, "height": height }))
        }
        None => None,
    }
}

fn severity_word(s: paddock_forensics::Severity) -> &'static str {
    use paddock_forensics::Severity::*;
    match s {
        Info => "info",
        Low => "low",
        Medium => "medium",
        High => "high",
        Critical => "critical",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use paddock_forensics::{Finding, Severity};

    fn one_finding() -> Vec<Finding> {
        vec![Finding::new(
            "ela",
            "ela_block_outliers",
            "compression inconsistency across blocks",
            Severity::High,
            0.8,
        )]
    }

    #[test]
    fn injection_carries_verdict_anchoring_and_weighting_for_every_content_type() {
        let meta = ForensicMeta {
            content_type: "photo",
            ..Default::default()
        };
        let s = format_injection(&meta, &one_finding()).expect("findings present");
        // header verdict + subject word + the evidence-weighting substance
        assert!(
            s.contains("Automated image forensics - paddock-forensics"),
            "{s}"
        );
        assert!(
            s.contains("anchor your judgment to it"),
            "verdict anchoring: {s}"
        );
        assert!(s.contains("CONFIRM or CONTRADICT"), "{s}");
        assert!(
            s.contains("likely false positive"),
            "evidence weighting: {s}"
        );
        // a photo must not carry the receipt/arithmetic block
        assert!(
            !s.contains("verify the arithmetic"),
            "no doc block on a photo: {s}"
        );
    }

    #[test]
    fn document_injection_adds_the_arithmetic_and_vat_block() {
        let meta = ForensicMeta {
            content_type: "document",
            ..Default::default()
        };
        let s = format_injection(&meta, &one_finding()).expect("findings present");
        assert!(
            s.contains("Automated document forensics"),
            "document subject: {s}"
        );
        assert!(s.contains("verify the arithmetic"), "arithmetic block: {s}");
        assert!(s.contains("PRINTED tax rate"), "VAT guidance: {s}");
        assert!(
            s.contains("multiple VAT rows"),
            "multi-rate VAT caveat: {s}"
        );
    }

    #[test]
    fn clean_attachment_injects_nothing() {
        let meta = ForensicMeta {
            content_type: "photo",
            ..Default::default()
        };
        assert!(format_injection(&meta, &[]).is_none());
    }

    #[test]
    fn tool_result_guidance_tracks_content_type() {
        let photo = tool_result_json(
            &ForensicMeta {
                content_type: "photo",
                ..Default::default()
            },
            &one_finding(),
        );
        assert!(photo.contains("CONFIRM or CONTRADICT"), "{photo}");
        assert!(
            !photo.contains("verify the arithmetic"),
            "no doc block on a photo tool result"
        );
        let doc = tool_result_json(
            &ForensicMeta {
                content_type: "document",
                ..Default::default()
            },
            &one_finding(),
        );
        assert!(
            doc.contains("verify the arithmetic"),
            "doc tool result carries arithmetic block"
        );
    }

    /// A heavily-flagged attachment must not balloon the ALWAYS-ON note: the raw
    /// per-analyzer list is capped (strongest kept, tail summarized) so an image
    /// that trips dozens of block-level signals can't push the injection to tens
    /// of thousands of tokens. The on-demand tool result stays COMPLETE - that is
    /// where the full list belongs. Guards the ~68k-token injection blow-up seen
    /// live on a critical/tampered image.
    #[test]
    fn injection_caps_the_raw_signal_list_while_the_tool_result_keeps_all() {
        let findings: Vec<Finding> = (0..50)
            .map(|i| Finding {
                analyzer: "ela".into(),
                code: format!("ela_{i}"),
                description: "block-level compression outlier".into(),
                severity: Severity::High,
                confidence: 0.99 - (i as f64) * 0.01, // strictly decreasing -> stable order
                region: None,
            })
            .collect();
        let meta = ForensicMeta {
            content_type: "photo",
            ..Default::default()
        };

        let s = format_injection(&meta, &findings).expect("findings present");
        // strongest few kept (PER_ANALYZER_CAP = 6), long tail summarized (50-6=44)
        assert!(
            s.contains("«ela_0»") && s.contains("«ela_5»"),
            "strongest signals kept: {s}"
        );
        assert!(
            !s.contains("«ela_6»"),
            "past the cap is dropped from the always-on note"
        );
        assert!(
            !s.contains("«ela_49»"),
            "tail signal dropped from the always-on note"
        );
        assert!(
            s.contains("and 44 more lower-severity"),
            "per-analyzer remainder noted: {s}"
        );
        assert!(
            s.contains("omitted from this summary"),
            "global omission note present: {s}"
        );
        // stays small even for 50 signals (the whole point)
        assert!(
            s.len() < 4000,
            "injection stays bounded, got {} bytes",
            s.len()
        );

        // the on-demand tool result is UNCAPPED - every signal survives there.
        let tr = tool_result_json(&meta, &findings);
        assert!(
            tr.contains("ela_49"),
            "tool result keeps the full per-signal list"
        );
        let v: serde_json::Value = serde_json::from_str(&tr).unwrap();
        assert_eq!(
            v["findings"].as_array().unwrap().len(),
            50,
            "all 50 signals present in the tool result"
        );
    }
}
