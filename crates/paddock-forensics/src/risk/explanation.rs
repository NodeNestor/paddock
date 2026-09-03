//! Human-readable forensic explanation, ported verbatim from the reference
//! implementation. Fully TEMPLATE-based (no VLM) - it maps finding codes
//! to forensic categories with static explanatory prose, builds a
//! cross-corroboration narrative when independent analyzer families agree, and
//! surfaces an anti-forensics warning. The reference's `f.source` (AnalysisStage) is
//! replaced by `stage_of(&f.analyzer)` (paddock has no per-finding stage field).

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::{Finding, Severity};

use super::stage_of;

/// One category's contribution to the explanation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExplanationCategory {
    pub name: String,
    pub finding_count: usize,
    pub max_severity: Severity,
    pub explanation: String,
    pub finding_codes: Vec<String>,
}

/// A plain-language forensic explanation of a report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ForensicExplanation {
    pub summary: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub visual_review: Option<String>,
    pub categories: Vec<ExplanationCategory>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub cross_corroboration: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub anti_forensics_warning: Option<String>,
}

struct CategoryDef {
    name: &'static str,
    code_prefixes: &'static [&'static str],
    explanation_template: &'static str,
}

const CATEGORIES: &[CategoryDef] = &[
    CategoryDef {
        name: "Compression Artifacts",
        code_prefixes: &["ela_", "jpeg_", "double_jpeg_", "qtable_"],
        explanation_template: "JPEG compression analysis reveals inconsistencies in how \
            the image was saved. Different regions show different compression histories, \
            which occurs when parts of the image were edited or spliced from other sources.",
    },
    CategoryDef {
        name: "Noise Inconsistency",
        code_prefixes: &["noise_", "prnu_"],
        explanation_template: "The image's sensor noise pattern is inconsistent across \
            regions. Each camera sensor produces a unique noise fingerprint - regions \
            with different noise characteristics likely originated from a different camera \
            or were digitally altered.",
    },
    CategoryDef {
        name: "Pixel Manipulation",
        code_prefixes: &["copy_move_", "splice_", "resampling_", "document_"],
        explanation_template: "Direct evidence of pixel-level manipulation was detected. \
            This includes copied/moved regions, splice boundaries, or signs of content \
            being resized, rotated, or pasted into the image.",
    },
    CategoryDef {
        name: "Texture & Color Anomaly",
        code_prefixes: &["texture_", "color_", "channel_", "histogram_"],
        explanation_template: "The image's texture patterns or color distributions are \
            inconsistent between regions. This suggests content was composited from \
            images with different cameras, lighting, or post-processing.",
    },
    CategoryDef {
        name: "Edge & Sharpness Anomaly",
        code_prefixes: &["edge_sharpness_", "dof_"],
        explanation_template: "Edge characteristics or blur patterns are inconsistent \
            with natural lens optics. Cut-paste operations leave unnaturally sharp \
            boundaries, while feathered blends create suspicious blur patterns.",
    },
    CategoryDef {
        name: "Geometric Inconsistency",
        code_prefixes: &["geometric_", "vanishing_point_"],
        explanation_template: "The image's geometric structure - line continuity and \
            perspective geometry - is inconsistent. This suggests content was \
            composited from scenes with different viewpoints or camera positions.",
    },
    CategoryDef {
        name: "Lighting & Shadow Inconsistency",
        code_prefixes: &["shadow_", "lighting_", "illumination_", "chromatic_"],
        explanation_template: "The lighting direction or shadow patterns are inconsistent \
            across the image. In a genuine photo, all shadows are cast by the same light \
            source(s). Conflicting shadow directions strongly indicate compositing.",
    },
    CategoryDef {
        name: "Frequency & Wavelet Anomaly",
        code_prefixes: &["frequency_", "wavelet_", "cfa_"],
        explanation_template: "The image's frequency content or sensor interpolation \
            patterns are inconsistent. This can indicate AI-generated content, \
            inpainting, or splicing from images captured by different cameras.",
    },
    CategoryDef {
        name: "Anti-Forensics Detected",
        code_prefixes: &["anti_forensics_"],
        explanation_template: "Counter-forensic techniques were detected - deliberate \
            attempts to hide manipulation traces. This is a strong indicator of \
            intentional fraud, as legitimate image editing rarely involves \
            anti-forensic processing.",
    },
    CategoryDef {
        name: "Source Anomaly",
        code_prefixes: &["screenshot_", "upscaling_"],
        explanation_template: "The image appears to be a screenshot, re-photographed \
            image, or upscaled from lower resolution. This is inconsistent with a \
            claimed original photograph.",
    },
    CategoryDef {
        name: "Document Text Anomaly",
        code_prefixes: &["font_", "text_"],
        explanation_template: "Text formatting inconsistencies were detected in the \
            document - different font characteristics or misaligned baselines suggest \
            text was pasted or edited from a different source.",
    },
    CategoryDef {
        name: "Metadata Anomaly",
        code_prefixes: &["exif_", "metadata_", "c2pa_", "thumbnail_", "exif_pixel_"],
        explanation_template: "Image metadata (EXIF, timestamps, device information) \
            shows inconsistencies or evidence of editing software. While metadata \
            alone is not conclusive, it provides context for other findings.",
    },
];

/// Generate a template-based forensic explanation from findings + score.
pub fn explain(findings: &[Finding], risk_score: f64, risk_level: Severity) -> ForensicExplanation {
    if findings.is_empty() {
        return ForensicExplanation {
            summary: "No forensic anomalies detected. The image appears authentic \
                based on all analyzed signals."
                .into(),
            visual_review: None,
            categories: Vec::new(),
            cross_corroboration: None,
            anti_forensics_warning: None,
        };
    }

    let categories = categorize(findings);
    let cross_corroboration = build_cross_corroboration(findings, &categories);
    let anti_forensics_warning = check_anti_forensics(findings);
    let summary = build_summary(
        findings,
        &categories,
        risk_score,
        risk_level,
        cross_corroboration.is_some(),
    );

    ForensicExplanation {
        summary,
        visual_review: None,
        categories,
        cross_corroboration,
        anti_forensics_warning,
    }
}

fn categorize(findings: &[Finding]) -> Vec<ExplanationCategory> {
    let mut categories = Vec::new();

    for cat_def in CATEGORIES {
        let matching: Vec<&Finding> = findings
            .iter()
            .filter(|f| {
                cat_def
                    .code_prefixes
                    .iter()
                    .any(|prefix| f.code.starts_with(prefix))
            })
            .collect();

        if matching.is_empty() {
            continue;
        }

        let max_severity = matching
            .iter()
            .map(|f| f.severity)
            .max()
            .unwrap_or(Severity::Info);

        // Only include categories with Low+ findings.
        if max_severity < Severity::Low {
            continue;
        }

        let finding_codes: Vec<String> = matching.iter().map(|f| f.code.clone()).collect();

        categories.push(ExplanationCategory {
            name: cat_def.name.into(),
            finding_count: matching.len(),
            max_severity,
            explanation: cat_def.explanation_template.into(),
            finding_codes,
        });
    }

    categories.sort_by(|a, b| b.max_severity.cmp(&a.max_severity));
    categories
}

fn build_cross_corroboration(
    findings: &[Finding],
    categories: &[ExplanationCategory],
) -> Option<String> {
    // Count independent analyzer families flagging material findings.
    let mut stages_with_findings: HashMap<super::Stage, usize> = HashMap::new();
    for f in findings {
        if f.severity >= Severity::Medium && f.confidence > 0.5 {
            *stages_with_findings
                .entry(stage_of(&f.analyzer))
                .or_insert(0) += 1;
        }
    }

    let active_stages = stages_with_findings.len();
    if active_stages < 2 {
        return None;
    }

    let high_severity_categories: Vec<&str> = categories
        .iter()
        .filter(|c| c.max_severity >= Severity::Medium)
        .map(|c| c.name.as_str())
        .collect();

    if high_severity_categories.len() < 2 {
        return None;
    }

    let cat_list = if high_severity_categories.len() <= 3 {
        high_severity_categories.join(" and ")
    } else {
        let (first, last) = high_severity_categories.split_at(high_severity_categories.len() - 1);
        format!("{}, and {}", first.join(", "), last[0])
    };

    Some(format!(
        "Multiple independent forensic signals corroborate manipulation: \
         {cat_list}. When {} different analysis categories independently flag \
         the same image, the probability of a false positive is extremely low. \
         Each signal alone might have an innocent explanation, but their \
         convergence strongly indicates intentional manipulation.",
        high_severity_categories.len(),
    ))
}

fn check_anti_forensics(findings: &[Finding]) -> Option<String> {
    let af_findings: Vec<&Finding> = findings
        .iter()
        .filter(|f| f.code.starts_with("anti_forensics_"))
        .collect();

    if af_findings.is_empty() {
        return None;
    }

    let techniques: Vec<&str> = af_findings
        .iter()
        .map(|f| {
            if f.code.contains("median_filter") {
                "median filtering (smooths JPEG artifacts)"
            } else if f.code.contains("noise_injection") {
                "noise injection (masks noise inconsistencies)"
            } else if f.code.contains("histogram_equalization") {
                "histogram equalization (flattens statistical anomalies)"
            } else {
                "unknown counter-forensic technique"
            }
        })
        .collect();

    Some(format!(
        "WARNING: Counter-forensic techniques detected - {}. \
         This indicates a sophisticated actor who is aware of forensic \
         detection methods and has deliberately attempted to conceal \
         manipulation. The presence of anti-forensics is itself strong \
         evidence of intentional fraud.",
        techniques.join("; "),
    ))
}

fn build_summary(
    findings: &[Finding],
    categories: &[ExplanationCategory],
    risk_score: f64,
    risk_level: Severity,
    has_corroboration: bool,
) -> String {
    let total_findings = findings.len();
    let high_findings = findings
        .iter()
        .filter(|f| f.severity >= Severity::High)
        .count();

    let level_word = match risk_level {
        Severity::Info => "clean",
        Severity::Low => "minor",
        Severity::Medium => "moderate",
        Severity::High => "significant",
        Severity::Critical => "critical",
    };

    let active_categories = categories.len();

    if risk_level <= Severity::Low {
        format!(
            "Analysis of {total_findings} forensic signals shows {level_word} \
             anomalies (risk score {risk_score:.0}/100). No strong evidence \
             of manipulation was found."
        )
    } else if has_corroboration {
        format!(
            "Analysis detected {level_word} manipulation evidence \
             (risk score {risk_score:.0}/100): {high_findings} high-severity \
             findings across {active_categories} independent forensic categories. \
             Multiple independent signals corroborate the detection."
        )
    } else {
        format!(
            "Analysis detected {level_word} anomalies \
             (risk score {risk_score:.0}/100): {total_findings} findings \
             with {high_findings} at high severity."
        )
    }
}
