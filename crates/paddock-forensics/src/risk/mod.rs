//! Risk scoring + report-level dedup for a forensic [`Report`](crate::Report).
//!
//! This is a **paddock-native** scorer derived from the reference scorer's
//! weighting, deliberately not a verbatim port: that scorer is coupled to
//! its VLM pipeline (stage-1/stage-2 `vlm_*` findings, `manipulation_probability`
//! blending, `JobId`/`AnalysisReport`). paddock-forensics is signal extraction
//! that *feeds* a model - there is no VLM stage here - so all of that is dropped.
//! What is kept is the useful core:
//!
//! - **weighted aggregation**: `severity_weight × confidence × stage_weight`,
//!   summed with a cross-corroboration bonus (independent analyzer families
//!   agreeing matters more than one family alone) and diminishing-returns
//!   normalization to 0..100;
//! - **report-level dedup** into [`KeyFinding`]s: raw findings are collapsed into
//!   semantic categories (so e.g. the several JPEG/quantization findings, or a
//!   copy_move + its per-cluster duplicates, become one key finding with all the
//!   contributing analyzers listed) - plus an exact-duplicate pass that removes
//!   identical (analyzer, code, description) findings emitted by more than one
//!   analyzer;
//! - a plain-language [`Verdict`].

use serde::{Deserialize, Serialize};

use crate::{Finding, Region, Severity};

pub mod explanation;
pub use explanation::{ExplanationCategory, ForensicExplanation};

/// Coarse analyzer family, used for stage weighting + cross-corroboration
/// (independent families agreeing is a stronger signal than one family alone).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Stage {
    Metadata,
    Pixel,
    Signal,
    AiDetect,
    Pdf,
}

impl Stage {
    fn weight(self) -> f64 {
        match self {
            Stage::Metadata => 0.8,
            Stage::Pixel => 1.2,
            Stage::Signal => 1.1,
            Stage::AiDetect => 1.3,
            Stage::Pdf => 1.0,
        }
    }
}

/// Map an analyzer name to its family.
fn stage_of(analyzer: &str) -> Stage {
    match analyzer {
        "metadata" | "exif_pixel" | "c2pa" => Stage::Metadata,
        "frequency" => Stage::AiDetect,
        "chromatic_aberration" | "illumination" | "prnu" | "prnu_cross_region" => Stage::Signal,
        a if a.starts_with("pdf") => Stage::Pdf,
        _ => Stage::Pixel,
    }
}

fn severity_weight(sev: Severity) -> f64 {
    match sev {
        Severity::Info => 0.0,
        Severity::Low => 10.0,
        Severity::Medium => 25.0,
        Severity::High => 50.0,
        Severity::Critical => 80.0,
    }
}

/// A deduplicated, human-facing finding aggregating one semantic category.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KeyFinding {
    pub title: String,
    pub description: String,
    pub severity: Severity,
    pub confidence: f64,
    /// Distinct analyzers that contributed to this category.
    pub sources: Vec<String>,
    /// The most specific (smallest) region among the contributing findings.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub region: Option<Region>,
    /// How many raw findings were collapsed into this key finding.
    pub count: usize,
}

/// The overall judgment for a report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Verdict {
    /// 0..100.
    pub risk_score: f64,
    pub risk_level: Severity,
    pub summary: String,
    /// Max confidence among key findings.
    pub confidence: f64,
    /// Number of independent analyzer families that flagged something material.
    pub corroborating_stages: u32,
}

/// The scored, deduplicated view of a forensic report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RiskReport {
    pub risk_score: f64,
    pub risk_level: Severity,
    pub verdict: Verdict,
    pub key_findings: Vec<KeyFinding>,
    /// Plain-language, template-based explanation (categories + narrative).
    pub explanation: ForensicExplanation,
}

struct Category {
    title: &'static str,
    code_patterns: &'static [&'static str],
}

/// Semantic categories, in priority order. Every analyzer this crate ships is
/// covered so its findings collapse into a stable, deduplicated key finding.
const CATEGORIES: &[Category] = &[
    Category {
        title: "Copy-move forgery",
        code_patterns: &["copy_move"],
    },
    Category {
        title: "Splice boundary detection",
        code_patterns: &["splice_boundary"],
    },
    Category {
        title: "Document text manipulation",
        code_patterns: &[
            "document_paste_corner",
            "document_block_anomaly",
            "font_stroke",
            "text_baseline",
            "paste_rectangle",
        ],
    },
    Category {
        title: "Noise inconsistency",
        code_patterns: &[
            "noise_inconsistency",
            "unnaturally_low_noise",
            "noise_distribution",
        ],
    },
    Category {
        title: "Compression history anomaly",
        code_patterns: &[
            "double_jpeg",
            "jpeg_ghost",
            "benford",
            "bimodal_quantization",
            "quantization_inconsistency",
            "qtable",
            "artifact_free",
            "ela_",
        ],
    },
    Category {
        title: "Texture/color anomaly",
        code_patterns: &["texture_", "color_histogram", "channel_correlation"],
    },
    Category {
        title: "Edge/resampling anomaly",
        code_patterns: &["edge_sharpness", "resampling", "upscaling"],
    },
    Category {
        title: "Shadow/lighting inconsistency",
        code_patterns: &["shadow_", "lighting_", "illumination_"],
    },
    Category {
        title: "Sensor/optics anomaly",
        code_patterns: &[
            "prnu",
            "cfa_",
            "no_cfa",
            "chromatic",
            "no_chromatic",
            "ca_model",
        ],
    },
    Category {
        title: "Frequency / AI-generation signature",
        code_patterns: &[
            "spectral_",
            "flat_spectrum",
            "gan_",
            "mid_freq",
            "low_high_freq",
        ],
    },
    Category {
        title: "Geometric inconsistency",
        code_patterns: &["geometric_", "vanishing_point"],
    },
    Category {
        title: "Depth-of-field inconsistency",
        code_patterns: &["dof_"],
    },
    Category {
        title: "Screenshot / synthetic capture",
        code_patterns: &["screenshot_"],
    },
    Category {
        title: "Histogram manipulation",
        code_patterns: &["histogram_"],
    },
    Category {
        title: "Anti-forensics detected",
        code_patterns: &["anti_forensics"],
    },
    Category {
        title: "Metadata / provenance anomaly",
        code_patterns: &[
            "ai_generator_tag",
            "editing_software_tag",
            "no_camera_metadata",
            "timestamp_inconsistency",
            "resolution_mismatch",
            "orientation_inconsistency",
            "color_space_clipping",
            "bit_depth_inconsistency",
            "c2pa_",
        ],
    },
    Category {
        title: "PDF structure / overlay anomaly",
        code_patterns: &["pdf_"],
    },
];

/// Remove redundant findings that carry the same message - identical
/// (code, description, severity) - even when emitted by different analyzers
/// (e.g. a shared jpeg-quality estimate surfaced by several JPEG analyzers).
/// Keeps the first occurrence and the max confidence seen; the analyzer that
/// remains is the first to have reported it.
fn dedup_exact(findings: &[Finding]) -> Vec<Finding> {
    let mut out: Vec<Finding> = Vec::with_capacity(findings.len());
    for f in findings {
        if let Some(existing) = out.iter_mut().find(|e| {
            e.code == f.code && e.description == f.description && e.severity == f.severity
        }) {
            if f.confidence > existing.confidence {
                existing.confidence = f.confidence;
            }
            continue;
        }
        out.push(f.clone());
    }
    out
}

/// Score + dedup a report's findings into a [`RiskReport`].
pub fn score(findings: &[Finding]) -> RiskReport {
    let findings = dedup_exact(findings);
    let risk_score = compute_score(&findings);
    let risk_level = score_to_level(risk_score);
    let key_findings = build_key_findings(&findings);
    let verdict = build_verdict(&findings, &key_findings, risk_score, risk_level);
    let explanation = explanation::explain(&findings, risk_score, risk_level);
    RiskReport {
        risk_score,
        risk_level,
        verdict,
        key_findings,
        explanation,
    }
}

fn compute_score(findings: &[Finding]) -> f64 {
    if findings.is_empty() {
        return 0.0;
    }

    let mut weighted_sum = 0.0;
    for f in findings {
        weighted_sum += severity_weight(f.severity) * f.confidence * stage_of(&f.analyzer).weight();
    }

    let bonus = cross_corroboration_bonus(findings);
    weighted_sum *= 1.0 + bonus;

    // Diminishing-returns normalization to 0..100.
    let normalized = 100.0 * (1.0 - (-weighted_sum / 80.0).exp());
    (normalized * 10.0).round() / 10.0
}

fn cross_corroboration_bonus(findings: &[Finding]) -> f64 {
    let mut families = std::collections::HashSet::new();
    for f in findings {
        if f.severity >= Severity::Medium && f.confidence > 0.5 {
            families.insert(stage_of(&f.analyzer));
        }
    }
    match families.len() {
        0..=1 => 0.0,
        2 => 0.15,
        3 => 0.30,
        4 => 0.45,
        _ => 0.60,
    }
}

fn score_to_level(score: f64) -> Severity {
    match score {
        s if s >= 80.0 => Severity::Critical,
        s if s >= 60.0 => Severity::High,
        s if s >= 35.0 => Severity::Medium,
        s if s >= 15.0 => Severity::Low,
        _ => Severity::Info,
    }
}

fn build_key_findings(findings: &[Finding]) -> Vec<KeyFinding> {
    let mut keys: Vec<KeyFinding> = Vec::new();

    for cat in CATEGORIES {
        let matching: Vec<&Finding> = findings
            .iter()
            .filter(|f| {
                cat.code_patterns.iter().any(|p| f.code.contains(p))
                    && f.severity >= Severity::Medium
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
        let max_confidence = matching
            .iter()
            .map(|f| f.confidence)
            .fold(0.0_f64, f64::max);

        let Some(best) = matching.iter().max_by(|a, b| {
            a.confidence
                .partial_cmp(&b.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        }) else {
            continue;
        };

        let mut sources: Vec<String> = matching
            .iter()
            .map(|f| f.analyzer.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        sources.sort();

        // Smallest bounding box among contributors = most specific localization.
        let region = matching
            .iter()
            .filter_map(|f| f.region.as_ref())
            .filter(|r| matches!(r, Region::BoundingBox { .. }))
            .min_by_key(|r| match r {
                Region::BoundingBox { width, height, .. } => (*width as u64) * (*height as u64),
                _ => u64::MAX,
            })
            .cloned();

        let description = best.description.chars().take(300).collect::<String>();

        let count = matching.len();
        let title = if count > 1 {
            format!("{} ({} signals)", cat.title, count)
        } else {
            cat.title.to_string()
        };

        keys.push(KeyFinding {
            title,
            description,
            severity: max_severity,
            confidence: max_confidence,
            sources,
            region,
            count,
        });
    }

    keys.sort_by(|a, b| {
        b.severity.cmp(&a.severity).then(
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal),
        )
    });

    keys
}

fn build_verdict(
    findings: &[Finding],
    key_findings: &[KeyFinding],
    risk_score: f64,
    risk_level: Severity,
) -> Verdict {
    let corroborating_stages = {
        let mut families = std::collections::HashSet::new();
        for f in findings {
            if f.severity >= Severity::Medium && f.confidence > 0.5 {
                families.insert(stage_of(&f.analyzer));
            }
        }
        families.len() as u32
    };

    let max_confidence = key_findings
        .iter()
        .map(|k| k.confidence)
        .fold(0.0_f64, f64::max);

    let summary = if risk_score < 15.0 {
        "No significant forensic anomalies detected. Image appears authentic.".to_string()
    } else if risk_score < 35.0 {
        "Minor anomalies detected, likely from normal image processing (compression, resizing)."
            .to_string()
    } else if risk_score < 60.0 {
        format!(
            "Moderate forensic anomalies detected across {corroborating_stages} analyzer \
             family(ies). Manual review recommended."
        )
    } else {
        let top_issue = key_findings
            .first()
            .map(|k| k.title.as_str())
            .unwrap_or("manipulation");
        format!(
            "Strong evidence of image manipulation detected. {top_issue} corroborated by \
             {corroborating_stages} independent analyzer family(ies)."
        )
    };

    Verdict {
        risk_score,
        risk_level,
        summary,
        confidence: max_confidence,
        corroborating_stages,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f(analyzer: &'static str, code: &str, sev: Severity, conf: f64) -> Finding {
        Finding::new(analyzer, code, "desc", sev, conf)
    }

    #[test]
    fn empty_is_zero_risk() {
        let r = score(&[]);
        assert_eq!(r.risk_score, 0.0);
        assert_eq!(r.risk_level, Severity::Info);
        assert!(r.key_findings.is_empty());
    }

    #[test]
    fn exact_duplicates_collapse() {
        // Same code+description from two analyzers -> one raw finding, and the
        // categories still collapse to a single key finding.
        let dups = vec![
            f(
                "jpeg_forensics",
                "jpeg_quality_estimate",
                Severity::Medium,
                0.5,
            ),
            f(
                "double_jpeg",
                "jpeg_quality_estimate",
                Severity::Medium,
                0.7,
            ),
        ];
        let deduped = dedup_exact(&dups);
        assert_eq!(deduped.len(), 1, "identical code+description collapses");
        assert_eq!(deduped[0].confidence, 0.7, "keeps max confidence");
    }

    #[test]
    fn corroboration_and_categories() {
        // Three distinct families with material findings -> corroboration bonus,
        // and each maps to its own key-finding category.
        let findings = vec![
            f("copy_move", "copy_move_detected", Severity::Critical, 0.9),
            f("prnu", "prnu_inconsistency", Severity::High, 0.8),
            f("metadata", "ai_generator_tag", Severity::Critical, 0.95),
        ];
        let r = score(&findings);
        assert!(
            r.risk_score > 60.0,
            "strong multi-family evidence, got {}",
            r.risk_score
        );
        assert_eq!(r.verdict.corroborating_stages, 3);
        assert_eq!(r.risk_level, Severity::Critical);
        // Distinct categories, highest severity first.
        assert!(r.key_findings.len() >= 3);
        assert_eq!(r.key_findings[0].severity, Severity::Critical);
    }

    #[test]
    fn info_findings_do_not_raise_risk() {
        let findings = vec![f("c2pa", "c2pa_manifest_present", Severity::Info, 0.95)];
        let r = score(&findings);
        assert_eq!(
            r.risk_score, 0.0,
            "info findings carry zero severity weight"
        );
        assert!(
            r.key_findings.is_empty(),
            "info findings are below the Medium key-finding gate"
        );
    }
}
