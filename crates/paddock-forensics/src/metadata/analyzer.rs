//! Image-metadata forensics (software tags, metadata presence, timestamp
//! consistency) via sift, ported verbatim from the reference. CPU-only.

use crate::analyzer::Analyzer;
use crate::{Context, Finding, Severity};

pub struct MetadataAnalyzer;

impl Analyzer for MetadataAnalyzer {
    fn name(&self) -> &'static str {
        "metadata"
    }

    fn cpu(&self, ctx: &Context) -> Vec<Finding> {
        let mut findings = Vec::new();
        Self::check_software_tags(&ctx.tags, &mut findings);
        Self::check_metadata_presence(&ctx.tags, &mut findings);
        Self::check_timestamp_consistency(&ctx.tags, &mut findings);
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

impl MetadataAnalyzer {
    fn check_software_tags(tags: &[sift::Tag], findings: &mut Vec<Finding>) {
        let software_tags: Vec<&sift::Tag> = tags
            .iter()
            .filter(|t| {
                let name = t.name.to_lowercase();
                name.contains("software") || name.contains("creator") || name.contains("tool")
            })
            .collect();

        for tag in &software_tags {
            let value = tag.value.to_lowercase();

            let ai_tools = [
                "midjourney",
                "dall-e",
                "stable diffusion",
                "comfyui",
                "automatic1111",
                "novelai",
                "leonardo",
                "firefly",
                "imagen",
                "flux",
                "ideogram",
            ];
            for tool in &ai_tools {
                if value.contains(tool) {
                    findings.push(Finding::new(
                        "metadata",
                        "ai_generator_tag",
                        format!(
                            "Metadata indicates AI generation tool: {} = {}",
                            tag.name, tag.value
                        ),
                        Severity::Critical,
                        0.95,
                    ));
                }
            }

            let edit_tools = [
                "photoshop",
                "gimp",
                "affinity",
                "pixelmator",
                "lightroom",
                "capture one",
                "paint.net",
                "canva",
                "snapseed",
            ];
            for tool in &edit_tools {
                if value.contains(tool) {
                    findings.push(Finding::new(
                        "metadata",
                        "editing_software_tag",
                        format!(
                            "Image processed with editing software: {} = {}",
                            tag.name, tag.value
                        ),
                        Severity::Medium,
                        0.9,
                    ));
                }
            }
        }
    }

    fn check_metadata_presence(tags: &[sift::Tag], findings: &mut Vec<Finding>) {
        // PDFs never carry camera EXIF.
        let is_pdf = tags.iter().any(|t| t.group == "PDF");
        if is_pdf {
            return;
        }

        let has_camera_make = tags.iter().any(|t| t.name == "Make");
        let has_camera_model = tags.iter().any(|t| t.name == "Model");
        let has_exif_version = tags.iter().any(|t| t.name == "ExifVersion");

        if !has_camera_make && !has_camera_model && !has_exif_version {
            findings.push(Finding::new(
                "metadata",
                "no_camera_metadata",
                "No camera metadata found - image may have been stripped or generated",
                Severity::Low,
                0.6,
            ));
        }
    }

    fn check_timestamp_consistency(tags: &[sift::Tag], findings: &mut Vec<Finding>) {
        // PDF creation vs modification differences are normal.
        let is_pdf = tags.iter().any(|t| t.group == "PDF");
        if is_pdf {
            return;
        }

        let date_fields: Vec<&sift::Tag> = tags
            .iter()
            .filter(|t| {
                let name = t.name.to_lowercase();
                name.contains("date") || name.contains("time")
            })
            .collect();

        if date_fields.len() >= 2 {
            let values: Vec<&str> = date_fields.iter().map(|t| t.value.as_str()).collect();
            let unique: std::collections::HashSet<&&str> = values.iter().collect();

            if unique.len() > 2 && date_fields.len() >= 3 {
                findings.push(Finding::new(
                    "metadata",
                    "timestamp_inconsistency",
                    format!(
                        "Multiple inconsistent timestamps found across {} date fields",
                        date_fields.len()
                    ),
                    Severity::Medium,
                    0.7,
                ));
            }
        }
    }
}
