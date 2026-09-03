//! Text-baseline alignment analysis (document-specific), ported verbatim from
//! the CPU reference. CPU-only (horizontal projection + robust spacing
//! stats; no GPU kernel), `gpu()` delegates.
//!
//! In typeset documents, text lines sit on an even baseline grid. Pasted text
//! often lands at subtly wrong vertical positions, breaking the regular
//! spacing. We build a horizontal projection profile (dark pixels per row),
//! find line peaks, and flag spacings that deviate from the median grid by a
//! robust (MAD) z-score. Only runs on Document/Mixed content.

use crate::analyzer::Analyzer;
use crate::context::ContentType;
use crate::{Context, Finding, Severity};

pub struct TextAlignmentAnalyzer {
    /// Dark-pixel threshold for text detection.
    dark_threshold: u8,
    /// Minimum peak height as a fraction of line width.
    min_peak_fraction: f64,
}

impl Default for TextAlignmentAnalyzer {
    fn default() -> Self {
        Self {
            dark_threshold: 100,
            min_peak_fraction: 0.03,
        }
    }
}

impl Analyzer for TextAlignmentAnalyzer {
    fn name(&self) -> &'static str {
        "text_alignment"
    }

    fn applies_to(&self, ctx: &Context) -> bool {
        // Document-only. A PDF classifies Document but carries no decoded
        // pixels, so it falls out on the dimension guard in `cpu`.
        matches!(ctx.content_type, ContentType::Document | ContentType::Mixed)
    }

    fn cpu(&self, ctx: &Context) -> Vec<Finding> {
        let (width, height) = (ctx.width as usize, ctx.height as usize);
        if width < 100 || height < 100 {
            return Vec::new();
        }

        let gray = ctx.gray();
        self.analyze_line_spacing(gray, width, height)
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

impl TextAlignmentAnalyzer {
    fn analyze_line_spacing(&self, gray: &[u8], width: usize, height: usize) -> Vec<Finding> {
        // Horizontal projection profile: dark pixels per row.
        let mut projection: Vec<u32> = vec![0; height];
        for y in 0..height {
            for x in 0..width {
                if gray[y * width + x] < self.dark_threshold {
                    projection[y] += 1;
                }
            }
        }

        let min_peak = (width as f64 * self.min_peak_fraction) as u32;

        // Text-line positions = peaks in the projection.
        let mut line_positions: Vec<usize> = Vec::new();
        let mut in_peak = false;
        let mut peak_max_val = 0u32;
        let mut peak_max_pos = 0;

        for (y, &proj_val) in projection.iter().enumerate() {
            if proj_val >= min_peak {
                if !in_peak {
                    in_peak = true;
                    peak_max_val = proj_val;
                    peak_max_pos = y;
                } else if proj_val > peak_max_val {
                    peak_max_val = proj_val;
                    peak_max_pos = y;
                }
            } else if in_peak {
                in_peak = false;
                line_positions.push(peak_max_pos);
            }
        }

        if line_positions.len() < 5 {
            return Vec::new(); // Not enough text lines.
        }

        let spacings: Vec<f64> = line_positions
            .windows(2)
            .map(|w| (w[1] - w[0]) as f64)
            .collect();

        if spacings.len() < 4 {
            return Vec::new();
        }

        let mut sorted_spacings = spacings.clone();
        sorted_spacings.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median_spacing = sorted_spacings[sorted_spacings.len() / 2];

        if median_spacing < 5.0 {
            return Vec::new(); // Too small to be meaningful.
        }

        let mad = {
            let mut devs: Vec<f64> = spacings
                .iter()
                .map(|s| (s - median_spacing).abs())
                .collect();
            devs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            devs[devs.len() / 2] * 1.4826
        };

        if mad < 0.5 {
            return Vec::new(); // Very regular spacing - no anomaly.
        }

        let mut findings = Vec::new();

        let mut anomaly_positions: Vec<(usize, f64)> = Vec::new();
        for (i, &spacing) in spacings.iter().enumerate() {
            let z = (spacing - median_spacing).abs() / mad;
            if z > 3.0 {
                anomaly_positions.push((i, z));
            }
        }

        if anomaly_positions.len() >= 2 {
            let max_z = anomaly_positions
                .iter()
                .fold(0.0_f64, |m, &(_, z)| m.max(z));

            findings.push(Finding::new(
                "text_alignment",
                "text_baseline_misalignment",
                format!(
                    "{} text line spacings deviate from document baseline grid \
                     (median spacing {:.0}px, z-score {:.1}) - possible pasted \
                     text lines from different source",
                    anomaly_positions.len(),
                    median_spacing,
                    max_z,
                ),
                if max_z > 5.0 {
                    Severity::High
                } else {
                    Severity::Medium
                },
                (0.45 + (max_z - 3.0) * 0.08).min(0.80),
            ));
        }

        findings
    }
}
