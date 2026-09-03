//! Font-consistency analysis (document-specific), ported verbatim from
//! the CPU reference. CPU-only (per-block stroke-width run lengths + MAD
//! outliers; no GPU kernel), `gpu()` delegates.
//!
//! Pasted text usually carries a different stroke width (bold vs regular, a
//! different font family) than the surrounding document. We estimate mean
//! stroke width per text block from horizontal dark-run lengths and flag blocks
//! that deviate from the document median by a robust (MAD) z-score. Only runs
//! on Document/Mixed content.

use crate::analyzer::Analyzer;
use crate::context::ContentType;
use crate::{Context, Finding, Region, Severity};

pub struct FontConsistencyAnalyzer {
    /// Block size for stroke-width analysis.
    block_size: usize,
    /// MAD z-score threshold.
    anomaly_z_threshold: f64,
}

impl Default for FontConsistencyAnalyzer {
    fn default() -> Self {
        Self {
            block_size: 32,
            anomaly_z_threshold: 3.5,
        }
    }
}

impl Analyzer for FontConsistencyAnalyzer {
    fn name(&self) -> &'static str {
        "font_consistency"
    }

    fn applies_to(&self, ctx: &Context) -> bool {
        matches!(ctx.content_type, ContentType::Document | ContentType::Mixed)
    }

    fn cpu(&self, ctx: &Context) -> Vec<Finding> {
        let (width, height) = (ctx.width as usize, ctx.height as usize);
        if width < self.block_size * 4 || height < self.block_size * 4 {
            return Vec::new();
        }

        let gray = ctx.gray();
        self.analyze_stroke_width(gray, width, height)
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

impl FontConsistencyAnalyzer {
    fn analyze_stroke_width(&self, gray: &[u8], width: usize, height: usize) -> Vec<Finding> {
        let bs = self.block_size;
        let blocks_x = width / bs;
        let blocks_y = height / bs;

        // Per-block mean stroke width, estimated from horizontal dark runs.
        let mut block_strokes: Vec<(usize, f64)> = Vec::new();

        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                let x0 = bx * bs;
                let y0 = by * bs;
                let block_idx = by * blocks_x + bx;

                // Does this block hold text (dark on light, bimodal)?
                let mut dark_count = 0_usize;
                let mut light_count = 0_usize;

                for dy in 0..bs {
                    for dx in 0..bs {
                        let x = x0 + dx;
                        let y = y0 + dy;
                        if x >= width || y >= height {
                            continue;
                        }

                        let v = gray[y * width + x];
                        if v < 80 {
                            dark_count += 1;
                        } else if v > 180 {
                            light_count += 1;
                        }
                    }
                }

                let total = bs * bs;
                let dark_ratio = dark_count as f64 / total as f64;
                let light_ratio = light_count as f64 / total as f64;

                if !(0.05..=0.60).contains(&dark_ratio) || light_ratio < 0.30 {
                    continue;
                }

                let mut run_lengths: Vec<usize> = Vec::new();

                for dy in 0..bs {
                    let y = y0 + dy;
                    if y >= height {
                        continue;
                    }

                    let mut run = 0;
                    for dx in 0..bs {
                        let x = x0 + dx;
                        if x >= width {
                            continue;
                        }

                        if gray[y * width + x] < 100 {
                            run += 1;
                        } else if run > 0 {
                            if (2..=20).contains(&run) {
                                run_lengths.push(run);
                            }
                            run = 0;
                        }
                    }
                }

                if run_lengths.len() >= 5 {
                    let mean_stroke: f64 =
                        run_lengths.iter().sum::<usize>() as f64 / run_lengths.len() as f64;
                    block_strokes.push((block_idx, mean_stroke));
                }
            }
        }

        let mut findings = Vec::new();

        if block_strokes.len() < 6 {
            return findings;
        }

        let values: Vec<f64> = block_strokes.iter().map(|&(_, s)| s).collect();
        let mut sorted = values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = sorted[sorted.len() / 2];
        let mad = {
            let mut devs: Vec<f64> = values.iter().map(|v| (v - median).abs()).collect();
            devs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            devs[devs.len() / 2] * 1.4826
        };

        if mad < 0.3 {
            return findings; // Very consistent strokes.
        }

        let anomalous: Vec<(usize, f64)> = block_strokes
            .iter()
            .filter_map(|&(idx, stroke)| {
                let z = (stroke - median).abs() / mad;
                if z > self.anomaly_z_threshold {
                    Some((idx, z))
                } else {
                    None
                }
            })
            .collect();

        if anomalous.len() >= 2 {
            let max_z = anomalous.iter().fold(0.0_f64, |m, &(_, z)| m.max(z));
            let strongest = anomalous
                .iter()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap();
            let sx = (strongest.0 % blocks_x) * bs;
            let sy = (strongest.0 / blocks_x) * bs;

            findings.push(
                Finding::new(
                    "font_consistency",
                    "font_stroke_width_inconsistency",
                    format!(
                        "{} text blocks show different stroke widths from document median \
                         ({median:.1}px, z-score {max_z:.1}) - possible mixed fonts or pasted \
                         text from different source",
                        anomalous.len(),
                    ),
                    if max_z > 5.0 {
                        Severity::High
                    } else {
                        Severity::Medium
                    },
                    (0.45 + (max_z - self.anomaly_z_threshold) * 0.08).min(0.80),
                )
                .with_region(Region::BoundingBox {
                    x: sx as u32,
                    y: sy as u32,
                    width: bs as u32,
                    height: bs as u32,
                }),
            );
        }

        findings
    }
}
