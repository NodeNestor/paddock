//! Depth-of-field consistency analysis, ported verbatim from the CPU
//! reference. CPU-only (per-block Sobel sharpness + adjacency jumps; no GPU
//! kernel), `gpu()` delegates.
//!
//! In a real photograph blur varies smoothly with distance from the focal
//! plane. Composites mix sharp regions from different scenes, producing abrupt
//! blur transitions no lens can make. We estimate per-block sharpness (mean
//! gradient magnitude at edges) and flag concentrated abrupt jumps between
//! neighbouring blocks. Camera-specific: skipped for documents (false positives
//! on text).

use crate::analyzer::Analyzer;
use crate::context::ContentType;
use crate::{Context, Finding, Region, Severity};

pub struct DofConsistencyAnalyzer {
    /// Block size for blur estimation.
    block_size: usize,
    /// Minimum edge pixels per block.
    min_edge_pixels: usize,
    /// Gradient threshold for edge detection.
    edge_threshold: f64,
}

impl Default for DofConsistencyAnalyzer {
    fn default() -> Self {
        Self {
            block_size: 64,
            min_edge_pixels: 20,
            edge_threshold: 30.0,
        }
    }
}

impl Analyzer for DofConsistencyAnalyzer {
    fn name(&self) -> &'static str {
        "dof_consistency"
    }

    fn applies_to(&self, ctx: &Context) -> bool {
        // Camera-specific -> skip documents (which includes PDFs).
        !matches!(ctx.content_type, ContentType::Document)
    }

    fn cpu(&self, ctx: &Context) -> Vec<Finding> {
        let (width, height) = (ctx.width as usize, ctx.height as usize);
        if width < self.block_size * 4 || height < self.block_size * 4 {
            return Vec::new();
        }

        let gray = ctx.gray();
        self.analyze_dof(gray, width, height)
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

impl DofConsistencyAnalyzer {
    fn analyze_dof(&self, gray: &[u8], width: usize, height: usize) -> Vec<Finding> {
        let bs = self.block_size;
        let blocks_x = width / bs;
        let blocks_y = height / bs;

        // Per-block sharpness: mean Sobel magnitude over edge pixels.
        let mut block_sharpness: Vec<(f64, usize)> = Vec::new(); // (mean_grad, edge_count)

        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                let x0 = bx * bs;
                let y0 = by * bs;
                let mut grad_sum = 0.0_f64;
                let mut count = 0_usize;

                for dy in 1..bs.saturating_sub(1) {
                    for dx in 1..bs.saturating_sub(1) {
                        let x = x0 + dx;
                        let y = y0 + dy;
                        if x >= width - 1 || y >= height - 1 {
                            continue;
                        }

                        let f = |px: usize, py: usize| gray[py * width + px] as f64;
                        let gx = -f(x - 1, y - 1) + f(x + 1, y - 1) - 2.0 * f(x - 1, y)
                            + 2.0 * f(x + 1, y)
                            - f(x - 1, y + 1)
                            + f(x + 1, y + 1);
                        let gy = -f(x - 1, y - 1) - 2.0 * f(x, y - 1) - f(x + 1, y - 1)
                            + f(x - 1, y + 1)
                            + 2.0 * f(x, y + 1)
                            + f(x + 1, y + 1);
                        let mag = (gx * gx + gy * gy).sqrt();

                        if mag >= self.edge_threshold {
                            grad_sum += mag;
                            count += 1;
                        }
                    }
                }

                let mean_grad = if count > 0 {
                    grad_sum / count as f64
                } else {
                    0.0
                };
                block_sharpness.push((mean_grad, count));
            }
        }

        let mut findings = Vec::new();

        // Abrupt blur transitions between adjacent blocks (natural DoF is
        // gradual; splices jump).
        let mut jump_count = 0;
        let mut max_jump = 0.0_f64;
        let mut max_jump_pos = (0_usize, 0_usize);

        for by in 0..blocks_y {
            for bx in 0..blocks_x - 1 {
                let idx = by * blocks_x + bx;
                let right = idx + 1;

                let (s1, c1) = block_sharpness[idx];
                let (s2, c2) = block_sharpness[right];

                if c1 < self.min_edge_pixels || c2 < self.min_edge_pixels {
                    continue;
                }

                let mean = (s1 + s2) / 2.0;
                if mean < 1.0 {
                    continue;
                }

                let jump = (s1 - s2).abs() / mean;
                if jump > 0.6 {
                    jump_count += 1;
                    if jump > max_jump {
                        max_jump = jump;
                        max_jump_pos = (bx, by);
                    }
                }
            }
        }

        for by in 0..blocks_y - 1 {
            for bx in 0..blocks_x {
                let idx = by * blocks_x + bx;
                let below = (by + 1) * blocks_x + bx;

                let (s1, c1) = block_sharpness[idx];
                let (s2, c2) = block_sharpness[below];

                if c1 < self.min_edge_pixels || c2 < self.min_edge_pixels {
                    continue;
                }

                let mean = (s1 + s2) / 2.0;
                if mean < 1.0 {
                    continue;
                }

                let jump = (s1 - s2).abs() / mean;
                if jump > 0.6 {
                    jump_count += 1;
                    if jump > max_jump {
                        max_jump = jump;
                        max_jump_pos = (bx, by);
                    }
                }
            }
        }

        let total_adjacencies = (blocks_x - 1) * blocks_y + blocks_x * (blocks_y - 1);
        let jump_ratio = jump_count as f64 / total_adjacencies.max(1) as f64;

        // Some jumps are natural object edges; flag only if concentrated.
        if jump_count >= 3 && jump_ratio > 0.02 && jump_ratio < 0.30 {
            findings.push(
                Finding::new(
                    "dof_consistency",
                    "dof_abrupt_transition",
                    format!(
                        "{jump_count} abrupt blur transitions detected ({:.1}% of block \
                         boundaries, max {:.0}% sharpness jump) - depth-of-field \
                         discontinuity inconsistent with natural lens optics",
                        jump_ratio * 100.0,
                        max_jump * 100.0,
                    ),
                    if max_jump > 1.0 {
                        Severity::High
                    } else {
                        Severity::Medium
                    },
                    (0.40 + jump_ratio * 3.0).min(0.75),
                )
                .with_region(Region::BoundingBox {
                    x: (max_jump_pos.0 * bs) as u32,
                    y: (max_jump_pos.1 * bs) as u32,
                    width: (bs * 2) as u32,
                    height: bs as u32,
                }),
            );
        }

        findings
    }
}
