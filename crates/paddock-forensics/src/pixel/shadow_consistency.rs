//! Shadow direction consistency, ported verbatim from the CPU reference.
//! CPU-only: the per-pixel loop makes discrete decisions from transcendentals -
//! `dir = atan2(gy,gx)`, then samples the neighbour at `round(cos(dir))`,
//! `round(sin(dir))` and accumulates `cos(dir)`/`sin(dir)`. CUDA's atan2/cos/sin
//! differ from Rust's libm in the last ULP, which can flip the rounded neighbour
//! (and thus the boundary accept/reject), so a bit-exact kernel of this algorithm
//! is not feasible (same class as double_jpeg); `gpu()` delegates to `cpu()`.
//! Camera/scene-specific -> skipped for documents.

use crate::analyzer::Analyzer;
use crate::context::ContentType;
use crate::{Context, Finding, Severity};

pub struct ShadowConsistencyAnalyzer {
    block_size: usize,
    dark_percentile: f64,
    min_boundary_pixels: usize,
    direction_tolerance_deg: f64,
}

impl Default for ShadowConsistencyAnalyzer {
    fn default() -> Self {
        Self {
            block_size: 64,
            dark_percentile: 0.30,
            min_boundary_pixels: 15,
            direction_tolerance_deg: 35.0,
        }
    }
}

struct BlockShadow {
    direction: f64,
    strength: f64,
    count: usize,
}

impl Analyzer for ShadowConsistencyAnalyzer {
    fn name(&self) -> &'static str {
        "shadow_consistency"
    }

    fn applies_to(&self, ctx: &Context) -> bool {
        // Camera/scene-specific -> skip documents (which includes PDFs).
        !matches!(ctx.content_type, ContentType::Document)
    }

    fn cpu(&self, ctx: &Context) -> Vec<Finding> {
        let (width, height) = (ctx.width as usize, ctx.height as usize);
        if width < self.block_size * 3 || height < self.block_size * 3 {
            return Vec::new();
        }

        let gray = ctx.gray();
        let dark_threshold = self.compute_dark_threshold(gray);
        let block_shadows = self.detect_shadows_cpu(gray, width, height, dark_threshold);
        self.analyze_directions(&block_shadows)
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

impl ShadowConsistencyAnalyzer {
    fn compute_dark_threshold(&self, gray: &[u8]) -> f64 {
        let mut histogram = [0u32; 256];
        for &v in gray {
            histogram[v as usize] += 1;
        }

        let target = (gray.len() as f64 * self.dark_percentile) as u32;
        let mut cumulative = 0u32;
        for (i, &count) in histogram.iter().enumerate() {
            cumulative += count;
            if cumulative >= target {
                return i as f64;
            }
        }
        128.0
    }

    fn detect_shadows_cpu(
        &self,
        gray: &[u8],
        width: usize,
        height: usize,
        dark_threshold: f64,
    ) -> Vec<BlockShadow> {
        let bs = self.block_size;
        let blocks_x = width / bs;
        let blocks_y = height / bs;
        let min_grad = 12.0; // ~0.05 * 255
        let max_grad = 100.0; // ~0.40 * 255
        let mut shadows = Vec::with_capacity(blocks_x * blocks_y);

        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                let x0 = bx * bs;
                let y0 = by * bs;
                let mut cos_sum = 0.0_f64;
                let mut sin_sum = 0.0_f64;
                let mut count = 0_usize;

                for dy in 1..bs.saturating_sub(1) {
                    for dx in 1..bs.saturating_sub(1) {
                        let x = x0 + dx;
                        let y = y0 + dy;
                        if x >= width - 1 || y >= height - 1 {
                            continue;
                        }

                        let center = gray[y * width + x] as f64;
                        if center > dark_threshold {
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
                        if mag < min_grad || mag > max_grad {
                            continue;
                        }

                        let dir = gy.atan2(gx);
                        let bx_off = dir.cos().round() as i64;
                        let by_off = dir.sin().round() as i64;
                        let nx = (x as i64 + bx_off).clamp(0, width as i64 - 1) as usize;
                        let ny = (y as i64 + by_off).clamp(0, height as i64 - 1) as usize;
                        let bright_side = gray[ny * width + nx] as f64;

                        if bright_side <= center * 1.3 {
                            continue;
                        }

                        cos_sum += dir.cos();
                        sin_sum += dir.sin();
                        count += 1;
                    }
                }

                let (direction, strength) = if count >= self.min_boundary_pixels {
                    let d = sin_sum.atan2(cos_sum);
                    let s = (cos_sum.powi(2) + sin_sum.powi(2)).sqrt() / count as f64;
                    (d, s)
                } else {
                    (0.0, 0.0)
                };

                shadows.push(BlockShadow {
                    direction,
                    strength,
                    count,
                });
            }
        }

        shadows
    }

    fn analyze_directions(&self, blocks: &[BlockShadow]) -> Vec<Finding> {
        let mut findings = Vec::new();

        let active: Vec<&BlockShadow> = blocks
            .iter()
            .filter(|b| b.count >= self.min_boundary_pixels && b.strength > 0.3)
            .collect();

        if active.len() < 4 {
            return findings;
        }

        let mut global_cos = 0.0_f64;
        let mut global_sin = 0.0_f64;
        for b in &active {
            global_cos += b.direction.cos() * b.strength;
            global_sin += b.direction.sin() * b.strength;
        }
        let global_dir = global_sin.atan2(global_cos);

        let tolerance = self.direction_tolerance_deg.to_radians();
        let mut inconsistent_count = 0;
        let mut max_deviation = 0.0_f64;

        for b in &active {
            let deviation = circular_diff(b.direction, global_dir);
            if deviation > tolerance {
                inconsistent_count += 1;
                max_deviation = max_deviation.max(deviation);
            }
        }

        let inconsistency_ratio = inconsistent_count as f64 / active.len() as f64;

        if inconsistency_ratio > 0.15 && inconsistent_count >= 3 {
            findings.push(Finding::new(
                "shadow_consistency",
                "shadow_direction_inconsistency",
                format!(
                    "{} of {} shadow-bearing blocks ({:.0}%) have shadow directions \
                     deviating >{:.0}° from dominant direction (max deviation {:.0}°) - \
                     possible composite with different light source positions",
                    inconsistent_count,
                    active.len(),
                    inconsistency_ratio * 100.0,
                    self.direction_tolerance_deg,
                    max_deviation.to_degrees(),
                ),
                if inconsistency_ratio > 0.30 {
                    Severity::High
                } else {
                    Severity::Medium
                },
                (0.40 + inconsistency_ratio * 0.8).min(0.80),
            ));
        }

        let bimodal = self.check_bimodal(&active);
        if let Some((dir1, dir2, separation)) = bimodal {
            findings.push(Finding::new(
                "shadow_consistency",
                "shadow_bimodal_direction",
                format!(
                    "Two dominant shadow directions detected at {:.0}° and {:.0}° \
                     (separation {:.0}°) - evidence of two different light sources, \
                     strongly indicating image compositing",
                    dir1.to_degrees(),
                    dir2.to_degrees(),
                    separation.to_degrees(),
                ),
                Severity::High,
                (0.55 + separation.to_degrees() / 180.0 * 0.3).min(0.85),
            ));
        }

        findings
    }

    fn check_bimodal(&self, blocks: &[&BlockShadow]) -> Option<(f64, f64, f64)> {
        if blocks.len() < 8 {
            return None;
        }

        let num_bins = 36;
        let mut hist = vec![0.0_f64; num_bins];

        for b in blocks {
            let mut angle = b.direction;
            if angle < 0.0 {
                angle += 2.0 * std::f64::consts::PI;
            }
            let bin = ((angle / (2.0 * std::f64::consts::PI)) * num_bins as f64) as usize;
            let bin = bin.min(num_bins - 1);
            hist[bin] += b.strength;
        }

        let mut smoothed = vec![0.0_f64; num_bins];
        for i in 0..num_bins {
            let prev = (i + num_bins - 1) % num_bins;
            let next = (i + 1) % num_bins;
            smoothed[i] = 0.25 * hist[prev] + 0.5 * hist[i] + 0.25 * hist[next];
        }

        let mut peaks: Vec<(usize, f64)> = Vec::new();
        for i in 0..num_bins {
            let prev = (i + num_bins - 1) % num_bins;
            let next = (i + 1) % num_bins;
            if smoothed[i] > smoothed[prev] && smoothed[i] > smoothed[next] && smoothed[i] > 0.1 {
                peaks.push((i, smoothed[i]));
            }
        }

        peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        if peaks.len() >= 2 {
            let bin_to_angle =
                |bin: usize| (bin as f64 / num_bins as f64) * 2.0 * std::f64::consts::PI;
            let dir1 = bin_to_angle(peaks[0].0);
            let dir2 = bin_to_angle(peaks[1].0);
            let separation = circular_diff(dir1, dir2);

            if peaks[1].1 > peaks[0].1 * 0.30 && separation > 30.0_f64.to_radians() {
                return Some((dir1, dir2, separation));
            }
        }

        None
    }
}

/// Circular difference in [0, π].
fn circular_diff(a: f64, b: f64) -> f64 {
    let diff = (a - b).abs();
    let diff = diff % (2.0 * std::f64::consts::PI);
    diff.min(2.0 * std::f64::consts::PI - diff)
}
