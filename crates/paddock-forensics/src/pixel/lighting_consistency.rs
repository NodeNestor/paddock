//! Lighting-direction consistency (Johnson & Farid 2007), ported from the reference.
//! Canonical = CPU; a bit-exact CUDA kernel (`lighting_plane_block`, one thread
//! per block, f64, `--fmad=false`) computes the identical per-block least-squares
//! plane gradient `(a,b)`. Both paths then derive direction/magnitude from `(a,b)`
//! through the same Rust `from_grad`, and all neighbour cos/sin/atan2 math runs
//! host-side - so no libm-vs-CUDA transcendental drift can enter, and GPU == CPU
//! exactly. (Not a copy of the reference's divergent device path.)
//!
//! A block's intensity plane gradient is the projected light direction; composite
//! regions from differently-lit scenes point differently even when brightness is
//! matched. Camera/scene-specific -> skipped for documents.

use crate::analyzer::Analyzer;
use crate::context::ContentType;
use crate::{Context, Finding, Region, Severity};

pub struct LightingConsistencyAnalyzer {
    block_size: usize,
    neighbor_radius: usize,
    min_gradient_magnitude: f64,
    anomaly_z_threshold: f64,
}

impl Default for LightingConsistencyAnalyzer {
    fn default() -> Self {
        Self {
            block_size: 64,
            neighbor_radius: 2,
            min_gradient_magnitude: 0.001,
            anomaly_z_threshold: 3.0,
        }
    }
}

struct BlockLighting {
    direction: f64,
    magnitude: f64,
}

impl BlockLighting {
    /// Derive direction/magnitude from the raw plane gradient - the one place
    /// both the CPU and GPU paths turn `(a,b)` into a block result, so identical
    /// `(a,b)` yield identical blocks (Rust transcendentals for both).
    fn from_grad(a: f64, b: f64) -> Self {
        Self {
            direction: b.atan2(a),
            magnitude: (a * a + b * b).sqrt(),
        }
    }
}

impl Analyzer for LightingConsistencyAnalyzer {
    fn name(&self) -> &'static str {
        "lighting_consistency"
    }

    fn applies_to(&self, ctx: &Context) -> bool {
        // Camera/scene-specific -> skip documents (which includes PDFs).
        !matches!(ctx.content_type, ContentType::Document)
    }

    fn cpu(&self, ctx: &Context) -> Vec<Finding> {
        let (width, height) = (ctx.width as usize, ctx.height as usize);
        if width < self.block_size * 4 || height < self.block_size * 4 {
            return Vec::new();
        }
        let blocks = self.fit_lighting_cpu(ctx.gray(), width, height);
        self.analyze_lighting(&blocks, width, height)
    }

    #[cfg(feature = "cuda")]
    fn gpu(
        &self,
        gpu: &crate::gpu::ForensicGpu,
        ctx: &Context,
    ) -> Result<Vec<Finding>, crate::gpu::GpuError> {
        use cudarc::driver::{LaunchConfig, PushKernelArg};

        let (width, height) = (ctx.width as usize, ctx.height as usize);
        if width < self.block_size * 4 || height < self.block_size * 4 {
            return Ok(Vec::new());
        }
        let bs = self.block_size;
        let (blocks_x, blocks_y) = (width / bs, height / bs);
        let total = blocks_x * blocks_y;

        let stream = gpu.stream();
        let d_gray = stream.clone_htod(ctx.gray())?;
        let mut d_a = stream.alloc_zeros::<f64>(total)?;
        let mut d_b = stream.alloc_zeros::<f64>(total)?;

        let (w_u, h_u, bs_u) = (width as u32, height as u32, bs as u32);
        let (bx_u, by_u) = (blocks_x as u32, blocks_y as u32);
        let f = gpu.function("lighting_consistency", "lighting_plane_block")?;
        let threads = 256u32;
        let cfg = LaunchConfig {
            grid_dim: ((total as u32).div_ceil(threads).max(1), 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(&f)
                .arg(&d_gray)
                .arg(&mut d_a)
                .arg(&mut d_b)
                .arg(&w_u)
                .arg(&h_u)
                .arg(&bs_u)
                .arg(&bx_u)
                .arg(&by_u)
                .launch(cfg)?;
        }
        let a: Vec<f64> = stream.clone_dtoh(&d_a)?;
        let b: Vec<f64> = stream.clone_dtoh(&d_b)?;
        stream.synchronize()?;

        let blocks: Vec<BlockLighting> = (0..total)
            .map(|i| BlockLighting::from_grad(a[i], b[i]))
            .collect();
        Ok(self.analyze_lighting(&blocks, width, height))
    }
}

impl LightingConsistencyAnalyzer {
    fn fit_lighting_cpu(&self, gray: &[u8], width: usize, height: usize) -> Vec<BlockLighting> {
        let bs = self.block_size;
        let blocks_x = width / bs;
        let blocks_y = height / bs;
        let mut blocks = Vec::with_capacity(blocks_x * blocks_y);

        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                let x0 = bx * bs;
                let y0 = by * bs;

                let mut sx = 0.0_f64;
                let mut sy = 0.0_f64;
                let mut si = 0.0_f64;
                let mut sxx = 0.0_f64;
                let mut syy = 0.0_f64;
                let mut sxy = 0.0_f64;
                let mut sxi = 0.0_f64;
                let mut syi = 0.0_f64;
                let mut n = 0.0_f64;

                for dy in 0..bs {
                    for dx in 0..bs {
                        let x = x0 + dx;
                        let y = y0 + dy;
                        if x >= width || y >= height {
                            continue;
                        }

                        let intensity = gray[y * width + x] as f64 / 255.0;
                        let fdx = dx as f64;
                        let fdy = dy as f64;

                        sx += fdx;
                        sy += fdy;
                        si += intensity;
                        sxx += fdx * fdx;
                        syy += fdy * fdy;
                        sxy += fdx * fdy;
                        sxi += fdx * intensity;
                        syi += fdy * intensity;
                        n += 1.0;
                    }
                }

                // Edge cases -> (a,b) = (0,0), exactly as the reference (direction 0,
                // magnitude 0). `si` (-> mean intensity) is unused downstream.
                let (a, b) = if n < 4.0 {
                    (0.0, 0.0)
                } else {
                    let det = sxx * (syy * n - sy * sy) - sxy * (sxy * n - sy * sx)
                        + sx * (sxy * sy - syy * sx);

                    if det.abs() < 1e-10 {
                        (0.0, 0.0)
                    } else {
                        let a = (sxi * (syy * n - sy * sy) - sxy * (syi * n - sy * si)
                            + sx * (syi * sy - syy * si))
                            / det;
                        let b = (sxx * (syi * n - sy * si) - sxi * (sxy * n - sy * sx)
                            + sx * (sxy * si - syi * sx))
                            / det;
                        (a, b)
                    }
                };

                blocks.push(BlockLighting::from_grad(a, b));
            }
        }

        blocks
    }

    fn analyze_lighting(
        &self,
        blocks: &[BlockLighting],
        width: usize,
        height: usize,
    ) -> Vec<Finding> {
        let bs = self.block_size;
        let blocks_x = width / bs;
        let blocks_y = height / bs;
        let num_blocks = blocks_x * blocks_y;
        let radius = self.neighbor_radius;
        let mut findings = Vec::new();

        if num_blocks < 9 {
            return findings;
        }

        let active: Vec<usize> = (0..num_blocks)
            .filter(|&b| blocks[b].magnitude > self.min_gradient_magnitude)
            .collect();

        if active.len() < 9 {
            return findings;
        }

        let mut deviations: Vec<(usize, f64)> = Vec::new();

        for &b in &active {
            let bx = b % blocks_x;
            let by = b / blocks_x;
            let block_dir = blocks[b].direction;

            let mut neighbor_cos = 0.0_f64;
            let mut neighbor_sin = 0.0_f64;
            let mut n_count = 0;

            let y_start = by.saturating_sub(radius);
            let y_end = (by + radius + 1).min(blocks_y);
            let x_start = bx.saturating_sub(radius);
            let x_end = (bx + radius + 1).min(blocks_x);

            for ny in y_start..y_end {
                for nx in x_start..x_end {
                    if nx == bx && ny == by {
                        continue;
                    }
                    let n_idx = ny * blocks_x + nx;
                    if n_idx < num_blocks && blocks[n_idx].magnitude > self.min_gradient_magnitude {
                        neighbor_cos += blocks[n_idx].direction.cos();
                        neighbor_sin += blocks[n_idx].direction.sin();
                        n_count += 1;
                    }
                }
            }

            if n_count >= 3 {
                let neighbor_dir = neighbor_sin.atan2(neighbor_cos);
                let dev = circular_diff(block_dir, neighbor_dir);
                deviations.push((b, dev));
            }
        }

        if deviations.len() < 9 {
            return findings;
        }

        let mut sorted: Vec<f64> = deviations.iter().map(|&(_, d)| d).collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = sorted[sorted.len() / 2];
        let mad = {
            let mut devs: Vec<f64> = sorted.iter().map(|d| (d - median).abs()).collect();
            devs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            devs[devs.len() / 2] * 1.4826
        };

        if mad < 1e-6 {
            return findings;
        }

        let anomalous: Vec<(usize, f64)> = deviations
            .iter()
            .filter_map(|&(b, d)| {
                let z = (d - median) / mad;
                if z > self.anomaly_z_threshold {
                    Some((b, z))
                } else {
                    None
                }
            })
            .collect();

        if anomalous.len() < 3 {
            return findings;
        }

        let max_z = anomalous.iter().fold(0.0_f64, |m, &(_, z)| m.max(z));
        let anomaly_ratio = anomalous.len() as f64 / deviations.len() as f64;

        let strongest = anomalous
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();
        let sx = (strongest.0 % blocks_x) * bs;
        let sy = (strongest.0 / blocks_x) * bs;

        findings.push(
            Finding::new(
                "lighting_consistency",
                "lighting_direction_inconsistency",
                format!(
                    "{} of {} blocks ({:.1}%) show lighting direction inconsistent with \
                     surroundings (z-score {max_z:.1}) - possible composite with different \
                     illumination geometry",
                    anomalous.len(),
                    deviations.len(),
                    anomaly_ratio * 100.0,
                ),
                if max_z > 5.0 || anomaly_ratio > 0.15 {
                    Severity::High
                } else {
                    Severity::Medium
                },
                (0.40 + anomaly_ratio * 1.5).min(0.80),
            )
            .with_region(Region::BoundingBox {
                x: sx as u32,
                y: sy as u32,
                width: bs as u32,
                height: bs as u32,
            }),
        );

        findings
    }
}

/// Circular difference in [0, π].
fn circular_diff(a: f64, b: f64) -> f64 {
    let diff = (a - b).abs();
    let diff = diff % (2.0 * std::f64::consts::PI);
    diff.min(2.0 * std::f64::consts::PI - diff)
}
