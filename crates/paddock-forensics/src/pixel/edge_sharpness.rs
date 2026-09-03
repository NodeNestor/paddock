//! Edge-sharpness profiling, ported from the reference. Canonical algorithm = the CPU
//! reference; a bit-exact CUDA kernel (`edge_sharpness_block`, one thread per
//! image block, f64, `--fmad=false`) computes the identical per-block mean edge
//! width + edge count, then the same host-side MAD-outlier logic runs on both
//! paths. GPU == CPU exactly (see tests/parity.rs).
//!
//! NOTE: this is not a copy of the reference's GPU kernel - its device path
//! diverges from its own CPU path; paddock defines one canonical algorithm (the
//! CPU one) and computes it identically on the GPU.
//!
//! Natural edges from one lens share a blur width set by the PSF. Cut-paste
//! makes unnaturally SHARP edges (small width); feathered blends make
//! unnaturally BLURRED ones (large width). Width ≈ gradient / |laplacian|.
//! Camera-specific -> skipped for documents (high false-positive on text).

use crate::analyzer::Analyzer;
use crate::context::ContentType;
use crate::{Context, Finding, Region, Severity};

pub struct EdgeSharpnessAnalyzer {
    /// Block size for analysis.
    block_size: usize,
    /// Sobel gradient threshold (on [0,1] grayscale).
    edge_threshold: f32,
    /// Minimum edge pixels per block to count as active.
    min_edge_pixels: usize,
    /// MAD z-score threshold for anomaly detection.
    anomaly_z_threshold: f64,
}

impl Default for EdgeSharpnessAnalyzer {
    fn default() -> Self {
        Self {
            block_size: 32,
            edge_threshold: 0.15,
            min_edge_pixels: 10,
            anomaly_z_threshold: 3.0,
        }
    }
}

/// Per-block sharpness statistics. Only `mean_width` and `edge_count` feed the
/// downstream logic, so those are all we compute (on either path).
struct BlockSharpness {
    mean_width: f64,
    edge_count: usize,
}

impl Analyzer for EdgeSharpnessAnalyzer {
    fn name(&self) -> &'static str {
        "edge_sharpness"
    }

    fn applies_to(&self, ctx: &Context) -> bool {
        // High false-positive on documents -> skip them.
        !matches!(ctx.content_type, ContentType::Document)
    }

    fn cpu(&self, ctx: &Context) -> Vec<Finding> {
        let (width, height) = (ctx.width as usize, ctx.height as usize);
        if width < self.block_size * 4 || height < self.block_size * 4 {
            return Vec::new();
        }

        let gray = ctx.gray();
        let block_stats = self.compute_sharpness_cpu(gray, width, height);
        self.analyze_blocks(&block_stats, width, height)
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
        let blocks_x = width / bs;
        let blocks_y = height / bs;
        let total = blocks_x * blocks_y;
        // Same f64 threshold the CPU derives: (edge_threshold * 255) in f32,
        // widened to f64.
        let threshold = (self.edge_threshold * 255.0) as f64;

        let stream = gpu.stream();
        let d_gray = stream.clone_htod(ctx.gray())?;
        let mut d_mean = stream.alloc_zeros::<f64>(total)?;
        let mut d_count = stream.alloc_zeros::<i32>(total)?;

        let (w_u, h_u, bs_u) = (width as u32, height as u32, bs as u32);
        let (bx_u, by_u) = (blocks_x as u32, blocks_y as u32);
        let f = gpu.function("edge_sharpness", "edge_sharpness_block")?;
        let threads = 256u32;
        let grid = (total as u32).div_ceil(threads).max(1);
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(&f)
                .arg(&d_gray)
                .arg(&mut d_mean)
                .arg(&mut d_count)
                .arg(&w_u)
                .arg(&h_u)
                .arg(&bs_u)
                .arg(&threshold)
                .arg(&bx_u)
                .arg(&by_u)
                .launch(cfg)?;
        }
        let means: Vec<f64> = stream.clone_dtoh(&d_mean)?;
        let counts: Vec<i32> = stream.clone_dtoh(&d_count)?;
        stream.synchronize()?;

        let block_stats: Vec<BlockSharpness> = means
            .iter()
            .zip(counts.iter())
            .map(|(&mean_width, &c)| BlockSharpness {
                mean_width,
                edge_count: c as usize,
            })
            .collect();

        Ok(self.analyze_blocks(&block_stats, width, height))
    }
}

impl EdgeSharpnessAnalyzer {
    fn compute_sharpness_cpu(
        &self,
        gray: &[u8],
        width: usize,
        height: usize,
    ) -> Vec<BlockSharpness> {
        let bs = self.block_size;
        let blocks_x = width / bs;
        let blocks_y = height / bs;
        let threshold = (self.edge_threshold * 255.0) as f64;
        let mut stats = Vec::with_capacity(blocks_x * blocks_y);

        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                let x0 = bx * bs;
                let y0 = by * bs;
                let mut w_sum = 0.0_f64;
                let mut count = 0_usize;

                for dy in 2..bs.saturating_sub(2) {
                    for dx in 2..bs.saturating_sub(2) {
                        let x = x0 + dx;
                        let y = y0 + dy;
                        if x >= width - 2 || y >= height - 2 {
                            continue;
                        }

                        let fetch = |px: usize, py: usize| gray[py * width + px] as f64;

                        let gx = -fetch(x - 1, y - 1) + fetch(x + 1, y - 1) - 2.0 * fetch(x - 1, y)
                            + 2.0 * fetch(x + 1, y)
                            - fetch(x - 1, y + 1)
                            + fetch(x + 1, y + 1);

                        let gy = -fetch(x - 1, y - 1) - 2.0 * fetch(x, y - 1) - fetch(x + 1, y - 1)
                            + fetch(x - 1, y + 1)
                            + 2.0 * fetch(x, y + 1)
                            + fetch(x + 1, y + 1);

                        let mag = (gx * gx + gy * gy).sqrt();
                        if mag < threshold {
                            continue;
                        }

                        let center = fetch(x, y);
                        let lap = -4.0 * center
                            + fetch(x - 1, y)
                            + fetch(x + 1, y)
                            + fetch(x, y - 1)
                            + fetch(x, y + 1);

                        let abs_lap = lap.abs();
                        let w = if abs_lap > 1e-3 {
                            (mag / abs_lap).min(10.0)
                        } else {
                            10.0
                        };

                        w_sum += w;
                        count += 1;
                    }
                }

                let mean_w = if count > 0 { w_sum / count as f64 } else { 0.0 };
                stats.push(BlockSharpness {
                    mean_width: mean_w,
                    edge_count: count,
                });
            }
        }

        stats
    }

    fn analyze_blocks(
        &self,
        blocks: &[BlockSharpness],
        width: usize,
        _height: usize,
    ) -> Vec<Finding> {
        let bs = self.block_size;
        let blocks_x = width / bs;
        let mut findings = Vec::new();

        let active: Vec<(usize, f64)> = blocks
            .iter()
            .enumerate()
            .filter(|(_, b)| b.edge_count >= self.min_edge_pixels)
            .map(|(i, b)| (i, b.mean_width))
            .collect();

        if active.len() < 9 {
            return findings;
        }

        let widths: Vec<f64> = active.iter().map(|&(_, w)| w).collect();
        let mut sorted = widths.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = sorted[sorted.len() / 2];
        let mad = {
            let mut devs: Vec<f64> = widths.iter().map(|w| (w - median).abs()).collect();
            devs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            devs[devs.len() / 2] * 1.4826
        };

        if mad < 1e-6 {
            return findings; // Uniform sharpness - no signal.
        }

        let mut sharp_blocks: Vec<(usize, f64)> = Vec::new();
        let mut blurred_blocks: Vec<(usize, f64)> = Vec::new();

        for &(idx, width_val) in &active {
            let z = (width_val - median) / mad;
            if z < -self.anomaly_z_threshold {
                sharp_blocks.push((idx, -z));
            } else if z > self.anomaly_z_threshold {
                blurred_blocks.push((idx, z));
            }
        }

        if sharp_blocks.len() >= 2 {
            let max_z = sharp_blocks.iter().fold(0.0_f64, |m, &(_, z)| m.max(z));
            let (min_x, min_y, max_x, max_y) = self.bounding_box(&sharp_blocks, blocks_x, bs);

            findings.push(
                Finding::new(
                    "edge_sharpness",
                    "edge_sharpness_unnaturally_sharp",
                    format!(
                        "{} blocks have edges {max_z:.1}× sharper than image median - \
                         possible cut-paste splice with hard boundaries",
                        sharp_blocks.len(),
                    ),
                    if max_z > 5.0 {
                        Severity::High
                    } else {
                        Severity::Medium
                    },
                    (0.45 + (max_z - self.anomaly_z_threshold) * 0.08).min(0.85),
                )
                .with_region(Region::BoundingBox {
                    x: min_x as u32,
                    y: min_y as u32,
                    width: (max_x - min_x) as u32,
                    height: (max_y - min_y) as u32,
                }),
            );
        }

        if blurred_blocks.len() >= 2 {
            let max_z = blurred_blocks.iter().fold(0.0_f64, |m, &(_, z)| m.max(z));
            let (min_x, min_y, max_x, max_y) = self.bounding_box(&blurred_blocks, blocks_x, bs);

            findings.push(
                Finding::new(
                    "edge_sharpness",
                    "edge_sharpness_unnaturally_blurred",
                    format!(
                        "{} blocks have edges {max_z:.1}× blurrier than image median - \
                         possible feathered splice or localized blur",
                        blurred_blocks.len(),
                    ),
                    if max_z > 5.0 {
                        Severity::High
                    } else {
                        Severity::Medium
                    },
                    (0.40 + (max_z - self.anomaly_z_threshold) * 0.07).min(0.80),
                )
                .with_region(Region::BoundingBox {
                    x: min_x as u32,
                    y: min_y as u32,
                    width: (max_x - min_x) as u32,
                    height: (max_y - min_y) as u32,
                }),
            );
        }

        let total_anomalous = sharp_blocks.len() + blurred_blocks.len();
        let anomaly_ratio = total_anomalous as f64 / active.len() as f64;
        if anomaly_ratio > 0.10 && total_anomalous > 5 {
            findings.push(Finding::new(
                "edge_sharpness",
                "edge_sharpness_bimodal",
                format!(
                    "{:.1}% of blocks have anomalous edge sharpness ({} sharp, {} blurred) - \
                     image contains regions processed with different blur/sharpening",
                    anomaly_ratio * 100.0,
                    sharp_blocks.len(),
                    blurred_blocks.len(),
                ),
                Severity::High,
                (0.50 + anomaly_ratio).min(0.85),
            ));
        }

        findings
    }

    fn bounding_box(
        &self,
        blocks: &[(usize, f64)],
        blocks_x: usize,
        bs: usize,
    ) -> (usize, usize, usize, usize) {
        let mut min_x = usize::MAX;
        let mut min_y = usize::MAX;
        let mut max_x = 0_usize;
        let mut max_y = 0_usize;

        for &(idx, _) in blocks {
            let bx = idx % blocks_x;
            let by = idx / blocks_x;
            min_x = min_x.min(bx * bs);
            min_y = min_y.min(by * bs);
            max_x = max_x.max((bx + 1) * bs);
            max_y = max_y.max((by + 1) * bs);
        }

        (min_x, min_y, max_x, max_y)
    }
}
