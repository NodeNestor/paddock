//! Color-channel correlation, ported from the reference. Canonical = CPU; a bit-exact
//! CUDA kernel (`channel_correlation_block`, one thread per block, f64,
//! `--fmad=false`) computes the identical per-block min R/G/B Pearson
//! correlation, then the same host-side neighbor/MAD logic runs on both paths.
//! (Not a copy of the reference's divergent device path.)
//!
//! Natural images correlate strongly across R/G/B (adjacent spectral bands see
//! similar radiance). Splices, white-balance mismatches, and AI content break it
//! locally. Photo-oriented -> skipped for PDFs.

use crate::analyzer::Analyzer;
use crate::{Context, Finding, Region, Severity};

pub struct ChannelCorrelationAnalyzer {
    block_size: usize,
    neighbor_radius: usize,
    anomaly_z_threshold: f64,
    min_anomaly_blocks: usize,
}

impl Default for ChannelCorrelationAnalyzer {
    fn default() -> Self {
        Self {
            block_size: 32,
            neighbor_radius: 2,
            anomaly_z_threshold: 3.0,
            min_anomaly_blocks: 3,
        }
    }
}

/// Only `min_corr` feeds the downstream logic.
struct BlockCorrelation {
    min_corr: f64,
}

impl Analyzer for ChannelCorrelationAnalyzer {
    fn name(&self) -> &'static str {
        "channel_correlation"
    }

    fn applies_to(&self, ctx: &Context) -> bool {
        !ctx.is_pdf()
    }

    fn cpu(&self, ctx: &Context) -> Vec<Finding> {
        let (width, height) = (ctx.width as usize, ctx.height as usize);
        if width < self.block_size * 4 || height < self.block_size * 4 {
            return Vec::new();
        }
        let rgb = ctx.image.to_rgb8();
        let blocks = self.compute_cpu(rgb.as_raw(), width, height);
        self.analyze_blocks(&blocks, width, height)
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

        let rgb = ctx.image.to_rgb8();
        let stream = gpu.stream();
        let d_rgb = stream.clone_htod(rgb.as_raw())?;
        let mut d_min = stream.alloc_zeros::<f64>(total)?;

        let (w_u, h_u, bs_u) = (width as u32, height as u32, bs as u32);
        let (bx_u, by_u) = (blocks_x as u32, blocks_y as u32);
        let f = gpu.function("channel_correlation", "channel_correlation_block")?;
        let threads = 256u32;
        let cfg = LaunchConfig {
            grid_dim: ((total as u32).div_ceil(threads).max(1), 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(&f)
                .arg(&d_rgb)
                .arg(&mut d_min)
                .arg(&w_u)
                .arg(&h_u)
                .arg(&bs_u)
                .arg(&bx_u)
                .arg(&by_u)
                .launch(cfg)?;
        }
        let mins: Vec<f64> = stream.clone_dtoh(&d_min)?;
        stream.synchronize()?;

        let blocks: Vec<BlockCorrelation> = mins
            .iter()
            .map(|&min_corr| BlockCorrelation { min_corr })
            .collect();
        Ok(self.analyze_blocks(&blocks, width, height))
    }
}

impl ChannelCorrelationAnalyzer {
    fn compute_cpu(&self, rgb: &[u8], width: usize, height: usize) -> Vec<BlockCorrelation> {
        let bs = self.block_size;
        let blocks_x = width / bs;
        let blocks_y = height / bs;
        let mut blocks = Vec::with_capacity(blocks_x * blocks_y);

        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                let x0 = bx * bs;
                let y0 = by * bs;

                let mut sr = 0.0_f64;
                let mut sg = 0.0_f64;
                let mut sb = 0.0_f64;
                let mut srr = 0.0_f64;
                let mut sgg = 0.0_f64;
                let mut sbb = 0.0_f64;
                let mut srg = 0.0_f64;
                let mut srb = 0.0_f64;
                let mut sgb = 0.0_f64;
                let mut n = 0.0_f64;

                for dy in 0..bs {
                    for dx in 0..bs {
                        let x = x0 + dx;
                        let y = y0 + dy;
                        if x >= width || y >= height {
                            continue;
                        }

                        let idx = (y * width + x) * 3;
                        let r = rgb[idx] as f64 / 255.0;
                        let g = rgb[idx + 1] as f64 / 255.0;
                        let b = rgb[idx + 2] as f64 / 255.0;

                        sr += r;
                        sg += g;
                        sb += b;
                        srr += r * r;
                        sgg += g * g;
                        sbb += b * b;
                        srg += r * g;
                        srb += r * b;
                        sgb += g * b;
                        n += 1.0;
                    }
                }

                if n < 4.0 {
                    blocks.push(BlockCorrelation { min_corr: 1.0 });
                    continue;
                }

                let pearson = |sx: f64, sy: f64, sxx: f64, syy: f64, sxy: f64| -> f64 {
                    let var_x = n * sxx - sx * sx;
                    let var_y = n * syy - sy * sy;
                    let cov = n * sxy - sx * sy;
                    let denom = (var_x * var_y).max(1e-10).sqrt();
                    cov / denom
                };

                let rg = pearson(sr, sg, srr, sgg, srg);
                let rb = pearson(sr, sb, srr, sbb, srb);
                let gb = pearson(sg, sb, sgg, sbb, sgb);

                blocks.push(BlockCorrelation {
                    min_corr: rg.min(rb).min(gb),
                });
            }
        }

        blocks
    }

    fn analyze_blocks(
        &self,
        blocks: &[BlockCorrelation],
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

        let mut deviations: Vec<(usize, f64)> = Vec::new();

        for b in 0..num_blocks {
            let bx = b % blocks_x;
            let by = b / blocks_x;
            let block_corr = blocks[b].min_corr;

            let mut n_sum = 0.0_f64;
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
                    n_sum += blocks[n_idx].min_corr;
                    n_count += 1;
                }
            }

            if n_count >= 3 {
                let neighbor_avg = n_sum / n_count as f64;
                let dev = (neighbor_avg - block_corr).max(0.0); // Low correlation = anomaly.
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

        if anomalous.len() < self.min_anomaly_blocks {
            return findings;
        }

        let max_z = anomalous.iter().fold(0.0_f64, |m, &(_, z)| m.max(z));
        let Some(strongest) = anomalous
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        else {
            return findings;
        };
        let sx = (strongest.0 % blocks_x) * bs;
        let sy = (strongest.0 / blocks_x) * bs;

        findings.push(
            Finding::new(
                "channel_correlation",
                "channel_correlation_anomaly",
                format!(
                    "{} blocks show reduced R/G/B channel correlation compared to \
                     surroundings (z-score {max_z:.1}) - possible splice from image with \
                     different color processing or white balance",
                    anomalous.len(),
                ),
                if max_z > 5.0 {
                    Severity::High
                } else {
                    Severity::Medium
                },
                (0.45 + (max_z - self.anomaly_z_threshold) * 0.08).min(0.85),
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
