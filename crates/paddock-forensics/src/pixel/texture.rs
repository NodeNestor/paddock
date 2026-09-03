//! Texture consistency via Local Binary Patterns, ported from the reference.
//! Canonical = CPU; a CUDA kernel (`lbp_histogram_block`, one thread per block)
//! computes the identical per-block 256-bin LBP histogram. The counts are
//! integers (order-independent, exact in f32), so GPU == CPU exactly with no
//! tolerance; the same host-side chi-squared neighbor MAD + clustering runs on
//! both paths. (Not a copy of the reference's divergent device path.)
//!
//! Every sensor/pipeline leaves a micro-texture fingerprint; a spliced region
//! has a different LBP histogram from its surroundings even when noise and color
//! match. Photo-oriented -> skipped for PDFs.

use crate::analyzer::Analyzer;
use crate::{Context, Finding, Region, Severity};

pub struct TextureConsistencyAnalyzer {
    block_size: usize,
    neighbor_radius: usize,
    anomaly_z_threshold: f64,
    min_anomaly_blocks: usize,
}

impl Default for TextureConsistencyAnalyzer {
    fn default() -> Self {
        Self {
            block_size: 32,
            neighbor_radius: 2,
            anomaly_z_threshold: 3.5,
            min_anomaly_blocks: 3,
        }
    }
}

const LBP_BINS: usize = 256;

impl Analyzer for TextureConsistencyAnalyzer {
    fn name(&self) -> &'static str {
        "texture_consistency"
    }

    fn applies_to(&self, ctx: &Context) -> bool {
        !ctx.is_pdf()
    }

    fn cpu(&self, ctx: &Context) -> Vec<Finding> {
        let (width, height) = (ctx.width as usize, ctx.height as usize);
        if width < self.block_size * 4 || height < self.block_size * 4 {
            return Vec::new();
        }
        let histograms = self.compute_lbp_histograms_cpu(ctx.gray(), width, height);
        self.analyze_histograms(&histograms, width, height)
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
        let mut d_hist = stream.alloc_zeros::<f32>(total * LBP_BINS)?;

        let (w_u, h_u, bs_u) = (width as u32, height as u32, bs as u32);
        let (bx_u, by_u) = (blocks_x as u32, blocks_y as u32);
        let f = gpu.function("texture", "lbp_histogram_block")?;
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
                .arg(&mut d_hist)
                .arg(&w_u)
                .arg(&h_u)
                .arg(&bs_u)
                .arg(&bx_u)
                .arg(&by_u)
                .launch(cfg)?;
        }
        let histograms: Vec<f32> = stream.clone_dtoh(&d_hist)?;
        stream.synchronize()?;

        Ok(self.analyze_histograms(&histograms, width, height))
    }
}

impl TextureConsistencyAnalyzer {
    fn compute_lbp_histograms_cpu(&self, gray: &[u8], width: usize, height: usize) -> Vec<f32> {
        let bs = self.block_size;
        let blocks_x = width / bs;
        let blocks_y = height / bs;
        let num_blocks = blocks_x * blocks_y;
        let mut histograms = vec![0.0_f32; num_blocks * LBP_BINS];

        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                let x0 = bx * bs;
                let y0 = by * bs;
                let block_idx = by * blocks_x + bx;

                for dy in 1..bs.saturating_sub(1) {
                    for dx in 1..bs.saturating_sub(1) {
                        let x = x0 + dx;
                        let y = y0 + dy;
                        if x >= width - 1 || y >= height - 1 {
                            continue;
                        }

                        let center = gray[y * width + x];
                        let fetch = |px: usize, py: usize| gray[py * width + px];

                        let mut code: u8 = 0;
                        code |= (fetch(x - 1, y - 1) >= center) as u8 * 128;
                        code |= (fetch(x, y - 1) >= center) as u8 * 64;
                        code |= (fetch(x + 1, y - 1) >= center) as u8 * 32;
                        code |= (fetch(x + 1, y) >= center) as u8 * 16;
                        code |= (fetch(x + 1, y + 1) >= center) as u8 * 8;
                        code |= (fetch(x, y + 1) >= center) as u8 * 4;
                        code |= (fetch(x - 1, y + 1) >= center) as u8 * 2;
                        code |= (fetch(x - 1, y) >= center) as u8;

                        histograms[block_idx * LBP_BINS + code as usize] += 1.0;
                    }
                }
            }
        }

        histograms
    }

    fn analyze_histograms(&self, histograms: &[f32], width: usize, height: usize) -> Vec<Finding> {
        let bs = self.block_size;
        let blocks_x = width / bs;
        let blocks_y = height / bs;
        let num_blocks = blocks_x * blocks_y;
        let radius = self.neighbor_radius;

        if num_blocks < 9 {
            return Vec::new();
        }

        let mut normalized = vec![0.0_f64; num_blocks * LBP_BINS];
        for b in 0..num_blocks {
            let src = &histograms[b * LBP_BINS..(b + 1) * LBP_BINS];
            let total: f64 = src.iter().map(|&v| v as f64).sum();
            if total > 0.0 {
                for i in 0..LBP_BINS {
                    normalized[b * LBP_BINS + i] = src[i] as f64 / total;
                }
            }
        }

        let mut block_scores = vec![0.0_f64; num_blocks];
        let mut block_active = vec![false; num_blocks];

        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                let block_idx = by * blocks_x + bx;
                let block_hist = &normalized[block_idx * LBP_BINS..(block_idx + 1) * LBP_BINS];
                let block_total: f64 = histograms[block_idx * LBP_BINS..(block_idx + 1) * LBP_BINS]
                    .iter()
                    .map(|&v| v as f64)
                    .sum();

                if block_total < 10.0 {
                    continue;
                }

                let mut chi_sum = 0.0_f64;
                let mut neighbor_count = 0;

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
                        let n_total: f64 = histograms[n_idx * LBP_BINS..(n_idx + 1) * LBP_BINS]
                            .iter()
                            .map(|&v| v as f64)
                            .sum();
                        if n_total < 10.0 {
                            continue;
                        }

                        let n_hist = &normalized[n_idx * LBP_BINS..(n_idx + 1) * LBP_BINS];
                        let chi_sq = chi_squared(block_hist, n_hist);
                        chi_sum += chi_sq;
                        neighbor_count += 1;
                    }
                }

                if neighbor_count > 0 {
                    block_scores[block_idx] = chi_sum / neighbor_count as f64;
                    block_active[block_idx] = true;
                }
            }
        }

        let active_scores: Vec<f64> = block_scores
            .iter()
            .zip(block_active.iter())
            .filter(|&(_, active)| *active)
            .map(|(&s, _)| s)
            .collect();

        if active_scores.len() < 9 {
            return Vec::new();
        }

        let mut sorted_scores = active_scores.clone();
        sorted_scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = sorted_scores[sorted_scores.len() / 2];
        let mad = {
            let mut deviations: Vec<f64> =
                active_scores.iter().map(|s| (s - median).abs()).collect();
            deviations.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            deviations[deviations.len() / 2] * 1.4826
        };

        if mad < 1e-6 {
            return Vec::new();
        }

        let mut anomalous_blocks: Vec<(usize, f64)> = Vec::new();
        for b in 0..num_blocks {
            if !block_active[b] {
                continue;
            }
            let z = (block_scores[b] - median) / mad;
            if z > self.anomaly_z_threshold {
                anomalous_blocks.push((b, z));
            }
        }

        if anomalous_blocks.len() < self.min_anomaly_blocks {
            return Vec::new();
        }

        let mut findings = Vec::new();
        let clusters = self.cluster_blocks(&anomalous_blocks, blocks_x, blocks_y);

        for (i, cluster) in clusters.iter().enumerate() {
            if cluster.len() < 2 {
                continue;
            }

            let mut min_x = usize::MAX;
            let mut min_y = usize::MAX;
            let mut max_x = 0_usize;
            let mut max_y = 0_usize;

            let max_z = cluster.iter().map(|&(_, z)| z).fold(f64::MIN, f64::max);

            for &(block_idx, _) in cluster {
                let bx = block_idx % blocks_x;
                let by = block_idx / blocks_x;
                min_x = min_x.min(bx * bs);
                min_y = min_y.min(by * bs);
                max_x = max_x.max((bx + 1) * bs);
                max_y = max_y.max((by + 1) * bs);
            }

            let severity = if max_z > 6.0 {
                Severity::High
            } else if max_z > 4.5 {
                Severity::Medium
            } else {
                Severity::Low
            };

            findings.push(
                Finding::new(
                    "texture_consistency",
                    "texture_lbp_inconsistency",
                    format!(
                        "Cluster of {} blocks shows micro-texture inconsistency \
                         (LBP chi-squared z-score {max_z:.1}) - region has different \
                         texture fingerprint from surroundings",
                        cluster.len(),
                    ),
                    severity,
                    (0.45 + (max_z - self.anomaly_z_threshold) * 0.08).min(0.85),
                )
                .with_region(Region::BoundingBox {
                    x: min_x as u32,
                    y: min_y as u32,
                    width: (max_x - min_x) as u32,
                    height: (max_y - min_y) as u32,
                }),
            );

            if i >= 4 {
                break;
            }
        }

        if clusters.iter().map(|c| c.len()).sum::<usize>() > 10 {
            let anomaly_ratio = anomalous_blocks.len() as f64 / active_scores.len() as f64;
            findings.push(Finding::new(
                "texture_consistency",
                "texture_widespread_inconsistency",
                format!(
                    "{:.1}% of image blocks show LBP texture anomalies - \
                     widespread micro-texture inconsistency indicates significant manipulation",
                    anomaly_ratio * 100.0,
                ),
                Severity::High,
                (0.55 + anomaly_ratio).min(0.85),
            ));
        }

        findings
    }

    fn cluster_blocks(
        &self,
        blocks: &[(usize, f64)],
        blocks_x: usize,
        blocks_y: usize,
    ) -> Vec<Vec<(usize, f64)>> {
        let num_blocks = blocks_x * blocks_y;
        let mut block_map = vec![false; num_blocks];
        let mut block_z = vec![0.0_f64; num_blocks];

        for &(idx, z) in blocks {
            if idx < num_blocks {
                block_map[idx] = true;
                block_z[idx] = z;
            }
        }

        let mut visited = vec![false; num_blocks];
        let mut clusters = Vec::new();

        for &(start_idx, _) in blocks {
            if visited[start_idx] {
                continue;
            }

            let mut cluster = Vec::new();
            let mut stack = vec![start_idx];

            while let Some(idx) = stack.pop() {
                if idx >= num_blocks || visited[idx] || !block_map[idx] {
                    continue;
                }
                visited[idx] = true;
                cluster.push((idx, block_z[idx]));

                let bx = idx % blocks_x;
                let by = idx / blocks_x;

                for dy in -1_i32..=1 {
                    for dx in -1_i32..=1 {
                        if dx == 0 && dy == 0 {
                            continue;
                        }
                        let nx = bx as i32 + dx;
                        let ny = by as i32 + dy;
                        if nx >= 0 && nx < blocks_x as i32 && ny >= 0 && ny < blocks_y as i32 {
                            let n_idx = ny as usize * blocks_x + nx as usize;
                            if !visited[n_idx] && block_map[n_idx] {
                                stack.push(n_idx);
                            }
                        }
                    }
                }
            }

            if !cluster.is_empty() {
                clusters.push(cluster);
            }
        }

        clusters.sort_by_key(|c| std::cmp::Reverse(c.len()));
        clusters
    }
}

/// Chi-squared distance between two normalized histograms.
fn chi_squared(h1: &[f64], h2: &[f64]) -> f64 {
    h1.iter()
        .zip(h2.iter())
        .map(|(&a, &b)| {
            let sum = a + b;
            if sum > 1e-10 {
                (a - b).powi(2) / sum
            } else {
                0.0
            }
        })
        .sum()
}
