//! Anti-forensics detection, ported from the reference. Canonical = CPU; a bit-exact
//! CUDA kernel (`anti_forensics_block`, one thread per block, f64,
//! `--fmad=false`) computes the identical per-block difference variance,
//! median-residual kurtosis, and histogram flatness, then the same host-side
//! low/high MAD-outlier + Gaussian-kurtosis logic runs on both paths. (Not a
//! copy of the reference's divergent device path.)
//!
//! Detects counter-forensic tampering: median filtering (low diff variance),
//! noise injection (kurtosis ≈ 3.0 Gaussian), and histogram equalization (flat
//! local histograms). Photo-oriented -> skipped for PDFs.

use crate::analyzer::Analyzer;
use crate::{Context, Finding, Region, Severity};

pub struct AntiForensicsDetector {
    block_size: usize,
    anomaly_z_threshold: f64,
    min_anomaly_blocks: usize,
}

impl Default for AntiForensicsDetector {
    fn default() -> Self {
        Self {
            block_size: 32,
            anomaly_z_threshold: 3.0,
            min_anomaly_blocks: 3,
        }
    }
}

struct BlockAntiForensics {
    diff_variance: f64,
    noise_kurtosis: f64,
    hist_flatness: f64,
    pixel_count: usize,
}

impl Analyzer for AntiForensicsDetector {
    fn name(&self) -> &'static str {
        "anti_forensics"
    }

    fn applies_to(&self, ctx: &Context) -> bool {
        !ctx.is_pdf()
    }

    fn cpu(&self, ctx: &Context) -> Vec<Finding> {
        let (width, height) = (ctx.width as usize, ctx.height as usize);
        if width < self.block_size * 4 || height < self.block_size * 4 {
            return Vec::new();
        }
        let blocks = self.compute_cpu(ctx.gray(), width, height);
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

        let stream = gpu.stream();
        let d_gray = stream.clone_htod(ctx.gray())?;
        let mut d_dv = stream.alloc_zeros::<f64>(total)?;
        let mut d_ku = stream.alloc_zeros::<f64>(total)?;
        let mut d_fl = stream.alloc_zeros::<f64>(total)?;
        let mut d_pc = stream.alloc_zeros::<i32>(total)?;

        let (w_u, h_u, bs_u) = (width as u32, height as u32, bs as u32);
        let (bx_u, by_u) = (blocks_x as u32, blocks_y as u32);
        let f = gpu.function("anti_forensics", "anti_forensics_block")?;
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
                .arg(&mut d_dv)
                .arg(&mut d_ku)
                .arg(&mut d_fl)
                .arg(&mut d_pc)
                .arg(&w_u)
                .arg(&h_u)
                .arg(&bs_u)
                .arg(&bx_u)
                .arg(&by_u)
                .launch(cfg)?;
        }
        let dv: Vec<f64> = stream.clone_dtoh(&d_dv)?;
        let ku: Vec<f64> = stream.clone_dtoh(&d_ku)?;
        let fl: Vec<f64> = stream.clone_dtoh(&d_fl)?;
        let pc: Vec<i32> = stream.clone_dtoh(&d_pc)?;
        stream.synchronize()?;

        let blocks: Vec<BlockAntiForensics> = (0..total)
            .map(|i| BlockAntiForensics {
                diff_variance: dv[i],
                noise_kurtosis: ku[i],
                hist_flatness: fl[i],
                pixel_count: pc[i] as usize,
            })
            .collect();
        Ok(self.analyze_blocks(&blocks, width, height))
    }
}

impl AntiForensicsDetector {
    fn compute_cpu(&self, gray: &[u8], width: usize, height: usize) -> Vec<BlockAntiForensics> {
        let bs = self.block_size;
        let blocks_x = width / bs;
        let blocks_y = height / bs;
        let mut blocks = Vec::with_capacity(blocks_x * blocks_y);

        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                let x0 = bx * bs;
                let y0 = by * bs;

                let mut diff_sum = 0.0_f64;
                let mut diff_sq = 0.0_f64;
                let mut diff_count = 0_usize;
                let mut res_sum = 0.0_f64;
                let mut res_sq = 0.0_f64;
                let mut res_4th = 0.0_f64;
                let mut res_count = 0_usize;
                let mut hist = [0u32; 32];

                for dy in 1..bs.saturating_sub(1) {
                    for dx in 1..bs.saturating_sub(1) {
                        let x = x0 + dx;
                        let y = y0 + dy;
                        if x >= width - 1 || y >= height - 1 {
                            continue;
                        }

                        let center = gray[y * width + x] as f64 / 255.0;

                        let dh = gray[y * width + x + 1] as f64 / 255.0 - center;
                        let dv = gray[(y + 1) * width + x] as f64 / 255.0 - center;
                        diff_sum += dh + dv;
                        diff_sq += dh * dh + dv * dv;
                        diff_count += 2;

                        // 3×3 median.
                        let mut v: Vec<u8> = Vec::with_capacity(9);
                        for ky in -1_i32..=1 {
                            for kx in -1_i32..=1 {
                                v.push(
                                    gray[(y as i32 + ky) as usize * width
                                        + (x as i32 + kx) as usize],
                                );
                            }
                        }
                        v.sort_unstable();
                        let residual = gray[y * width + x] as f64 / 255.0 - v[4] as f64 / 255.0;
                        res_sum += residual;
                        res_sq += residual * residual;
                        res_4th += residual.powi(4);
                        res_count += 1;

                        let bin = ((center * 31.999) as usize).min(31);
                        hist[bin] += 1;
                    }
                }

                let diff_var = if diff_count > 1 {
                    let mean = diff_sum / diff_count as f64;
                    diff_sq / diff_count as f64 - mean * mean
                } else {
                    0.0
                };

                let kurtosis = if res_count > 3 {
                    let var = res_sq / res_count as f64 - (res_sum / res_count as f64).powi(2);
                    if var > 1e-10 {
                        (res_4th / res_count as f64) / (var * var)
                    } else {
                        0.0
                    }
                } else {
                    0.0
                };

                let total: u32 = hist.iter().sum();
                let max_h = *hist.iter().max().unwrap_or(&1) as f64;
                let flatness = if total > 0 && max_h > 0.0 {
                    (total as f64 / 32.0) / max_h
                } else {
                    0.0
                };

                blocks.push(BlockAntiForensics {
                    diff_variance: diff_var,
                    noise_kurtosis: kurtosis,
                    hist_flatness: flatness,
                    pixel_count: res_count,
                });
            }
        }

        blocks
    }

    fn analyze_blocks(
        &self,
        blocks: &[BlockAntiForensics],
        width: usize,
        _height: usize,
    ) -> Vec<Finding> {
        let bs = self.block_size;
        let blocks_x = width / bs;
        let mut findings = Vec::new();

        let active: Vec<(usize, &BlockAntiForensics)> = blocks
            .iter()
            .enumerate()
            .filter(|(_, b)| b.pixel_count > 10)
            .collect();

        if active.len() < 9 {
            return findings;
        }

        // 1. Median filter: unusually LOW diff_variance.
        self.detect_low_outliers(
            &active,
            |b| b.diff_variance,
            blocks_x,
            bs,
            "anti_forensics_median_filter",
            "unusually low first-order difference variance",
            "median filtering applied to hide JPEG splice artifacts",
            &mut findings,
        );

        // 2. Noise injection: kurtosis near 3.0 (Gaussian).
        let gaussian_blocks: Vec<(usize, f64)> = active
            .iter()
            .filter(|(_, b)| (b.noise_kurtosis - 3.0).abs() < 0.5 && b.noise_kurtosis > 0.0)
            .map(|&(i, b)| (i, (b.noise_kurtosis - 3.0).abs()))
            .collect();

        let gaussian_ratio = gaussian_blocks.len() as f64 / active.len() as f64;
        if gaussian_ratio > 0.15 && gaussian_blocks.len() >= self.min_anomaly_blocks {
            let (min_x, min_y, max_x, max_y) = bbox(&gaussian_blocks, blocks_x, bs);
            findings.push(
                Finding::new(
                    "anti_forensics",
                    "anti_forensics_noise_injection",
                    format!(
                        "{} blocks ({:.1}%) show Gaussian noise characteristics (kurtosis ≈ 3.0) - \
                         possible artificial noise injection to mask manipulation traces",
                        gaussian_blocks.len(),
                        gaussian_ratio * 100.0,
                    ),
                    Severity::High,
                    (0.45 + gaussian_ratio * 0.8).min(0.80),
                )
                .with_region(Region::BoundingBox {
                    x: min_x as u32,
                    y: min_y as u32,
                    width: (max_x - min_x) as u32,
                    height: (max_y - min_y) as u32,
                }),
            );
        }

        // 3. Histogram equalization: unusually high flatness.
        self.detect_high_outliers(
            &active,
            |b| b.hist_flatness,
            blocks_x,
            bs,
            "anti_forensics_histogram_equalization",
            "unnaturally flat local histogram",
            "histogram equalization applied to mask statistical anomalies",
            &mut findings,
        );

        findings
    }

    #[allow(clippy::too_many_arguments)]
    fn detect_low_outliers(
        &self,
        active: &[(usize, &BlockAntiForensics)],
        metric: impl Fn(&BlockAntiForensics) -> f64,
        blocks_x: usize,
        bs: usize,
        code: &str,
        metric_desc: &str,
        implication: &str,
        findings: &mut Vec<Finding>,
    ) {
        let values: Vec<f64> = active.iter().map(|(_, b)| metric(b)).collect();
        let (median, mad) = robust_stats(&values);
        if mad < 1e-10 {
            return;
        }

        let anomalous: Vec<(usize, f64)> = active
            .iter()
            .filter_map(|&(i, b)| {
                let z = (median - metric(b)) / mad; // negative z for LOW outliers
                if z > self.anomaly_z_threshold {
                    Some((i, z))
                } else {
                    None
                }
            })
            .collect();

        if anomalous.len() >= self.min_anomaly_blocks {
            let max_z = anomalous.iter().fold(0.0_f64, |m, &(_, z)| m.max(z));
            let (min_x, min_y, max_x, max_y) = bbox(&anomalous, blocks_x, bs);
            findings.push(
                Finding::new(
                    "anti_forensics",
                    code,
                    format!(
                        "{} blocks show {metric_desc} (z-score {max_z:.1}) - {implication}",
                        anomalous.len(),
                    ),
                    if max_z > 5.0 {
                        Severity::High
                    } else {
                        Severity::Medium
                    },
                    (0.50 + (max_z - self.anomaly_z_threshold) * 0.08).min(0.85),
                )
                .with_region(Region::BoundingBox {
                    x: min_x as u32,
                    y: min_y as u32,
                    width: (max_x - min_x) as u32,
                    height: (max_y - min_y) as u32,
                }),
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn detect_high_outliers(
        &self,
        active: &[(usize, &BlockAntiForensics)],
        metric: impl Fn(&BlockAntiForensics) -> f64,
        blocks_x: usize,
        bs: usize,
        code: &str,
        metric_desc: &str,
        implication: &str,
        findings: &mut Vec<Finding>,
    ) {
        let values: Vec<f64> = active.iter().map(|(_, b)| metric(b)).collect();
        let (median, mad) = robust_stats(&values);
        if mad < 1e-10 {
            return;
        }

        let anomalous: Vec<(usize, f64)> = active
            .iter()
            .filter_map(|&(i, b)| {
                let z = (metric(b) - median) / mad;
                if z > self.anomaly_z_threshold {
                    Some((i, z))
                } else {
                    None
                }
            })
            .collect();

        if anomalous.len() >= self.min_anomaly_blocks {
            let max_z = anomalous.iter().fold(0.0_f64, |m, &(_, z)| m.max(z));
            let (min_x, min_y, max_x, max_y) = bbox(&anomalous, blocks_x, bs);
            findings.push(
                Finding::new(
                    "anti_forensics",
                    code,
                    format!(
                        "{} blocks show {metric_desc} (z-score {max_z:.1}) - {implication}",
                        anomalous.len(),
                    ),
                    if max_z > 5.0 {
                        Severity::High
                    } else {
                        Severity::Medium
                    },
                    (0.50 + (max_z - self.anomaly_z_threshold) * 0.08).min(0.85),
                )
                .with_region(Region::BoundingBox {
                    x: min_x as u32,
                    y: min_y as u32,
                    width: (max_x - min_x) as u32,
                    height: (max_y - min_y) as u32,
                }),
            );
        }
    }
}

fn robust_stats(values: &[f64]) -> (f64, f64) {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = sorted[sorted.len() / 2];
    let mut devs: Vec<f64> = sorted.iter().map(|v| (v - median).abs()).collect();
    devs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mad = devs[devs.len() / 2] * 1.4826;
    (median, mad)
}

fn bbox(blocks: &[(usize, f64)], blocks_x: usize, bs: usize) -> (usize, usize, usize, usize) {
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
