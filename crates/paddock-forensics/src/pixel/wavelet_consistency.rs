//! Wavelet subband consistency, ported from the reference. Canonical = CPU; a
//! bit-exact CUDA kernel (`wavelet_subband_block`, one thread per block, f64,
//! `--fmad=false`) computes the identical per-block LL/LH/HL/HH energies, then
//! the same host-side metric + MAD logic runs on both paths. (Not a copy of
//! the reference's divergent device path.)
//!
//! Measures energy distribution across Haar subbands per block: AI inpainting
//! suppresses HH (diagonal) detail; resampling shifts the detail ratio; splices
//! shift directional balance. Photo-oriented -> skipped for PDFs.

use crate::analyzer::Analyzer;
use crate::{Context, Finding, Region, Severity};

pub struct WaveletConsistencyAnalyzer {
    block_size: usize,
    neighbor_radius: usize,
    anomaly_z_threshold: f64,
    min_anomaly_blocks: usize,
}

impl Default for WaveletConsistencyAnalyzer {
    fn default() -> Self {
        Self {
            block_size: 32,
            neighbor_radius: 2,
            anomaly_z_threshold: 3.0,
            min_anomaly_blocks: 3,
        }
    }
}

struct SubbandEnergy {
    ll: f64,
    lh: f64,
    hl: f64,
    hh: f64,
}

impl SubbandEnergy {
    fn detail_total(&self) -> f64 {
        self.lh + self.hl + self.hh
    }

    fn hh_ratio(&self) -> f64 {
        let total = self.detail_total();
        if total > 1e-10 { self.hh / total } else { 0.0 }
    }

    fn detail_ratio(&self) -> f64 {
        if self.ll > 1e-10 {
            self.detail_total() / self.ll
        } else {
            0.0
        }
    }

    fn directional_balance(&self) -> f64 {
        let sum = self.lh + self.hl;
        if sum > 1e-10 {
            (self.lh - self.hl).abs() / sum
        } else {
            0.0
        }
    }
}

impl Analyzer for WaveletConsistencyAnalyzer {
    fn name(&self) -> &'static str {
        "wavelet_consistency"
    }

    fn applies_to(&self, ctx: &Context) -> bool {
        !ctx.is_pdf()
    }

    fn cpu(&self, ctx: &Context) -> Vec<Finding> {
        let (width, height) = (ctx.width as usize, ctx.height as usize);
        if width < self.block_size * 4 || height < self.block_size * 4 {
            return Vec::new();
        }
        let energies = self.compute_wavelet_cpu(ctx.gray(), width, height);
        self.analyze_energies(&energies, width, height)
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
        let mut d_ll = stream.alloc_zeros::<f64>(total)?;
        let mut d_lh = stream.alloc_zeros::<f64>(total)?;
        let mut d_hl = stream.alloc_zeros::<f64>(total)?;
        let mut d_hh = stream.alloc_zeros::<f64>(total)?;

        let (w_u, h_u, bs_u) = (width as u32, height as u32, bs as u32);
        let (bx_u, by_u) = (blocks_x as u32, blocks_y as u32);
        let f = gpu.function("wavelet_consistency", "wavelet_subband_block")?;
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
                .arg(&mut d_ll)
                .arg(&mut d_lh)
                .arg(&mut d_hl)
                .arg(&mut d_hh)
                .arg(&w_u)
                .arg(&h_u)
                .arg(&bs_u)
                .arg(&bx_u)
                .arg(&by_u)
                .launch(cfg)?;
        }
        let ll: Vec<f64> = stream.clone_dtoh(&d_ll)?;
        let lh: Vec<f64> = stream.clone_dtoh(&d_lh)?;
        let hl: Vec<f64> = stream.clone_dtoh(&d_hl)?;
        let hh: Vec<f64> = stream.clone_dtoh(&d_hh)?;
        stream.synchronize()?;

        let energies: Vec<SubbandEnergy> = (0..total)
            .map(|i| SubbandEnergy {
                ll: ll[i],
                lh: lh[i],
                hl: hl[i],
                hh: hh[i],
            })
            .collect();
        Ok(self.analyze_energies(&energies, width, height))
    }
}

impl WaveletConsistencyAnalyzer {
    fn compute_wavelet_cpu(&self, gray: &[u8], width: usize, height: usize) -> Vec<SubbandEnergy> {
        let bs = self.block_size;
        let blocks_x = width / bs;
        let blocks_y = height / bs;
        let half = bs / 2;
        let mut energies = Vec::with_capacity(blocks_x * blocks_y);

        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                let x0 = bx * bs;
                let y0 = by * bs;
                let mut ll_sum = 0.0_f64;
                let mut lh_sum = 0.0_f64;
                let mut hl_sum = 0.0_f64;
                let mut hh_sum = 0.0_f64;
                let mut count = 0;

                for py in 0..half {
                    for px in 0..half {
                        let x = x0 + px * 2;
                        let y = y0 + py * 2;
                        if x + 1 >= width || y + 1 >= height {
                            continue;
                        }

                        let a = gray[y * width + x] as f64 / 255.0;
                        let b = gray[y * width + x + 1] as f64 / 255.0;
                        let c = gray[(y + 1) * width + x] as f64 / 255.0;
                        let d = gray[(y + 1) * width + x + 1] as f64 / 255.0;

                        let ll = (a + b + c + d) * 0.5;
                        let lh = (a + b - c - d) * 0.5;
                        let hl = (a - b + c - d) * 0.5;
                        let hh = (a - b - c + d) * 0.5;

                        ll_sum += ll * ll;
                        lh_sum += lh * lh;
                        hl_sum += hl * hl;
                        hh_sum += hh * hh;
                        count += 1;
                    }
                }

                let n = count.max(1) as f64;
                energies.push(SubbandEnergy {
                    ll: ll_sum / n,
                    lh: lh_sum / n,
                    hl: hl_sum / n,
                    hh: hh_sum / n,
                });
            }
        }

        energies
    }

    fn analyze_energies(
        &self,
        energies: &[SubbandEnergy],
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
            .filter(|&b| energies[b].detail_total() > 1e-8)
            .collect();

        if active.len() < 9 {
            return findings;
        }

        type MetricCheck<'a> = (&'a dyn Fn(&SubbandEnergy) -> f64, &'a str, &'a str, &'a str);
        let checks: &[MetricCheck] = &[
            (
                &|e| e.hh_ratio(),
                "wavelet_hh_suppression",
                "HH (diagonal detail) energy suppression",
                "possible AI inpainting or diffusion model artifacts",
            ),
            (
                &|e| e.detail_ratio(),
                "wavelet_detail_anomaly",
                "detail-to-approximation energy ratio anomaly",
                "possible resampling or frequency content manipulation",
            ),
            (
                &|e| e.directional_balance(),
                "wavelet_directional_anomaly",
                "directional texture balance anomaly",
                "possible splice from source with different texture orientation",
            ),
        ];

        for &(metric_fn, code, metric_name, implication) in checks {
            self.detect_anomalies(
                energies,
                &active,
                blocks_x,
                blocks_y,
                radius,
                metric_fn,
                code,
                metric_name,
                implication,
                &mut findings,
            );
        }

        findings
    }

    #[allow(clippy::too_many_arguments)]
    fn detect_anomalies(
        &self,
        energies: &[SubbandEnergy],
        active: &[usize],
        blocks_x: usize,
        blocks_y: usize,
        radius: usize,
        metric_fn: &dyn Fn(&SubbandEnergy) -> f64,
        code: &str,
        metric_name: &str,
        implication: &str,
        findings: &mut Vec<Finding>,
    ) {
        let num_blocks = blocks_x * blocks_y;
        let mut deviations: Vec<(usize, f64)> = Vec::new();

        for &b in active {
            let bx = b % blocks_x;
            let by = b / blocks_x;
            let block_metric = metric_fn(&energies[b]);

            let mut neighbor_sum = 0.0_f64;
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
                    if n_idx < num_blocks && energies[n_idx].detail_total() > 1e-8 {
                        neighbor_sum += metric_fn(&energies[n_idx]);
                        n_count += 1;
                    }
                }
            }

            if n_count >= 3 {
                let neighbor_avg = neighbor_sum / n_count as f64;
                let deviation = (block_metric - neighbor_avg).abs();
                deviations.push((b, deviation));
            }
        }

        if deviations.len() < 9 {
            return;
        }

        let mut sorted: Vec<f64> = deviations.iter().map(|&(_, d)| d).collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = sorted[sorted.len() / 2];
        let mad = {
            let mut devs: Vec<f64> = sorted.iter().map(|d| (d - median).abs()).collect();
            devs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            devs[devs.len() / 2] * 1.4826
        };

        if mad < 1e-8 {
            return;
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
            return;
        }

        let max_z = anomalous.iter().fold(0.0_f64, |m, &(_, z)| m.max(z));
        let bs = self.block_size;

        let Some(strongest) = anomalous
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        else {
            return;
        };
        let sx = (strongest.0 % blocks_x) * bs;
        let sy = (strongest.0 / blocks_x) * bs;

        findings.push(
            Finding::new(
                "wavelet_consistency",
                code,
                format!(
                    "{} blocks show {metric_name} (z-score {max_z:.1}) - {implication}",
                    anomalous.len(),
                ),
                if max_z > 5.0 {
                    Severity::High
                } else {
                    Severity::Medium
                },
                (0.40 + (max_z - self.anomaly_z_threshold) * 0.07).min(0.80),
            )
            .with_region(Region::BoundingBox {
                x: sx as u32,
                y: sy as u32,
                width: bs as u32,
                height: bs as u32,
            }),
        );
    }
}
