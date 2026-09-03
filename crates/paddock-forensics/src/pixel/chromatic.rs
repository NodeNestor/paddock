//! Chromatic-aberration consistency with RANSAC (Johnson&Farid 2006, Gloe 2010),
//! ported verbatim from the CPU reference. CPU-only: the reference's device
//! path is a stub, and the sub-pixel/RANSAC pipeline is serial - `gpu()`
//! delegates to `cpu()`. The RANSAC here uses a fixed-seed LCG (seed 42), so it
//! is deterministic (unlike copy_move).
//!
//! Real lenses produce radial CA following a polynomial centered at the optical
//! axis. Per-block R-G/B-G lateral displacement is fit to displacement =
//! a·r²+b·r+c via RANSAC; blocks deviating from the model -> spliced from a
//! different lens; total absence of CA -> synthetic. Camera-specific -> skipped
//! for documents.

use crate::analyzer::Analyzer;
use crate::context::ContentType;
use crate::{Context, Finding, Severity};

pub struct ChromaticAberrationAnalyzer {
    block_size: usize,
    edge_threshold: f64,
    ransac_iterations: usize,
    ransac_inlier_threshold: f64,
}

impl Default for ChromaticAberrationAnalyzer {
    fn default() -> Self {
        Self {
            block_size: 64,
            edge_threshold: 20.0,
            ransac_iterations: 500,
            ransac_inlier_threshold: 0.3,
        }
    }
}

struct CaMeasurement {
    radius: f64,
    rg_displacement: f64,
    bg_displacement: f64,
    #[allow(dead_code)]
    edge_density: f64,
}

/// RANSAC-fitted radial CA model: displacement = a·r² + b·r + c.
struct CaModel {
    a: f64,
    b: f64,
    c: f64,
}

impl Analyzer for ChromaticAberrationAnalyzer {
    fn name(&self) -> &'static str {
        "chromatic_aberration"
    }

    fn applies_to(&self, ctx: &Context) -> bool {
        !matches!(ctx.content_type, ContentType::Document)
    }

    fn cpu(&self, ctx: &Context) -> Vec<Finding> {
        let w = ctx.width as usize;
        let h = ctx.height as usize;

        if w < self.block_size * 3 || h < self.block_size * 3 {
            return vec![];
        }

        let rgb = ctx.image.to_rgb8();
        let pixels = rgb.as_raw();

        let cx = w as f64 / 2.0;
        let cy = h as f64 / 2.0;
        let max_radius = (cx * cx + cy * cy).sqrt();

        let blocks_x = w / self.block_size;
        let blocks_y = h / self.block_size;

        let mut measurements: Vec<CaMeasurement> = Vec::new();

        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                let x0 = bx * self.block_size;
                let y0 = by * self.block_size;

                if let Some(m) = self.measure_block_ca(pixels, w, h, x0, y0, cx, cy)
                    && m.edge_density > 0.05
                {
                    measurements.push(m);
                }
            }
        }

        let mut findings = Vec::new();

        if measurements.len() < 6 {
            return findings;
        }

        let mean_rg: f64 = measurements
            .iter()
            .map(|m| m.rg_displacement.abs())
            .sum::<f64>()
            / measurements.len() as f64;
        let mean_bg: f64 = measurements
            .iter()
            .map(|m| m.bg_displacement.abs())
            .sum::<f64>()
            / measurements.len() as f64;

        if mean_rg < 0.01 && mean_bg < 0.01 {
            findings.push(Finding::new(
                "chromatic_aberration",
                "no_chromatic_aberration",
                format!(
                    "No detectable chromatic aberration (R-G: {mean_rg:.4}px, B-G: {mean_bg:.4}px) - \
                     inconsistent with real camera optics, suggests synthetic generation"
                ),
                Severity::Medium,
                0.55,
            ));
            return findings;
        }

        let rg_data: Vec<(f64, f64)> = measurements
            .iter()
            .map(|m| (m.radius / max_radius, m.rg_displacement))
            .collect();

        let (rg_model, rg_inliers) = self.ransac_fit_ca_model(&rg_data);

        let mut residuals: Vec<(usize, f64)> = Vec::new();
        for (i, m) in measurements.iter().enumerate() {
            let r = m.radius / max_radius;
            let predicted = rg_model.a * r * r + rg_model.b * r + rg_model.c;
            let residual = (m.rg_displacement - predicted).abs();
            residuals.push((i, residual));
        }

        let median_residual = {
            let mut sorted: Vec<f64> = residuals.iter().map(|&(_, r)| r).collect();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            sorted[sorted.len() / 2]
        };

        let outlier_threshold = (median_residual * 3.0).max(self.ransac_inlier_threshold);
        let outliers: Vec<usize> = residuals
            .iter()
            .filter(|&&(_, r)| r > outlier_threshold)
            .map(|&(i, _)| i)
            .collect();

        let outlier_ratio = outliers.len() as f64 / measurements.len() as f64;
        let inlier_ratio = rg_inliers as f64 / measurements.len() as f64;

        if outlier_ratio > 0.1 && outlier_ratio < 0.6 {
            findings.push(Finding::new(
                "chromatic_aberration",
                "chromatic_aberration_inconsistency",
                format!(
                    "Chromatic aberration inconsistency: {:.1}% of edge-rich blocks \
                     deviate from RANSAC-fitted radial CA model (a={:.4}, b={:.4}, \
                     {:.1}% inliers) - indicates spliced content from a different lens",
                    outlier_ratio * 100.0,
                    rg_model.a,
                    rg_model.b,
                    inlier_ratio * 100.0
                ),
                Severity::High,
                (0.55 + outlier_ratio * 0.4).min(0.85),
            ));
        }

        if inlier_ratio < 0.4 && measurements.len() >= 10 {
            findings.push(Finding::new(
                "chromatic_aberration",
                "ca_model_poor_fit",
                format!(
                    "Radial CA model fits only {:.1}% of blocks - chromatic aberration \
                     pattern does not follow any consistent lens model",
                    inlier_ratio * 100.0
                ),
                Severity::Medium,
                0.50,
            ));
        }

        findings
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

impl ChromaticAberrationAnalyzer {
    fn measure_block_ca(
        &self,
        pixels: &[u8],
        img_w: usize,
        img_h: usize,
        x0: usize,
        y0: usize,
        img_cx: f64,
        img_cy: f64,
    ) -> Option<CaMeasurement> {
        let bs = self.block_size;
        let block_cx = x0 as f64 + bs as f64 / 2.0;
        let block_cy = y0 as f64 + bs as f64 / 2.0;
        let rel_x = block_cx - img_cx;
        let rel_y = block_cy - img_cy;
        let radius = (rel_x * rel_x + rel_y * rel_y).sqrt();

        let mut r_gx = Vec::new();
        let mut r_gy = Vec::new();
        let mut g_gx = Vec::new();
        let mut g_gy = Vec::new();
        let mut b_gx = Vec::new();
        let mut b_gy = Vec::new();
        let mut edge_count = 0;

        for dy in 1..bs - 1 {
            for dx in 1..bs - 1 {
                let x = x0 + dx;
                let y = y0 + dy;
                if x + 1 >= img_w || y + 1 >= img_h {
                    continue;
                }

                for (ch, gx_vec, gy_vec) in [
                    (0, &mut r_gx, &mut r_gy),
                    (1, &mut g_gx, &mut g_gy),
                    (2, &mut b_gx, &mut b_gy),
                ] {
                    let gx = pixels[((y - 1) * img_w + (x + 1)) * 3 + ch] as f64
                        - pixels[((y - 1) * img_w + (x - 1)) * 3 + ch] as f64
                        + 2.0 * pixels[(y * img_w + (x + 1)) * 3 + ch] as f64
                        - 2.0 * pixels[(y * img_w + (x - 1)) * 3 + ch] as f64
                        + pixels[((y + 1) * img_w + (x + 1)) * 3 + ch] as f64
                        - pixels[((y + 1) * img_w + (x - 1)) * 3 + ch] as f64;

                    let gy = pixels[((y + 1) * img_w + (x - 1)) * 3 + ch] as f64
                        - pixels[((y - 1) * img_w + (x - 1)) * 3 + ch] as f64
                        + 2.0 * pixels[((y + 1) * img_w + x) * 3 + ch] as f64
                        - 2.0 * pixels[((y - 1) * img_w + x) * 3 + ch] as f64
                        + pixels[((y + 1) * img_w + (x + 1)) * 3 + ch] as f64
                        - pixels[((y - 1) * img_w + (x + 1)) * 3 + ch] as f64;

                    gx_vec.push(gx);
                    gy_vec.push(gy);

                    if ch == 1 {
                        let mag = (gx * gx + gy * gy).sqrt();
                        if mag > self.edge_threshold {
                            edge_count += 1;
                        }
                    }
                }
            }
        }

        let total = ((bs - 2) * (bs - 2)) as f64;
        let edge_density = edge_count as f64 / total;

        if r_gx.is_empty() || edge_density < 0.05 {
            return None;
        }

        let rg_dx = Self::subpixel_shift_1d(&r_gx, &g_gx);
        let rg_dy = Self::subpixel_shift_1d(&r_gy, &g_gy);
        let rg_displacement = (rg_dx * rg_dx + rg_dy * rg_dy).sqrt();

        let bg_dx = Self::subpixel_shift_1d(&b_gx, &g_gx);
        let bg_dy = Self::subpixel_shift_1d(&b_gy, &g_gy);
        let bg_displacement = (bg_dx * bg_dx + bg_dy * bg_dy).sqrt();

        Some(CaMeasurement {
            radius,
            rg_displacement,
            bg_displacement,
            edge_density,
        })
    }

    fn subpixel_shift_1d(a: &[f64], b: &[f64]) -> f64 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }

        let n = a.len() as f64;
        let mean_a: f64 = a.iter().sum::<f64>() / n;
        let mean_b: f64 = b.iter().sum::<f64>() / n;

        let mut cc0 = 0.0_f64;
        let mut cc1 = 0.0_f64;
        let mut cc_neg1 = 0.0_f64;

        for i in 0..a.len() {
            let da = a[i] - mean_a;
            let db = b[i] - mean_b;
            cc0 += da * db;
            if i + 1 < a.len() {
                cc1 += da * (b[i + 1] - mean_b);
            }
            if i > 0 {
                cc_neg1 += da * (b[i - 1] - mean_b);
            }
        }

        let denom = 2.0 * (2.0 * cc0 - cc1 - cc_neg1);
        if denom.abs() > 1e-10 {
            (cc_neg1 - cc1) / denom
        } else {
            0.0
        }
    }

    fn ransac_fit_ca_model(&self, data: &[(f64, f64)]) -> (CaModel, usize) {
        let n = data.len();
        if n < 3 {
            return (
                CaModel {
                    a: 0.0,
                    b: 0.0,
                    c: 0.0,
                },
                0,
            );
        }

        let mut best_model = CaModel {
            a: 0.0,
            b: 0.0,
            c: 0.0,
        };
        let mut best_inliers = 0;

        // Fixed-seed LCG -> reproducible RANSAC (deterministic).
        let mut rng_state = 42_u64;
        let next_rand = |state: &mut u64| -> usize {
            *state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((*state >> 33) as usize) % n
        };

        for _ in 0..self.ransac_iterations {
            let i0 = next_rand(&mut rng_state);
            let i1 = next_rand(&mut rng_state);
            let i2 = next_rand(&mut rng_state);

            if i0 == i1 || i1 == i2 || i0 == i2 {
                continue;
            }

            let model = Self::fit_quadratic(
                data[i0].0, data[i0].1, data[i1].0, data[i1].1, data[i2].0, data[i2].1,
            );

            let model = match model {
                Some(m) => m,
                None => continue,
            };

            let inliers = data
                .iter()
                .filter(|&&(r, d)| {
                    let predicted = model.a * r * r + model.b * r + model.c;
                    (d - predicted).abs() < self.ransac_inlier_threshold
                })
                .count();

            if inliers > best_inliers {
                best_inliers = inliers;
                best_model = model;
            }
        }

        let inlier_data: Vec<(f64, f64)> = data
            .iter()
            .filter(|&&(r, d)| {
                let predicted = best_model.a * r * r + best_model.b * r + best_model.c;
                (d - predicted).abs() < self.ransac_inlier_threshold
            })
            .copied()
            .collect();

        if inlier_data.len() >= 3
            && let Some(refined) = Self::fit_quadratic_least_squares(&inlier_data)
        {
            best_model = refined;
        }

        (best_model, best_inliers)
    }

    fn fit_quadratic(r0: f64, d0: f64, r1: f64, d1: f64, r2: f64, d2: f64) -> Option<CaModel> {
        let det = r0 * r0 * (r1 - r2) - r1 * r1 * (r0 - r2) + r2 * r2 * (r0 - r1);
        if det.abs() < 1e-12 {
            return None;
        }

        let a = (d0 * (r1 - r2) - d1 * (r0 - r2) + d2 * (r0 - r1)) / det;
        let b =
            (d0 * (r2 * r2 - r1 * r1) + d1 * (r0 * r0 - r2 * r2) + d2 * (r1 * r1 - r0 * r0)) / det;
        let c = (d0 * (r1 * r1 * r2 - r2 * r2 * r1)
            + d1 * (r2 * r2 * r0 - r0 * r0 * r2)
            + d2 * (r0 * r0 * r1 - r1 * r1 * r0))
            / det;

        Some(CaModel { a, b, c })
    }

    fn fit_quadratic_least_squares(data: &[(f64, f64)]) -> Option<CaModel> {
        let n = data.len() as f64;
        if n < 3.0 {
            return None;
        }

        let mut s0 = 0.0_f64;
        let mut s1 = 0.0_f64;
        let mut s2 = 0.0_f64;
        let mut s3 = 0.0_f64;
        let mut s4 = 0.0_f64;
        let mut sd0 = 0.0_f64;
        let mut sd1 = 0.0_f64;
        let mut sd2 = 0.0_f64;

        for &(r, d) in data {
            let r2 = r * r;
            s0 += 1.0;
            s1 += r;
            s2 += r2;
            s3 += r2 * r;
            s4 += r2 * r2;
            sd0 += d;
            sd1 += r * d;
            sd2 += r2 * d;
        }

        let det = s4 * (s2 * s0 - s1 * s1) - s3 * (s3 * s0 - s1 * s2) + s2 * (s3 * s1 - s2 * s2);
        if det.abs() < 1e-15 {
            return None;
        }

        let a = (sd2 * (s2 * s0 - s1 * s1) - s3 * (sd1 * s0 - s1 * sd0)
            + s2 * (sd1 * s1 - s2 * sd0))
            / det;
        let b = (s4 * (sd1 * s0 - s1 * sd0) - sd2 * (s3 * s0 - s1 * s2)
            + s2 * (s3 * sd0 - sd1 * s2))
            / det;
        let c = (s4 * (s2 * sd0 - sd1 * s1) - s3 * (s3 * sd0 - sd1 * s2)
            + sd2 * (s3 * s1 - s2 * s2))
            / det;

        Some(CaModel { a, b, c })
    }
}
