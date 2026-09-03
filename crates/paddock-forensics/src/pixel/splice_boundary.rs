//! Splice-boundary detection, ported verbatim from the CPU reference.
//! CPU-only: the reference's splice-boundary GPU path is a CPU-fallback stub, so
//! `gpu()` delegates to `cpu()`.
//!
//! Sobel-detects strong edges, then for each edge pixel compares the two sides
//! for noise-variance, JPEG-blocking, and CFA-residual discontinuity; edges with
//! ≥2 indicators cluster into candidate splice boundaries. Uses `content_type`
//! to tighten the suspicious-pixel percentile on documents (text edges are
//! uniformly "suspicious"). Photo/document raster only: skipped for PDFs.

use crate::analyzer::Analyzer;
use crate::context::ContentType;
use crate::{Context, Finding, Region, Severity};

pub struct SpliceBoundaryDetector {
    analysis_radius: usize,
    edge_threshold: f64,
    noise_ratio_threshold: f64,
    suspicious_edge_ratio: f64,
}

impl Default for SpliceBoundaryDetector {
    fn default() -> Self {
        Self {
            analysis_radius: 8,
            edge_threshold: 60.0,
            noise_ratio_threshold: 2.0,
            suspicious_edge_ratio: 0.03,
        }
    }
}

struct EdgeAnalysis {
    x: usize,
    y: usize,
    noise_ratio: f64,
    blocking_diff: f64,
    cfa_diff: f64,
}

struct SpatialCluster {
    x_min: usize,
    y_min: usize,
    x_max: usize,
    y_max: usize,
    count: usize,
}

impl Analyzer for SpliceBoundaryDetector {
    fn name(&self) -> &'static str {
        "splice_boundary"
    }

    fn applies_to(&self, ctx: &Context) -> bool {
        !ctx.is_pdf()
    }

    fn cpu(&self, ctx: &Context) -> Vec<Finding> {
        let w = ctx.width as usize;
        let h = ctx.height as usize;

        let min_dim = self.analysis_radius * 4;
        if w < min_dim || h < min_dim {
            return vec![];
        }

        let gray = ctx.gray();
        let rgb = ctx.image.to_rgb8();
        let rgb_pixels = rgb.as_raw();

        let (gradient_mag, gradient_dir) = self.sobel_edge_detect(gray, w, h);

        let margin = self.analysis_radius + 1;
        let mut edge_pixels: Vec<(usize, usize, f64)> = Vec::new();

        for y in margin..h.saturating_sub(margin) {
            for x in margin..w.saturating_sub(margin) {
                let mag = gradient_mag[y * w + x];
                if mag > self.edge_threshold {
                    edge_pixels.push((x, y, gradient_dir[y * w + x]));
                }
            }
        }

        if edge_pixels.is_empty() {
            return vec![];
        }

        let max_edge_pixels = 5000;
        let step = if edge_pixels.len() > max_edge_pixels {
            edge_pixels.len() / max_edge_pixels
        } else {
            1
        };

        let mut analyses: Vec<EdgeAnalysis> = Vec::new();

        for i in (0..edge_pixels.len()).step_by(step) {
            let (x, y, direction) = edge_pixels[i];
            if let Some(analysis) = self.analyze_edge_pixel(gray, rgb_pixels, w, h, x, y, direction)
            {
                analyses.push(analysis);
            }
        }

        if analyses.is_empty() {
            return vec![];
        }

        let mut noise_suspicious = 0;
        let mut blocking_suspicious = 0;
        let mut cfa_suspicious = 0;
        let mut multi_indicator = 0;
        let mut scored_positions: Vec<(usize, usize, f64)> = Vec::new();

        for a in &analyses {
            let mut indicators = 0;
            let mut score = 0.0_f64;

            if a.noise_ratio > self.noise_ratio_threshold
                || a.noise_ratio < 1.0 / self.noise_ratio_threshold
            {
                noise_suspicious += 1;
                indicators += 1;
                score += (a.noise_ratio - 1.0).abs().min(10.0);
            }

            if a.blocking_diff > 0.3 {
                blocking_suspicious += 1;
                indicators += 1;
                score += a.blocking_diff;
            }

            if a.cfa_diff > 0.4 {
                cfa_suspicious += 1;
                indicators += 1;
                score += a.cfa_diff;
            }

            if indicators >= 2 {
                multi_indicator += 1;
                scored_positions.push((a.x, a.y, score));
            }
        }

        // Documents: text edges are uniformly suspicious -> keep only top 10%.
        // Photos: keep top 25%.
        let is_document = matches!(ctx.content_type, ContentType::Document);
        let suspicious_positions: Vec<(usize, usize, f64)> = if scored_positions.len() > 20 {
            let mut scores: Vec<f64> = scored_positions.iter().map(|p| p.2).collect();
            scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let percentile = if is_document { 90 } else { 75 };
            let threshold = scores[scores.len() * percentile / 100];
            scored_positions
                .iter()
                .filter(|p| p.2 >= threshold)
                .copied()
                .collect()
        } else {
            scored_positions.clone()
        };

        let total_analyzed = analyses.len() as f64;
        let noise_ratio = noise_suspicious as f64 / total_analyzed;
        let blocking_ratio = blocking_suspicious as f64 / total_analyzed;
        let cfa_ratio = cfa_suspicious as f64 / total_analyzed;
        let multi_ratio = multi_indicator as f64 / total_analyzed;

        let mut findings = Vec::new();

        let positions_xy: Vec<(usize, usize)> =
            suspicious_positions.iter().map(|p| (p.0, p.1)).collect();
        let mut clusters = Self::cluster_positions(&positions_xy, w, h);

        if is_document {
            clusters.sort_by(|a, b| b.count.cmp(&a.count));
            clusters.truncate(10);
        }

        if multi_ratio > self.suspicious_edge_ratio {
            let severity = if multi_ratio > 0.15 {
                Severity::Critical
            } else if multi_ratio > 0.08 {
                Severity::High
            } else {
                Severity::Medium
            };

            let max_score = suspicious_positions
                .iter()
                .map(|p| p.2)
                .fold(0.0_f64, f64::max);

            if clusters.is_empty() {
                findings.push(Finding::new(
                    "splice_boundary",
                    "splice_boundary_detected",
                    format!(
                        "Splice boundary detected: {:.1}% of strong edges show multi-indicator \
                         forensic discontinuity (noise: {:.1}%, blocking: {:.1}%, CFA: {:.1}%) - \
                         distinct image regions with different provenance are joined",
                        multi_ratio * 100.0,
                        noise_ratio * 100.0,
                        blocking_ratio * 100.0,
                        cfa_ratio * 100.0
                    ),
                    severity,
                    (0.5 + multi_ratio * 3.0).min(0.90),
                ));
            } else {
                for (i, cluster) in clusters.iter().enumerate() {
                    let cluster_scores: Vec<f64> = suspicious_positions
                        .iter()
                        .filter(|(x, y, _)| {
                            *x >= cluster.x_min
                                && *x <= cluster.x_max
                                && *y >= cluster.y_min
                                && *y <= cluster.y_max
                        })
                        .map(|(_, _, s)| *s)
                        .collect();

                    let cluster_mean_score = if cluster_scores.is_empty() {
                        0.5
                    } else {
                        cluster_scores.iter().sum::<f64>() / cluster_scores.len() as f64
                    };

                    let relative_strength = if max_score > 0.0 {
                        cluster_mean_score / max_score
                    } else {
                        0.5
                    };
                    let confidence = (0.4 + relative_strength * 0.5).min(0.95);

                    findings.push(
                        Finding::new(
                            "splice_boundary",
                            "splice_boundary_detected",
                            format!(
                                "Splice boundary region {} of {}: {} suspicious edge pixels at ({},{})->({},{}) \
                                 (noise: {:.1}%, blocking: {:.1}%, CFA: {:.1}%)",
                                i + 1,
                                clusters.len(),
                                cluster.count,
                                cluster.x_min,
                                cluster.y_min,
                                cluster.x_max,
                                cluster.y_max,
                                noise_ratio * 100.0,
                                blocking_ratio * 100.0,
                                cfa_ratio * 100.0
                            ),
                            severity,
                            confidence,
                        )
                        .with_region(Region::BoundingBox {
                            x: cluster.x_min as u32,
                            y: cluster.y_min as u32,
                            width: (cluster.x_max - cluster.x_min + 1) as u32,
                            height: (cluster.y_max - cluster.y_min + 1) as u32,
                        }),
                    );
                }
            }
        }

        if noise_ratio > self.suspicious_edge_ratio * 2.0
            && multi_ratio <= self.suspicious_edge_ratio
        {
            let region = clusters.first().map(|c| Region::BoundingBox {
                x: c.x_min as u32,
                y: c.y_min as u32,
                width: (c.x_max - c.x_min + 1) as u32,
                height: (c.y_max - c.y_min + 1) as u32,
            });

            let mut finding = Finding::new(
                "splice_boundary",
                "splice_boundary_detected",
                format!(
                    "Noise variance transitions detected along {:.1}% of strong edges - \
                     possible splice boundary where regions with different noise \
                     characteristics meet",
                    noise_ratio * 100.0
                ),
                Severity::Medium,
                (0.4 + noise_ratio * 2.0).min(0.75),
            );
            if let Some(region) = region {
                finding = finding.with_region(region);
            }
            findings.push(finding);
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

impl SpliceBoundaryDetector {
    fn sobel_edge_detect(&self, gray: &[u8], w: usize, h: usize) -> (Vec<f64>, Vec<f64>) {
        let mut magnitude = vec![0.0_f64; w * h];
        let mut direction = vec![0.0_f64; w * h];

        for y in 1..h - 1 {
            for x in 1..w - 1 {
                let p00 = gray[(y - 1) * w + (x - 1)] as f64;
                let p01 = gray[(y - 1) * w + x] as f64;
                let p02 = gray[(y - 1) * w + (x + 1)] as f64;
                let p10 = gray[y * w + (x - 1)] as f64;
                let p12 = gray[y * w + (x + 1)] as f64;
                let p20 = gray[(y + 1) * w + (x - 1)] as f64;
                let p21 = gray[(y + 1) * w + x] as f64;
                let p22 = gray[(y + 1) * w + (x + 1)] as f64;

                let gx = -p00 + p02 - 2.0 * p10 + 2.0 * p12 - p20 + p22;
                let gy = -p00 - 2.0 * p01 - p02 + p20 + 2.0 * p21 + p22;

                magnitude[y * w + x] = (gx * gx + gy * gy).sqrt();
                direction[y * w + x] = gy.atan2(gx);
            }
        }

        (magnitude, direction)
    }

    fn analyze_edge_pixel(
        &self,
        gray: &[u8],
        rgb_pixels: &[u8],
        w: usize,
        h: usize,
        x: usize,
        y: usize,
        direction: f64,
    ) -> Option<EdgeAnalysis> {
        let r = self.analysis_radius;

        let nx = direction.cos();
        let ny = direction.sin();

        let offset = r as f64 * 0.7;
        let cx1 = (x as f64 + nx * offset).round() as isize;
        let cy1 = (y as f64 + ny * offset).round() as isize;
        let cx2 = (x as f64 - nx * offset).round() as isize;
        let cy2 = (y as f64 - ny * offset).round() as isize;

        let half_r = (r / 2) as isize;

        if cx1 - half_r < 0
            || cy1 - half_r < 0
            || cx2 - half_r < 0
            || cy2 - half_r < 0
            || (cx1 + half_r) as usize >= w
            || (cy1 + half_r) as usize >= h
            || (cx2 + half_r) as usize >= w
            || (cy2 + half_r) as usize >= h
        {
            return None;
        }

        let mean1 = self.local_mean_luminance(gray, w, cx1 as usize, cy1 as usize, r);
        let mean2 = self.local_mean_luminance(gray, w, cx2 as usize, cy2 as usize, r);
        let luma_diff = (mean1 - mean2).abs();
        if luma_diff > 80.0 {
            return None; // Object boundary, not splice.
        }

        let noise1 = self.local_noise_variance(gray, w, cx1 as usize, cy1 as usize, r);
        let noise2 = self.local_noise_variance(gray, w, cx2 as usize, cy2 as usize, r);

        let noise_ratio = if noise2 > 1e-6 {
            noise1 / noise2
        } else if noise1 > 1e-6 {
            noise1 * 1000.0
        } else {
            1.0
        };

        let blocking1 = self.local_blocking_strength(gray, w, cx1 as usize, cy1 as usize, r);
        let blocking2 = self.local_blocking_strength(gray, w, cx2 as usize, cy2 as usize, r);

        let max_blocking = blocking1.max(blocking2).max(1e-6);
        let blocking_diff = (blocking1 - blocking2).abs() / max_blocking;

        let cfa1 = self.local_cfa_energy(rgb_pixels, w, cx1 as usize, cy1 as usize, r);
        let cfa2 = self.local_cfa_energy(rgb_pixels, w, cx2 as usize, cy2 as usize, r);

        let max_cfa = cfa1.max(cfa2).max(1e-6);
        let cfa_diff = (cfa1 - cfa2).abs() / max_cfa;

        Some(EdgeAnalysis {
            x,
            y,
            noise_ratio,
            blocking_diff,
            cfa_diff,
        })
    }

    fn local_mean_luminance(
        &self,
        gray: &[u8],
        w: usize,
        cx: usize,
        cy: usize,
        radius: usize,
    ) -> f64 {
        let half = radius / 2;
        let mut sum = 0.0_f64;
        let mut count = 0_u32;
        for dy in 0..half * 2 {
            for dx in 0..half * 2 {
                let px = cx.wrapping_sub(half).wrapping_add(dx);
                let py = cy.wrapping_sub(half).wrapping_add(dy);
                if px < w && py < gray.len() / w {
                    sum += gray[py * w + px] as f64;
                    count += 1;
                }
            }
        }
        if count > 0 { sum / count as f64 } else { 128.0 }
    }

    fn local_noise_variance(
        &self,
        gray: &[u8],
        w: usize,
        cx: usize,
        cy: usize,
        radius: usize,
    ) -> f64 {
        let half = radius / 2;
        let x0 = cx.saturating_sub(half);
        let y0 = cy.saturating_sub(half);

        let mut residuals: Vec<f64> = Vec::new();

        for dy in 1..radius.saturating_sub(1) {
            for dx in 1..radius.saturating_sub(1) {
                let x = x0 + dx;
                let y = y0 + dy;

                let center = gray[y * w + x] as f64;

                let n = gray[(y - 1) * w + x] as f64;
                let s = gray[(y + 1) * w + x] as f64;
                let e = gray[y * w + (x + 1)] as f64;
                let ww = gray[y * w + (x - 1)] as f64;

                let residual = center - (n + s + e + ww) / 4.0;
                residuals.push(residual);
            }
        }

        if residuals.is_empty() {
            return 0.0;
        }

        let mean: f64 = residuals.iter().sum::<f64>() / residuals.len() as f64;
        residuals.iter().map(|&r| (r - mean).powi(2)).sum::<f64>() / residuals.len() as f64
    }

    fn local_blocking_strength(
        &self,
        gray: &[u8],
        w: usize,
        cx: usize,
        cy: usize,
        radius: usize,
    ) -> f64 {
        let half = radius / 2;
        let x0 = cx.saturating_sub(half);
        let y0 = cy.saturating_sub(half);

        let mut boundary_sum = 0.0_f64;
        let mut boundary_count = 0_u64;
        let mut interior_sum = 0.0_f64;
        let mut interior_count = 0_u64;

        for dy in 0..radius {
            for dx in 1..radius {
                let x = x0 + dx;
                let y = y0 + dy;

                let diff = (gray[y * w + x] as f64 - gray[y * w + (x - 1)] as f64).abs();

                if x.is_multiple_of(8) {
                    boundary_sum += diff;
                    boundary_count += 1;
                } else {
                    interior_sum += diff;
                    interior_count += 1;
                }
            }
        }

        for dy in 1..radius {
            for dx in 0..radius {
                let x = x0 + dx;
                let y = y0 + dy;

                let diff = (gray[y * w + x] as f64 - gray[(y - 1) * w + x] as f64).abs();

                if y.is_multiple_of(8) {
                    boundary_sum += diff;
                    boundary_count += 1;
                } else {
                    interior_sum += diff;
                    interior_count += 1;
                }
            }
        }

        if boundary_count == 0 || interior_count == 0 {
            return 0.0;
        }

        let boundary_avg = boundary_sum / boundary_count as f64;
        let interior_avg = interior_sum / interior_count as f64;

        if interior_avg > 0.0 {
            boundary_avg / interior_avg
        } else {
            0.0
        }
    }

    fn local_cfa_energy(
        &self,
        rgb_pixels: &[u8],
        w: usize,
        cx: usize,
        cy: usize,
        radius: usize,
    ) -> f64 {
        let half = radius / 2;
        let x0 = cx.saturating_sub(half);
        let y0 = cy.saturating_sub(half);

        let mut even_residuals: Vec<f64> = Vec::new();
        let mut odd_residuals: Vec<f64> = Vec::new();

        for dy in 1..radius.saturating_sub(1) {
            for dx in 1..radius.saturating_sub(1) {
                let x = x0 + dx;
                let y = y0 + dy;

                let idx = (y * w + x) * 3 + 1;
                let center = rgb_pixels[idx] as f64;

                let north = rgb_pixels[((y - 1) * w + x) * 3 + 1] as f64;
                let south = rgb_pixels[((y + 1) * w + x) * 3 + 1] as f64;
                let west = rgb_pixels[(y * w + (x - 1)) * 3 + 1] as f64;
                let east = rgb_pixels[(y * w + (x + 1)) * 3 + 1] as f64;

                let predicted = (north + south + west + east) / 4.0;
                let residual = center - predicted;

                if (x + y).is_multiple_of(2) {
                    even_residuals.push(residual);
                } else {
                    odd_residuals.push(residual);
                }
            }
        }

        if even_residuals.is_empty() || odd_residuals.is_empty() {
            return 0.0;
        }

        let even_mean: f64 = even_residuals.iter().sum::<f64>() / even_residuals.len() as f64;
        let odd_mean: f64 = odd_residuals.iter().sum::<f64>() / odd_residuals.len() as f64;

        let even_var: f64 = even_residuals
            .iter()
            .map(|&r| (r - even_mean).powi(2))
            .sum::<f64>()
            / even_residuals.len() as f64;
        let odd_var: f64 = odd_residuals
            .iter()
            .map(|&r| (r - odd_mean).powi(2))
            .sum::<f64>()
            / odd_residuals.len() as f64;

        let total_var = (even_var + odd_var) / 2.0;
        if total_var < 1e-10 {
            return 0.0;
        }

        let mean_diff = (even_mean - odd_mean).abs();
        let var_diff = (even_var - odd_var).abs();

        mean_diff / total_var.sqrt() + var_diff / total_var
    }

    fn cluster_positions(
        positions: &[(usize, usize)],
        img_w: usize,
        img_h: usize,
    ) -> Vec<SpatialCluster> {
        if positions.is_empty() {
            return vec![];
        }

        let cell_size = 32_usize;
        let grid_w = img_w.div_ceil(cell_size);
        let grid_h = img_h.div_ceil(cell_size);

        let mut grid = vec![0_u32; grid_w * grid_h];
        for &(x, y) in positions {
            let gx = x / cell_size;
            let gy = y / cell_size;
            if gx < grid_w && gy < grid_h {
                grid[gy * grid_w + gx] += 1;
            }
        }

        let mut labels = vec![0_u32; grid_w * grid_h];
        let mut label_id = 0_u32;

        for gy in 0..grid_h {
            for gx in 0..grid_w {
                let idx = gy * grid_w + gx;
                if grid[idx] > 0 && labels[idx] == 0 {
                    label_id += 1;
                    let mut queue = vec![(gx, gy)];
                    labels[idx] = label_id;
                    while let Some((cx, cy)) = queue.pop() {
                        for &(dx, dy) in &[
                            (1_isize, 0),
                            (-1, 0),
                            (0, 1),
                            (0, -1),
                            (1, 1),
                            (1, -1),
                            (-1, 1),
                            (-1, -1),
                        ] {
                            let nx = cx as isize + dx;
                            let ny = cy as isize + dy;
                            if nx >= 0 && ny >= 0 {
                                let nx = nx as usize;
                                let ny = ny as usize;
                                if nx < grid_w && ny < grid_h {
                                    let ni = ny * grid_w + nx;
                                    if grid[ni] > 0 && labels[ni] == 0 {
                                        labels[ni] = label_id;
                                        queue.push((nx, ny));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        if label_id == 0 {
            return vec![];
        }

        let mut clusters: Vec<SpatialCluster> = (0..label_id)
            .map(|_| SpatialCluster {
                x_min: usize::MAX,
                y_min: usize::MAX,
                x_max: 0,
                y_max: 0,
                count: 0,
            })
            .collect();

        for &(x, y) in positions {
            let gx = x / cell_size;
            let gy = y / cell_size;
            if gx < grid_w && gy < grid_h {
                let label = labels[gy * grid_w + gx];
                if label > 0 {
                    let c = &mut clusters[(label - 1) as usize];
                    c.x_min = c.x_min.min(x);
                    c.y_min = c.y_min.min(y);
                    c.x_max = c.x_max.max(x);
                    c.y_max = c.y_max.max(y);
                    c.count += 1;
                }
            }
        }

        clusters.retain(|c| c.count >= 3);
        clusters.sort_by(|a, b| b.count.cmp(&a.count));
        clusters
    }
}
