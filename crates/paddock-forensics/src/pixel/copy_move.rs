//! Copy-move (clone) detection via rotation-invariant Zernike moments + RANSAC,
//! ported verbatim from the CPU reference. CPU-only: the reference's copy-move
//! GPU path is a CPU-fallback stub, and the RANSAC step is inherently serial, so
//! `gpu()` delegates to `cpu()`.
//!
//! Pipeline: overlapping circular blocks -> Zernike magnitude features
//! (rotation-invariant) -> lexicographic-sort nearest-neighbor matching -> RANSAC
//! translation consistency -> cluster by displacement. Photo/document raster
//! only: skipped for PDFs.
//!
//! NOTE (verbatim): the RANSAC sampler seeds from `SystemTime` nanoseconds, so
//! the result is not deterministic run-to-run - this is the reference's own behavior,
//! kept as-is rather than "fixed" so the port matches the reference.

use crate::analyzer::Analyzer;
use crate::{Context, Finding, Region, Severity};

pub struct CopyMoveDetector {
    pub block_radius: usize,
    pub stride: usize,
    pub similarity_threshold: f64,
    pub min_cluster_size: usize,
    pub ransac_iterations: usize,
    pub ransac_inlier_threshold: f64,
}

impl Default for CopyMoveDetector {
    fn default() -> Self {
        Self {
            block_radius: 12,
            stride: 4,
            similarity_threshold: 0.15,
            min_cluster_size: 8,
            ransac_iterations: 200,
            ransac_inlier_threshold: 8.0,
        }
    }
}

impl Analyzer for CopyMoveDetector {
    fn name(&self) -> &'static str {
        "copy_move"
    }

    fn applies_to(&self, ctx: &Context) -> bool {
        !ctx.is_pdf()
    }

    fn cpu(&self, ctx: &Context) -> Vec<Finding> {
        let w = ctx.width as usize;
        let h = ctx.height as usize;
        let r = self.block_radius;

        if w < r * 6 || h < r * 6 {
            return vec![];
        }

        // Downsample large images; rebuild the equivalent GrayImage from
        // ctx.image (bit-identical to ctx.gray()).
        let (gray_data, work_w, work_h, _scale) = if w * h > 400_000 {
            let scale = ((w * h) as f64 / 400_000.0).sqrt();
            let new_w = (w as f64 / scale) as u32;
            let new_h = (h as f64 / scale) as u32;
            let luma = ctx.image.to_luma8();
            let resized =
                image::imageops::resize(&luma, new_w, new_h, image::imageops::FilterType::Triangle);
            (resized.into_raw(), new_w as usize, new_h as usize, scale)
        } else {
            (ctx.gray().to_vec(), w, h, 1.0)
        };
        let gray = &gray_data;
        let (w, h) = (work_w, work_h);

        let basis = ZernikeBasis::new(r);

        let blocks = self.extract_zernike_features(gray, w, h, &basis);
        if blocks.len() < 10 {
            return vec![];
        }

        let raw_matches = self.find_matches(&blocks);
        if raw_matches.is_empty() {
            return vec![];
        }

        let consistent_matches = self.ransac_filter(&raw_matches);
        let clusters = self.cluster_by_displacement(&consistent_matches);

        let mut findings = Vec::new();

        for (displacement, count, group) in &clusters {
            if *count >= self.min_cluster_size {
                let x_min = group.iter().map(|m| m.x1 as u32).min().unwrap_or(0);
                let x_max = group.iter().map(|m| m.x1 as u32).max().unwrap_or(0);
                let y_min = group.iter().map(|m| m.y1 as u32).min().unwrap_or(0);
                let y_max = group.iter().map(|m| m.y1 as u32).max().unwrap_or(0);

                findings.push(
                    Finding::new(
                        "copy_move",
                        "copy_move_detected",
                        format!(
                            "Copy-move forgery detected: {count} spatially consistent matching pairs \
                             with displacement ({:.0}, {:.0}) - region has been cloned \
                             (rotation-invariant Zernike moment matching with RANSAC verification)",
                            displacement.0, displacement.1
                        ),
                        Severity::Critical,
                        (0.65 + (*count as f64 / 80.0).min(0.30)).min(0.95),
                    )
                    .with_region(Region::BoundingBox {
                        x: x_min,
                        y: y_min,
                        width: x_max - x_min + 1,
                        height: y_max - y_min + 1,
                    }),
                );
            }
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

/// A block's position and Zernike moment feature vector.
struct ZernikeBlock {
    cx: usize,
    cy: usize,
    features: Vec<f64>,
}

/// A matched pair of blocks.
#[derive(Clone)]
struct Match {
    x1: f64,
    y1: f64,
    x2: f64,
    y2: f64,
}

impl CopyMoveDetector {
    fn extract_zernike_features(
        &self,
        gray: &[u8],
        w: usize,
        h: usize,
        basis: &ZernikeBasis,
    ) -> Vec<ZernikeBlock> {
        let r = self.block_radius;
        let mut blocks = Vec::new();

        let max_y = h.saturating_sub(r);
        let max_x = w.saturating_sub(r);

        for cy in (r..max_y).step_by(self.stride) {
            for cx in (r..max_x).step_by(self.stride) {
                let features = basis.compute_magnitudes(gray, w, cx, cy);
                blocks.push(ZernikeBlock { cx, cy, features });
            }
        }

        blocks
    }

    fn find_matches(&self, blocks: &[ZernikeBlock]) -> Vec<Match> {
        let mut indices: Vec<usize> = (0..blocks.len()).collect();
        indices.sort_by(|&a, &b| {
            let fa = &blocks[a].features;
            let fb = &blocks[b].features;
            for (va, vb) in fa.iter().zip(fb.iter()) {
                match va.partial_cmp(vb) {
                    Some(std::cmp::Ordering::Equal) => continue,
                    Some(ord) => return ord,
                    None => return std::cmp::Ordering::Equal,
                }
            }
            std::cmp::Ordering::Equal
        });

        let mut matches = Vec::new();
        let window = 6;

        for i in 0..indices.len() {
            for j in 1..=window {
                if i + j >= indices.len() {
                    break;
                }

                let a = &blocks[indices[i]];
                let b = &blocks[indices[i + j]];

                let spatial_dist = ((a.cx as f64 - b.cx as f64).powi(2)
                    + (a.cy as f64 - b.cy as f64).powi(2))
                .sqrt();
                if spatial_dist < (self.block_radius * 3) as f64 {
                    continue;
                }

                let feature_dist = Self::normalized_distance(&a.features, &b.features);

                if feature_dist < self.similarity_threshold {
                    matches.push(Match {
                        x1: a.cx as f64,
                        y1: a.cy as f64,
                        x2: b.cx as f64,
                        y2: b.cy as f64,
                    });
                }
            }
        }

        matches
    }

    fn normalized_distance(a: &[f64], b: &[f64]) -> f64 {
        let mut sum_sq = 0.0_f64;
        let mut norm_a = 0.0_f64;
        let mut norm_b = 0.0_f64;

        for (va, vb) in a.iter().zip(b.iter()) {
            sum_sq += (va - vb) * (va - vb);
            norm_a += va * va;
            norm_b += vb * vb;
        }

        let norm = (norm_a.sqrt() + norm_b.sqrt()) / 2.0;
        if norm > 1e-10 {
            sum_sq.sqrt() / norm
        } else {
            f64::MAX
        }
    }

    /// RANSAC translation-consistency filtering. True copy-move regions share a
    /// consistent displacement.
    fn ransac_filter(&self, matches: &[Match]) -> Vec<Match> {
        if matches.len() < 4 {
            return matches.to_vec();
        }

        let mut best_inliers = Vec::new();

        for _ in 0..self.ransac_iterations {
            let idx = (std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .subsec_nanos() as usize)
                % matches.len();
            let sample = &matches[idx];

            let dx = sample.x2 - sample.x1;
            let dy = sample.y2 - sample.y1;

            let inliers: Vec<Match> = matches
                .iter()
                .filter(|m| {
                    let mdx = m.x2 - m.x1;
                    let mdy = m.y2 - m.y1;
                    let error = ((mdx - dx).powi(2) + (mdy - dy).powi(2)).sqrt();
                    error < self.ransac_inlier_threshold
                })
                .map(|m| Match {
                    x1: m.x1,
                    y1: m.y1,
                    x2: m.x2,
                    y2: m.y2,
                })
                .collect();

            if inliers.len() > best_inliers.len() {
                best_inliers = inliers;
            }
        }

        best_inliers
    }

    fn cluster_by_displacement<'a>(
        &self,
        matches: &'a [Match],
    ) -> Vec<((f64, f64), usize, Vec<&'a Match>)> {
        let grid = 16;
        let mut displacement_groups: std::collections::HashMap<(i32, i32), Vec<&Match>> =
            std::collections::HashMap::new();

        for m in matches {
            let dx = ((m.x2 - m.x1) / grid as f64).round() as i32 * grid;
            let dy = ((m.y2 - m.y1) / grid as f64).round() as i32 * grid;
            displacement_groups.entry((dx, dy)).or_default().push(m);
        }

        let mut clusters: Vec<((f64, f64), usize, Vec<&Match>)> = displacement_groups
            .into_iter()
            .map(|((dx, dy), group)| ((dx as f64, dy as f64), group.len(), group))
            .collect();
        clusters.sort_by_key(|c| std::cmp::Reverse(c.1));

        let mut merged: Vec<((f64, f64), usize, Vec<&'a Match>)> = Vec::new();
        for cluster in clusters {
            let dominated = merged.iter().any(|(d, _, _)| {
                let ddx = (d.0 - cluster.0.0).abs();
                let ddy = (d.1 - cluster.0.1).abs();
                ddx <= 32.0 && ddy <= 32.0
            });
            if !dominated {
                merged.push(cluster);
            }
        }

        merged.into_iter().take(3).collect()
    }
}

/// Precomputed Zernike polynomial basis for a given radius.
struct ZernikeBasis {
    pixels: Vec<(i32, i32, Vec<(usize, i32, f64, f64)>)>,
    num_moments: usize,
}

/// Zernike (n, m) orders up to order 12 (even orders for rotation invariance).
const ZERNIKE_ORDERS: [(usize, i32); 20] = [
    (0, 0),
    (1, 1),
    (2, 0),
    (2, 2),
    (3, 1),
    (3, 3),
    (4, 0),
    (4, 2),
    (4, 4),
    (5, 1),
    (5, 3),
    (5, 5),
    (6, 0),
    (6, 2),
    (6, 4),
    (6, 6),
    (8, 0),
    (8, 2),
    (10, 0),
    (12, 0),
];

impl ZernikeBasis {
    fn new(radius: usize) -> Self {
        let r = radius as f64;
        let mut pixels = Vec::new();

        for dy in -(radius as i32)..=(radius as i32) {
            for dx in -(radius as i32)..=(radius as i32) {
                let x = dx as f64 / r;
                let y = dy as f64 / r;
                let rho = (x * x + y * y).sqrt();

                if rho > 1.0 {
                    continue;
                }

                let theta = y.atan2(x);

                let mut basis_values = Vec::new();
                for &(n, m) in &ZERNIKE_ORDERS {
                    let radial = Self::radial_polynomial(n, m.unsigned_abs() as usize, rho);
                    let v_real = radial * (m as f64 * theta).cos();
                    let v_imag = radial * (m as f64 * theta).sin();
                    basis_values.push((n, m, v_real, v_imag));
                }

                pixels.push((dx, dy, basis_values));
            }
        }

        Self {
            pixels,
            num_moments: ZERNIKE_ORDERS.len(),
        }
    }

    fn compute_magnitudes(&self, gray: &[u8], w: usize, cx: usize, cy: usize) -> Vec<f64> {
        let mut moments_real = vec![0.0_f64; self.num_moments];
        let mut moments_imag = vec![0.0_f64; self.num_moments];
        let mut count = 0.0_f64;

        for (dx, dy, basis_values) in &self.pixels {
            let x = cx as i32 + dx;
            let y = cy as i32 + dy;

            if x < 0 || y < 0 || x as usize >= w || y as usize >= (gray.len() / w) {
                continue;
            }

            let pixel = gray[y as usize * w + x as usize] as f64 / 255.0;
            count += 1.0;

            for (idx, &(_, _, v_real, v_imag)) in basis_values.iter().enumerate() {
                moments_real[idx] += pixel * v_real;
                moments_imag[idx] += pixel * v_imag;
            }
        }

        if count < 1.0 {
            return vec![0.0; self.num_moments];
        }

        (0..self.num_moments)
            .map(|i| {
                let (n, _) = ZERNIKE_ORDERS[i];
                let scale = (n + 1) as f64 / (std::f64::consts::PI * count);
                let re = moments_real[i] * scale;
                let im = moments_imag[i] * scale;
                (re * re + im * im).sqrt()
            })
            .collect()
    }

    fn radial_polynomial(n: usize, m: usize, rho: f64) -> f64 {
        if !(n - m).is_multiple_of(2) {
            return 0.0;
        }

        let mut sum = 0.0_f64;
        let upper = (n - m) / 2;

        for s in 0..=upper {
            let sign = if s % 2 == 0 { 1.0 } else { -1.0 };
            let num = Self::factorial(n - s);
            let den = Self::factorial(s)
                * Self::factorial((n + m) / 2 - s)
                * Self::factorial((n - m) / 2 - s);

            if den > 0.0 {
                sum += sign * (num / den) * rho.powi((n - 2 * s) as i32);
            }
        }

        sum
    }

    fn factorial(n: usize) -> f64 {
        (1..=n).fold(1.0, |acc, x| acc * x as f64)
    }
}
