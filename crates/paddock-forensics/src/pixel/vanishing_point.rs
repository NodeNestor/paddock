//! Vanishing-point consistency analysis, ported verbatim from the CPU
//! reference. CPU-only (Hough transform + intersection clustering; no GPU
//! kernel), `gpu()` delegates.
//!
//! Parallel 3D lines converge to vanishing points in a 2D image. A composite
//! from different scenes has lines that converge to inconsistent VPs across
//! regions. We Hough-detect strong lines, intersect non-parallel pairs into VP
//! candidates, and check whether the left/right and top/bottom image halves
//! agree on a VP. Camera-specific: skipped for documents.
//!
//! Note: paddock's Context stores gray as a raw `Vec<u8>`, whereas the
//! reference's is a `GrayImage`; the downsample path rebuilds the equivalent
//! `GrayImage` via `ctx.image.to_luma8()` (bit-identical to `ctx.gray()`), so
//! the resized bytes match the reference's exactly.

use crate::analyzer::Analyzer;
use crate::context::ContentType;
use crate::{Context, Finding, Severity};

pub struct VanishingPointAnalyzer {
    /// Sobel gradient threshold (on [0,255] grayscale).
    edge_threshold: f64,
    /// Number of Hough theta bins.
    num_theta: usize,
    /// Minimum votes for a Hough peak as a fraction of the max dimension.
    peak_vote_fraction: f64,
    /// Angle tolerance for grouping parallel lines (degrees).
    _angle_group_tolerance: f64,
    /// Maximum VP distance from center (as a fraction of the diagonal).
    max_vp_distance_fraction: f64,
}

impl Default for VanishingPointAnalyzer {
    fn default() -> Self {
        Self {
            edge_threshold: 80.0,
            num_theta: 180,
            peak_vote_fraction: 0.06,
            _angle_group_tolerance: 15.0,
            max_vp_distance_fraction: 5.0,
        }
    }
}

#[derive(Clone)]
struct DetectedLine {
    rho: f64,
    theta: f64,
    votes: u32,
}

#[allow(dead_code)]
struct VanishingPointCandidate {
    x: f64,
    y: f64,
    support: usize,
}

impl Analyzer for VanishingPointAnalyzer {
    fn name(&self) -> &'static str {
        "vanishing_point"
    }

    fn applies_to(&self, ctx: &Context) -> bool {
        // Camera-specific -> skip documents (which includes PDFs).
        !matches!(ctx.content_type, ContentType::Document)
    }

    fn cpu(&self, ctx: &Context) -> Vec<Finding> {
        let (orig_width, orig_height) = (ctx.width as usize, ctx.height as usize);

        if orig_width < 200 || orig_height < 200 {
            return Vec::new();
        }

        // Downsample large images: Hough cost is O(W*H*num_theta); ~500K pixels
        // is a big speedup with negligible VP-accuracy loss.
        let (gray_data, width, height);
        if orig_width * orig_height > 500_000 {
            let scale = ((orig_width * orig_height) as f64 / 500_000.0).sqrt();
            let new_w = (orig_width as f64 / scale) as u32;
            let new_h = (orig_height as f64 / scale) as u32;
            let luma = ctx.image.to_luma8();
            let resized =
                image::imageops::resize(&luma, new_w, new_h, image::imageops::FilterType::Triangle);
            gray_data = resized.into_raw();
            width = new_w as usize;
            height = new_h as usize;
        } else {
            gray_data = ctx.gray().to_vec();
            width = orig_width;
            height = orig_height;
        }
        let gray = &gray_data;

        // Step 1: Hough line detection.
        let rho_max = ((width * width + height * height) as f64).sqrt();
        let num_rho = (2.0 * rho_max) as usize + 1;
        let accumulator = self.hough_accumulate(gray, width, height, num_rho, rho_max);
        let min_votes = (self.peak_vote_fraction * width.max(height) as f64) as u32;
        let lines = self.extract_peaks(&accumulator, num_rho, min_votes);

        if lines.len() < 4 {
            return Vec::new();
        }

        // Step 2: VP consistency across image halves.
        self.check_consistency(&lines, width, height)
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

impl VanishingPointAnalyzer {
    fn hough_accumulate(
        &self,
        gray: &[u8],
        width: usize,
        height: usize,
        num_rho: usize,
        rho_max: f64,
    ) -> Vec<u32> {
        let num_theta = self.num_theta;
        let mut acc = vec![0u32; num_theta * num_rho];

        let theta_step = std::f64::consts::PI / num_theta as f64;
        let cos_t: Vec<f64> = (0..num_theta)
            .map(|t| (t as f64 * theta_step).cos())
            .collect();
        let sin_t: Vec<f64> = (0..num_theta)
            .map(|t| (t as f64 * theta_step).sin())
            .collect();

        for y in 1..height - 1 {
            for x in 1..width - 1 {
                let f = |px: usize, py: usize| gray[py * width + px] as f64;

                let gx = -f(x - 1, y - 1) + f(x + 1, y - 1) - 2.0 * f(x - 1, y) + 2.0 * f(x + 1, y)
                    - f(x - 1, y + 1)
                    + f(x + 1, y + 1);
                let gy = -f(x - 1, y - 1) - 2.0 * f(x, y - 1) - f(x + 1, y - 1)
                    + f(x - 1, y + 1)
                    + 2.0 * f(x, y + 1)
                    + f(x + 1, y + 1);

                if (gx * gx + gy * gy).sqrt() < self.edge_threshold {
                    continue;
                }

                for t in 0..num_theta {
                    let rho = x as f64 * cos_t[t] + y as f64 * sin_t[t];
                    let ri =
                        ((rho + rho_max) / (2.0 * rho_max) * (num_rho - 1) as f64 + 0.5) as usize;
                    acc[t * num_rho + ri.min(num_rho - 1)] += 1;
                }
            }
        }

        acc
    }

    fn extract_peaks(&self, acc: &[u32], num_rho: usize, min_votes: u32) -> Vec<DetectedLine> {
        let num_theta = self.num_theta;
        let rho_max = (num_rho as f64 - 1.0) / 2.0;
        let theta_step = std::f64::consts::PI / num_theta as f64;
        let nms: i32 = 5;
        let mut peaks = Vec::new();

        for t in 0..num_theta {
            for r in 0..num_rho {
                let v = acc[t * num_rho + r];
                if v < min_votes {
                    continue;
                }

                let mut is_max = true;
                'nms: for dt in -nms..=nms {
                    for dr in -nms..=nms {
                        if dt == 0 && dr == 0 {
                            continue;
                        }
                        let nt = (t as i32 + dt).rem_euclid(num_theta as i32) as usize;
                        let nr = r as i32 + dr;
                        if nr < 0 || nr >= num_rho as i32 {
                            continue;
                        }
                        if acc[nt * num_rho + nr as usize] > v {
                            is_max = false;
                            break 'nms;
                        }
                    }
                }

                if is_max {
                    peaks.push(DetectedLine {
                        rho: r as f64 - rho_max,
                        theta: t as f64 * theta_step,
                        votes: v,
                    });
                }
            }
        }

        peaks.sort_by(|a, b| b.votes.cmp(&a.votes));
        peaks.truncate(40);
        peaks
    }

    /// VP consistency across image halves.
    fn check_consistency(
        &self,
        lines: &[DetectedLine],
        width: usize,
        height: usize,
    ) -> Vec<Finding> {
        let mut findings = Vec::new();

        let cx = width as f64 / 2.0;
        let cy = height as f64 / 2.0;

        // Intersections of non-parallel line pairs -> VP candidates.
        let mut intersections: Vec<(f64, f64, usize, usize)> = Vec::new();
        let max_dist =
            self.max_vp_distance_fraction * ((width * width + height * height) as f64).sqrt();

        for i in 0..lines.len() {
            for j in (i + 1)..lines.len() {
                let diff = angle_diff(lines[i].theta, lines[j].theta);
                if diff < 5.0_f64.to_radians() {
                    continue; // Too parallel.
                }

                if let Some((x, y)) = line_intersection(&lines[i], &lines[j]) {
                    let dist = ((x - cx).powi(2) + (y - cy).powi(2)).sqrt();
                    if dist < max_dist {
                        intersections.push((x, y, i, j));
                    }
                }
            }
        }

        if intersections.len() < 5 {
            return findings;
        }

        // Grid-density clustering (kept for parity with the reference; the quadrant
        // coverage it feeds is exploratory and does not itself emit a finding).
        let cell_size = width.max(height) as f64 / 10.0;
        let clusters = cluster_intersections(&intersections, cell_size);

        if clusters.len() < 2 {
            return findings;
        }

        let quadrant_of_line = |line: &DetectedLine| -> u8 {
            let cos_t = line.theta.cos();
            let sin_t = line.theta.sin();

            let mut mask = 0u8;
            for qy in 0..2 {
                for qx in 0..2 {
                    let test_x = cx * (qx as f64 + 0.5);
                    let test_y = cy * (qy as f64 + 0.5);
                    let dist = (test_x * cos_t + test_y * sin_t - line.rho).abs();
                    if dist < width.max(height) as f64 / 4.0 {
                        mask |= 1 << (qy * 2 + qx);
                    }
                }
            }
            mask
        };

        let top_clusters: Vec<&Vec<usize>> = clusters.iter().take(3).collect();
        for cluster in &top_clusters {
            let mut cluster_line_indices: Vec<usize> = Vec::new();
            for &int_idx in cluster.iter() {
                let (_, _, li, lj) = intersections[int_idx];
                if !cluster_line_indices.contains(&li) {
                    cluster_line_indices.push(li);
                }
                if !cluster_line_indices.contains(&lj) {
                    cluster_line_indices.push(lj);
                }
            }

            let quadrant_masks: Vec<u8> = cluster_line_indices
                .iter()
                .map(|&li| quadrant_of_line(&lines[li]))
                .collect();

            let combined_mask: u8 = quadrant_masks.iter().fold(0, |acc, &m| acc | m);
            let _coverage = combined_mask.count_ones();
        }

        // Left vs right: lines in each half should converge to the same VP.
        let mut left_intersections: Vec<(f64, f64)> = Vec::new();
        let mut right_intersections: Vec<(f64, f64)> = Vec::new();

        for &(x, y, li, lj) in &intersections {
            let line_i_center_x = if lines[li].theta.sin().abs() > 0.01 {
                (lines[li].rho - cy * lines[li].theta.sin()) / lines[li].theta.cos()
            } else {
                cx
            };
            let line_j_center_x = if lines[lj].theta.sin().abs() > 0.01 {
                (lines[lj].rho - cy * lines[lj].theta.sin()) / lines[lj].theta.cos()
            } else {
                cx
            };

            if line_i_center_x < cx && line_j_center_x < cx {
                left_intersections.push((x, y));
            }
            if line_i_center_x >= cx && line_j_center_x >= cx {
                right_intersections.push((x, y));
            }
        }

        if left_intersections.len() >= 3 && right_intersections.len() >= 3 {
            let left_vp = median_point(&left_intersections);
            let right_vp = median_point(&right_intersections);

            let vp_distance =
                ((left_vp.0 - right_vp.0).powi(2) + (left_vp.1 - right_vp.1).powi(2)).sqrt();

            let diag = ((width * width + height * height) as f64).sqrt();
            let relative_dist = vp_distance / diag;

            if relative_dist > 0.15 {
                findings.push(Finding::new(
                    "vanishing_point",
                    "vanishing_point_inconsistency",
                    format!(
                        "Vanishing points estimated from left and right image halves \
                         diverge by {vp_distance:.0} pixels ({:.1}% of diagonal) - \
                         possible composite from scenes with different perspective geometry",
                        relative_dist * 100.0,
                    ),
                    if relative_dist > 0.30 {
                        Severity::High
                    } else {
                        Severity::Medium
                    },
                    (0.40 + relative_dist).min(0.80),
                ));
            }
        }

        // Top vs bottom.
        let mut top_intersections: Vec<(f64, f64)> = Vec::new();
        let mut bottom_intersections: Vec<(f64, f64)> = Vec::new();

        for &(x, y, li, lj) in &intersections {
            let line_i_center_y = if lines[li].theta.cos().abs() > 0.01 {
                (lines[li].rho - cx * lines[li].theta.cos()) / lines[li].theta.sin()
            } else {
                cy
            };
            let line_j_center_y = if lines[lj].theta.cos().abs() > 0.01 {
                (lines[lj].rho - cx * lines[lj].theta.cos()) / lines[lj].theta.sin()
            } else {
                cy
            };

            if line_i_center_y < cy && line_j_center_y < cy {
                top_intersections.push((x, y));
            }
            if line_i_center_y >= cy && line_j_center_y >= cy {
                bottom_intersections.push((x, y));
            }
        }

        if top_intersections.len() >= 3 && bottom_intersections.len() >= 3 {
            let top_vp = median_point(&top_intersections);
            let bottom_vp = median_point(&bottom_intersections);

            let vp_distance =
                ((top_vp.0 - bottom_vp.0).powi(2) + (top_vp.1 - bottom_vp.1).powi(2)).sqrt();

            let diag = ((width * width + height * height) as f64).sqrt();
            let relative_dist = vp_distance / diag;

            if relative_dist > 0.15 {
                findings.push(Finding::new(
                    "vanishing_point",
                    "vanishing_point_vertical_inconsistency",
                    format!(
                        "Vanishing points from top and bottom halves diverge by \
                         {vp_distance:.0} pixels ({:.1}% of diagonal) - possible vertical splice \
                         combining scenes with different perspective",
                        relative_dist * 100.0,
                    ),
                    if relative_dist > 0.30 {
                        Severity::High
                    } else {
                        Severity::Medium
                    },
                    (0.40 + relative_dist).min(0.80),
                ));
            }
        }

        findings
    }
}

/// Angle difference in [0, π/2] (lines are undirected).
fn angle_diff(a: f64, b: f64) -> f64 {
    let diff = (a - b).abs();
    diff.min(std::f64::consts::PI - diff)
}

/// Intersection of two Hough lines (rho, theta).
fn line_intersection(l1: &DetectedLine, l2: &DetectedLine) -> Option<(f64, f64)> {
    let sin_diff = (l1.theta - l2.theta).sin();
    if sin_diff.abs() < 1e-10 {
        return None; // Parallel.
    }

    let x = (l2.rho * l1.theta.sin() - l1.rho * l2.theta.sin()) / sin_diff;
    let y = (l1.rho * l2.theta.cos() - l2.rho * l1.theta.cos()) / sin_diff;
    Some((x, y))
}

/// Cluster intersection points by grid density (merging adjacent cells).
fn cluster_intersections(
    intersections: &[(f64, f64, usize, usize)],
    cell_size: f64,
) -> Vec<Vec<usize>> {
    use std::collections::HashMap;

    let mut grid: HashMap<(i64, i64), Vec<usize>> = HashMap::new();

    for (i, &(x, y, _, _)) in intersections.iter().enumerate() {
        let gx = (x / cell_size).floor() as i64;
        let gy = (y / cell_size).floor() as i64;
        grid.entry((gx, gy)).or_default().push(i);
    }

    let mut cells: Vec<((i64, i64), Vec<usize>)> = grid.into_iter().collect();
    cells.sort_by_key(|&((gx, gy), _)| (gx, gy));

    let mut clusters: Vec<Vec<usize>> = Vec::new();
    let mut used = vec![false; cells.len()];

    for i in 0..cells.len() {
        if used[i] {
            continue;
        }
        let mut cluster = cells[i].1.clone();
        used[i] = true;

        for j in (i + 1)..cells.len() {
            if used[j] {
                continue;
            }
            let dx = (cells[i].0.0 - cells[j].0.0).abs();
            let dy = (cells[i].0.1 - cells[j].0.1).abs();
            if dx <= 1 && dy <= 1 {
                cluster.extend(&cells[j].1);
                used[j] = true;
            }
        }

        if cluster.len() >= 2 {
            clusters.push(cluster);
        }
    }

    clusters.sort_by_key(|c| std::cmp::Reverse(c.len()));
    clusters
}

/// Coordinate-wise median of a set of 2D points.
fn median_point(points: &[(f64, f64)]) -> (f64, f64) {
    let mut xs: Vec<f64> = points.iter().map(|p| p.0).collect();
    let mut ys: Vec<f64> = points.iter().map(|p| p.1).collect();
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    ys.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    (xs[xs.len() / 2], ys[ys.len() / 2])
}
