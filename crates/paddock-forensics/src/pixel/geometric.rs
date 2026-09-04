//! Geometric consistency via Hough line transform, ported verbatim from
//! the CPU reference. CPU-only: the canonical algorithm bins per-block
//! edge orientation on `atan2(gy,gx)` - a discrete decision driven by a
//! transcendental, so a device kernel using CUDA's atan2 could bin a pixel
//! differently than Rust's libm and diverge. An exact-parity GPU kernel of this
//! algorithm is therefore not feasible (same class as double_jpeg's DCT), so
//! `gpu()` delegates to `cpu()`. Camera/scene-specific -> skipped for documents.

use crate::analyzer::Analyzer;
use crate::context::ContentType;
use crate::{Context, Finding, Region, Severity};

pub struct GeometricConsistencyAnalyzer {
    edge_threshold: f32,
    num_theta: u32,
    peak_vote_fraction: f64,
    orient_block_size: usize,
    num_orient_bins: usize,
    orient_chi_sq_threshold: f64,
}

impl Default for GeometricConsistencyAnalyzer {
    fn default() -> Self {
        Self {
            edge_threshold: 0.3,
            num_theta: 180,
            peak_vote_fraction: 0.08,
            orient_block_size: 64,
            num_orient_bins: 18,
            orient_chi_sq_threshold: 50.0,
        }
    }
}

struct HoughLine {
    rho: f64,
    theta: f64,
    votes: u32,
}

/// A detected gap in a Hough line's edge support.
struct LineGap {
    start: usize,
    length: usize,
    bounding_box: Option<Region>,
}

impl Analyzer for GeometricConsistencyAnalyzer {
    fn name(&self) -> &'static str {
        "geometric_consistency"
    }

    fn applies_to(&self, ctx: &Context) -> bool {
        // Camera/scene-specific -> skip documents (which includes PDFs).
        !matches!(ctx.content_type, ContentType::Document)
    }

    fn cpu(&self, ctx: &Context) -> Vec<Finding> {
        let (orig_width, orig_height) = (ctx.width as usize, ctx.height as usize);

        if orig_width < self.orient_block_size * 3 || orig_height < self.orient_block_size * 3 {
            return Vec::new();
        }

        // Downsample large images (Hough is O(W*H*num_theta)); rebuild the
        // equivalent GrayImage from ctx.image.
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

        let rho_max = ((width * width + height * height) as f64).sqrt();
        let num_rho = (2.0 * rho_max) as usize + 1;

        let accumulator = self.hough_accumulate_cpu(gray, width, height, num_rho, rho_max);
        let orient_histograms = self.block_orientations_cpu(gray, width, height);

        let mut findings = Vec::new();

        let min_votes = (self.peak_vote_fraction * width.max(height) as f64) as u32;
        let peaks = self.extract_peaks(&accumulator, self.num_theta as usize, num_rho, min_votes);

        if !peaks.is_empty() {
            self.analyze_line_gaps(&peaks, gray, width, height, &mut findings);
        }

        self.analyze_orientation_consistency(&orient_histograms, width, height, &mut findings);

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

impl GeometricConsistencyAnalyzer {
    fn hough_accumulate_cpu(
        &self,
        gray: &[u8],
        width: usize,
        height: usize,
        num_rho: usize,
        rho_max: f64,
    ) -> Vec<u32> {
        let num_theta = self.num_theta as usize;
        let mut accumulator = vec![0u32; num_theta * num_rho];
        let threshold = self.edge_threshold * 255.0;

        let theta_step = std::f64::consts::PI / num_theta as f64;
        let cos_table: Vec<f64> = (0..num_theta)
            .map(|t| (t as f64 * theta_step).cos())
            .collect();
        let sin_table: Vec<f64> = (0..num_theta)
            .map(|t| (t as f64 * theta_step).sin())
            .collect();

        for y in 1..height - 1 {
            for x in 1..width - 1 {
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

                if mag < threshold as f64 {
                    continue;
                }

                for t in 0..num_theta {
                    let rho = x as f64 * cos_table[t] + y as f64 * sin_table[t];
                    let rho_idx =
                        ((rho + rho_max) / (2.0 * rho_max) * (num_rho - 1) as f64 + 0.5) as usize;
                    let rho_idx = rho_idx.min(num_rho - 1);
                    accumulator[t * num_rho + rho_idx] += 1;
                }
            }
        }

        accumulator
    }

    fn block_orientations_cpu(&self, gray: &[u8], width: usize, height: usize) -> Vec<f32> {
        let bs = self.orient_block_size;
        let blocks_x = width / bs;
        let blocks_y = height / bs;
        let nbins = self.num_orient_bins;
        let mut histograms = vec![0.0f32; blocks_x * blocks_y * nbins];
        let threshold = self.edge_threshold * 255.0;

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
                        if mag < threshold as f64 {
                            continue;
                        }

                        let mut angle = gy.atan2(gx);
                        if angle < 0.0 {
                            angle += std::f64::consts::PI;
                        }
                        let bin = ((angle / std::f64::consts::PI) * nbins as f64) as usize;
                        let bin = bin.min(nbins - 1);
                        histograms[block_idx * nbins + bin] += 1.0;
                    }
                }
            }
        }

        histograms
    }

    fn extract_peaks(
        &self,
        accumulator: &[u32],
        num_theta: usize,
        num_rho: usize,
        min_votes: u32,
    ) -> Vec<HoughLine> {
        let rho_max = (num_rho as f64 - 1.0) / 2.0;
        let theta_step = std::f64::consts::PI / num_theta as f64;
        let nms_radius: i32 = 5;

        let mut peaks = Vec::new();

        for t in 0..num_theta {
            for r in 0..num_rho {
                let votes = accumulator[t * num_rho + r];
                if votes < min_votes {
                    continue;
                }

                let mut is_max = true;
                for dt in -nms_radius..=nms_radius {
                    for dr in -nms_radius..=nms_radius {
                        if dt == 0 && dr == 0 {
                            continue;
                        }
                        let nt = (t as i32 + dt).rem_euclid(num_theta as i32) as usize;
                        let nr = r as i32 + dr;
                        if nr < 0 || nr >= num_rho as i32 {
                            continue;
                        }
                        if accumulator[nt * num_rho + nr as usize] > votes {
                            is_max = false;
                            break;
                        }
                    }
                    if !is_max {
                        break;
                    }
                }

                if is_max {
                    peaks.push(HoughLine {
                        rho: r as f64 - rho_max,
                        theta: t as f64 * theta_step,
                        votes,
                    });
                }
            }
        }

        peaks.sort_by_key(|p| std::cmp::Reverse(p.votes));
        peaks.truncate(50);
        peaks
    }

    fn analyze_line_gaps(
        &self,
        peaks: &[HoughLine],
        gray: &[u8],
        width: usize,
        height: usize,
        findings: &mut Vec<Finding>,
    ) {
        let threshold = (self.edge_threshold * 255.0) as f64;
        let min_segment_len = width.min(height) / 10;
        let min_gap_len = 8_usize;
        let max_gap_len = width.max(height) / 4;
        let mut gap_findings = 0;

        for line in peaks.iter().take(20) {
            let cos_t = line.theta.cos();
            let sin_t = line.theta.sin();

            let support = if sin_t.abs() > cos_t.abs() {
                self.trace_line_along_x(line, gray, width, height, threshold)
            } else {
                self.trace_line_along_y(line, gray, width, height, threshold)
            };

            let gaps = self.find_gaps(&support, min_segment_len, min_gap_len, max_gap_len);

            for gap in &gaps {
                gap_findings += 1;
                if gap_findings <= 3 {
                    let mut finding = Finding::new(
                        "geometric_consistency",
                        "geometric_line_break",
                        format!(
                            "Strong line (theta={:.0}°, {} votes) breaks at pixel offset {} \
                             with gap of {} pixels - possible splice boundary",
                            line.theta.to_degrees(),
                            line.votes,
                            gap.start,
                            gap.length,
                        ),
                        Severity::Medium,
                        (0.45 + 0.1 * (line.votes as f64 / peaks[0].votes as f64)).min(0.75),
                    );
                    if let Some(region) = gap.bounding_box.clone() {
                        finding = finding.with_region(region);
                    }
                    findings.push(finding);
                }
            }
        }

        if gap_findings > 3 {
            findings.push(Finding::new(
                "geometric_consistency",
                "geometric_multiple_line_breaks",
                format!(
                    "{gap_findings} line break events detected across dominant image lines - \
                     significant geometric discontinuity"
                ),
                Severity::High,
                (0.55 + 0.05 * gap_findings as f64).min(0.85),
            ));
        }
    }

    fn trace_line_along_x(
        &self,
        line: &HoughLine,
        gray: &[u8],
        width: usize,
        height: usize,
        threshold: f64,
    ) -> Vec<(bool, usize, usize)> {
        let mut support = Vec::new();

        for x in 0..width {
            let y = ((line.rho - x as f64 * line.theta.cos()) / line.theta.sin()) as i64;
            if y < 1 || y >= height as i64 - 1 {
                continue;
            }
            let y = y as usize;

            let has_edge = self.check_edge_at(gray, x, y, width, height, threshold);
            support.push((has_edge, x, y));
        }

        support
    }

    fn trace_line_along_y(
        &self,
        line: &HoughLine,
        gray: &[u8],
        width: usize,
        height: usize,
        threshold: f64,
    ) -> Vec<(bool, usize, usize)> {
        let mut support = Vec::new();

        for y in 0..height {
            let x = ((line.rho - y as f64 * line.theta.sin()) / line.theta.cos()) as i64;
            if x < 1 || x >= width as i64 - 1 {
                continue;
            }
            let x = x as usize;

            let has_edge = self.check_edge_at(gray, x, y, width, height, threshold);
            support.push((has_edge, x, y));
        }

        support
    }

    fn check_edge_at(
        &self,
        gray: &[u8],
        x: usize,
        y: usize,
        width: usize,
        height: usize,
        threshold: f64,
    ) -> bool {
        if x < 1 || x >= width - 1 || y < 1 || y >= height - 1 {
            return false;
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

        (gx * gx + gy * gy).sqrt() >= threshold
    }

    fn find_gaps(
        &self,
        support: &[(bool, usize, usize)],
        min_segment: usize,
        min_gap: usize,
        max_gap: usize,
    ) -> Vec<LineGap> {
        if support.is_empty() {
            return Vec::new();
        }

        let mut runs: Vec<(bool, usize, usize)> = Vec::new();
        let mut current = support[0].0;
        let mut start = 0;

        for (i, &(has_edge, _, _)) in support.iter().enumerate().skip(1) {
            if has_edge != current {
                runs.push((current, start, i - start));
                current = has_edge;
                start = i;
            }
        }
        runs.push((current, start, support.len() - start));

        let mut gaps = Vec::new();
        for i in 1..runs.len().saturating_sub(1) {
            let (is_edge, gap_start, gap_len) = runs[i];
            if is_edge {
                continue;
            }
            if gap_len < min_gap || gap_len > max_gap {
                continue;
            }

            let (prev_edge, _, prev_len) = runs[i - 1];
            let (next_edge, _, next_len) = runs[i + 1];

            if prev_edge && next_edge && prev_len >= min_segment && next_len >= min_segment {
                let gap_end = gap_start + gap_len;
                let mut min_x = usize::MAX;
                let mut min_y = usize::MAX;
                let mut max_x = 0_usize;
                let mut max_y = 0_usize;

                for &(_, x, y) in &support[gap_start..gap_end] {
                    min_x = min_x.min(x);
                    min_y = min_y.min(y);
                    max_x = max_x.max(x);
                    max_y = max_y.max(y);
                }

                let pad = 16;
                let bb_x = min_x.saturating_sub(pad) as u32;
                let bb_y = min_y.saturating_sub(pad) as u32;
                let bb_w = (max_x - min_x + 2 * pad) as u32;
                let bb_h = (max_y - min_y + 2 * pad) as u32;

                gaps.push(LineGap {
                    start: gap_start,
                    length: gap_len,
                    bounding_box: Some(Region::BoundingBox {
                        x: bb_x,
                        y: bb_y,
                        width: bb_w,
                        height: bb_h,
                    }),
                });
            }
        }

        gaps
    }

    fn analyze_orientation_consistency(
        &self,
        histograms: &[f32],
        width: usize,
        height: usize,
        findings: &mut Vec<Finding>,
    ) {
        let bs = self.orient_block_size;
        let nbins = self.num_orient_bins;
        let blocks_x = width / bs;
        let blocks_y = height / bs;
        let num_blocks = blocks_x * blocks_y;

        if num_blocks < 4 {
            return;
        }

        let mut global_hist = vec![0.0_f64; nbins];
        let mut active_blocks = 0;

        for b in 0..num_blocks {
            let block_hist = &histograms[b * nbins..(b + 1) * nbins];
            let total: f32 = block_hist.iter().sum();
            if total < 10.0 {
                continue;
            }
            active_blocks += 1;
            for i in 0..nbins {
                global_hist[i] += block_hist[i] as f64 / total as f64;
            }
        }

        if active_blocks < 4 {
            return;
        }

        for v in &mut global_hist {
            *v /= active_blocks as f64;
        }

        let mut anomaly_scores: Vec<(usize, f64)> = Vec::new();

        for b in 0..num_blocks {
            let block_hist = &histograms[b * nbins..(b + 1) * nbins];
            let total: f64 = block_hist.iter().map(|&v| v as f64).sum();
            if total < 10.0 {
                continue;
            }

            let mut chi_sq = 0.0_f64;
            for i in 0..nbins {
                let observed = block_hist[i] as f64 / total;
                let expected = global_hist[i];
                if expected > 1e-6 {
                    chi_sq += (observed - expected).powi(2) / expected;
                }
            }

            anomaly_scores.push((b, chi_sq));
        }

        if anomaly_scores.is_empty() {
            return;
        }

        let mut scores: Vec<f64> = anomaly_scores.iter().map(|&(_, s)| s).collect();
        scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = scores[scores.len() / 2];
        let mad: f64 = {
            let mut deviations: Vec<f64> = scores.iter().map(|s| (s - median).abs()).collect();
            deviations.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            deviations[deviations.len() / 2] * 1.4826
        };

        let outlier_threshold = (median + 3.0 * mad).max(self.orient_chi_sq_threshold);

        let outlier_blocks: Vec<(usize, f64)> = anomaly_scores
            .iter()
            .filter(|&&(_, score)| score > outlier_threshold)
            .copied()
            .collect();

        if outlier_blocks.is_empty() {
            return;
        }

        let outlier_ratio = outlier_blocks.len() as f64 / active_blocks as f64;

        if outlier_ratio < 0.02 || outlier_blocks.len() < 2 {
            return;
        }

        let Some(strongest) = outlier_blocks
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        else {
            return;
        };
        let sx = (strongest.0 % blocks_x) * bs;
        let sy = (strongest.0 / blocks_x) * bs;

        findings.push(
            Finding::new(
                "geometric_consistency",
                "geometric_orientation_inconsistency",
                format!(
                    "{} of {} blocks ({:.1}%) show edge orientation distributions inconsistent \
                     with the image - possible composite from different scenes",
                    outlier_blocks.len(),
                    active_blocks,
                    outlier_ratio * 100.0,
                ),
                if outlier_ratio > 0.10 {
                    Severity::High
                } else {
                    Severity::Medium
                },
                (0.45 + outlier_ratio * 2.0).min(0.80),
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
