//! Document-specific forensic analysis, ported verbatim from the CPU
//! reference. CPU-only (median-residual corner detection + per-block anomaly
//! map; no GPU kernel), `gpu()` delegates.
//!
//! Catches localized document tampering (a swapped digit/letter) by flagging
//! blocks whose noise/sharpness/edge statistics diverge from their local
//! neighborhood, plus paste-corner candidates where V and H noise boundaries
//! meet. Runs only on images this analyzer classifies as documents. Skipped for
//! PDFs (dedicated pdf/* analyzers handle those).
//!
//! PARITY NOTE: the reference's `analyze` declares `findings` twice - the second
//! declaration shadows the first, so when block anomalies are found the earlier
//! `document_detected` + paste-corner findings are dropped. This is faithfully
//! reproduced here so the port's output matches the reference byte-for-byte; do not
//! "fix" the shadow without re-blessing the oracle.

use crate::analyzer::Analyzer;
use crate::{Context, Finding, Region, Severity};

pub struct DocumentForensicsAnalyzer {
    /// Block size for local analysis (pixels).
    block_size: usize,
    /// Neighborhood radius in blocks for local comparison.
    neighborhood: usize,
    /// Minimum fraction of white-ish pixels to classify as a document.
    document_threshold: f64,
}

impl Default for DocumentForensicsAnalyzer {
    fn default() -> Self {
        Self {
            block_size: 8,
            neighborhood: 3,
            document_threshold: 0.40,
        }
    }
}

impl Analyzer for DocumentForensicsAnalyzer {
    fn name(&self) -> &'static str {
        "document_forensics"
    }

    fn applies_to(&self, ctx: &Context) -> bool {
        !ctx.is_pdf()
    }

    fn cpu(&self, ctx: &Context) -> Vec<Finding> {
        let w = ctx.width as usize;
        let h = ctx.height as usize;

        if w < self.block_size * 8 || h < self.block_size * 8 {
            return vec![];
        }

        // Step 1: classify as document vs photo.
        let gray = ctx.gray();
        if !self.is_document(gray, w, h) {
            return vec![];
        }

        let mut findings = Vec::new();

        findings.push(Finding::new(
            "document_forensics",
            "document_detected",
            "Image classified as document/screenshot - \
             running document-specific forensic analysis",
            Severity::Info,
            0.9,
        ));

        // Pass 1: paste-corner detection via noise-residual boundaries.
        findings.extend(self.detect_paste_corners(gray, w, h));

        // Pass 2: block anomaly detection.
        let blocks_x = w / self.block_size;
        let blocks_y = h / self.block_size;
        let block_features = self.compute_block_features(gray, w, h, blocks_x, blocks_y);
        let anomaly_map = self.compute_anomaly_map(&block_features, blocks_x, blocks_y);

        let mut scores: Vec<f64> = anomaly_map.to_vec();
        scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let median = scores[scores.len() / 2];
        let mad: f64 = {
            let mut deviations: Vec<f64> = scores.iter().map(|s| (s - median).abs()).collect();
            deviations.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            deviations[deviations.len() / 2] * 1.4826
        };

        if mad < 1e-6 {
            return findings;
        }

        // Blocks with anomaly score > median + 3*MAD.
        let threshold = median + 3.0 * mad;

        let mut outlier_positions: Vec<(usize, usize, f64)> = Vec::new();
        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                let score = anomaly_map[by * blocks_x + bx];
                if score > threshold {
                    let px = bx * self.block_size + self.block_size / 2;
                    let py = by * self.block_size + self.block_size / 2;
                    outlier_positions.push((px, py, score));
                }
            }
        }

        if outlier_positions.is_empty() {
            return vec![];
        }

        // Cluster outlier blocks spatially.
        let mut clusters = self.cluster_outliers(&outlier_positions, w, h);

        // Keep only the top 5 clusters by peak score.
        clusters.sort_by(|a, b| {
            b.peak_score
                .partial_cmp(&a.peak_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        clusters.truncate(5);

        // PARITY: second `findings` intentionally shadows the first (see the
        // module note) - the document_detected + paste-corner findings above are
        // dropped when block anomalies exist, matching the reference exactly.
        let mut findings = Vec::new();

        for (i, cluster) in clusters.iter().enumerate() {
            let pad = self.block_size as u32 / 2;
            let x = (cluster.x_min as u32).saturating_sub(pad);
            let y = (cluster.y_min as u32).saturating_sub(pad);
            let x_max = (cluster.x_max as u32 + pad).min(w as u32 - 1);
            let y_max = (cluster.y_max as u32 + pad).min(h as u32 - 1);

            findings.push(
                Finding::new(
                    "document_forensics",
                    "document_block_anomaly",
                    format!(
                        "Document forensics: anomalous region {} of {} at ({x},{y})->({x_max},{y_max}) - \
                         {} blocks with noise/sharpness inconsistent with surrounding content \
                         (peak anomaly score {:.2}, threshold {threshold:.2})",
                        i + 1,
                        clusters.len(),
                        cluster.count,
                        cluster.peak_score,
                    ),
                    if cluster.peak_score > threshold * 2.0 {
                        Severity::High
                    } else {
                        Severity::Medium
                    },
                    (0.4 + (cluster.peak_score / threshold - 1.0) * 0.3).min(0.85),
                )
                .with_region(Region::BoundingBox {
                    x,
                    y,
                    width: x_max - x + 1,
                    height: y_max - y + 1,
                }),
            );
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

/// Per-block forensic feature vector.
struct BlockFeatures {
    /// Noise variance (high-pass residual variance).
    noise_var: f64,
    /// Edge density (fraction of strong-gradient pixels).
    edge_density: f64,
    /// Local sharpness (mean gradient magnitude).
    sharpness: f64,
    /// Intensity mean.
    #[allow(dead_code)]
    intensity_mean: f64,
    /// Intensity variance.
    intensity_var: f64,
}

struct DocCluster {
    x_min: usize,
    y_min: usize,
    x_max: usize,
    y_max: usize,
    count: usize,
    peak_score: f64,
}

impl DocumentForensicsAnalyzer {
    /// Paste-corner candidates via noise-residual V+H boundary co-location.
    fn detect_paste_corners(&self, gray: &[u8], w: usize, h: usize) -> Vec<Finding> {
        if w < 20 || h < 20 {
            return vec![];
        }

        // Step 1: noise residual (original - 3×3 median filtered).
        let mut residual = vec![0.0_f64; w * h];
        for y in 1..h - 1 {
            for x in 1..w - 1 {
                let mut window = [0u8; 9];
                let mut k = 0;
                for dy in -1_isize..=1 {
                    for dx in -1_isize..=1 {
                        window[k] =
                            gray[(y as isize + dy) as usize * w + (x as isize + dx) as usize];
                        k += 1;
                    }
                }
                window.sort_unstable();
                residual[y * w + x] = (gray[y * w + x] as f64 - window[4] as f64).abs();
            }
        }

        // Step 2: per-pixel V-boundary and H-boundary strengths.
        let mut v_strength = vec![0.0_f64; w * h];
        let mut h_strength = vec![0.0_f64; w * h];

        for y in 2..h - 2 {
            for x in 2..w - 2 {
                let left_avg = (residual[y * w + (x - 2)] + residual[y * w + (x - 1)]) / 2.0;
                let right_avg = (residual[y * w + (x + 1)] + residual[y * w + (x + 2)]) / 2.0;
                v_strength[y * w + x] = (residual[y * w + x] - (left_avg + right_avg) / 2.0).abs();

                let top_avg = (residual[(y - 2) * w + x] + residual[(y - 1) * w + x]) / 2.0;
                let bot_avg = (residual[(y + 1) * w + x] + residual[(y + 2) * w + x]) / 2.0;
                h_strength[y * w + x] = (residual[y * w + x] - (top_avg + bot_avg) / 2.0).abs();
            }
        }

        // Step 3: corner strength = max V in a strip × max H in a strip.
        let corner_r = 6_usize;
        let mut corners: Vec<(usize, usize, f64)> = Vec::new();

        for y in corner_r..h.saturating_sub(corner_r) {
            for x in corner_r..w.saturating_sub(corner_r) {
                let mut max_v = 0.0_f64;
                for dy in -(corner_r as isize)..=(corner_r as isize) {
                    for dx in -2_isize..=2 {
                        let ny = (y as isize + dy) as usize;
                        let nx = (x as isize + dx) as usize;
                        let v = v_strength[ny * w + nx];
                        if v > max_v {
                            max_v = v;
                        }
                    }
                }

                let mut max_h = 0.0_f64;
                for dx in -(corner_r as isize)..=(corner_r as isize) {
                    for dy in -2_isize..=2 {
                        let ny = (y as isize + dy) as usize;
                        let nx = (x as isize + dx) as usize;
                        let h_val = h_strength[ny * w + nx];
                        if h_val > max_h {
                            max_h = h_val;
                        }
                    }
                }

                let corner_score = max_v * max_h;
                if corner_score > 100.0 {
                    corners.push((x, y, corner_score));
                }
            }
        }

        // Step 4: NMS - keep corners > 20px apart.
        corners.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

        let nms_radius = 20_usize;
        let mut kept: Vec<(usize, usize, f64)> = Vec::new();

        for (x, y, score) in &corners {
            let too_close = kept.iter().any(|(kx, ky, _)| {
                (*x as isize - *kx as isize).unsigned_abs() < nms_radius
                    && (*y as isize - *ky as isize).unsigned_abs() < nms_radius
            });
            if !too_close {
                kept.push((*x, *y, *score));
            }
            if kept.len() >= 15 {
                break;
            }
        }

        // Step 5: report top corners.
        let mut findings = Vec::new();
        let report_count = kept.len().min(10);
        let pad = 15_u32;

        for (i, &(x, y, score)) in kept.iter().take(report_count).enumerate() {
            let bx = (x as u32).saturating_sub(pad);
            let by = (y as u32).saturating_sub(pad);
            let bx2 = (x as u32 + pad).min(w as u32 - 1);
            let by2 = (y as u32 + pad).min(h as u32 - 1);

            findings.push(
                Finding::new(
                    "document_forensics",
                    "document_paste_corner",
                    format!(
                        "Paste corner candidate {} at ({x},{y}) - noise residual shows \
                         co-located V+H boundary transitions (score {score:.0})",
                        i + 1,
                    ),
                    if i == 0 {
                        Severity::High
                    } else if i < 3 {
                        Severity::Medium
                    } else {
                        Severity::Low
                    },
                    (0.4 + (score / 30000.0).min(0.4)).min(0.80),
                )
                .with_region(Region::BoundingBox {
                    x: bx,
                    y: by,
                    width: bx2 - bx,
                    height: by2 - by,
                }),
            );
        }

        findings
    }

    /// Classify document (text on background) vs photo.
    fn is_document(&self, gray: &[u8], _w: usize, _h: usize) -> bool {
        let total = gray.len() as f64;

        let near_white = gray.iter().filter(|&&p| p > 230).count() as f64 / total;
        let near_black = gray.iter().filter(|&&p| p < 25).count() as f64 / total;

        let background_ratio = near_white + near_black;
        if background_ratio > self.document_threshold {
            return true;
        }

        let mut histogram = [0_u32; 256];
        for &p in gray {
            histogram[p as usize] += 1;
        }

        let peak1 = histogram[..64].iter().max().copied().unwrap_or(0);
        let peak2 = histogram[192..].iter().max().copied().unwrap_or(0);
        let valley = histogram[96..160].iter().min().copied().unwrap_or(u32::MAX);

        if peak1 > 0 && peak2 > 0 {
            let min_peak = peak1.min(peak2);
            if min_peak > 0 && valley < min_peak / 2 {
                return true;
            }
        }

        false
    }

    /// Forensic features for each block.
    fn compute_block_features(
        &self,
        gray: &[u8],
        w: usize,
        _h: usize,
        blocks_x: usize,
        blocks_y: usize,
    ) -> Vec<BlockFeatures> {
        let bs = self.block_size;
        let mut features = Vec::with_capacity(blocks_x * blocks_y);

        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                let x0 = bx * bs;
                let y0 = by * bs;

                let mut sum = 0.0_f64;
                let mut sum_sq = 0.0_f64;
                let mut gradient_sum = 0.0_f64;
                let mut edge_count = 0_u32;
                let mut noise_residuals = Vec::new();
                let n = (bs * bs) as f64;

                for dy in 0..bs {
                    for dx in 0..bs {
                        let val = gray[(y0 + dy) * w + (x0 + dx)] as f64;
                        sum += val;
                        sum_sq += val * val;
                    }
                }

                let intensity_mean = sum / n;
                let intensity_var = sum_sq / n - intensity_mean * intensity_mean;

                for dy in 1..bs.saturating_sub(1) {
                    for dx in 1..bs.saturating_sub(1) {
                        let x = x0 + dx;
                        let y = y0 + dy;
                        let c = gray[y * w + x] as f64;
                        let n_val = gray[(y - 1) * w + x] as f64;
                        let s_val = gray[(y + 1) * w + x] as f64;
                        let e_val = gray[y * w + (x + 1)] as f64;
                        let w_val = gray[y * w + (x - 1)] as f64;

                        let gx = e_val - w_val;
                        let gy = s_val - n_val;
                        let grad = (gx * gx + gy * gy).sqrt();
                        gradient_sum += grad;

                        if grad > 30.0 {
                            edge_count += 1;
                        }

                        let predicted = (n_val + s_val + e_val + w_val) / 4.0;
                        noise_residuals.push(c - predicted);
                    }
                }

                let inner_pixels = ((bs - 2) * (bs - 2)) as f64;
                let sharpness = if inner_pixels > 0.0 {
                    gradient_sum / inner_pixels
                } else {
                    0.0
                };
                let edge_density = if inner_pixels > 0.0 {
                    edge_count as f64 / inner_pixels
                } else {
                    0.0
                };

                let noise_var = if !noise_residuals.is_empty() {
                    let mean_r: f64 =
                        noise_residuals.iter().sum::<f64>() / noise_residuals.len() as f64;
                    noise_residuals
                        .iter()
                        .map(|&r| (r - mean_r).powi(2))
                        .sum::<f64>()
                        / noise_residuals.len() as f64
                } else {
                    0.0
                };

                features.push(BlockFeatures {
                    noise_var,
                    edge_density,
                    sharpness,
                    intensity_mean,
                    intensity_var,
                });
            }
        }

        features
    }

    /// Per-block anomaly score vs the local neighborhood.
    fn compute_anomaly_map(
        &self,
        features: &[BlockFeatures],
        blocks_x: usize,
        blocks_y: usize,
    ) -> Vec<f64> {
        let nr = self.neighborhood as isize;
        let mut anomaly_map = vec![0.0_f64; blocks_x * blocks_y];

        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                let idx = by * blocks_x + bx;
                let block = &features[idx];

                // Skip near-uniform background blocks.
                if block.edge_density < 0.01 && block.intensity_var < 5.0 {
                    continue;
                }

                let mut neighbor_noise: Vec<f64> = Vec::new();
                let mut neighbor_sharpness: Vec<f64> = Vec::new();
                let mut neighbor_edge: Vec<f64> = Vec::new();

                for dy in -nr..=nr {
                    for dx in -nr..=nr {
                        if dx == 0 && dy == 0 {
                            continue;
                        }
                        let nx = bx as isize + dx;
                        let ny = by as isize + dy;
                        if nx >= 0
                            && ny >= 0
                            && (nx as usize) < blocks_x
                            && (ny as usize) < blocks_y
                        {
                            let ni = ny as usize * blocks_x + nx as usize;
                            let nb = &features[ni];
                            // Compare only against similar-content blocks.
                            if nb.edge_density > 0.01 || nb.intensity_var > 5.0 {
                                neighbor_noise.push(nb.noise_var);
                                neighbor_sharpness.push(nb.sharpness);
                                neighbor_edge.push(nb.edge_density);
                            }
                        }
                    }
                }

                if neighbor_noise.len() < 3 {
                    continue;
                }

                let noise_dev = Self::robust_deviation(block.noise_var, &neighbor_noise);
                let sharp_dev = Self::robust_deviation(block.sharpness, &neighbor_sharpness);
                let edge_dev = Self::robust_deviation(block.edge_density, &neighbor_edge);

                // Weight noise highest - forged content usually differs there.
                anomaly_map[idx] = noise_dev * 2.0 + sharp_dev * 1.5 + edge_dev * 1.0;
            }
        }

        anomaly_map
    }

    /// How many MADs a value deviates from the median of a set.
    fn robust_deviation(value: f64, neighbors: &[f64]) -> f64 {
        if neighbors.is_empty() {
            return 0.0;
        }

        let mut sorted = neighbors.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = sorted[sorted.len() / 2];

        let mut deviations: Vec<f64> = sorted.iter().map(|v| (v - median).abs()).collect();
        deviations.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mad = deviations[deviations.len() / 2] * 1.4826;

        if mad < 1e-6 {
            // All neighbors identical - any difference is suspicious.
            return (value - median).abs() * 10.0;
        }

        (value - median).abs() / mad
    }

    /// Cluster outlier positions via connected-component flood fill on a grid.
    fn cluster_outliers(
        &self,
        positions: &[(usize, usize, f64)],
        img_w: usize,
        img_h: usize,
    ) -> Vec<DocCluster> {
        if positions.is_empty() {
            return vec![];
        }

        // 2× block size for tolerance.
        let cell_size = self.block_size * 2;
        let grid_w = img_w.div_ceil(cell_size);
        let grid_h = img_h.div_ceil(cell_size);

        let mut grid_scores: Vec<f64> = vec![0.0; grid_w * grid_h];
        for &(x, y, score) in positions {
            let gx = x / cell_size;
            let gy = y / cell_size;
            if gx < grid_w && gy < grid_h {
                let gi = gy * grid_w + gx;
                if score > grid_scores[gi] {
                    grid_scores[gi] = score;
                }
            }
        }

        let mut labels = vec![0_u32; grid_w * grid_h];
        let mut label_id = 0_u32;

        for gy in 0..grid_h {
            for gx in 0..grid_w {
                let idx = gy * grid_w + gx;
                if grid_scores[idx] > 0.0 && labels[idx] == 0 {
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
                                    if grid_scores[ni] > 0.0 && labels[ni] == 0 {
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

        let mut clusters: Vec<DocCluster> = (0..label_id)
            .map(|_| DocCluster {
                x_min: usize::MAX,
                y_min: usize::MAX,
                x_max: 0,
                y_max: 0,
                count: 0,
                peak_score: 0.0,
            })
            .collect();

        for &(x, y, score) in positions {
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
                    if score > c.peak_score {
                        c.peak_score = score;
                    }
                }
            }
        }

        clusters.retain(|c| c.count >= 1);
        clusters.sort_by(|a, b| {
            b.peak_score
                .partial_cmp(&a.peak_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        clusters
    }
}
