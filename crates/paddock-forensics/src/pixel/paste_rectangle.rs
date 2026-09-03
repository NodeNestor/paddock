//! Rectangular paste detection via OCR-masked noise-correlation analysis, ported
//! verbatim from the CPU reference. CPU-only (integral-image filters +
//! tiled Pearson correlation; no GPU kernel), `gpu()` delegates.
//!
//! A text stroke mask (nordocr-style adaptive threshold) isolates paper-only
//! pixels; at a paste boundary adjacent paper pixels carry uncorrelated noise
//! (different sensor captures). Masking is essential - unmasked text edges swamp
//! the subtle boundary signal. Pipeline: adaptive threshold -> box-filter noise
//! residual -> tiled paper-only correlation edges -> pair into rectangles ->
//! validate with a noise-variance ratio -> NMS. Photo/document raster only:
//! skipped for PDFs.

use crate::analyzer::Analyzer;
use crate::{Context, Finding, Region, Severity};

pub struct PasteRectangleDetector {
    /// Block size for adaptive thresholding (nordocr). Default 15.
    adaptive_block_size: usize,
    /// Constant subtracted from the local mean (nordocr). Default 4.0.
    adaptive_c: f64,
    /// Minimum rectangle dimension (pixels).
    min_rect: usize,
    /// Maximum rectangle dimension (pixels).
    max_rect: usize,
}

impl Default for PasteRectangleDetector {
    fn default() -> Self {
        Self {
            adaptive_block_size: 15,
            adaptive_c: 4.0,
            min_rect: 50,
            max_rect: 800,
        }
    }
}

impl Analyzer for PasteRectangleDetector {
    fn name(&self) -> &'static str {
        "paste_rectangle"
    }

    fn applies_to(&self, ctx: &Context) -> bool {
        !ctx.is_pdf()
    }

    fn cpu(&self, ctx: &Context) -> Vec<Finding> {
        let w = ctx.width as usize;
        let h = ctx.height as usize;

        if w < self.min_rect * 3 || h < self.min_rect * 3 {
            return vec![];
        }

        let gray = ctx.gray();

        // Downsample large images to ~6M pixels.
        let (work, work_w, work_h, scale) = if w * h > 6_000_000 {
            let s = ((w * h) as f64 / 6_000_000.0).sqrt();
            let nw = (w as f64 / s) as usize;
            let nh = (h as f64 / s) as usize;
            let mut dst = vec![0u8; nw * nh];
            for y in 0..nh {
                for x in 0..nw {
                    let sx = (x as f64 * s) as usize;
                    let sy = (y as f64 * s) as usize;
                    dst[y * nw + x] = gray[sy.min(h - 1) * w + sx.min(w - 1)];
                }
            }
            (dst, nw, nh, s)
        } else {
            (gray.to_vec(), w, h, 1.0)
        };

        // Step 1: text stroke mask (nordocr-style adaptive threshold).
        let mut blurred = vec![0u8; work_w * work_h];
        gaussian_blur_5x5(&work, &mut blurred, work_w, work_h);
        let text_mask = adaptive_threshold_mean(
            &blurred,
            work_w,
            work_h,
            self.adaptive_block_size,
            self.adaptive_c,
        );
        // Dilate to cover anti-aliased edges.
        let text_mask = dilate_binary(&text_mask, work_w, work_h, 3);

        // Step 2: noise residual via box filter.
        let noise = fast_noise_residual(&work, work_w, work_h, 5);

        // Step 3: paper-only noise-correlation edges in tiles.
        let v_edges = find_paper_noise_edges_v(&noise, &text_mask, work_w, work_h);
        let h_edges = find_paper_noise_edges_h(&noise, &text_mask, work_w, work_h);

        // Step 4: form rectangles from edge pairs.
        let min_r = (self.min_rect as f64 / scale) as usize;
        let max_r = (self.max_rect as f64 / scale) as usize;
        let mut candidates = form_rectangles(&v_edges, &h_edges, min_r, max_r);

        // Step 5: validate with the noise-variance ratio.
        for cand in &mut candidates {
            let var_score =
                noise_variance_ratio(&noise, work_w, work_h, cand.0, cand.1, cand.2, cand.3);
            cand.4 = cand.4 * 0.5 + var_score * 0.5;
        }

        candidates.retain(|c| c.4 > 0.4);
        candidates.sort_by(|a, b| b.4.partial_cmp(&a.4).unwrap_or(std::cmp::Ordering::Equal));

        let kept = nms(&candidates, 0.3);

        let mut findings = Vec::new();
        for (i, &(x, y, rw, rh, score)) in kept.iter().take(5).enumerate() {
            let orig_x = (x as f64 * scale) as u32;
            let orig_y = (y as f64 * scale) as u32;
            let orig_w = (rw as f64 * scale) as u32;
            let orig_h = (rh as f64 * scale) as u32;
            let confidence = (0.4 + score * 0.15).clamp(0.3, 0.95);

            findings.push(
                Finding::new(
                    "paste_rectangle",
                    "paste_rectangle_detected",
                    format!(
                        "Noise correlation boundary discontinuity {} at ({orig_x},{orig_y})->({},{}) - \
                         rectangular region where noise patterns change abruptly at edges, \
                         indicating copy-pasted content (score {score:.2})",
                        i + 1,
                        orig_x + orig_w,
                        orig_y + orig_h,
                    ),
                    if score > 2.0 {
                        Severity::Critical
                    } else if score > 1.0 {
                        Severity::High
                    } else {
                        Severity::Medium
                    },
                    confidence,
                )
                .with_region(Region::BoundingBox {
                    x: orig_x,
                    y: orig_y,
                    width: orig_w,
                    height: orig_h,
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

// ── Noise residual ──────────────────────────────────────────────────────────

/// O(n) noise residual via integral-image box-filter subtraction.
fn fast_noise_residual(gray: &[u8], w: usize, h: usize, radius: usize) -> Vec<f64> {
    let iw = w + 1;
    let mut integral = vec![0.0_f64; iw * (h + 1)];
    for y in 0..h {
        let mut row_sum = 0.0_f64;
        for x in 0..w {
            row_sum += gray[y * w + x] as f64;
            integral[(y + 1) * iw + (x + 1)] = row_sum + integral[y * iw + (x + 1)];
        }
    }

    let mut noise = vec![0.0_f64; w * h];
    for y in 0..h {
        for x in 0..w {
            let y1 = y.saturating_sub(radius);
            let x1 = x.saturating_sub(radius);
            let y2 = (y + radius + 1).min(h);
            let x2 = (x + radius + 1).min(w);
            let area = ((y2 - y1) * (x2 - x1)) as f64;
            let sum = integral[y2 * iw + x2] - integral[y1 * iw + x2] - integral[y2 * iw + x1]
                + integral[y1 * iw + x1];
            noise[y * w + x] = gray[y * w + x] as f64 - sum / area;
        }
    }
    noise
}

// ── Text masking (adapted from nordocr) ─────────────────────────────────────

/// Separable 5×5 Gaussian blur, binomial kernel [1,4,6,4,1]/16.
fn gaussian_blur_5x5(input: &[u8], output: &mut [u8], w: usize, h: usize) {
    const WEIGHTS: [u32; 5] = [1, 4, 6, 4, 1];
    const SUM: u32 = 16;
    let mut temp = vec![0u8; w * h];

    for y in 0..h {
        for x in 0..w {
            let mut acc = 0u32;
            for k in 0..5usize {
                let sx = (x as isize + k as isize - 2).clamp(0, w as isize - 1) as usize;
                acc += input[y * w + sx] as u32 * WEIGHTS[k];
            }
            temp[y * w + x] = (acc / SUM) as u8;
        }
    }
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0u32;
            for k in 0..5usize {
                let sy = (y as isize + k as isize - 2).clamp(0, h as isize - 1) as usize;
                acc += temp[sy * w + x] as u32 * WEIGHTS[k];
            }
            output[y * w + x] = (acc / SUM) as u8;
        }
    }
}

/// Adaptive mean thresholding via integral images. `true` = dark text pixel.
fn adaptive_threshold_mean(
    gray: &[u8],
    w: usize,
    h: usize,
    block_size: usize,
    c: f64,
) -> Vec<bool> {
    let iw = w + 1;
    let mut integral = vec![0i64; iw * (h + 1)];
    for y in 0..h {
        let mut row_sum = 0i64;
        for x in 0..w {
            row_sum += gray[y * w + x] as i64;
            integral[(y + 1) * iw + (x + 1)] = row_sum + integral[y * iw + (x + 1)];
        }
    }

    let half = (block_size / 2) as isize;
    let mut mask = vec![false; w * h];

    for y in 0..h {
        for x in 0..w {
            let y0 = (y as isize - half).max(0) as usize;
            let x0 = (x as isize - half).max(0) as usize;
            let y1 = ((y as isize + half).min(h as isize - 1) + 1) as usize;
            let x1 = ((x as isize + half).min(w as isize - 1) + 1) as usize;
            let area = ((y1 - y0) * (x1 - x0)) as f64;
            let sum = integral[y1 * iw + x1] as f64
                - integral[y0 * iw + x1] as f64
                - integral[y1 * iw + x0] as f64
                + integral[y0 * iw + x0] as f64;
            let mean = sum / area;
            // BINARY_INV: dark text below threshold -> true.
            mask[y * w + x] = (gray[y * w + x] as f64) <= mean - c;
        }
    }
    mask
}

/// Box dilation of a boolean mask with a square kernel.
fn dilate_binary(mask: &[bool], w: usize, h: usize, kernel: usize) -> Vec<bool> {
    let half = kernel / 2;
    let mut out = vec![false; w * h];
    for y in 0..h {
        for x in 0..w {
            if mask[y * w + x] {
                for dy in 0..kernel {
                    for dx in 0..kernel {
                        let ny = (y as isize + dy as isize - half as isize).clamp(0, h as isize - 1)
                            as usize;
                        let nx = (x as isize + dx as isize - half as isize).clamp(0, w as isize - 1)
                            as usize;
                        out[ny * w + nx] = true;
                    }
                }
            }
        }
    }
    out
}

// ── Paper-only edge detection ───────────────────────────────────────────────

/// Vertical paste edges: paper-only cross-column noise correlation in tiles.
fn find_paper_noise_edges_v(
    noise: &[f64],
    text_mask: &[bool],
    w: usize,
    h: usize,
) -> Vec<(usize, f64)> {
    if w < 20 || h < 20 {
        return vec![];
    }

    let tile = 128.min(h).min(w);
    let stride = tile / 2;

    let mut col_scores = vec![0.0_f64; w.saturating_sub(1)];
    let mut col_counts = vec![0_u32; w.saturating_sub(1)];

    let mut y_start = 0;
    while y_start + tile <= h {
        let mut x_start = 0;
        while x_start + tile <= w {
            let mut corrs = Vec::new();
            for x in 0..tile - 1 {
                let ax = x_start + x;
                let mut c1 = Vec::new();
                let mut c2 = Vec::new();
                for y in y_start..y_start + tile {
                    if !text_mask[y * w + ax] && !text_mask[y * w + ax + 1] {
                        c1.push(noise[y * w + ax]);
                        c2.push(noise[y * w + ax + 1]);
                    }
                }
                if c1.len() >= 12 {
                    let corr = pearson_correlation(&c1, &c2);
                    if corr.is_finite() {
                        corrs.push((ax, corr));
                    }
                }
            }

            if corrs.len() >= 5 {
                let vals: Vec<f64> = corrs.iter().map(|c| c.1).collect();
                let mean: f64 = vals.iter().sum::<f64>() / vals.len() as f64;
                let std: f64 = (vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                    / vals.len() as f64)
                    .sqrt();
                if std > 0.02 {
                    for &(ax, corr) in &corrs {
                        let z = (mean - corr) / std;
                        if z > 1.5 {
                            col_scores[ax] += z;
                            col_counts[ax] += 1;
                        }
                    }
                }
            }

            x_start += stride;
        }
        y_start += stride;
    }

    let mut avg = vec![0.0_f64; w.saturating_sub(1)];
    for x in 0..avg.len() {
        if col_counts[x] > 0 {
            avg[x] = col_scores[x] / col_counts[x] as f64;
        }
    }
    let smoothed = smooth(&avg, 5);
    find_anomaly_peaks(&smoothed, 15, 2.0)
}

/// Horizontal paste edges: paper-only cross-row noise correlation in tiles.
fn find_paper_noise_edges_h(
    noise: &[f64],
    text_mask: &[bool],
    w: usize,
    h: usize,
) -> Vec<(usize, f64)> {
    if w < 20 || h < 20 {
        return vec![];
    }

    let tile = 128.min(h).min(w);
    let stride = tile / 2;

    let mut row_scores = vec![0.0_f64; h.saturating_sub(1)];
    let mut row_counts = vec![0_u32; h.saturating_sub(1)];

    let mut y_start = 0;
    while y_start + tile <= h {
        let mut x_start = 0;
        while x_start + tile <= w {
            let mut corrs = Vec::new();
            for y in 0..tile - 1 {
                let ay = y_start + y;
                let mut r1 = Vec::new();
                let mut r2 = Vec::new();
                for x in x_start..x_start + tile {
                    if !text_mask[ay * w + x] && !text_mask[(ay + 1) * w + x] {
                        r1.push(noise[ay * w + x]);
                        r2.push(noise[(ay + 1) * w + x]);
                    }
                }
                if r1.len() >= 12 {
                    let corr = pearson_correlation(&r1, &r2);
                    if corr.is_finite() {
                        corrs.push((ay, corr));
                    }
                }
            }

            if corrs.len() >= 5 {
                let vals: Vec<f64> = corrs.iter().map(|c| c.1).collect();
                let mean: f64 = vals.iter().sum::<f64>() / vals.len() as f64;
                let std: f64 = (vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                    / vals.len() as f64)
                    .sqrt();
                if std > 0.02 {
                    for &(ay, corr) in &corrs {
                        let z = (mean - corr) / std;
                        if z > 1.5 {
                            row_scores[ay] += z;
                            row_counts[ay] += 1;
                        }
                    }
                }
            }

            x_start += stride;
        }
        y_start += stride;
    }

    let mut avg = vec![0.0_f64; h.saturating_sub(1)];
    for y in 0..avg.len() {
        if row_counts[y] > 0 {
            avg[y] = row_scores[y] / row_counts[y] as f64;
        }
    }
    let smoothed = smooth(&avg, 5);
    find_anomaly_peaks(&smoothed, 15, 2.0)
}

// ── Rectangle formation ─────────────────────────────────────────────────────

fn form_rectangles(
    v_edges: &[(usize, f64)],
    h_edges: &[(usize, f64)],
    min_size: usize,
    max_size: usize,
) -> Vec<(usize, usize, usize, usize, f64)> {
    let mut rects = Vec::new();
    let max_edges = 20;

    for i in 0..v_edges.len().min(max_edges) {
        for j in (i + 1)..v_edges.len().min(max_edges) {
            let (x1, sx1) = v_edges[i];
            let (x2, sx2) = v_edges[j];
            let (xl, xr) = if x1 < x2 { (x1, x2) } else { (x2, x1) };
            let rw = xr - xl;
            if rw < min_size || rw > max_size {
                continue;
            }

            for k in 0..h_edges.len().min(max_edges) {
                for l in (k + 1)..h_edges.len().min(max_edges) {
                    let (y1, sy1) = h_edges[k];
                    let (y2, sy2) = h_edges[l];
                    let (yt, yb) = if y1 < y2 { (y1, y2) } else { (y2, y1) };
                    let rh = yb - yt;
                    if rh < min_size || rh > max_size {
                        continue;
                    }

                    let edge_score = (sx1 * sx2 * sy1 * sy2).sqrt().sqrt();
                    rects.push((xl, yt, rw, rh, edge_score));
                }
            }
        }
    }

    rects
}

// ── Validation scoring ──────────────────────────────────────────────────────

fn noise_variance_ratio(
    noise: &[f64],
    w: usize,
    h: usize,
    x: usize,
    y: usize,
    rw: usize,
    rh: usize,
) -> f64 {
    let pad = 15;
    let mut in_sum = 0.0_f64;
    let mut in_n = 0_u32;
    let mut out_sum = 0.0_f64;
    let mut out_n = 0_u32;

    for py in (y + 3)..(y + rh).saturating_sub(3).min(h) {
        for px in (x + 3)..(x + rw).saturating_sub(3).min(w) {
            let v = noise[py * w + px];
            in_sum += v * v;
            in_n += 1;
        }
    }

    // Outside: top / bottom / left / right strips.
    for py in y.saturating_sub(pad)..y.min(h) {
        for px in x..(x + rw).min(w) {
            let v = noise[py * w + px];
            out_sum += v * v;
            out_n += 1;
        }
    }
    for py in (y + rh).min(h)..(y + rh + pad).min(h) {
        for px in x..(x + rw).min(w) {
            let v = noise[py * w + px];
            out_sum += v * v;
            out_n += 1;
        }
    }
    for py in y..(y + rh).min(h) {
        for px in x.saturating_sub(pad)..x.min(w) {
            let v = noise[py * w + px];
            out_sum += v * v;
            out_n += 1;
        }
        for px in (x + rw).min(w)..(x + rw + pad).min(w) {
            let v = noise[py * w + px];
            out_sum += v * v;
            out_n += 1;
        }
    }

    if in_n < 10 || out_n < 10 {
        return 0.0;
    }

    let in_var = in_sum / in_n as f64;
    let out_var = out_sum / out_n as f64;
    if out_var < 0.01 {
        return 0.0;
    }

    let log_ratio = (in_var / out_var).ln().abs();
    (log_ratio * 2.0).min(3.0)
}

// ── Utilities ───────────────────────────────────────────────────────────────

fn nms(
    candidates: &[(usize, usize, usize, usize, f64)],
    iou_threshold: f64,
) -> Vec<(usize, usize, usize, usize, f64)> {
    let mut kept = Vec::new();
    for cand in candidates {
        let dominated = kept.iter().any(|k: &(usize, usize, usize, usize, f64)| {
            let ox1 = cand.0.max(k.0);
            let oy1 = cand.1.max(k.1);
            let ox2 = (cand.0 + cand.2).min(k.0 + k.2);
            let oy2 = (cand.1 + cand.3).min(k.1 + k.3);
            if ox2 > ox1 && oy2 > oy1 {
                let overlap = (ox2 - ox1) * (oy2 - oy1);
                let area = cand.2 * cand.3;
                overlap as f64 / area as f64 > iou_threshold
            } else {
                false
            }
        });
        if !dominated {
            kept.push(*cand);
        }
    }
    kept
}

fn smooth(data: &[f64], window: usize) -> Vec<f64> {
    let half = window / 2;
    let mut out = vec![0.0; data.len()];
    for i in 0..data.len() {
        let start = i.saturating_sub(half);
        let end = (i + half + 1).min(data.len());
        let sum: f64 = data[start..end].iter().sum();
        out[i] = sum / (end - start) as f64;
    }
    out
}

fn find_anomaly_peaks(
    signal: &[f64],
    min_distance: usize,
    sigma_threshold: f64,
) -> Vec<(usize, f64)> {
    if signal.len() < 5 {
        return vec![];
    }

    let mean: f64 = signal.iter().sum::<f64>() / signal.len() as f64;
    let std: f64 =
        (signal.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / signal.len() as f64).sqrt();
    let threshold = mean + sigma_threshold * std;

    let mut peaks = Vec::new();
    for i in 2..signal.len() - 2 {
        if signal[i] > threshold && signal[i] >= signal[i - 1] && signal[i] >= signal[i + 1] {
            let too_close = peaks.iter().any(|(p, _): &(usize, f64)| {
                (*p as isize - i as isize).unsigned_abs() < min_distance
            });
            if !too_close {
                peaks.push((i, signal[i]));
            }
        }
    }

    peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    peaks.truncate(30);
    peaks
}

fn pearson_correlation(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n < 3 {
        return 1.0;
    }

    let n_f = n as f64;
    let mean_a: f64 = a[..n].iter().sum::<f64>() / n_f;
    let mean_b: f64 = b[..n].iter().sum::<f64>() / n_f;

    let mut cov = 0.0_f64;
    let mut var_a = 0.0_f64;
    let mut var_b = 0.0_f64;
    for i in 0..n {
        let da = a[i] - mean_a;
        let db = b[i] - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }

    let denom = (var_a * var_b).sqrt();
    if denom < 1e-10 {
        return 1.0;
    }

    cov / denom
}
