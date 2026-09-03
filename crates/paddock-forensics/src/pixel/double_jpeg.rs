//! Double-JPEG-compression detection via DCT coefficient analysis, ported from
//! the CPU reference (Bianchi & Piva 2012 histogram periodicity; Li et
//! al. 2009 Benford's-Law divergence; Fan & de Queiroz 2003 blocking-grid).
//!
//! CPU-only, deliberately. The DCT-histogram core uses `rustdct`; a GPU DCT
//! would not reproduce it bit-for-bit, which would break both exact GPU==CPU
//! parity AND the reference oracle (different coefficients -> different histogram
//! bins -> different findings). The reference's own GPU path is a stub for the same
//! reason. Per the "CUDA alternative only when a real one exists" rule, there is
//! no sensible CUDA alternative here, so `gpu()` delegates to `cpu()`.

use rustdct::DctPlanner;

use crate::analyzer::Analyzer;
use crate::{Context, Finding, Severity};

const MAX_COEFF_VALUE: i32 = 1024;
const HIST_SIZE: usize = (2 * MAX_COEFF_VALUE + 1) as usize;

/// Standard JPEG luminance quantization table (Annex K).
const STANDARD_LUMA: [u16; 64] = [
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113,
    92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
];

pub struct DoubleJpegDetector;

struct CoefficientHistogram {
    position: usize,
    bins: Vec<u32>,
    offset: i32,
}

struct PeriodicityResult {
    period: u32,
    strength: f64,
    significant: bool,
}

impl Analyzer for DoubleJpegDetector {
    fn name(&self) -> &'static str {
        "double_jpeg"
    }

    fn applies_to(&self, ctx: &Context) -> bool {
        ctx.is_jpeg()
    }

    fn cpu(&self, ctx: &Context) -> Vec<Finding> {
        let mut findings = Vec::new();
        let qtables = extract_quantization_tables(&ctx.raw_bytes);
        let dct_blocks = self.compute_dct_blocks(ctx);
        if !dct_blocks.is_empty() {
            let histograms = self.build_coefficient_histograms(&dct_blocks);
            self.test_histogram_periodicity(&histograms, &qtables, &mut findings);
            self.test_benfords_law(&dct_blocks, &mut findings);
        }
        self.test_blocking_grid(ctx, &mut findings);
        if !qtables.is_empty() {
            self.analyze_qtable_forensics(&qtables, &mut findings);
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

impl DoubleJpegDetector {
    /// 8x8 DCT blocks from the grayscale image (rustdct, separable + JPEG norm).
    fn compute_dct_blocks(&self, ctx: &Context) -> Vec<[f64; 64]> {
        let (w, h) = (ctx.width as usize, ctx.height as usize);
        let gray = ctx.gray();
        let (blocks_x, blocks_y) = (w / 8, h / 8);
        let mut planner = DctPlanner::new();
        let dct = planner.plan_dct2(8);
        let s0 = (1.0 / 8.0_f64).sqrt();
        let s1 = (2.0 / 8.0_f64).sqrt();

        let mut blocks = Vec::with_capacity(blocks_x * blocks_y);
        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                let (x0, y0) = (bx * 8, by * 8);
                let mut block = [0.0_f64; 64];
                for dy in 0..8 {
                    for dx in 0..8 {
                        block[dy * 8 + dx] = gray[(y0 + dy) * w + (x0 + dx)] as f64 - 128.0;
                    }
                }
                let mut row_buf = [0.0_f64; 8];
                for row in 0..8 {
                    row_buf.copy_from_slice(&block[row * 8..(row + 1) * 8]);
                    dct.process_dct2(&mut row_buf);
                    row_buf[0] *= s0;
                    for v in row_buf.iter_mut().skip(1) {
                        *v *= s1;
                    }
                    block[row * 8..(row + 1) * 8].copy_from_slice(&row_buf);
                }
                let mut col_buf = [0.0_f64; 8];
                for col in 0..8 {
                    for row in 0..8 {
                        col_buf[row] = block[row * 8 + col];
                    }
                    dct.process_dct2(&mut col_buf);
                    col_buf[0] *= s0;
                    for v in col_buf.iter_mut().skip(1) {
                        *v *= s1;
                    }
                    for row in 0..8 {
                        block[row * 8 + col] = col_buf[row];
                    }
                }
                blocks.push(block);
            }
        }
        blocks
    }

    fn build_coefficient_histograms(&self, blocks: &[[f64; 64]]) -> Vec<CoefficientHistogram> {
        let offset = MAX_COEFF_VALUE;
        let mut histograms: Vec<CoefficientHistogram> = (1..64)
            .map(|pos| CoefficientHistogram {
                position: pos,
                bins: vec![0u32; HIST_SIZE],
                offset,
            })
            .collect();
        for block in blocks {
            for (hist_idx, hist) in histograms.iter_mut().enumerate() {
                let pos = hist_idx + 1;
                let val = block[pos].round() as i32;
                let bin = (val + offset) as usize;
                if bin < HIST_SIZE {
                    hist.bins[bin] += 1;
                }
            }
        }
        histograms
    }

    fn test_histogram_periodicity(
        &self,
        histograms: &[CoefficientHistogram],
        qtables: &[[u16; 64]],
        findings: &mut Vec<Finding>,
    ) {
        let current_q = qtables.first();
        let mut periodic_positions = 0;
        let mut total_tested = 0;
        let mut detected: Vec<(usize, u32, f64)> = Vec::new();

        for hist in histograms {
            let non_zero: u32 = hist.bins.iter().sum::<u32>() - hist.bins[hist.offset as usize];
            if non_zero < 100 {
                continue;
            }
            total_tested += 1;
            let current_step = current_q.map(|q| q[hist.position] as u32).unwrap_or(1);
            let result = self.detect_periodicity(&hist.bins, hist.offset, current_step);
            if result.significant {
                periodic_positions += 1;
                detected.push((hist.position, result.period, result.strength));
            }
        }
        if total_tested == 0 {
            return;
        }
        let periodic_ratio = periodic_positions as f64 / total_tested as f64;
        if periodic_ratio > 0.15 {
            // Deterministic dominant period: max count, then smallest period
            // (the reference used HashMap::max_by_key, whose tie order is unstable).
            let mut counts: std::collections::BTreeMap<u32, usize> =
                std::collections::BTreeMap::new();
            for &(_, period, _) in &detected {
                *counts.entry(period).or_insert(0) += 1;
            }
            let dominant_period = counts
                .iter()
                .max_by(|a, b| a.1.cmp(b.1).then(b.0.cmp(a.0)))
                .map(|(&p, _)| p)
                .unwrap_or(0);
            let avg_strength: f64 =
                detected.iter().map(|&(_, _, s)| s).sum::<f64>() / detected.len() as f64;
            findings.push(Finding::new(
                "double_jpeg",
                "double_jpeg_periodicity",
                format!(
                    "Double JPEG compression detected: {:.1}% of DCT coefficient histograms show \
                     periodic patterns (dominant period Q1≈{dominant_period}, avg strength \
                     {avg_strength:.3}) - image was decompressed and re-compressed",
                    periodic_ratio * 100.0
                ),
                Severity::High,
                (0.55 + periodic_ratio * 0.4).min(0.90),
            ));
        }
    }

    fn detect_periodicity(
        &self,
        bins: &[u32],
        offset: i32,
        current_step: u32,
    ) -> PeriodicityResult {
        let none = || PeriodicityResult {
            period: 0,
            strength: 0.0,
            significant: false,
        };
        let range = 200.min(offset as usize);
        let start = (offset as usize).saturating_sub(range);
        let end = (offset as usize + range).min(bins.len());
        let segment: Vec<f64> = bins[start..end].iter().map(|&v| v as f64).collect();
        if segment.len() < 16 {
            return none();
        }
        let n = segment.len();
        let mean: f64 = segment.iter().sum::<f64>() / n as f64;
        let detrended: Vec<f64> = segment.iter().map(|&v| v - mean).collect();

        let mut power = vec![0.0_f64; n / 2];
        for (k, p) in power.iter_mut().enumerate().skip(1) {
            let mut re = 0.0_f64;
            let mut im = 0.0_f64;
            for (i, &val) in detrended.iter().enumerate() {
                let angle = 2.0 * std::f64::consts::PI * k as f64 * i as f64 / n as f64;
                re += val * angle.cos();
                im += val * (-angle.sin());
            }
            *p = re * re + im * im;
        }
        let total_power: f64 = power.iter().sum();
        if total_power < 1e-10 {
            return none();
        }
        let min_k = 2;
        let max_k = (n / 4).min(power.len());
        let mut best_k = 0;
        let mut best_power = 0.0_f64;
        for k in min_k..max_k {
            if power[k] > best_power {
                let period = n as f64 / k as f64;
                if (period - current_step as f64).abs() < 1.0 {
                    continue;
                }
                best_power = power[k];
                best_k = k;
            }
        }
        if best_k == 0 {
            return none();
        }
        let period = (n as f64 / best_k as f64).round() as u32;
        let strength = best_power / total_power;
        let noise_floor: f64 = {
            let mut sorted: Vec<f64> = power[min_k..max_k].to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            sorted[sorted.len() / 2]
        };
        let significant = strength > 0.05 && best_power > noise_floor * 5.0 && period >= 2;
        PeriodicityResult {
            period,
            strength,
            significant,
        }
    }

    fn test_benfords_law(&self, blocks: &[[f64; 64]], findings: &mut Vec<Finding>) {
        let benford: [f64; 9] = std::array::from_fn(|i| (1.0 + 1.0 / (i as f64 + 1.0)).log10());
        let mut digit_counts = [0_u64; 9];
        let mut total = 0_u64;
        for block in blocks {
            for &coeff in block[1..].iter() {
                let abs_val = coeff.abs() as u64;
                if abs_val == 0 {
                    continue;
                }
                let mut d = abs_val;
                while d >= 10 {
                    d /= 10;
                }
                if (1..=9).contains(&d) {
                    digit_counts[(d - 1) as usize] += 1;
                    total += 1;
                }
            }
        }
        if total < 1000 {
            return;
        }
        let mut chi_squared = 0.0_f64;
        for d in 0..9 {
            let observed = digit_counts[d] as f64 / total as f64;
            let expected = benford[d];
            chi_squared += (observed - expected).powi(2) / expected;
        }
        if chi_squared > 0.02 {
            let mut kl = 0.0_f64;
            for d in 0..9 {
                let p = digit_counts[d] as f64 / total as f64;
                let q = benford[d];
                if p > 0.0 {
                    kl += p * (p / q).ln();
                }
            }
            findings.push(Finding::new(
                "double_jpeg",
                "benfords_law_violation",
                format!(
                    "DCT coefficient first-digit distribution deviates from Benford's Law \
                     (χ²={chi_squared:.4}, KL divergence={kl:.4}) - indicates double compression or \
                     non-camera-original content"
                ),
                if kl > 0.1 { Severity::High } else { Severity::Medium },
                (0.4 + (kl * 3.0).min(0.45)).min(0.85),
            ));
        }
    }

    fn test_blocking_grid(&self, ctx: &Context, findings: &mut Vec<Finding>) {
        let (w, h) = (ctx.width as usize, ctx.height as usize);
        if w < 32 || h < 32 {
            return;
        }
        let gray = ctx.gray();
        let mut grid = [[0.0_f64; 8]; 8];
        let step = if w * h > 500_000 { 4 } else { 1 };
        for offset_y in 0..8usize {
            for offset_x in 0..8usize {
                let (mut bsum, mut bcnt, mut isum, mut icnt) = (0.0_f64, 0u64, 0.0_f64, 0u64);
                for y in (0..h).step_by(step) {
                    for x in 1..w {
                        let diff = (gray[y * w + x] as f64 - gray[y * w + x - 1] as f64).abs();
                        if (x + offset_x) % 8 == 0 {
                            bsum += diff;
                            bcnt += 1;
                        } else {
                            isum += diff;
                            icnt += 1;
                        }
                    }
                }
                for y in 1..h {
                    for x in (0..w).step_by(step) {
                        let diff = (gray[y * w + x] as f64 - gray[(y - 1) * w + x] as f64).abs();
                        if (y + offset_y) % 8 == 0 {
                            bsum += diff;
                            bcnt += 1;
                        } else {
                            isum += diff;
                            icnt += 1;
                        }
                    }
                }
                if bcnt > 0 && icnt > 0 {
                    let (ba, ia) = (bsum / bcnt as f64, isum / icnt as f64);
                    grid[offset_y][offset_x] = if ia > 0.0 { ba / ia } else { 0.0 };
                }
            }
        }
        let standard = grid[0][0];
        let mut best_off = (0usize, 0usize);
        let mut best = 0.0_f64;
        for oy in 0..8usize {
            for ox in 0..8usize {
                if (oy, ox) == (0, 0) {
                    continue;
                }
                if grid[oy][ox] > best {
                    best = grid[oy][ox];
                    best_off = (ox, oy);
                }
            }
        }
        if best > standard * 1.05 && best > 1.02 {
            findings.push(Finding::new(
                "double_jpeg",
                "shifted_jpeg_grid",
                format!(
                    "JPEG blocking grid at offset ({},{}) is {:.1}% stronger than standard (0,0) \
                     grid - image was JPEG-compressed, cropped by non-8-pixel boundary, then \
                     re-compressed",
                    best_off.0,
                    best_off.1,
                    (best / standard - 1.0) * 100.0
                ),
                Severity::High,
                0.80,
            ));
        }
    }

    fn analyze_qtable_forensics(&self, qtables: &[[u16; 64]], findings: &mut Vec<Finding>) {
        for (idx, qtable) in qtables.iter().enumerate() {
            let mut ratios = Vec::new();
            for i in 0..64 {
                if STANDARD_LUMA[i] > 0 && qtable[i] > 0 {
                    ratios.push(qtable[i] as f64 / STANDARD_LUMA[i] as f64);
                }
            }
            if ratios.is_empty() {
                continue;
            }
            ratios.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let median_ratio = ratios[ratios.len() / 2];
            let residuals: Vec<f64> = ratios
                .iter()
                .map(|&r| ((r / median_ratio) - 1.0).abs())
                .collect();
            let mean_residual: f64 = residuals.iter().sum::<f64>() / residuals.len() as f64;
            if mean_residual > 0.15 {
                findings.push(Finding::new(
                    "double_jpeg",
                    "non_standard_quantization",
                    format!(
                        "Quantization table {idx} is not a uniform scaling of standard JPEG tables \
                         (mean residual {mean_residual:.3}) - may indicate custom processing or \
                         double compression"
                    ),
                    Severity::Medium,
                    0.55,
                ));
            }
            let estimated_quality = if median_ratio <= 1.0 {
                (50.0 / median_ratio).round().min(100.0) as u8
            } else {
                (50.0 * (2.0 - median_ratio)).round().max(1.0) as u8
            };
            findings.push(Finding::new(
                "double_jpeg",
                "jpeg_quality_estimate",
                format!("Estimated JPEG quality factor: ~{estimated_quality} (table {idx})"),
                Severity::Info,
                0.85,
            ));
        }
    }
}

/// Parse JPEG DQT markers into quantization tables.
fn extract_quantization_tables(data: &[u8]) -> Vec<[u16; 64]> {
    let mut tables = Vec::new();
    let mut pos = 2;
    while pos + 4 < data.len() {
        if data[pos] != 0xFF {
            pos += 1;
            continue;
        }
        let marker = data[pos + 1];
        pos += 2;
        if marker == 0x00 || marker == 0xFF || (0xD0..=0xD9).contains(&marker) {
            continue;
        }
        if pos + 2 > data.len() {
            break;
        }
        let length = u16::from_be_bytes([data[pos], data[pos + 1]]) as usize;
        if length < 2 || pos + length > data.len() {
            break;
        }
        if marker == 0xDB {
            let mut tpos = pos + 2;
            while tpos < pos + length {
                if tpos >= data.len() {
                    break;
                }
                let precision = (data[tpos] >> 4) & 0x0F;
                tpos += 1;
                let mut table = [0u16; 64];
                for entry in table.iter_mut() {
                    if precision == 0 {
                        if tpos >= data.len() {
                            break;
                        }
                        *entry = data[tpos] as u16;
                        tpos += 1;
                    } else {
                        if tpos + 1 >= data.len() {
                            break;
                        }
                        *entry = u16::from_be_bytes([data[tpos], data[tpos + 1]]);
                        tpos += 2;
                    }
                }
                tables.push(table);
            }
        }
        if marker == 0xDA {
            break;
        }
        pos += length;
    }
    tables
}
