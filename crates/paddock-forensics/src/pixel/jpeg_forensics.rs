//! JPEG bitstream-level forensics via quantized-coefficient recovery, ported
//! verbatim from the CPU reference. CPU-only (rustdct 8×8 DCT + residual
//! statistics; a GPU DCT would not reproduce it bit-for-bit - the same reason
//! double_jpeg is CPU-only), `gpu()` delegates.
//!
//! jpeg-decoder does not expose raw DCT coefficients, so we recover them:
//! DCT(pixels) / QTable ≈ the original quantized integers (Fridrich et al.).
//! The fractional part reveals compression history - single compression clusters
//! at 0.0; pasted non-JPEG content loses grid alignment; double compression goes
//! bimodal. JPEG-only (skipped for non-JPEG).

use rustdct::DctPlanner;

use crate::analyzer::Analyzer;
use crate::{Context, Finding, Severity};

pub struct JpegForensicsAnalyzer {
    #[allow(dead_code)]
    block_size_threshold: f64,
}

impl Default for JpegForensicsAnalyzer {
    fn default() -> Self {
        Self {
            block_size_threshold: 0.1,
        }
    }
}

impl Analyzer for JpegForensicsAnalyzer {
    fn name(&self) -> &'static str {
        "jpeg_forensics"
    }

    fn applies_to(&self, ctx: &Context) -> bool {
        ctx.is_jpeg()
    }

    fn cpu(&self, ctx: &Context) -> Vec<Finding> {
        // Belt-and-braces SOI check (applies_to already gates JPEG).
        if ctx.raw_bytes.len() < 2 || ctx.raw_bytes[0] != 0xFF || ctx.raw_bytes[1] != 0xD8 {
            return vec![];
        }

        let qtables = Self::extract_quantization_tables(&ctx.raw_bytes);
        if qtables.is_empty() {
            return vec![];
        }

        let w = ctx.width as usize;
        let h = ctx.height as usize;
        let gray = ctx.gray();

        let blocks_x = w / 8;
        let blocks_y = h / 8;

        if blocks_x < 4 || blocks_y < 4 {
            return vec![];
        }

        let qtable = &qtables[0]; // Luminance table.

        let block_stats = self.compute_block_residual_stats(gray, w, h, qtable);

        let mut findings = Vec::new();
        self.detect_artifact_free_regions(&block_stats, blocks_x, blocks_y, &mut findings);
        self.detect_quantization_inconsistency(
            &block_stats,
            blocks_x,
            blocks_y,
            qtable,
            &mut findings,
        );
        self.detect_double_compression_coefficients(&block_stats, &mut findings);

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

/// Per-block quantization-residual statistics.
struct BlockResidualStats {
    /// Mean fractional part of DCT/QTable (0.0 = perfect grid alignment).
    mean_fractional: f64,
    /// Standard deviation of the fractional parts.
    #[allow(dead_code)]
    std_fractional: f64,
    /// Fraction of coefficients landing very close to an integer.
    integer_alignment_ratio: f64,
    /// Number of non-zero quantized coefficients.
    non_zero_count: usize,
    /// Mean absolute value of the quantized coefficients.
    #[allow(dead_code)]
    mean_abs_quantized: f64,
}

impl JpegForensicsAnalyzer {
    /// Quantization-residual statistics for each 8×8 block.
    fn compute_block_residual_stats(
        &self,
        gray: &[u8],
        w: usize,
        h: usize,
        qtable: &[u16; 64],
    ) -> Vec<BlockResidualStats> {
        let blocks_x = w / 8;
        let blocks_y = h / 8;

        let mut planner = DctPlanner::new();
        let dct = planner.plan_dct2(8);

        let mut stats = Vec::with_capacity(blocks_x * blocks_y);

        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                let x0 = bx * 8;
                let y0 = by * 8;

                // Extract + level-shift the block.
                let mut block = [0.0_f64; 64];
                for dy in 0..8 {
                    for dx in 0..8 {
                        block[dy * 8 + dx] = gray[(y0 + dy) * w + (x0 + dx)] as f64 - 128.0;
                    }
                }

                // Separable 2D DCT-II with orthonormal scaling.
                let mut row_buf = [0.0_f64; 8];
                for row in 0..8 {
                    row_buf.copy_from_slice(&block[row * 8..(row + 1) * 8]);
                    dct.process_dct2(&mut row_buf);
                    row_buf[0] *= (1.0 / 8.0_f64).sqrt();
                    for v in row_buf[1..].iter_mut() {
                        *v *= (2.0 / 8.0_f64).sqrt();
                    }
                    block[row * 8..(row + 1) * 8].copy_from_slice(&row_buf);
                }

                let mut col_buf = [0.0_f64; 8];
                for col in 0..8 {
                    for row in 0..8 {
                        col_buf[row] = block[row * 8 + col];
                    }
                    dct.process_dct2(&mut col_buf);
                    col_buf[0] *= (1.0 / 8.0_f64).sqrt();
                    for v in col_buf[1..].iter_mut() {
                        *v *= (2.0 / 8.0_f64).sqrt();
                    }
                    for row in 0..8 {
                        block[row * 8 + col] = col_buf[row];
                    }
                }

                // Divide by the Q-table and analyze the residuals.
                let mut frac_sum = 0.0_f64;
                let mut frac_sq_sum = 0.0_f64;
                let mut integer_aligned = 0_usize;
                let mut non_zero = 0_usize;
                let mut abs_sum = 0.0_f64;
                let mut ac_count = 0_usize;

                for i in 1..64 {
                    // Skip the DC coefficient.
                    let q = qtable[i] as f64;
                    if q < 1.0 {
                        continue;
                    }

                    let quantized_continuous = block[i] / q;
                    let quantized_integer = quantized_continuous.round();
                    let fractional = (quantized_continuous - quantized_integer).abs();

                    frac_sum += fractional;
                    frac_sq_sum += fractional * fractional;
                    ac_count += 1;

                    if quantized_integer.abs() > 0.5 {
                        non_zero += 1;
                        abs_sum += quantized_integer.abs();
                    }

                    if fractional < 0.1 {
                        integer_aligned += 1;
                    }
                }

                let n = ac_count as f64;
                let mean_frac = if n > 0.0 { frac_sum / n } else { 0.5 };
                let var_frac = if n > 1.0 {
                    (frac_sq_sum / n) - (mean_frac * mean_frac)
                } else {
                    0.0
                };

                stats.push(BlockResidualStats {
                    mean_fractional: mean_frac,
                    std_fractional: var_frac.max(0.0).sqrt(),
                    integer_alignment_ratio: if ac_count > 0 {
                        integer_aligned as f64 / ac_count as f64
                    } else {
                        0.0
                    },
                    non_zero_count: non_zero,
                    mean_abs_quantized: if non_zero > 0 {
                        abs_sum / non_zero as f64
                    } else {
                        0.0
                    },
                });
            }
        }

        stats
    }

    /// Blocks whose DCT coefficients do not align with the quantization grid -
    /// a strong sign of pasted non-JPEG content.
    fn detect_artifact_free_regions(
        &self,
        stats: &[BlockResidualStats],
        _blocks_x: usize,
        _blocks_y: usize,
        findings: &mut Vec<Finding>,
    ) {
        if stats.is_empty() {
            return;
        }

        let mean_alignment: f64 =
            stats.iter().map(|s| s.integer_alignment_ratio).sum::<f64>() / stats.len() as f64;

        // Only meaningful if the image overall shows JPEG artifacts.
        if mean_alignment < 0.5 {
            return;
        }

        let alignment_threshold = (mean_alignment * 0.6).max(0.3);
        let artifact_free: Vec<usize> = stats
            .iter()
            .enumerate()
            .filter(|(_, s)| {
                s.integer_alignment_ratio < alignment_threshold && s.non_zero_count > 5
            })
            .map(|(i, _)| i)
            .collect();

        let af_ratio = artifact_free.len() as f64 / stats.len() as f64;

        if af_ratio > 0.03 && af_ratio < 0.6 {
            findings.push(Finding::new(
                "jpeg_forensics",
                "jpeg_artifact_free_regions",
                format!(
                    "{:.1}% of blocks lack JPEG quantization grid alignment \
                     (alignment ratio below {alignment_threshold:.2} vs image mean {mean_alignment:.2}) - \
                     these regions were likely pasted from a non-JPEG source \
                     (PNG, screenshot, AI output)",
                    af_ratio * 100.0,
                ),
                Severity::Critical,
                (0.6 + af_ratio * 0.5).min(0.90),
            ));
        }
    }

    /// Blocks with quantization characteristics unlike the majority.
    fn detect_quantization_inconsistency(
        &self,
        stats: &[BlockResidualStats],
        _blocks_x: usize,
        _blocks_y: usize,
        _qtable: &[u16; 64],
        findings: &mut Vec<Finding>,
    ) {
        if stats.is_empty() {
            return;
        }

        let mut sorted_frac: Vec<f64> = stats.iter().map(|s| s.mean_fractional).collect();
        sorted_frac.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let median_frac = sorted_frac[sorted_frac.len() / 2];
        let q1 = sorted_frac[sorted_frac.len() / 4];
        let q3 = sorted_frac[3 * sorted_frac.len() / 4];
        let iqr = q3 - q1;

        if iqr < 0.01 {
            return; // Very uniform - single compression.
        }

        let upper_fence = q3 + 1.5 * iqr;
        let outlier_blocks = stats
            .iter()
            .filter(|s| s.mean_fractional > upper_fence)
            .count();
        let outlier_ratio = outlier_blocks as f64 / stats.len() as f64;

        if outlier_ratio > 0.05 && outlier_ratio < 0.5 {
            findings.push(Finding::new(
                "jpeg_forensics",
                "quantization_inconsistency",
                format!(
                    "{:.1}% of blocks show anomalous DCT quantization residuals \
                     (median fractional {median_frac:.3}, IQR {iqr:.3}, {outlier_blocks} outlier blocks) - \
                     indicates regions with different compression history",
                    outlier_ratio * 100.0,
                ),
                Severity::High,
                (0.55 + outlier_ratio * 0.5).min(0.85),
            ));
        }
    }

    /// Double compression from coefficient-level analysis: the fractional
    /// residual distribution goes bimodal rather than unimodal.
    fn detect_double_compression_coefficients(
        &self,
        stats: &[BlockResidualStats],
        findings: &mut Vec<Finding>,
    ) {
        if stats.len() < 50 {
            return;
        }

        let num_bins = 50;
        let mut histogram = vec![0_u32; num_bins];

        for stat in stats {
            let bin = (stat.mean_fractional * num_bins as f64) as usize;
            let bin = bin.min(num_bins - 1);
            histogram[bin] += 1;
        }

        let smoothed = Self::smooth_histogram(&histogram, 3);
        let peaks = Self::count_peaks(&smoothed);

        if peaks >= 2 {
            findings.push(Finding::new(
                "jpeg_forensics",
                "bimodal_quantization_residual",
                format!(
                    "Bimodal distribution of DCT quantization residuals ({peaks} peaks detected) - \
                     image contains blocks with different compression histories, \
                     strong indicator of editing or compositing"
                ),
                Severity::High,
                0.75,
            ));
        }
    }

    fn smooth_histogram(hist: &[u32], radius: usize) -> Vec<f64> {
        hist.iter()
            .enumerate()
            .map(|(i, _)| {
                let start = i.saturating_sub(radius);
                let end = (i + radius + 1).min(hist.len());
                let sum: f64 = hist[start..end].iter().map(|&v| v as f64).sum();
                sum / (end - start) as f64
            })
            .collect()
    }

    fn count_peaks(smoothed: &[f64]) -> usize {
        if smoothed.len() < 3 {
            return 0;
        }

        let max_val = smoothed.iter().cloned().fold(0.0_f64, f64::max);
        let threshold = max_val * 0.15; // Minimum peak height.

        let mut peaks = 0;
        for i in 1..smoothed.len() - 1 {
            if smoothed[i] > smoothed[i - 1]
                && smoothed[i] > smoothed[i + 1]
                && smoothed[i] > threshold
            {
                peaks += 1;
            }
        }

        peaks
    }

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
}
