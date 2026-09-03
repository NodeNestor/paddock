//! Histogram-gap analysis, ported verbatim from the CPU reference.
//! CPU-only (per-block integer histograms; no GPU kernel), `gpu()` delegates.
//!
//! Level adjustment, contrast stretching, and recompression punch gaps into the
//! intensity histogram - runs of values that should exist but do not. Natural
//! images have smooth, gap-free histograms. Per-block analysis localizes the
//! manipulated region. Photo/sensor-oriented, so skipped for PDFs.

use crate::analyzer::Analyzer;
use crate::{Context, Finding, Region, Severity};

pub struct HistogramGapAnalyzer {
    block_size: usize,
    /// Minimum consecutive zero bins to count as a gap.
    min_gap_length: usize,
    /// Minimum total gaps per block to flag.
    min_gaps_per_block: usize,
}

impl Default for HistogramGapAnalyzer {
    fn default() -> Self {
        Self {
            block_size: 64,
            min_gap_length: 2,
            min_gaps_per_block: 3,
        }
    }
}

impl Analyzer for HistogramGapAnalyzer {
    fn name(&self) -> &'static str {
        // The reference's stable name is "histogram_gaps" (the file is
        // histogram_analysis.rs); keep the name so findings + gating line up.
        "histogram_gaps"
    }

    fn applies_to(&self, ctx: &Context) -> bool {
        !ctx.is_pdf()
    }

    fn cpu(&self, ctx: &Context) -> Vec<Finding> {
        let (width, height) = (ctx.width as usize, ctx.height as usize);
        if width < self.block_size * 3 || height < self.block_size * 3 {
            return Vec::new();
        }

        let gray = ctx.gray();
        let bs = self.block_size;
        let blocks_x = width / bs;
        let blocks_y = height / bs;

        // Global histogram for reference.
        let global_gaps =
            self.count_gaps(&self.compute_histogram(gray, 0, 0, width, height, width));

        // Per-block gap analysis: (block_idx, gap_count).
        let mut block_gaps: Vec<(usize, usize)> = Vec::new();
        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                let x0 = bx * bs;
                let y0 = by * bs;
                let hist = self.compute_histogram(gray, x0, y0, bs, bs, width);
                let gaps = self.count_gaps(&hist);
                if gaps >= self.min_gaps_per_block {
                    block_gaps.push((by * blocks_x + bx, gaps));
                }
            }
        }

        let mut findings = Vec::new();

        if global_gaps >= 5 {
            findings.push(Finding::new(
                "histogram_gaps",
                "histogram_global_gaps",
                format!(
                    "Global intensity histogram has {global_gaps} gaps (runs of missing \
                     values) - indicates level adjustment, contrast stretching, or \
                     value manipulation applied to the image"
                ),
                if global_gaps > 15 {
                    Severity::Medium
                } else {
                    Severity::Low
                },
                (0.40 + global_gaps as f64 * 0.02).min(0.70),
            ));
        }

        if !block_gaps.is_empty() {
            let gap_ratio = block_gaps.len() as f64 / (blocks_x * blocks_y) as f64;

            // Only interesting if gaps are localized (not everywhere).
            if gap_ratio > 0.05 && gap_ratio < 0.60 {
                let max_gaps = block_gaps.iter().map(|&(_, g)| g).max().unwrap_or(0);
                let mut min_x = usize::MAX;
                let mut min_y = usize::MAX;
                let mut max_x = 0_usize;
                let mut max_y = 0_usize;

                for &(idx, _) in &block_gaps {
                    let bx = idx % blocks_x;
                    let by = idx / blocks_x;
                    min_x = min_x.min(bx * bs);
                    min_y = min_y.min(by * bs);
                    max_x = max_x.max((bx + 1) * bs);
                    max_y = max_y.max((by + 1) * bs);
                }

                findings.push(
                    Finding::new(
                        "histogram_gaps",
                        "histogram_localized_gaps",
                        format!(
                            "{} of {} blocks ({:.1}%) show histogram gaps (max {} gaps/block) - \
                             localized level manipulation or spliced content with different \
                             value distribution",
                            block_gaps.len(),
                            blocks_x * blocks_y,
                            gap_ratio * 100.0,
                            max_gaps,
                        ),
                        Severity::Medium,
                        (0.45 + gap_ratio * 0.5).min(0.80),
                    )
                    .with_region(Region::BoundingBox {
                        x: min_x as u32,
                        y: min_y as u32,
                        width: (max_x - min_x) as u32,
                        height: (max_y - min_y) as u32,
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

impl HistogramGapAnalyzer {
    fn compute_histogram(
        &self,
        gray: &[u8],
        x0: usize,
        y0: usize,
        w: usize,
        h: usize,
        stride: usize,
    ) -> [u32; 256] {
        let mut hist = [0u32; 256];
        for dy in 0..h {
            for dx in 0..w {
                let x = x0 + dx;
                let y = y0 + dy;
                if x < stride && y * stride + x < gray.len() {
                    hist[gray[y * stride + x] as usize] += 1;
                }
            }
        }
        hist
    }

    fn count_gaps(&self, hist: &[u32; 256]) -> usize {
        let first_nonzero = hist.iter().position(|&v| v > 0).unwrap_or(0);
        let last_nonzero = hist.iter().rposition(|&v| v > 0).unwrap_or(255);

        if last_nonzero <= first_nonzero + 10 {
            return 0; // Too narrow a range to be meaningful.
        }

        let mut gap_count = 0;
        let mut run_length = 0;

        for &count in &hist[first_nonzero..=last_nonzero] {
            if count == 0 {
                run_length += 1;
            } else {
                if run_length >= self.min_gap_length {
                    gap_count += 1;
                }
                run_length = 0;
            }
        }
        if run_length >= self.min_gap_length {
            gap_count += 1;
        }

        gap_count
    }
}
