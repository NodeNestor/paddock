//! Error Level Analysis (ELA) - the canonical, multi-scale, complexity-normalized
//! algorithm, ported from the CPU reference.
//!
//! **One algorithm, two backends.** `cpu()` and (under `cuda`) `gpu()` compute
//! the *same* quantities; only *where* the per-pixel/per-block math runs differs.
//! The GPU path does the embarrassingly-parallel work in three deterministic
//! kernels (`cuda/ela.cu`) and shares the tiny cross-block reduction and the
//! finding-emission logic with the CPU path verbatim - so `tests/parity.rs` can
//! prove they agree.
//!
//! This deliberately does not reproduce the reference's *GPU* ELA, which was a
//! cruder, single-quality, normalized-space variant whose findings never matched
//! its own CPU path. Two documented, intentional refinements over the reference
//! CPU code, applied identically to both paths so parity holds:
//!   1. luminance variance ("complexity") is clamped at 0 before `sqrt` (a
//!      variance cannot be negative; the unclamped `sqrt` could yield NaN);
//!   2. the global error variance uses the one-pass `E[x^2] - E[x]^2` form so
//!      the GPU reduction and the CPU reduction compute the identical statistic.

use std::io::Cursor;

use image::codecs::jpeg::JpegEncoder;
use image::{DynamicImage, GenericImageView, ImageFormat, ImageReader};

use crate::analyzer::Analyzer;
use crate::{Context, Finding, Severity};

/// Downsample cap: ELA artifact patterns are coarse and survive 2× downsampling,
/// so we cap the long edge to bound JPEG encode/decode and per-pixel cost.
const MAX_LONG_EDGE: u32 = 800;

/// Standard JPEG luminance quantization table (Annex K) - the reference for
/// estimating a JPEG's quality from its own quantization table.
const STANDARD_LUMA: [u16; 64] = [
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113,
    92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
];

/// Error Level Analysis analyzer.
pub struct ErrorLevelAnalyzer {
    /// Fallback re-save quality when JPEG quality estimation fails.
    resave_quality: u8,
    /// Minimum fraction of outlier/hotspot blocks to trigger a finding.
    hotspot_ratio_threshold: f64,
    /// Block size for local analysis (pixels).
    block_size: usize,
    /// Standard deviations for statistical outlier detection.
    outlier_sigma: f64,
}

impl Default for ErrorLevelAnalyzer {
    fn default() -> Self {
        Self {
            resave_quality: 92,
            hotspot_ratio_threshold: 0.05,
            block_size: 16,
            outlier_sigma: 2.5,
        }
    }
}

/// Shared per-run setup: the (possibly downsampled) work image, its dimensions,
/// the estimated JPEG quality, and the multi-scale quality levels.
struct ElaPrep {
    work: DynamicImage,
    width: usize,
    height: usize,
    estimated_quality: u8,
    qualities: Vec<u8>,
}

/// One block's statistics.
#[derive(Clone, Copy)]
struct BlockStat {
    mean_error: f64,
    /// Luminance variance over the block (already clamped at 0).
    complexity: f64,
}

impl BlockStat {
    /// ELA error normalized by local complexity - high-texture regions naturally
    /// carry higher ELA residuals and must not be read as manipulation.
    fn normalized(&self) -> f64 {
        if self.complexity > 1.0 {
            self.mean_error / self.complexity.sqrt()
        } else {
            self.mean_error
        }
    }
}

/// The reduced quantities both backends feed into [`ErrorLevelAnalyzer::emit_findings`].
struct Reduced {
    blocks: Vec<BlockStat>,
    /// Total adaptive-threshold hotspot pixel count across all blocks.
    adaptive_hotspots: u64,
    /// Sum and sum-of-squares of the combined error map over all pixels.
    global_sum: f64,
    global_sum_sq: f64,
    total_pixels: usize,
}

impl Analyzer for ErrorLevelAnalyzer {
    fn name(&self) -> &'static str {
        "ela"
    }

    /// JPEG-only, matching the reference's `should_skip` (`ela => !is_jpeg`). ELA
    /// re-saves at a fixed quality and diffs; on a non-JPEG source (PNG/WebP/...)
    /// every block shows uniform re-compression error rather than localized
    /// tampering, so the reference suppresses it - paddock matches to avoid the same
    /// false positives.
    fn applies_to(&self, ctx: &Context) -> bool {
        ctx.is_jpeg()
    }

    fn cpu(&self, ctx: &Context) -> Vec<Finding> {
        let prep = self.prepare(ctx);
        let work_rgb = prep.work.to_rgb8();

        if self.too_small(&prep) {
            return self.small_image_findings(&prep);
        }
        let resaved = match self.resaved_planes(&prep) {
            Some(r) => r,
            None => return Vec::new(),
        };
        let reduced = self.reduce_cpu(work_rgb.as_raw(), &resaved, prep.width, prep.height);
        self.emit_findings(&reduced, prep.estimated_quality)
    }

    #[cfg(feature = "cuda")]
    fn gpu(
        &self,
        gpu: &crate::gpu::ForensicGpu,
        ctx: &Context,
    ) -> Result<Vec<Finding>, crate::gpu::GpuError> {
        let prep = self.prepare(ctx);
        let work_rgb = prep.work.to_rgb8();

        if self.too_small(&prep) {
            // Not worth the GPU; the small-image path is a cheap single-quality
            // global ELA identical on both backends.
            return Ok(self.small_image_findings(&prep));
        }
        let resaved = match self.resaved_planes(&prep) {
            Some(r) => r,
            None => return Ok(Vec::new()),
        };
        let reduced = self.reduce_gpu(gpu, work_rgb.as_raw(), &resaved, prep.width, prep.height)?;
        Ok(self.emit_findings(&reduced, prep.estimated_quality))
    }
}

impl ErrorLevelAnalyzer {
    // ---- shared setup -----------------------------------------------------

    fn prepare(&self, ctx: &Context) -> ElaPrep {
        let estimated_quality =
            Self::estimate_jpeg_quality(&ctx.raw_bytes).unwrap_or(self.resave_quality);

        let (orig_w, orig_h) = (ctx.width, ctx.height);
        let long_edge = orig_w.max(orig_h);
        let work = if long_edge > MAX_LONG_EDGE {
            ctx.image.resize(
                MAX_LONG_EDGE * orig_w / long_edge,
                MAX_LONG_EDGE * orig_h / long_edge,
                image::imageops::FilterType::Triangle,
            )
        } else {
            ctx.image.clone()
        };

        let width = work.width() as usize;
        let height = work.height() as usize;
        // Two scales (Q-5, Q): cross-quality averaging without a third encode.
        let qualities = vec![
            estimated_quality.saturating_sub(5).max(1),
            estimated_quality,
        ];

        ElaPrep {
            work,
            width,
            height,
            estimated_quality,
            qualities,
        }
    }

    fn too_small(&self, prep: &ElaPrep) -> bool {
        prep.width < self.block_size * 3 || prep.height < self.block_size * 3
    }

    /// Re-save the work image at each quality and return the decoded interleaved
    /// RGB planes (one per quality). `None` if any encode/decode fails or a
    /// decoded plane's dimensions do not match the work image.
    fn resaved_planes(&self, prep: &ElaPrep) -> Option<Vec<Vec<u8>>> {
        let (w, h) = (prep.width as u32, prep.height as u32);
        let mut planes = Vec::with_capacity(prep.qualities.len());
        for &q in &prep.qualities {
            let mut buf = Cursor::new(Vec::new());
            let encoder = JpegEncoder::new_with_quality(&mut buf, q);
            prep.work.write_with_encoder(encoder).ok()?;
            buf.set_position(0);
            let decoded = ImageReader::with_format(buf, ImageFormat::Jpeg)
                .decode()
                .ok()?;
            if decoded.dimensions() != (w, h) {
                return None;
            }
            planes.push(decoded.to_rgb8().into_raw());
        }
        Some(planes)
    }

    // ---- CPU reduction ----------------------------------------------------

    fn reduce_cpu(
        &self,
        orig_rgb: &[u8],
        resaved: &[Vec<u8>],
        width: usize,
        height: usize,
    ) -> Reduced {
        let n = width * height;
        let k = resaved.len().max(1);

        // Combined, multi-scale per-pixel error map (in [0,255] space).
        let mut map = vec![0.0_f64; n];
        for (i, m) in map.iter_mut().enumerate() {
            let oi = i * 3;
            let mut acc = 0.0_f64;
            for plane in resaved {
                let dr = orig_rgb[oi] as i32 - plane[oi] as i32;
                let dg = orig_rgb[oi + 1] as i32 - plane[oi + 1] as i32;
                let db = orig_rgb[oi + 2] as i32 - plane[oi + 2] as i32;
                let s = dr.unsigned_abs() + dg.unsigned_abs() + db.unsigned_abs();
                acc += s as f64 / 3.0;
            }
            *m = acc / k as f64;
        }

        let bs = self.block_size;
        let blocks_x = width / bs;
        let blocks_y = height / bs;

        let mut blocks = Vec::with_capacity(blocks_x * blocks_y);
        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                let x0 = bx * bs;
                let y0 = by * bs;

                let mut sum_error = 0.0_f64;
                for dy in 0..bs {
                    for dx in 0..bs {
                        sum_error += map[(y0 + dy) * width + (x0 + dx)];
                    }
                }
                let mean_error = sum_error / (bs * bs) as f64;
                let complexity = Self::block_complexity(orig_rgb, width, x0, y0, bs);
                blocks.push(BlockStat {
                    mean_error,
                    complexity,
                });
            }
        }

        // Adaptive hotspots: per-block local threshold from this block's mean +
        // 2·sqrt(complexity), counting pixels above it and above 15.
        let mut adaptive_hotspots = 0_u64;
        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                let block = &blocks[by * blocks_x + bx];
                let local_threshold = block.mean_error + 2.0 * block.complexity.sqrt();
                let x0 = bx * bs;
                let y0 = by * bs;
                for dy in 0..bs {
                    for dx in 0..bs {
                        let e = map[(y0 + dy) * width + (x0 + dx)];
                        if e > local_threshold && e > 15.0 {
                            adaptive_hotspots += 1;
                        }
                    }
                }
            }
        }

        let global_sum: f64 = map.iter().sum();
        let global_sum_sq: f64 = map.iter().map(|&e| e * e).sum();

        Reduced {
            blocks,
            adaptive_hotspots,
            global_sum,
            global_sum_sq,
            total_pixels: n,
        }
    }

    /// Luminance variance over a block (BT.601), clamped at 0.
    fn block_complexity(rgb: &[u8], width: usize, x0: usize, y0: usize, bs: usize) -> f64 {
        let mut sum = 0.0_f64;
        let mut sum_sq = 0.0_f64;
        for dy in 0..bs {
            for dx in 0..bs {
                let idx = ((y0 + dy) * width + (x0 + dx)) * 3;
                let lum = 0.299 * rgb[idx] as f64
                    + 0.587 * rgb[idx + 1] as f64
                    + 0.114 * rgb[idx + 2] as f64;
                sum += lum;
                sum_sq += lum * lum;
            }
        }
        let count = (bs * bs) as f64;
        let mean = sum / count;
        (sum_sq / count - mean * mean).max(0.0)
    }

    // ---- GPU reduction ----------------------------------------------------

    #[cfg(feature = "cuda")]
    fn reduce_gpu(
        &self,
        gpu: &crate::gpu::ForensicGpu,
        orig_rgb: &[u8],
        resaved: &[Vec<u8>],
        width: usize,
        height: usize,
    ) -> Result<Reduced, crate::gpu::GpuError> {
        use cudarc::driver::{LaunchConfig, PushKernelArg};

        let n = width * height;
        let bs = self.block_size;
        let blocks_x = width / bs;
        let blocks_y = height / bs;
        let nblocks = blocks_x * blocks_y;
        let k = resaved.len().max(1);
        let stream = gpu.stream();

        // Uploads: original work image + the k resaved planes, concatenated.
        let d_orig = stream.clone_htod(orig_rgb)?;
        let mut resaved_cat = Vec::with_capacity(k * n * 3);
        for plane in resaved {
            resaved_cat.extend_from_slice(plane);
        }
        let d_resaved = stream.clone_htod(&resaved_cat)?;
        let mut d_map = stream.alloc_zeros::<f32>(n)?;

        // Kernel 1 - multi-scale error map.
        let n_u = n as u32;
        let k_u = k as u32;
        let f_map = gpu.function("ela", "fela_error_map")?;
        unsafe {
            stream
                .launch_builder(&f_map)
                .arg(&d_orig)
                .arg(&d_resaved)
                .arg(&mut d_map)
                .arg(&n_u)
                .arg(&k_u)
                .launch(LaunchConfig::for_num_elems(n_u))?;
        }

        // Kernel 2 - per-block mean/complexity/hotspots (one CTA per block).
        let mut d_blocks = stream.alloc_zeros::<ElaBlockStatsRaw>(nblocks)?;
        let (w_u, h_u, bs_u, bx_u) = (width as u32, height as u32, bs as u32, blocks_x as u32);
        let f_blk = gpu.function("ela", "fela_block_stats")?;
        let cfg_blocks = LaunchConfig {
            grid_dim: (blocks_x as u32, blocks_y as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(&f_blk)
                .arg(&d_map)
                .arg(&d_orig)
                .arg(&w_u)
                .arg(&h_u)
                .arg(&bs_u)
                .arg(&bx_u)
                .arg(&mut d_blocks)
                .launch(cfg_blocks)?;
        }

        // Kernel 3 - global sum / sum-of-squares (single CTA -> deterministic).
        let mut d_global = stream.alloc_zeros::<f32>(2)?;
        let f_glob = gpu.function("ela", "fela_global")?;
        let cfg_global = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(&f_glob)
                .arg(&d_map)
                .arg(&n_u)
                .arg(&mut d_global)
                .launch(cfg_global)?;
        }

        let blocks_raw: Vec<ElaBlockStatsRaw> = stream.clone_dtoh(&d_blocks)?;
        let global: Vec<f32> = stream.clone_dtoh(&d_global)?;
        stream.synchronize()?;

        let mut blocks = Vec::with_capacity(nblocks);
        let mut adaptive_hotspots = 0_u64;
        for b in &blocks_raw {
            blocks.push(BlockStat {
                mean_error: b.mean_error as f64,
                complexity: b.complexity as f64,
            });
            adaptive_hotspots += b.hotspot_count as u64;
        }

        Ok(Reduced {
            blocks,
            adaptive_hotspots,
            global_sum: global[0] as f64,
            global_sum_sq: global[1] as f64,
            total_pixels: n,
        })
    }

    // ---- shared finding emission -----------------------------------------

    fn emit_findings(&self, r: &Reduced, estimated_quality: u8) -> Vec<Finding> {
        let mut findings = Vec::new();
        if r.blocks.is_empty() || r.total_pixels == 0 {
            return findings;
        }
        let total_pixels = r.total_pixels as f64;
        let nblocks = r.blocks.len() as f64;

        // Cross-block statistical outlier detection on complexity-normalized error.
        let norm_mean: f64 = r.blocks.iter().map(BlockStat::normalized).sum::<f64>() / nblocks;
        let norm_var: f64 = r
            .blocks
            .iter()
            .map(|b| (b.normalized() - norm_mean).powi(2))
            .sum::<f64>()
            / nblocks;
        let norm_std = norm_var.sqrt();
        let threshold_high = norm_mean + self.outlier_sigma * norm_std;
        let threshold_low = (norm_mean - self.outlier_sigma * norm_std).max(0.0);

        let outlier_high = r
            .blocks
            .iter()
            .filter(|b| b.normalized() > threshold_high)
            .count();
        let outlier_low = r
            .blocks
            .iter()
            .filter(|b| b.normalized() < threshold_low && b.mean_error > 0.5)
            .count();
        let outlier_ratio = (outlier_high + outlier_low) as f64 / nblocks;

        // Global statistics (one-pass variance so GPU and CPU agree exactly).
        let global_mean = r.global_sum / total_pixels;
        let global_std = (r.global_sum_sq / total_pixels - global_mean * global_mean)
            .max(0.0)
            .sqrt();

        let adaptive_hotspot_ratio = r.adaptive_hotspots as f64 / total_pixels;

        if global_mean > 20.0 {
            findings.push(Finding::new(
                "ela",
                "ela_high_mean_error",
                format!(
                    "Multi-scale ELA mean error level {global_mean:.1} (estimated Q={estimated_quality}) exceeds \
                     threshold - possible re-save or composite"
                ),
                Severity::Low,
                0.5,
            ));
        }

        if global_std > 25.0 {
            findings.push(Finding::new(
                "ela",
                "ela_uneven_error",
                format!(
                    "Uneven ELA distribution (std dev {global_std:.1}) across multi-scale analysis \
                     suggests regions saved at different quality levels"
                ),
                Severity::High,
                0.75,
            ));
        }

        if adaptive_hotspot_ratio > self.hotspot_ratio_threshold {
            findings.push(Finding::new(
                "ela",
                "ela_hotspot_regions",
                format!(
                    "{:.1}% of pixels show elevated error levels (adaptive threshold) - \
                     localized manipulation suspected",
                    adaptive_hotspot_ratio * 100.0
                ),
                Severity::High,
                0.7,
            ));
        }

        if outlier_ratio > self.hotspot_ratio_threshold {
            findings.push(Finding::new(
                "ela",
                "ela_block_outliers",
                format!(
                    "{:.1}% of blocks are statistical ELA outliers ({} high, {} low out \
                     of {} blocks) - complexity-normalized analysis indicates tampering",
                    outlier_ratio * 100.0,
                    outlier_high,
                    outlier_low,
                    r.blocks.len()
                ),
                if outlier_ratio > 0.15 {
                    Severity::High
                } else {
                    Severity::Medium
                },
                (0.55 + outlier_ratio * 2.0).min(0.85),
            ));
        }

        findings
    }

    // ---- small-image (global, single-quality) fallback --------------------

    /// For images too small for block analysis: a single-quality global ELA.
    /// Identical on both backends (it is cheap and not worth a kernel launch).
    fn small_image_findings(&self, prep: &ElaPrep) -> Vec<Finding> {
        let map = match self.single_quality_map(&prep.work, prep.estimated_quality) {
            Some(m) => m,
            None => return Vec::new(),
        };
        let total = map.len() as f64;
        if total == 0.0 {
            return Vec::new();
        }
        let mean_error: f64 = map.iter().sum::<f64>() / total;
        let sum_sq: f64 = map.iter().map(|&e| e * e).sum();
        let std_dev = (sum_sq / total - mean_error * mean_error).max(0.0).sqrt();
        let hotspot_ratio = map.iter().filter(|&&e| e > 30.0).count() as f64 / total;

        let mut findings = Vec::new();
        if mean_error > 20.0 {
            findings.push(Finding::new(
                "ela",
                "ela_high_mean_error",
                format!(
                    "ELA mean error level {mean_error:.1} exceeds threshold - possible re-save or composite"
                ),
                Severity::Low,
                0.5,
            ));
        }
        if std_dev > 25.0 {
            findings.push(Finding::new(
                "ela",
                "ela_uneven_error",
                format!(
                    "Uneven ELA distribution (std dev {std_dev:.1}) suggests regions saved at \
                     different quality levels"
                ),
                Severity::High,
                0.75,
            ));
        }
        if hotspot_ratio > self.hotspot_ratio_threshold {
            findings.push(Finding::new(
                "ela",
                "ela_hotspot_regions",
                format!(
                    "{:.1}% of pixels show elevated error levels - localized manipulation suspected",
                    hotspot_ratio * 100.0
                ),
                Severity::High,
                0.7,
            ));
        }
        findings
    }

    /// Single-quality per-pixel error map in [0,255] space.
    fn single_quality_map(&self, work: &DynamicImage, quality: u8) -> Option<Vec<f64>> {
        let orig = work.to_rgb8();
        let (w, h) = orig.dimensions();
        let orig_raw = orig.as_raw();

        let mut buf = Cursor::new(Vec::new());
        let encoder = JpegEncoder::new_with_quality(&mut buf, quality);
        work.write_with_encoder(encoder).ok()?;
        buf.set_position(0);
        let resaved = ImageReader::with_format(buf, ImageFormat::Jpeg)
            .decode()
            .ok()?;
        if resaved.dimensions() != (w, h) {
            return None;
        }
        let resaved_raw = resaved.to_rgb8().into_raw();

        let total = (w * h) as usize;
        let mut map = Vec::with_capacity(total);
        let mut idx = 0usize;
        for _ in 0..total {
            let dr = orig_raw[idx] as i32 - resaved_raw[idx] as i32;
            let dg = orig_raw[idx + 1] as i32 - resaved_raw[idx + 1] as i32;
            let db = orig_raw[idx + 2] as i32 - resaved_raw[idx + 2] as i32;
            let e = (dr.unsigned_abs() + dg.unsigned_abs() + db.unsigned_abs()) as f64 / 3.0;
            map.push(e);
            idx += 3;
        }
        Some(map)
    }

    // ---- JPEG quality estimation ------------------------------------------

    /// Estimate the JPEG quality factor from the first quantization table.
    fn estimate_jpeg_quality(raw_bytes: &[u8]) -> Option<u8> {
        if raw_bytes.len() < 2 || raw_bytes[0] != 0xFF || raw_bytes[1] != 0xD8 {
            return None;
        }
        let qtables = Self::extract_quantization_tables(raw_bytes);
        let qtable = qtables.first()?;

        let mut ratios = Vec::with_capacity(64);
        for i in 0..64 {
            if STANDARD_LUMA[i] > 0 && qtable[i] > 0 {
                ratios.push(qtable[i] as f64 / STANDARD_LUMA[i] as f64);
            }
        }
        if ratios.is_empty() {
            return None;
        }
        ratios.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median_ratio = ratios[ratios.len() / 2];

        let quality = if median_ratio < 1.0 {
            (50.0 / median_ratio).round().min(100.0) as u8
        } else {
            (50.0 * (2.0 - median_ratio)).round().max(1.0) as u8
        };
        Some(quality.clamp(1, 100))
    }

    /// Parse JPEG markers and extract DQT (Define Quantization Table) contents.
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

/// GPU per-block statistics - repr must match `FElaBlockStats` in `cuda/ela.cu`.
#[cfg(feature = "cuda")]
#[repr(C)]
#[derive(Clone, Copy)]
struct ElaBlockStatsRaw {
    mean_error: f32,
    complexity: f32,
    hotspot_count: u32,
    _pad: u32,
}

#[cfg(feature = "cuda")]
unsafe impl cudarc::driver::DeviceRepr for ElaBlockStatsRaw {}
#[cfg(feature = "cuda")]
unsafe impl cudarc::driver::ValidAsZeroBits for ElaBlockStatsRaw {}
