//! JPEG Ghost splice detection, ported from the CPU reference. Sweeps
//! re-save qualities; a block spliced from a differently-compressed source
//! shows minimum re-save error at a different quality than the rest of the
//! image, revealing the splice.
//!
//! Canonical algorithm = the CPU sweep. The reference's GPU path was a stub (called
//! CPU); paddock's GPU path computes the per-block, per-quality error via a
//! kernel whose exact integer sum-of-squares (fits u32 for a 64×64 block)
//! matches the CPU bit-for-bit - so the best-quality/histogram reduction, run
//! host-side, is identical. Exact GPU==CPU parity.
//!
//! One deliberate refinement over the reference CPU code, applied to both paths:
//! the dominant quality is chosen with a deterministic tie-break (highest count,
//! then lowest quality) instead of `HashMap::max_by_key`, whose tie order is
//! non-deterministic across runs.

use std::io::Cursor;

use image::codecs::jpeg::JpegEncoder;
use image::{DynamicImage, GenericImageView, ImageFormat, ImageReader};

use crate::analyzer::Analyzer;
use crate::{Context, Finding, Severity};

/// Cap on the working-image pixel count for the sweep (600×450). JPEG ghost is a
/// block-level technique; a reduced-resolution sweep cuts encode+MSE cost with
/// no meaningful loss.
const MAX_SWEEP_PIXELS: u32 = 270_000;

pub struct JpegGhostDetector {
    quality_start: u8,
    quality_end: u8,
    quality_step: u8,
    block_size: u32,
}

impl Default for JpegGhostDetector {
    fn default() -> Self {
        Self {
            quality_start: 60,
            quality_end: 98,
            quality_step: 6,
            block_size: 64,
        }
    }
}

/// Shared per-run setup: the (possibly downsampled) sweep image + its dims + the
/// quality ladder + block grid.
struct GhostPrep {
    sweep: DynamicImage,
    width: u32,
    /// Read by the GPU lane only.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    height: u32,
    qualities: Vec<u8>,
    blocks_x: u32,
    blocks_y: u32,
}

impl Analyzer for JpegGhostDetector {
    fn name(&self) -> &'static str {
        "jpeg_ghost"
    }

    /// JPEG-only (matches the reference's `should_skip`): the technique keys on JPEG
    /// compression history.
    fn applies_to(&self, ctx: &Context) -> bool {
        ctx.is_jpeg()
    }

    fn cpu(&self, ctx: &Context) -> Vec<Finding> {
        let prep = match self.prepare(ctx) {
            Some(p) => p,
            None => return Vec::new(),
        };
        let sweep_rgb = prep.sweep.to_rgb8();
        let sweep_raw = sweep_rgb.as_raw();
        let num_blocks = (prep.blocks_x * prep.blocks_y) as usize;

        let mut block_mse: Vec<Vec<f64>> = Vec::with_capacity(prep.qualities.len());
        for &q in &prep.qualities {
            let resaved = match resave_rgb(&prep.sweep, q) {
                Some(r) => r,
                None => return Vec::new(),
            };
            let mut errs = Vec::with_capacity(num_blocks);
            for by in 0..prep.blocks_y {
                for bx in 0..prep.blocks_x {
                    errs.push(block_mse_cpu(
                        sweep_raw,
                        &resaved,
                        prep.width,
                        bx * self.block_size,
                        by * self.block_size,
                        self.block_size,
                    ));
                }
            }
            block_mse.push(errs);
        }
        self.reduce_findings(&block_mse, &prep.qualities, num_blocks)
    }

    #[cfg(feature = "cuda")]
    fn gpu(
        &self,
        gpu: &crate::gpu::ForensicGpu,
        ctx: &Context,
    ) -> Result<Vec<Finding>, crate::gpu::GpuError> {
        use cudarc::driver::{LaunchConfig, PushKernelArg};

        let prep = match self.prepare(ctx) {
            Some(p) => p,
            None => return Ok(Vec::new()),
        };
        let sweep_rgb = prep.sweep.to_rgb8();
        let sweep_raw = sweep_rgb.as_raw();
        let n = (prep.width * prep.height) as usize;
        let k = prep.qualities.len();
        let num_blocks = (prep.blocks_x * prep.blocks_y) as usize;
        let stream = gpu.stream();

        // Re-save each quality on the host, concatenate for one upload.
        let mut resaved_cat = Vec::with_capacity(k * n * 3);
        for &q in &prep.qualities {
            match resave_rgb(&prep.sweep, q) {
                Some(r) => resaved_cat.extend_from_slice(&r),
                None => return Ok(Vec::new()),
            }
        }
        let d_sweep = stream.clone_htod(sweep_raw)?;
        let d_resaved = stream.clone_htod(&resaved_cat)?;
        let mut d_out = stream.alloc_zeros::<u32>(k * num_blocks)?;

        let (w_u, h_u, bs_u, bx_u) = (prep.width, prep.height, self.block_size, prep.blocks_x);
        let f = gpu.function("jpeg_ghost", "jghost_block_sse")?;
        let cfg = LaunchConfig {
            grid_dim: (prep.blocks_x, prep.blocks_y, k as u32),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(&f)
                .arg(&d_sweep)
                .arg(&d_resaved)
                .arg(&mut d_out)
                .arg(&w_u)
                .arg(&h_u)
                .arg(&bs_u)
                .arg(&bx_u)
                .launch(cfg)?;
        }
        let sums: Vec<u32> = stream.clone_dtoh(&d_out)?;
        stream.synchronize()?;

        // sums[q*nblocks + block] -> MSE, identical divisor to the CPU path.
        let denom = (self.block_size * self.block_size * 3) as f64;
        let mut block_mse: Vec<Vec<f64>> = Vec::with_capacity(k);
        for q in 0..k {
            let mut errs = Vec::with_capacity(num_blocks);
            for b in 0..num_blocks {
                errs.push(sums[q * num_blocks + b] as f64 / denom);
            }
            block_mse.push(errs);
        }
        Ok(self.reduce_findings(&block_mse, &prep.qualities, num_blocks))
    }
}

impl JpegGhostDetector {
    fn prepare(&self, ctx: &Context) -> Option<GhostPrep> {
        let (ow, oh) = (ctx.width, ctx.height);
        if ow < self.block_size * 2 || oh < self.block_size * 2 {
            return None;
        }
        let sweep = {
            let pixels = ow * oh;
            if pixels > MAX_SWEEP_PIXELS {
                let scale = (MAX_SWEEP_PIXELS as f64 / pixels as f64).sqrt();
                let sw = ((ow as f64 * scale) as u32).max(self.block_size * 2);
                let sh = ((oh as f64 * scale) as u32).max(self.block_size * 2);
                ctx.image
                    .resize(sw, sh, image::imageops::FilterType::Triangle)
            } else {
                ctx.image.clone()
            }
        };
        let (width, height) = (sweep.width(), sweep.height());
        let qualities: Vec<u8> = (self.quality_start..=self.quality_end)
            .step_by(self.quality_step as usize)
            .collect();
        let blocks_x = width / self.block_size;
        let blocks_y = height / self.block_size;
        if blocks_x == 0 || blocks_y == 0 || qualities.is_empty() {
            return None;
        }
        Some(GhostPrep {
            sweep,
            width,
            height,
            qualities,
            blocks_x,
            blocks_y,
        })
    }

    /// Shared reduction: best quality per block -> dominant -> ghost blocks ->
    /// findings. Identical on both paths (they only differ in how block MSE was
    /// computed, which is exact-equal).
    fn reduce_findings(
        &self,
        block_mse: &[Vec<f64>],
        qualities: &[u8],
        num_blocks: usize,
    ) -> Vec<Finding> {
        // best quality per block (min MSE across the ladder)
        let best_q: Vec<u8> = (0..num_blocks)
            .map(|b| {
                let mut bq = qualities[0];
                let mut be = f64::MAX;
                for (qi, &q) in qualities.iter().enumerate() {
                    let e = block_mse[qi][b];
                    if e < be {
                        be = e;
                        bq = q;
                    }
                }
                bq
            })
            .collect();

        // Histogram over the fixed quality ladder -> dominant with a
        // DETERMINISTIC tie-break (highest count, then lowest quality).
        let mut counts = vec![0u32; qualities.len()];
        for &q in &best_q {
            if let Some(i) = qualities.iter().position(|&x| x == q) {
                counts[i] += 1;
            }
        }
        let dominant = counts
            .iter()
            .enumerate()
            .max_by(|(ia, a), (ib, b)| a.cmp(b).then(ib.cmp(ia)))
            .map(|(i, _)| qualities[i])
            .unwrap_or(85);

        let mut ghost_blocks = 0u32;
        let mut diffs: Vec<i16> = Vec::new();
        for &q in &best_q {
            let d = q as i16 - dominant as i16;
            if d.unsigned_abs() > 6 {
                ghost_blocks += 1;
                diffs.push(d);
            }
        }

        let mut findings = Vec::new();
        let ghost_ratio = ghost_blocks as f64 / num_blocks as f64;
        if ghost_ratio > 0.05 && ghost_ratio < 0.7 {
            findings.push(Finding::new(
                "jpeg_ghost",
                "jpeg_ghost_detected",
                format!(
                    "{:.1}% of image blocks show different JPEG compression history (dominant quality \
                     ≈{dominant}, ghost regions deviate by {:+} to {:+}) - indicates splicing from \
                     differently-compressed source",
                    ghost_ratio * 100.0,
                    diffs.iter().min().unwrap_or(&0),
                    diffs.iter().max().unwrap_or(&0),
                ),
                Severity::Critical,
                0.80,
            ));
        }
        findings.push(Finding::new(
            "jpeg_ghost",
            "jpeg_quality_estimate",
            format!("Estimated original JPEG quality factor: ~{dominant}"),
            Severity::Info,
            0.85,
        ));
        findings
    }
}

/// Re-save an image at `quality` and return the decoded interleaved RGB.
fn resave_rgb(img: &DynamicImage, quality: u8) -> Option<Vec<u8>> {
    let mut buf = Cursor::new(Vec::new());
    let encoder = JpegEncoder::new_with_quality(&mut buf, quality);
    img.write_with_encoder(encoder).ok()?;
    buf.set_position(0);
    let decoded = ImageReader::with_format(buf, ImageFormat::Jpeg)
        .decode()
        .ok()?;
    if decoded.dimensions() != img.dimensions() {
        return None;
    }
    Some(decoded.to_rgb8().into_raw())
}

/// Mean squared error over a block (integer sum / count) - exact-equal to the
/// GPU kernel's u32 sum divided by the same count.
fn block_mse_cpu(a: &[u8], b: &[u8], img_width: u32, x0: u32, y0: u32, size: u32) -> f64 {
    let stride = (img_width * 3) as usize;
    let mut sum = 0u64;
    for dy in 0..size as usize {
        let row = (y0 as usize + dy) * stride + x0 as usize * 3;
        for i in 0..size as usize * 3 {
            let diff = a[row + i] as i32 - b[row + i] as i32;
            sum += (diff * diff) as u64;
        }
    }
    sum as f64 / (size * size * 3) as f64
}
