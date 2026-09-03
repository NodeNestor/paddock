//! PaddleOCR-VL image preprocessing - an exact port of the checkpoint's
//! `image_processing_paddleocr_vl.py` (the pipeline vLLM runs), not
//! llama.cpp's mtmd variant. Three steps, each with a sharp parity story:
//!
//! 1. `smart_resize`: both edges to multiples of 28 (patch 14 × merge 2),
//!    area clamped to [min_pixels, max_pixels]. Python `round()` is
//!    round-half-EVEN, so the port uses `round_ties_even` - 350px rounds to
//!    336 (12.5 -> 12) where a naive round would give 364.
//! 2. PIL bicubic resize on u8 (`resample: 3`) - `pillow::resize_rgb8`, the
//!    same int32 fixed-point + u8 mid-pass path the deepseek_ocr lane gates
//!    byte-exact.
//! 3. Normalize entirely in f32: `(px/255 - 0.5) / 0.5`. The oracle dump
//!    tested both f32 and f64 formula classes against the HF processor's own
//!    output and f32 matched bit-exactly (`normalize_formula: "f32_all"` in
//!    the manifest) - do not "improve" this to f64.
//!
//! The gate for all three is `gpu_paddleocr_vl_vision.rs` against the oracle
//! artifacts under `<models>/ocr-battery/paddle-oracle/`.

use crate::gpu_model::pillow::{self, Filter};

/// Patch side of the vision tower (SigLIP-so400m shape).
pub const PATCH: usize = 14;
/// Both resize targets are multiples of patch × spatial-merge.
pub const FACTOR: usize = PATCH * 2;

/// Pixel-area budget `smart_resize` clamps into. The checkpoint's
/// `preprocessor_config.json` says 112896 / 1003520 (28²·144 / 28²·1280);
/// Spotting is served with a LARGER per-request max (1605632) - the budget is
/// a parameter precisely so the serving surface can pass that through.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PixelBudget {
    pub min_pixels: usize,
    pub max_pixels: usize,
}

impl PixelBudget {
    /// The checkpoint's own defaults (also written into the GGUF mmproj as
    /// `clip.vision.image_{min,max}_pixels`).
    pub const DEFAULT: Self = Self {
        min_pixels: 112_896,
        max_pixels: 1_003_520,
    };
}

/// Exact port of the checkpoint's `smart_resize(height, width, factor,
/// min_pixels, max_pixels)`. Returns (height_bar, width_bar).
///
/// Order matters and is the reference's: the `< factor` guards mutate
/// height/width before the aspect check and the budget math, and the height
/// guard runs first (so a tiny-both-ways image is scaled off the updated
/// width). All float math is f64 like CPython's.
pub fn smart_resize(
    height: usize,
    width: usize,
    factor: usize,
    budget: PixelBudget,
) -> Result<(usize, usize), String> {
    let f = factor as f64;
    let (mut h, mut w) = (height as f64, width as f64);
    if h < f {
        w = ((w * f) / h).round_ties_even();
        h = f;
    }
    if w < f {
        h = ((h * f) / w).round_ties_even();
        w = f;
    }
    let (mn, mx) = (h.min(w), h.max(w));
    if mx / mn > 200.0 {
        return Err(format!(
            "absolute aspect ratio must be smaller than 200, got {}",
            mx / mn
        ));
    }
    let mut h_bar = (h / f).round_ties_even() * f;
    let mut w_bar = (w / f).round_ties_even() * f;
    if h_bar * w_bar > budget.max_pixels as f64 {
        let beta = ((h * w) / budget.max_pixels as f64).sqrt();
        h_bar = (h / beta / f).floor() * f;
        w_bar = (w / beta / f).floor() * f;
    } else if h_bar * w_bar < budget.min_pixels as f64 {
        let beta = (budget.min_pixels as f64 / (h * w)).sqrt();
        h_bar = (h * beta / f).ceil() * f;
        w_bar = (w * beta / f).ceil() * f;
    }
    if h_bar < f || w_bar < f {
        // the reference would emit a 0-patch grid here and fall over later;
        // fail loudly at the door instead
        return Err(format!(
            "degenerate resize target {w_bar}x{h_bar} for {width}x{height}"
        ));
    }
    Ok((h_bar as usize, w_bar as usize))
}

/// Full preprocessing for one interleaved-RGB u8 image: smart-resize, PIL
/// bicubic, f32 normalize into the PLANAR `[3][h][w]` layout
/// `vision::VisionModel::encode` consumes. Returns (planar, width, height).
pub fn preprocess_rgb(
    rgb: &[u8],
    w: usize,
    h: usize,
    budget: PixelBudget,
) -> Result<(Vec<f32>, usize, usize), String> {
    assert_eq!(rgb.len(), 3 * w * h, "expected tightly-packed RGB8");
    let (th, tw) = smart_resize(h, w, FACTOR, budget)?;
    let resized_owned;
    let resized: &[u8] = if (tw, th) == (w, h) {
        rgb
    } else {
        resized_owned = pillow::resize_rgb8(rgb, w, h, tw, th, Filter::Bicubic);
        &resized_owned
    };
    Ok((normalize_rgb(resized, tw, th), tw, th))
}

/// `(px/255 - 0.5) / 0.5`, f32 end to end (the oracle-verified formula),
/// interleaved HWC u8 -> planar CHW f32.
pub fn normalize_rgb(rgb: &[u8], w: usize, h: usize) -> Vec<f32> {
    assert_eq!(rgb.len(), 3 * w * h);
    let mut out = vec![0f32; 3 * w * h];
    for i in 0..w * h {
        for c in 0..3 {
            let v = rgb[i * 3 + c] as f32 / 255.0f32;
            out[c * w * h + i] = (v - 0.5f32) / 0.5f32;
        }
    }
    out
}

/// The deterministic probe pixels the oracle dump uses - regenerated here so
/// the gate carries no image fixtures: closed-form hash over the interleaved
/// byte index.
pub fn hash_pixels(w: usize, h: usize, seed: u32) -> Vec<u8> {
    (0..3 * w * h)
        .map(|i| ((i as u32).wrapping_mul(2_654_435_761).wrapping_add(seed) >> 24) as u8)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference table generated by the checkpoint's own `smart_resize`. The three
    /// square cases pin Python's round-half-even: 350/28 = 12.5 -> 12,
    /// 378/28 = 13.5 -> 14, 406/28 = 14.5 -> 14.
    #[test]
    fn smart_resize_matches_reference() {
        let b = PixelBudget::DEFAULT;
        for (h, w, want) in [
            (500, 700, (504, 700)),
            (900, 300, (896, 308)),
            (900, 1400, (784, 1232)), // over max: floor branch
            (80, 100, (308, 392)),    // under min: ceil branch
            (350, 350, (336, 336)),   // 12.5 tie -> even
            (406, 406, (392, 392)),   // 14.5 tie -> even
            (378, 378, (392, 392)),   // 13.5 tie -> even
            (20, 500, (84, 1680)),    // height < factor guard
            (500, 20, (1680, 84)),    // width < factor guard
            (1080, 1920, (728, 1316)),
            (2000, 2000, (980, 980)),
            (297, 420, (308, 420)),
            (28, 28, (336, 336)),
        ] {
            assert_eq!(smart_resize(h, w, FACTOR, b).unwrap(), want, "for {w}x{h}");
        }
    }

    #[test]
    fn extreme_aspect_is_refused() {
        assert!(smart_resize(30, 20_000, FACTOR, PixelBudget::DEFAULT).is_err());
    }

    /// The spotting budget must lift the area cap without touching the math.
    #[test]
    fn budget_is_a_parameter() {
        let spotting = PixelBudget {
            min_pixels: 112_896,
            max_pixels: 1_605_632,
        };
        let (h, w) = smart_resize(900, 1400, FACTOR, spotting).unwrap();
        assert_eq!(
            (h, w),
            (896, 1400),
            "in budget at the spotting cap: round only"
        );
        assert!(h * w <= spotting.max_pixels);
    }

    #[test]
    fn normalize_is_the_f32_formula() {
        let rgb = [0u8, 128, 255];
        let out = normalize_rgb(&rgb, 1, 1);
        assert_eq!(out[0], (0.0f32 / 255.0 - 0.5) / 0.5);
        assert_eq!(out[1], (128.0f32 / 255.0 - 0.5) / 0.5);
        assert_eq!(out[2], 1.0);
    }
}
