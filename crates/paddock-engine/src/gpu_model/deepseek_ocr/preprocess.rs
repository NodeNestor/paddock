//! PIL-exact image preprocessing for the DeepSeek-OCR views.
//!
//! The reference (`modeling_unlimitedocr.py`) builds every view with PIL on
//! uint8 pixels:
//!
//!   crops:  `image.resize((cols*640, rows*640))`   - PIL default = BICUBIC
//!           then 640×640 crops in row-major tile order
//!   global: `ImageOps.pad(image, (1024, 1024), color=(127,127,127))`
//!           - aspect-preserving contain + center pad with the mean color
//!   then ToTensor (x/255) and Normalize(mean=std=0.5) per channel.
//!
//! PIL's resize is not float bicubic: it is a two-pass (horizontal, then
//! vertical) separable convolution in **int32 fixed point with a uint8
//! intermediate between the passes**, coefficients normalized per output
//! pixel and quantized to 22 fractional bits, support scaled by the downscale
//! factor (PIL is always antialiased). The cubic is A = -0.5. Reimplementing
//! that with float math produces off-by-one bytes everywhere, a different
//! numeric class from what the reference tower was trained and gated on - so
//! this file replicates `libImaging/Resample.c` exactly and is gated
//! byte-for-byte against PIL's own output (`preprocess_fixtures.rs`,
//! generated out of tree).
//!
//! Host-side by design: this is admission work on request bytes, the same
//! class as granite/gemma4 preprocessing and the ASR mel frontend.

/// Fixed-point fraction bits, Resample.c's `PRECISION_BITS (32 - 8 - 2)`.
const PRECISION_BITS: u32 = 22;

/// The bicubic kernel, A = -0.5, support 2.0 - PIL's `bicubic_filter`.
fn bicubic(x: f64) -> f64 {
    const A: f64 = -0.5;
    let x = x.abs();
    if x < 1.0 {
        ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        ((((x - 5.0) * x) + 8.0) * x - 4.0) * A
    } else {
        0.0
    }
}

/// One axis's resampling plan: per output coordinate, the first source index,
/// the tap count, and `taps`-strided quantized coefficients.
struct Coeffs {
    taps: usize,
    bounds: Vec<(usize, usize)>,
    kk: Vec<i32>,
}

/// `precompute_coeffs` + `normalize_coeffs_8bpc`, bit-faithful: C's `(int)`
/// casts truncate toward zero (mirrored with `as`), the window is normalized
/// in f64 before quantization, and quantization rounds half away from zero.
fn coeffs(in_size: usize, out_size: usize) -> Coeffs {
    const SUPPORT: f64 = 2.0;
    let scale = in_size as f64 / out_size as f64;
    let filterscale = scale.max(1.0); // downscale widens the kernel = antialias
    let support = SUPPORT * filterscale;
    let taps = support.ceil() as usize * 2 + 1;
    let mut bounds = Vec::with_capacity(out_size);
    let mut kk = vec![0i32; out_size * taps];
    let mut w = vec![0f64; taps];
    for xx in 0..out_size {
        let center = (xx as f64 + 0.5) * scale;
        let ss = 1.0 / filterscale;
        let xmin = ((center - support + 0.5) as i64).max(0);
        let xmax = ((center + support + 0.5) as i64).min(in_size as i64);
        let cnt = (xmax - xmin) as usize;
        let mut wsum = 0.0;
        for (x, wx) in w[..cnt].iter_mut().enumerate() {
            *wx = bicubic(((x as i64 + xmin) as f64 - center + 0.5) * ss);
            wsum += *wx;
        }
        for (x, &wx) in w[..cnt].iter().enumerate() {
            let kf = if wsum != 0.0 { wx / wsum } else { wx };
            let q = kf * (1i64 << PRECISION_BITS) as f64;
            kk[xx * taps + x] = (if kf < 0.0 { q - 0.5 } else { q + 0.5 }) as i32;
        }
        bounds.push((xmin as usize, cnt));
    }
    Coeffs { taps, bounds, kk }
}

/// Resample.c's `clip8`: round via the pre-added half-ulp, shift, clamp.
#[inline]
fn clip8(ss: i64) -> u8 {
    if ss >= 1 << (PRECISION_BITS + 8) {
        255
    } else if ss <= 0 {
        0
    } else {
        (ss >> PRECISION_BITS) as u8
    }
}

fn pass_horizontal(src: &[u8], sw: usize, h: usize, dw: usize) -> Vec<u8> {
    let c = coeffs(sw, dw);
    let mut out = vec![0u8; dw * h * 3];
    for y in 0..h {
        let row = &src[y * sw * 3..][..sw * 3];
        let orow = &mut out[y * dw * 3..][..dw * 3];
        for xx in 0..dw {
            let (xmin, cnt) = c.bounds[xx];
            let k = &c.kk[xx * c.taps..][..cnt];
            for ch in 0..3 {
                let mut ss: i64 = 1 << (PRECISION_BITS - 1);
                for (x, &kv) in k.iter().enumerate() {
                    ss += row[(xmin + x) * 3 + ch] as i64 * kv as i64;
                }
                orow[xx * 3 + ch] = clip8(ss);
            }
        }
    }
    out
}

fn pass_vertical(src: &[u8], w: usize, sh: usize, dh: usize) -> Vec<u8> {
    let c = coeffs(sh, dh);
    let mut out = vec![0u8; w * dh * 3];
    for yy in 0..dh {
        let (ymin, cnt) = c.bounds[yy];
        let k = &c.kk[yy * c.taps..][..cnt];
        let orow = &mut out[yy * w * 3..][..w * 3];
        for x in 0..w * 3 {
            let mut ss: i64 = 1 << (PRECISION_BITS - 1);
            for (y, &kv) in k.iter().enumerate() {
                ss += src[(ymin + y) * w * 3 + x] as i64 * kv as i64;
            }
            orow[x] = clip8(ss);
        }
    }
    out
}

/// `Image.resize((dw, dh))` on interleaved RGB8 - PIL's default BICUBIC,
/// horizontal pass first, uint8 clip between the passes, and a pass whose
/// size is unchanged is SKIPPED (PIL's `need_horizontal`/`need_vertical`),
/// which matters because a skipped pass never re-rounds.
pub(crate) fn resize_rgb8(src: &[u8], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<u8> {
    assert_eq!(src.len(), sw * sh * 3, "src is not {sw}x{sh} RGB8");
    assert!(dw > 0 && dh > 0);
    if dw == sw && dh == sh {
        return src.to_vec();
    }
    let horiz = (dw != sw).then(|| pass_horizontal(src, sw, sh, dw));
    let hbuf: &[u8] = horiz.as_deref().unwrap_or(src);
    if dh != sh {
        pass_vertical(hbuf, dw, sh, dh)
    } else {
        hbuf.to_vec()
    }
}

/// `ImageOps.pad(image, (out, out), color=fill)` - contain-resize preserving
/// aspect ratio, then paste centered on a fill-colored square. Python's
/// `round()` is banker's rounding (ties to even), used both for the contained
/// size and the paste offset; `round_ties_even` mirrors it.
pub(crate) fn pad_to_square(
    src: &[u8],
    sw: usize,
    sh: usize,
    out: usize,
    fill: [u8; 3],
) -> Vec<u8> {
    assert_eq!(src.len(), sw * sh * 3, "src is not {sw}x{sh} RGB8");
    let im_ratio = sw as f64 / sh as f64;
    let (rw, rh) = if im_ratio == 1.0 {
        (out, out)
    } else if im_ratio > 1.0 {
        let nh = (sh as f64 / sw as f64 * out as f64).round_ties_even() as usize;
        (out, if nh != out { nh } else { out })
    } else {
        let nw = (sw as f64 / sh as f64 * out as f64).round_ties_even() as usize;
        (if nw != out { nw } else { out }, out)
    };
    let resized = resize_rgb8(src, sw, sh, rw, rh);
    let mut buf: Vec<u8> = fill.iter().copied().cycle().take(out * out * 3).collect();
    let (x0, y0) = if rw != out {
        ((((out - rw) as f64) * 0.5).round_ties_even() as usize, 0)
    } else {
        (0, (((out - rh) as f64) * 0.5).round_ties_even() as usize)
    };
    for y in 0..rh {
        let s = &resized[y * rw * 3..][..rw * 3];
        buf[((y0 + y) * out + x0) * 3..][..rw * 3].copy_from_slice(s);
    }
    buf
}

/// One `tile_px` crop out of the resized grid image, `(col, row)` of the tile
/// grid - `resized_img.crop(box)` with the reference's row-major box walk.
pub(crate) fn crop_tile(src: &[u8], w: usize, tile_px: usize, col: usize, row: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(tile_px * tile_px * 3);
    for y in 0..tile_px {
        let off = ((row * tile_px + y) * w + col * tile_px) * 3;
        out.extend_from_slice(&src[off..off + tile_px * 3]);
    }
    out
}

/// ToTensor + Normalize: `(x/255 - mean_c) / std_c`, f32, channel-interleaved
/// in and out (the layout [`super::vision::DeepEncoder::patch_rows`] eats).
// Serving normalizes on device now (`pd_ocr_patches_u8`, built
// to be bit-identical to this expression); this stays as the host reference
// the preprocess tests pin the math against.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn normalize_rgb8(rgb: &[u8], mean: [f32; 3], std: [f32; 3]) -> Vec<f32> {
    rgb.iter()
        .enumerate()
        .map(|(i, &b)| (b as f32 / 255.0 - mean[i % 3]) / std[i % 3])
        .collect()
}

#[cfg(test)]
mod fixtures {
    include!("preprocess_fixtures.rs");
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Byte LCG shared with `gen-preprocess-fixtures.py`, so the big-geometry
    /// checks regenerate their input instead of committing megabytes.
    fn lcg_rgb(n: usize) -> Vec<u8> {
        let mut s = 0x00c0_ffeeu64;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (s >> 40) as u8
            })
            .collect()
    }

    fn fnv1a(bytes: &[u8]) -> u64 {
        let mut h = 0xcbf29ce484222325u64;
        for &b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        h
    }

    /// Awkward-prime geometries, full output committed: downscale (the
    /// antialias kernel-widening path) and upscale, byte-for-byte vs PIL.
    #[test]
    fn resize_matches_pil_exactly() {
        let src = lcg_rgb(37 * 53 * 3);
        assert_eq!(resize_rgb8(&src, 37, 53, 29, 41), fixtures::RS_DOWN_29X41);
        assert_eq!(resize_rgb8(&src, 37, 53, 64, 80), fixtures::RS_UP_64X80);
    }

    /// ImageOps.pad byte-parity, including the banker's-rounded contain size
    /// and paste offset, on both orientations.
    #[test]
    fn pad_matches_pil_exactly() {
        let src = lcg_rgb(37 * 53 * 3);
        assert_eq!(
            pad_to_square(&src, 37, 53, 64, [127; 3]),
            fixtures::PAD_TALL_64
        );
        let src = lcg_rgb(53 * 37 * 3);
        assert_eq!(
            pad_to_square(&src, 53, 37, 64, [127; 3]),
            fixtures::PAD_WIDE_64
        );
    }

    /// The real request geometries (battery page 1240×1754): the crop-grid
    /// resize to 1280×1920 and the padded 1024² global view, digest-pinned.
    #[test]
    fn battery_geometries_match_pil() {
        let src = lcg_rgb(1240 * 1754 * 3);
        let crops = resize_rgb8(&src, 1240, 1754, 1280, 1920);
        assert_eq!(
            fnv1a(&crops),
            fixtures::RS_BATTERY_FNV,
            "1240x1754 -> 1280x1920"
        );
        let global = pad_to_square(&src, 1240, 1754, 1024, [127; 3]);
        assert_eq!(
            fnv1a(&global),
            fixtures::PAD_BATTERY_FNV,
            "1240x1754 -> pad 1024"
        );
    }

    /// A pass whose size is unchanged must be skipped, not run as identity -
    /// running it would re-round every byte through the fixed-point path.
    #[test]
    fn unchanged_axis_skips_its_pass() {
        let src = lcg_rgb(24 * 16 * 3);
        assert_eq!(resize_rgb8(&src, 24, 16, 24, 16), src);
        assert_eq!(resize_rgb8(&src, 24, 16, 24, 8), fixtures::RS_VONLY_24X8);
    }

    #[test]
    fn crop_and_normalize_shapes() {
        let src = lcg_rgb(20 * 10 * 3);
        let t = crop_tile(&src, 20, 10, 1, 0);
        assert_eq!(t.len(), 10 * 10 * 3);
        assert_eq!(t[0..3], src[10 * 3..10 * 3 + 3]);
        let n = normalize_rgb8(&t, [0.5; 3], [0.5; 3]);
        assert_eq!(n.len(), t.len());
        assert!((n[0] - (t[0] as f32 / 255.0 - 0.5) / 0.5).abs() < 1e-7);
    }
}
