//! Granite-Vision AnyRes preprocessing: turning one picture into the tile set
//! the tower eats, and into the exact row layout the LLM expects.
//!
//! Two halves, both host-side and both free of device buffers so they can be
//! tested directly:
//!
//!   **Tiling** (LLaVA-NeXT AnyRes). Pick the best of 27 grid pinpoints for the
//!   image's aspect, aspect-resize the image into it, centre-pad to the pinpoint,
//!   cut it into 384px tiles - and prepend a **base tile** that is the whole
//!   image squashed to 384×384. So a 640×480 photo becomes 1 base + a 2×2 grid
//!   = 5 tiles, each 576 SigLIP tokens -> 144 projected tokens.
//!
//!   **Packing** (`pack_and_unpad_image_features`). The base tile's 144 rows go
//!   first, untouched. The grid tiles are then read as one feature map - tile
//!   (ty,tx) occupying cells [ty·12, ty·12+12) × [tx·12, tx·12+12) - the padding
//!   that the resize added is cut back off, and the map is emitted **row-major
//!   across tiles** with a learned `image_newline` row closing every feature row.
//!
//! ## Why this does not follow llama.cpp
//!
//! llama.cpp serves this model too, and its granite path packs differently:
//! one newline per TILE appended at that tile's end, tiles emitted whole and
//! in order, no unpadding, and an image ≤384px on both axes served as a lone
//! 384×384 tile with no base/grid split at all (`mtmd-image.cpp`
//! `get_slice_instructions`: `has_slices = w > 384 || h > 384`). Upstream has no
//! such case - `image_size_to_num_patches` always returns ≥2, so every image
//! goes through the base+grid path. For a 640×480 photo that is 594 rows here
//! against 725 there, in a different order.
//!
//! We follow upstream, because that is the layout the model was trained on and
//! the count the HF processor's `_get_number_of_features` gives a client. The
//! cost is that llama.cpp stops being a usable greedy oracle for image prompts
//! - the third place granite-vision departs from it, after the `granite-docling`
//!   pre-tokenizer and the f16-vs-f32 mmproj, and like both
//!   of those we are the side that matches the spec, so the divergence is not
//!   chased as a parity bug.
//!
//! ## Trap: the pinpoint list is (height, width)
//!
//! `config.json` and the GGUF's `clip.vision.image_grid_pinpoints` store pairs
//! in HF's `(height, width)` order; llama.cpp reads the same array as
//! `(width, height)` (`clip.cpp` builds `clip_image_size{pinpoints[i],
//! pinpoints[i+1]}`). It happens not to matter for this model because the list
//! is closed under transpose - every `384×768` has a `768×384` - so both
//! readings enumerate the same set of geometric sizes. It would matter for any
//! checkpoint whose list is not symmetric. This module works in geometric
//! `(width, height)` and the loader transposes at the file boundary.

/// Geometry the plan needs, lifted out of `VisionHparams` so the arithmetic is
/// testable without a GPU or a model file.
#[derive(Clone, Copy, Debug)]
pub struct TileGeom {
    /// Tile side in pixels - 384.
    pub image_size: usize,
    /// Projected tokens per tile side - 12 (24 patches × the 4/8 downsample).
    pub tokens_side: usize,
}

/// One row of the packed stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PackRow {
    /// Row `idx` of tile `tile`, indexing the encoder's tile-major output.
    Feature { tile: usize, idx: usize },
    /// The learned `image_newline` parameter.
    Newline,
}

/// Everything about one image's layout, decided before a single pixel moves.
#[derive(Clone, Debug)]
pub struct AnyResPlan {
    /// Source size in pixels, (width, height).
    pub orig: (usize, usize),
    /// The selected pinpoint, geometric (width, height).
    pub best: (usize, usize),
    /// Grid tiles across and down - `best / image_size`.
    pub grid: (usize, usize),
    /// The aspect-preserving resize applied before centre-padding to `best`.
    pub resized: (usize, usize),
    /// The unpadded window over the combined feature map, in projected cells:
    /// `(x0, y0, x1, y1)` half-open.
    pub win: (usize, usize, usize, usize),
    pub geom: TileGeom,
}

impl AnyResPlan {
    /// `pinpoints` are geometric `(width, height)`, in FILE ORDER - the order
    /// decides ties in `select_best_resolution`, so it must not be sorted.
    pub fn new(w: usize, h: usize, geom: TileGeom, pinpoints: &[(usize, usize)]) -> Option<Self> {
        if w == 0 || h == 0 || pinpoints.is_empty() {
            return None;
        }
        let best = select_best_resolution(w, h, pinpoints)?;
        let s = geom.image_size;
        let grid = (best.0 / s, best.1 / s);
        let resized = patch_output_size(w, h, best.0, best.1);
        let win = unpad_window(w, h, grid, geom.tokens_side);
        Some(Self {
            orig: (w, h),
            best,
            grid,
            resized,
            win,
            geom,
        })
    }

    /// Tiles the tower must encode: 1 base + the grid.
    pub fn n_tiles(&self) -> usize {
        1 + self.grid.0 * self.grid.1
    }

    /// Rows the LLM must reserve - base + unpadded grid cells + one newline per
    /// surviving feature row. Matches `LlavaNextProcessor._get_number_of_features`.
    pub fn n_tokens(&self) -> usize {
        let (x0, y0, x1, y1) = self.win;
        let ts = self.geom.tokens_side;
        ts * ts + (y1 - y0) * ((x1 - x0) + 1)
    }

    /// The packing order. Base tile first and whole, then the unpadded feature
    /// map row-major across tiles with a newline closing each row.
    ///
    /// The across-tiles walk is the part that is silent when wrong: emitting
    /// tile-by-tile instead would produce the same row count for a 1×N grid and
    /// a plausible-looking wrong answer for every other shape.
    pub fn rows(&self) -> Vec<PackRow> {
        let ts = self.geom.tokens_side;
        let (x0, y0, x1, y1) = self.win;
        let mut out = Vec::with_capacity(self.n_tokens());
        out.extend((0..ts * ts).map(|idx| PackRow::Feature { tile: 0, idx }));
        for y in y0..y1 {
            for x in x0..x1 {
                out.push(PackRow::Feature {
                    tile: 1 + (y / ts) * self.grid.0 + (x / ts),
                    idx: (y % ts) * ts + (x % ts),
                });
            }
            out.push(PackRow::Newline);
        }
        out
    }

    /// Cut `rgb` (tightly packed interleaved RGB8, `w`×`h`) into the tiles the
    /// tower encodes, each `image_size²` RGB8. Tile 0 is the base.
    ///
    /// The base tile is a plain squash to 384×384 - aspect is not preserved
    /// there, deliberately: it is the overview, and the grid tiles carry the
    /// aspect-correct detail.
    pub fn tiles(&self, rgb: &[u8], w: usize, h: usize) -> Vec<Vec<u8>> {
        assert_eq!(rgb.len(), 3 * w * h, "expected tightly-packed RGB8");
        assert_eq!(
            (w, h),
            self.orig,
            "plan was built for a different image size"
        );
        let s = self.geom.image_size;
        let mut out = Vec::with_capacity(self.n_tiles());
        out.push(resize_bicubic_pillow(rgb, w, h, s, s));

        // aspect-resize into the pinpoint, then centre-pad with black. The odd
        // pixel goes right/bottom, matching `_get_padding_size`'s divmod.
        let (rw, rh) = self.resized;
        let scaled = resize_bicubic_pillow(rgb, w, h, rw, rh);
        let (bw, bh) = self.best;
        let (ox, oy) = ((bw - rw) / 2, (bh - rh) / 2);
        let mut padded = vec![0u8; 3 * bw * bh];
        for y in 0..rh {
            let src = y * rw * 3;
            let dst = ((y + oy) * bw + ox) * 3;
            padded[dst..dst + rw * 3].copy_from_slice(&scaled[src..src + rw * 3]);
        }

        // raster order over the grid - `divide_to_patches` walks rows outer.
        for ty in 0..self.grid.1 {
            for tx in 0..self.grid.0 {
                let mut tile = vec![0u8; 3 * s * s];
                for y in 0..s {
                    let src = ((ty * s + y) * bw + tx * s) * 3;
                    tile[y * s * 3..(y + 1) * s * 3].copy_from_slice(&padded[src..src + s * 3]);
                }
                out.push(tile);
            }
        }
        out
    }
}

/// `select_best_resolution` - maximize the effective (post-downscale) pixel
/// count, break ties on least wasted canvas, and on a full tie keep the first
/// candidate. HF works in (height, width); this is the same loop transposed,
/// walking `pinpoints` in file order so ties resolve identically.
fn select_best_resolution(
    ow: usize,
    oh: usize,
    pinpoints: &[(usize, usize)],
) -> Option<(usize, usize)> {
    let (mut best, mut max_eff, mut min_waste) = (None, 0usize, usize::MAX);
    for &(cw, ch) in pinpoints {
        if cw == 0 || ch == 0 {
            continue;
        }
        let scale = (cw as f64 / ow as f64).min(ch as f64 / oh as f64);
        // int() truncation, not rounding - HF's `int(original_width * scale)`
        let (dw, dh) = ((ow as f64 * scale) as usize, (oh as f64 * scale) as usize);
        let eff = (dw * dh).min(ow * oh);
        let waste = cw * ch - eff;
        if eff > max_eff || (eff == max_eff && waste < min_waste) {
            (best, max_eff, min_waste) = (Some((cw, ch)), eff, waste);
        }
    }
    best
}

/// `get_patch_output_size` - fit the image inside the pinpoint preserving
/// aspect. Ceil on the free axis, clamped so it can never exceed the canvas.
fn patch_output_size(ow: usize, oh: usize, tw: usize, th: usize) -> (usize, usize) {
    let (sw, sh) = (tw as f64 / ow as f64, th as f64 / oh as f64);
    if sw < sh {
        (tw, ((oh as f64 * sw).ceil() as usize).min(th))
    } else {
        (((ow as f64 * sh).ceil() as usize).min(tw), th)
    }
}

/// `unpad_image`, expressed over projected cells instead of pixels: which
/// window of the `grid × tokens_side` feature map is real image rather than the
/// black padding the resize added.
///
/// Two details that decide the token count:
///
/// - `int(round(x, 7))` is a TRUNCATION after rounding, so a surviving extent
///   of 0.96 cells becomes 0, not 1. That is how an extreme strip (5000×40)
///   collapses its entire grid to newline rows and serves only the base tile's
///   144 features.
/// - the slice is `[pad, current - pad)`, so an odd shortfall leaves one extra
///   cell rather than being split - the window is not always `new_extent` wide.
fn unpad_window(
    ow: usize,
    oh: usize,
    grid: (usize, usize),
    ts: usize,
) -> (usize, usize, usize, usize) {
    let (cw, ch) = (grid.0 * ts, grid.1 * ts);
    let orig_ar = ow as f64 / oh as f64;
    let cur_ar = cw as f64 / ch as f64;
    let trunc7 = |v: f64| -> usize {
        let r = (v * 1e7).round() / 1e7;
        if r <= 0.0 { 0 } else { r as usize }
    };
    if orig_ar > cur_ar {
        let new_h = trunc7(oh as f64 * (cw as f64 / ow as f64));
        let pad = ch.saturating_sub(new_h) / 2;
        (0, pad, cw, ch - pad)
    } else {
        let new_w = trunc7(ow as f64 * (ch as f64 / oh as f64));
        let pad = cw.saturating_sub(new_w) / 2;
        (pad, 0, cw - pad, ch)
    }
}

// ---------------------------------------------------------------------------
// Pillow-compatible bicubic
// ---------------------------------------------------------------------------

/// Pillow's bicubic kernel, a = -0.5 (the Catmull-Rom parameter PIL and
/// torchvision both use). Support is 2.0 before the antialias stretch.
fn bicubic(x: f32) -> f32 {
    const A: f32 = -0.5;
    let x = x.abs();
    if x < 1.0 {
        ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * A
    } else {
        0.0
    }
}

/// Per-output-pixel filter taps: `(first_source_index, weights)`.
///
/// This is Pillow's `precompute_coeffs`: on a downscale the kernel is STRETCHED
/// by the scale factor (that stretch is the antialiasing - a plain 4-tap
/// bicubic aliases badly when shrinking a photo to 384px, which is most of what
/// this model sees), then the taps are renormalized to sum to 1.
fn coeffs(in_size: usize, out_size: usize) -> Vec<(usize, Vec<f32>)> {
    let scale = in_size as f64 / out_size as f64;
    let filterscale = scale.max(1.0);
    let support = 2.0 * filterscale;
    let inv = 1.0 / filterscale;
    (0..out_size)
        .map(|xx| {
            let center = (xx as f64 + 0.5) * scale;
            // C truncation toward zero, then clamp - identical to Pillow's
            // `(int)(center - support + 0.5)` followed by the bounds check.
            let xmin = ((center - support + 0.5) as i64).max(0) as usize;
            let xmax = (((center + support + 0.5) as i64).max(0) as usize).min(in_size);
            let mut w: Vec<f32> = (xmin..xmax)
                .map(|x| bicubic((((x as f64 - center) + 0.5) * inv) as f32))
                .collect();
            let sum: f32 = w.iter().sum();
            if sum != 0.0 {
                for v in &mut w {
                    *v /= sum;
                }
            }
            (xmin, w)
        })
        .collect()
}

/// Separable bicubic resize of tightly-packed RGB8, following Pillow's
/// horizontal-then-vertical order INCLUDING the u8 round-trip between passes
/// (PIL's 8-bit path stores the intermediate as an image, and that rounding is
/// observable in the output).
///
/// INTERIM, per the SOTA-implementation rule: Pillow accumulates in 22-bit
/// fixed point and we accumulate in f32. The filter taps, support, bounds and
/// normalization - everything that decides the filter's SHAPE - are identical;
/// only the last-bit rounding can differ, and by at most 1 LSB. That is two
/// orders of magnitude under the f16-vs-f32 gap the mmproj already carries,
/// so it is not worth carrying an integer path for.
pub fn resize_bicubic_pillow(src: &[u8], sw: usize, sh: usize, tw: usize, th: usize) -> Vec<u8> {
    assert_eq!(src.len(), 3 * sw * sh, "expected tightly-packed RGB8");
    assert!(tw > 0 && th > 0 && sw > 0 && sh > 0);
    if sw == tw && sh == th {
        return src.to_vec();
    }
    let round8 = |v: f32| -> u8 { (v + 0.5).clamp(0.0, 255.0) as u8 };

    // horizontal: sw -> tw, height unchanged
    let mid: Vec<u8> = if sw == tw {
        src.to_vec()
    } else {
        let cx = coeffs(sw, tw);
        let mut out = vec![0u8; 3 * tw * sh];
        for y in 0..sh {
            let row = y * sw * 3;
            for (x, (xmin, w)) in cx.iter().enumerate() {
                for c in 0..3 {
                    let mut acc = 0f32;
                    for (j, &k) in w.iter().enumerate() {
                        acc += k * src[row + (xmin + j) * 3 + c] as f32;
                    }
                    out[(y * tw + x) * 3 + c] = round8(acc);
                }
            }
        }
        out
    };

    // vertical: sh -> th
    if sh == th {
        return mid;
    }
    let cy = coeffs(sh, th);
    let mut out = vec![0u8; 3 * tw * th];
    for (y, (ymin, w)) in cy.iter().enumerate() {
        for x in 0..tw {
            for c in 0..3 {
                let mut acc = 0f32;
                for (j, &k) in w.iter().enumerate() {
                    acc += k * mid[((ymin + j) * tw + x) * 3 + c] as f32;
                }
                out[(y * tw + x) * 3 + c] = round8(acc);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The model's 27 pinpoints, transposed out of the file's (height, width)
    /// into geometric (width, height) exactly as the loader does.
    fn pinpoints() -> Vec<(usize, usize)> {
        const FILE_ORDER_HW: &[(usize, usize)] = &[
            (384, 384),
            (384, 768),
            (384, 1152),
            (384, 1536),
            (384, 1920),
            (384, 2304),
            (384, 2688),
            (384, 3072),
            (384, 3456),
            (384, 3840),
            (768, 384),
            (768, 768),
            (768, 1152),
            (768, 1536),
            (768, 1920),
            (1152, 384),
            (1152, 768),
            (1152, 1152),
            (1536, 384),
            (1536, 768),
            (1920, 384),
            (1920, 768),
            (2304, 384),
            (2688, 384),
            (3072, 384),
            (3456, 384),
            (3840, 384),
        ];
        FILE_ORDER_HW.iter().map(|&(h, w)| (w, h)).collect()
    }

    fn geom() -> TileGeom {
        TileGeom {
            image_size: 384,
            tokens_side: 12,
        }
    }

    /// Generated by `scratchpad/gv-anyres-oracle.py`, which calls transformers'
    /// own `select_best_resolution` / `get_patch_output_size` /
    /// `image_size_to_num_patches` plus `LlavaNextProcessor._get_unpadded_features`
    /// verbatim. Columns:
    /// (w, h, best_w, best_h, grid_x, grid_y, n_tiles, resized_w, resized_h,
    ///  cell_w, cell_h, n_tokens)
    const HF: &[(
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
    )] = &[
        (384, 384, 384, 384, 1, 1, 2, 384, 384, 12, 12, 300),
        (383, 383, 384, 384, 1, 1, 2, 384, 384, 12, 12, 300),
        (385, 385, 768, 768, 2, 2, 5, 768, 768, 24, 24, 744),
        (200, 200, 384, 384, 1, 1, 2, 384, 384, 12, 12, 300),
        (64, 64, 384, 384, 1, 1, 2, 384, 384, 12, 12, 300),
        (1, 1, 384, 384, 1, 1, 2, 384, 384, 12, 12, 300),
        (1000, 1000, 1152, 1152, 3, 3, 10, 1152, 1152, 36, 36, 1476),
        (1152, 1152, 1152, 1152, 3, 3, 10, 1152, 1152, 36, 36, 1476),
        (2000, 2000, 1152, 1152, 3, 3, 10, 1152, 1152, 36, 36, 1476),
        (4096, 4096, 1152, 1152, 3, 3, 10, 1152, 1152, 36, 36, 1476),
        (800, 400, 1152, 768, 3, 2, 7, 1152, 576, 36, 18, 810),
        (1024, 300, 1152, 384, 3, 1, 4, 1152, 338, 36, 10, 514),
        (3840, 384, 3840, 384, 10, 1, 11, 3840, 384, 120, 12, 1596),
        (1920, 200, 1920, 384, 5, 1, 6, 1920, 200, 60, 6, 510),
        (640, 480, 768, 768, 2, 2, 5, 768, 576, 24, 18, 594),
        (1600, 400, 1920, 768, 5, 2, 11, 1920, 480, 60, 16, 1120),
        (400, 800, 768, 1152, 2, 3, 7, 576, 1152, 18, 36, 828),
        (300, 1024, 384, 1152, 1, 3, 4, 338, 1152, 10, 36, 540),
        (384, 3840, 384, 3840, 1, 10, 11, 384, 3840, 12, 120, 1704),
        (200, 1920, 384, 1920, 1, 5, 6, 200, 1920, 6, 60, 564),
        (480, 640, 768, 768, 2, 2, 5, 576, 768, 18, 24, 600),
        (400, 1600, 768, 1920, 2, 5, 11, 480, 1920, 16, 60, 1164),
        (4000, 100, 3840, 384, 10, 1, 11, 3840, 96, 120, 4, 628),
        (100, 4000, 384, 3840, 1, 10, 11, 96, 3840, 4, 120, 744),
        (5000, 40, 3840, 384, 10, 1, 11, 3840, 31, 120, 0, 144),
        (40, 5000, 384, 3840, 1, 10, 11, 31, 3840, 0, 120, 264),
        (777, 333, 1152, 384, 3, 1, 4, 896, 384, 28, 12, 492),
        (333, 777, 384, 1152, 1, 3, 4, 384, 896, 12, 28, 508),
        (1023, 767, 1152, 768, 3, 2, 7, 1025, 768, 32, 24, 936),
        (511, 1021, 768, 1152, 2, 3, 7, 577, 1152, 18, 36, 828),
    ];

    #[test]
    fn plan_matches_the_hf_processor_on_every_fixture() {
        let (pp, g) = (pinpoints(), geom());
        for &(w, h, bw, bh, gx, gy, nt, rw, rh, cw, ch, tok) in HF {
            let p = AnyResPlan::new(w, h, g, &pp).expect("plan");
            assert_eq!(p.best, (bw, bh), "{w}x{h} best resolution");
            assert_eq!(p.grid, (gx, gy), "{w}x{h} grid");
            assert_eq!(p.n_tiles(), nt, "{w}x{h} tile count");
            assert_eq!(p.resized, (rw, rh), "{w}x{h} aspect resize");
            let (x0, y0, x1, y1) = p.win;
            assert_eq!((x1 - x0, y1 - y0), (cw, ch), "{w}x{h} unpadded cells");
            assert_eq!(p.n_tokens(), tok, "{w}x{h} token count");
        }
    }

    #[test]
    fn rows_length_always_equals_the_advertised_token_count() {
        let (pp, g) = (pinpoints(), geom());
        for &(w, h, ..) in HF {
            let p = AnyResPlan::new(w, h, g, &pp).unwrap();
            assert_eq!(p.rows().len(), p.n_tokens(), "{w}x{h}");
        }
    }

    /// The base tile is 144 rows, in order, and carries no newline. Getting a
    /// newline in there (llama.cpp's single-tile shape) would shift every
    /// subsequent row by one with no error anywhere.
    #[test]
    fn base_tile_comes_first_whole_and_bare() {
        let p = AnyResPlan::new(640, 480, geom(), &pinpoints()).unwrap();
        let rows = p.rows();
        for (i, r) in rows[..144].iter().enumerate() {
            assert_eq!(*r, PackRow::Feature { tile: 0, idx: i });
        }
        assert!(!matches!(rows[144], PackRow::Newline));
    }

    /// The grid walk is row-major across tiles, not tile-by-tile. On a 2×2 grid
    /// the first feature row must read tile 1's row 0 (12 cells) then tile 2's
    /// row 0 (12 cells) and only then a newline.
    #[test]
    fn feature_map_is_walked_across_tiles_not_tile_by_tile() {
        // 768x768 -> 2x2 grid, square so nothing is unpadded
        let p = AnyResPlan::new(768, 768, geom(), &pinpoints()).unwrap();
        assert_eq!(p.grid, (2, 2));
        let rows = p.rows();
        let first_line = &rows[144..144 + 25];
        for x in 0..12 {
            assert_eq!(
                first_line[x],
                PackRow::Feature { tile: 1, idx: x },
                "left half col {x}"
            );
        }
        for x in 0..12 {
            assert_eq!(
                first_line[12 + x],
                PackRow::Feature { tile: 2, idx: x },
                "right half col {x}"
            );
        }
        assert_eq!(first_line[24], PackRow::Newline);
        // and the row below crosses back to tile 1, row 1
        assert_eq!(rows[144 + 25], PackRow::Feature { tile: 1, idx: 12 });
    }

    /// A newline closes every feature row and nothing else.
    #[test]
    fn newlines_land_only_at_row_ends() {
        let p = AnyResPlan::new(1024, 300, geom(), &pinpoints()).unwrap();
        let (x0, _, x1, _) = p.win;
        let stride = (x1 - x0) + 1;
        for (i, r) in p.rows()[144..].iter().enumerate() {
            assert_eq!(
                matches!(r, PackRow::Newline),
                i % stride == stride - 1,
                "row {i} of a {stride}-wide line"
            );
        }
    }

    /// Padding really is cut off: a 4000x100 strip lands in a 3840x384 canvas,
    /// so most of the feature map is black bars and only 4 of 12 cell rows
    /// survive - centred, so the first surviving row is not row 0.
    #[test]
    fn unpadding_drops_the_black_bars_and_stays_centred() {
        let p = AnyResPlan::new(4000, 100, geom(), &pinpoints()).unwrap();
        assert_eq!(p.win, (0, 4, 120, 8));
        let rows = p.rows();
        // first grid row reads cell y=4 => tile row 4 of the top tile band
        assert_eq!(
            rows[144],
            PackRow::Feature {
                tile: 1,
                idx: 4 * 12
            }
        );
    }

    /// The degenerate end of `int(round(x, 7))`: an extreme strip keeps zero
    /// grid columns, so the image reduces to the base tile plus bare newlines.
    /// Upstream does this; it is not a bug on our side and the count has to
    /// match or the placeholder run desyncs.
    #[test]
    fn extreme_strip_collapses_the_grid_to_newlines() {
        let p = AnyResPlan::new(40, 5000, geom(), &pinpoints()).unwrap();
        assert_eq!(p.win.0, p.win.2, "no columns survive");
        assert_eq!(p.n_tokens(), 144 + 120);
        assert!(p.rows()[144..].iter().all(|r| *r == PackRow::Newline));

        // the transpose keeps zero ROWS, so there is no grid contribution at all
        let p = AnyResPlan::new(5000, 40, geom(), &pinpoints()).unwrap();
        assert_eq!(p.n_tokens(), 144);
        assert_eq!(p.rows().len(), 144);
    }

    /// Every row index a plan emits must exist in the encoder's tile-major
    /// output, or the gather reads out of bounds on some aspect ratio nobody
    /// tested by hand.
    #[test]
    fn every_feature_row_is_in_range_for_every_fixture() {
        let (pp, g) = (pinpoints(), geom());
        for &(w, h, ..) in HF {
            let p = AnyResPlan::new(w, h, g, &pp).unwrap();
            let cap = p.n_tiles() * g.tokens_side * g.tokens_side;
            for r in p.rows() {
                if let PackRow::Feature { tile, idx } = r {
                    assert!(
                        tile < p.n_tiles(),
                        "{w}x{h}: tile {tile} >= {}",
                        p.n_tiles()
                    );
                    assert!(idx < g.tokens_side * g.tokens_side, "{w}x{h}: idx {idx}");
                    assert!(tile * g.tokens_side * g.tokens_side + idx < cap);
                }
            }
        }
    }

    #[test]
    fn tiles_are_the_right_count_and_size() {
        let (w, h) = (200usize, 137usize);
        let rgb: Vec<u8> = (0..3 * w * h).map(|i| (i % 251) as u8).collect();
        let p = AnyResPlan::new(w, h, geom(), &pinpoints()).unwrap();
        let tiles = p.tiles(&rgb, w, h);
        assert_eq!(tiles.len(), p.n_tiles());
        for t in &tiles {
            assert_eq!(t.len(), 3 * 384 * 384);
        }
    }

    /// The grid tile must carry the image in the middle with black bars, and
    /// the base tile must be a full-bleed squash. A 800x400 image goes into a
    /// 1152x768 canvas as 1152x576, so 96 black rows top and bottom.
    #[test]
    fn grid_tiles_are_centre_padded_and_the_base_tile_is_not() {
        let (w, h) = (800usize, 400usize);
        let rgb = vec![200u8; 3 * w * h];
        let p = AnyResPlan::new(w, h, geom(), &pinpoints()).unwrap();
        assert_eq!(p.resized, (1152, 576));
        let tiles = p.tiles(&rgb, w, h);
        // base: uniform grey everywhere, no bars
        assert!(
            tiles[0].iter().all(|&v| v > 150),
            "base tile should be full bleed"
        );
        // grid tile 0 (top-left of a 3x2 grid): rows 0..96 are pad, 96.. are image
        let t = &tiles[1];
        assert!(
            t[..3 * 384 * 90].iter().all(|&v| v == 0),
            "top bar should be black"
        );
        assert!(
            t[3 * 384 * 100..3 * 384 * 200].iter().all(|&v| v > 150),
            "image band"
        );
    }

    /// A flat image must survive any resize unchanged - catches normalization
    /// bugs in the filter taps, which would show up as edge darkening.
    #[test]
    fn bicubic_preserves_a_constant_image() {
        for (sw, sh, tw, th) in [
            (100, 100, 384, 384),
            (1000, 700, 384, 384),
            (37, 300, 384, 384),
        ] {
            let src = vec![137u8; 3 * sw * sh];
            let out = resize_bicubic_pillow(&src, sw, sh, tw, th);
            assert_eq!(out.len(), 3 * tw * th);
            assert!(
                out.iter().all(|&v| v == 137),
                "{sw}x{sh} -> {tw}x{th} drifted"
            );
        }
    }

    /// Downscaling must ANTIALIAS: a 1px checkerboard shrunk 8x should average
    /// to mid grey, not sample one phase and come back as a checkerboard. This
    /// is the whole reason the kernel is stretched by the scale factor, and a
    /// plain 4-tap bicubic fails it.
    #[test]
    fn bicubic_antialiases_on_downscale() {
        let (sw, sh) = (512usize, 512usize);
        let mut src = vec![0u8; 3 * sw * sh];
        for y in 0..sh {
            for x in 0..sw {
                let v = if (x + y) % 2 == 0 { 255 } else { 0 };
                for c in 0..3 {
                    src[(y * sw + x) * 3 + c] = v;
                }
            }
        }
        let out = resize_bicubic_pillow(&src, sw, sh, 64, 64);
        // interior only - the edges legitimately ring
        for y in 4..60 {
            for x in 4..60 {
                let v = out[(y * 64 + x) * 3];
                assert!(
                    (110..=145).contains(&v),
                    "pixel {x},{y} = {v}, expected ~128"
                );
            }
        }
    }

    /// Upscaling stays sharp and monotone: a horizontal ramp must come back a
    /// ramp, non-decreasing left to right.
    #[test]
    fn bicubic_upscale_keeps_a_ramp_monotone() {
        let (sw, sh) = (16usize, 4usize);
        let mut src = vec![0u8; 3 * sw * sh];
        for y in 0..sh {
            for x in 0..sw {
                for c in 0..3 {
                    src[(y * sw + x) * 3 + c] = (x * 17) as u8;
                }
            }
        }
        let out = resize_bicubic_pillow(&src, sw, sh, 128, 32);
        for y in 0..32 {
            for x in 1..128 {
                let (a, b) = (out[(y * 128 + x - 1) * 3], out[(y * 128 + x) * 3]);
                assert!(b >= a, "ramp dipped at x={x}: {a} -> {b}");
            }
        }
    }

    /// The claim "Pillow-compatible" checked against actual Pillow (11.3.0, the
    /// resampler `transformers`' `resample=3` reaches). Fixture from
    /// `scratchpad/gv-bicubic-oracle.py`: the mean over every byte, plus a
    /// prime-stride walk so edge pixels are sampled too - the tap bounds are
    /// where a resampler goes wrong, and an interior-only check would miss it.
    ///
    /// Tolerance is ±1 LSB, which is the fixed-point-vs-f32 gap the module note
    /// predicts. Anything structural (wrong support, missing antialias stretch,
    /// unnormalized taps) moves pixels by tens of levels, not one.
    #[test]
    fn bicubic_matches_pillow() {
        #[allow(clippy::type_complexity)]
        const PIL: &[(usize, usize, usize, usize, f64, usize, &[u8])] = &[
            (
                137,
                91,
                384,
                384,
                107.219073,
                4409,
                &[
                    0, 186, 89, 211, 237, 22, 168, 191, 82, 192, 146, 6, 150, 166, 72, 173, 202, 7,
                    131, 175,
                ],
            ),
            (
                640,
                480,
                384,
                384,
                126.075342,
                4409,
                &[
                    2, 30, 171, 155, 75, 74, 50, 72, 172, 77, 166, 75, 229, 99, 53, 145, 92, 121,
                    151, 99,
                ],
            ),
            (91, 137, 64, 48, 107.459852, 4409, &[5, 101, 212]),
            (
                33,
                400,
                384,
                384,
                119.325272,
                4409,
                &[
                    0, 232, 21, 142, 83, 17, 33, 109, 21, 206, 90, 46, 96, 125, 37, 13, 107, 69,
                    160, 10,
                ],
            ),
        ];
        for &(sw, sh, tw, th, mean, step, samples) in PIL {
            let mut src = vec![0u8; 3 * sw * sh];
            for y in 0..sh {
                for x in 0..sw {
                    let i = (y * sw + x) * 3;
                    src[i] = ((x * 7 + y * 3) % 256) as u8;
                    src[i + 1] = ((x ^ y) % 256) as u8;
                    src[i + 2] = ((x * x + y * y) % 256) as u8;
                }
            }
            let out = resize_bicubic_pillow(&src, sw, sh, tw, th);
            let got = out.iter().map(|&v| v as f64).sum::<f64>() / out.len() as f64;
            assert!(
                (got - mean).abs() < 0.05,
                "{sw}x{sh} -> {tw}x{th}: mean {got:.6} vs Pillow {mean:.6}"
            );
            for (k, &want) in samples.iter().enumerate() {
                let got = out[k * step];
                assert!(
                    got.abs_diff(want) <= 1,
                    "{sw}x{sh} -> {tw}x{th}: byte {} = {got}, Pillow says {want}",
                    k * step
                );
            }
        }
    }

    /// x and y must not be swapped anywhere in the two-pass resize. A left-half
    /// white image stays left-half white; a top-half white image stays top-half.
    #[test]
    fn bicubic_does_not_transpose() {
        let (sw, sh) = (64usize, 32usize);
        let mut left = vec![0u8; 3 * sw * sh];
        let mut top = vec![0u8; 3 * sw * sh];
        for y in 0..sh {
            for x in 0..sw {
                let i = (y * sw + x) * 3;
                if x < sw / 2 {
                    left[i..i + 3].fill(255);
                }
                if y < sh / 2 {
                    top[i..i + 3].fill(255);
                }
            }
        }
        let l = resize_bicubic_pillow(&left, sw, sh, 384, 384);
        assert!(l[(200 * 384 + 20) * 3] > 200 && l[(200 * 384 + 360) * 3] < 55);
        let t = resize_bicubic_pillow(&top, sw, sh, 384, 384);
        assert!(t[(20 * 384 + 200) * 3] > 200 && t[(360 * 384 + 200) * 3] < 55);
    }
}
