//! Annotation overlay generation, ported from the reference implementation's
//! annotation layer. Produces PNG overlays a VLM (or a human) can
//! look at - an ELA heatmap, a noise-variance map, a CFA map, a copy-move
//! border, a signal-composite forensic heatmap, and a severity overview.
//!
//! The VLM-coupled renderers are deliberately not ported: the reference's
//! smart-forensic and VLM-region renderers drive the overlay from
//! stage-2 `vlm_*` bounding boxes, and paddock-forensics has no VLM stage. The
//! signal-composite ("forensic") heatmap is the paddock primary overlay.
//!
//! CPU-only image compositing; there is no GPU path (and none is warranted).

use std::collections::HashMap;
use std::io::Cursor;

use image::codecs::jpeg::JpegEncoder;
use image::{GenericImageView, ImageFormat, ImageReader, Rgb, RgbImage};

use crate::{Context, Finding, Region, Severity};

/// Generated annotation overlays: `type name -> PNG bytes`.
pub struct AnnotationSet {
    pub images: HashMap<String, Vec<u8>>,
}

/// Generate every applicable annotation overlay for an analyzed context.
pub fn render(ctx: &Context, findings: &[Finding]) -> AnnotationSet {
    let mut images = HashMap::new();

    if ctx.is_pdf() {
        // PDFs have no single decoded raster to overlay.
        return AnnotationSet { images };
    }

    if let Some(png) = render_ela(ctx) {
        images.insert("ela".into(), png);
    }
    if let Some(png) = render_noise_map(ctx) {
        images.insert("noise".into(), png);
    }
    if findings.iter().any(|f| f.code == "copy_move_detected")
        && let Some(png) = render_copy_move(ctx)
    {
        images.insert("copy_move".into(), png);
    }
    if let Some(png) = render_cfa_map(ctx) {
        images.insert("cfa".into(), png);
    }
    // Signal-composite forensic heatmap (paddock primary - no VLM smart variant).
    if let Some(png) = render_forensic_composite(ctx, findings) {
        images.insert("forensic".into(), png);
    }
    if let Some(png) = render_overview(ctx, findings) {
        images.insert("overview".into(), png);
    }

    AnnotationSet { images }
}

// ── heatmap helpers (from annotation/heatmap.rs) ─────────────────────────────

/// Blue -> green -> yellow -> red rainbow map for signal overlays.
fn value_to_color(value: f64) -> [u8; 4] {
    let v = value.clamp(0.0, 1.0);
    let (r, g, b) = if v < 0.25 {
        let t = v / 0.25;
        (0.0, t, 1.0 - t * 0.5)
    } else if v < 0.5 {
        let t = (v - 0.25) / 0.25;
        (0.0, 1.0, 0.5 - t * 0.5)
    } else if v < 0.75 {
        let t = (v - 0.5) / 0.25;
        (t, 1.0 - t * 0.3, 0.0)
    } else {
        let t = (v - 0.75) / 0.25;
        (1.0, 0.7 - t * 0.7, 0.0)
    };
    [(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8, 180]
}

/// Alpha-composite a heatmap color onto a base pixel.
fn blend_pixel(base: [u8; 3], overlay: [u8; 4]) -> [u8; 3] {
    let alpha = overlay[3] as f64 / 255.0;
    let inv_alpha = 1.0 - alpha;
    [
        (base[0] as f64 * inv_alpha + overlay[0] as f64 * alpha) as u8,
        (base[1] as f64 * inv_alpha + overlay[1] as f64 * alpha) as u8,
        (base[2] as f64 * inv_alpha + overlay[2] as f64 * alpha) as u8,
    ]
}

fn encode_png(image: &RgbImage) -> Option<Vec<u8>> {
    let mut buf = Cursor::new(Vec::new());
    image.write_to(&mut buf, ImageFormat::Png).ok()?;
    Some(buf.into_inner())
}

// ── renderers (from annotation/renderer.rs) ──────────────────────────────────

fn render_ela(ctx: &Context) -> Option<Vec<u8>> {
    let quality = 92_u8;
    let mut jpeg_buf = Cursor::new(Vec::new());
    let encoder = JpegEncoder::new_with_quality(&mut jpeg_buf, quality);
    ctx.image.write_with_encoder(encoder).ok()?;

    jpeg_buf.set_position(0);
    let resaved = ImageReader::with_format(jpeg_buf, ImageFormat::Jpeg)
        .decode()
        .ok()?;

    let (w, h) = ctx.image.dimensions();
    let mut output = ctx.image.to_rgb8();

    let mut max_error = 0.0_f64;
    let mut errors = vec![0.0_f64; (w * h) as usize];

    for y in 0..h {
        for x in 0..w {
            let orig = ctx.image.get_pixel(x, y).0;
            let resv = resaved.get_pixel(x, y).0;
            let err: f64 = (0..3)
                .map(|c| (orig[c] as f64 - resv[c] as f64).abs())
                .sum::<f64>()
                / 3.0;
            errors[(y * w + x) as usize] = err;
            if err > max_error {
                max_error = err;
            }
        }
    }

    if max_error < 1.0 {
        return None;
    }

    for y in 0..h {
        for x in 0..w {
            let err = errors[(y * w + x) as usize];
            let normalized = (err / max_error).min(1.0);
            if normalized > 0.1 {
                let color = value_to_color(normalized);
                let base = output.get_pixel(x, y).0;
                let blended = blend_pixel(base, color);
                output.put_pixel(x, y, Rgb(blended));
            }
        }
    }

    encode_png(&output)
}

fn render_noise_map(ctx: &Context) -> Option<Vec<u8>> {
    let w = ctx.width as usize;
    let h = ctx.height as usize;
    let block_size = 32;

    if w < block_size * 3 || h < block_size * 3 {
        return None;
    }

    let gray = ctx.gray();
    let blocks_x = w / block_size;
    let blocks_y = h / block_size;

    let mut block_vars: Vec<f64> = Vec::new();
    for by in 0..blocks_y {
        for bx in 0..blocks_x {
            let mut sum = 0.0_f64;
            let mut sum_sq = 0.0_f64;
            let count = (block_size * block_size) as f64;

            for dy in 0..block_size {
                for dx in 0..block_size {
                    let val = gray[(by * block_size + dy) * w + bx * block_size + dx] as f64;
                    sum += val;
                    sum_sq += val * val;
                }
            }

            let mean = sum / count;
            let var = (sum_sq / count) - (mean * mean);
            block_vars.push(var);
        }
    }

    let max_var = block_vars.iter().cloned().fold(0.0_f64, f64::max);
    if max_var < 1.0 {
        return None;
    }

    let mut output = ctx.image.to_rgb8();

    for by in 0..blocks_y {
        for bx in 0..blocks_x {
            let var = block_vars[by * blocks_x + bx];
            let normalized = (var / max_var).min(1.0);
            let color = value_to_color(normalized);

            for dy in 0..block_size {
                for dx in 0..block_size {
                    let x = (bx * block_size + dx) as u32;
                    let y = (by * block_size + dy) as u32;
                    if x < ctx.width && y < ctx.height {
                        let base = output.get_pixel(x, y).0;
                        let blended = blend_pixel(base, color);
                        output.put_pixel(x, y, Rgb(blended));
                    }
                }
            }
        }
    }

    encode_png(&output)
}

fn render_copy_move(ctx: &Context) -> Option<Vec<u8>> {
    let mut output = ctx.image.to_rgb8();
    let w = ctx.width;
    let h = ctx.height;
    if w < 4 || h < 4 {
        return None;
    }

    let border_color = Rgb([255_u8, 0, 0]);
    for x in 0..w {
        output.put_pixel(x, 0, border_color);
        output.put_pixel(x, 1, border_color);
        output.put_pixel(x, h - 1, border_color);
        output.put_pixel(x, h - 2, border_color);
    }
    for y in 0..h {
        output.put_pixel(0, y, border_color);
        output.put_pixel(1, y, border_color);
        output.put_pixel(w - 1, y, border_color);
        output.put_pixel(w - 2, y, border_color);
    }

    encode_png(&output)
}

fn render_cfa_map(ctx: &Context) -> Option<Vec<u8>> {
    let w = ctx.width as usize;
    let h = ctx.height as usize;
    let block_size = 64;

    if w < block_size * 3 || h < block_size * 3 {
        return None;
    }

    let rgb = ctx.image.to_rgb8();
    let pixels = rgb.as_raw();
    let blocks_x = w / block_size;
    let blocks_y = h / block_size;

    let mut strengths: Vec<f64> = Vec::new();
    for by in 0..blocks_y {
        for bx in 0..blocks_x {
            let mut even_sum = 0.0_f64;
            let mut odd_sum = 0.0_f64;
            let mut even_count = 0;
            let mut odd_count = 0;

            for dy in 1..block_size - 1 {
                for dx in 1..block_size - 1 {
                    let x = bx * block_size + dx;
                    let y = by * block_size + dy;
                    let idx = (y * w + x) * 3 + 1; // Green channel.
                    let center = pixels[idx] as f64;
                    let neighbors = (pixels[((y - 1) * w + x) * 3 + 1] as f64
                        + pixels[((y + 1) * w + x) * 3 + 1] as f64
                        + pixels[(y * w + x - 1) * 3 + 1] as f64
                        + pixels[(y * w + x + 1) * 3 + 1] as f64)
                        / 4.0;
                    let residual = (center - neighbors).abs();

                    if (x + y) % 2 == 0 {
                        even_sum += residual;
                        even_count += 1;
                    } else {
                        odd_sum += residual;
                        odd_count += 1;
                    }
                }
            }

            let strength = if even_count > 0 && odd_count > 0 {
                let even_avg = even_sum / even_count as f64;
                let odd_avg = odd_sum / odd_count as f64;
                (even_avg - odd_avg).abs()
            } else {
                0.0
            };
            strengths.push(strength);
        }
    }

    let max_strength = strengths.iter().cloned().fold(0.0_f64, f64::max);
    if max_strength < 0.01 {
        return None;
    }

    let mut output = ctx.image.to_rgb8();

    for by in 0..blocks_y {
        for bx in 0..blocks_x {
            let strength = strengths[by * blocks_x + bx];
            // Low CFA = suspicious -> shown hot.
            let normalized = 1.0 - (strength / max_strength).min(1.0);

            if normalized > 0.5 {
                let color = value_to_color(normalized);
                for dy in 0..block_size {
                    for dx in 0..block_size {
                        let x = (bx * block_size + dx) as u32;
                        let y = (by * block_size + dy) as u32;
                        if x < ctx.width && y < ctx.height {
                            let base = output.get_pixel(x, y).0;
                            let blended = blend_pixel(base, color);
                            output.put_pixel(x, y, Rgb(blended));
                        }
                    }
                }
            }
        }
    }

    encode_png(&output)
}

/// Signal-composite forensic heatmap: the top spatial findings, feathered and
/// normalized, with rectangle outlines for High/Critical regions.
fn render_forensic_composite(ctx: &Context, findings: &[Finding]) -> Option<Vec<u8>> {
    let w = ctx.width as usize;
    let h = ctx.height as usize;
    let total_pixels = (w * h) as f64;

    let mut spatial_findings: Vec<&Finding> = findings
        .iter()
        .filter(|f| {
            if f.severity < Severity::Medium {
                return false;
            }
            if f.code.contains("document_block_anomaly") {
                return false;
            }
            if let Some(Region::BoundingBox {
                width: bw,
                height: bh,
                ..
            }) = &f.region
            {
                let region_pixels = (*bw as f64) * (*bh as f64);
                region_pixels / total_pixels < 0.02
            } else {
                f.region.is_some()
            }
        })
        .collect();

    if spatial_findings.is_empty() {
        return None;
    }

    let mut heat_map = vec![0.0_f64; w * h];

    let severity_weight = |s: Severity| -> f64 {
        match s {
            Severity::Info => 0.1,
            Severity::Low => 0.3,
            Severity::Medium => 0.6,
            Severity::High => 0.9,
            Severity::Critical => 1.0,
        }
    };

    spatial_findings.sort_by(|a, b| {
        let wa = severity_weight(a.severity) * a.confidence;
        let wb = severity_weight(b.severity) * b.confidence;
        wb.partial_cmp(&wa).unwrap_or(std::cmp::Ordering::Equal)
    });
    spatial_findings.truncate(15);

    for finding in &spatial_findings {
        let weight = severity_weight(finding.severity) * finding.confidence;
        let Some(region) = finding.region.as_ref() else {
            continue;
        };

        match region {
            Region::BoundingBox {
                x,
                y,
                width: bw,
                height: bh,
            } => {
                let x0 = (*x as usize).min(w);
                let y0 = (*y as usize).min(h);
                let x1 = (x0 + *bw as usize).min(w);
                let y1 = (y0 + *bh as usize).min(h);

                let feather = 8.min((x1.saturating_sub(x0)).min(y1.saturating_sub(y0)) / 4);

                for py in y0..y1 {
                    for px in x0..x1 {
                        let dx = (px - x0).min(x1 - 1 - px);
                        let dy = (py - y0).min(y1 - 1 - py);
                        let edge_dist = dx.min(dy);
                        let feather_factor = if feather > 0 {
                            (edge_dist as f64 / feather as f64).min(1.0)
                        } else {
                            1.0
                        };
                        heat_map[py * w + px] += weight * feather_factor;
                    }
                }
            }
            Region::Points { points } => {
                let radius = 16_usize;
                for point in points {
                    let cx = point[0] as usize;
                    let cy = point[1] as usize;

                    let x0 = cx.saturating_sub(radius);
                    let y0 = cy.saturating_sub(radius);
                    let x1 = (cx + radius).min(w);
                    let y1 = (cy + radius).min(h);

                    for py in y0..y1 {
                        for px in x0..x1 {
                            let dx = px as f64 - cx as f64;
                            let dy = py as f64 - cy as f64;
                            let dist_sq = dx * dx + dy * dy;
                            let r_sq = (radius * radius) as f64;
                            if dist_sq < r_sq {
                                let falloff = 1.0 - dist_sq / r_sq;
                                heat_map[py * w + px] += weight * falloff;
                            }
                        }
                    }
                }
            }
            Region::Mask {
                width: mw,
                height: mh,
                data,
            } => {
                // paddock's Mask carries raw bytes (0/255), not base64.
                let mw = *mw as usize;
                let mh = *mh as usize;
                for my in 0..mh.min(h) {
                    for mx in 0..mw.min(w) {
                        let mask_idx = my * mw + mx;
                        if mask_idx < data.len() && data[mask_idx] > 127 {
                            heat_map[my * w + mx] += weight;
                        }
                    }
                }
            }
        }
    }

    let max_heat = heat_map.iter().cloned().fold(0.0_f64, f64::max);
    if max_heat < 0.01 {
        return None;
    }

    let mut output = ctx.image.to_rgb8();

    for y in 0..h {
        for x in 0..w {
            let heat = heat_map[y * w + x];
            if heat < 0.01 {
                continue;
            }
            let normalized = (heat / max_heat).min(1.0);
            let color = value_to_color(normalized);
            let base = output.get_pixel(x as u32, y as u32).0;
            let blended = blend_pixel(base, color);
            output.put_pixel(x as u32, y as u32, Rgb(blended));
        }
    }

    // Rectangle outlines for High/Critical (heat alone is invisible for small
    // regions on large pages).
    for finding in &spatial_findings {
        if finding.severity < Severity::High {
            continue;
        }
        if let Some(Region::BoundingBox {
            x,
            y,
            width: bw,
            height: bh,
        }) = &finding.region
        {
            let x0 = (*x as usize).min(w.saturating_sub(1));
            let y0 = (*y as usize).min(h.saturating_sub(1));
            let x1 = (x0 + *bw as usize).min(w.saturating_sub(1));
            let y1 = (y0 + *bh as usize).min(h.saturating_sub(1));

            let color = if finding.severity == Severity::Critical {
                [255, 0, 0]
            } else {
                [255, 140, 0]
            };
            let thickness = 3;

            for t in 0..thickness {
                for px in x0.saturating_sub(t)..=(x1 + t).min(w - 1) {
                    if y0 >= t && y0 - t < h {
                        output.put_pixel(px as u32, (y0 - t) as u32, Rgb(color));
                    }
                    if y1 + t < h {
                        output.put_pixel(px as u32, (y1 + t) as u32, Rgb(color));
                    }
                }
                for py in y0.saturating_sub(t)..=(y1 + t).min(h - 1) {
                    if x0 >= t && x0 - t < w {
                        output.put_pixel((x0 - t) as u32, py as u32, Rgb(color));
                    }
                    if x1 + t < w {
                        output.put_pixel((x1 + t) as u32, py as u32, Rgb(color));
                    }
                }
            }
        }
    }

    // Small finding-count badge in the top-left corner.
    let count = spatial_findings.len();
    let badge_w = 8 + count.to_string().len() * 7;
    let badge_h = 16_usize;
    for by in 2..badge_h.min(h) + 2 {
        for bx in 2..badge_w.min(w) + 2 {
            if bx < w && by < h {
                output.put_pixel(bx as u32, by as u32, Rgb([40, 40, 40]));
            }
        }
    }

    encode_png(&output)
}

/// Severity-colored border + darkened top strip.
fn render_overview(ctx: &Context, findings: &[Finding]) -> Option<Vec<u8>> {
    let mut output = ctx.image.to_rgb8();
    let (w, h) = (ctx.width, ctx.height);
    if w == 0 || h == 0 {
        return None;
    }

    let max_severity = findings
        .iter()
        .map(|f| f.severity)
        .max()
        .unwrap_or(Severity::Info);

    let border_color = match max_severity {
        Severity::Info => Rgb([100_u8, 200, 100]),
        Severity::Low => Rgb([200_u8, 200, 50]),
        Severity::Medium => Rgb([255_u8, 165, 0]),
        Severity::High => Rgb([255_u8, 80, 0]),
        Severity::Critical => Rgb([255_u8, 0, 0]),
    };

    let border = 4;
    for x in 0..w {
        for b in 0..border {
            if b < h {
                output.put_pixel(x, b, border_color);
                output.put_pixel(x, h - 1 - b, border_color);
            }
        }
    }
    for y in 0..h {
        for b in 0..border {
            if b < w {
                output.put_pixel(b, y, border_color);
                output.put_pixel(w - 1 - b, y, border_color);
            }
        }
    }

    let strip_height = 24.min(h);
    for y in 0..strip_height {
        for x in 0..w {
            let base = output.get_pixel(x, y).0;
            let darkened = [
                (base[0] as f64 * 0.3) as u8,
                (base[1] as f64 * 0.3) as u8,
                (base[2] as f64 * 0.3) as u8,
            ];
            output.put_pixel(x, y, Rgb(darkened));
        }
    }

    encode_png(&output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Finding, Region};
    use image::{ExtendedColorType, ImageEncoder};

    /// A gradient|checker PNG large enough for the block-based overlays.
    fn synth_png() -> Vec<u8> {
        let (w, h) = (256u32, 256u32);
        let mut rgb = vec![0u8; (w * h * 3) as usize];
        for y in 0..h {
            for x in 0..w {
                let i = ((y * w + x) * 3) as usize;
                let v: u8 = if x < 128 {
                    if (x + y) & 1 == 0 { 0 } else { 255 }
                } else {
                    (y * 255 / h) as u8
                };
                rgb[i] = v;
                rgb[i + 1] = v;
                rgb[i + 2] = v;
            }
        }
        let mut png = std::io::Cursor::new(Vec::new());
        image::codecs::png::PngEncoder::new(&mut png)
            .write_image(&rgb, w, h, ExtendedColorType::Rgb8)
            .unwrap();
        png.into_inner()
    }

    fn valid_png(bytes: &[u8]) -> bool {
        bytes.len() > 8 && &bytes[0..8] == b"\x89PNG\r\n\x1a\n"
    }

    #[test]
    fn renders_overlays_as_valid_pngs() {
        let ctx = Context::from_bytes(synth_png()).unwrap();
        let findings = vec![
            Finding::new(
                "copy_move",
                "copy_move_detected",
                "d",
                Severity::Critical,
                0.9,
            )
            .with_region(Region::BoundingBox {
                x: 10,
                y: 10,
                width: 20,
                height: 20,
            }),
            Finding::new("ela", "ela_block_outliers", "d", Severity::High, 0.8).with_region(
                Region::BoundingBox {
                    x: 40,
                    y: 40,
                    width: 16,
                    height: 16,
                },
            ),
        ];
        let set = render(&ctx, &findings);
        // overview + forensic composite always render on this synth; every
        // emitted overlay must be a valid PNG.
        assert!(set.images.contains_key("overview"));
        assert!(set.images.contains_key("forensic"));
        for (name, png) in &set.images {
            assert!(valid_png(png), "{name} is not a valid PNG");
        }
    }

    #[test]
    fn pdf_context_produces_no_overlays() {
        let ctx = Context::from_bytes(b"%PDF-1.4\ntrailer<<>>\n%%EOF".to_vec()).unwrap();
        let set = render(&ctx, &[]);
        assert!(set.images.is_empty(), "PDFs have no raster to overlay");
    }
}
