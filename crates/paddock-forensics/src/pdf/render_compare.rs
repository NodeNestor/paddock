//! Render-vs-scan comparison: the strongest PDF-overlay signal. Render the full
//! page (scan + any overlaid text/graphics) and diff it against the embedded
//! scan image - any difference is content added on top of the scan (the classic
//! "type new digits over a scanned receipt" fraud). Ported from the reference
//! implementation.
//!
//! Page rendering needs pdfium, which is process-global (one binding per
//! process, owned by the runner). So this does not bind pdfium itself - it takes
//! a [`PageRenderer`] the caller supplies (the runner wraps its own pdfium;
//! tests wrap a dedicated one). That keeps the single-Pdfium invariant and the
//! forensics crate free of a pdfium runtime dependency.

use image::{DynamicImage, RgbImage};

use crate::{Finding, Region, Severity};

/// Renders a PDF page to pixels. Implemented by whoever owns pdfium.
pub trait PageRenderer {
    /// Render 0-based `page` at approximately `dpi`, returning interleaved RGB8
    /// plus its dimensions, or `None` if the page cannot be rendered.
    fn render_page(&self, pdf_bytes: &[u8], page: u32, dpi: f32) -> Option<(Vec<u8>, u32, u32)>;
}

/// Tunables (the reference's defaults).
pub struct RenderCompareOpts {
    pub render_dpi: f32,
    /// Per-pixel mean-channel difference (0-255) to count as modified.
    pub diff_threshold: u8,
    /// Minimum fraction of the page that must differ to report.
    pub min_diff_ratio: f64,
    /// Max pages compared (cost guard).
    pub max_pages: u32,
}

impl Default for RenderCompareOpts {
    fn default() -> Self {
        Self {
            render_dpi: 150.0,
            diff_threshold: 10,
            min_diff_ratio: 0.0001,
            max_pages: 10,
        }
    }
}

/// Compare each page's rendered output against its embedded scan. Returns
/// findings (with spatial regions) for pages carrying overlaid content.
pub fn render_compare(
    pdf_bytes: &[u8],
    renderer: &dyn PageRenderer,
    opts: &RenderCompareOpts,
) -> Vec<Finding> {
    let doc = match sift::read(pdf_bytes) {
        Ok(d) => d,
        Err(_) => return Vec::new(),
    };
    let images = doc.images().unwrap_or_default();
    if images.is_empty() {
        return Vec::new();
    }

    let mut findings = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let mut processed = 0u32;

    for page_num in images.iter().map(|i| i.page) {
        if !seen.insert(page_num) {
            continue;
        }
        processed += 1;
        if processed > opts.max_pages {
            findings.push(Finding::new(
                "pdf_render_compare",
                "pdf_render_compare_limit",
                format!(
                    "PDF has more than {0} pages with images - render comparison limited to first {0}",
                    opts.max_pages
                ),
                Severity::Info,
                1.0,
            ));
            break;
        }

        let Some((rgb, rw, rh)) = renderer.render_page(pdf_bytes, page_num, opts.render_dpi) else {
            continue;
        };
        let Some(rendered) = RgbImage::from_raw(rw, rh, rgb) else {
            continue;
        };

        // Largest embedded image on the page = the scan.
        let Some(scan) = images
            .iter()
            .filter(|i| i.page == page_num)
            .max_by_key(|i| i.width as u64 * i.height as u64)
        else {
            continue;
        };
        if scan.width < 200 || scan.height < 200 {
            continue;
        }
        let Some(scan_img) = decode_sift_image(scan) else {
            continue;
        };
        let scan_resized = image::imageops::resize(
            &scan_img.to_rgb8(),
            rw,
            rh,
            image::imageops::FilterType::Lanczos3,
        );

        let total = rw as u64 * rh as u64;
        let mut diff_pixels = 0u64;
        let mut diff_map = vec![false; (rw * rh) as usize];
        for y in 0..rh {
            for x in 0..rw {
                let pr = rendered.get_pixel(x, y).0;
                let ps = scan_resized.get_pixel(x, y).0;
                let dr = (pr[0] as i32 - ps[0] as i32).unsigned_abs();
                let dg = (pr[1] as i32 - ps[1] as i32).unsigned_abs();
                let db = (pr[2] as i32 - ps[2] as i32).unsigned_abs();
                if (dr + dg + db) / 3 > opts.diff_threshold as u32 {
                    diff_pixels += 1;
                    diff_map[(y * rw + x) as usize] = true;
                }
            }
        }
        let diff_ratio = diff_pixels as f64 / total as f64;
        if diff_ratio < opts.min_diff_ratio {
            continue;
        }

        let regions = find_diff_regions(&diff_map, rw, rh);
        findings.push(Finding::new(
            "pdf_render_compare",
            "pdf_overlay_content_detected",
            format!(
                "Page {} (scan {}x{}): {:.2}% of pixels differ between embedded scan and rendered \
                 page - content was added on top of the scanned image ({} modified region{})",
                page_num + 1,
                scan.width,
                scan.height,
                diff_ratio * 100.0,
                regions.len(),
                if regions.len() == 1 { "" } else { "s" }
            ),
            Severity::Critical,
            0.92,
        ));

        // Map diff-region boxes back to original scan coordinates.
        let sx = scan.width as f64 / rw as f64;
        let sy = scan.height as f64 / rh as f64;
        for (i, r) in regions.iter().take(10).enumerate() {
            let x = (r.x_min as f64 * sx) as u32;
            let y = (r.y_min as f64 * sy) as u32;
            let w = ((r.x_max - r.x_min + 1) as f64 * sx) as u32;
            let h = ((r.y_max - r.y_min + 1) as f64 * sy) as u32;
            findings.push(
                Finding::new(
                    "pdf_render_compare",
                    "pdf_overlay_region",
                    format!(
                        "Page {}: overlay region {} at ({x},{y}) - {} pixels added on top of scan",
                        page_num + 1,
                        i + 1,
                        r.pixel_count
                    ),
                    Severity::Critical,
                    0.90,
                )
                .with_region(Region::BoundingBox {
                    x,
                    y,
                    width: w,
                    height: h,
                }),
            );
        }
    }
    findings
}

struct DiffRegion {
    x_min: u32,
    y_min: u32,
    x_max: u32,
    y_max: u32,
    pixel_count: u32,
}

/// Connected-component bounding boxes over the diff mask (4-connectivity),
/// tiny (<20px) regions dropped, sorted largest-first.
fn find_diff_regions(diff_map: &[bool], w: u32, h: u32) -> Vec<DiffRegion> {
    let mut labels = vec![false; diff_map.len()];
    let mut regions = Vec::new();
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) as usize;
            if !diff_map[idx] || labels[idx] {
                continue;
            }
            let mut queue = vec![(x, y)];
            labels[idx] = true;
            let mut r = DiffRegion {
                x_min: x,
                y_min: y,
                x_max: x,
                y_max: y,
                pixel_count: 0,
            };
            while let Some((cx, cy)) = queue.pop() {
                r.x_min = r.x_min.min(cx);
                r.y_min = r.y_min.min(cy);
                r.x_max = r.x_max.max(cx);
                r.y_max = r.y_max.max(cy);
                r.pixel_count += 1;
                for &(dx, dy) in &[(1i32, 0), (-1, 0), (0, 1), (0, -1)] {
                    let nx = cx as i32 + dx;
                    let ny = cy as i32 + dy;
                    if nx >= 0 && ny >= 0 && (nx as u32) < w && (ny as u32) < h {
                        let ni = (ny as u32 * w + nx as u32) as usize;
                        if diff_map[ni] && !labels[ni] {
                            labels[ni] = true;
                            queue.push((nx as u32, ny as u32));
                        }
                    }
                }
            }
            if r.pixel_count >= 20 {
                regions.push(r);
            }
        }
    }
    regions.sort_by(|a, b| b.pixel_count.cmp(&a.pixel_count));
    regions
}

fn decode_sift_image(image: &sift::Image) -> Option<DynamicImage> {
    match &image.data {
        sift::ImageData::Jpeg(b) | sift::ImageData::Jpeg2000(b) => {
            image::ImageReader::new(std::io::Cursor::new(b))
                .with_guessed_format()
                .ok()?
                .decode()
                .ok()
        }
        sift::ImageData::Pixels(b) => match image.components {
            1 => image::GrayImage::from_raw(image.width, image.height, b.to_vec())
                .map(DynamicImage::ImageLuma8),
            3 => RgbImage::from_raw(image.width, image.height, b.to_vec())
                .map(DynamicImage::ImageRgb8),
            _ => None,
        },
        _ => None,
    }
}
