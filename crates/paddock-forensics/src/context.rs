//! The shared analysis context: original bytes + decoded pixels + metadata.
//!
//! Forensics is byte-exact, so a [`Context`] keeps the *original* upload bytes
//! (`raw_bytes`) alongside the decoded image and the sift-derived metadata
//! (tags, EXIF thumbnail). It mirrors the reference's analysis context so the ported
//! analyzers see the same inputs: cached grayscale, quantization-relevant raw
//! bytes, EXIF/XMP tags, the embedded thumbnail, and a classified content type.

use std::io::Cursor;

use image::{DynamicImage, ImageFormat, ImageReader};

use crate::error::ContextError;

/// PDF magic (`%PDF-`).
fn is_pdf_bytes(b: &[u8]) -> bool {
    b.len() >= 5 && &b[0..5] == b"%PDF-"
}

/// HEIC/HEIF/AVIF detection: an ISO-BMFF `ftyp` box with a HEIF-family brand.
/// The `image` crate cannot decode these; the `heic` feature adds a libheif
/// path, otherwise HEIC input is an honest decode error.
fn is_heic_bytes(b: &[u8]) -> bool {
    b.len() >= 12
        && &b[4..8] == b"ftyp"
        && (b[8..12].starts_with(b"heic")
            || b[8..12].starts_with(b"heix")
            || b[8..12].starts_with(b"mif1")
            || b[8..12].starts_with(b"msf1")
            || b[8..12].starts_with(b"avif"))
}

/// Decode HEIC/HEIF/AVIF via libheif (feature `heic`). Ported from the
/// reference's libheif decoder; the Python pillow-heif fallback is deliberately not
/// ported (no Python in paddock) - a strict-libheif reject falls through as a
/// decode error rather than a subprocess.
#[cfg(feature = "heic")]
fn decode_heic(bytes: &[u8]) -> Result<DynamicImage, ContextError> {
    use libheif_rs::{ColorSpace, HeifContext, RgbChroma};

    let heif_err = |e: libheif_rs::HeifError| ContextError::Decode(format!("HEIC: {e}"));

    let ctx = HeifContext::read_from_bytes(bytes).map_err(heif_err)?;
    let handle = ctx.primary_image_handle().map_err(heif_err)?;
    let w = handle.width();
    let h = handle.height();

    let decoded = handle
        .decode(ColorSpace::Rgb(RgbChroma::Rgb), None)
        .map_err(heif_err)?;
    let plane = decoded
        .planes()
        .interleaved
        .ok_or_else(|| ContextError::Decode("HEIC: no interleaved plane".to_string()))?;

    let stride = plane.stride;
    let mut rgb_data = Vec::with_capacity((w as usize) * (h as usize) * 3);
    for y in 0..h as usize {
        let row_start = y * stride;
        let row_end = row_start + (w as usize * 3);
        rgb_data.extend_from_slice(&plane.data[row_start..row_end]);
    }

    let rgb_img = image::RgbImage::from_raw(w, h, rgb_data)
        .ok_or_else(|| ContextError::Decode("HEIC: invalid RGB buffer".to_string()))?;
    Ok(DynamicImage::ImageRgb8(rgb_img))
}

/// Content classification (ported from the reference's weighted classifier). Many
/// analyzers gate on this - document-specific ones (text alignment, font
/// consistency) run only on Document/Mixed; camera/sensor ones skip documents.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContentType {
    Photo,
    Document,
    Mixed,
    Unknown,
}

/// Decoded image + original bytes + sift metadata, at reference parity.
pub struct Context {
    /// The original, undecoded upload bytes. Never re-encoded.
    pub raw_bytes: Vec<u8>,
    /// Decoded pixels (a 1×1 placeholder for PDFs, which have no single image).
    pub image: DynamicImage,
    /// Row-major 8-bit luminance of `image` (cached; empty for PDFs).
    pub gray: Vec<u8>,
    pub width: u32,
    pub height: u32,
    /// Decoded raster format, when known (drives content classification).
    pub format: Option<ImageFormat>,
    /// EXIF/XMP/IPTC/PDF metadata tags (via sift).
    pub tags: Vec<sift::Tag>,
    /// Embedded EXIF thumbnail bytes, if present (for thumbnail-mismatch checks).
    pub thumbnail_bytes: Option<Vec<u8>>,
    pub content_type: ContentType,
}

impl Context {
    /// Decode raster bytes (JPEG/PNG/WebP/GIF/TIFF/BMP) or accept a PDF, building
    /// the full analysis context. [`ContextError::Decode`] for raster formats this
    /// crate cannot decode (HEIC has a dedicated lane in a later wave).
    pub fn from_bytes(bytes: impl Into<Vec<u8>>) -> Result<Self, ContextError> {
        let raw_bytes = bytes.into();

        // PDFs carry no single decoded image - the document-forensics analyzers
        // work off `raw_bytes` (via sift); pixel analyzers gate `applies_to =
        // !is_pdf`. A 1×1 placeholder keeps `image` non-optional (the reference's shape).
        if is_pdf_bytes(&raw_bytes) {
            let tags = sift::read(&raw_bytes).map(|d| d.tags()).unwrap_or_default();
            return Ok(Self {
                raw_bytes,
                image: DynamicImage::new_rgb8(1, 1),
                gray: Vec::new(),
                width: 1,
                height: 1,
                format: None,
                tags,
                thumbnail_bytes: None,
                content_type: ContentType::Document,
            });
        }

        // HEIC/HEIF/AVIF: the `image` crate can't decode these - take the
        // libheif path when the `heic` feature is on, else an honest error.
        let (image, format): (DynamicImage, Option<ImageFormat>) = if is_heic_bytes(&raw_bytes) {
            #[cfg(feature = "heic")]
            {
                (decode_heic(&raw_bytes)?, None)
            }
            #[cfg(not(feature = "heic"))]
            {
                return Err(ContextError::Decode(
                    "HEIC/HEIF input requires the `heic` cargo feature (libheif)".to_string(),
                ));
            }
        } else {
            let reader = ImageReader::new(Cursor::new(&raw_bytes))
                .with_guessed_format()
                .map_err(|e| ContextError::Decode(e.to_string()))?;
            let format = reader.format();
            let image = reader
                .decode()
                .map_err(|e| ContextError::Decode(e.to_string()))?;
            (image, format)
        };
        let width = image.width();
        let height = image.height();
        if width == 0 || height == 0 {
            return Err(ContextError::TooSmall { width, height });
        }

        let gray = image.to_luma8().into_raw();
        // sift metadata is best-effort enrichment - a file with no readable EXIF
        // still analyzes; it just classifies from pixels alone.
        let (tags, thumbnail_bytes) = match sift::read(&raw_bytes) {
            Ok(doc) => (doc.tags(), doc.thumbnail()),
            Err(_) => (Vec::new(), None),
        };
        let content_type = classify_content(&tags, format, &gray, width, height);

        Ok(Self {
            raw_bytes,
            image,
            gray,
            width,
            height,
            format,
            tags,
            thumbnail_bytes,
            content_type,
        })
    }

    /// Whether the ORIGINAL bytes are a PDF (`%PDF-`).
    pub fn is_pdf(&self) -> bool {
        is_pdf_bytes(&self.raw_bytes)
    }

    /// Whether the ORIGINAL bytes are a JPEG (SOI marker).
    pub fn is_jpeg(&self) -> bool {
        self.raw_bytes.len() >= 3
            && self.raw_bytes[0] == 0xFF
            && self.raw_bytes[1] == 0xD8
            && self.raw_bytes[2] == 0xFF
    }

    /// Cached row-major 8-bit luminance of the decoded image.
    pub fn gray(&self) -> &[u8] {
        &self.gray
    }
}

/// Weighted content-type classifier, ported from the reference: metadata (EXIF
/// camera vs software vs PDF), format, histogram bimodality, and horizontal
/// projection profile -> Photo / Document / Unknown.
fn classify_content(
    tags: &[sift::Tag],
    format: Option<ImageFormat>,
    gray: &[u8],
    width: u32,
    height: u32,
) -> ContentType {
    let (w, h) = (width as usize, height as usize);

    // Signal 1: metadata (-1 photo ... +1 document)
    let has_camera = tags.iter().any(|t| {
        t.group == "EXIF"
            && matches!(
                t.name.as_str(),
                "Make" | "Model" | "LensModel" | "FocalLength"
            )
    });
    let has_software = tags
        .iter()
        .any(|t| t.group == "EXIF" && t.name == "Software");
    let has_pdf_tags = tags.iter().any(|t| t.group == "PDF");
    let metadata_score = if has_pdf_tags {
        1.0
    } else if has_camera && !has_software {
        -1.0
    } else if has_camera && has_software {
        -0.3
    } else if has_software {
        0.5
    } else {
        0.3
    };

    // Signal 2: format
    let format_score = match format {
        Some(ImageFormat::Png) => 0.5,
        Some(ImageFormat::Bmp) => 0.3,
        Some(ImageFormat::Jpeg) if has_camera => -0.5,
        Some(ImageFormat::Jpeg) => 0.0,
        _ => 0.0,
    };

    // Signal 3: histogram bimodality
    let mut histogram = [0u32; 256];
    for &p in gray {
        histogram[p as usize] += 1;
    }
    let total = gray.len().max(1) as f64;
    let near_white = histogram[230..].iter().sum::<u32>() as f64 / total;
    let near_black = histogram[..25].iter().sum::<u32>() as f64 / total;
    let background_ratio = near_white + near_black;
    let peak_dark = histogram[..80].iter().max().copied().unwrap_or(0);
    let peak_light = histogram[180..].iter().max().copied().unwrap_or(0);
    let valley = histogram[80..180].iter().min().copied().unwrap_or(u32::MAX);
    let histogram_score = if peak_dark > 0 && peak_light > 0 {
        let min_peak = peak_dark.min(peak_light);
        if min_peak > 0 && valley < min_peak / 2 {
            1.0
        } else if background_ratio > 0.5 {
            0.6
        } else {
            0.0
        }
    } else if background_ratio > 0.6 {
        0.7
    } else {
        0.0
    };

    // Signal 4: horizontal projection profile (comb pattern of text lines)
    let projection_score = if h > 20 && w > 20 {
        let mut row_means = vec![0.0_f64; h];
        for (y, rm) in row_means.iter_mut().enumerate() {
            let sum: f64 = (0..w).map(|x| gray[y * w + x] as f64).sum();
            *rm = sum / w as f64;
        }
        let diffs: Vec<f64> = (1..h)
            .map(|y| (row_means[y] - row_means[y - 1]).abs())
            .collect();
        if diffs.is_empty() {
            0.0
        } else {
            let mean_diff: f64 = diffs.iter().sum::<f64>() / diffs.len() as f64;
            let threshold = mean_diff * 2.0;
            let sharp = diffs.iter().filter(|&&d| d > threshold).count();
            let ratio = sharp as f64 / h as f64;
            if ratio > 0.05 {
                ((ratio - 0.05) / 0.15).min(1.0)
            } else {
                0.0
            }
        }
    } else {
        0.0
    };

    let doc_score = metadata_score * 0.40
        + format_score * 0.20
        + histogram_score * 0.20
        + projection_score * 0.20;

    if doc_score > 0.35 {
        ContentType::Document
    } else if doc_score < -0.15 {
        ContentType::Photo
    } else {
        ContentType::Unknown
    }
}
