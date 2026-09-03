//! Multi-page TIFF -> per-page image parts (document semantics).
//!
//! A multi-page TIFF is a scanned document wearing an image extension: fax
//! archives, office scanners and "print to TIFF" all append one IFD per page.
//! The `image` crate - and every API vendor - decodes exactly IFD 0, so pages
//! 2+ used to vanish with no signal, against the no-silent-failures
//! principle. The fix mirrors the PDF raster path (`crate::pdf`): REPLACE the
//! single image part with a header note + `[page k]` markers + one PNG
//! `image_url` part per page, so the exact multi-image flow carries it and the
//! model knows what (and how much) it got. A single-page TIFF stays a plain
//! image part on the ordinary lane - nothing about it changes.
//!
//! Page decoding uses the `tiff` crate directly (the same one `image` is built
//! on): `Decoder::new` + the `more_images`/`next_image` IFD walk that image's
//! single-frame API hides. Color conversion covers the types that occur in real
//! scanned documents - bilevel (Gray 1, the fax class), Gray 8/16 ± alpha,
//! RGB/RGBA 8/16, CMYK 8 - and refuses the rest by name (planar layouts,
//! palette, floats), matching the honesty of the single-image lane rather than
//! guessing at channels. Runs only when the model can see: on a text-only
//! server image parts are already refused loudly up front, so nothing is
//! silently lost there either.

use std::io::Cursor;

use serde_json::{Value, json};
use tiff::ColorType;
use tiff::decoder::{Decoder, DecodingResult};

/// TIFF magic: classic II/MM + BigTIFF (43) variants - the decoder reads all
/// four. The BYTES decide, never the declared media type (same rule as
/// `chat::decode_image_url`; browsers routinely mislabel TIFF uploads).
fn is_tiff(bytes: &[u8]) -> bool {
    matches!(
        bytes,
        [0x49, 0x49, 0x2A, 0x00, ..]
            | [0x4D, 0x4D, 0x00, 0x2A, ..]
            | [0x49, 0x49, 0x2B, 0x00, ..]
            | [0x4D, 0x4D, 0x00, 0x2B, ..]
    )
}

/// Does this image part carry TIFF bytes? Decodes only the first 8 base64
/// chars (6 bytes) - enough for the magic - so the pre-check never pays a
/// full base64 pass.
fn part_is_tiff(part: &Value) -> bool {
    use base64::Engine as _;
    let Some(b64) = crate::doc::image_part_b64(part) else {
        return false;
    };
    let head = b64.trim_start();
    let Some(head) = head.get(..8) else {
        return false;
    };
    base64::engine::general_purpose::STANDARD
        .decode(head)
        .is_ok_and(|b| is_tiff(&b))
}

/// Cheap pre-check so a TIFF-less request never enters the blocking pass.
pub(crate) fn has_tiff_parts(messages: &[Value]) -> bool {
    messages.iter().any(|m| {
        m.get("content")
            .and_then(Value::as_array)
            .is_some_and(|parts| parts.iter().any(part_is_tiff))
    })
}

/// Page count of a TIFF, or `None` when the bytes aren't TIFF (or the header
/// walk fails - the real decode will state the real error). Walks IFDs only
/// (tag tables, no pixel data), so it is cheap enough for the Studio's
/// extract-preview panel.
pub(crate) fn page_count(bytes: &[u8]) -> Option<usize> {
    if !is_tiff(bytes) {
        return None;
    }
    let mut dec = Decoder::new(Cursor::new(bytes)).ok()?;
    let mut n = 1;
    while dec.more_images() {
        dec.next_image().ok()?;
        n += 1;
    }
    Some(n)
}

/// One decoded page, converted to the RGB8 everything downstream speaks.
/// Alpha is dropped uncomposited - the same treatment transparent PNGs have
/// always had on the single-image lane.
fn page_to_rgb(
    w: u32,
    h: u32,
    ct: ColorType,
    buf: DecodingResult,
) -> Result<image::RgbImage, String> {
    use image::DynamicImage as D;
    let px = (w as usize) * (h as usize);
    // from_raw's own too-small check guards every arm; exact sizing is the
    // decoder's contract
    let img = match (ct, buf) {
        // bilevel (fax/scan class): rows are MSB-first packed bits, one row per
        // div_ceil(w,8) bytes; WhiteIsZero was already inverted by the decoder
        (ColorType::Gray(1), DecodingResult::U8(v)) => {
            let row_bytes = (w as usize).div_ceil(8);
            if v.len() < row_bytes * h as usize {
                return Err("TIFF page buffer size mismatch".into());
            }
            let mut out = Vec::with_capacity(px);
            for row in v.chunks_exact(row_bytes).take(h as usize) {
                for x in 0..w as usize {
                    let bit = (row[x / 8] >> (7 - (x % 8))) & 1;
                    out.push(if bit != 0 { 0xFF } else { 0x00 });
                }
            }
            image::GrayImage::from_raw(w, h, out).map(D::ImageLuma8)
        }
        (ColorType::Gray(8), DecodingResult::U8(v)) => {
            image::GrayImage::from_raw(w, h, v).map(D::ImageLuma8)
        }
        (ColorType::Gray(16), DecodingResult::U16(v)) => {
            image::ImageBuffer::from_raw(w, h, v).map(D::ImageLuma16)
        }
        (ColorType::GrayA(8), DecodingResult::U8(v)) => {
            image::ImageBuffer::from_raw(w, h, v).map(D::ImageLumaA8)
        }
        (ColorType::GrayA(16), DecodingResult::U16(v)) => {
            image::ImageBuffer::from_raw(w, h, v).map(D::ImageLumaA16)
        }
        (ColorType::RGB(8), DecodingResult::U8(v)) => {
            image::RgbImage::from_raw(w, h, v).map(D::ImageRgb8)
        }
        (ColorType::RGB(16), DecodingResult::U16(v)) => {
            image::ImageBuffer::from_raw(w, h, v).map(D::ImageRgb16)
        }
        (ColorType::RGBA(8), DecodingResult::U8(v)) => {
            image::RgbaImage::from_raw(w, h, v).map(D::ImageRgba8)
        }
        (ColorType::RGBA(16), DecodingResult::U16(v)) => {
            image::ImageBuffer::from_raw(w, h, v).map(D::ImageRgba16)
        }
        (ColorType::CMYK(8), DecodingResult::U8(v)) => {
            if v.len() < px * 4 {
                return Err("TIFF page buffer size mismatch".into());
            }
            let mut out = Vec::with_capacity(px * 3);
            for cmyk in v.as_chunks::<4>().0.iter().take(px) {
                let k = 255 - cmyk[3] as u16;
                out.extend([
                    ((255 - cmyk[0] as u16) * k / 255) as u8,
                    ((255 - cmyk[1] as u16) * k / 255) as u8,
                    ((255 - cmyk[2] as u16) * k / 255) as u8,
                ]);
            }
            image::RgbImage::from_raw(w, h, out).map(D::ImageRgb8)
        }
        (ct, _) => {
            return Err(format!(
                "unsupported TIFF color type {ct:?} - convert the file to PNG or JPEG (one \
                 image per page) and resend"
            ));
        }
    }
    .ok_or_else(|| "TIFF page buffer size mismatch".to_string())?;
    Ok(img.to_rgb8())
}

struct TiffPages {
    pages: Vec<image::RgbImage>,
    total_pages: usize,
    /// 1-based number of the first decoded page (markers show real numbers).
    first_page: usize,
    /// True when `server_cap` cut the decode below what the caller's own
    /// selection asked for - same semantics as `RenderedPdf::ceiling_clipped`.
    ceiling_clipped: bool,
}

/// Decode the selected pages (skipped pages cost only an IFD tag read, never
/// a pixel decode), then keep walking so `total_pages` is the honest source
/// count (the disclosure needs it). `server_cap` bounds how many pages
/// decode whatever the selection asked. Errors name the page; a selection
/// starting past the end is a loud error, never an empty result.
fn decode_pages(
    bytes: &[u8],
    sel: crate::pdf::PageSel,
    server_cap: usize,
) -> Result<TiffPages, String> {
    use crate::pdf::PageSel;
    let mut dec = Decoder::new(Cursor::new(bytes)).map_err(|e| format!("TIFF decode: {e}"))?;
    // the selection can't be fully resolved before the walk (total unknown),
    // so resolve what's knowable now and check start > total at the end
    let (start, want_end) = match sel {
        PageSel::All => (1, usize::MAX),
        PageSel::First(n) => (1, n.max(1)),
        PageSel::Range(a, b) => (a, b),
    };
    let end = want_end.min(start.saturating_add(server_cap.max(1) - 1));
    let mut pages = Vec::new();
    let mut total = 0usize;
    loop {
        total += 1;
        if total >= start && total <= end {
            let at_page = |e: &dyn std::fmt::Display| format!("TIFF page {total}: {e}");
            let (w, h) = dec.dimensions().map_err(|e| at_page(&e))?;
            let ct = dec.colortype().map_err(|e| at_page(&e))?;
            let mut buf = DecodingResult::U8(Vec::new());
            let layout = dec
                .read_image_to_buffer(&mut buf)
                .map_err(|e| at_page(&e))?;
            if layout.planes > 1 {
                // read_image_to_buffer hands planar data back as separate
                // planes; interleaving every (colortype × depth) combination is
                // not worth it for a layout scanners never write - refuse by name
                return Err(at_page(
                    &"planar (PlanarConfiguration=2) TIFF is not supported - \
                                     convert to the standard interleaved layout",
                ));
            }
            pages.push(page_to_rgb(w, h, ct, buf).map_err(|e| at_page(&e))?);
        }
        if !dec.more_images() {
            break;
        }
        dec.next_image()
            .map_err(|e| format!("TIFF page {}: {e}", total + 1))?;
    }
    // total == 1 is exempt: a single-page TIFF is a plain picture and passes
    // through to the ordinary image lane untouched - a page range on it is
    // moot, not an error
    if start > total && total > 1 {
        return Err(format!(
            "the TIFF has only {total} pages - the requested pages start at {start}"
        ));
    }
    // the cap bound the walk below the selection's ask AND the file really
    // has pages past the cut - the same "server ceiling, not caller intent"
    // split as `RenderedPdf::ceiling_clipped`
    let ceiling_clipped = end < want_end && total > end;
    Ok(TiffPages {
        pages,
        total_pages: total,
        first_page: start.min(total),
        ceiling_clipped,
    })
}

/// Replace every MULTI-page TIFF image part with a header note + `[page k]`
/// markers + one PNG `image_url` part per decoded page (the caller's `detail`
/// carried onto each). Single-page TIFFs - and every non-TIFF part - pass
/// through untouched. The page cap resolves per PART: a flat `max_pages` key
/// on the image part wins, then the request-level extension, and the server's
/// own rendering ceiling (`server_cap`) bounds both; a cut is disclosed in
/// the note, never silent. **Blocking** (full base64 + page decode) - runs
/// under the same `spawn_blocking` as the rest of attachment expansion.
///
/// `plain_pages`: bare page images, no note or `[page k]` markers, and a
/// server-ceiling clip is a loud error - the deepseek2-ocr route, same
/// semantics as `crate::pdf::expand_in_messages`.
pub(crate) fn expand_in_messages(
    messages: &mut [Value],
    server_cap: usize,
    req_max: Option<usize>,
    summary: &mut crate::pdf::PdfSummary,
    plain_pages: bool,
) -> Result<(), String> {
    use base64::Engine as _;
    for msg in messages.iter_mut() {
        let Some(parts) = msg.get_mut("content").and_then(Value::as_array_mut) else {
            continue;
        };
        if !parts.iter().any(part_is_tiff) {
            continue;
        }
        let taken = std::mem::take(parts);
        let mut out = Vec::with_capacity(taken.len() + 4);
        for part in taken {
            if !part_is_tiff(&part) {
                out.push(part);
                continue;
            }
            let b64 = crate::doc::image_part_b64(&part).expect("part_is_tiff implies b64");
            let bytes = base64::engine::general_purpose::STANDARD
                .decode(b64.trim())
                .map_err(|e| format!("TIFF image base64: {e}"))?;
            // the caller's `detail` governs each replacement page exactly as it
            // governed the original part
            let detail = part
                .get("detail")
                .or_else(|| part.get("image_url").and_then(|v| v.get("detail")))
                .cloned();
            let sel = crate::pdf::part_page_sel(&part, req_max)?;
            let tiff = decode_pages(&bytes, sel, server_cap)?;
            if tiff.total_pages <= 1 {
                // a plain picture that happens to be TIFF - the ordinary
                // single-image lane decodes it; nothing to disclose
                out.push(part);
                continue;
            }
            let n = tiff.pages.len();
            summary.tiffs += 1;
            summary.total_pages += tiff.total_pages;
            summary.rendered_pages += n;
            summary.truncated |= n < tiff.total_pages;
            let (a, b) = (tiff.first_page, tiff.first_page + n.max(1) - 1);
            let total = tiff.total_pages;
            if plain_pages {
                if tiff.ceiling_clipped {
                    return Err(format!(
                        "the TIFF holds {total} pages but this server parses at most \
                         {server_cap} per request - select pages (e.g. \"pages\": \
                         \"1-{server_cap}\") and send the rest separately, or raise the \
                         server's page ceiling"
                    ));
                }
                for page in tiff.pages {
                    let mut img_part = crate::pdf::png_image_part(page)?;
                    if let Some(d) = &detail {
                        img_part["image_url"]["detail"] = d.clone();
                    }
                    out.push(img_part);
                }
                continue;
            }
            let mut note = format!("[Attached TIFF - {total} pages]");
            if n < total {
                // loud, never silent: the model is told which pages it got
                note.push_str(&if a == b {
                    format!("\n[Only page {a} of {total} is shown below.]")
                } else {
                    format!("\n[Only pages {a}-{b} of {total} are shown below.]")
                });
            }
            out.push(json!({"type": "text", "text": note}));
            for (i, page) in tiff.pages.into_iter().enumerate() {
                out.push(json!({"type": "text", "text": format!("[page {}]", a + i)}));
                let mut img_part = crate::pdf::png_image_part(page)?;
                if let Some(d) = &detail {
                    img_part["image_url"]["detail"] = d.clone();
                }
                out.push(img_part);
            }
        }
        if let Some(parts) = msg.get_mut("content").and_then(Value::as_array_mut) {
            *parts = out;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::Engine as _;
    use tiff::encoder::{TiffEncoder, colortype};

    /// n RGB8 pages (each a distinct solid color) in one TIFF, via the tiff
    /// crate's own encoder - `write_image` appends one IFD per call.
    fn rgb_tiff(n: usize, w: u32, h: u32) -> Vec<u8> {
        let mut out = Cursor::new(Vec::new());
        let mut enc = TiffEncoder::new(&mut out).expect("encoder");
        for i in 0..n {
            let px = [40 * (i as u8 + 1), 10, 200 - 40 * (i as u8)];
            let data: Vec<u8> = px
                .iter()
                .copied()
                .cycle()
                .take((w * h * 3) as usize)
                .collect();
            enc.write_image::<colortype::RGB8>(w, h, &data)
                .expect("page");
        }
        drop(enc);
        out.into_inner()
    }

    /// A hand-rolled minimal bilevel (Gray 1) TIFF, 8×2, uncompressed,
    /// BlackIsZero, rows `11110000` / `00001111` - the fax/scanner class the
    /// encoder can't write. Little-endian, IFD after the 2 data bytes.
    fn bilevel_tiff() -> Vec<u8> {
        let mut b: Vec<u8> = vec![0x49, 0x49, 0x2A, 0x00, 12, 0, 0, 0]; // II*, IFD @ 12
        b.extend([0b1111_0000, 0b0000_1111, 0, 0]); // strip data @ 8 (+2 pad)
        let entry = |tag: u16, ty: u16, val: u32| {
            let mut e = Vec::with_capacity(12);
            e.extend(tag.to_le_bytes());
            e.extend(ty.to_le_bytes());
            e.extend(1u32.to_le_bytes());
            e.extend(val.to_le_bytes());
            e
        };
        const SHORT: u16 = 3;
        const LONG: u16 = 4;
        let entries = [
            entry(256, SHORT, 8), // ImageWidth
            entry(257, SHORT, 2), // ImageLength
            entry(258, SHORT, 1), // BitsPerSample
            entry(259, SHORT, 1), // Compression: none
            entry(262, SHORT, 1), // Photometric: BlackIsZero
            entry(273, LONG, 8),  // StripOffsets
            entry(278, SHORT, 2), // RowsPerStrip
            entry(279, LONG, 2),  // StripByteCounts
        ];
        b.extend((entries.len() as u16).to_le_bytes());
        for e in &entries {
            b.extend_from_slice(e);
        }
        b.extend(0u32.to_le_bytes()); // no next IFD
        b
    }

    fn image_part(bytes: &[u8]) -> Value {
        let b64 = base64::engine::general_purpose::STANDARD.encode(bytes);
        json!({"type": "image_url",
               "image_url": {"url": format!("data:image/tiff;base64,{b64}"), "detail": "high"}})
    }

    #[test]
    fn sniff_finds_tiff_parts_and_ignores_the_rest() {
        let msgs = vec![json!({"role":"user","content":[image_part(&rgb_tiff(1, 2, 2))]})];
        assert!(has_tiff_parts(&msgs));
        // PNG magic -> not TIFF, whatever the media type claims
        let png = json!({"role":"user","content":[{"type":"image_url","image_url":
            {"url": format!("data:image/tiff;base64,{}",
             base64::engine::general_purpose::STANDARD.encode([0x89, b'P', b'N', b'G', 13, 10, 26, 10]))}}]});
        assert!(!has_tiff_parts(&[png]));
    }

    #[test]
    fn page_count_walks_ifds() {
        assert_eq!(page_count(&rgb_tiff(3, 2, 2)), Some(3));
        assert_eq!(page_count(&rgb_tiff(1, 2, 2)), Some(1));
        assert_eq!(page_count(b"\x89PNG"), None);
    }

    #[test]
    fn multi_page_expands_with_truncation_note_and_detail() {
        let mut msgs = vec![json!({"role":"user","content":[
            {"type":"text","text":"read this scan"},
            image_part(&rgb_tiff(3, 4, 2)),
        ]})];
        let mut summary = crate::pdf::PdfSummary::default();
        expand_in_messages(&mut msgs, 2, None, &mut summary, false).expect("expand");
        assert_eq!(summary.tiffs, 1);
        assert_eq!(summary.total_pages, 3);
        assert_eq!(summary.rendered_pages, 2);
        assert!(summary.truncated);

        let parts = msgs[0]["content"].as_array().unwrap();
        // text + note + 2×([page k] + image) = 6
        assert_eq!(parts.len(), 6, "parts: {parts:#?}");
        assert!(parts[1]["text"].as_str().unwrap().contains("3 pages"));
        assert!(
            parts[1]["text"]
                .as_str()
                .unwrap()
                .contains("Only pages 1-2")
        );
        let pages: Vec<&Value> = parts
            .iter()
            .filter(|p| p.get("image_url").is_some())
            .collect();
        assert_eq!(pages.len(), 2);
        for p in &pages {
            let url = p["image_url"]["url"].as_str().unwrap();
            assert!(
                url.starts_with("data:image/png;base64,"),
                "pages re-encode as PNG"
            );
            assert_eq!(
                p["image_url"]["detail"], "high",
                "caller detail carried per page"
            );
        }
        // page 1 and page 2 carry different pixels (each page really decoded,
        // not IFD 0 repeated) - decode and compare
        let px = |v: &Value| {
            let b64 = v["image_url"]["url"]
                .as_str()
                .unwrap()
                .split_once(',')
                .unwrap()
                .1;
            let bytes = base64::engine::general_purpose::STANDARD
                .decode(b64)
                .unwrap();
            image::load_from_memory(&bytes)
                .unwrap()
                .to_rgb8()
                .into_raw()
        };
        assert_ne!(px(pages[0]), px(pages[1]));
    }

    /// The deepseek2-ocr route: bare page images with the caller's `detail`
    /// carried, no note or markers; a server-ceiling clip is a loud error
    /// (a user selection stays quiet) - mirror of the PDF plain route.
    #[test]
    fn plain_pages_emits_bare_images_and_refuses_the_ceiling_clip() {
        let mut msgs = vec![json!({"role":"user","content":[image_part(&rgb_tiff(3, 4, 2))]})];
        let mut summary = crate::pdf::PdfSummary::default();
        expand_in_messages(&mut msgs, 8, None, &mut summary, true).expect("expand");
        let parts = msgs[0]["content"].as_array().unwrap();
        assert_eq!(parts.len(), 3, "three bare page images: {parts:#?}");
        assert!(
            parts.iter().all(|p| p.get("image_url").is_some()),
            "no text parts at all"
        );
        assert_eq!(
            parts[0]["image_url"]["detail"], "high",
            "caller detail still carried"
        );

        // user selection: intent, not truncation
        let mut part = image_part(&rgb_tiff(3, 4, 2));
        part["pages"] = json!("2-3");
        let mut msgs = vec![json!({"role":"user","content":[part]})];
        expand_in_messages(
            &mut msgs,
            8,
            None,
            &mut crate::pdf::PdfSummary::default(),
            true,
        )
        .expect("expand");
        assert_eq!(
            msgs[0]["content"].as_array().unwrap().len(),
            2,
            "pages 2-3 only"
        );

        // server ceiling below the ask: loud error naming the cap
        let mut msgs = vec![json!({"role":"user","content":[image_part(&rgb_tiff(3, 4, 2))]})];
        let err = expand_in_messages(
            &mut msgs,
            2,
            None,
            &mut crate::pdf::PdfSummary::default(),
            true,
        )
        .unwrap_err();
        assert!(err.contains("at most 2"), "{err}");
    }

    #[test]
    fn single_page_tiff_passes_through_untouched() {
        let part = image_part(&rgb_tiff(1, 2, 2));
        let mut msgs = vec![json!({"role":"user","content":[part.clone()]})];
        let mut summary = crate::pdf::PdfSummary::default();
        expand_in_messages(&mut msgs, 8, None, &mut summary, false).expect("expand");
        assert_eq!(
            msgs[0]["content"],
            json!([part]),
            "plain picture stays on the image lane"
        );
        assert!(!summary.any());
    }

    /// A flat `max_pages` on the image part itself caps this file, beating
    /// the request-level extension (the setting belongs to the file).
    #[test]
    fn part_level_max_pages_caps_this_tiff() {
        let mut part = image_part(&rgb_tiff(3, 4, 2));
        part["max_pages"] = json!(1);
        let mut msgs = vec![json!({"role":"user","content":[part]})];
        let mut summary = crate::pdf::PdfSummary::default();
        expand_in_messages(&mut msgs, 8, Some(2), &mut summary, false).expect("expand");
        assert_eq!(
            summary.rendered_pages, 1,
            "the part's own cap wins over the request's 2"
        );
        assert!(summary.truncated);
        let note = msgs[0]["content"][0]["text"].as_str().unwrap();
        assert!(note.contains("Only page 1 of 3"), "{note}");
    }

    /// A `pages` RANGE on the part decodes only those IFDs and keeps the real
    /// page numbers on the markers; a start past the end is a loud error.
    #[test]
    fn part_level_pages_range_selects_the_middle() {
        let mut part = image_part(&rgb_tiff(3, 4, 2));
        part["pages"] = json!("2-3");
        let mut msgs = vec![json!({"role":"user","content":[part]})];
        let mut summary = crate::pdf::PdfSummary::default();
        expand_in_messages(&mut msgs, 8, None, &mut summary, false).expect("expand");
        assert_eq!(summary.rendered_pages, 2);
        let parts = msgs[0]["content"].as_array().unwrap();
        assert!(
            parts[0]["text"]
                .as_str()
                .unwrap()
                .contains("Only pages 2-3 of 3")
        );
        assert_eq!(
            parts[1]["text"], "[page 2]",
            "real page numbers: {parts:#?}"
        );
        assert_eq!(parts[3]["text"], "[page 3]");

        let mut part = image_part(&rgb_tiff(3, 4, 2));
        part["pages"] = json!("7-9");
        let mut msgs = vec![json!({"role":"user","content":[part]})];
        let err = expand_in_messages(
            &mut msgs,
            8,
            None,
            &mut crate::pdf::PdfSummary::default(),
            false,
        )
        .unwrap_err();
        assert!(err.contains("only 3 pages"), "{err}");
    }

    /// The bilevel conversion agrees with the `image` crate's own L1 decode of
    /// the identical bytes - the reference for bit order and expansion.
    #[test]
    fn bilevel_page_matches_image_crate_reference() {
        let bytes = bilevel_tiff();
        let reference = image::load_from_memory(&bytes)
            .expect("image crate decodes it")
            .to_rgb8();
        let ours = decode_pages(&bytes, crate::pdf::PageSel::All, 4).expect("decode");
        assert_eq!(ours.total_pages, 1);
        assert_eq!(ours.pages[0].dimensions(), (8, 2));
        assert_eq!(ours.pages[0].as_raw(), reference.as_raw());
        // and the pattern itself: row 0 = 4 white then 4 black
        let p = &ours.pages[0];
        assert_eq!(p.get_pixel(0, 0).0, [255, 255, 255]);
        assert_eq!(p.get_pixel(7, 0).0, [0, 0, 0]);
        assert_eq!(p.get_pixel(0, 1).0, [0, 0, 0]);
        assert_eq!(p.get_pixel(7, 1).0, [255, 255, 255]);
    }
}
