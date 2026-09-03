//! Everything a file says about itself, as one grouped view.
//!
//! Two readers, one shape. **sift** (truespar's pure-Rust ExifTool/Poppler
//! replacement) covers photos and PDFs - EXIF, MakerNotes, XMP, IPTC, ICC,
//! PDF Info, QuickTime, HEIF, plus its derived Composite tags. **scriptor**
//! covers OPC packages (docx/xlsx/pptx), whose properties live inside a zip
//! that sift does not open: `core.xml`, `app.xml`, and the provenance sleeper
//! `custom.xml` (DMS client/matter stamps, compare-tool markers). They are
//! not interchangeable and never overlap - a file is one or the other - so
//! callers ask once and get whichever applies.
//!
//! Measured on the corpus (release build): a Nikon COOLPIX
//! JPEG yields 115 tags in 5 groups, a Canon 40D 79 in 4, a distiller PDF 26
//! in 2, a docx 9 in 3 - each in well under a millisecond.
//!
//! One field here is not read out of the file at all: `location` resolves a
//! photo's coordinates to a place name offline (`paddock-geo`), and carries
//! the two f64s beside it. Both halves are needed downstream - a map cannot
//! plot the degrees-minutes-seconds string sift derives, and "43.467448,
//! 11.885127" is not an answer to "where was this".
//!
//! This is deliberately not the prompt-injection metadata. `paddock-runner`'s
//! `doc.rs` curates six PDF fields and three photo bits for the model, because
//! a tag zoo in a prompt is noise a model does not answer questions from. This
//! crate is the other audience: a person who asked what is in their file, and
//! wants all 121 tags.
//!
//! ## Why its own crate
//!
//! Both halves of the split need it. The runner serves it as API surface
//! (`POST /api/metadata`), so an API client with no manager still gets it; the
//! manager serves it off the attachment blobs it already stores
//! (`GET /api/attachments/{id}/metadata`), so reading a photo's EXIF does not
//! require a GPU server to be running. One crate is what keeps the two answers
//! identical.
//!
//! Nothing here is cached. Measured (release build): tags off a 22.7 MB PDF
//! take ~21 ms *including* a ~31 ms process-spawn baseline - the
//! parse is below the noise floor of starting the process that does it. A cache
//! would buy nothing and would guarantee that an extractor fix leaves users
//! looking at stale metadata.

use serde::Serialize;

/// Per-value ceiling. XMP happily carries kilobyte descriptions (and, rarely,
/// an embedded thumbnail as text) - a metadata table is not the place to
/// render those, and the response rides a browser. Truncation is DISCLOSED on
/// the tag rather than silent.
const MAX_VALUE_CHARS: usize = 4096;

/// Per-name ceiling. Names come from the file's own tag tables, so a corrupt
/// or hostile file can propose an arbitrarily long one.
const MAX_NAME_CHARS: usize = 96;

/// One file's embedded metadata, grouped by where it came from.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct FileMetadata {
    /// What the BYTES turned out to be - never the declared mime, which is
    /// whatever the browser guessed from a file extension. `None` when the
    /// magic bytes match nothing we read.
    pub format: Option<&'static str>,
    /// Which reader answered: `"sift"`, `"scriptor"`, or `"none"` when the
    /// file carries nothing. Callers show it as provenance; the upstream
    /// attribution principle applies to what a user sees, not just licences.
    pub reader: &'static str,
    /// Groups in reader order. The names come from the reader and the file,
    /// so treat the set as open: sift emits EXIF, MakerNotes, XMP, IPTC, ICC,
    /// PDF, QuickTime, HEIF, Composite (values it derives) and File (container
    /// facts like byte order); scriptor emits Document, Application, Custom.
    /// Empty is an ordinary answer, not a failure: a screenshot, a code file,
    /// a stripped export.
    ///
    /// A group is a namespace, so the same NAME in two groups is information,
    /// not duplication - a PDF states CreateDate in both its Info dictionary
    /// and its XMP, in different formats, and both are worth showing. Two tags
    /// with the same name in one group is the bug, and sift guards against it
    /// (`tests/tag_uniqueness.rs` there).
    pub groups: Vec<Group>,
    /// Where the file says it was, if it says. The coordinates are also in
    /// `groups` as text, so this exists for the two things text cannot do:
    /// carry NUMBERS a map can plot, and carry a place name that is not in
    /// the file at all (layer 2).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub location: Option<Location>,
}

/// A coordinate the file carries, plus what it turned out to be.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Location {
    /// Decimal degrees, negative south.
    pub latitude: f64,
    /// Decimal degrees, negative west.
    pub longitude: f64,
    /// Metres above sea level, when the file states it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub altitude: Option<f64>,
    /// Nearest populated place, resolved offline. `None` only when the lookup
    /// table could not be read - every real coordinate on Earth has a nearest
    /// place, even if it is 900 km away across water.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub place: Option<Place>,
}

/// A place name for a coordinate - DERIVED, not read out of the file.
///
/// Nearest populated place rather than point-in-polygon, so `region` is the
/// region of the matched CITY and not a boundary test on the point itself.
/// Everything here keeps that honest: the distance is carried so a caller can
/// state it, and [`Place::description`] is the one phrase that already words
/// it correctly, so nothing downstream has to invent its own.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Place {
    pub city: String,
    /// First-level division ("Tuscany"). Empty when the source has no name.
    pub region: String,
    pub country: String,
    /// Great-circle km from the coordinate to the place.
    pub distance_km: f64,
    /// Compass direction from the place to the coordinate ("NE").
    pub bearing: &'static str,
    /// "in Arezzo (Tuscany, Italy)", or "12 km NE of X (...)" when far enough
    /// that "in" would be a claim. Identical to the phrase the prompt line
    /// carries, deliberately: two audiences, one wording.
    pub description: String,
}

/// One source of tags within a file.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Group {
    pub name: String,
    pub tags: Vec<Tag>,
}

/// One metadata field, display-ready.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Tag {
    pub name: String,
    pub value: String,
    /// Set when the value hit [`MAX_VALUE_CHARS`] - so a UI can say the value
    /// continues rather than presenting a cut string as the whole truth.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub truncated: bool,
}

impl FileMetadata {
    /// The honest empty answer: we read the bytes and they say nothing.
    fn none() -> FileMetadata {
        FileMetadata {
            format: None,
            reader: "none",
            groups: Vec::new(),
            location: None,
        }
    }

    /// No metadata found. Not an error - most screenshots land here.
    pub fn is_empty(&self) -> bool {
        self.groups.is_empty()
    }

    /// Total tags across every group.
    pub fn tag_count(&self) -> usize {
        self.groups.iter().map(|g| g.tags.len()).sum()
    }

    /// Append one tag, creating its group on first sight. Group order is
    /// therefore the reader's own order, which is the order a person expects
    /// to read them in (identity first, derived values last).
    fn push(&mut self, group: &str, name: &str, value: &str) {
        let (value, truncated) = clean(value, MAX_VALUE_CHARS);
        let (name, _) = clean(name, MAX_NAME_CHARS);
        // A named field with no value is noise: the file has the slot, not the
        // fact. Same for the reverse - sift never emits it, but a corrupt tag
        // table could.
        if name.is_empty() || value.is_empty() {
            return;
        }
        let tag = Tag {
            name,
            value,
            truncated,
        };
        match self.groups.iter_mut().find(|g| g.name == group) {
            Some(g) => g.tags.push(tag),
            None => self.groups.push(Group {
                name: group.to_owned(),
                tags: vec![tag],
            }),
        }
    }
}

/// Read whatever the file says about itself. Never fails and never panics on
/// content: metadata is a garnish, and an unreadable file is an empty answer,
/// not an error the caller has to render.
///
/// **Blocking** - it parses a document. Call it off the async executor.
pub fn read(bytes: &[u8]) -> FileMetadata {
    // OPC first, by magic: every OOXML package is a zip, and sift reads no zip
    // container, so the order is a routing fact rather than a preference.
    if bytes.starts_with(b"PK\x03\x04") {
        let opc = read_opc(bytes);
        if !opc.is_empty() {
            return opc;
        }
        // A zip that is not an Office package (or an Office package with no
        // property parts). Claiming a format we did not confirm would be a
        // guess, so say nothing.
        return FileMetadata::none();
    }
    read_sift(bytes)
}

/// Photos, PDFs, video containers - everything sift's magic-byte detection
/// recognizes.
fn read_sift(bytes: &[u8]) -> FileMetadata {
    let Ok(mut doc) = sift::read(bytes) else {
        return FileMetadata::none();
    };
    // Permissions-only encryption with an empty user password is the
    // Word/Acrobat "protect" default; without this its Info dict reads as
    // nothing at all. Mirrors the runner's text lane (`doc.rs::extract_text`).
    // A real password fails here and simply yields fewer tags - the honest
    // refusal for that belongs to the text lane, not to a metadata panel.
    doc.authenticate(b"");
    let mut out = FileMetadata {
        format: doc.file_type().map(format_name),
        reader: "sift",
        groups: Vec::new(),
        // `doc.gps()` does duplicate the Composite GPSPosition tag as text, and
        // that used to be the argument for not calling it. That has changed
        // the answer: a map needs two f64s, and no amount of string parsing on
        // "43 deg 28' 2.81\" N, 11 deg 53' 6.46\" E" should be the way a pane
        // gets them.
        location: doc.gps().map(|g| Location {
            latitude: g.latitude,
            longitude: g.longitude,
            altitude: g.altitude,
            place: paddock_geo::nearest(g.latitude, g.longitude).map(|p| Place {
                description: p.describe(),
                city: p.city,
                region: p.region,
                country: p.country,
                distance_km: p.distance_km,
                bearing: p.bearing,
            }),
        }),
    };
    for tag in doc.tags() {
        out.push(tag.group, &tag.name, &tag.value);
    }
    if out.is_empty() && out.format.is_none() {
        return FileMetadata::none();
    }
    out
}

/// OOXML document properties. Group names name the PART they came from in
/// plain words, because that is the distinction that matters when two of them
/// disagree: `Document` is what the author typed, `Application` is what Word
/// counted, `Custom` is what some system stamped on the way past.
fn read_opc(bytes: &[u8]) -> FileMetadata {
    let core = scriptor_crdt::extract::core_properties(bytes);
    let ext = scriptor_crdt::extract::extended_properties(bytes);
    let mut out = FileMetadata {
        format: Some("Office Open XML"),
        reader: "scriptor",
        groups: Vec::new(),
        // OOXML has no coordinate the format defines. A custom property could
        // hold one, but reading a place out of an arbitrary string is a guess.
        location: None,
    };
    for (name, value) in [
        ("Title", core.title),
        ("Subject", core.subject),
        ("Author", core.creator),
        ("Keywords", core.keywords),
        ("Last modified by", core.last_modified_by),
        ("Created", core.created),
        ("Modified", core.modified),
    ] {
        if let Some(v) = value {
            out.push("Document", name, &v);
        }
    }
    for (name, value) in [
        ("Pages", ext.pages),
        ("Words", ext.words),
        ("Company", ext.company),
        ("Manager", ext.manager),
    ] {
        if let Some(v) = value {
            out.push("Application", name, &v);
        }
    }
    // Uncapped deliberately, unlike the prompt path (which takes 16): a person
    // inspecting a file wants every stamp on it, and the per-value cap plus
    // the drop-empties rule already bound what one property can cost.
    for (name, value) in &ext.custom {
        out.push("Custom", name, value);
    }
    if out.is_empty() {
        return FileMetadata::none();
    }
    out
}

/// One value -> one display string. Control characters collapse to spaces
/// (these are single-line table cells, and an embedded newline in an XMP
/// description would otherwise break the row); the result is capped, and the
/// caller is told when that happened.
fn clean(value: &str, cap: usize) -> (String, bool) {
    let mut out = String::new();
    let mut truncated = false;
    for (i, c) in value.chars().enumerate() {
        if i == cap {
            truncated = true;
            break;
        }
        out.push(if c.is_control() { ' ' } else { c });
    }
    (out.trim().to_owned(), truncated)
}

/// The format's own name, as a person writes it.
///
/// Deliberately exhaustive with no wildcard arm: when sift learns a format,
/// this stops compiling and someone names it here, instead of the new format
/// silently reporting nothing. One line to fix, and the compiler points at it.
fn format_name(t: sift::core::FileType) -> &'static str {
    use sift::core::FileType as F;
    match t {
        F::Jpeg => "JPEG",
        F::Png => "PNG",
        F::Gif => "GIF",
        F::Bmp => "BMP",
        F::Tiff => "TIFF",
        F::WebP => "WebP",
        F::Heif => "HEIF",
        F::Pdf => "PDF",
        F::Icc => "ICC profile",
        F::QuickTime => "QuickTime",
        F::Cr2 => "Canon CR2",
        F::Cr3 => "Canon CR3",
        F::Nef => "Nikon NEF",
        F::Arw => "Sony ARW",
        F::Dng => "Adobe DNG",
        F::Orf => "Olympus ORF",
        F::Rw2 => "Panasonic RW2",
        F::Pef => "Pentax PEF",
        F::Raf => "Fujifilm RAF",
        F::Srw => "Samsung SRW",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal JPEG carrying one EXIF tag (Make = TestCam), hand-built so the
    /// test needs no fixture file. Same construction as the runner's
    /// `doc.rs` photo tests.
    fn exif_jpeg() -> Vec<u8> {
        let tiff: Vec<u8> = [
            &[0x49, 0x49, 0x2A, 0x00, 0x08, 0x00, 0x00, 0x00][..], // II*\0, IFD0 @8
            &[0x01, 0x00][..],                                     // 1 entry
            // tag 0x010F (Make), ASCII, count 8, value @ offset 26
            &[
                0x0F, 0x01, 0x02, 0x00, 0x08, 0x00, 0x00, 0x00, 0x1A, 0x00, 0x00, 0x00,
            ][..],
            &[0x00, 0x00, 0x00, 0x00][..], // no next IFD
            b"TestCam\0",
        ]
        .concat();
        let mut jpeg = vec![0xFF, 0xD8]; // SOI
        jpeg.extend_from_slice(&[0xFF, 0xE1]); // APP1
        let len = (2 + 6 + tiff.len()) as u16;
        jpeg.extend_from_slice(&len.to_be_bytes());
        jpeg.extend_from_slice(b"Exif\0\0");
        jpeg.extend_from_slice(&tiff);
        jpeg.extend_from_slice(&[0xFF, 0xD9]); // EOI
        jpeg
    }

    /// A JPEG whose only EXIF content is a GPS IFD, hand-built for the same
    /// reason as the one above: the real corpus is a separate download, so a
    /// test that only ran against it would not run on CI or on a fresh
    /// checkout. Coordinates are the Tuscany photo's, rounded to whole
    /// seconds - 43°28'03" N, 11°53'06" E.
    fn gps_jpeg() -> Vec<u8> {
        // offsets are from the start of the TIFF header: IFD0 at 8 (one entry,
        // 18 bytes), GPS IFD at 26 (four entries, 54 bytes), then the two
        // rational triples it points at.
        const GPS_IFD: u32 = 26;
        const LAT_AT: u32 = 80;
        const LON_AT: u32 = 104;
        let rational3 = |a: u32, b: u32, c: u32| -> Vec<u8> {
            [a, 1, b, 1, c, 1]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect()
        };
        let entry = |tag: u16, ty: u16, count: u32, val: [u8; 4]| -> Vec<u8> {
            [
                &tag.to_le_bytes()[..],
                &ty.to_le_bytes()[..],
                &count.to_le_bytes()[..],
                &val[..],
            ]
            .concat()
        };
        let tiff: Vec<u8> = [
            &[0x49, 0x49, 0x2A, 0x00, 0x08, 0x00, 0x00, 0x00][..], // II*\0, IFD0 @8
            &[0x01, 0x00][..],                                     // IFD0: 1 entry
            &entry(0x8825, 4, 1, GPS_IFD.to_le_bytes())[..],       // GPSInfo pointer
            &[0x00, 0x00, 0x00, 0x00][..],                         // no next IFD
            &[0x04, 0x00][..],                                     // GPS IFD: 4 entries
            &entry(0x0001, 2, 2, *b"N\0\0\0")[..],                 // GPSLatitudeRef
            &entry(0x0002, 5, 3, LAT_AT.to_le_bytes())[..],        // GPSLatitude
            &entry(0x0003, 2, 2, *b"E\0\0\0")[..],                 // GPSLongitudeRef
            &entry(0x0004, 5, 3, LON_AT.to_le_bytes())[..],        // GPSLongitude
            &[0x00, 0x00, 0x00, 0x00][..],                         // no next IFD
            &rational3(43, 28, 3)[..],
            &rational3(11, 53, 6)[..],
        ]
        .concat();
        let mut jpeg = vec![0xFF, 0xD8, 0xFF, 0xE1];
        let len = (2 + 6 + tiff.len()) as u16;
        jpeg.extend_from_slice(&len.to_be_bytes());
        jpeg.extend_from_slice(b"Exif\0\0");
        jpeg.extend_from_slice(&tiff);
        jpeg.extend_from_slice(&[0xFF, 0xD9]);
        jpeg
    }

    /// An OPC package with all three property parts, incl. the custom stamp
    /// that is the whole reason `custom.xml` is read at all.
    fn docx() -> Vec<u8> {
        let core = r#"<?xml version="1.0"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/">
<dc:title>Fee Schedule</dc:title><dc:creator>Legal Team</dc:creator>
</cp:coreProperties>"#;
        let app = r#"<?xml version="1.0"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties">
<Pages>2</Pages><Words>310</Words><Company>ACME Law</Company>
</Properties>"#;
        let custom = r#"<?xml version="1.0"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/custom-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
<property fmtid="{D5CDD505-2E9C-101B-9397-08002B2CF9AE}" pid="2" name="Matter"><vt:lpwstr>ACME-0042</vt:lpwstr></property>
</Properties>"#;
        scriptor_ooxml::write_parts_bytes(&[
            scriptor_ooxml::Part {
                name: "word/document.xml".into(),
                data: b"<w:document/>".to_vec(),
            },
            scriptor_ooxml::Part {
                name: "docProps/core.xml".into(),
                data: core.as_bytes().to_vec(),
            },
            scriptor_ooxml::Part {
                name: "docProps/app.xml".into(),
                data: app.as_bytes().to_vec(),
            },
            scriptor_ooxml::Part {
                name: "docProps/custom.xml".into(),
                data: custom.as_bytes().to_vec(),
            },
        ])
        .expect("zip")
    }

    fn tag<'a>(m: &'a FileMetadata, group: &str, name: &str) -> Option<&'a Tag> {
        m.groups
            .iter()
            .find(|g| g.name == group)?
            .tags
            .iter()
            .find(|t| t.name == name)
    }

    #[test]
    fn a_photo_reads_through_sift() {
        let m = read(&exif_jpeg());
        assert_eq!(m.format, Some("JPEG"));
        assert_eq!(m.reader, "sift");
        assert_eq!(
            tag(&m, "EXIF", "Make").map(|t| t.value.as_str()),
            Some("TestCam")
        );
    }

    #[test]
    fn an_office_package_reads_through_scriptor() {
        let m = read(&docx());
        assert_eq!(m.format, Some("Office Open XML"));
        assert_eq!(m.reader, "scriptor");
        assert_eq!(
            tag(&m, "Document", "Title").map(|t| t.value.as_str()),
            Some("Fee Schedule")
        );
        assert_eq!(
            tag(&m, "Application", "Words").map(|t| t.value.as_str()),
            Some("310")
        );
        // custom.xml is the provenance sleeper - the prompt path caps it at 16
        // fields, this one carries them all
        assert_eq!(
            tag(&m, "Custom", "Matter").map(|t| t.value.as_str()),
            Some("ACME-0042")
        );
    }

    #[test]
    fn the_group_order_is_the_readers_order() {
        // Identity before statistics before stamps: what a person reads first.
        let m = read(&docx());
        let names: Vec<&str> = m.groups.iter().map(|g| g.name.as_str()).collect();
        assert_eq!(names, ["Document", "Application", "Custom"]);
    }

    #[test]
    fn nothing_readable_is_an_empty_answer_not_an_error() {
        for bytes in [
            &b""[..],
            &b"just some text, no container at all"[..],
            &[0xFF, 0xD8, 0xFF, 0xD9][..], // a JPEG with no metadata segments
            &b"PK\x03\x04not really a zip"[..],
        ] {
            let m = read(bytes);
            assert!(m.is_empty(), "expected no tags, got {m:?}");
        }
    }

    #[test]
    fn a_bare_jpeg_still_names_its_format() {
        // No tags, but the bytes are a JPEG - the panel can say so.
        let m = read(&[0xFF, 0xD8, 0xFF, 0xD9]);
        assert_eq!(m.format, Some("JPEG"));
        assert!(m.is_empty());
    }

    #[test]
    fn control_characters_never_reach_the_value() {
        // A newline inside a value would break out of its table row; the
        // prompt path has the same rule for a stronger reason (a forged
        // metadata line), so the discipline is shared.
        let (v, truncated) = clean("Fee\nSchedule\u{0}", MAX_VALUE_CHARS);
        assert_eq!(v, "Fee Schedule");
        assert!(!truncated);
    }

    #[test]
    fn a_long_value_is_capped_and_says_so() {
        let (v, truncated) = clean(&"x".repeat(MAX_VALUE_CHARS + 10), MAX_VALUE_CHARS);
        assert_eq!(v.len(), MAX_VALUE_CHARS);
        assert!(truncated, "truncation must be disclosed, never silent");
    }

    #[test]
    fn a_photos_coordinates_come_back_as_numbers_and_a_place() {
        let m = read(&gps_jpeg());
        let loc = m.location.expect("a GPS IFD is a location");
        // seconds-resolution fixture, so this is the coordinate to ~30 m
        assert!((loc.latitude - 43.4675).abs() < 1e-4, "{loc:?}");
        assert!((loc.longitude - 11.885).abs() < 1e-4, "{loc:?}");
        let place = loc.place.expect("resolved offline");
        assert_eq!(place.city, "Arezzo");
        assert_eq!(place.country, "Italy");
        // the phrase is worded once, in paddock-geo, and shared with the
        // prompt line - a pane must never re-derive its own version
        assert_eq!(place.description, "in Arezzo (Tuscany, Italy)");
    }

    #[test]
    fn a_photo_without_gps_has_no_location() {
        // Most photos. The field is absent, not a zero-zero coordinate off the
        // coast of Africa, which is what a defaulted lat/lon would have meant.
        assert!(read(&exif_jpeg()).location.is_none());
        assert!(read(&docx()).location.is_none());
    }

    #[test]
    fn the_wire_shape_is_flat_and_skips_the_default() {
        let m = read(&exif_jpeg());
        let v = serde_json::to_value(&m).expect("serializes");
        assert_eq!(v["format"], "JPEG");
        assert_eq!(v["groups"][0]["name"], "EXIF");
        assert!(v["groups"][0]["tags"][0].get("name").is_some());
        // `truncated` is absent unless it happened - a false on every tag is
        // noise in a response that carries a hundred of them
        assert!(v["groups"][0]["tags"][0].get("truncated").is_none());
        // same rule for the whole location block: a photo without one says
        // nothing rather than sending a null the client has to test for
        assert!(v.get("location").is_none());

        let g = serde_json::to_value(read(&gps_jpeg())).expect("serializes");
        assert!(g["location"]["latitude"].is_f64());
        assert_eq!(g["location"]["place"]["city"], "Arezzo");
        // no altitude in the fixture, so no altitude on the wire
        assert!(g["location"].get("altitude").is_none());
    }
}
