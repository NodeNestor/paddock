//! Metadata-domain forensic analyzers (Wave 5), ported from the reference
//! implementation. All operate on the sift-derived `tags` + the original
//! `raw_bytes` (plus the decoded image for EXIF↔pixel cross-checks); they are
//! pure-CPU (no GPU kernel exists or is warranted), so each `gpu()` delegates to
//! `cpu()`.

pub mod analyzer;
pub mod c2pa;
pub mod exif_pixel;
