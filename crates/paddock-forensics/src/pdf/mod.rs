//! PDF document-forensics analyzers, ported from the reference. Unlike
//! the pixel lane these are CPU-only (structural / metadata / embedded-image
//! work, no pixel-parallel kernels), so each `gpu()` delegates to `cpu()` -
//! except `image_pipeline`, which runs the pixel analyzers (ELA/noise) over each
//! embedded image and therefore inherits their GPU path.
//!
//! All read the ORIGINAL bytes via `sift::read` (from-bytes, no temp file) plus
//! raw-byte scanning for content-stream / object-level patterns sift does not
//! surface. `applies_to = ctx.is_pdf()`.

pub mod image_pipeline;
pub mod overlay;
pub mod render_compare;
pub mod structure;
