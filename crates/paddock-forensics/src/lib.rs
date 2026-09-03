//! # paddock-forensics
//!
//! Image / document forensic **signal extraction** as a runner-gated library.
//! It ports the analyzer *algorithms* from a dedicated forensics service (ELA,
//! noise, PRNU, splice, ...) into paddock so the serving runner can, on request,
//! run them over an attachment's **original bytes** and feed the findings to the
//! local model - the same job that service does, minus the external VLM.
//!
//! Design invariants:
//! - **GPU-first, CPU fallback.** Each [`Analyzer`] implements both a `cpu` and
//!   (under the `cuda` feature) a `gpu` path computing the *same* canonical
//!   algorithm. [`run`] prefers GPU when a [`gpu::ForensicGpu`] is supplied and
//!   falls back to CPU transparently on any GPU error.
//! - **Off the LLM ABI.** The GPU path loads a multi-arch fatbin through cudarc
//!   ([`gpu`]); it never touches paddock's `KernelTableV1`.
//! - **Byte-exact.** Analyzers run on the original ingestion bytes - ELA/PRNU/
//!   JPEG-ghost die under re-encode/resize, so the [`Context`] holds the raw
//!   bytes and the decoded pixels, nothing re-compressed.
//!
//! Parity between the two paths is enforced by `tests/parity.rs`.
//!
//! ## Coverage audit
//!
//! Every signal analyzer in the reference implementation is ported (40 registered in
//! [`analyzer::default_analyzers`] + `pdf_render_compare`, wired by the runner
//! via a [`PageRenderer`]): all of `pixel/`, `ai_detect/`, `metadata/`, `pdf/`,
//! plus `risk/` (scoring + report dedup + template `explanation`) and
//! `annotation/` (overlay generation, [`annotate`]). HEIC/HEIF input decodes
//! under the `heic` feature.
//!
//! Intentionally **not** ported, with reasons:
//! - `ml/manipulation` (Mesorch, AAAI 2025 ONNX): would need the `ort` ONNX
//!   Runtime dependency + a 340 MB model published to the registry. paddock-
//!   forensics feeds a bigger local model that is the AI-judgment layer, so a
//!   separate CNN is redundant here.
//! - `ml/deepfake`: a stub in the reference (`analyze` returns `[]`) - no capability.
//! - `ml/forensic_model` (DINOv3 7B): disabled even in the reference (≈27 GB VRAM,
//!   domain gap).
//! - `ocr/` (Tesseract, external binary): the consuming model does OCR;
//!   paste_rectangle's adaptive-threshold text masking is already ported inline.
//! - `vlm/`, `webhook/`, `api/`, `storage/`: the reference's service layer - out
//!   of scope by design (this is a library that feeds a model, not a service).
//! - The reference's VLM-coupled bits within ported modules: the risk scorer's
//!   stage-2 probability blend, and the smart-forensic / VLM-region overlays.

mod analyzer;
pub mod annotate;
mod context;
mod error;
pub mod metadata;
pub mod pdf;
pub mod pixel;
pub mod risk;

pub mod gpu;

pub use analyzer::{Analyzer, Report, run, run_analyzer};
pub use context::{ContentType, Context};
pub use error::ContextError;
pub use pdf::render_compare::{PageRenderer, RenderCompareOpts, render_compare};

use serde::{Deserialize, Serialize};

/// Severity of a finding, ordered from least to most serious.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Severity {
    Info,
    Low,
    Medium,
    High,
    Critical,
}

/// A spatial region within the image where an issue was localized. Callers use
/// it to draw overlays; the model can be told "the anomaly is at ...".
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Region {
    /// Axis-aligned bounding box, pixel coordinates, origin top-left.
    BoundingBox {
        x: u32,
        y: u32,
        width: u32,
        height: u32,
    },
    /// A set of specific pixel coordinates.
    Points { points: Vec<[u32; 2]> },
    /// Per-pixel binary mask (row-major, same dimensions as the source), where
    /// each byte is 0 or 255.
    Mask {
        width: u32,
        height: u32,
        data: Vec<u8>,
    },
}

/// A single forensic finding.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Finding {
    /// Name of the analyzer that produced it (e.g. `"ela"`).
    pub analyzer: String,
    /// Short stable identifier for the finding type (e.g. `"ela_block_outliers"`).
    pub code: String,
    /// Human-readable description (this string is what feeds the model prompt).
    pub description: String,
    pub severity: Severity,
    /// Confidence in the finding, 0.0..=1.0.
    pub confidence: f64,
    /// Optional spatial location.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub region: Option<Region>,
}

impl Finding {
    /// Build a finding without a spatial region.
    pub fn new(
        analyzer: &'static str,
        code: impl Into<String>,
        description: impl Into<String>,
        severity: Severity,
        confidence: f64,
    ) -> Self {
        Self {
            analyzer: analyzer.to_string(),
            code: code.into(),
            description: description.into(),
            severity,
            confidence,
            region: None,
        }
    }

    /// Attach a spatial region.
    pub fn with_region(mut self, region: Region) -> Self {
        self.region = Some(region);
        self
    }
}
