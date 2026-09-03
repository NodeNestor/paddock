//! Error types. `GpuError` lives in [`crate::gpu`] and is cuda-gated; the errors
//! here are always available.

use thiserror::Error;

/// Failure building an [`crate::Context`] from bytes.
#[derive(Debug, Error)]
pub enum ContextError {
    #[error("unsupported or undecodable image: {0}")]
    Decode(String),
    #[error("image too small: {width}x{height}")]
    TooSmall { width: u32, height: u32 },
}
