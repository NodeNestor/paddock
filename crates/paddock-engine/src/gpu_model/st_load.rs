//! Reading plain (unquantized) tensors out of a safetensors checkpoint.
//!
//! The safetensors-primary lanes all need the same three things - widen a bf16
//! tensor to f32, hand back raw bf16 bytes for a device-resident bf16 plane,
//! and say so loudly when a tensor is missing or the wrong dtype - so they live
//! here instead of once per family.
//!
//! This exists because the codebase was already growing copies: `bf16_to_f32`
//! had three independent definitions when granite needed a fourth. The rule
//! is to grep the API rather than the symbol name; the two
//! stragglers (`nemotron/dflash.rs`, `qwen3_asr/aligner.rs`) predate this and
//! should fold in when either is next touched.

use paddock_models::safetensors::{ShardedSafetensors, StDtype};

use crate::gpu_model::gpt_oss::GpuModelError;

/// Widen bf16 to f32 exactly: bf16 is the top 16 bits of an f32, so this is a
/// shift, never a rounding.
pub(crate) fn bf16_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| f32::from_bits((u16::from_le_bytes(*c) as u32) << 16))
        .collect()
}

/// Raw bf16 bytes for a device-resident bf16 plane. Values are identical to
/// the f32 widening - the widen just moves into the consuming kernel's
/// registers - so this is a residency choice, not a numeric one.
pub(crate) fn bf16_bytes<'a>(
    st: &'a ShardedSafetensors,
    name: &str,
    want_elems: usize,
) -> Result<&'a [u8], GpuModelError> {
    let (t, bytes) = st
        .bytes(name)
        .ok_or_else(|| GpuModelError::Unsupported(format!("{name}: tensor missing")))?;
    if t.dtype != StDtype::Bf16 {
        return Err(GpuModelError::Unsupported(format!(
            "{name}: expected bf16, got {:?}",
            t.dtype
        )));
    }
    if bytes.len() != want_elems * 2 {
        return Err(GpuModelError::Unsupported(format!(
            "{name}: {} bytes, expected {want_elems} bf16 elements",
            bytes.len()
        )));
    }
    Ok(bytes)
}

/// Read a tensor as f32, widening bf16 exactly; f32 passes through.
pub(crate) fn f32_tensor(
    st: &ShardedSafetensors,
    name: &str,
    want_elems: usize,
) -> Result<Vec<f32>, GpuModelError> {
    let (t, bytes) = st
        .bytes(name)
        .ok_or_else(|| GpuModelError::Unsupported(format!("{name}: tensor missing")))?;
    let v = match t.dtype {
        StDtype::Bf16 => bf16_to_f32(bytes),
        StDtype::F32 => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect(),
        other => {
            return Err(GpuModelError::Unsupported(format!(
                "{name}: expected bf16/f32, got {other:?}"
            )));
        }
    };
    if v.len() != want_elems {
        return Err(GpuModelError::Unsupported(format!(
            "{name}: {} elements, expected {want_elems}",
            v.len()
        )));
    }
    Ok(v)
}
