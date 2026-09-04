//! Host-side f32 tensor loading for GPU parity TESTS - not a serving path.
//!
//! The CPU reference models that once lived here (llama, gpt-oss) are gone:
//! correctness is same-weights greedy parity against the newest llama.cpp
//! release binary, and the no-CPU-code rule abolished the in-house CPU
//! reference convention. What remains is the
//! smallest thing the GPU parity tests need: dequant a GGUF tensor to f32 on
//! the host and run a scalar matvec to diff a kernel against.

mod tensor;

pub use tensor::HostTensor;

use paddock_models::ggml_type::GgmlType;
use paddock_models::mapped::{MapError, MappedGguf};

#[derive(Debug, thiserror::Error)]
pub enum RefModelError {
    #[error(transparent)]
    Map(#[from] MapError),
    #[error("tensor {name}: dequant of {ty:?} not supported in the reference path")]
    UnsupportedType { name: String, ty: GgmlType },
    #[error(transparent)]
    Dequant(#[from] paddock_kernels::reference::DequantError),
}

/// Load any supported tensor as f32, whatever its on-disk type.
pub fn load_f32(map: &MappedGguf, name: &str) -> Result<HostTensor, RefModelError> {
    let (info, bytes) = map.tensor_bytes(name)?;
    let n = info.element_count() as usize;
    let mut data = vec![0f32; n];
    match info.ggml_type {
        GgmlType::F32 => {
            for (i, chunk) in bytes.as_chunks::<4>().0.iter().enumerate() {
                // as_chunks::<4>() hands out whole words only
                data[i] = f32::from_le_bytes(*chunk);
            }
        }
        GgmlType::F16 => {
            for (i, chunk) in bytes.as_chunks::<2>().0.iter().enumerate() {
                data[i] = half::f16::from_le_bytes(*chunk).to_f32();
            }
        }
        GgmlType::Q8_0 => paddock_kernels::reference::dequant_q8_0(bytes, &mut data)?,
        GgmlType::Mxfp4 => paddock_kernels::reference::dequant_mxfp4(bytes, &mut data)?,
        ty => {
            return Err(RefModelError::UnsupportedType {
                name: name.to_owned(),
                ty,
            });
        }
    }
    Ok(HostTensor::new(
        data,
        info.dims.iter().map(|&d| d as usize).collect(),
    ))
}
