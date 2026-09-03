//! weight upload / dequant / Q8 repack.

use super::error::*;
use super::types::narrow_to_f16;
use super::*;
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use half::f16;
use paddock_kernels::abi::DequantF32Fn;
use paddock_models::ggml_type::GgmlType;
use paddock_models::mapped::MappedGguf;

impl GpuExecutor {
    pub(super) fn dequant_for(
        &self,
        ty: GgmlType,
        name: &str,
    ) -> Result<(DequantF32Fn, usize), GpuError> {
        let (f, block_elems) = match ty {
            GgmlType::Mxfp4 => (self.kernels.mxfp4_dequant_f32, 32),
            GgmlType::Q8_0 => (self.kernels.q8_0_dequant_f32, 32),
            // not a quant type at all - a bf16 plane's "dequant" is a widen.
            // It lives here so a mixed UD file's bf16 token_embd serves the
            // single-row gather (dequant_slice) with no call-site change.
            // 32 elems/"block" is the table's unit, not a bf16 property.
            GgmlType::Bf16 => (self.kernels.bf16_dequant_f32, 32),
            _ => (None, 0),
        };
        f.map(|f| (f, block_elems))
            .ok_or_else(|| GpuError::NoKernel {
                name: name.to_owned(),
                ty,
            })
    }

    /// Upload a tensor fully dequanted to f32 on device.
    pub fn upload(&self, map: &MappedGguf, name: &str) -> Result<DeviceTensor, GpuError> {
        let (info, bytes) = map.tensor_bytes(name)?;
        let dims: Vec<usize> = info.dims.iter().map(|&d| d as usize).collect();
        let n = info.element_count() as usize;

        let buf = match info.ggml_type {
            GgmlType::F32 => {
                let host: Vec<f32> = bytes
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().expect("4 bytes")))
                    .collect();
                self.stream.clone_htod(&host).map_err(drv)?
            }
            // bf16 = the top half of an f32 - host-side widen (small side
            // tensors; the 35B's nextn router/norms ship as bf16 where the
            // backbone's are f32)
            GgmlType::Bf16 => {
                let host: Vec<f32> = bytes
                    .chunks_exact(2)
                    .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
                    .collect();
                self.stream.clone_htod(&host).map_err(drv)?
            }
            // f16 host-side widen (exact) - UD k-quant exports ship tiny side
            // projections (qwen35 ssm_alpha/ssm_beta, [embd, 32]) as F16.
            GgmlType::F16 => {
                let host: Vec<f32> = bytes
                    .chunks_exact(2)
                    .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                    .collect();
                self.stream.clone_htod(&host).map_err(drv)?
            }
            // k-quant family: the pack's generic dequant takes (dtype, n_super)
            // instead of the per-type fn - separate arm, same transient-upload shape.
            ty if kq_params(ty).is_some() => {
                let (raw_id, _, _) = kq_params(ty).expect("guarded");
                let f = self
                    .kernels
                    .kquant_dequant
                    .ok_or(GpuError::MissingOp("kquant_dequant"))?;
                let d_raw = self.stream.clone_htod(bytes).map_err(drv)?;
                let mut d_out: CudaSlice<f32> = self.stream.alloc_zeros(n).map_err(drv)?;
                {
                    let (in_ptr, _g1) = d_raw.device_ptr(&self.stream);
                    let (out_ptr, _g2) = d_out.device_ptr_mut(&self.stream);
                    // SAFETY: pack ABI v1 contract; pointers + stream live across the call
                    check(unsafe {
                        f(
                            in_ptr as *const core::ffi::c_void,
                            out_ptr as *mut core::ffi::c_void,
                            (n / 256) as u64,
                            raw_id,
                            self.stream_ptr(),
                        )
                    })?;
                }
                d_out
            }
            ty => {
                let (dequant, block_elems) = self.dequant_for(ty, name)?;
                let d_raw = self.stream.clone_htod(bytes).map_err(drv)?;
                let mut d_out: CudaSlice<f32> = self.stream.alloc_zeros(n).map_err(drv)?;
                let n_blocks = (n / block_elems) as u64;
                {
                    let (in_ptr, _g1) = d_raw.device_ptr(&self.stream);
                    let (out_ptr, _g2) = d_out.device_ptr_mut(&self.stream);
                    // SAFETY: pack ABI v1 contract; pointers + stream live across the call
                    check(unsafe {
                        dequant(
                            in_ptr as *const core::ffi::c_void,
                            out_ptr as *mut core::ffi::c_void,
                            n_blocks,
                            self.stream_ptr(),
                        )
                    })?;
                }
                d_out
            }
        };
        Ok(DeviceTensor { buf, dims })
    }

    /// Upload a dense tensor keeping it at **f16** on device.
    ///
    /// For an F16 source this is the file's own storage class - no conversion
    /// happens at all, and the plane costs exactly what the GGUF holds instead
    /// of twice that. F32/BF16 sources are narrowed on the host, which is a
    /// precision step and is why this is not the default upload: only call it
    /// for planes whose sole consumer is a tensor-core GEMM that accumulates in
    /// f32 (`gemm_f16_f32`), where the f16 operand class is the intended one.
    ///
    /// Quantized types are refused rather than routed through a dequant: a
    /// k-quant plane wants the quant-aware GEMM, and silently widening it to
    /// f16 here would hide that the caller picked the wrong lane.
    pub fn upload_f16(&self, map: &MappedGguf, name: &str) -> Result<HalfTensor, GpuError> {
        let (info, bytes) = map.tensor_bytes(name)?;
        let dims: Vec<usize> = info.dims.iter().map(|&d| d as usize).collect();
        let host: Vec<f16> = match info.ggml_type {
            GgmlType::F16 => bytes
                .chunks_exact(2)
                .map(|c| f16::from_le_bytes([c[0], c[1]]))
                .collect(),
            // F32/BF16 sources go through the checked narrow: both carry f32's
            // exponent range, so an out-of-range weight would land as `inf`
            // here and poison every row it touches.
            GgmlType::F32 => {
                let f32s: Vec<f32> = bytes
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().expect("4 bytes")))
                    .collect();
                narrow_to_f16(&f32s, name)?
            }
            GgmlType::Bf16 => {
                let f32s: Vec<f32> = bytes
                    .chunks_exact(2)
                    .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
                    .collect();
                narrow_to_f16(&f32s, name)?
            }
            ty => {
                return Err(GpuError::Driver(format!(
                    "upload_f16({name}): {ty:?} is a quantized type - it belongs on a quant-aware \
                     GEMM lane, not an f16 plane"
                )));
            }
        };
        Ok(HalfTensor {
            buf: self.stream.clone_htod(&host).map_err(drv)?,
            dims,
        })
    }

    /// Upload a tensor keeping its quantized bytes on device (dequant per use).
    pub fn upload_raw(&self, map: &MappedGguf, name: &str) -> Result<QuantTensor, GpuError> {
        let (info, bytes) = map.tensor_bytes(name)?;
        // Q4_0 lands as its exact Q8_0 transcode (see repack_q8): the raw-
        // layout consumers (dequant_slice, the embed gathers) speak Q8_0,
        // and the transcode dequants bit-identically.
        if info.ggml_type == GgmlType::Q4_0 {
            let q8 = crate::gpu::q40_to_q8_blocks(bytes);
            return Ok(QuantTensor {
                bytes: self.stream.clone_htod(&q8).map_err(drv)?,
                ty: GgmlType::Q8_0,
                dims: info.dims.iter().map(|&d| d as usize).collect(),
            });
        }
        Ok(QuantTensor {
            bytes: self.stream.clone_htod(bytes).map_err(drv)?,
            ty: info.ggml_type,
            dims: info.dims.iter().map(|&d| d as usize).collect(),
        })
    }

    /// Upload a tensor's raw bytes into a TRANSIENT allocation, run `work`
    /// with the device pointer, then free it. DEFAULT: stream-ordered pool
    /// staging (clone + drop) - the pre-Act-81 path. The classic
    /// cuMemAlloc/cuMemFree variant (PADDOCK_CLASSIC_STAGING=1, debug only)
    /// would keep the mempool hole-free (3.25 GB un-trimmable staging holes
    /// measured on the 35B-A3B-Q8 load) but was REVERTED as the default:
    /// it corrupts repacked planes on the 35B-A3B (greedy text
    /// degrades under spec; bisected via the same-weights llama oracle;
    /// mechanism undiagnosed - kquant parity green, 9B clean, only the MoE
    /// expert planes visibly wrong). Do not re-enable without a same-weights
    /// greedy-parity gate on the 35B; the hole-free win needs a different
    /// design (staging arena / two-phase load).
    pub(super) fn with_staged_raw<R>(
        &self,
        bytes: &[u8],
        work: impl FnOnce(u64) -> Result<R, GpuError>,
    ) -> Result<R, GpuError> {
        // CAP: the buffer serves tensors up to `staging_cap` and the
        // few above it keep the old per-tensor path. Grow-to-the-largest left a
        // residue exactly the size of the biggest plane, because a buffer that
        // large is never handed back - `release_staging` moved those bytes out
        // of `pool live` and into the ctx bucket with every total unchanged.
        // Measured on sm_86, one box, one binary, model resident total:
        //
        //                     no buffer   uncapped   capped
        //   qwen3.8/q4          21.07      20.10      19.06
        //   qwen3.8/q8          29.39      30.50      29.23
        //
        // Q8 is the lane that proves the cap: uncapped it is a REGRESSION there
        // (+1.11 GB over no buffer at all), because the buffer grows to the
        // 1.29 GiB token_embd while that lane only ever had 0.30 GB of holes to
        // save. Capped it is -0.16.
        //
        // Do not read this off the residency split alone - it cannot see it. On
        // the uncapped q8 lane the split says `retained-not-live 0.06 GB (0.2%
        // of live)`, identical to capped, while the process holds 1.27 GB more:
        // `release_staging` drops the buffer from the POOL, so retained goes
        // quiet, but the driver does not hand the pages back to the process.
        // The resident total is the only line that shows it.
        //
        // 128 MiB is read off the models, not picked round: qwen3.8's q4 and q8
        // are 866 tensors each with exactly two over the cap - `token_embd` and
        // `output`, the vocab-sized planes - and the largest under it is 58.4 MiB
        // (q4) / 90.3 MiB (q8). So 864 of 866 still ride the buffer, which is
        // where the holes come from; the giants only ever set the size.
        let staging_cap: usize = paddock_models::dev_var!("PADDOCK_STAGING_MAX")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(128 << 20);
        if paddock_models::dev_var_os!("PADDOCK_CLASSIC_STAGING").is_none()
            && bytes.len() <= staging_cap
        {
            // One staging buffer for the whole load, grown to the largest
            // tensor and reused. The old shape - a fresh clone_htod per tensor,
            // freed on return - allocated each weight plane with a same-sized
            // staging buffer beside it and then freed the staging, so every
            // hole landed between two live planes and cuMemPoolTrimTo could
            // never hand the block back. See `GpuExecutor::staging`.
            let mut slot = self.staging.lock().unwrap_or_else(|e| e.into_inner());
            let need = bytes.len();
            if slot.as_ref().is_none_or(|b| b.len() < need) {
                // drop the old one before allocating, so the grow reuses it
                *slot = None;
                *slot = Some(self.alloc_u8(need)?);
            }
            let buf = slot.as_mut().expect("just filled");
            let mut view = buf
                .try_slice_mut(0..need)
                .ok_or_else(|| GpuError::Unsupported("staging slice".into()))?;
            self.stream.memcpy_htod(bytes, &mut view).map_err(drv)?;
            let (sp, _g) = view.device_ptr(&self.stream);
            return work(sp);
        }
        if paddock_models::dev_var_os!("PADDOCK_CLASSIC_STAGING").is_none() {
            // Over the cap: the pre-arena per-tensor shape. It does leave a
            // hole, which is the whole reason the buffer exists - but it is two
            // tensors per model, not 864, and a bounded hole beats a permanent
            // buffer the trim can never reclaim. Not the classic path below:
            // that one is reverted for corrupting the 35B-A3B expert planes and
            // must not be reached by falling off the end of a size test.
            let raw = self.stream.clone_htod(bytes).map_err(drv)?;
            let (sp, _g) = raw.device_ptr(&self.stream);
            return work(sp);
        }
        unsafe {
            let dp = cudarc::driver::result::malloc_sync(bytes.len()).map_err(drv)?;
            let r = (|| {
                cudarc::driver::result::memcpy_htod_sync(dp, bytes).map_err(drv)?;
                let out = work(dp)?;
                // the repack kernel reads dp on our stream - it must retire
                // before the free (free_sync is immediate, not stream-ordered)
                self.stream.synchronize().map_err(drv)?;
                Ok(out)
            })();
            let _ = cudarc::driver::result::free_sync(dp);
            r
        }
    }

    /// Upload a Q8_0 weight and repack it into the aligned data + f16-scale streams
    /// the vectorized decode GEMV reads. The staged upload is freed on return.
    pub fn repack_q8(&self, map: &MappedGguf, name: &str) -> Result<RepackedQ8, GpuError> {
        let (info, bytes) = map.tensor_bytes(name)?;
        let dims: Vec<usize> = info.dims.iter().map(|&d| d as usize).collect();
        // Q4_0 (the QAT lineage) rides the Q8 lane through the exact host
        // transcode: same 32-weight block, same f16 scale, int8 = nibble - 8,
        // so dequant is bit-identical - a compressed Q8_0, not a requant.
        if info.ggml_type == GgmlType::Q4_0 {
            let q8 = crate::gpu::q40_to_q8_blocks(bytes);
            return self.repack_q8_blocks(&q8, dims);
        }
        if info.ggml_type != GgmlType::Q8_0 {
            return Err(GpuError::NoKernel {
                name: name.to_owned(),
                ty: info.ggml_type,
            });
        }
        self.repack_q8_blocks(bytes, dims)
    }

    /// Repack raw Q8_0 blocks (34-byte GGUF layout: f16 scale + 32 int8) that
    /// did not come from a GGUF tensor - the load-time-quant seam (the DFlash
    /// drafter's bf16 checkpoint planes quantize on host and land here).
    pub fn repack_q8_blocks(&self, bytes: &[u8], dims: Vec<usize>) -> Result<RepackedQ8, GpuError> {
        // full element count: expert tensors are 3D ([in, out, n_expert]) and
        // repack as one flat stream of (e*out + o) rows
        let n_blocks = dims.iter().product::<usize>() / 32;
        debug_assert_eq!(bytes.len(), n_blocks * 34, "raw Q8_0 block stream size");
        let f = self
            .kernels
            .q8_0_repack
            .ok_or(GpuError::MissingOp("q8_0_repack"))?;
        let mut data = self.alloc_u8(n_blocks * 32)?;
        let mut scale = self.alloc_u8(n_blocks * 2)?; // f16 per block
        self.with_staged_raw(bytes, |sp| {
            let (dp, _g2) = data.device_ptr_mut(&self.stream);
            let (scp, _g3) = scale.device_ptr_mut(&self.stream);
            check(unsafe {
                f(
                    sp as *const _,
                    dp as *mut _,
                    scp as *mut _,
                    n_blocks as u64,
                    self.stream_ptr(),
                )
            })
        })?;
        Ok(RepackedQ8 { data, scale, dims })
    }

    /// Upload + repack two Q8_0 weights of the same in_dim into one fused
    /// plane concatenated along out_dim ([a-rows | b-rows] - the repack block
    /// stream is per-output-row contiguous, so concatenation is offset math).
    /// The merged-projection (vLLM-style qkv / gate_up) building block.
    pub fn repack_q8_concat2(
        &self,
        map: &MappedGguf,
        name_a: &str,
        name_b: &str,
    ) -> Result<RepackedQ8, GpuError> {
        let (ia, ba) = map.tensor_bytes(name_a)?;
        let (ib, bb) = map.tensor_bytes(name_b)?;
        for (i, n) in [(&ia, name_a), (&ib, name_b)] {
            if i.ggml_type != GgmlType::Q8_0 {
                return Err(GpuError::NoKernel {
                    name: n.to_string(),
                    ty: i.ggml_type,
                });
            }
        }
        assert_eq!(ia.dims[0], ib.dims[0], "fused planes need one in_dim");
        let na = ia.dims.iter().product::<u64>() as usize / 32;
        let nb = ib.dims.iter().product::<u64>() as usize / 32;
        let f = self
            .kernels
            .q8_0_repack
            .ok_or(GpuError::MissingOp("q8_0_repack"))?;
        let mut data = self.alloc_u8((na + nb) * 32)?;
        let mut scale = self.alloc_u8((na + nb) * 2)?;
        self.with_staged_raw(ba, |sp| {
            let (dp, _g2) = data.device_ptr_mut(&self.stream);
            let (scp, _g3) = scale.device_ptr_mut(&self.stream);
            check(unsafe {
                f(
                    sp as *const _,
                    dp as *mut _,
                    scp as *mut _,
                    na as u64,
                    self.stream_ptr(),
                )
            })
        })?;
        self.with_staged_raw(bb, |sp| {
            let (dp, _g2) = data.device_ptr_mut(&self.stream);
            let (scp, _g3) = scale.device_ptr_mut(&self.stream);
            check(unsafe {
                f(
                    sp as *const _,
                    (dp as usize + na * 32) as *mut _,
                    (scp as usize + na * 2) as *mut _,
                    nb as u64,
                    self.stream_ptr(),
                )
            })
        })?;
        Ok(RepackedQ8 {
            data,
            scale,
            dims: vec![ia.dims[0] as usize, (ia.dims[1] + ib.dims[1]) as usize],
        })
    }
}
