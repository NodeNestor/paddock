//! BF16 dense weight planes - per-TENSOR quant dispatch.
//!
//! UD quant files are MIXED: muse-glimmer's UD-Q8_K_XL keeps `token_embd`,
//! `output`, `attn_k` and `attn_v` at bf16 while everything else is Q8_0,
//! because the quantizer judged those planes worth the bytes. The project's rule
//! for that is per-tensor dispatch rather than a per-model switch, and the
//! correctness spine is same-weights parity against llama.cpp on the identical
//! file - so down-quantizing a bf16 plane into the Q8_0 lane at load is out on
//! both counts (different weights, no exact greedy target, and a deliberate
//! quality choice silently overridden).
//!
//! A bf16 plane is carried by the plain [`QuantTensor`] the raw uploader
//! already produces (`bytes` + `ty` + `dims`) - no new plane struct, and
//! `dims` keeps the house convention `[in_dim, out_dim]`.

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};

use super::error::check;
use super::{GpuError, GpuExecutor, QuantTensor};
use paddock_models::ggml_type::GgmlType;

impl GpuExecutor {
    /// True when the pack carries the bf16 dense lane. Every consumer checks
    /// this before electing a bf16 plane, so an older pack degrades to a loud
    /// `MissingOp` at load rather than a wrong answer at serve.
    pub fn has_bf16_dense(&self) -> bool {
        self.kernels.bf16_gemv_f32.is_some()
            && self.kernels.bf16_gemm_f32.is_some()
            && self.kernels.bf16_dequant_f32.is_some()
    }

    /// True when the pack can gather rows out of a bf16 embedding table -
    /// loaders elect the bf16-resident table only behind this, so an older
    /// pack falls back to the widened f32 table instead of erroring.
    pub fn has_embed_gather_bf16(&self) -> bool {
        self.kernels.embed_gather_bf16.is_some()
    }

    /// Guard for the plane class itself: a `QuantTensor` reaching these
    /// entry points must actually hold bf16 bytes. The loader decides the
    /// class from the file, so a mismatch here is a routing bug, not user
    /// input - but it would read garbage weights silently, so name it.
    fn bf16_plane<'a>(&self, w: &'a QuantTensor, who: &str) -> Result<&'a QuantTensor, GpuError> {
        if w.ty != GgmlType::Bf16 {
            return Err(GpuError::NoKernel {
                name: format!("{who}: plane is not bf16"),
                ty: w.ty,
            });
        }
        Ok(w)
    }

    /// `y = W x` over a bf16 weight plane - the r==1 twin of
    /// `q8_0_gemv_repacked`, same operand convention (`dims[0]` = in_dim).
    pub fn bf16_gemv(
        &self,
        w: &QuantTensor,
        bias: Option<&CudaSlice<f32>>,
        x: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
    ) -> Result<(), GpuError> {
        let w = self.bf16_plane(w, "bf16_gemv")?;
        let f = self
            .kernels
            .bf16_gemv_f32
            .ok_or(GpuError::MissingOp("bf16_gemv_f32"))?;
        let (in_dim, out_dim) = (w.dims[0], w.dims[1]);
        let (wp, _g1) = w.bytes.device_ptr(&self.stream);
        let (xp, _g2) = x.device_ptr(&self.stream);
        let (yp, _g3) = y.device_ptr_mut(&self.stream);
        let (bias_ptr, _gb);
        let bp: *const core::ffi::c_void = match bias {
            Some(b) => {
                (bias_ptr, _gb) = b.device_ptr(&self.stream);
                bias_ptr as *const _
            }
            None => core::ptr::null(),
        };
        // SAFETY: ABI contract; shapes come from the plane's own dims
        check(unsafe {
            f(
                wp as *const _,
                bp,
                xp as *const _,
                yp as *mut _,
                in_dim as u32,
                out_dim as u32,
                self.stream_ptr(),
            )
        })
    }

    /// `bf16_gemv` over a row SEGMENT of a fused plane: rows
    /// `[first_row, first_row + out_dim)` of `w` (house dims `[in, out]`).
    /// The nemotron attn twins live as one load-time-concatenated `[q;k;v]`
    /// plane (thin-k/v rung), and the serial decode row still
    /// wants three per-projection GEMVs - this is that read, a plain base
    /// pointer offset (row-major planes make a row range a byte range).
    pub fn bf16_gemv_rows(
        &self,
        w: &QuantTensor,
        first_row: usize,
        out_dim: usize,
        x: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
    ) -> Result<(), GpuError> {
        let w = self.bf16_plane(w, "bf16_gemv_rows")?;
        let f = self
            .kernels
            .bf16_gemv_f32
            .ok_or(GpuError::MissingOp("bf16_gemv_f32"))?;
        let in_dim = w.dims[0];
        debug_assert!(first_row + out_dim <= w.dims[1], "segment exceeds plane");
        let (wp, _g1) = w.bytes.device_ptr(&self.stream);
        let (xp, _g2) = x.device_ptr(&self.stream);
        let (yp, _g3) = y.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; the segment is a contiguous byte range of
        // the plane (2 bytes per bf16 element, in_dim elements per row)
        check(unsafe {
            f(
                (wp + (first_row * in_dim * 2) as u64) as *const _,
                core::ptr::null(),
                xp as *const _,
                yp as *mut _,
                in_dim as u32,
                out_dim as u32,
                self.stream_ptr(),
            )
        })
    }

    /// `bf16_gemv` writing at an output-row offset inside `y` - the twin of
    /// `q8_0_gemv_repacked_at`, for the fused `[q|k|v]` decode row.
    pub fn bf16_gemv_at(
        &self,
        w: &QuantTensor,
        x: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        y_off: usize,
    ) -> Result<(), GpuError> {
        let w = self.bf16_plane(w, "bf16_gemv_at")?;
        let f = self
            .kernels
            .bf16_gemv_f32
            .ok_or(GpuError::MissingOp("bf16_gemv_f32"))?;
        let (in_dim, out_dim) = (w.dims[0], w.dims[1]);
        let (wp, _g1) = w.bytes.device_ptr(&self.stream);
        let (xp, _g2) = x.device_ptr(&self.stream);
        let (yp, _g3) = y.device_ptr_mut(&self.stream);
        // SAFETY: same allocation, offset output rowlet; caller sizes y
        check(unsafe {
            f(
                wp as *const _,
                core::ptr::null(),
                xp as *const _,
                (yp + (y_off * 4) as u64) as *mut _,
                in_dim as u32,
                out_dim as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Batched twin: `x` is `[batch, in_dim]`, `y` is `[batch, out_dim]` -
    /// the same row-major activation layout `q8_0_gemm_repacked` uses.
    pub fn bf16_gemm(
        &self,
        w: &QuantTensor,
        bias: Option<&CudaSlice<f32>>,
        x: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        batch: usize,
    ) -> Result<(), GpuError> {
        let out_dim = w.dims[1];
        self.bf16_gemm_dispatch(w, 0, out_dim, bias, x, y, batch)
    }

    /// `bf16_gemm` over a row segment of a fused plane (see
    /// `bf16_gemv_rows`) - the per-projection fallback when the pack lacks
    /// the fused `bf16_qkv_gemm_mma` slot, and the 2..=8 decode band's
    /// path (which stays on the multi-row GEMV class either way).
    pub fn bf16_gemm_rows(
        &self,
        w: &QuantTensor,
        first_row: usize,
        out_dim: usize,
        x: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        batch: usize,
    ) -> Result<(), GpuError> {
        if batch == 1 {
            return self.bf16_gemv_rows(w, first_row, out_dim, x, y);
        }
        self.bf16_gemm_dispatch(w, first_row, out_dim, None, x, y, batch)
    }

    fn bf16_gemm_dispatch(
        &self,
        w: &QuantTensor,
        first_row: usize,
        out_dim: usize,
        bias: Option<&CudaSlice<f32>>,
        x: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        batch: usize,
    ) -> Result<(), GpuError> {
        if batch == 1 {
            debug_assert_eq!(first_row, 0, "gemv segment goes via bf16_gemv_rows");
            return self.bf16_gemv(w, bias, x, y);
        }
        let w = self.bf16_plane(w, "bf16_gemm")?;
        let in_dim = w.dims[0];
        debug_assert!(first_row + out_dim <= w.dims[1], "segment exceeds plane");
        // byte offset of the segment's first row (2 bytes per bf16 element)
        let woff = (first_row * in_dim * 2) as u64;
        // Decode band (2..=8 rows): the 64x64 tile GEMM's grid degenerates to
        // out_dim/64 blocks here - 4 blocks for a 256-wide K/V plane, measured
        // at 794us for a 0.5 MB weight read on a paddleocr c4 leg.
        // The multi-row GEMV twin streams the plane at DRAM pace. Same
        // f32-product class; warp-local summation order (the sanctioned
        // reorder class - serial gemv vs batched gemm already differ the same
        // way and hold greedy parity).
        // NB=8 (batch 5..8) is the mr kernel's issue-bound arm: 16 cvt +
        // 128 FMA per 32 weight bytes per lane - 806 GB/s on the granite
        // lm_head (1028us/tick, the top c8 cost after the wqkv fusion) while
        // the BN=32 tile holds 1476 (bench/bf16_head_mr_bench.cu, cold
        // 4-plane rotation; b2/b4 are tied at the wall). Big outs have
        // die-filling tile grids (out/32 CTAs), so they take the tile; the
        // mr class keeps the small-out planes it was built for (256-wide
        // K/V heads = 4-CTA tile grids).
        let mr_band = (2..=8).contains(&batch) && !(batch >= 5 && out_dim >= 8192);
        if mr_band
            && in_dim % 16 == 0
            && let Some(f) = self.kernels.bf16_gemv_mr_f32
        {
            let (wp, _g1) = w.bytes.device_ptr(&self.stream);
            let (xp, _g2) = x.device_ptr(&self.stream);
            let (yp, _g3) = y.device_ptr_mut(&self.stream);
            let (bias_ptr, _gb);
            let bp: *const core::ffi::c_void = match bias {
                Some(b) => {
                    (bias_ptr, _gb) = b.device_ptr(&self.stream);
                    bias_ptr as *const _
                }
                None => core::ptr::null(),
            };
            // SAFETY: ABI contract; segment offset stays inside the plane
            return check(unsafe {
                f(
                    (wp + woff) as *const _,
                    bp,
                    xp as *const _,
                    yp as *mut _,
                    in_dim as u32,
                    out_dim as u32,
                    batch as u32,
                    self.stream_ptr(),
                )
            });
        }
        // Prefill band (batch > 8): the tensor-core arm. Casts the f32
        // activations to bf16 in its smem stage - the parity reference's own
        // batched-BF16 class (llama.cpp = cublasGemmEx bf16xbf16, f32
        // compute), gated by the parity battery. The f32-FMA tile below stays
        // the fallback for older packs and ragged in_dim.
        if batch >= 2 && in_dim % 16 == 0 {
            // the pack admits 2..8 since the head-band fix; small-out 2..8
            // returned via the mr branch above
            if let Some(f) = self.kernels.bf16_gemm_mma {
                let (wp, _g1) = w.bytes.device_ptr(&self.stream);
                let (xp, _g2) = x.device_ptr(&self.stream);
                let (yp, _g3) = y.device_ptr_mut(&self.stream);
                let (bias_ptr, _gb);
                let bp: *const core::ffi::c_void = match bias {
                    Some(b) => {
                        (bias_ptr, _gb) = b.device_ptr(&self.stream);
                        bias_ptr as *const _
                    }
                    None => core::ptr::null(),
                };
                // SAFETY: ABI contract; segment offset stays inside the plane
                return check(unsafe {
                    f(
                        (wp + woff) as *const _,
                        bp,
                        xp as *const _,
                        yp as *mut _,
                        in_dim as u32,
                        out_dim as u32,
                        batch as u32,
                        self.stream_ptr(),
                    )
                });
            }
        }
        let f = self
            .kernels
            .bf16_gemm_f32
            .ok_or(GpuError::MissingOp("bf16_gemm_f32"))?;
        let (wp, _g1) = w.bytes.device_ptr(&self.stream);
        let (xp, _g2) = x.device_ptr(&self.stream);
        let (yp, _g3) = y.device_ptr_mut(&self.stream);
        let (bias_ptr, _gb);
        let bp: *const core::ffi::c_void = match bias {
            Some(b) => {
                (bias_ptr, _gb) = b.device_ptr(&self.stream);
                bias_ptr as *const _
            }
            None => core::ptr::null(),
        };
        // SAFETY: ABI contract; segment offset stays inside the plane
        check(unsafe {
            f(
                (wp + woff) as *const _,
                bp,
                xp as *const _,
                yp as *mut _,
                in_dim as u32,
                out_dim as u32,
                batch as u32,
                self.stream_ptr(),
            )
        })
    }

    /// True when the pack carries the fused q|k|v decode-band GEMM (slot
    /// 424) - the caller keeps three per-segment launches otherwise.
    pub fn has_bf16_qkv_gemm(&self) -> bool {
        self.kernels.bf16_qkv_gemm_mma.is_some()
    }

    /// One launch computing q|k|v against the shared `x` over the fused
    /// `[q;k;v]` plane (dims `[in, oq + 2*okv]`), segmented store into the
    /// three y planes. Decode-band only (the launcher declines batch <= 1);
    /// per out-row bit-identical to `bf16_gemm` on the matching segment.
    #[allow(clippy::too_many_arguments)]
    pub fn bf16_qkv_gemm(
        &self,
        w: &QuantTensor,
        x: &CudaSlice<f32>,
        yq: &mut CudaSlice<f32>,
        yk: &mut CudaSlice<f32>,
        yv: &mut CudaSlice<f32>,
        oq: usize,
        okv: usize,
        batch: usize,
    ) -> Result<(), GpuError> {
        let w = self.bf16_plane(w, "bf16_qkv_gemm")?;
        let f = self
            .kernels
            .bf16_qkv_gemm_mma
            .ok_or(GpuError::MissingOp("bf16_qkv_gemm_mma"))?;
        let in_dim = w.dims[0];
        debug_assert_eq!(oq + 2 * okv, w.dims[1], "fused plane rows");
        let (wp, _g1) = w.bytes.device_ptr(&self.stream);
        let (xp, _g2) = x.device_ptr(&self.stream);
        let (qp, _g3) = yq.device_ptr_mut(&self.stream);
        let (kp, _g4) = yk.device_ptr_mut(&self.stream);
        let (vp, _g5) = yv.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; y planes are [batch, seg] sized by the caller
        check(unsafe {
            f(
                wp as *const _,
                xp as *const _,
                qp as *mut _,
                kp as *mut _,
                vp as *mut _,
                in_dim as u32,
                oq as u32,
                okv as u32,
                batch as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Embedding gather that picks its kernel from the TABLE's own class -
    /// a mixed UD file can hand either a Q8_0 or a bf16 `token_embd`, and the
    /// call sites (decode input, prefill rows, multimodal splice, drafter)
    /// should not each have to remember that.
    pub fn embed_gather_plane(
        &self,
        table: &QuantTensor,
        tokens: &CudaSlice<u32>,
        out: &mut CudaSlice<f32>,
        embd: usize,
        n_tokens: usize,
        scale: f32,
    ) -> Result<(), GpuError> {
        match table.ty {
            GgmlType::Bf16 => self.embed_gather_bf16(table, tokens, out, embd, n_tokens, scale),
            _ => self.embed_gather_q8(table, tokens, out, embd, n_tokens, scale),
        }
    }

    /// bf16 twin of `embed_gather_q8`: device-selected token rows out of a
    /// bf16 embedding table with the fused output scale (graph-capturable).
    pub fn embed_gather_bf16(
        &self,
        table: &QuantTensor,
        tokens: &CudaSlice<u32>,
        out: &mut CudaSlice<f32>,
        embd: usize,
        n_tokens: usize,
        scale: f32,
    ) -> Result<(), GpuError> {
        let table = self.bf16_plane(table, "embed_gather_bf16")?;
        let f = self
            .kernels
            .embed_gather_bf16
            .ok_or(GpuError::MissingOp("embed_gather_bf16"))?;
        let (tp, _g1) = table.bytes.device_ptr(&self.stream);
        let (kp, _g2) = tokens.device_ptr(&self.stream);
        let (op, _g3) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; out is [n_tokens, embd]
        check(unsafe {
            f(
                tp as *const _,
                kp as *const _,
                op as *mut _,
                embd as u32,
                n_tokens as u32,
                scale,
                self.stream_ptr(),
            )
        })
    }
}
