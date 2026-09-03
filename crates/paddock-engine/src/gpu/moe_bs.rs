//! block-scale (mxf8f6f4) MoE arms.

use super::error::*;
use super::*;
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};

impl GpuExecutor {
    /// sm_120a block-scale sorted-MoE gate+up+swiglu (mxFP4 x FP8, hardware
    /// ue8m0 scaling). fq/fs receive the e4m3 + ue8m0 swiglu output.
    #[allow(clippy::too_many_arguments)]
    pub fn mxfp4_moe_gate_up_bs(
        &self,
        gate_w: &RepackedMxfp4,
        gate_bias: &CudaSlice<f32>,
        up_w: &RepackedMxfp4,
        up_bias: &CudaSlice<f32>,
        sorted_row: &CudaSlice<u32>,
        block_expert: &CudaSlice<u32>,
        yq: &CudaSlice<i8>,
        ys: &CudaSlice<u8>,
        fq: &mut CudaSlice<i8>,
        fs: &mut CudaSlice<u8>,
        in_dim: usize,
        ff: usize,
        max_blocks: usize,
        rows: usize,
        alpha: f32,
        limit: f32,
        // SwiGLU up-term: 1.0 = gpt-oss (u+1); 0.0 = qwen plain silu(g)*u.
        up_add: f32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .mxfp4_moe_gate_up_bs
            .ok_or(GpuError::MissingOp("mxfp4_moe_gate_up_bs"))?;
        let (gdp, _g1) = gate_w.data.device_ptr(&self.stream);
        let (gsp, _g1s) = gate_w.scale.device_ptr(&self.stream);
        let (gbp, _g2) = gate_bias.device_ptr(&self.stream);
        let (udp, _g3) = up_w.data.device_ptr(&self.stream);
        let (usp, _g3s) = up_w.scale.device_ptr(&self.stream);
        let (ubp, _g4) = up_bias.device_ptr(&self.stream);
        let (srp, _g5) = sorted_row.device_ptr(&self.stream);
        let (bep, _g6) = block_expert.device_ptr(&self.stream);
        let (yqp, _g7) = yq.device_ptr(&self.stream);
        let (ysp, _g8) = ys.device_ptr(&self.stream);
        let (fqp, _g9) = fq.device_ptr_mut(&self.stream);
        let (fsp, _g10) = fs.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; sorted layout from moe_align
        check(unsafe {
            f(
                gdp as *const _,
                gsp as *const _,
                gbp as *const _,
                udp as *const _,
                usp as *const _,
                ubp as *const _,
                srp as *const _,
                bep as *const _,
                yqp as *const _,
                ysp as *const _,
                fqp as *mut _,
                fsp as *mut _,
                in_dim as u32,
                ff as u32,
                max_blocks as u32,
                rows as u32,
                alpha,
                limit,
                up_add,
                self.stream_ptr(),
            )
        })
    }

    /// Prefill-config gate_up_bs (64-token sorted blocks, 64-row weight
    /// tiles) - see the pack-side notes; pair with `mxfp4_moe_down_bs64` and
    /// `moe_align_bm(bm=64)`.
    #[allow(clippy::too_many_arguments)]
    pub fn mxfp4_moe_gate_up_bs64(
        &self,
        gate_w: &RepackedMxfp4,
        gate_bias: &CudaSlice<f32>,
        up_w: &RepackedMxfp4,
        up_bias: &CudaSlice<f32>,
        sorted_row: &CudaSlice<u32>,
        block_expert: &CudaSlice<u32>,
        yq: &CudaSlice<i8>,
        ys: &CudaSlice<u8>,
        fq: &mut CudaSlice<i8>,
        fs: &mut CudaSlice<u8>,
        in_dim: usize,
        ff: usize,
        max_blocks: usize,
        rows: usize,
        alpha: f32,
        limit: f32,
        // SwiGLU up-term: 1.0 = gpt-oss (u+1); 0.0 = qwen plain silu(g)*u.
        up_add: f32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .mxfp4_moe_gate_up_bs64
            .ok_or(GpuError::MissingOp("mxfp4_moe_gate_up_bs64"))?;
        let (gdp, _g1) = gate_w.data.device_ptr(&self.stream);
        let (gsp, _g1s) = gate_w.scale.device_ptr(&self.stream);
        let (gbp, _g2) = gate_bias.device_ptr(&self.stream);
        let (udp, _g3) = up_w.data.device_ptr(&self.stream);
        let (usp, _g3s) = up_w.scale.device_ptr(&self.stream);
        let (ubp, _g4) = up_bias.device_ptr(&self.stream);
        let (srp, _g5) = sorted_row.device_ptr(&self.stream);
        let (bep, _g6) = block_expert.device_ptr(&self.stream);
        let (yqp, _g7) = yq.device_ptr(&self.stream);
        let (ysp, _g8) = ys.device_ptr(&self.stream);
        let (fqp, _g9) = fq.device_ptr_mut(&self.stream);
        let (fsp, _g10) = fs.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; sorted layout from moe_align
        check(unsafe {
            f(
                gdp as *const _,
                gsp as *const _,
                gbp as *const _,
                udp as *const _,
                usp as *const _,
                ubp as *const _,
                srp as *const _,
                bep as *const _,
                yqp as *const _,
                ysp as *const _,
                fqp as *mut _,
                fsp as *mut _,
                in_dim as u32,
                ff as u32,
                max_blocks as u32,
                rows as u32,
                alpha,
                limit,
                up_add,
                self.stream_ptr(),
            )
        })
    }

    /// Prefill-config down_bs - see `mxfp4_moe_gate_up_bs64`.
    #[allow(clippy::too_many_arguments)]
    pub fn mxfp4_moe_down_bs64(
        &self,
        down_w: &RepackedMxfp4,
        down_bias: &CudaSlice<f32>,
        sorted_row: &CudaSlice<u32>,
        sorted_slot: &CudaSlice<u32>,
        block_expert: &CudaSlice<u32>,
        topk_w: &CudaSlice<f32>,
        fq: &CudaSlice<i8>,
        fs: &CudaSlice<u8>,
        part: &mut CudaSlice<f32>,
        ff: usize,
        embd: usize,
        n_active: usize,
        max_blocks: usize,
        rows: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .mxfp4_moe_down_bs64
            .ok_or(GpuError::MissingOp("mxfp4_moe_down_bs64"))?;
        let (ddp, _g1) = down_w.data.device_ptr(&self.stream);
        let (dsp, _g1s) = down_w.scale.device_ptr(&self.stream);
        let (dbp, _g2) = down_bias.device_ptr(&self.stream);
        let (srp, _g3) = sorted_row.device_ptr(&self.stream);
        let (ssp, _g4) = sorted_slot.device_ptr(&self.stream);
        let (bep, _g5) = block_expert.device_ptr(&self.stream);
        let (twp, _g6) = topk_w.device_ptr(&self.stream);
        let (fqp, _g7) = fq.device_ptr(&self.stream);
        let (fsp, _g8) = fs.device_ptr(&self.stream);
        let (pp, _g9) = part.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; part sized [rows, n_active, embd]
        check(unsafe {
            f(
                ddp as *const _,
                dsp as *const _,
                dbp as *const _,
                srp as *const _,
                ssp as *const _,
                bep as *const _,
                twp as *const _,
                fqp as *const _,
                fsp as *const _,
                pp as *mut _,
                ff as u32,
                embd as u32,
                n_active as u32,
                max_blocks as u32,
                rows as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Whether the pack ships the bs64 prefill MoE pair + bm-align.
    pub fn has_moe_bs64(&self) -> bool {
        self.kernels.moe_align_bm.is_some()
            && self.kernels.mxfp4_moe_gate_up_bs64.is_some()
            && self.kernels.mxfp4_moe_down_bs64.is_some()
    }

    /// Block-scale sorted-MoE down: consumes gate_up_bs output, emits the
    /// deterministic per-(token, slot) partials (fold with moe_slot_combine).
    #[allow(clippy::too_many_arguments)]
    pub fn mxfp4_moe_down_bs(
        &self,
        down_w: &RepackedMxfp4,
        down_bias: &CudaSlice<f32>,
        sorted_row: &CudaSlice<u32>,
        sorted_slot: &CudaSlice<u32>,
        block_expert: &CudaSlice<u32>,
        topk_w: &CudaSlice<f32>,
        fq: &CudaSlice<i8>,
        fs: &CudaSlice<u8>,
        part: &mut CudaSlice<f32>,
        ff: usize,
        embd: usize,
        n_active: usize,
        max_blocks: usize,
        rows: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .mxfp4_moe_down_bs
            .ok_or(GpuError::MissingOp("mxfp4_moe_down_bs"))?;
        let (ddp, _g1) = down_w.data.device_ptr(&self.stream);
        let (dsp, _g1s) = down_w.scale.device_ptr(&self.stream);
        let (dbp, _g2) = down_bias.device_ptr(&self.stream);
        let (srp, _g3) = sorted_row.device_ptr(&self.stream);
        let (ssp, _g4) = sorted_slot.device_ptr(&self.stream);
        let (bep, _g5) = block_expert.device_ptr(&self.stream);
        let (twp, _g6) = topk_w.device_ptr(&self.stream);
        let (fqp, _g7) = fq.device_ptr(&self.stream);
        let (fsp, _g8) = fs.device_ptr(&self.stream);
        let (pp, _g9) = part.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; part sized [rows, n_active, embd]
        check(unsafe {
            f(
                ddp as *const _,
                dsp as *const _,
                dbp as *const _,
                srp as *const _,
                ssp as *const _,
                bep as *const _,
                twp as *const _,
                fqp as *const _,
                fsp as *const _,
                pp as *mut _,
                ff as u32,
                embd as u32,
                n_active as u32,
                max_blocks as u32,
                rows as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Whether the pack ships `mxfp4_moe_down_bs_res` (the fused residual
    /// fold); older packs fall back to down_bs + `moe_slot_combine`.
    pub fn has_moe_down_bs_res(&self) -> bool {
        self.kernels.mxfp4_moe_down_bs_res.is_some()
    }

    /// `mxfp4_moe_down_bs` with the slot_combine fold fused into the epilogue
    /// (bit-exact vs the two-launch pair). `cnt` is one u32 per (token,
    /// 128-col y-tile), zeroed once at alloc and never reset.
    #[allow(clippy::too_many_arguments)]
    pub fn mxfp4_moe_down_bs_res(
        &self,
        down_w: &RepackedMxfp4,
        down_bias: &CudaSlice<f32>,
        sorted_row: &CudaSlice<u32>,
        sorted_slot: &CudaSlice<u32>,
        block_expert: &CudaSlice<u32>,
        topk_w: &CudaSlice<f32>,
        fq: &CudaSlice<i8>,
        fs: &CudaSlice<u8>,
        part: &mut CudaSlice<f32>,
        residual: &mut CudaSlice<f32>,
        cnt: &mut CudaSlice<u32>,
        ff: usize,
        embd: usize,
        n_active: usize,
        max_blocks: usize,
        rows: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .mxfp4_moe_down_bs_res
            .ok_or(GpuError::MissingOp("mxfp4_moe_down_bs_res"))?;
        let (ddp, _g1) = down_w.data.device_ptr(&self.stream);
        let (dsp, _g1s) = down_w.scale.device_ptr(&self.stream);
        let (dbp, _g2) = down_bias.device_ptr(&self.stream);
        let (srp, _g3) = sorted_row.device_ptr(&self.stream);
        let (ssp, _g4) = sorted_slot.device_ptr(&self.stream);
        let (bep, _g5) = block_expert.device_ptr(&self.stream);
        let (twp, _g6) = topk_w.device_ptr(&self.stream);
        let (fqp, _g7) = fq.device_ptr(&self.stream);
        let (fsp, _g8) = fs.device_ptr(&self.stream);
        let (pp, _g9) = part.device_ptr_mut(&self.stream);
        let (rp, _g10) = residual.device_ptr_mut(&self.stream);
        let (cp, _g11) = cnt.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; part sized [rows, n_active, embd], cnt sized
        // [rows, ceil(embd/128)]
        check(unsafe {
            f(
                ddp as *const _,
                dsp as *const _,
                dbp as *const _,
                srp as *const _,
                ssp as *const _,
                bep as *const _,
                twp as *const _,
                fqp as *const _,
                fsp as *const _,
                pp as *mut _,
                rp as *mut _,
                cp as *mut _,
                ff as u32,
                embd as u32,
                n_active as u32,
                max_blocks as u32,
                rows as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Whether the block-scale dense route is fully available: the requant,
    /// activation-quantize and GEMM entries are all non-NULL (sm_120a packs).
    /// Callers must also gate on `compute_capability().0 == 12` - the fatbin
    /// carries sm_120a SASS only for these kernels.
    pub fn has_mxfp4_gemm_bs(&self) -> bool {
        self.kernels.mxfp4_gemm_bs.is_some()
            && self.kernels.q8_0_to_mxfp4.is_some()
            && self.kernels.quantize_e4m3.is_some()
            && self.kernels.quantize_e4m3_swiglu.is_some()
    }

    /// Whether the mxf4 (fp4 x fp4, m16n8k64) dense route is fully available:
    /// requant, e2m1 activation quantizers and the k64 GEMM. Same cc-12
    /// caller obligation as [`has_mxfp4_gemm_bs`].
    pub fn has_mxfp4_gemm_f4(&self) -> bool {
        self.kernels.mxfp4_gemm_f4.is_some()
            && self.kernels.q8_0_to_fp4p.is_some()
            && self.kernels.quantize_e2m1.is_some()
            && self.kernels.quantize_e2m1_swiglu.is_some()
    }
}
