//! fused gemma QKV, f8/fp4 GEMV ladder, vision attn glue.

use super::error::*;
use super::*;
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};

impl GpuExecutor {
    /// Gemma4/muse-glimmer fused decode QKV epilogue: per-head norms, rope,
    /// and the K/V append, one launch for what was seven.
    ///
    /// One wrapper over the pack's `gemma_qkv_nra3` - the nra/nra2/nra2s
    /// ladder that preceded it grew one entry per new argument (kv_dtype, then
    /// qkv_stride) and every caller had to know which rung carried what. The
    /// superset takes them all, including the three constants the epilogue
    /// used to bake:
    ///
    /// * `rope` - the family's `(theta_scale, freq_scale, ..)` tuple. Only the
    ///   first two matter here; this kernel has never carried the yarn ramp
    ///   because every consumer runs `ext_factor 0 / mscale 1`. `freq_scale`
    ///   is what makes a NoPE layer a bit-exact identity, and its absence from
    ///   the old entries is why muse-glimmer's full-attention layers were
    ///   re-roped on every DECODE step while prefill left them alone.
    /// * `neox` - pair layout (half-split vs interleaved).
    /// * `vnorm` - whether V gets the weightless per-head RMS norm.
    ///
    /// `qkv_stride` non-zero when q/k/v are `(buffer, offset)` views into one
    /// `[batch][qkv_stride]` concatenated GEMM plane.
    #[allow(clippy::too_many_arguments)]
    pub fn gemma_qkv_nra(
        &self,
        q: (&mut CudaSlice<f32>, usize),
        k: (&mut CudaSlice<f32>, usize),
        v: (&mut CudaSlice<f32>, usize),
        wq_norm: &CudaSlice<f32>,
        wk_norm: &CudaSlice<f32>,
        q_out: &mut CudaSlice<f32>,
        kc: &mut CudaSlice<u8>,
        vc: &mut CudaSlice<u8>,
        positions: &CudaSlice<u32>,
        slots: Option<&CudaSlice<u32>>,
        factors: Option<&CudaSlice<f32>>,
        block_tables: Option<&CudaSlice<u32>>,
        bps: usize,
        n_head: usize,
        n_kv: usize,
        head_dim: usize,
        max_ctx: usize,
        batch: usize,
        eps: f32,
        rope: (f32, f32, f32, f32, f32, f32),
        kv_dtype: KvDtype,
        qkv_stride: usize,
        neox: bool,
        vnorm: bool,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .gemma_qkv_nra3
            .ok_or(GpuError::MissingOp("gemma_qkv_nra3"))?;
        let (theta_scale, freq_scale, ..) = rope;
        let (qbp, _g1a) = q.0.device_ptr_mut(&self.stream);
        let qkvp = qbp + (q.1 * 4) as u64;
        let (kbp, _g1b) = k.0.device_ptr_mut(&self.stream);
        let kptr = kbp + (k.1 * 4) as u64;
        let (vbp, _g1c) = v.0.device_ptr_mut(&self.stream);
        let vptr = vbp + (v.1 * 4) as u64;
        let (wqp, _g2) = wq_norm.device_ptr(&self.stream);
        let (wkp, _g3) = wk_norm.device_ptr(&self.stream);
        let (qop, _g4) = q_out.device_ptr_mut(&self.stream);
        let (kp, _g5) = kc.device_ptr_mut(&self.stream);
        let (vp, _g6) = vc.device_ptr_mut(&self.stream);
        let (pp, _g7) = positions.device_ptr(&self.stream);
        let sg = slots.map(|s| s.device_ptr(&self.stream));
        let fg = factors.map(|s| s.device_ptr(&self.stream));
        let bg = block_tables.map(|s| s.device_ptr(&self.stream));
        let opt = |g: &Option<(u64, _)>| {
            g.as_ref()
                .map_or(core::ptr::null(), |(p, _)| *p as *const core::ffi::c_void)
        };
        // SAFETY: ABI contract; planes sized [batch * (qkv_stride | dims)]
        check(unsafe {
            f(
                qkvp as *mut _,
                kptr as *mut _,
                vptr as *mut _,
                wqp as *const _,
                wkp as *const _,
                qop as *mut _,
                kp as *mut _,
                vp as *mut _,
                pp as *const _,
                opt(&sg),
                opt(&fg),
                opt(&bg),
                bps as u32,
                n_head as u32,
                n_kv as u32,
                head_dim as u32,
                max_ctx as u32,
                batch as u32,
                eps,
                theta_scale,
                kv_dtype as u32,
                qkv_stride as u32,
                freq_scale,
                neox as u32,
                vnorm as u32,
                self.stream_ptr(),
            )
        })
    }

    pub fn has_gemma_qkv_nra2s(&self) -> bool {
        self.kernels.gemma_qkv_nra3.is_some()
    }

    /// `gemma_qkv_nra` twin whose q/k/v planes are PACKED bf16 (the b16-D
    /// election's p16 convention: f32 element indexing, half the bytes) -
    /// slot 420. Same epilogue; `q_out`/cache writes unchanged.
    /// Offsets are in ELEMENTS of the packed plane, so bytes are ×2 here
    /// where the f32 wrapper does ×4.
    #[allow(clippy::too_many_arguments)]
    pub fn gemma_qkv_nra_b16(
        &self,
        q: (&mut CudaSlice<f32>, usize),
        k: (&mut CudaSlice<f32>, usize),
        v: (&mut CudaSlice<f32>, usize),
        wq_norm: &CudaSlice<f32>,
        wk_norm: &CudaSlice<f32>,
        q_out: &mut CudaSlice<f32>,
        kc: &mut CudaSlice<u8>,
        vc: &mut CudaSlice<u8>,
        positions: &CudaSlice<u32>,
        slots: Option<&CudaSlice<u32>>,
        factors: Option<&CudaSlice<f32>>,
        block_tables: Option<&CudaSlice<u32>>,
        bps: usize,
        n_head: usize,
        n_kv: usize,
        head_dim: usize,
        max_ctx: usize,
        batch: usize,
        eps: f32,
        rope: (f32, f32, f32, f32, f32, f32),
        kv_dtype: KvDtype,
        qkv_stride: usize,
        neox: bool,
        vnorm: bool,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .gemma_qkv_nra3_b16
            .ok_or(GpuError::MissingOp("gemma_qkv_nra3_b16"))?;
        let (theta_scale, freq_scale, ..) = rope;
        let (qbp, _g1a) = q.0.device_ptr_mut(&self.stream);
        let qkvp = qbp + (q.1 * 2) as u64;
        let (kbp, _g1b) = k.0.device_ptr_mut(&self.stream);
        let kptr = kbp + (k.1 * 2) as u64;
        let (vbp, _g1c) = v.0.device_ptr_mut(&self.stream);
        let vptr = vbp + (v.1 * 2) as u64;
        let (wqp, _g2) = wq_norm.device_ptr(&self.stream);
        let (wkp, _g3) = wk_norm.device_ptr(&self.stream);
        let (qop, _g4) = q_out.device_ptr_mut(&self.stream);
        let (kp, _g5) = kc.device_ptr_mut(&self.stream);
        let (vp, _g6) = vc.device_ptr_mut(&self.stream);
        let (pp, _g7) = positions.device_ptr(&self.stream);
        let sg = slots.map(|s| s.device_ptr(&self.stream));
        let fg = factors.map(|s| s.device_ptr(&self.stream));
        let bg = block_tables.map(|s| s.device_ptr(&self.stream));
        let opt = |g: &Option<(u64, _)>| {
            g.as_ref()
                .map_or(core::ptr::null(), |(p, _)| *p as *const core::ffi::c_void)
        };
        // SAFETY: ABI contract; packed-bf16 planes sized
        // [batch * (qkv_stride | dims)] elements at 2 bytes each
        check(unsafe {
            f(
                qkvp as *mut _,
                kptr as *mut _,
                vptr as *mut _,
                wqp as *const _,
                wkp as *const _,
                qop as *mut _,
                kp as *mut _,
                vp as *mut _,
                pp as *const _,
                opt(&sg),
                opt(&fg),
                opt(&bg),
                bps as u32,
                n_head as u32,
                n_kv as u32,
                head_dim as u32,
                max_ctx as u32,
                batch as u32,
                eps,
                theta_scale,
                kv_dtype as u32,
                qkv_stride as u32,
                freq_scale,
                neox as u32,
                vnorm as u32,
                self.stream_ptr(),
            )
        })
    }

    pub fn has_gemma_qkv_nra3_b16(&self) -> bool {
        self.kernels.gemma_qkv_nra3_b16.is_some()
    }

    /// e4m3 GEMV over f8w planes (f32 x in, f32 y out at `y_off` elements -
    /// the offset serves the concatenated [gate|up] decode lane).
    pub fn f8_gemv_at(
        &self,
        w: &RepackedMxfp4,
        x: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        y_off: usize,
        in_dim: usize,
        out_dim: usize,
    ) -> Result<(), GpuError> {
        let f = self.kernels.f8_gemv.ok_or(GpuError::MissingOp("f8_gemv"))?;
        let (dp, _g1) = w.data.device_ptr(&self.stream);
        let (sp, _g2) = w.scale.device_ptr(&self.stream);
        let (xp, _g3) = x.device_ptr(&self.stream);
        let (yp, _g4) = y.device_ptr_mut(&self.stream);
        let yptr = yp + (y_off * 4) as u64;
        check(unsafe {
            f(
                dp as *const _,
                sp as *const _,
                core::ptr::null(),
                xp as *const _,
                yptr as *mut _,
                in_dim as u32,
                out_dim as u32,
                self.stream_ptr(),
            )
        })
    }

    /// f8_gemv_at over a ROW-OFFSET sub-view of a fused plane (row_off
    /// output rows in): plain out-row-major layout makes the sub-plane just
    /// pointer offsets (data += row_off*in, scale += row_off*in/32).
    #[allow(clippy::too_many_arguments)]
    pub fn f8_gemv_at_off(
        &self,
        w: &RepackedMxfp4,
        row_off: usize,
        x: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        y_off: usize,
        in_dim: usize,
        out_dim: usize,
    ) -> Result<(), GpuError> {
        let f = self.kernels.f8_gemv.ok_or(GpuError::MissingOp("f8_gemv"))?;
        let (dp, _g1) = w.data.device_ptr(&self.stream);
        let (sp, _g2) = w.scale.device_ptr(&self.stream);
        let (xp, _g3) = x.device_ptr(&self.stream);
        let (yp, _g4) = y.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                (dp + (row_off * in_dim) as u64) as *const _,
                (sp + (row_off * (in_dim / 32)) as u64) as *const _,
                core::ptr::null(),
                xp as *const _,
                (yp + (y_off * 4) as u64) as *mut _,
                in_dim as u32,
                out_dim as u32,
                self.stream_ptr(),
            )
        })
    }

    /// fp4 (e2m1) GEMV over a packed mxfp4 plane - f8_gemv_at's contract
    /// (f32 x, f32 y at y_off), half the weight bytes. row_off selects a
    /// sub-plane of a fused plane (packed rows = in_dim/2 bytes).
    #[allow(clippy::too_many_arguments)]
    pub fn fp4_gemv_at_off(
        &self,
        w: &RepackedMxfp4,
        row_off: usize,
        x: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        y_off: usize,
        in_dim: usize,
        out_dim: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .fp4_gemv
            .ok_or(GpuError::MissingOp("fp4_gemv"))?;
        let (dp, _g1) = w.data.device_ptr(&self.stream);
        let (sp, _g2) = w.scale.device_ptr(&self.stream);
        let (xp, _g3) = x.device_ptr(&self.stream);
        let (yp, _g4) = y.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                (dp + (row_off * (in_dim / 2)) as u64) as *const _,
                (sp + (row_off * (in_dim / 32)) as u64) as *const _,
                core::ptr::null(),
                xp as *const _,
                (yp + (y_off * 4) as u64) as *mut _,
                in_dim as u32,
                out_dim as u32,
                self.stream_ptr(),
            )
        })
    }

    /// fp4 mma_ks twin - f8_gemm_mma_ks's contract (part >= 8*out*batch f32).
    #[allow(clippy::too_many_arguments)]
    pub fn fp4_gemm_mma_ks(
        &self,
        w: &RepackedMxfp4,
        xq: &CudaSlice<i8>,
        xs: &CudaSlice<u8>,
        part: &mut CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        in_dim: usize,
        out_dim: usize,
        batch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .fp4_gemm_mma_ks
            .ok_or(GpuError::MissingOp("fp4_gemm_mma_ks"))?;
        let (wdp, _g1) = w.data.device_ptr(&self.stream);
        let (wsp, _g2) = w.scale.device_ptr(&self.stream);
        let (xqp, _g3) = xq.device_ptr(&self.stream);
        let (xsp, _g4) = xs.device_ptr(&self.stream);
        let (pp, _g5) = part.device_ptr_mut(&self.stream);
        let (yp, _g6) = y.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                wdp as *const _,
                wsp as *const _,
                xqp as *const _,
                xsp as *const _,
                pp as *mut _,
                yp as *mut _,
                in_dim as u32,
                out_dim as u32,
                batch as u32,
                self.stream_ptr(),
            )
        })
    }

    /// mxfp4_gemm_bs over a ROW-OFFSET sub-view (the fp4 prefill/TMA lane;
    /// with PADDOCK_FP4_TMA=1 the pack routes this to the fp4-TMA kernel).
    #[allow(clippy::too_many_arguments)]
    pub fn mxfp4_gemm_bs_off(
        &self,
        w: &RepackedMxfp4,
        row_off: usize,
        xq: &CudaSlice<i8>,
        xs: &CudaSlice<u8>,
        y: &mut CudaSlice<f32>,
        in_dim: usize,
        out_dim: usize,
        batch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .mxfp4_gemm_bs
            .ok_or(GpuError::MissingOp("mxfp4_gemm_bs"))?;
        let (wdp, _g1) = w.data.device_ptr(&self.stream);
        let (wsp, _g2) = w.scale.device_ptr(&self.stream);
        let (xqp, _g3) = xq.device_ptr(&self.stream);
        let (xsp, _g4) = xs.device_ptr(&self.stream);
        let (yp, _g5) = y.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                (wdp + (row_off * (in_dim / 2)) as u64) as *const _,
                (wsp + (row_off * (in_dim / 32)) as u64) as *const _,
                xqp as *const _,
                xsp as *const _,
                yp as *mut _,
                in_dim as u32,
                out_dim as u32,
                batch as u32,
                self.stream_ptr(),
            )
        })
    }

    pub fn has_fp4_ladder(&self) -> bool {
        self.kernels.fp4_gemv.is_some()
            && self.kernels.fp4_gemm_mma_ks.is_some()
            && self.kernels.mxfp4_gemm_bs.is_some()
    }

    /// batched e4m3 GEMV (2..16 rows, weights read once for each block).
    pub fn f8_gemv_batch(
        &self,
        w: &RepackedMxfp4,
        x: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        in_dim: usize,
        out_dim: usize,
        batch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .f8_gemv_batch
            .ok_or(GpuError::MissingOp("f8_gemv_batch"))?;
        let (dp, _g1) = w.data.device_ptr(&self.stream);
        let (sp, _g2) = w.scale.device_ptr(&self.stream);
        let (xp, _g3) = x.device_ptr(&self.stream);
        let (yp, _g4) = y.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                dp as *const _,
                sp as *const _,
                xp as *const _,
                yp as *mut _,
                in_dim as u32,
                out_dim as u32,
                batch as u32,
                self.stream_ptr(),
            )
        })
    }

    pub fn has_f8_gemv(&self) -> bool {
        self.kernels.f8_gemv.is_some() && self.kernels.f8_gemv_batch.is_some()
    }

    pub fn softcap(&self, x: &mut CudaSlice<f32>, n: usize, cap: f32) -> Result<(), GpuError> {
        let f = self.kernels.softcap.ok_or(GpuError::MissingOp("softcap"))?;
        let (xp, _g) = x.device_ptr_mut(&self.stream);
        check(unsafe { f(xp as *mut _, n as u32, cap, self.stream_ptr()) })
    }

    /// Gemma4 vision 2D rope: two independent NEOX blocks per head - dims
    /// [0,hd/2) roped by pos_x, [hd/2,hd) by pos_y, pairs (i, i+hd/4).
    #[allow(clippy::too_many_arguments)]
    pub fn rope2d_neox(
        &self,
        x: &mut CudaSlice<f32>,
        pos_x: &CudaSlice<u32>,
        pos_y: &CudaSlice<u32>,
        n_tokens: usize,
        n_heads: usize,
        head_dim: usize,
        theta_scale: f32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .rope2d_neox
            .ok_or(GpuError::MissingOp("rope2d_neox"))?;
        let (xp, _g1) = x.device_ptr_mut(&self.stream);
        let (pxp, _g2) = pos_x.device_ptr(&self.stream);
        let (pyp, _g3) = pos_y.device_ptr(&self.stream);
        check(unsafe {
            f(
                xp as *mut _,
                pxp as *const _,
                pyp as *const _,
                n_tokens as u32,
                n_heads as u32,
                head_dim as u32,
                theta_scale,
                self.stream_ptr(),
            )
        })
    }

    /// [`Self::rope2d_neox`] with the pair layout as an argument: `neox=true`
    /// pairs `(i, i+hd/4)` (gemma4v), `neox=false` pairs `(2i, 2i+1)`
    /// (muse-glimmer). Both halves stay independent and width-then-height.
    ///
    /// The layout is the reference clip graph's `ggml_rope_ext` mode, i.e. an
    /// architecture constant - no GGUF key carries it, and getting it wrong is
    /// silent (the tower still produces plausible features).
    #[allow(clippy::too_many_arguments)]
    pub fn rope2d(
        &self,
        x: &mut CudaSlice<f32>,
        pos_x: &CudaSlice<u32>,
        pos_y: &CudaSlice<u32>,
        n_tokens: usize,
        n_heads: usize,
        head_dim: usize,
        theta_scale: f32,
        neox: bool,
    ) -> Result<(), GpuError> {
        let f = self.kernels.rope2d.ok_or(GpuError::MissingOp("rope2d"))?;
        let (xp, _g1) = x.device_ptr_mut(&self.stream);
        let (pxp, _g2) = pos_x.device_ptr(&self.stream);
        let (pyp, _g3) = pos_y.device_ptr(&self.stream);
        check(unsafe {
            f(
                xp as *mut _,
                pxp as *const _,
                pyp as *const _,
                n_tokens as u32,
                n_heads as u32,
                head_dim as u32,
                theta_scale,
                u32::from(neox),
                self.stream_ptr(),
            )
        })
    }

    /// Pixel-shuffle merge: `out[o][c*k + s] = src[idx[o*k + s]][c]`.
    ///
    /// CHANNEL-outer, which is what makes it different from qwen3vl's merger
    /// (`out[o][s*width + c]`, spatial-outer) as well as from
    /// [`Self::gather_rows_avg`] with `k == 4` (which POOLS the k rows).
    /// `idx` holds `rows * k` i32 source row indices.
    pub fn pixel_shuffle_rows(
        &self,
        src: &CudaSlice<f32>,
        idx: &CudaSlice<u32>,
        out: &mut CudaSlice<f32>,
        rows: usize,
        k: usize,
        width: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .pixel_shuffle_rows
            .ok_or(GpuError::MissingOp("pixel_shuffle_rows"))?;
        let (sp, _g1) = src.device_ptr(&self.stream);
        let (ip, _g2) = idx.device_ptr(&self.stream);
        let (op, _g3) = out.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                sp as *const _,
                ip as *const _,
                op as *mut _,
                rows as u32,
                k as u32,
                width as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Non-causal ViT attention: q/k/v [n, heads, hd] -> out [n, heads*hd].
    #[allow(clippy::too_many_arguments)]
    pub fn vision_attn(
        &self,
        q: &CudaSlice<f32>,
        k: &CudaSlice<f32>,
        v: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        n: usize,
        n_heads: usize,
        head_dim: usize,
        scale: f32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .vision_attn
            .ok_or(GpuError::MissingOp("vision_attn"))?;
        let (qp, _g1) = q.device_ptr(&self.stream);
        let (kp, _g2) = k.device_ptr(&self.stream);
        let (vp, _g3) = v.device_ptr(&self.stream);
        let (op, _g4) = out.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                qp as *const _,
                kp as *const _,
                vp as *const _,
                op as *mut _,
                n as u32,
                n_heads as u32,
                head_dim as u32,
                scale,
                self.stream_ptr(),
            )
        })
    }

    /// Batched cross/self attention over `n_batch` independent groups: q/out are
    /// `[n_batch, nq, heads, hd]`, k/v are `[n_batch, nkv, heads, hd]`. Self-
    /// attention passes `nq == nkv` with the same buffer three times. One launch
    /// covers every group - granite-vision's per-window Q-Former shapes (16
    /// queries over 16 or 64 keys) are far too small to fill the GPU alone.
    #[allow(clippy::too_many_arguments)]
    pub fn vision_attn_x(
        &self,
        q: &CudaSlice<f32>,
        k: &CudaSlice<f32>,
        v: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        nq: usize,
        nkv: usize,
        n_heads: usize,
        head_dim: usize,
        n_batch: usize,
        scale: f32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .vision_attn_x
            .ok_or(GpuError::MissingOp("vision_attn_x"))?;
        let (qp, _g1) = q.device_ptr(&self.stream);
        let (kp, _g2) = k.device_ptr(&self.stream);
        let (vp, _g3) = v.device_ptr(&self.stream);
        let (op, _g4) = out.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                qp as *const _,
                kp as *const _,
                vp as *const _,
                op as *mut _,
                nq as u32,
                nkv as u32,
                n_heads as u32,
                head_dim as u32,
                n_batch as u32,
                scale,
                self.stream_ptr(),
            )
        })
    }

    /// [`Self::vision_attn_x`] over a row window - `n_batch` consecutive groups
    /// of `n` rows each, starting at `row_off`.
    ///
    /// Muse-glimmer's window attention is a block-diagonal mask over a permuted
    /// token order, and a block-diagonal mask over CONTIGUOUS runs is exactly a
    /// batch of independent attentions. So the mask never materializes: the
    /// windows are laid out consecutively at preprocess time and each run of
    /// equal-sized windows becomes one launch. Ragged edge windows are their own
    /// (smaller) runs - hence the offset, which `vision_attn_x` lacks.
    #[allow(clippy::too_many_arguments)]
    pub fn vision_attn_x_at(
        &self,
        q: &CudaSlice<f32>,
        k: &CudaSlice<f32>,
        v: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        row_off: usize,
        n: usize,
        n_heads: usize,
        head_dim: usize,
        n_batch: usize,
        scale: f32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .vision_attn_x
            .ok_or(GpuError::MissingOp("vision_attn_x"))?;
        let off = (row_off * n_heads * head_dim * std::mem::size_of::<f32>()) as u64;
        let (qp, _g1) = q.device_ptr(&self.stream);
        let (kp, _g2) = k.device_ptr(&self.stream);
        let (vp, _g3) = v.device_ptr(&self.stream);
        let (op, _g4) = out.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                (qp + off) as *const _,
                (kp + off) as *const _,
                (vp + off) as *const _,
                (op + off) as *mut _,
                n as u32,
                n as u32,
                n_heads as u32,
                head_dim as u32,
                n_batch as u32,
                scale,
                self.stream_ptr(),
            )
        })
    }

    /// `vision_attn` over rows [row_off, row_off+n) of batched q/k/v/out buffers
    /// - the per-image attention call inside a multi-image encode (each image's
    ///   patches attend only among themselves; every other tower op is
    ///   row-independent, so batching needs only this windowed view).
    #[allow(clippy::too_many_arguments)]
    pub fn vision_attn_at(
        &self,
        q: &CudaSlice<f32>,
        k: &CudaSlice<f32>,
        v: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        row_off: usize,
        n: usize,
        n_heads: usize,
        head_dim: usize,
        scale: f32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .vision_attn
            .ok_or(GpuError::MissingOp("vision_attn"))?;
        let off = (row_off * n_heads * head_dim * std::mem::size_of::<f32>()) as u64;
        let (qp, _g1) = q.device_ptr(&self.stream);
        let (kp, _g2) = k.device_ptr(&self.stream);
        let (vp, _g3) = v.device_ptr(&self.stream);
        let (op, _g4) = out.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                (qp + off) as *const _,
                (kp + off) as *const _,
                (vp + off) as *const _,
                (op + off) as *mut _,
                n as u32,
                n_heads as u32,
                head_dim as u32,
                scale,
                self.stream_ptr(),
            )
        })
    }

    /// SAM ViTDet attention with the decomposed relative-position bias
    /// (DeepSeek-OCR's first tower):
    /// `out = softmax(q·kᵀ·scale + rel_h + rel_w)·v`, the bias being the RAW
    /// (unscaled) query contracted against per-axis tables.
    ///
    /// q/k/v/out are `[n_batch, side², heads, hd]` f32. `rh`/`rw` are
    /// `[side, side, hd]` f32 bias tables - `get_rel_pos` already applied on
    /// the host, so they are indexed `[qy][ky]` absolutely, and they are
    /// shared by every batch element (rel-pos is relative within a window).
    /// Windowed SAM blocks pass `n_batch` = windows with `side` = 14; global
    /// blocks pass `n_batch` = views with `side` = the full grid.
    #[allow(clippy::too_many_arguments)]
    pub fn sam_attn(
        &self,
        q: &CudaSlice<f32>,
        k: &CudaSlice<f32>,
        v: &CudaSlice<f32>,
        rh: &CudaSlice<f32>,
        rw: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        n_batch: usize,
        side: usize,
        n_heads: usize,
        head_dim: usize,
        scale: f32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .sam_attn
            .ok_or(GpuError::MissingOp("sam_attn"))?;
        let (qp, _g1) = q.device_ptr(&self.stream);
        let (kp, _g2) = k.device_ptr(&self.stream);
        let (vp, _g3) = v.device_ptr(&self.stream);
        let (rhp, _g4) = rh.device_ptr(&self.stream);
        let (rwp, _g5) = rw.device_ptr(&self.stream);
        let (op, _g6) = out.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                qp as *const _,
                kp as *const _,
                vp as *const _,
                rhp as *const _,
                rwp as *const _,
                op as *mut _,
                n_batch as u32,
                side as u32,
                n_heads as u32,
                head_dim as u32,
                scale,
                self.stream_ptr(),
            )
        })
    }

    /// Multi-column (2..=8) dp4a GEMV - llama's mmvq shape: one block per output
    /// row, weight streamed once, int8 activation columns. The B=2..8 decode
    /// matmul (gemv-class DRAM efficiency where the tiled MT kernel is latency-bound).
    pub fn q8_0_gemv_dp4a_nc(
        &self,
        w: &RepackedQ8,
        xq: &CudaSlice<i8>,
        xs: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        ncols: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .q8_0_gemv_dp4a_nc
            .ok_or(GpuError::MissingOp("q8_0_gemv_dp4a_nc"))?;
        let (in_dim, out_dim) = (w.dims[0], w.dims[1]);
        let (dp, _g1) = w.data.device_ptr(&self.stream);
        let (scp, _g2) = w.scale.device_ptr(&self.stream);
        let (xqp, _g3) = xq.device_ptr(&self.stream);
        let (xsp, _g4) = xs.device_ptr(&self.stream);
        let (yp, _g5) = y.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                dp as *const _,
                scp as *const _,
                xqp as *const _,
                xsp as *const _,
                yp as *mut _,
                in_dim as u32,
                out_dim as u32,
                ncols as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Bias-carrying single/multi-column repacked dp4a GEMV - the nc kernel
    /// body plus a nullable bias. The B=1 decode projections use this with the
    /// repacked weights: block-per-row fills big dies where the warp-per-row
    /// 34-byte-block GEMV tops out under half of DRAM.
    pub fn q8_0_gemv_dp4a_nc_b(
        &self,
        w: &RepackedQ8,
        bias: Option<&CudaSlice<f32>>,
        xq: &CudaSlice<i8>,
        xs: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        ncols: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .q8_0_gemv_dp4a_nc_b
            .ok_or(GpuError::MissingOp("q8_0_gemv_dp4a_nc_b"))?;
        let (in_dim, out_dim) = (w.dims[0], w.dims[1]);
        let (dp, _g1) = w.data.device_ptr(&self.stream);
        let (scp, _g2) = w.scale.device_ptr(&self.stream);
        let (xqp, _g3) = xq.device_ptr(&self.stream);
        let (xsp, _g4) = xs.device_ptr(&self.stream);
        let (yp, _g5) = y.device_ptr_mut(&self.stream);
        let (bias_ptr, _gb);
        let bp: *const core::ffi::c_void = match bias {
            Some(b) => {
                (bias_ptr, _gb) = b.device_ptr(&self.stream);
                bias_ptr as *const _
            }
            None => core::ptr::null(),
        };
        // SAFETY: ABI contract; activation pre-quantized to [ncols, in_dim]
        check(unsafe {
            f(
                dp as *const _,
                scp as *const _,
                bp,
                xqp as *const _,
                xsp as *const _,
                yp as *mut _,
                in_dim as u32,
                out_dim as u32,
                ncols as u32,
                self.stream_ptr(),
            )
        })
    }

    /// f32 -> e4m3 + ue8m0 per-32 activation quantize (block-scale mma B side).
    pub fn quantize_e4m3(
        &self,
        x: &CudaSlice<f32>,
        q: &mut CudaSlice<i8>,
        scale: &mut CudaSlice<u8>,
        n: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .quantize_e4m3
            .ok_or(GpuError::MissingOp("quantize_e4m3"))?;
        let (xp, _g1) = x.device_ptr(&self.stream);
        let (qp, _g2) = q.device_ptr_mut(&self.stream);
        let (sp, _g3) = scale.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; n % 32 == 0, buffers sized [n]/[n/32]
        check(unsafe {
            f(
                xp as *const _,
                qp as *mut _,
                sp as *mut _,
                n as u32,
                self.stream_ptr(),
            )
        })
    }

    /// f16-in twin of [`Self::quantize_e4m3`] (attention streams):
    /// x is an f16 plane held in the f32-typed scratch.
    pub fn quantize_e4m3_f16in(
        &self,
        x: &CudaSlice<f32>,
        q: &mut CudaSlice<i8>,
        scale: &mut CudaSlice<u8>,
        n: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .quantize_e4m3_i16
            .ok_or(GpuError::MissingOp("quantize_e4m3_i16"))?;
        let (xp, _g1) = x.device_ptr(&self.stream);
        let (qp, _g2) = q.device_ptr_mut(&self.stream);
        let (sp, _g3) = scale.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; x holds f16 in the f32-typed scratch
        check(unsafe {
            f(
                xp as *const _,
                qp as *mut _,
                sp as *mut _,
                n as u32,
                1u32,
                self.stream_ptr(),
            )
        })
    }
}
