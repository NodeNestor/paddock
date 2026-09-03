//! Whisper-family decode ops  - the flash-decoding attention over
//! f16 K/V slot planes plus the fused epilogues that collapse the decode
//! step's launch train. Kernel side: `packs/cuda/src/asr/whisper.cuh`.
//!
//! Everything here is batched over decode SLOTS: `q`/`out` are compact
//! `[batch, ...]` in active order, while the K/V planes are `[cap, stride, d]`
//! addressed through a `slots` index vector - so a finished slot leaves the
//! active set without any cache moving.

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use half::f16;

use super::error::*;
use super::*;

// The K/V planes are BYTE buffers sized `elems * kv_dtype.bytes()`, the same
// shape every other family's cache has: one allocation that means f16 or
// fp8-e4m3 depending on the flag riding the launcher, rather than two typed
// pools and a branch at every call site.

/// Per-row flags for `whisper_ts_rules` - must match `PD_WTS_*` in
/// `packs/cuda/src/asr/whisper.cuh`. They describe three facts about the
/// tokens sampled so FAR, which is all whisper's timestamp grammar needs.
pub mod ts_flags {
    /// this row wants timestamps (a batch can mix)
    pub const ON: u32 = 1;
    /// nothing sampled yet - the window's first emitted token
    pub const BEGIN: u32 = 2;
    /// the last sampled token was a timestamp
    pub const LAST: u32 = 4;
    /// the one before it was - or there is only one, which the reference
    /// counts as true so a lone opening timestamp is followed by text
    pub const PENULT: u32 = 8;
    /// at least one timestamp has been sampled, so the floor is meaningful
    pub const HAVE: u32 = 16;
}

impl GpuExecutor {
    /// Single-query attention for every active slot at once. `lens` gives the
    /// live key count as `lens[b] + len_bias` (self-attention passes the
    /// position cursor with bias 1); `None` uses `kv_len` for all slots.
    /// `part` is the split-partial scratch - pass `None` to force the
    /// single-chunk form.
    #[allow(clippy::too_many_arguments)]
    pub fn whisper_dec_attn(
        &self,
        q: &CudaSlice<f32>,
        qbias: Option<&CudaSlice<f32>>,
        k: &CudaSlice<u8>,
        v: &CudaSlice<u8>,
        slots: &CudaSlice<u32>,
        lens: Option<&CudaSlice<u32>>,
        out: &mut CudaSlice<f16>,
        part: Option<&mut CudaSlice<f32>>,
        kv_stride: usize,
        kv_len: usize,
        len_bias: usize,
        n_heads: usize,
        head_dim: usize,
        batch: usize,
        scale: f32,
        kv_dtype: KvDtype,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .whisper_dec_attn
            .ok_or(GpuError::MissingOp("whisper_dec_attn"))?;
        let (qp, _g1) = q.device_ptr(&self.stream);
        let (kp, _g2) = k.device_ptr(&self.stream);
        let (vp, _g3) = v.device_ptr(&self.stream);
        let (sp, _g4) = slots.device_ptr(&self.stream);
        let (op, _g5) = out.device_ptr_mut(&self.stream);
        let (qbp, _gq);
        let qbias_ptr: *const core::ffi::c_void = match qbias {
            Some(b) => {
                (qbp, _gq) = b.device_ptr(&self.stream);
                qbp as *const _
            }
            None => core::ptr::null(),
        };
        let (lp, _gl);
        let lens_ptr: *const core::ffi::c_void = match lens {
            Some(l) => {
                (lp, _gl) = l.device_ptr(&self.stream);
                lp as *const _
            }
            None => core::ptr::null(),
        };
        let (pp, _gp);
        let part_ptr: *mut core::ffi::c_void = match part {
            Some(p) => {
                (pp, _gp) = p.device_ptr_mut(&self.stream);
                pp as *mut _
            }
            None => core::ptr::null_mut(),
        };
        check(unsafe {
            f(
                qp as *const _,
                qbias_ptr,
                kp as *const _,
                vp as *const _,
                sp as *const _,
                lens_ptr,
                op as *mut _,
                part_ptr,
                kv_stride as u32,
                kv_len as u32,
                len_bias as u32,
                n_heads as u32,
                head_dim as u32,
                batch as u32,
                scale,
                kv_dtype as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Whisper's `ApplyTimestampRules`, in place on the logits, before the
    /// greedy pick. `state` is `[rows, 2]` u32: `{flags, lowest allowed
    /// timestamp id}`, flags built from `ts_flags::*`. `max_init` is the
    /// largest initial-timestamp OFFSET from `ts_begin`, or `u32::MAX` for no
    /// limit.
    ///
    /// Rows without `ts_flags::ON` are left untouched, so a batch mixing
    /// timestamped and plain requests runs one launch and only the rows that
    /// asked for times are constrained.
    #[allow(clippy::too_many_arguments)]
    pub fn whisper_ts_rules(
        &self,
        logits: &mut CudaSlice<f32>,
        state: &CudaSlice<u32>,
        rows: usize,
        vocab: usize,
        eot: u32,
        no_ts: u32,
        ts_begin: u32,
        max_init: u32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .whisper_ts_rules
            .ok_or(GpuError::MissingOp("whisper_ts_rules"))?;
        let (lp, _g1) = logits.device_ptr_mut(&self.stream);
        let (sp, _g2) = state.device_ptr(&self.stream);
        check(unsafe {
            f(
                lp as *mut _,
                sp as *const _,
                rows as u32,
                vocab as u32,
                eot,
                no_ts,
                ts_begin,
                max_init,
                self.stream_ptr(),
            )
        })
    }

    /// `x[b] = tok_embd[tokens[b]] + dec_pos[pos[b]]` for every active slot.
    pub fn whisper_embed_pos(
        &self,
        tok: &CudaSlice<f32>,
        postab: &CudaSlice<f32>,
        tokens: &CudaSlice<u32>,
        pos: &CudaSlice<u32>,
        x: &mut CudaSlice<f32>,
        d: usize,
        batch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .whisper_embed_pos
            .ok_or(GpuError::MissingOp("whisper_embed_pos"))?;
        let (tp, _g1) = tok.device_ptr(&self.stream);
        let (pp, _g2) = postab.device_ptr(&self.stream);
        let (kp, _g3) = tokens.device_ptr(&self.stream);
        let (op, _g4) = pos.device_ptr(&self.stream);
        let (xp, _g5) = x.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                tp as *const _,
                pp as *const _,
                kp as *const _,
                op as *const _,
                xp as *mut _,
                d as u32,
                batch as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Split the merged `[batch, 3*d]` q|k|v landing and append this step's
    /// K/V to the slot caches at `kv_dtype`. `bq`/`bv` may be `None` - whisper's
    /// k_proj genuinely carries no bias.
    #[allow(clippy::too_many_arguments)]
    pub fn whisper_qkv_split(
        &self,
        qkv: &CudaSlice<f32>,
        bq: Option<&CudaSlice<f32>>,
        bv: Option<&CudaSlice<f32>>,
        q: &mut CudaSlice<f32>,
        kc: &mut CudaSlice<u8>,
        vc: &mut CudaSlice<u8>,
        slots: &CudaSlice<u32>,
        pos: &CudaSlice<u32>,
        d: usize,
        ctx: usize,
        batch: usize,
        kv_dtype: KvDtype,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .whisper_qkv_split
            .ok_or(GpuError::MissingOp("whisper_qkv_split"))?;
        let (xp, _g1) = qkv.device_ptr(&self.stream);
        let (qp, _g2) = q.device_ptr_mut(&self.stream);
        let (kp, _g3) = kc.device_ptr_mut(&self.stream);
        let (vp, _g4) = vc.device_ptr_mut(&self.stream);
        let (sp, _g5) = slots.device_ptr(&self.stream);
        let (pp, _g6) = pos.device_ptr(&self.stream);
        let (bqp, _gq);
        let bq_ptr: *const core::ffi::c_void = match bq {
            Some(b) => {
                (bqp, _gq) = b.device_ptr(&self.stream);
                bqp as *const _
            }
            None => core::ptr::null(),
        };
        let (bvp, _gv);
        let bv_ptr: *const core::ffi::c_void = match bv {
            Some(b) => {
                (bvp, _gv) = b.device_ptr(&self.stream);
                bvp as *const _
            }
            None => core::ptr::null(),
        };
        check(unsafe {
            f(
                xp as *const _,
                bq_ptr,
                bv_ptr,
                qp as *mut _,
                kp as *mut _,
                vp as *mut _,
                sp as *const _,
                pp as *const _,
                d as u32,
                ctx as u32,
                batch as u32,
                kv_dtype as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Store a window's cross-attention K or V into its slot plane at
    /// `kv_dtype`.
    #[allow(clippy::too_many_arguments)]
    pub fn whisper_kv_store(
        &self,
        src: &CudaSlice<f32>,
        bias: Option<&CudaSlice<f32>>,
        dst: &mut CudaSlice<u8>,
        slots: &CudaSlice<u32>,
        rows: usize,
        d: usize,
        stride: usize,
        batch: usize,
        kv_dtype: KvDtype,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .whisper_kv_store
            .ok_or(GpuError::MissingOp("whisper_kv_store"))?;
        let (sp, _g1) = src.device_ptr(&self.stream);
        let (dp, _g2) = dst.device_ptr_mut(&self.stream);
        let (lp, _g3) = slots.device_ptr(&self.stream);
        let (bp, _gb);
        let bias_ptr: *const core::ffi::c_void = match bias {
            Some(b) => {
                (bp, _gb) = b.device_ptr(&self.stream);
                bp as *const _
            }
            None => core::ptr::null(),
        };
        check(unsafe {
            f(
                sp as *const _,
                bias_ptr,
                dp as *mut _,
                lp as *const _,
                rows as u32,
                d as u32,
                stride as u32,
                batch as u32,
                kv_dtype as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Split the encoder's FUSED `[rows, 3*d]` q|k|v GEMM landing into the
    /// three planes attention consumes, `bq`/`bv` folded in (k_proj carries
    /// no bias - architecture). Exists so the encoder's three per-layer
    /// projections run as one M=3*d GEMM (the split GEMMs left
    /// half the clusters idle at 1500 rows).
    #[allow(clippy::too_many_arguments)]
    pub fn whisper_enc_qkv_split(
        &self,
        qkv: &CudaSlice<f32>,
        bq: Option<&CudaSlice<f32>>,
        bv: Option<&CudaSlice<f32>>,
        q: &mut CudaSlice<f32>,
        k: &mut CudaSlice<f32>,
        v: &mut CudaSlice<f32>,
        d: usize,
        rows: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .whisper_enc_qkv_split
            .ok_or(GpuError::MissingOp("whisper_enc_qkv_split"))?;
        let (xp, _g1) = qkv.device_ptr(&self.stream);
        let (qp, _g2) = q.device_ptr_mut(&self.stream);
        let (kp, _g3) = k.device_ptr_mut(&self.stream);
        let (vp, _g4) = v.device_ptr_mut(&self.stream);
        let (bqp, _gq);
        let bq_ptr: *const core::ffi::c_void = match bq {
            Some(b) => {
                (bqp, _gq) = b.device_ptr(&self.stream);
                bqp as *const _
            }
            None => core::ptr::null(),
        };
        let (bvp, _gv);
        let bv_ptr: *const core::ffi::c_void = match bv {
            Some(b) => {
                (bvp, _gv) = b.device_ptr(&self.stream);
                bvp as *const _
            }
            None => core::ptr::null(),
        };
        check(unsafe {
            f(
                xp as *const _,
                bq_ptr,
                bv_ptr,
                qp as *mut _,
                kp as *mut _,
                vp as *mut _,
                d as u32,
                rows as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Store every decoder layer's cross-attention K (or V) slot plane from
    /// one layer-batched `[rows, n_layer*d]` GEMM landing (- the 64
    /// per-layer cross GEMMs share the encoder states, so the runner batches
    /// them into one call per plane set). `dsts` is the device array of
    /// per-layer plane base pointers (uploaded once at pool alloc), `bias`
    /// the concatenated `[n_layer*d]` plane (V) or `None` (K).
    #[allow(clippy::too_many_arguments)]
    pub fn whisper_kv_store_batch(
        &self,
        src: &CudaSlice<f32>,
        bias: Option<&CudaSlice<f32>>,
        dsts: &CudaSlice<u64>,
        slots: &CudaSlice<u32>,
        rows: usize,
        d: usize,
        n_layer: usize,
        stride: usize,
        kv_dtype: KvDtype,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .whisper_kv_store_batch
            .ok_or(GpuError::MissingOp("whisper_kv_store_batch"))?;
        let (sp, _g1) = src.device_ptr(&self.stream);
        let (dp, _g2) = dsts.device_ptr(&self.stream);
        let (lp, _g3) = slots.device_ptr(&self.stream);
        let (bp, _gb);
        let bias_ptr: *const core::ffi::c_void = match bias {
            Some(b) => {
                (bp, _gb) = b.device_ptr(&self.stream);
                bp as *const _
            }
            None => core::ptr::null(),
        };
        check(unsafe {
            f(
                sp as *const _,
                bias_ptr,
                dp as *const _,
                lp as *const _,
                rows as u32,
                d as u32,
                n_layer as u32,
                stride as u32,
                kv_dtype as u32,
                self.stream_ptr(),
            )
        })
    }

    /// [`Self::whisper_kv_store_batch`] off an AUDIO-MAJOR batched landing
    /// row r stores into `slots[r / rows_per_slot]` at row
    /// `r % rows_per_slot` of that slot's plane.
    #[allow(clippy::too_many_arguments)]
    pub fn whisper_kv_store_slots(
        &self,
        src: &CudaSlice<f32>,
        bias: Option<&CudaSlice<f32>>,
        dsts: &CudaSlice<u64>,
        slots: &CudaSlice<u32>,
        rows: usize,
        d: usize,
        n_layer: usize,
        stride: usize,
        kv_dtype: KvDtype,
        rows_per_slot: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .whisper_kv_store_slots
            .ok_or(GpuError::MissingOp("whisper_kv_store_slots"))?;
        let (sp, _g1) = src.device_ptr(&self.stream);
        let (dp, _g2) = dsts.device_ptr(&self.stream);
        let (lp, _g3) = slots.device_ptr(&self.stream);
        let (bp, _gb);
        let bias_ptr: *const core::ffi::c_void = match bias {
            Some(b) => {
                (bp, _gb) = b.device_ptr(&self.stream);
                bp as *const _
            }
            None => core::ptr::null(),
        };
        check(unsafe {
            f(
                sp as *const _,
                bias_ptr,
                dp as *const _,
                lp as *const _,
                rows as u32,
                d as u32,
                n_layer as u32,
                stride as u32,
                kv_dtype as u32,
                rows_per_slot as u32,
                self.stream_ptr(),
            )
        })
    }

    /// LayerNorm with an f16 landing (same reduction structure as
    /// [`Self::layernorm`], the cast folded in).
    #[allow(clippy::too_many_arguments)]
    pub fn whisper_ln_f16(
        &self,
        x: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        out: &mut CudaSlice<f16>,
        rows: usize,
        n: usize,
        eps: f32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .whisper_ln_f16
            .ok_or(GpuError::MissingOp("whisper_ln_f16"))?;
        let (xp, _g1) = x.device_ptr(&self.stream);
        let (wp, _g2) = w.device_ptr(&self.stream);
        let (bp, _g3) = b.device_ptr(&self.stream);
        let (op, _g4) = out.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                xp as *const _,
                wp as *const _,
                bp as *const _,
                op as *mut _,
                rows as u32,
                n as u32,
                eps,
                self.stream_ptr(),
            )
        })
    }

    /// `x += proj + bias`, then the next block's pre-norm out of the updated
    /// residual, at f16 - whisper's residual seam, one launch instead of four.
    #[allow(clippy::too_many_arguments)]
    pub fn whisper_res_ln_f16(
        &self,
        x: &mut CudaSlice<f32>,
        proj: &CudaSlice<f32>,
        bias: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        out: &mut CudaSlice<f16>,
        rows: usize,
        n: usize,
        eps: f32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .whisper_res_ln_f16
            .ok_or(GpuError::MissingOp("whisper_res_ln_f16"))?;
        let (pp, _g1) = proj.device_ptr(&self.stream);
        let (bip, _g2) = bias.device_ptr(&self.stream);
        let (wp, _g3) = w.device_ptr(&self.stream);
        let (bp, _g4) = b.device_ptr(&self.stream);
        let (xp, _g5) = x.device_ptr_mut(&self.stream);
        let (op, _g6) = out.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                xp as *mut _,
                pp as *const _,
                bip as *const _,
                wp as *const _,
                bp as *const _,
                op as *mut _,
                rows as u32,
                n as u32,
                eps,
                self.stream_ptr(),
            )
        })
    }

    /// bias + exact-erf GELU + f16 cast on the fc1 landing.
    pub fn whisper_bias_gelu_f16(
        &self,
        x: &CudaSlice<f32>,
        bias: &CudaSlice<f32>,
        out: &mut CudaSlice<f16>,
        rows: usize,
        n: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .whisper_bias_gelu_f16
            .ok_or(GpuError::MissingOp("whisper_bias_gelu_f16"))?;
        let (xp, _g1) = x.device_ptr(&self.stream);
        let (bp, _g2) = bias.device_ptr(&self.stream);
        let (op, _g3) = out.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                xp as *const _,
                bp as *const _,
                op as *mut _,
                rows as u32,
                n as u32,
                self.stream_ptr(),
            )
        })
    }

    /// True when the whole granite-speech conformer set is present - the
    /// tower is all-or-nothing, so the loader checks this once instead of
    /// each call site discovering a hole halfway through an encode.
    pub fn has_granite_speech(&self) -> bool {
        self.kernels.gs_bias_silu_f16.is_some()
            && self.kernels.gs_bias_glu.is_some()
            && self.kernels.gs_dwconv_bn_silu_f16.is_some()
            && self.kernels.gs_conf_attn.is_some()
            && self.kernels.gs_bias_softmax_f16.is_some()
            && self.kernels.gs_res_ln_f16.is_some()
            && self.kernels.gs_post_ln_f16.is_some()
    }

    /// bias + SiLU + f16 cast on a macaron FFN landing `[rows, n]`.
    pub fn gs_bias_silu_f16(
        &self,
        x: &CudaSlice<f32>,
        bias: &CudaSlice<f32>,
        out: &mut CudaSlice<f16>,
        rows: usize,
        n: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .gs_bias_silu_f16
            .ok_or(GpuError::MissingOp("gs_bias_silu_f16"))?;
        let (xp, _g1) = x.device_ptr(&self.stream);
        let (bp, _g2) = bias.device_ptr(&self.stream);
        let (op, _g3) = out.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                xp as *const _,
                bp as *const _,
                op as *mut _,
                rows as u32,
                n as u32,
                self.stream_ptr(),
            )
        })
    }

    /// bias + sigmoid-GLU over the conv module's `[rows, 2*d]` landing.
    pub fn gs_bias_glu(
        &self,
        x: &CudaSlice<f32>,
        bias: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        rows: usize,
        d: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .gs_bias_glu
            .ok_or(GpuError::MissingOp("gs_bias_glu"))?;
        let (xp, _g1) = x.device_ptr(&self.stream);
        let (bp, _g2) = bias.device_ptr(&self.stream);
        let (op, _g3) = out.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                xp as *const _,
                bp as *const _,
                op as *mut _,
                rows as u32,
                d as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Centered depthwise conv over time + folded BatchNorm + SiLU, f16 out.
    /// `w` is tap-major `[k, d]` (transposed at load).
    #[allow(clippy::too_many_arguments)]
    pub fn gs_dwconv_bn_silu_f16(
        &self,
        x: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        bnw: &CudaSlice<f32>,
        bnb: &CudaSlice<f32>,
        out: &mut CudaSlice<f16>,
        rows: usize,
        d: usize,
        k: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .gs_dwconv_bn_silu_f16
            .ok_or(GpuError::MissingOp("gs_dwconv_bn_silu_f16"))?;
        let (xp, _g1) = x.device_ptr(&self.stream);
        let (wp, _g2) = w.device_ptr(&self.stream);
        let (nwp, _g3) = bnw.device_ptr(&self.stream);
        let (nbp, _g4) = bnb.device_ptr(&self.stream);
        let (op, _g5) = out.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                xp as *const _,
                wp as *const _,
                nwp as *const _,
                nbp as *const _,
                op as *mut _,
                rows as u32,
                d as u32,
                k as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Conformer blockwise attention with Shaw relative position embeddings
    /// over a merged `[rows, 3*n_heads*hd]` q|k|v landing. `rel` is the
    /// layer's `[2*max_pos+1, hd]` table, shared by every head.
    #[allow(clippy::too_many_arguments)]
    pub fn gs_conf_attn(
        &self,
        qkv: &CudaSlice<f32>,
        out: &mut CudaSlice<f16>,
        rel: &CudaSlice<f32>,
        rows: usize,
        ctx: usize,
        n_heads: usize,
        head_dim: usize,
        max_pos: usize,
        scale: f32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .gs_conf_attn
            .ok_or(GpuError::MissingOp("gs_conf_attn"))?;
        let (qp, _g1) = qkv.device_ptr(&self.stream);
        let (rp, _g2) = rel.device_ptr(&self.stream);
        let (op, _g3) = out.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                qp as *const _,
                op as *mut _,
                rp as *const _,
                rows as u32,
                ctx as u32,
                n_heads as u32,
                head_dim as u32,
                max_pos as u32,
                scale,
                self.stream_ptr(),
            )
        })
    }

    /// bias + row softmax + f16 cast - the CTC branch head.
    pub fn gs_bias_softmax_f16(
        &self,
        x: &CudaSlice<f32>,
        bias: &CudaSlice<f32>,
        out: &mut CudaSlice<f16>,
        rows: usize,
        n: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .gs_bias_softmax_f16
            .ok_or(GpuError::MissingOp("gs_bias_softmax_f16"))?;
        let (xp, _g1) = x.device_ptr(&self.stream);
        let (bp, _g2) = bias.device_ptr(&self.stream);
        let (op, _g3) = out.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                xp as *const _,
                bp as *const _,
                op as *mut _,
                rows as u32,
                n as u32,
                self.stream_ptr(),
            )
        })
    }

    /// `x += s*(proj + bias)`, then the next block's pre-norm out of it at
    /// f16. Pass `norm = None` for a bare residual update.
    #[allow(clippy::too_many_arguments)]
    pub fn gs_res_ln_f16(
        &self,
        x: &mut CudaSlice<f32>,
        proj: &CudaSlice<f32>,
        bias: &CudaSlice<f32>,
        norm: Option<(&CudaSlice<f32>, &CudaSlice<f32>, &mut CudaSlice<f16>)>,
        rows: usize,
        n: usize,
        s: f32,
        eps: f32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .gs_res_ln_f16
            .ok_or(GpuError::MissingOp("gs_res_ln_f16"))?;
        self.gs_ln_seam(f, x, proj, bias, norm, rows, n, s, eps)
    }

    /// `x = LN(x + s*(proj + bias))` in place - the post-LN contract, where
    /// the residual stream itself becomes the normalized value. The f16
    /// landing is optional (granite's own post-norm feeds another LayerNorm).
    #[allow(clippy::too_many_arguments)]
    pub fn gs_post_ln_f16(
        &self,
        x: &mut CudaSlice<f32>,
        proj: &CudaSlice<f32>,
        bias: &CudaSlice<f32>,
        norm: (
            &CudaSlice<f32>,
            &CudaSlice<f32>,
            Option<&mut CudaSlice<f16>>,
        ),
        rows: usize,
        n: usize,
        s: f32,
        eps: f32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .gs_post_ln_f16
            .ok_or(GpuError::MissingOp("gs_post_ln_f16"))?;
        let (w, b, out) = norm;
        let (xp, _g1) = x.device_ptr_mut(&self.stream);
        let (pp, _g2) = proj.device_ptr(&self.stream);
        let (bip, _g3) = bias.device_ptr(&self.stream);
        let (wp, _g4) = w.device_ptr(&self.stream);
        let (bp, _g5) = b.device_ptr(&self.stream);
        let op = match out {
            Some(o) => o.device_ptr_mut(&self.stream).0 as *mut core::ffi::c_void,
            None => std::ptr::null_mut(),
        };
        check(unsafe {
            f(
                xp as *mut _,
                pp as *const _,
                bip as *const _,
                wp as *const _,
                bp as *const _,
                op,
                rows as u32,
                n as u32,
                s,
                eps,
                self.stream_ptr(),
            )
        })
    }

    /// Shared body for the two residual seams - they differ only in which
    /// value the residual keeps, which is a kernel-side decision.
    #[allow(clippy::too_many_arguments)]
    fn gs_ln_seam(
        &self,
        f: unsafe extern "C" fn(
            *mut core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            u32,
            f32,
            f32,
            *mut core::ffi::c_void,
        ) -> i32,
        x: &mut CudaSlice<f32>,
        proj: &CudaSlice<f32>,
        bias: &CudaSlice<f32>,
        norm: Option<(&CudaSlice<f32>, &CudaSlice<f32>, &mut CudaSlice<f16>)>,
        rows: usize,
        n: usize,
        s: f32,
        eps: f32,
    ) -> Result<(), GpuError> {
        let (xp, _g1) = x.device_ptr_mut(&self.stream);
        let (pp, _g2) = proj.device_ptr(&self.stream);
        let (bip, _g3) = bias.device_ptr(&self.stream);
        let (wp, bp, op) = match norm {
            Some((w, b, out)) => {
                let (wp, _g4) = w.device_ptr(&self.stream);
                let (bp, _g5) = b.device_ptr(&self.stream);
                let (op, _g6) = out.device_ptr_mut(&self.stream);
                (
                    wp as *const core::ffi::c_void,
                    bp as *const core::ffi::c_void,
                    op as *mut core::ffi::c_void,
                )
            }
            None => (std::ptr::null(), std::ptr::null(), std::ptr::null_mut()),
        };
        check(unsafe {
            f(
                xp as *mut _,
                pp as *const _,
                bip as *const _,
                wp,
                bp,
                op,
                rows as u32,
                n as u32,
                s,
                eps,
                self.stream_ptr(),
            )
        })
    }

    /// `softmax(QK^T)` over the encoder frames for the alignment heads of one
    /// cross-attention layer - the read-out word-level timing comes from
    ///
    /// `heads` names which of this layer's heads to dump, and `out` is
    /// `[batch, heads.len(), n_enc]` with every row summing to 1. The decode
    /// kernel never materialises these probabilities (it is flash-style and
    /// consumes them inside its online loop), so this is a second, plain pass
    /// that runs only when a caller asked for word times.
    #[allow(clippy::too_many_arguments)]
    pub fn whisper_xattn_probs(
        &self,
        q: &CudaSlice<f32>,
        qbias: Option<&CudaSlice<f32>>,
        k: &CudaSlice<u8>,
        slots: &CudaSlice<u32>,
        heads: &CudaSlice<u32>,
        out: &mut CudaSlice<f32>,
        // element offset into `out` the first row lands at - the word-timing
        // pass accumulates many chunks into one plane, and a shifted pointer
        // is cheaper than staging each chunk and copying it into place
        out_off: usize,
        kv_stride: usize,
        n_enc: usize,
        n_heads: usize,
        head_dim: usize,
        n_sel: usize,
        batch: usize,
        scale: f32,
        kv_dtype: KvDtype,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .whisper_xattn_probs
            .ok_or(GpuError::MissingOp("whisper_xattn_probs"))?;
        let (qp, _g1) = q.device_ptr(&self.stream);
        let (kp, _g2) = k.device_ptr(&self.stream);
        let (sp, _g3) = slots.device_ptr(&self.stream);
        let (hp, _g4) = heads.device_ptr(&self.stream);
        let mut ov = out
            .try_slice_mut(out_off..out_off + batch * n_sel * n_enc)
            .ok_or_else(|| GpuError::Driver("whisper_xattn_probs: out range".into()))?;
        let (op, _g5) = ov.device_ptr_mut(&self.stream);
        let (qbp, _gq);
        let qbias_ptr: *const core::ffi::c_void = match qbias {
            Some(b) => {
                (qbp, _gq) = b.device_ptr(&self.stream);
                qbp as *const _
            }
            None => core::ptr::null(),
        };
        check(unsafe {
            f(
                qp as *const _,
                qbias_ptr,
                kp as *const _,
                sp as *const _,
                hp as *const _,
                op as *mut _,
                kv_stride as u32,
                n_enc as u32,
                n_heads as u32,
                head_dim as u32,
                n_sel as u32,
                batch as u32,
                scale,
                kv_dtype as u32,
                self.stream_ptr(),
            )
        })
    }
}
