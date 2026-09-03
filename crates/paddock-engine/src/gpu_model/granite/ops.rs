//! Weight-class dispatch for granite's linear layers.
//!
//! Granite serves the same decoder from two checkpoint families, and they do
//! not share a weight class (see [`GraniteW`]). Every GEMM/GEMV in
//! `forward.rs` and `batch.rs` goes through one of the four entry points here
//! so that the class lives in one file instead of at ~58 call sites.
//!
//! The shape of the dispatch is forced by what the classes eat:
//!
//! - **Quant** (Q8_0 / k-quant) takes INT8 activations on the batch paths -
//!   the caller pre-quantizes `xn` into `xq`/`xs` and the kernel consumes
//!   those. This is the existing qwen35 machinery, reached unchanged.
//! - **Nvf4** is W4A16: the checkpoint's own recipe carries an
//!   `input_global_scale` for a W4A4 lane we deliberately do not serve, so the
//!   activations stay f32 and the int8 pair is simply unused. That is why each
//!   entry point takes `xn` (the f32 activations) ALONGSIDE the int8 pair -
//!   not redundancy, but the only buffer the fp4 kernels can read.
//! - **Bf16** likewise reads f32 activations.
//!
//! The int8 buffers are still quantized by the caller on an NVFP4 model. That
//! is wasted work rather than wrong work, and it keeps the call sites free of
//! per-class branching; if it ever shows up in a profile the fix is to hoist
//! the class test above the quantize, not to fork the dispatch.

use cudarc::driver::CudaSlice;

use paddock_models::ggml_type::GgmlType;

use crate::gpu::{GpuExecutor, QuantW};
use crate::gpu_model::gpt_oss::GpuModelError;
use crate::gpu_model::qwen35::{
    gemv_any, gemv8_any, mmq_pre_any, nvf4_mm, nvf4_w4a4_min_rows, prefill_mm_pre_any,
    prefill_mm_pre_p,
};

use super::GraniteW;

/// The SPLIT gate/up pair, for the seats that cannot take a merged plane
/// (every int8 class: the merge is exact only because nvfp4 carries a
/// per-tensor scale that granite ships identical for the two, and a k-quant
/// plane has no such scalar). Returns an error rather than panicking so a
/// future loader that merges a class one of these seats still handles cannot
/// fail silently.
pub(crate) fn split_ffn(
    layer: &super::GraniteLayer,
) -> Result<(&GraniteW, &GraniteW), GpuModelError> {
    match (&layer.gate, &layer.up) {
        (Some(g), Some(u)) => Ok((g, u)),
        _ => Err(GpuModelError::Unsupported(
            "granite: this FFN seat needs split gate/up planes but the layer is merged".into(),
        )),
    }
}

/// `y = W x`, one row. The serial spine and the captured decode graph.
pub(crate) fn gemv(
    exec: &GpuExecutor,
    w: &GraniteW,
    x: &CudaSlice<f32>,
    y: &mut CudaSlice<f32>,
) -> Result<(), GpuModelError> {
    match w {
        GraniteW::Quant(q) => gemv_any(exec, q, x, y),
        GraniteW::Nvf4(p) => nvf4_mm(exec, p, x, y, 1),
        GraniteW::Nvf4Fused { .. } => Err(GpuModelError::Unsupported(
            "granite: q/k/v live in the fused NVFP4 q|k|v plane; this seat must go through nvf4_qkv_into".into(),
        )),
        // the fp8 rowwise plane has a native f32-in gemv, so r==1 needs no
        // activation staging at all
        GraniteW::Fp8 { plane, out_dim, in_dim } => {
            exec.f8r_gemv(plane, x, y, *in_dim, *out_dim)?;
            Ok(())
        }
        // r=1 on the lin class needs a partials plane - the walker's own
        // arms call `gemv_f8lin`; reaching this seat is a routing bug.
        GraniteW::F8Lin { .. } => Err(GpuModelError::Unsupported(
            "granite f8lin: r=1 rides gemv_f8lin (partials plane required)".into(),
        )),
        GraniteW::Bf16(t) => {
            exec.bf16_gemv(t, None, x, y)?;
            Ok(())
        }
    }
}

/// Whether a batched nvf4 GEMM at `r` rows takes the W4A4 (fp4-activation)
/// path - i.e. whether it is safe to hand it a PRE-STAGED nvf4 activation pair
/// instead of an f32 `xn`. Below this the GEMM wants f32 activations (W4A16),
/// so the norm/swiglu fusions must not pre-stage.
pub(crate) fn nvf4_prestaged_ok(exec: &GpuExecutor, r: usize) -> bool {
    r >= nvf4_w4a4_min_rows() && exec.has_nvf4_gemm_f4()
}

/// Run an nvf4 GEMM over a PRE-STAGED activation pair (`xq`/`xs4` already
/// hold the quantize_nvf4 output - from a fused norm or swiglu), skipping the
/// internal quantize. Same election and output as the quantizing path; the
/// caller guarantees `nvf4_prestaged_ok`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn nvf4_mm_prestaged(
    exec: &GpuExecutor,
    p: &crate::gpu::Nvf4Plane,
    xq: &CudaSlice<i8>,
    xs4: &CudaSlice<u8>,
    _part: &mut CudaSlice<f32>,
    y: &mut CudaSlice<f32>,
    r: usize,
) -> Result<(), GpuModelError> {
    // part = None to match the prefill (pf_mm) Nvf4 arm exactly: with part the
    // f4c arm K-SPLITS and folds partials in a different summation order than
    // the unsplit GEMM, a near-tie divergence from the fused-off path. The
    // mixed tick is a prefill tick, so None is the bit-exact choice.
    exec.nvf4_gemm_f4(p, xq, xs4, y, None, r, None)?;
    Ok(())
}

/// The fused q|k|v NVFP4 seat, every width. Stages `xn` once
/// and runs one GEMM over the load-time [q|k|v] plane, then tells the rope
/// site how to read the result: `(nz, scale, in_part)`. `in_part` means raw
/// K-split slices sit in `part`, to be folded `nz` deep and scaled by the
/// plane's `scale2` by the consumer (pd_nvf4_sk_reduce's own arithmetic,
/// without its launch and its y round trip); otherwise the finished plane is
/// in `y` (nz 1, scale 1). r=1 rides the contiguous-plane GEMV; r>=2 takes the
/// W4A4 mma from the same row floor as `mm_pre`'s Nvf4 arm, and below that
/// floor the W4A16 route -- the same class ladder as the split planes, so the
/// numbers per element are the split path's bit for bit.
#[allow(clippy::too_many_arguments)]
pub(crate) fn nvf4_qkv_into(
    exec: &GpuExecutor,
    qkv: &crate::gpu::Nvf4Plane,
    xn: &CudaSlice<f32>,
    xq: &mut CudaSlice<i8>,
    xs4: &mut CudaSlice<u8>,
    part: &mut CudaSlice<f32>,
    y: &mut CudaSlice<f32>,
    r: usize,
    prestaged: bool,
) -> Result<(u32, f32, bool), GpuModelError> {
    if r == 1 {
        exec.nvf4_gemv(qkv, xn, y, None)?;
        return Ok((1, 1.0, false));
    }
    if r < nvf4_w4a4_min_rows() || !exec.has_nvf4_gemm_f4() {
        nvf4_mm(exec, qkv, xn, y, r)?;
        return Ok((1, 1.0, false));
    }
    // `prestaged`: the attn-norm fold (add_rmsnorm_quant_nvf4_from_parts)
    // already wrote xq/xs4 for this row block -- the caller only sets it on
    // the W4A4 path this function would take (nvf4_prestaged_ok).
    if !prestaged {
        exec.quantize_nvf4(xn, xq, xs4, r * qkv.in_dim)?;
    }
    if let Some(nz) = exec.nvf4_gemm_f4_raw_parts(qkv, xq, xs4, part, r)? {
        return Ok((nz, qkv.scale2, true));
    }
    exec.nvf4_gemm_f4(qkv, xq, xs4, y, None, r, Some(part))?;
    Ok((1, 1.0, false))
}

/// The r=1 q|k|v seat: one launch when all three planes are checkpoint NVFP4.
///
/// They share `xn`, and granite's k/v are `out_dim` 1024 - 128 CTAs on a
/// 188-SM die, so separately they pay a full launch's ramp for a quarter of
/// the bytes (at c1, q/k/v/o all measure ~8.5 us despite k/v being 4x smaller).
/// Merging gives one 6144-row grid. Anything that is not a uniform NVFP4
/// triple falls back to the three separate calls, which is what the Q8 and
/// k-quant classes already do through their own fused arms upstream.
pub(crate) fn gemv_qkv(
    exec: &GpuExecutor,
    wq: &GraniteW,
    wk: &GraniteW,
    wv: &GraniteW,
    xn: &CudaSlice<f32>,
    q: &mut CudaSlice<f32>,
    k: &mut CudaSlice<f32>,
    v: &mut CudaSlice<f32>,
) -> Result<(), GpuModelError> {
    if let (GraniteW::Nvf4(pq), GraniteW::Nvf4(pk), GraniteW::Nvf4(pv)) = (wq, wk, wv)
        && exec.has_nvf4_gemv_multi()
        && pq.in_dim == pk.in_dim
        && pk.in_dim == pv.in_dim
    {
        exec.nvf4_gemv_multi(&[(pq, q), (pk, k), (pv, v)], xn, pq.in_dim)?;
        return Ok(());
    }
    gemv(exec, wq, xn, q)?;
    gemv(exec, wk, xn, k)?;
    gemv(exec, wv, xn, v)
}

/// `y = W x` over `rows` rows, f32 activations. The class-generic multi-row
/// path - what the checkpoint classes use everywhere the quant classes would
/// take an int8 kernel.
pub(crate) fn mm(
    exec: &GpuExecutor,
    w: &GraniteW,
    x: &CudaSlice<f32>,
    y: &mut CudaSlice<f32>,
    rows: usize,
) -> Result<(), GpuModelError> {
    match w {
        GraniteW::Quant(q) => {
            if rows == 1 {
                gemv_any(exec, q, x, y)
            } else {
                // no int8 pair here - callers with one take `mm_pre`
                Err(GpuModelError::Unsupported(
                    "granite: quantized multi-row GEMM needs pre-quantized activations".into(),
                ))
            }
        }
        GraniteW::Nvf4(p) => nvf4_mm(exec, p, x, y, rows),
        GraniteW::Nvf4Fused { .. } => Err(GpuModelError::Unsupported(
            "granite: q/k/v live in the fused NVFP4 q|k|v plane; this seat must go through nvf4_qkv_into".into(),
        )),
        // no staging buffers here - `mm_pre`/`pf_mm` are the seats that have
        // them, and every fp8 caller goes through one of those
        GraniteW::F8Lin { .. } => Err(GpuModelError::Unsupported(
            "granite f8lin: use mm_pre/pf_mm (staged e4m3 + partials)".into(),
        )),
        GraniteW::Fp8 { .. } if rows == 1 => gemv(exec, w, x, y),
        GraniteW::Fp8 { .. } => Err(GpuModelError::Unsupported(
            "granite fp8: multi-row needs the e4m3 activation pair - use mm_pre/pf_mm".into(),
        )),
        GraniteW::Bf16(t) => {
            exec.bf16_gemm(t, None, x, y, rows)?;
            Ok(())
        }
    }
}

/// r=1 over the tile-linear f8 plane: the lin GEMV is built for exactly this
/// shape (qwen35's b=1 arm, slot 481). `ticket=None` pins nz=1 - `part` is
/// the kernel's workspace, y is written directly. Non-F8Lin weights fall to
/// the plain [`gemv`] so call sites can stay uniform.
pub(crate) fn gemv_f8lin(
    exec: &GpuExecutor,
    w: &GraniteW,
    x: &CudaSlice<f32>,
    xq: &mut CudaSlice<i8>,
    xs4: &mut CudaSlice<u8>,
    part: &mut CudaSlice<f32>,
    y: &mut CudaSlice<f32>,
) -> Result<(), GpuModelError> {
    match w {
        GraniteW::F8Lin {
            plane,
            out_dim,
            in_dim,
        } => {
            if plane.is_lin() {
                exec.f8lin_gemv(plane, x, part, y, None, *in_dim, *out_dim)?;
            } else {
                // row-major w8: e4m3 PRE-STAGED by the caller's group site
                exec.f8d_gemm_mma_ks(plane, *in_dim, *out_dim, xq, xs4, part, y, 1)?;
            }
            Ok(())
        }
        _ => gemv(exec, w, x, y),
    }
}

/// The 2..=8 decode band: int8 activations for the quant classes, f32 for the
/// checkpoint classes.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gemv8(
    exec: &GpuExecutor,
    w: &GraniteW,
    xn: &CudaSlice<f32>,
    xq: &mut CudaSlice<i8>,
    xs: &mut CudaSlice<f32>,
    ssums: &CudaSlice<f32>,
    y: &mut CudaSlice<f32>,
) -> Result<(), GpuModelError> {
    match w {
        GraniteW::Quant(q) => gemv8_any(exec, q, xn, xq, xs, ssums, y),
        // r==1 on fp8 takes the f32-in gemv; the staged pair is left alone
        _ => gemv(exec, w, xn, y),
    }
}

/// Multi-row with the caller's pre-quantized activations (the chunked-prefill
/// and wide-decode path).
#[allow(clippy::too_many_arguments)]
pub(crate) fn mm_pre(
    exec: &GpuExecutor,
    w: &GraniteW,
    xn: &CudaSlice<f32>,
    xq: &mut CudaSlice<i8>,
    xs: &mut CudaSlice<f32>,
    xs4: &mut CudaSlice<u8>,
    ssums: &mut CudaSlice<f32>,
    part: &mut CudaSlice<f32>,
    y: &mut CudaSlice<f32>,
    r: usize,
) -> Result<(), GpuModelError> {
    match w {
        GraniteW::Quant(q) => mmq_pre_any(exec, q, xq, xs, ssums, part, y, r),
        // W4A4 - fp4 x fp4 on the block-scale mma. This is the checkpoint's
        // own declared recipe (`input_activations` 4-bit, group 16, e4m3
        // scales) and the class the fp4 tensor cores actually want. Without
        // it the lane runs W4A16 software dequant, which lands behind our own
        // Q8_0 lane -- a 4-bit path has no business being there.
        GraniteW::Nvf4(p) if r >= nvf4_w4a4_min_rows() && exec.has_nvf4_gemm_f4() => {
            exec.quantize_nvf4(xn, xq, xs4, r * p.in_dim)?;
            exec.nvf4_gemm_f4(p, xq, xs4, y, None, r, Some(part))?;
            Ok(())
        }
        // e4m3-row pair PRE-STAGED by the walker's group site (one
        // quantize_e4m3_row per shared input - the group dedup; is_f8row at
        // the staging site is the single source, same contract as is_quant/
        // is_f8w).
        GraniteW::Fp8 {
            plane,
            out_dim,
            in_dim,
        } => {
            exec.f8row_gemm(plane, xq, xs, y, *in_dim, *out_dim, r)?;
            Ok(())
        }
        // e4m3 restage in place (same reasoning as the Fp8 arm above). The
        // f8d wrapper routes lin planes by width: legacy lin <= 32 rows, the
        // kt arm above (~94% of the DRAM roof on this die, 801-discovered).
        GraniteW::F8Lin {
            plane,
            out_dim,
            in_dim,
        } => {
            // activations PRE-STAGED e4m3 by the walker's group site (one
            // quantize per shared input - is_f8w at the staging site is the
            // single source; restaging here would double the launches the
            // dedup exists to remove).
            // Width split: f8d's batch tiles RE-READ weights per 64 rows -
            // fine to 64, ~17 weight sweeps per GEMM at r~1100 mixed-tick
            // waves.
            // The wide band streams weights once via f8_gemm_w8.
            if r <= 64 || plane.is_lin() {
                exec.f8d_gemm_mma_ks(plane, *in_dim, *out_dim, xq, xs4, part, y, r)?;
            } else {
                exec.f8_gemm_w8(plane, 0, xq, xs4, y, *in_dim, *out_dim, r)?;
            }
            Ok(())
        }
        _ => mm(exec, w, xn, y, r),
    }
}

/// The prefill matmul (Q8 activations + the y-quant staging buffers).
#[allow(clippy::too_many_arguments)]
pub(crate) fn pf_mm(
    exec: &GpuExecutor,
    w: &GraniteW,
    xn: &CudaSlice<f32>,
    xq: &mut CudaSlice<i8>,
    xs: &mut CudaSlice<f32>,
    xs4: &mut CudaSlice<u8>,
    yq: &CudaSlice<u8>,
    xsums: &mut CudaSlice<f32>,
    ssums: &mut CudaSlice<f32>,
    skfix: &mut CudaSlice<f32>,
    part: &mut CudaSlice<f32>,
    y: &mut CudaSlice<f32>,
    r: usize,
) -> Result<(), GpuModelError> {
    match w {
        // Q8 takes the partials-carrying entry: 65..=192-row chunks ride the
        // mcol mma rung (weights read once) instead of base mmq - the
        // concurrent-admission prefill fix, see prefill_mm_pre_sk.
        GraniteW::Quant(QuantW::Q8(q8)) => {
            prefill_mm_pre_p(exec, q8, xq, xs, yq, skfix, part, y, r)
        }
        GraniteW::Quant(q) => prefill_mm_pre_any(exec, q, xq, xs, yq, xsums, ssums, skfix, y, r),
        GraniteW::Nvf4(p) if r >= nvf4_w4a4_min_rows() && exec.has_nvf4_gemm_f4() => {
            exec.quantize_nvf4(xn, xq, xs4, r * p.in_dim)?;
            exec.nvf4_gemm_f4(p, xq, xs4, y, None, r, None)?;
            Ok(())
        }
        GraniteW::Fp8 {
            plane,
            out_dim,
            in_dim,
        } => {
            // e4m3-row pair pre-staged by the walker's pf group site
            exec.f8row_gemm(plane, xq, xs, y, *in_dim, *out_dim, r)?;
            Ok(())
        }
        GraniteW::F8Lin {
            plane,
            out_dim,
            in_dim,
        } => {
            // pre-staged e4m3 + the same width split as mm_pre's arm
            if r <= 64 || plane.is_lin() {
                exec.f8d_gemm_mma_ks(plane, *in_dim, *out_dim, xq, xs4, part, y, r)?;
            } else {
                exec.f8_gemm_w8(plane, 0, xq, xs4, y, *in_dim, *out_dim, r)?;
            }
            Ok(())
        }
        _ => mm(exec, w, xn, y, r),
    }
}

/// Gather token rows out of the embedding table, applying granite's
/// `embedding_scale` in the same kernel.
///
/// The table's own `ty` picks the kernel: the GGUF lane lands Q8_0 or a
/// k-quant, the NVFP4 checkpoint lands bf16 (IBM leaves the embeddings
/// unquantized). Routing on the tensor rather than on the lane is what lets
/// both share this.
pub(crate) fn embed_gather(
    exec: &GpuExecutor,
    tbl: &super::TokEmbd,
    tokens: &CudaSlice<u32>,
    out: &mut CudaSlice<f32>,
    embd: usize,
    n: usize,
) -> Result<(), GpuModelError> {
    match tbl {
        super::TokEmbd::Q8(t) if t.ty == GgmlType::Bf16 => {
            exec.embed_gather_bf16(t, tokens, out, embd, n, 1.0)?;
        }
        super::TokEmbd::Q8(t) => exec.embed_gather_batch_q8(t, tokens, out, embd, n)?,
        super::TokEmbd::Kq(t) => exec.kquant_gather(t, tokens, out, embd, n)?,
    }
    Ok(())
}

/// Does this weight take the int8-activation lane at all? The fast paths in
/// `batch.rs` ask before pre-quantizing.
pub(crate) fn is_quant(w: &GraniteW) -> bool {
    matches!(w, GraniteW::Quant(_))
}

/// The tuned-f8 plane class test - the walker's staging sites ask this the
/// way they ask [`is_quant`]: one class test decides both the staging format
/// and the GEMM route (the staging-contract law - a mismatch is fluent
/// garbage with green gates).
/// Fused-plane f8row GEMM into a caller-owned output plane (the pf-side
/// fused-qkv rung): consumes the seat's PRE-STAGED e4m3-row pair, dispatches
/// through pd_f8row_gemm (mma <=64 / tw >=65 / kt tail).
#[allow(clippy::too_many_arguments)]
pub(crate) fn f8row_mm_into(
    exec: &GpuExecutor,
    plane: &crate::gpu::F8RowPlane,
    in_dim: usize,
    out_dim: usize,
    xq: &CudaSlice<i8>,
    xrs: &CudaSlice<f32>,
    y: &mut CudaSlice<f32>,
    r: usize,
) -> Result<(), GpuModelError> {
    exec.f8row_gemm(plane, xq, xrs, y, in_dim, out_dim, r)?;
    Ok(())
}

pub(crate) fn is_f8w(w: &GraniteW) -> bool {
    matches!(w, GraniteW::F8Lin { .. })
}

/// The f8row class test - same single-source staging contract as
/// [`is_f8w`], for the per-ROW e4m3 activation pair.
pub(crate) fn is_f8row(w: &GraniteW) -> bool {
    matches!(w, GraniteW::Fp8 { .. })
}

/// Both weights are the same Q8/k-quant class - the precondition every fused
/// multi-projection fast path needs.
pub(crate) fn both_quant<'a>(a: &'a GraniteW, b: &'a GraniteW) -> Option<(&'a QuantW, &'a QuantW)> {
    Some((a.quant()?, b.quant()?))
}

/// Three-way twin of [`both_quant`] for the fused q/k/v seats.
pub(crate) fn tri_quant<'a>(
    a: &'a GraniteW,
    b: &'a GraniteW,
    c: &'a GraniteW,
) -> Option<(&'a QuantW, &'a QuantW, &'a QuantW)> {
    Some((a.quant()?, b.quant()?, c.quant()?))
}
