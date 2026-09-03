//! Empirical verify: `q8_0_gemm_mma` / `q8_0_gemm_mmq` must handle a
//! non-16-aligned `out_dim` correctly.
//!
//! Both launchers used to reject any `out_dim` not a multiple of 16
//! (`CUDA_ERROR_INVALID_VALUE`) - laguna S-2.1's `g_proj` (the per-head
//! softplus attention gate) has `out_dim = 72` (its SWA head count), which
//! broke the first real >64-row batched prefill. Reading both
//! kernels' tile staging and writeback shows every row is already
//! bounds-checked individually (`row_base + row < out_dim`) with
//! out-of-range rows zero-padded during staging - the tail past the old
//! 16-row boundary was never actually unsafe, just rejected up front. This
//! test proves it empirically rather than trusting that reading alone:
//! rows 64..72 (past the old alignment boundary) must compute as correctly
//! as rows 0..64, against the independently-trusted `q8_0_gemv_repacked`
//! kernel (the same one the byte-exact greedy-parity sweep already
//! validates against llama.cpp).
//!
//! Numeric-class note: `q8_0_gemv_repacked` is exact-f32-activation (W8A16);
//! `q8_0_gemm_mma`/`q8_0_gemm_mmq` quantize the activation to int8 too
//! (W8A8) for tensor-core throughput - a real, expected difference
//! (activation-quantization noise, ~1/127 per element), not a bug. This
//! test bounds that noise with a generous relative tolerance and checks the
//! tail rows show the same error profile as the aligned rows, not a
//! qualitatively different (i.e. broken) one.
//!
//! Gated: skips (loudly) when the pack isn't built or no CUDA device exists.

mod common;

use cudarc::driver::{DevicePtr, DevicePtrMut};
use paddock_kernels::KernelPack;

/// Deterministic xorshift64 - no external rand dependency, reproducible.
struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn next_i8_range(&mut self, span: i32) -> i8 {
        ((self.next_u64() % (2 * span as u64 + 1)) as i32 - span) as i8
    }
    fn next_unit_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

fn as_void(p: u64) -> *const core::ffi::c_void {
    p as *const core::ffi::c_void
}
fn as_void_mut(p: u64) -> *mut core::ffi::c_void {
    p as *mut core::ffi::c_void
}

#[test]
fn q8_0_gemm_tensorcore_handles_ragged_out_dim() {
    let Some(path) = common::pack() else {
        return;
    };
    let Some(ctx) = common::cuda() else {
        return;
    };
    let pack = KernelPack::load(&path).expect("pack load");
    let k = pack.kernels_v1().expect("v1 kernel table");
    let stream = ctx.default_stream();

    let in_dim: u32 = 256;
    let out_dim: u32 = 72; // laguna S-2.1's g_proj SWA head count: 72 & 15 == 8, not 16-aligned
    let n_blocks = (in_dim / 32) as usize;

    let mut rng = Rng(0x5EED_1234_ABCD_EF01);
    // Repacked Q8_0 weight: int8 data [out_dim, in_dim] contiguous + f16 scale
    // [out_dim, n_blocks] (matches pd_q8_0_repack's output layout).
    let w_data: Vec<u8> = (0..(out_dim as usize * in_dim as usize))
        .map(|_| rng.next_i8_range(100) as u8)
        .collect();
    let w_scale_bytes: Vec<u8> = (0..(out_dim as usize * n_blocks))
        .flat_map(|_| half::f16::from_f32(0.01 + 0.09 * rng.next_unit_f32().abs()).to_le_bytes())
        .collect();

    let d_wdata = stream.clone_htod(&w_data).expect("htod w_data");
    let d_wscale = stream.clone_htod(&w_scale_bytes).expect("htod w_scale");

    let mut run_case = |batch: u32, label: &str| {
        let x: Vec<f32> = (0..(batch as usize * in_dim as usize))
            .map(|_| rng.next_unit_f32())
            .collect();
        let d_x = stream.clone_htod(&x).expect("htod x");

        // --- reference: q8_0_gemv_repacked, one row at a time (exact-f32 activation) ---
        let gemv_ref = k
            .q8_0_gemv_repacked
            .expect("pack provides q8_0_gemv_repacked");
        let mut y_ref = vec![0f32; batch as usize * out_dim as usize];
        for b in 0..batch as usize {
            let mut d_y_row = stream
                .alloc_zeros::<f32>(out_dim as usize)
                .expect("alloc y_row");
            {
                let (wd, _g1) = d_wdata.device_ptr(&stream);
                let (ws, _g2) = d_wscale.device_ptr(&stream);
                let (xr, _g3) = d_x.device_ptr(&stream);
                let (yr, _g4) = d_y_row.device_ptr_mut(&stream);
                let x_row_ptr = xr + (b * in_dim as usize * 4) as u64;
                // SAFETY: device pointers + stream are live for the call; ABI v1 contract
                let status = unsafe {
                    gemv_ref(
                        as_void(wd),
                        as_void(ws),
                        std::ptr::null(),
                        as_void(x_row_ptr),
                        as_void_mut(yr),
                        in_dim,
                        out_dim,
                        stream.cu_stream() as *mut core::ffi::c_void,
                    )
                };
                assert_eq!(status, 0, "{label}: reference gemv failed row {b}");
            }
            stream.synchronize().expect("sync ref");
            let row = stream.clone_dtoh(&d_y_row).expect("dtoh ref row");
            y_ref[b * out_dim as usize..(b + 1) * out_dim as usize].copy_from_slice(&row);
        }
        (x, y_ref)
    };

    // --- candidate A: q8_0_gemm_mma at batch<=64 (the 64x64-tile branch) ---
    {
        let batch: u32 = 40;
        let (x, y_ref) = run_case(batch, "mma");
        let d_x = stream.clone_htod(&x).expect("htod x mma");
        let quantize = k.quantize_q8.expect("pack provides quantize_q8");
        let mut d_xq = stream
            .alloc_zeros::<u8>(batch as usize * in_dim as usize)
            .expect("alloc xq");
        let mut d_xs = stream
            .alloc_zeros::<f32>(batch as usize * n_blocks)
            .expect("alloc xs");
        {
            let (xr, _g1) = d_x.device_ptr(&stream);
            let (xq, _g2) = d_xq.device_ptr_mut(&stream);
            let (xs, _g3) = d_xs.device_ptr_mut(&stream);
            let status = unsafe {
                quantize(
                    as_void(xr),
                    as_void_mut(xq),
                    as_void_mut(xs),
                    batch * in_dim, // element count - the launcher itself divides by 32
                    stream.cu_stream() as *mut core::ffi::c_void,
                )
            };
            assert_eq!(status, 0, "quantize_q8 failed");
        }
        let mma = k.q8_0_gemm_mma.expect("pack provides q8_0_gemm_mma");
        let mut d_y = stream
            .alloc_zeros::<f32>(batch as usize * out_dim as usize)
            .expect("alloc y mma");
        {
            let (wd, _g1) = d_wdata.device_ptr(&stream);
            let (ws, _g2) = d_wscale.device_ptr(&stream);
            let (xq, _g3) = d_xq.device_ptr(&stream);
            let (xs, _g4) = d_xs.device_ptr(&stream);
            let (yy, _g5) = d_y.device_ptr_mut(&stream);
            let status = unsafe {
                mma(
                    as_void(wd),
                    as_void(ws),
                    as_void(xq),
                    as_void(xs),
                    as_void_mut(yy),
                    in_dim,
                    out_dim,
                    batch,
                    stream.cu_stream() as *mut core::ffi::c_void,
                )
            };
            assert_eq!(
                status, 0,
                "q8_0_gemm_mma rejected out_dim={out_dim} (the bug under test)"
            );
        }
        stream.synchronize().expect("sync mma");
        let y_mma = stream.clone_dtoh(&d_y).expect("dtoh y mma");
        check_tail_not_worse(&y_ref, &y_mma, batch, out_dim, "q8_0_gemm_mma");
    }

    // --- candidate B: q8_0_gemm_mmq at batch>64 (the ACTUAL kernel that crashed) ---
    {
        let batch: u32 = 96;
        let (x, y_ref) = run_case(batch, "mmq");
        let d_x = stream.clone_htod(&x).expect("htod x mmq");
        let quantize_mmq = k.quantize_q8_mmq.expect("pack provides quantize_q8_mmq");
        let n_chunks = in_dim.div_ceil(128) as usize;
        let batch_pad = (batch as usize).div_ceil(128) * 128;
        let mut d_yq = stream
            .alloc_zeros::<u8>(n_chunks * batch_pad * 144)
            .expect("alloc yq");
        {
            let (xr, _g1) = d_x.device_ptr(&stream);
            let (yq, _g2) = d_yq.device_ptr_mut(&stream);
            let status = unsafe {
                quantize_mmq(
                    as_void(xr),
                    as_void_mut(yq),
                    in_dim,
                    batch,
                    stream.cu_stream() as *mut core::ffi::c_void,
                )
            };
            assert_eq!(status, 0, "quantize_q8_mmq failed");
        }
        let mmq = k.q8_0_gemm_mmq.expect("pack provides q8_0_gemm_mmq");
        let mut d_y = stream
            .alloc_zeros::<f32>(batch as usize * out_dim as usize)
            .expect("alloc y mmq");
        {
            let (wd, _g1) = d_wdata.device_ptr(&stream);
            let (ws, _g2) = d_wscale.device_ptr(&stream);
            let (yq, _g3) = d_yq.device_ptr(&stream);
            let (yy, _g4) = d_y.device_ptr_mut(&stream);
            // fixup = null forces plain tiled mode (skips the stream-k partition) -
            // the ragged-boundary staging/writeback logic under test is identical
            // either way; this just avoids needing a real stream-k fixup buffer.
            let status = unsafe {
                mmq(
                    as_void(wd),
                    as_void(ws),
                    as_void(yq),
                    std::ptr::null_mut(),
                    as_void_mut(yy),
                    in_dim,
                    out_dim,
                    batch,
                    stream.cu_stream() as *mut core::ffi::c_void,
                )
            };
            assert_eq!(
                status, 0,
                "q8_0_gemm_mmq rejected out_dim={out_dim} (the bug under test)"
            );
        }
        stream.synchronize().expect("sync mmq");
        let y_mmq = stream.clone_dtoh(&d_y).expect("dtoh y mmq");
        check_tail_not_worse(&y_ref, &y_mmq, batch, out_dim, "q8_0_gemm_mmq");

        // --- candidate D: q8_0_gemm_mmq_pipe, same yq (the kernel a real
        // serving traffic pattern crashed: RepackedQ8.dims[0] is in_dim, not
        // out_dim, so the caller's `dims[0] % 128 == 0` gate does not exclude
        // a narrow out_dim like g_proj's - the earlier "caller-gated, skip
        // this family" call was wrong) ---
        let pipe = k
            .q8_0_gemm_mmq_pipe
            .expect("pack provides q8_0_gemm_mmq_pipe");
        let mut d_y2 = stream
            .alloc_zeros::<f32>(batch as usize * out_dim as usize)
            .expect("alloc y pipe");
        {
            let (wd, _g1) = d_wdata.device_ptr(&stream);
            let (ws, _g2) = d_wscale.device_ptr(&stream);
            let (yq, _g3) = d_yq.device_ptr(&stream);
            let (yy, _g4) = d_y2.device_ptr_mut(&stream);
            let status = unsafe {
                pipe(
                    as_void(wd),
                    as_void(ws),
                    as_void(yq),
                    std::ptr::null(),
                    as_void_mut(yy),
                    in_dim,
                    out_dim,
                    batch,
                    stream.cu_stream() as *mut core::ffi::c_void,
                )
            };
            assert_eq!(
                status, 0,
                "q8_0_gemm_mmq_pipe rejected out_dim={out_dim} (the bug under test)"
            );
        }
        stream.synchronize().expect("sync pipe");
        let y_pipe = stream.clone_dtoh(&d_y2).expect("dtoh y pipe");
        check_tail_not_worse(&y_ref, &y_pipe, batch, out_dim, "q8_0_gemm_mmq_pipe");
    }

    // --- candidate C: q8_0_gemm_mma_ks at batch=32 (the decode-tick config
    // that actually crashed: laguna's CUDA-graph-captured step, --max-batch
    // 32, mmq_pre -> q8_0_gemm_mma_ks) ---
    {
        let batch: u32 = 32;
        let (x, y_ref) = run_case(batch, "mma_ks");
        let d_x = stream.clone_htod(&x).expect("htod x mma_ks");
        let quantize = k.quantize_q8.expect("pack provides quantize_q8");
        let mut d_xq = stream
            .alloc_zeros::<u8>(batch as usize * in_dim as usize)
            .expect("alloc xq ks");
        let mut d_xs = stream
            .alloc_zeros::<f32>(batch as usize * n_blocks)
            .expect("alloc xs ks");
        {
            let (xr, _g1) = d_x.device_ptr(&stream);
            let (xq, _g2) = d_xq.device_ptr_mut(&stream);
            let (xs, _g3) = d_xs.device_ptr_mut(&stream);
            let status = unsafe {
                quantize(
                    as_void(xr),
                    as_void_mut(xq),
                    as_void_mut(xs),
                    batch * in_dim,
                    stream.cu_stream() as *mut core::ffi::c_void,
                )
            };
            assert_eq!(status, 0, "quantize_q8 failed (ks)");
        }
        let mma_ks = k.q8_0_gemm_mma_ks.expect("pack provides q8_0_gemm_mma_ks");
        // nz (K-split factor) is computed internally and capped at 8; size the
        // partial-planes buffer for the worst case so any internal choice fits.
        let mut d_part = stream
            .alloc_zeros::<f32>(8 * batch as usize * out_dim as usize)
            .expect("alloc part");
        let mut d_y = stream
            .alloc_zeros::<f32>(batch as usize * out_dim as usize)
            .expect("alloc y mma_ks");
        {
            let (wd, _g1) = d_wdata.device_ptr(&stream);
            let (ws, _g2) = d_wscale.device_ptr(&stream);
            let (xq, _g3) = d_xq.device_ptr(&stream);
            let (xs, _g4) = d_xs.device_ptr(&stream);
            let (pp, _g5) = d_part.device_ptr_mut(&stream);
            let (yy, _g6) = d_y.device_ptr_mut(&stream);
            let status = unsafe {
                mma_ks(
                    as_void(wd),
                    as_void(ws),
                    as_void(xq),
                    as_void(xs),
                    as_void_mut(pp),
                    as_void_mut(yy),
                    in_dim,
                    out_dim,
                    batch,
                    stream.cu_stream() as *mut core::ffi::c_void,
                )
            };
            assert_eq!(
                status, 0,
                "q8_0_gemm_mma_ks rejected out_dim={out_dim} (the bug under test)"
            );
        }
        stream.synchronize().expect("sync mma_ks");
        let y_mma_ks = stream.clone_dtoh(&d_y).expect("dtoh y mma_ks");
        check_tail_not_worse(&y_ref, &y_mma_ks, batch, out_dim, "q8_0_gemm_mma_ks");
    }
}

/// Compares candidate output against the exact-f32 reference row-by-row,
/// bounding the expected int8-activation-quantization noise, and - the
/// actual point of this test - asserting the tail rows (64..out_dim, past
/// the old 16-alignment boundary) show the same error profile as the
/// aligned rows (0..64), not a qualitatively worse one.
fn check_tail_not_worse(y_ref: &[f32], y_got: &[f32], batch: u32, out_dim: u32, label: &str) {
    let od = out_dim as usize;
    let aligned = 64usize.min(od);
    // Per-element relative error blows up near ref's natural zero-crossings
    // (256-term random dot products cancel to near-zero often; a normal
    // quantization deviation there produces a huge RATIO despite a normal
    // ABSOLUTE deviation). Normalize by the array's overall scale instead -
    // standard practice for numerical kernel comparisons.
    let scale = y_ref.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
    let mut err_aligned = Vec::new();
    let mut err_tail = Vec::new();
    for b in 0..batch as usize {
        for o in 0..od {
            let r = y_ref[b * od + o];
            let g = y_got[b * od + o];
            assert!(
                g.is_finite(),
                "{label}: batch {b} out {o} produced non-finite {g}"
            );
            let rel = (g - r).abs() / scale;
            if o < aligned {
                err_aligned.push(rel);
            } else {
                err_tail.push(rel);
            }
        }
    }
    let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len() as f32;
    let max = |v: &[f32]| v.iter().cloned().fold(0f32, f32::max);
    let (ma, xa) = (mean(&err_aligned), max(&err_aligned));
    let (mt, xt) = if err_tail.is_empty() {
        (0.0, 0.0)
    } else {
        (mean(&err_tail), max(&err_tail))
    };
    eprintln!(
        "{label}: aligned rows [0,{aligned}) mean_rel={ma:.5} max_rel={xa:.5}; \
         tail rows [{aligned},{od}) mean_rel={mt:.5} max_rel={xt:.5}"
    );
    // Measured (seed-fixed): both kernels land under 0.5% max
    // relative error (pure int8-activation-quantization noise on a 256-dim
    // dot product). 2% leaves real margin for different sizes/seeds while
    // still catching a real ragged-boundary bug outright - those produce
    // order-of-magnitude errors or exact zeros (an unwritten tile), not
    // ~0.3-0.5% noise. (An earlier revision of this test asserted <15% and
    // passed on a bug - a units error in this file's own quantize_q8 call
    // left 92% of activations unquantized; fixed, then tightened here.)
    assert!(
        xa < 0.02,
        "{label}: aligned-row max relative error {xa:.5} too high"
    );
    if !err_tail.is_empty() {
        assert!(
            xt < 0.02,
            "{label}: TAIL-row max relative error {xt:.5} too high - the ragged boundary is broken"
        );
        assert!(
            mt < ma * 3.0 + 0.005,
            "{label}: tail mean error {mt:.5} is qualitatively worse than aligned mean {ma:.5} - tail-specific bug"
        );
    }
}
