//! Parity for DFlash2's grouped dynamic convolution (`pd_dflash_conv`, slot
//! 459) against the CPU reference in `paddock_kernels::reference::dflash`.
//!
//! Three shapes, because the kernel has three ways to be wrong:
//!   - the real muse geometry (embd 6656, 2 taps, groups of 16) at a full
//!     16-row block, which is what a trained checkpoint expects;
//!   - a TRUNCATED runtime block (`rows` = k+1 < block_size) over several
//!     blocks, which is what a paddock draft round actually runs - this is
//!     where a `& (block_size-1)` mask would silently convolve one slot's
//!     leading row against another slot's trailing one;
//!   - both `side` values, since one projection row feeds both wraps and
//!     transposing the halves is an easy and invisible mistake.
//!     Gated on a CUDA device + built pack.

mod common;

use paddock_engine::gpu::GpuExecutor;
use paddock_kernels::reference::dflash::grouped_conv;

fn det(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / (1u64 << 31) as f32) - 0.5
        })
        .collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[allow(clippy::too_many_arguments)]
fn check(
    exec: &GpuExecutor,
    tag: &str,
    embd: usize,
    taps: usize,
    group_size: usize,
    rows_per_block: usize,
    nblk: usize,
    side: usize,
) {
    let r = rows_per_block * nblk;
    let ng = embd / group_size;
    let h = det(r * embd, 0x51 + side as u64);
    let base = det(2 * taps * embd, 0xa7);
    let delta = det(r * 2 * taps * ng, 0xd3);

    let mut want = vec![0.0f32; r * embd];
    grouped_conv(
        &h,
        &mut want,
        &base,
        &delta,
        side,
        embd,
        taps,
        group_size,
        rows_per_block,
        r,
    );

    let d_h = exec.to_device(&h).expect("h");
    let d_base = exec.to_device(&base).expect("base");
    let d_delta = exec.to_device(&delta).expect("delta");
    // Poisoned, so a row the kernel never writes fails loudly instead of
    // matching a zeroed reference by accident.
    let mut d_out = exec.to_device(&vec![f32::NAN; r * embd]).expect("out");
    exec.dflash_conv(
        &d_h,
        &mut d_out,
        &d_base,
        &d_delta,
        side,
        embd,
        taps,
        ng,
        group_size,
        rows_per_block,
        r,
    )
    .expect("dflash_conv");
    let got = exec.to_host(&d_out).expect("dtoh");

    let diff = max_abs_diff(&got, &want);
    eprintln!("dflash_conv parity [{tag}] side {side}: max_abs_diff {diff:.2e}");
    assert!(
        diff < 1e-5,
        "dflash_conv [{tag}] side {side} max_abs_diff {diff} too high"
    );
}

#[test]
fn conv_matches_cpu_on_muse_geometry() {
    let Some(exec) = common::gpu() else { return };
    if !exec.has_dflash_conv() {
        eprintln!("pack has no dflash_conv (slot 459) - skipping");
        return;
    }
    // Real muse DFlash2: hidden 6656, conv_kernel_size 2, conv_group_size 16,
    // block_size 16. Two blocks so the boundary is exercised at full width.
    for side in 0..2 {
        check(&exec, "muse full block", 6656, 2, 16, 16, 2, side);
    }
}

#[test]
fn conv_masks_a_truncated_runtime_block() {
    let Some(exec) = common::gpu() else { return };
    if !exec.has_dflash_conv() {
        return;
    }
    // rows = k+1 = 4 over 5 slots: the shape a real draft round runs when the
    // service asks for 3 drafts. A block_size-derived mask would be wrong for
    // every row here.
    for side in 0..2 {
        check(&exec, "truncated rows=4 x5", 6656, 2, 16, 4, 5, side);
    }
    // rows = 2 is the tightest: every odd row is a block leader.
    check(&exec, "truncated rows=2 x8", 6656, 2, 16, 2, 8, 0);
}

/// The kernel must refuse geometry its float4 walk cannot express, rather than
/// producing quietly wrong rows.
#[test]
fn conv_refuses_unsupported_geometry() {
    let Some(exec) = common::gpu() else { return };
    if !exec.has_dflash_conv() {
        return;
    }
    let (embd, taps, gs, rows, r) = (12usize, 2usize, 6usize, 4usize, 4usize);
    let d_h = exec.to_device(&det(r * embd, 1)).expect("h");
    let d_base = exec.to_device(&det(2 * taps * embd, 2)).expect("base");
    let d_delta = exec
        .to_device(&det(r * 2 * taps * (embd / gs), 3))
        .expect("delta");
    let mut d_out = exec.to_device(&vec![0.0f32; r * embd]).expect("out");
    // group_size 6 is not a multiple of 4 -> rc -2.
    let rc = exec.dflash_conv(
        &d_h,
        &mut d_out,
        &d_base,
        &d_delta,
        0,
        embd,
        taps,
        embd / gs,
        gs,
        rows,
        r,
    );
    assert!(
        rc.is_err(),
        "group_size 6 must be refused, not silently mis-walked"
    );
}
