//! Parity for DFlash2's candidate-selector walk (`pd_dflash_select`, slot 461)
//! against `paddock_kernels::reference::dflash::select_walk`.
//!
//! The walk is a CHAIN - each row's choice picks which predecessor row of the
//! edge matrix the next row reads - so an off-by-one in the carry produces
//! output that is still well-formed and still plausible, and shows up only as
//! lost acceptance. These cases therefore check the carry explicitly, at the
//! real muse geometry (rank 256, top-16) and at truncated runtime blocks.
//! Gated on a CUDA device + built pack.

mod common;

use paddock_engine::gpu::GpuExecutor;
use paddock_kernels::reference::dflash::select_walk;

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

#[allow(clippy::too_many_arguments)]
fn check(exec: &GpuExecutor, tag: &str, rank: usize, k: usize, rows: usize, nblk: usize, cap: f32) {
    let r = rows * nblk;
    let nk = r * k;
    let ids: Vec<u32> = (0..nk)
        .map(|i| ((i * 2654435761usize) % 200000) as u32)
        .collect();
    let logits = det(nk, 0x9e);
    let pred = det((nk + nblk) * rank, 0x11);
    let succ = det(nk * rank, 0x22);
    let hs = det(r * rank, 0x33);
    let scale = 0.196_116_13f32;

    let mut want = vec![0u32; r];
    select_walk(
        &ids, &logits, &pred, &succ, &hs, &mut want, scale, cap, rank, k, rows, r,
    );

    // pd_topk_rows' layout: interleaved (id, raw-logit bits) pairs.
    let mut topk = vec![0u32; nk * 2];
    for i in 0..nk {
        topk[i * 2] = ids[i];
        topk[i * 2 + 1] = logits[i].to_bits();
    }
    let d_topk = exec.to_device_u32(&topk).expect("topk");
    let d_pred = exec.to_device(&pred).expect("pred");
    let d_succ = exec.to_device(&succ).expect("succ");
    let d_hs = exec.to_device(&hs).expect("hs");
    let mut d_out = exec.to_device_u32(&vec![0xDEADBEEFu32; r]).expect("out");
    exec.dflash_select(
        &d_topk, &d_pred, &d_succ, &d_hs, &mut d_out, scale, cap, rank, k, rows, r,
    )
    .expect("dflash_select");
    let got = exec.to_host_u32(&d_out).expect("dtoh");

    // Row 0 of every block is the anchor and is never written by the walk.
    let mut bad = 0usize;
    for b in 0..nblk {
        for j in 1..rows {
            let row = b * rows + j;
            if got[row] != want[row] {
                bad += 1;
                if bad <= 3 {
                    eprintln!("  [{tag}] row {row}: got {} want {}", got[row], want[row]);
                }
            }
        }
    }
    eprintln!(
        "dflash_select parity [{tag}]: {} mismatched of {}",
        bad,
        nblk * (rows - 1)
    );
    assert_eq!(
        bad, 0,
        "dflash_select [{tag}] diverged from the reference walk"
    );
}

#[test]
fn select_matches_cpu_on_muse_geometry() {
    let Some(exec) = common::gpu() else { return };
    if !exec.has_dflash_select() {
        eprintln!("pack has no dflash_select (slot 461) - skipping");
        return;
    }
    // Real muse DFlash2: selector_rank 256, selector_top_k 16, block 16.
    check(&exec, "muse full block", 256, 16, 16, 3, 20.0);
    // Uncapped: the epilogue branch must not be the only path that works.
    check(&exec, "muse no softcap", 256, 16, 16, 3, 0.0);
}

#[test]
fn select_matches_cpu_on_truncated_blocks() {
    let Some(exec) = common::gpu() else { return };
    if !exec.has_dflash_select() {
        return;
    }
    // A real draft round runs rows = k+1 <= block_size, one block per slot.
    check(&exec, "rows=4 x5", 256, 16, 4, 5, 20.0);
    check(&exec, "rows=2 x8", 256, 16, 2, 8, 20.0);
    // 32 blocks is the c32 shape.
    check(&exec, "rows=16 x32", 256, 16, 16, 32, 20.0);
}

/// `k` not dividing the CTA's warp count must still be exact - every warp
/// iterates on a warp-uniform bound, so a warp that runs out of candidates
/// skips the shuffles wholesale rather than stranding a partial mask.
#[test]
fn select_handles_k_indivisible_by_warps() {
    let Some(exec) = common::gpu() else { return };
    if !exec.has_dflash_select() {
        return;
    }
    for k in [1usize, 3, 5, 7, 11, 13] {
        check(&exec, &format!("k={k}"), 256, k, 8, 4, 20.0);
    }
}
