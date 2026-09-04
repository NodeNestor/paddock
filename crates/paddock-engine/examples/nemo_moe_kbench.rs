//! Nemotron Q8 MoE decode-band crossover.
//!
//! The r>1 tick serves the experts on the SORTED tile - `moe_align` packs
//! (row, expert) pairs into BM=32 blocks and the tile computes all 32 rows.
//! At decode widths nearly every block holds one real row, so the routed pair
//! leaves most of the tile idle and reaches only a fraction of the stream roof.
//!
//! This prices the candidate routes at the real shape (hidden 2688, moe_ff
//! 1856, shared_ff 3712, 128 experts top-6) so the crossover and the
//! rows-per-CTA election are measured, not assumed:
//!
//!   sorted  align + up_relu2_sorted + quantize + down_sorted + combine
//!   dec2/N  up_relu2_dec2(N rows/CTA) + quantize + dn_dec2 + add
//!
//! dec2 does not dedup: two rows on the same expert stream its plane twice,
//! so it must lose above the decode band. Finding that r is the point.
//!
//! Usage (static pack):
//!   cargo run --release -p paddock-engine --features static-pack \
//!     --example nemo_moe_kbench
//! Usage (pack file):
//!   PADDOCK_PACK=packs/cuda/build/pd-cuda-sm86.dll cargo run --release \
//!     -p paddock-engine --example nemo_moe_kbench
// A development probe: it runs on a box its author is looking at, and a
// failure should stop it where it happened rather than be reported.
#![allow(clippy::unwrap_used)]

use std::sync::Arc;

use paddock_engine::gpu::GpuExecutor;
use paddock_models::mapped::MappedGguf;

/// min-of-`reps` mean-of-`iters` - one pass puts two launches of the same
/// config 20% apart on this die.
fn time_us(exec: &GpuExecutor, reps: usize, iters: usize, mut f: impl FnMut()) -> f64 {
    for _ in 0..5 {
        f();
    }
    exec.synchronize().expect("sync");
    let mut best = f64::MAX;
    for _ in 0..reps {
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            f();
        }
        exec.synchronize().expect("sync");
        best = best.min(t0.elapsed().as_secs_f64() * 1e6 / iters as f64);
    }
    best
}

/// qwen35's dense q8 rung ladder, mirrored here because `mmq_pre` is
/// crate-private: nc GEMV to r=4, K-split mma to 64, mt tile above. The
/// engine's shared-expert route calls the real one - this must stay the same
/// band or the lab is pricing a different thing.
fn dense_pre(
    exec: &GpuExecutor,
    w: &paddock_engine::gpu::RepackedQ8,
    xq: &cudarc::driver::CudaSlice<i8>,
    xs: &cudarc::driver::CudaSlice<f32>,
    part: &mut cudarc::driver::CudaSlice<f32>,
    y: &mut cudarc::driver::CudaSlice<f32>,
    r: usize,
) {
    if r <= 4 {
        exec.q8_0_gemv_dp4a_nc(w, xq, xs, y, r).unwrap();
    } else if r <= 64 && part.len() >= 8 * 64 * w.dims[1] {
        exec.q8_0_gemm_mma_ks(w, xq, xs, part, y, r).unwrap();
    } else {
        exec.q8_0_gemm_mt_dp4a(w, xq, xs, y, r).unwrap();
    }
}

/// The engine's own live-block bound (gpu_model/nemotron/batch.rs) - an
/// UPPER bound on what moe_align can fill, so the lab launches exactly what
/// the tick launches.
fn live_blocks(rows: usize, picks: usize, experts: usize) -> usize {
    experts.min(rows * picks) * rows.div_ceil(32)
}

fn main() {
    let model = std::env::var("NEMO_GGUF").unwrap_or_else(|_| {
        concat!(
            r"E:\paddock\models\NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF",
            r"\NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q8_0.gguf"
        )
        .to_string()
    });
    let exec = match std::env::var_os("PADDOCK_PACK") {
        Some(p) => Arc::new(GpuExecutor::new(0, std::path::Path::new(&p)).expect("executor")),
        None => Arc::new(GpuExecutor::with_pack(0, None).expect("executor (static pack)")),
    };
    println!("sm_count={}", exec.sm_count());
    let map = MappedGguf::open(std::path::Path::new(&model)).expect("open gguf");

    // blk.1 is the checkpoint's first MoE block
    let up = exec
        .repack_q8(&map, "blk.1.ffn_up_exps.weight")
        .expect("up");
    let down = exec
        .repack_q8(&map, "blk.1.ffn_down_exps.weight")
        .expect("down");
    let sh_up = exec
        .repack_q8(&map, "blk.1.ffn_up_shexp.weight")
        .expect("sh_up");
    let sh_down = exec
        .repack_q8(&map, "blk.1.ffn_down_shexp.weight")
        .expect("sh_down");
    let (embd, moe_ff, n_expert) = (up.dims[0], up.dims[1], up.dims[2]);
    let shared_ff = sh_up.dims[1];
    let n_active = 6usize;
    println!("embd={embd} moe_ff={moe_ff} shared_ff={shared_ff} experts={n_expert} top-{n_active}");
    // one expert's up+down at Q8_0: int8 data + one f16 scale per 32
    let e_bytes = 2.0 * moe_ff as f64 * embd as f64 * (1.0 + 2.0 / 32.0);
    let sh_bytes = 2.0 * shared_ff as f64 * embd as f64 * (1.0 + 2.0 / 32.0);

    let rows: &[usize] = &[1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64];
    let r_max = 64usize;
    let nb_max = live_blocks(r_max, n_active, n_expert);
    let nbs_max = live_blocks(r_max, 1, 1);

    // deterministic activation + routing. Picks are distinct within a row
    // (a real top-k is) and spread over all 128 experts, so the collision
    // statistics dec2 pays for are the real ones.
    let pat: Vec<f32> = (0..r_max * embd)
        .map(|i| ((i as u64).wrapping_mul(2654435761) % 1009) as f32 / 1009.0 - 0.5)
        .collect();
    let mut idx_h = vec![0u32; r_max * n_active];
    for b in 0..r_max {
        let mut seen: Vec<u32> = Vec::with_capacity(n_active);
        let mut h = (b as u64)
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(1);
        while seen.len() < n_active {
            h = h
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let e = ((h >> 33) % n_expert as u64) as u32;
            if !seen.contains(&e) {
                seen.push(e);
            }
        }
        idx_h[b * n_active..(b + 1) * n_active].copy_from_slice(&seen);
    }
    let wt_h: Vec<f32> = (0..r_max * n_active)
        .map(|i| 0.1 + (i % 7) as f32 * 0.05)
        .collect();

    let xf = exec.to_device(&pat).expect("xf");
    let d_idx = exec.to_device_u32(&idx_h).expect("idx");
    let d_w = exec.to_device(&wt_h).expect("w");
    let d_sh_idx = exec.to_device_u32(&vec![0u32; r_max]).expect("sh_idx");
    let d_sh_w = exec.to_device(&vec![1.0f32; r_max]).expect("sh_w");

    let mut xq = exec.alloc_i8(r_max * embd).expect("xq");
    let mut xs = exec.alloc(r_max * embd / 32).expect("xs");
    let mut srow = exec.alloc_u32(nb_max * 32).expect("srow");
    let mut sslot = exec.alloc_u32(nb_max * 32).expect("sslot");
    let mut bexp = exec.alloc_u32(nb_max).expect("bexp");
    let mut srow_s = exec.alloc_u32(nbs_max * 32).expect("srow_s");
    let mut sslot_s = exec.alloc_u32(nbs_max * 32).expect("sslot_s");
    let mut bexp_s = exec.alloc_u32(nbs_max).expect("bexp_s");
    let mut fu_r = exec.alloc(nb_max * 32 * moe_ff).expect("fu_r");
    let mut fq_r = exec.alloc_i8(nb_max * 32 * moe_ff).expect("fq_r");
    let mut fs_r = exec.alloc(nb_max * 32 * moe_ff / 32).expect("fs_r");
    let mut fu_s = exec.alloc(nbs_max * 32 * shared_ff).expect("fu_s");
    let mut fq_s = exec.alloc_i8(nbs_max * 32 * shared_ff).expect("fq_s");
    let mut fs_s = exec.alloc(nbs_max * 32 * shared_ff / 32).expect("fs_s");
    let mut part = exec.alloc(r_max * n_active * embd).expect("part");
    let mut proj = exec.alloc(r_max * embd).expect("proj");
    let mut proj2 = exec.alloc(r_max * embd).expect("proj2");
    let mut resid = exec.alloc(r_max * embd).expect("resid");
    // dec2 activation planes are r*n_active wide, not nb*32 - the epilogue
    // quantize shrinks with them, which is a third of what this route saves
    let mut act_r = exec.alloc(r_max * n_active * moe_ff).expect("act_r");
    let mut aq_r = exec.alloc_i8(r_max * n_active * moe_ff).expect("aq_r");
    let mut as_r = exec.alloc(r_max * n_active * moe_ff / 32).expect("as_r");
    let mut act_s = exec.alloc(r_max * shared_ff).expect("act_s");
    let mut aq_s = exec.alloc_i8(r_max * shared_ff).expect("aq_s");
    let mut as_s = exec.alloc(r_max * shared_ff / 32).expect("as_s");

    let rows_pb: Vec<u32> = std::env::var("ROWS_PB")
        .ok()
        .map(|s| s.split(',').filter_map(|v| v.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![2, 4, 8, 16]);

    println!("\n== routed pair: sorted vs dec2 (us; GB/s over DEDUPED expert bytes) ==");
    print!("{:>3} {:>4} {:>4} {:>12}", "r", "nblk", "dist", "sorted");
    for p in &rows_pb {
        print!(" {:>12}", format!("dec2/{p}"));
    }
    println!();
    for &r in rows {
        exec.quantize_q8(&xf, &mut xq, &mut xs, r * embd).unwrap();
        let nbr = live_blocks(r, n_active, n_expert);
        // deduped bytes = what the sorted tile actually streams
        let mut seen: Vec<u32> = Vec::new();
        for e in &idx_h[..r * n_active] {
            if !seen.contains(e) {
                seen.push(*e);
            }
        }
        let ideal = seen.len() as f64 * e_bytes;
        let (reps, iters) = (5, 30);
        let t_sorted = time_us(&exec, reps, iters, || {
            exec.moe_align(
                &d_idx, &mut srow, &mut sslot, &mut bexp, r, n_active, n_expert, nbr,
            )
            .unwrap();
            exec.q8_0_moe_up_relu2_sorted(&up, &srow, &bexp, &xq, &xs, &mut fu_r, nbr)
                .unwrap();
            exec.quantize_q8(&fu_r, &mut fq_r, &mut fs_r, nbr * 32 * moe_ff)
                .unwrap();
            exec.q8_0_moe_down_sorted(
                &down, &srow, &sslot, &bexp, &d_w, &fq_r, &fs_r, &mut part, n_active, nbr,
            )
            .unwrap();
            exec.moe_slot_combine(&part, &mut resid, embd, n_active, r)
                .unwrap();
        });
        print!(
            "{r:>3} {nbr:>4} {:>4} {:>6.1}/{:>4.0}",
            seen.len(),
            t_sorted,
            ideal / (t_sorted * 1e-6) / 1e9
        );
        for &p in &rows_pb {
            let t = time_us(&exec, reps, iters, || {
                exec.q8_0_moe_up_relu2_dec2(&up, &d_idx, &xq, &xs, &mut act_r, n_active, r, p)
                    .unwrap();
                exec.quantize_q8(&act_r, &mut aq_r, &mut as_r, r * n_active * moe_ff)
                    .unwrap();
                exec.q8_0_moe_dn_dec2(&down, &d_idx, &d_w, &aq_r, &as_r, &mut proj, n_active, r)
                    .unwrap();
                exec.add(&mut resid, &proj, r * embd).unwrap();
            });
            print!(" {:>6.1}/{:>4.0}", t, ideal / (t * 1e-6) / 1e9);
        }
        println!();
    }

    println!("\n== shared expert: sorted 1-block vs dec2 vs DENSE ladder (us; GB/s) ==");
    println!(
        "{:>3} {:>12} {:>12} {:>12}",
        "r", "sorted", "dec2/8", "dense"
    );
    for &r in rows {
        exec.quantize_q8(&xf, &mut xq, &mut xs, r * embd).unwrap();
        let nbs = live_blocks(r, 1, 1);
        let (reps, iters) = (5, 40);
        let t_sorted = time_us(&exec, reps, iters, || {
            exec.moe_align(
                &d_sh_idx,
                &mut srow_s,
                &mut sslot_s,
                &mut bexp_s,
                r,
                1,
                1,
                nbs,
            )
            .unwrap();
            exec.q8_0_moe_up_relu2_sorted(&sh_up, &srow_s, &bexp_s, &xq, &xs, &mut fu_s, nbs)
                .unwrap();
            exec.quantize_q8(&fu_s, &mut fq_s, &mut fs_s, nbs * 32 * shared_ff)
                .unwrap();
            exec.q8_0_moe_down_sorted(
                &sh_down, &srow_s, &sslot_s, &bexp_s, &d_sh_w, &fq_s, &fs_s, &mut proj, 1, nbs,
            )
            .unwrap();
            exec.moe_slot_combine(&proj, &mut resid, embd, 1, r)
                .unwrap();
        });
        let t_dec2 = time_us(&exec, reps, iters, || {
            exec.q8_0_moe_up_relu2_dec2(&sh_up, &d_sh_idx, &xq, &xs, &mut act_s, 1, r, 8)
                .unwrap();
            exec.quantize_q8(&act_s, &mut aq_s, &mut as_s, r * shared_ff)
                .unwrap();
            exec.q8_0_moe_dn_dec2(&sh_down, &d_sh_idx, &d_sh_w, &aq_s, &as_s, &mut proj2, 1, r)
                .unwrap();
            exec.add(&mut resid, &proj2, r * embd).unwrap();
        });
        // the shared expert is a DENSE ffn - every row uses it - so the right
        // shape is the plain q8 ladder: one weight pass per tick regardless of
        // r, with relu^2 folded into the quantize between up and down
        let t_dense = time_us(&exec, reps, iters, || {
            dense_pre(&exec, &sh_up, &xq, &xs, &mut part, &mut act_s, r);
            exec.quantize_q8_relu2(&act_s, &mut aq_s, &mut as_s, r * shared_ff)
                .unwrap();
            dense_pre(&exec, &sh_down, &aq_s, &as_s, &mut part, &mut proj2, r);
            exec.add(&mut resid, &proj2, r * embd).unwrap();
        });
        println!(
            "{r:>3} {:>6.1}/{:>4.0} {:>6.1}/{:>4.0} {:>6.1}/{:>4.0}",
            t_sorted,
            sh_bytes / (t_sorted * 1e-6) / 1e9,
            t_dec2,
            sh_bytes / (t_dec2 * 1e-6) / 1e9,
            t_dense,
            sh_bytes / (t_dense * 1e-6) / 1e9
        );
    }

    // ---- agreement: the two routes are the same math in a different order.
    // Both fold the same topk weights over the same experts; only the
    // reduction grouping differs, so a large gap here is a layout bug.
    println!("\n== routed agreement at r=4 (sorted vs dec2/8) ==");
    let r = 4usize;
    exec.quantize_q8(&xf, &mut xq, &mut xs, r * embd).unwrap();
    let nbr = live_blocks(r, n_active, n_expert);
    exec.zero_region(&mut resid, 0, r * embd).unwrap();
    exec.moe_align(
        &d_idx, &mut srow, &mut sslot, &mut bexp, r, n_active, n_expert, nbr,
    )
    .unwrap();
    exec.q8_0_moe_up_relu2_sorted(&up, &srow, &bexp, &xq, &xs, &mut fu_r, nbr)
        .unwrap();
    exec.quantize_q8(&fu_r, &mut fq_r, &mut fs_r, nbr * 32 * moe_ff)
        .unwrap();
    exec.q8_0_moe_down_sorted(
        &down, &srow, &sslot, &bexp, &d_w, &fq_r, &fs_r, &mut part, n_active, nbr,
    )
    .unwrap();
    exec.moe_slot_combine(&part, &mut resid, embd, n_active, r)
        .unwrap();
    let a = exec.to_host_len(&resid, r * embd).unwrap();
    exec.q8_0_moe_up_relu2_dec2(&up, &d_idx, &xq, &xs, &mut act_r, n_active, r, 8)
        .unwrap();
    exec.quantize_q8(&act_r, &mut aq_r, &mut as_r, r * n_active * moe_ff)
        .unwrap();
    exec.q8_0_moe_dn_dec2(&down, &d_idx, &d_w, &aq_r, &as_r, &mut proj, n_active, r)
        .unwrap();
    let b = exec.to_host_len(&proj, r * embd).unwrap();
    let (mut md, mut mv) = (0f32, 0f32);
    for (x, z) in a.iter().zip(b.iter()) {
        md = md.max((x - z).abs());
        mv = mv.max(x.abs());
    }
    println!(
        "max|d| {md:.3e} over max|y| {mv:.3e}  (rel {:.2e})",
        md / mv.max(1e-30)
    );

    // ---- shared-expert agreement: the dense ladder must reproduce what the
    // sorted 1-block pair computes. Different reduction grouping, and on the
    // up half a different kernel entirely, so this is a tolerance check - but
    // a LAYOUT error reads as orders of magnitude, not ulps. It is also the
    // only coverage of the ks rung at in_dim 3712 (sh_down's K).
    println!("== shared agreement at r=4 (sorted vs dense) ==");
    let nbs = live_blocks(r, 1, 1);
    exec.zero_region(&mut resid, 0, r * embd).unwrap();
    exec.moe_align(
        &d_sh_idx,
        &mut srow_s,
        &mut sslot_s,
        &mut bexp_s,
        r,
        1,
        1,
        nbs,
    )
    .unwrap();
    exec.q8_0_moe_up_relu2_sorted(&sh_up, &srow_s, &bexp_s, &xq, &xs, &mut fu_s, nbs)
        .unwrap();
    exec.quantize_q8(&fu_s, &mut fq_s, &mut fs_s, nbs * 32 * shared_ff)
        .unwrap();
    exec.q8_0_moe_down_sorted(
        &sh_down, &srow_s, &sslot_s, &bexp_s, &d_sh_w, &fq_s, &fs_s, &mut proj, 1, nbs,
    )
    .unwrap();
    exec.moe_slot_combine(&proj, &mut resid, embd, 1, r)
        .unwrap();
    let a = exec.to_host_len(&resid, r * embd).unwrap();
    dense_pre(&exec, &sh_up, &xq, &xs, &mut part, &mut act_s, r);
    exec.quantize_q8_relu2(&act_s, &mut aq_s, &mut as_s, r * shared_ff)
        .unwrap();
    dense_pre(&exec, &sh_down, &aq_s, &as_s, &mut part, &mut proj2, r);
    let b = exec.to_host_len(&proj2, r * embd).unwrap();
    let (mut md, mut mv) = (0f32, 0f32);
    for (x, z) in a.iter().zip(b.iter()) {
        md = md.max((x - z).abs());
        mv = mv.max(x.abs());
    }
    println!(
        "max|d| {md:.3e} over max|y| {mv:.3e}  (rel {:.2e})",
        md / mv.max(1e-30)
    );

    // ---- the fused quantize must be BIT-identical to relu^2-into-f32
    // followed by the plain quantize; that equality is what lets the shared
    // expert change kernels without changing its numeric class.
    println!("== quantize_q8_relu2 vs relu2 + quantize_q8 ==");
    dense_pre(&exec, &sh_up, &xq, &xs, &mut part, &mut act_s, r);
    exec.quantize_q8_relu2(&act_s, &mut aq_s, &mut as_s, r * shared_ff)
        .unwrap();
    let fq_fused = exec.to_host_i8(&aq_s).unwrap();
    let fs_fused = exec.to_host_len(&as_s, r * shared_ff / 32).unwrap();
    let pre = exec.to_host_len(&act_s, r * shared_ff).unwrap();
    let relu2: Vec<f32> = pre
        .iter()
        .map(|v| {
            let t = v.max(0.0);
            t * t
        })
        .collect();
    let d_relu2 = exec.to_device(&relu2).unwrap();
    exec.quantize_q8(&d_relu2, &mut aq_s, &mut as_s, r * shared_ff)
        .unwrap();
    let fq_plain = exec.to_host_i8(&aq_s).unwrap();
    let fs_plain = exec.to_host_len(&as_s, r * shared_ff / 32).unwrap();
    let n = r * shared_ff;
    let qdiff = fq_fused[..n]
        .iter()
        .zip(&fq_plain[..n])
        .filter(|(a, b)| a != b)
        .count();
    let sdiff = fs_fused
        .iter()
        .zip(&fs_plain)
        .filter(|(a, b)| a != b)
        .count();
    println!(
        "int8 mismatches {qdiff} / {n}, scale mismatches {sdiff} / {}",
        n / 32
    );
}
