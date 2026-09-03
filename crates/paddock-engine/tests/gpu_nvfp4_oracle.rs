//! Checkpoint-NVFP4 tensor-level oracle: the CUDA consumers of a
//! modelopt W4A16_NVFP4 triple against the host reference in
//! `paddock_models::modelopt` (which is itself pinned to an independent
//! Python/numpy implementation via exact f32 bit patterns).
//!
//!   1. `nvf4_dequant_f32`  -> BIT-exact vs the host reference, whole tensors
//!   2. `nvf4_gemv`         -> f64 host matvec, rel-err gated (accumulation
//!      order differs; nibble/scale bugs are O(1) off)
//!
//! Gated on: CUDA device + built pack + the Nemotron NVFP4 checkpoint.

mod common;

use paddock_models::modelopt::nvfp4_view;
use paddock_models::safetensors::ShardedSafetensors;

const CKPT_ENV: &str = "NEMOTRON_NVFP4_DIR";
const CKPT_DIR: &str = "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4";

/// Real planes across the geometry range: first expert, a far expert in the
/// last moe layer, and the wide shared expert.
const PREFIXES: [&str; 3] = [
    "backbone.layers.1.mixer.experts.0.up_proj",
    "backbone.layers.51.mixer.experts.127.down_proj",
    "backbone.layers.1.mixer.shared_experts.up_proj",
];

fn checkpoint() -> Option<ShardedSafetensors> {
    let dir = common::model_dir(CKPT_ENV, &[CKPT_DIR])?;
    match ShardedSafetensors::open_dir(&dir) {
        Ok(st) => Some(st),
        Err(e) => {
            common::missing(&format!("nvfp4 checkpoint unreadable: {e}"));
            None
        }
    }
}

fn deterministic_input(n: usize, seed: u64) -> Vec<f32> {
    // LCG -> [-1, 1); enough spread to exercise every scale block.
    let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15).max(1);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / (1u64 << 31) as f32) - 1.0
        })
        .collect()
}

#[test]
fn nvf4_dequant_bit_matches_host_reference() {
    let Some(exec) = common::gpu() else { return };
    let Some(st) = checkpoint() else { return };
    if !exec.has_nvf4_ckpt() {
        common::missing("pack has no nvf4 checkpoint consumers (cc != 12.0?)");
        return;
    }
    for prefix in PREFIXES {
        let v = nvfp4_view(&st, prefix).expect(prefix);
        let plane = exec
            .nvf4_upload(v.packed, v.scales, v.scale2, v.n, v.k)
            .expect("upload");
        let mut d_y = exec.alloc(v.n * v.k).expect("y");
        exec.nvf4_dequant_f32(&plane, &mut d_y).expect("dequant");
        let gpu = exec.to_host(&d_y).expect("host");
        let mut diffs = 0usize;
        for row in 0..v.n {
            let host = v.dequant_row_f32(row);
            for col in 0..v.k {
                if gpu[row * v.k + col].to_bits() != host[col].to_bits() {
                    if diffs == 0 {
                        eprintln!(
                            "{prefix} [{row}, {col}]: gpu {:#010x} host {:#010x}",
                            gpu[row * v.k + col].to_bits(),
                            host[col].to_bits()
                        );
                    }
                    diffs += 1;
                }
            }
        }
        assert_eq!(
            diffs,
            0,
            "{prefix}: {diffs} of {} elements differ",
            v.n * v.k
        );
        println!("{prefix}: [{}, {}] bit-exact", v.n, v.k);
    }
}

#[test]
fn nvf4_gemv_matches_f64_reference() {
    let Some(exec) = common::gpu() else { return };
    let Some(st) = checkpoint() else { return };
    if !exec.has_nvf4_ckpt() {
        common::missing("pack has no nvf4 checkpoint consumers (cc != 12.0?)");
        return;
    }
    for prefix in PREFIXES {
        let v = nvfp4_view(&st, prefix).expect(prefix);
        let plane = exec
            .nvf4_upload(v.packed, v.scales, v.scale2, v.n, v.k)
            .expect("upload");
        let x = deterministic_input(v.k, 11);
        let d_x = exec.to_device(&x).expect("x");
        let mut d_y = exec.alloc(v.n).expect("y");
        exec.nvf4_gemv(&plane, &d_x, &mut d_y, None).expect("gemv");
        let y = exec.to_host(&d_y).expect("host");
        let mut max_rel = 0f64;
        for row in 0..v.n {
            let host = v.dequant_row_f32(row);
            let (mut want, mut mag) = (0f64, 0f64);
            for (&w, &xi) in host.iter().zip(&x) {
                let t = w as f64 * xi as f64;
                want += t;
                mag += t.abs();
            }
            let got = y[row] as f64;
            // Gate against the dot's CONDITION (sum of |terms|), not |result|:
            // a near-cancelling row has |result| << mag and f32 accumulation
            // error scales with mag. Format bugs (nibble order, scale
            // addressing, scale2) are O(mag) and still fail loudly.
            let rel = (got - want).abs() / mag.max(1e-6);
            if rel > max_rel {
                max_rel = rel;
            }
            assert!(
                rel < 1e-5,
                "{prefix} row {row}: gemv {got:e} vs reference {want:e} (rel-to-mag {rel:e})"
            );
        }
        println!(
            "{prefix}: [{}, {}] max rel-to-mag err {max_rel:.3e}",
            v.n, v.k
        );
    }
}

/// Decode-band bandwidth spot on the largest real plane (lm_head,
/// 131072 x 2688): the SOTA acceptance metric for the GEMV is streamed
/// weight+scale bytes vs the card's measured practical roof (~1531 GB/s on
/// GB202). Prints; asserts only a sanity floor so slow
/// regressions surface without making the gate flaky.
#[test]
fn nvf4_gemv_bandwidth_spot() {
    let Some(exec) = common::gpu() else { return };
    let Some(st) = checkpoint() else { return };
    if !exec.has_nvf4_ckpt() {
        common::missing("pack has no nvf4 checkpoint consumers (cc != 12.0?)");
        return;
    }
    let v = nvfp4_view(&st, "lm_head").expect("lm_head");
    let plane = exec
        .nvf4_upload(v.packed, v.scales, v.scale2, v.n, v.k)
        .expect("upload");
    let x = deterministic_input(v.k, 3);
    let d_x = exec.to_device(&x).expect("x");
    let mut d_y = exec.alloc(v.n).expect("y");
    for _ in 0..3 {
        exec.nvf4_gemv(&plane, &d_x, &mut d_y, None)
            .expect("warmup");
    }
    exec.synchronize().expect("sync");
    let iters = 20u32;
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        exec.nvf4_gemv(&plane, &d_x, &mut d_y, None).expect("gemv");
    }
    exec.synchronize().expect("sync");
    let dt = t0.elapsed().as_secs_f64() / iters as f64;
    let bytes = (v.packed.len() + v.scales.len()) as f64;
    let gbs = bytes / dt / 1e9;
    println!(
        "lm_head [{}, {}] gemv: {:.1} us, {gbs:.0} GB/s streamed (roof ~1531)",
        v.n,
        v.k,
        dt * 1e6
    );
    // Tuning ladder (ncu/bench-guided): 621 (one-row CTA, one loop iter at
    // K=2688 = latency-bound) -> 910 (warp-per-row x8) -> 1077
    // (warp-COHERENT lanes; the lane-owns-chunk layout was at the L1/TEX
    // wall, 97.9% with DRAM at 46.6%) -> 1442 = 94% of the practical roof
    // (the in-loop ragged-tail break blocked ptxas load
    // pipelining - hoisted; bench/nemo_lmhead_gemv_bench.cu swept the
    // alternatives and none beat it). Floor trips under the measured 1442.
    //
    // The floor is PER-DIE, because GB/s is. The 1442 above and the ~1531
    // roof it is 94% of are GB202 numbers; asserting them on another card
    // measures the card, not a regression.
    //
    // sm_100 (B200) bootstrap, the first legs after the NVFP4
    // family was un-gated for this die: 1229 GB/s. That is not a healthy
    // number and this floor must not be read as an acceptance bar - B200
    // streams ~8 TB/s of HBM3e, so the same kernel that reaches 94% of roof
    // on GB202 is at roughly a SIXTH of roof here. The geometry is tuned for
    // a 1.5-1.8 TB/s part: 8 warp-rows per CTA and a 128-element coherent
    // step were sized against GB202's latency/bandwidth product. Re-tuning it
    // is a named lever for nemotron on that die, not a defect to hide, so the
    // floor is set just under the measured value to catch a REGRESSION while
    // the headroom stays visible in the printed line above.
    let floor = if exec.compute_capability().0 >= 12 {
        1300.0
    } else {
        1150.0
    };
    assert!(
        gbs > floor,
        "nvf4 gemv regressed: {gbs:.0} GB/s, floor {floor:.0} \
         (sm_{}{} - GB202 measured 1442, B200 bootstrap 1229)",
        exec.compute_capability().0,
        exec.compute_capability().1,
    );
}

/// W4A4 GEMM (slot 426): device-quantize activations to nvf4,
/// run the fp4 x fp4 block-scale mma, and check every sampled output against
/// an f64 host GEMM over the same quantized operands - the host dequants the
/// read-back xq/xs planes with the pinned e2m1/e4m3 decoders, so activation
/// quantization cancels out of the diff and the gate isolates fragment
/// layout, scale addressing, and the scale2 epilogue. Batches cover the
/// dispatch band edge (9), the c32 decode shape (32), and a two-col-tile
/// case with a ragged tail (200).
#[test]
fn nvf4_gemm_f4_matches_quantized_reference() {
    let Some(exec) = common::gpu() else { return };
    let Some(st) = checkpoint() else { return };
    if !exec.has_nvf4_ckpt() || !exec.has_nvf4_gemm_f4() {
        common::missing("pack has no nvf4 W4A4 GEMM (cc != 12.0?)");
        return;
    }
    use paddock_models::modelopt::{e2m1_to_f32, e4m3_to_f32};
    for prefix in PREFIXES {
        let v = nvfp4_view(&st, prefix).expect(prefix);
        let plane = exec
            .nvf4_upload(v.packed, v.scales, v.scale2, v.n, v.k)
            .expect("upload");
        for &batch in &[9usize, 32, 200] {
            let x = deterministic_input(batch * v.k, 7 + batch as u64);
            let d_x = exec.to_device(&x).expect("x");
            let mut d_xq = exec.alloc_i8(batch * v.k / 2).expect("xq");
            let mut d_xs = exec.alloc_u8(batch * v.k / 16).expect("xs");
            exec.quantize_nvf4(&d_x, &mut d_xq, &mut d_xs, batch * v.k)
                .expect("quantize");
            let mut d_y = exec.alloc(batch * v.n).expect("y");
            // part present -> the split/v2 routes; the expert-plane grids
            // here are all under 64 tiles, so this exercises split-K
            let mut d_part = exec.alloc(4 * batch * v.n).expect("part");
            exec.nvf4_gemm_f4(
                &plane,
                &d_xq,
                &d_xs,
                &mut d_y,
                None,
                batch,
                Some(&mut d_part),
            )
            .expect("gemm");
            let y = exec.to_host(&d_y).expect("y host");
            let xq = exec.to_host_i8(&d_xq).expect("xq host");
            let xs = exec
                .to_host_range_u8(&d_xs, 0, batch * v.k / 16)
                .expect("xs host");
            // exact host dequant of the quantized activations
            let mut a = vec![0f64; batch * v.k];
            for (e, ae) in a.iter_mut().enumerate() {
                let nib = if e & 1 == 0 {
                    xq[e >> 1] as u8 & 0xF
                } else {
                    (xq[e >> 1] as u8) >> 4
                };
                *ae = (e2m1_to_f32(nib) * e4m3_to_f32(xs[e >> 4])) as f64;
            }
            // sample rows: full coverage is O(n*k*batch) host f64 - the first
            // and last CTA row-tiles plus a mid stride cover every fragment
            // position without minutes of host math
            let rows: Vec<usize> = (0..v.n)
                .filter(|r| *r < 130 || *r >= v.n - 130 || r % 97 == 0)
                .collect();
            let mut max_rel = 0f64;
            for &row in &rows {
                let host_w = v.dequant_row_f32(row);
                for col in 0..batch {
                    let (mut want, mut mag) = (0f64, 0f64);
                    for k in 0..v.k {
                        let t = host_w[k] as f64 * a[col * v.k + k];
                        want += t;
                        mag += t.abs();
                    }
                    let got = y[col * v.n + row] as f64;
                    let rel = (got - want).abs() / mag.max(1e-6);
                    if rel > max_rel {
                        max_rel = rel;
                    }
                    assert!(
                        rel < 1e-5,
                        "{prefix} b{batch} [{row}, {col}]: {got:e} vs {want:e} (rel-to-mag {rel:e})"
                    );
                }
            }
            println!(
                "{prefix} b{batch}: [{}, {}] {} rows sampled, max rel-to-mag {max_rel:.3e}",
                v.n,
                v.k,
                rows.len()
            );
        }
    }
}
