//! Router f32 GEMM A/B: the tiled `pd_gemm_f32_nt_kernel` (PADDOCK_ROUTER_GEMM)
//! vs the legacy matvec-tile path, at the qwen35 router shape ([batch, 2048] x
//! [256, 2048]^T). Checks max rel err against a CPU f64 reference (the two GPU
//! paths differ in accumulation order, so compare both to the oracle, not to
//! each other bit-wise) and times each. Env PADDOCK_ROUTER_GEMM is latched
//! statically in the pack, so run twice (on/off) for the speed A/B - the
//! parity check is self-contained either way.
// A development probe: it runs on a box its author is looking at, and a
// failure should stop it where it happened rather than be reported.
#![allow(clippy::unwrap_used)]

use std::sync::Arc;

use paddock_engine::gpu::{DeviceTensor, GpuExecutor};

fn main() {
    let pack = std::env::var_os("PADDOCK_PACK")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../packs/cuda/build/pd-cuda-sm120.so")
        });
    let exec = Arc::new(GpuExecutor::new(0, &pack).expect("executor"));
    let (in_dim, out_dim) = (2048usize, 256usize);
    // deterministic pseudo-random fill, logits-scale values
    let fill = |n: usize, seed: u32| -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                ((s >> 8) as f32 / (1u32 << 24) as f32) - 0.5
            })
            .collect()
    };
    let w = fill(out_dim * in_dim, 1);
    let d_w = DeviceTensor {
        buf: exec.to_device(&w).expect("w"),
        dims: vec![in_dim, out_dim],
    };
    let mode = if std::env::var_os("PADDOCK_ROUTER_GEMM").is_some() {
        "GEMM"
    } else {
        "matvec"
    };
    for &batch in &[32usize, 256, 1024, 2048] {
        let x = fill(batch * in_dim, 2);
        let d_x = exec.to_device(&x).expect("x");
        let mut d_out = exec.alloc(batch * out_dim).expect("out");
        exec.matvec_f32_batch(&d_w, &d_x, &mut d_out, batch)
            .expect("router");
        let got = exec.to_host(&d_out).expect("dtoh");
        // f64 oracle on sampled rows
        let mut max_rel = 0f64;
        for m in (0..batch).step_by((batch / 16).max(1)) {
            for o in 0..out_dim {
                let mut acc = 0f64;
                for i in 0..in_dim {
                    acc += w[o * in_dim + i] as f64 * x[m * in_dim + i] as f64;
                }
                let g = got[m * out_dim + o] as f64;
                let rel = (g - acc).abs() / acc.abs().max(1e-3);
                if rel > max_rel {
                    max_rel = rel;
                }
            }
        }
        exec.synchronize().expect("sync");
        let t0 = std::time::Instant::now();
        for _ in 0..50 {
            exec.matvec_f32_batch(&d_w, &d_x, &mut d_out, batch)
                .expect("router");
        }
        exec.synchronize().expect("sync");
        let us = t0.elapsed().as_secs_f64() * 1e6 / 50.0;
        println!("batch {batch:5}: {us:8.1} us  max_rel_err {max_rel:.2e}  ({mode})");
    }
}
