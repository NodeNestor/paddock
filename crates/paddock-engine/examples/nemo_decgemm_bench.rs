//! Nemotron BATCHED-DECODE dense-projection crossover.
//!
//! The GGUF lane's r>1 arm goes `prefill_quant` + `prefill_mm_pre_any`, which
//! for batch <= 64 lands on `q8_0_gemm_mma` - the shared-staging MT tile built
//! for prefill row counts. qwen35's decode ladder instead takes
//! `q8_0_gemv_dp4a_nc` at r <= 4 and the K-split `q8_0_gemm_mma_ks` up to 64.
//! This prices all three (plus the naive r-separate GEMVs) at nemotron's real
//! dense shapes so the crossover is measured, not assumed.
//!
//! Shapes come from the 30B-A3B checkpoint: hidden 2688, mamba in_proj rows
//! 10304 (z 4096 | xBC 6144 | dt 64), d_inner 4096, 32 q-heads / 2 kv-heads at
//! head_dim 128, shared expert ff 3712, vocab 131072. Layer 0 is mamba, 1 is
//! MoE, 5 is the first attention block.
//!
//! Usage (static pack):
//!   cargo run --release -p paddock-engine --features static-pack \
//!     --example nemo_decgemm_bench
//! Usage (pack file):
//!   PADDOCK_PACK=packs/cuda/build/pd-cuda-sm86.dll cargo run --release \
//!     -p paddock-engine --example nemo_decgemm_bench

use std::sync::Arc;

use paddock_engine::gpu::GpuExecutor;
use paddock_models::mapped::MappedGguf;

/// min-of-`reps` mean-of-`iters` - a single pass puts two launches of the
/// identical config 20% apart on this die.
fn time_us(exec: &GpuExecutor, reps: usize, iters: usize, mut f: impl FnMut()) -> f64 {
    for _ in 0..10 {
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
        let us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
        best = best.min(us);
    }
    best
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

    // GGUF dims are [in, out]. Every plane a nemotron decode tick touches on
    // the Q8_0 lane, in per-tick byte order.
    let shapes: [(&str, &str); 7] = [
        ("ssm_in   2688->10304", "blk.0.ssm_in.weight"),
        ("ssm_out  4096->2688", "blk.0.ssm_out.weight"),
        ("attn_q   2688->4096", "blk.5.attn_q.weight"),
        ("attn_k   2688->256", "blk.5.attn_k.weight"),
        ("attn_o   4096->2688", "blk.5.attn_output.weight"),
        ("shexp_up 2688->3712", "blk.1.ffn_up_shexp.weight"),
        ("lm_head  2688->131072", "output.weight"),
    ];
    let rows: &[usize] = &[1, 2, 3, 4, 5, 6, 8, 12, 16, 24, 32];

    let (max_in, max_out, max_r) = (4096usize, 131072usize, 32usize);
    // deterministic non-zero activation so the parity numbers mean something
    let pat: Vec<f32> = (0..max_r * max_in)
        .map(|i| ((i as u64).wrapping_mul(2654435761) % 1009) as f32 / 1009.0 - 0.5)
        .collect();
    let xf = exec.to_device(&pat).expect("xf");
    let mut xq = exec.alloc_i8(max_r * max_in).expect("xq");
    let mut xs = exec.alloc(max_r * max_in / 32).expect("xs");
    let mut y = exec.alloc(max_r * max_out).expect("y");
    // K-split fixup plane: the 64-row envelope over every shape but lm_head
    let ks_cap = 8 * 64 * 10304;
    let mut part = exec.alloc(ks_cap).expect("part");

    println!("\n-- us (GB/s over weight bytes); r*gemv = r separate q8_0_gemv_repacked --");
    println!(
        "{:<22} {:>3} {:>10} {:>10} {:>10} {:>10}",
        "shape", "r", "r*gemv", "nc", "mma", "mma_ks"
    );
    for (name, tensor) in shapes {
        let w = exec.repack_q8(&map, tensor).expect("repack");
        let (in_dim, out_dim) = (w.dims[0], w.dims[1]);
        // weight bytes: int8 data + one f16 scale per 32
        let wbytes = in_dim as f64 * out_dim as f64 * (1.0 + 2.0 / 32.0);
        let ks_fits = part.len() >= 8 * 64 * out_dim;
        for &r in rows {
            exec.quantize_q8(&xf, &mut xq, &mut xs, r * in_dim).unwrap();
            let reps = 5;
            let iters = if out_dim > 60000 { 20 } else { 60 };
            let t_gemv = time_us(&exec, reps, iters, || {
                for _ in 0..r {
                    exec.q8_0_gemv_repacked(&w, None, &xf, &mut y).unwrap()
                }
            });
            let t_nc = if r <= 8 {
                time_us(&exec, reps, iters, || {
                    exec.q8_0_gemv_dp4a_nc(&w, &xq, &xs, &mut y, r).unwrap()
                })
            } else {
                f64::NAN
            };
            let t_mma = time_us(&exec, reps, iters, || {
                exec.q8_0_gemm_mma(&w, &xq, &xs, &mut y, r).unwrap()
            });
            let t_ks = if ks_fits {
                time_us(&exec, reps, iters, || {
                    exec.q8_0_gemm_mma_ks(&w, &xq, &xs, &mut part, &mut y, r)
                        .unwrap()
                })
            } else {
                f64::NAN
            };
            let bw = |t: f64| wbytes / (t * 1e-6) / 1e9;
            println!(
                "{name:<22} {r:>3} {:>6.1}/{:>3.0} {:>6.1}/{:>3.0} {:>6.1}/{:>3.0} {:>6.1}/{:>3.0}",
                t_gemv,
                bw(t_gemv),
                t_nc,
                bw(t_nc),
                t_mma,
                bw(t_mma),
                t_ks,
                bw(t_ks)
            );
        }
        // agreement check at r=4: nc vs mma read the same xq/xs, so they differ
        // only by accumulation order - a big gap here means a layout bug, not
        // rounding.
        exec.quantize_q8(&xf, &mut xq, &mut xs, 4 * in_dim).unwrap();
        exec.q8_0_gemv_dp4a_nc(&w, &xq, &xs, &mut y, 4).unwrap();
        let a = exec.to_host_len(&y, 4 * out_dim).unwrap();
        exec.q8_0_gemm_mma(&w, &xq, &xs, &mut y, 4).unwrap();
        let b = exec.to_host_len(&y, 4 * out_dim).unwrap();
        let (mut md, mut mv) = (0f32, 0f32);
        for (x, z) in a.iter().zip(b.iter()) {
            md = md.max((x - z).abs());
            mv = mv.max(x.abs());
        }
        println!("  r=4 nc vs mma: max|d| {md:.3e} over max|y| {mv:.3e}");
    }
}
