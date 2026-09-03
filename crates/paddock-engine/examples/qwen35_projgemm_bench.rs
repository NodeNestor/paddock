//! Prefill dense-projection GEMM microbench for qwen35 - the large-batch
//! prefill GEMM is the lever here. This times the current Q8 int8-MMA pipe
//! (`q8_0_gemm_mmq_pipe`) against the existing per-32-fold fp8 path
//! (`f8_gemm_w8`) at PREFILL batch sizes (512..4096) on the real 35B
//! dense-proj shapes, so we have a baseline and a clean single-kernel ncu
//! target. Cost is data-independent (zeroed acts fine).
//! Usage: QWEN35_GGUF=<path>/Qwen3.6-35B-A3B-Q8_0.gguf
//!        PADDOCK_PACK=packs/cuda/build/pd-cuda-sm120.so
//!        [PROJGEMM_NCU=1 to run one kernel/shape/batch for ncu]
//!        cargo run --release --example qwen35_projgemm_bench

use std::sync::Arc;

use paddock_engine::gpu::GpuExecutor;
use paddock_models::mapped::MappedGguf;

fn time_us(exec: &GpuExecutor, iters: usize, mut f: impl FnMut()) -> f64 {
    for _ in 0..10 {
        f();
    }
    exec.synchronize().expect("sync");
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        f();
    }
    exec.synchronize().expect("sync");
    t0.elapsed().as_secs_f64() * 1e6 / iters as f64
}

fn main() {
    let model = std::env::var("QWEN35_GGUF")
        .unwrap_or_else(|_| "/llms/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-Q8_0.gguf".to_string());
    let pack = std::env::var_os("PADDOCK_PACK")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../packs/cuda/build/pd-cuda-sm120.so")
        });
    let exec = Arc::new(GpuExecutor::new(0, &pack).expect("executor"));
    println!(
        "sm_count = {} has_f8_gemm_w8 = {}",
        exec.sm_count(),
        exec.has_f8_gemm_w8()
    );
    let map = MappedGguf::open(std::path::Path::new(&model)).expect("open gguf");

    // real 35B dense-proj shapes (GGUF dims are [in, out]); blk.0 = Linear, blk.3 = Full.
    let shapes: [(&str, &str); 3] = [
        ("in_qkv 2048->8192", "blk.0.attn_qkv.weight"),
        ("ssm_out 4096->2048", "blk.0.ssm_out.weight"),
        ("attn_q 2048->8192", "blk.3.attn_q.weight"),
    ];
    let ncu = std::env::var_os("PROJGEMM_NCU").is_some();
    let batches: &[usize] = if ncu {
        &[2048]
    } else {
        &[512, 1024, 2048, 4096]
    };

    let stages: u32 = std::env::var("PADDOCK_F8W8_STAGES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2);
    println!("PADDOCK_F8W8_STAGES = {stages}");

    // scratch sized for the largest shape/batch
    let (max_in, max_out, max_b) = (4096usize, 8192usize, 4096usize);
    // deterministic NON-ZERO activation so the parity checksum is meaningful
    let pat: Vec<f32> = (0..max_b * max_in)
        .map(|i| ((i as u64).wrapping_mul(2654435761) % 1009) as f32 / 1009.0 - 0.5)
        .collect();
    let xf = exec.to_device(&pat).expect("xf");
    // mmq activation plane: 144B per (128-K-chunk, padded-col); size generously.
    let mut yq_mmq = exec
        .alloc_u8(max_in.div_ceil(128) * ((max_b + 127) & !127) * 144)
        .expect("yq_mmq");
    // e4m3 activation planes (per-32 fold): i8 data [b*in], u8 scale [b*in/32]
    let mut xq_f8 = exec.alloc_i8(max_b * max_in).expect("xq_f8");
    let mut xs_f8 = exec.alloc_u8(max_b * max_in / 32).expect("xs_f8");
    let mut y = exec.alloc(max_b * max_out).expect("y");

    println!("-- prefill proj GEMM (us): mmq_pipe(Q8) | f8_gemm_w8(fp8 per-32) --");
    for (name, tensor) in shapes {
        let w = exec.repack_q8(&map, tensor).expect("repack");
        let (in_dim, out_dim) = (w.dims[0], w.dims[1]);
        let w8 = if exec.has_f8_gemm_w8() {
            exec.q8_0_to_f8w(&w).ok()
        } else {
            None
        };
        for &b in batches {
            // quantize activations for this batch (buffers oversized; kernel writes n)
            exec.quantize_q8_mmq(&xf, &mut yq_mmq, in_dim, b).unwrap();
            exec.quantize_e4m3(&xf, &mut xq_f8, &mut xs_f8, b * in_dim)
                .ok();

            let iters = if ncu { 1 } else { 200 };
            let t_pipe = time_us(&exec, iters, || {
                exec.q8_0_gemm_mmq_pipe(&w, None, &yq_mmq, &mut y, b)
                    .unwrap()
            });
            let (t_f8, csum) = if let Some(w8) = &w8 {
                let t = time_us(&exec, iters, || {
                    exec.f8_gemm_w8(w8, 0, &xq_f8, &xs_f8, &mut y, in_dim, out_dim, b)
                        .unwrap()
                });
                // parity checksum: sum of raw f32 bits of the b*out output (order-free)
                let hv = exec.to_host_len(&y, b * out_dim).expect("y->host");
                let c = hv
                    .iter()
                    .fold(0u64, |a, &v| a.wrapping_add(v.to_bits() as u64));
                (t, c)
            } else {
                (f64::NAN, 0)
            };
            // 2*M*N*K flops; report effective TFLOP/s for a rough SOL feel
            let flop = 2.0 * b as f64 * out_dim as f64 * in_dim as f64;
            let tf_pipe = flop / (t_pipe * 1e-6) / 1e12;
            let tf_f8 = flop / (t_f8 * 1e-6) / 1e12;
            println!(
                "{name:<20} B={b:<5} {t_pipe:>8.1} us ({tf_pipe:>5.1} TF) | f8 {t_f8:>8.1} us ({tf_f8:>5.1} TF) csum={csum:016x}"
            );
        }
    }
}
