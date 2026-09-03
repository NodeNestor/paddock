//! Chunked DeltaNet PREFILL bench at the pf8 serving shape: one 2048-token
//! span, H=32, D=128, chunk C=64 -> nc=32. Times `gated_delta_chunked` (the
//! stage1 + stage2 pipeline the unified tick calls per Linear layer) with
//! L2-COLD per-layer state (30 cycling state buffers, like dn_v2_dec_bench)
//! and checks parity against the sequential `gated_delta_recurrent_v2_at`
//! oracle on the same inputs. PADDOCK_DNC_MMA / _MMA_G / _SCAN are process-
//! latched in the pack - run once per env combo for the A/B. Span length via
//! argv[1] (default 2048).
use std::sync::Arc;

use paddock_engine::gpu::GpuExecutor;

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(2048);
    let pack = std::env::var_os("PADDOCK_PACK")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../packs/cuda/build/pd-cuda-sm120.so")
        });
    let exec = Arc::new(GpuExecutor::new(0, &pack).expect("executor"));
    let (h, d) = (
        std::env::var("DNC_H")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(32usize),
        128usize,
    );
    let nc = n.div_ceil(64);
    let fill = |sz: usize, seed: u32| -> Vec<f32> {
        let mut s = seed;
        (0..sz)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                ((s >> 8) as f32 / (1u32 << 24) as f32) - 0.5
            })
            .collect()
    };
    let rows = n * h * d;
    let d_q = exec.to_device(&fill(rows, 1)).expect("q");
    let d_k = exec.to_device(&fill(rows, 2)).expect("k");
    let d_v = exec.to_device(&fill(rows, 3)).expect("v");
    // g in (-0.2, 0), beta in (0, 1) - the post-gate serving ranges
    let d_g = exec
        .to_device(
            &fill(n * h, 4)
                .iter()
                .map(|x| x * 0.2 - 0.1)
                .collect::<Vec<_>>(),
        )
        .expect("g");
    let d_beta = exec
        .to_device(&fill(n * h, 5).iter().map(|x| x + 0.5).collect::<Vec<_>>())
        .expect("beta");
    let mut d_out = exec.alloc(rows).expect("out");
    let mut d_out_ref = exec.alloc(rows).expect("out_ref");
    // chunked scratch, sized as qwen35.rs does
    let mut dw = exec.alloc(nc * h * 64 * d).expect("dw");
    let mut du = exec.alloc(nc * h * 64 * d).expect("du");
    let mut aqk = exec.alloc(nc * h * 64 * 64).expect("aqk");
    let mut cg = exec.alloc_f64(nc * h * 64).expect("cg");

    // parity: chunked vs sequential v2 from the same zero state
    let zeros = vec![0f32; h * d * d];
    let mut st_a = exec.to_device(&zeros).expect("st_a");
    let mut st_b = exec.to_device(&zeros).expect("st_b");
    exec.gated_delta_chunked(
        &d_q, &d_k, &d_v, &d_g, &d_beta, &mut st_a, 0, &mut d_out, &mut dw, &mut du, &mut aqk,
        &mut cg, n, h, d,
    )
    .expect("chunked");
    exec.gated_delta_recurrent_v2_at(
        &d_q,
        &d_k,
        &d_v,
        &d_g,
        &d_beta,
        &mut st_b,
        0,
        &mut d_out_ref,
        0,
        n,
        h,
        d,
    )
    .expect("v2 ref");
    exec.synchronize().expect("sync");
    let a = exec.to_host(&d_out).expect("a");
    let b = exec.to_host(&d_out_ref).expect("b");
    let (mut max_abs, mut max_rel) = (0f64, 0f64);
    for i in 0..a.len() {
        let e = (a[i] as f64 - b[i] as f64).abs();
        max_abs = max_abs.max(e);
        max_rel = max_rel.max(e / (b[i] as f64).abs().max(1e-3));
    }
    let sa = exec.to_host(&st_a).expect("sa");
    let sb = exec.to_host(&st_b).expect("sb");
    let mut st_err = 0f64;
    for i in 0..sa.len() {
        st_err = st_err.max((sa[i] as f64 - sb[i] as f64).abs());
    }
    println!(
        "parity vs v2: out max_abs {max_abs:.2e} max_rel {max_rel:.2e}  state max_abs {st_err:.2e}"
    );

    // v2 (the default) vs v1 walk, BITWISE (the kill switch is read per
    // launch, not latched)
    let a1 = exec.to_host(&d_out).expect("v2 out");
    let s1 = exec.to_host(&st_a).expect("v2 st");
    unsafe { std::env::set_var("PADDOCK_NO_DNC_MMA_V2", "1") };
    let mut st_c = exec.to_device(&zeros).expect("st_c");
    exec.gated_delta_chunked(
        &d_q, &d_k, &d_v, &d_g, &d_beta, &mut st_c, 0, &mut d_out, &mut dw, &mut du, &mut aqk,
        &mut cg, n, h, d,
    )
    .expect("v2 walk");
    exec.synchronize().expect("sync");
    let a2 = exec.to_host(&d_out).expect("v1 out");
    let s2 = exec.to_host(&st_c).expect("v1 st");
    let bad_o = a1
        .iter()
        .zip(&a2)
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count();
    let bad_s = s1
        .iter()
        .zip(&s2)
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count();
    println!(
        "v2 vs v1 walk: out {bad_o}/{} state {bad_s}/{} words differ {}",
        a1.len(),
        s1.len(),
        if bad_o + bad_s == 0 {
            "(bit-exact)"
        } else {
            "*** NOT BIT-EXACT ***"
        }
    );
    unsafe { std::env::remove_var("PADDOCK_NO_DNC_MMA_V2") };

    // timing, L2-cold states (each of ~30 Linear layers owns its state slot)
    const NL: usize = 30;
    let mut states: Vec<_> = (0..NL)
        .map(|_| exec.to_device(&zeros).expect("st"))
        .collect();
    let flop = nc as f64 * h as f64 * (4.0 * (d * 64 * d) as f64 + 2.0 * (64 * 64 * d) as f64);
    for pass in ["v1", "v2"] {
        if pass == "v1" {
            unsafe { std::env::set_var("PADDOCK_NO_DNC_MMA_V2", "1") };
        } else {
            unsafe { std::env::remove_var("PADDOCK_NO_DNC_MMA_V2") };
        }
        for li in 0..NL {
            exec.gated_delta_chunked(
                &d_q,
                &d_k,
                &d_v,
                &d_g,
                &d_beta,
                &mut states[li],
                0,
                &mut d_out,
                &mut dw,
                &mut du,
                &mut aqk,
                &mut cg,
                n,
                h,
                d,
            )
            .expect("warm");
        }
        exec.synchronize().expect("sync");
        let t0 = std::time::Instant::now();
        let iters = 90;
        for i in 0..iters {
            exec.gated_delta_chunked(
                &d_q,
                &d_k,
                &d_v,
                &d_g,
                &d_beta,
                &mut states[i % NL],
                0,
                &mut d_out,
                &mut dw,
                &mut du,
                &mut aqk,
                &mut cg,
                n,
                h,
                d,
            )
            .expect("bench");
        }
        exec.synchronize().expect("sync");
        let us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
        println!(
            "chunked[{pass}] n={n} nc={nc} COLD: {us:8.1} us  (~{:.1} TF effective f32-class)",
            flop / us / 1e6
        );
    }
}
