//! B3-1 probe: chain (matvec_f32_raw + topk_scaled) vs the cooperative
//! router stage. Logit math is claimed VERBATIM => idx must be identical,
//! w bit-identical. Plus timing at r=64/128.
// A development probe: it runs on a box its author is looking at, and a
// failure should stop it where it happened rather than be reported.
#![allow(clippy::unwrap_used)]

use std::sync::Arc;

use paddock_engine::gpu::GpuExecutor;

fn time_us(exec: &GpuExecutor, iters: usize, mut f: impl FnMut()) -> f64 {
    for _ in 0..10 {
        f();
    }
    exec.stream.synchronize().expect("sync");
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        f();
    }
    exec.stream.synchronize().expect("sync");
    t0.elapsed().as_secs_f64() * 1e6 / iters as f64
}

fn main() {
    let pack = std::env::var("PADDOCK_PACK").expect("set PADDOCK_PACK");
    let exec = Arc::new(GpuExecutor::new(0, pack.as_ref()).expect("executor"));
    let (n, n_expert, k) = (2816usize, 128usize, 8usize);
    let up = |host: Vec<f32>| {
        let mut d = exec.stream.alloc_zeros::<f32>(host.len()).expect("a");
        exec.stream.memcpy_htod(&host, &mut d).expect("h");
        d
    };
    let rw_s = up((0..n_expert * n)
        .map(|i| ((i * 131 + 7) % 997) as f32 / 997.0 - 0.5)
        .collect());
    let dscale = up((0..n_expert)
        .map(|i| 0.9 + 0.001 * (i % 11) as f32)
        .collect());
    let rw = paddock_engine::gpu::DeviceTensor {
        buf: {
            let h: Vec<f32> = exec.stream.clone_dtoh(&rw_s).expect("d");
            let mut d = exec.stream.alloc_zeros::<f32>(h.len()).expect("a");
            exec.stream.memcpy_htod(&h, &mut d).expect("h");
            d
        },
        dims: vec![n, n_expert],
    };
    for r in [64usize, 128] {
        let x = up((0..r * n)
            .map(|i| (((i * 61 + 13) % 1009) as f32 / 1009.0 - 0.5) * 3.0)
            .collect());
        let mut lg1 = exec.stream.alloc_zeros::<f32>(r * n_expert).expect("l1");
        let mut idx1 = exec.alloc_u32(r * k).expect("i1");
        let mut w1 = exec.stream.alloc_zeros::<f32>(r * k).expect("w1");
        exec.matvec_f32_raw(&rw_s, n, n_expert, &x, &mut lg1, r)
            .expect("mv");
        exec.moe_topk_scaled(&lg1, &dscale, n_expert, k, &mut idx1, &mut w1, r)
            .expect("tk");
        let mut lg2 = exec.stream.alloc_zeros::<f32>(r * n_expert).expect("l2");
        let mut idx2 = exec.alloc_u32(r * k).expect("i2");
        let mut w2 = exec.stream.alloc_zeros::<f32>(r * k).expect("w2");
        exec.moe_router_stage(
            &rw, &x, &mut lg2, &dscale, &mut idx2, &mut w2, n, n_expert, r, k,
        )
        .expect("stage");
        let (a1, b1): (Vec<u32>, Vec<f32>) = (
            exec.stream.clone_dtoh(&idx1).expect("d"),
            exec.stream.clone_dtoh(&w1).expect("d"),
        );
        let (a2, b2): (Vec<u32>, Vec<f32>) = (
            exec.stream.clone_dtoh(&idx2).expect("d"),
            exec.stream.clone_dtoh(&w2).expect("d"),
        );
        let l1h: Vec<f32> = exec.stream.clone_dtoh(&lg1).expect("d");
        let l2h: Vec<f32> = exec.stream.clone_dtoh(&lg2).expect("d");
        let ml = l1h
            .iter()
            .zip(&l2h)
            .filter(|(p, q)| p.to_bits() != q.to_bits())
            .count();
        let mi = a1.iter().zip(&a2).filter(|(p, q)| p != q).count();
        let mw = b1
            .iter()
            .zip(&b2)
            .filter(|(p, q)| p.to_bits() != q.to_bits())
            .count();
        let t_chain = time_us(&exec, 300, || {
            exec.matvec_f32_raw(&rw_s, n, n_expert, &x, &mut lg1, r)
                .expect("mv");
            exec.moe_topk_scaled(&lg1, &dscale, n_expert, k, &mut idx1, &mut w1, r)
                .expect("tk");
        });
        let t_stage = time_us(&exec, 300, || {
            exec.moe_router_stage(
                &rw, &x, &mut lg2, &dscale, &mut idx2, &mut w2, n, n_expert, r, k,
            )
            .expect("st");
        });
        println!(
            "r={r}: logits mism {ml}/{}  idx mism {mi}/{}  w mism {mw}/{}  | chain {t_chain:6.1}us  stage {t_stage:6.1}us ({:+.1}%)",
            r * n_expert,
            r * k,
            r * k,
            100.0 * (t_stage / t_chain - 1.0)
        );
    }
}
