//! hibatch-lane M1 probe: the head->matvec->topk chain vs pd_moe_head_router_hb
//! on identical synthetic inputs. hb is PRECISION-class (bf16 smem normed rows
//! feeding the router dot), so the gates are top-k index agreement + weight/
//! output maxrel - not bitwise. Also times chain vs hb (the M1 speed gate).
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
    let eps = 1e-6f32;
    let up = |host: Vec<f32>| {
        let mut d = exec.stream.alloc_zeros::<f32>(host.len()).expect("alloc");
        exec.stream.memcpy_htod(&host, &mut d).expect("htod");
        d
    };
    let gamma = up((0..n).map(|i| 0.5 + 0.001 * (i % 37) as f32).collect());
    let pre2 = up((0..n).map(|i| 0.4 + 0.002 * (i % 29) as f32).collect());
    let rw_s = up((0..n_expert * n)
        .map(|i| ((i * 131 + 7) % 997) as f32 / 997.0 - 0.5)
        .collect());
    let dscale = up((0..n_expert)
        .map(|i| 0.9 + 0.001 * (i % 11) as f32)
        .collect());
    let rw = paddock_engine::gpu::DeviceTensor {
        buf: {
            let host: Vec<f32> = exec.stream.clone_dtoh(&rw_s).expect("d");
            let mut d = exec.stream.alloc_zeros::<f32>(host.len()).expect("a");
            exec.stream.memcpy_htod(&host, &mut d).expect("h");
            d
        },
        dims: vec![n, n_expert],
    };

    for r in [48usize, 64, 128] {
        let x = up((0..r * n)
            .map(|i| (((i * 61 + 13) % 1009) as f32 / 1009.0 - 0.5) * 3.0)
            .collect());
        // chain
        let mut rn = exec.stream.alloc_zeros::<f32>(r * n).expect("rn");
        let mut pn = exec.stream.alloc_zeros::<f32>(r * n).expect("pn");
        let mut q = exec.alloc_i8(r * n).expect("q");
        let mut qs = exec.stream.alloc_zeros::<f32>(r * n / 32).expect("qs");
        let mut logits = exec.stream.alloc_zeros::<f32>(r * n_expert).expect("lg");
        let mut idx = exec.alloc_u32(r * k).expect("idx");
        let mut w = exec.stream.alloc_zeros::<f32>(r * k).expect("w");
        exec.moe_head(
            &x, &gamma, &pre2, &mut rn, &mut pn, &mut q, &mut qs, n, eps, r,
        )
        .expect("head");
        exec.matvec_f32_raw(&rw_s, n, n_expert, &rn, &mut logits, r)
            .expect("mv");
        exec.moe_topk_scaled(&logits, &dscale, n_expert, k, &mut idx, &mut w, r)
            .expect("topk");
        let ci: Vec<u32> = exec.stream.clone_dtoh(&idx).expect("d");
        let cw: Vec<f32> = exec.stream.clone_dtoh(&w).expect("d");
        let cp: Vec<f32> = exec.stream.clone_dtoh(&pn).expect("d");
        let cq: Vec<i8> = exec.stream.clone_dtoh(&q).expect("d");
        let cs: Vec<f32> = exec.stream.clone_dtoh(&qs).expect("d");
        // hb
        let mut pn2 = exec.stream.alloc_zeros::<f32>(r * n).expect("pn2");
        let mut q2 = exec.alloc_i8(r * n).expect("q2");
        let mut qs2 = exec.stream.alloc_zeros::<f32>(r * n / 32).expect("qs2");
        let mut idx2 = exec.alloc_u32(r * k).expect("idx2");
        let mut w2 = exec.stream.alloc_zeros::<f32>(r * k).expect("w2");
        exec.moe_head_router_hb(
            &x, &gamma, &pre2, &rw, &dscale, &mut pn2, &mut q2, &mut qs2, &mut idx2, &mut w2, n,
            n_expert, k, eps, r,
        )
        .expect("hb");
        let fi: Vec<u32> = exec.stream.clone_dtoh(&idx2).expect("d");
        let fw: Vec<f32> = exec.stream.clone_dtoh(&w2).expect("d");
        let fp: Vec<f32> = exec.stream.clone_dtoh(&pn2).expect("d");
        let fq2: Vec<i8> = exec.stream.clone_dtoh(&q2).expect("d");
        let fs: Vec<f32> = exec.stream.clone_dtoh(&qs2).expect("d");
        let mi = ci.iter().zip(&fi).filter(|(a, b)| a != b).count();
        let mrel = |a: &Vec<f32>, b: &Vec<f32>| {
            a.iter()
                .zip(b)
                .map(|(x, y)| (x - y).abs() / x.abs().max(1e-3))
                .fold(0.0f32, f32::max)
        };
        let mq = cq.iter().zip(&fq2).filter(|(a, b)| a != b).count();
        println!(
            "r={r}: idx agree {}/{} (miss {mi})  w maxrel {:.2e}  pn maxrel {:.2e}  q mism {mq}/{}  qs maxrel {:.2e}",
            r * k - mi,
            r * k,
            mrel(&cw, &fw),
            mrel(&cp, &fp),
            r * n,
            mrel(&cs, &fs)
        );
        // timing (r=128 = the c128 shape)
        let t_chain = time_us(&exec, 200, || {
            exec.moe_head(
                &x, &gamma, &pre2, &mut rn, &mut pn, &mut q, &mut qs, n, eps, r,
            )
            .expect("head");
            exec.matvec_f32_raw(&rw_s, n, n_expert, &rn, &mut logits, r)
                .expect("mv");
            exec.moe_topk_scaled(&logits, &dscale, n_expert, k, &mut idx, &mut w, r)
                .expect("topk");
        });
        let t_hb = time_us(&exec, 200, || {
            exec.moe_head_router_hb(
                &x, &gamma, &pre2, &rw, &dscale, &mut pn2, &mut q2, &mut qs2, &mut idx2, &mut w2,
                n, n_expert, k, eps, r,
            )
            .expect("hb");
        });
        println!(
            "      chain {t_chain:7.1}us  hb {t_hb:7.1}us  ({:+.1}%)",
            100.0 * (t_hb / t_chain - 1.0)
        );
    }
}
