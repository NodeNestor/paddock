//! Offline bitwise probe for the head+router+topk fusion (slot 487): the
//! three-launch chain vs pd_moe_head_router on identical synthetic inputs.
//! The serve-side burst gate is unsound for bitwise claims (admission timing
//! legitimately changes batched temp-0 outputs) - this is the sound one.
// A development probe: it runs on a box its author is looking at, and a
// failure should stop it where it happened rather than be reported.
#![allow(clippy::unwrap_used)]

use std::sync::Arc;

use paddock_engine::gpu::GpuExecutor;

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
    let rw = up((0..n_expert * n)
        .map(|i| ((i * 131 + 7) % 997) as f32 / 997.0 - 0.5)
        .collect());
    let dscale = up((0..n_expert)
        .map(|i| 0.9 + 0.001 * (i % 11) as f32)
        .collect());

    for r in [16usize, 32, 128] {
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
        exec.matvec_f32_raw(&rw, n, n_expert, &rn, &mut logits, r)
            .expect("mv");
        exec.moe_topk_scaled(&logits, &dscale, n_expert, k, &mut idx, &mut w, r)
            .expect("topk");
        let (ci, cw): (Vec<u32>, Vec<f32>) = (
            exec.stream.clone_dtoh(&idx).expect("d"),
            exec.stream.clone_dtoh(&w).expect("d"),
        );
        let (cp, cq, cs): (Vec<f32>, Vec<i8>, Vec<f32>) = (
            exec.stream.clone_dtoh(&pn).expect("d"),
            exec.stream.clone_dtoh(&q).expect("d"),
            exec.stream.clone_dtoh(&qs).expect("d"),
        );
        // fused (fresh outputs)
        let mut pn2 = exec.stream.alloc_zeros::<f32>(r * n).expect("pn2");
        let mut q2 = exec.alloc_i8(r * n).expect("q2");
        let mut qs2 = exec.stream.alloc_zeros::<f32>(r * n / 32).expect("qs2");
        let mut idx2 = exec.alloc_u32(r * k).expect("idx2");
        let mut w2 = exec.stream.alloc_zeros::<f32>(r * k).expect("w2");
        let rwt = paddock_engine::gpu::DeviceTensor {
            buf: rw_clone(&exec, &rw),
            dims: vec![n, n_expert],
        };
        exec.moe_head_router(
            &x, &gamma, &pre2, &rwt, &dscale, &mut pn2, &mut q2, &mut qs2, &mut idx2, &mut w2, n,
            n_expert, k, eps, r,
        )
        .expect("fused");
        let (fi, fw): (Vec<u32>, Vec<f32>) = (
            exec.stream.clone_dtoh(&idx2).expect("d"),
            exec.stream.clone_dtoh(&w2).expect("d"),
        );
        let (fp, fq2, fs): (Vec<f32>, Vec<i8>, Vec<f32>) = (
            exec.stream.clone_dtoh(&pn2).expect("d"),
            exec.stream.clone_dtoh(&q2).expect("d"),
            exec.stream.clone_dtoh(&qs2).expect("d"),
        );
        let mi = ci.iter().zip(&fi).filter(|(a, b)| a != b).count();
        let mw = cw
            .iter()
            .zip(&fw)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        let mp = cp
            .iter()
            .zip(&fp)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        let mq = cq.iter().zip(&fq2).filter(|(a, b)| a != b).count();
        let ms = cs
            .iter()
            .zip(&fs)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        println!("r={r}: idx {mi} w {mw} pn {mp} q {mq} qs {ms} mismatches");
        if mi + mw > 0 {
            for s in 0..r * k {
                if ci[s] != fi[s] || cw[s].to_bits() != fw[s].to_bits() {
                    println!(
                        "  first: slot {s} chain=({}, {:.9e}) fused=({}, {:.9e})",
                        ci[s], cw[s], fi[s], fw[s]
                    );
                    break;
                }
            }
        }
    }
}

fn rw_clone(
    exec: &GpuExecutor,
    src: &cudarc::driver::CudaSlice<f32>,
) -> cudarc::driver::CudaSlice<f32> {
    let host: Vec<f32> = exec.stream.clone_dtoh(src).expect("d");
    let mut d = exec.stream.alloc_zeros::<f32>(host.len()).expect("a");
    exec.stream.memcpy_htod(&host, &mut d).expect("h");
    d
}
