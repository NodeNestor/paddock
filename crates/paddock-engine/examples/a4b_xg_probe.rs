//! P1-2 probe: per-32 chain (mma2) vs per-128 xg pair (mma2g with host-made
//! per-128 xs) on identical synthetic weights + activations. Reports fq
//! maxrel (expect quantize-class ~1.3x of per-32 error) and gu timing.
use std::sync::Arc;

use paddock_engine::gpu::{GpuExecutor, RepackedQ8};

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
    let (n_e, ff, embd, k) = (128usize, 704usize, 2816usize, 8usize);
    let mk = |in_d: usize, out: usize, seed: usize| -> RepackedQ8 {
        let blocks = n_e * out * in_d / 32;
        let host: Vec<u8> = (0..blocks * 32)
            .map(|i| ((i * 13 + 5 + seed) % 255) as u8)
            .collect();
        let mut d = exec.alloc_u8(blocks * 32).expect("data");
        exec.stream.memcpy_htod(&host, &mut d).expect("h");
        const H: [u16; 8] = [
            0x3c00, 0x3800, 0x3a00, 0x3e00, 0x3400, 0x4000, 0x3d00, 0x3900,
        ];
        let hs: Vec<u16> = (0..blocks).map(|i| H[(i + seed) % 8]).collect();
        let bytes: Vec<u8> = hs.iter().flat_map(|h| h.to_le_bytes()).collect();
        let mut sc = exec.alloc_u8(blocks * 2).expect("scale");
        exec.stream.memcpy_htod(&bytes, &mut sc).expect("h");
        RepackedQ8 {
            data: d,
            scale: sc,
            dims: vec![in_d, out, n_e],
        }
    };
    let gate = mk(embd, ff, 0);
    let up = mk(embd, ff, 7);

    for r in [64usize, 128] {
        // realistic-ish activations with mild outliers
        let x: Vec<f32> = (0..r * embd)
            .map(|i| {
                let base = (((i * 61 + 13) % 1009) as f32 / 1009.0 - 0.5) * 2.0;
                if i % 977 == 0 { base * 9.0 } else { base }
            })
            .collect();
        // per-32 quantize (host)
        let q32: Vec<i8> = (0..r * embd)
            .map(|i| {
                let g0 = (i / 32) * 32;
                let a = x[g0..g0 + 32].iter().fold(0f32, |m, v| m.max(v.abs()));
                let s = a / 127.0;
                if s == 0.0 {
                    0
                } else {
                    (x[i] / s).round().clamp(-127.0, 127.0) as i8
                }
            })
            .collect();
        let s32: Vec<f32> = (0..r * embd / 32)
            .map(|b| {
                x[b * 32..b * 32 + 32]
                    .iter()
                    .fold(0f32, |m, v| m.max(v.abs()))
                    / 127.0
            })
            .collect();
        // per-128 quantize (host)
        let q128: Vec<i8> = (0..r * embd)
            .map(|i| {
                let g0 = (i / 128) * 128;
                let a = x[g0..g0 + 128].iter().fold(0f32, |m, v| m.max(v.abs()));
                let s = a / 127.0;
                if s == 0.0 {
                    0
                } else {
                    (x[i] / s).round().clamp(-127.0, 127.0) as i8
                }
            })
            .collect();
        let s128: Vec<f32> = (0..r * embd / 128)
            .map(|b| {
                x[b * 128..b * 128 + 128]
                    .iter()
                    .fold(0f32, |m, v| m.max(v.abs()))
                    / 127.0
            })
            .collect();
        let up_i8 = |h: &Vec<i8>| {
            let mut d = exec.alloc_i8(h.len()).expect("a");
            exec.stream.memcpy_htod(h, &mut d).expect("h");
            d
        };
        let up_f = |h: &Vec<f32>| {
            let mut d = exec.stream.alloc_zeros::<f32>(h.len()).expect("a");
            exec.stream.memcpy_htod(h, &mut d).expect("h");
            d
        };
        let (xq_a, xs_a) = (up_i8(&q32), up_f(&s32));
        let (xq_b, xs_b) = (up_i8(&q128), up_f(&s128));
        // routing + CSR
        let idx_h: Vec<u32> = (0..r * k).map(|i| ((i * 5) % n_e) as u32).collect();
        let mut idx = exec.alloc_u32(r * k).expect("i");
        exec.stream.memcpy_htod(&idx_h, &mut idx).expect("h");
        let mb32 = (r * k + n_e * 31) / 32;
        let srp = mb32 * 32;
        let mut srow = exec.alloc_u32(srp).expect("sr");
        let mut sslot = exec.alloc_u32(srp).expect("ss");
        let mut bexp = exec.alloc_u32(mb32).expect("be");
        exec.moe_align(&idx, &mut srow, &mut sslot, &mut bexp, r, k, n_e, mb32)
            .expect("align");
        let mut fq_a = exec.alloc_i8(srp * ff).expect("fa");
        let mut fs_a = exec.stream.alloc_zeros::<f32>(srp * ff / 32).expect("sa");
        let mut fq_b = exec.alloc_i8(srp * ff).expect("fb");
        let mut fs_b = exec.stream.alloc_zeros::<f32>(srp * ff / 32).expect("sb");
        exec.q8_0_moe_gate_up_mma2_geglu(
            &gate, &up, &srow, &bexp, &xq_a, &xs_a, &mut fq_a, &mut fs_a, mb32, 32,
        )
        .expect("v2");
        exec.q8_0_moe_gate_up_mma2g_geglu(
            &gate, &up, &srow, &bexp, &xq_b, &xs_b, &mut fq_b, &mut fs_b, mb32, 32,
        )
        .expect("xg");
        let sr_h: Vec<u32> = exec.stream.clone_dtoh(&srow).expect("d");
        let (qa, sa): (Vec<i8>, Vec<f32>) = (
            exec.stream.clone_dtoh(&fq_a).expect("d"),
            exec.stream.clone_dtoh(&fs_a).expect("d"),
        );
        let (qb, sb2): (Vec<i8>, Vec<f32>) = (
            exec.stream.clone_dtoh(&fq_b).expect("d"),
            exec.stream.clone_dtoh(&fs_b).expect("d"),
        );
        let mut nlive = 0usize;
        for &s in sr_h.iter() {
            if s == u32::MAX {
                continue;
            }
            nlive += 1;
        }
        // simpler: dequant compare
        let deq = |q: &Vec<i8>, s: &Vec<f32>, i: usize, j: usize| {
            q[i * ff + j] as f32 * s[i * (ff / 32) + j / 32]
        };
        let mut num = 0f64;
        let mut den = 0f64;
        for (i, &sr0) in sr_h.iter().enumerate() {
            if sr0 == u32::MAX {
                continue;
            }
            for j in 0..ff {
                let a = deq(&qa, &sa, i, j) as f64;
                let b = deq(&qb, &sb2, i, j) as f64;
                num += (a - b).abs();
                den += a.abs();
            }
        }
        let mr = (num / den.max(1e-30)) as f32;
        let t_a = time_us(&exec, 200, || {
            exec.q8_0_moe_gate_up_mma2_geglu(
                &gate, &up, &srow, &bexp, &xq_a, &xs_a, &mut fq_a, &mut fs_a, mb32, 32,
            )
            .expect("v2");
        });
        let t_b = time_us(&exec, 200, || {
            exec.q8_0_moe_gate_up_mma2g_geglu(
                &gate, &up, &srow, &bexp, &xq_b, &xs_b, &mut fq_b, &mut fs_b, mb32, 32,
            )
            .expect("xg");
        });
        println!(
            "r={r}: live {nlive}  fq rel-err(per32 vs per128 path) {mr:.5}  | gu per32 {t_a:7.1}us  xg {t_b:7.1}us  ({:+.1}%)",
            100.0 * (t_b / t_a - 1.0)
        );
    }
}
