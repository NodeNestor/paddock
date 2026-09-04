//! P1 dn64 probe: shipped xg lane (mma2g per-32 fs + down_mma2) vs the dn64
//! pair (mma2g_y64 per-64 fs + down_mma2_fs64) on identical synthetic
//! weights + per-128 activations. Reports fq dequant rel-err between arms
//! (census bound: per-64 adds 0.9-1.4%), part-vs-f64-reference maxrel for
//! both arms (consumer correctness), and gu/down timings.
// A development probe: it runs on a box its author is looking at, and a
// failure should stop it where it happened rather than be reported.
#![allow(clippy::unwrap_used)]

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

fn half_to_f64(h: u16) -> f64 {
    let s = if h >> 15 == 1 { -1.0f64 } else { 1.0 };
    let e = ((h >> 10) & 0x1f) as i32;
    let m = (h & 0x3ff) as f64;
    if e == 0 {
        s * m * (2f64).powi(-24)
    } else {
        s * (1.0 + m / 1024.0) * (2f64).powi(e - 15)
    }
}

fn main() {
    let pack = std::env::var("PADDOCK_PACK").expect("set PADDOCK_PACK");
    let exec = Arc::new(GpuExecutor::new(0, pack.as_ref()).expect("executor"));
    let (n_e, ff, embd, k) = (128usize, 704usize, 2816usize, 8usize);
    const H: [u16; 8] = [
        0x3c00, 0x3800, 0x3a00, 0x3e00, 0x3400, 0x4000, 0x3d00, 0x3900,
    ];
    let mk = |in_d: usize, out: usize, seed: usize| -> (RepackedQ8, Vec<u8>, Vec<u16>) {
        let blocks = n_e * out * in_d / 32;
        let host: Vec<u8> = (0..blocks * 32)
            .map(|i| ((i * 13 + 5 + seed) % 255) as u8)
            .collect();
        let mut d = exec.alloc_u8(blocks * 32).expect("data");
        exec.stream.memcpy_htod(&host, &mut d).expect("h");
        let hs: Vec<u16> = (0..blocks).map(|i| H[(i + seed) % 8]).collect();
        let bytes: Vec<u8> = hs.iter().flat_map(|h| h.to_le_bytes()).collect();
        let mut sc = exec.alloc_u8(blocks * 2).expect("scale");
        exec.stream.memcpy_htod(&bytes, &mut sc).expect("h");
        (
            RepackedQ8 {
                data: d,
                scale: sc,
                dims: vec![in_d, out, n_e],
            },
            host,
            hs,
        )
    };
    let (gate, _, _) = mk(embd, ff, 0);
    let (up, _, _) = mk(embd, ff, 7);
    let (down, dn_bytes, dn_scales) = mk(ff, embd, 3);

    for r in [64usize, 128] {
        // realistic-ish activations with mild outliers, per-128 quantized
        let x: Vec<f32> = (0..r * embd)
            .map(|i| {
                let base = (((i * 61 + 13) % 1009) as f32 / 1009.0 - 0.5) * 2.0;
                if i % 977 == 0 { base * 9.0 } else { base }
            })
            .collect();
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
        let (xq, xs) = (up_i8(&q128), up_f(&s128));
        // routing + CSR + topk weights
        let idx_h: Vec<u32> = (0..r * k).map(|i| ((i * 5) % n_e) as u32).collect();
        let mut idx = exec.alloc_u32(r * k).expect("i");
        exec.stream.memcpy_htod(&idx_h, &mut idx).expect("h");
        let tw_h: Vec<f32> = (0..r * k).map(|i| 0.5 + (i % 7) as f32 * 0.25).collect();
        let tw = up_f(&tw_h);
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
        let mut part_a = exec.stream.alloc_zeros::<f32>(r * k * embd).expect("pa");
        let mut part_b = exec.stream.alloc_zeros::<f32>(r * k * embd).expect("pb");
        exec.q8_0_moe_gate_up_mma2g_geglu(
            &gate, &up, &srow, &bexp, &xq, &xs, &mut fq_a, &mut fs_a, mb32, 32,
        )
        .expect("xg");
        exec.q8_0_moe_gate_up_mma2g_y64_geglu(
            &gate, &up, &srow, &bexp, &xq, &xs, &mut fq_b, &mut fs_b, mb32, 32,
        )
        .expect("y64");
        exec.q8_0_moe_down_mma2(
            &down,
            &srow,
            &sslot,
            &bexp,
            &tw,
            &fq_a,
            &fs_a,
            &mut part_a,
            k,
            mb32,
            32,
        )
        .expect("dn");
        exec.q8_0_moe_down_mma2_fs64(
            &down,
            &srow,
            &sslot,
            &bexp,
            &tw,
            &fq_b,
            &fs_b,
            &mut part_b,
            k,
            mb32,
            32,
            false,
        )
        .expect("dn64");

        let sr_h: Vec<u32> = exec.stream.clone_dtoh(&srow).expect("d");
        let sl_h: Vec<u32> = exec.stream.clone_dtoh(&sslot).expect("d");
        let be_h: Vec<u32> = exec.stream.clone_dtoh(&bexp).expect("d");
        let (qa, sa): (Vec<i8>, Vec<f32>) = (
            exec.stream.clone_dtoh(&fq_a).expect("d"),
            exec.stream.clone_dtoh(&fs_a).expect("d"),
        );
        let (qb, sb): (Vec<i8>, Vec<f32>) = (
            exec.stream.clone_dtoh(&fq_b).expect("d"),
            exec.stream.clone_dtoh(&fs_b).expect("d"),
        );
        let (pa, pb): (Vec<f32>, Vec<f32>) = (
            exec.stream.clone_dtoh(&part_a).expect("d"),
            exec.stream.clone_dtoh(&part_b).expect("d"),
        );

        // fq dequant rel-err between arms (per-64 quantize delta; census class)
        let n32 = ff / 32;
        let n64 = ff / 64;
        let mut num = 0f64;
        let mut den = 0f64;
        for (i, &sr0) in sr_h.iter().enumerate() {
            if sr0 == u32::MAX {
                continue;
            }
            for j in 0..ff {
                let a = qa[i * ff + j] as f64 * sa[i * n32 + j / 32] as f64;
                let b = qb[i * ff + j] as f64 * sb[i * n64 + j / 64] as f64;
                num += (a - b).abs();
                den += a.abs();
            }
        }
        let fq_rel = num / den.max(1e-30);

        // part vs f64 reference (each arm vs its own fq/fs) on sampled rows
        let deq_w = |e: usize, o: usize, j: usize| -> f64 {
            let row = e * embd + o;
            (dn_bytes[row * ff + j] as i8) as f64 * half_to_f64(dn_scales[row * n32 + j / 32])
        };
        let mut mr_a = 0f64;
        let mut mr_b = 0f64;
        let mut checked = 0usize;
        for i in (0..srp).step_by(srp / 4 + 1) {
            let sr0 = sr_h[i];
            if sr0 == u32::MAX {
                continue;
            }
            let e = be_h[i / 32] as usize;
            let w = tw_h[sr0 as usize * k + sl_h[i] as usize] as f64;
            for o in (0..embd).step_by(embd / 32 + 1) {
                let mut ra = 0f64;
                let mut rb = 0f64;
                for j in 0..ff {
                    ra += qa[i * ff + j] as f64 * sa[i * n32 + j / 32] as f64 * deq_w(e, o, j);
                    rb += qb[i * ff + j] as f64 * sb[i * n64 + j / 64] as f64 * deq_w(e, o, j);
                }
                let pidx = (sr0 as usize * k + sl_h[i] as usize) * embd + o;
                let da = (pa[pidx] as f64 - w * ra).abs() / (w * ra).abs().max(1e-6);
                let db = (pb[pidx] as f64 - w * rb).abs() / (w * rb).abs().max(1e-6);
                if da > mr_a {
                    mr_a = da;
                }
                if db > mr_b {
                    mr_b = db;
                }
                checked += 1;
            }
        }

        let t_gu_a = time_us(&exec, 200, || {
            exec.q8_0_moe_gate_up_mma2g_geglu(
                &gate, &up, &srow, &bexp, &xq, &xs, &mut fq_a, &mut fs_a, mb32, 32,
            )
            .expect("xg");
        });
        let t_gu_b = time_us(&exec, 200, || {
            exec.q8_0_moe_gate_up_mma2g_y64_geglu(
                &gate, &up, &srow, &bexp, &xq, &xs, &mut fq_b, &mut fs_b, mb32, 32,
            )
            .expect("y64");
        });
        let t_dn_a = time_us(&exec, 200, || {
            exec.q8_0_moe_down_mma2(
                &down,
                &srow,
                &sslot,
                &bexp,
                &tw,
                &fq_a,
                &fs_a,
                &mut part_a,
                k,
                mb32,
                32,
            )
            .expect("dn");
        });
        let t_dn_b = time_us(&exec, 200, || {
            exec.q8_0_moe_down_mma2_fs64(
                &down,
                &srow,
                &sslot,
                &bexp,
                &tw,
                &fq_b,
                &fs_b,
                &mut part_b,
                k,
                mb32,
                32,
                false,
            )
            .expect("dn64");
        });
        println!(
            "r={r}: fq rel(per32-arm vs per64-arm) {fq_rel:.5} | part-vs-ref maxrel a {mr_a:.2e} b {mr_b:.2e} ({checked} pts) | gu {t_gu_a:6.1} -> {t_gu_b:6.1}us ({:+.1}%) | dn {t_dn_a:6.1} -> {t_dn_b:6.1}us ({:+.1}%)",
            100.0 * (t_gu_b / t_gu_a - 1.0),
            100.0 * (t_dn_b / t_dn_a - 1.0)
        );
    }
}
