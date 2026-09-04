//! A4B global-layer (hd512 V-less) prefill-attention lab: times the hd512
//! arm at the pf8 serving shape (2048-row chunks, e4m3 KV - the KV8 serving
//! default) across chunk positions, with a sampled-row CPU parity gate.
//!
//! Geometry: 16 q-heads / 2 kv-heads (G8), head_dim 512, full causal
//! (swa_window 0), score scale 1.0 - exactly the A4B's 5 global layers.
//!
//! Usage: PADDOCK_PACK=... a4b_attn_kbench [--parity]
// A development probe: it runs on a box its author is looking at, and a
// failure should stop it where it happened rather than be reported.
#![allow(clippy::unwrap_used)]

use std::sync::Arc;

use paddock_engine::gpu::{GpuExecutor, KvDtype};

/// f32 -> e4m3 byte (round-to-nearest-even, saturate to +-448, no NaN care -
/// the lab never produces NaN inputs). Mirrors the device encode class.
fn e4m3_encode(v: f32) -> u8 {
    if v == 0.0 {
        return 0;
    }
    let s = if v < 0.0 { 0x80u8 } else { 0 };
    let a = v.abs().min(448.0);
    // subnormal boundary: 2^-6; mant step 2^-9
    if a < 2f32.powi(-6) {
        let m = (a / 2f32.powi(-9)).round() as u8;
        return s | m.min(7);
    }
    let e = a.log2().floor() as i32;
    let mut exp = e.clamp(-6, 8);
    let mut mant = ((a / 2f32.powi(exp) - 1.0) * 8.0).round() as i32;
    if mant == 8 {
        exp += 1;
        mant = 0;
    }
    if exp > 8 || (exp == 8 && mant > 6) {
        // saturate at 448 = 2^8 * 1.75
        return s | 0x7E;
    }
    s | (((exp + 7) as u8) << 3) | mant as u8
}

fn e4m3_decode(b: u8) -> f32 {
    let s = if b & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let exp = ((b >> 3) & 0xF) as i32;
    let mant = (b & 7) as f32;
    if exp == 0 {
        return s * mant * 2f32.powi(-9);
    }
    s * (1.0 + mant / 8.0) * 2f32.powi(exp - 7)
}

fn fill(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            ((s >> 8) as f32 / (1u32 << 24) as f32) - 0.5
        })
        .collect()
}

fn main() {
    let parity = std::env::args().any(|a| a == "--parity");
    // --swa: the A4B's 25 sliding-window layers (16q/8kv G2, hd256, win 1024)
    // instead of the 5 hd512 globals
    let swa = std::env::args().any(|a| a == "--swa");
    let pack = std::env::var_os("PADDOCK_PACK")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../packs/cuda/build/pd-cuda-sm120.so")
        });
    let exec = Arc::new(GpuExecutor::new(0, &pack).expect("executor"));
    let (h, kvh, hd) = if swa {
        (16usize, 8usize, 256usize)
    } else {
        (16, 2, 512)
    };
    let window = if swa { 1024usize } else { 0 };
    let kv_dim = kvh * hd;
    let rows = 2048usize;
    let max_ctx = 8192usize;
    let scale = 1.0f32; // gemma4 attention scale

    // e4m3 KV pool (the KV8 serving default), identity block table (16-row pages)
    let bps = max_ctx / 16;
    let kf: Vec<f32> = fill(max_ctx * kv_dim, 7);
    let vf: Vec<f32> = fill(max_ctx * kv_dim, 11);
    let kb: Vec<u8> = kf.iter().map(|&v| e4m3_encode(v)).collect();
    let vb: Vec<u8> = vf.iter().map(|&v| e4m3_encode(v)).collect();
    let d_k = exec.to_device_u8(&kb).expect("k");
    let d_v = exec.to_device_u8(&vb).expect("v");
    let bt: Vec<u32> = (0..bps as u32).collect();
    let d_bt = exec.to_device_u32(&bt).expect("bt");
    let sinks = vec![-1e30f32; h];
    let d_sinks = exec.to_device(&sinks).expect("sinks");
    let qf = fill(rows * h * hd, 3);
    let d_q = exec.to_device(&qf).expect("q");
    let mut d_out = exec.alloc(rows * h * hd).expect("out");
    let d_slots = exec.to_device_u32(&vec![0u32; rows]).expect("slots");

    if parity {
        // done=2048 exercises the causal edge + a full history band;
        // PF_DONE=0 gives single-tile early rows (no rescale path) for debug
        let done: usize = std::env::var("PF_DONE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(2048);
        let pos: Vec<u32> = (done as u32..(done + rows) as u32).collect();
        let d_pos = exec.to_device_u32(&pos).expect("pos");
        exec.attn_prefill_f16_paged(
            &d_q,
            &d_k,
            &d_v,
            &d_sinks,
            &mut d_out,
            &d_pos,
            &d_slots,
            &d_bt,
            bps,
            h,
            kvh,
            hd,
            kv_dim,
            window,
            rows,
            scale,
            KvDtype::Fp8E4m3,
        )
        .expect("attn");
        let gpu = exec.to_host(&d_out).expect("dtoh");
        // sampled rows: chunk head, interior, causal tail
        let sample: Vec<usize> = vec![0, 1, 7, 63, 500, 1024, 1500, 2040, 2046, 2047];
        let (mut worst, mut worst_at) = (0f64, (0usize, 0usize, 0usize));
        for &r in &sample {
            let mut row_worst = 0f64;
            let p = done + r; // absolute position
            // valid keys: the sliding window (or full causal when window=0)
            let lo = if window > 0 && p + 1 > window {
                p + 1 - window
            } else {
                0
            };
            for head in 0..h {
                let kvg = head / (h / kvh);
                // f16-rounded Q, decoded-e4m3 K/V - the kernel's operand class
                let q: Vec<f32> = (0..hd)
                    .map(|d| f32::from(half::f16::from_f32(qf[(r * h + head) * hd + d])))
                    .collect();
                let mut scores = vec![0f64; p + 1 - lo];
                let mut m = f64::NEG_INFINITY;
                for key in lo..=p {
                    let mut dot = 0f64;
                    for d in 0..hd {
                        dot += q[d] as f64 * e4m3_decode(kb[key * kv_dim + kvg * hd + d]) as f64;
                    }
                    let sc = dot * scale as f64;
                    scores[key - lo] = sc;
                    m = m.max(sc);
                }
                // the f8 arms quantize P-tilde to e4m3 at store while l stays
                // the exact f32 sum (the TRT/SGLang normalizer split); pf6g
                // is the default hd512 route (NO_PF6G restores c2's f16 P);
                // the SWA twin models e4m3 P only once pf6s is armed
                let p_e4m3 = if window > 0 {
                    std::env::var_os("PADDOCK_PF6S").is_some()
                } else {
                    std::env::var_os("PADDOCK_NO_PF6G").is_none()
                };
                let mut l = 0f64;
                for s in scores.iter_mut() {
                    *s = (*s - m).exp();
                    l += *s;
                    if p_e4m3 {
                        *s = e4m3_decode(e4m3_encode(*s as f32)) as f64;
                    }
                }
                for d in (0..hd).step_by(37) {
                    let mut o = 0f64;
                    for key in lo..=p {
                        o += scores[key - lo] * e4m3_decode(vb[key * kv_dim + kvg * hd + d]) as f64;
                    }
                    o /= l;
                    let g = gpu[(r * h + head) * hd + d] as f64;
                    let diff = (g - o).abs();
                    row_worst = row_worst.max(diff);
                    if diff > worst {
                        worst = diff;
                        worst_at = (r, head, d);
                    }
                }
            }
            println!("  row {r:4}: worst {row_worst:.3e}");
        }
        println!(
            "parity done={done}: worst |diff| {worst:.3e} at (row {}, head {}, d {})",
            worst_at.0, worst_at.1, worst_at.2
        );
        let gate = 5e-2; // f16 QK + e4m3 KV + f32 online softmax class
        println!(
            "gate({gate:.0e}): {}",
            if worst < gate { "PASS" } else { "FAIL" }
        );
        return;
    }

    for &done in &[0usize, 2048, 4096, 6144] {
        let pos: Vec<u32> = (done as u32..(done + rows) as u32).collect();
        let d_pos = exec.to_device_u32(&pos).expect("pos");
        for _ in 0..3 {
            exec.attn_prefill_f16_paged(
                &d_q,
                &d_k,
                &d_v,
                &d_sinks,
                &mut d_out,
                &d_pos,
                &d_slots,
                &d_bt,
                bps,
                h,
                kvh,
                hd,
                kv_dim,
                window,
                rows,
                scale,
                KvDtype::Fp8E4m3,
            )
            .expect("attn");
        }
        exec.synchronize().expect("sync");
        let t0 = std::time::Instant::now();
        let iters = 30;
        for _ in 0..iters {
            exec.attn_prefill_f16_paged(
                &d_q,
                &d_k,
                &d_v,
                &d_sinks,
                &mut d_out,
                &d_pos,
                &d_slots,
                &d_bt,
                bps,
                h,
                kvh,
                hd,
                kv_dim,
                window,
                rows,
                scale,
                KvDtype::Fp8E4m3,
            )
            .expect("attn");
        }
        exec.synchronize().expect("sync");
        let us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
        // causal (window-clipped) key count; x4 = QK + PV MACs
        let keys: f64 = (0..rows)
            .map(|i| {
                let n = done + i + 1;
                if window > 0 && n > window {
                    window as f64
                } else {
                    n as f64
                }
            })
            .sum();
        let flop = 4.0 * keys * (h * hd) as f64;
        println!("done={done:4}: {us:8.1} us  {:.0} TF", flop / us / 1e6);
    }
}
