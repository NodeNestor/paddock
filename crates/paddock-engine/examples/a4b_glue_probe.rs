//! MoE glue probe: times pd_moe_tail_combine at the decode
//! shape (batch x k x embd part fold + dual-rms tail) and
//! pd_quantize_q8_geglu_remap at the prefill-wave shape. Synthetic data;
//! perf only (numerics are bitwise-by-construction edits upstream).
//! Usage: PADDOCK_PACK=... a4b_glue_probe [batch]
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
    let (embd, ff, k) = (2816usize, 704usize, 8usize);
    let batches: Vec<usize> = std::env::args()
        .nth(1)
        .map(|v| vec![v.parse().unwrap()])
        .unwrap_or_else(|| vec![32, 64, 128]);

    for &b in &batches {
        let fill = |n: usize, s: f64| -> Vec<f32> {
            (0..n)
                .map(|i| ((i as f64 * s).sin() * 0.1) as f32)
                .collect()
        };
        let mut x = exec.stream.alloc_zeros::<f32>(b * embd).expect("x");
        exec.stream
            .memcpy_htod(&fill(b * embd, 0.13), &mut x)
            .expect("h");
        let mut proj = exec.stream.alloc_zeros::<f32>(b * embd).expect("p");
        exec.stream
            .memcpy_htod(&fill(b * embd, 0.29), &mut proj)
            .expect("h");
        let mut part = exec.stream.alloc_zeros::<f32>(b * k * embd).expect("pt");
        exec.stream
            .memcpy_htod(&fill(b * k * embd, 0.07), &mut part)
            .expect("h");
        let mut pn1 = exec.stream.alloc_zeros::<f32>(embd).expect("n1");
        exec.stream
            .memcpy_htod(&fill(embd, 0.41), &mut pn1)
            .expect("h");
        let mut pn2 = exec.stream.alloc_zeros::<f32>(embd).expect("n2");
        exec.stream
            .memcpy_htod(&fill(embd, 0.43), &mut pn2)
            .expect("h");
        let mut pw = exec.stream.alloc_zeros::<f32>(embd).expect("pw");
        exec.stream
            .memcpy_htod(&fill(embd, 0.47), &mut pw)
            .expect("h");
        let t = time_us(&exec, 500, || {
            exec.moe_tail_combine(&mut x, &proj, &part, &pn1, &pn2, &pw, embd, k, 1e-6, 1.0, b)
                .expect("combine");
        });
        let mb = (b * k * embd * 4) as f64 / 1e6;
        println!(
            "tail_combine b={b:>3}: {t:>7.1}us  (part {mb:.1} MB, {:.0} GB/s part-once)",
            mb / t * 1000.0
        );
    }

    // remap at the prefill-wave shape: r tokens, k=8, bm128 sort with pad
    let r = 2400usize;
    let pairs = r * k;
    let live = pairs; // all live, identity map
    let srp128 = 31616usize; // the serve grid (pad rows exit immediately)
    let mut srow = exec.alloc_u32(srp128).expect("srow");
    let mut sslot = exec.alloc_u32(srp128).expect("sslot");
    {
        let mut h = vec![0xffff_ffffu32; srp128];
        let mut hs = vec![0u32; srp128];
        for i in 0..live {
            h[i] = (i / k) as u32;
            hs[i] = (i % k) as u32;
        }
        exec.stream.memcpy_htod(&h, &mut srow).expect("h");
        exec.stream.memcpy_htod(&hs, &mut sslot).expect("h");
    }
    let mut map = exec.stream.alloc_zeros::<f32>(r * k).expect("map");
    {
        let h: Vec<f32> = (0..r * k).map(|i| f32::from_bits(i as u32)).collect();
        exec.stream.memcpy_htod(&h, &mut map).expect("h");
    }
    let mut gu = exec.stream.alloc_zeros::<f32>(srp128 * 2 * ff).expect("gu");
    {
        let h: Vec<f32> = (0..srp128 * 2 * ff)
            .map(|i| ((i % 251) as f32) * 0.01 - 1.2)
            .collect();
        exec.stream.memcpy_htod(&h, &mut gu).expect("h");
    }
    let mut fq = exec.alloc_i8(pairs * ff).expect("fq");
    let mut fs = exec.stream.alloc_zeros::<f32>(pairs * ff / 32).expect("fs");
    let t = time_us(&exec, 200, || {
        exec.quantize_q8_geglu_remap(&gu, &srow, &sslot, &map, &mut fq, &mut fs, ff, k, srp128, 0)
            .expect("remap");
    });
    let mb = (live * 2 * ff * 4) as f64 / 1e6;
    println!(
        "geglu_remap srp128={srp128} live={live}: {t:>7.1}us  (gu {mb:.0} MB, {:.0} GB/s)",
        mb / t * 1000.0
    );
    // bitwise receipt: checksum fq (i64 sum) and fs (bits xor) for cross-pack compare
    {
        let hq: Vec<i8> = exec.stream.clone_dtoh(&fq).expect("dtoh fq");
        let hs: Vec<f32> = exec.stream.clone_dtoh(&fs).expect("dtoh fs");
        let qsum: i64 = hq.iter().map(|&v| v as i64).sum();
        let qabs: i64 = hq.iter().map(|&v| (v as i64).abs()).sum();
        let sxor = hs.iter().fold(0u32, |acc, v| acc ^ v.to_bits());
        println!("remap checksum: qsum={qsum} qabs={qabs} fs_xor={sxor:08x}");
    }
}
