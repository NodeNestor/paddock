//! Bit-parity for granite's residual fusion (`add_rmsnorm_q8_xn`).
//!
//! The fused kernel replaces three launches - `scale_add` + `rmsnorm_batch` +
//! `quantize_q8_sums` - so the only acceptable result is that all five outputs
//! (`x`, `xn`, `q`, `scale`, `sums`) are BIT-IDENTICAL to running those three.
//! Not close: identical. A fused norm that is a few ulp off its unfused
//! sequence is exactly the kind of drift that costs days to track down later.
//!
//! Bit-exactness is only free here because the sumsq accumulator is a
//! double-float and therefore width-invariant: the fused
//! kernel is one row-per-CTA norm, where `rmsnorm_batch` runs its own width
//! election, and under the previous f64 accumulator the two widths would have
//! had to be forced to match. So this test is also the standing check that the
//! DF property still holds - if someone elects F32 or re-widens a reduction,
//! this goes red before a board does.
//!
//! Shapes cover granite-30b's real decode widths (n_embd 4096, n_ff 32768) plus
//! a batch>1 row and the no-residual (entry norm) form. Light - no model load.
// Test code: a failed assumption stops the test where it happened.
#![allow(clippy::unwrap_used)]

mod common;

use paddock_engine::gpu::GpuExecutor;

fn det(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / (1u64 << 31) as f32) - 0.5
        })
        .collect()
}

fn exec() -> Option<GpuExecutor> {
    common::gpu()
}

/// One (n, batch, with_residual) case: run both paths on identical inputs and
/// demand bitwise agreement on every output.
fn case(e: &GpuExecutor, n: usize, batch: usize, residual: bool, seed: u64) {
    let eps = 1e-5f32;
    // granite's residual_multiplier is not 1.0 - using 1.0 would let an
    // "ignored the scale entirely" bug pass, which is the whole reason this
    // kernel exists.
    let res_scale = 0.223_606_8_f32;

    let x0 = det(n * batch, seed);
    let proj = det(n * batch, seed ^ 0x9e37);
    let w = det(n, seed ^ 0x51ed);

    // ---- unfused: the three launches the fusion replaces -----------------
    let mut ux = e.to_device(&x0).expect("ux");
    let up = e.to_device(&proj).expect("up");
    let uw = e.to_device(&w).expect("uw");
    let mut uxn = e.alloc(n * batch).expect("uxn");
    if residual {
        e.scale_add(&mut ux, &up, res_scale, n * batch)
            .expect("scale_add");
    }
    e.rmsnorm_batch(&ux, &uw, &mut uxn, n, eps, batch)
        .expect("rmsnorm");
    let mut uq = e.alloc_i8(n * batch).expect("uq");
    let mut us = e.alloc((n >> 5) * batch).expect("us");
    let mut um = e.alloc((n >> 4) * batch).expect("um");
    // quantize_q8_sums takes a flat element count; the rows are contiguous so
    // batch rows quantize as one run of batch*n elements.
    e.quantize_q8_sums(&uxn, &mut uq, &mut us, &mut um, n * batch)
        .expect("quantize");

    // ---- fused ------------------------------------------------------------
    let mut fx = e.to_device(&x0).expect("fx");
    let fp = e.to_device(&proj).expect("fp");
    let fw = e.to_device(&w).expect("fw");
    let mut fxn = e.alloc(n * batch).expect("fxn");
    let mut fq = e.alloc_i8(n * batch).expect("fq");
    let mut fs = e.alloc((n >> 5) * batch).expect("fs");
    let mut fm = e.alloc((n >> 4) * batch).expect("fm");
    e.add_rmsnorm_q8_xn(
        &mut fx,
        if residual { Some(&fp) } else { None },
        &fw,
        &mut fxn,
        &mut fq,
        &mut fs,
        &mut fm,
        n,
        batch,
        eps,
        res_scale,
    )
    .expect("fused");

    let tag = format!("n={n} batch={batch} residual={residual}");
    let (hux, hfx) = (e.to_host(&ux).unwrap(), e.to_host(&fx).unwrap());
    let (huxn, hfxn) = (e.to_host(&uxn).unwrap(), e.to_host(&fxn).unwrap());
    let (huq, hfq) = (e.to_host_i8(&uq).unwrap(), e.to_host_i8(&fq).unwrap());
    let (hus, hfs) = (e.to_host(&us).unwrap(), e.to_host(&fs).unwrap());
    let (hum, hfm) = (e.to_host(&um).unwrap(), e.to_host(&fm).unwrap());

    // Compare BITS, not values: f32 == would call two different NaNs equal and
    // -0.0 == 0.0, and a quantizer that emits -0.0 where the reference emits
    // 0.0 has changed the bytes a GEMV reads.
    let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
    let first_diff = |a: &[u32], b: &[u32]| a.iter().zip(b).position(|(x, y)| x != y);

    if let Some(i) = first_diff(&bits(&hux), &bits(&hfx)) {
        panic!("{tag}: x differs at {i}: {} vs {}", hux[i], hfx[i]);
    }
    if let Some(i) = first_diff(&bits(&huxn), &bits(&hfxn)) {
        panic!("{tag}: xn differs at {i}: {} vs {}", huxn[i], hfxn[i]);
    }
    if let Some(i) = huq.iter().zip(&hfq).position(|(a, b)| a != b) {
        panic!("{tag}: q differs at {i}: {} vs {}", huq[i], hfq[i]);
    }
    if let Some(i) = first_diff(&bits(&hus), &bits(&hfs)) {
        panic!(
            "{tag}: scale differs at block {i}: {} vs {}",
            hus[i], hfs[i]
        );
    }
    if let Some(i) = first_diff(&bits(&hum), &bits(&hfm)) {
        panic!("{tag}: sums differ at half {i}: {} vs {}", hum[i], hfm[i]);
    }
}

#[test]
fn add_rmsnorm_q8_xn_bitmatches_the_three_it_replaces() {
    let Some(e) = exec() else {
        common::missing("gpu");
        return;
    };
    if !e.has_add_rmsnorm_q8_xn() {
        common::missing("add_rmsnorm_q8_xn");
        return;
    }
    // granite-30b decode shapes: n_embd for the attn/ffn norms, n_ff for the
    // swiglu landing, plus a small n to exercise the n4 < nth corner where
    // whole lanes never enter the epilogue loop (the group-mask case).
    for (n, batch) in [(4096, 1), (32768, 1), (4096, 3), (128, 1), (32, 1)] {
        case(&e, n, batch, true, 0xC0FFEE ^ n as u64);
        case(&e, n, batch, false, 0xBEEF ^ n as u64);
    }
}
