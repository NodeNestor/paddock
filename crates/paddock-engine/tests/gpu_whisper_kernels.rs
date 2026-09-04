//! Whisper decode-lane kernel gates. Every kernel in
//! `packs/cuda/src/asr/whisper.cuh` replaces a SEQUENCE of ops the bring-up
//! lane already ran, so each one is gated against that sequence rather than
//! against a hand-rolled expectation: the fused epilogues must be bit-equal
//! (same arithmetic, one launch), and the flash-decoding attention must match
//! the prefill-shaped `vision_attn_x` it replaces to f16-KV tolerance (it is
//! the same math with the K/V rounded to the cache's dtype, which is the
//! numeric class change the lane is making deliberately).
//!
//! Needs a CUDA device and `PADDOCK_PACK`; skips cleanly without them, the
//! same contract as the other gpu_* tests.
// Test code: a failed assumption stops the test where it happened.
#![allow(clippy::unwrap_used)]

mod common;

use half::f16;
use paddock_engine::gpu::{GpuExecutor, KvDtype};

/// Deterministic pseudo-random fill - no rand dep, and a fixed sequence keeps
/// a failure reproducible.
fn fill(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            ((s >> 8) as f32 / (1u32 << 24) as f32) - 0.5
        })
        .collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

/// Distance between two f16 values in representable steps. f16 is
/// sign-magnitude, so the raw bit difference is meaningless across zero; mapping
/// to a monotone key makes "1" mean genuinely adjacent.
fn f16_ulps(a: f16, b: f16) -> i32 {
    let key = |x: f16| -> i32 {
        let bits = x.to_bits() as i32;
        if bits & 0x8000 != 0 {
            -(bits & 0x7fff)
        } else {
            bits
        }
    };
    (key(a) - key(b)).abs()
}

/// f16 bit-equality with a failure a human can read. `assert_eq!` on a 1.9M-wide
/// vector prints 1.9M halves and says nothing; what tells you whether a drift is
/// a rounding-order slip or a broken kernel is how many moved, by how many ulps,
/// and where the first one is. (A real kernel bug here was diagnosed by
/// instrumenting exactly this by hand - so it lives in the gate now.)
fn assert_f16_bit_equal(want: &[f16], got: &[f16], what: &str) {
    assert_eq!(want.len(), got.len(), "{what}: length mismatch");
    let (mut n_diff, mut worst, mut first) = (0usize, 0.0f32, None);
    let mut hist: std::collections::BTreeMap<i32, usize> = std::collections::BTreeMap::new();
    for (i, (a, b)) in want.iter().zip(got).enumerate() {
        if a.to_bits() == b.to_bits() {
            continue;
        }
        n_diff += 1;
        *hist.entry(f16_ulps(*a, *b)).or_default() += 1;
        worst = worst.max((a.to_f32() - b.to_f32()).abs());
        first.get_or_insert((i, *a, *b));
    }
    if n_diff == 0 {
        return;
    }
    let (i, a, b) = first.unwrap();
    panic!(
        "{what}: {n_diff} of {} f16 values differ ({:.4}%); ulp histogram {hist:?}; \
         worst |diff| {worst:e}; first at index {i}: want {} got {}",
        want.len(),
        100.0 * n_diff as f64 / want.len() as f64,
        a.to_f32(),
        b.to_f32(),
    );
}

fn exec() -> Option<GpuExecutor> {
    common::gpu()
}

#[test]
fn ln_f16_matches_layernorm_then_convert() {
    let Some(e) = exec() else { return };
    // 1500 rows is the ENCODER's own width - the shape that actually runs,
    // and the one the register-staged body has to be exact at.
    let (rows, n) = (1500usize, 1280usize);
    let x = e.to_device(&fill(rows * n, 7)).unwrap();
    let w = e.to_device(&fill(n, 11)).unwrap();
    let b = e.to_device(&fill(n, 13)).unwrap();

    let mut want32 = e.alloc(rows * n).unwrap();
    let mut want = e.alloc_f16(rows * n).unwrap();
    e.layernorm(&x, &w, &b, &mut want32, rows, n, 1e-5).unwrap();
    e.convert_f32_f16(&want32, &mut want, rows * n).unwrap();

    let mut got = e.alloc_f16(rows * n).unwrap();
    e.whisper_ln_f16(&x, &w, &b, &mut got, rows, n, 1e-5)
        .unwrap();

    assert_f16_bit_equal(
        &e.to_host_f16_len(&want, rows * n).unwrap(),
        &e.to_host_f16_len(&got, rows * n).unwrap(),
        "fused LN->f16 must be bit-equal to layernorm + convert",
    );
}

#[test]
fn res_ln_f16_matches_bias_add_then_add_then_layernorm() {
    let Some(e) = exec() else { return };
    // the encoder residual seam runs 1500 rows wide; gate the real shape
    let (rows, n) = (1500usize, 1280usize);
    let xh = fill(rows * n, 21);
    let projh = fill(rows * n, 23);
    let biash = fill(n, 27);
    let wh = fill(n, 29);
    let bh = fill(n, 31);

    // unfused: tmp = proj + bias; x += tmp; out = f16(LN(x))
    let mut x0 = e.to_device(&xh).unwrap();
    let mut tmp = e.to_device(&projh).unwrap();
    let bias = e.to_device(&biash).unwrap();
    let w = e.to_device(&wh).unwrap();
    let b = e.to_device(&bh).unwrap();
    e.bias_add(&mut tmp, &bias, rows, n).unwrap();
    e.add(&mut x0, &tmp, rows * n).unwrap();
    let mut want32 = e.alloc(rows * n).unwrap();
    let mut want = e.alloc_f16(rows * n).unwrap();
    e.layernorm(&x0, &w, &b, &mut want32, rows, n, 1e-5)
        .unwrap();
    e.convert_f32_f16(&want32, &mut want, rows * n).unwrap();

    let mut x1 = e.to_device(&xh).unwrap();
    let proj = e.to_device(&projh).unwrap();
    let mut got = e.alloc_f16(rows * n).unwrap();
    e.whisper_res_ln_f16(&mut x1, &proj, &bias, &w, &b, &mut got, rows, n, 1e-5)
        .unwrap();

    assert_f16_bit_equal(
        &e.to_host_f16_len(&want, rows * n).unwrap(),
        &e.to_host_f16_len(&got, rows * n).unwrap(),
        "fused residual+norm must be bit-equal to bias_add + add + layernorm + convert",
    );
    assert_eq!(
        e.to_host_len(&x0, rows * n).unwrap(),
        e.to_host_len(&x1, rows * n).unwrap(),
        "the residual stream itself must land identically"
    );
}

#[test]
fn bias_gelu_f16_matches_bias_add_then_gelu_erf() {
    let Some(e) = exec() else { return };
    let (rows, n) = (3usize, 5120usize);
    let xh = fill(rows * n, 41);
    let biash = fill(n, 43);
    let bias = e.to_device(&biash).unwrap();

    let mut x0 = e.to_device(&xh).unwrap();
    e.bias_add(&mut x0, &bias, rows, n).unwrap();
    e.gelu_erf(&mut x0, rows * n).unwrap();
    let mut want = e.alloc_f16(rows * n).unwrap();
    e.convert_f32_f16(&x0, &mut want, rows * n).unwrap();

    let x1 = e.to_device(&xh).unwrap();
    let mut got = e.alloc_f16(rows * n).unwrap();
    e.whisper_bias_gelu_f16(&x1, &bias, &mut got, rows, n)
        .unwrap();

    assert_eq!(
        e.to_host_f16_len(&want, rows * n).unwrap(),
        e.to_host_f16_len(&got, rows * n).unwrap(),
        "fused bias+erf-GELU+cast must be bit-equal to the three-op chain"
    );
}

/// Fill a whole `[cap, rows, d]` KV plane at `kv` from host f32, through the
/// production writer - so the gate exercises the same store path serving
/// uses, and the plane's byte layout is whatever `kv` says it is.
fn kv_plane(
    e: &GpuExecutor,
    host: &[f32],
    cap: usize,
    rows: usize,
    d: usize,
    kv: KvDtype,
) -> cudarc::driver::CudaSlice<u8> {
    let src = e.to_device(host).unwrap();
    let slots: Vec<u32> = (0..cap as u32).collect();
    let dslots = e.to_device_u32(&slots).unwrap();
    let mut plane = e.alloc_u8(cap * rows * d * kv.bytes()).unwrap();
    e.whisper_kv_store(&src, None, &mut plane, &dslots, rows, d, rows, cap, kv)
        .unwrap();
    plane
}

#[test]
fn dec_attn_matches_the_prefill_shaped_kernel() {
    let Some(e) = exec() else { return };
    // whisper-large-v3 decode geometry, two slots, cross-attention lengths
    let (heads, hd) = (20usize, 64usize);
    let d = heads * hd;
    let (cap, batch, nkv) = (3usize, 2usize, 1500usize);
    let scale = 1.0 / (hd as f32).sqrt();

    let qh = fill(batch * d, 51);
    let q = e.to_device(&qh).unwrap();
    // K/V live at f16 in the cache; the reference reads the same rounded
    // values back at f32, so the only difference under test is the kernel
    let mut planes = Vec::new();
    for seed in [61u32, 67] {
        let h = fill(cap * nkv * d, seed);
        let plane = kv_plane(&e, &h, cap, nkv, d, KvDtype::Fp16);
        let rounded: Vec<f32> = e
            .to_host_f16_from_u8(&plane, cap * nkv * d)
            .unwrap()
            .iter()
            .map(|v| v.to_f32())
            .collect();
        planes.push((plane, rounded));
    }
    // slot map: active row 0 -> plane row 2, active row 1 -> plane row 0
    let slot_ids = [2u32, 0];
    let slots = e.to_device_u32(&slot_ids).unwrap();

    let mut got = e.alloc_f16(batch * d).unwrap();
    let mut part = e.alloc(cap * heads * 32 * (hd + 2)).unwrap();
    e.whisper_dec_attn(
        &q,
        None,
        &planes[0].0,
        &planes[1].0,
        &slots,
        None,
        &mut got,
        Some(&mut part),
        nkv,
        nkv,
        0,
        heads,
        hd,
        batch,
        scale,
        KvDtype::Fp16,
    )
    .unwrap();
    let got: Vec<f32> = e
        .to_host_f16_len(&got, batch * d)
        .unwrap()
        .iter()
        .map(|v| v.to_f32())
        .collect();

    // reference: the prefill-shaped kernel, one launch per active row over
    // that row's own plane slice
    for (b, &sid) in slot_ids.iter().enumerate() {
        let base = sid as usize * nkv * d;
        let k = e.to_device(&planes[0].1[base..base + nkv * d]).unwrap();
        let v = e.to_device(&planes[1].1[base..base + nkv * d]).unwrap();
        let qrow = e.to_device(&qh[b * d..(b + 1) * d]).unwrap();
        let mut want = e.alloc(d).unwrap();
        e.vision_attn_x(&qrow, &k, &v, &mut want, 1, nkv, heads, hd, 1, scale)
            .unwrap();
        let want = e.to_host_len(&want, d).unwrap();
        let diff = max_abs_diff(&want, &got[b * d..(b + 1) * d]);
        assert!(
            diff < 4e-3,
            "slot {b}: dec_attn vs vision_attn_x max |diff| {diff}"
        );
    }
}

#[test]
fn dec_attn_honours_per_slot_lengths() {
    let Some(e) = exec() else { return };
    // the self-attention shape: short, ragged, per-slot key counts
    let (heads, hd) = (20usize, 64usize);
    let d = heads * hd;
    let (cap, batch, ctx) = (4usize, 3usize, 448usize);
    let scale = 1.0 / (hd as f32).sqrt();
    let lens = [1u32, 37, 200]; // pos values; live keys are pos+1
    let slot_ids = [3u32, 1, 0];

    let qh = fill(batch * d, 71);
    let q = e.to_device(&qh).unwrap();
    let mut planes = Vec::new();
    for seed in [73u32, 79] {
        let h = fill(cap * ctx * d, seed);
        let plane = kv_plane(&e, &h, cap, ctx, d, KvDtype::Fp16);
        let rounded: Vec<f32> = e
            .to_host_f16_from_u8(&plane, cap * ctx * d)
            .unwrap()
            .iter()
            .map(|v| v.to_f32())
            .collect();
        planes.push((plane, rounded));
    }
    let slots = e.to_device_u32(&slot_ids).unwrap();
    let dlens = e.to_device_u32(&lens).unwrap();

    let mut got = e.alloc_f16(batch * d).unwrap();
    let mut part = e.alloc(cap * heads * 32 * (hd + 2)).unwrap();
    e.whisper_dec_attn(
        &q,
        None,
        &planes[0].0,
        &planes[1].0,
        &slots,
        Some(&dlens),
        &mut got,
        Some(&mut part),
        ctx,
        0,
        1,
        heads,
        hd,
        batch,
        scale,
        KvDtype::Fp16,
    )
    .unwrap();
    let got: Vec<f32> = e
        .to_host_f16_len(&got, batch * d)
        .unwrap()
        .iter()
        .map(|v| v.to_f32())
        .collect();

    for (b, (&sid, &len)) in slot_ids.iter().zip(&lens).enumerate() {
        let live = len as usize + 1;
        let base = sid as usize * ctx * d;
        let k = e.to_device(&planes[0].1[base..base + live * d]).unwrap();
        let v = e.to_device(&planes[1].1[base..base + live * d]).unwrap();
        let qrow = e.to_device(&qh[b * d..(b + 1) * d]).unwrap();
        let mut want = e.alloc(d).unwrap();
        e.vision_attn_x(&qrow, &k, &v, &mut want, 1, live, heads, hd, 1, scale)
            .unwrap();
        let want = e.to_host_len(&want, d).unwrap();
        let diff = max_abs_diff(&want, &got[b * d..(b + 1) * d]);
        assert!(diff < 4e-3, "slot {b} (len {live}): max |diff| {diff}");
    }
}

#[test]
fn qkv_split_lands_q_and_appends_the_caches() {
    let Some(e) = exec() else { return };
    let (cap, batch, d, ctx) = (3usize, 2usize, 1280usize, 448usize);
    let qkvh = fill(batch * 3 * d, 91);
    let bqh = fill(d, 93);
    let bvh = fill(d, 97);
    let qkv = e.to_device(&qkvh).unwrap();
    let bq = e.to_device(&bqh).unwrap();
    let bv = e.to_device(&bvh).unwrap();
    let slot_ids = [2u32, 0];
    let pos = [5u32, 11];
    let slots = e.to_device_u32(&slot_ids).unwrap();
    let dpos = e.to_device_u32(&pos).unwrap();

    let mut q = e.alloc(cap * d).unwrap();
    let mut kc = e.alloc_u8(cap * ctx * d * 2).unwrap();
    let mut vc = e.alloc_u8(cap * ctx * d * 2).unwrap();
    e.whisper_qkv_split(
        &qkv,
        Some(&bq),
        Some(&bv),
        &mut q,
        &mut kc,
        &mut vc,
        &slots,
        &dpos,
        d,
        ctx,
        batch,
        KvDtype::Fp16,
    )
    .unwrap();

    let qout = e.to_host_len(&q, cap * d).unwrap();
    let kout = e.to_host_f16_from_u8(&kc, cap * ctx * d).unwrap();
    let vout = e.to_host_f16_from_u8(&vc, cap * ctx * d).unwrap();
    for b in 0..batch {
        let row = slot_ids[b] as usize * ctx * d + pos[b] as usize * d;
        for i in 0..d {
            assert_eq!(
                qout[b * d + i],
                qkvh[b * 3 * d + i] + bqh[i],
                "q row {b} dim {i}"
            );
            assert_eq!(
                kout[row + i],
                f16::from_f32(qkvh[b * 3 * d + d + i]),
                "k row {b} dim {i} (no k bias in whisper)"
            );
            assert_eq!(
                vout[row + i],
                f16::from_f32(qkvh[b * 3 * d + 2 * d + i] + bvh[i]),
                "v row {b} dim {i}"
            );
        }
    }
}

#[test]
fn embed_pos_matches_two_row_copies_and_an_add() {
    let Some(e) = exec() else { return };
    let (vocab, ctx, d, batch) = (300usize, 64usize, 1280usize, 3usize);
    let toks = fill(vocab * d, 101);
    let poss = fill(ctx * d, 103);
    let tok = e.to_device(&toks).unwrap();
    let ptab = e.to_device(&poss).unwrap();
    let ids = [7u32, 299, 0];
    let pos = [0u32, 63, 12];
    let dids = e.to_device_u32(&ids).unwrap();
    let dpos = e.to_device_u32(&pos).unwrap();

    let mut x = e.alloc(batch * d).unwrap();
    e.whisper_embed_pos(&tok, &ptab, &dids, &dpos, &mut x, d, batch)
        .unwrap();
    let got = e.to_host_len(&x, batch * d).unwrap();
    for b in 0..batch {
        for i in 0..d {
            let want = toks[ids[b] as usize * d + i] + poss[pos[b] as usize * d + i];
            assert_eq!(got[b * d + i], want, "row {b} dim {i}");
        }
    }
}

#[test]
fn kv_store_lands_the_window_in_its_slot_plane() {
    let Some(e) = exec() else { return };
    let (cap, rows, d) = (3usize, 1500usize, 1280usize);
    let srch = fill(rows * d, 111);
    let biash = fill(d, 113);
    let src = e.to_device(&srch).unwrap();
    let bias = e.to_device(&biash).unwrap();
    let slot = [2u32];
    let slots = e.to_device_u32(&slot).unwrap();

    let mut dst = e.alloc_u8(cap * rows * d * 2).unwrap();
    e.whisper_kv_store(
        &src,
        Some(&bias),
        &mut dst,
        &slots,
        rows,
        d,
        rows,
        1,
        KvDtype::Fp16,
    )
    .unwrap();
    let out = e.to_host_f16_from_u8(&dst, cap * rows * d).unwrap();
    let base = slot[0] as usize * rows * d;
    for r in [0usize, 1, 749, 1499] {
        for i in [0usize, 1, 639, 1279] {
            assert_eq!(
                out[base + r * d + i],
                f16::from_f32(srch[r * d + i] + biash[i]),
                "row {r} dim {i}"
            );
        }
    }
}

/// What `load.rs`'s `self_attn` merge has to guarantee is a LAYOUT: plane p's
/// outputs land at `[b, p*d_out + o]`, so one GEMM replaces three. That splits
/// into two claims with different natures, and conflating them is what made this
/// gate wrong.
///
/// The layout claim is exact and stays exact - the merged plane must be the
/// three planes concatenated, byte for byte.
///
/// The arithmetic claim is not bit-equal and cannot be: both sides are cuBLAS
/// `gemm_ex`, whose algorithm is chosen per problem SHAPE, so N=3840 and N=1280
/// legitimately reduce K=1280 in different orders. Measured here at ~2e-6
/// relative - reassociation noise for an f32-accumulated 1280-term dot, and the
/// original `assert_eq!` was demanding a determinism guarantee no vendor BLAS
/// makes. The bound below is ~10x that and ~5 orders below what a misrouted
/// plane produces (unrelated values, O(1) apart), which is the failure this
/// actually catches.
#[test]
fn merged_qkv_plane_matches_three_separate_gemms() {
    let Some(e) = exec() else { return };
    // exactly what load.rs's `self_attn` builds, on the whisper shape
    let (d_in, d_out, batch) = (1280usize, 1280usize, 3usize);
    let n = d_in * d_out;
    let planes: Vec<Vec<f32>> = [121u32, 127, 131].iter().map(|&s| fill(n, s)).collect();
    let devs: Vec<_> = planes
        .iter()
        .map(|h| paddock_engine::gpu::HalfTensor {
            buf: e.to_device_f16(h, "plane").unwrap(),
            dims: vec![d_in, d_out],
        })
        .collect();
    let mut merged = e.alloc_f16(3 * n).unwrap();
    for (i, t) in devs.iter().enumerate() {
        e.copy_region(&t.buf, 0, &mut merged, i * n, n).unwrap();
    }

    // claim 1, exact: the merged plane is the three planes concatenated
    let merged_host = e.to_host_f16_len(&merged, 3 * n).unwrap();
    for (i, t) in devs.iter().enumerate() {
        assert_f16_bit_equal(
            &e.to_host_f16_len(&t.buf, n).unwrap(),
            &merged_host[i * n..(i + 1) * n],
            &format!("plane {i} did not land at offset {}", i * n),
        );
    }

    let merged = paddock_engine::gpu::HalfTensor {
        buf: merged,
        dims: vec![d_in, 3 * d_out],
    };

    let xh = fill(batch * d_in, 137);
    let mut x = e.alloc_f16(batch * d_in).unwrap();
    let x32 = e.to_device(&xh).unwrap();
    e.convert_f32_f16(&x32, &mut x, batch * d_in).unwrap();

    let mut fused = e.alloc(batch * 3 * d_out).unwrap();
    e.matvec_batch_f16(&merged, &x, &mut fused, batch).unwrap();
    let fused = e.to_host_len(&fused, batch * 3 * d_out).unwrap();

    // claim 2, to the reassociation bound: the fused slice is that plane's GEMM
    for (p, t) in devs.iter().enumerate() {
        let mut y = e.alloc(batch * d_out).unwrap();
        e.matvec_batch_f16(t, &x, &mut y, batch).unwrap();
        let y = e.to_host_len(&y, batch * d_out).unwrap();
        let scale = y.iter().fold(0f32, |m, v| m.max(v.abs()));
        assert!(
            scale > 0.0,
            "plane {p} reference is all zeros - the gate would prove nothing"
        );
        let (mut worst, mut at) = (0f32, 0usize);
        for b in 0..batch {
            for o in 0..d_out {
                let d = (fused[b * 3 * d_out + p * d_out + o] - y[b * d_out + o]).abs();
                if d > worst {
                    worst = d;
                    at = b * d_out + o;
                }
            }
        }
        assert!(
            worst / scale < 2e-5,
            "plane {p}: fused slice drifted {worst} on a {scale}-scale output \
             (relative {}, worst at row {} out {}) - cuBLAS reassociation is ~2e-6; \
             this size means the slice is reading another plane",
            worst / scale,
            at / d_out,
            at % d_out
        );
    }
}

#[test]
fn dec_attn_single_slot_single_key_is_just_v() {
    // The exact c1 first-step shape: one slot, one live key. The answer is
    // V[0] verbatim (softmax over one key is 1.0), which makes any partial
    // or combine mistake unmissable.
    let Some(e) = exec() else { return };
    let (heads, hd) = (20usize, 64usize);
    let d = heads * hd;
    let (cap, batch, ctx) = (1usize, 1usize, 448usize);
    let scale = 1.0 / (hd as f32).sqrt();

    let q = e.to_device(&fill(batch * d, 151)).unwrap();
    let kh = fill(cap * ctx * d, 157);
    let vh = fill(cap * ctx * d, 163);
    let kc = kv_plane(&e, &kh, cap, ctx, d, KvDtype::Fp16);
    let vc = kv_plane(&e, &vh, cap, ctx, d, KvDtype::Fp16);

    let slots = e.to_device_u32(&[0u32]).unwrap();
    let dpos = e.to_device_u32(&[0u32]).unwrap();
    let mut out = e.alloc_f16(cap * d).unwrap();
    let mut part = e.alloc(cap * heads * 32 * (hd + 2)).unwrap();
    e.whisper_dec_attn(
        &q,
        None,
        &kc,
        &vc,
        &slots,
        Some(&dpos),
        &mut out,
        Some(&mut part),
        ctx,
        0,
        1,
        heads,
        hd,
        batch,
        scale,
        KvDtype::Fp16,
    )
    .unwrap();
    let got = e.to_host_f16_len(&out, d).unwrap();
    for i in 0..d {
        let want = f16::from_f32(vh[i]).to_f32();
        assert!(
            (got[i].to_f32() - want).abs() < 1e-3,
            "dim {i}: got {} want {want} (one key means out == V[0])",
            got[i].to_f32()
        );
    }
}

/// The fp8-e4m3 KV arm, on the shape that motivated it: the 1500-frame cross
/// plane at c32 width. e4m3 carries 3 mantissa bits, so this can never be a
/// bit gate - the contract is that the fp8 path reads the same bytes the f16
/// path does (a wrong stride or slot offset at 1-byte width is the failure
/// mode this catches) and lands within the format's own error.
///
/// The bound is loose deliberately relative to what it measures: attention
/// output is a softmax-weighted mean over 1500 independently-rounded rows, so
/// the per-element e4m3 error (up to ~6% half-ulp) averages down hard. A
/// stride/index bug does not average down - it reads unrelated rows and blows
/// past this by orders of magnitude.
#[test]
fn dec_attn_fp8_tracks_the_f16_plane() {
    let Some(e) = exec() else { return };
    let (heads, hd) = (20usize, 64usize);
    let d = heads * hd;
    let (cap, batch, nkv) = (3usize, 2usize, 1500usize);
    let scale = 1.0 / (hd as f32).sqrt();

    let q = e.to_device(&fill(batch * d, 51)).unwrap();
    let kh = fill(cap * nkv * d, 61);
    let vh = fill(cap * nkv * d, 67);
    let slots = e.to_device_u32(&[2u32, 0]).unwrap();
    let mut part = e.alloc(cap * heads * 32 * (hd + 2)).unwrap();

    let mut run = |kv: KvDtype| -> Vec<f32> {
        let k = kv_plane(&e, &kh, cap, nkv, d, kv);
        let v = kv_plane(&e, &vh, cap, nkv, d, kv);
        let mut out = e.alloc_f16(batch * d).unwrap();
        e.whisper_dec_attn(
            &q,
            None,
            &k,
            &v,
            &slots,
            None,
            &mut out,
            Some(&mut part),
            nkv,
            nkv,
            0,
            heads,
            hd,
            batch,
            scale,
            kv,
        )
        .unwrap();
        e.to_host_f16_len(&out, batch * d)
            .unwrap()
            .iter()
            .map(|v| v.to_f32())
            .collect()
    };
    let want = run(KvDtype::Fp16);
    let got = run(KvDtype::Fp8E4m3);

    let scale_ref = want.iter().fold(0f32, |m, v| m.max(v.abs()));
    let diff = max_abs_diff(&want, &got);
    assert!(
        scale_ref > 0.0,
        "reference output is all zeros - the gate would prove nothing"
    );
    assert!(
        diff / scale_ref < 0.10,
        "fp8 cross-attention drifted {diff} on a {scale_ref}-scale output \
         (relative {}) - that is a wiring bug, not e4m3 rounding",
        diff / scale_ref
    );
}

/// The self-attention append at fp8 width: `qkv_split` writes K/V into the
/// slot cache at `pos`, and one live key means the attention output is that
/// V row. Any byte-offset mistake in the 1-byte store lands somewhere else
/// entirely, so this pins the store address, not just the value.
///
/// The bound is e4m3's own half-ulp, because with a single key the softmax
/// weight is exactly 1 and the only arithmetic between `want` and `got` is the
/// store's round to e4m3 (f16 then holds every e4m3 value exactly). Three
/// mantissa bits give half an ulp = |want|/16 in the normal range, and a FLAT
/// 2^-10 below 2^-6 where the subnormals are spaced 2^-9 apart - so anything
/// under ~9.8e-4 rounds to zero and there is nothing else it could round to.
///
/// That last part is what had this gate red since it was written:
/// it floored a relative test at 1e-3, below e4m3's smallest subnormal, so every
/// element the format cannot represent failed by construction. dim 101 wants
/// 3.19e-4 and got the only answer available, 0. Stating the format's guarantee
/// is not a widened tolerance - it is 10x TIGHTER than the 0.10 it replaces
/// everywhere e4m3 can actually hold the number.
#[test]
fn qkv_split_appends_at_fp8_width() {
    let Some(e) = exec() else { return };
    let (heads, hd) = (20usize, 64usize);
    let d = heads * hd;
    let (cap, batch, ctx) = (3usize, 1usize, 448usize);
    let scale = 1.0 / (hd as f32).sqrt();

    let qkvh = fill(batch * 3 * d, 91);
    let qkv = e.to_device(&qkvh).unwrap();
    let bv = e.to_device(&fill(d, 97)).unwrap();
    let bvh = e.to_host_len(&bv, d).unwrap();
    let slots = e.to_device_u32(&[2u32]).unwrap();
    let dpos = e.to_device_u32(&[0u32]).unwrap();
    let mut part = e.alloc(cap * heads * 32 * (hd + 2)).unwrap();

    let mut q = e.alloc(cap * d).unwrap();
    let mut kc = e.alloc_u8(cap * ctx * d).unwrap();
    let mut vc = e.alloc_u8(cap * ctx * d).unwrap();
    e.whisper_qkv_split(
        &qkv,
        None,
        Some(&bv),
        &mut q,
        &mut kc,
        &mut vc,
        &slots,
        &dpos,
        d,
        ctx,
        batch,
        KvDtype::Fp8E4m3,
    )
    .unwrap();

    let mut out = e.alloc_f16(cap * d).unwrap();
    e.whisper_dec_attn(
        &q,
        None,
        &kc,
        &vc,
        &slots,
        Some(&dpos),
        &mut out,
        Some(&mut part),
        ctx,
        0,
        1,
        heads,
        hd,
        batch,
        scale,
        KvDtype::Fp8E4m3,
    )
    .unwrap();

    let got = e.to_host_f16_len(&out, d).unwrap();
    // half an ulp of e4m3: |w|/16 where the format is normal (>= 2^-6), a flat
    // 2^-10 in the subnormal range below it
    let tol = |w: f32| (w.abs() / 16.0).max(1.0 / 1024.0);
    for i in 0..d {
        let want = qkvh[2 * d + i] + bvh[i];
        let diff = (got[i].to_f32() - want).abs();
        assert!(
            diff <= tol(want),
            "dim {i}: got {} want {want} (off by {diff}, e4m3 half-ulp is {}) - \
             one key means out == the appended V row",
            got[i].to_f32(),
            tol(want)
        );
    }
}

#[test]
fn xattn_probs_matches_a_host_softmax_for_the_nominated_heads_only() {
    // The word-timing read-out. Three properties, and the third is
    // the one a careless implementation gets wrong:
    //
    //   1. the values are softmax(q·k*scale) over the encoder frames
    //   2. each row sums to 1
    //   3. row `sel` holds the head named by `heads[sel]` - Not head `sel`.
    //      Getting that wrong still produces a plausible, normalised, entirely
    //      wrong alignment, and nothing downstream could tell.
    let Some(e) = exec() else { return };
    let (heads_n, hd) = (20usize, 64usize);
    let d = heads_n * hd;
    let (cap, batch, n_enc) = (2usize, 2usize, 96usize);
    let scale = 1.0 / (hd as f32).sqrt();

    let qh = fill(batch * d, 211);
    let bh = fill(d, 217);
    let kh = fill(cap * n_enc * d, 223);
    let q = e.to_device(&qh).unwrap();
    let qb = e.to_device(&bh).unwrap();
    let kc = kv_plane(&e, &kh, cap, n_enc, d, KvDtype::Fp16);

    // deliberately not in order and not the low heads: a `sel`-for-`h` slip
    // survives an identity list
    let sel: Vec<u32> = vec![17, 3, 11];
    let n_sel = sel.len();
    // slot 1 answers active row 0 - the plane row must follow `slots`, not `b`
    let slots = e.to_device_u32(&[1u32, 0u32]).unwrap();
    let dheads = e.to_device_u32(&sel).unwrap();
    let mut out = e.alloc(batch * n_sel * n_enc).unwrap();

    e.whisper_xattn_probs(
        &q,
        Some(&qb),
        &kc,
        &slots,
        &dheads,
        &mut out,
        0,
        n_enc,
        n_enc,
        heads_n,
        hd,
        n_sel,
        batch,
        scale,
        KvDtype::Fp16,
    )
    .unwrap();
    let got = e.to_host_len(&out, batch * n_sel * n_enc).unwrap();

    let slot_of = [1usize, 0usize];
    for b in 0..batch {
        for (s, &h) in sel.iter().enumerate() {
            let h = h as usize;
            let slot = slot_of[b];
            // host oracle in f64, through the same f16 rounding the plane took
            let mut scores = vec![0.0f64; n_enc];
            for (r, sc) in scores.iter_mut().enumerate() {
                let mut dot = 0.0f64;
                for i in 0..hd {
                    let qv = (qh[b * d + h * hd + i] + bh[h * hd + i]) * scale;
                    let kv = f16::from_f32(kh[(slot * n_enc + r) * d + h * hd + i]).to_f32();
                    dot += qv as f64 * kv as f64;
                }
                *sc = dot;
            }
            let m = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let den: f64 = scores.iter().map(|v| (v - m).exp()).sum();

            let row = &got[(b * n_sel + s) * n_enc..(b * n_sel + s + 1) * n_enc];
            let sum: f32 = row.iter().sum();
            assert!(
                (sum - 1.0).abs() < 2e-3,
                "b={b} sel={s}: row sums to {sum}, not 1"
            );
            for (r, &g) in row.iter().enumerate() {
                let want = ((scores[r] - m).exp() / den) as f32;
                assert!(
                    (g - want).abs() < 2e-4,
                    "b={b} head={h} frame={r}: {g} vs host {want}"
                );
            }
        }
    }
}

#[test]
fn xattn_probs_on_an_empty_plane_is_zeros_not_nan() {
    // A window with no encoder mass is a real input (silence, or a clip
    // shorter than the frame it was padded into). A nan here would poison the
    // whole DTW and surface as a transcript with no times at all rather than
    // as an error anyone could read.
    let Some(e) = exec() else { return };
    let (heads_n, hd, n_enc) = (20usize, 64usize, 32usize);
    let d = heads_n * hd;
    let q = e.to_device(&vec![0.0f32; d]).unwrap();
    let kc = kv_plane(&e, &vec![0.0f32; n_enc * d], 1, n_enc, d, KvDtype::Fp16);
    let slots = e.to_device_u32(&[0u32]).unwrap();
    let dheads = e.to_device_u32(&[0u32]).unwrap();
    let mut out = e.alloc(n_enc).unwrap();
    e.whisper_xattn_probs(
        &q,
        None,
        &kc,
        &slots,
        &dheads,
        &mut out,
        0,
        n_enc,
        n_enc,
        heads_n,
        hd,
        1,
        1,
        1.0,
        KvDtype::Fp16,
    )
    .unwrap();
    let got = e.to_host_len(&out, n_enc).unwrap();
    assert!(got.iter().all(|v| v.is_finite()), "non-finite in {got:?}");
    // all-zero scores are a real uniform distribution, so this is 1/n_enc
    let want = 1.0 / n_enc as f32;
    assert!(
        got.iter().all(|v| (v - want).abs() < 1e-5),
        "not uniform: {got:?}"
    );
}

#[test]
fn enc_qkv_split_matches_the_slices_it_replaces() {
    // the encoder's fused q|k|v GEMM landing splits into the three
    // planes vision_attn eats, biases folded (k has none). The claim is
    // bit-exact: each output element is one f32 read (+ one f32 add), the
    // same arithmetic as the bias_add launches this replaces.
    let Some(e) = exec() else { return };
    let (rows, d) = (1500usize, 1280usize);
    let fused_h = fill(rows * 3 * d, 211);
    let bqh = fill(d, 213);
    let bvh = fill(d, 217);
    let fused = e.to_device(&fused_h).unwrap();
    let bq = e.to_device(&bqh).unwrap();
    let bv = e.to_device(&bvh).unwrap();
    let mut q = e.alloc(rows * d).unwrap();
    let mut k = e.alloc(rows * d).unwrap();
    let mut v = e.alloc(rows * d).unwrap();
    e.whisper_enc_qkv_split(
        &fused,
        Some(&bq),
        Some(&bv),
        &mut q,
        &mut k,
        &mut v,
        d,
        rows,
    )
    .unwrap();
    let qh = e.to_host_len(&q, rows * d).unwrap();
    let kh = e.to_host_len(&k, rows * d).unwrap();
    let vh = e.to_host_len(&v, rows * d).unwrap();
    for r in [0usize, 1, 749, 1499] {
        for i in [0usize, 1, 639, 1279] {
            assert_eq!(
                qh[r * d + i],
                fused_h[r * 3 * d + i] + bqh[i],
                "q row {r} dim {i}"
            );
            assert_eq!(
                kh[r * d + i],
                fused_h[r * 3 * d + d + i],
                "k row {r} dim {i}"
            );
            assert_eq!(
                vh[r * d + i],
                fused_h[r * 3 * d + 2 * d + i] + bvh[i],
                "v row {r} dim {i}"
            );
        }
    }
}

#[test]
fn kv_store_batch_matches_per_layer_stores() {
    // one launch stores every layer's cross plane off the
    // layer-batched [rows, n_layer*d] landing. Same claim as the kv_store
    // gate above, per layer: element (r, i) of layer li's slot plane is
    // f16(src[r, li*d + i] + bias[li*d + i]).
    let Some(e) = exec() else { return };
    let (cap, rows, d, nl) = (2usize, 96usize, 64usize, 4usize);
    let srch = fill(rows * nl * d, 311);
    let bvh = fill(nl * d, 313);
    let src = e.to_device(&srch).unwrap();
    let bias = e.to_device(&bvh).unwrap();
    let slots = e.to_device_u32(&[1u32]).unwrap();
    let planes: Vec<_> = (0..nl)
        .map(|_| e.alloc_u8(cap * rows * d * 2).unwrap())
        .collect();
    let ptrs = e.pointer_table(&planes).unwrap();
    e.whisper_kv_store_batch(
        &src,
        Some(&bias),
        &ptrs,
        &slots,
        rows,
        d,
        nl,
        rows,
        KvDtype::Fp16,
    )
    .unwrap();
    for (li, plane) in planes.iter().enumerate() {
        let out = e.to_host_f16_from_u8(plane, cap * rows * d).unwrap();
        let base = rows * d; // slot 1
        for r in [0usize, 1, 95] {
            for i in [0usize, 1, 63] {
                assert_eq!(
                    out[base + r * d + i],
                    f16::from_f32(srch[r * nl * d + li * d + i] + bvh[li * d + i]),
                    "layer {li} row {r} dim {i}"
                );
            }
        }
    }
}
