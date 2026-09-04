//! Parity gate for `sam_attn` - SAM ViTDet attention with the DECOMPOSED
//! relative-position bias (DeepSeek-OCR's first tower):
//!
//!   out = softmax(q·kᵀ·scale + rel_h + rel_w) · v
//!   rel_h[i, ky] = Σ_d q[i,d] · Rh[qy(i), ky, d]     - RAW q, not scaled
//!   rel_w[i, kx] = Σ_d q[i,d] · Rw[qx(i), kx, d]
//!
//! The kernel is exact-f32 class (same lineage as the scalar
//! `pd_vision_attn_kernel`), so the gate is 1e-4 against a plain two-pass
//! reference. The bias is the part no other kernel covers and the part that is
//! silent when wrong - zero tables reduce this to ordinary attention, so one
//! leg pins exactly that reduction against `vision_attn_x` too.
//!
//! Shapes are the family's real ones: side 14 (windowed blocks, batch = the
//! 25 windows of a 1024px view), side 64 (global blocks at 1024px), side 40
//! (global at a 640px crop), all at SAM ViT-B's 12 heads × hd 64.
// Test code: a failed assumption stops the test where it happened.
#![allow(clippy::unwrap_used)]

mod common;

use paddock_engine::gpu::GpuExecutor;

/// deterministic LCG floats in [-1, 1)
fn lcg_fill(seed: &mut u64, n: usize) -> Vec<f32> {
    (0..n)
        .map(|_| {
            *seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((*seed >> 40) as f32) / ((1u64 << 23) as f32) - 1.0
        })
        .collect()
}

struct Shape {
    side: usize,
    heads: usize,
    hd: usize,
    batch: usize,
}

impl Shape {
    fn n(&self) -> usize {
        self.side * self.side
    }
}

/// Two-pass reference with the decomposed bias, in the exact order the
/// checkpoint's `Attention.forward` composes it: rel dots on raw q, scale on
/// the q·k term only.
fn reference(
    s: &Shape,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    rh: &[f32],
    rw: &[f32],
    scale: f32,
) -> Vec<f32> {
    let (n, side, hd) = (s.n(), s.side, s.hd);
    let mut out = vec![0.0f32; s.batch * n * s.heads * hd];
    for b in 0..s.batch {
        let base = b * n * s.heads * hd;
        for h in 0..s.heads {
            for i in 0..n {
                let qi = &q[base + (i * s.heads + h) * hd..][..hd];
                let (qy, qx) = (i / side, i % side);
                // per-axis bias vectors for this query row
                let relh: Vec<f32> = (0..side)
                    .map(|ky| {
                        let t = &rh[(qy * side + ky) * hd..][..hd];
                        qi.iter().zip(t).map(|(a, b)| a * b).sum()
                    })
                    .collect();
                let relw: Vec<f32> = (0..side)
                    .map(|kx| {
                        let t = &rw[(qx * side + kx) * hd..][..hd];
                        qi.iter().zip(t).map(|(a, b)| a * b).sum()
                    })
                    .collect();
                let mut scores = vec![0.0f32; n];
                let mut m = f32::NEG_INFINITY;
                for (j, sc) in scores.iter_mut().enumerate() {
                    let kj = &k[base + (j * s.heads + h) * hd..][..hd];
                    let dot: f32 = qi.iter().zip(kj).map(|(a, b)| a * b).sum();
                    *sc = dot * scale + relh[j / side] + relw[j % side];
                    m = m.max(*sc);
                }
                let mut l = 0.0f32;
                for sc in scores.iter_mut() {
                    *sc = (*sc - m).exp();
                    l += *sc;
                }
                for d in 0..hd {
                    let mut acc = 0.0f32;
                    for (j, sc) in scores.iter().enumerate() {
                        acc += sc * v[base + (j * s.heads + h) * hd + d];
                    }
                    out[base + i * s.heads * hd + h * hd + d] = acc / l;
                }
            }
        }
    }
    out
}

fn run_shape(
    exec: &GpuExecutor,
    s: &Shape,
    seed: &mut u64,
    zero_bias: bool,
) -> (Vec<f32>, Vec<f32>) {
    let n = s.n();
    let qkv = s.batch * n * s.heads * s.hd;
    let q = lcg_fill(seed, qkv);
    let k = lcg_fill(seed, qkv);
    let v = lcg_fill(seed, qkv);
    let tbl = s.side * s.side * s.hd;
    let (rh, rw) = if zero_bias {
        (vec![0.0f32; tbl], vec![0.0f32; tbl])
    } else {
        // rel-pos tables are small learned values; keep them O(0.1) so the
        // bias is comparable to the scaled q·k term rather than drowning it
        let mut t1 = lcg_fill(seed, tbl);
        let mut t2 = lcg_fill(seed, tbl);
        for x in t1.iter_mut().chain(t2.iter_mut()) {
            *x *= 0.1;
        }
        (t1, t2)
    };
    let scale = 1.0 / (s.hd as f32).sqrt();

    let want = reference(s, &q, &k, &v, &rh, &rw, scale);

    let dq = exec.to_device(&q).unwrap();
    let dk = exec.to_device(&k).unwrap();
    let dv = exec.to_device(&v).unwrap();
    let drh = exec.to_device(&rh).unwrap();
    let drw = exec.to_device(&rw).unwrap();
    let mut dout = exec.alloc(qkv).unwrap();
    exec.sam_attn(
        &dq, &dk, &dv, &drh, &drw, &mut dout, s.batch, s.side, s.heads, s.hd, scale,
    )
    .expect("sam_attn launch");
    let got = exec.to_host(&dout).unwrap();
    (want, got)
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

#[test]
fn sam_attn_matches_the_reference_on_the_family_shapes() {
    let Some(exec) = common::gpu() else { return };

    // (side, heads, batch): the three geometries the family actually
    // launches. Window batch is trimmed from 25 (the kernel is batch-parallel
    // on grid.z, 4 exercises the same paths), and the GLOBAL legs run 1 head:
    // the host reference is O(side⁴·hd·heads) and 4 heads at side 64 put this
    // gate at 190 s in debug Rust for zero structural gain - the head stride
    // is already pinned 12-wide by the windowed leg.
    //
    // Two numeric classes through the same shapes (the vision_attn pattern):
    // the exact-f32 kernel gates at 1e-4, the mma arm (f16 fragments + f16
    // Pw bias table, f32 accumulate/softmax) at its class tolerance. 1e-3 is
    // what the vision mma arm uses; any fragment- or bias-indexing mistake is
    // O(1) wrong, not O(1e-3).
    let shapes = [
        (14usize, 12usize, 4usize, "windowed"),
        (64, 1, 1, "global@1024"),
        (40, 2, 1, "global@640"),
    ];

    unsafe { std::env::set_var("PADDOCK_NO_SAM_MMA", "1") };
    let mut seed = 0x5eed_0c11u64;
    for (side, heads, batch, label) in shapes {
        let s = Shape {
            side,
            heads,
            hd: 64,
            batch,
        };
        let (want, got) = run_shape(&exec, &s, &mut seed, false);
        let d = max_abs_diff(&want, &got);
        assert!(d < 1e-4, "{label}: exact-f32 class drifted, max |Δ| = {d}");
    }
    unsafe { std::env::remove_var("PADDOCK_NO_SAM_MMA") };

    let mut seed = 0x5eed_0c11u64;
    for (side, heads, batch, label) in shapes {
        let s = Shape {
            side,
            heads,
            hd: 64,
            batch,
        };
        let (want, got) = run_shape(&exec, &s, &mut seed, false);
        let d = max_abs_diff(&want, &got);
        assert!(d < 1e-3, "{label}: mma class drifted, max |Δ| = {d}");
    }
}

/// Zero tables must reduce the kernel to plain bidirectional attention - and
/// that reduction is also pinned against the existing `vision_attn_x` kernel,
/// so the two implementations cannot drift apart on their shared math.
#[test]
fn zero_bias_reduces_to_plain_attention() {
    let Some(exec) = common::gpu() else { return };
    let mut seed = 0xb1a5_0000u64;
    let s = Shape {
        side: 14,
        heads: 12,
        hd: 64,
        batch: 3,
    };
    let (want, got) = run_shape(&exec, &s, &mut seed, true);
    let d = max_abs_diff(&want, &got);
    assert!(
        d < 1e-4,
        "zero-bias leg drifted from the reference, max |Δ| = {d}"
    );

    // same inputs through vision_attn_x (scalar class forced so both legs are
    // exact-f32 - the mma default is a different numeric class)
    let mut seed2 = 0xb1a5_0000u64;
    let n = s.n();
    let qkv = s.batch * n * s.heads * s.hd;
    let q = lcg_fill(&mut seed2, qkv);
    let k = lcg_fill(&mut seed2, qkv);
    let v = lcg_fill(&mut seed2, qkv);
    let scale = 1.0 / (s.hd as f32).sqrt();
    let dq = exec.to_device(&q).unwrap();
    let dk = exec.to_device(&k).unwrap();
    let dv = exec.to_device(&v).unwrap();
    let mut dout = exec.alloc(qkv).unwrap();
    // SAFETY of the env flip: this test binary runs its fns on threads, but
    // nothing else in this file launches vision attention, so the flip cannot
    // leak into a concurrent mma leg.
    unsafe { std::env::set_var("PADDOCK_NO_VIS_MMA", "1") };
    exec.vision_attn_x(
        &dq, &dk, &dv, &mut dout, n, n, s.heads, s.hd, s.batch, scale,
    )
    .expect("vision_attn_x launch");
    unsafe { std::env::remove_var("PADDOCK_NO_VIS_MMA") };
    let plain = exec.to_host(&dout).unwrap();
    let d2 = max_abs_diff(&got, &plain);
    assert!(
        d2 < 1e-4,
        "sam_attn(zero bias) != vision_attn_x, max |Δ| = {d2}"
    );
}
