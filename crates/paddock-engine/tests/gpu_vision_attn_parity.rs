//! Shape-level parity for the tower's bidirectional attention - both entry
//! points (`vision_attn`, `vision_attn_x`) and both numeric classes.
//!
//! This kernel had no direct coverage for a long while - it was pinned only
//! end-to-end through the llama-mtmd oracle, which stays the arbiter of the
//! whole encode - and the original shape silently truncated the q.k dot to
//! floor(hd/32)*32 dims (n_warps = hd>>5 dropped the partial warp; head_dim is
//! 72 on the qwen3-vl tower). Plain softmax attention is what decides here.
//!
//! Two implementations answer these calls:
//!   - the f32 scalar kernel, which is exact and gated at 1e-4;
//!   - the mma/tensor-core kernel, the default, which rounds q/k/v to f16 into
//!     its fragments and is gated at that class's tolerance.
//!     Both legs run in one test because the selector is an env var and cargo runs
//!     test fns on concurrent threads - splitting them would let one leg's setting
//!     leak into the other's launches.

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

/// One attention shape: `n_batch` independent groups, each `nq` query rows
/// against `nkv` KV rows, laid out [batch][row][head][dim].
struct Shape {
    nq: usize,
    nkv: usize,
    heads: usize,
    hd: usize,
    batch: usize,
}

/// Plain two-pass softmax attention, the reference every leg diffs against.
fn reference(s: &Shape, q: &[f32], k: &[f32], v: &[f32], scale: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; s.batch * s.nq * s.heads * s.hd];
    for b in 0..s.batch {
        let qb = b * s.nq * s.heads * s.hd;
        let kb = b * s.nkv * s.heads * s.hd;
        for h in 0..s.heads {
            for i in 0..s.nq {
                let qi = &q[qb + (i * s.heads + h) * s.hd..][..s.hd];
                let mut scores = vec![0.0f32; s.nkv];
                let mut m = f32::NEG_INFINITY;
                for (j, sc) in scores.iter_mut().enumerate() {
                    let kj = &k[kb + (j * s.heads + h) * s.hd..][..s.hd];
                    *sc = qi.iter().zip(kj).map(|(a, b)| a * scale * b).sum();
                    m = m.max(*sc);
                }
                let mut l = 0.0f32;
                for sc in scores.iter_mut() {
                    *sc = (*sc - m).exp();
                    l += *sc;
                }
                for d in 0..s.hd {
                    let mut acc = 0.0f32;
                    for (j, sc) in scores.iter().enumerate() {
                        acc += sc * v[kb + (j * s.heads + h) * s.hd + d];
                    }
                    out[qb + (i * s.heads + h) * s.hd + d] = acc / l;
                }
            }
        }
    }
    out
}

/// Run one shape through whichever kernel the env currently selects and return
/// the largest absolute divergence from the reference. `batched` picks the
/// `vision_attn_x` export (cross-attention + grid.z) over plain `vision_attn`.
fn max_diff(exec: &GpuExecutor, s: &Shape, batched: bool) -> f32 {
    let mut seed = 0x9e3779b97f4a7c15u64;
    let q = lcg_fill(&mut seed, s.batch * s.nq * s.heads * s.hd);
    let k = lcg_fill(&mut seed, s.batch * s.nkv * s.heads * s.hd);
    let v = lcg_fill(&mut seed, s.batch * s.nkv * s.heads * s.hd);
    let scale = 1.0f32 / (s.hd as f32).sqrt();

    let d_q = exec.to_device(&q).expect("q");
    let d_k = exec.to_device(&k).expect("k");
    let d_v = exec.to_device(&v).expect("v");
    let mut d_o = exec.alloc(s.batch * s.nq * s.heads * s.hd).expect("o");
    if batched {
        exec.vision_attn_x(
            &d_q, &d_k, &d_v, &mut d_o, s.nq, s.nkv, s.heads, s.hd, s.batch, scale,
        )
        .expect("attn_x");
    } else {
        exec.vision_attn(&d_q, &d_k, &d_v, &mut d_o, s.nq, s.heads, s.hd, scale)
            .expect("attn");
    }
    let gpu = exec.to_host(&d_o).expect("out");

    let cpu = reference(s, &q, &k, &v, scale);
    cpu.iter()
        .zip(&gpu)
        .fold(0.0f32, |acc, (r, g)| acc.max((r - g).abs()))
}

#[test]
fn vision_attn_matches_reference_softmax() {
    let Some(exec) = common::gpu_arc() else {
        return;
    };

    // nq 67/129 span a partial final query tile and a second block; nkv 67 and
    // 129 span a partial final KEY tile (the mask that keeps zero-filled pad
    // rows from carrying exp(0-m) of softmax mass); hd 72 is the qwen3-vl head
    // that needs the 80-dim mma pad, 128 the widest we take. The two batched
    // rows are granite-vision's Q-Former: the 16x64 cross shape, and a batched
    // self-attention with a partial query tile.
    let shapes: &[(Shape, bool)] = &[
        (
            Shape {
                nq: 67,
                nkv: 67,
                heads: 3,
                hd: 72,
                batch: 1,
            },
            false,
        ),
        (
            Shape {
                nq: 129,
                nkv: 129,
                heads: 2,
                hd: 64,
                batch: 1,
            },
            false,
        ),
        (
            Shape {
                nq: 40,
                nkv: 40,
                heads: 4,
                hd: 128,
                batch: 1,
            },
            false,
        ),
        (
            Shape {
                nq: 16,
                nkv: 64,
                heads: 2,
                hd: 64,
                batch: 3,
            },
            true,
        ),
        (
            Shape {
                nq: 24,
                nkv: 24,
                heads: 2,
                hd: 72,
                batch: 2,
            },
            true,
        ),
    ];

    // Leg 1: the exact f32 kernel. Nothing here is allowed to drift.
    unsafe { std::env::set_var("PADDOCK_NO_VIS_MMA", "1") };
    for (s, batched) in shapes {
        let d = max_diff(&exec, s, *batched);
        eprintln!(
            "f32  nq={} nkv={} heads={} hd={} batch={}: max_abs_diff {d:.2e}",
            s.nq, s.nkv, s.heads, s.hd, s.batch
        );
        assert!(d < 1e-4, "f32 vision attn diverges from softmax: {d}");
    }

    // Leg 2: the mma kernel that actually serves. q/k/v round to f16 into the
    // fragments (f32 accumulate, f32 softmax), which measures ~1e-4 absolute on
    // [-1,1) activations - the same class the tower's f16 GEMMs already run in.
    // The gate is loose enough for data and shape variation and far tighter
    // than any fragment-indexing mistake, which is O(1) wrong, not O(1e-3).
    unsafe { std::env::remove_var("PADDOCK_NO_VIS_MMA") };
    for (s, batched) in shapes {
        let d = max_diff(&exec, s, *batched);
        eprintln!(
            "mma  nq={} nkv={} heads={} hd={} batch={}: max_abs_diff {d:.2e}",
            s.nq, s.nkv, s.heads, s.hd, s.batch
        );
        assert!(d < 1e-3, "mma vision attn diverges from softmax: {d}");
    }
}
