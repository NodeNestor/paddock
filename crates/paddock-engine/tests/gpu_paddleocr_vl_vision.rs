//! PaddleOCR-VL vision tower + preprocessing oracle gate: our
//! tower vs the checkpoint's own modeling_paddleocr_vl.py (f32, eager, CUDA)
//! on four deterministic probes covering every smart_resize branch and both
//! aspect orders.
//!
//! Oracle artifacts come from an out-of-tree dump tool into
//! `<models>/ocr-battery/paddle-oracle/` - outside the repo, same convention
//! as the DeepEncoder gate. Skips cleanly when absent; fails under
//! `PADDOCK_STRICT_GATES=1`.
//!
//! Two numeric classes deliberately:
//! * preprocessing is exact - the resized u8 bytes and the f32 pixel_values
//!   must match the HF processor bit-for-bit (PIL fixed-point bicubic +
//!   the oracle-verified all-f32 normalize);
//! * the tower is CLASS tolerance - f16 weight planes with f32 accumulation
//!   against an f32 reference, 27 layers deep. Same thresholds as the
//!   DeepEncoder gate: a real graph bug (wrong eps, swapped GELU, transposed
//!   h/w rope) moves relL2 by orders of magnitude, not percent. The `embd`
//!   tap (patch GEMM + interpolated positions, one GEMM deep) is gated an
//!   order tighter to catch stem/position bugs before they smear.

mod common;

use std::sync::Arc;

use paddock_engine::gpu_model::paddleocr_vl::preprocess::{self, FACTOR, PixelBudget};
use paddock_engine::gpu_model::paddleocr_vl::vision::{EncodeTaps, VisionModel};
use paddock_models::mapped::MappedGguf;

const MMPROJ: &[&str] = &["PaddleOCR-VL-1.6-GGUF/PaddleOCR-VL-1.6-GGUF-mmproj.gguf"];

fn oracle_dir() -> Option<std::path::PathBuf> {
    common::model_roots()
        .iter()
        .map(|r| r.join("ocr-battery").join("paddle-oracle"))
        .find(|p| p.join("manifest.json").exists())
}

fn read_f32(path: &std::path::Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

struct Probe {
    tag: &'static str,
    src_w: usize,
    src_h: usize,
    seed: u32,
}

/// The dump script's PROBES table, verbatim.
const PROBES: &[Probe] = &[
    Probe {
        tag: "a_700x500",
        src_w: 700,
        src_h: 500,
        seed: 1,
    },
    Probe {
        tag: "b_300x900",
        src_w: 300,
        src_h: 900,
        seed: 2,
    },
    Probe {
        tag: "c_1400x900",
        src_w: 1400,
        src_h: 900,
        seed: 3,
    },
    Probe {
        tag: "d_100x80",
        src_w: 100,
        src_h: 80,
        seed: 4,
    },
];

fn manifest(dir: &std::path::Path) -> serde_json::Value {
    let text = std::fs::read_to_string(dir.join("manifest.json")).expect("manifest");
    serde_json::from_str(&text).expect("manifest json")
}

/// Preprocessing must be exact: resized bytes and pixel_values bit-for-bit.
#[test]
fn preprocessing_matches_the_hf_processor_bit_for_bit() {
    let Some(dir) = oracle_dir() else {
        common::missing("no PaddleOCR-VL oracle");
        return;
    };
    let m = manifest(&dir);
    let budget = PixelBudget {
        min_pixels: m["min_pixels"].as_u64().unwrap() as usize,
        max_pixels: m["max_pixels"].as_u64().unwrap() as usize,
    };

    for p in PROBES {
        let info = &m["probes"][p.tag];
        let (rw, rh) = (
            info["resized_w"].as_u64().unwrap() as usize,
            info["resized_h"].as_u64().unwrap() as usize,
        );

        let rgb = preprocess::hash_pixels(p.src_w, p.src_h, p.seed);
        let (th, tw) =
            preprocess::smart_resize(p.src_h, p.src_w, FACTOR, budget).expect("smart_resize");
        assert_eq!((tw, th), (rw, rh), "{}: smart_resize target", p.tag);

        // the sharp anchor: PIL bicubic output bytes
        let resized = paddock_engine::gpu_model::pillow::resize_rgb8(
            &rgb,
            p.src_w,
            p.src_h,
            tw,
            th,
            paddock_engine::gpu_model::pillow::Filter::Bicubic,
        );
        let want_bytes = std::fs::read(dir.join(format!("resized_{}.bin", p.tag))).unwrap();
        assert_eq!(
            resized.len(),
            want_bytes.len(),
            "{}: resized byte count",
            p.tag
        );
        let bad = resized
            .iter()
            .zip(&want_bytes)
            .filter(|(a, b)| a != b)
            .count();
        assert_eq!(
            bad,
            0,
            "{}: {bad} of {} resized bytes differ from PIL",
            p.tag,
            resized.len()
        );

        // normalize (all-f32) must reproduce pixel_values bit-for-bit; pv is
        // (L, 3, 14, 14) in RASTER patch order, ours is planar CHW - compare
        // through the layout mapping.
        let planar = preprocess::normalize_rgb(&resized, tw, th);
        let want_pv = read_f32(&dir.join(format!("pv_{}.bin", p.tag)));
        let (gw, gh) = (tw / 14, th / 14);
        assert_eq!(
            want_pv.len(),
            gw * gh * 3 * 196,
            "{}: pv element count",
            p.tag
        );
        let mut bad = 0usize;
        for gy in 0..gh {
            for gx in 0..gw {
                let patch_base = ((gy * gw) + gx) * 3 * 196;
                for c in 0..3 {
                    for ky in 0..14 {
                        for kx in 0..14 {
                            let ours = planar[c * tw * th + (gy * 14 + ky) * tw + gx * 14 + kx];
                            let ref_v = want_pv[patch_base + c * 196 + ky * 14 + kx];
                            if ours.to_bits() != ref_v.to_bits() {
                                bad += 1;
                            }
                        }
                    }
                }
            }
        }
        assert_eq!(
            bad,
            0,
            "{}: {bad} of {} pixel_values differ",
            p.tag,
            want_pv.len()
        );
        eprintln!(
            "{}: {}x{} -> {}x{} preprocess bit-exact",
            p.tag, p.src_w, p.src_h, tw, th
        );
    }
}

struct Diff {
    max_abs: f32,
    mean_abs: f64,
    rel_l2: f64,
    cosine: f64,
}

fn diff(got: &[f32], want: &[f32]) -> Diff {
    assert_eq!(got.len(), want.len());
    let (mut max_abs, mut sum_abs, mut d2, mut w2, mut g2, mut dot) =
        (0f32, 0f64, 0f64, 0f64, 0f64, 0f64);
    for (&g, &w) in got.iter().zip(want) {
        let d = g - w;
        max_abs = max_abs.max(d.abs());
        sum_abs += d.abs() as f64;
        d2 += (d as f64) * (d as f64);
        w2 += (w as f64) * (w as f64);
        g2 += (g as f64) * (g as f64);
        dot += (g as f64) * (w as f64);
    }
    Diff {
        max_abs,
        mean_abs: sum_abs / got.len() as f64,
        rel_l2: (d2 / w2.max(1e-30)).sqrt(),
        cosine: dot / (w2.sqrt() * g2.sqrt()).max(1e-30),
    }
}

/// Reorder a reference tensor from RASTER row order into our merged
/// 2×2-block row order so the two sides compare row-for-row.
fn raster_to_merged(want: &[f32], gw: usize, gh: usize, e: usize) -> Vec<f32> {
    assert_eq!(want.len(), gw * gh * e);
    let mut out = Vec::with_capacity(want.len());
    for yb in (0..gh).step_by(2) {
        for xb in (0..gw).step_by(2) {
            for dy in 0..2 {
                for dx in 0..2 {
                    let src = ((yb + dy) * gw + (xb + dx)) * e;
                    out.extend_from_slice(&want[src..src + e]);
                }
            }
        }
    }
    out
}

/// The projector rows: reference token (h', w') raster over the MERGED grid.
/// Our merged-block walk emits exactly one output token per 4 consecutive
/// tower rows, in (yb, xb) raster order - the same order. Identity map.
#[test]
fn tower_and_projector_match_the_reference() {
    let Some(dir) = oracle_dir() else {
        common::missing("no PaddleOCR-VL oracle");
        return;
    };
    let Some(path) = common::model("PADDLEOCR_VL_MMPROJ", MMPROJ) else {
        return;
    };
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    let m = manifest(&dir);
    let map = MappedGguf::open(&path).expect("open mmproj");
    let mut vm = VisionModel::load(Arc::clone(&exec), &map).expect("load vision tower");
    let budget = vm.budget;

    for p in PROBES {
        let info = &m["probes"][p.tag];
        let (gw, gh) = (
            info["grid_w"].as_u64().unwrap() as usize,
            info["grid_h"].as_u64().unwrap() as usize,
        );

        let rgb = preprocess::hash_pixels(p.src_w, p.src_h, p.seed);
        let (planar, tw, th) =
            preprocess::preprocess_rgb(&rgb, p.src_w, p.src_h, budget).expect("preprocess");
        assert_eq!((tw / 14, th / 14), (gw, gh), "{}: grid", p.tag);

        let mut taps = EncodeTaps::default();
        let out = vm
            .encode_batch_taps(&[(&planar, tw, th)], Some(&mut taps))
            .expect("encode")
            .pop()
            .unwrap();
        assert_eq!((out.nx, out.ny), (gw / 2, gh / 2), "{}: merged grid", p.tag);

        // stage 1: encoder input (patch GEMM + interpolated positions) - one
        // GEMM deep, so the f16 floor is tight and a stem/pos bug is loud
        let want_embd = raster_to_merged(
            &read_f32(&dir.join(format!("embd_{}.bin", p.tag))),
            gw,
            gh,
            1152,
        );
        let d = diff(&taps.embd, &want_embd);
        eprintln!(
            "{}: embd  max|Δ| {:.4}  mean|Δ| {:.5}  relL2 {:.5}  cos {:.6}",
            p.tag, d.max_abs, d.mean_abs, d.rel_l2, d.cosine
        );
        assert!(
            d.rel_l2 < 1e-3,
            "{}: embd relL2 {} over the stem gate",
            p.tag,
            d.rel_l2
        );

        // layer 0 attention half (pre-residual) vs the reference's self_attn
        // hook - splits the first layer's two halves
        {
            let want = raster_to_merged(
                &read_f32(&dir.join(format!("attn0_{}.bin", p.tag))),
                gw,
                gh,
                1152,
            );
            let d = diff(&taps.attn0, &want);
            eprintln!(
                "{}: attn0 full  max|Δ| {:.4}  relL2 {:.5}  cos {:.6}",
                p.tag, d.max_abs, d.rel_l2, d.cosine
            );
            // one attention half deep: measured 1.2e-4..1.6e-4 (f16-plane class)
            assert!(
                d.rel_l2 < 2e-3,
                "{}: attn0 relL2 {} over the gate",
                p.tag,
                d.rel_l2
            );
        }

        // early-depth full-tensor anchors (layers 0/3/9): sharp because the
        // divergence hasn't compounded yet - a graph bug is a step here, a
        // numeric-class difference is a smooth ramp
        for li in [0usize, 3, 9] {
            let want = raster_to_merged(
                &read_f32(&dir.join(format!("layer{li}_{}.bin", p.tag))),
                gw,
                gh,
                1152,
            );
            let d = diff(&taps.layers[&li], &want);
            eprintln!(
                "{}: layer{li} full  max|Δ| {:.4}  relL2 {:.5}  cos {:.6}",
                p.tag, d.max_abs, d.rel_l2, d.cosine
            );
            // measured 1.8e-4 (layer 0) to 5.4e-4 (layer 9); ~10x headroom
            assert!(
                d.rel_l2 < 5e-3,
                "{}: layer{li} relL2 {} over the gate",
                p.tag,
                d.rel_l2
            );
        }

        // layer-level bisect: our per-layer sums vs the reference hooks'
        for (li, ours) in taps.layer_sums.iter().enumerate() {
            let want = info["stage_sums"][format!("layer{li}")].as_f64().unwrap();
            let rel = ((ours - want) / want.abs().max(1.0)).abs();
            if rel > 1e-2 || li == 0 || li + 1 == taps.layer_sums.len() {
                eprintln!(
                    "{}: layer{li:02} sum ours {ours:.1} ref {want:.1} rel {rel:.2e} (attn-half {:.1})",
                    p.tag, taps.attn_sums[li]
                );
            }
        }

        // stage 2: post-post_ln tower output, 27 layers deep - class gate
        let want_vit = raster_to_merged(
            &read_f32(&dir.join(format!("vit_{}.bin", p.tag))),
            gw,
            gh,
            1152,
        );
        let d = diff(&taps.vit, &want_vit);
        eprintln!(
            "{}: vit   max|Δ| {:.4}  mean|Δ| {:.5}  relL2 {:.5}  cos {:.6}",
            p.tag, d.max_abs, d.mean_abs, d.rel_l2, d.cosine
        );
        assert!(
            d.rel_l2 < 0.01,
            "{}: vit relL2 {} over the class gate",
            p.tag,
            d.rel_l2
        );
        assert!(
            d.cosine > 0.999,
            "{}: vit cosine {} under the class gate",
            p.tag,
            d.cosine
        );

        // stage 3: projector output - what the decoder actually eats
        let got = exec.to_host(&out.embd).expect("readback");
        let want_proj = read_f32(&dir.join(format!("proj_{}.bin", p.tag)));
        assert_eq!(got.len(), want_proj.len(), "{}: proj element count", p.tag);
        let d = diff(&got, &want_proj);
        eprintln!(
            "{}: proj  max|Δ| {:.4}  mean|Δ| {:.5}  relL2 {:.5}  cos {:.6}",
            p.tag, d.max_abs, d.mean_abs, d.rel_l2, d.cosine
        );
        assert!(
            d.rel_l2 < 0.01,
            "{}: proj relL2 {} over the class gate",
            p.tag,
            d.rel_l2
        );
        assert!(
            d.cosine > 0.999,
            "{}: proj cosine {} under the class gate",
            p.tag,
            d.cosine
        );
    }
}

/// An image's rows must be BITWISE independent of what else shares the
/// encode batch (per-image attention windows) - the qwen35 tower contract,
/// re-gated here because this family will serve batched pages.
#[test]
fn batched_encode_matches_serial() {
    let Some(path) = common::model("PADDLEOCR_VL_MMPROJ", MMPROJ) else {
        return;
    };
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    let map = MappedGguf::open(&path).expect("open mmproj");
    let mut vm = VisionModel::load(Arc::clone(&exec), &map).expect("load vision tower");

    // two different same-size images (the batch contract groups by size)
    let (w, h) = (336usize, 280usize);
    let a = preprocess::normalize_rgb(&preprocess::hash_pixels(w, h, 11), w, h);
    let b = preprocess::normalize_rgb(&preprocess::hash_pixels(w, h, 22), w, h);

    let serial_a = exec
        .to_host(&vm.encode(&a, w, h).expect("serial a").embd)
        .unwrap();
    let serial_b = exec
        .to_host(&vm.encode(&b, w, h).expect("serial b").embd)
        .unwrap();
    let batch = vm.encode_batch(&[(&a, w, h), (&b, w, h)]).expect("batch");
    let got_a = exec.to_host(&batch[0].embd).unwrap();
    let got_b = exec.to_host(&batch[1].embd).unwrap();

    // batched GEMMs reduce in a different order at 2n rows (cuBLAS kernel
    // election) - same class note as the qwen35 tower - so this is a
    // tolerance check, not bit-identity; the per-image ATTENTION window is
    // what's really under test and a window bug is a gross error.
    let d_a = diff(&got_a, &serial_a);
    let d_b = diff(&got_b, &serial_b);
    eprintln!(
        "batch-vs-serial relL2: a {:.2e}  b {:.2e}",
        d_a.rel_l2, d_b.rel_l2
    );
    assert!(
        d_a.rel_l2 < 5e-3,
        "image a diverges batched: {}",
        d_a.rel_l2
    );
    assert!(
        d_b.rel_l2 < 5e-3,
        "image b diverges batched: {}",
        d_b.rel_l2
    );
}

/// Direct semantics check of the `pd_mrope_vision` kernel against the
/// reference formula (`SigLIPRotaryEmbedding` + `apply_rotary_pos_emb_vision`
/// in modeling_paddleocr_vl.py): pair p < 18 rotates by the ROW position at
/// freq exponent p, pair 18..36 by the COLUMN at p-18, NeoX pair (p, p+36).
/// The kernel's only other consumer (qwen35 tower) has no numeric gate, so
/// this is the op's first hard anchor.
#[test]
fn mrope_vision_kernel_matches_reference_math() {
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    let (rows, n_heads, head_dim) = (3usize, 2usize, 72usize);
    let half = head_dim / 2;
    let sect = head_dim / 4;
    let ys = [0u32, 3, 7];
    let xs = [0u32, 5, 2];
    // axis-major [4, rows] = [y, x, y, x] - the encode fill
    let mut pos = vec![0u32; 4 * rows];
    for r in 0..rows {
        pos[r] = ys[r];
        pos[rows + r] = xs[r];
        pos[2 * rows + r] = ys[r];
        pos[3 * rows + r] = xs[r];
    }
    let n = rows * n_heads * head_dim;
    let x: Vec<f32> = (0..n)
        .map(|i| ((i * 37 % 101) as f32 - 50.0) / 25.0)
        .collect();

    let mut d_x = exec.to_device(&x).unwrap();
    let d_pos = exec.to_device_u32(&pos).unwrap();
    let theta_scale = 10000f32.powf(-2.0 / half as f32);
    exec.mrope_vision(&mut d_x, &d_pos, rows, n_heads, head_dim, theta_scale)
        .expect("mrope_vision");
    let got = exec.to_host(&d_x).unwrap();

    let mut want = x.clone();
    for r in 0..rows {
        for h in 0..n_heads {
            let base = (r * n_heads + h) * head_dim;
            for p in 0..half {
                let posv = if p < sect { ys[r] } else { xs[r] } as f32;
                let exp = if p < sect { p } else { p - sect } as u32;
                // the kernel reconstructs theta by repeated multiply - do the
                // same so the comparison is bit-tight, not just close
                let mut theta = posv;
                for _ in 0..exp {
                    theta *= theta_scale;
                }
                let (sn, cs) = (theta.sin(), theta.cos());
                let a = x[base + p];
                let b = x[base + p + half];
                want[base + p] = a * cs - b * sn;
                want[base + p + half] = a * sn + b * cs;
            }
        }
    }
    let worst = got
        .iter()
        .zip(&want)
        .map(|(g, w)| (g - w).abs())
        .fold(0f32, f32::max);
    assert!(
        worst < 1e-5,
        "mrope_vision diverges from the reference formula: max|d| {worst}"
    );
}

/// `vision_attn_at` vs host softmax attention. The op's gated consumers
/// (whisper, qwen3-asr) run head_dim 64; this tower runs 72 - check both so
/// a shape-conditional kernel bug shows up as a 64-pass/72-fail split.
#[test]
fn vision_attn_matches_host_softmax_at_both_head_dims() {
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    for head_dim in [64usize, 72] {
        let (n, n_heads) = (37usize, 2usize);
        let width = n_heads * head_dim;
        let mk = |seed: usize| -> Vec<f32> {
            (0..n * width)
                .map(|i| (((i * 131 + seed * 977) % 211) as f32 - 105.0) / 70.0)
                .collect()
        };
        let (q, k, v) = (mk(1), mk(2), mk(3));
        let d_q = exec.to_device(&q).unwrap();
        let d_k = exec.to_device(&k).unwrap();
        let d_v = exec.to_device(&v).unwrap();
        let mut d_o = exec.alloc(n * width).unwrap();
        let scale = 1.0 / (head_dim as f32).sqrt();
        exec.vision_attn_at(&d_q, &d_k, &d_v, &mut d_o, 0, n, n_heads, head_dim, scale)
            .expect("vision_attn_at");
        let got = exec.to_host(&d_o).unwrap();

        let mut worst = 0f32;
        for h in 0..n_heads {
            for i in 0..n {
                let qi = &q[(i * n_heads + h) * head_dim..][..head_dim];
                let mut logits = vec![0f32; n];
                for j in 0..n {
                    let kj = &k[(j * n_heads + h) * head_dim..][..head_dim];
                    logits[j] = qi.iter().zip(kj).map(|(a, b)| a * b).sum::<f32>() * scale;
                }
                let m = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exps: Vec<f32> = logits.iter().map(|l| (l - m).exp()).collect();
                let denom: f32 = exps.iter().sum();
                for d in 0..head_dim {
                    let mut acc = 0f32;
                    for j in 0..n {
                        acc += exps[j] * v[(j * n_heads + h) * head_dim + d];
                    }
                    let want = acc / denom;
                    let gotv = got[(i * n_heads + h) * head_dim + d];
                    worst = worst.max((gotv - want).abs());
                }
            }
        }
        eprintln!("vision_attn hd={head_dim}: max|d| {worst:.2e}");
        // the kernel stages through f16 (its gated consumers pass transcripts
        // with exactly this class) - 5e-4 measured at hd 64; a real
        // shape-conditional bug is orders worse
        assert!(
            worst < 2e-3,
            "vision_attn diverges at head_dim {head_dim}: max|d| {worst}"
        );
    }
}

/// `vision_attn_at` at this family's real sequence lengths. The op's gated
/// consumers top out at n=1500 (whisper); these grids run 1800-4928 rows, so
/// probe past every plausible internal tile boundary with spot-row host
/// references (full host attention at n=4928 would take minutes in debug).
#[test]
fn vision_attn_holds_at_large_n() {
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    let (n_heads, head_dim) = (2usize, 72usize);
    let width = n_heads * head_dim;
    for n in [1500usize, 1800, 2048, 4928] {
        let mk = |seed: usize| -> Vec<f32> {
            (0..n * width)
                .map(|i| (((i * 131 + seed * 977) % 211) as f32 - 105.0) / 70.0)
                .collect()
        };
        let (q, k, v) = (mk(1), mk(2), mk(3));
        let d_q = exec.to_device(&q).unwrap();
        let d_k = exec.to_device(&k).unwrap();
        let d_v = exec.to_device(&v).unwrap();
        let mut d_o = exec.alloc(n * width).unwrap();
        let scale = 1.0 / (head_dim as f32).sqrt();
        exec.vision_attn_at(&d_q, &d_k, &d_v, &mut d_o, 0, n, n_heads, head_dim, scale)
            .expect("vision_attn_at");
        let got = exec.to_host(&d_o).unwrap();

        let mut worst = 0f32;
        // spot rows: ends, middle, and both sides of common tile sizes
        for &i in &[0usize, 1, 511, 512, 1023, 1024, 1499, n / 2, n - 2, n - 1] {
            let i = i.min(n - 1);
            for h in 0..n_heads {
                let qi = &q[(i * n_heads + h) * head_dim..][..head_dim];
                let mut logits = vec![0f32; n];
                for j in 0..n {
                    let kj = &k[(j * n_heads + h) * head_dim..][..head_dim];
                    logits[j] = qi.iter().zip(kj).map(|(a, b)| a * b).sum::<f32>() * scale;
                }
                let m = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exps: Vec<f32> = logits.iter().map(|l| (l - m).exp()).collect();
                let denom: f32 = exps.iter().sum();
                for d in 0..head_dim {
                    let mut acc = 0f32;
                    for j in 0..n {
                        acc += exps[j] * v[(j * n_heads + h) * head_dim + d];
                    }
                    let want = acc / denom;
                    let gotv = got[(i * n_heads + h) * head_dim + d];
                    worst = worst.max((gotv - want).abs());
                }
            }
        }
        eprintln!("vision_attn n={n}: spot max|d| {worst:.2e}");
        assert!(
            worst < 2e-3,
            "vision_attn diverges at n={n}: max|d| {worst}"
        );
    }
}
