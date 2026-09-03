//! Op-parity gate for the qwen4_exp (Qwen3.8-Flash-Next) kernel family,
//! pack slots 506-516 - every new op against `reference::qwen4exp`, the
//! host reference that already holds ARBITER parity end to end
//! (stage 2).
//!
//! Synthetic weights throughout: these gates test the KERNELS, not the
//! checkpoint (that is `gpu_qwen4exp_load.rs`), so they need no model dir and
//! run in milliseconds. Output buffers are poisoned with NaN so an unwritten
//! element fails loudly instead of matching a zeroed reference.

mod common;

use paddock_engine::gpu::GpuExecutor;
use paddock_kernels::reference::qwen4exp as rq;

/// The house deterministic input LCG.
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

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "length mismatch");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// Executor + the family gate, the pair every test here opens with.
fn exec_with_family() -> Option<GpuExecutor> {
    let exec = common::gpu()?;
    if !exec.has_qwen4exp_ops() {
        common::missing("pack has no qwen4exp kernels (rebuild packs/cuda)");
        return None;
    }
    Some(exec)
}

const EPS: f32 = 1e-6;

#[test]
fn group_norm_1p_matches_reference() {
    let Some(exec) = exec_with_family() else {
        return;
    };
    let (rows, hc, hidden) = (3usize, 4usize, 2560usize);
    let hw = hc * hidden;

    let x = det(rows * hw, 11);
    let w = det(hw, 12);

    let mut want = x.clone();
    for r in 0..rows {
        rq::group_rms_norm_1p(&mut want[r * hw..(r + 1) * hw], &w, hc, EPS);
    }

    let d_x = exec.to_device(&x).expect("x");
    let d_w = exec.to_device(&w).expect("w");
    let mut d_out = exec.to_device(&vec![f32::NAN; rows * hw]).expect("out");
    exec.q4x_group_norm_1p(&d_x, &d_w, &mut d_out, rows, hc, hidden, EPS)
        .expect("q4x_group_norm_1p");
    let got = exec.to_host(&d_out).expect("dtoh");

    let diff = max_abs_diff(&got, &want);
    eprintln!("group_norm_1p parity: max_abs_diff {diff:.2e}");
    // The whole difference is the 2560-wide sum-of-squares landing in a
    // different order (block tree vs the reference's ascending loop): one
    // shared `1/sqrt(ms+eps)` per group, so the deviation is UNIFORM across
    // the group and ~5 ulps of the output magnitude. 1e-6 was under that
    // floor and measured 1.67e-6 on the first run.
    assert!(diff < 1e-5, "group_norm_1p max_abs_diff {diff} too high");
}

/// The whole hyper-connection mix, composed the way the forward graph will:
/// grouped (1+w) norm -> low-rank down -> scale+silu -> up -> gated reduce,
/// plus the raw inject logits. Compared against the reference's `hc_mix`.
#[test]
fn hc_mix_matches_reference() {
    let Some(exec) = exec_with_family() else {
        return;
    };
    let (rows, hc, hidden, lowrank) = (3usize, 4usize, 320usize, 80usize);
    let hw = hc * hidden;

    let h = det(rows * hw, 21);
    let norm_w = det(hw, 22);
    // reference matvec convention is row-major [out, in] - the same layout
    // matvec_f32_raw reads (y = x·Wᵀ per row).
    let w_down = det(lowrank * hw, 23);
    let w_up = det(hw * lowrank, 24);
    let w_inj = det(hc * hw, 25);

    let (mut want_bi, mut want_inj) = (Vec::new(), Vec::new());
    for r in 0..rows {
        let (bi, inj) = rq::hc_mix(
            &h[r * hw..(r + 1) * hw],
            &norm_w,
            &w_down,
            &w_up,
            Some(&w_inj),
            hc,
            lowrank,
            EPS,
        );
        want_bi.extend(bi);
        want_inj.extend(inj.expect("inject"));
    }

    let d_h = exec.to_device(&h).expect("h");
    let d_norm = exec.to_device(&norm_w).expect("norm");
    let d_down = exec.to_device(&w_down).expect("down");
    let d_up = exec.to_device(&w_up).expect("up");
    let d_inj_w = exec.to_device(&w_inj).expect("inj w");
    let mut d_xn = exec.to_device(&vec![f32::NAN; rows * hw]).expect("xn");
    let mut d_m = exec.to_device(&vec![f32::NAN; rows * lowrank]).expect("m");
    let mut d_gate = exec.to_device(&vec![f32::NAN; rows * hw]).expect("gate");
    let mut d_bi = exec.to_device(&vec![f32::NAN; rows * hidden]).expect("bi");
    let mut d_inj = exec.to_device(&vec![f32::NAN; rows * hc]).expect("inj");

    exec.q4x_group_norm_1p(&d_h, &d_norm, &mut d_xn, rows, hc, hidden, EPS)
        .expect("norm");
    exec.matvec_f32_raw(&d_down, hw, lowrank, &d_xn, &mut d_m, rows)
        .expect("down");
    exec.q4x_scale_silu(&mut d_m, rows * lowrank, 1.0 / hc as f32)
        .expect("scale_silu");
    exec.matvec_f32_raw(&d_up, lowrank, hw, &d_m, &mut d_gate, rows)
        .expect("up");
    exec.q4x_hc_mix(&d_xn, &d_gate, &mut d_bi, rows, hc, hidden)
        .expect("hc_mix");
    exec.matvec_f32_raw(&d_inj_w, hw, hc, &d_xn, &mut d_inj, rows)
        .expect("inject");

    let got_bi = exec.to_host(&d_bi).expect("dtoh bi");
    let got_inj = exec.to_host(&d_inj).expect("dtoh inj");
    let (db, di) = (
        max_abs_diff(&got_bi, &want_bi),
        max_abs_diff(&got_inj, &want_inj),
    );
    eprintln!("hc_mix parity: block_input {db:.2e}, inject {di:.2e}");
    assert!(db < 1e-5, "hc_mix block_input max_abs_diff {db} too high");
    assert!(di < 1e-4, "hc_mix inject max_abs_diff {di} too high");
}

#[test]
fn hc_combine_matches_reference() {
    let Some(exec) = exec_with_family() else {
        return;
    };
    let (rows, hc, hidden) = (3usize, 4usize, 640usize);
    let hw = hc * hidden;

    let h = det(rows * hw, 31);
    let block_out = det(rows * hidden, 32);
    let inj = det(rows * hc, 33);

    let mut want = h.clone();
    for r in 0..rows {
        rq::hc_combine(
            &mut want[r * hw..(r + 1) * hw],
            &block_out[r * hidden..(r + 1) * hidden],
            &inj[r * hc..(r + 1) * hc],
            hc,
        );
    }

    let mut d_h = exec.to_device(&h).expect("h");
    let d_b = exec.to_device(&block_out).expect("block_out");
    let d_i = exec.to_device(&inj).expect("inj");
    exec.q4x_hc_combine(&mut d_h, &d_b, &d_i, rows, hc, hidden)
        .expect("q4x_hc_combine");
    let got = exec.to_host(&d_h).expect("dtoh");

    let diff = max_abs_diff(&got, &want);
    eprintln!("hc_combine parity: max_abs_diff {diff:.2e}");
    assert!(diff < 1e-6, "hc_combine max_abs_diff {diff} too high");
}

/// PLE gate: two grouped (1+w) norms then the signed-sqrt scaled dot.
#[test]
fn ple_gate_matches_reference() {
    let Some(exec) = exec_with_family() else {
        return;
    };
    let (rows, hc, hidden) = (4usize, 4usize, 2560usize);
    let hw = hc * hidden;

    let h = det(rows * hw, 41);
    let key = det(rows * hw, 42);
    let value = det(rows * hidden, 43);
    let norm_key = det(hw, 44);
    let norm_query = det(hw, 45);

    let mut want = Vec::with_capacity(rows * hw);
    for r in 0..rows {
        want.extend(rq::ple_gate(
            &h[r * hw..(r + 1) * hw],
            &key[r * hw..(r + 1) * hw],
            &value[r * hidden..(r + 1) * hidden],
            &norm_key,
            &norm_query,
            hc,
            EPS,
        ));
    }

    let d_h = exec.to_device(&h).expect("h");
    let d_key = exec.to_device(&key).expect("key");
    let d_val = exec.to_device(&value).expect("value");
    let d_nk = exec.to_device(&norm_key).expect("nk");
    let d_nq = exec.to_device(&norm_query).expect("nq");
    let mut d_kn = exec.to_device(&vec![f32::NAN; rows * hw]).expect("kn");
    let mut d_qn = exec.to_device(&vec![f32::NAN; rows * hw]).expect("qn");
    let mut d_gv = exec.to_device(&vec![f32::NAN; rows * hw]).expect("gv");

    exec.q4x_group_norm_1p(&d_key, &d_nk, &mut d_kn, rows, hc, hidden, EPS)
        .expect("key norm");
    exec.q4x_group_norm_1p(&d_h, &d_nq, &mut d_qn, rows, hc, hidden, EPS)
        .expect("query norm");
    exec.q4x_ple_gate(&d_kn, &d_qn, &d_val, &mut d_gv, rows, hc, hidden)
        .expect("q4x_ple_gate");
    let got = exec.to_host(&d_gv).expect("dtoh");

    let diff = max_abs_diff(&got, &want);
    eprintln!("ple_gate parity: max_abs_diff {diff:.2e}");
    assert!(diff < 1e-5, "ple_gate max_abs_diff {diff} too high");
}

/// The PLE conv: k=4 at dilation 3 (a nine-token ring). The pack had no
/// dilated conv at all before this family, so there is no twin to diff - the
/// reference is the gate.
#[test]
fn conv_dil_matches_reference() {
    let Some(exec) = exec_with_family() else {
        return;
    };
    let (n, dim, k, dil) = (9usize, 640usize, 4usize, 3usize);

    let src = det(n * dim, 51);
    let w = det(dim * k, 52);

    let mut want = src.clone();
    rq::conv1d_causal_silu(&mut want, &w, n, dim, k, dil);

    let d_src = exec.to_device(&src).expect("src");
    let d_w = exec.to_device(&w).expect("w");
    let mut d_out = exec.to_device(&vec![f32::NAN; n * dim]).expect("out");
    exec.q4x_conv_dil(&d_src, &d_w, &mut d_out, n, dim, k, dil)
        .expect("q4x_conv_dil");
    let got = exec.to_host(&d_out).expect("dtoh");

    let diff = max_abs_diff(&got, &want);
    eprintln!("conv_dil parity: max_abs_diff {diff:.2e}");
    assert!(diff < 1e-5, "conv_dil max_abs_diff {diff} too high");
}

/// The windowed one-token twin must reproduce the sequence form's last row
/// bit for bit - that equality is what lets decode continue a prefill.
#[test]
fn conv_dil_step_matches_sequence() {
    let Some(exec) = exec_with_family() else {
        return;
    };
    let (n, dim, k, dil) = (12usize, 320usize, 4usize, 3usize);
    let wrows = (k - 1) * dil;

    let src = det(n * dim, 61);
    let w = det(dim * k, 62);

    // the host reference, for the numeric bound
    let mut host = src.clone();
    rq::conv1d_causal_silu(&mut host, &w, n, dim, k, dil);
    let host_last = &host[(n - 1) * dim..];

    let d_src = exec.to_device(&src).expect("src");
    let d_w = exec.to_device(&w).expect("w");
    let mut d_seq = exec.to_device(&vec![f32::NAN; n * dim]).expect("seq");
    exec.q4x_conv_dil(&d_src, &d_w, &mut d_seq, n, dim, k, dil)
        .expect("q4x_conv_dil");
    let seq = exec.to_host(&d_seq).expect("dtoh seq");
    let seq_last = &seq[(n - 1) * dim..];

    // window = the wrows pre-conv rows before the last token, oldest first
    let win: Vec<f32> = src[(n - 1 - wrows) * dim..(n - 1) * dim].to_vec();
    let d_x = exec.to_device(&src[(n - 1) * dim..]).expect("x");
    let d_win = exec.to_device(&win).expect("win");
    let mut d_out = exec.to_device(&vec![f32::NAN; dim]).expect("out");
    exec.q4x_conv_dil_step(&d_x, &d_win, &d_w, &mut d_out, dim, k, dil)
        .expect("q4x_conv_dil_step");
    let got = exec.to_host(&d_out).expect("dtoh");

    // The load-bearing invariant is GPU-vs-GPU: a decode step off a carried
    // window must reproduce what the prefill pass wrote for that same token,
    // to the bit - otherwise the handoff introduces drift no gate would see.
    // (Against the host the two forms agree only to ~1 ulp: nvcc contracts
    // `acc += w*v` into an FMA and the reference does not.)
    assert_eq!(
        got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        seq_last.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "windowed step is not bit-identical to the sequence form's last row"
    );
    let hdiff = max_abs_diff(&got, host_last);
    eprintln!(
        "conv_dil_step: bit-identical to the sequence form ({dim} channels);          vs host max_abs_diff {hdiff:.2e}"
    );
    assert!(
        hdiff < 1e-5,
        "conv_dil_step vs host max_abs_diff {hdiff} too high"
    );
}

#[test]
fn gdn_gated_norm_matches_reference() {
    let Some(exec) = exec_with_family() else {
        return;
    };
    // rows = tokens * v_heads, d = head_dim (the real geometry: 48 heads x 128)
    let (rows, d) = (3usize * 48, 128usize);

    let x = det(rows * d, 71);
    let z = det(rows * d, 72);
    let w = det(d, 73);

    let mut want = x.clone();
    rq::gdn_gated_norm(&mut want, &z, &w, d, EPS);

    let d_x = exec.to_device(&x).expect("x");
    let d_z = exec.to_device(&z).expect("z");
    let d_w = exec.to_device(&w).expect("w");
    let mut d_out = exec.to_device(&vec![f32::NAN; rows * d]).expect("out");
    exec.q4x_gdn_gated_norm(&d_x, &d_z, &d_w, &mut d_out, rows, d, EPS)
        .expect("q4x_gdn_gated_norm");
    let got = exec.to_host(&d_out).expect("dtoh");

    let diff = max_abs_diff(&got, &want);
    eprintln!("gdn_gated_norm parity: max_abs_diff {diff:.2e}");
    assert!(diff < 1e-6, "gdn_gated_norm max_abs_diff {diff} too high");
}

/// The widening law: raw safetensors GDN planes serve value head `vh` from
/// key head `vh / (hv/hk)` (repeat_interleave). Getting this wrong is not a
/// small numeric error - it silently reads a different head, which is exactly
/// the bug that cost stage 2 a debugging pass.
#[test]
fn gdn_split_widen_is_repeat_interleave() {
    let Some(exec) = exec_with_family() else {
        return;
    };
    // the real GDN geometry: 16 key heads -> 48 value heads, 128-wide
    let (rows, hk, hv, kd, vd) = (3usize, 16usize, 48usize, 128usize, 128usize);
    let qkv = 2 * hk * kd + hv * vd;

    let conv = det(rows * qkv, 81);

    // host oracle, mirroring examples/q38fn_host_forward.rs
    let (mut want_q, mut want_k, mut want_v) = (
        vec![0f32; rows * hv * kd],
        vec![0f32; rows * hv * kd],
        vec![0f32; rows * hv * vd],
    );
    for t in 0..rows {
        let row = &conv[t * qkv..(t + 1) * qkv];
        for vh in 0..hv {
            let kh = vh / (hv / hk);
            let o = (t * hv + vh) * kd;
            want_q[o..o + kd].copy_from_slice(&row[kh * kd..(kh + 1) * kd]);
            want_k[o..o + kd].copy_from_slice(&row[hk * kd + kh * kd..hk * kd + (kh + 1) * kd]);
        }
        want_v[t * hv * vd..(t + 1) * hv * vd]
            .copy_from_slice(&row[2 * hk * kd..2 * hk * kd + hv * vd]);
    }

    let d_conv = exec.to_device(&conv).expect("conv");
    let mut d_q = exec.to_device(&vec![f32::NAN; rows * hv * kd]).expect("q");
    let mut d_k = exec.to_device(&vec![f32::NAN; rows * hv * kd]).expect("k");
    let mut d_v = exec.to_device(&vec![f32::NAN; rows * hv * vd]).expect("v");
    exec.q4x_gdn_split_widen(&d_conv, &mut d_q, &mut d_k, &mut d_v, rows, hk, hv, kd, vd)
        .expect("q4x_gdn_split_widen");

    for (name, dev, want) in [
        ("q", &d_q, &want_q),
        ("k", &d_k, &want_k),
        ("v", &d_v, &want_v),
    ] {
        let got = exec.to_host(dev).expect("dtoh");
        assert_eq!(
            got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            want.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "{name} is not a bit-exact repeat_interleave gather"
        );
    }
    eprintln!("gdn_split_widen: bit-exact repeat_interleave over {hk}->{hv} heads");
}

#[test]
fn add_gated_row_matches_reference() {
    let Some(exec) = exec_with_family() else {
        return;
    };
    let (rows, n) = (5usize, 2560usize);

    let y0 = det(rows * n, 91);
    let x = det(rows * n, 92);
    let s = det(rows, 93);

    let mut want = y0.clone();
    for r in 0..rows {
        let g = rq::sigmoid(s[r]);
        for i in 0..n {
            want[r * n + i] += x[r * n + i] * g;
        }
    }

    let mut d_y = exec.to_device(&y0).expect("y");
    let d_x = exec.to_device(&x).expect("x");
    let d_s = exec.to_device(&s).expect("s");
    exec.q4x_add_gated_row(&mut d_y, &d_x, &d_s, rows, n)
        .expect("q4x_add_gated_row");
    let got = exec.to_host(&d_y).expect("dtoh");

    let diff = max_abs_diff(&got, &want);
    eprintln!("add_gated_row parity: max_abs_diff {diff:.2e}");
    assert!(diff < 1e-6, "add_gated_row max_abs_diff {diff} too high");
}

/// NVFP4 e2m1 code -> value. The checkpoint's own code book (0, 0.5, 1, 1.5,
/// 2, 3, 4, 6 with a sign bit) - the same one `pd_nvf4_dot4w`'s byte-perm
/// table spells as e4m3 constants.
fn nvf4_code(nib: u8) -> f32 {
    const MAG: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let v = MAG[(nib & 0x07) as usize];
    if nib & 0x08 != 0 { -v } else { v }
}

/// The MoE gate+up swiglu over real NVFP4 planes: synthetic nibbles and
/// per-16 e4m3 scales uploaded through `nvf4_moe_upload`, checked against a
/// host dequant-and-dot. This kernel is the one place where a wrong nibble or
/// scale stride would be silent - the outputs stay plausible.
#[test]
fn moe_gu_swiglu_matches_host_dequant() {
    let Some(exec) = exec_with_family() else {
        return;
    };
    // the real MoE shape, trimmed in expert count: in_dim = hidden 2560,
    // ff = moe_ff 640, k = 10 picks
    let (n_expert, ff, in_dim, k, batch) = (8usize, 640usize, 2560usize, 4usize, 3usize);
    let rows = n_expert * ff;

    // deterministic nibble/scale bytes - every code and both nibble halves get
    // exercised by construction
    let bytes = |seed: u64, len: usize| -> Vec<u8> {
        let mut s = seed;
        (0..len)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (s >> 40) as u8
            })
            .collect()
    };
    let gp = bytes(101, rows * in_dim / 2);
    let up = bytes(102, rows * in_dim / 2);
    // scale bytes: keep the e4m3 exponent modest so products stay in range
    let scale_bytes = |seed: u64, len: usize| -> Vec<u8> {
        bytes(seed, len).iter().map(|b| 0x30 | (b & 0x07)).collect()
    };
    let gs = scale_bytes(103, rows * in_dim / 16);
    let us = scale_bytes(104, rows * in_dim / 16);
    let g2: Vec<f32> = (0..n_expert).map(|e| 0.5 + 0.1 * e as f32).collect();
    let u2: Vec<f32> = (0..n_expert).map(|e| 0.7 - 0.03 * e as f32).collect();

    let gate = exec
        .nvf4_moe_upload(&gp, &gs, &g2, n_expert, ff, in_dim)
        .expect("gate plane");
    let upp = exec
        .nvf4_moe_upload(&up, &us, &u2, n_expert, ff, in_dim)
        .expect("up plane");

    let x = det(batch * in_dim, 105);
    let idx: Vec<u32> = (0..batch * k)
        .map(|i| ((i * 3) % n_expert) as u32)
        .collect();

    // host oracle: dequantize the selected rows and dot, then swiglu
    let dot = |packed: &[u8], scales: &[u8], sc2: f32, e: usize, r: usize, xr: &[f32]| -> f32 {
        let pb = &packed[(e * ff + r) * in_dim / 2..][..in_dim / 2];
        let sb = &scales[(e * ff + r) * in_dim / 16..][..in_dim / 16];
        let mut acc = 0f32;
        for i in 0..in_dim {
            let byte = pb[i / 2];
            let nib = if i % 2 == 0 { byte & 0x0F } else { byte >> 4 };
            acc += nvf4_code(nib) * rq::e4m3_to_f32(sb[i / 16]) * xr[i];
        }
        acc * sc2
    };
    let mut want = vec![0f32; batch * k * ff];
    for slot in 0..batch * k {
        let e = idx[slot] as usize;
        let xr = &x[(slot / k) * in_dim..(slot / k + 1) * in_dim];
        for r in 0..ff {
            let g = dot(&gp, &gs, g2[e], e, r, xr);
            let u = dot(&up, &us, u2[e], e, r, xr);
            want[slot * ff + r] = g * rq::sigmoid(g) * u;
        }
    }

    let d_idx = exec.to_device_u32(&idx).expect("idx");
    let d_x = exec.to_device(&x).expect("x");
    let mut d_y = exec.to_device(&vec![f32::NAN; batch * k * ff]).expect("y");
    exec.q4x_moe_gu_swiglu(&gate, &upp, &d_idx, &d_x, &mut d_y, k, batch)
        .expect("q4x_moe_gu_swiglu");
    let got = exec.to_host(&d_y).expect("dtoh");

    // relative: the dot is a 2560-term f32 reduction in a different order
    let scale = want.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let diff = max_abs_diff(&got, &want);
    eprintln!(
        "moe_gu_swiglu parity: max_abs_diff {diff:.2e} over a {scale:.2e} range \
         (rel {:.2e})",
        diff / scale
    );
    assert!(
        diff / scale < 1e-4,
        "moe_gu_swiglu relative deviation {} too high",
        diff / scale
    );
}

/// The router must consider every expert. `pd_moe_topk_warp` held each lane's
/// logits in `float v[8]` = 8 x 32 = 256 experts, so on this family's 512 the
/// second half of the expert set was never eligible - silently, because the
/// truncated routing still produces a plausible model. This gate puts the
/// winning logits in the upper half deliberately.
#[test]
fn moe_topk_batch_covers_every_expert() {
    let Some(exec) = exec_with_family() else {
        return;
    };
    let (batch, n_expert, k) = (3usize, 512usize, 10usize);

    let mut logits = det(batch * n_expert, 121);
    // plant each token's clear winners above index 255
    for t in 0..batch {
        for (j, e) in [500usize, 480, 300, 256].iter().enumerate() {
            logits[t * n_expert + e] = 5.0 - j as f32 * 0.25;
        }
    }

    // host oracle: softmax over all experts, top-k, renormalized over the picks
    let mut want_idx = Vec::new();
    let mut want_w = Vec::new();
    for t in 0..batch {
        let row = &logits[t * n_expert..(t + 1) * n_expert];
        let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let p: Vec<f32> = row.iter().map(|v| (v - m).exp()).collect();
        let den: f32 = p.iter().sum();
        let probs: Vec<f32> = p.iter().map(|v| v / den).collect();
        let mut order: Vec<usize> = (0..n_expert).collect();
        order.sort_unstable_by(|&a, &b| probs[b].total_cmp(&probs[a]));
        let top = &order[..k];
        let wsum: f32 = top.iter().map(|&e| probs[e]).sum();
        want_idx.extend(top.iter().map(|&e| e as u32));
        want_w.extend(top.iter().map(|&e| probs[e] / wsum));
    }

    let d_logits = exec.to_device(&logits).expect("logits");
    let d_bias = exec.to_device(&vec![0f32; n_expert]).expect("bias");
    let mut d_idx = exec.to_device_u32(&vec![0u32; batch * k]).expect("idx");
    let mut d_w = exec.to_device(&vec![f32::NAN; batch * k]).expect("w");
    exec.moe_topk_batch(&d_logits, &d_bias, n_expert, k, &mut d_idx, &mut d_w, batch)
        .expect("moe_topk_batch");
    let got_idx = exec.to_host_u32(&d_idx).expect("dtoh idx");
    let got_w = exec.to_host(&d_w).expect("dtoh w");

    assert_eq!(got_idx, want_idx, "router picked a different expert set");
    let diff = max_abs_diff(&got_w, &want_w);
    eprintln!("moe_topk over {n_expert} experts: ids exact, weight max_abs_diff {diff:.2e}");
    assert!(diff < 1e-5, "router weights off by {diff}");
}
