//! Parity tests for the canonical spec rejection-sampling kernels:
//! `draft_rs` (sampled draft draw + fp16 q-store materialization) and
//! `spec_rs_resolve` (accept w.p. min(1, p/q) + residual recovery).
//!
//! Oracle strategy: the TOKEN contracts are replayed on host over the
//! DEVICE'S own stored q values (read back), with the kernel's exact chunk
//! structure (256 contiguous chunks, per-chunk partials summed in chunk
//! order) - so fp16 rounding never enters the oracle. Light (no model load).

mod common;

use paddock_engine::gpu::GpuExecutor;

const TPB: usize = 256;

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

fn f16_to_f32(h: u16) -> f32 {
    let e = (h >> 10) & 0x1f;
    let m = (h & 0x3ff) as f32;
    if e == 0 {
        m * 2f32.powi(-24)
    } else {
        (1024.0 + m) * 2f32.powi(e as i32 - 25)
    }
}

/// The kernels' quantile walk, mirrored: per-chunk partials (256 contiguous
/// chunks) in element order, chunk prefix scan in chunk order, owner walk.
/// `mass(i)` must be the exact per-element value the kernel uses.
fn host_owner_walk(n: usize, u_times_total: f32, mass: impl Fn(usize) -> f32) -> Option<u32> {
    let chunk = n.div_ceil(TPB);
    let csum: Vec<f32> = (0..TPB)
        .map(|t| {
            let (lo, hi) = (t * chunk, ((t + 1) * chunk).min(n));
            let mut s = 0.0f32;
            for i in lo..hi {
                s += mass(i);
            }
            s
        })
        .collect();
    let total: f32 = csum.iter().sum();
    // a NaN total must bail too, which `total <= 0.0` would not do
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    if !(total > 0.0) {
        return None;
    }
    let r = u_times_total;
    let mut pre = 0.0f32;
    let mut own = TPB;
    let mut rr = 0.0f32;
    for (t, &c) in csum.iter().enumerate() {
        if c > 0.0 && pre + c >= r {
            own = t;
            rr = r - pre;
            break;
        }
        pre += c;
    }
    if own == TPB {
        for t in (0..TPB).rev() {
            if csum[t] > 0.0 {
                own = t;
                rr = csum[t];
                break;
            }
        }
    }
    let (lo, hi) = (own * chunk, ((own + 1) * chunk).min(n));
    let mut last = None;
    for i in lo..hi {
        let e = mass(i);
        if e > 0.0 {
            last = Some(i as u32);
            rr -= e;
            if rr <= 0.0 {
                break;
            }
        }
    }
    last
}

/// Launch draft_rs once at a given device step value.
#[allow(clippy::too_many_arguments)]
fn run_draft(
    exec: &GpuExecutor,
    logits: &[f32],
    invt: &[f32],
    uplane: &[f32],
    step: u32,
    rows: usize,
    n: usize,
    rmax: usize,
    kmax: usize,
) -> (Vec<u32>, Vec<u16>, Vec<f32>) {
    let d_l = exec.to_device(logits).expect("logits");
    let d_it = exec.to_device(invt).expect("invt");
    let d_up = exec.to_device(uplane).expect("uplane");
    let d_step = exec.to_device_u32(&[step]).expect("step");
    let mut d_qs = exec
        .to_device_u16(&vec![0u16; kmax * rmax * n])
        .expect("qstore");
    let mut d_sum = exec.to_device(&vec![0.0f32; kmax * rmax]).expect("qsum");
    let mut d_tok = exec.to_device_u32(&vec![u32::MAX; rows]).expect("tok");
    exec.draft_rs(
        &d_l, &d_it, &d_up, &d_step, &mut d_qs, &mut d_sum, &mut d_tok, rows, n, rmax,
    )
    .expect("draft_rs");
    (
        exec.to_host_u32(&d_tok).expect("tok dtoh"),
        exec.to_host_u16(&d_qs).expect("qstore dtoh"),
        exec.to_host(&d_sum).expect("qsum dtoh"),
    )
}

fn exec() -> Option<GpuExecutor> {
    common::gpu()
}

#[test]
fn draft_greedy_rows_argmax_with_zero_marker() {
    let Some(exec) = exec() else { return };
    let (rows, n, rmax, kmax) = (4usize, 4096 + 7, 8usize, 2usize);
    let mut logits = det(rows * n, 11);
    logits[2 * n + 77] = 8.0;
    logits[2 * n + 2000] = 8.0; // tie -> lowest index
    let invt = vec![0.0f32; rows];
    let uplane = vec![0.5f32; kmax * rows];
    let (tok, _qs, qsum) = run_draft(&exec, &logits, &invt, &uplane, 1, rows, n, rmax, kmax);
    for r in 0..rows {
        let row = &logits[r * n..(r + 1) * n];
        let mut best = 0usize;
        for (i, v) in row.iter().enumerate() {
            if *v > row[best] {
                best = i;
            }
        }
        assert_eq!(tok[r], best as u32, "row {r}");
        assert_eq!(qsum[rmax + r], 0.0, "greedy marker row {r} (step 1)");
    }
    assert_eq!(tok[2], 77, "tie must pick the lowest index");
    eprintln!("draft_rs greedy: argmax + zero markers OK");
}

#[test]
fn draft_categorical_draw_matches_own_store() {
    let Some(exec) = exec() else { return };
    let (rows, n, rmax, kmax) = (6usize, 4096usize, 8usize, 3usize);
    let inv_t = 1.0f32 / 0.7;
    let logits: Vec<f32> = det(rows * n, 42).iter().map(|v| v * 6.0).collect();
    let invt = vec![inv_t; rows];
    let us: Vec<f32> = det(rows, 77).iter().map(|v| v + 0.5).collect();
    let step = 2u32;
    let mut uplane = vec![0.25f32; kmax * rows];
    for r in 0..rows {
        uplane[step as usize * rows + r] = us[r];
    }
    let (tok, qs, qsum) = run_draft(&exec, &logits, &invt, &uplane, step, rows, n, rmax, kmax);
    for r in 0..rows {
        let qrow = &qs[(step as usize * rmax + r) * n..(step as usize * rmax + r) * n + n];
        // S must equal the chunk-ordered sum of the STORED values
        let chunk = n.div_ceil(TPB);
        let s_host: f32 = (0..TPB)
            .map(|t| {
                let (lo, hi) = (t * chunk, ((t + 1) * chunk).min(n));
                let mut s = 0.0f32;
                for &q in &qrow[lo..hi] {
                    s += f16_to_f32(q);
                }
                s
            })
            .sum();
        let s_dev = qsum[step as usize * rmax + r];
        assert!(
            (s_dev - s_host).abs() <= s_host * 1e-6 + 1e-6,
            "row {r}: qsum {s_dev} vs host {s_host}"
        );
        // the draw must replay exactly over the device's own store
        let want =
            host_owner_walk(n, us[r] * s_dev, |i| f16_to_f32(qrow[i])).expect("store has mass");
        assert_eq!(tok[r], want, "row {r} draw");
        assert!(
            f16_to_f32(qrow[tok[r] as usize]) > 0.0,
            "drawn token has mass"
        );
    }
    eprintln!("draft_rs categorical: draw + store + sum replay OK on {rows} rows");
}

/// par layout: {vrow, jstep, srow, invt, u1, u2, 0, 0}
#[allow(clippy::too_many_arguments)]
fn run_resolve(
    exec: &GpuExecutor,
    logits: &[f32],
    drafts: &[u32],
    qstore: &[u16],
    qsum: &[f32],
    par: &[u32],
    out_len: usize,
    rr: usize,
    n: usize,
    rmax: usize,
) -> Vec<u32> {
    let d_l = exec.to_device(logits).expect("logits");
    let d_d = exec.to_device_u32(drafts).expect("drafts");
    let d_qs = exec.to_device_u16(qstore).expect("qstore");
    let d_sum = exec.to_device(qsum).expect("qsum");
    let d_par = exec.to_device_u32(par).expect("par");
    let mut d_out = exec.to_device_u32(&vec![u32::MAX; out_len]).expect("out");
    exec.spec_rs_resolve(
        &d_l,
        &d_d,
        &d_qs,
        &d_sum,
        &d_par,
        &mut d_out,
        par.len() / 8,
        rr,
        n,
        rmax,
    )
    .expect("resolve");
    exec.to_host_u32(&d_out).expect("out dtoh")
}

#[test]
fn resolve_accept_reject_and_residual() {
    let Some(exec) = exec() else { return };
    let (n, rr, rmax) = (4096usize, 2usize, 8usize);
    let inv_t = 1.0f32;
    // target row: two dominant tokens A=100 (large p) and B=200 (small p)
    let mut logits = vec![-20.0f32; 2 * n];
    logits[100] = 2.0; // A: p ~ .73 of the two-mass
    logits[200] = 1.0; // B: p ~ .27
    // verify row 1: same shape shifted - dominant C=300, draft elsewhere
    logits[n + 300] = 4.0;
    logits[n + 400] = 0.0;
    // q rows (fp16 exactly-representable): draft = B with q(B) = .75, q(A) = .25
    let mut qstore = vec![0u16; rmax * n]; // jstep 0 only
    let qsum_row = |a: f32, b: f32| a + b;
    qstore[100] = 0x3400; // row 0: 0.25
    qstore[200] = 0x3B00; // row 0: 0.875
    let mut qsum = vec![0.0f32; rmax];
    qsum[0] = qsum_row(0.25, 0.875);
    // srow 1 = greedy marker (qsum 0): point-q rule on verify row 1
    let drafts = vec![200u32, 400u32]; // jstep 0: srow0 drafts B, srow1 drafts 400
    // host p for row 0 (chunk-ordered Z)
    let chunk = n.div_ceil(TPB);
    let z0: f32 = (0..TPB)
        .map(|t| {
            let (lo, hi) = (t * chunk, ((t + 1) * chunk).min(n));
            let mut s = 0.0f32;
            let m = 2.0f32; // max logit row 0
            for &l in &logits[lo..hi] {
                s += ((l - m) * inv_t).exp();
            }
            s
        })
        .sum();
    let p_b = ((1.0f32 - 2.0) * inv_t).exp() / z0;
    let q_b = 0.875f32 / qsum[0];
    let thresh = p_b / q_b; // accept iff u1 < this
    // case 1: u1 just below threshold -> accept -> out = draft B
    let par_acc = vec![
        0u32,
        0,
        0,
        inv_t.to_bits(),
        (thresh * 0.9).to_bits(),
        0.5f32.to_bits(),
        0,
        0,
    ];
    let got = run_resolve(
        &exec, &logits, &drafts, &qstore, &qsum, &par_acc, 2, rr, n, rmax,
    );
    assert_eq!(got[0], 200, "accept must emit the draft");
    // case 2: u1 just above -> reject -> residual max(p-q, 0): A holds nearly
    // all residual mass (p(A)≈.73 vs q(A)≈.22; B has p<q -> 0)
    let par_rej = vec![
        0u32,
        0,
        0,
        inv_t.to_bits(),
        (thresh * 1.1).min(0.999).to_bits(),
        0.3f32.to_bits(),
        0,
        0,
    ];
    let got = run_resolve(
        &exec, &logits, &drafts, &qstore, &qsum, &par_rej, 2, rr, n, rmax,
    );
    assert_ne!(got[0], 200, "reject must never emit the draft");
    assert_eq!(got[0], 100, "residual mass concentrates at A");
    // case 3: point-q row (greedy marker): accept iff u1 < p(d). draft 400
    // has p ≈ e^-4 of the mass -> u1 = .9 rejects; recovery = p masked at
    // 400 -> argmax-of-p class pick = 300 for u2 anywhere in its mass
    let par_point = vec![
        1u32,
        0,
        1,
        inv_t.to_bits(),
        0.9f32.to_bits(),
        0.1f32.to_bits(),
        0,
        0,
    ];
    let got = run_resolve(
        &exec, &logits, &drafts, &qstore, &qsum, &par_point, 2, rr, n, rmax,
    );
    assert_ne!(got[1], 400, "point-q reject must not emit the draft");
    assert_eq!(got[1], 300, "masked-p recovery picks the dominant token");
    eprintln!("spec_rs_resolve: accept / residual / point-q rules OK");
}
