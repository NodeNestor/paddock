//! Parity test for the Qwen3.5 Gated DeltaNet recurrence kernel vs the CPU
//! reference (`paddock_kernels::reference::delta_net`). Gated on a CUDA device +
//! built pack. The GPU runs the whole sequence for all heads in one launch (one
//! block per head, thread j owns state column j); the reference runs the same
//! sequential recurrence in plain f32.

mod common;

use paddock_kernels::reference::delta_net::{gated_delta_chunked, gated_delta_recurrent};

/// Deterministic pseudo-random f32 in (-0.5, 0.5) - same LCG the other parity
/// tests use, so inputs are reproducible.
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

#[test]
fn gated_delta_recurrent_matches_cpu_reference() {
    let Some(exec) = common::gpu() else {
        return;
    };

    // qwen3.5 geometry: head_dim = ssm.state_size = 128. A few heads + a short
    // sequence exercise the full recurrence and the state carry across tokens.
    let (t, h, d) = (16usize, 4usize, 128usize);
    let n = t * h * d;

    let q = det(n, 1);
    let k = det(n, 2);
    let v = det(n, 3);
    // g is the log-decay: keep it negative so exp(g) ∈ (0,1); beta ∈ (0,1).
    let g: Vec<f32> = det(t * h, 4).iter().map(|x| x - 0.5).collect(); // (-1, 0)
    let beta: Vec<f32> = det(t * h, 5).iter().map(|x| x + 0.5).collect(); // (0, 1)

    // CPU reference
    let mut ref_state = vec![0f32; h * d * d];
    let mut ref_out = vec![0f32; n];
    gated_delta_recurrent(&q, &k, &v, &g, &beta, &mut ref_state, &mut ref_out, t, h, d);

    // GPU (state zero-initialized via an uploaded zero buffer)
    let d_q = exec.to_device(&q).expect("q");
    let d_k = exec.to_device(&k).expect("k");
    let d_v = exec.to_device(&v).expect("v");
    let d_g = exec.to_device(&g).expect("g");
    let d_beta = exec.to_device(&beta).expect("beta");
    let mut d_state = exec.to_device(&vec![0f32; h * d * d]).expect("state");
    let mut d_out = exec.to_device(&vec![0f32; n]).expect("out");
    exec.gated_delta_recurrent(
        &d_q,
        &d_k,
        &d_v,
        &d_g,
        &d_beta,
        &mut d_state,
        &mut d_out,
        t,
        h,
        d,
    )
    .expect("gated_delta_recurrent");
    let got_out = exec.to_host(&d_out).expect("out dtoh");
    let got_state = exec.to_host(&d_state).expect("state dtoh");

    let maxd = |a: &[f32], b: &[f32]| {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    };
    let out_diff = maxd(&got_out, &ref_out);
    let state_diff = maxd(&got_state, &ref_state);
    eprintln!("delta-net parity: max|out| {out_diff:.2e}  max|state| {state_diff:.2e}");
    // The GPU L2-norm reduction sums in a different order than the sequential
    // reference and uses rsqrtf, but the recurrence otherwise matches - observed
    // ~2e-9 (out) / ~2e-8 (state); gate at 1e-5 to catch real regressions.
    assert!(out_diff < 1e-5, "out max_abs_diff {out_diff} too high");
    assert!(
        state_diff < 1e-5,
        "state max_abs_diff {state_diff} too high"
    );
}

/// L2-normalize q,k per (t,h) row on the host the way `deltanet_split_gqa_norm`
/// does (the v2 kernel takes q,k pre-normalized; q carries the 1/sqrt(D) scale).
fn norm_qk(q: &[f32], k: &[f32], rows: usize, d: usize) -> (Vec<f32>, Vec<f32>) {
    let (mut qn, mut kn) = (q.to_vec(), k.to_vec());
    for r in 0..rows {
        let row = &q[r * d..(r + 1) * d];
        let qs = row.iter().map(|x| x * x).sum::<f32>();
        let krow = &k[r * d..(r + 1) * d];
        let ks = krow.iter().map(|x| x * x).sum::<f32>();
        let qi = 1.0 / (qs + 1e-6).sqrt() / (d as f32).sqrt();
        let ki = 1.0 / (ks + 1e-6).sqrt();
        for j in 0..d {
            qn[r * d + j] = row[j] * qi;
            kn[r * d + j] = krow[j] * ki;
        }
    }
    (qn, kn)
}

/// Transpose each per-head [D, D] state tile (v2 stores column-contiguous;
/// the CPU reference is row-major).
fn transpose_tiles(s: &[f32], tiles: usize, d: usize) -> Vec<f32> {
    let mut out = vec![0f32; s.len()];
    for tl in 0..tiles {
        let b = tl * d * d;
        for i in 0..d {
            for j in 0..d {
                out[b + i * d + j] = s[b + j * d + i];
            }
        }
    }
    out
}

/// v2 kernel, single-sequence chunk mode with per-token snapshots: out and the
/// final state must match the CPU reference; snapshot t must equal the reference
/// state after t+1 tokens (the speculative-rollback contract).
#[test]
fn gated_delta_recurrent_v2_chunk_and_snapshots_match_cpu_reference() {
    let Some(exec) = common::gpu() else {
        return;
    };

    let (t, h, d) = (8usize, 4usize, 128usize);
    let n = t * h * d;
    let q = det(n, 11);
    let k = det(n, 12);
    let v = det(n, 13);
    let g: Vec<f32> = det(t * h, 14).iter().map(|x| x - 0.5).collect();
    let beta: Vec<f32> = det(t * h, 15).iter().map(|x| x + 0.5).collect();

    let (qn, kn) = norm_qk(&q, &k, t * h, d);
    let d_q = exec.to_device(&qn).expect("q");
    let d_k = exec.to_device(&kn).expect("k");
    let d_v = exec.to_device(&v).expect("v");
    let d_g = exec.to_device(&g).expect("g");
    let d_beta = exec.to_device(&beta).expect("beta");
    let mut d_state = exec.to_device(&vec![0f32; h * d * d]).expect("state");
    let mut d_snap = exec.to_device(&vec![0f32; t * h * d * d]).expect("snap");
    let mut d_out = exec.to_device(&vec![0f32; n]).expect("out");
    exec.gated_delta_recurrent_v2(
        &d_q,
        &d_k,
        &d_v,
        &d_g,
        &d_beta,
        None,
        &mut d_state,
        0,
        Some(&mut d_snap),
        &mut d_out,
        1,
        t,
        h,
        d,
    )
    .expect("v2");
    let got_out = exec.to_host(&d_out).expect("out dtoh");
    let got_state = transpose_tiles(&exec.to_host(&d_state).expect("state dtoh"), h, d);
    let got_snap = exec.to_host(&d_snap).expect("snap dtoh");

    let maxd = |a: &[f32], b: &[f32]| {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    };

    let mut ref_state = vec![0f32; h * d * d];
    let mut ref_out = vec![0f32; n];
    gated_delta_recurrent(&q, &k, &v, &g, &beta, &mut ref_state, &mut ref_out, t, h, d);
    let out_diff = maxd(&got_out, &ref_out);
    let state_diff = maxd(&got_state, &ref_state);

    // snapshots: prefix-run the reference to each length t+1
    let mut snap_diff = 0f32;
    for tt in 0..t {
        let rows = (tt + 1) * h * d;
        let mut ps = vec![0f32; h * d * d];
        let mut po = vec![0f32; rows];
        gated_delta_recurrent(
            &q[..rows],
            &k[..rows],
            &v[..rows],
            &g[..(tt + 1) * h],
            &beta[..(tt + 1) * h],
            &mut ps,
            &mut po,
            tt + 1,
            h,
            d,
        );
        let sn = transpose_tiles(&got_snap[tt * h * d * d..(tt + 1) * h * d * d], h, d);
        snap_diff = snap_diff.max(maxd(&sn, &ps));
    }
    eprintln!(
        "delta-net v2 parity: max|out| {out_diff:.2e}  max|state| {state_diff:.2e}  max|snap| {snap_diff:.2e}"
    );
    assert!(out_diff < 1e-5, "out max_abs_diff {out_diff} too high");
    assert!(
        state_diff < 1e-5,
        "state max_abs_diff {state_diff} too high"
    );
    assert!(snap_diff < 1e-5, "snap max_abs_diff {snap_diff} too high");
}

/// v2 kernel, slots mode: B sequences advance their own (pre-seeded, distinct)
/// state slots by one token; each must match an independent CPU reference run
/// continued from the same initial state.
#[test]
fn gated_delta_recurrent_v2_slots_match_cpu_reference() {
    let Some(exec) = common::gpu() else {
        return;
    };

    let (b, h, d) = (3usize, 4usize, 128usize);
    let n = b * h * d;
    let q = det(n, 21);
    let k = det(n, 22);
    let v = det(n, 23);
    let g: Vec<f32> = det(b * h, 24).iter().map(|x| x - 0.5).collect();
    let beta: Vec<f32> = det(b * h, 25).iter().map(|x| x + 0.5).collect();
    // distinct non-zero initial states per slot (scaled into a sane range)
    let init: Vec<Vec<f32>> = (0..b)
        .map(|s| {
            det(h * d * d, 30 + s as u64)
                .iter()
                .map(|x| x * 0.1)
                .collect()
        })
        .collect();
    // permuted slot assignment proves the indirection
    let slots: Vec<u32> = vec![2, 0, 1];

    let mut states_host = vec![0f32; b * h * d * d];
    for (seq, &slot) in slots.iter().enumerate() {
        let tr = transpose_tiles(&init[seq], h, d);
        states_host[slot as usize * h * d * d..(slot as usize + 1) * h * d * d]
            .copy_from_slice(&tr);
    }

    let (qn, kn) = norm_qk(&q, &k, b * h, d);
    let d_q = exec.to_device(&qn).expect("q");
    let d_k = exec.to_device(&kn).expect("k");
    let d_v = exec.to_device(&v).expect("v");
    let d_g = exec.to_device(&g).expect("g");
    let d_beta = exec.to_device(&beta).expect("beta");
    let d_slots = exec.to_device_u32(&slots).expect("slots");
    let mut d_states = exec.to_device(&states_host).expect("states");
    let mut d_out = exec.to_device(&vec![0f32; n]).expect("out");
    exec.gated_delta_recurrent_v2(
        &d_q,
        &d_k,
        &d_v,
        &d_g,
        &d_beta,
        Some(&d_slots),
        &mut d_states,
        0,
        None,
        &mut d_out,
        b,
        1,
        h,
        d,
    )
    .expect("v2 slots");
    let got_out = exec.to_host(&d_out).expect("out dtoh");
    let got_states = exec.to_host(&d_states).expect("states dtoh");

    let maxd = |a: &[f32], b: &[f32]| {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    };
    let (mut out_diff, mut state_diff) = (0f32, 0f32);
    for (seq, &slot) in slots.iter().enumerate() {
        let rows = h * d;
        let mut rs = init[seq].clone();
        let mut ro = vec![0f32; rows];
        gated_delta_recurrent(
            &q[seq * rows..(seq + 1) * rows],
            &k[seq * rows..(seq + 1) * rows],
            &v[seq * rows..(seq + 1) * rows],
            &g[seq * h..(seq + 1) * h],
            &beta[seq * h..(seq + 1) * h],
            &mut rs,
            &mut ro,
            1,
            h,
            d,
        );
        out_diff = out_diff.max(maxd(&got_out[seq * rows..(seq + 1) * rows], &ro));
        let gs = transpose_tiles(
            &got_states[slot as usize * h * d * d..(slot as usize + 1) * h * d * d],
            h,
            d,
        );
        state_diff = state_diff.max(maxd(&gs, &rs));
    }
    eprintln!("delta-net v2 slots parity: max|out| {out_diff:.2e}  max|state| {state_diff:.2e}");
    assert!(out_diff < 1e-5, "out max_abs_diff {out_diff} too high");
    assert!(
        state_diff < 1e-5,
        "state max_abs_diff {state_diff} too high"
    );
}

/// Packed multi-span kernel: decode rows (len-1 items) + short span walks in
/// one launch must be BIT-EXACT vs the composition it replaces (the
/// slots-mode step for the decode rows + a per-span `_at` walk each) - the
/// body is the v2 kernel's verbatim, so any bit drift is a bug - and match
/// the CPU sequential reference at the usual tolerance.
#[test]
fn gated_delta_recurrent_v2_packed_matches_composition() {
    let Some(exec) = common::gpu() else {
        return;
    };
    if !exec.has_dn_recurrent_packed() {
        eprintln!("pack lacks gated_delta_recurrent_v2_packed - skipping");
        return;
    }

    let (h, d) = (4usize, 128usize);
    // two decode rows, spans of 5 and 3 rows, then a fused-ckpt tail CHAIN of
    // 4+2 rows (one item, mid-snapshot after row 4); permuted distinct slots
    let lens = [1usize, 1, 5, 3, 6];
    let slots = [3u32, 1, 0, 2, 4];
    let chain_cut = 4usize; // chain item = rows 10..16, seam after 4 rows
    let rows: usize = lens.iter().sum();
    let n = rows * h * d;
    let q = det(n, 41);
    let k = det(n, 42);
    let v = det(n, 43);
    let g: Vec<f32> = det(rows * h, 44).iter().map(|x| x - 0.5).collect();
    let beta: Vec<f32> = det(rows * h, 45).iter().map(|x| x + 0.5).collect();
    let init: Vec<Vec<f32>> = (0..lens.len())
        .map(|s| {
            det(h * d * d, 50 + s as u64)
                .iter()
                .map(|x| x * 0.1)
                .collect()
        })
        .collect();
    let mut states_host = vec![0f32; lens.len() * h * d * d];
    for (seq, &slot) in slots.iter().enumerate() {
        let tr = transpose_tiles(&init[seq], h, d);
        states_host[slot as usize * h * d * d..(slot as usize + 1) * h * d * d]
            .copy_from_slice(&tr);
    }

    let (qn, kn) = norm_qk(&q, &k, rows * h, d);
    let d_q = exec.to_device(&qn).expect("q");
    let d_k = exec.to_device(&kn).expect("k");
    let d_v = exec.to_device(&v).expect("v");
    let d_g = exec.to_device(&g).expect("g");
    let d_beta = exec.to_device(&beta).expect("beta");

    // arm A: one packed launch over all five items (stride-8 descriptors;
    // the chain snapshots into blob1 at a nonzero region offset)
    let snap_off = 512usize;
    let mut items: Vec<u32> = Vec::new();
    let mut row0 = 0u32;
    for (i, &len) in lens.iter().enumerate() {
        let (sat, sas) = if i == 4 {
            (chain_cut as u32, 1u32)
        } else {
            (0, 0)
        };
        items.extend_from_slice(&[row0, len as u32, slots[i], sat, sas, 0, 0, 0]);
        row0 += len as u32;
    }
    let d_items = exec.to_device_u32(&items).expect("items");
    let mut d_states_a = exec.to_device(&states_host).expect("states a");
    let mut d_out_a = exec.to_device(&vec![0f32; n]).expect("out a");
    let mut d_snap0 = exec
        .to_device(&vec![0f32; snap_off + h * d * d])
        .expect("snap0");
    let mut d_snap1 = exec
        .to_device(&vec![0f32; snap_off + h * d * d])
        .expect("snap1");
    exec.gated_delta_recurrent_v2_packed(
        &d_q,
        &d_k,
        &d_v,
        &d_g,
        &d_beta,
        &d_items,
        &mut d_states_a,
        &mut d_out_a,
        Some((&mut d_snap0, snap_off)),
        Some((&mut d_snap1, snap_off)),
        lens.len(),
        h,
        d,
    )
    .expect("packed");

    // arm B: the composition it replaces - slots step for rows 0..2, per-span
    // _at walks, and the chain as two _at walks with a state copy between
    let d_slots2 = exec.to_device_u32(&slots[0..2]).expect("slots2");
    let mut d_states_b = exec.to_device(&states_host).expect("states b");
    let mut d_out_b = exec.to_device(&vec![0f32; n]).expect("out b");
    exec.gated_delta_recurrent_v2(
        &d_q,
        &d_k,
        &d_v,
        &d_g,
        &d_beta,
        Some(&d_slots2),
        &mut d_states_b,
        0,
        None,
        &mut d_out_b,
        2,
        1,
        h,
        d,
    )
    .expect("v2 slots");
    let mut rb = 2usize;
    let mut ref_snap = vec![0f32; h * d * d];
    for (i, &len) in lens.iter().enumerate().skip(2) {
        let off = slots[i] as usize * h * d * d;
        if i == 4 {
            exec.gated_delta_recurrent_v2_at(
                &d_q,
                &d_k,
                &d_v,
                &d_g,
                &d_beta,
                &mut d_states_b,
                off,
                &mut d_out_b,
                rb,
                chain_cut,
                h,
                d,
            )
            .expect("v2 at chain head");
            let st_mid = exec.to_host(&d_states_b).expect("mid state");
            ref_snap.copy_from_slice(&st_mid[off..off + h * d * d]);
            exec.gated_delta_recurrent_v2_at(
                &d_q,
                &d_k,
                &d_v,
                &d_g,
                &d_beta,
                &mut d_states_b,
                off,
                &mut d_out_b,
                rb + chain_cut,
                len - chain_cut,
                h,
                d,
            )
            .expect("v2 at chain tail");
        } else {
            exec.gated_delta_recurrent_v2_at(
                &d_q,
                &d_k,
                &d_v,
                &d_g,
                &d_beta,
                &mut d_states_b,
                off,
                &mut d_out_b,
                rb,
                len,
                h,
                d,
            )
            .expect("v2 at");
        }
        rb += len;
    }

    let out_a = exec.to_host(&d_out_a).expect("out a dtoh");
    let out_b = exec.to_host(&d_out_b).expect("out b dtoh");
    let st_a = exec.to_host(&d_states_a).expect("st a dtoh");
    let st_b = exec.to_host(&d_states_b).expect("st b dtoh");
    let snap1_h = exec.to_host(&d_snap1).expect("snap1 dtoh");
    let bitdiff = |a: &[f32], b: &[f32]| {
        a.iter()
            .zip(b)
            .filter(|(x, y)| x.to_bits() != y.to_bits())
            .count()
    };
    let (od, sd) = (bitdiff(&out_a, &out_b), bitdiff(&st_a, &st_b));
    let np = bitdiff(&snap1_h[snap_off..snap_off + h * d * d], &ref_snap);
    eprintln!(
        "delta-net v2 packed vs composition: out bitdiff {od}  state bitdiff {sd}  seam-snap bitdiff {np}"
    );
    assert_eq!(od, 0, "packed out not bit-exact vs composition");
    assert_eq!(sd, 0, "packed state not bit-exact vs composition");
    assert_eq!(
        np, 0,
        "in-kernel seam snapshot not bit-exact vs staged copy"
    );

    // CPU sequential reference per item (tolerance-level)
    let maxd = |a: &[f32], b: &[f32]| {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    };
    let (mut out_diff, mut state_diff) = (0f32, 0f32);
    let mut r0 = 0usize;
    for (seq, &len) in lens.iter().enumerate() {
        let rows_i = len * h * d;
        let mut rs = init[seq].clone();
        let mut ro = vec![0f32; rows_i];
        gated_delta_recurrent(
            &q[r0 * h * d..r0 * h * d + rows_i],
            &k[r0 * h * d..r0 * h * d + rows_i],
            &v[r0 * h * d..r0 * h * d + rows_i],
            &g[r0 * h..(r0 + len) * h],
            &beta[r0 * h..(r0 + len) * h],
            &mut rs,
            &mut ro,
            len,
            h,
            d,
        );
        out_diff = out_diff.max(maxd(&out_a[r0 * h * d..r0 * h * d + rows_i], &ro));
        let slot = slots[seq] as usize;
        let gs = transpose_tiles(&st_a[slot * h * d * d..(slot + 1) * h * d * d], h, d);
        state_diff = state_diff.max(maxd(&gs, &rs));
        r0 += len;
    }
    eprintln!(
        "delta-net v2 packed cpu parity: max|out| {out_diff:.2e}  max|state| {state_diff:.2e}"
    );
    assert!(out_diff < 1e-5, "out max_abs_diff {out_diff} too high");
    assert!(
        state_diff < 1e-5,
        "state max_abs_diff {state_diff} too high"
    );
}

/// Chunked prefill kernel: for a spread of sequence lengths (whole chunks,
/// partial tails, single/short chunks) and a nonzero incoming state, the GPU
/// chunked scan must match both its CPU oracle (`gated_delta_chunked`, same
/// algorithm and numeric recipe - tight) and the plain sequential recurrence
/// (tolerance-level: the reformulation is not bit-identical). The t=100 case
/// also runs at a nonzero state_elem_off to prove the slot pointer math.
#[test]
fn gated_delta_chunked_matches_cpu_references() {
    let Some(exec) = common::gpu() else {
        return;
    };

    let (h, d) = (4usize, 128usize);
    let maxd = |a: &[f32], b: &[f32]| {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    };

    // (t, decay scale): full-range g kills cross-chunk coupling within a few
    // tokens, so the t=512 leg also runs near-zero decay - every chunk feeds
    // the next and accumulation drift has nowhere to hide.
    for (case, &(t, gscale)) in [
        (512usize, 1.0f32),
        (512, 0.02),
        (100, 1.0),
        (64, 1.0),
        (65, 1.0),
        (33, 1.0),
    ]
    .iter()
    .enumerate()
    {
        let n = t * h * d;
        let seed = 100 * (case as u64 + 1);
        let q = det(n, seed + 1);
        let k = det(n, seed + 2);
        let v = det(n, seed + 3);
        let g: Vec<f32> = det(t * h, seed + 4)
            .iter()
            .map(|x| (x - 0.5) * gscale)
            .collect();
        let beta: Vec<f32> = det(t * h, seed + 5).iter().map(|x| x + 0.5).collect();
        let init: Vec<f32> = det(h * d * d, seed + 6).iter().map(|x| x * 0.1).collect();

        // CPU references, both continued from the same nonzero state
        let mut seq_state = init.clone();
        let mut seq_out = vec![0f32; n];
        gated_delta_recurrent(&q, &k, &v, &g, &beta, &mut seq_state, &mut seq_out, t, h, d);
        let mut chk_state = init.clone();
        let mut chk_out = vec![0f32; n];
        gated_delta_chunked(
            &q,
            &k,
            &v,
            &g,
            &beta,
            &mut chk_state,
            &mut chk_out,
            t,
            h,
            d,
            64,
        );

        // GPU: pre-normalized q/k, transposed state, slot 1 of a 2-slot buffer
        // for the t=100 case (elem-offset path), slot 0 elsewhere
        let off_slots = if t == 100 { 2usize } else { 1usize };
        let off = (off_slots - 1) * h * d * d;
        let (qn, kn) = norm_qk(&q, &k, t * h, d);
        let d_q = exec.to_device(&qn).expect("q");
        let d_k = exec.to_device(&kn).expect("k");
        let d_v = exec.to_device(&v).expect("v");
        let d_g = exec.to_device(&g).expect("g");
        let d_beta = exec.to_device(&beta).expect("beta");
        let mut states_host = vec![0f32; off_slots * h * d * d];
        states_host[off..].copy_from_slice(&transpose_tiles(&init, h, d));
        let mut d_state = exec.to_device(&states_host).expect("state");
        let mut d_out = exec.to_device(&vec![0f32; n]).expect("out");
        let nc = t.div_ceil(64);
        let mut d_dw = exec.to_device(&vec![0f32; nc * h * 64 * d]).expect("dw");
        let mut d_du = exec.to_device(&vec![0f32; nc * h * 64 * d]).expect("du");
        let mut d_aqk = exec.to_device(&vec![0f32; nc * h * 64 * 64]).expect("aqk");
        let mut d_cg = exec.alloc_f64(nc * h * 64).expect("cg");
        exec.gated_delta_chunked(
            &d_q,
            &d_k,
            &d_v,
            &d_g,
            &d_beta,
            &mut d_state,
            off,
            &mut d_out,
            &mut d_dw,
            &mut d_du,
            &mut d_aqk,
            &mut d_cg,
            t,
            h,
            d,
        )
        .expect("gated_delta_chunked");
        let got_out = exec.to_host(&d_out).expect("out dtoh");
        let got_states = exec.to_host(&d_state).expect("state dtoh");
        let got_state = transpose_tiles(&got_states[off..], h, d);
        if off > 0 {
            assert!(
                got_states[..off].iter().all(|&x| x == 0.0),
                "state slot 0 clobbered at offset run"
            );
        }

        let (o_chk, s_chk) = (maxd(&got_out, &chk_out), maxd(&got_state, &chk_state));
        let (o_seq, s_seq) = (maxd(&got_out, &seq_out), maxd(&got_state, &seq_state));
        eprintln!(
            "chunked t={t:3} gs={gscale}: vs cpu-chunked out {o_chk:.2e} state {s_chk:.2e} | vs sequential out {o_seq:.2e} state {s_seq:.2e}"
        );
        assert!(o_chk < 2e-5, "t={t} out vs cpu-chunked {o_chk} too high");
        assert!(s_chk < 2e-5, "t={t} state vs cpu-chunked {s_chk} too high");
        assert!(o_seq < 5e-5, "t={t} out vs sequential {o_seq} too high");
        assert!(s_seq < 5e-5, "t={t} state vs sequential {s_seq} too high");
    }
}

/// Heavy timing probe at the real 9B geometry (T=512, 32 heads, D=128): the
/// chunked scan vs the v2 sequential recurrence, per-layer wall time.
#[test]
fn gated_delta_chunked_speed_vs_v2() {
    if !common::heavy() {
        return;
    }
    let Some(exec) = common::gpu() else {
        return;
    };

    let (t, h, d) = (2048usize, 32usize, 128usize);
    let n = t * h * d;
    let q = det(n, 71);
    let k = det(n, 72);
    let v = det(n, 73);
    let g: Vec<f32> = det(t * h, 74).iter().map(|x| (x - 0.5) * 0.1).collect();
    let beta: Vec<f32> = det(t * h, 75).iter().map(|x| x + 0.5).collect();
    let (qn, kn) = norm_qk(&q, &k, t * h, d);
    let d_q = exec.to_device(&qn).expect("q");
    let d_k = exec.to_device(&kn).expect("k");
    let d_v = exec.to_device(&v).expect("v");
    let d_g = exec.to_device(&g).expect("g");
    let d_beta = exec.to_device(&beta).expect("beta");
    let mut d_state = exec.to_device(&vec![0f32; h * d * d]).expect("state");
    let mut d_out = exec.to_device(&vec![0f32; n]).expect("out");
    let nc = t.div_ceil(64);
    let mut d_dw = exec.to_device(&vec![0f32; nc * h * 64 * d]).expect("dw");
    let mut d_du = exec.to_device(&vec![0f32; nc * h * 64 * d]).expect("du");
    let mut d_aqk = exec.to_device(&vec![0f32; nc * h * 64 * 64]).expect("aqk");
    let mut d_cg = exec.alloc_f64(nc * h * 64).expect("cg");

    let iters = 50;
    for &tl in &[64usize, 128, 256, 512, 1024, 2048] {
        for probe in ["v2", "chunked"] {
            // warm up, then time
            for _ in 0..3 {
                match probe {
                    "v2" => exec
                        .gated_delta_recurrent_v2(
                            &d_q,
                            &d_k,
                            &d_v,
                            &d_g,
                            &d_beta,
                            None,
                            &mut d_state,
                            0,
                            None,
                            &mut d_out,
                            1,
                            tl,
                            h,
                            d,
                        )
                        .expect("v2"),
                    _ => exec
                        .gated_delta_chunked(
                            &d_q,
                            &d_k,
                            &d_v,
                            &d_g,
                            &d_beta,
                            &mut d_state,
                            0,
                            &mut d_out,
                            &mut d_dw,
                            &mut d_du,
                            &mut d_aqk,
                            &mut d_cg,
                            tl,
                            h,
                            d,
                        )
                        .expect("chunked"),
                }
            }
            exec.synchronize().unwrap();
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                match probe {
                    "v2" => exec
                        .gated_delta_recurrent_v2(
                            &d_q,
                            &d_k,
                            &d_v,
                            &d_g,
                            &d_beta,
                            None,
                            &mut d_state,
                            0,
                            None,
                            &mut d_out,
                            1,
                            tl,
                            h,
                            d,
                        )
                        .expect("v2"),
                    _ => exec
                        .gated_delta_chunked(
                            &d_q,
                            &d_k,
                            &d_v,
                            &d_g,
                            &d_beta,
                            &mut d_state,
                            0,
                            &mut d_out,
                            &mut d_dw,
                            &mut d_du,
                            &mut d_aqk,
                            &mut d_cg,
                            tl,
                            h,
                            d,
                        )
                        .expect("chunked"),
                }
            }
            exec.synchronize().unwrap();
            let us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
            eprintln!("{probe:8} T={tl:3} H=32 D=128: {us:.1} us/layer");
        }
    }
}
