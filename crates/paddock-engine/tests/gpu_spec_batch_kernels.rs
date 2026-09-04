//! Bit-exactness tests for the P6b spec-batch kernels against their
//! copy_region / host-scan equivalents: the batched conv-ext build + segmented
//! conv must reproduce the single-slot ext+conv path exactly, the ragged
//! commit kernels must equal the explicit copies, and the device row argmax
//! must equal the host argmax. Light (no model load).

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

#[test]
fn conv_ext_chain_matches_single_slot_path() {
    let Some(exec) = exec() else { return };
    // 27B-like geometry, scaled down: conv_dim 96, k 4, chunk 5 rows, 3 slots
    let (n_slots, b, km1, r, conv_dim, k) = (4usize, 3usize, 3usize, 5usize, 96usize, 4usize);
    let seg = km1 + r;

    let wins = det(n_slots * km1 * conv_dim, 1);
    let mixed = det(b * r * conv_dim, 2);
    let w = det(conv_dim * k, 3);
    let slots: Vec<u32> = vec![2, 0, 3]; // permuted slot assignment

    let d_wins = exec.to_device(&wins).expect("wins");
    let d_mixed = exec.to_device(&mixed).expect("mixed");
    let d_w = exec.to_device(&w).expect("w");
    let d_slots = exec.to_device_u32(&slots).expect("slots");

    // batched path: ext build kernel + segmented conv
    let mut d_ext = exec
        .to_device(&vec![0f32; b * seg * conv_dim])
        .expect("ext");
    exec.conv_ext_build_slots(&d_wins, &d_slots, &d_mixed, &mut d_ext, b, km1, r, conv_dim)
        .expect("ext build");
    let mut d_out = exec.to_device(&vec![0f32; b * r * conv_dim]).expect("out");
    exec.conv_chunk_ext(&d_ext, &d_w, &mut d_out, b, km1, r, conv_dim, k)
        .expect("conv chunk");
    let got_ext = exec.to_host(&d_ext).expect("ext dtoh");
    let got = exec.to_host(&d_out).expect("out dtoh");

    // single-slot path per slot: copy_region ext + causal_conv1d_silu + slice
    for (seq, &slot) in slots.iter().enumerate() {
        let mut d_ext1 = exec.to_device(&vec![0f32; seg * conv_dim]).expect("ext1");
        exec.copy_region(
            &d_wins,
            slot as usize * km1 * conv_dim,
            &mut d_ext1,
            0,
            km1 * conv_dim,
        )
        .expect("cp win");
        exec.copy_region(
            &d_mixed,
            seq * r * conv_dim,
            &mut d_ext1,
            km1 * conv_dim,
            r * conv_dim,
        )
        .expect("cp mixed");
        let mut d_conv1 = exec.to_device(&vec![0f32; seg * conv_dim]).expect("conv1");
        exec.causal_conv1d_silu(&d_ext1, &d_w, &mut d_conv1, seg, conv_dim, k)
            .expect("conv single");
        let ext1 = exec.to_host(&d_ext1).expect("ext1 dtoh");
        let conv1 = exec.to_host(&d_conv1).expect("conv1 dtoh");

        let ext_b = &got_ext[seq * seg * conv_dim..(seq + 1) * seg * conv_dim];
        assert_eq!(ext_b, &ext1[..], "ext mismatch for seq {seq}");
        let out_b = &got[seq * r * conv_dim..(seq + 1) * r * conv_dim];
        let want = &conv1[km1 * conv_dim..];
        assert!(
            out_b.iter().zip(want).all(|(a, c)| a == c),
            "conv mismatch for seq {seq}"
        );
    }
    eprintln!("conv ext chain: BIT-EXACT vs single-slot path");
}

#[test]
fn commit_kernels_match_copy_region() {
    let Some(exec) = exec() else { return };
    let (n_slots, b, r, h, d) = (4usize, 3usize, 5usize, 3usize, 128usize);
    let state_elems = h * d * d;
    let slots: Vec<u32> = vec![1, 3, 0];
    let committed: Vec<u32> = vec![2, 5, 1]; // middle slot = full acceptance (no restore)

    let states0 = det(n_slots * state_elems, 10);
    let snap = det(b * r * state_elems, 11);
    let d_slots = exec.to_device_u32(&slots).expect("slots");
    let d_comm = exec.to_device_u32(&committed).expect("committed");

    // kernel path
    let mut d_states = exec.to_device(&states0).expect("states");
    let d_snap = exec.to_device(&snap).expect("snap");
    exec.state_restore_slots(&mut d_states, &d_snap, &d_slots, &d_comm, b, r, h, d)
        .expect("restore");
    let got = exec.to_host(&d_states).expect("states dtoh");

    // reference: explicit copies
    let mut want = states0.clone();
    for (seq, &slot) in slots.iter().enumerate() {
        let c = committed[seq] as usize;
        if c < r {
            let src = &snap[(seq * r + c - 1) * state_elems..(seq * r + c) * state_elems];
            want[slot as usize * state_elems..(slot as usize + 1) * state_elems]
                .copy_from_slice(src);
        }
    }
    assert!(
        got.iter().zip(&want).all(|(a, c)| a == c),
        "state restore mismatch"
    );

    // conv window commit
    let (km1, conv_dim) = (3usize, 96usize);
    let seg = km1 + r;
    let wins0 = det(n_slots * km1 * conv_dim, 12);
    let ext = det(b * seg * conv_dim, 13);
    let mut d_wins = exec.to_device(&wins0).expect("wins");
    let d_ext = exec.to_device(&ext).expect("ext");
    exec.conv_commit_slots(&d_ext, &mut d_wins, &d_slots, &d_comm, b, km1, r, conv_dim)
        .expect("conv commit");
    let got_w = exec.to_host(&d_wins).expect("wins dtoh");
    let mut want_w = wins0.clone();
    for (seq, &slot) in slots.iter().enumerate() {
        let c = committed[seq] as usize;
        for j in 0..km1 {
            let src = &ext[(seq * seg + c + j) * conv_dim..(seq * seg + c + j + 1) * conv_dim];
            want_w[(slot as usize * km1 + j) * conv_dim..(slot as usize * km1 + j + 1) * conv_dim]
                .copy_from_slice(src);
        }
    }
    assert!(
        got_w.iter().zip(&want_w).all(|(a, c)| a == c),
        "conv commit mismatch"
    );
    eprintln!("commit kernels: BIT-EXACT vs copy_region semantics");
}

#[test]
fn argmax_rows_matches_host() {
    let Some(exec) = exec() else { return };
    // vocab-scale rows with adversarial near-ties: duplicate the max value at a
    // later index - the LOWER index must win, as in the host scan
    let (rows, n) = (13usize, 248320usize);
    let mut x = det(rows * n, 42);
    for row in 0..rows {
        let base = row * n;
        // find the row max, then plant an exact duplicate later in the row
        let (mut bi, mut bv) = (0usize, f32::NEG_INFINITY);
        for i in 0..n {
            if x[base + i] > bv {
                bv = x[base + i];
                bi = i;
            }
        }
        if bi + 1000 < n {
            x[base + bi + 1000] = bv;
        }
    }
    let d_x = exec.to_device(&x).expect("x");
    let mut d_out = exec.to_device_u32(&vec![0u32; rows]).expect("out");
    exec.argmax_rows(&d_x, &mut d_out, rows, n)
        .expect("argmax rows");
    let got = exec.to_host_u32(&d_out).expect("dtoh");
    for (row, &got_row) in got.iter().enumerate() {
        let base = row * n;
        let (mut bi, mut bv) = (0usize, f32::NEG_INFINITY);
        for i in 0..n {
            if x[base + i] > bv {
                bv = x[base + i];
                bi = i;
            }
        }
        assert_eq!(
            got_row, bi as u32,
            "row {row}: device {} host {}",
            got_row, bi
        );
    }
    eprintln!("argmax_rows: matches host argmax on {rows} rows incl exact ties");
}

/// The snapshot-free verify pair (the dflash width work) must be
/// BIT-EXACT vs the legacy snapshot-rollback path: verify_hold's out[] equals
/// v2's, and commit_walk's final state equals the snapshot state_restore_slots
/// would have picked - for every committed length 1..=k1 (c == k1 exercises
/// the "whole chunk stood" convention, where the legacy kernel keeps the
/// advanced live state and the walk must reproduce it from round-start).
#[test]
fn snapshot_free_verify_walk_matches_snapshot_rollback() {
    let Some(exec) = exec() else { return };
    if !exec.has_gated_delta_commit_walk() {
        eprintln!("SKIP: pack has no gated_delta_verify_hold/commit_walk (need pack >= 0.20)");
        return;
    }
    // D is fixed at 128 by the v2 kernel family; small head/batch counts keep
    // the host reference cheap. Permuted slots exercise the slots indirection.
    let (n_slots, b, k1, n_heads, d) = (6usize, 4usize, 5usize, 3usize, 128usize);
    let rows = b * k1;

    let q = det(rows * n_heads * d, 11);
    let k = det(rows * n_heads * d, 12);
    let v = det(rows * n_heads * d, 13);
    let g: Vec<f32> = det(rows * n_heads, 14)
        .iter()
        .map(|x| x * 0.1 - 0.05)
        .collect();
    let beta: Vec<f32> = det(rows * n_heads, 15)
        .iter()
        .map(|x| x * 0.5 + 0.5)
        .collect();
    let states0 = det(n_slots * n_heads * d * d, 16);
    let slots: Vec<u32> = vec![4, 0, 5, 2];

    let d_q = exec.to_device(&q).expect("q");
    let d_k = exec.to_device(&k).expect("k");
    let d_v = exec.to_device(&v).expect("v");
    let d_g = exec.to_device(&g).expect("g");
    let d_beta = exec.to_device(&beta).expect("beta");
    let d_slots = exec.to_device_u32(&slots).expect("slots");

    for c in 1..=k1 {
        let committed: Vec<u32> = (0..b).map(|i| (((i + c) % k1) + 1).min(c) as u32).collect();
        // include the uniform-c row pattern too on the last lap
        let committed = if c == k1 {
            vec![k1 as u32; b]
        } else {
            committed
        };
        let d_committed = exec.to_device_u32(&committed).expect("committed");

        // legacy path: v2 snap-mode (advances live state) + snapshot rollback
        let mut d_states_a = exec.to_device(&states0).expect("states a");
        let mut d_snap = exec
            .to_device(&vec![0f32; b * k1 * n_heads * d * d])
            .expect("snap");
        let mut d_out_a = exec
            .to_device(&vec![0f32; rows * n_heads * d])
            .expect("out a");
        exec.gated_delta_recurrent_v2(
            &d_q,
            &d_k,
            &d_v,
            &d_g,
            &d_beta,
            Some(&d_slots),
            &mut d_states_a,
            0,
            Some(&mut d_snap),
            &mut d_out_a,
            b,
            k1,
            n_heads,
            d,
        )
        .expect("v2 snap");
        exec.state_restore_slots(
            &mut d_states_a,
            &d_snap,
            &d_slots,
            &d_committed,
            b,
            k1,
            n_heads,
            d,
        )
        .expect("restore");

        // snapshot-free path: hold verify + commit walk from round-start
        let mut d_states_b = exec.to_device(&states0).expect("states b");
        let mut d_out_b = exec
            .to_device(&vec![0f32; rows * n_heads * d])
            .expect("out b");
        exec.gated_delta_verify_hold(
            &d_q,
            &d_k,
            &d_v,
            &d_g,
            &d_beta,
            Some(&d_slots),
            &d_states_b,
            &mut d_out_b,
            b,
            k1,
            n_heads,
            d,
        )
        .expect("verify hold");
        exec.gated_delta_commit_walk(
            &d_k,
            &d_v,
            &d_g,
            &d_beta,
            Some(&d_slots),
            &d_committed,
            &mut d_states_b,
            b,
            k1,
            n_heads,
            d,
        )
        .expect("commit walk");

        let out_a = exec.to_host(&d_out_a).expect("out a dtoh");
        let out_b = exec.to_host(&d_out_b).expect("out b dtoh");
        assert!(
            out_a
                .iter()
                .zip(&out_b)
                .all(|(x, y)| x.to_bits() == y.to_bits()),
            "c={c}: verify out[] diverged from v2"
        );
        let sa = exec.to_host(&d_states_a).expect("states a dtoh");
        let sb = exec.to_host(&d_states_b).expect("states b dtoh");
        let bad = sa
            .iter()
            .zip(&sb)
            .filter(|(x, y)| x.to_bits() != y.to_bits())
            .count();
        assert_eq!(
            bad, 0,
            "c={c}: {bad} state words diverge from the snapshot rollback"
        );
    }
    eprintln!(
        "snapshot-free verify pair: bit-exact vs snapshot rollback across c=1..={k1} \
         (b={b}, H={n_heads}, D={d}, permuted slots)"
    );
}

#[test]
fn dflash_cond_append_matches_norm_rope_append_chain() {
    let Some(exec) = exec() else { return };
    if !exec.has_dflash_cond_append() {
        eprintln!("SKIP: pack has no dflash_cond_append (pre-0.22.0)");
        return;
    }
    use paddock_engine::gpu::KvDtype;
    let (n_kv, hd, bps, n_blocks) = (3usize, 128usize, 8usize, 8usize);
    let kv_dim = n_kv * hd;
    let eps = 1e-6f32;
    // ext_factor 0 pins ramp=0 (angle = freq_scale*theta) - still walks the
    // full theta chain; theta_scale = 10000^(-2/hd), the yarn default shape.
    let params = (
        10000f32.powf(-2.0 / hd as f32),
        1.0f32,
        0.0f32,
        1.0f32,
        0.0f32,
        1.0f32,
    );

    // two shapes: norm_batch < 64 (decode-nth path) and >= 64 (wide-nth path)
    for (cfg, r, cuts) in [
        ("narrow", 14usize, vec![(2usize, 4usize), (9, 3)]),
        ("wide", 30usize, vec![(0usize, 7usize), (12, 6), (25, 5)]),
    ] {
        // rows: slot 0 at positions 3.., slot 1 at 10.. (crosses the block-16
        // boundary on the wide shape) - block table maps to distinct blocks
        let half = r / 2;
        let mut pos: Vec<u32> = Vec::new();
        let mut slots: Vec<u32> = Vec::new();
        for i in 0..r {
            if i < half {
                slots.push(0);
                pos.push(3 + i as u32);
            } else {
                slots.push(1);
                pos.push(10 + (i - half) as u32);
            }
        }
        let mut bt = vec![0u32; 2 * bps];
        bt[0] = 2; // slot 0, block 0
        bt[1] = 3; // slot 0, block 1 (positions 16..31)
        bt[bps] = 5; // slot 1, block 0
        bt[bps + 1] = 7; // slot 1, block 1
        let fk = det(r * kv_dim, 11);
        let fv = det(r * kv_dim, 12);
        let kw = det(hd, 13);
        let d_fk = exec.to_device(&fk).expect("fk");
        let d_fv = exec.to_device(&fv).expect("fv");
        let d_kw = exec.to_device(&kw).expect("kw");
        let d_pos = exec.to_device_u32(&pos).expect("pos");
        let d_slots = exec.to_device_u32(&slots).expect("slots");
        let d_bt = exec.to_device_u32(&bt).expect("bt");
        let pool_bytes = n_blocks * 16 * kv_dim * 2;
        let fill = vec![0xABu8; pool_bytes];
        let mut a_k = exec.to_device_u8(&fill).expect("a_k");
        let mut a_v = exec.to_device_u8(&fill).expect("a_v");
        let mut b_k = exec.to_device_u8(&fill).expect("b_k");
        let mut b_v = exec.to_device_u8(&fill).expect("b_v");

        // legacy chain: norm(all rows) -> rope(all rows) -> per-cut appends
        let mut fkn = exec.to_device(&vec![0f32; r * kv_dim]).expect("fkn");
        exec.rmsnorm_batch(&d_fk, &d_kw, &mut fkn, hd, eps, r * n_kv)
            .expect("norm");
        exec.rope_yarn_batch(&mut fkn, &d_pos, n_kv, hd, params, r)
            .expect("rope");
        for &(off, len) in &cuts {
            exec.kv_append_batch_paged_rows(
                &fkn,
                &mut a_k,
                &d_pos,
                Some(&d_slots),
                &d_bt,
                bps,
                kv_dim,
                off,
                len,
                KvDtype::Fp16,
            )
            .expect("append k");
            exec.kv_append_batch_paged_rows(
                &d_fv,
                &mut a_v,
                &d_pos,
                Some(&d_slots),
                &d_bt,
                bps,
                kv_dim,
                off,
                len,
                KvDtype::Fp16,
            )
            .expect("append v");
        }

        // fused fold: one launch over the flattened cut rows
        let mut rows_w: Vec<u32> = Vec::new();
        for &(off, len) in &cuts {
            rows_w.extend(off as u32..(off + len) as u32);
        }
        let d_rows = exec.to_device_u32(&rows_w).expect("rows_w");
        exec.dflash_cond_append(
            &d_fk,
            &d_fv,
            &d_kw,
            &mut b_k,
            &mut b_v,
            &d_rows,
            &d_pos,
            &d_slots,
            &d_bt,
            bps,
            n_kv,
            hd,
            eps,
            params,
            rows_w.len(),
            r * n_kv,
        )
        .expect("fused");

        let ha_k = exec
            .to_host_range_u8(&a_k, 0, pool_bytes)
            .expect("a_k dtoh");
        let hb_k = exec
            .to_host_range_u8(&b_k, 0, pool_bytes)
            .expect("b_k dtoh");
        let ha_v = exec
            .to_host_range_u8(&a_v, 0, pool_bytes)
            .expect("a_v dtoh");
        let hb_v = exec
            .to_host_range_u8(&b_v, 0, pool_bytes)
            .expect("b_v dtoh");
        assert_eq!(ha_k, hb_k, "k pool mismatch ({cfg})");
        assert_eq!(ha_v, hb_v, "v pool mismatch ({cfg})");
        // sanity: the writes actually landed (pools differ from the fill)
        assert_ne!(
            ha_k, fill,
            "k pool untouched ({cfg}) - kernel did not write"
        );
        eprintln!("dflash cond fold ({cfg}): BIT-EXACT vs norm+rope+append chain");
    }
}

/// Rung E2: the dflash ring attention as one batched-runs launch
/// (grid.z over the blocks, the v4 hd128 arm's run prologue) must be
/// bit-identical to the per-block loop it replaces - same kernel, same tile
/// per block. Also the regression pin for the sm_120a prologue miscompile
/// (nvcc 13.3 lowered `slots += roff` to a 32-bit ULEA, caught under
/// memcheck) - a wrong slot reads another sequence's KV and the outputs
/// part, or the read faults outright.
#[test]
fn dflash_ring_attention_runs_matches_per_block_loop() {
    let Some(exec) = exec() else { return };
    if !exec.kernels_pf_runs_available() {
        eprintln!("SKIP: pack has no pf_runs_register (pre-0.17)");
        return;
    }
    use paddock_engine::gpu::KvDtype;
    // the DFlash2 drafter geometry: 32q / 8kv / hd128, f16 ring, 2048 window
    let (n_heads, n_kv, hd, bps, n_blocks) = (32usize, 8usize, 128usize, 8usize, 12usize);
    let (kv_dim, window) = (n_kv * hd, 2048usize);
    let scale = 1.0 / (hd as f32).sqrt();
    let sinks = exec.alloc_no_sinks(n_heads).expect("sinks");
    // 3 blocks x 8 rows, three different slots at different depths (slot 2's
    // block crosses the 16-token page boundary); the block end is every
    // row's attention bound (non-causal block), exactly as the drafter
    // stages d_apos
    let (n, rows) = (3usize, 8usize);
    let r = n * rows;
    let depth = [5u32, 21, 12];
    let slot_of = [2u32, 0, 1];
    let mut apos: Vec<u32> = Vec::new();
    let mut slots: Vec<u32> = Vec::new();
    for b in 0..n {
        apos.extend(std::iter::repeat_n(depth[b] + rows as u32 - 1, rows));
        slots.extend(std::iter::repeat_n(slot_of[b], rows));
    }
    // distinct pages per slot so a wrong slot reads different KV
    let mut bt = vec![0u32; 3 * bps];
    for s in 0..3 {
        for j in 0..bps {
            bt[s * bps + j] = ((s * 4 + j) % n_blocks) as u32;
        }
    }
    let q = det(r * n_heads * hd, 21);
    let d_q = exec.to_device(&q).expect("q");
    let d_apos = exec.to_device_u32(&apos).expect("apos");
    let d_slots = exec.to_device_u32(&slots).expect("slots");
    let d_bt = exec.to_device_u32(&bt).expect("bt");
    // f16 pools with deterministic, finite contents (small magnitudes)
    let pool_elems = n_blocks * 16 * kv_dim;
    let kf = det(pool_elems, 31);
    let vf = det(pool_elems, 32);
    let to_f16 = |v: &[f32]| -> Vec<u8> {
        v.iter()
            .flat_map(|&x| half::f16::from_f32(x * 0.25).to_le_bytes())
            .collect()
    };
    let pool_k = exec.to_device_u8(&to_f16(&kf)).expect("pool_k");
    let pool_v = exec.to_device_u8(&to_f16(&vf)).expect("pool_v");

    // A: the per-block loop
    let mut out_a = exec
        .to_device(&vec![0f32; r * n_heads * hd])
        .expect("out_a");
    for b in 0..n {
        exec.attn_prefill_f16_rows_paged(
            &d_q,
            &pool_k,
            &pool_v,
            &sinks,
            &mut out_a,
            &d_apos,
            &d_slots,
            &d_bt,
            bps,
            n_heads,
            n_kv,
            hd,
            kv_dim,
            window,
            b * rows,
            rows,
            scale,
            KvDtype::Fp16,
        )
        .expect("per-block");
    }
    // B: one batched-runs launch over the same rows
    let offs: Vec<u32> = (0..=n).map(|i| (i * rows) as u32).collect();
    let d_offs = exec.to_device_u32(&offs).expect("offs");
    let mut out_b = exec
        .to_device(&vec![0f32; r * n_heads * hd])
        .expect("out_b");
    exec.pf_runs_register(Some((&d_offs, n as u32, rows as u32)))
        .expect("arm");
    let res = exec.attn_prefill_f16_rows_paged(
        &d_q,
        &pool_k,
        &pool_v,
        &sinks,
        &mut out_b,
        &d_apos,
        &d_slots,
        &d_bt,
        bps,
        n_heads,
        n_kv,
        hd,
        kv_dim,
        window,
        0,
        r,
        scale,
        KvDtype::Fp16,
    );
    exec.pf_runs_register(None).expect("disarm");
    res.expect("runs launch");

    let ha = exec.to_host(&out_a).expect("a dtoh");
    let hb = exec.to_host(&out_b).expect("b dtoh");
    assert!(
        ha.iter().all(|x| x.is_finite()),
        "per-block output has non-finite values"
    );
    let bits = |v: &[f32]| -> Vec<u32> { v.iter().map(|x| x.to_bits()).collect() };
    assert_eq!(
        bits(&ha),
        bits(&hb),
        "batched-runs ring attention differs from the per-block loop"
    );
    // sanity: the blocks genuinely attend different slots (a slot mix-up
    // would still be self-consistent between A and B, so prove the inputs
    // were distinct: block 0 vs block 1 outputs differ)
    let blk = n_heads * hd * rows;
    assert_ne!(
        &ha[..blk],
        &ha[blk..2 * blk],
        "blocks 0 and 1 produced identical attention"
    );
    eprintln!(
        "dflash ring attention: batched-runs launch BIT-EXACT vs per-block loop (n={n}, rows={rows})"
    );
}
