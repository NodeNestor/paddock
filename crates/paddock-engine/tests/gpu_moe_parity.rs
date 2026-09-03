//! Equivalence test for the fused MoE kernels (mxfp4_moe_gate_up +
//! mxfp4_moe_down) against the original unfused chain (per-expert
//! mxfp4_gemv_indexed + swiglu_oai + scale_add_dev) on real gpt-oss expert
//! weights. Same inputs -> same output within GEMV reduction-order noise.
//!
//! Gated on: CUDA device + built pack + the gpt-oss download.
#![allow(clippy::unwrap_used)]

mod common;

use paddock_engine::gpu::RepackedMxfp4;
use paddock_models::mapped::MappedGguf;

const ALPHA: f32 = 1.702;
const LIMIT: f32 = 7.0;

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

fn rel_err(a: &[f32], b: &[f32]) -> f32 {
    let mut num = 0f64;
    let mut den = 0f64;
    for (x, y) in a.iter().zip(b) {
        num += ((x - y) as f64).powi(2);
        den += (*y as f64).powi(2);
    }
    (num.sqrt() / den.sqrt().max(1e-12)) as f32
}

/// dp4a MXFP4 GEMV (quantized activation + integer __dp4a, in-register unpack)
/// vs the float mxfp4_gemv_indexed, on a real expert. Checks the accuracy cost
/// (should be < 1%) and reports the speedup - the MoE is compute-bound, so this
/// is where the integer path pays the most.
#[test]
fn dp4a_mxfp4_close_to_f32() {
    let Some(model) = common::model("PADDOCK_MODEL", common::GPT_OSS_20B) else {
        return;
    };
    let Some(exec) = common::gpu() else {
        return;
    };
    let map = MappedGguf::open(&model).expect("open gguf");
    let gate = exec.upload_raw(&map, "blk.0.ffn_gate_exps.weight").unwrap();
    let (embd, ff) = (gate.dims[0], gate.dims[1]);
    let slot = 0usize;
    let idx: Vec<u32> = vec![7]; // expert 7
    let d_idx = exec.stream.clone_htod(&idx).unwrap();
    let x = det(embd, 11);
    let d_x = exec.to_device(&x).unwrap();

    // float reference
    let mut d_yf = exec.alloc(ff).unwrap();
    exec.mxfp4_gemv_indexed(&gate.bytes, None, &d_idx, slot, &d_x, &mut d_yf, embd, ff)
        .unwrap();
    let yf = exec.to_host(&d_yf).unwrap();

    // dp4a path
    let mut xq = exec.alloc_i8(embd).unwrap();
    let mut xs = exec.alloc(embd / 32).unwrap();
    exec.quantize_q8(&d_x, &mut xq, &mut xs, embd).unwrap();
    let mut d_yd = exec.alloc(ff).unwrap();
    exec.mxfp4_gemv_indexed_dp4a(
        &gate.bytes,
        None,
        &d_idx,
        slot,
        &xq,
        &xs,
        &mut d_yd,
        embd,
        ff,
    )
    .unwrap();
    let yd = exec.to_host(&d_yd).unwrap();

    let err = rel_err(&yd, &yf);
    eprintln!("dp4a MXFP4 vs f32: rel_err {err:.2e} over {ff} outputs");
    assert!(err < 1e-2, "dp4a MXFP4 rel err {err} exceeds 1%");

    if std::env::var_os("PADDOCK_HEAVY_TESTS").is_some() {
        let iters = 400;
        for _ in 0..20 {
            exec.mxfp4_gemv_indexed(&gate.bytes, None, &d_idx, slot, &d_x, &mut d_yf, embd, ff)
                .unwrap();
        }
        exec.to_host(&d_yf).unwrap();
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            exec.mxfp4_gemv_indexed(&gate.bytes, None, &d_idx, slot, &d_x, &mut d_yf, embd, ff)
                .unwrap();
        }
        exec.to_host(&d_yf).unwrap();
        let f32_ms = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;
        for _ in 0..20 {
            exec.quantize_q8(&d_x, &mut xq, &mut xs, embd).unwrap();
            exec.mxfp4_gemv_indexed_dp4a(
                &gate.bytes,
                None,
                &d_idx,
                slot,
                &xq,
                &xs,
                &mut d_yd,
                embd,
                ff,
            )
            .unwrap();
        }
        exec.to_host(&d_yd).unwrap();
        let t1 = std::time::Instant::now();
        for _ in 0..iters {
            exec.quantize_q8(&d_x, &mut xq, &mut xs, embd).unwrap();
            exec.mxfp4_gemv_indexed_dp4a(
                &gate.bytes,
                None,
                &d_idx,
                slot,
                &xq,
                &xs,
                &mut d_yd,
                embd,
                ff,
            )
            .unwrap();
        }
        exec.to_host(&d_yd).unwrap();
        let dp4a_ms = t1.elapsed().as_secs_f64() * 1e3 / iters as f64;
        eprintln!(
            "MXFP4 GEMV: f32 {f32_ms:.4} ms | dp4a {dp4a_ms:.4} ms | speedup {:.2}×",
            f32_ms / dp4a_ms
        );
    }
}

/// Batched fused MoE (gate_up_batch + down_batch over B tokens) must equal
/// running the single-token fused MoE on each token independently - the MoE for
/// the batched decode forward.
#[test]
fn batched_moe_matches_per_token() {
    let Some(model) = common::model("PADDOCK_MODEL", common::GPT_OSS_20B) else {
        return;
    };
    let Some(exec) = common::gpu() else {
        return;
    };
    let map = MappedGguf::open(&model).expect("open gguf");
    let gate = exec.upload_raw(&map, "blk.0.ffn_gate_exps.weight").unwrap();
    let up = exec.upload_raw(&map, "blk.0.ffn_up_exps.weight").unwrap();
    let down = exec.upload_raw(&map, "blk.0.ffn_down_exps.weight").unwrap();
    let gate_b = exec.upload(&map, "blk.0.ffn_gate_exps.bias").unwrap();
    let up_b = exec.upload(&map, "blk.0.ffn_up_exps.bias").unwrap();
    let down_b = exec.upload(&map, "blk.0.ffn_down_exps.bias").unwrap();
    let (embd, ff) = (gate.dims[0], gate.dims[1]);
    // the plain fused kernels now read the repacked (data, scale) layout
    let gate_rp = exec.repack_mxfp4(&gate).unwrap();
    let up_rp = exec.repack_mxfp4(&up).unwrap();
    let down_rp = exec.repack_mxfp4(&down).unwrap();
    let n_active = 4usize;
    let batch = 4usize;

    // per-token activations, expert selections, weights
    let mut xrows = Vec::new();
    let mut idx = Vec::new();
    let mut wts = Vec::new();
    for b in 0..batch {
        xrows.extend(det(embd, 11 + b as u64));
        idx.extend([(b * 3) as u32 % 32, (b * 7 + 1) as u32 % 32, 2, 30]);
        wts.extend([0.4f32, 0.3, 0.2, 0.1]);
    }
    let d_x = exec.to_device(&xrows).unwrap();
    let d_idx = exec.stream.clone_htod(&idx).unwrap();
    let d_w = exec.to_device(&wts).unwrap();

    // batched
    let mut d_gu = exec.alloc(batch * n_active * ff).unwrap();
    let mut d_res = exec.alloc(batch * embd).unwrap(); // zeroed
    exec.mxfp4_moe_gate_up_batch(
        &gate,
        &gate_b.buf,
        &up,
        &up_b.buf,
        &d_idx,
        &d_x,
        &mut d_gu,
        embd,
        ff,
        n_active,
        batch,
        ALPHA,
        LIMIT,
    )
    .unwrap();
    exec.mxfp4_moe_down_batch(
        &down,
        &down_b.buf,
        &d_idx,
        &d_w,
        &d_gu,
        &mut d_res,
        ff,
        embd,
        n_active,
        batch,
    )
    .unwrap();
    let batched = exec.to_host(&d_res).unwrap();

    // per-token single fused MoE
    for b in 0..batch {
        let d_xb = exec.to_device(&xrows[b * embd..(b + 1) * embd]).unwrap();
        let d_idxb = exec
            .stream
            .clone_htod(&idx[b * n_active..(b + 1) * n_active].to_vec())
            .unwrap();
        let d_wb = exec
            .to_device(&wts[b * n_active..(b + 1) * n_active])
            .unwrap();
        let mut d_gub = exec.alloc(n_active * ff).unwrap();
        let mut d_resb = exec.alloc(embd).unwrap();
        exec.mxfp4_moe_gate_up(
            &gate_rp,
            &gate_b.buf,
            &up_rp,
            &up_b.buf,
            &d_idxb,
            &d_xb,
            &mut d_gub,
            embd,
            ff,
            n_active,
            ALPHA,
            LIMIT,
        )
        .unwrap();
        exec.mxfp4_moe_down(
            &down_rp,
            &down_b.buf,
            &d_idxb,
            &d_wb,
            &d_gub,
            &mut d_resb,
            ff,
            embd,
            n_active,
        )
        .unwrap();
        let single = exec.to_host(&d_resb).unwrap();
        let err = rel_err(&batched[b * embd..(b + 1) * embd], &single);
        eprintln!("token {b}: batched vs single rel_err {err:.2e}");
        assert!(err < 1e-5, "token {b}: {err} too high");
    }
}

#[test]
fn fused_moe_matches_unfused_chain() {
    let Some(model) = common::model("PADDOCK_MODEL", common::GPT_OSS_20B) else {
        return;
    };
    let Some(exec) = common::gpu() else {
        return;
    };
    let map = MappedGguf::open(&model).expect("open gguf");

    let gate = exec.upload_raw(&map, "blk.0.ffn_gate_exps.weight").unwrap();
    let up = exec.upload_raw(&map, "blk.0.ffn_up_exps.weight").unwrap();
    let down = exec.upload_raw(&map, "blk.0.ffn_down_exps.weight").unwrap();
    let gate_b = exec.upload(&map, "blk.0.ffn_gate_exps.bias").unwrap();
    let up_b = exec.upload(&map, "blk.0.ffn_up_exps.bias").unwrap();
    let down_b = exec.upload(&map, "blk.0.ffn_down_exps.bias").unwrap();

    let (embd, ff) = (gate.dims[0], gate.dims[1]);
    // the plain fused kernels now read the repacked (data, scale) layout
    let gate_rp = exec.repack_mxfp4(&gate).unwrap();
    let up_rp = exec.repack_mxfp4(&up).unwrap();
    let down_rp = exec.repack_mxfp4(&down).unwrap();
    let n_active = 4usize;
    let x = det(embd, 11);
    let d_x = exec.to_device(&x).unwrap();

    // route to experts {5, 17, 2, 30} with arbitrary normalized weights
    let idx: Vec<u32> = vec![5, 17, 2, 30];
    let w: Vec<f32> = vec![0.4, 0.3, 0.2, 0.1];
    let d_idx = exec.stream.clone_htod(&idx).unwrap();
    let d_w = exec.to_device(&w).unwrap();

    // ---- fused path
    let mut d_gate_up = exec.alloc(n_active * ff).unwrap();
    let mut fused_out = exec.alloc(embd).unwrap(); // residual starts at 0
    exec.mxfp4_moe_gate_up(
        &gate_rp,
        &gate_b.buf,
        &up_rp,
        &up_b.buf,
        &d_idx,
        &d_x,
        &mut d_gate_up,
        embd,
        ff,
        n_active,
        ALPHA,
        LIMIT,
    )
    .unwrap();
    exec.mxfp4_moe_down(
        &down_rp,
        &down_b.buf,
        &d_idx,
        &d_w,
        &d_gate_up,
        &mut fused_out,
        ff,
        embd,
        n_active,
    )
    .unwrap();
    let fused = exec.to_host(&fused_out).unwrap();

    // ---- unfused reference: per-expert gate/up GEMV + swiglu + down + scale_add
    let mut d_moe = exec.alloc(embd).unwrap(); // alloc_zeros
    let mut d_gate = exec.alloc(ff).unwrap();
    let mut d_up = exec.alloc(ff).unwrap();
    let mut d_down = exec.alloc(embd).unwrap();
    for slot in 0..n_active {
        exec.mxfp4_gemv_indexed(
            &gate.bytes,
            Some(&gate_b.buf),
            &d_idx,
            slot,
            &d_x,
            &mut d_gate,
            embd,
            ff,
        )
        .unwrap();
        exec.mxfp4_gemv_indexed(
            &up.bytes,
            Some(&up_b.buf),
            &d_idx,
            slot,
            &d_x,
            &mut d_up,
            embd,
            ff,
        )
        .unwrap();
        exec.swiglu_oai(&mut d_gate, &d_up, ff, ALPHA, LIMIT)
            .unwrap();
        exec.mxfp4_gemv_indexed(
            &down.bytes,
            Some(&down_b.buf),
            &d_idx,
            slot,
            &d_gate,
            &mut d_down,
            ff,
            embd,
        )
        .unwrap();
        exec.scale_add_dev(&mut d_moe, &d_down, &d_w, slot, embd)
            .unwrap();
    }
    let unfused = exec.to_host(&d_moe).unwrap();

    let err = rel_err(&fused, &unfused);
    eprintln!("fused vs unfused MoE: rel_err {err:.2e} over {embd} dims");
    assert!(err < 1e-4, "fused MoE rel err {err} too high");
}

/// The int8 mmq sorted MoE pair (gate_up_mmq + down_mmq) against the f32
/// CUDA-core sorted pair on real blk.0 expert weights and a prefill-shaped
/// batch. The mmq path quantizes the activations twice (input + swiglu
/// output), so the bar is the dp4a class bound (~1%), not f32 exactness -
/// same policy as `dp4a_mxfp4_close_to_f32`.
#[test]
fn sorted_mmq_moe_matches_sorted_f32() {
    let Some(model) = common::model("PADDOCK_MODEL", common::GPT_OSS_20B) else {
        return;
    };
    let Some(exec) = common::gpu() else {
        return;
    };
    let map = MappedGguf::open(&model).expect("open gguf");
    let gate = exec.upload_raw(&map, "blk.0.ffn_gate_exps.weight").unwrap();
    let up = exec.upload_raw(&map, "blk.0.ffn_up_exps.weight").unwrap();
    let down = exec.upload_raw(&map, "blk.0.ffn_down_exps.weight").unwrap();
    let gate_b = exec.upload(&map, "blk.0.ffn_gate_exps.bias").unwrap();
    let up_b = exec.upload(&map, "blk.0.ffn_up_exps.bias").unwrap();
    let down_b = exec.upload(&map, "blk.0.ffn_down_exps.bias").unwrap();
    let (embd, ff) = (gate.dims[0], gate.dims[1]);
    let gate_rp = exec.repack_mxfp4(&gate).unwrap();
    let up_rp = exec.repack_mxfp4(&up).unwrap();
    let down_rp = exec.repack_mxfp4(&down).unwrap();
    let (n_experts, n_active, batch) = (32usize, 4usize, 64usize);
    let bm = 32usize; // PD_MOE_BM / moe_align tile
    let max_blocks = (batch * n_active + n_experts * (bm - 1)).div_ceil(bm);

    // rows, expert picks (spread over all 32 experts), mix weights
    let x = det(batch * embd, 42);
    let mut idx = Vec::new();
    let mut wts = Vec::new();
    for b in 0..batch {
        idx.extend([
            (b % n_experts) as u32,
            ((b * 5 + 3) % n_experts) as u32,
            ((b * 11 + 7) % n_experts) as u32,
            ((b * 17 + 13) % n_experts) as u32,
        ]);
        wts.extend([0.4f32, 0.3, 0.2, 0.1]);
    }
    let d_x = exec.to_device(&x).unwrap();
    let d_idx = exec.stream.clone_htod(&idx).unwrap();
    let d_w = exec.to_device(&wts).unwrap();
    let mut d_sorted_row = exec.alloc_u32(max_blocks * bm).unwrap();
    let mut d_sorted_slot = exec.alloc_u32(max_blocks * bm).unwrap();
    let mut d_block_expert = exec.alloc_u32(max_blocks).unwrap();
    exec.moe_align(
        &d_idx,
        &mut d_sorted_row,
        &mut d_sorted_slot,
        &mut d_block_expert,
        batch,
        n_active,
        n_experts,
        max_blocks,
    )
    .unwrap();

    // f32 CUDA-core sorted reference
    let mut d_fused_f = exec.alloc(max_blocks * bm * ff).unwrap();
    let mut d_res_f = exec.alloc(batch * embd).unwrap(); // zeroed
    exec.mxfp4_moe_gate_up_gemm_sorted(
        &gate_rp,
        &gate_b.buf,
        &up_rp,
        &up_b.buf,
        &d_sorted_row,
        &d_block_expert,
        &d_x,
        &mut d_fused_f,
        embd,
        ff,
        max_blocks,
        ALPHA,
        LIMIT,
        false,
    )
    .unwrap();
    exec.mxfp4_moe_down_gemm_sorted(
        &down_rp,
        &down_b.buf,
        &d_sorted_row,
        &d_sorted_slot,
        &d_block_expert,
        &d_w,
        &d_fused_f,
        &mut d_res_f,
        ff,
        embd,
        n_active,
        max_blocks,
        false,
    )
    .unwrap();
    let res_f = exec.to_host(&d_res_f).unwrap();
    let fused_f = exec.to_host(&d_fused_f).unwrap();

    // The same f32 gate_up again on the tensor-core arm. There are three
    // implementations of this stage in the tree - f32 CUDA-core, f32
    // tensor-core, int8 mmq. Only mmq still serves (removed the
    // f32 pair's serving lanes with the exact-f32 pins); the f32 kernels
    // live on as this gate's independent plain-layout reference. A three-way
    // comparison says which one is the odd one out; comparing only two says
    // a disagreement exists and nothing about where it lives.
    let mut d_fused_tc = exec.alloc(max_blocks * bm * ff).unwrap();
    exec.mxfp4_moe_gate_up_gemm_sorted(
        &gate_rp,
        &gate_b.buf,
        &up_rp,
        &up_b.buf,
        &d_sorted_row,
        &d_block_expert,
        &d_x,
        &mut d_fused_tc,
        embd,
        ff,
        max_blocks,
        ALPHA,
        LIMIT,
        true,
    )
    .unwrap();
    let fused_tc = exec.to_host(&d_fused_tc).unwrap();

    // int8 mmq pair (gate_up emits the swiglu output already quantized).
    // LAYOUT: the mmq gate_up kernel streams the g||u INTERLEAVED plane
    // through the gate_data pointer alone - up_data is
    // never dereferenced (the gpt-oss loader keeps a 16-byte dummy there).
    // The f32 sorted pair above reads the plain per-plane layout, which is
    // why gate_rp/up_rp feed it directly. Handing the mmq kernel the plain
    // plane instead was this gate's own rel_err 2.11: the kernel
    // ILV-addresses whatever bytes it is given.
    let gu = exec
        .gu_interleave(&gate_rp, &up_rp, embd / 32, n_experts * ff)
        .unwrap();
    let RepackedMxfp4 {
        data: _gate_drop,
        scale: gate_scale,
    } = gate_rp;
    let RepackedMxfp4 {
        data: _up_drop,
        scale: up_scale,
    } = up_rp;
    let gate_ilv = RepackedMxfp4 {
        data: gu,
        scale: gate_scale,
    };
    let up_ilv = RepackedMxfp4 {
        data: exec.alloc_u8(16).unwrap(),
        scale: up_scale,
    };
    let mut xq = exec.alloc_i8(batch * embd).unwrap();
    let mut xs = exec.alloc(batch * embd / 32).unwrap();
    exec.quantize_q8(&d_x, &mut xq, &mut xs, batch * embd)
        .unwrap();
    let mut fq = exec.alloc_i8(max_blocks * bm * ff).unwrap();
    let mut fs = exec.alloc(max_blocks * bm * ff / 32).unwrap();
    exec.mxfp4_moe_gate_up_mmq(
        &gate_ilv,
        &gate_b.buf,
        &up_ilv,
        &up_b.buf,
        &d_sorted_row,
        &d_block_expert,
        &xq,
        &xs,
        &mut fq,
        &mut fs,
        embd,
        ff,
        max_blocks,
        ALPHA,
        LIMIT,
        1.0,
    )
    .unwrap();
    let mut d_part = exec.alloc(batch * n_active * embd).unwrap();
    let mut d_res_q = exec.alloc(batch * embd).unwrap(); // zeroed
    exec.mxfp4_moe_down_mmq(
        &down_rp,
        &down_b.buf,
        &d_sorted_row,
        &d_sorted_slot,
        &d_block_expert,
        &d_w,
        &fq,
        &fs,
        &mut d_part,
        ff,
        embd,
        n_active,
        max_blocks,
    )
    .unwrap();
    exec.moe_slot_combine(&d_part, &mut d_res_q, embd, n_active, batch)
        .unwrap();
    let res_q = exec.to_host(&d_res_q).unwrap();
    // determinism: the mmq down + fixed-order fold must be bit-reproducible
    // (the atomic-scatter design flipped near-tie greedy tokens run to run)
    let mut d_res_q2 = exec.alloc(batch * embd).unwrap();
    exec.mxfp4_moe_down_mmq(
        &down_rp,
        &down_b.buf,
        &d_sorted_row,
        &d_sorted_slot,
        &d_block_expert,
        &d_w,
        &fq,
        &fs,
        &mut d_part,
        ff,
        embd,
        n_active,
        max_blocks,
    )
    .unwrap();
    exec.moe_slot_combine(&d_part, &mut d_res_q2, embd, n_active, batch)
        .unwrap();
    let res_q2 = exec.to_host(&d_res_q2).unwrap();
    assert!(
        res_q
            .iter()
            .zip(&res_q2)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "mmq down + slot_combine must be deterministic"
    );
    exec.synchronize().unwrap();
    let fq_h: Vec<i8> = exec.stream.clone_dtoh(&fq).unwrap();
    let fs_h = exec.to_host(&fs).unwrap();

    // dequantized gate_up intermediate on real rows only (mmq zeroes PAD
    // rows, the f32 kernel leaves them stale - different by design)
    let srow = exec.to_host_u32(&d_sorted_row).unwrap();
    // The row filter is BLOCK_EXPERT, not sorted_row. moe_align PAD-fills
    // block_expert for every block but sorted_row only up to bacc*bm - the
    // tail past the used blocks is never written, and on a fresh (zeroed)
    // allocation it reads back as row 0, which is a perfectly valid token id.
    // Filtering on sorted_row therefore swept 224 rows of untouched scratch
    // into the comparison and made two agreeing kernels look uncorrelated
    // (rel_err 2.11). Every real consumer early-outs on block_expert; so does
    // this gate now.
    let bexp = exec.to_host_u32(&d_block_expert).unwrap();
    let mut fr = Vec::new();
    let mut ftc = Vec::new();
    let mut fqv = Vec::new();
    for (r, &t) in srow.iter().enumerate() {
        if bexp[r / bm] != u32::MAX && t != u32::MAX {
            fr.extend_from_slice(&fused_f[r * ff..(r + 1) * ff]);
            ftc.extend_from_slice(&fused_tc[r * ff..(r + 1) * ff]);
            fqv.extend((0..ff).map(|i| fq_h[r * ff + i] as f32 * fs_h[(r * ff + i) / 32]));
        }
    }
    let e_gu = rel_err(&fqv, &fr);
    let e_res = rel_err(&res_q, &res_f);
    eprintln!("sorted mmq MoE vs f32: gate_up rel_err {e_gu:.2e}, residual rel_err {e_res:.2e}");
    // A rel_err over 1 means "uncorrelated", not "imprecise", and the three
    // shapes it can take are worth telling apart without a rebuild: a constant
    // factor (norm ratio far from 1, samples proportional), a dead output (one
    // norm ~0), or a permutation (norms agree, samples do not).
    let l2 = |v: &[f32]| v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    eprintln!(
        "  rows real {} of {}, |mmq| {:.4e} |f32| {:.4e} ratio {:.4}",
        fr.len() / ff,
        srow.len(),
        l2(&fqv),
        l2(&fr),
        l2(&fqv) / l2(&fr).max(1e-30)
    );
    eprintln!("  mmq[0..6] {:?}", &fqv[..6.min(fqv.len())]);
    eprintln!("  f32[0..6] {:?}", &fr[..6.min(fr.len())]);
    eprintln!("  tc [0..6] {:?}", &ftc[..6.min(ftc.len())]);
    eprintln!(
        "  three-way: mmq-vs-cudacore {:.2e}, mmq-vs-tc {:.2e}, tc-vs-cudacore {:.2e}",
        rel_err(&fqv, &fr),
        rel_err(&fqv, &ftc),
        rel_err(&ftc, &fr)
    );
    // The reference's own health, asserted first so the failure order says
    // which side to look at: two independent f32 implementations of the same
    // math must agree to reduction-order noise. If this one goes red, the
    // reference moved and the mmq verdicts below mean nothing.
    let e_f32 = rel_err(&ftc, &fr);
    assert!(
        e_f32 < 1e-2,
        "the two f32 arms disagree ({e_f32}) - the reference itself moved"
    );
    assert!(e_gu < 1e-2, "gate_up mmq rel err {e_gu} exceeds 1%");
    assert!(e_res < 1e-2, "down mmq rel err {e_res} exceeds 1%");
}
/// The batched (grid.z = token) fused dp4a MoE pair against the single-token
/// launchers run per token - must be BIT-identical: the batched kernel at
/// token t executes the same warp shapes and per-row math as the b1 kernel on
/// t's inputs (token 0 literally is the b1 launch). This is what lets the
/// serving path use it at b=2..3 without leaving the B=1 numeric class.
#[test]
fn batched_dp4a_moe_matches_single() {
    let Some(model) = common::model("PADDOCK_MODEL", common::GPT_OSS_20B) else {
        return;
    };
    let Some(exec) = common::gpu() else {
        return;
    };
    let map = MappedGguf::open(&model).expect("open gguf");
    let gate = exec.upload_raw(&map, "blk.0.ffn_gate_exps.weight").unwrap();
    let up = exec.upload_raw(&map, "blk.0.ffn_up_exps.weight").unwrap();
    let down = exec.upload_raw(&map, "blk.0.ffn_down_exps.weight").unwrap();
    let gate_b = exec.upload(&map, "blk.0.ffn_gate_exps.bias").unwrap();
    let up_b = exec.upload(&map, "blk.0.ffn_up_exps.bias").unwrap();
    let down_b = exec.upload(&map, "blk.0.ffn_down_exps.bias").unwrap();
    let (embd, ff) = (gate.dims[0], gate.dims[1]);
    let gate_rp = exec.repack_mxfp4(&gate).unwrap();
    let up_rp = exec.repack_mxfp4(&up).unwrap();
    let down_rp = exec.repack_mxfp4(&down).unwrap();
    // Both dp4a gate_up arms stream the g||u ILV plane through gate_data
    // (up_data is a loader dummy) - same layout serving builds. Fed the plain
    // repacks instead, the two arms still bit-match (they misread the same
    // bytes the same way) but validate nothing real and read past the
    // plain-sized buffers (ILV pitch is 2x). 32 = n_experts.
    let gu = exec
        .gu_interleave(&gate_rp, &up_rp, embd / 32, 32 * ff)
        .unwrap();
    let RepackedMxfp4 {
        data: _gd,
        scale: gate_scale,
    } = gate_rp;
    let RepackedMxfp4 {
        data: _ud,
        scale: up_scale,
    } = up_rp;
    let gate_ilv = RepackedMxfp4 {
        data: gu,
        scale: gate_scale,
    };
    let up_ilv = RepackedMxfp4 {
        data: exec.alloc_u8(16).unwrap(),
        scale: up_scale,
    };
    let (n_active, batch) = (4usize, 3usize);

    let x = det(batch * embd, 77);
    let resid0 = det(batch * embd, 99); // pre-MoE residual rows
    let idx: Vec<u32> = vec![5, 17, 2, 30, 0, 31, 9, 12, 21, 3, 3, 28];
    let wts: Vec<f32> = vec![
        0.4, 0.3, 0.2, 0.1, 0.7, 0.1, 0.1, 0.1, 0.25, 0.25, 0.25, 0.25,
    ];
    let d_x = exec.to_device(&x).unwrap();
    let d_idx = exec.stream.clone_htod(&idx).unwrap();
    let d_w = exec.to_device(&wts).unwrap();

    // ---- batched pair
    let mut xq = exec.alloc_i8(batch * embd).unwrap();
    let mut xs = exec.alloc(batch * embd / 32).unwrap();
    exec.quantize_q8(&d_x, &mut xq, &mut xs, batch * embd)
        .unwrap();
    let mut d_gu = exec.alloc(batch * n_active * ff).unwrap();
    exec.mxfp4_moe_gate_up_dp4a_b(
        &gate_ilv,
        &gate_b.buf,
        &up_ilv,
        &up_b.buf,
        &d_idx,
        &xq,
        &xs,
        &mut d_gu,
        embd,
        ff,
        n_active,
        batch,
        ALPHA,
        LIMIT,
    )
    .unwrap();
    let mut fq = exec.alloc_i8(batch * n_active * ff).unwrap();
    let mut fs = exec.alloc(batch * n_active * ff / 32).unwrap();
    exec.quantize_q8(&d_gu, &mut fq, &mut fs, batch * n_active * ff)
        .unwrap();
    let mut d_res = exec.to_device(&resid0).unwrap();
    exec.mxfp4_moe_down_dp4a_b(
        &down_rp,
        &down_b.buf,
        &d_idx,
        &d_w,
        &fq,
        &fs,
        &mut d_res,
        ff,
        embd,
        n_active,
        batch,
    )
    .unwrap();
    let gu_b = exec.to_host(&d_gu).unwrap();
    let res_b = exec.to_host(&d_res).unwrap();

    // ---- single-token reference, one launch pair per token
    for t in 0..batch {
        let d_xt = exec.to_device(&x[t * embd..(t + 1) * embd]).unwrap();
        let d_it = exec
            .stream
            .clone_htod(&idx[t * n_active..(t + 1) * n_active])
            .unwrap();
        let d_wt = exec
            .to_device(&wts[t * n_active..(t + 1) * n_active])
            .unwrap();
        let mut xq1 = exec.alloc_i8(embd).unwrap();
        let mut xs1 = exec.alloc(embd / 32).unwrap();
        exec.quantize_q8(&d_xt, &mut xq1, &mut xs1, embd).unwrap();
        let mut d_gu1 = exec.alloc(n_active * ff).unwrap();
        exec.mxfp4_moe_gate_up_dp4a(
            &gate_ilv,
            &gate_b.buf,
            &up_ilv,
            &up_b.buf,
            &d_it,
            &xq1,
            &xs1,
            &mut d_gu1,
            embd,
            ff,
            n_active,
            ALPHA,
            LIMIT,
        )
        .unwrap();
        let mut fq1 = exec.alloc_i8(n_active * ff).unwrap();
        let mut fs1 = exec.alloc(n_active * ff / 32).unwrap();
        exec.quantize_q8(&d_gu1, &mut fq1, &mut fs1, n_active * ff)
            .unwrap();
        let mut d_res1 = exec.to_device(&resid0[t * embd..(t + 1) * embd]).unwrap();
        exec.mxfp4_moe_down_dp4a(
            &down_rp,
            &down_b.buf,
            &d_it,
            &d_wt,
            &fq1,
            &fs1,
            &mut d_res1,
            ff,
            embd,
            n_active,
        )
        .unwrap();
        let gu_1 = exec.to_host(&d_gu1).unwrap();
        let res_1 = exec.to_host(&d_res1).unwrap();
        let gu_bt = &gu_b[t * n_active * ff..(t + 1) * n_active * ff];
        let res_bt = &res_b[t * embd..(t + 1) * embd];
        assert!(
            gu_bt
                .iter()
                .zip(&gu_1)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "token {t}: batched gate_up_dp4a != single-token kernel"
        );
        assert!(
            res_bt
                .iter()
                .zip(&res_1)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "token {t}: batched down_dp4a != single-token kernel"
        );
    }
    eprintln!("batched dp4a MoE == single-token dp4a MoE (bitwise) over {batch} tokens");
}
