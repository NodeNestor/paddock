//! Bit-exactness gates for the deepseek-ocr decode launch folds:
//!
//! 1. `rope_qk_append_paged_ring` (ABI 384) vs the four-kernel chain
//!    rope(q) + rope(k) + append(k)@wpos + append(v)@wpos - q plane and both
//!    pool planes must be byte-identical, with the ring's two position
//!    streams DIVERGENT (rope by true pos, append at the write slot).
//! 2. `add_rmsnorm_quant_q8_batch` (ABI 385) vs add_rmsnorm_batch +
//!    quantize_q8 - x, xn, q, scales all bit-equal.
//! 3. `swiglu_quant_q8` (ABI 386) vs swiglu + quantize_q8 - q/scales
//!    bit-equal.
//!
//! Gated on a CUDA device + built pack (common::gpu()).

mod common;

use cudarc::driver::CudaSlice;
use paddock_engine::gpu::{GpuExecutor, KvDtype};
use paddock_kernels::reference::ops::YarnRope;

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

fn dev_f32(exec: &GpuExecutor, data: &[f32]) -> CudaSlice<f32> {
    exec.stream.clone_htod(data).expect("htod f32")
}

fn dev_u32(exec: &GpuExecutor, data: &[u32]) -> CudaSlice<u32> {
    exec.stream.clone_htod(data).expect("htod u32")
}

fn host_f32(exec: &GpuExecutor, d: &CudaSlice<f32>) -> Vec<f32> {
    exec.stream.clone_dtoh(d).expect("dtoh f32")
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// deepseek-ocr decode geometry: 10 q heads == 10 kv heads, head_dim 128.
const NH: usize = 10;
const NKV: usize = 10;
const HD: usize = 128;
const KV_DIM: usize = NKV * HD;

fn rope_params() -> (f32, f32, f32, f32, f32, f32) {
    YarnRope::new(HD, 10_000.0, 1.0, 4096, 0.0, 1.0, 32.0, 1.0).kernel_params()
}

#[test]
fn rope_qk_append_ring_matches_split_chain() {
    let Some(exec) = common::gpu() else {
        return;
    };
    if !exec.has_rope_qk_append_paged_ring() {
        common::missing("pack lacks rope_qk_append_paged_ring (slot 384)");
        return;
    }

    let batch = 8usize;
    let bps = 4usize; // blocks per slot
    let n_blocks = batch * bps;
    let pool_bytes = n_blocks * 16 * KV_DIM * 2; // f16 KV
    let rope = rope_params();

    // Ring-divergent streams: rows 0..4 pre-boundary (wpos == pos), rows 4..8
    // ring-engaged (wpos = pf + (pos - pf) % w with pf=16, w=32 -> wpos != pos).
    let pos: Vec<u32> = vec![3, 9, 17, 40, 55, 63, 70, 100];
    let wpos: Vec<u32> = pos
        .iter()
        .map(|&p| if p >= 48 { 16 + (p - 16) % 32 } else { p })
        .collect();
    assert!(
        wpos.iter().zip(&pos).any(|(w, p)| w != p),
        "test must exercise divergence"
    );
    let slots: Vec<u32> = (0..batch as u32).collect();
    // identity-ish block table: slot s block j -> physical block s*bps+j
    let bt: Vec<u32> = (0..batch)
        .flat_map(|s| (0..bps).map(move |j| (s * bps + j) as u32))
        .collect();

    let q_h = det(batch * NH * HD, 1);
    let k_h = det(batch * KV_DIM, 2);
    let v_h = det(batch * KV_DIM, 3);

    let d_pos = dev_u32(&exec, &pos);
    let d_wpos = dev_u32(&exec, &wpos);
    let d_slots = dev_u32(&exec, &slots);
    let d_bt = dev_u32(&exec, &bt);
    let d_v = dev_f32(&exec, &v_h);

    // path A: the split four-kernel chain
    let mut qa = dev_f32(&exec, &q_h);
    let mut ka = dev_f32(&exec, &k_h);
    let mut pk_a = exec.alloc_u8(pool_bytes).expect("pool_k A");
    let mut pv_a = exec.alloc_u8(pool_bytes).expect("pool_v A");
    exec.stream.memset_zeros(&mut pk_a).expect("zero");
    exec.stream.memset_zeros(&mut pv_a).expect("zero");
    exec.rope_yarn_batch(&mut qa, &d_pos, NH, HD, rope, batch)
        .expect("rope q");
    exec.rope_yarn_batch(&mut ka, &d_pos, NKV, HD, rope, batch)
        .expect("rope k");
    exec.kv_append_batch_paged(
        &ka,
        &mut pk_a,
        &d_wpos,
        Some(&d_slots),
        &d_bt,
        bps,
        KV_DIM,
        batch,
        KvDtype::Fp16,
    )
    .expect("append k");
    exec.kv_append_batch_paged(
        &d_v,
        &mut pv_a,
        &d_wpos,
        Some(&d_slots),
        &d_bt,
        bps,
        KV_DIM,
        batch,
        KvDtype::Fp16,
    )
    .expect("append v");

    // path B: the fused ring kernel (NEOX arm)
    let mut qb = dev_f32(&exec, &q_h);
    let mut kb = dev_f32(&exec, &k_h);
    let mut pk_b = exec.alloc_u8(pool_bytes).expect("pool_k B");
    let mut pv_b = exec.alloc_u8(pool_bytes).expect("pool_v B");
    exec.stream.memset_zeros(&mut pk_b).expect("zero");
    exec.stream.memset_zeros(&mut pv_b).expect("zero");
    exec.rope_qk_append_paged_ring(
        &mut qb,
        &mut kb,
        &d_v,
        &mut pk_b,
        &mut pv_b,
        &d_pos,
        &d_wpos,
        Some(&d_slots),
        &d_bt,
        bps,
        NH,
        NKV,
        HD,
        rope,
        batch,
        true,
        KvDtype::Fp16,
    )
    .expect("fused ring");

    let qa_h = host_f32(&exec, &qa);
    let qb_h = host_f32(&exec, &qb);
    assert_eq!(
        bits(&qa_h),
        bits(&qb_h),
        "roped q plane must be bit-identical"
    );
    let pk_a_h: Vec<u8> = exec.stream.clone_dtoh(&pk_a).expect("dtoh");
    let pk_b_h: Vec<u8> = exec.stream.clone_dtoh(&pk_b).expect("dtoh");
    assert_eq!(pk_a_h, pk_b_h, "K pool bytes must be bit-identical");
    let pv_a_h: Vec<u8> = exec.stream.clone_dtoh(&pv_a).expect("dtoh");
    let pv_b_h: Vec<u8> = exec.stream.clone_dtoh(&pv_b).expect("dtoh");
    assert_eq!(pv_a_h, pv_b_h, "V pool bytes must be bit-identical");
}

#[test]
fn add_rmsnorm_quant_q8_matches_chain() {
    let Some(exec) = common::gpu() else {
        return;
    };
    if !exec.has_add_rmsnorm_quant_q8() {
        common::missing("pack lacks add_rmsnorm_quant_q8_batch (slot 385)");
        return;
    }
    let embd = 1280usize;
    let eps = 1e-6f32;
    for batch in [1usize, 8, 96] {
        let x_h = det(batch * embd, 10 + batch as u64);
        let p_h = det(batch * embd, 20 + batch as u64);
        let w_h = det(embd, 30);
        let d_p = dev_f32(&exec, &p_h);
        let d_w = dev_f32(&exec, &w_h);

        // path A: add_rmsnorm_batch then quantize_q8
        let mut xa = dev_f32(&exec, &x_h);
        let mut oa = exec.alloc(batch * embd).expect("out A");
        let mut qa = exec.alloc_i8(batch * embd).expect("q A");
        let mut sa = exec.alloc(batch * embd / 32).expect("s A");
        exec.add_rmsnorm_batch(&mut xa, &d_p, &d_w, &mut oa, embd, eps, batch)
            .expect("chain norm");
        exec.quantize_q8(&oa, &mut qa, &mut sa, batch * embd)
            .expect("chain quant");

        // path B: the fused kernel
        let mut xb = dev_f32(&exec, &x_h);
        let mut ob = exec.alloc(batch * embd).expect("out B");
        let mut qb = exec.alloc_i8(batch * embd).expect("q B");
        let mut sb = exec.alloc(batch * embd / 32).expect("s B");
        exec.add_rmsnorm_quant_q8_batch(
            &mut xb, &d_p, &d_w, &mut ob, &mut qb, &mut sb, embd, eps, batch,
        )
        .expect("fused");

        let (xa_h, xb_h) = (host_f32(&exec, &xa), host_f32(&exec, &xb));
        assert_eq!(
            bits(&xa_h),
            bits(&xb_h),
            "residual write-back (batch {batch})"
        );
        let (oa_h, ob_h) = (host_f32(&exec, &oa), host_f32(&exec, &ob));
        assert_eq!(bits(&oa_h), bits(&ob_h), "normed plane (batch {batch})");
        let qa_h: Vec<i8> = exec.stream.clone_dtoh(&qa).expect("dtoh");
        let qb_h: Vec<i8> = exec.stream.clone_dtoh(&qb).expect("dtoh");
        assert_eq!(qa_h, qb_h, "int8 plane (batch {batch})");
        let (sa_h, sb_h) = (host_f32(&exec, &sa), host_f32(&exec, &sb));
        assert_eq!(bits(&sa_h), bits(&sb_h), "scales (batch {batch})");
    }
}

#[test]
fn swiglu_quant_q8_matches_chain() {
    let Some(exec) = common::gpu() else {
        return;
    };
    if !exec.has_swiglu_quant_q8() {
        common::missing("pack lacks swiglu_quant_q8 (slot 386)");
        return;
    }
    for n in [1280usize, 8 * 3584, 8 * 896 * 6] {
        assert_eq!(n % 32, 0);
        let g_h = det(n, 40);
        let u_h = det(n, 50);
        let d_u = dev_f32(&exec, &u_h);

        // path A: swiglu in place then quantize
        let mut ga = dev_f32(&exec, &g_h);
        let mut qa = exec.alloc_i8(n).expect("q A");
        let mut sa = exec.alloc(n / 32).expect("s A");
        exec.swiglu(&mut ga, &d_u, n).expect("chain swiglu");
        exec.quantize_q8(&ga, &mut qa, &mut sa, n)
            .expect("chain quant");

        // path B: fused, gate untouched
        let gb = dev_f32(&exec, &g_h);
        let mut qb = exec.alloc_i8(n).expect("q B");
        let mut sb = exec.alloc(n / 32).expect("s B");
        exec.swiglu_quant_q8(&gb, &d_u, &mut qb, &mut sb, n)
            .expect("fused");

        let qa_h: Vec<i8> = exec.stream.clone_dtoh(&qa).expect("dtoh");
        let qb_h: Vec<i8> = exec.stream.clone_dtoh(&qb).expect("dtoh");
        assert_eq!(qa_h, qb_h, "int8 plane (n {n})");
        let (sa_h, sb_h) = (host_f32(&exec, &sa), host_f32(&exec, &sb));
        assert_eq!(bits(&sa_h), bits(&sb_h), "scales (n {n})");
        // the fused form leaves gate raw - the plane must not have been
        // activated in place
        let gb_h = host_f32(&exec, &gb);
        assert_eq!(bits(&gb_h), bits(&g_h), "gate must stay unmodified");
    }
}
