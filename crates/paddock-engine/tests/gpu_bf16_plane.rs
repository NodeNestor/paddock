//! BF16 dense weight-plane parity on real bf16 weights.
//!
//! The engine serves bf16 tensors in their own class rather than
//! down-quantizing them at load (`Plane::Bf16` in gemma4/mod.rs). That
//! decision is only worth anything if the bf16 lane computes the right
//! numbers, so this gate checks all four entry points against an in-test host
//! reference:
//!
//!   1. `bf16_gemv`             -> host dot products (r == 1 decode arm)
//!   2. `bf16_gemm`             -> the same reference, r = 3 / 64 / 129
//!   3. `bf16_gemv_at`          -> gemv landing at a row offset (fused qkv row)
//!   4. `dequant_slice` (bf16)  -> exact widen of one embedding row
//!   5. `embed_gather_bf16`     -> gathered rows with the fused output scale
//!
//! 4 and 5 are widens and must be exact - bf16 -> f32 loses nothing. 1-3 are
//! reductions, so they are gated on relative error instead: the kernels
//! accumulate in f32 in a different order than the host loop.
//!
//! The host file is muse's **mmproj** (303 BF16 tensors) - see
//! `common::MUSE_GLIMMER_MMPROJ` for why it, and not a `UD-*_XL` model file.
//! Entries 4/5 are row-gather kernels, so they are exercised here against a
//! projection plane rather than a literal embedding table; nothing in either
//! kernel cares which the rows came from.
//!
//! Gated on: CUDA device + built pack + the Muse Glimmer mmproj download.

mod common;

use paddock_models::ggml_type::GgmlType;
use paddock_models::mapped::MappedGguf;

/// tiny LCG - no rand dep, fully reproducible
fn deterministic_input(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f32 / (1u64 << 31) as f32) - 0.5
        })
        .collect()
}

fn bf16_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
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

/// Host `y[r][o] = sum_k W[o][k] * x[r][k]` over the widened plane.
fn host_matmul(w: &[f32], x: &[f32], in_dim: usize, out_dim: usize, rows: usize) -> Vec<f32> {
    let mut y = vec![0f32; rows * out_dim];
    for r in 0..rows {
        for o in 0..out_dim {
            let mut acc = 0f32;
            for k in 0..in_dim {
                acc += w[o * in_dim + k] * x[r * in_dim + k];
            }
            y[r * out_dim + o] = acc;
        }
    }
    y
}

#[test]
fn bf16_plane_matches_host_reference() {
    let Some(path) = common::model("PADDOCK_MUSE_MMPROJ", common::MUSE_GLIMMER_MMPROJ) else {
        return;
    };
    let Some(exec) = common::gpu() else { return };
    // the lane is the point of the gate - a pack without it is a red flag,
    // not a skip (the whole reason common/mod.rs exists is gates that passed
    // having executed nothing)
    assert!(
        exec.has_bf16_dense(),
        "pack has no bf16 dense lane - rebuild packs/cuda"
    );
    let map = MappedGguf::open(&path).expect("open muse-glimmer mmproj");

    // A square tower plane: [1536, 1536]. Small enough that the O(n^3) host
    // reference below stays cheap at rows=129.
    let name = "v.blk.0.attn_k.weight";
    let (info, bytes) = map.tensor_bytes(name).expect("v.blk.0.attn_k present");
    assert_eq!(
        info.ggml_type,
        GgmlType::Bf16,
        "{name} is not bf16 - this gate's premise is that the mmproj ships bf16 planes"
    );
    let (in_dim, out_dim) = (info.dims[0] as usize, info.dims[1] as usize);
    let w_host = bf16_to_f32(bytes);
    assert_eq!(w_host.len(), in_dim * out_dim);

    let plane = exec.upload_raw(&map, name).expect("upload bf16 plane");
    assert_eq!(plane.dims, vec![in_dim, out_dim]);

    // ── 1/2: gemv and gemm against the same host reference. 2..=8 rides the
    // decode-band multi-row GEMV: 2/5/8 hit all three template
    // widths and 3/5 exercise the batch < NB rounded-up guard; 64/129 ride
    // the tensor-core prefill arm, which casts the f32
    // activations to bf16 in its smem stage - so the host reference for
    // those rows casts x the same way (products then match exactly and only
    // the summation order differs, the same 1e-5 class as the other rows).
    // 17/32 land in the decode-band BN=32/KT=128 tier (sweep);
    // 64/129 keep the narrow/fat prefill election
    for &rows in &[1usize, 2, 3, 5, 8, 17, 32, 64, 129] {
        let x = deterministic_input(rows * in_dim, 0x5eed_0000 + rows as u64);
        // round-to-nearest-even, matching __float2bfloat16 (truncation would
        // be off by one bf16 ulp on half the values)
        let x_ref: Vec<f32> = if rows > 8 {
            x.iter()
                .map(|v| {
                    let b = v.to_bits();
                    f32::from_bits(b.wrapping_add(0x7FFF + ((b >> 16) & 1)) & 0xFFFF_0000)
                })
                .collect()
        } else {
            x.clone()
        };
        let want = host_matmul(&w_host, &x_ref, in_dim, out_dim, rows);
        let xd = exec.stream.clone_htod(&x).expect("x htod");
        let mut yd = exec
            .stream
            .alloc_zeros::<f32>(rows * out_dim)
            .expect("y alloc");
        if rows == 1 {
            exec.bf16_gemv(&plane, None, &xd, &mut yd)
                .expect("bf16_gemv");
        } else {
            exec.bf16_gemm(&plane, None, &xd, &mut yd, rows)
                .expect("bf16_gemm");
        }
        let got = exec.to_host(&yd).expect("y dtoh");
        let err = rel_err(&got, &want);
        assert!(
            err < 1e-5,
            "bf16 plane rows={rows}: rel err {err:e} (want < 1e-5)"
        );
    }

    // ── 3: the offset-landing twin writes exactly where it is told and
    // touches nothing else (the fused [q|k|v] decode row's contract).
    {
        let x = deterministic_input(in_dim, 0xa11ce);
        let want = host_matmul(&w_host, &x, in_dim, out_dim, 1);
        let xd = exec.stream.clone_htod(&x).expect("x htod");
        let off = 512usize;
        let pad = 777f32;
        let mut yd = exec
            .stream
            .clone_htod(&vec![pad; off + out_dim + 64])
            .expect("y alloc");
        exec.bf16_gemv_at(&plane, &xd, &mut yd, off)
            .expect("bf16_gemv_at");
        let got = exec.to_host(&yd).expect("y dtoh");
        assert!(
            got[..off].iter().all(|v| *v == pad),
            "gemv_at wrote before its offset"
        );
        assert!(
            got[off + out_dim..].iter().all(|v| *v == pad),
            "gemv_at wrote past its segment"
        );
        let err = rel_err(&got[off..off + out_dim], &want);
        assert!(err < 1e-5, "bf16 gemv_at: rel err {err:e} (want < 1e-5)");
    }

    // ── 4/5: the row widen and gather. Both are EXACT.
    //
    // `mm.2.weight` is [4096, 6656] - 6656 rows of 4096. These two kernels are
    // row gathers, so a projection plane stands in for an embedding table
    // perfectly well; what is being checked is that a gathered bf16 row widens
    // bit-for-bit and picks up the fused scale.
    let embd = exec.upload_raw(&map, "mm.2.weight").expect("upload mm.2");
    assert_eq!(embd.ty, GgmlType::Bf16);
    let n_embd = embd.dims[0];
    let n_rows = embd.dims[1];
    let (_, e_bytes) = map.tensor_bytes("mm.2.weight").expect("mm.2 present");
    let row_bytes = embd.row_bytes(n_embd);
    assert_eq!(row_bytes, n_embd * 2, "bf16 row stride is 2 bytes/element");

    // first, last, and two interior rows
    let toks: Vec<u32> = vec![0, 1, (n_rows / 2) as u32, (n_rows - 1) as u32];
    for &t in &toks {
        let want = bf16_to_f32(&e_bytes[t as usize * row_bytes..(t as usize + 1) * row_bytes]);
        let mut d = exec.stream.alloc_zeros::<f32>(n_embd).expect("row alloc");
        exec.dequant_slice(&embd, t as usize * row_bytes, &mut d)
            .expect("bf16 dequant_slice");
        let got = exec.to_host(&d).expect("row dtoh");
        assert_eq!(
            got, want,
            "bf16 dequant_slice row {t} is not an exact widen"
        );
    }

    let scale = 0.75f32;
    let td = exec.stream.clone_htod(&toks).expect("tokens htod");
    let mut od = exec
        .stream
        .alloc_zeros::<f32>(toks.len() * n_embd)
        .expect("gather alloc");
    exec.embed_gather_bf16(&embd, &td, &mut od, n_embd, toks.len(), scale)
        .expect("embed_gather_bf16");
    let got = exec.to_host(&od).expect("gather dtoh");
    for (i, &t) in toks.iter().enumerate() {
        let want: Vec<f32> =
            bf16_to_f32(&e_bytes[t as usize * row_bytes..(t as usize + 1) * row_bytes])
                .iter()
                .map(|v| v * scale)
                .collect();
        assert_eq!(
            &got[i * n_embd..(i + 1) * n_embd],
            &want[..],
            "embed_gather_bf16 row {t} (scale {scale}) is not exact"
        );
    }
}
