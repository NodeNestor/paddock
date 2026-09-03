//! Isolate-at-geometry timing probe for the OCR decode-attention split chain.
//! Profiler spans are not trustworthy for this chain: PDL-dependent
//! spans bill the predecessor wait as duration (the pdl-duration-inflation
//! trap). This probe launches the exact serve-geometry chain (deepseek-ocr:
//! 10q/10kv hd128 f16-KV, ~1013-row R-SWA depth, 16 splits) back-to-back
//! with event-free wall timing around a synchronized loop, no PDL
//! predecessor to wait on - the number is the kernel work.
//!
//! `#[ignore]`d: a measurement tool, not a gate. Run with
//!   PADDOCK_PACK=<.so> cargo test --release -p paddock-engine \
//!     --test gpu_ocr_attn_isolate -- --ignored --nocapture

mod common;

use cudarc::driver::CudaSlice;
use half::f16;
use paddock_engine::gpu::{GpuExecutor, KvDtype};

fn f16_dev_u8(exec: &GpuExecutor, data: &[f32]) -> CudaSlice<u8> {
    let bytes: Vec<u8> = data
        .iter()
        .flat_map(|&x| f16::from_f32(x).to_le_bytes())
        .collect();
    exec.stream.clone_htod(&bytes).expect("f16 bytes htod")
}

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
#[ignore = "timing probe, not a gate - run with --ignored --nocapture"]
fn ocr_decode_attn_chain_at_serve_geometry() {
    let Some(exec) = common::gpu() else {
        return;
    };
    let (n_heads, n_kv_heads, head_dim) = (10usize, 10usize, 128usize);
    let kv_dim = n_kv_heads * head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();
    // the c8 board cell's R-SWA depth: ~1013 live KV rows per slot
    let pos = 1013u32;
    let max_ctx = 1024usize;
    let bps = max_ctx / 16;
    let n_splits = 16usize; // attn_splits_for(10, batch<=8, 188 SMs) = 16

    for &batch in &[1usize, 6, 8] {
        let qdim = n_heads * head_dim;
        let q = det(batch * qdim, 1);
        let kc = det(batch * max_ctx * kv_dim, 2);
        let vc = det(batch * max_ctx * kv_dim, 3);
        let d_q = exec.to_device(&q).expect("q");
        let d_k = f16_dev_u8(&exec, &kc);
        let d_v = f16_dev_u8(&exec, &vc);
        let d_sinks = exec.alloc(n_heads).expect("sinks");
        let positions = vec![pos; batch];
        let d_pos = exec.stream.clone_htod(&positions).expect("pos");
        let bt_host: Vec<u32> = (0..(batch * bps) as u32).collect();
        let d_bt = exec.stream.clone_htod(&bt_host).expect("bt");
        let mut d_o = exec
            .alloc(n_heads * batch * n_splits * head_dim)
            .expect("o");
        let mut d_ml = exec.alloc(n_heads * batch * n_splits * 2).expect("ml");
        let mut d_out = exec.alloc(batch * qdim).expect("out");

        let chain =
            |d_o: &mut CudaSlice<f32>, d_ml: &mut CudaSlice<f32>, d_out: &mut CudaSlice<f32>| {
                exec.attn_partial_batch_paged(
                    &d_q,
                    &d_k,
                    &d_v,
                    d_o,
                    d_ml,
                    &d_pos,
                    None,
                    &d_bt,
                    bps,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    kv_dim,
                    0,
                    n_splits,
                    batch,
                    scale,
                    KvDtype::Fp16,
                )
                .expect("partial");
                exec.attn_combine_batch(
                    d_o, d_ml, &d_sinks, d_out, n_heads, head_dim, n_splits, batch,
                )
                .expect("combine");
            };

        for _ in 0..50 {
            chain(&mut d_o, &mut d_ml, &mut d_out);
        }
        exec.synchronize().expect("warmup sync");
        let reps = 2000usize;
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            chain(&mut d_o, &mut d_ml, &mut d_out);
        }
        exec.synchronize().expect("timed sync");
        let chain_us = t0.elapsed().as_secs_f64() * 1e6 / reps as f64;

        // partial alone, for the split vs combine attribution
        exec.synchronize().expect("pre sync");
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            exec.attn_partial_batch_paged(
                &d_q,
                &d_k,
                &d_v,
                &mut d_o,
                &mut d_ml,
                &d_pos,
                None,
                &d_bt,
                bps,
                n_heads,
                n_kv_heads,
                head_dim,
                kv_dim,
                0,
                n_splits,
                batch,
                scale,
                KvDtype::Fp16,
            )
            .expect("partial");
        }
        exec.synchronize().expect("partial sync");
        let partial_us = t0.elapsed().as_secs_f64() * 1e6 / reps as f64;

        // byte floor: each row reads its slot's K+V once
        let bytes = batch as f64 * pos as f64 * kv_dim as f64 * 2.0 * 2.0;
        let floor_us = bytes / 1531e9 * 1e6;
        eprintln!(
            "batch {batch} splits {n_splits}: chain {chain_us:.1} us (partial {partial_us:.1} + \
             combine {:.1}) | bytes {:.1} MB floor {floor_us:.1} us -> {:.0}% of floor",
            chain_us - partial_us,
            bytes / 1e6,
            floor_us / chain_us * 100.0
        );
    }
}
