//! Prefill-attention A/B at the pf8 serving shape: 2048-row span, qwen35
//! geometry (16 q-heads / 2 kv-heads GQA, head_dim 256), fp16 KV, paged with
//! an identity block table, spans at done=0 (ctx 2048) and done=2048 (ctx
//! 4096). PADDOCK_ATTN_PF_V2 is process-latched - run twice for the A/B.
use std::sync::Arc;

use paddock_engine::gpu::{GpuExecutor, KvDtype};

fn main() {
    let pack = std::env::var_os("PADDOCK_PACK")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../packs/cuda/build/pd-cuda-sm120.so")
        });
    let exec = Arc::new(GpuExecutor::new(0, &pack).expect("executor"));
    let (h, kvh, hd) = (16usize, 2usize, 256usize);
    let kv_dim = kvh * hd;
    let rows = 2048usize;
    let max_ctx = 8192usize;
    let scale = 1.0 / (hd as f32).sqrt();
    let fill = |n: usize, seed: u32| -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                ((s >> 8) as f32 / (1u32 << 24) as f32) - 0.5
            })
            .collect()
    };
    // fp16 KV pool, identity block table (page = 16 rows)
    let bps = max_ctx / 16;
    let kvf: Vec<f32> = fill(max_ctx * kv_dim, 7);
    let kvh16: Vec<u16> = kvf
        .iter()
        .map(|&v| half::f16::from_f32(v).to_bits())
        .collect();
    let d_k = exec.to_device_u8(bytemuck_cast(&kvh16)).expect("k");
    let d_v = exec.to_device_u8(bytemuck_cast(&kvh16)).expect("v");
    let bt: Vec<u32> = (0..bps as u32).collect();
    let d_bt = exec.to_device_u32(&bt).expect("bt");
    let sinks = vec![-1e30f32; h]; // no-sink semantics
    let d_sinks = exec.to_device(&sinks).expect("sinks");
    let d_q = exec.to_device(&fill(rows * h * hd, 3)).expect("q");
    let mut d_out = exec.alloc(rows * h * hd).expect("out");
    let d_slots = exec.to_device_u32(&vec![0u32; rows]).expect("slots");
    let mode = if std::env::var_os("PADDOCK_ATTN_PF_V2").is_some() {
        "v2"
    } else {
        "wmma"
    };
    for &done in &[0usize, 2048] {
        let pos: Vec<u32> = (done as u32..(done + rows) as u32).collect();
        let d_pos = exec.to_device_u32(&pos).expect("pos");
        for _ in 0..3 {
            exec.attn_prefill_f16_paged(
                &d_q,
                &d_k,
                &d_v,
                &d_sinks,
                &mut d_out,
                &d_pos,
                &d_slots,
                &d_bt,
                bps,
                h,
                kvh,
                hd,
                kv_dim,
                0,
                rows,
                scale,
                KvDtype::Fp16,
            )
            .expect("attn");
        }
        exec.synchronize().expect("sync");
        let t0 = std::time::Instant::now();
        let iters = 30;
        for _ in 0..iters {
            exec.attn_prefill_f16_paged(
                &d_q,
                &d_k,
                &d_v,
                &d_sinks,
                &mut d_out,
                &d_pos,
                &d_slots,
                &d_bt,
                bps,
                h,
                kvh,
                hd,
                kv_dim,
                0,
                rows,
                scale,
                KvDtype::Fp16,
            )
            .expect("attn");
        }
        exec.synchronize().expect("sync");
        let us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
        // causal FLOP: sum over rows of ctx(row) = rows*done + rows*(rows+1)/2
        let keys: f64 = rows as f64 * done as f64 + rows as f64 * (rows as f64 + 1.0) / 2.0;
        let flop = 4.0 * keys * (h * hd) as f64;
        println!(
            "{mode} done={done:4}: {us:8.1} us  {:.0} TF",
            flop / us / 1e6
        );
    }
}

fn bytemuck_cast(v: &[u16]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 2) }
}
