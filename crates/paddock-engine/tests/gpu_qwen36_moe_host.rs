//! MoE expert offload (PADDOCK_MOE_HOST) on the serial path: load the 35B-A3B
//! UD file with its routed experts host-mapped and greedy-generate. The
//! all-resident twin does not fit a 16 GB card, so this gate checks the path
//! runs and prints tokens + tok/s; bit-parity against resident is the
//! kernel-level gate in gpu_kquant_parity.
mod common;

use std::time::Instant;

use paddock_engine::gpu_model::qwen35::GpuQwen35;
use paddock_models::mapped::MappedGguf;

#[test]
fn moe_host_mapped_generates() {
    if !common::heavy() {
        return;
    }
    let Some(path) = common::model("QWEN36_MOE_UD_GGUF", common::QWEN36_35B_A3B_UD_Q4) else {
        return;
    };
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    // SAFETY: single-threaded test binary, set before any reader.
    unsafe { std::env::set_var("PADDOCK_MOE_HOST", "1") };
    let map = MappedGguf::open(&path).expect("open gguf");
    let t0 = Instant::now();
    let mut m = GpuQwen35::load(exec, &map, 2048).expect("load moe host-mapped");
    // PADDOCK_MOE_CACHE_SLOTS pins the count; default 64 per layer (5 GB)
    if paddock_engine::gpu::moe_cache_slots_pin().is_none() {
        // SAFETY: single-threaded test binary, set before any reader.
        unsafe { std::env::set_var("PADDOCK_MOE_CACHE_SLOTS", "64") };
    }
    let seated = m.enable_moe_cache(u64::MAX).expect("seat expert cache");
    eprintln!(
        "load {:.1}s, cache {seated} slots/layer",
        t0.elapsed().as_secs_f64()
    );
    let prompt: Vec<u32> = vec![760, 6511, 314, 9338, 369];
    let n = 32usize;
    let t1 = Instant::now();
    let out = m.generate_greedy(&prompt, n, None).expect("generate");
    let s = t1.elapsed().as_secs_f64();
    eprintln!("tokens {out:?}");
    eprintln!(
        "{n} tokens in {s:.2}s = {:.2} tok/s (incl. prefill + graph capture)",
        n as f64 / s
    );
    let t2 = Instant::now();
    let out2 = m.generate_greedy(&prompt, n, None).expect("generate 2");
    let s2 = t2.elapsed().as_secs_f64();
    eprintln!(
        "rerun: {:.2} tok/s, deterministic={}",
        n as f64 / s2,
        out == out2
    );
    if let Some((rows, misses)) = m.moe_cache_stats().expect("cache stats") {
        eprintln!(
            "expert cache: {rows} rows resolved, {misses} misses, hit rate {:.1}%",
            100.0 * (1.0 - misses as f64 / rows.max(1) as f64)
        );
    }
    assert_eq!(out, out2, "greedy must be deterministic run-to-run");
}
