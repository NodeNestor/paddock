//! Qwen3.8 DFlash2 bring-up smoke: load the target, sideload the drafter
//! GGUF, run the synthetic tap->fuse->ring->draft selftest. Exercises the whole
//! sideloader (81-tensor audit, geometry agreement, the `fc` band split, the
//! conv + selector planes), the ring + paged plumbing, and the captured
//! draft round - with deterministic pseudo-features, so it says nothing
//! about acceptance. That needs a live serve.
//!
//! Usage:
//!   QWEN_GGUF=/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q8_0.gguf \
//!   QWEN_DFLASH=/models/Qwen3.8-27B-DFlash2/Qwen3.8-27B-DFlash2-Q4_K_M.gguf \
//!   PADDOCK_PACK=packs/cuda/build/pd-cuda-sm120.so \
//!   cargo run --release --example qwen35_dflash
// A development probe: it runs on a box its author is looking at, and a
// failure should stop it where it happened rather than be reported.
#![allow(clippy::unwrap_used)]

use std::sync::Arc;

use paddock_engine::gpu::GpuExecutor;
use paddock_engine::gpu_model::qwen35::GpuQwen35;
use paddock_models::mapped::MappedGguf;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();
    let model = std::env::var("QWEN_GGUF").expect("set QWEN_GGUF");
    let drafter = std::env::var("QWEN_DFLASH").expect("set QWEN_DFLASH");
    let pack = std::env::var("PADDOCK_PACK").expect("set PADDOCK_PACK");
    let max_ctx: usize = std::env::var("MAX_CTX")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4096);
    let slots: usize = std::env::var("MAX_BATCH")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4);

    let map = MappedGguf::open(model.as_ref()).expect("open gguf");
    let t0 = std::time::Instant::now();
    let exec = Arc::new(GpuExecutor::new(0, pack.as_ref()).expect("executor"));
    let mut m = GpuQwen35::load(exec, &map, max_ctx).expect("load target");
    eprintln!("target loaded in {:.1}s", t0.elapsed().as_secs_f32());

    let t1 = std::time::Instant::now();
    m.attach_dflash(std::path::Path::new(&drafter))
        .expect("attach drafter");
    eprintln!("drafter attached in {:.1}s", t1.elapsed().as_secs_f32());

    // the rings size off the slot count, so the batch lane comes first
    m.enable_batch(slots).expect("enable_batch");

    let r = m.dflash_selftest().expect("selftest");
    println!("drafts        : {:?}", r.drafts);
    println!("repeat-identical: {}", r.repeat_identical);
    println!("ms/round      : {:.3}", r.ms_per_round);
    assert!(
        r.repeat_identical,
        "two identical rounds disagreed - ring append race"
    );
}
