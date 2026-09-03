//! Muse Glimmer DFlash bring-up smoke: load the target,
//! sideload the drafter GGUF, then run the synthetic tap->fuse->ring->draft
//! selftest. Exercises the whole sideloader (58-tensor audit, geometry
//! agreement with the target, the `fc` superblock band split), the ring +
//! paged plumbing, both drafter passes, and the captured draft round -
//! driving the ring with deterministic pseudo-features rather than real
//! ones, so it says nothing about acceptance. That needs a live serve.
//!
//! Usage:
//!   MUSE_GGUF=/models/Muse-Glimmer-30B-GGUF/Muse-Glimmer-30B-Q8_0.gguf \
//!   MUSE_DFLASH=/models/Muse-Glimmer-30B-GGUF/dflash-kquant.gguf \
//!   PADDOCK_PACK=packs/cuda/build/pd-cuda-sm120.so \
//!   cargo run --release --example muse_dflash

use std::sync::Arc;

use paddock_engine::generator::Generator;
use paddock_engine::gpu::GpuExecutor;
use paddock_engine::gpu_model::gemma4::GpuGemma4;
use paddock_models::mapped::MappedGguf;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();
    let model = std::env::var("MUSE_GGUF").expect("set MUSE_GGUF");
    let drafter = std::env::var("MUSE_DFLASH").expect("set MUSE_DFLASH");
    let pack = std::env::var("PADDOCK_PACK").expect("set PADDOCK_PACK");
    let max_ctx: usize = std::env::var("MAX_CTX")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8192);
    let slots: usize = std::env::var("MAX_BATCH")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4);

    let map = MappedGguf::open(model.as_ref()).expect("open gguf");
    let t0 = std::time::Instant::now();
    let exec = Arc::new(GpuExecutor::new(0, pack.as_ref()).expect("executor"));
    let mut m = GpuGemma4::load(exec, &map, max_ctx).expect("load target");
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
