//! DFlash stage-B smoke: load the target + sideload the drafter, run the
//! synthetic ring/draft selftest.
//! Validates the whole sideloader (57-tensor audit, bf16->f16 range check,
//! fused qkv + fc band splits), the ring/paged plumbing, and both drafter
//! forwards end to end; prints the eager round cost. Real-feature acceptance
//! needs the aux-feature capture - this probe drives the ring with
//! deterministic pseudo-features.
//!
//! Usage: LAGUNA_GGUF=<path>\Laguna-XS-2.1-Q4_K_M.gguf
//!        LAGUNA_DFLASH=<path to the DFlash drafter dir>
//!        PADDOCK_PACK=packs\cuda\build\pd-cuda-sm86.dll
//!        cargo run --release --example laguna_dflash

use std::sync::Arc;

use paddock_engine::generator::Generator;
use paddock_engine::gpu::GpuExecutor;
use paddock_engine::gpu_model::laguna::GpuLaguna;
use paddock_models::mapped::MappedGguf;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();
    let model = std::env::var("LAGUNA_GGUF").expect("set LAGUNA_GGUF");
    let drafter = std::env::var("LAGUNA_DFLASH").expect("set LAGUNA_DFLASH");
    let pack = std::env::var("PADDOCK_PACK").expect("set PADDOCK_PACK");
    let max_ctx: usize = std::env::var("MAX_CTX")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4096);

    let map = MappedGguf::open(model.as_ref()).expect("open gguf");
    let t0 = std::time::Instant::now();
    let exec = Arc::new(GpuExecutor::new(0, pack.as_ref()).expect("executor"));
    let mut m = GpuLaguna::load(exec, &map, max_ctx).expect("load target");
    eprintln!("target loaded in {:.1}s", t0.elapsed().as_secs_f32());

    let t1 = std::time::Instant::now();
    m.attach_dflash(drafter.as_ref()).expect("attach dflash");
    eprintln!("drafter attached in {:.1}s", t1.elapsed().as_secs_f32());

    let slots = m.enable_batch(4).expect("enable batch");
    assert!(slots > 1, "batch lane didn't enable (slots {slots})");

    let r = m.dflash_selftest().expect("selftest");
    eprintln!(
        "dflash selftest: {} drafts {:?}\n  repeat identical: {}\n  eager draft round: {:.2} ms",
        r.drafts.len(),
        r.drafts,
        r.repeat_identical,
        r.ms_per_round
    );
    assert!(
        r.repeat_identical,
        "draft rounds not deterministic - ring append race?"
    );
    eprintln!("stage B smoke PASSED");
}
