//! Report gemma4 KV budget: per-slot bytes and the enable_batch slot count
//! at a given context - the paged-SWA WindowRing's memory win made visible.
//!
//! Usage: GEMMA4_GGUF=... PADDOCK_PACK=... [MAX_CTX=8192] [ASK=32] gemma4_kv_budget
// A development probe: it runs on a box its author is looking at, and a
// failure should stop it where it happened rather than be reported.
#![allow(clippy::unwrap_used)]

use std::sync::Arc;

use paddock_engine::generator::Generator;
use paddock_engine::gpu::GpuExecutor;
use paddock_engine::gpu_model::gemma4::GpuGemma4;
use paddock_models::mapped::MappedGguf;

fn main() {
    let model = std::env::var("GEMMA4_GGUF").expect("set GEMMA4_GGUF");
    let pack = std::env::var("PADDOCK_PACK").expect("set PADDOCK_PACK");
    let max_ctx: usize = std::env::var("MAX_CTX")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8192);
    let ask: usize = std::env::var("ASK")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(32);

    let map = MappedGguf::open(model.as_ref()).expect("open gguf");
    let exec = Arc::new(GpuExecutor::new(0, pack.as_ref()).expect("executor"));
    let mut m = GpuGemma4::load(exec, &map, max_ctx).expect("load");
    let slots = m.enable_batch(ask).expect("enable_batch");
    println!(
        "max_ctx={max_ctx} paging={} asked={ask} enabled_slots={slots}",
        std::env::var_os("PADDOCK_NO_PAGED_KV").is_none()
    );
}
