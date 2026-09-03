//! Print the resident breakdown for a real model file, so a "won't fit"
//! verdict can be read rather than guessed at.
//!
//!   cargo run -p paddock-estimator --example fit_breakdown -- <gguf> [budget_gib] [batch]
//!
//! `resident` is what the fit is judged on - the KV pool is elastic and sizes
//! itself to what is left, so it never decides whether a model fits. That
//! distinction is exactly what this dump exists to make visible.

use paddock_estimator::{Device, Envelope, KvDtype, ModelKind, ModelShape, estimate};

fn gib(b: u64) -> f64 {
    b as f64 / (1u64 << 30) as f64
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = std::path::Path::new(args.get(1).expect("usage: <gguf> [budget_gib] [batch]"));
    let budget: f64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(20.0);
    let batch: u64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(4);

    let report = paddock_models::probe::probe_path(path).expect("probe");
    let weight_bytes = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    let shape = ModelShape::from_report(&report, weight_bytes, ModelKind::Generative);
    let dev = Device {
        free_bytes: (budget * (1u64 << 30) as f64) as u64,
        total_bytes: 48 << 30,
    };

    for (label, spec, offload) in [
        ("spec off, no offload", None, None),
        (
            "spec ON,  no offload",
            Some(paddock_estimator::SpecCost::default()),
            None,
        ),
        (
            "spec off, offload ON",
            None,
            Some(paddock_estimator::OffloadCost::armed(24 << 30)),
        ),
    ] {
        let env = Envelope {
            concurrency: batch,
            kv_dtype: KvDtype::F16,
            spec,
            offload,
        };
        let e = estimate(&shape, &env, &dev);
        println!("\n=== {label} (budget {budget} GiB, batch {batch}) ===");
        println!("  weights   {:>8.2} GiB", gib(e.weights));
        println!("  tower     {:>8.2}", gib(e.tower));
        println!("  workspace {:>8.2}", gib(e.workspace));
        println!(
            "  state     {:>8.2}  <- per-slot, FLAT in context",
            gib(e.state)
        );
        println!("  overhead  {:>8.2}  <- itemized below", gib(e.overhead));
        let o = e.overhead_parts;
        println!(
            "     slack  {:>8.2}  (a measured 8% of the planes)",
            gib(o.allocator_slack)
        );
        println!(
            "     ckpt   {:>8.2}  <- self-sized prefix checkpoints + staging",
            gib(o.prefix_checkpoints)
        );
        println!("     conv   {:>8.2}", gib(o.conv_scratch));
        println!("     logits {:>8.2}", gib(o.logits));
        println!("     btable {:>8.2}", gib(o.block_tables));
        println!(
            "     spec   {:>8.2}  <- draft state + draft logits",
            gib(o.spec_state)
        );
        println!(
            "     stage  {:>8.2}  <- kv-offload staging",
            gib(o.offload_staging)
        );
        println!(
            "  ckpt pool {:>8.2}  <- above the floor; comes off the KV pool",
            gib(o.prefix_pool_extra)
        );
        println!(
            "  fixed     {:>8.2}  <- CUDA context + graph margin (flat)",
            gib(o.fixed)
        );
        println!("  ---------------------");
        println!(
            "  resident  {:>8.2}  <- what the fit is judged on",
            gib(e.resident)
        );
        println!(
            "  kv_pool   {:>8.2}  <- elastic; never decides the verdict",
            gib(e.kv_pool)
        );
        println!("  host_ram  {:>8.2}  <- NOT vram", gib(e.host_ram));
        println!("  max_ctx   {:>8}  ({:?})", e.max_ctx, e.limited_by);
        println!("  verdict   {:?}", e.fit);
    }
}
