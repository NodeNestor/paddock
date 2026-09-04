//! Gemma 4 decode-tick bench: the serving hot path (forward_batch_sampled,
//! graph-replayed) timed over N steps at batch R. When profiling this, pass
//! --cuda-graph-trace=node or the graph ticks vanish from the trace and the
//! busy% lies.
//!
//! Usage: GEMMA4_GGUF=... PADDOCK_PACK=... [R=1] [N=64] gemma4_decode_bench
// A development probe: it runs on a box its author is looking at, and a
// failure should stop it where it happened rather than be reported.
#![allow(clippy::unwrap_used)]

use std::sync::Arc;

use paddock_engine::generator::{Generator, RowSample};
use paddock_engine::gpu::GpuExecutor;
use paddock_engine::gpu_model::gemma4::GpuGemma4;
use paddock_engine::sampler::DevicePlan;
use paddock_models::mapped::MappedGguf;

fn main() {
    let model = std::env::var("GEMMA4_GGUF").expect("GEMMA4_GGUF");
    let pack = std::env::var("PADDOCK_PACK").expect("PADDOCK_PACK");
    let r: usize = std::env::var("R")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    let n: usize = std::env::var("N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(64);

    let map = MappedGguf::open(model.as_ref()).expect("open");
    let exec = Arc::new(GpuExecutor::new(0, pack.as_ref()).expect("executor"));
    let mctx: usize = std::env::var("MAX_CTX")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4096);
    let mut m = GpuGemma4::load(exec, &map, mctx).expect("load");
    let slots: usize = std::env::var("SLOTS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(r);
    let cap = m.enable_batch(slots).expect("enable_batch");
    assert!(cap >= r);

    // PROMPT_LEN synthetic tokens per slot (realistic decode context)
    let plen: usize = std::env::var("PROMPT_LEN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4);
    let mut prompt: Vec<u32> = vec![2];
    prompt.extend((0..plen.saturating_sub(1)).map(|i| 9259 + (i as u32 % 1000)));
    // per-slot salted prompts (serving-shape: no prefix reuse across slots)
    let salted: Vec<Vec<u32>> = (0..r)
        .map(|s| {
            let mut p = prompt.clone();
            for (j, t) in p.iter_mut().enumerate().skip(1) {
                *t = 9259 + ((j + s * 331) as u32 % 1000);
            }
            p
        })
        .collect();
    let tp = std::time::Instant::now();
    if std::env::var_os("BATCH_PF").is_some() {
        // the serving path: coalesced multi-prompt pass; WAVES>1 replays
        // fresh salted cohorts to expose cross-wave state degradation
        let waves: usize = std::env::var("WAVES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1);
        for w in 0..waves {
            let items: Vec<(usize, Vec<u32>)> = (0..r)
                .map(|s| {
                    let mut p = salted[s].clone();
                    for t in p.iter_mut().skip(1) {
                        *t = (*t + (w as u32 * 7919)) % 200000 + 1000;
                    }
                    (s, p)
                })
                .collect();
            let t1 = std::time::Instant::now();
            m.forward_prefill_batch(&items).expect("batch prefill");
            eprintln!(
                "wave {w}: batch prefill x{r}: {:.1} ms",
                t1.elapsed().as_secs_f32() * 1000.0
            );
        }
    } else {
        for (slot, s) in salted.iter().enumerate() {
            let t1 = std::time::Instant::now();
            m.forward_prefill(slot, s).expect("prefill");
            eprintln!(
                "prefill slot {slot}: {:.1} ms",
                t1.elapsed().as_secs_f32() * 1000.0
            );
        }
    }
    eprintln!(
        "prefill total: {:.1} ms = {:.0} tok/s",
        tp.elapsed().as_secs_f32() * 1000.0,
        (r * plen) as f32 / tp.elapsed().as_secs_f32()
    );

    let mut tokens = vec![9259u32; r];
    let mut positions: Vec<u32> = vec![prompt.len() as u32; r];
    let plans: Vec<RowSample> = (0..r)
        .map(|_| RowSample::Device(DevicePlan::Greedy))
        .collect();

    // warmup (captures the graph)
    for _ in 0..4 {
        let s = m
            .forward_batch_sampled(&tokens, &positions, &plans)
            .expect("step");
        tokens[..r].copy_from_slice(&s.ids[..r]);
        for p in positions.iter_mut() {
            *p += 1;
        }
    }
    let t0 = std::time::Instant::now();
    for _ in 0..n {
        let s = m
            .forward_batch_sampled(&tokens, &positions, &plans)
            .expect("step");
        tokens[..r].copy_from_slice(&s.ids[..r]);
        for p in positions.iter_mut() {
            *p += 1;
        }
    }
    let dt = t0.elapsed().as_secs_f32();
    println!(
        "R={r} N={n}: {:.2} ms/step, {:.1} tok/s aggregate",
        dt * 1000.0 / n as f32,
        (r * n) as f32 / dt
    );
}
