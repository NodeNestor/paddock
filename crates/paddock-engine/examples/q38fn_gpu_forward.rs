//! GPU forward / greedy generation for Qwen3.8-Flash-Next - the triage twin of
//! `q38fn_host_forward`, so a question ("what does the model predict after
//! these ids?") can be asked of either side with the same arguments.
//!
//! Usage:
//!   q38fn_gpu_forward --dir <ckpt> --ids 760,6511,... [--topk 8] [--gen N]
//!                     [--split K]
//!
//! `--gen N` greedily continues for N tokens. `--split K` prefills only the
//! first K ids and steps the rest, which is how the prefill/decode equality
//! gets exercised by hand.
// A development probe: it runs on a box its author is looking at, and a
// failure should stop it where it happened rather than be reported.
#![allow(clippy::unwrap_used)]

use std::sync::Arc;

use paddock_engine::gpu::GpuExecutor;
use paddock_engine::gpu_model::qwen4exp::Qwen4ExpGpu;

/// O(n) argmax. The greedy loop must not use `top_k`: a full sort of the
/// 248320-wide vocab costs ~9.6 ms per step and lands inside the decode timing
/// window, which inflated every c1 number this lane has ever reported.
fn argmax1(logits: &[f32]) -> usize {
    let mut best = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &x) in logits.iter().enumerate() {
        if x > bv {
            bv = x;
            best = i;
        }
    }
    best
}

fn top_k(logits: &[f32], k: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]));
    order.truncate(k);
    order
}

fn main() {
    let (mut dir, mut pack) = (None, None);
    let mut ids: Vec<u32> = Vec::new();
    let (mut topk, mut n_gen, mut split) = (8usize, 0usize, 0usize);
    let mut batch = 0usize;
    let mut widths_arg: Vec<usize> = Vec::new();
    let mut pfsweep: Vec<usize> = Vec::new();
    let mut maxctx = 512usize;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--dir" => dir = args.next(),
            "--pack" => pack = args.next(),
            "--ids" => {
                ids = args
                    .next()
                    .unwrap()
                    .split(',')
                    .map(|s| s.parse().unwrap())
                    .collect()
            }
            "--topk" => topk = args.next().unwrap().parse().unwrap(),
            "--gen" => n_gen = args.next().unwrap().parse().unwrap(),
            "--split" => split = args.next().unwrap().parse().unwrap(),
            "--batch" => batch = args.next().unwrap().parse().unwrap(),
            "--maxctx" => maxctx = args.next().unwrap().parse().unwrap(),
            "--pfsweep" => {
                pfsweep = args
                    .next()
                    .unwrap()
                    .split(',')
                    .map(|s| s.parse().unwrap())
                    .collect()
            }
            "--widths" => {
                widths_arg = args
                    .next()
                    .unwrap()
                    .split(',')
                    .map(|s| s.parse().unwrap())
                    .collect()
            }
            other => panic!("unknown arg {other}"),
        }
    }
    let dir = std::path::PathBuf::from(dir.expect("--dir required"));
    assert!(!ids.is_empty(), "--ids required");
    let pack = pack
        .map(std::path::PathBuf::from)
        .or_else(|| std::env::var_os("PADDOCK_PACK").map(std::path::PathBuf::from))
        .expect("--pack or PADDOCK_PACK required");

    let exec = Arc::new(GpuExecutor::new(0, &pack).expect("gpu executor"));
    let t0 = std::time::Instant::now();
    let mut m = Qwen4ExpGpu::load(&exec, &dir, maxctx).expect("load");
    eprintln!("loaded in {:.1}s", t0.elapsed().as_secs_f32());

    if !pfsweep.is_empty() {
        // Prefill amortization curve off one load: how a fused wave of N rows
        // compares with N/128 serial 128-row prefills is the whole question
        // behind `forward_prefill_batch`, and the single-sequence walk answers
        // the weight-amortization half of it directly.
        for &n in &pfsweep {
            let n = n.min(ids.len());
            // two legs, report the second: the first at a new length pays any
            // lazy warm-up in the walk
            for leg in 0..2 {
                let t = std::time::Instant::now();
                let _ = m.forward_prompt(&ids[..n]).expect("prefill");
                let dt = t.elapsed().as_secs_f64();
                if leg == 1 {
                    eprintln!(
                        "prefill {n:5} tok: {:8.2} ms  {:9.1} tok/s  {:6.4} ms/row",
                        dt * 1e3,
                        n as f64 / dt,
                        dt * 1e3 / n as f64
                    );
                }
            }
        }
        return;
    }
    let split = if split == 0 {
        ids.len()
    } else {
        split.min(ids.len())
    };
    let t_pf = std::time::Instant::now();
    let mut logits = m.forward_prompt(&ids[..split]).expect("prefill");
    let pf = t_pf.elapsed().as_secs_f64();
    eprintln!(
        "prefill {split} tok in {:.1} ms ({:.1} tok/s)",
        pf * 1e3,
        split as f64 / pf
    );
    for &id in &ids[split..] {
        logits = m.decode_step(id).expect("decode step");
    }

    println!(
        "gpu top-{topk} after {} ids (prefill {split} + {} steps):",
        ids.len(),
        ids.len() - split
    );
    for i in top_k(&logits, topk) {
        println!("  token {i}  logit {:.4}", logits[i]);
    }

    if batch > 0 {
        // Batched decode: `batch` independent slots, one token each per step.
        // A BARE LOOP number - no HTTP, no scheduler, no sampling - so it is
        // not a harness cell and must never be stamped as one.
        //
        // The single-slot instance goes first: this family carries a 51.2 GB
        // device-resident PLE table, so two live instances do not fit and the
        // second would quietly fall back to the host gather - i.e. measure a
        // different lane than the one it names.
        drop(m);
        drop(logits);
        let mut mb = Qwen4ExpGpu::load_with_slots(&exec, &dir, 512, batch).expect("load batched");
        let mut last: Vec<Vec<f32>> = Vec::with_capacity(batch);
        for s in 0..batch {
            // vary the prompt per slot so the slots are genuinely different
            let p: Vec<u32> = ids.iter().map(|&t| t + s as u32).collect();
            last.push(mb.prefill_slot(s, &p).expect("prefill slot"));
        }
        let steps = if n_gen > 0 { n_gen } else { 32 };
        // Width SWEEP off one load: every rung runs the same prefilled slots,
        // so the whole scaling curve costs one 50 s model load instead of
        // five. Widths are the ladder's, clamped to the slots we hold.
        let widths: Vec<usize> = if widths_arg.is_empty() {
            [1usize, 4, 8, 16, 32]
                .into_iter()
                .filter(|&w| w <= batch)
                .chain(if [1, 4, 8, 16, 32].contains(&batch) {
                    None
                } else {
                    Some(batch)
                })
                .collect()
        } else {
            widths_arg.iter().copied().filter(|&w| w <= batch).collect()
        };
        for &w in &widths {
            let mut toks: Vec<(usize, u32)> =
                (0..w).map(|s| (s, argmax1(&last[s]) as u32)).collect();
            // warm: the first tick at a width pays its graph capture
            let mut cur = mb.decode_step_batch(&toks).expect("warm step");
            let t0 = std::time::Instant::now();
            for _ in 0..steps {
                toks = (0..w).map(|s| (s, argmax1(&cur[s]) as u32)).collect();
                cur = mb.decode_step_batch(&toks).expect("batched step");
            }
            let dt = t0.elapsed().as_secs_f64();
            // CORRECTNESS GATE: the batched greedy stream is deterministic for
            // a fixed width, so it discriminates numerics changes that the
            // serve-level greedy check cannot (concurrent requests join at
            // different ticks, so 8 identical prompts legitimately produce 2
            // distinct texts there - band-on and band-off both do).
            if std::env::var_os("PADDOCK_Q38FN_GREEDY_DUMP").is_some() {
                let stream: Vec<u32> = (0..w).map(|s| argmax1(&cur[s]) as u32).collect();
                let sum: u64 = cur
                    .iter()
                    .flat_map(|r| r.iter())
                    .map(|v| v.to_bits() as u64)
                    .sum();
                eprintln!("greedy-batch {w:3}: {stream:?} bits={sum:#x}");
            }
            eprintln!(
                "batch {w:3}: {steps} steps in {:.2} s - {:6.2} ms/step, {:8.1} tok/s aggregate",
                dt,
                dt * 1e3 / steps as f64,
                (steps * w) as f64 / dt
            );
        }
        return;
    }

    if n_gen > 0 {
        let mut out = Vec::with_capacity(n_gen);
        // first step separately: it is the one that pays any lazy warm-up, and
        // folding it into the mean would understate steady-state decode
        let mut first = 0f64;
        let t_all = std::time::Instant::now();
        for i in 0..n_gen {
            let id = argmax1(&logits) as u32;
            out.push(id);
            if m.config().eos_ids.contains(&id) {
                break;
            }
            let t = std::time::Instant::now();
            logits = m.decode_step(id).expect("decode step");
            if i == 0 {
                first = t.elapsed().as_secs_f64();
            }
        }
        let all = t_all.elapsed().as_secs_f64();
        let steps = out.len().saturating_sub(1).max(1);
        eprintln!(
            "decode {steps} steps in {:.2} s - first {:.1} ms, steady {:.1} ms/tok              ({:.2} tok/s)",
            all,
            first * 1e3,
            (all - first) * 1e3 / (steps.saturating_sub(1).max(1)) as f64,
            steps as f64 / all
        );
        println!("greedy {n_gen}: {out:?}");
    }
}
