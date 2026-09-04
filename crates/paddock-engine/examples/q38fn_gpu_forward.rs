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
    let mut m = Qwen4ExpGpu::load(&exec, &dir, 512).expect("load");
    eprintln!("loaded in {:.1}s", t0.elapsed().as_secs_f32());

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

    if n_gen > 0 {
        let mut out = Vec::with_capacity(n_gen);
        // first step separately: it is the one that pays any lazy warm-up, and
        // folding it into the mean would understate steady-state decode
        let mut first = 0f64;
        let t_all = std::time::Instant::now();
        for i in 0..n_gen {
            let id = top_k(&logits, 1)[0] as u32;
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
