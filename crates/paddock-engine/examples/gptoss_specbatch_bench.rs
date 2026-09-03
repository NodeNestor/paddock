//! G5 serving spec-decode benchmark: per-slot n-gram drafting in the batched
//! decode round (forward_spec_batch) vs the plain per-tick forward_batch loop
//! a greedy server runs, on gpt-oss-20b. Workloads: all-repeat (best case),
//! mixed repeat/prose (realistic), all-prose (regression floor). Args:
//! `<B>` (default 8) `<tokens per seq>` (default 128) `<K>` (default 3).

use std::sync::Arc;

use paddock_engine::gpu::GpuExecutor;
use paddock_engine::gpu_model::gpt_oss::GpuGptOss;
use paddock_engine::spec::NgramDraft;
use paddock_models::mapped::MappedGguf;
use paddock_tokenizer::GgufTokenizer;

const REPEAT: &str = "Repeat the following paragraph exactly, word for word, \
    over and over without stopping: The quick brown fox jumps over the lazy \
    dog while the cat watches from the window and the birds sing in the \
    garden by the old stone wall (copy #%).\n\nThe quick brown fox jumps \
    over the lazy dog while the cat watches from the window and the birds \
    sing in the garden by the old stone wall (copy #%). The quick brown fox";

const PROSE: &str = "Once upon a time, in a quiet village nestled between \
    rolling hills numbered % on the old maps, there lived a clockmaker who";

fn argmax(l: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, v) in l.iter().enumerate() {
        if *v > l[best] {
            best = i;
        }
    }
    best as u32
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let b: usize = args.get(1).map(|s| s.parse().expect("B")).unwrap_or(8);
    let n_tok: usize = args
        .get(2)
        .map(|s| s.parse().expect("tokens"))
        .unwrap_or(128);
    let k_draft: usize = args.get(3).map(|s| s.parse().expect("K")).unwrap_or(3);
    // total verify rows must fit SPEC_BATCH_MAX_ROWS (32): B x (K+1) <= 32
    let k_draft = k_draft.min(32 / b - 1);
    let model_path = std::env::var_os("PADDOCK_MODEL")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            let home = std::env::var("USERPROFILE")
                .or_else(|_| std::env::var("HOME"))
                .expect("USERPROFILE or HOME");
            std::path::PathBuf::from(home).join("paddock/models/gpt-oss-20b-mxfp4.gguf")
        });
    let pack = std::env::var_os("PADDOCK_PACK")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from("packs/cuda/build/pd-cuda-sm86.dll"));
    let exec = Arc::new(GpuExecutor::new(0, &pack).expect("executor"));
    let map = MappedGguf::open(&model_path).expect("open gguf");
    let tok = GgufTokenizer::from_gguf(map.gguf()).expect("tokenizer");
    let mut m = GpuGptOss::load(exec, &map, 4096).expect("load");
    m.enable_batch(b).expect("enable_batch");
    let vocab = 201088usize;

    for (wl, mk) in [
        ("repeat", 0usize), // all slots repeat
        ("mixed", 1),       // even slots repeat, odd slots prose
        ("prose", 2),       // all slots prose
    ] {
        // distinct prompt per slot (the % marker) so streams differ per slot
        let prompts: Vec<Vec<u32>> = (0..b)
            .map(|s| {
                let text = match (mk, s % 2) {
                    (0, _) | (1, 0) => REPEAT.replace('%', &s.to_string()),
                    _ => PROSE.replace('%', &s.to_string()),
                };
                tok.encode(&text).expect("encode")
            })
            .collect();

        // ---- plain greedy serving loop: prefill each slot, then one
        // forward_batch per tick + host argmax per row (what a greedy
        // server does today)
        let mut pending = vec![0u32; b];
        let mut pos = vec![0u32; b];
        for (s, p) in prompts.iter().enumerate() {
            let logits = m.forward_prefill(s, p).expect("prefill");
            pending[s] = argmax(&logits);
            pos[s] = p.len() as u32;
        }
        let mut plain: Vec<Vec<u32>> = (0..b).map(|s| vec![pending[s]]).collect();
        let t0 = std::time::Instant::now();
        let mut produced = b; // first tokens came from prefill
        while plain.iter().any(|st| st.len() < n_tok) {
            let logits = m.forward_batch(&pending, &pos).expect("fwd");
            for s in 0..b {
                if plain[s].len() >= n_tok {
                    // keep feeding the row (a live server would drop it; the
                    // dense loop can't) but stop recording
                    pos[s] += 1;
                    pending[s] = argmax(&logits[s * vocab..(s + 1) * vocab]);
                    continue;
                }
                pos[s] += 1;
                pending[s] = argmax(&logits[s * vocab..(s + 1) * vocab]);
                plain[s].push(pending[s]);
                produced += 1;
            }
        }
        let plain_rate = produced as f64 / t0.elapsed().as_secs_f64();

        // ---- spec serving loop: per-slot drafters, ragged verify rounds
        let mut pending = vec![0u32; b];
        let mut pos = vec![0usize; b];
        let mut drs: Vec<NgramDraft> = (0..b).map(|_| NgramDraft::default()).collect();
        for (s, p) in prompts.iter().enumerate() {
            let logits = m.forward_prefill(s, p).expect("prefill");
            pending[s] = argmax(&logits);
            pos[s] = p.len();
            for &t in p.iter().chain([pending[s]].iter()) {
                drs[s].push(t);
            }
        }
        let mut spec: Vec<Vec<u32>> = (0..b).map(|s| vec![pending[s]]).collect();
        // per-slot adaptive draft length (the G3 rule: double on full
        // accept, shrink to the observed run on a reject)
        let mut k_now = vec![k_draft.min(2).max(1); b];
        let t0 = std::time::Instant::now();
        let mut produced = b;
        let (mut rounds, mut rows_total, mut drafted, mut accepted) =
            (0usize, 0usize, 0usize, 0usize);
        while spec.iter().any(|st| st.len() < n_tok) {
            let mut reqs: Vec<(usize, usize, Vec<u32>)> = Vec::with_capacity(b);
            let mut live: Vec<usize> = Vec::with_capacity(b);
            for s in 0..b {
                if spec[s].len() >= n_tok {
                    continue; // finished slots contribute no rows
                }
                let mut chunk = vec![pending[s]];
                chunk.extend(drs[s].draft(k_now[s]));
                drafted += chunk.len() - 1;
                reqs.push((s, pos[s], chunk));
                live.push(s);
            }
            let picks = m.forward_spec_batch(&reqs).expect("spec batch");
            rounds += 1;
            let mut base = 0usize;
            for (ri, s) in live.iter().copied().enumerate() {
                let chunk = &reqs[ri].2;
                let mut a = 0usize;
                while a + 1 < chunk.len() && chunk[a + 1] == picks[base + a] {
                    a += 1;
                }
                accepted += a;
                if chunk.len() > 1 {
                    if a == chunk.len() - 1 {
                        k_now[s] = (k_now[s] * 2).min(k_draft);
                    } else {
                        k_now[s] = (a + 1).clamp(1, k_now[s]);
                    }
                }
                for &t in picks[base..=base + a].iter() {
                    spec[s].push(t);
                    drs[s].push(t);
                    produced += 1;
                }
                pos[s] += a + 1;
                pending[s] = picks[base + a];
                base += chunk.len();
            }
            rows_total += base;
        }
        let spec_rate = produced as f64 / t0.elapsed().as_secs_f64();

        let same = (0..b)
            .all(|s| plain[s][..n_tok.min(plain[s].len())] == spec[s][..n_tok.min(spec[s].len())]);
        println!(
            "{wl:>7} B={b}: plain {plain_rate:7.1} tok/s | spec(K={k_draft}) {spec_rate:7.1} tok/s | {:.2}x | {rounds} rounds avg {:.1} rows, acc {:.0}% | streams {}",
            spec_rate / plain_rate,
            rows_total as f64 / rounds.max(1) as f64,
            100.0 * accepted as f64 / drafted.max(1) as f64,
            if same {
                "IDENTICAL"
            } else {
                "differ (class near-tie)"
            }
        );
    }
}
