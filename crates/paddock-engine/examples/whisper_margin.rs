//! Greedy decode with the top-2 logit margin printed at every step.
//!
//!   cargo run --release --example whisper_margin -- <model.gguf> <clip.wav> [lang]
//!
//! WHY. When two engines running the same weights disagree on a word, the
//! question that decides whether it is a defect or not is how close the two
//! candidates were. A flip at a 1e-3 relative margin is the model being
//! undecided and the two engines rounding differently; a flip at a wide
//! margin means one of them computed something wrong. Transcript diffs alone
//! cannot tell those apart, and neither can WER.
//!
//! This drives the same public decode API the serving scheduler uses
//! (`prepare_batch` / `encode_into` / `step_batch`), so the numbers are the
//! served path's numbers - one slot, greedy, no sampling. `logits_row` is the
//! same host readback language detection already uses.
//!
//! It doubles as the top-2 kernel's REAL-MODEL check: having both
//! a full host logits row and the device's own runner-up at every step, it
//! asserts they agree. Whisper serves its per-word alternatives off that
//! kernel, so a disagreement here would put the wrong word in a tooltip.
//!
//! Set `PADDOCK_KV_CACHE_DTYPE=f16|fp8_e4m3` to compare KV widths on the same
//! clip; the default is whatever the loader elects.
// A development probe: it runs on a box its author is looking at, and a
// failure should stop it where it happened rather than be reported.
#![allow(clippy::unwrap_used)]

use std::sync::Arc;

use paddock_engine::audio::{PAD_SAMPLES, wav, whisper_features};
use paddock_engine::gpu::{GpuExecutor, KvDtype};
use paddock_engine::gpu_model::whisper::{self, GpuWhisper};
use paddock_models::mapped::MappedGguf;
use paddock_tokenizer::GgufTokenizer;

/// Top-2 of a logits row, and the softmax probabilities behind them. The
/// probability pair is what makes margins comparable across steps - raw
/// logit gaps mean different things at different entropies.
fn top2(row: &[f32]) -> (usize, f32, usize, f32, f32, f32) {
    let (mut i1, mut i2) = (0usize, 0usize);
    let (mut l1, mut l2) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
    for (i, &v) in row.iter().enumerate() {
        if v > l1 {
            l2 = l1;
            i2 = i1;
            l1 = v;
            i1 = i;
        } else if v > l2 {
            l2 = v;
            i2 = i;
        }
    }
    // denom is sum(exp(v - l1)), so the top-1 probability is exp(0)/denom
    let denom: f32 = row.iter().map(|&v| (v - l1).exp()).sum();
    (i1, l1, i2, l2, 1.0 / denom, (l2 - l1).exp() / denom)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (gguf, wav_path, lang) = match args.as_slice() {
        [g, w] => (g, w, None),
        [g, w, l] => (g, w, Some(l.as_str())),
        _ => {
            eprintln!("usage: whisper_margin <model.gguf> <clip.wav> [lang]");
            std::process::exit(2);
        }
    };
    let pack = std::env::var("PADDOCK_PACK").expect("set PADDOCK_PACK");

    let clip = wav::decode_wav(&std::fs::read(wav_path).expect("read wav")).expect("decode wav");
    assert_eq!(clip.sample_rate, 16000, "feed 16 kHz mono");

    let map = MappedGguf::open(gguf.as_ref()).expect("open gguf");
    let tok = GgufTokenizer::from_gguf(map.gguf()).expect("tokenizer");
    let exec = Arc::new(GpuExecutor::new(0, pack.as_ref()).expect("cuda executor"));
    let mut m = GpuWhisper::load(exec, &map, 448).expect("load whisper");
    if let Ok(s) = std::env::var("PADDOCK_KV_CACHE_DTYPE") {
        let d = match s.as_str() {
            "f16" | "fp16" => KvDtype::Fp16,
            "fp8_e4m3" | "fp8" => KvDtype::Fp8E4m3,
            other => panic!("unknown PADDOCK_KV_CACHE_DTYPE {other:?}"),
        };
        m.set_kv_dtype(d);
    }
    // The pool cap is a knob because the decode kernel's work split follows
    // the ACTIVE ROW COUNT: a tie decided at 1e-3 relative can land either way
    // depending on how the reduction was partitioned, and the serving path
    // runs a wide pool while this probe would default to one slot.
    let cap: usize = std::env::var("PADDOCK_MARGIN_CAP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    m.prepare_batch(cap).expect("pool");

    let (sot, eot) = m.contract_tokens();
    let (transcribe, no_ts) = m.prompt_tail();
    let ctx = m.text_ctx();
    let scale = m.time_scale();
    let slots = [0u32];

    let mut off = 0usize;
    let mut window = 0usize;
    while off < clip.samples.len().max(1) {
        let end = (off + PAD_SAMPLES).min(clip.samples.len());
        let mel = whisper_features(&clip.samples[off..end]).expect("mel");
        m.encode_into(0, &mel).expect("encode");
        println!(
            "== window {window} ({:.2}s .. {:.2}s)",
            off as f64 / 16000.0,
            end as f64 / 16000.0
        );

        let mut pos = 0u32;
        // `sampled` is what the timestamp grammar reads: the tokens emitted so
        // far in this window, prompt excluded. Empty on the prompt steps.
        let step = |m: &mut GpuWhisper, t: u32, p: &mut u32, sampled: &[u32], rules: bool| {
            let st = whisper::ts_state(sampled, &scale, rules);
            let out = m
                .step_batch(&slots, &[t], &[*p], rules.then_some(&st[..]))
                .expect("step");
            *p += 1;
            (out.next[0], out.runner_up[0])
        };

        let mut sampled: Vec<u32> = Vec::new();
        let (mut next, mut alt) = step(&mut m, sot, &mut pos, &sampled, false);
        // The `<|startoftranscript|>` step is where OpenAI reads
        // `no_speech_prob`, so this is the one place the number can be
        // checked against what the model actually thought. Printed with the
        // step's top-3 because the answer "0" is only meaningful next to what
        // did win - on a fine-tune that never saw `<|nospeech|>` the mass
        // goes entirely to the language tokens.
        {
            let row = m.logits_row(0).expect("logits");
            let mx = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let den: f32 = row.iter().map(|&v| (v - mx).exp()).sum();
            let mut idx: Vec<usize> = (0..row.len()).collect();
            idx.sort_by(|&a, &b| row[b].total_cmp(&row[a]));
            let top: Vec<String> = idx[..3]
                .iter()
                .map(|&i| {
                    format!(
                        "{:?}({i}) p={:.4}",
                        tok.decode(&[i as u32], false).unwrap_or_default(),
                        (row[i] - mx).exp() / den
                    )
                })
                .collect();
            let ns = m.nospeech_token() as usize;
            println!(
                "  sot: p(<|nospeech|>)={:.6e} (logit {:.3}, max {:.3}) | top3 {}",
                (row[ns] - mx).exp() / den,
                row[ns],
                mx,
                top.join("  ")
            );
        }
        let lang_tok = match lang {
            Some(code) => m.lang_token(code).expect("language not in this checkpoint"),
            None => {
                let row = m.logits_row(0).expect("logits");
                m.detect_language(&row).expect("detect").1
            }
        };
        // PADDOCK_MARGIN_TIMESTAMPS=1 drops `<|notimestamps|>`, which is how
        // whisper is asked to emit its timestamp tokens. Worth a
        // flag here rather than only in the server: this probe prints the
        // per-step candidates, so it is what tells you whether a FINE-TUNE
        // still emits `<|0.00|>` at all - a checkpoint trained without
        // timestamps just decodes straight into text, and the difference is
        // invisible in a transcript.
        // The grammar runs from the step that CONSUMES `<|transcribe|>`
        // onwards - that step's own argmax is the window's first emitted
        // token, and it is the one the reference forces to be a timestamp.
        let want_ts = std::env::var("PADDOCK_MARGIN_TIMESTAMPS").is_ok_and(|v| v != "0");
        let prompt_tail: &[u32] = if want_ts {
            &[lang_tok, transcribe]
        } else {
            &[lang_tok, transcribe, no_ts]
        };
        for (i, &t) in prompt_tail.iter().enumerate() {
            let last = i + 1 == prompt_tail.len();
            (next, alt) = step(&mut m, t, &mut pos, &sampled, want_ts && last);
        }

        // per-token margins over the transcript proper - the prompt steps
        // above are contract tokens and carry no interesting choice
        let mut ties = 0usize;
        for i in 0..448 {
            if next == eot || pos as usize + 1 >= ctx {
                break;
            }
            let row = m.logits_row(0).expect("logits");
            let (i1, l1, i2, l2, p1, p2) = top2(&row);
            debug_assert_eq!(i1 as u32, next, "argmax_rows disagrees with the host top-1");
            // The device top-2 against this host top-2, on the real model.
            // The synthetic gate (gpu_argmax_top2) covers the reduction on
            // constructed rows; only here does it run at whisper's true 51866
            // width, on distributions the model actually produces, inside the
            // graph-CAPTURED tick - three things a unit test cannot reach.
            match alt {
                Some((id, lp2)) => {
                    assert_eq!(
                        id as usize,
                        i2,
                        "step {i}: kernel runner-up {id} vs host {i2} ({:?} vs {:?})",
                        tok.decode(&[id], true).unwrap_or_default(),
                        tok.decode(&[i2 as u32], true).unwrap_or_default()
                    );
                    assert!(
                        (lp2 - p2.ln()).abs() < 2e-3,
                        "step {i}: kernel log p2 {lp2} vs host {}",
                        p2.ln()
                    );
                }
                None => panic!("step {i}: no runner-up on a {}-token vocabulary", row.len()),
            }
            let t1 = tok.decode(&[i1 as u32], true).unwrap_or_default();
            let t2 = tok.decode(&[i2 as u32], true).unwrap_or_default();
            // relative gap: how big the winning margin is next to the logit
            // scale itself, so it is comparable across steps and models
            let rel = (l1 - l2) / l1.abs().max(1e-6);
            if p2 > 0.02 {
                ties += 1;
            }
            println!(
                "  {i:3}  {t1:?} ({i1}) l={l1:.4} p={p1:.4}   vs  {t2:?} ({i2}) l={l2:.4} p={p2:.4}   \
                 gap={:.4} rel={rel:.4}{}",
                l1 - l2,
                if p2 > 0.02 { "  <-- CONTESTED" } else { "" }
            );
            sampled.push(next);
            (next, alt) = step(&mut m, next, &mut pos, &sampled, want_ts);
        }
        println!("  ({ties} contested steps, p(runner-up) > 2%)\n");
        off = end;
        window += 1;
    }
}
