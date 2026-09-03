//! Gemma 4 prefix-cache acceptance: turn 1 caches a long prompt; turn 2
//! (same prompt + a new user turn) must RESUME from the checkpoint and
//! produce a greedy continuation byte-identical to the cache-free
//! single-stream lane (the oracle-locked reference). Reports the reused
//! token count and the prefill-time drop.
//!
//! Usage: GEMMA4_GGUF=... PADDOCK_PACK=... [MAX_CTX=8192] gemma4_prefix_check

use std::sync::Arc;

use paddock_engine::generator::Generator;
use paddock_engine::gpu::GpuExecutor;
use paddock_engine::gpu_model::gemma4::GpuGemma4;
use paddock_models::mapped::MappedGguf;
use paddock_tokenizer::GgufTokenizer;

fn argmax(l: &[f32]) -> u32 {
    let mut b = 0usize;
    for i in 1..l.len() {
        if l[i] > l[b] {
            b = i;
        }
    }
    b as u32
}

fn main() {
    let model = std::env::var("GEMMA4_GGUF").expect("set GEMMA4_GGUF");
    let pack = std::env::var("PADDOCK_PACK").expect("set PADDOCK_PACK");
    let max_ctx: usize = std::env::var("MAX_CTX")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8192);
    let n_gen = 24usize;

    let map = MappedGguf::open(model.as_ref()).expect("open gguf");
    let tok = GgufTokenizer::from_gguf(map.gguf()).expect("tokenizer");
    let bos = tok.bos_id.expect("bos");

    // turn 1: a long "conversation so far"; turn 2 appends a new user turn
    let long_text = "The history of computing spans mechanical calculators, \
        electromechanical relays, vacuum tubes, transistors and integrated \
        circuits. Each generation multiplied speed and shrank cost. "
        .repeat(40);
    let turn1: Vec<u32> = {
        let mut ids = vec![bos];
        ids.extend(tok.encode(&long_text).expect("encode"));
        ids
    };
    let turn2: Vec<u32> = {
        let mut ids = turn1.clone();
        ids.extend(
            tok.encode("Summarize the above in one sentence.")
                .expect("encode"),
        );
        ids
    };
    eprintln!("turn1 {} tokens, turn2 {} tokens", turn1.len(), turn2.len());

    let exec = Arc::new(GpuExecutor::new(0, pack.as_ref()).expect("executor"));
    let mut m = GpuGemma4::load(exec, &map, max_ctx).expect("load");
    let cap = m.enable_batch(2).expect("enable_batch");
    assert!(cap >= 1, "no slots: {cap}");
    let vocab = m.vocab();

    // turn 1 into slot 0 (cold; inserts pages + tail checkpoint)
    let t0 = std::time::Instant::now();
    let _ = m.forward_prefill(0, &turn1).expect("turn1 prefill");
    let cold1 = t0.elapsed().as_secs_f32();
    assert_eq!(m.take_prefill_reused(0), 0, "turn1 must be cold");

    // turn 2 back into slot 0 (the conversation's slot - and row 0 of
    // forward_batch drives slot 0): must resume from turn1's checkpoint
    let t1 = std::time::Instant::now();
    let logits = m.forward_prefill(0, &turn2).expect("turn2 prefill");
    let warm = t1.elapsed().as_secs_f32();
    let reused = m.take_prefill_reused(0);
    eprintln!(
        "turn1 cold {cold1:.2}s | turn2 warm {warm:.2}s | reused {reused}/{} tokens",
        turn2.len()
    );
    assert!(reused > 0, "turn 2 did not resume from the cache");

    // greedy continuation through the batch lane from the resumed state
    let mut tokens = vec![argmax(&logits)];
    let mut positions = vec![turn2.len() as u32];
    let mut warm_out = vec![tokens[0]];
    for _ in 1..n_gen {
        let l = m.forward_batch(&tokens, &positions).expect("decode");
        let next = argmax(&l[..vocab]);
        warm_out.push(next);
        tokens[0] = next;
        positions[0] += 1;
    }

    // reference: the cache-free single-stream lane (oracle-locked)
    m.reset();
    let mut logits = m.forward_prefill_stream(&turn2).expect("stream prefill");
    let mut cold_out = Vec::new();
    for _ in 0..n_gen {
        let next = argmax(&logits);
        cold_out.push(next);
        logits = m.forward(next).expect("decode");
    }

    if warm_out == cold_out {
        println!(
            "PREFIX-CACHE-OK reused={reused} cold={cold1:.2}s warm={warm:.2}s ({:.1}x)",
            cold1 / warm
        );
    } else {
        let d = warm_out.iter().zip(&cold_out).position(|(a, b)| a != b);
        println!("PREFIX-CACHE-DIFF at {d:?}");
        println!(
            "  warm: {:?}",
            tok.decode(&warm_out, false).unwrap_or_default()
        );
        println!(
            "  cold: {:?}",
            tok.decode(&cold_out, false).unwrap_or_default()
        );
    }
}
