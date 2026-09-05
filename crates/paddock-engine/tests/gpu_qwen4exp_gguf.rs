//! Qwen3.8-Flash-Next off a llama.cpp GGUF (the Unsloth UD exports) on the
//! consumer-card lane: k-quant dense planes, k-quant / i-quant expert seats
//! host-mapped behind the `[moe_offload]` slot cache, PLE table gathered
//! from the mmap. Heavy: `QWEN38FN_GGUF=<first shard>` names the file.
//!
//! The gate is greedy continuation of a fixed prompt against llama.cpp on
//! the same file (`QWEN38FN_GGUF_REF` = the expected token ids, comma
//! separated, from `llama-cli --temp 0`); without a reference it prints the
//! continuation and the decode rate.

mod common;

use std::time::Instant;

use paddock_engine::gpu_model::qwen4exp::Qwen4ExpGpu;
use paddock_models::mapped::MappedGguf;
use paddock_tokenizer::GgufTokenizer;

const PROMPT: &str = "The capital of France is";

#[test]
fn gguf_greedy_continuation() {
    if !common::heavy() {
        return;
    }
    let Some(path) = common::model("QWEN38FN_GGUF", &[]) else {
        common::missing("QWEN38FN_GGUF");
        return;
    };
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    // host-mapped experts + the slot cache are what this lane exists for
    unsafe { std::env::set_var("PADDOCK_MOE_HOST", "1") };
    let map = MappedGguf::open(&path).expect("open gguf");
    let tok = GgufTokenizer::from_gguf(map.gguf()).expect("tokenizer");
    drop(map);
    let t0 = Instant::now();
    let mut m = Qwen4ExpGpu::load_gguf_with_slots(&exec, &path, 4096, 1).expect("load gguf");
    let load_s = t0.elapsed().as_secs_f64();
    let headroom = exec.vram_headroom().unwrap_or(0);
    let seated = m
        .enable_moe_cache(headroom.saturating_sub(512 << 20))
        .expect("seat expert cache");
    eprintln!(
        "load {load_s:.1}s, {:.1} GiB experts host-mapped, cache on {seated} layers ({:.2} GiB headroom)",
        m.expert_host_bytes() as f64 / (1u64 << 30) as f64,
        headroom as f64 / (1u64 << 30) as f64
    );
    let prompt = tok.encode(PROMPT).expect("encode");
    let n = 32usize;
    let t1 = Instant::now();
    let out = m.generate_greedy(&prompt, n).expect("generate");
    let gen_s = t1.elapsed().as_secs_f64();
    let text = tok.decode(&out, false).unwrap_or_default();
    eprintln!("prompt ids {prompt:?}");
    eprintln!("greedy ids {out:?}");
    eprintln!("greedy text {text:?}");
    eprintln!("{n} tokens in {gen_s:.2}s = {:.1} tok/s (prefill included)", n as f64 / gen_s);
    if let Ok(reference) = std::env::var("QWEN38FN_GGUF_REF") {
        let want: Vec<u32> = reference
            .split(',')
            .filter(|s| !s.trim().is_empty())
            .map(|s| s.trim().parse().expect("reference token id"))
            .collect();
        let k = want.len().min(out.len());
        assert_eq!(
            &out[..k],
            &want[..k],
            "greedy continuation diverges from the llama.cpp reference"
        );
    }
    assert!(
        out.iter().any(|&t| t != out[0]),
        "degenerate continuation (every token {})",
        out[0]
    );
}
