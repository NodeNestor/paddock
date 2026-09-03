//! Geometry gate for the DeepSeek-OCR family: parse the real
//! converted GGUF's metadata and assert the shape we designed against.
//!
//! Cheap - metadata only, no uploads, no CUDA - so it runs anywhere the file is
//! present and skips cleanly where it is not. It exists because three of the
//! numbers below are traps rather than trivia, and each one is silent when
//! wrong: a rope dimension read from the file is 0, the shared experts arrive
//! pre-merged at twice the per-expert width, and the sliding-window key means
//! R-SWA rather than the gemma4/laguna window it looks like.

mod common;

use paddock_engine::gpu_model::deepseek_ocr::{ARCH, Hparams};
use paddock_models::mapped::MappedGguf;

fn model_path() -> Option<std::path::PathBuf> {
    if let Ok(p) = std::env::var("UNLIMITED_OCR_GGUF") {
        let p = std::path::PathBuf::from(p);
        return p.exists().then_some(p);
    }
    common::model_roots().iter().find_map(|r| {
        let p = r.join("Unlimited-OCR-GGUF").join("Unlimited-OCR-Q8_0.gguf");
        p.exists().then_some(p)
    })
}

#[test]
fn unlimited_ocr_geometry_matches_the_design() {
    let Some(path) = model_path() else {
        common::missing("no Unlimited-OCR GGUF (set UNLIMITED_OCR_GGUF)");
        return;
    };
    let map = MappedGguf::open(&path).expect("open gguf");
    assert_eq!(map.gguf().architecture(), Some(ARCH), "arch string");

    let hp = Hparams::from_gguf(&map).expect("parse hparams");

    // Language config, identical across the whole family.
    assert_eq!(hp.n_layer, 12);
    assert_eq!(hp.n_embd, 1280);
    assert_eq!(hp.n_head, 10);
    assert_eq!(hp.n_head_kv, 10, "plain MHA - no GQA on this family");
    assert_eq!(hp.n_vocab, 129_280);
    assert_eq!(hp.n_ctx_train, 32_768);

    // TRAP 1. The file's own `rope.dimension_count` is 0, because llama.cpp
    // derives it from qk_rope_head_dim, which is 0 precisely because this model
    // is not MLA. Anything that reads that key gets a dead RoPE. head_dim must
    // be computed.
    assert_eq!(hp.head_dim, 128, "head_dim must be n_embd/n_head, not read");
    let file_rope_dim = map
        .gguf()
        .metadata
        .get("deepseek2-ocr.rope.dimension_count")
        .and_then(|v| v.as_u64());
    assert_eq!(
        file_rope_dim,
        Some(0),
        "if this stops being 0 the converter changed - re-check whether reading it is now safe"
    );

    // MoE: 64 routed, top-6, one leading dense layer.
    assert_eq!(hp.n_expert, 64);
    assert_eq!(hp.n_expert_used, 6);
    assert_eq!(hp.n_ff_exp, 896);
    assert_eq!(hp.n_ff, 6848, "dense width, used by layer 0 only");
    assert_eq!(hp.first_k_dense, 1);
    assert!(!hp.is_moe_layer(0));
    assert!(hp.is_moe_layer(1) && hp.is_moe_layer(hp.n_layer - 1));

    // TRAP 2. Two shared experts, but the GGUF stores them PRE-MERGED as one
    // 1792-wide plane, so the graph runs a single dense MLP. Assert the merged
    // width against the tensor that actually exists rather than trusting arithmetic.
    assert_eq!(hp.n_expert_shared, 2);
    assert_eq!(hp.shexp_ff(), 1792);
    let shexp = map
        .gguf()
        .tensors
        .iter()
        .find(|t| t.name == "blk.1.ffn_gate_shexp.weight")
        .expect("blk.1.ffn_gate_shexp.weight");
    let dims: Vec<usize> = shexp.dims.iter().map(|d| *d as usize).collect();
    assert!(
        dims.contains(&hp.shexp_ff()),
        "shared-expert plane {dims:?} does not carry the merged width {}",
        hp.shexp_ff()
    );

    // TRAP 3. R-SWA, not a gemma4-style sliding window: the prefill prefix stays
    // globally visible and only generated tokens ring.
    assert_eq!(hp.rswa_window, Some(128));
    let prefill = 907; // a representative document page's prefill length
    assert_eq!(
        hp.kv_rows(prefill, 32),
        prefill + 32,
        "below the ring KV still grows"
    );
    assert_eq!(
        hp.kv_rows(prefill, 1_000_000),
        prefill + 128,
        "R-SWA must pin KV no matter how long the document runs"
    );
}
