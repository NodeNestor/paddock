//! Nemotron 3.5 Lightning tokenizer, built straight from the HF checkpoint
//! dir (the safetensors-primary path - no GGUF exists in this lane).
//!
//! The encoding anchor is an arbiter oracle's own prompt_ids: vLLM tokenized
//! this exact string with this exact tokenizer.json, so an id mismatch here is
//! a construction bug, not a model question.

use paddock_tokenizer::GgufTokenizer;
use std::path::PathBuf;

fn ckpt() -> Option<PathBuf> {
    let p = std::env::var("NEMOTRON_NVFP4_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"));
    p.join("tokenizer.json").exists().then_some(p)
}

#[test]
fn nemotron_hf_dir_tokenizer_matches_the_arbiter_oracle() {
    let Some(dir) = ckpt() else {
        eprintln!("skip: no nemotron checkpoint on this machine");
        return;
    };
    let tok = GgufTokenizer::from_hf_dir(&dir).expect("builds from HF dir");

    // decode contract: eos set [2, 11] = </s> + <|im_end|>, no auto-BOS
    assert_eq!(tok.eos_id, Some(2));
    assert_eq!(tok.eot_id, Some(11));
    assert_eq!(tok.bos_id, Some(1));
    assert!(!tok.add_bos);
    assert_eq!(tok.vocab_size, 131072);
    let template = tok
        .chat_template
        .as_deref()
        .expect("chat_template.jinja loads");
    assert!(template.contains("<|im_start|>"));
    assert!(template.contains("enable_thinking"));

    // the oracle prompt, ids pinned from the vLLM decoder oracle
    let ids = tok
        .encode("The quick brown fox jumps over the lazy dog. The capital of Sweden is")
        .expect("encodes");
    assert_eq!(
        ids,
        vec![
            1784, 7586, 22980, 94137, 72993, 2136, 1278, 42757, 10575, 1046, 1531, 8961, 1307,
            27453, 1395
        ]
    );

    // ChatML markers must survive as unsplittable specials
    let marker_ids = tok
        .encode("<|im_start|>user\nhej<|im_end|>")
        .expect("encodes markers");
    assert_eq!(marker_ids.first(), Some(&10));
    assert_eq!(marker_ids.last(), Some(&11));

    // round-trip
    let text = tok.decode(&ids, false).expect("decodes");
    assert_eq!(
        text,
        "The quick brown fox jumps over the lazy dog. The capital of Sweden is"
    );
}
