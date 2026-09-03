//! IBM Granite-Vision 4.1 declares `tokenizer.ggml.pre = "granite-docling"`,
//! which is not the `dbrx` the granite text models ship - a separate registry
//! entry, and one where we deliberately part company with llama.cpp.
//!
//! Upstream routes LLAMA_VOCAB_PRE_TYPE_GRANITE_DOCLING onto its GPT2 case,
//! whose ` ?\p{N}+` takes unbounded digit runs. The model's own tokenizer.json
//! says `\p{N}{1,3}`, and the vocab agrees: exactly 1000 three-digit tokens
//! (the whole 000-999 set) and nothing with four or more digits. So digits are
//! the assertion that matters here - get the routing wrong and "12345" becomes
//! a pre-token no vocab entry covers.

use paddock_models::mapped::MappedGguf;
use paddock_tokenizer::GgufTokenizer;

/// `GRANITE_VISION_GGUF`, else `PADDOCK_MODELS_DIR`, else `models/` under the
/// workspace root (gitignored, so a symlink to your own store works).
fn model_path() -> std::path::PathBuf {
    if let Ok(p) = std::env::var("GRANITE_VISION_GGUF") {
        return p.into();
    }
    std::env::var_os("PADDOCK_MODELS_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models")
        })
        .join("granite-vision-4.1-4b-GGUF/granite-vision-4.1-4b-Q8_0.gguf")
}

#[test]
fn granite_vision_tokenizer_builds_and_groups_digits_in_threes() {
    let path = model_path();
    if !path.exists() {
        eprintln!("model {path:?} missing - skipping");
        return;
    }
    let map = MappedGguf::open(&path).expect("open gguf");
    assert_eq!(
        map.gguf()
            .metadata
            .get("tokenizer.ggml.pre")
            .and_then(|v| v.as_str()),
        Some("granite-docling"),
        "expected the granite-docling pre-tokenizer granite-vision 4.1 ships"
    );

    // Must not return UnknownPreTokenizer - the gate that blocked serving.
    let tok = GgufTokenizer::from_gguf(map.gguf()).expect("build granite-vision tokenizer");

    // The discriminating case, and it has to be this long. "12345" does not
    // separate the two: llama.cpp hands its GPT2 pre-token to BPE, which
    // reassembles [123, 45] from the merges anyway - measured against
    // llama-server, both give 2 tokens. Ten digits is
    // where the pre-tokenizer boundaries actually show through:
    //
    //   \p{N}{1,3}          -> 123 | 456 | 789 | 0
    //   llama.cpp GPT2+BPE  -> 123 | 456 | 78  | 90   (measured)
    //
    // Same token COUNT, different tokens - which is exactly the kind of
    // divergence a count-only assertion would sail straight past.
    let ids = tok.encode("1234567890").expect("encode digits");
    let pieces: Vec<String> = ids
        .iter()
        .map(|&i| tok.decode(&[i], false).unwrap_or_default())
        .collect();
    assert_eq!(
        pieces,
        vec!["123", "456", "789", "0"],
        "expected \\p{{N}}{{1,3}} boundaries 123|456|789|0, got {pieces:?} - if this \
         reads 123|456|78|90 the registry is routing granite-docling to llama.cpp's \
         GPT2 case instead of the model's own tokenizer.json"
    );
    assert_eq!(tok.decode(&ids, false).expect("decode"), "1234567890");

    // Digits carry no leading space in this split (the regex has no ` ?`
    // before \p{N}), unlike GPT2's ` ?\p{N}+`.
    let spaced = tok.encode(" 500").expect("encode spaced digits");
    assert_eq!(
        tok.decode(&spaced, false).expect("decode"),
        " 500",
        "round-trip must survive a space-prefixed number"
    );

    // General round-trip: contraction, mixed script, punctuation.
    let text = "Hello, world! It's 12345 你好，世界 café";
    let ids = tok.encode(text).expect("encode");
    assert!(!ids.is_empty(), "empty encoding");
    assert_eq!(tok.decode(&ids, false).expect("decode"), text);
    eprintln!(
        "granite-vision tokenizer: {} tokens for {text:?}",
        ids.len()
    );
}
