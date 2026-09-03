//! IBM Granite 4.1 declares `tokenizer.ggml.pre = "dbrx"`, which llama.cpp
//! serves from its llama3 case. Before that entry existed here, `from_gguf`
//! hard-errored with UnknownPreTokenizer - that error is the designed
//! behaviour for an unknown pre, so the registry entry is a real prerequisite
//! for serving granite, not a formality. Load the real 8b GGUF's tokenizer,
//! confirm it builds, and round-trip text through the split + byte-level BPE.

use paddock_models::mapped::MappedGguf;
use paddock_tokenizer::GgufTokenizer;

/// `GRANITE_GGUF`, else `PADDOCK_MODELS_DIR`, else `models/` under the
/// workspace root (gitignored, so a symlink to your own store works).
fn model_path() -> std::path::PathBuf {
    if let Ok(p) = std::env::var("GRANITE_GGUF") {
        return p.into();
    }
    std::env::var_os("PADDOCK_MODELS_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models")
        })
        .join("granite-4.1-8b-GGUF/granite-4.1-8b-Q8_0.gguf")
}

#[test]
fn granite_tokenizer_builds_and_roundtrips() {
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
        Some("dbrx"),
        "expected the dbrx pre-tokenizer granite 4.1 ships"
    );

    // The key assertion: this does not return UnknownPreTokenizer.
    let tok = GgufTokenizer::from_gguf(map.gguf()).expect("build granite tokenizer");

    // Mixed script + a contraction (the [sS] alternation) + digits, which the
    // llama3 split chunks at {1,3} unlike qwen's single-digit \p{N}.
    let text = "Hello, world! It's 12345 你好，世界 café";
    let ids = tok.encode(text).expect("encode");
    assert!(!ids.is_empty(), "empty encoding");
    let back = tok.decode(&ids, false).expect("decode");
    assert_eq!(back, text, "round-trip mismatch: {back:?}");
    eprintln!("granite tokenizer: {} tokens for {:?}", ids.len(), text);
}
