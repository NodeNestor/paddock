//! The `qwen35` pre-tokenizer must be in the registry (else `from_gguf` errors
//! with UnknownPreTokenizer). Load the real 9B GGUF's tokenizer, confirm it
//! builds, and round-trips text through the qwen35 split regex + byte-level BPE.

use paddock_models::mapped::MappedGguf;
use paddock_tokenizer::GgufTokenizer;

fn model_path() -> std::path::PathBuf {
    std::env::var("QWEN35_GGUF")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            std::path::PathBuf::from("C:/dev/models/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q8_0.gguf")
        })
}

#[test]
fn qwen35_tokenizer_builds_and_roundtrips() {
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
        Some("qwen35"),
        "expected qwen35 pre-tokenizer"
    );

    // The key assertion: this does not return UnknownPreTokenizer.
    let tok = GgufTokenizer::from_gguf(map.gguf()).expect("build qwen35 tokenizer");

    let text = "Hello, world! 你好，世界 café";
    let ids = tok.encode(text).expect("encode");
    assert!(!ids.is_empty(), "empty encoding");
    let back = tok.decode(&ids, false).expect("decode");
    assert_eq!(back, text, "round-trip mismatch: {back:?}");
    eprintln!("qwen35 tokenizer: {} tokens for {:?}", ids.len(), text);
}
