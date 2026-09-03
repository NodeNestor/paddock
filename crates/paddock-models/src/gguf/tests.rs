//! Parser tests: a tiny in-memory GGUF writer builds well-formed and hostile
//! inputs; a gated test parses a real llama.cpp-produced file when present.

use super::*;
use crate::testutil::Writer;

#[test]
fn parses_metadata_tensors_and_alignment() {
    let mut w = Writer::new(2, 3);
    w.kv_str("general.architecture", "llama");
    w.kv_u32("llama.context_length", 4096);
    w.kv_str_array("tokenizer.ggml.tokens", &["<s>", "</s>", "hej"]);
    // two 4x2 F32 tensors: 32 bytes each, second at offset 32
    w.tensor_f32("blk.0.attn_q.weight", &[4, 2], 0);
    w.tensor_f32("blk.0.attn_k.weight", &[4, 2], 32);
    let bytes = w.finish_with_data(64);

    let f = GgufFile::parse(&bytes).expect("parses");
    assert_eq!(f.version, 3);
    assert_eq!(f.architecture(), Some("llama"));
    assert_eq!(
        f.arch_field("context_length").and_then(Value::as_u64),
        Some(4096)
    );
    assert_eq!(f.tensors.len(), 2);
    assert_eq!(f.tensors[1].offset, 32);
    assert_eq!(f.tensors[0].byte_size(), Some(32));
    match &f.metadata["tokenizer.ggml.tokens"] {
        Value::Array(items) => assert_eq!(items.len(), 3),
        other => panic!("expected array, got {other:?}"),
    }
}

#[test]
fn bad_magic_is_an_error_not_a_panic() {
    assert!(matches!(
        GgufFile::parse(b"NOPE\x03\x00\x00\x00"),
        Err(GgufError::BadMagic)
    ));
}

#[test]
fn hostile_kv_count_is_capped_before_allocation() {
    // header claims u64::MAX metadata entries; must fail on the cap, fast
    let mut buf = Vec::new();
    buf.extend_from_slice(&0x4655_4747u32.to_le_bytes());
    buf.extend_from_slice(&3u32.to_le_bytes());
    buf.extend_from_slice(&0u64.to_le_bytes());
    buf.extend_from_slice(&u64::MAX.to_le_bytes());
    assert!(matches!(
        GgufFile::parse(&buf),
        Err(GgufError::CountTooLarge {
            what: "metadata kv",
            ..
        })
    ));
}

#[test]
fn truncated_file_reports_where() {
    let w = Writer::new(0, 1); // claims one kv, provides none
    let bytes = w.buf;
    assert!(matches!(
        GgufFile::parse(&bytes),
        Err(GgufError::Truncated { .. })
    ));
}

#[test]
fn tensor_past_end_of_file_is_rejected() {
    let mut w = Writer::new(1, 0);
    w.tensor_f32("huge.weight", &[1024, 1024], 0); // 4 MiB claimed
    let bytes = w.finish_with_data(64); // 64 bytes provided
    assert!(matches!(
        GgufFile::parse(&bytes),
        Err(GgufError::BadTensor { .. })
    ));
}

#[test]
fn unaligned_tensor_offset_is_rejected() {
    let mut w = Writer::new(1, 0);
    w.tensor_f32("t.weight", &[8], 7); // 7 % 32 != 0
    let bytes = w.finish_with_data(64);
    assert!(matches!(
        GgufFile::parse(&bytes),
        Err(GgufError::BadTensor { .. })
    ));
}

/// Parses a real llama.cpp-produced file when one is available locally
/// (PADDOCK_TEST_GGUF env var, or the tiny stories260K from the Unsloth
/// install). Skips silently in CI where no fixture exists.
#[test]
fn parses_real_gguf_when_available() {
    let candidates: Vec<std::path::PathBuf> = std::env::var_os("PADDOCK_TEST_GGUF")
        .map(|p| vec![p.into()])
        .unwrap_or_else(|| {
            std::env::var_os("USERPROFILE")
                .map(|home| {
                    vec![std::path::PathBuf::from(&home).join(".unsloth/.cache/stories260K.gguf")]
                })
                .unwrap_or_default()
        });

    let Some(path) = candidates.iter().find(|p| p.exists()) else {
        eprintln!("no local GGUF fixture found - skipping real-file test");
        return;
    };
    let bytes = std::fs::read(path).expect("read fixture");
    let f = GgufFile::parse(&bytes).expect("real GGUF parses");
    assert!(
        f.architecture().is_some(),
        "real files declare an architecture"
    );
    assert!(!f.tensors.is_empty());
    // every tensor must be sized or explicitly unsizable - and none past EOF
    // (validate_tensor_bounds already ran inside parse)
    eprintln!(
        "parsed {}: arch={:?} tensors={} kv={}",
        path.display(),
        f.architecture(),
        f.tensors.len(),
        f.metadata.len()
    );
}
