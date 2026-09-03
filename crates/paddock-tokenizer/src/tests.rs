//! Synthetic construction test + a gated real-model test against gpt-oss-20b.
//! Golden parity is the greedy-parity suite's job (paddock-bench, against a
//! llama.cpp release on the identical GGUF); these tests prove construction,
//! round-tripping, and special-token handling.

use std::collections::HashMap;

use paddock_models::gguf::{GgufFile, Value};

use crate::GgufTokenizer;

fn str_array(items: &[&str]) -> Value {
    Value::Array(items.iter().map(|s| Value::Str((*s).to_string())).collect())
}

/// Hand-built minimal byte-level BPE model: enough to merge "hello".
fn tiny_gguf() -> GgufFile {
    let mut metadata = HashMap::new();
    metadata.insert("tokenizer.ggml.model".into(), Value::Str("gpt2".into()));
    metadata.insert("tokenizer.ggml.pre".into(), Value::Str("gpt-4o".into()));
    metadata.insert(
        "tokenizer.ggml.tokens".into(),
        str_array(&["h", "e", "l", "o", "he", "ll", "hell", "hello", "<|end|>"]),
    );
    metadata.insert(
        "tokenizer.ggml.merges".into(),
        str_array(&["h e", "l l", "he ll", "hell o"]),
    );
    metadata.insert(
        "tokenizer.ggml.token_type".into(),
        Value::Array(vec![
            Value::U32(1),
            Value::U32(1),
            Value::U32(1),
            Value::U32(1),
            Value::U32(1),
            Value::U32(1),
            Value::U32(1),
            Value::U32(1),
            Value::U32(3), // <|end|> is control
        ]),
    );
    metadata.insert("tokenizer.ggml.eos_token_id".into(), Value::U32(8));
    GgufFile {
        version: 3,
        alignment: 32,
        metadata,
        tensors: vec![],
        data_offset: 0,
    }
}

#[test]
fn builds_from_gguf_and_merges_bpe() {
    let tok = GgufTokenizer::from_gguf(&tiny_gguf()).expect("builds");
    assert_eq!(tok.vocab_size, 9);
    assert_eq!(tok.eos_id, Some(8));
    // full merge chain: h e -> he, l l -> ll, he ll -> hell, hell o -> hello
    assert_eq!(tok.encode("hello").expect("encodes"), vec![7]);
}

#[test]
fn control_tokens_survive_as_single_ids() {
    let tok = GgufTokenizer::from_gguf(&tiny_gguf()).expect("builds");
    assert_eq!(tok.token_to_id("<|end|>"), Some(8));
    let ids = tok.encode("hello<|end|>").expect("encodes");
    assert_eq!(ids, vec![7, 8], "control token must not be split by BPE");
    // and skip_special drops it on decode
    assert_eq!(tok.decode(&ids, true).expect("decodes"), "hello");
}

#[test]
fn unknown_pre_tokenizer_is_a_hard_error() {
    let mut f = tiny_gguf();
    f.metadata.insert(
        "tokenizer.ggml.pre".into(),
        Value::Str("some-new-family".into()),
    );
    let err = GgufTokenizer::from_gguf(&f).expect_err("must refuse");
    assert!(matches!(err, crate::TokenizerError::UnknownPreTokenizer(_)));
}

/// Whisper-family construction: the GGUF (our own schema -
/// our whisper converter) embeds the complete HF
/// tokenizer.json, so `from_gguf` takes the embedded-json arm instead of the
/// tokenizer.ggml.* rebuild. Gated on the local KB-Whisper conversion.
#[test]
fn whisper_tokenizer_builds_from_embedded_json() {
    let path = std::env::var("WHISPER_GGUF")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            std::path::PathBuf::from("/models/kb-whisper-large/kb-whisper-large-f16.gguf")
        });
    if !path.exists() {
        eprintln!("KB-Whisper GGUF not present - skipping whisper tokenizer test");
        return;
    }

    use std::io::Read;
    let file = std::fs::File::open(&path).expect("open");
    let file_len = file.metadata().expect("meta").len();
    let mut head = vec![0u8; (64 << 20).min(file_len) as usize];
    std::io::BufReader::new(file)
        .read_exact(&mut head)
        .expect("read prefix");
    let f = GgufFile::parse_prefix(&head, file_len).expect("parses");

    let tok = GgufTokenizer::from_gguf(&f).expect("builds from embedded tokenizer.json");
    // decode contract: generation ends at <|endoftext|>; prompting is
    // explicit special tokens, never a text-side BOS or a chat template
    assert_eq!(tok.eos_id, Some(50_257), "eot from whisper.token.eot");
    assert!(!tok.add_bos);
    assert!(tok.chat_template.is_none());
    // the special-token soup must resolve as single ids (the whole point of
    // shipping tokenizer.json instead of rebuilding): sot, the language
    // tokens the Nordic lane serves, and the task token
    assert_eq!(tok.token_to_id("<|startoftranscript|>"), Some(50_258));
    assert_eq!(tok.token_to_id("<|sv|>"), Some(50_273));
    assert_eq!(tok.token_to_id("<|no|>"), Some(50_288));
    assert_eq!(tok.token_to_id("<|da|>"), Some(50_285));
    assert_eq!(tok.token_to_id("<|transcribe|>"), Some(50_360));

    // transcripts must round-trip faithfully (Swedish orthography included).
    // whisper's tokenizer.json pre-tokenizes with GPT-2 add_prefix_space: a
    // string not already leading with a space gains exactly one - that is
    // the model's native transcript shape (references strip it at the
    // endpoint layer, which is the earlier serving concern, not the tokenizer's).
    let samples = [
        "Det var en gång en katt som hette Måns.",
        "Smörgåsbord på fjärden - åäö ÅÄÖ",
        "  two leading spaces and trailing  ",
        "12345 tokens: 1 22 333 4444",
    ];
    for s in samples {
        let ids = tok.encode(s).expect("encode");
        assert!(!ids.is_empty());
        let back = tok.decode(&ids, false).expect("decode");
        let want = if s.starts_with(' ') {
            s.to_string()
        } else {
            format!(" {s}")
        };
        assert_eq!(
            back, want,
            "round-trip must be add-prefix-space faithful for {s:?}"
        );
    }
}

/// Real-model test, gated on the local gpt-oss download.
#[test]
fn gpt_oss_tokenizer_round_trips() {
    let Some(home) = std::env::var_os("USERPROFILE") else {
        return;
    };
    let path = std::path::PathBuf::from(home).join("paddock/models/gpt-oss-20b-mxfp4.gguf");
    if !path.exists() {
        eprintln!("gpt-oss GGUF not present - skipping real-model tokenizer test");
        return;
    }

    // header (incl. full vocab) fits comfortably in a 64 MiB prefix
    use std::io::Read;
    let file = std::fs::File::open(&path).expect("open");
    let file_len = file.metadata().expect("meta").len();
    let mut head = vec![0u8; (64 << 20).min(file_len) as usize];
    std::io::BufReader::new(file)
        .read_exact(&mut head)
        .expect("read prefix");
    let f = GgufFile::parse_prefix(&head, file_len).expect("parses");

    let tok = GgufTokenizer::from_gguf(&f).expect("builds from real vocab");
    assert_eq!(tok.vocab_size, 201_088);
    assert_eq!(tok.eos_id, Some(200_002));

    // eos id must resolve to a real token and encode as itself
    let eos_tok = tok.id_to_token(200_002).expect("eos resolves");
    assert_eq!(tok.token_to_id(&eos_tok), Some(200_002));

    // round-trip identity across the nasty cases: unicode, emoji, CRLF,
    // leading/trailing whitespace, code, contractions (the (?i:) branch)
    let samples = [
        "Hello, world!",
        "  two leading spaces and trailing  ",
        "Smörgåsbord på fjärden - åäö ÅÄÖ",
        "🦀🚀 emoji and ¡unicode!",
        "fn main() { println!(\"hi\"); }",
        "line one\r\nline two\n\nline four",
        "I'LL don't WE'RE it's",
        "12345 tokens: 1 22 333 4444",
    ];
    for s in samples {
        let ids = tok.encode(s).expect("encode");
        assert!(!ids.is_empty());
        let back = tok.decode(&ids, false).expect("decode");
        assert_eq!(back, s, "round-trip must be lossless for {s:?}");
    }
}

/// StreamDecoder must match the full decode at every step - it feeds the
/// streaming handlers' emit offsets, so a single divergent byte would ship
/// wrong deltas. Tiny-vocab arm: walk id sequences over the whole vocab
/// (specials included) both skip_special ways.
#[test]
fn stream_decoder_matches_full_decode_tiny() {
    let tok = GgufTokenizer::from_gguf(&tiny_gguf()).expect("builds");
    let n = tok.vocab_size as u32;
    // a repeating walk long enough to cross several flush seams
    let ids: Vec<u32> = (0..400u32).map(|i| (i * 7 + 3) % n).collect();
    for skip in [false, true] {
        let mut sd = tok.stream_decoder(skip);
        for k in 0..ids.len() {
            let got = sd.push(&tok, ids[k]);
            let want = tok.decode(&ids[..=k], skip).expect("decode");
            assert_eq!(got, want, "step {k} skip_special={skip}");
        }
    }
}

/// Real-model arm (gated on the local Qwen3.8 GGUF): multilingual + emoji
/// content forces multi-byte scalars split across byte-BPE tokens, which is
/// exactly the seam hazard the flush back-off must absorb.
#[test]
fn stream_decoder_matches_full_decode_qwen38() {
    let path = std::path::PathBuf::from("/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q8_0.gguf");
    if !path.exists() {
        eprintln!("Qwen3.8 GGUF not present - skipping stream-decoder real test");
        return;
    }
    use std::io::Read;
    let file = std::fs::File::open(&path).expect("open");
    let file_len = file.metadata().expect("meta").len();
    let mut head = vec![0u8; (64 << 20).min(file_len) as usize];
    std::io::BufReader::new(file)
        .read_exact(&mut head)
        .expect("read prefix");
    let f = GgufFile::parse_prefix(&head, file_len).expect("parses");
    let tok = GgufTokenizer::from_gguf(&f).expect("builds");

    let corpus = "Smörgåsbord på fjärden - åäö. 计算机的历史始于算盘。\
                  🦀🚀 fn main() { println!(\"héllo\"); } Привет, мир! \
                  日本語のテキストと한국어 텍스트가 섞여 있다. \u{12000}\u{12001}";
    let ids = tok.encode(&corpus.repeat(4)).expect("encode");
    assert!(ids.len() > 200, "corpus should cross several flush seams");
    for skip in [false, true] {
        let mut sd = tok.stream_decoder(skip);
        for k in 0..ids.len() {
            let got = sd.push(&tok, ids[k]);
            let want = tok.decode(&ids[..=k], skip).expect("decode");
            assert_eq!(got, want, "step {k} skip_special={skip}");
        }
    }
}
