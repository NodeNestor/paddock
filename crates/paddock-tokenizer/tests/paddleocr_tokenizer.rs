//! PaddleOCR-VL tokenizer gate: the official GGUF declares
//! `tokenizer.ggml.model = "llama"` with vocab + rank scores but no merges -
//! there is no faithful rebuild from the GGUF alone, so the family reads the
//! checkpoint's tokenizer.json as a SIDECAR next to the weights (the
//! converted SPM-flavoured BPE vLLM itself runs).
//!
//! Fixtures come from the checkpoint's own processor: a round-trip corpus
//! (Nordic + CJK + byte-fallback + literal-special-lookalikes) and the six
//! task-prompt renderings with their exact input ids. Skips cleanly when the
//! model or fixtures are absent.

use paddock_models::mapped::MappedGguf;
use paddock_tokenizer::{GgufTokenizer, TokenizerError};

const IMAGE_TOKEN: u32 = 100295;

fn model_path() -> std::path::PathBuf {
    std::env::var("PADDLEOCR_VL_GGUF")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            std::path::PathBuf::from("/models/PaddleOCR-VL-1.6-GGUF/PaddleOCR-VL-1.6-GGUF.gguf")
        })
}

fn fixtures_path() -> std::path::PathBuf {
    std::env::var("PADDLEOCR_VL_FIXTURES")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            std::path::PathBuf::from("/models/ocr-battery/paddle-oracle/template_fixtures.json")
        })
}

fn load() -> Option<(GgufTokenizer, serde_json::Value)> {
    let (mp, fp) = (model_path(), fixtures_path());
    if !mp.exists() || !fp.exists() {
        eprintln!("model {mp:?} or fixtures {fp:?} missing - skipping");
        return None;
    }
    let map = MappedGguf::open(&mp).expect("open gguf");

    // the GGUF-only path must refuse loudly, not build something wrong
    match GgufTokenizer::from_gguf(map.gguf()) {
        Err(TokenizerError::UnsupportedModel(m)) => assert_eq!(m, "llama"),
        other => panic!("expected UnsupportedModel from the GGUF-only path, got {other:?}"),
    }

    let tok = GgufTokenizer::from_gguf_with_sidecar(map.gguf(), mp.parent().unwrap())
        .expect("sidecar tokenizer");
    let fixtures =
        serde_json::from_str(&std::fs::read_to_string(&fp).expect("fixtures")).expect("json");
    Some((tok, fixtures))
}

#[test]
fn sidecar_builds_with_the_gguf_decode_contract() {
    let Some((tok, _)) = load() else { return };
    assert_eq!(tok.bos_id, Some(1));
    assert_eq!(tok.eos_id, Some(2));
    assert!(
        !tok.add_bos,
        "the ERNIE template emits <|begin_of_sentence|> itself"
    );
    assert!(
        tok.chat_template.is_some(),
        "GGUF carries the family template"
    );
    assert_eq!(
        tok.vocab_size, 103_424,
        "model logit width, padding rows included"
    );
    assert_eq!(tok.token_to_id("<|IMAGE_PLACEHOLDER|>"), Some(IMAGE_TOKEN));
    assert_eq!(tok.token_to_id("<|begin_of_sentence|>"), Some(100_273));
}

/// Every corpus line must encode to the reference ids exactly - one wrong id
/// is a different prompt on the first forward.
#[test]
fn round_trip_matches_the_reference_tokenizer() {
    let Some((tok, fixtures)) = load() else {
        return;
    };
    for case in fixtures["round_trip"].as_array().unwrap() {
        let text = case["text"].as_str().unwrap();
        let want: Vec<u32> = case["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let got = tok.encode(text).expect("encode");
        assert_eq!(got, want, "ids diverged for {text:?}");
        let decoded = tok.decode(&got, false).expect("decode");
        assert_eq!(
            decoded,
            case["decoded"].as_str().unwrap(),
            "decode diverged for {text:?}"
        );
    }
}

/// The six task-prompt renderings (and the text-only / multi-turn shapes)
/// tokenize to the processor's exact ids. Image expansion is the engine's
/// job, so the fixture's 144-token image run is collapsed back to the single
/// <|IMAGE_PLACEHOLDER|> the rendered string carries.
#[test]
fn task_prompt_renderings_tokenize_exactly() {
    let Some((tok, fixtures)) = load() else {
        return;
    };

    let collapse = |ids: &[u32]| {
        let mut out = Vec::with_capacity(ids.len());
        for &id in ids {
            if id == IMAGE_TOKEN && out.last() == Some(&IMAGE_TOKEN) {
                continue;
            }
            out.push(id);
        }
        out
    };

    for (task, fix) in fixtures["tasks"].as_object().unwrap() {
        let rendered = fix["rendered"].as_str().unwrap();
        let want: Vec<u32> = fix["input_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let got = tok.encode(rendered).expect("encode");
        assert_eq!(got, collapse(&want), "ids diverged for task {task:?}");
    }

    for key in ["text_only", "multi_turn"] {
        let fix = &fixtures[key];
        let want: Vec<u32> = fix["input_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let got = tok
            .encode(fix["rendered"].as_str().unwrap())
            .expect("encode");
        assert_eq!(got, want, "ids diverged for {key}");
    }
}
