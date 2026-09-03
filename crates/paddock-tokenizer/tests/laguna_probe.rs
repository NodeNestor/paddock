//! Ad-hoc laguna tokenizer probe against the real GGUF (no GPU). Runs only
//! when LAGUNA_GGUF is set:
//!   LAGUNA_GGUF=E:\...\Laguna-XS-2.1-Q4_K_M.gguf \
//!   cargo test -p paddock-tokenizer --test laguna_probe -- --nocapture

use paddock_models::mapped::MappedGguf;
use paddock_tokenizer::GgufTokenizer;

#[test]
fn probe_real_gguf() {
    let Ok(path) = std::env::var("LAGUNA_GGUF") else {
        eprintln!("LAGUNA_GGUF unset - skipping");
        return;
    };
    let map = MappedGguf::open(path.as_ref()).expect("open gguf");
    let tok = GgufTokenizer::from_gguf(map.gguf()).expect("tokenizer");
    for text in [
        "The capital of France is",
        "hello world",
        "Reply with exactly: ok",
        "line one\nline two",
    ] {
        let ids = tok.encode(text).expect("encode");
        let round = tok.decode(&ids, false).unwrap_or_default();
        eprintln!("{text:?} -> {ids:?} -> {round:?}");
        for &id in &ids {
            eprintln!("   {id} = {:?}", tok.id_to_token(id));
        }
    }
    // llama.cpp reference (verified live): with BOS,
    // "The capital of France is" = [2, 785, 9626, 377, 15360, 395]
    let ids = tok.encode("The capital of France is").expect("encode");
    assert_eq!(
        ids,
        vec![785, 9626, 377, 15360, 395],
        "laguna encode parity"
    );
}
