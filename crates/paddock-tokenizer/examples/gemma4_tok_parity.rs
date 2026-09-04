//! Gemma 4 tokenizer parity probe: encode argv[2] with the GGUF-constructed
//! tokenizer from argv[1] and print one token id per line (BOS included when
//! the model asks for it) - the same shape `llama-tokenize` prints, so the
//! parity harness can diff the two directly.
//!
//! Usage: gemma4_tok_parity <model.gguf> <text>
// A development probe: it runs on a box its author is looking at, and a
// failure should stop it where it happened rather than be reported.
#![allow(clippy::unwrap_used)]

use paddock_models::mapped::MappedGguf;
use paddock_tokenizer::GgufTokenizer;

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .expect("usage: gemma4_tok_parity <model.gguf> <text>");
    let text = args
        .next()
        .expect("usage: gemma4_tok_parity <model.gguf> <text>");

    let map = MappedGguf::open(path.as_ref()).expect("open gguf");
    let tok = GgufTokenizer::from_gguf(map.gguf()).expect("build tokenizer");

    if tok.add_bos
        && let Some(bos) = tok.bos_id
    {
        println!("{bos}");
    }
    for id in tok.encode(&text).expect("encode") {
        println!("{id}");
    }
}
