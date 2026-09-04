//! Render the gemma4 chat template for one user message and print the exact
//! prompt string + token ids - diffed against `llama-completion
//! --verbose-prompt` (minja) to pin template-engine divergences.
//!
//! Usage: gemma4_template_check <model.gguf> [message]
// A development probe: it runs on a box its author is looking at, and a
// failure should stop it where it happened rather than be reported.
#![allow(clippy::unwrap_used)]

use paddock_models::mapped::MappedGguf;
use paddock_tokenizer::GgufTokenizer;

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .expect("usage: gemma4_template_check <gguf> [msg]");
    let msg = args
        .next()
        .unwrap_or_else(|| "Vad heter Sveriges huvudstad? Svara med ett ord.".to_owned());

    let map = MappedGguf::open(path.as_ref()).expect("open");
    let tok = GgufTokenizer::from_gguf(map.gguf()).expect("tokenizer");
    let template = tok.chat_template.clone().expect("model ships a template");

    let messages = vec![serde_json::json!({"role": "user", "content": msg})];
    let prompt =
        paddock_runner::chat_template::render(&template, &messages, None, None).expect("render");
    println!("prompt: {prompt:?}");
    let mut ids = Vec::new();
    if tok.add_bos {
        ids.push(tok.bos_id.expect("bos"));
    }
    ids.extend(tok.encode(&prompt).expect("encode"));
    println!("n_tokens(with bos): {}", ids.len());
    for id in &ids {
        println!("{id} -> {:?}", tok.id_to_token(*id).unwrap_or_default());
    }
}
