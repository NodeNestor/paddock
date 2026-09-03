//! Parse + render a model's embedded chat template through the same minijinja
//! setup the server uses, and report exactly where it fails.
//!
//! Every new family arrives with an HF Jinja template written against Python's
//! Jinja2, and the ones that use constructs minijinja lacks fail at request
//! time with a one-line "parse error ... (in chat:1)" - templates ship as a
//! single line, so that says nothing about WHERE. This bisects to the offending
//! span and prints the source around it.
//!
//! Usage: chat_template_lint <model.gguf> [--render]

use paddock_models::mapped::MappedGguf;

fn main() {
    let mut args = std::env::args().skip(1);
    let model = args
        .next()
        .expect("usage: chat_template_lint <model.gguf> [--render]");
    let do_render = args.any(|a| a == "--render");

    let map = MappedGguf::open(model.as_ref()).expect("open model");
    let tok = paddock_tokenizer::GgufTokenizer::from_gguf(map.gguf()).expect("tokenizer");
    let Some(tmpl) = tok.chat_template.clone() else {
        println!("model carries no tokenizer.chat_template");
        return;
    };
    println!(
        "template: {} bytes, {} lines",
        tmpl.len(),
        tmpl.lines().count()
    );

    let msgs = vec![serde_json::json!({"role": "user", "content": "hi"})];
    match paddock_runner::chat_template::render(&tmpl, &msgs, None, None) {
        Ok(out) if do_render => println!("--- rendered ---\n{out}\n--- end ---"),
        Ok(out) => println!(
            "OK - renders ({} chars); pass --render to print it",
            out.len()
        ),
        Err(e) => {
            println!("FAIL: {e}");
            // Bisect on the PREFIX: the shortest prefix that already fails
            // ends at (or just past) the construct minijinja rejects. Jinja
            // blocks can be cut mid-tag, so an unclosed-block error is
            // expected noise - we look for the first prefix whose error is
            // the same class as the whole template's.
            let bytes = tmpl.as_bytes();
            let mut lo = 0usize;
            let mut hi = bytes.len();
            let fails = |n: usize| -> Option<String> {
                let Some(s) = tmpl.get(..n) else { return None };
                paddock_runner::chat_template::render(s, &msgs, None, None).err()
            };
            // Compare the full message minus the location. Matching only
            // "syntax error" would hit every mid-tag truncation and bisect to
            // byte 2.
            let want = e.split(" (in chat").next().unwrap_or(&e).to_owned();
            while lo + 1 < hi {
                let mid = (lo + hi) / 2;
                match fails(mid) {
                    Some(err) if err.contains(&want) => hi = mid,
                    _ => lo = mid,
                }
            }
            let at = hi.min(tmpl.len());
            let from = at.saturating_sub(160);
            println!(
                "\nfirst failing prefix ends at byte {at}; source around it:\n...{}<<<HERE>>>{}...",
                &tmpl[from..at],
                &tmpl[at..(at + 80).min(tmpl.len())]
            );
        }
    }
}
