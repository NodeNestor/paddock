//! Print the probe's ModelReport for a GGUF - what the catalog/estimator
//! will see. `cargo run -p paddock-models --example probe -- <file.gguf>`
// A development probe: it runs on a box its author is looking at, and a
// failure should stop it where it happened rather than be reported.
#![allow(clippy::unwrap_used)]

fn main() {
    let path = std::env::args().nth(1).expect("usage: probe <file.gguf>");
    match paddock_models::probe::probe_path(std::path::Path::new(&path)) {
        Ok(r) => println!("{r:#?}"),
        Err(e) => {
            eprintln!("probe failed: {e}");
            std::process::exit(1);
        }
    }
}
