//! End-to-end smoke for the Qwen3.5 serving path: real GGUF -> qwen35 tokenizer
//! + GpuQwen35 -> greedy generation -> detokenize. Not a correctness proof (that
//!   is the HF greedy-parity gate), but coherent English continuation is a strong
//!   signal the whole hybrid forward wired up right. Heavy + CUDA-gated.

use std::sync::Arc;

use paddock_engine::gpu::GpuExecutor;
use paddock_engine::gpu_model::qwen35::GpuQwen35;
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
fn qwen35_greedy_generates_text() {
    let pack = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../paddock-engine/../../packs/cuda/build/pd-cuda-sm86.dll");
    if !pack.exists() {
        eprintln!("pack {pack:?} not built - skipping");
        return;
    }
    let path = model_path();
    if !path.exists() {
        eprintln!("model {path:?} missing - skipping");
        return;
    }
    let exec = match GpuExecutor::new(0, &pack) {
        Ok(e) => Arc::new(e),
        Err(e) => {
            eprintln!("no CUDA ({e}) - skipping");
            return;
        }
    };

    let map = MappedGguf::open(&path).expect("open gguf");
    let tok = GgufTokenizer::from_gguf(map.gguf()).expect("tokenizer");
    let mut model = GpuQwen35::load(exec, &map, 4096).expect("load qwen35");

    let prompt = "The capital of France is";
    let ids = tok.encode(prompt).expect("encode");
    eprintln!("prompt {prompt:?} -> {} tokens", ids.len());

    let out = model.generate_greedy(&ids, 12, None).expect("generate");
    let text = tok.decode(&out, true).expect("decode");
    eprintln!("continuation: {text:?}");

    assert!(!out.is_empty(), "no tokens generated");
    // must not collapse to a single repeated id
    assert!(
        out.iter().collect::<std::collections::HashSet<_>>().len() > 1,
        "degenerate (all-same) generation: {out:?}"
    );
}
