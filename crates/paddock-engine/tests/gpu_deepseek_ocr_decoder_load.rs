//! Loader milestone for the DeepSeek-OCR decoder: open the real
//! `Unlimited-OCR-Q8_0.gguf`, bring every decoder plane resident - Q8_0
//! attention/dense/lm_head planes, repacked Q8_0 expert stacks, F32 routers -
//! and check the declared geometry survived the trip. The forward graph and
//! the R-SWA pool are the next commits; this is the "weights resident with an
//! honest ledger" gate, same discipline as the whisper and tower loaders.
//!
//! Heavy (uploads ~3.1 GB) - gated on the model file, a built pack, and a
//! CUDA device; skips cleanly like the sibling load tests.

mod common;

use std::sync::Arc;

use paddock_engine::gpu_model::deepseek_ocr::GpuDeepseekOcr;
use paddock_models::mapped::MappedGguf;

fn model_path() -> Option<std::path::PathBuf> {
    if let Ok(p) = std::env::var("UNLIMITED_OCR_GGUF") {
        let p = std::path::PathBuf::from(p);
        return p.exists().then_some(p);
    }
    common::model_roots().iter().find_map(|r| {
        let p = r.join("Unlimited-OCR-GGUF").join("Unlimited-OCR-Q8_0.gguf");
        p.exists().then_some(p)
    })
}

#[test]
fn loads_the_decoder_and_geometry_matches_the_design() {
    let Some(path) = model_path() else {
        common::missing("no Unlimited-OCR GGUF (set UNLIMITED_OCR_GGUF)");
        return;
    };
    let Some(exec) = common::gpu_arc() else {
        return;
    };

    let map = MappedGguf::open(&path).expect("open gguf");
    let m = GpuDeepseekOcr::load(Arc::clone(&exec), &map, 32_768).expect("load decoder");

    // Geometry is already pinned metadata-level by gpu_deepseek_ocr_hparams;
    // here the same numbers are asserted through the loaded model, so a
    // loader that quietly re-derived anything diverges visibly.
    assert_eq!(m.hp.n_layer, 12);
    assert_eq!(m.hp.n_embd, 1280);
    assert_eq!(m.hp.n_head, 10);
    assert_eq!(m.hp.n_head_kv, 10, "G=1 - plain MHA");
    assert_eq!(m.hp.head_dim, 128, "computed, never the file's rope key");
    assert_eq!(m.hp.n_expert, 64);
    assert_eq!(m.hp.n_expert_used, 6);
    assert_eq!(m.hp.shexp_ff(), 1792);
    assert_eq!(m.hp.rswa_window, Some(128));
    assert_eq!(m.max_ctx, 32_768);

    // The ledger: a 3.1 GB file's planes plus repack overhead must land in
    // the same ballpark - an empty ledger means the free-VRAM probes lied,
    // and a 2x one means something got duplicated.
    let file_gb = std::fs::metadata(&path).expect("stat").len() as f64 / 1e9;
    let resident_gb = m.weights_bytes as f64 / 1e9;
    assert!(
        resident_gb > 0.8 * file_gb && resident_gb < 1.6 * file_gb,
        "decoder resident {resident_gb:.2} GB vs file {file_gb:.2} GB"
    );
    eprintln!("decoder resident {resident_gb:.2} GB (file {file_gb:.2} GB)");

    // R-SWA is the family's point - the fit arithmetic must pin.
    assert_eq!(m.kv_rows(907, 4), 911, "warmup still grows");
    assert_eq!(m.kv_rows(907, 1_000_000), 907 + 128, "ring must pin KV");
}
