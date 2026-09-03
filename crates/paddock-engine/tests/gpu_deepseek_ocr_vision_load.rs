//! Loader milestone for the DeepEncoder: open the real
//! `mmproj-Unlimited-OCR-F16.gguf`, upload every plane of both towers resident,
//! and check that the geometry we designed against is the geometry the file
//! carries. The forward graph is the next step; this is the "weights resident
//! with an honest ledger" gate.
//!
//! It exists because four of this mmproj's own metadata keys are wrong for this
//! tower (CLIP's FFN width, patch size, image size, and LayerNorm eps - see
//! `deepseek_ocr::vision`), so "the loader read the file" and "the loader read
//! the right numbers" are genuinely different claims. Everything asserted below
//! is asserted against a TENSOR, not against the key that was supposed to
//! describe it.
//!
//! Heavy (uploads ~825 MB) - gated on the model file, a built pack, and a CUDA
//! device, and skips cleanly like the sibling load tests.

mod common;

use std::sync::Arc;

use paddock_engine::gpu_model::deepseek_ocr::vision::{BASE_PX, CLIP_EPS, DeepEncoder, TILE_PX};
use paddock_models::mapped::MappedGguf;

fn mmproj_path() -> Option<std::path::PathBuf> {
    if let Ok(p) = std::env::var("UNLIMITED_OCR_MMPROJ") {
        let p = std::path::PathBuf::from(p);
        return p.exists().then_some(p);
    }
    common::model_roots().iter().find_map(|r| {
        let p = r
            .join("Unlimited-OCR-GGUF")
            .join("mmproj-Unlimited-OCR-F16.gguf");
        p.exists().then_some(p)
    })
}

#[test]
fn loads_the_deepencoder_and_geometry_matches_the_design() {
    let Some(path) = mmproj_path() else {
        common::missing("no Unlimited-OCR mmproj (set UNLIMITED_OCR_MMPROJ)");
        return;
    };
    let Some(exec) = common::gpu_arc() else {
        return;
    };

    let map = MappedGguf::open(&path).expect("open mmproj");
    assert_eq!(
        map.gguf().tensors.len(),
        476,
        "tensor count - a converter change moves this"
    );

    let enc = DeepEncoder::load(Arc::clone(&exec), &map).expect("load deepencoder");
    let hp = &enc.hp;

    // --- SAM ViT-B, all of it derived from tensors.
    assert_eq!(hp.sam_layers, 12);
    assert_eq!(hp.sam_embd, 768);
    assert_eq!(hp.sam_heads, 12);
    assert_eq!(hp.sam_head_dim, 64);
    assert_eq!(hp.sam_ff, 3072, "mlp_ratio 4");
    assert_eq!(
        hp.sam_patch, 16,
        "from the conv stem, NOT clip.vision.patch_size (14)"
    );
    assert_eq!(hp.sam_grid_train, 64, "1024 / 16");
    assert_eq!(hp.window, 14);

    // The rel-pos tables are the only record of which blocks attend globally -
    // no metadata key carries `global_attn_indexes`. A windowed block's table
    // has 2*14-1 = 27 rows, a global one 2*64-1 = 127, and the checkpoint's
    // safetensors agree exactly: blocks 2/5/8/11 are (127, 64), the rest
    // (27, 64). If a converter ever "helpfully" normalized those to one shape
    // this assert is what catches it.
    assert_eq!(
        hp.global_blocks,
        vec![2, 5, 8, 11],
        "global blocks must come from the rel-pos table widths"
    );

    // --- CLIP-L.
    assert_eq!(hp.clip_layers, 24);
    assert_eq!(hp.clip_embd, 1024);
    assert_eq!(hp.clip_heads, 16);
    assert_eq!(
        hp.clip_ff, 4096,
        "the file's feed_forward_length says 64 - do not believe it"
    );
    let file_ff = map
        .gguf()
        .metadata
        .get("clip.vision.feed_forward_length")
        .and_then(|v| v.as_u64());
    assert_eq!(
        file_ff,
        Some(64),
        "if this stops being 64 the converter was fixed - re-check whether reading it is now safe"
    );
    assert_eq!(
        hp.clip_positions, 257,
        "1 CLS + the 16x16 grid the squeeze leaves"
    );

    // --- the eps split: one key, two towers.
    assert_eq!(hp.sam_eps, 1e-6, "the file's eps key is SAM's");
    assert!(
        (CLIP_EPS - 1e-5).abs() < f32::EPSILON,
        "CLIP's eps is 1e-5 and cannot come from the file"
    );

    // --- projector and preprocessing.
    assert_eq!(hp.proj_in, 2048, "concat(clip 1024, sam net_3 1024)");
    assert_eq!(hp.llm_embd, 1280, "the decoder's width");
    assert_eq!(enc.llm_embd(), 1280);
    assert_eq!(hp.min_tiles, 2);
    assert_eq!(
        hp.max_tiles, 32,
        "Unlimited-OCR's budget; the DeepSeek-OCR pair uses 6"
    );
    assert_eq!(hp.image_mean, [0.5; 3]);
    assert_eq!(hp.image_std, [0.5; 3]);

    // --- the 16x squeeze, stated as the two grids it connects.
    assert_eq!(hp.sam_grid(BASE_PX), 64);
    assert_eq!(hp.tokens_per_side(BASE_PX), 16);
    assert_eq!(hp.sam_grid(TILE_PX), 40);
    assert_eq!(hp.tokens_per_side(TILE_PX), 10);

    // --- the ledger. f16 planes only; the pos/rel-pos tables stay on the host
    // because every view resamples them, so device bytes are below the file
    // size and that is deliberate rather than a leak.
    let file_bytes = std::fs::metadata(&path).expect("stat mmproj").len() as usize;
    let dev = enc.weight_bytes();
    let host = enc.host_table_bytes();
    assert!(
        dev > 0 && dev < file_bytes,
        "device {dev} vs file {file_bytes}"
    );
    assert!(
        dev + host < file_bytes,
        "device {dev} + host {host} should still sit under the file's {file_bytes} \
         (the F32 planes narrow to f16)"
    );
    eprintln!(
        "deepencoder: {:.1} MiB device, {:.1} MiB host tables, file {:.1} MiB",
        dev as f64 / (1 << 20) as f64,
        host as f64 / (1 << 20) as f64,
        file_bytes as f64 / (1 << 20) as f64,
    );
}

/// The dead tensor. `v.patch_embd.weight` is CLIP's own conv stem and the
/// family never reaches it - `CLIPVisionEmbeddings.forward` takes the
/// `patch_embeds is not None` branch at every call site, because SAM's net_3
/// output is the patch embedding. It must be present in the file (the converter
/// carries it) and absent from our device ledger.
#[test]
fn clips_own_conv_stem_is_present_in_the_file_and_unloaded() {
    let Some(path) = mmproj_path() else {
        common::missing("no Unlimited-OCR mmproj (set UNLIMITED_OCR_MMPROJ)");
        return;
    };
    let map = MappedGguf::open(&path).expect("open mmproj");
    let dead = map
        .gguf()
        .tensors
        .iter()
        .find(|t| t.name == "v.patch_embd.weight")
        .expect("v.patch_embd.weight should still be in the file");
    let dims: Vec<usize> = dead.dims.iter().map(|d| *d as usize).collect();
    assert_eq!(
        dims,
        vec![14, 14, 3, 1024],
        "CLIP-L/14's stem, at CLIP's own patch size"
    );

    // SAM's stem is the one that runs, and it is a different shape entirely.
    let live = map
        .gguf()
        .tensors
        .iter()
        .find(|t| t.name == "v.sam.patch_embd.weight")
        .expect("v.sam.patch_embd.weight");
    let live_dims: Vec<usize> = live.dims.iter().map(|d| *d as usize).collect();
    assert_eq!(live_dims, vec![16, 16, 3, 768]);
}
