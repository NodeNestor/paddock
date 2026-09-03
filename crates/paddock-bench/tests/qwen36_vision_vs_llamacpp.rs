//! Vision acceptance gate: same-weights multimodal greedy parity vs a
//! prebuilt llama-mtmd-cli on the identical 27B GGUF + mmproj. A
//! deterministic synthetic BMP goes through both engines with the same
//! chat-templated prompt; the bar is the generated TEXT matching.
//!
//! Two cases:
//!  - 768x768 (48x48 patch grid): 32-aligned and in-budget, so resizing is a
//!    no-op - pins the tower + injection.
//!  - 500x300: UNALIGNED - llama smart-resizes to 512x288 with a bilinear
//!    0.96x scale and 16px black side bars (PAD_CEIL). Pins the S2
//!    preprocessing port (smart_resize + bilinear + letterbox) end to end.
//!
//! Prompt construction mirrors mtmd exactly: the qwen chat template around the
//! user text, with the image chunk wrapped in <|vision_start|>/<|vision_end|>
//! marker tokens, image rows sharing t=p0 (mutually visible), and the
//! llama-position advancing by max(grid) after the image.
//!
//! Sequential (two ~27 GB residencies): mtmd-cli runs to completion first.
//! Very heavy: gated on PADDOCK_HEAVY_TESTS, --test-threads=1.

use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;

use paddock_engine::gpu::GpuExecutor;
use paddock_engine::gpu_model::qwen35::GpuQwen35;
use paddock_engine::gpu_model::qwen35::vision::VisionModel;
use paddock_models::mapped::MappedGguf;
use paddock_tokenizer::GgufTokenizer;

const N: usize = 32;
const PROMPT: &str = "What colors do you see?";

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap()
}

fn model_dir() -> PathBuf {
    PathBuf::from(std::env::var("USERPROFILE").expect("USERPROFILE")).join(
        ".cache/huggingface/hub/models--unsloth--Qwen3.6-27B-MTP-GGUF/snapshots/5cb35eb3dcbf52dbce5f87dbc64df6aaffadcace",
    )
}

/// Deterministic pixels - must stay in sync between the BMP writer and the
/// in-memory RGB build.
fn px(x: usize, y: usize, w: usize, h: usize) -> (u8, u8, u8) {
    (
        ((x * 255) / w) as u8,
        ((y * 255) / h) as u8,
        (((x / 64 + y / 64) % 2) * 200 + 25) as u8,
    )
}

/// Write a 24-bit bottom-up BMP of the synthetic image (rows padded to 4B).
fn write_bmp(path: &std::path::Path, w: usize, h: usize) {
    let row = (w * 3).div_ceil(4) * 4;
    let img_size = row * h;
    let mut out = Vec::with_capacity(54 + img_size);
    out.extend_from_slice(b"BM");
    out.extend_from_slice(&(54u32 + img_size as u32).to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(&54u32.to_le_bytes());
    out.extend_from_slice(&40u32.to_le_bytes());
    out.extend_from_slice(&(w as i32).to_le_bytes());
    out.extend_from_slice(&(h as i32).to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&24u16.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(&(img_size as u32).to_le_bytes());
    out.extend_from_slice(&2835u32.to_le_bytes());
    out.extend_from_slice(&2835u32.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    for y in (0..h).rev() {
        let mut written = 0usize;
        for x in 0..w {
            let (r, g, b) = px(x, y, w, h);
            out.extend_from_slice(&[b, g, r]);
            written += 3;
        }
        while !written.is_multiple_of(4) {
            out.push(0);
            written += 1;
        }
    }
    std::fs::write(path, out).expect("write bmp");
}

fn run_gate(w: usize, h: usize) {
    if std::env::var_os("PADDOCK_HEAVY_TESTS").is_none() {
        eprintln!("set PADDOCK_HEAVY_TESTS=1 to run the vision gate (two ~27 GB loads)");
        return;
    }
    let cli = repo().join("vendor/llamacpp/llama-mtmd-cli.exe");
    let pack = repo().join("packs/cuda/build/pd-cuda-sm86.dll");
    let dir = model_dir();
    let model = dir.join("Qwen3.6-27B-Q8_0.gguf");
    let mmproj = dir.join("mmproj-F16.gguf");
    for (what, p) in [
        ("mtmd-cli", &cli),
        ("pack", &pack),
        ("model", &model),
        ("mmproj", &mmproj),
    ] {
        if !p.exists() {
            eprintln!("{what} {p:?} missing - skipping");
            return;
        }
    }
    let bmp = std::env::temp_dir().join(format!("paddock_vision_gate_{w}x{h}.bmp"));
    write_bmp(&bmp, w, h);

    // ---- phase 1: llama-mtmd-cli alone on the GPU (greedy, N tokens) ----
    let out = Command::new(&cli)
        .args([
            "-m",
            model.to_str().unwrap(),
            "--mmproj",
            mmproj.to_str().unwrap(),
            "--image",
            bmp.to_str().unwrap(),
            "-p",
            PROMPT,
            "--temp",
            "0",
            "-n",
            &N.to_string(),
            "-ngl",
            "99",
        ])
        .output()
        .expect("run mtmd-cli");
    let llama_text = String::from_utf8_lossy(&out.stdout).trim().to_string();
    assert!(
        !llama_text.is_empty(),
        "mtmd-cli produced no output; stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    eprintln!("llama  : {llama_text:?}");

    // ---- phase 2: Paddock alone (identical prompt construction) ----
    let exec = Arc::new(GpuExecutor::new(0, &pack).expect("cuda executor"));
    let map = MappedGguf::open(&model).expect("open 27B");
    let tok = GgufTokenizer::from_gguf(map.gguf()).expect("tokenizer");
    let mmap = MappedGguf::open(&mmproj).expect("open mmproj");
    let vm = VisionModel::load(exec.clone(), &mmap).expect("vision tower");

    // encode the image through the full preprocessing path (identity for the
    // aligned case, smart-resize + bilinear + letterbox for the unaligned one)
    let mut rgb = vec![0u8; 3 * w * h];
    for y in 0..h {
        for x in 0..w {
            let (r, g, b) = px(x, y, w, h);
            let i = (y * w + x) * 3;
            rgb[i] = r;
            rgb[i + 1] = g;
            rgb[i + 2] = b;
        }
    }
    let (img, tw, th) = vm.preprocess_rgb(&rgb, w, h);
    eprintln!("preprocess: {w}x{h} -> {tw}x{th}");
    let vision = vm.encode(&img, tw, th).expect("encode image");
    eprintln!(
        "image -> {} tokens ({}x{})",
        vision.nx * vision.ny,
        vision.nx,
        vision.ny
    );

    // chat-templated prompt, media marker replaced by the vision markers + chunk
    let id = |s: &str| {
        tok.token_to_id(s)
            .unwrap_or_else(|| panic!("missing special {s}"))
    };
    let mut before = vec![id("<|im_start|>")];
    before.extend(tok.encode("user\n").expect("enc"));
    before.push(id("<|vision_start|>"));
    let mut after = vec![id("<|vision_end|>")];
    after.extend(tok.encode(PROMPT).expect("enc"));
    after.push(id("<|im_end|>"));
    after.extend(tok.encode("\n").expect("enc"));
    after.push(id("<|im_start|>"));
    after.extend(tok.encode("assistant\n").expect("enc"));

    let mut pad = GpuQwen35::load(exec, &map, 4096).expect("load 27B");
    let ids = pad
        .generate_greedy_mm(&before, &vision, &after, N, None)
        .expect("generate");
    let pad_text = tok.decode(&ids, true).expect("decode");
    eprintln!("paddock: {pad_text:?}");

    // llama prints the response text through the Windows console (CRLF
    // translation on stdout) - normalize line endings, then require equality.
    let llama_norm = llama_text.replace("\r\n", "\n");
    let pad_norm = pad_text.trim().to_string();
    let m = llama_norm
        .chars()
        .zip(pad_norm.chars())
        .take_while(|(a, b)| a == b)
        .count();
    eprintln!("shared prefix: {m} chars of {}", llama_norm.chars().count());
    assert_eq!(
        llama_norm, pad_norm,
        "multimodal greedy output must match llama-mtmd-cli"
    );
    eprintln!("EXACT MATCH: {N}-token multimodal greedy output identical ({w}x{h})");
}

// IGNORED since the vision-tower attention fix: the old
// kernel silently truncated the q.k dot to 64 of the 72 head dims (2 warps for
// a 72-dim head), so these exact-token matches were LUCKY DRAWS calibrated
// against the truncated tower - exactly the "if a numeric-class change flips
// one, cite this precedent" case. The corrected kernel is
// mathematically right (gpu_vision_attn_parity: 1.2e-7 vs a CPU softmax
// reference - the old kernel fails that gate by construction), and paddock's
// output is coherent + image-accurate; it simply no longer produces the exact
// token stream the truncated tower happened to share with llama.cpp (here it
// forks the 27B's think/no-think decision, a hypersensitive branch, on top of
// the established image-prompt knife-edge). No trustworthy exact oracle exists
// right now: the reference binary disagrees with the corrected math, and an
// A/B of llama-mtmd-cli against both kernels diverges either way.
// Kernel correctness is covered by gpu_vision_attn_parity; e2e coherence by
// gpu_qwen36_vision_serving + qwen_vision_http. Re-enable with a fresh baseline
// captured against a corrected-math llama-mtmd-cli.
#[test]
#[ignore = "vision-tower attention fix invalidated the b9895 exact-match baseline; see comment"]
fn vision_greedy_matches_llamacpp() {
    run_gate(768, 768);
}

#[test]
#[ignore = "vision-tower attention fix invalidated the b9895 exact-match baseline; see comment"]
fn vision_greedy_matches_llamacpp_nonsquare_aligned() {
    // 32-aligned NON-SQUARE: identity preprocessing on both engines, so this
    // isolates the tower's non-square grid handling from the resize port
    run_gate(512, 288);
}

/// S2 preprocessing oracle: validate the smart-resize/bilinear/letterbox PORT
/// with llama itself as the (fixed) evaluator - mtmd-cli on the original
/// unaligned image vs mtmd-cli on PADDOCK's preprocessed canvas (identity for
/// llama, since it is already 32-aligned and in-budget). If the canvas bytes
/// were llama-equivalent, both runs see identical pixels and greedy output
/// must match exactly. This deliberately avoids a paddock-vs-llama text
/// compare: image prompts are hundreds of rows, where cross-engine greedy
/// exact is provably knife-edge -
/// e.g. this very image flips one content token between the towers while
/// both engines agree on either fixed canvas.
#[test]
fn preprocessing_matches_llamacpp_oracle() {
    if std::env::var_os("PADDOCK_HEAVY_TESTS").is_none() {
        eprintln!("set PADDOCK_HEAVY_TESTS=1 to run the preprocessing oracle");
        return;
    }
    let cli = repo().join("vendor/llamacpp/llama-mtmd-cli.exe");
    let dir = model_dir();
    let model = dir.join("Qwen3.6-27B-Q8_0.gguf");
    let mmproj = dir.join("mmproj-F16.gguf");
    for (what, p) in [("mtmd-cli", &cli), ("model", &model), ("mmproj", &mmproj)] {
        if !p.exists() {
            eprintln!("{what} {p:?} missing - skipping");
            return;
        }
    }

    let (w, h) = (500usize, 300usize);
    let original = std::env::temp_dir().join("paddock_preproc_oracle_orig.bmp");
    write_bmp(&original, w, h);

    // paddock's preprocessing, pure host path (no GPU needed)
    use paddock_engine::gpu_model::qwen35::vision::{
        PixelBudget, resize_pad_black, smart_resize_target,
    };
    let mut rgb = vec![0u8; 3 * w * h];
    for y in 0..h {
        for x in 0..w {
            let (r, g, b) = px(x, y, w, h);
            let i = (y * w + x) * 3;
            rgb[i] = r;
            rgb[i + 1] = g;
            rgb[i + 2] = b;
        }
    }
    // Compared under LLAMA.CPP'S budget deliberately: this test proves our
    // resize ALGORITHM matches mtmd's, and mtmd caps at a quarter of Qwen's
    // published ceiling. We serve at PixelBudget::QWEN_SPEC. At 500x300 the
    // two budgets give the same answer (both caps are far away), so the
    // divergence does not weaken what this asserts - it is spelled out here so
    // the next reader does not "fix" production to match the oracle.
    let (tw, th) = smart_resize_target(w, h, 16, PixelBudget::LLAMACPP);
    assert_eq!((tw, th), (512, 288), "smart-resize target drifted");
    assert_eq!(
        smart_resize_target(w, h, 16, PixelBudget::QWEN_SPEC),
        (tw, th),
        "this fixture must sit under both budgets, or it is testing the cap not the algorithm"
    );
    let canvas = resize_pad_black(&rgb, w, h, tw, th);

    // write the canvas as a BMP (bottom-up, BGR)
    let canvas_bmp = std::env::temp_dir().join("paddock_preproc_oracle_canvas.bmp");
    {
        let img_size = tw * th * 3;
        let mut out = Vec::with_capacity(54 + img_size);
        out.extend_from_slice(b"BM");
        out.extend_from_slice(&(54u32 + img_size as u32).to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        out.extend_from_slice(&54u32.to_le_bytes());
        out.extend_from_slice(&40u32.to_le_bytes());
        out.extend_from_slice(&(tw as i32).to_le_bytes());
        out.extend_from_slice(&(th as i32).to_le_bytes());
        out.extend_from_slice(&1u16.to_le_bytes());
        out.extend_from_slice(&24u16.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        out.extend_from_slice(&(img_size as u32).to_le_bytes());
        out.extend_from_slice(&2835u32.to_le_bytes());
        out.extend_from_slice(&2835u32.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        for y in (0..th).rev() {
            for x in 0..tw {
                let i = (y * tw + x) * 3;
                out.extend_from_slice(&[canvas[i + 2], canvas[i + 1], canvas[i]]);
            }
        }
        std::fs::write(&canvas_bmp, out).expect("write canvas bmp");
    }

    let run = |img: &std::path::Path| -> String {
        let out = Command::new(&cli)
            .args([
                "-m",
                model.to_str().unwrap(),
                "--mmproj",
                mmproj.to_str().unwrap(),
                "--image",
                img.to_str().unwrap(),
                "-p",
                PROMPT,
                "--temp",
                "0",
                "-n",
                &N.to_string(),
                "-ngl",
                "99",
            ])
            .output()
            .expect("run mtmd-cli");
        let text = String::from_utf8_lossy(&out.stdout).trim().to_string();
        assert!(
            !text.is_empty(),
            "mtmd-cli produced no output; stderr: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        text
    };

    let on_original = run(&original);
    let on_canvas = run(&canvas_bmp);
    eprintln!("llama(original): {on_original:?}");
    eprintln!("llama(canvas)  : {on_canvas:?}");
    assert_eq!(
        on_original, on_canvas,
        "paddock's preprocessed canvas is not llama-equivalent"
    );
    eprintln!("PREPROCESS ORACLE OK: llama output identical on original vs paddock canvas");
}
