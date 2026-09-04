//! Gemma 4 multimodal greedy parity: replicate llama-mtmd-cli's exact chunk
//! stream (system+think turn, user turn opening, IMAGE, prompt text, model
//! turn) and greedy-decode - the continuation diffs against the oracle's.
//!
//! Usage: gemma4_mm_greedy <model.gguf> <mmproj.gguf> <image> [prompt] [n]
// A development probe: it runs on a box its author is looking at, and a
// failure should stop it where it happened rather than be reported.
#![allow(clippy::unwrap_used)]

use std::sync::Arc;

use paddock_engine::generator::Generator;
use paddock_engine::gpu::GpuExecutor;
use paddock_engine::gpu_model::gemma4::GpuGemma4;
use paddock_engine::service::MmChunk;
use paddock_models::mapped::MappedGguf;
use paddock_tokenizer::GgufTokenizer;

fn argmax(l: &[f32]) -> u32 {
    let mut b = 0usize;
    for i in 1..l.len() {
        if l[i] > l[b] {
            b = i;
        }
    }
    b as u32
}

fn main() {
    let mut args = std::env::args().skip(1);
    let model = args.next().expect("model");
    let mmproj = args.next().expect("mmproj");
    let image = args.next().expect("image");
    let prompt = args
        .next()
        .unwrap_or_else(|| "Describe what this image shows in one sentence.".to_owned());
    let n_gen: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or(48);

    let img = image::open(&image).expect("decode image").to_rgb8();
    let (w, h) = (img.width() as usize, img.height() as usize);

    let map = MappedGguf::open(model.as_ref()).expect("open model");
    let tok = GgufTokenizer::from_gguf(map.gguf()).expect("tokenizer");
    let bos = tok.bos_id.expect("bos");

    // llama-mtmd-cli --jinja layout: enable_thinking system turn, user turn,
    // "<__media__>" prepended bare to the prompt text (PR #17616), then the
    // turn close + generation prompt. mtmd tokenizes each text piece
    // separately - mirrored here so boundaries match.
    let pre = "<|turn>system\n<|think|>\n<turn|>\n<|turn>user\n";
    let post = format!("{prompt}<turn|>\n<|turn>model\n");
    let mut pre_ids = vec![bos];
    pre_ids.extend(tok.encode(pre).expect("encode pre"));
    let post_ids = tok.encode(&post).expect("encode post");

    let chunks = vec![
        MmChunk::Text(pre_ids),
        MmChunk::Image {
            rgb: img.as_raw().clone(),
            w,
            h,
        },
        MmChunk::Text(post_ids),
    ];

    let exec = Arc::new(
        GpuExecutor::new(0, "packs/cuda/build/pd-cuda-sm120.so".as_ref()).expect("executor"),
    );
    let mut m = GpuGemma4::load(exec, &map, 4096).expect("load");
    let mmap = MappedGguf::open(mmproj.as_ref()).expect("open mmproj");
    m.attach_vision(&mmap).expect("attach vision");

    let t0 = std::time::Instant::now();
    // the mm prefill reports its ROW count beside the logits - image rows
    // included, which is what the serving layer bills
    let (mut logits, rows) = m
        .forward_multimodal(&chunks)
        .expect("mm prefill")
        .expect("vision attached");
    eprintln!(
        "mm prefill ({rows} rows) in {:.2}s",
        t0.elapsed().as_secs_f32()
    );

    let mut out = Vec::new();
    for _ in 0..n_gen {
        let next = argmax(&logits);
        out.push(next);
        logits = m.forward(next).expect("decode");
    }
    println!("==={:?}", tok.decode(&out, false).expect("decode text"));
    if let Ok(path) = std::env::var("OUT_FILE") {
        std::fs::write(path, tok.decode(&out, false).unwrap()).unwrap();
    }
}
