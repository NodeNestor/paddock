//! Gemma 4 vision tower smoke: load the mmproj, encode a real image (the
//! 1×1-PNG trap is documented), print output shape + value stats. The exact
//! gate is end-to-end token parity vs llama-mtmd-cli once the splice lands.
//!
//! Usage: gemma4_vision_smoke <mmproj.gguf> <image> [pack.so]
// A development probe: it runs on a box its author is looking at, and a
// failure should stop it where it happened rather than be reported.
#![allow(clippy::unwrap_used)]

use std::sync::Arc;

use paddock_engine::gpu::GpuExecutor;
use paddock_engine::gpu_model::gemma4::vision::VisionModel;
use paddock_models::mapped::MappedGguf;

fn main() {
    let mut args = std::env::args().skip(1);
    let mmproj = args
        .next()
        .expect("usage: gemma4_vision_smoke <mmproj> <image> [pack]");
    let image = args.next().expect("image path");
    let pack = args
        .next()
        .unwrap_or_else(|| "packs/cuda/build/pd-cuda-sm120.so".to_owned());

    let img = image::open(&image).expect("decode image").to_rgb8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    eprintln!("image {w}x{h}");

    let exec = Arc::new(GpuExecutor::new(0, pack.as_ref()).expect("executor"));
    let map = MappedGguf::open(mmproj.as_ref()).expect("open mmproj");
    let vm = VisionModel::load(exec.clone(), &map).expect("load tower");

    let t0 = std::time::Instant::now();
    let (patches, gw, gh) = vm.preprocess_rgb(img.as_raw(), w, h);
    let out = vm.encode(&patches, gw, gh).expect("encode");
    let dt = t0.elapsed().as_secs_f32();

    let host = exec
        .to_host_len(&out.embd, out.n_tokens * vm.llm_embd())
        .expect("readback");
    let finite = host.iter().all(|v| v.is_finite());
    let mean = host.iter().sum::<f32>() / host.len() as f32;
    let rms = (host.iter().map(|v| v * v).sum::<f32>() / host.len() as f32).sqrt();
    println!(
        "grid {gw}x{gh} -> {} tokens x {} dims in {dt:.2}s | finite={finite} mean={mean:.4} rms={rms:.4}",
        out.n_tokens,
        vm.llm_embd()
    );
    println!("first8: {:?}", &host[..8]);
}
