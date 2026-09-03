//! Vision tower smoke test: load the qwen3vl mmproj, encode a synthetic
//! 768×768 image (48×48 patch grid - identity pos-embd path), and check the
//! output geometry + numeric sanity. Token-level parity vs llama-mtmd-cli is
//! the follow-up gate; this pins shapes, ordering, and finiteness.

mod common;

use paddock_engine::gpu_model::qwen35::vision::VisionModel;
use paddock_models::mapped::MappedGguf;

/// Deterministic synthetic RGB test image: smooth gradients + a few blocks, so
/// different patches get distinct embeddings.
pub fn synth_rgb(w: usize, h: usize) -> Vec<u8> {
    let mut px = vec![0u8; 3 * w * h];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 3;
            px[i] = ((x * 255) / w) as u8;
            px[i + 1] = ((y * 255) / h) as u8;
            px[i + 2] = (((x / 64 + y / 64) % 2) * 200 + 25) as u8;
        }
    }
    px
}

#[test]
fn vision_encode_shapes_and_sanity() {
    let Some(path) = common::model("QWEN36_MMPROJ", common::QWEN36_MMPROJ) else {
        return;
    };
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    let map = MappedGguf::open(&path).expect("open mmproj");
    let vm = VisionModel::load(exec.clone(), &map).expect("load vision tower");

    let (w, h) = (768usize, 768usize);
    let rgb = synth_rgb(w, h);
    let img = vm.normalize_rgb(&rgb, w, h);
    let t0 = std::time::Instant::now();
    let out = vm.encode(&img, w, h).expect("encode");
    let host = exec.to_host(&out.embd).expect("to host");
    // merger width comes from the mmproj (27B: 5120, 35B-A3B: 2048)
    let ed = host.len() / (out.nx * out.ny);
    eprintln!(
        "encoded {}x{} in {:?} -> [{} x {}] grid {}x{}",
        w,
        h,
        t0.elapsed(),
        host.len() / ed,
        ed,
        out.nx,
        out.ny
    );

    assert_eq!(out.nx, 24);
    assert_eq!(out.ny, 24);
    assert_eq!(host.len(), 24 * 24 * ed);
    assert!(host.iter().all(|v| v.is_finite()), "non-finite embeddings");

    // distinct patches must land at distinct embeddings; norms in a sane band
    let row = |i: usize| &host[i * ed..(i + 1) * ed];
    let norm = |r: &[f32]| r.iter().map(|v| v * v).sum::<f32>().sqrt();
    let (n0, nmid) = (norm(row(0)), norm(row(300)));
    eprintln!("row norms: [0]={n0:.2} [300]={nmid:.2}");
    assert!(n0 > 1e-2 && n0 < 1e4, "degenerate embedding norm {n0}");
    let d: f32 = row(0)
        .iter()
        .zip(row(300))
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(
        d > 1.0,
        "distinct patches produced near-identical embeddings"
    );

    // smaller, non-native grid exercises the pos-embd interpolation path
    let (w2, h2) = (256usize, 256usize);
    let rgb2 = synth_rgb(w2, h2);
    let img2 = vm.normalize_rgb(&rgb2, w2, h2);
    let out2 = vm.encode(&img2, w2, h2).expect("encode 256");
    let host2 = exec.to_host(&out2.embd).expect("to host");
    assert_eq!((out2.nx, out2.ny), (8, 8));
    assert!(host2.iter().all(|v| v.is_finite()));
    eprintln!("256x256 OK -> {} tokens", host2.len() / ed);
}
