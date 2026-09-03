//! Muse Glimmer vision-tower stage oracle.
//!
//! Feeds the same synthetic images `llama-mtmd-debug -p encode` feeds, through
//! the same tower weights, and prints the same per-stage sums - so every one of
//! this tower's silent architecture constants (rope pair layout, window
//! geometry, erf-vs-tanh GELU, the channel-outer merge, the 3:1 sparse pattern)
//! is separately falsifiable instead of hiding behind an end-to-end token diff.
//!
//! The reference tool feeds RAW f32 pixel values - no resize, no mean/std - so
//! this bypasses `preprocess_rgb` and builds the patch rows directly, which is
//! what makes the two comparable. Preprocessing parity is a separate question
//! and its own check.
//!
//! Reference side:
//!   llama-mtmd-debug -m <model.gguf> --mmproj <mmproj.gguf> \
//!       -p encode --image gray -n 896 -ngl 0 --no-warmup
//!
//! This side:
//!   MUSE_MMPROJ=.../mmproj-Muse-Glimmer-30B-BF16.gguf \
//!   PADDOCK_PACK=packs/cuda/build/pd-cuda-sm120.so \
//!   PADDOCK_MUSE_VIS_DUMP=1 \
//!   cargo run --release --example muse_vision_probe -- gray 896

use std::sync::Arc;

use paddock_engine::gpu::GpuExecutor;
use paddock_engine::gpu_model::gemma4::muse_vision::VisionModel;
use paddock_models::mapped::MappedGguf;

/// The reference tool's image generators, value for value (mtmd-debug.cpp).
fn synth(kind: &str, size: usize) -> Vec<f32> {
    let mut img = vec![0f32; size * size * 3];
    let put = |img: &mut Vec<f32>, x: usize, y: usize, r: f32, g: f32, b: f32| {
        let i = (y * size + x) * 3;
        img[i] = r;
        img[i + 1] = g;
        img[i + 2] = b;
    };
    match kind {
        "black" => {}
        "white" => img.fill(1.0),
        "gray" => img.fill(0.5),
        "red" => {
            for i in 0..size * size {
                img[i * 3] = 1.0
            }
        }
        "green" => {
            for i in 0..size * size {
                img[i * 3 + 1] = 1.0
            }
        }
        "blue" => {
            for i in 0..size * size {
                img[i * 3 + 2] = 1.0
            }
        }
        "cb" => {
            for y in 0..size {
                for x in 0..size {
                    let v = if (x + y) % 2 == 1 { 0.0 } else { 1.0 };
                    put(&mut img, x, y, v, v, v);
                }
            }
        }
        "rainbow" => {
            let (cx, cy) = (size as f32 / 2.0, size as f32 / 2.0);
            let max_dist = (cx * cx + cy * cy).sqrt();
            for y in 0..size {
                for x in 0..size {
                    let (dx, dy) = (x as f32 - cx, y as f32 - cy);
                    let mut hue = dy.atan2(dx) / std::f32::consts::TAU;
                    if hue < 0.0 {
                        hue += 1.0;
                    }
                    let sat = ((dx * dx + dy * dy).sqrt() / max_dist).min(1.0);
                    let h6 = hue * 6.0;
                    let i6 = h6 as i32;
                    let f = h6 - i6 as f32;
                    let (p, q, t) = (1.0 - sat, 1.0 - sat * f, 1.0 - sat * (1.0 - f));
                    let (r, g, b) = match i6 % 6 {
                        0 => (1.0, t, p),
                        1 => (q, 1.0, p),
                        2 => (p, 1.0, t),
                        3 => (p, q, 1.0),
                        4 => (t, p, 1.0),
                        _ => (1.0, p, q),
                    };
                    put(&mut img, x, y, r, g, b);
                }
            }
        }
        other => panic!("unknown image kind {other:?}"),
    }
    img
}

fn main() {
    let mut args = std::env::args().skip(1);
    let kind = args.next().unwrap_or_else(|| "gray".into());
    let size: usize = args.next().map(|s| s.parse().unwrap()).unwrap_or(896);
    let mmproj = std::env::var("MUSE_MMPROJ").expect("set MUSE_MMPROJ");
    let pack = std::env::var("PADDOCK_PACK").expect("set PADDOCK_PACK");

    let map = MappedGguf::open(mmproj.as_ref()).expect("open mmproj gguf");
    let exec = Arc::new(GpuExecutor::new(0, pack.as_ref()).expect("cuda executor"));
    let v = VisionModel::load(Arc::clone(&exec), &map).expect("load muse tower");

    // The reference tool hands the tower a square image directly, so the grid
    // is size/patch and the tower's own aspect fit never runs. Assert rather
    // than silently comparing two different geometries.
    let patch = v.patch_size();
    assert_eq!(
        size % (patch * v.merge_size()),
        0,
        "pick -n a multiple of {}",
        patch * v.merge_size()
    );
    let (gw, gh) = (size / patch, size / patch);

    let img = synth(&kind, size);
    let patches = v.patch_rows_raw(&img, size, size);
    println!(
        "muse-vision probe: {kind} {size}x{size} -> grid {gw}x{gh} = {} patches, {} out tokens",
        gw * gh,
        (gw / v.merge_size()) * (gh / v.merge_size())
    );

    let t0 = std::time::Instant::now();
    let out = v.encode(&patches, gw, gh).expect("encode");
    let ms = t0.elapsed().as_secs_f64() * 1e3;
    let host = exec
        .to_host_len(&out.embd, out.n_tokens * v.llm_embd())
        .expect("readback");
    let sum: f64 = host.iter().map(|&x| x as f64).sum();
    let d = v.llm_embd();
    println!(
        "projected: {} x {d} sum={sum:.4}  ({ms:.0} ms)",
        out.n_tokens
    );
    // The same 3x3 corners ggml's debug callback prints for the `projected`
    // node - a sum can cancel a structural error, six named elements cannot.
    println!("projected corners (rows x cols, matching the mtmd-debug dump):");
    let rows = [
        0,
        1,
        2,
        out.n_tokens - 3,
        out.n_tokens - 2,
        out.n_tokens - 1,
    ];
    for r in rows {
        let row = &host[r * d..(r + 1) * d];
        println!(
            "  [{r:4}] {:9.4} {:9.4} {:9.4}  ...  {:9.4} {:9.4} {:9.4}",
            row[0],
            row[1],
            row[2],
            row[d - 3],
            row[d - 2],
            row[d - 1]
        );
    }
}
