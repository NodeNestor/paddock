//! Granite-Vision loader-milestone probe: parse the mmproj's geometry, upload
//! every tower and Q-Former weight, print the audit. First rung of the
//! bring-up ladder - the tower/projector forward comes next, then parity
//! against llama.cpp's mmproj on the same image.
//!
//! The point of this rung is cheap: a loader that silently mis-resolves the
//! tap layers or the downsampler selectors would still "work" here and produce
//! confidently wrong features later, so print the resolved mapping and eyeball
//! it against the checkpoint's own geometry.
//!
//! Usage: GRANITE_MMPROJ=<path to mmproj-model-f16.gguf>
//!        PADDOCK_PACK=packs\cuda\build\pd-cuda-sm86.dll
//!        cargo run --release --example granite_vision_load

use std::sync::Arc;

use paddock_engine::gpu::GpuExecutor;
use paddock_engine::gpu_model::granite::preprocess::PackRow;
use paddock_engine::gpu_model::granite::vision::{QFORMER_EPS, QFORMER_HEAD_DIM, VisionModel};
use paddock_models::mapped::MappedGguf;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();
    let mmproj = std::env::var("GRANITE_MMPROJ").expect("set GRANITE_MMPROJ");
    let pack = std::env::var("PADDOCK_PACK").expect("set PADDOCK_PACK");

    let map = MappedGguf::open(mmproj.as_ref()).expect("open mmproj gguf");
    let exec = Arc::new(GpuExecutor::new(0, pack.as_ref()).expect("cuda executor"));

    let t0 = std::time::Instant::now();
    let v = VisionModel::load(Arc::clone(&exec), &map).expect("load granite-vision tower");
    let load_ms = t0.elapsed().as_millis();

    let hp = &v.hp;
    println!("\n=== granite-vision mmproj loaded in {load_ms} ms ===");
    println!(
        "tower      : {} blocks, embd {}, {} heads x {} dims, ffn {}, LN eps {:e}",
        hp.n_layers, hp.embd, hp.n_heads, hp.head_dim, hp.ff, hp.eps
    );
    println!(
        "image      : {}px / patch {} -> {}x{} = {} tokens per tile",
        hp.image_size,
        hp.patch,
        hp.grid,
        hp.grid,
        hp.grid * hp.grid
    );
    println!(
        "qformer    : window {} -> query {}, d_head {} => {} heads, eps {:e}",
        hp.window_side,
        hp.query_side,
        QFORMER_HEAD_DIM,
        hp.embd / QFORMER_HEAD_DIM,
        QFORMER_EPS
    );
    println!(
        "projected  : {} tokens per tile, out width {}",
        v.tokens_per_tile(),
        hp.proj_dim
    );
    println!(
        "anyres     : {} grid pinpoints {:?}",
        hp.grid_pinpoints.len(),
        hp.grid_pinpoints
    );
    println!(
        "normalize  : mean {:?} std {:?}",
        hp.image_mean, hp.image_std
    );

    // The tap-to-downsampler mapping, printed because a shift here is the
    // laguna-DFlash failure repeating, and it is silent at runtime.
    println!("\n{} projectors:", v.projs.len());
    println!("  {:>3}  {:>12}  {:>16}", "idx", "tower tap", "downsampler");
    for (i, p) in v.projs.iter().enumerate() {
        let ds = match p.spatial_offset {
            -1 => "area interp".to_string(),
            o @ 0..=3 => {
                format!(
                    "2x2 offset {} ({})",
                    o,
                    ["TL", "TR", "BL", "BR"][o as usize]
                )
            }
            o => format!("UNKNOWN {o}"),
        };
        println!(
            "  {i:>3}  {:>12}  {ds:>16}",
            format!("block {}", p.feature_layer)
        );
    }

    // Cheap shape audit: every projector must carry the same geometry, since
    // they all consume the same grid and emit the same token count.
    let q_len = hp.query_side * hp.query_side;
    let enc_len = hp.window_side * hp.window_side;
    for (i, p) in v.projs.iter().enumerate() {
        assert_eq!(
            p.query.element_count(),
            q_len * hp.embd,
            "proj {i}: query should be [{q_len}, {}]",
            hp.embd
        );
        assert_eq!(
            p.img_pos.element_count(),
            enc_len * hp.embd,
            "proj {i}: img_pos should be [{enc_len}, {}]",
            hp.embd
        );
        assert_eq!(
            p.linear_w.element_count(),
            hp.embd * hp.proj_dim,
            "proj {i}: linear should be [{}, {}]",
            hp.embd,
            hp.proj_dim
        );
    }
    println!(
        "\nshape audit OK: all {} projectors agree on query/img_pos/linear",
        v.projs.len()
    );
    println!(
        "image_newline: {} elements (LLM width)",
        v.image_newline.element_count()
    );

    // ---- AnyRes plans for a spread of real aspect ratios ----
    //
    // The token count is what a prompt builder must reserve as <image>
    // placeholders, and it is decided here with no pixels touched. Note how far
    // it moves with aspect: a 640x480 photo is 594 rows, a 3840x384 banner is
    // 1596, and a 5000x40 sliver keeps only the base tile's 144 because the
    // unpad math discards every grid row.
    println!("\nAnyRes plans (rows a prompt must reserve):");
    println!(
        "  {:>11}  {:>11}  {:>6}  {:>6}  {:>9}  {:>7}",
        "image", "pinpoint", "grid", "tiles", "cells", "tokens"
    );
    for (w, h) in [
        (384, 384),
        (640, 480),
        (1024, 300),
        (800, 400),
        (1000, 1000),
        (3840, 384),
        (5000, 40),
    ] {
        let p = v.plan(w, h).expect("plan");
        let (x0, y0, x1, y1) = p.win;
        println!(
            "  {:>11}  {:>11}  {:>6}  {:>6}  {:>9}  {:>7}",
            format!("{w}x{h}"),
            format!("{}x{}", p.best.0, p.best.1),
            format!("{}x{}", p.grid.0, p.grid.1),
            p.n_tiles(),
            format!("{}x{}", x1 - x0, y1 - y0),
            p.n_tokens()
        );
    }

    // ---- encode a synthetic tile and report it the way llama.cpp does ----
    //
    // The image is deliberately HIGH-FREQUENCY and asymmetric in x/y. A flat or
    // symmetric image would make every windowing permutation produce the same
    // answer, so a transposed win/qwin/unwin table would sail through - the one
    // bug class this rung exists to catch.
    let side = hp.image_size;
    let mut rgb = vec![0u8; 3 * side * side];
    for y in 0..side {
        for x in 0..side {
            let p = (y * side + x) * 3;
            rgb[p] = ((x * 7 + y * 3) % 256) as u8;
            rgb[p + 1] = ((x ^ y) % 256) as u8;
            rgb[p + 2] = ((x * x + y * y) % 256) as u8;
        }
    }
    if let Ok(path) = std::env::var("GRANITE_PROBE_PNG") {
        image::save_buffer(
            &path,
            &rgb,
            side as u32,
            side as u32,
            image::ColorType::Rgb8,
        )
        .expect("write probe png");
        println!("\nprobe image written: {path} ({side}x{side})");
    }

    let tile = v.normalize_rgb(&rgb, side, side);
    let t1 = std::time::Instant::now();
    let feats = v.encode(&[tile]).expect("encode tile");
    println!("\nencode: {} ms", t1.elapsed().as_millis());

    // llama.cpp emits one tensor of shape [8*2560, tokens] - the 8 streams
    // concatenated along the row. Rebuild that row layout so the numbers are
    // directly comparable to its MTMD_DEBUG_EMBEDDINGS dump.
    let host: Vec<Vec<f32>> = feats
        .streams
        .iter()
        .map(|s| exec.stream.clone_dtoh(s).expect("stream to host"))
        .collect();
    let (tokens, w) = (feats.tokens, feats.width);
    let n_embd = w * host.len();
    println!("Shape: [{n_embd}, {tokens}]   (8 projectors x {w})");

    print!("Token 0 (first 16 values): ");
    for i in 0..16 {
        print!("{:.6} ", host[0][i]);
    }
    print!("\nToken 0 (last 16 values):  ");
    for i in w - 16..w {
        print!("{:.6} ", host[host.len() - 1][i]);
    }
    println!();

    // Stats over every value of every stream, matching llama.cpp's reduction.
    let stats = |extra_newline: bool| {
        let (mut sum, mut sum_sq) = (0f64, 0f64);
        let (mut lo, mut hi) = (f32::MAX, f32::MIN);
        let mut n = 0usize;
        for s in &host {
            for &val in s.iter() {
                sum += val as f64;
                sum_sq += (val as f64) * (val as f64);
                lo = lo.min(val);
                hi = hi.max(val);
                n += 1;
            }
        }
        if extra_newline {
            let nl = exec
                .stream
                .clone_dtoh(&v.image_newline.buf)
                .expect("newline to host");
            for _ in 0..host.len() {
                for &val in &nl {
                    sum += val as f64;
                    sum_sq += (val as f64) * (val as f64);
                    lo = lo.min(val);
                    hi = hi.max(val);
                    n += 1;
                }
            }
        }
        let mean = sum / n as f64;
        let var = sum_sq / n as f64 - mean * mean;
        (mean, var.max(0.0).sqrt(), lo, hi, sum)
    };
    for (label, nl) in [("no newline row", false), ("+1 newline row", true)] {
        let (mean, std, lo, hi, sum) = stats(nl);
        println!(
            "Stats ({label}): mean={mean:.6}, std={std:.6}, min={lo:.6}, max={hi:.6}, sum={sum:.6}"
        );
    }
    // Per-stream stats + pairwise distinctness. llama.cpp's dump only shows
    // token 0 of projector 0 and projector 7 plus one global reduction, so
    // projectors 1..6 ride on the aggregate alone. These two checks cover them
    // directly: if two spatial index tables were accidentally identical (all
    // four 2×2 offsets picking TL, say), or two projectors read the same tap
    // through a copy-paste, the streams would coincide and the global stats
    // would barely move.
    println!("\nper-stream (tap / downsampler):");
    for (i, s) in host.iter().enumerate() {
        let p = &v.projs[i];
        let mean = s.iter().map(|&x| x as f64).sum::<f64>() / s.len() as f64;
        let var = s.iter().map(|&x| (x as f64 - mean).powi(2)).sum::<f64>() / s.len() as f64;
        let lo = s.iter().cloned().fold(f32::MAX, f32::min);
        let hi = s.iter().cloned().fold(f32::MIN, f32::max);
        println!(
            "  {i}  blk {:>2}  ds {:>2}   mean {:>10.6}  std {:>9.6}  min {:>10.6}  max {:>9.6}",
            p.feature_layer,
            p.spatial_offset,
            mean,
            var.sqrt(),
            lo,
            hi
        );
    }
    let mut dup = false;
    for a in 0..host.len() {
        for b in a + 1..host.len() {
            let d = host[a]
                .iter()
                .zip(&host[b])
                .map(|(x, y)| ((x - y) as f64).abs())
                .fold(0f64, f64::max);
            if d < 1e-9 {
                println!("  !! stream {a} and {b} are IDENTICAL - index tables or taps collided");
                dup = true;
            }
        }
    }
    if !dup {
        println!("  all {} streams pairwise distinct", host.len());
    }

    // ---- AnyRes end-to-end: tile, encode, pack (multi-tile, non-square) ----
    //
    // The plan arithmetic has unit tests against the HF processor; what those
    // cannot cover is the GPU gather that turns tile-major streams into the
    // packed row layout. So encode a NON-SQUARE image (700x300 -> a 3x1 grid,
    // 4 tiles) both ways and check every packed row against its source: feature
    // rows must be bit-identical to the tile-major encode, and newline rows bit-
    // identical to the learned parameter. A transposed index table or an
    // off-by-one in the newline splice fails this and nothing else catches it.
    {
        let (iw, ih) = (700usize, 300usize);
        let mut img = vec![0u8; 3 * iw * ih];
        for y in 0..ih {
            for x in 0..iw {
                let p = (y * iw + x) * 3;
                img[p] = ((x * 5 + y * 11) % 256) as u8;
                img[p + 1] = ((x * x + y) % 256) as u8;
                img[p + 2] = ((x ^ (y * 3)) % 256) as u8;
            }
        }
        let plan = v.plan(iw, ih).expect("plan");
        println!(
            "\nanyres e2e: {iw}x{ih} -> pinpoint {}x{}, {} tiles, {} rows",
            plan.best.0,
            plan.best.1,
            plan.n_tiles(),
            plan.n_tokens()
        );
        let packed = v.encode_image(&img, iw, ih).expect("encode_image");
        assert_eq!(
            packed.tokens,
            plan.n_tokens(),
            "packed rows != planned rows"
        );

        // the same tiles, encoded without packing, as the reference
        let tiles: Vec<Vec<f32>> = plan
            .tiles(&img, iw, ih)
            .iter()
            .map(|t| v.normalize_rgb(t, hp.image_size, hp.image_size))
            .collect();
        let raw = v.encode(&tiles).expect("encode tiles");
        let w = packed.width;
        let nl = exec
            .stream
            .clone_dtoh(&v.image_newline.buf)
            .expect("newline");
        let (mut n_feat, mut n_nl) = (0usize, 0usize);
        for k in [0usize, 3, 7] {
            let got = exec.stream.clone_dtoh(&packed.streams[k]).expect("packed");
            let want = exec.stream.clone_dtoh(&raw.streams[k]).expect("raw");
            for (r, row) in plan.rows().iter().enumerate() {
                let dst = &got[r * w..(r + 1) * w];
                match row {
                    PackRow::Feature { tile, idx } => {
                        let s = (tile * v.tokens_per_tile() + idx) * w;
                        assert_eq!(
                            dst,
                            &want[s..s + w],
                            "stream {k} row {r} (tile {tile} idx {idx})"
                        );
                        n_feat += 1;
                    }
                    PackRow::Newline => {
                        assert_eq!(dst, &nl[..], "stream {k} row {r} should be image_newline");
                        n_nl += 1;
                    }
                }
            }
        }
        println!("  pack verified on streams 0/3/7: {n_feat} feature rows, {n_nl} newline rows");
    }

    // ---- DeepStack wiring against the real text model (optional leg) ----
    //
    // Checks the two files agree the way the engine assumes: the text model's
    // deepstack_mapping must name exactly the streams the mmproj produces, and
    // the per-layer targets must come out of the FILE rather than the 3*k
    // pattern that happens to hold for this checkpoint.
    if let Ok(text) = std::env::var("GRANITE_TEXT") {
        use paddock_engine::gpu_model::granite::GpuGranite;
        let tmap = MappedGguf::open(text.as_ref()).expect("open text gguf");
        let mut m = GpuGranite::load(Arc::clone(&exec), &tmap, 4096).expect("load text model");
        let targets: Vec<(usize, i32)> = m
            .deepstack_map()
            .iter()
            .enumerate()
            .filter(|(_, k)| **k >= 0)
            .map(|(li, k)| (li, *k))
            .collect();
        println!("\ndeepstack targets (LLM layer <- vision stream), read from the file:");
        for (li, k) in &targets {
            let p = &v.projs[*k as usize];
            println!(
                "  layer {li:>2} <- stream {k}  (tower blk {}, ds {})",
                p.feature_layer, p.spatial_offset
            );
        }
        println!("  stream 0 is not listed: it is the image's input embedding, not an injection");
        m.attach_vision(&map).expect("attach mmproj");
        // Reserve-then-place, the way a prompt builder has to: the placeholder
        // count is decided from the size alone, and the encode must agree.
        let want = m.image_tokens(side, side).expect("token count");
        let t = m.place_image(0, 5, &rgb, side, side).expect("place image");
        println!("placed 1 image ({side}x{side}) at slot 0 pos 5 -> {t} rows (reserved {want})");
        assert_eq!(t, want, "placeholder reservation and encode disagree");
    } else {
        println!("\n(set GRANITE_TEXT=<text gguf> to also check the deepstack mapping)");
    }

    println!(
        "\ncompare against: MTMD_DEBUG_EMBEDDINGS=1 llama-server --mmproj <this file>\n\
         (llama.cpp reduces in f32 and we reduce in f64, so expect its sum to\n\
         drift in the last digits over 2.9M values - the per-value rows are the\n\
         exact check.)"
    );
}
