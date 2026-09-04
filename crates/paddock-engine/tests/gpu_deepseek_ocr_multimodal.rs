//! Multimodal splice gate: a real document page through the
//! engine's preprocess -> DeepEncoder -> splice -> pool prefill -> ring decode,
//! against the reference-driven f32 oracle.
//!
//! Beyond logits parity, the legs pin the two claims that make this lane
//! worth having:
//!  * a SECOND request with the same page resumes off the radix and re-runs
//!    the tower for one view (the global tail the block-aligned resume left),
//!    not all 7;
//!  * the pool footprint still PINS at ⌈(907 + W)/16⌉ blocks after 200
//!    generated tokens with the vision prefix in place.
// Test code: a failed assumption stops the test where it happened.
#![allow(clippy::unwrap_used)]

mod common;

use std::sync::Arc;

use paddock_engine::gpu_model::deepseek_ocr::GpuDeepseekOcr;
use paddock_engine::service::MmChunk;
use paddock_models::mapped::MappedGguf;

fn oracle_dir() -> Option<std::path::PathBuf> {
    common::model_roots()
        .iter()
        .map(|r| r.join("ocr-battery").join("oracle"))
        .find(|p| p.join("mm.json").exists())
}

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

fn read_u32s(p: &std::path::Path) -> Vec<u32> {
    std::fs::read(p)
        .unwrap_or_else(|e| panic!("{}: {e}", p.display()))
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect()
}

fn read_f32s(p: &std::path::Path) -> Vec<f32> {
    std::fs::read(p)
        .unwrap_or_else(|e| panic!("{}: {e}", p.display()))
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

fn json_usize(txt: &str, key: &str) -> usize {
    let s = &txt[txt.find(&format!("\"{key}\":")).unwrap() + key.len() + 3..];
    s[..s.find([',', '}', '\n']).unwrap()]
        .trim()
        .parse()
        .unwrap()
}

fn argmax(v: &[f32]) -> u32 {
    let mut bi = 0usize;
    for (i, &x) in v.iter().enumerate() {
        if x > v[bi] {
            bi = i;
        }
    }
    bi as u32
}

/// max|Δ| over the oracle's top-50 logit positions - the head-anchored gate
/// (a full-vector cosine on 129k logits is dominated by near-zero noise).
fn head_dmax(oracle: &[f32], got: &[f32]) -> f32 {
    let mut widx: Vec<usize> = (0..oracle.len()).collect();
    widx.sort_by(|&a, &b| oracle[b].partial_cmp(&oracle[a]).unwrap());
    widx[..50]
        .iter()
        .map(|&i| (got[i] - oracle[i]).abs())
        .fold(0f32, f32::max)
}

#[test]
fn battery_page_splice_matches_the_oracle() {
    let Some(dir) = oracle_dir() else {
        common::missing("no mm oracle");
        return;
    };
    let (Some(path), Some(mmproj)) = (model_path(), mmproj_path()) else {
        common::missing("no Unlimited-OCR GGUF pair");
        return;
    };
    let Some(exec) = common::gpu_arc() else {
        return;
    };

    let meta = std::fs::read_to_string(dir.join("mm.json")).expect("mm.json");
    let n_rows = json_usize(&meta, "n_rows");
    let w = json_usize(&meta, "ring_window");
    let prefix = read_u32s(&dir.join("mm_prefix_ids.bin"));
    let suffix = read_u32s(&dir.join("mm_suffix_ids.bin"));
    let greedy = read_u32s(&dir.join("mm_greedy_ids.bin"));
    let t5_ids = read_u32s(&dir.join("mm_top5_ids.bin"));
    let oracle_logits = read_f32s(&dir.join("mm_prefill_logits.bin"));
    let rgb = std::fs::read(dir.join("battery_rgb.bin")).expect("battery_rgb.bin");
    assert_eq!(rgb.len(), 1240 * 1754 * 3, "battery page is 1240x1754 RGB8");
    let steps = greedy.len();

    let map = MappedGguf::open(&path).expect("open gguf");
    let mut m = GpuDeepseekOcr::load(Arc::clone(&exec), &map, 32_768).expect("load decoder");
    let mm = MappedGguf::open(&mmproj).expect("open mmproj");
    m.attach_vision(&mm).expect("attach vision");
    assert_eq!(m.enable_batch(2).expect("enable_batch"), 2);

    let chunks = vec![
        MmChunk::Text(prefix.clone()),
        MmChunk::Image {
            rgb: rgb.clone(),
            w: 1240,
            h: 1754,
        },
        MmChunk::Text(suffix.clone()),
    ];

    // ── leg 1: cold prefill. Row count is the arbiter-measured 907, the
    // tower runs exactly 7 views (6 crops + global), argmax matches the f32
    // oracle, and the distribution head stays close (tower f16 GEMMs + Q8_0
    // decoder vs f32 - same class the encoder/decoder gates arbitrate).
    let (logits, rows) = m.multimodal_prefill_slot(0, &chunks).expect("mm prefill");
    assert_eq!(
        rows, n_rows,
        "row count vs the oracle (arbiter measured 907)"
    );
    assert_eq!(
        m.mm_tower_views, 7,
        "battery page = 6 crops + 1 global view"
    );
    let dmax = head_dmax(&oracle_logits, &logits);
    eprintln!("cold mm prefill: head50 max|Δ| {dmax:.4}");
    assert_eq!(argmax(&logits), greedy[0], "prefill argmax vs oracle");
    assert!(
        dmax < 1.5,
        "prefill head drift {dmax} (measured 0.35 at landing)"
    );

    // ── leg 2: teacher-forced decode through warmup AND ring steady state
    // on a real document distribution (the text-only gate used a garbage
    // prompt; this is the one that speaks for served output).
    let (mut top1, mut top1_ring) = (0usize, 0usize);
    for s in 0..steps {
        m.batch_step_slots(&[greedy[s]], &[(n_rows + s) as u32], &[0])
            .expect("decode step");
        let l = m.read_batch_logits(1).expect("logits");
        let ok = argmax(&l) == t5_ids[s * 5];
        top1 += ok as usize;
        if s >= w {
            top1_ring += ok as usize;
        }
    }
    let ring_steps = steps - w;
    eprintln!("mm decode: top1 {top1}/{steps} (ring {top1_ring}/{ring_steps})");
    assert!(top1 as f64 / steps as f64 >= 0.95, "mm top-1 agreement");
    assert!(
        top1_ring as f64 / ring_steps as f64 >= 0.95,
        "mm ring agreement"
    );

    // ── leg 3: the footprint PINS with a vision prefix in place.
    assert_eq!(
        m.pool_slot_blocks(0).expect("slot blocks"),
        (n_rows + w).div_ceil(16),
        "ring did not pin the pool footprint under the spliced prefix"
    );

    // ── leg 4: same page into slot 1 - the same-document-many-questions
    // case. The radix resume adopts every full block (896 of 907 rows), so
    // the recomputed tail touches only the global view's last rows: the
    // tower re-runs one view, not 7, and crops are never re-encoded.
    let before = m.mm_tower_views;
    let (logits1, rows1) = m
        .multimodal_prefill_slot(1, &chunks)
        .expect("resumed mm prefill");
    assert_eq!(rows1, n_rows);
    assert_eq!(
        m.mm_tower_views - before,
        1,
        "resume must re-encode only the global view, got {} views",
        m.mm_tower_views - before
    );
    assert_eq!(argmax(&logits1), greedy[0], "resumed prefill argmax");
    let d1 = head_dmax(&oracle_logits, &logits1);
    eprintln!("resumed mm prefill: head50 max|Δ| {d1:.4}");
    // wider than the cold bar: the recomputed tail rides the sanctioned
    // rmsnorm width seam (see batch.rs::prefill_resume_rows) on top of the
    // tower class - measured 0.73 at landing
    assert!(d1 < 1.5, "resumed head drift {d1}");

    // ── leg 5: the crop-override directive routes end to end. Forced base on
    // the same page is the multi-page layout of one: 273 image rows + the
    // text, one tower view (the padded global), no crops. And the geometry
    // key fold means none of the gundam encode's cached rows may be adopted -
    // a cross-mode radix hit was the corruption this fold exists to prevent
    // (image keys differ; only the shared text prefix may resume, and it is
    // shorter than MIN_CACHE_PREFIX here).
    let mut base_chunks = chunks.clone();
    base_chunks.insert(
        0,
        MmChunk::OcrCrop(paddock_engine::service::OcrCropMode::Base),
    );
    let before = m.mm_tower_views;
    let (_, rows_b) = m
        .multimodal_prefill_slot(1, &base_chunks)
        .expect("forced-base prefill");
    assert_eq!(
        rows_b,
        prefix.len() + 273 + suffix.len(),
        "forced base = one 273-row page view plus the text"
    );
    assert_eq!(m.mm_tower_views - before, 1, "forced base encodes ONE view");
}

/// The chunked-mm lane: the same battery page admitted through
/// `prefill_begin_multimodal` + the encoder budget + mixed ticks must match
/// the oracle the classic wave matches - and a live decode stream must keep
/// its agreement while a page admits beside it (the lane's whole purpose).
#[test]
fn chunked_mm_lane_matches_the_oracle_and_survives_interleave() {
    use paddock_engine::generator::{Generator, MmAdmit};

    let Some(dir) = oracle_dir() else {
        common::missing("no mm oracle");
        return;
    };
    let (Some(path), Some(mmproj)) = (model_path(), mmproj_path()) else {
        common::missing("no Unlimited-OCR GGUF pair");
        return;
    };
    let Some(exec) = common::gpu_arc() else {
        return;
    };

    let meta = std::fs::read_to_string(dir.join("mm.json")).expect("mm.json");
    let n_rows = json_usize(&meta, "n_rows");
    let w = json_usize(&meta, "ring_window");
    let prefix = read_u32s(&dir.join("mm_prefix_ids.bin"));
    let suffix = read_u32s(&dir.join("mm_suffix_ids.bin"));
    let greedy = read_u32s(&dir.join("mm_greedy_ids.bin"));
    let t5_ids = read_u32s(&dir.join("mm_top5_ids.bin"));
    let oracle_logits = read_f32s(&dir.join("mm_prefill_logits.bin"));
    let rgb = std::fs::read(dir.join("battery_rgb.bin")).expect("battery_rgb.bin");
    let steps = greedy.len();

    let map = MappedGguf::open(&path).expect("open gguf");
    let mut m = GpuDeepseekOcr::load(Arc::clone(&exec), &map, 32_768).expect("load decoder");
    let mm = MappedGguf::open(&mmproj).expect("open mmproj");
    m.attach_vision(&mm).expect("attach vision");
    assert_eq!(m.enable_batch(2).expect("enable_batch"), 2);
    assert!(m.supports_chunked_prefill() && m.supports_chunked_multimodal());

    let chunks = vec![
        MmChunk::Text(prefix.clone()),
        MmChunk::Image {
            rgb: rgb.clone(),
            w: 1240,
            h: 1754,
        },
        MmChunk::Text(suffix.clone()),
    ];

    // Drive one slot's admission through the encoder budget to completion:
    // encode_step until it reports, then mixed ticks (with `dec` decode rows
    // riding) until the finisher lands. Prep runs on worker threads, so the
    // encode loop polls - a real scheduler tick does other work between calls.
    let admit_and_finish = |m: &mut GpuDeepseekOcr,
                            slot: usize,
                            req: &[MmChunk],
                            budget: usize,
                            mut dec: Option<(&mut usize, &mut usize)>|
     -> (Vec<f32>, usize, usize, usize) {
        let verdicts = m.prefill_begin_multimodal(vec![(slot, req.to_vec())]);
        assert_eq!(verdicts.len(), 1);
        let mut queued = matches!(verdicts[0], (_, MmAdmit::Queued));
        assert!(
            queued || matches!(verdicts[0], (_, MmAdmit::Encoding)),
            "admission failed outright"
        );
        let mut spins = 0usize;
        while m.encoding_pending() {
            for (k, res) in m.encode_step() {
                assert_eq!(k, slot);
                assert!(matches!(res, MmAdmit::Queued), "encode failed");
                queued = true;
            }
            spins += 1;
            assert!(spins < 20_000, "encoder budget never finished");
            std::thread::sleep(std::time::Duration::from_micros(200));
        }
        assert!(queued, "entry left the encode queue without reporting");
        let mut ticks = 0usize;
        loop {
            // the decode stream keeps stepping through the same mixed ticks
            // that advance the admission - the interleave under test
            let band: Vec<(usize, u32, u32)> = match &dec {
                Some((s, _)) => vec![(0usize, greedy[**s], (n_rows + **s) as u32)],
                None => Vec::new(),
            };
            let (dl, fin) = m.forward_mixed(&band, budget).expect("mixed tick");
            if let Some((s, top1)) = dec.as_mut() {
                **top1 += (argmax(&dl) == t5_ids[**s * 5]) as usize;
                **s += 1;
            }
            ticks += 1;
            assert!(ticks < 200, "chunked prefill never finished");
            if let Some((k, logits, rows)) = fin.into_iter().next() {
                assert_eq!(k, slot);
                return (logits, rows, ticks, spins);
            }
        }
    };

    // ── leg 1: COLD chunked admission, no decode load. Same bars as the
    // classic wave: 907 rows, 7 tower views, oracle argmax, bounded head
    // drift - and the rows really advanced across several ticks.
    let (logits, rows, ticks, _) = admit_and_finish(&mut m, 0, &chunks, 256, None);
    assert_eq!(rows, n_rows, "row count vs the oracle");
    assert_eq!(
        m.mm_tower_views, 7,
        "cold chunked admission = 6 crops + 1 global"
    );
    assert_eq!(
        argmax(&logits),
        greedy[0],
        "chunked prefill argmax vs oracle"
    );
    let dmax = head_dmax(&oracle_logits, &logits);
    eprintln!("chunked mm prefill: head50 max|Δ| {dmax:.4} over {ticks} ticks");
    assert!(dmax < 1.5, "chunked prefill head drift {dmax}");
    assert!(
        ticks >= 3,
        "907 rows at a 256-row budget must span ticks, got {ticks}"
    );

    // ── leg 2a: same page into slot 1, decode riding. The admission resumes
    // off slot 0's insert (one tower view, not 7) - the chunked finisher's
    // prefix_insert must be adoptable, or the radix story broke. The decode
    // stream's agreement bar is the classic gate's own 0.95 - fused-tick
    // decode rows ride the r-invariant prefill GEMM class, the sanctioned
    // near-tie seam.
    let before = m.mm_tower_views;
    let (mut s, mut top1) = (0usize, 0usize);
    let (logits1, rows1, _, _) =
        admit_and_finish(&mut m, 1, &chunks, 64, Some((&mut s, &mut top1)));
    assert_eq!(rows1, n_rows);
    assert_eq!(
        m.mm_tower_views - before,
        1,
        "resumed chunked admission re-encodes ONE view"
    );
    assert_eq!(argmax(&logits1), greedy[0], "interleaved admission argmax");
    assert!(s >= 1, "no decode rows rode the admission ticks");

    // ── leg 2b: a COLD page (one pixel flipped - new content hash, same
    // geometry) admits into slot 1 while slot 0 keeps decoding: the full
    // encoder-budget ladder and every prefill chunk run under a live stream.
    // This is the lane's reason to exist; the stream's agreement is checked
    // over the whole run below.
    let mut rgb_b = rgb.clone();
    rgb_b[0] ^= 1;
    let chunks_b = vec![
        MmChunk::Text(prefix.clone()),
        MmChunk::Image {
            rgb: rgb_b,
            w: 1240,
            h: 1754,
        },
        MmChunk::Text(suffix.clone()),
    ];
    let before = m.mm_tower_views;
    let rode_before = s;
    let (_, rows_b, ticks_b, _) =
        admit_and_finish(&mut m, 1, &chunks_b, 64, Some((&mut s, &mut top1)));
    assert_eq!(
        rows_b, n_rows,
        "one flipped pixel must not move the geometry"
    );
    assert_eq!(
        m.mm_tower_views - before,
        7,
        "a cold page encodes all 7 views"
    );
    let rode = s - rode_before;
    assert!(
        rode >= 5,
        "decode rows must ride a cold admission's ticks, got {rode}"
    );
    eprintln!("interleaved: {top1}/{s} decode top-1 during admissions ({ticks_b} cold ticks)");

    // ...and the stream keeps decoding to steady state after the admission,
    // through the queue-empty mixed path (the captured decode graph).
    while s < steps {
        let (dl, fin) = m
            .forward_mixed(&[(0, greedy[s], (n_rows + s) as u32)], 64)
            .expect("decode tick");
        assert!(fin.is_empty());
        top1 += (argmax(&dl) == t5_ids[s * 5]) as usize;
        s += 1;
    }
    let agree = top1 as f64 / steps as f64;
    eprintln!("interleaved decode: top1 {top1}/{steps}");
    assert!(
        agree >= 0.95,
        "decode agreement through an interleaved admission: {agree}"
    );

    // ── leg 3: the ring still pins the footprint under the chunked lane.
    assert_eq!(
        m.pool_slot_blocks(0).expect("slot blocks"),
        (n_rows + w).div_ceil(16),
        "ring did not pin the pool footprint for a chunked-lane sequence"
    );

    // ── leg 4: the text queue + abort. A text prompt rides the same queue;
    // an aborted one leaves no trace and never inserts.
    let text: Vec<u32> = prefix.iter().chain(suffix.iter()).copied().collect();
    m.prefill_begin(1, text.clone())
        .expect("text prefill_begin");
    assert!(m.prefill_abort(1), "abort must drop the queued entry");
    m.prefill_begin(1, text.clone())
        .expect("re-begin after abort");
    let mut t_logits = None;
    for _ in 0..50 {
        let (_, fin) = m.forward_mixed(&[], 48).expect("text mixed tick");
        if let Some((k, l, r)) = fin.into_iter().next() {
            assert_eq!(k, 1);
            assert_eq!(r, text.len());
            t_logits = Some(l);
            break;
        }
    }
    let t_logits = t_logits.expect("text chunked prefill finished");
    // classic whole-prompt prefill of the same text into slot 1 (resumes off
    // the chunked insert - the two lanes must agree where it matters)
    let c_logits = paddock_engine::generator::Generator::forward_prefill(&mut m, 1, &text)
        .expect("classic prefill");
    assert_eq!(
        argmax(&t_logits),
        argmax(&c_logits),
        "text lanes disagree on the next token"
    );
}
