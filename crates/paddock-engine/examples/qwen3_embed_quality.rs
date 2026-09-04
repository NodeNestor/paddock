//! Retrieval-quality gate for lossy encoder numeric classes (the sm_120a
//! block-scale FP4 route): embed a synthetic corpus of distinct passages plus
//! paraphrase queries with KNOWN relevant docs, in one ragged batch (so the
//! prefill-batch GEMM path under test actually engages), and report
//! recall@1/recall@10. Run twice - PADDOCK_NO_BS=1 pins the Q8_0 baseline on
//! the same binary - and compare; a dump/ref file pair also reports the
//! cross-path embedding cosine and top-10 overlap:
//!
//!   qwen3_embed_quality /tmp/bs.bin                # dump embeddings
//!   PADDOCK_NO_BS=1 qwen3_embed_quality /tmp/q8.bin /tmp/bs.bin   # + compare
//!
//! The gate: recall@10 unchanged vs the Q8_0 baseline (FP4 weights are lossy;
//! the cos smoke test alone is far too blunt to catch retrieval regressions).
// A development probe: it runs on a box its author is looking at, and a
// failure should stop it where it happened rather than be reported.
#![allow(clippy::unwrap_used)]

use std::io::{Read, Write};
use std::sync::Arc;

use paddock_engine::gpu::GpuExecutor;
use paddock_engine::gpu_model::qwen3::GpuQwen3;
use paddock_models::mapped::MappedGguf;
use paddock_tokenizer::GgufTokenizer;

const SUBJECTS: [&str; 8] = [
    "the migration scheduler",
    "a coral reef ecosystem",
    "the bond portfolio",
    "a convolutional network",
    "the volcanic monitoring array",
    "an ancient trade route",
    "the fermentation process",
    "a distributed cache layer",
];
const VERBS: [&str; 8] = [
    "coordinates",
    "degrades under",
    "hedges against",
    "classifies",
    "detects precursors of",
    "connected",
    "converts sugars during",
    "invalidates entries after",
];
const OBJECTS: [&str; 6] = [
    "seasonal workload spikes across regions",
    "sustained thermal stress and acidification",
    "interest rate shocks in emerging markets",
    "handwritten postal codes at scale",
    "major eruptions weeks in advance",
    "inland cities with coastal ports",
];

/// 384 distinct passages: every (subject, verb, object) combination, with a
/// clause keyed to the index so no two docs are token-identical.
fn corpus() -> Vec<String> {
    let mut docs = Vec::new();
    for (i, s) in SUBJECTS.iter().enumerate() {
        for (j, v) in VERBS.iter().enumerate() {
            for (k, o) in OBJECTS.iter().enumerate() {
                let idx = (i * VERBS.len() + j) * OBJECTS.len() + k;
                docs.push(format!(
                    "Report {idx}: field observations confirm that {s} {v} {o}, \
                     which analysts consider significant for planning cycle {}.",
                    idx % 17
                ));
            }
        }
    }
    docs
}

/// One query per stride-6 doc: same subject+verb+object restated as a
/// question - the source doc is the known-relevant target.
fn queries() -> Vec<(String, usize)> {
    let mut qs = Vec::new();
    let n_v = VERBS.len();
    let n_o = OBJECTS.len();
    for qi in 0..64 {
        let doc = qi * 6;
        let i = doc / (n_v * n_o);
        let j = (doc / n_o) % n_v;
        let k = doc % n_o;
        qs.push((
            format!(
                "Which report documents that {} {} {}?",
                SUBJECTS[i], VERBS[j], OBJECTS[k]
            ),
            doc,
        ));
    }
    qs
}

fn cos(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn main() {
    let dump_path = std::env::args().nth(1);
    let ref_path = std::env::args().nth(2);
    let pack = std::env::var_os("PADDOCK_PACK")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../packs/cuda/build/pd-cuda-sm120.so")
        });
    let model = std::env::var("QWEN3_EMBED_GGUF").unwrap_or_else(|_| {
        "C:/dev/models/Qwen3-Embedding-0.6B-GGUF/Qwen3-Embedding-0.6B-Q8_0.gguf".into()
    });
    let exec = Arc::new(GpuExecutor::new(0, &pack).expect("cuda executor"));
    let map = MappedGguf::open(std::path::Path::new(&model)).expect("open gguf");
    let tok = GgufTokenizer::from_gguf(map.gguf()).expect("tokenizer");
    let mut m = GpuQwen3::load(exec, &map, 4096).expect("load qwen3");

    let eos = 151643u32;
    let docs = corpus();
    let qs = queries();
    let n_docs = docs.len();

    // one ragged batch over docs + queries: the row count (~17k) is what
    // routes the FFN GEMMs onto the prefill path under test
    let mut texts: Vec<&str> = docs.iter().map(String::as_str).collect();
    texts.extend(qs.iter().map(|(q, _)| q.as_str()));
    let seqs: Vec<Vec<u32>> = texts
        .iter()
        .map(|t| {
            let mut e = tok.encode(t).expect("encode");
            e.push(eos);
            e
        })
        .collect();
    let rows: usize = seqs.iter().map(Vec::len).sum();
    let embs = m.embed(&seqs).expect("embed");
    let dim = embs[0].len();
    eprintln!(
        "embedded {} docs + {} queries ({rows} rows, dim {dim})",
        n_docs,
        qs.len()
    );

    // recall@k of the known-relevant doc
    let mut top10: Vec<Vec<usize>> = Vec::new();
    let (mut r1, mut r10) = (0usize, 0usize);
    for (qi, (_, rel)) in qs.iter().enumerate() {
        let qe = &embs[n_docs + qi];
        let mut scored: Vec<(f32, usize)> = (0..n_docs).map(|d| (cos(qe, &embs[d]), d)).collect();
        scored.sort_by(|a, b| b.0.total_cmp(&a.0));
        let ranked: Vec<usize> = scored[..10].iter().map(|&(_, d)| d).collect();
        if ranked[0] == *rel {
            r1 += 1;
        }
        if ranked.contains(rel) {
            r10 += 1;
        }
        top10.push(ranked);
    }
    println!(
        "recall@1 {}/{} = {:.3}   recall@10 {}/{} = {:.3}",
        r1,
        qs.len(),
        r1 as f32 / qs.len() as f32,
        r10,
        qs.len(),
        r10 as f32 / qs.len() as f32
    );

    // dump embeddings for the cross-path compare
    if let Some(path) = &dump_path {
        let mut f = std::fs::File::create(path).expect("create dump");
        for e in &embs {
            let bytes: Vec<u8> = e.iter().flat_map(|v| v.to_le_bytes()).collect();
            f.write_all(&bytes).unwrap();
        }
        eprintln!("dumped {} x {dim} embeddings to {path}", embs.len());
    }

    // compare against a reference dump (the other numeric class)
    if let Some(path) = ref_path {
        let mut buf = Vec::new();
        std::fs::File::open(&path)
            .expect("open ref")
            .read_to_end(&mut buf)
            .unwrap();
        assert_eq!(buf.len(), embs.len() * dim * 4, "ref dump shape mismatch");
        let refs: Vec<Vec<f32>> = (0..embs.len())
            .map(|i| {
                buf[i * dim * 4..(i + 1) * dim * 4]
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect()
            })
            .collect();
        let mut min_c = f32::INFINITY;
        let mut sum_c = 0f64;
        for (e, r) in embs.iter().zip(&refs) {
            let c = cos(e, r);
            min_c = min_c.min(c);
            sum_c += c as f64;
        }
        // top-10 agreement per query against the ref embeddings' ranking
        let mut overlap = 0usize;
        for (qi, ranked) in top10.iter().enumerate() {
            let qe = &refs[n_docs + qi];
            let mut scored: Vec<(f32, usize)> =
                (0..n_docs).map(|d| (cos(qe, &refs[d]), d)).collect();
            scored.sort_by(|a, b| b.0.total_cmp(&a.0));
            overlap += scored[..10]
                .iter()
                .filter(|&&(_, d)| ranked.contains(&d))
                .count();
        }
        println!(
            "vs {path}: cos mean {:.5} min {:.5}   top10 overlap {:.3}",
            sum_c / embs.len() as f64,
            min_c,
            overlap as f32 / (top10.len() * 10) as f32
        );
    }
}
