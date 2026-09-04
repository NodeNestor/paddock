//! Dev probe: UD-Q4_K_XL forward vs the Q8_0 forward of the same model.
//! Quantization moves logits a little; a wiring bug moves them completely.
//! Prints top-5 ids + cosine similarity for a few prompts - triage tool, only
//! asserts the correlation floor.
// Test code: a failed assumption stops the test where it happened.
#![allow(clippy::unwrap_used)]

mod common;

use paddock_engine::gpu::GpuExecutor;
use paddock_engine::gpu_model::qwen35::GpuQwen35;
use paddock_models::mapped::MappedGguf;
use std::sync::Arc;

fn top5(v: &[f32]) -> Vec<(usize, f32)> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_unstable_by(|&a, &b| v[b].partial_cmp(&v[a]).unwrap());
    idx[..5].iter().map(|&i| (i, v[i])).collect()
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        dot += *x as f64 * *y as f64;
        na += (*x as f64).powi(2);
        nb += (*y as f64).powi(2);
    }
    dot / (na.sqrt() * nb.sqrt()).max(1e-12)
}

#[test]
fn ud_logits_track_q8_logits() {
    // the pack path stays: this probe builds one executor per model
    let Some(pack) = common::pack() else {
        return;
    };
    let Some(ud) = common::model("QWEN35_UD_GGUF", common::QWEN35_9B_UD_Q4) else {
        return;
    };
    let q8 = ud.with_file_name("Qwen3.5-9B-Q8_0.gguf");
    if !q8.exists() {
        common::missing(&format!("no Q8_0 beside {}", ud.display()));
        return;
    }

    // eager prefill + per-layer residual norms - the triage signal
    // SAFETY: test-local, set before any engine thread spawns
    unsafe {
        std::env::set_var("PADDOCK_NO_PREFILL_GRAPH", "1");
        std::env::set_var("PADDOCK_DEBUG_LAYER_NORMS", "1");
    }

    let prompts: Vec<Vec<u32>> = vec![
        vec![9707],                           // single token
        vec![785, 6722, 315, 23190, 374],     // a short phrase
        vec![151644, 872, 198, 9707, 151645], // template-ish ids
    ];

    let run = |path: &std::path::Path| -> Vec<Vec<f32>> {
        let exec = Arc::new(GpuExecutor::new(0, &pack).expect("exec"));
        let map = MappedGguf::open(path).expect("open");
        let mut m = GpuQwen35::load(exec, &map, 4096).expect("load");
        let mut out = Vec::new();
        for p in &prompts {
            m.reset();
            out.push(m.prefill(p).expect("prefill"));
        }
        out
    };

    eprintln!("== loading UD ==");
    let lud = run(&ud);
    eprintln!("== loading Q8 ==");
    let lq8 = run(&q8);

    for (i, (a, b)) in lud.iter().zip(&lq8).enumerate() {
        let c = cosine(a, b);
        eprintln!(
            "prompt {i}: cosine {c:.4}\n  UD  top5 {:?}\n  Q8  top5 {:?}",
            top5(a),
            top5(b)
        );
        assert!(
            c > 0.90,
            "prompt {i}: UD logits decorrelated from Q8 (cos {c:.4})"
        );
    }
}
