//! Parity for the Qwen3.5 partial sectioned M-RoPE kernel (`pd_mrope`) and the
//! sigmoid output gate (`pd_mul_sigmoid`) against their CPU references. The rope
//! is exercised in both modes that matter for a multimodal model:
//!   - text: all four position axes carry the token index (collapses to plain
//!     partial NEOX rope), and
//!   - vision: distinct temporal/height/width position ids per token,
//!     so the section->axis wiring is verified, not just the text special case.
//!     Gated on a CUDA device + built pack.

mod common;

use paddock_engine::gpu::GpuExecutor;
use paddock_kernels::reference::ops::{YarnRope, swiglu};
use paddock_kernels::reference::qwen35_attn::{mrope, sigmoid_gate};

fn det(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / (1u64 << 31) as f32) - 0.5
        })
        .collect()
}

fn exec() -> Option<GpuExecutor> {
    common::gpu()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

// Run pd_mrope for one position layout and diff against the CPU reference.
fn check_mrope(exec: &GpuExecutor, positions: &[u32], tag: &str) {
    // Real Qwen3.5-9B attention geometry: head_dim 256, n_rot 64, sections
    // [11,11,10,0], freq_base 1e7, no yarn.
    let (t, n_heads, head_dim, n_rot) = (12usize, 16usize, 256usize, 64usize);
    let sections = [11u32, 11, 10, 0];
    let rope = YarnRope::new(n_rot, 1e7, 1.0, 4096, 0.0, 1.0, 32.0, 1.0);

    let x = det(t * n_heads * head_dim, 42);
    let pos_f: Vec<f32> = positions.iter().map(|&p| p as f32).collect();

    let mut want = x.clone();
    mrope(&mut want, &pos_f, &sections, t, n_heads, head_dim, &rope);

    let mut d_x = exec.to_device(&x).expect("x");
    let d_pos = exec.stream.clone_htod(&positions.to_vec()).expect("pos");
    exec.mrope(
        &mut d_x,
        &d_pos,
        t,
        n_heads,
        head_dim,
        n_rot,
        rope.kernel_params(),
        sections,
    )
    .expect("mrope");
    let got = exec.to_host(&d_x).expect("dtoh");

    let diff = max_abs_diff(&got, &want);
    eprintln!("mrope parity ({tag}): max_abs_diff {diff:.2e}");
    assert!(diff < 1e-4, "mrope {tag} max_abs_diff {diff} too high");
}

#[test]
fn mrope_text_and_vision_match_cpu() {
    let Some(exec) = exec() else { return };
    let t = 12usize;

    // text: every axis is the token index.
    let mut text = vec![0u32; 4 * t];
    for axis in 0..4 {
        for ti in 0..t {
            text[axis * t + ti] = ti as u32;
        }
    }
    check_mrope(&exec, &text, "text");

    // vision: distinct temporal / height / width position ids (extra axis 0).
    let mut vis = vec![0u32; 4 * t];
    for ti in 0..t {
        vis[ti] = (ti / 3) as u32; // temporal: coarse
        vis[t + ti] = (ti % 4) as u32; // height
        vis[2 * t + ti] = (ti % 3) as u32; // width
        vis[3 * t + ti] = 0;
    }
    check_mrope(&exec, &vis, "vision");
}

#[test]
fn mul_sigmoid_matches_cpu() {
    let Some(exec) = exec() else { return };
    let n = 4096usize;
    let x = det(n, 7);
    let gate = det(n, 8).iter().map(|v| v * 12.0).collect::<Vec<_>>(); // wide range

    let mut want = x.clone();
    sigmoid_gate(&mut want, &gate);

    let mut d_x = exec.to_device(&x).expect("x");
    let d_gate = exec.to_device(&gate).expect("gate");
    exec.mul_sigmoid(&mut d_x, &d_gate, n).expect("mul_sigmoid");
    let got = exec.to_host(&d_x).expect("dtoh");

    let diff = max_abs_diff(&got, &want);
    eprintln!("mul_sigmoid parity: max_abs_diff {diff:.2e}");
    assert!(diff < 1e-5, "mul_sigmoid max_abs_diff {diff} too high");
}

#[test]
fn swiglu_matches_cpu() {
    let Some(exec) = exec() else { return };
    let n = 12288usize; // real Qwen3.5-9B FFN width
    let gate = det(n, 11);
    let up = det(n, 12);

    let mut want = gate.clone();
    swiglu(&mut want, &up);

    let mut d_gate = exec.to_device(&gate).expect("gate");
    let d_up = exec.to_device(&up).expect("up");
    exec.swiglu(&mut d_gate, &d_up, n).expect("swiglu");
    let got = exec.to_host(&d_gate).expect("dtoh");

    let diff = max_abs_diff(&got, &want);
    eprintln!("swiglu parity: max_abs_diff {diff:.2e}");
    assert!(diff < 1e-5, "swiglu max_abs_diff {diff} too high");
}
