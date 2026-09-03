//! Parity tests for the Qwen3.5 DeltaNet-layer support ops (causal conv1d+SiLU,
//! gate math, gated RMSNorm) vs their CPU references in
//! `paddock_kernels::reference::delta_net`. Gated on a CUDA device + built pack.

mod common;

use paddock_engine::gpu::GpuExecutor;
use paddock_kernels::reference::delta_net::{causal_conv1d_silu, delta_gate, gated_rmsnorm};

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

fn exec_or_skip() -> Option<GpuExecutor> {
    common::gpu()
}

fn maxd(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[test]
fn causal_conv1d_silu_matches_cpu() {
    let Some(exec) = exec_or_skip() else { return };
    let (t, conv_dim, k) = (12usize, 64usize, 4usize);
    let x = det(t * conv_dim, 1);
    let w = det(conv_dim * k, 2);

    let mut ref_out = vec![0f32; t * conv_dim];
    causal_conv1d_silu(&x, &w, &mut ref_out, t, conv_dim, k);

    let d_x = exec.to_device(&x).expect("x");
    let d_w = exec.to_device(&w).expect("w");
    let mut d_out = exec.to_device(&vec![0f32; t * conv_dim]).expect("out");
    exec.causal_conv1d_silu(&d_x, &d_w, &mut d_out, t, conv_dim, k)
        .expect("causal_conv1d_silu");
    let got = exec.to_host(&d_out).expect("dtoh");

    let diff = maxd(&got, &ref_out);
    eprintln!("conv1d+silu parity: max_abs_diff {diff:.2e}");
    assert!(diff < 1e-4, "conv1d max_abs_diff {diff} too high");
}

#[test]
fn delta_gate_matches_cpu() {
    let Some(exec) = exec_or_skip() else { return };
    let (t, h) = (12usize, 32usize);
    let a = det(t * h, 1);
    let b = det(t * h, 2);
    let ssm_a: Vec<f32> = det(h, 3).iter().map(|x| -x.abs() - 0.1).collect(); // < 0
    let dt = det(h, 4);

    let mut g_ref = vec![0f32; t * h];
    let mut beta_ref = vec![0f32; t * h];
    delta_gate(&a, &b, &ssm_a, &dt, &mut g_ref, &mut beta_ref, t, h);

    let d_a = exec.to_device(&a).expect("a");
    let d_b = exec.to_device(&b).expect("b");
    let d_sa = exec.to_device(&ssm_a).expect("ssm_a");
    let d_dt = exec.to_device(&dt).expect("dt");
    let mut d_g = exec.to_device(&vec![0f32; t * h]).expect("g");
    let mut d_beta = exec.to_device(&vec![0f32; t * h]).expect("beta");
    exec.delta_gate(&d_a, &d_b, &d_sa, &d_dt, &mut d_g, &mut d_beta, t, h)
        .expect("delta_gate");
    let g_got = exec.to_host(&d_g).expect("g dtoh");
    let beta_got = exec.to_host(&d_beta).expect("beta dtoh");

    let gd = maxd(&g_got, &g_ref);
    let bd = maxd(&beta_got, &beta_ref);
    eprintln!("delta_gate parity: max|g| {gd:.2e}  max|beta| {bd:.2e}");
    assert!(
        gd < 1e-4 && bd < 1e-4,
        "delta_gate diff too high (g {gd}, beta {bd})"
    );
}

#[test]
fn gated_rmsnorm_matches_cpu() {
    let Some(exec) = exec_or_skip() else { return };
    let (n_rows, d, eps) = (16usize, 128usize, 1e-6f32);
    let x = det(n_rows * d, 1);
    let z = det(n_rows * d, 2);
    let w = det(d, 3);

    let mut ref_out = vec![0f32; n_rows * d];
    gated_rmsnorm(&x, &z, &w, &mut ref_out, n_rows, d, eps);

    let d_x = exec.to_device(&x).expect("x");
    let d_z = exec.to_device(&z).expect("z");
    let d_w = exec.to_device(&w).expect("w");
    let mut d_out = exec.to_device(&vec![0f32; n_rows * d]).expect("out");
    exec.gated_rmsnorm(&d_x, &d_z, &d_w, &mut d_out, n_rows, d, eps)
        .expect("gated_rmsnorm");
    let got = exec.to_host(&d_out).expect("dtoh");

    let diff = maxd(&got, &ref_out);
    eprintln!("gated_rmsnorm parity: max_abs_diff {diff:.2e}");
    assert!(diff < 1e-4, "gated_rmsnorm max_abs_diff {diff} too high");
}
