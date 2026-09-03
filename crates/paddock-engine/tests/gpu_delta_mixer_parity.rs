//! Parity test for the assembled Qwen3.5 DeltaNet mixer core: chain the five GPU
//! kernels (conv+silu -> split/GQA -> gate -> recurrence -> gated RMSNorm) and
//! diff against the CPU composition `reference::delta_net::deltanet_mixer_core`.
//! Covers the on-device wiring (split offsets, GQA repeat, z alignment). Gated on
//! a CUDA device + built pack.

mod common;

use paddock_kernels::reference::delta_net::deltanet_mixer_core;

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

#[test]
fn deltanet_mixer_core_matches_cpu() {
    let Some(exec) = common::gpu() else {
        return;
    };

    // Real Qwen3.5-9B DeltaNet geometry: 16 key heads, 32 value heads (GQA 2x),
    // head dim (state size) 128, conv kernel 4.
    let (t, n_k, n_v, s, k) = (16usize, 16usize, 32usize, 128usize, 4usize);
    let key_dim = s * n_k;
    let value_dim = s * n_v;
    let conv_dim = 2 * key_dim + value_dim;
    let eps = 1e-6f32;

    let mixed = det(t * conv_dim, 1);
    let z = det(t * value_dim, 2);
    let a = det(t * n_v, 3);
    let b = det(t * n_v, 4);
    let conv_w = det(conv_dim * k, 5);
    let ssm_a: Vec<f32> = det(n_v, 6).iter().map(|x| -x.abs() - 0.1).collect(); // < 0
    let dt = det(n_v, 7);
    let norm_w = det(s, 8);

    // CPU reference (the whole mixer core in one call)
    let mut ref_core = vec![0f32; t * value_dim];
    deltanet_mixer_core(
        &mixed,
        &z,
        &a,
        &b,
        &conv_w,
        &ssm_a,
        &dt,
        &norm_w,
        &mut ref_core,
        t,
        n_k,
        n_v,
        s,
        k,
        eps,
    );

    // GPU: the five kernels chained.
    let d_mixed = exec.to_device(&mixed).expect("mixed");
    let d_convw = exec.to_device(&conv_w).expect("conv_w");
    let mut d_conv = exec.to_device(&vec![0f32; t * conv_dim]).expect("conv");
    exec.causal_conv1d_silu(&d_mixed, &d_convw, &mut d_conv, t, conv_dim, k)
        .expect("conv");

    let hv_elems = t * n_v * s;
    let mut d_q = exec.to_device(&vec![0f32; hv_elems]).expect("q");
    let mut d_k = exec.to_device(&vec![0f32; hv_elems]).expect("k");
    let mut d_v = exec.to_device(&vec![0f32; hv_elems]).expect("v");
    exec.deltanet_split_gqa(&d_conv, &mut d_q, &mut d_k, &mut d_v, t, n_k, n_v, s)
        .expect("split_gqa");

    let d_a = exec.to_device(&a).expect("a");
    let d_b = exec.to_device(&b).expect("b");
    let d_sa = exec.to_device(&ssm_a).expect("ssm_a");
    let d_dt = exec.to_device(&dt).expect("dt");
    let mut d_g = exec.to_device(&vec![0f32; t * n_v]).expect("g");
    let mut d_beta = exec.to_device(&vec![0f32; t * n_v]).expect("beta");
    exec.delta_gate(&d_a, &d_b, &d_sa, &d_dt, &mut d_g, &mut d_beta, t, n_v)
        .expect("gate");

    let mut d_state = exec.to_device(&vec![0f32; n_v * s * s]).expect("state");
    let mut d_attn = exec.to_device(&vec![0f32; hv_elems]).expect("attn");
    exec.gated_delta_recurrent(
        &d_q,
        &d_k,
        &d_v,
        &d_g,
        &d_beta,
        &mut d_state,
        &mut d_attn,
        t,
        n_v,
        s,
    )
    .expect("recurrent");

    let d_z = exec.to_device(&z).expect("z");
    let d_normw = exec.to_device(&norm_w).expect("norm_w");
    let mut d_core = exec.to_device(&vec![0f32; t * value_dim]).expect("core");
    exec.gated_rmsnorm(&d_attn, &d_z, &d_normw, &mut d_core, t * n_v, s, eps)
        .expect("gated_rmsnorm");

    let got = exec.to_host(&d_core).expect("dtoh");
    let diff = got
        .iter()
        .zip(&ref_core)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    eprintln!("deltanet mixer core parity: max_abs_diff {diff:.2e}");
    assert!(diff < 1e-4, "mixer core max_abs_diff {diff} too high");
}

/// Same mixer composition through the PRODUCTION chain since the v2 rewrite:
/// conv+silu -> split/GQA with fused L2-norm -> gate -> v2 recurrence -> gated
/// RMSNorm. Covers split_gqa_norm's GQA indexing + normalization and the v2
/// kernel end-to-end against the same CPU reference.
#[test]
fn deltanet_mixer_core_matches_cpu_v2_chain() {
    let Some(exec) = common::gpu() else {
        return;
    };

    let (t, n_k, n_v, s, k) = (16usize, 16usize, 32usize, 128usize, 4usize);
    let key_dim = s * n_k;
    let value_dim = s * n_v;
    let conv_dim = 2 * key_dim + value_dim;
    let eps = 1e-6f32;

    let mixed = det(t * conv_dim, 1);
    let z = det(t * value_dim, 2);
    let a = det(t * n_v, 3);
    let b = det(t * n_v, 4);
    let conv_w = det(conv_dim * k, 5);
    let ssm_a: Vec<f32> = det(n_v, 6).iter().map(|x| -x.abs() - 0.1).collect();
    let dt = det(n_v, 7);
    let norm_w = det(s, 8);

    let mut ref_core = vec![0f32; t * value_dim];
    deltanet_mixer_core(
        &mixed,
        &z,
        &a,
        &b,
        &conv_w,
        &ssm_a,
        &dt,
        &norm_w,
        &mut ref_core,
        t,
        n_k,
        n_v,
        s,
        k,
        eps,
    );

    let d_mixed = exec.to_device(&mixed).expect("mixed");
    let d_convw = exec.to_device(&conv_w).expect("conv_w");
    let mut d_conv = exec.to_device(&vec![0f32; t * conv_dim]).expect("conv");
    exec.causal_conv1d_silu(&d_mixed, &d_convw, &mut d_conv, t, conv_dim, k)
        .expect("conv");

    let hv_elems = t * n_v * s;
    let mut d_q = exec.to_device(&vec![0f32; hv_elems]).expect("q");
    let mut d_k = exec.to_device(&vec![0f32; hv_elems]).expect("k");
    let mut d_v = exec.to_device(&vec![0f32; hv_elems]).expect("v");
    exec.deltanet_split_gqa_norm(&d_conv, &mut d_q, &mut d_k, &mut d_v, t, n_k, n_v, s)
        .expect("split_gqa_norm");

    let d_a = exec.to_device(&a).expect("a");
    let d_b = exec.to_device(&b).expect("b");
    let d_sa = exec.to_device(&ssm_a).expect("ssm_a");
    let d_dt = exec.to_device(&dt).expect("dt");
    let mut d_g = exec.to_device(&vec![0f32; t * n_v]).expect("g");
    let mut d_beta = exec.to_device(&vec![0f32; t * n_v]).expect("beta");
    exec.delta_gate(&d_a, &d_b, &d_sa, &d_dt, &mut d_g, &mut d_beta, t, n_v)
        .expect("gate");

    let mut d_state = exec.to_device(&vec![0f32; n_v * s * s]).expect("state");
    let mut d_attn = exec.to_device(&vec![0f32; hv_elems]).expect("attn");
    exec.gated_delta_recurrent_v2(
        &d_q,
        &d_k,
        &d_v,
        &d_g,
        &d_beta,
        None,
        &mut d_state,
        0,
        None,
        &mut d_attn,
        1,
        t,
        n_v,
        s,
    )
    .expect("recurrent v2");

    let d_z = exec.to_device(&z).expect("z");
    let d_normw = exec.to_device(&norm_w).expect("norm_w");
    let mut d_core = exec.to_device(&vec![0f32; t * value_dim]).expect("core");
    exec.gated_rmsnorm(&d_attn, &d_z, &d_normw, &mut d_core, t * n_v, s, eps)
        .expect("gated_rmsnorm");

    let got = exec.to_host(&d_core).expect("dtoh");
    let diff = got
        .iter()
        .zip(&ref_core)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    eprintln!("deltanet mixer core parity (v2 chain): max_abs_diff {diff:.2e}");
    assert!(
        diff < 1e-4,
        "mixer core (v2 chain) max_abs_diff {diff} too high"
    );
}
