//! GPU weight-pipeline parity on real gpt-oss weights: upload quantized
//! bytes -> on-device pack dequant -> in-house matvec (cuBLAS was deleted in
//! phase C), compared against the CPU reference (load_f32 + scalar
//! matvec). Not bit-exact (GEMM reduction order
//! differs) - gated by relative error, tight enough to catch any layout,
//! transpose, or dequant mistake instantly.
//!
//! Gated on: CUDA device + built pack + the gpt-oss download.

mod common;

use paddock_engine::gpu::GpuExecutor;
use paddock_engine::reference::load_f32;
use paddock_models::mapped::MappedGguf;

fn deterministic_input(n: usize, seed: u64) -> Vec<f32> {
    // tiny LCG - no rand dep, fully reproducible
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f32 / (1u64 << 31) as f32) - 0.5
        })
        .collect()
}

fn rel_err(a: &[f32], b: &[f32]) -> f32 {
    let mut num = 0f64;
    let mut den = 0f64;
    for (x, y) in a.iter().zip(b) {
        num += ((x - y) as f64).powi(2);
        den += (*y as f64).powi(2);
    }
    (num.sqrt() / den.sqrt().max(1e-12)) as f32
}

#[test]
fn gpu_matvec_matches_cpu_reference_on_real_weights() {
    let Some(model) = common::model("PADDOCK_MODEL", common::GPT_OSS_20B) else {
        return;
    };
    let Some(exec) = common::gpu() else {
        return;
    };
    let map = MappedGguf::open(&model).expect("open gguf");

    // Case 1: Q8_0 attention weight [2880 -> 4096]
    for name in ["blk.0.attn_q.weight", "blk.7.attn_output.weight"] {
        let w_gpu = exec.upload(&map, name).expect("upload+dequant");
        let w_cpu = load_f32(&map, name).expect("cpu load");
        let x = deterministic_input(w_cpu.dims[0], 42);

        let d_x = exec.to_device(&x).expect("x to device");
        let mut d_y = exec.alloc(w_cpu.dims[1]).expect("alloc y");
        exec.matvec_f32_batch(&w_gpu, &d_x, &mut d_y, 1)
            .expect("gemv");
        let gpu_y = exec.to_host(&d_y).expect("y to host");

        let mut cpu_y = vec![0f32; w_cpu.dims[1]];
        w_cpu.matvec(&x, &mut cpu_y);

        let err = rel_err(&gpu_y, &cpu_y);
        eprintln!("{name}: rel_err {err:.2e} over {} outputs", cpu_y.len());
        assert!(err < 1e-4, "{name}: rel err {err} too high");
    }

    // Case 1b: the same Q8_0 weights through the fused q8_0_gemv (A3) - no f32
    // materialization; must match the CPU reference as tightly as cuBLAS does.
    for name in ["blk.0.attn_q.weight", "blk.7.attn_output.weight"] {
        let w_q = exec.upload_raw(&map, name).expect("upload raw");
        let w_cpu = load_f32(&map, name).expect("cpu load");
        let x = deterministic_input(w_cpu.dims[0], 42);

        let d_x = exec.to_device(&x).expect("x to device");
        let mut d_y = exec.alloc(w_cpu.dims[1]).expect("alloc y");
        exec.q8_0_gemv(&w_q, None, &d_x, &mut d_y)
            .expect("q8_0 gemv");
        let gpu_y = exec.to_host(&d_y).expect("y to host");

        let mut cpu_y = vec![0f32; w_cpu.dims[1]];
        w_cpu.matvec(&x, &mut cpu_y);

        let err = rel_err(&gpu_y, &cpu_y);
        eprintln!(
            "{name} [q8_0_gemv]: rel_err {err:.2e} over {} outputs",
            cpu_y.len()
        );
        assert!(err < 1e-4, "{name}: q8_0_gemv rel err {err} too high");
    }

    // Case 1c: q8_0_gemv with the bias folded in - the model path. Compare to
    // CPU matvec + bias add.
    {
        let name = "blk.0.attn_q.weight";
        let w_q = exec.upload_raw(&map, name).expect("upload raw");
        let w_cpu = load_f32(&map, name).expect("cpu load");
        let bias_cpu = load_f32(&map, "blk.0.attn_q.bias").expect("cpu load bias");
        let bias_gpu = exec.upload(&map, "blk.0.attn_q.bias").expect("upload bias");
        let x = deterministic_input(w_cpu.dims[0], 42);

        let d_x = exec.to_device(&x).expect("x to device");
        let mut d_y = exec.alloc(w_cpu.dims[1]).expect("alloc y");
        exec.q8_0_gemv(&w_q, Some(&bias_gpu.buf), &d_x, &mut d_y)
            .expect("q8_0 gemv + bias");
        let gpu_y = exec.to_host(&d_y).expect("y to host");

        let mut cpu_y = vec![0f32; w_cpu.dims[1]];
        w_cpu.matvec(&x, &mut cpu_y);
        for (o, b) in cpu_y.iter_mut().zip(&bias_cpu.data) {
            *o += b;
        }

        let err = rel_err(&gpu_y, &cpu_y);
        eprintln!("{name} [q8_0_gemv+bias]: rel_err {err:.2e}");
        assert!(err < 1e-4, "{name}: q8_0_gemv+bias rel err {err} too high");
    }

    // Case 1d: the lm_head (out=201088) - the largest per-token GEMV and the one
    // whose argmax drives greedy decode, so its accuracy is what token parity
    // rides on. (The router ffn_gate_inp is F32, not Q8_0 - it rides
    // matvec_f32_batch; the q8_0_gemv type guard rejects it, so it isn't a
    // case here.)
    {
        let name = "output.weight";
        let w_q = exec.upload_raw(&map, name).expect("upload raw");
        let w_cpu = load_f32(&map, name).expect("cpu load");
        let x = deterministic_input(w_cpu.dims[0], 42);

        let d_x = exec.to_device(&x).expect("x to device");
        let mut d_y = exec.alloc(w_cpu.dims[1]).expect("alloc y");
        exec.q8_0_gemv(&w_q, None, &d_x, &mut d_y)
            .expect("q8_0 gemv");
        let gpu_y = exec.to_host(&d_y).expect("y to host");

        let mut cpu_y = vec![0f32; w_cpu.dims[1]];
        w_cpu.matvec(&x, &mut cpu_y);

        let err = rel_err(&gpu_y, &cpu_y);
        eprintln!(
            "{name} [q8_0_gemv]: rel_err {err:.2e} over {} outputs",
            cpu_y.len()
        );
        assert!(err < 1e-4, "{name}: q8_0_gemv rel err {err} too high");
    }

    // Case 2: MXFP4 expert tensor - full 3-D dequant on device, then matvec
    // through expert slice 3 of blk.0.ffn_gate_exps [2880, 2880, 32]
    let name = "blk.0.ffn_gate_exps.weight";
    let w_gpu = exec.upload(&map, name).expect("upload+dequant experts");
    let w_cpu = load_f32(&map, name).expect("cpu load experts");
    let (in_dim, ff_dim) = (w_cpu.dims[0], w_cpu.dims[1]);
    let expert = 3usize;
    let x = deterministic_input(in_dim, 7);

    // slice the device buffer at the expert offset; cuBLAS sees an [in, ff] weight
    let slice_len = in_dim * ff_dim;
    let d_slice = w_gpu
        .buf
        .try_slice(expert * slice_len..(expert + 1) * slice_len)
        .expect("expert slice");
    // device-to-device copy of the view into an owned buffer (test-only; the
    // real graph will gemm straight from the offset view)
    let mut owned = exec.alloc(slice_len).expect("alloc slice copy");
    exec.stream
        .memcpy_dtod(&d_slice, &mut owned)
        .expect("dtod copy");
    let slice_tensor = paddock_engine::gpu::DeviceTensor {
        buf: owned,
        dims: vec![in_dim, ff_dim],
    };

    let d_x = exec.to_device(&x).expect("x to device");
    let mut d_y = exec.alloc(ff_dim).expect("alloc y");
    exec.matvec_f32_batch(&slice_tensor, &d_x, &mut d_y, 1)
        .expect("gemv expert");
    let gpu_y = exec.to_host(&d_y).expect("y to host");

    let mut cpu_y = vec![0f32; ff_dim];
    let w_slice = &w_cpu.data[expert * slice_len..(expert + 1) * slice_len];
    for (o, row) in cpu_y.iter_mut().zip(w_slice.chunks_exact(in_dim)) {
        *o = row.iter().zip(&x).map(|(a, b)| a * b).sum();
    }

    let err = rel_err(&gpu_y, &cpu_y);
    eprintln!("{name}[expert {expert}]: rel_err {err:.2e}");
    assert!(err < 1e-4, "expert slice rel err {err} too high");
}

/// Empirically measure the machine's per-kernel launch overhead (the WDDM tax that
/// CUDA graphs eliminate). Launches a trivial kernel many times back-to-back and
/// divides by the count. If this is several µs, our ~400 launches/token cost
/// milliseconds - the untouched single-stream lever (llama.cpp replays one graph).
#[test]
fn launch_overhead_probe() {
    if !common::heavy() {
        return;
    }
    let Some(exec) = common::gpu() else {
        return;
    };
    let x = exec.to_device(&[0.1f32; 32]).expect("x");
    let w = exec.to_device(&[1.0f32; 32]).expect("w");
    let mut y = exec.alloc(32).expect("y");
    for _ in 0..50 {
        exec.rmsnorm(&x, &w, &mut y, 32, 1e-5).expect("warm");
    }
    exec.to_host(&y).expect("sync");
    let n = 20_000;
    let t0 = std::time::Instant::now();
    for _ in 0..n {
        exec.rmsnorm(&x, &w, &mut y, 32, 1e-5).expect("launch");
    }
    exec.to_host(&y).expect("sync");
    let per_us = t0.elapsed().as_secs_f64() * 1e6 / n as f64;
    eprintln!(
        "per-kernel launch: {per_us:.2} µs -> ~450 launches/token ≈ {:.2} ms/token launch tax",
        per_us * 450.0 / 1e3
    );
}

/// dp4a Q8_0 GEMV (quantized activation + integer dot, the llama.cpp/mistral.rs
/// method) vs the f32 dequant GEMV. Not bit-exact - the activation is quantized
/// to int8 - so this measures the accuracy cost of the method (expected well
/// under 1%). This is the "perplexity-parity, not token-parity" bar in miniature.
#[test]
fn q8_0_gemv_dp4a_close_to_f32() {
    let Some(model) = common::model("PADDOCK_MODEL", common::GPT_OSS_20B) else {
        return;
    };
    let Some(exec) = common::gpu() else {
        return;
    };
    let map = MappedGguf::open(&model).expect("open gguf");
    for name in ["blk.0.attn_q.weight", "output.weight"] {
        let w = exec.upload_raw(&map, name).expect("upload raw");
        let (in_dim, out_dim) = (w.dims[0], w.dims[1]);
        let x = deterministic_input(in_dim, 42);
        let d_x = exec.to_device(&x).expect("x");

        // f32 reference
        let mut d_yf = exec.alloc(out_dim).expect("yf");
        exec.q8_0_gemv(&w, None, &d_x, &mut d_yf).expect("gemv f32");
        let yf = exec.to_host(&d_yf).expect("yf host");

        // dp4a path: quantize activation, then integer GEMV
        let mut xq = exec.alloc_i8(in_dim).expect("xq");
        let mut xs = exec.alloc(in_dim / 32).expect("xs");
        exec.quantize_q8(&d_x, &mut xq, &mut xs, in_dim)
            .expect("quantize");
        let mut d_yd = exec.alloc(out_dim).expect("yd");
        exec.q8_0_gemv_dp4a(&w, None, &xq, &xs, &mut d_yd)
            .expect("gemv dp4a");
        let yd = exec.to_host(&d_yd).expect("yd host");

        let err = rel_err(&yd, &yf);
        eprintln!("{name} [dp4a vs f32]: rel_err {err:.2e} over {out_dim} outputs");
        assert!(err < 1e-2, "{name}: dp4a rel err {err} exceeds 1%");
    }
}

/// Speed check (run with --nocapture): dp4a Q8_0 GEMV vs f32 dequant GEMV on the
/// lm_head. Validates the whole thesis - the integer path should be materially
/// faster because it's ~10× fewer compute instructions per weight element.
#[test]
fn q8_0_gemv_dp4a_speed() {
    if !common::heavy() {
        return;
    }
    let Some(model) = common::model("PADDOCK_MODEL", common::GPT_OSS_20B) else {
        return;
    };
    let Some(exec) = common::gpu() else {
        return;
    };
    let map = MappedGguf::open(&model).expect("open gguf");
    let w = exec.upload_raw(&map, "output.weight").expect("upload raw");
    let (in_dim, out_dim) = (w.dims[0], w.dims[1]);
    let d_x = exec.to_device(&deterministic_input(in_dim, 42)).expect("x");
    let mut xq = exec.alloc_i8(in_dim).expect("xq");
    let mut xs = exec.alloc(in_dim / 32).expect("xs");
    let mut d_y = exec.alloc(out_dim).expect("y");
    let iters = 200;

    // warm + time f32
    for _ in 0..10 {
        exec.q8_0_gemv(&w, None, &d_x, &mut d_y).expect("gpu op");
    }
    exec.to_host(&d_y).expect("gpu op");
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        exec.q8_0_gemv(&w, None, &d_x, &mut d_y).expect("gpu op");
    }
    exec.to_host(&d_y).expect("gpu op");
    let f32_ms = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;

    // warm + time dp4a (quantize + gemv, both counted)
    for _ in 0..10 {
        exec.quantize_q8(&d_x, &mut xq, &mut xs, in_dim)
            .expect("gpu op");
        exec.q8_0_gemv_dp4a(&w, None, &xq, &xs, &mut d_y)
            .expect("gpu op");
    }
    exec.to_host(&d_y).expect("gpu op");
    let t1 = std::time::Instant::now();
    for _ in 0..iters {
        exec.quantize_q8(&d_x, &mut xq, &mut xs, in_dim)
            .expect("gpu op");
        exec.q8_0_gemv_dp4a(&w, None, &xq, &xs, &mut d_y)
            .expect("gpu op");
    }
    exec.to_host(&d_y).expect("gpu op");
    let dp4a_ms = t1.elapsed().as_secs_f64() * 1e3 / iters as f64;

    eprintln!(
        "lm_head GEMV: f32 {f32_ms:.3} ms | dp4a {dp4a_ms:.3} ms | speedup {:.2}×",
        f32_ms / dp4a_ms
    );
}

/// Vectorized repacked Q8_0 GEMV (the exact-f32 decode fast path) vs the current
/// interleaved f32 GEMV and the dp4a path - on the real Qwen3.5-9B lm_head (the
/// largest, decode-dominating GEMV). Reports sustained GB/s so the bandwidth win
/// is visible, and asserts the repacked path stays f32-tight (rel_err ~1e-6, so
/// the greedy-parity gate holds - unlike dp4a, which quantizes the activation).
#[test]
fn q8_0_gemv_repacked_speed() {
    if !common::heavy() {
        return;
    }
    let Some(model) = common::model("QWEN35_GGUF", common::QWEN35_9B_Q8) else {
        return;
    };
    let Some(exec) = common::gpu() else {
        return;
    };
    let map = MappedGguf::open(&model).expect("open gguf");
    // Every distinct GEMV shape in the decode path, real qwen weights. Reports
    // sustained GB/s per shape so we can see which shapes fall below the lm_head's
    // share of roofline and drag the in-model average down - gemv dominates
    // decode GPU time, so these shapes are where decode speed is decided.
    // layer 0 is DeltaNet ((0+1)%4!=0); layer 3 is full-attn.
    let shapes: &[(&str, &str)] = &[
        ("lm_head    ", "output.weight"),
        ("ffn_gate   ", "blk.0.ffn_gate.weight"),
        ("ffn_down   ", "blk.0.ffn_down.weight"),
        ("in_qkv     ", "blk.0.attn_qkv.weight"),
        ("ssm_alpha  ", "blk.0.ssm_alpha.weight"),
        ("ssm_out    ", "blk.0.ssm_out.weight"),
        ("attn_q(3)  ", "blk.3.attn_q.weight"),
        ("attn_k(3)  ", "blk.3.attn_k.weight"),
    ];
    let iters = 300;
    for (label, name) in shapes {
        let Some(_) = map.tensor_info(name) else {
            eprintln!("{label} {name}: absent - skip");
            continue;
        };
        let rp = exec.repack_q8(&map, name).expect("repack");
        let (in_dim, out_dim) = (rp.dims[0], rp.dims[1]);
        let bytes = (out_dim * in_dim / 32 * 34) as f64;
        let d_x = exec.to_device(&deterministic_input(in_dim, 42)).expect("x");
        let mut d_yr = exec.alloc(out_dim).expect("yr");

        // one-time correctness vs interleaved f32
        let w = exec.upload_raw(&map, name).expect("raw");
        let mut d_yf = exec.alloc(out_dim).expect("yf");
        exec.q8_0_gemv(&w, None, &d_x, &mut d_yf).unwrap();
        exec.q8_0_gemv_repacked(&rp, None, &d_x, &mut d_yr).unwrap();
        let er = rel_err(&exec.to_host(&d_yr).unwrap(), &exec.to_host(&d_yf).unwrap());
        assert!(er < 1e-4, "{label}: repacked rel_err {er} not f32-tight");

        for _ in 0..15 {
            exec.q8_0_gemv_repacked(&rp, None, &d_x, &mut d_yr).unwrap();
        }
        exec.synchronize().unwrap();
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            exec.q8_0_gemv_repacked(&rp, None, &d_x, &mut d_yr).unwrap();
        }
        exec.synchronize().unwrap();
        let s = t0.elapsed().as_secs_f64() / iters as f64;
        let gbs = bytes / s / 1e9;
        eprintln!(
            "{label} [{in_dim:>5}x{out_dim:<6}] {:>6.3} ms  {:>3.0} GB/s  ({:>2.0}% peak)  err {er:.1e}",
            s * 1e3,
            gbs,
            gbs / 768.0 * 100.0
        );
    }
}

/// Batched Q8_0 GEMM (the weight-amortizing kernel behind concurrent decode)
/// must equal running the single-row GEMV on each batch row independently.
#[test]
fn q8_0_gemm_matches_per_row_gemv() {
    let Some(model) = common::model("PADDOCK_MODEL", common::GPT_OSS_20B) else {
        return;
    };
    let Some(exec) = common::gpu() else {
        return;
    };
    let map = MappedGguf::open(&model).expect("open gguf");
    let w = exec
        .upload_raw(&map, "blk.0.attn_q.weight")
        .expect("upload raw");
    let bias = exec.upload(&map, "blk.0.attn_q.bias").expect("upload bias");
    let (in_dim, out_dim) = (w.dims[0], w.dims[1]);

    // odd batch so the tiling remainder path (batch % 8 != 0) is exercised
    let batch = 5usize;
    let mut xb = Vec::with_capacity(batch * in_dim);
    for b in 0..batch {
        xb.extend(deterministic_input(in_dim, 100 + b as u64));
    }
    let d_xb = exec.to_device(&xb).expect("xb to device");
    let mut d_yb = exec.alloc(batch * out_dim).expect("alloc yb");
    exec.q8_0_gemm(&w, Some(&bias.buf), &d_xb, &mut d_yb, batch)
        .expect("gemm");
    let gemm = exec.to_host(&d_yb).expect("yb to host");

    for b in 0..batch {
        let d_x = exec
            .to_device(&xb[b * in_dim..(b + 1) * in_dim])
            .expect("x row");
        let mut d_y = exec.alloc(out_dim).expect("alloc y");
        exec.q8_0_gemv(&w, Some(&bias.buf), &d_x, &mut d_y)
            .expect("gemv");
        let gemv = exec.to_host(&d_y).expect("y to host");
        let err = rel_err(&gemm[b * out_dim..(b + 1) * out_dim], &gemv);
        eprintln!("row {b}: gemm vs gemv rel_err {err:.2e}");
        assert!(err < 1e-5, "row {b}: gemm vs gemv rel err {err} too high");
    }
}

/// ncu profiling target: run only the MMA GEMM at prefill scale (B=512) on
/// lm_head, in a tight loop, so `ncu --kernel-name pd_q8_0_gemm_mma_kernel`
/// captures the prefill config in isolation. Gated on PADDOCK_MMA_PROF.
#[test]
fn mma_prefill_profile() {
    if std::env::var_os("PADDOCK_MMA_PROF").is_none() {
        return;
    }
    let Some(model) = common::model("QWEN35_GGUF", common::QWEN35_9B_Q8) else {
        return;
    };
    let Some(exec) = common::gpu() else {
        return;
    };
    let map = MappedGguf::open(&model).expect("gguf");
    let rp = exec
        .repack_q8(&map, "blk.0.ffn_gate.weight")
        .expect("repack");
    let (in_dim, out_dim) = (rp.dims[0], rp.dims[1]);
    let batch = 512usize;
    let mut xb = Vec::with_capacity(batch * in_dim);
    for b in 0..batch {
        xb.extend(deterministic_input(in_dim, 7 + b as u64));
    }
    let d_xb = exec.to_device(&xb).unwrap();
    let mut xq = exec.alloc_i8(batch * in_dim).unwrap();
    let mut xs = exec.alloc(batch * in_dim / 32).unwrap();
    exec.quantize_q8(&d_xb, &mut xq, &mut xs, batch * in_dim)
        .unwrap();
    let mut y = exec.alloc(batch * out_dim).unwrap();
    for _ in 0..30 {
        exec.q8_0_gemm_mma(&rp, &xq, &xs, &mut y, batch).unwrap();
    }
    exec.synchronize().unwrap();
}

/// The int8 tensor-core GEMM (`q8_0_gemm_mma`) must agree with the dp4a MT GEMM
/// to the shared int8-with-f32-per-block-scale numeric class: both compute exact
/// int32 block dots then f32-scale-accumulate, so they differ only in the f32
/// summation grouping (~1e-6). This is the fragment-layout gate - a wrong
/// A/B/D map produces gross error, not 1e-6. Speed reported with --nocapture.
#[test]
fn q8_0_gemm_mma_matches_dp4a() {
    let Some(model) = common::model("QWEN35_GGUF", common::QWEN35_9B_Q8) else {
        return;
    };
    let Some(exec) = common::gpu() else {
        return;
    };
    let map = MappedGguf::open(&model).expect("open gguf");
    // lm_head (out%16==0: 248320/16), ffn_gate (12288), in_qkv (5120) - the
    // verify path's shapes; batches spanning the spec-verify + serving regime
    let shapes = [
        "output.weight",
        "blk.0.ffn_gate.weight",
        "blk.0.attn_qkv.weight",
    ];
    for name in shapes {
        let Some(_) = map.tensor_info(name) else {
            eprintln!("{name}: absent - skip");
            continue;
        };
        let rp = exec.repack_q8(&map, name).expect("repack");
        let (in_dim, out_dim) = (rp.dims[0], rp.dims[1]);
        for &batch in &[8usize, 24, 32, 64, 128] {
            let mut xb = Vec::with_capacity(batch * in_dim);
            for b in 0..batch {
                xb.extend(deterministic_input(in_dim, 100 + b as u64));
            }
            let d_xb = exec.to_device(&xb).expect("xb");
            let mut xq = exec.alloc_i8(batch * in_dim).expect("xq");
            let mut xs = exec.alloc(batch * in_dim / 32).expect("xs");
            exec.quantize_q8(&d_xb, &mut xq, &mut xs, batch * in_dim)
                .expect("quantize");

            let mut d_ref = exec.alloc(batch * out_dim).expect("ref");
            let mut d_mma = exec.alloc(batch * out_dim).expect("mma");
            exec.q8_0_gemm_mt_dp4a(&rp, &xq, &xs, &mut d_ref, batch)
                .expect("dp4a");
            exec.q8_0_gemm_mma(&rp, &xq, &xs, &mut d_mma, batch)
                .expect("mma");
            let err = rel_err(
                &exec.to_host(&d_mma).unwrap(),
                &exec.to_host(&d_ref).unwrap(),
            );

            // speed (warm min-of-batches, --nocapture only)
            let mut speed = String::new();
            if std::env::var_os("PADDOCK_HEAVY_TESTS").is_some() {
                let bytes = (out_dim * in_dim / 32 * 34) as f64;
                let time = |f: &dyn Fn(&GpuExecutor)| -> f64 {
                    for _ in 0..10 {
                        f(&exec);
                    }
                    exec.synchronize().unwrap();
                    let mut best = f64::MAX;
                    for _ in 0..8 {
                        let t = std::time::Instant::now();
                        for _ in 0..20 {
                            f(&exec);
                        }
                        exec.synchronize().unwrap();
                        best = best.min(t.elapsed().as_secs_f64() / 20.0);
                    }
                    best
                };
                let td = time(&|e| {
                    e.q8_0_gemm_mt_dp4a(
                        &rp,
                        &xq,
                        &xs,
                        &mut exec.alloc(batch * out_dim).unwrap(),
                        batch,
                    )
                    .unwrap();
                });
                let tm = time(&|e| {
                    e.q8_0_gemm_mma(
                        &rp,
                        &xq,
                        &xs,
                        &mut exec.alloc(batch * out_dim).unwrap(),
                        batch,
                    )
                    .unwrap();
                });
                speed = format!(
                    "  dp4a {:.3}ms {:.0}GB/s | mma {:.3}ms {:.0}GB/s ({:.2}x)",
                    td * 1e3,
                    bytes / td / 1e9,
                    tm * 1e3,
                    bytes / tm / 1e9,
                    td / tm
                );
            }
            eprintln!("{name} B={batch} [mma vs dp4a]: rel_err {err:.2e}{speed}");
            assert!(
                err < 1e-4,
                "{name} B={batch}: mma vs dp4a rel_err {err} too high"
            );
        }

        // prefill regime: MMA vs the cublas f16 staging path (what prefill uses).
        // decides whether the current BN=64 MMA (re-reads weight ceil(B/64)x) can
        // win prefill or whether a wider-BN register-blocked GEMM is needed.
        if std::env::var_os("PADDOCK_HEAVY_TESTS").is_some() {
            let batch = 512usize;
            let mut xb = Vec::with_capacity(batch * in_dim);
            for b in 0..batch {
                xb.extend(deterministic_input(in_dim, 7 + b as u64));
            }
            let d_xb = exec.to_device(&xb).expect("xb");
            let mut xq = exec.alloc_i8(batch * in_dim).expect("xq");
            let mut xs = exec.alloc(batch * in_dim / 32).expect("xs");
            exec.quantize_q8(&d_xb, &mut xq, &mut xs, batch * in_dim)
                .expect("q");
            let mut d_w16 = exec.alloc_f16(in_dim * out_dim).expect("w16");
            let mut d_x16 = exec.alloc_f16(batch * in_dim).expect("x16");
            let bytes = (out_dim * in_dim / 32 * 34) as f64;
            let time = |f: &mut dyn FnMut()| -> f64 {
                for _ in 0..5 {
                    f();
                }
                exec.synchronize().unwrap();
                let mut best = f64::MAX;
                for _ in 0..6 {
                    let t = std::time::Instant::now();
                    for _ in 0..10 {
                        f();
                    }
                    exec.synchronize().unwrap();
                    best = best.min(t.elapsed().as_secs_f64() / 10.0);
                }
                best
            };
            let mut y1 = exec.alloc(batch * out_dim).unwrap();
            let mut y2 = exec.alloc(batch * out_dim).unwrap();
            let tm = time(&mut || {
                exec.q8_0_gemm_mma(&rp, &xq, &xs, &mut y1, batch).unwrap();
            });
            let tc = time(&mut || {
                exec.q8_0_repacked_to_f16(&rp, &mut d_w16).unwrap();
                exec.convert_f32_f16(&d_xb, &mut d_x16, batch * in_dim)
                    .unwrap();
                exec.gemm_f16_f32(&d_w16, &d_x16, &mut y2, in_dim, out_dim, batch)
                    .unwrap();
            });
            eprintln!(
                "{name} B=512 PREFILL: cublas {:.3}ms {:.0}GB/s | mma {:.3}ms {:.0}GB/s ({:.2}x)",
                tc * 1e3,
                bytes / tc / 1e9,
                tm * 1e3,
                bytes / tm / 1e9,
                tc / tm
            );
        }
    }
}

/// The mmq-class GEMM (`q8_0_gemm_mmq`, K-tile 256 / ntx=2 / one block per SM)
/// must agree with `q8_0_gemm_mma` - its activations are quantized by
/// `quantize_q8_mmq`, whose int8/scale values are bit-identical to
/// `quantize_q8` (only the placement differs), and the per-block f32
/// accumulation runs in the same k-major order, so the agreement should be at
/// or near bit-exact and is gated at the shared numeric class (~1e-6).
/// Speed vs mma/cublas reported under PADDOCK_HEAVY_TESTS (Gate 2 of P6e).
#[test]
fn q8_0_gemm_mmq_matches_mma() {
    let Some(model) = common::model("QWEN35_GGUF", common::QWEN35_9B_Q8) else {
        return;
    };
    let Some(exec) = common::gpu() else {
        return;
    };
    let map = MappedGguf::open(&model).expect("open gguf");
    // ffn_down (out 4096 -> 32 row tiles) is the stream-k shape: its B=512
    // grid is 128 tiles = 76% wave efficiency, so the launcher goes stream-k
    // (split-tile sums differ from mma by ~1e-7 instead of bit-exact).
    let shapes = [
        "output.weight",
        "blk.0.ffn_gate.weight",
        "blk.0.attn_qkv.weight",
        "blk.0.ffn_down.weight",
    ];
    // +256: the stream-k fold-flags tail (the in-kernel deferred fold reads
    // its per-block flag words at fixup + 256*16384 floats - every serving
    // allocation carries it). Sized without the tail, the flags land in
    // whatever pool memory follows: garbage flags corrupt the split-tile
    // folds, which was this gate's own shape-dependent 0.26-0.29 rel_err on
    // exactly the stream-k shapes.
    let mut skfix = exec.alloc(256 * 128 * 128 + 256).expect("skfix");
    for name in shapes {
        let Some(_) = map.tensor_info(name) else {
            eprintln!("{name}: absent - skip");
            continue;
        };
        let rp = exec.repack_q8(&map, name).expect("repack");
        let (in_dim, out_dim) = (rp.dims[0], rp.dims[1]);
        let n_chunks = in_dim.div_ceil(128);
        for &batch in &[96usize, 128, 512] {
            let batch_pad = batch.next_multiple_of(128);
            let mut xb = Vec::with_capacity(batch * in_dim);
            for b in 0..batch {
                xb.extend(deterministic_input(in_dim, 100 + b as u64));
            }
            let d_xb = exec.to_device(&xb).expect("xb");
            let mut xq = exec.alloc_i8(batch * in_dim).expect("xq");
            let mut xs = exec.alloc(batch * in_dim / 32).expect("xs");
            exec.quantize_q8(&d_xb, &mut xq, &mut xs, batch * in_dim)
                .expect("quantize");
            let mut yq = exec.alloc_u8(n_chunks * batch_pad * 144).expect("yq");
            exec.quantize_q8_mmq(&d_xb, &mut yq, in_dim, batch)
                .expect("quantize mmq");

            let mut d_ref = exec.alloc(batch * out_dim).expect("ref");
            let mut d_new = exec.alloc(batch * out_dim).expect("new");
            exec.q8_0_gemm_mma(&rp, &xq, &xs, &mut d_ref, batch)
                .expect("mma");
            exec.q8_0_gemm_mmq(&rp, &yq, Some(&mut skfix), &mut d_new, batch)
                .expect("mmq");
            let err = rel_err(
                &exec.to_host(&d_new).unwrap(),
                &exec.to_host(&d_ref).unwrap(),
            );

            let mut speed = String::new();
            if std::env::var_os("PADDOCK_HEAVY_TESTS").is_some() {
                let bytes = (out_dim * in_dim / 32 * 34) as f64;
                let time = |f: &mut dyn FnMut()| -> f64 {
                    for _ in 0..5 {
                        f();
                    }
                    exec.synchronize().unwrap();
                    let mut best = f64::MAX;
                    for _ in 0..6 {
                        let t = std::time::Instant::now();
                        for _ in 0..10 {
                            f();
                        }
                        exec.synchronize().unwrap();
                        best = best.min(t.elapsed().as_secs_f64() / 10.0);
                    }
                    best
                };
                let mut y1 = exec.alloc(batch * out_dim).unwrap();
                let mut y2 = exec.alloc(batch * out_dim).unwrap();
                // both timed with their quantize step - that is what prefill pays
                let tm = time(&mut || {
                    exec.quantize_q8(&d_xb, &mut xq, &mut xs, batch * in_dim)
                        .unwrap();
                    exec.q8_0_gemm_mma(&rp, &xq, &xs, &mut y1, batch).unwrap();
                });
                let tn = time(&mut || {
                    exec.quantize_q8_mmq(&d_xb, &mut yq, in_dim, batch).unwrap();
                    exec.q8_0_gemm_mmq(&rp, &yq, Some(&mut skfix), &mut y2, batch)
                        .unwrap();
                });
                // fixup=None forces plain tiling - isolates what stream-k buys
                let tt = time(&mut || {
                    exec.quantize_q8_mmq(&d_xb, &mut yq, in_dim, batch).unwrap();
                    exec.q8_0_gemm_mmq(&rp, &yq, None, &mut y2, batch).unwrap();
                });
                speed = format!(
                    "  mma {:.3}ms {:.0}GB/s | mmq {:.3}ms {:.0}GB/s ({:.2}x) | tiled {:.3}ms (sk {:.2}x)",
                    tm * 1e3,
                    bytes / tm / 1e9,
                    tn * 1e3,
                    bytes / tn / 1e9,
                    tm / tn,
                    tt * 1e3,
                    tt / tn
                );
            }
            eprintln!("{name} B={batch} [mmq vs mma]: rel_err {err:.2e}{speed}");
            assert!(
                err < 1e-4,
                "{name} B={batch}: mmq vs mma rel_err {err} too high"
            );
        }
    }
}

/// The one-launch f32 router matvec (pd_matvec_f32_batch) against a CPU f64
/// reference on the real gpt-oss router weight (the cuBLAS lane it used to
/// A/B against was deleted in phase C) - same math, different
/// summation order, so the bar is f32 reduction noise, not bitwise. One ulp
/// here can flip a top-4-of-32 expert pick, so anything above noise level is
/// a real bug.
#[test]
fn matvec_f32_batch_matches_reference() {
    let Some(model) = common::model("PADDOCK_MODEL", common::GPT_OSS_20B) else {
        return;
    };
    let Some(exec) = common::gpu() else {
        return;
    };
    let map = MappedGguf::open(&model).expect("open gguf");
    let w = exec
        .upload(&map, "blk.0.ffn_gate_inp.weight")
        .expect("router weight");
    let w_cpu = load_f32(&map, "blk.0.ffn_gate_inp.weight").expect("cpu router weight");
    let (in_dim, out_dim) = (w.dims[0], w.dims[1]);
    for &b in &[1usize, 4, 37, 64] {
        let x = deterministic_input(b * in_dim, 7 + b as u64);
        let d_x = exec.to_device(&x).unwrap();
        let mut y_k = exec.alloc(b * out_dim).unwrap();
        exec.matvec_f32_batch(&w, &d_x, &mut y_k, b).unwrap();
        let k = exec.to_host(&y_k).unwrap();
        // f64 CPU reference, row by row (w_cpu is [out][in] row-major)
        let mut c = vec![0f32; b * out_dim];
        for t in 0..b {
            for o in 0..out_dim {
                let mut acc = 0f64;
                for i in 0..in_dim {
                    acc += w_cpu.data[o * in_dim + i] as f64 * x[t * in_dim + i] as f64;
                }
                c[t * out_dim + o] = acc as f32;
            }
        }
        let err = rel_err(&k[..b * out_dim], &c[..b * out_dim]);
        eprintln!("matvec_f32_batch vs f64 reference b={b}: rel_err {err:.2e}");
        assert!(
            err < 1e-5,
            "router matvec rel err {err} above f32 reduction noise at b={b}"
        );
    }
}
