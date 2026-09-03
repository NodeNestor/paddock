//! Per-op GPU kernel parity vs the shared CPU reference ops. Elementwise ops
//! must match to float-libm noise; reduction ops to reduction-order noise;
//! rope uses the same multiplicative theta chain on both sides so only
//! sinf/cosf ulps differ (looser bound at large positions where argument
//! reduction differs).

mod common;

use paddock_engine::gpu::GpuExecutor;
use paddock_kernels::reference::ops;

fn det(n: usize, seed: u64) -> Vec<f32> {
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

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn executor() -> Option<GpuExecutor> {
    common::gpu()
}

#[test]
fn gpu_ops_match_cpu_reference() {
    let Some(exec) = executor() else { return };
    let table = exec.kernels().expect("kernel table");
    let stream_ptr = || exec.stream.cu_stream() as *mut core::ffi::c_void;

    // ---- rmsnorm (n = 2880, the gpt-oss embd)
    {
        let x = det(2880, 1);
        let w = det(2880, 2);
        let expected = ops::rms_norm(&x, &w, 1e-5);

        let d_x = exec.to_device(&x).expect("x");
        let d_w = exec.to_device(&w).expect("w");
        let mut d_out = exec.alloc(2880).expect("out");
        let f = table.rmsnorm_f32.expect("rmsnorm in pack");
        let status = unsafe {
            use cudarc::driver::{DevicePtr, DevicePtrMut};
            let (xp, _g1) = d_x.device_ptr(&exec.stream);
            let (wp, _g2) = d_w.device_ptr(&exec.stream);
            let (op, _g3) = d_out.device_ptr_mut(&exec.stream);
            f(
                xp as *const _,
                wp as *const _,
                op as *mut _,
                2880,
                1e-5,
                stream_ptr(),
            )
        };
        assert_eq!(status, 0);
        let got = exec.to_host(&d_out).expect("dtoh");
        let d = max_abs_diff(&got, &expected);
        eprintln!("rmsnorm max_abs_diff {d:.2e}");
        assert!(d < 1e-6);
    }

    // ---- rope yarn (gpt-oss params), positions incl. beyond original ctx
    {
        let yarn = ops::YarnRope::new(64, 150_000.0, 1.0 / 32.0, 4096, 1.0, 1.0, 32.0, 1.0);
        let (theta_scale, freq_scale, corr_low, corr_high, ext_factor, mscale) =
            yarn.kernel_params();
        for (pos, tol) in [(0u32, 1e-6f32), (5, 1e-6), (4097, 1e-4)] {
            let x = det(64 * 64, 3 + pos as u64);
            let mut expected = x.clone();
            for h in 0..64 {
                yarn.apply(&mut expected[h * 64..(h + 1) * 64], pos as usize);
            }
            let mut d_x = exec.to_device(&x).expect("x");
            let f = table.rope_yarn_f32.expect("rope in pack");
            let status = unsafe {
                use cudarc::driver::DevicePtrMut;
                let (xp, _g) = d_x.device_ptr_mut(&exec.stream);
                f(
                    xp as *mut _,
                    64,
                    64,
                    pos,
                    theta_scale,
                    freq_scale,
                    corr_low,
                    corr_high,
                    ext_factor,
                    mscale,
                    stream_ptr(),
                )
            };
            assert_eq!(status, 0);
            let got = exec.to_host(&d_x).expect("dtoh");
            let d = max_abs_diff(&got, &expected);
            eprintln!("rope pos {pos}: max_abs_diff {d:.2e}");
            assert!(d < tol, "pos {pos}: {d} >= {tol}");
        }
    }

    // ---- softmax with sink (sizes incl. a single-element edge)
    for n in [1usize, 128, 2048] {
        let mut expected = det(n, 9);
        let x = expected.clone();
        ops::softmax_with_sink(&mut expected, 0.7);

        let mut d_x = exec.to_device(&x).expect("x");
        let f = table.softmax_sink_f32.expect("softmax in pack");
        let status = unsafe {
            use cudarc::driver::DevicePtrMut;
            let (xp, _g) = d_x.device_ptr_mut(&exec.stream);
            f(xp as *mut _, n as u32, 0.7, stream_ptr())
        };
        assert_eq!(status, 0);
        let got = exec.to_host(&d_x).expect("dtoh");
        let d = max_abs_diff(&got, &expected);
        eprintln!("softmax_sink n={n}: max_abs_diff {d:.2e}");
        assert!(d < 1e-6);
    }

    // ---- swiglu_oai (values spanning the clamp range)
    {
        let mut gate: Vec<f32> = det(2880, 11).iter().map(|v| v * 30.0).collect();
        let up: Vec<f32> = det(2880, 12).iter().map(|v| v * 30.0).collect();
        let mut expected = gate.clone();
        ops::swiglu_oai(&mut expected, &up, 1.702, 7.0);

        let mut d_g = exec.to_device(&gate).expect("g");
        let d_u = exec.to_device(&up).expect("u");
        let f = table.swiglu_oai_f32.expect("swiglu in pack");
        let status = unsafe {
            use cudarc::driver::{DevicePtr, DevicePtrMut};
            let (gp, _g1) = d_g.device_ptr_mut(&exec.stream);
            let (upp, _g2) = d_u.device_ptr(&exec.stream);
            f(
                gp as *mut _,
                upp as *const _,
                2880,
                1.702,
                7.0,
                stream_ptr(),
            )
        };
        assert_eq!(status, 0);
        let got = exec.to_host(&d_g).expect("dtoh");
        let d = max_abs_diff(&got, &expected);
        eprintln!("swiglu_oai max_abs_diff {d:.2e}");
        assert!(d < 1e-5);
        gate.clear(); // silence unused-mut style concerns
    }

    // ---- add_inplace (must be bit-exact)
    {
        let x = det(4096, 20);
        let y = det(4096, 21);
        let expected: Vec<f32> = x.iter().zip(&y).map(|(a, b)| a + b).collect();

        let mut d_x = exec.to_device(&x).expect("x");
        let d_y = exec.to_device(&y).expect("y");
        let f = table.add_inplace_f32.expect("add in pack");
        let status = unsafe {
            use cudarc::driver::{DevicePtr, DevicePtrMut};
            let (xp, _g1) = d_x.device_ptr_mut(&exec.stream);
            let (yp, _g2) = d_y.device_ptr(&exec.stream);
            f(xp as *mut _, yp as *const _, 4096, stream_ptr())
        };
        assert_eq!(status, 0);
        let got = exec.to_host(&d_x).expect("dtoh");
        for (g, e) in got.iter().zip(&expected) {
            assert_eq!(g.to_bits(), e.to_bits());
        }
        eprintln!("add_inplace: bit-exact");
    }
}
