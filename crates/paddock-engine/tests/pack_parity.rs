//! GPU kernel-pack parity: the CUDA pack's dequant output must be BIT-EXACT
//! against the CPU reference implementations. This is the pattern every future
//! kernel follows - reference first, GPU diffed against it forever.
//!
//! Gated: skips (loudly) when the pack isn't built or no CUDA device exists.

mod common;

use cudarc::driver::{DevicePtr, DevicePtrMut};
use paddock_kernels::reference;
use paddock_kernels::{KernelPack, PackError};

#[test]
fn cuda_pack_dequants_match_cpu_reference_bit_exactly() {
    let Some(path) = common::pack() else {
        return;
    };
    let Some(ctx) = common::cuda() else {
        return;
    };

    let pack = match KernelPack::load(&path) {
        Ok(p) => p,
        Err(e @ PackError::AbiMismatch { .. }) => panic!("stale pack, rebuild it: {e}"),
        Err(e) => panic!("pack load failed: {e}"),
    };
    // the pack is a multi-arch fatbin (see packs/cuda/build.ps1); the file name
    // keeps the historical pd-cuda-sm86.dll for compatibility
    assert_eq!(pack.info().arch_str(), Some("cuda-multi"));
    let kernels = pack.kernels_v1().expect("v1 kernel table");

    // --- MXFP4: three blocks exercising scale paths (normal, 0.5, denormal)
    let mut input = Vec::new();
    for (e, fill) in [(128u8, 0x10u8), (127, 0xFF), (1, 0x73)] {
        input.push(e);
        input.extend((0..16).map(|j| fill.wrapping_add(j * 0x11)));
    }
    let n_blocks = 3u64;
    let mut expected = vec![0f32; 96];
    reference::dequant_mxfp4(&input, &mut expected).expect("cpu reference");

    let stream = ctx.default_stream();
    let d_in = stream.clone_htod(&input).expect("htod");
    let mut d_out = stream.alloc_zeros::<f32>(96).expect("alloc");

    let mxfp4 = kernels.mxfp4_dequant_f32.expect("pack provides mxfp4");
    {
        let (in_ptr, _g1) = d_in.device_ptr(&stream);
        let (out_ptr, _g2) = d_out.device_ptr_mut(&stream);
        // SAFETY: device pointers + stream are live for the call; ABI contract v1
        let status = unsafe {
            mxfp4(
                in_ptr as *const core::ffi::c_void,
                out_ptr as *mut core::ffi::c_void,
                n_blocks,
                stream.cu_stream() as *mut core::ffi::c_void,
            )
        };
        assert_eq!(status, 0, "mxfp4 launcher returned CUDA error {status}");
    }
    stream.synchronize().expect("sync");
    let got = stream.clone_dtoh(&d_out).expect("dtoh");

    for (i, (g, e)) in got.iter().zip(&expected).enumerate() {
        assert_eq!(
            g.to_bits(),
            e.to_bits(),
            "mxfp4 elem {i}: gpu {g} != cpu {e}"
        );
    }

    // --- Q8_0: one block, scale 0.5, values spanning the i8 range
    let mut q8: Vec<u8> = half::f16::from_f32(0.5).to_le_bytes().to_vec();
    q8.extend((0..32).map(|j| (j * 8 - 128) as i8 as u8));
    let mut q8_expected = vec![0f32; 32];
    reference::dequant_q8_0(&q8, &mut q8_expected).expect("cpu reference");

    let d_in8 = stream.clone_htod(&q8).expect("htod");
    let mut d_out8 = stream.alloc_zeros::<f32>(32).expect("alloc");
    let q8_fn = kernels.q8_0_dequant_f32.expect("pack provides q8_0");
    {
        let (in8, _g3) = d_in8.device_ptr(&stream);
        let (out8, _g4) = d_out8.device_ptr_mut(&stream);
        // SAFETY: as above
        let status = unsafe {
            q8_fn(
                in8 as *const core::ffi::c_void,
                out8 as *mut core::ffi::c_void,
                1,
                stream.cu_stream() as *mut core::ffi::c_void,
            )
        };
        assert_eq!(status, 0);
    }
    stream.synchronize().expect("sync");
    let got8 = stream.clone_dtoh(&d_out8).expect("dtoh");
    for (i, (g, e)) in got8.iter().zip(&q8_expected).enumerate() {
        assert_eq!(
            g.to_bits(),
            e.to_bits(),
            "q8_0 elem {i}: gpu {g} != cpu {e}"
        );
    }

    eprintln!("GPU pack parity: mxfp4 96/96 and q8_0 32/32 elements bit-exact");
}
