//! `pd_q8_0_gemv_dp4a_nc_multi` (entry 320) must be BIT-exact against the
//! single-plane `pd_q8_0_gemv_dp4a_nc` it merges: the multi kernel resolves
//! (segment, row) per block and runs the identical extracted row body on the
//! same staged activation, so every output float must match to the bit -
//! anything less means the merge changed math, not just launch economics
//! (the laguna q|k|v|g and shexp gate|up merges).
//!
//! Gated: skips (loudly) when the pack isn't built or no CUDA device exists.

mod common;

use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use paddock_kernels::KernelPack;

/// Deterministic xorshift64 - no external rand dependency, reproducible.
struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn next_i8_range(&mut self, span: i32) -> i8 {
        ((self.next_u64() % (2 * span as u64 + 1)) as i32 - span) as i8
    }
    fn next_unit_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

fn as_void(p: u64) -> *const core::ffi::c_void {
    p as *const core::ffi::c_void
}
fn as_void_mut(p: u64) -> *mut core::ffi::c_void {
    p as *mut core::ffi::c_void
}

/// One synthetic repacked-Q8_0 plane: int8 data [out_dim, in_dim] + f16 scale
/// [out_dim, in_dim/32] (pd_q8_0_repack's output layout).
struct Plane {
    data: CudaSlice<u8>,
    scale: CudaSlice<u8>,
    out_dim: u32,
}

fn make_plane(
    stream: &std::sync::Arc<CudaStream>,
    rng: &mut Rng,
    in_dim: u32,
    out_dim: u32,
) -> Plane {
    let n_blocks = (in_dim / 32) as usize;
    let data: Vec<u8> = (0..(out_dim as usize * in_dim as usize))
        .map(|_| rng.next_i8_range(100) as u8)
        .collect();
    let scale: Vec<u8> = (0..(out_dim as usize * n_blocks))
        .flat_map(|_| half::f16::from_f32(0.01 + 0.09 * rng.next_unit_f32().abs()).to_le_bytes())
        .collect();
    Plane {
        data: stream.clone_htod(&data).expect("htod plane data"),
        scale: stream.clone_htod(&scale).expect("htod plane scale"),
        out_dim,
    }
}

#[test]
fn q8_nc_multi_matches_single() {
    let Some(path) = common::pack() else {
        return;
    };
    let Some(ctx) = common::cuda() else {
        return;
    };
    let pack = KernelPack::load(&path).expect("pack load");
    let k = pack.kernels_v1().expect("v1 kernel table");
    let Some(multi) = k.q8_0_gemv_dp4a_nc_multi else {
        eprintln!("pack has no q8_0_gemv_dp4a_nc_multi - skipping (rebuild the pack)");
        return;
    };
    let single = k
        .q8_0_gemv_dp4a_nc
        .expect("pack provides q8_0_gemv_dp4a_nc");
    let quantize = k.quantize_q8.expect("pack provides quantize_q8");
    let stream = ctx.default_stream();
    let mut rng = Rng(0x5EED_0086_0806_2026);

    let in_dim: u32 = 2048;
    let n_blocks = (in_dim / 32) as usize;
    // laguna-shaped mix: distinct out_dims incl. the ragged 72 (g_proj's SWA
    // head count) so non-16-aligned segment boundaries are exercised too
    let plane_dims: [u32; 4] = [1152, 288, 288, 72];
    let planes: Vec<Plane> = plane_dims
        .iter()
        .map(|&od| make_plane(&stream, &mut rng, in_dim, od))
        .collect();

    // (n_segs, ncols): the c4 tick (4,4), a ramp tick (2,3), degenerate
    // single-seg (1,1), and padded-slot exercise at r=1 (4,1)
    for &(n_segs, ncols) in &[(4usize, 4u32), (2, 3), (1, 1), (4, 1)] {
        let label = format!("n_segs={n_segs} ncols={ncols}");
        let x: Vec<f32> = (0..(ncols as usize * in_dim as usize))
            .map(|_| rng.next_unit_f32())
            .collect();
        let d_x = stream.clone_htod(&x).expect("htod x");
        let mut d_xq = stream
            .alloc_zeros::<u8>(ncols as usize * in_dim as usize)
            .expect("alloc xq");
        let mut d_xs = stream
            .alloc_zeros::<f32>(ncols as usize * n_blocks)
            .expect("alloc xs");
        {
            let (xr, _g1) = d_x.device_ptr(&stream);
            let (xq, _g2) = d_xq.device_ptr_mut(&stream);
            let (xs, _g3) = d_xs.device_ptr_mut(&stream);
            let status = unsafe {
                quantize(
                    as_void(xr),
                    as_void_mut(xq),
                    as_void_mut(xs),
                    ncols * in_dim,
                    stream.cu_stream() as *mut core::ffi::c_void,
                )
            };
            assert_eq!(status, 0, "{label}: quantize_q8 failed");
        }

        // reference: one single-plane nc launch per segment
        let mut y_ref: Vec<Vec<f32>> = Vec::new();
        for p in planes.iter().take(n_segs) {
            let mut d_y = stream
                .alloc_zeros::<f32>(ncols as usize * p.out_dim as usize)
                .expect("alloc y_ref");
            {
                let (wd, _g1) = p.data.device_ptr(&stream);
                let (ws, _g2) = p.scale.device_ptr(&stream);
                let (xq, _g3) = d_xq.device_ptr(&stream);
                let (xs, _g4) = d_xs.device_ptr(&stream);
                let (yy, _g5) = d_y.device_ptr_mut(&stream);
                // SAFETY: device pointers + stream live for the call; ABI v1 contract
                let status = unsafe {
                    single(
                        as_void(wd),
                        as_void(ws),
                        as_void(xq),
                        as_void(xs),
                        as_void_mut(yy),
                        in_dim,
                        p.out_dim,
                        ncols,
                        stream.cu_stream() as *mut core::ffi::c_void,
                    )
                };
                assert_eq!(status, 0, "{label}: single nc launch failed");
            }
            stream.synchronize().expect("sync single");
            y_ref.push(stream.clone_dtoh(&d_y).expect("dtoh y_ref"));
        }

        // candidate: one multi launch over the same segments
        let mut d_ys: Vec<CudaSlice<f32>> = planes
            .iter()
            .take(n_segs)
            .map(|p| {
                stream
                    .alloc_zeros::<f32>(ncols as usize * p.out_dim as usize)
                    .expect("alloc y_multi")
            })
            .collect();
        {
            let null = std::ptr::null::<core::ffi::c_void>();
            let mut dp = [null; 4];
            let mut sp = [null; 4];
            let mut yp = [std::ptr::null_mut::<core::ffi::c_void>(); 4];
            let mut outs = [0u32; 4];
            let mut guards = Vec::new();
            for (i, (p, d_y)) in planes.iter().take(n_segs).zip(d_ys.iter_mut()).enumerate() {
                let (d, g1) = p.data.device_ptr(&stream);
                let (s, g2) = p.scale.device_ptr(&stream);
                let (yy, g3) = d_y.device_ptr_mut(&stream);
                dp[i] = as_void(d);
                sp[i] = as_void(s);
                yp[i] = as_void_mut(yy);
                outs[i] = p.out_dim;
                guards.push(g1);
                guards.push(g2);
                guards.push(g3);
            }
            let (xq, _g1) = d_xq.device_ptr(&stream);
            let (xs, _g2) = d_xs.device_ptr(&stream);
            // SAFETY: ABI contract; unused trailing segments pass nulls/0
            let status = unsafe {
                multi(
                    dp[0],
                    sp[0],
                    null,
                    yp[0],
                    outs[0],
                    dp[1],
                    sp[1],
                    null,
                    yp[1],
                    outs[1],
                    dp[2],
                    sp[2],
                    null,
                    yp[2],
                    outs[2],
                    dp[3],
                    sp[3],
                    null,
                    yp[3],
                    outs[3],
                    as_void(xq),
                    as_void(xs),
                    in_dim,
                    n_segs as u32,
                    ncols,
                    stream.cu_stream() as *mut core::ffi::c_void,
                )
            };
            assert_eq!(status, 0, "{label}: multi nc launch failed");
        }
        stream.synchronize().expect("sync multi");
        for (i, d_y) in d_ys.iter().enumerate() {
            let y = stream.clone_dtoh(d_y).expect("dtoh y_multi");
            assert_eq!(y.len(), y_ref[i].len(), "{label}: seg {i} length");
            for (j, (a, b)) in y.iter().zip(y_ref[i].iter()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "{label}: seg {i} elem {j} differs: multi {a} vs single {b}"
                );
            }
        }
        println!("{label}: bit-exact across {n_segs} segment(s)");
    }
}
