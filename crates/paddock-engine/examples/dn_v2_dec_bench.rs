//! DeltaNet v2 DECODE-step floor probe: the serving shape is batch=32 slots x
//! n_tokens=1 (one recurrence step per tick), H=32, D=128 - state traffic is
//! 32*32*128*128*4 B read + the same written = ~134 MB per layer call, DRAM
//! floor ~75 us at 1.79 TB/s. The serving trace's 56.5 us AVERAGE mixes these
//! with unified-span calls (T<384 spans amortize the state over many tokens),
//! so this probe times the pure decode shape at several batches to settle
//! whether the kernel is at its floor (=> only byte-reduction could help) or
//! latency-bound (=> kernel work exists).
// A development probe: it runs on a box its author is looking at, and a
// failure should stop it where it happened rather than be reported.
#![allow(clippy::unwrap_used)]

use std::sync::Arc;

use paddock_engine::gpu::GpuExecutor;

fn main() {
    let pack = std::env::var_os("PADDOCK_PACK")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../packs/cuda/build/pd-cuda-sm120.so")
        });
    let exec = Arc::new(GpuExecutor::new(0, &pack).expect("executor"));
    let (h, d) = (32usize, 128usize);
    let fill = |n: usize, seed: u32| -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                ((s >> 8) as f32 / (1u32 << 24) as f32) - 0.5
            })
            .collect()
    };
    for &b in &[8usize, 16, 32] {
        let n = b * h * d; // n_tokens = 1: one row per slot
        let d_q = exec.to_device(&fill(n, 1)).expect("q");
        let d_k = exec.to_device(&fill(n, 2)).expect("k");
        let d_v = exec.to_device(&fill(n, 3)).expect("v");
        let d_g = exec
            .to_device(
                &fill(b * h, 4)
                    .iter()
                    .map(|x| x * 0.1 - 0.1)
                    .collect::<Vec<_>>(),
            )
            .expect("g");
        let d_beta = exec
            .to_device(&fill(b * h, 5).iter().map(|x| x + 0.5).collect::<Vec<_>>())
            .expect("beta");
        let slots: Vec<u32> = (0..b as u32).collect();
        let d_slots = exec.to_device_u32(&slots).expect("slots");
        // serving reality: each of ~30 Linear layers has its own state, so the
        // per-call working set is L2-COLD. Cycle NL layer-sized buffers.
        const NL: usize = 30;
        let mut states: Vec<_> = (0..NL)
            .map(|_| exec.alloc(b * h * d * d).expect("state"))
            .collect();
        let mut d_out = exec.alloc(n).expect("out");
        // warm
        for state in states.iter_mut() {
            exec.gated_delta_recurrent_v2(
                &d_q,
                &d_k,
                &d_v,
                &d_g,
                &d_beta,
                Some(&d_slots),
                state,
                0,
                None,
                &mut d_out,
                b,
                1,
                h,
                d,
            )
            .expect("v2");
        }
        exec.synchronize().expect("sync");
        let t0 = std::time::Instant::now();
        let iters = 150;
        for i in 0..iters {
            let li = i % NL;
            exec.gated_delta_recurrent_v2(
                &d_q,
                &d_k,
                &d_v,
                &d_g,
                &d_beta,
                Some(&d_slots),
                &mut states[li],
                0,
                None,
                &mut d_out,
                b,
                1,
                h,
                d,
            )
            .expect("v2");
        }
        exec.synchronize().expect("sync");
        let us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
        let mb = (b * h * d * d * 4 * 2) as f64 / 1e6; // state read+write
        println!(
            "b={b:2} n=1 COLD: {us:7.1} us  state r+w {mb:6.1} MB -> {:.2} TB/s effective (floor ~{:.1} us @1.79TB/s)",
            mb / us * 1e-3 * 1e3,
            mb * 1e6 / 1.79e12 * 1e6 / 1e6
        );
    }
}
