//! Measure the resident VRAM a forensics GPU context costs, so the Studio
//! will-it-fit estimate (`paddock-manager::estimate`) can price the forensics pass.
//!
//! Run on a box with a free-ish GPU:
//!   cargo run -p paddock-forensics --features cuda --example vram_footprint -- <device>
//!
//! `nvidia-smi` reads free memory without a CUDA context of our own, so the
//! before/after step is the forensics context + its 10 kernel modules - the
//! resident cost of turning forensics on (per-image scratch is transient and not
//! measured here). Other GPU processes add noise; the step is hundreds of MiB,
//! well above it.

#[cfg(feature = "cuda")]
fn free_mib(dev: usize) -> i64 {
    let out = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
            "-i",
            &dev.to_string(),
        ])
        .output()
        .expect("run nvidia-smi");
    String::from_utf8_lossy(&out.stdout)
        .trim()
        .parse()
        .unwrap_or(-1)
}

#[cfg(feature = "cuda")]
fn main() {
    let dev: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let settle = || std::thread::sleep(std::time::Duration::from_millis(800));

    let before = free_mib(dev);
    // First context: includes the process's PRIMARY-context baseline - what a
    // standalone measurement sees.
    let gpu1 = paddock_forensics::gpu::ForensicGpu::new(dev).expect("init forensics gpu 1");
    std::hint::black_box(&gpu1);
    settle();
    let after1 = free_mib(dev);

    // Second context: the primary context already exists (as it does in the
    // RUNNER, where the engine holds it), so this delta is the true INCREMENTAL
    // cost of adding forensics - modules + scratch, minus the shared baseline.
    let gpu2 = paddock_forensics::gpu::ForensicGpu::new(dev).expect("init forensics gpu 2");
    std::hint::black_box(&gpu2);
    settle();
    let after2 = free_mib(dev);

    println!("device                    : {dev}");
    println!("free before        (MiB)  : {before}");
    println!("free after ctx #1  (MiB)  : {after1}");
    println!("free after ctx #2  (MiB)  : {after2}");
    println!("standalone (ctx+modules)  : {} MiB", before - after1);
    println!("INCREMENTAL (2nd ctx)     : {} MiB", after1 - after2);
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("build with --features cuda to measure the GPU footprint");
}
