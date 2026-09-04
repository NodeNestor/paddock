//! Compile the forensic CUDA kernels to a **multi-arch fatbin** - the same
//! packaging paddock's own kernel pack uses (`packs/cuda/build.sh`), trimmed to
//! what forensics needs (no block-scale/tcgen05 MoE defines).
//!
//! Why a fatbin and not the `-arch=compute_80` PTX/JIT shortcut: paddock
//! is GPU-first across sm_86/89/90/100/120, so we bake real SASS per arch and
//! keep a PTX fallback only for an unseen future arch. The resulting `.fatbin`
//! is `include_bytes!`'d into the binary and handed to cudarc via
//! `Ptx::from_binary` (`cuModuleLoadData`) - no runtime nvcc, no JIT on any
//! supported device, and it never enters the LLM `KernelTableV1`.
//!
//! This runs only when the `cuda` feature is on (Cargo sets `CARGO_FEATURE_CUDA`).
//! A GPU-less / CPU-only build compiles nothing here.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Arches we target, in priority order. Mirrors `packs/cuda/build.sh`'s list
/// (minus the ones paddock only needs for its MoE families). Any arch the
/// installed toolkit does not know is skipped with a warning, exactly like the
/// shell twin, so an older CUDA still builds a valid (smaller) fatbin.
const TARGET_ARCHES: &[&str] = &["86", "89", "90", "100", "120", "121"];

/// The lowest virtual arch we also emit as PTX, so a GPU newer than anything in
/// TARGET_ARCHES still runs (the driver JITs the PTX). sm_86 PTX runs on every
/// arch we care about and beyond.
const PTX_FALLBACK_ARCH: &str = "86";

fn main() {
    // Rebuild if the CUDA sources change regardless of feature, so toggling the
    // feature on later picks them up.
    println!("cargo:rerun-if-changed=cuda/");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");

    if env::var_os("CARGO_FEATURE_CUDA").is_none() {
        // CPU-only build: nothing to compile. The GPU code is #[cfg(feature =
        // "cuda")] and will not reference any FORENSICS_FATBIN_* env.
        return;
    }

    compile_fatbins();
}

fn compile_fatbins() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR set by cargo"));
    let cuda_dir = Path::new("cuda");

    let nvcc = find_nvcc();

    let gencode = build_gencode_flags(&nvcc);
    if gencode.is_empty() {
        panic!(
            "{} could not report its architectures, so none of {TARGET_ARCHES:?} \
             could be targeted. Either it is not a working nvcc, or it is not on \
             PATH - set CUDA_PATH/CUDA_HOME to the toolkit root.",
            nvcc.display()
        );
    }

    let mut cu_files: Vec<PathBuf> = std::fs::read_dir(cuda_dir)
        .expect("cuda/ directory not found")
        .filter_map(|e| {
            let p = e.ok()?.path();
            (p.extension().and_then(|x| x.to_str()) == Some("cu")).then_some(p)
        })
        .collect();
    // Deterministic order so the build is reproducible.
    cu_files.sort();

    for cu in &cu_files {
        let stem = cu
            .file_stem()
            .and_then(|s| s.to_str())
            .expect("kernel source path has a UTF-8 stem");
        let fatbin = out_dir.join(format!("{stem}.fatbin"));
        println!("cargo:rerun-if-changed={}", cu.display());

        let mut cmd = Command::new(&nvcc);
        cmd.arg("--fatbin")
            .arg("-O3")
            // Precise math: forensics parity (GPU == CPU) matters more than the
            // last few % - no --use_fast_math (the reference used it; we don't).
            // `--fmad=false` disables FMA contraction so a device f64 sum is
            // bit-identical to the Rust reference's separate mul+add - the
            // per-block reduction kernels (edge_sharpness, texture, ...) rely on
            // this so the same blocks flag on GPU and CPU, not just close ones.
            .arg("--fmad=false")
            .args(&gencode)
            .arg(format!("-I{}", cuda_dir.display()))
            .arg("-o")
            .arg(&fatbin)
            .arg(cu);

        let status = cmd
            .status()
            .unwrap_or_else(|e| panic!("failed to run nvcc for {}: {e}", cu.display()));
        if !status.success() {
            panic!("nvcc --fatbin failed for {}", cu.display());
        }

        // The Rust side does include_bytes!(env!("FORENSICS_FATBIN_ELA")) etc.
        println!(
            "cargo:rustc-env=FORENSICS_FATBIN_{}={}",
            stem.to_uppercase(),
            fatbin.display()
        );
    }
}

/// Assemble `-gencode` flags for every TARGET_ARCH the toolkit supports, plus
/// the `a`-suffixed feature targets paddock uses for Blackwell, plus one PTX
/// fallback. Queries `nvcc --list-gpu-arch` so an unknown arch is skipped
/// rather than failing the build - the shell twin's behaviour.
/// Locate nvcc: `CUDA_PATH`/`CUDA_HOME` first, then the platform default, then
/// bare `nvcc` off PATH.
///
/// The extension is load-bearing on Windows. This used to join `"bin/nvcc"` and
/// test `exists()`, which is false next to a perfectly good `nvcc.exe` - so the
/// `cuda` feature could never build on Windows at all, and it failed with
/// "nvcc not found at <dir that plainly contains it>". That is why the reference
/// implementation has only ever run its CPU path on Windows.
///
/// The PATH fallback matches `packs/cuda/build.ps1`, which calls bare `nvcc` on
/// purpose so the machine's elected toolkit wins: `CUDA_PATH` is inherited per
/// process, and an older shell can carry a stale one long after the machine
/// moved to a newer toolkit. A stale variable should not be fatal when a
/// working nvcc is one lookup away.
fn find_nvcc() -> PathBuf {
    let exe = if cfg!(windows) { "nvcc.exe" } else { "nvcc" };
    let roots = [env::var_os("CUDA_PATH"), env::var_os("CUDA_HOME")];
    for root in roots.into_iter().flatten() {
        let p = Path::new(&root).join("bin").join(exe);
        if p.exists() {
            return p;
        }
    }
    let p = Path::new("/usr/local/cuda").join("bin").join(exe);
    if p.exists() {
        return p;
    }
    PathBuf::from(exe)
}

fn build_gencode_flags(nvcc: &Path) -> Vec<String> {
    let listed = Command::new(nvcc)
        .arg("--list-gpu-arch")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).into_owned())
        .unwrap_or_default();
    // Lines look like "compute_86"; collect the numeric suffixes.
    let supported: Vec<String> = listed
        .lines()
        .filter_map(|l| l.trim().strip_prefix("compute_").map(str::to_string))
        .collect();
    let is_supported = |a: &str| supported.iter().any(|s| s == a);

    let mut flags = Vec::new();
    for &a in TARGET_ARCHES {
        if !is_supported(a) {
            println!("cargo:warning=paddock-forensics: toolkit does not support sm_{a}, skipping");
            continue;
        }
        flags.push(format!("-gencode=arch=compute_{a},code=sm_{a}"));
        // Feature targets - same rationale as packs/cuda/build.sh. The 'a'
        // targets carry the Blackwell-specific MMA features; forensics kernels
        // don't use them today, but compiling the feature target keeps us in
        // lockstep with the pack's arch policy and is free for these kernels.
        if a == "120" {
            flags.push("-gencode=arch=compute_120a,code=sm_120a".to_string());
        }
        if a == "100" {
            flags.push("-gencode=arch=compute_100a,code=sm_100a".to_string());
        }
    }
    // PTX fallback for a future/unseen arch (driver JITs it).
    if is_supported(PTX_FALLBACK_ARCH) {
        flags.push(format!(
            "-gencode=arch=compute_{PTX_FALLBACK_ARCH},code=compute_{PTX_FALLBACK_ARCH}"
        ));
    }
    flags
}
