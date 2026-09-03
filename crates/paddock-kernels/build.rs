//! Links the CUDA kernel pack into the binary, when asked.
//!
//! Inert unless the `static-pack` feature is on, which is the whole point: a
//! default `cargo build` behaves exactly as it did before this file existed -
//! no nvcc, no .lib on disk, no GPU needed. That keeps GPU-less CI building,
//! while a RELEASE build (`--features static-pack`) welds the kernels in so a
//! user's install is two binaries and nothing else.
//!
//! The .lib is a BUILD INPUT, never a shipped artifact. There is no download
//! for it and there should not be: it is a ~3-minute `packs/cuda/build.ps1`
//! run, and anyone building paddock-runner already has nvcc because they
//! already build the pack. (pdfium is fetched from R2 because building that
//! means depot_tools and a Chromium toolchain - the analogy does not carry.)

use std::path::PathBuf;

fn main() {
    // Both are read even when the feature is off, so toggling it re-runs us.
    println!("cargo:rerun-if-env-changed=PADDOCK_PACK_LIB");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");

    if std::env::var_os("CARGO_FEATURE_STATIC_PACK").is_none() {
        return;
    }

    let dir = pack_dir();
    // A missing .lib is the common first failure here, and rustc's own
    // "could not find native static library" says nothing about how to make
    // one. Say it here instead, with the command.
    let stem = if cfg!(target_env = "msvc") {
        "pd-cuda.lib"
    } else {
        "libpd-cuda.a"
    };
    if !dir.join(stem).exists() {
        panic!(
            "static-pack is on but {} is not there.\n\
             Build it first:  {}\n\
             Or point PADDOCK_PACK_LIB at the directory holding it.",
            dir.join(stem).display(),
            if cfg!(windows) {
                "powershell -File packs\\cuda\\build.ps1 -Static"
            } else {
                "packs/cuda/build.sh --static"
            }
        );
    }
    println!("cargo:rustc-link-search=native={}", dir.display());
    println!("cargo:rustc-link-lib=static=pd-cuda");
    println!("cargo:rerun-if-changed={}", dir.join(stem).display());

    // The pack calls into the CUDA RUNTIME (cudaGetDevice/cudaLaunchKernel and
    // friends). As a shared object nvcc linked that in for us; as an archive it
    // is our job. Static, not shared, so nothing lands beside the exe - the
    // point of the exercise. Note this is the runtime API, entirely separate
    // from the DRIVER API cudarc opens by name (nvcuda.dll / libcuda.so, which
    // belongs to the display driver). Both in one process is normal.
    println!(
        "cargo:rustc-link-search=native={}",
        cuda_lib_dir().display()
    );
    println!("cargo:rustc-link-lib=static=cudart_static");

    if cfg!(target_env = "msvc") {
        // cudart_static's own imports. Missing ones surface as unresolved
        // externals at link time - loud and trivially fixable, which is why
        // this list is deliberately minimal rather than speculative.
        for l in ["kernel32", "user32", "advapi32"] {
            println!("cargo:rustc-link-lib=dylib={l}");
        }
    } else {
        for l in ["dl", "rt", "pthread", "stdc++"] {
            println!("cargo:rustc-link-lib=dylib={l}");
        }
    }
}

/// Where the pack build script leaves its artifacts.
fn pack_dir() -> PathBuf {
    if let Some(p) = std::env::var_os("PADDOCK_PACK_LIB") {
        return PathBuf::from(p);
    }
    // crates/paddock-kernels -> repo root -> packs/cuda/build
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../packs/cuda/build")
}

/// The toolkit's link libraries. CUDA_PATH is set by the Windows installer;
/// CUDA_HOME then the conventional prefix elsewhere.
fn cuda_lib_dir() -> PathBuf {
    let root = std::env::var_os("CUDA_PATH")
        .or_else(|| std::env::var_os("CUDA_HOME"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/usr/local/cuda"));
    if cfg!(windows) {
        root.join("lib/x64")
    } else {
        root.join("lib64")
    }
}
