//! Engine-elected env defaults that the KERNEL PACK must also see.
//!
//! The pack's launchers read election/kill envs with C `getenv`. On Windows,
//! Rust's `std::env::set_var` writes the Win32 environment only - the UCRT
//! keeps its own copy, snapshotted at process start, and the pack DLL's
//! `getenv` reads THAT. So every default the engine elected at model load was
//! invisible to the pack on Windows: env-gated arms silently ran their
//! fallbacks, and the first launcher that REFUSES instead of falling back
//! (the DNC varlen rs route, engine gate true / pack gate false) surfaced it
//! as a bare CUDA 801 on every >=128-row qwen35 prefill span.
//!
//! `_putenv_s` updates the UCRT copy; both the exe and the nvcc-built pack
//! link the DYNAMIC ucrt (no crt-static in this workspace), so there is one
//! copy per process and both worlds agree after this. On non-Windows,
//! setenv/getenv already share one environ and set_var alone suffices.

/// Truthy read for election/kill envs: set, non-empty, and not `"0"`.
///
/// The house contract for engine-elected defaults is "the env always wins,
/// `FOO=0` reverts" (see gemma4's LIN_KTZ). A bare presence check can't honor
/// that once `set_env` fills a default - there is no way left to say off - so
/// every env that can be defaulted must read through this instead of
/// `var_os(..).is_some()`. The pack-side twin is `pd_env_on` in abi.cuh.
pub fn env_on(key: &str) -> bool {
    std::env::var_os(key).is_some_and(|v| !v.is_empty() && v != "0")
}

/// Set an env var so both Rust readers and the pack's C `getenv` see it.
///
/// Same contract as the raw `set_var` calls this replaces: call at model
/// load, before serving threads spawn.
pub fn set_env(key: &str, value: &str) {
    unsafe { std::env::set_var(key, value) };
    #[cfg(windows)]
    {
        unsafe extern "C" {
            fn _putenv_s(k: *const std::os::raw::c_char, v: *const std::os::raw::c_char) -> i32;
        }
        let (Ok(k), Ok(v)) = (std::ffi::CString::new(key), std::ffi::CString::new(value)) else {
            return; // interior NUL can't be a real env pair
        };
        unsafe { _putenv_s(k.as_ptr(), v.as_ptr()) };
    }
}
