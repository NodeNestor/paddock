//! GPU error type + launch-status helpers.

use paddock_kernels::PackError;
use paddock_models::ggml_type::GgmlType;
use paddock_models::mapped::MapError;

#[derive(Debug, thiserror::Error)]
pub enum GpuError {
    #[error("CUDA driver: {0}")]
    Driver(String),
    /// The driver reported `CUDA_ERROR_OUT_OF_MEMORY` (device result code 2) on
    /// an allocation. Kept as its own variant - Not folded into `Driver(String)`
    /// - so an OOM is classified by the driver's numeric result code at this
    ///   boundary (see [`from_driver`]) and threaded up as a type the whole way,
    ///   instead of being re-recognized later by sniffing the driver's rendered
    ///   text. An OOM during a large prefill (a many-page PDF or a high-detail
    ///   image on a VRAM-constrained or busy device) is a CAPACITY event the
    ///   caller can act on, not an engine bug: the API layer maps it to a
    ///   retryable `Overloaded` naming the levers (fewer/smaller pages), via
    ///   `GenError::OutOfMemory` -> `EngineError::from_gen`.
    #[error("CUDA out of memory (CUDA_ERROR_OUT_OF_MEMORY)")]
    OutOfMemory,
    /// A paged-KV budget pool ran dry (after eviction). The engine maps this
    /// to `GenError::PoolExhausted` so the scheduler can preempt instead of
    /// failing the batch.
    #[error("KV pool exhausted")]
    PoolExhausted,
    #[error("cuBLAS: {0}")]
    Blas(String),
    #[error(transparent)]
    Pack(#[from] PackError),
    #[error(transparent)]
    Map(#[from] MapError),
    #[error("tensor {name}: no dequant kernel for {ty:?} in the loaded pack")]
    NoKernel { name: String, ty: GgmlType },
    #[error("kernel {0} missing from the loaded pack")]
    MissingOp(&'static str),
    #[error("kernel launcher returned CUDA error {0}")]
    Launch(i32),
    /// Preflight rejection: this GPU / pack combination can't serve. Carries
    /// a complete user-facing sentence (device, cc, what to do) - the
    /// no-silent-failures principle applied to bring-up.
    #[error("{0}")]
    Unsupported(String),
}

/// Convert AND CLASSIFY a cudarc driver error - the short name the op modules
/// call by (`.map_err(drv)`). Every real driver call in `gpu/` routes its error
/// here, so an allocation that hits `CUDA_ERROR_OUT_OF_MEMORY` becomes a typed
/// `GpuError::OutOfMemory` rather than an opaque `Driver(String)` the API layer
/// would have to sniff. Same classification as [`from_driver`]; this is the
/// alias the hot paths use.
pub(super) fn drv(e: cudarc::driver::DriverError) -> GpuError {
    from_driver(e)
}

/// A bounds/range refusal that is OURS, not the driver's - a slice view that
/// didn't fit, a host buffer longer than its device target. These guard the
/// driver calls; they are logic errors, never an OOM, so they carry a plain
/// message. (Split out of `drv` when `drv` narrowed to `DriverError` so the OOM
/// code could be read off the driver result - a `&str` has no result code.)
pub(super) fn oob(msg: &str) -> GpuError {
    GpuError::Driver(msg.to_string())
}

/// Classify a cudarc `DriverError` at the cudarc->`GpuError` boundary. This is
/// the one place a driver result becomes an engine error, and it decides OOM by
/// the numeric result code (`CUDA_ERROR_OUT_OF_MEMORY` == 2) rather than by the
/// enum name cudarc happens to render - so the OOM signal survives as a type
/// (`GpuError::OutOfMemory`) up through `GenError::OutOfMemory` to the API
/// funnel, and is not hostage to the driver's `Display` string. Every
/// DriverError->GpuError conversion routes through here (directly or via the
/// `From` impl below); non-OOM faults keep their rendered text for diagnostics.
pub(crate) fn from_driver(e: cudarc::driver::DriverError) -> GpuError {
    if e.0 == cudarc::driver::sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY {
        GpuError::OutOfMemory
    } else {
        GpuError::Driver(e.to_string())
    }
}

impl From<cudarc::driver::DriverError> for GpuError {
    fn from(e: cudarc::driver::DriverError) -> Self {
        from_driver(e)
    }
}

#[track_caller]
pub(super) fn check(status: i32) -> Result<(), GpuError> {
    if status == 0 {
        Ok(())
    } else {
        // GpuError::Launch carries only the rc, so a refusal like
        // 801/NotSupported used to be unattributable - and a release backtrace
        // is empty. #[track_caller] costs nothing on the success path and names
        // the wrapper that launched, which is the question every time.
        let loc = std::panic::Location::caller();
        tracing::warn!(
            status,
            at = format!("{}:{}", loc.file(), loc.line()),
            "kernel launcher refused"
        );
        // diagnostic aid: the Launch error is bare (no op name), so a refusal
        // like 801/NotSupported can't be attributed. Opt-in panic captures the
        // call site in the backtrace. Harmless in production - env-gated.
        if paddock_models::dev_var_os!("PADDOCK_PANIC_LAUNCH").is_some() {
            panic!("kernel launcher returned CUDA error {status} (PADDOCK_PANIC_LAUNCH)");
        }
        Err(GpuError::Launch(status))
    }
}
