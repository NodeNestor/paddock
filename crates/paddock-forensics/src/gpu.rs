//! GPU context for forensics: a self-owned cudarc context + a **dedicated side
//! stream**, with the forensic kernels loaded from a multi-arch fatbin.
//!
//! This is deliberately independent of `paddock-engine`: forensics never enters
//! the LLM `KernelTableV1`. The fatbin (built by `build.rs`) is `include_bytes!`'d
//! and handed to cudarc via `Ptx::from_binary` (`cuModuleLoadData`) - real SASS
//! on every supported arch, no runtime JIT, no libnvrtc call.
//!
//! Without the `cuda` feature, [`ForensicGpu`] is an uninhabited type so the
//! rest of the crate compiles unchanged and every caller simply passes `None`.

#[cfg(not(feature = "cuda"))]
/// Uninhabited placeholder when built without the `cuda` feature.
pub enum ForensicGpu {}

#[cfg(feature = "cuda")]
pub use cuda_impl::{ForensicGpu, GpuError};

#[cfg(feature = "cuda")]
mod cuda_impl {
    use std::collections::HashMap;
    use std::sync::Arc;

    use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaStream};
    use cudarc::nvrtc::Ptx;
    use thiserror::Error;

    /// The forensic kernel fatbins, welded in at compile time. `build.rs` emits
    /// one `FORENSICS_FATBIN_<STEM>` per `cuda/*.cu`.
    const ELA_FATBIN: &[u8] = include_bytes!(env!("FORENSICS_FATBIN_ELA"));
    const NOISE_FATBIN: &[u8] = include_bytes!(env!("FORENSICS_FATBIN_NOISE"));
    const JPEG_GHOST_FATBIN: &[u8] = include_bytes!(env!("FORENSICS_FATBIN_JPEG_GHOST"));
    const EDGE_SHARPNESS_FATBIN: &[u8] = include_bytes!(env!("FORENSICS_FATBIN_EDGE_SHARPNESS"));
    const CHANNEL_CORRELATION_FATBIN: &[u8] =
        include_bytes!(env!("FORENSICS_FATBIN_CHANNEL_CORRELATION"));
    const WAVELET_CONSISTENCY_FATBIN: &[u8] =
        include_bytes!(env!("FORENSICS_FATBIN_WAVELET_CONSISTENCY"));
    const TEXTURE_FATBIN: &[u8] = include_bytes!(env!("FORENSICS_FATBIN_TEXTURE"));
    const COLOR_CONSISTENCY_FATBIN: &[u8] =
        include_bytes!(env!("FORENSICS_FATBIN_COLOR_CONSISTENCY"));
    const ANTI_FORENSICS_FATBIN: &[u8] = include_bytes!(env!("FORENSICS_FATBIN_ANTI_FORENSICS"));
    const LIGHTING_CONSISTENCY_FATBIN: &[u8] =
        include_bytes!(env!("FORENSICS_FATBIN_LIGHTING_CONSISTENCY"));

    #[derive(Debug, Error)]
    pub enum GpuError {
        #[error("CUDA driver error: {0}")]
        Driver(#[from] cudarc::driver::DriverError),
        #[error("kernel not found: {0}")]
        KernelNotFound(String),
        #[error("GPU forensics: {0}")]
        Other(String),
    }

    /// A forensic GPU context. Cheap to hold; build one per serving runner when
    /// `[forensics] enabled` and hand `Some(&gpu)` to [`crate::run`].
    pub struct ForensicGpu {
        #[allow(dead_code)]
        ctx: Arc<CudaContext>,
        /// Dedicated non-default stream so forensic kernels overlap LLM work
        /// rather than blocking the engine's default stream.
        stream: Arc<CudaStream>,
        modules: HashMap<&'static str, Arc<CudaModule>>,
    }

    impl ForensicGpu {
        /// Initialize on the given device ordinal, loading every forensic module.
        pub fn new(device_ordinal: usize) -> Result<Self, GpuError> {
            let ctx = CudaContext::new(device_ordinal)?;
            // A fresh stream (not ctx.default_stream()) - forensics runs on its
            // own stream to overlap with, never serialize behind, the engine.
            let stream = ctx.new_stream()?;

            let mut modules = HashMap::new();
            modules.insert(
                "ela",
                ctx.load_module(Ptx::from_binary(ELA_FATBIN.to_vec()))?,
            );
            modules.insert(
                "noise",
                ctx.load_module(Ptx::from_binary(NOISE_FATBIN.to_vec()))?,
            );
            modules.insert(
                "jpeg_ghost",
                ctx.load_module(Ptx::from_binary(JPEG_GHOST_FATBIN.to_vec()))?,
            );
            modules.insert(
                "edge_sharpness",
                ctx.load_module(Ptx::from_binary(EDGE_SHARPNESS_FATBIN.to_vec()))?,
            );
            modules.insert(
                "channel_correlation",
                ctx.load_module(Ptx::from_binary(CHANNEL_CORRELATION_FATBIN.to_vec()))?,
            );
            modules.insert(
                "wavelet_consistency",
                ctx.load_module(Ptx::from_binary(WAVELET_CONSISTENCY_FATBIN.to_vec()))?,
            );
            modules.insert(
                "texture",
                ctx.load_module(Ptx::from_binary(TEXTURE_FATBIN.to_vec()))?,
            );
            modules.insert(
                "color_consistency",
                ctx.load_module(Ptx::from_binary(COLOR_CONSISTENCY_FATBIN.to_vec()))?,
            );
            modules.insert(
                "anti_forensics",
                ctx.load_module(Ptx::from_binary(ANTI_FORENSICS_FATBIN.to_vec()))?,
            );
            modules.insert(
                "lighting_consistency",
                ctx.load_module(Ptx::from_binary(LIGHTING_CONSISTENCY_FATBIN.to_vec()))?,
            );

            Ok(Self {
                ctx,
                stream,
                modules,
            })
        }

        /// The dedicated forensic stream.
        pub fn stream(&self) -> &Arc<CudaStream> {
            &self.stream
        }

        /// Look up a kernel by module and function name.
        pub fn function(&self, module: &str, function: &str) -> Result<CudaFunction, GpuError> {
            let m = self
                .modules
                .get(module)
                .ok_or_else(|| GpuError::KernelNotFound(format!("module `{module}` not loaded")))?;
            m.load_function(function)
                .map_err(|e| GpuError::KernelNotFound(format!("{module}::{function}: {e}")))
        }
    }
}
