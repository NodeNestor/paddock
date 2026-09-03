//! The backend seam.
//!
//! A Backend is a device execution target for our engine (CUDA now; CPU, Metal,
//! Vulkan later) - explicitly not a slot for third-party engines. External
//! providers (OpenAI/Anthropic proxying) will get their own seam at the router
//! level, not here.
//!
//! Deliberately narrow in P0: just identity/health. The generate/cache surface
//! gets designed in the architecture doc with the P1 forward pass, where real
//! usage can push back on the shape - committing to a session API before we've
//! run a model would just bake in guesses.

/// Static facts about an execution backend.
#[derive(Debug, Clone)]
pub struct BackendInfo {
    /// e.g. "cuda"
    pub name: &'static str,
    /// e.g. "NVIDIA RTX A6000" once the device is probed
    pub device: String,
    /// total device memory in bytes; the estimator's starting point
    pub memory_total: u64,
}

pub trait Backend: Send + Sync {
    fn info(&self) -> BackendInfo;

    /// Cheap liveness check - device still responsive, context not poisoned.
    fn healthy(&self) -> bool;
}
