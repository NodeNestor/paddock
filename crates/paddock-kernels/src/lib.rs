//! Kernel packs: offline-compiled kernel libraries (CUDA first) loaded at
//! runtime behind a stable C ABI.
//!
//! Why packs instead of compiled-in kernels: hardware-matched downloads keep the
//! app binary small (the Unsloth/LM Studio distribution model, see research), and
//! the ABI boundary is what makes kernels swappable per-arch without rebuilding
//! the engine. Kernel policy: everything in-house, zero copied code.

//! CUDA's own redistributables used to live here as `cuda_runtime`, then in a
//! `paddock-cuda-runtime` crate of their own. Both are gone:
//! paddock ships no NVIDIA redistributable and loads none, so
//! there is nothing to find on disk. A pack is self-contained - nvcc links
//! cudart statically, and `pd-cuda-sm86.dll` imports KERNEL32 and nothing else.

pub mod abi;
pub mod loader;
pub mod reference;

pub use abi::{PACK_ABI_VERSION, PackInfo};
pub use loader::{KernelPack, PackError};
