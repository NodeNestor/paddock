//! One way in to NVML, because the obvious way is wrong on Linux.
//!
//! `Nvml::init()` loads `libnvidia-ml.so` on Linux - the UNVERSIONED name. The
//! NVIDIA driver installs `libnvidia-ml.so.1`; the bare `.so` is a development
//! symlink that comes with the headers package and is absent from every
//! runtime-only install, which includes every container the NVIDIA Container
//! Toolkit builds.
//!
//! Seen on a Linux release under `--gpus all` with a
//! working A6000: the manager announced "no usable NVIDIA graphics card found
//! - models cannot run on this computer". `nvidia-smi` worked in the same
//!   container, `libnvidia-ml.so.1` was in the loader cache, and adding the one
//!   symlink flipped it to "graphics card supported". So the product told a user
//!   with a supported card that their card did not exist - a confidently wrong
//!   answer, which is worse than the silent failure the principles already ban.
//!
//! Windows is unaffected: there the name is `nvml.dll`, versionless, and
//! present whenever the driver is. Hence a fallback rather than a replacement
//! - the default is right on one platform and right on developer Linux boxes
//!   too, and this only catches the case where it is not.
use nvml_wrapper::{Nvml, error::NvmlError};

/// Initialise NVML, looking under the versioned soname if the default name is
/// not installed.
pub fn init() -> Result<Nvml, NvmlError> {
    match Nvml::init() {
        Ok(n) => Ok(n),
        // Keep the first error to report: it names the library a reader will
        // recognise, and on a box with no driver at all both attempts fail
        // for the same reason anyway.
        Err(first) => {
            for name in FALLBACKS {
                if let Ok(n) = Nvml::builder().lib_path(name.as_ref()).init() {
                    return Ok(n);
                }
            }
            Err(first)
        }
    }
}

#[cfg(unix)]
const FALLBACKS: &[&str] = &["libnvidia-ml.so.1"];

#[cfg(not(unix))]
const FALLBACKS: &[&str] = &[];
