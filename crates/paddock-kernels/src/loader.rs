//! Loads and validates kernel packs.
//!
//! Failure here must be loud and specific (no-silent-failures principle): a user
//! staring at "pack rejected: abi 3, engine expects 0" can fix their install; a
//! user staring at a segfault cannot.
//!
//! Two ways a pack arrives:
//!
//!   - **built in** - the `static-pack` feature links the CUDA archive into
//!     this binary, so the symbols are simply there. This is what a RELEASE
//!     does: the install is two binaries and nothing else beside them.
//!   - **loaded** - `dlopen` a .dll/.so at a path, which is what development
//!     does. Rebuild the pack, restart the runner, no relink; and it is the
//!     only way to run a pack carrying an arch the shipped binary deliberately
//!     omits, which per-generation bring-up needs.
//!
//! The VALIDATION is not REDUNDANT in the BUILT-IN CASE, which is worth saying
//! because it looks it. The archive is still built by a separate `nvcc` run at
//! a separate time - edit `abi.rs`, forget `build.ps1`, and cargo happily links
//! yesterday's `pd-cuda.lib`. The magic/abi check catches a version bump and
//! `table_fit` catches silent table GROWTH (the 3056 -> 3064 class), exactly as
//! they do for a stale .dll. Same protocol, both sources.

use std::path::{Path, PathBuf};

use crate::abi::{
    KernelTableV1, PACK_ABI_VERSION, PACK_INFO_SYMBOL, PACK_KERNELS_V1_SYMBOL, PACK_MAGIC, PackInfo,
};

/// The linked-in pack's exports. Same two symbols the dlopen path resolves by
/// name - `PACK_INFO_SYMBOL` / `PACK_KERNELS_V1_SYMBOL` are these spellings.
#[cfg(feature = "static-pack")]
mod builtin {
    use super::{KernelTableV1, PackInfo};
    unsafe extern "C" {
        pub fn paddock_pack_info() -> *const PackInfo;
        pub fn paddock_pack_kernels_v1() -> *const KernelTableV1;
    }
}

#[derive(Debug, thiserror::Error)]
pub enum PackError {
    #[error("failed to open kernel pack {path}: {source}")]
    Open {
        path: PathBuf,
        source: libloading::Error,
    },
    #[error("{path} is not a Paddock kernel pack (missing {symbol} symbol)")]
    NotAPack { path: PathBuf, symbol: String },
    #[error("{path} returned a null/invalid descriptor")]
    BadDescriptor { path: PathBuf },
    #[error("{path} has wrong magic - not a pack or corrupted download")]
    BadMagic { path: PathBuf },
    #[error("{path} was built for pack ABI {found}, this engine expects {expected}")]
    AbiMismatch {
        path: PathBuf,
        found: u32,
        expected: u32,
    },
    #[error("{path} exports no kernel table (missing {symbol})")]
    NoKernelTable { path: PathBuf, symbol: String },
    #[error("{path} kernel table is null or truncated (size {size})")]
    BadKernelTable { path: PathBuf, size: u32 },
}

/// How much of this build's kernel table the loaded pack actually fills.
///
/// The table grows append-only, so a pack older than the engine simply
/// declares fewer bytes and every entry past that reads as `None`. That is
/// the designed behaviour and it is safe - but it is also INVISIBLE until
/// some request happens to need one of the missing entries, and then it
/// surfaces as a kernel name in front of whoever asked. A pack months out of
/// date shipped in a portable install exactly that way and announced itself
/// as "kernel whisper_xattn_probs missing from the loaded pack".
#[derive(Debug, Clone, Copy)]
pub struct TableFit {
    /// bytes the pack says it filled
    pub declared: usize,
    /// bytes this build's table has
    pub expected: usize,
}

impl TableFit {
    /// Entries this pack cannot answer for. Pointer-sized because the table
    /// past the header is all `Option<fn>` - an approximation only if the
    /// gap ever straddles the header, which it cannot.
    pub fn missing_entries(&self) -> usize {
        self.expected.saturating_sub(self.declared) / std::mem::size_of::<usize>()
    }
    /// The pack predates this build.
    pub fn is_stale(&self) -> bool {
        self.declared < self.expected
    }
}

/// Where a pack's symbols came from. Not public: callers care about the
/// kernels, and every message that needs to name the source goes through
/// `origin()`.
enum Source {
    /// Linked into this binary at build time (`static-pack`).
    #[cfg(feature = "static-pack")]
    Builtin,
    /// Held for the lifetime of the pack - dropping it unloads the library.
    Loaded(libloading::Library),
}

/// A validated kernel pack. Kernel entry points get resolved lazily by
/// the engine as the P1 kernel table lands.
pub struct KernelPack {
    info: PackInfo,
    source: Source,
    /// What to call this in an error. A real path when loaded, a marker when
    /// built in - the error type predates the built-in case and its variants
    /// all carry a `PathBuf`, so this keeps the messages readable without
    /// splitting every variant in two.
    origin: PathBuf,
}

impl std::fmt::Debug for KernelPack {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KernelPack")
            .field("info", &self.info)
            .field("origin", &self.origin)
            .finish_non_exhaustive()
    }
}

impl KernelPack {
    /// The kernels compiled into this binary. Only exists under `static-pack`;
    /// the engine picks it when no pack path was configured.
    #[cfg(feature = "static-pack")]
    pub fn builtin() -> Result<Self, PackError> {
        let origin = PathBuf::from("<built-in kernels>");
        // SAFETY: linked in from our own archive; the contract is that this
        // returns a pointer to a static PackInfo with 'static lifetime.
        let info_ptr = unsafe { builtin::paddock_pack_info() };
        Self::finish(info_ptr, Source::Builtin, origin)
    }

    pub fn load(path: &Path) -> Result<Self, PackError> {
        // SAFETY: loading a library runs its init sections; we only load packs the
        // user configured or that our downloader hash-verified. That trust boundary
        // is documented in the plan (packs are versioned + SHA-256 checked).
        let lib = unsafe { libloading::Library::new(path) }.map_err(|source| PackError::Open {
            path: path.to_path_buf(),
            source,
        })?;

        type InfoFn = unsafe extern "C" fn() -> *const PackInfo;
        // SAFETY: symbol type is part of the ABI contract keyed by PACK_INFO_SYMBOL
        let info_fn =
            unsafe { lib.get::<InfoFn>(PACK_INFO_SYMBOL) }.map_err(|_| PackError::NotAPack {
                path: path.to_path_buf(),
                symbol: sym(PACK_INFO_SYMBOL),
            })?;

        // SAFETY: contract says the pointer is to a static PackInfo inside the lib
        let info_ptr = unsafe { info_fn() };
        Self::finish(info_ptr, Source::Loaded(lib), path.to_path_buf())
    }

    /// The checks both sources share: non-null descriptor, magic, ABI version.
    fn finish(
        info_ptr: *const PackInfo,
        source: Source,
        origin: PathBuf,
    ) -> Result<Self, PackError> {
        if info_ptr.is_null() {
            return Err(PackError::BadDescriptor { path: origin });
        }
        // SAFETY: non-null, repr(C), the pack keeps it alive as long as we live
        let info = unsafe { *info_ptr };

        if info.magic != PACK_MAGIC {
            return Err(PackError::BadMagic { path: origin });
        }
        if info.abi_version != PACK_ABI_VERSION {
            return Err(PackError::AbiMismatch {
                path: origin,
                found: info.abi_version,
                expected: PACK_ABI_VERSION,
            });
        }
        Ok(Self {
            info,
            source,
            origin,
        })
    }

    pub fn info(&self) -> &PackInfo {
        &self.info
    }

    /// What to call this pack in a message - a path, or the built-in marker.
    pub fn origin(&self) -> &Path {
        &self.origin
    }

    /// Whether the kernels are welded into this binary rather than loaded.
    pub fn is_builtin(&self) -> bool {
        #[cfg(feature = "static-pack")]
        if matches!(self.source, Source::Builtin) {
            return true;
        }
        false
    }

    /// Resolve the exported table. Both sources land on the same pointer
    /// contract, so everything downstream of here is shared.
    fn table_ptr(&self) -> Result<*const KernelTableV1, PackError> {
        type TableFn = unsafe extern "C" fn() -> *const KernelTableV1;
        let ptr = match &self.source {
            #[cfg(feature = "static-pack")]
            // SAFETY: linked in from our own archive, same contract as below
            Source::Builtin => unsafe { builtin::paddock_pack_kernels_v1() },
            Source::Loaded(lib) => {
                // SAFETY: symbol type is part of the versioned ABI contract
                let f = unsafe { lib.get::<TableFn>(PACK_KERNELS_V1_SYMBOL) }.map_err(|_| {
                    PackError::NoKernelTable {
                        path: self.origin.clone(),
                        symbol: sym(PACK_KERNELS_V1_SYMBOL),
                    }
                })?;
                // SAFETY: contract - pointer to a static table living inside the lib
                unsafe { f() }
            }
        };
        if ptr.is_null() {
            return Err(PackError::BadKernelTable {
                path: self.origin.clone(),
                size: 0,
            });
        }
        Ok(ptr)
    }

    /// How much of this build's table the pack fills - the append-only
    /// contract's staleness, made answerable before anything needs a kernel.
    ///
    /// Runs for the built-in pack too: the archive is a separate nvcc run, so
    /// "linked in" does not mean "same vintage as abi.rs" (see the module
    /// header).
    pub fn table_fit(&self) -> Result<TableFit, PackError> {
        let ptr = self.table_ptr()?;
        // SAFETY: header (size + reserved) is always present per contract
        let declared = unsafe { *(ptr as *const u32) } as usize;
        Ok(TableFit {
            declared,
            expected: std::mem::size_of::<KernelTableV1>(),
        })
    }

    /// Resolve the v1 kernel table. Append-only growth contract: we copy
    /// min(pack's size, our size) bytes into a zeroed table, so entries the
    /// pack predates read as None (Option<fn> null niche) - the engine reports
    /// "kernel X not in pack Y", never a null-pointer jump.
    pub fn kernels_v1(&self) -> Result<KernelTableV1, PackError> {
        let ptr = self.table_ptr()?;
        // SAFETY: header (size + reserved) is always present per contract
        let declared = unsafe { *(ptr as *const u32) } as usize;
        if declared < 8 {
            return Err(PackError::BadKernelTable {
                path: self.origin.clone(),
                size: declared as u32,
            });
        }
        // SAFETY: zeroed KernelTableV1 is valid (u32s zero, Option<fn> None via
        // null niche); we copy only bytes the pack declares it filled
        let mut table: KernelTableV1 = unsafe { std::mem::zeroed() };
        let copy = declared.min(std::mem::size_of::<KernelTableV1>());
        // SAFETY: src has `declared` valid bytes per contract; dst sized above
        unsafe {
            std::ptr::copy_nonoverlapping(
                ptr as *const u8,
                &mut table as *mut KernelTableV1 as *mut u8,
                copy,
            );
        }
        Ok(table)
    }
}

/// A NUL-terminated ABI symbol name, as something printable.
fn sym(bytes: &[u8]) -> String {
    String::from_utf8_lossy(&bytes[..bytes.len() - 1]).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loading_a_non_library_fails_with_open_error() {
        let err =
            KernelPack::load(Path::new("Z:/definitely/not/a/pack.dll")).expect_err("must fail");
        assert!(matches!(err, PackError::Open { .. }));
    }

    /// The built-in pack has to pass the same magic/ABI/table checks a loaded
    /// one does - that is what catches a `pd-cuda.lib` built before the last
    /// `abi.rs` edit, which static linking does not prevent on its own.
    #[cfg(feature = "static-pack")]
    #[test]
    fn the_builtin_pack_validates_and_fills_the_whole_table() {
        let pack = KernelPack::builtin().expect("built-in pack must validate");
        assert!(pack.is_builtin());
        let fit = pack.table_fit().expect("table");
        assert!(
            !fit.is_stale(),
            "pd-cuda.lib is older than this build: {} of {} entries absent - rebuild the pack",
            fit.missing_entries(),
            fit.expected / std::mem::size_of::<usize>(),
        );
    }
}
