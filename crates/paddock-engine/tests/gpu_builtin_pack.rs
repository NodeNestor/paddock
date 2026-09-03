//! The kernels welded into this binary actually run.
//!
//! Only exists under `static-pack`, and it is the one check that linkage
//! cannot be reasoned about. Everything else about static linking is ordinary
//! - the ABI is unchanged, the table is the same table - but device code in an
//!   ARCHIVE registers itself through `__cudaRegisterFatBinary` in a static
//!   initializer, and a static initializer only runs if the linker pulled its
//!   object in. Drop the object and every launch comes back
//!   `cudaErrorNoKernelImageForDevice`, which reads exactly like an unsupported
//!   GPU. Somebody would then go hunting through gpu/arch.rs for a gate bug that
//!   is not there.
//!
//! `GpuExecutor::with_pack(.., None)` ends in `preflight`, whose trial launch
//! of the elementwise add is precisely the proof: a kernel from the archive
//! reached the device and came back. So this test is one line of intent and a
//! paragraph of why.
//!
//! Gated: skips when there is no CUDA device, like every other GPU suite here.

#![cfg(feature = "static-pack")]

mod common;

#[test]
fn the_linked_in_kernels_launch_on_this_device() {
    if common::cuda().is_none() {
        return;
    }
    let exec = match paddock_engine::gpu::GpuExecutor::with_pack(0, None) {
        Ok(e) => e,
        // An unvalidated arch refuses before the trial launch, which says
        // nothing either way about registration - that is the allowlist doing
        // its job, not a linkage failure. Anything else is a real result.
        Err(e) if e.to_string().contains("has not validated") => {
            eprintln!("skipping: {e}");
            return;
        }
        Err(e) => panic!("built-in kernels did not come up: {e}"),
    };
    // Named, so a failure report says which card it was proved on.
    let (maj, min) = exec.compute_capability();
    eprintln!("built-in pack launched on sm_{maj}{min}");
    assert!(
        exec.pack_is_builtin(),
        "with_pack(None) must use the linked-in kernels"
    );
}
