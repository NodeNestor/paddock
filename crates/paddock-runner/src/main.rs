//! `paddock-runner` - the data-plane binary. One model, one port, the full
//! inference API, until stopped. Headless: the Studio, catalog, and telemetry
//! UI live in the manager (`paddock`), which spawns and supervises runners.
//!
//! Config precedence: CLI flags > PADDOCK_* env > paddock.toml > defaults.
//! All the wiring (clap, layering, banner) lives in `startup`.

// The statically-linked pdfium ships Chromium's PartitionAlloc, whose
// allocator_shim takes over the process malloc/free. Two problems
// with serving through it: any single host allocation >= 2 GiB int3-traps the
// process (PA's internal guard), and its SpinningMutex serializes the hot
// serving path - a perf profile of a 32-way serving run put
// native_queued_spin_lock_slowpath at 20.8% of host CPU on the tokio workers
// plus ~14% inside PartitionAlloc Malloc/Free/AcquireSpinThenBlock. Routing
// Rust's global allocator to mimalloc bypasses malloc for every Rust
// allocation (tokenizers, serde, SSE streaming, tick assembly), which removes
// both the contention and the 2 GiB trap from the Rust side; pdfium's own
// internal C++ allocations keep using PA, which is what it was built for.
// The full shim removal (use_partition_alloc_as_malloc=false) stays.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn main() -> std::process::ExitCode {
    paddock_runner::startup::run()
}
