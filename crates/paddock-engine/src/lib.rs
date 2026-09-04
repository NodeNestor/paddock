//! The native inference engine.
//!
//! Design contracts:
//! - engine core owns its own loop and all mutable state (vLLM-V1 pattern);
//!   the server talks to it over channels only
//! - KV allocator supports per-layer cache kinds from day one:
//!   paged full-attention / sliding-window ring / recurrent state slots
//! - kernels come from versioned kernel packs (paddock-kernels), all in-house
//!
//! The first cut ships the skeleton: the backend trait and placeholder types,
//! so the server and harness can compile against stable seams while the
//! implementations fill in behind them.
pub mod align;
pub mod audio;
pub mod backend;
pub mod cuda;
pub mod encoder;
pub mod envset;
pub mod generator;
pub mod gpu;
pub mod gpu_model;
pub mod kv_plan;
pub mod kv_pool;
pub mod kv_tier;
pub mod metrics;
pub mod paged_radix;
pub mod reference;
pub mod sampler;
pub mod service;
pub mod spec;
pub mod spec_policy;
pub mod tickseg;
pub mod transcriber;

pub use backend::{Backend, BackendInfo};
