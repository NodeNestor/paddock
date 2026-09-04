//! NVIDIA Nemotron 3.5 Lightning 30B-A3B (`nemotron_h_moe`) - the first
//! safetensors-primary family: the NVFP4 checkpoint is served directly, no
//! GGUF anywhere in the lane.
//!
//! Architecture (pinned from the vLLM and llama.cpp study references plus
//! the checkpoint itself): 52
//! homogeneous single-residual blocks - each layer is one mixer,
//! `x = x + mixer(rms_norm(x))` - interleaving 23 mamba-2, 23 MoE and 6
//! attention layers (5, 12, 19, 26, 33, 42). NoPE attention (no rotary at
//! all), 32 Q / 2 KV heads at head_dim 128. Final RMS norm into an NVFP4
//! lm_head (untied).
//!
//! Weight residencies (byte-exact to the checkpoint everywhere):
//!   - experts + shared expert + lm_head: NVFP4 triples served packed
//!     (W4A16-class; the dequant kernel is oracle-only)
//!   - mamba in/out_proj: checkpoint e4m3 bytes as F8RowPlanes, the
//!     per-tensor weight_scale broadcast into the row-scale array
//!   - attention q/k/v/o: f32 planes (exact widening) for prefill + checkpoint
//!     bf16 twins for the decode GEMVs; embeddings: checkpoint bf16 (the
//!     gather widens in-kernel); routers, norms, conv, A/D/dt_bias:
//!     bf16/f32 -> f32 (exact widening)

pub(crate) mod batch;
pub(crate) mod dflash;
mod forward;
mod load;
pub(crate) mod mtp;
mod prefix;
mod spec;
pub(crate) mod ssm_arena;

use std::sync::Arc;

use cudarc::driver::CudaSlice;

use crate::gpu::{
    DeviceTensor, F8RowPlane, GpuExecutor, KvDtype, Nvf4MoePlane, Nvf4Plane, QuantTensor, QuantW,
    RepackedQ8,
};
use paddock_models::nemotron::NemotronConfig;

pub(crate) use forward::{DecodeState, PipeState, PrefillScratch, Scratch, SendGraph};

/// One linear plane in either weight class: the NVFP4 checkpoint's fp8
/// e4m3 rows (W8A8 prefill / W8A16 decode), or a GGUF quant plane (Q8_0
/// int8 dp4a class; k-quant loads too but the MoE planes refuse non-Q8_0
/// upstream, so a UD file errors at load rather than serving half-classed).
pub(crate) enum LinW {
    F8(F8RowPlane),
    Qw(QuantW),
}

/// Mamba-2 mixer weights. `in_proj` rows are `[z | x B C | dt]` (d_inner +
/// conv_dim + n_heads = 10304); the forward slices that output by offset
/// rather than splitting.
pub(crate) struct MambaWeights {
    pub in_proj: LinW,
    pub out_proj: LinW,
    /// conv weight [conv_dim, k] (checkpoint [conv_dim, 1, k] squeezed)
    pub conv_w: CudaSlice<f32>,
    pub conv_b: CudaSlice<f32>,
    /// -exp(A_log), per head (the load-time transform vLLM also applies)
    pub a: CudaSlice<f32>,
    pub d: CudaSlice<f32>,
    pub dt_bias: CudaSlice<f32>,
    /// gated-norm weight [d_inner]
    pub norm_w: CudaSlice<f32>,
}

/// NoPE attention mixer weights, in either weight class.
///
/// `F32` is the NVFP4 lane: f32 planes (bf16 widened exactly) carry prefill
/// byte-identical to the original lane; `bf16` holds the checkpoint bytes
/// for the decode GEMVs (half the DRAM per tick), None when the pack lacks
/// the lane or PADDOCK_NO_NEMO_BF16 pins the f32 baseline.
///
/// `Qw` is the GGUF lane: one quant plane per projection (Q8_0 - decode
/// rides the repacked GEMV, prefill the mmq int8 ladder).
pub(crate) enum AttnWeights {
    F32 {
        wq: DeviceTensor,
        wk: DeviceTensor,
        wv: DeviceTensor,
        wo: DeviceTensor,
        bf16: Option<AttnBf16>,
    },
    Qw {
        wq: QuantW,
        wk: QuantW,
        wv: QuantW,
        wo: QuantW,
    },
}

/// Checkpoint-byte twins of the attention planes ([in, out] dims, the bf16
/// lane's convention). q/k/v live as one load-time-concatenated `[q;k;v]`
/// plane (thin-k/v rung): the 256-row k/v planes are
/// latency-starved on their own decode-band grids (~40 us floor at 1-3% of
/// the DRAM roof, regardless of kernel shape - probed in
/// bf16_thin_probe.cu), so the batched tick computes all three in one
/// launch riding the fused grid; the serial row reads per-projection
/// segments of the same bytes (row-major planes make a row range a byte
/// range). Same total VRAM as three separate planes.
pub(crate) struct AttnBf16 {
    /// fused plane, dims `[hidden, q_dim + 2*kv_dim]`; rows are q, then k,
    /// then v
    pub wqkv: QuantTensor,
    pub q_dim: usize,
    pub kv_dim: usize,
    pub wo: QuantTensor,
}

/// The batched decode tick's q/k/v projections over the bf16 twins - one
/// fused launch when the pack carries slot 424, three per-segment launches
/// otherwise (and under PADDOCK_NVQKV=0, the A/B pin).
///
/// The fused arm runs at all decode batches >= 2: the probe had it at 20.8
/// us (b8) / 34.6 us (b32) for the whole q|k|v read vs ~93/131 us for the
/// three separate launches - the thin k/v planes are latency-starved alone
/// at every batch. At rows 2..=8 this moves those rows from the multi-row
/// GEMV's f32-product class onto the mma's bf16-activation-cast class (the
/// same class rows > 8 always used);
/// the serve battery gates it like every twin-class election. rows == 1
/// stays on the exact per-segment GEMV.
pub(crate) fn attn_qkv_batch(
    exec: &GpuExecutor,
    b: &AttnBf16,
    x: &CudaSlice<f32>,
    yq: &mut CudaSlice<f32>,
    yk: &mut CudaSlice<f32>,
    yv: &mut CudaSlice<f32>,
    rows: usize,
) -> Result<(), crate::gpu::GpuError> {
    static QKV: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let on = *QKV.get_or_init(|| {
        paddock_models::dev_var!("PADDOCK_NVQKV")
            .map(|v| v != "0")
            .unwrap_or(true)
    });
    if on && rows > 1 && exec.has_bf16_qkv_gemm() {
        return exec.bf16_qkv_gemm(&b.wqkv, x, yq, yk, yv, b.q_dim, b.kv_dim, rows);
    }
    exec.bf16_gemm_rows(&b.wqkv, 0, b.q_dim, x, yq, rows)?;
    exec.bf16_gemm_rows(&b.wqkv, b.q_dim, b.kv_dim, x, yk, rows)?;
    exec.bf16_gemm_rows(&b.wqkv, b.q_dim + b.kv_dim, b.kv_dim, x, yv, rows)
}

/// Embedding table residency: bf16 keeps the checkpoint bytes (the gather
/// widens in-kernel - bit-identical rows at half the VRAM); f32 is the widened
/// fallback for older packs and the kill switch; Q8 is the GGUF lane's raw
/// Q8_0 table (granite's resident gather path - never repacked).
pub(crate) enum TokEmbd {
    F32(CudaSlice<f32>),
    Bf16(QuantTensor),
    Q8(QuantTensor),
}

/// MoE mixer weights: sigmoid router with a selection-only correction bias,
/// top-6 of 128 renormalized then x2.5 on the routed output only; experts
/// are squared-relu with no gate matrix; one always-on unscaled shared
/// expert in parallel.
pub(crate) struct MoeWeights {
    pub router: DeviceTensor,
    pub bias: DeviceTensor,
    pub planes: MoePlanes,
}

/// Expert plane residency: NVFP4 triples (W4A16), or repacked Q8_0 (W8A8
/// dp4a - the GGUF lane; up/down are the flat [e*ff + o] expert streams,
/// sh_* the 1-expert shared planes the relu2 kernels address with a zero
/// idx).
// one instance per MoE layer, resident for the model's life - the variant size
// gap is noise next to the device planes it names
#[allow(clippy::large_enum_variant)]
pub(crate) enum MoePlanes {
    Nvf4 {
        up: Nvf4MoePlane,
        down: Nvf4MoePlane,
        sh_up: Nvf4MoePlane,
        sh_down: Nvf4MoePlane,
    },
    Q8 {
        up: RepackedQ8,
        down: RepackedQ8,
        sh_up: RepackedQ8,
        sh_down: RepackedQ8,
    },
}

/// lm_head residency (untied): NVFP4 plane, or a GGUF quant plane.
pub(crate) enum HeadW {
    Nvf4(Nvf4Plane),
    Qw(QuantW),
}

/// Batched NVFP4 lm_head election, shared by the decode
/// tick and the spec verify pass. Wide planes from two rows up take the
/// tensor-core class - exact-dequant bf16 weights on m16n8k16 mma, where
/// the only numeric change vs the scalar lane is the f32->bf16 activation
/// cast (the same cast vLLM's bf16 serving applies before its lm_head) -
/// because the scalar classes lose everywhere batch exists: at the vocab
/// shape the probe measures tc 253-291 us vs mr 577 (b8) / 1155 (b32).
/// rows==1 keeps the per-row exact-f32 gemv (138 us at 91% of roof - tc's
/// staged ring only reaches ~50% at b1), so single-stream serving keeps
/// both its speed and its exact class. The boundary does not cost
/// composition-determinism: a probed fact is that concurrent greedy serving
/// diverges across identical reruns with the scalar head too (10-12/16 c16
/// outputs, PADDOCK_NVF4_TC=0) - batch invariance at concurrency was never a
/// live property of the graph.
/// PADDOCK_NVF4_TC=0 is the A/B kill switch. The election lives here, not
/// in the generic wrapper, so every other nvf4_gemv_batch consumer stays in
/// the exact class.
pub(crate) fn head_nvf4_batch(
    exec: &GpuExecutor,
    h: &Nvf4Plane,
    x: &CudaSlice<f32>,
    y: &mut CudaSlice<f32>,
    rows: usize,
) -> Result<(), crate::gpu::GpuError> {
    static TC: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let on = *TC.get_or_init(|| {
        paddock_models::dev_var!("PADDOCK_NVF4_TC")
            .map(|v| v != "0")
            .unwrap_or(true)
    });
    if on && rows > 1 && h.out_dim >= 4096 && h.in_dim.is_multiple_of(16) && exec.has_nvf4_gemm_tc()
    {
        exec.nvf4_gemm_tc(h, x, y, None, rows)
    } else {
        exec.nvf4_gemv_batch(h, x, y, None, rows)
    }
}

// one per layer; the Mamba/Attn/Moe weight bundles differ in size by design
#[allow(clippy::large_enum_variant)]
pub(crate) enum Mixer {
    Mamba(MambaWeights),
    Attn(AttnWeights),
    Moe(MoeWeights),
}

pub(crate) struct NemotronLayer {
    /// pre-mixer RMS norm weight [hidden]
    pub norm: DeviceTensor,
    pub mixer: Mixer,
}

pub struct GpuNemotron {
    pub(crate) exec: Arc<GpuExecutor>,
    pub(crate) hp: NemotronConfig,
    pub(crate) layers: Vec<NemotronLayer>,
    /// embedding table [vocab, hidden] (input lookup only)
    pub(crate) tok_embd: TokEmbd,
    pub(crate) final_norm: DeviceTensor,
    pub(crate) lm_head: HeadW,
    pub(crate) kv_dtype: KvDtype,
    /// Storage class for the recurrent SSM state. f16 is the elected default
    /// (NLL-gated quality-neutral); PADDOCK_SSM_DTYPE=f32 (dev builds only) selects the
    /// checkpoint-conservative reference class (arithmetic is f32 in both).
    pub(crate) ssm_dtype: ssm_arena::SsmDtype,
    pub(crate) max_ctx: usize,
    pub(crate) weights_bytes: u64,
    /// Content identity of the loaded weights and tokenizer, captured at
    /// load - the cache namespace's answer to "are these the same bytes?".
    /// Geometry alone stopped being a sufficient key when the tier gained a
    /// store that survives restarts (see `kv_tier::fingerprint`).
    pub(crate) content_id: ([u8; 32], [u8; 32]),
    pub(crate) decode: Option<DecodeState>,
    pub(crate) scratch: Option<Scratch>,
    pub(crate) prefill: Option<PrefillScratch>,
    /// in-flight serial decode pipe  - always None between requests
    pub(crate) pipe: Option<PipeState>,
    /// Continuous-batching state (batch.rs): paged KV on the 6
    /// attention layers + per-slot mamba arenas. None until `enable_batch`
    /// succeeds - the serial lane above keeps working without it.
    pub(crate) batch: Option<batch::NemoBatch>,
    /// Prompts mid-prefill, FIFO (batch.rs, stall-free batching).
    pub(crate) chunked: Vec<batch::ChunkedPrefill>,
    /// Per-slot rows the radix served this prefill (telemetry via
    /// `take_prefill_reused`); sized at enable, empty on the serial lane.
    pub(crate) last_reused: Vec<usize>,
    /// In-flight batched depth-2 decode pipe (stage E) - always None
    /// between pipe sessions.
    pub(crate) pipe_b: Option<batch::PipeB>,
    /// DFlash drafter (C2) - attached via --mtp, None otherwise.
    pub(crate) dflash: Option<dflash::DflashDrafter>,
    /// In-file MTP drafter (C3) - the GGUF's blk.52 nextn block,
    /// loaded whenever the file carries it and spec isn't killed
    /// (PADDOCK_NO_SPEC); an explicit DFlash attach evicts it.
    pub(crate) mtp: Option<mtp::MtpDrafter>,
}
