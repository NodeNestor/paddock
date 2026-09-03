//! Laguna (poolside XS-2.1 / S-2.1) - the agentic-coding MoE family.
//!
//! Architecture, verified against config.json + modeling_laguna.py + the real
//! XS-2.1 Q4_K_M GGUF (llama.cpp `src/models/laguna.cpp` and the HF modeling
//! file are the study references):
//!
//! - hybrid attention repeating [full, SWA, SWA, SWA] (window 512), with
//!   PER-LAYER Q-head counts (XS: 48 on full / 64 on SWA layers) over a
//!   uniform 8 KV heads × head_dim 128 (GQA 6:1 and 8:1);
//! - per-head QK-RMSNorm before rope (qwen3-shaped);
//! - full layers: partial-rotary YaRN - first `n_rot`=64 of 128 dims rotate
//!   (NEOX half-split, tail passes through), θ=500k, factor 32 (XS) /
//!   128 (S), mscale derived (`yarn_attn_factor` stamped 1.0);
//!   SWA layers: plain full-rotary rope, θ=10k;
//! - per-head SOFTPLUS output gate from a separate `attn_gate` projection
//!   [embd, n_heads], computed from the same normed input as q/k/v and
//!   applied to the attention output (broadcast over head_dim) before wo;
//! - MoE on every layer but layer 0 (dense 8192): 256 experts top-8 (XS) /
//!   top-10 (S), SIGMOID router scores, selection-only score-correction bias
//!   (`exp_probs_b`), sum-normalized unbiased weights, ×2.5 routed scaling,
//!   plus an ALWAYS-ON ungated shared expert (unlike qwen35moe's scalar-gated
//!   one);
//! - multi-EOS: eos 2 (= bos) and eot 24 (`</assistant>`).
//!
//! Bring-up mirrors qwen35: loader first, validated against the real GGUF;
//! forward + Generator next. Correctness bar = same-weights greedy match vs
//! the NEWEST llama.cpp release binary serving the identical file (no CPU
//! references).

use std::sync::Arc;

use crate::gpu::{DeviceTensor, GpuExecutor, KvDtype, QuantTensor, QuantW, RepackedKQ, RepackedQ8};

mod batch;
mod dflash;
mod forward;
mod load;
mod prefix;

pub use dflash::DflashSelftest;

/// A captured decode tick. The model is single-threaded on the engine's
/// thread (same argument as qwen35's SendGraph).
pub(crate) struct SendGraph(pub(crate) crate::gpu::CapturedGraph);
// SAFETY: never accessed from two threads at once; see above.
unsafe impl Send for SendGraph {}

/// Routed-expert seat, per tensor: the official XS-2.1 Q4_K_M ships Q4_K
/// gate/up + Q6_K down (k-quant-resident); the official S-2.1 Q8_0 file
/// ships Q8_0 experts. Same split qwen35moe serves.
pub(crate) enum ExpW {
    Q8(RepackedQ8),
    Kq(RepackedKQ),
}

/// 256-expert MoE FFN + always-on shared expert. Router math (from the HF
/// reference): scores = sigmoid(router logits, f32); selection =
/// top-k over scores + probs_bias; weights = the UNBIASED scores of the
/// selected experts, sum-normalized, then ×`routed_scale`; layer output =
/// routed combine + shared-expert output (no gate on the shared branch).
pub(crate) struct MoeWeights {
    /// `ffn_gate_inp` [embd, n_expert] F32 - router matvec plane.
    pub router_w: DeviceTensor,
    /// `exp_probs_b.bias` [n_expert] F32 - aux-loss-free selection bias
    /// (DeepSeek-V3 class). Biases expert CHOICE only, never the weights.
    pub probs_bias: DeviceTensor,
    /// `ffn_{gate,up}_exps` [embd, moe_ff, n_expert]; expert row (e, o) sits
    /// at e*moe_ff + o in the repacked stream.
    pub gate_exps: ExpW,
    pub up_exps: ExpW,
    /// `ffn_down_exps` [moe_ff, embd, n_expert].
    pub down_exps: ExpW,
    /// `ffn_{gate,up,down}_shexp` - plain SwiGLU FFN of width shexp_ff.
    pub shexp_gate: QuantW,
    pub shexp_up: QuantW,
    pub shexp_down: QuantW,
    /// Merged [gate | up] plane for the r==1 GEMV lane (one launch + the
    /// swiglu_fused epilogue instead of three small kernels). Built at load
    /// when both parts share one k-quant type; duplicate residency - the
    /// r>1 mmq lane keeps the split tensors.
    pub shexp_gateup: Option<RepackedKQ>,
}

/// Per-layer FFN: layer 0 is dense (ffn 8192), everything else is MoE.
pub(crate) enum Ffn {
    Dense {
        gate: QuantW,
        up: QuantW,
        down: QuantW,
    },
    Moe(MoeWeights),
}

/// One transformer block.
pub(crate) struct LagunaLayer {
    pub attn_norm: DeviceTensor,
    /// `attn_q` [embd, n_heads*head_dim] - width varies per layer.
    pub wq: QuantW,
    /// `attn_k` / `attn_v` [embd, n_kv_heads*head_dim] (uniform).
    pub wk: QuantW,
    pub wv: QuantW,
    /// `attn_output` [n_heads*head_dim, embd].
    pub wo: QuantW,
    /// `attn_gate` [embd, n_heads] - the per-head softplus output gate.
    pub g_proj: QuantW,
    /// `attn_{q,k}_norm` [head_dim] F32 - per-head QK-RMSNorm (pre-rope).
    pub q_norm: DeviceTensor,
    pub k_norm: DeviceTensor,
    pub ffn_norm: DeviceTensor,
    pub ffn: Ffn,
    /// SWA-512 layer (il % 4 != 0). Full layers run partial-rotary YaRN.
    pub is_swa: bool,
    /// This layer's Q-head count (48 full / 64 SWA on XS-2.1).
    pub n_heads: usize,
    /// Merged [q | k | gate] plane for the r==1 GEMV lane: one big launch
    /// (near-roof bandwidth, like the lm head) replaces three small GEMVs.
    /// v stays separate (Q6_K on the XS file). Duplicate residency (~400 MB
    /// on XS) - the r>1 mmq lane keeps the split tensors; the profile showed
    /// the decode tick is small-kernel-bound, not bandwidth-bound.
    pub qkg: Option<crate::gpu::RepackedKQ>,
}

#[derive(Clone, Copy)]
pub(crate) struct MoeDims {
    pub n_expert: usize,
    pub n_active: usize,
    pub moe_ff: usize,
    pub shexp_ff: usize,
    /// `expert_weights_scale` (2.5) - multiplies the routed combine.
    pub routed_scale: f32,
}

/// Geometry + rope constants from the GGUF header.
pub(crate) struct Hparams {
    pub n_layer: usize,
    pub n_embd: usize,
    /// Per-layer Q-head counts (`attention.head_count` ships as an ARRAY).
    pub n_heads: Vec<usize>,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub n_vocab: usize,
    pub eps: f32,
    pub swa_window: usize,
    /// Full-attn rope: YaRN over the first `n_rot` dims (64), tail untouched.
    /// Tuple = YarnRope::kernel_params order (theta_scale, freq_scale,
    /// corr_low, corr_high, ext_factor, mscale).
    pub n_rot: usize,
    pub rope_full: (f32, f32, f32, f32, f32, f32),
    pub rope_swa: (f32, f32, f32, f32, f32, f32),
    pub moe: MoeDims,
}

/// Token-embedding residency: gathered from its file quant (Q4_K on the
/// XS-2.1 election, Q8_0 on plain exports) - same split as qwen35.
pub(crate) enum TokEmbd {
    Q8(QuantTensor),
    Kq(RepackedKQ),
}

/// The Laguna GPU model. Loader-only milestone: weights resident + geometry
/// parsed; forward/batch/spec land next (order).
pub struct GpuLaguna {
    pub(crate) exec: Arc<GpuExecutor>,
    pub(crate) hp: Hparams,
    pub(crate) tok_embd: TokEmbd,
    pub(crate) layers: Vec<LagunaLayer>,
    pub(crate) output_norm: DeviceTensor,
    /// `output` [embd, vocab] (Q6_K on the XS election).
    pub(crate) lm_head: QuantW,
    pub(crate) max_ctx: usize,
    /// VRAM the load phase consumed (ledger delta) - feeds
    /// Generator::weights_mem_bytes.
    pub(crate) weights_bytes: u64,
    /// Content identity of the loaded weights and tokenizer, captured at
    /// load - the cache namespace's answer to "are these the same bytes?".
    /// Geometry alone stopped being a sufficient key when the tier gained a
    /// store that survives restarts (see `kv_tier::fingerprint`).
    pub(crate) content_id: ([u8; 32], [u8; 32]),
    /// KV cache element type (default [`KvDtype::Fp16`], greedy-exact).
    /// [`KvDtype::Fp8E4m3`] is a lossy opt-in throughput/memory mode, set via
    /// `set_kv_dtype` before serving. Independent of the DFlash drafter's own
    /// aux-feature KV (dflash.rs), which stays pinned to f16.
    pub(crate) kv_dtype: KvDtype,
    /// Lazily-built serial decode state + scratch (forward.rs).
    pub(crate) decode: Option<forward::DecodeState>,
    pub(crate) scratch: Option<forward::Scratch>,
    /// Continuous-batching state (batch.rs): paged KV (SWA rings + full-layer
    /// budget pool), per-slot tables, batched scratch. None until
    /// enable_batch succeeds with capacity > 1.
    pub(crate) batch: Option<batch::BatchState>,
    /// In-flight pipelined decode (batch.rs decode_pipe_*). None between
    /// pipes; every other forward path requires it drained first.
    pub(crate) pipe: Option<batch::PipeState>,
    /// Sideloaded DFlash block-diffusion drafter (dflash.rs) - poolside's
    /// official speculator checkpoint, attached via serving config. None =
    /// no model drafter (spec rounds fall back to the service's n-gram).
    pub(crate) dflash: Option<dflash::DflashDrafter>,
    /// Prompts mid-prefill, FIFO (batch.rs, stall-free batching). Each entry
    /// carries a CURSOR, so a tick advances a prompt partway and the next
    /// tick resumes it - decode rows never wait for a whole prompt, let
    /// alone a whole admission wave.
    pub(crate) chunked: Vec<batch::ChunkedPrefill>,
}
