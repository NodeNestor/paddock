//! Qwen3 dense transformer (GGUF arch `qwen3`) as an ENCODER: the backbone for
//! Qwen3-Embedding and Qwen3-Reranker (the private-RAG capability). It reuses
//! qwen35's TUNED prefill kernel path - fused residual+norm+quantize, int8
//! mma/mmq GEMMs, the f16/mma prefill attention class, and the fused
//! SwiGLU-quantize FFN tail - so it inherits every per-die optimization. The
//! only differences from qwen35's full-attn layer:
//!   - ungated attention (no query||gate split, no sigmoid gate);
//!   - `ffn_norm` (not `post_attention_norm`);
//!   - plain 1D RoPE (`rope_yarn_batch`, not sectioned mRoPE);
//!   - tied lm_head (reuse the Q8_0 `token_embd`).
//!
//! Embeddings/reranking are non-autoregressive, so encoding is a single RAGGED
//! BATCHED prefill over many texts: each text takes a KV slot, all rows run in
//! one weight-amortized pass with per-slot attention isolation, then each
//! sequence's last-token hidden is pooled. That batch amortization - not a
//! per-text loop - is the throughput lever for RAG-scale embedding. Max
//! performance, no simplified fallback path.

use std::sync::Arc;

use cudarc::driver::{CudaSlice, DevicePtr};
use paddock_kernels::reference::ops::YarnRope;
use paddock_models::gguf::Value;
use paddock_models::mapped::MappedGguf;

use crate::gpu::{DeviceTensor, GpuError, GpuExecutor, KvDtype, RepackedMxfp4, RepackedQ8};
use crate::gpu_model::gpt_oss::GpuModelError;
use crate::gpu_model::qwen35::{
    prefill_add_norm_quant, prefill_ffn_down_sk, prefill_mm, prefill_mm_pre_sk,
};

/// Prefill batch above which the sm_120a block-scale FP4 GEMMs take the FFN
/// (same boundary the mmq_hi/mmq_pipe rung uses - below it the int8 classes
/// already saturate the die and the fp4 numeric class buys nothing).
const BS_MIN_BATCH: usize = 1024;

/// Numeric class of one block-scale FFN GEMM site (weights are packed per
/// class at load; the plane bytes are not interchangeable between classes).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum BsClass {
    /// fp4 weights x e4m3 activations (mxf8f6f4 k32) - the precision rung.
    F8,
    /// e4m3 weights x e4m3 activations (W8A8-FP8): the weight-precision rung
    /// between q8_0 and the fp4 classes - ~4x finer weight mantissa than F8's
    /// e2m1, still the full block-scale MMA rate. For models whose gates
    /// reject every 4-bit-weight rung (the 0.6B reranker).
    W8,
    /// fp4 x fp4, ue8m0 per-32 (mxf4 k64) - 2x issue rate, harshest class.
    F4,
    /// fp4 x fp4, E4M3 scales per-16 (mxf4nvf4 k64) - the NVFP4 recipe:
    /// mxf4's speed with outlier-tolerant scaling.
    Nv4,
    /// Nv4 with a fused orthonormal H128 rotation on both sides (the QuaRot
    /// treatment): tames the post-norm hidden's outlier channels so even
    /// that input runs fp4 x fp4. Weights and activations must both be the
    /// rotated variant - the rotations cancel inside the GEMM.
    Nv4Rot,
    /// Nv4 with SmoothQuant per-channel scale migration: activations divide
    /// by a calibrated s[c] inside the quantizer, the site's weights
    /// multiply by s[c] at requant - outlier channels shrink toward the
    /// rest before the per-16 quantize. Calibration-only (needs the
    /// activation statistics a calibration pass collects).
    Nv4Smooth,
    /// F8 with the SmoothQuant fold - here the migration usually runs
    /// WEIGHT-ward (low alpha ~ 1/w_colmax): the fp4 weights' per-32 ue8m0
    /// blocks are the coarse side, and normalizing their columns lets the
    /// finer e4m3 activations absorb the range. The small models' rung:
    /// they fail plain fp4 weights on recall. Calibration-only.
    F8Smooth,
}

/// The calibration ladder, fastest profile first: (name, gate/up, down,
/// qkv, wo). Names are the on-disk cache vocabulary - additions are fine,
/// renames/semantic changes must bump the server's calib cache schema.
///
/// Measured verdicts across the Qwen3 encoder family (GB202, Q8_0 GGUFs,
/// confirmed by two independent forced recalibrations) - fp4
/// tolerance grows with model size, depends on the task (near-tie
/// reranking under hard negatives is stricter than embedding retrieval),
/// and the FAILURE MODE differs by size: the larger models are limited by
/// fp4 ACTIVATIONS on the hidden (nvf4-ward smoothing fixes it), the small
/// ones by fp4 WEIGHT blocks (weight-ward F8Smooth fixes the embedder;
/// the 0.6B reranker tolerates neither direction - nor even 8-bit e4m3
/// weights (w8a8 rank1 12 vs baseline 15) - and floors on Q8_0. Verdicts
/// re-measured under the WMMA-f16 attention class:
///   Embedding-0.6B  all-f8s         Reranker-0.6B  q8_0 (floor)
///   Embedding-4B    all-f8          Reranker-4B    all-f8s
///   Embedding-8B    nv4-smooth-all  Reranker-8B    nv4-smooth-all
const BS_LADDER: [(
    &str,
    Option<BsClass>,
    BsClass,
    Option<BsClass>,
    Option<BsClass>,
); 9] = [
    // every site smoothed, INCLUDING the down projection (rerank-8B's last
    // lost query traced to the unsmoothed down site: nv4-down alone cost
    // rank1 16 -> 15 while all-f8 held 16/16)
    (
        "nv4-smooth-all",
        Some(BsClass::Nv4Smooth),
        BsClass::Nv4Smooth,
        Some(BsClass::Nv4Smooth),
        Some(BsClass::Nv4Smooth),
    ),
    // hidden readers + wo smoothed, down plain nvf4
    (
        "nv4-smooth",
        Some(BsClass::Nv4Smooth),
        BsClass::Nv4,
        Some(BsClass::Nv4Smooth),
        Some(BsClass::Nv4Smooth),
    ),
    // nvf4 FFN + nvf4 wo (the measured 8B profile)
    (
        "nv4+wo4",
        Some(BsClass::Nv4),
        BsClass::Nv4,
        Some(BsClass::F8),
        Some(BsClass::Nv4),
    ),
    // conservative mix: e4m3 activations wherever the hidden is read
    (
        "nv4-down",
        Some(BsClass::F8),
        BsClass::Nv4,
        Some(BsClass::F8),
        Some(BsClass::F8),
    ),
    // same speed shape as nv4-down but every fp4 plane smooth-folded - the
    // weight-quantization-quality rung for smaller models
    (
        "f8s-nv4dn",
        Some(BsClass::F8Smooth),
        BsClass::Nv4Smooth,
        Some(BsClass::F8Smooth),
        Some(BsClass::F8Smooth),
    ),
    // precision rung everywhere
    (
        "all-f8",
        Some(BsClass::F8),
        BsClass::F8,
        Some(BsClass::F8),
        Some(BsClass::F8),
    ),
    // ... and its smooth-folded twin
    (
        "all-f8s",
        Some(BsClass::F8Smooth),
        BsClass::F8Smooth,
        Some(BsClass::F8Smooth),
        Some(BsClass::F8Smooth),
    ),
    // last rung above the q8_0 floor: 8-bit e4m3 weights keep quality where
    // every 4-bit-weight rung fails, and the block-scale MMA still beats mmq
    (
        "w8a8",
        Some(BsClass::W8),
        BsClass::W8,
        Some(BsClass::W8),
        Some(BsClass::W8),
    ),
    // hybrid tail: W8A8-FP8 only on the tame sites (attention output + FFN
    // down - whose narrow GEMMs are also the wave-starved ones at serving
    // batches), exact Q8_0 on the outlier-sensitive hidden readers (qkv +
    // gate/up). The chance for weight-precision-bound small models to get
    // off the floor: all-w8 costs rank1 12 vs baseline 15 on the 0.6B
    // reranker; sparing the hidden readers spares most of that.
    ("w8-tail", None, BsClass::W8, None, Some(BsClass::W8)),
];

/// An encode submitted to the GPU but not yet read back: every kernel is
/// enqueued (forward, pool gather, output norm into a dedicated flip
/// buffer) and `ev` marks the pooled rows' completion. The encoder thread
/// pipelines these - submit batch k+1 on the compute stream, then collect
/// batch k through the side stream - so the GPU never idles on the CPU
/// tail (readback, rerank head, replies) or client turnaround.
pub struct PendingPooled {
    ev: cudarc::driver::CudaEvent,
    flip: usize,
    n: usize,
    lane: usize,
}

/// A captured encode graph; CapturedGraph is not Send, but the model lives on
/// one thread (the decode paths' SendGraph convention).
struct SendGraph(crate::gpu::CapturedGraph);
// SAFETY: the model (and thus every graph) is owned and used by exactly one
// thread - the encoder thread that built it.
unsafe impl Send for SendGraph {}

/// The swappable half of a second execution lane (see `GpuQwen3::lane1`).
struct LaneState {
    exec: Arc<GpuExecutor>,
    scratch: Option<EncScratch>,
    prefix_kv: Option<PrefixKv>,
    pool_out: [Option<CudaSlice<f32>>; 2],
    pool_flip: usize,
    graphs: std::collections::HashMap<(usize, usize, usize, usize), (u64, SendGraph)>,
}

/// Saved shared-prefix KV rows (slot-0 snapshot) for cross-encode reuse.
struct PrefixKv {
    tokens: Vec<u32>,
    epoch: u64,
    /// per-layer buffers, each holding `tokens.len() * kv_dim * kv_bytes`
    k: Vec<CudaSlice<u8>>,
    v: Vec<CudaSlice<u8>>,
    /// allocation capacity in prefix rows (buffers are reused grow-only)
    cap_rows: usize,
}

/// One dense transformer block: ungated GQA attention + SwiGLU FFN.
struct Qwen3Layer {
    attn_norm: DeviceTensor,
    /// Fused QKV: q/k/v rows concatenated into one plane so the projections
    /// run as a single wide GEMM (three narrow GEMMs were 1.2-wave launches
    /// idling ~40% of the GPU at serving batch sizes). Built at load when
    /// the pack carries the fused consumer kernel; the separate planes
    /// below are the fallback and are None when fused (never both).
    wqkv: Option<RepackedQ8>,
    wq: Option<RepackedQ8>,
    wk: Option<RepackedQ8>,
    wv: Option<RepackedQ8>,
    q_norm: DeviceTensor,
    k_norm: DeviceTensor,
    wo: RepackedQ8,
    ffn_norm: DeviceTensor,
    ffn_gate: RepackedQ8,
    ffn_up: RepackedQ8,
    ffn_down: RepackedQ8,
    /// sm_120a block-scale FP4 twins of the FFN weights. None off Blackwell
    /// or under PADDOCK_NO_BS; the Q8_0 tensors above stay resident for the
    /// sub-[`BS_MIN_BATCH`] rungs either way.
    ffn_gate_mx: Option<RepackedMxfp4>,
    ffn_up_mx: Option<RepackedMxfp4>,
    ffn_down_mx: Option<RepackedMxfp4>,
    /// FP4 twins of the attention projections (class in `bs_attn` - rotated
    /// nvf4 by default, e4m3-activation as the pinned fallback).
    wqkv_mx: Option<RepackedMxfp4>,
    wq_mx: Option<RepackedMxfp4>,
    wk_mx: Option<RepackedMxfp4>,
    wv_mx: Option<RepackedMxfp4>,
    wo_mx: Option<RepackedMxfp4>,
    /// SmoothQuant per-channel scale vectors for this layer's three
    /// fp4-quantized inputs (attn hidden, ffn hidden, wo input), built from
    /// calibration statistics. `s` folds into weights at requant, `sinv`
    /// into the activation quantizer.
    smooth: Option<SmoothVecs>,
}

/// Per-layer SmoothQuant scales: (s, 1/s) per site input channel - one set
/// tuned for the fp4-activation classes (`_s`), one for the e4m3-activation
/// class (`_s8`, weight-ward alphas).
struct SmoothVecs {
    attn_s: CudaSlice<f32>,
    attn_sinv: CudaSlice<f32>,
    gu_s: CudaSlice<f32>,
    gu_sinv: CudaSlice<f32>,
    wo_s: CudaSlice<f32>,
    wo_sinv: CudaSlice<f32>,
    dn_s: CudaSlice<f32>,
    dn_sinv: CudaSlice<f32>,
    attn_s8: CudaSlice<f32>,
    attn_s8inv: CudaSlice<f32>,
    gu_s8: CudaSlice<f32>,
    gu_s8inv: CudaSlice<f32>,
    wo_s8: CudaSlice<f32>,
    wo_s8inv: CudaSlice<f32>,
    dn_s8: CudaSlice<f32>,
    dn_s8inv: CudaSlice<f32>,
}

/// Per-layer activation abs-max accumulators (calibration stats pass).
struct StatBufs {
    attn: CudaSlice<f32>,
    gu: CudaSlice<f32>,
    wo: CudaSlice<f32>,
    dn: CudaSlice<f32>,
}

/// Lazily-(re)allocated encode scratch, grown to the row / slot high-water
/// mark. Prefill-only: KV is a single per-layer slot-strided buffer (no
/// persistent decode state, no recurrent/conv buffers).
struct EncScratch {
    rows: usize,
    slots: usize,
    /// KV cache stride (tokens): the batch's max sequence length, not max_ctx -
    /// encoder sequences are short, so sizing KV to max_ctx would waste GBs.
    kv_stride: usize,
    d_x: CudaSlice<f32>,
    d_xn: CudaSlice<f32>,
    d_q: CudaSlice<f32>,
    /// combined qkv GEMM output [rows, q_dim + 2*kv_dim] (fused-QKV route)
    d_qkv: CudaSlice<f32>,
    /// persistent host-upload targets (tokens/positions/slots, attention
    /// tiles, pool gather indices, batched-copy descriptors): per-encode
    /// transient allocs are stream-ordered pool ops whose cross-stream reuse
    /// ordering serializes the dual-lane streams
    up_tokens: CudaSlice<u32>,
    up_pos: CudaSlice<u32>,
    up_slots: CudaSlice<u32>,
    up_trow0: CudaSlice<u32>,
    up_tslot: CudaSlice<u32>,
    up_idx: CudaSlice<u32>,
    up_descs: CudaSlice<u64>,
    /// split-K partials for the tail-wave mmq (tail<=sm/2 x 4 splits x 128^2)
    d_skpart: CudaSlice<f32>,
    d_k: CudaSlice<f32>,
    d_v: CudaSlice<f32>,
    d_qn: CudaSlice<f32>,
    d_kn: CudaSlice<f32>,
    d_attn: CudaSlice<f32>,
    d_proj: CudaSlice<f32>,
    d_ffn_gate: CudaSlice<f32>,
    d_ffn_up: CudaSlice<f32>,
    d_h: CudaSlice<f32>,
    /// Gathered per-slot pooled rows ([slots, embd]) - the only plane that
    /// crosses to the host. Copying `d_h` wholesale cost a fixed ~50 MB
    /// pageable D2H per encode at the scratch high-water mark (~25 ms, which
    /// dwarfed small requests and taxed every big one).
    // (a slots*embd staging pool lived here to dodge a ~50 MB pageable D2H
    // per encode; the row-ranged to_host_len path made it redundant)
    // quantize staging the tuned prefill helpers read/write
    d_pxq: CudaSlice<i8>,
    d_pxs: CudaSlice<f32>,
    d_yq: CudaSlice<u8>,
    d_skfix: CudaSlice<f32>,
    /// ue8m0 scale plane for the block-scale route (the e4m3 bytes reuse
    /// `d_pxq` as a raw byte plane, like gpt-oss). None off the bs path.
    d_exs: Option<CudaSlice<u8>>,
    /// Output planes of the fused gate+up+swiglu->nvf4 kernel (the down
    /// GEMM's input) - separate from `d_pxq`/`d_exs`, which still hold the
    /// hidden the fused kernel is reading. None off the fused path.
    d_fq: Option<CudaSlice<i8>>,
    d_fs: Option<CudaSlice<u8>>,
    // per-layer KV [slots, max_ctx, kv_dim] in raw bytes (kv_dtype-sized)
    kv_k: Vec<CudaSlice<u8>>,
    kv_v: Vec<CudaSlice<u8>>,
}

pub struct GpuQwen3 {
    exec: Arc<GpuExecutor>,
    n_layers: usize,
    embd: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    ff: usize,
    pub vocab: usize,
    max_ctx: usize,
    rms_eps: f32,
    /// YaRN/RoPE kernel params (ext_factor 0 => plain RoPE).
    rope: (f32, f32, f32, f32, f32, f32),
    kv_dtype: KvDtype,
    tok_embd: DeviceTensor,
    layers: Vec<Qwen3Layer>,
    out_norm: DeviceTensor,
    /// lm_head - tied to `token_embd` when `output.weight` is absent.
    output: RepackedQ8,
    /// no-op attention sinks (Qwen3 has none): the -inf per-head identity, so
    /// the sink term contributes exactly 0 mass. See `alloc_no_sinks`.
    sinks: CudaSlice<f32>,
    /// Which block-scale class each GEMM site's `*_mx` planes were packed
    /// for (see [`BsClass`]).
    bs_gu: Option<BsClass>,
    bs_dn: BsClass,
    bs_attn: Option<BsClass>,
    bs_wo: Option<BsClass>,
    /// True when the classes came from explicit env pins - `calibrate_bs`
    /// then refuses to override them.
    bs_pinned: bool,
    /// Dequanted (yes, no) lm_head rows for the reranker score - cached so
    /// each rerank call is two CPU dots per document, not a vocab GEMV.
    head_rows_cache: Option<(u32, u32, Vec<f32>, Vec<f32>)>,
    /// When Some, run_layers accumulates per-channel activation abs-max at
    /// the three fp4 input sites (the SmoothQuant statistics pass).
    stat_bufs: Option<Vec<StatBufs>>,
    /// Cross-encode prefix-KV cache: per-layer snapshots of the shared-prefix
    /// KV rows keyed by the exact prefix tokens. A repeated query (rerank
    /// pagination, successive coalesced chunks of the same request stream)
    /// skips the whole prefix pass. Numerics-invalidated by `bs_epoch`.
    prefix_kv: Option<PrefixKv>,
    /// Double-buffered pooled-output planes ([slots, embd] each) so a
    /// submitted-but-uncollected batch's rows survive the next submit.
    pool_out: [Option<CudaSlice<f32>>; 2],
    pool_flip: usize,
    /// Multi-lane execution: extra streams + scratch/pool/prefix state so
    /// several batches overlap on the GPU (wave-starved small batches
    /// backfill each other's idle SMs, and one lane's clients can turn
    /// around while the others crunch). `set_lane` SWAPS the target lane's
    /// state into the primary fields, so every encode path stays
    /// lane-oblivious. Parked lanes live at index lane-1; weights and
    /// quantized planes are shared (read-only after load; plane changes
    /// synchronize the lane-0 stream). Empty until the encoder asks.
    lanes_extra: Vec<LaneState>,
    cur_lane: usize,
    /// Captured encode graphs for the CURRENT lane, keyed by padded shape
    /// (rows bucket, fixed tile count, kv_stride, tile width); the entry
    /// carries the bs_epoch it was captured under. Swapped with lane state
    /// (graphs bake lane-specific scratch pointers); cleared when the
    /// lane's scratch rebuilds.
    graphs: std::collections::HashMap<(usize, usize, usize, usize), (u64, SendGraph)>,
    /// shape-recurrence counter gating capture (heuristic; shared across lanes)
    graph_hits: std::collections::HashMap<(usize, usize, usize, usize), u32>,
    /// Bumped whenever the block-scale planes change (set_bs_planes, profile
    /// application, ladder plane drops) - the prefix KV depends on the qkv
    /// GEMM class, so a stale snapshot must never survive a plane swap.
    bs_epoch: u64,
    /// Whether the default FFN front half runs the FUSED gate+up+swiglu
    /// kernel (bit-identical to the unfused chain; requires the F8-gu +
    /// Nv4-dn class pair and the pack entry).
    bs_gu_fused: bool,
    scratch: Option<EncScratch>,
    /// Resident weight bytes. Stamped absolutely at the end of `load` (no
    /// scratch, no pooled planes yet, so the pool's used counter is the
    /// weights) and then kept current by `set_bs_planes`, which is the only
    /// thing that changes the weight set afterwards - the startup calibration
    /// re-quantizes every plane to a different class, and F8 and Nv4 planes
    /// are not the same size. Feeds `/api/stats`' weights line, which was
    /// simply absent on every encoder lane before.
    weights_bytes: Option<u64>,
}

impl GpuQwen3 {
    pub fn load(
        exec: Arc<GpuExecutor>,
        map: &MappedGguf,
        max_ctx: usize,
    ) -> Result<Self, GpuModelError> {
        exec.disable_event_tracking();
        let u = |k: &str| {
            map.gguf()
                .arch_field(k)
                .and_then(Value::as_u64)
                .ok_or_else(|| GpuModelError::MissingMeta(k.to_owned()))
        };
        let f = |k: &str| map.gguf().arch_field(k).and_then(Value::as_f32);

        let n_layers = u("block_count")? as usize;
        let embd = u("embedding_length")? as usize;
        let n_heads = u("attention.head_count")? as usize;
        let n_kv_heads = u("attention.head_count_kv")? as usize;
        let head_dim = u("attention.key_length")? as usize;
        let ff = u("feed_forward_length")? as usize;
        let rms_eps = f("attention.layer_norm_rms_epsilon").unwrap_or(1e-6);
        // Qwen3 rotates the full head_dim (no partial-rope key) with plain RoPE.
        let n_rot = map
            .gguf()
            .arch_field("rope.dimension_count")
            .and_then(Value::as_u64)
            .map(|v| v as usize)
            .unwrap_or(head_dim);
        let rope_base = f("rope.freq_base").unwrap_or(1e6);
        let ctx_train = u("context_length").unwrap_or(max_ctx as u64) as usize;
        let rope =
            YarnRope::new(n_rot, rope_base, 1.0, ctx_train, 0.0, 1.0, 32.0, 1.0).kernel_params();

        let tok_embd = exec.upload(map, "token_embd.weight")?;
        let vocab = tok_embd.dims[1];

        // sm_120a block-scale FP4 route for the FFN GEMMs: re-quantize the
        // Q8_0 weights to mxfp4 (e2m1 + ue8m0) at load and run the hardware
        // block-scale MMA at prefill batch. LOSSY numeric classes, held to
        // the retrieval-quality gate (qwen3_embed_quality: recall@10 AND
        // recall@1 vs the Q8_0 baseline on 4B/8B). Per-GEMM-site rungs:
        //   fp4 x e4m3 (mxf8f6f4 k32)  - precision rung
        //   fp4 x fp4  (mxf4 k64)      - 2x the MMA issue rate, but e2m1
        //                                activations are brutal (2-bit
        //                                mantissa, 50% steps)
        // Measured on the GB202 gate: the POST-NORM HIDDEN (gate/up input,
        // known outlier channels) does not survive e2m1 - recall@1 dropped
        // 0.875 -> 0.719 (4B) with fp4 everywhere - while the SWIGLU OUTPUT
        // (down input) held recall@1 at-or-above baseline on both models.
        // So the DEFAULT is the mixed rung: gate/up on e4m3 activations,
        // down on fp4 x fp4. PADDOCK_BS_MODE overrides per site ("f4" both
        // fp4, "gu4" the reverse mix, "f8" both e4m3), PADDOCK_BS_F8=1 pins
        // all-e4m3, PADDOCK_NO_BS=1 pins the Q8_0 path on the same binary.
        // SOTA follow-up: the nvf4 kind (e2m1 with e4m3 scales per 16) is
        // built for exactly the outlier-channel problem - it could make the
        // gate/up input fp4-clean and unlock the full-f4 rate by default.
        // The cc gate matters: the pack entries are non-NULL on any
        // PD_BS_HOST build, but only sm_120a carries SASS for them.
        let bs_ok = exec.compute_capability().0 == 12
            && paddock_models::dev_var_os!("PADDOCK_NO_BS").is_none();
        let f8_pin = paddock_models::dev_var_os!("PADDOCK_BS_F8").is_some();
        let f4_ok = bs_ok && exec.has_mxfp4_gemm_f4() && !f8_pin;
        let nv4_ok = bs_ok && exec.has_mxfp4_gemm_nv4() && !f8_pin;
        let rot_ok = bs_ok && exec.has_nvf4_rot() && !f8_pin;
        let mode = paddock_models::dev_var!("PADDOCK_BS_MODE").unwrap_or_default();
        use BsClass::{F4, F8, Nv4, Nv4Rot};
        let (bs_gu, bs_dn): (Option<BsClass>, BsClass) = match mode.as_str() {
            "nv4" if nv4_ok => (Some(Nv4), Nv4),
            "f4" if f4_ok => (Some(F4), F4),
            "gu4" if f4_ok => (Some(F4), F8),
            "dn4" if f4_ok => (Some(F8), F4),
            "f8" => (Some(F8), F8),
            // MEASURED NEGATIVE RESULT, kept reproducible: "rot" runs the
            // hidden-reading GEMMs on H128 QuaRot-rotated nvf4. On this
            // encoder it does not rescue the hidden (4B recall@1 0.766 =
            // the unrotated fp4 number; 8B 0.672, worse than unrotated) -
            // per-16 e4m3 scales already contain outliers block-locally,
            // and the rotation spreads their energy so every block
            // quantizes coarser. QuaRot's W4A4 wins assume coarse int4
            // scale granularity, not the NVFP4 class.
            "rot" if rot_ok => (Some(Nv4Rot), Nv4),
            "nv4f8" if nv4_ok => (Some(F8), Nv4),
            "w8" if exec.has_f8_gemm_w8() => (Some(BsClass::W8), BsClass::W8),
            // default: e4m3 activations for the GEMMs reading the post-norm
            // hidden (fp4 activations fail the recall@1 gate there raw
            // under both scale flavors AND rotated - see "rot"), nvf4 for
            // the down site (its swiglu input holds recall@1 at-or-above
            // baseline).
            _ if nv4_ok => (Some(F8), Nv4),
            _ if f4_ok => (Some(F8), F4),
            _ => (Some(F8), F8),
        };
        // Attention projections join the block-scale route: wq/wk/wv read
        // the hidden -> e4m3-activation class; wo reads the ATTENTION
        // OUTPUT, which (like the swiglu output) tolerates fp4 -> nvf4.
        // PADDOCK_NO_BS_ATTN pins all four on Q8_0 mmq; PADDOCK_BS_ATTN=f8
        // pins wo back to the e4m3 class; ="rot" reproduces the rotated
        // experiment on all four.
        let attn_mode = paddock_models::dev_var!("PADDOCK_BS_ATTN").unwrap_or_default();
        let (bs_attn, bs_wo) =
            if !bs_ok || paddock_models::dev_var_os!("PADDOCK_NO_BS_ATTN").is_some() {
                (None, None)
            } else if attn_mode == "rot" && rot_ok {
                (Some(Nv4Rot), Some(Nv4Rot))
            } else if attn_mode == "nv4" && nv4_ok {
                // the everything-fp4 attention rung (8B-class profiles only -
                // the 4B hidden is not fp4-clean)
                (Some(Nv4), Some(Nv4))
            } else if exec.has_mxfp4_gemm_bs() {
                // wo=Nv4 measured 4B-dirty too (recall@1 0.828 vs 0.875 baseline;
                // 8B was fine at 0.859) - fp4 activations stay opt-in there
                let wo = if nv4_ok && attn_mode == "wo4" {
                    Nv4
                } else {
                    F8
                };
                (Some(F8), Some(wo))
            } else {
                (None, None)
            };
        let use_bs = bs_ok && (nv4_ok || f4_ok || exec.has_mxfp4_gemm_bs());
        // Explicit env pins skip the load-time calibration below.
        let explicit = !mode.is_empty() || !attn_mode.is_empty() || f8_pin;

        let mut layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let dt = |name: &str| exec.upload(map, &format!("blk.{i}.{name}"));
            let qt = |name: &str| exec.repack_q8(map, &format!("blk.{i}.{name}"));
            let (wq, wk, wv) = (
                qt("attn_q.weight")?,
                qt("attn_k.weight")?,
                qt("attn_v.weight")?,
            );
            // PADDOCK_NO_QKV_FUSE pins the separate-plane route for A/B
            let fuse_qkv = exec.has_qkv_norm_rope_append()
                && head_dim == 128
                && paddock_models::dev_var_os!("PADDOCK_NO_QKV_FUSE").is_none();
            let (wqkv, wq, wk, wv) = if fuse_qkv {
                (Some(exec.concat_q8(&[&wq, &wk, &wv])?), None, None, None)
            } else {
                (None, Some(wq), Some(wk), Some(wv))
            };
            layers.push(Qwen3Layer {
                attn_norm: dt("attn_norm.weight")?,
                wqkv,
                wq,
                wk,
                wv,
                q_norm: dt("attn_q_norm.weight")?,
                k_norm: dt("attn_k_norm.weight")?,
                wo: qt("attn_output.weight")?,
                ffn_norm: dt("ffn_norm.weight")?,
                ffn_gate: qt("ffn_gate.weight")?,
                ffn_up: qt("ffn_up.weight")?,
                ffn_down: qt("ffn_down.weight")?,
                // block-scale planes are built after construction, by
                // set_bs_planes (explicit env classes) or calibrate (the
                // measured per-model default)
                ffn_gate_mx: None,
                ffn_up_mx: None,
                ffn_down_mx: None,
                wqkv_mx: None,
                wq_mx: None,
                wk_mx: None,
                wv_mx: None,
                wo_mx: None,
                smooth: None,
            });
        }

        let out_norm = exec.upload(map, "output_norm.weight")?;
        let output = if map.tensor_info("output.weight").is_some() {
            exec.repack_q8(map, "output.weight")?
        } else {
            exec.repack_q8(map, "token_embd.weight")?
        };
        let sinks = exec.alloc_no_sinks(n_heads)?;

        let mut me = Self {
            exec,
            n_layers,
            embd,
            n_heads,
            n_kv_heads,
            head_dim,
            ff,
            vocab,
            max_ctx,
            rms_eps,
            rope,
            kv_dtype: KvDtype::Fp16,
            tok_embd,
            layers,
            out_norm,
            output,
            sinks,
            bs_gu,
            bs_dn,
            bs_attn,
            bs_wo,
            bs_pinned: false,
            bs_gu_fused: false,
            head_rows_cache: None,
            stat_bufs: None,
            prefix_kv: None,
            pool_out: [None, None],
            pool_flip: 0,
            lanes_extra: Vec::new(),
            cur_lane: 0,
            graphs: std::collections::HashMap::new(),
            graph_hits: std::collections::HashMap::new(),
            bs_epoch: 0,
            scratch: None,
            weights_bytes: None,
        };
        // base-vs-derived split for the ledger line below: the `_mx`
        // duplicates were ~3.7 GB on Embedding-8B with no line saying so
        let base_bytes = me.exec.settled_mem_used();
        if use_bs {
            // env-pinned or static-default classes; a caller with a
            // tokenizer (the server) can upgrade the default per model by
            // running `calibrate_bs` on real text afterwards. `explicit`
            // env pins mark the classes as final (calibrate_bs refuses).
            me.bs_pinned = explicit;
            me.set_bs_planes()?;
        }
        // MEASURED NEGATIVE RESULT, kept reproducible: the fused gate+up+
        // swiglu->nvf4 kernel is bit-identical to the chain but slower on
        // GB202 - dual accumulators + two A-fragment sets need ~130 regs,
        // which spills at 2 blocks/SM (361.7 vs 392.6 texts/s on 4B) and
        // exposes barriers at 1 block/SM (339.0). Opt-in for revisits.
        me.bs_gu_fused = me.bs_gu == Some(F8)
            && me.bs_dn == Nv4
            && me.layers.first().is_some_and(|l| l.ffn_down_mx.is_some())
            && me.exec.has_mxfp4_gemm_bs_gu()
            && paddock_models::dev_var_os!("PADDOCK_BS_GU_FUSED").is_some();
        // Resident-weight line: nothing above allocates scratch, pooled output
        // planes or prefix KV, so the pool's used counter here is the weights
        // (set_bs_planes above included). `settled_` owns the sync-then-trim
        // that keeps freed repack staging out of the number.
        me.weights_bytes = me.exec.settled_mem_used();
        if let (Some(b), Some(t)) = (base_bytes, me.weights_bytes) {
            let gb = |v: u64| v as f64 / 1e9;
            if t > b {
                tracing::info!(
                    "qwen3 VRAM  base planes {:.2} GB + bs decode/batch duplicates \
                     (_mx; Q8 originals stay for rows <= {}) {:.2} GB = {:.2} GB resident",
                    gb(b),
                    BS_MIN_BATCH,
                    gb(t - b),
                    gb(t)
                );
            } else {
                tracing::info!(
                    "qwen3 VRAM  resident weights {:.2} GB (no bs planes)",
                    gb(t)
                );
            }
        }
        Ok(me)
    }

    /// Resident weight bytes - see the `weights_bytes` field. Encoder lanes
    /// have no `Generator`, so the runner reads this through `Encoder`.
    pub fn weights_mem_bytes(&self) -> Option<u64> {
        self.weights_bytes
    }

    /// Device bytes this process holds live (weights + scratch + pooled
    /// planes + prefix KV) - the encoder's `model_mem` line.
    pub fn device_mem_used(&self) -> Option<u64> {
        self.exec.process_mem_used()
    }

    /// (Re)quantize every layer's block-scale planes to the classes in
    /// `bs_gu`/`bs_dn`/`bs_attn`/`bs_wo`, replacing whatever was there.
    fn set_bs_planes(&mut self) -> Result<(), GpuModelError> {
        self.set_lane(0);
        // The startup calibration re-enters this with different classes, and
        // F8/Nv4/W8 planes are not the same size - so track the resident-weight
        // line by DELTA rather than re-reading it absolutely. By then scratch
        // and pooled planes exist and an absolute read would swallow them;
        // nothing but the planes moves across this call, so the delta is
        // exactly the weight change. None during load (the end-of-load stamp
        // takes it absolutely instead).
        let before = self.weights_bytes.and_then(|_| self.exec.synced_mem_used());
        let exec = self.exec.clone();
        self.bs_epoch += 1; // prefix-KV snapshots predate the new planes
        let (gu, dn, attn, wo) = (self.bs_gu, self.bs_dn, self.bs_attn, self.bs_wo);
        for l in &mut self.layers {
            let sm = l.smooth.as_ref();
            let mk = |w: &RepackedQ8,
                      c: BsClass,
                      svec: Option<&CudaSlice<f32>>,
                      svec8: Option<&CudaSlice<f32>>|
             -> Result<RepackedMxfp4, GpuModelError> {
                Ok(match c {
                    BsClass::Nv4Smooth => exec.q8_0_to_nvf4_smooth(
                        w,
                        svec.ok_or(crate::gpu::GpuError::MissingOp("smooth stats"))?,
                    )?,
                    BsClass::F8Smooth => exec.q8_0_to_mxfp4_smooth(
                        w,
                        svec8.ok_or(crate::gpu::GpuError::MissingOp("smooth stats"))?,
                    )?,
                    BsClass::Nv4Rot => exec.q8_0_to_nvf4_rot(w)?,
                    BsClass::Nv4 => exec.q8_0_to_nvf4(w)?,
                    BsClass::F4 => exec.q8_0_to_fp4p(w)?,
                    BsClass::F8 => exec.q8_0_to_mxfp4(w)?,
                    BsClass::W8 => exec.q8_0_to_f8w(w)?,
                })
            };
            l.ffn_gate_mx = gu
                .map(|c| mk(&l.ffn_gate, c, sm.map(|m| &m.gu_s), sm.map(|m| &m.gu_s8)))
                .transpose()?;
            l.ffn_up_mx = gu
                .map(|c| mk(&l.ffn_up, c, sm.map(|m| &m.gu_s), sm.map(|m| &m.gu_s8)))
                .transpose()?;
            l.ffn_down_mx = Some(mk(
                &l.ffn_down,
                dn,
                sm.map(|m| &m.dn_s),
                sm.map(|m| &m.dn_s8),
            )?);
            if let Some(wqkv) = &l.wqkv {
                l.wqkv_mx = attn
                    .map(|c| mk(wqkv, c, sm.map(|m| &m.attn_s), sm.map(|m| &m.attn_s8)))
                    .transpose()?;
            } else {
                l.wq_mx = attn
                    .map(|c| {
                        mk(
                            l.wq.as_ref().expect("unfused"),
                            c,
                            sm.map(|m| &m.attn_s),
                            sm.map(|m| &m.attn_s8),
                        )
                    })
                    .transpose()?;
                l.wk_mx = attn
                    .map(|c| {
                        mk(
                            l.wk.as_ref().expect("unfused"),
                            c,
                            sm.map(|m| &m.attn_s),
                            sm.map(|m| &m.attn_s8),
                        )
                    })
                    .transpose()?;
                l.wv_mx = attn
                    .map(|c| {
                        mk(
                            l.wv.as_ref().expect("unfused"),
                            c,
                            sm.map(|m| &m.attn_s),
                            sm.map(|m| &m.attn_s8),
                        )
                    })
                    .transpose()?;
            }
            l.wo_mx = wo
                .map(|c| mk(&l.wo, c, sm.map(|m| &m.wo_s), sm.map(|m| &m.wo_s8)))
                .transpose()?;
        }
        if !self.lanes_extra.is_empty() {
            // the requants ran on lane 0; drain it so lane 1's next batch
            // never reads half-built planes (event tracking is off)
            self.exec.synchronize()?;
        }
        if let (Some(b), Some(a), Some(w)) =
            (before, self.exec.synced_mem_used(), self.weights_bytes)
        {
            self.weights_bytes = Some(w.saturating_add(a).saturating_sub(b));
        }
        Ok(())
    }

    /// Allocate zeroed per-layer activation-stat accumulators and turn the
    /// run_layers collection hooks on.
    fn begin_smooth_stats(&mut self) -> Result<(), GpuModelError> {
        let exec = self.exec.clone();
        let (embd, q_dim) = (self.embd, self.n_heads * self.head_dim);
        let zeros_e = vec![0f32; embd];
        let zeros_q = vec![0f32; q_dim];
        let zeros_f = vec![0f32; self.ff];
        let mut bufs = Vec::with_capacity(self.n_layers);
        for _ in 0..self.n_layers {
            bufs.push(StatBufs {
                attn: exec.to_device(&zeros_e)?,
                gu: exec.to_device(&zeros_e)?,
                wo: exec.to_device(&zeros_q)?,
                dn: exec.to_device(&zeros_f)?,
            });
        }
        self.stat_bufs = Some(bufs);
        Ok(())
    }

    /// Turn the collected activation stats + weight column maxima into the
    /// per-layer SmoothQuant vectors (s = sqrt(a_max/w_max), alpha = 0.5,
    /// clamped) and drop the accumulators.
    fn build_smooth(&mut self) -> Result<(), GpuModelError> {
        let exec = self.exec.clone();
        let (embd, q_dim, ff) = (self.embd, self.n_heads * self.head_dim, self.ff);
        let Some(stats) = self.stat_bufs.take() else {
            return Ok(());
        };
        let zeros_e = vec![0f32; embd];
        let zeros_q = vec![0f32; q_dim];
        let zeros_f = vec![0f32; ff];
        // Per-SITE alpha, chosen from a small grid by a local proxy for
        // per-16-block quantization difficulty: after migrating with
        // s = a^alpha / w^(1-alpha), each block's damage tracks how far its
        // hottest channel sits above the block's typical magnitude, on both
        // sides - J(alpha) = mean_block[max/rms](a') + mean_block[max/rms](w').
        // Free (uses only the collected range vectors, no extra encodes);
        // the recall gate still owns acceptance.
        let mk = |amax: &[f32], wmax: &[f32], grid: &[f32], act_w: f64| -> (Vec<f32>, Vec<f32>) {
            let spread = |v: &[f32]| -> f64 {
                let mut j = 0f64;
                let nb = v.len() / 16;
                for b in 0..nb {
                    let blk = &v[b * 16..(b + 1) * 16];
                    let mx = blk.iter().cloned().fold(0f32, f32::max) as f64;
                    let ms = blk.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>() / 16.0;
                    if ms > 0.0 {
                        j += mx / ms.sqrt();
                    }
                }
                j / nb.max(1) as f64
            };
            let mut best = (f64::INFINITY, 0.5f32);
            for &alpha in grid {
                let a1: Vec<f32> = amax
                    .iter()
                    .zip(wmax)
                    .map(|(&a, &w)| {
                        if a > 0.0 && w > 0.0 {
                            a / (a.powf(alpha) / w.powf(1.0 - alpha)).clamp(1e-2, 1e3)
                        } else {
                            a
                        }
                    })
                    .collect();
                let w1: Vec<f32> = amax
                    .iter()
                    .zip(wmax)
                    .map(|(&a, &w)| {
                        if a > 0.0 && w > 0.0 {
                            w * (a.powf(alpha) / w.powf(1.0 - alpha)).clamp(1e-2, 1e3)
                        } else {
                            w
                        }
                    })
                    .collect();
                // weight the activation side by its format's relative step
                // size (e4m3 is ~4x finer than e2m1)
                let j = act_w * spread(&a1) + spread(&w1);
                if j < best.0 {
                    best = (j, alpha);
                }
            }
            let alpha = best.1;
            let mut sv = Vec::with_capacity(amax.len());
            let mut si = Vec::with_capacity(amax.len());
            for (&a, &w) in amax.iter().zip(wmax) {
                let s = if a > 0.0 && w > 0.0 {
                    (a.powf(alpha) / w.powf(1.0 - alpha)).clamp(1e-2, 1e3)
                } else {
                    1.0
                };
                sv.push(s);
                si.push(1.0 / s);
            }
            (sv, si)
        };
        for (li, st) in stats.into_iter().enumerate() {
            // weight column maxima, shared per site across its consumers
            let mut w_attn = exec.to_device(&zeros_e)?;
            let mut w_gu = exec.to_device(&zeros_e)?;
            let mut w_wo = exec.to_device(&zeros_q)?;
            let mut w_dn = exec.to_device(&zeros_f)?;
            {
                let l = &self.layers[li];
                if let Some(wqkv) = &l.wqkv {
                    // fused plane: one pass is the max over q/k/v rows
                    exec.q8_0_col_absmax(wqkv, &mut w_attn)?;
                } else {
                    exec.q8_0_col_absmax(l.wq.as_ref().expect("unfused"), &mut w_attn)?;
                    exec.q8_0_col_absmax(l.wk.as_ref().expect("unfused"), &mut w_attn)?;
                    exec.q8_0_col_absmax(l.wv.as_ref().expect("unfused"), &mut w_attn)?;
                }
                exec.q8_0_col_absmax(&l.ffn_gate, &mut w_gu)?;
                exec.q8_0_col_absmax(&l.ffn_up, &mut w_gu)?;
                exec.q8_0_col_absmax(&l.wo, &mut w_wo)?;
                exec.q8_0_col_absmax(&l.ffn_down, &mut w_dn)?;
            }
            let (a_attn, a_gu, a_wo, a_dn) = (
                exec.to_host(&st.attn)?,
                exec.to_host(&st.gu)?,
                exec.to_host(&st.wo)?,
                exec.to_host(&st.dn)?,
            );
            let (wa, wg, ww, wd) = (
                exec.to_host(&w_attn)?,
                exec.to_host(&w_gu)?,
                exec.to_host(&w_wo)?,
                exec.to_host(&w_dn)?,
            );
            // fp4-activation set: both sides coarse, alpha near balance
            const G_NV4: [f32; 4] = [0.35, 0.5, 0.65, 0.8];
            // e4m3-activation set: weight-ward alphas (the fp4 weights are
            // the coarse side; alpha ~ 0 normalizes their columns)
            const G_F8: [f32; 4] = [0.0, 0.15, 0.3, 0.5];
            let (attn_s, attn_sinv) = mk(&a_attn, &wa, &G_NV4, 1.0);
            let (gu_s, gu_sinv) = mk(&a_gu, &wg, &G_NV4, 1.0);
            let (wo_s, wo_sinv) = mk(&a_wo, &ww, &G_NV4, 1.0);
            let (dn_s, dn_sinv) = mk(&a_dn, &wd, &G_NV4, 1.0);
            let (attn_s8, attn_s8inv) = mk(&a_attn, &wa, &G_F8, 0.25);
            let (gu_s8, gu_s8inv) = mk(&a_gu, &wg, &G_F8, 0.25);
            let (wo_s8, wo_s8inv) = mk(&a_wo, &ww, &G_F8, 0.25);
            let (dn_s8, dn_s8inv) = mk(&a_dn, &wd, &G_F8, 0.25);
            self.layers[li].smooth = Some(SmoothVecs {
                attn_s: exec.to_device(&attn_s)?,
                attn_sinv: exec.to_device(&attn_sinv)?,
                gu_s: exec.to_device(&gu_s)?,
                gu_sinv: exec.to_device(&gu_sinv)?,
                wo_s: exec.to_device(&wo_s)?,
                wo_sinv: exec.to_device(&wo_sinv)?,
                dn_s: exec.to_device(&dn_s)?,
                dn_sinv: exec.to_device(&dn_sinv)?,
                attn_s8: exec.to_device(&attn_s8)?,
                attn_s8inv: exec.to_device(&attn_s8inv)?,
                gu_s8: exec.to_device(&gu_s8)?,
                gu_s8inv: exec.to_device(&gu_s8inv)?,
                wo_s8: exec.to_device(&wo_s8)?,
                wo_s8inv: exec.to_device(&wo_s8inv)?,
                dn_s8: exec.to_device(&dn_s8)?,
                dn_s8inv: exec.to_device(&dn_s8inv)?,
            });
        }
        Ok(())
    }

    /// Load-time quality calibration on real text (the caller - the server,
    /// which owns a tokenizer - supplies the tokenized retrieval task:
    /// distractor docs then paraphrase queries with known source docs).
    /// FP4 tolerance is model-specific (measured: the 8B encoder holds
    /// recall with nvf4 on the hidden's consumers and wo, the 4B does not),
    /// so instead of hard-coding the most conservative class mix, walk the
    /// profile ladder fastest-first and keep the first whose recall@1 and
    /// recall@10 hold at-or-above the pure-Q8_0 baseline - the same
    /// acceptance rule as the offline qwen3_embed_quality gate. Encodes are
    /// deterministic, so the verdict is stable per (model, task). No-op
    /// (keeps the static default) when classes were env-pinned or the
    /// block-scale route is off. Returns the selected profile name.
    ///
    /// NOTE: a synthetic token-space task (random-token docs, no tokenizer
    /// needed) was tried first and measured USELESS as a verdict source:
    /// random-token activations do not reproduce real-text outlier
    /// structure, and its verdicts contradicted the semantic gate on both
    /// 4B (over-strict) and 8B (rejected everything).
    pub fn calibrate_bs(
        &mut self,
        seqs: &[Vec<u32>],
        n_docs: usize,
        rel: &[usize],
    ) -> Result<&'static str, GpuModelError> {
        let seqs = seqs.to_vec();
        let rel = rel.to_vec();
        self.calibrate_ladder("recall@1", "recall@10", rel.len(), move |m| {
            m.calib_recall(&seqs, n_docs, &rel)
        })
    }

    /// Reranker twin of [`calibrate_bs`]: the metric is the yes/no judge's
    /// RANKING quality, not embedding recall (a reranker's embedding space
    /// is not a retrieval space - measured recall@1 0/64 there, pure
    /// noise). `seqs` holds `rel.len()` groups of `group` scored prompts;
    /// per query, count the known-relevant doc at rank 1 and in the top 3.
    pub fn calibrate_bs_rerank(
        &mut self,
        seqs: &[Vec<u32>],
        yes: u32,
        no: u32,
        group: usize,
        rel: &[usize],
    ) -> Result<&'static str, GpuModelError> {
        let seqs = seqs.to_vec();
        let rel = rel.to_vec();
        self.calibrate_ladder("rank1", "rank3", rel.len(), move |m| {
            let scores = m.rerank(&seqs, yes, no)?;
            let (mut r1, mut r3) = (0u32, 0u32);
            for (qi, &target) in rel.iter().enumerate() {
                let g = &scores[qi * group..(qi + 1) * group];
                let mut order: Vec<usize> = (0..group).collect();
                order.sort_by(|&a, &b| g[b].total_cmp(&g[a]));
                if order[0] == target {
                    r1 += 1;
                }
                if order[..3].contains(&target) {
                    r3 += 1;
                }
            }
            Ok((r1, r3))
        })
    }

    /// The shared ladder walk: measure the metric on a pure-Q8_0 baseline,
    /// then keep the fastest profile whose both metric components hold
    /// at-or-above it.
    fn calibrate_ladder<F>(
        &mut self,
        m1: &str,
        m2: &str,
        q: usize,
        mut measure: F,
    ) -> Result<&'static str, GpuModelError>
    where
        F: FnMut(&mut Self) -> Result<(u32, u32), GpuModelError>,
    {
        self.set_lane(0);
        if self.bs_pinned || self.layers.first().is_none_or(|l| l.ffn_down_mx.is_none()) {
            return Ok("pinned");
        }
        // baseline: drop the planes -> the pure Q8_0 path
        let saved = (self.bs_gu, self.bs_dn, self.bs_attn, self.bs_wo);
        self.bs_epoch += 1;
        for l in &mut self.layers {
            l.ffn_gate_mx = None;
            l.ffn_up_mx = None;
            l.ffn_down_mx = None;
            l.wqkv_mx = None;
            l.wq_mx = None;
            l.wk_mx = None;
            l.wv_mx = None;
            l.wo_mx = None;
        }
        // the baseline encode doubles as the SmoothQuant statistics pass
        let want_smooth = self.exec.has_nvf4_smooth();
        if want_smooth {
            self.begin_smooth_stats()?;
        }
        let (b1, b2) = measure(self)?;
        if want_smooth {
            self.build_smooth()?;
        }
        self.stat_bufs = None;
        for (name, gu, dn, attn, wo) in BS_LADDER {
            if matches!(gu, Some(BsClass::Nv4Smooth | BsClass::F8Smooth))
                && self.layers.first().is_none_or(|l| l.smooth.is_none())
            {
                continue; // no stats -> the smooth rungs cannot be built
            }
            if (matches!(gu, Some(BsClass::W8)) || matches!(dn, BsClass::W8))
                && !self.exec.has_f8_gemm_w8()
            {
                continue; // pack lacks the W8A8 kernels
            }
            (self.bs_gu, self.bs_dn, self.bs_attn, self.bs_wo) = (gu, dn, attn, wo);
            self.set_bs_planes()?;
            let (r1, r2) = measure(self)?;
            let pass = r1 >= b1 && r2 >= b2;
            tracing::info!(
                "paddock: bs calibration '{name}': {m1} {r1}/{q} vs baseline {b1}, \
                 {m2} {r2}/{q} vs {b2} -> {}",
                if pass { "selected" } else { "rejected" },
            );
            if pass {
                return Ok(name);
            }
        }
        // floor: no block-scale planes at all (pure Q8_0)
        tracing::info!("paddock: bs calibration: every profile rejected, staying on Q8_0");
        (self.bs_gu, self.bs_dn, self.bs_attn, self.bs_wo) = saved;
        self.bs_epoch += 1;
        for l in &mut self.layers {
            l.ffn_gate_mx = None;
            l.ffn_up_mx = None;
            l.ffn_down_mx = None;
            l.wqkv_mx = None;
            l.wq_mx = None;
            l.wk_mx = None;
            l.wv_mx = None;
            l.wo_mx = None;
        }
        Ok("q8_0")
    }

    /// Serialize the per-layer SmoothQuant s-vectors (attn, gu, wo, dn per
    /// layer, f32 LE, dims header) for the caller's verdict cache - a warm
    /// start can then apply a smooth profile without re-running the
    /// statistics encode. None until a calibration built them.
    pub fn export_smooth(&self) -> Result<Option<Vec<u8>>, GpuModelError> {
        if self.layers.iter().any(|l| l.smooth.is_none()) {
            return Ok(None);
        }
        let (embd, q_dim, ff) = (self.embd, self.n_heads * self.head_dim, self.ff);
        let mut out = Vec::with_capacity(16 + self.n_layers * 2 * (2 * embd + q_dim + ff) * 4);
        for d in [self.n_layers as u32, embd as u32, q_dim as u32, ff as u32] {
            out.extend_from_slice(&d.to_le_bytes());
        }
        for l in &self.layers {
            let sm = l.smooth.as_ref().unwrap();
            for buf in [
                &sm.attn_s,
                &sm.gu_s,
                &sm.wo_s,
                &sm.dn_s,
                &sm.attn_s8,
                &sm.gu_s8,
                &sm.wo_s8,
                &sm.dn_s8,
            ] {
                for v in self.exec.to_host(buf)? {
                    out.extend_from_slice(&v.to_le_bytes());
                }
            }
        }
        Ok(Some(out))
    }

    /// Load cached SmoothQuant s-vectors (the `export_smooth` format).
    /// Returns false - leaving the model untouched - on any dimension or
    /// value mismatch; the caller then recalibrates.
    pub fn import_smooth(&mut self, bytes: &[u8]) -> Result<bool, GpuModelError> {
        self.set_lane(0);
        let (embd, q_dim, ff) = (self.embd, self.n_heads * self.head_dim, self.ff);
        if bytes.len() < 16 {
            return Ok(false);
        }
        let rd = |o: usize| {
            u32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]) as usize
        };
        if rd(0) != self.n_layers || rd(4) != embd || rd(8) != q_dim || rd(12) != ff {
            return Ok(false);
        }
        let per = 2 * (2 * embd + q_dim + ff);
        if bytes.len() != 16 + self.n_layers * per * 4 {
            return Ok(false);
        }
        let exec = self.exec.clone();
        let mut off = 16usize;
        let mut read_vec = |n: usize| -> Option<Vec<f32>> {
            let v: Vec<f32> = bytes[off..off + n * 4]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            off += n * 4;
            v.iter().all(|x| x.is_finite() && *x > 0.0).then_some(v)
        };
        let mut vecs = Vec::with_capacity(self.n_layers);
        for _ in 0..self.n_layers {
            let (Some(a), Some(g), Some(w), Some(d)) = (
                read_vec(embd),
                read_vec(embd),
                read_vec(q_dim),
                read_vec(ff),
            ) else {
                return Ok(false);
            };
            let (Some(a8), Some(g8), Some(w8), Some(d8)) = (
                read_vec(embd),
                read_vec(embd),
                read_vec(q_dim),
                read_vec(ff),
            ) else {
                return Ok(false);
            };
            vecs.push((a, g, w, d, a8, g8, w8, d8));
        }
        let inv = |v: &[f32]| -> Vec<f32> { v.iter().map(|&x| 1.0 / x).collect() };
        for (li, (a, g, w, d, a8, g8, w8, d8)) in vecs.into_iter().enumerate() {
            self.layers[li].smooth = Some(SmoothVecs {
                attn_s: exec.to_device(&a)?,
                attn_sinv: exec.to_device(&inv(&a))?,
                gu_s: exec.to_device(&g)?,
                gu_sinv: exec.to_device(&inv(&g))?,
                wo_s: exec.to_device(&w)?,
                wo_sinv: exec.to_device(&inv(&w))?,
                dn_s: exec.to_device(&d)?,
                dn_sinv: exec.to_device(&inv(&d))?,
                attn_s8: exec.to_device(&a8)?,
                attn_s8inv: exec.to_device(&inv(&a8))?,
                gu_s8: exec.to_device(&g8)?,
                gu_s8inv: exec.to_device(&inv(&g8))?,
                wo_s8: exec.to_device(&w8)?,
                wo_s8inv: exec.to_device(&inv(&w8))?,
                dn_s8: exec.to_device(&d8)?,
                dn_s8inv: exec.to_device(&inv(&d8))?,
            });
        }
        Ok(true)
    }

    /// Apply a previously-calibrated profile by name (the disk-cache fast
    /// path - skips the calibration encodes entirely). Returns false when
    /// the profile is unknown, classes are env-pinned, or the block-scale
    /// route is off - the caller should then fall back to `calibrate_bs`.
    pub fn apply_bs_profile(&mut self, profile: &str) -> Result<bool, GpuModelError> {
        self.set_lane(0);
        if self.bs_pinned || self.layers.first().is_none_or(|l| l.ffn_down_mx.is_none()) {
            return Ok(false);
        }
        if profile == "q8_0" {
            self.bs_epoch += 1;
            for l in &mut self.layers {
                l.ffn_gate_mx = None;
                l.ffn_up_mx = None;
                l.ffn_down_mx = None;
                l.wqkv_mx = None;
                l.wq_mx = None;
                l.wk_mx = None;
                l.wv_mx = None;
                l.wo_mx = None;
            }
            return Ok(true);
        }
        for (name, gu, dn, attn, wo) in BS_LADDER {
            if name == profile {
                if matches!(gu, Some(BsClass::Nv4Smooth | BsClass::F8Smooth))
                    && self.layers.first().is_none_or(|l| l.smooth.is_none())
                {
                    // smooth planes need the calibration statistics - the
                    // caller falls back to calibrate_bs (whose ladder tests
                    // this rung first anyway, so the cost is ~two encodes)
                    return Ok(false);
                }
                if (matches!(gu, Some(BsClass::W8)) || matches!(dn, BsClass::W8))
                    && !self.exec.has_f8_gemm_w8()
                {
                    return Ok(false); // pack lacks the W8A8 kernels
                }
                if (self.bs_gu, self.bs_dn, self.bs_attn, self.bs_wo) == (gu, dn, attn, wo) {
                    return Ok(true); // already the static default's planes
                }
                (self.bs_gu, self.bs_dn, self.bs_attn, self.bs_wo) = (gu, dn, attn, wo);
                self.set_bs_planes()?;
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Encode the calibration task with the CURRENT classes/planes and score
    /// retrieval: (recall@1, recall@10) of each query's known source doc.
    fn calib_recall(
        &mut self,
        seqs: &[Vec<u32>],
        n_docs: usize,
        rel: &[usize],
    ) -> Result<(u32, u32), GpuModelError> {
        let embs = self.embed(seqs)?;
        let (mut r1, mut r10) = (0u32, 0u32);
        for (qi, &target) in rel.iter().enumerate() {
            let qe = &embs[n_docs + qi];
            let mut scored: Vec<(f32, usize)> = (0..n_docs)
                .map(|d| (qe.iter().zip(&embs[d]).map(|(a, b)| a * b).sum::<f32>(), d))
                .collect();
            scored.sort_by(|a, b| b.0.total_cmp(&a.0));
            if scored[0].1 == target {
                r1 += 1;
            }
            if scored[..10].iter().any(|&(_, d)| d == target) {
                r10 += 1;
            }
        }
        Ok((r1, r10))
    }

    /// (Re)allocate encode scratch + per-layer KV for `rows` total tokens
    /// across `slots` sequences with a `kv_stride`-token KV cache. Grows
    /// monotonically (cached across calls).
    fn ensure_scratch(
        &mut self,
        rows: usize,
        slots: usize,
        kv_stride: usize,
    ) -> Result<(), GpuModelError> {
        // Quantize the requested shape up to coarse buckets: serving batches
        // jitter a few percent request to request (ragged tokenization,
        // varying merge widths), and a multi-GB scratch rebuild costs
        // ~100+ ms - measured as a permanent ~30% serving tax when every
        // request re-allocated. With grow-only reuse, bucketing means the
        // high-water mark converges after a handful of rebuilds instead of
        // thrashing on every row-count wiggle.
        let rows = rows.next_multiple_of(8192);
        // Slots bucketed to a POWER of two, not to 128. Rounding a slot count
        // up to 128 is not the same class of approximation as rounding rows up
        // to 8192: rows is a sum of token counts and lands in the thousands,
        // while slots is a sequence count that starts at 1 and is bounded by
        // the scheduler's max batch - so the 128-multiple was a 64x overshoot
        // on the common single-text request, and every one of those phantom
        // slots carries a full kv_stride of K and V in every layer:
        //
        //     128 slots x 2048 stride x 1024 kv_dim x 2 B = 537 MB / layer / tensor
        //                                    x2 x36 layers = 38.7 GB
        //
        // which is how a 2003-token embedding request took the pool from 10.7
        // to 70.4 GB, and a 4000-token one asked for more than the card has
        // (measured cold, walking the length up: 253/503/1003/2003 tokens
        // fine, ~4003 dead with CUDA_ERROR_OUT_OF_MEMORY on a 96 GB card).
        // A power of two keeps the anti-thrash bucketing the comment above
        // wants - the high-water still converges in a handful of steps - while
        // bounding the overshoot at 2x instead of 64x.
        let slots = slots.next_power_of_two().max(8);
        // the fused-path planes are part of the shape: a scratch built before
        // calibration/env selection enabled them must be rebuilt
        if matches!(&self.scratch, Some(sc) if sc.rows >= rows && sc.slots >= slots
            && sc.kv_stride >= kv_stride && (!self.bs_gu_fused || sc.d_fq.is_some()))
        {
            return Ok(());
        }
        // a rebuild moves every scratch pointer the current lane's graphs
        // baked - drop them (they re-capture on the next encode)
        self.graphs.clear();
        // ...and drop the old scratch before allocating the new one, so a grow
        // does not need both resident at once. This was not what caused the
        // OOM above - that was the slot bucket, and this change alone did not
        // move it - but it halves the transient of every rebuild and lets a
        // failed grow retry against the full budget instead of half of it.
        self.scratch = None;
        let e = &self.exec;
        let (embd, ff, head_dim) = (self.embd, self.ff, self.head_dim);
        let q_dim = self.n_heads * head_dim;
        let kv_dim = self.n_kv_heads * head_dim;
        let kv_bytes = self.kv_dtype.bytes();
        let wide = ff.max(embd);
        let kv_slot = slots * kv_stride * kv_dim * kv_bytes;
        let (mut kv_k, mut kv_v) = (
            Vec::with_capacity(self.n_layers),
            Vec::with_capacity(self.n_layers),
        );
        for _ in 0..self.n_layers {
            kv_k.push(e.alloc_u8(kv_slot)?);
            kv_v.push(e.alloc_u8(kv_slot)?);
        }
        self.scratch = Some(EncScratch {
            rows,
            slots,
            kv_stride,
            d_x: e.alloc(rows * embd)?,
            d_xn: e.alloc(rows * embd)?,
            d_q: e.alloc(rows * q_dim)?,
            d_qkv: e.alloc(rows * (q_dim + 2 * kv_dim))?,
            up_tokens: e.alloc_u32(rows)?,
            up_pos: e.alloc_u32(rows)?,
            up_slots: e.alloc_u32(rows)?,
            up_trow0: e.alloc_u32(rows.div_ceil(16) + slots)?,
            up_tslot: e.alloc_u32(rows.div_ceil(16) + slots)?,
            up_idx: e.alloc_u32(slots)?,
            up_descs: e.alloc_u64(self.n_layers * slots * 6)?,
            d_skpart: e.alloc((e.sm_count() / 2 + 1) * 4 * 128 * 128)?,
            d_k: e.alloc(rows * kv_dim)?,
            d_v: e.alloc(rows * kv_dim)?,
            d_qn: e.alloc(rows * q_dim)?,
            d_kn: e.alloc(rows * kv_dim)?,
            d_attn: e.alloc(rows * q_dim)?,
            d_proj: e.alloc(rows * embd)?,
            d_ffn_gate: e.alloc(rows * ff)?,
            d_ffn_up: e.alloc(rows * ff)?,
            d_h: e.alloc(rows * embd)?,
            d_pxq: e.alloc_i8(rows * wide)?,
            d_pxs: e.alloc(rows * wide / 32)?,
            d_yq: e.alloc_u8(wide.div_ceil(128) * rows.next_multiple_of(128) * 144)?,
            d_skfix: e.alloc(256 * 128 * 128 + 256)?, // +256: stream-k fold flags tail
            // sized for the finest plane (nvf4: one byte per 16 elems);
            // unconditional - the load-time calibration flips block-scale
            // planes on after its baseline encode built the scratch
            d_exs: Some(e.alloc_u8(rows * wide / 16)?),
            d_fq: if self.bs_gu_fused {
                Some(e.alloc_i8(rows * wide / 2)?)
            } else {
                None
            },
            d_fs: if self.bs_gu_fused {
                Some(e.alloc_u8(rows * wide / 16)?)
            } else {
                None
            },
            kv_k,
            kv_v,
        });
        Ok(())
    }

    /// Ragged batched prefill: run every sequence through the tuned prefill
    /// stack in one pass (per-slot KV isolation), returning each sequence's
    /// last-token hidden after `output_norm` - the pooled representation both
    /// the embedding (normalize) and reranker (lm_head) heads build on.
    /// Returns a flat `[slots][embd]`.
    /// Run embed + all transformer layers over `rows` tokens laid out by the
    /// per-row (token, position, slot) triples, appending each layer's K/V into
    /// the per-slot cache and leaving the final pre-`output_norm` hidden in
    /// scratch `d_x`. Scratch must already be sized (`ensure_scratch`). Shared
    /// by [`encode_pooled`] and the prefix-cached path.
    fn run_layers(
        &mut self,
        tokens: &[u32],
        positions: &[u32],
        slot_of: &[u32],
        rows: usize,
    ) -> Result<(), GpuModelError> {
        let exec = self.exec.clone();
        // uploads land in persistent scratch buffers - per-encode transient
        // allocs are stream-ordered pool ops whose cross-stream reuse
        // ordering serializes the dual-lane streams
        {
            let sc = self.scratch.as_mut().expect("scratch");
            exec.upload_u32(tokens, &mut sc.up_tokens)?;
            exec.upload_u32(positions, &mut sc.up_pos)?;
            exec.upload_u32(slot_of, &mut sc.up_slots)?;
        }
        let (head_dim, kv_dtype) = (self.head_dim, self.kv_dtype);

        // Fast attention: the tiled multi-slot flash pass (10-20x the decode
        // class for prefill) instead of attn_decode_batch. Tile each contiguous
        // slot-run so no query tile crosses a slot boundary; rows spilled past
        // a text are masked by slots[b]==slot. Two tiers: the WMMA f16 kernel
        // (32-row tiles; tensor-core QK^T/AV, the scalar kernel ran at ~11% of
        // f32 peak and dominated the reranker forward) when the pack has it
        // and the KV cache is fp16, else the scalar f32 kernel (16-row tiles).
        // PADDOCK_ATTN_SCALAR=1 pins the scalar path for A/B checks. Falls
        // back to the decode kernel if the pack lacks both or head_dim != 128.
        let use_flash = head_dim == 128 && exec.has_attn_prefill_batch();
        let use_flash_f16 = head_dim == 128
            && exec.has_attn_prefill_batch_f16()
            && kv_dtype == KvDtype::Fp16
            && self
                .scratch
                .as_ref()
                .is_some_and(|sc| sc.kv_stride % 64 == 0)
            && paddock_models::dev_var_os!("PADDOCK_ATTN_SCALAR").is_none();
        let tile_q = if use_flash_f16 { 32 } else { 16 };
        let n_qtiles = if use_flash || use_flash_f16 {
            let mut trow0 = Vec::new();
            let mut tslot = Vec::new();
            let mut i = 0usize;
            while i < rows {
                let s = slot_of[i];
                let mut j = i;
                while j < rows && slot_of[j] == s {
                    j += 1;
                }
                let mut r0 = i;
                while r0 < j {
                    trow0.push(r0 as u32);
                    tslot.push(s);
                    r0 += tile_q;
                }
                i = j;
            }
            let n = trow0.len();
            let sc = self.scratch.as_mut().expect("scratch");
            exec.upload_u32(&trow0, &mut sc.up_trow0)?;
            exec.upload_u32(&tslot, &mut sc.up_tslot)?;
            n
        } else {
            0
        };
        let have_tiles = n_qtiles > 0;
        let _ = have_tiles;

        // CUDA-graph replay - MEASURED NEGATIVE on this encoder, kept
        // reproducible behind PADDOCK_GRAPH=1. The machinery works (bit-
        // identical replays; rows pad to 128 into a reserved garbage slot,
        // the tile array pads with dead sentinels the tile kernels early-
        // exit on, entries invalidate on scratch rebuild / bs_epoch, and
        // capture waits for a shape to recur 3x). But the encoder's
        // pipelined submission already hides all host launch cost - even
        // one-request latency measured no win (rerank B=1 p50 16.5 ->
        // 17.2 ms, p90 17.0 -> 23.4 ms from capture stalls; C=16,B=8
        // serving median dropped ~5%). Graphs pay on decode loops with
        // host round-trips per step, not on a prefill path whose launches
        // ride ahead of a busy GPU.
        let timing = paddock_models::dev_var_os!("PADDOCK_TIME_LAYERS").is_some();
        let graphable = !self.stat_bufs.is_some()
            && !timing
            && (use_flash || use_flash_f16)
            && paddock_models::dev_var_os!("PADDOCK_GRAPH").is_some();
        if graphable {
            let bucket = rows.next_multiple_of(128);
            let garbage = slot_of.iter().copied().max().unwrap_or(0) as usize + 1;
            let (slots_alloc, kv_stride) = {
                let sc = self.scratch.as_ref().expect("scratch");
                (sc.slots, sc.kv_stride)
            };
            debug_assert!(garbage < slots_alloc, "callers reserve a spare slot");
            // tile bound: sum ceil(len/tq) <= rows/tq + n_texts (+1 pad run).
            // Bucketing the text count keeps the key stable across small
            // wobbles without over-padding the grid with dead sentinels.
            let slots_b = (garbage + 1).next_multiple_of(32).min(slots_alloc);
            let n_tiles_fixed = bucket / tile_q + slots_b + 1;
            // pad + re-upload the shape-varying planes (contents are graph
            // inputs; only pointers and grid dims are baked)
            {
                let mut t = tokens.to_vec();
                let mut pp = positions.to_vec();
                let mut sl = slot_of.to_vec();
                t.resize(bucket, 0);
                pp.resize(bucket, 0);
                sl.resize(bucket, garbage as u32);
                let sc = self.scratch.as_mut().expect("scratch");
                exec.upload_u32(&t, &mut sc.up_tokens)?;
                exec.upload_u32(&pp, &mut sc.up_pos)?;
                exec.upload_u32(&sl, &mut sc.up_slots)?;
            }
            {
                // rebuild tiles over the padded rows, then sentinel-pad
                let mut trow0 = Vec::with_capacity(n_tiles_fixed);
                let mut tslot = Vec::with_capacity(n_tiles_fixed);
                let mut i = 0usize;
                while i < rows {
                    let sslot = slot_of[i];
                    let mut j = i;
                    while j < rows && slot_of[j] == sslot {
                        j += 1;
                    }
                    let mut r0 = i;
                    while r0 < j {
                        trow0.push(r0 as u32);
                        tslot.push(sslot);
                        r0 += tile_q;
                    }
                    i = j;
                }
                let mut r0 = rows;
                while r0 < bucket {
                    trow0.push(r0 as u32);
                    tslot.push(garbage as u32);
                    r0 += tile_q;
                }
                trow0.resize(n_tiles_fixed, u32::MAX - 64);
                tslot.resize(n_tiles_fixed, 0);
                let sc = self.scratch.as_mut().expect("scratch");
                exec.upload_u32(&trow0, &mut sc.up_trow0)?;
                exec.upload_u32(&tslot, &mut sc.up_tslot)?;
            }
            let key = (bucket, n_tiles_fixed, kv_stride, tile_q);
            if let Some((epoch, g)) = self.graphs.get(&key) {
                if *epoch == self.bs_epoch {
                    g.0.launch()
                        .map_err(|e| GpuError::Driver(format!("graph launch: {e}")))?;
                    return Ok(());
                }
                self.graphs.remove(&key);
            }
            // capture only shapes that RECUR: a capture syncs the lane and
            // replays every launch through the capture machinery, so paying
            // it for a one-off batch shape is a net loss (measured: capture
            // storms during batch-size wobble erased the replay win)
            let hits = self.graph_hits.entry(key).or_insert(0);
            *hits += 1;
            if *hits < 3 {
                return self.record_layers(bucket, n_tiles_fixed, use_flash_f16);
            }
            // capture miss: record the padded shape into a fresh graph
            exec.stream
                .synchronize()
                .map_err(|e| GpuError::Driver(format!("pre-capture sync: {e}")))?;
            exec.stream
                .begin_capture(
                    cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
                )
                .map_err(|e| GpuError::Driver(format!("begin_capture: {e}")))?;
            let rec = self.record_layers(bucket, n_tiles_fixed, use_flash_f16);
            let graph = crate::gpu::end_capture_no_flags(&exec.stream)
                .map_err(|e| GpuError::Driver(format!("end_capture: {e}")));
            rec?; // surface a record failure only after capture cleanly ends
            let graph = graph?
                .ok_or_else(|| GpuError::Driver("encode capture produced no graph".into()))?;
            if self.graphs.len() >= 32 {
                self.graphs.clear();
            }
            graph
                .launch()
                .map_err(|e| GpuError::Driver(format!("graph launch: {e}")))?;
            self.graphs.insert(key, (self.bs_epoch, SendGraph(graph)));
            return Ok(());
        }
        self.record_layers(rows, n_qtiles, use_flash_f16)
    }

    /// The pure-launch encode body (embed gather + every layer): no host
    /// syncs, no allocations on the serving path - CUDA-graph capturable.
    /// `r`/`n_qtiles` may be the PADDED shape (see run_layers).
    fn record_layers(
        &mut self,
        r: usize,
        n_qtiles: usize,
        use_flash_f16: bool,
    ) -> Result<(), GpuModelError> {
        let exec = self.exec.clone();
        let (embd, ff, n_heads, n_kv_heads, head_dim) = (
            self.embd,
            self.ff,
            self.n_heads,
            self.n_kv_heads,
            self.head_dim,
        );
        let (eps, rope, kv_dtype) = (self.rms_eps, self.rope, self.kv_dtype);
        let kv_dim = n_kv_heads * head_dim;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let have_tiles = n_qtiles > 0;

        // disjoint field borrows (config already copied out)
        let stat_on = self.stat_bufs.is_some();
        let mut stat_bufs = self.stat_bufs.as_mut();
        // swiglu output is fused away on the serving paths - the stats pass
        // materializes it into a transient plane to measure the down site
        let mut stat_tmp = if stat_on {
            Some(exec.alloc(r * ff)?)
        } else {
            None
        };
        let layers = &self.layers;
        let sinks = &self.sinks;
        let tok_embd = &self.tok_embd;
        let sc = self.scratch.as_mut().expect("scratch");
        let kv_ctx = sc.kv_stride;

        exec.embed_gather_batch(&tok_embd.buf, &sc.up_tokens, &mut sc.d_x, embd, r)?;

        // PADDOCK_TIME_LAYERS=1: aggregate per-section GPU time across all
        // layers (sync-heavy; distorts wall time, reveals shares)
        let timing = paddock_models::dev_var_os!("PADDOCK_TIME_LAYERS").is_some();
        let mut sect = [0f64; 4]; // qkv, attention, wo, ffn
        let mut tmark = std::time::Instant::now();
        macro_rules! lap {
            ($i:expr) => {
                if timing {
                    let _ = exec.stream.synchronize();
                    sect[$i] += tmark.elapsed().as_secs_f64();
                    tmark = std::time::Instant::now();
                }
            };
        }

        for (li, layer) in layers.iter().enumerate() {
            // pre-attn norm + qkv projections: block-scale FP4 (e4m3
            // activations) at prefill batch, else the tuned Q8_0 stack.
            // Fused-QKV route first: one wide GEMM into d_qkv, then one
            // consumer kernel (q norm+rope packed, k norm+rope+scatter,
            // v scatter) - bit-exact with the separate route it replaces.
            let fused_qkv_done = if let Some(wqkv) = &layer.wqkv {
                let qkv_dim = n_heads * head_dim + 2 * kv_dim;
                if let (true, Some(qkvmx)) = (r > BS_MIN_BATCH, &layer.wqkv_mx) {
                    let exs = sc.d_exs.as_mut().expect("bs scratch");
                    exec.rmsnorm_batch(&sc.d_x, &layer.attn_norm.buf, &mut sc.d_xn, embd, eps, r)?;
                    match self.bs_attn {
                        Some(BsClass::F8Smooth) => {
                            let sm = layer.smooth.as_ref().expect("smooth vecs");
                            exec.quantize_e4m3_smooth(
                                &sc.d_xn,
                                &sm.attn_s8inv,
                                &mut sc.d_pxq,
                                exs,
                                r * embd,
                                embd,
                            )?;
                            exec.mxfp4_gemm_bs(
                                qkvmx,
                                &sc.d_pxq,
                                exs,
                                &mut sc.d_qkv,
                                embd,
                                qkv_dim,
                                r,
                            )?;
                        }
                        Some(BsClass::Nv4Smooth) => {
                            let sm = layer.smooth.as_ref().expect("smooth vecs");
                            exec.quantize_nvf4_smooth(
                                &sc.d_xn,
                                &sm.attn_sinv,
                                &mut sc.d_pxq,
                                exs,
                                r * embd,
                                embd,
                            )?;
                            exec.mxfp4_gemm_nv4(
                                qkvmx,
                                &sc.d_pxq,
                                exs,
                                &mut sc.d_qkv,
                                embd,
                                qkv_dim,
                                r,
                            )?;
                        }
                        Some(BsClass::Nv4) => {
                            exec.quantize_nvf4(&sc.d_xn, &mut sc.d_pxq, exs, r * embd)?;
                            exec.mxfp4_gemm_nv4(
                                qkvmx,
                                &sc.d_pxq,
                                exs,
                                &mut sc.d_qkv,
                                embd,
                                qkv_dim,
                                r,
                            )?;
                        }
                        Some(BsClass::Nv4Rot) => {
                            exec.quantize_nvf4_rot(&sc.d_xn, &mut sc.d_pxq, exs, r * embd)?;
                            exec.mxfp4_gemm_nv4(
                                qkvmx,
                                &sc.d_pxq,
                                exs,
                                &mut sc.d_qkv,
                                embd,
                                qkv_dim,
                                r,
                            )?;
                        }
                        Some(BsClass::W8) => {
                            exec.quantize_e4m3(&sc.d_xn, &mut sc.d_pxq, exs, r * embd)?;
                            exec.f8_gemm_w8(
                                qkvmx,
                                0,
                                &sc.d_pxq,
                                exs,
                                &mut sc.d_qkv,
                                embd,
                                qkv_dim,
                                r,
                            )?;
                        }
                        _ => {
                            exec.quantize_e4m3(&sc.d_xn, &mut sc.d_pxq, exs, r * embd)?;
                            exec.mxfp4_gemm_bs(
                                qkvmx,
                                &sc.d_pxq,
                                exs,
                                &mut sc.d_qkv,
                                embd,
                                qkv_dim,
                                r,
                            )?;
                        }
                    }
                } else {
                    prefill_add_norm_quant(
                        &exec,
                        &mut sc.d_x,
                        None,
                        false,
                        &layer.attn_norm.buf,
                        &mut sc.d_xn,
                        stat_on,
                        &mut sc.d_pxq,
                        &mut sc.d_pxs,
                        &mut sc.d_yq,
                        embd,
                        r,
                        eps,
                    )?;
                    prefill_mm_pre_sk(
                        &exec,
                        wqkv,
                        &sc.d_pxq,
                        &sc.d_pxs,
                        &sc.d_yq,
                        &mut sc.d_skfix,
                        Some(&mut sc.d_skpart),
                        &mut sc.d_qkv,
                        r,
                    )?;
                }
                if let Some(st) = stat_bufs.as_deref_mut() {
                    exec.col_absmax(&sc.d_xn, &mut st[li].attn, r, embd)?;
                }
                exec.qkv_norm_rope_append(
                    &sc.d_qkv,
                    &layer.q_norm.buf,
                    &layer.k_norm.buf,
                    &mut sc.d_qn,
                    &mut sc.kv_k[li],
                    &mut sc.kv_v[li],
                    &sc.up_pos,
                    Some(&sc.up_slots),
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    kv_ctx,
                    eps,
                    rope,
                    r,
                    kv_dtype,
                )?;
                true
            } else {
                false
            };
            if !fused_qkv_done {
                if let (true, Some(qmx), Some(kmx), Some(vmx)) =
                    (r > BS_MIN_BATCH, &layer.wq_mx, &layer.wk_mx, &layer.wv_mx)
                {
                    let exs = sc.d_exs.as_mut().expect("bs scratch");
                    exec.rmsnorm_batch(&sc.d_x, &layer.attn_norm.buf, &mut sc.d_xn, embd, eps, r)?;
                    match self.bs_attn {
                        Some(BsClass::F8Smooth) => {
                            let sm = layer.smooth.as_ref().expect("smooth vecs");
                            exec.quantize_e4m3_smooth(
                                &sc.d_xn,
                                &sm.attn_s8inv,
                                &mut sc.d_pxq,
                                exs,
                                r * embd,
                                embd,
                            )?;
                            exec.mxfp4_gemm_bs(
                                qmx,
                                &sc.d_pxq,
                                exs,
                                &mut sc.d_q,
                                embd,
                                n_heads * head_dim,
                                r,
                            )?;
                            exec.mxfp4_gemm_bs(kmx, &sc.d_pxq, exs, &mut sc.d_k, embd, kv_dim, r)?;
                            exec.mxfp4_gemm_bs(vmx, &sc.d_pxq, exs, &mut sc.d_v, embd, kv_dim, r)?;
                        }
                        Some(BsClass::Nv4Smooth) => {
                            let sm = layer.smooth.as_ref().expect("smooth vecs");
                            exec.quantize_nvf4_smooth(
                                &sc.d_xn,
                                &sm.attn_sinv,
                                &mut sc.d_pxq,
                                exs,
                                r * embd,
                                embd,
                            )?;
                            exec.mxfp4_gemm_nv4(
                                qmx,
                                &sc.d_pxq,
                                exs,
                                &mut sc.d_q,
                                embd,
                                n_heads * head_dim,
                                r,
                            )?;
                            exec.mxfp4_gemm_nv4(kmx, &sc.d_pxq, exs, &mut sc.d_k, embd, kv_dim, r)?;
                            exec.mxfp4_gemm_nv4(vmx, &sc.d_pxq, exs, &mut sc.d_v, embd, kv_dim, r)?;
                        }
                        Some(BsClass::Nv4) => {
                            exec.quantize_nvf4(&sc.d_xn, &mut sc.d_pxq, exs, r * embd)?;
                            exec.mxfp4_gemm_nv4(
                                qmx,
                                &sc.d_pxq,
                                exs,
                                &mut sc.d_q,
                                embd,
                                n_heads * head_dim,
                                r,
                            )?;
                            exec.mxfp4_gemm_nv4(kmx, &sc.d_pxq, exs, &mut sc.d_k, embd, kv_dim, r)?;
                            exec.mxfp4_gemm_nv4(vmx, &sc.d_pxq, exs, &mut sc.d_v, embd, kv_dim, r)?;
                        }
                        Some(BsClass::Nv4Rot) => {
                            exec.quantize_nvf4_rot(&sc.d_xn, &mut sc.d_pxq, exs, r * embd)?;
                            exec.mxfp4_gemm_nv4(
                                qmx,
                                &sc.d_pxq,
                                exs,
                                &mut sc.d_q,
                                embd,
                                n_heads * head_dim,
                                r,
                            )?;
                            exec.mxfp4_gemm_nv4(kmx, &sc.d_pxq, exs, &mut sc.d_k, embd, kv_dim, r)?;
                            exec.mxfp4_gemm_nv4(vmx, &sc.d_pxq, exs, &mut sc.d_v, embd, kv_dim, r)?;
                        }
                        Some(BsClass::W8) => {
                            exec.quantize_e4m3(&sc.d_xn, &mut sc.d_pxq, exs, r * embd)?;
                            exec.f8_gemm_w8(
                                qmx,
                                0,
                                &sc.d_pxq,
                                exs,
                                &mut sc.d_q,
                                embd,
                                n_heads * head_dim,
                                r,
                            )?;
                            exec.f8_gemm_w8(kmx, 0, &sc.d_pxq, exs, &mut sc.d_k, embd, kv_dim, r)?;
                            exec.f8_gemm_w8(vmx, 0, &sc.d_pxq, exs, &mut sc.d_v, embd, kv_dim, r)?;
                        }
                        _ => {
                            exec.quantize_e4m3(&sc.d_xn, &mut sc.d_pxq, exs, r * embd)?;
                            exec.mxfp4_gemm_bs(
                                qmx,
                                &sc.d_pxq,
                                exs,
                                &mut sc.d_q,
                                embd,
                                n_heads * head_dim,
                                r,
                            )?;
                            exec.mxfp4_gemm_bs(kmx, &sc.d_pxq, exs, &mut sc.d_k, embd, kv_dim, r)?;
                            exec.mxfp4_gemm_bs(vmx, &sc.d_pxq, exs, &mut sc.d_v, embd, kv_dim, r)?;
                        }
                    }
                } else {
                    // (ungated: xn materialized only for the stats pass)
                    prefill_add_norm_quant(
                        &exec,
                        &mut sc.d_x,
                        None,
                        false,
                        &layer.attn_norm.buf,
                        &mut sc.d_xn,
                        stat_on,
                        &mut sc.d_pxq,
                        &mut sc.d_pxs,
                        &mut sc.d_yq,
                        embd,
                        r,
                        eps,
                    )?;
                    prefill_mm_pre_sk(
                        &exec,
                        layer.wq.as_ref().expect("unfused"),
                        &sc.d_pxq,
                        &sc.d_pxs,
                        &sc.d_yq,
                        &mut sc.d_skfix,
                        Some(&mut sc.d_skpart),
                        &mut sc.d_q,
                        r,
                    )?;
                    prefill_mm_pre_sk(
                        &exec,
                        layer.wk.as_ref().expect("unfused"),
                        &sc.d_pxq,
                        &sc.d_pxs,
                        &sc.d_yq,
                        &mut sc.d_skfix,
                        Some(&mut sc.d_skpart),
                        &mut sc.d_k,
                        r,
                    )?;
                    prefill_mm_pre_sk(
                        &exec,
                        layer.wv.as_ref().expect("unfused"),
                        &sc.d_pxq,
                        &sc.d_pxs,
                        &sc.d_yq,
                        &mut sc.d_skfix,
                        Some(&mut sc.d_skpart),
                        &mut sc.d_v,
                        r,
                    )?;
                }
                if let Some(st) = stat_bufs.as_deref_mut() {
                    exec.col_absmax(&sc.d_xn, &mut st[li].attn, r, embd)?;
                }
                // q/k head-norm + rope (+ k cache scatter) fused when the pack
                // has the kernels - three global passes per side collapse to one
                // (bit-exact with the separate sequence; the elementwise passes
                // outweighed the qkv GEMMs at large prefill batches)
                if head_dim == 128 && exec.has_fused_qk_norm_rope() {
                    exec.q_norm_rope(
                        &sc.d_q,
                        &layer.q_norm.buf,
                        &mut sc.d_qn,
                        &sc.up_pos,
                        n_heads,
                        head_dim,
                        eps,
                        rope,
                        r,
                    )?;
                    exec.k_norm_rope_append(
                        &sc.d_k,
                        &layer.k_norm.buf,
                        &mut sc.kv_k[li],
                        &sc.up_pos,
                        Some(&sc.up_slots),
                        n_kv_heads,
                        head_dim,
                        kv_ctx,
                        eps,
                        rope,
                        r,
                        kv_dtype,
                    )?;
                } else {
                    exec.rmsnorm_batch(
                        &sc.d_q,
                        &layer.q_norm.buf,
                        &mut sc.d_qn,
                        head_dim,
                        eps,
                        r * n_heads,
                    )?;
                    exec.rmsnorm_batch(
                        &sc.d_k,
                        &layer.k_norm.buf,
                        &mut sc.d_kn,
                        head_dim,
                        eps,
                        r * n_kv_heads,
                    )?;
                    exec.rope_yarn_batch(&mut sc.d_qn, &sc.up_pos, n_heads, head_dim, rope, r)?;
                    exec.rope_yarn_batch(&mut sc.d_kn, &sc.up_pos, n_kv_heads, head_dim, rope, r)?;
                    exec.kv_append_batch(
                        &sc.d_kn,
                        &mut sc.kv_k[li],
                        &sc.up_pos,
                        Some(&sc.up_slots),
                        kv_dim,
                        kv_ctx,
                        r,
                        kv_dtype,
                    )?;
                }
                exec.kv_append_batch(
                    &sc.d_v,
                    &mut sc.kv_v[li],
                    &sc.up_pos,
                    Some(&sc.up_slots),
                    kv_dim,
                    kv_ctx,
                    r,
                    kv_dtype,
                )?;
            }
            lap!(0);
            // per-row-slot isolated attention: each row attends to its slot's
            // KV up to its position - the plain prefill kernel is single-slot
            // (all rows share one region), which cross-contaminates a ragged
            // multi-sequence batch. attn_decode_batch isolates by (slot,
            // position) per row, and all of a sequence's K/V is appended
            // before its rows read it (causal within the slot). swa_window=0
            // (dense, no sliding window). GEMMs above stay batched over all R
            // rows - only attention, which is per-sequence-independent anyway,
            // dispatches per row-slot.
            if use_flash_f16 && have_tiles {
                exec.attn_prefill_batch_f16(
                    &sc.d_qn,
                    &sc.kv_k[li],
                    &sc.kv_v[li],
                    sinks,
                    &mut sc.d_attn,
                    &sc.up_pos,
                    &sc.up_slots,
                    &sc.up_trow0,
                    &sc.up_tslot,
                    n_qtiles,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    kv_ctx,
                    kv_dim,
                    0,
                    r,
                    scale,
                    kv_dtype,
                )?;
            } else if have_tiles {
                exec.attn_prefill_batch(
                    &sc.d_qn,
                    &sc.kv_k[li],
                    &sc.kv_v[li],
                    sinks,
                    &mut sc.d_attn,
                    &sc.up_pos,
                    &sc.up_slots,
                    &sc.up_trow0,
                    &sc.up_tslot,
                    n_qtiles,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    kv_ctx,
                    kv_dim,
                    0,
                    r,
                    scale,
                    kv_dtype,
                )?;
            } else {
                exec.attn_decode_batch(
                    &sc.d_qn,
                    &sc.kv_k[li],
                    &sc.kv_v[li],
                    sinks,
                    &mut sc.d_attn,
                    &sc.up_pos,
                    Some(&sc.up_slots),
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    kv_ctx,
                    kv_dim,
                    0,
                    r,
                    scale,
                    kv_dtype,
                )?;
            }
            lap!(1);
            // ungated: no mul_sigmoid; attn output projection
            if let Some(st) = stat_bufs.as_deref_mut() {
                exec.col_absmax(&sc.d_attn, &mut st[li].wo, r, n_heads * head_dim)?;
            }
            if let (true, Some(omx)) = (r > BS_MIN_BATCH, &layer.wo_mx) {
                let exs = sc.d_exs.as_mut().expect("bs scratch");
                match self.bs_wo {
                    Some(BsClass::F8Smooth) => {
                        let sm = layer.smooth.as_ref().expect("smooth vecs");
                        exec.quantize_e4m3_smooth(
                            &sc.d_attn,
                            &sm.wo_s8inv,
                            &mut sc.d_pxq,
                            exs,
                            r * n_heads * head_dim,
                            n_heads * head_dim,
                        )?;
                        exec.mxfp4_gemm_bs(
                            omx,
                            &sc.d_pxq,
                            exs,
                            &mut sc.d_proj,
                            n_heads * head_dim,
                            embd,
                            r,
                        )?;
                    }
                    Some(BsClass::Nv4Smooth) => {
                        let sm = layer.smooth.as_ref().expect("smooth vecs");
                        exec.quantize_nvf4_smooth(
                            &sc.d_attn,
                            &sm.wo_sinv,
                            &mut sc.d_pxq,
                            exs,
                            r * n_heads * head_dim,
                            n_heads * head_dim,
                        )?;
                        exec.mxfp4_gemm_nv4(
                            omx,
                            &sc.d_pxq,
                            exs,
                            &mut sc.d_proj,
                            n_heads * head_dim,
                            embd,
                            r,
                        )?;
                    }
                    Some(BsClass::Nv4) => {
                        exec.quantize_nvf4(&sc.d_attn, &mut sc.d_pxq, exs, r * n_heads * head_dim)?;
                        exec.mxfp4_gemm_nv4(
                            omx,
                            &sc.d_pxq,
                            exs,
                            &mut sc.d_proj,
                            n_heads * head_dim,
                            embd,
                            r,
                        )?;
                    }
                    Some(BsClass::Nv4Rot) => {
                        exec.quantize_nvf4_rot(
                            &sc.d_attn,
                            &mut sc.d_pxq,
                            exs,
                            r * n_heads * head_dim,
                        )?;
                        exec.mxfp4_gemm_nv4(
                            omx,
                            &sc.d_pxq,
                            exs,
                            &mut sc.d_proj,
                            n_heads * head_dim,
                            embd,
                            r,
                        )?;
                    }
                    Some(BsClass::W8) => {
                        exec.quantize_e4m3(&sc.d_attn, &mut sc.d_pxq, exs, r * n_heads * head_dim)?;
                        exec.f8_gemm_w8(
                            omx,
                            0,
                            &sc.d_pxq,
                            exs,
                            &mut sc.d_proj,
                            n_heads * head_dim,
                            embd,
                            r,
                        )?;
                    }
                    _ => {
                        exec.quantize_e4m3(&sc.d_attn, &mut sc.d_pxq, exs, r * n_heads * head_dim)?;
                        exec.mxfp4_gemm_bs(
                            omx,
                            &sc.d_pxq,
                            exs,
                            &mut sc.d_proj,
                            n_heads * head_dim,
                            embd,
                            r,
                        )?;
                    }
                }
            } else {
                prefill_mm(
                    &exec,
                    &layer.wo,
                    &mut sc.d_pxq,
                    &mut sc.d_pxs,
                    &mut sc.d_yq,
                    &mut sc.d_skfix,
                    &sc.d_attn,
                    &mut sc.d_proj,
                    r,
                )?;
            }
            lap!(2);
            // residual + FFN norm + quantize, then SwiGLU FFN. At prefill
            // batch on sm_120a the three FFN GEMMs ride the block-scale FP4
            // tensor path (mxfp4 weight twins + e4m3 activations, hardware
            // ue8m0 scaling); every other rung keeps the tuned Q8_0 stack.
            if let (true, Some(dmx)) = (r > BS_MIN_BATCH, &layer.ffn_down_mx) {
                // gate/up planes are per-rung optional (the w8-tail rung
                // keeps the outlier-sensitive hidden readers on the exact
                // Q8_0 path and rides block-scale only on the tame down
                // site); both halves leave f32 gate/up in the same buffers.
                if let (Some(gmx), Some(umx)) = (&layer.ffn_gate_mx, &layer.ffn_up_mx) {
                    let exs = sc.d_exs.as_mut().expect("bs scratch");
                    exec.add_rmsnorm_batch(
                        &mut sc.d_x,
                        &sc.d_proj,
                        &layer.ffn_norm.buf,
                        &mut sc.d_xn,
                        embd,
                        eps,
                        r,
                    )?;
                    if let Some(st) = stat_bufs.as_deref_mut() {
                        exec.col_absmax(&sc.d_xn, &mut st[li].gu, r, embd)?;
                    }
                    if self.bs_gu_fused {
                        // fused front half: one e4m3 staging feeds both matrices,
                        // silu(g)*u quantizes in registers straight into the down
                        // GEMM's planes - the f32 gate/up planes never exist
                        let fq = sc.d_fq.as_mut().expect("fused scratch");
                        let fs = sc.d_fs.as_mut().expect("fused scratch");
                        exec.quantize_e4m3(&sc.d_xn, &mut sc.d_pxq, exs, r * embd)?;
                        exec.mxfp4_gemm_bs_gu(gmx, umx, &sc.d_pxq, exs, fq, fs, embd, ff, r)?;
                        exec.mxfp4_gemm_nv4(dmx, fq, fs, &mut sc.d_proj, ff, embd, r)?;
                        exec.add(&mut sc.d_x, &sc.d_proj, r * embd)?;
                        continue;
                    }
                    match self.bs_gu.expect("gate planes imply a class") {
                        BsClass::F8Smooth => {
                            let sm = layer.smooth.as_ref().expect("smooth vecs");
                            exec.quantize_e4m3_smooth(
                                &sc.d_xn,
                                &sm.gu_s8inv,
                                &mut sc.d_pxq,
                                exs,
                                r * embd,
                                embd,
                            )?;
                            exec.mxfp4_gemm_bs(
                                gmx,
                                &sc.d_pxq,
                                exs,
                                &mut sc.d_ffn_gate,
                                embd,
                                ff,
                                r,
                            )?;
                            exec.mxfp4_gemm_bs(umx, &sc.d_pxq, exs, &mut sc.d_ffn_up, embd, ff, r)?;
                        }
                        BsClass::Nv4Smooth => {
                            let sm = layer.smooth.as_ref().expect("smooth vecs");
                            exec.quantize_nvf4_smooth(
                                &sc.d_xn,
                                &sm.gu_sinv,
                                &mut sc.d_pxq,
                                exs,
                                r * embd,
                                embd,
                            )?;
                            exec.mxfp4_gemm_nv4(
                                gmx,
                                &sc.d_pxq,
                                exs,
                                &mut sc.d_ffn_gate,
                                embd,
                                ff,
                                r,
                            )?;
                            exec.mxfp4_gemm_nv4(
                                umx,
                                &sc.d_pxq,
                                exs,
                                &mut sc.d_ffn_up,
                                embd,
                                ff,
                                r,
                            )?;
                        }
                        BsClass::Nv4Rot => {
                            exec.quantize_nvf4_rot(&sc.d_xn, &mut sc.d_pxq, exs, r * embd)?;
                            exec.mxfp4_gemm_nv4(
                                gmx,
                                &sc.d_pxq,
                                exs,
                                &mut sc.d_ffn_gate,
                                embd,
                                ff,
                                r,
                            )?;
                            exec.mxfp4_gemm_nv4(
                                umx,
                                &sc.d_pxq,
                                exs,
                                &mut sc.d_ffn_up,
                                embd,
                                ff,
                                r,
                            )?;
                        }
                        BsClass::Nv4 => {
                            exec.quantize_nvf4(&sc.d_xn, &mut sc.d_pxq, exs, r * embd)?;
                            exec.mxfp4_gemm_nv4(
                                gmx,
                                &sc.d_pxq,
                                exs,
                                &mut sc.d_ffn_gate,
                                embd,
                                ff,
                                r,
                            )?;
                            exec.mxfp4_gemm_nv4(
                                umx,
                                &sc.d_pxq,
                                exs,
                                &mut sc.d_ffn_up,
                                embd,
                                ff,
                                r,
                            )?;
                        }
                        BsClass::F4 => {
                            exec.quantize_e2m1(&sc.d_xn, &mut sc.d_pxq, exs, r * embd)?;
                            exec.mxfp4_gemm_f4(
                                gmx,
                                &sc.d_pxq,
                                exs,
                                &mut sc.d_ffn_gate,
                                embd,
                                ff,
                                r,
                            )?;
                            exec.mxfp4_gemm_f4(umx, &sc.d_pxq, exs, &mut sc.d_ffn_up, embd, ff, r)?;
                        }
                        BsClass::F8 => {
                            exec.quantize_e4m3(&sc.d_xn, &mut sc.d_pxq, exs, r * embd)?;
                            exec.mxfp4_gemm_bs(
                                gmx,
                                &sc.d_pxq,
                                exs,
                                &mut sc.d_ffn_gate,
                                embd,
                                ff,
                                r,
                            )?;
                            exec.mxfp4_gemm_bs(umx, &sc.d_pxq, exs, &mut sc.d_ffn_up, embd, ff, r)?;
                        }
                        BsClass::W8 => {
                            exec.quantize_e4m3(&sc.d_xn, &mut sc.d_pxq, exs, r * embd)?;
                            exec.f8_gemm_w8(
                                gmx,
                                0,
                                &sc.d_pxq,
                                exs,
                                &mut sc.d_ffn_gate,
                                embd,
                                ff,
                                r,
                            )?;
                            exec.f8_gemm_w8(umx, 0, &sc.d_pxq, exs, &mut sc.d_ffn_up, embd, ff, r)?;
                        }
                    }
                } else {
                    // gu on the exact Q8_0 mmq stack (w8-tail)
                    prefill_add_norm_quant(
                        &exec,
                        &mut sc.d_x,
                        Some(&sc.d_proj),
                        false,
                        &layer.ffn_norm.buf,
                        &mut sc.d_xn,
                        stat_on,
                        &mut sc.d_pxq,
                        &mut sc.d_pxs,
                        &mut sc.d_yq,
                        embd,
                        r,
                        eps,
                    )?;
                    if let Some(st) = stat_bufs.as_deref_mut() {
                        exec.col_absmax(&sc.d_xn, &mut st[li].gu, r, embd)?;
                    }
                    prefill_mm_pre_sk(
                        &exec,
                        &layer.ffn_gate,
                        &sc.d_pxq,
                        &sc.d_pxs,
                        &sc.d_yq,
                        &mut sc.d_skfix,
                        Some(&mut sc.d_skpart),
                        &mut sc.d_ffn_gate,
                        r,
                    )?;
                    prefill_mm_pre_sk(
                        &exec,
                        &layer.ffn_up,
                        &sc.d_pxq,
                        &sc.d_pxs,
                        &sc.d_yq,
                        &mut sc.d_skfix,
                        Some(&mut sc.d_skpart),
                        &mut sc.d_ffn_up,
                        r,
                    )?;
                }
                let exs = sc.d_exs.as_mut().expect("bs scratch");
                match self.bs_dn {
                    BsClass::F8Smooth => {
                        let sm = layer.smooth.as_ref().expect("smooth vecs");
                        exec.quantize_e4m3_swiglu_smooth(
                            &sc.d_ffn_gate,
                            &sc.d_ffn_up,
                            &sm.dn_s8inv,
                            &mut sc.d_pxq,
                            exs,
                            r * ff,
                            ff,
                        )?;
                        exec.mxfp4_gemm_bs(dmx, &sc.d_pxq, exs, &mut sc.d_proj, ff, embd, r)?;
                    }
                    BsClass::Nv4Smooth => {
                        let sm = layer.smooth.as_ref().expect("smooth vecs");
                        exec.quantize_nvf4_swiglu_smooth(
                            &sc.d_ffn_gate,
                            &sc.d_ffn_up,
                            &sm.dn_sinv,
                            &mut sc.d_pxq,
                            exs,
                            r * ff,
                            ff,
                        )?;
                        exec.mxfp4_gemm_nv4(dmx, &sc.d_pxq, exs, &mut sc.d_proj, ff, embd, r)?;
                    }
                    // the down site never rotates (the ladder never emits it)
                    BsClass::Nv4 | BsClass::Nv4Rot => {
                        exec.quantize_nvf4_swiglu(
                            &sc.d_ffn_gate,
                            &sc.d_ffn_up,
                            &mut sc.d_pxq,
                            exs,
                            r * ff,
                        )?;
                        exec.mxfp4_gemm_nv4(dmx, &sc.d_pxq, exs, &mut sc.d_proj, ff, embd, r)?;
                    }
                    BsClass::F4 => {
                        exec.quantize_e2m1_swiglu(
                            &sc.d_ffn_gate,
                            &sc.d_ffn_up,
                            &mut sc.d_pxq,
                            exs,
                            r * ff,
                        )?;
                        exec.mxfp4_gemm_f4(dmx, &sc.d_pxq, exs, &mut sc.d_proj, ff, embd, r)?;
                    }
                    BsClass::F8 => {
                        exec.quantize_e4m3_swiglu(
                            &sc.d_ffn_gate,
                            &sc.d_ffn_up,
                            &mut sc.d_pxq,
                            exs,
                            r * ff,
                        )?;
                        exec.mxfp4_gemm_bs(dmx, &sc.d_pxq, exs, &mut sc.d_proj, ff, embd, r)?;
                    }
                    BsClass::W8 => {
                        exec.quantize_e4m3_swiglu(
                            &sc.d_ffn_gate,
                            &sc.d_ffn_up,
                            &mut sc.d_pxq,
                            exs,
                            r * ff,
                        )?;
                        exec.f8_gemm_w8(dmx, 0, &sc.d_pxq, exs, &mut sc.d_proj, ff, embd, r)?;
                    }
                }
            } else {
                prefill_add_norm_quant(
                    &exec,
                    &mut sc.d_x,
                    Some(&sc.d_proj),
                    false,
                    &layer.ffn_norm.buf,
                    &mut sc.d_xn,
                    false,
                    &mut sc.d_pxq,
                    &mut sc.d_pxs,
                    &mut sc.d_yq,
                    embd,
                    r,
                    eps,
                )?;
                prefill_mm_pre_sk(
                    &exec,
                    &layer.ffn_gate,
                    &sc.d_pxq,
                    &sc.d_pxs,
                    &sc.d_yq,
                    &mut sc.d_skfix,
                    Some(&mut sc.d_skpart),
                    &mut sc.d_ffn_gate,
                    r,
                )?;
                prefill_mm_pre_sk(
                    &exec,
                    &layer.ffn_up,
                    &sc.d_pxq,
                    &sc.d_pxs,
                    &sc.d_yq,
                    &mut sc.d_skfix,
                    Some(&mut sc.d_skpart),
                    &mut sc.d_ffn_up,
                    r,
                )?;
                if let (Some(st), Some(tmp)) = (stat_bufs.as_deref_mut(), stat_tmp.as_mut()) {
                    exec.copy_f32(&sc.d_ffn_gate, tmp, r * ff)?;
                    exec.swiglu(tmp, &sc.d_ffn_up, r * ff)?;
                    exec.col_absmax(tmp, &mut st[li].dn, r, ff)?;
                }
                prefill_ffn_down_sk(
                    &exec,
                    &layer.ffn_down,
                    &mut sc.d_pxq,
                    &mut sc.d_pxs,
                    &mut sc.d_yq,
                    &mut sc.d_skfix,
                    Some(&mut sc.d_skpart),
                    &mut sc.d_ffn_gate,
                    &sc.d_ffn_up,
                    &mut sc.d_proj,
                    ff,
                    r,
                )?;
            }
            exec.add(&mut sc.d_x, &sc.d_proj, r * embd)?;
            lap!(3);
        }
        if timing {
            tracing::info!(
                "  [layer-timing r={r}] qkv {:.1} ms | attn {:.1} ms | wo {:.1} ms | ffn {:.1} ms",
                sect[0] * 1e3,
                sect[1] * 1e3,
                sect[2] * 1e3,
                sect[3] * 1e3
            );
        }
        Ok(())
    }

    /// `output_norm` + last-token pooling over the `d_x` hidden the last
    /// `run_layers` produced. `lens` are the per-slot row counts of that pass
    /// (the last row of each slot is its sequence's pooled token). Returns a
    /// flat `[slots][embd]`.
    /// Swap the requested lane's state into the primary fields - every
    /// encode path below is lane-oblivious; only the entry points pick.
    fn set_lane(&mut self, lane: usize) {
        if lane == self.cur_lane {
            return;
        }
        let swap_with = |this: &mut Self, park: usize| {
            let l = &mut this.lanes_extra[park];
            std::mem::swap(&mut this.exec, &mut l.exec);
            std::mem::swap(&mut this.scratch, &mut l.scratch);
            std::mem::swap(&mut this.prefix_kv, &mut l.prefix_kv);
            std::mem::swap(&mut this.pool_out, &mut l.pool_out);
            std::mem::swap(&mut this.pool_flip, &mut l.pool_flip);
            std::mem::swap(&mut this.graphs, &mut l.graphs);
        };
        if self.cur_lane != 0 {
            let park = self.cur_lane - 1;
            swap_with(self, park); // put the current lane back in its slot
        }
        if lane != 0 {
            swap_with(self, lane - 1); // take the target lane out
        }
        self.cur_lane = lane;
    }

    /// How many execution lanes are available, forking extras lazily.
    /// Each lane carries its own scratch+KV memory. PADDOCK_LANES picks the
    /// count (default 2, clamped 1..=4); PADDOCK_NO_DUAL_LANE pins one.
    pub fn lanes(&mut self) -> usize {
        if paddock_models::dev_var_os!("PADDOCK_NO_DUAL_LANE").is_some() {
            return 1;
        }
        // default: quad-lane for small models (each lane costs a scratch+KV
        // set - a few GB at 0.6B, tens at 8B), dual for the big ones
        let default = if self.ff <= 4096 { 4 } else { 2 };
        let want = paddock_models::dev_var!("PADDOCK_LANES")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(default)
            .clamp(1, 4);
        while self.lanes_extra.len() + 1 < want {
            match self.exec.fork_stream() {
                Ok(exec) => self.lanes_extra.push(LaneState {
                    exec: Arc::new(exec),
                    scratch: None,
                    prefix_kv: None,
                    pool_out: [None, None],
                    pool_flip: 0,
                    graphs: std::collections::HashMap::new(),
                }),
                Err(e) => {
                    tracing::warn!(
                        "paddock: lane fork failed ({e}); staying at {}",
                        self.lanes_extra.len() + 1
                    );
                    break;
                }
            }
        }
        self.lanes_extra.len() + 1
    }

    /// Enqueue the pooled-output tail (device gather + output norm into a
    /// flip buffer) and record its completion event - no host sync. The
    /// matching `pool_collect` reads the rows through the side stream.
    fn pool_submit(&mut self, lens: &[usize]) -> Result<PendingPooled, GpuModelError> {
        let exec = self.exec.clone();
        let (embd, eps) = (self.embd, self.rms_eps);
        let n = lens.len();
        let flip = self.pool_flip;
        self.pool_flip ^= 1;
        // grow-only per flip; the other buffer may belong to an uncollected
        // batch and is never touched here (frees are stream-ordered anyway)
        if self.pool_out[flip]
            .as_ref()
            .is_none_or(|b| b.len() < n * embd)
        {
            self.pool_out[flip] = Some(exec.alloc(n.next_multiple_of(128) * embd)?);
        }
        // last-token pooling per sequence (pooling_type=3; add_eos_token=true):
        // gather the last row of each slot on device, norm just those -
        // [slots, embd], not the whole rows-sized hidden plane.
        let mut idx = Vec::with_capacity(n);
        let mut off = 0usize;
        for &len in lens {
            off += len;
            idx.push((off - 1) as u32);
        }
        let sc = self.scratch.as_mut().expect("scratch");
        exec.upload_u32(&idx, &mut sc.up_idx)?;
        exec.embed_gather_batch(&sc.d_x, &sc.up_idx, &mut sc.d_h, embd, n)?;
        let pool = self.pool_out[flip].as_mut().expect("pool buffer");
        exec.rmsnorm_batch(&sc.d_h, &self.out_norm.buf, pool, embd, eps, n)?;
        let ev = exec.record_event()?;
        Ok(PendingPooled {
            ev,
            flip,
            n,
            lane: self.cur_lane,
        })
    }

    /// Non-blocking: is a submitted batch's GPU work complete?
    pub fn pool_ready(&self, p: &PendingPooled) -> bool {
        // event query is lane-agnostic (any context-bound thread may ask)
        self.exec.event_done(&p.ev)
    }

    /// Read a submitted batch's pooled rows (blocks only on its event; later
    /// batches on the compute streams keep running).
    pub fn pool_collect(&mut self, p: &PendingPooled) -> Result<Vec<Vec<f32>>, GpuModelError> {
        self.set_lane(p.lane);
        let embd = self.embd;
        let buf = self.pool_out[p.flip].as_ref().expect("pool buffer");
        let flat = self.exec.to_host_len_after(&p.ev, buf, p.n * embd)?;
        Ok(flat.chunks_exact(embd).map(<[f32]>::to_vec).collect())
    }

    /// Ragged batched prefill of every sequence in one pass (per-slot KV
    /// isolation), returning each sequence's `output_norm`-ed last-token hidden.
    fn encode_pooled(&mut self, seqs: &[Vec<u32>]) -> Result<PendingPooled, GpuModelError> {
        let slots = seqs.len();
        assert!(slots > 0, "empty batch");
        let lens: Vec<usize> = seqs.iter().map(|s| s.len()).collect();
        assert!(lens.iter().all(|&l| l > 0), "empty sequence");
        let rows: usize = lens.iter().sum();
        let max_seq = lens.iter().max().copied().unwrap_or(0);
        assert!(max_seq <= self.max_ctx, "sequence exceeds max_ctx");
        // KV cache stride = batch's max length (aligned), not max_ctx: encoder
        // sequences are short, so max_ctx-sized KV would waste GBs at batch.
        let kv_stride = max_seq.next_multiple_of(64).max(64);
        // +1: the graph path pads rows into a reserved garbage slot
        self.ensure_scratch(rows, slots + 1, kv_stride)?;

        // per-row token / position-within-sequence / KV slot
        let mut tokens = Vec::with_capacity(rows);
        let mut positions = Vec::with_capacity(rows);
        let mut slot_of = Vec::with_capacity(rows);
        for (i, seq) in seqs.iter().enumerate() {
            for (p, &t) in seq.iter().enumerate() {
                tokens.push(t);
                positions.push(p as u32);
                slot_of.push(i as u32);
            }
        }
        self.run_layers(&tokens, &positions, &slot_of, rows)?;
        self.pool_submit(&lens)
    }

    /// Prefix-cached encode: the shared token prefix (identical across every
    /// sequence - a rerank request repeats the same instruct/query before each
    /// document) is prefilled once into slot 0, its per-layer KV broadcast into
    /// every slot, then only the per-sequence SUFFIXES are prefilled - each
    /// attending to the shared prefix KV. Numerically identical to encoding
    /// each full `prefix ++ suffix` (same tokens, positions, weights; the
    /// prefix KV is a bit-exact copy) at a fraction of the tokens: 60 + N*30
    /// instead of N*90 for a typical rerank batch.
    fn encode_pooled_cached(
        &mut self,
        prefix: &[u32],
        suffixes: &[Vec<u32>],
    ) -> Result<PendingPooled, GpuModelError> {
        let slots = suffixes.len();
        let prefix_len = prefix.len();
        let suf_lens: Vec<usize> = suffixes.iter().map(|s| s.len()).collect();
        let sum_suffix: usize = suf_lens.iter().sum();
        let max_full = prefix_len + suf_lens.iter().max().copied().unwrap_or(0);
        assert!(max_full <= self.max_ctx, "sequence exceeds max_ctx");
        let kv_stride = max_full.next_multiple_of(64).max(64);
        // +1: the graph path pads rows into a reserved garbage slot
        self.ensure_scratch(prefix_len.max(sum_suffix), slots + 1, kv_stride)?;

        let timing = paddock_models::dev_var_os!("PADDOCK_TIME_RERANK").is_some();
        let mut t = std::time::Instant::now();
        let mut lap = |label: &str, exec: &GpuExecutor| {
            if timing {
                let _ = exec.stream.synchronize();
                tracing::info!(
                    "  [rerank-timing] {label}: {:.1} ms",
                    t.elapsed().as_secs_f64() * 1e3
                );
                t = std::time::Instant::now();
            }
        };
        let exec0 = self.exec.clone();
        // Pass 1: the shared prefix into slot 0 (positions 0..prefix_len) -
        // restored from the cross-encode snapshot when the same prefix
        // repeats (rerank streams reuse one query across many requests),
        // else prefilled and snapshotted for the next encode.
        let hit = self
            .prefix_kv
            .as_ref()
            .is_some_and(|c| c.epoch == self.bs_epoch && c.tokens == prefix);
        if hit {
            self.copy_prefix_kv(prefix_len, false)?;
            lap("prefix restore", &exec0);
        } else {
            let p_pos: Vec<u32> = (0..prefix_len as u32).collect();
            let p_slot = vec![0u32; prefix_len];
            self.run_layers(prefix, &p_pos, &p_slot, prefix_len)?;
            lap("prefix pass", &exec0);
            self.save_prefix_kv(prefix)?;
            lap("prefix snapshot", &exec0);
        }

        // Broadcast the prefix KV (slot 0) into every other slot's [0:prefix_len]
        self.broadcast_prefix_kv(prefix_len, slots)?;
        lap("broadcast", &exec0);

        // Pass 2: prefill the per-sequence suffixes; positions continue from
        // prefix_len so RoPE matches the full sequence, and each suffix row
        // attends over its slot's [prefix KV ++ own suffix KV].
        let mut tokens = Vec::with_capacity(sum_suffix);
        let mut positions = Vec::with_capacity(sum_suffix);
        let mut slot_of = Vec::with_capacity(sum_suffix);
        for (i, suf) in suffixes.iter().enumerate() {
            for (p, &t) in suf.iter().enumerate() {
                tokens.push(t);
                positions.push((prefix_len + p) as u32);
                slot_of.push(i as u32);
            }
        }
        self.run_layers(&tokens, &positions, &slot_of, sum_suffix)?;
        lap("suffix pass", &exec0);
        let out = self.pool_submit(&suf_lens);
        lap("pool submit", &exec0);
        out
    }

    /// Copy the prefix KV computed in slot 0 into every other slot's
    /// `[0:prefix_len]` region - one `batched_copy` launch across all layers
    /// (the per-(layer,slot) memcpy storm otherwise dominates).
    fn broadcast_prefix_kv(
        &mut self,
        prefix_len: usize,
        slots: usize,
    ) -> Result<(), GpuModelError> {
        if slots <= 1 || prefix_len == 0 {
            return Ok(());
        }
        let exec = self.exec.clone();
        let kv_dim = self.n_kv_heads * self.head_dim;
        let kv_bytes = self.kv_dtype.bytes();
        let n_layers = self.n_layers;
        let sc = self.scratch.as_ref().expect("scratch");
        let slot_stride = (sc.kv_stride * kv_dim * kv_bytes) as u64;
        let prefix_bytes = (prefix_len * kv_dim * kv_bytes) as u64; // %16==0 (kv_dim%8==0)
        let base = |b: &CudaSlice<u8>| -> u64 {
            let (p, _g) = b.device_ptr(&exec.stream);
            p
        };
        let mut descs: Vec<u64> = Vec::with_capacity(n_layers * (slots - 1) * 2 * 3);
        for li in 0..n_layers {
            let kb = base(&sc.kv_k[li]);
            let vb = base(&sc.kv_v[li]);
            for j in 1..slots as u64 {
                descs.extend([kb, kb + j * slot_stride, prefix_bytes]);
                descs.extend([vb, vb + j * slot_stride, prefix_bytes]);
            }
        }
        let sc = self.scratch.as_mut().expect("scratch");
        exec.upload_u64(&descs, &mut sc.up_descs)?;
        exec.batched_copy(&sc.up_descs, descs.len() / 3)?;
        Ok(())
    }

    /// Snapshot slot 0's `[0:prefix_len]` KV rows into the cross-encode
    /// cache (buffers reused grow-only across queries).
    fn save_prefix_kv(&mut self, prefix: &[u32]) -> Result<(), GpuModelError> {
        let rows = prefix.len();
        let kv_dim = self.n_kv_heads * self.head_dim;
        let kv_bytes = self.kv_dtype.bytes();
        let need = rows * kv_dim * kv_bytes;
        let reuse = self.prefix_kv.as_ref().is_some_and(|c| c.cap_rows >= rows);
        if !reuse {
            let mut k = Vec::with_capacity(self.n_layers);
            let mut v = Vec::with_capacity(self.n_layers);
            for _ in 0..self.n_layers {
                k.push(self.exec.alloc_u8(need)?);
                v.push(self.exec.alloc_u8(need)?);
            }
            self.prefix_kv = Some(PrefixKv {
                tokens: Vec::new(),
                epoch: 0,
                k,
                v,
                cap_rows: rows,
            });
        }
        let c = self.prefix_kv.as_mut().expect("prefix cache");
        c.tokens = prefix.to_vec();
        c.epoch = self.bs_epoch;
        self.copy_prefix_kv(rows, true)
    }

    /// Move the prefix KV between slot 0 and the snapshot buffers - one
    /// batched_copy across all layers. `to_cache` picks the direction.
    fn copy_prefix_kv(&mut self, prefix_len: usize, to_cache: bool) -> Result<(), GpuModelError> {
        let exec = self.exec.clone();
        let kv_dim = self.n_kv_heads * self.head_dim;
        let kv_bytes = self.kv_dtype.bytes();
        let n_layers = self.n_layers;
        let bytes = (prefix_len * kv_dim * kv_bytes) as u64;
        let sc = self.scratch.as_ref().expect("scratch");
        let c = self.prefix_kv.as_ref().expect("prefix cache");
        let base = |b: &CudaSlice<u8>| -> u64 {
            let (p, _g) = b.device_ptr(&exec.stream);
            p
        };
        let mut descs: Vec<u64> = Vec::with_capacity(n_layers * 2 * 3);
        for li in 0..n_layers {
            for (slot0, cache) in [(&sc.kv_k[li], &c.k[li]), (&sc.kv_v[li], &c.v[li])] {
                let (s0, cb) = (base(slot0), base(cache));
                if to_cache {
                    descs.extend([s0, cb, bytes]);
                } else {
                    descs.extend([cb, s0, bytes]);
                }
            }
        }
        let sc = self.scratch.as_mut().expect("scratch");
        exec.upload_u64(&descs, &mut sc.up_descs)?;
        exec.batched_copy(&sc.up_descs, descs.len() / 3)?;
        Ok(())
    }

    /// Encode with automatic shared-prefix caching: if the sequences share a
    /// long common token prefix (rerank repeats the instruct/query prompt),
    /// prefill it once and reuse; otherwise fall back to the plain batched
    /// encode. Both paths are numerically identical - caching only factors the
    /// shared work out. Also safe for embeddings (distinct texts share no
    /// prefix -> falls back).
    fn encode_pooled_prefix_shared(
        &mut self,
        seqs: &[Vec<u32>],
    ) -> Result<PendingPooled, GpuModelError> {
        const MIN_SHARED_PREFIX: usize = 16; // below this the broadcast overhead outweighs
        if seqs.len() > 1 {
            let min_len = seqs.iter().map(Vec::len).min().unwrap_or(0);
            // cap at min_len-1 so every sequence keeps >=1 suffix token (the
            // pooled last token must live in the suffix pass)
            let common = longest_common_prefix(seqs).min(min_len.saturating_sub(1));
            if common >= MIN_SHARED_PREFIX {
                let prefix = seqs[0][..common].to_vec();
                let suffixes: Vec<Vec<u32>> = seqs.iter().map(|s| s[common..].to_vec()).collect();
                return self.encode_pooled_cached(&prefix, &suffixes);
            }
        }
        self.encode_pooled(seqs)
    }

    /// Embedding vectors: last-token pooled hidden, L2-normalized (the
    /// Qwen3-Embedding output convention).
    /// How many rows (tokens) a coalesced encode may hold. Scratch planes
    /// scale as rows x ff x 4 bytes; budgeting the largest plane at ~3.2 GB
    /// keeps the 8B model at its proven cap while letting small models merge
    /// far more concurrent requests (the 0.6B reranker needs ~129k rows to
    /// fuse 8 concurrent 128-doc requests into one weight-amortized pass).
    pub fn coalesce_row_budget(&self) -> usize {
        ((3200usize << 20) / (self.ff * 4)).clamp(16_384, 262_144)
    }

    pub fn embed(&mut self, seqs: &[Vec<u32>]) -> Result<Vec<Vec<f32>>, GpuModelError> {
        let p = self.embed_submit(seqs, 0)?;
        self.embed_collect(&p)
    }

    /// Enqueue an embedding batch without reading it back - the encoder
    /// thread pipelines submits ahead of collects, alternating lanes.
    pub fn embed_submit(
        &mut self,
        seqs: &[Vec<u32>],
        lane: usize,
    ) -> Result<PendingPooled, GpuModelError> {
        self.set_lane(lane);
        self.encode_pooled_prefix_shared(seqs)
    }

    pub fn embed_collect(&mut self, p: &PendingPooled) -> Result<Vec<Vec<f32>>, GpuModelError> {
        let mut pooled = self.pool_collect(p)?;
        for v in &mut pooled {
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
            for x in v.iter_mut() {
                *x /= norm;
            }
        }
        Ok(pooled)
    }

    /// Reranker scores: the lm_head logit gap between the `yes` and `no` tokens
    /// at each sequence's last position, as sigmoid(logit_yes - logit_no) in
    /// [0,1]. (Qwen3-Reranker is a generative yes/no relevance judge.)
    pub fn rerank(
        &mut self,
        seqs: &[Vec<u32>],
        yes_id: u32,
        no_id: u32,
    ) -> Result<Vec<f32>, GpuModelError> {
        let p = self.rerank_submit(seqs, yes_id, no_id, 0)?;
        self.rerank_collect(&p, yes_id, no_id)
    }

    /// Enqueue a rerank batch (also pre-warms the yes/no head-row cache -
    /// its dequant readback does a full stream sync, which must never land
    /// in a collect while a later batch is running).
    pub fn rerank_submit(
        &mut self,
        seqs: &[Vec<u32>],
        yes_id: u32,
        no_id: u32,
        lane: usize,
    ) -> Result<PendingPooled, GpuModelError> {
        self.set_lane(lane);
        self.head_rows(yes_id, no_id)?;
        self.encode_pooled_prefix_shared(seqs)
    }

    pub fn rerank_collect(
        &mut self,
        p: &PendingPooled,
        yes_id: u32,
        no_id: u32,
    ) -> Result<Vec<f32>, GpuModelError> {
        let pooled = self.pool_collect(p)?;
        // Only two of the vocab logits matter, so score on the CPU against
        // the dequanted yes/no head rows (cached). The old path ran a full
        // vocab x embd GEMV per document, uploading each pooled vector back
        // to the device and downloading vocab f32 logits to read two of
        // them - several ms of pure waste per document that rivalled the
        // whole batched encode at rerank batch sizes.
        let (yw, nw) = self.head_rows(yes_id, no_id)?;
        let mut scores = Vec::with_capacity(pooled.len());
        for v in &pooled {
            let y: f32 = v.iter().zip(&yw).map(|(a, b)| a * b).sum();
            let n: f32 = v.iter().zip(&nw).map(|(a, b)| a * b).sum();
            scores.push(1.0 / (1.0 + (n - y).exp()));
        }
        Ok(scores)
    }

    /// Dequant two rows of the Q8_0 lm_head to f32 on the host (each row is
    /// embd int8 + embd/32 f16 scales - a few KB), cached per (yes, no).
    fn head_rows(
        &mut self,
        yes_id: u32,
        no_id: u32,
    ) -> Result<(Vec<f32>, Vec<f32>), GpuModelError> {
        if let Some((y, n, yw, nw)) = &self.head_rows_cache
            && (*y, *n) == (yes_id, no_id)
        {
            return Ok((yw.clone(), nw.clone()));
        }
        let exec = self.exec.clone();
        let embd = self.embd;
        let n_blocks = embd / 32;
        let row = |t: u32| -> Result<Vec<f32>, GpuModelError> {
            let t = t as usize;
            let data = exec.to_host_range_u8(&self.output.data, t * embd, embd)?;
            let scale =
                exec.to_host_range_u8(&self.output.scale, t * n_blocks * 2, n_blocks * 2)?;
            let mut w = Vec::with_capacity(embd);
            for b in 0..n_blocks {
                let d = f16_to_f32(u16::from_le_bytes([scale[b * 2], scale[b * 2 + 1]]));
                for k in 0..32 {
                    w.push(data[b * 32 + k] as i8 as f32 * d);
                }
            }
            Ok(w)
        };
        let (yw, nw) = (row(yes_id)?, row(no_id)?);
        self.head_rows_cache = Some((yes_id, no_id, yw.clone(), nw.clone()));
        Ok((yw, nw))
    }
}

/// f16 bits -> f32 (the repacked Q8_0 scale format; no half crate at this
/// layer). Handles normals, subnormals and zero - scales are never inf/nan.
fn f16_to_f32(bits: u16) -> f32 {
    let s = ((bits >> 15) & 1) as u32;
    let e = ((bits >> 10) & 0x1f) as u32;
    let m = (bits & 0x3ff) as u32;
    let f = if e == 0 {
        // subnormal: m * 2^-24
        (m as f32) * 2.0f32.powi(-24)
    } else {
        f32::from_bits(((e + 112) << 23) | (m << 13))
    };
    if s == 1 { -f } else { f }
}

/// Length of the longest token prefix shared by all sequences (0 if they
/// diverge at the first token). Used to factor a rerank batch's repeated
/// instruct/query prompt out of the per-document encode.
fn longest_common_prefix(seqs: &[Vec<u32>]) -> usize {
    let Some((first, rest)) = seqs.split_first() else {
        return 0;
    };
    let mut n = first.len();
    for s in rest {
        n = n.min(s.len());
        let mut i = 0;
        while i < n && s[i] == first[i] {
            i += 1;
        }
        n = i;
        if n == 0 {
            break;
        }
    }
    n
}
