//! IBM Granite 4.1 (3b / 8b / 30b) - a plain dense llama-class family.
//!
//! Architecture, verified against a real official `granite-4.1-8b-Q8_0.gguf`
//! metadata dump (llama.cpp `src/models/granite.cpp` is the
//! study reference):
//!
//! - dense, full attention on every layer - no SWA, no hybrid, no MoE, no
//!   QK-norm, no attention sinks, no biases anywhere;
//! - GQA 32 q / 8 kv heads over head_dim 128, SwiGLU FFN, RMSNorm;
//! - rope theta 1e7, full rotary (`rope.dimension_count` == head_dim), no
//!   rope scaling.
//!
//! The only structural delta from a llama-style dense stack is four scalar
//! multipliers, all of which fail silently if missed - the model keeps
//! producing fluent text, just wrong text - which is exactly why the
//! same-weights greedy oracle is the acceptance gate:
//!
//! ```text
//! embd  = tok_embd[t] · embedding_scale                 (12.0)
//! layer:  h = x + residual_scale · Attn(RMSNorm(x))     (0.22)
//!         x = h + residual_scale · SwiGLU(RMSNorm(h))   (0.22)
//! head:   logits = (lm_head · RMSNorm(x)) / logit_scale (16.0)
//! ```
//!
//! plus `attention_scale` (0.0078125) used as the KQ scale in place of
//! 1/sqrt(head_dim). Note 0.0078125 is 1/128 = 1/head_dim, not 1/sqrt(128)
//! ≈ 0.0884 - close enough to a plausible default to slip through review,
//! far enough to wreck the logits.
//!
//! ROPE CONVENTION: granite is llama.cpp's `LLAMA_ROPE_TYPE_NORM` - rotating
//! INTERLEAVED pairs `(2k, 2k+1)` - while every other family we serve is NEOX
//! (half-split `(k, k+half)`). Wrong convention = fluent, confidently wrong
//! text, no error. The pack's rope kernel is templated on the convention and
//! granite calls `rope_yarn_batch_norm`; weights stay in file order. That is
//! how llama.cpp (separate `rope_norm` / `rope_neox` kernels) and vLLM (an
//! `IS_NEOX` template arg) both do it, and it costs the NEOX families nothing
//! since the instantiations are separate compiled kernels.
//!
//! Rejected alternative, recorded so nobody re-derives it: un-permuting q/k
//! rows at load so NEOX rope would "just work". It is exact for Q8_0 (row
//! reorder, blocks run along in_dim) and needs no kernel - but it only has a
//! bytes-level repack seam for Q8_0, so Q4_K_M (the quant most people run)
//! would have needed extra plumbing; it breaks the invariant that resident
//! weights match file bytes, which any later fused-QKV / W4A8-repack / TP
//! sharding path would trip over silently; and it makes intermediate-tensor
//! comparison against the parity oracle read permuted.
//!
//! Shapes are read from the file, never hard-coded: the same code serves 3b
//! (32 layers), 8b (40) and 30b (64 layers, ff 32768) - the 4.1 sizes differ
//! only in depth and FFN width.

use std::sync::Arc;

use crate::gpu::{DeviceTensor, GpuExecutor, KvDtype, QuantTensor, QuantW, RepackedKQ};

pub mod audio;
pub(crate) mod batch;
pub(crate) mod deepstack;
mod encode;
mod forward;
mod load;
mod load_hf;
mod multimodal;
pub(crate) mod ops;
mod prefix;
pub mod preprocess;
pub mod vision;

/// A captured decode graph. `CapturedGraph` is not `Send` by construction
/// (CUDA handles are context-bound), but the engine service owns the model on
/// one dedicated thread for exactly that reason, so the wrapper is sound
/// here - same rationale as gemma4/qwen35/laguna.
pub(crate) struct SendGraph(pub(crate) crate::gpu::CapturedGraph);
// SAFETY: the graph never leaves the engine thread that captured it; the
// Generator itself is only ever driven from that thread.
unsafe impl Send for SendGraph {}

/// Token-embedding residency: gathered from its file quant.
///
/// `Q8` is a misnomer kept for churn's sake - it holds a [`QuantTensor`], whose
/// `ty` the gather dispatches on, so the NVFP4 checkpoint's **bf16** embedding
/// table rides this variant unchanged. See `ops::embed_gather`.
pub(crate) enum TokEmbd {
    Q8(QuantTensor),
    Kq(RepackedKQ),
}

/// One linear weight, in whichever class the checkpoint shipped it.
///
/// Granite serves two lanes off one decoder and they carry different weight
/// classes, so the layer holds this rather than a bare [`QuantW`]:
///
/// - **GGUF lane** - everything is `Quant` (Q8_0 or a k-quant).
/// - **NVFP4 lane** (`ibm-granite/granite-4.2-*-nvfp4`, compressed-tensors) -
///   the seven per-layer projections are `Nvf4`, served W4A16 **exactly as
///   shipped, with no requantization**, while `lm_head` and the embedding
///   table stay `Bf16` because IBM's recipe deliberately leaves them
///   unquantized. Requantizing those two to Q8 to avoid a variant here would
///   be a quality change we were never asked for.
/// - **FP8 lane** (`ibm-granite/granite-4.2-*-fp8`) - the same seven
///   projections as `Fp8`, e4m3 bytes with one scale per output row, served
///   W8A8 against e4m3-quantized activations. Same rule: head, embeddings and
///   norms stay bf16 as shipped.
///
/// Fast paths in `batch.rs` match on `quant()` and simply fall through for the
/// other two classes - an NVFP4 plane has no Q8 twin to fuse with, so the
/// generic dispatch in `ops` is the correct answer, not a missed optimization.
pub(crate) enum GraniteW {
    Quant(QuantW),
    /// Checkpoint NVFP4: e2m1 nibbles + per-16 e4m3 scales + a global scale.
    Nvf4(crate::gpu::Nvf4Plane),
    /// q, k or v served from the layer's FUSED [q|k|v] NVFP4 plane
    /// (`GraniteLayer::qkv_nv4`) - carries only its geometry, the
    /// bytes live once in the fused plane. Every seat routes the three
    /// projections through `ops::nvf4_qkv_into` when the fused plane exists,
    /// so a generic call site that reaches a plane of this class is a wiring
    /// bug and the `ops` arms say so instead of computing garbage.
    Nvf4Fused {
        out_dim: usize,
        in_dim: usize,
    },
    /// Checkpoint FP8: e4m3 bytes + one scale per output row. `F8RowPlane`
    /// carries no shape, so the dims ride along - a transposed GEMM here is
    /// silent.
    Fp8 {
        plane: crate::gpu::F8RowPlane,
        out_dim: usize,
        in_dim: usize,
    },
    /// Checkpoint FP8 rebuilt at load into the tile-linear STRIP box layout -
    /// the tuned f8 lin family (f8lin_gemv b=1, f8_gemm_lin <=64, lin_kt
    /// above) serves every width from one plane. Arbitrary row scales fold
    /// to ue8m0 at load: exponent to the strip, mantissa residual into the
    /// e4m3 bytes (<= half-ULP, identity for pow2 scales) - see load_hf's
    /// `f8lin_requant`. PADDOCK_G42_F8ROW=1 keeps the old f8row class.
    F8Lin {
        plane: crate::gpu::RepackedMxfp4,
        out_dim: usize,
        in_dim: usize,
    },
    /// Unquantized checkpoint plane (`ty == GgmlType::Bf16`).
    Bf16(QuantTensor),
}

impl GraniteW {
    /// The Q8/k-quant weight, when this is one. `None` for the checkpoint
    /// classes - which is what makes a fast-path `match` fall through instead
    /// of needing an arm per class.
    pub(crate) fn quant(&self) -> Option<&QuantW> {
        match self {
            GraniteW::Quant(q) => Some(q),
            _ => None,
        }
    }

    /// Exact resident bytes, for the VRAM ledger. Same reason as
    /// [`QuantW::bytes`]: a free-VRAM delta measures pool growth, not tensors.
    pub(crate) fn bytes(&self) -> u64 {
        match self {
            GraniteW::Quant(q) => q.bytes(),
            GraniteW::Nvf4(p) => (p.data.len() + p.scale.len()) as u64,
            // counted once, on the fused plane
            GraniteW::Nvf4Fused { .. } => 0,
            GraniteW::Fp8 { plane, .. } => (plane.data.len() + plane.scale.len() * 4) as u64,
            GraniteW::F8Lin { plane, .. } => (plane.data.len() + plane.scale.len()) as u64,
            GraniteW::Bf16(t) => t.bytes.len() as u64,
        }
    }

    /// `[in_dim, out_dim]`. GGUF order puts in_dim first and the NVFP4 view
    /// carries `[out, in]`, so this normalizes the two conventions in one
    /// place - getting it wrong is a silently transposed GEMM. Named to match
    /// [`QuantW::dims`] so the call sites read the same in both lanes.
    pub(crate) fn dims(&self) -> [usize; 2] {
        match self {
            GraniteW::Quant(q) => [q.dims()[0], q.dims()[1]],
            GraniteW::Nvf4(p) => [p.in_dim, p.out_dim],
            GraniteW::Nvf4Fused { out_dim, in_dim } => [*in_dim, *out_dim],
            GraniteW::Fp8 {
                out_dim, in_dim, ..
            } => [*in_dim, *out_dim],
            GraniteW::F8Lin {
                out_dim, in_dim, ..
            } => [*in_dim, *out_dim],
            GraniteW::Bf16(t) => [t.dims[0], t.dims[1]],
        }
    }
}

impl TokEmbd {
    /// Exact resident device bytes - the VRAM ledger line. Same reason the
    /// layer groups sum `QuantW::bytes()`: a free-VRAM delta around a load
    /// measures mempool GROWTH, and the staging block each repack allocates
    /// stays pool-held until the next trim, so the delta double-counts every
    /// tensor it brackets.
    pub(crate) fn resident_bytes(&self) -> usize {
        match self {
            TokEmbd::Q8(t) => t.bytes.len(),
            TokEmbd::Kq(t) => t.data.len() + t.scales.len(),
        }
    }
}

/// One transformer block - uniform across the stack (no per-layer variation
/// at all, unlike laguna's hybrid pattern).
pub(crate) struct GraniteLayer {
    pub attn_norm: DeviceTensor,
    /// `attn_q` [embd, n_heads*head_dim] - file order, NORM rope at runtime.
    pub wq: GraniteW,
    /// `attn_k` [embd, n_kv_heads*head_dim].
    pub wk: GraniteW,
    /// `attn_v` [embd, n_kv_heads*head_dim] (v never ropes).
    ///
    /// Not merged into one q|k|v plane, though the measurement says it would
    /// pay: `pd_nvf4_gemv_multi` (slot 492) already merges the LAUNCHES, but
    /// its segmented walk costs 18.0 us in the serve profile against 14.16 us
    /// for a contiguous [6144, 4096] plane -- ~3.8 us/layer, ~2.6% of the c1
    /// tick. What blocks it is not the decode kernel (a 3-output store over a
    /// contiguous plane is trivial) but the r>1 seats: prefill and batched
    /// decode run q, k and v as separate multi-row GEMMs, and serving those
    /// from a merged plane needs a row OFFSET threaded through the nvf4
    /// multi-row wrappers. Duplicating the planes instead would cost ~1 GB on
    /// the 8b, which the VRAM rule forbids. Do the offset first.
    pub wv: GraniteW,
    /// f8row-class q|k|v concat (rows: q, then k, then v) for the fused
    /// wqkv decode GEMM+rope+append - built at load when all three
    /// projections are checkpoint FP8. Duplicates the three planes
    /// (~25 MB/layer on the 8b); the per-plane originals still serve r=1
    /// and prefill widths.
    pub qkv_f8: Option<crate::gpu::F8RowPlane>,
    /// NVFP4-class q|k|v as one contiguous [q_out + 2*kv_out, embd] plane
    /// built at load exactly like `gate_up`: the three tensors
    /// ship with the same `weight_global_scale` in every layer of both 4.2
    /// exports (checked per layer; a mismatch keeps the split planes), so the
    /// concat needs one `scale2` and every element keeps its own block scale.
    /// MEMORY-NEUTRAL: when this exists `wq`/`wk`/`wv` are `Nvf4Fused`
    /// markers and the separate planes are never uploaded. Serves every
    /// width: r=1 as one 6144-row GEMV (the segmented `pd_nvf4_gemv_multi`
    /// walk measured 18.0 us vs 14.2 for a contiguous plane on the c1
    /// profile), r>=2 as one W4A4 GEMM whose raw K-split partials the
    /// rope+append kernel folds itself (as separate GEMMs k/v sat at the
    /// launch floor, 7.9 us for 2.4 MB, and every GEMM paid a separate reduce
    /// launch).
    pub qkv_nv4: Option<crate::gpu::Nvf4Plane>,
    pub wo: GraniteW,
    pub ffn_norm: DeviceTensor,
    /// `ffn_gate` / `ffn_up`, SPLIT. `None` exactly when `gate_up` carries them
    /// merged - never both, so the planes are never resident twice.
    pub gate: Option<GraniteW>,
    pub up: Option<GraniteW>,
    /// gate and up as one [2*n_ff, n_embd] plane, gate rows first.
    ///
    /// This is a SHAPE optimisation, not a launch-count one: the nvf4 GEMV's
    /// efficiency is a function of out_dim, and 2*n_ff sits far higher on that
    /// curve than n_ff twice. Measured DRAM-cold on the 8b's geometry:
    /// [12800, 4096] runs 25.86 us at 1141 GB/s, so the split pair costs 51.72
    /// us, while [25600, 4096] runs 43.34 us at 1361 GB/s (89% of the 1531
    /// practical roof) -- 8.4 us per layer, 0.34 ms/token at 40 layers.
    ///
    /// It is exact, not a reorder: nvfp4 carries one `weight_global_scale` per
    /// tensor and granite ships gate and up with the same value in every layer
    /// (checked: 40/40 on the 8b, 62/62 on the 30b), so the merged plane needs
    /// one `scale2` and every element keeps its own e4m3 block scale. A family
    /// whose two scales differ cannot take this path -- the loader checks and
    /// falls back to split rather than rescaling into e4m3.
    ///
    /// Not to be confused with `pd_nvf4_gemv_multi` (slot 492), which merges
    /// the LAUNCHES but keeps the segments and their separate base pointers;
    /// that lost 6% on gate|up, twice. Here the rows are genuinely adjacent.
    pub gate_up: Option<GraniteW>,
    pub down: GraniteW,
}

/// Geometry + the four Granite scalars, all from the GGUF header.
pub(crate) struct Hparams {
    pub n_layer: usize,
    pub n_embd: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    /// Derived n_embd/n_heads - granite does not stamp attention.key_length /
    /// value_length, so there is nothing to read here (cross-checked against
    /// rope.dimension_count at load).
    pub head_dim: usize,
    pub n_ff: usize,
    pub n_vocab: usize,
    pub eps: f32,
    /// Plain rope (YarnRope with ext_factor 0 collapses to it), theta 1e7.
    pub rope: (f32, f32, f32, f32, f32, f32),
    /// `granite.embedding_scale` - multiplies the gathered embedding.
    pub embedding_scale: f32,
    /// `granite.residual_scale` - multiplies attn-out AND ffn-out before each
    /// residual add.
    pub residual_scale: f32,
    /// `granite.logit_scale` - logits are DIVIDED by this at the head.
    pub logit_scale: f32,
    /// `granite.attention.scale` - used as the KQ scale (replaces 1/sqrt(hd)).
    pub attention_scale: f32,
    /// `granite.deepstack_mapping`, one entry per layer: which vision stream to
    /// ADD into the image rows before this layer runs, or -1 for none. All -1
    /// on text-only checkpoints. Stream 0 never appears here - it is the image
    /// input embedding, not a layer injection.
    pub deepstack: Vec<i32>,
}

impl Hparams {
    /// Does this checkpoint inject vision features mid-stack?
    pub(crate) fn has_deepstack(&self) -> bool {
        self.deepstack.iter().any(|&k| k >= 0)
    }
}

/// The Granite GPU model. Bring-up milestone: weights resident + serial
/// batch-1 forward, validated by greedy parity against the newest llama.cpp
/// release binary on the identical GGUF. Batch/paged/graph lanes land after
/// parity, the same order gemma4 and laguna took.
/// Depth-2 decode-pipe state (see the `pipe` field).
pub(crate) struct PipeStateG {
    pub(crate) b: usize,
    pub(crate) tick: u64,
    pub(crate) ev: [Option<cudarc::driver::CudaEvent>; 2],
    /// tick-0 positions (row-major, len b); tick j writes KV at `pos0[i] + j`.
    pub(crate) pos0: Vec<u32>,
    /// explicit row->slot mapping for a pipe over an arbitrary slot set
    /// (the scheduler's churn-phase decode set). None = identity.
    pub(crate) slots: Option<Vec<u32>>,
}

pub struct GpuGranite {
    pub(crate) exec: Arc<GpuExecutor>,
    pub(crate) hp: Hparams,
    pub(crate) tok_embd: TokEmbd,
    pub(crate) layers: Vec<GraniteLayer>,
    pub(crate) output_norm: DeviceTensor,
    /// `output` [embd, vocab]. Granite 4.1 ships this as a real tensor even
    /// though the HF config says tie_word_embeddings - the loader still falls
    /// back to token_embd when it is absent (llama.cpp duplicates likewise).
    pub(crate) lm_head: GraniteW,
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
    /// `set_kv_dtype` before serving.
    pub(crate) kv_dtype: KvDtype,
    pub(crate) decode: Option<forward::DecodeState>,
    pub(crate) scratch: Option<forward::Scratch>,
    /// Continuous-batching state (batch.rs): the single full-attention block
    /// pool, per-slot tables, batched scratch and captured decode graphs.
    /// None until `enable_batch` succeeds with capacity > 1 - the serial
    /// decode/scratch above are torn down when it does.
    pub(crate) batch: Option<batch::BatchState>,
    /// Prompts mid-prefill, FIFO (batch.rs, stall-free batching). Each entry
    /// carries a CURSOR, so a tick advances a prompt partway and the next
    /// tick resumes it - decode rows never wait for a whole prompt, let
    /// alone a whole admission wave.
    pub(crate) chunked: Vec<batch::ChunkedPrefill>,
    /// Prompt rows the last prefill of each slot served from the prefix cache
    /// (usage reporting; taken and zeroed by `take_prefill_reused`). Sized at
    /// `enable_batch`; empty on the serial lane.
    pub(crate) last_reused: Vec<usize>,
    /// 1b.1 decode-page sealing: per-slot mirror of the radix KEY vector for
    /// every KV row this slot has actually BACKED - the resumed prefix at
    /// admission, then each fed chunk/decode row as its KV lands. Published
    /// into the radix at slot release, which turns the generated tail into
    /// the next turn's prefix and
    /// makes a preempted sequence's committed prefix instantly adoptable by
    /// its own recompute. Grows only on SUCCESSFUL feeds, so it can never
    /// claim rows that were not written.
    pub(crate) seal_hist: Vec<Vec<u32>>,
    /// Poisoned (false) on any positional gap - an uncovered feed path (e.g.
    /// a spec round, until 1b.4 wires it) disables publication for that
    /// sequence rather than ever publishing a wrong chain.
    pub(crate) seal_ok: Vec<bool>,
    /// granite-vision only: the loaded mmproj, and which slot holds which
    /// image where. Empty on text-only checkpoints, and every DeepStack
    /// operation short-circuits on that.
    pub(crate) vision: Option<vision::VisionModel>,
    /// granite-speech only: the conformer + Q-Former audio tower from the
    /// companion mmproj. Mutually exclusive with `vision` - the
    /// two checkpoints are different models that happen to share this decoder
    /// code, and `attach_audio`/`attach_vision` each refuse the other's file.
    pub(crate) audio: Option<audio::SpeechTower>,
    /// The `<image>` token id from the vocab (vision checkpoints only). The mm
    /// lane writes it into an image's rows before DeepStack replaces them - see
    /// `multimodal.rs` for why a real id matters even though it is overwritten.
    pub(crate) img_pad_id: Option<u32>,
    /// The `<|audio|>` token id (speech checkpoints only) - same role as
    /// `img_pad_id`: what the audio's rows carry until the tower's embeddings
    /// replace them.
    pub(crate) audio_pad_id: Option<u32>,
    /// In-flight depth-2 decode pipe (the idle-edge door): tick
    /// N+1 is enqueued before tick N's sampled ids reach the host, so the
    /// scheduler's commit/SSE work overlaps the GPU instead of gapping ~0.6 ms
    /// between step-graph replays (every steady-state c1 gap terminated at
    /// the next replay's first node). qwen35's pipe is the
    /// template; granite is simpler (no mrope, no lane fork).
    pub(crate) pipe: Option<PipeStateG>,
    pub(crate) media: deepstack::MediaRegistry,
    /// Encoded pictures held for reuse across turns, byte-budgeted. Without it
    /// every turn of an image conversation re-runs the tower and all 8
    /// Q-Formers over bytes it already had - 361 ms per repeat, measured.
    pub(crate) img_cache: Vec<deepstack::GraniteImageCacheEntry>,
    pub(crate) img_cache_bytes: usize,
    pub(crate) img_cache_clock: u64,
    pub(crate) img_cache_reused: u64,
    /// Admission waves mid-ENCODE (encode.rs). The front one is in
    /// flight and advances by one tile budget per tick; the rest are registered
    /// and waiting. Non-empty means the scheduler is holding those slots for us
    /// - they are neither admitted nor re-offerable until a step reports them.
    pub(crate) enc: std::collections::VecDeque<encode::WaveEncode>,
}
