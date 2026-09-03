//! Qwen3-ASR (GGUF arch `qwen3` + companion `qwen3a` audio mmproj) - the
//! first voice family. Reference: transformers 5.14.1
//! `modeling_qwen3_asr.py`; llama.cpp b10327 `models/qwen3a.cpp` studied as
//! the second reference.
//!
//! The decoder is BONE-STOCK Qwen3 dense - 28 layers, hidden 2048, GQA
//! 16 q / 8 kv over head_dim 128, per-head q/k RMS-norm, SwiGLU 6144,
//! RMSNorm 1e-6, NEOX rope theta 1e6, TIED lm_head, vocab 151936. The
//! serial lane follows granite's bring-up shape (batch-1 decode + one
//! batched-row prefill; plain launches, correctness first) minus granite's
//! four scalar multipliers, plus the qk-norm/rope-NEOX deltas; the tuned
//! Q8_0 prefill stack comes from qwen35::ops exactly as qwen3.rs (the
//! embeddings encoder twin of this decoder) already proved.
//!
//! Audio rides the multimodal splice: the runner decodes WAV to 16 kHz mono
//! f32 (`MmChunk::Audio`), the host mel frontend (`crate::audio`) produces
//! the normalized log-mel frames, the tower (`audio.rs`) encodes them into
//! `[n_tokens, 2048]` embeddings, and the prefill overwrites the
//! `<|audio_token|>` rows with them - decoder attention stays CAUSAL over
//! audio rows (upstream treats them as ordinary input embeddings).

use std::sync::Arc;

use crate::gpu::{DeviceTensor, GpuExecutor, KvDtype, QuantTensor, QuantW, RepackedKQ};
use crate::gpu_model::gpt_oss::GpuModelError;

pub mod aligner;
pub mod audio;
mod batch;
mod forward;
mod load;

pub use aligner::AlignerMeta;
pub use audio::{AudioOutput, AudioTower};
pub(crate) use forward::{DecodeState, PrefillScratch, Scratch};

/// A captured decode graph. `CapturedGraph` is not `Send` by construction
/// (CUDA handles are context-bound), but the engine service owns the model on
/// one dedicated thread for exactly that reason - same rationale as
/// granite/gemma4/qwen35/laguna.
pub(crate) struct SendGraph(pub(crate) crate::gpu::CapturedGraph);
// SAFETY: the graph never leaves the engine thread that captured it; the
// Generator itself is only ever driven from that thread.
unsafe impl Send for SendGraph {}

/// Input-embedding table with a resident gather path (granite's pattern).
pub(crate) enum TokEmbd {
    Q8(QuantTensor),
    Kq(RepackedKQ),
}

impl TokEmbd {
    pub(crate) fn resident_bytes(&self) -> usize {
        match self {
            TokEmbd::Q8(t) => t.bytes.len(),
            TokEmbd::Kq(t) => t.data.len() + t.scales.len(),
        }
    }
}

pub(crate) struct AsrLayer {
    pub attn_norm: DeviceTensor,
    pub wq: QuantW,
    pub wk: QuantW,
    pub wv: QuantW,
    pub q_norm: DeviceTensor,
    pub k_norm: DeviceTensor,
    pub wo: QuantW,
    pub ffn_norm: DeviceTensor,
    pub gate: QuantW,
    pub up: QuantW,
    pub down: QuantW,
}

pub(crate) struct Hparams {
    pub n_layer: usize,
    pub n_embd: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub n_ff: usize,
    pub n_vocab: usize,
    pub eps: f32,
    /// YaRN/rope kernel params (ext_factor 0 => plain NEOX rope).
    pub rope: (f32, f32, f32, f32, f32, f32),
}

pub struct GpuQwen3Asr {
    pub(crate) exec: Arc<GpuExecutor>,
    pub(crate) hp: Hparams,
    pub(crate) max_ctx: usize,
    pub(crate) kv_dtype: KvDtype,
    pub(crate) tok_embd: TokEmbd,
    pub(crate) layers: Vec<AsrLayer>,
    pub(crate) output_norm: DeviceTensor,
    /// lm_head - tied to `token_embd` when `output.weight` is absent (it is
    /// on every published Qwen3-ASR checkpoint).
    pub(crate) lm_head: QuantW,
    pub(crate) tower: Option<AudioTower>,
    pub(crate) decode: Option<DecodeState>,
    pub(crate) scratch: Option<Scratch>,
    pub(crate) prefill: Option<PrefillScratch>,
    /// Continuous-batching state (batch.rs). Serving is batched
    /// whenever this is built; the serial lane above stays the batch-off
    /// parity spine.
    pub(crate) batch: Option<batch::BatchState>,
    /// Prompts advancing under stall-free chunked prefill  -
    /// their rows ride the mixed ticks.
    pub(crate) chunked: Vec<batch::ChunkedPrefill>,
    pub(crate) weights_bytes: u64,
}

impl GpuQwen3Asr {
    /// Attach the audio tower from the companion mmproj GGUF. Hard-fails on
    /// a non-qwen3a projector so a mispaired file cannot half-serve.
    pub fn attach_audio(
        &mut self,
        mmproj: &paddock_models::mapped::MappedGguf,
    ) -> Result<(), GpuModelError> {
        let tower = AudioTower::load(self.exec.clone(), mmproj)?;
        if tower.out_dim != self.hp.n_embd {
            return Err(GpuModelError::Unsupported(format!(
                "audio tower projects to {} but the decoder embeds {} - mismatched files",
                tower.out_dim, self.hp.n_embd
            )));
        }
        self.weights_bytes += tower.weight_bytes() as u64;
        self.tower = Some(tower);
        Ok(())
    }

    pub fn has_audio(&self) -> bool {
        self.tower.is_some()
    }
}
