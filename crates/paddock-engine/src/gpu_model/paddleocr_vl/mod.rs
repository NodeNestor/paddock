//! PaddleOCR-VL-1.6 - the Nordic/multilingual OCR family.
//!
//! 0.9B two-part model: a SigLIP-so400m-shaped NaViT encoder (27 layers,
//! hidden 1152, full attention, native-resolution patching driven by the
//! image grid) + 2-layer MLP projector with 2×2 spatial merge, feeding an
//! ERNIE-4.5-0.3B decoder (18 layers, hd128 decoupled GQA, 3D M-RoPE
//! [16,24,24]). Reference: the checkpoint's own `modeling_paddleocr_vl.py`;
//! vLLM's implementation and llama.cpp's
//! (`tools/mtmd/models/paddleocr.cpp`) are the cross-references.
//!
//! Both parts ingest the OFFICIAL GGUF (`PaddleOCR-VL-1.6-GGUF`): the oracle
//! dump (our OCR oracle tool) verified its planes are
//! byte-identical to the reference safetensors, so the parity chain
//! runs on literally the same weights end to end.
//!
//! Engine lane 1 (vision tower + projector) landed first; lane 2 (the
//! decoder + loader) brought the ERNIE side up on the official decoder
//! GGUF - the oracle byte-verified its planes against the reference
//! safetensors too, so both files of the pair are proven same-weights. The
//! serving surface landed next (exclusive serial path); the batched
//! serving lane (paged KV + continuous batching + chunked-mm) lives in
//! `batch`/`chunked`/`prefix` on the deepseek-ocr shape.

pub mod batch;
pub mod chunked;
pub mod forward;
pub mod load;
pub mod multimodal;
pub mod prefix;
pub mod preprocess;
pub mod vision;

pub use load::GpuPaddleOcrVl;

/// A captured decode tick. Single-threaded on the engine's thread (the same
/// argument every other family's SendGraph makes).
pub(crate) struct SendGraph(pub(crate) crate::gpu::CapturedGraph);
// SAFETY: never accessed from two threads at once; see above.
unsafe impl Send for SendGraph {}

/// The image-placeholder token (`<|IMAGE_PLACEHOLDER|>`) - the checkpoint
/// config's `image_token_id`, verified equal to the tokenizer's id for that
/// string. Not in the GGUF header; the decoder splices projector rows at
/// runs of this id.
pub const IMAGE_TOKEN: u32 = 100_295;

use paddock_models::gguf::Value;
use paddock_models::mapped::MappedGguf;

/// The GGUF architecture string of the official decoder file.
pub const ARCH: &str = "paddleocr";

/// Every decoder key this family reads, so a stale or foreign file fails by
/// NAME (the deepseek-ocr discipline).
mod key {
    pub const BLOCK_COUNT: &str = "paddleocr.block_count";
    pub const EMBD: &str = "paddleocr.embedding_length";
    pub const HEAD: &str = "paddleocr.attention.head_count";
    pub const HEAD_KV: &str = "paddleocr.attention.head_count_kv";
    pub const KEY_LEN: &str = "paddleocr.attention.key_length";
    pub const VALUE_LEN: &str = "paddleocr.attention.value_length";
    pub const CTX: &str = "paddleocr.context_length";
    pub const FF: &str = "paddleocr.feed_forward_length";
    pub const RMS_EPS: &str = "paddleocr.attention.layer_norm_rms_epsilon";
    pub const ROPE_BASE: &str = "paddleocr.rope.freq_base";
    pub const SECTIONS: &str = "paddleocr.rope.dimension_sections";
}

#[derive(Debug, thiserror::Error)]
pub enum HparamsError {
    #[error("paddleocr-vl: missing GGUF key {0}")]
    MissingKey(String),
    #[error("paddleocr-vl: GGUF key {0} has the wrong type")]
    BadKey(String),
    #[error("paddleocr-vl: {0}")]
    Geometry(String),
}

fn usize_key(map: &MappedGguf, k: &str) -> Result<usize, HparamsError> {
    map.gguf()
        .metadata
        .get(k)
        .ok_or_else(|| HparamsError::MissingKey(k.to_owned()))?
        .as_u64()
        .map(|v| v as usize)
        .ok_or_else(|| HparamsError::BadKey(k.to_owned()))
}

fn f32_key(map: &MappedGguf, k: &str) -> Result<f32, HparamsError> {
    match map.gguf().metadata.get(k) {
        Some(Value::F32(f)) => Ok(*f),
        Some(Value::F64(f)) => Ok(*f as f32),
        Some(_) => Err(HparamsError::BadKey(k.to_owned())),
        None => Err(HparamsError::MissingKey(k.to_owned())),
    }
}

/// ERNIE-4.5-0.3B decoder geometry, pinned from the checkpoint's config.json
/// AND llama.cpp's converter output (they agree).
#[derive(Debug, Clone)]
pub struct Hparams {
    pub n_layer: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_kv_heads: usize,
    /// DECOUPLED from the hidden size: 128, while n_embd/n_head = 64. Read
    /// from `attention.key_length` - deriving it is the classic silent-garbage
    /// trap (deepseek-ocr hit the mirror image of it).
    pub head_dim: usize,
    pub n_ff: usize,
    /// Not in the header - measured off `token_embd.weight`'s slow dim.
    pub n_vocab: usize,
    pub n_ctx_train: usize,
    pub eps: f32,
    pub rope_base: f32,
    /// M-RoPE per-axis rotary-PAIR counts (t, h, w, extra) - [16, 24, 24, 0].
    /// 2·sum == n_rot == head_dim: the full head rotates (the reference
    /// asserts n_embd_head == n_rot, and HF cat(freqs,freqs) covers all 128).
    pub sections: [u32; 4],
    pub n_rot: usize,
}

impl Hparams {
    pub fn from_gguf(map: &MappedGguf) -> Result<Self, HparamsError> {
        let arch = map.gguf().architecture().unwrap_or_default();
        if arch != ARCH {
            return Err(HparamsError::Geometry(format!(
                "architecture {arch:?} is not {ARCH:?}"
            )));
        }
        let n_head = usize_key(map, key::HEAD)?;
        let n_kv_heads = usize_key(map, key::HEAD_KV)?;
        if n_kv_heads == 0 || n_head % n_kv_heads != 0 {
            return Err(HparamsError::Geometry(format!(
                "head_count {n_head} not divisible by head_count_kv {n_kv_heads}"
            )));
        }
        let head_dim = usize_key(map, key::KEY_LEN)?;
        let value_len = usize_key(map, key::VALUE_LEN)?;
        if value_len != head_dim {
            return Err(HparamsError::Geometry(format!(
                "value_length {value_len} != key_length {head_dim} - not this family"
            )));
        }
        let sections = match map.gguf().metadata.get(key::SECTIONS) {
            Some(Value::Array(a)) if a.len() == 4 => {
                let mut s = [0u32; 4];
                for (i, v) in a.iter().enumerate() {
                    s[i] = v
                        .as_u64()
                        .ok_or_else(|| HparamsError::BadKey(key::SECTIONS.to_owned()))?
                        as u32;
                }
                s
            }
            Some(_) => return Err(HparamsError::BadKey(key::SECTIONS.to_owned())),
            None => return Err(HparamsError::MissingKey(key::SECTIONS.to_owned())),
        };
        // full-head rotation: pairs sum to head_dim/2 exactly. A file where
        // they don't is a different rope regime, not a tolerable variation.
        let n_rot = 2 * sections.iter().sum::<u32>() as usize;
        if n_rot != head_dim {
            return Err(HparamsError::Geometry(format!(
                "rope sections {sections:?} rotate {n_rot} dims, head_dim is {head_dim}"
            )));
        }
        let n_vocab = map
            .tensor_info("token_embd.weight")
            .ok_or_else(|| HparamsError::Geometry("no token_embd.weight".into()))?
            .dims
            .get(1)
            .copied()
            .ok_or_else(|| HparamsError::Geometry("token_embd is not 2-D".into()))?
            as usize;

        Ok(Hparams {
            n_layer: usize_key(map, key::BLOCK_COUNT)?,
            n_embd: usize_key(map, key::EMBD)?,
            n_head,
            n_kv_heads,
            head_dim,
            n_ff: usize_key(map, key::FF)?,
            n_vocab,
            n_ctx_train: usize_key(map, key::CTX)?,
            eps: f32_key(map, key::RMS_EPS)?,
            rope_base: f32_key(map, key::ROPE_BASE)?,
            sections,
            n_rot,
        })
    }
}
