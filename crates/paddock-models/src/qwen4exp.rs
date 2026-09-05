//! Qwen3.8-Flash-Next (`qwen4_exp`) checkpoint config.
//!
//! Safetensors-primary family (no GGUF exists; llama.cpp #27742 unmerged).
//! Geometry facts and tensor inventory:
//! 176B total = 125B main + 51B PLE n-gram table, 6B active; 48 layers =
//! 36 GDN + 12 full-attention-with-QSA (every 4th), 512 routed experts
//! top-10 + 1 shared expert per layer, 4-stream gated hyper-connections,
//! one PLE n-gram layer, 1 MTP layer, SigLIP-class ViT.
//!
//! Parsed from the real RadixArk NVFP4 config, not the model
//! card: `rope_parameters` is a nested dict, `text_config.model_type` is
//! `qwen4_exp_text` (top-level `qwen4_exp`), `mtp` is a nested dict, and
//! `ple_embedding_dtype` = `float8_e4m3fn` is present (the sglang PR#40
//! knob). `ple_layer_ids` is ONE-indexed (id 2 = decoder layer index 1 -
//! proven by the shard audit: the ple.* tensors live under
//! `model.language_model.layers.1.`).

use std::path::Path;

use crate::safetensors::StError;

/// One decoder layer's mixer kind, from `layer_types`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Qwen4ExpBlock {
    /// `linear_attention` - gated DeltaNet (GDN), sigmoid output gate.
    Gdn,
    /// `full_attention` - gated attention + QSA sparse indexer.
    Attention,
}

/// Parsed `config.json` `text_config` (+ eos set from generation_config).
/// Field names follow the checkpoint's vocabulary where sane.
#[derive(Debug, Clone)]
pub struct Qwen4ExpConfig {
    pub hidden: usize,  // 2560
    pub n_layer: usize, // 48
    pub blocks: Vec<Qwen4ExpBlock>,
    pub vocab: usize,   // 248320
    pub max_pos: usize, // 262144
    pub eps: f32,       // rms_norm_eps 1e-6 (zero-centered (1+w) norms)

    // attention (12 layers; gated: q_proj rows = 2 * n_heads * head_dim)
    pub n_heads: usize,    // 24
    pub n_kv_heads: usize, // 2  (G12)
    pub head_dim: usize,   // 256
    pub rotary_dim: usize, // 64 (partial_rotary_factor 0.25)
    pub rope_theta: f32,   // 1e7

    // QSA indexer (per attention layer)
    pub idx_heads: usize,    // 4
    pub idx_kv_heads: usize, // 1
    pub idx_head_dim: usize, // 128
    pub idx_budget: usize,   // 2048 (dense fast-path while visible <= budget+3)
    pub idx_compress: usize, // 4 (block size; KV page size must be a multiple)

    // GDN (36 layers; split planes, sigmoid gate - Not the qwen35 silu)
    pub gdn_k_heads: usize, // 16
    pub gdn_v_heads: usize, // 48
    pub gdn_k_dim: usize,   // 128
    pub gdn_v_dim: usize,   // 128
    pub gdn_conv: usize,    // 4

    // MoE (every layer): softmax router top-10 renormed + sigmoid-gated
    // shared expert
    pub n_expert: usize,  // 512
    pub n_active: usize,  // 10
    pub moe_ff: usize,    // 640
    pub shared_ff: usize, // 640

    // hyper-connections (4-stream gated residual, low-rank mix)
    pub hc_count: usize,   // 4
    pub hc_lowrank: usize, // 320

    // PLE n-gram layer(s), ZERO-indexed decoder layer indices
    pub ple_layers: Vec<usize>, // [1]
    pub ple_embed: usize,       // 2560 (16 heads x 160)
    pub ple_conv: usize,        // 4 (dilation 3 -> 9-token ring)
    pub ngram_size: usize,      // 3
    pub heads_per_ngram: usize, // 8 (x2 ngram orders = 16 heads)
    pub ngram_vocab_base: u64,  // 20000000
    pub ngram_split: usize,     // 128 shards
    /// `ple_embedding_dtype` as written ("float8_e4m3fn")
    pub ple_dtype: String,

    // MTP (1 full-attention layer, bf16 fused experts, shared lm_head)
    pub mtp_layers: usize,

    /// generation_config eos set - [248046 <|im_end|>, 248044 <|endoftext|>]
    pub eos_ids: Vec<u32>,
    pub bos_id: u32, // 248044
}

impl Qwen4ExpConfig {
    /// GDN in_proj_qkv rows: q 2048 | k 2048 | v 6144 = 10240.
    pub fn gdn_qkv_rows(&self) -> usize {
        2 * self.gdn_k_heads * self.gdn_k_dim + self.gdn_v_heads * self.gdn_v_dim
    }
    /// GDN z (output gate) rows = v plane width (6144).
    pub fn gdn_z_rows(&self) -> usize {
        self.gdn_v_heads * self.gdn_v_dim
    }
    /// attention q_proj rows: q 6144 | output gate 6144 = 12288.
    pub fn attn_q_rows(&self) -> usize {
        2 * self.n_heads * self.head_dim
    }
    /// attention o_proj input width (6144).
    pub fn attn_o_in(&self) -> usize {
        self.n_heads * self.head_dim
    }
    /// hyper-connection stream state width (4 x 2560 = 10240).
    pub fn hc_width(&self) -> usize {
        self.hc_count * self.hidden
    }
    /// PLE n-gram head count (16) - bigram 8 + trigram 8.
    pub fn ple_heads(&self) -> usize {
        (self.ngram_size - 1) * self.heads_per_ngram
    }

    pub fn read(dir: &Path) -> Result<Self, StError> {
        let v: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)
            .map_err(|e| StError::Header(e.to_string()))?;
        let top_type = v.get("model_type").and_then(|x| x.as_str()).unwrap_or("");
        if top_type != "qwen4_exp" {
            return Err(StError::Header(format!(
                "qwen4exp: model_type is {top_type:?}, not qwen4_exp"
            )));
        }
        let tc = v
            .get("text_config")
            .ok_or_else(|| StError::Header("qwen4exp: no text_config".into()))?;
        let miss = |k: &str| StError::Header(format!("qwen4exp text_config: missing {k}"));
        let u = |k: &str| {
            tc.get(k)
                .and_then(|x| x.as_u64())
                .map(|x| x as usize)
                .ok_or_else(|| miss(k))
        };
        let f = |k: &str| {
            tc.get(k)
                .and_then(|x| x.as_f64())
                .map(|x| x as f32)
                .ok_or_else(|| miss(k))
        };
        let s = |k: &str| {
            tc.get(k)
                .and_then(|x| x.as_str())
                .map(str::to_owned)
                .ok_or_else(|| miss(k))
        };

        let n_layer = u("num_hidden_layers")?;
        let layer_types = tc
            .get("layer_types")
            .and_then(|x| x.as_array())
            .ok_or_else(|| miss("layer_types"))?;
        if layer_types.len() != n_layer {
            return Err(StError::Header(format!(
                "qwen4exp: layer_types has {} entries for {n_layer} layers",
                layer_types.len()
            )));
        }
        let blocks = layer_types
            .iter()
            .map(|t| match t.as_str() {
                Some("linear_attention") => Ok(Qwen4ExpBlock::Gdn),
                // transformers rewrites full_attention -> qwen_sparse_attention;
                // accept both spellings, the checkpoint writes the former.
                Some("full_attention") | Some("qwen_sparse_attention") => {
                    Ok(Qwen4ExpBlock::Attention)
                }
                other => Err(StError::Header(format!(
                    "qwen4exp: unknown layer type {other:?}"
                ))),
            })
            .collect::<Result<Vec<_>, _>>()?;

        // rope facts live in the nested rope_parameters dict (rope_theta,
        // partial_rotary_factor are also mirrored at text_config top level;
        // read the nested dict as the source of truth).
        let rp = tc
            .get("rope_parameters")
            .ok_or_else(|| miss("rope_parameters"))?;
        let rope_theta = rp
            .get("rope_theta")
            .and_then(|x| x.as_f64())
            .ok_or_else(|| miss("rope_parameters.rope_theta"))? as f32;
        let partial = rp
            .get("partial_rotary_factor")
            .and_then(|x| x.as_f64())
            .ok_or_else(|| miss("rope_parameters.partial_rotary_factor"))?;
        let head_dim = u("head_dim")?;
        let rotary_dim = ((head_dim as f64) * partial).round() as usize;

        let out_gate = s("output_gate_type")?;
        if out_gate != "sigmoid" {
            return Err(StError::Header(format!(
                "qwen4exp: output_gate_type {out_gate:?} (graph assumes sigmoid)"
            )));
        }

        // ple_layer_ids is ONE-indexed (id 2 = decoder layer index 1; the
        // ple.* tensors sit under layers.1 in the shard headers).
        let ple_layers = tc
            .get("ple_layer_ids")
            .and_then(|x| x.as_array())
            .ok_or_else(|| miss("ple_layer_ids"))?
            .iter()
            .map(|x| {
                x.as_u64()
                    .filter(|&id| id >= 1)
                    .map(|id| id as usize - 1)
                    .ok_or_else(|| StError::Header("qwen4exp: bad ple_layer_ids".into()))
            })
            .collect::<Result<Vec<_>, _>>()?;

        let mtp_layers = u("mtp_num_hidden_layers")?;

        // eos SET from generation_config ([248046, 248044]); decode stops on
        // any of these. bos from text_config.
        let g: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("generation_config.json"))?)
                .map_err(|e| StError::Header(e.to_string()))?;
        let eos_ids: Vec<u32> = match g.get("eos_token_id") {
            Some(serde_json::Value::Array(a)) => a
                .iter()
                .filter_map(|x| x.as_u64())
                .map(|x| x as u32)
                .collect(),
            Some(serde_json::Value::Number(n)) => {
                n.as_u64().map(|x| x as u32).into_iter().collect()
            }
            _ => return Err(StError::Header("qwen4exp: generation_config eos".into())),
        };
        let bos_id = tc
            .get("bos_token_id")
            .and_then(|x| x.as_u64())
            .ok_or_else(|| miss("bos_token_id"))? as u32;

        Ok(Self {
            hidden: u("hidden_size")?,
            n_layer,
            blocks,
            vocab: u("vocab_size")?,
            max_pos: u("max_position_embeddings")?,
            eps: f("rms_norm_eps")?,
            n_heads: u("num_attention_heads")?,
            n_kv_heads: u("num_key_value_heads")?,
            head_dim,
            rotary_dim,
            rope_theta,
            idx_heads: u("indexer_n_heads")?,
            idx_kv_heads: u("indexer_kv_heads")?,
            idx_head_dim: u("indexer_head_dim")?,
            idx_budget: u("indexer_budget")?,
            idx_compress: u("indexer_compress_ratio")?,
            gdn_k_heads: u("linear_num_key_heads")?,
            gdn_v_heads: u("linear_num_value_heads")?,
            gdn_k_dim: u("linear_key_head_dim")?,
            gdn_v_dim: u("linear_value_head_dim")?,
            gdn_conv: u("linear_conv_kernel_dim")?,
            n_expert: u("num_experts")?,
            n_active: u("num_experts_per_tok")?,
            moe_ff: u("moe_intermediate_size")?,
            shared_ff: u("shared_expert_intermediate_size")?,
            hc_count: u("hc_count")?,
            hc_lowrank: u("hc_lowrank")?,
            ple_layers,
            ple_embed: u("ple_embed_dim")?,
            ple_conv: u("ple_conv_kernel_size")?,
            ngram_size: u("ngram_size")?,
            heads_per_ngram: u("heads_per_ngram")?,
            ngram_vocab_base: tc
                .get("ngram_vocab_size_base")
                .and_then(|x| x.as_u64())
                .ok_or_else(|| miss("ngram_vocab_size_base"))?,
            ngram_split: u("split_ngram_parts")?,
            ple_dtype: s("ple_embedding_dtype")?,
            mtp_layers,
            eos_ids,
            bos_id,
        })
    }
}

/// The PLE n-gram hash constants a GGUF carries as metadata arrays (the
/// safetensors checkpoint ships them as I64 tensors instead).
#[derive(Debug, Clone)]
pub struct Qwen4ExpPleHash {
    pub multipliers: Vec<i64>,
    pub head_offsets: Vec<i64>,
    pub head_vocab_sizes: Vec<i64>,
}

impl Qwen4ExpConfig {
    /// The same config off a llama.cpp `qwen4exp` GGUF's metadata (the
    /// Unsloth exports). Every key is required; a missing or ill-typed one
    /// is an error, never a default - the only derivations are the ones the
    /// GGUF vocabulary does not spell out: the block kinds come from
    /// `attention.compress_ratios` (non-zero = the QSA attention layer),
    /// the GDN value head dim from `ssm.inner_size / time_step_rank`, the
    /// indexer kv-head count from the first `indexer.k_proj` tensor, the
    /// vocab from `token_embd`, and the PLE width from
    /// `embedding_length_per_layer_input` x the head count.
    pub fn from_gguf(
        g: &crate::gguf::GgufFile,
        tensor_dims_of: impl Fn(&str) -> Option<Vec<usize>>,
    ) -> Result<Self, StError> {
        use crate::gguf::Value;
        let arch = g.architecture().unwrap_or("");
        if arch != "qwen4exp" {
            return Err(StError::Header(format!(
                "qwen4exp: general.architecture is {arch:?}, not qwen4exp"
            )));
        }
        let miss = |k: &str| StError::Header(format!("qwen4exp gguf: missing {arch}.{k}"));
        let u = |k: &str| -> Result<usize, StError> {
            g.arch_field(k)
                .and_then(Value::as_u64)
                .map(|x| x as usize)
                .ok_or_else(|| miss(k))
        };
        let f = |k: &str| -> Result<f32, StError> {
            g.arch_field(k)
                .and_then(Value::as_f32)
                .ok_or_else(|| miss(k))
        };
        let arr_u = |k: &str| -> Result<Vec<u64>, StError> {
            match g.arch_field(k) {
                Some(Value::Array(items)) => items
                    .iter()
                    .map(|v| v.as_u64().ok_or_else(|| miss(k)))
                    .collect(),
                _ => Err(miss(k)),
            }
        };
        // tensor geometry comes through the caller's lookup: a split family's
        // first shard carries the metadata and none of the tensors
        let tensor_dims = |name: &str| -> Result<Vec<usize>, StError> {
            tensor_dims_of(name)
                .ok_or_else(|| StError::Header(format!("qwen4exp gguf: no tensor {name}")))
        };

        let n_layer = u("block_count")?;
        let hidden = u("embedding_length")?;
        // block kinds: the QSA compress ratio is written per layer and is
        // non-zero exactly on the full-attention layers
        let ratios = arr_u("attention.compress_ratios")?;
        if ratios.len() != n_layer {
            return Err(StError::Header(format!(
                "qwen4exp gguf: attention.compress_ratios has {} entries for {n_layer} layers",
                ratios.len()
            )));
        }
        let blocks: Vec<Qwen4ExpBlock> = ratios
            .iter()
            .map(|&r| {
                if r > 0 {
                    Qwen4ExpBlock::Attention
                } else {
                    Qwen4ExpBlock::Gdn
                }
            })
            .collect();
        let idx_compress = ratios.iter().copied().max().unwrap_or(0) as usize;
        let first_attn = blocks
            .iter()
            .position(|b| *b == Qwen4ExpBlock::Attention)
            .ok_or_else(|| StError::Header("qwen4exp gguf: no attention layer".into()))?;
        let idx_head_dim = u("attention.indexer.key_length")?;
        let idx_kv_heads = {
            let d = tensor_dims(&format!("blk.{first_attn}.indexer.k_proj.weight"))?;
            if d.len() != 2 || d[1] % idx_head_dim != 0 {
                return Err(StError::Header(format!(
                    "qwen4exp gguf: indexer.k_proj is {d:?}, want [hidden, n*{idx_head_dim}]"
                )));
            }
            d[1] / idx_head_dim
        };
        let vocab = {
            let d = tensor_dims("token_embd.weight")?;
            if d.len() != 2 || d[0] != hidden {
                return Err(StError::Header(format!(
                    "qwen4exp gguf: token_embd is {d:?}, want [{hidden}, vocab]"
                )));
            }
            d[1]
        };
        let gdn_v_heads = u("ssm.time_step_rank")?;
        let gdn_inner = u("ssm.inner_size")?;
        if gdn_v_heads == 0 || gdn_inner % gdn_v_heads != 0 {
            return Err(StError::Header(format!(
                "qwen4exp gguf: ssm.inner_size {gdn_inner} is not a multiple of {gdn_v_heads} value heads"
            )));
        }
        let ple_layers: Vec<usize> = arr_u("ple.layers")?
            .into_iter()
            .map(|x| x as usize)
            .collect();
        let ngram_size = u("ple.ngram_size")?;
        let heads_per_ngram = u("ple.heads_per_ngram")?;
        let ple_heads = (ngram_size - 1) * heads_per_ngram;
        let ple_row = u("embedding_length_per_layer_input")?;
        let head_vocab = arr_u("ple.head_vocab_sizes")?;
        if head_vocab.len() != ple_heads {
            return Err(StError::Header(format!(
                "qwen4exp gguf: ple.head_vocab_sizes has {} entries for {ple_heads} heads",
                head_vocab.len()
            )));
        }
        let ngram_vocab_base = head_vocab.iter().copied().min().unwrap_or(0);
        let eos_ids: Vec<u32> = {
            let mut v = Vec::new();
            if let Some(e) = g
                .metadata
                .get("tokenizer.ggml.eos_token_id")
                .and_then(Value::as_u64)
            {
                v.push(e as u32);
            }
            if let Some(Value::Array(items)) = g.metadata.get("tokenizer.ggml.eos_token_ids") {
                v.extend(items.iter().filter_map(Value::as_u64).map(|x| x as u32));
            }
            if v.is_empty() {
                return Err(StError::Header(
                    "qwen4exp gguf: no tokenizer.ggml.eos_token_id".into(),
                ));
            }
            v.sort_unstable();
            v.dedup();
            v
        };
        // the PLE hash primes its n-gram window with this id (vLLM's
        // `ngram_context`); the converter writes it from the HF eos, which is
        // the checkpoint's bos as well
        let bos_id = u("ple.eos_token_id")? as u32;
        Ok(Self {
            hidden,
            n_layer,
            blocks,
            vocab,
            max_pos: u("context_length")?,
            eps: f("attention.layer_norm_rms_epsilon")?,
            n_heads: u("attention.head_count")?,
            n_kv_heads: u("attention.head_count_kv")?,
            head_dim: u("attention.key_length")?,
            rotary_dim: u("rope.dimension_count")?,
            rope_theta: f("rope.freq_base")?,
            idx_heads: u("attention.indexer.head_count")?,
            idx_kv_heads,
            idx_head_dim,
            idx_budget: u("attention.indexer.top_k")?,
            idx_compress,
            gdn_k_heads: u("ssm.group_count")?,
            gdn_v_heads,
            gdn_k_dim: u("ssm.state_size")?,
            gdn_v_dim: gdn_inner / gdn_v_heads,
            gdn_conv: u("ssm.conv_kernel")?,
            n_expert: u("expert_count")?,
            n_active: u("expert_used_count")?,
            moe_ff: u("expert_feed_forward_length")?,
            shared_ff: u("expert_shared_feed_forward_length")?,
            hc_count: u("hyper_connection.count")?,
            hc_lowrank: u("hyper_connection.low_rank")?,
            ple_layers,
            ple_embed: ple_row * ple_heads,
            ple_conv: u("ple.conv_kernel")?,
            ngram_size,
            heads_per_ngram,
            ngram_vocab_base,
            // one tensor, not the checkpoint's 128 shards
            ngram_split: 1,
            ple_dtype: "gguf".to_owned(),
            mtp_layers: 0,
            eos_ids,
            bos_id,
        })
    }

    /// The PLE hash constants off the GGUF metadata (see [`Qwen4ExpPleHash`]).
    pub fn ple_hash_from_gguf(g: &crate::gguf::GgufFile) -> Result<Qwen4ExpPleHash, StError> {
        use crate::gguf::Value;
        let arr_i = |k: &str| -> Result<Vec<i64>, StError> {
            match g.arch_field(k) {
                Some(Value::Array(items)) => items
                    .iter()
                    .map(|v| {
                        v.as_i64()
                            .ok_or_else(|| StError::Header(format!("qwen4exp gguf: {k}: not i64")))
                    })
                    .collect(),
                _ => Err(StError::Header(format!("qwen4exp gguf: missing {k}"))),
            }
        };
        Ok(Qwen4ExpPleHash {
            multipliers: arr_i("ple.layer_multipliers")?,
            head_offsets: arr_i("ple.head_offsets")?,
            head_vocab_sizes: arr_i("ple.head_vocab_sizes")?,
        })
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    /// The downloaded RadixArk NVFP4 checkpoint (B200 box path first, RTX
    /// box convention second); QWEN4EXP_DIR overrides.
    pub(crate) fn checkpoint_dir() -> Option<std::path::PathBuf> {
        if let Ok(d) = std::env::var("QWEN4EXP_DIR") {
            return Some(d.into());
        }
        for d in [
            "/models/Qwen3.8-Flash-Next-NVFP4",
            "/models/Qwen3.8-Flash-Next-NVFP4",
        ] {
            let p = std::path::PathBuf::from(d);
            if p.join("config.json").exists() {
                return Some(p);
            }
        }
        None
    }

    #[test]
    fn parses_flash_next_checkpoint() {
        let Some(dir) = checkpoint_dir() else {
            eprintln!("skip: no Qwen3.8-Flash-Next checkpoint present");
            return;
        };
        let c = Qwen4ExpConfig::read(&dir).expect("config parses");
        assert_eq!((c.hidden, c.n_layer, c.vocab), (2560, 48, 248320));
        let attn = c
            .blocks
            .iter()
            .filter(|b| **b == Qwen4ExpBlock::Attention)
            .count();
        assert_eq!((attn, c.n_layer - attn), (12, 36), "12 attn / 36 GDN");
        // 3 linear then 1 full, repeating: indices 3, 7, 11, ...
        assert_eq!(c.blocks[0], Qwen4ExpBlock::Gdn);
        assert_eq!(c.blocks[3], Qwen4ExpBlock::Attention);
        assert_eq!(c.blocks[47], Qwen4ExpBlock::Attention);
        assert_eq!(
            (c.n_heads, c.n_kv_heads, c.head_dim, c.rotary_dim),
            (24, 2, 256, 64)
        );
        assert!((c.rope_theta - 1e7).abs() < 1.0);
        assert_eq!((c.gdn_k_heads, c.gdn_v_heads, c.gdn_k_dim), (16, 48, 128));
        assert_eq!(c.gdn_qkv_rows(), 10240);
        assert_eq!(c.gdn_z_rows(), 6144);
        assert_eq!(c.attn_q_rows(), 12288);
        assert_eq!(
            (c.n_expert, c.n_active, c.moe_ff, c.shared_ff),
            (512, 10, 640, 640)
        );
        assert_eq!((c.hc_count, c.hc_lowrank, c.hc_width()), (4, 320, 10240));
        assert_eq!(c.ple_layers, vec![1], "one-indexed id 2 -> decoder layer 1");
        assert_eq!((c.ple_embed, c.ple_heads(), c.ngram_split), (2560, 16, 128));
        assert_eq!(c.ple_dtype, "float8_e4m3fn");
        assert_eq!(c.mtp_layers, 1);
        assert_eq!(c.eos_ids, vec![248046, 248044]);
        assert_eq!(c.bos_id, 248044);
    }
}

#[cfg(test)]
mod inventory_tests {
    use super::tests::checkpoint_dir;
    use super::*;
    use crate::modelopt::nvfp4_view;
    use crate::safetensors::{ShardedSafetensors, StDtype};

    /// Tensor-level oracle (the pattern): every plane the loader will
    /// walk must exist with the exact dtype+shape the graph assumes, checked
    /// against the real checkpoint before any GPU code exists. Collects every
    /// mismatch instead of stopping at the first.
    #[test]
    fn flash_next_full_inventory() {
        let Some(dir) = checkpoint_dir() else {
            eprintln!("skip: no Qwen3.8-Flash-Next checkpoint present");
            return;
        };
        let c = Qwen4ExpConfig::read(&dir).expect("config");
        let st = ShardedSafetensors::open_dir(&dir).expect("shards");
        let mut errs: Vec<String> = Vec::new();
        let mut chk = |errs: &mut Vec<String>, name: &str, dt: StDtype, shape: &[usize]| match st
            .bytes(name)
        {
            None => errs.push(format!("{name}: MISSING")),
            Some((t, _)) => {
                if t.dtype != dt || t.shape != shape {
                    errs.push(format!(
                        "{name}: want {dt:?} {shape:?}, got {:?} {:?}",
                        t.dtype, t.shape
                    ));
                }
            }
        };
        let h = c.hidden; // 2560
        let hw = c.hc_width(); // 10240
        let lr = c.hc_lowrank; // 320

        let hc = |errs: &mut Vec<String>,
                  chk: &mut dyn FnMut(&mut Vec<String>, &str, StDtype, &[usize]),
                  pfx: String| {
            chk(errs, &format!("{pfx}.hc_norm.weight"), StDtype::Bf16, &[hw]);
            chk(
                errs,
                &format!("{pfx}.input_mix_weight_down.weight"),
                StDtype::Bf16,
                &[lr, hw],
            );
            chk(
                errs,
                &format!("{pfx}.input_mix_weight_up.weight"),
                StDtype::Bf16,
                &[hw, lr],
            );
            chk(
                errs,
                &format!("{pfx}.block_inject_weight.weight"),
                StDtype::Bf16,
                &[c.hc_count, hw],
            );
        };

        for li in 0..c.n_layer {
            let p = format!("model.language_model.layers.{li}");
            hc(&mut errs, &mut chk, format!("{p}.attn_hyper_connection"));
            hc(&mut errs, &mut chk, format!("{p}.mlp_hyper_connection"));
            match c.blocks[li] {
                Qwen4ExpBlock::Gdn => {
                    let g = format!("{p}.linear_attn");
                    chk(
                        &mut errs,
                        &format!("{g}.in_proj_qkv.weight"),
                        StDtype::Bf16,
                        &[c.gdn_qkv_rows(), h],
                    );
                    chk(
                        &mut errs,
                        &format!("{g}.in_proj_z.weight"),
                        StDtype::Bf16,
                        &[c.gdn_z_rows(), h],
                    );
                    chk(
                        &mut errs,
                        &format!("{g}.in_proj_a.weight"),
                        StDtype::Bf16,
                        &[c.gdn_v_heads, h],
                    );
                    chk(
                        &mut errs,
                        &format!("{g}.in_proj_b.weight"),
                        StDtype::Bf16,
                        &[c.gdn_v_heads, h],
                    );
                    chk(
                        &mut errs,
                        &format!("{g}.conv1d.weight"),
                        StDtype::Bf16,
                        &[c.gdn_qkv_rows(), 1, c.gdn_conv],
                    );
                    chk(
                        &mut errs,
                        &format!("{g}.A_log"),
                        StDtype::Bf16,
                        &[c.gdn_v_heads],
                    );
                    chk(
                        &mut errs,
                        &format!("{g}.dt_bias"),
                        StDtype::Bf16,
                        &[c.gdn_v_heads],
                    );
                    chk(
                        &mut errs,
                        &format!("{g}.norm.weight"),
                        StDtype::Bf16,
                        &[c.gdn_v_dim],
                    );
                    chk(
                        &mut errs,
                        &format!("{g}.out_proj.weight"),
                        StDtype::Bf16,
                        &[h, c.gdn_z_rows()],
                    );
                }
                Qwen4ExpBlock::Attention => {
                    let a = format!("{p}.self_attn");
                    chk(
                        &mut errs,
                        &format!("{a}.q_proj.weight"),
                        StDtype::Bf16,
                        &[c.attn_q_rows(), h],
                    );
                    chk(
                        &mut errs,
                        &format!("{a}.k_proj.weight"),
                        StDtype::Bf16,
                        &[c.n_kv_heads * c.head_dim, h],
                    );
                    chk(
                        &mut errs,
                        &format!("{a}.v_proj.weight"),
                        StDtype::Bf16,
                        &[c.n_kv_heads * c.head_dim, h],
                    );
                    chk(
                        &mut errs,
                        &format!("{a}.o_proj.weight"),
                        StDtype::Bf16,
                        &[h, c.attn_o_in()],
                    );
                    chk(
                        &mut errs,
                        &format!("{a}.q_norm.weight"),
                        StDtype::Bf16,
                        &[c.head_dim],
                    );
                    chk(
                        &mut errs,
                        &format!("{a}.k_norm.weight"),
                        StDtype::Bf16,
                        &[c.head_dim],
                    );
                    chk(
                        &mut errs,
                        &format!("{a}.indexer.index_qk_proj.weight"),
                        StDtype::Bf16,
                        &[
                            c.idx_heads * c.idx_head_dim + c.idx_kv_heads * c.idx_head_dim,
                            h,
                        ],
                    );
                    chk(
                        &mut errs,
                        &format!("{a}.indexer.q_layernorm.weight"),
                        StDtype::Bf16,
                        &[c.idx_head_dim],
                    );
                    chk(
                        &mut errs,
                        &format!("{a}.indexer.k_layernorm.weight"),
                        StDtype::Bf16,
                        &[c.idx_head_dim],
                    );
                }
            }
            // MoE: router + shared expert bf16; 512 routed experts NVFP4.
            chk(
                &mut errs,
                &format!("{p}.mlp.gate.weight"),
                StDtype::Bf16,
                &[c.n_expert, h],
            );
            chk(
                &mut errs,
                &format!("{p}.mlp.shared_expert.gate_proj.weight"),
                StDtype::Bf16,
                &[c.shared_ff, h],
            );
            chk(
                &mut errs,
                &format!("{p}.mlp.shared_expert.up_proj.weight"),
                StDtype::Bf16,
                &[c.shared_ff, h],
            );
            chk(
                &mut errs,
                &format!("{p}.mlp.shared_expert.down_proj.weight"),
                StDtype::Bf16,
                &[h, c.shared_ff],
            );
            chk(
                &mut errs,
                &format!("{p}.mlp.shared_expert_gate.weight"),
                StDtype::Bf16,
                &[1, h],
            );
            for e in 0..c.n_expert {
                for (plane, n, k) in [
                    ("gate_proj", c.moe_ff, h),
                    ("up_proj", c.moe_ff, h),
                    ("down_proj", h, c.moe_ff),
                ] {
                    match nvfp4_view(&st, &format!("{p}.mlp.experts.{e}.{plane}")) {
                        Err(e2) => errs.push(format!("L{li} expert {e} {plane}: {e2}")),
                        Ok(v) if (v.n, v.k) != (n, k) => errs.push(format!(
                            "L{li} expert {e} {plane}: [{}, {}] want [{n}, {k}]",
                            v.n, v.k
                        )),
                        Ok(_) => {}
                    }
                }
                if errs.len() > 40 {
                    break; // enough signal
                }
            }
            if errs.len() > 40 {
                break;
            }
        }

        // PLE layer (decoder index 1)
        for &pl in &c.ple_layers {
            let p = format!("model.language_model.layers.{pl}.ple");
            chk(
                &mut errs,
                &format!("{p}.key_proj.weight"),
                StDtype::Bf16,
                &[hw, h],
            );
            chk(
                &mut errs,
                &format!("{p}.value_proj.weight"),
                StDtype::Bf16,
                &[h, h],
            );
            chk(
                &mut errs,
                &format!("{p}.conv1d.weight"),
                StDtype::Bf16,
                &[hw, 1, c.ple_conv],
            );
            for nrm in ["norm_key", "norm_query", "norm_conv"] {
                chk(
                    &mut errs,
                    &format!("{p}.{nrm}.weight"),
                    StDtype::Bf16,
                    &[hw],
                );
            }
            let emb = format!("{p}.ple_embedding");
            chk(
                &mut errs,
                &format!("{emb}.ngram_embedding.weight_scale"),
                StDtype::Bf16,
                &[1],
            );
            // shard rows: base rounded up to a multiple of 128 then / 128
            let rows = {
                let d = 128u64;
                ((c.ngram_vocab_base * 16).div_ceil(d) * d / c.ngram_split as u64) as usize
            };
            for sh in [0, 63, 127] {
                match st.bytes(&format!("{emb}.ngram_embedding.shard_{sh}.weight")) {
                    None => errs.push(format!("ple shard_{sh}: MISSING")),
                    Some((t, _)) => {
                        if t.dtype != StDtype::F8E4m3 || t.shape[1] != c.ple_embed / c.ple_heads() {
                            errs.push(format!("ple shard_{sh}: {:?} {:?}", t.dtype, t.shape));
                        }
                        let _ = rows; // documented, not asserted: per-head vocab
                        // primes make exact rows shard-dependent
                    }
                }
            }
            // I64 hash buffers: values pinned from the checkpoint audit.
            let (t, b) = st
                .bytes(&format!("{emb}.layer_multipliers"))
                .expect("multipliers");
            assert_eq!((t.dtype, t.shape.as_slice()), (StDtype::I64, &[3usize][..]));
            let vals: Vec<i64> = b
                .as_chunks::<8>()
                .0
                .iter()
                .map(|c| i64::from_le_bytes(*c))
                .collect();
            assert_eq!(vals, vec![23703573157769, 20109073645365, 8052911324071]);
            chk(
                &mut errs,
                &format!("{emb}.ngram_heads_vocab_sizes"),
                StDtype::I64,
                &[c.ple_heads()],
            );
            chk(
                &mut errs,
                &format!("{emb}.ngram_heads_offsets"),
                StDtype::I64,
                &[c.ple_heads()],
            );
        }

        // top level + mixer + MTP spots + vision spot
        chk(
            &mut errs,
            "model.language_model.embed_tokens.weight",
            StDtype::Bf16,
            &[c.vocab, h],
        );
        chk(&mut errs, "lm_head.weight", StDtype::Bf16, &[c.vocab, h]);
        chk(
            &mut errs,
            "model.language_model.hyper_connection_mixer.hc_norm.weight",
            StDtype::Bf16,
            &[hw],
        );
        chk(&mut errs, "mtp.fc_hidden.weight", StDtype::Bf16, &[h, h]);
        chk(&mut errs, "mtp.fc_embedding.weight", StDtype::Bf16, &[h, h]);
        chk(
            &mut errs,
            "mtp.pre_fc_norm_hidden.weight",
            StDtype::Bf16,
            &[hw],
        );
        chk(
            &mut errs,
            "mtp.pre_fc_norm_embedding.weight",
            StDtype::Bf16,
            &[h],
        );
        // MTP experts: bf16 FUSED planes, bare names (no .weight suffix)
        chk(
            &mut errs,
            "mtp.layers.0.mlp.experts.gate_up_proj",
            StDtype::Bf16,
            &[c.n_expert, 2 * c.moe_ff, h],
        );
        chk(
            &mut errs,
            "mtp.layers.0.mlp.experts.down_proj",
            StDtype::Bf16,
            &[c.n_expert, h, c.moe_ff],
        );
        chk(
            &mut errs,
            "model.visual.patch_embed.proj.weight",
            StDtype::Bf16,
            &[1152, 3, 2, 16, 16],
        );

        assert!(
            errs.is_empty(),
            "{} inventory mismatches:\n{}",
            errs.len(),
            errs.join("\n")
        );
    }
}
