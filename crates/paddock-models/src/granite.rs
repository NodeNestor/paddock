//! `GraniteForCausalLM` config.json - the safetensors-primary lane's geometry.
//!
//! The GGUF path reads all of this out of the header instead (see the engine's
//! `granite/load.rs`); this exists for checkpoints that ship no GGUF at all,
//! which today means IBM's NVFP4 exports of granite 4.2.
//!
//! Granite is a plain dense decoder - GQA + SwiGLU MLP + RMSNorm, no Mamba and
//! no experts, verified against the 4.2 tensor index (1923 tensors, zero
//! matching `mamba|expert|ssm|conv1d`). What makes it Granite
//! rather than Llama is four SCALARS, and every one of them fails silently if
//! it is defaulted rather than read:
//!
//! | scalar | 4.1-30b | 4.2-30b | what it multiplies |
//! |---|---|---|---|
//! | `embedding_multiplier` | 12.0 | 1.0 | token embeddings after the gather |
//! | `residual_multiplier`  | 0.175 | 1.0 | both residual adds |
//! | `logits_scaling`       | 16.0 | 1.0 | DIVIDES the logits at the head |
//! | `attention_multiplier` | 0.0078125 | 0.0078125 | replaces 1/sqrt(head_dim) as the KQ scale |
//!
//! 4.2 sets three of them to identity, which is exactly why they are required
//! here rather than optional-with-a-default: a reader that defaults a missing
//! multiplier to 1.0 cannot tell 4.2 (which means 1.0) from a 4.1-shaped file
//! whose key it failed to parse (which means 12.0, and fluent wrong output).
//!
//! Note `attention_multiplier` is 1/128 = 1/head_dim, not 1/sqrt(128) ≈ 0.0884.
//! It is not a rounding of the usual scale; it is a different one.

use std::path::Path;

use crate::safetensors::StError;

/// Geometry + the four Granite scalars, read from `config.json`.
#[derive(Debug, Clone)]
pub struct GraniteConfig {
    pub hidden: usize,
    pub n_layer: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub n_ff: usize,
    pub vocab: usize,
    pub max_pos: usize,
    pub eps: f32,
    pub rope_theta: f32,

    /// The four scalars. See the module docs - none of these may be defaulted.
    pub embedding_scale: f32,
    pub residual_scale: f32,
    pub logit_scale: f32,
    pub attention_scale: f32,

    /// `false` on 4.2 (a real `lm_head.weight` ships); `true` on 4.1, where the
    /// head reuses the embedding matrix. Read rather than inferred, but the
    /// loader still branches on the TENSOR'S PRESENCE - 4.1's GGUF ships an
    /// `output.weight` regardless of what its config says.
    pub tie_word_embeddings: bool,

    pub bos_id: u32,
    pub eos_id: u32,
}

impl GraniteConfig {
    pub fn read(dir: &Path) -> Result<Self, StError> {
        let v: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)
            .map_err(|e| StError::Header(e.to_string()))?;
        let miss = |k: &str| StError::Header(format!("granite config.json: missing {k}"));
        let u = |k: &str| {
            v.get(k)
                .and_then(|x| x.as_u64())
                .map(|x| x as usize)
                .ok_or_else(|| miss(k))
        };
        let f = |k: &str| {
            v.get(k)
                .and_then(|x| x.as_f64())
                .map(|x| x as f32)
                .ok_or_else(|| miss(k))
        };

        let model_type = v.get("model_type").and_then(|x| x.as_str()).unwrap_or("");
        if model_type != "granite" {
            return Err(StError::Header(format!(
                "not a granite checkpoint (model_type {model_type:?})"
            )));
        }

        let hidden = u("hidden_size")?;
        let n_heads = u("num_attention_heads")?;
        if n_heads == 0 || hidden % n_heads != 0 {
            return Err(StError::Header(format!(
                "granite: hidden_size {hidden} is not a multiple of num_attention_heads {n_heads}"
            )));
        }
        let n_kv_heads = u("num_key_value_heads")?;
        if n_kv_heads == 0 || n_heads % n_kv_heads != 0 {
            return Err(StError::Header(format!(
                "granite: num_attention_heads {n_heads} is not a multiple of \
                 num_key_value_heads {n_kv_heads}"
            )));
        }

        // transformers 4.57 moved rope into a `rope_parameters` object and kept
        // the flat key alongside it. Accept either, prefer the nested one --
        // whichever a future export drops, the other still answers.
        let rope_theta = v
            .get("rope_parameters")
            .and_then(|r| r.get("rope_theta"))
            .and_then(|x| x.as_f64())
            .map(|x| x as f32)
            .or_else(|| {
                v.get("rope_theta")
                    .and_then(|x| x.as_f64())
                    .map(|x| x as f32)
            })
            .ok_or_else(|| miss("rope_theta (or rope_parameters.rope_theta)"))?;
        if v.get("rope_scaling").is_some_and(|x| !x.is_null()) {
            return Err(StError::Header(
                "granite: rope_scaling is set - no shipped granite geometry uses it, and \
                 serving it as plain rope would be silently wrong"
                    .into(),
            ));
        }

        let logit_scale = f("logits_scaling")?;
        if logit_scale == 0.0 {
            return Err(StError::Header(
                "granite: logits_scaling 0 would divide the logits by zero".into(),
            ));
        }

        let id = |k: &str, d: u32| v.get(k).and_then(|x| x.as_u64()).map_or(d, |x| x as u32);

        Ok(Self {
            hidden,
            n_layer: u("num_hidden_layers")?,
            n_heads,
            n_kv_heads,
            head_dim: hidden / n_heads,
            n_ff: u("intermediate_size")?,
            vocab: u("vocab_size")?,
            max_pos: u("max_position_embeddings")?,
            eps: f("rms_norm_eps")?,
            rope_theta,
            embedding_scale: f("embedding_multiplier")?,
            residual_scale: f("residual_multiplier")?,
            logit_scale,
            attention_scale: f("attention_multiplier")?,
            tie_word_embeddings: v
                .get("tie_word_embeddings")
                .and_then(|x| x.as_bool())
                .unwrap_or(false),
            bos_id: id("bos_token_id", 0),
            eos_id: id("eos_token_id", 0),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(dir: &Path, body: &str) {
        std::fs::write(dir.join("config.json"), body).unwrap();
    }

    fn cfg_42_30b() -> String {
        r#"{
          "architectures": ["GraniteForCausalLM"], "model_type": "granite",
          "hidden_size": 4096, "num_hidden_layers": 64,
          "num_attention_heads": 32, "num_key_value_heads": 8,
          "intermediate_size": 32768, "vocab_size": 100352,
          "max_position_embeddings": 131072, "rms_norm_eps": 1e-05,
          "rope_parameters": {"rope_theta": 50000000, "rope_type": "default"},
          "rope_scaling": null,
          "attention_multiplier": 0.0078125, "embedding_multiplier": 1.0,
          "residual_multiplier": 1.0, "logits_scaling": 1.0,
          "tie_word_embeddings": false,
          "bos_token_id": 100283, "eos_token_id": 100257
        }"#
        .to_owned()
    }

    #[test]
    fn reads_the_shipped_42_30b_geometry() {
        let d = tempfile::tempdir().unwrap();
        write(d.path(), &cfg_42_30b());
        let c = GraniteConfig::read(d.path()).unwrap();
        assert_eq!(
            (c.hidden, c.n_layer, c.n_heads, c.n_kv_heads),
            (4096, 64, 32, 8)
        );
        assert_eq!(c.head_dim, 128);
        assert_eq!(c.n_ff, 32768);
        assert_eq!(c.vocab, 100352);
        assert_eq!(c.rope_theta, 50_000_000.0);
        // the scalar that is not 1/sqrt(head_dim)
        assert_eq!(c.attention_scale, 0.0078125);
        assert_eq!(c.attention_scale, 1.0 / 128.0);
        assert!(!c.tie_word_embeddings);
    }

    /// A missing multiplier must be an error, never a 1.0 default: on a
    /// 4.1-shaped file that silently swaps ×12 embeddings for ×1 and produces
    /// fluent wrong output instead of a failure.
    #[test]
    fn a_missing_scalar_is_an_error_not_a_default() {
        for key in [
            "embedding_multiplier",
            "residual_multiplier",
            "logits_scaling",
            "attention_multiplier",
        ] {
            let d = tempfile::tempdir().unwrap();
            let mut v: serde_json::Value = serde_json::from_str(&cfg_42_30b()).unwrap();
            v.as_object_mut().unwrap().remove(key);
            write(d.path(), &v.to_string());
            let e = GraniteConfig::read(d.path()).expect_err("must refuse");
            assert!(
                format!("{e:?}").contains(key),
                "error should name {key}: {e:?}"
            );
        }
    }

    #[test]
    fn another_family_is_refused_by_model_type() {
        let d = tempfile::tempdir().unwrap();
        write(
            d.path(),
            r#"{"model_type": "nemotron_h", "hidden_size": 2688}"#,
        );
        let e = GraniteConfig::read(d.path()).expect_err("must refuse");
        assert!(format!("{e:?}").contains("nemotron_h"));
    }

    #[test]
    fn a_zero_logit_scale_is_refused_before_it_divides() {
        let d = tempfile::tempdir().unwrap();
        let mut v: serde_json::Value = serde_json::from_str(&cfg_42_30b()).unwrap();
        v["logits_scaling"] = serde_json::json!(0.0);
        write(d.path(), &v.to_string());
        assert!(GraniteConfig::read(d.path()).is_err());
    }
}
