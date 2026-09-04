//! Qwen3-ForcedAligner-0.6B on the qwen3_asr chassis. Reference:
//! transformers main `modeling_qwen3_asr.py` (`Qwen3ASRForTokenClassification`
//! is literally `GenericForTokenClassification` over the same base model) and
//! `processing_qwen3_asr.py` (packing + decode, ported runner-side).
//!
//! The checkpoint is a Qwen3-ASR: the same audio tower, the same projector,
//! a stock Qwen3-0.6B text stack - the one new weight is `score.weight`
//! [n_labels=5000, hidden], a bias-free classification head over 80 ms time
//! bins that replaces the LM head. So this file does not build a new model:
//! it builds a `GpuQwen3Asr` from the HF safetensors (first safetensors-
//! PRIMARY lane - the fp8-native and dflash loaders were sideloads) with
//! `score` mounted as `lm_head` and `hp.n_vocab = n_labels`, and adds the one
//! forward the aligner needs: a single causal prefill that reads the head at
//! the `<timestamp>` rows instead of the last row. No decode loop, no
//! sampling, no paged pool - each request is one prefill at slot 0.
//!
//! Weights ride the same residency classes the GGUF path produces: big
//! planes host-quantized to genuine Q8_0 blocks onto the int8 ladder (the
//! laguna dflash stage-D recipe), norms f32. That is a deliberate deviation
//! from exact-logit parity with the BF16 reference - the acceptance level is
//! decoded TIMESTAMPS on the battery oracle, not logits (llama.cpp cannot
//! serve this class, so transformers is the reference
//! and same-weights-exact would need a bf16 GEMM lane this head does not
//! earn).

use std::path::Path;
use std::sync::Arc;

use half::f16;
use paddock_kernels::reference::ops::YarnRope;
use paddock_models::ggml_type::GgmlType;
use paddock_models::safetensors::{AlignerConfig, ShardedSafetensors, StDtype};

use crate::audio::MelFeatures;
use crate::gpu::{DeviceTensor, GpuExecutor, KvDtype, QuantTensor, QuantW};
use crate::gpu_model::gpt_oss::GpuModelError;
use crate::gpu_model::qwen35::{prefill_add_norm_quant, prefill_mm_pre_any};

use super::{AsrLayer, AudioTower, GpuQwen3Asr, Hparams, TokEmbd};

/// What the serving layer needs beyond the model itself: the packing ids and
/// the bin width, straight off the checkpoint's config.
pub struct AlignerMeta {
    pub audio_token_id: u32,
    pub timestamp_token_id: u32,
    /// milliseconds per predicted class
    pub segment_ms: f32,
    pub n_labels: usize,
}

// bf16 conversion + Q8_0 host-quant - the laguna dflash recipe (dflash.rs),
// kept local because that module is private to its family. Same math, same
// loud handling of non-finite values.

pub(super) fn bf16_to_f16(bytes: &[u8]) -> (Vec<f16>, usize) {
    let mut out = Vec::with_capacity(bytes.len() / 2);
    let mut bad = 0usize;
    for c in bytes.as_chunks::<2>().0 {
        let bits = u16::from_le_bytes(*c);
        let f = f32::from_bits((bits as u32) << 16);
        if !f.is_finite() || f.abs() > f16::MAX.to_f32() {
            bad += 1;
        }
        out.push(f16::from_f32(f));
    }
    (out, bad)
}

pub(super) fn bf16_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| f32::from_bits((u16::from_le_bytes(*c) as u32) << 16))
        .collect()
}

fn q8_0_blocks(vals: &[f32], bad: &mut usize) -> Vec<u8> {
    debug_assert_eq!(vals.len() % 32, 0);
    let mut out = Vec::with_capacity(vals.len() / 32 * 34);
    for blk in vals.as_chunks::<32>().0 {
        let mut amax = 0.0f32;
        for &v in blk {
            if !v.is_finite() {
                *bad += 1;
            }
            amax = amax.max(v.abs());
        }
        let d = amax / 127.0;
        let id = if d > 0.0 { 1.0 / d } else { 0.0 };
        out.extend_from_slice(&f16::from_f32(d).to_le_bytes());
        for &v in blk {
            out.push((v * id).round() as i8 as u8);
        }
    }
    out
}

/// Fetch a named BF16 tensor's raw bytes, shape-checked.
pub(super) fn st_bf16<'a>(
    st: &'a ShardedSafetensors,
    name: &str,
    shape: &[usize],
) -> Result<&'a [u8], GpuModelError> {
    let (t, b) = st
        .bytes(name)
        .ok_or_else(|| GpuModelError::MissingMeta(format!("aligner tensor {name}")))?;
    if t.dtype != StDtype::Bf16 || t.shape != shape {
        return Err(GpuModelError::Unsupported(format!(
            "aligner {name}: {:?} {:?} (want BF16 {shape:?})",
            t.dtype, t.shape
        )));
    }
    Ok(b)
}

impl GpuQwen3Asr {
    /// Build the aligner from the HF checkpoint directory (model.safetensors
    /// + config.json). `score` mounts as `lm_head`, `n_vocab` becomes the
    ///   label count - everything downstream of the loader is the stock family.
    pub fn load_aligner(
        exec: Arc<GpuExecutor>,
        dir: &Path,
        max_ctx: usize,
    ) -> Result<(Self, AlignerMeta), GpuModelError> {
        let cfg = AlignerConfig::read(&dir.join("config.json"))
            .map_err(|e| GpuModelError::Unsupported(format!("aligner config: {e}")))?;
        let st = ShardedSafetensors::open_dir(dir)
            .map_err(|e| GpuModelError::Unsupported(format!("aligner safetensors: {e}")))?;
        let total: u64 = std::fs::read_dir(dir)
            .ok()
            .into_iter()
            .flatten()
            .flatten()
            .filter(|e| e.path().extension().is_some_and(|x| x == "safetensors"))
            .filter_map(|e| e.metadata().ok().map(|m| m.len()))
            .sum();
        exec.vram_load_gate(total, "qwen3-aligner")
            .map_err(GpuModelError::WontFit)?;
        // Single-stream engine - must precede all allocs (see load.rs).
        exec.disable_event_tracking();

        if cfg.n_kv_heads == 0 || cfg.n_heads % cfg.n_kv_heads != 0 {
            return Err(GpuModelError::Unsupported(format!(
                "aligner: head_count {} not a multiple of head_count_kv {}",
                cfg.n_heads, cfg.n_kv_heads
            )));
        }
        if max_ctx > cfg.max_pos {
            return Err(GpuModelError::Unsupported(format!(
                "aligner: requested context {max_ctx} exceeds the trained {}",
                cfg.max_pos
            )));
        }
        let rope = YarnRope::new(
            cfg.head_dim,
            cfg.rope_theta,
            1.0,
            cfg.max_pos,
            0.0,
            1.0,
            32.0,
            1.0,
        )
        .kernel_params();
        let drv = |e: cudarc::driver::DriverError| crate::gpu::from_driver(e);

        let mut bad = 0usize;
        let mut bytes = 0u64;
        // Big plane -> genuine Q8_0 blocks -> the int8 ladder. dims follow the
        // GGUF convention the family reads: [in, out].
        let e = exec.clone();
        let mut qw = |name: &str, out: usize, inn: usize| -> Result<QuantW, GpuModelError> {
            let b = st_bf16(&st, name, &[out, inn])?;
            let blocks = q8_0_blocks(&bf16_to_f32(b), &mut bad);
            let w = e.repack_q8_blocks(&blocks, vec![inn, out])?;
            bytes += (w.data.len() + w.scale.len()) as u64;
            Ok(QuantW::Q8(w))
        };
        let e2 = exec.clone();
        let norm = |name: &str, n: usize| -> Result<DeviceTensor, GpuModelError> {
            let b = st_bf16(&st, name, &[n])?;
            let v = bf16_to_f32(b);
            Ok(DeviceTensor {
                buf: e2.stream.clone_htod(&v).map_err(drv)?,
                dims: vec![n],
            })
        };

        let (h, hd, ff) = (cfg.hidden, cfg.head_dim, cfg.intermediate);
        let q_dim = cfg.n_heads * hd;
        let kv_dim = cfg.n_kv_heads * hd;
        let mut layers = Vec::with_capacity(cfg.n_layer);
        for i in 0..cfg.n_layer {
            let p = |s: &str| format!("model.language_model.layers.{i}.{s}");
            layers.push(AsrLayer {
                attn_norm: norm(&p("input_layernorm.weight"), h)?,
                wq: qw(&p("self_attn.q_proj.weight"), q_dim, h)?,
                wk: qw(&p("self_attn.k_proj.weight"), kv_dim, h)?,
                wv: qw(&p("self_attn.v_proj.weight"), kv_dim, h)?,
                q_norm: norm(&p("self_attn.q_norm.weight"), hd)?,
                k_norm: norm(&p("self_attn.k_norm.weight"), hd)?,
                wo: qw(&p("self_attn.o_proj.weight"), h, q_dim)?,
                ffn_norm: norm(&p("post_attention_layernorm.weight"), h)?,
                gate: qw(&p("mlp.gate_proj.weight"), ff, h)?,
                up: qw(&p("mlp.up_proj.weight"), ff, h)?,
                down: qw(&p("mlp.down_proj.weight"), h, ff)?,
            });
        }
        let output_norm = norm("model.language_model.norm.weight", h)?;
        // The head: `score`, not an lm_head - but plane-shaped exactly like
        // one, so it mounts in the lm_head seat and the label count becomes
        // the "vocab". Every head kernel downstream is width-generic.
        let lm_head = qw("score.weight", cfg.n_labels, h)?;

        // Embedding table, host-quantized to Q8_0 in row chunks (the whole
        // table as f32 would be a ~600 MB transient for nothing).
        let tok_embd = {
            let b = st_bf16(
                &st,
                "model.language_model.embed_tokens.weight",
                &[cfg.vocab, h],
            )?;
            let mut blocks = Vec::with_capacity(cfg.vocab * h / 32 * 34);
            for chunk in b.chunks(8192 * h * 2) {
                blocks.extend(q8_0_blocks(&bf16_to_f32(chunk), &mut bad));
            }
            bytes += blocks.len() as u64;
            TokEmbd::Q8(QuantTensor {
                bytes: exec.stream.clone_htod(&blocks).map_err(drv)?,
                ty: GgmlType::Q8_0,
                dims: vec![h, cfg.vocab],
            })
        };

        if bad > 0 {
            return Err(GpuModelError::Unsupported(format!(
                "aligner: {bad} non-finite weight values - refusing a poisoned checkpoint"
            )));
        }

        let tower = AudioTower::load_st(exec.clone(), &st, &cfg)?;
        if tower.out_dim != h {
            return Err(GpuModelError::Unsupported(format!(
                "aligner: tower projects {} but the decoder is {h}-wide",
                tower.out_dim
            )));
        }

        let weights_bytes = bytes + tower.weight_bytes() as u64;
        tracing::info!(
            layers = cfg.n_layer,
            labels = cfg.n_labels,
            weight_mib = weights_bytes / (1 << 20),
            "qwen3 forced aligner loaded (safetensors, Q8_0 load-time repack)"
        );

        let meta = AlignerMeta {
            audio_token_id: cfg.audio_token_id,
            timestamp_token_id: cfg.timestamp_token_id,
            segment_ms: cfg.segment_ms,
            n_labels: cfg.n_labels,
        };
        Ok((
            Self {
                exec,
                hp: Hparams {
                    n_layer: cfg.n_layer,
                    n_embd: h,
                    n_heads: cfg.n_heads,
                    n_kv_heads: cfg.n_kv_heads,
                    head_dim: hd,
                    n_ff: ff,
                    n_vocab: cfg.n_labels,
                    eps: cfg.eps,
                    rope,
                },
                max_ctx,
                kv_dtype: KvDtype::Fp16,
                tok_embd,
                layers,
                output_norm,
                lm_head,
                tower: Some(tower),
                decode: None,
                scratch: None,
                prefill: None,
                batch: None,
                chunked: Vec::new(),
                weights_bytes,
            },
            meta,
        ))
    }

    /// One alignment: encode the clip, run the packed sequence through the
    /// causal stack at slot 0, and return the argmax time-bin at each
    /// `<timestamp>` row (`ts_rows`, ascending indices into `ids`). The pad
    /// run at `splice_at` must be exactly the tower's token count for this
    /// clip - a mismatch means the caller's packing math diverged from the
    /// conv stem, which must fail loudly, not misalign silently.
    pub fn align_bins(
        &mut self,
        ids: &[u32],
        mel: &MelFeatures,
        splice_at: usize,
        n_audio: usize,
        ts_rows: &[usize],
    ) -> Result<Vec<u32>, GpuModelError> {
        assert!(!ts_rows.is_empty() && !ids.is_empty());
        // every alignment is an independent sequence at slot 0; attention is
        // position-bounded so rewinding the cursor is the reset
        if let Some(ds) = self.decode.as_mut() {
            ds.pos = 0;
        }
        let out = self
            .tower
            .as_mut()
            .expect("aligner always loads its tower")
            .encode(mel)?;
        if out.n_tokens != n_audio {
            return Err(GpuModelError::Unsupported(format!(
                "aligner: packed {n_audio} audio rows but the tower produced {}",
                out.n_tokens
            )));
        }
        self.prefill_body(ids, &[(splice_at, out)])?;

        // ── the timestamp head ──
        // Stage the <timestamp> rows compactly, one norm+quant, one head GEMM
        // [n_ts, n_labels], argmax on host. Reuses the prefill scratch: d_xn
        // as the compact stage, d_proj as the norm output (both are dead
        // after the layer loop, and both are ≥ n_ts rows by construction).
        let exec = self.exec.clone();
        let (embd, eps, n_labels) = (self.hp.n_embd, self.hp.eps, self.hp.n_vocab);
        let n_ts = ts_rows.len();
        let sc = self.prefill.as_mut().expect("prefill scratch");
        for (i, &row) in ts_rows.iter().enumerate() {
            debug_assert!(row < ids.len());
            exec.copy_region(&sc.d_x, row * embd, &mut sc.d_xn, i * embd, embd)?;
        }
        std::mem::swap(&mut sc.d_x, &mut sc.d_xn);
        prefill_add_norm_quant(
            &exec,
            &mut sc.d_x,
            None,
            false,
            &self.output_norm.buf,
            &mut sc.d_proj,
            false,
            &mut sc.d_pxq,
            &mut sc.d_pxs,
            &mut sc.d_yq,
            embd,
            n_ts,
            eps,
        )?;
        let mut d_log = exec.alloc(n_ts * n_labels)?;
        prefill_mm_pre_any(
            &exec,
            &self.lm_head,
            &sc.d_pxq,
            &sc.d_pxs,
            &sc.d_yq,
            &mut sc.xsums,
            &mut sc.ssums,
            &mut sc.d_skfix,
            &mut d_log,
            n_ts,
        )?;
        let host = exec.to_host_len(&d_log, n_ts * n_labels)?;
        Ok(host
            .chunks_exact(n_labels)
            .map(|row| {
                let mut best = 0usize;
                for (j, &v) in row.iter().enumerate() {
                    if v > row[best] {
                        best = j;
                    }
                }
                best as u32
            })
            .collect())
    }
}
