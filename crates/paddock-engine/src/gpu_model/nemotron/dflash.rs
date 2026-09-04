//! DFlash drafter for nemotron (C2) - the official
//! `nvidia/...-NVFP4-DFlash` checkpoint: a 6-layer qwen3-class dense GQA
//! model (hidden 2688, 32/2 heads at hd 128, QK-norm, yarn rope θ=1e4
//! factor 128, NVFP4 W4A16 MLPs + fc, bf16 attention planes, no lm_head -
//! the TARGET's head runs on top). vLLM's `qwen3_dflash.py` is the
//! reference: committed positions become per-layer K/V from one fused
//! context state per row, `hidden_norm(fc(concat_i aux_i))`, where aux_i
//! are the target's post-block residuals at `target_layer_ids`
//! [1,5,19,29,41,51]; a draft round embeds `[committed, k × mask(990)]`
//! rows and attends NON-CAUSALLY over [context ∥ block] - expressed on the
//! causal kernels by giving every block row the block-end position (the
//! muse splice).
//!
//! The fc plane loads SPLIT into 6 per-aux column bands so the fusion runs
//! as 6 NVFP4 GEMMs over the contiguous aux planes (no interleave copy):
//! fc(concat_i aux_i) = Σ_i fc_band_i(aux_i).

use std::path::Path;

use cudarc::driver::CudaSlice;

use crate::gpu::{DeviceTensor, GpuError, KvDtype, Nvf4Plane};
use crate::gpu_model::gpt_oss::GpuModelError;
use crate::gpu_model::qwen35::{prefill_mm_pre_any, prefill_quant};
use paddock_kernels::reference::ops::YarnRope;
use paddock_models::modelopt::nvfp4_view;
use paddock_models::nemotron::NemotronDflashConfig;
use paddock_models::safetensors::{ShardedSafetensors, StDtype};

use super::*;

/// Draft block cap: 1 committed row + up to `MAX_DRAFT` masks per round.
pub(crate) const MAX_DRAFT: usize = 15;

pub(crate) struct DfLayer {
    pub in_norm: CudaSlice<f32>,
    pub post_norm: CudaSlice<f32>,
    pub q_norm: CudaSlice<f32>,
    pub k_norm: CudaSlice<f32>,
    pub wq: DeviceTensor,
    pub wk: DeviceTensor,
    pub wv: DeviceTensor,
    pub wo: DeviceTensor,
    pub gate: Nvf4Plane,
    pub up: Nvf4Plane,
    pub down: Nvf4Plane,
}

pub(crate) struct DflashDrafter {
    pub n_layers: usize,
    pub mask_token: u32,
    pub target_layers: Vec<usize>,
    pub inter: usize,
    /// yarn kernel params (neox convention - qwen3)
    pub rope: (f32, f32, f32, f32, f32, f32),
    pub eps: f32,
    /// fc split into one [hidden -> hidden] NVFP4 band per aux stream
    pub fc_bands: Vec<Nvf4Plane>,
    pub hidden_norm: CudaSlice<f32>,
    pub final_norm: CudaSlice<f32>,
    /// the TRAINED mask embedding - the drafter's embed_tokens row
    /// `mask_token` (byte-probed: the table is identical to the
    /// target's EXCEPT this row; reusing the target's row 990 collapsed
    /// acceptance from ~2.3 to ~1.3 at k=7)
    pub mask_embd: CudaSlice<f32>,
    pub layers: Vec<DfLayer>,
    pub state: Option<DflashState>,
}

/// Serving-time drafter state (built at enable_batch when attached).
pub(crate) struct DflashState {
    /// per-drafter-layer dense context KV [n_slots, max_ctx, kv_dim] f16
    pub kv_k: Vec<CudaSlice<u8>>,
    pub kv_v: Vec<CudaSlice<u8>>,
    /// per-slot contiguous feature coverage [start, end)
    pub feat: Vec<(u32, u32)>,
    /// aux bands, one [band_rows, embd] plane per target layer (band_rows
    /// == the batch scratch row capacity - walk rows index straight in)
    pub aux: Vec<CudaSlice<f32>>,
    /// fused context rows [band, embd]: raw fc sum + the normed rows
    pub d_ctx: CudaSlice<f32>,
    pub d_ctxn: CudaSlice<f32>,
    pub d_acc: CudaSlice<f32>,
    /// per-append K/V projections [band, kv_dim]
    pub d_kp: CudaSlice<f32>,
    pub d_vp: CudaSlice<f32>,
    // draft-round planes at MAX_DRAFT+1 rows per live slot (single-req
    // rounds for now; multi-req rounds loop)
    pub d_tok: CudaSlice<u32>,
    pub d_pos: CudaSlice<u32>,
    pub d_apos: CudaSlice<u32>,
    pub d_slots: CudaSlice<u32>,
    pub d_x: CudaSlice<f32>,
    pub d_xn: CudaSlice<f32>,
    pub d_q: CudaSlice<f32>,
    pub d_qn: CudaSlice<f32>,
    pub d_k: CudaSlice<f32>,
    pub d_kn: CudaSlice<f32>,
    pub d_v: CudaSlice<f32>,
    pub d_attn: CudaSlice<f32>,
    pub d_proj: CudaSlice<f32>,
    pub d_g: CudaSlice<f32>,
    pub d_u: CudaSlice<f32>,
    pub d_sinks: CudaSlice<f32>,
    pub d_logits: CudaSlice<f32>,
    pub d_picks: CudaSlice<u32>,
}

fn bf16_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| f32::from_bits((u16::from_le_bytes(*c) as u32) << 16))
        .collect()
}

impl GpuNemotron {
    /// Attach the official DFlash drafter (safetensors dir). Validates the
    /// geometry against the target and splits the fc plane into per-aux
    /// NVFP4 bands. `state` builds at enable_batch.
    pub fn attach_dflash(&mut self, path: &Path) -> Result<(), GpuModelError> {
        let dir = if path.is_dir() {
            path.to_path_buf()
        } else {
            path.parent()
                .ok_or_else(|| {
                    GpuModelError::Unsupported(format!(
                        "dflash path has no parent directory: {}",
                        path.display()
                    ))
                })?
                .to_path_buf()
        };
        let cfg = NemotronDflashConfig::read(&dir)
            .map_err(|e| GpuModelError::Unsupported(format!("dflash config: {e}")))?;
        let hp = &self.hp;
        if cfg.hidden != hp.hidden
            || cfg.n_heads != hp.n_heads
            || cfg.n_kv_heads != hp.n_kv_heads
            || cfg.head_dim != hp.head_dim
        {
            return Err(GpuModelError::Unsupported(
                "dflash geometry disagrees with the target".into(),
            ));
        }
        let (n_layers, inter, eps, mask_token) = (cfg.n_layers, cfg.inter, cfg.eps, cfg.mask_token);
        let target_layers = cfg.target_layers.clone();
        if target_layers.iter().any(|&l| l >= hp.n_layer)
            || !target_layers.windows(2).all(|w| w[0] < w[1])
        {
            return Err(GpuModelError::Unsupported(format!(
                "dflash target_layer_ids {target_layers:?} do not index the target"
            )));
        }
        let rope = YarnRope::new(
            hp.head_dim,
            cfg.rope_theta,
            1.0 / cfg.rope_factor,
            cfg.rope_orig,
            1.0,
            1.0,
            32.0,
            1.0,
        )
        .kernel_params();

        let st = ShardedSafetensors::open_dir(&dir)
            .map_err(|e| GpuModelError::Unsupported(format!("dflash shards: {e}")))?;
        let exec = self.exec.clone();

        let f32t = |name: &str, want: usize| -> Result<CudaSlice<f32>, GpuModelError> {
            let (t, bytes) = st
                .bytes(name)
                .ok_or_else(|| GpuModelError::Unsupported(format!("{name}: missing")))?;
            if t.dtype != StDtype::Bf16 {
                return Err(GpuModelError::Unsupported(format!(
                    "{name}: expected bf16, got {:?}",
                    t.dtype
                )));
            }
            let v = bf16_to_f32(bytes);
            if v.len() != want {
                return Err(GpuModelError::Unsupported(format!(
                    "{name}: {} elems, expected {want}",
                    v.len()
                )));
            }
            exec.to_device(&v).map_err(GpuModelError::from)
        };
        let plane = |name: &str, out: usize, inn: usize| -> Result<DeviceTensor, GpuModelError> {
            let buf = f32t(name, out * inn)?;
            Ok(DeviceTensor {
                buf,
                dims: vec![inn, out],
            })
        };
        let nvf4 = |name: &str, out: usize, inn: usize| -> Result<Nvf4Plane, GpuModelError> {
            let v = nvfp4_view(&st, name)
                .map_err(|e| GpuModelError::Unsupported(format!("{name}: {e}")))?;
            if (v.n, v.k) != (out, inn) {
                return Err(GpuModelError::Unsupported(format!(
                    "{name} is [{}, {}], expected [{out}, {inn}]",
                    v.n, v.k
                )));
            }
            exec.nvf4_upload(v.packed, v.scales, v.scale2, v.n, v.k)
                .map_err(GpuModelError::from)
        };

        // fc [hidden, n_aux*hidden] NVFP4 -> per-aux bands (packed bytes and
        // scales split per row on host; scale2 shared)
        let n_aux = target_layers.len();
        let fcv =
            nvfp4_view(&st, "fc").map_err(|e| GpuModelError::Unsupported(format!("fc: {e}")))?;
        if (fcv.n, fcv.k) != (hp.hidden, n_aux * hp.hidden) {
            return Err(GpuModelError::Unsupported(format!(
                "fc is [{}, {}], expected [{}, {}]",
                fcv.n,
                fcv.k,
                hp.hidden,
                n_aux * hp.hidden
            )));
        }
        let (pb, sb) = (hp.hidden / 2, hp.hidden / 16); // packed/scale bytes per band-row
        let (pk, sk) = (fcv.k / 2, fcv.k / 16);
        let mut fc_bands = Vec::with_capacity(n_aux);
        for ai in 0..n_aux {
            let mut p = Vec::with_capacity(fcv.n * pb);
            let mut s = Vec::with_capacity(fcv.n * sb);
            for row in 0..fcv.n {
                p.extend_from_slice(&fcv.packed[row * pk + ai * pb..row * pk + (ai + 1) * pb]);
                s.extend_from_slice(&fcv.scales[row * sk + ai * sb..row * sk + (ai + 1) * sb]);
            }
            fc_bands.push(exec.nvf4_upload(&p, &s, fcv.scale2, fcv.n, hp.hidden)?);
        }

        let q_dim = hp.n_heads * hp.head_dim;
        let kv_dim = hp.n_kv_heads * hp.head_dim;
        let mut layers = Vec::with_capacity(n_layers);
        for l in 0..n_layers {
            let p = format!("layers.{l}");
            layers.push(DfLayer {
                in_norm: f32t(&format!("{p}.input_layernorm.weight"), hp.hidden)?,
                post_norm: f32t(&format!("{p}.post_attention_layernorm.weight"), hp.hidden)?,
                q_norm: f32t(&format!("{p}.self_attn.q_norm.weight"), hp.head_dim)?,
                k_norm: f32t(&format!("{p}.self_attn.k_norm.weight"), hp.head_dim)?,
                wq: plane(&format!("{p}.self_attn.q_proj.weight"), q_dim, hp.hidden)?,
                wk: plane(&format!("{p}.self_attn.k_proj.weight"), kv_dim, hp.hidden)?,
                wv: plane(&format!("{p}.self_attn.v_proj.weight"), kv_dim, hp.hidden)?,
                wo: plane(&format!("{p}.self_attn.o_proj.weight"), hp.hidden, q_dim)?,
                gate: nvf4(&format!("{p}.mlp.gate_proj"), inter, hp.hidden)?,
                up: nvf4(&format!("{p}.mlp.up_proj"), inter, hp.hidden)?,
                down: nvf4(&format!("{p}.mlp.down_proj"), hp.hidden, inter)?,
            });
        }
        // embed_tokens is untrained and identical to the target's table
        // EXCEPT the mask row (byte-probed) - the rounds embed via the
        // target's arms and then overwrite the mask rows with this vector
        let mask_embd = {
            let (t, bytes) = st
                .bytes("embed_tokens.weight")
                .ok_or_else(|| GpuModelError::Unsupported("embed_tokens.weight: missing".into()))?;
            if t.dtype != StDtype::Bf16 {
                return Err(GpuModelError::Unsupported(format!(
                    "embed_tokens.weight: expected bf16, got {:?}",
                    t.dtype
                )));
            }
            let row = mask_token as usize * hp.hidden * 2;
            let v = bf16_to_f32(&bytes[row..row + hp.hidden * 2]);
            exec.to_device(&v)?
        };
        self.dflash = Some(DflashDrafter {
            n_layers,
            mask_token,
            target_layers,
            inter,
            rope,
            eps,
            fc_bands,
            hidden_norm: f32t("hidden_norm.weight", hp.hidden)?,
            final_norm: f32t("norm.weight", hp.hidden)?,
            mask_embd,
            layers,
            state: None,
        });
        tracing::info!(
            layers = n_layers,
            aux = n_aux,
            mask = mask_token,
            "nemotron DFlash drafter attached"
        );
        // an explicit sideload wins the drafter seat - drop the in-file
        // nextn block so its weights and walk hooks don't ride for free
        if self.mtp.take().is_some() {
            tracing::info!("nemotron: in-file MTP block released (DFlash attached)");
        }
        Ok(())
    }

    /// Build (or rebuild) the drafter's serving state - called from
    /// enable_batch so the aux taps are live from the first walk.
    pub(crate) fn dflash_ensure_state(&mut self) -> Result<(), GpuModelError> {
        let Some(df) = self.dflash.as_mut() else {
            return Ok(());
        };
        let (n_slots, band) = {
            let bs = self.batch.as_ref().expect("batch enabled");
            (bs.n_slots, bs.cap)
        };
        let hp = &self.hp;
        let kv_dim = hp.n_kv_heads * hp.head_dim;
        let q_dim = hp.n_heads * hp.head_dim;
        let e = &self.exec;
        let n_aux = df.target_layers.len();
        let rows = MAX_DRAFT + 1;
        let kv_bytes = n_slots * self.max_ctx * kv_dim * 2;
        let mut kv_k = Vec::with_capacity(df.n_layers);
        let mut kv_v = Vec::with_capacity(df.n_layers);
        for _ in 0..df.n_layers {
            kv_k.push(e.alloc_u8(kv_bytes)?);
            kv_v.push(e.alloc_u8(kv_bytes)?);
        }
        df.state = Some(DflashState {
            kv_k,
            kv_v,
            feat: vec![(0, 0); n_slots],
            aux: (0..n_aux)
                .map(|_| e.alloc(band * hp.hidden))
                .collect::<Result<Vec<_>, _>>()?,
            d_ctx: e.alloc(band * hp.hidden)?,
            d_ctxn: e.alloc(band * hp.hidden)?,
            d_acc: e.alloc(band * hp.hidden)?,
            d_kp: e.alloc(band * kv_dim)?,
            d_vp: e.alloc(band * kv_dim)?,
            d_tok: e.alloc_u32(rows)?,
            d_pos: e.alloc_u32(rows)?,
            d_apos: e.alloc_u32(rows)?,
            d_slots: e.alloc_u32(rows)?,
            d_x: e.alloc(rows * hp.hidden)?,
            d_xn: e.alloc(rows * hp.hidden)?,
            d_q: e.alloc(rows * q_dim)?,
            d_qn: e.alloc(rows * q_dim)?,
            d_k: e.alloc(rows * kv_dim)?,
            d_kn: e.alloc(rows * kv_dim)?,
            d_v: e.alloc(rows * kv_dim)?,
            d_attn: e.alloc(rows * q_dim)?,
            d_proj: e.alloc(rows * hp.hidden)?,
            d_g: e.alloc(rows * df.inter)?,
            d_u: e.alloc(rows * df.inter)?,
            d_sinks: e.alloc_no_sinks(hp.n_heads)?,
            d_logits: e.alloc(rows * hp.vocab)?,
            d_picks: e.alloc_u32(rows)?,
        });
        Ok(())
    }

    /// Clear one slot's feature coverage (fresh sequence / release).
    pub(crate) fn dflash_clear_slot(&mut self, slot: usize) {
        if let Some(st) = self.dflash.as_mut().and_then(|d| d.state.as_mut())
            && slot < st.feat.len()
        {
            st.feat[slot] = (0, 0);
        }
    }

    /// Trim a slot's coverage to `keep` rows (prefix restore - gemma4's
    /// trim-not-clear: the ring rows below the resume point still describe
    /// the same tokens).
    pub(crate) fn dflash_trim_slot(&mut self, slot: usize, keep: usize) {
        if let Some(st) = self.dflash.as_mut().and_then(|d| d.state.as_mut())
            && slot < st.feat.len()
        {
            let (s, e) = st.feat[slot];
            let keep = keep as u32;
            if s == 0 && e > keep {
                st.feat[slot] = (0, keep);
            } else if s > 0 {
                // non-zero start never survives a restore
                st.feat[slot] = (0, 0);
            }
        }
    }

    /// Coverage-warm: features cover exactly [0, pos).
    pub(crate) fn dflash_warm(&self, slot: usize, pos: usize) -> bool {
        self.dflash
            .as_ref()
            .and_then(|d| d.state.as_ref())
            .is_some_and(|st| st.feat[slot] == (0, pos as u32))
    }

    /// Fuse the tapped aux rows into context states and append their K/V to
    /// every drafter layer's cache. `rows` are the batch walk's rows
    /// (positions/slots still live in the batch scratch's d_pos/d_slots);
    /// `runs` gives contiguous same-slot spans for the coverage bookkeeping.
    pub(crate) fn dflash_append_features(&mut self, r: usize) -> Result<(), GpuModelError> {
        let hp = self.hp.clone();
        let exec = self.exec.clone();
        let max_ctx = self.max_ctx;
        let kv_dim = hp.n_kv_heads * hp.head_dim;
        let Some(df) = self.dflash.as_mut() else {
            return Ok(());
        };
        let Some(st) = df.state.as_mut() else {
            return Ok(());
        };
        let bs = self.batch.as_ref().expect("batch enabled");
        let sc = &bs.sc;

        // fused context state: Σ_ai fc_band_ai(aux_ai), then hidden_norm
        for (ai, band) in df.fc_bands.iter().enumerate() {
            if ai == 0 {
                exec.nvf4_gemv_batch(band, &st.aux[0], &mut st.d_ctx, None, r)?;
            } else {
                exec.nvf4_gemv_batch(band, &st.aux[ai], &mut st.d_acc, None, r)?;
                exec.add(&mut st.d_ctx, &st.d_acc, r * hp.hidden)?;
            }
        }
        exec.rmsnorm_batch(
            &st.d_ctx,
            &df.hidden_norm,
            &mut st.d_ctxn,
            hp.hidden,
            df.eps,
            r,
        )?;

        for l in 0..df.n_layers {
            let ly = &df.layers[l];
            exec.gemm_f32(&ly.wk.buf, hp.hidden, kv_dim, &st.d_ctxn, &mut st.d_kp, r)?;
            exec.gemm_f32(&ly.wv.buf, hp.hidden, kv_dim, &st.d_ctxn, &mut st.d_vp, r)?;
            exec.rmsnorm_batch(
                &st.d_kp,
                &ly.k_norm,
                &mut st.d_acc,
                hp.head_dim,
                df.eps,
                r * hp.n_kv_heads,
            )?;
            exec.rope_yarn_batch(
                &mut st.d_acc,
                &sc.d_pos,
                hp.n_kv_heads,
                hp.head_dim,
                df.rope,
                r,
            )?;
            exec.kv_append_batch(
                &st.d_acc,
                &mut st.kv_k[l],
                &sc.d_pos,
                Some(&sc.d_slots),
                kv_dim,
                max_ctx,
                r,
                KvDtype::Fp16,
            )?;
            exec.kv_append_batch(
                &st.d_vp,
                &mut st.kv_v[l],
                &sc.d_pos,
                Some(&sc.d_slots),
                kv_dim,
                max_ctx,
                r,
                KvDtype::Fp16,
            )?;
        }
        Ok(())
    }

    /// Host-side coverage bookkeeping after an append: rows [start, end)
    /// for `slot` now carry features. Coverage only extends contiguously.
    pub(crate) fn dflash_note_rows(&mut self, slot: usize, start: usize, end: usize) {
        if let Some(st) = self.dflash.as_mut().and_then(|d| d.state.as_mut())
            && slot < st.feat.len()
        {
            let (s, e) = st.feat[slot];
            if (s == 0 && start <= e as usize && end as u32 > e) || start == 0 {
                st.feat[slot] = (0, end as u32);
            }
        }
    }

    /// Draft one slot's block: rows = [committed, k × mask] at positions
    /// pos..pos+k; every row attends [0, pos+k] (the non-causal splice).
    /// Returns the k drafts (rows 1..).
    pub(crate) fn dflash_draft(
        &mut self,
        slot: usize,
        pos: usize,
        committed: u32,
        k: usize,
    ) -> Result<Vec<u32>, GpuModelError> {
        assert!((1..=MAX_DRAFT).contains(&k));
        let hp = self.hp.clone();
        let exec = self.exec.clone();
        let max_ctx = self.max_ctx;
        let embd = hp.hidden;
        let kv_dim = hp.n_kv_heads * hp.head_dim;
        let q_dim = hp.n_heads * hp.head_dim;
        let scale = 1.0 / (hp.head_dim as f32).sqrt();
        let rows = k + 1;
        let drv = |e: cudarc::driver::DriverError| crate::gpu::from_driver(e);

        // stage row streams
        {
            let df = self.dflash.as_mut().expect("dflash");
            let mask = df.mask_token;
            let st = df.state.as_mut().expect("dflash state");
            let toks: Vec<u32> = std::iter::once(committed)
                .chain(std::iter::repeat_n(mask, k))
                .collect();
            let positions: Vec<u32> = (pos as u32..(pos + rows) as u32).collect();
            // the splice: every row's ATTENTION position is the block end
            let apos: Vec<u32> = vec![(pos + rows - 1) as u32; rows];
            let slots: Vec<u32> = vec![slot as u32; rows];
            let stm = &self.exec.stream;
            let mut t = st
                .d_tok
                .try_slice_mut(0..rows)
                .ok_or_else(|| GpuError::Driver("tok".into()))?;
            stm.memcpy_htod(&toks, &mut t).map_err(drv)?;
            let mut p = st
                .d_pos
                .try_slice_mut(0..rows)
                .ok_or_else(|| GpuError::Driver("pos".into()))?;
            stm.memcpy_htod(&positions, &mut p).map_err(drv)?;
            let mut ap = st
                .d_apos
                .try_slice_mut(0..rows)
                .ok_or_else(|| GpuError::Driver("apos".into()))?;
            stm.memcpy_htod(&apos, &mut ap).map_err(drv)?;
            let mut s = st
                .d_slots
                .try_slice_mut(0..rows)
                .ok_or_else(|| GpuError::Driver("slots".into()))?;
            stm.memcpy_htod(&slots, &mut s).map_err(drv)?;
        }

        // embed via the target's table (the drafter's embed is untrained)
        {
            let df = self.dflash.as_mut().expect("dflash");
            let st = df.state.as_mut().expect("dflash state");
            match &self.tok_embd {
                TokEmbd::F32(tab) => {
                    exec.embed_gather_batch(tab, &st.d_tok, &mut st.d_x, embd, rows)?
                }
                TokEmbd::Bf16(tab) => {
                    exec.embed_gather_bf16(tab, &st.d_tok, &mut st.d_x, embd, rows, 1.0)?
                }
                TokEmbd::Q8(tab) => {
                    exec.embed_gather_batch_q8(tab, &st.d_tok, &mut st.d_x, embd, rows)?
                }
            }
            // the mask rows take the drafter's TRAINED mask embedding
            for i in 1..rows {
                exec.copy_region(&df.mask_embd, 0, &mut st.d_x, i * embd, embd)?;
            }
            for l in 0..df.n_layers {
                let ly = &df.layers[l];
                exec.rmsnorm_batch(&st.d_x, &ly.in_norm, &mut st.d_xn, embd, df.eps, rows)?;
                exec.gemm_f32(&ly.wq.buf, embd, q_dim, &st.d_xn, &mut st.d_q, rows)?;
                exec.gemm_f32(&ly.wk.buf, embd, kv_dim, &st.d_xn, &mut st.d_k, rows)?;
                exec.gemm_f32(&ly.wv.buf, embd, kv_dim, &st.d_xn, &mut st.d_v, rows)?;
                exec.rmsnorm_batch(
                    &st.d_q,
                    &ly.q_norm,
                    &mut st.d_qn,
                    hp.head_dim,
                    df.eps,
                    rows * hp.n_heads,
                )?;
                exec.rmsnorm_batch(
                    &st.d_k,
                    &ly.k_norm,
                    &mut st.d_kn,
                    hp.head_dim,
                    df.eps,
                    rows * hp.n_kv_heads,
                )?;
                exec.rope_yarn_batch(
                    &mut st.d_qn,
                    &st.d_pos,
                    hp.n_heads,
                    hp.head_dim,
                    df.rope,
                    rows,
                )?;
                exec.rope_yarn_batch(
                    &mut st.d_kn,
                    &st.d_pos,
                    hp.n_kv_heads,
                    hp.head_dim,
                    df.rope,
                    rows,
                )?;
                // block K/V land in the context cache at their true
                // positions (overwritten by real context rows on commit)
                exec.kv_append_batch(
                    &st.d_kn,
                    &mut st.kv_k[l],
                    &st.d_pos,
                    Some(&st.d_slots),
                    kv_dim,
                    max_ctx,
                    rows,
                    KvDtype::Fp16,
                )?;
                exec.kv_append_batch(
                    &st.d_v,
                    &mut st.kv_v[l],
                    &st.d_pos,
                    Some(&st.d_slots),
                    kv_dim,
                    max_ctx,
                    rows,
                    KvDtype::Fp16,
                )?;
                exec.attn_decode_batch(
                    &st.d_qn,
                    &st.kv_k[l],
                    &st.kv_v[l],
                    &st.d_sinks,
                    &mut st.d_attn,
                    &st.d_apos,
                    Some(&st.d_slots),
                    hp.n_heads,
                    hp.n_kv_heads,
                    hp.head_dim,
                    max_ctx,
                    kv_dim,
                    0,
                    rows,
                    scale,
                    KvDtype::Fp16,
                )?;
                exec.gemm_f32(&ly.wo.buf, q_dim, embd, &st.d_attn, &mut st.d_proj, rows)?;
                exec.add(&mut st.d_x, &st.d_proj, rows * embd)?;
                exec.rmsnorm_batch(&st.d_x, &ly.post_norm, &mut st.d_xn, embd, df.eps, rows)?;
                exec.nvf4_gemv_batch(&ly.gate, &st.d_xn, &mut st.d_g, None, rows)?;
                exec.nvf4_gemv_batch(&ly.up, &st.d_xn, &mut st.d_u, None, rows)?;
                exec.swiglu(&mut st.d_g, &st.d_u, rows * df.inter)?;
                exec.nvf4_gemv_batch(&ly.down, &st.d_g, &mut st.d_proj, None, rows)?;
                exec.add(&mut st.d_x, &st.d_proj, rows * embd)?;
            }
            exec.rmsnorm_batch(&st.d_x, &df.final_norm, &mut st.d_xn, embd, df.eps, rows)?;
        }

        // the TARGET's head over the drafter rows
        {
            let df = self.dflash.as_mut().expect("dflash");
            let st = df.state.as_mut().expect("dflash state");
            match &self.lm_head {
                HeadW::Nvf4(h) => {
                    exec.nvf4_gemv_batch(h, &st.d_xn, &mut st.d_logits, None, rows)?
                }
                HeadW::Qw(q) => {
                    let bs = self.batch.as_mut().expect("batch enabled");
                    let s8 = bs.sc.q8.as_mut().expect("q8 batch scratch");
                    prefill_quant(
                        &exec, &mut s8.xq, &mut s8.xs, &mut s8.yq, &st.d_xn, embd, rows,
                    )?;
                    prefill_mm_pre_any(
                        &exec,
                        q,
                        &s8.xq,
                        &s8.xs,
                        &s8.yq,
                        &mut s8.xsums,
                        &mut s8.ssums,
                        &mut s8.skfix,
                        &mut st.d_logits,
                        rows,
                    )?;
                }
            }
            exec.argmax_rows(&st.d_logits, &mut st.d_picks, rows, hp.vocab)?;
            // true block diffusion: mask@p+i predicts its own position's
            // token, so drafts are the mask rows' picks. (Measured: with the
            // wrong mask embedding the rows degenerate into next-token
            // predictors - the earlier shifted read was chasing that bug.)
            let view = st
                .d_picks
                .try_slice(1..rows)
                .ok_or_else(|| GpuError::Driver("picks view".into()))?;
            let drafts: Vec<u32> = self
                .exec
                .stream
                .clone_dtoh(&view)
                .map_err(|e| GpuError::Driver(e.to_string()))?;
            Ok(drafts)
        }
    }
}
