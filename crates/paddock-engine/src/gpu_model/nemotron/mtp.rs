//! In-file MTP drafter for nemotron (C3) - the GGUF's blk.52
//! nextn block, DeepSeek-style glue on the trunk: one combined transformer
//! block `x0 = eh_proj(cat[enorm(embed(tok)), hnorm(h)])`, `x1 = x0 +
//! attn(attn_norm(x0))` (NoPE, own dense KV), `x2 = x1 +
//! moe(post_attention_norm(x1))` (the trunk MoE recipe), `h_out =
//! shared_head_norm(x2)` into the TRUNK lm_head. vLLM's `nemotron_h_mtp.py`
//! is the vendor reference (NVIDIA's own contributed code): the end norm is
//! RMSNorm - llama.cpp builds LLM_NORM but never serves the tensors, so the
//! recon's flagged discrepancy resolves to the vendor choice.
//!
//! Pairing convention (the qwen35 lane's, certified there against llama.cpp
//! b9895): the MTP consumes pair `(token at pos i, target h at pos i-1)` at
//! its own KV position i, `h` = the trunk's POST-final-norm hidden, with
//! h_{-1} = zeros for row 0 of a fresh sequence. The chain state per slot is
//! one vector (`pending_h` = h at the last covered position); a prefix-
//! restore trim below the covered end zeroes it - the one resume row then
//! pairs with zeros (defined, reproducible, same class as row 0) instead of
//! a stale vector.
//!
//! Serving shape follows DFlash, not qwen35's spec_batch machinery: the
//! block advances on every batched walk (prefill chunks, decode ticks,
//! verify rounds) using the walk's own staged rows, coverage extends
//! contiguously per slot, and drafting chains k sequential r=1 passes
//! feeding the head's own post-shared_head_norm output back as the next h
//! (qwen35's draft()). The pass reuses the batch scratch wholesale but keeps
//! its own residual plane (`d_mx`) - callers consume `sc.d_x` through the
//! head after rows_pass_body returns, so the trunk residuals must survive.
//!
//! Decode ticks are not graph-captured for the MTP: the h chain needs
//! per-tick host staging (gather pending_h by the tick's actual slots), so
//! the append runs as plain launches after `step_replay` - and pipe_b is
//! disabled while the MTP is the active drafter (spec rounds are the decode
//! path when spec is on; `--no-spec` serves never load the block at all).

use cudarc::driver::CudaSlice;

use crate::gpu::{DeviceTensor, GpuError, KvDtype, QuantW};
use crate::gpu_model::gpt_oss::GpuModelError;
use crate::gpu_model::qwen35::{gemv_any, prefill_mm_pre_any, prefill_quant};

use super::*;

/// Draft chain cap per round (mirrors DFlash's block cap).
pub(crate) const MTP_MAX_DRAFT: usize = 15;

/// The blk.52 nextn weights, Q8_0 residency (the GGUF lane's classes: qw
/// planes decode on the repacked GEMV / prefill on the mmq ladder, experts
/// on the repacked relu2 pair - byte-identical dispatch to a trunk layer).
pub(crate) struct MtpWeights {
    pub enorm: DeviceTensor,
    pub hnorm: DeviceTensor,
    /// [2*hidden -> hidden], e-first concat (vLLM order)
    pub eh_proj: QuantW,
    pub attn_norm: DeviceTensor,
    pub attn: AttnWeights,
    pub post_norm: DeviceTensor,
    pub moe: MoeWeights,
    /// shared_head_norm - RMSNorm per the vendor reference
    pub head_norm: DeviceTensor,
}

pub(crate) struct MtpDrafter {
    pub w: MtpWeights,
    pub state: Option<MtpState>,
}

/// Serving-time state (built at enable_batch when the MTP is active).
pub(crate) struct MtpState {
    /// dense per-slot KV for the single attention sub-layer,
    /// [n_slots, max_ctx, kv_dim] f16 (DFlash's cache shape)
    pub kv_k: CudaSlice<u8>,
    pub kv_v: CudaSlice<u8>,
    /// per-slot contiguous coverage [start, end) - same discipline as
    /// DFlash's feat
    pub feat: Vec<(u32, u32)>,
    /// per-slot h chain: the trunk h at the last covered position,
    /// [n_slots, embd] (zeros = fresh / post-trim)
    pub pending_h: CudaSlice<f32>,
    /// walk h tap: rmsnorm(sc.d_x, final_norm) for all pass rows [band, embd]
    pub d_h: CudaSlice<f32>,
    /// shifted h inputs (row 0 of a run = pending_h) [band, embd]
    pub d_hin: CudaSlice<f32>,
    /// the MTP block's own residual stream [band, embd] - sc.d_x stays the
    /// trunk's (callers head it after the hook)
    pub d_mx: CudaSlice<f32>,
    /// draft-loop head-normed row (the chained h) [embd]
    pub d_hd: CudaSlice<f32>,
    pub d_pick: CudaSlice<u32>,
    /// stays zero - pending_h resets copy from here
    pub d_zero: CudaSlice<f32>,
}

impl GpuNemotron {
    /// The in-file MTP is the drafter iff no DFlash sideload took the seat
    /// (an explicit `--mtp` attach wins over the in-file block).
    pub(crate) fn mtp_active(&self) -> bool {
        self.mtp.is_some() && self.dflash.is_none()
    }

    /// Build the serving state - called from enable_batch so the walk hooks
    /// are live from the first pass.
    pub(crate) fn mtp_ensure_state(&mut self) -> Result<(), GpuModelError> {
        if !self.mtp_active() {
            return Ok(());
        }
        let (n_slots, band) = {
            let bs = self.batch.as_ref().expect("batch enabled");
            (bs.n_slots, bs.cap)
        };
        let hp = &self.hp;
        let kv_dim = hp.n_kv_heads * hp.head_dim;
        let e = self.exec.clone();
        let mtp = self.mtp.as_mut().expect("mtp weights");
        if mtp.state.is_some() {
            return Ok(());
        }
        let kv_bytes = n_slots * self.max_ctx * kv_dim * 2;
        mtp.state = Some(MtpState {
            kv_k: e.alloc_u8(kv_bytes)?,
            kv_v: e.alloc_u8(kv_bytes)?,
            feat: vec![(0, 0); n_slots],
            pending_h: e.alloc(n_slots * hp.hidden)?, // alloc = zeroed
            d_h: e.alloc(band * hp.hidden)?,
            d_hin: e.alloc(band * hp.hidden)?,
            d_mx: e.alloc(band * hp.hidden)?,
            d_hd: e.alloc(hp.hidden)?,
            d_pick: e.alloc_u32(1)?,
            d_zero: e.alloc(hp.hidden)?,
        });
        tracing::info!(
            kv_gib = (2 * kv_bytes) as f64 / (1u64 << 30) as f64,
            "nemotron MTP drafter state up"
        );
        Ok(())
    }

    /// Fresh sequence / slot release: coverage gone, chain zeroed.
    pub(crate) fn mtp_clear_slot(&mut self, slot: usize) -> Result<(), GpuModelError> {
        let embd = self.hp.hidden;
        let exec = self.exec.clone();
        if let Some(st) = self.mtp.as_mut().and_then(|m| m.state.as_mut())
            && slot < st.feat.len()
        {
            st.feat[slot] = (0, 0);
            exec.copy_region(&st.d_zero, 0, &mut st.pending_h, slot * embd, embd)?;
        }
        Ok(())
    }

    /// Prefix-restore trim: KV cells below `keep` still describe the same
    /// tokens (coverage survives), but pending_h held h at the old end - the
    /// chain can't be rebuilt from one vector, so it resets to zeros and the
    /// single resume row pairs off-distribution once (drafter-quality-only;
    /// the verify re-judges everything).
    pub(crate) fn mtp_trim_slot(&mut self, slot: usize, keep: usize) -> Result<(), GpuModelError> {
        let embd = self.hp.hidden;
        let exec = self.exec.clone();
        if let Some(st) = self.mtp.as_mut().and_then(|m| m.state.as_mut())
            && slot < st.feat.len()
        {
            let (s, e) = st.feat[slot];
            if s == 0 && e > keep as u32 {
                st.feat[slot] = (0, keep as u32);
                exec.copy_region(&st.d_zero, 0, &mut st.pending_h, slot * embd, embd)?;
            } else if s > 0 {
                st.feat[slot] = (0, 0);
                exec.copy_region(&st.d_zero, 0, &mut st.pending_h, slot * embd, embd)?;
            }
        }
        Ok(())
    }

    /// Coverage-warm: the block consumed exactly [0, pos).
    pub(crate) fn mtp_warm(&self, slot: usize, pos: usize) -> bool {
        self.mtp
            .as_ref()
            .and_then(|m| m.state.as_ref())
            .is_some_and(|st| st.feat[slot] == (0, pos as u32))
    }

    /// Advance one slot after an append: extend coverage to `end` and move
    /// pending_h to the h of pass row `h_row` - GUARDED on actually
    /// extending, so a re-walked stale row (hole-row class) can never
    /// regress the chain.
    pub(crate) fn mtp_advance(
        &mut self,
        slot: usize,
        start: usize,
        end: usize,
        h_row: usize,
    ) -> Result<(), GpuModelError> {
        let embd = self.hp.hidden;
        let exec = self.exec.clone();
        if let Some(st) = self.mtp.as_mut().and_then(|m| m.state.as_mut())
            && slot < st.feat.len()
        {
            let (s, e) = st.feat[slot];
            if s == 0 && start <= e as usize && end as u32 > e {
                st.feat[slot] = (0, end as u32);
                exec.copy_region(&st.d_h, h_row * embd, &mut st.pending_h, slot * embd, embd)?;
            }
        }
        Ok(())
    }

    /// The walk hook: run the MTP block over the pass's `runs` (contiguous
    /// same-slot spans as (slot, row offset, len), rows 0..r in order),
    /// consuming the walk's staged tokens/positions/slots and the trunk's
    /// final residuals still in sc.d_x. Appends the block's KV for every
    /// row; coverage/pending advance separately via `mtp_advance` so verify
    /// rounds move only through ACCEPTED rows (stale cells past the accept
    /// are overwritten before any draft reads them - the draft pass writes
    /// its own cells before attending).
    pub(crate) fn mtp_append_rows(
        &mut self,
        runs: &[(usize, usize, usize)],
    ) -> Result<(), GpuModelError> {
        let r: usize = runs.iter().map(|&(_, _, l)| l).sum();
        if r == 0 || self.mtp.as_ref().is_none_or(|m| m.state.is_none()) {
            return Ok(());
        }
        let exec = self.exec.clone();
        let (embd, eps) = (self.hp.hidden, self.hp.eps);
        let final_norm = self.final_norm.buf.clone();

        // h tap: the trunk's post-final-norm rows for this pass
        {
            let bs = self.batch.as_ref().expect("batch enabled");
            let st = self
                .mtp
                .as_mut()
                .expect("mtp weights")
                .state
                .as_mut()
                .expect("mtp state");
            exec.rmsnorm_batch(&bs.sc.d_x, &final_norm, &mut st.d_h, embd, eps, r)?;
            // shifted h inputs: row 0 of each run pairs with the slot's
            // pending_h, row i with the run's own h row i-1
            for &(slot, off, len) in runs {
                exec.copy_region(&st.pending_h, slot * embd, &mut st.d_hin, off * embd, embd)?;
                if len > 1 {
                    let n = (len - 1) * embd;
                    exec.copy_region(&st.d_h, off * embd, &mut st.d_hin, (off + 1) * embd, n)?;
                }
            }
        }
        self.mtp_block_rows(r, false)
    }

    /// Decode-tick hook (graph-replayed ticks - the MTP runs outside the
    /// capture, see the module doc): every row is one new token of its slot.
    pub(crate) fn mtp_append_ticks(
        &mut self,
        slots: &[u32],
        positions: &[u32],
    ) -> Result<(), GpuModelError> {
        if self.mtp.as_ref().is_none_or(|m| m.state.is_none()) {
            return Ok(());
        }
        let runs: Vec<(usize, usize, usize)> = slots
            .iter()
            .enumerate()
            .map(|(i, &s)| (s as usize, i, 1))
            .collect();
        self.mtp_append_rows(&runs)?;
        for (i, (&s, &p)) in slots.iter().zip(positions).enumerate() {
            self.mtp_advance(s as usize, p as usize, p as usize + 1, i)?;
        }
        Ok(())
    }

    /// One MTP block pass over rows 0..r: inputs are sc.d_tok/d_pos/d_slots
    /// (already staged) + st.d_hin. Reuses the batch scratch except the
    /// residual (st.d_mx). `head_out` additionally produces the
    /// post-shared_head_norm row 0 into st.d_hd (the draft chain).
    fn mtp_block_rows(&mut self, r: usize, head_out: bool) -> Result<(), GpuModelError> {
        let exec = self.exec.clone();
        let hp = self.hp.clone();
        let (embd, eps) = (hp.hidden, hp.eps);
        let q_dim = hp.n_heads * hp.head_dim;
        let kv_dim = hp.n_kv_heads * hp.head_dim;
        let scale = 1.0 / (hp.head_dim as f32).sqrt();
        let max_ctx = self.max_ctx;
        let dec1 = r == 1;
        let tok_embd = &self.tok_embd;
        let mtp = self.mtp.as_mut().expect("mtp weights");
        let m = &mtp.w;
        let st = mtp.state.as_mut().expect("mtp state");
        let bs = self.batch.as_mut().expect("batch enabled");
        let sc = &mut bs.sc;

        // x0 = eh_proj(cat[enorm(embed(tok)), hnorm(h)]) - e first. The
        // concat stages packed [r, 2*embd] rows in d_zxbcdt (in_proj_rows
        // per row is wider), per-row copies like qwen35's project.
        match tok_embd {
            TokEmbd::F32(tab) => exec.embed_gather_batch(tab, &sc.d_tok, &mut st.d_mx, embd, r)?,
            TokEmbd::Bf16(tab) => {
                exec.embed_gather_bf16(tab, &sc.d_tok, &mut st.d_mx, embd, r, 1.0)?
            }
            TokEmbd::Q8(tab) => {
                exec.embed_gather_batch_q8(tab, &sc.d_tok, &mut st.d_mx, embd, r)?
            }
        }
        exec.rmsnorm_batch(&st.d_mx, &m.enorm.buf, &mut sc.d_xn, embd, eps, r)?;
        exec.rmsnorm_batch(&st.d_hin, &m.hnorm.buf, &mut sc.d_proj, embd, eps, r)?;
        for i in 0..r {
            exec.copy_region(&sc.d_xn, i * embd, &mut sc.d_zxbcdt, i * 2 * embd, embd)?;
            exec.copy_region(
                &sc.d_proj,
                i * embd,
                &mut sc.d_zxbcdt,
                i * 2 * embd + embd,
                embd,
            )?;
        }
        if dec1 {
            gemv_any(&exec, &m.eh_proj, &sc.d_zxbcdt, &mut st.d_mx)?;
        } else {
            let s8 = sc.q8.as_mut().expect("q8 batch scratch");
            prefill_quant(
                &exec,
                &mut s8.xq,
                &mut s8.xs,
                &mut s8.yq,
                &sc.d_zxbcdt,
                2 * embd,
                r,
            )?;
            prefill_mm_pre_any(
                &exec,
                &m.eh_proj,
                &s8.xq,
                &s8.xs,
                &s8.yq,
                &mut s8.xsums,
                &mut s8.ssums,
                &mut s8.skfix,
                &mut st.d_mx,
                r,
            )?;
        }

        // x1 = x0 + attn(attn_norm(x0)) - NoPE, dense per-slot KV, per-row
        // causal window [0, pos_i] (the decode kernel is exactly that)
        exec.rmsnorm_batch(&st.d_mx, &m.attn_norm.buf, &mut sc.d_xn, embd, eps, r)?;
        let AttnWeights::Qw { wq, wk, wv, wo } = &m.attn else {
            unreachable!("in-file MTP is the GGUF lane (Qw planes)");
        };
        if dec1 {
            gemv_any(&exec, wq, &sc.d_xn, &mut sc.d_q)?;
            gemv_any(&exec, wk, &sc.d_xn, &mut sc.d_k)?;
            gemv_any(&exec, wv, &sc.d_xn, &mut sc.d_v)?;
        } else {
            let s8 = sc.q8.as_mut().expect("q8 batch scratch");
            prefill_quant(&exec, &mut s8.xq, &mut s8.xs, &mut s8.yq, &sc.d_xn, embd, r)?;
            prefill_mm_pre_any(
                &exec,
                wq,
                &s8.xq,
                &s8.xs,
                &s8.yq,
                &mut s8.xsums,
                &mut s8.ssums,
                &mut s8.skfix,
                &mut sc.d_q,
                r,
            )?;
            prefill_mm_pre_any(
                &exec,
                wk,
                &s8.xq,
                &s8.xs,
                &s8.yq,
                &mut s8.xsums,
                &mut s8.ssums,
                &mut s8.skfix,
                &mut sc.d_k,
                r,
            )?;
            prefill_mm_pre_any(
                &exec,
                wv,
                &s8.xq,
                &s8.xs,
                &s8.yq,
                &mut s8.xsums,
                &mut s8.ssums,
                &mut s8.skfix,
                &mut sc.d_v,
                r,
            )?;
        }
        exec.kv_append_batch(
            &sc.d_k,
            &mut st.kv_k,
            &sc.d_pos,
            Some(&sc.d_slots),
            kv_dim,
            max_ctx,
            r,
            KvDtype::Fp16,
        )?;
        exec.kv_append_batch(
            &sc.d_v,
            &mut st.kv_v,
            &sc.d_pos,
            Some(&sc.d_slots),
            kv_dim,
            max_ctx,
            r,
            KvDtype::Fp16,
        )?;
        exec.attn_decode_batch(
            &sc.d_q,
            &st.kv_k,
            &st.kv_v,
            &sc.d_sinks,
            &mut sc.d_attn,
            &sc.d_pos,
            Some(&sc.d_slots),
            hp.n_heads,
            hp.n_kv_heads,
            hp.head_dim,
            max_ctx,
            kv_dim,
            0,
            r,
            scale,
            KvDtype::Fp16,
        )?;
        if dec1 {
            gemv_any(&exec, wo, &sc.d_attn, &mut sc.d_proj)?;
        } else {
            let s8 = sc.q8.as_mut().expect("q8 batch scratch");
            prefill_quant(
                &exec, &mut s8.xq, &mut s8.xs, &mut s8.yq, &sc.d_attn, q_dim, r,
            )?;
            prefill_mm_pre_any(
                &exec,
                wo,
                &s8.xq,
                &s8.xs,
                &s8.yq,
                &mut s8.xsums,
                &mut s8.ssums,
                &mut s8.skfix,
                &mut sc.d_proj,
                r,
            )?;
        }
        exec.add(&mut st.d_mx, &sc.d_proj, r * embd)?;

        // x2 = x1 + moe(post_norm(x1)) - the trunk MoE recipe verbatim
        exec.rmsnorm_batch(&st.d_mx, &m.post_norm.buf, &mut sc.d_xn, embd, eps, r)?;
        exec.matvec_f32_batch(&m.moe.router, &sc.d_xn, &mut sc.d_logits_r, r)?;
        exec.moe_topk_sigmoid_batch(
            &sc.d_logits_r,
            &m.moe.bias.buf,
            hp.routed_scale,
            hp.n_expert,
            hp.n_active,
            &mut sc.d_idx,
            &mut sc.d_w,
            r,
        )?;
        let MoePlanes::Q8 {
            up,
            down,
            sh_up,
            sh_down,
        } = &m.moe.planes
        else {
            unreachable!("in-file MTP is the GGUF lane (Q8 experts)");
        };
        let s8 = sc.q8.as_mut().expect("q8 batch scratch");
        if dec1 {
            exec.quantize_q8(&sc.d_xn, &mut s8.xq, &mut s8.xs, embd)?;
            exec.q8_0_moe_up_relu2(up, &sc.d_idx, &s8.xq, &s8.xs, &mut s8.act_r, hp.n_active, 1)?;
            exec.quantize_q8(
                &s8.act_r,
                &mut s8.fq_r1,
                &mut s8.fs_r1,
                hp.n_active * hp.moe_ff,
            )?;
            exec.q8_0_moe_down(
                down,
                &sc.d_idx,
                &sc.d_w,
                &s8.fq_r1,
                &s8.fs_r1,
                &mut sc.d_proj,
                hp.n_active,
                1,
            )?;
            exec.q8_0_moe_up_relu2(sh_up, &sc.d_sh_idx, &s8.xq, &s8.xs, &mut s8.act_s, 1, 1)?;
            exec.quantize_q8(&s8.act_s, &mut s8.fq_s1, &mut s8.fs_s1, hp.shared_ff)?;
            exec.q8_0_moe_down(
                sh_down,
                &sc.d_sh_idx,
                &sc.d_sh_w,
                &s8.fq_s1,
                &s8.fs_s1,
                &mut s8.shproj,
                1,
                1,
            )?;
            exec.add(&mut sc.d_proj, &s8.shproj, embd)?;
            exec.add(&mut st.d_mx, &sc.d_proj, embd)?;
        } else {
            exec.quantize_q8(&sc.d_xn, &mut s8.xq, &mut s8.xs, r * embd)?;
            exec.moe_align(
                &sc.d_idx,
                &mut sc.d_srow,
                &mut sc.d_sslot,
                &mut sc.d_bexp,
                r,
                hp.n_active,
                hp.n_expert,
                sc.nb_r,
            )?;
            exec.q8_0_moe_up_relu2_sorted(
                up,
                &sc.d_srow,
                &sc.d_bexp,
                &s8.xq,
                &s8.xs,
                &mut s8.fu_r,
                sc.nb_r,
            )?;
            exec.quantize_q8(
                &s8.fu_r,
                &mut s8.fq_r,
                &mut s8.fs_r,
                sc.nb_r * 32 * hp.moe_ff,
            )?;
            exec.q8_0_moe_down_sorted(
                down,
                &sc.d_srow,
                &sc.d_sslot,
                &sc.d_bexp,
                &sc.d_w,
                &s8.fq_r,
                &s8.fs_r,
                &mut sc.d_part,
                hp.n_active,
                sc.nb_r,
            )?;
            exec.moe_slot_combine(&sc.d_part, &mut st.d_mx, embd, hp.n_active, r)?;
            exec.moe_align(
                &sc.d_sh_idx,
                &mut sc.d_srow_s,
                &mut sc.d_sslot_s,
                &mut sc.d_bexp_s,
                r,
                1,
                1,
                sc.nb_s,
            )?;
            exec.q8_0_moe_up_relu2_sorted(
                sh_up,
                &sc.d_srow_s,
                &sc.d_bexp_s,
                &s8.xq,
                &s8.xs,
                &mut s8.fu_s,
                sc.nb_s,
            )?;
            exec.quantize_q8(
                &s8.fu_s,
                &mut s8.fq_s,
                &mut s8.fs_s,
                sc.nb_s * 32 * hp.shared_ff,
            )?;
            exec.q8_0_moe_down_sorted(
                sh_down,
                &sc.d_srow_s,
                &sc.d_sslot_s,
                &sc.d_bexp_s,
                &sc.d_sh_w,
                &s8.fq_s,
                &s8.fs_s,
                &mut sc.d_proj,
                1,
                sc.nb_s,
            )?;
            exec.moe_slot_combine(&sc.d_proj, &mut st.d_mx, embd, 1, r)?;
        }

        if head_out {
            exec.rmsnorm_batch(&st.d_mx, &m.head_norm.buf, &mut st.d_hd, embd, eps, 1)?;
        }
        Ok(())
    }

    /// Draft up to k tokens for one slot: chain (pending token, pending_h)
    /// -> block -> trunk head -> pick, feeding the head's own normed output
    /// back as the next h (qwen35's single-head chain). The chain's KV cells
    /// at pos.. are speculative - the verify round's append overwrites the
    /// accepted span, and the next draft rewrites its own cells before
    /// attending, so nothing stale is ever read.
    pub(crate) fn mtp_draft(
        &mut self,
        slot: usize,
        pos: usize,
        committed: u32,
        k: usize,
    ) -> Result<Vec<u32>, GpuModelError> {
        assert!((1..=MTP_MAX_DRAFT).contains(&k));
        let exec = self.exec.clone();
        let (embd, vocab) = (self.hp.hidden, self.hp.vocab);
        let drv = |e: cudarc::driver::DriverError| crate::gpu::from_driver(e);
        let mut drafts = Vec::with_capacity(k);
        let mut tok = committed;
        for i in 0..k {
            {
                let bs = self.batch.as_mut().expect("batch enabled");
                let sc = &mut bs.sc;
                let stm = &exec.stream;
                let mut t = sc
                    .d_tok
                    .try_slice_mut(0..1)
                    .ok_or_else(|| GpuError::Driver("tok".into()))?;
                stm.memcpy_htod(&[tok], &mut t).map_err(drv)?;
                let mut p = sc
                    .d_pos
                    .try_slice_mut(0..1)
                    .ok_or_else(|| GpuError::Driver("pos".into()))?;
                stm.memcpy_htod(&[(pos + i) as u32], &mut p).map_err(drv)?;
                let mut s = sc
                    .d_slots
                    .try_slice_mut(0..1)
                    .ok_or_else(|| GpuError::Driver("slots".into()))?;
                stm.memcpy_htod(&[slot as u32], &mut s).map_err(drv)?;
            }
            {
                let st = self
                    .mtp
                    .as_mut()
                    .expect("mtp weights")
                    .state
                    .as_mut()
                    .expect("mtp state");
                if i == 0 {
                    exec.copy_region(&st.pending_h, slot * embd, &mut st.d_hin, 0, embd)?;
                } else {
                    exec.copy_region(&st.d_hd, 0, &mut st.d_hin, 0, embd)?;
                }
            }
            self.mtp_block_rows(1, true)?;
            {
                let head = match &self.lm_head {
                    HeadW::Qw(q) => q,
                    HeadW::Nvf4(_) => {
                        return Err(GpuModelError::Unsupported(
                            "in-file MTP is the GGUF lane".into(),
                        ));
                    }
                };
                let st = self
                    .mtp
                    .as_mut()
                    .expect("mtp weights")
                    .state
                    .as_mut()
                    .expect("mtp state");
                let bs = self.batch.as_mut().expect("batch enabled");
                let sc = &mut bs.sc;
                gemv_any(&exec, head, &st.d_hd, &mut sc.head_logits)?;
                exec.argmax_rows(&sc.head_logits, &mut st.d_pick, 1, vocab)?;
                let view = st
                    .d_pick
                    .try_slice(0..1)
                    .ok_or_else(|| GpuError::Driver("pick view".into()))?;
                let ids: Vec<u32> = exec
                    .stream
                    .clone_dtoh(&view)
                    .map_err(|e| GpuError::Driver(e.to_string()))?;
                tok = ids[0];
            }
            drafts.push(tok);
        }
        Ok(drafts)
    }
}
