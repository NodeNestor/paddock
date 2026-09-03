//! PaddleOCR-VL continuous batching: paged KV, batched decode,
//! chunked prefill - the deepseek-ocr lane reshaped for this family's
//! two deltas, with everything MoE/R-SWA deleted (the ERNIE decoder is a
//! plain dense causal stack):
//!
//! 1. **M-RoPE rides every pass as a fourth row stream.** Next to granite's
//!    (token, position, slot) triplet each pass uploads `d_mrope` - axis-major
//!    `[4, r]` (t, h, w, unused) - and the rope kernels turn on it while KV
//!    append and attention bounds keep reading the SEQUENCE position. Prefill
//!    rows take their axes from the request's `build_positions` plan; decode
//!    rows take `pos + mrope_delta[slot]` on all axes - qwen35's constant-
//!    offset recipe, where the delta is fixed at prefill completion as
//!    `positions.next - prefill_rows` (0 for text, negative once an image
//!    compressed the position space).
//! 2. **The matmul planes stay on the gemma4 `Plane` seam** (bf16 verbatim,
//!    `bf16_gemv` at r==1 / `bf16_gemm` otherwise) - no quantization ladder,
//!    so the whole lane keeps the same-weights parity story.
//!
//! Resume numeric class, stated where it happens (the deepseek-ocr note
//! carries the full argument): a radix-resumed tail is not bitwise-identical
//! to its cold pass - `pd_rmsnorm_batch` elects its reduction width at the
//! 64-row boundary, and gemv (serial spine) vs gemm (this lane) are separate
//! last-ulp classes too. All of it is the sanctioned f16-plane near-tie
//! class - it shows up as a ±1 LOC bin at most, and the OCR battery judges
//! by CER delta, not bit equality.
//!
//! Decode ticks are captured into per-r CUDA graphs (granite's recipe). The
//! M-RoPE divergence lives entirely in the pre-replay `d_mrope` upload, so
//! the captured body is delta-agnostic.

use std::collections::HashMap;

use cudarc::driver::CudaSlice;
use cudarc::driver::sys::CUstreamCaptureMode;

use crate::gpu::{GpuError, KvDtype};
use crate::gpu_model::gpt_oss::GpuModelError;
use crate::gpu_model::granite::batch::pf_rows;
use crate::kv_plan;
use crate::kv_pool::{BlockTable, KvPool};

use super::load::GpuPaddleOcrVl;

/// VRAM slack the slot-fit math leaves untouched (graph/scratch churn).
const VRAM_HEADROOM: usize = 1 << 30;

/// FlashDecoding split ceiling (partial-scratch sizing).
const MAX_ATTN_SPLITS: usize = 16;

/// KV splits for batched decode attention - position-INDEPENDENT so a
/// captured per-r graph can bake it. 16 q-heads × ≤8 rows is a ≤128-block
/// grid on a 188-SM die; splits fill it (deepseek-ocr's lesson: the
/// serial decode walk was work-bound, not launch-bound).
fn attn_splits_for(n_heads: usize, batch: usize, sm_count: usize) -> usize {
    if paddock_models::dev_var_os!("PADDOCK_NO_ATTN_SPLIT").is_some() {
        return 1;
    }
    if n_heads * batch >= 2 * 3 * sm_count {
        return 1; // die already saturated
    }
    if let Some(n) = paddock_models::dev_var!("PADDOCK_ATTN_SPLITS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n >= 1)
    {
        return n.min(MAX_ATTN_SPLITS);
    }
    MAX_ATTN_SPLITS
}

pub(crate) struct LayerKv {
    pub k: CudaSlice<u8>,
    pub v: CudaSlice<u8>,
}

/// Batched-lane scratch, sized once at enable for `cap`-row passes.
pub(crate) struct BatchScratch {
    pub x: CudaSlice<f32>,
    pub xn: CudaSlice<f32>,
    /// [cap, 2048] - the decoupled q width, WIDER than the hidden 1024.
    pub q: CudaSlice<f32>,
    pub k: CudaSlice<f32>,
    pub v: CudaSlice<f32>,
    pub attn: CudaSlice<f32>,
    pub proj: CudaSlice<f32>,
    /// no-op sinks [n_heads] - -1e30, the softmax identity (granite's burn).
    pub sinks: CudaSlice<f32>,
    pub ffn_gate: CudaSlice<f32>,
    pub ffn_up: CudaSlice<f32>,
    /// FlashDecoding partial planes, sized by slots (decode rows only).
    pub attn_o: CudaSlice<f32>,
    pub attn_ml: CudaSlice<f32>,
    // row streams: tokens, sequence positions, the [4, cap] M-RoPE axes, slots
    pub d_toks: CudaSlice<u32>,
    pub d_pos: CudaSlice<u32>,
    pub d_mrope: CudaSlice<u32>,
    pub d_slots: CudaSlice<u32>,
    pub head_logits: CudaSlice<f32>,
    // device sampling: rows × {inv_t, u, mode, pad} params and the picked ids
    pub d_par: CudaSlice<u32>,
    pub d_out: CudaSlice<u32>,
}

/// The whole batching state: pool + tables + scratch + the per-slot M-RoPE
/// deltas.
pub(crate) struct BatchState {
    pub n_slots: usize,
    /// scratch row capacity (pf_rows + slots) - mixed ticks size by it
    pub cap: usize,
    pub bps: usize,
    pub pool: KvPool,
    pub tables: Vec<BlockTable>,
    pub bt_host: Vec<u32>,
    pub d_bt: CudaSlice<u32>,
    pub kv: Vec<LayerKv>,
    pub sc: BatchScratch,
    pub prefix: Option<crate::paged_radix::PagedRadix>,
    pub kv_bytes: u64,
    pub graphs: HashMap<usize, super::SendGraph>,
    /// Slot 0's decode cursor for the serial `Generator::forward` surface
    /// (warmup and any serial-path caller) - qwen3_asr's pattern.
    pub slot0_pos: usize,
    /// Per-slot M-RoPE offset: decode roping runs at `pos + delta` on all
    /// axes. Set at prefill completion (`next - rows`); 0 for text prompts.
    pub mrope_delta: Vec<i64>,
    /// Rows the last `prefix_resume` adopted, per slot - `take_prefill_reused`.
    pub last_reused: Vec<usize>,
}

/// Contiguous same-slot runs over a chunk's prefill rows - an attention
/// launch never mixes two slots' query rows.
pub(super) struct PfCuts {
    pub(super) dec: usize,
    pub(super) runs: Vec<(usize, usize)>,
}

fn drv(e: cudarc::driver::DriverError) -> GpuError {
    crate::gpu::from_driver(e)
}

impl GpuPaddleOcrVl {
    /// Allocate the paged-KV + scratch state for up to `max_batch` slots.
    /// A pack without paged KV gets a real error - the service's documented
    /// "genuinely can't" signal, which routes serving onto the exclusive
    /// serial lane (its contract; the trait-default `Ok(1)` trap).
    pub fn enable_batch(&mut self, max_batch: usize) -> Result<usize, GpuModelError> {
        if !self.exec.has_paged_kv() {
            return Err(GpuModelError::Unsupported(
                "paddleocr-vl batched serving needs the paged-KV pack (serial lane only)".into(),
            ));
        }
        // the serial dense KV makes way for the paged stores
        self.decode = None;
        self.scratch = None;
        self.batch = None;
        self.exec.trim_mem_pool();

        let hp = self.hp.clone();
        let (embd, nh, hd) = (hp.n_embd, hp.n_head, hp.head_dim);
        let q_dim = nh * hd;
        let kv_dim = hp.n_kv_heads * hd;
        let kvb = self.kv_dtype.bytes();
        let bps = self.max_ctx.div_ceil(16);
        let n_layer = hp.n_layer;
        let block_bytes = n_layer * 16 * kv_dim * 2 * kvb;
        let cap = pf_rows() + max_batch;
        let slots = max_batch;

        let scratch_est = cap * (4 * embd + 2 * q_dim + 2 * kv_dim + 2 * hp.n_ff) * 4
            + slots.max(1) * hp.n_vocab * 4
            + (128 << 20);
        let px_on = !super::prefix::prefix_disabled();
        let retain = if px_on {
            super::prefix::retention_blocks()
        } else {
            0
        };
        // One arbiter sizes the KV store: crate::kv_plan.
        let grant = self
            .exec
            .vram_headroom()
            .ok_or_else(|| GpuError::Driver("no free-VRAM reading".into()))?;
        let demand = kv_plan::Demand {
            family: "paddleocr-vl",
            max_ctx: self.max_ctx,
            slots: max_batch,
            blocks_per_slot: bps,
            block_bytes: block_bytes as u64,
            // one block id addresses every layer, so no KV is per-slot here
            per_slot_bytes: 0,
            retention_blocks: retain,
            // every slot must at least hold a full chunk's worth of prompt, or
            // admission deadlocks on its own first chunk
            floor_blocks_per_slot: pf_rows().div_ceil(16),
            floor_blocks_min: 256,
            reserves: vec![
                kv_plan::Reserve::new("graph/scratch slack", VRAM_HEADROOM as u64),
                kv_plan::Reserve::new("prefill scratch", scratch_est as u64),
            ],
            ..Default::default()
        };
        let plan = demand
            .plan(grant)
            .map_err(|e| GpuModelError::WontFit(e.message))?;
        plan.report(&demand, grant);
        let pool_blocks = plan.pool_blocks;
        // the fit may have seated fewer than asked; everything below sizes to
        // what we actually got
        let slots = plan.slots;

        let e = &self.exec;
        let mut kv = Vec::with_capacity(n_layer);
        let mut kv_bytes = 0u64;
        for _ in 0..n_layer {
            let bytes = pool_blocks * 16 * kv_dim * kvb;
            kv_bytes += 2 * bytes as u64;
            kv.push(LayerKv {
                k: e.alloc_u8(bytes)?,
                v: e.alloc_u8(bytes)?,
            });
        }

        let sc = BatchScratch {
            x: e.alloc(cap * embd)?,
            xn: e.alloc(cap * embd)?,
            q: e.alloc(cap * q_dim)?,
            k: e.alloc(cap * kv_dim)?,
            v: e.alloc(cap * kv_dim)?,
            attn: e.alloc(cap * q_dim)?,
            proj: e.alloc(cap * embd)?,
            sinks: e.alloc_no_sinks(nh)?,
            ffn_gate: e.alloc(cap * hp.n_ff)?,
            ffn_up: e.alloc(cap * hp.n_ff)?,
            attn_o: e.alloc(nh * slots.max(1) * MAX_ATTN_SPLITS * hd)?,
            attn_ml: e.alloc(nh * slots.max(1) * MAX_ATTN_SPLITS * 2)?,
            d_toks: e.alloc_u32(cap)?,
            d_pos: e.alloc_u32(cap)?,
            d_mrope: e.alloc_u32(4 * cap)?,
            d_slots: e.alloc_u32(cap)?,
            head_logits: e.alloc(slots.max(1) * hp.n_vocab)?,
            d_par: e.alloc_u32(slots.max(1) * 4)?,
            d_out: e.alloc_u32(slots.max(1))?,
        };

        self.batch = Some(BatchState {
            n_slots: slots,
            cap,
            bps,
            pool: KvPool::with_blocks(pool_blocks as u32),
            tables: (0..slots).map(|_| BlockTable::new()).collect(),
            bt_host: vec![0u32; slots * bps],
            d_bt: e.alloc_u32(slots * bps)?,
            kv,
            sc,
            prefix: px_on.then(crate::paged_radix::PagedRadix::new),
            kv_bytes,
            graphs: HashMap::new(),
            slot0_pos: 0,
            mrope_delta: vec![0; slots],
            last_reused: vec![0; slots],
        });
        tracing::info!(
            "paddleocr-vl batch: {slots} slots, {n_layer}-layer pool {pool_blocks} blocks \
             ({:.2} GiB, {} tokens), {} rows/chunk",
            (pool_blocks * block_bytes) as f64 / (1u64 << 30) as f64,
            pool_blocks * 16,
            pf_rows(),
        );
        Ok(slots)
    }

    /// Back every `(slot, position)` with a physical pool block.
    /// PoolExhausted sheds radix retention first.
    pub(super) fn ensure_rows(&mut self, slots: &[u32], pos: &[u32]) -> Result<(), GpuModelError> {
        let bs = self.batch.as_mut().expect("batch enabled");
        let mut grew = false;
        for (i, &s) in slots.iter().enumerate() {
            let s = s as usize;
            let before = bs.tables[s].blocks().len();
            loop {
                match bs.tables[s].ensure(pos[i] as usize, &mut bs.pool) {
                    Ok(()) => break,
                    Err(_) => {
                        let shed = bs
                            .prefix
                            .as_mut()
                            .and_then(|r| r.evict_lru(&mut bs.pool))
                            .is_some();
                        if !shed {
                            return Err(GpuModelError::PoolExhausted);
                        }
                    }
                }
            }
            let now = bs.tables[s].blocks().len();
            // A table past its bt stripe would write the NEIGHBOR slot's
            // entries (silent cross-slot corruption) before the flat indexing
            // even panics on the last slot. The service clamps generation to
            // the window rows-based at prefill finish, so this firing means
            // that contract broke - refuse loudly instead of spilling.
            if now > bs.bps {
                return Err(GpuModelError::ContextExceeded {
                    got: now * 16,
                    max: bs.bps * 16,
                });
            }
            if now > before {
                grew = true;
                let base = s * bs.bps;
                for j in before..now {
                    bs.bt_host[base + j] = bs.tables[s].blocks()[j];
                }
            }
        }
        if grew {
            self.exec
                .stream
                .memcpy_htod(&bs.bt_host, &mut bs.d_bt)
                .map_err(drv)?;
        }
        Ok(())
    }

    pub fn release_inactive_slots(&mut self, occupied: &[bool]) {
        let Some(bs) = self.batch.as_mut() else {
            return;
        };
        for (s, occ) in occupied.iter().enumerate() {
            if !occ && s < bs.tables.len() && !bs.tables[s].blocks().is_empty() {
                bs.tables[s].clear(&mut bs.pool);
                bs.mrope_delta[s] = 0;
            }
        }
    }

    pub fn pool_free_blocks(&self) -> Option<usize> {
        self.batch
            .as_ref()
            .map(|b| b.pool.free_blocks() + self.prefix_evictable())
    }

    // ── prefill ─────────────────────────────────────────────────────────────

    /// Prefill a whole TEXT prompt into `slot` (chunked at `pf_rows`),
    /// publish the prefix, and return the last token's logits. Text M-RoPE
    /// degenerates to the sequence arange, so the slot's delta pins at 0.
    pub fn forward_prefill(
        &mut self,
        slot: usize,
        tokens: &[u32],
    ) -> Result<Vec<f32>, GpuModelError> {
        self.admit(slot, tokens.len())?;
        let mut base = self.prefill_resume_rows(slot, tokens, tokens.len())?;
        let mut last_len = 0usize;
        for chunk in tokens[base..].chunks(pf_rows()) {
            let rows: Vec<(u32, u32, u32)> = chunk
                .iter()
                .enumerate()
                .map(|(j, &t)| (slot as u32, (base + j) as u32, t))
                .collect();
            let mrope = text_mrope(base, chunk.len());
            self.rows_pass_body(&rows, &mrope, 0)?;
            base += chunk.len();
            last_len = chunk.len();
        }
        self.prefix_insert(slot, tokens);
        let bs = self.batch.as_mut().expect("batch enabled");
        bs.mrope_delta[slot] = 0;
        self.head_row(last_len - 1)
    }

    /// Admission checks + slot reset for an `n_rows`-row prompt (row count,
    /// not token count: the multimodal path admits image rows too).
    pub(super) fn admit(&mut self, slot: usize, n_rows: usize) -> Result<(), GpuModelError> {
        let n_slots = self.batch.as_ref().expect("batch enabled").n_slots;
        if slot >= n_slots {
            return Err(GpuModelError::Unsupported(format!(
                "slot {slot} >= enabled {n_slots}"
            )));
        }
        if n_rows == 0 {
            return Err(GpuModelError::Unsupported("empty prompt".into()));
        }
        if n_rows > self.max_ctx {
            return Err(GpuModelError::Unsupported(format!(
                "prompt {n_rows} rows > max_ctx {}",
                self.max_ctx
            )));
        }
        {
            let bs = self.batch.as_mut().expect("batch enabled");
            bs.tables[slot].clear(&mut bs.pool);
            bs.mrope_delta[slot] = 0;
        }
        self.ensure_rows(&[slot as u32], &[(n_rows - 1) as u32])
    }

    pub(super) fn prefill_resume_rows(
        &mut self,
        slot: usize,
        keys: &[u32],
        n_rows: usize,
    ) -> Result<usize, GpuModelError> {
        let start = self.prefix_resume(slot, keys)?;
        self.batch.as_mut().expect("batch enabled").last_reused[slot] = start;
        if start > 0 {
            self.ensure_rows(&[slot as u32], &[(n_rows - 1) as u32])?;
        }
        Ok(start)
    }

    /// One pass over `chunk` rows (slot, position, token) with their M-RoPE
    /// stream (`mrope` is axis-major [4, r], built by the caller - decode
    /// rows carry pos+delta, prefill rows their plan's axes). The leading
    /// `dec` rows are the decode band. Rows belonging to a QUEUED image
    /// prompt have their placeholder embeddings overwritten from the entry's
    /// feature planes by (slot, position) - see chunked.rs.
    pub(super) fn rows_pass_body(
        &mut self,
        chunk: &[(u32, u32, u32)],
        mrope: &[u32],
        dec: usize,
    ) -> Result<(), GpuModelError> {
        let r = chunk.len();
        debug_assert_eq!(mrope.len(), 4 * r);
        let toks: Vec<u32> = chunk.iter().map(|x| x.2).collect();
        let slots_v: Vec<u32> = chunk.iter().map(|x| x.0).collect();
        let pos: Vec<u32> = chunk.iter().map(|x| x.1).collect();
        let mut runs: Vec<(usize, usize)> = Vec::new();
        for (i, x) in chunk.iter().enumerate().skip(dec) {
            match runs.last_mut() {
                Some((off, n)) if chunk[*off].0 == x.0 => *n += 1,
                _ => runs.push((i, 1)),
            }
        }
        self.ensure_rows(&slots_v, &pos)?;
        self.upload_rows(&toks, &pos, mrope, &slots_v)?;
        self.embed_rows(r)?;
        self.splice_queued_rows(chunk, dec)?;
        self.layer_walk(r, Some(&PfCuts { dec, runs }))?;
        Ok(())
    }

    // ── decode ──────────────────────────────────────────────────────────────

    /// One batched decode step; row i drives `slots[i]` at sequence position
    /// `positions[i]` and M-RoPE position `positions[i] + delta[slot]`.
    /// Leaves [r, vocab] logits in head_logits.
    pub fn batch_step_slots(
        &mut self,
        tokens: &[u32],
        positions: &[u32],
        slots: &[u32],
    ) -> Result<(), GpuModelError> {
        let r = tokens.len();
        assert_eq!(r, positions.len());
        assert_eq!(r, slots.len());
        let n_slots = self.batch.as_ref().expect("batch enabled").n_slots;
        assert!(r <= n_slots, "rows {r} > enabled {n_slots}");
        let mrope = self.decode_mrope(positions, slots);
        self.ensure_rows(slots, positions)?;
        self.upload_rows(tokens, positions, &mrope, slots)?;
        self.step_replay(r)
    }

    pub fn batch_step(&mut self, tokens: &[u32], positions: &[u32]) -> Result<(), GpuModelError> {
        let ident: Vec<u32> = (0..tokens.len() as u32).collect();
        self.batch_step_slots(tokens, positions, &ident)
    }

    /// Decode-row M-RoPE stream: `pos + delta[slot]` on all four axes
    /// (axis 3 has a zero section and is never read; keeping it equal is
    /// simplest and matches the serial spine's upload).
    pub(super) fn decode_mrope(&self, positions: &[u32], slots: &[u32]) -> Vec<u32> {
        let bs = self.batch.as_ref().expect("batch enabled");
        let r = positions.len();
        let mut m = vec![0u32; 4 * r];
        for i in 0..r {
            let p = (positions[i] as i64 + bs.mrope_delta[slots[i] as usize]) as u32;
            for ax in 0..4 {
                m[ax * r + i] = p;
            }
        }
        m
    }

    fn step_body(&mut self, r: usize) -> Result<(), GpuModelError> {
        self.embed_rows(r)?;
        self.layer_walk(r, None)?;
        self.head_rows(r)
    }

    fn capture_body(
        &mut self,
        body: impl FnOnce(&mut Self) -> Result<(), GpuModelError>,
        what: &str,
    ) -> Result<super::SendGraph, GpuModelError> {
        let exec = self.exec.clone();
        exec.stream
            .synchronize()
            .map_err(|e| GpuError::Driver(format!("{what} pre-capture sync: {e}")))?;
        exec.stream
            .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
            .map_err(|e| GpuError::Driver(format!("{what} begin_capture: {e}")))?;
        let rec = body(self);
        let graph = crate::gpu::end_capture_no_flags(&exec.stream)
            .map_err(|e| GpuError::Driver(format!("{what} end_capture: {e}")));
        rec?;
        let graph =
            graph?.ok_or_else(|| GpuError::Driver(format!("{what} capture produced no graph")))?;
        Ok(super::SendGraph(graph))
    }

    fn step_replay(&mut self, r: usize) -> Result<(), GpuModelError> {
        if !self
            .batch
            .as_ref()
            .expect("batch enabled")
            .graphs
            .contains_key(&r)
        {
            let g = self.capture_body(|s| s.step_body(r), "decode")?;
            self.batch
                .as_mut()
                .expect("batch enabled")
                .graphs
                .insert(r, g);
        }
        self.batch.as_ref().expect("batch enabled").graphs[&r]
            .0
            .launch()
            .map_err(|e| GpuError::Driver(format!("decode graph launch: {e}")))?;
        Ok(())
    }

    pub(super) fn upload_rows(
        &mut self,
        tokens: &[u32],
        pos: &[u32],
        mrope: &[u32],
        slots: &[u32],
    ) -> Result<(), GpuModelError> {
        let r = tokens.len();
        let bs = self.batch.as_mut().expect("batch enabled");
        self.exec.upload_u32(tokens, &mut bs.sc.d_toks)?;
        self.exec.upload_u32(pos, &mut bs.sc.d_pos)?;
        self.exec.upload_u32(slots, &mut bs.sc.d_slots)?;
        // d_mrope is read as [4, r] AXIS-major by the kernel, so the four
        // axes must land contiguously at the CURRENT r, not at cap.
        let mut v = bs.sc.d_mrope.slice_mut(0..4 * r);
        self.exec.stream.memcpy_htod(mrope, &mut v).map_err(drv)?;
        Ok(())
    }

    pub(super) fn embed_rows(&mut self, r: usize) -> Result<(), GpuModelError> {
        let embd = self.hp.n_embd;
        let bs = self.batch.as_mut().expect("batch enabled");
        self.exec
            .embed_gather_plane(&self.tok_embd, &bs.sc.d_toks, &mut bs.sc.x, embd, r, 1.0)?;
        Ok(())
    }

    pub(super) fn layer_walk(
        &mut self,
        r: usize,
        cuts: Option<&PfCuts>,
    ) -> Result<(), GpuModelError> {
        let exec = self.exec.clone();
        let hp = self.hp.clone();
        let (embd, nh, n_kv, hd) = (hp.n_embd, hp.n_head, hp.n_kv_heads, hp.head_dim);
        let kv_dim = n_kv * hd;
        let (eps, n_ff, n_rot, sections) = (hp.eps, hp.n_ff, hp.n_rot, hp.sections);
        let scale = 1.0 / (hd as f32).sqrt();
        let yarn = self.mrope_params();
        let kv_dtype = self.kv_dtype;
        let pf = cuts.is_some();
        let r1 = r == 1 && !pf;
        // hd128 G=8 WMMA-class prefill tile (the v4 family; the <128,8>
        // instantiation is this task's pack rung - an older pack falls back
        // to the scalar paged tile inside the same launcher, correct at a
        // worse speed). f16 caches only, the deepseek-ocr gate.
        let wmma_pf = pf
            && hd == 128
            && matches!(kv_dtype, KvDtype::Fp16)
            && exec.has_attn_prefill_f16_paged()
            && paddock_models::dev_var_os!("PADDOCK_OCR_NO_WMMA").is_none();
        let bs = self.batch.as_mut().expect("batch enabled");
        let bps = bs.bps;

        for (li, layer) in self.layers.iter().enumerate() {
            let sc = &mut bs.sc;
            exec.rmsnorm_batch(&sc.x, &layer.attn_norm.buf, &mut sc.xn, embd, eps, r)?;
            if r1 {
                layer.wq.gemv(&exec, &sc.xn, &mut sc.q)?;
                layer.wk.gemv(&exec, &sc.xn, &mut sc.k)?;
                layer.wv.gemv(&exec, &sc.xn, &mut sc.v)?;
            } else {
                layer.wq.gemm(&exec, &sc.xn, &mut sc.q, r)?;
                layer.wk.gemm(&exec, &sc.xn, &mut sc.k, r)?;
                layer.wv.gemm(&exec, &sc.xn, &mut sc.v, r)?;
            }
            // sectioned 3-axis rope on the M-RoPE stream; KV lands at the
            // SEQUENCE position (d_pos) - the two diverge past any image.
            exec.mrope(&mut sc.q, &sc.d_mrope, r, nh, hd, n_rot, yarn, sections)?;
            exec.mrope(&mut sc.k, &sc.d_mrope, r, n_kv, hd, n_rot, yarn, sections)?;
            let kvs = &mut bs.kv[li];
            exec.kv_append_batch_paged(
                &sc.k,
                &mut kvs.k,
                &sc.d_pos,
                Some(&sc.d_slots),
                &bs.d_bt,
                bps,
                kv_dim,
                r,
                kv_dtype,
            )?;
            exec.kv_append_batch_paged(
                &sc.v,
                &mut kvs.v,
                &sc.d_pos,
                Some(&sc.d_slots),
                &bs.d_bt,
                bps,
                kv_dim,
                r,
                kv_dtype,
            )?;
            match cuts {
                Some(c) => {
                    let all: &[(usize, usize)] = if c.runs.len() == 1 && c.dec == 0 {
                        &[(0, r)]
                    } else {
                        &c.runs
                    };
                    if c.dec > 0 {
                        let ns = attn_splits_for(nh, c.dec, exec.sm_count());
                        if ns > 1 && exec.has_attn_partial_batch_paged() {
                            exec.attn_partial_batch_paged(
                                &sc.q,
                                &kvs.k,
                                &kvs.v,
                                &mut sc.attn_o,
                                &mut sc.attn_ml,
                                &sc.d_pos,
                                Some(&sc.d_slots),
                                &bs.d_bt,
                                bps,
                                nh,
                                n_kv,
                                hd,
                                kv_dim,
                                0,
                                ns,
                                c.dec,
                                scale,
                                kv_dtype,
                            )?;
                            exec.attn_combine_batch(
                                &sc.attn_o,
                                &sc.attn_ml,
                                &sc.sinks,
                                &mut sc.attn,
                                nh,
                                hd,
                                ns,
                                c.dec,
                            )?;
                        } else {
                            exec.attn_decode_batch_rows_paged(
                                &sc.q,
                                &kvs.k,
                                &kvs.v,
                                &sc.sinks,
                                &mut sc.attn,
                                &sc.d_pos,
                                Some(&sc.d_slots),
                                &bs.d_bt,
                                bps,
                                nh,
                                n_kv,
                                hd,
                                kv_dim,
                                0,
                                0,
                                c.dec,
                                scale,
                                kv_dtype,
                            )?;
                        }
                    }
                    for &(off, len) in all {
                        if off < c.dec {
                            continue;
                        }
                        if wmma_pf {
                            exec.attn_prefill_f16_paged_at(
                                &sc.q,
                                &kvs.k,
                                &kvs.v,
                                &sc.sinks,
                                &mut sc.attn,
                                &sc.d_pos,
                                &sc.d_slots,
                                off,
                                &bs.d_bt,
                                bps,
                                nh,
                                n_kv,
                                hd,
                                kv_dim,
                                0,
                                len,
                                scale,
                                kv_dtype,
                            )?;
                        } else {
                            exec.attn_decode_batch_rows_paged(
                                &sc.q,
                                &kvs.k,
                                &kvs.v,
                                &sc.sinks,
                                &mut sc.attn,
                                &sc.d_pos,
                                Some(&sc.d_slots),
                                &bs.d_bt,
                                bps,
                                nh,
                                n_kv,
                                hd,
                                kv_dim,
                                0,
                                off,
                                len,
                                scale,
                                kv_dtype,
                            )?;
                        }
                    }
                }
                None => {
                    let ns = attn_splits_for(nh, r, exec.sm_count());
                    if ns > 1 && exec.has_attn_partial_batch_paged() {
                        exec.attn_partial_batch_paged(
                            &sc.q,
                            &kvs.k,
                            &kvs.v,
                            &mut sc.attn_o,
                            &mut sc.attn_ml,
                            &sc.d_pos,
                            Some(&sc.d_slots),
                            &bs.d_bt,
                            bps,
                            nh,
                            n_kv,
                            hd,
                            kv_dim,
                            0,
                            ns,
                            r,
                            scale,
                            kv_dtype,
                        )?;
                        exec.attn_combine_batch(
                            &sc.attn_o,
                            &sc.attn_ml,
                            &sc.sinks,
                            &mut sc.attn,
                            nh,
                            hd,
                            ns,
                            r,
                        )?;
                    } else {
                        exec.attn_decode_batch_paged(
                            &sc.q,
                            &kvs.k,
                            &kvs.v,
                            &sc.sinks,
                            &mut sc.attn,
                            &sc.d_pos,
                            Some(&sc.d_slots),
                            &bs.d_bt,
                            bps,
                            nh,
                            n_kv,
                            hd,
                            kv_dim,
                            0,
                            r,
                            scale,
                            kv_dtype,
                        )?;
                    }
                }
            }
            if r1 {
                layer.wo.gemv(&exec, &sc.attn, &mut sc.proj)?;
            } else {
                layer.wo.gemm(&exec, &sc.attn, &mut sc.proj, r)?;
            }
            exec.add(&mut sc.x, &sc.proj, r * embd)?;

            exec.rmsnorm_batch(&sc.x, &layer.ffn_norm.buf, &mut sc.xn, embd, eps, r)?;
            if r1 {
                layer.gate.gemv(&exec, &sc.xn, &mut sc.ffn_gate)?;
                layer.up.gemv(&exec, &sc.xn, &mut sc.ffn_up)?;
                exec.swiglu(&mut sc.ffn_gate, &sc.ffn_up, n_ff)?;
                layer.down.gemv(&exec, &sc.ffn_gate, &mut sc.proj)?;
            } else {
                layer.gate.gemm(&exec, &sc.xn, &mut sc.ffn_gate, r)?;
                layer.up.gemm(&exec, &sc.xn, &mut sc.ffn_up, r)?;
                exec.swiglu(&mut sc.ffn_gate, &sc.ffn_up, r * n_ff)?;
                layer.down.gemm(&exec, &sc.ffn_gate, &mut sc.proj, r)?;
            }
            exec.add(&mut sc.x, &sc.proj, r * embd)?;
        }
        Ok(())
    }

    /// Norm rows 0..rows and run the untied head into head_logits [rows, vocab].
    pub(super) fn head_rows(&mut self, rows: usize) -> Result<(), GpuModelError> {
        let exec = self.exec.clone();
        let (embd, eps) = (self.hp.n_embd, self.hp.eps);
        let bs = self.batch.as_mut().expect("batch enabled");
        let sc = &mut bs.sc;
        exec.rmsnorm_batch(&sc.x, &self.output_norm.buf, &mut sc.xn, embd, eps, rows)?;
        if rows == 1 {
            self.lm_head.gemv(&exec, &sc.xn, &mut sc.head_logits)?;
        } else {
            self.lm_head
                .gemm(&exec, &sc.xn, &mut sc.head_logits, rows)?;
        }
        Ok(())
    }

    fn head_row_at(&mut self, row: usize) -> Result<(), GpuModelError> {
        let n_embd = self.hp.n_embd;
        if row > 0 {
            let exec = self.exec.clone();
            let bs = self.batch.as_mut().expect("batch enabled");
            let sc = &mut bs.sc;
            let src =
                sc.x.try_slice(row * n_embd..(row + 1) * n_embd)
                    .ok_or_else(|| GpuError::Driver("x row slice".into()))?;
            let mut dst = sc
                .proj
                .try_slice_mut(0..n_embd)
                .ok_or_else(|| GpuError::Driver("proj row slice".into()))?;
            exec.stream.memcpy_dtod(&src, &mut dst).map_err(drv)?;
            let ps = sc
                .proj
                .try_slice(0..n_embd)
                .ok_or_else(|| GpuError::Driver("proj src slice".into()))?;
            let mut xd =
                sc.x.try_slice_mut(0..n_embd)
                    .ok_or_else(|| GpuError::Driver("x dst slice".into()))?;
            exec.stream.memcpy_dtod(&ps, &mut xd).map_err(drv)?;
        }
        self.head_rows(1)
    }

    /// One prefill finisher's logits: bounce row `row` to the front, head it.
    pub(super) fn head_row(&mut self, row: usize) -> Result<Vec<f32>, GpuModelError> {
        let n_vocab = self.hp.n_vocab;
        self.head_row_at(row)?;
        let bs = self.batch.as_ref().expect("batch enabled");
        let v = bs
            .sc
            .head_logits
            .try_slice(0..n_vocab)
            .ok_or_else(|| GpuError::Driver("head row slice".into()))?;
        Ok(self.exec.stream.clone_dtoh(&v).map_err(drv)?)
    }

    /// Read the [rows, vocab] logits back to the host.
    pub fn read_batch_logits(&mut self, rows: usize) -> Result<Vec<f32>, GpuModelError> {
        let vocab = self.hp.n_vocab;
        let bs = self.batch.as_ref().expect("batch enabled");
        let v = bs
            .sc
            .head_logits
            .try_slice(0..rows * vocab)
            .ok_or_else(|| GpuError::Driver("batch logits slice".into()))?;
        if crate::tickseg::on() {
            let t = std::time::Instant::now();
            let out = self.exec.stream.clone_dtoh(&v).map_err(drv)?;
            crate::tickseg::rb(t.elapsed(), out.len() * 4);
            return Ok(out);
        }
        Ok(self.exec.stream.clone_dtoh(&v).map_err(drv)?)
    }

    // ── device sampling (deepseek-ocr's recipe verbatim) ───────────────

    /// Pack per-row sampler params (inv_t, u, mode, pad). Host/Hole rows stay
    /// mode 0 = untouched.
    fn pack_samp_par(plans: &[crate::generator::RowSample]) -> Vec<u32> {
        use crate::generator::RowSample;
        use crate::sampler::DevicePlan;
        let mut par = vec![0u32; plans.len() * 4];
        for (i, p) in plans.iter().enumerate() {
            match p {
                RowSample::Hole | RowSample::Host => {}
                RowSample::Device(DevicePlan::Greedy) => par[i * 4 + 2] = 1,
                RowSample::Device(DevicePlan::Categorical { inv_t, u }) => {
                    par[i * 4] = inv_t.to_bits();
                    par[i * 4 + 1] = u.to_bits();
                    par[i * 4 + 2] = 2;
                }
                // P65 TruncCat is qwen35-only (supports_host_head); skip-safe
                RowSample::Device(DevicePlan::TruncCat { .. }) => {}
                RowSample::Device(DevicePlan::RsVerify { .. })
                | RowSample::Device(DevicePlan::RsTrunc { .. }) => {}
            }
        }
        par
    }

    pub(crate) fn supports_device_sampling_impl(&self) -> bool {
        self.batch.is_some() && self.exec.has_sample_rows()
    }

    /// Sample head_logits rows 0..r on device with `plans`; only Host-plan
    /// rows pay a vocab-row readback. Assumes the head has already run.
    pub(super) fn sample_head_rows(
        &mut self,
        r: usize,
        plans: &[crate::generator::RowSample],
    ) -> Result<crate::generator::SampledStep, GpuModelError> {
        use crate::generator::{RowSample, SampledStep};
        assert_eq!(plans.len(), r, "one plan per row");
        let exec = self.exec.clone();
        let vocab = self.hp.n_vocab;
        let par = Self::pack_samp_par(plans);
        {
            let sc = &mut self.batch.as_mut().expect("batch enabled").sc;
            let mut v = sc
                .d_par
                .try_slice_mut(0..r * 4)
                .ok_or_else(|| GpuError::Driver("d_par slice".into()))?;
            exec.stream.memcpy_htod(&par, &mut v).map_err(drv)?;
            exec.sample_rows_at(&sc.head_logits, &sc.d_par, 0, &mut sc.d_out, 0, r, vocab)?;
        }
        let sc = &self.batch.as_ref().expect("batch enabled").sc;
        let ids_view = sc
            .d_out
            .try_slice(0..r)
            .ok_or_else(|| GpuError::Driver("d_out slice".into()))?;
        let ids = exec.stream.clone_dtoh(&ids_view).map_err(drv)?;
        let mut host_rows = Vec::new();
        for (i, p) in plans.iter().enumerate() {
            if matches!(p, RowSample::Host) {
                let v = sc
                    .head_logits
                    .try_slice(i * vocab..(i + 1) * vocab)
                    .ok_or_else(|| GpuError::Driver("host row slice".into()))?;
                host_rows.push((i, exec.stream.clone_dtoh(&v).map_err(drv)?));
            }
        }
        Ok(SampledStep { ids, host_rows })
    }

    /// Device-sampled decode tick: the dense batch step + sample_rows.
    pub(crate) fn forward_batch_sampled_impl(
        &mut self,
        tokens: &[u32],
        positions: &[u32],
        plans: &[crate::generator::RowSample],
    ) -> Result<crate::generator::SampledStep, GpuModelError> {
        self.batch_step(tokens, positions)?;
        self.sample_head_rows(tokens.len(), plans)
    }

    /// Plain-rope-through-yarn kernel params for the sectioned M-RoPE, θ from
    /// the header - one definition for the serial spine and this lane.
    pub(crate) fn mrope_params(&self) -> (f32, f32, f32, f32, f32, f32) {
        use paddock_kernels::reference::ops::YarnRope;
        YarnRope::new(
            self.hp.n_rot,
            self.hp.rope_base,
            1.0,
            self.hp.n_ctx_train,
            0.0,
            1.0,
            32.0,
            1.0,
        )
        .kernel_params()
    }
}

/// Text-row M-RoPE stream: all four axes = the sequence arange from `base`.
pub(super) fn text_mrope(base: usize, len: usize) -> Vec<u32> {
    let mut m = vec![0u32; 4 * len];
    for i in 0..len {
        let p = (base + i) as u32;
        for ax in 0..4 {
            m[ax * len + i] = p;
        }
    }
    m
}
