//! Vision splice: interleaved text/image prefill (the exclusive multimodal
//! path - the engine drains all slots first, so this runs single-sequence
//! in slot 0 from position 0).
//!
//! Semantics mirror llama.cpp mtmd for gemma4v exactly:
//! - the template's per-image `<|image|>` placeholder becomes
//!   `<|image>` (begin) + the encoder's soft tokens + `<image|>` (end)
//! - image rows enter the residual stream UNSCALED (ggml only multiplies
//!   token lookups by √n_embd - `ubatch.token ? sqrtf(n_embd) : 1.0`)
//! - image rows decode NON-CAUSALLY within their span
//!   (`mtmd_decode_use_non_causal` = true for gemma4v): our attention
//!   kernels take positions only as CAUSAL BOUNDS (rope is a separate
//!   pass), so image rows carry an attention-bound override = the span's
//!   last position while keeping their true positions for rope + KV writes.

use cudarc::driver::CudaSlice;

use crate::gpu::GpuError;
use crate::gpu_model::prefix_cache::{BLOCK_TOKENS, cut_outside_image_spans, image_key_row};
use crate::service::MmChunk;

use super::{Arch, GpuGemma4};

/// This module serves two families and they do not share a vision tower.
/// gemma-4's is a 27-layer SigLIP (RMS norms, GEGLU, NEOX rope, 3×3 avg-pool);
/// muse-glimmer's is a 50-layer Perception Encoder (LayerNorms with bias, an
/// erf-GELU MLP, NORM rope, 32×32 window attention, channel-outer pixel
/// shuffle). They share exactly one thing - the output shape, `[n, llm_embd]`
/// rows to splice - and that is where this enum joins them.
///
/// EXHAUSTIVE on PURPOSE - no `_` arm (see [`Arch`]). A third architecture in
/// this module must not compile until someone has opened its clip graph and
/// decided which tower it gets; falling through to gemma4's would produce
/// plausible features from the wrong encoder.
/// Boxed: the two towers differ ~2x in inline size and this enum lives in the
/// model struct, so the smaller arm would carry the larger one's footprint.
pub(crate) enum VisionTower {
    Gemma4(Box<super::vision::VisionModel>),
    Muse(Box<super::muse_vision::VisionModel>),
}

impl VisionTower {
    /// The projector's output width, which must equal the LLM's residual width.
    pub(crate) fn llm_embd(&self) -> usize {
        match self {
            VisionTower::Gemma4(v) => v.llm_embd(),
            VisionTower::Muse(v) => v.llm_embd(),
        }
    }

    pub(crate) fn budget(&self) -> crate::generator::VisionBudget {
        match self {
            VisionTower::Gemma4(v) => v.budget(),
            VisionTower::Muse(v) => v.budget(),
        }
    }

    /// Preprocess + encode one RGB8 image -> (device rows, row count).
    /// Preprocessing is part of the tower, not of the caller: the two disagree
    /// on the resize filter (bilinear vs LANCZOS), on whether the image is
    /// letterboxed or stretched, and on the grid the patches come out in.
    fn encode_rgb(&self, rgb: &[u8], w: usize, h: usize) -> Result<EncodedImage, GpuError> {
        match self {
            VisionTower::Gemma4(v) => {
                let (patches, gw, gh) = v.preprocess_rgb(rgb, w, h);
                let o = v.encode(&patches, gw, gh)?;
                Ok(EncodedImage {
                    embd: o.embd,
                    n_tokens: o.n_tokens,
                })
            }
            VisionTower::Muse(v) => {
                let (patches, gw, gh) = v.preprocess_rgb(rgb, w, h);
                let o = v.encode(&patches, gw, gh)?;
                Ok(EncodedImage {
                    embd: o.embd,
                    n_tokens: o.n_tokens,
                })
            }
        }
    }
}

/// One tower's output rows, whichever tower produced them - the only shape the
/// splice below cares about.
pub(crate) struct EncodedImage {
    pub embd: CudaSlice<f32>,
    pub n_tokens: usize,
}

/// One prefill row of the interleaved stream.
enum Row {
    Token(u32),
    /// (image index, row within that image's encoded embeddings)
    Image(usize, usize),
}

/// One cached vision-tower output: projected soft-token rows device-side,
/// raw RGB host-side for the exact-bytes verify (see the field note on
/// [`GpuGemma4::img_cache`]).
pub(crate) struct G4ImageCacheEntry {
    pub hash: u64,
    pub w: usize,
    pub h: usize,
    pub rgb: Vec<u8>,
    pub embd: CudaSlice<f32>,
    pub n_tokens: usize,
    pub last_used: u64,
}

/// Images held in the tower-output cache - sized so a multi-image request
/// (a document sent as page images) stays resident across turns.
const G4_IMAGE_CACHE_ENTRIES: usize = 16;

/// FNV-1a over the raw image bytes (dims folded in by the caller).
fn hash_bytes(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

impl GpuGemma4 {
    pub fn attach_vision(
        &mut self,
        map: &paddock_models::mapped::MappedGguf,
    ) -> Result<(), GpuError> {
        // The tower is elected by the TEXT model's arch, not by the mmproj's
        // projector string: they must agree, and the text side is what already
        // decided every other constant.
        let vm = match self.hp.arch {
            Arch::Gemma4 => VisionTower::Gemma4(Box::new(super::vision::VisionModel::load(
                self.exec.clone(),
                map,
            )?)),
            Arch::MuseGlimmer => VisionTower::Muse(Box::new(
                super::muse_vision::VisionModel::load(self.exec.clone(), map)?,
            )),
        };
        if vm.llm_embd() != self.hp.n_embd {
            return Err(GpuError::Driver(format!(
                "mmproj projects to {} but the model embd is {}",
                vm.llm_embd(),
                self.hp.n_embd
            )));
        }
        if self.img_beg_id.is_none() || self.img_end_id.is_none() {
            let (b, e) = super::image_markers(self.hp.arch);
            return Err(GpuError::Driver(format!(
                "vocab lacks the {b}/{e} markers - not a {} vision model",
                self.hp.arch.key()
            )));
        }
        self.vision = Some(vm);
        Ok(())
    }

    /// Encode one image through the tower, or serve it from the output cache
    /// (hash + exact-bytes verify). Hits return a dtod CLONE of the cached
    /// rows (~3 MB - noise next to the tower run it saves) so the splice code
    /// below never holds a cache borrow.
    ///
    /// Also returns the CONTENT hash, which is what the prefix radix keys this
    /// picture's rows on - the same number the cache identifies it by, so
    /// "the tower cache hit" and "the KV cache hit" can never disagree about
    /// whether two pictures are the same picture.
    fn encode_image_cached(
        &mut self,
        rgb: &[u8],
        w: usize,
        h: usize,
    ) -> Result<(EncodedImage, u64), GpuError> {
        let hash = hash_bytes(rgb)
            ^ (w as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
            ^ (h as u64).rotate_left(32);
        self.img_cache_clock += 1;
        let clock = self.img_cache_clock;
        if let Some(e) = self
            .img_cache
            .iter_mut()
            .find(|e| e.hash == hash && e.w == w && e.h == h && e.rgb == rgb)
        {
            e.last_used = clock;
            let n_tokens = e.n_tokens;
            let mut embd = self
                .exec
                .stream
                .alloc_zeros::<f32>(e.embd.len())
                .map_err(|er| GpuError::Driver(er.to_string()))?;
            self.exec
                .stream
                .memcpy_dtod(&e.embd, &mut embd)
                .map_err(|er| GpuError::Driver(er.to_string()))?;
            self.img_cache_reused += 1;
            if paddock_models::dev_var_os!("PADDOCK_VISION_DEBUG").is_some() {
                tracing::info!(
                    "[img-cache] HIT ({}x{}, {} soft tokens; reused {} total)",
                    w,
                    h,
                    n_tokens,
                    self.img_cache_reused
                );
            }
            return Ok((EncodedImage { embd, n_tokens }, hash));
        }
        let vision = self
            .vision
            .as_ref()
            .ok_or_else(|| GpuError::Driver("no mmproj attached".into()))?;
        let out = vision.encode_rgb(rgb, w, h)?;
        // cache a device copy (the fresh buffer goes to the caller)
        let mut copy = self
            .exec
            .stream
            .alloc_zeros::<f32>(out.embd.len())
            .map_err(|er| GpuError::Driver(er.to_string()))?;
        self.exec
            .stream
            .memcpy_dtod(&out.embd, &mut copy)
            .map_err(|er| GpuError::Driver(er.to_string()))?;
        if self.img_cache.len() >= G4_IMAGE_CACHE_ENTRIES {
            let lru = self
                .img_cache
                .iter()
                .enumerate()
                .min_by_key(|(_, e)| e.last_used)
                .map(|(i, _)| i)
                .expect("non-empty");
            self.img_cache.swap_remove(lru);
        }
        self.img_cache.push(G4ImageCacheEntry {
            hash,
            w,
            h,
            rgb: rgb.to_vec(),
            embd: copy,
            n_tokens: out.n_tokens,
            last_used: clock,
        });
        Ok((out, hash))
    }

    /// Exclusive multimodal prefill: encode every image, splice
    /// begin + soft tokens + end around each, prefill the whole stream from
    /// position 0 (slot 0), return the last row's logits and the ROW COUNT.
    ///
    /// The row count goes back to the caller rather than staying an internal
    /// cursor because the serial engine reports usage from it - text tokens
    /// alone under-count an image prompt by an order of magnitude.
    pub(crate) fn multimodal_prefill(
        &mut self,
        chunks: &[MmChunk],
    ) -> Result<(Vec<f32>, usize), GpuError> {
        let (logits, rows) = self.multimodal_prefill_slot(0, chunks)?;
        self.pos = rows;
        Ok((logits, rows))
    }

    /// The mm prefill against batch slot `slot` (S8 shape, qwen35 parity).
    /// Returns the last row's logits and the total row count - image rows
    /// included, so it differs from the prompt's token count and the service
    /// must use it as the slot's KV position.
    ///
    /// PREFIX CACHING APPLIES here, keyed on content rather than on
    /// row tokens: every image row carries the same placeholder, so a radix
    /// keyed on the row stream would serve one picture's KV for another. Text
    /// rows key as themselves and image rows key off the picture's content hash
    /// - see [`crate::gpu_model::prefix_cache::image_key_row`]. That is what
    ///   makes the document workload work: same page, many questions, and every
    ///   turn after the first resumes past the whole picture rather than
    ///   re-prefilling its soft tokens.
    ///
    /// gemma4v is the awkward one, and the reason granite went first. Its image
    /// rows decode NON-CAUSALLY: each attends to its span's last position, so a
    /// resume landing strictly inside a picture would re-prefill rows whose
    /// attention bound points at keys the adopted blocks already hold under a
    /// different write order. Rather than reason about whether that is benign,
    /// no CHECKPOINT is ever attached inside an image span - and since a resume
    /// position is exactly a checkpoint position, mid-span resumes cannot occur
    /// at all. See [`Self::mm_prefix_cut`].
    pub(crate) fn multimodal_prefill_slot(
        &mut self,
        slot: usize,
        chunks: &[MmChunk],
    ) -> Result<(Vec<f32>, usize), GpuError> {
        if self.vision.is_none() {
            return Err(GpuError::Driver("no mmproj attached".into()));
        }
        assert!(
            slot < self.n_slots.max(1),
            "slot {slot} >= enabled {}",
            self.n_slots
        );
        let (beg, end) = (self.img_beg_id.unwrap(), self.img_end_id.unwrap());

        // pass 1: resolve every image (tower or cache) - needs &mut self,
        // so it runs before the row stream takes any other borrows
        let mut encoded: Vec<EncodedImage> = Vec::new();
        let mut hashes: Vec<u64> = Vec::new();
        for ch in chunks {
            if let MmChunk::Image { rgb, w, h } = ch {
                let (out, hash) = self.encode_image_cached(rgb, *w, *h)?;
                encoded.push(out);
                hashes.push(hash);
            }
        }
        // pass 2: the interleaved row stream, and the radix key vector beside
        // it - one key per row, image rows keyed on content (see the doc above)
        let mut rows: Vec<Row> = Vec::new();
        let mut keys: Vec<u32> = Vec::new();
        // soft-token row range of each picture, for the mid-span guard
        let mut img_spans: Vec<(usize, usize)> = Vec::new();
        let mut img_k = 0usize;
        for ch in chunks {
            match ch {
                MmChunk::Text(ids) => {
                    rows.extend(ids.iter().map(|&t| Row::Token(t)));
                    keys.extend_from_slice(ids);
                }
                MmChunk::Image { .. } => {
                    rows.push(Row::Token(beg));
                    keys.push(beg);
                    let first = rows.len();
                    let n = encoded[img_k].n_tokens;
                    rows.extend((0..n).map(|r| Row::Image(img_k, r)));
                    keys.extend((0..n).map(|r| image_key_row(hashes[img_k], r)));
                    img_spans.push((first, first + n));
                    rows.push(Row::Token(end));
                    keys.push(end);
                    img_k += 1;
                }
                MmChunk::Audio { .. } => {
                    return Err(GpuError::Driver(
                        "gemma4 serves images, not audio - routing bug".into(),
                    ));
                }
                MmChunk::OcrCrop(_) => {
                    return Err(GpuError::Driver(
                        "OCR crop directive on gemma4 - routing bug".into(),
                    ));
                }
                MmChunk::VisionPixels { .. } => {
                    return Err(GpuError::Driver(
                        "pixel-budget directive on gemma4 - routing bug".into(),
                    ));
                }
            }
        }
        debug_assert_eq!(rows.len(), keys.len(), "one radix key per prefill row");
        if rows.is_empty() {
            return Err(GpuError::Driver("empty multimodal prompt".into()));
        }
        if rows.len() > self.max_ctx {
            return Err(GpuError::Driver(format!(
                "multimodal prompt is {} rows but max_ctx is {}",
                rows.len(),
                self.max_ctx
            )));
        }

        // image spans (for the non-causal attention-bound override):
        // span_end[i] = the absolute position of image i's last soft token
        let mut span_end = vec![0usize; encoded.len()];
        for (pos, row) in rows.iter().enumerate() {
            if let Row::Image(i, _) = row {
                span_end[*i] = pos;
            }
        }

        // Same admission shape as the text path (batch.rs): clear, try to
        // resume, then grow the table to cover the whole prompt. `start` is a
        // block-aligned row count already resident in KV - 0 on a cold prompt.
        self.gpool_clear_slot(slot);
        let start = self.prefix_resume(slot, &keys)?;
        // `keys` are cache keys (image spans hash into them), not token ids,
        // so the DFlash coverage trim `prefix_resume` just ran compared two
        // different id spaces and its agreement means nothing here. Multimodal
        // resumes stay cold until the walk refills the ring. Rebuilding this
        // properly wants the mm path to record its own key mirror - the same
        // seam, just keyed the way the resume asks about it.
        self.dflash_clear_slot(slot);
        self.ensure_global_rows(&[slot as u32], &[(rows.len() - 1) as u32])?;
        let cut = self.mm_prefix_cut(rows.len(), start, &img_spans);
        let slot_fill = vec![slot as u32; self.pf_rows];
        self.exec
            .stream
            .memcpy_htod(&slot_fill, &mut self.scratch.pf_slots)
            .map_err(|e| GpuError::Driver(e.to_string()))?;

        let n_embd = self.hp.n_embd;
        let row_bytes = self.token_embd.row_bytes(n_embd);
        // resume: rows [0, start) are already in KV, so the tail starts there
        // and `base` stays ABSOLUTE - positions, rope and the non-causal
        // attention bounds are all indexed off the full prompt, not the tail.
        let mut base = start;
        let mut last_len = 0usize;
        for chunk in rows[start..].chunks(self.pf_rows) {
            let r = chunk.len();
            let positions: Vec<u32> = (0..r).map(|i| (base + i) as u32).collect();
            // attention bounds: image rows see through their whole span
            let attn_pos: Vec<u32> = chunk
                .iter()
                .enumerate()
                .map(|(i, row)| match row {
                    Row::Token(_) => (base + i) as u32,
                    Row::Image(img, _) => span_end[*img].max(base + i) as u32,
                })
                .collect();
            {
                let sc = &mut self.scratch;
                self.exec
                    .stream
                    .memcpy_htod(&positions, &mut sc.pf_pos)
                    .map_err(|e| GpuError::Driver(e.to_string()))?;
                self.exec
                    .stream
                    .memcpy_htod(&attn_pos, &mut sc.pf_attn_pos)
                    .map_err(|e| GpuError::Driver(e.to_string()))?;
                // token rows stage in pf_tmp (zeroed -> image slots stay 0),
                // one √embd scale covers them, then image rows overwrite
                self.exec
                    .stream
                    .memset_zeros(&mut sc.pf_tmp)
                    .map_err(|e| GpuError::Driver(e.to_string()))?;
            }
            for (i, row) in chunk.iter().enumerate() {
                if let Row::Token(t) = row {
                    let sc = &mut self.scratch;
                    self.exec.dequant_slice(
                        &self.token_embd,
                        *t as usize * row_bytes,
                        &mut sc.pf_row,
                    )?;
                    self.exec
                        .copy_region(&sc.pf_row, 0, &mut sc.pf_tmp, i * n_embd, n_embd)?;
                }
            }
            {
                let sc = &mut self.scratch;
                self.exec
                    .stream
                    .memset_zeros(&mut sc.pf_x)
                    .map_err(|e| GpuError::Driver(e.to_string()))?;
                self.exec
                    .scale_add(&mut sc.pf_x, &sc.pf_tmp, self.hp.embd_scale(), r * n_embd)?;
            }
            for (i, row) in chunk.iter().enumerate() {
                if let Row::Image(img, ir) = row {
                    let src = &encoded[*img].embd;
                    let sc = &mut self.scratch;
                    self.exec
                        .copy_region(src, ir * n_embd, &mut sc.pf_x, i * n_embd, n_embd)?;
                }
            }
            // After the image splice, deliberately: the reference norms
            // `inpL` once, and mtmd has already substituted the projected
            // image rows into it by then - so image rows get normalized too.
            // (The sqrt scale above is text-only for the opposite reason: the
            // projector's output is not a raw embedding lookup.) No-op on
            // gemma4.
            {
                let sc = &mut self.scratch;
                super::GpuGemma4::embd_preamble(
                    &self.exec,
                    &self.hp,
                    self.embd_ones.as_ref(),
                    &mut sc.pf_x,
                    r,
                )?;
            }
            // image-aware SWA sub-spans: cut every ~swa_span() rows, but never
            // inside an image span - its rows attend NON-CAUSALLY to the
            // span's end, so a mid-image cut would read keys not yet
            // appended. Extension is bounded by the encoder's <=280 soft
            // tokens (< IMG_SPAN_MAX, which the ring absorbs).
            let spans = {
                let mut spans: Vec<(usize, usize)> = Vec::new();
                let mut o = 0usize;
                while o < r {
                    let mut end = (o + self.swa_span).min(r);
                    while end < r {
                        match (&chunk[end - 1], &chunk[end]) {
                            (Row::Image(a, _), Row::Image(b, _)) if a == b => end += 1,
                            _ => break,
                        }
                    }
                    spans.push((o, end - o));
                    o = end;
                }
                spans
            };
            self.prefill_layers(r, &[(0, r)], &spans, 0)?;
            base += r;
            last_len = r;
        }
        let logits = self.logits_from_pf_row(last_len - 1)?;
        self.prefix_insert(slot, &keys, cut)?;
        Ok((logits, rows.len()))
    }

    /// The checkpoint cut for a multimodal prompt: [`Self::prefix_cut`]'s
    /// answer, walked back to a page boundary that is not strictly inside an
    /// image span.
    ///
    /// [`cut_outside_image_spans`] is what keeps gemma4v's non-causal image
    /// rows safe, and qwen35 shares it - the rule and its tests live in
    /// `gpu_model/prefix_cache.rs` so the two families cannot drift on it.
    fn mm_prefix_cut(
        &self,
        n_rows: usize,
        start: usize,
        img_spans: &[(usize, usize)],
    ) -> Option<usize> {
        let cut = cut_outside_image_spans(self.prefix_cut(n_rows, start)?, img_spans);
        (cut > start && cut >= BLOCK_TOKENS).then_some(cut)
    }
}
