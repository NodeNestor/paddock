//! PaddleOCR-VL prefix cache: a [`PagedRadix`] over the full-attention block
//! pool - the deepseek-ocr shape minus the R-SWA argument (this decoder is
//! plain causal, so the whole prompt is an ordinary shareable prefix and
//! decode rows append past it without ever rewriting a prefill row).
//!
//! Keys are row tokens for text and content-derived for image rows
//! ([`crate::gpu_model::prefix_cache::image_key_row`] over the picture's
//! byte hash with the RESOLVED grid folded in - see multimodal.rs for why
//! the grid fold is load-bearing on this family: the per-request pixel
//! budget can change the grid, and with it every row, of the same bytes).
//!
//! M-RoPE and resume: a resumed prefix's rows carry positions derived purely
//! from ids + grids, both of which the key stream pins (same keys => same
//! text tokens, same image content, same grid => same `build_positions`
//! output), so adopted KV was roped exactly as the recompute would rope it.

use crate::gpu::GpuError;
use crate::gpu_model::gpt_oss::GpuModelError;
use crate::kv_pool::BLOCK_TOKENS;

use super::load::GpuPaddleOcrVl;

/// Don't resume prefixes shorter than this (radix churn floor).
const MIN_CACHE_PREFIX: usize = 32;

/// Blocks kept in reserve for radix retention when sizing the pool. The
/// 18-layer kv_dim-256 shape puts one block-set at ~0.28 MiB f16, so the
/// shared OCR default of 512 costs ~0.14 GiB. `PADDOCK_OCR_PREFIX_BLOCKS`
/// overrides (the knob is shared with the deepseek-ocr lane deliberately -
/// one OCR retention policy per box); 0 disables.
pub(crate) fn retention_blocks() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        paddock_models::dev_var!("PADDOCK_OCR_PREFIX_BLOCKS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(512)
    })
}

/// Free blocks the insert path evicts down to (`PADDOCK_OCR_PX_MARGIN`,
/// shared with deepseek-ocr like the retention knob).
fn px_margin() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        paddock_models::dev_var!("PADDOCK_OCR_PX_MARGIN")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(256)
    })
}

/// Engine-wide off switch, honoured by every family.
pub(crate) fn prefix_disabled() -> bool {
    paddock_models::dev_var_os!("PADDOCK_NO_PREFIX_CACHE").is_some()
}

impl GpuPaddleOcrVl {
    /// Match `keys` against the radix and adopt the matched blocks into
    /// `slot`'s table. Returns the block-aligned resume position.
    pub(crate) fn prefix_resume(
        &mut self,
        slot: usize,
        keys: &[u32],
    ) -> Result<usize, GpuModelError> {
        let bs = self.batch.as_mut().expect("batch enabled");
        let Some(radix) = bs.prefix.as_mut() else {
            return Ok(0);
        };
        let blocks = radix.match_prefix(keys);
        let pos = blocks.len() * BLOCK_TOKENS;
        if pos < MIN_CACHE_PREFIX || pos >= keys.len() {
            return Ok(0);
        }
        bs.tables[slot].clear(&mut bs.pool);
        bs.tables[slot].share_prefix(&blocks, &mut bs.pool);
        let base = slot * bs.bps;
        for (j, &b) in bs.tables[slot].blocks().iter().enumerate() {
            bs.bt_host[base + j] = b;
        }
        self.exec
            .stream
            .memcpy_htod(&bs.bt_host, &mut bs.d_bt)
            .map_err(|e| GpuError::Driver(e.to_string()))?;
        Ok(pos)
    }

    /// Non-adopting twin of [`Self::prefix_resume`]: how far `keys` would
    /// resume right now, touching nothing but the radix LRU. Admission uses
    /// it to decide which images to preprocess/encode - a conservative LOWER
    /// bound on the execute-time resume (inserts between probe and resume
    /// only lengthen the match; eviction shrinking it is caught by the
    /// inline-prep/re-encode fallbacks).
    pub(crate) fn prefix_probe(&mut self, keys: &[u32]) -> usize {
        let bs = self.batch.as_mut().expect("batch enabled");
        let Some(radix) = bs.prefix.as_mut() else {
            return 0;
        };
        let pos = radix.match_prefix(keys).len() * BLOCK_TOKENS;
        if pos < MIN_CACHE_PREFIX || pos >= keys.len() {
            0
        } else {
            pos
        }
    }

    /// Publish a finished prompt's blocks, then evict down to the margin.
    pub(crate) fn prefix_insert(&mut self, slot: usize, keys: &[u32]) {
        let bs = self.batch.as_mut().expect("batch enabled");
        let Some(radix) = bs.prefix.as_mut() else {
            return;
        };
        let blocks = bs.tables[slot].blocks().to_vec();
        radix.insert(keys, &blocks, &mut bs.pool);
        let margin = crate::gpu_model::prefix_cache::evict_ahead_margin(
            px_margin(),
            bs.pool.capacity() as usize,
        );
        while margin > 0 && bs.pool.free_blocks() < margin {
            let Some(radix) = bs.prefix.as_mut() else {
                break;
            };
            if radix.evict_lru(&mut bs.pool).is_none() {
                break;
            }
        }
    }

    /// Blocks the radix holds that could be reclaimed - reclaimable capacity,
    /// not a reservation, for admission accounting.
    pub(crate) fn prefix_evictable(&self) -> usize {
        self.batch
            .as_ref()
            .and_then(|bs| bs.prefix.as_ref().map(|r| r.evictable_blocks(&bs.pool)))
            .unwrap_or(0)
    }

    /// Rows the last `prefix_resume` adopted into `slot`, consumed once -
    /// `Generator::take_prefill_reused`, the usage report's `cached` field.
    pub fn take_prefill_reused(&mut self, slot: usize) -> usize {
        self.batch
            .as_mut()
            .and_then(|bs| bs.last_reused.get_mut(slot))
            .map_or(0, std::mem::take)
    }
}
