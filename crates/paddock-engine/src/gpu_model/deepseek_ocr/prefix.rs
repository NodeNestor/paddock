//! DeepSeek-OCR prefix cache: a [`PagedRadix`] over the one full-attention
//! block pool - granite's easy case (no recurrent state, no checkpoints, no
//! copy-on-write), plus the R-SWA argument that lets it stay on at all:
//!
//! Only whole 16-token blocks are ever adopted or inserted, so the block
//! containing `prefill_len` is always slot-private (a match covering it would
//! need more rows than the prompt has, and `insert` stops at
//! `tokens.len()/16`). Ring rows live at `[pf, pf+W)` - the boundary block
//! plus fresh blocks - so a steady-state overwrite can never touch a shared
//! block, and prefill rows are never rewritten. The cacheable region and the
//! reference's globally-visible prefix are the same region, by construction.
//!
//! Keys are the row tokens for now. When the multimodal splice lands, image
//! rows switch to content-derived keys via
//! [`crate::gpu_model::prefix_cache::image_key_row`] (granite's scheme) -
//! that is what makes "same page, many questions" resume past the whole
//! picture, which on this family is most of the prompt.

use crate::gpu::GpuError;
use crate::gpu_model::gpt_oss::GpuModelError;
use crate::kv_pool::BLOCK_TOKENS;

use super::load::GpuDeepseekOcr;

/// Don't resume prefixes shorter than this (radix churn floor).
const MIN_CACHE_PREFIX: usize = 32;

/// Blocks kept in reserve for radix retention when sizing the pool. The
/// 12-layer hd128 shape puts one block-set at ~1.5 MiB f16, so the default
/// 512 costs ~0.8 GiB. `PADDOCK_OCR_PREFIX_BLOCKS` overrides; 0 disables.
pub(crate) fn retention_blocks() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        paddock_models::dev_var!("PADDOCK_OCR_PREFIX_BLOCKS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(512)
    })
}

/// Free blocks the insert path evicts down to.
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

impl GpuDeepseekOcr {
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
    /// resume right now, touching nothing but the radix LRU. The pipelined
    /// admission wave uses it to decide which images to host-preprocess -
    /// it is a conservative LOWER bound on the execute-time resume (same-wave
    /// inserts can only lengthen the match), so it may preprocess too much
    /// but never too little; eviction between probe and resume shrinking the
    /// match is caught by the wave's inline-prep fallback.
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
    /// Called at prefill completion - Before any ring row exists, so what is
    /// published is exactly the globally-visible prefix and nothing else.
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
