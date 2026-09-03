//! Nemotron radix prefix cache -  stage D.
//!
//! The attention half is granite's shape: a [`PagedRadix`] over the 6-layer
//! block pool, a hit ADOPTS blocks by refcount, nothing copies. The mamba
//! half is the qwen35 hybrid precedent: recurrent state is not per-block
//! sharable - a sequence can only resume at a position whose 23-layer
//! (SSM state + conv window) snapshot was CHECKPOINTED, so the resume point
//! is the deepest checkpoint under the block match (`PagedMatch::ckpt`),
//! never the raw match length. Checkpoints land at `ckpt_cuts` - the last
//! two page boundaries of a prompt (qwen35's two-boundary law: a re-rendered
//! next turn diverges inside the trailing generation header, which ~5/16 of
//! the time crosses the last boundary; a checkpoint only there is
//! unreachable and reuse deterministically drops to 0%).
//!
//! Snapshots are STAGED during the pass, never by splitting it: the mamba
//! run walk in `layer_walk` pauses its conv/scan advance at a break row,
//! copies that layer's slot state into a staging blob, and continues - the
//! GEMM passes and the tick structure stay whole (splitting a chunk at a
//! cut would re-stream all 20 GiB of weights per split; qwen35's
//! d_ckpt_stage exists for exactly this reason). After the pass, the blob
//! flat-copies into the checkpoint pool under the radix node.

use crate::gpu::GpuError;
use crate::gpu_model::gpt_oss::GpuModelError;
use crate::kv_pool::BLOCK_TOKENS;

use super::GpuNemotron;

/// Don't resume prefixes shorter than this (granite's floor - the restore
/// here also pays 23 state copies, so trivial prompts aren't worth churn).
pub(super) const MIN_CACHE_PREFIX: usize = 32;

/// Staging blobs available per pass. A pass stages one blob per checkpoint
/// cut that lands inside it; cuts beyond this are skipped (reuse loss only,
/// never an error). 4 covers a full admission wave's trailing cuts.
pub(super) const CKPT_STAGES: usize = 4;

/// Blocks kept in reserve for radix retention when sizing the pool. Cheap
/// here - a nemotron block-set is 96 KiB (6 attention layers), so the 512
/// default is ~48 MB, not granite-30b's 2 GiB.
pub(crate) fn retention_blocks() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        paddock_models::dev_var!("PADDOCK_NEMO_PREFIX_BLOCKS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(512)
    })
}

/// Engine-wide off switch, honoured by every family.
pub(crate) fn prefix_disabled() -> bool {
    paddock_models::dev_var_os!("PADDOCK_NO_PREFIX_CACHE").is_some()
}

/// The checkpoint boundaries for a prompt: its last two full page
/// boundaries, ascending (0 entries collapse when the prompt is short).
/// Keeps at least one token to prefill, matching the radix matcher.
pub(super) fn ckpt_cuts(t_len: usize, step: usize) -> [usize; 2] {
    // `step` = the tier's run span when armed (both boundaries must sit at
    // run granularity or their blobs cannot demote - qwen35's precedent),
    // BLOCK_TOKENS otherwise (the historical behavior, unchanged).
    let step = step.max(BLOCK_TOKENS);
    let b1 = if t_len > 1 {
        (t_len - 1) / BLOCK_TOKENS * BLOCK_TOKENS
    } else {
        0
    };
    let b1 = b1 / step * step;
    [b1.saturating_sub(step), b1]
}

impl GpuNemotron {
    /// f32 elements one checkpoint holds: every mamba layer's SSM state +
    /// conv window, in layer order (the staging blob and the pool share this
    /// layout).
    pub(super) fn state_ckpt_f32(&self) -> usize {
        let hp = &self.hp;
        let state_elems = hp.mamba_heads * hp.mamba_head_dim * hp.d_state;
        let win_elems = (hp.d_conv - 1) * hp.conv_dim();
        let n_mamba = hp
            .blocks
            .iter()
            .filter(|b| matches!(b, paddock_models::nemotron::NemotronBlock::Mamba))
            .count();
        n_mamba * (state_elems + win_elems)
    }

    /// Match `keys` against the radix; on a hit with a reachable checkpoint,
    /// adopt the KV blocks, restore the mamba state snapshot, re-back the
    /// tail, and return the resume position. 0 = cold (admission already
    /// zeroed the arenas). Called after `admit_rows`.
    pub(super) fn prefix_resume_rows(
        &mut self,
        slot: usize,
        keys: &[u32],
        n_rows: usize,
    ) -> Result<usize, GpuModelError> {
        self.last_reused[slot] = 0;
        let m = {
            let bs = self.batch.as_mut().expect("batch enabled");
            let Some(radix) = bs.prefix.as_mut() else {
                return Ok(0);
            };
            let mut m = radix.match_full(keys);
            // TIER (D5 park/wake): the restore is consulted and PARKED at
            // admission (`tier_prefix_loading`); an elected restore has
            // already published + attached by the time prefill runs. Pump
            // for freshness and re-match, so paths that skip the consult
            // still pick up published prefixes.
            if m.ckpt.is_none()
                && let Some(tier) = bs.tier.as_mut()
            {
                tier.pump_completions(radix, &mut bs.pool);
                m = radix.match_full(keys);
            }
            m
        };
        // hybrid law: resume only where state was snapshotted - the deepest
        // checkpoint under the match, never the raw block-match length
        let Some((pos, idx)) = m.ckpt else {
            return Ok(0);
        };
        if pos < MIN_CACHE_PREFIX || pos >= keys.len() {
            return Ok(0);
        }
        {
            let bs = self.batch.as_mut().expect("batch enabled");
            // release the admission's fresh backing, adopt the shared blocks
            bs.tables[slot].clear(&mut bs.pool);
            bs.tables[slot].share_prefix(&m.blocks[..pos / BLOCK_TOKENS], &mut bs.pool);
            let base = slot * bs.bps;
            for (j, &b) in bs.tables[slot].blocks().iter().enumerate() {
                bs.bt_host[base + j] = b;
            }
            self.exec
                .stream
                .memcpy_htod(&bs.bt_host, &mut bs.d_bt)
                .map_err(|e| GpuError::Driver(e.to_string()))?;
        }
        self.restore_state(slot, idx)?;
        // re-back the tail the prompt will still write
        self.ensure_rows(&[slot as u32], &[(n_rows - 1) as u32])?;
        self.last_reused[slot] = pos;
        // drafter coverage trims to the resume point (gemma4's trim-not-
        // clear: rows below the resume still describe the same tokens; a
        // clear here made every prefix hit a cold drafter). The MTP twin
        // also zeroes its pending_h - the chain vector belonged to the old
        // end (see mtp_trim_slot).
        self.dflash_trim_slot(slot, pos);
        self.mtp_trim_slot(slot, pos)?;
        Ok(pos)
    }

    /// Publish a finished prompt's full pages into the radix (idempotent for
    /// pages the checkpoint commits already inserted), then evict down to a
    /// free margin so the next admission does not pay for it.
    pub(super) fn prefix_insert(&mut self, slot: usize, keys: &[u32]) {
        let bs = self.batch.as_mut().expect("batch enabled");
        let Some(radix) = bs.prefix.as_mut() else {
            return;
        };
        let blocks = bs.tables[slot].blocks().to_vec();
        radix.insert(keys, &blocks, &mut bs.pool);
        let margin =
            crate::gpu_model::prefix_cache::evict_ahead_margin(256, bs.pool.capacity() as usize);
        if margin > 0 && bs.pool.free_blocks() < margin {
            match (bs.tier.as_mut(), bs.prefix.as_mut()) {
                (Some(tier), Some(radix)) => {
                    // tier-aware evict-ahead: closing runs AND their mamba
                    // checkpoint blobs demote before eviction - a plain
                    // evict_lru here discarded the blobs and left the tier
                    // restore-blind (probe hit, aux None, every repeat
                    // recomputed)
                    let exec = self.exec.clone();
                    let state_bytes = (bs.state_ckpt_f32 * 4) as u64;
                    let state = bs.d_state_pool.as_ref().map(|sp| {
                        use cudarc::driver::DevicePtr;
                        let (pp, _g) = sp.device_ptr(&exec.stream);
                        (pp, state_bytes)
                    });
                    tier.press(radix, &mut bs.pool, margin, state, &mut || {
                        exec.record_event().ok()
                    });
                    tier.pump_completions(radix, &mut bs.pool);
                }
                (None, Some(radix)) => {
                    while bs.pool.free_blocks() < margin {
                        if radix.evict_lru(&mut bs.pool).is_none() {
                            break;
                        }
                    }
                }
                _ => {}
            }
        }
    }

    /// Commit one staged checkpoint after its pass: insert the pages up to
    /// `cut`, attach a state index under the radix node, and flat-copy the
    /// staging blob into the pool at that index. No-ops (reuse loss only)
    /// when the cache is off or the node can't take a checkpoint.
    pub(super) fn commit_stage(&mut self, stage: usize, slot: usize, keys: &[u32], cut: usize) {
        let n = self.state_ckpt_f32();
        let exec = self.exec.clone();
        let bs = self.batch.as_mut().expect("batch enabled");
        let Some(radix) = bs.prefix.as_mut() else {
            return;
        };
        let blocks: Vec<u32> = match bs.tables[slot].blocks().get(..cut / BLOCK_TOKENS) {
            Some(b) => b.to_vec(),
            None => return,
        };
        radix.insert(&keys[..cut], &blocks, &mut bs.pool);
        let Some(idx) = radix.attach_state(keys, cut) else {
            return;
        };
        let Some(sp) = bs.d_state_pool.as_mut() else {
            return;
        };
        if let Err(e) = exec.copy_region(&bs.d_ckpt_stage[stage], 0, sp, idx as usize * n, n) {
            tracing::warn!("nemotron ckpt commit failed (stage {stage}): {e}");
        }
    }

    /// Restore `slot`'s mamba state from checkpoint `idx` - the reverse of
    /// the staged snapshot: pool blob -> each mamba layer's slot arena
    /// windows, in the same layer order the blob was written in.
    fn restore_state(&mut self, slot: usize, idx: u32) -> Result<(), GpuModelError> {
        let exec = self.exec.clone();
        let hp = self.hp.clone();
        let state_elems = hp.mamba_heads * hp.mamba_head_dim * hp.d_state;
        let win_elems = (hp.d_conv - 1) * hp.conv_dim();
        let n = self.state_ckpt_f32();
        let bs = self.batch.as_mut().expect("batch enabled");
        let Some(sp) = bs.d_state_pool.as_ref() else {
            return Err(GpuModelError::Unsupported(
                "resume without a state pool".into(),
            ));
        };
        let mut boff = idx as usize * n;
        for li in 0..hp.n_layer {
            let Some(s) = bs.ssm[li].as_mut() else {
                continue;
            };
            s.restore_from_blob(&exec, sp, boff, slot * state_elems, state_elems)?;
            boff += state_elems;
            let w = bs.conv_win[li].as_mut().expect("mamba layer has window");
            exec.copy_region(sp, boff, w, slot * win_elems, win_elems)?;
            boff += win_elems;
        }
        Ok(())
    }

    /// Blocks the radix could give back - added to the free count for
    /// admission accounting (the cache is reclaimable capacity, not a
    /// reservation - the gemma4 c8 lesson).
    /// The D5 admission consult (park/wake): probe + elect the hybrid
    /// two-round restore and, when elected, START it and PARK the request -
    /// qwen35's recipe verbatim with the mamba state blob in place of the
    /// DeltaNet one. `true` = skip this slot this tick; the per-pass
    /// `tier_pump` drives the flow and the wake re-enters admission.
    pub(crate) fn tier_consult_impl(&mut self, slot: usize, tokens: &[u32]) -> bool {
        use crate::kv_tier::{Election, FlowStatus};
        use cudarc::driver::DevicePtr;
        let exec = self.exec.clone();
        let Some(bs) = self.batch.as_mut() else {
            return false;
        };
        let state_bytes = (bs.state_ckpt_f32 * 4) as u64;
        let (Some(tier), Some(pr)) = (bs.tier.as_mut(), bs.prefix.as_mut()) else {
            return false;
        };
        let Some(sp) = bs.d_state_pool.as_ref() else {
            return false;
        };
        tier.pump_completions(pr, &mut bs.pool);
        {
            let exec2 = exec.clone();
            tier.pump_flows(pr, &mut || exec2.record_event().ok());
        }
        match tier.flow_status(slot, tokens) {
            FlowStatus::Loading => return true,
            FlowStatus::Done { .. } => return false,
            FlowStatus::None => {}
        }
        // a resident usable checkpoint makes the tier moot for this prompt
        if pr.match_full(tokens).ckpt.is_some() {
            return false;
        }
        let hit = tier.probe(tokens, 0);
        let afford = |pool: &crate::kv_pool::KvPool, r: usize| {
            pool.free_blocks().saturating_sub(2 * r) / r * r
        };
        let r = tier.run_blocks();
        let mut afford_blocks = afford(&bs.pool, r);
        let deepest = hit
            .as_ref()
            .and_then(|h| tier.probe_aux(tokens, h.end_block))
            .filter(|a| a.end_block * BLOCK_TOKENS >= MIN_CACHE_PREFIX && a.end_block % r == 0);
        if let Some(a) = &deepest
            && afford_blocks < a.end_block
        {
            // retention crowds the destination: pressure-demote it (the
            // prefix cache is reclaimable capacity)
            let want = a.end_block + 2 * r;
            let after = exec.record_event().ok();
            let (_e, taken) = tier.pressure_demote(pr, &mut bs.pool, want, after);
            let (cp, _g) = sp.device_ptr(&exec.stream);
            for t in taken {
                if t.end_block % r == 0 {
                    let blob = cp + t.state_idx as u64 * state_bytes;
                    let ev = exec.record_event().ok();
                    tier.demote_aux(pr, t, blob, state_bytes, ev);
                } else {
                    pr.recycle_state(t.state_idx);
                }
            }
            let deadline = std::time::Instant::now() + std::time::Duration::from_millis(50);
            while bs.pool.free_blocks() < want && tier.stats().2 > 0 {
                tier.pump_completions(pr, &mut bs.pool);
                if std::time::Instant::now() >= deadline {
                    break;
                }
                std::thread::sleep(std::time::Duration::from_micros(200));
            }
            afford_blocks = afford(&bs.pool, r);
        }
        let aux = deepest.filter(|a| a.end_block <= afford_blocks);
        tracing::debug!(
            free = bs.pool.free_blocks(),
            afford_blocks,
            hit_end = hit.as_ref().map(|h| h.end_block),
            aux_end = aux.as_ref().map(|a| a.end_block),
            "nemotron tier gate"
        );
        let (Some(hit), Some(aux)) = (hit, aux) else {
            return false;
        };
        let n_runs = aux.end_block / r;
        let per_run = hit.bytes / hit.keys.len().max(1) as u64;
        let hit = crate::kv_tier::TierHit {
            start_block: 0,
            end_block: aux.end_block,
            bytes: per_run * n_runs as u64,
            keys: hit.keys[..n_runs.min(hit.keys.len())].to_vec(),
            // runs are equal-sized, so the truncated hit keeps the same
            // share of disk-sourced bytes as the full one
            nvme_bytes: hit.nvme_bytes.min(per_run * n_runs as u64),
        };
        let shape = crate::kv_tier::HitShape {
            restore_bytes: hit.bytes + aux.bytes,
            restore_tokens: (aux.end_block * BLOCK_TOKENS) as u32,
            queued_bytes: tier.catalog.ledger(crate::kv_tier::Tier::Ram).in_flight,
            nvme_bytes: hit.nvme_bytes,
        };
        let Election::Restore { est_us, .. } = tier.cost.elect(shape) else {
            return false;
        };
        let after = exec.record_event().ok();
        let (cp, _g) = sp.device_ptr(&exec.stream);
        let plan = crate::kv_tier::AuxPlan {
            hit: aux,
            state_base: cp,
            state_stride: state_bytes,
        };
        match crate::kv_tier::RestoreFlow::begin(
            tier,
            &mut bs.pool,
            tokens,
            &hit,
            Some(plan),
            est_us,
            after,
        ) {
            Some(flow) => {
                tier.park_flow(slot, flow);
                tracing::debug!(
                    slot,
                    boundary = hit.end_block,
                    "nemotron tier: restore parked (D5)"
                );
                true
            }
            None => false,
        }
    }

    /// The per-tick tier pump (see `Generator::tier_pump`).
    pub(crate) fn tier_pump_impl(&mut self) {
        let exec = self.exec.clone();
        let Some(bs) = self.batch.as_mut() else {
            return;
        };
        let (Some(tier), Some(pr)) = (bs.tier.as_mut(), bs.prefix.as_mut()) else {
            return;
        };
        tier.pump_completions(pr, &mut bs.pool);
        tier.pump_flows(pr, &mut || exec.record_event().ok());
        // 2.3 write-through: retained chains AND live state blobs
        // pre-store in slack so eviction (and ckpt-slot recycling) is free
        let state = bs.d_state_pool.as_ref().map(|sp| {
            use cudarc::driver::DevicePtr;
            let (cp, _g) = sp.device_ptr(&exec.stream);
            (cp, (bs.state_ckpt_f32 * 4) as u64)
        });
        tier.mirror_slack(pr, &mut bs.pool, exec.record_event().ok(), 2, state);
    }

    pub(crate) fn tier_stats_impl(&self) -> Option<crate::kv_tier::TierStats> {
        self.batch.as_ref()?.tier.as_ref().map(|t| t.tier_stats())
    }

    /// The checkpoint step: the tier's run span when armed (boundaries must
    /// sit at run granularity to demote), BLOCK_TOKENS otherwise.
    pub(crate) fn tier_ckpt_step(&self) -> usize {
        self.batch
            .as_ref()
            .and_then(|b| b.tier.as_ref())
            .map(|t| t.run_blocks() * BLOCK_TOKENS)
            .unwrap_or(BLOCK_TOKENS)
    }

    pub(crate) fn prefix_evictable(&self) -> usize {
        self.batch
            .as_ref()
            .and_then(|bs| bs.prefix.as_ref().map(|r| r.evictable_blocks(&bs.pool)))
            .unwrap_or(0)
    }
}
