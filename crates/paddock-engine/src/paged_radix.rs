//! Zero-copy radix prefix cache over the paged `KvPool` (P5c).
//!
//! **This module is the CPU-side bookkeeping only** (the same discipline as
//! `kv_pool` / P1): it maps block-aligned token prefixes to `KvPool` block ids so
//! a new sequence that shares a prefix ADOPTS the cached blocks (refcount++, via
//! [`crate::kv_pool::BlockTable::share_prefix`]) instead of recomputing or copying
//! them. It owns no device memory and touches no kernel - the blocks it names are
//! the same `KvPool` blocks the slots write, which is the whole point of
//! zero-copy sharing (vs. `prefix_cache::RadixKvCache`, which keeps its own store
//! and COPIES).
//!
//! Refcount lifecycle (all through `KvPool`):
//! - `insert` RETAINS each new cached block -> the tree holds one reference.
//! - a sharing slot RETAINS via `BlockTable::share_prefix` -> +1 per slot.
//! - a slot finishing RELEASES via `BlockTable::clear` (free-on-completion, P5b).
//! - [`PagedRadix::evict_lru`] RELEASES the tree's reference on the LRU leaf.
//! - a block returns to the free-list only at refcount 0 (no tree node, no slot).
//!
//! Only **full 16-token blocks** are cached: a block-aligned prefix means the
//! adopting slot never writes into a shared block (its own writes start at the
//! next block boundary), so **no copy-on-write is needed** for this path. The
//! token-granular partial-tail reuse (needs `BlockTable::cow_at`) and the DeltaNet
//! recurrent-state checkpoints (hybrid resume) are the device-wiring follow-up
//! (P5c-P2); this module is the allocator/tree bookkeeping they build on.

use crate::kv_pool::{BLOCK_TOKENS, BlockId, KvPool};
use crate::kv_tier::LogicalKey;
use std::collections::HashMap;

/// FNV-1a over a block's tokens - the child key under its parent. (Was
/// mirrored from the dense cache's hash_block; that cache is gone and this
/// is the one definition of block identity now.) Exact tokens are
/// also compared on match, so a collision costs a miss, never a wrong reuse.
fn hash_block(tokens: &[u32]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for &t in tokens {
        h ^= t as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

struct Node {
    parent: u32,
    /// the `KvPool` block this node caches (retained while the node is alive).
    block: BlockId,
    /// hash of this node's 16 tokens (its key in `parent.children`).
    key: u64,
    /// the exact tokens, for the hash-collision check on match.
    tokens: Vec<u32>,
    children: HashMap<u64, u32>,
    last_used: u64,
    alive: bool,
    /// This prefix has been MATCHED again after being cached, i.e. it recurs.
    ///
    /// The signal is the page match, deliberately not "a checkpoint was handed
    /// to a resume". Keying on successful resumes is chicken-and-egg: a prefix
    /// can only prove itself by hitting, and thrash is precisely what stops it
    /// hitting. Measured on a c32 leg with the resume-keyed
    /// version: 158 requests per leg matched 176 tokens of pages and found
    /// `ckpt None` - exactly the prefixes that deserved protection, and every
    /// one of them invisible to the flag, so admission control never fired once.
    ///
    /// Survives the checkpoint being stolen: the proof is about the PREFIX, not
    /// the state blob. See `protect_proven`.
    recurred: bool,
    /// P5c hybrid resume: index of the DeltaNet recurrent-state checkpoint for
    /// the prefix ending at this node (a block-boundary position), or `None`.
    /// The device state blob lives in the model's paged state pool at this index;
    /// this is CPU-side bookkeeping only.
    state_blk: Option<u32>,
    /// KV tier content-chain key for the prefix ending at this node
    /// `parent.tkey.child(tokens)`, rooted in the cache
    /// namespace via [`PagedRadix::set_tier_root`]. `None` when the tier is
    /// off or the node predates arming - such nodes simply never demote.
    tkey: Option<LogicalKey>,
}

/// The prefix-cache hit for a prompt: the shared KV block ids to adopt, plus the
/// deepest DeltaNet state checkpoint along the matched path (its block-boundary
/// position and the state-pool index) for a hybrid resume.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct PagedMatch {
    pub blocks: Vec<BlockId>,
    pub ckpt: Option<(usize, u32)>,
}

/// One block on the LRU leaf's root-to-leaf path (see
/// [`PagedRadix::lru_leaf_path`]). `depth` is 1-based: the prefix ending at
/// this block spans `depth` full 16-token blocks.
#[derive(Debug, Clone, Copy)]
pub struct LruPathEntry {
    pub node: u32,
    pub depth: usize,
    pub block: BlockId,
    pub tkey: Option<LogicalKey>,
    /// State-checkpoint index at this boundary (hybrid families) - what the
    /// tier's demote arm captures before eviction would recycle it.
    pub state_blk: Option<u32>,
}

/// A radix tree of block-aligned token prefixes over the shared `KvPool`. Node 0
/// is the dummy root (no block).
pub struct PagedRadix {
    nodes: Vec<Node>,
    free_nodes: Vec<u32>,
    clock: u64,
    /// CPU free-list of DeltaNet state-checkpoint indices (into the model's paged
    /// state pool). Empty until `set_state_capacity` - text-only / non-hybrid
    /// models never checkpoint state.
    state_free: Vec<u32>,
    /// Admission control on the state pool: once a checkpoint belongs to a
    /// prefix that has recurred, stop stealing it - let the pool fill and hold.
    ///
    /// Plain LRU steal is pathological exactly where this cache lives. When the
    /// distinct-prefix working set is a little LARGER than the pool and requests
    /// cycle through it, every arrival evicts the entry that would have been the
    /// next hit - the classic cyclic-thrash degenerate case, hit rate ~0 rather
    /// than the ~capacity/working-set you would expect. Measured on the qwen3.8
    /// c32 leg: 128 distinct prefixes, 88 checkpoint slots,
    /// **0 usable resumes out of 224 requests** on the first leg, while 180
    /// later requests matched 176 tokens of KV pages and found `ckpt None` -
    /// their state had been stolen before they came back. On a hybrid model
    /// matched pages without the recurrent state are worthless: those requests
    /// re-prefill anyway, having paid the match and the adopt.
    ///
    /// Holding the recurring set makes it stick, so the cache serves a stable
    /// `capacity/working-set` fraction instead of churning every entry out just
    /// before its next hit; the refused admissions also skip their state
    /// snapshot, which is the dominant write cost (~170 MiB per checkpoint at
    /// 27B - 48 GDN layers of state + conv window).
    ///
    /// The trade is adaptivity: a resident set that has all recurred never
    /// yields to a newly hot prefix. That is the right trade only where the
    /// pool is smaller than the working set, which is why this is opt-in -
    /// qwen35 arms it from `PADDOCK_CKPT_PROTECT`. Sizing the pool to the
    /// working set is the better fix where the memory exists.
    protect_proven: bool,
    /// Counters for the admission policy, read by the engine's prefix-stats
    /// witness: (state writes, LRU steals, refused admissions).
    st_writes: u64,
    st_steals: u64,
    st_refused: u64,
}

impl Default for PagedRadix {
    fn default() -> Self {
        Self::new()
    }
}

impl PagedRadix {
    pub fn new() -> Self {
        Self {
            nodes: vec![Node {
                parent: 0,
                block: 0,
                key: 0,
                tokens: Vec::new(),
                children: HashMap::new(),
                last_used: 0,
                alive: true,
                recurred: false,
                state_blk: None,
                tkey: None,
            }],
            free_nodes: Vec::new(),
            clock: 0,
            state_free: Vec::new(),
            protect_proven: false,
            st_writes: 0,
            st_steals: 0,
            st_refused: 0,
        }
    }

    /// Arm state-pool admission control (see `protect_proven`). Call at pool
    /// setup, next to `set_state_capacity`.
    pub fn set_protect_proven(&mut self, on: bool) {
        self.protect_proven = on;
    }

    /// (state writes, LRU steals, refused admissions) since boot.
    pub fn state_stats(&self) -> (u64, u64, u64) {
        (self.st_writes, self.st_steals, self.st_refused)
    }

    /// Enable DeltaNet state checkpoints (hybrid models): `n` state-pool indices
    /// become available for `attach_state`. Idempotent-ish - resets the free-list
    /// to `0..n` (call once at pool setup, alongside the device state pool alloc).
    pub fn set_state_capacity(&mut self, n: u32) {
        self.state_free = (0..n).rev().collect();
    }

    fn tick(&mut self) -> u64 {
        self.clock += 1;
        self.clock
    }

    /// The block ids of the longest block-aligned prefix of `tokens` already
    /// cached (one per 16-token block, in order). Bumps LRU on the matched path.
    /// Keeps at least one token unmatched to prefill, so a prompt equal to a
    /// cached sequence still has a token to run.
    pub fn match_prefix(&mut self, tokens: &[u32]) -> Vec<BlockId> {
        let cap = tokens.len().saturating_sub(1);
        let full = cap / BLOCK_TOKENS;
        let mut node = 0u32;
        let mut blocks = Vec::new();
        for bi in 0..full {
            let chunk = &tokens[bi * BLOCK_TOKENS..(bi + 1) * BLOCK_TOKENS];
            let h = hash_block(chunk);
            let Some(&child) = self.nodes[node as usize].children.get(&h) else {
                break;
            };
            if self.nodes[child as usize].tokens != chunk {
                break; // hash collision - treat as a miss
            }
            blocks.push(self.nodes[child as usize].block);
            let t = self.tick();
            self.nodes[child as usize].last_used = t;
            node = child;
        }
        blocks
    }

    /// The full prefix-cache hit for `tokens`: the shared KV block ids AND the
    /// deepest DeltaNet state checkpoint along the matched path. A hybrid resume
    /// needs both - the KV blocks to adopt and the recurrent state at that
    /// block-boundary position. LRU-bumped like `match_prefix`.
    pub fn match_full(&mut self, tokens: &[u32]) -> PagedMatch {
        let cap = tokens.len().saturating_sub(1);
        let full = cap / BLOCK_TOKENS;
        let mut node = 0u32;
        let mut blocks = Vec::new();
        let mut ckpt = None;
        for bi in 0..full {
            let chunk = &tokens[bi * BLOCK_TOKENS..(bi + 1) * BLOCK_TOKENS];
            let h = hash_block(chunk);
            let Some(&child) = self.nodes[node as usize].children.get(&h) else {
                break;
            };
            if self.nodes[child as usize].tokens != chunk {
                break;
            }
            blocks.push(self.nodes[child as usize].block);
            if let Some(sb) = self.nodes[child as usize].state_blk {
                ckpt = Some(((bi + 1) * BLOCK_TOKENS, sb));
            }
            // Reaching this node at all means the prefix came back - mark it
            // whether or not its state survived. See `Node::recurred`.
            self.nodes[child as usize].recurred = true;
            let t = self.tick();
            self.nodes[child as usize].last_used = t;
            node = child;
        }
        PagedMatch { blocks, ckpt }
    }

    /// Claim a free state-pool index without attaching it - the tier's aux
    /// restore lands the blob first and attaches after (an attach-then-fill
    /// order would leave a garbage checkpoint visible on failure). Same
    /// steal/protect policy as `attach_state`. Undo with
    /// [`Self::recycle_state`].
    pub fn reserve_state_slot(&mut self) -> Option<u32> {
        if self.state_free.is_empty() && self.count_state() == 0 {
            return None; // state capacity never enabled
        }
        self.alloc_state()
    }

    /// Attach a RESERVED state index to the cached node ending at block
    /// boundary `pos`. False - and the caller recycles the index - if the
    /// node is missing (evicted between publication and now) or already
    /// checkpointed.
    pub fn attach_state_at(&mut self, tokens: &[u32], pos: usize, idx: u32) -> bool {
        let want = pos / BLOCK_TOKENS;
        if want == 0 || !pos.is_multiple_of(BLOCK_TOKENS) {
            return false;
        }
        let mut node = 0u32;
        for bi in 0..want {
            let chunk = &tokens[bi * BLOCK_TOKENS..(bi + 1) * BLOCK_TOKENS];
            let h = hash_block(chunk);
            let Some(&child) = self.nodes[node as usize].children.get(&h) else {
                return false;
            };
            if self.nodes[child as usize].tokens != chunk {
                return false;
            }
            node = child;
        }
        if node == 0 || self.nodes[node as usize].state_blk.is_some() {
            return false;
        }
        self.nodes[node as usize].state_blk = Some(idx);
        true
    }

    /// Attach a DeltaNet state checkpoint to the cached node ending at block-
    /// boundary `pos` (a `BLOCK_TOKENS` multiple), returning the state-pool index
    /// for the model to write the state blob into. `None` if `pos` isn't a cached
    /// node, already has a checkpoint, or state capacity is off. Steals the LRU
    /// node's checkpoint when the state pool is exhausted (that node + its KV page
    /// stay - only its checkpoint moves).
    pub fn attach_state(&mut self, tokens: &[u32], pos: usize) -> Option<u32> {
        if self.state_free.is_empty() && self.count_state() == 0 {
            return None; // state capacity never enabled
        }
        let want = pos / BLOCK_TOKENS;
        if want == 0 || !pos.is_multiple_of(BLOCK_TOKENS) {
            return None;
        }
        // walk to the node at depth `want`
        let mut node = 0u32;
        for bi in 0..want {
            let chunk = &tokens[bi * BLOCK_TOKENS..(bi + 1) * BLOCK_TOKENS];
            let h = hash_block(chunk);
            let child = *self.nodes[node as usize].children.get(&h)?;
            if self.nodes[child as usize].tokens != chunk {
                return None;
            }
            node = child;
        }
        if node == 0 || self.nodes[node as usize].state_blk.is_some() {
            return None;
        }
        let sb = self.alloc_state()?;
        self.nodes[node as usize].state_blk = Some(sb);
        Some(sb)
    }

    fn count_state(&self) -> usize {
        self.nodes.iter().filter(|n| n.state_blk.is_some()).count()
    }

    /// A free state-pool index, stealing the LRU checkpointed node's if exhausted
    /// (that node + its KV page survive; only the checkpoint is reclaimed).
    ///
    /// Under `protect_proven`, victims are ordered never-recurred first and a
    /// recurred checkpoint is not stolen at all - the pool fills, then holds.
    /// A refused admission also skips the caller's state snapshot, which is the
    /// dominant write cost, so refusing is cheaper than serving.
    fn alloc_state(&mut self) -> Option<u32> {
        if let Some(b) = self.state_free.pop() {
            self.st_writes += 1;
            return Some(b);
        }
        let victim = self
            .nodes
            .iter()
            .enumerate()
            .filter(|(i, n)| *i != 0 && n.alive && n.state_blk.is_some())
            .min_by_key(|(_, n)| (n.recurred, n.last_used))
            .map(|(i, _)| i)?;
        if self.protect_proven && self.nodes[victim].recurred {
            // Every resident checkpoint belongs to a prefix that came back, so
            // stealing one only moves the miss around - the cyclic-thrash
            // trade LRU makes by default. Hold the resident set instead.
            self.st_refused += 1;
            return None;
        }
        self.st_writes += 1;
        self.st_steals += 1;
        self.nodes[victim].state_blk.take()
    }

    /// Cache `tokens`' full 16-token blocks from a slot's `blocks` (logical block
    /// `i` backs tokens `[i*16, i*16+16)`). A block not already in the tree gets a
    /// node and is RETAINED in `pool` (the tree's reference); a block already
    /// cached just bumps LRU (its node keeps the earlier block - the caller's
    /// duplicate can be released by the slot as usual). `blocks.len()` must cover
    /// `tokens.len()/16` full blocks.
    pub fn insert(&mut self, tokens: &[u32], blocks: &[BlockId], pool: &mut KvPool) {
        let full = (tokens.len() / BLOCK_TOKENS).min(blocks.len());
        let mut node = 0u32;
        for bi in 0..full {
            let chunk = &tokens[bi * BLOCK_TOKENS..(bi + 1) * BLOCK_TOKENS];
            let h = hash_block(chunk);
            if let Some(&child) = self.nodes[node as usize].children.get(&h) {
                if self.nodes[child as usize].tokens == chunk {
                    let t = self.tick();
                    self.nodes[child as usize].last_used = t;
                    node = child;
                    continue;
                }
                // hash collision with different tokens: don't cache further (the
                // trie can't disambiguate) - stop extending this path.
                break;
            }
            pool.retain(blocks[bi]); // the tree now holds a reference
            let t = self.tick();
            let nid = self.new_node(Node {
                parent: node,
                block: blocks[bi],
                key: h,
                tokens: chunk.to_vec(),
                children: HashMap::new(),
                last_used: t,
                alive: true,
                recurred: false,
                state_blk: None,
                tkey: self.nodes[node as usize].tkey.map(|k| k.child(chunk)),
            });
            self.nodes[node as usize].children.insert(h, nid);
            node = nid;
        }
    }

    /// Arm KV-tier chain keys: every node inserted from now
    /// on carries `parent.tkey.child(tokens)`, rooted here. Call once at pool
    /// setup, before any insert - nodes created unarmed never demote.
    pub fn set_tier_root(&mut self, root: LogicalKey) {
        self.nodes[0].tkey = Some(root);
    }

    /// The current LRU childless leaf's full root-to-leaf path - what the
    /// tier's demote arm inspects before eviction (it needs every block of a
    /// run alive while the gather reads it; `evict_lru` would already have
    /// released the leaf). Entries are root-first; `depth` is 1-based (=
    /// the number of 16-token blocks the prefix ending here spans).
    /// The `max` least-recently-used leaf paths, oldest first - the
    /// write-through mirror pass walks these: the chains
    /// eviction would pick first are the ones worth pre-storing.
    pub fn lru_leaf_paths(&self, max: usize) -> Vec<Vec<LruPathEntry>> {
        let mut leaves: Vec<(u64, u32)> = self
            .nodes
            .iter()
            .enumerate()
            .filter(|(i, n)| *i != 0 && n.alive && n.children.is_empty())
            .map(|(i, n)| (n.last_used, i as u32))
            .collect();
        leaves.sort_unstable();
        leaves
            .into_iter()
            .take(max)
            .map(|(_, leaf)| self.path_to(leaf))
            .collect()
    }

    /// Every live checkpoint attachment: (depth-blocks, tier chain key,
    /// state index). Bounded by the state-pool capacity, so the blob
    /// write-through can scan it every pass - the LRU leaves are exactly
    /// the chains whose checkpoints have already recycled.
    pub fn state_attachments(&self) -> Vec<(usize, Option<LogicalKey>, u32)> {
        self.nodes
            .iter()
            .enumerate()
            .filter(|(i, n)| *i != 0 && n.alive && n.state_blk.is_some())
            .map(|(i, n)| {
                let mut d = 0usize;
                let mut node = i as u32;
                while node != 0 {
                    d += 1;
                    node = self.nodes[node as usize].parent;
                }
                (d, n.tkey, n.state_blk.expect("filtered"))
            })
            .collect()
    }

    pub fn lru_leaf_path(&self) -> Vec<LruPathEntry> {
        let Some(leaf) = self
            .nodes
            .iter()
            .enumerate()
            .filter(|(i, n)| *i != 0 && n.alive && n.children.is_empty())
            .min_by_key(|(_, n)| n.last_used)
            .map(|(i, _)| i as u32)
        else {
            return Vec::new();
        };
        self.path_to(leaf)
    }

    /// Root-first path entries for `leaf` (depth 1-based) - shared by the
    /// single-victim LRU walk and the mirror pass's multi-leaf variant.
    fn path_to(&self, leaf: u32) -> Vec<LruPathEntry> {
        let mut path = Vec::new();
        let mut node = leaf;
        while node != 0 {
            let n = &self.nodes[node as usize];
            path.push(LruPathEntry {
                node,
                depth: 0,
                block: n.block,
                tkey: n.tkey,
                state_blk: n.state_blk,
            });
            node = n.parent;
        }
        path.reverse();
        for (i, e) in path.iter_mut().enumerate() {
            e.depth = i + 1;
        }
        path
    }

    /// How many full blocks of `tokens` are currently cached (read-only - no
    /// LRU bump, no recurrence marking). The tier's restore publication uses
    /// it to verify a chain is attachable / already published.
    pub fn chain_depth(&self, tokens: &[u32]) -> usize {
        let full = tokens.len() / BLOCK_TOKENS;
        let mut node = 0u32;
        let mut depth = 0;
        for bi in 0..full {
            let chunk = &tokens[bi * BLOCK_TOKENS..(bi + 1) * BLOCK_TOKENS];
            let h = hash_block(chunk);
            let Some(&child) = self.nodes[node as usize].children.get(&h) else {
                break;
            };
            if self.nodes[child as usize].tokens != chunk {
                break;
            }
            depth = bi + 1;
            node = child;
        }
        depth
    }

    /// Publish a restored run: attach `blocks` as chain
    /// blocks `[start_block, start_block + blocks.len())` of `tokens`,
    /// RETAINING each in `pool` exactly like `insert`. Returns `false` -
    /// and touches nothing - unless the chain is present through
    /// `start_block` (the prefix may have been evicted while the restore
    /// was in flight; publishing into a hole would attach content at wrong
    /// positions). Positions already cached keep their existing blocks
    /// (the caller releases its surplus copies, same as `insert`).
    pub fn insert_extension(
        &mut self,
        tokens: &[u32],
        start_block: usize,
        blocks: &[BlockId],
        pool: &mut KvPool,
    ) -> bool {
        if self.chain_depth(tokens) < start_block {
            return false;
        }
        let end = (start_block + blocks.len()).min(tokens.len() / BLOCK_TOKENS);
        // walk to start_block (present per the check), then insert onward -
        // same node lifecycle as `insert`
        let mut node = 0u32;
        for bi in 0..end {
            let chunk = &tokens[bi * BLOCK_TOKENS..(bi + 1) * BLOCK_TOKENS];
            let h = hash_block(chunk);
            if let Some(&child) = self.nodes[node as usize].children.get(&h) {
                if self.nodes[child as usize].tokens == chunk {
                    let t = self.tick();
                    self.nodes[child as usize].last_used = t;
                    node = child;
                    continue;
                }
                return bi > start_block; // collision - stop; partial publish stands
            }
            debug_assert!(bi >= start_block, "chain_depth guaranteed presence");
            pool.retain(blocks[bi - start_block]);
            let t = self.tick();
            let tkey = self.nodes[node as usize].tkey.map(|k| k.child(chunk));
            let nid = self.new_node(Node {
                parent: node,
                block: blocks[bi - start_block],
                key: h,
                tokens: chunk.to_vec(),
                children: HashMap::new(),
                last_used: t,
                alive: true,
                recurred: false,
                state_blk: None,
                tkey,
            });
            self.nodes[node as usize].children.insert(h, nid);
            node = nid;
        }
        true
    }

    /// Detach a node's state checkpoint without recycling its pool index -
    /// the tier's demote arm claims it so the blob's bytes survive until the
    /// store's gather has read them; the index returns via
    /// [`Self::recycle_state`] at store completion. `None` if the node has
    /// no checkpoint (or was already claimed).
    pub fn take_state(&mut self, node: u32) -> Option<u32> {
        self.nodes.get_mut(node as usize)?.state_blk.take()
    }

    /// Return a state index claimed by [`Self::take_state`] to the free list
    /// (the tier calls this once the demote's store completed - or failed;
    /// either way the blob region is no longer read).
    pub fn recycle_state(&mut self, idx: u32) {
        self.state_free.push(idx);
    }

    /// Evict a SPECIFIC live childless leaf (the tier's demote arm walks the
    /// LRU path itself and evicts bottom-up as it goes). Returns the released
    /// block, `None` if `node` is not currently evictable.
    pub fn evict_leaf(&mut self, node: u32, pool: &mut KvPool) -> Option<BlockId> {
        let n = self.nodes.get(node as usize)?;
        if node == 0 || !n.alive || !n.children.is_empty() {
            return None;
        }
        Some(self.evict_node(node, pool))
    }

    /// Evict the least-recently-used childless leaf, RELEASING its block back to
    /// `pool` (the tree's reference; the block frees only if no slot still holds
    /// it). Returns the evicted block id, or `None` if the tree has no leaf. Call
    /// on pool exhaustion to reclaim cached-but-idle prefixes.
    pub fn evict_lru(&mut self, pool: &mut KvPool) -> Option<BlockId> {
        let victim = self
            .nodes
            .iter()
            .enumerate()
            .filter(|(i, n)| *i != 0 && n.alive && n.children.is_empty())
            .min_by_key(|(_, n)| n.last_used)
            .map(|(i, _)| i as u32)?;
        Some(self.evict_node(victim, pool))
    }

    /// Shared teardown for `evict_lru` / `evict_leaf`. Caller guarantees the
    /// node is a live childless non-root leaf.
    fn evict_node(&mut self, victim: u32, pool: &mut KvPool) -> BlockId {
        let (parent, key, blk) = {
            let n = &self.nodes[victim as usize];
            (n.parent, n.key, n.block)
        };
        self.nodes[parent as usize].children.remove(&key);
        self.nodes[victim as usize].alive = false;
        self.nodes[victim as usize].children = HashMap::new();
        if let Some(sb) = self.nodes[victim as usize].state_blk.take() {
            self.state_free.push(sb); // reclaim the checkpoint index
        }
        self.free_nodes.push(victim);
        pool.release(blk);
        blk
    }

    /// Number of cached blocks (alive non-root nodes).
    pub fn cached_blocks(&self) -> usize {
        self.nodes.iter().skip(1).filter(|n| n.alive).count()
    }

    /// Blocks the tree could reclaim under pressure: alive nodes whose block
    /// only the tree references (refcount 1 - not shared with a live slot).
    /// Admission accounting adds this to the pool's free count so the prefix
    /// cache behaves as reclaimable capacity, not a reservation - otherwise a
    /// retention-heavy workload (salted benches, many one-shot prompts) drives
    /// `free` to ~0 and the admission watermark serializes the whole server
    /// behind slot completions (found live: gemma4 c8 TTFT 3.3 s -> 52 s).
    pub fn evictable_blocks(&self, pool: &KvPool) -> usize {
        self.nodes
            .iter()
            .skip(1)
            .filter(|n| n.alive && pool.refcount(n.block) == 1)
            .count()
    }

    fn new_node(&mut self, n: Node) -> u32 {
        if let Some(id) = self.free_nodes.pop() {
            self.nodes[id as usize] = n;
            id
        } else {
            self.nodes.push(n);
            (self.nodes.len() - 1) as u32
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_pool::{BlockTable, KvPool};

    // 16-token block of a constant value (distinct per block seed).
    fn block_toks(seed: u32) -> Vec<u32> {
        (0..BLOCK_TOKENS as u32).map(|i| seed * 100 + i).collect()
    }

    /// A slot prefills `n_blocks` fresh blocks (alloc from the pool) and returns
    /// its block table.
    fn prefill(pool: &mut KvPool, n_blocks: usize) -> BlockTable {
        let mut t = BlockTable::new();
        t.ensure(n_blocks * BLOCK_TOKENS - 1, pool).expect("alloc");
        t
    }

    #[test]
    fn empty_match_is_empty() {
        let mut r = PagedRadix::new();
        assert!(r.match_prefix(&block_toks(1)).is_empty());
    }

    #[test]
    fn insert_then_match_hits_and_tree_holds_a_ref() {
        let mut pool = KvPool::with_blocks(16);
        let mut r = PagedRadix::new();
        // slot prefills 2 blocks, then inserts them
        let table = prefill(&mut pool, 2);
        let mut toks: Vec<u32> = block_toks(1);
        toks.extend(block_toks(2));
        toks.push(999); // +1 so 2 full blocks are cacheable (cap keeps 1 token)
        r.insert(&toks, table.blocks(), &mut pool);
        assert_eq!(r.cached_blocks(), 2);
        // both blocks now held by the slot AND the tree
        for &b in table.blocks() {
            assert_eq!(pool.refcount(b), 2, "slot + tree");
        }
        // a fresh prompt with the same prefix matches both blocks
        let got = r.match_prefix(&toks);
        assert_eq!(got, table.blocks());
    }

    #[test]
    fn match_keeps_at_least_one_token() {
        let mut pool = KvPool::with_blocks(16);
        let mut r = PagedRadix::new();
        let table = prefill(&mut pool, 2);
        let toks: Vec<u32> = [block_toks(1), block_toks(2)].concat(); // exactly 32
        r.insert(&toks, table.blocks(), &mut pool);
        // matching the same 32 tokens keeps the last block unmatched (cap=31 ->
        // 1 full block), so there is always a token left to prefill.
        assert_eq!(r.match_prefix(&toks).len(), 1);
    }

    #[test]
    fn branching_prefix_shares_the_common_block() {
        let mut pool = KvPool::with_blocks(16);
        let mut r = PagedRadix::new();
        // seq A: block1, block2
        let ta = prefill(&mut pool, 2);
        let a: Vec<u32> = [block_toks(1), block_toks(2), vec![7]].concat();
        r.insert(&a, ta.blocks(), &mut pool);
        // seq B shares block1 then diverges to block3
        let b_prefix: Vec<u32> = [block_toks(1), block_toks(3), vec![7]].concat();
        let shared = r.match_prefix(&b_prefix);
        assert_eq!(
            shared,
            &ta.blocks()[..1],
            "shares only the common first block"
        );
        // B adopts the shared block into its own table (refcount++)
        let mut tb = BlockTable::new();
        tb.share_prefix(&shared, &mut pool);
        assert_eq!(pool.refcount(shared[0]), 3, "A-slot + tree + B-slot");
    }

    #[test]
    fn evict_lru_releases_tree_ref_and_frees_when_unshared() {
        let mut pool = KvPool::with_blocks(4);
        let mut r = PagedRadix::new();
        let mut table = prefill(&mut pool, 2); // blocks used by the slot
        let toks: Vec<u32> = [block_toks(1), block_toks(2), vec![7]].concat();
        r.insert(&toks, table.blocks(), &mut pool);
        let free_after_insert = pool.free_blocks();
        // the slot finishes: its refs drop (free-on-completion). Blocks stay
        // pinned by the tree (refcount 1), so still not free.
        table.clear(&mut pool);
        assert_eq!(
            pool.free_blocks(),
            free_after_insert,
            "tree still pins them"
        );
        // evict both leaves -> their blocks return to the pool
        assert!(r.evict_lru(&mut pool).is_some());
        assert!(r.evict_lru(&mut pool).is_some());
        assert_eq!(r.cached_blocks(), 0);
        assert_eq!(pool.free_blocks(), 4, "all blocks back");
    }

    #[test]
    fn evict_lru_targets_the_least_recently_used() {
        let mut pool = KvPool::with_blocks(8);
        let mut r = PagedRadix::new();
        // two independent single-block prefixes
        let mut t1 = BlockTable::new();
        t1.ensure(BLOCK_TOKENS - 1, &mut pool).unwrap();
        let a: Vec<u32> = [block_toks(1), vec![7]].concat();
        r.insert(&a, t1.blocks(), &mut pool);
        let mut t2 = BlockTable::new();
        t2.ensure(BLOCK_TOKENS - 1, &mut pool).unwrap();
        let b: Vec<u32> = [block_toks(2), vec![7]].concat();
        r.insert(&b, t2.blocks(), &mut pool);
        // touch A (more recent) -> B is the LRU and must be evicted first
        let _ = r.match_prefix(&a);
        let evicted = r.evict_lru(&mut pool).unwrap();
        assert_eq!(evicted, t2.blocks()[0], "LRU (B) evicted, not A");
    }

    #[test]
    fn attach_state_and_match_full_returns_the_deepest_checkpoint() {
        let mut pool = KvPool::with_blocks(16);
        let mut r = PagedRadix::new();
        r.set_state_capacity(4);
        let table = prefill(&mut pool, 3);
        let toks: Vec<u32> = [block_toks(1), block_toks(2), block_toks(3), vec![9]].concat();
        r.insert(&toks, table.blocks(), &mut pool);
        // no state yet
        assert!(r.match_full(&toks).ckpt.is_none());
        // checkpoint at the 2-block boundary (pos 32)
        let sb = r.attach_state(&toks, 2 * BLOCK_TOKENS).expect("attach");
        // re-attaching the same node is a no-op
        assert!(r.attach_state(&toks, 2 * BLOCK_TOKENS).is_none());
        // match now reports the checkpoint (pos 32, index sb); blocks intact
        let m = r.match_full(&toks);
        assert_eq!(m.ckpt, Some((2 * BLOCK_TOKENS, sb)));
        assert_eq!(m.blocks, table.blocks());
    }

    #[test]
    fn no_state_capacity_means_no_checkpoints() {
        let mut pool = KvPool::with_blocks(8);
        let mut r = PagedRadix::new(); // state capacity not enabled
        let table = prefill(&mut pool, 2);
        let toks: Vec<u32> = [block_toks(1), block_toks(2), vec![9]].concat();
        r.insert(&toks, table.blocks(), &mut pool);
        assert!(r.attach_state(&toks, BLOCK_TOKENS).is_none());
        assert!(r.match_full(&toks).ckpt.is_none());
    }

    #[test]
    fn state_pool_exhaustion_steals_the_lru_checkpoint() {
        let mut pool = KvPool::with_blocks(16);
        let mut r = PagedRadix::new();
        r.set_state_capacity(1); // room for one checkpoint
        // two independent 1-block prefixes, each checkpointed at pos 16
        let mut t1 = BlockTable::new();
        t1.ensure(BLOCK_TOKENS - 1, &mut pool).unwrap();
        let a: Vec<u32> = [block_toks(1), vec![9]].concat();
        r.insert(&a, t1.blocks(), &mut pool);
        let s1 = r.attach_state(&a, BLOCK_TOKENS).expect("a state");
        let mut t2 = BlockTable::new();
        t2.ensure(BLOCK_TOKENS - 1, &mut pool).unwrap();
        let b: Vec<u32> = [block_toks(2), vec![9]].concat();
        r.insert(&b, t2.blocks(), &mut pool);
        // pool exhausted -> steals A's checkpoint (LRU), reusing the same index
        let s2 = r.attach_state(&b, BLOCK_TOKENS).expect("b steals a");
        assert_eq!(s1, s2, "reused the stolen index");
        // A no longer has a checkpoint; B does
        assert!(r.match_full(&a).ckpt.is_none());
        assert_eq!(r.match_full(&b).ckpt, Some((BLOCK_TOKENS, s2)));
    }

    /// Two 1-block prefixes and room for one checkpoint. `on_a` decides whether
    /// A is resumed (which marks it proven) before B asks for the slot.
    /// Returns (radix, a, b, a's original state index, B's attach result).
    /// Two 1-block prefixes and room for one checkpoint. `recur_a` decides
    /// whether A is matched again (which marks it recurred) before B asks for
    /// the slot. Returns (radix, a, b, a's state index, B's attach result).
    fn two_prefixes_one_slot(
        protect: bool,
        recur_a: bool,
    ) -> (PagedRadix, Vec<u32>, Vec<u32>, u32, Option<u32>) {
        let mut pool = KvPool::with_blocks(16);
        let mut r = PagedRadix::new();
        r.set_state_capacity(1);
        r.set_protect_proven(protect);
        let mut t1 = BlockTable::new();
        t1.ensure(BLOCK_TOKENS - 1, &mut pool).unwrap();
        let a: Vec<u32> = [block_toks(1), vec![9]].concat();
        r.insert(&a, t1.blocks(), &mut pool);
        let s1 = r.attach_state(&a, BLOCK_TOKENS).expect("a state");
        if recur_a {
            r.match_full(&a); // A comes back - this is what marks it recurred
        }
        let mut t2 = BlockTable::new();
        t2.ensure(BLOCK_TOKENS - 1, &mut pool).unwrap();
        let b: Vec<u32> = [block_toks(2), vec![9]].concat();
        r.insert(&b, t2.blocks(), &mut pool);
        let sb = r.attach_state(&b, BLOCK_TOKENS);
        (r, a, b, s1, sb)
    }

    #[test]
    fn protect_proven_refuses_to_evict_a_recurring_prefix() {
        let (mut r, a, b, s1, sb) = two_prefixes_one_slot(true, true);
        // A has come back once, so its checkpoint is the one about to be hit
        // again - under plain LRU B evicts exactly that. Refuse instead.
        assert!(
            sb.is_none(),
            "must not evict a recurring prefix's checkpoint"
        );
        assert_eq!(
            r.match_full(&a).ckpt,
            Some((BLOCK_TOKENS, s1)),
            "A survives"
        );
        assert!(r.match_full(&b).ckpt.is_none(), "B got no checkpoint");
        let (_, _, refused) = r.state_stats();
        assert_eq!(refused, 1);
    }

    #[test]
    fn protect_proven_still_steals_from_a_prefix_that_never_came_back() {
        // A was checkpointed on its first and only sighting, so it has shown no
        // recurrence and carries no protection: B takes its slot as before.
        let (mut r, a, b, s1, sb) = two_prefixes_one_slot(true, false);
        assert_eq!(sb, Some(s1), "reused the stolen index");
        assert!(r.match_full(&a).ckpt.is_none());
        assert_eq!(r.match_full(&b).ckpt, Some((BLOCK_TOKENS, s1)));
        let (_, steals, refused) = r.state_stats();
        assert_eq!((steals, refused), (1, 0));
    }

    #[test]
    fn protect_proven_is_off_by_default_and_steals() {
        // The unarmed path must behave exactly as it did before the policy.
        let (mut r, a, _b, s1, sb) = two_prefixes_one_slot(false, true);
        assert_eq!(
            sb,
            Some(s1),
            "default policy still steals from a recurring A"
        );
        assert!(r.match_full(&a).ckpt.is_none());
    }

    #[test]
    fn a_page_match_marks_recurrence_even_when_the_state_was_stolen() {
        // The regression that made the first version of this policy inert: a
        // prefix whose checkpoint is gone still matches its PAGES, and that is
        // the signal. Keyed on successful resumes instead, it stays invisible.
        let mut pool = KvPool::with_blocks(16);
        let mut r = PagedRadix::new();
        r.set_state_capacity(1);
        let mut t1 = BlockTable::new();
        t1.ensure(BLOCK_TOKENS - 1, &mut pool).unwrap();
        let a: Vec<u32> = [block_toks(1), vec![9]].concat();
        r.insert(&a, t1.blocks(), &mut pool);
        r.attach_state(&a, BLOCK_TOKENS).expect("a state");
        // B steals A's checkpoint (policy off), so A's pages survive but its
        // state does not.
        let mut t2 = BlockTable::new();
        t2.ensure(BLOCK_TOKENS - 1, &mut pool).unwrap();
        let b: Vec<u32> = [block_toks(2), vec![9]].concat();
        r.insert(&b, t2.blocks(), &mut pool);
        r.attach_state(&b, BLOCK_TOKENS).expect("b steals");
        let m = r.match_full(&a);
        assert!(m.ckpt.is_none(), "state gone");
        assert_eq!(
            m.blocks.len(),
            1,
            "pages still match - the recurrence signal"
        );
        // Armed, the roles now invert on that evidence alone: A has come back
        // and B never has, so A reclaims the slot from the one-hit-wonder.
        r.set_protect_proven(true);
        assert!(
            r.attach_state(&a, BLOCK_TOKENS).is_some(),
            "A reclaims from non-recurring B"
        );
        assert!(
            r.match_full(&b).ckpt.is_none(),
            "B lost the slot it never earned"
        );
        // ...and now that B has been matched too, A is protected from it.
        assert!(
            r.attach_state(&b, BLOCK_TOKENS).is_none(),
            "B refused against recurring A"
        );
        let (_, _, refused) = r.state_stats();
        assert_eq!(refused, 1);
    }

    #[test]
    fn evict_lru_reclaims_the_checkpoint_index() {
        let mut pool = KvPool::with_blocks(8);
        let mut r = PagedRadix::new();
        r.set_state_capacity(1);
        let table = prefill(&mut pool, 1);
        let toks: Vec<u32> = [block_toks(1), vec![9]].concat();
        r.insert(&toks, table.blocks(), &mut pool);
        r.attach_state(&toks, BLOCK_TOKENS).expect("state");
        // evicting the node reclaims its state index (free-list refilled), so a
        // fresh prefix can checkpoint again.
        assert!(r.evict_lru(&mut pool).is_some());
        let mut t2 = BlockTable::new();
        t2.ensure(BLOCK_TOKENS - 1, &mut pool).unwrap();
        let b: Vec<u32> = [block_toks(2), vec![9]].concat();
        r.insert(&b, t2.blocks(), &mut pool);
        assert!(
            r.attach_state(&b, BLOCK_TOKENS).is_some(),
            "index reclaimed"
        );
    }

    #[test]
    fn reinsert_of_cached_prefix_does_not_double_retain() {
        let mut pool = KvPool::with_blocks(8);
        let mut r = PagedRadix::new();
        let table = prefill(&mut pool, 1);
        let toks: Vec<u32> = [block_toks(1), vec![7]].concat();
        r.insert(&toks, table.blocks(), &mut pool);
        let rc = pool.refcount(table.blocks()[0]);
        // a second request with the same block re-inserts: the node already
        // exists, so no extra retain (the tree holds exactly one reference).
        r.insert(&toks, table.blocks(), &mut pool);
        assert_eq!(pool.refcount(table.blocks()[0]), rc, "no double-retain");
        assert_eq!(r.cached_blocks(), 1);
    }
}
