//! Shared prefix-block primitives for the paged radix caches.
//!
//! The dense `RadixKvCache` that used to live here (a copy-in/copy-out block
//! store for the non-paged KV mode) is gone: the paged pool +
//! `PagedRadix` zero-copy cache is the only prefix cache now - pool mode is
//! the serving default for every family, the newer families (gemma4, laguna,
//! granite) never had a dense lane, and the dense fallback was a silent
//! second regime waiting to swallow a config. What remains is what all the
//! paged caches share: the page-size constant, the image-content radix keys,
//! the image-span checkpoint guard, and the evict-ahead margin clamp.

/// Tokens per cache page. Prompts share their common leading full pages; the last
/// partial page (< this) is never cached, and re-prefilled (cheap).
pub const BLOCK_TOKENS: usize = 16;

/// The radix key for row `j` of an image whose content hashes to `h`.
///
/// Every image row of every prompt carries the same `<image>` placeholder id, so
/// a radix keyed on the row tokens treats two different pictures as the same
/// prefix and serves one image's KV for the other - "of two concurrent image
/// requests, the blue-image slot answered red". That is why multimodal prompts
/// were excluded from prefix caching engine-wide. The fix is to key image rows
/// on the picture's CONTENT instead, and this is the one definition of how, so
/// the families cannot drift apart on it.
///
/// Two properties, both load-bearing:
///
/// 1. **Different pictures differ.** `h` is the content hash each family's image
///    cache already computes, so two pictures agree here exactly when they agree
///    byte for byte.
/// 2. **It can never equal a token id.** The high bit is set, and real ids are
///    bounded by the vocab (~100-260k), so the image and text key spaces do not
///    overlap. Without this a text prompt could - however improbably - hash into
///    an image's path and adopt KV for rows it never wrote.
///
/// Folding in `j` keeps a picture's run internally ordered, so a prompt that
/// reuses only PART of an image still matches position by position rather than
/// matching any row against any other.
pub fn image_key_row(h: u64, j: usize) -> u32 {
    let mixed = h
        .wrapping_mul(0x9e37_79b9_7f4a_7c15)
        .wrapping_add((j as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9));
    // fold to 31 bits, then set the high bit
    0x8000_0000 | ((mixed ^ (mixed >> 32)) as u32 & 0x7fff_ffff)
}

/// The evict-ahead free margin an insert may hold back, clamped to what this
/// pool can actually honour.
///
/// Every family evicts down to a free margin after inserting, so eviction
/// happens off the admission path and freed ids come back through the free
/// list's LIFO reuse. The margins were tuned as ABSOLUTE block counts against
/// big-context servers (gemma4 2048, laguna 1024, granite 256) - and an
/// absolute count is a silent trap on a small one: a runner with max_ctx 8192
/// and one slot has a pool of a few hundred blocks, so `free < margin` holds
/// even with the pool completely empty. The loop then evicts everything the
/// insert just published, every time, and the prefix cache is not degraded but
/// off, with nothing anywhere saying so. Measured on gemma4 at 8k/1 slot: a
/// repeated identical prompt logged `matched 0` on every request, and `matched
/// 80` once the margin was manually zeroed.
///
/// A quarter of the pool is the ceiling: enough free blocks that admissions
/// still draw from the LIFO list, with the other three quarters left for the
/// retention the cache exists to hold. Warn once when the configured value had
/// to be cut, because "your margin is bigger than your pool" is a
/// configuration mistake the user cannot otherwise see.
pub fn evict_ahead_margin(configured: usize, pool_capacity: usize) -> usize {
    if configured == 0 {
        return 0; // explicitly disabled: evict only under real pressure
    }
    let ceiling = pool_capacity / 4;
    if configured > ceiling {
        static WARNED: std::sync::Once = std::sync::Once::new();
        WARNED.call_once(|| {
            tracing::warn!(
                "prefix cache: evict-ahead margin {configured} blocks exceeds a quarter of the \
                 {pool_capacity}-block pool; using {ceiling}. An unclamped margin this large \
                 would evict every prefix as soon as it was cached."
            );
        });
    }
    configured.min(ceiling)
}

/// Walk a checkpoint cut back to a page boundary at or before the start of any
/// image span it lands strictly inside.
///
/// Both vision families that cache across pictures give an image's rows MUTUAL
/// visibility rather than causal-within-the-span: gemma4v decodes them
/// non-causally, and qwen35 gives every row of one picture the same mRoPE `t`
/// plus an attention bound pointing at the span's last row. Either way a resume
/// landing strictly inside a picture would re-prefill rows whose attention
/// reaches keys the adopted blocks already hold, written in a different order.
///
/// Rather than reason about whether that is benign, no CHECKPOINT is ever
/// attached inside an image span - and since a resume position is exactly a
/// checkpoint position, mid-span resumes cannot occur at all. No guard is then
/// needed on the resume side or in the prefill loop, and there is no state
/// where "the cut moved" and "the resume moved" can disagree.
///
/// Walking BACK rather than forward matters: forward would place the cut past
/// rows the tail must then re-prefill, and the tail is what re-writes the
/// picture. Back lands before the picture, so a span is either entirely adopted
/// or entirely re-prefilled.
///
/// Spans arrive in row order, so one reverse pass settles it: each step moves
/// the cut before a picture, which can only expose an earlier picture, never a
/// later one. Pure, so the rule is testable without a GPU.
pub fn cut_outside_image_spans(mut cut: usize, img_spans: &[(usize, usize)]) -> usize {
    for &(a, b) in img_spans.iter().rev() {
        if cut > a && cut < b {
            cut = a / BLOCK_TOKENS * BLOCK_TOKENS;
        }
    }
    cut
}

#[cfg(test)]
mod span_tests {
    use super::{BLOCK_TOKENS, cut_outside_image_spans as cut_outside};

    /// The property the whole guard exists for: an image's rows attend to their
    /// span's last position, so a checkpoint inside a picture would let a later
    /// turn resume mid-picture and re-prefill rows whose attention bound points
    /// past the cut. No checkpoint inside a span, no such resume.
    #[test]
    fn a_cut_never_lands_inside_a_picture() {
        // one picture occupying rows [20, 300)
        let spans = [(20usize, 300usize)];
        for cut in (0..400).step_by(BLOCK_TOKENS) {
            let out = cut_outside(cut, &spans);
            assert!(
                out <= 20 || out >= 300,
                "cut {cut} landed at {out}, inside [20, 300)"
            );
            assert!(out <= cut, "the cut may only move BACK, {cut} -> {out}");
            assert_eq!(out % BLOCK_TOKENS, 0, "cut {out} left the page grid");
        }
    }

    /// A cut clear of every picture is untouched - the common document shape is
    /// image first then a long text tail, and that tail must stay resumable.
    #[test]
    fn a_cut_past_the_picture_is_left_alone() {
        let spans = [(20usize, 300usize)];
        assert_eq!(cut_outside(320, &spans), 320);
        assert_eq!(cut_outside(300, &spans), 300);
        // and one before it
        assert_eq!(cut_outside(16, &spans), 16);
    }

    /// Several pictures: stepping back out of one must not strand the cut
    /// inside an earlier one.
    #[test]
    fn stepping_back_out_of_one_picture_clears_the_earlier_ones() {
        // pictures at [16, 100) and [104, 200) with only 4 rows between them
        let spans = [(16usize, 100usize), (104usize, 200usize)];
        for cut in (0..240).step_by(BLOCK_TOKENS) {
            let out = cut_outside(cut, &spans);
            for &(a, b) in &spans {
                assert!(out <= a || out >= b, "cut {cut} -> {out} inside [{a}, {b})");
            }
        }
    }

    /// A text-only prompt has no spans, so nothing moves.
    #[test]
    fn without_pictures_the_cut_is_the_text_cut() {
        for cut in [0usize, 16, 64, 1024] {
            assert_eq!(cut_outside(cut, &[]), cut);
        }
    }

    /// qwen35's shape: a system preamble, then a big picture, then the question.
    /// Both of the two-boundary rule's cuts land in the text tail and must
    /// survive untouched, or a document conversation resumes nowhere.
    #[test]
    fn the_text_tail_after_a_picture_stays_resumable() {
        // 24 rows of preamble, a 1440-row picture, then a 400-row tail
        let spans = [(24usize, 1464usize)];
        for cut in [1856usize, 1840] {
            assert_eq!(
                cut_outside(cut, &spans),
                cut,
                "tail cut {cut} was walked back"
            );
        }
    }
}

#[cfg(test)]
mod margin_tests {
    use super::evict_ahead_margin;

    /// The bug this closes: a margin tuned for a big-context server is larger
    /// than a small server's entire pool, so the evict-ahead loop runs until
    /// the radix is empty on every insert and the cache is silently off.
    /// Measured on gemma4 at max_ctx 8192 with one slot before the clamp.
    #[test]
    fn a_margin_larger_than_the_pool_cannot_empty_it() {
        // gemma4's default against a small server's few hundred blocks
        let pool = 400;
        let m = evict_ahead_margin(2048, pool);
        assert!(
            m < pool,
            "margin {m} still covers the whole {pool}-block pool"
        );
        // and it leaves the majority of the pool for retention, which is the
        // point - a cache that can hold nothing is not a cache
        assert!(
            pool - m >= pool * 3 / 4,
            "only {} blocks left to cache with",
            pool - m
        );
    }

    /// On the servers these numbers were tuned for, nothing changes.
    #[test]
    fn a_margin_the_pool_can_afford_is_untouched() {
        assert_eq!(evict_ahead_margin(2048, 40_000), 2048);
        assert_eq!(evict_ahead_margin(1024, 8_192), 1024);
        assert_eq!(evict_ahead_margin(256, 4_096), 256);
        // exactly at the ceiling is still affordable
        assert_eq!(evict_ahead_margin(1024, 4_096), 1024);
    }

    /// 0 means "evict only under real pressure" and must stay exactly that -
    /// the clamp must not turn a deliberate opt-out into a quarter-pool margin.
    #[test]
    fn zero_stays_disabled() {
        assert_eq!(evict_ahead_margin(0, 40_000), 0);
        assert_eq!(evict_ahead_margin(0, 0), 0);
    }

    /// A pool too small to spare anything asks for no margin at all, rather
    /// than one block that evicts the only page there was room for.
    #[test]
    fn a_tiny_pool_gets_no_margin() {
        for cap in 0..4 {
            assert_eq!(evict_ahead_margin(2048, cap), 0, "capacity {cap}");
        }
    }
}
