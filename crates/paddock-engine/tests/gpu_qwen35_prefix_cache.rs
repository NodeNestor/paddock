//! Qwen3.5 hybrid prefix-cache gates. A hybrid model can only RESUME a cached
//! prefix at a DeltaNet state checkpoint (the recurrent state has no rollback),
//! so the cache pairs full-attn KV pages with state snapshots at the last two
//! page boundaries of every prefill.
//!
//! Gate 1 (bit-exact): re-prefilling the same prompt resumes from its own
//! checkpoint - the resumed chunk replays the cold run's final chunk with a
//! bit-exact restored state over byte-identical KV pages, so the logits must
//! match EXACTLY.
//!
//! Gate 2 (multi-turn shape): a prompt sharing all but its trailing tokens
//! with a cached one (the re-rendered-history case) must reuse a checkpoint
//! and stay greedy-identical to the pinned-off (PADDOCK_NO_PREFIX_CACHE)
//! single-chunk path. The resume geometry differs from the cold run's, so the
//! DeltaNet chunked-scan grouping differs - greedy + loose L2 is the honest
//! bar (the S2/G2 gate lesson), on a clear-winner prompt.
//!
//! Heavy GPU test: PADDOCK_HEAVY_TESTS=1, --release, --test-threads=1.

mod common;

use paddock_engine::gpu_model::qwen35::GpuQwen35;
use paddock_models::mapped::MappedGguf;
use paddock_tokenizer::GgufTokenizer;

fn setup() -> Option<(GpuQwen35, GgufTokenizer)> {
    if !common::heavy() {
        return None;
    }
    let Some(path) = common::model("QWEN35_GGUF", common::QWEN35_9B_Q8) else {
        return None;
    };
    let Some(exec) = common::gpu_arc() else {
        return None;
    };
    let map = MappedGguf::open(&path).expect("open gguf");
    let tok = GgufTokenizer::from_gguf(map.gguf()).expect("tokenizer");
    let m = GpuQwen35::load(exec, &map, 4096).expect("load 9B");
    Some((m, tok))
}

/// ~`n`-token prompt of varied real text (clear-winner continuations).
fn long_prompt(tok: &GgufTokenizer, n: usize) -> Vec<u32> {
    let base = tok
        .encode(
            "The reference manual describes a distributed consensus protocol in which \
             every participant maintains a monotonically increasing term counter and \
             exchanges signed heartbeat messages over authenticated channels. ",
        )
        .expect("enc");
    base.iter().copied().cycle().take(n).collect()
}

fn amax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map_or(0, |(i, _)| i)
}

fn rel(x: &[f32], y: &[f32]) -> f32 {
    let num: f64 = x.iter().zip(y).map(|(p, q)| ((p - q) as f64).powi(2)).sum();
    let den: f64 = y.iter().map(|p| (*p as f64).powi(2)).sum();
    (num.sqrt() / den.sqrt().max(1e-12)) as f32
}

#[test]
fn resume_from_own_checkpoint_is_bit_exact() {
    let Some((mut m, tok)) = setup() else { return };
    m.enable_batch(4).expect("enable_batch");

    let b = long_prompt(&tok, 203); // non-page-aligned deliberately
    let cold = m.forward_prefill_slot(0, &b).expect("cold prefill");
    assert_eq!(m.take_prefill_reused(0), 0, "cold run must not reuse");

    let warm = m.forward_prefill_slot(1, &b).expect("warm prefill");
    let reused = m.take_prefill_reused(1);
    let b1 = (b.len() - 1) / 16 * 16;
    eprintln!(
        "PREFIX CACHE: reused {reused} of {} tokens (checkpoint at {b1})",
        b.len()
    );
    assert_eq!(reused, b1, "must resume at the deepest checkpoint");
    // restored state is a bit-exact snapshot + KV pages are byte-identical +
    // the resumed chunk has the cold run's exact geometry => logits identical
    let n_diff = cold.iter().zip(&warm).filter(|(a, b)| a != b).count();
    assert_eq!(
        n_diff,
        0,
        "resume must be BIT-EXACT; {} of {} logits differ (rel {:.2e})",
        n_diff,
        cold.len(),
        rel(&warm, &cold)
    );
}

/// Loose L2 bound, per the established gate policy (gpu_gpt_oss_parity's
/// F16_KV_REL): a resume changes the DeltaNet chunked-scan grouping, so
/// logits legitimately drift ~1e-2 while the greedy token holds; 3e-1 still
/// catches O(1) breakage (wrong window / state / slot mapping).
const LOOSE_REL: f32 = 3e-1;

#[test]
fn multi_turn_shape_reuses_and_matches_pinned_reference() {
    let Some((mut m, tok)) = setup() else { return };

    let a = long_prompt(&tok, 199);
    // the multi-turn shape: everything but the trailing generation header is
    // re-sent verbatim, then diverges (new turn) - three independent tails
    // make the greedy gate three independent trials
    let tails = [
        " The committee voted to adopt the second proposal because",
        " In summary, the protocol guarantees safety by requiring",
        " The final tally showed forty-two delegates voting in favor of",
    ];
    let bs: Vec<Vec<u32>> = tails
        .iter()
        .map(|t| {
            let mut b: Vec<u32> = a[..a.len() - 6].to_vec();
            b.extend(tok.encode(t).expect("enc"));
            b
        })
        .collect();

    // pinned references: the pre-cache single-chunk path
    unsafe { std::env::set_var("PADDOCK_NO_PREFIX_CACHE", "1") };
    m.enable_batch(4).expect("enable_batch pinned");
    let refs: Vec<Vec<f32>> = bs
        .iter()
        .enumerate()
        .map(|(i, b)| {
            let r = m.forward_prefill_slot(i, b).expect("pinned prefill");
            assert_eq!(m.take_prefill_reused(i), 0, "pinned path must not reuse");
            r
        })
        .collect();
    unsafe { std::env::remove_var("PADDOCK_NO_PREFIX_CACHE") };

    // cached run: A fills the cache (checkpoints at its last two page
    // boundaries), each B resumes from the deepest one under the shared prefix
    m.enable_batch(4).expect("enable_batch cached");
    let _ = m.forward_prefill_slot(0, &a).expect("prefill A");
    assert_eq!(m.take_prefill_reused(0), 0);
    for (i, (b, reference)) in bs.iter().zip(&refs).enumerate() {
        let got = m.forward_prefill_slot(1 + i, b).expect("prefill B");
        let reused = m.take_prefill_reused(1 + i);
        let r = rel(&got, reference);
        eprintln!(
            "PREFIX CACHE multi-turn[{i}]: reused {reused} of {} shared; rel {:.2e}; \
             greedy {} vs {}",
            a.len() - 6,
            r,
            amax(&got),
            amax(reference)
        );
        assert!(reused >= 16, "expected checkpoint reuse, got {reused}");
        assert!(
            reused <= a.len() - 6,
            "reuse cannot exceed the shared prefix"
        );
        assert_eq!(
            amax(&got),
            amax(reference),
            "greedy token flipped (tail {i})"
        );
        assert!(r < LOOSE_REL, "diverged: rel {r} (tail {i})");
    }
}
