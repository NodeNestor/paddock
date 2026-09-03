//! The race suite: the replica state machine + catalog proven
//! against the deterministic fake transport before any CUDA or disk exists.
//!
//! Three layers:
//! 1. directed transition tests - every documented rule, one test each;
//! 2. exhaustive interleavings - every permutation of the adversarial event
//!    sets (cancel / complete / evict / join racing one another);
//! 3. seeded property fuzz - thousands of random op streams, with
//!    `check_invariants` after every event plus waiter-liveness bookkeeping
//!    (every waiter woken exactly once, or cancelled by its own hand).

use rand_core::{Rng, SeedableRng};

use super::catalog::*;
use super::digest::*;
use super::transport::*;
use super::*;

const KB: u64 = 1024;

fn ns() -> CacheNamespace {
    let id = IdentityDigest::compute(&IdentityFields {
        model_tensors: b"model",
        adapter: b"",
        architecture: b"test-arch",
        cache_schema: b"schema-v1",
        layout_abi: 1,
        tokenizer: b"tok",
    });
    CacheNamespace {
        identity: id,
        scope: PrivacyScope::Shared,
    }
}

/// The on-disk cache directory must change when the WEIGHTS or the TOKENIZER
/// change, not only when the geometry does. Every family used to pass `b""`
/// for both, so two checkpoints of one model at different quants
/// shared a directory - identical geometry, different activations, adopted
/// silently across a restart. This test is what makes that un-shippable
/// again: it fails the moment a field stops reaching the digest.
#[test]
fn the_cache_directory_separates_different_weights_and_tokenizers() {
    use super::nvme_store::NvmeStore;
    let data = std::path::Path::new("/data");
    let mk = |tensors: &[u8], tok: &[u8]| {
        let id = IdentityDigest::compute(&IdentityFields {
            model_tensors: tensors,
            adapter: b"",
            // deliberately CONSTANT: geometry is what used to be the whole
            // key, and it must not be what saves us here
            architecture: b"same-arch layers=32 kv_dim=1024 dtype=Fp16 max_ctx=8192",
            cache_schema: b"schema-v1",
            layout_abi: 1,
            tokenizer: tok,
        });
        NvmeStore::dir_for(
            data,
            &CacheNamespace {
                identity: id,
                scope: PrivacyScope::Shared,
            },
        )
    };
    let q8 = mk(b"weights-of-the-q8-file", b"tok-v1");
    let q4 = mk(b"weights-of-the-q4-file", b"tok-v1");
    let retok = mk(b"weights-of-the-q8-file", b"tok-v2");
    assert_ne!(
        q8, q4,
        "two quants of one model must not share a cache directory"
    );
    assert_ne!(
        q8, retok,
        "a tokenizer revision must not share a cache directory"
    );
    assert_ne!(q4, retok);
    // and identical inputs still agree, or every restart would be cold
    assert_eq!(q8, mk(b"weights-of-the-q8-file", b"tok-v1"));
}

/// An adapter is part of what produced the activations, and the field is
/// already in the digest - check it is load-bearing too, so a LoRA lane
/// cannot repeat the same mistake later.
#[test]
fn an_adapter_changes_the_namespace() {
    let mk = |adapter: &[u8]| {
        IdentityDigest::compute(&IdentityFields {
            model_tensors: b"w",
            adapter,
            architecture: b"a",
            cache_schema: b"s",
            layout_abi: 1,
            tokenizer: b"t",
        })
    };
    assert_ne!(mk(b""), mk(b"lora-1"));
    assert_ne!(mk(b"lora-1"), mk(b"lora-2"));
}

fn keys(n: usize) -> Vec<LogicalKey> {
    let mut out = Vec::with_capacity(n);
    let mut k = ns().root();
    for i in 0..n {
        k = k.child(&[i as u32; 16]);
        out.push(k);
    }
    out
}

fn cat(ram: u64, nvme: u64) -> TierCatalog {
    TierCatalog::new(TierCatalogConfig {
        ram_capacity: ram,
        nvme_capacity: nvme,
    })
}

/// The checksum the fake transport will genuinely produce for this key+len -
/// what an honest producer passes to `begin_store`.
fn honest(key: &LogicalKey, bytes: u64) -> Checksum {
    Checksum::of_payload(&synth_payload(key, bytes))
}

/// reserve + begin_store + submit + deliver + complete: one Ready replica.
fn store_ready(
    c: &mut TierCatalog,
    t: &mut FakeTransport,
    key: LogicalKey,
    tier: Tier,
    bytes: u64,
) {
    c.reserve(key, tier, bytes).unwrap();
    let (op, job) = c
        .begin_store(key, tier, bytes, honest(&key, bytes))
        .unwrap();
    t.submit(job).unwrap();
    t.deliver(op);
    for done in t.poll() {
        assert!(c.on_completion(done).is_empty());
    }
    c.check_invariants();
    assert_eq!(c.ready_bytes(&key, tier), Some(bytes));
}

// ---------------------------------------------------------------------------
// 1. directed transitions
// ---------------------------------------------------------------------------

#[test]
fn store_load_happy_path_balances_the_ledger() {
    let (mut c, mut t) = (cat(10 * KB, 10 * KB), FakeTransport::new());
    let k = keys(1)[0];
    // reserve holds a BOUND; begin_store trims to the sealed size
    c.reserve(k, Tier::Ram, 4 * KB).unwrap();
    assert_eq!(c.ledger(Tier::Ram).reserved, 4 * KB);
    let (op, job) = c
        .begin_store(k, Tier::Ram, 3 * KB, honest(&k, 3 * KB))
        .unwrap();
    let l = c.ledger(Tier::Ram);
    assert_eq!(
        (l.reserved, l.in_flight, l.free),
        (0, 3 * KB, 7 * KB),
        "surplus returns at seal"
    );
    t.submit(job).unwrap();
    t.deliver(op);
    for done in t.poll() {
        c.on_completion(done);
    }
    c.check_invariants();
    let l = c.ledger(Tier::Ram);
    assert_eq!((l.in_flight, l.ready, l.free), (0, 3 * KB, 7 * KB));

    // load toward the GPU: pins the source for the whole op
    let LoadStart::Started { op, job } = c
        .begin_load(k, Tier::Ram, LoadDst::Gpu, WaiterId(1))
        .unwrap()
    else {
        panic!("fresh load must start")
    };
    assert_eq!(c.ledger(Tier::Ram).pinned, 3 * KB);
    t.submit(job).unwrap();
    t.deliver(op);
    let mut woken = Vec::new();
    for done in t.poll() {
        woken.extend(c.on_completion(done));
    }
    assert_eq!(
        woken,
        vec![Wake {
            waiter: WaiterId(1),
            result: LoadResult::Ok
        }]
    );
    assert_eq!(c.ledger(Tier::Ram).pinned, 0, "pin released with the op");
    c.check_invariants();
}

#[test]
fn reservation_is_exactly_once_and_bounded() {
    let mut c = cat(4 * KB, 0);
    let k = keys(2);
    assert_eq!(
        c.reserve(k[0], Tier::Ram, 5 * KB),
        Err(ReserveError::Insufficient {
            free: 4 * KB,
            wanted: 5 * KB
        })
    );
    c.reserve(k[0], Tier::Ram, 3 * KB).unwrap();
    assert_eq!(
        c.reserve(k[0], Tier::Ram, KB),
        Err(ReserveError::AlreadyPresent)
    );
    assert_eq!(
        c.reserve(k[1], Tier::Ram, 2 * KB),
        Err(ReserveError::Insufficient {
            free: KB,
            wanted: 2 * KB
        })
    );
    assert!(c.release_reservation(&k[0], Tier::Ram));
    assert!(
        !c.release_reservation(&k[0], Tier::Ram),
        "second release is a no-op"
    );
    assert_eq!(c.ledger(Tier::Ram).free, 4 * KB);
    // nvme has zero capacity: disabled
    assert!(matches!(
        c.reserve(k[0], Tier::Nvme, 1),
        Err(ReserveError::Insufficient { .. })
    ));
    c.check_invariants();
}

#[test]
fn begin_store_rejects_oversized_seal_and_unreserved_keys() {
    let mut c = cat(4 * KB, 0);
    let k = keys(1)[0];
    assert!(matches!(
        c.begin_store(k, Tier::Ram, KB, honest(&k, KB)),
        Err(BeginStoreError::NotReserved)
    ));
    c.reserve(k, Tier::Ram, 2 * KB).unwrap();
    assert!(matches!(
        c.begin_store(k, Tier::Ram, 3 * KB, honest(&k, 3 * KB)),
        Err(BeginStoreError::ExceedsReservation {
            reserved: 2048,
            sealed: 3072
        })
    ));
    // nothing moved on the failed begin
    assert_eq!(c.ledger(Tier::Ram).reserved, 2 * KB);
    c.check_invariants();
}

#[test]
fn short_write_never_publishes() {
    let (mut c, mut t) = (cat(8 * KB, 0), FakeTransport::new());
    let k = keys(1)[0];
    c.reserve(k, Tier::Ram, 2 * KB).unwrap();
    let (op, job) = c
        .begin_store(k, Tier::Ram, 2 * KB, honest(&k, 2 * KB))
        .unwrap();
    t.submit(job).unwrap();
    t.deliver_short(op, KB);
    for done in t.poll() {
        c.on_completion(done);
    }
    assert_eq!(c.counters.short_writes, 1);
    assert_eq!(c.ready_bytes(&k, Tier::Ram), None);
    assert_eq!(c.ledger(Tier::Ram).free, 8 * KB, "everything released");
    c.check_invariants();
}

#[test]
fn store_checksum_mismatch_never_publishes() {
    let (mut c, mut t) = (cat(8 * KB, 0), FakeTransport::new());
    let k = keys(1)[0];
    c.reserve(k, Tier::Ram, 2 * KB).unwrap();
    // producer claims a checksum the transport's bytes won't hash to
    let (op, job) = c
        .begin_store(k, Tier::Ram, 2 * KB, Checksum([0xAB; 32]))
        .unwrap();
    t.submit(job).unwrap();
    t.deliver(op);
    for done in t.poll() {
        c.on_completion(done);
    }
    assert_eq!(c.counters.integrity_failures, 1);
    assert_eq!(c.ready_bytes(&k, Tier::Ram), None);
    assert_eq!(c.ledger(Tier::Ram).free, 8 * KB);
    c.check_invariants();
}

#[test]
fn adopt_store_publishes_the_transport_checksum_and_reads_still_verify() {
    // begin_store_adopt (the real-transport mode): no producer checksum -
    // the transport's reported one is published, and at-rest corruption is
    // still caught on the next read because loads verify against it.
    let (mut c, mut t) = (cat(8 * KB, 0), FakeTransport::new());
    let k = keys(1)[0];
    c.reserve(k, Tier::Ram, 2 * KB).unwrap();
    let (op, job) = c.begin_store_adopt(k, Tier::Ram, 2 * KB).unwrap();
    t.submit(job).unwrap();
    t.deliver(op);
    for done in t.poll() {
        c.on_completion(done);
    }
    assert_eq!(c.counters.integrity_failures, 0);
    assert_eq!(
        c.ready_bytes(&k, Tier::Ram),
        Some(2 * KB),
        "adopted and published"
    );
    // corrupt at rest, then load: the adopted checksum must catch it
    let loc = t.locs()[0];
    t.corrupt_at_rest(loc);
    let start = c
        .begin_load(k, Tier::Ram, LoadDst::Gpu, WaiterId(1))
        .unwrap();
    let LoadStart::Started { op, job } = start else {
        panic!("fresh load must start")
    };
    t.submit(job).unwrap();
    t.deliver(op);
    let mut woke = Vec::new();
    for done in t.poll() {
        woke.extend(c.on_completion(done));
    }
    assert_eq!(
        woke,
        vec![Wake {
            waiter: WaiterId(1),
            result: LoadResult::Integrity
        }]
    );
    assert_eq!(c.counters.integrity_failures, 1);
    assert_eq!(c.ready_bytes(&k, Tier::Ram), None, "source poisoned");
    c.check_invariants();
}

#[test]
fn duplicate_completion_is_stale_and_changes_nothing() {
    let (mut c, mut t) = (cat(8 * KB, 0), FakeTransport::new());
    let k = keys(1)[0];
    c.reserve(k, Tier::Ram, 2 * KB).unwrap();
    let (op, job) = c
        .begin_store(k, Tier::Ram, 2 * KB, honest(&k, 2 * KB))
        .unwrap();
    t.submit(job).unwrap();
    t.deliver(op);
    t.duplicate_last();
    let completions = t.poll();
    assert_eq!(completions.len(), 2);
    for done in completions {
        c.on_completion(done);
    }
    assert_eq!(c.counters.stale_completions, 1);
    assert_eq!(c.ready_bytes(&k, Tier::Ram), Some(2 * KB));
    c.check_invariants();
}

#[test]
fn late_completion_cannot_publish_into_a_new_owner() {
    let (mut c, mut t) = (cat(8 * KB, 0), FakeTransport::new());
    let k = keys(1)[0];
    c.reserve(k, Tier::Ram, 2 * KB).unwrap();
    let (op_a, job_a) = c
        .begin_store(k, Tier::Ram, 2 * KB, honest(&k, 2 * KB))
        .unwrap();
    t.submit(job_a).unwrap();
    assert!(c.cancel_store(op_a));
    assert!(!c.cancel_store(op_a), "cancel is idempotent");
    t.cancel(op_a); // catalog contract: caller cancels the transport op

    // a new owner takes the slot
    c.reserve(k, Tier::Ram, 2 * KB).unwrap();
    let (op_b, job_b) = c
        .begin_store(k, Tier::Ram, 2 * KB, honest(&k, 2 * KB))
        .unwrap();
    t.submit(job_b).unwrap();

    // the RACED completion of the cancelled op arrives first - must be stale
    t.deliver(op_a);
    for done in t.poll() {
        c.on_completion(done);
    }
    assert_eq!(c.counters.stale_completions, 1);
    assert_eq!(
        c.ready_bytes(&k, Tier::Ram),
        None,
        "old op must not publish"
    );
    c.check_invariants();

    t.deliver(op_b);
    for done in t.poll() {
        c.on_completion(done);
    }
    assert_eq!(c.ready_bytes(&k, Tier::Ram), Some(2 * KB));
    c.check_invariants();
}

#[test]
fn single_flight_joins_and_wakes_every_waiter() {
    let (mut c, mut t) = (cat(8 * KB, 0), FakeTransport::new());
    let k = keys(1)[0];
    store_ready(&mut c, &mut t, k, Tier::Ram, 2 * KB);
    let LoadStart::Started { op, job } = c
        .begin_load(k, Tier::Ram, LoadDst::Gpu, WaiterId(1))
        .unwrap()
    else {
        panic!()
    };
    let LoadStart::Joined { op: op2 } = c
        .begin_load(k, Tier::Ram, LoadDst::Gpu, WaiterId(2))
        .unwrap()
    else {
        panic!("second load of the same key must join")
    };
    assert_eq!(op, op2);
    assert_eq!(c.counters.single_flight_joins, 1);
    assert_eq!(
        c.ledger(Tier::Ram).pinned,
        2 * KB,
        "one pin per OP, not per waiter"
    );
    t.submit(job).unwrap();
    t.deliver(op);
    let mut woken = Vec::new();
    for done in t.poll() {
        woken.extend(c.on_completion(done));
    }
    woken.sort_by_key(|w| w.waiter.0);
    assert_eq!(
        woken,
        vec![
            Wake {
                waiter: WaiterId(1),
                result: LoadResult::Ok
            },
            Wake {
                waiter: WaiterId(2),
                result: LoadResult::Ok
            },
        ]
    );
    c.check_invariants();
}

#[test]
fn waiter_cancel_is_individual_teardown_is_last_out() {
    let (mut c, mut t) = (cat(8 * KB, 0), FakeTransport::new());
    let k = keys(1)[0];
    store_ready(&mut c, &mut t, k, Tier::Ram, 2 * KB);
    let LoadStart::Started { op, job } = c
        .begin_load(k, Tier::Ram, LoadDst::Gpu, WaiterId(1))
        .unwrap()
    else {
        panic!()
    };
    c.begin_load(k, Tier::Ram, LoadDst::Gpu, WaiterId(2))
        .unwrap();
    t.submit(job).unwrap();

    // first waiter elects recompute - the shared op keeps running
    assert_eq!(c.cancel_waiter(op, WaiterId(1)), CancelOutcome::OpContinues);
    assert_eq!(
        c.cancel_waiter(op, WaiterId(1)),
        CancelOutcome::AlreadyDone,
        "idempotent"
    );
    c.check_invariants();
    assert_eq!(
        c.ledger(Tier::Ram).pinned,
        2 * KB,
        "op still pins its source"
    );

    // last waiter leaves - teardown, caller cancels IO
    assert_eq!(c.cancel_waiter(op, WaiterId(2)), CancelOutcome::CancelIo);
    t.cancel(op);
    assert_eq!(c.ledger(Tier::Ram).pinned, 0, "pin released exactly once");
    c.check_invariants();

    // the raced completion after teardown is stale
    t.deliver(op);
    for done in t.poll() {
        assert!(c.on_completion(done).is_empty(), "no waiters left to wake");
    }
    assert_eq!(c.counters.stale_completions, 1);
    // source is intact and evictable
    assert_eq!(c.evict(&k, Tier::Ram), Ok(2 * KB));
    c.check_invariants();
}

#[test]
fn load_failure_wakes_iofailed_and_source_survives() {
    let (mut c, mut t) = (cat(8 * KB, 0), FakeTransport::new());
    let k = keys(1)[0];
    store_ready(&mut c, &mut t, k, Tier::Ram, 2 * KB);
    let LoadStart::Started { op, job } = c
        .begin_load(k, Tier::Ram, LoadDst::Gpu, WaiterId(7))
        .unwrap()
    else {
        panic!()
    };
    t.submit(job).unwrap();
    t.deliver_failed(op);
    let mut woken = Vec::new();
    for done in t.poll() {
        woken.extend(c.on_completion(done));
    }
    assert_eq!(
        woken,
        vec![Wake {
            waiter: WaiterId(7),
            result: LoadResult::IoFailed
        }]
    );
    assert_eq!(c.counters.io_failures, 1);
    assert_eq!(
        c.ready_bytes(&k, Tier::Ram),
        Some(2 * KB),
        "source replica intact"
    );
    c.check_invariants();
}

#[test]
fn at_rest_corruption_poisons_the_source() {
    let (mut c, mut t) = (cat(8 * KB, 0), FakeTransport::new());
    let k = keys(1)[0];
    store_ready(&mut c, &mut t, k, Tier::Ram, 2 * KB);
    // silent media corruption between store and load
    for loc in t.locs() {
        t.corrupt_at_rest(loc);
    }
    let LoadStart::Started { op, job } = c
        .begin_load(k, Tier::Ram, LoadDst::Gpu, WaiterId(1))
        .unwrap()
    else {
        panic!()
    };
    t.submit(job).unwrap();
    t.deliver(op);
    let mut woken = Vec::new();
    for done in t.poll() {
        woken.extend(c.on_completion(done));
    }
    assert_eq!(
        woken,
        vec![Wake {
            waiter: WaiterId(1),
            result: LoadResult::Integrity
        }]
    );
    assert_eq!(c.counters.integrity_failures, 1);
    // the replica is Bad now: not readable, not double-counted
    assert_eq!(c.ready_bytes(&k, Tier::Ram), None);
    assert!(matches!(
        c.begin_load(k, Tier::Ram, LoadDst::Gpu, WaiterId(2)),
        Err(LoadError::NotReady)
    ));
    assert_eq!(c.ledger(Tier::Ram).ready, 0, "poisoned bytes released");
    c.check_invariants();
    // and the slot can be re-owned
    c.reserve(k, Tier::Ram, 2 * KB).unwrap();
    c.check_invariants();
}

#[test]
fn eviction_respects_pins_and_live_ops() {
    let (mut c, mut t) = (cat(8 * KB, 0), FakeTransport::new());
    let k = keys(1)[0];
    assert_eq!(c.evict(&k, Tier::Ram), Err(EvictError::NotFound));
    c.reserve(k, Tier::Ram, 2 * KB).unwrap();
    assert_eq!(c.evict(&k, Tier::Ram), Err(EvictError::Busy));
    let (op, job) = c
        .begin_store(k, Tier::Ram, 2 * KB, honest(&k, 2 * KB))
        .unwrap();
    assert_eq!(c.evict(&k, Tier::Ram), Err(EvictError::Busy));
    t.submit(job).unwrap();
    t.deliver(op);
    for done in t.poll() {
        c.on_completion(done);
    }
    let LoadStart::Started { op, job } = c
        .begin_load(k, Tier::Ram, LoadDst::Gpu, WaiterId(1))
        .unwrap()
    else {
        panic!()
    };
    assert_eq!(
        c.evict(&k, Tier::Ram),
        Err(EvictError::Pinned),
        "never yank under a reader"
    );
    t.submit(job).unwrap();
    t.deliver(op);
    for done in t.poll() {
        c.on_completion(done);
    }
    assert_eq!(c.evict(&k, Tier::Ram), Ok(2 * KB));
    assert_eq!(c.counters.evictions, 1);
    c.check_invariants();
}

#[test]
fn tier_to_tier_promotion_inherits_generation_and_charges_dst() {
    let (mut c, mut t) = (cat(8 * KB, 8 * KB), FakeTransport::new());
    let k = keys(1)[0];
    store_ready(&mut c, &mut t, k, Tier::Nvme, 2 * KB);

    assert!(matches!(
        c.begin_load(k, Tier::Nvme, LoadDst::Tier(Tier::Nvme), WaiterId(1)),
        Err(LoadError::SameTier)
    ));
    let LoadStart::Started { op, job } = c
        .begin_load(k, Tier::Nvme, LoadDst::Tier(Tier::Ram), WaiterId(1))
        .unwrap()
    else {
        panic!()
    };
    // dst charged up front - a load must never finish into an exhausted pool
    assert_eq!(c.ledger(Tier::Ram).in_flight, 2 * KB);
    assert_eq!(c.ledger(Tier::Nvme).pinned, 2 * KB);
    t.submit(job).unwrap();
    t.deliver(op);
    for done in t.poll() {
        c.on_completion(done);
    }
    c.check_invariants();
    // both tiers now hold the content; NVMe copy remains independently evictable
    assert_eq!(c.ready_bytes(&k, Tier::Ram), Some(2 * KB));
    assert_eq!(c.ready_bytes(&k, Tier::Nvme), Some(2 * KB));
    assert_eq!(c.evict(&k, Tier::Nvme), Ok(2 * KB));
    assert_eq!(c.ready_bytes(&k, Tier::Ram), Some(2 * KB));
    c.check_invariants();
}

#[test]
fn tier_dst_conflicts_are_rejected_up_front() {
    let (mut c, mut t) = (cat(3 * KB, 8 * KB), FakeTransport::new());
    let k = keys(2);
    store_ready(&mut c, &mut t, k[0], Tier::Nvme, 2 * KB);
    store_ready(&mut c, &mut t, k[1], Tier::Nvme, 2 * KB);
    // k0 promotes fine; k1 then lacks dst bytes (3 KB RAM, 2 in flight)
    c.begin_load(k[0], Tier::Nvme, LoadDst::Tier(Tier::Ram), WaiterId(1))
        .unwrap();
    assert!(matches!(
        c.begin_load(k[1], Tier::Nvme, LoadDst::Tier(Tier::Ram), WaiterId(2)),
        Err(LoadError::DstInsufficient {
            free: 1024,
            wanted: 2048
        })
    ));
    // occupied dst (in flight counts as occupied)
    assert!(
        matches!(
            c.begin_load(k[0], Tier::Nvme, LoadDst::Tier(Tier::Ram), WaiterId(3)),
            Ok(LoadStart::Joined { .. })
        ),
        "same (key,dst) is the single-flight JOIN, not a conflict"
    );
    c.check_invariants();
}

#[test]
fn cancelled_tier_promotion_releases_dst_and_late_completion_is_stale() {
    let (mut c, mut t) = (cat(8 * KB, 8 * KB), FakeTransport::new());
    let k = keys(1)[0];
    store_ready(&mut c, &mut t, k, Tier::Nvme, 2 * KB);
    let LoadStart::Started { op, job } = c
        .begin_load(k, Tier::Nvme, LoadDst::Tier(Tier::Ram), WaiterId(1))
        .unwrap()
    else {
        panic!()
    };
    t.submit(job).unwrap();
    assert_eq!(c.cancel_waiter(op, WaiterId(1)), CancelOutcome::CancelIo);
    t.cancel(op);
    assert_eq!(
        c.ledger(Tier::Ram).in_flight,
        0,
        "dst reservation released exactly once"
    );
    assert_eq!(c.ledger(Tier::Nvme).pinned, 0);
    // raced completion delivers anyway
    t.deliver(op);
    for done in t.poll() {
        assert!(c.on_completion(done).is_empty());
    }
    assert_eq!(c.counters.stale_completions, 1);
    assert_eq!(
        c.ready_bytes(&k, Tier::Ram),
        None,
        "cancelled promotion must not publish"
    );
    c.check_invariants();
}

#[test]
fn mark_bad_respects_pins() {
    let (mut c, mut t) = (cat(8 * KB, 0), FakeTransport::new());
    let k = keys(1)[0];
    store_ready(&mut c, &mut t, k, Tier::Ram, 2 * KB);
    let LoadStart::Started { op, job } = c
        .begin_load(k, Tier::Ram, LoadDst::Gpu, WaiterId(1))
        .unwrap()
    else {
        panic!()
    };
    assert_eq!(c.mark_bad(&k, Tier::Ram), Err(EvictError::Pinned));
    t.submit(job).unwrap();
    t.deliver(op);
    for done in t.poll() {
        c.on_completion(done);
    }
    assert_eq!(c.mark_bad(&k, Tier::Ram), Ok(()));
    assert_eq!(c.counters.bad_marks, 1);
    assert_eq!(
        c.evict(&k, Tier::Ram),
        Ok(0),
        "Bad marker evicts as zero bytes"
    );
    c.check_invariants();
}

// ---------------------------------------------------------------------------
// 2. exhaustive interleavings
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
enum Ev {
    Deliver,
    CancelW1,
    CancelW2,
    EvictSrc,
    JoinW3,
}

fn permutations<T: Clone>(v: &[T]) -> Vec<Vec<T>> {
    if v.len() <= 1 {
        return vec![v.to_vec()];
    }
    let mut out = Vec::new();
    for i in 0..v.len() {
        let mut rest = v.to_vec();
        let x = rest.remove(i);
        for mut p in permutations(&rest) {
            p.insert(0, x.clone());
            out.push(p);
        }
    }
    out
}

/// Every ordering of {complete, cancel-waiter-1, cancel-waiter-2, evict-src,
/// late-join-waiter-3} against one in-flight two-waiter load. 120 orderings;
/// after every event the invariants must hold, and terminally every waiter
/// must have been woken exactly once or have cancelled itself.
#[test]
fn load_race_interleavings_exhaustive() {
    let events = [
        Ev::Deliver,
        Ev::CancelW1,
        Ev::CancelW2,
        Ev::EvictSrc,
        Ev::JoinW3,
    ];
    for perm in permutations(&events) {
        let (mut c, mut t) = (cat(8 * KB, 0), FakeTransport::new());
        let k = keys(1)[0];
        store_ready(&mut c, &mut t, k, Tier::Ram, 2 * KB);
        let LoadStart::Started { op, job } = c
            .begin_load(k, Tier::Ram, LoadDst::Gpu, WaiterId(1))
            .unwrap()
        else {
            panic!()
        };
        c.begin_load(k, Tier::Ram, LoadDst::Gpu, WaiterId(2))
            .unwrap();
        t.submit(job).unwrap();

        // waiters not yet resolved (woken or self-cancelled)
        let mut unresolved = vec![WaiterId(1), WaiterId(2)];
        let mut wakes: Vec<Wake> = Vec::new();
        for ev in &perm {
            match ev {
                Ev::Deliver => {
                    t.deliver(op);
                    for done in t.poll() {
                        wakes.extend(c.on_completion(done));
                    }
                }
                Ev::CancelW1 | Ev::CancelW2 => {
                    let w = if *ev == Ev::CancelW1 {
                        WaiterId(1)
                    } else {
                        WaiterId(2)
                    };
                    match c.cancel_waiter(op, w) {
                        CancelOutcome::OpContinues => unresolved.retain(|x| *x != w),
                        CancelOutcome::CancelIo => {
                            unresolved.retain(|x| *x != w);
                            t.cancel(op);
                        }
                        // op already resolved: the waiter must have been woken
                        CancelOutcome::AlreadyDone => {
                            assert!(
                                wakes.iter().any(|x| x.waiter == w) || !unresolved.contains(&w),
                                "AlreadyDone for a waiter that was never woken ({perm:?})"
                            );
                        }
                    }
                }
                Ev::EvictSrc => {
                    // legal only when unpinned - either way, consistent after
                    let _ = c.evict(&k, Tier::Ram);
                }
                Ev::JoinW3 => {
                    // a third request shows up mid-race; every outcome is legal
                    // (fresh load, join, or NotReady after eviction) - but it
                    // must resolve by the end like the others.
                    match c.begin_load(k, Tier::Ram, LoadDst::Gpu, WaiterId(3)) {
                        Ok(LoadStart::Joined { .. }) => unresolved.push(WaiterId(3)),
                        Ok(LoadStart::Started { op: op3, job }) => {
                            t.submit(job).unwrap();
                            t.deliver(op3);
                            for done in t.poll() {
                                wakes.extend(c.on_completion(done));
                            }
                        }
                        Err(_) => {}
                    }
                }
            }
            c.check_invariants();
        }
        // drain: anything still pending resolves now
        t.deliver_all();
        for done in t.poll() {
            wakes.extend(c.on_completion(done));
        }
        c.check_invariants();
        for w in unresolved {
            assert_eq!(
                wakes.iter().filter(|x| x.waiter == w).count(),
                1,
                "waiter {w:?} must be woken exactly once ({perm:?}); wakes: {wakes:?}"
            );
        }
        // no waiter is ever woken twice
        for w in [WaiterId(1), WaiterId(2), WaiterId(3)] {
            assert!(
                wakes.iter().filter(|x| x.waiter == w).count() <= 1,
                "waiter {w:?} woken twice ({perm:?})"
            );
        }
    }
}

/// Every ordering of {deliver, cancel_store, evict, re-reserve} against one
/// in-flight store: the slot must end consistent, the ledger balanced, and a
/// cancelled op's completion stale.
#[test]
fn store_race_interleavings_exhaustive() {
    #[derive(Clone, Copy, Debug, PartialEq)]
    enum SEv {
        Deliver,
        Cancel,
        Evict,
        Rereserve,
    }
    let events = [SEv::Deliver, SEv::Cancel, SEv::Evict, SEv::Rereserve];
    for perm in permutations(&events) {
        let (mut c, mut t) = (cat(8 * KB, 0), FakeTransport::new());
        let k = keys(1)[0];
        c.reserve(k, Tier::Ram, 2 * KB).unwrap();
        let (op, job) = c
            .begin_store(k, Tier::Ram, 2 * KB, honest(&k, 2 * KB))
            .unwrap();
        t.submit(job).unwrap();
        let mut cancelled = false;
        for ev in &perm {
            match ev {
                SEv::Deliver => {
                    t.deliver(op);
                    for done in t.poll() {
                        c.on_completion(done);
                    }
                }
                SEv::Cancel => {
                    cancelled = c.cancel_store(op);
                    if cancelled {
                        t.cancel(op);
                    }
                }
                SEv::Evict => {
                    let _ = c.evict(&k, Tier::Ram);
                }
                SEv::Rereserve => {
                    let _ = c.reserve(k, Tier::Ram, KB);
                }
            }
            c.check_invariants();
        }
        t.deliver_all();
        for done in t.poll() {
            c.on_completion(done);
        }
        c.check_invariants();
        if cancelled {
            // deliver-after-cancel must have been stale in this permutation
            assert!(c.counters.stale_completions <= 1);
        }
        let l = c.ledger(Tier::Ram);
        assert_eq!(l.free + l.reserved + l.in_flight + l.ready, l.capacity);
    }
}

// ---------------------------------------------------------------------------
// 3. seeded property fuzz
// ---------------------------------------------------------------------------

/// Random op streams against small tiers, invariants after every event, waiter
/// wake-exactly-once bookkeeping across the whole run. Deterministic seeds so
/// a failure replays.
#[test]
fn fuzz_random_op_streams_hold_invariants() {
    for seed in 0u64..6 {
        fuzz_one(seed, 4000);
    }
}

fn fuzz_one(seed: u64, steps: u32) {
    use std::collections::{BTreeMap, HashMap};
    let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(seed);
    let mut next = move |n: u64| -> u64 { rng.next_u64() % n };

    let ks = keys(6);
    let (mut c, mut t) = (cat(16 * KB, 16 * KB), FakeTransport::new());
    // ops we started and may still be pending in the transport
    let mut live_stores: Vec<OpId> = Vec::new();
    let mut live_loads: Vec<OpId> = Vec::new();
    // waiter -> op they are parked on; removed when woken or self-cancelled.
    // BTreeMap: iteration order is part of the event stream, so it must be
    // deterministic for the seed to replay.
    let mut parked: BTreeMap<WaiterId, OpId> = BTreeMap::new();
    let mut woken: HashMap<WaiterId, u32> = HashMap::new();
    let mut next_waiter = 0u64;

    for step in 0..steps {
        let ctx = format!("seed {seed} step {step}");
        match next(14) {
            0 | 1 => {
                // reserve a random key with a random bound
                let k = ks[next(6) as usize];
                let tier = if next(2) == 0 { Tier::Ram } else { Tier::Nvme };
                let _ = c.reserve(k, tier, (1 + next(4)) * KB);
            }
            2 => {
                let k = ks[next(6) as usize];
                let tier = if next(2) == 0 { Tier::Ram } else { Tier::Nvme };
                let _ = c.release_reservation(&k, tier);
            }
            3 | 4 => {
                // begin_store on a random key (only succeeds if Reserved);
                // occasionally lie about the checksum to exercise integrity
                let k = ks[next(6) as usize];
                let tier = if next(2) == 0 { Tier::Ram } else { Tier::Nvme };
                let bytes = (1 + next(3)) * KB;
                let sum = if next(8) == 0 {
                    Checksum([0x5A; 32])
                } else {
                    honest(&k, bytes)
                };
                if let Ok((op, job)) = c.begin_store(k, tier, bytes, sum) {
                    t.submit(job).unwrap();
                    live_stores.push(op);
                }
            }
            5 => {
                if !live_stores.is_empty() {
                    let op = live_stores[next(live_stores.len() as u64) as usize];
                    if c.cancel_store(op) {
                        t.cancel(op);
                    }
                }
            }
            6 | 7 => {
                let k = ks[next(6) as usize];
                let src = if next(2) == 0 { Tier::Ram } else { Tier::Nvme };
                let dst = match next(3) {
                    0 => LoadDst::Tier(Tier::Ram),
                    1 => LoadDst::Tier(Tier::Nvme),
                    _ => LoadDst::Gpu,
                };
                let w = WaiterId(next_waiter);
                match c.begin_load(k, src, dst, w) {
                    Ok(LoadStart::Started { op, job }) => {
                        t.submit(job).unwrap();
                        live_loads.push(op);
                        parked.insert(w, op);
                        next_waiter += 1;
                    }
                    Ok(LoadStart::Joined { op }) => {
                        parked.insert(w, op);
                        next_waiter += 1;
                    }
                    Err(_) => {}
                }
            }
            8 => {
                // a parked waiter elects recompute
                if let Some((&w, &op)) = parked.iter().next() {
                    match c.cancel_waiter(op, w) {
                        CancelOutcome::OpContinues => {
                            parked.remove(&w);
                        }
                        CancelOutcome::CancelIo => {
                            parked.remove(&w);
                            t.cancel(op);
                        }
                        CancelOutcome::AlreadyDone => {
                            panic!("{ctx}: parked waiter {w:?} got AlreadyDone - lost wake");
                        }
                    }
                }
            }
            9 => {
                let k = ks[next(6) as usize];
                let tier = if next(2) == 0 { Tier::Ram } else { Tier::Nvme };
                let _ = c.evict(&k, tier);
            }
            10 => {
                let k = ks[next(6) as usize];
                let tier = if next(2) == 0 { Tier::Ram } else { Tier::Nvme };
                let _ = c.mark_bad(&k, tier);
            }
            11 => {
                // deliver a random pending op - success, short, or failure
                let pending = t.pending_ops();
                if !pending.is_empty() {
                    let op = pending[next(pending.len() as u64) as usize];
                    match next(5) {
                        0 => t.deliver_failed(op),
                        1 => t.deliver_short(op, 1),
                        2 => t.drop_op(op),
                        _ => t.deliver(op),
                    }
                }
            }
            12 => {
                // corrupt something at rest
                let locs = t.locs();
                if !locs.is_empty() {
                    t.corrupt_at_rest(locs[next(locs.len() as u64) as usize]);
                }
            }
            _ => {
                // poll + feed completions; occasionally duplicate one first
                if next(4) == 0 {
                    t.duplicate_last();
                }
                for done in t.poll() {
                    for wake in c.on_completion(done) {
                        *woken.entry(wake.waiter).or_default() += 1;
                        assert!(
                            parked.remove(&wake.waiter).is_some(),
                            "{ctx}: wake for a waiter not parked (double wake?)"
                        );
                    }
                }
            }
        }
        c.check_invariants();
        live_stores.retain(|op| t.pending_ops().contains(op));
        live_loads.retain(|op| t.pending_ops().contains(op));
    }

    // drain everything and settle
    t.deliver_all();
    for done in t.poll() {
        for wake in c.on_completion(done) {
            *woken.entry(wake.waiter).or_default() += 1;
            assert!(
                parked.remove(&wake.waiter).is_some(),
                "seed {seed}: final drain double wake"
            );
        }
    }
    c.check_invariants();
    for (w, n) in &woken {
        assert_eq!(*n, 1, "seed {seed}: waiter {w:?} woken {n} times");
    }
    // waiters still parked here were on ops whose completions the fuzz
    // deliberately DROPPED (lost interrupt) - the Phase-0 contract is that
    // nothing corrupts; catalog-level timeouts arrive with the durable tier.
    for (&w, &op) in &parked {
        assert!(
            c.has_op(op),
            "seed {seed}: waiter {w:?} parked on op {op:?} that no longer exists - lost wake"
        );
    }
}
