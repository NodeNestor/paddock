//! Stage-B gate for the nemotron continuous-batching substrate (
//! `enable_batch_impl` must build the
//! attention-layer pool + mamba slot arenas within granite's budget/floor
//! contract, and the admission/release plumbing must move pool blocks the
//! way the scheduler will assume. No forward ticks here - those are stage C
//! and get their own serial-parity gate.

mod common;

use paddock_engine::generator::Generator;
use paddock_engine::gpu_model::nemotron::GpuNemotron;

const CKPT_ENV: &str = "NEMOTRON_NVFP4_DIR";
const CKPT_DIR: &str = "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4";
const ORACLE: &str = "/models/nemotron-battery/oracle/decoder-oracle.json";
const MAX_CTX: usize = 2048;
const PROMPT_LEN: usize = 700;
const GREEDY_STEPS: usize = 24;

fn argmax(l: &[f32]) -> u32 {
    let mut bi = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &v) in l.iter().enumerate() {
        if v > bv {
            bv = v;
            bi = i;
        }
    }
    bi as u32
}

fn oracle_prompt(n: usize) -> Option<Vec<u32>> {
    let path = std::env::var("NEMOTRON_ORACLE").unwrap_or_else(|_| ORACLE.into());
    let raw = std::fs::read(&path).ok()?;
    let oracle: serde_json::Value = serde_json::from_slice(&raw).ok()?;
    let seed: Vec<u32> = oracle["prompt_ids"]
        .as_array()?
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    Some((0..n).map(|i| seed[i % seed.len()]).collect())
}

#[test]
fn enable_batch_builds_pool_and_arenas() {
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    if !exec.has_paged_kv()
        || !exec.has_mamba2_batch()
        || !exec.has_nvf4_gemv_batch()
        || !exec.has_nvf4_ckpt()
    {
        common::missing("pack lacks the nemotron batch kernel set (cc != 12.0?)");
        return;
    }
    let Some(dir) = common::model_dir(CKPT_ENV, &[CKPT_DIR]) else {
        return;
    };
    let mut model = GpuNemotron::load_dir(exec, &dir, MAX_CTX).expect("load");

    let slots = model.batch_enable_probe(4).expect("enable_batch");
    assert_eq!(slots, 4, "20 GiB model on this card must seat 4 slots");
    let (free, capacity) = model.batch_pool_stats().expect("batch populated");
    assert!(capacity >= 256, "pool below the floor: {capacity}");
    // ceiling = addressable (slots × bps) + the stage-D radix retention slack
    assert!(
        capacity <= 4 * MAX_CTX.div_ceil(16) + 512,
        "pool over the addressable + retention ceiling: {capacity}"
    );
    assert_eq!(free, capacity, "fresh pool must be all-free");

    // admission backs the whole prompt up front (100 rows = 7 blocks)
    model.batch_admit_probe(0, 100).expect("admit");
    let (f, _) = model.batch_pool_stats().unwrap();
    assert_eq!(capacity - f, 100usize.div_ceil(16));

    // slot reuse: the old sequence's blocks return before the new backing
    model.batch_admit_probe(0, 20).expect("re-admit");
    let (f, _) = model.batch_pool_stats().unwrap();
    assert_eq!(capacity - f, 2);

    // release-on-completion frees exactly the idle slots' blocks
    model.batch_admit_probe(3, 33).expect("admit slot 3");
    model.release_inactive_slots(&[false, true, false, true]);
    let (f, _) = model.batch_pool_stats().unwrap();
    assert_eq!(capacity - f, 3, "slot 0 released, slot 3 (3 blocks) kept");
    model.release_inactive_slots(&[false; 4]);
    let (f, _) = model.batch_pool_stats().unwrap();
    assert_eq!(f, capacity, "all slots released -> pool all-free");

    // refusals: out-of-range slot, empty prompt, over-context prompt
    assert!(model.batch_admit_probe(4, 10).is_err());
    assert!(model.batch_admit_probe(0, 0).is_err());
    assert!(model.batch_admit_probe(0, MAX_CTX + 1).is_err());

    // accounting is visible through the Generator surface
    let kv = model.kv_mem_bytes().expect("kv accounting");
    assert!(kv > 0);
    assert_eq!(model.pool_free_blocks(), Some(capacity));
}

/// Stage-C parity gate: the batch lane's c1 output must reproduce the
/// serial parity spine through the same gate class the bulk-prefill gate
/// uses (exactness is impossible by construction - the paged wmma prefill
/// tile and the W8A8 dynamic-act projections change summation order - so
/// the gates are boundary top-1, a tight mean-|delta|/rms band, and an
/// identical greedy continuation on the r=1 decode graph). Plus a
/// coalesced c3 wave + batched-decode + mixed-tick smoke: the slot serving
/// the same prompt must land the same boundary pick.
#[test]
fn batch_lane_c1_matches_serial() {
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    if !exec.has_paged_kv()
        || !exec.has_mamba2_batch()
        || !exec.has_nvf4_gemv_batch()
        || !exec.has_nvf4_ckpt()
        || !exec.has_nemotron_prefill_f8()
    {
        common::missing("pack lacks the nemotron batch kernel set (cc != 12.0?)");
        return;
    }
    let Some(dir) = common::model_dir(CKPT_ENV, &[CKPT_DIR]) else {
        return;
    };
    let Some(prompt) = oracle_prompt(PROMPT_LEN) else {
        common::missing("no oracle dump for prompt ids");
        return;
    };
    let mut model = GpuNemotron::load_dir(exec, &dir, MAX_CTX).expect("load");

    // ---- serial reference: token-by-token walk + greedy continuation -----
    model.reset();
    let mut logits_s = Vec::new();
    for &t in &prompt {
        logits_s = model.forward(t).expect("serial forward");
    }
    let mut ids_s = Vec::with_capacity(GREEDY_STEPS);
    let mut l = logits_s.clone();
    for _ in 0..GREEDY_STEPS {
        let tok = argmax(&l);
        ids_s.push(tok);
        l = model.forward(tok).expect("serial decode");
    }

    // ---- batch lane: slot prefill + r=1 decode graph ----------------------
    let slots = model.batch_enable_probe(4).expect("enable_batch");
    assert_eq!(slots, 4);
    let logits_b = model.forward_prefill(0, &prompt).expect("batch prefill");
    let mut ids_b = Vec::with_capacity(GREEDY_STEPS);
    let mut l = logits_b.clone();
    for i in 0..GREEDY_STEPS {
        let tok = argmax(&l);
        ids_b.push(tok);
        l = model
            .forward_batch(&[tok], &[(PROMPT_LEN + i) as u32])
            .expect("batch decode");
    }

    let rms = (logits_s
        .iter()
        .map(|v| (*v as f64) * (*v as f64))
        .sum::<f64>()
        / logits_s.len() as f64)
        .sqrt();
    let mean_abs = logits_s
        .iter()
        .zip(logits_b.iter())
        .map(|(a, b)| (*a as f64 - *b as f64).abs())
        .sum::<f64>()
        / logits_s.len() as f64;
    println!(
        "boundary: mean |d| {mean_abs:.5} vs rms {rms:.3} ({:.2}%)",
        100.0 * mean_abs / rms
    );
    println!("greedy serial: {ids_s:?}\ngreedy batch:  {ids_b:?}");
    assert_eq!(
        argmax(&logits_s),
        argmax(&logits_b),
        "top-1 flipped at the prompt boundary"
    );
    assert!(
        mean_abs / rms.max(1e-3) < 0.10,
        "boundary logits drifted structurally: mean |delta| {mean_abs:.4} vs rms {rms:.3}"
    );
    assert_eq!(ids_s, ids_b, "greedy continuation diverged");

    // ---- coalesced wave (c3): same prompt in slot 1 + two different ------
    let p_short: Vec<u32> = prompt[..97].to_vec();
    let p_mid: Vec<u32> = prompt[..333].to_vec();
    let items = vec![(1usize, prompt.clone()), (2usize, p_short), (3usize, p_mid)];
    let out = model.forward_prefill_batch(&items).expect("coalesced wave");
    assert_eq!(out.len(), 3);
    assert_eq!(
        argmax(&out[0]),
        ids_b[0],
        "same prompt through the coalesced wave flipped its boundary pick"
    );

    // ---- batched decode smoke (r=3, identity slots 0..2) ------------------
    // Row 0 = slot 0's next unconsumed token (a token must never replay
    // through the tick: the mamba state advance is not idempotent). Rows
    // 1..2 = slots 1..2's first decode off their wave boundary picks.
    let toks: Vec<u32> = out.iter().map(|l| argmax(l)).collect();
    let toks3 = [argmax(&l), toks[0], toks[1]];
    let pos3 = [(PROMPT_LEN + GREEDY_STEPS) as u32, prompt.len() as u32, 97];
    let logits3 = model.forward_batch(&toks3, &pos3).expect("r=3 decode");
    let vocab = logits3.len() / 3;
    assert_eq!(vocab * 3, logits3.len());
    println!(
        "r=3 decode: slot0 pick {} slot1 pick {} slot2 pick {}",
        argmax(&logits3[..vocab]),
        argmax(&logits3[vocab..2 * vocab]),
        argmax(&logits3[2 * vocab..])
    );

    // ---- mixed tick: slot 2 re-prefills chunked while slot 0 decodes ------
    let p_new: Vec<u32> = prompt[..600].to_vec();
    model.prefill_begin(2, p_new).expect("prefill_begin");
    let mut finished = Vec::new();
    let mut tick = 0usize;
    let mut dec_tok = argmax(&logits3[..vocab]);
    let mut dec_pos = (PROMPT_LEN + GREEDY_STEPS + 1) as u32;
    while finished.is_empty() {
        assert!(tick < 8, "chunked prefill never finished");
        let (step, fin) = model
            .forward_mixed_sampled(
                &[(0usize, dec_tok, dec_pos)],
                usize::MAX,
                &[paddock_engine::generator::RowSample::Device(
                    paddock_engine::sampler::DevicePlan::Greedy,
                )],
                &[],
            )
            .expect("mixed tick");
        assert_eq!(step.ids.len(), 1, "one decode row per tick");
        dec_tok = step.ids[0];
        dec_pos += 1;
        finished = fin;
        tick += 1;
    }
    // 600 rows at 512-row chunks = 2 ticks cold; the stage-D prefix cache
    // may resume off the earlier same-prefix prompts and finish in 1
    assert!(tick <= 2, "600-token prompt took {tick} mixed ticks");
    let (slot, fs, n) = &finished[0];
    assert_eq!((*slot, *n), (2usize, 600usize));
    match fs {
        paddock_engine::generator::FinishSample::Logits(l) => {
            // 600 tokens of the same text: its boundary pick is a fresh
            // computation on the prefill class - just sanity it's a valid id
            assert!((argmax(l) as usize) < vocab);
        }
        paddock_engine::generator::FinishSample::Sampled(id) => {
            assert!((*id as usize) < vocab);
        }
    }
    println!("mixed tick: finished slot 2 (600 rows) on tick {tick}, decode kept flowing");
}

/// Stage-D gate: the radix prefix cache + mamba state-snapshot resume.
/// ckpt_cuts(700) = [672, 688], so an exact repeat must resume at 688 (the
/// deepest checkpoint), a prompt sharing only 680 tokens must resume at 672,
/// and a resumed prefill must reproduce the cold boundary pick + greedy
/// stream (adopted KV blocks are the same physical bytes; the restored
/// state is the staged snapshot of the cold run's own state at the cut).
#[test]
fn prefix_cache_resumes_with_state_snapshot() {
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    if !exec.has_paged_kv()
        || !exec.has_mamba2_batch()
        || !exec.has_nvf4_gemv_batch()
        || !exec.has_nvf4_ckpt()
        || !exec.has_nemotron_prefill_f8()
    {
        common::missing("pack lacks the nemotron batch kernel set (cc != 12.0?)");
        return;
    }
    let Some(dir) = common::model_dir(CKPT_ENV, &[CKPT_DIR]) else {
        return;
    };
    let Some(prompt) = oracle_prompt(PROMPT_LEN) else {
        common::missing("no oracle dump for prompt ids");
        return;
    };
    let mut model = GpuNemotron::load_dir(exec, &dir, MAX_CTX).expect("load");
    assert_eq!(model.batch_enable_probe(4).expect("enable"), 4);

    let greedy8 = |model: &mut GpuNemotron, slot: usize, first: &[f32], pos0: usize| {
        let mut ids = Vec::new();
        let mut tok = argmax(first);
        let mut pos = pos0 as u32;
        for _ in 0..8 {
            ids.push(tok);
            let (step, _) = model
                .forward_mixed_sampled(
                    &[(slot, tok, pos)],
                    usize::MAX,
                    &[paddock_engine::generator::RowSample::Device(
                        paddock_engine::sampler::DevicePlan::Greedy,
                    )],
                    &[],
                )
                .expect("decode tick");
            tok = step.ids[0];
            pos += 1;
        }
        ids
    };

    // cold prefill (slot 0): plants checkpoints at 672 and 688
    let cold = model.forward_prefill(0, &prompt).expect("cold prefill");
    assert_eq!(model.take_prefill_reused(0), 0, "first sight cannot reuse");
    let cold_ids = greedy8(&mut model, 0, &cold, PROMPT_LEN);

    // exact repeat (slot 1): resumes at the deepest checkpoint
    let warm = model.forward_prefill(1, &prompt).expect("warm prefill");
    let reused = model.take_prefill_reused(1);
    assert_eq!(
        reused, 688,
        "exact repeat must resume at the deep checkpoint"
    );
    let rms =
        (cold.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / cold.len() as f64).sqrt();
    let mean_abs = cold
        .iter()
        .zip(warm.iter())
        .map(|(a, b)| (*a as f64 - *b as f64).abs())
        .sum::<f64>()
        / cold.len() as f64;
    println!("resume boundary: mean |d| {mean_abs:.6} vs rms {rms:.3}");
    assert_eq!(
        argmax(&cold),
        argmax(&warm),
        "resume flipped the boundary pick"
    );
    // The 12-row resumed tail crosses pd_matvec_f32_batch's r=16 rung
    // boundary (the MoE router's lane-strided vs thread-strided sums - a
    // sanctioned order change), so 23 MoE layers compound a
    // smooth few-percent drift (measured 4.5% of rms). The band gates that
    // it stays in the sanctioned class; the STRUCTURAL gate - a wrong
    // snapshot, a stale window, a misaligned resume - is the greedy-stream
    // equality below, which numeric reorder does not break.
    assert!(
        mean_abs / rms.max(1e-3) < 0.10,
        "resumed logits drifted past the reorder class: mean |d| {mean_abs:.5} vs rms {rms:.3}"
    );
    let warm_ids = greedy8(&mut model, 1, &warm, PROMPT_LEN);
    assert_eq!(cold_ids, warm_ids, "greedy stream diverged after resume");

    // divergent tail (slot 2): shares 680 tokens -> page 43 differs -> the
    // 672 checkpoint is the deepest reachable one
    let mut p2 = prompt[..680].to_vec();
    p2.extend((0..20).map(|i| 2000 + i as u32));
    model.forward_prefill(2, &p2).expect("divergent prefill");
    assert_eq!(
        model.take_prefill_reused(2),
        672,
        "shared-680 prompt must resume at 672"
    );

    // chunked lane (slot 3): same prompt through prefill_begin + mixed ticks
    model
        .prefill_begin(3, prompt.clone())
        .expect("prefill_begin");
    let mut fin = Vec::new();
    let mut ticks = 0;
    while fin.is_empty() {
        assert!(ticks < 4, "resumed chunked prefill should finish fast");
        let (_, f) = model
            .forward_mixed_sampled(&[], usize::MAX, &[], &[])
            .expect("mixed tick");
        fin = f;
        ticks += 1;
    }
    assert_eq!(
        model.take_prefill_reused(3),
        688,
        "chunked lane must resume too"
    );
    let (slot, fs, n) = &fin[0];
    assert_eq!((*slot, *n), (3usize, PROMPT_LEN));
    if let paddock_engine::generator::FinishSample::Logits(l) = fs {
        assert_eq!(
            argmax(l),
            argmax(&cold),
            "chunked resume flipped the boundary pick"
        );
    }
    println!(
        "prefix cache: exact repeat 688, divergent-tail 672, chunked lane resumed in {ticks} tick(s)"
    );
}

/// Stage-E fp8-KV smoke: the batch lane serves the checkpoint's own KV spec
/// through the paged pool (the v4 tile's raw-e4m3 hd128 G=16 arm + the fp8
/// paged decode walk). fp8 KV is a LOSSY class, so the gates are the
/// boundary top-1 (a flip there would be beyond the class) + halved KV
/// accounting; the greedy stream is reported, not asserted (near-ties may
/// flip - the serve-level arbiter compare judges the class).
#[test]
fn fp8_kv_batch_lane_smoke() {
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    if !exec.has_paged_kv()
        || !exec.has_mamba2_batch()
        || !exec.has_nvf4_gemv_batch()
        || !exec.has_nvf4_ckpt()
        || !exec.has_nemotron_prefill_f8()
    {
        common::missing("pack lacks the nemotron batch kernel set (cc != 12.0?)");
        return;
    }
    let Some(dir) = common::model_dir(CKPT_ENV, &[CKPT_DIR]) else {
        return;
    };
    let Some(prompt) = oracle_prompt(PROMPT_LEN) else {
        common::missing("no oracle dump for prompt ids");
        return;
    };
    let mut model = GpuNemotron::load_dir(exec, &dir, MAX_CTX).expect("load");

    // f16 reference on the same batch lane
    assert_eq!(model.batch_enable_probe(4).expect("enable f16"), 4);
    let l16 = model.forward_prefill(0, &prompt).expect("f16 prefill");
    let kv16 = model.kv_mem_bytes().expect("f16 kv accounting");
    let mut ids16 = Vec::new();
    let mut l = l16.clone();
    for i in 0..8 {
        let tok = argmax(&l);
        ids16.push(tok);
        l = model
            .forward_batch(&[tok], &[(PROMPT_LEN + i) as u32])
            .expect("f16 decode");
    }

    // flip to the checkpoint's own KV spec and rebuild the lane
    model.set_kv_dtype(paddock_engine::gpu::KvDtype::Fp8E4m3);
    assert_eq!(model.batch_enable_probe(4).expect("enable fp8"), 4);
    let l8 = model.forward_prefill(0, &prompt).expect("fp8 prefill");
    let kv8 = model.kv_mem_bytes().expect("fp8 kv accounting");
    let mut ids8 = Vec::new();
    let mut l = l8.clone();
    for i in 0..8 {
        let tok = argmax(&l);
        ids8.push(tok);
        l = model
            .forward_batch(&[tok], &[(PROMPT_LEN + i) as u32])
            .expect("fp8 decode");
    }

    let rms =
        (l16.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / l16.len() as f64).sqrt();
    let mean_abs = l16
        .iter()
        .zip(l8.iter())
        .map(|(a, b)| (*a as f64 - *b as f64).abs())
        .sum::<f64>()
        / l16.len() as f64;
    let same = ids16
        .iter()
        .zip(ids8.iter())
        .filter(|(a, b)| a == b)
        .count();
    println!(
        "fp8 KV: boundary mean |d| {mean_abs:.4} vs rms {rms:.3} ({:.2}%), greedy8 {same}/8 match",
        100.0 * mean_abs / rms
    );
    println!("f16: {ids16:?}\nfp8: {ids8:?}");
    assert_eq!(
        argmax(&l16),
        argmax(&l8),
        "fp8 KV flipped the boundary top-1"
    );
    // the paged KV part of the accounting must halve (arenas stay f32);
    // pool block count is equal by construction (same max_batch/max_ctx)
    assert!(
        kv8 < kv16,
        "fp8 KV accounting did not shrink: {kv8} vs {kv16}"
    );
}
