//! Gate for `whisper_ts_rules`  - whisper's `ApplyTimestampRules`
//! as a device-side logit filter.
//!
//! This kernel decides which TOKENS are LEGAL, so a bug in it does not show up
//! as noise: it shows up as a transcript with no times in it, or with times
//! that run backwards. The oracle is the reference implementation
//! (openai/whisper `decoding.py`), rule by rule, re-expressed on the host and
//! diffed against the device result over the whole 51866-token row.
//!
//! Light (no model load).

mod common;

use paddock_engine::gpu::GpuExecutor;
use paddock_engine::gpu_model::whisper::{TimeScale, ts_state};

// kb/nb/Røst all share this layout, checked against the converted GGUFs
const VOCAB: usize = 51866;
const EOT: u32 = 50257;
const NO_TS: u32 = 50364;
const TS_BEGIN: u32 = 50365;
const MAX_INIT: u32 = 50; // 1.0 s at 0.02 s a step

fn scale() -> TimeScale {
    TimeScale {
        begin: TS_BEGIN,
        precision: 0.02,
        window_s: 30.0,
    }
}

fn det(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / (1u64 << 31) as f32) * 8.0 - 4.0
        })
        .collect()
}

/// The reference filter, host side. Deliberately a transliteration of
/// `ApplyTimestampRules.apply` rather than a tidied-up version - the point is
/// to be able to read it against the original.
fn host_rules(row: &mut [f32], sampled: &[u32]) {
    let ts = |t: u32| t >= TS_BEGIN;
    row[NO_TS as usize] = f32::NEG_INFINITY;

    let last_was_ts = sampled.last().is_some_and(|&t| ts(t));
    let penult_was_ts = sampled.len() < 2 || ts(sampled[sampled.len() - 2]);
    if last_was_ts {
        if penult_was_ts {
            for v in row.iter_mut().skip(TS_BEGIN as usize) {
                *v = f32::NEG_INFINITY;
            }
        } else {
            for v in row.iter_mut().take(EOT as usize) {
                *v = f32::NEG_INFINITY;
            }
        }
    }
    if let Some(&last_ts) = sampled.iter().rev().find(|&&t| ts(t)) {
        let stop = if last_was_ts && !penult_was_ts {
            last_ts
        } else {
            last_ts + 1
        };
        for v in row.iter_mut().take(stop as usize).skip(TS_BEGIN as usize) {
            *v = f32::NEG_INFINITY;
        }
    }
    if sampled.is_empty() {
        for v in row.iter_mut().take(TS_BEGIN as usize) {
            *v = f32::NEG_INFINITY;
        }
        for v in row.iter_mut().skip((TS_BEGIN + MAX_INIT + 1) as usize) {
            *v = f32::NEG_INFINITY;
        }
    }
    // "if the total probability of the timestamps beats the best text token"
    let lse = |sl: &[f32]| -> f64 {
        let m = sl.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        if !m.is_finite() {
            return f64::NEG_INFINITY;
        }
        m + sl.iter().map(|&v| (v as f64 - m).exp()).sum::<f64>().ln()
    };
    let ts_lp = lse(&row[TS_BEGIN as usize..]);
    let best_text = row[..TS_BEGIN as usize]
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    if ts_lp > best_text as f64 {
        for v in row.iter_mut().take(TS_BEGIN as usize) {
            *v = f32::NEG_INFINITY;
        }
    }
}

/// Which tokens survived - that is the only thing the greedy pick can see,
/// and comparing the SET makes a near-tie in the mass rule readable instead of
/// showing up as one mystery index.
fn legal(row: &[f32]) -> Vec<usize> {
    row.iter()
        .enumerate()
        .filter(|(_, v)| v.is_finite())
        .map(|(i, _)| i)
        .collect()
}

fn exec() -> Option<GpuExecutor> {
    common::gpu()
}

#[test]
fn matches_the_reference_rules_at_every_stage_of_a_window() {
    let Some(e) = exec() else { return };
    let s = scale();
    let t = |sec: f32| TS_BEGIN + (sec / 0.02).round() as u32;
    // one row per state a real window passes through, run as one batch so the
    // per-row indexing is under test too
    let states: Vec<Vec<u32>> = vec![
        vec![],                                 // opening: must be a timestamp
        vec![t(0.0)],                           // just opened: must be text
        vec![t(0.0), 462, 828],                 // mid-text
        vec![t(0.0), 462, t(2.0)],              // closed a segment
        vec![t(0.0), 462, t(2.0), t(2.0)],      // opened the next at the same instant
        vec![t(0.0), 462, t(2.0), t(2.0), 951], // and back into text
    ];
    let rows = states.len();
    let mut host: Vec<f32> = det(rows * VOCAB, 0xabcd);
    let d_logits = e.to_device(&host).expect("upload");
    let flat: Vec<u32> = states
        .iter()
        .flat_map(|st| ts_state(st, &s, true))
        .collect();
    let d_state = e.to_device_u32(&flat).expect("state");
    let mut d_logits = d_logits;
    e.whisper_ts_rules(
        &mut d_logits,
        &d_state,
        rows,
        VOCAB,
        EOT,
        NO_TS,
        TS_BEGIN,
        MAX_INIT,
    )
    .expect("ts_rules");
    let got = e.to_host_len(&d_logits, rows * VOCAB).expect("back");

    for (r, sampled) in states.iter().enumerate() {
        let want_row = &mut host[r * VOCAB..(r + 1) * VOCAB];
        host_rules(want_row, sampled);
        let got_row = &got[r * VOCAB..(r + 1) * VOCAB];
        let (a, b) = (legal(want_row), legal(got_row));
        assert_eq!(
            a.len(),
            b.len(),
            "row {r} ({sampled:?}): {} tokens legal on device, {} on the reference",
            b.len(),
            a.len()
        );
        assert_eq!(
            a, b,
            "row {r} ({sampled:?}): a different set of tokens survived"
        );
        // the surviving VALUES must be untouched - this filter masks, it does
        // not rescale
        for &i in &a {
            assert_eq!(want_row[i], got_row[i], "row {r}: value at {i} changed");
        }
    }
}

#[test]
fn the_opening_step_can_only_emit_an_early_timestamp() {
    let Some(e) = exec() else { return };
    let s = scale();
    // The rule that makes the whole feature work. Without it KB-Whisper's
    // greedy argmax here is `<|notimestamps|>` (measured at p=0.794) and the
    // window decodes with no times at all.
    let mut host: Vec<f32> = det(VOCAB, 7);
    host[NO_TS as usize] = 99.0; // exactly the trap: the mode token as argmax
    let mut d = e.to_device(&host).expect("upload");
    let st = ts_state(&[], &s, true);
    let d_state = e.to_device_u32(&st).expect("state");
    e.whisper_ts_rules(&mut d, &d_state, 1, VOCAB, EOT, NO_TS, TS_BEGIN, MAX_INIT)
        .expect("rules");
    let got = e.to_host_len(&d, VOCAB).expect("back");
    let survivors = legal(&got);
    assert_eq!(
        survivors.first().copied(),
        Some(TS_BEGIN as usize),
        "the window must be able to open at 0.00"
    );
    assert_eq!(
        survivors.last().copied(),
        Some((TS_BEGIN + MAX_INIT) as usize),
        "max_initial_timestamp must cap the opening time at 1.0 s"
    );
    assert!(
        !got[NO_TS as usize].is_finite(),
        "`<|notimestamps|>` survived the opening step"
    );
}

#[test]
fn a_disabled_row_is_left_exactly_alone() {
    let Some(e) = exec() else { return };
    // Mixed batches are the point of the per-row flag: a plain-text request
    // sharing a step with a timestamped one must decode as if the filter were
    // not there at all.
    let host: Vec<f32> = det(2 * VOCAB, 31);
    let mut d = e.to_device(&host).expect("upload");
    let s = scale();
    let mut flat = ts_state(&[], &s, false).to_vec();
    flat.extend_from_slice(&ts_state(&[], &s, true));
    let d_state = e.to_device_u32(&flat).expect("state");
    e.whisper_ts_rules(&mut d, &d_state, 2, VOCAB, EOT, NO_TS, TS_BEGIN, MAX_INIT)
        .expect("rules");
    let got = e.to_host_len(&d, 2 * VOCAB).expect("back");
    assert_eq!(
        &got[..VOCAB],
        &host[..VOCAB],
        "the disabled row was modified"
    );
    assert!(
        legal(&got[VOCAB..]).len() < VOCAB,
        "the enabled row alongside it was not filtered"
    );
}
