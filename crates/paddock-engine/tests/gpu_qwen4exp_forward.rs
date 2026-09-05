//! Full-forward parity gate for the Qwen3.8-Flash-Next GPU graph.
//!
//! The oracle is `examples/q38fn_host_forward.rs` - the host-exact f32 forward
//! that holds ARBITER parity (vLLM PR #53899 on the same NVFP4 bytes, temp 0):
//! argmax and the top-4 ranks are identical there, and those ids are the gate
//! here. When a reference dump is present (`--dump` on that example, pointed at
//! by `Q38FN_REF_DUMP`) the gate additionally bounds the numeric deviation.
//!
//! HEAVY: this uploads the whole 76.6 GiB text model, so it needs a card with
//! nothing else resident. Run it with:
//!
//! ```text
//! PADDOCK_MODELS=/models PADDOCK_HEAVY_TESTS=1 PADDOCK_STRICT_GATES=1 \
//!   cargo test --release -p paddock-engine --test gpu_qwen4exp_forward -- --nocapture
//! ```

mod common;

use paddock_engine::gpu_model::qwen4exp::Qwen4ExpGpu;

/// "The capital of Sweden is" - the stamped positive control. Host reference
/// and arbiter agree on the argmax (50332 " Stockholm") and the next three.
const PROMPT_A: &[u32] = &[760, 6511, 314, 22466, 369];
const TOP4_A: &[usize] = &[50332, 271, 7172, 198];

/// A 14-token numpy code context; the host reference's top-4 for it.
const PROMPT_B: &[u32] = &[
    464, 8328, 430, 2510, 198, 87, 283, 2510, 7007, 2477, 16, 11, 220, 17,
];
const TOP4_B: &[usize] = &[11, 2387, 1089, 13];

/// Which routed-MoE class this binary loads. The default (`W4A4`) quantizes
/// ACTIVATIONS to fp4 on the grouped tensor-core lane - the class this
/// checkpoint was quantized for (`input_activations.num_bits = 4` in its own
/// `quantization_config`) and the class the rival serves off the same bytes.
/// `PADDOCK_Q38FN_MOE_G=0` selects the per-pair walk, which dequantizes the
/// expert weights against f32 activations: a more accurate class than the
/// checkpoint's own, and the one the host oracle models.
///
/// Two classes, two gates. Rank ORDER inside a near-tied tail is a property of
/// the numeric class, not of the model, so it is asserted only where the class
/// matches the oracle's; what both classes must hold is the argmax, the top-k
/// SET, the greedy chain, graph == eager, and batch == single.
fn w4a4_lane() -> bool {
    !matches!(
        std::env::var("PADDOCK_Q38FN_MOE_G").ok().as_deref(),
        Some("0") | Some("off")
    )
}

fn top_k(logits: &[f32], k: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]));
    order.truncate(k);
    order
}

#[test]
fn forward_matches_host_reference_top4() {
    let Some(dir) = common::model_dir("QWEN4EXP_DIR", &["Qwen3.8-Flash-Next-NVFP4"]) else {
        return;
    };
    if !common::heavy() {
        return;
    }
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    if !exec.has_qwen4exp_ops() {
        common::missing("pack has no qwen4exp kernels (rebuild packs/cuda)");
        return;
    }

    let mut m = Qwen4ExpGpu::load(&exec, &dir, 512).expect("load qwen4exp");
    for (name, ids, want) in [
        ("capital-of-sweden", PROMPT_A, TOP4_A),
        ("numpy-code", PROMPT_B, TOP4_B),
    ] {
        let logits = m.forward_prompt(ids).expect("forward");
        assert_eq!(logits.len(), m.config().vocab, "logit width");
        let got = top_k(&logits, 8);
        eprintln!(
            "{name}: top-8 {:?}  (argmax logit {:.4})",
            got, logits[got[0]]
        );
        assert_eq!(
            &got[..4],
            want,
            "{name}: GPU top-4 differs from the host-exact reference"
        );
    }
}

/// Numeric bound against the reference's own final logits, when the dump made
/// by `q38fn_host_forward --dump <dir>` is available. The f16 KV cache is the
/// dominant deviation (there is no f32 KV kernel), so this bounds the RELATIVE
/// spread over the top of the distribution rather than asking for f32 equality.
#[test]
fn forward_logits_track_the_reference_dump() {
    let Some(dump) = std::env::var_os("Q38FN_REF_DUMP").map(std::path::PathBuf::from) else {
        common::missing("Q38FN_REF_DUMP not set (run q38fn_host_forward --dump <dir>)");
        return;
    };
    let Some(dir) = common::model_dir("QWEN4EXP_DIR", &["Qwen3.8-Flash-Next-NVFP4"]) else {
        return;
    };
    if !common::heavy() {
        return;
    }
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    if !exec.has_qwen4exp_ops() {
        common::missing("pack has no qwen4exp kernels (rebuild packs/cuda)");
        return;
    }
    let raw = std::fs::read(dump.join("logits.bin")).expect("reference logits.bin");
    let want: Vec<f32> = raw
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();

    let mut m = Qwen4ExpGpu::load(&exec, &dir, 512).expect("load qwen4exp");
    let got = m.forward_prompt(PROMPT_A).expect("forward");
    assert_eq!(got.len(), want.len(), "vocab width");

    // spread over the top-64: the tail is dominated by near-equal small logits
    // where an absolute difference says nothing about the distribution
    let top = top_k(&want, 64);
    let span = want[top[0]] - want[top[63]];
    let worst = top
        .iter()
        .map(|&i| (got[i] - want[i]).abs())
        .fold(0.0f32, f32::max);
    eprintln!(
        "forward vs host reference: worst |Δlogit| over the top-64 = {worst:.4} \
         (top-64 span {span:.4}, ratio {:.3}%)",
        100.0 * worst / span
    );
    assert!(
        worst / span < 0.05,
        "GPU logits drift {worst} over a {span}-wide top-64 - more than the f16 KV can explain"
    );
}

/// The decode invariant: prefill(n+k) and prefill(n) + k decode steps must
/// land in the same place. They run different kernels - sequence-form convs
/// and tiled prefill attention versus windowed conv steps and the decode
/// attention walk - so agreement here is what says the carried state (GDN
/// recurrence, both conv windows, KV, the PLE stream) is complete and correct.
#[test]
fn decode_continues_prefill_exactly() {
    let Some(dir) = common::model_dir("QWEN4EXP_DIR", &["Qwen3.8-Flash-Next-NVFP4"]) else {
        return;
    };
    if !common::heavy() {
        return;
    }
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    if !exec.has_qwen4exp_ops() {
        common::missing("pack has no qwen4exp kernels (rebuild packs/cuda)");
        return;
    }
    let mut m = Qwen4ExpGpu::load(&exec, &dir, 512).expect("load qwen4exp");

    // split the 14-token prompt: prefill 12, then step the last two
    let split = PROMPT_B.len() - 2;
    m.forward_prompt(&PROMPT_B[..split]).expect("prefill");
    assert_eq!(m.position(), split);
    m.decode_step(PROMPT_B[split]).expect("step 1");
    let stepped = m.decode_step(PROMPT_B[split + 1]).expect("step 2");
    assert_eq!(m.position(), PROMPT_B.len());

    let whole = m.forward_prompt(PROMPT_B).expect("whole prefill");

    let (ts, tw) = (top_k(&stepped, 8), top_k(&whole, 8));
    let worst = tw
        .iter()
        .map(|&i| (stepped[i] - whole[i]).abs())
        .fold(0.0f32, f32::max);
    let span = whole[tw[0]] - whole[tw[7]];
    eprintln!(
        "decode vs prefill: top-8 {ts:?} vs {tw:?}; worst |Δlogit| over the top-8 \
         {worst:.4} (span {span:.4})"
    );
    assert_eq!(ts[..4], tw[..4], "decode path predicts a different top-4");
    // Same lane both sides, so this is pure shape noise - but under W4A4 the
    // two sides do not run the same arithmetic: a 1-row group and a 12-row
    // group take different CUTLASS tiles, and an fp4 accumulation resolves a
    // near-tie differently. Measured 2026-08-28: 0.070 of span on the W4A4
    // lane against 0.009 on the reference lane, both with identical top-4.
    let drift = if w4a4_lane() { 0.15 } else { 0.05 };
    assert!(
        worst / span < drift,
        "decode drifts {worst} from prefill over a {span}-wide top-8"
    );
}

/// One position PAST the stamped prompt, against the host-exact reference run
/// on the same six ids (`q38fn_host_forward --ids 760,6511,314,22466,369,50332`).
/// This is the position where the ARBITER diverges from us: it
/// continues with 11 (",") where both of our implementations rank 13 (".")
/// first - and they rank it first by 0.237, a near-tie the two sides resolve
/// differently because vLLM's Blackwell NVFP4 path quantizes ACTIVATIONS to
/// fp4 (W4A4) while this lane dequantizes the weights against f32 activations.
/// Our GPU and our f32 host reference agree here to 1e-4, which is the claim
/// this gate actually defends.
#[test]
fn continuation_matches_host_reference() {
    let Some(dir) = common::model_dir("QWEN4EXP_DIR", &["Qwen3.8-Flash-Next-NVFP4"]) else {
        return;
    };
    if !common::heavy() {
        return;
    }
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    if !exec.has_qwen4exp_ops() {
        common::missing("pack has no qwen4exp kernels (rebuild packs/cuda)");
        return;
    }
    // host reference top-8 and its logits at this position
    const IDS: &[u32] = &[760, 6511, 314, 22466, 369, 50332];
    const WANT: &[usize] = &[13, 11, 321, 198, 271, 641, 864, 318];
    /// Bound on the cross-class logit move at this position. Set from the
    /// measured value with a 2x margin, not from what makes the gate pass.
    const W4A4_HOST_LOGIT_BOUND: f32 = 0.7;
    const WANT_L: &[f32] = &[
        17.7683, 17.5311, 14.7101, 14.2866, 14.0679, 13.8832, 13.8455, 13.8403,
    ];

    let mut m = Qwen4ExpGpu::load(&exec, &dir, 512).expect("load qwen4exp");
    let logits = m.forward_prompt(IDS).expect("forward");
    let got = top_k(&logits, 8);
    let worst = WANT
        .iter()
        .zip(WANT_L)
        .map(|(&i, &l)| (logits[i] - l).abs())
        .fold(0.0f32, f32::max);
    eprintln!(
        "continuation[{}]: top-8 {got:?}; worst |Δlogit| vs host {worst:.4}",
        if w4a4_lane() { "w4a4" } else { "ref" }
    );
    if w4a4_lane() {
        // The oracle is an f32-activation run; W4A4 is a different numeric
        // class and this position is the KNOWN near-tie (ranks 4..8 sit inside
        // a 0.45-wide band under a 17.77 top logit, and the arbiter itself
        // resolves the top-2 the other way). What survives the class change is
        // asserted; the tail permutation is not.
        assert_eq!(got[..3], WANT[..3], "W4A4 moved the top-3");
        // The host's ranks 4..8 sit in a 0.45-wide band (14.2866, 14.0679,
        // 13.8832, 13.8455, 13.8403) under a 17.77 top logit, and rank 8 is
        // 0.005 above rank 9 - a boundary that no numeric-class change
        // preserves, so membership is asserted with one slot of give rather
        // than exactly. Ranks 1..3 have real margins (0.24, 2.82, 0.42) and
        // are asserted in order above.
        let kept = WANT.iter().filter(|t| got.contains(t)).count();
        assert!(
            kept >= WANT.len() - 1,
            "W4A4 kept only {kept}/{} of the host top-8 - that is a move, not a tie-break",
            WANT.len()
        );
        assert!(
            worst < W4A4_HOST_LOGIT_BOUND,
            "W4A4 drifts {worst} from the host reference"
        );
    } else {
        assert_eq!(got, WANT, "top-8 differs from the host-exact reference");
        assert!(worst < 0.01, "logits drift {worst} from the host reference");
    }
}

/// Greedy generation, gated on SELF-CONSISTENCY rather than on matching the
/// arbiter token for token: one near-tie resolved differently (see
/// `continuation_matches_host_reference`) re-routes every token after it, so
/// an exact-match gate against a different numeric class would be a coin flip
/// dressed as a bound. What must hold is that the generated chain is the same
/// chain a single prefill of prompt+generation would predict.
#[test]
fn greedy_generation_is_self_consistent() {
    let Some(dir) = common::model_dir("QWEN4EXP_DIR", &["Qwen3.8-Flash-Next-NVFP4"]) else {
        return;
    };
    if !common::heavy() {
        return;
    }
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    if !exec.has_qwen4exp_ops() {
        common::missing("pack has no qwen4exp kernels (rebuild packs/cuda)");
        return;
    }
    let mut m = Qwen4ExpGpu::load(&exec, &dir, 512).expect("load qwen4exp");
    let chain = m.generate_greedy(PROMPT_A, 8).expect("generate");
    eprintln!("greedy: {chain:?}");
    assert_eq!(
        chain[0], 50332,
        "first generated token is not \" Stockholm\""
    );

    // re-run the whole thing as one prefill and check every step's argmax
    let mut all = PROMPT_A.to_vec();
    all.extend_from_slice(&chain[..chain.len() - 1]);
    let logits = m
        .forward_prompt(&all)
        .expect("prefill of prompt+generation");
    let want_last = *chain.last().expect("generated at least one token");
    let got_last = top_k(&logits, 1)[0] as u32;
    assert_eq!(
        got_last, want_last,
        "prefilling the generated chain predicts a different next token \
         than the decode walk did"
    );
}

/// The captured decode tick must be the eager walk, exactly. Capture bakes
/// device ADDRESSES, so anything that silently reallocated or that read a
/// per-token value from the host instead of from device memory would show up
/// here as a chain that diverges after the first replay - or as a chain frozen
/// at whatever the capture saw.
#[test]
fn decode_graph_matches_eager() {
    let Some(dir) = common::model_dir("QWEN4EXP_DIR", &["Qwen3.8-Flash-Next-NVFP4"]) else {
        return;
    };
    if !common::heavy() {
        return;
    }
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    if !exec.has_qwen4exp_ops() {
        common::missing("pack has no qwen4exp kernels (rebuild packs/cuda)");
        return;
    }
    let mut m = Qwen4ExpGpu::load(&exec, &dir, 512).expect("load qwen4exp");

    m.set_graph_capture(false);
    let eager = m.generate_greedy(PROMPT_A, 10).expect("eager generation");
    assert!(!m.graph_active(), "capture ran with the switch off");

    m.set_graph_capture(true);
    let graphed = m.generate_greedy(PROMPT_A, 10).expect("graphed generation");
    assert!(m.graph_active(), "the decode tick was never captured");

    eprintln!("eager   {eager:?}\ngraphed {graphed:?}");
    assert_eq!(
        graphed, eager,
        "the captured tick diverges from the eager walk"
    );
}

/// The batched-decode gate: N slots advancing together must produce exactly the
/// tokens each slot produces on its own. This is what makes a serving lane
/// possible - every carried state (GDN recurrence, both conv windows, the KV
/// cache, the position cursor) has to be per-slot, and a bug in any of them
/// shows up as cross-talk between rows.
#[test]
fn batched_slots_match_single_slot_runs() {
    let Some(dir) = common::model_dir("QWEN4EXP_DIR", &["Qwen3.8-Flash-Next-NVFP4"]) else {
        return;
    };
    if !common::heavy() {
        return;
    }
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    if !exec.has_qwen4exp_ops() {
        common::missing("pack has no qwen4exp kernels (rebuild packs/cuda)");
        return;
    }
    let mut m = Qwen4ExpGpu::load_with_slots(&exec, &dir, 512, 2).expect("load qwen4exp x2");
    assert_eq!(m.slot_count(), 2);

    const K: usize = 6;
    let pa: Vec<u32> = PROMPT_A.to_vec();
    let pb: Vec<u32> = PROMPT_B.to_vec();

    // reference: each slot driven alone, one row per batched step
    let solo = |m: &mut Qwen4ExpGpu, slot: usize, p: &[u32]| -> Vec<u32> {
        let mut lg = m.prefill_slot(slot, p).expect("prefill slot");
        let mut out = Vec::new();
        for _ in 0..K {
            let t = top_k(&lg, 1)[0] as u32;
            out.push(t);
            lg = m.decode_step_batch(&[(slot, t)]).expect("solo step")[0].clone();
        }
        out
    };
    let ref_a = solo(&mut m, 0, &pa);
    let ref_b = solo(&mut m, 1, &pb);

    // now both slots advancing in one walk
    let mut la = m.prefill_slot(0, &pa).expect("prefill 0");
    let mut lb = m.prefill_slot(1, &pb).expect("prefill 1");
    let (mut got_a, mut got_b) = (Vec::new(), Vec::new());
    for _ in 0..K {
        let ta = top_k(&la, 1)[0] as u32;
        let tb = top_k(&lb, 1)[0] as u32;
        got_a.push(ta);
        got_b.push(tb);
        let out = m
            .decode_step_batch(&[(0, ta), (1, tb)])
            .expect("batched step");
        la = out[0].clone();
        lb = out[1].clone();
    }
    assert_eq!(got_a, ref_a, "slot 0 diverged when batched with slot 1");
    assert_eq!(got_b, ref_b, "slot 1 diverged when batched with slot 0");
}

/// A prompt prefilled inside a WAVE must land on the same distribution as the
/// same prompt prefilled on its own. The wave shares one walk across several
/// slots: every row-parallel op sees all the rows at once, the two convs run
/// at a row offset, and the recurrence runs grid (heads, runs) - so this gate
/// is what says the fused shape did not leak one prompt's state into
/// another's.
///
/// Bit-equality is not asserted and would be wrong to assert: the routed MoE
/// sorts a 3-prompt wave into different 128-row groups than a single prompt,
/// so an fp4 accumulation resolves a near-tie differently - measured here as
/// ranks 2/3 of slot 2 swapping inside a band narrower than the class's own
/// logit noise. What is asserted: the argmax, the top-4 SET, a bound on the
/// logit move, and - the actual failure mode this gate exists for - that each
/// waved slot is FAR closer to its own serial run than to any other slot's.
/// A state leak between runs cannot survive that last one.
#[test]
fn prefill_wave_matches_serial_prefill() {
    let Some(dir) = common::model_dir("QWEN4EXP_DIR", &["Qwen3.8-Flash-Next-NVFP4"]) else {
        return;
    };
    if !common::heavy() {
        return;
    }
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    if !exec.has_qwen4exp_ops() {
        common::missing("pack has no qwen4exp kernels (rebuild packs/cuda)");
        return;
    }
    if !exec.has_gated_delta_recurrent_runs() {
        common::missing("pack has no gated_delta_recurrent_runs (slot 534)");
        return;
    }
    // Bound on the wave-vs-serial logit move over the top-4, with a ~2x margin
    // over the worst MEASURED value. Two controls fix it, and they are what
    // says the move is the wave's SHAPE and not any lane:
    //   * routed MoE on the ungrouped per-pair lane, short prompts: 0.07-0.10,
    //     top-4 identical, greedy chains identical. The runs walk is right.
    //   * these prompts, dense lane forced to bf16 on both sides: 1.139.
    //     Same lane, same weights, and rank 4 still moves - so a top-4 SET
    //     assertion here would be asserting a near-tie, not a property.
    //     The shipped Dual class measures 0.885 on the same comparison, i.e.
    //     CLOSER to the serial reference, which is what f16 activations
    //     carrying 11 mantissa bits against bf16's 8 should do.
    let bound: f32 = 2.5;
    // Three different lengths so no run offset is a multiple of the tile
    // height - AND each run under the dense lane's f16 batch threshold while
    // the FUSED wave is over it. Under the shipped `Dual` class that makes
    // this gate a direct f16-tensor-core vs bf16-tile comparison: the serial
    // reference below runs every prompt on the bf16 tile (28/24/33 rows each),
    // the wave runs all 85 rows on the tcgen05 f16 twin. Nothing else in this
    // file crosses that threshold.
    // SIX runs of different lengths, so no run offset is a multiple of the
    // tile height and the fused wave (77 rows) crosses the dense lane's f16
    // batch threshold while every serial run (5..19) stays under it. Under the
    // shipped `Dual` class that makes this gate a direct tcgen05-f16 vs
    // bf16-tile comparison, and nothing else in this file crosses it.
    //
    // Width, not length, is what buys the rows, and the runs are ARRANGEMENTS
    // of the two stamped prompts rather than repeats or padding: repeating a
    // stamped prompt flattens its own next-token distribution (five copies of
    // "the capital of Sweden is" put 760 and 271 inside 0.1 of each other) and
    // a nonsense tail (A+A+B[..5]) drifts 3.1 logits under the wave, so both
    // turn every rank assertion into a coin flip. These keep the margins the
    // file's other gates rely on.
    let prompts: Vec<Vec<u32>> = vec![
        PROMPT_A.to_vec(),                   // 5
        PROMPT_B.to_vec(),                   // 14
        [PROMPT_A, &PROMPT_B[..7]].concat(), // 12
        [PROMPT_B, PROMPT_A].concat(),       // 19
        [&PROMPT_B[..7], PROMPT_A].concat(), // 12
        [PROMPT_A, PROMPT_B].concat(),       // 19
    ];
    let mut m =
        Qwen4ExpGpu::load_with_slots(&exec, &dir, 512, prompts.len()).expect("load qwen4exp");
    eprintln!(
        "wave gate: dense class {:?}, wave rows {} (serial runs {:?})",
        paddock_engine::gpu_model::qwen4exp::dense_class_from_env(),
        prompts.iter().map(|p| p.len()).sum::<usize>(),
        prompts.iter().map(|p| p.len()).collect::<Vec<_>>()
    );

    // serial reference first: prefill each slot alone and take three greedy
    // steps, so the chain comparison below is against a run that never saw a
    // wave
    let mut serial: Vec<Vec<f32>> = Vec::new();
    let mut serial_chain: Vec<Vec<u32>> = Vec::new();
    for (i, p) in prompts.iter().enumerate() {
        let l = m.prefill_slot(i, p).expect("serial prefill");
        let mut ch = vec![top_k(&l, 1)[0] as u32];
        for _ in 0..3 {
            let out = m
                .decode_step_batch(&[(i, *ch.last().unwrap())])
                .expect("serial step");
            ch.push(top_k(&out[0], 1)[0] as u32);
        }
        serial.push(l);
        serial_chain.push(ch);
    }

    let items: Vec<(usize, Vec<u32>)> = prompts.iter().cloned().enumerate().collect();
    let waved = m.prefill_slots(&items).expect("wave prefill");
    assert_eq!(waved.len(), prompts.len());

    // distance to own serial run vs to every other slot's: the leak test
    let rms = |a: &[f32], b: &[f32]| -> f32 {
        (a.iter()
            .zip(b)
            .map(|(x, y)| ((x - y) as f64).powi(2))
            .sum::<f64>()
            / a.len() as f64)
            .sqrt() as f32
    };
    for (i, l) in waved.iter().enumerate() {
        let got = top_k(l, 4);
        let want = top_k(&serial[i], 4);
        let worst = want
            .iter()
            .chain(got.iter())
            .map(|&t| (l[t] - serial[i][t]).abs())
            .fold(0.0f32, f32::max);
        let own = rms(l, &serial[i]);
        let other = (0..prompts.len())
            .filter(|&j| j != i)
            .map(|j| rms(l, &serial[j]))
            .fold(f32::INFINITY, f32::min);
        eprintln!(
            "wave slot {i}: top-4 {got:?} vs serial {want:?}; worst |Δlogit| {worst:.4}; \
             rms own {own:.4} vs nearest-other {other:.4}"
        );
        // The ARGMAX is what a served token is, and it is asserted. Rank
        // order below it is not: a 77-row wave puts every row in a different
        // tile than its 5..19-row serial run did, and the bf16-on-both-sides
        // control moves ranks 3-4 by itself (slot 3: [50332, 11, 18, 220] vs
        // [50332, 11, 2387, 18], drift 1.07). What is asserted below it is
        // membership with one slot of give, plus the logit bound and the leak
        // check - the three things a real defect cannot pass.
        assert_eq!(got[0], want[0], "slot {i} argmax moved under the wave");
        let kept = want.iter().filter(|t| got.contains(t)).count();
        assert!(
            kept >= want.len() - 1,
            "slot {i} kept only {kept}/{} of the serial top-4",
            want.len()
        );
        assert!(
            worst < bound,
            "slot {i} logits moved {worst} under the wave"
        );
        // Isolation: a waved row must be NEAREST to its own serial run. Stated
        // as a minimum and not a ratio deliberately - arrangements of two
        // stamped prompts necessarily share endings, and two runs that end in
        // the same five tokens have next-token distributions 1.4x apart for
        // reasons that have nothing to do with the wave.
        //
        // Honest about what this catches: not the `pd_attn_prefill` slots[0]
        // bug this gate was written after. There every run still tracked its
        // own tokens through everything but attention, so `own` stayed the
        // minimum - what caught it was the DRIFT bound (0.79 on the ungrouped
        // control, where the class explains 0.10). This is the coarser net,
        // for a run that lands on another's state outright.
        assert!(
            own < other,
            "slot {i} is not closest to its OWN serial run \
             (rms own {own}, nearest other {other}) - a run leaked into another"
        );
    }

    // the carried state: three greedy steps off the wave-prefilled slots, all
    // three advancing together, against the serial chains above
    let mut chain: Vec<Vec<u32>> = waved.iter().map(|l| vec![top_k(l, 1)[0] as u32]).collect();
    for _ in 0..3 {
        let rows: Vec<(usize, u32)> = (0..prompts.len())
            .map(|i| (i, *chain[i].last().unwrap()))
            .collect();
        let out = m.decode_step_batch(&rows).expect("wave step");
        for (i, o) in out.iter().enumerate() {
            chain[i].push(top_k(o, 1)[0] as u32);
        }
    }
    for (i, ch) in chain.iter().enumerate() {
        eprintln!(
            "wave slot {i}: chain {ch:?} vs serial {:?}",
            serial_chain[i]
        );
        // Exact on the reference lane. Under W4A4 a rank-2/3 near-tie can
        // resolve the other way and every token after it re-routes, which is
        // the same reason `greedy_generation_is_self_consistent` gates on
        // self-consistency rather than on matching another numeric class; the
        // first token is the argmax already asserted above.
        if w4a4_lane() {
            assert_eq!(
                ch[0], serial_chain[i][0],
                "slot {i} first token moved under the wave"
            );
        } else {
            assert_eq!(
                ch, &serial_chain[i],
                "slot {i} continues differently after a wave prefill"
            );
        }
    }
}
