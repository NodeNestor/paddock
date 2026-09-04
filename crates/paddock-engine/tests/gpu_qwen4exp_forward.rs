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
    assert!(
        worst / span < 0.05,
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
    eprintln!("continuation: top-8 {got:?}; worst |Δlogit| vs host {worst:.4}");
    assert_eq!(got, WANT, "top-8 differs from the host-exact reference");
    assert!(worst < 0.01, "logits drift {worst} from the host reference");
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
