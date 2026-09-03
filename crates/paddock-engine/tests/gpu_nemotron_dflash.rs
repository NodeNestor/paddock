//! DFlash drafter gates for nemotron (C2): attach the official
//! `nvidia/...-NVFP4-DFlash` checkpoint to the NVFP4 target (its trained
//! pairing), verify feature coverage flows from the batched walks, drafts
//! are deterministic, and - the invariant that matters - a full spec loop
//! (draft -> trunk verify -> accept walk -> commit) produces exactly the
//! no-spec greedy stream. Acceptance quality is reported, not asserted
//! (it is the model's business); correctness is asserted.

mod common;

use paddock_engine::generator::Generator;
use paddock_engine::gpu_model::nemotron::GpuNemotron;

const CKPT_ENV: &str = "NEMOTRON_NVFP4_DIR";
const CKPT_DIR: &str = "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4";
const DFLASH_ENV: &str = "NEMOTRON_DFLASH_DIR";
const DFLASH_DIR: &str = "/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DFlash";
const ORACLE: &str = "/models/nemotron-battery/oracle/decoder-oracle.json";
const PROMPT_LEN: usize = 700;

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
fn dflash_spec_loop_matches_greedy() {
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    if !exec.has_paged_kv()
        || !exec.has_mamba2_batch()
        || !exec.has_nvf4_gemv_batch()
        || !exec.has_nvf4_ckpt()
        || !exec.has_spec_verify_mamba()
        || !exec.has_argmax_rows()
    {
        common::missing("pack lacks the nemotron dflash kernel set");
        return;
    }
    let Some(dir) = common::model_dir(CKPT_ENV, &[CKPT_DIR]) else {
        return;
    };
    let df_dir = std::env::var(DFLASH_ENV).unwrap_or_else(|_| DFLASH_DIR.into());
    if !std::path::Path::new(&df_dir)
        .join("model.safetensors")
        .exists()
    {
        eprintln!("skip: no dflash checkpoint at {df_dir}");
        return;
    }
    let Some(prompt) = oracle_prompt(PROMPT_LEN) else {
        common::missing("no oracle dump for prompt ids");
        return;
    };
    let mut model = GpuNemotron::load_dir(exec, &dir, 4096).expect("load");
    model
        .attach_dflash(std::path::Path::new(&df_dir))
        .expect("attach dflash");
    assert!(
        model.spec_capable(),
        "drafter attached + verify kernels => spec capable"
    );
    assert_eq!(model.batch_enable_probe(4).expect("enable"), 4);

    // slot 0: fully walked prefill - features cover [0, 700); the spec
    // loop runs here. (A prefix-resumed slot honestly starts feature-cold
    // for the resumed span: those rows were walked on another slot.)
    let l0 = model.forward_prefill(0, &prompt).expect("prefill 0");
    let b0 = argmax(&l0);
    assert!(
        model
            .spec_ensure_warm(0, &[], (PROMPT_LEN - 1) as u32)
            .expect("warm probe"),
        "slot 0 must be feature-warm after its full prefill"
    );

    // no-spec baseline stream on slot 1 (prefix resume gives it the same
    // state; its own drafter coverage is irrelevant here)
    let l1 = model.forward_prefill(1, &prompt).expect("prefill 1");
    assert_eq!(argmax(&l1), b0);
    let mut b = vec![b0];
    for i in 0..32 {
        let (step, _) = model
            .forward_mixed_sampled(
                &[(1usize, b[i], (PROMPT_LEN + i) as u32)],
                usize::MAX,
                &[paddock_engine::generator::RowSample::Device(
                    paddock_engine::sampler::DevicePlan::Greedy,
                )],
                &[],
            )
            .expect("baseline decode");
        b.push(step.ids[0]);
    }

    // drafts must be deterministic (the laguna selftest property)
    let d1 = model
        .spec_draft_batch(&[(0usize, b[0])], 4)
        .expect("draft")
        .expect("engaged");
    let d2 = model
        .spec_draft_batch(&[(0usize, b[0])], 4)
        .expect("draft 2")
        .expect("engaged");
    assert_eq!(d1, d2, "repeat drafts diverged (ring append race?)");
    assert_eq!(d1[0].len(), 4);
    let hits = d1[0].iter().zip(&b[1..5]).filter(|(a, c)| a == c).count();
    eprintln!(
        "first-round drafts {:?} vs truth {:?} - {hits}/4 on-stream",
        d1[0],
        &b[1..5]
    );

    // full spec loop on slot 0: committed stream must equal the no-spec
    // stream REGARDLESS of draft quality
    let mut committed: Vec<u32> = Vec::new();
    let mut pending = b[0];
    let mut pos = PROMPT_LEN;
    let mut drafted = 0usize;
    let mut accepted = 0usize;
    let mut rounds = 0usize;
    while committed.len() < 24 {
        rounds += 1;
        let k = 7usize;
        let drafts = model
            .spec_draft_batch(&[(0usize, pending)], k)
            .expect("draft")
            .expect("engaged");
        let mut chunk = vec![pending];
        chunk.extend(&drafts[0]);
        drafted += drafts[0].len();
        let picks = model
            .forward_spec_batch(&[(0usize, pos, chunk.clone())])
            .expect("verify")
            .expect("engaged");
        let mut a = 0usize;
        while a + 1 < chunk.len() && chunk[a + 1] == picks[a] {
            a += 1;
        }
        accepted += a;
        // service semantics: rows 0..=a of the chunk are committed, and the
        // pick after the last accepted row becomes the next pending
        for i in 0..=a {
            committed.push(if i == 0 { chunk[0] } else { chunk[i] });
        }
        pending = picks[a];
        pos += a + 1;
        // the committed tokens so far must be the no-spec stream
        assert_eq!(
            &committed[..],
            &b[..committed.len()],
            "spec-committed stream diverged from the no-spec greedy stream"
        );
        assert_eq!(pending, b[committed.len()], "next pending off-stream");
    }
    eprintln!(
        "spec loop: {} committed in {rounds} rounds (acceptance length {:.2}), drafted {drafted}, accepted {accepted} ({:.0}%)",
        committed.len(),
        committed.len() as f64 / rounds as f64,
        100.0 * accepted as f64 / drafted.max(1) as f64
    );
}
