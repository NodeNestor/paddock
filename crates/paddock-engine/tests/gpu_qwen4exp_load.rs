//! qwen4exp loader oracle (the GPU side of the loader-oracle pattern): upload one GDN
//! layer, one attention layer and their MoE planes, then read every plane
//! BACK and compare against the checkpoint bytes / an exact host widen.
//! Byte-equality is the whole gate - the charter is bf16 parity, so any
//! difference is a loader bug, never tolerance.
//!
//! Sized to run beside a live serve: two layers ~= 1.5 GB (the 51 GB PLE
//! table is deliberately not loaded here; its projections are).

mod common;

use paddock_engine::gpu_model::qwen4exp::{MixerW, load_layer, load_ple_projections};
use paddock_models::modelopt::nvfp4_view;
use paddock_models::qwen4exp::{Qwen4ExpBlock, Qwen4ExpConfig};
use paddock_models::safetensors::ShardedSafetensors;

fn bf16_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
        .collect()
}

#[test]
fn qwen4exp_layer_planes_round_trip() {
    let Some(dir) = common::model_dir("QWEN4EXP_DIR", &["Qwen3.8-Flash-Next-NVFP4"]) else {
        return;
    };
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    let c = Qwen4ExpConfig::read(&dir).expect("config");
    let st = ShardedSafetensors::open_dir(&dir).expect("shards");
    assert_eq!(c.blocks[0], Qwen4ExpBlock::Gdn);
    assert_eq!(c.blocks[3], Qwen4ExpBlock::Attention);

    for li in [0usize, 3] {
        let layer = load_layer(&exec, &st, &c, li).expect("layer loads");
        let p = format!("model.language_model.layers.{li}");

        // one representative bf16 plane per mixer kind: device bytes must be
        // identical to the checkpoint's
        let (name, plane) = match &layer.mixer {
            MixerW::Gdn(g) => (format!("{p}.linear_attn.in_proj_qkv.weight"), &g.qkv),
            MixerW::Attn(a) => (format!("{p}.self_attn.q_proj.weight"), &a.q),
        };
        let (_, want) = st.bytes(&name).expect("checkpoint bytes");
        let raw = plane.raw_bf16().expect("parity class is bf16-resident");
        let got: Vec<u8> = exec.to_host_range_u8(raw, 0, want.len()).expect("dtoh");
        assert!(got == want, "{name}: device bytes differ from checkpoint");

        // the LAUNCH FOLD is a byte concatenation and nothing more: the hc
        // plane's first `lowrank` rows must be the checkpoint's down plane and
        // its tail rows the inject plane, both unchanged. A fold that quietly
        // reordered or re-encoded either half would still produce plausible
        // logits, so it is gated here rather than inferred from a forward.
        let hcw = &layer.attn_hc;
        assert_eq!(
            hcw.inject_rows, c.hc_count,
            "attn hc did not fold its inject"
        );
        let hw = c.hc_width();
        let fused = hcw.down.raw_bf16().expect("folded plane is bf16");
        let (_, down_want) = st
            .bytes(&format!(
                "{p}.attn_hyper_connection.input_mix_weight_down.weight"
            ))
            .expect("down bytes");
        let (_, inj_want) = st
            .bytes(&format!(
                "{p}.attn_hyper_connection.block_inject_weight.weight"
            ))
            .expect("inject bytes");
        let head: Vec<u8> = exec
            .to_host_range_u8(fused, 0, down_want.len())
            .expect("dtoh");
        assert!(
            head == down_want,
            "folded hc plane's head is not the down plane"
        );
        let tail: Vec<u8> = exec
            .to_host_range_u8(fused, c.hc_lowrank * hw * 2, inj_want.len())
            .expect("dtoh");
        assert!(
            tail == inj_want,
            "folded hc plane's tail is not the inject plane"
        );

        // the GDN a||b fold, same claim
        if let MixerW::Gdn(g) = &layer.mixer {
            let (_, a_want) = st
                .bytes(&format!("{p}.linear_attn.in_proj_a.weight"))
                .expect("a bytes");
            let (_, b_want) = st
                .bytes(&format!("{p}.linear_attn.in_proj_b.weight"))
                .expect("b bytes");
            let ab: Vec<f32> = exec.to_host(&g.ab.buf).expect("dtoh");
            let hv = c.gdn_v_heads * c.hidden;
            assert_eq!(ab.len(), 2 * hv, "a||b plane width");
            assert_eq!(ab[..hv], bf16_to_f32(a_want)[..], "a half of the fold");
            assert_eq!(ab[hv..], bf16_to_f32(b_want)[..], "b half of the fold");
        }

        // the router||shared-gate fold: last row is the shared expert's gate
        let router: Vec<f32> = exec.to_host(&layer.moe.router.buf).expect("dtoh");
        let (_, sg_want) = st
            .bytes(&format!("{p}.mlp.shared_expert_gate.weight"))
            .expect("shared gate bytes");
        assert_eq!(
            router.len(),
            (c.n_expert + 1) * c.hidden,
            "router plane width"
        );
        assert_eq!(
            router[c.n_expert * c.hidden..],
            bf16_to_f32(sg_want)[..],
            "router plane's last row is not the shared-expert gate"
        );

        // hyper-connection norm: exact f32 widen
        let (_, nb) = st
            .bytes(&format!("{p}.attn_hyper_connection.hc_norm.weight"))
            .unwrap();
        let got_n: Vec<f32> = exec.to_host(&layer.attn_hc.norm.buf).expect("dtoh");
        assert_eq!(got_n, bf16_to_f32(nb), "{p} hc_norm widen not exact");

        // MoE gate plane: concatenated nibbles equal per-expert checkpoint
        // views at both ends of the expert range; scale2 array matches
        let paddock_engine::gpu_model::qwen4exp::ExpertSeats::Nvf4 { gate: plane, .. } =
            &layer.moe.seats
        else {
            panic!("{p}: the safetensors lane seats NVFP4 experts");
        };
        assert_eq!(
            (plane.n_expert, plane.ff, plane.in_dim),
            (c.n_expert, c.moe_ff, c.hidden)
        );
        let data: Vec<u8> = exec
            .to_host_range_u8(&plane.data, 0, c.n_expert * c.moe_ff * c.hidden / 2)
            .expect("dtoh");
        let s2: Vec<f32> = exec.to_host(&plane.scale2).expect("dtoh");
        let stride = c.moe_ff * c.hidden / 2;
        for e in [0usize, 255, c.n_expert - 1] {
            let v = nvfp4_view(&st, &format!("{p}.mlp.experts.{e}.gate_proj")).unwrap();
            assert!(
                data[e * stride..(e + 1) * stride] == *v.packed,
                "L{li} expert {e} gate nibbles differ"
            );
            assert_eq!(s2[e], v.scale2, "L{li} expert {e} scale2");
        }
    }
    eprintln!("qwen4exp layer oracle: layers 0 (GDN) + 3 (attn) byte-exact");
}

#[test]
fn qwen4exp_ple_projections_load() {
    let Some(dir) = common::model_dir("QWEN4EXP_DIR", &["Qwen3.8-Flash-Next-NVFP4"]) else {
        return;
    };
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    let c = Qwen4ExpConfig::read(&dir).expect("config");
    let st = ShardedSafetensors::open_dir(&dir).expect("shards");
    // projections + hash buffers only - pin the audited I64 values; the
    // 51 GB table upload is exercised by the (heavy) full-load gate later.
    let ple = load_ple_projections(&exec, &st, &c, c.ple_layers[0]).expect("ple");
    assert_eq!(
        ple.multipliers,
        vec![23703573157769, 20109073645365, 8052911324071]
    );
    assert_eq!(ple.head_vocab.len(), 16);
    assert_eq!(ple.head_offset[0], 0);
    assert!(ple.table_scale.is_finite() && ple.table_scale > 0.0);
}
