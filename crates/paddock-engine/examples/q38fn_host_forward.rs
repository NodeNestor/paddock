//! Host-exact forward for Qwen3.8-Flash-Next (parity anchor).
//!
//! Runs a short token-id prompt through the entire model on CPU in f32,
//! straight from checkpoint bytes (bf16 widened exactly, NVFP4 experts
//! host-dequanted, PLE fp8 rows decoded) using the source-verified reference
//! ops. Prints the final position's top-k logits - compared against the vLLM
//! arbiter's /v1/completions logprobs on the same ids (temp 0).
//!
//! Geometry facts and every formula live in `reference::qwen4exp`.
//! GDN head widening is hk = hv % n_k_heads (the tile
//! mapping our shipped deltanet kernel uses - packs/cuda/src/deltanet/
//! core.cuh:317 - parity-proven on the 27B's identical 48V/16K geometry).
//!
//! Usage: q38fn_host_forward --dir <checkpoint> --ids 760,6511,314,22466,369
// A development probe: it runs on a box its author is looking at, and a
// failure should stop it where it happened rather than be reported.
#![allow(clippy::unwrap_used)]

use paddock_kernels::reference::delta_net::gated_delta_recurrent;
use paddock_kernels::reference::ops::YarnRope;
use paddock_kernels::reference::qwen4exp as rq;
use paddock_kernels::reference::qwen35_attn::gated_attention_core;
use paddock_models::modelopt::nvfp4_view;
use paddock_models::qwen4exp::{Qwen4ExpBlock, Qwen4ExpConfig};
use paddock_models::safetensors::{ShardedSafetensors, StDtype};

fn bf16_to_f32(b: &[u8]) -> Vec<f32> {
    b.as_chunks::<2>()
        .0
        .iter()
        .map(|c| f32::from_bits((u16::from_le_bytes(*c) as u32) << 16))
        .collect()
}

/// bf16 plane [rows, k] matvec against x[k], threaded over row chunks.
fn bf16_matvec(st: &ShardedSafetensors, name: &str, x: &[f32], rows: usize, k: usize) -> Vec<f32> {
    let (t, b) = st.bytes(name).unwrap_or_else(|| panic!("{name} missing"));
    assert_eq!(t.dtype, StDtype::Bf16, "{name}");
    assert_eq!(b.len(), rows * k * 2, "{name}");
    let nthr = 8.min(rows.max(1));
    let chunk = rows.div_ceil(nthr);
    let mut y = vec![0f32; rows];
    std::thread::scope(|s| {
        for (ti, ych) in y.chunks_mut(chunk).enumerate() {
            let b = &b[ti * chunk * k * 2..];
            s.spawn(move || {
                for (r, yr) in ych.iter_mut().enumerate() {
                    let row = &b[r * k * 2..(r + 1) * k * 2];
                    let mut acc = 0f32;
                    for (i, c) in row.as_chunks::<2>().0.iter().enumerate() {
                        let w = f32::from_bits((u16::from_le_bytes(*c) as u32) << 16);
                        acc += w * x[i];
                    }
                    *yr = acc;
                }
            });
        }
    });
    y
}

fn tensor_f32(st: &ShardedSafetensors, name: &str, want: usize) -> Vec<f32> {
    let (t, b) = st.bytes(name).unwrap_or_else(|| panic!("{name} missing"));
    match t.dtype {
        StDtype::Bf16 => {
            assert_eq!(b.len(), want * 2, "{name}");
            bf16_to_f32(b)
        }
        StDtype::F32 => {
            assert_eq!(b.len(), want * 4, "{name}");
            b.as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect()
        }
        other => panic!("{name}: dtype {other:?}"),
    }
}

fn i64s(st: &ShardedSafetensors, name: &str) -> Vec<i64> {
    let (t, b) = st.bytes(name).unwrap_or_else(|| panic!("{name} missing"));
    assert_eq!(t.dtype, StDtype::I64, "{name}");
    b.as_chunks::<8>()
        .0
        .iter()
        .map(|c| i64::from_le_bytes(*c))
        .collect()
}

/// Per-op dump sink for the GPU parity gate: `<dir>/L{li}.{tag}.bin`, raw
/// little-endian f32, `n_tokens` rows of whatever width the tag carries.
/// Off unless `--dump <dir>` is given (the parity anchor's own run is
/// unaffected - nothing here feeds back into the math).
struct Dump(Option<std::path::PathBuf>);

impl Dump {
    fn put(&self, li: usize, tag: &str, v: &[f32]) {
        let Some(dir) = &self.0 else { return };
        let mut b = Vec::with_capacity(v.len() * 4);
        for x in v {
            b.extend_from_slice(&x.to_le_bytes());
        }
        let name = if li == usize::MAX {
            format!("{tag}.bin")
        } else {
            format!("L{li}.{tag}.bin")
        };
        std::fs::write(dir.join(name), b).expect("dump write");
    }
    /// Flatten a per-token Vec<Vec<f32>> and dump it as one [n, width] file.
    fn put_rows(&self, li: usize, tag: &str, rows: &[Vec<f32>]) {
        if self.0.is_none() {
            return;
        }
        self.put(li, tag, &rows.concat());
    }
    fn on(&self) -> bool {
        self.0.is_some()
    }
}

fn softmax(x: &mut [f32]) {
    let m = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut d = 0f32;
    for v in x.iter_mut() {
        *v = (*v - m).exp();
        d += *v;
    }
    for v in x.iter_mut() {
        *v /= d;
    }
}

fn main() {
    let mut dir = None;
    let mut ids: Vec<u32> = Vec::new();
    let mut topk = 8usize;
    let mut dump_dir: Option<String> = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--dir" => dir = args.next(),
            "--ids" => {
                ids = args
                    .next()
                    .unwrap()
                    .split(',')
                    .map(|s| s.parse().unwrap())
                    .collect()
            }
            "--topk" => topk = args.next().unwrap().parse().unwrap(),
            "--dump" => dump_dir = args.next(),
            other => panic!("unknown arg {other}"),
        }
    }
    let dump = Dump(dump_dir.map(|d| {
        let p = std::path::PathBuf::from(d);
        std::fs::create_dir_all(&p).expect("dump dir");
        p
    }));
    let dir = std::path::PathBuf::from(dir.expect("--dir required"));
    assert!(!ids.is_empty(), "--ids required");
    let n = ids.len();

    let c = Qwen4ExpConfig::read(&dir).expect("config");
    let st = ShardedSafetensors::open_dir(&dir).expect("shards");
    let h = c.hidden;
    let hw = c.hc_width();
    let eps = c.eps;

    // ---- embeddings -> 4-stream state per token --------------------------
    let (et, eb) = st
        .bytes("model.language_model.embed_tokens.weight")
        .expect("embed");
    assert_eq!(et.dtype, StDtype::Bf16);
    let mut hst: Vec<Vec<f32>> = ids
        .iter()
        .map(|&id| {
            let row = bf16_to_f32(&eb[id as usize * h * 2..(id as usize + 1) * h * 2]);
            let mut s = Vec::with_capacity(hw);
            for _ in 0..c.hc_count {
                s.extend_from_slice(&row);
            }
            s
        })
        .collect();

    dump.put_rows(usize::MAX, "h_embed", &hst);

    let rope = YarnRope::new(
        c.rotary_dim,
        c.rope_theta,
        1.0,
        c.max_pos,
        0.0,
        1.0,
        32.0,
        1.0,
    );
    let sections: [u32; 4] = [11, 11, 10, 0];
    // text: all four mrope axes equal the token index
    let mut positions = vec![0f32; 4 * n];
    for t in 0..n {
        for ax in 0..4 {
            positions[ax * n + t] = t as f32;
        }
    }

    for li in 0..c.n_layer {
        let p = format!("model.language_model.layers.{li}");
        eprint!("layer {li} ({:?})...\r", c.blocks[li]);

        // ---- PLE (decoder layer 1): H += ple_out -------------------------
        if c.ple_layers.contains(&li) {
            let pl = format!("{p}.ple");
            let emb_p = format!("{pl}.ple_embedding");
            let mult = i64s(&st, &format!("{emb_p}.layer_multipliers"));
            let sizes = i64s(&st, &format!("{emb_p}.ngram_heads_vocab_sizes"));
            let offs = i64s(&st, &format!("{emb_p}.ngram_heads_offsets"));
            let scale = {
                let (t, b) = st
                    .bytes(&format!("{emb_p}.ngram_embedding.weight_scale"))
                    .unwrap();
                assert_eq!(t.dtype, StDtype::Bf16);
                bf16_to_f32(b)[0]
            };
            let width = c.ple_embed / c.ple_heads(); // 160
            let rows_per_shard = 2_500_012usize;
            let norm_key_w = tensor_f32(&st, &format!("{pl}.norm_key.weight"), hw);
            let norm_query_w = tensor_f32(&st, &format!("{pl}.norm_query.weight"), hw);
            let norm_conv_w = tensor_f32(&st, &format!("{pl}.norm_conv.weight"), hw);
            let conv_w = tensor_f32(&st, &format!("{pl}.conv1d.weight"), hw * c.ple_conv);
            // token stream with the 2-token EOS priming
            let eos = c.bos_id as i64; // 248044 <|endoftext|> primes the window
            let mut stream = vec![eos; 2];
            stream.extend(ids.iter().map(|&i| i as i64));
            let mut gvs: Vec<Vec<f32>> = Vec::with_capacity(n);
            for (t, ht) in hst.iter().enumerate() {
                let w3 = rq::ple_window(&stream, t + 2, eos);
                let row_ids = rq::ple_ngram_ids(&w3, &mult, &sizes, &offs, c.heads_per_ngram);
                let mut emb = Vec::with_capacity(c.ple_embed);
                for &rid in &row_ids {
                    let rid = rid as usize;
                    let (sh, local) = (rid / rows_per_shard, rid % rows_per_shard);
                    let (_, sb) = st
                        .bytes(&format!("{emb_p}.ngram_embedding.shard_{sh}.weight"))
                        .unwrap_or_else(|| panic!("ple shard {sh} missing"));
                    for &b in &sb[local * width..(local + 1) * width] {
                        emb.push(rq::e4m3_to_f32(b) * scale);
                    }
                }
                let key = bf16_matvec(&st, &format!("{pl}.key_proj.weight"), &emb, hw, h);
                let value = bf16_matvec(&st, &format!("{pl}.value_proj.weight"), &emb, h, h);
                let gv = rq::ple_gate(
                    ht,
                    &key,
                    &value,
                    &norm_key_w,
                    &norm_query_w,
                    c.hc_count,
                    eps,
                );
                gvs.push(gv);
            }
            // conv over the sequence of norm_conv(gv); out = gv + conv
            let mut seq = vec![0f32; n * hw];
            for t in 0..n {
                let mut ng = gvs[t].clone();
                rq::group_rms_norm_1p(&mut ng, &norm_conv_w, c.hc_count, eps);
                seq[t * hw..(t + 1) * hw].copy_from_slice(&ng);
            }
            rq::conv1d_causal_silu(&mut seq, &conv_w, n, hw, c.ple_conv, 3);
            dump.put_rows(li, "ple_gv", &gvs);
            dump.put(li, "ple_conv", &seq);
            for t in 0..n {
                for d in 0..hw {
                    hst[t][d] += gvs[t][d] + seq[t * hw + d];
                }
            }
            dump.put_rows(li, "h_ple", &hst);
        }

        // ---- attn hyper-connection mix -----------------------------------
        let ah = format!("{p}.attn_hyper_connection");
        let a_norm = tensor_f32(&st, &format!("{ah}.hc_norm.weight"), hw);
        let a_down = tensor_f32(
            &st,
            &format!("{ah}.input_mix_weight_down.weight"),
            c.hc_lowrank * hw,
        );
        let a_up = tensor_f32(
            &st,
            &format!("{ah}.input_mix_weight_up.weight"),
            hw * c.hc_lowrank,
        );
        let a_inj = tensor_f32(
            &st,
            &format!("{ah}.block_inject_weight.weight"),
            c.hc_count * hw,
        );
        let mut block_in: Vec<Vec<f32>> = Vec::with_capacity(n);
        let mut injs: Vec<Vec<f32>> = Vec::with_capacity(n);
        for ht in &hst {
            let (bi, inj) = rq::hc_mix(
                ht,
                &a_norm,
                &a_down,
                &a_up,
                Some(&a_inj),
                c.hc_count,
                c.hc_lowrank,
                eps,
            );
            block_in.push(bi);
            injs.push(inj.unwrap());
        }
        dump.put_rows(li, "attn_bi", &block_in);
        dump.put_rows(li, "attn_inj", &injs);

        // ---- mixer -------------------------------------------------------
        let mixer_out: Vec<Vec<f32>> = match c.blocks[li] {
            Qwen4ExpBlock::Gdn => {
                let g = format!("{p}.linear_attn");
                let (kd, vd) = (c.gdn_k_dim, c.gdn_v_dim);
                let (hk, hv) = (c.gdn_k_heads, c.gdn_v_heads);
                let qkv_rows = c.gdn_qkv_rows();
                let mut qkv_seq = vec![0f32; n * qkv_rows];
                let mut z_seq = vec![0f32; n * c.gdn_z_rows()];
                let mut ax = vec![0f32; n * hv];
                let mut bx = vec![0f32; n * hv];
                for t in 0..n {
                    let x = &block_in[t];
                    qkv_seq[t * qkv_rows..(t + 1) * qkv_rows].copy_from_slice(&bf16_matvec(
                        &st,
                        &format!("{g}.in_proj_qkv.weight"),
                        x,
                        qkv_rows,
                        h,
                    ));
                    z_seq[t * c.gdn_z_rows()..(t + 1) * c.gdn_z_rows()].copy_from_slice(
                        &bf16_matvec(&st, &format!("{g}.in_proj_z.weight"), x, c.gdn_z_rows(), h),
                    );
                    ax[t * hv..(t + 1) * hv].copy_from_slice(
                        &tensor_f32(&st, &format!("{g}.in_proj_a.weight"), hv * h)
                            .chunks(h)
                            .map(|r| r.iter().zip(x.iter()).map(|(a, b)| a * b).sum::<f32>())
                            .collect::<Vec<_>>(),
                    );
                    bx[t * hv..(t + 1) * hv].copy_from_slice(
                        &tensor_f32(&st, &format!("{g}.in_proj_b.weight"), hv * h)
                            .chunks(h)
                            .map(|r| r.iter().zip(x.iter()).map(|(a, b)| a * b).sum::<f32>())
                            .collect::<Vec<_>>(),
                    );
                }
                let conv_w = tensor_f32(&st, &format!("{g}.conv1d.weight"), qkv_rows * c.gdn_conv);
                rq::conv1d_causal_silu(&mut qkv_seq, &conv_w, n, qkv_rows, c.gdn_conv, 1);
                // split + widen 16 qk heads to 48 v heads by REPEAT_INTERLEAVE
                // (modeling_qwen3_5.py:504: qk head j serves v heads 3j..3j+3;
                // the shipped kernel's % mapping is for the GGUF lane's
                // load-permuted heads, not raw safetensors planes)
                let mut q = vec![0f32; n * hv * kd];
                let mut k = vec![0f32; n * hv * kd];
                let mut v = vec![0f32; n * hv * vd];
                for t in 0..n {
                    let row = &qkv_seq[t * qkv_rows..(t + 1) * qkv_rows];
                    for vh in 0..hv {
                        let kh = vh / (hv / hk);
                        q[(t * hv + vh) * kd..(t * hv + vh + 1) * kd]
                            .copy_from_slice(&row[kh * kd..(kh + 1) * kd]);
                        k[(t * hv + vh) * kd..(t * hv + vh + 1) * kd]
                            .copy_from_slice(&row[hk * kd + kh * kd..hk * kd + (kh + 1) * kd]);
                    }
                    v[t * hv * vd..(t + 1) * hv * vd]
                        .copy_from_slice(&row[2 * hk * kd..2 * hk * kd + hv * vd]);
                }
                let a_log = tensor_f32(&st, &format!("{g}.A_log"), hv);
                let dt_bias = tensor_f32(&st, &format!("{g}.dt_bias"), hv);
                let (gg, beta) = rq::gdn_gates(&ax, &bx, &a_log, &dt_bias, n, hv);
                let mut state = vec![0f32; hv * kd * vd];
                let mut out = vec![0f32; n * hv * vd];
                gated_delta_recurrent(&q, &k, &v, &gg, &beta, &mut state, &mut out, n, hv, kd);
                let norm_w = tensor_f32(&st, &format!("{g}.norm.weight"), vd);
                (0..n)
                    .map(|t| {
                        let mut o = out[t * hv * vd..(t + 1) * hv * vd].to_vec();
                        let z = &z_seq[t * c.gdn_z_rows()..(t + 1) * c.gdn_z_rows()];
                        rq::gdn_gated_norm(&mut o, z, &norm_w, vd, eps);
                        bf16_matvec(&st, &format!("{g}.out_proj.weight"), &o, h, c.gdn_z_rows())
                    })
                    .collect()
            }
            Qwen4ExpBlock::Attention => {
                let a = format!("{p}.self_attn");
                let (nh, nkv, hd) = (c.n_heads, c.n_kv_heads, c.head_dim);
                let mut q_full = vec![0f32; n * nh * 2 * hd];
                let mut kx = vec![0f32; n * nkv * hd];
                let mut vx = vec![0f32; n * nkv * hd];
                for t in 0..n {
                    let x = &block_in[t];
                    q_full[t * nh * 2 * hd..(t + 1) * nh * 2 * hd].copy_from_slice(&bf16_matvec(
                        &st,
                        &format!("{a}.q_proj.weight"),
                        x,
                        nh * 2 * hd,
                        h,
                    ));
                    kx[t * nkv * hd..(t + 1) * nkv * hd].copy_from_slice(&bf16_matvec(
                        &st,
                        &format!("{a}.k_proj.weight"),
                        x,
                        nkv * hd,
                        h,
                    ));
                    vx[t * nkv * hd..(t + 1) * nkv * hd].copy_from_slice(&bf16_matvec(
                        &st,
                        &format!("{a}.v_proj.weight"),
                        x,
                        nkv * hd,
                        h,
                    ));
                }
                // (1+w) Gemma norms: pass w+1 into the plain-w reference
                let qn: Vec<f32> = tensor_f32(&st, &format!("{a}.q_norm.weight"), hd)
                    .iter()
                    .map(|w| w + 1.0)
                    .collect();
                let kn: Vec<f32> = tensor_f32(&st, &format!("{a}.k_norm.weight"), hd)
                    .iter()
                    .map(|w| w + 1.0)
                    .collect();
                let mut out = vec![0f32; n * nh * hd];
                gated_attention_core(
                    &q_full, &kx, &vx, &qn, &kn, &positions, &sections, &mut out, n, nh, nkv, hd,
                    &rope, eps,
                );
                (0..n)
                    .map(|t| {
                        bf16_matvec(
                            &st,
                            &format!("{a}.o_proj.weight"),
                            &out[t * nh * hd..(t + 1) * nh * hd],
                            h,
                            nh * hd,
                        )
                    })
                    .collect()
            }
        };
        dump.put_rows(li, "mix_out", &mixer_out);
        for t in 0..n {
            rq::hc_combine(&mut hst[t], &mixer_out[t], &injs[t], c.hc_count);
        }
        dump.put_rows(li, "h_mid", &hst);

        // ---- mlp hyper-connection mix + MoE ------------------------------
        let mh = format!("{p}.mlp_hyper_connection");
        let m_norm = tensor_f32(&st, &format!("{mh}.hc_norm.weight"), hw);
        let m_down = tensor_f32(
            &st,
            &format!("{mh}.input_mix_weight_down.weight"),
            c.hc_lowrank * hw,
        );
        let m_up = tensor_f32(
            &st,
            &format!("{mh}.input_mix_weight_up.weight"),
            hw * c.hc_lowrank,
        );
        let m_inj = tensor_f32(
            &st,
            &format!("{mh}.block_inject_weight.weight"),
            c.hc_count * hw,
        );
        let router = tensor_f32(&st, &format!("{p}.mlp.gate.weight"), c.n_expert * h);
        let sh_gate_v = tensor_f32(&st, &format!("{p}.mlp.shared_expert_gate.weight"), h);
        let (mut mlp_bis, mut mlp_injs, mut moes, mut topks) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for ht in hst.iter_mut() {
            let (x, inj) = rq::hc_mix(
                ht,
                &m_norm,
                &m_down,
                &m_up,
                Some(&m_inj),
                c.hc_count,
                c.hc_lowrank,
                eps,
            );
            let inj = inj.unwrap();
            if dump.on() {
                mlp_bis.push(x.clone());
                mlp_injs.push(inj.clone());
            }
            // router: softmax over all 512 (f32), top-10, renormalize
            let mut logits = rq::matvec(&router, &x, c.n_expert, h);
            softmax(&mut logits);
            let mut idx: Vec<usize> = (0..c.n_expert).collect();
            idx.sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]));
            let top = &idx[..c.n_active];
            let wsum: f32 = top.iter().map(|&e| logits[e]).sum();
            if dump.on() {
                // [n, 2*n_active]: expert ids then their renormalized weights
                let mut row: Vec<f32> = top.iter().map(|&e| e as f32).collect();
                row.extend(top.iter().map(|&e| logits[e] / wsum));
                topks.push(row);
            }
            let mut moe = vec![0f32; h];
            for &e in top {
                let ep = format!("{p}.mlp.experts.{e}");
                let gv = nvfp4_view(&st, &format!("{ep}.gate_proj")).expect("gate");
                let uv = nvfp4_view(&st, &format!("{ep}.up_proj")).expect("up");
                let dv = nvfp4_view(&st, &format!("{ep}.down_proj")).expect("down");
                let mut act = vec![0f32; c.moe_ff];
                for (r, a) in act.iter_mut().enumerate() {
                    let gr = gv.dequant_row_f32(r);
                    let ur = uv.dequant_row_f32(r);
                    let gd: f32 = gr.iter().zip(&x).map(|(a, b)| a * b).sum();
                    let ud: f32 = ur.iter().zip(&x).map(|(a, b)| a * b).sum();
                    *a = gd * rq::sigmoid(gd) * ud; // swiglu
                }
                let w = logits[e] / wsum;
                for (r, m) in moe.iter_mut().enumerate() {
                    let dr = dv.dequant_row_f32(r);
                    *m += w * dr.iter().zip(&act).map(|(a, b)| a * b).sum::<f32>();
                }
            }
            // shared expert with sigmoid scalar gate
            let sg = rq::sigmoid(sh_gate_v.iter().zip(&x).map(|(a, b)| a * b).sum());
            let sgate = bf16_matvec(
                &st,
                &format!("{p}.mlp.shared_expert.gate_proj.weight"),
                &x,
                c.shared_ff,
                h,
            );
            let sup = bf16_matvec(
                &st,
                &format!("{p}.mlp.shared_expert.up_proj.weight"),
                &x,
                c.shared_ff,
                h,
            );
            let mut sact = vec![0f32; c.shared_ff];
            for r in 0..c.shared_ff {
                sact[r] = sgate[r] * rq::sigmoid(sgate[r]) * sup[r];
            }
            let sdown = bf16_matvec(
                &st,
                &format!("{p}.mlp.shared_expert.down_proj.weight"),
                &sact,
                h,
                c.shared_ff,
            );
            for r in 0..h {
                moe[r] += sg * sdown[r];
            }
            if dump.on() {
                moes.push(moe.clone());
            }
            rq::hc_combine(ht, &moe, &inj, c.hc_count);
        }
        dump.put_rows(li, "mlp_bi", &mlp_bis);
        dump.put_rows(li, "mlp_inj", &mlp_injs);
        dump.put_rows(li, "moe_topk", &topks);
        dump.put_rows(li, "moe_out", &moes);
        dump.put_rows(li, "h_out", &hst);
    }

    // ---- final mixer (no inject) -> lm_head ------------------------------
    let fx = "model.language_model.hyper_connection_mixer";
    let f_norm = tensor_f32(&st, &format!("{fx}.hc_norm.weight"), hw);
    let f_down = tensor_f32(
        &st,
        &format!("{fx}.input_mix_weight_down.weight"),
        c.hc_lowrank * hw,
    );
    let f_up = tensor_f32(
        &st,
        &format!("{fx}.input_mix_weight_up.weight"),
        hw * c.hc_lowrank,
    );
    let (fin, _) = rq::hc_mix(
        &hst[n - 1],
        &f_norm,
        &f_down,
        &f_up,
        None,
        c.hc_count,
        c.hc_lowrank,
        eps,
    );
    let logits = bf16_matvec(&st, "lm_head.weight", &fin, c.vocab, h);
    dump.put(usize::MAX, "fin", &fin);
    dump.put(usize::MAX, "logits", &logits);

    let mut order: Vec<usize> = (0..c.vocab).collect();
    order.sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]));
    eprintln!();
    println!("host-forward top-{topk} for final position (prompt len {n}):");
    for &i in &order[..topk] {
        println!("  token {i}  logit {:.4}", logits[i]);
    }
}
