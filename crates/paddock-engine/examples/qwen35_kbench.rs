//! Per-kernel decode microbench for the qwen3.5 DeltaNet family - the qwen
//! sibling of gptoss_kbench. Times each kernel class on the b=1 decode path in
//! isolation (real blk weights, zeroed activations), prints us/call + effective
//! GB/s, and sums a per-token estimate to compare against qwen35_profile - the
//! gap is whatever this list doesn't cover. Also A/Bs the DeltaNet prefill
//! recurrence (sequential v2 vs chunked scan) across span lengths, since the
//! r>=128 dispatch boundary was tuned on an 84-SM A6000 and this family has
//! never seen a 188-SM die.
//! Usage: qwen35_kbench   (QWEN35_GGUF/PADDOCK_PACK override paths)
// A development probe: it runs on a box its author is looking at, and a
// failure should stop it where it happened rather than be reported.
#![allow(clippy::unwrap_used)]

use std::sync::Arc;

use paddock_engine::gpu::{GpuExecutor, KvDtype};
use paddock_models::mapped::MappedGguf;

fn time_us(exec: &GpuExecutor, iters: usize, mut f: impl FnMut()) -> f64 {
    for _ in 0..10 {
        f();
    }
    exec.synchronize().expect("sync");
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        f();
    }
    exec.synchronize().expect("sync");
    t0.elapsed().as_secs_f64() * 1e6 / iters as f64
}

fn row(name: &str, us: f64, bytes: usize) {
    let gbps = bytes as f64 / (us * 1e-6) / 1e9;
    println!(
        "{name:<30} {us:>9.2} us  {gbps:>8.1} GB/s  ({:.2} MB)",
        bytes as f64 / 1e6
    );
}

/// repacked Q8_0 read bytes: int8 data + f16 scale per 32-block, + activations/out
fn q8b(i: usize, o: usize) -> usize {
    i * o + i * o / 32 * 2 + i * 4 + o * 4
}

fn main() {
    let model = std::env::var("QWEN35_GGUF")
        .unwrap_or_else(|_| "C:/dev/models/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q8_0.gguf".to_string());
    let pack = std::env::var_os("PADDOCK_PACK")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../packs/cuda/build/pd-cuda-sm86.dll")
        });
    let exec = Arc::new(GpuExecutor::new(0, &pack).expect("executor"));
    let map = MappedGguf::open(std::path::Path::new(&model)).expect("open gguf");
    println!("sm_count = {}", exec.sm_count());

    // dims (9B; the 27B differs but the kernel classes are identical).
    // full-attn head_dim is 256 (attention.key_length) - Not the DeltaNet 128.
    let embd = 4096usize;
    let ff = 12288usize;
    let (n_heads, n_kv_heads, head_dim) = (16usize, 4usize, 256usize);
    let (n_k_heads, n_v_heads, s) = (16usize, 32usize, 128usize);
    let conv_dim = 2 * n_k_heads * s + n_v_heads * s; // q+k+v pre-conv = 8192
    let conv_k = 4usize;
    let value_dim = n_v_heads * s; // 4096
    let q_dim = n_heads * head_dim; // 4096
    let kv_dim = n_kv_heads * head_dim; // 1024
    let n_rot = 64usize;
    let sections = [11u32, 11, 10, 0];
    // YarnRope::kernel_params order (ext_factor 0 => plain rope, corr_* unused)
    let yarn = (
        (1e7f32).powf(-2.0 / n_rot as f32),
        1.0f32,
        0.0f32,
        0.0f32,
        0.0f32,
        1.0f32,
    );
    let max_ctx = 8192usize;

    // blk.0 is a DeltaNet layer, blk.3 a full-attn layer (interval 4)
    let in_qkv = exec
        .repack_q8(&map, "blk.0.attn_qkv.weight")
        .expect("in_qkv");
    let conv_w = exec
        .upload(&map, "blk.0.ssm_conv1d.weight")
        .expect("conv_w");
    let alpha_w = exec
        .repack_q8(&map, "blk.0.ssm_alpha.weight")
        .expect("alpha");
    let beta_w = exec.repack_q8(&map, "blk.0.ssm_beta.weight").expect("beta");
    let ssm_a = exec.upload(&map, "blk.0.ssm_a").expect("ssm_a");
    let dt_bias = exec.upload(&map, "blk.0.ssm_dt.bias").expect("dt_bias");
    let ssm_norm = exec
        .upload(&map, "blk.0.ssm_norm.weight")
        .expect("ssm_norm");
    let gate_w = exec
        .repack_q8(&map, "blk.0.attn_gate.weight")
        .expect("gate_w");
    let out_w = exec.repack_q8(&map, "blk.0.ssm_out.weight").expect("out_w");
    let attn_norm = exec
        .upload(&map, "blk.0.attn_norm.weight")
        .expect("attn_norm");
    let ffn_gate = exec
        .repack_q8(&map, "blk.0.ffn_gate.weight")
        .expect("ffn_gate");
    let ffn_up = exec.repack_q8(&map, "blk.0.ffn_up.weight").expect("ffn_up");
    let ffn_down = exec
        .repack_q8(&map, "blk.0.ffn_down.weight")
        .expect("ffn_down");
    let wq = exec.repack_q8(&map, "blk.3.attn_q.weight").expect("wq");
    let wk = exec.repack_q8(&map, "blk.3.attn_k.weight").expect("wk");
    let wv = exec.repack_q8(&map, "blk.3.attn_v.weight").expect("wv");
    let wo = exec
        .repack_q8(&map, "blk.3.attn_output.weight")
        .expect("wo");
    let q_norm = exec
        .upload(&map, "blk.3.attn_q_norm.weight")
        .expect("q_norm");
    let k_norm = exec
        .upload(&map, "blk.3.attn_k_norm.weight")
        .expect("k_norm");
    let output_w = exec.repack_q8(&map, "output.weight").expect("output");
    let vocab = output_w.dims[1];
    println!("embd={embd} ff={ff} conv_dim={conv_dim} q_dim={q_dim} kv_dim={kv_dim} vocab={vocab}");

    // activations / state (alloc_* are zeroed - zero activations keep the math
    // finite; kernel cost is data-independent for everything timed here)
    let x = exec.alloc(embd).expect("x");
    let mut xn = exec.alloc(embd).expect("xn");
    let mut mixed = exec.alloc(conv_dim).expect("mixed");
    let mut conv_win = exec.alloc((conv_k - 1) * conv_dim).expect("conv_win");
    let mut conv_out = exec.alloc(conv_dim).expect("conv_out");
    let mut dq = exec.alloc(n_v_heads * s).expect("dq");
    let mut dk = exec.alloc(n_v_heads * s).expect("dk");
    let mut dv = exec.alloc(n_v_heads * s).expect("dv");
    let mut g = exec.alloc(n_v_heads).expect("g");
    let mut beta = exec.alloc(n_v_heads).expect("beta");
    let mut state = exec.alloc(n_v_heads * s * s).expect("state");
    let mut dattn = exec.alloc(value_dim).expect("dattn");
    let mut z = exec.alloc(value_dim).expect("z");
    let mut core = exec.alloc(value_dim).expect("core");
    let mut proj = exec.alloc(embd).expect("proj");
    let mut ffn_g = exec.alloc(ff).expect("ffn_g");
    let mut ffn_u = exec.alloc(ff).expect("ffn_u");
    let mut logits = exec.alloc(vocab).expect("logits");
    let mut qg = exec.alloc(2 * q_dim).expect("qg");
    let mut q = exec.alloc(q_dim).expect("q");
    let mut gate = exec.alloc(q_dim).expect("gate");
    let mut k = exec.alloc(kv_dim).expect("k");
    let mut v = exec.alloc(kv_dim).expect("v");
    let mut qn = exec.alloc(q_dim).expect("qn");
    let mut kn = exec.alloc(kv_dim).expect("kn");
    let mut kc = exec.alloc_u8(max_ctx * kv_dim * 2).expect("kc");
    let vc = exec.alloc_u8(max_ctx * kv_dim * 2).expect("vc");
    let mut attn = exec.alloc(q_dim).expect("attn");
    // no-op sink sentinel (plain softmax), same as the model's buffer
    let sinks = exec.to_device(&vec![-1e30f32; n_heads]).expect("sinks");
    let d_mrope = exec.alloc_u32(4).expect("d_mrope");
    let eps = 1e-6f32;
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    println!("-- DeltaNet mixer (24/32 layers) --");
    let us = time_us(&exec, 400, || {
        exec.rmsnorm_batch(&x, &attn_norm.buf, &mut xn, embd, eps, 1)
            .unwrap()
    });
    row("rmsnorm(embd)", us, embd * 8);
    let t_rms = us;
    let us = time_us(&exec, 400, || {
        exec.q8_0_gemv_repacked(&in_qkv, None, &xn, &mut mixed)
            .unwrap()
    });
    row("gemv in_qkv 4096->8192", us, q8b(embd, conv_dim));
    let t_qkv = us;
    let us = time_us(&exec, 400, || {
        exec.conv_step(
            &mut conv_win,
            &mixed,
            &conv_w.buf,
            &mut conv_out,
            conv_dim,
            conv_k,
        )
        .unwrap()
    });
    // window shift r+w, x_new read, weights read, out write
    row(
        "conv_step",
        us,
        ((conv_k - 1) * 2 + 1 + conv_k + 1) * conv_dim * 4,
    );
    let t_conv = us;
    let us = time_us(&exec, 400, || {
        exec.deltanet_split_gqa_norm(
            &conv_out, &mut dq, &mut dk, &mut dv, 1, n_k_heads, n_v_heads, s,
        )
        .unwrap()
    });
    row("split_gqa_norm", us, (conv_dim + 3 * value_dim) * 4);
    let t_split = us;
    let us = time_us(&exec, 400, || {
        exec.deltanet_alpha_beta_gate(
            &alpha_w,
            &beta_w,
            &xn,
            &ssm_a.buf,
            &dt_bias.buf,
            &mut g,
            &mut beta,
            n_v_heads,
        )
        .unwrap()
    });
    row("alpha_beta_gate", us, 2 * q8b(embd, n_v_heads));
    let t_ab = us;
    let us = time_us(&exec, 400, || {
        exec.gated_delta_recurrent_v2(
            &dq, &dk, &dv, &g, &beta, None, &mut state, 0, None, &mut dattn, 1, 1, n_v_heads, s,
        )
        .unwrap()
    });
    // state read+write dominates: [H, D, D] f32 both ways
    row("gated_delta_recurrent T=1", us, 2 * n_v_heads * s * s * 4);
    let t_recur = us;
    let us = time_us(&exec, 400, || {
        exec.q8_0_gemv_repacked(&gate_w, None, &xn, &mut z).unwrap()
    });
    row("gemv attn_gate 4096->4096", us, q8b(embd, value_dim));
    let t_gate = us;
    let us = time_us(&exec, 400, || {
        exec.gated_rmsnorm(&dattn, &z, &ssm_norm.buf, &mut core, n_v_heads, s, eps)
            .unwrap()
    });
    row("gated_rmsnorm", us, 3 * value_dim * 4);
    let t_gn = us;
    let us = time_us(&exec, 400, || {
        exec.q8_0_gemv_repacked(&out_w, None, &core, &mut proj)
            .unwrap()
    });
    row("gemv ssm_out 4096->4096", us, q8b(value_dim, embd));
    let t_out = us;

    println!("-- full attention (8/32 layers) --");
    let us = time_us(&exec, 400, || {
        exec.q8_0_gemv_repacked(&wq, None, &xn, &mut qg).unwrap()
    });
    row("gemv wq 4096->8192", us, q8b(embd, 2 * q_dim));
    let t_wq = us;
    let us = time_us(&exec, 400, || {
        exec.split_qg(&qg, &mut q, &mut gate, 1, n_heads, head_dim)
            .unwrap()
    });
    row("split_qg", us, 2 * 2 * q_dim * 4);
    let t_sq = us;
    let us = time_us(&exec, 400, || {
        exec.q8_0_gemv_repacked(&wk, None, &xn, &mut k).unwrap()
    });
    row("gemv wk 4096->1024", us, q8b(embd, kv_dim));
    let t_wk = us;
    let us = time_us(&exec, 400, || {
        exec.q8_0_gemv_repacked(&wv, None, &xn, &mut v).unwrap()
    });
    row("gemv wv 4096->1024", us, q8b(embd, kv_dim));
    let t_wv = us;
    let us = time_us(&exec, 400, || {
        exec.rmsnorm_batch(&q, &q_norm.buf, &mut qn, head_dim, eps, n_heads)
            .unwrap()
    });
    row("qk-norm q (16x256)", us, 2 * q_dim * 4);
    let t_qn = us;
    let us = time_us(&exec, 400, || {
        exec.rmsnorm_batch(&k, &k_norm.buf, &mut kn, head_dim, eps, n_kv_heads)
            .unwrap()
    });
    row("qk-norm k (4x256)", us, 2 * kv_dim * 4);
    let t_kn = us;
    let us = time_us(&exec, 400, || {
        exec.mrope(
            &mut qn, &d_mrope, 1, n_heads, head_dim, n_rot, yarn, sections,
        )
        .unwrap()
    });
    row("mrope q (16 heads)", us, 2 * q_dim * 4);
    let t_mq = us;
    let us = time_us(&exec, 400, || {
        exec.mrope(
            &mut kn, &d_mrope, 1, n_kv_heads, head_dim, n_rot, yarn, sections,
        )
        .unwrap()
    });
    row("mrope k (4 heads)", us, 2 * kv_dim * 4);
    let t_mk = us;
    let pos0 = exec.to_device_u32(&[0u32]).expect("pos0");
    let slots = exec.to_device_u32(&[0u32]).expect("slots");
    let us = time_us(&exec, 400, || {
        exec.kv_append_batch(
            &kn,
            &mut kc,
            &pos0,
            Some(&slots),
            kv_dim,
            max_ctx,
            1,
            KvDtype::Fp16,
        )
        .unwrap()
    });
    row("kv_append x2", 2.0 * us, 2 * kv_dim * 6);
    let t_kv = 2.0 * us;
    let mut t_attn = 0.0;
    // FlashDecoding split A/B: the unsplit kernel is n_heads=16 blocks on a
    // 188-SM die; sweep fixed split counts (row-count-independent, so the
    // spec-vs-dense / resume bit-exactness pairs stay aligned).
    let mut attn_po = exec.alloc(n_heads * 32 * head_dim).expect("attn_po");
    let mut attn_pml = exec.alloc(n_heads * 32 * 2).expect("attn_pml");
    for ctx in [128usize, 256, 2048, 8192] {
        let pos = exec.to_device_u32(&[ctx as u32 - 1]).expect("pos");
        let us = time_us(&exec, 400, || {
            exec.attn_decode_batch(
                &qn,
                &kc,
                &vc,
                &sinks,
                &mut attn,
                &pos,
                Some(&slots),
                n_heads,
                n_kv_heads,
                head_dim,
                max_ctx,
                kv_dim,
                0,
                1,
                scale,
                KvDtype::Fp16,
            )
            .unwrap()
        });
        row(&format!("attn_decode ctx={ctx}"), us, 2 * ctx * kv_dim * 2);
        if ctx == 256 {
            t_attn = us; // short-ctx operating point, like the profile run
        }
        for splits in [8usize, 16, 32] {
            let us = time_us(&exec, 400, || {
                exec.attn_partial_batch(
                    &qn,
                    &kc,
                    &vc,
                    &mut attn_po,
                    &mut attn_pml,
                    &pos,
                    Some(&slots),
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    max_ctx,
                    kv_dim,
                    0,
                    splits,
                    1,
                    scale,
                    KvDtype::Fp16,
                )
                .unwrap();
                exec.attn_combine_batch(
                    &attn_po, &attn_pml, &sinks, &mut attn, n_heads, head_dim, splits, 1,
                )
                .unwrap();
            });
            row(
                &format!("attn_split{splits} ctx={ctx}"),
                us,
                2 * ctx * kv_dim * 2,
            );
        }
    }
    let us = time_us(&exec, 400, || {
        exec.mul_sigmoid(&mut attn, &gate, q_dim).unwrap()
    });
    row("mul_sigmoid", us, 3 * q_dim * 4);
    let t_ms = us;
    let us = time_us(&exec, 400, || {
        exec.q8_0_gemv_repacked(&wo, None, &attn, &mut proj)
            .unwrap()
    });
    row("gemv wo 4096->4096", us, q8b(q_dim, embd));
    let t_wo = us;

    println!("-- FFN (every layer) --");
    let us = time_us(&exec, 400, || {
        exec.q8_0_gemv_repacked(&ffn_gate, None, &xn, &mut ffn_g)
            .unwrap()
    });
    row("gemv ffn_gate 4096->12288", us, q8b(embd, ff));
    let t_fg = us;
    let us = time_us(&exec, 400, || {
        exec.q8_0_gemv_repacked(&ffn_up, None, &xn, &mut ffn_u)
            .unwrap()
    });
    row("gemv ffn_up 4096->12288", us, q8b(embd, ff));
    let t_fu = us;
    let us = time_us(&exec, 400, || exec.swiglu(&mut ffn_g, &ffn_u, ff).unwrap());
    row("swiglu", us, 3 * ff * 4);
    let t_sw = us;
    let us = time_us(&exec, 400, || {
        exec.q8_0_gemv_repacked(&ffn_down, None, &ffn_g, &mut proj)
            .unwrap()
    });
    row("gemv ffn_down 12288->4096", us, q8b(ff, embd));
    let t_fd = us;
    let us = time_us(&exec, 400, || exec.add(&mut proj, &x, embd).unwrap());
    row("add residual", us, embd * 12);
    let t_add = us;

    // ---- serving-B GEMM ladder: mt_dp4a vs plain mma vs K-split mma ----
    // The G4 lesson on gpt-oss: at B=9..64 the plain mma grid is N-tiles only
    // (wk here: 16 blocks) and idles the die; mma_ks z-splits K into partial
    // planes + a fixed-order combine. Measure the crossover for QWEN's shapes.
    // CAVEAT: same-weight loops are L2-resident on this 128 MB-L2 die, which
    // flatters weight-re-reading kernels (mt re-reads ceil(B/24)x) - use for
    // ordering, confirm end-to-end with the serving sweep.
    println!("-- serving B ladder (us): mt_dp4a | mma | mma_ks --");
    let bmax = 64usize;
    let sxq = exec.alloc_i8(bmax * ff).expect("sxq");
    let sxs = exec.alloc(bmax * ff / 32).expect("sxs");
    let mut sy = exec.alloc(bmax * ff).expect("sy");
    let mut part = exec.alloc(8 * ff * bmax).expect("part");
    let shapes: [(&str, &paddock_engine::gpu::RepackedQ8); 5] = [
        ("in_qkv 4096->8192", &in_qkv),
        ("wk    4096->1024", &wk),
        ("gate  4096->4096", &gate_w),
        ("ffn_g 4096->12288", &ffn_gate),
        ("ffn_d 12288->4096", &ffn_down),
    ];
    for (name, w) in shapes {
        for b in [8usize, 16, 24, 32, 48, 64] {
            let t_mt = time_us(&exec, 200, || {
                exec.q8_0_gemm_mt_dp4a(w, &sxq, &sxs, &mut sy, b).unwrap()
            });
            let t_mma = time_us(&exec, 200, || {
                exec.q8_0_gemm_mma(w, &sxq, &sxs, &mut sy, b).unwrap()
            });
            let t_ks = time_us(&exec, 200, || {
                exec.q8_0_gemm_mma_ks(w, &sxq, &sxs, &mut part, &mut sy, b)
                    .unwrap()
            });
            let best = if t_ks < t_mt && t_ks < t_mma {
                "ks"
            } else if t_mma < t_mt {
                "mma"
            } else {
                "mt"
            };
            println!("{name} B={b:<3} {t_mt:>8.1} | {t_mma:>8.1} | {t_ks:>8.1}   best={best}");
        }
    }

    println!("-- head --");
    let us = time_us(&exec, 100, || {
        exec.q8_0_gemv_repacked(&output_w, None, &xn, &mut logits)
            .unwrap()
    });
    row("lm_head 4096->248320", us, q8b(embd, vocab));
    let t_head = us;

    // per-token estimate: 24 DeltaNet layers + 8 full-attn layers + head.
    // every layer: attn_norm rms + FFN(post rms + gate + up + swiglu + down) + 2 adds
    let common = 2.0 * t_rms + t_fg + t_fu + t_sw + t_fd + 2.0 * t_add;
    let lin = common + t_qkv + t_conv + t_split + t_ab + t_recur + t_gate + t_gn + t_out;
    let full = common
        + t_wq
        + t_sq
        + t_wk
        + t_wv
        + t_qn
        + t_kn
        + t_mq
        + t_mk
        + t_kv
        + t_attn
        + t_ms
        + t_wo;
    let token_us = 24.0 * lin + 8.0 * full + t_rms + t_head;
    println!(
        "estimate: lin {lin:.1} us x24 + full {full:.1} us x8 + head {t_head:.1} = {:.0} us/token ({:.1} tok/s)",
        token_us,
        1e6 / token_us
    );

    // ---- DeltaNet prefill recurrence: sequential v2 vs chunked scan ----
    // The r>=128 dispatch boundary and the chunked kernel's internal shapes
    // (C=64 chunk, G=32 state columns/block) were tuned on 84 SMs.
    println!("-- DeltaNet prefill: v2 vs chunked --");
    let t_max = 2048usize;
    let big_q = exec.alloc(t_max * n_v_heads * s).expect("big_q");
    let big_k = exec.alloc(t_max * n_v_heads * s).expect("big_k");
    let big_v = exec.alloc(t_max * n_v_heads * s).expect("big_v");
    let big_g = exec.alloc(t_max * n_v_heads).expect("big_g");
    let big_b = exec.alloc(t_max * n_v_heads).expect("big_b");
    let mut big_o = exec.alloc(t_max * n_v_heads * s).expect("big_o");
    let nc_max = t_max.div_ceil(64);
    let mut dnc_dw = exec.alloc(nc_max * n_v_heads * 64 * s).expect("dnc_dw");
    let mut dnc_du = exec.alloc(nc_max * n_v_heads * 64 * s).expect("dnc_du");
    let mut dnc_aqk = exec.alloc(nc_max * n_v_heads * 64 * 64).expect("dnc_aqk");
    let mut dnc_cg = exec.alloc_f64(nc_max * n_v_heads * 64).expect("dnc_cg");
    for t in [64usize, 128, 256, 384, 448, 512, 1024, 2048] {
        let us_v2 = time_us(&exec, 20, || {
            exec.gated_delta_recurrent_v2(
                &big_q, &big_k, &big_v, &big_g, &big_b, None, &mut state, 0, None, &mut big_o, 1,
                t, n_v_heads, s,
            )
            .unwrap()
        });
        let us_ch = time_us(&exec, 20, || {
            exec.gated_delta_chunked(
                &big_q,
                &big_k,
                &big_v,
                &big_g,
                &big_b,
                &mut state,
                0,
                &mut big_o,
                &mut dnc_dw,
                &mut dnc_du,
                &mut dnc_aqk,
                &mut dnc_cg,
                t,
                n_v_heads,
                s,
            )
            .unwrap()
        });
        println!(
            "T={t:<5} v2 {us_v2:>9.1} us ({:>6.1} tok/ms)   chunked {us_ch:>9.1} us ({:>6.1} tok/ms)  ratio {:.2}x",
            t as f64 / us_v2 * 1000.0,
            t as f64 / us_ch * 1000.0,
            us_v2 / us_ch
        );
    }

    // prefill conv at span shapes
    println!("-- prefill causal conv --");
    let big_x = exec.alloc(t_max * conv_dim).expect("big_x");
    let mut big_co = exec.alloc(t_max * conv_dim).expect("big_co");
    for t in [128usize, 512, 2048] {
        let us = time_us(&exec, 50, || {
            exec.causal_conv1d_silu(&big_x, &conv_w.buf, &mut big_co, t, conv_dim, conv_k)
                .unwrap()
        });
        row(&format!("causal_conv1d T={t}"), us, t * conv_dim * 2 * 4);
    }
}
