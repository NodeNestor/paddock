//! Per-kernel decode microbench: time each kernel class on the b=1 gpt-oss
//! decode path in isolation (real blk.0 weights, zeroed activations) and print
//! us/call + effective GB/s. Sums a per-token estimate to compare against
//! gptoss_decode_bench - the gap is whatever this list doesn't cover.
//! Usage: gptoss_kbench   (PADDOCK_MODEL/PADDOCK_PACK override paths)

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
        "{name:<28} {us:>9.1} us  {gbps:>8.1} GB/s  ({:.2} MB)",
        bytes as f64 / 1e6
    );
}

fn main() {
    let model_path = std::env::var_os("PADDOCK_MODEL")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            let home = std::env::var("USERPROFILE")
                .or_else(|_| std::env::var("HOME"))
                .expect("USERPROFILE or HOME");
            std::path::PathBuf::from(home).join("paddock/models/gpt-oss-20b-mxfp4.gguf")
        });
    let pack = std::env::var_os("PADDOCK_PACK")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from("packs/cuda/build/pd-cuda-sm86.dll"));
    let exec = Arc::new(GpuExecutor::new(0, &pack).expect("executor"));
    let map = MappedGguf::open(&model_path).expect("open gguf");

    // blk.0 weights, real bytes
    let wq = exec.upload_raw(&map, "blk.0.attn_q.weight").expect("wq");
    let bq = exec.upload(&map, "blk.0.attn_q.bias").expect("bq");
    let wk = exec.upload_raw(&map, "blk.0.attn_k.weight").expect("wk");
    let bk = exec.upload(&map, "blk.0.attn_k.bias").expect("bk");
    let wo = exec
        .upload_raw(&map, "blk.0.attn_output.weight")
        .expect("wo");
    let bo = exec.upload(&map, "blk.0.attn_output.bias").expect("bo");
    let attn_norm = exec
        .upload(&map, "blk.0.attn_norm.weight")
        .expect("attn_norm");
    let router_w = exec
        .upload(&map, "blk.0.ffn_gate_inp.weight")
        .expect("router_w");
    let router_b = exec
        .upload(&map, "blk.0.ffn_gate_inp.bias")
        .expect("router_b");
    let gate_exps_q = exec
        .upload_raw(&map, "blk.0.ffn_gate_exps.weight")
        .expect("gate");
    let up_exps_q = exec
        .upload_raw(&map, "blk.0.ffn_up_exps.weight")
        .expect("up");
    let down_exps_q = exec
        .upload_raw(&map, "blk.0.ffn_down_exps.weight")
        .expect("down");
    let gate_b = exec
        .upload(&map, "blk.0.ffn_gate_exps.bias")
        .expect("gate_b");
    let up_b = exec.upload(&map, "blk.0.ffn_up_exps.bias").expect("up_b");
    let down_b = exec
        .upload(&map, "blk.0.ffn_down_exps.bias")
        .expect("down_b");
    let sinks = exec.upload(&map, "blk.0.attn_sinks.weight").expect("sinks");
    let output_r = exec.repack_q8(&map, "output.weight").expect("output_r");
    let gate_exps = exec.repack_mxfp4(&gate_exps_q).expect("repack gate");
    let up_exps = exec.repack_mxfp4(&up_exps_q).expect("repack up");
    let down_exps = exec.repack_mxfp4(&down_exps_q).expect("repack down");

    let embd = wq.dims[0]; // 2880
    let q_dim = wq.dims[1]; // 4096
    let kv_dim = wk.dims[1]; // 512
    let vocab = output_r.dims[1];
    let (n_heads, n_kv_heads, head_dim) = (64usize, 8usize, 64usize);
    let n_experts = *gate_exps_q.dims.last().unwrap(); // 32 on the 20b, 128 on the 120b
    let (n_active, ff) = (4usize, embd);
    let max_ctx = 4096usize;
    println!(
        "embd={embd} q_dim={q_dim} kv_dim={kv_dim} vocab={vocab} ff={ff} n_experts={n_experts}"
    );

    // g||u fused ILV plane: the dp4a/dp4a_b/mmq/bs gate_up
    // kernels stream every weight byte through gate_data at the fused 128 B
    // pair pitch - up_data is never dereferenced. Feeding them the plain
    // per-plane repacks (as this bench once did) reads 2x past the
    // plain buffer and times garbage traffic. The sorted f32/TC pair below
    // keeps the plain planes - it is the one plain-layout reader left.
    use paddock_engine::gpu::RepackedMxfp4;
    let gate_src = exec.repack_mxfp4(&gate_exps_q).expect("repack gate ilv");
    let up_src = exec.repack_mxfp4(&up_exps_q).expect("repack up ilv");
    let gu = exec
        .gu_interleave(&gate_src, &up_src, embd / 32, n_experts * ff)
        .expect("gu");
    let RepackedMxfp4 {
        data: _gate_drop,
        scale: gate_scale,
    } = gate_src;
    let RepackedMxfp4 {
        data: _up_drop,
        scale: up_scale,
    } = up_src;
    let gate_ilv = RepackedMxfp4 {
        data: gu,
        scale: gate_scale,
    };
    let up_ilv = RepackedMxfp4 {
        data: exec.alloc_u8(16).expect("dummy"),
        scale: up_scale,
    };

    // activations / scratch (alloc_* are zeroed)
    let x = exec.alloc(embd).expect("x");
    let mut xn = exec.alloc(embd).expect("xn");
    let mut xq = exec.alloc_i8(n_active * ff).expect("xq");
    let mut xs = exec.alloc(n_active * ff / 32).expect("xs");
    let mut y_q = exec.alloc(q_dim).expect("y_q");
    let mut y_kv = exec.alloc(kv_dim).expect("y_kv");
    let mut y_o = exec.alloc(embd).expect("y_o");
    let mut router_y = exec.alloc(n_experts).expect("router_y");
    let mut topk_idx = exec.alloc_u32(n_active).expect("topk_idx");
    let mut topk_w = exec.alloc(n_active).expect("topk_w");
    let mut gate_up = exec.alloc(n_active * ff).expect("gate_up");
    let mut resid = exec.alloc(embd).expect("resid");
    let mut logits = exec.alloc(vocab).expect("logits");
    let kc = exec.alloc_u8(max_ctx * kv_dim * 2).expect("kc");
    let vc = exec.alloc_u8(max_ctx * kv_dim * 2).expect("vc");
    let q_in = exec.alloc(q_dim).expect("q_in");
    let mut attn_o = exec.alloc(n_heads * 16 * head_dim).expect("attn_o");
    let mut attn_ml = exec.alloc(n_heads * 16 * 2).expect("attn_ml");
    let mut attn_out = exec.alloc(q_dim).expect("attn_out");

    // valid expert ids via the real router (zero activation -> bias argmax)
    exec.matvec_f32_batch(&router_w, &xn, &mut router_y, 1)
        .expect("router");
    exec.moe_topk_batch(
        &router_y,
        &router_b.buf,
        n_experts,
        n_active,
        &mut topk_idx,
        &mut topk_w,
        1,
    )
    .expect("topk");
    exec.synchronize().expect("sync");

    let q8b = |i: usize, o: usize| i * o / 32 * 34 + i + i / 32 * 4 + o * 4;
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    println!("-- attention block --");
    let us = time_us(&exec, 400, || {
        exec.rmsnorm_batch(&x, &attn_norm.buf, &mut xn, embd, 1e-5, 1)
            .unwrap()
    });
    row("rmsnorm", us, embd * 8);
    let t_rms = us;
    let us = time_us(&exec, 400, || {
        exec.quantize_q8(&xn, &mut xq, &mut xs, embd).unwrap()
    });
    row("quantize_q8(embd)", us, embd * 5);
    let t_quant = us;
    let us = time_us(&exec, 400, || {
        exec.q8_0_gemv_dp4a(&wq, Some(&bq.buf), &xq, &xs, &mut y_q)
            .unwrap()
    });
    row("gemv wq 2880->4096", us, q8b(embd, q_dim));
    let t_wq = us;
    let us = time_us(&exec, 400, || {
        exec.q8_0_gemv_dp4a(&wk, Some(&bk.buf), &xq, &xs, &mut y_kv)
            .unwrap()
    });
    row("gemv wk 2880->512", us, q8b(embd, kv_dim));
    let t_wk = us;
    let us = time_us(&exec, 400, || {
        exec.q8_0_gemv_dp4a(&wo, Some(&bo.buf), &xq, &xs, &mut y_o)
            .unwrap()
    });
    row("gemv wo 4096->2880", us, q8b(q_dim, embd));
    let t_wo = us;

    let mut t_attn = 0.0;
    for (ctx, splits) in [(256usize, 8usize), (256, 16), (2048, 8), (2048, 16)] {
        let pos = exec.to_device_u32(&[ctx as u32 - 1]).expect("pos");
        let us_p = time_us(&exec, 400, || {
            exec.attn_partial_batch(
                &q_in,
                &kc,
                &vc,
                &mut attn_o,
                &mut attn_ml,
                &pos,
                None,
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
            .unwrap()
        });
        let us_c = time_us(&exec, 400, || {
            exec.attn_combine_batch(
                &attn_o,
                &attn_ml,
                &sinks.buf,
                &mut attn_out,
                n_heads,
                head_dim,
                splits,
                1,
            )
            .unwrap()
        });
        row(
            &format!("attn ctx={ctx} splits={splits}"),
            us_p + us_c,
            2 * ctx * kv_dim * 2,
        );
        if ctx == 256 && splits == 8 {
            t_attn = us_p + us_c; // ~the decode-bench operating point
        }
    }

    println!("-- MoE block --");
    let us = time_us(&exec, 400, || {
        exec.matvec_f32_batch(&router_w, &xn, &mut router_y, 1)
            .unwrap()
    });
    row("router 2880->32", us, embd * n_experts * 4);
    let t_router = us;
    let us = time_us(&exec, 400, || {
        exec.moe_topk_batch(
            &router_y,
            &router_b.buf,
            n_experts,
            n_active,
            &mut topk_idx,
            &mut topk_w,
            1,
        )
        .unwrap()
    });
    row("moe_topk", us, n_experts * 8);
    let t_topk = us;
    let moe_gu_bytes = 2 * n_active * ff * embd / 32 * 17 + 2 * n_active * ff * 4;
    let us = time_us(&exec, 200, || {
        exec.mxfp4_moe_gate_up_dp4a(
            &gate_ilv,
            &gate_b.buf,
            &up_ilv,
            &up_b.buf,
            &topk_idx,
            &xq,
            &xs,
            &mut gate_up,
            embd,
            ff,
            n_active,
            1.702,
            7.0,
        )
        .unwrap()
    });
    row("moe gate_up dp4a", us, moe_gu_bytes);
    let t_gu = us;
    let us = time_us(&exec, 400, || {
        exec.quantize_q8(&gate_up, &mut xq, &mut xs, n_active * ff)
            .unwrap()
    });
    row("quantize_q8(4*ff)", us, n_active * ff * 5);
    let t_quant4 = us;
    let moe_dn_bytes = n_active * embd * ff / 32 * 17 + n_active * ff + embd * 8;
    let us = time_us(&exec, 200, || {
        exec.mxfp4_moe_down_dp4a(
            &down_exps,
            &down_b.buf,
            &topk_idx,
            &topk_w,
            &xq,
            &xs,
            &mut resid,
            ff,
            embd,
            n_active,
        )
        .unwrap()
    });
    row("moe down dp4a", us, moe_dn_bytes);
    let t_dn = us;
    let us = time_us(&exec, 400, || exec.add(&mut resid, &y_o, embd).unwrap());
    row("add residual", us, embd * 12);
    let t_add = us;

    println!("-- repacked block-per-row A/B (nc kernel, ncols=1, no bias) --");
    let wq_r = exec.repack_q8(&map, "blk.0.attn_q.weight").expect("wq_r");
    let wk_r = exec.repack_q8(&map, "blk.0.attn_k.weight").expect("wk_r");
    let wo_r = exec
        .repack_q8(&map, "blk.0.attn_output.weight")
        .expect("wo_r");
    let us = time_us(&exec, 400, || {
        exec.q8_0_gemv_dp4a_nc(&wq_r, &xq, &xs, &mut y_q, 1)
            .unwrap()
    });
    row("nc wq 2880->4096", us, q8b(embd, q_dim));
    let us = time_us(&exec, 400, || {
        exec.q8_0_gemv_dp4a_nc(&wk_r, &xq, &xs, &mut y_kv, 1)
            .unwrap()
    });
    row("nc wk 2880->512", us, q8b(embd, kv_dim));
    let us = time_us(&exec, 400, || {
        exec.q8_0_gemv_dp4a_nc(&wo_r, &xq, &xs, &mut y_o, 1)
            .unwrap()
    });
    row("nc wo 4096->2880", us, q8b(q_dim, embd));

    // ---- prefill shapes (b = 512, the PREFILL_CHUNK): the mmq path ----
    println!("-- prefill b=512 --");
    let b512 = 512usize;
    let max_blocks = (b512 * n_active + n_experts * 31) / 32;
    let x512 = exec.alloc(b512 * q_dim.max(embd)).expect("x512");
    let mut yq512 = exec
        .alloc_u8(q_dim.max(embd).div_ceil(128) * b512 * 144)
        .expect("yq512");
    let mut y512 = exec.alloc(b512 * q_dim).expect("y512");
    let mut skfix = exec.alloc(256 * 128 * 128).expect("skfix");
    let mut idx512 = exec.alloc_u32(b512 * n_active).expect("idx512");
    let mut w512 = exec.alloc(b512 * n_active).expect("w512");
    let mut sorted_row = exec.alloc_u32(max_blocks * 32).expect("sorted_row");
    let mut sorted_slot = exec.alloc_u32(max_blocks * 32).expect("sorted_slot");
    let mut block_expert = exec.alloc_u32(max_blocks).expect("block_expert");
    let mut pxq = exec.alloc_i8(b512 * embd).expect("pxq");
    let mut pxs = exec.alloc(b512 * embd / 32).expect("pxs");
    let mut moe_xq = exec.alloc_i8(max_blocks * 32 * ff).expect("moe_xq");
    let mut moe_xs = exec.alloc(max_blocks * 32 * ff / 32).expect("moe_xs");
    let mut part512 = exec.alloc(b512 * n_active * ff).expect("part512");
    let mut resid512 = exec.alloc(b512 * embd).expect("resid512");
    // realistic expert spread: random router logits (LCG), real top-k
    let mut seed = 0x2545f4914f6cdd1du64;
    let logits512: Vec<f32> = (0..b512 * n_experts)
        .map(|_| {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 40) as f32) / ((1u64 << 24) as f32)
        })
        .collect();
    let d_logits512 = exec.to_device(&logits512).expect("logits512");
    let zero_bias = exec.alloc(n_experts).expect("zero_bias");
    exec.moe_topk_batch(
        &d_logits512,
        &zero_bias,
        n_experts,
        n_active,
        &mut idx512,
        &mut w512,
        b512,
    )
    .expect("topk512");
    exec.synchronize().expect("sync");

    let us = time_us(&exec, 200, || {
        exec.rmsnorm_batch(&x512, &attn_norm.buf, &mut xn, embd, 1e-5, 1)
            .unwrap()
    });
    row("rmsnorm b=512(x1 row)", us, embd * 8);
    let us = time_us(&exec, 200, || {
        exec.quantize_q8_mmq(&x512, &mut yq512, embd, b512).unwrap()
    });
    row("quantize_q8_mmq embd", us, b512 * embd * 5);
    let t_pq = us;
    let us = time_us(&exec, 100, || {
        exec.q8_0_gemm_mmq(&wq_r, &yq512, Some(&mut skfix), &mut y512, b512)
            .unwrap()
    });
    row("mmq wq 2880->4096 b512", us, embd * q_dim / 32 * 34);
    let t_pwq = us;
    let us = time_us(&exec, 100, || {
        exec.q8_0_gemm_mmq(&wk_r, &yq512, Some(&mut skfix), &mut y512, b512)
            .unwrap()
    });
    row("mmq wk 2880->512 b512", us, embd * kv_dim / 32 * 34);
    let t_pwk = us;
    let us = time_us(&exec, 100, || {
        exec.q8_0_gemm_mmq(&wo_r, &yq512, Some(&mut skfix), &mut y512, b512)
            .unwrap()
    });
    row("mmq wo 4096->2880 b512", us, q_dim * embd / 32 * 34);
    let t_pwo = us;
    let us = time_us(&exec, 200, || {
        exec.moe_topk_batch(
            &d_logits512,
            &zero_bias,
            n_experts,
            n_active,
            &mut idx512,
            &mut w512,
            b512,
        )
        .unwrap()
    });
    row("moe_topk b512", us, b512 * n_experts * 4);
    let t_ptopk = us;
    let us = time_us(&exec, 200, || {
        exec.moe_align(
            &idx512,
            &mut sorted_row,
            &mut sorted_slot,
            &mut block_expert,
            b512,
            n_active,
            n_experts,
            max_blocks,
        )
        .unwrap()
    });
    row("moe_align b512", us, b512 * n_active * 8);
    let t_palign = us;
    let us = time_us(&exec, 200, || {
        exec.quantize_q8(&x512, &mut pxq, &mut pxs, b512 * embd)
            .unwrap()
    });
    row("quantize_q8 512*embd", us, b512 * embd * 5);
    let t_pq2 = us;
    let moe_touch = n_experts.min(b512 * n_active); // experts touched (weight reads)
    let pgu_bytes = 2 * moe_touch * ff * embd / 32 * 17;
    let us = time_us(&exec, 50, || {
        exec.mxfp4_moe_gate_up_mmq(
            &gate_ilv,
            &gate_b.buf,
            &up_ilv,
            &up_b.buf,
            &sorted_row,
            &block_expert,
            &pxq,
            &pxs,
            &mut moe_xq,
            &mut moe_xs,
            embd,
            ff,
            max_blocks,
            1.702,
            7.0,
            1.0,
        )
        .unwrap()
    });
    row("moe gate_up_mmq b512", us, pgu_bytes);
    let t_pgu = us;
    let us = time_us(&exec, 50, || {
        exec.mxfp4_moe_down_mmq(
            &down_exps,
            &down_b.buf,
            &sorted_row,
            &sorted_slot,
            &block_expert,
            &w512,
            &moe_xq,
            &moe_xs,
            &mut part512,
            ff,
            embd,
            n_active,
            max_blocks,
        )
        .unwrap()
    });
    row("moe down_mmq b512", us, moe_touch * embd * ff / 32 * 17);
    let t_pdn = us;
    let us = time_us(&exec, 200, || {
        exec.moe_slot_combine(&part512, &mut resid512, embd, n_active, b512)
            .unwrap()
    });
    row(
        "moe_slot_combine b512",
        us,
        b512 * (n_active + 1) * embd * 4,
    );
    let t_pcomb = us;
    // probe: same rows routed to only 4 experts (1/8 the weight bytes). If
    // time collapses proportionally, the kernel is weight-fetch-bound.
    let logits4: Vec<f32> = logits512
        .chunks(n_experts)
        .flat_map(|row| {
            let mut v = vec![-1e30f32; n_experts];
            for e in 0..4.min(n_experts) {
                v[e] = row[e];
            }
            v
        })
        .collect();
    let d_logits4 = exec.to_device(&logits4).expect("logits4");
    let mut idx4 = exec.alloc_u32(b512 * n_active).expect("idx4");
    let mut w4 = exec.alloc(b512 * n_active).expect("w4");
    exec.moe_topk_batch(
        &d_logits4, &zero_bias, n_experts, n_active, &mut idx4, &mut w4, b512,
    )
    .expect("topk4");
    let mut sorted_row4 = exec.alloc_u32(max_blocks * 32).expect("sr4");
    let mut sorted_slot4 = exec.alloc_u32(max_blocks * 32).expect("ss4");
    let mut block_expert4 = exec.alloc_u32(max_blocks).expect("be4");
    exec.moe_align(
        &idx4,
        &mut sorted_row4,
        &mut sorted_slot4,
        &mut block_expert4,
        b512,
        n_active,
        n_experts,
        max_blocks,
    )
    .expect("align4");
    exec.synchronize().expect("sync");
    let us = time_us(&exec, 50, || {
        exec.mxfp4_moe_gate_up_mmq(
            &gate_ilv,
            &gate_b.buf,
            &up_ilv,
            &up_b.buf,
            &sorted_row4,
            &block_expert4,
            &pxq,
            &pxs,
            &mut moe_xq,
            &mut moe_xs,
            embd,
            ff,
            max_blocks,
            1.702,
            7.0,
            1.0,
        )
        .unwrap()
    });
    row("moe gate_up_mmq 4-expert", us, 2 * 4 * ff * embd / 32 * 17);

    // A/B: the sorted WMMA f16 tensor-core MoE (f32 activations; test-reference
    // class since  - serving routes b>dp4a_max to mmq/bs instead)
    let mut fused_sorted = exec.alloc(max_blocks * 32 * ff).expect("fused_sorted");
    let us = time_us(&exec, 50, || {
        exec.mxfp4_moe_gate_up_gemm_sorted(
            &gate_exps,
            &gate_b.buf,
            &up_exps,
            &up_b.buf,
            &sorted_row,
            &block_expert,
            &x512,
            &mut fused_sorted,
            embd,
            ff,
            max_blocks,
            1.702,
            7.0,
            true,
        )
        .unwrap()
    });
    row("moe gate_up sortedTC b512", us, pgu_bytes);
    let per_layer_p =
        t_pq + t_pwq + 2.0 * t_pwk + t_pwo + t_ptopk + t_palign + t_pq2 + t_pgu + t_pdn + t_pcomb;
    println!(
        "prefill est (no attn/rope/norm/router): {per_layer_p:.0} us/layer x 24 = {:.1} ms; llama-CUDA pp512 budget is 2.82 ms/layer TOTAL",
        24.0 * per_layer_p / 1000.0
    );

    // ---- serving shapes (B tokens, one decode step) ----
    for bsrv in [32usize, 64] {
        println!("-- serving B={bsrv} --");
        let mb = (bsrv * n_active + n_experts * 31) / 32;
        let kcb = exec.alloc_u8(bsrv * 1024 * kv_dim * 2).expect("kcb");
        let vcb = exec.alloc_u8(bsrv * 1024 * kv_dim * 2).expect("vcb");
        let posb = exec.to_device_u32(&vec![71u32; bsrv]).expect("posb");
        let mut ry = exec.alloc(bsrv * n_experts).expect("ry");
        let mut xs8 = exec.alloc_u8(b512 * embd / 32).expect("xs8");
        let mut moe_xs8 = exec.alloc_u8(max_blocks * 32 * ff / 32).expect("moe_xs8");
        let mut big_logits = exec.alloc(bsrv * vocab).expect("big_logits");
        let mut tsum = 0.0;
        let us = time_us(&exec, 200, || {
            exec.rmsnorm_batch(&x512, &attn_norm.buf, &mut xn, embd, 1e-5, bsrv)
                .unwrap()
        });
        row(&format!("rmsnorm b{bsrv}"), us, bsrv * embd * 8);
        tsum += 2.0 * us;
        let us = time_us(&exec, 200, || {
            exec.quantize_q8(&x512, &mut pxq, &mut pxs, bsrv * embd)
                .unwrap()
        });
        row(&format!("quantize b{bsrv}*embd"), us, bsrv * embd * 5);
        tsum += 2.0 * us;
        let us = time_us(&exec, 200, || {
            exec.q8_0_gemm_mt_dp4a(&wq_r, &pxq, &pxs, &mut y512, bsrv)
                .unwrap()
        });
        row(&format!("mt_dp4a wq b{bsrv}"), us, embd * q_dim / 32 * 34);
        tsum += us;
        let us = time_us(&exec, 200, || {
            exec.q8_0_gemm_mt_dp4a(&wk_r, &pxq, &pxs, &mut y512, bsrv)
                .unwrap()
        });
        row(&format!("mt_dp4a wk b{bsrv}"), us, embd * kv_dim / 32 * 34);
        tsum += 2.0 * us;
        let us = time_us(&exec, 200, || {
            exec.q8_0_gemm_mt_dp4a(&wo_r, &pxq, &pxs, &mut y512, bsrv)
                .unwrap()
        });
        row(&format!("mt_dp4a wo b{bsrv}"), us, q_dim * embd / 32 * 34);
        tsum += us;
        let us = time_us(&exec, 200, || {
            exec.q8_0_gemm_mma_ks(&wq_r, &pxq, &pxs, &mut skfix, &mut y512, bsrv)
                .unwrap()
        });
        row(&format!("mma_ks wq b{bsrv}"), us, embd * q_dim / 32 * 34);
        let us = time_us(&exec, 200, || {
            exec.q8_0_gemm_mma_ks(&wk_r, &pxq, &pxs, &mut skfix, &mut y512, bsrv)
                .unwrap()
        });
        row(&format!("mma_ks wk b{bsrv}"), us, embd * kv_dim / 32 * 34);
        let us = time_us(&exec, 200, || {
            exec.q8_0_gemm_mma_ks(&wo_r, &pxq, &pxs, &mut skfix, &mut y512, bsrv)
                .unwrap()
        });
        row(&format!("mma_ks wo b{bsrv}"), us, q_dim * embd / 32 * 34);
        let us = time_us(&exec, 200, || {
            exec.q8_0_gemm_mma(&wq_r, &pxq, &pxs, &mut y512, bsrv)
                .unwrap()
        });
        row(&format!("mma wq b{bsrv}"), us, embd * q_dim / 32 * 34);
        let touched0 = n_experts.min(bsrv * n_active);
        let us = time_us(&exec, 100, || {
            exec.mxfp4_moe_gate_up_dp4a_b(
                &gate_ilv,
                &gate_b.buf,
                &up_ilv,
                &up_b.buf,
                &idx512,
                &pxq,
                &pxs,
                &mut part512,
                embd,
                ff,
                n_active,
                bsrv,
                1.702,
                7.0,
            )
            .unwrap()
        });
        row(
            &format!("gate_up_dp4a_b b{bsrv}"),
            us,
            2 * touched0 * ff * embd / 32 * 17,
        );
        exec.quantize_q8(&part512, &mut pxq, &mut pxs, bsrv * n_active * ff)
            .unwrap();
        let us = time_us(&exec, 100, || {
            exec.mxfp4_moe_down_dp4a_b(
                &down_exps,
                &down_b.buf,
                &idx512,
                &w512,
                &pxq,
                &pxs,
                &mut resid512,
                ff,
                embd,
                n_active,
                bsrv,
            )
            .unwrap()
        });
        row(
            &format!("down_dp4a_b b{bsrv}"),
            us,
            touched0 * embd * ff / 32 * 17,
        );
        let us = time_us(&exec, 200, || {
            exec.attn_decode_batch(
                &x512,
                &kcb,
                &vcb,
                &sinks.buf,
                &mut y512,
                &posb,
                None,
                n_heads,
                n_kv_heads,
                head_dim,
                1024,
                kv_dim,
                0,
                bsrv,
                scale,
                KvDtype::Fp16,
            )
            .unwrap()
        });
        row(
            &format!("attn b{bsrv} ctx=72"),
            us,
            bsrv * 2 * 72 * kv_dim * 2,
        );
        tsum += us;
        let us = time_us(&exec, 200, || {
            exec.matvec_f32_batch(&router_w, &x512, &mut ry, bsrv)
                .unwrap()
        });
        row(&format!("router b{bsrv}"), us, embd * n_experts * 4);
        tsum += us;
        let us = time_us(&exec, 200, || {
            exec.moe_topk_batch(
                &ry,
                &router_b.buf,
                n_experts,
                n_active,
                &mut idx512,
                &mut w512,
                bsrv,
            )
            .unwrap()
        });
        row(&format!("topk b{bsrv}"), us, bsrv * n_experts * 4);
        tsum += us;
        exec.moe_topk_batch(
            &d_logits512,
            &zero_bias,
            n_experts,
            n_active,
            &mut idx512,
            &mut w512,
            bsrv,
        )
        .unwrap();
        let us = time_us(&exec, 200, || {
            exec.moe_align(
                &idx512,
                &mut sorted_row,
                &mut sorted_slot,
                &mut block_expert,
                bsrv,
                n_active,
                n_experts,
                mb,
            )
            .unwrap()
        });
        row(&format!("moe_align b{bsrv}"), us, bsrv * n_active * 8);
        tsum += us;
        let us = time_us(&exec, 200, || {
            exec.quantize_e4m3(&x512, &mut pxq, &mut xs8, bsrv * embd)
                .unwrap()
        });
        row(&format!("quantize_e4m3 b{bsrv}"), us, bsrv * embd * 5);
        tsum += us;
        let touched = n_experts.min(bsrv * n_active);
        let us = time_us(&exec, 100, || {
            exec.mxfp4_moe_gate_up_bs(
                &gate_ilv,
                &gate_b.buf,
                &up_ilv,
                &up_b.buf,
                &sorted_row,
                &block_expert,
                &pxq,
                &xs8,
                &mut moe_xq,
                &mut moe_xs8,
                embd,
                ff,
                mb,
                bsrv,
                1.702,
                7.0,
                1.0,
            )
            .unwrap()
        });
        row(
            &format!("gate_up_bs b{bsrv}"),
            us,
            2 * touched * ff * embd / 32 * 17,
        );
        tsum += us;
        let us = time_us(&exec, 100, || {
            exec.mxfp4_moe_down_bs(
                &down_exps,
                &down_b.buf,
                &sorted_row,
                &sorted_slot,
                &block_expert,
                &w512,
                &moe_xq,
                &moe_xs8,
                &mut part512,
                ff,
                embd,
                n_active,
                mb,
                bsrv,
            )
            .unwrap()
        });
        row(
            &format!("down_bs b{bsrv}"),
            us,
            touched * embd * ff / 32 * 17,
        );
        tsum += us;
        let us = time_us(&exec, 200, || {
            exec.moe_slot_combine(&part512, &mut resid512, embd, n_active, bsrv)
                .unwrap()
        });
        row(
            &format!("slot_combine b{bsrv}"),
            us,
            bsrv * (n_active + 1) * embd * 4,
        );
        tsum += us;
        exec.quantize_q8_mmq(&x512, &mut yq512, embd, bsrv).unwrap();
        let us = time_us(&exec, 50, || {
            exec.q8_0_gemm_mma(&output_r, &pxq, &pxs, &mut big_logits, bsrv)
                .unwrap()
        });
        row(&format!("lm_head mma b{bsrv}"), us, embd * vocab / 32 * 34);
        let head = us;
        let us = time_us(&exec, 50, || {
            exec.q8_0_gemm_mt_dp4a_wide(&output_r, &pxq, &pxs, &mut big_logits, bsrv)
                .unwrap()
        });
        row(&format!("lm_head wide b{bsrv}"), us, embd * vocab / 32 * 34);
        let us = time_us(&exec, 200, || {
            exec.q8_0_gemm_mt_dp4a_wide(&wq_r, &pxq, &pxs, &mut y512, bsrv)
                .unwrap()
        });
        row(&format!("wide wq b{bsrv}"), us, embd * q_dim / 32 * 34);
        let us = time_us(&exec, 200, || {
            exec.q8_0_gemm_mt_dp4a_wide(&wk_r, &pxq, &pxs, &mut y512, bsrv)
                .unwrap()
        });
        row(&format!("wide wk b{bsrv}"), us, embd * kv_dim / 32 * 34);
        let us = time_us(&exec, 200, || {
            exec.q8_0_gemm_mt_dp4a_wide(&wo_r, &pxq, &pxs, &mut y512, bsrv)
                .unwrap()
        });
        row(&format!("wide wo b{bsrv}"), us, q_dim * embd / 32 * 34);
        let us = time_us(&exec, 50, || {
            exec.q8_0_gemm_mmq(&output_r, &yq512, Some(&mut skfix), &mut big_logits, bsrv)
                .unwrap()
        });
        row(&format!("lm_head mmq b{bsrv}"), us, embd * vocab / 32 * 34);
        println!(
            "serving est B={bsrv}: {:.2} ms/layer x 24 + head {:.2} ms = {:.1} ms/step (measured {} ms)",
            tsum / 1000.0,
            head / 1000.0,
            (24.0 * tsum + head) / 1000.0,
            if bsrv == 32 { "24.0" } else { "33.9" }
        );
    }

    println!("-- head --");
    let nc_bytes = embd * vocab / 32 * 34 + vocab * 4;
    let us = time_us(&exec, 100, || {
        exec.q8_0_gemv_dp4a_nc(&output_r, &xq, &xs, &mut logits, 1)
            .unwrap()
    });
    row("lm_head 2880->vocab", us, nc_bytes);
    let t_head = us;

    // per-token estimate at the decode-bench operating point (short ctx)
    let n_layers = 24.0;
    let per_layer = 2.0 * t_rms
        + 2.0 * t_quant
        + t_quant4
        + t_wq
        + 2.0 * t_wk
        + t_wo
        + t_attn
        + t_router
        + t_topk
        + t_gu
        + t_dn
        + t_add;
    let token_us = n_layers * per_layer + t_rms + t_quant + t_head;
    println!(
        "estimate: {:.1} us/layer x {n_layers} + head {t_head:.1} = {:.0} us/token ({:.1} tok/s)",
        per_layer,
        token_us,
        1e6 / token_us
    );
}
