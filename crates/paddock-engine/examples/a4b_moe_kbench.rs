//! A4B decode-expert kernel isolation bench (88/91): the dp4a
//! token-batched pair vs the dec2 intensity twins vs the dec3 bulk-streamed
//! pair vs the tc5 e4m3 chains (f8s BM=128, f8d BM=32 decode shapes) at
//! decode shapes, synthetic weights.
//!
//! Parity gates run before timing at every r: dec3 gate_up must be BITWISE
//! dec2 (same per-lane pass pattern, same shfl tree, same GEGLU); dec3 down
//! is a declared reorder class (per-pair partials + fixed-order combine), so
//! its gate is a small max-rel bound vs dec2. The f8d chain must be BITWISE
//! the f8s chain end-to-end (same mma k-order per output - only the block
//! geometry differs); both are the e4m3 class vs dec2 (maxrel printed,
//! coherence/greedy arbitrate in serving).
//!
//! Byte model: the printed GB/s uses the no-dedup worst case (r*k routed
//! slabs) for comparability with the /88 numbers. dec3 streams each
//! touched expert once, so where pairs share experts (r=32 here: every
//! expert hit twice) its worst-case number can exceed the DRAM roof - the
//! `uniq` column is the honest unique-byte rate.
//!
//! Usage: PADDOCK_PACK=... a4b_moe_kbench

use std::sync::Arc;

use paddock_engine::gpu::{GpuExecutor, RepackedQ8};

fn time_us(exec: &GpuExecutor, iters: usize, mut f: impl FnMut()) -> f64 {
    // warm
    for _ in 0..10 {
        f();
    }
    exec.stream.synchronize().expect("sync");
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        f();
    }
    exec.stream.synchronize().expect("sync");
    t0.elapsed().as_secs_f64() * 1e6 / iters as f64
}

fn main() {
    let pack = std::env::var("PADDOCK_PACK").expect("set PADDOCK_PACK");
    let exec = Arc::new(GpuExecutor::new(0, pack.as_ref()).expect("executor"));
    let (n_e, ff, embd, k) = (128usize, 704usize, 2816usize, 8usize);
    let dec3 = exec.has_moe_dec3();
    if !dec3 {
        println!("pack has no dec3 trio - dec2/orig legs only");
    }

    // synthetic expert planes: patterned int8 data + a small set of exact f16
    // scales, so parity compares exercise real accumulate math
    let mk = |in_d: usize, out: usize, seed: usize| -> RepackedQ8 {
        let blocks = n_e * out * in_d / 32;
        let data = {
            // bytes are int8 weights; the u8 pattern covers the full range
            let host: Vec<u8> = (0..blocks * 32)
                .map(|i| ((i * 13 + 5 + seed) % 255) as u8)
                .collect();
            let mut d = exec.alloc_u8(blocks * 32).expect("data");
            exec.stream.memcpy_htod(&host, &mut d).expect("htod");
            d
        };
        let scale = {
            // exact halves: 1.0, 0.5, 0.75, 1.5, 0.25, 2.0, 1.25, 0.625
            const H: [u16; 8] = [
                0x3c00, 0x3800, 0x3a00, 0x3e00, 0x3400, 0x4000, 0x3d00, 0x3900,
            ];
            let host: Vec<u16> = (0..blocks).map(|i| H[(i + seed) % 8]).collect();
            let bytes: Vec<u8> = host.iter().flat_map(|h| h.to_le_bytes()).collect();
            let mut s = exec.alloc_u8(blocks * 2).expect("scale");
            exec.stream.memcpy_htod(&bytes, &mut s).expect("htod");
            s
        };
        RepackedQ8 {
            data,
            scale,
            dims: vec![in_d, out, n_e],
        }
    };
    let gate = mk(embd, ff, 0);
    let up = mk(embd, ff, 7);
    let down = mk(ff, embd, 3);

    // f8 chains: fused [gate|up] Q8 plane built from the split
    // planes by d2d (load.rs's split, reversed), then the pack's e4m3
    // converters - identical weights to the dec2 legs modulo the e4m3 class
    let ffp = ff.next_multiple_of(128);
    let f8 = exec.has_f8d_moe();
    if !f8 {
        println!("pack has no f8d trio - f8 legs skipped");
    }
    let f8_planes = if f8 {
        let half = ff * embd / 32; // blocks per half-plane per expert
        let frb = 2 * half;
        let mut fdata = exec.alloc_u8(n_e * frb * 32).expect("fused data");
        let mut fscale = exec.alloc_u8(n_e * frb * 2).expect("fused scale");
        for e in 0..n_e {
            for (src, off) in [(&gate, 0usize), (&up, half)] {
                let sv = src
                    .data
                    .try_slice(e * half * 32..(e + 1) * half * 32)
                    .expect("sv");
                let mut dv = fdata
                    .try_slice_mut((e * frb + off) * 32..(e * frb + off + half) * 32)
                    .expect("dv");
                exec.stream.memcpy_dtod(&sv, &mut dv).expect("d2d");
                let ss = src
                    .scale
                    .try_slice(e * half * 2..(e + 1) * half * 2)
                    .expect("ss");
                let mut ds = fscale
                    .try_slice_mut((e * frb + off) * 2..(e * frb + off + half) * 2)
                    .expect("ds");
                exec.stream.memcpy_dtod(&ss, &mut ds).expect("d2d");
            }
        }
        let fused = RepackedQ8 {
            data: fdata,
            scale: fscale,
            dims: vec![embd, 2 * ff, n_e],
        };
        let gu_f8 = exec.q8_0_to_f8w(&fused).expect("gu f8");
        let dn_f8 = exec
            .q8_0_to_f8w_pad(&down, ff / 32, ffp / 32)
            .expect("dn f8");
        Some((gu_f8, dn_f8))
    } else {
        None
    };

    // PD_KB_ONLY="u64:32" runs a single (pattern, r) cell - the ncu hook.
    let only = std::env::var("PD_KB_ONLY").ok();
    for (pat, r) in [
        ("uni", 1usize),
        ("uni", 4),
        ("uni", 8),
        ("uni", 32),
        ("hot", 4),
        ("hot", 8),
        ("u64", 32),
        ("u48", 8),
        // r=128: the spec-verify shape (32 slots x k+1=4 rows -> 1024 pairs,
        // the >=512 f8s band) - prices v2-vs-f8s for the spec cells
        ("uni", 128),
        // r=256 (2048 pairs): the v2-vs-f8s crossover probe for the band
        // routing bound (PADDOCK_QMMA2_MAX)
        ("uni", 256),
        // prefill-chunk widths (4k/8k pairs): does v2 keep beating f8s at
        // the wave shapes the imax prefill band runs?
        ("uni", 512),
        ("uni", 1024),
        // 16384 pairs: the c32 BURST prefill wave - the serve's
        // ILLEGAL_ADDRESS shape
        ("uni", 2048),
    ] {
        if let Some(o) = &only
            && *o != format!("{pat}:{r}")
        {
            continue;
        }
        // uni: (row*k + slot)*5 % 128 - perfectly spread (the pattern that
        // HID the skew problem). hot: every row picks the same 8
        // experts (slot*16) - uniq_e=8 at any r, np maxed out.
        let idx_host: Vec<u32> = (0..r * k)
            .map(|i| {
                if pat == "hot" {
                    ((i % k) * 16) as u32
                } else if pat == "u64" {
                    // real c32 routing shape (PADDOCK_MOE_UNIQ, 08-25): ~64
                    // uniq experts x ~4 pairs each
                    (((i >> 2) * 2) % n_e) as u32
                } else if pat == "u48" {
                    // real c8 shape: 64 pairs over ~48 uniq experts
                    ((i % 48) * 2) as u32
                } else {
                    ((i * 5) % n_e) as u32
                }
            })
            .collect();
        let uniq_e = {
            let mut seen = vec![false; n_e];
            idx_host.iter().for_each(|&e| seen[e as usize] = true);
            seen.iter().filter(|&&s| s).count()
        };
        let idx = {
            let mut d = exec.alloc_u32(r * k).expect("idx");
            exec.stream.memcpy_htod(&idx_host, &mut d).expect("htod");
            d
        };
        let w = {
            let host: Vec<f32> = (0..r * k).map(|i| 0.0625 * ((i % 4) + 1) as f32).collect();
            let mut d = exec.stream.alloc_zeros::<f32>(r * k).expect("w");
            exec.stream.memcpy_htod(&host, &mut d).expect("htod");
            d
        };
        let xq = {
            let host: Vec<i8> = (0..r * embd)
                .map(|i| ((i * 37 + 11) % 255) as u8 as i8)
                .collect();
            let mut d = exec.alloc_i8(r * embd).expect("xq");
            exec.stream.memcpy_htod(&host, &mut d).expect("htod");
            d
        };
        let xs = {
            let host: Vec<f32> = (0..r * embd / 32)
                .map(|i| 0.01 * ((i % 7) + 1) as f32)
                .collect();
            let mut d = exec.stream.alloc_zeros::<f32>(r * embd / 32).expect("xs");
            exec.stream.memcpy_htod(&host, &mut d).expect("htod");
            d
        };
        let mut fused = exec.stream.alloc_zeros::<f32>(r * k * ff).expect("fused");
        let mut fq = exec.alloc_i8(r * k * ff).expect("fq");
        let mut fs = exec.stream.alloc_zeros::<f32>(r * k * ff / 32).expect("fs");
        let mut out = exec.stream.alloc_zeros::<f32>(r * embd).expect("out");

        // dec3 CSR (moe_align at BM=2 - skew balance) + partials
        let mb8 = (r * k + n_e).div_ceil(2);
        let mut srow = exec.alloc_u32(mb8 * 2).expect("srow");
        let mut sslot = exec.alloc_u32(mb8 * 2).expect("sslot");
        let mut bexp = exec.alloc_u32(mb8).expect("bexp");
        let mut part = exec.stream.alloc_zeros::<f32>(r * k * embd).expect("part");

        // ---- parity gates ---------------------------------------------------
        // dec3's reorder tripwire is calibrated for r <= 32 (5e-3); at the
        // r=128 spec-verify cell the cross-slot spread legitimately exceeds
        // it - skip the gate there (dec3 is not under test at that shape).
        if dec3 && r <= 32 {
            exec.q8_0_moe_gu_dec2_geglu(&gate, &up, &idx, &xq, &xs, &mut fused, k, r)
                .expect("gu dec2");
            let ref2: Vec<f32> = exec.stream.clone_dtoh(&fused).expect("dtoh");
            exec.moe_align_bm(&idx, &mut srow, &mut sslot, &mut bexp, r, k, n_e, 2, mb8)
                .expect("align");
            exec.q8_0_moe_gu_dec3_geglu(
                &gate,
                &up,
                &bexp,
                &srow,
                &sslot,
                &xq,
                &xs,
                &mut fused,
                k,
                mb8,
                r * k,
            )
            .expect("gu dec3");
            let ref3: Vec<f32> = exec.stream.clone_dtoh(&fused).expect("dtoh");
            let mism = ref2
                .iter()
                .zip(&ref3)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            assert_eq!(
                mism, 0,
                "r={r}: gu dec3 not bitwise dec2 ({mism} mismatches)"
            );

            exec.quantize_q8(&fused, &mut fq, &mut fs, r * k * ff)
                .expect("quant");
            exec.q8_0_moe_dn_dec2(&down, &idx, &w, &fq, &fs, &mut out, k, r)
                .expect("dn dec2");
            let o2: Vec<f32> = exec.stream.clone_dtoh(&out).expect("dtoh");
            exec.q8_0_moe_dn_dec3(
                &down,
                &bexp,
                &srow,
                &sslot,
                &w,
                &fq,
                &fs,
                &mut part,
                k,
                mb8,
                r * k,
            )
            .expect("dn dec3");
            exec.moe_combine_dec3(&part, &mut out, embd, k, r)
                .expect("combine");
            let o3: Vec<f32> = exec.stream.clone_dtoh(&out).expect("dtoh");
            let maxrel = o2
                .iter()
                .zip(&o3)
                .map(|(a, b)| (a - b).abs() / a.abs().max(1e-3))
                .fold(0.0f32, f32::max);
            // reorder-class jitter: the cross-slot sum order changes, and the
            // max is over r*embd samples with a 1e-3 denominator floor -
            // measured 2e-5 (r=1) .. 1.1e-3 (r=32). 5e-3 is the lab
            // tripwire; greedy/coherence arbitrate in serving.
            assert!(
                maxrel < 5e-3,
                "r={r}: dn dec3 reorder drift {maxrel} > 5e-3"
            );
            println!("{pat} r={r:>2}: parity OK (gu bitwise, dn maxrel {maxrel:.2e})");
        }

        // ---- timing ---------------------------------------------------------
        // PD_KB_FAST=1: sanitizer/parity runs - 2 iters instead of 200
        let iters = if std::env::var_os("PD_KB_FAST").is_some() {
            2
        } else {
            200
        };
        let gu_o = time_us(&exec, iters, || {
            exec.q8_0_moe_gate_up_geglu(&gate, &up, &idx, &xq, &xs, &mut fused, k, r)
                .expect("gu orig");
        });
        let gu_2 = time_us(&exec, iters, || {
            exec.q8_0_moe_gu_dec2_geglu(&gate, &up, &idx, &xq, &xs, &mut fused, k, r)
                .expect("gu dec2");
        });
        let dn_o = time_us(&exec, iters, || {
            exec.q8_0_moe_down(&down, &idx, &w, &fq, &fs, &mut out, k, r)
                .expect("dn orig");
        });
        let dn_2 = time_us(&exec, iters, || {
            exec.q8_0_moe_dn_dec2(&down, &idx, &w, &fq, &fs, &mut out, k, r)
                .expect("dn dec2");
        });
        // dec3 legs carry their serving-chain overheads: align rides the gu
        // leg (it feeds both halves), combine rides the dn leg
        let (gu_3, dn_3) = if dec3 {
            let g = time_us(&exec, iters, || {
                exec.moe_align_bm(&idx, &mut srow, &mut sslot, &mut bexp, r, k, n_e, 2, mb8)
                    .expect("align");
                exec.q8_0_moe_gu_dec3_geglu(
                    &gate,
                    &up,
                    &bexp,
                    &srow,
                    &sslot,
                    &xq,
                    &xs,
                    &mut fused,
                    k,
                    mb8,
                    r * k,
                )
                .expect("gu dec3");
            });
            let d = time_us(&exec, iters, || {
                exec.q8_0_moe_dn_dec3(
                    &down,
                    &bexp,
                    &srow,
                    &sslot,
                    &w,
                    &fq,
                    &fs,
                    &mut part,
                    k,
                    mb8,
                    r * k,
                )
                .expect("dn dec3");
                exec.moe_combine_dec3(&part, &mut out, embd, k, r)
                    .expect("combine");
            });
            (g, d)
        } else {
            (f64::NAN, f64::NAN)
        };

        // bytes touched per launch: worst case (no dedup) and unique-expert
        let gu_mb = (r * k * 2 * ff * embd) as f64 / 1e6 * 1.0625;
        let dn_mb = (r * k * ff * embd) as f64 / 1e6 * 1.0625;
        let gu_ub = (uniq_e * 2 * ff * embd) as f64 / 1e6 * 1.0625;
        let dn_ub = (uniq_e * ff * embd) as f64 / 1e6 * 1.0625;
        println!(
            "{pat} r={r:>2}: gate_up orig {gu_o:8.1}us ({:6.0} GB/s) dec2 {gu_2:8.1}us ({:6.0} GB/s) | \
             down orig {dn_o:8.1}us ({:6.0} GB/s) dec2 {dn_2:8.1}us ({:6.0} GB/s)",
            gu_mb / gu_o * 1e3,
            gu_mb / gu_2 * 1e3,
            dn_mb / dn_o * 1e3,
            dn_mb / dn_2 * 1e3,
        );
        if dec3 {
            println!(
                "      gate_up dec3 {gu_3:8.1}us ({:6.0} GB/s, uniq {:6.0}) | \
                 down dec3 {dn_3:8.1}us ({:6.0} GB/s, uniq {:6.0})   [uniq_e={uniq_e}]",
                gu_mb / gu_3 * 1e3,
                gu_ub / gu_3 * 1e3,
                dn_mb / dn_3 * 1e3,
                dn_ub / dn_3 * 1e3,
            );
        }

        // ---- shipped sorted q8 mma pair (the c32 serving class, BM=32) ------
        // The kernels g4_moe_tail elects at r*k >= 128. Parity vs dec2 is the
        // reorder class (mma k-fold order differs); the v2 ring twins gate
        // BITWISE against these legs' outputs.
        if exec.has_q8_moe_geglu_sorted() {
            let mb32f = (r * k + n_e * 31).div_ceil(32);
            let srp = mb32f * 32;
            let mut srowq = exec.alloc_u32(srp).expect("srowq");
            let mut sslotq = exec.alloc_u32(srp).expect("sslotq");
            let mut bexpq = exec.alloc_u32(mb32f).expect("bexpq");
            let mut sfq = exec.alloc_i8(srp * ff).expect("sfq");
            let mut sfs = exec.stream.alloc_zeros::<f32>(srp * ff / 32).expect("sfs");
            exec.moe_align(&idx, &mut srowq, &mut sslotq, &mut bexpq, r, k, n_e, mb32f)
                .expect("alignq");
            exec.q8_0_moe_gate_up_mma_geglu(
                &gate, &up, &srowq, &bexpq, &xq, &xs, &mut sfq, &mut sfs, mb32f, 32,
            )
            .expect("gu qmma");
            exec.q8_0_moe_down_mma(
                &down, &srowq, &sslotq, &bexpq, &w, &sfq, &sfs, &mut part, k, mb32f, 32,
            )
            .expect("dn qmma");
            exec.stream.memset_zeros(&mut out).expect("zero");
            exec.moe_slot_combine(&part, &mut out, embd, k, r)
                .expect("combine");
            let oq: Vec<f32> = exec.stream.clone_dtoh(&out).expect("dtoh");
            exec.q8_0_moe_gu_dec2_geglu(&gate, &up, &idx, &xq, &xs, &mut fused, k, r)
                .expect("gu dec2");
            exec.quantize_q8(&fused, &mut fq, &mut fs, r * k * ff)
                .expect("quant");
            exec.q8_0_moe_dn_dec2(&down, &idx, &w, &fq, &fs, &mut out, k, r)
                .expect("dn dec2");
            let o2: Vec<f32> = exec.stream.clone_dtoh(&out).expect("dtoh");
            let maxrel = o2
                .iter()
                .zip(&oq)
                .map(|(a, b)| (a - b).abs() / a.abs().max(1e-3))
                .fold(0.0f32, f32::max);
            // align rides the gu leg (it feeds both halves), like dec3
            let gu_q = time_us(&exec, iters, || {
                exec.moe_align(&idx, &mut srowq, &mut sslotq, &mut bexpq, r, k, n_e, mb32f)
                    .expect("alignq");
                exec.q8_0_moe_gate_up_mma_geglu(
                    &gate, &up, &srowq, &bexpq, &xq, &xs, &mut sfq, &mut sfs, mb32f, 32,
                )
                .expect("gu qmma");
            });
            let dn_q = time_us(&exec, iters, || {
                exec.q8_0_moe_down_mma(
                    &down, &srowq, &sslotq, &bexpq, &w, &sfq, &sfs, &mut part, k, mb32f, 32,
                )
                .expect("dn qmma");
            });
            println!(
                "      qmma gu {gu_q:8.1}us ({:6.0} GB/s uniq) dn {dn_q:8.1}us ({:6.0}) | \
                 maxrel-vs-dec2 {maxrel:.2e} [mb {mb32f}]",
                gu_ub / gu_q * 1e3,
                dn_ub / dn_q * 1e3,
            );

            // ---- v2 ring twins: BITWISE vs the shipped pair -----------------
            // Dead-quarter fq/fs are unwritten by design, so the gu compare
            // masks to live sorted rows; part compares fully (live writes
            // only + the alloc_zeros ground).
            if exec.has_q8_moe_qmma2() {
                // shipped references (state left by the legs above)
                let ppart: Vec<f32> = exec.stream.clone_dtoh(&part).expect("dtoh");
                let pfq: Vec<i8> = exec.stream.clone_dtoh(&sfq).expect("dtoh");
                let pfs: Vec<f32> = exec.stream.clone_dtoh(&sfs).expect("dtoh");
                let srow_h: Vec<u32> = exec.stream.clone_dtoh(&srowq).expect("dtoh");

                // v2 down over the SHIPPED fq/fs: part must be bitwise
                exec.q8_0_moe_down_mma2(
                    &down, &srowq, &sslotq, &bexpq, &w, &sfq, &sfs, &mut part, k, mb32f, 32,
                )
                .expect("dn v2");
                let p2: Vec<f32> = exec.stream.clone_dtoh(&part).expect("dtoh");
                let mism = ppart
                    .iter()
                    .zip(&p2)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
                assert_eq!(
                    mism, 0,
                    "{pat} r={r}: dn v2 not bitwise ({mism} mismatches)"
                );

                // v2 gate_up: live-row fq/fs bitwise
                let mut sfq2 = exec.alloc_i8(srp * ff).expect("sfq2");
                let mut sfs2 = exec.stream.alloc_zeros::<f32>(srp * ff / 32).expect("sfs2");
                exec.q8_0_moe_gate_up_mma2_geglu(
                    &gate, &up, &srowq, &bexpq, &xq, &xs, &mut sfq2, &mut sfs2, mb32f, 32,
                )
                .expect("gu v2");
                let qfq: Vec<i8> = exec.stream.clone_dtoh(&sfq2).expect("dtoh");
                let qfs: Vec<f32> = exec.stream.clone_dtoh(&sfs2).expect("dtoh");
                let (mut mq, mut ms) = (0usize, 0usize);
                for (i, &sr) in srow_h.iter().enumerate() {
                    if sr == u32::MAX {
                        continue;
                    }
                    for j in 0..ff {
                        if pfq[i * ff + j] != qfq[i * ff + j] {
                            mq += 1;
                        }
                    }
                    for j in 0..ff / 32 {
                        if pfs[i * ff / 32 + j].to_bits() != qfs[i * ff / 32 + j].to_bits() {
                            ms += 1;
                        }
                    }
                }
                assert_eq!(
                    (mq, ms),
                    (0, 0),
                    "{pat} r={r}: gu v2 not bitwise on live rows"
                );

                // full v2 chain: dn v2 over the v2 fq (dead-quarter garbage
                // must stay confined to dead columns) -- part still bitwise
                exec.q8_0_moe_down_mma2(
                    &down, &srowq, &sslotq, &bexpq, &w, &sfq2, &sfs2, &mut part, k, mb32f, 32,
                )
                .expect("dn v2 chain");
                let p3: Vec<f32> = exec.stream.clone_dtoh(&part).expect("dtoh");
                let mism3 = ppart
                    .iter()
                    .zip(&p3)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
                assert_eq!(
                    mism3, 0,
                    "{pat} r={r}: v2 chain leaked pad garbage ({mism3})"
                );

                let gu_v = time_us(&exec, iters, || {
                    exec.moe_align(&idx, &mut srowq, &mut sslotq, &mut bexpq, r, k, n_e, mb32f)
                        .expect("alignq");
                    exec.q8_0_moe_gate_up_mma2_geglu(
                        &gate, &up, &srowq, &bexpq, &xq, &xs, &mut sfq2, &mut sfs2, mb32f, 32,
                    )
                    .expect("gu v2");
                });
                let dn_v = time_us(&exec, iters, || {
                    exec.q8_0_moe_down_mma2(
                        &down, &srowq, &sslotq, &bexpq, &w, &sfq2, &sfs2, &mut part, k, mb32f, 32,
                    )
                    .expect("dn v2");
                });
                println!(
                    "      qmma2 gu {gu_v:8.1}us ({:6.0} GB/s uniq) dn {dn_v:8.1}us ({:6.0}) | \
                 BITWISE OK (gu live rows, dn, chain)",
                    gu_ub / gu_v * 1e3,
                    dn_ub / dn_v * 1e3,
                );

                // v3t TMA twins (slots 502/503): bitwise vs v2 on the
                // CURRENT CSR (same align-race rule as the v5 leg below).
                if exec.has_q8_moe_qmma2t() {
                    exec.q8_0_moe_gate_up_mma2_geglu(
                        &gate, &up, &srowq, &bexpq, &xq, &xs, &mut sfq2, &mut sfs2, mb32f, 32,
                    )
                    .expect("gu v2 ref t");
                    exec.q8_0_moe_down_mma2(
                        &down, &srowq, &sslotq, &bexpq, &w, &sfq2, &sfs2, &mut part, k, mb32f, 32,
                    )
                    .expect("dn v2 ref t");
                    let qfq: Vec<i8> = exec.stream.clone_dtoh(&sfq2).expect("dtoh");
                    let qfs: Vec<f32> = exec.stream.clone_dtoh(&sfs2).expect("dtoh");
                    let qpart: Vec<f32> = exec.stream.clone_dtoh(&part).expect("dtoh");
                    let srow_h: Vec<u32> = exec.stream.clone_dtoh(&srowq).expect("dtoh");
                    let mut sfqt = exec.alloc_i8(srp * ff).expect("sfqt");
                    let mut sfst = exec.stream.alloc_zeros::<f32>(srp * ff / 32).expect("sfst");
                    exec.q8_0_moe_gate_up_mma2t_geglu(
                        &gate, &up, &srowq, &bexpq, &xq, &xs, &mut sfqt, &mut sfst, mb32f, 32,
                    )
                    .expect("gu v3t");
                    let tfq: Vec<i8> = exec.stream.clone_dtoh(&sfqt).expect("dtoh");
                    let tfs: Vec<f32> = exec.stream.clone_dtoh(&sfst).expect("dtoh");
                    let (mut mtq, mut mts) = (0usize, 0usize);
                    for (i, &sr) in srow_h.iter().enumerate() {
                        if sr == u32::MAX {
                            continue;
                        }
                        for j in 0..ff {
                            if qfq[i * ff + j] != tfq[i * ff + j] {
                                mtq += 1;
                            }
                        }
                        for j in 0..ff / 32 {
                            if qfs[i * ff / 32 + j].to_bits() != tfs[i * ff / 32 + j].to_bits() {
                                mts += 1;
                            }
                        }
                    }
                    if (mtq, mts) != (0, 0) {
                        // localize: histogram fq mismatches by output-row
                        // residues (j&7 = swizzle atom row, (j>>4)&7 = chunk)
                        let mut h8 = [0usize; 8];
                        let mut hc = [0usize; 8];
                        for (i, &sr) in srow_h.iter().enumerate() {
                            if sr == u32::MAX {
                                continue;
                            }
                            for j in 0..ff {
                                if qfq[i * ff + j] != tfq[i * ff + j] {
                                    h8[j & 7] += 1;
                                    hc[(j / 16) & 7] += 1;
                                }
                            }
                        }
                        println!(
                            "      v3t MISMATCH mtq={mtq} mts={mts} by j&7 {h8:?} by chunk {hc:?}"
                        );
                        let i0 = srow_h.iter().position(|&x| x != u32::MAX).unwrap();
                        println!("      row{}: v2 {:?}", i0, &qfq[i0 * ff..i0 * ff + 24]);
                        println!("      row{}: t  {:?}", i0, &tfq[i0 * ff..i0 * ff + 24]);
                    }
                    assert_eq!((mtq, mts), (0, 0), "{pat} r={r}: gu v3t not bitwise vs v2");
                    exec.q8_0_moe_down_mma2t(
                        &down, &srowq, &sslotq, &bexpq, &w, &sfq2, &sfs2, &mut part, k, mb32f, 32,
                    )
                    .expect("dn v3t");
                    let tpart: Vec<f32> = exec.stream.clone_dtoh(&part).expect("dtoh");
                    let mtp = qpart
                        .iter()
                        .zip(&tpart)
                        .filter(|(x, y)| x.to_bits() != y.to_bits())
                        .count();
                    assert_eq!(mtp, 0, "{pat} r={r}: dn v3t not bitwise vs v2 ({mtp})");
                    let gu_t = time_us(&exec, iters, || {
                        exec.moe_align(&idx, &mut srowq, &mut sslotq, &mut bexpq, r, k, n_e, mb32f)
                            .expect("alignq");
                        exec.q8_0_moe_gate_up_mma2t_geglu(
                            &gate, &up, &srowq, &bexpq, &xq, &xs, &mut sfqt, &mut sfst, mb32f, 32,
                        )
                        .expect("gu v3t");
                    });
                    let dn_t = time_us(&exec, iters, || {
                        exec.q8_0_moe_down_mma2t(
                            &down, &srowq, &sslotq, &bexpq, &w, &sfqt, &sfst, &mut part, k, mb32f,
                            32,
                        )
                        .expect("dn v3t");
                    });
                    println!(
                        "      qmma2T gu {gu_t:8.1}us ({:6.0} GB/s uniq) dn {dn_t:8.1}us ({:6.0}) | BITWISE OK (gu live rows, dn)",
                        gu_ub / gu_t * 1e3,
                        dn_ub / dn_t * 1e3,
                    );
                }

                // v5 gate_up (slot 488): bitwise vs the v2 outputs on live rows.
                // moe_align's scatter order is atomic (racy across warps), and
                // the timing loops above re-ran it 200x - so REFERENCE v2 on
                // the CURRENT CSR right here, or the compare chases a
                // permuted layout (the u48/uni128 false "not bitwise").
                if exec.has_q8_moe_mma3() {
                    exec.q8_0_moe_gate_up_mma2_geglu(
                        &gate, &up, &srowq, &bexpq, &xq, &xs, &mut sfq2, &mut sfs2, mb32f, 32,
                    )
                    .expect("gu v2 ref");
                    let qfq: Vec<i8> = exec.stream.clone_dtoh(&sfq2).expect("dtoh");
                    let qfs: Vec<f32> = exec.stream.clone_dtoh(&sfs2).expect("dtoh");
                    let srow_h: Vec<u32> = exec.stream.clone_dtoh(&srowq).expect("dtoh");
                    let mut sfq3 = exec.alloc_i8(srp * ff).expect("sfq3");
                    let mut sfs3 = exec.stream.alloc_zeros::<f32>(srp * ff / 32).expect("sfs3");
                    exec.q8_0_moe_gate_up_mma3_geglu(
                        &gate, &up, &srowq, &bexpq, &xq, &xs, &mut sfq3, &mut sfs3, mb32f, 32,
                    )
                    .expect("gu v5");
                    let vfq: Vec<i8> = exec.stream.clone_dtoh(&sfq3).expect("dtoh");
                    let vfs: Vec<f32> = exec.stream.clone_dtoh(&sfs3).expect("dtoh");
                    let (mut m3q, mut m3s) = (0usize, 0usize);
                    for (i, &sr) in srow_h.iter().enumerate() {
                        if sr == u32::MAX {
                            continue;
                        }
                        for j in 0..ff {
                            if qfq[i * ff + j] != vfq[i * ff + j] {
                                m3q += 1;
                            }
                        }
                        for j in 0..ff / 32 {
                            if qfs[i * ff / 32 + j].to_bits() != vfs[i * ff / 32 + j].to_bits() {
                                m3s += 1;
                            }
                        }
                    }
                    assert_eq!((m3q, m3s), (0, 0), "{pat} r={r}: gu v5 not bitwise vs v2");
                    let gu_5 = time_us(&exec, iters, || {
                        exec.moe_align(&idx, &mut srowq, &mut sslotq, &mut bexpq, r, k, n_e, mb32f)
                            .expect("alignq");
                        exec.q8_0_moe_gate_up_mma3_geglu(
                            &gate, &up, &srowq, &bexpq, &xq, &xs, &mut sfq3, &mut sfs3, mb32f, 32,
                        )
                        .expect("gu v5");
                    });
                    println!(
                        "      qmma3 gu {gu_5:8.1}us ({:6.0} GB/s uniq) | BITWISE vs v2 OK",
                        gu_ub / gu_5 * 1e3,
                    );
                }
            }
        }

        // ---- f8 chains: f8s BM=128 vs f8d BM=32 --------------------
        if let Some((gu_f8, dn_f8)) = f8_planes.as_ref() {
            // same real-valued x the q8 legs see (xq * xs), e4m3-quantized
            let x32 = {
                let host: Vec<f32> = (0..r * embd)
                    .map(|i| {
                        (((i * 37 + 11) % 255) as u8 as i8) as f32
                            * (0.01 * ((i / 32 % 7) + 1) as f32)
                    })
                    .collect();
                let mut d = exec.stream.alloc_zeros::<f32>(r * embd).expect("x32");
                exec.stream.memcpy_htod(&host, &mut d).expect("htod");
                d
            };
            let mut e4q = exec.alloc_i8(r * embd).expect("e4q");
            let mut e4s = exec.alloc_u8(r * embd / 32).expect("e4s");
            exec.quantize_e4m3(&x32, &mut e4q, &mut e4s, r * embd)
                .expect("qe4");

            let mb128 = (r * k + n_e * 127).div_ceil(128);
            let srp128 = mb128 * 128;
            // tight bound: every live block holds >= 1 pair, so blocks <= r*k
            // - at decode r this beats the histogram worst case by 2-16x and
            // shrinks every srp-scaled interstitial with it
            let mb32 = (r * k + n_e * 31).div_ceil(32).min(r * k);
            let srp32 = mb32 * 32;
            let mut srowf = exec.alloc_u32(srp128).expect("srowf");
            let mut sslotf = exec.alloc_u32(srp128).expect("sslotf");
            let mut bexpf = exec.alloc_u32(mb128.max(mb32)).expect("bexpf");
            let mut xg = exec.alloc_u8(srp128 * embd).expect("xg");
            let mut sg = exec.alloc_u8(srp128 * embd / 32).expect("sg");
            let mut guf = exec
                .stream
                .alloc_zeros::<f32>(srp128 * 2 * ff)
                .expect("guf");
            let mut fq8 = exec.alloc_u8(srp128 * ffp).expect("fq8");
            let mut fs8 = exec.alloc_u8(srp128 * ffp / 32).expect("fs8");
            let mut outf = exec.stream.alloc_zeros::<f32>(r * embd).expect("outf");
            let mut outd = exec.stream.alloc_zeros::<f32>(r * embd).expect("outd");

            // f8s chain once (the >=512-band serving shape at this r)
            exec.moe_align_bm(
                &idx,
                &mut srowf,
                &mut sslotf,
                &mut bexpf,
                r,
                k,
                n_e,
                128,
                mb128,
            )
            .expect("align128");
            exec.moe_gather_e4m3(&e4q, &e4s, &srowf, &mut xg, &mut sg, embd, srp128)
                .expect("gather128");
            exec.f8bs_moe_gemm_gu(
                gu_f8,
                &xg,
                &sg,
                &bexpf,
                &mut guf,
                embd,
                2 * ff,
                n_e,
                srp128,
                mb128,
            )
            .expect("f8s gu");
            exec.quantize_e4m3_geglu2_pad(&guf, &mut fq8, &mut fs8, ff, ffp, srp128)
                .expect("f8s geglu");
            exec.f8bs_moe_gemm_dn(
                dn_f8, &fq8, &fs8, &bexpf, &srowf, &sslotf, &w, &mut part, ffp, embd, n_e, srp128,
                mb128, k,
            )
            .expect("f8s dn");
            exec.stream.memset_zeros(&mut outf).expect("zero");
            exec.moe_slot_combine(&part, &mut outf, embd, k, r)
                .expect("combine");
            let of: Vec<f32> = exec.stream.clone_dtoh(&outf).expect("dtoh");

            // f8s GEMM timing while the BM=128 CSR is still the live layout
            let gu_s = time_us(&exec, iters, || {
                exec.f8bs_moe_gemm_gu(
                    gu_f8,
                    &xg,
                    &sg,
                    &bexpf,
                    &mut guf,
                    embd,
                    2 * ff,
                    n_e,
                    srp128,
                    mb128,
                )
                .expect("f8s gu");
            });
            let dn_s = time_us(&exec, iters, || {
                exec.f8bs_moe_gemm_dn(
                    dn_f8, &fq8, &fs8, &bexpf, &srowf, &sslotf, &w, &mut part, ffp, embd, n_e,
                    srp128, mb128, k,
                )
                .expect("f8s dn");
            });
            // dn_s ran over the fq8 the f8s geglu wrote - restore before the
            // d32 chain reuses the plane
            exec.quantize_e4m3_geglu2_pad(&guf, &mut fq8, &mut fs8, ff, ffp, srp128)
                .expect("f8s geglu");

            // f8d chain once (BM=32 decode shapes)
            exec.moe_align_bm(
                &idx,
                &mut srowf,
                &mut sslotf,
                &mut bexpf,
                r,
                k,
                n_e,
                32,
                mb32,
            )
            .expect("align32");
            exec.moe_gather_e4m3(&e4q, &e4s, &srowf, &mut xg, &mut sg, embd, srp32)
                .expect("gather32");
            exec.f8bs_moe_gemm_gu_d32(
                gu_f8,
                &xg,
                &sg,
                &bexpf,
                &mut guf,
                embd,
                2 * ff,
                n_e,
                srp32,
                mb32,
            )
            .expect("f8d gu");
            exec.quantize_e4m3_geglu2_pad_b(&guf, &mut fq8, &mut fs8, &bexpf, ff, ffp, 32, srp32)
                .expect("f8d geglu");
            exec.f8bs_moe_gemm_dn_d32(
                dn_f8, &fq8, &fs8, &bexpf, &srowf, &sslotf, &w, &mut part, ffp, embd, n_e, srp32,
                mb32, k,
            )
            .expect("f8d dn");
            exec.stream.memset_zeros(&mut outd).expect("zero");
            exec.moe_slot_combine(&part, &mut outd, embd, k, r)
                .expect("combine");
            let od: Vec<f32> = exec.stream.clone_dtoh(&outd).expect("dtoh");

            let mism = of
                .iter()
                .zip(&od)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            assert_eq!(
                mism, 0,
                "{pat} r={r}: f8d not bitwise f8s ({mism} mismatches)"
            );
            // e4m3-class distance vs dec2 (info only; serving gates arbitrate).
            // NOTE: the dec2 reference eats the ORIGINAL q8 activations while
            // the f8 chain eats the e4m3 requant of the same values, so this
            // is the full serving-class delta, not a kernel-parity number.
            exec.q8_0_moe_gu_dec2_geglu(&gate, &up, &idx, &xq, &xs, &mut fused, k, r)
                .expect("gu dec2");
            exec.quantize_q8(&fused, &mut fq, &mut fs, r * k * ff)
                .expect("quant");
            exec.q8_0_moe_dn_dec2(&down, &idx, &w, &fq, &fs, &mut out, k, r)
                .expect("dn dec2");
            let o2: Vec<f32> = exec.stream.clone_dtoh(&out).expect("dtoh");
            let maxrel = o2
                .iter()
                .zip(&od)
                .map(|(a, b)| (a - b).abs() / a.abs().max(1e-2))
                .fold(0.0f32, f32::max);

            // timing: GEMMs isolated; interstitials separate; chain = the
            // full serving sequence (quant + align + gather + gu + geglu +
            // dn + memset + combine). The BM=32 layout is live here (the f8d
            // parity chain built it last).
            let gu_d = time_us(&exec, iters, || {
                exec.f8bs_moe_gemm_gu_d32(
                    gu_f8,
                    &xg,
                    &sg,
                    &bexpf,
                    &mut guf,
                    embd,
                    2 * ff,
                    n_e,
                    srp32,
                    mb32,
                )
                .expect("f8d gu");
            });
            let ge_d = time_us(&exec, iters, || {
                exec.quantize_e4m3_geglu2_pad_b(
                    &guf, &mut fq8, &mut fs8, &bexpf, ff, ffp, 32, srp32,
                )
                .expect("f8d geglu");
            });
            let dn_d = time_us(&exec, iters, || {
                exec.f8bs_moe_gemm_dn_d32(
                    dn_f8, &fq8, &fs8, &bexpf, &srowf, &sslotf, &w, &mut part, ffp, embd, n_e,
                    srp32, mb32, k,
                )
                .expect("f8d dn");
            });
            let ag_d = time_us(&exec, iters, || {
                exec.moe_align_bm(
                    &idx,
                    &mut srowf,
                    &mut sslotf,
                    &mut bexpf,
                    r,
                    k,
                    n_e,
                    32,
                    mb32,
                )
                .expect("align32");
                exec.moe_gather_e4m3(&e4q, &e4s, &srowf, &mut xg, &mut sg, embd, srp32)
                    .expect("gather32");
            });
            let chain = time_us(&exec, iters, || {
                exec.quantize_e4m3(&x32, &mut e4q, &mut e4s, r * embd)
                    .expect("qe4");
                exec.moe_align_bm(
                    &idx,
                    &mut srowf,
                    &mut sslotf,
                    &mut bexpf,
                    r,
                    k,
                    n_e,
                    32,
                    mb32,
                )
                .expect("align32");
                exec.moe_gather_e4m3(&e4q, &e4s, &srowf, &mut xg, &mut sg, embd, srp32)
                    .expect("gather32");
                exec.f8bs_moe_gemm_gu_d32(
                    gu_f8,
                    &xg,
                    &sg,
                    &bexpf,
                    &mut guf,
                    embd,
                    2 * ff,
                    n_e,
                    srp32,
                    mb32,
                )
                .expect("f8d gu");
                exec.quantize_e4m3_geglu2_pad_b(
                    &guf, &mut fq8, &mut fs8, &bexpf, ff, ffp, 32, srp32,
                )
                .expect("f8d geglu");
                exec.f8bs_moe_gemm_dn_d32(
                    dn_f8, &fq8, &fs8, &bexpf, &srowf, &sslotf, &w, &mut part, ffp, embd, n_e,
                    srp32, mb32, k,
                )
                .expect("f8d dn");
                exec.stream.memset_zeros(&mut outd).expect("zero");
                exec.moe_slot_combine(&part, &mut outd, embd, k, r)
                    .expect("combine");
            });

            // honest bytes: e4m3 planes, one slab read per live BLOCK (a hot
            // expert with >32 pairs re-reads per extra block)
            // ---- prefill dn hybrid (slots 489/490): f8s-gu -> q8 remap ->
            // v2 down; compare vs the f8s chain output (e4m3-vs-q8 class =>
            // maxrel). Self-contained: re-runs the bm128 align + gather +
            // f8s gu (the f8d chain above rewrote those buffers).
            if exec.has_pf_dn_hybrid() && exec.has_q8_moe_qmma2() {
                exec.moe_align_bm(
                    &idx,
                    &mut srowf,
                    &mut sslotf,
                    &mut bexpf,
                    r,
                    k,
                    n_e,
                    128,
                    mb128,
                )
                .expect("align128h");
                exec.moe_gather_e4m3(&e4q, &e4s, &srowf, &mut xg, &mut sg, embd, srp128)
                    .expect("gatherh");
                exec.f8bs_moe_gemm_gu(
                    gu_f8,
                    &xg,
                    &sg,
                    &bexpf,
                    &mut guf,
                    embd,
                    2 * ff,
                    n_e,
                    srp128,
                    mb128,
                )
                .expect("f8s gu h");
                let mb32h = (r * k + n_e * 31).div_ceil(32);
                let mut hsrow = exec.alloc_u32(mb32h * 32).expect("hsrow");
                let mut hsslot = exec.alloc_u32(mb32h * 32).expect("hsslot");
                let mut hbexp = exec.alloc_u32(mb32h).expect("hbexp");
                exec.moe_align(&idx, &mut hsrow, &mut hsslot, &mut hbexp, r, k, n_e, mb32h)
                    .expect("align32h");
                let mut map = exec.stream.alloc_zeros::<f32>(r * k).expect("map");
                exec.moe_pair_map(&hsrow, &hsslot, &mut map, k, mb32h * 32)
                    .expect("pairmap");
                let mut hfq = exec.alloc_i8(mb32h * 32 * ff).expect("hfq");
                let mut hfs = exec
                    .stream
                    .alloc_zeros::<f32>(mb32h * 32 * ff / 32)
                    .expect("hfs");
                exec.quantize_q8_geglu_remap(
                    &guf, &srowf, &sslotf, &map, &mut hfq, &mut hfs, ff, k, srp128, 0,
                )
                .expect("remap");
                exec.q8_0_moe_down_mma2(
                    &down, &hsrow, &hsslot, &hbexp, &w, &hfq, &hfs, &mut part, k, mb32h, 32,
                )
                .expect("dn hy");
                exec.stream.memset_zeros(&mut outd).expect("zero");
                exec.moe_slot_combine(&part, &mut outd, embd, k, r)
                    .expect("combine");
                let oh: Vec<f32> = exec.stream.clone_dtoh(&outd).expect("dtoh");
                let maxrel = of
                    .iter()
                    .zip(&oh)
                    .map(|(a, b)| (a - b).abs() / a.abs().max(1e-2))
                    .fold(0.0f32, f32::max);
                let nz = oh.iter().filter(|v| **v != 0.0).count();
                println!(
                    "      hybrid dn: maxrel-vs-f8s {maxrel:.2e} nonzero {nz}/{}",
                    oh.len()
                );
                // isolate: v2-gu (q8 exact) on the same bm32 CSR -> same down
                let mut hfq2 = exec.alloc_i8(mb32h * 32 * ff).expect("hfq2");
                let mut hfs2 = exec
                    .stream
                    .alloc_zeros::<f32>(mb32h * 32 * ff / 32)
                    .expect("hfs2");
                exec.q8_0_moe_gate_up_mma2_geglu(
                    &gate, &up, &hsrow, &hbexp, &xq, &xs, &mut hfq2, &mut hfs2, mb32h, 32,
                )
                .expect("gu v2 h");
                exec.q8_0_moe_down_mma2(
                    &down, &hsrow, &hsslot, &hbexp, &w, &hfq2, &hfs2, &mut part, k, mb32h, 32,
                )
                .expect("dn v2 h");
                exec.stream.memset_zeros(&mut outd).expect("zero");
                exec.moe_slot_combine(&part, &mut outd, embd, k, r)
                    .expect("combine");
                let ov: Vec<f32> = exec.stream.clone_dtoh(&outd).expect("dtoh");
                let mr2 = of
                    .iter()
                    .zip(&ov)
                    .map(|(a, b)| (a - b).abs() / a.abs().max(1e-2))
                    .fold(0.0f32, f32::max);
                println!("      isolate: v2gu+v2dn on hybrid CSR maxrel-vs-f8s {mr2:.2e}");
                // fq/fs spot-compare on the first live bm32 row
                let hr: Vec<u32> = exec.stream.clone_dtoh(&hsrow).expect("d");
                let fq_h: Vec<i8> = exec.stream.clone_dtoh(&hfq).expect("d");
                let fs_h: Vec<f32> = exec.stream.clone_dtoh(&hfs).expect("d");
                let fq_v: Vec<i8> = exec.stream.clone_dtoh(&hfq2).expect("d");
                let fs_v: Vec<f32> = exec.stream.clone_dtoh(&hfs2).expect("d");
                if let Some(li) = hr.iter().position(|&t| t != u32::MAX) {
                    println!(
                        "      row {li}: remap fs[0]={:.4e} v2 fs[0]={:.4e} | remap fq[0..6]={:?} v2 fq[0..6]={:?}",
                        fs_h[li * ff / 32],
                        fs_v[li * ff / 32],
                        &fq_h[li * ff..li * ff + 6],
                        &fq_v[li * ff..li * ff + 6]
                    );
                }
            }

            let f8b = 33.0 / 32.0;
            let gu_lb = (uniq_e * 2 * ff * embd) as f64 / 1e6 * f8b;
            let dn_lb = (uniq_e * embd * ffp) as f64 / 1e6 * f8b;
            println!(
                "      f8s gu {gu_s:8.1}us dn {dn_s:8.1}us | f8d gu {gu_d:8.1}us ({:6.0} GB/s uniq) \
                 dn {dn_d:8.1}us ({:6.0}) geglu {ge_d:6.1}us align+gather {ag_d:6.1}us chain {chain:8.1}us",
                gu_lb / gu_d * 1e3,
                dn_lb / dn_d * 1e3,
            );
            println!("      f8d==f8s BITWISE OK; e4m3 class vs dec2 maxrel {maxrel:.2e}");
        }
    }
}
