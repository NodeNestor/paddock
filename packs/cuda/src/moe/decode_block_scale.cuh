// moe/decode_block_scale.cuh (formerly 13_moe_dec_bs.cuh) - decode-class fp4 MoE pair (refuted, opt-in) + block-scale MoE gate_up/down
// Textually-included segment of the single pack translation unit.
// Not standalone-compilable: include order is defined by ../pack.cu.
// ---- DECODE-class fp4 MoE pair.
// At decode (~1-4 real rows per 32-slot align block)
// the warp-specialized _bs tiles are memory-LATENCY-bound at 1 CTA/SM
// (DRAM 19.8%, SM 7.8%, occupancy 21.9%) and spend their machinery on 90%+
// PAD rows. These kernels take the opposite posture: shallow and WIDE -
// plain streamed weight reads, real rows only, ~17 KB smem (4-5 CTAs/SM),
// grid (blocks, out/32) = thousands of self-staggering CTAs whose combined
// pending loads keep DRAM fed. Same align/activation/output contracts as
// the _bs pair (they drop in inside the same launchers). NUMERIC CLASS
// differs (f32 FMA over exactly-dequantized operands vs block-scale MMA) -
// env-gated PADDOCK_MOE_DEC, PPL-gated before any default-on.
#define PD_DEC_TT 4u

// e2m1 nibble -> float, branch-light ALU (a table would serialize on the
// divergent per-lane indices): s|ee|m, mag = ee==0 ? 0.5*m : (1+m/2)*2^(ee-1)
static __device__ __forceinline__ float pd_e2m1_val(uint32_t nib) {
    const uint32_t m = nib & 1u, ee = (nib >> 1) & 3u;
    const uint32_t m2 = ee == 0u ? m : ((2u + m) << (ee - 1u));
    const float v = 0.5f * (float)m2;
    return (nib & 8u) ? -v : v;
}

__global__ void __launch_bounds__(256)
pd_mxfp4_moe_gate_up_dec_kernel(
    const unsigned char* __restrict__ gate_data, const unsigned char* __restrict__ gate_scale,
    const float* __restrict__ gate_bias,
    const unsigned char* __restrict__ up_data, const unsigned char* __restrict__ up_scale,
    const float* __restrict__ up_bias,
    const unsigned int* __restrict__ sorted_row, const unsigned int* __restrict__ block_expert,
    const unsigned char* __restrict__ yq, const unsigned char* __restrict__ ys,
    unsigned char* __restrict__ fq, unsigned char* __restrict__ fs,
    uint32_t in_dim, uint32_t ff, float alpha, float limit, float up_add) {
    const uint32_t blk = blockIdx.x;
    const uint32_t e = block_expert[blk];
    if (e == PD_MOE_PAD) return;
    const uint32_t r0 = blockIdx.y * 32u;  // this block's 32 out rows = one fs block
    const uint32_t tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
    const uint32_t nkb = in_dim >> 5;
    const uint32_t n_sb = ff >> 5;
    extern __shared__ unsigned char dsh[];
    __half* xh = reinterpret_cast<__half*>(dsh);  // [TT][in_dim], scale folded in
    float* fusedv = reinterpret_cast<float*>(xh + PD_DEC_TT * in_dim);  // [TT][32]
    __shared__ unsigned int tokv[PD_DEC_TT];
    __shared__ unsigned int nt_sh;

    for (uint32_t s0 = 0; s0 < 32u; s0 += PD_DEC_TT) {
        // align fills a block's slots contiguously, so the first PAD ends it
        if (tid < PD_DEC_TT) tokv[tid] = sorted_row[(size_t)blk * 32u + s0 + tid];
        if (tid == 0) nt_sh = 0;
        __syncthreads();
        if (tid == 0) {
            uint32_t n = 0;
            while (n < PD_DEC_TT && tokv[n] != PD_MOE_PAD) ++n;
            nt_sh = n;
        }
        __syncthreads();
        const uint32_t nt = nt_sh;
        if (nt == 0) return;
        // stage activations dequantized to half (e4m3 value x ue8m0 pow2 is
        // exact in fp16 - no extra rounding class)
        for (uint32_t u = tid; u < nt * (in_dim >> 2); u += 256u) {
            const uint32_t t = u / (in_dim >> 2), k = (u % (in_dim >> 2)) * 4u;
            const size_t rowb = (size_t)tokv[t] * in_dim;
            const float sc = ldexpf(1.0f, (int)ys[(rowb + k) >> 5] - 127);
            uint32_t q4;
            memcpy(&q4, yq + rowb + k, 4);
#pragma unroll
            for (uint32_t j = 0; j < 4u; ++j) {
                __nv_fp8_e4m3 f8;
                f8.__x = (unsigned char)(q4 >> (8u * j));
                xh[t * in_dim + k + j] = __float2half(float(f8) * sc);
            }
        }
        __syncthreads();

        // 8 warps x 4 sequential local rows cover the 32 out rows; a warp's
        // lanes stride the k32 blocks (each lane owns kb = lane, lane+32, ..)
        for (uint32_t rr = 0; rr < 4u; ++rr) {
            const uint32_t rloc = warp * 4u + rr;
            const uint32_t r = r0 + rloc;
            float accg[PD_DEC_TT] = {0.f, 0.f, 0.f, 0.f};
            float accu[PD_DEC_TT] = {0.f, 0.f, 0.f, 0.f};
            if (r < ff) {
                // gate_data is the INTERLEAVED gate+up plane ([gate 64 B |
                // up 64 B] per 4-kb group per row - gu_interleave); up_data
                // is a dummy the bs family never derefs. Scales stay flat
                // per-plane.
                const size_t wsb = ((size_t)e * ff + r) * (size_t)nkb;
                const size_t rowbytes =
                    ((size_t)e * ff + r) * (size_t)(((nkb + 3u) >> 2) * 128u);
                for (uint32_t kb = lane; kb < nkb; kb += 32u) {
                    const size_t goff =
                        rowbytes + (size_t)(kb >> 2) * 128u + (size_t)(kb & 3u) * 16u;
                    uint4 gq = *reinterpret_cast<const uint4*>(gate_data + goff);
                    uint4 uq = *reinterpret_cast<const uint4*>(gate_data + goff + 64u);
                    const float gsc = ldexpf(1.0f, (int)gate_scale[wsb + kb] - 127);
                    const float usc = ldexpf(1.0f, (int)up_scale[wsb + kb] - 127);
                    const unsigned char* gb = reinterpret_cast<const unsigned char*>(&gq);
                    const unsigned char* ub = reinterpret_cast<const unsigned char*>(&uq);
                    float pg[PD_DEC_TT] = {0.f, 0.f, 0.f, 0.f};
                    float pu[PD_DEC_TT] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
                    for (uint32_t d = 0; d < 16u; ++d) {
                        const float glo = pd_e2m1_val(gb[d] & 15u);
                        const float ghi = pd_e2m1_val(gb[d] >> 4);
                        const float ulo = pd_e2m1_val(ub[d] & 15u);
                        const float uhi = pd_e2m1_val(ub[d] >> 4);
#pragma unroll
                        for (uint32_t t = 0; t < PD_DEC_TT; ++t) {
                            if (t >= nt) break;
                            const __half* xr = xh + t * in_dim + kb * 32u;
                            const float xlo = __half2float(xr[d]);
                            const float xhi = __half2float(xr[d + 16u]);
                            pg[t] = fmaf(glo, xlo, fmaf(ghi, xhi, pg[t]));
                            pu[t] = fmaf(ulo, xlo, fmaf(uhi, xhi, pu[t]));
                        }
                    }
#pragma unroll
                    for (uint32_t t = 0; t < PD_DEC_TT; ++t) {
                        if (t >= nt) break;
                        accg[t] = fmaf(pg[t], gsc, accg[t]);
                        accu[t] = fmaf(pu[t], usc, accu[t]);
                    }
                }
            }
#pragma unroll
            for (uint32_t t = 0; t < PD_DEC_TT; ++t) {
                if (t >= nt) break;
                float g = accg[t], u2 = accu[t];
                for (uint32_t s = 16; s > 0; s >>= 1) {
                    g += __shfl_down_sync(0xffffffffu, g, s);
                    u2 += __shfl_down_sync(0xffffffffu, u2, s);
                }
                if (lane == 0 && r < ff) {
                    // Exact _bs epilogue math (alpha/limit/up_add semantics)
                    const float gv = g + gate_bias[r];
                    const float uv = u2 + up_bias[r];
                    const float xg = fminf(gv, limit);
                    const float yu = fminf(fmaxf(uv, -limit), limit);
                    fusedv[t * 32u + rloc] =
                        (xg / (1.0f + expf(-alpha * xg))) * (yu + up_add);
                }
            }
        }
        __syncthreads();
        // e4m3 requant, one warp per real token; the frexpf shared-exponent
        // pick is copied verbatim from the _bs epilogue (identical fs class)
        if (warp < nt) {
            const float v = fusedv[warp * 32u + lane];
            float a = fabsf(v);
            for (uint32_t o = 16; o > 0; o >>= 1)
                a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
            int ee = 0;
            if (a > 0.0f) {
                int ex;
                float m = frexpf(a, &ex);
                ee = ex - 9 + (m > 0.875f ? 1 : 0);
            }
            const float inv = ldexpf(1.0f, -ee);
            const size_t row = (size_t)blk * 32u + s0 + warp;
            fq[row * ff + r0 + lane] = __nv_fp8_e4m3(v * inv).__x;
            if (lane == 0) fs[row * n_sb + (r0 >> 5)] = (unsigned char)(ee + 127);
        }
        __syncthreads();
        if (nt < PD_DEC_TT) return;
    }
}

__global__ void __launch_bounds__(256)
pd_mxfp4_moe_down_dec_kernel(
    const unsigned char* __restrict__ down_data, const unsigned char* __restrict__ down_scale,
    const float* __restrict__ down_bias,
    const unsigned int* __restrict__ sorted_row, const unsigned int* __restrict__ sorted_slot,
    const unsigned int* __restrict__ block_expert, const float* __restrict__ topk_w,
    const unsigned char* __restrict__ fq, const unsigned char* __restrict__ fs,
    float* __restrict__ part, uint32_t ff, uint32_t embd, uint32_t n_active) {
    const uint32_t blk = blockIdx.x;
    const uint32_t e = block_expert[blk];
    if (e == PD_MOE_PAD) return;
    const uint32_t r0 = blockIdx.y * 32u;  // 32 embd out rows
    const uint32_t tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
    const uint32_t nkb = ff >> 5;  // 16 at ff=512
    extern __shared__ unsigned char dsh[];
    __half* xh = reinterpret_cast<__half*>(dsh);  // [TT][ff]
    __shared__ unsigned int tokv[PD_DEC_TT], sltv[PD_DEC_TT];
    __shared__ unsigned int nt_sh;

    for (uint32_t s0 = 0; s0 < 32u; s0 += PD_DEC_TT) {
        if (tid < PD_DEC_TT) {
            tokv[tid] = sorted_row[(size_t)blk * 32u + s0 + tid];
            sltv[tid] = sorted_slot[(size_t)blk * 32u + s0 + tid];
        }
        if (tid == 0) nt_sh = 0;
        __syncthreads();
        if (tid == 0) {
            uint32_t n = 0;
            while (n < PD_DEC_TT && tokv[n] != PD_MOE_PAD) ++n;
            nt_sh = n;
        }
        __syncthreads();
        const uint32_t nt = nt_sh;
        if (nt == 0) return;
        // stage the fused activations (SORTED-major fq/fs, gate_up's output)
        for (uint32_t u = tid; u < nt * (ff >> 2); u += 256u) {
            const uint32_t t = u / (ff >> 2), k = (u % (ff >> 2)) * 4u;
            const size_t rowb = ((size_t)blk * 32u + s0 + t) * ff;
            const float sc =
                ldexpf(1.0f, (int)fs[((size_t)blk * 32u + s0 + t) * (ff >> 5) + (k >> 5)] - 127);
            uint32_t q4;
            memcpy(&q4, fq + rowb + k, 4);
#pragma unroll
            for (uint32_t j = 0; j < 4u; ++j) {
                __nv_fp8_e4m3 f8;
                f8.__x = (unsigned char)(q4 >> (8u * j));
                xh[t * ff + k + j] = __float2half(float(f8) * sc);
            }
        }
        __syncthreads();

        for (uint32_t rr = 0; rr < 4u; ++rr) {
            const uint32_t rloc = warp * 4u + rr;
            const uint32_t r = r0 + rloc;
            float acc[PD_DEC_TT] = {0.f, 0.f, 0.f, 0.f};
            if (r < embd && lane < nkb) {
                const size_t wb = ((size_t)e * embd + r) * (size_t)nkb;
                for (uint32_t kb = lane; kb < nkb; kb += 32u) {
                    uint4 dq = *reinterpret_cast<const uint4*>(down_data + (wb + kb) * 16u);
                    const float dsc = ldexpf(1.0f, (int)down_scale[wb + kb] - 127);
                    const unsigned char* db = reinterpret_cast<const unsigned char*>(&dq);
                    float pd[PD_DEC_TT] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
                    for (uint32_t d = 0; d < 16u; ++d) {
                        const float wlo = pd_e2m1_val(db[d] & 15u);
                        const float whi = pd_e2m1_val(db[d] >> 4);
#pragma unroll
                        for (uint32_t t = 0; t < PD_DEC_TT; ++t) {
                            if (t >= nt) break;
                            const __half* xr = xh + t * ff + kb * 32u;
                            pd[t] = fmaf(wlo, __half2float(xr[d]),
                                         fmaf(whi, __half2float(xr[d + 16u]), pd[t]));
                        }
                    }
#pragma unroll
                    for (uint32_t t = 0; t < PD_DEC_TT; ++t) {
                        if (t >= nt) break;
                        acc[t] = fmaf(pd[t], dsc, acc[t]);
                    }
                }
            }
#pragma unroll
            for (uint32_t t = 0; t < PD_DEC_TT; ++t) {
                if (t >= nt) break;
                float v = acc[t];
                for (uint32_t s = 16; s > 0; s >>= 1)
                    v += __shfl_down_sync(0xffffffffu, v, s);
                if (lane == 0 && r < embd) {
                    const uint32_t token = tokv[t];
                    const float w = topk_w[(size_t)token * n_active + sltv[t]];
                    part[((size_t)token * n_active + sltv[t]) * embd + r] =
                        w * (v + down_bias[r]);
                }
            }
        }
        __syncthreads();
        if (nt < PD_DEC_TT) return;
    }
}

PD_EXPORT
int pd_mxfp4_moe_gate_up_bs(const void* gate_data, const void* gate_scale,
                            const void* gate_bias, const void* up_data,
                            const void* up_scale, const void* up_bias,
                            const void* sorted_row, const void* block_expert,
                            const void* yq, const void* ys, void* fq, void* fs,
                            uint32_t in_dim, uint32_t ff, uint32_t max_blocks,
                            uint32_t rows, float alpha, float limit, float up_add,
                            void* stream) {
#ifndef PD_BS_HOST
    (void)gate_data; (void)gate_scale; (void)gate_bias; (void)up_data; (void)up_scale;
    (void)up_bias; (void)sorted_row; (void)block_expert; (void)yq; (void)ys; (void)fq;
    (void)fs; (void)in_dim; (void)ff; (void)max_blocks; (void)rows; (void)alpha;
    (void)limit; (void)up_add; (void)stream;
    return cudaErrorNotSupported;
#else
    if (max_blocks == 0 || ff == 0) return 0;
    if ((in_dim & 31u) != 0 || (ff & 31u) != 0) return cudaErrorInvalidValue;
    // decode-class kernel (PADDOCK_MOE_DEC, see the pair above): real rows
    // only, wide shallow grid; smem 4x in_dim halves + the fused pane
    static const bool dec_env = pd_env("PADDOCK_MOE_DEC") != nullptr;
    if (dec_env && rows < 256u) {
        const size_t sh = (size_t)PD_DEC_TT * in_dim * 2u + (size_t)PD_DEC_TT * 32u * 4u;
        dim3 g(max_blocks, ff >> 5);
        pd_mxfp4_moe_gate_up_dec_kernel<<<g, 256, sh, (cudaStream_t)stream>>>(
            (const unsigned char*)gate_data, (const unsigned char*)gate_scale,
            (const float*)gate_bias, (const unsigned char*)up_data,
            (const unsigned char*)up_scale, (const float*)up_bias,
            (const unsigned int*)sorted_row, (const unsigned int*)block_expert,
            (const unsigned char*)yq, (const unsigned char*)ys, (unsigned char*)fq,
            (unsigned char*)fs, in_dim, ff, alpha, limit, up_add);
        return pd_launch_status();
    }
    static int nsm_gu = 0;
    static bool smem_gu = false, no_persist = false;
    if (!smem_gu) {
        // the 256-deep K chunk puts the tile past the 48 KB default window
        cudaFuncSetAttribute((const void*)pd_mxfp4_moe_gate_up_bs_kernel<PD_BS_BM, 128u, 2u>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, PD_BS_GU_SMEM);
        cudaFuncSetAttribute(
            (const void*)pd_mxfp4_moe_gate_up_bs_kernel<PD_BS_BM, 128u, 2u, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, PD_BS_GU_SMEM);
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&nsm_gu, cudaDevAttrMultiProcessorCount, dev);
        if (nsm_gu <= 0) nsm_gu = 128;
        no_persist = pd_env("PADDOCK_NO_MOE_PERSIST") != nullptr;
        smem_gu = true;
    }
    const uint32_t ny = (ff + 127u) / 128u;
    // persistent work loop: PREFILL-class launches only (rows >= 256 - the
    // harness A/B: +11% at the pp512 shape, -4.5% at the 1-row decode shape,
    // where the grid-stride items phase-lock every CTA's transition into a
    // global DRAM dip; decode keeps the self-staggering 2-D grid). Also
    // needs nk > PD_BS_S + 1 so the item-parity metadata slots can never be
    // overwritten under a consumer (see kernel note).
    const uint32_t nk = (in_dim + PD_BS_KC_GU - 1u) / PD_BS_KC_GU;
    if (!no_persist && rows >= 256u && nk > PD_BS_S + 1u) {
        const uint32_t nit = max_blocks * ny;
        const uint32_t g1 = nit < (uint32_t)nsm_gu ? nit : (uint32_t)nsm_gu;
        // The NCW=8 consumer config (RPW=32/NF=2) is the shipped geometry.
        // The wide NCW=16 config (RPW=16/NF=1, 608 threads) was built to
        // double the MMA drain rate at the pp512 roofline corner (s17b spec);
        // it is BIT-EXACT but measured slower - pp512 321us vs 293 in-harness,
        // and at the server c32 1118 (vs 1127) + pf8 285 (vs 290). The extra
        // 8 consumer warps contend the same stage smem faster than they drain
        // MMA, and the two bar.sync-3 half-epilogue barriers/item on 512
        // consumer threads add latency the narrow config never pays.
        // FALSIFIED - kept as a compilable template lever, not launched.
        pd_mxfp4_moe_gate_up_bs_kernel<PD_BS_BM, 128u, 2u, true><<<g1,
                                         PD_BS_TH_GU,
                                         PD_BS_GU_SMEM, (cudaStream_t)stream>>>(
            (const unsigned char*)gate_data, (const unsigned char*)gate_scale,
            (const float*)gate_bias, (const unsigned char*)up_data,
            (const unsigned char*)up_scale, (const float*)up_bias,
            (const unsigned int*)sorted_row, (const unsigned int*)block_expert,
            (const unsigned char*)yq, (const unsigned char*)ys, (unsigned char*)fq,
            (unsigned char*)fs, in_dim, ff, alpha, limit, up_add, max_blocks);
        return pd_launch_status();
    }
    dim3 grid(max_blocks, ny);
    pd_mxfp4_moe_gate_up_bs_kernel<PD_BS_BM, 128u, 2u><<<grid, PD_BS_TH_GU, PD_BS_GU_SMEM,
                                     (cudaStream_t)stream>>>(
        (const unsigned char*)gate_data, (const unsigned char*)gate_scale,
        (const float*)gate_bias, (const unsigned char*)up_data,
        (const unsigned char*)up_scale, (const float*)up_bias,
        (const unsigned int*)sorted_row, (const unsigned int*)block_expert,
        (const unsigned char*)yq, (const unsigned char*)ys, (unsigned char*)fq,
        (unsigned char*)fs, in_dim, ff, alpha, limit, up_add, max_blocks);
    return pd_launch_status();
#endif
}

// Prefill-config gate_up: 64-token blocks on 64-row weight tiles. Per-launch
// weight traffic is unchanged (it scales with block count x KC, not tile
// height) but fat experts need half the blocks, so a 2048-pair tick reads
// the touched experts' weights ~half as often. Same KC=256 read granularity,
// same per-warp fragment shape (register-neutral). Pairs with
// pd_mxfp4_moe_down_bs64 - fq/fs are indexed by 64-row sorted blocks.
#define PD_BS64_GU_STAGE (2u * 64u * PD_BS_WROW_GU + 64u * PD_BS_YROW_GU                           + 2u * 64u * PD_BS_KB_GU + 64u * PD_BS_KB_GU)
#define PD_BS64_GU_SMEM (32u + 4u * 64u * 4u + PD_BS_S * PD_BS64_GU_STAGE)
PD_EXPORT
int pd_mxfp4_moe_gate_up_bs64(const void* gate_data, const void* gate_scale,
                              const void* gate_bias, const void* up_data,
                              const void* up_scale, const void* up_bias,
                              const void* sorted_row, const void* block_expert,
                              const void* yq, const void* ys, void* fq, void* fs,
                              uint32_t in_dim, uint32_t ff, uint32_t max_blocks,
                              uint32_t rows, float alpha, float limit, float up_add,
                              void* stream) {
    (void)rows;
#ifndef PD_BS_HOST
    (void)gate_data; (void)gate_scale; (void)gate_bias; (void)up_data; (void)up_scale;
    (void)up_bias; (void)sorted_row; (void)block_expert; (void)yq; (void)ys; (void)fq;
    (void)fs; (void)in_dim; (void)ff; (void)max_blocks; (void)alpha; (void)limit;
    (void)up_add; (void)stream;
    return cudaErrorNotSupported;
#else
    if (max_blocks == 0 || ff == 0) return 0;
    if ((in_dim & 31u) != 0 || (ff & 31u) != 0) return cudaErrorInvalidValue;
    static int nsm_gu64 = 0;
    static bool smem_gu64 = false, no_persist64 = false;
    if (!smem_gu64) {
        cudaFuncSetAttribute((const void*)pd_mxfp4_moe_gate_up_bs_kernel<64u, 64u, 4u>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, PD_BS64_GU_SMEM);
        cudaFuncSetAttribute(
            (const void*)pd_mxfp4_moe_gate_up_bs_kernel<64u, 64u, 4u, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, PD_BS64_GU_SMEM);
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&nsm_gu64, cudaDevAttrMultiProcessorCount, dev);
        if (nsm_gu64 <= 0) nsm_gu64 = 128;
        no_persist64 = pd_env("PADDOCK_NO_MOE_PERSIST") != nullptr;
        smem_gu64 = true;
    }
    // persistent work loop, same gating as the BM=32 launcher above - without
    // it the bs64 config pays the 1-CTA/SM epilogue DRAM-silence the persistent
    // redesign removed (first serving A/B of plain-2D bs64: c8 -2.1%, c32 -2.2%
    // vs persistent bs32 - the +11% persistence win outweighed the halved
    // weight re-reads).
    const uint32_t nk64 = (in_dim + PD_BS_KC_GU - 1u) / PD_BS_KC_GU;
    if (!no_persist64 && rows >= 256u && nk64 > PD_BS_S + 1u) {
        const uint32_t nit = max_blocks * ((ff + 63u) / 64u);
        const uint32_t g1 = nit < (uint32_t)nsm_gu64 ? nit : (uint32_t)nsm_gu64;
        pd_mxfp4_moe_gate_up_bs_kernel<64u, 64u, 4u, true><<<g1, PD_BS_TH_GU,
                                         PD_BS64_GU_SMEM, (cudaStream_t)stream>>>(
            (const unsigned char*)gate_data, (const unsigned char*)gate_scale,
            (const float*)gate_bias, (const unsigned char*)up_data,
            (const unsigned char*)up_scale, (const float*)up_bias,
            (const unsigned int*)sorted_row, (const unsigned int*)block_expert,
            (const unsigned char*)yq, (const unsigned char*)ys, (unsigned char*)fq,
            (unsigned char*)fs, in_dim, ff, alpha, limit, up_add, max_blocks);
        return pd_launch_status();
    }
    dim3 grid(max_blocks, (ff + 63u) / 64u);
    pd_mxfp4_moe_gate_up_bs_kernel<64u, 64u, 4u><<<grid, PD_BS_TH_GU, PD_BS64_GU_SMEM,
                                     (cudaStream_t)stream>>>(
        (const unsigned char*)gate_data, (const unsigned char*)gate_scale,
        (const float*)gate_bias, (const unsigned char*)up_data,
        (const unsigned char*)up_scale, (const float*)up_bias,
        (const unsigned int*)sorted_row, (const unsigned int*)block_expert,
        (const unsigned char*)yq, (const unsigned char*)ys, (unsigned char*)fq,
        (unsigned char*)fs, in_dim, ff, alpha, limit, up_add, max_blocks);
    return pd_launch_status();
#endif
}

// Block-scale down over the sorted layout: B = the gate_up_bs e4m3 output
// (sorted-row indexed, no gather), A = down expert strips. WARP-SPECIALIZED
// like gate_up above (8 consumer warps keep the original MMA + partials
// epilogue verbatim - memcmp bit-exact - and
// PD_BS_PW_DN producer warps own the staging behind per-stage mbarriers).
// Emits the same per-(token, slot) f32 partials as down_mmq -
// pd_moe_slot_combine folds them in fixed order, keeping the MoE
// bit-reproducible run to run.
template <uint32_t BM, uint32_t BMR, uint32_t CW, bool PERSIST = false,
          bool PBF16 = false>
__global__ void __launch_bounds__(PD_BS_TH_DN, PD_BS_MINCTA(PD_BS_DN_SMEM))
pd_mxfp4_moe_down_bs_kernel(
    const unsigned char* __restrict__ down_data, const unsigned char* __restrict__ down_scale,
    const float* __restrict__ down_bias,
    const unsigned int* __restrict__ sorted_row, const unsigned int* __restrict__ sorted_slot,
    const unsigned int* __restrict__ block_expert, const float* __restrict__ topk_w,
    const unsigned char* __restrict__ fq, const unsigned char* __restrict__ fs,
    float* __restrict__ part, float* __restrict__ residual,
    unsigned int* __restrict__ cnt, uint32_t ff, uint32_t embd, uint32_t n_active,
    uint32_t nb) {
#if PD_BS_OK
    // same warp-grid scheme as gate_up: 8 consumer warps x (32 rows x 16
    // cols) fragments; <32,128,2> = decode (bit-identical to pre-template),
    // <64,64,4> = prefill (half the blocks at fat experts, same KC).
    static_assert(CW * (8u / CW) == 8u && 32u * (8u / CW) == BMR && 16u * CW == BM,
                  "warp grid must cover the tile");
    static_assert((BMR & (BMR - 1u)) == 0u, "scale-prefetch mask needs pow2 BMR");
    constexpr uint32_t DN_STAGE = BMR * PD_BS_WROW_DN + BM * PD_BS_YROW_DN +
                                  BMR * PD_BS_KB_DN + BM * PD_BS_KB_DN;
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u, warp = tid >> 5;
    const uint32_t n_kb = ff >> 5;
    const uint32_t nk = (ff + PD_BS_KC_DN - 1u) / PD_BS_KC_DN;
    // persistent work loop - see gate_up_bs for the full design note. Down's
    // producers never read tok/slt (fq/fs index by blk), so metadata needs
    // no producer barrier here; parity slots + the nk > PD_BS_S pipeline
    // depth keep the consumer's epilogue reads safe.
    const uint32_t ny = (embd + BMR - 1u) / BMR;
    const uint32_t nit = nb * ny;
    const uint32_t it0 = PERSIST ? blockIdx.x : blockIdx.y * nb + blockIdx.x;
    const uint32_t itstep = PERSIST ? gridDim.x : nit;

    extern __shared__ unsigned char pd_bs_sh[];
    uint64_t* bfull = (uint64_t*)pd_bs_sh;   // [PD_BS_S]
    uint64_t* bempty = bfull + PD_BS_S;      // [PD_BS_S]
    // four item-parity metadata slots (gate_up keeps 2): down's K is only
    // ff=512 -> nk=2 at KC=256, so the 2-slot guard (nk > S+1) could never
    // let the persistent path engage and every prefill down launch paid the
    // 2-D grid's per-item epilogue silence (626us vs gate_up's 580 at
    // half the FLOPs, 17.5% tensor). With 4 slots a clobber needs the
    // producer 4 items ahead, but the S-stage ring caps runahead at
    // ceil((S+1)/nk)=2 items -- safe at any nk >= S, so the launcher gate
    // relaxes to nk >= PD_BS_S.
    float* bds = (float*)(pd_bs_sh + 32u);   // [4][128] down bias (item parity)
    unsigned char* tiles = pd_bs_sh + 32u + 4u * BMR * 4u;   // 16-aligned stages
    __shared__ unsigned int tok[4][BM];
    __shared__ unsigned int slt[4][BM];
    if (tid == 0) {
        #pragma unroll
        for (uint32_t s = 0; s < PD_BS_S; ++s) {
            pd_bs_bar_init(&bfull[s], 2u * 32u * PD_BS_PW_DN);
            pd_bs_bar_init(&bempty[s], 256u);
        }
    }
    __syncthreads();  // the only full-CTA barrier in the kernel

    if (warp >= 8u) {
        // ------------- producers: PD_BS_PW_DN warps own all staging -------------
        const uint32_t ptid = tid - 256u;
        const uint32_t pth = 32u * PD_BS_PW_DN;
        uint32_t eph[PD_BS_S] = {};
        uint32_t gkt = 0, ipar = 0;  // global chunk counter + item parity
        #define PD_BS_SCD_N (BMR * PD_BS_KB_DN + BM * PD_BS_KB_DN)
        #define PD_BS_SCD_V ((PD_BS_SCD_N + 32u * PD_BS_PW_DN - 1u) / (32u * PD_BS_PW_DN))
        #define PD_BS_LDG_SCD(regs, kt)                                                   \
            _Pragma("unroll") for (uint32_t v = 0; v < PD_BS_SCD_V; ++v) {                \
                const uint32_t u = ptid + v * pth;                                        \
                const uint32_t w = BMR * PD_BS_KB_DN;                                    \
                unsigned char b = 0u;                                                     \
                if (u < w) {                                                              \
                    const uint32_t row = u / PD_BS_KB_DN, kb = u % PD_BS_KB_DN;           \
                    if ((row_base + row) < embd && (kt) * PD_BS_KB_DN + kb < n_kb)        \
                        b = down_scale[(wrow0 + row) * n_kb + (kt) * PD_BS_KB_DN + kb];   \
                } else if (u < PD_BS_SCD_N) {                                             \
                    const uint32_t t = (u - w) / PD_BS_KB_DN, kb = u % PD_BS_KB_DN;       \
                    if ((kt) * PD_BS_KB_DN + kb < n_kb)                                   \
                        b = fs[((size_t)blk * BM + t) * n_kb +                      \
                               (kt) * PD_BS_KB_DN + kb];                                  \
                }                                                                         \
                (regs)[v] = b;                                                            \
            }
        for (uint32_t it = it0; it < nit; it += itstep) {
            const uint32_t blk = it % nb;
            const uint32_t e = block_expert[blk];
            if (e == PD_MOE_PAD) continue;
            const uint32_t row_base = (it / nb) * BMR;
            const size_t wrow0 = (size_t)e * embd + row_base;
            const uint32_t p = ipar & 3u;
            ++ipar;
            // consumer-only metadata into parity slot p (visible via the
            // bfull release/acquire chain - stores precede this thread's
            // first arrive of the item; the epilogue runs after all waits)
            for (uint32_t u = ptid; u < BM; u += pth) {
                tok[p][u] = sorted_row[(size_t)blk * BM + u];
                slt[p][u] = sorted_slot[(size_t)blk * BM + u];
            }
            for (uint32_t u = ptid; u < BMR; u += pth) {
                const bool ok = row_base + u < embd;
                bds[p * BMR + u] = ok ? down_bias[(size_t)e * embd + row_base + u] : 0.0f;
            }
            unsigned char screg[PD_BS_SCD_V];
            PD_BS_LDG_SCD(screg, 0u)
            for (uint32_t kt = 0; kt < nk; ++kt) {
                const uint32_t s = gkt % PD_BS_S;
                unsigned char* wds = tiles + s * DN_STAGE;
                unsigned char* ybs = wds + BMR * PD_BS_WROW_DN;
                unsigned char* wsd = ybs + BM * PD_BS_YROW_DN;
                unsigned char* ysc = wsd + BMR * PD_BS_KB_DN;
                if (gkt >= PD_BS_S) { pd_bs_bar_wait(&bempty[s], eph[s]); eph[s] ^= 1u; }
                ++gkt;
                for (uint32_t u = ptid; u < BMR * PD_BS_WSEG_DN; u += pth) {
                    const uint32_t row = u / PD_BS_WSEG_DN, seg = u % PD_BS_WSEG_DN;
                    const bool ok = (row_base + row) < embd && kt * PD_BS_KB_DN + seg < n_kb;
                    pd_cp_async16((int*)(wds + row * PD_BS_WROW_DN +
                                         (seg ^ PD_BS_SWZ(row, PD_BS_WSEG_DN)) * 16u),
                                  down_data + (wrow0 + row) * (ff >> 1) +
                                      kt * PD_BS_WROW_DN + seg * 16u, ok);
                }
                for (uint32_t u = ptid; u < BM * PD_BS_YSEG_DN; u += pth) {
                    const uint32_t t = u / PD_BS_YSEG_DN, seg = u % PD_BS_YSEG_DN;
                    const bool ok = (kt * PD_BS_YSEG_DN + seg) * 16u < ff;
                    pd_cp_async16((int*)(ybs + t * PD_BS_YROW_DN + seg * 16u),
                                  fq + ((size_t)blk * BM + t) * ff + kt * PD_BS_KC_DN +
                                      seg * 16u, ok);
                }
                #pragma unroll
                for (uint32_t v = 0; v < PD_BS_SCD_V; ++v) {
                    const uint32_t u = ptid + v * pth;
                    const uint32_t w = BMR * PD_BS_KB_DN;
                    if (u < w) wsd[u] = screg[v];
                    else if (u < PD_BS_SCD_N) ysc[u - w] = screg[v];
                }
                pd_bs_cp_arrive_noinc(&bfull[s]);
                pd_bs_bar_arrive(&bfull[s]);
                if (kt + 1u < nk) PD_BS_LDG_SCD(screg, kt + 1u)
            }
        }
        #undef PD_BS_LDG_SCD
        #undef PD_BS_SCD_V
        #undef PD_BS_SCD_N
        return;
    }

    // ------------- consumers: the original MMA + epilogue, verbatim -------------
    const uint32_t g = lane >> 2, tq = lane & 3u;
    const uint32_t i0 = (warp / CW) * 32u;
    const uint32_t joff = (warp % CW) * 16u;
    uint32_t fph[PD_BS_S] = {};
    uint32_t gct = 0, ipar = 0;
    for (uint32_t it = it0; it < nit; it += itstep) {
    const uint32_t blk = it % nb;
    if (block_expert[blk] == PD_MOE_PAD) continue;
    const uint32_t row_base = (it / nb) * BMR;
    const uint32_t p = ipar & 3u;
    ++ipar;
    float acc[2][2][4] = {};
    for (uint32_t kt = 0; kt < nk; ++kt) {
        const uint32_t s = gct % PD_BS_S;
        ++gct;
        unsigned char* wd = tiles + s * DN_STAGE;
        unsigned char* yb = wd + BMR * PD_BS_WROW_DN;
        unsigned char* wsd = yb + BM * PD_BS_YROW_DN;
        unsigned char* ysc = wsd + BMR * PD_BS_KB_DN;
        pd_bs_bar_wait(&bfull[s], fph[s]); fph[s] ^= 1u;
        #pragma unroll
        for (uint32_t kb = 0; kb < PD_BS_KB_DN; ++kb) {
            uint32_t b0[2], b1[2], sfb[2];
            #pragma unroll
            for (uint32_t j = 0; j < 2u; ++j) {
                uint32_t t = joff + j * 8u + g;
                const unsigned char* yr = yb + t * PD_BS_YROW_DN + kb * 32u;
                b0[j] = *(const uint32_t*)(yr + 4u * tq);
                b1[j] = *(const uint32_t*)(yr + 16u + 4u * tq);
                sfb[j] = ysc[t * PD_BS_KB_DN + kb];
            }
            #pragma unroll
            for (uint32_t n = 0; n < 2u; ++n) {
                uint32_t r0 = i0 + n * 16u + g;
                uint32_t da[4];
                // afrag_split: split-order repacked bytes (see gate_up_bs);
                // kb ^ swz undoes the store-side seg swizzle
                uint32_t kbs = kb ^ PD_BS_SWZ(r0, PD_BS_WSEG_DN);
                pd_bs_afrag_split(da, wd + r0 * PD_BS_WROW_DN + kbs * 16u,
                                  wd + (r0 + 8u) * PD_BS_WROW_DN + kbs * 16u, tq);
                uint32_t rs = (tq & 1u) ? r0 + 8u : r0;
                uint32_t sfad = wsd[rs * PD_BS_KB_DN + kb];
                #pragma unroll
                for (uint32_t j = 0; j < 2u; ++j)
                    pd_bs_mma(acc[n][j], da[0], da[1], da[2], da[3], b0[j], b1[j], sfad,
                              sfb[j]);
            }
        }
        pd_bs_bar_arrive(&bempty[s]);
    }

    // deterministic per-(token, slot) partials, folded by pd_moe_slot_combine
    #pragma unroll
    for (uint32_t j = 0; j < 2u; ++j) {
        const uint32_t c0 = joff + j * 8u + 2u * tq;
        #pragma unroll
        for (uint32_t n = 0; n < 2u; ++n) {
            const uint32_t rl0 = i0 + n * 16u + g;
            #pragma unroll
            for (uint32_t q = 0; q < 4u; ++q) {
                const uint32_t rloc = rl0 + ((q & 2u) ? 8u : 0u);
                const uint32_t r = row_base + rloc;
                const uint32_t c = c0 + (q & 1u);
                const unsigned int token = tok[p][c];
                if (r >= embd || token == PD_MOE_PAD) continue;
                const float w = topk_w[(size_t)token * n_active + slt[p][c]];
                const float v = acc[n][j][q] + bds[p * BMR + rloc];
                const size_t pidx = ((size_t)token * n_active + slt[p][c]) * embd + r;
                // PADDOCK_MOE_PART_BF16 (prefill-only, launcher-gated, never
                // with the fused fold): halves the partials round trip; the
                // bf16 round is the only numeric change, combine still sums
                // f32 in fixed slot order (PPL-gated trade).
                if (PBF16)
                    ((__nv_bfloat16*)part)[pidx] = __float2bfloat16(w * v);
                else
                    part[pidx] = w * v;
            }
        }
    }

    // Fused slot_combine (the _res entry point): last-arrival fold. Each CTA
    // covers one (expert block, 128-col y-tile); a token's n_active slots sit
    // in n_active DISTINCT expert blocks (topk experts are distinct), so per
    // CTA a token appears at most once. After the partial stores, one thread
    // per row bumps cnt[token * gridDim.y + blockIdx.y]; whoever completes
    // the n_active-th arrival owns the fold for that (token, y-tile) and sums
    // the slot partials in FIXED slot order into the residual - per-element
    // identical to pd_moe_slot_combine (bit-exact), one launch earlier.
    // Counters are never reset: every launch adds exactly n_active per key,
    // so "last" is (old % n_active) == n_active-1. That stays aligned across
    // the u32 wrap only for power-of-two n_active - the launcher rejects the
    // rest. Producers returned above; bar.sync 1 counts the 256 consumers.
    if (residual != nullptr) {
        // (virtual y-tile index: under PERSIST there is no blockIdx.y; the
        // computed value equals the old gridDim.y/blockIdx.y pair exactly)
        __shared__ unsigned char winner[BM];
        __threadfence();  // publish this CTA's partials before any counter bump
        asm volatile("bar.sync 1, 256;" ::: "memory");
        if (tid < BM) {
            winner[tid] = 0u;
            const unsigned int token = tok[p][tid];
            if (token != PD_MOE_PAD) {
                unsigned int old =
                    atomicAdd(&cnt[(size_t)token * ny + row_base / BMR], 1u);
                winner[tid] = (old % n_active) == n_active - 1u;
            }
        }
        asm volatile("bar.sync 1, 256;" ::: "memory");
        __threadfence();  // acquire the other slot CTAs' partials before reading
        for (uint32_t c = 0; c < BM; ++c) {
            if (!winner[c]) continue;
            const unsigned int token = tok[p][c];
            for (uint32_t r = tid; r < BMR; r += 256u) {
                const uint32_t rr = row_base + r;
                if (rr >= embd) continue;
                float accv = 0.0f;
                for (uint32_t k = 0; k < n_active; ++k)
                    accv += part[((size_t)token * n_active + k) * embd + rr];
                residual[(size_t)token * embd + rr] += accv;
            }
        }
    }
    }  // item loop
#else
    (void)down_data; (void)down_scale; (void)down_bias; (void)sorted_row; (void)sorted_slot;
    (void)block_expert; (void)topk_w; (void)fq; (void)fs; (void)part; (void)residual;
    (void)cnt; (void)ff; (void)embd; (void)n_active; (void)nb;
#endif
}

static int pd_mxfp4_moe_down_bs_impl(const void* down_data, const void* down_scale,
                                     const void* down_bias, const void* sorted_row,
                                     const void* sorted_slot, const void* block_expert,
                                     const void* topk_w, const void* fq, const void* fs,
                                     void* part, void* residual, void* cnt, uint32_t ff,
                                     uint32_t embd, uint32_t n_active,
                                     uint32_t max_blocks, uint32_t rows, void* stream) {
#ifndef PD_BS_HOST
    (void)down_data; (void)down_scale; (void)down_bias; (void)sorted_row; (void)sorted_slot;
    (void)block_expert; (void)topk_w; (void)fq; (void)fs; (void)part; (void)residual;
    (void)cnt; (void)ff; (void)embd; (void)n_active; (void)max_blocks; (void)stream;
    return cudaErrorNotSupported;
#else
    if (max_blocks == 0 || embd == 0) return 0;
    if ((ff & 31u) != 0 || (embd & 31u) != 0) return cudaErrorInvalidValue;
    // the fused fold's no-reset counter trick needs n_active | 2^32
    if (residual != nullptr && (n_active == 0 || (n_active & (n_active - 1u))))
        return cudaErrorInvalidValue;
    // decode-class kernel (PADDOCK_MOE_DEC): plain part writes only - the
    // fused-residual fold (down_bs_res) keeps the _bs path
    static const bool dec_env_dn = pd_env("PADDOCK_MOE_DEC") != nullptr;
    if (dec_env_dn && rows < 256u && residual == nullptr) {
        const size_t sh = (size_t)PD_DEC_TT * ff * 2u;
        dim3 g(max_blocks, embd >> 5);
        pd_mxfp4_moe_down_dec_kernel<<<g, 256, sh, (cudaStream_t)stream>>>(
            (const unsigned char*)down_data, (const unsigned char*)down_scale,
            (const float*)down_bias, (const unsigned int*)sorted_row,
            (const unsigned int*)sorted_slot, (const unsigned int*)block_expert,
            (const float*)topk_w, (const unsigned char*)fq, (const unsigned char*)fs,
            (float*)part, ff, embd, n_active);
        return pd_launch_status();
    }
    static int nsm_dn = 0;
    static bool smem_dn = false, no_persist_dn = false;
    if (!smem_dn) {
        cudaFuncSetAttribute((const void*)pd_mxfp4_moe_down_bs_kernel<PD_BS_BM, 128u, 2u>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, PD_BS_DN_SMEM);
        cudaFuncSetAttribute(
            (const void*)pd_mxfp4_moe_down_bs_kernel<PD_BS_BM, 128u, 2u, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, PD_BS_DN_SMEM);
        cudaFuncSetAttribute(
            (const void*)pd_mxfp4_moe_down_bs_kernel<PD_BS_BM, 128u, 2u, true, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, PD_BS_DN_SMEM);
        cudaFuncSetAttribute(
            (const void*)pd_mxfp4_moe_down_bs_kernel<PD_BS_BM, 128u, 2u, false, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, PD_BS_DN_SMEM);
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&nsm_dn, cudaDevAttrMultiProcessorCount, dev);
        if (nsm_dn <= 0) nsm_dn = 128;
        no_persist_dn = pd_env("PADDOCK_NO_MOE_PERSIST") != nullptr;
        smem_dn = true;
    }
    const uint32_t ny = (embd + 127u) / 128u;
    const uint32_t nk = (ff + PD_BS_KC_DN - 1u) / PD_BS_KC_DN;
    // prefill-class only - same A/B rationale as gate_up above. Gate is
    // nk >= PD_BS_S (not > S+1): the down kernel carries four item-parity
    // metadata slots (see the kernel note), so shallow-K launches (qwen ff=512
    // -> nk=2) get the persistent path too - without it every prefill down
    // paid the 2-D grid's per-item epilogue silence (17.5% tensor vs gate_up's
    // 40.5% at half the FLOPs).
    // bf16 partials (PADDOCK_MOE_PART_BF16): prefill-class only (rows >= 256,
    // matching the engine's combine-side gate) and never with the fused fold
    // (its gather reads f32). Read per call for in-process A/Bs.
    const bool pbf16 = rows >= 256u && residual == nullptr &&
                       pd_env("PADDOCK_MOE_PART_BF16") != nullptr;
    if (!no_persist_dn && rows >= 256u && nk >= PD_BS_S) {
        const uint32_t nit = max_blocks * ny;
        const uint32_t g1 = nit < (uint32_t)nsm_dn ? nit : (uint32_t)nsm_dn;
        if (pbf16)
            pd_mxfp4_moe_down_bs_kernel<PD_BS_BM, 128u, 2u, true, true><<<g1,
                                          PD_BS_TH_DN, PD_BS_DN_SMEM,
                                          (cudaStream_t)stream>>>(
                (const unsigned char*)down_data, (const unsigned char*)down_scale,
                (const float*)down_bias, (const unsigned int*)sorted_row,
                (const unsigned int*)sorted_slot, (const unsigned int*)block_expert,
                (const float*)topk_w, (const unsigned char*)fq, (const unsigned char*)fs,
                (float*)part, (float*)residual, (unsigned int*)cnt, ff, embd, n_active,
                max_blocks);
        else
            pd_mxfp4_moe_down_bs_kernel<PD_BS_BM, 128u, 2u, true><<<g1, PD_BS_TH_DN,
                                          PD_BS_DN_SMEM, (cudaStream_t)stream>>>(
                (const unsigned char*)down_data, (const unsigned char*)down_scale,
                (const float*)down_bias, (const unsigned int*)sorted_row,
                (const unsigned int*)sorted_slot, (const unsigned int*)block_expert,
                (const float*)topk_w, (const unsigned char*)fq, (const unsigned char*)fs,
                (float*)part, (float*)residual, (unsigned int*)cnt, ff, embd, n_active,
                max_blocks);
        return pd_launch_status();
    }
    dim3 grid(max_blocks, ny);
    if (pbf16)
        pd_mxfp4_moe_down_bs_kernel<PD_BS_BM, 128u, 2u, false, true><<<grid,
                                      PD_BS_TH_DN, PD_BS_DN_SMEM, (cudaStream_t)stream>>>(
            (const unsigned char*)down_data, (const unsigned char*)down_scale,
            (const float*)down_bias, (const unsigned int*)sorted_row,
            (const unsigned int*)sorted_slot, (const unsigned int*)block_expert,
            (const float*)topk_w, (const unsigned char*)fq, (const unsigned char*)fs,
            (float*)part, (float*)residual, (unsigned int*)cnt, ff, embd, n_active,
            max_blocks);
    else
        pd_mxfp4_moe_down_bs_kernel<PD_BS_BM, 128u, 2u><<<grid, PD_BS_TH_DN,
                                      PD_BS_DN_SMEM, (cudaStream_t)stream>>>(
            (const unsigned char*)down_data, (const unsigned char*)down_scale,
            (const float*)down_bias, (const unsigned int*)sorted_row,
            (const unsigned int*)sorted_slot, (const unsigned int*)block_expert,
            (const float*)topk_w, (const unsigned char*)fq, (const unsigned char*)fs,
            (float*)part, (float*)residual, (unsigned int*)cnt, ff, embd, n_active,
            max_blocks);
    return pd_launch_status();
#endif
}

PD_EXPORT
int pd_mxfp4_moe_down_bs(const void* down_data, const void* down_scale,
                         const void* down_bias, const void* sorted_row,
                         const void* sorted_slot, const void* block_expert,
                         const void* topk_w, const void* fq, const void* fs, void* part,
                         uint32_t ff, uint32_t embd, uint32_t n_active,
                         uint32_t max_blocks, uint32_t rows, void* stream) {
    return pd_mxfp4_moe_down_bs_impl(down_data, down_scale, down_bias, sorted_row,
                                     sorted_slot, block_expert, topk_w, fq, fs, part,
                                     nullptr, nullptr, ff, embd, n_active, max_blocks,
                                     rows, stream);
}

// down_bs with the slot_combine fold fused into the epilogue (last-arrival
// via `cnt`, one u32 per (token, 128-col y-tile), zeroed once at alloc and
// never reset). Saves the separate combine launch + its cold re-read of the
// partials; bit-exact vs down_bs + pd_moe_slot_combine.
PD_EXPORT
int pd_mxfp4_moe_down_bs_res(const void* down_data, const void* down_scale,
                             const void* down_bias, const void* sorted_row,
                             const void* sorted_slot, const void* block_expert,
                             const void* topk_w, const void* fq, const void* fs,
                             void* part, void* residual, void* cnt, uint32_t ff,
                             uint32_t embd, uint32_t n_active, uint32_t max_blocks,
                             uint32_t rows, void* stream) {
    return pd_mxfp4_moe_down_bs_impl(down_data, down_scale, down_bias, sorted_row,
                                     sorted_slot, block_expert, topk_w, fq, fs, part,
                                     residual, cnt, ff, embd, n_active, max_blocks,
                                     rows, stream);
}

// Prefill-config down: pairs with pd_mxfp4_moe_gate_up_bs64 (64-row sorted
// blocks in fq/fs, 64-row weight tiles, grid.y = embd/64). No fused fold on
// this path - it exists for fat-expert prefill ticks where the combine is a
// vanishing fraction of the pass.
#define PD_BS64_DN_STAGE (64u * PD_BS_WROW_DN + 64u * PD_BS_YROW_DN \
                          + 64u * PD_BS_KB_DN + 64u * PD_BS_KB_DN)
#define PD_BS64_DN_SMEM (32u + 4u * 64u * 4u + PD_BS_S * PD_BS64_DN_STAGE)
PD_EXPORT
int pd_mxfp4_moe_down_bs64(const void* down_data, const void* down_scale,
                           const void* down_bias, const void* sorted_row,
                           const void* sorted_slot, const void* block_expert,
                           const void* topk_w, const void* fq, const void* fs,
                           void* part, uint32_t ff, uint32_t embd, uint32_t n_active,
                           uint32_t max_blocks, uint32_t rows, void* stream) {
    (void)rows;
#ifndef PD_BS_HOST
    (void)down_data; (void)down_scale; (void)down_bias; (void)sorted_row; (void)sorted_slot;
    (void)block_expert; (void)topk_w; (void)fq; (void)fs; (void)part; (void)ff; (void)embd;
    (void)n_active; (void)max_blocks; (void)stream;
    return cudaErrorNotSupported;
#else
    if (max_blocks == 0 || embd == 0) return 0;
    if ((ff & 31u) != 0 || (embd & 31u) != 0) return cudaErrorInvalidValue;
    static int nsm_dn64 = 0;
    static bool smem_dn64 = false, no_persist_dn64 = false;
    if (!smem_dn64) {
        cudaFuncSetAttribute((const void*)pd_mxfp4_moe_down_bs_kernel<64u, 64u, 4u>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, PD_BS64_DN_SMEM);
        cudaFuncSetAttribute(
            (const void*)pd_mxfp4_moe_down_bs_kernel<64u, 64u, 4u, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, PD_BS64_DN_SMEM);
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&nsm_dn64, cudaDevAttrMultiProcessorCount, dev);
        if (nsm_dn64 <= 0) nsm_dn64 = 128;
        no_persist_dn64 = pd_env("PADDOCK_NO_MOE_PERSIST") != nullptr;
        smem_dn64 = true;
    }
    // persistent work loop, same nk >= PD_BS_S gate as the BM=32 down above
    // (the 4 item-parity metadata slots make shallow K safe; qwen ff=512 ->
    // nk=2). Without this branch the bs64 pair was only ever serving-A/B'd
    // with a plain-2D down - 8192 CTAs at the 2048-row qwen35 tick, each
    // paying the per-item epilogue silence the persistent redesign removed.
    const uint32_t nk64 = (ff + PD_BS_KC_DN - 1u) / PD_BS_KC_DN;
    if (!no_persist_dn64 && rows >= 256u && nk64 >= PD_BS_S) {
        const uint32_t nit = max_blocks * ((embd + 63u) / 64u);
        const uint32_t g1 = nit < (uint32_t)nsm_dn64 ? nit : (uint32_t)nsm_dn64;
        pd_mxfp4_moe_down_bs_kernel<64u, 64u, 4u, true><<<g1, PD_BS_TH_DN,
                                      PD_BS64_DN_SMEM, (cudaStream_t)stream>>>(
            (const unsigned char*)down_data, (const unsigned char*)down_scale,
            (const float*)down_bias, (const unsigned int*)sorted_row,
            (const unsigned int*)sorted_slot, (const unsigned int*)block_expert,
            (const float*)topk_w, (const unsigned char*)fq, (const unsigned char*)fs,
            (float*)part, nullptr, nullptr, ff, embd, n_active, max_blocks);
        return pd_launch_status();
    }
    dim3 grid(max_blocks, (embd + 63u) / 64u);
    pd_mxfp4_moe_down_bs_kernel<64u, 64u, 4u><<<grid, PD_BS_TH_DN, PD_BS64_DN_SMEM,
                                  (cudaStream_t)stream>>>(
        (const unsigned char*)down_data, (const unsigned char*)down_scale,
        (const float*)down_bias, (const unsigned int*)sorted_row,
        (const unsigned int*)sorted_slot, (const unsigned int*)block_expert,
        (const float*)topk_w, (const unsigned char*)fq, (const unsigned char*)fs,
        (float*)part, nullptr, nullptr, ff, embd, n_active, max_blocks);
    return pd_launch_status();
#endif
}

