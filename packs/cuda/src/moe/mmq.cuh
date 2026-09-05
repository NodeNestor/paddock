// moe/mmq.cuh (formerly 10_moe_mmq.cuh) - MoE expert mmq GEMMs, slot combine, batched copy
// Textually-included segment of the single pack translation unit.
// Not standalone-compilable: include order is defined by ../pack.cu.
// --------------------------------------------------- MoE expert mmq GEMMs
// int8 tensor-core MoE GEMMs over the sorted (moe_align) layout - the mmq
// kernel's structure (K staged 256 deep, fully-unrolled compile-time staging,
// m16n8k32 s8 MMA, per-k32-block scale folding) applied to the MXFP4 expert
// weights. Each block owns one 32-token sorted block x one 128-wide strip of
// expert output columns and walks the full K; the fp4 nibbles unpack straight
// to int8 through pd_fp4_unpack8 (the doubled-value table; pd_e8m0_half's
// halved scale compensates, same convention as the dp4a decode kernels).
// Activations arrive pre-quantized in the STRIDED per-row int8 layout
// (pd_quantize_q8): token rows are gathered through sorted_row, so no flat
// mmq activation buffer is needed. Token blocks ride the fast grid axis so
// the same expert's consecutive blocks reuse the weight strip through L2
// (the P6g lesson). The f16 WMMA `_sorted_tc` kernels stay as the pinned
// fallback class; this kernel replaces them wherever mmq numerics are
// acceptable (the greedy-gate bar).
#define PD_MOEQ_BM 32                            // tokens per sorted block
// y-tile in the unpacked weight-tile layout (64 data int32 + 8 scale f32,
// PD_MMQ_XK stride) staged at full 256-K depth: one stage + one barrier per
// kt (the 32-col tile has only 2 j-groups of mma per barrier, so barrier
// count is what bounds it).
#define PD_MOEQ_Y_INT32 (PD_MOEQ_BM * PD_MMQ_XK)
// PACKED weight tile: 32 data int32 (8 fp4 blocks x 16 B, still packed) +
// 2 int32 of raw e8m0 scale bytes + 2 pad -> stride 36 int32. Rows stay
// 16B-aligned (144 B) for the 16-byte async copies, and the 4g bank offset
// avoids the stride-32 full-conflict. fp4 words unpack to int8 at FRAGMENT
// LOAD (one pd_fp4_unpack8 yields both k-halves of an A-fragment pair).
// The data words arrive via a cp.async double BUFFER (issue kt+1 while kt
// computes - the kernels were latency-bound at ~40% DRAM/SM with fully
// synchronous staging); the packed size is what lets two w-buffers + the
// y-tile fit 2 blocks/SM (2 x 46.6 KB). gate/up stay two K-passes over one
// buffer pair: both matrices in flight at once doubles the concurrent
// blocks' L2 working set past the 6 MB L2 and re-reads spill to DRAM
// (measured 87% DRAM / 2x duration on the single-walk variant).
#define PD_MOEQ_WK 36
#define PD_MOEQ_W_INT32 (128u * PD_MOEQ_WK)
#define PD_MOEQ_GU_SMEM ((2u * PD_MOEQ_W_INT32 + PD_MOEQ_Y_INT32) * 4u)
// (measured MUCH worse: single buffer + __launch_bounds__(256,3) for 3
// blocks/SM - the 85-register cap spills and the lost cp.async overlap cost
// ~30% e2e. The shared-scoreboard stall at 2 blocks/SM is the cheaper evil.)
#define PD_MOEQ_DN_SMEM PD_MOEQ_GU_SMEM

// 16-byte global->shared async copy; src_ok=false zero-fills (src-size 0).
__device__ __forceinline__ void pd_cp_async16(void* smem, const void* gmem, bool src_ok) {
#if PD_MMA_OK
    const unsigned sm = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(sm), "l"(gmem),
                 "r"(src_ok ? 16u : 0u));
#endif
}

// Issue the packed data words of one weight strip chunk (128 rows x 8 fp4
// blocks, one 16-byte cp.async per (row, block)); pair with a commit_group
// + wait_group before compute. Scale bytes are not part of this (they are
// unaligned byte loads - staged synchronously with the y-tile).
__device__ __forceinline__ void pd_moeq_issue_w(
    int* __restrict__ tile, const unsigned char* __restrict__ data, size_t wrow0,
    uint32_t row_base, uint32_t out_dim, uint32_t n_blocks, uint32_t kt, uint32_t tid,
    uint32_t plane) {
#if PD_MMA_OK
    // plane 0/1 = the g||u ILV layout (block gb of a row sits in 128 B pair
    // gb/4 at +plane*64; `data` is the fused plane). plane 2 = the plain
    // single-plane layout (the down experts keep it).
    const size_t pitch = (size_t)(((n_blocks + 3u) >> 2) * 128u);
    #pragma unroll
    for (uint32_t it = 0; it < 4u; ++it) {
        const uint32_t i = it * 256u + tid;
        const uint32_t row = i >> 3, b = i & 7u, gb = kt * 8u + b;
        const bool ok = gb < n_blocks && (row_base + row) < out_dim;
        const uint32_t sb = ok ? gb : 0u;
        const size_t src = plane == 2u
            ? ((wrow0 + row) * n_blocks + sb) * 16u
            : (wrow0 + row) * pitch + (size_t)(sb >> 2) * 128u + plane * 64u +
                  (sb & 3u) * 16u;
        pd_cp_async16(tile + row * PD_MOEQ_WK + b * 4u, data + src, ok);
    }
#endif
}

// Stage one chunk's raw scale bytes (8 per row; unaligned in the stream, so
// plain byte loads). Out-of-range slots get byte 0: the data words are zero,
// so the (denormal) e8m0(0) scale multiplies a zero dot - exact zero.
__device__ __forceinline__ void pd_moeq_stage_ws(
    int* __restrict__ tile, const unsigned char* __restrict__ scale, size_t wrow0,
    uint32_t row_base, uint32_t out_dim, uint32_t n_blocks, uint32_t kt, uint32_t tid) {
#if PD_MMA_OK
    #pragma unroll
    for (uint32_t it = 0; it < 4u; ++it) {
        const uint32_t i = it * 256u + tid;
        const uint32_t row = i >> 3, b = i & 7u, gb = kt * 8u + b;
        ((uint8_t*)(tile + row * PD_MOEQ_WK + 32u))[b] =
            (gb < n_blocks && (row_base + row) < out_dim)
                ? scale[(wrow0 + row) * n_blocks + gb]
                : (uint8_t)0;
    }
#endif
}

// Stage one full 256-K activation chunk for the 32 sorted rows into the
// weight-tile layout (64 data int32 + 8 f32 scales per col, PD_MMQ_XK
// stride). `xq`/`xs` are the strided quantize_q8 layout; `rows[c]` maps
// tile col c to its activation row (PD_MOE_PAD -> zeros).
// Templated on BM (tokens/block). Default 32 keeps every existing caller
// bit-identical (the BM=32 loop bounds reduce to the original 8 data iters +
// one full-warp scale pass). BM=64 (the wider prefill block) needs 16 data
// iters and 2 scale passes.
template <uint32_t BM = PD_MOEQ_BM>
__device__ __forceinline__ void pd_moeq_stage_y(
    int* __restrict__ tile_y, const int* __restrict__ xq32,
    const float* __restrict__ xs, const unsigned int* __restrict__ rows,
    uint32_t in_dim, uint32_t kt, uint32_t tid) {
#if PD_MMA_OK
    const uint32_t n_k32 = in_dim >> 2, n_blocks = in_dim >> 5;
    // BM cols x 64 data int32: BM/4 compile-time iterations
    #pragma unroll
    for (uint32_t it = 0; it < (BM * 64u / 256u); ++it) {
        const uint32_t idx = it * 256u + tid;
        const uint32_t c = idx >> 6, k = idx & 63u, gk = kt * 64u + k;
        const unsigned int row = rows[c];
        tile_y[c * PD_MMQ_XK + k] =
            (row != PD_MOE_PAD && gk < n_k32) ? xq32[(size_t)row * n_k32 + gk] : 0;
    }
    // BM cols x 8 scales
    #pragma unroll
    for (uint32_t it = 0; it < (BM * 8u + 255u) / 256u; ++it) {
        const uint32_t idx = it * 256u + tid;
        if (idx < BM * 8u) {
            const uint32_t c = idx >> 3, b = idx & 7u, gb = kt * 8u + b;
            const unsigned int row = rows[c];
            ((float*)tile_y)[c * PD_MMQ_XK + 64u + b] =
                (row != PD_MOE_PAD && gb < n_blocks) ? xs[(size_t)row * n_blocks + gb] : 0.f;
        }
    }
#endif
}

__global__ void __launch_bounds__(256, 2) pd_mxfp4_moe_gate_up_mmq_kernel(
    const unsigned char* __restrict__ gate_data, const unsigned char* __restrict__ gate_scale,
    const float* __restrict__ gate_bias,
    const unsigned char* __restrict__ up_data, const unsigned char* __restrict__ up_scale,
    const float* __restrict__ up_bias,
    const unsigned int* __restrict__ sorted_row, const unsigned int* __restrict__ block_expert,
    const int8_t* __restrict__ xq, const float* __restrict__ xs,
    int8_t* __restrict__ fq, float* __restrict__ fs, uint32_t in_dim, uint32_t ff,
    float alpha, float limit, float up_add) {
#if PD_MMA_OK
    const uint32_t blk = blockIdx.x;                 // token block (fast axis: L2 strip reuse)
    const uint32_t e = block_expert[blk];
    if (e == PD_MOE_PAD) return;
    const uint32_t row_base = blockIdx.y * 128u;     // output-column strip
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u, warp = tid >> 5;
    const uint32_t g = lane >> 2, t = lane & 3u;
    const uint32_t i0 = (warp >> 1) * 32u;
    const uint32_t joff = (warp & 1u) * 8u;
    const uint32_t n_blocks = in_dim >> 5;
    const uint32_t nk = (in_dim + 255u) >> 8;

    extern __shared__ int pd_moeq_sh[];
    int* tile_y = pd_moeq_sh;                        // 32 cols x 76 int32 (unpacked)
    int* wbuf0 = pd_moeq_sh + PD_MOEQ_Y_INT32;       // 2 x (128 rows x 36 int32, packed)
    int* wbuf1 = wbuf0 + PD_MOEQ_W_INT32;
    __shared__ unsigned int tok[PD_MOEQ_BM];
    if (tid < PD_MOEQ_BM) tok[tid] = sorted_row[(size_t)blk * PD_MOEQ_BM + tid];
    __syncthreads();

    float acc_g[4][4] = {}, acc_u[4][4] = {};
    const size_t wrow0 = (size_t)e * ff + row_base;
    #pragma unroll
    for (uint32_t mat = 0; mat < 2u; ++mat) {
        const unsigned char* ws = mat ? up_scale : gate_scale;
        float (*acc)[4] = mat ? acc_u : acc_g;
        pd_moeq_issue_w(wbuf0, gate_data, wrow0, row_base, ff, n_blocks, 0, tid, mat);
        asm volatile("cp.async.commit_group;");
        for (uint32_t kt = 0; kt < nk; ++kt) {
            int* tw = (kt & 1u) ? wbuf1 : wbuf0;
            asm volatile("cp.async.wait_group 0;");
            pd_moeq_stage_ws(tw, ws, wrow0, row_base, ff, n_blocks, kt, tid);
            pd_moeq_stage_y(tile_y, (const int*)xq, xs, tok, in_dim, kt, tid);
            __syncthreads();
            if (kt + 1u < nk) {
                // next chunk's data words fly while this chunk computes; the
                // target buffer's last reader finished before the sync above
                pd_moeq_issue_w((kt & 1u) ? wbuf0 : wbuf1, gate_data, wrow0, row_base, ff,
                                n_blocks, kt + 1u, tid, mat);
                asm volatile("cp.async.commit_group;");
            }

            #pragma unroll
            for (uint32_t h = 0; h < 2u; ++h) {
                const uint32_t k00 = h * 32u;
                int A[2][4][4];
                float dA[2][2][4];
                #pragma unroll
                for (uint32_t n = 0; n < 2u; ++n) {
                    const uint32_t r0 = (i0 + n * 16u + g) * PD_MOEQ_WK;
                    const uint32_t r8 = (i0 + n * 16u + 8u + g) * PD_MOEQ_WK;
                    #pragma unroll
                    for (uint32_t kk = 0; kk < 4u; ++kk) {
                        const uint32_t bb = (k00 >> 3) + kk;    // fp4 block in chunk
                        const int2 u0 = pd_fp4_unpack8((uint32_t)tw[r0 + bb * 4u + t]);
                        const int2 u1 = pd_fp4_unpack8((uint32_t)tw[r8 + bb * 4u + t]);
                        A[n][kk][0] = u0.x;
                        A[n][kk][2] = u0.y;
                        A[n][kk][1] = u1.x;
                        A[n][kk][3] = u1.y;
                        dA[n][0][kk] = pd_e8m0_half(((const uint8_t*)(tw + r0 + 32u))[bb]);
                        dA[n][1][kk] = pd_e8m0_half(((const uint8_t*)(tw + r8 + 32u))[bb]);
                    }
                }
                #pragma unroll
                for (uint32_t j0 = 0; j0 < PD_MOEQ_BM; j0 += 16u) {
                    const uint32_t jc = j0 + joff;
                    #pragma unroll
                    for (uint32_t kk = 0; kk < 4u; ++kk) {
                        const uint32_t ko = k00 + kk * 8u;
                        const int b0 = tile_y[(jc + g) * PD_MMQ_XK + ko + t];
                        const int b1 = tile_y[(jc + g) * PD_MMQ_XK + ko + 4u + t];
                        const float dB0 =
                            ((const float*)tile_y)[(jc + 2u * t) * PD_MMQ_XK + 64u + (k00 >> 3) + kk];
                        const float dB1 =
                            ((const float*)tile_y)[(jc + 2u * t + 1u) * PD_MMQ_XK + 64u + (k00 >> 3) + kk];
                        #pragma unroll
                        for (uint32_t n = 0; n < 2u; ++n) {
                            int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
                            asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                                : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
                                : "r"(A[n][kk][0]), "r"(A[n][kk][1]), "r"(A[n][kk][2]),
                                  "r"(A[n][kk][3]), "r"(b0), "r"(b1));
                            acc[(j0 >> 3) + n][0] += dA[n][0][kk] * dB0 * (float)d0;
                            acc[(j0 >> 3) + n][1] += dA[n][0][kk] * dB1 * (float)d1;
                            acc[(j0 >> 3) + n][2] += dA[n][1][kk] * dB0 * (float)d2;
                            acc[(j0 >> 3) + n][3] += dA[n][1][kk] * dB1 * (float)d3;
                        }
                    }
                }
            }
            __syncthreads();  // tile_y + the buffers are rewritten next kt
        }
    }

    // bias + OAI-clamp swiglu, quantized per-32 output block in REGISTERS and
    // emitted as int8 + scales - the down GEMM's direct input; the f32
    // activation never lands in memory (P6j applied to the MoE). For a fixed
    // token col c, the 32-row block [i0, i0+32) lives in the 8 lanes sharing
    // t (g = 0..7), 4 regs each - amax reduces over shfl_xor strides 4/8/16.
    // fmax is order-independent and the scale expression matches
    // pd_quantize_q8, so fq/fs are bit-identical to the two-step path.
    // PAD rows get exact zeros (data and scale).
    const uint32_t n_sb = ff >> 5;
    #pragma unroll
    for (uint32_t j0 = 0; j0 < PD_MOEQ_BM; j0 += 16u) {
        #pragma unroll
        for (uint32_t qc = 0; qc < 2u; ++qc) {
            const uint32_t c = j0 + joff + 2u * t + qc;
            const bool pad = tok[c] == PD_MOE_PAD;
            const uint32_t rb = row_base + i0;   // this group's 32-row block base
            float sw[4];
            #pragma unroll
            for (uint32_t n = 0; n < 2u; ++n) {
                #pragma unroll
                for (uint32_t hq = 0; hq < 2u; ++hq) {  // +0 / +8 rows
                    const uint32_t q = qc + 2u * hq;
                    const uint32_t r = rb + n * 16u + hq * 8u + g;
                    float out = 0.f;
                    if (!pad && r < ff) {
                        const float gv = acc_g[(j0 >> 3) + n][q] + gate_bias[(size_t)e * ff + r];
                        const float uv = acc_u[(j0 >> 3) + n][q] + up_bias[(size_t)e * ff + r];
                        const float xg = fminf(gv, limit);
                        const float yu = fminf(fmaxf(uv, -limit), limit);
                        // up_add: 1.0 = gpt-oss (u+1); 0.0 = qwen plain silu(g)*u
                        out = (xg / (1.0f + expf(-alpha * xg))) * (yu + up_add);
                    }
                    sw[n * 2u + hq] = out;
                }
            }
            float a = fmaxf(fmaxf(fabsf(sw[0]), fabsf(sw[1])), fmaxf(fabsf(sw[2]), fabsf(sw[3])));
            #pragma unroll
            for (uint32_t o = 4; o <= 16u; o <<= 1)
                a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
            const float scl = a * (1.0f / 127.0f);
            const float invs = scl > 0.f ? 1.0f / scl : 0.f;
            const size_t row = (size_t)blk * PD_MOEQ_BM + c;
            if (rb < ff) {   // ff % 32 == 0: a 32-block is fully in or out
                #pragma unroll
                for (uint32_t v = 0; v < 4u; ++v) {
                    const uint32_t r = rb + (v >> 1) * 16u + (v & 1u) * 8u + g;
                    int qi = __float2int_rn(sw[v] * invs);
                    qi = qi < -127 ? -127 : (qi > 127 ? 127 : qi);
                    fq[row * ff + r] = (int8_t)qi;
                }
                if (g == 0) fs[row * n_sb + (rb >> 5)] = scl;
            }
        }
    }
#else
    (void)gate_data; (void)gate_scale; (void)gate_bias; (void)up_data; (void)up_scale;
    (void)up_bias; (void)sorted_row; (void)block_expert; (void)xq; (void)xs;
    (void)fq; (void)fs; (void)in_dim; (void)ff; (void)alpha; (void)limit;
#endif
}

// Down half: same tile shape over K = ff; the activation rows are the sorted
// fused_sorted rows quantized in place (strided layout, indexed directly by
// blk*32+c - no gather), the epilogue applies down bias, weights by topk_w
// and atomic-adds into the residual (same semantics as the _sorted kernels).
__global__ void __launch_bounds__(256, 2) pd_mxfp4_moe_down_mmq_kernel(
    const unsigned char* __restrict__ down_data, const unsigned char* __restrict__ down_scale,
    const float* __restrict__ down_bias,
    const unsigned int* __restrict__ sorted_row, const unsigned int* __restrict__ sorted_slot,
    const unsigned int* __restrict__ block_expert, const float* __restrict__ topk_w,
    const int8_t* __restrict__ fq, const float* __restrict__ fs,
    float* __restrict__ part, uint32_t ff, uint32_t embd, uint32_t n_active) {
#if PD_MMA_OK
    const uint32_t blk = blockIdx.x;
    const uint32_t e = block_expert[blk];
    if (e == PD_MOE_PAD) return;
    const uint32_t row_base = blockIdx.y * 128u;
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u, warp = tid >> 5;
    const uint32_t g = lane >> 2, t = lane & 3u;
    const uint32_t i0 = (warp >> 1) * 32u;
    const uint32_t joff = (warp & 1u) * 8u;
    const uint32_t n_blocks = ff >> 5;
    const uint32_t nk = (ff + 255u) >> 8;

    extern __shared__ int pd_moeq_sh[];
    int* tile_y = pd_moeq_sh;
    int* wbuf0 = pd_moeq_sh + PD_MOEQ_Y_INT32;
    int* wbuf1 = wbuf0 + PD_MOEQ_W_INT32;
    __shared__ unsigned int tok[PD_MOEQ_BM], slt[PD_MOEQ_BM], srow[PD_MOEQ_BM];
    if (tid < PD_MOEQ_BM) {
        tok[tid] = sorted_row[(size_t)blk * PD_MOEQ_BM + tid];
        slt[tid] = sorted_slot[(size_t)blk * PD_MOEQ_BM + tid];
        srow[tid] = blk * PD_MOEQ_BM + tid;   // direct row: fq is already sorted
    }
    __syncthreads();

    float acc[4][4] = {};
    const size_t wrow0 = (size_t)e * embd + row_base;
    pd_moeq_issue_w(wbuf0, down_data, wrow0, row_base, embd, n_blocks, 0, tid, 2u);
    asm volatile("cp.async.commit_group;");
    for (uint32_t kt = 0; kt < nk; ++kt) {
        int* tile_d = (kt & 1u) ? wbuf1 : wbuf0;
        asm volatile("cp.async.wait_group 0;");
        pd_moeq_stage_ws(tile_d, down_scale, wrow0, row_base, embd, n_blocks, kt, tid);
        pd_moeq_stage_y(tile_y, (const int*)fq, fs, srow, ff, kt, tid);
        __syncthreads();
        if (kt + 1u < nk) {
            pd_moeq_issue_w((kt & 1u) ? wbuf0 : wbuf1, down_data, wrow0, row_base, embd,
                            n_blocks, kt + 1u, tid, 2u);
            asm volatile("cp.async.commit_group;");
        }

        #pragma unroll
        for (uint32_t h = 0; h < 2u; ++h) {
            const uint32_t k00 = h * 32u;
            int A[2][4][4];
            float dA[2][2][4];
            #pragma unroll
            for (uint32_t n = 0; n < 2u; ++n) {
                const uint32_t r0 = (i0 + n * 16u + g) * PD_MOEQ_WK;
                const uint32_t r8 = (i0 + n * 16u + 8u + g) * PD_MOEQ_WK;
                #pragma unroll
                for (uint32_t kk = 0; kk < 4u; ++kk) {
                    const uint32_t bb = (k00 >> 3) + kk;
                    const int2 u0 = pd_fp4_unpack8((uint32_t)tile_d[r0 + bb * 4u + t]);
                    const int2 u1 = pd_fp4_unpack8((uint32_t)tile_d[r8 + bb * 4u + t]);
                    A[n][kk][0] = u0.x;
                    A[n][kk][2] = u0.y;
                    A[n][kk][1] = u1.x;
                    A[n][kk][3] = u1.y;
                    dA[n][0][kk] = pd_e8m0_half(((const uint8_t*)(tile_d + r0 + 32u))[bb]);
                    dA[n][1][kk] = pd_e8m0_half(((const uint8_t*)(tile_d + r8 + 32u))[bb]);
                }
            }
            #pragma unroll
            for (uint32_t j0 = 0; j0 < PD_MOEQ_BM; j0 += 16u) {
                const uint32_t jc = j0 + joff;
                #pragma unroll
                for (uint32_t kk = 0; kk < 4u; ++kk) {
                    const uint32_t ko = k00 + kk * 8u;
                    const int b0 = tile_y[(jc + g) * PD_MMQ_XK + ko + t];
                    const int b1 = tile_y[(jc + g) * PD_MMQ_XK + ko + 4u + t];
                    const float dB0 =
                        ((const float*)tile_y)[(jc + 2u * t) * PD_MMQ_XK + 64u + (k00 >> 3) + kk];
                    const float dB1 =
                        ((const float*)tile_y)[(jc + 2u * t + 1u) * PD_MMQ_XK + 64u + (k00 >> 3) + kk];
                    #pragma unroll
                    for (uint32_t n = 0; n < 2u; ++n) {
                        int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
                        asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                            : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
                            : "r"(A[n][kk][0]), "r"(A[n][kk][1]), "r"(A[n][kk][2]),
                              "r"(A[n][kk][3]), "r"(b0), "r"(b1));
                        acc[(j0 >> 3) + n][0] += dA[n][0][kk] * dB0 * (float)d0;
                        acc[(j0 >> 3) + n][1] += dA[n][0][kk] * dB1 * (float)d1;
                        acc[(j0 >> 3) + n][2] += dA[n][1][kk] * dB0 * (float)d2;
                        acc[(j0 >> 3) + n][3] += dA[n][1][kk] * dB1 * (float)d3;
                    }
                }
            }
        }
        __syncthreads();  // tiles are overwritten next kt
    }

    // DETERMINISTIC epilogue: each (token, slot, r) has exactly one writer in
    // the sorted layout (a token appears once per chosen expert; strips are
    // disjoint in r), so the weighted result lands in a per-slot partials
    // buffer with plain stores. pd_moe_slot_combine then folds the n_active
    // partials into the residual in FIXED slot order - no atomicAdd, so
    // logits are bit-reproducible run to run (the atomic scatter flipped a
    // near-tie greedy token ~50% of runs at the b9895 prefill gate).
    #pragma unroll
    for (uint32_t j0 = 0; j0 < PD_MOEQ_BM; j0 += 16u) {
        const uint32_t c0 = j0 + joff + 2u * t;
        #pragma unroll
        for (uint32_t n = 0; n < 2u; ++n) {
            const uint32_t r0 = row_base + i0 + n * 16u + g;
            const uint32_t r8 = r0 + 8u;
            #pragma unroll
            for (uint32_t q = 0; q < 4u; ++q) {
                const uint32_t r = (q & 2u) ? r8 : r0;
                const uint32_t c = c0 + (q & 1u);
                const unsigned int token = tok[c];
                if (r >= embd || token == PD_MOE_PAD) continue;
                const float w = topk_w[(size_t)token * n_active + slt[c]];
                const float v = acc[(j0 >> 3) + n][q] + down_bias[(size_t)e * embd + r];
                part[((size_t)token * n_active + slt[c]) * embd + r] = w * v;
            }
        }
    }
#else
    (void)down_data; (void)down_scale; (void)down_bias; (void)sorted_row; (void)sorted_slot;
    (void)block_expert; (void)topk_w; (void)fq; (void)fs; (void)part;
    (void)ff; (void)embd; (void)n_active;
#endif
}

// Fold the per-(token, slot) down partials into the residual in FIXED slot
// order - the deterministic tail of pd_mxfp4_moe_down_mmq. float4 lanes
// (embd % 4 == 0): the scalar version measured ~175 GB/s.
__global__ void pd_moe_slot_combine_kernel(
    const float4* __restrict__ part, float4* __restrict__ residual,
    uint32_t embd4, uint32_t n_active, uint32_t rows) {
    PD_PDL_ARM();  // consumer-safe for early PDL launches (2026-08-31)
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)rows * embd4) return;
    const size_t t = i / embd4, r = i % embd4;
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    for (uint32_t k = 0; k < n_active; ++k) {
        const float4 p = part[((size_t)t * n_active + k) * embd4 + r];
        acc.x += p.x; acc.y += p.y; acc.z += p.z; acc.w += p.w;
    }
    float4 d = residual[i];
    d.x += acc.x; d.y += acc.y; d.z += acc.z; d.w += acc.w;
    residual[i] = d;
}

PD_EXPORT
int pd_moe_slot_combine(const void* part, void* residual, uint32_t embd,
                        uint32_t n_active, uint32_t rows, void* stream) {
    if (rows == 0 || embd == 0) return 0;
    if (embd & 3u) return cudaErrorInvalidValue;
    const size_t total = (size_t)rows * (embd >> 2);
    // 256 threads stands. MEASURED (A4B c32, 2 interleaved reps):
    // narrowing to 64 threads takes this from 88 blocks (0.47 blocks/SM on a
    // 188-SM part) to 352, and bought +0.11% -- inside the control's own 0.23%
    // spread, i.e. nothing. The low wave count is real but it is not the
    // binding constraint: this kernel folds n_active partials per element, so
    // it is far closer to bandwidth-bound than occupancy-bound and extra
    // blocks add no bandwidth. Do not re-derive "0.47 blocks/SM => refill it".
    const uint32_t blocks = (uint32_t)((total + 255u) / 256u);
    pd_moe_slot_combine_kernel<<<blocks, 256, 0, (cudaStream_t)stream>>>(
        (const float4*)part, (float4*)residual, embd >> 2, n_active, rows);
    return pd_launch_status();
}

// Write-out twin of slot_combine (slot 485): residual[i] = sum (no read,
// no pre-zero). BITWISE the memset_zeros + pd_moe_slot_combine chain it
// replaces (0.0f + x == x exactly, fold order unchanged) - kills the
// ~8.8us/tick driver memset a gap attribution caught (282us/round idle
// attributed after down_mma2).
__global__ void pd_moe_slot_combine_init_kernel(
    const float4* __restrict__ part, float4* __restrict__ residual,
    uint32_t embd4, uint32_t n_active, uint32_t rows) {
    PD_PDL_ARM();  // consumer-safe for early PDL launches (2026-08-31)
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)rows * embd4) return;
    const size_t t = i / embd4, r = i % embd4;
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    for (uint32_t k = 0; k < n_active; ++k) {
        const float4 p = part[((size_t)t * n_active + k) * embd4 + r];
        acc.x += p.x; acc.y += p.y; acc.z += p.z; acc.w += p.w;
    }
    residual[i] = acc;
}

PD_EXPORT
int pd_moe_slot_combine_init(const void* part, void* residual, uint32_t embd,
                             uint32_t n_active, uint32_t rows, void* stream) {
    if (rows == 0 || embd == 0) return 0;
    if (embd & 3u) return cudaErrorInvalidValue;
    const size_t total = (size_t)rows * (embd >> 2);
    const uint32_t blocks = (uint32_t)((total + 255u) / 256u);
    pd_moe_slot_combine_init_kernel<<<blocks, 256, 0, (cudaStream_t)stream>>>(
        (const float4*)part, (float4*)residual, embd >> 2, n_active, rows);
    return pd_launch_status();
}

// bf16-partials twin of slot_combine (PADDOCK_MOE_PART_BF16): partials were
// rounded to bf16 at the down store; the fold still sums f32 in FIXED slot
// order. uint2 lanes = 4 bf16 per load.
__global__ void pd_moe_slot_combine_bf16_kernel(
    const uint2* __restrict__ part, float4* __restrict__ residual,
    uint32_t embd4, uint32_t n_active, uint32_t rows) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)rows * embd4) return;
    const size_t t = i / embd4, r = i % embd4;
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    for (uint32_t k = 0; k < n_active; ++k) {
        const uint2 pk = part[((size_t)t * n_active + k) * embd4 + r];
        const __nv_bfloat162 lo = *reinterpret_cast<const __nv_bfloat162*>(&pk.x);
        const __nv_bfloat162 hi = *reinterpret_cast<const __nv_bfloat162*>(&pk.y);
        acc.x += __bfloat162float(lo.x); acc.y += __bfloat162float(lo.y);
        acc.z += __bfloat162float(hi.x); acc.w += __bfloat162float(hi.y);
    }
    float4 d = residual[i];
    d.x += acc.x; d.y += acc.y; d.z += acc.z; d.w += acc.w;
    residual[i] = d;
}

PD_EXPORT
int pd_moe_slot_combine_bf16(const void* part, void* residual, uint32_t embd,
                             uint32_t n_active, uint32_t rows, void* stream) {
    if (rows == 0 || embd == 0) return 0;
    if (embd & 3u) return cudaErrorInvalidValue;
    const size_t total = (size_t)rows * (embd >> 2);
    const uint32_t blocks = (uint32_t)((total + 255u) / 256u);
    pd_moe_slot_combine_bf16_kernel<<<blocks, 256, 0, (cudaStream_t)stream>>>(
        (const uint2*)part, (float4*)residual, embd >> 2, n_active, rows);
    return pd_launch_status();
}

PD_EXPORT
int pd_mxfp4_moe_gate_up_mmq(const void* gate_data, const void* gate_scale,
                             const void* gate_bias, const void* up_data,
                             const void* up_scale, const void* up_bias,
                             const void* sorted_row, const void* block_expert,
                             const void* xq, const void* xs, void* fq, void* fs,
                             uint32_t in_dim, uint32_t ff, uint32_t max_blocks,
                             float alpha, float limit, float up_add, void* stream) {
    if (ff == 0 || max_blocks == 0) return 0;
    if ((in_dim & 31u) || (ff & 31u)) return cudaErrorInvalidValue;
    static cudaError_t attr = cudaFuncSetAttribute(
        (const void*)pd_mxfp4_moe_gate_up_mmq_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)PD_MOEQ_GU_SMEM);
    if (attr != cudaSuccess) return attr;
    dim3 grid(max_blocks, (ff + 127u) / 128u);
    pd_mxfp4_moe_gate_up_mmq_kernel<<<grid, 256, PD_MOEQ_GU_SMEM, (cudaStream_t)stream>>>(
        (const unsigned char*)gate_data, (const unsigned char*)gate_scale, (const float*)gate_bias,
        (const unsigned char*)up_data, (const unsigned char*)up_scale, (const float*)up_bias,
        (const unsigned int*)sorted_row, (const unsigned int*)block_expert,
        (const int8_t*)xq, (const float*)xs, (int8_t*)fq, (float*)fs, in_dim, ff, alpha, limit,
        up_add);
    return pd_launch_status();
}

// ------------------------------------------------------------ batched copy
// One launch for a batch of device-to-device copies (the radix prefix-cache
// insert/load issue hundreds of 32 KB page copies per prompt; as individual
// memcpys the host submit cost alone was ~10 ms per pp512 prefill). Each
// block owns one descriptor SLICE {src, dst, bytes}; bytes % 16 == 0 (KV
// pages). One-block-per-descriptor ran ~24 GB/s at the c16 wave's
// typical n~96 (24K threads on a 148-SM die, ~125us/launch = 4.1 ms/wave)
// - split each descriptor across `split` blocks so small batches still
// fill the die; disjoint 16B ranges, bitwise-identical output.
__global__ void pd_batched_copy_kernel(const ulonglong3* __restrict__ descs,
                                       uint32_t split) {
    const ulonglong3 d = descs[blockIdx.x / split];
    const uint4* __restrict__ src = (const uint4*)(uintptr_t)d.x;
    uint4* __restrict__ dst = (uint4*)(uintptr_t)d.y;
    const unsigned long long n16 = d.z >> 4;
    const uint32_t part = blockIdx.x % split;
    const unsigned long long lo = n16 * part / split;
    const unsigned long long hi = n16 * (part + 1u) / split;
    for (unsigned long long i = lo + threadIdx.x; i < hi; i += blockDim.x)
        dst[i] = src[i];
}

PD_EXPORT
int pd_batched_copy(const void* descs, uint32_t n, void* stream) {
    if (n == 0) return 0;
    uint32_t split = n >= 2048u ? 1u : (2048u + n - 1u) / n;
    if (split > 32u) split = 32u;
    pd_batched_copy_kernel<<<n * split, 256, 0, (cudaStream_t)stream>>>(
        (const ulonglong3*)descs, split);
    return pd_launch_status();
}

PD_EXPORT
int pd_mxfp4_moe_down_mmq(const void* down_data, const void* down_scale,
                          const void* down_bias, const void* sorted_row,
                          const void* sorted_slot, const void* block_expert,
                          const void* topk_w, const void* fq, const void* fs,
                          void* part, uint32_t ff, uint32_t embd,
                          uint32_t n_active, uint32_t max_blocks, void* stream) {
    if (embd == 0 || max_blocks == 0) return 0;
    if (ff & 31u) return cudaErrorInvalidValue;
    static cudaError_t attr = cudaFuncSetAttribute(
        (const void*)pd_mxfp4_moe_down_mmq_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)PD_MOEQ_DN_SMEM);
    if (attr != cudaSuccess) return attr;
    dim3 grid(max_blocks, (embd + 127u) / 128u);
    pd_mxfp4_moe_down_mmq_kernel<<<grid, 256, PD_MOEQ_DN_SMEM, (cudaStream_t)stream>>>(
        (const unsigned char*)down_data, (const unsigned char*)down_scale, (const float*)down_bias,
        (const unsigned int*)sorted_row, (const unsigned int*)sorted_slot,
        (const unsigned int*)block_expert, (const float*)topk_w,
        (const int8_t*)fq, (const float*)fs, (float*)part, ff, embd, n_active);
    return pd_launch_status();
}

