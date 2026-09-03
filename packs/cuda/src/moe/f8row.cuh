// moe/f8row.cuh - flat-scale (per-output-ROW) e4m3 expert GEMM for the
// gemma4-A4B routed experts on sm_89+/sm_120a. Textually-included segment of
// the single pack translation unit; needs moe/mmq.cuh (pd_cp_async16,
// pd_moeq_stage_y, PD_MMQ_XK) and gemm/int8_mma.cuh (PD_MMA_OK).
//
// Why (change A): our Q8_0 expert mma runs at 69-75% of DRAM peak with SM
// throughput at 45-49%, well short of what a fused MoE kernel of this class
// reaches on the same shapes (87-90% of peak at under 9% SM).
// The difference is not occupancy (change B reached 3 CTA/SM honestly and
// the cell got 0.86% slower) and it is not the tile shape.
// It is the SCALE FORMAT. Q8_0 carries an f16 scale per 32 weights, which has
// to be
//   (a) streamed from DRAM      -> 2 B per 32 B of data = 6.25% more bytes,
//   (b) staged into shared      -> pd_qmma_stage_ws, one extra pass per chunk,
//   (c) loaded + converted + multiplied inside the k loop
//       -> 2 LDS.U16 + 2 CVT + 4 I2F + 8 FMUL + 4 FADD per mma.
// A per-ROW scale is loop-invariant: it leaves the k walk entirely and lands
// once in the epilogue. int8 cannot carry a flat scale (no per-element
// exponent), so flat means e4m3 - and sm_89+ has the m16n8k32 e4m3 mma with
// an f32 accumulator, which the dense f8 lane (gemm/f8_lin.cuh) already
// drives, so the fragment/lane mapping is proven in-house.
//
// Deliberately not flat on the activation side. The B operand keeps Q8_0's
// per-32 f32 scale (same tile_y layout, pd_moeq_stage_y reused verbatim) for
// two reasons: activations are a rounding error of this kernel's DRAM traffic
// (32 tokens x 2816 against ~75 experts x 3.97M weights), so per-32 costs
// nothing in bytes; and it is 4 FFMA per mma against a 4096-MAC instruction,
// which is free. What per-32 buys is accuracy - a flat activation scale is
// the coarse half of vLLM's per-tensor scheme and there is no reason to
// inherit it when the fine half is free. The whole win here is on the WEIGHT
// side.
//
// PRECISION CLASS: e4m3 weights with a per-row f32 scale, requantized from
// the Q8_0 plane at load. Coarser than Q8_0's per-32 f16 in the mantissa
// (3 bits vs ~7) but finer in reach (4-bit exponent per element vs one shared
// block scale), so small weights inside a block gain what large ones lose.
// This is a lossy class and it is gated like every other one: greedy parity
// vs llama.cpp on the same GGUF before it can be anything but opt-in.

#if PD_MMA_OK && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
#define PD_F8R_MMA_OK 1
#else
#define PD_F8R_MMA_OK 0
#endif

// Weight tile row stride, int32: 64 data + 4 pad. The Q8 twin needs 76
// (64 data + 4 scale + 8 pad); with the scales gone the pad only has to keep
// the g-lane stride off the bank diagonal, and 68 % 32 == 4 does that - the
// 8 lanes of g land on banks 0,4,8..28 and t fills 0..3, so all 32 lanes of a
// fragment read hit distinct banks, same property PD_QMMA_WK bought with 12.
#define PD_F8R_WK 68u
#define PD_F8R_ROWS 64u
#define PD_F8R_W_INT32 (PD_F8R_ROWS * PD_F8R_WK)

// The weight plane comes from the EXISTING dense-lane converter
// pd_q8_0_to_f8row (gemm/dense_fp4_w8.cuh): Q8_0 -> e4m3 with one
// power-of-two f32 scale per output row. Power-of-two is the right pick here
// and not a compromise - for a floating-point target a pow2 scale shifts
// every element's exponent without touching its mantissa, so unlike an
// amax/448 scale it introduces no rounding of its own; the only cost is that
// the row's max lands in [224, 448] instead of exactly on 448, i.e. one
// binade of the underflow floor, which e4m3's 4-bit exponent has to spare.
// An expert plane is just rows: out_dim = n_expert * ff, in_dim = n_embd.

// ---- activations: e4m3 with a per-32 f32 scale ------------------------------
// Same plane shape as pd_quantize_q8 (data row-major, one f32 scale per 32
// elements at [row*n_blocks + b]) so pd_moeq_stage_y stages it unchanged -
// only the byte encoding differs. One warp per 32-block.
__global__ void pd_quantize_e4m3_b32f_kernel(const float* __restrict__ x,
                                             unsigned char* __restrict__ q,
                                             float* __restrict__ s, uint32_t n) {
    const uint32_t b = blockIdx.x * 8u + (threadIdx.x >> 5);
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t i = b * 32u + lane;
    if (i >= n) return;
    const float v = x[i];
    float a = fabsf(v);
    #pragma unroll
    for (uint32_t off = 16; off > 0; off >>= 1)
        a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, off));
    const float scl = a * (1.0f / 448.0f);
    const float inv = scl > 0.f ? 1.0f / scl : 0.f;
    q[i] = __nv_fp8_e4m3(v * inv).__x;
    if (lane == 0) s[b] = scl;
}

PD_EXPORT
int pd_quantize_e4m3_b32f(const void* x, void* q, void* scale, uint32_t n,
                          void* stream) {
    if (n == 0) return 0;
    if ((n & 31u) != 0) return cudaErrorInvalidValue;
    const uint32_t nb = n >> 5;
    pd_quantize_e4m3_b32f_kernel<<<(nb + 7u) / 8u, 256, 0, (cudaStream_t)stream>>>(
        (const float*)x, (unsigned char*)q, (float*)scale, n);
    return pd_launch_status();
}

// Issue one chunk's e4m3 weight data (64 rows x 8 k32-blocks x 32 B, two
// 16-byte cp.asyncs per (row, block)). Byte-for-byte the Q8 twin's staging -
// e4m3 and int8 are both one byte per element and the fragment word order is
// the same - minus the scale words the tile no longer carries.
__device__ __forceinline__ void pd_f8r_issue_w(
    int* __restrict__ tile, const unsigned char* __restrict__ data, size_t wrow0,
    uint32_t row_base, uint32_t out_dim, uint32_t in_dim, uint32_t kt, uint32_t tid) {
#if PD_F8R_MMA_OK
    const uint32_t n_blocks = in_dim >> 5;
    #pragma unroll
    for (uint32_t it = 0; it < 4u; ++it) {
        const uint32_t i = it * 256u + tid;
        const uint32_t row = i >> 4, half = i & 15u;
        const uint32_t b = half >> 1, h16 = half & 1u, gb = kt * 8u + b;
        const bool ok = gb < n_blocks && (row_base + row) < out_dim;
        pd_cp_async16(tile + row * PD_F8R_WK + b * 8u + h16 * 4u,
                      data + ((wrow0 + row) * (size_t)in_dim) + (ok ? gb : 0u) * 32u + h16 * 16u,
                      ok);
    }
#endif
}

// Stage the strip's per-row weight scales once per matrix (64 f32). These are
// the whole point: loop-invariant, so they never touch the k walk.
__device__ __forceinline__ void pd_f8r_stage_rs(
    float* __restrict__ dst, const float* __restrict__ rs, size_t wrow0,
    uint32_t row_base, uint32_t out_dim, uint32_t tid) {
    for (uint32_t i = tid; i < PD_F8R_ROWS; i += 256u)
        dst[i] = (row_base + i) < out_dim ? rs[wrow0 + i] : 0.f;
}

// Flat-scale twin of pd_q8_0_moe_gate_up_mma_kernel. Same sorted layout, same
// grid, same in-register geglu+quantize epilogue - the only differences are the
// weight plane's scale format and what that removes from the inner loop.
// GELU=true is the gemma4-A4B gelu_tanh(gate)*up; the SwiGLU arm is kept for
// the qwen shape.
// F8OUT picks what the epilogue hands the down GEMM: false = int8 per-32 (the
// Q8_0 down kernel's input, byte-for-byte the original epilogue), true = e4m3
// per-32 with the same f32 scale plane, which is what the flat-scale down
// kernel needs as its B operand. Same buffer sizes either way (1 B/elem), so
// the caller's fq/fs allocation does not change.
template <uint32_t BM, bool DB, bool GELU = false, bool F8OUT = false>
__global__ void __launch_bounds__(256, 2) pd_f8row_moe_gate_up_mma_kernel(
    const unsigned char* __restrict__ gate_data, const float* __restrict__ gate_rs,
    const unsigned char* __restrict__ up_data, const float* __restrict__ up_rs,
    const unsigned int* __restrict__ sorted_row, const unsigned int* __restrict__ block_expert,
    const unsigned char* __restrict__ xq, const float* __restrict__ xs,
    int8_t* __restrict__ fq, float* __restrict__ fs, uint32_t in_dim, uint32_t ff) {
#if PD_F8R_MMA_OK
    const uint32_t blk = blockIdx.x;                 // token block (fast axis: L2 strip reuse)
    const uint32_t e = block_expert[blk];
    if (e == PD_MOE_PAD) return;
    const uint32_t row_base = blockIdx.y * PD_F8R_ROWS;
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u, warp = tid >> 5;
    const uint32_t g = lane >> 2, t = lane & 3u;
    const uint32_t i0 = (warp >> 2) * 32u;           // 32-row group (0 or 32)
    const uint32_t joff = (warp & 3u) * 8u;          // 8-token column quarter
    const uint32_t nk = (in_dim + 255u) >> 8;

    constexpr uint32_t NH = BM / 32u;                // 32-token halves per block
    extern __shared__ int pd_f8rm_sh[];
    int* tile_y = pd_f8rm_sh;
    int* wbuf0 = pd_f8rm_sh + BM * PD_MMQ_XK;
    int* wbuf1 = DB ? (wbuf0 + PD_F8R_W_INT32) : wbuf0;   // SB aliases wbuf0
    __shared__ unsigned int tok[BM];
    __shared__ float wsr[2][PD_F8R_ROWS];            // [gate|up] row scales
    for (uint32_t i = tid; i < BM; i += 256u) tok[i] = sorted_row[(size_t)blk * BM + i];
    const size_t wrow0 = (size_t)e * ff + row_base;
    pd_f8r_stage_rs(wsr[0], gate_rs, wrow0, row_base, ff, tid);
    pd_f8r_stage_rs(wsr[1], up_rs, wrow0, row_base, ff, tid);
    __syncthreads();

    float acc_g[NH][2][4] = {}, acc_u[NH][2][4] = {};
    // One weight stream over 2*nk chunks, s = kt*2 + mat, so gate and up for the
    // same kt land next to each other and share a single activation stage. The
    // old form ran `mat` outermost and re-staged tile_y for every (mat,kt) pair
    // -- but the tile is identical for gate and up, so half of that staging was
    // pure waste (-17% of this kernel's global load traffic). Barrier discipline
    // below is unchanged from that version deliberately: wait_group then a sync
    // before anyone reads the buffer, one more sync once every warp is done with
    // it. `mat` is unrolled, so s&1 == mat and every buffer pick is compile-time.
    pd_f8r_issue_w(wbuf0, gate_data, wrow0, row_base, ff, in_dim, 0, tid);
    asm volatile("cp.async.commit_group;");
    for (uint32_t kt = 0; kt < nk; ++kt) {
        #pragma unroll
        for (uint32_t mat = 0; mat < 2u; ++mat) {
            const uint32_t s = kt * 2u + mat;         // position in the weight stream
            int* tw = (DB && mat) ? wbuf1 : wbuf0;
            // the chunk after this one: mat flips, kt advances only after `up`
            const unsigned char* nd = mat ? gate_data : up_data;
            const uint32_t nkt = mat ? kt + 1u : kt;
            asm volatile("cp.async.wait_group 0;");
            if (mat == 0u)
                pd_moeq_stage_y<BM>(tile_y, (const int*)xq, xs, tok, in_dim, kt, tid);
            __syncthreads();
            // DB: prefetch s+1 into the other buffer now (overlaps this mma).
            // SB: the single buffer is still being read below -- defer to after.
            if (DB && s + 1u < 2u * nk) {
                pd_f8r_issue_w(mat ? wbuf0 : wbuf1, nd, wrow0, row_base, ff,
                               in_dim, nkt, tid);
                asm volatile("cp.async.commit_group;");
            }

            #pragma unroll
            for (uint32_t th = 0; th < NH; ++th) {
                const uint32_t jb = th * 32u + joff;       // this half's token base
                float (*acc)[4] = mat ? acc_u[th] : acc_g[th];
                #pragma unroll
                for (uint32_t h = 0; h < 2u; ++h) {
                    const uint32_t k00 = h * 32u;
                    #pragma unroll
                    for (uint32_t kk = 0; kk < 4u; ++kk) {
                        const uint32_t bb = (k00 >> 3) + kk;
                        const uint32_t ko = k00 + kk * 8u;
                        const int b0 = tile_y[(jb + g) * PD_MMQ_XK + ko + t];
                        const int b1 = tile_y[(jb + g) * PD_MMQ_XK + ko + 4u + t];
                        const float dB0 =
                            ((const float*)tile_y)[(jb + 2u * t) * PD_MMQ_XK + 64u + bb];
                        const float dB1 =
                            ((const float*)tile_y)[(jb + 2u * t + 1u) * PD_MMQ_XK + 64u + bb];
                        #pragma unroll
                        for (uint32_t n = 0; n < 2u; ++n) {
                            const uint32_t r0 = (i0 + n * 16u + g) * PD_F8R_WK;
                            const uint32_t r8 = (i0 + n * 16u + 8u + g) * PD_F8R_WK;
                            // Same A layout as the s8 twin: k-half0 = elems
                            // {4t..4t+3} (word t of the block), k-half1 =
                            // {16+4t..} (word 4+t). The 8-bit fragment map is
                            // the type-independent one.
                            const int A0 = tw[r0 + bb * 8u + t];
                            const int A2 = tw[r0 + bb * 8u + 4u + t];
                            const int A1 = tw[r8 + bb * 8u + t];
                            const int A3 = tw[r8 + bb * 8u + 4u + t];
                            float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
                            asm("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                                : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                                : "r"(A0), "r"(A1), "r"(A2), "r"(A3), "r"(b0), "r"(b1));
                            // only the per-32 ACTIVATION scale rides the loop;
                            // the weight row scale lands in the epilogue
                            acc[n][0] += dB0 * d0;
                            acc[n][1] += dB1 * d1;
                            acc[n][2] += dB0 * d2;
                            acc[n][3] += dB1 * d3;
                        }
                    }
                }
            }
            __syncthreads();  // tile_y + the buffers are rewritten next step
            // SB: buffer now free to reload; issue s+1 (no compute overlap).
            if (!DB && s + 1u < 2u * nk) {
                pd_f8r_issue_w(wbuf0, nd, wrow0, row_base, ff, in_dim, nkt, tid);
                asm volatile("cp.async.commit_group;");
            }
        }
    }

    // Identical epilogue to the s8 twin (in-register geglu + per-32 int8
    // quantize straight into the down GEMM's fq/fs) with the weight row scale
    // folded in here - acc holds sum(qA*qB)*scaleB, so one multiply by
    // scaleA[row] completes the dot.
    const uint32_t n_sb = ff >> 5;
    #pragma unroll
    for (uint32_t th = 0; th < NH; ++th) {
        const uint32_t jb = th * 32u + joff;
        #pragma unroll
        for (uint32_t qc = 0; qc < 2u; ++qc) {
            const uint32_t c = jb + 2u * t + qc;
            const bool pad = tok[c] == PD_MOE_PAD;
            const uint32_t rb = row_base + i0;
            float sw[4];
            #pragma unroll
            for (uint32_t n = 0; n < 2u; ++n) {
                #pragma unroll
                for (uint32_t hq = 0; hq < 2u; ++hq) {
                    const uint32_t q = qc + 2u * hq;
                    const uint32_t lr = i0 + n * 16u + hq * 8u + g;  // row in strip
                    const uint32_t r = row_base + lr;
                    float out = 0.f;
                    if (!pad && r < ff) {
                        const float gv = acc_g[th][n][q] * wsr[0][lr];
                        const float uv = acc_u[th][n][q] * wsr[1][lr];
                        out = GELU
                            ? 0.5f * gv
                                  * (1.0f
                                     + tanhf(0.79788456080286535587989211986876f * gv
                                             * (1.0f + 0.044715f * gv * gv)))
                                  * uv
                            : (gv / (1.0f + __expf(-gv))) * uv;
                    }
                    sw[n * 2u + hq] = out;
                }
            }
            float a = fmaxf(fmaxf(fabsf(sw[0]), fabsf(sw[1])), fmaxf(fabsf(sw[2]), fabsf(sw[3])));
            #pragma unroll
            for (uint32_t o = 4; o <= 16u; o <<= 1)
                a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
            const float scl = a * (F8OUT ? (1.0f / 448.0f) : (1.0f / 127.0f));
            const float invs = scl > 0.f ? 1.0f / scl : 0.f;
            const size_t row = (size_t)blk * BM + c;
            if (rb < ff) {
                #pragma unroll
                for (uint32_t v = 0; v < 4u; ++v) {
                    const uint32_t r = rb + (v >> 1) * 16u + (v & 1u) * 8u + g;
                    if (F8OUT) {
                        // PAD rows carry sw=0 and scl=0, so this stores a
                        // clean +0.0 byte - the down K-guard's zero-fill
                        // relies on 0 x anything accumulating exactly 0.
                        ((unsigned char*)fq)[row * ff + r] =
                            __nv_fp8_e4m3(sw[v] * invs).__x;
                    } else {
                        int qi = __float2int_rn(sw[v] * invs);
                        qi = qi < -127 ? -127 : (qi > 127 ? 127 : qi);
                        fq[row * ff + r] = (int8_t)qi;
                    }
                }
                if (g == 0) fs[row * n_sb + (rb >> 5)] = scl;
            }
        }
    }
#else
    (void)gate_data; (void)gate_rs; (void)up_data; (void)up_rs; (void)sorted_row;
    (void)block_expert; (void)xq; (void)xs; (void)fq; (void)fs; (void)in_dim; (void)ff;
#endif
}

// smem: activation tile (unchanged 76-int32 rows - it still carries the
// per-32 B scales) + 1 or 2 scale-free weight buffers.
// (32,true) = 9728 + 34816 = 44,544 B; (64,false) = 19,456 + 17,408 = 36,864 B.
// Both stay under the ~51 KB 2-CTA/SM line, same tier as the Q8 twin.
template <uint32_t BM, bool DB>
static constexpr uint32_t pd_f8r_smem() {
    return (BM * PD_MMQ_XK + (DB ? 2u : 1u) * PD_F8R_W_INT32) * 4u;
}

template <uint32_t BM, bool DB, bool GELU = false, bool F8OUT = false>
static int pd_launch_f8r_gu(const unsigned char* gd, const float* grs,
                            const unsigned char* ud, const float* urs,
                            const unsigned int* sr, const unsigned int* be,
                            const unsigned char* xq, const float* xs, int8_t* fq,
                            float* fs, uint32_t in_dim, uint32_t ff,
                            uint32_t max_blocks, cudaStream_t stream) {
    constexpr uint32_t smem = pd_f8r_smem<BM, DB>();
    static bool attr = false;   // per-instantiation (template statics)
    if (!attr) {
        cudaFuncSetAttribute(
            (const void*)pd_f8row_moe_gate_up_mma_kernel<BM, DB, GELU, F8OUT>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        attr = true;
    }
    dim3 grid(max_blocks, (ff + PD_F8R_ROWS - 1u) / PD_F8R_ROWS);
    pd_f8row_moe_gate_up_mma_kernel<BM, DB, GELU, F8OUT><<<grid, 256, smem, stream>>>(
        gd, grs, ud, urs, sr, be, xq, xs, fq, fs, in_dim, ff);
    return pd_launch_status();
}

// GEGLU (gemma4-A4B) launcher. Same bm contract as the Q8 twin: 64 -> wider
// prefill block (single-buffered weights), else the 32-token serving default
// (double-buffered). The sorted layout and fq/fs sizing at the call site must
// match bm.
PD_EXPORT
int pd_f8row_moe_gate_up_mma_geglu(const void* gate_data, const void* gate_rs,
                                   const void* up_data, const void* up_rs,
                                   const void* sorted_row, const void* block_expert,
                                   const void* xq, const void* xs, void* fq, void* fs,
                                   uint32_t in_dim, uint32_t ff, uint32_t max_blocks,
                                   uint32_t bm, void* stream) {
    if (ff == 0 || max_blocks == 0) return 0;
    if ((in_dim & 255u) != 0 || (ff & 31u) != 0) return cudaErrorInvalidValue;
    const unsigned char* gd = (const unsigned char*)gate_data;
    const unsigned char* ud = (const unsigned char*)up_data;
    const float* grs = (const float*)gate_rs; const float* urs = (const float*)up_rs;
    const unsigned int* sr = (const unsigned int*)sorted_row;
    const unsigned int* be = (const unsigned int*)block_expert;
    const unsigned char* xqp = (const unsigned char*)xq; const float* xsp = (const float*)xs;
    int8_t* fqp = (int8_t*)fq; float* fsp = (float*)fs;
    cudaStream_t st = (cudaStream_t)stream;
    if (bm >= 64u)
        return pd_launch_f8r_gu<64u, false, true>(gd, grs, ud, urs, sr, be, xqp, xsp,
                                                  fqp, fsp, in_dim, ff, max_blocks, st);
    return pd_launch_f8r_gu<32u, true, true>(gd, grs, ud, urs, sr, be, xqp, xsp, fqp,
                                             fsp, in_dim, ff, max_blocks, st);
}

// e4m3-output twin of the launcher above: identical GEMM, the epilogue hands
// the down half e4m3 per-32 instead of int8 per-32. Pairs with
// pd_f8row_moe_down_mma; the int8 export above pairs with the Q8_0 down.
PD_EXPORT
int pd_f8row_moe_gate_up_mma_geglu_f8(const void* gate_data, const void* gate_rs,
                                      const void* up_data, const void* up_rs,
                                      const void* sorted_row, const void* block_expert,
                                      const void* xq, const void* xs, void* fq, void* fs,
                                      uint32_t in_dim, uint32_t ff, uint32_t max_blocks,
                                      uint32_t bm, void* stream) {
    if (ff == 0 || max_blocks == 0) return 0;
    if ((in_dim & 255u) != 0 || (ff & 31u) != 0) return cudaErrorInvalidValue;
    const unsigned char* gd = (const unsigned char*)gate_data;
    const unsigned char* ud = (const unsigned char*)up_data;
    const float* grs = (const float*)gate_rs; const float* urs = (const float*)up_rs;
    const unsigned int* sr = (const unsigned int*)sorted_row;
    const unsigned int* be = (const unsigned int*)block_expert;
    const unsigned char* xqp = (const unsigned char*)xq; const float* xsp = (const float*)xs;
    int8_t* fqp = (int8_t*)fq; float* fsp = (float*)fs;
    cudaStream_t st = (cudaStream_t)stream;
    if (bm >= 64u)
        return pd_launch_f8r_gu<64u, false, true, true>(gd, grs, ud, urs, sr, be, xqp,
                                                        xsp, fqp, fsp, in_dim, ff,
                                                        max_blocks, st);
    return pd_launch_f8r_gu<32u, true, true, true>(gd, grs, ud, urs, sr, be, xqp, xsp,
                                                   fqp, fsp, in_dim, ff, max_blocks, st);
}

// Down half, flat-scale: same tile shape over K = ff, activation rows are the
// sorted fused rows (indexed directly, no gather), weights are e4m3 with one
// power-of-two f32 scale per OUTPUT row (n_expert * embd rows). Epilogue is the
// Q8_0 twin's, unchanged: one writer per (token, slot, r), plain stores into
// the partials buffer, pd_moe_slot_combine folds in fixed slot order.
//
// This half is where the scale format should matter most and where it is also
// least likely to be the whole story: the Q8_0 down measured 58.0/60.2% of
// DRAM peak against gate_up's 78.6%, because its K walk is only nk=3 chunks
// (K = ff = 704) against gate_up's 11 - the per-block prelude and the BM x K
// activation stage amortize over a third as much weight traffic. The scale
// format takes 5.9% of the bytes out; the prelude/staging structure is a
// separate change and gets measured separately.
template <uint32_t BM, bool DB>
__global__ void __launch_bounds__(256, 2) pd_f8row_moe_down_mma_kernel(
    const unsigned char* __restrict__ down_data, const float* __restrict__ down_rs,
    const unsigned int* __restrict__ sorted_row, const unsigned int* __restrict__ sorted_slot,
    const unsigned int* __restrict__ block_expert, const float* __restrict__ topk_w,
    const unsigned char* __restrict__ fq, const float* __restrict__ fs,
    float* __restrict__ part, uint32_t ff, uint32_t embd, uint32_t n_active) {
#if PD_F8R_MMA_OK
    const uint32_t blk = blockIdx.x;
    const uint32_t e = block_expert[blk];
    if (e == PD_MOE_PAD) return;
    const uint32_t row_base = blockIdx.y * PD_F8R_ROWS;
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u, warp = tid >> 5;
    const uint32_t g = lane >> 2, t = lane & 3u;
    const uint32_t i0 = (warp >> 2) * 32u;
    const uint32_t joff = (warp & 3u) * 8u;
    const uint32_t nk = (ff + 255u) >> 8;

    constexpr uint32_t NH = BM / 32u;
    extern __shared__ int pd_f8rm_sh[];
    int* tile_y = pd_f8rm_sh;
    int* wbuf0 = pd_f8rm_sh + BM * PD_MMQ_XK;
    int* wbuf1 = DB ? (wbuf0 + PD_F8R_W_INT32) : wbuf0;
    __shared__ unsigned int tok[BM], slt[BM], idn[BM];
    __shared__ float wsr[PD_F8R_ROWS];
    for (uint32_t i = tid; i < BM; i += 256u) {
        tok[i] = sorted_row[(size_t)blk * BM + i];
        slt[i] = sorted_slot[(size_t)blk * BM + i];
        idn[i] = blk * BM + i;  // fq rows are sorted-contiguous
    }
    const size_t wrow0 = (size_t)e * embd + row_base;
    pd_f8r_stage_rs(wsr, down_rs, wrow0, row_base, embd, tid);
    __syncthreads();

    float acc[NH][2][4] = {};
    pd_f8r_issue_w(wbuf0, down_data, wrow0, row_base, embd, ff, 0, tid);
    asm volatile("cp.async.commit_group;");
    for (uint32_t kt = 0; kt < nk; ++kt) {
        int* tw = (DB && (kt & 1u)) ? wbuf1 : wbuf0;
        asm volatile("cp.async.wait_group 0;");
        pd_moeq_stage_y<BM>(tile_y, (const int*)fq, fs, idn, ff, kt, tid);
        __syncthreads();
        if (DB && kt + 1u < nk) {
            pd_f8r_issue_w((kt & 1u) ? wbuf0 : wbuf1, down_data, wrow0, row_base, embd,
                           ff, kt + 1u, tid);
            asm volatile("cp.async.commit_group;");
        }
        #pragma unroll
        for (uint32_t th = 0; th < NH; ++th) {
            const uint32_t jb = th * 32u + joff;
            #pragma unroll
            for (uint32_t h = 0; h < 2u; ++h) {
                const uint32_t k00 = h * 32u;
                #pragma unroll
                for (uint32_t kk = 0; kk < 4u; ++kk) {
                    const uint32_t bb = (k00 >> 3) + kk;
                    const uint32_t ko = k00 + kk * 8u;
                    const int b0 = tile_y[(jb + g) * PD_MMQ_XK + ko + t];
                    const int b1 = tile_y[(jb + g) * PD_MMQ_XK + ko + 4u + t];
                    const float dB0 =
                        ((const float*)tile_y)[(jb + 2u * t) * PD_MMQ_XK + 64u + bb];
                    const float dB1 =
                        ((const float*)tile_y)[(jb + 2u * t + 1u) * PD_MMQ_XK + 64u + bb];
                    #pragma unroll
                    for (uint32_t n = 0; n < 2u; ++n) {
                        const uint32_t r0 = (i0 + n * 16u + g) * PD_F8R_WK;
                        const uint32_t r8 = (i0 + n * 16u + 8u + g) * PD_F8R_WK;
                        const int A0 = tw[r0 + bb * 8u + t];
                        const int A2 = tw[r0 + bb * 8u + 4u + t];
                        const int A1 = tw[r8 + bb * 8u + t];
                        const int A3 = tw[r8 + bb * 8u + 4u + t];
                        float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
                        asm("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                            : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                            : "r"(A0), "r"(A1), "r"(A2), "r"(A3), "r"(b0), "r"(b1));
                        acc[th][n][0] += dB0 * d0;
                        acc[th][n][1] += dB1 * d1;
                        acc[th][n][2] += dB0 * d2;
                        acc[th][n][3] += dB1 * d3;
                    }
                }
            }
        }
        __syncthreads();
        if (!DB && kt + 1u < nk) {
            pd_f8r_issue_w(wbuf0, down_data, wrow0, row_base, embd, ff, kt + 1u, tid);
            asm volatile("cp.async.commit_group;");
        }
    }

    #pragma unroll
    for (uint32_t th = 0; th < NH; ++th) {
        const uint32_t c0 = th * 32u + joff + 2u * t;
        #pragma unroll
        for (uint32_t n = 0; n < 2u; ++n) {
            const uint32_t l0 = i0 + n * 16u + g, l8 = l0 + 8u;   // rows in strip
            #pragma unroll
            for (uint32_t q = 0; q < 4u; ++q) {
                const uint32_t lr = (q & 2u) ? l8 : l0;
                const uint32_t r = row_base + lr;
                const uint32_t c = c0 + (q & 1u);
                const unsigned int token = tok[c];
                if (r >= embd || token == PD_MOE_PAD) continue;
                const float w = topk_w[(size_t)token * n_active + slt[c]];
                part[((size_t)token * n_active + slt[c]) * embd + r] =
                    w * acc[th][n][q] * wsr[lr];
            }
        }
    }
#else
    (void)down_data; (void)down_rs; (void)sorted_row; (void)sorted_slot;
    (void)block_expert; (void)topk_w; (void)fq; (void)fs; (void)part;
    (void)ff; (void)embd; (void)n_active;
#endif
}

template <uint32_t BM, bool DB>
static int pd_launch_f8r_dn(const unsigned char* dd, const float* drs,
                            const unsigned int* sr, const unsigned int* sl,
                            const unsigned int* be, const float* tw,
                            const unsigned char* fq, const float* fs, float* part,
                            uint32_t ff, uint32_t embd, uint32_t n_active,
                            uint32_t max_blocks, cudaStream_t stream) {
    constexpr uint32_t smem = pd_f8r_smem<BM, DB>();
    static bool attr = false;
    if (!attr) {
        cudaFuncSetAttribute((const void*)pd_f8row_moe_down_mma_kernel<BM, DB>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        attr = true;
    }
    dim3 grid(max_blocks, (embd + PD_F8R_ROWS - 1u) / PD_F8R_ROWS);
    pd_f8row_moe_down_mma_kernel<BM, DB><<<grid, 256, smem, stream>>>(
        dd, drs, sr, sl, be, tw, fq, fs, part, ff, embd, n_active);
    return pd_launch_status();
}

PD_EXPORT
int pd_f8row_moe_down_mma(const void* down_data, const void* down_rs,
                          const void* sorted_row, const void* sorted_slot,
                          const void* block_expert, const void* topk_w, const void* fq,
                          const void* fs, void* part, uint32_t ff, uint32_t embd,
                          uint32_t n_active, uint32_t max_blocks, uint32_t bm,
                          void* stream) {
    if (embd == 0 || max_blocks == 0) return 0;
    // K = ff only needs 32-granularity: the K walk is fully guarded (issue_w
    // zero-fills past n_blocks, stage_y zero-fills data AND scales), and an
    // e4m3 zero byte is +0.0, so 0 x anything accumulates exactly 0. A4B's
    // ff_exp = 704 is the ragged-K consumer (22 blocks over 24 slots).
    if ((ff & 31u) != 0 || (embd & 31u) != 0) return cudaErrorInvalidValue;
    const unsigned char* dd = (const unsigned char*)down_data;
    const float* drs = (const float*)down_rs;
    const unsigned int* sr = (const unsigned int*)sorted_row;
    const unsigned int* sl = (const unsigned int*)sorted_slot;
    const unsigned int* be = (const unsigned int*)block_expert;
    const float* tw = (const float*)topk_w;
    const unsigned char* fqp = (const unsigned char*)fq;
    const float* fsp = (const float*)fs; float* pp = (float*)part;
    cudaStream_t st = (cudaStream_t)stream;
    if (bm >= 64u)
        return pd_launch_f8r_dn<64u, false>(dd, drs, sr, sl, be, tw, fqp, fsp, pp, ff,
                                            embd, n_active, max_blocks, st);
    return pd_launch_f8r_dn<32u, true>(dd, drs, sr, sl, be, tw, fqp, fsp, pp, ff, embd,
                                       n_active, max_blocks, st);
}
