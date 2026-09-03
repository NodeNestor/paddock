// fmha16: trtllm-gen-class tensor-core decode attention for the muse-glimmer
// geometry (hd128, G=16, fp8-e4m3 paged KV).
//
// WHY. Attributing a c32 tick by DIE-TIME (duration x min(1, CTAs/148)) put
// muse's whole decode deficit in attention - the GEMM band and the packing
// were already fine, so attention was the entire cell.
//
// HOW. NVIDIA's own kernel name states the formulation:
//   fmhaSm100fKernel_...H128PagedKvSlidingOrChunkedCausalP16MultiCtasKvCga
//     VarSeqQ16Kv128StaticSwapsAbForGen
// **Q16 x Kv128** + **SwapsAbForGen**: at G=16 the sixteen q-heads sharing a
// kv-head are an M=16 MMA dimension, so decode attention is two ordinary
// m16n8k16 GEMMs rather than a gemv with 16-fold redundant KV traffic plus an
// f32 partial round-trip and a second full-die combine kernel:
//     S[16 heads][t tokens] = Q[16][128 dim] @ K[t][128 dim]^T
//     O[16 heads][128 dim] += P[16][t]       @ V[t][128 dim]
// Q and K are already [outer][k] row-major so ldmatrix feeds them directly; V
// is needed as [dim][token] and rides `ldmatrix .trans` (staging a V^T buffer
// instead cost 8-way bank conflicts per store - measured 22.30 -> 18.39 us).
//
// vs the SHIPPED pair (pd_attn_decode_vec8_paged splits=2 + combine), us/layer,
// NWARP=16 / 512 thr:
//     B  ctx |   prod |  this | speedup
//     1  128 |  13.60 |  7.69 | 1.77x
//     8  256 |  22.59 | 10.25 | 2.20x
//    16  256 |  35.75 | 10.90 | 3.28x
//    32  128 |  34.36 |  8.21 | 4.19x
//    32  256 |  64.74 | 12.11 | 5.35x
//    32  512 | 124.39 | 18.57 | 6.70x
// At the live c32 point ctx ramps 128->256, so this runs 8.2-12.1 us.
// It wins at every RUNG INCLUDING B=1, so unlike
// pd_attn_decode_fused_gqa16 (which lost below B=24) it needs no row gate.
//
// Two STRUCTURAL NOTES.
//  - CONSTANT SMEM, so no CONTEXT BAND GATE. The KV walk is chunked KVT=128
//    deep, so smem does not grow with context - unlike fused_gqa16, which
//    staged the whole window and had to be gated to pos_max <= 768 by the smem
//    cap. This arm is valid at any context.
//  - More WARPS, not more CTAs. The grid is n_kv_heads*batch = 64 CTAs at c32
//    (43% of 148 SMs), which looks like the binding constraint and is not: a
//    KV split across CTAs cost a FLAT ~12 us (partial round-trip +
//    __threadfence + a single merging CTA is a serial tail). w4 -> w16 is
//    18.34 -> 12.11 at B=32/ctx256. Hence NWARP=16 and no split path here.
//
// NUMERICS. K and V are exact (e4m3 -> bf16 is lossless); only Q and P round.
// NXQ/NXP=2 carry each as bf16 big + bf16 residual (one extra MMA per plane):
//     q1p1 12.14us relRMS 1.7e-3 | q2p1 12.30 1.5e-3
//     q1p2 12.29    7.7e-4       | q2p2 12.44 2.5e-6   <- shipped
// The single splits barely move because the errors are INDEPENDENT and
// comparable (eP~1.5e-3, eQ~7.7e-4, sqrt of squares = 1.7e-3 = q1p1); both
// must be fixed. +2.5% time for 680x accuracy, and 2.5e-6 is near the f32
// f32 reference's 2.4e-7 and ~5600x tighter than the PADDOCK_LAGD_F16 class.
//
// FINAL-output contract, exactly like fused_gqa16: sinks folded in-kernel, no
// partials, the caller skips the combine entirely.

// padded leading dim: 136 elems = 272 B rows, 16 B aligned as ldmatrix wants
static constexpr uint32_t PD_FMHA16_LD = 136;

#if PD_BF16MMA_OK
// B-operand load with transpose (V is [token][dim], the second GEMM wants
// [dim][token]). x2 addressing: lanes 0-15 supply row (lane&15) of the 16-token
// k-window; tile0 -> b0 (k 0..7), tile1 -> b1 (k 8..15).
__device__ __forceinline__ void pd_bf16m_ldm_x2_trans(const void* p, uint32_t& r0,
                                                      uint32_t& r1) {
    const unsigned a = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
                 : "=r"(r0), "=r"(r1)
                 : "r"(a));
}
#endif

template <uint32_t KVT, uint32_t NWARP, uint32_t NXQ = 1u, uint32_t NXP = 1u>
__global__ void __launch_bounds__(32 * NWARP, 1) pd_attn_fmha16_kernel(
    const float* __restrict__ q, const __nv_fp8_e4m3* __restrict__ pool_k,
    const __nv_fp8_e4m3* __restrict__ pool_v, float* __restrict__ out,
    const unsigned int* __restrict__ positions,
    const unsigned int* __restrict__ slots,
    const uint32_t* __restrict__ block_tables, uint32_t blocks_per_slot,
    uint32_t n_heads, uint32_t kv_dim, uint32_t swa_window, float scale,
    const float* __restrict__ sinks) {
#if PD_BF16MMA_OK
    PD_PDL_ARM();
    // fixed by the launcher's shape gate: hd128, GQA-16, 16-token paged blocks
    constexpr uint32_t HD = 128u, G = 16u, BLK = 16u;
    const uint32_t kvh = blockIdx.x, b = blockIdx.y;
    const uint32_t slot = slots ? slots[b] : b;
    const uint32_t tid = threadIdx.x, warp = tid >> 5, lane = tid & 31u;
    // ldmatrix address map, transcribed from gemm/bf16_dense.cuh
    const uint32_t l7 = lane & 7u;
    const uint32_t a_roff = ((lane & 8u) ? 8u : 0u) + l7;
    const uint32_t a_kof = (lane & 16u) ? 8u : 0u;
    const uint32_t b_kof = (lane & 8u) ? 8u : 0u;
    // m16n8k16 D-fragment map: d0=(m=gg,n=2t) d1=(m=gg,2t+1)
    //                          d2=(m=gg+8,2t) d3=(m=gg+8,2t+1)
    const uint32_t gg = lane >> 2, t4 = lane & 3u;
    // n8 groups per warp: KVT/8 token groups (and HD/8 dim groups) shared out
    constexpr uint32_t NJG = 16u / NWARP;
    const uint32_t wbase = warp * (NJG * 8u);

    const uint32_t pos = positions[b];
    const uint32_t first_pos =
        (swa_window > 0 && pos + 1 > swa_window) ? (pos + 1 - swa_window) : 0;
    const uint32_t n_pos = pos + 1 - first_pos;

    extern __shared__ char smem_raw[];
    __nv_bfloat16* sm_q = (__nv_bfloat16*)smem_raw;      // [NXQ][16][PD_FMHA16_LD]
    __nv_bfloat16* sm_k = sm_q + NXQ * 16u * PD_FMHA16_LD;         // [KVT][PD_FMHA16_LD]
    __nv_bfloat16* sm_v = sm_k + KVT * PD_FMHA16_LD;               // [KVT][PD_FMHA16_LD] (not V^T)
    __nv_bfloat16* sm_p = sm_v + KVT * PD_FMHA16_LD;               // [NXP][16][PD_FMHA16_LD]
    float* sm_m = (float*)(sm_p + NXP * 16u * PD_FMHA16_LD);       // [16] running max
    float* sm_l = sm_m + 16;                             // [16] running sum
    float* sm_c = sm_l + 16;                             // [16] rescale corr
    float* sm_wm = sm_c + 16;                            // [NWARP][16]
    float* sm_wl = sm_wm + NWARP * 16;                   // [NWARP][16]

    for (uint32_t e = tid; e < 16u * HD; e += blockDim.x) {
        const uint32_t r = e / HD, d = e - r * HD;
        const float qv = q[((size_t)b * n_heads + kvh * G + r) * HD + d] * scale;
        const __nv_bfloat16 qb = __float2bfloat16(qv);
        sm_q[r * PD_FMHA16_LD + d] = qb;
        if (NXQ > 1u)   // big + residual: ~22-bit q, lagd's NXQ=2 class
            sm_q[16u * PD_FMHA16_LD + r * PD_FMHA16_LD + d] = __float2bfloat16(qv - __bfloat162float(qb));
    }
    if (tid < 16u) { sm_m[tid] = -INFINITY; sm_l[tid] = 0.f; }

    float o[NJG][4];
#pragma unroll
    for (uint32_t i = 0; i < NJG; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) o[i][j] = 0.f;

    const uint32_t* bt = block_tables + (size_t)slot * blocks_per_slot;
    __syncthreads();

    for (uint32_t t0 = 0; t0 < n_pos; t0 += KVT) {
        const uint32_t nt = min(KVT, n_pos - t0);
        // stage K and V both as [nt][PD_FMHA16_LD], fp8 -> bf16 (V is transposed at
        // ldmatrix time, not here)
        for (uint32_t c = tid; c < nt * (HD / 16u); c += blockDim.x) {
            const uint32_t tk = c / (HD / 16u), off = (c % (HD / 16u)) * 16u;
            const uint32_t tok = first_pos + t0 + tk;
            const size_t row = (size_t)bt[tok >> 4] * BLK + (tok & 15u);
            const size_t src = row * kv_dim + kvh * HD + off;
            const uint4 kk = *(const uint4*)(pool_k + src);
            const uint4 vv = *(const uint4*)(pool_v + src);
            const __nv_fp8_e4m3* kp = (const __nv_fp8_e4m3*)&kk;
            const __nv_fp8_e4m3* vp = (const __nv_fp8_e4m3*)&vv;
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                sm_k[tk * PD_FMHA16_LD + off + i] = __float2bfloat16(float(kp[i]));
                sm_v[tk * PD_FMHA16_LD + off + i] = __float2bfloat16(float(vp[i]));
            }
        }
        __syncthreads();

        // ── S = Q @ K^T. warp owns tokens [warp*32, +32) as 4 n8 groups
        float s[NJG][4];
#pragma unroll
        for (uint32_t jg = 0; jg < NJG; ++jg)
            s[jg][0] = s[jg][1] = s[jg][2] = s[jg][3] = 0.f;
#pragma unroll
        for (uint32_t k0 = 0; k0 < HD; k0 += 16u) {
            uint32_t af[NXQ][4];
#pragma unroll
            for (uint32_t xq = 0; xq < NXQ; ++xq)
                pd_bf16m_ldm_x4(&sm_q[xq * 16u * PD_FMHA16_LD + a_roff * PD_FMHA16_LD + k0 + a_kof],
                                af[xq][0], af[xq][1], af[xq][2], af[xq][3]);
#pragma unroll
            for (uint32_t jg = 0; jg < NJG; ++jg) {
                uint32_t bf[2];
                pd_bf16m_ldm_x2(&sm_k[(wbase + jg * 8u + l7) * PD_FMHA16_LD + k0 + b_kof],
                                bf[0], bf[1]);
#pragma unroll
                for (uint32_t xq = 0; xq < NXQ; ++xq) pd_bf16m_mma(s[jg], af[xq], bf);
            }
        }
        // mask the tail of the last chunk (sm_k rows >= nt hold stale data)
#pragma unroll
        for (uint32_t jg = 0; jg < NJG; ++jg) {
            const uint32_t c0 = wbase + jg * 8u + 2u * t4;
            if (c0 >= nt) { s[jg][0] = -INFINITY; s[jg][2] = -INFINITY; }
            if (c0 + 1 >= nt) { s[jg][1] = -INFINITY; s[jg][3] = -INFINITY; }
        }

        // ── row max: reduce over the 4 n8 groups, then the 4 lanes sharing gg
        float mx_lo = -INFINITY, mx_hi = -INFINITY;
#pragma unroll
        for (uint32_t jg = 0; jg < NJG; ++jg) {
            mx_lo = fmaxf(mx_lo, fmaxf(s[jg][0], s[jg][1]));
            mx_hi = fmaxf(mx_hi, fmaxf(s[jg][2], s[jg][3]));
        }
        mx_lo = fmaxf(mx_lo, __shfl_xor_sync(0xffffffffu, mx_lo, 1));
        mx_lo = fmaxf(mx_lo, __shfl_xor_sync(0xffffffffu, mx_lo, 2));
        mx_hi = fmaxf(mx_hi, __shfl_xor_sync(0xffffffffu, mx_hi, 1));
        mx_hi = fmaxf(mx_hi, __shfl_xor_sync(0xffffffffu, mx_hi, 2));
        if (t4 == 0) { sm_wm[warp * 16 + gg] = mx_lo; sm_wm[warp * 16 + gg + 8] = mx_hi; }
        __syncthreads();
        if (tid < 16u) {
            const float mo = sm_m[tid];
            float mm = mo;
#pragma unroll
            for (uint32_t w = 0; w < NWARP; ++w) mm = fmaxf(mm, sm_wm[w * 16 + tid]);
            sm_c[tid] = (mo == -INFINITY || mm == -INFINITY) ? 0.f : __expf(mo - mm);
            sm_m[tid] = mm;
        }
        __syncthreads();

        // ── P = exp(S - m), row sums, and P staged for the second GEMM
        const float m_lo = sm_m[gg], m_hi = sm_m[gg + 8];
        float sl_lo = 0.f, sl_hi = 0.f;
#pragma unroll
        for (uint32_t jg = 0; jg < NJG; ++jg) {
            const float p0 = (s[jg][0] == -INFINITY) ? 0.f : __expf(s[jg][0] - m_lo);
            const float p1 = (s[jg][1] == -INFINITY) ? 0.f : __expf(s[jg][1] - m_lo);
            const float p2 = (s[jg][2] == -INFINITY) ? 0.f : __expf(s[jg][2] - m_hi);
            const float p3 = (s[jg][3] == -INFINITY) ? 0.f : __expf(s[jg][3] - m_hi);
            sl_lo += p0 + p1; sl_hi += p2 + p3;
            const uint32_t c0 = wbase + jg * 8u + 2u * t4;
            const __nv_bfloat16 b0 = __float2bfloat16(p0), b1 = __float2bfloat16(p1);
            const __nv_bfloat16 b2 = __float2bfloat16(p2), b3 = __float2bfloat16(p3);
            sm_p[gg * PD_FMHA16_LD + c0] = b0;
            sm_p[gg * PD_FMHA16_LD + c0 + 1] = b1;
            sm_p[(gg + 8) * PD_FMHA16_LD + c0] = b2;
            sm_p[(gg + 8) * PD_FMHA16_LD + c0 + 1] = b3;
            if (NXP > 1u) {
                __nv_bfloat16* r1 = sm_p + 16u * PD_FMHA16_LD;
                r1[gg * PD_FMHA16_LD + c0] = __float2bfloat16(p0 - __bfloat162float(b0));
                r1[gg * PD_FMHA16_LD + c0 + 1] = __float2bfloat16(p1 - __bfloat162float(b1));
                r1[(gg + 8) * PD_FMHA16_LD + c0] = __float2bfloat16(p2 - __bfloat162float(b2));
                r1[(gg + 8) * PD_FMHA16_LD + c0 + 1] = __float2bfloat16(p3 - __bfloat162float(b3));
            }
        }
        sl_lo += __shfl_xor_sync(0xffffffffu, sl_lo, 1);
        sl_lo += __shfl_xor_sync(0xffffffffu, sl_lo, 2);
        sl_hi += __shfl_xor_sync(0xffffffffu, sl_hi, 1);
        sl_hi += __shfl_xor_sync(0xffffffffu, sl_hi, 2);
        if (t4 == 0) { sm_wl[warp * 16 + gg] = sl_lo; sm_wl[warp * 16 + gg + 8] = sl_hi; }

        // rescale this warp's O columns by the per-ROW correction
        const float c_lo = sm_c[gg], c_hi = sm_c[gg + 8];
#pragma unroll
        for (uint32_t jg = 0; jg < NJG; ++jg) {
            o[jg][0] *= c_lo; o[jg][1] *= c_lo;
            o[jg][2] *= c_hi; o[jg][3] *= c_hi;
        }
        __syncthreads();
        if (tid < 16u) {
            float ll = sm_l[tid] * sm_c[tid];
#pragma unroll
            for (uint32_t w = 0; w < NWARP; ++w) ll += sm_wl[w * 16 + tid];
            sm_l[tid] = ll;
        }

        // ── O += P @ V^T. warp owns dims [warp*32, +32) as 4 n8 groups.
        // P past nt is exactly 0 (masked above), so stale sm_vt cannot leak.
        const uint32_t kend = ((nt + 15u) / 16u) * 16u;
#pragma unroll 1
        for (uint32_t k0 = 0; k0 < kend; k0 += 16u) {
            uint32_t af[NXP][4];
#pragma unroll
            for (uint32_t xp = 0; xp < NXP; ++xp)
                pd_bf16m_ldm_x4(&sm_p[xp * 16u * PD_FMHA16_LD + a_roff * PD_FMHA16_LD + k0 + a_kof],
                                af[xp][0], af[xp][1], af[xp][2], af[xp][3]);
#pragma unroll
            for (uint32_t jg = 0; jg < NJG; ++jg) {
                uint32_t bf[2];
                pd_bf16m_ldm_x2_trans(&sm_v[(k0 + (lane & 15u)) * PD_FMHA16_LD + wbase + jg * 8u],
                             bf[0], bf[1]);
#pragma unroll
                for (uint32_t xp = 0; xp < NXP; ++xp) pd_bf16m_mma(o[jg], af[xp], bf);
            }
        }
        __syncthreads();
    }

    // ── single-split epilogue: fold the sink in, normalize, store FINAL
    if (tid < 16u) {
        const float sk = sinks ? sinks[kvh * G + tid] : -INFINITY;
        float ll = sm_l[tid];
        if (sk != -INFINITY) ll += __expf(sk - sm_m[tid]);
        sm_l[tid] = (ll > 0.f) ? (1.0f / ll) : 0.f;
    }
    __syncthreads();
    const float inv_lo = sm_l[gg], inv_hi = sm_l[gg + 8];
    const size_t ob_lo = ((size_t)b * n_heads + kvh * G + gg) * HD;
    const size_t ob_hi = ((size_t)b * n_heads + kvh * G + gg + 8) * HD;
#pragma unroll
    for (uint32_t jg = 0; jg < NJG; ++jg) {
        const uint32_t d0 = wbase + jg * 8u + 2u * t4;
        out[ob_lo + d0] = o[jg][0] * inv_lo;
        out[ob_lo + d0 + 1] = o[jg][1] * inv_lo;
        out[ob_hi + d0] = o[jg][2] * inv_hi;
        out[ob_hi + d0 + 1] = o[jg][3] * inv_hi;
    }
#else
    (void)q; (void)pool_k; (void)pool_v; (void)out; (void)positions;
    (void)slots; (void)block_tables; (void)blocks_per_slot; (void)n_heads;
    (void)kv_dim; (void)swa_window; (void)scale; (void)sinks;
#endif
}

// Host entry. Shape-gated to the muse geometry (fp8-e4m3 KV, hd128, G=16) -
// rc -2 otherwise. No context gate: smem is constant in ctx (KVT-chunked), so
// unlike pd_attn_decode_fused_gqa16 this arm has no pos_max ceiling.
// FINAL-output contract: sinks are folded in-kernel, so the caller must not
// run a combine. Election is the ENGINE's job; this only refuses shapes it
// cannot serve.
PD_EXPORT
int pd_attn_decode_fmha16(const void* q, const void* pool_k, const void* pool_v,
                          const void* sinks, void* out, const void* positions,
                          const void* slots, const void* block_tables,
                          uint32_t blocks_per_slot, uint32_t n_heads,
                          uint32_t n_kv_heads, uint32_t head_dim, uint32_t kv_dim,
                          uint32_t swa_window, uint32_t batch, float scale,
                          uint32_t kv_dtype, void* stream) {
    if (n_heads == 0 || batch == 0) return 0;
    // bf16 m16n8k16 needs sm_80+; the device body compiles to a no-op below
    // that, so refuse on the host rather than launching an empty kernel.
    static int cc_major = -1;
    if (cc_major < 0)
        cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, 0);
    if (cc_major < 8) return -2;
    const uint32_t group = n_kv_heads ? n_heads / n_kv_heads : 1u;
    if (kv_dtype != PD_KV_FP8_E4M3 || head_dim != 128u || group != 16u
        || n_heads != n_kv_heads * group)
        return -2;
    constexpr uint32_t KVT = 128u, NWARP = 16u, NXQ = 2u, NXP = 2u;
    const uint32_t smem = (NXQ * 16u + KVT + KVT + NXP * 16u) * PD_FMHA16_LD * 2u
                        + (16u * 3u + NWARP * 32u) * 4u;
    static int smem_cap = -1;
    if (smem_cap < 0)
        cudaDeviceGetAttribute(&smem_cap, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
    if ((int)smem > smem_cap) return -3;
    static uint32_t fsmem_set = 0;
    if (smem > 48u * 1024u && smem > fsmem_set) {
        cudaFuncSetAttribute(
            (const void*)pd_attn_fmha16_kernel<KVT, NWARP, NXQ, NXP>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        fsmem_set = smem;
    }
    dim3 grid(n_kv_heads, batch);
    pd_pdl_go(pd_attn_fmha16_kernel<KVT, NWARP, NXQ, NXP>,
        grid, 32u * NWARP, smem, (cudaStream_t)stream,
            (const float*)q, (const __nv_fp8_e4m3*)pool_k,
            (const __nv_fp8_e4m3*)pool_v, (float*)out,
            (const unsigned int*)positions, (const unsigned int*)slots,
            (const uint32_t*)block_tables, blocks_per_slot, n_heads,
            kv_dim, swa_window, scale, (const float*)sinks);
    return pd_launch_status();
}
