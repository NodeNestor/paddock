// ── lagd: the hd128 decode partial, v5-class ───────────────────────────────
//
// Replaces the TILE-32 GQA walk (tf32 sc_mma + tf32 PV) for hd128/f16
// fused decode shapes. The walk was ~2x its own memory floor at B=1:
// profiling the A6000 serve grid (8 kvh x 1 x splits, 1 CTA/SM achieved) showed
// barrier 1.93 + wait 1.90 cycles/inst - five __syncthreads per tile and
// the scalar convert chains stall with nothing co-resident to interleave.
// This kernel is the v8-class structure on Ampere primitives (the v3->v5
// lineage from the B200 work): ldmatrix +
// f16 m16n8k16 for scores AND PV, softmax folded into the score warps
// (scores never round-trip smem), 4 barriers/tile, cp.async double-buffer,
// PV widened to all 8 warps (16 dims each - v5's NW_V=HD/32 left half the
// block idle at hd128), and smem sized so 2 CTAs co-reside per SM.
//
// Measured at the laguna serve shapes (B=1, partial+combine
// pair): full 21.4 -> 16.5 us, SWA 20.2 -> 15.4-18.2 us; the kernel alone
// runs at its stage-only floor (9.7/8.5 us). B=8 (c8 gate): full@8 72.9 ->
// 56.8, SWA@4 48.8 -> 40.5. Combine (~6 us isolated) is now the pair's
// biggest slice - the LCO door is the recorded next step.
//
// Numerics (template knobs, measured in the harness):
//   NXQ=2 / NXP=3 (DEFAULT): q as f16 big+residual (~22-bit, the walk's
//     2-split-tf32 score class), P as f16 big+mid+small (~33-bit carry,
//     the walk's accepted 3-split-tf32 PV class). maxrel vs the walk
//     3.3e-5 / relRMS 8e-7 - same class, extra mmas hide under staging
//     (full: free; SWA: ~+2.8 us at B=1, free at B=8).
//   NXQ=1 / NXP=1 (PADDOCK_LAGD_F16): straight f16 q and P - llama
//     fattn's own class; maxrel ~1.4e-2. The perf experiment latch.
// f16 K/V only (the pool dtype is the operand dtype - no converts).
// G = n_heads/n_kv_heads <= 8 (launcher gate); q rows pad to 16 mma rows.
//
// PAGED switches the stage() source addressing (block table vs dense
// slot base) - everything else is byte-identical, same partial/combine
// layout as the GQA walk, so dense<->paged outputs stay bit-equal.
//
// KVT (head-packed rung): __half is the laguna pool
// class, __nv_fp8_e4m3 the nemotron/granite KV8 class. The fp8 arm stages
// RAW e4m3 rows into the FRONT of the destination buffer's own f16 row
// slots (raw row = 128 B inside the 272 B slot - no extra smem, so the
// 2-CTA/SM co-residency survives) and expands in place after the wait:
// every thread reads its chunks to registers, one barrier, writes f16.
// e4m3 -> f16 is exact, so the expanded operands are bit-identical to an
// f16 pool holding the same values; the cost is +2 barriers per tile.
//
// WIDE (same rung): G in 9..16 - the mma d-frags' half-1 rows (8..15),
// which the G<=8 build zero-pads and ignores, become real heads: softmax
// state doubles to [16] and the fold/P-store/corr/o-store paths handle the
// second row per lane. Nemotron's G=16 fills the m16 tile exactly.
template <uint32_t TILE, uint32_t NXQ, uint32_t NXP, bool PAGED,
          typename KVT = __half, bool WIDE = false>
__global__ void pd_attn_decode_lagd_kernel(
    const float* __restrict__ q, const KVT* __restrict__ kc,
    const KVT* __restrict__ vc, float* __restrict__ out_o,
    float* __restrict__ out_ml, const unsigned int* __restrict__ positions,
    const unsigned int* __restrict__ slots,
    const uint32_t* __restrict__ block_tables, uint32_t blocks_per_slot,
    uint32_t max_ctx, uint32_t n_heads, uint32_t n_kv_heads, uint32_t kv_dim,
    uint32_t swa_window, uint32_t n_splits, float scale) {
#if PD_FA_OK
    constexpr uint32_t HD = 128u;
    constexpr bool F8 = sizeof(KVT) == 1u;
    // decode-graph PDL: no-op under the plain laguna launches; the fp8 arm
    // replaces a pd_pdl_go'd kernel (vec8) and must keep the cascade armed
    PD_PDL_ARM();
    const uint32_t kvh = blockIdx.x, b = blockIdx.y, sp = blockIdx.z;
    const uint32_t d = threadIdx.x, nth = blockDim.x;
    const uint32_t warp = d >> 5, lane = d & 31u;
    const uint32_t G = n_heads / n_kv_heads;

    const uint32_t slot = slots ? slots[b] : b;
    const uint32_t pos = positions[b];
    const uint32_t first_pos =
        (swa_window > 0 && pos + 1 > swa_window) ? (pos + 1 - swa_window) : 0;
    const uint32_t n_pos = pos + 1 - first_pos;
    const uint32_t chunk = (n_pos + n_splits - 1) / n_splits;
    const uint32_t lo = sp * chunk;
    uint32_t hi = lo + chunk;
    if (hi > n_pos) hi = n_pos;

    extern __shared__ __align__(16) unsigned char lagd_smraw[];
    constexpr uint32_t row_e = HD + 8u;            // +8-half row pad (bank law)
    constexpr uint32_t w_s = TILE + 8u;
    constexpr uint32_t NSW = TILE / 8u;            // score warps
    __half* s_qh = (__half*)lagd_smraw;                 // [NXQ][16][row_e]
    __half* s_wh = s_qh + (size_t)NXQ * 16u * row_e;    // [NXP][16][w_s]
    __half* s_kv = s_wh + (size_t)NXP * 16u * w_s;      // [2][K,V][TILE][row_e]
    __shared__ float s_m[16], s_l[16], s_corr[16];
    __shared__ float s_pmax[NSW][16], s_psum[NSW][16];

    const float* qb = q + (size_t)b * n_heads * HD;
    const uint32_t* bt =
        PAGED ? block_tables + (size_t)slot * blocks_per_slot : nullptr;
    const KVT* kcb = PAGED ? nullptr : kc + (size_t)slot * max_ctx * kv_dim;
    const KVT* vcb = PAGED ? nullptr : vc + (size_t)slot * max_ctx * kv_dim;

    // f16 q planes: NXQ=2 adds the residual plane (q - f16(q) is exactly
    // representable - the f16 twin of the walk's tf32 big+small split)
    for (uint32_t i = d; i < 16u * row_e; i += nth) {
        const uint32_t r = i / row_e, c = i % row_e;
        const float v = (r < G && c < HD)
            ? qb[((size_t)kvh * G + r) * HD + c] : 0.0f;
        const __half h = __float2half(v);
        s_qh[(size_t)r * row_e + c] = h;
        if (NXQ == 2u)
            s_qh[(size_t)16u * row_e + (size_t)r * row_e + c] =
                __float2half(v - __half2float(h));
    }
    for (uint32_t i = d; i < NXP * 16u * w_s; i += nth) s_wh[i] = __half(0.f);
    if (d < 16u) { s_m[d] = -INFINITY; s_l[d] = 0.0f; }

    // PV o-frags: every warp owns 16 dims as two n8 subtiles (8 x 16 = 128)
    float o_acc[2][4];
    #pragma unroll
    for (uint32_t i = 0; i < 2u; ++i)
        #pragma unroll
        for (uint32_t j = 0; j < 4u; ++j) o_acc[i][j] = 0.0f;

    // 16-byte cp.async lines per row: 16 for the f16 pool, 8 raw for e4m3
    // (the raw bytes land at the FRONT of the destination f16 row slot and
    // expand in place after the wait - see the header note)
    constexpr uint32_t lines = (HD * sizeof(KVT)) >> 4;
    auto stage = [&](uint32_t bf, uint32_t t0) {
        const uint32_t n_t = hi - t0 < TILE ? hi - t0 : TILE;
        if (n_t < TILE) {
            // zero the stale tail rows: PV multiplies them by exact-0
            // weights, but uninitialized smem can be NaN and 0*NaN = NaN
            for (uint32_t i = d; i < 2u * (TILE - n_t) * lines; i += nth) {
                const uint32_t kvsel = i / ((TILE - n_t) * lines);
                const uint32_t j = i - kvsel * (TILE - n_t) * lines;
                const uint32_t p = n_t + j / lines, l = j % lines;
                *(uint4*)((char*)(s_kv
                    + ((size_t)(bf * 2u + kvsel) * TILE + p) * row_e) + l * 16u)
                    = make_uint4(0u, 0u, 0u, 0u);
            }
        }
        for (uint32_t i = d; i < 2u * n_t * lines; i += nth) {
            const uint32_t kvsel = i / (n_t * lines);
            const uint32_t j = i - kvsel * n_t * lines;
            const uint32_t p = j / lines, l = j - p * lines;
            const uint32_t gpos = first_pos + t0 + p;
            const KVT* src;
            if (PAGED) {
                const uint32_t blk = bt[gpos >> 4];
                src = (kvsel ? vc : kc)
                    + (size_t)blk * 16u * kv_dim + (size_t)(gpos & 15u) * kv_dim
                    + (size_t)kvh * HD;
            } else {
                src = (kvsel ? vcb : kcb)
                    + (size_t)gpos * kv_dim + (size_t)kvh * HD;
            }
            __half* dst = s_kv + ((size_t)(bf * 2u + kvsel) * TILE + p) * row_e;
            pd_attn_cpa16((char*)dst + l * 16u, (const char*)src + l * 16u);
        }
        pd_attn_cpa_commit();
    };

    __syncthreads();
    if (lo < hi) stage(0u, lo);
    uint32_t bf = 0;
    for (uint32_t t0 = lo; t0 < hi; t0 += TILE, bf ^= 1u) {
        const uint32_t n_t = hi - t0 < TILE ? hi - t0 : TILE;
        const bool more = t0 + TILE < hi;
        if (more) stage(bf ^ 1u, t0 + TILE);
        if (more) pd_attn_cpa_wait1(); else pd_attn_cpa_wait0();
        __syncthreads();
        if constexpr (F8) {
            // expand buffer bf's raw e4m3 rows (front 128 B of each f16 row
            // slot) to f16 in place. Reads and writes overlap inside a row,
            // so: every thread loads its 16-byte chunks to registers, one
            // barrier, then writes the 32-byte f16 chunks. Covers the full
            // TILE including the zero tail (raw zeros expand to f16 zeros).
            constexpr uint32_t NCH = 2u * TILE * (HD / 16u);
            uint4 rv[(NCH + 255u) / 256u];
            uint32_t nc = 0;
            for (uint32_t c = d; c < NCH; c += nth, ++nc) {
                const uint32_t kvsel = c / (TILE * (HD / 16u));
                const uint32_t rem = c - kvsel * TILE * (HD / 16u);
                const uint32_t p = rem / (HD / 16u), j = rem % (HD / 16u);
                const char* rb = (const char*)(s_kv
                    + ((size_t)(bf * 2u + kvsel) * TILE + p) * row_e);
                rv[nc] = *(const uint4*)(rb + (size_t)j * 16u);
            }
            __syncthreads();
            nc = 0;
            for (uint32_t c = d; c < NCH; c += nth, ++nc) {
                const uint32_t kvsel = c / (TILE * (HD / 16u));
                const uint32_t rem = c - kvsel * TILE * (HD / 16u);
                const uint32_t p = rem / (HD / 16u), j = rem % (HD / 16u);
                __half* wb = s_kv
                    + ((size_t)(bf * 2u + kvsel) * TILE + p) * row_e
                    + (size_t)j * 16u;
                const unsigned char* by = (const unsigned char*)&rv[nc];
                #pragma unroll
                for (uint32_t e = 0; e < 16u; ++e) {
                    __nv_fp8_e4m3 t;
                    t.__x = by[e];
                    wb[e] = __float2half(float(t));   // both hops exact
                }
            }
            __syncthreads();
        }
        const __half* kbuf = s_kv + (size_t)(bf * 2u) * TILE * row_e;
        const __half* vbuf = s_kv + ((size_t)(bf * 2u) + 1u) * TILE * row_e;
        // scores in registers: warp w owns cols [w*8, +8); m16n8k16 over the
        // head dim, q frags from the f16 plane(s), K rows as B (no trans)
        float dfr[4] = {0.f, 0.f, 0.f, 0.f};
        if (warp < NSW) {
            const uint32_t p0 = warp * 8u;
            #pragma unroll
            for (uint32_t kk = 0; kk < HD; kk += 16u) {
                uint32_t af[4];
                const __half* ap = s_qh + (size_t)(lane & 15u) * row_e
                                 + kk + ((lane >> 4) ? 8u : 0u);
                pd_ldm_x4(af, (const unsigned char*)ap);
                uint32_t bfr[2];
                const __half* bp = kbuf + (size_t)(p0 + (lane & 7u)) * row_e
                                 + kk + (((lane >> 3) & 1u) ? 8u : 0u);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                             : "=r"(bfr[0]), "=r"(bfr[1])
                             : "r"((unsigned)__cvta_generic_to_shared(bp)));
                pd_fa_mma16(dfr, af[0], af[1], af[2], af[3], bfr[0], bfr[1]);
                if (NXQ == 2u) {
                    uint32_t ar[4];
                    const __half* rp = ap + (size_t)16u * row_e;
                    pd_ldm_x4(ar, (const unsigned char*)rp);
                    pd_fa_mma16(dfr, ar[0], ar[1], ar[2], ar[3], bfr[0], bfr[1]);
                }
            }
            // mask + scale in regs; quad-shfl partial row max (rows < G all
            // live in frag half 0 for G <= 8)
            #pragma unroll
            for (uint32_t half = 0; half < 2u; ++half)
                #pragma unroll
                for (uint32_t cc = 0; cc < 2u; ++cc) {
                    const uint32_t pp = p0 + 2u * (lane & 3u) + cc;
                    dfr[half * 2u + cc] = pp < n_t
                        ? dfr[half * 2u + cc] * scale : -INFINITY;
                }
            float pm = fmaxf(dfr[0], dfr[1]);
            #pragma unroll
            for (uint32_t off = 1; off <= 2; off <<= 1)
                pm = fmaxf(pm, __shfl_xor_sync(0xffffffffu, pm, off));
            const uint32_t rr = lane >> 2;
            if ((lane & 3u) == 0 && rr < G) s_pmax[warp][rr] = pm;
            if constexpr (WIDE) {
                // frag half 1 = rows 8..15, real heads at G > 8
                float pm1 = fmaxf(dfr[2], dfr[3]);
                #pragma unroll
                for (uint32_t off = 1; off <= 2; off <<= 1)
                    pm1 = fmaxf(pm1, __shfl_xor_sync(0xffffffffu, pm1, off));
                if ((lane & 3u) == 0 && 8u + rr < G) s_pmax[warp][8u + rr] = pm1;
            }
        }
        __syncthreads();
        // fold + weights, still in the score warps (redundant per warp);
        // scaled scores never left registers
        if (warp < NSW) {
            const uint32_t p0 = warp * 8u;
            const uint32_t rr = lane >> 2;
            // per-row fold + P store, shared by both frag halves (row is rr
            // or 8+rr, scores d0/d1 are the half's two columns)
            auto fold_row = [&](uint32_t row, float d0, float d1, float& mnew,
                                float& corr, float& w0, float& w1) {
                float m = s_m[row];
                #pragma unroll
                for (uint32_t sw = 0; sw < NSW; ++sw)
                    m = fmaxf(m, s_pmax[sw][row]);
                mnew = m;
                corr = __expf(s_m[row] - m);
                w0 = d0 > -INFINITY ? __expf(d0 - m) : 0.f;
                w1 = d1 > -INFINITY ? __expf(d1 - m) : 0.f;
                const uint32_t pp = p0 + 2u * (lane & 3u);
                const __half h0 = __float2half(w0), h1 = __float2half(w1);
                *(__half2*)(s_wh + (size_t)row * w_s + pp) = __halves2half2(h0, h1);
                if (NXP == 3u) {
                    // f16 3-split of P: big + mid + small carries ~33 bits
                    // (> f32's 24) - the accepted PV class, in ldmatrix-
                    // consumable planes
                    const float r0 = w0 - __half2float(h0);
                    const float r1 = w1 - __half2float(h1);
                    const __half m0 = __float2half(r0), m1 = __float2half(r1);
                    *(__half2*)(s_wh + (size_t)16u * w_s + (size_t)row * w_s + pp) =
                        __halves2half2(m0, m1);
                    *(__half2*)(s_wh + (size_t)32u * w_s + (size_t)row * w_s + pp) =
                        __halves2half2(__float2half(r0 - __half2float(m0)),
                                       __float2half(r1 - __half2float(m1)));
                }
            };
            float mnew = 0.f, corr = 1.f, w0 = 0.f, w1 = 0.f;
            if (rr < G) fold_row(rr, dfr[0], dfr[1], mnew, corr, w0, w1);
            float ps = w0 + w1;
            #pragma unroll
            for (uint32_t off = 1; off <= 2; off <<= 1)
                ps += __shfl_xor_sync(0xffffffffu, ps, off);
            if ((lane & 3u) == 0 && rr < G) {
                s_psum[warp][rr] = ps;
                if (warp == 0) s_corr[rr] = corr;
                if (warp == 0) s_m[rr] = mnew;   // safe: all readers folded
            }
            if constexpr (WIDE) {
                float mnew1 = 0.f, corr1 = 1.f, w2 = 0.f, w3 = 0.f;
                if (8u + rr < G)
                    fold_row(8u + rr, dfr[2], dfr[3], mnew1, corr1, w2, w3);
                float ps1 = w2 + w3;
                #pragma unroll
                for (uint32_t off = 1; off <= 2; off <<= 1)
                    ps1 += __shfl_xor_sync(0xffffffffu, ps1, off);
                if ((lane & 3u) == 0 && 8u + rr < G) {
                    s_psum[warp][8u + rr] = ps1;
                    if (warp == 0) s_corr[8u + rr] = corr1;
                    if (warp == 0) s_m[8u + rr] = mnew1;
                }
            }
        }
        __syncthreads();
        // o update: all 8 warps, 16 dims each (two n8 subtiles); A = the P
        // plane(s), B = V rows transposed
        {
            const uint32_t n_base_w = warp * 16u;
            {
                const uint32_t rr = lane >> 2;
                const float corr = rr < G ? s_corr[rr] : 1.0f;
                #pragma unroll
                for (uint32_t sub = 0; sub < 2u; ++sub) {
                    o_acc[sub][0] *= corr;
                    o_acc[sub][1] *= corr;
                }
                if constexpr (WIDE) {
                    // half-1 rows: o_acc[sub][2..3] carry row 8+rr
                    const float c1 = 8u + rr < G ? s_corr[8u + rr] : 1.0f;
                    #pragma unroll
                    for (uint32_t sub = 0; sub < 2u; ++sub) {
                        o_acc[sub][2] *= c1;
                        o_acc[sub][3] *= c1;
                    }
                }
            }
            #pragma unroll
            for (uint32_t kk = 0; kk < TILE; kk += 16u) {
                uint32_t af[NXP][4];
                #pragma unroll
                for (uint32_t x = 0; x < NXP; ++x) {
                    const __half* ap = s_wh + (size_t)x * 16u * w_s
                                     + (size_t)(lane & 15u) * w_s
                                     + kk + ((lane >> 4) ? 8u : 0u);
                    pd_ldm_x4(af[x], (const unsigned char*)ap);
                }
                #pragma unroll
                for (uint32_t sub = 0; sub < 2u; ++sub) {
                    uint32_t bfr[2];
                    const __half* bp = vbuf + (size_t)(kk + (lane & 15u)) * row_e
                                     + n_base_w + sub * 8u;
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
                                 : "=r"(bfr[0]), "=r"(bfr[1])
                                 : "r"((unsigned)__cvta_generic_to_shared(bp)));
                    #pragma unroll
                    for (uint32_t x = 0; x < NXP; ++x)
                        pd_fa_mma16(o_acc[sub], af[x][0], af[x][1], af[x][2],
                                    af[x][3], bfr[0], bfr[1]);
                }
            }
        }
        if (d < G) {
            float ws = 0.0f;
            #pragma unroll
            for (uint32_t sw = 0; sw < NSW; ++sw) ws += s_psum[sw][d];
            s_l[d] = s_l[d] * s_corr[d] + ws;
        }
        __syncthreads();
    }
    __syncthreads();
    // partial store in the production combine layout (o frag rows = heads;
    // only frag half 0 is real for G <= 8, half 1 rows accumulate exact 0)
    {
        const uint32_t n_base_w = warp * 16u;
        const uint32_t rr = lane >> 2;
        if (rr < G) {
            const size_t pidx = ((size_t)(kvh * G + rr) * gridDim.y + b) * n_splits + sp;
            #pragma unroll
            for (uint32_t sub = 0; sub < 2u; ++sub) {
                float* dst = out_o + pidx * HD + n_base_w + sub * 8u + 2u * (lane & 3u);
                dst[0] = o_acc[sub][0];
                dst[1] = o_acc[sub][1];
            }
        }
        if constexpr (WIDE) {
            if (8u + rr < G) {
                const size_t pidx =
                    ((size_t)(kvh * G + 8u + rr) * gridDim.y + b) * n_splits + sp;
                #pragma unroll
                for (uint32_t sub = 0; sub < 2u; ++sub) {
                    float* dst =
                        out_o + pidx * HD + n_base_w + sub * 8u + 2u * (lane & 3u);
                    dst[0] = o_acc[sub][2];
                    dst[1] = o_acc[sub][3];
                }
            }
        }
    }
    if (d < G) {
        const size_t pidx = ((size_t)(kvh * G + d) * gridDim.y + b) * n_splits + sp;
        out_ml[pidx * 2 + 0] = s_m[d];
        out_ml[pidx * 2 + 1] = s_l[d];
    }
#else
    (void)q; (void)kc; (void)vc; (void)out_o; (void)out_ml; (void)positions;
    (void)slots; (void)block_tables; (void)blocks_per_slot; (void)max_ctx;
    (void)n_heads; (void)n_kv_heads; (void)kv_dim; (void)swa_window;
    (void)n_splits; (void)scale;
#endif
}
