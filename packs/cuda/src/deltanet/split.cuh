// deltanet/split.cuh - chunked stage-2 SPLIT walk: the serial
// chunk walk reduced to the state chain plus the o1 fold, and a chunk-
// parallel coef pass. Textually-included segment of the single pack
// translation unit; not standalone-compilable (order: after
// gemm/int8_mma.cuh for the PD_DNC_* defines, before
// deltanet/stage2_sample.cuh whose launcher dispatches these kernels).
//
// Why (attribution): vLLM's
// GDN prefill runs the same 192-CTA one-wave serial state kernel geometry at
// a fraction of the classic v2 walk's ~15 us/chunk: the v2 walk carries the
// whole output assembly (q fold, coef pass, out writeback, their ring slots
// and barriers) on the serial chain. Iteration history: the two-level scan
// (~2x slower, transfer FLOP), the fla DO_OUT/DO_STATE split (-24/-34%,
// S0-stash traffic), and iteration 1 of this kernel (walk3 83 us but a
// 55 us o-pass - bandwidth-bound on a 33 MB f32 entry-state stash) all lost
// to the classic walk. This iteration keeps the o1 = gam*(q S0) term in the
// walk on warps 4-7 (idle during pass 1; q comes straight from L2, the
// state slice is already resident - the entry-state stash disappears
// entirely) and moves only the coef x deltaT correction to the parallel
// pass, which SEEDS its accumulator from the walk's out rows so the
// fragment chain keeps the classic order.
//
// BIT-EXACTNESS vs the classic v2 walk (the serving default), by
// construction: every mma chain keeps the classic k-grouping and operand
// bits. Pass 1 folds k = 0,8,...,120 per fragment in classic order (two
// KS=64 dw slabs; slab width does not regroup a sequential acc chain); the
// o1 chain on warps 4-7 reads the same q bits from gmem; deltas subtract du
// read straight from gmem (prefetched to registers at chunk top); pass 3 is
// the classic body verbatim; the coef pass loads gam*o1 back from out
// (exact f32 round-trip) and continues the classic pass-2 chain. Gates:
// dnc_pf_bench word-compare vs classic, greedy, suite.
//
// Ring: 4 items/chunk (dw k-slab x2, raw-k a-half x2), slot = C x (64+4)
// floats, 4 slots = one chunk of lookahead. The dw pair and the k pair are
// each consumed under one wait_group 2 - two waits and five barriers per
// chunk (classic: eight waits, ~sixteen barriers).
// smem: state G x (D+4) + dT G x (C+4) + 4 slots = 95232 B (sm_120 dynamic
// cap is 99 KB).

// ---- tf32 mma helpers (moved from stage2_sample.cuh - shared by the v1/v2
// walks, the scan, and the split kernels below; definitions unchanged) ----
static __device__ __forceinline__ uint32_t pd_tf32(float v) {
    uint32_t r;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(v));
    return r;
}
static __device__ __forceinline__ uint32_t pd_tf32_small(float v, uint32_t big) {
    float bf = __uint_as_float(big);
    return pd_tf32(v - bf);
}

template <uint32_t PREC>
static __device__ __forceinline__ void pd_dnm_mma(float (&acc)[4], const float araw[4],
                                                  const float braw[2]) {
    uint32_t ab[4], bb[2];
#pragma unroll
    for (uint32_t i = 0; i < 4; ++i) ab[i] = pd_tf32(araw[i]);
#pragma unroll
    for (uint32_t i = 0; i < 2; ++i) bb[i] = pd_tf32(braw[i]);
    asm("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(ab[0]), "r"(ab[1]), "r"(ab[2]), "r"(ab[3]), "r"(bb[0]), "r"(bb[1]));
    if (PREC == 3u) {
        uint32_t as[4], bs[2];
#pragma unroll
        for (uint32_t i = 0; i < 4; ++i) as[i] = pd_tf32_small(araw[i], ab[i]);
#pragma unroll
        for (uint32_t i = 0; i < 2; ++i) bs[i] = pd_tf32_small(braw[i], bb[i]);
        asm("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
            : "r"(ab[0]), "r"(ab[1]), "r"(ab[2]), "r"(ab[3]), "r"(bs[0]), "r"(bs[1]));
        asm("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
            : "r"(as[0]), "r"(as[1]), "r"(as[2]), "r"(as[3]), "r"(bb[0]), "r"(bb[1]));
    }
}

// 3xTF32 with the SMALL-term mmas in swapped order: (a_small, b_big) before
// (a_big, b_small) - pass 3 flips operand roles vs v1 (A=deltaT, B=wk); this
// keeps v1's exact chain wk_b*dT_b, wk_b*dT_s, wk_s*dT_b.
template <uint32_t PREC>
static __device__ __forceinline__ void pd_dnm_mma_sw(float (&acc)[4], const float araw[4],
                                                     const float braw[2]) {
    uint32_t ab[4], bb[2];
#pragma unroll
    for (uint32_t i = 0; i < 4; ++i) ab[i] = pd_tf32(araw[i]);
#pragma unroll
    for (uint32_t i = 0; i < 2; ++i) bb[i] = pd_tf32(braw[i]);
    asm("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(ab[0]), "r"(ab[1]), "r"(ab[2]), "r"(ab[3]), "r"(bb[0]), "r"(bb[1]));
    if (PREC == 3u) {
        uint32_t as[4], bs[2];
#pragma unroll
        for (uint32_t i = 0; i < 4; ++i) as[i] = pd_tf32_small(araw[i], ab[i]);
#pragma unroll
        for (uint32_t i = 0; i < 2; ++i) bs[i] = pd_tf32_small(braw[i], bb[i]);
        asm("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
            : "r"(as[0]), "r"(as[1]), "r"(as[2]), "r"(as[3]), "r"(bb[0]), "r"(bb[1]));
        asm("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
            : "r"(ab[0]), "r"(ab[1]), "r"(ab[2]), "r"(ab[3]), "r"(bs[0]), "r"(bs[1]));
    }
}

static __device__ __forceinline__ void pd_dnc_cpa16(void* smem, const void* gmem,
                                                    uint32_t bytes) {
    const unsigned sm = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(sm), "l"(gmem),
                 "r"(bytes));
}

#define PD_DNS3_KS 64u
#define PD_DNS3_SLOT (PD_DNC_C * (PD_DNS3_KS + 4u))
#define PD_DNS3_WALK_SMEM                                                      \
    ((PD_DNC_G * (PD_DNC_D + 4u) + PD_DNC_G * (PD_DNC_C + 4u) +                \
      4u * PD_DNS3_SLOT) *                                                     \
     4u)

// ---- split walk: grid (H, D/G), 256 threads. Per chunk: warps 0-3 resolve
// delta = du - dw S0 (dw in two KS=64 ring slabs under one wait, du
// prefetched to registers at chunk top), warps 4-7 build the o1 = gam*(q S0)
// rows concurrently (q from L2 at fragment coords, state slice resident)
// and write them to out; pass 3 (all warps, classic body, both a-halves
// under one wait) hops the state. deltaT goes to the stash for the coef
// pass. No coef item, no entry-state stash.
template <uint32_t PREC, typename ST = float>
__global__ void __launch_bounds__(256)
pd_dnc_walk3_kernel(const float* __restrict__ q, const float* __restrict__ k,
                    ST* __restrict__ state, const float* __restrict__ dw,
                    const float* __restrict__ du, const double* __restrict__ cg,
                    float* __restrict__ out, float* __restrict__ dstash,
                    uint32_t n_tokens, uint32_t n_heads) {
    constexpr uint32_t D = PD_DNC_D, C = PD_DNC_C, G = PD_DNC_G;
    constexpr uint32_t KS = PD_DNS3_KS;
    constexpr uint32_t NT = G / 8u;
    constexpr uint32_t SD = D + 4u, SC = C + 4u, SK = KS + 4u;
    const uint32_t h = blockIdx.x, col0 = blockIdx.y * G;
    const uint32_t tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
    const uint32_t g8 = lane >> 2, t4 = lane & 3u;
    const uint32_t nc = (n_tokens + C - 1u) / C;

    extern __shared__ float shm[];
    float* sh_s = shm;                 // [G][SD] resident state columns
    float* sh_dT = sh_s + G * SD;      // [G][SC] deltas, c-major
    float* sh_ring = sh_dT + G * SC;   // 4 x PD_DNS3_SLOT
    __shared__ float sh_w[C], sh_gam[C];
    __shared__ float sh_gall;

    // items (positional, 4/chunk): 0/1 = dw K-slabs, 2/3 = raw-k a-halves
    auto issue_item = [&](uint32_t it) {
        const uint32_t ch = it / 4u;
        if (ch < nc) {
            const uint32_t ty = it % 4u;
            const uint32_t c0i = ch * C;
            const uint32_t cli = min(C, n_tokens - c0i);
            const size_t tbi = (size_t)ch * n_heads + h;
            float* pane = sh_ring + (size_t)(it & 3u) * PD_DNS3_SLOT;
            if (ty < 2u) {
                const uint32_t k0s = ty * KS;
                for (uint32_t u = tid; u < C * (KS / 4u); u += 256u) {
                    const uint32_t r = u / (KS / 4u), c4 = (u % (KS / 4u)) * 4u;
                    pd_dnc_cpa16(pane + r * SK + c4,
                                 dw + (tbi * C + r) * D + k0s + c4,
                                 r < cli ? 16u : 0u);
                }
            } else {
                const uint32_t a0 = (ty - 2u) * 64u;
                for (uint32_t u = tid; u < C * 16u; u += 256u) {
                    const uint32_t j = u / 16u, c4 = (u % 16u) * 4u;
                    pd_dnc_cpa16(pane + j * SC + c4,
                                 k + ((size_t)(c0i + j) * n_heads + h) * D + a0 + c4,
                                 j < cli ? 16u : 0u);
                }
            }
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
    };
    // consume a PAIR of items under one wait: at most two groups (the other
    // pair) stay pending
    auto pair_wait = [&] {
        asm volatile("cp.async.wait_group 2;" ::: "memory");
        __syncthreads();
    };

    ST* s_head = state + (size_t)h * D * D;
    for (uint32_t idx = tid; idx < G * (D / 4u); idx += 256u) {
        const uint32_t c = idx / (D / 4u), a4 = (idx % (D / 4u)) * 4u;
        *reinterpret_cast<float4*>(&sh_s[c * SD + a4]) =
            pd_dns_ld4(s_head + (size_t)(col0 + c) * D + a4);
    }
    issue_item(0); issue_item(1); issue_item(2); issue_item(3);
    __syncthreads();

    for (uint32_t ch = 0; ch < nc; ++ch) {
        const uint32_t c0 = ch * C;
        const uint32_t cl = min(C, n_tokens - c0);
        const size_t tb = (size_t)ch * n_heads + h;
        const uint32_t it0 = 4u * ch;

        // decay vectors: single-buffered like the classic walk (first read
        // is >=2 barriers away, last read precedes the pass-3 end barrier)
        if (tid < C) {
            sh_w[tid] = tid < cl
                ? expf((float)(cg[tb * C + cl - 1u] - cg[tb * C + tid]))
                : 0.0f;
            sh_gam[tid] = tid < cl ? expf((float)cg[tb * C + tid]) : 0.0f;
        }
        if (tid == 0) sh_gall = expf((float)(cg[tb * C + cl - 1u]));

        // du prefetch (warps 0-3): fragment-coord loads issued before pass 1
        // hides the gmem latency behind the mma work
        float rdu[NT][4];
        if (warp < 4u) {
#pragma unroll
            for (uint32_t nt = 0; nt < NT; ++nt) {
#pragma unroll
                for (uint32_t e = 0; e < 4; ++e) {
                    const uint32_t i = warp * 16u + g8 + (e >= 2 ? 8u : 0u);
                    const uint32_t c = nt * 8u + 2u * t4 + (e & 1u);
                    rdu[nt][e] = i < cl ? du[(tb * C + i) * D + col0 + c] : 0.f;
                }
            }
        }

        // ---- pass 1, both dw slabs under one wait. Warps 0-3: dw x S0.
        // Warps 4-7: the o1 = q x S0 fold over the same k range, q read at
        // fragment coords from gmem (L2; the chunk's q is 32 KB shared by
        // the head's four blocks) - classic bits, classic k order.
        float acc1[NT][4];
#pragma unroll
        for (uint32_t nt = 0; nt < NT; ++nt)
#pragma unroll
            for (uint32_t e = 0; e < 4; ++e) acc1[nt][e] = 0.f;
        pair_wait();
        for (uint32_t slab = 0; slab < 2; ++slab) {
            const float* pane = sh_ring + (size_t)((it0 + slab) & 3u) * PD_DNS3_SLOT;
            const uint32_t k0s = slab * KS;
            if (warp < 4u) {
                const uint32_t p1r = warp * 16u;
#pragma unroll
                for (uint32_t kk = 0; kk < KS; kk += 8u) {
                    float ar[4];
                    ar[0] = pane[(p1r + g8) * SK + kk + t4];
                    ar[1] = pane[(p1r + g8 + 8u) * SK + kk + t4];
                    ar[2] = pane[(p1r + g8) * SK + kk + t4 + 4u];
                    ar[3] = pane[(p1r + g8 + 8u) * SK + kk + t4 + 4u];
#pragma unroll
                    for (uint32_t nt = 0; nt < NT; ++nt) {
                        float br[2];
                        br[0] = sh_s[(nt * 8u + g8) * SD + k0s + kk + t4];
                        br[1] = sh_s[(nt * 8u + g8) * SD + k0s + kk + t4 + 4u];
                        pd_dnm_mma<PREC>(acc1[nt], ar, br);
                    }
                }
            } else {
                const uint32_t qr = (warp - 4u) * 16u;
                const uint32_t i0 = qr + g8, i1 = qr + g8 + 8u;
                const float* q0 = q + ((size_t)(c0 + i0) * n_heads + h) * D;
                const float* q1 = q + ((size_t)(c0 + i1) * n_heads + h) * D;
                const bool v0 = i0 < cl, v1 = i1 < cl;
#pragma unroll
                for (uint32_t kk = 0; kk < KS; kk += 8u) {
                    float ar[4];
                    ar[0] = v0 ? q0[k0s + kk + t4] : 0.f;
                    ar[1] = v1 ? q1[k0s + kk + t4] : 0.f;
                    ar[2] = v0 ? q0[k0s + kk + t4 + 4u] : 0.f;
                    ar[3] = v1 ? q1[k0s + kk + t4 + 4u] : 0.f;
#pragma unroll
                    for (uint32_t nt = 0; nt < NT; ++nt) {
                        float br[2];
                        br[0] = sh_s[(nt * 8u + g8) * SD + k0s + kk + t4];
                        br[1] = sh_s[(nt * 8u + g8) * SD + k0s + kk + t4 + 4u];
                        pd_dnm_mma<PREC>(acc1[nt], ar, br);
                    }
                }
            }
        }
        __syncthreads();
        issue_item(it0 + 4u); issue_item(it0 + 5u);

        // ---- split: warps 0-3 resolve deltas into sh_dT; warps 4-7
        // gam-scale o1 (classic split-phase math) and write the out rows -
        // the coef pass accumulates on top of them.
        if (warp < 4u) {
#pragma unroll
            for (uint32_t nt = 0; nt < NT; ++nt) {
#pragma unroll
                for (uint32_t e = 0; e < 4; ++e) {
                    const uint32_t i = warp * 16u + g8 + (e >= 2 ? 8u : 0u);
                    const uint32_t c = nt * 8u + 2u * t4 + (e & 1u);
                    float dlt = 0.f;
                    if (i < cl) dlt = rdu[nt][e] - acc1[nt][e];
                    sh_dT[c * SC + i] = dlt;
                }
            }
        } else {
#pragma unroll
            for (uint32_t nt = 0; nt < NT; ++nt) {
#pragma unroll
                for (uint32_t e = 0; e < 4; ++e) {
                    const uint32_t i = (warp - 4u) * 16u + g8 + (e >= 2 ? 8u : 0u);
                    const uint32_t c = nt * 8u + 2u * t4 + (e & 1u);
                    if (i < cl)
                        out[((size_t)(c0 + i) * n_heads + h) * D + col0 + c] =
                            acc1[nt][e] * sh_gam[i];
                }
            }
        }
        __syncthreads();
        {
            // deltaT slice to the stash (the coef pass's B operand)
            float* dst = dstash + ((size_t)ch * n_heads + h) * (size_t)(C * D) +
                         (size_t)col0 * C;
            for (uint32_t idx = tid; idx < G * (C / 4u); idx += 256u) {
                const uint32_t c = idx / (C / 4u), i4 = (idx % (C / 4u)) * 4u;
                *reinterpret_cast<float4*>(&dst[(size_t)c * C + i4]) =
                    *reinterpret_cast<const float4*>(&sh_dT[c * SC + i4]);
            }
        }

        // ---- pass 3 (classic body), both a-halves under one wait:
        // S(c-rows) = gall*S + deltaT (G x C) x wk (C x D, raw k pane, w_j
        // folded at the B load)
        pair_wait();
        for (uint32_t half = 0; half < 2; ++half) {
            const float* pane =
                sh_ring + (size_t)((it0 + 2u + half) & 3u) * PD_DNS3_SLOT;
            const uint32_t mt = warp & 1u;
            const uint32_t nt0 = (warp >> 1) * 2u;
            float acc3[2][4];
#pragma unroll
            for (uint32_t nn = 0; nn < 2; ++nn) {
#pragma unroll
                for (uint32_t e = 0; e < 4; ++e) {
                    const uint32_t c = mt * 16u + g8 + (e >= 2 ? 8u : 0u);
                    const uint32_t a = half * 64u + (nt0 + nn) * 8u + 2u * t4 + (e & 1u);
                    acc3[nn][e] = sh_gall * sh_s[c * SD + a];
                }
            }
#pragma unroll
            for (uint32_t kk = 0; kk < C; kk += 8u) {
                float ar[4];
                ar[0] = sh_dT[(mt * 16u + g8) * SC + kk + t4];
                ar[1] = sh_dT[(mt * 16u + g8 + 8u) * SC + kk + t4];
                ar[2] = sh_dT[(mt * 16u + g8) * SC + kk + t4 + 4u];
                ar[3] = sh_dT[(mt * 16u + g8 + 8u) * SC + kk + t4 + 4u];
                const float w0 = sh_w[kk + t4], w4 = sh_w[kk + t4 + 4u];
#pragma unroll
                for (uint32_t nn = 0; nn < 2; ++nn) {
                    float br[2];
                    br[0] = w0 * pane[(kk + t4) * SC + (nt0 + nn) * 8u + g8];
                    br[1] = w4 * pane[(kk + t4 + 4u) * SC + (nt0 + nn) * 8u + g8];
                    pd_dnm_mma_sw<PREC>(acc3[nn], ar, br);
                }
            }
#pragma unroll
            for (uint32_t nn = 0; nn < 2; ++nn) {
#pragma unroll
                for (uint32_t e = 0; e < 4; ++e) {
                    const uint32_t c = mt * 16u + g8 + (e >= 2 ? 8u : 0u);
                    const uint32_t a = half * 64u + (nt0 + nn) * 8u + 2u * t4 + (e & 1u);
                    sh_s[c * SD + a] = acc3[nn][e];
                }
            }
        }
        __syncthreads();
        issue_item(it0 + 6u); issue_item(it0 + 7u);
    }

    for (uint32_t idx = tid; idx < G * (D / 4u); idx += 256u) {
        const uint32_t c = idx / (D / 4u), a4 = (idx % (D / 4u)) * 4u;
        pd_dns_st4(s_head + (size_t)(col0 + c) * D + a4,
                   *reinterpret_cast<const float4*>(&sh_s[c * SD + a4]));
    }
}

#define PD_DNS3_OPASS_SMEM                                                     \
    ((PD_DNC_C * (PD_DNC_C + 4u) + PD_DNC_G * (PD_DNC_C + 4u)) * 4u)

// ---- coef pass: grid (H, D/G, nc), 256 threads, one block per (head,
// G-slice, chunk). out += coef x deltaT, seeded from the walk's gam*o1 rows
// (exact f32 round-trip through gmem) so the fragment chain continues the
// classic pass-2 order. Warps 4-7 own the row tiles (their classic roles);
// warps 0-3 only help stage.
template <uint32_t PREC>
__global__ void __launch_bounds__(256)
pd_dnc_opass_kernel(const float* __restrict__ coef, const float* __restrict__ dstash,
                    float* __restrict__ out, uint32_t n_tokens, uint32_t n_heads) {
    constexpr uint32_t D = PD_DNC_D, C = PD_DNC_C, G = PD_DNC_G;
    constexpr uint32_t NT = G / 8u;
    constexpr uint32_t SC = C + 4u;
    const uint32_t h = blockIdx.x, col0 = blockIdx.y * G;
    const uint32_t ch = blockIdx.z;
    const uint32_t tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
    const uint32_t g8 = lane >> 2, t4 = lane & 3u;
    const uint32_t c0 = ch * C;
    const uint32_t cl = min(C, n_tokens - c0);
    const size_t tb = (size_t)ch * n_heads + h;

    extern __shared__ float shm3[];
    float* sh_cf = shm3;               // [C][SC] coef rows
    float* sh_dT = sh_cf + C * SC;     // [G][SC] deltaT, c-major

    for (uint32_t u = tid; u < C * (C / 4u); u += 256u) {
        const uint32_t r = u / (C / 4u), c4 = (u % (C / 4u)) * 4u;
        uint32_t by = 0u;
        if (r < cl && c4 < cl) by = min(16u, (cl - c4) * 4u);
        pd_dnc_cpa16(sh_cf + r * SC + c4, coef + (tb * C + r) * C + c4, by);
    }
    {
        const float* src = dstash + ((size_t)ch * n_heads + h) * (size_t)(C * D) +
                           (size_t)col0 * C;
        for (uint32_t u = tid; u < G * (C / 4u); u += 256u) {
            const uint32_t c = u / (C / 4u), i4 = (u % (C / 4u)) * 4u;
            pd_dnc_cpa16(sh_dT + c * SC + i4, src + (size_t)c * C + i4, 16u);
        }
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();

    if (warp >= 4u) {
        const uint32_t mt = warp - 4u;
        float acc[NT][4];
        // seed from the walk's gam*o1 rows - the classic chain continues
#pragma unroll
        for (uint32_t nt = 0; nt < NT; ++nt) {
#pragma unroll
            for (uint32_t e = 0; e < 4; ++e) {
                const uint32_t i = mt * 16u + g8 + (e >= 2 ? 8u : 0u);
                const uint32_t c = nt * 8u + 2u * t4 + (e & 1u);
                acc[nt][e] = i < cl
                    ? out[((size_t)(c0 + i) * n_heads + h) * D + col0 + c]
                    : 0.f;
            }
        }
#pragma unroll
        for (uint32_t kk = 0; kk < C; kk += 8u) {
            float ar[4];
            ar[0] = sh_cf[(mt * 16u + g8) * SC + kk + t4];
            ar[1] = sh_cf[(mt * 16u + g8 + 8u) * SC + kk + t4];
            ar[2] = sh_cf[(mt * 16u + g8) * SC + kk + t4 + 4u];
            ar[3] = sh_cf[(mt * 16u + g8 + 8u) * SC + kk + t4 + 4u];
#pragma unroll
            for (uint32_t nt = 0; nt < NT; ++nt) {
                float br[2];
                br[0] = sh_dT[(nt * 8u + g8) * SC + kk + t4];
                br[1] = sh_dT[(nt * 8u + g8) * SC + kk + t4 + 4u];
                pd_dnm_mma<PREC>(acc[nt], ar, br);
            }
        }
#pragma unroll
        for (uint32_t nt = 0; nt < NT; ++nt) {
#pragma unroll
            for (uint32_t e = 0; e < 4; ++e) {
                const uint32_t i = mt * 16u + g8 + (e >= 2 ? 8u : 0u);
                const uint32_t c = nt * 8u + 2u * t4 + (e & 1u);
                if (i < cl)
                    out[((size_t)(c0 + i) * n_heads + h) * D + col0 + c] = acc[nt][e];
            }
        }
    }
}

// ---- stage 1 v2, round 2 (occupancy layout): round 1 replaced
// the scalar dots and the two 2016-step substitution chains with 3xTF32 mma
// + a hierarchical explicit (I+M)^-1, but its 84.5 KB smem (full K + Q
// panes + the densified T) held one CTA per SM - the serial phases (f64
// cumsum, base solves, merges) idled the die and the win was only
// 143.7 -> 120.3 us. This round stages the operands in [C][32+4] a-column
// SLABS (four rounds per matmul, classic stage1's own staging shape; the
// per-fragment k chain is the identical global sequence, so the numerics
// match round 1 bit for bit) and reads T PACKED straight from the M
// buffer's upper triangle at the A-fragment loads (j<i -> mt[j*C+i], j==i
// -> 1, else 0) - no densified pane. smem 35328 B -> two CTAs per SM
// (launch_bounds(256,2) caps registers accordingly).
// sh_mt row stride 72 (C + 8): C=64 floats is a 2-bank-wrap stride, so the
// dw/du packed-T A-loads were 4-way and the M-emit stores 8-way conflicted
// (profiled: sh_mt sites were the top shared-excessive lines). 72*4B
// offsets consecutive rows by 8 banks -> t4-strided reads land on distinct
// banks. Pure address remap, bit-exact, +2304 B smem (still 2 CTAs/SM).
#define PD_DNS1_SMT 72u
#define PD_DNS1_SMEM                                                           \
    ((2u * PD_DNC_C * 36u + PD_DNC_C * PD_DNS1_SMT + 2u * PD_DNC_C) * 4u)

// PREC mirrors the walk's class knob (1 = plain tf32, 3 = 3xTF32 near-f32).
// The walk elected 1 (+1.7% pf8 / +1.1% c32, oracle+PPL gated);
// stage1 shipped hardcoded 3x and never got the same election - its dots and
// dw/du products are the same class as the walk's pass-1/3 chains (tf32's
// 10-bit mantissa is still finer than the bf16 vLLM's fla kernels run for
// these products), so PREC=1 rides the identical gate set. The T=(I+M)^-1
// build stays scalar f32 either way - only the mma chains change class.
// At: the aqk/coef output type. The register-state
// walk consumes coef as a bf16 mma operand, so its route instantiates
// at=bf16 and stage1 writes the bf16 payload straight into the f32-sized
// call-internal buffer (the DWB16 reuse precedent). qb16/kb16 (nullptr
// off) are that route's bf16 q/k copies, emitted by the tail epilogue
// below from the just-read (L2-hot) rows.
// paired fragment store: e0/e1 (and e2/e3) of an m16n8 acc land on ADJACENT
// output columns, but predicated scalar stores kept them as 2x4B STG
// (41% excessive global sectors). One aligned 2-elem store per pair instead.
template <typename T> struct alignas(2 * sizeof(T)) pd_dns1_pair { T x, y; };

template <uint32_t PREC = 3u, typename IT = float, typename OT = float,
          typename VT = IT, typename AT = float>
__global__ void __launch_bounds__(256, 2)
pd_dnc_stage1_v2_kernel(const IT* __restrict__ q, const IT* __restrict__ k,
                        const VT* __restrict__ v, const float* __restrict__ g,
                        const float* __restrict__ beta, OT* __restrict__ dw,
                        OT* __restrict__ du, AT* __restrict__ aqk,
                        double* __restrict__ cg, uint32_t n_tokens,
                        uint32_t n_heads,
                        __nv_bfloat16* __restrict__ qb16 = nullptr,
                        __nv_bfloat16* __restrict__ kb16 = nullptr,
                        float* __restrict__ gsh = nullptr,
                        const uint32_t* __restrict__ vl_items = nullptr) {
    constexpr uint32_t D = PD_DNC_D, C = PD_DNC_C;
    // slab stride: 32 a-cols + pad; bf16 pads to 40 so every cp.async row
    // start stays 16B-aligned (40*2 = 80 | 16; f32: 36*4 = 144 | 16)
    constexpr uint32_t SL = sizeof(IT) == 2 ? 40u : 36u;
    constexpr uint32_t SMT = PD_DNS1_SMT;  // sh_mt padded row stride
    // (a cp.async chunk is 16B = 16/sizeof(it) elems - the staging loops below
    // spell that out at each use rather than through a shared constant)
    const uint32_t ch = blockIdx.x, h = blockIdx.y;
    // varlen chunk items (GDN formulation band): one
    // launch covers every eligible span of the tick; items are (global row0,
    // chunk len) pairs per launch chunk. Scratch (dw/du/aqk/cg/gsh and the
    // qb16/kb16 copies) stays indexed by the LAUNCH chunk (tb / ch*C), so
    // spans' partial chunks never collide; only the q/k/v/g/beta row base
    // comes from the item. Legacy launches (nullptr) are bit-identical.
    uint32_t c0, cl;
    if (vl_items != nullptr) {
        c0 = vl_items[ch * 2u];
        cl = vl_items[ch * 2u + 1u];
    } else {
        c0 = ch * C;
        cl = min(C, n_tokens - c0);
    }
    const uint32_t tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
    const uint32_t g8 = lane >> 2, t4 = lane & 3u;
    const size_t tb = (size_t)ch * n_heads + h;

    extern __shared__ float sh1[];
    IT* sh_sa = (IT*)sh1;               // [C][SL]: k slab; also B slab later
    IT* sh_sb = sh_sa + C * SL;         // [C][SL]: q slab (dots only)
    // W merge scratch overlays the sa pane as f32 (the T-build runs between
    // the dots and dw/du when both slab panes are dead)
    float* sh_w32 = (float*)sh1;
    float* sh_mt = (float*)(sh_sb + C * SL);  // [C][SMT]: M lower, T packed upper
    double* sh_cg = (double*)(sh_mt + C * SMT);  // [C]
    __shared__ float sh_b[PD_DNC_C], sh_bg[PD_DNC_C];

    // slab stager: rows x 32 a-cols starting at a0, zero-filled past cl
    auto stage_slab = [&](auto* pane, const auto* src, uint32_t a0) {
        const uint32_t epc = 16u / (uint32_t)sizeof(src[0]);
        const uint32_t sl = sizeof(src[0]) == 2u ? 40u : 36u;
        const uint32_t nch = 32u / epc;
        for (uint32_t u = tid; u < C * nch; u += 256u) {
            const uint32_t r = u / nch, ce = (u % nch) * epc;
            pd_dnc_cpa16((void*)(pane + r * sl + ce),
                         src + ((size_t)(c0 + r) * n_heads + h) * D + a0 + ce,
                         r < cl ? 16u : 0u);
        }
    };

    // g -> f64 cumsum (thread 0) overlapped with the first slab loads
    if (tid < C) sh_b[tid] = tid < cl ? g[(size_t)(c0 + tid) * n_heads + h] : 0.f;
    __syncthreads();
    stage_slab(sh_sa, k, 0u);
    stage_slab(sh_sb, q, 0u);
    asm volatile("cp.async.commit_group;" ::: "memory");
    if (tid == 0) {
        double run = 0.0;
        for (uint32_t i = 0; i < C; ++i) {
            if (i < cl) run += (double)sh_b[i];
            sh_cg[i] = i < cl ? run : 0.0;
            if (i < cl) cg[tb * C + i] = run;
        }
    }
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
    if (tid < C) {
        const float b = tid < cl ? beta[(size_t)(c0 + tid) * n_heads + h] : 0.f;
        sh_b[tid] = b;
        // __expf (ex2.approx class) is what the reference uses too: Triton
        // tl.exp lowers to ex2.approx, and the full expf's INF-guard FSETP
        // chains were ~10% of this kernel's instruction stream.
        // Arguments are always <= 0 here, so approx underflow-to-zero at
        // the -87 tail matches what the f32 math produced anyway.
        sh_bg[tid] = tid < cl ? b * __expf((float)sh_cg[tid]) : 0.f;
    }
    // RS-walk gate vectors: the walk's per-chunk gates need
    // (float)(cg[cl-1] - cg[i]) and (float)cg[i] - emit them here from the
    // f64 cumsum already in shared, so the walk never touches f64 (its
    // LDG.64 + F2F.F64 chains were 6%+ of walk stalls). Same subtract,
    // same rounding -> the walk's gates are bit-identical.
    if (gsh != nullptr && tid < C) {
        const double cgl = sh_cg[cl - 1u];
        gsh[tb * 2u * C + tid] = tid < cl ? (float)(cgl - sh_cg[tid]) : 0.f;
        gsh[tb * 2u * C + C + tid] = tid < cl ? (float)sh_cg[tid] : 0.f;
    }

    // dots as mma over four a-slabs: warps 0-3 akk (A = k slab), 4-7 aqk
    // (A = q slab); B = k slab ([n=j][k=a] via the row-major slab). The
    // global k chain is a = 0,8,...,120 - identical to a full-pane loop.
    {
        const uint32_t mt = warp & 3u;
        const IT* apane = warp < 4u ? sh_sa : sh_sb;
        float acc[8][4];
#pragma unroll
        for (uint32_t nt = 0; nt < 8u; ++nt)
#pragma unroll
            for (uint32_t e = 0; e < 4; ++e) acc[nt][e] = 0.f;
        for (uint32_t rd = 0; rd < 4u; ++rd) {
#pragma unroll
            for (uint32_t kk = 0; kk < 32u; kk += 8u) {
                float ar[4];
                ar[0] = (float)apane[(mt * 16u + g8) * SL + kk + t4];
                ar[1] = (float)apane[(mt * 16u + g8 + 8u) * SL + kk + t4];
                ar[2] = (float)apane[(mt * 16u + g8) * SL + kk + t4 + 4u];
                ar[3] = (float)apane[(mt * 16u + g8 + 8u) * SL + kk + t4 + 4u];
#pragma unroll
                for (uint32_t nt = 0; nt < 8u; ++nt) {
                    float br[2];
                    br[0] = (float)sh_sa[(nt * 8u + g8) * SL + kk + t4];
                    br[1] = (float)sh_sa[(nt * 8u + g8) * SL + kk + t4 + 4u];
                    pd_dnm_mma<PREC>(acc[nt], ar, br);
                }
            }
            // RS-route bf16 q/k emission folded into the dots rounds: the
            // slabs already sit in shared, so the old tail epilogue's global
            // f32 re-read of q/k (+49 us at T=2048, measured)
            // becomes a shared read + one bfloat162 store per column pair.
            // Values are bit-identical to the epilogue's (same f32 words,
            // same __float2bfloat16). Reads race nothing: the panes are
            // stable until the restage barrier below.
            if (qb16 != nullptr) {
                for (uint32_t u = tid; u < C * 16u; u += 256u) {
                    const uint32_t r = u >> 4, c = (u & 15u) * 2u;
                    if (r < cl) {
                        // LAUNCH-chunk row space (== c0+r on legacy launches;
                        // on varlen launches c0 is the span's global row and
                        // the scratch copies stay chunk-padded)
                        const size_t dst =
                            ((size_t)(ch * C + r) * n_heads + h) * D + rd * 32u + c;
                        *(__nv_bfloat162*)(kb16 + dst) = __floats2bfloat162_rn(
                            (float)sh_sa[r * SL + c], (float)sh_sa[r * SL + c + 1u]);
                        *(__nv_bfloat162*)(qb16 + dst) = __floats2bfloat162_rn(
                            (float)sh_sb[r * SL + c], (float)sh_sb[r * SL + c + 1u]);
                    }
                }
            }
            if (rd < 3u) {
                __syncthreads();
                stage_slab(sh_sa, k, (rd + 1u) * 32u);
                stage_slab(sh_sb, q, (rd + 1u) * 32u);
                asm volatile("cp.async.commit_group;" ::: "memory");
                asm volatile("cp.async.wait_group 0;" ::: "memory");
                __syncthreads();
            }
        }
        __syncthreads();
        // emit: aqk warps write coef (decay ratio folded, j > i zeroed)
        // straight to global; akk warps write M = b_i ratio akk, j < i, to
        // the strict lower of sh_mt. Column-adjacent element pairs go out
        // as one aligned store; ratios use the reference-class __expf (args
        // <= 0 by construction - cg is non-increasing, j <= i).
#pragma unroll
        for (uint32_t nt = 0; nt < 8u; ++nt) {
#pragma unroll
            for (uint32_t ep = 0; ep < 2u; ++ep) {
                const uint32_t i = mt * 16u + g8 + (ep ? 8u : 0u);
                const uint32_t j = nt * 8u + 2u * t4;
                const float r0 =
                    j <= i ? __expf((float)(sh_cg[i] - sh_cg[j])) : 0.f;
                const float r1 =
                    j + 1u <= i ? __expf((float)(sh_cg[i] - sh_cg[j + 1u])) : 0.f;
                const float v0 = r0 * acc[nt][2u * ep];
                const float v1 = r1 * acc[nt][2u * ep + 1u];
                if (warp >= 4u)
                    *reinterpret_cast<pd_dns1_pair<AT>*>(
                        aqk + tb * C * C + i * C + j) =
                        pd_dns1_pair<AT>{(AT)v0, (AT)v1};
                else {
                    if (j < i) sh_mt[i * SMT +j] = sh_b[i] * v0;
                    if (j + 1u < i) sh_mt[i * SMT +j + 1u] = sh_b[i] * v1;
                }
            }
        }
    }
    __syncthreads();

    // T = (I+M)^-1, unit-lower, built hierarchically; strict-lower entries
    // pack into the strict UPPER triangle (T[i][j] at mt[j*C+i]).
    // Base: four 16x16 column-parallel solves, REGISTER-RESIDENT
    // (T-build door): the old per-column loop chained ~105
    // serially-dependent fma+LDS (element i re-read T[j][c] from shared) -
    // with only 2 warps active, the other 6 idled at the next barrier and
    // that wait, not the merge arithmetic, was the T-build's binding band.
    // Forward substitution per RHS column is column-independent, so column
    // c keeps its elements in a[e] (e = i-c-1): step s folds term j = c+s
    // into every e >= s, and the multiplier T[c+s][c] is this thread's own
    // a[s-1], final since step s-1. Same fma set, same per-element
    // j-ascending order -> bit-exact; the serial path shrinks to the
    // a[0]->a[14] chain and everything else is ILP. M loads go
    // conflict-free (addr stride 73 across lanes -> 32 distinct banks).
    if (tid < C) {
        const uint32_t cl0 = tid & 15u, c = (tid >> 4) * 16u + cl0;
        float a[15];
#pragma unroll
        for (uint32_t e = 0; e < 15u; ++e)
            a[e] = cl0 + 1u + e < 16u ? -sh_mt[(c + 1u + e) * SMT + c] : 0.f;
#pragma unroll
        for (uint32_t s = 1; s < 15u; ++s) {
            const float tv = a[s - 1u];
#pragma unroll
            for (uint32_t e = s; e < 15u; ++e)
                if (cl0 + 1u + e < 16u)
                    a[e] = fmaf(-sh_mt[(c + 1u + e) * SMT + c + s], tv, a[e]);
        }
#pragma unroll
        for (uint32_t e = 0; e < 15u; ++e)
            if (cl0 + 1u + e < 16u) sh_mt[c * SMT + c + 1u + e] = a[e];
    }
    __syncthreads();
    // merge 16->32 (pairs (0,1),(2,3)) then 32->64: W = M21 T11 into the sa
    // pane, then T21 = -T22 W. Restructured (T-build door): the
    // packed-T inner loops were per-lane triangular - divergent trip counts
    // (BSSY/BRA lines topped both merge buckets, together 19.9% of kernel
    // stalls) - and their T11 reads walked stride-SMT rows across lanes
    // (8c-bank pattern, 4-8-way replays). Each merge now first densifies
    // its T11 STRICT-lower into the dead front of the sa pane in a [j][c]
    // lane-consecutive layout (one-time, conflict-free consumption), then:
    //   W:   uniform full-trip unrolled j loop, per-instruction predication
    //        (j > c ? fmaf : keep) - the skipped terms are exact no-ops and
    //        the surviving partial-sum order is the old c+1..end order, so
    //        the result is bit-exact (incl. -0); the rr output rows ride as
    //        independent fma chains (ILP) off one shared T11d load.
    //   T21: telescoped segments ([base,i(0)), [i(0),i(1)), ...) - each rr
    //        chain covers exactly its old rj range in the old order, no
    //        predicates, no wasted terms; seeds from the W registers (this
    //        thread computed that word). A-operand reads are warp-uniform
    //        broadcasts, W reads lane-consecutive - conflict-free.
    {
        const uint32_t pair = tid >> 7, r = (tid >> 4) & 7u, c = tid & 15u;
        const uint32_t bg = pair * 32u, rg = bg + 16u + r;
        // densify: T11d16[b][j][c] = T[b*32+j][b*32+c] (strict lower)
        for (uint32_t u = tid; u < 512u; u += 256u) {
            const uint32_t b = u >> 8, j = (u >> 4) & 15u, cc = u & 15u;
            const float tv = sh_mt[(b * 32u + cc) * SMT + b * 32u + j];
            sh_w32[b * 272u + j * 17u + cc] = j > cc ? tv : 0.f;
        }
        __syncthreads();
        float w[2];
#pragma unroll
        for (uint32_t rr = 0; rr < 2u; ++rr)
            w[rr] = sh_mt[(rg + rr * 8u) * SMT + bg + c];
#pragma unroll
        for (uint32_t j = 0; j < 16u; ++j) {
            const float tv = sh_w32[pair * 272u + j * 17u + c];
#pragma unroll
            for (uint32_t rr = 0; rr < 2u; ++rr) {
                const float m = sh_mt[(rg + rr * 8u) * SMT + bg + j];
                w[rr] = j > c ? fmaf(m, tv, w[rr]) : w[rr];
            }
        }
#pragma unroll
        for (uint32_t rr = 0; rr < 2u; ++rr)
            sh_w32[(rg + rr * 8u) * 36u + c] = w[rr];
        __syncthreads();
        float t2[2] = {w[0], w[1]};
        for (uint32_t rj = bg + 16u; rj < rg; ++rj) {  // [base, i(0)): both
            const float wv = sh_w32[rj * 36u + c];
            t2[0] = fmaf(sh_mt[rj * SMT + rg], wv, t2[0]);
            t2[1] = fmaf(sh_mt[rj * SMT + rg + 8u], wv, t2[1]);
        }
#pragma unroll
        for (uint32_t kk = 0; kk < 8u; ++kk) {  // [i(0), i(1)): chain 1
            const uint32_t rj = rg + kk;
            t2[1] = fmaf(sh_mt[rj * SMT + rg + 8u], sh_w32[rj * 36u + c], t2[1]);
        }
        sh_mt[(bg + c) * SMT + rg] = -t2[0];
        sh_mt[(bg + c) * SMT + rg + 8u] = -t2[1];
    }
    __syncthreads();
    {
        const uint32_t r = tid >> 5, c = tid & 31u;
        // densify: T11d32[j][c] = T[j][c] (strict lower of the 32x32 head)
        for (uint32_t u = tid; u < 1024u; u += 256u) {
            const uint32_t j = u >> 5, cc = u & 31u;
            const float tv = sh_mt[cc * SMT + j];
            sh_w32[j * 33u + cc] = j > cc ? tv : 0.f;
        }
        __syncthreads();
        float w[4];
#pragma unroll
        for (uint32_t rr = 0; rr < 4u; ++rr)
            w[rr] = sh_mt[(32u + r + rr * 8u) * SMT + c];
#pragma unroll
        for (uint32_t j = 0; j < 32u; ++j) {
            const float tv = sh_w32[j * 33u + c];
#pragma unroll
            for (uint32_t rr = 0; rr < 4u; ++rr) {
                const float m = sh_mt[(32u + r + rr * 8u) * SMT + j];
                w[rr] = j > c ? fmaf(m, tv, w[rr]) : w[rr];
            }
        }
#pragma unroll
        for (uint32_t rr = 0; rr < 4u; ++rr)
            sh_w32[(32u + r + rr * 8u) * 36u + c] = w[rr];
        __syncthreads();
        float tt4[4] = {w[0], w[1], w[2], w[3]};
        for (uint32_t rj = 32u; rj < 32u + r; ++rj) {  // [32, i(0)): all 4
            const float wv = sh_w32[rj * 36u + c];
#pragma unroll
            for (uint32_t rr = 0; rr < 4u; ++rr)
                tt4[rr] = fmaf(sh_mt[rj * SMT + 32u + r + rr * 8u], wv, tt4[rr]);
        }
#pragma unroll
        for (uint32_t s = 0; s < 3u; ++s)
#pragma unroll
            for (uint32_t kk = 0; kk < 8u; ++kk) {  // [i(s), i(s+1)): rr > s
                const uint32_t rj = 32u + r + s * 8u + kk;
                const float wv = sh_w32[rj * 36u + c];
#pragma unroll
                for (uint32_t rr = s + 1u; rr < 4u; ++rr)
                    tt4[rr] = fmaf(sh_mt[rj * SMT + 32u + r + rr * 8u], wv, tt4[rr]);
            }
#pragma unroll
        for (uint32_t rr = 0; rr < 4u; ++rr)
            sh_mt[c * SMT + 32u + r + rr * 8u] = -tt4[rr];
    }
    __syncthreads();

    // dw = T (bg.K) and du = T (b.V), n(=a)-slabbed in four rounds. A = T
    // read PACKED from sh_mt with predicates; B = the slab with the
    // per-token scale folded at the load. Warp w: m-tile w&3, n-tile pair
    // (w>>2)*2 within the round's four n-tiles.
    for (uint32_t pass = 0; pass < 2u; ++pass) {
        const float* scale = pass == 0u ? sh_bg : sh_b;
        OT* dst = pass == 0u ? dw : du;
        constexpr uint32_t SLV = sizeof(VT) == 2 ? 40u : 36u;
        const uint32_t mt = warp & 3u, np = (warp >> 2) * 2u;
        // sa/sb alternate as a 2-deep ring (sb is free after the dots): the
        // next round's slab stages during this round's mma - the 4 staging
        // stalls per pass collapse to the first. Same bytes, same order.
        if (pass == 0u) stage_slab(sh_sa, k, 0u);
        else stage_slab((VT*)sh_sa, v, 0u);
        asm volatile("cp.async.commit_group;" ::: "memory");
        for (uint32_t rd = 0; rd < 4u; ++rd) {
            IT* pane = (rd & 1u) ? sh_sb : sh_sa;
            if (rd < 3u) {
                if (pass == 0u)
                    stage_slab((rd & 1u) ? sh_sa : sh_sb, k, (rd + 1u) * 32u);
                else
                    stage_slab((VT*)((rd & 1u) ? sh_sa : sh_sb), v, (rd + 1u) * 32u);
            }
            asm volatile("cp.async.commit_group;" ::: "memory");
            asm volatile("cp.async.wait_group 1;" ::: "memory");
            __syncthreads();
            float acc[2][4];
#pragma unroll
            for (uint32_t nn = 0; nn < 2u; ++nn)
#pragma unroll
                for (uint32_t e = 0; e < 4; ++e) acc[nn][e] = 0.f;
#pragma unroll
            for (uint32_t kk = 0; kk < C; kk += 8u) {
                const uint32_t i0r = mt * 16u + g8, j0 = kk + t4;
                float ar[4];
                ar[0] = j0 < i0r ? sh_mt[j0 * SMT +i0r] : (j0 == i0r ? 1.f : 0.f);
                ar[1] = j0 < i0r + 8u ? sh_mt[j0 * SMT +i0r + 8u]
                                      : (j0 == i0r + 8u ? 1.f : 0.f);
                ar[2] = j0 + 4u < i0r ? sh_mt[(j0 + 4u) * SMT +i0r]
                                      : (j0 + 4u == i0r ? 1.f : 0.f);
                ar[3] = j0 + 4u < i0r + 8u ? sh_mt[(j0 + 4u) * SMT +i0r + 8u]
                                           : (j0 + 4u == i0r + 8u ? 1.f : 0.f);
                const float s0 = scale[kk + t4], s4 = scale[kk + t4 + 4u];
#pragma unroll
                for (uint32_t nn = 0; nn < 2u; ++nn) {
                    float br[2];
                    br[0] = s0 * (pass == 0u
                        ? (float)pane[(kk + t4) * SL + (np + nn) * 8u + g8]
                        : (float)((const VT*)pane)[(kk + t4) * SLV + (np + nn) * 8u + g8]);
                    br[1] = s4 * (pass == 0u
                        ? (float)pane[(kk + t4 + 4u) * SL + (np + nn) * 8u + g8]
                        : (float)((const VT*)pane)[(kk + t4 + 4u) * SLV + (np + nn) * 8u + g8]);
                    pd_dnm_mma_sw<PREC>(acc[nn], ar, br);
                }
            }
#pragma unroll
            for (uint32_t nn = 0; nn < 2u; ++nn) {
#pragma unroll
                for (uint32_t ep = 0; ep < 2u; ++ep) {
                    const uint32_t i = mt * 16u + g8 + (ep ? 8u : 0u);
                    const uint32_t a = rd * 32u + (np + nn) * 8u + 2u * t4;
                    if (i < cl)
                        *reinterpret_cast<pd_dns1_pair<OT>*>(
                            dst + (tb * C + i) * D + a) =
                            pd_dns1_pair<OT>{(OT)acc[nn][2u * ep],
                                             (OT)acc[nn][2u * ep + 1u]};
                }
            }
            __syncthreads();
        }
    }
}

