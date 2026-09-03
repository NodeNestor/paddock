// NVFP4 MoE consumers over the TILED expert-plane layout (the skinny-tile
// pair). A mechanism probe left one candidate for marlin's per-byte edge on
// the c32 decode MoE band: the tile SHAPE.
// Pricing it out left every arm BIT-EXACT vs the
// shipped bs pair: the winning axis is the LAYOUT (contiguous per-stage
// spans -> 512 B-class warp loads; the shipped row-major pair reads 128 B
// rows at ~1.4 KB stride), with the skinny BM=8 tile second and ring depth
// a wash (ST 2/3/4 within +-0.5%). At the c32-realistic uniq 96 the pair
// runs 1461-1474 GB/s-wt (92-93% of the 1.58 TB/s practical roof) vs the
// shipped 1280/1309 (81/83%) - pair time -11.7%..-17.3% across uniq 24-128,
// ~= the full ~1.1 ms/tick prize at marlin's per-byte rate. Persistent
// grids and deep 1-CTA rings stay dead; these are ordinary
// multi-wave grids, 2-stage ring, 2 CTAs/SM.
//
// TILED PLANE LAYOUT (the engine's nvf4_moe_upload_tiled contract; both
// nemotron planes tile exactly - 1856 = 29*64 rows, 2688 = 42*64 - so no
// pad bytes exist and DRAM traffic is identical to row-major):
//   data : [e][rt][ks][piece 2][row 64][16 B]   (2048 B per (e,rt,ks) block)
//   scale: [e][rt][ks][row 64][4 B]             (256 B per block)
// rt = 64-row output tile, ks = 64-element K block (piece = its 32-element
// half). A K-chunk fetch for one row tile is one contiguous span (data) plus
// one for scales; ldmatrix reads rows at 16 B stride (4-bank steps, conflict
// free, better than the row-major WROW=144 stride).
//
// Three consumer classes, one layout (a tiled plane must never exist without
// every class able to read it - the lm_head has_nvf4_tm law):
//   - _st  (BM=8, 64 rows, 128 thr): the DECODE pair. pairs/uniq at c32 is
//     ~2.4, so 32-wide blocks are ~7.5% live; 8-wide quarters the Y/fq
//     staging and the mma count. Fed by pd_moe_align_bm(bm=8).
//   - _stw (BM=32, 128 rows, 256 thr): the PREFILL pair, same geometry class
//     as the shipped bs pair (full 32-token blocks; BM=8 there would re-read
//     each expert strip per 8 tokens). Same align as today.
//   - _mtt: the r=1 serial-decode GEMV twins (W4A16 f32 class, same numeric
//     class as the mt pair). CTA per 16-row group, K split by warp, lane =
//     (row, piece) so a warp's ks-block read is 512 B contiguous. The
//     grouping (not the math) differs from mt, so its gates are the mt
//     class: rel-to-rms vs the row-major twin + determinism bit-gated.
// Accumulate order in _st/_stw is the shipped pair's (kt asc, k64 asc)
// verbatim -> both are BIT-EXACT vs the row-major bs pair on identical
// routing; the unit gates lean on that.
// All six launchers are cc12-only (block-scale mma / e4m3 decode on the
// tiled layout); every other die keeps row-major planes and the shipped
// consumers, which is what the engine's layout election checks.

#define PD_STT_KB 8u                    // KC = 256 elements per ring stage
#define PD_STT_K64S (PD_STT_KB / 2u)    // ks blocks per stage
#define PD_STT_TDATA (PD_STT_K64S * 2048u)   // per-64-row-tile stage bytes
#define PD_STT_TSCL (PD_STT_K64S * 256u)
#define PD_STT_YROW (16u + PD_STT_KB * 16u)
#define PD_STT_STAGE(BM, ROWS) \
    ((ROWS / 64u) * (PD_STT_TDATA + PD_STT_TSCL) + (BM) * PD_STT_YROW)
#define PD_STT_SMEM(BM, ROWS) (2u * PD_STT_STAGE(BM, ROWS))

// Sorted-tile expert up + squared-relu + nvf4 requantize over the tiled
// plane. Geometry: CTA = ROWS output rows x BM token columns, ROWS*2
// threads, each warp owns one 16-row m-tile across all BM columns. 2-stage
// cp.async ring with the scale bytes folded into the ring (pd_cpa4p) - the
// stage is contiguous so there is no strided-scale tax to defer.
template <uint32_t BM, uint32_t ROWS>
__global__ void __launch_bounds__(ROWS * 2u, 2) pd_nv4st_up_kernel(
    const uint8_t* __restrict__ data, const uint8_t* __restrict__ scale,
    const float* __restrict__ scale2, const uint32_t* __restrict__ sorted_row,
    const uint32_t* __restrict__ block_expert, const uint8_t* __restrict__ xq,
    const uint8_t* __restrict__ xs, uint8_t* __restrict__ fq,
    uint8_t* __restrict__ fs, uint32_t in_dim, uint32_t ff) {
#if PD_BS_OK
    constexpr uint32_t THR = ROWS * 2u;
    constexpr uint32_t YROW = PD_STT_YROW;
    constexpr uint32_t K64S = PD_STT_K64S;
    constexpr uint32_t STAGE = PD_STT_STAGE(BM, ROWS);
    const uint32_t blk = blockIdx.x;
    const uint32_t e = block_expert[blk];
    if (e == PD_MOE_PAD) return;
    const uint32_t row_base = blockIdx.y * ROWS;

    extern __shared__ unsigned char pd_stt_sh[];
    __shared__ uint32_t tok[BM];

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u, warp = tid >> 5;
    const uint32_t g = lane >> 2, tq = lane & 3u;
    const uint32_t i0 = warp * 16u;
    const uint32_t n_kb = in_dim >> 5;
    const uint32_t n_k16 = in_dim >> 4;
    const uint32_t nks = in_dim >> 6;
    const uint32_t nrt = ff >> 6;   // exact: the layout requires ff % 64 == 0
    const uint32_t nk = (in_dim + PD_STT_KB * 32u - 1u) / (PD_STT_KB * 32u);
    const size_t trt0 = ((size_t)e * nrt + (row_base >> 6)) * nks;

    if (tid < BM) tok[tid] = sorted_row[(size_t)blk * BM + tid];
    __syncthreads();

    float acc[BM / 8u][4] = {};

    #define PD_STT_ISSUE_W(dst, kt)                                                   \
        for (uint32_t u = tid; u < (ROWS / 64u) * (PD_STT_TDATA / 16u); u += THR) {   \
            const uint32_t h = u / (PD_STT_TDATA / 16u), v = u % (PD_STT_TDATA / 16u);\
            const uint32_t ks = (kt) * K64S + v / 128u;                               \
            const bool ok = ks < nks && (row_base >> 6) + h < nrt;                    \
            pd_cp_async16(                                                            \
                (int*)((dst) + h * (PD_STT_TDATA + PD_STT_TSCL) + v * 16u),           \
                data + (trt0 + h * (size_t)nks + (kt) * K64S) * 2048u + v * 16u,      \
                ok);                                                                  \
        }                                                                             \
        for (uint32_t u = tid; u < (ROWS / 64u) * (PD_STT_TSCL / 4u); u += THR) {     \
            const uint32_t h = u / (PD_STT_TSCL / 4u), v = u % (PD_STT_TSCL / 4u);    \
            const uint32_t ks = (kt) * K64S + v / 64u;                                \
            const bool ok = ks < nks && (row_base >> 6) + h < nrt;                    \
            pd_cpa4p((dst) + h * (PD_STT_TDATA + PD_STT_TSCL) + PD_STT_TDATA + v * 4u,\
                     scale + (trt0 + h * (size_t)nks + (kt) * K64S) * 256u + v * 4u,  \
                     ok);                                                             \
        }
    #define PD_STT_ISSUE_Y(dst, kt)                                                   \
        for (uint32_t u = tid; u < BM * PD_STT_KB; u += THR) {                        \
            const uint32_t col = u / PD_STT_KB, seg = u % PD_STT_KB;                  \
            const uint32_t r = tok[col];                                              \
            const bool ok = r != PD_MOE_PAD && (kt) * PD_STT_KB + seg < n_kb;         \
            pd_cp_async16((int*)((dst) + col * YROW + 16u + seg * 16u),               \
                          xq + ((size_t)(ok ? r : 0u) * in_dim >> 1) +                \
                              (kt) * (PD_STT_KB * 16u) + seg * 16u,                   \
                          ok);                                                        \
        }                                                                             \
        for (uint32_t u = tid; u < BM * (PD_STT_KB / 2u); u += THR) {                 \
            const uint32_t col = u / (PD_STT_KB / 2u), q = u % (PD_STT_KB / 2u);      \
            const uint32_t r = tok[col];                                              \
            const bool ok = r != PD_MOE_PAD &&                                        \
                            (kt) * (PD_STT_KB * 2u) + q * 4u + 4u <= n_k16;           \
            pd_cpa4p((dst) + col * YROW + q * 4u,                                     \
                     xs + (size_t)(ok ? r : 0u) * n_k16 +                             \
                         (kt) * (PD_STT_KB * 2u) + q * 4u,                            \
                     ok);                                                             \
        }
    #define PD_STT_WBUF(s) (pd_stt_sh + ((s) & 1u) * STAGE)
    #define PD_STT_YBUF(s) (PD_STT_WBUF(s) + (ROWS / 64u) * (PD_STT_TDATA + PD_STT_TSCL))

    PD_STT_ISSUE_W(PD_STT_WBUF(0), 0u)
    PD_STT_ISSUE_Y(PD_STT_YBUF(0), 0u)
    asm volatile("cp.async.commit_group;");
    for (uint32_t kt = 0; kt < nk; ++kt) {
        unsigned char* tw = PD_STT_WBUF(kt);
        unsigned char* ty = PD_STT_YBUF(kt);
        if (kt + 1u < nk) {
            PD_STT_ISSUE_W(PD_STT_WBUF(kt + 1u), kt + 1u)
            PD_STT_ISSUE_Y(PD_STT_YBUF(kt + 1u), kt + 1u)
            asm volatile("cp.async.commit_group;");
            asm volatile("cp.async.wait_group 1;");
        } else {
            asm volatile("cp.async.wait_group 0;");
        }
        __syncthreads();

        uint32_t am[K64S][4], sa[K64S];
        const uint32_t rl = ((lane >> 3) & 1u) * 8u + (lane & 7u);
        const uint32_t pl = lane >> 4;
        const uint32_t rs = (tq & 1u) ? (i0 & 63u) + g + 8u : (i0 & 63u) + g;
        const uint32_t h = i0 >> 6;
        #pragma unroll
        for (uint32_t k64 = 0; k64 < K64S; ++k64) {
            pd_ldm_x4(am[k64], tw + h * (PD_STT_TDATA + PD_STT_TSCL) +
                                   (k64 * 2u + pl) * 1024u +
                                   ((i0 & 63u) + rl) * 16u);
            sa[k64] = *(const uint32_t*)(tw + h * (PD_STT_TDATA + PD_STT_TSCL) +
                                         PD_STT_TDATA + k64 * 256u + rs * 4u);
        }
        #pragma unroll
        for (uint32_t j0 = 0; j0 < BM; j0 += 8u) {
            uint32_t bm[2u * K64S];
            #pragma unroll
            for (uint32_t q = 0; q < PD_STT_KB / 4u; ++q)
                pd_ldm_x4(bm + q * 4u, ty + (j0 + (lane & 7u)) * YROW + 16u +
                                           q * 64u + (lane >> 3) * 16u);
            const unsigned char* ysr = ty + (j0 + g) * YROW;
            #pragma unroll
            for (uint32_t k64 = 0; k64 < K64S; ++k64) {
                const uint32_t sb = *(const uint32_t*)(ysr + k64 * 4u);
                pd_nv4_mma(acc[j0 >> 3], am[k64][0], am[k64][1], am[k64][2],
                           am[k64][3], bm[k64 * 2u], bm[k64 * 2u + 1u],
                           sa[k64], sb);
            }
        }
        __syncthreads();
    }
    #undef PD_STT_ISSUE_W
    #undef PD_STT_ISSUE_Y
    #undef PD_STT_WBUF
    #undef PD_STT_YBUF

    // epilogue: the shipped bs kernel's per-16-along-ff quantize, verbatim
    // math - one 16-row block per warp, 8 token columns per j0 group.
    const float s2 = scale2[e];
    const uint32_t tmask = 0x11111111u << tq;
    const uint32_t rb = row_base + i0;
    #pragma unroll
    for (uint32_t j0 = 0; j0 < BM; j0 += 8u) {
        #pragma unroll
        for (uint32_t qc = 0; qc < 2u; ++qc) {
            const uint32_t c = j0 + 2u * tq + qc;
            const bool pad = tok[c] == PD_MOE_PAD;
            const float a0 = acc[j0 >> 3][qc] * s2;
            const float a1 = acc[j0 >> 3][qc + 2u] * s2;
            const float r0v = fmaxf(a0, 0.0f);
            const float r1v = fmaxf(a1, 0.0f);
            const float v0 = pad ? 0.0f : r0v * r0v;
            const float v1 = pad ? 0.0f : r1v * r1v;
            float a = fmaxf(v0, v1);
            a = fmaxf(a, __shfl_xor_sync(tmask, a, 4));
            a = fmaxf(a, __shfl_xor_sync(tmask, a, 8));
            a = fmaxf(a, __shfl_xor_sync(tmask, a, 16));
            float inv;
            const unsigned sbyte = pd_nvf4_scale(a, &inv);
            const uint32_t n0 = pd_e2m1_rn(v0 * inv);
            const uint32_t n1 = pd_e2m1_rn(v1 * inv);
            const uint32_t m = (g & 3u) * 2u;
            const uint32_t lo0 = __shfl_sync(0xffffffffu, n0, m * 4u + tq);
            const uint32_t hi0 = __shfl_sync(0xffffffffu, n0, (m + 1u) * 4u + tq);
            const uint32_t lo1 = __shfl_sync(0xffffffffu, n1, m * 4u + tq);
            const uint32_t hi1 = __shfl_sync(0xffffffffu, n1, (m + 1u) * 4u + tq);
            const uint32_t lo = (g < 4u) ? lo0 : lo1;
            const uint32_t hi = (g < 4u) ? hi0 : hi1;
            if (rb < ff) {
                const size_t srow = (size_t)blk * BM + c;
                fq[srow * (ff >> 1) + (rb >> 1) + g] =
                    (unsigned char)(lo | (hi << 4));
                if (g == 0)
                    fs[srow * (ff >> 4) + (rb >> 4)] = (unsigned char)sbyte;
            }
        }
    }
#else
    (void)data; (void)scale; (void)scale2; (void)sorted_row; (void)block_expert;
    (void)xq; (void)xs; (void)fq; (void)fs; (void)in_dim; (void)ff;
#endif
}

// Down + weighted scatter over the tiled plane. B = fq/fs by sorted position
// ([nb*BM, ff/2], the bs contract at this BM); W = down plane, K = ff.
template <uint32_t BM, uint32_t ROWS>
__global__ void __launch_bounds__(ROWS * 2u, 2) pd_nv4st_dn_kernel(
    const uint8_t* __restrict__ data, const uint8_t* __restrict__ scale,
    const float* __restrict__ scale2, const uint32_t* __restrict__ sorted_row,
    const uint32_t* __restrict__ sorted_slot,
    const uint32_t* __restrict__ block_expert, const float* __restrict__ topk_w,
    const uint8_t* __restrict__ fq, const uint8_t* __restrict__ fs,
    float* __restrict__ part, uint32_t ff, uint32_t embd, uint32_t kw,
    uint32_t np, uint32_t slot_off) {
#if PD_BS_OK
    constexpr uint32_t THR = ROWS * 2u;
    constexpr uint32_t YROW = PD_STT_YROW;
    constexpr uint32_t K64S = PD_STT_K64S;
    constexpr uint32_t STAGE = PD_STT_STAGE(BM, ROWS);
    const uint32_t blk = blockIdx.x;
    const uint32_t e = block_expert[blk];
    if (e == PD_MOE_PAD) return;
    const uint32_t row_base = blockIdx.y * ROWS;

    extern __shared__ unsigned char pd_stt_sh[];

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u, warp = tid >> 5;
    const uint32_t g = lane >> 2, tq = lane & 3u;
    const uint32_t i0 = warp * 16u;
    const uint32_t n_kb = ff >> 5;
    const uint32_t n_k16 = ff >> 4;
    const uint32_t nks = ff >> 6;
    const uint32_t nrt = embd >> 6;
    const uint32_t nk = (ff + PD_STT_KB * 32u - 1u) / (PD_STT_KB * 32u);
    const size_t trt0 = ((size_t)e * nrt + (row_base >> 6)) * nks;

    float acc[BM / 8u][4] = {};

    #define PD_STT_ISSUE_W(dst, kt)                                                   \
        for (uint32_t u = tid; u < (ROWS / 64u) * (PD_STT_TDATA / 16u); u += THR) {   \
            const uint32_t h = u / (PD_STT_TDATA / 16u), v = u % (PD_STT_TDATA / 16u);\
            const uint32_t ks = (kt) * K64S + v / 128u;                               \
            const bool ok = ks < nks && (row_base >> 6) + h < nrt;                    \
            pd_cp_async16(                                                            \
                (int*)((dst) + h * (PD_STT_TDATA + PD_STT_TSCL) + v * 16u),           \
                data + (trt0 + h * (size_t)nks + (kt) * K64S) * 2048u + v * 16u,      \
                ok);                                                                  \
        }                                                                             \
        for (uint32_t u = tid; u < (ROWS / 64u) * (PD_STT_TSCL / 4u); u += THR) {     \
            const uint32_t h = u / (PD_STT_TSCL / 4u), v = u % (PD_STT_TSCL / 4u);    \
            const uint32_t ks = (kt) * K64S + v / 64u;                                \
            const bool ok = ks < nks && (row_base >> 6) + h < nrt;                    \
            pd_cpa4p((dst) + h * (PD_STT_TDATA + PD_STT_TSCL) + PD_STT_TDATA + v * 4u,\
                     scale + (trt0 + h * (size_t)nks + (kt) * K64S) * 256u + v * 4u,  \
                     ok);                                                             \
        }
    #define PD_STT_ISSUE_Y(dst, kt)                                                   \
        for (uint32_t u = tid; u < BM * PD_STT_KB; u += THR) {                        \
            const uint32_t col = u / PD_STT_KB, seg = u % PD_STT_KB;                  \
            const bool ok = (kt) * PD_STT_KB + seg < n_kb;                            \
            pd_cp_async16((int*)((dst) + col * YROW + 16u + seg * 16u),               \
                          fq + ((size_t)blk * BM + col) * (size_t)(ff >> 1) +         \
                              (kt) * (PD_STT_KB * 16u) + seg * 16u,                   \
                          ok);                                                        \
        }                                                                             \
        for (uint32_t u = tid; u < BM * (PD_STT_KB / 2u); u += THR) {                 \
            const uint32_t col = u / (PD_STT_KB / 2u), q = u % (PD_STT_KB / 2u);      \
            const bool ok = (kt) * (PD_STT_KB * 2u) + q * 4u + 4u <= n_k16;           \
            pd_cpa4p((dst) + col * YROW + q * 4u,                                     \
                     fs + ((size_t)blk * BM + col) * n_k16 +                          \
                         (kt) * (PD_STT_KB * 2u) + q * 4u,                            \
                     ok);                                                             \
        }
    #define PD_STT_WBUF(s) (pd_stt_sh + ((s) & 1u) * STAGE)
    #define PD_STT_YBUF(s) (PD_STT_WBUF(s) + (ROWS / 64u) * (PD_STT_TDATA + PD_STT_TSCL))

    PD_STT_ISSUE_W(PD_STT_WBUF(0), 0u)
    PD_STT_ISSUE_Y(PD_STT_YBUF(0), 0u)
    asm volatile("cp.async.commit_group;");
    for (uint32_t kt = 0; kt < nk; ++kt) {
        unsigned char* tw = PD_STT_WBUF(kt);
        unsigned char* ty = PD_STT_YBUF(kt);
        if (kt + 1u < nk) {
            PD_STT_ISSUE_W(PD_STT_WBUF(kt + 1u), kt + 1u)
            PD_STT_ISSUE_Y(PD_STT_YBUF(kt + 1u), kt + 1u)
            asm volatile("cp.async.commit_group;");
            asm volatile("cp.async.wait_group 1;");
        } else {
            asm volatile("cp.async.wait_group 0;");
        }
        __syncthreads();

        uint32_t am[K64S][4], sa[K64S];
        const uint32_t rl = ((lane >> 3) & 1u) * 8u + (lane & 7u);
        const uint32_t pl = lane >> 4;
        const uint32_t rs = (tq & 1u) ? (i0 & 63u) + g + 8u : (i0 & 63u) + g;
        const uint32_t h = i0 >> 6;
        #pragma unroll
        for (uint32_t k64 = 0; k64 < K64S; ++k64) {
            pd_ldm_x4(am[k64], tw + h * (PD_STT_TDATA + PD_STT_TSCL) +
                                   (k64 * 2u + pl) * 1024u +
                                   ((i0 & 63u) + rl) * 16u);
            sa[k64] = *(const uint32_t*)(tw + h * (PD_STT_TDATA + PD_STT_TSCL) +
                                         PD_STT_TDATA + k64 * 256u + rs * 4u);
        }
        #pragma unroll
        for (uint32_t j0 = 0; j0 < BM; j0 += 8u) {
            uint32_t bm[2u * K64S];
            #pragma unroll
            for (uint32_t q = 0; q < PD_STT_KB / 4u; ++q)
                pd_ldm_x4(bm + q * 4u, ty + (j0 + (lane & 7u)) * YROW + 16u +
                                           q * 64u + (lane >> 3) * 16u);
            const unsigned char* ysr = ty + (j0 + g) * YROW;
            #pragma unroll
            for (uint32_t k64 = 0; k64 < K64S; ++k64) {
                const uint32_t sb = *(const uint32_t*)(ysr + k64 * 4u);
                pd_nv4_mma(acc[j0 >> 3], am[k64][0], am[k64][1], am[k64][2],
                           am[k64][3], bm[k64 * 2u], bm[k64 * 2u + 1u],
                           sa[k64], sb);
            }
        }
        __syncthreads();
    }
    #undef PD_STT_ISSUE_W
    #undef PD_STT_ISSUE_Y
    #undef PD_STT_WBUF
    #undef PD_STT_YBUF

    // weighted scatter to the per-(token, slot) partial rows - the shipped
    // down epilogue with the j0 groups walked 8 columns at a time.
    const float s2 = scale2[e];
    #pragma unroll
    for (uint32_t j0 = 0; j0 < BM; j0 += 8u) {
        #pragma unroll
        for (uint32_t qc = 0; qc < 2u; ++qc) {
            const uint32_t c = j0 + 2u * tq + qc;
            const uint32_t t = sorted_row[(size_t)blk * BM + c];
            if (t == PD_MOE_PAD) continue;
            const uint32_t slt = sorted_slot[(size_t)blk * BM + c];
            const float w = (topk_w ? topk_w[(size_t)t * kw + slt] : 1.0f) * s2;
            float* prow = part + ((size_t)t * np + slt + slot_off) * embd;
            const uint32_t r0 = row_base + i0 + g;
            const uint32_t r8 = r0 + 8u;
            if (r0 < embd) prow[r0] = acc[j0 >> 3][qc] * w;
            if (r8 < embd) prow[r8] = acc[j0 >> 3][qc + 2u] * w;
        }
    }
#else
    (void)data; (void)scale; (void)scale2; (void)sorted_row; (void)sorted_slot;
    (void)block_expert; (void)topk_w; (void)fq; (void)fs; (void)part; (void)ff;
    (void)embd; (void)kw; (void)np; (void)slot_off;
#endif
}

// ---- r=1 serial-decode twins over the tiled plane (the mt class) ----------
// Same numeric class as pd_nvf4_moe_up_relu2_mt / down_part (W4A16: f32
// activations, pd_nvf4_dot4w per element quad, relu^2 / weighted partial
// epilogues). The GROUPING differs: CTA per 16-row group (a task-per-row
// GEMV on the tiled layout would read 16 B at 1 KB stride - the reason
// gemv_batch_tf regrouped the lm_head). 128 threads; warp w owns ks blocks
// w, w+4, ...; lane = (row = lane>>1, piece = lane&1) so each warp's
// ks-block read is 512 B contiguous. Per-lane K order ascends; combine is
// fixed-order (piece pair -> ascending warps), deterministic.
__global__ void pd_nv4st_mt_up_kernel(
    const uint8_t* __restrict__ rdata, const uint8_t* __restrict__ rscale,
    const float* __restrict__ rscale2, const uint8_t* __restrict__ sdata,
    const uint8_t* __restrict__ sscale, const float* __restrict__ sscale2,
    const uint32_t* __restrict__ idx, const float* __restrict__ x,
    float* __restrict__ act, uint32_t in_dim, uint32_t ff_r, uint32_t ff_s,
    uint32_t k) {
#if PD_NV4_OK
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const uint32_t gr_r = ff_r >> 4, gr_s = ff_s >> 4;
    const uint32_t task = blockIdx.x;
    if (task >= k * gr_r + gr_s) return;
    const uint8_t* base;
    const uint8_t* sbase;
    float s2;
    uint32_t rg, nrt, aoff;
    if (task < k * gr_r) {
        const uint32_t slot = task / gr_r;
        rg = task - slot * gr_r;
        const uint32_t e = idx[slot];
        nrt = ff_r >> 6;
        const uint32_t nks = in_dim >> 6;
        base = rdata + (size_t)e * nrt * nks * 2048u;
        sbase = rscale + (size_t)e * nrt * nks * 256u;
        s2 = rscale2[e];
        aoff = slot * ff_r;
    } else {
        rg = task - k * gr_r;
        nrt = ff_s >> 6;
        base = sdata;
        sbase = sscale;
        s2 = sscale2[0];
        aoff = k * ff_r;
    }
    const uint32_t nks = in_dim >> 6;
    const uint32_t rt = rg >> 2, r0 = (rg & 3u) * 16u;
    const uint32_t r6 = r0 + (lane >> 1), p = lane & 1u;

    float acc = 0.0f;
    for (uint32_t ks = warp; ks < nks; ks += 4u) {
        const size_t blk = (size_t)rt * nks + ks;
        const uint4 wv = *reinterpret_cast<const uint4*>(
            base + blk * 2048u + p * 1024u + r6 * 16u);
        const uint32_t sw = *reinterpret_cast<const uint32_t*>(
            sbase + blk * 256u + r6 * 4u);
        const uint32_t s0 = (sw >> (p * 16u)) & 0xFFu;
        const uint32_t s1 = (sw >> (p * 16u + 8u)) & 0xFFu;
        const uint32_t e0 = ks * 64u + p * 32u;
        acc += pd_nvf4_dot4w(wv.x & 0xFFFFu, s0, x, e0);
        acc += pd_nvf4_dot4w(wv.x >> 16, s0, x, e0 + 4u);
        acc += pd_nvf4_dot4w(wv.y & 0xFFFFu, s0, x, e0 + 8u);
        acc += pd_nvf4_dot4w(wv.y >> 16, s0, x, e0 + 12u);
        acc += pd_nvf4_dot4w(wv.z & 0xFFFFu, s1, x, e0 + 16u);
        acc += pd_nvf4_dot4w(wv.z >> 16, s1, x, e0 + 20u);
        acc += pd_nvf4_dot4w(wv.w & 0xFFFFu, s1, x, e0 + 24u);
        acc += pd_nvf4_dot4w(wv.w >> 16, s1, x, e0 + 28u);
    }
    // piece pair first (lane, lane^1), then ascending warps through shared -
    // fixed summation order per output row.
    acc += __shfl_xor_sync(0xffffffffu, acc, 1);
    __shared__ float psum[4][16];
    if (p == 0) psum[warp][lane >> 1] = acc;
    __syncthreads();
    if (threadIdx.x < 16u) {
        const float total = ((psum[0][threadIdx.x] + psum[1][threadIdx.x]) +
                             psum[2][threadIdx.x]) + psum[3][threadIdx.x];
        const float v = fmaxf(total * s2, 0.0f);
        act[aoff + rg * 16u + threadIdx.x] = v * v;
    }
#else
    (void)rdata; (void)rscale; (void)rscale2; (void)sdata; (void)sscale;
    (void)sscale2; (void)idx; (void)x; (void)act; (void)in_dim; (void)ff_r;
    (void)ff_s; (void)k;
#endif
}

__global__ void pd_nv4st_mt_dn_kernel(
    const uint8_t* __restrict__ rdata, const uint8_t* __restrict__ rscale,
    const float* __restrict__ rscale2, const uint8_t* __restrict__ sdata,
    const uint8_t* __restrict__ sscale, const float* __restrict__ sscale2,
    const uint32_t* __restrict__ idx, const float* __restrict__ topk_w,
    const float* __restrict__ act, float* __restrict__ part, uint32_t ff_r,
    uint32_t ff_s, uint32_t embd, uint32_t k) {
#if PD_NV4_OK
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const uint32_t gr_e = embd >> 4;
    const uint32_t task = blockIdx.x;
    if (task >= (k + 1u) * gr_e) return;
    const uint32_t slot = task / gr_e, rg = task - slot * gr_e;
    const uint8_t* base;
    const uint8_t* sbase;
    const float* xrow;
    float w;
    uint32_t kk;
    const uint32_t nrt = embd >> 6;
    if (slot < k) {
        const uint32_t e = idx[slot];
        w = topk_w[slot] * rscale2[e];
        kk = ff_r;
        const uint32_t nks = kk >> 6;
        base = rdata + (size_t)e * nrt * nks * 2048u;
        sbase = rscale + (size_t)e * nrt * nks * 256u;
        xrow = act + (size_t)slot * ff_r;
    } else {
        w = sscale2[0];
        kk = ff_s;
        base = sdata;
        sbase = sscale;
        xrow = act + (size_t)k * ff_r;
    }
    const uint32_t nks = kk >> 6;
    const uint32_t rt = rg >> 2, r0 = (rg & 3u) * 16u;
    const uint32_t r6 = r0 + (lane >> 1), p = lane & 1u;

    float psm = 0.0f;
    for (uint32_t ks = warp; ks < nks; ks += 4u) {
        const size_t blk = (size_t)rt * nks + ks;
        const uint4 wv = *reinterpret_cast<const uint4*>(
            base + blk * 2048u + p * 1024u + r6 * 16u);
        const uint32_t sw = *reinterpret_cast<const uint32_t*>(
            sbase + blk * 256u + r6 * 4u);
        const uint32_t s0 = (sw >> (p * 16u)) & 0xFFu;
        const uint32_t s1 = (sw >> (p * 16u + 8u)) & 0xFFu;
        const uint32_t e0 = ks * 64u + p * 32u;
        psm += pd_nvf4_dot4w(wv.x & 0xFFFFu, s0, xrow, e0);
        psm += pd_nvf4_dot4w(wv.x >> 16, s0, xrow, e0 + 4u);
        psm += pd_nvf4_dot4w(wv.y & 0xFFFFu, s0, xrow, e0 + 8u);
        psm += pd_nvf4_dot4w(wv.y >> 16, s0, xrow, e0 + 12u);
        psm += pd_nvf4_dot4w(wv.z & 0xFFFFu, s1, xrow, e0 + 16u);
        psm += pd_nvf4_dot4w(wv.z >> 16, s1, xrow, e0 + 20u);
        psm += pd_nvf4_dot4w(wv.w & 0xFFFFu, s1, xrow, e0 + 24u);
        psm += pd_nvf4_dot4w(wv.w >> 16, s1, xrow, e0 + 28u);
    }
    // the down_part fold shape: w per lane before the combine
    float acc = w * psm;
    acc += __shfl_xor_sync(0xffffffffu, acc, 1);
    __shared__ float psum[4][16];
    if (p == 0) psum[warp][lane >> 1] = acc;
    __syncthreads();
    if (threadIdx.x < 16u)
        part[(size_t)slot * embd + rg * 16u + threadIdx.x] =
            0.0f + (((psum[0][threadIdx.x] + psum[1][threadIdx.x]) +
                     psum[2][threadIdx.x]) + psum[3][threadIdx.x]);
#else
    (void)rdata; (void)rscale; (void)rscale2; (void)sdata; (void)sscale;
    (void)sscale2; (void)idx; (void)topk_w; (void)act; (void)part; (void)ff_r;
    (void)ff_s; (void)embd; (void)k;
#endif
}

// ---- launchers (ABI 472-477; cc12-only, see exports.cuh) -------------------

// Skinny decode pair: BM=8 blocks from pd_moe_align_bm(bm=8). Same argument
// contract as the bs pair at BM=8 strides (sorted_row/sorted_slot are
// [nb*8], fq/fs are [nb*8, ff/16th]).
PD_EXPORT
int pd_nvf4_moe_up_relu2_st(const void* data, const void* scale,
                            const void* scale2, const void* sorted_row,
                            const void* block_expert, const void* xq,
                            const void* xs, void* fq, void* fs, uint32_t in_dim,
                            uint32_t ff, uint32_t nb, void* stream) {
#ifndef PD_BS_HOST
    (void)data; (void)scale; (void)scale2; (void)sorted_row; (void)block_expert;
    (void)xq; (void)xs; (void)fq; (void)fs; (void)in_dim; (void)ff; (void)nb;
    (void)stream;
    return cudaErrorNotSupported;
#else
    if (nb == 0) return 0;
    if ((in_dim & 63u) != 0 || (ff & 63u) != 0) return cudaErrorInvalidValue;
    dim3 grid(nb, ff >> 6);
    pd_nv4st_up_kernel<8u, 64u>
        <<<grid, 128u, PD_STT_SMEM(8u, 64u), (cudaStream_t)stream>>>(
            (const uint8_t*)data, (const uint8_t*)scale, (const float*)scale2,
            (const uint32_t*)sorted_row, (const uint32_t*)block_expert,
            (const uint8_t*)xq, (const uint8_t*)xs, (uint8_t*)fq, (uint8_t*)fs,
            in_dim, ff);
    return pd_launch_status();
#endif
}

PD_EXPORT
int pd_nvf4_moe_down_st(const void* data, const void* scale, const void* scale2,
                        const void* sorted_row, const void* sorted_slot,
                        const void* block_expert, const void* topk_w,
                        const void* fq, const void* fs, void* part, uint32_t ff,
                        uint32_t embd, uint32_t kw, uint32_t np,
                        uint32_t slot_off, uint32_t nb, void* stream) {
#ifndef PD_BS_HOST
    (void)data; (void)scale; (void)scale2; (void)sorted_row; (void)sorted_slot;
    (void)block_expert; (void)topk_w; (void)fq; (void)fs; (void)part; (void)ff;
    (void)embd; (void)kw; (void)np; (void)slot_off; (void)nb; (void)stream;
    return cudaErrorNotSupported;
#else
    if (nb == 0) return 0;
    if ((ff & 63u) != 0 || (embd & 63u) != 0) return cudaErrorInvalidValue;
    dim3 grid(nb, embd >> 6);
    pd_nv4st_dn_kernel<8u, 64u>
        <<<grid, 128u, PD_STT_SMEM(8u, 64u), (cudaStream_t)stream>>>(
            (const uint8_t*)data, (const uint8_t*)scale, (const float*)scale2,
            (const uint32_t*)sorted_row, (const uint32_t*)sorted_slot,
            (const uint32_t*)block_expert, (const float*)topk_w,
            (const uint8_t*)fq, (const uint8_t*)fs, (float*)part, ff, embd, kw,
            np, slot_off);
    return pd_launch_status();
#endif
}

// Wide prefill pair: BM=32 blocks (the shipped align), 128-row CTAs over
// the tiled plane. Bit-exact vs the row-major bs pair on identical routing.
PD_EXPORT
int pd_nvf4_moe_up_relu2_stw(const void* data, const void* scale,
                             const void* scale2, const void* sorted_row,
                             const void* block_expert, const void* xq,
                             const void* xs, void* fq, void* fs,
                             uint32_t in_dim, uint32_t ff, uint32_t nb,
                             void* stream) {
#ifndef PD_BS_HOST
    (void)data; (void)scale; (void)scale2; (void)sorted_row; (void)block_expert;
    (void)xq; (void)xs; (void)fq; (void)fs; (void)in_dim; (void)ff; (void)nb;
    (void)stream;
    return cudaErrorNotSupported;
#else
    if (nb == 0) return 0;
    if ((in_dim & 63u) != 0 || (ff & 63u) != 0) return cudaErrorInvalidValue;
    dim3 grid(nb, (ff + 127u) >> 7);
    pd_nv4st_up_kernel<32u, 128u>
        <<<grid, 256u, PD_STT_SMEM(32u, 128u), (cudaStream_t)stream>>>(
            (const uint8_t*)data, (const uint8_t*)scale, (const float*)scale2,
            (const uint32_t*)sorted_row, (const uint32_t*)block_expert,
            (const uint8_t*)xq, (const uint8_t*)xs, (uint8_t*)fq, (uint8_t*)fs,
            in_dim, ff);
    return pd_launch_status();
#endif
}

PD_EXPORT
int pd_nvf4_moe_down_stw(const void* data, const void* scale,
                         const void* scale2, const void* sorted_row,
                         const void* sorted_slot, const void* block_expert,
                         const void* topk_w, const void* fq, const void* fs,
                         void* part, uint32_t ff, uint32_t embd, uint32_t kw,
                         uint32_t np, uint32_t slot_off, uint32_t nb,
                         void* stream) {
#ifndef PD_BS_HOST
    (void)data; (void)scale; (void)scale2; (void)sorted_row; (void)sorted_slot;
    (void)block_expert; (void)topk_w; (void)fq; (void)fs; (void)part; (void)ff;
    (void)embd; (void)kw; (void)np; (void)slot_off; (void)nb; (void)stream;
    return cudaErrorNotSupported;
#else
    if (nb == 0) return 0;
    if ((ff & 63u) != 0 || (embd & 63u) != 0) return cudaErrorInvalidValue;
    dim3 grid(nb, (embd + 127u) >> 7);
    pd_nv4st_dn_kernel<32u, 128u>
        <<<grid, 256u, PD_STT_SMEM(32u, 128u), (cudaStream_t)stream>>>(
            (const uint8_t*)data, (const uint8_t*)scale, (const float*)scale2,
            (const uint32_t*)sorted_row, (const uint32_t*)sorted_slot,
            (const uint32_t*)block_expert, (const float*)topk_w,
            (const uint8_t*)fq, (const uint8_t*)fs, (float*)part, ff, embd, kw,
            np, slot_off);
    return pd_launch_status();
#endif
}

// r=1 twins: same argument contract as the mt pair, tiled planes.
PD_EXPORT
int pd_nvf4_moe_up_relu2_mtt(const void* rdata, const void* rscale,
                             const void* rscale2, const void* sdata,
                             const void* sscale, const void* sscale2,
                             const void* idx, const void* x, void* act,
                             uint32_t in_dim, uint32_t ff_r, uint32_t ff_s,
                             uint32_t k, void* stream) {
#ifndef PD_BS_HOST
    (void)rdata; (void)rscale; (void)rscale2; (void)sdata; (void)sscale;
    (void)sscale2; (void)idx; (void)x; (void)act; (void)in_dim; (void)ff_r;
    (void)ff_s; (void)k; (void)stream;
    return cudaErrorNotSupported;
#else
    if (ff_r == 0 || k == 0) return 0;
    if ((in_dim & 63u) != 0 || (ff_r & 63u) != 0 || (ff_s & 63u) != 0)
        return cudaErrorInvalidValue;
    const uint32_t grid = k * (ff_r >> 4) + (ff_s >> 4);
    pd_nv4st_mt_up_kernel<<<grid, 128u, 0, (cudaStream_t)stream>>>(
        (const uint8_t*)rdata, (const uint8_t*)rscale, (const float*)rscale2,
        (const uint8_t*)sdata, (const uint8_t*)sscale, (const float*)sscale2,
        (const uint32_t*)idx, (const float*)x, (float*)act, in_dim, ff_r, ff_s,
        k);
    return pd_launch_status();
#endif
}

PD_EXPORT
int pd_nvf4_moe_down_part_tt(const void* rdata, const void* rscale,
                             const void* rscale2, const void* sdata,
                             const void* sscale, const void* sscale2,
                             const void* idx, const void* topk_w,
                             const void* act, void* part, uint32_t ff_r,
                             uint32_t ff_s, uint32_t embd, uint32_t k,
                             void* stream) {
#ifndef PD_BS_HOST
    (void)rdata; (void)rscale; (void)rscale2; (void)sdata; (void)sscale;
    (void)sscale2; (void)idx; (void)topk_w; (void)act; (void)part; (void)ff_r;
    (void)ff_s; (void)embd; (void)k; (void)stream;
    return cudaErrorNotSupported;
#else
    if (embd == 0 || k == 0) return 0;
    if ((ff_r & 63u) != 0 || (ff_s & 63u) != 0 || (embd & 63u) != 0)
        return cudaErrorInvalidValue;
    const uint32_t grid = (k + 1u) * (embd >> 4);
    pd_nv4st_mt_dn_kernel<<<grid, 128u, 0, (cudaStream_t)stream>>>(
        (const uint8_t*)rdata, (const uint8_t*)rscale, (const float*)rscale2,
        (const uint8_t*)sdata, (const uint8_t*)sscale, (const float*)sscale2,
        (const uint32_t*)idx, (const float*)topk_w, (const float*)act,
        (float*)part, ff_r, ff_s, embd, k);
    return pd_launch_status();
#endif
}
