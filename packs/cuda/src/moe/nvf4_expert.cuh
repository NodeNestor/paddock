// moe/nvf4_expert.cuh - NVFP4 MoE expert consumers: the modelopt-checkpoint
// expert GEMMs, the persistent raw-ring twin, and the tile-major / fragment
// plane twins.
// Textually-included segment of the single pack translation unit.
// Not standalone-compilable: include order is defined by ../pack.cu.
//
// Split out of quant/nvf4.cuh (5680 lines against the ~2500
// ceiling, decision 3). Cut on that
// file's own banners; all three pieces sit under the ceiling.
//
// Lives under moe/ by domain (the layout law), next to moe/nvf4_st.cuh - even
// though its helpers come from quant/nvf4.cuh. Format lane and op domain point
// different ways here; the layout law follows the OP.
//
// Include after quant/nvf4.cuh - uses its quantizers and mma/dot4w helpers -
// and before moe/nvf4_sorted.cuh. Nothing here is used by quant/nvf4.cuh; the
// dependency runs one way. Basename is nvf4_expert, not nvf4, deliberately:
// quant/nvf4.cuh already owns that name and two `nvf4.cuh` in one TU is a
// reader trap even though the include paths disambiguate.
// ---- NVFP4 MoE expert consumers (nemotron_h_moe) ---------------
// Expert planes are one contiguous residency per role (row of expert e at
// e*ff + r - the house MoE layout), scales flat alongside, and scale2 is a
// per-EXPERT f32 array (modelopt quantizes each expert separately; folding the
// per-expert factor into e4m3 would be lossy, so it rides the epilogue).
// The K walk is pd_nvf4_gemv's warp-coherent step verbatim.

// The arithmetic half of pd_nvf4_dot4, taking the 16-bit weight word and
// scale byte already loaded - the expression is dot4's verbatim, so any
// caller that hands it the same wb/sb/x[e..e+3] lands the identical f32.
// The fragment-layout twins use this: their weight words come out of the
// permuted blocks via prmt instead of a row-pointer u16 load.
__device__ __forceinline__ float pd_nvf4_dot4w(uint32_t wb, uint32_t sb,
                                               const float* __restrict__ x,
                                               uint32_t e) {
#if PD_NV4_OK
    constexpr uint32_t T0 = 0x3C383000u, T1 = 0x4C484440u;
    const float s = (float)reinterpret_cast<const __nv_fp8_e4m3&>(sb);
    const float4 xv = *reinterpret_cast<const float4*>(x + e);
    const uint32_t v = (wb & 0xFu) | ((wb & 0xF0u) << 4)
                     | ((wb & 0xF00u) << 8) | ((wb & 0xF000u) << 12);
    const uint32_t mag = v & 0x07070707u;
    const uint32_t t = (mag | (mag >> 4)) & 0x00FF00FFu;
    const uint32_t e4 = __byte_perm(T0, T1, (t | (t >> 8)) & 0xFFFFu)
                      | ((v & 0x08080808u) << 4);
    const __nv_fp8_e4m3* eb = reinterpret_cast<const __nv_fp8_e4m3*>(&e4);
    return s * ((float)eb[0] * xv.x + (float)eb[1] * xv.y
              + (float)eb[2] * xv.z + (float)eb[3] * xv.w);
#else
    (void)wb; (void)sb; (void)x; (void)e;
    return 0.0f;
#endif
}

// Decode the 4 adjacent-packed e2m1 nibbles at row[e>>1] against scale byte
// srow[e>>4] and dot with x[e..e+3] - shared by the up/down kernels below.
__device__ __forceinline__ float pd_nvf4_dot4(const uint8_t* __restrict__ row,
                                              const uint8_t* __restrict__ srow,
                                              const float* __restrict__ x,
                                              uint32_t e) {
#if PD_NV4_OK
    const uint32_t wb = (uint32_t)*reinterpret_cast<const uint16_t*>(row + (e >> 1));
    return pd_nvf4_dot4w(wb, (uint32_t)srow[e >> 4], x, e);
#else
    (void)row; (void)srow; (void)x; (void)e;
    return 0.0f;
#endif
}

// Row-batched twin of pd_nvf4_gemv_kernel (stage A):
// x [batch, in_dim] f32, y [batch, out_dim], grid.y = row. Same warp-
// coherent walk (pd_nvf4_dot4 is that step verbatim), same hoisted tail -
// bit-exact per row vs the 1-row kernel. Weight bytes re-stream per row
// (no cross-row tile), which is fine at the small decode widths this
// serves; a W4A16 GEMM tile is the follow-up if lm_head ever binds
// at width.
//
// TM=true reads the TILE-MAJOR plane layout (the lm_head repack
// rung): row o's 64 weight bytes for K-step k0 live at block
// (o/128 * nk + k0/128), offset (o%128)*64, and its 8 scale bytes at the
// same block index * 8. The pointers are REBASED per step so pd_nvf4_dot4's
// e-relative indexing - and with it the whole FMA walk and reduction - stays
// verbatim: bit-exact vs TM=false on the same logical plane. Requires
// in_dim % 128 == 0 (launcher-gated), so the ragged tail never runs.
template <bool TM = false>
__global__ void pd_nvf4_gemv_batch_kernel(
    const uint8_t* __restrict__ data, const uint8_t* __restrict__ scale,
    const float* __restrict__ bias, const float* __restrict__ x,
    float* __restrict__ y, float scale2, uint32_t in_dim, uint32_t out_dim) {
#if PD_NV4_OK
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const uint32_t o = blockIdx.x * (blockDim.x >> 5) + warp;
    if (o >= out_dim) return;
    const uint8_t* row = data;
    const uint8_t* srow = scale;
    size_t trb = 0;  // TM: (tile*nk + 0)*128 + row-in-tile, the block-0 slot
    if constexpr (TM) {
        trb = (size_t)(o >> 7) * (in_dim >> 7) * 128u + (o & 127u);
    } else {
        row += (size_t)o * (in_dim >> 1);
        srow += (size_t)o * (in_dim >> 4);
    }
    const float* xr = x + (size_t)blockIdx.y * in_dim;
    float acc = 0.0f;
    const uint32_t full = in_dim & ~127u;
    #pragma unroll 4
    for (uint32_t k0 = 0; k0 < full; k0 += 128u) {
        if constexpr (TM) {
            // step block = trb + (k0/128)*128 rows; subtracting the row-major
            // in-step offset keeps dot4's `e`-based indexing correct
            const size_t stp = trb + (size_t)(k0 >> 7) * 128u;
            acc += pd_nvf4_dot4(data + stp * 64u - (k0 >> 1),
                                scale + stp * 8u - (k0 >> 4), xr,
                                k0 + lane * 4u);
        } else {
            acc += pd_nvf4_dot4(row, srow, xr, k0 + lane * 4u);
        }
    }
    if constexpr (!TM) {
        if (full < in_dim) {
            const uint32_t e = full + lane * 4u;
            if (e < in_dim) acc += pd_nvf4_dot4(row, srow, xr, e);
        }
    }
    for (uint32_t s = 16; s > 0; s >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, s);
    if (lane == 0) {
        float v = acc * scale2;
        if (bias) v += bias[o];
        y[(size_t)blockIdx.y * out_dim + o] = v;
    }
#else
    (void)data; (void)scale; (void)bias; (void)x; (void)y; (void)scale2;
    (void)in_dim; (void)out_dim;
#endif
}

PD_EXPORT
int pd_nvf4_gemv_batch(const void* data, const void* scale, const void* bias,
                       const void* x, void* y, float scale2, uint32_t in_dim,
                       uint32_t out_dim, uint32_t batch, void* stream) {
#ifndef PD_BS_HOST
    (void)data; (void)scale; (void)bias; (void)x; (void)y; (void)scale2;
    (void)in_dim; (void)out_dim; (void)batch; (void)stream;
    return cudaErrorNotSupported;
#else
    if (out_dim == 0 || batch == 0) return 0;
    if ((in_dim & 31u) != 0) return cudaErrorInvalidValue;
    const uint32_t rows_per_cta = 8u;
    dim3 grid((out_dim + rows_per_cta - 1u) / rows_per_cta, batch);
    pd_nvf4_gemv_batch_kernel<false><<<grid, rows_per_cta * 32u, 0,
                                       (cudaStream_t)stream>>>(
        (const uint8_t*)data, (const uint8_t*)scale, (const float*)bias,
        (const float*)x, (float*)y, scale2, in_dim, out_dim);
    return pd_launch_status();
#endif
}

// FRAGMENT-layout gemv (the fragment rung): the plane's 8 KB
// (tile, stage) blocks are permuted to [w:8][k16:8][g:8][u32 t0..t3] with
// u32 = [row g byte t, row g+8 byte t, row g byte 4+t, row g+8 byte 4+t]
// (scales stay tile-major [row][8B] per block). The gemv reshapes to match:
// one CTA per 16-row group (blockIdx.x = tile*8 + w), and each warp owns
// the ROW PAIR (16w+j, 16w+j+8) - the pair's weight words live in the same
// u64s (only the extracted byte differs) and the pair shares each x float4,
// so one u64 + one float4 feed two rows' FMA chains. Per row the chain is
// pd_nvf4_dot4w over elements 4l..4l+3 in ascending k0 - dot4's expression
// and order verbatim - so each output row is BIT-EXACT vs the row-major
// gemv. Per stage a CTA touches exactly its contiguous 1 KB slice
// (offsets w*1024..+1024), so the DRAM walk is sequential per CTA and the
// plane streams once. Requires in_dim % 128 == 0 (the layout's contract).
__global__ void pd_nvf4_gemv_batch_tf_kernel(
    const uint8_t* __restrict__ data, const uint8_t* __restrict__ scale,
    const float* __restrict__ bias, const float* __restrict__ x,
    float* __restrict__ y, float scale2, uint32_t in_dim, uint32_t out_dim) {
#if PD_NV4_OK
    const uint32_t j = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const uint32_t grp = blockIdx.x;              // tile*8 + w
    const size_t tile = grp >> 3;
    const uint32_t w = grp & 7u;
    const uint32_t rA = grp * 16u + j;            // g = j, low half
    const uint32_t rB = rA + 8u;                  // high half of the pair
    const uint32_t nk = in_dim >> 7;
    // lane l covers elements 4l..4l+3 = bytes 2l, 2l+1 of the stage row:
    // k16 group sk = l>>2, u32 pair t = (l&1)*2, byte half hb = (l>>1)&1
    const uint32_t sk = lane >> 2;
    const uint32_t tp = (lane & 1u) * 2u;
    const uint32_t hb = (lane >> 1) & 1u;
    // prmt selectors building the u16 [byte 2l, byte 2l+1] for each row:
    // byte q of u32 t, byte q of u32 t+1 (q = hb*2 + row-half)
    const uint32_t qA = hb * 2u;
    const uint32_t selA = qA | ((qA + 4u) << 4);
    const uint32_t selB = (qA + 1u) | ((qA + 5u) << 4);
    const float* xr = x + (size_t)blockIdx.y * in_dim;
    float accA = 0.0f, accB = 0.0f;
    #pragma unroll 4
    for (uint32_t ks = 0; ks < nk; ++ks) {
        const uint8_t* blk = data + (((tile * nk + ks) << 13) | (w << 10));
        const uint2 v = *reinterpret_cast<const uint2*>(
            blk + sk * 128u + j * 16u + tp * 4u);
        const uint8_t* sblk = scale + ((tile * nk + ks) << 10) + w * 128u;
        const uint32_t sA = sblk[j * 8u + sk];
        const uint32_t sB = sblk[(j + 8u) * 8u + sk];
        const uint32_t e = ks * 128u + lane * 4u;
        accA += pd_nvf4_dot4w(__byte_perm(v.x, v.y, selA) & 0xFFFFu, sA, xr, e);
        accB += pd_nvf4_dot4w(__byte_perm(v.x, v.y, selB) & 0xFFFFu, sB, xr, e);
    }
    for (uint32_t s = 16; s > 0; s >>= 1) {
        accA += __shfl_down_sync(0xffffffffu, accA, s);
        accB += __shfl_down_sync(0xffffffffu, accB, s);
    }
    if (lane == 0) {
        if (rA < out_dim) {
            float vv = accA * scale2;
            if (bias) vv += bias[rA];
            y[(size_t)blockIdx.y * out_dim + rA] = vv;
        }
        if (rB < out_dim) {
            float vv = accB * scale2;
            if (bias) vv += bias[rB];
            y[(size_t)blockIdx.y * out_dim + rB] = vv;
        }
    }
#else
    (void)data; (void)scale; (void)bias; (void)x; (void)y; (void)scale2;
    (void)in_dim; (void)out_dim;
#endif
}

PD_EXPORT
int pd_nvf4_gemv_batch_tf(const void* data, const void* scale,
                          const void* bias, const void* x, void* y,
                          float scale2, uint32_t in_dim, uint32_t out_dim,
                          uint32_t batch, void* stream) {
#ifndef PD_BS_HOST
    (void)data; (void)scale; (void)bias; (void)x; (void)y; (void)scale2;
    (void)in_dim; (void)out_dim; (void)batch; (void)stream;
    return cudaErrorNotSupported;
#else
    if (out_dim == 0 || batch == 0) return 0;
    if ((in_dim & 127u) != 0) return cudaErrorInvalidValue;
    // one CTA per 16-row group over the PADDED plane (pad rows are
    // zero-filled and their stores are guarded)
    const uint32_t mt = (out_dim + 127u) / 128u;
    dim3 grid(mt * 8u, batch);
    pd_nvf4_gemv_batch_tf_kernel<<<grid, 256u, 0, (cudaStream_t)stream>>>(
        (const uint8_t*)data, (const uint8_t*)scale, (const float*)bias,
        (const float*)x, (float*)y, scale2, in_dim, out_dim);
    return pd_launch_status();
#endif
}

// Multi-row W4A16 twin: decode each weight fragment once and dot
// it against up to BR resident activation rows. The follow-up the gemv_batch
// comment above promised - at c32 it was re-streaming the
// 177 MB lm_head plane per ROW (med 4.19 ms per 32-row tick, 15.9% of GPU
// time). grid.y tiles the batch in BR-row groups, so the plane streams
// ceil(batch/BR) times instead of `batch` times. The K walk, per-row FMA
// order and shuffle reduction are pd_nvf4_gemv_batch's verbatim - bit-exact
// per row vs that kernel (and so vs the 1-row gemv).
//
// BN/KS: BN output rows per warp against the staged tile, and KS
// 128-element sub-steps staged per barrier. The first cut's 21 syncthreads
// pairs per CTA (one per 128-K step) fenced the weight stream to one 64 B
// load in flight per warp per interval - measured 113 GB/s at the lm_head
// shape while the barrier-free single-row gemv does 1076 on the same plane.
// The KS-wide span cuts the barrier count by KS, and all KS x BN weight
// words + scale bytes are staged into register arrays before any
// consumption (the qwen pf7 lesson: ptxas serializes memory loops through
// one temp cluster unless the loads land in distinct staging slots), so a
// warp keeps KS*BN 64 B weight loads in flight per interval. The per-row
// FMA chain (ks ascending = the same K order) and reduction stay verbatim
// -> still bit-exact per row at every (BN, KS). Thin planes keep BN=1 via
// the launcher's width gate (a 256-wide plane at BN=4 would be 8 CTAs).
// TM=true is the same tile-major read as pd_nvf4_gemv_batch_kernel<TM>:
// only where the weight word and scale byte come from changes - the staged
// register arrays, the FMA chain and the reduction are shared code, so the
// TM arm is bit-exact vs TM=false on the same logical plane. Requires
// in_dim % 128 == 0 (launcher-gated). FRAG=true reads the FRAGMENT-ordered
// blocks instead (see pd_nvf4_gemv_batch_tf_kernel for the layout): the
// weight u16 comes out of a u64 via prmt, the scale byte keeps the
// tile-major addressing (scales are not fragment-permuted) - still the
// same staged arrays and FMA chain, so still bit-exact.
template <uint32_t BR, uint32_t BN, uint32_t KS, bool TM = false,
          bool FRAG = false>
__global__ void pd_nvf4_gemm_mr_kernel(
    const uint8_t* __restrict__ data, const uint8_t* __restrict__ scale,
    const float* __restrict__ bias, const float* __restrict__ x,
    float* __restrict__ y, float scale2, uint32_t in_dim, uint32_t out_dim,
    uint32_t batch) {
#if PD_NV4_OK
    constexpr uint32_t T0 = 0x3C383000u, T1 = 0x4C484440u;
    constexpr uint32_t SPAN = KS * 128u;
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const uint32_t o0 = (blockIdx.x * (blockDim.x >> 5) + warp) * BN;
    const uint32_t r0 = blockIdx.y * BR;
    // x K-span staged in smem once per CTA per barrier interval - v1 had
    // every warp re-reading the tile from L2 (8x redundant) and measured
    // 110 GB/s; the weight stream must be the only long-latency stream.
    // [BR][SPAN] f32 = 8 KB at KS=1, 32 KB at KS=4 (still under the 48 KB
    // static window; occupancy is register-bound either way).
    __shared__ float xs[BR][SPAN];
    float acc[BN][BR];
    #pragma unroll
    for (uint32_t n = 0; n < BN; ++n)
        #pragma unroll
        for (uint32_t i = 0; i < BR; ++i) acc[n][i] = 0.0f;
    for (uint32_t k0 = 0; k0 < in_dim; k0 += SPAN) {
        __syncthreads();
        // 256 threads x BR*SPAN elems: thread t loads elems t, t+256, ...
        for (uint32_t idx = threadIdx.x; idx < BR * SPAN; idx += blockDim.x) {
            const uint32_t ri = idx / SPAN, ci = idx % SPAN;
            xs[ri][ci] = (r0 + ri < batch && k0 + ci < in_dim)
                ? x[(size_t)(r0 + ri) * in_dim + k0 + ci]
                : 0.0f;
        }
        __syncthreads();
        // stage all sub-steps' weight words + scale bytes, then consume
        uint32_t wbv[KS][BN];
        float sv[KS][BN];
        #pragma unroll
        for (uint32_t ks = 0; ks < KS; ++ks) {
            const uint32_t e = k0 + ks * 128u + lane * 4u;
            #pragma unroll
            for (uint32_t n = 0; n < BN; ++n) {
                const bool ok = o0 + n < out_dim && e < in_dim;
                if constexpr (FRAG) {
                    // fragment block of stage e>>7; the row's byte pair
                    // (e>>1, +1) sits in u32s t, t+1 at byte q
                    const uint32_t oo = o0 + n;
                    const uint32_t el = (e >> 1) & 63u;   // stage-row byte
                    const uint32_t skf = el >> 3;
                    const uint32_t tf_ = el & 3u;         // even: t of pair
                    const uint32_t qf = (((el >> 2) & 1u) << 1)
                                      | ((oo >> 3) & 1u);
                    const size_t blkb =
                        (((size_t)(oo >> 7) * (in_dim >> 7) + (e >> 7))
                         << 13) | (((oo >> 4) & 7u) << 10);
                    const size_t trs =
                        ((size_t)(oo >> 7) * (in_dim >> 7) + (e >> 7))
                            * 128u + (oo & 127u);
                    if (ok) {
                        const uint2 vv = *reinterpret_cast<const uint2*>(
                            data + blkb + skf * 128u + (oo & 7u) * 16u
                            + tf_ * 4u);
                        wbv[ks][n] = __byte_perm(vv.x, vv.y,
                                                 qf | ((qf + 4u) << 4))
                                   & 0xFFFFu;
                        sv[ks][n] = (float)reinterpret_cast<
                            const __nv_fp8_e4m3&>(
                                scale[trs * 8u + ((e >> 4) & 7u)]);
                    } else {
                        wbv[ks][n] = 0u;
                        sv[ks][n] = 0.0f;
                    }
                } else if constexpr (TM) {
                    // block (o/128 * nk + e/128), row-in-tile o%128; in-block
                    // byte offsets are e's low bits (lane*2 / lane/4)
                    const uint32_t oo = o0 + n;
                    const size_t tr =
                        ((size_t)(oo >> 7) * (in_dim >> 7) + (e >> 7)) * 128u
                        + (oo & 127u);
                    wbv[ks][n] = ok
                        ? (uint32_t)*reinterpret_cast<const uint16_t*>(
                              data + tr * 64u + ((e >> 1) & 63u))
                        : 0u;
                    sv[ks][n] = ok
                        ? (float)reinterpret_cast<const __nv_fp8_e4m3&>(
                              scale[tr * 8u + ((e >> 4) & 7u)])
                        : 0.0f;
                } else {
                    const uint8_t* row =
                        data + (size_t)(o0 + n) * (in_dim >> 1);
                    const uint8_t* srow =
                        scale + (size_t)(o0 + n) * (in_dim >> 4);
                    wbv[ks][n] = ok
                        ? (uint32_t)*reinterpret_cast<const uint16_t*>(
                              row + (e >> 1))
                        : 0u;
                    sv[ks][n] = ok
                        ? (float)reinterpret_cast<const __nv_fp8_e4m3&>(
                              srow[e >> 4])
                        : 0.0f;
                }
            }
        }
        #pragma unroll
        for (uint32_t ks = 0; ks < KS; ++ks) {
            const uint32_t e = k0 + ks * 128u + lane * 4u;
            if (e >= in_dim) continue;
            const uint32_t c = ks * 128u + lane * 4u;
            #pragma unroll
            for (uint32_t n = 0; n < BN; ++n) {
                const uint32_t wb = wbv[ks][n];
                const float s = sv[ks][n];
                const uint32_t v = (wb & 0xFu) | ((wb & 0xF0u) << 4)
                                 | ((wb & 0xF00u) << 8) | ((wb & 0xF000u) << 12);
                const uint32_t mag = v & 0x07070707u;
                const uint32_t t = (mag | (mag >> 4)) & 0x00FF00FFu;
                const uint32_t e4 = __byte_perm(T0, T1, (t | (t >> 8)) & 0xFFFFu)
                                  | ((v & 0x08080808u) << 4);
                const __nv_fp8_e4m3* eb =
                    reinterpret_cast<const __nv_fp8_e4m3*>(&e4);
                #pragma unroll
                for (uint32_t i = 0; i < BR; ++i) {
                    // the per-row gemv walk's expression VERBATIM (same
                    // contraction shape -> bit-exact per row); the smem tile
                    // only moves where x is read from, never the arithmetic
                    acc[n][i] += s * ((float)eb[0] * xs[i][c]
                                    + (float)eb[1] * xs[i][c + 1u]
                                    + (float)eb[2] * xs[i][c + 2u]
                                    + (float)eb[3] * xs[i][c + 3u]);
                }
            }
        }
    }
    #pragma unroll
    for (uint32_t n = 0; n < BN; ++n) {
        if (o0 + n >= out_dim) return;
        #pragma unroll
        for (uint32_t i = 0; i < BR; ++i) {
            float a = acc[n][i];
            for (uint32_t s = 16; s > 0; s >>= 1)
                a += __shfl_down_sync(0xffffffffu, a, s);
            if (lane == 0 && r0 + i < batch) {
                float v = a * scale2;
                if (bias) v += bias[o0 + n];
                y[(size_t)(r0 + i) * out_dim + o0 + n] = v;
            }
        }
    }
#else
    (void)data; (void)scale; (void)bias; (void)x; (void)y; (void)scale2;
    (void)in_dim; (void)out_dim; (void)batch;
#endif
}

PD_EXPORT
int pd_nvf4_gemm_mr(const void* data, const void* scale, const void* bias,
                    const void* x, void* y, float scale2, uint32_t in_dim,
                    uint32_t out_dim, uint32_t batch, void* stream) {
#ifndef PD_BS_HOST
    (void)data; (void)scale; (void)bias; (void)x; (void)y; (void)scale2;
    (void)in_dim; (void)out_dim; (void)batch; (void)stream;
    return cudaErrorNotSupported;
#else
    if (out_dim == 0 || batch == 0) return 0;
    if ((in_dim & 31u) != 0) return cudaErrorInvalidValue;
    constexpr uint32_t BR = 16u;
    const uint32_t rows_per_cta = 8u;
    // width gate: wide planes (the lm_head class) run the BN=2 arm - the
    // election measured at the vocab shape: b1k1 1710 us,
    // b2k1 1204, b2k2/b2k4 ~1155, b4 worse (occupancy). At b2k1: Issue
    // Slots Busy 63.6% vs DRAM 17% - this scalar-FMA W4A16 class is
    // COMPUTE-bound at batch 16, so ~1.15 ms is its structural floor at the
    // lm_head shape (the KS barrier-thinning axis is a wash for the same
    // reason). Going lower needs a tensor-core head class (exact-dequant
    // bf16 or W4A4) - a numeric class change behind a quality gate, not a
    // knob here. Thin planes keep BN=1 so the grid stays dense.
    if (out_dim >= 4096u) {
        constexpr uint32_t BN = 2u;
        dim3 grid((out_dim + rows_per_cta * BN - 1u) / (rows_per_cta * BN),
                  (batch + BR - 1u) / BR);
        pd_nvf4_gemm_mr_kernel<BR, BN, 1u><<<grid, rows_per_cta * 32u, 0,
                                             (cudaStream_t)stream>>>(
            (const uint8_t*)data, (const uint8_t*)scale, (const float*)bias,
            (const float*)x, (float*)y, scale2, in_dim, out_dim, batch);
        return pd_launch_status();
    }
    dim3 grid((out_dim + rows_per_cta - 1u) / rows_per_cta,
              (batch + BR - 1u) / BR);
    pd_nvf4_gemm_mr_kernel<BR, 1u, 1u><<<grid, rows_per_cta * 32u, 0,
                                         (cudaStream_t)stream>>>(
        (const uint8_t*)data, (const uint8_t*)scale, (const float*)bias,
        (const float*)x, (float*)y, scale2, in_dim, out_dim, batch);
    return pd_launch_status();
#endif
}

// Tensor-core head class. The scalar mr kernel above is
// ISSUE-bound at batch (63.6% issue slots vs 17% DRAM at the lm_head
// shape), so its ~1.15 ms is a class floor - the contraction has to move
// onto the tensor pipe to go lower. This twin is pd_bf16_gemm_mma_kernel's
// ring/compute/store VERBATIM with the A-stage swapped for an NVFP4 dequant:
// packed e2m1 nibbles + e4m3 per-16 scales load once per element (u64 per
// 16-element scale block), decode to bf16 into the shared A tile, and every
// k16 fragment feeds m16n8k16 bf16 mma instead of BR scalar FFMA chains.
//
// Numerics: an e2m1 value (2-bit mantissa) times an e4m3 scale (3-bit
// mantissa) carries at most 5 mantissa bits, so the staged bf16 weights are
// exact dequants of the checkpoint plane - the only class change is the
// activation cast f32->bf16, the same cast bf16 serving applies to
// every hidden state before its lm_head GEMM, so this concedes nothing
// to that class. scale2 stays a f32 epilogue multiply on the
// f32 accumulator, ordered exactly like the mr epilogue (acc*scale2 then
// +bias). Not bit-comparable to the mr kernel on general inputs (mma k16
// trees reassociate the sum) - the unit gate pins it two ways: bit-exact on
// integer-lattice inputs (where every accumulation order is exact in f32)
// and tolerance-gated on gaussian inputs.
template <uint32_t BM, uint32_t BN, uint32_t NWARP, uint32_t ST, uint32_t KT,
          uint32_t RG, uint32_t CG>
__global__ void __launch_bounds__(NWARP * 32) pd_nvf4_gemm_tc_kernel(
        const uint8_t* __restrict__ data, const uint8_t* __restrict__ scale,
        const float* __restrict__ bias, const float* __restrict__ X,
        float* __restrict__ Y, float scale2, uint32_t K, uint32_t M,
        uint32_t N) {
#if PD_BF16MMA_OK
    constexpr uint32_t T0 = 0x3C383000u, T1 = 0x4C484440u;
    constexpr uint32_t NTH = NWARP * 32u;
    constexpr uint32_t WM = RG * 16u;      // warp tile rows
    constexpr uint32_t WN = CG * 8u;       // warp tile cols
    constexpr uint32_t WR = BM / WM;       // warp-rows in the CTA tile
    constexpr uint32_t WC = BN / WN;       // warp-cols in the CTA tile
    constexpr uint32_t NSUBK = KT / 16u;   // k16 sub-tiles per stage
    constexpr uint32_t KPAD = KT + 8u;     // padded shared K-stride (bf16s)
    constexpr uint32_t GPR = KT / 16u;     // 16-elem scale blocks per row
    constexpr uint32_t H8PR = KT / 8u;     // 8-elem groups per B row
    constexpr uint32_t AIT = (BM * GPR + NTH - 1u) / NTH;  // A groups/thread
    static_assert(WR * WC == NWARP, "warp grid");
    static_assert(WM * WR == BM && WN * WC == BN, "tile cover");
    static_assert(KT % 16u == 0u, "KT k16-multiple");
    static_assert(ST >= 2u && ST <= 4u, "stage count (ring only)");

    extern __shared__ __align__(16) __nv_bfloat16 pd_nv4tc_dyn[];
    auto sh_a = reinterpret_cast<__nv_bfloat16(*)[BM * KPAD]>(pd_nv4tc_dyn);
    auto sh_b = reinterpret_cast<__nv_bfloat16(*)[BN * KPAD]>(pd_nv4tc_dyn +
                                                              ST * BM * KPAD);

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u, warp = tid >> 5;
    const uint32_t g = lane >> 2, t = lane & 3u;
    const uint32_t wr = (warp % WR) * WM;
    const uint32_t wc = (warp / WR) * WN;
    const uint32_t row_base = blockIdx.x * BM;
    const uint32_t col_base = blockIdx.y * BN;
    const __nv_bfloat16 zero = __float2bfloat16(0.0f);

    // Split stage: LOAD pulls the packed u64s + e4m3 scale bytes + B float4s
    // into registers (all AIT+BIT long-latency loads in flight per thread -
    // the qwen pf7 staging-array lesson); STORE decodes and lands them in
    // shared. The main loop puts compute() between the two, so the whole
    // DRAM latency of stage k+ST-1 hides under stage k's mma work - the
    // plain-load substitute for the cp.async overlap the bf16 twin gets for
    // free (cp.async cannot feed a dequant). K%16==0 (launcher gate) means
    // an A scale-block or B 8-group is always fully in or fully out of K.
    constexpr uint32_t BIT = (BN * H8PR + NTH - 1u) / NTH;
    uint64_t a_wq[AIT];
    float a_s[AIT];
    bool a_ok[AIT];
    float4 b_v0[BIT], b_v1[BIT];
    bool b_ok[BIT];
    auto stage_load = [&](uint32_t k0) {
        #pragma unroll
        for (uint32_t it = 0; it < AIT; ++it) {
            const uint32_t i = tid + it * NTH;
            const uint32_t row = i / GPR, g16 = (i % GPR) * 16u;
            const uint32_t gr = row_base + row, gk = k0 + g16;
            const bool ok = i < BM * GPR && gr < M && gk < K;
            a_ok[it] = ok;
            a_wq[it] = ok ? *reinterpret_cast<const uint64_t*>(
                                data + (size_t)gr * (K >> 1) + (gk >> 1))
                          : 0ull;
            a_s[it] = ok ? (float)reinterpret_cast<const __nv_fp8_e4m3&>(
                               scale[(size_t)gr * (K >> 4) + (gk >> 4)])
                         : 0.0f;
        }
        #pragma unroll
        for (uint32_t it = 0; it < BIT; ++it) {
            const uint32_t i = tid + it * NTH;
            const uint32_t col = i / H8PR, h8 = (i % H8PR) * 8u, gk = k0 + h8;
            const bool ok =
                i < BN * H8PR && (col_base + col) < N && gk < K;
            b_ok[it] = ok;
            if (ok) {
                const float* src = X + (size_t)(col_base + col) * K + gk;
                b_v0[it] = *reinterpret_cast<const float4*>(src);
                b_v1[it] = *reinterpret_cast<const float4*>(src + 4);
            }
        }
    };
    auto stage_store = [&](uint32_t buf) {
        #pragma unroll
        for (uint32_t it = 0; it < AIT; ++it) {
            const uint32_t i = tid + it * NTH;
            if (i >= BM * GPR) continue;
            const uint32_t row = i / GPR, g16 = (i % GPR) * 16u;
            __nv_bfloat16* dst = &sh_a[buf][row * KPAD + g16];
            __nv_bfloat16 tmp[16];
            if (a_ok[it]) {
                const uint64_t wq = a_wq[it];
                const float s = a_s[it];
                #pragma unroll
                for (uint32_t q = 0; q < 4u; ++q) {
                    // 4 nibbles -> 4 e4m3 bytes, the mr kernel's decode
                    // verbatim; then exact f32 s-fold + exact bf16 cast
                    const uint32_t wb = (uint32_t)(wq >> (16u * q)) & 0xFFFFu;
                    const uint32_t v = (wb & 0xFu) | ((wb & 0xF0u) << 4)
                                     | ((wb & 0xF00u) << 8)
                                     | ((wb & 0xF000u) << 12);
                    const uint32_t mag = v & 0x07070707u;
                    const uint32_t tt = (mag | (mag >> 4)) & 0x00FF00FFu;
                    const uint32_t e4 =
                        __byte_perm(T0, T1, (tt | (tt >> 8)) & 0xFFFFu)
                        | ((v & 0x08080808u) << 4);
                    const __nv_fp8_e4m3* eb =
                        reinterpret_cast<const __nv_fp8_e4m3*>(&e4);
                    #pragma unroll
                    for (uint32_t e = 0; e < 4u; ++e)
                        tmp[q * 4u + e] =
                            __float2bfloat16((float)eb[e] * s);
                }
            } else {
                #pragma unroll
                for (uint32_t e = 0; e < 16u; ++e) tmp[e] = zero;
            }
            reinterpret_cast<int4*>(dst)[0] =
                reinterpret_cast<const int4*>(tmp)[0];
            reinterpret_cast<int4*>(dst)[1] =
                reinterpret_cast<const int4*>(tmp)[1];
        }
        #pragma unroll
        for (uint32_t it = 0; it < BIT; ++it) {
            const uint32_t i = tid + it * NTH;
            if (i >= BN * H8PR) continue;
            const uint32_t col = i / H8PR, h8 = (i % H8PR) * 8u;
            __nv_bfloat16* dst = &sh_b[buf][col * KPAD + h8];
            if (b_ok[it]) {
                const float4 v0 = b_v0[it], v1 = b_v1[it];
                __nv_bfloat16 tmpb[8] = {
                    __float2bfloat16(v0.x), __float2bfloat16(v0.y),
                    __float2bfloat16(v0.z), __float2bfloat16(v0.w),
                    __float2bfloat16(v1.x), __float2bfloat16(v1.y),
                    __float2bfloat16(v1.z), __float2bfloat16(v1.w)};
                *reinterpret_cast<int4*>(dst) =
                    *reinterpret_cast<const int4*>(tmpb);
            } else {
                #pragma unroll
                for (uint32_t e = 0; e < 8u; ++e) dst[e] = zero;
            }
        }
    };

    const uint32_t l7 = lane & 7u;
    const uint32_t a_roff = ((lane & 8u) ? 8u : 0u) + l7;
    const uint32_t a_kof = (lane & 16u) ? 8u : 0u;
    const uint32_t b_kof = (lane & 8u) ? 8u : 0u;

    float acc[RG][CG][4] = {};
    auto compute = [&](uint32_t buf) {
        #pragma unroll
        for (uint32_t sk = 0; sk < NSUBK; ++sk) {
            const uint32_t ko = sk * 16u;
            uint32_t a[RG][4];
            #pragma unroll
            for (uint32_t rg = 0; rg < RG; ++rg)
                pd_bf16m_ldm_x4(
                    &sh_a[buf][(wr + rg * 16u + a_roff) * KPAD + ko + a_kof],
                    a[rg][0], a[rg][1], a[rg][2], a[rg][3]);
            uint32_t b[CG][2];
            #pragma unroll
            for (uint32_t cg = 0; cg < CG; ++cg)
                pd_bf16m_ldm_x2(
                    &sh_b[buf][(wc + cg * 8u + l7) * KPAD + ko + b_kof],
                    b[cg][0], b[cg][1]);
            #pragma unroll
            for (uint32_t rg = 0; rg < RG; ++rg)
                #pragma unroll
                for (uint32_t cg = 0; cg < CG; ++cg)
                    pd_bf16m_mma(acc[rg][cg], a[rg], b[cg]);
        }
    };

    // ST-deep ring, one barrier per K-step: sync -> issue stage k+ST-1's
    // loads -> compute(k) on shared while they fly -> decode+store. The
    // barrier at the top of step i orders both hazards: the store of buffer
    // b in step i-1 vs compute(b) in a later step (RAW, >=1 sync between),
    // and compute(b) in step i-1 vs the store overwriting b in step i (WAR
    // - with ST=2 the buffer computed in step i-1 is exactly the one
    // restaged in step i).
    #pragma unroll
    for (uint32_t s = 0; s < ST - 1u; ++s) {
        const uint32_t k0 = s * KT;
        if (k0 < K) {
            stage_load(k0);
            stage_store(s);
        }
    }
    uint32_t p = 0;
    for (uint32_t k0 = 0; k0 < K; k0 += KT) {
        const uint32_t pre = k0 + (ST - 1u) * KT;
        __syncthreads();
        if (pre < K) stage_load(pre);
        compute(p);
        if (pre < K) stage_store((p + ST - 1u) % ST);
        p = (p + 1u) % ST;
    }

    // store: element (m=out row, n=batch col) -> Y[n*M + m]. The epilogue is
    // the mr kernel's VERBATIM order - acc*scale2, then bias only when one
    // exists (an unconditional +0.0 would flip a -0.0 accumulator and break
    // the lattice bit-exactness gate).
    const bool hb = bias != nullptr;
    #pragma unroll
    for (uint32_t rg = 0; rg < RG; ++rg) {
        const uint32_t r0 = row_base + wr + rg * 16u + g;
        const uint32_t r8 = r0 + 8u;
        const float b0 = (hb && r0 < M) ? bias[r0] : 0.0f;
        const float b8 = (hb && r8 < M) ? bias[r8] : 0.0f;
        #pragma unroll
        for (uint32_t cg = 0; cg < CG; ++cg) {
            const uint32_t c0 = col_base + wc + cg * 8u + 2u * t;
            const uint32_t c1 = c0 + 1u;
            #pragma unroll
            for (uint32_t e = 0; e < 4u; ++e) {
                const uint32_t r = (e < 2u) ? r0 : r8;
                const uint32_t c = (e & 1u) ? c1 : c0;
                if (r < M && c < N) {
                    float v = acc[rg][cg][e] * scale2;
                    if (hb) v += (e < 2u) ? b0 : b8;
                    Y[(size_t)c * M + r] = v;
                }
            }
        }
    }
#else
    (void)data; (void)scale; (void)bias; (void)X; (void)Y; (void)scale2;
    (void)K; (void)M; (void)N;
#endif
}

#ifdef PD_BS_HOST
template <uint32_t BM, uint32_t BN, uint32_t NW, uint32_t ST, uint32_t KT,
          uint32_t RG, uint32_t CG>
static int pd_nvf4_tc_cfg(const uint8_t* data, const uint8_t* scale,
                          const float* bias, const float* x, float* y,
                          float scale2, uint32_t in_dim, uint32_t out_dim,
                          uint32_t batch, cudaStream_t st) {
    constexpr uint32_t KPAD = KT + 8u;
    constexpr uint32_t smem = ST * (BM * KPAD + BN * KPAD) * 2u;
    static bool set = false;
    if (!set) {
        cudaFuncSetAttribute(pd_nvf4_gemm_tc_kernel<BM, BN, NW, ST, KT, RG, CG>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        set = true;
    }
    dim3 grid((out_dim + BM - 1u) / BM, (batch + BN - 1u) / BN);
    pd_nvf4_gemm_tc_kernel<BM, BN, NW, ST, KT, RG, CG><<<grid, NW * 32u, smem,
                                                         st>>>(
        data, scale, bias, x, y, scale2, in_dim, out_dim, batch);
    return pd_launch_status();
}
#endif

// ---- persistent raw-ring twin (lm_head rung) ----
//
// The tc kernel above runs the 131072x2688 b32 head at ~280 us = ~48% of the
// measured 1484 GB/s roof, where a marlin-class schedule moves the same
// 198 MB at ~93% of it. Profiling attributes the difference to the
// SCHEDULE, not the class: 1024 one-shot CTAs at 1 CTA/SM re-pay the ring
// prologue every wave, and the register-staged plain loads make stage_store
// a hard wait that the ST=2 ring can't hide. This twin keeps the tc kernel's
// math bit-for-bit and replaces the schedule with marlin's design points -
// the same ones pd_kquant_w4a8_pipe_kernel already proved on this die:
//   - the ring stages RAW bytes (packed nibbles + e4m3 scale records + f32
//     activations) via cp.async - 27.5 KB/stage instead of 87 KB decoded, no
//     decode-then-store round trip, and the copy retires asynchronously so
//     nothing waits on DRAM but the ring depth itself (ST=3 ~ 2 stages of
//     lookahead);
//   - the dequant moves into the mma loop, per fragment register: an
//     m16n8k16 A register holds 2 k-adjacent elements, which is exactly one
//     packed byte, so a fragment is 4 shared byte-loads + one __byte_perm
//     against the same T0/T1 e4m3 table, the same f32 scale fold, and the
//     same rn bf16 cast as stage_store - identical staged values, and ALU
//     that dual-issues under the tensor pipe instead of serializing after it;
//   - One persistent CTA per SM walks row tiles (strided), so the prologue
//     is paid once per ~5.5 tiles instead of per tile, and the warp grid
//     flips to RG=1/CG=4 (8 warp-rows, 1 warp-col) so every A byte is
//     decoded exactly once per CTA (the tc grid's WC=2 decodes A twice).
// Accumulation order (k16 sub-tiles ascending, stages ascending, same mma
// chain per output element) and the epilogue are the tc kernel's verbatim,
// so the output is BIT-IDENTICAL to pd_nvf4_gemm_tc_kernel - the probe and
// the unit gate both memcmp the two. Launcher gates K % KT == 0 (no k-tail
// in the hot loop; every served NVFP4 plane has K % 32 == 0 and the lm_head
// K = 2688 = 21*128), everything else falls back to the tc config above.
// Kill switch: PADDOCK_NO_NVF4_TCP.

// 8-byte predicated cp.async for the scale records (KT/16 = 8 B per row per
// stage; 16 B copies would misalign on odd k0/16 offsets). src-size 0
// zero-fills without dereferencing gmem.
__device__ __forceinline__ void pd_nv4_cpa8(void* smem, const void* gmem,
                                            bool ok) {
#if PD_BF16MMA_OK
    const unsigned sm = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;" ::"r"(sm),
                 "l"(gmem), "r"(ok ? 8u : 0u));
#endif
}

#if PD_BF16MMA_OK
// one packed byte -> one A fragment register (2 adjacent bf16 weights): the
// tc stage_store decode restricted to a nibble pair. Same T0/T1 table, same
// sign lift, same f32 scale multiply, same round-to-nearest bf16 cast
// (__floats2bfloat162_rn is the paired form of __float2bfloat16 - both rn),
// so the fragment bits match what ldmatrix would read off a staged tile.
__device__ __forceinline__ uint32_t pd_nv4tcp_dec2(uint32_t by, float s) {
    constexpr uint32_t T0 = 0x3C383000u, T1 = 0x4C484440u;
    const uint32_t sel = (by & 7u) | (((by >> 4) & 7u) << 4);
    const uint32_t e4 = __byte_perm(T0, T1, sel) | ((by & 0x08u) << 4)
                      | ((by & 0x80u) << 8);
    const __nv_fp8_e4m3* eb = reinterpret_cast<const __nv_fp8_e4m3*>(&e4);
    const __nv_bfloat162 v =
        __floats2bfloat162_rn((float)eb[0] * s, (float)eb[1] * s);
    return reinterpret_cast<const uint32_t&>(v);
}
#endif

#if PD_BF16MMA_OK
// v2 fragment decode (the fragment-order rung): the e2m1 pair decodes
// STRAIGHT to bf16 through two byte tables - TH/TL hold the hi/lo bytes of
// the exact bf16 encoding of each e2m1 magnitude - and the e4m3 block scale
// multiplies in bf16. Both factors are exact in bf16 and the product
// carries <= 5 mantissa bits, so __hmul2's rn of an exact product lands the
// same bits as pd_nv4tcp_dec2's f32-multiply-then-rn path: bit-identical
// fragments, ~11 ops instead of ~16, and the two f8->f32 converts + two
// FMULs + pack collapse into one HMUL2. `s2` is the scale broadcast as
// bf16x2 (hoisted by the caller).
__device__ __forceinline__ uint32_t pd_nv4tcv_dec2(uint32_t by, uint32_t s2) {
    constexpr uint32_t TH0 = 0x3F3F3F00u, TH1 = 0x40404040u;
    constexpr uint32_t TL0 = 0xC0800000u, TL1 = 0xC0804000u;
    const uint32_t sel = (by & 7u) | (((by >> 4) & 7u) << 4);
    const uint32_t hb = __byte_perm(TH0, TH1, sel);
    const uint32_t lb = __byte_perm(TL0, TL1, sel);
    uint32_t r = __byte_perm(lb, hb, 0x5140u)
               | ((by & 0x08u) << 12) | ((by & 0x80u) << 24);
    const __nv_bfloat162 v =
        __hmul2(reinterpret_cast<const __nv_bfloat162&>(r),
                reinterpret_cast<const __nv_bfloat162&>(s2));
    return reinterpret_cast<const uint32_t&>(v);
}
#endif

// The fragment-order rung's probe kernel: tcp's persistent
// raw-ring over the TILE-MAJOR layout, with two restructures the marlin
// comparison pointed at. (1) A decodes via pd_nv4tcv_dec2 (bf16 tables +
// HMUL2 - bit-identical fragments, far fewer ops). (2) BONCE=true drops the
// f32 B ring entirely: B converts f32->bf16 COOPERATIVELY once per stage
// into a 2-slot shared bf16 tile (read back via ldmatrix - the proven
// BREG fragment path), sourced straight from global f32 X, which is
// L2-resident at every decode batch (b32 x 2688 x 4 = 344 KB). That kills
// the 8x per-warp B re-read (73% of tcp's L1 wavefronts, WC=1) without
// BREG's register staging, needs no extra barrier (the conversion for step
// ks+1 runs during step ks; the tile-step syncthreads already orders the
// cross-warp visibility), and shrinks the stage enough that ST=2 fits two
// CTAs per SM (MINB=2) - the occupancy door the BREG arms couldn't afford.
// REPK layout only: the row-major tc/tcp family keeps serving the untiled
// planes.
//
// FRAG=true additionally reads a FRAGMENT-ORDERED block: the 8 KB (tile,
// stage) block is permuted OFFLINE to [w:8][k16:8][g:8][u32 t0..t3], each
// u32 = the 4 bytes lane (g,t) feeds its a0..a3 fragment registers - one
// conflict-free LDS.32 per (sk, rg) replaces two LDS.64 + two dynamic
// prmts, the A stage drops its padding (8 KB flat, sequential cp.async),
// and ST=3 fits 2 CTAs/SM. Same bytes, same decode, same bits.
template <uint32_t BM, uint32_t BN, uint32_t NWARP, uint32_t ST, uint32_t KT,
          uint32_t RG, uint32_t CG, bool BONCE = false, uint32_t MINB = 1u,
          bool FRAG = false>
__global__ void __launch_bounds__(NWARP * 32, MINB) pd_nvf4_gemm_tcv_kernel(
        const uint8_t* __restrict__ data, const uint8_t* __restrict__ scale,
        const float* __restrict__ bias, const float* __restrict__ X,
        float* __restrict__ Y, float scale2, uint32_t K, uint32_t M,
        uint32_t N) {
#if PD_BF16MMA_OK
    constexpr uint32_t NTH = NWARP * 32u;
    constexpr uint32_t WM = RG * 16u, WN = CG * 8u;
    constexpr uint32_t WR = BM / WM, WC = BN / WN;
    constexpr uint32_t NSUBK = KT / 16u;
    constexpr uint32_t ARS = KT / 2u + 16u;
    constexpr uint32_t SRS = KT / 16u;
    constexpr uint32_t BRS = KT + 4u;   // f32 B ring (only when !BONCE)
    constexpr uint32_t BKP = KT + 8u;   // bf16 B tile stride (tc's KPAD)
    // FRAG stages are the flat 8 KB block (lane u32s are already
    // conflict-free: word = const + lane); row-wise stages keep the pad
    constexpr uint32_t ABYT = FRAG ? BM * (KT / 2u) : BM * ARS;
    constexpr uint32_t SBYT = BM * SRS;
    constexpr uint32_t ACH = BM * (KT / 32u);
    constexpr uint32_t BCH = BN * (KT / 4u);
    static_assert(WR * WC == NWARP, "warp grid");
    static_assert(WM * WR == BM && WN * WC == BN, "tile cover");
    static_assert(ST >= 2u && ST <= 4u, "ring depth");
    static_assert(BM <= NTH, "one scale record per thread");
    static_assert(!FRAG || RG == 1u, "frag u32 packs one 16-row group");

    extern __shared__ __align__(16) uint8_t pd_nv4tcv_dyn[];
    uint8_t* const sa = pd_nv4tcv_dyn;
    uint8_t* const ss = sa + (size_t)ST * ABYT;
    float* const sb = reinterpret_cast<float*>(ss + (size_t)ST * SBYT);
    __nv_bfloat16* const sbh =
        reinterpret_cast<__nv_bfloat16*>(ss + (size_t)ST * SBYT);

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u, warp = tid >> 5;
    const uint32_t g = lane >> 2, t = lane & 3u;
    const uint32_t l7 = lane & 7u;
    const uint32_t b_kof = (lane & 8u) ? 8u : 0u;
    const uint32_t wr = (warp % WR) * WM;
    const uint32_t wc = (warp / WR) * WN;
    const uint32_t nk = K / KT;
    const uint32_t mtiles = (M + BM - 1u) / BM;
    const uint32_t ntiles = (N + BN - 1u) / BN;
    const bool hb = bias != nullptr;

    for (uint32_t tile = blockIdx.x; tile < mtiles * ntiles;
         tile += gridDim.x) {
        const uint32_t row_base = (tile % mtiles) * BM;
        const uint32_t col_base = (tile / mtiles) * BN;
        float acc_[RG][CG][4] = {};

        auto stage = [&](uint32_t k0, uint32_t buf) {
            const size_t blk = (size_t)(row_base / BM) * nk + k0 / KT;
            #pragma unroll
            for (uint32_t it = 0; it < (ACH + NTH - 1u) / NTH; ++it) {
                const uint32_t i = tid + it * NTH;
                if (i < ACH) {
                    if constexpr (FRAG) {
                        pd_bf16m_cpa16(sa + buf * ABYT + i * 16u,
                                       data + blk * (BM * (KT >> 1))
                                           + i * 16u,
                                       true);
                    } else {
                        const uint32_t row = i / (KT / 32u),
                                       ch = i % (KT / 32u);
                        pd_bf16m_cpa16(sa + buf * ABYT + row * ARS + ch * 16u,
                                       data + (blk * BM + row) * (KT >> 1)
                                           + ch * 16u,
                                       true);
                    }
                }
            }
            if (tid < BM)
                pd_nv4_cpa8(ss + buf * SBYT + tid * SRS,
                            scale + (blk * BM + tid) * (KT >> 4), true);
            if constexpr (!BONCE) {
                #pragma unroll
                for (uint32_t it = 0; it < (BCH + NTH - 1u) / NTH; ++it) {
                    const uint32_t i = tid + it * NTH;
                    if (i < BCH) {
                        const uint32_t col = i / (KT / 4u), ch = i % (KT / 4u);
                        const uint32_t gc = col_base + col;
                        pd_bf16m_cpa16(sb + (size_t)buf * BN * BRS + col * BRS
                                           + ch * 4u,
                                       X + (size_t)gc * K + k0 + ch * 4u,
                                       gc < N);
                    }
                }
            }
            pd_bf16m_cpa_commit();
        };

        // BONCE: cooperative f32->bf16 conversion of one stage's B tile,
        // straight from global X (L2-hot at decode widths) into slot
        // (k0/KT)&1 - the same __float2bfloat16 per element as tc's
        // stage_store, so ldmatrix reads the identical fragment bits.
        auto b_conv = [&](uint32_t k0) {
            const uint32_t slot = (k0 / KT) & 1u;
            #pragma unroll
            for (uint32_t it = 0; it < (BN * (KT / 8u) + NTH - 1u) / NTH;
                 ++it) {
                const uint32_t i = tid + it * NTH;
                if (i >= BN * (KT / 8u)) continue;
                const uint32_t col = i / (KT / 8u), h8 = (i % (KT / 8u)) * 8u;
                __nv_bfloat16* dst =
                    &sbh[(size_t)slot * BN * BKP + col * BKP + h8];
                if (col_base + col < N) {
                    const float* src =
                        X + (size_t)(col_base + col) * K + k0 + h8;
                    const float4 v0 = *reinterpret_cast<const float4*>(src);
                    const float4 v1 =
                        *reinterpret_cast<const float4*>(src + 4);
                    __nv_bfloat16 tmpb[8] = {
                        __float2bfloat16(v0.x), __float2bfloat16(v0.y),
                        __float2bfloat16(v0.z), __float2bfloat16(v0.w),
                        __float2bfloat16(v1.x), __float2bfloat16(v1.y),
                        __float2bfloat16(v1.z), __float2bfloat16(v1.w)};
                    *reinterpret_cast<int4*>(dst) =
                        *reinterpret_cast<const int4*>(tmpb);
                } else {
                    #pragma unroll
                    for (uint32_t e = 0; e < 8u; ++e)
                        dst[e] = __float2bfloat16(0.0f);
                }
            }
        };

        auto compute = [&](uint32_t buf, uint32_t ks) {
            const uint8_t* const Ar = sa + buf * ABYT;
            const uint8_t* const Sr = ss + buf * SBYT;
            const float* const Br = sb + (size_t)buf * BN * BRS;
            const __nv_bfloat16* const Bh =
                sbh + (size_t)(ks & 1u) * BN * BKP;
            uint2 sv0[RG], sv8[RG];
            #pragma unroll
            for (uint32_t rg = 0; rg < RG; ++rg) {
                const uint32_t r0 = wr + rg * 16u + g;
                sv0[rg] = *reinterpret_cast<const uint2*>(Sr + r0 * SRS);
                sv8[rg] =
                    *reinterpret_cast<const uint2*>(Sr + (r0 + 8u) * SRS);
            }
            #pragma unroll
            for (uint32_t sk = 0; sk < NSUBK; ++sk) {
                const uint32_t ko = sk * 16u;
                uint32_t a[RG][4];
                #pragma unroll
                for (uint32_t rg = 0; rg < RG; ++rg) {
                    const uint32_t r0 = wr + rg * 16u + g, r8 = r0 + 8u;
                    const uint32_t se0 =
                        __byte_perm(sv0[rg].x, sv0[rg].y, sk) & 0xFFu;
                    const uint32_t se8 =
                        __byte_perm(sv8[rg].x, sv8[rg].y, sk) & 0xFFu;
                    const float s0 = (float)reinterpret_cast<
                        const __nv_fp8_e4m3&>(se0);
                    const float s8 = (float)reinterpret_cast<
                        const __nv_fp8_e4m3&>(se8);
                    const __nv_bfloat162 s0b = __floats2bfloat162_rn(s0, s0);
                    const __nv_bfloat162 s8b = __floats2bfloat162_rn(s8, s8);
                    const uint32_t s0u =
                        reinterpret_cast<const uint32_t&>(s0b);
                    const uint32_t s8u =
                        reinterpret_cast<const uint32_t&>(s8b);
                    if constexpr (FRAG) {
                        // one conflict-free LDS.32: word = const + lane
                        const uint32_t av = *reinterpret_cast<const uint32_t*>(
                            Ar + ((((warp % WR) * NSUBK + sk) * 8u + g) * 4u
                                  + t) * 4u);
                        a[rg][0] = pd_nv4tcv_dec2(av & 0xFFu, s0u);
                        a[rg][1] = pd_nv4tcv_dec2((av >> 8) & 0xFFu, s8u);
                        a[rg][2] = pd_nv4tcv_dec2((av >> 16) & 0xFFu, s0u);
                        a[rg][3] = pd_nv4tcv_dec2(av >> 24, s8u);
                    } else {
                        const uint2 av0 = *reinterpret_cast<const uint2*>(
                            Ar + r0 * ARS + 8u * sk);
                        const uint2 av8 = *reinterpret_cast<const uint2*>(
                            Ar + r8 * ARS + 8u * sk);
                        const uint32_t psel = 0x40u + t * 0x11u;
                        const uint32_t p0 = __byte_perm(av0.x, av0.y, psel);
                        const uint32_t p8 = __byte_perm(av8.x, av8.y, psel);
                        a[rg][0] = pd_nv4tcv_dec2(p0 & 0xFFu, s0u);
                        a[rg][1] = pd_nv4tcv_dec2(p8 & 0xFFu, s8u);
                        a[rg][2] = pd_nv4tcv_dec2((p0 >> 8) & 0xFFu, s0u);
                        a[rg][3] = pd_nv4tcv_dec2((p8 >> 8) & 0xFFu, s8u);
                    }
                }
                uint32_t b[CG][2];
                if constexpr (BONCE) {
                    #pragma unroll
                    for (uint32_t cg = 0; cg < CG; ++cg)
                        pd_bf16m_ldm_x2(
                            &Bh[(wc + cg * 8u + l7) * BKP + ko + b_kof],
                            b[cg][0], b[cg][1]);
                } else {
                    #pragma unroll
                    for (uint32_t cg = 0; cg < CG; ++cg) {
                        const float* c = Br + (wc + cg * 8u + g) * BRS + ko;
                        const __nv_bfloat162 b0 =
                            __floats2bfloat162_rn(c[2u * t], c[2u * t + 1u]);
                        const __nv_bfloat162 b1 =
                            __floats2bfloat162_rn(c[8u + 2u * t],
                                                  c[9u + 2u * t]);
                        b[cg][0] = reinterpret_cast<const uint32_t&>(b0);
                        b[cg][1] = reinterpret_cast<const uint32_t&>(b1);
                    }
                }
                #pragma unroll
                for (uint32_t rg = 0; rg < RG; ++rg)
                    #pragma unroll
                    for (uint32_t cg = 0; cg < CG; ++cg)
                        pd_bf16m_mma(acc_[rg][cg], a[rg], b[cg]);
            }
        };

        __syncthreads();
        if constexpr (BONCE) b_conv(0);
        #pragma unroll
        for (uint32_t s = 0; s + 1u < ST; ++s) {
            if (s < nk) stage(s * KT, s);
            else pd_bf16m_cpa_commit();
        }

        uint32_t p = 0;
        for (uint32_t ks = 0; ks < nk; ++ks) {
            pd_bf16m_cpa_waitN<(int)ST - 2>();
            __syncthreads();
            const uint32_t pre = ks + ST - 1u;
            if (pre < nk) stage(pre * KT, (p + ST - 1u) % ST);
            else pd_bf16m_cpa_commit();
            // next step's B conversion rides under this step's mma work; the
            // slot it writes was last READ two steps ago, ordered by the
            // syncthreads above
            if constexpr (BONCE)
                if (ks + 1u < nk) b_conv((ks + 1u) * KT);
            compute(p, ks);
            p = (p + 1u) % ST;
        }

        #pragma unroll
        for (uint32_t rg = 0; rg < RG; ++rg) {
            const uint32_t r0 = row_base + wr + rg * 16u + g;
            const uint32_t r8 = r0 + 8u;
            const float b0 = (hb && r0 < M) ? bias[r0] : 0.0f;
            const float b8 = (hb && r8 < M) ? bias[r8] : 0.0f;
            #pragma unroll
            for (uint32_t cg = 0; cg < CG; ++cg) {
                const uint32_t c0 = col_base + wc + cg * 8u + 2u * t;
                const uint32_t c1 = c0 + 1u;
                #pragma unroll
                for (uint32_t e = 0; e < 4u; ++e) {
                    const uint32_t r = (e < 2u) ? r0 : r8;
                    const uint32_t c = (e & 1u) ? c1 : c0;
                    if (r < M && c < N) {
                        float v = acc_[rg][cg][e] * scale2;
                        if (hb) v += (e < 2u) ? b0 : b8;
                        Y[(size_t)c * M + r] = v;
                    }
                }
            }
        }
    }
#else
    (void)data; (void)scale; (void)bias; (void)X; (void)Y; (void)scale2;
    (void)K; (void)M; (void)N;
#endif
}

// REPK=true reads a TILE-MAJOR weight layout - [row_tile][k_stage][row][...]
// with weights and scale records each contiguous per (tile, stage) block -
// so every stage's cp.async is one sequential 10.25 KB pull per CTA instead
// of 128 rows x 64 B at 1344 B stride. Requires M % BM == 0 and K % KT == 0
// (the repack pads to that). Same bytes, same decode, same bits.
template <uint32_t BM, uint32_t BN, uint32_t NWARP, uint32_t ST, uint32_t KT,
          uint32_t RG, uint32_t CG, bool BREG = false, uint32_t MINB = 1u,
          bool REPK = false, bool SPLIT = false>
__global__ void __launch_bounds__(NWARP * 32, MINB) pd_nvf4_gemm_tcp_kernel(
        const uint8_t* __restrict__ data, const uint8_t* __restrict__ scale,
        const float* __restrict__ bias, const float* __restrict__ X,
        float* __restrict__ Y, float scale2, uint32_t K, uint32_t M,
        uint32_t N) {
#if PD_BF16MMA_OK
    constexpr uint32_t NTH = NWARP * 32u;
    constexpr uint32_t WM = RG * 16u, WN = CG * 8u;
    constexpr uint32_t WR = BM / WM, WC = BN / WN;
    constexpr uint32_t NSUBK = KT / 16u;
    // raw shared strides. A rows pad to 16 B (cp.async chunk alignment) AND
    // an odd word count (20 words at KT=128) so the 8 rows a fragment phase
    // touches land on 8 distinct banks; B pads +4 f32 for the same reason.
    constexpr uint32_t ARS = KT / 2u + 16u;
    constexpr uint32_t SRS = KT / 16u;
    constexpr uint32_t BRS = KT + 4u;
    // BREG=true swaps the B ring from cp.async'd raw f32 to the tc kernel's
    // register-staged decoded-bf16 tile (KPAD stride, ldmatrix consumption).
    // That costs B its async copy (a short wait - X is L2-hot, ~350 KB for
    // the whole plane) but shrinks the stage by 8.2 KB, which at ST=2 puts
    // the CTA at ~39 KB and buys the SECOND resident CTA per SM - the
    // verdict on the ST=3 raw arm was occupancy-bound (16.6%, issue slots
    // idle 55%), not bandwidth- or compute-bound. MINB=2 pins the 128-reg
    // compile that residency needs.
    constexpr uint32_t BKP = KT + 8u;  // bf16 B stride (tc's KPAD)
    constexpr uint32_t ABYT = BM * ARS;
    constexpr uint32_t SBYT = BM * SRS;
    constexpr uint32_t ACH = BM * (KT / 32u);  // 16 B A chunks per stage
    constexpr uint32_t BCH = BN * (KT / 4u);   // 16 B B chunks per stage
    constexpr uint32_t BIT = (BN * (KT / 8u) + NTH - 1u) / NTH;  // BREG 8-groups/thread
    static_assert(WR * WC == NWARP, "warp grid");
    static_assert(WM * WR == BM && WN * WC == BN, "tile cover");
    static_assert(ST >= 2u && ST <= 4u, "ring depth");
    static_assert(BM <= NTH, "one scale record per thread");

    extern __shared__ __align__(16) uint8_t pd_nv4tcp_dyn[];
    uint8_t* const sa = pd_nv4tcp_dyn;
    uint8_t* const ss = sa + (size_t)ST * ABYT;
    float* const sb = reinterpret_cast<float*>(ss + (size_t)ST * SBYT);
    __nv_bfloat16* const sbh =
        reinterpret_cast<__nv_bfloat16*>(ss + (size_t)ST * SBYT);

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u, warp = tid >> 5;
    const uint32_t g = lane >> 2, t = lane & 3u;
    const uint32_t l7 = lane & 7u;
    const uint32_t b_kof = (lane & 8u) ? 8u : 0u;
    const uint32_t wr = (warp % WR) * WM;
    const uint32_t wc = (warp / WR) * WN;
    const uint32_t nk = K / KT;  // launcher gates K % KT == 0
    // SPLIT (the decode-starved grids: `down` at batch<=32 is 40 tiles on 148
    // SMs, so the persistent loop leaves 108 SMs idle). grid.y slices the
    // K-chunk walk; each slice accumulates only its share and writes RAW
    // partials (no scale2, no bias) at slice stride N*M, which
    // pd_nvf4_sk_reduce_kernel folds in fixed slice order - deterministic,
    // and the same numeric class as the unsplit kernel (f32 accumulate, the
    // only change is where the adds are grouped). Same design as
    // pd_mxfp4_gemm_nv4b_kernel's SPLIT.
    uint32_t k_lo = 0, nks = nk;
    if constexpr (SPLIT) {
        const uint32_t ck = (nk + gridDim.y - 1u) / gridDim.y;
        k_lo = blockIdx.y * ck;
        const uint32_t k_hi = (k_lo + ck) < nk ? (k_lo + ck) : nk;
        nks = k_hi > k_lo ? k_hi - k_lo : 0u;
    }
    const uint32_t mtiles = (M + BM - 1u) / BM;
    const uint32_t ntiles = (N + BN - 1u) / BN;
    const bool hb = bias != nullptr;

    for (uint32_t tile = blockIdx.x; tile < mtiles * ntiles;
         tile += gridDim.x) {
        const uint32_t row_base = (tile % mtiles) * BM;
        const uint32_t col_base = (tile / mtiles) * BN;

        // one stage = one commit group: A nibbles + scale records + B f32.
        // OOB rows/cols ride the size-0 zero-fill, so a dead row decodes as
        // 0 * (scale byte 0 -> 0.0f) = 0 - the tc a_ok=false path exactly.
        auto stage = [&](uint32_t k0, uint32_t buf) {
            #pragma unroll
            for (uint32_t it = 0; it < (ACH + NTH - 1u) / NTH; ++it) {
                const uint32_t i = tid + it * NTH;
                if (i < ACH) {
                    const uint32_t row = i / (KT / 32u), ch = i % (KT / 32u);
                    if constexpr (REPK) {
                        const size_t blk =
                            (size_t)(row_base / BM) * nk + k0 / KT;
                        pd_bf16m_cpa16(sa + buf * ABYT + row * ARS + ch * 16u,
                                       data + (blk * BM + row) * (KT >> 1)
                                           + ch * 16u,
                                       true);
                    } else {
                        const uint32_t gr = row_base + row;
                        pd_bf16m_cpa16(
                            sa + buf * ABYT + row * ARS + ch * 16u,
                            data + (size_t)gr * (K >> 1) + (k0 >> 1)
                                + ch * 16u,
                            gr < M);
                    }
                }
            }
            if (tid < BM) {
                if constexpr (REPK) {
                    const size_t blk = (size_t)(row_base / BM) * nk + k0 / KT;
                    pd_nv4_cpa8(ss + buf * SBYT + tid * SRS,
                                scale + (blk * BM + tid) * (KT >> 4), true);
                } else {
                    pd_nv4_cpa8(
                        ss + buf * SBYT + tid * SRS,
                        scale
                            + (size_t)(row_base + tid) * (K >> 4) + (k0 >> 4),
                        row_base + tid < M);
                }
            }
            if constexpr (!BREG) {
                #pragma unroll
                for (uint32_t it = 0; it < (BCH + NTH - 1u) / NTH; ++it) {
                    const uint32_t i = tid + it * NTH;
                    if (i < BCH) {
                        const uint32_t col = i / (KT / 4u), ch = i % (KT / 4u);
                        const uint32_t gc = col_base + col;
                        pd_bf16m_cpa16(sb + (size_t)buf * BN * BRS + col * BRS
                                           + ch * 4u,
                                       X + (size_t)gc * K + k0 + ch * 4u,
                                       gc < N);
                    }
                }
            }
            pd_bf16m_cpa_commit();
        };

        // BREG B path: the tc kernel's register-staged decoded-bf16 tile,
        // verbatim (same __float2bfloat16 per element, ldmatrix consumption
        // -> same fragment bits as the inline f32 pack). Dead code when
        // !BREG - the arrays fold away with the uncalled lambdas.
        float4 b_v0[BIT], b_v1[BIT];
        bool b_ok[BIT];
        auto b_load = [&](uint32_t k0) {
            #pragma unroll
            for (uint32_t it = 0; it < BIT; ++it) {
                const uint32_t i = tid + it * NTH;
                const uint32_t col = i / (KT / 8u), h8 = (i % (KT / 8u)) * 8u;
                const bool ok = i < BN * (KT / 8u) && (col_base + col) < N;
                b_ok[it] = ok;
                if (ok) {
                    const float* src =
                        X + (size_t)(col_base + col) * K + k0 + h8;
                    b_v0[it] = *reinterpret_cast<const float4*>(src);
                    b_v1[it] = *reinterpret_cast<const float4*>(src + 4);
                }
            }
        };
        auto b_store = [&](uint32_t buf) {
            #pragma unroll
            for (uint32_t it = 0; it < BIT; ++it) {
                const uint32_t i = tid + it * NTH;
                if (i >= BN * (KT / 8u)) continue;
                const uint32_t col = i / (KT / 8u), h8 = (i % (KT / 8u)) * 8u;
                __nv_bfloat16* dst =
                    &sbh[(size_t)buf * BN * BKP + col * BKP + h8];
                if (b_ok[it]) {
                    const float4 v0 = b_v0[it], v1 = b_v1[it];
                    __nv_bfloat16 tmpb[8] = {
                        __float2bfloat16(v0.x), __float2bfloat16(v0.y),
                        __float2bfloat16(v0.z), __float2bfloat16(v0.w),
                        __float2bfloat16(v1.x), __float2bfloat16(v1.y),
                        __float2bfloat16(v1.z), __float2bfloat16(v1.w)};
                    *reinterpret_cast<int4*>(dst) =
                        *reinterpret_cast<const int4*>(tmpb);
                } else {
                    #pragma unroll
                    for (uint32_t e = 0; e < 8u; ++e)
                        dst[e] = __float2bfloat16(0.0f);
                }
            }
        };

        auto compute = [&](uint32_t buf, float acc[RG][CG][4]) {
            const uint8_t* const Ar = sa + buf * ABYT;
            const uint8_t* const Sr = ss + buf * SBYT;
            const float* const Br = sb + (size_t)buf * BN * BRS;
            // scale records load once per stage per fragment row pair (u64),
            // one byte peeled per k16 below - the per-sk byte-load form put
            // 6 narrow LDS ops on every fragment and pinned the kernel on
            // L1/TEX slots (70.8% L1 vs 50% DRAM, 4.12 cyc/inst).
            uint2 sv0[RG], sv8[RG];
            #pragma unroll
            for (uint32_t rg = 0; rg < RG; ++rg) {
                const uint32_t r0 = wr + rg * 16u + g;
                sv0[rg] = *reinterpret_cast<const uint2*>(Sr + r0 * SRS);
                sv8[rg] =
                    *reinterpret_cast<const uint2*>(Sr + (r0 + 8u) * SRS);
            }
            #pragma unroll
            for (uint32_t sk = 0; sk < NSUBK; ++sk) {
                const uint32_t ko = sk * 16u;
                uint32_t a[RG][4];
                #pragma unroll
                for (uint32_t rg = 0; rg < RG; ++rg) {
                    const uint32_t r0 = wr + rg * 16u + g, r8 = r0 + 8u;
                    const uint32_t se0 =
                        __byte_perm(sv0[rg].x, sv0[rg].y, sk) & 0xFFu;
                    const uint32_t se8 =
                        __byte_perm(sv8[rg].x, sv8[rg].y, sk) & 0xFFu;
                    const float s0 = (float)reinterpret_cast<
                        const __nv_fp8_e4m3&>(se0);
                    const float s8 = (float)reinterpret_cast<
                        const __nv_fp8_e4m3&>(se8);
                    // one u64 per row covers the k16 group; both fragment
                    // bytes (t and t+4) peel with a single dynamic prmt
                    const uint2 av0 = *reinterpret_cast<const uint2*>(
                        Ar + r0 * ARS + 8u * sk);
                    const uint2 av8 = *reinterpret_cast<const uint2*>(
                        Ar + r8 * ARS + 8u * sk);
                    const uint32_t psel = 0x40u + t * 0x11u;
                    const uint32_t p0 = __byte_perm(av0.x, av0.y, psel);
                    const uint32_t p8 = __byte_perm(av8.x, av8.y, psel);
                    a[rg][0] = pd_nv4tcp_dec2(p0 & 0xFFu, s0);
                    a[rg][1] = pd_nv4tcp_dec2(p8 & 0xFFu, s8);
                    a[rg][2] = pd_nv4tcp_dec2((p0 >> 8) & 0xFFu, s0);
                    a[rg][3] = pd_nv4tcp_dec2((p8 >> 8) & 0xFFu, s8);
                }
                uint32_t b[CG][2];
                if constexpr (BREG) {
                    const __nv_bfloat16* const Bh =
                        sbh + (size_t)buf * BN * BKP;
                    #pragma unroll
                    for (uint32_t cg = 0; cg < CG; ++cg)
                        pd_bf16m_ldm_x2(
                            &Bh[(wc + cg * 8u + l7) * BKP + ko + b_kof],
                            b[cg][0], b[cg][1]);
                } else {
                    #pragma unroll
                    for (uint32_t cg = 0; cg < CG; ++cg) {
                        const float* c = Br + (wc + cg * 8u + g) * BRS + ko;
                        const __nv_bfloat162 b0 =
                            __floats2bfloat162_rn(c[2u * t], c[2u * t + 1u]);
                        const __nv_bfloat162 b1 =
                            __floats2bfloat162_rn(c[8u + 2u * t],
                                                  c[9u + 2u * t]);
                        b[cg][0] = reinterpret_cast<const uint32_t&>(b0);
                        b[cg][1] = reinterpret_cast<const uint32_t&>(b1);
                    }
                }
                #pragma unroll
                for (uint32_t rg = 0; rg < RG; ++rg)
                    #pragma unroll
                    for (uint32_t cg = 0; cg < CG; ++cg)
                        pd_bf16m_mma(acc[rg][cg], a[rg], b[cg]);
            }
        };

        float acc[RG][CG][4] = {};

        // tile-top barrier: every warp is out of the previous tile's compute
        // before the prologue restages those buffers (the step-level barrier
        // below only orders within a tile).
        __syncthreads();
        #pragma unroll
        for (uint32_t s = 0; s + 1u < ST; ++s) {
            if (s < nks) {
                stage((k_lo + s) * KT, s);
                if constexpr (BREG) {
                    b_load((k_lo + s) * KT);
                    b_store(s);
                }
            } else pd_bf16m_cpa_commit();  // empty group keeps counting uniform
        }

        // per step: wait until stage ks arrived (ST-2 newer groups may still
        // fly), barrier, issue stage ks+ST-1 into the buffer computed at step
        // ks-1 (the barrier just ordered that WAR), then compute. Tail steps
        // issue empty commit groups so wait_group<ST-2> keeps meaning "stage
        // ks is home" all the way to ks = nk-1 - without them the last ST-2
        // steps would compute a stage the wait no longer covers.
        uint32_t p = 0;
        for (uint32_t ks = 0; ks < nks; ++ks) {
            pd_bf16m_cpa_waitN<(int)ST - 2>();
            __syncthreads();
            const uint32_t pre = ks + ST - 1u;
            if (pre < nks) {
                stage((k_lo + pre) * KT, (p + ST - 1u) % ST);
                if constexpr (BREG) b_load((k_lo + pre) * KT);
            } else pd_bf16m_cpa_commit();
            compute(p, acc);
            // BREG: land the B regs after compute so the loads fly under the
            // mma work - tc's stage_store placement, B slice only
            if constexpr (BREG)
                if (pre < nks) b_store((p + ST - 1u) % ST);
            p = (p + 1u) % ST;
        }

        // epilogue: the tc kernel's VERBATIM order (acc*scale2, then bias
        // only when one exists - the lattice bit-exactness contract).
        // SPLIT writes the RAW slice partial instead; pd_nvf4_sk_reduce_kernel
        // owns scale2+bias once, in the same order, after the fold.
        #pragma unroll
        for (uint32_t rg = 0; rg < RG; ++rg) {
            const uint32_t r0 = row_base + wr + rg * 16u + g;
            const uint32_t r8 = r0 + 8u;
            const float b0 = (hb && r0 < M) ? bias[r0] : 0.0f;
            const float b8 = (hb && r8 < M) ? bias[r8] : 0.0f;
            #pragma unroll
            for (uint32_t cg = 0; cg < CG; ++cg) {
                const uint32_t c0 = col_base + wc + cg * 8u + 2u * t;
                const uint32_t c1 = c0 + 1u;
                #pragma unroll
                for (uint32_t e = 0; e < 4u; ++e) {
                    const uint32_t r = (e < 2u) ? r0 : r8;
                    const uint32_t c = (e & 1u) ? c1 : c0;
                    if (r < M && c < N) {
                        if constexpr (SPLIT) {
                            Y[(size_t)blockIdx.y * N * M + (size_t)c * M + r] =
                                acc[rg][cg][e];
                        } else {
                            float v = acc[rg][cg][e] * scale2;
                            if (hb) v += (e < 2u) ? b0 : b8;
                            Y[(size_t)c * M + r] = v;
                        }
                    }
                }
            }
        }
    }
#else
    (void)data; (void)scale; (void)bias; (void)X; (void)Y; (void)scale2;
    (void)K; (void)M; (void)N;
#endif
}

#ifdef PD_BS_HOST
template <uint32_t BM, uint32_t BN, uint32_t NW, uint32_t ST, uint32_t KT,
          uint32_t RG, uint32_t CG, bool BREG = false, uint32_t MINB = 1u,
          bool REPK = false>
static int pd_nvf4_tcp_cfg(const uint8_t* data, const uint8_t* scale,
                           const float* bias, const float* x, float* y,
                           float scale2, uint32_t in_dim, uint32_t out_dim,
                           uint32_t batch, cudaStream_t st) {
    constexpr uint32_t smem =
        ST * (BM * (KT / 2u + 16u) + BM * (KT / 16u)
              + (BREG ? BN * (KT + 8u) * 2u : BN * (KT + 4u) * 4u));
    static bool set = false;
    if (!set) {
        cudaFuncSetAttribute(
            pd_nvf4_gemm_tcp_kernel<BM, BN, NW, ST, KT, RG, CG, BREG, MINB,
                                    REPK>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        set = true;
    }
    static int nsm = 0;
    if (nsm == 0) {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
        if (nsm <= 0) nsm = 128;
    }
    const uint32_t total =
        ((out_dim + BM - 1u) / BM) * ((batch + BN - 1u) / BN);
    const uint32_t cap = (uint32_t)nsm * MINB;
    const uint32_t grid = total < cap ? total : cap;
    pd_nvf4_gemm_tcp_kernel<BM, BN, NW, ST, KT, RG, CG, BREG, MINB, REPK>
        <<<grid, NW * 32u, smem, st>>>(data, scale, bias, x, y, scale2,
                                       in_dim, out_dim, batch);
    return pd_launch_status();
}

// Pack-owned partial-sum scratch for the split-K arm, keyed by STREAM so two
// concurrent lanes can never fold into each other's partials.
//
// Two hard constraints, both learned the expensive way (qwen3.8
// c32 fell 700 -> 17.9 tok/s on the first cut of this):
//  1. The decode tick is CUDA-GRAPH CAPTURED. cudaMalloc inside a capture is
//     illegal and takes the whole capture down with it, so the buffer is
//     never allocated while `st` is capturing - the caller falls back to the
//     unsplit arm for that pass instead.
//  2. qwen35/batch.rs:563 - "the captured decode graphs bake scratch
//     ADDRESSES". So the buffer is allocated once at a size that covers the
//     whole decode band and is never freed or grown: freeing it after a
//     graph baked the pointer is a use-after-free on replay. A request that
//     exceeds what we already hold returns null (unsplit arm), it does not
//     reallocate.
// The buffer is small (sk 8 x 64 rows x out_dim f32; 35.7 MB at the widest
// FFN plane) and the pack outlives every consumer.
static float* pd_nvf4_sk_scratch(size_t nfloats, size_t reserve,
                                 cudaStream_t st) {
    struct Slot { cudaStream_t st; float* p; size_t n; };
    static Slot slots[8] = {};
    static int used = 0;
    for (int i = 0; i < used; ++i) {
        if (slots[i].st != st) continue;
        return slots[i].n >= nfloats ? slots[i].p : nullptr;  // never regrow
    }
    if (used >= 8) return nullptr;
    // allocation is only ever legal outside a capture
    cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(st, &cs) != cudaSuccess ||
        cs != cudaStreamCaptureStatusNone)
        return nullptr;
    const size_t want = reserve > nfloats ? reserve : nfloats;
    float* p = nullptr;
    if (cudaMalloc(&p, want * sizeof(float)) != cudaSuccess) {
        cudaGetLastError();
        return nullptr;
    }
    slots[used] = Slot{st, p, want};
    ++used;
    return p;
}

// Split-K twin: grid.y slices the K walk into `sk` ranges writing raw
// partials into `part` (>= sk * batch * out_dim floats), then one
// elementwise reduce folds them with the epilogue (scale2 + bias), in fixed
// slice order. For the decode-band FFN grids the unsplit persistent loop
// starves the machine - `down` (out 5120 => 40 tiles) leaves 108 of 148 SMs
// idle while each CTA walks all K=17408.
template <uint32_t BM, uint32_t BN, uint32_t NW, uint32_t ST, uint32_t KT,
          uint32_t RG, uint32_t CG, bool BREG = false, uint32_t MINB = 1u,
          bool REPK = false>
static int pd_nvf4_tcp_sk_cfg(const uint8_t* data, const uint8_t* scale,
                              const float* bias, const float* x, float* part,
                              float* y, float scale2, uint32_t in_dim,
                              uint32_t out_dim, uint32_t batch, uint32_t sk,
                              cudaStream_t st) {
    constexpr uint32_t smem =
        ST * (BM * (KT / 2u + 16u) + BM * (KT / 16u)
              + (BREG ? BN * (KT + 8u) * 2u : BN * (KT + 4u) * 4u));
    static bool set = false;
    if (!set) {
        cudaFuncSetAttribute(
            pd_nvf4_gemm_tcp_kernel<BM, BN, NW, ST, KT, RG, CG, BREG, MINB,
                                    REPK, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        set = true;
    }
    static int nsm = 0;
    if (nsm == 0) {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
        if (nsm <= 0) nsm = 128;
    }
    if (sk < 2u || in_dim / KT < sk) return cudaErrorInvalidValue;
    const uint32_t total =
        ((out_dim + BM - 1u) / BM) * ((batch + BN - 1u) / BN);
    // the per-slice cap is the whole-die cap divided by the slice count:
    // grid.x * sk CTAs are resident at once
    const uint32_t cap = (uint32_t)nsm * MINB;
    const uint32_t gx = total < cap ? total : cap;
    dim3 grid(gx, sk);
    pd_nvf4_gemm_tcp_kernel<BM, BN, NW, ST, KT, RG, CG, BREG, MINB, REPK, true>
        <<<grid, NW * 32u, smem, st>>>(data, scale, bias, x, part, scale2,
                                       in_dim, out_dim, batch);
    const int rc = pd_launch_status();
    if (rc != 0) return rc;
    const uint32_t n = batch * out_dim;
    pd_nvf4_sk_reduce_kernel<<<(n + 255u) / 256u, 256u, 0, st>>>(
        part, bias, y, scale2, n, out_dim, sk);
    return pd_launch_status();
}
#endif

// Tensor-core NVFP4 GEMM entry (the batched lm_head class). The Rust route
// elects this over the scalar lane for wide planes at batch >= 2 - the
// probe has tc at 253-291 us vs mr's 577 (b8) / 1155 (b32) at the vocab
// shape, while batch 1 keeps the per-row exact-f32 gemv (138 us at 91% of
// roof; this kernel's barrier-paced ring only reaches ~50% at b1). K%16 is
// a hard gate (the u64 nibble loads and B float4 loads both need it; every
// served NVFP4 plane has K%32==0).
PD_EXPORT
int pd_nvf4_gemm_tc(const void* data, const void* scale, const void* bias,
                    const void* x, void* y, float scale2, uint32_t in_dim,
                    uint32_t out_dim, uint32_t batch, void* stream) {
#ifndef PD_BS_HOST
    (void)data; (void)scale; (void)bias; (void)x; (void)y; (void)scale2;
    (void)in_dim; (void)out_dim; (void)batch; (void)stream;
    return cudaErrorNotSupported;
#else
    if (out_dim == 0 || batch == 0) return 0;
    if ((in_dim & 15u) != 0) return cudaErrorInvalidValue;
    // Persistent raw-ring arm first (the lm_head rung): K % 128
    // is its full-stage contract (every served NVFP4 plane satisfies it -
    // the lm_head K = 2688 = 21*128). Kill: PADDOCK_NO_NVF4_TCP.
    static const bool tcp_off = pd_env("PADDOCK_NO_NVF4_TCP") != nullptr;
    if (!tcp_off && (in_dim & 127u) == 0u) {
        // Split-K for the machine-starved grids. This kernel was
        // tuned at the lm_head shape (out 131072 => 1024 row-tiles, K 2688);
        // the qwen3.8 FFN `down` plane is its inverse - out 5120 => 40 tiles
        // on 148 SMs with K 17408, so the persistent loop leaves 108 SMs idle
        // for the whole walk. Measured at the live shapes, b32:
        //   down   239.5 us -> sk8  90.0 us  (2.66x)      [40 tiles]
        //   gate   73.8 us  -> sk8  71.8 us  (flat)       [136 tiles]
        // so the split is elected only on the starved grids, where it is a
        // pure parallelism fix. Above ~64 tiles the die is already full and
        // the extra partial-plane write/read is a wash-to-loss.
        // Numerics: the split REGROUPS the f32 accumulate (fixed slice order,
        // deterministic run to run, but not bit-identical to the unsplit
        // walk - probe max rel 1.3e-3 on adversarial random planes). It is
        // therefore off the lm_head/nemotron lattice-gate path and on by
        // default only for the FFN decode grids.
        //
        // OPT-IN (PADDOCK_NVF4_TCPSK=1), not default-on, and deliberately so:
        // the partial buffer has to be alive and at a STABLE address before
        // the decode graph captures (qwen35/batch.rs:563 bakes scratch
        // addresses), which a pack-owned lazy allocation cannot guarantee on
        // the capture stream - it can only allocate on a pass that runs
        // eagerly. Banking this arm by default needs the buffer plumbed from
        // the caller the way `part` already is for pd_nvf4_gemm_f4s. Until
        // then the knob exists for measurement, and the shipped path is the
        // unsplit walk. Measured worth: `down` 239.5 -> 90.0 us at b32
        // (2.66x), ~1.36x on the whole tick.
        static const bool sk_off = pd_env("PADDOCK_NVF4_TCPSK") == nullptr;
        const uint32_t tiles = ((out_dim + 127u) / 128u) * ((batch + 31u) / 32u);
        const uint32_t sk = 8u;
        float* part = nullptr;
        if (!sk_off && tiles < 64u && in_dim / 128u >= sk)
            part = pd_nvf4_sk_scratch((size_t)sk * batch * out_dim,
                                      (size_t)sk * 64u * out_dim,
                                      (cudaStream_t)stream);
        if (part)
            return pd_nvf4_tcp_sk_cfg<128u, 32u, 8u, 3u, 128u, 1u, 4u>(
                (const uint8_t*)data, (const uint8_t*)scale,
                (const float*)bias, (const float*)x, part, (float*)y, scale2,
                in_dim, out_dim, batch, sk, (cudaStream_t)stream);
        return pd_nvf4_tcp_cfg<128u, 32u, 8u, 3u, 128u, 1u, 4u>(
            (const uint8_t*)data, (const uint8_t*)scale, (const float*)bias,
            (const float*)x, (float*)y, scale2, in_dim, out_dim, batch,
            (cudaStream_t)stream);
    }
    // election measured at the lm_head shape (131072 x
    // 2688 b32): tcC = ST=2/KT=128 299.5 us vs tcA (ST=2/KT=64, 2 CTA/SM)
    // 395.6, tcB (ST=3/KT=64) 413.2, tcD (BM=64, 3 CTA/SM) 405.9 - the
    // barrier count (21 vs 42) beats the extra CTA residency; the scalar
    // b2 arm above does 1155 on the same plane.
    return pd_nvf4_tc_cfg<128u, 32u, 8u, 2u, 128u, 2u, 2u>(
        (const uint8_t*)data, (const uint8_t*)scale, (const float*)bias,
        (const float*)x, (float*)y, scale2, in_dim, out_dim, batch,
        (cudaStream_t)stream);
#endif
}

// ---- tile-major (TM) plane twins (lm_head repack rung) --------
// The loader repacks a plane to [row_tile 128][k_stage KT=128][row] with
// weights (64 B/row/stage) and e4m3 scale records (8 B/row/stage) each
// contiguous per (tile, stage) block, out_dim padded to 128 rows and
// ZERO-filled (a pad row decodes 0 * scale-byte-0 = 0, and every consumer
// guards its stores at out_dim). That turns the tcp stage's 128 x 64 B
// strided pulls into one sequential 10.25 KB block per (tile, stage) -
// probe: tcpC 225 -> tcpR 205 us b32 / 180 us b8, +9-12%. These entries are
// the same kernels reading that layout; each is bit-exact vs its row-major
// twin (shared dot walk / mma chain - only the weight+scale addressing
// moves). in_dim % 128 == 0 is the layout's contract, checked here so a
// mis-elected upload fails loudly instead of walking garbage.

PD_EXPORT
int pd_nvf4_gemv_batch_tm(const void* data, const void* scale,
                          const void* bias, const void* x, void* y,
                          float scale2, uint32_t in_dim, uint32_t out_dim,
                          uint32_t batch, void* stream) {
#ifndef PD_BS_HOST
    (void)data; (void)scale; (void)bias; (void)x; (void)y; (void)scale2;
    (void)in_dim; (void)out_dim; (void)batch; (void)stream;
    return cudaErrorNotSupported;
#else
    if (out_dim == 0 || batch == 0) return 0;
    if ((in_dim & 127u) != 0) return cudaErrorInvalidValue;
    const uint32_t rows_per_cta = 8u;
    dim3 grid((out_dim + rows_per_cta - 1u) / rows_per_cta, batch);
    pd_nvf4_gemv_batch_kernel<true><<<grid, rows_per_cta * 32u, 0,
                                      (cudaStream_t)stream>>>(
        (const uint8_t*)data, (const uint8_t*)scale, (const float*)bias,
        (const float*)x, (float*)y, scale2, in_dim, out_dim);
    return pd_launch_status();
#endif
}

PD_EXPORT
int pd_nvf4_gemm_mr_tm(const void* data, const void* scale, const void* bias,
                       const void* x, void* y, float scale2, uint32_t in_dim,
                       uint32_t out_dim, uint32_t batch, void* stream) {
#ifndef PD_BS_HOST
    (void)data; (void)scale; (void)bias; (void)x; (void)y; (void)scale2;
    (void)in_dim; (void)out_dim; (void)batch; (void)stream;
    return cudaErrorNotSupported;
#else
    if (out_dim == 0 || batch == 0) return 0;
    if ((in_dim & 127u) != 0) return cudaErrorInvalidValue;
    constexpr uint32_t BR = 16u;
    const uint32_t rows_per_cta = 8u;
    // same width gate as pd_nvf4_gemm_mr - the arms differ only in layout
    if (out_dim >= 4096u) {
        constexpr uint32_t BN = 2u;
        dim3 grid((out_dim + rows_per_cta * BN - 1u) / (rows_per_cta * BN),
                  (batch + BR - 1u) / BR);
        pd_nvf4_gemm_mr_kernel<BR, BN, 1u, true>
            <<<grid, rows_per_cta * 32u, 0, (cudaStream_t)stream>>>(
                (const uint8_t*)data, (const uint8_t*)scale,
                (const float*)bias, (const float*)x, (float*)y, scale2,
                in_dim, out_dim, batch);
        return pd_launch_status();
    }
    dim3 grid((out_dim + rows_per_cta - 1u) / rows_per_cta,
              (batch + BR - 1u) / BR);
    pd_nvf4_gemm_mr_kernel<BR, 1u, 1u, true>
        <<<grid, rows_per_cta * 32u, 0, (cudaStream_t)stream>>>(
            (const uint8_t*)data, (const uint8_t*)scale, (const float*)bias,
            (const float*)x, (float*)y, scale2, in_dim, out_dim, batch);
    return pd_launch_status();
#endif
}

#ifdef PD_BS_HOST
template <uint32_t BM, uint32_t BN, uint32_t NW, uint32_t ST, uint32_t KT,
          uint32_t RG, uint32_t CG, bool BONCE, uint32_t MINB,
          bool FRAG = false>
static int pd_nvf4_tcv_cfg(const uint8_t* data, const uint8_t* scale,
                           const float* bias, const float* x, float* y,
                           float scale2, uint32_t in_dim, uint32_t out_dim,
                           uint32_t batch, cudaStream_t st) {
    constexpr uint32_t smem =
        ST * ((FRAG ? BM * (KT / 2u) : BM * (KT / 2u + 16u))
              + BM * (KT / 16u))
        + (BONCE ? 2u * BN * (KT + 8u) * 2u : ST * BN * (KT + 4u) * 4u);
    static bool set = false;
    if (!set) {
        cudaFuncSetAttribute(
            pd_nvf4_gemm_tcv_kernel<BM, BN, NW, ST, KT, RG, CG, BONCE, MINB,
                                    FRAG>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        set = true;
    }
    static int nsm = 0;
    if (nsm == 0) {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
        if (nsm <= 0) nsm = 128;
    }
    const uint32_t total =
        ((out_dim + BM - 1u) / BM) * ((batch + BN - 1u) / BN);
    const uint32_t cap = (uint32_t)nsm * MINB;
    const uint32_t grid = total < cap ? total : cap;
    pd_nvf4_gemm_tcv_kernel<BM, BN, NW, ST, KT, RG, CG, BONCE, MINB, FRAG>
        <<<grid, NW * 32u, smem, st>>>(data, scale, bias, x, y, scale2,
                                       in_dim, out_dim, batch);
    return pd_launch_status();
}
#endif

// The tc entry's TM twin. Election (the fragment-order rung):
// the tcv arm - v2 bf16-table decode (pd_nv4tcv_dec2, bit-identical
// fragments) + B converted once per stage into a shared bf16 tile read via
// ldmatrix (kills the 8x per-warp B re-read that owned 73% of tcp's L1
// wavefronts) - at ST=2 the stage shrinks to 39.9 KB and two CTAs fit per
// SM: probe 205.3 (tcp-REPK) -> 167.0 us b32 / 151.3 b8, bit-exact vs the
// whole tc family. Kill PADDOCK_NO_NVF4_TCV pins the tcp REPK arm for an
// A/B leg on the same plane layout. There is no row-major fallback here
// (the TM lane's own switch is the Rust-side PADDOCK_NO_NVF4_TM, which
// keeps the plane row-major so every row-major election stays reachable).
PD_EXPORT
int pd_nvf4_gemm_tc_tm(const void* data, const void* scale, const void* bias,
                       const void* x, void* y, float scale2, uint32_t in_dim,
                       uint32_t out_dim, uint32_t batch, void* stream) {
#ifndef PD_BS_HOST
    (void)data; (void)scale; (void)bias; (void)x; (void)y; (void)scale2;
    (void)in_dim; (void)out_dim; (void)batch; (void)stream;
    return cudaErrorNotSupported;
#else
    if (out_dim == 0 || batch == 0) return 0;
    if ((in_dim & 127u) != 0) return cudaErrorInvalidValue;
    static const bool tcv_off = pd_env("PADDOCK_NO_NVF4_TCV") != nullptr;
    if (!tcv_off)
        return pd_nvf4_tcv_cfg<128u, 32u, 8u, 2u, 128u, 1u, 4u, true, 2u>(
            (const uint8_t*)data, (const uint8_t*)scale, (const float*)bias,
            (const float*)x, (float*)y, scale2, in_dim, out_dim, batch,
            (cudaStream_t)stream);
    return pd_nvf4_tcp_cfg<128u, 32u, 8u, 3u, 128u, 1u, 4u, false, 1u, true>(
        (const uint8_t*)data, (const uint8_t*)scale, (const float*)bias,
        (const float*)x, (float*)y, scale2, in_dim, out_dim, batch,
        (cudaStream_t)stream);
#endif
}

// ---- FRAGMENT-layout (TF) plane twins (fragment rung) ----------
// The loader permutes each tile-major 8 KB (tile, stage) block to
// [w:8][k16:8][g:8][u32 t0..t3] - u32 (w,sk,g,t) holds lane (g,t)'s a0..a3
// mma fragment bytes - while scales keep the tile-major [row][8B] order.
// The tc arm reads one conflict-free LDS.32 per (sk, rg) with flat 8 KB
// stages (probe: tcvC 167.0 -> tcvF 159.2 us b32, tcvF8 144.2 b8 = marlin
// parity on the kernel); the gemv/mr twins re-address the same bytes and
// stay bit-exact per class.

PD_EXPORT
int pd_nvf4_gemm_mr_tf(const void* data, const void* scale, const void* bias,
                       const void* x, void* y, float scale2, uint32_t in_dim,
                       uint32_t out_dim, uint32_t batch, void* stream) {
#ifndef PD_BS_HOST
    (void)data; (void)scale; (void)bias; (void)x; (void)y; (void)scale2;
    (void)in_dim; (void)out_dim; (void)batch; (void)stream;
    return cudaErrorNotSupported;
#else
    if (out_dim == 0 || batch == 0) return 0;
    if ((in_dim & 127u) != 0) return cudaErrorInvalidValue;
    constexpr uint32_t BR = 16u;
    const uint32_t rows_per_cta = 8u;
    if (out_dim >= 4096u) {
        constexpr uint32_t BN = 2u;
        dim3 grid((out_dim + rows_per_cta * BN - 1u) / (rows_per_cta * BN),
                  (batch + BR - 1u) / BR);
        pd_nvf4_gemm_mr_kernel<BR, BN, 1u, false, true>
            <<<grid, rows_per_cta * 32u, 0, (cudaStream_t)stream>>>(
                (const uint8_t*)data, (const uint8_t*)scale,
                (const float*)bias, (const float*)x, (float*)y, scale2,
                in_dim, out_dim, batch);
        return pd_launch_status();
    }
    dim3 grid((out_dim + rows_per_cta - 1u) / rows_per_cta,
              (batch + BR - 1u) / BR);
    pd_nvf4_gemm_mr_kernel<BR, 1u, 1u, false, true>
        <<<grid, rows_per_cta * 32u, 0, (cudaStream_t)stream>>>(
            (const uint8_t*)data, (const uint8_t*)scale, (const float*)bias,
            (const float*)x, (float*)y, scale2, in_dim, out_dim, batch);
    return pd_launch_status();
#endif
}

// The tc entry over a fragment plane: the tcv arm's FRAG instantiation at
// the probe-elected <ST=2, 2 CTA/SM, BONCE> config. No sub-switch here -
// the TF lane's A/B switch is the Rust-side PADDOCK_NO_NVF4_TF, which
// falls back to the tiled upload and its whole election tree.
PD_EXPORT
int pd_nvf4_gemm_tc_tf(const void* data, const void* scale, const void* bias,
                       const void* x, void* y, float scale2, uint32_t in_dim,
                       uint32_t out_dim, uint32_t batch, void* stream) {
#ifndef PD_BS_HOST
    (void)data; (void)scale; (void)bias; (void)x; (void)y; (void)scale2;
    (void)in_dim; (void)out_dim; (void)batch; (void)stream;
    return cudaErrorNotSupported;
#else
    if (out_dim == 0 || batch == 0) return 0;
    if ((in_dim & 127u) != 0) return cudaErrorInvalidValue;
    return pd_nvf4_tcv_cfg<128u, 32u, 8u, 2u, 128u, 1u, 4u, true, 2u, true>(
        (const uint8_t*)data, (const uint8_t*)scale, (const float*)bias,
        (const float*)x, (float*)y, scale2, in_dim, out_dim, batch,
        (cudaStream_t)stream);
#endif
}

// Token-batched expert up GEMV + fused squared-relu (nemotron experts carry
// no gate matrix - the activation is relu(up(x))^2, applied on the scaled
// value so the epilogue order matches the reference: scale2 first, then
// relu, then square). One warp per output row, 8 rows/CTA, one slot
// (= token x pick) per blockIdx.y. Serves the shared expert too: k=1 with
// idx pointing at a constant 0 over the 1-expert shared plane.
__global__ void pd_nvf4_moe_up_relu2_kernel(
    const uint8_t* __restrict__ data, const uint8_t* __restrict__ scale,
    const float* __restrict__ scale2, const uint32_t* __restrict__ idx,
    const float* __restrict__ x, float* __restrict__ y, uint32_t in_dim,
    uint32_t ff, uint32_t k) {
#if PD_NV4_OK
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const uint32_t r = blockIdx.x * (blockDim.x >> 5) + warp;
    if (r >= ff) return;
    const uint32_t slot = blockIdx.y;
    const uint32_t e = idx[slot];
    const float* xrow = x + (size_t)(slot / k) * in_dim;
    const uint8_t* row = data + ((size_t)e * ff + r) * (in_dim >> 1);
    const uint8_t* srow = scale + ((size_t)e * ff + r) * (in_dim >> 4);
    float acc = 0.0f;
    // tail hoisted out of the walk (the pd_nvf4_gemv_kernel fix: an in-loop
    // per-lane break blocks load pipelining, -24% on the DRAM-cold sweep)
    const uint32_t full = in_dim & ~127u;
    #pragma unroll 4
    for (uint32_t k0 = 0; k0 < full; k0 += 128u)
        acc += pd_nvf4_dot4(row, srow, xrow, k0 + lane * 4u);
    if (full < in_dim) {
        const uint32_t el = full + lane * 4u;
        if (el < in_dim) acc += pd_nvf4_dot4(row, srow, xrow, el);
    }
    for (uint32_t s = 16; s > 0; s >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, s);
    if (lane == 0) {
        const float v = fmaxf(acc * scale2[e], 0.0f);
        y[(size_t)slot * ff + r] = v * v;
    }
#else
    (void)data; (void)scale; (void)scale2; (void)idx; (void)x; (void)y;
    (void)in_dim; (void)ff; (void)k;
#endif
}

PD_EXPORT
int pd_nvf4_moe_up_relu2(const void* data, const void* scale,
                         const void* scale2, const void* idx, const void* x,
                         void* y, uint32_t in_dim, uint32_t ff, uint32_t k,
                         uint32_t batch, void* stream) {
#ifndef PD_BS_HOST
    (void)data; (void)scale; (void)scale2; (void)idx; (void)x; (void)y;
    (void)in_dim; (void)ff; (void)k; (void)batch; (void)stream;
    return cudaErrorNotSupported;
#else
    if (ff == 0 || batch == 0 || k == 0) return 0;
    if ((in_dim & 31u) != 0) return cudaErrorInvalidValue;
    const uint32_t rows_per_cta = 8u;
    dim3 grid((ff + rows_per_cta - 1u) / rows_per_cta, batch * k);
    pd_nvf4_moe_up_relu2_kernel<<<grid, rows_per_cta * 32u, 0,
                                  (cudaStream_t)stream>>>(
        (const uint8_t*)data, (const uint8_t*)scale, (const float*)scale2,
        (const uint32_t*)idx, (const float*)x, (float*)y, in_dim, ff, k);
    return pd_launch_status();
#endif
}

// Token-batched expert GATE+up GEMV with a fused swiglu - the qwen4_exp
// (Qwen3.8-Flash-Next) MoE shape. Every existing NVFP4 expert consumer here is
// nemotron's `relu(up(x))^2`, which has no gate matrix; this family has both a
// gate and an up plane and needs `silu(gate(x)) * up(x)`.
//
// One warp per output row computes both dots so the token's x row is read once
// for the pair (it stays in L1 across the two weight streams). Same walk, same
// hoisted tail, same scale2-then-activation epilogue order as the relu2 twin -
// so a row of this kernel is the relu2 kernel's row arithmetic twice, and any
// numeric question about one answers the other.
__global__ void pd_q4x_moe_gu_swiglu_kernel(
    const uint8_t* __restrict__ gdata, const uint8_t* __restrict__ gscale,
    const float* __restrict__ gscale2, const uint8_t* __restrict__ udata,
    const uint8_t* __restrict__ uscale, const float* __restrict__ uscale2,
    const uint32_t* __restrict__ idx, const float* __restrict__ x,
    float* __restrict__ y, uint32_t in_dim, uint32_t ff, uint32_t k) {
#if PD_NV4_OK
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const uint32_t r = blockIdx.x * (blockDim.x >> 5) + warp;
    if (r >= ff) return;
    const uint32_t slot = blockIdx.y;
    const uint32_t e = idx[slot];
    const float* xrow = x + (size_t)(slot / k) * in_dim;
    const size_t wo = ((size_t)e * ff + r) * (in_dim >> 1);
    const size_t so = ((size_t)e * ff + r) * (in_dim >> 4);
    const uint8_t* grow = gdata + wo;
    const uint8_t* gsrow = gscale + so;
    const uint8_t* urow = udata + wo;
    const uint8_t* usrow = uscale + so;
    float ga = 0.0f, ua = 0.0f;
    const uint32_t full = in_dim & ~127u;
    #pragma unroll 4
    for (uint32_t k0 = 0; k0 < full; k0 += 128u) {
        const uint32_t el = k0 + lane * 4u;
        ga += pd_nvf4_dot4(grow, gsrow, xrow, el);
        ua += pd_nvf4_dot4(urow, usrow, xrow, el);
    }
    if (full < in_dim) {
        const uint32_t el = full + lane * 4u;
        if (el < in_dim) {
            ga += pd_nvf4_dot4(grow, gsrow, xrow, el);
            ua += pd_nvf4_dot4(urow, usrow, xrow, el);
        }
    }
    for (uint32_t s = 16; s > 0; s >>= 1) {
        ga += __shfl_down_sync(0xffffffffu, ga, s);
        ua += __shfl_down_sync(0xffffffffu, ua, s);
    }
    if (lane == 0) {
        const float g = ga * gscale2[e];
        const float u = ua * uscale2[e];
        y[(size_t)slot * ff + r] = g * (1.0f / (1.0f + expf(-g))) * u;
    }
#else
    (void)gdata; (void)gscale; (void)gscale2; (void)udata; (void)uscale;
    (void)uscale2; (void)idx; (void)x; (void)y; (void)in_dim; (void)ff; (void)k;
#endif
}

PD_EXPORT
int pd_q4x_moe_gu_swiglu(const void* gdata, const void* gscale,
                         const void* gscale2, const void* udata,
                         const void* uscale, const void* uscale2,
                         const void* idx, const void* x, void* y,
                         uint32_t in_dim, uint32_t ff, uint32_t k,
                         uint32_t batch, void* stream) {
#ifndef PD_BS_HOST
    (void)gdata; (void)gscale; (void)gscale2; (void)udata; (void)uscale;
    (void)uscale2; (void)idx; (void)x; (void)y; (void)in_dim; (void)ff;
    (void)k; (void)batch; (void)stream;
    return cudaErrorNotSupported;
#else
    if (ff == 0 || batch == 0 || k == 0) return 0;
    if ((in_dim & 31u) != 0) return cudaErrorInvalidValue;
    const uint32_t rows_per_cta = 8u;
    dim3 grid((ff + rows_per_cta - 1u) / rows_per_cta, batch * k);
    pd_q4x_moe_gu_swiglu_kernel<<<grid, rows_per_cta * 32u, 0,
                                  (cudaStream_t)stream>>>(
        (const uint8_t*)gdata, (const uint8_t*)gscale, (const float*)gscale2,
        (const uint8_t*)udata, (const uint8_t*)uscale, (const float*)uscale2,
        (const uint32_t*)idx, (const float*)x, (float*)y, in_dim, ff, k);
    return pd_launch_status();
#endif
}

// Token-batched expert down GEMV + weighted combine. One warp per (token,
// out row); the k picks walk in fixed ascending slot order inside the lane
// accumulator (each partial scaled by topk_w[slot]*scale2[e] before the
// warp reduce - linear, so one reduction covers all k, deterministic).
// accumulate=1 adds onto y (the shared-expert pass: topk_w=1, k=1, same
// kernel); accumulate=0 overwrites (the routed pass goes first).
__global__ void pd_nvf4_moe_down_acc_kernel(
    const uint8_t* __restrict__ data, const uint8_t* __restrict__ scale,
    const float* __restrict__ scale2, const uint32_t* __restrict__ idx,
    const float* __restrict__ topk_w, const float* __restrict__ xr,
    float* __restrict__ y, uint32_t ff, uint32_t embd, uint32_t k,
    uint32_t accumulate) {
#if PD_NV4_OK
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const uint32_t r = blockIdx.x * (blockDim.x >> 5) + warp;
    if (r >= embd) return;
    const uint32_t t = blockIdx.y;
    float acc = 0.0f;
    for (uint32_t j = 0; j < k; ++j) {
        const uint32_t slot = t * k + j;
        const uint32_t e = idx[slot];
        const float w = topk_w[slot] * scale2[e];
        const uint8_t* row = data + ((size_t)e * embd + r) * (ff >> 1);
        const uint8_t* srow = scale + ((size_t)e * embd + r) * (ff >> 4);
        const float* xrow = xr + (size_t)slot * ff;
        float part = 0.0f;
        // tail hoisted (same fix as the gemv/up walks - see pd_nvf4_gemv_kernel)
        const uint32_t full = ff & ~127u;
        #pragma unroll 4
        for (uint32_t k0 = 0; k0 < full; k0 += 128u)
            part += pd_nvf4_dot4(row, srow, xrow, k0 + lane * 4u);
        if (full < ff) {
            const uint32_t el = full + lane * 4u;
            if (el < ff) part += pd_nvf4_dot4(row, srow, xrow, el);
        }
        acc += w * part;
    }
    for (uint32_t s = 16; s > 0; s >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, s);
    if (lane == 0) {
        float* out = y + (size_t)t * embd + r;
        *out = (accumulate ? *out : 0.0f) + acc;
    }
#else
    (void)data; (void)scale; (void)scale2; (void)idx; (void)topk_w; (void)xr;
    (void)y; (void)ff; (void)embd; (void)k; (void)accumulate;
#endif
}

PD_EXPORT
int pd_nvf4_moe_down_acc(const void* data, const void* scale,
                         const void* scale2, const void* idx,
                         const void* topk_w, const void* xr, void* y,
                         uint32_t ff, uint32_t embd, uint32_t k,
                         uint32_t batch, uint32_t accumulate, void* stream) {
#ifndef PD_BS_HOST
    (void)data; (void)scale; (void)scale2; (void)idx; (void)topk_w; (void)xr;
    (void)y; (void)ff; (void)embd; (void)k; (void)batch; (void)accumulate;
    (void)stream;
    return cudaErrorNotSupported;
#else
    if (embd == 0 || batch == 0 || k == 0) return 0;
    if ((ff & 31u) != 0) return cudaErrorInvalidValue;
    const uint32_t rows_per_cta = 8u;
    dim3 grid((embd + rows_per_cta - 1u) / rows_per_cta, batch);
    pd_nvf4_moe_down_acc_kernel<<<grid, rows_per_cta * 32u, 0,
                                  (cudaStream_t)stream>>>(
        (const uint8_t*)data, (const uint8_t*)scale, (const float*)scale2,
        (const uint32_t*)idx, (const float*)topk_w, (const float*)xr,
        (float*)y, ff, embd, k, accumulate);
    return pd_launch_status();
#endif
}

