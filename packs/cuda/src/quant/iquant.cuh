// quant/iquant.cuh - the ggml i-quant family (IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS,
// IQ2_S, IQ3_XXS, IQ3_S) and IQ4_NL on the k-quant streams.
//
// These are the 1-4 bpw formats of the Unsloth UD files: codebook-indexed
// 8-weight groups (a grid entry per group, a sign byte per group, a scale per
// 16 or 32 weights) rather than packed nibbles. They ride the SAME repacked
// stream pair the k-quant family uses - a data stream of 16-byte-aligned
// per-super-block payloads and a 24-byte scale record - so the token-batched
// MoE pair (moe/kquant.cuh) serves them through pd_kq_win_unpack with no
// kernel of its own. What a window unpack yields is what the dp4a lane
// wants: 16 packed s8 weights and an f32 scale. IQ1's +-0.125 delta folds
// into the integers as 8*grid +- 1 with the scale divided by 8 (exact in
// f32: a power-of-two rescale), so no mu term is needed.
//
// Layouts are ggml's (ggml-common.h); the dequant is a line-for-line port of
// ggml-quants.c's dequantize_row_* so the reference gate in
// tests/gpu_kquant_parity.rs is a bit-identity check, not a tolerance.
//
// Repacked data per 256-weight super-block (offsets inside the payload):
//   IQ2_XXS  64: qs u16[32]                                   rec: d
//   IQ2_XS   64: qs u16[32]                                   rec: d, scales[8] @2
//   IQ2_S    80: qs idx[32] @0, qh[8] @32, signs[32] @40, pad rec: d, scales[8] @2
//            (raw: d, qs[64] = 32 idx + 32 signs, qh[8], scales[8])
//   IQ3_XXS  96: qs[64] @0, scales_and_signs[32] @64          rec: d
//   IQ3_S   112: qs[64] @0, qh[8] @64, signs[32] @72, pad     rec: d, scales[4] @2
//   IQ1_S    48: qs[32] @0, qh u16[8] @32                      rec: d
//   IQ1_M    48: qs[32] @0, qh[16] @32                         rec: d (from scales) @0, scales u16[4] @2
//   IQ4_NL  128: 8 x qs[16] (per 32-weight block)             rec: d[8] f16 @0
//
// Not served here: the dense GEMV / mma lanes (quant/kquant.cuh,
// kquant_w4a8.cuh) - the engine routes i-quant tensors to the token-batched
// MoE pair only and refuses them as dense weights.

#define PD_KQ_IQ2XXS 16u
#define PD_KQ_IQ2XS 17u
#define PD_KQ_IQ3XXS 18u
#define PD_KQ_IQ1S 19u
#define PD_KQ_IQ4NL_ID 20u
#define PD_KQ_IQ3S 21u
#define PD_KQ_IQ2S 22u
#define PD_KQ_IQ1M 29u

#define PD_IQ1S_DELTA 0.125f

__host__ __device__ __forceinline__ bool pd_kq_valid_iq(uint32_t dt) {
    return dt == PD_KQ_IQ2XXS || dt == PD_KQ_IQ2XS || dt == PD_KQ_IQ2S ||
           dt == PD_KQ_IQ3XXS || dt == PD_KQ_IQ3S || dt == PD_KQ_IQ1S ||
           dt == PD_KQ_IQ1M || dt == PD_KQ_IQ4NL_ID;
}

// raw (GGUF) bytes per 256-weight super-block
__host__ __device__ __forceinline__ uint32_t pd_iq_srcb(uint32_t dt) {
    switch (dt) {
        case PD_KQ_IQ2XXS: return 66u;
        case PD_KQ_IQ2XS: return 74u;
        case PD_KQ_IQ2S: return 82u;
        case PD_KQ_IQ3XXS: return 98u;
        case PD_KQ_IQ3S: return 110u;
        case PD_KQ_IQ1S: return 50u;
        case PD_KQ_IQ1M: return 56u;
        default: return 144u;  // IQ4_NL: 8 x 18
    }
}

// repacked scale-record bytes per super-block: what pd_iq_repack_super writes
// and the window unpack reads (d f16 [+ the sub-block scale bytes]), rounded
// to 4. The k-quant family keeps its fixed 24-byte record (PD_KQ_SCB); the
// slimmer i-quant records are what let a 1-2 bit expert set stay near its
// raw size in host-mapped memory (a 24-byte record is +48% on IQ1_S).
__host__ __device__ __forceinline__ uint32_t pd_iq_scb(uint32_t dt) {
    switch (dt) {
        case PD_KQ_IQ2XXS: return 4u;   // d
        case PD_KQ_IQ2XS: return 12u;   // d + scales[8]
        case PD_KQ_IQ2S: return 12u;    // d + scales[8]
        case PD_KQ_IQ3XXS: return 4u;   // d
        case PD_KQ_IQ3S: return 8u;     // d + scales[4]
        case PD_KQ_IQ1S: return 4u;     // d
        case PD_KQ_IQ1M: return 12u;    // d (folded) + scales[8]
        default: return 16u;            // IQ4_NL: 8 x f16 d
    }
}

// repacked data bytes per super-block (16-byte multiples)
__host__ __device__ __forceinline__ uint32_t pd_iq_datab(uint32_t dt) {
    switch (dt) {
        case PD_KQ_IQ2XXS: return 64u;
        case PD_KQ_IQ2XS: return 64u;
        case PD_KQ_IQ2S: return 80u;
        case PD_KQ_IQ3XXS: return 96u;
        case PD_KQ_IQ3S: return 112u;
        case PD_KQ_IQ1S: return 48u;
        case PD_KQ_IQ1M: return 48u;
        default: return 128u;
    }
}

__device__ __forceinline__ float pd_iq_f16(const uint8_t* p) {
    __half h;
    memcpy(&h, p, 2u);
    return __half2float(h);
}

__device__ __forceinline__ uint16_t pd_iq_u16(const uint8_t* p) {
    uint16_t v;
    memcpy(&v, p, 2u);
    return v;
}

__device__ __forceinline__ uint32_t pd_iq_u32(const uint8_t* p) {
    uint32_t v;
    memcpy(&v, p, 4u);
    return v;
}

// IQ1_M's d is spread over the top nibbles of its four scale words.
__device__ __forceinline__ float pd_iq1m_d(const uint8_t* scales) {
    const uint16_t s0 = pd_iq_u16(scales), s1 = pd_iq_u16(scales + 2u);
    const uint16_t s2 = pd_iq_u16(scales + 4u), s3 = pd_iq_u16(scales + 6u);
    const uint16_t u = (uint16_t)((s0 >> 12) | ((s1 >> 8) & 0x00f0) | ((s2 >> 4) & 0x0f00) | (s3 & 0xf000));
    __half h;
    memcpy(&h, &u, 2u);
    return __half2float(h);
}

// ---- repack: raw super-block -> (data payload, pd_iq_scb-byte scale record) --
__device__ __forceinline__ void pd_iq_repack_super(uint32_t dt, const uint8_t* __restrict__ s,
                                                   uint8_t* __restrict__ d,
                                                   uint8_t* __restrict__ rec) {
    for (uint32_t i = 0; i < pd_iq_scb(dt); ++i) rec[i] = 0;
    switch (dt) {
        case PD_KQ_IQ2XXS:
            for (uint32_t i = 0; i < 64u; ++i) d[i] = s[2u + i];
            rec[0] = s[0]; rec[1] = s[1];
            break;
        case PD_KQ_IQ2XS:
            for (uint32_t i = 0; i < 64u; ++i) d[i] = s[2u + i];
            rec[0] = s[0]; rec[1] = s[1];
            for (uint32_t i = 0; i < 8u; ++i) rec[2u + i] = s[66u + i];
            break;
        case PD_KQ_IQ2S:
            // ggml's qs[64] is 32 grid-index bytes THEN 32 sign bytes; qh follows
            for (uint32_t i = 0; i < 32u; ++i) d[i] = s[2u + i];          // qs (indices)
            for (uint32_t i = 0; i < 8u; ++i) d[32u + i] = s[66u + i];    // qh
            for (uint32_t i = 0; i < 32u; ++i) d[40u + i] = s[34u + i];   // signs
            for (uint32_t i = 72u; i < 80u; ++i) d[i] = 0;
            rec[0] = s[0]; rec[1] = s[1];
            for (uint32_t i = 0; i < 8u; ++i) rec[2u + i] = s[74u + i];
            break;
        case PD_KQ_IQ3XXS:
            for (uint32_t i = 0; i < 96u; ++i) d[i] = s[2u + i];
            rec[0] = s[0]; rec[1] = s[1];
            break;
        case PD_KQ_IQ3S:
            for (uint32_t i = 0; i < 64u; ++i) d[i] = s[2u + i];          // qs
            for (uint32_t i = 0; i < 8u; ++i) d[64u + i] = s[66u + i];    // qh
            for (uint32_t i = 0; i < 32u; ++i) d[72u + i] = s[74u + i];   // signs
            for (uint32_t i = 104u; i < 112u; ++i) d[i] = 0;
            rec[0] = s[0]; rec[1] = s[1];
            for (uint32_t i = 0; i < 4u; ++i) rec[2u + i] = s[106u + i];
            break;
        case PD_KQ_IQ1S:
            for (uint32_t i = 0; i < 48u; ++i) d[i] = s[2u + i];          // qs + qh
            rec[0] = s[0]; rec[1] = s[1];
            break;
        case PD_KQ_IQ1M: {
            for (uint32_t i = 0; i < 48u; ++i) d[i] = s[i];               // qs + qh
            const float dd = pd_iq1m_d(s + 48u);
            const __half h = __float2half(dd);
            memcpy(rec, &h, 2u);
            for (uint32_t i = 0; i < 8u; ++i) rec[2u + i] = s[48u + i];
            break;
        }
        default:  // IQ4_NL: 8 blocks of {f16 d, 16 qs}
            for (uint32_t j = 0; j < 8u; ++j) {
                rec[2u * j] = s[j * 18u];
                rec[2u * j + 1u] = s[j * 18u + 1u];
                for (uint32_t i = 0; i < 16u; ++i) d[j * 16u + i] = s[j * 18u + 2u + i];
            }
            break;
    }
}

// ---- full-tensor dequant of a raw super-block (ggml dequantize_row_* port) --
__device__ __forceinline__ void pd_iq_dequant_super(uint32_t dt, const uint8_t* __restrict__ s,
                                                    float* __restrict__ y) {
    switch (dt) {
        case PD_KQ_IQ2XXS: {
            const float d = pd_iq_f16(s);
            const uint8_t* qs = s + 2u;
            for (uint32_t ib32 = 0; ib32 < 8u; ++ib32) {
                const uint32_t a0 = pd_iq_u32(qs + 8u * ib32), a1 = pd_iq_u32(qs + 8u * ib32 + 4u);
                const float db = d * (0.5f + (float)(a1 >> 28)) * 0.25f;
                for (uint32_t l = 0; l < 4u; ++l) {
                    const unsigned long long grid = PD_IQ2XXS_GRID[(a0 >> (8u * l)) & 0xFFu];
                    const uint8_t signs = PD_KSIGNS_IQ2XS[(a1 >> (7u * l)) & 127u];
                    for (uint32_t j = 0; j < 8u; ++j)
                        y[ib32 * 32u + l * 8u + j] =
                            db * (float)((grid >> (8u * j)) & 0xFFu) * ((signs >> j) & 1u ? -1.f : 1.f);
                }
            }
            break;
        }
        case PD_KQ_IQ2XS: {
            const float d = pd_iq_f16(s);
            const uint8_t* qs = s + 2u;
            const uint8_t* scales = s + 66u;
            for (uint32_t ib32 = 0; ib32 < 8u; ++ib32) {
                const float db0 = d * (0.5f + (float)(scales[ib32] & 0xFu)) * 0.25f;
                const float db1 = d * (0.5f + (float)(scales[ib32] >> 4u)) * 0.25f;
                for (uint32_t l = 0; l < 4u; ++l) {
                    const uint16_t q = pd_iq_u16(qs + 8u * ib32 + 2u * l);
                    const unsigned long long grid = PD_IQ2XS_GRID[q & 511u];
                    const uint8_t signs = PD_KSIGNS_IQ2XS[q >> 9u];
                    const float dl = l < 2u ? db0 : db1;
                    for (uint32_t j = 0; j < 8u; ++j)
                        y[ib32 * 32u + l * 8u + j] =
                            dl * (float)((grid >> (8u * j)) & 0xFFu) * ((signs >> j) & 1u ? -1.f : 1.f);
                }
            }
            break;
        }
        case PD_KQ_IQ2S: {
            const float d = pd_iq_f16(s);
            const uint8_t* qs = s + 2u;
            const uint8_t* signs = qs + 32u;
            const uint8_t* qh = s + 66u;
            const uint8_t* scales = s + 74u;
            for (uint32_t ib32 = 0; ib32 < 8u; ++ib32) {
                const float db0 = d * (0.5f + (float)(scales[ib32] & 0xFu)) * 0.25f;
                const float db1 = d * (0.5f + (float)(scales[ib32] >> 4u)) * 0.25f;
                for (uint32_t l = 0; l < 4u; ++l) {
                    const uint32_t idx = qs[4u * ib32 + l] | (((uint32_t)qh[ib32] << (8u - 2u * l)) & 0x300u);
                    const unsigned long long grid = PD_IQ2S_GRID[idx];
                    const uint8_t sg = signs[4u * ib32 + l];
                    const float dl = l < 2u ? db0 : db1;
                    for (uint32_t j = 0; j < 8u; ++j)
                        y[ib32 * 32u + l * 8u + j] =
                            dl * (float)((grid >> (8u * j)) & 0xFFu) * ((sg >> j) & 1u ? -1.f : 1.f);
                }
            }
            break;
        }
        case PD_KQ_IQ3XXS: {
            const float d = pd_iq_f16(s);
            const uint8_t* qs = s + 2u;
            const uint8_t* sas = qs + 64u;
            for (uint32_t ib32 = 0; ib32 < 8u; ++ib32) {
                const uint32_t aux = pd_iq_u32(sas + 4u * ib32);
                const float db = d * (0.5f + (float)(aux >> 28)) * 0.5f;
                for (uint32_t l = 0; l < 4u; ++l) {
                    const uint8_t signs = PD_KSIGNS_IQ2XS[(aux >> (7u * l)) & 127u];
                    const uint32_t g1 = PD_IQ3XXS_GRID[qs[8u * ib32 + 2u * l]];
                    const uint32_t g2 = PD_IQ3XXS_GRID[qs[8u * ib32 + 2u * l + 1u]];
                    for (uint32_t j = 0; j < 4u; ++j) {
                        y[ib32 * 32u + l * 8u + j] =
                            db * (float)((g1 >> (8u * j)) & 0xFFu) * ((signs >> j) & 1u ? -1.f : 1.f);
                        y[ib32 * 32u + l * 8u + 4u + j] =
                            db * (float)((g2 >> (8u * j)) & 0xFFu) * ((signs >> (j + 4u)) & 1u ? -1.f : 1.f);
                    }
                }
            }
            break;
        }
        case PD_KQ_IQ3S: {
            const float d = pd_iq_f16(s);
            const uint8_t* qs = s + 2u;
            const uint8_t* qh = s + 66u;
            const uint8_t* signs = s + 74u;
            const uint8_t* scales = s + 106u;
            for (uint32_t ib32 = 0; ib32 < 8u; ++ib32) {
                const uint8_t sc = scales[ib32 >> 1u];
                const float db = d * (1.f + 2.f * (float)((ib32 & 1u) ? (sc >> 4u) : (sc & 0xFu)));
                for (uint32_t l = 0; l < 4u; ++l) {
                    const uint32_t i1 = qs[8u * ib32 + 2u * l] | (((uint32_t)qh[ib32] << (8u - 2u * l)) & 256u);
                    const uint32_t i2 = qs[8u * ib32 + 2u * l + 1u] | (((uint32_t)qh[ib32] << (7u - 2u * l)) & 256u);
                    const uint32_t g1 = PD_IQ3S_GRID[i1], g2 = PD_IQ3S_GRID[i2];
                    const uint8_t sg = signs[4u * ib32 + l];
                    for (uint32_t j = 0; j < 4u; ++j) {
                        y[ib32 * 32u + l * 8u + j] =
                            db * (float)((g1 >> (8u * j)) & 0xFFu) * ((sg >> j) & 1u ? -1.f : 1.f);
                        y[ib32 * 32u + l * 8u + 4u + j] =
                            db * (float)((g2 >> (8u * j)) & 0xFFu) * ((sg >> (j + 4u)) & 1u ? -1.f : 1.f);
                    }
                }
            }
            break;
        }
        case PD_KQ_IQ1S: {
            const float d = pd_iq_f16(s);
            const uint8_t* qs = s + 2u;
            const uint8_t* qh = s + 34u;
            for (uint32_t ib = 0; ib < 8u; ++ib) {
                const uint16_t h = pd_iq_u16(qh + 2u * ib);
                const float dl = d * (float)(2u * ((h >> 12) & 7u) + 1u);
                const float delta = (h & 0x8000u) ? -PD_IQ1S_DELTA : PD_IQ1S_DELTA;
                for (uint32_t l = 0; l < 4u; ++l) {
                    const unsigned long long grid = PD_IQ1S_GRID[qs[4u * ib + l] | (((h >> (3u * l)) & 7u) << 8u)];
                    for (uint32_t j = 0; j < 8u; ++j)
                        y[ib * 32u + l * 8u + j] = dl * ((float)(int8_t)((grid >> (8u * j)) & 0xFFu) + delta);
                }
            }
            break;
        }
        case PD_KQ_IQ1M: {
            const uint8_t* qs = s;
            const uint8_t* qh = s + 32u;
            const uint8_t* scb = s + 48u;
            const float d = pd_iq1m_d(scb);
            for (uint32_t ib = 0; ib < 8u; ++ib) {
                const uint16_t sc = pd_iq_u16(scb + 2u * (ib >> 1u));
                const uint32_t sh = 6u * (ib & 1u);
                const float dl1 = d * (float)(2u * ((sc >> sh) & 7u) + 1u);
                const float dl2 = d * (float)(2u * ((sc >> (sh + 3u)) & 7u) + 1u);
                const uint8_t h0 = qh[2u * ib], h1 = qh[2u * ib + 1u];
                const uint32_t idx[4] = {
                    qs[4u * ib] | (((uint32_t)h0 << 8u) & 0x700u),
                    qs[4u * ib + 1u] | (((uint32_t)h0 << 4u) & 0x700u),
                    qs[4u * ib + 2u] | (((uint32_t)h1 << 8u) & 0x700u),
                    qs[4u * ib + 3u] | (((uint32_t)h1 << 4u) & 0x700u)};
                const float delta[4] = {(h0 & 0x08u) ? -PD_IQ1S_DELTA : PD_IQ1S_DELTA,
                                        (h0 & 0x80u) ? -PD_IQ1S_DELTA : PD_IQ1S_DELTA,
                                        (h1 & 0x08u) ? -PD_IQ1S_DELTA : PD_IQ1S_DELTA,
                                        (h1 & 0x80u) ? -PD_IQ1S_DELTA : PD_IQ1S_DELTA};
                for (uint32_t l = 0; l < 4u; ++l) {
                    const unsigned long long grid = PD_IQ1S_GRID[idx[l]];
                    const float dl = l < 2u ? dl1 : dl2;
                    for (uint32_t j = 0; j < 8u; ++j)
                        y[ib * 32u + l * 8u + j] = dl * ((float)(int8_t)((grid >> (8u * j)) & 0xFFu) + delta[l]);
                }
            }
            break;
        }
        default: {  // IQ4_NL
            for (uint32_t j = 0; j < 8u; ++j) {
                const uint8_t* blk = s + j * 18u;
                const float d = pd_iq_f16(blk);
                for (uint32_t l = 0; l < 16u; ++l) {
                    y[j * 32u + l] = d * (float)PD_KVALUES_IQ4NL[blk[2u + l] & 0xFu];
                    y[j * 32u + 16u + l] = d * (float)PD_KVALUES_IQ4NL[blk[2u + l] >> 4u];
                }
            }
            break;
        }
    }
}

// ---- window unpack over the REPACKED streams: 16 weights -> 4 packed s8 ----
// words + f32 scale (the token-batched dp4a lane's contract; g = 0 always).
// Window w (0..15) of a super-block covers 32-weight block ib = w >> 1, its
// 8-weight groups l = 2*(w & 1) and 2*(w & 1) + 1.
__device__ __forceinline__ int pd_iq_pack_signed(unsigned long long grid, uint32_t signs, uint32_t half) {
    // bytes 4*half..4*half+3 of the grid, negated where the sign bit is set
    int out = 0;
    #pragma unroll
    for (uint32_t j = 0; j < 4u; ++j) {
        const uint32_t jj = 4u * half + j;
        int v = (int)((grid >> (8u * jj)) & 0xFFu);
        if ((signs >> jj) & 1u) v = -v;
        out |= (v & 0xFF) << (8u * j);
    }
    return out;
}

__device__ __forceinline__ int pd_iq_pack_signed32(uint32_t grid, uint32_t signs, uint32_t half) {
    int out = 0;
    #pragma unroll
    for (uint32_t j = 0; j < 4u; ++j) {
        int v = (int)((grid >> (8u * j)) & 0xFFu);
        if ((signs >> (4u * half + j)) & 1u) v = -v;
        out |= (v & 0xFF) << (8u * j);
    }
    return out;
}

// IQ1: 8*grid +- 1 per byte (grid in {-1, 0, 1}), scale carries the /8
__device__ __forceinline__ int pd_iq1_pack(unsigned long long grid, bool neg, uint32_t half) {
    int out = 0;
    #pragma unroll
    for (uint32_t j = 0; j < 4u; ++j) {
        const int g = (int)(int8_t)((grid >> (8u * (4u * half + j))) & 0xFFu);
        const int v = 8 * g + (neg ? -1 : 1);
        out |= (v & 0xFF) << (8u * j);
    }
    return out;
}

__device__ __forceinline__ void pd_iq_win_unpack(uint32_t dt, const uint8_t* __restrict__ sb,
                                                 const uint8_t* __restrict__ rec, uint32_t w,
                                                 int wq[4], float* f) {
    const uint32_t ib = w >> 1u, l0 = 2u * (w & 1u);
    switch (dt) {
        case PD_KQ_IQ2XXS: {
            const uint32_t a0 = pd_iq_u32(sb + 8u * ib), a1 = pd_iq_u32(sb + 8u * ib + 4u);
            *f = pd_iq_f16(rec) * (0.5f + (float)(a1 >> 28)) * 0.25f;
            #pragma unroll
            for (uint32_t i = 0; i < 2u; ++i) {
                const uint32_t l = l0 + i;
                const unsigned long long grid = PD_IQ2XXS_GRID[(a0 >> (8u * l)) & 0xFFu];
                const uint32_t signs = PD_KSIGNS_IQ2XS[(a1 >> (7u * l)) & 127u];
                wq[2u * i] = pd_iq_pack_signed(grid, signs, 0u);
                wq[2u * i + 1u] = pd_iq_pack_signed(grid, signs, 1u);
            }
            break;
        }
        case PD_KQ_IQ2XS: {
            const uint8_t sc = rec[2u + ib];
            *f = pd_iq_f16(rec) * (0.5f + (float)((w & 1u) ? (sc >> 4u) : (sc & 0xFu))) * 0.25f;
            #pragma unroll
            for (uint32_t i = 0; i < 2u; ++i) {
                const uint16_t q = pd_iq_u16(sb + 8u * ib + 2u * (l0 + i));
                const unsigned long long grid = PD_IQ2XS_GRID[q & 511u];
                const uint32_t signs = PD_KSIGNS_IQ2XS[q >> 9u];
                wq[2u * i] = pd_iq_pack_signed(grid, signs, 0u);
                wq[2u * i + 1u] = pd_iq_pack_signed(grid, signs, 1u);
            }
            break;
        }
        case PD_KQ_IQ2S: {
            const uint8_t sc = rec[2u + ib];
            *f = pd_iq_f16(rec) * (0.5f + (float)((w & 1u) ? (sc >> 4u) : (sc & 0xFu))) * 0.25f;
            const uint8_t qh = sb[32u + ib];
            #pragma unroll
            for (uint32_t i = 0; i < 2u; ++i) {
                const uint32_t l = l0 + i;
                const uint32_t idx = sb[4u * ib + l] | (((uint32_t)qh << (8u - 2u * l)) & 0x300u);
                const unsigned long long grid = PD_IQ2S_GRID[idx];
                const uint32_t signs = sb[40u + 4u * ib + l];
                wq[2u * i] = pd_iq_pack_signed(grid, signs, 0u);
                wq[2u * i + 1u] = pd_iq_pack_signed(grid, signs, 1u);
            }
            break;
        }
        case PD_KQ_IQ3XXS: {
            const uint32_t aux = pd_iq_u32(sb + 64u + 4u * ib);
            *f = pd_iq_f16(rec) * (0.5f + (float)(aux >> 28)) * 0.5f;
            #pragma unroll
            for (uint32_t i = 0; i < 2u; ++i) {
                const uint32_t l = l0 + i;
                const uint32_t signs = PD_KSIGNS_IQ2XS[(aux >> (7u * l)) & 127u];
                wq[2u * i] = pd_iq_pack_signed32(PD_IQ3XXS_GRID[sb[8u * ib + 2u * l]], signs, 0u);
                wq[2u * i + 1u] = pd_iq_pack_signed32(PD_IQ3XXS_GRID[sb[8u * ib + 2u * l + 1u]], signs, 1u);
            }
            break;
        }
        case PD_KQ_IQ3S: {
            const uint8_t sc = rec[2u + (ib >> 1u)];
            *f = pd_iq_f16(rec) * (1.f + 2.f * (float)((ib & 1u) ? (sc >> 4u) : (sc & 0xFu)));
            const uint8_t qh = sb[64u + ib];
            #pragma unroll
            for (uint32_t i = 0; i < 2u; ++i) {
                const uint32_t l = l0 + i;
                const uint32_t i1 = sb[8u * ib + 2u * l] | (((uint32_t)qh << (8u - 2u * l)) & 256u);
                const uint32_t i2 = sb[8u * ib + 2u * l + 1u] | (((uint32_t)qh << (7u - 2u * l)) & 256u);
                const uint32_t signs = sb[72u + 4u * ib + l];
                wq[2u * i] = pd_iq_pack_signed32(PD_IQ3S_GRID[i1], signs, 0u);
                wq[2u * i + 1u] = pd_iq_pack_signed32(PD_IQ3S_GRID[i2], signs, 1u);
            }
            break;
        }
        case PD_KQ_IQ1S: {
            const uint16_t h = pd_iq_u16(sb + 32u + 2u * ib);
            *f = pd_iq_f16(rec) * (float)(2u * ((h >> 12) & 7u) + 1u) * 0.125f;
            const bool neg = (h & 0x8000u) != 0u;
            #pragma unroll
            for (uint32_t i = 0; i < 2u; ++i) {
                const uint32_t l = l0 + i;
                const unsigned long long grid = PD_IQ1S_GRID[sb[4u * ib + l] | (((h >> (3u * l)) & 7u) << 8u)];
                wq[2u * i] = pd_iq1_pack(grid, neg, 0u);
                wq[2u * i + 1u] = pd_iq1_pack(grid, neg, 1u);
            }
            break;
        }
        case PD_KQ_IQ1M: {
            const uint16_t sc = pd_iq_u16(rec + 2u + 2u * (ib >> 1u));
            const uint32_t sh = 6u * (ib & 1u) + 3u * (w & 1u);
            *f = pd_iq_f16(rec) * (float)(2u * ((sc >> sh) & 7u) + 1u) * 0.125f;
            const uint8_t hq = sb[32u + 2u * ib + (w & 1u)];
            #pragma unroll
            for (uint32_t i = 0; i < 2u; ++i) {
                const uint32_t l = l0 + i;
                const uint32_t idx = sb[4u * ib + l] | (((uint32_t)hq << (i ? 4u : 8u)) & 0x700u);
                const bool neg = (hq & (i ? 0x80u : 0x08u)) != 0u;
                const unsigned long long grid = PD_IQ1S_GRID[idx];
                wq[2u * i] = pd_iq1_pack(grid, neg, 0u);
                wq[2u * i + 1u] = pd_iq1_pack(grid, neg, 1u);
            }
            break;
        }
        default: {  // IQ4_NL: block ib's lo (w even) / hi (w odd) nibbles
            *f = pd_iq_f16(rec + 2u * ib);
            const uint4 qa = __ldcs((const uint4*)(sb + ib * 16u));
            const uint32_t qw[4] = {qa.x, qa.y, qa.z, qa.w};
            const bool hi = (w & 1u) != 0u;
            #pragma unroll
            for (uint32_t v = 0; v < 4u; ++v) {
                const uint32_t nib = (hi ? qw[v] >> 4u : qw[v]) & 0x0F0F0F0Fu;
                int out = 0;
                #pragma unroll
                for (uint32_t j = 0; j < 4u; ++j)
                    out |= ((int)PD_KVALUES_IQ4NL[(nib >> (8u * j)) & 0xFu] & 0xFF) << (8u * j);
                wq[v] = out;
            }
            break;
        }
    }
}
