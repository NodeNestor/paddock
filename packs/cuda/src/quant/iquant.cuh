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
// the two low-bit K-QUANTS ride the same lanes: 16-weight sub-block scales,
// a 24-byte record, no codebook
#define PD_KQ_Q2K_ID 10u
#define PD_KQ_Q3K_ID 11u

#define PD_IQ1S_DELTA 0.125f

__host__ __device__ __forceinline__ bool pd_kq_valid_iq(uint32_t dt) {
    return dt == PD_KQ_IQ2XXS || dt == PD_KQ_IQ2XS || dt == PD_KQ_IQ2S ||
           dt == PD_KQ_IQ3XXS || dt == PD_KQ_IQ3S || dt == PD_KQ_IQ1S ||
           dt == PD_KQ_IQ1M || dt == PD_KQ_IQ4NL_ID ||
           dt == PD_KQ_Q2K_ID || dt == PD_KQ_Q3K_ID;
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
        case PD_KQ_Q2K_ID: return 84u;   // scales[16] qs[64] d dmin
        case PD_KQ_Q3K_ID: return 110u;  // hmask[32] qs[64] scales[12] d
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
        case PD_KQ_Q2K_ID: return 24u;  // d, dmin + scales[16] (4|4-bit sc|m per 16)
        case PD_KQ_Q3K_ID: return 24u;  // d + 16 unpacked int8 scales (6-bit, -32 applied)
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
        case PD_KQ_Q2K_ID: return 64u;   // qs
        case PD_KQ_Q3K_ID: return 96u;   // qs[64] + hmask[32]
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

// ksigns_iq2xs, computed: the table is the 7-bit index with its parity in
// bit 7 (so every 8-sign word has even parity). Two instructions beat a
// dependent table load in every lane that used it.
__device__ __forceinline__ uint32_t pd_iq_ksign(uint32_t idx7) {
    return idx7 | ((__popc(idx7) & 1u) << 7u);
}

__device__ __forceinline__ uint32_t pd_iq_u32(const uint8_t* p) {
    uint32_t v;
    memcpy(&v, p, 4u);
    return v;
}

// The REPACKED streams are 16-byte aligned per super-block and every offset
// the window unpack reads is a multiple of the width, so these are single
// aligned loads (the byte-assembling twins above stay for the raw GGUF
// super-blocks, whose f16 header leaves the payload 2-aligned).
__device__ __forceinline__ uint16_t pd_iq_u16a(const uint8_t* p) {
    return *reinterpret_cast<const uint16_t*>(p);
}
__device__ __forceinline__ uint32_t pd_iq_u32a(const uint8_t* p) {
    return *reinterpret_cast<const uint32_t*>(p);
}
__device__ __forceinline__ float pd_iq_f16a(const uint8_t* p) {
    __half h;
    const uint16_t u = pd_iq_u16a(p);
    memcpy(&h, &u, 2u);
    return __half2float(h);
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

// Q3_K's 16 six-bit scales out of the packed 12 bytes (ggml dequantize_row_q3_K's
// kmask1/kmask2 shuffle), as signed int8 with the -32 applied.
__device__ __forceinline__ void pd_q3k_unpack_scales(const uint8_t* __restrict__ sc,
                                                              int8_t out[16]) {
    uint32_t aux[4];
    aux[0] = pd_iq_u32(sc);
    aux[1] = pd_iq_u32(sc + 4u);
    const uint32_t tmp = pd_iq_u32(sc + 8u);
    const uint32_t kmask1 = 0x03030303u, kmask2 = 0x0f0f0f0fu;
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    for (uint32_t i = 0; i < 16u; ++i)
        out[i] = (int8_t)((int)((aux[i >> 2] >> (8u * (i & 3u))) & 0xFFu) - 32);
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
        case PD_KQ_Q2K_ID:
            for (uint32_t i = 0; i < 64u; ++i) d[i] = s[16u + i];        // qs
            rec[0] = s[80]; rec[1] = s[81];                              // d
            rec[2] = s[82]; rec[3] = s[83];                              // dmin
            for (uint32_t i = 0; i < 16u; ++i) rec[4u + i] = s[i];        // sc|m per 16
            break;
        case PD_KQ_Q3K_ID: {
            for (uint32_t i = 0; i < 64u; ++i) d[i] = s[32u + i];        // qs (low 2 bits)
            for (uint32_t i = 0; i < 32u; ++i) d[64u + i] = s[i];        // hmask (high bit)
            rec[0] = s[108]; rec[1] = s[109];                            // d
            int8_t scs[16];
            pd_q3k_unpack_scales(s + 96u, scs);
            for (uint32_t i = 0; i < 16u; ++i) rec[2u + i] = (uint8_t)scs[i];
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
        case PD_KQ_Q2K_ID: {
            // ggml dequantize_row_q2_K: y = d*sc*q - dmin*m, sub-block of 16
            const float d = pd_iq_f16(s + 80u), dmin = pd_iq_f16(s + 82u);
            const uint8_t* q = s + 16u;
            uint32_t is = 0;
            for (uint32_t n = 0; n < 256u; n += 128u) {
                for (uint32_t j = 0; j < 4u; ++j) {
                    for (uint32_t lb = 0; lb < 2u; ++lb) {
                        const uint8_t sc = s[is++];
                        const float dl = d * (float)(sc & 0xFu), ml = dmin * (float)(sc >> 4u);
                        for (uint32_t l = 0; l < 16u; ++l)
                            y[n + 32u * j + 16u * lb + l] =
                                dl * (float)((q[n / 4u + 16u * lb + l] >> (2u * j)) & 3u) - ml;
                    }
                }
            }
            break;
        }
        case PD_KQ_Q3K_ID: {
            // ggml dequantize_row_q3_K: q = low2 - (hbit ? 0 : 4), y = d*scale*q
            const float d = pd_iq_f16(s + 108u);
            const uint8_t* hm = s;
            const uint8_t* q = s + 32u;
            int8_t scs[16];
            pd_q3k_unpack_scales(s + 96u, scs);
            uint32_t is = 0, mbit = 0;
            for (uint32_t n = 0; n < 256u; n += 128u) {
                for (uint32_t j = 0; j < 4u; ++j, ++mbit) {
                    for (uint32_t lb = 0; lb < 2u; ++lb) {
                        const float dl = d * (float)scs[is++];
                        for (uint32_t l = 0; l < 16u; ++l) {
                            const uint32_t i = 16u * lb + l;
                            const int qv = (int)((q[n / 4u + i] >> (2u * j)) & 3u) -
                                           (((hm[i] >> mbit) & 1u) ? 0 : 4);
                            y[n + 32u * j + i] = dl * (float)qv;
                        }
                    }
                }
            }
            break;
        }
        case PD_KQ_IQ2XXS: {
            const float d = pd_iq_f16(s);
            const uint8_t* qs = s + 2u;
            for (uint32_t ib32 = 0; ib32 < 8u; ++ib32) {
                const uint32_t a0 = pd_iq_u32(qs + 8u * ib32), a1 = pd_iq_u32(qs + 8u * ib32 + 4u);
                const float db = d * (0.5f + (float)(a1 >> 28)) * 0.25f;
                for (uint32_t l = 0; l < 4u; ++l) {
                    const unsigned long long grid = PD_IQ2XXS_GRID[(a0 >> (8u * l)) & 0xFFu];
                    const uint8_t signs = (uint8_t)pd_iq_ksign((a1 >> (7u * l)) & 127u);
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
                    const uint8_t signs = (uint8_t)pd_iq_ksign(q >> 9u);
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
                    const uint8_t signs = (uint8_t)pd_iq_ksign((aux >> (7u * l)) & 127u);
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
// Sign application is SIMD: the 4 sign bits of a half spread to one byte
// each (a multiply by 1 + 2^7 + 2^14 + 2^21 lands bit i at 8i with no
// carries), widen to 0x00/0xFF, and (v ^ m) - m negates per byte.
__device__ __forceinline__ uint32_t pd_iq_sign_mask4(uint32_t bits4) {
    return ((bits4 * 0x00204081u) & 0x01010101u) * 0xFFu;
}
__device__ __forceinline__ int pd_iq_apply_signs(uint32_t word, uint32_t bits4) {
    const uint32_t m = pd_iq_sign_mask4(bits4);
    return (int)__vsub4(word ^ m, m);
}
__device__ __forceinline__ int pd_iq_pack_signed(uint32_t word, uint32_t signs, uint32_t half) {
    // grid bytes 4*half..4*half+3 (`word`), negated where the sign bit is set
    return pd_iq_apply_signs(word, (signs >> (4u * half)) & 0xFu);
}

__device__ __forceinline__ int pd_iq_pack_signed32(uint32_t grid, uint32_t signs, uint32_t half) {
    return pd_iq_apply_signs(grid, (signs >> (4u * half)) & 0xFu);
}


// IQ1: 8*grid +- 1 per byte (grid in {-1, 0, 1}), scale carries the /8:
// a per-byte shift by 3 (mask keeps the bytes apart), then a SIMD +-1.
__device__ __forceinline__ int pd_iq1_pack(uint32_t word, bool neg) {
    const uint32_t g8 = (word << 3u) & 0xF8F8F8F8u;
    return (int)__vadd4(g8, neg ? 0xFFFFFFFFu : 0x01010101u);
}

// The codebooks one format reads. The window unpack takes them through this
// struct so a kernel can hand it a SHARED-MEMORY copy (quant/iquant_dense.cuh
// does: 2-16 KB per block, one broadcast-free lookup per window instead of a
// dependent L1 load); `pd_iq_tabs_global` is the default set.
struct PdIqTabs {
    const unsigned long long* g64;  // IQ2_XXS / IQ2_XS / IQ2_S / IQ1_S grids
    const uint32_t* g32;            // IQ3_XXS / IQ3_S grids
};
__device__ __forceinline__ PdIqTabs pd_iq_tabs_global(uint32_t dt) {
    PdIqTabs t;
    t.g64 = nullptr; t.g32 = nullptr;
    switch (dt) {
        case PD_KQ_IQ2XXS: t.g64 = PD_IQ2XXS_GRID; break;
        case PD_KQ_IQ2XS: t.g64 = PD_IQ2XS_GRID; break;
        case PD_KQ_IQ2S: t.g64 = PD_IQ2S_GRID; break;
        case PD_KQ_IQ1S: t.g64 = PD_IQ1S_GRID; break;
        case PD_KQ_IQ3XXS: t.g32 = PD_IQ3XXS_GRID; break;
        case PD_KQ_IQ3S: t.g32 = PD_IQ3S_GRID; break;
        default: break;
    }
    return t;
}
// bytes of the format's grid (0 = no codebook: IQ1_M's grid is IQ1_S's, handled below)
__host__ __device__ __forceinline__ uint32_t pd_iq_grid_bytes(uint32_t dt) {
    switch (dt) {
        case PD_KQ_IQ2XXS: return 256u * 8u;
        case PD_KQ_IQ2XS: return 512u * 8u;
        case PD_KQ_IQ2S: return 1024u * 8u;
        case PD_KQ_IQ1S: case PD_KQ_IQ1M: return 2048u * 8u;
        case PD_KQ_IQ3XXS: return 256u * 4u;
        case PD_KQ_IQ3S: return 512u * 4u;
        default: return 0u;
    }
}
// smem bytes a block needs for its table copy. (A nibble-coded copy - every
// grid draws its bytes from an alphabet of <= 8 values, so an entry fits a
// 32-bit word and a lookup is a one-bank load + two byte-permutes - was
// measured 10% SLOWER on IQ2_XXS: the lanes are issue-bound, not
// bank-conflict-bound, and the permutes cost more than the wider gather.)
__host__ __device__ __forceinline__ uint32_t pd_iq_tabs_bytes(uint32_t dt) {
    return pd_iq_grid_bytes(dt);
}
// the 8-byte grid entry `i` as two packed words
__device__ __forceinline__ void pd_iq_grid8(uint32_t dt, const PdIqTabs& tabs, uint32_t i,
                                            uint32_t* lo, uint32_t* hi) {
    (void)dt;
    const unsigned long long g = tabs.g64[i];
    *lo = (uint32_t)g;
    *hi = (uint32_t)(g >> 32u);
}
// the 4-byte grid entry `i`
__device__ __forceinline__ uint32_t pd_iq_grid4(uint32_t dt, const PdIqTabs& tabs, uint32_t i) {
    (void)dt;
    return tabs.g32[i];
}
// Copy the format's tables into `dst` (block-cooperative; caller syncs) and
// return the set pointing at the copy. Falls back to the globals when the
// format has no codebook.
__device__ __forceinline__ PdIqTabs pd_iq_tabs_stage(uint32_t dt, uint8_t* dst) {
    PdIqTabs g = pd_iq_tabs_global(dt);
    if (dt == PD_KQ_IQ1M) g.g64 = PD_IQ1S_GRID;
    const uint32_t gb = pd_iq_grid_bytes(dt);
    if (gb == 0u) return g;
    const uint32_t* src = g.g64 ? reinterpret_cast<const uint32_t*>(g.g64)
                                : reinterpret_cast<const uint32_t*>(g.g32);
    uint32_t* d32 = reinterpret_cast<uint32_t*>(dst);
    for (uint32_t i = threadIdx.x; i < (gb >> 2); i += blockDim.x) d32[i] = src[i];
    PdIqTabs t;
    t.g64 = g.g64 ? reinterpret_cast<const unsigned long long*>(dst) : nullptr;
    t.g32 = g.g32 ? reinterpret_cast<const uint32_t*>(dst) : nullptr;
    return t;
}

__device__ __forceinline__ void pd_iq_win_unpack_t(uint32_t dt, const uint8_t* __restrict__ sb,
                                                   const uint8_t* __restrict__ rec, uint32_t w,
                                                   int wq[4], float* f, float* g,
                                                   const PdIqTabs& tabs) {
    const uint32_t ib = w >> 1u, l0 = 2u * (w & 1u);
    *g = 0.0f;
    switch (dt) {
        case PD_KQ_Q2K_ID: {
            // window w = sub-block (n = w>>3, j = (w>>1)&3, lb = w&1) of 16 weights
            const uint32_t n = w >> 3u, j = (w >> 1u) & 3u, lb = w & 1u;
            const uint8_t sc = rec[4u + w];
            *f = pd_iq_f16a(rec) * (float)(sc & 0xFu);
            *g = -pd_iq_f16a(rec + 2u) * (float)(sc >> 4u);
            const uint4 qa = __ldcs((const uint4*)(sb + 32u * n + 16u * lb));
            const uint32_t qw[4] = {qa.x, qa.y, qa.z, qa.w};
            #pragma unroll
            for (uint32_t k = 0; k < 4u; ++k) wq[k] = (int)((qw[k] >> (2u * j)) & 0x03030303u);
            break;
        }
        case PD_KQ_Q3K_ID: {
            const uint32_t n = w >> 3u, j = (w >> 1u) & 3u, lb = w & 1u;
            const uint32_t mbit = 4u * n + j;
            *f = pd_iq_f16a(rec) * (float)((const int8_t*)rec)[2u + w];
            const uint4 qa = __ldcs((const uint4*)(sb + 32u * n + 16u * lb));
            const uint4 ha = __ldcs((const uint4*)(sb + 64u + 16u * lb));
            const uint32_t qw[4] = {qa.x, qa.y, qa.z, qa.w};
            const uint32_t hw[4] = {ha.x, ha.y, ha.z, ha.w};
            #pragma unroll
            for (uint32_t k = 0; k < 4u; ++k) {
                // per byte: low2 | (hbit << 2), then -4 without cross-byte borrow
                const uint32_t lo = (qw[k] >> (2u * j)) & 0x03030303u;
                const uint32_t hb = ((hw[k] >> mbit) & 0x01010101u) << 2u;
                wq[k] = (int)__vsub4(lo | hb, 0x04040404u);
            }
            break;
        }
        case PD_KQ_IQ2XXS: {
            const uint32_t a0 = pd_iq_u32a(sb + 8u * ib), a1 = pd_iq_u32a(sb + 8u * ib + 4u);
            *f = pd_iq_f16a(rec) * (0.5f + (float)(a1 >> 28)) * 0.25f;
            #pragma unroll
            for (uint32_t i = 0; i < 2u; ++i) {
                const uint32_t l = l0 + i;
                uint32_t g0, g1;
                pd_iq_grid8(dt, tabs, (a0 >> (8u * l)) & 0xFFu, &g0, &g1);
                const uint32_t signs = pd_iq_ksign((a1 >> (7u * l)) & 127u);
                wq[2u * i] = pd_iq_pack_signed(g0, signs, 0u);
                wq[2u * i + 1u] = pd_iq_pack_signed(g1, signs, 1u);
            }
            break;
        }
        case PD_KQ_IQ2XS: {
            const uint8_t sc = rec[2u + ib];
            *f = pd_iq_f16a(rec) * (0.5f + (float)((w & 1u) ? (sc >> 4u) : (sc & 0xFu))) * 0.25f;
            #pragma unroll
            for (uint32_t i = 0; i < 2u; ++i) {
                const uint16_t q = pd_iq_u16a(sb + 8u * ib + 2u * (l0 + i));
                uint32_t g0, g1;
                pd_iq_grid8(dt, tabs, q & 511u, &g0, &g1);
                const uint32_t signs = pd_iq_ksign(q >> 9u);
                wq[2u * i] = pd_iq_pack_signed(g0, signs, 0u);
                wq[2u * i + 1u] = pd_iq_pack_signed(g1, signs, 1u);
            }
            break;
        }
        case PD_KQ_IQ2S: {
            const uint8_t sc = rec[2u + ib];
            *f = pd_iq_f16a(rec) * (0.5f + (float)((w & 1u) ? (sc >> 4u) : (sc & 0xFu))) * 0.25f;
            const uint8_t qh = sb[32u + ib];
            #pragma unroll
            for (uint32_t i = 0; i < 2u; ++i) {
                const uint32_t l = l0 + i;
                const uint32_t idx = sb[4u * ib + l] | (((uint32_t)qh << (8u - 2u * l)) & 0x300u);
                uint32_t g0, g1;
                pd_iq_grid8(dt, tabs, idx, &g0, &g1);
                const uint32_t signs = sb[40u + 4u * ib + l];
                wq[2u * i] = pd_iq_pack_signed(g0, signs, 0u);
                wq[2u * i + 1u] = pd_iq_pack_signed(g1, signs, 1u);
            }
            break;
        }
        case PD_KQ_IQ3XXS: {
            const uint32_t aux = pd_iq_u32a(sb + 64u + 4u * ib);
            *f = pd_iq_f16a(rec) * (0.5f + (float)(aux >> 28)) * 0.5f;
            #pragma unroll
            for (uint32_t i = 0; i < 2u; ++i) {
                const uint32_t l = l0 + i;
                const uint32_t signs = pd_iq_ksign((aux >> (7u * l)) & 127u);
                const uint32_t g0 = pd_iq_grid4(dt, tabs, sb[8u * ib + 2u * l]);
                const uint32_t g1 = pd_iq_grid4(dt, tabs, sb[8u * ib + 2u * l + 1u]);
                wq[2u * i] = pd_iq_pack_signed32(g0, signs, 0u);
                wq[2u * i + 1u] = pd_iq_pack_signed32(g1, signs, 1u);
            }
            break;
        }
        case PD_KQ_IQ3S: {
            const uint8_t sc = rec[2u + (ib >> 1u)];
            *f = pd_iq_f16a(rec) * (1.f + 2.f * (float)((ib & 1u) ? (sc >> 4u) : (sc & 0xFu)));
            const uint8_t qh = sb[64u + ib];
            #pragma unroll
            for (uint32_t i = 0; i < 2u; ++i) {
                const uint32_t l = l0 + i;
                const uint32_t i1 = sb[8u * ib + 2u * l] | (((uint32_t)qh << (8u - 2u * l)) & 256u);
                const uint32_t i2 = sb[8u * ib + 2u * l + 1u] | (((uint32_t)qh << (7u - 2u * l)) & 256u);
                const uint32_t signs = sb[72u + 4u * ib + l];
                const uint32_t g0 = pd_iq_grid4(dt, tabs, i1), g1 = pd_iq_grid4(dt, tabs, i2);
                wq[2u * i] = pd_iq_pack_signed32(g0, signs, 0u);
                wq[2u * i + 1u] = pd_iq_pack_signed32(g1, signs, 1u);
            }
            break;
        }
        case PD_KQ_IQ1S: {
            const uint16_t h = pd_iq_u16a(sb + 32u + 2u * ib);
            *f = pd_iq_f16a(rec) * (float)(2u * ((h >> 12) & 7u) + 1u) * 0.125f;
            const bool neg = (h & 0x8000u) != 0u;
            #pragma unroll
            for (uint32_t i = 0; i < 2u; ++i) {
                const uint32_t l = l0 + i;
                uint32_t g0, g1;
                pd_iq_grid8(dt, tabs, sb[4u * ib + l] | (((h >> (3u * l)) & 7u) << 8u), &g0, &g1);
                wq[2u * i] = pd_iq1_pack(g0, neg);
                wq[2u * i + 1u] = pd_iq1_pack(g1, neg);
            }
            break;
        }
        case PD_KQ_IQ1M: {
            const uint16_t sc = pd_iq_u16a(rec + 2u + 2u * (ib >> 1u));
            const uint32_t sh = 6u * (ib & 1u) + 3u * (w & 1u);
            *f = pd_iq_f16a(rec) * (float)(2u * ((sc >> sh) & 7u) + 1u) * 0.125f;
            const uint8_t hq = sb[32u + 2u * ib + (w & 1u)];
            #pragma unroll
            for (uint32_t i = 0; i < 2u; ++i) {
                const uint32_t l = l0 + i;
                const uint32_t idx = sb[4u * ib + l] | (((uint32_t)hq << (i ? 4u : 8u)) & 0x700u);
                const bool neg = (hq & (i ? 0x80u : 0x08u)) != 0u;
                uint32_t g0, g1;
                pd_iq_grid8(dt, tabs, idx, &g0, &g1);
                wq[2u * i] = pd_iq1_pack(g0, neg);
                wq[2u * i + 1u] = pd_iq1_pack(g1, neg);
            }
            break;
        }
        default: {  // IQ4_NL: block ib's lo (w even) / hi (w odd) nibbles
            *f = pd_iq_f16a(rec + 2u * ib);
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

// The default set (the global tables).
__device__ __forceinline__ void pd_iq_win_unpack(uint32_t dt, const uint8_t* __restrict__ sb,
                                                 const uint8_t* __restrict__ rec, uint32_t w,
                                                 int wq[4], float* f, float* g) {
    pd_iq_win_unpack_t(dt, sb, rec, w, wq, f, g, pd_iq_tabs_global(dt));
}
