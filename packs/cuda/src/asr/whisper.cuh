// asr/whisper.cuh - whisper-family (encoder-decoder ASR) serving kernels.
// Textually-included segment of the single pack translation unit.
// Not standalone-compilable: include order is defined by ../../pack.cu.
// ---------------------------------------------------------------- asr/whisper
//
// The bring-up decode lane drove whisper's two attentions
// through `pd_vision_attn_mma_kernel`, which is a PREFILL shape: its grid is
// ((nq+127)/128, heads, batch), so a decode step (nq == 1) launches 20 blocks
// of 8 warps in which 7 warps own no query row at all - and the 1500-frame
// cross-attention then streams 15.4 MB of f32 K/V through them at ~175 GB/s.
// Traced on a 21 s clip: 2944 of those calls, 88 us each, 51.6% of all GPU
// time in the request. That is the shape this file replaces.
//
// The replacement is flash-decoding: one query row per (slot, head), K/V
// resident at `kv_dtype`, the key range split across `splits` blocks that
// each carry an online-softmax partial, and a combine pass that merges them.
// Bytes drop 2x and the grid goes from 20 blocks to a few hundred, which is
// the whole gap.
//
// KV DTYPE. f16 is the default and the reference class (vLLM serves
// whisper at --dtype float16 and its KV cache follows the model dtype);
// fp8-e4m3 is the explicit `--kv-cache-dtype fp8_e4m3` opt-in, the same
// switch every other family carries. It matters more here than anywhere
// else in the engine: whisper's CROSS planes are 1500 frames per slot per
// layer and never shrink, so at c32 the decode step reads 32 slots x 32
// layers x 1500 x 1280 x 2 planes = 7.9 GB per TOKEN, and that one kernel
// measured 27% of all GPU time in a c32 battery at the card's achievable
// read bandwidth. Halving its element width is the only lever left on it -
// key splitting was tried and falsified (see `pd_whisper_splits`).
//
// Everything here is batched over decode SLOTS from the start: q/out are
// compact [batch, ...] in ACTIVE order while the K/V planes are indexed
// through `slots[b]`, so a finished slot leaves the active set without
// moving anyone's cache. The rest of the file is the launch-train collapse
// that goes with it - whisper's decode step ran 32 launches per layer
// (1024 per token) and every one of the elementwise ones was pure launch
// latency at 1280 floats.
//
// Numerics: f32 accumulate everywhere, fixed summation order (rows walk in
// index order per warp, warps merge in index order, splits merge in index
// order), `__expf` as in every other attention kernel we own. The fused
// epilogues below reproduce their unfused sequences term for term - same
// LayerNorm reduction structure as `pd_layernorm_kernel`, same erf-GELU
// constant as `pd_gelu_erf_kernel`, same `x += (proj + bias)` association.
// "Term for term" includes matching their FMA CONTRACTION, not just their term
// order: a select sitting between a multiply and its add silently costs the
// fma and rounds differently (see `pd_whisper_ln_staged` - that is what
// put four of these gates red).

#define PD_WD_NTH 256u
#define PD_WD_WARPS (PD_WD_NTH / 32u)
#define PD_WD_MAX_SPLITS 32u
// Key rows a warp takes per iteration. The first cut walked one row at a
// time and measured 790 GB/s on the cross planes - the loop was
// load -> warp-reduce -> load -> accumulate with every step depending on the
// last, so a warp had exactly one 128 B request in flight. Taking four rows
// per turn, through explicit staging arrays (ptxas serializes a
// memory loop through one temp cluster unless the loads land in named
// registers first), puts four independent requests in flight before the
// first reduction retires. At c32 this kernel is the decode step - 32 slots
// x 32 layers x 1500 frames of K and V is 7.9 GB per token.
#define PD_WD_RU 4u

// One (slot, head, key-chunk) partial. DPL = head dims per lane = hd/32, so
// a lane's slice of a K/V row is DPL contiguous halves and a warp's load is
// one 32*DPL*2 B contiguous run - the whole point of the lane mapping.
//
// `lens[b] + len_bias` is the live key count for slot b. Self-attention
// passes the position cursor as `lens` with len_bias 1 (keys 0..=pos), which
// is why there is no separate length buffer to keep in step with `pos`.
// Cross-attention passes lens == NULL and the window's frame count.
template <uint32_t DPL, typename KVT>
__global__ void __launch_bounds__(PD_WD_NTH) pd_whisper_dec_attn_kernel(
    const float* __restrict__ q, const float* __restrict__ qbias,
    const KVT* __restrict__ k, const KVT* __restrict__ v,
    const uint32_t* __restrict__ slots, const uint32_t* __restrict__ lens,
    __half* __restrict__ out, float* __restrict__ part, uint32_t kv_stride,
    uint32_t kv_len_def, uint32_t len_bias, uint32_t n_heads, uint32_t splits,
    float scale) {
    constexpr uint32_t HD = DPL * 32u;
    const uint32_t tid = threadIdx.x, warp = tid >> 5, lane = tid & 31u;
    const uint32_t h = blockIdx.x, b = blockIdx.y, sp = blockIdx.z;
    const uint32_t slot = slots ? slots[b] : b;
    const uint32_t len = (lens ? lens[b] : kv_len_def) + len_bias;

    // chunk boundaries are computed on device from a device-resident length:
    // nothing here reads host state, so the step is graph-capturable once the
    // rest of the loop is.
    const uint32_t chunk = (len + splits - 1u) / splits;
    const uint32_t r0 = sp * chunk;
    const uint32_t r1 = (r0 + chunk < len) ? (r0 + chunk) : len;

    // q is compact in ACTIVE order; K/V are indexed by the slot's plane row.
    const float* qp = q + ((size_t)b * n_heads + h) * HD;
    const size_t kvbase = ((size_t)slot * kv_stride) * (size_t)(n_heads * HD) + (size_t)h * HD;
    const uint32_t kvrow = n_heads * HD;

    float qv[DPL];
#pragma unroll
    for (uint32_t i = 0; i < DPL; ++i) {
        const uint32_t d = lane * DPL + i;
        // bias then scale, the order the unfused (bias_add -> attn) pair had
        qv[i] = (qbias ? (qp[d] + qbias[h * HD + d]) : qp[d]) * scale;
    }

    float m = -INFINITY, l = 0.f, acc[DPL];
#pragma unroll
    for (uint32_t i = 0; i < DPL; ++i) acc[i] = 0.f;

    // warp w owns rows [r0 + w*RU, ...) stepping WARPS*RU - a fixed order, so
    // the partial is bit-reproducible run to run. Within a turn the RU rows
    // are STAGED first and folded second: the staging is what buys the
    // memory-level parallelism, the fold order is what keeps the softmax
    // deterministic.
    for (uint32_t r = r0 + warp * PD_WD_RU; r < r1; r += PD_WD_WARPS * PD_WD_RU) {
        KVT ks[PD_WD_RU][DPL], vs[PD_WD_RU][DPL];
        uint32_t live = 0;
#pragma unroll
        for (uint32_t u = 0; u < PD_WD_RU; ++u) {
            const uint32_t rr = r + u;
            const bool ok = rr < r1;
            live += ok ? 1u : 0u;
            // out-of-range rows stage zeros and are masked out of the fold
            // below, so no branch divergence rides into the reduction
            const size_t off = kvbase + (size_t)(ok ? rr : r) * kvrow + lane * DPL;
#pragma unroll
            for (uint32_t i = 0; i < DPL; ++i) {
                ks[u][i] = k[off + i];
                vs[u][i] = v[off + i];
            }
        }
        float dots[PD_WD_RU];
#pragma unroll
        for (uint32_t u = 0; u < PD_WD_RU; ++u) {
            float dot = 0.f;
#pragma unroll
            for (uint32_t i = 0; i < DPL; ++i) dot += qv[i] * pd_kv_load(ks[u][i]);
#pragma unroll
            for (uint32_t s = 16u; s > 0u; s >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, s);
            dots[u] = dot;
        }
#pragma unroll
        for (uint32_t u = 0; u < PD_WD_RU; ++u) {
            if (u >= live) break;
            const float mn = fmaxf(m, dots[u]);
            const float corr = __expf(m - mn);
            const float p = __expf(dots[u] - mn);
            l = l * corr + p;
#pragma unroll
            for (uint32_t i = 0; i < DPL; ++i)
                acc[i] = acc[i] * corr + p * pd_kv_load(vs[u][i]);
            m = mn;
        }
    }

    // merge the block's four warp partials, warp order 0..3 (fixed)
    __shared__ float sh_acc[PD_WD_WARPS][HD];
    __shared__ float sh_m[PD_WD_WARPS], sh_l[PD_WD_WARPS];
#pragma unroll
    for (uint32_t i = 0; i < DPL; ++i) sh_acc[warp][lane * DPL + i] = acc[i];
    if (lane == 0) {
        sh_m[warp] = m;
        sh_l[warp] = l;
    }
    __syncthreads();
    if (warp != 0) return;

    float bm = sh_m[0], bl = 0.f, bacc[DPL];
#pragma unroll
    for (uint32_t i = 0; i < DPL; ++i) bacc[i] = 0.f;
#pragma unroll
    for (uint32_t w = 1; w < PD_WD_WARPS; ++w) bm = fmaxf(bm, sh_m[w]);
    if (bm == -INFINITY) {
        // this chunk started past the live length - emit a partial that the
        // combine can only ignore (weight exp(-1e30 - m) is a hard zero, and
        // an empty l keeps it out of the denominator). Subtracting -inf from
        // itself here would hand the combine a nan instead.
        bm = -1e30f;
    } else {
#pragma unroll
        for (uint32_t w = 0; w < PD_WD_WARPS; ++w) {
            const float cw = __expf(sh_m[w] - bm);
            bl += sh_l[w] * cw;
#pragma unroll
            for (uint32_t i = 0; i < DPL; ++i) bacc[i] += sh_acc[w][lane * DPL + i] * cw;
        }
    }

    if (splits == 1u) {
        // single chunk: no combine pass, normalize straight into `out`
        const float inv = bl > 0.f ? 1.f / bl : 0.f;
        __half* op = out + ((size_t)b * n_heads + h) * HD;
#pragma unroll
        for (uint32_t i = 0; i < DPL; ++i) op[lane * DPL + i] = __float2half(bacc[i] * inv);
        return;
    }
    float* pp = part + (((size_t)b * n_heads + h) * splits + sp) * (HD + 2u);
#pragma unroll
    for (uint32_t i = 0; i < DPL; ++i) pp[lane * DPL + i] = bacc[i];
    if (lane == 0) {
        pp[HD] = bm;
        pp[HD + 1u] = bl;
    }
}

// Merge `splits` partials per (slot, head). One block per (head, slot); the
// split walk is in index order, matching the partial kernel's own order.
__global__ void pd_whisper_dec_attn_combine_kernel(const float* __restrict__ part,
                                                   __half* __restrict__ out, uint32_t hd,
                                                   uint32_t n_heads, uint32_t splits) {
    const uint32_t h = blockIdx.x, b = blockIdx.y, d = threadIdx.x;
    if (d >= hd) return;
    const float* pb = part + ((size_t)b * n_heads + h) * splits * (size_t)(hd + 2u);
    float m = pb[hd];
    for (uint32_t s = 1; s < splits; ++s) m = fmaxf(m, pb[(size_t)s * (hd + 2u) + hd]);
    float acc = 0.f, l = 0.f;
    for (uint32_t s = 0; s < splits; ++s) {
        const float* ps = pb + (size_t)s * (hd + 2u);
        const float w = __expf(ps[hd] - m);
        acc += ps[d] * w;
        l += ps[hd + 1u] * w;
    }
    out[((size_t)b * n_heads + h) * hd + d] = __float2half(l > 0.f ? acc / l : 0.f);
}

// Split budget: aim for ~2 blocks per SM without cutting a chunk below one
// block-turn's worth of rows (WARPS*RU = 32) times four - a shorter chunk
// pays more combine than it buys in parallelism. It
// is a function of the PLANE stride, never of the live length, so a captured
// graph keeps its launch shape while lengths grow.
static uint32_t pd_whisper_splits(uint32_t batch, uint32_t n_heads, uint32_t kv_stride) {
    static int sms = 0;
    if (sms == 0) {
        int dev = 0, n = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev);
        sms = n > 0 ? n : 64;
    }
    // "Split until the grid reaches ~2 blocks per SM" - and no further. That
    // stops at s=1 for the c32 cross-attention (batch 32 x 20 heads = 640
    // blocks on 188 SMs), which LOOKS like a 3.4-blocks-per-SM quantization
    // tail: the kernel measures 245.8 MB in 174.8 us = 1406 GB/s, ~78% of the
    // 1792 GB/s spec number. FALSIFIED - forcing s = 2/4/8 through
    // a temporary override moved the c32 battery 29.96 -> 28.84 / 29.81 /
    // 30.07 req/s, i.e. nothing outside noise. 1406 GB/s is this card's
    // achievable read bandwidth for this access pattern; the spec peak was
    // the wrong roof to measure against. Do not re-try key splitting here
    // without a new reason - the byte volume is the only lever left.
    //
    // NUMERICAL CONSEQUENCE, measured. This returns 11 / 2 / 1
    // splits at batch 1 / 8 / 32 for the cross planes, and a different partial
    // count merges the online softmax in a different order. The active row count
    // is not pinned by the offered concurrency - it moves as slots fill and drain
    // - so two copies of the same clip inside one run can decode under different
    // split counts and come back with different text. Measured over 90 battery
    // legs: every c1 leg is clean, while 9/30 of the f16 concurrent legs have
    // at least one clip returning two distinct transcripts - other engines
    // behave the same way. Always a near-tie - the token behind Røst's is a 47.8% vs 45.2%
    // coin flip, and the whole c32 battery moves 0.02 pp of WER. Use
    // `examples/whisper_margin.rs` (top-2 margin per step) before treating any
    // such flip as a defect. Pinning the split count would buy reproducible
    // transcripts under load at the cost of the c1 cell, a guarantee no other
    // engine offers; left elected until that is a stated
    // product requirement. Transcript GATES run at c1 for this reason.
    const uint32_t want = (uint32_t)(2 * sms) / (batch * n_heads);
    const uint32_t cap = kv_stride / (4u * PD_WD_WARPS * PD_WD_RU);
    uint32_t s = want < 1u ? 1u : want;
    if (s > cap) s = cap;
    if (s < 1u) s = 1u;
    if (s > PD_WD_MAX_SPLITS) s = PD_WD_MAX_SPLITS;
    return s;
}

// 328: batched single-query decode attention over f16 K/V planes.
//   q     f32 [batch, n_heads, hd]           active order
//   qbias f32 [n_heads*hd] or NULL           folded in (whisper q/cross-q have one)
//   k, v  f16 [slots_cap, kv_stride, n_heads*hd]
//   slots u32 [batch] or NULL (identity)     which plane row each active b reads
//   lens  u32 [batch] or NULL                live keys = lens[b] + len_bias
//   out   f16 [batch, n_heads, hd]        the out_proj GEMM's operand: writing
//                                         f16 here is the convert the unfused
//                                         path did as its own launch
//   part  f32 [batch, n_heads, splits, hd+2] scratch; may be NULL iff splits==1
PD_EXPORT
int pd_whisper_dec_attn(const void* q, const void* qbias, const void* k, const void* v,
                        const void* slots, const void* lens, void* out, void* part,
                        uint32_t kv_stride, uint32_t kv_len_def, uint32_t len_bias,
                        uint32_t n_heads, uint32_t hd, uint32_t batch, float scale,
                        uint32_t kv_dtype, void* stream) {
    if (batch == 0 || n_heads == 0 || hd == 0) return 0;
    if ((hd % 64u) != 0u || hd > 128u) return cudaErrorInvalidValue;
    const uint32_t splits = part ? pd_whisper_splits(batch, n_heads, kv_stride) : 1u;
    dim3 grid(n_heads, batch, splits);
    cudaStream_t st = (cudaStream_t)stream;
#define PD_WD_LAUNCH(DPL, KVT)                                                            \
    pd_whisper_dec_attn_kernel<DPL, KVT><<<grid, PD_WD_NTH, 0, st>>>(                      \
        (const float*)q, (const float*)qbias, (const KVT*)k, (const KVT*)v,                \
        (const uint32_t*)slots, (const uint32_t*)lens, (__half*)out, (float*)part,         \
        kv_stride, kv_len_def, len_bias, n_heads, splits, scale)
    if (kv_dtype == PD_KV_FP8_E4M3) {
        if (hd == 64u) {
            PD_WD_LAUNCH(2u, __nv_fp8_e4m3);
        } else {
            PD_WD_LAUNCH(4u, __nv_fp8_e4m3);
        }
    } else if (hd == 64u) {
        PD_WD_LAUNCH(2u, __half);
    } else {
        PD_WD_LAUNCH(4u, __half);
    }
#undef PD_WD_LAUNCH
    if (splits > 1u) {
        dim3 cg(n_heads, batch);
        pd_whisper_dec_attn_combine_kernel<<<cg, 128, 0, st>>>((const float*)part,
                                                               (__half*)out, hd, n_heads,
                                                               splits);
    }
    return pd_launch_status();
}

// 329: x[b] = tok_table[tokens[b]] + pos_table[pos[b]] - whisper's decoder
// embedding is a plain row copy plus a LEARNED position row, and doing them
// as two gathers plus an add was three launches of 1280 floats.
__global__ void pd_whisper_embed_pos_kernel(const float* __restrict__ tok,
                                            const float* __restrict__ postab,
                                            const uint32_t* __restrict__ tokens,
                                            const uint32_t* __restrict__ pos,
                                            float* __restrict__ x, uint32_t d) {
    const uint32_t b = blockIdx.x;
    const float* tr = tok + (size_t)tokens[b] * d;
    const float* pr = postab + (size_t)pos[b] * d;
    float* xr = x + (size_t)b * d;
    for (uint32_t i = threadIdx.x; i < d; i += blockDim.x) xr[i] = tr[i] + pr[i];
}

PD_EXPORT
int pd_whisper_embed_pos(const void* tok, const void* postab, const void* tokens,
                         const void* pos, void* x, uint32_t d, uint32_t batch,
                         void* stream) {
    if (batch == 0 || d == 0) return 0;
    pd_whisper_embed_pos_kernel<<<batch, 256, 0, (cudaStream_t)stream>>>(
        (const float*)tok, (const float*)postab, (const uint32_t*)tokens,
        (const uint32_t*)pos, (float*)x, d);
    return pd_launch_status();
}

// 330: split a merged q|k|v landing and append this step's K/V to the slot
// caches. Replaces two bias adds and two device-to-device row copies per
// layer per token. k_proj has no bias anywhere in whisper - passing NULL for
// `bk` is the architecture, not an omission.
template <typename KVT>
__global__ void pd_whisper_qkv_split_kernel(const float* __restrict__ qkv,
                                            const float* __restrict__ bq,
                                            const float* __restrict__ bv,
                                            float* __restrict__ q, KVT* __restrict__ kc,
                                            KVT* __restrict__ vc,
                                            const uint32_t* __restrict__ slots,
                                            const uint32_t* __restrict__ pos, uint32_t d,
                                            uint32_t ctx) {
    const uint32_t b = blockIdx.x;
    const size_t src = (size_t)b * 3u * d;
    const size_t dst = ((size_t)(slots ? slots[b] : b) * ctx + pos[b]) * d;
    for (uint32_t i = threadIdx.x; i < d; i += blockDim.x) {
        q[(size_t)b * d + i] = qkv[src + i] + (bq ? bq[i] : 0.f);
        pd_kv_store(&kc[dst + i], qkv[src + d + i]);
        pd_kv_store(&vc[dst + i], qkv[src + 2u * d + i] + (bv ? bv[i] : 0.f));
    }
}

PD_EXPORT
int pd_whisper_qkv_split(const void* qkv, const void* bq, const void* bv, void* q, void* kc,
                         void* vc, const void* slots, const void* pos, uint32_t d,
                         uint32_t ctx, uint32_t batch, uint32_t kv_dtype, void* stream) {
    if (batch == 0 || d == 0) return 0;
    cudaStream_t st = (cudaStream_t)stream;
#define PD_WQ_LAUNCH(KVT)                                                                 \
    pd_whisper_qkv_split_kernel<KVT><<<batch, 256, 0, st>>>(                               \
        (const float*)qkv, (const float*)bq, (const float*)bv, (float*)q, (KVT*)kc,        \
        (KVT*)vc, (const uint32_t*)slots, (const uint32_t*)pos, d, ctx)
    if (kv_dtype == PD_KV_FP8_E4M3) {
        PD_WQ_LAUNCH(__nv_fp8_e4m3);
    } else {
        PD_WQ_LAUNCH(__half);
    }
#undef PD_WQ_LAUNCH
    return pd_launch_status();
}

// 331: store a window's cross-attention K or V into its slot plane as f16,
// bias folded in. Runs once per layer per window (not per token), but it is
// what lets the cross planes live at f16 without a staging round trip.
template <typename KVT>
__global__ void pd_whisper_kv_store_kernel(const float* __restrict__ src,
                                           const float* __restrict__ bias,
                                           KVT* __restrict__ dst,
                                           const uint32_t* __restrict__ slots, uint32_t rows,
                                           uint32_t d, uint32_t stride) {
    const uint32_t r = blockIdx.x, b = blockIdx.y;
    const size_t s = ((size_t)b * rows + r) * d;
    const size_t t = ((size_t)(slots ? slots[b] : b) * stride + r) * d;
    for (uint32_t i = threadIdx.x; i < d; i += blockDim.x)
        pd_kv_store(&dst[t + i], src[s + i] + (bias ? bias[i] : 0.f));
}

PD_EXPORT
int pd_whisper_kv_store(const void* src, const void* bias, void* dst, const void* slots,
                        uint32_t rows, uint32_t d, uint32_t stride, uint32_t batch,
                        uint32_t kv_dtype, void* stream) {
    if (batch == 0 || rows == 0 || d == 0) return 0;
    dim3 grid(rows, batch);
    cudaStream_t st = (cudaStream_t)stream;
#define PD_WS_LAUNCH(KVT)                                                                 \
    pd_whisper_kv_store_kernel<KVT><<<grid, 256, 0, st>>>(                                 \
        (const float*)src, (const float*)bias, (KVT*)dst, (const uint32_t*)slots, rows, d,  \
        stride)
    if (kv_dtype == PD_KV_FP8_E4M3) {
        PD_WS_LAUNCH(__nv_fp8_e4m3);
    } else {
        PD_WS_LAUNCH(__half);
    }
#undef PD_WS_LAUNCH
    return pd_launch_status();
}

// 388: split the encoder's FUSED q|k|v GEMM landing into the three planes
// vision_attn consumes, biases folded (k_proj has no bias - architecture,
// same as 330). Exists so the encoder's three per-layer M=1280 projections
// can run as one M=3840 tc5p call: at 1500x1280 the split GEMMs leave 46% of
// the clusters idle (3x12.60us -> 19.09 measured), and this launch
// replaces the two full-width bias_add launches that followed them.
__global__ void pd_whisper_enc_qkv_split_kernel(const float* __restrict__ qkv,
                                                const float* __restrict__ bq,
                                                const float* __restrict__ bv,
                                                float* __restrict__ q,
                                                float* __restrict__ k,
                                                float* __restrict__ v, uint32_t d) {
    const uint32_t r = blockIdx.x;
    const size_t src = (size_t)r * 3u * d, dst = (size_t)r * d;
    for (uint32_t i = threadIdx.x; i < d; i += blockDim.x) {
        q[dst + i] = qkv[src + i] + (bq ? bq[i] : 0.f);
        k[dst + i] = qkv[src + d + i];
        v[dst + i] = qkv[src + 2u * d + i] + (bv ? bv[i] : 0.f);
    }
}

PD_EXPORT
int pd_whisper_enc_qkv_split(const void* qkv, const void* bq, const void* bv,
                             void* q, void* k, void* v, uint32_t d,
                             uint32_t rows, void* stream) {
    if (rows == 0 || d == 0) return 0;
    pd_whisper_enc_qkv_split_kernel<<<rows, 256, 0, (cudaStream_t)stream>>>(
        (const float*)qkv, (const float*)bq, (const float*)bv, (float*)q,
        (float*)k, (float*)v, d);
    return pd_launch_status();
}

// 389: cross-K/V store off a LAYER-BATCHED landing. Every
// decoder layer's cross K (or V) projection reads the same encoder states,
// so the runner concatenates the 32 weight planes and runs one M=n_layer*d
// GEMM (64 x 12.60us of half-idle launches -> 2 x 135 measured); this then
// lands every layer's slot plane from that fused [rows, n_layer*d] landing
// in one launch. `dsts` is a device array of n_layer plane base pointers
// (uploaded once at pool alloc - capture-safe); `bias` is the concatenated
// [n_layer*d] plane (V) or NULL (K - whisper ships no k bias). Single-slot
// like 331's admission use: slot = slots[0].
template <typename KVT>
__global__ void pd_whisper_kv_store_batch_kernel(
    const float* __restrict__ src, const float* __restrict__ bias,
    KVT* const* __restrict__ dsts, const uint32_t* __restrict__ slots,
    uint32_t d, uint32_t ld, uint32_t stride, uint32_t rps) {
    const uint32_t r = blockIdx.x, li = blockIdx.y;
    // batched admission: the landing carries rps rows per audio,
    // audio-major; row r belongs to audio r/rps and stores into its slot
    const float* s = src + (size_t)r * ld + (size_t)li * d;
    KVT* t = dsts[li] + ((size_t)(slots ? slots[r / rps] : 0u) * stride + r % rps) * d;
    const float* b = bias ? bias + (size_t)li * d : nullptr;
    for (uint32_t i = threadIdx.x; i < d; i += blockDim.x)
        pd_kv_store(&t[i], s[i] + (b ? b[i] : 0.f));
}

static int pd_whisper_kv_store_impl(const void* src, const void* bias,
                                    const void* dsts, const void* slots,
                                    uint32_t rows, uint32_t d, uint32_t n_layer,
                                    uint32_t stride, uint32_t kv_dtype,
                                    uint32_t rps, void* stream) {
    if (rows == 0 || d == 0 || n_layer == 0) return 0;
    dim3 grid(rows, n_layer);
    cudaStream_t st = (cudaStream_t)stream;
    if (kv_dtype == PD_KV_FP8_E4M3)
        pd_whisper_kv_store_batch_kernel<__nv_fp8_e4m3><<<grid, 256, 0, st>>>(
            (const float*)src, (const float*)bias,
            (__nv_fp8_e4m3* const*)dsts, (const uint32_t*)slots, d,
            n_layer * d, stride, rps);
    else
        pd_whisper_kv_store_batch_kernel<__half><<<grid, 256, 0, st>>>(
            (const float*)src, (const float*)bias, (__half* const*)dsts,
            (const uint32_t*)slots, d, n_layer * d, stride, rps);
    return pd_launch_status();
}

// slot 389, signature frozen (append-only ABI): rps == rows -> every row
// indexes slots[0], the original single-slot admission behavior exactly
PD_EXPORT
int pd_whisper_kv_store_batch(const void* src, const void* bias, const void* dsts,
                              const void* slots, uint32_t rows, uint32_t d,
                              uint32_t n_layer, uint32_t stride, uint32_t kv_dtype,
                              void* stream) {
    return pd_whisper_kv_store_impl(src, bias, dsts, slots, rows, d, n_layer,
                                    stride, kv_dtype, rows, stream);
}

// slot 405 (batched admission): the landing is audio-major with
// rows_per_slot rows per audio; row r stores into slots[r / rows_per_slot]
PD_EXPORT
int pd_whisper_kv_store_slots(const void* src, const void* bias, const void* dsts,
                              const void* slots, uint32_t rows, uint32_t d,
                              uint32_t n_layer, uint32_t stride, uint32_t kv_dtype,
                              uint32_t rows_per_slot, void* stream) {
    if (rows_per_slot == 0) return -1;
    return pd_whisper_kv_store_impl(src, bias, dsts, slots, rows, d, n_layer,
                                    stride, kv_dtype, rows_per_slot, stream);
}

// Rows per thread the staged norms hold in registers. 256 threads x 8 covers
// every whisper d_model (large-v3 1280, medium 1024, small 768) with room
// over; wider rows fall back to the re-reading body below.
#define PD_WLN_RU 8u

// Block-wide sum with the warp-partial staging `pd_layernorm_kernel` uses -
// warp-shuffle tree, one slot per warp, then the partials folded in warp
// order. Every thread folds the partials rather than thread 0 folding and
// broadcasting through shared memory: same terms in the same order, so the
// result is bit-identical, and it drops a `__syncthreads` plus a shared
// round trip from a kernel that is nothing but latency.
__device__ __forceinline__ float pd_wln_block_sum(float acc, float* wsum) {
    const uint32_t lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    const uint32_t nwarps = (blockDim.x + 31u) >> 5;
    for (uint32_t s = 16; s > 0; s >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, s);
    if (lane == 0) wsum[warp] = acc;
    __syncthreads();
    float t = 0.0f;
    for (uint32_t i = 0; i < nwarps; ++i) t += wsum[i];
    return t;
}

// The fused norms over a row already STAGED in registers: `xs[u]` is
// x[tid + u*blockDim.x], zero past the row end.
//
// Why staging is the whole kernel: one CTA over a ~5 KB row is pure memory
// latency, and the re-reading form below walks that row three times (mean,
// variance, write-out) - four for the residual seam, which updates x first.
// Each walk is a dependent DRAM round trip, and it traced at 5.7 us for
// 32x1280 (~29 GB/s). Held in registers there is exactly one. Named array,
// not a loop temp: ptxas otherwise funnels loop-carried loads through a
// single temp cluster and serializes them (staging).
//
// Arithmetic is unchanged: thread tid still folds its slice in u order, the
// same warp tree reduces it, and the out-of-range slots contribute a
// trailing +0.0 (exact for any finite partial), so this is term-for-term
// what the unfused bias_add + add + layernorm chain produced.
__device__ __forceinline__ void pd_whisper_ln_staged(const float (&xs)[PD_WLN_RU],
                                                     const float* __restrict__ w,
                                                     const float* __restrict__ b,
                                                     __half* __restrict__ orow, uint32_t n,
                                                     float eps) {
    const uint32_t tid = threadIdx.x, nth = blockDim.x;
    __shared__ float wm[32], wv[32];
    float acc = 0.0f;
#pragma unroll
    for (uint32_t u = 0; u < PD_WLN_RU; ++u) acc += xs[u];
    const float mean = pd_wln_block_sum(acc, wm) / (float)n;
    float vacc = 0.0f;
#pragma unroll
    for (uint32_t u = 0; u < PD_WLN_RU; ++u) {
        // Dead slots hold 0, whose (0 - mean)^2 is not zero - so they have to
        // be guarded or the tail threads inject mean^2 into the variance. Zero
        // the DIFFERENCE, not the product: `vacc += ok ? dd*dd : 0.f` puts a
        // select between the multiply and the add, and ptxas will not contract
        // across it - the PTX reads `mul.f32` + `selp.f32` + `add.f32`
        // where pd_layernorm_kernel gets one `fma.rn.f32`. That is a different
        // rounding of the variance, which is what put four whisper gates red
        // 62 of 1.92M f16 outputs off by exactly one ulp. A zeroed
        // difference squares and adds exactly, so the guard costs nothing and
        // the expression stays the single unconditional multiply-add the
        // unfused kernel contracts.
        const float dd = (tid + u * nth) < n ? (xs[u] - mean) : 0.0f;
        vacc += dd * dd;
    }
    const float inv = rsqrtf(pd_wln_block_sum(vacc, wv) / (float)n + eps);
#pragma unroll
    for (uint32_t u = 0; u < PD_WLN_RU; ++u) {
        const uint32_t i = tid + u * nth;
        if (i < n) orow[i] = __float2half((xs[u] - mean) * inv * w[i] + b[i]);
    }
}

// The LayerNorm body both fused norms share - identical reduction structure
// to `pd_layernorm_kernel` (block-wide two-pass mean/variance through the
// same 32-slot warp-sum staging, summed by thread 0 in warp order), so the
// fused forms are term-for-term what the unfused pair produced. The staged
// form above is what actually runs at every whisper width; this stays as the
// fallback for rows too wide to hold.
__device__ __forceinline__ void pd_whisper_ln_body(const float* __restrict__ xr,
                                                   const float* __restrict__ w,
                                                   const float* __restrict__ b,
                                                   __half* __restrict__ orow, uint32_t n,
                                                   float eps) {
    const uint32_t tid = threadIdx.x, nth = blockDim.x;
    const uint32_t lane = tid & 31u, warp = tid >> 5, nwarps = (nth + 31u) >> 5;
    __shared__ float wsum[32];
    __shared__ float s_mean, s_inv;

    float acc = 0.0f;
    for (uint32_t i = tid; i < n; i += nth) acc += xr[i];
    for (uint32_t s = 16; s > 0; s >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, s);
    if (lane == 0) wsum[warp] = acc;
    __syncthreads();
    if (tid == 0) {
        float m = 0.0f;
        for (uint32_t i = 0; i < nwarps; ++i) m += wsum[i];
        s_mean = m / (float)n;
    }
    __syncthreads();
    const float mean = s_mean;
    float vacc = 0.0f;
    for (uint32_t i = tid; i < n; i += nth) {
        const float dd = xr[i] - mean;
        vacc += dd * dd;
    }
    for (uint32_t s = 16; s > 0; s >>= 1) vacc += __shfl_down_sync(0xffffffffu, vacc, s);
    if (lane == 0) wsum[warp] = vacc;
    __syncthreads();
    if (tid == 0) {
        float v = 0.0f;
        for (uint32_t i = 0; i < nwarps; ++i) v += wsum[i];
        s_inv = rsqrtf(v / (float)n + eps);
    }
    __syncthreads();
    const float inv = s_inv;
    for (uint32_t i = tid; i < n; i += nth)
        orow[i] = __float2half((xr[i] - mean) * inv * w[i] + b[i]);
}

// 332: LayerNorm straight to f16 - every whisper GEMM eats f16 activations,
// so the unfused path always paid a second launch to convert 1280 floats.
__global__ void pd_whisper_ln_f16_kernel(const float* __restrict__ x,
                                         const float* __restrict__ w,
                                         const float* __restrict__ b, __half* __restrict__ out,
                                         uint32_t n, float eps) {
    const float* xr = x + (size_t)blockIdx.x * n;
    __half* orow = out + (size_t)blockIdx.x * n;
    const uint32_t tid = threadIdx.x, nth = blockDim.x;
    if (n <= nth * PD_WLN_RU) {
        float xs[PD_WLN_RU];
#pragma unroll
        for (uint32_t u = 0; u < PD_WLN_RU; ++u) {
            const uint32_t i = tid + u * nth;
            xs[u] = i < n ? xr[i] : 0.0f;
        }
        pd_whisper_ln_staged(xs, w, b, orow, n, eps);
        return;
    }
    pd_whisper_ln_body(xr, w, b, orow, n, eps);
}

PD_EXPORT
int pd_whisper_ln_f16(const void* x, const void* w, const void* b, void* out, uint32_t rows,
                      uint32_t n, float eps, void* stream) {
    if (rows == 0 || n == 0) return 0;
    pd_whisper_ln_f16_kernel<<<rows, 256, 0, (cudaStream_t)stream>>>(
        (const float*)x, (const float*)w, (const float*)b, (__half*)out, n, eps);
    return pd_launch_status();
}

// 333: the whole residual seam in one launch - `x += proj + bias`, then the
// next block's pre-norm out of the updated residual, at f16. Whisper's
// decoder has three of these per layer (self-out -> cross-norm, cross-out ->
// mlp-norm, mlp-out -> next layer's norm / the final norm), so this collapses
// twelve launches per layer into three. The association is exactly what the
// unfused chain had: bias_add(proj) then add(x, proj).
__global__ void pd_whisper_res_ln_f16_kernel(float* __restrict__ x,
                                             const float* __restrict__ proj,
                                             const float* __restrict__ bias,
                                             const float* __restrict__ w,
                                             const float* __restrict__ b,
                                             __half* __restrict__ out, uint32_t n, float eps) {
    const uint32_t row = blockIdx.x;
    float* xr = x + (size_t)row * n;
    const float* pr = proj + (size_t)row * n;
    __half* orow = out + (size_t)row * n;
    const uint32_t tid = threadIdx.x, nth = blockDim.x;
    if (n <= nth * PD_WLN_RU) {
        // the updated residual goes to DRAM once (the next layer reads it)
        // and stays in registers for the norm - and since every thread now
        // only reads its own slice, the barrier the re-reading form needed
        // between the update and the norm is gone too
        float xs[PD_WLN_RU];
#pragma unroll
        for (uint32_t u = 0; u < PD_WLN_RU; ++u) {
            const uint32_t i = tid + u * nth;
            xs[u] = i < n ? xr[i] + (pr[i] + bias[i]) : 0.0f;
        }
#pragma unroll
        for (uint32_t u = 0; u < PD_WLN_RU; ++u) {
            const uint32_t i = tid + u * nth;
            if (i < n) xr[i] = xs[u];
        }
        pd_whisper_ln_staged(xs, w, b, orow, n, eps);
        return;
    }
    for (uint32_t i = tid; i < n; i += nth) xr[i] += pr[i] + bias[i];
    __syncthreads();
    pd_whisper_ln_body(xr, w, b, orow, n, eps);
}

PD_EXPORT
int pd_whisper_res_ln_f16(void* x, const void* proj, const void* bias, const void* w,
                          const void* b, void* out, uint32_t rows, uint32_t n, float eps,
                          void* stream) {
    if (rows == 0 || n == 0) return 0;
    pd_whisper_res_ln_f16_kernel<<<rows, 256, 0, (cudaStream_t)stream>>>(
        (float*)x, (const float*)proj, (const float*)bias, (const float*)w, (const float*)b,
        (__half*)out, n, eps);
    return pd_launch_status();
}

// 334: fc1's epilogue - bias, exact-erf GELU, and the f16 cast fc2 needs, in
// one pass over the 5120-wide landing. Same constant and form as
// `pd_gelu_erf_kernel` (PyTorch's approximate="none").
__global__ void pd_whisper_bias_gelu_f16_kernel(const float* __restrict__ x,
                                                const float* __restrict__ bias,
                                                __half* __restrict__ out, uint32_t n,
                                                uint64_t total) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const float xi = x[i] + bias[i % n];
    out[i] = __float2half(0.5f * xi * (1.0f + erff(xi * 0.70710678118654752440084436210484f)));
}

PD_EXPORT
int pd_whisper_bias_gelu_f16(const void* x, const void* bias, void* out, uint32_t rows,
                             uint32_t n, void* stream) {
    if (rows == 0 || n == 0) return 0;
    const uint64_t total = (uint64_t)rows * n;
    const uint64_t blocks = (total + 255ull) / 256ull;
    pd_whisper_bias_gelu_f16_kernel<<<(uint32_t)blocks, 256, 0, (cudaStream_t)stream>>>(
        (const float*)x, (const float*)bias, (__half*)out, n, total);
    return pd_launch_status();
}

// ---------------------------------------------------------------- timestamps
//
// Whisper's ApplyTimestampRules, on device.
//
// Why this is not OPTIONAL. Whisper emits timestamps as vocabulary tokens, and
// the prompt only OFFERS the mode: dropping `<|notimestamps|>` from the prompt
// lets the model emit times, it does not make it. Measured on KB-Whisper
// (examples/whisper_margin with PADDOCK_MARGIN_TIMESTAMPS=1): at the first
// sampled position the greedy argmax is `<|notimestamps|>` itself at p=0.794,
// with `<|0.00|>` the runner-up at p=0.204 - the fine-tune's own prior is to
// opt out. Plain greedy decoding therefore returns a transcript with no times
// in it and no error, which is the silent-failure shape. The reference
// implementation (openai/whisper decoding.py `ApplyTimestampRules`, mirrored by
// faster-whisper and whisper.cpp) constrains the logits instead, and this is
// that filter, rule for rule.
//
// On device because the alternative is reading a 51866-float row back per slot
// per step (6.6 MB at c32) and breaking the captured graph to do it. The
// per-row STATE the rules need is three facts about the tokens sampled so far,
// which the scheduler already tracks, so it rides in as two u32 per row.
//
// state[row*2+0] = flags, state[row*2+1] = ts_floor (the lowest timestamp id
// still allowed; only read when PD_WTS_HAVE is set).
#define PD_WTS_ON 1u     // this row wants timestamps at all (mixed batches)
#define PD_WTS_BEGIN 2u  // nothing sampled yet - the first emitted token
#define PD_WTS_LAST 4u   // the last sampled token was a timestamp
#define PD_WTS_PENULT 8u // the one before it was (or there is only one)
#define PD_WTS_HAVE 16u  // at least one timestamp has been sampled

__global__ void pd_whisper_ts_rules_kernel(float* __restrict__ logits,
                                           const uint32_t* __restrict__ state, uint32_t vocab,
                                           uint32_t eot, uint32_t no_ts, uint32_t ts_begin,
                                           uint32_t max_init) {
    const uint32_t row = blockIdx.x, tid = threadIdx.x, nth = blockDim.x;
    const uint32_t flags = state[row * 2u];
    if (!(flags & PD_WTS_ON)) return;
    const uint32_t floor_ts = state[row * 2u + 1u];
    float* x = logits + (size_t)row * vocab;
    const bool at_begin = (flags & PD_WTS_BEGIN) != 0u;
    const bool last_ts = (flags & PD_WTS_LAST) != 0u;
    const bool penult_ts = (flags & PD_WTS_PENULT) != 0u;
    const bool have_ts = (flags & PD_WTS_HAVE) != 0u;

    // ---- the hard masks, in the reference's own order ----
    for (uint32_t i = tid; i < vocab; i += nth) {
        bool kill = (i == no_ts); // the mode token is never re-emitted here
        // timestamps come in PAIRS: one closes a segment, the next opens the
        // following one. So after a lone timestamp only a timestamp (or eot)
        // may follow, and after a pair only text may.
        if (last_ts && penult_ts && i >= ts_begin) kill = true;
        if (last_ts && !penult_ts && i < eot) kill = true;
        // and they never go backwards
        if (have_ts && i >= ts_begin && i < floor_ts) kill = true;
        if (at_begin) {
            // a window OPENS on a timestamp - that is the rule that makes the
            // model emit them at all rather than falling into plain text
            if (i < ts_begin) kill = true;
            // ...and not an arbitrarily late one: `max_initial_timestamp`
            // stops the model from silently skipping the start of a window
            if (max_init != 0xffffffffu && i > ts_begin + max_init) kill = true;
        }
        if (kill) x[i] = -INFINITY;
    }
    __syncthreads();

    // ---- the probability-mass rule ----
    // "if the total probability of all timestamps beats the single best text
    // token, emit a timestamp." Log-softmax subtracts the same normaliser from
    // both sides, so comparing logsumexp(timestamps) against max(text) on the
    // raw (already masked) logits is the same comparison the reference makes.
    // Skipped where it cannot matter: at_begin has no text left, and a closed
    // pair has no timestamps left.
    if (at_begin || (last_ts && penult_ts)) return;

    __shared__ float sh_m[8], sh_s[8], sh_t[8];
    float mts = -INFINITY, sts = 0.f, mtx = -INFINITY;
    for (uint32_t i = tid + ts_begin; i < vocab; i += nth) {
        const float v = x[i];
        if (v > mts) {
            sts = sts * __expf(mts - v) + 1.f;
            mts = v;
        } else if (v > -INFINITY) {
            sts += __expf(v - mts);
        }
    }
    for (uint32_t i = tid; i < ts_begin; i += nth) mtx = fmaxf(mtx, x[i]);
#pragma unroll
    for (uint32_t off = 16; off > 0; off >>= 1) {
        const float om = __shfl_down_sync(0xffffffffu, mts, off);
        const float os = __shfl_down_sync(0xffffffffu, sts, off);
        if (om > mts) {
            sts = sts * __expf(mts - om) + os;
            mts = om;
        } else if (om > -INFINITY) {
            sts += os * __expf(om - mts);
        }
        mtx = fmaxf(mtx, __shfl_down_sync(0xffffffffu, mtx, off));
    }
    const uint32_t lane = tid & 31u, warp = tid >> 5, nwarp = (nth + 31u) >> 5;
    if (lane == 0) {
        sh_m[warp] = mts;
        sh_s[warp] = sts;
        sh_t[warp] = mtx;
    }
    __syncthreads();
    __shared__ bool ts_wins;
    if (tid == 0) {
        for (uint32_t w = 1; w < nwarp; ++w) {
            if (sh_m[w] > mts) {
                sts = sts * __expf(mts - sh_m[w]) + sh_s[w];
                mts = sh_m[w];
            } else if (sh_m[w] > -INFINITY) {
                sts += sh_s[w] * __expf(sh_m[w] - mts);
            }
            mtx = fmaxf(mtx, sh_t[w]);
        }
        // logsumexp over the timestamp range vs the best single text logit
        ts_wins = (sts > 0.f) && ((mts + logf(sts)) > mtx);
    }
    __syncthreads();
    if (!ts_wins) return;
    // NOTE this masks eot too (eot < ts_begin) - the reference's own behaviour:
    // a window that wants a timestamp here does not get to stop instead.
    for (uint32_t i = tid; i < ts_begin; i += nth) x[i] = -INFINITY;
}

// 343: whisper's ApplyTimestampRules over a [rows, vocab] logits block, in
// place, per row, before the greedy pick.
//   logits f32 [rows, vocab]   modified in place
//   state  u32 [rows, 2]       {flags (PD_WTS_*), lowest allowed timestamp id}
//   max_init                   largest initial-timestamp OFFSET from ts_begin,
//                              or 0xffffffff for no limit
PD_EXPORT
int pd_whisper_ts_rules(void* logits, const void* state, uint32_t rows, uint32_t vocab,
                        uint32_t eot, uint32_t no_ts, uint32_t ts_begin, uint32_t max_init,
                        void* stream) {
    if (rows == 0 || vocab == 0) return 0;
    if (ts_begin >= vocab || eot >= vocab || no_ts >= vocab) return cudaErrorInvalidValue;
    pd_whisper_ts_rules_kernel<<<rows, 256, 0, (cudaStream_t)stream>>>(
        (float*)logits, (const uint32_t*)state, vocab, eot, no_ts, ts_begin, max_init);
    return pd_launch_status();
}

// Cross-attention probabilities for the ALIGNMENT HEADS - the read-out that
// makes word-level timing possible.
//
// Whisper has no frame-level emission head, so the only thing in the model that
// knows when a word was spoken is where the decoder LOOKED while emitting it.
// `pd_whisper_dec_attn` is flash-style: it consumes softmax(QK^T) inside its
// online loop and never materialises it, which is right for the hot path and
// useless here. So this is a separate, plain kernel that computes the same
// probabilities for a HANDFUL of nominated heads and writes them out.
//
// Deliberately not folded into the decode kernel. Word timing is opt-in and the
// alignment pass runs off the latency path, so making every transcription carry
// a materialised attention row - or a second graph variant - to serve a feature
// most requests never ask for is the wrong trade. Dumping only the alignment
// heads is also what keeps this cheap: large-v3 nominates 10 of its 640
// (layer, head) pairs, so a 30 s window costs ~12 MB rather than ~750.
//
// One WARP per key row. A thread per row would stride each load by a whole
// K row and shred the coalescing; a warp's slice is `hd` contiguous elements,
// which is the same access shape the decode kernel is built around.
template <typename KVT>
__global__ void __launch_bounds__(256) pd_whisper_xattn_probs_kernel(
    const float* __restrict__ q, const float* __restrict__ qbias,
    const KVT* __restrict__ k, const uint32_t* __restrict__ slots,
    const uint32_t* __restrict__ heads, float* __restrict__ out, uint32_t kv_stride,
    uint32_t n_enc, uint32_t n_heads, uint32_t hd, uint32_t n_sel, float scale) {
    constexpr uint32_t NW = 8u;  // 256 threads
    const uint32_t sel = blockIdx.x, b = blockIdx.y;
    const uint32_t h = heads[sel];
    const uint32_t slot = slots ? slots[b] : b;
    const uint32_t tid = threadIdx.x, warp = tid >> 5, lane = tid & 31u;

    const float* qp = q + ((size_t)b * n_heads + h) * hd;
    const size_t kbase = (size_t)slot * kv_stride * (size_t)(n_heads * hd) + (size_t)h * hd;
    const uint32_t krow = n_heads * hd;
    float* op = out + ((size_t)b * n_sel + sel) * n_enc;

    // bias then scale - the order pd_whisper_dec_attn folds them in, so the
    // scores here are the ones the decode actually attended with
    __shared__ float sq[128];
    for (uint32_t i = tid; i < hd; i += 256u)
        sq[i] = (qbias ? (qp[i] + qbias[h * hd + i]) : qp[i]) * scale;
    __syncthreads();

    // pass 1: raw scores into `out`, and the block max. The butterfly leaves
    // the full dot in every lane, so `mymax` is the warp's max already.
    float mymax = -INFINITY;
    for (uint32_t r = warp; r < n_enc; r += NW) {
        const KVT* kr = k + kbase + (size_t)r * krow;
        float dot = 0.f;
        for (uint32_t i = lane; i < hd; i += 32u) dot += sq[i] * pd_kv_load(kr[i]);
#pragma unroll
        for (uint32_t s = 16u; s > 0u; s >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, s);
        if (lane == 0) op[r] = dot;
        mymax = fmaxf(mymax, dot);
    }
    __shared__ float smax[NW], ssum[NW];
    if (lane == 0) smax[warp] = mymax;
    // also the barrier that publishes pass 1's `op` writes to pass 2
    __syncthreads();
    float mx = smax[0];
#pragma unroll
    for (uint32_t w = 1; w < NW; ++w) mx = fmaxf(mx, smax[w]);

    // pass 2: exponentiate in place and total. Reading `out` back rather than
    // K again: the row is 1500 floats against 1500 * hd halves of keys.
    float mysum = 0.f;
    for (uint32_t r = tid; r < n_enc; r += 256u) {
        const float e = __expf(op[r] - mx);
        op[r] = e;
        mysum += e;
    }
#pragma unroll
    for (uint32_t s = 16u; s > 0u; s >>= 1) mysum += __shfl_xor_sync(0xffffffffu, mysum, s);
    if (lane == 0) ssum[warp] = mysum;
    __syncthreads();
    float tot = 0.f;
#pragma unroll
    for (uint32_t w = 0; w < NW; ++w) tot += ssum[w];

    // a row with no mass at all writes zeros rather than nan - an empty
    // window is a real input, and a nan here would poison the whole DTW
    const float inv = tot > 0.f ? (1.f / tot) : 0.f;
    for (uint32_t r = tid; r < n_enc; r += 256u) op[r] *= inv;
}

// 355: softmax(QK^T) over the encoder frames, for nominated cross-attention
// heads only. The word-timing read-out.
//   q      f32 [batch, n_heads, hd]          active order, as pd_whisper_dec_attn
//   qbias  f32 [n_heads*hd] or NULL          folded in, same order as the decode
//   k      KVT [slots_cap, kv_stride, n_heads*hd]   the layer's cross-K plane
//   slots  u32 [batch] or NULL (identity)
//   heads  u32 [n_sel]                       which heads of this layer to dump
//   out    f32 [batch, n_sel, n_enc]         rows sum to 1 (or to 0 if empty)
PD_EXPORT
int pd_whisper_xattn_probs(const void* q, const void* qbias, const void* k, const void* slots,
                           const void* heads, void* out, uint32_t kv_stride, uint32_t n_enc,
                           uint32_t n_heads, uint32_t hd, uint32_t n_sel, uint32_t batch,
                           float scale, uint32_t kv_dtype, void* stream) {
    if (batch == 0 || n_sel == 0 || n_enc == 0 || n_heads == 0 || hd == 0) return 0;
    if (hd > 128u) return cudaErrorInvalidValue;
    dim3 grid(n_sel, batch);
    cudaStream_t st = (cudaStream_t)stream;
    if (kv_dtype == PD_KV_FP8_E4M3) {
        pd_whisper_xattn_probs_kernel<__nv_fp8_e4m3><<<grid, 256, 0, st>>>(
            (const float*)q, (const float*)qbias, (const __nv_fp8_e4m3*)k,
            (const uint32_t*)slots, (const uint32_t*)heads, (float*)out, kv_stride, n_enc,
            n_heads, hd, n_sel, scale);
    } else {
        pd_whisper_xattn_probs_kernel<__half><<<grid, 256, 0, st>>>(
            (const float*)q, (const float*)qbias, (const __half*)k, (const uint32_t*)slots,
            (const uint32_t*)heads, (float*)out, kv_stride, n_enc, n_heads, hd, n_sel, scale);
    }
    return pd_launch_status();
}
