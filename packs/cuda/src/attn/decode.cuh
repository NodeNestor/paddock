// attn/decode.cuh (formerly 03_attn_decode.cuh) - fused per-head RMSNorm+YaRN rope; decode attention family (partial/combine, batch, paged)
// Textually-included segment of the single pack translation unit.
// Not standalone-compilable: include order is defined by ../pack.cu.
#ifdef PD_KRS_DUMP
// rung-4 score-dump scratch (debug-only define - see the in-kernel hook)
__device__ float pd_krs_dump[16 * 33];
#endif
// ------------------------------------- fused per-head RMSNorm + YaRN rope
// The encoder's q/k pipeline ran three separate global passes per side
// (rmsnorm_batch -> rope_yarn_batch -> kv_append_batch); at prefill batches
// of ~60k rows those elementwise passes cost more than the qkv GEMMs
// themselves, and rmsnorm_batch's one-256-thread-block-per-row shape leaves
// 224 threads idle on a 128-dim head row. One warp per head instead: lane
// holds dims [4l, 4l+4) as a float4, the square-sum reduction is the same
// lane-order shfl chain the old kernel used on its single live warp (bit-
// exact), and the rope pair (k, k+64) is exchanged via shfl_xor(16) with the
// theta chain replicated per pair index exactly as rope_yarn_batch does -
// the fused output is bit-identical to the three-pass sequence. head_dim
// must be 128 (every Qwen3 encoder head); the engine falls back to the
// separate passes otherwise.
struct PdRopeArgs {
    float theta_scale, freq_scale, corr_low, corr_high, ext_factor, mscale;
};

__device__ __forceinline__ float4 pd_norm_rope_head(
    const float* __restrict__ head, const float* __restrict__ w, float eps,
    unsigned int pos, PdRopeArgs rp, uint32_t lane) {
    float4 v = reinterpret_cast<const float4*>(head)[lane];
    float acc = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    #pragma unroll
    for (uint32_t s = 16; s > 0; s >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, s);
    const float sum = __shfl_sync(0xffffffffu, acc, 0);
    const float inv = 1.0f / sqrtf(sum / 128.0f + eps);
    const float4 wv = reinterpret_cast<const float4*>(w)[lane];
    float4 n;
    n.x = v.x * inv * wv.x;
    n.y = v.y * inv * wv.y;
    n.z = v.z * inv * wv.z;
    n.w = v.w * inv * wv.w;
    // rope pairs (k, k+64): lanes 0-15 hold the low half, 16-31 the high;
    // both sides derive the same per-pair angle (theta chain of k0+j
    // multiplies, matching rope_yarn_batch's per-lane chain bit-exactly)
    const uint32_t k0 = 4u * (lane & 15u);
    float theta = (float)pos;
    for (uint32_t i = 0; i < k0; ++i) theta *= rp.theta_scale;
    float4 p;
    p.x = __shfl_xor_sync(0xffffffffu, n.x, 16);
    p.y = __shfl_xor_sync(0xffffffffu, n.y, 16);
    p.z = __shfl_xor_sync(0xffffffffu, n.z, 16);
    p.w = __shfl_xor_sync(0xffffffffu, n.w, 16);
    const bool low = lane < 16u;
    float* nn = &n.x;
    const float* pp = &p.x;
    #pragma unroll
    for (uint32_t j = 0; j < 4u; ++j) {
        const float k = (float)(k0 + j);
        const float y = (k - rp.corr_low) / fmaxf(0.001f, rp.corr_high - rp.corr_low);
        const float ramp = (1.0f - fminf(1.0f, fmaxf(0.0f, y))) * rp.ext_factor;
        const float angle = (rp.freq_scale * theta) * (1.0f - ramp) + theta * ramp;
        const float s = sinf(angle) * rp.mscale;
        const float c = cosf(angle) * rp.mscale;
        // low lane owns a (dim k), partner p is b (dim k+64); high lane owns
        // b with partner a - same formulas as the in-place rope
        nn[j] = low ? nn[j] * c - pp[j] * s : pp[j] * s + nn[j] * c;
        theta *= rp.theta_scale;
    }
    return n;
}

// q side: [batch, n_heads*128] -> out (normed + roped). grid ceil(heads/8).
__global__ void pd_q_norm_rope_kernel(
    const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ out,
    const unsigned int* __restrict__ positions, uint32_t n_heads, float eps,
    PdRopeArgs rp, uint32_t total_heads) {
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const uint32_t idx = blockIdx.x * (blockDim.x >> 5) + warp;
    if (idx >= total_heads) return;
    const uint32_t b = idx / n_heads;
    const float4 r = pd_norm_rope_head(x + (size_t)idx * 128u, w, eps,
                                       positions[b], rp, lane);
    reinterpret_cast<float4*>(out + (size_t)idx * 128u)[lane] = r;
}

// k side: norm + rope + scatter straight into the KV cache at (slot, pos) -
// the intermediate kn buffer and the separate append pass disappear.
template<typename KV>
__global__ void pd_k_norm_rope_append_kernel(
    const float* __restrict__ x, const float* __restrict__ w, KV* __restrict__ cache,
    const unsigned int* __restrict__ positions, const unsigned int* __restrict__ slots,
    uint32_t n_kv_heads, uint32_t max_ctx, float eps, PdRopeArgs rp,
    uint32_t total_heads) {
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const uint32_t idx = blockIdx.x * (blockDim.x >> 5) + warp;
    if (idx >= total_heads) return;
    const uint32_t b = idx / n_kv_heads, h = idx % n_kv_heads;
    const float4 r = pd_norm_rope_head(x + (size_t)idx * 128u, w, eps,
                                       positions[b], rp, lane);
    const uint32_t slot = slots ? slots[b] : b;
    const uint32_t kv_dim = n_kv_heads * 128u;
    KV* row = cache + (size_t)slot * max_ctx * kv_dim +
              (size_t)positions[b] * kv_dim + (size_t)h * 128u;
    pd_kv_store(&row[4u * lane + 0u], r.x);
    pd_kv_store(&row[4u * lane + 1u], r.y);
    pd_kv_store(&row[4u * lane + 2u], r.z);
    pd_kv_store(&row[4u * lane + 3u], r.w);
}

// FlashDecoding - attention split over the KV sequence to fill the GPU at batch
// 1. At decode there's one query token and few heads, so pd_attn_decode_kernel's
// "one block per head" launches only n_heads blocks that each walk the whole KV
// range serially - the GPU sits ~97% idle and time grows with context. Here the
// KV range is sliced into n_splits chunks: grid (n_heads, n_splits), each block
// runs online softmax over its chunk and writes an UNnormalized partial. A
// second kernel (combine) merges the partials with the flash log-sum-exp rule.
//
// The sink is not applied here - it joins once, in the combine, as a virtual
// split (m=sink, l=1, o=0). Dot reduction is warp-shuffle + a tiny cross-warp
// combine; head_dim must be a multiple of 32.
__global__ void pd_attn_decode_partial_kernel(
    const float* __restrict__ q, const __half* __restrict__ kc,
    const __half* __restrict__ vc, float* __restrict__ out_o, float* __restrict__ out_ml,
    uint32_t n_heads, uint32_t n_kv_heads, uint32_t head_dim,
    uint32_t first_pos, uint32_t n_pos, uint32_t n_splits, uint32_t kv_dim, float scale)
{
    uint32_t h = blockIdx.x, s = blockIdx.y;
    uint32_t d = threadIdx.x;                 // 0..head_dim-1
    uint32_t group = n_heads / n_kv_heads;
    uint32_t kvh = h / group;
    uint32_t n_warps = head_dim >> 5;

    // this split's slice of [0, n_pos), chunk-contiguous so KV reads stay local
    uint32_t chunk = (n_pos + n_splits - 1) / n_splits;
    uint32_t lo = s * chunk;
    uint32_t hi = lo + chunk; if (hi > n_pos) hi = n_pos;

    __shared__ float red[32];                 // one slot per warp (head_dim<=1024)
    __shared__ float s_m, s_l, s_score;

    float qd = q[(size_t)h * head_dim + d];
    float acc = 0.0f;
    if (d == 0) { s_m = -INFINITY; s_l = 0.0f; }
    __syncthreads();

    for (uint32_t i = lo; i < hi; ++i) {
        size_t base = (size_t)(first_pos + i) * kv_dim + (size_t)kvh * head_dim;
        float v = qd * __half2float(kc[base + d]);
        for (uint32_t off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
        if ((d & 31u) == 0) red[d >> 5] = v;
        __syncthreads();
        if (d == 0) {
            float sc = 0.0f;
            for (uint32_t w = 0; w < n_warps; ++w) sc += red[w];
            s_score = sc * scale;
        }
        __syncthreads();
        float score = s_score, m_old = s_m;
        float m_new = fmaxf(m_old, score);
        float corr = __expf(m_old - m_new);
        float w = __expf(score - m_new);
        acc = acc * corr + w * __half2float(vc[base + d]);
        __syncthreads();
        if (d == 0) { s_l = s_l * corr + w; s_m = m_new; }
        __syncthreads();
    }
    out_o[((size_t)(h * n_splits + s)) * head_dim + d] = acc;
    if (d == 0) {
        out_ml[((size_t)(h * n_splits + s)) * 2 + 0] = s_m;
        out_ml[((size_t)(h * n_splits + s)) * 2 + 1] = s_l;
    }
}

// FlashDecoding combine: merge the n_splits partials for a head into the final
// attention output. One block per head; blockDim.x = head_dim. The per-head sink
// joins the denominator here (it holds no value). Empty splits carry m=-inf so
// exp(m-gm)=0 and drop out. n_splits is small (<=16) so the per-thread loops are
// cheap; each thread owns output dim d and reads its o partials coalesced.
__global__ void pd_attn_decode_combine_kernel(
    const float* __restrict__ in_o, const float* __restrict__ in_ml,
    const float* __restrict__ sinks, float* __restrict__ out,
    uint32_t n_heads, uint32_t head_dim, uint32_t n_splits)
{
    uint32_t h = blockIdx.x;
    uint32_t d = threadIdx.x;

    float gm = sinks[h];
    for (uint32_t s = 0; s < n_splits; ++s)
        gm = fmaxf(gm, in_ml[((size_t)(h * n_splits + s)) * 2 + 0]);

    float acc = 0.0f, l = 0.0f;
    for (uint32_t s = 0; s < n_splits; ++s) {
        float m = in_ml[((size_t)(h * n_splits + s)) * 2 + 0];
        if (m == -INFINITY) continue;  // empty split: exact +0 contribution -
                                       // skip the head_dim-float o read (and
                                       // its store may have been elided)
        float ls = in_ml[((size_t)(h * n_splits + s)) * 2 + 1];
        float sc = __expf(m - gm);
        acc += sc * in_o[((size_t)(h * n_splits + s)) * head_dim + d];
        l += sc * ls;
    }
    l += __expf(sinks[h] - gm);               // sink: denominator only
    out[(size_t)h * head_dim + d] = acc / l;
}

// Batched FlashDecoding partial: the per-sequence analog of pd_attn_decode_partial.
// grid (n_heads, batch, n_splits); block (h, b, s) runs a partial online softmax
// over sequence b's KV slice s (its own position/slot/window) and writes an
// UNnormalized partial. The sink joins later in the combine. Partial layout is
// indexed by pidx = (h*batch + b)*n_splits + s. Fills the GPU at low batch + long
// context (n_heads*batch*n_splits blocks) where the one-block-per-(head,seq)
// pd_attn_decode_batch would leave the GPU idle and grow with context.
template<typename KV>
__global__ void pd_attn_decode_batch_partial_kernel(
    const float* __restrict__ q, const KV* __restrict__ kc,
    const KV* __restrict__ vc, float* __restrict__ out_o, float* __restrict__ out_ml,
    const unsigned int* __restrict__ positions, const unsigned int* __restrict__ slots,
    uint32_t n_heads, uint32_t n_kv_heads, uint32_t head_dim, uint32_t max_ctx,
    uint32_t kv_dim, uint32_t swa_window, uint32_t n_splits, float scale) {
    uint32_t h = blockIdx.x, b = blockIdx.y, s = blockIdx.z;
    uint32_t d = threadIdx.x;
    uint32_t group = n_heads / n_kv_heads;
    uint32_t kvh = h / group;

    uint32_t slot = slots ? slots[b] : b;
    uint32_t pos = positions[b];
    uint32_t first_pos = (swa_window > 0 && pos + 1 > swa_window) ? (pos + 1 - swa_window) : 0;
    uint32_t n_pos = pos + 1 - first_pos;
    uint32_t chunk = (n_pos + n_splits - 1) / n_splits;
    uint32_t lo = s * chunk;
    uint32_t hi = lo + chunk; if (hi > n_pos) hi = n_pos;
    size_t pidx = (size_t)(h * gridDim.y + b) * n_splits + s;

    extern __shared__ float smem[];
    __shared__ float s_m, s_l;

    const float* qb = q + (size_t)b * n_heads * head_dim;
    const KV* kcb = kc + (size_t)slot * max_ctx * kv_dim;
    const KV* vcb = vc + (size_t)slot * max_ctx * kv_dim;

    if (d < head_dim) smem[d] = qb[(size_t)h * head_dim + d];
    float acc = 0.0f;
    if (d == 0) { s_m = -INFINITY; s_l = 0.0f; }
    // pd_attn_tile_walk's leading __syncthreads() orders the q stage + m/l init

    if (head_dim > 128u)
        pd_attn_tile_walk<KV, PD_ATTN_TILE_HD256>(kcb, vcb, first_pos, lo, hi, kv_dim,
                                                  kvh, head_dim, scale, smem, &s_m, &s_l, acc);
    else
        pd_attn_tile_walk<KV, PD_ATTN_TILE>(kcb, vcb, first_pos, lo, hi, kv_dim, kvh,
                                            head_dim, scale, smem, &s_m, &s_l, acc);
    __syncthreads();
    if (d < head_dim) out_o[pidx * head_dim + d] = acc;
    if (d == 0) {
        out_ml[pidx * 2 + 0] = s_m;
        out_ml[pidx * 2 + 1] = s_l;
    }
}

// Paged twin of pd_attn_decode_batch_partial_kernel (P3b): same split online-
// softmax, but K/V come from the block pool via each slot's block table (walked
// by pd_attn_tile_walk_paged). Bit-exact vs the dense partial - only the per-
// token base differs. Needed on ≥128-SM dies where attn_splits engages. This is
// the plain (non-GQA-fused) variant; a paged GQA-fused partial for perf parity
// is P3b-2.
// TQ (attention streams): f16 q plane; partials stay f32.
template<typename KV, typename TQ = float>
__global__ void pd_attn_decode_batch_partial_paged_kernel(
    const TQ* __restrict__ q, const KV* __restrict__ pool_k,
    const KV* __restrict__ pool_v, float* __restrict__ out_o, float* __restrict__ out_ml,
    const unsigned int* __restrict__ positions, const unsigned int* __restrict__ slots,
    const uint32_t* __restrict__ block_tables, uint32_t blocks_per_slot,
    uint32_t n_heads, uint32_t n_kv_heads, uint32_t head_dim,
    uint32_t kv_dim, uint32_t swa_window, uint32_t n_splits, float scale) {
    // cascade (laguna chain): q + the pool's newest row
    // come from the predecessor GEMV/append - same arming as the GQA twin
    PD_PDL_ARM();
    uint32_t h = blockIdx.x, b = blockIdx.y, s = blockIdx.z;
    uint32_t d = threadIdx.x;
    uint32_t group = n_heads / n_kv_heads;
    uint32_t kvh = h / group;

    uint32_t slot = slots ? slots[b] : b;
    uint32_t pos = positions[b];
    uint32_t first_pos = (swa_window > 0 && pos + 1 > swa_window) ? (pos + 1 - swa_window) : 0;
    uint32_t n_pos = pos + 1 - first_pos;
    uint32_t chunk = (n_pos + n_splits - 1) / n_splits;
    uint32_t lo = s * chunk;
    uint32_t hi = lo + chunk; if (hi > n_pos) hi = n_pos;
    size_t pidx = (size_t)(h * gridDim.y + b) * n_splits + s;

    extern __shared__ float smem[];
    __shared__ float s_m, s_l;

    const TQ* qb = q + (size_t)b * n_heads * head_dim;
    const uint32_t* bt = block_tables + (size_t)slot * blocks_per_slot;

    if (d < head_dim) smem[d] = (float)qb[(size_t)h * head_dim + d];
    float acc = 0.0f;
    if (d == 0) { s_m = -INFINITY; s_l = 0.0f; }

    if (head_dim > 128u)
        pd_attn_tile_walk_paged<KV, PD_ATTN_TILE_HD256>(pool_k, pool_v, bt, first_pos, lo, hi, kv_dim,
                                                        kvh, head_dim, scale, smem, &s_m, &s_l, acc);
    else
        pd_attn_tile_walk_paged<KV, PD_ATTN_TILE>(pool_k, pool_v, bt, first_pos, lo, hi, kv_dim, kvh,
                                                  head_dim, scale, smem, &s_m, &s_l, acc);
    __syncthreads();
    if (d < head_dim) out_o[pidx * head_dim + d] = acc;
    if (d == 0) {
        out_ml[pidx * 2 + 0] = s_m;
        out_ml[pidx * 2 + 1] = s_l;
    }
}


// GQA-fused batched FlashDecoding partial: one block per (KV head, seq,
// split) stages each K/V tile once and serves every q-head of the group.
// The per-q-head partial walk re-reads KV group_size x (4x/6x/8x on the
// qwen models) - at hd=256 depth decode that was the traffic gap vs llama's
// flat tg-at-depth (partial walk measured 240-290 GB/s of useful bytes).
// Per-head numerics replicate pd_attn_tile_walk exactly: same tile size,
// same thread-serial dot in dim order, same per-tile m/l fold sequence and
// summation orders - partials are bit-identical to the per-q-head kernel,
// land in the same pidx layout, and the combine is unchanged.
// Raw-KV per-element converts - the same scalar conversions pd_kv_load4
// does, so folding at the read keeps every result bit-identical to the
// f32-staged walk.
__device__ __forceinline__ float pd_kv_to_f32(__half v) { return __half2float(v); }
__device__ __forceinline__ float pd_kv_to_f32(__nv_fp8_e4m3 v) { return float(v); }
// Paired variant (dd, dd+1) for the vectorized walks: one 4 B/2 B shared
// load instead of two, per-element conversions unchanged. `p` must be
// 2-element aligned (even dd on an even row stride).
__device__ __forceinline__ float2 pd_kv_to_f32x2(const __half* p) {
    return __half22float2(*reinterpret_cast<const __half2*>(p));
}
__device__ __forceinline__ float2 pd_kv_to_f32x2(const __nv_fp8_e4m3* p) {
    return make_float2(float(p[0]), float(p[1]));
}

// Self-contained cp.async helpers for the attention region (the mmq_pipe
// ones live further down the file). Pre-sm_80 falls back to a synchronous
// 16B copy with no-op group markers.
__device__ __forceinline__ void pd_attn_cpa16(void* smem, const void* gmem) {
#if __CUDA_ARCH__ >= 800
    const unsigned sm = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(sm), "l"(gmem));
#else
    *reinterpret_cast<float4*>(smem) = *reinterpret_cast<const float4*>(gmem);
#endif
}
__device__ __forceinline__ void pd_attn_cpa_commit() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;");
#endif
}
__device__ __forceinline__ void pd_attn_cpa_wait1() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 1;");
#endif
}
__device__ __forceinline__ void pd_attn_cpa_wait0() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 0;");
#endif
}
// KVS wait levels: with K and V as separate groups the
// K-wait under a DBK prefetch allows 3 pending (V, K', V') and the
// deferred V-wait allows 2 (K', V')
__device__ __forceinline__ void pd_attn_cpa_wait2() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 2;");
#endif
}
__device__ __forceinline__ void pd_attn_cpa_wait3() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 3;");
#endif
}

#define PD_GQA_MAX_GROUP 8u
// f32 q/scores/weights + DOUBLE-BUFFERED raw K/V tiles (16-byte row pad;
// sized for the 2-byte f16 worst case - fp8 uses less than allocated).
// q rows stride hd+4 and score/weight rows stride T+1: the unpadded
// [G][hd] / [G][T] layouts put every same-dd q element (hd = 64/128/256,
// all bank-stride multiples) and every same-p score in one bank - an 8-way
// conflict on every dot load/score store that made the walk 2x slower
// (373 -> 189 us on the B=32 depth-1100 bisect harness). The +4 (not +1)
// keeps q rows 16-byte ALIGNED as well, so the dot reads float4 spans -
// the walk is LDS-issue-bound and 4-wide loads quarter the instruction
// count.
// The f32 planes ahead of the K/V staging area, in FLOATS, rounded up to a
// multiple of 4 so the staging base lands on a 16-byte boundary. That padding
// is not cosmetic: the stage does cp.async in 16-byte lines, which FAULTS
// (CUDA_ERROR_MISALIGNED_ADDRESS) on an 8-byte base.
//
// The raw count is G*(hd+4) + 2*G*(T+1) floats. hd is a multiple of 4 on every
// head we serve, so the first term is always 4-aligned; the second is 2*G*(T+1)
// with T in {16, 32}, i.e. 2G*odd - 4-aligned only when G is EVEN. Every shape
// served before granite-vision-4.1-4b had an even GQA group (4, 6, 8), so the
// base happened to be aligned and nothing ever noticed. That model is 40 q / 8
// kv = group 5, and it faulted on the first decode.
#define PD_GQA_FPLANES(G, hd, T) \
    ((((G) * ((hd) + 4u) + 2u * (G) * ((T) + 1u)) + 3u) & ~3u)
#define PD_GQA_SMEM(G, hd, T) \
    ((PD_GQA_FPLANES(G, hd, T) * sizeof(float)) + 4u * (T) * ((hd) * 2u + 16u))
template<typename KV, uint32_t TILE, bool PVM = false, bool SCM = true>
__global__ void pd_attn_decode_batch_partial_gqa_kernel(
    const float* __restrict__ q, const KV* __restrict__ kc,
    const KV* __restrict__ vc, float* __restrict__ out_o, float* __restrict__ out_ml,
    const unsigned int* __restrict__ positions, const unsigned int* __restrict__ slots,
    uint32_t n_heads, uint32_t n_kv_heads, uint32_t head_dim, uint32_t max_ctx,
    uint32_t kv_dim, uint32_t swa_window, uint32_t n_splits, float scale) {
    const uint32_t kvh = blockIdx.x, b = blockIdx.y, s = blockIdx.z;
    const uint32_t d = threadIdx.x, nth = blockDim.x;
    const uint32_t G = n_heads / n_kv_heads;

    const uint32_t slot = slots ? slots[b] : b;
    const uint32_t pos = positions[b];
    const uint32_t first_pos =
        (swa_window > 0 && pos + 1 > swa_window) ? (pos + 1 - swa_window) : 0;
    const uint32_t n_pos = pos + 1 - first_pos;
    // Adaptive effective splits (GB202 wave profile): the grid's
    // n_splits is graph-baked for max_ctx, but at short kv (~300 tokens) a
    // 32-way split hands each CTA ~1 tile and the per-CTA prelude (q stage,
    // table walk, barriers) dominates - 66 us vs flash-splitkv's 26. Target
    // >= 4 tiles per live CTA; surplus CTAs see lo >= hi, skip staging, and
    // write the (-inf, 0) empty partial the combine already folds (the same
    // path short sequences exercised at fixed splits). Split-boundary regroup
    // = the existing reorder class (splits already vary by env/die).
    uint32_t s_eff = (n_pos + 4u * TILE - 1u) / (4u * TILE);
    if (s_eff > n_splits) s_eff = n_splits;
    if (s_eff < 1u) s_eff = 1u;
    const uint32_t chunk = (n_pos + s_eff - 1u) / s_eff;
    const uint32_t lo = s * chunk;
    uint32_t hi = lo + chunk;
    if (hi > n_pos) hi = n_pos;

    // cp.async double-buffered RAW-KV walk: tile t+1's K/V bytes stream into
    // the alternate buffer while tile t computes, so the stage's global
    // latency never serializes with the math (the f32-staged walk exposed a
    // full DRAM round-trip per tile and ran at ~11-23% of bandwidth). KV
    // bytes convert to f32 at the READ - same scalar conversion as
    // pd_kv_load4, so every fold is bit-identical to the staged walk.
    extern __shared__ __align__(16) unsigned char gqa_smraw[];
    const uint32_t q_s = head_dim + 4u;                 //  padded strides (see
    const uint32_t t_s = TILE + 1u;                     //  PD_GQA_SMEM note)
    float* s_q = (float*)gqa_smraw;                     //  [G][q_s] (unscaled)
    float* s_sc = s_q + (size_t)G * q_s;                //  [G][t_s]
    float* s_w = s_sc + (size_t)G * t_s;                //  [G][t_s]
    // 16B-aligned base (PD_GQA_FPLANES): cp.async stages 16-byte lines
    KV* s_kv = (KV*)((float*)gqa_smraw + PD_GQA_FPLANES(G, head_dim, TILE));
    const uint32_t row_e = head_dim + 16u / (uint32_t)sizeof(KV);  // 16 B row pad
    __shared__ float s_m[PD_GQA_MAX_GROUP], s_l[PD_GQA_MAX_GROUP];

    const float* qb = q + (size_t)b * n_heads * head_dim;
    const KV* kcb = kc + (size_t)slot * max_ctx * kv_dim;
    const KV* vcb = vc + (size_t)slot * max_ctx * kv_dim;

    if (lo < hi)
        for (uint32_t idx = d; idx < G * head_dim; idx += nth)
            s_q[(idx / head_dim) * q_s + idx % head_dim] =
                qb[((size_t)kvh * G + idx / head_dim) * head_dim + idx % head_dim];
    if (d < G) { s_m[d] = -INFINITY; s_l[d] = 0.0f; }
    // mma score path (serving head_dims): K is f16 -> exact in tf32; Q splits
    // big+small (3xTF32) so the pair of MMAs stays in the scalar path's ~1e-6
    // numeric class. hd128 (laguna/qwen3 shape) is served at TILE 32 -
    // the m16n8k8 map covers a 16-token slab, so warps 0..TILE/16-1 each own
    // one slab in parallel (TILE-16-at-hd128 was measured slower: doubling
    // stage/barrier rounds beat the score savings, full layers 49->70 µs).
    // hd256 keeps its exact original gate (TILE 16 only - gemma4 unperturbed).
    // SCM=false is the launcher's PADDOCK_NO_ATTN_MMA128 escape instantiation.
    const bool sc_mma = SCM && (sizeof(KV) == 2u)
        && ((head_dim == 256u && TILE == 16u)
            || (head_dim == 128u && (TILE == 16u || TILE == 32u)));
    // tensor-core PV path: sc_mma shapes whose fragment map exactly covers the
    // head - hd256/T16 (warp w owns dims [w*32, w*32+32) as two 16-dim
    // M-tiles) and hd128/T32 (one M-tile each - 8 warps span exactly 128
    // dims), both at an exact 8-warp block. The explicit TILE<->hd
    // bijection here (not sc_mma's looser gate - hd256/T32 and hd128/T16
    // shapes exist) is what lets the PV arm derive MTN and its chunk count
    // from TILE alone at compile time. Twin of the paged kernel's arm.
    const bool pv_on = PVM && sc_mma && nth == 256u
        && ((head_dim == 256u && TILE == 16u) || (head_dim == 128u && TILE == 32u));

    // Per-thread accumulator slots over (head, dim) pairs: e = h*head_dim+dd,
    // thread t owns e = t, t+nth, ... - Every thread carries V-accumulation
    // work (the old d<head_dim layout left 3/4 of a 256-thread block idle and
    // serialized G*TILE FMAs on the rest; at (G=8, hd=64) the walk ran at
    // ~9% of DRAM bandwidth, latency-bound on exactly that chain). Each
    // (h, dd) element still folds its tile weights in ascending-p order, so
    // results stay bit-identical to pd_attn_tile_walk.
    float acc[PD_GQA_MAX_GROUP];
    #pragma unroll
    for (uint32_t h = 0; h < PD_GQA_MAX_GROUP; ++h) acc[h] = 0.0f;
    __shared__ float s_mnew[PD_GQA_MAX_GROUP], s_corr[PD_GQA_MAX_GROUP];

    // stage tile [t0, t0+n_t) raw K+V rows into buffer `bf`, one commit group
    const uint32_t lines = (head_dim * (uint32_t)sizeof(KV)) >> 4;  // 16B/row
    auto stage = [&](uint32_t bf, uint32_t t0) {
        const uint32_t n_t = hi - t0 < TILE ? hi - t0 : TILE;
        for (uint32_t i = d; i < 2u * n_t * lines; i += nth) {
            const uint32_t kvsel = i / (n_t * lines);
            const uint32_t j = i - kvsel * n_t * lines;
            const uint32_t p = j / lines, l = j - p * lines;
            const KV* src = (kvsel ? vcb : kcb)
                + (size_t)(first_pos + t0 + p) * kv_dim + (size_t)kvh * head_dim;
            KV* dst = s_kv + ((size_t)(bf * 2u + kvsel) * TILE + p) * row_e;
            pd_attn_cpa16((char*)dst + l * 16u, (const char*)src + l * 16u);
        }
        pd_attn_cpa_commit();
    };

    if (lo < hi) stage(0u, lo);
    uint32_t bf = 0;
    for (uint32_t t0 = lo; t0 < hi; t0 += TILE, bf ^= 1u) {
        const uint32_t n_t = hi - t0 < TILE ? hi - t0 : TILE;
        const bool more = t0 + TILE < hi;
        if (more) stage(bf ^ 1u, t0 + TILE);  // next tile streams while we compute
        if (more) pd_attn_cpa_wait1(); else pd_attn_cpa_wait0();
        __syncthreads();  // this tile's bytes visible block-wide
        const KV* kbuf = s_kv + (size_t)(bf * 2u) * TILE * row_e;
        const KV* vbuf = s_kv + ((size_t)(bf * 2u) + 1u) * TILE * row_e;
        // one thread per (position, head): the walk's dim-order dot, K and q
        // read as 16-byte spans (the walk is LDS-ISSUE-bound, not latency-
        // bound: scalar reads were 96 shared-load instructions per thread
        // per tile; this is 24). The FMA sequence per (p, h) is unchanged -
        // BIT-IDENTICAL to pd_attn_tile_walk. (A 4-way split of the CHAIN
        // was tried and measured slower; widening the loads keeps the chain.)
        if (sc_mma) {
            // tensor-core scores: warps 0..TILE/16-1 each compute one
            // (16-token x <=8-head) slab via m16n8k8 tf32 pairs (Q = big +
            // small; K f16 exact in tf32). TILE 16 = the original single-warp
            // shape; TILE 32 (hd128) runs two slabs in parallel. Stale
            // rows/cols land in s_sc slots no consumer reads (softmax reduce
            // masks lane < n_t; exp writes < n_t*G).
            const uint32_t warp_ = d >> 5, lane_ = d & 31u;
            if (warp_ < (TILE >> 4)) {
                const uint32_t pb = warp_ * 16u;  // slab position base
                const uint32_t g8m = lane_ >> 2, t4m = lane_ & 3u;
                float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
                const __half* kb16 = (const __half*)kbuf;
                auto tf = [](float v) { uint32_t r; asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(v)); return r; };
                for (uint32_t kk = 0; kk < head_dim; kk += 8u) {
                    const uint32_t a0 = tf(__half2float(kb16[(size_t)(pb + g8m) * row_e + kk + t4m]));
                    const uint32_t a1 = tf(__half2float(kb16[(size_t)(pb + g8m + 8u) * row_e + kk + t4m]));
                    const uint32_t a2 = tf(__half2float(kb16[(size_t)(pb + g8m) * row_e + kk + 4u + t4m]));
                    const uint32_t a3 = tf(__half2float(kb16[(size_t)(pb + g8m + 8u) * row_e + kk + 4u + t4m]));
                    const float q0f = g8m < G ? s_q[g8m * q_s + kk + t4m] : 0.0f;
                    const float q1f = g8m < G ? s_q[g8m * q_s + kk + 4u + t4m] : 0.0f;
                    const uint32_t b0 = tf(q0f), b1 = tf(q1f);
                    const uint32_t b0s = tf(q0f - __uint_as_float(b0));
                    const uint32_t b1s = tf(q1f - __uint_as_float(b1));
                    asm("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
                    asm("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0s), "r"(b1s));
                }
                s_sc[(2u * t4m) * t_s + pb + g8m] = d0 * scale;
                s_sc[(2u * t4m + 1u) * t_s + pb + g8m] = d1 * scale;
                s_sc[(2u * t4m) * t_s + pb + g8m + 8u] = d2 * scale;
                s_sc[(2u * t4m + 1u) * t_s + pb + g8m + 8u] = d3 * scale;
            }
        } else
        for (uint32_t idx = d; idx < n_t * G; idx += nth) {
            const uint32_t p = idx / G, h = idx % G;
            float sc = 0.0f;
            const float* qrow = s_q + h * q_s;
            const KV* krow = kbuf + (size_t)p * row_e;
            if (sizeof(KV) == 2u) {
                // f16: one uint4 = 8 K elements; two float4 = 8 q elements
                for (uint32_t dd = 0; dd < head_dim; dd += 8) {
                    const uint4 kr = *(const uint4*)(krow + dd);
                    const float4 q0 = *(const float4*)(qrow + dd);
                    const float4 q1 = *(const float4*)(qrow + dd + 4);
                    const __half2* kh = (const __half2*)&kr;
                    float2 k0 = __half22float2(kh[0]), k1 = __half22float2(kh[1]);
                    float2 k2 = __half22float2(kh[2]), k3 = __half22float2(kh[3]);
                    sc += q0.x * k0.x;
                    sc += q0.y * k0.y;
                    sc += q0.z * k1.x;
                    sc += q0.w * k1.y;
                    sc += q1.x * k2.x;
                    sc += q1.y * k2.y;
                    sc += q1.z * k3.x;
                    sc += q1.w * k3.y;
                }
            } else {
                // fp8: 8-wide (one uint2 smem read = 8 e4m3), mirroring the
                // f16 uint4 branch - the 2-elem loop halved score throughput
                for (uint32_t dd = 0; dd < head_dim; dd += 8) {
                    const uint2 kr8 = *(const uint2*)(krow + dd);
                    const __nv_fp8_e4m3* kb = (const __nv_fp8_e4m3*)&kr8;
                    #pragma unroll
                    for (uint32_t j = 0; j < 8u; ++j)
                        sc += qrow[dd + j] * (float)kb[j];
                }
            }
            s_sc[h * t_s + p] = sc * scale;
        }
        __syncthreads();
        // per-head m fold as a warp-shuffle max: warp w owns head w (launch
        // guarantees nth/32 >= G). fmax is exact under any reduction order,
        // so the fold matches the walk's serial max bit-for-bit; corr/m land
        // in shared for every thread to read (one exp per head total).
        {
            const uint32_t warp = d >> 5, lane = d & 31u;
            if (warp < G) {
                float v = (lane < n_t) ? s_sc[warp * t_s + lane] : -INFINITY;
                #pragma unroll
                for (uint32_t off = 16; off > 0; off >>= 1)
                    v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, off));
                if (lane == 0) {
                    const float m_new = fmaxf(s_m[warp], v);
                    s_mnew[warp] = m_new;
                    s_corr[warp] = __expf(s_m[warp] - m_new);
                }
            }
        }
        __syncthreads();
        for (uint32_t idx = d; idx < n_t * G; idx += nth) {
            const uint32_t p = idx / G, h = idx % G;
            s_w[h * t_s + p] = __expf(s_sc[h * t_s + p] - s_mnew[h]);
        }
        __syncthreads();
        if (pv_on) {
            // tensor-core PV - byte-for-byte the paged kernel's arm (see its
            // comment for the fragment map and the zero-both-operands rule).
            // MTN M-tiles per warp (hd256/T16 = 2, hd128/T32 = 1) and TILE>>3
            // 8-token chunks - compile-time via the gate's TILE<->hd bijection.
            constexpr uint32_t MTN = TILE == 16u ? 2u : 1u;
            const uint32_t warp_ = d >> 5, lane_ = d & 31u;
            const uint32_t g8 = lane_ >> 2, t4 = lane_ & 3u;
            const uint32_t h0 = 2u * t4, h1 = h0 + 1u;
            const float c0 = h0 < G ? s_corr[h0] : 0.0f;
            const float c1 = h1 < G ? s_corr[h1] : 0.0f;
            #pragma unroll
            for (uint32_t mt = 0; mt < MTN; ++mt) {
                acc[mt * 4u + 0] *= c0; acc[mt * 4u + 1] *= c1;
                acc[mt * 4u + 2] *= c0; acc[mt * 4u + 3] *= c1;
            }
            auto tf = [](float v) { uint32_t r; asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(v)); return r; };
            const __half* vb16 = (const __half*)vbuf;
            #pragma unroll
            for (uint32_t kc_ = 0; kc_ < (TILE >> 3); ++kc_) {
                const uint32_t p0 = kc_ * 8u + t4, p1 = p0 + 4u;
                const float w0 = (p0 < n_t && g8 < G) ? s_w[g8 * t_s + p0] : 0.0f;
                const float w1 = (p1 < n_t && g8 < G) ? s_w[g8 * t_s + p1] : 0.0f;
                const uint32_t b0 = tf(w0), b1 = tf(w1);
                const float r0 = w0 - __uint_as_float(b0);
                const float r1 = w1 - __uint_as_float(b1);
                const uint32_t b0s = tf(r0), b1s = tf(r1);
                // third split term: big+mid+small carries P at ~33 bits (>
                // f32's 24), so the PV products round no worse than the
                // scalar f32 fold - two terms measured +0.5% PPL via chaos
                // amplification over long teacher-forced runs.
                const uint32_t b0t = tf(r0 - __uint_as_float(b0s));
                const uint32_t b1t = tf(r1 - __uint_as_float(b1s));
                #pragma unroll
                for (uint32_t mt = 0; mt < MTN; ++mt) {
                    const uint32_t db = warp_ * (16u * MTN) + mt * 16u + g8;
                    const uint32_t a0 = p0 < n_t ? tf(__half2float(vb16[(size_t)p0 * row_e + db])) : 0u;
                    const uint32_t a1 = p0 < n_t ? tf(__half2float(vb16[(size_t)p0 * row_e + db + 8u])) : 0u;
                    const uint32_t a2 = p1 < n_t ? tf(__half2float(vb16[(size_t)p1 * row_e + db])) : 0u;
                    const uint32_t a3 = p1 < n_t ? tf(__half2float(vb16[(size_t)p1 * row_e + db + 8u])) : 0u;
                    asm("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                        : "+f"(acc[mt * 4u + 0]), "+f"(acc[mt * 4u + 1]),
                          "+f"(acc[mt * 4u + 2]), "+f"(acc[mt * 4u + 3])
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
                    asm("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                        : "+f"(acc[mt * 4u + 0]), "+f"(acc[mt * 4u + 1]),
                          "+f"(acc[mt * 4u + 2]), "+f"(acc[mt * 4u + 3])
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0s), "r"(b1s));
                    asm("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                        : "+f"(acc[mt * 4u + 0]), "+f"(acc[mt * 4u + 1]),
                          "+f"(acc[mt * 4u + 2]), "+f"(acc[mt * 4u + 3])
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0t), "r"(b1t));
                }
            }
        } else
        // V accumulate over (head, dim-PAIR) slots: nth threads cover
        // G*head_dim/2 pairs, one 2-element V load per p per pair; each
        // element's p-order fold is still the walk's exact sequence
        for (uint32_t e2 = d, j = 0; e2 < G * (head_dim >> 1); e2 += nth, j += 2) {
            const uint32_t hd2 = head_dim >> 1;
            const uint32_t h = e2 / hd2, dd = (e2 - h * hd2) << 1;
            float a0 = acc[j] * s_corr[h];
            float a1 = acc[j + 1] * s_corr[h];
            const float* wrow = s_w + h * t_s;
            const KV* vrow = vbuf + dd;
            for (uint32_t p = 0; p < n_t; ++p) {
                const float2 vv = pd_kv_to_f32x2(vrow + (size_t)p * row_e);
                const float wp = wrow[p];
                a0 += wp * vv.x;
                a1 += wp * vv.y;
            }
            acc[j] = a0;
            acc[j + 1] = a1;
        }
        if (d < G) {
            float ws = 0.0f;
            for (uint32_t p = 0; p < n_t; ++p) ws += s_w[d * t_s + p];
            s_l[d] = s_l[d] * s_corr[d] + ws;
            s_m[d] = s_mnew[d];
        }
        // all reads of this buffer complete before the next iteration's
        // stage() streams tile t+2 into it
        __syncthreads();
    }
    __syncthreads();
    if (pv_on) {
        // fragment-layout writeback - twin of the paged kernel's.
        constexpr uint32_t MTN = TILE == 16u ? 2u : 1u;
        const uint32_t warp_ = d >> 5, lane_ = d & 31u;
        const uint32_t g8 = lane_ >> 2, t4 = lane_ & 3u;
        const uint32_t h0 = 2u * t4, h1 = h0 + 1u;
        #pragma unroll
        for (uint32_t mt = 0; mt < MTN; ++mt) {
            const uint32_t db = warp_ * (16u * MTN) + mt * 16u + g8;
            if (h0 < G) {
                const size_t pidx = ((size_t)(kvh * G + h0) * gridDim.y + b) * n_splits + s;
                out_o[pidx * head_dim + db] = acc[mt * 4u + 0];
                out_o[pidx * head_dim + db + 8u] = acc[mt * 4u + 2];
            }
            if (h1 < G) {
                const size_t pidx = ((size_t)(kvh * G + h1) * gridDim.y + b) * n_splits + s;
                out_o[pidx * head_dim + db] = acc[mt * 4u + 1];
                out_o[pidx * head_dim + db + 8u] = acc[mt * 4u + 3];
            }
        }
    } else
    for (uint32_t e2 = d, j = 0; e2 < G * (head_dim >> 1); e2 += nth, j += 2) {
        const uint32_t hd2 = head_dim >> 1;
        const uint32_t h = e2 / hd2, dd = (e2 - h * hd2) << 1;
        const size_t pidx = ((size_t)(kvh * G + h) * gridDim.y + b) * n_splits + s;
        out_o[pidx * head_dim + dd] = acc[j];
        out_o[pidx * head_dim + dd + 1] = acc[j + 1];
    }
    if (d < G) {
        const size_t pidx = ((size_t)(kvh * G + d) * gridDim.y + b) * n_splits + s;
        out_ml[pidx * 2 + 0] = s_m[d];
        out_ml[pidx * 2 + 1] = s_l[d];
    }
}

// Paged twin of pd_attn_decode_batch_partial_gqa_kernel (P3b-2): identical
// GQA-fused cp.async double-buffered walk, but K/V come from the block pool via
// the slot's block table. The only changes are the base setup and the per-token
// `src` in stage() (dense slot base -> block-table lookup); the cp.async, the
// dot/fold, and every syncthreads are byte-identical, so partials are bit-exact
// vs the dense GQA-fused kernel (which is itself bit-identical to the plain walk).
template<typename KV, uint32_t TILE, bool PVM = false, bool SCM = true>
__global__ void pd_attn_decode_batch_partial_gqa_paged_kernel(
    const float* __restrict__ q, const KV* __restrict__ pool_k,
    const KV* __restrict__ pool_v, float* __restrict__ out_o, float* __restrict__ out_ml,
    const unsigned int* __restrict__ positions, const unsigned int* __restrict__ slots,
    const uint32_t* __restrict__ block_tables, uint32_t blocks_per_slot,
    uint32_t n_heads, uint32_t n_kv_heads, uint32_t head_dim,
    uint32_t kv_dim, uint32_t swa_window, uint32_t n_splits, float scale) {
    PD_PDL_ARM();  // cascade (granite chain)
    const uint32_t kvh = blockIdx.x, b = blockIdx.y, s = blockIdx.z;
    const uint32_t d = threadIdx.x, nth = blockDim.x;
    const uint32_t G = n_heads / n_kv_heads;

    const uint32_t slot = slots ? slots[b] : b;
    const uint32_t pos = positions[b];
    const uint32_t first_pos =
        (swa_window > 0 && pos + 1 > swa_window) ? (pos + 1 - swa_window) : 0;
    const uint32_t n_pos = pos + 1 - first_pos;
    // Adaptive effective splits (GB202 wave profile): the grid's
    // n_splits is graph-baked for max_ctx, but at short kv (~300 tokens) a
    // 32-way split hands each CTA ~1 tile and the per-CTA prelude (q stage,
    // table walk, barriers) dominates - 66 us vs flash-splitkv's 26. Target
    // >= 4 tiles per live CTA; surplus CTAs see lo >= hi, skip staging, and
    // write the (-inf, 0) empty partial the combine already folds (the same
    // path short sequences exercised at fixed splits). Split-boundary regroup
    // = the existing reorder class (splits already vary by env/die).
    uint32_t s_eff = (n_pos + 4u * TILE - 1u) / (4u * TILE);
    if (s_eff > n_splits) s_eff = n_splits;
    if (s_eff < 1u) s_eff = 1u;
    const uint32_t chunk = (n_pos + s_eff - 1u) / s_eff;
    const uint32_t lo = s * chunk;
    uint32_t hi = lo + chunk;
    if (hi > n_pos) hi = n_pos;

    extern __shared__ __align__(16) unsigned char gqa_smraw[];
    const uint32_t q_s = head_dim + 4u;
    const uint32_t t_s = TILE + 1u;
    float* s_q = (float*)gqa_smraw;
    float* s_sc = s_q + (size_t)G * q_s;
    float* s_w = s_sc + (size_t)G * t_s;
    // 16B-aligned base (PD_GQA_FPLANES): cp.async stages 16-byte lines
    KV* s_kv = (KV*)((float*)gqa_smraw + PD_GQA_FPLANES(G, head_dim, TILE));
    const uint32_t row_e = head_dim + 16u / (uint32_t)sizeof(KV);
    __shared__ float s_m[PD_GQA_MAX_GROUP], s_l[PD_GQA_MAX_GROUP];

    const float* qb = q + (size_t)b * n_heads * head_dim;
    // paged: the slot's block table replaces the dense slot base.
    const uint32_t* bt = block_tables + (size_t)slot * blocks_per_slot;

    if (lo < hi)
        for (uint32_t idx = d; idx < G * head_dim; idx += nth)
            s_q[(idx / head_dim) * q_s + idx % head_dim] =
                qb[((size_t)kvh * G + idx / head_dim) * head_dim + idx % head_dim];
    if (d < G) { s_m[d] = -INFINITY; s_l[d] = 0.0f; }
    // mma score path (serving head_dims): K is f16 -> exact in tf32; Q splits
    // big+small (3xTF32) so the pair of MMAs stays in the scalar path's ~1e-6
    // numeric class. hd128 (laguna/qwen3 shape) is served at TILE 32
    // via per-warp 16-token slabs (TILE-16-at-hd128 measured slower: doubled
    // stage/barrier rounds beat the score savings). hd256 keeps its exact
    // original gate. SCM=false = the PADDOCK_NO_ATTN_MMA128 escape.
    const bool sc_mma = SCM && (sizeof(KV) == 2u)
        && ((head_dim == 256u && TILE == 16u)
            || (head_dim == 128u && (TILE == 16u || TILE == 32u)));
    // tensor-core PV path (opt-in launcher instantiation): sc_mma shapes whose
    // fragment map exactly covers the head - hd256/T16 (warp w owns dims
    // [w*32, w*32+32) as two 16-dim M-tiles) and hd128/T32 (one M-tile each -
    // 8 warps span exactly 128 dims), both at an exact
    // 8-warp block. The explicit TILE<->hd bijection (not sc_mma's looser
    // gate - hd256/T32 and hd128/T16 shapes exist) lets the PV arm derive MTN
    // and its chunk count from TILE alone at compile time.
    const bool pv_on = PVM && sc_mma && nth == 256u
        && ((head_dim == 256u && TILE == 16u) || (head_dim == 128u && TILE == 32u));

    float acc[PD_GQA_MAX_GROUP];
    #pragma unroll
    for (uint32_t h = 0; h < PD_GQA_MAX_GROUP; ++h) acc[h] = 0.0f;
    __shared__ float s_mnew[PD_GQA_MAX_GROUP], s_corr[PD_GQA_MAX_GROUP];

    const uint32_t lines = (head_dim * (uint32_t)sizeof(KV)) >> 4;
    auto stage = [&](uint32_t bf, uint32_t t0) {
        const uint32_t n_t = hi - t0 < TILE ? hi - t0 : TILE;
        for (uint32_t i = d; i < 2u * n_t * lines; i += nth) {
            const uint32_t kvsel = i / (n_t * lines);
            const uint32_t j = i - kvsel * n_t * lines;
            const uint32_t p = j / lines, l = j - p * lines;
            // paged per-token base: resolve the physical block for this position.
            const uint32_t gpos = first_pos + t0 + p;
            const uint32_t blk = bt[gpos >> 4];
            const KV* src = (kvsel ? pool_v : pool_k)
                + (size_t)blk * 16u * kv_dim + (size_t)(gpos & 15u) * kv_dim
                + (size_t)kvh * head_dim;
            KV* dst = s_kv + ((size_t)(bf * 2u + kvsel) * TILE + p) * row_e;
            pd_attn_cpa16((char*)dst + l * 16u, (const char*)src + l * 16u);
        }
        pd_attn_cpa_commit();
    };

    if (lo < hi) stage(0u, lo);
    uint32_t bf = 0;
    for (uint32_t t0 = lo; t0 < hi; t0 += TILE, bf ^= 1u) {
        const uint32_t n_t = hi - t0 < TILE ? hi - t0 : TILE;
        const bool more = t0 + TILE < hi;
        if (more) stage(bf ^ 1u, t0 + TILE);
        if (more) pd_attn_cpa_wait1(); else pd_attn_cpa_wait0();
        __syncthreads();
        const KV* kbuf = s_kv + (size_t)(bf * 2u) * TILE * row_e;
        const KV* vbuf = s_kv + ((size_t)(bf * 2u) + 1u) * TILE * row_e;
        if (sc_mma) {
            // tensor-core scores: warps 0..TILE/16-1 each compute one
            // (16-token x <=8-head) slab via m16n8k8 tf32 pairs (Q = big +
            // small; K f16 exact in tf32). TILE 16 = the original single-warp
            // shape; TILE 32 (hd128) runs two slabs in parallel. Stale
            // rows/cols land in s_sc slots no consumer reads (softmax reduce
            // masks lane < n_t; exp writes < n_t*G).
            const uint32_t warp_ = d >> 5, lane_ = d & 31u;
            if (warp_ < (TILE >> 4)) {
                const uint32_t pb = warp_ * 16u;  // slab position base
                const uint32_t g8m = lane_ >> 2, t4m = lane_ & 3u;
                float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
                const __half* kb16 = (const __half*)kbuf;
                auto tf = [](float v) { uint32_t r; asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(v)); return r; };
                for (uint32_t kk = 0; kk < head_dim; kk += 8u) {
                    const uint32_t a0 = tf(__half2float(kb16[(size_t)(pb + g8m) * row_e + kk + t4m]));
                    const uint32_t a1 = tf(__half2float(kb16[(size_t)(pb + g8m + 8u) * row_e + kk + t4m]));
                    const uint32_t a2 = tf(__half2float(kb16[(size_t)(pb + g8m) * row_e + kk + 4u + t4m]));
                    const uint32_t a3 = tf(__half2float(kb16[(size_t)(pb + g8m + 8u) * row_e + kk + 4u + t4m]));
                    const float q0f = g8m < G ? s_q[g8m * q_s + kk + t4m] : 0.0f;
                    const float q1f = g8m < G ? s_q[g8m * q_s + kk + 4u + t4m] : 0.0f;
                    const uint32_t b0 = tf(q0f), b1 = tf(q1f);
                    const uint32_t b0s = tf(q0f - __uint_as_float(b0));
                    const uint32_t b1s = tf(q1f - __uint_as_float(b1));
                    asm("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
                    asm("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0s), "r"(b1s));
                }
                s_sc[(2u * t4m) * t_s + pb + g8m] = d0 * scale;
                s_sc[(2u * t4m + 1u) * t_s + pb + g8m] = d1 * scale;
                s_sc[(2u * t4m) * t_s + pb + g8m + 8u] = d2 * scale;
                s_sc[(2u * t4m + 1u) * t_s + pb + g8m + 8u] = d3 * scale;
            }
        } else
        for (uint32_t idx = d; idx < n_t * G; idx += nth) {
            const uint32_t p = idx / G, h = idx % G;
            float sc = 0.0f;
            const float* qrow = s_q + h * q_s;
            const KV* krow = kbuf + (size_t)p * row_e;
            if (sizeof(KV) == 2u) {
                for (uint32_t dd = 0; dd < head_dim; dd += 8) {
                    const uint4 kr = *(const uint4*)(krow + dd);
                    const float4 q0 = *(const float4*)(qrow + dd);
                    const float4 q1 = *(const float4*)(qrow + dd + 4);
                    const __half2* kh = (const __half2*)&kr;
                    float2 k0 = __half22float2(kh[0]), k1 = __half22float2(kh[1]);
                    float2 k2 = __half22float2(kh[2]), k3 = __half22float2(kh[3]);
                    sc += q0.x * k0.x;
                    sc += q0.y * k0.y;
                    sc += q0.z * k1.x;
                    sc += q0.w * k1.y;
                    sc += q1.x * k2.x;
                    sc += q1.y * k2.y;
                    sc += q1.z * k3.x;
                    sc += q1.w * k3.y;
                }
            } else {
                // fp8: 8-wide (one uint2 smem read = 8 e4m3), mirroring the
                // f16 uint4 branch - the 2-elem loop halved score throughput
                for (uint32_t dd = 0; dd < head_dim; dd += 8) {
                    const uint2 kr8 = *(const uint2*)(krow + dd);
                    const __nv_fp8_e4m3* kb = (const __nv_fp8_e4m3*)&kr8;
                    #pragma unroll
                    for (uint32_t j = 0; j < 8u; ++j)
                        sc += qrow[dd + j] * (float)kb[j];
                }
            }
            s_sc[h * t_s + p] = sc * scale;
        }
        __syncthreads();
        {
            const uint32_t warp = d >> 5, lane = d & 31u;
            if (warp < G) {
                float v = (lane < n_t) ? s_sc[warp * t_s + lane] : -INFINITY;
                #pragma unroll
                for (uint32_t off = 16; off > 0; off >>= 1)
                    v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, off));
                if (lane == 0) {
                    const float m_new = fmaxf(s_m[warp], v);
                    s_mnew[warp] = m_new;
                    s_corr[warp] = __expf(s_m[warp] - m_new);
                }
            }
        }
        __syncthreads();
        for (uint32_t idx = d; idx < n_t * G; idx += nth) {
            const uint32_t p = idx / G, h = idx % G;
            s_w[h * t_s + p] = __expf(s_sc[h * t_s + p] - s_mnew[h]);
        }
        __syncthreads();
        if (pv_on) {
            // tensor-core PV: O[dim][head] += V^T (16-dim x 8-token) x P
            // (8-token x <=8-head) per m16n8k8 tf32 mma. V f16 is exact in
            // tf32; P (f32 exp weights) splits big+small like the score
            // path's Q, so the fold stays in the same ~1e-6 class. All 8
            // warps work (the scalar fold serialized 16 token-FMAs per
            // dim-pair): warp w owns dims [w*32, w*32+32) as two 16-dim
            // M-tiles; acc[] holds the two D fragments (dims g8/g8+8 x heads
            // 2*t4/2*t4+1 -- the same 8 registers the scalar path uses as
            // dim-pair slots). Tokens/heads outside (n_t, G) zero both
            // operands: stale first-tile smem can be NaN and NaN*0 = NaN.
            // MTN M-tiles per warp (hd256/T16 = 2, hd128/T32 = 1) and TILE>>3
            // 8-token chunks - compile-time via the gate's TILE<->hd bijection.
            constexpr uint32_t MTN = TILE == 16u ? 2u : 1u;
            const uint32_t warp_ = d >> 5, lane_ = d & 31u;
            const uint32_t g8 = lane_ >> 2, t4 = lane_ & 3u;
            const uint32_t h0 = 2u * t4, h1 = h0 + 1u;
            const float c0 = h0 < G ? s_corr[h0] : 0.0f;
            const float c1 = h1 < G ? s_corr[h1] : 0.0f;
            #pragma unroll
            for (uint32_t mt = 0; mt < MTN; ++mt) {
                acc[mt * 4u + 0] *= c0; acc[mt * 4u + 1] *= c1;
                acc[mt * 4u + 2] *= c0; acc[mt * 4u + 3] *= c1;
            }
            auto tf = [](float v) { uint32_t r; asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(v)); return r; };
            const __half* vb16 = (const __half*)vbuf;
            #pragma unroll
            for (uint32_t kc = 0; kc < (TILE >> 3); ++kc) {
                const uint32_t p0 = kc * 8u + t4, p1 = p0 + 4u;
                const float w0 = (p0 < n_t && g8 < G) ? s_w[g8 * t_s + p0] : 0.0f;
                const float w1 = (p1 < n_t && g8 < G) ? s_w[g8 * t_s + p1] : 0.0f;
                const uint32_t b0 = tf(w0), b1 = tf(w1);
                const float r0 = w0 - __uint_as_float(b0);
                const float r1 = w1 - __uint_as_float(b1);
                const uint32_t b0s = tf(r0), b1s = tf(r1);
                // third split term: big+mid+small carries P at ~33 bits (>
                // f32's 24), so the PV products round no worse than the
                // scalar f32 fold - two terms measured +0.5% PPL via chaos
                // amplification over long teacher-forced runs.
                const uint32_t b0t = tf(r0 - __uint_as_float(b0s));
                const uint32_t b1t = tf(r1 - __uint_as_float(b1s));
                #pragma unroll
                for (uint32_t mt = 0; mt < MTN; ++mt) {
                    const uint32_t db = warp_ * (16u * MTN) + mt * 16u + g8;
                    const uint32_t a0 = p0 < n_t ? tf(__half2float(vb16[(size_t)p0 * row_e + db])) : 0u;
                    const uint32_t a1 = p0 < n_t ? tf(__half2float(vb16[(size_t)p0 * row_e + db + 8u])) : 0u;
                    const uint32_t a2 = p1 < n_t ? tf(__half2float(vb16[(size_t)p1 * row_e + db])) : 0u;
                    const uint32_t a3 = p1 < n_t ? tf(__half2float(vb16[(size_t)p1 * row_e + db + 8u])) : 0u;
                    asm("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                        : "+f"(acc[mt * 4u + 0]), "+f"(acc[mt * 4u + 1]),
                          "+f"(acc[mt * 4u + 2]), "+f"(acc[mt * 4u + 3])
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
                    asm("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                        : "+f"(acc[mt * 4u + 0]), "+f"(acc[mt * 4u + 1]),
                          "+f"(acc[mt * 4u + 2]), "+f"(acc[mt * 4u + 3])
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0s), "r"(b1s));
                    asm("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                        : "+f"(acc[mt * 4u + 0]), "+f"(acc[mt * 4u + 1]),
                          "+f"(acc[mt * 4u + 2]), "+f"(acc[mt * 4u + 3])
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0t), "r"(b1t));
                }
            }
        } else
        for (uint32_t e2 = d, j = 0; e2 < G * (head_dim >> 1); e2 += nth, j += 2) {
            const uint32_t hd2 = head_dim >> 1;
            const uint32_t h = e2 / hd2, dd = (e2 - h * hd2) << 1;
            float a0 = acc[j] * s_corr[h];
            float a1 = acc[j + 1] * s_corr[h];
            const float* wrow = s_w + h * t_s;
            const KV* vrow = vbuf + dd;
            for (uint32_t p = 0; p < n_t; ++p) {
                const float2 vv = pd_kv_to_f32x2(vrow + (size_t)p * row_e);
                const float wp = wrow[p];
                a0 += wp * vv.x;
                a1 += wp * vv.y;
            }
            acc[j] = a0;
            acc[j + 1] = a1;
        }
        if (d < G) {
            float ws = 0.0f;
            for (uint32_t p = 0; p < n_t; ++p) ws += s_w[d * t_s + p];
            s_l[d] = s_l[d] * s_corr[d] + ws;
            s_m[d] = s_mnew[d];
        }
        __syncthreads();
    }
    __syncthreads();
    if (pv_on) {
        // fragment-layout writeback: each thread owns (dims g8, g8+8 of its
        // warp's MTN 16-dim M-tiles) x (heads 2*t4, 2*t4+1) - every (dim,
        // head) pair lands exactly once, heads >= G skipped.
        constexpr uint32_t MTN = TILE == 16u ? 2u : 1u;
        const uint32_t warp_ = d >> 5, lane_ = d & 31u;
        const uint32_t g8 = lane_ >> 2, t4 = lane_ & 3u;
        const uint32_t h0 = 2u * t4, h1 = h0 + 1u;
        #pragma unroll
        for (uint32_t mt = 0; mt < MTN; ++mt) {
            const uint32_t db = warp_ * (16u * MTN) + mt * 16u + g8;
            if (h0 < G) {
                const size_t pidx = ((size_t)(kvh * G + h0) * gridDim.y + b) * n_splits + s;
                out_o[pidx * head_dim + db] = acc[mt * 4u + 0];
                out_o[pidx * head_dim + db + 8u] = acc[mt * 4u + 2];
            }
            if (h1 < G) {
                const size_t pidx = ((size_t)(kvh * G + h1) * gridDim.y + b) * n_splits + s;
                out_o[pidx * head_dim + db] = acc[mt * 4u + 1];
                out_o[pidx * head_dim + db + 8u] = acc[mt * 4u + 3];
            }
        }
    } else
    for (uint32_t e2 = d, j = 0; e2 < G * (head_dim >> 1); e2 += nth, j += 2) {
        const uint32_t hd2 = head_dim >> 1;
        const uint32_t h = e2 / hd2, dd = (e2 - h * hd2) << 1;
        const size_t pidx = ((size_t)(kvh * G + h) * gridDim.y + b) * n_splits + s;
        out_o[pidx * head_dim + dd] = acc[j];
        out_o[pidx * head_dim + dd + 1] = acc[j + 1];
    }
    if (d < G) {
        const size_t pidx = ((size_t)(kvh * G + d) * gridDim.y + b) * n_splits + s;
        out_ml[pidx * 2 + 0] = s_m[d];
        out_ml[pidx * 2 + 1] = s_l[d];
    }
}

// vec8 decode walk: register-resident per-(q-head,
// split) fp8-KV walk for hd128 short-context serving. The GQA-fused walk
// above optimizes DRAM bytes that don't bind at serve contexts (64 layers'
// hot KV sits in L2) while its geometry starves the die: 8 kv-head CTAs x
// s_eff(ceil(n_pos/128)) = 8-16 live CTAs on 188 SMs, each paying smem
// staging + 2 barriers per 32-token tile. This class (llama.cpp b10327
// fattn-vec studied as reference; original implementation) launches one
// 128-thread CTA per (q-head, split): Q pre-scaled in registers, one warp
// reads exactly one 128 B K line per token, warp-shfl score reduce, online
// softmax per warp with zero block barriers in the walk, 4-way smem merge at
// the epilogue only. Partials land in the production combine layout so
// pd_attn_decode_batch_combine* run unchanged. Numerics: same values as the
// fused walk (e4m3->f32 cvt is exact), different reduction ORDER - the
// sanctioned split-regroup class (parity gates at 2e-7).
// Isolated, combine included: 14.4 -> 8.2 us at ctx<=512 B=1, and it wins
// every swept (ctx, B) cell with q-head-budgeted splits.
template<uint32_t HD, uint32_t TPC>
__global__ void pd_attn_decode_vec8_paged_kernel(
    const float* __restrict__ q, const __nv_fp8_e4m3* __restrict__ pool_k,
    const __nv_fp8_e4m3* __restrict__ pool_v, float* __restrict__ out_o,
    float* __restrict__ out_ml, const unsigned int* __restrict__ positions,
    const unsigned int* __restrict__ slots,
    const uint32_t* __restrict__ block_tables, uint32_t blocks_per_slot,
    uint32_t n_heads, uint32_t n_kv_heads, uint32_t kv_dim,
    uint32_t swa_window, uint32_t n_splits, float scale) {
    PD_PDL_ARM();
    const uint32_t h = blockIdx.x, b = blockIdx.y, s = blockIdx.z;
    const uint32_t G = n_heads / n_kv_heads;
    const uint32_t kvh = h / G;
    const uint32_t lane = threadIdx.x & 31u, w = threadIdx.x >> 5;
    const uint32_t slot = slots ? slots[b] : b;
    const uint32_t pos = positions[b];
    const uint32_t first_pos =
        (swa_window > 0 && pos + 1 > swa_window) ? (pos + 1 - swa_window) : 0;
    const uint32_t n_pos = pos + 1 - first_pos;

    uint32_t s_eff = (n_pos + TPC - 1u) / TPC;
    if (s_eff > n_splits) s_eff = n_splits;
    if (s_eff < 1u) s_eff = 1u;
    const uint32_t chunk = (n_pos + s_eff - 1u) / s_eff;
    const uint32_t lo = s * chunk;
    uint32_t hi = lo + chunk;
    if (hi > n_pos) hi = n_pos;

    float qr[4];
    const float* qb = q + ((size_t)b * n_heads + h) * HD + lane * 4u;
    #pragma unroll
    for (int j = 0; j < 4; ++j) qr[j] = qb[j] * scale;

    const uint32_t* bt = block_tables + (size_t)slot * blocks_per_slot;
    float m = -INFINITY, l = 0.0f, vacc[4] = {0.f, 0.f, 0.f, 0.f};

    for (uint32_t t = lo + w; t < hi; t += 4u) {
        const uint32_t tok = first_pos + t;
        const size_t row = (size_t)bt[tok >> 4] * 16u + (tok & 15u);
        const uint32_t kw =
            *(const uint32_t*)(pool_k + row * kv_dim + kvh * HD + lane * 4u);
        float d0 = 0.f;
        #pragma unroll
        for (int j = 0; j < 4; ++j)
            d0 += qr[j] * float(*((const __nv_fp8_e4m3*)&kw + j));
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) d0 += __shfl_xor_sync(0xffffffffu, d0, o);
        const float mn = fmaxf(m, d0);
        const float corr = __expf(m - mn), wt = __expf(d0 - mn);
        l = l * corr + wt;
        m = mn;
        const uint32_t vw =
            *(const uint32_t*)(pool_v + row * kv_dim + kvh * HD + lane * 4u);
        #pragma unroll
        for (int j = 0; j < 4; ++j)
            vacc[j] = vacc[j] * corr + wt * float(*((const __nv_fp8_e4m3*)&vw + j));
    }

    __shared__ float sm_m[4], sm_l[4], sm_acc[4][HD];
    sm_m[w] = m; sm_l[w] = l;
    #pragma unroll
    for (int j = 0; j < 4; ++j) sm_acc[w][lane * 4u + j] = vacc[j];
    __syncthreads();
    const size_t pidx = ((size_t)h * gridDim.y + b) * n_splits + s;
    if (w == 0) {
        float M = fmaxf(fmaxf(sm_m[0], sm_m[1]), fmaxf(sm_m[2], sm_m[3]));
        if (M == -INFINITY) {                  // empty split: combine folds it
            #pragma unroll
            for (int j = 0; j < 4; ++j) out_o[pidx * HD + lane * 4u + j] = 0.0f;
            if (lane == 0) { out_ml[pidx * 2] = -INFINITY; out_ml[pidx * 2 + 1] = 0.0f; }
            return;
        }
        float wgt[4], L = 0.0f;
        #pragma unroll
        for (int ww = 0; ww < 4; ++ww) {
            wgt[ww] = __expf(sm_m[ww] - M);
            L += sm_l[ww] * wgt[ww];
        }
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint32_t d = lane * 4u + j;
            float o = 0.0f;
            #pragma unroll
            for (int ww = 0; ww < 4; ++ww) o += wgt[ww] * sm_acc[ww][d];
            out_o[pidx * HD + d] = o;
        }
        if (lane == 0) { out_ml[pidx * 2] = M; out_ml[pidx * 2 + 1] = L; }
    }
}


// Batched FlashDecoding combine: merge the n_splits partials for each (head, seq)
// into the final output, folding the per-head sink into the denominator. grid
// (n_heads, batch); one thread per output dim. Empty splits carry m=-inf -> drop.
// To (attention streams): f16 final plane; partials stay f32.
template<typename TO = float>
__global__ void pd_attn_decode_batch_combine_kernel(
    const float* __restrict__ in_o, const float* __restrict__ in_ml,
    const float* __restrict__ sinks, TO* __restrict__ out,
    uint32_t n_heads, uint32_t head_dim, uint32_t n_splits) {
    PD_PDL_ARM();
    uint32_t h = blockIdx.x, b = blockIdx.y;
    uint32_t d = threadIdx.x;
    size_t pbase = (size_t)(h * gridDim.y + b) * n_splits;

    float gm = sinks[h];
    for (uint32_t s = 0; s < n_splits; ++s)
        gm = fmaxf(gm, in_ml[(pbase + s) * 2 + 0]);

    float acc = 0.0f, l = 0.0f;
    for (uint32_t s = 0; s < n_splits; ++s) {
        float m = in_ml[(pbase + s) * 2 + 0];
        if (m == -INFINITY) continue;  // empty split: sc = expf(-inf-gm) = 0
                                       // and o = 0 - an exact +0 term. Skip
                                       // the head_dim-float o read; at short
                                       // kv the GV adaptive-split arm leaves
                                       // most splits empty (s_eff << n_splits)
                                       // and this drops ~7/8 of the o traffic.
        float ls = in_ml[(pbase + s) * 2 + 1];
        float sc = __expf(m - gm);
        acc += sc * in_o[(pbase + s) * head_dim + d];
        l += sc * ls;
    }
    l += __expf(sinks[h] - gm);               // sink: denominator only
    out[((size_t)b * n_heads + h) * head_dim + d] = (TO)(acc / l);
}

// Fused single-pass GQA decode attention (the trtllm-gen chase): one CTA
// per (kv-head, row), 32*GG threads = GG warps = the
// whole q-group, the row's entire windowed K/V run staged to dynamic smem
// once (uint4), each warp's head walked register-resident from smem in
// 32-token score tiles - the OWNER lane keeps its token's score so the
// softmax exp runs once per token in parallel across lanes (the per-lane-
// redundant exp measured +20%) - and the FINAL output written
// in-kernel with the sink folded into the denominator. No partial planes,
// no combine launch, 1/GG the pool traffic of the per-(q-head, split) vec8
// walk. Measured on muse 32q/2kv hd128 fp8, B=32, 48 layers cycling L2:
// 22.6/30.9/40.8 us at ctx 128/192/256, against 33.2/49.9/65.0 for the
// vec8-splits=2 + combine path it replaces. Loses at
// B<=16 (few CTAs, latency-bound: 0.14 waves with nothing saturated) -
// the engine elects it at rows>=24 && band<=768 only; vec8+combine keep
// every other cell. Numerics: e4m3->f32 exact cvt, f32 online softmax -
// token-serial reduction order per head (the sanctioned split-regroup
// class; parity gates 2.4-4.3e-7).
template<uint32_t HD, uint32_t GG>
__global__ void __launch_bounds__(32 * GG, 1) pd_attn_decode_fused_gqa16_kernel(
    const float* __restrict__ q, const __nv_fp8_e4m3* __restrict__ pool_k,
    const __nv_fp8_e4m3* __restrict__ pool_v, float* __restrict__ out,
    const unsigned int* __restrict__ positions,
    const unsigned int* __restrict__ slots,
    const uint32_t* __restrict__ block_tables, uint32_t blocks_per_slot,
    uint32_t n_heads, uint32_t n_kv_heads, uint32_t kv_dim,
    uint32_t swa_window, float scale, const float* __restrict__ sinks) {
    PD_PDL_ARM();
    const uint32_t kvh = blockIdx.x, b = blockIdx.y;
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const uint32_t slot = slots ? slots[b] : b;
    const uint32_t pos = positions[b];
    const uint32_t first_pos =
        (swa_window > 0 && pos + 1 > swa_window) ? (pos + 1 - swa_window) : 0;
    const uint32_t n_pos = pos + 1 - first_pos;

    // layout: K[n_pos][HD] fp8 | V[n_pos][HD] fp8 | Q[GG][HD] f32. The
    // warp-cooperative reads (addr/4 = t*HD/4 + lane) are conflict-free
    // unpadded.
    extern __shared__ __nv_fp8_e4m3 sm_kv[];
    __nv_fp8_e4m3* sm_k = sm_kv;
    __nv_fp8_e4m3* sm_v = sm_kv + (size_t)n_pos * HD;
    float* sm_q = (float*)(sm_v + (size_t)n_pos * HD);

    const uint32_t* bt = block_tables + (size_t)slot * blocks_per_slot;
    for (uint32_t e = threadIdx.x; e < GG * HD; e += blockDim.x) {
        const uint32_t qh = kvh * GG + e / HD;
        sm_q[e] = q[((size_t)b * n_heads + qh) * HD + (e % HD)] * scale;
    }
    const uint32_t CHV = HD / 16u;                      // uint4 chunks per row
    for (uint32_t c = threadIdx.x; c < n_pos * CHV; c += blockDim.x) {
        const uint32_t t = c / CHV, off = (c - t * CHV) * 16u;
        const uint32_t tok = first_pos + t;
        const size_t row = (size_t)bt[tok >> 4] * 16u + (tok & 15u);
        const size_t src = row * kv_dim + kvh * HD + off;
        *(uint4*)(sm_k + (size_t)t * HD + off) = *(const uint4*)(pool_k + src);
        *(uint4*)(sm_v + (size_t)t * HD + off) = *(const uint4*)(pool_v + src);
    }
    __syncthreads();

    // warp == q-head within the group; q dims live in registers
    const uint32_t h = kvh * GG + warp;
    const float* qw = sm_q + (size_t)warp * HD;
    float qr[4];
    #pragma unroll
    for (int j = 0; j < 4; ++j) qr[j] = qw[lane * 4u + j];

    float m = -INFINITY, l = 0.0f, vacc[4] = {0.f, 0.f, 0.f, 0.f};
    for (uint32_t t0 = 0; t0 < n_pos; t0 += 32u) {
        const uint32_t nt = min(32u, n_pos - t0);
        // pass A: warp-cooperative dot per token, owner lane keeps the score
        float s_own = -INFINITY;
        #pragma unroll 8
        for (uint32_t i = 0; i < nt; ++i) {
            const uint32_t kw =
                *(const uint32_t*)(sm_k + (size_t)(t0 + i) * HD + lane * 4u);
            float d0 = 0.f;
            #pragma unroll
            for (int j = 0; j < 4; ++j)
                d0 += qr[j] * float(*((const __nv_fp8_e4m3*)&kw + j));
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1)
                d0 += __shfl_xor_sync(0xffffffffu, d0, o);
            if (i == lane) s_own = d0;
        }
        // tile max -> one corr exp + one parallel exp per token
        float mt = s_own;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1)
            mt = fmaxf(mt, __shfl_xor_sync(0xffffffffu, mt, o));
        const float mn = fmaxf(m, mt);
        const float corr = __expf(m - mn);
        const float w_own = (lane < nt) ? __expf(s_own - mn) : 0.0f;
        float ws = w_own;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) ws += __shfl_xor_sync(0xffffffffu, ws, o);
        l = l * corr + ws;
        m = mn;
        #pragma unroll
        for (int j = 0; j < 4; ++j) vacc[j] *= corr;
        // pass B: broadcast each token's weight, accumulate its V row
        #pragma unroll 8
        for (uint32_t i = 0; i < nt; ++i) {
            const float wt = __shfl_sync(0xffffffffu, w_own, i);
            const uint32_t vw =
                *(const uint32_t*)(sm_v + (size_t)(t0 + i) * HD + lane * 4u);
            #pragma unroll
            for (int j = 0; j < 4; ++j)
                vacc[j] += wt * float(*((const __nv_fp8_e4m3*)&vw + j));
        }
    }

    // epilogue: fold the sink into the denominator, write the FINAL output
    const float snk = sinks ? sinks[h] : -INFINITY;
    const float gm = fmaxf(m, snk);
    const float corr = __expf(m - gm);
    const float L = l * corr + __expf(snk - gm);
    #pragma unroll
    for (int j = 0; j < 4; ++j)
        out[((size_t)b * n_heads + h) * HD + lane * 4u + j] = vacc[j] * corr / L;
}

// Largest expert count the routers handle (gpt-oss-20b = 32, 120b = 128,
// qwen3.6-A3B = 256). The host wrappers reject anything larger instead of
// silently corrupting.
#define PD_MOE_MAX_EXPERT 256

// Warp top-k: each lane holds up to 8 of the <=256 (biased) logits in
// registers; k rounds of a warp argmax (shfl tree). Tie-break and softmax
// summation order exactly match the old one-thread scan (strict >, ascending
// index - the lane-local scan walks i, i+32, i+64, ... and the cross-lane
// combine prefers the lower index on equal values), so the outputs are
// bit-identical and greedy streams are unchanged (models with <= 128 experts
// see -1e30 in the extra registers - selection is untouched). The
// one-thread version's k x n_expert dependent walk through spilled locals
// measured 19.9 us at 128 experts - x36 layers it was the 120b decode gap
// (~0.7 ms/token).
// VJ = per-lane register slots, so the walk covers 32*VJ experts. VJ=8 (256)
// was the only shape until qwen4_exp arrived with 512 routed experts: the
// cap was silent - experts past 256 were simply never eligible, and every
// model in the fleet until then had 128 or fewer, so nothing caught it.
// VJ=8 remains the default instantiation, so every existing caller keeps its
// exact register footprint and codegen.
template <uint32_t VJ>
__device__ __forceinline__ void pd_moe_topk_warp_t(const float* __restrict__ logits,
                                                 const float* __restrict__ bias,
                                                 uint32_t n_expert, uint32_t k,
                                                 uint32_t* __restrict__ out_idx,
                                                 float* __restrict__ out_w) {
    const uint32_t lane = threadIdx.x & 31u;
    float v[VJ];
    #pragma unroll
    for (uint32_t j = 0; j < VJ; ++j) {
        uint32_t i = lane + 32u * j;
        v[j] = i < n_expert ? logits[i] + (bias ? bias[i] : 0.0f) : -1e30f;
    }
    float sel_logit[16];
    for (uint32_t s = 0; s < k; ++s) {
        float best = -1e30f;
        uint32_t bi = 0;
        #pragma unroll
        for (uint32_t j = 0; j < VJ; ++j) {
            uint32_t i = lane + 32u * j;
            if (v[j] > best) { best = v[j]; bi = i; }
        }
        for (uint32_t off = 16; off > 0; off >>= 1) {
            float ov = __shfl_down_sync(0xffffffffu, best, off);
            uint32_t oi = __shfl_down_sync(0xffffffffu, bi, off);
            if (ov > best || (ov == best && oi < bi)) { best = ov; bi = oi; }
        }
        best = __shfl_sync(0xffffffffu, best, 0);
        bi = __shfl_sync(0xffffffffu, bi, 0);
        sel_logit[s] = best;
        if (lane == 0) out_idx[s] = bi;
        if ((bi & 31u) == lane) v[bi >> 5] = -1e30f;
    }
    if (lane == 0) {
        float mm = sel_logit[0];
        for (uint32_t s = 1; s < k; ++s) mm = fmaxf(mm, sel_logit[s]);
        float sum = 0.0f;
        for (uint32_t s = 0; s < k; ++s) {
            float e = __expf(sel_logit[s] - mm);
            out_w[s] = e;
            sum += e;
        }
        for (uint32_t s = 0; s < k; ++s) out_w[s] /= sum;
    }
}

// The 256-expert instantiation, under the historic name: every pre-existing
// caller (moe/f8.cuh, gemm/f32_qkv.cuh, the kernels below) binds to this and
// is bit-identical to before the template split.
__device__ __forceinline__ void pd_moe_topk_warp(const float* __restrict__ logits,
                                                 const float* __restrict__ bias,
                                                 uint32_t n_expert, uint32_t k,
                                                 uint32_t* __restrict__ out_idx,
                                                 float* __restrict__ out_w) {
    pd_moe_topk_warp_t<8u>(logits, bias, n_expert, k, out_idx, out_w);
}

// Widest expert count any router walk here covers. Past this the launchers
// REFUSE - a silent truncation of the expert set reads as a plausible model.
#define PD_MOE_TOPK_MAX_EXPERTS 512u

// MoE top-k router, single token (no bias fold). One warp; see pd_moe_topk_warp.
template <uint32_t VJ>
__global__ void pd_moe_topk_kernel_t(const float* __restrict__ logits, uint32_t n_expert,
                                   uint32_t k, uint32_t* __restrict__ out_idx,
                                   float* __restrict__ out_w) {
    if (blockIdx.x != 0) return;
    pd_moe_topk_warp_t<VJ>(logits, (const float*)0, n_expert, k, out_idx, out_w);
}

// Batched MoE top-k router: grid `batch`, block b does token b's top-k over
// biased logits [batch, n_expert] -> out_idx/out_w [batch, k]. Bias is folded in.
template <uint32_t VJ>
__global__ void pd_moe_topk_batch_kernel_t(const float* __restrict__ logits,
                                         const float* __restrict__ bias, uint32_t n_expert,
                                         uint32_t k, uint32_t* __restrict__ out_idx,
                                         float* __restrict__ out_w) {
    uint32_t b = blockIdx.x;
    pd_moe_topk_warp_t<VJ>(logits + (size_t)b * n_expert, bias, n_expert, k,
                     out_idx + (size_t)b * k, out_w + (size_t)b * k);
}

// f32 batched matvec for the tiny router GEMM: out[t][o] = dot(w[o], x[t]),
// grid (out_dim, batch), block-per-output warp-shuffle reduce. Replaces the
// cuBLAS gemmSN + splitKreduce pair at serving batch, whose two launches cost
// ~19 us/layer of pure latency on a 368 KB weight (0.45 ms/step at any B).
// 8-token tile: one block per (output, 8-token group). The block-per-(o,t)
// original re-walked the weight row per token and ran 4096 tiny blocks at
// c32/128e - 21.7 us of pure wave latency for a 1.5 MB L2-resident weight
// Each token's summation here
// is BIT-IDENTICAL to the original (same per-thread i stride, same shfl
// tree, same serial cross-warp sum) - the router feeds topk, where a last-
// ulp change could flip a tie - so this is a pure schedule change.
template <uint32_t BT>
__global__ void pd_matvec_f32_batch_kernel(const float* __restrict__ w,
                                           const float* __restrict__ x,
                                           float* __restrict__ out,
                                           uint32_t in_dim, uint32_t out_dim,
                                           uint32_t batch) {
    // cascade (laguna router): x is the predecessor rmsnorm's output
    PD_PDL_ARM();
    const uint32_t o = blockIdx.x, t0 = blockIdx.y * BT;
    const uint32_t tid = threadIdx.x, nth = blockDim.x;
    const float* wr = w + (size_t)o * in_dim;
    float acc[BT] = {};
    for (uint32_t i = tid; i < in_dim; i += nth) {
        const float wv = wr[i];
        #pragma unroll
        for (uint32_t b = 0; b < BT; ++b)
            if (t0 + b < batch) acc[b] += wv * x[(size_t)(t0 + b) * in_dim + i];
    }
    __shared__ float wsum[8][BT];
    const uint32_t warp = tid >> 5, lane = tid & 31u;
    #pragma unroll
    for (uint32_t b = 0; b < BT; ++b) {
        float v = acc[b];
        for (uint32_t s = 16; s > 0; s >>= 1) v += __shfl_down_sync(0xffffffffu, v, s);
        if (lane == 0) wsum[warp][b] = v;
    }
    __syncthreads();
    if (tid == 0) {
        const uint32_t nwarps = (nth + 31u) >> 5;
        #pragma unroll
        for (uint32_t b = 0; b < BT; ++b) {
            if (t0 + b >= batch) break;
            float v = 0.0f;
            for (uint32_t i = 0; i < nwarps; ++i) v += wsum[i][b];
            out[(size_t)(t0 + b) * out_dim + o] = v;
        }
    }
}

// Fused MXFP4-dequant + GEMV for one expert selected by a device index (so the
// whole MoE loop stays on-device - no sync). Weight is the full 3-D expert
// tensor (MXFP4); expert e = idx[slot]. y[o] = bias[e][o] + Σ_i dequant(W[e][o][i])·x[i].
// Rows are block-aligned (in_dim % 32 == 0), so no cross-row block straddling.
//
// One block per output row, one element per thread (so x reads coalesce). Two
// things that measurably hurt the naive version: (1) PD_FP4_VALUES lives in
// __constant__ memory and the nibble index is divergent across a warp, which
// serializes constant loads - staged into a shared 16-entry LUT here so lookups
// broadcast from shared instead; (2) the shared-memory tree reduction cost 8
// __syncthreads - replaced with a warp-shuffle reduce plus one cross-warp combine.
__global__ void pd_mxfp4_gemv_indexed_kernel(
    const uint8_t* __restrict__ W, const float* __restrict__ bias,
    const uint32_t* __restrict__ idx, uint32_t slot,
    const float* __restrict__ x, float* __restrict__ y,
    uint32_t in_dim, uint32_t out_dim) {
    uint32_t o = blockIdx.x;
    if (o >= out_dim) return;
    uint32_t e = idx[slot];
    uint32_t tid = threadIdx.x, nth = blockDim.x;

    __shared__ float lut[16];
    __shared__ float wsum[32];
    if (tid < 16) lut[tid] = PD_FP4_VALUES[tid];
    __syncthreads();

    size_t elem_base = (size_t)e * in_dim * out_dim + (size_t)o * in_dim;
    size_t byte_base = (elem_base / 32) * 17;   // in_dim % 32 == 0 -> exact

    float acc = 0.0f;
    for (uint32_t i = tid; i < in_dim; i += nth) {
        const uint8_t* bp = W + byte_base + (size_t)(i >> 5) * 17;
        float d = pd_e8m0_half(bp[0]);
        uint32_t j = i & 31u;
        uint8_t packed = bp[1 + (j & 15u)];
        uint8_t nib = (j < 16u) ? (packed & 0x0F) : (packed >> 4);
        acc += lut[nib] * d * x[i];
    }

    for (uint32_t s = 16; s > 0; s >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, s);
    uint32_t warp = tid >> 5, lane = tid & 31u;
    if (lane == 0) wsum[warp] = acc;
    __syncthreads();
    if (tid == 0) {
        float v = 0.0f;
        uint32_t nwarps = (nth + 31u) >> 5;
        for (uint32_t w = 0; w < nwarps; ++w) v += wsum[w];
        if (bias) v += bias[(size_t)e * out_dim + o];
        y[o] = v;
    }
}

// Fused MoE gate+up+swiglu, batched over the active experts. grid (ff, n_active),
// one block per (ff row o, expert slot). Reads x once and drives both the gate and
// up projections for expert idx[slot], then applies swiglu_oai inline and writes
// out[slot*ff + o]. Replaces 4×(gate gemv + up gemv + swiglu) = 12 launches with
// one, kills the d_gate/d_up round-trips, and reuses the shared-LUT + warp-shuffle
// shape from the single-expert GEMV.
__global__ void pd_mxfp4_moe_gate_up_kernel(
    const uint8_t* __restrict__ gate_data, const uint8_t* __restrict__ gate_scale,
    const float* __restrict__ gate_bias,
    const uint8_t* __restrict__ up_data, const uint8_t* __restrict__ up_scale,
    const float* __restrict__ up_bias,
    const uint32_t* __restrict__ idx, const float* __restrict__ x,
    float* __restrict__ out, uint32_t in_dim, uint32_t ff, float alpha, float limit) {
    uint32_t o = blockIdx.x, slot = blockIdx.y;
    uint32_t e = idx[slot];
    uint32_t tid = threadIdx.x, nth = blockDim.x;
    uint32_t n_blocks = in_dim >> 5;

    // dynamic shared: [lut 16][wg 32][wu 32][gate scales n_blocks][up scales n_blocks].
    // Precompute each block's e8m0 scale once (was recomputed per byte, 16× per
    // block) and read it back from shared in the hot loop.
    extern __shared__ float smem[];
    float* lut = smem;
    float* wg = lut + 16;
    float* wu = wg + 32;
    float* sgs = wu + 32;
    float* sus = sgs + n_blocks;
    if (tid < 16) lut[tid] = PD_FP4_VALUES[tid];

    size_t blk_base = (size_t)((size_t)e * ff + o) * (in_dim / 32);   // first block index
    for (uint32_t b = tid; b < n_blocks; b += nth) {
        sgs[b] = pd_e8m0_half(gate_scale[blk_base + b]);
        sus[b] = pd_e8m0_half(up_scale[blk_base + b]);
    }
    __syncthreads();

    float acc_g = 0.0f, acc_u = 0.0f;
    // one thread per NIBBLE BYTE (not per element): each byte packs two elements
    // (low nibble = element blk*32+p, high = +16), so we read the byte once, drop
    // the j<16 branch, and halve the iteration count vs per-element. Unroll so the
    // compiler pipelines several independent weight loads to hide their latency.
#pragma unroll 4
    for (uint32_t nb = tid; nb < n_blocks * 16u; nb += nth) {
        uint32_t blk = nb >> 4, p = nb & 15u;
        size_t doff = (blk_base + blk) * 16 + p;
        float dg = sgs[blk], du = sus[blk];
        uint8_t pg = gate_data[doff], pu = up_data[doff];
        uint32_t lo = blk * 32u + p, hi = lo + 16u;
        float xl = x[lo], xh = x[hi];
        acc_g += dg * (lut[pg & 0x0F] * xl + lut[pg >> 4] * xh);
        acc_u += du * (lut[pu & 0x0F] * xl + lut[pu >> 4] * xh);
    }
    for (uint32_t s = 16; s > 0; s >>= 1) {
        acc_g += __shfl_down_sync(0xffffffffu, acc_g, s);
        acc_u += __shfl_down_sync(0xffffffffu, acc_u, s);
    }
    uint32_t warp = tid >> 5, lane = tid & 31u;
    if (lane == 0) { wg[warp] = acc_g; wu[warp] = acc_u; }
    __syncthreads();
    if (tid == 0) {
        float g = 0.0f, u = 0.0f;
        uint32_t nwarps = (nth + 31u) >> 5;
        for (uint32_t w = 0; w < nwarps; ++w) { g += wg[w]; u += wu[w]; }
        g += gate_bias[(size_t)e * ff + o];
        u += up_bias[(size_t)e * ff + o];
        float xg = fminf(g, limit);
        float yu = fminf(fmaxf(u, -limit), limit);
        out[(size_t)slot * ff + o] = (xg / (1.0f + expf(-alpha * xg))) * (yu + 1.0f);
    }
}

// Fused MoE down + weighted expert-mix + residual add. One block per output dim o
// (grid embd). Each block sums over the active experts: Σ_slot w[slot]·(down_e·
// fused[slot] + down_bias[e][o]), then adds straight into the residual. Replaces
// 4×(down gemv + scale_add) + memset + residual-add with a single launch and no
// d_down/d_moe scratch. Weight folded into the per-element accumulate.
__global__ void pd_mxfp4_moe_down_kernel(
    const uint8_t* __restrict__ down_data, const uint8_t* __restrict__ down_scale,
    const float* __restrict__ down_bias,
    const uint32_t* __restrict__ idx, const float* __restrict__ topk_w,
    const float* __restrict__ fused, float* __restrict__ residual,
    uint32_t ff, uint32_t embd, uint32_t n_active) {
    uint32_t o = blockIdx.x;
    uint32_t tid = threadIdx.x, nth = blockDim.x;
    uint32_t n_blocks = ff >> 5;

    // dynamic shared: [lut 16][wsum 32][per-slot block scales n_active*n_blocks].
    // Precompute each active expert's block scales once (was per byte, 16×/block).
    extern __shared__ float smem[];
    float* lut = smem;
    float* wsum = lut + 16;
    float* sc = wsum + 32;
    if (tid < 16) lut[tid] = PD_FP4_VALUES[tid];
    for (uint32_t slot = 0; slot < n_active; ++slot) {
        size_t blk_base = (size_t)((size_t)idx[slot] * embd + o) * (ff / 32);
        for (uint32_t b = tid; b < n_blocks; b += nth)
            sc[slot * n_blocks + b] = pd_e8m0_half(down_scale[blk_base + b]);
    }
    __syncthreads();

    float acc = 0.0f;   // Σ_slot w · Σ_i dequant(down)·fused
    for (uint32_t slot = 0; slot < n_active; ++slot) {
        uint32_t e = idx[slot];
        float w = topk_w[slot];
        size_t blk_base = (size_t)((size_t)e * embd + o) * (ff / 32);
        const float* fs = fused + (size_t)slot * ff;
        const float* scs = sc + slot * n_blocks;
        // one thread per nibble byte -> two elements per read, no j<16 branch
#pragma unroll 4
        for (uint32_t nb = tid; nb < n_blocks * 16u; nb += nth) {
            uint32_t blk = nb >> 4, p = nb & 15u;
            uint8_t packed = down_data[(blk_base + blk) * 16 + p];
            uint32_t lo = blk * 32u + p, hi = lo + 16u;
            acc += w * scs[blk] * (lut[packed & 0x0F] * fs[lo] + lut[packed >> 4] * fs[hi]);
        }
    }
    for (uint32_t s = 16; s > 0; s >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, s);
    uint32_t warp = tid >> 5, lane = tid & 31u;
    if (lane == 0) wsum[warp] = acc;
    __syncthreads();
    if (tid == 0) {
        float v = 0.0f;
        uint32_t nwarps = (nth + 31u) >> 5;
        for (uint32_t w = 0; w < nwarps; ++w) v += wsum[w];
        float biasacc = 0.0f;
        for (uint32_t slot = 0; slot < n_active; ++slot)
            biasacc += topk_w[slot] * down_bias[(size_t)idx[slot] * embd + o];
        residual[o] += v + biasacc;
    }
}

// dp4a version of the fused gate+up+swiglu MoE kernel. Same shape as
// pd_mxfp4_moe_gate_up but the activation arrives pre-quantized (xq/xs) and the
// two projections run on integer __dp4a with in-register nibble unpack.
// Warp-per-output shape: in_dim/32 is small (90 on the 20b), so the old
// block-per-output 256-thread strided loop left ~2/3 of every block idle and
// paid a shared-tree combine. One warp owns one ff row (lanes stride the
// k-blocks, ~3 each), 8 outputs per 256-thread block, pure shuffle reduce.
__global__ void pd_mxfp4_moe_gate_up_dp4a_kernel(
    const unsigned char* __restrict__ gate_data, const unsigned char* __restrict__ gate_scale,
    const float* __restrict__ gate_bias,
    const unsigned char* __restrict__ up_data, const unsigned char* __restrict__ up_scale,
    const float* __restrict__ up_bias,
    const unsigned int* __restrict__ idx, const signed char* __restrict__ xq,
    const float* __restrict__ xs, float* __restrict__ out,
    uint32_t in_dim, uint32_t ff, float alpha, float limit) {
    uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    uint32_t o = blockIdx.x * (blockDim.x >> 5) + warp;
    uint32_t slot = blockIdx.y;
    if (o >= ff) return;
    // batched grid: blockIdx.z = token (the llama mmvq-with-ids shape for tiny
    // serving batches). z = 0 zeroes every offset, so the single-token launch
    // stays bit-identical to the pre-batch kernel (B=1 graphs replay it).
    uint32_t t = blockIdx.z, nact = gridDim.y;
    xq += (size_t)t * in_dim;
    xs += (size_t)t * (in_dim >> 5);
    out += (size_t)t * nact * ff;
    uint32_t e = idx[(size_t)t * nact + slot];
    uint32_t n_blocks = in_dim >> 5;

    size_t blk_base = (size_t)((size_t)e * ff + o) * (in_dim / 32);
    float acc_g = 0.0f, acc_u = 0.0f;
    for (uint32_t b = lane; b < n_blocks; b += 32) {
        // one streamed 16B load per matrix per k-block (weights are never
        // reused by this block - bypass L1) instead of 4x4B memcpy loads;
        // identical bytes and dp4a order, so the sums are bit-identical
        // g||u ILV layout (see gate_up_bs): block b of this row sits in
        // pair b/4 at +0 (gate) / +64 (up); everything reads via gate_data
        const size_t ivb = (size_t)((size_t)e * ff + o) * (size_t)(((n_blocks + 3u) >> 2) * 128u) +
                           (size_t)(b >> 2) * 128u + (b & 3u) * 16u;
        uint4 gw = __ldcs(reinterpret_cast<const uint4*>(gate_data + ivb));
        uint4 uw = __ldcs(reinterpret_cast<const uint4*>(gate_data + ivb + 64u));
        float gd = pd_e8m0_half(gate_scale[blk_base + b]) * xs[b];
        float ud = pd_e8m0_half(up_scale[blk_base + b]) * xs[b];
        const int4 alo = *reinterpret_cast<const int4*>(xq + (size_t)b * 32);
        const int4 ahi = *reinterpret_cast<const int4*>(xq + (size_t)b * 32 + 16);
        int sg = 0, su = 0;
#pragma unroll
        for (uint32_t i = 0; i < 4; ++i) {
            int2 vg = pd_fp4_unpack8((&gw.x)[i]);
            int2 vu = pd_fp4_unpack8((&uw.x)[i]);
            int a_lo = (&alo.x)[i];
            int a_hi = (&ahi.x)[i];
            sg = __dp4a(vg.x, a_lo, sg);
            sg = __dp4a(vg.y, a_hi, sg);
            su = __dp4a(vu.x, a_lo, su);
            su = __dp4a(vu.y, a_hi, su);
        }
        acc_g += gd * (float)sg;
        acc_u += ud * (float)su;
    }
    for (uint32_t s = 16; s > 0; s >>= 1) {
        acc_g += __shfl_down_sync(0xffffffffu, acc_g, s);
        acc_u += __shfl_down_sync(0xffffffffu, acc_u, s);
    }
    if (lane == 0) {
        float g = acc_g + gate_bias[(size_t)e * ff + o];
        float u = acc_u + up_bias[(size_t)e * ff + o];
        float xg = fminf(g, limit);
        float yu = fminf(fmaxf(u, -limit), limit);
        out[(size_t)slot * ff + o] = (xg / (1.0f + expf(-alpha * xg))) * (yu + 1.0f);
    }
}

// dp4a version of the fused down + weighted mix + residual kernel. The per-expert
// swiglu output arrives pre-quantized (fused_q/fused_s, per-slot int8 + scale).
// Warp-per-output shape (see gate_up above): one warp owns one embd row across
// all active experts' ff/32 k-blocks, 8 outputs per 256-thread block.
// (A block-per-row reshape - grid embd, 256-thread tree reduce, the shape that
// fixed the Q8_0 decode GEMVs on 188-SM parts - measured slower here, 20.0 ->
// 23.4 us on that die: this grid already runs 8 warps/block with aligned 16B
// streams, and the per-block reduce+bias overhead amortizes over only ~1.4
// stride iterations. Fill is not this kernel's limiter.)
__global__ void pd_mxfp4_moe_down_dp4a_kernel(
    const unsigned char* __restrict__ down_data, const unsigned char* __restrict__ down_scale,
    const float* __restrict__ down_bias,
    const unsigned int* __restrict__ idx, const float* __restrict__ topk_w,
    const signed char* __restrict__ fused_q, const float* __restrict__ fused_s,
    float* __restrict__ residual, uint32_t ff, uint32_t embd, uint32_t n_active) {
    uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    uint32_t o = blockIdx.x * (blockDim.x >> 5) + warp;
    if (o >= embd) return;
    // batched grid: blockIdx.y = token; y = 0 zeroes every offset, keeping the
    // single-token launch bit-identical (see gate_up above). One writer per
    // (token, o) - plain residual store, deterministic at any batch.
    uint32_t t = blockIdx.y;
    idx += (size_t)t * n_active;
    topk_w += (size_t)t * n_active;
    fused_q += (size_t)t * n_active * ff;
    fused_s += (size_t)t * n_active * (ff >> 5);
    residual += (size_t)t * embd;
    uint32_t n_blocks = ff >> 5;

    // (A b-outer/slot-inner interchange measured NEUTRAL - 42.2 -> 43.2 µs -
    // so the simpler slot-outer walk stays; the inner i-unroll already gives the
    // memory system enough outstanding loads.)
    float acc = 0.0f;
    for (uint32_t slot = 0; slot < n_active; ++slot) {
        uint32_t e = idx[slot];
        float w = topk_w[slot];
        size_t blk_base = (size_t)((size_t)e * embd + o) * (ff / 32);
        const signed char* fq = fused_q + (size_t)slot * ff;
        const float* fs = fused_s + (size_t)slot * n_blocks;
        for (uint32_t b = lane; b < n_blocks; b += 32) {
            // one streamed 16B weight load per k-block (see gate_up above) -
            // bit-identical sums, fewer transactions
            uint4 dw = __ldcs(reinterpret_cast<const uint4*>(down_data + (blk_base + b) * 16));
            float dd = pd_e8m0_half(down_scale[blk_base + b]) * fs[b] * w;
            const int4 alo = *reinterpret_cast<const int4*>(fq + (size_t)b * 32);
            const int4 ahi = *reinterpret_cast<const int4*>(fq + (size_t)b * 32 + 16);
            int sumi = 0;
#pragma unroll
            for (uint32_t i = 0; i < 4; ++i) {
                int2 v = pd_fp4_unpack8((&dw.x)[i]);
                sumi = __dp4a(v.x, (&alo.x)[i], sumi);
                sumi = __dp4a(v.y, (&ahi.x)[i], sumi);
            }
            acc += dd * (float)sumi;
        }
    }
    for (uint32_t s = 16; s > 0; s >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, s);
    if (lane == 0) {
        float biasacc = 0.0f;
        for (uint32_t slot = 0; slot < n_active; ++slot)
            biasacc += topk_w[slot] * down_bias[(size_t)idx[slot] * embd + o];
        residual[o] += acc + biasacc;
    }
}

// Reverse routing map for the grouped MoE: slot_of[b][e] = the slot at which
// token b selected expert e, or 255 if it didn't. grid `batch`, one block per
// token. Lets the grouped kernels iterate tokens-per-expert without a sort.
__global__ void pd_moe_slot_map_kernel(const unsigned int* __restrict__ idx,
                                       unsigned char* __restrict__ slot_of,
                                       uint32_t n_active, uint32_t n_expert, uint32_t batch) {
    uint32_t b = blockIdx.x;
    if (b >= batch || threadIdx.x != 0) return;
    unsigned char* row = slot_of + (size_t)b * n_expert;
    for (uint32_t e = 0; e < n_expert; ++e) row[e] = 255;
    const unsigned int* ib = idx + (size_t)b * n_active;
    for (uint32_t s = 0; s < n_active; ++s) row[ib[s]] = (unsigned char)s;
}

// Grouped MoE gate+up+swiglu: grid (ff, n_expert). Block (o, e) dequants expert
// e's gate & up row o once into shared f32, then reuses it across every token
// that selected e (skipping the rest via slot_of). This is the weight-read +
// dequant amortization that makes batched MoE scale: an expert row is read once
// per step, not once per token. Output -> fused[b][slot][o] (matches down layout).
__global__ void pd_mxfp4_moe_gate_up_grouped_kernel(
    const unsigned char* __restrict__ gate_data, const unsigned char* __restrict__ gate_scale,
    const float* __restrict__ gate_bias,
    const unsigned char* __restrict__ up_data, const unsigned char* __restrict__ up_scale,
    const float* __restrict__ up_bias,
    const unsigned char* __restrict__ slot_of, const float* __restrict__ x,
    float* __restrict__ out, uint32_t in_dim, uint32_t ff, uint32_t n_expert,
    uint32_t n_active, uint32_t batch, float alpha, float limit) {
    uint32_t o = blockIdx.x, e = blockIdx.y;
    uint32_t tid = threadIdx.x, nth = blockDim.x;
    uint32_t n_blocks = in_dim >> 5;

    // Skip experts no token selected this step - avoids dequanting all 32 experts
    // at low batch (keeps grouped competitive with naive there, faster at high B).
    __shared__ int any_tok;
    if (tid == 0) {
        int a = 0;
        for (uint32_t b = 0; b < batch; ++b)
            if (slot_of[(size_t)b * n_expert + e] != 255) { a = 1; break; }
        any_tok = a;
    }
    __syncthreads();
    if (!any_tok) return;

    // Dequant expert e's gate & up row o into shared f32 once (reused across all
    // the batch's tokens below). Uses the A5 tricks that make the single-stream
    // MoE fast: shared 16-entry LUT (no constant-mem serialization), per-block
    // e8m0 scales precomputed once (not 16×/block), and one-thread-per-nibble-byte
    // so all 256 threads dequant (the old per-block loop left ~166 idle).
    extern __shared__ float smem[];
    float* lut = smem;              // [16] FP4 value table
    float* sg = lut + 16;           // dequanted gate row [in_dim]
    float* su = sg + in_dim;        // dequanted up row [in_dim]
    float* sgs = su + in_dim;       // [n_blocks] gate block scales
    float* sus = sgs + n_blocks;    // [n_blocks] up block scales
    float* red = sus + n_blocks;    // [2*nwarps] reduction scratch (g | u)
    if (tid < 16) lut[tid] = PD_FP4_VALUES[tid];

    size_t blk_base = (size_t)((size_t)e * ff + o) * (in_dim / 32);
    for (uint32_t bl = tid; bl < n_blocks; bl += nth) {
        sgs[bl] = pd_e8m0_half(gate_scale[blk_base + bl]);
        sus[bl] = pd_e8m0_half(up_scale[blk_base + bl]);
    }
    __syncthreads();
#pragma unroll 4
    for (uint32_t nb = tid; nb < n_blocks * 16u; nb += nth) {
        uint32_t blk = nb >> 4, p = nb & 15u;
        size_t doff = (blk_base + blk) * 16 + p;
        uint8_t pg = gate_data[doff];
        uint8_t pu = up_data[doff];
        uint32_t lo = (blk << 5) + p, hi = lo + 16u;
        float dg = sgs[blk], du = sus[blk];
        sg[lo] = dg * lut[pg & 0x0F];
        sg[hi] = dg * lut[pg >> 4];
        su[lo] = du * lut[pu & 0x0F];
        su[hi] = du * lut[pu >> 4];
    }
    __syncthreads();

    uint32_t warp = tid >> 5, lane = tid & 31u, nwarps = (nth + 31u) >> 5;
    for (uint32_t b = 0; b < batch; ++b) {
        uint32_t s = slot_of[(size_t)b * n_expert + e];
        if (s == 255) continue;               // token b didn't select expert e
        const float* xb = x + (size_t)b * in_dim;
        float ag = 0.0f, au = 0.0f;
        for (uint32_t i = tid; i < in_dim; i += nth) { ag += sg[i] * xb[i]; au += su[i] * xb[i]; }
        for (uint32_t r = 16; r > 0; r >>= 1) {
            ag += __shfl_down_sync(0xffffffffu, ag, r);
            au += __shfl_down_sync(0xffffffffu, au, r);
        }
        if (lane == 0) { red[warp] = ag; red[nwarps + warp] = au; }
        __syncthreads();
        if (tid == 0) {
            float g = gate_bias[(size_t)e * ff + o], u = up_bias[(size_t)e * ff + o];
            for (uint32_t w = 0; w < nwarps; ++w) { g += red[w]; u += red[nwarps + w]; }
            float xg = fminf(g, limit);
            float yu = fminf(fmaxf(u, -limit), limit);
            out[((size_t)b * n_active + s) * ff + o] = (xg / (1.0f + expf(-alpha * xg))) * (yu + 1.0f);
        }
        __syncthreads();
    }
}

// Grouped MoE down + weighted mix + residual add: grid (embd, n_expert). Block
// (o, e) dequants expert e's down row o once, reuses it across e's tokens, and
// atomic-adds each token's weighted contribution into residual[b][o]. `residual`
// [batch, embd] must already hold the post-attention hidden state - the expert
// mix accumulates on TOP of it (do not zero it, or the residual is lost).
__global__ void pd_mxfp4_moe_down_grouped_kernel(
    const unsigned char* __restrict__ down_data, const unsigned char* __restrict__ down_scale,
    const float* __restrict__ down_bias,
    const unsigned char* __restrict__ slot_of, const float* __restrict__ topk_w,
    const float* __restrict__ fused, float* __restrict__ residual, uint32_t ff, uint32_t embd,
    uint32_t n_expert, uint32_t n_active, uint32_t batch) {
    uint32_t o = blockIdx.x, e = blockIdx.y;
    uint32_t tid = threadIdx.x, nth = blockDim.x;
    uint32_t n_blocks = ff >> 5;

    // Skip experts no token selected this step (see gate_up_grouped).
    __shared__ int any_tok;
    if (tid == 0) {
        int a = 0;
        for (uint32_t b = 0; b < batch; ++b)
            if (slot_of[(size_t)b * n_expert + e] != 255) { a = 1; break; }
        any_tok = a;
    }
    __syncthreads();
    if (!any_tok) return;

    // Dequant expert e's down row o into shared f32 once (A5 tricks: shared LUT,
    // per-block scales precomputed once, one-thread-per-nibble-byte -> all threads
    // active). See pd_mxfp4_moe_gate_up_grouped_kernel.
    extern __shared__ float smem[];
    float* lut = smem;         // [16]
    float* sd = lut + 16;      // dequanted down row [ff]
    float* sc = sd + ff;       // [n_blocks] block scales
    float* red = sc + n_blocks;// [nwarps]
    if (tid < 16) lut[tid] = PD_FP4_VALUES[tid];

    size_t blk_base = (size_t)((size_t)e * embd + o) * (ff / 32);
    for (uint32_t bl = tid; bl < n_blocks; bl += nth)
        sc[bl] = pd_e8m0_half(down_scale[blk_base + bl]);
    __syncthreads();
#pragma unroll 4
    for (uint32_t nb = tid; nb < n_blocks * 16u; nb += nth) {
        uint32_t blk = nb >> 4, p = nb & 15u;
        uint8_t packed = down_data[(blk_base + blk) * 16 + p];
        uint32_t lo = (blk << 5) + p, hi = lo + 16u;
        float d = sc[blk];
        sd[lo] = d * lut[packed & 0x0F];
        sd[hi] = d * lut[packed >> 4];
    }
    __syncthreads();

    uint32_t warp = tid >> 5, lane = tid & 31u, nwarps = (nth + 31u) >> 5;
    for (uint32_t b = 0; b < batch; ++b) {
        uint32_t s = slot_of[(size_t)b * n_expert + e];
        if (s == 255) continue;
        const float* fb = fused + ((size_t)b * n_active + s) * ff;
        float acc = 0.0f;
        for (uint32_t i = tid; i < ff; i += nth) acc += sd[i] * fb[i];
        for (uint32_t r = 16; r > 0; r >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, r);
        if (lane == 0) red[warp] = acc;
        __syncthreads();
        if (tid == 0) {
            float v = down_bias[(size_t)e * embd + o];
            for (uint32_t w = 0; w < nwarps; ++w) v += red[w];
            float wt = topk_w[(size_t)b * n_active + s];
            atomicAdd(&residual[(size_t)b * embd + o], wt * v);
        }
        __syncthreads();
    }
}

// Tiled grouped MoE gate+up+swiglu (SGEMM shape). Replaces the one-output-per-block
// grouped kernel: each block computes a BN-wide output tile for one expert across
// its routed tokens, K-tiled so staged activations are reused across BN outputs and
// staged dequanted weights across BM tokens (register 2x2 micro-tiles) - high
// arithmetic intensity vs the memory-bound per-element kernel, and 32x fewer blocks.
// grid (ceil(ff/BN), n_expert); 256 threads = 16x16, each a 2x2 tile. BK=32 = one
// MXFP4 block per K-step. Output layout matches the grouped kernel (fused[(b*n_active
// +slot)*ff + o]) so the down kernel is unchanged.
#define PD_MOE_BM 32
#define PD_MOE_BN 32
#define PD_MOE_BK 32
// Sorted-GEMM output tile: BM stays 32 (tokens/expert; larger wastes on align
// padding), but BN grows to 128 so each of 256 threads owns a 4x4 register
// micro-tile (arithmetic intensity 2 vs the 2x2's 1). Thread grid 8x32.
#define PD_MOE_SBN 128
// Shared-tile leading dims, padded +1 so the store stride isn't a multiple of 32:
// the K-major layout (As[kk*BM+row], Bg[kk*SBN+col]) otherwise makes consecutive
// threads write the same bank -> a 32-way conflict (96% of store wavefronts).
// Reads stay broadcast/conflict-free. This is the sorted-GEMM's dominant cost.
#define PD_MOE_LDA (PD_MOE_BM + 1)
#define PD_MOE_LDB (PD_MOE_SBN + 1)
// Tensor-core (WMMA) variant: f16 A/B staged in shared with an 8-element pad so
// each 16×16 fragment row starts 16-byte-aligned and dodges shared bank conflicts.
// A is row-major [BM][BK], B is row-major [BK][SBN]. m16n16k16, f32 accumulate.
#define PD_MOE_TC_LDA (PD_MOE_BK + 8)
#define PD_MOE_TC_LDB (PD_MOE_SBN + 8)
__global__ void pd_mxfp4_moe_gate_up_gemm_kernel(
    const unsigned char* __restrict__ gate_W, const float* __restrict__ gate_bias,
    const unsigned char* __restrict__ up_W, const float* __restrict__ up_bias,
    const unsigned char* __restrict__ slot_of, const float* __restrict__ x,
    float* __restrict__ out, uint32_t in_dim, uint32_t ff, uint32_t n_expert,
    uint32_t n_active, uint32_t batch, float alpha, float limit) {
    uint32_t nt = blockIdx.x, e = blockIdx.y;
    uint32_t tid = threadIdx.x;
    uint32_t ty = tid >> 4, tx = tid & 15u;   // 16x16 thread grid
    uint32_t n_kblocks = in_dim >> 5;

    extern __shared__ float smem[];
    float* lut = smem;                                  // [16]
    float* As = lut + 16;                               // [BM*BK]
    float* Bg = As + PD_MOE_BM * PD_MOE_BK;             // [BN*BK] gate
    float* Bu = Bg + PD_MOE_BN * PD_MOE_BK;             // [BN*BK] up
    int* gb = (int*)(Bu + PD_MOE_BN * PD_MOE_BK);      // [batch] token index
    int* gs = gb + batch;                               // [batch] its slot
    __shared__ int T;

    if (tid < 16) lut[tid] = PD_FP4_VALUES[tid];
    // gather this expert's routed tokens (one thread; tiny vs the GEMM)
    if (tid == 0) {
        int cnt = 0;
        for (uint32_t b = 0; b < batch; ++b) {
            unsigned char s = slot_of[(size_t)b * n_expert + e];
            if (s != 255) { gb[cnt] = (int)b; gs[cnt] = (int)s; ++cnt; }
        }
        T = cnt;
    }
    __syncthreads();
    if (T == 0) return;

    uint32_t o0 = nt * PD_MOE_BN;
    for (uint32_t c = 0; c < (uint32_t)T; c += PD_MOE_BM) {
        float ag[2][2] = {{0, 0}, {0, 0}}, au[2][2] = {{0, 0}, {0, 0}};
        for (uint32_t kb = 0; kb < n_kblocks; ++kb) {
            uint32_t kbase = kb << 5;
            // stage A [BM x BK] (activations; padding rows = 0)
            for (uint32_t idx = tid; idx < PD_MOE_BM * PD_MOE_BK; idx += 256) {
                uint32_t row = idx / PD_MOE_BK, kk = idx % PD_MOE_BK;
                uint32_t ti = c + row;
                As[idx] = (ti < (uint32_t)T) ? x[(size_t)gb[ti] * in_dim + kbase + kk] : 0.0f;
            }
            // stage + dequant B gate/up [BN x BK] (one MXFP4 block per row)
            for (uint32_t idx = tid; idx < PD_MOE_BN * PD_MOE_BK; idx += 256) {
                uint32_t row = idx / PD_MOE_BK, kk = idx % PD_MOE_BK;
                uint32_t o = o0 + row;
                float vg = 0.0f, vu = 0.0f;
                if (o < ff) {
                    size_t bo = ((size_t)((size_t)e * ff + o) * in_dim / 32 + kb) * 17;
                    const unsigned char* gp = gate_W + bo;
                    const unsigned char* up = up_W + bo;
                    unsigned char bgb = gp[1 + (kk & 15u)], bub = up[1 + (kk & 15u)];
                    uint32_t ng = (kk < 16) ? (bgb & 0x0F) : (bgb >> 4);
                    uint32_t nu = (kk < 16) ? (bub & 0x0F) : (bub >> 4);
                    vg = pd_e8m0_half(gp[0]) * lut[ng];
                    vu = pd_e8m0_half(up[0]) * lut[nu];
                }
                Bg[idx] = vg;
                Bu[idx] = vu;
            }
            __syncthreads();
            for (uint32_t kk = 0; kk < PD_MOE_BK; ++kk) {
                float a0 = As[ty * PD_MOE_BK + kk], a1 = As[(ty + 16) * PD_MOE_BK + kk];
                float g0 = Bg[tx * PD_MOE_BK + kk], g1 = Bg[(tx + 16) * PD_MOE_BK + kk];
                float u0 = Bu[tx * PD_MOE_BK + kk], u1 = Bu[(tx + 16) * PD_MOE_BK + kk];
                ag[0][0] += a0 * g0; ag[0][1] += a0 * g1; ag[1][0] += a1 * g0; ag[1][1] += a1 * g1;
                au[0][0] += a0 * u0; au[0][1] += a0 * u1; au[1][0] += a1 * u0; au[1][1] += a1 * u1;
            }
            __syncthreads();
        }
        // swiglu epilogue -> scatter to fused[(b*n_active+slot)*ff + o]
        for (uint32_t i = 0; i < 2; ++i) {
            uint32_t ti = c + ty + i * 16;
            if (ti >= (uint32_t)T) continue;
            uint32_t bt = (uint32_t)gb[ti], st = (uint32_t)gs[ti];
            for (uint32_t j = 0; j < 2; ++j) {
                uint32_t o = o0 + tx + j * 16;
                if (o >= ff) continue;
                float g = ag[i][j] + gate_bias[(size_t)e * ff + o];
                float u = au[i][j] + up_bias[(size_t)e * ff + o];
                float xg = fminf(g, limit);
                float yu = fminf(fmaxf(u, -limit), limit);
                out[((size_t)bt * n_active + st) * ff + o] =
                    (xg / (1.0f + expf(-alpha * xg))) * (yu + 1.0f);
            }
        }
    }
}

#define PD_MOE_PAD 0xFFFFFFFFu

// Repack MXFP4 from the on-disk 17-byte block (1 e8m0 scale + 16 data bytes) into
// two aligned streams: `dst_data` (16 bytes/block, 16-aligned -> a coalesced load)
// and `dst_scale` (1 byte/block, contiguous). The 17-byte stride otherwise misaligns
// every data read (11.5/32 sectors); this makes the sorted-GEMM weight load
// coalesced. Run once at load, per expert-weight tensor.
__global__ void pd_mxfp4_repack_kernel(const unsigned char* __restrict__ src,
                                       unsigned char* __restrict__ dst_data,
                                       unsigned char* __restrict__ dst_scale, uint64_t n_blocks) {
    uint64_t blk = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (blk >= n_blocks) return;
    const unsigned char* s = src + blk * 17;
    dst_scale[blk] = s[0];
    unsigned char* d = dst_data + blk * 16;
#pragma unroll
    for (int i = 0; i < 16; ++i) d[i] = s[1 + i];
}

// Group the MoE token-expert pairs by expert into contiguous, BM-padded blocks -
// the "moe_align" pass. Turns the scattered routing (idx[rows,n_active]) into a
// sorted layout the tiled GEMM reads directly: no per-block routing scan, tokens
// contiguous per expert, M-padding only the per-expert tail. Single block.
//   sorted_row[max_blocks*BM]  : source token row per entry (PD_MOE_PAD = padding)
//   sorted_slot[max_blocks*BM] : which of the token's n_active picks (topk_w/output)
//   block_expert[max_blocks]   : expert id per BM-block (PD_MOE_PAD = unused block)
__global__ void pd_moe_align_kernel(
    const unsigned int* __restrict__ idx, unsigned int* __restrict__ sorted_row,
    unsigned int* __restrict__ sorted_slot, unsigned int* __restrict__ block_expert,
    uint32_t rows, uint32_t n_active, uint32_t n_expert, uint32_t bm, uint32_t max_blocks) {
    // histogram -> block-wide scan -> scatter. The first cut had one thread
    // serially walk all experts for the prefix AND the block_expert writes -
    // 21.7 us at c32/128e with the whole block idle
    // behind ~256 dependent global writes. Now: warp-shuffle exclusive scan
    // of nb = ceil(count/bm) (chunked over blockDim windows so any n_expert
    // works), per-expert parallel block_expert writes, and PAD-fill of only
    // the used entry region (the tail past bacc*bm is never read - its
    // block_expert says PAD and every consumer early-outs on that).
    extern __shared__ unsigned int ash[];
    unsigned int* count = ash;              // [n_expert]
    unsigned int* boff = count + n_expert;  // [n_expert] first block index
    unsigned int* fill = boff + n_expert;   // [n_expert] running scatter counter
    __shared__ unsigned int wsum[32];
    __shared__ unsigned int carry_sh;       // running block total across windows
    uint32_t tid = threadIdx.x, nth = blockDim.x;
    uint32_t lane = tid & 31u, warp = tid >> 5, nwarp = (nth + 31u) >> 5;
    uint32_t npairs = rows * n_active;

    for (uint32_t e = tid; e < n_expert; e += nth) { count[e] = 0; fill[e] = 0; }
    if (tid == 0) carry_sh = 0;
    __syncthreads();
    for (uint32_t p = tid; p < npairs; p += nth) atomicAdd(&count[idx[p]], 1u);
    __syncthreads();

    for (uint32_t base = 0; base < n_expert; base += nth) {
        uint32_t e = base + tid;
        uint32_t nb = (e < n_expert) ? (count[e] + bm - 1u) / bm : 0u;
        uint32_t x = nb;   // warp-inclusive scan
        #pragma unroll
        for (uint32_t d = 1; d < 32u; d <<= 1) {
            uint32_t v = __shfl_up_sync(0xffffffffu, x, d);
            if (lane >= d) x += v;
        }
        if (lane == 31u) wsum[warp] = x;
        __syncthreads();
        if (warp == 0) {
            uint32_t w = (lane < nwarp) ? wsum[lane] : 0u;
            #pragma unroll
            for (uint32_t d = 1; d < 32u; d <<= 1) {
                uint32_t v = __shfl_up_sync(0xffffffffu, w, d);
                if (lane >= d) w += v;
            }
            wsum[lane] = w;
        }
        __syncthreads();
        uint32_t excl = x - nb + (warp ? wsum[warp - 1u] : 0u) + carry_sh;
        if (e < n_expert) boff[e] = excl;
        __syncthreads();
        if (tid == nth - 1u) carry_sh = excl + nb;  // window's inclusive total
        __syncthreads();
    }
    const uint32_t bacc = carry_sh;

    for (uint32_t e = tid; e < n_expert; e += nth) {
        uint32_t nb = (count[e] + bm - 1u) / bm, b0 = boff[e];
        for (uint32_t b = 0; b < nb; ++b) block_expert[b0 + b] = e;
    }
    for (uint32_t b = bacc + tid; b < max_blocks; b += nth) block_expert[b] = PD_MOE_PAD;
    for (uint32_t i = tid; i < bacc * bm; i += nth) sorted_row[i] = PD_MOE_PAD;
    __syncthreads();
    for (uint32_t p = tid; p < npairs; p += nth) {
        unsigned int e = idx[p];
        unsigned int pos = boff[e] * bm + atomicAdd(&fill[e], 1u);
        sorted_row[pos] = p / n_active;
        sorted_slot[pos] = p % n_active;
    }
}

// Sorted tiled MoE gate+up+swiglu: like pd_mxfp4_moe_gate_up_gemm but reads the
// moe_align layout (contiguous tokens per block, expert from block_expert) instead
// of scanning slot_of - kills the redundant per-block gather. Writes swiglu output
// contiguously to fused_sorted[(blk*BM+row)*ff + o] (down reads it directly).
// grid (ceil(ff/BN), max_blocks).
__global__ void pd_mxfp4_moe_gate_up_gemm_sorted_kernel(
    const unsigned char* __restrict__ gate_data, const unsigned char* __restrict__ gate_scale,
    const float* __restrict__ gate_bias,
    const unsigned char* __restrict__ up_data, const unsigned char* __restrict__ up_scale,
    const float* __restrict__ up_bias,
    const unsigned int* __restrict__ sorted_row, const unsigned int* __restrict__ block_expert,
    const float* __restrict__ x, float* __restrict__ fused_sorted,
    uint32_t in_dim, uint32_t ff, float alpha, float limit) {
    uint32_t nt = blockIdx.x, blk = blockIdx.y;
    uint32_t e = block_expert[blk];
    if (e == PD_MOE_PAD) return;
    uint32_t tid = threadIdx.x, ry = tid >> 5, rx = tid & 31u;   // 8x32; each owns 4x4
    uint32_t n_kblocks = in_dim >> 5;

    extern __shared__ char csmem[];
    float* lut = (float*)csmem;
    float* As = lut + 16;                                 // [BK][LDA] f32 (K-major, padded)
    __half* Bg = (__half*)(As + PD_MOE_BK * PD_MOE_LDA);  // [BK][LDB] f16
    __half* Bu = Bg + PD_MOE_BK * PD_MOE_LDB;             // [BK][LDB] f16
    __shared__ unsigned int tok[PD_MOE_BM];
    if (tid < 16) lut[tid] = PD_FP4_VALUES[tid];
    if (tid < PD_MOE_BM) tok[tid] = sorted_row[(size_t)blk * PD_MOE_BM + tid];
    __syncthreads();

    uint32_t o0 = nt * PD_MOE_SBN;
    float ag[4][4], au[4][4];
    for (uint32_t i = 0; i < 4; ++i)
        for (uint32_t j = 0; j < 4; ++j) { ag[i][j] = 0.0f; au[i][j] = 0.0f; }

    for (uint32_t kb = 0; kb < n_kblocks; ++kb) {
        uint32_t kbase = kb << 5;
        for (uint32_t idx = tid; idx < PD_MOE_BM * PD_MOE_BK; idx += 256) {
            uint32_t row = idx / PD_MOE_BK, kk = idx % PD_MOE_BK;
            uint32_t token = tok[row];
            As[kk * PD_MOE_LDA + row] =
                (token != PD_MOE_PAD) ? x[(size_t)token * in_dim + kbase + kk] : 0.0f;
        }
        // one thread per DATA BYTE (not per nibble): read the byte once, emit both
        // nibbles (K-positions b and b+16). Halves the redundant byte reads (the two
        // nibble halves of a block previously loaded the same 16 bytes twice) and
        // reads 16 consecutive bytes per block instead of a scattered pattern.
        for (uint32_t idx = tid; idx < PD_MOE_SBN * 16u; idx += 256) {
            uint32_t col = idx / 16u, b = idx & 15u;
            uint32_t o = o0 + col;
            float g0 = 0.0f, g1 = 0.0f, u0 = 0.0f, u1 = 0.0f;
            if (o < ff) {
                size_t blk = (size_t)((size_t)e * ff + o) * (in_dim / 32) + kb;
                float sg = pd_e8m0_half(gate_scale[blk]), su = pd_e8m0_half(up_scale[blk]);
                unsigned char bgb = gate_data[blk * 16 + b], bub = up_data[blk * 16 + b];
                g0 = sg * lut[bgb & 0x0F]; g1 = sg * lut[bgb >> 4];
                u0 = su * lut[bub & 0x0F]; u1 = su * lut[bub >> 4];
            }
            Bg[b * PD_MOE_LDB + col] = __float2half(g0);
            Bg[(b + 16) * PD_MOE_LDB + col] = __float2half(g1);
            Bu[b * PD_MOE_LDB + col] = __float2half(u0);
            Bu[(b + 16) * PD_MOE_LDB + col] = __float2half(u1);
        }
        __syncthreads();
        for (uint32_t kk = 0; kk < PD_MOE_BK; ++kk) {
            float a[4], bg[4], bu[4];
#pragma unroll
            for (uint32_t i = 0; i < 4; ++i) a[i] = As[kk * PD_MOE_LDA + ry * 4 + i];
#pragma unroll
            for (uint32_t j = 0; j < 4; ++j) {
                // strided columns (rx + j*32, not rx*4+j): shared reads hit bank rx
                // (conflict-free) and the epilogue store below is warp-coalesced.
                bg[j] = __half2float(Bg[kk * PD_MOE_LDB + rx + j * 32]);
                bu[j] = __half2float(Bu[kk * PD_MOE_LDB + rx + j * 32]);
            }
#pragma unroll
            for (uint32_t i = 0; i < 4; ++i)
#pragma unroll
                for (uint32_t j = 0; j < 4; ++j) {
                    ag[i][j] += a[i] * bg[j];
                    au[i][j] += a[i] * bu[j];
                }
        }
        __syncthreads();
    }
    for (uint32_t i = 0; i < 4; ++i) {
        uint32_t row = ry * 4 + i;
        for (uint32_t j = 0; j < 4; ++j) {
            uint32_t o = o0 + rx + j * 32;
            if (o >= ff) continue;
            float g = ag[i][j] + gate_bias[(size_t)e * ff + o];
            float u = au[i][j] + up_bias[(size_t)e * ff + o];
            float xg = fminf(g, limit);
            float yu = fminf(fmaxf(u, -limit), limit);
            fused_sorted[((size_t)blk * PD_MOE_BM + row) * ff + o] =
                (xg / (1.0f + expf(-alpha * xg))) * (yu + 1.0f);
        }
    }
}

// Tensor-core (WMMA) variant of the sorted gate+up+swiglu GEMM. Same tile shape
// (BM×SBN output, BK-chunked over in_dim) and same MXFP4->f16 shared staging, but
// the accumulate runs on the FP16 tensor cores (m16n16k16, f32 acc) instead of the
// 4×4 CUDA-core micro-tile. 256 threads = 8 warps; the BM×SBN=32×128 output is 2
// M-tiles × 8 N-tiles, one N-tile per warp (both M-tiles). Grid (ceil(ff/SBN),
// max_blocks). Requires sm_70+ (WMMA); on Ampere the FP16 cores run ~2× the FMA
// rate. M is often padding at decode, so the win is largest at prefill/high batch.
__global__ void pd_mxfp4_moe_gate_up_gemm_sorted_tc_kernel(
    const unsigned char* __restrict__ gate_data, const unsigned char* __restrict__ gate_scale,
    const float* __restrict__ gate_bias,
    const unsigned char* __restrict__ up_data, const unsigned char* __restrict__ up_scale,
    const float* __restrict__ up_bias,
    const unsigned int* __restrict__ sorted_row, const unsigned int* __restrict__ block_expert,
    const float* __restrict__ x, float* __restrict__ fused_sorted,
    uint32_t in_dim, uint32_t ff, float alpha, float limit) {
    uint32_t nt = blockIdx.x, blk = blockIdx.y;
    uint32_t e = block_expert[blk];
    if (e == PD_MOE_PAD) return;
    uint32_t tid = threadIdx.x, warp = tid >> 5, lane = tid & 31u;
    uint32_t n_kblocks = in_dim >> 5;

    extern __shared__ char csmem[];
    float* lut = (float*)csmem;
    __half* As = (__half*)(lut + 16);                       // [BM][TC_LDA] f16, row-major
    __half* Bg = As + PD_MOE_BM * PD_MOE_TC_LDA;            // [BK][TC_LDB] f16, row-major
    __half* Bu = Bg + PD_MOE_BK * PD_MOE_TC_LDB;            // [BK][TC_LDB] f16
    __shared__ unsigned int tok[PD_MOE_BM];
    if (tid < 16) lut[tid] = PD_FP4_VALUES[tid];
    if (tid < PD_MOE_BM) tok[tid] = sorted_row[(size_t)blk * PD_MOE_BM + tid];
    __syncthreads();

    uint32_t o0 = nt * PD_MOE_SBN;
    uint32_t nw = warp * 16;                                // this warp's N-tile column in [0,SBN)
    // acc[m] for gate/up, m = 0 (rows 0..15) and 1 (rows 16..31)
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_g[2], acc_u[2];
    for (int m = 0; m < 2; ++m) {
        wmma::fill_fragment(acc_g[m], 0.0f);
        wmma::fill_fragment(acc_u[m], 0.0f);
    }

    for (uint32_t kb = 0; kb < n_kblocks; ++kb) {
        uint32_t kbase = kb << 5;
        // A tile: row-major [BM][BK], f16. rows are tokens (0 for padding).
        for (uint32_t idx = tid; idx < PD_MOE_BM * PD_MOE_BK; idx += 256) {
            uint32_t row = idx / PD_MOE_BK, kk = idx % PD_MOE_BK;
            uint32_t token = tok[row];
            float v = (token != PD_MOE_PAD) ? x[(size_t)token * in_dim + kbase + kk] : 0.0f;
            As[row * PD_MOE_TC_LDA + kk] = __float2half(v);
        }
        // B tiles: dequant MXFP4 -> f16 into row-major [BK][SBN] (one thread per data
        // byte, both nibbles -> K-positions b and b+16). Same as the CUDA-core kernel.
        for (uint32_t idx = tid; idx < PD_MOE_SBN * 16u; idx += 256) {
            uint32_t col = idx / 16u, b = idx & 15u;
            uint32_t o = o0 + col;
            float g0 = 0.0f, g1 = 0.0f, u0 = 0.0f, u1 = 0.0f;
            if (o < ff) {
                size_t wb = (size_t)((size_t)e * ff + o) * (in_dim / 32) + kb;
                float sg = pd_e8m0_half(gate_scale[wb]), su = pd_e8m0_half(up_scale[wb]);
                unsigned char bgb = gate_data[wb * 16 + b], bub = up_data[wb * 16 + b];
                g0 = sg * lut[bgb & 0x0F]; g1 = sg * lut[bgb >> 4];
                u0 = su * lut[bub & 0x0F]; u1 = su * lut[bub >> 4];
            }
            Bg[b * PD_MOE_TC_LDB + col] = __float2half(g0);
            Bg[(b + 16) * PD_MOE_TC_LDB + col] = __float2half(g1);
            Bu[b * PD_MOE_TC_LDB + col] = __float2half(u0);
            Bu[(b + 16) * PD_MOE_TC_LDB + col] = __float2half(u1);
        }
        __syncthreads();
        // accumulate over the two 16-wide K sub-tiles of this BK=32 chunk
        for (uint32_t k0 = 0; k0 < PD_MOE_BK; k0 += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> fbg, fbu;
            wmma::load_matrix_sync(fa[0], As + 0 * PD_MOE_TC_LDA + k0, PD_MOE_TC_LDA);
            wmma::load_matrix_sync(fa[1], As + 16 * PD_MOE_TC_LDA + k0, PD_MOE_TC_LDA);
            wmma::load_matrix_sync(fbg, Bg + k0 * PD_MOE_TC_LDB + nw, PD_MOE_TC_LDB);
            wmma::load_matrix_sync(fbu, Bu + k0 * PD_MOE_TC_LDB + nw, PD_MOE_TC_LDB);
            for (int m = 0; m < 2; ++m) {
                wmma::mma_sync(acc_g[m], fa[m], fbg, acc_g[m]);
                wmma::mma_sync(acc_u[m], fa[m], fbu, acc_u[m]);
            }
        }
        __syncthreads();   // done reading Bg/Bu for this kb before next overwrite / epilogue
    }

    // Epilogue: reuse the Bg/Bu region as per-warp f32 scratch (accumulation done).
    // Store each acc tile to shared, then apply bias + swiglu and scatter to output.
    __syncthreads();
    float* epi = (float*)Bg + warp * (2 * 256);             // [gate 16×16][up 16×16] per warp
    for (int m = 0; m < 2; ++m) {
        wmma::store_matrix_sync(epi, acc_g[m], 16, wmma::mem_row_major);
        wmma::store_matrix_sync(epi + 256, acc_u[m], 16, wmma::mem_row_major);
        __syncwarp();
        for (uint32_t el = lane; el < 256; el += 32) {
            uint32_t r = el >> 4, c = el & 15u;             // 16×16
            uint32_t o = o0 + nw + c;
            uint32_t row = m * 16 + r;
            uint32_t token = tok[row];
            if (token == PD_MOE_PAD || o >= ff) continue;
            float g = epi[el] + gate_bias[(size_t)e * ff + o];
            float u = epi[256 + el] + up_bias[(size_t)e * ff + o];
            float xg = fminf(g, limit);
            float yu = fminf(fmaxf(u, -limit), limit);
            fused_sorted[((size_t)blk * PD_MOE_BM + row) * ff + o] =
                (xg / (1.0f + expf(-alpha * xg))) * (yu + 1.0f);
        }
        __syncwarp();
    }
}

// Sorted tiled MoE down + weighted mix + residual add. Reads fused_sorted (from the
// sorted gate_up) contiguously, GEMMs against the expert's down weight, then
// scatter-adds each real token's weighted result into residual[token][o] (atomic).
// residual must already hold the post-attention hidden state. grid (ceil(embd/BN),
// max_blocks).
__global__ void pd_mxfp4_moe_down_gemm_sorted_kernel(
    const unsigned char* __restrict__ down_data, const unsigned char* __restrict__ down_scale,
    const float* __restrict__ down_bias,
    const unsigned int* __restrict__ sorted_row, const unsigned int* __restrict__ sorted_slot,
    const unsigned int* __restrict__ block_expert, const float* __restrict__ topk_w,
    const float* __restrict__ fused_sorted, float* __restrict__ residual,
    uint32_t ff, uint32_t embd, uint32_t n_active) {
    uint32_t nt = blockIdx.x, blk = blockIdx.y;
    uint32_t e = block_expert[blk];
    if (e == PD_MOE_PAD) return;
    uint32_t tid = threadIdx.x, ry = tid >> 5, rx = tid & 31u;   // 8x32; each owns 4x4
    uint32_t n_kblocks = ff >> 5;

    extern __shared__ char csmem[];
    float* lut = (float*)csmem;
    float* As = lut + 16;                                // fused rows [BK][LDA] f32 (padded)
    __half* Bd = (__half*)(As + PD_MOE_BK * PD_MOE_LDA); // down weight [BK][LDB] f16
    __shared__ unsigned int tok[PD_MOE_BM], slt[PD_MOE_BM];
    if (tid < 16) lut[tid] = PD_FP4_VALUES[tid];
    if (tid < PD_MOE_BM) {
        tok[tid] = sorted_row[(size_t)blk * PD_MOE_BM + tid];
        slt[tid] = sorted_slot[(size_t)blk * PD_MOE_BM + tid];
    }
    __syncthreads();

    uint32_t o0 = nt * PD_MOE_SBN;
    float ad[4][4];
    for (uint32_t i = 0; i < 4; ++i)
        for (uint32_t j = 0; j < 4; ++j) ad[i][j] = 0.0f;

    for (uint32_t kb = 0; kb < n_kblocks; ++kb) {
        uint32_t kbase = kb << 5;
        for (uint32_t idx = tid; idx < PD_MOE_BM * PD_MOE_BK; idx += 256) {
            uint32_t row = idx / PD_MOE_BK, kk = idx % PD_MOE_BK;
            As[kk * PD_MOE_LDA + row] =
                fused_sorted[((size_t)blk * PD_MOE_BM + row) * ff + kbase + kk];
        }
        // one thread per data byte, both nibbles (see gate_up)
        for (uint32_t idx = tid; idx < PD_MOE_SBN * 16u; idx += 256) {
            uint32_t col = idx / 16u, b = idx & 15u;
            uint32_t o = o0 + col;
            float d0 = 0.0f, d1 = 0.0f;
            if (o < embd) {
                size_t blkb = (size_t)((size_t)e * embd + o) * (ff / 32) + kb;
                float sd = pd_e8m0_half(down_scale[blkb]);
                unsigned char db = down_data[blkb * 16 + b];
                d0 = sd * lut[db & 0x0F]; d1 = sd * lut[db >> 4];
            }
            Bd[b * PD_MOE_LDB + col] = __float2half(d0);
            Bd[(b + 16) * PD_MOE_LDB + col] = __float2half(d1);
        }
        __syncthreads();
        for (uint32_t kk = 0; kk < PD_MOE_BK; ++kk) {
            float a[4], d[4];
#pragma unroll
            for (uint32_t i = 0; i < 4; ++i) a[i] = As[kk * PD_MOE_LDA + ry * 4 + i];
#pragma unroll
            for (uint32_t j = 0; j < 4; ++j) d[j] = __half2float(Bd[kk * PD_MOE_LDB + rx + j * 32]);
#pragma unroll
            for (uint32_t i = 0; i < 4; ++i)
#pragma unroll
                for (uint32_t j = 0; j < 4; ++j) ad[i][j] += a[i] * d[j];
        }
        __syncthreads();
    }
    for (uint32_t i = 0; i < 4; ++i) {
        uint32_t row = ry * 4 + i;
        uint32_t token = tok[row];
        if (token == PD_MOE_PAD) continue;
        float w = topk_w[(size_t)token * n_active + slt[row]];
        for (uint32_t j = 0; j < 4; ++j) {
            uint32_t o = o0 + rx + j * 32;
            if (o >= embd) continue;
            float v = ad[i][j] + down_bias[(size_t)e * embd + o];
            atomicAdd(&residual[(size_t)token * embd + o], w * v);
        }
    }
}

// Tensor-core (WMMA) variant of the sorted down GEMM. Same tile + MXFP4->f16 shared
// staging as the CUDA-core kernel, but the accumulate runs on the FP16 tensor cores
// (m16n16k16, f32 acc). One B (down weight), so one acc set per M-tile; the epilogue
// weights each real token's output by topk_w and atomic-adds into the residual.
// 8 warps = 2 M-tiles × 8 N-tiles, one N-tile/warp. Grid (ceil(embd/SBN), max_blocks).
__global__ void pd_mxfp4_moe_down_gemm_sorted_tc_kernel(
    const unsigned char* __restrict__ down_data, const unsigned char* __restrict__ down_scale,
    const float* __restrict__ down_bias,
    const unsigned int* __restrict__ sorted_row, const unsigned int* __restrict__ sorted_slot,
    const unsigned int* __restrict__ block_expert, const float* __restrict__ topk_w,
    const float* __restrict__ fused_sorted, float* __restrict__ residual,
    uint32_t ff, uint32_t embd, uint32_t n_active) {
    uint32_t nt = blockIdx.x, blk = blockIdx.y;
    uint32_t e = block_expert[blk];
    if (e == PD_MOE_PAD) return;
    uint32_t tid = threadIdx.x, warp = tid >> 5, lane = tid & 31u;
    uint32_t n_kblocks = ff >> 5;

    extern __shared__ char csmem[];
    float* lut = (float*)csmem;
    __half* As = (__half*)(lut + 16);                       // [BM][TC_LDA] f16, row-major
    __half* Bd = As + PD_MOE_BM * PD_MOE_TC_LDA;            // [BK][TC_LDB] f16, row-major
    __shared__ unsigned int tok[PD_MOE_BM], slt[PD_MOE_BM];
    if (tid < 16) lut[tid] = PD_FP4_VALUES[tid];
    if (tid < PD_MOE_BM) {
        tok[tid] = sorted_row[(size_t)blk * PD_MOE_BM + tid];
        slt[tid] = sorted_slot[(size_t)blk * PD_MOE_BM + tid];
    }
    __syncthreads();

    uint32_t o0 = nt * PD_MOE_SBN;
    uint32_t nw = warp * 16;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2];
    wmma::fill_fragment(acc[0], 0.0f);
    wmma::fill_fragment(acc[1], 0.0f);

    for (uint32_t kb = 0; kb < n_kblocks; ++kb) {
        uint32_t kbase = kb << 5;
        for (uint32_t idx = tid; idx < PD_MOE_BM * PD_MOE_BK; idx += 256) {
            uint32_t row = idx / PD_MOE_BK, kk = idx % PD_MOE_BK;
            As[row * PD_MOE_TC_LDA + kk] =
                __float2half(fused_sorted[((size_t)blk * PD_MOE_BM + row) * ff + kbase + kk]);
        }
        for (uint32_t idx = tid; idx < PD_MOE_SBN * 16u; idx += 256) {
            uint32_t col = idx / 16u, b = idx & 15u;
            uint32_t o = o0 + col;
            float d0 = 0.0f, d1 = 0.0f;
            if (o < embd) {
                size_t blkb = (size_t)((size_t)e * embd + o) * (ff / 32) + kb;
                float sd = pd_e8m0_half(down_scale[blkb]);
                unsigned char db = down_data[blkb * 16 + b];
                d0 = sd * lut[db & 0x0F]; d1 = sd * lut[db >> 4];
            }
            Bd[b * PD_MOE_TC_LDB + col] = __float2half(d0);
            Bd[(b + 16) * PD_MOE_TC_LDB + col] = __float2half(d1);
        }
        __syncthreads();
        for (uint32_t k0 = 0; k0 < PD_MOE_BK; k0 += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> fbd;
            wmma::load_matrix_sync(fa[0], As + 0 * PD_MOE_TC_LDA + k0, PD_MOE_TC_LDA);
            wmma::load_matrix_sync(fa[1], As + 16 * PD_MOE_TC_LDA + k0, PD_MOE_TC_LDA);
            wmma::load_matrix_sync(fbd, Bd + k0 * PD_MOE_TC_LDB + nw, PD_MOE_TC_LDB);
            wmma::mma_sync(acc[0], fa[0], fbd, acc[0]);
            wmma::mma_sync(acc[1], fa[1], fbd, acc[1]);
        }
        __syncthreads();
    }

    // Epilogue: store each acc tile to a per-warp f32 scratch (reusing Bd), weight
    // by topk_w, and atomic-add each real token's contribution into the residual.
    __syncthreads();
    float* epi = (float*)Bd + warp * 256;
    for (int m = 0; m < 2; ++m) {
        wmma::store_matrix_sync(epi, acc[m], 16, wmma::mem_row_major);
        __syncwarp();
        for (uint32_t el = lane; el < 256; el += 32) {
            uint32_t r = el >> 4, c = el & 15u;
            uint32_t o = o0 + nw + c;
            uint32_t row = m * 16 + r;
            uint32_t token = tok[row];
            if (token == PD_MOE_PAD || o >= embd) continue;
            float wgt = topk_w[(size_t)token * n_active + slt[row]];
            float v = epi[el] + down_bias[(size_t)e * embd + o];
            atomicAdd(&residual[(size_t)token * embd + o], wgt * v);
        }
        __syncwarp();
    }
}

// Batched fused MoE gate+up+swiglu. grid (ff, n_active, batch): block (o, slot, b)
// drives expert idx[b][slot] for token b's activation x[b]. Correct batched MoE
// (each block reads its expert row; expert-overlap amortization is a follow-up).
__global__ void pd_mxfp4_moe_gate_up_batch_kernel(
    const unsigned char* __restrict__ gate_W, const float* __restrict__ gate_bias,
    const unsigned char* __restrict__ up_W, const float* __restrict__ up_bias,
    const unsigned int* __restrict__ idx, const float* __restrict__ x,
    float* __restrict__ out, uint32_t in_dim, uint32_t ff, uint32_t n_active,
    float alpha, float limit) {
    uint32_t o = blockIdx.x, slot = blockIdx.y, b = blockIdx.z;
    uint32_t e = idx[b * n_active + slot];
    uint32_t tid = threadIdx.x, nth = blockDim.x;
    uint32_t n_blocks = in_dim >> 5;
    __shared__ float wg[32], wu[32];
    const float* xb = x + (size_t)b * in_dim;

    size_t base = ((size_t)((size_t)e * ff + o) * in_dim / 32) * 17;
    float acc_g = 0.0f, acc_u = 0.0f;
    for (uint32_t bl = tid; bl < n_blocks; bl += nth) {
        const unsigned char* gp = gate_W + base + (size_t)bl * 17;
        const unsigned char* upp = up_W + base + (size_t)bl * 17;
        float dg = pd_e8m0_half(gp[0]), du = pd_e8m0_half(upp[0]);
        uint32_t base_i = bl << 5;
        for (uint32_t j = 0; j < 16; ++j) {
            uint8_t pg = gp[1 + j], pu = upp[1 + j];
            float xl = xb[base_i + j], xh = xb[base_i + j + 16];
            acc_g += dg * (PD_FP4_VALUES[pg & 0x0F] * xl + PD_FP4_VALUES[pg >> 4] * xh);
            acc_u += du * (PD_FP4_VALUES[pu & 0x0F] * xl + PD_FP4_VALUES[pu >> 4] * xh);
        }
    }
    for (uint32_t s = 16; s > 0; s >>= 1) {
        acc_g += __shfl_down_sync(0xffffffffu, acc_g, s);
        acc_u += __shfl_down_sync(0xffffffffu, acc_u, s);
    }
    uint32_t warp = tid >> 5, lane = tid & 31u;
    if (lane == 0) { wg[warp] = acc_g; wu[warp] = acc_u; }
    __syncthreads();
    if (tid == 0) {
        float g = 0.0f, u = 0.0f;
        uint32_t nwarps = (nth + 31u) >> 5;
        for (uint32_t w = 0; w < nwarps; ++w) { g += wg[w]; u += wu[w]; }
        g += gate_bias[(size_t)e * ff + o];
        u += up_bias[(size_t)e * ff + o];
        float xg = fminf(g, limit);
        float yu = fminf(fmaxf(u, -limit), limit);
        out[(size_t)b * n_active * ff + (size_t)slot * ff + o] =
            (xg / (1.0f + expf(-alpha * xg))) * (yu + 1.0f);
    }
}

// Batched fused MoE down + weighted mix + residual add. grid (embd, batch):
// block (o, b) sums Σ_slot w[b][slot]·(down_e·fused[b][slot] + bias) into
// residual[b][o]. `residual` is [batch, embd] and must be pre-zeroed by caller.
__global__ void pd_mxfp4_moe_down_batch_kernel(
    const unsigned char* __restrict__ down_W, const float* __restrict__ down_bias,
    const unsigned int* __restrict__ idx, const float* __restrict__ topk_w,
    const float* __restrict__ fused, float* __restrict__ residual,
    uint32_t ff, uint32_t embd, uint32_t n_active) {
    uint32_t o = blockIdx.x, b = blockIdx.y;
    uint32_t tid = threadIdx.x, nth = blockDim.x;
    uint32_t n_blocks = ff >> 5;
    __shared__ float wsum[32];
    const unsigned int* idx_b = idx + (size_t)b * n_active;
    const float* w_b = topk_w + (size_t)b * n_active;
    const float* fused_b = fused + (size_t)b * n_active * ff;

    float acc = 0.0f;
    for (uint32_t slot = 0; slot < n_active; ++slot) {
        uint32_t e = idx_b[slot];
        float w = w_b[slot];
        size_t base = ((size_t)((size_t)e * embd + o) * ff / 32) * 17;
        const float* fs = fused_b + (size_t)slot * ff;
        for (uint32_t bl = tid; bl < n_blocks; bl += nth) {
            const unsigned char* bp = down_W + base + (size_t)bl * 17;
            float d = pd_e8m0_half(bp[0]);
            uint32_t base_i = bl << 5;
            for (uint32_t j = 0; j < 16; ++j) {
                uint8_t packed = bp[1 + j];
                acc += w * d * (PD_FP4_VALUES[packed & 0x0F] * fs[base_i + j] +
                                PD_FP4_VALUES[packed >> 4] * fs[base_i + j + 16]);
            }
        }
    }
    for (uint32_t s = 16; s > 0; s >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, s);
    uint32_t warp = tid >> 5, lane = tid & 31u;
    if (lane == 0) wsum[warp] = acc;
    __syncthreads();
    if (tid == 0) {
        float v = 0.0f;
        uint32_t nwarps = (nth + 31u) >> 5;
        for (uint32_t w = 0; w < nwarps; ++w) v += wsum[w];
        float biasacc = 0.0f;
        for (uint32_t slot = 0; slot < n_active; ++slot)
            biasacc += w_b[slot] * down_bias[(size_t)idx_b[slot] * embd + o];
        residual[(size_t)b * embd + o] += v + biasacc;
    }
}

// Fused Q8_0-dequant + GEMV for a dense weight (attention q/k/v/o, router,
// lm_head). Weight is Q8_0 [in_dim, out_dim] in GGUF layout: out_dim rows of
// in_dim contiguous elements. y[o] = bias[o] + Σ_i dequant(W[o][i])·x[i]. Rows
// are block-aligned (in_dim % 32 == 0). `bias` may be null. One block per row;
// the block-strided read keeps each warp on one Q8_0 block (broadcast scale,
// coalesced 32-byte int8 load) - the same shape as the MXFP4 indexed GEMV.
__global__ void pd_q8_0_gemv_kernel(
    const uint8_t* __restrict__ W, const float* __restrict__ bias,
    const float* __restrict__ x, float* __restrict__ y,
    uint32_t in_dim, uint32_t out_dim) {
    uint32_t o = blockIdx.x;
    if (o >= out_dim) return;
    uint32_t tid = threadIdx.x, nth = blockDim.x;
    uint32_t n_blocks = in_dim >> 5;
    extern __shared__ float ssc[];   // per-block f16->f32 scales, computed once
    __shared__ float wsum[32];

    size_t byte_base = ((size_t)o * in_dim / 32) * 34;   // in_dim % 32 == 0 -> exact
    for (uint32_t b = tid; b < n_blocks; b += nth) {
        __half h;
        memcpy(&h, W + byte_base + (size_t)b * 34, sizeof(h));
        ssc[b] = __half2float(h);
    }
    __syncthreads();
    float acc = 0.0f;
#pragma unroll 4
    for (uint32_t i = tid; i < in_dim; i += nth) {
        const uint8_t* bp = W + byte_base + (size_t)(i >> 5) * 34;
        acc += (float)((int8_t)bp[2 + (i & 31u)]) * ssc[i >> 5] * x[i];
    }
    // warp-shuffle reduce + one cross-warp combine (replaces the 8-sync tree)
    for (uint32_t s = 16; s > 0; s >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, s);
    uint32_t warp = tid >> 5, lane = tid & 31u;
    if (lane == 0) wsum[warp] = acc;
    __syncthreads();
    if (tid == 0) {
        float v = 0.0f;
        uint32_t nwarps = (nth + 31u) >> 5;
        for (uint32_t w = 0; w < nwarps; ++w) v += wsum[w];
        if (bias) v += bias[o];
        y[o] = v;
    }
}

// Batched Q8_0 GEMM: y[b][o] = bias[o] + Σ_i dequant(W[o][i])·x[b][i] for b in
// 0..batch. This is the weight-read amortization that makes concurrent decode
// scale: one block per output row o dequants the weight row once and applies it
// to a tile of batch rows, so B sequences cost ~one weight read instead of B.
// x is row-major [batch, in_dim]; y is [batch, out_dim]. Rows block-aligned.
#define PD_GEMM_TILE 8
__global__ void __launch_bounds__(256) pd_q8_0_gemm_kernel(
    const uint8_t* __restrict__ W, const float* __restrict__ bias,
    const float* __restrict__ x, float* __restrict__ y,
    uint32_t in_dim, uint32_t out_dim, uint32_t batch) {
    uint32_t o = blockIdx.x;
    if (o >= out_dim) return;
    uint32_t tid = threadIdx.x, nth = blockDim.x;
    uint32_t n_blocks = in_dim >> 5;
    extern __shared__ float ssc[];              // per-block scales, computed once
    __shared__ float red[32];
    uint32_t lane = tid & 31u, warp = tid >> 5, nwarps = (nth + 31u) >> 5;

    size_t byte_base = ((size_t)o * in_dim / 32) * 34;
    for (uint32_t b = tid; b < n_blocks; b += nth) {
        __half h;
        memcpy(&h, W + byte_base + (size_t)b * 34, sizeof(h));
        ssc[b] = __half2float(h);
    }
    __syncthreads();

    for (uint32_t b0 = 0; b0 < batch; b0 += PD_GEMM_TILE) {
        uint32_t tb = (batch - b0 < PD_GEMM_TILE) ? (batch - b0) : PD_GEMM_TILE;
        float acc[PD_GEMM_TILE];
#pragma unroll
        for (uint32_t t = 0; t < PD_GEMM_TILE; ++t) acc[t] = 0.0f;
        // weight element dequanted once, applied across the tile of batch rows
        for (uint32_t i = tid; i < in_dim; i += nth) {
            const uint8_t* bp = W + byte_base + (size_t)(i >> 5) * 34;
            float w = (float)((int8_t)bp[2 + (i & 31u)]) * ssc[i >> 5];
            for (uint32_t t = 0; t < tb; ++t)
                acc[t] += w * x[(size_t)(b0 + t) * in_dim + i];
        }
        for (uint32_t t = 0; t < tb; ++t) {
            float a = acc[t];
            for (uint32_t s = 16; s > 0; s >>= 1) a += __shfl_down_sync(0xffffffffu, a, s);
            if (lane == 0) red[warp] = a;
            __syncthreads();
            if (tid == 0) {
                float v = 0.0f;
                for (uint32_t w = 0; w < nwarps; ++w) v += red[w];
                if (bias) v += bias[o];
                y[(size_t)(b0 + t) * out_dim + o] = v;
            }
            __syncthreads();
        }
    }
}

// Quantize an f32 activation x[n] to symmetric int8 with a per-32-block scale -
// the Q8_1-style activation quantization that lets weight×activation dot products
// run on the __dp4a integer unit (the method llama.cpp and mistral.rs/candle both
// use). One warp per block: warp-reduce max|x|, d = max/127, q = round(x/d).
// `q` is int8[n] (contiguous, 4-aligned for int loads); `scale` is f32[n/32].
__global__ void pd_quantize_q8_kernel(const float* __restrict__ x, signed char* __restrict__ q,
                                      float* __restrict__ scale, uint32_t n_blocks) {
    // cascade: required - the launcher goes through pd_pdl_go, and a
    // PSS-launched kernel without the wait races its predecessor (exactly the
    // bug this comment's commit fixed: launcher converted, body not armed).
    PD_PDL_ARM();
    // 8 warps per CTA, one 32-block per warp: the old 1-warp CTA
    // made the launch CTA-dispatch-bound - a 224x5120 verify plane was 35840
    // CTAs of 160 useful bytes each, 36.9 us for ~5.8 MB (~10% of DRAM roof)
    // in the c32 spec round, x75 launches/round. Same lanes, same shfl tree,
    // same rounding per block -> bit-identical output; only the grid shrinks.
    uint32_t b = blockIdx.x * 8u + (threadIdx.x >> 5);
    if (b >= n_blocks) return;
    uint32_t d = threadIdx.x & 31u;           // 0..31, one warp per 32-block
    float v = x[b * 32u + d];
    float a = fabsf(v);
    for (uint32_t s = 16; s > 0; s >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, s));
    float scl = a * (1.0f / 127.0f);
    if (d == 0) scale[b] = scl;
    float inv = scl > 0.0f ? 1.0f / scl : 0.0f;
    int qi = __float2int_rn(v * inv);
    qi = qi < -127 ? -127 : (qi > 127 ? 127 : qi);
    q[b * 32u + d] = (signed char)qi;
}

// relu^2 twin: applies relu(x)^2 before the per-32 amax, so a
// dense FFN whose activation is squared-relu (nemotron_h_moe's shared expert)
// can run its up plane on the ordinary q8 GEMM ladder and still hand the down
// plane a properly quantized activation. Bit-identical to writing relu^2 to
// f32 and calling pd_quantize_q8 on it - same values, same warp amax, same
// rounding - it just skips the round trip.
//
// Note the outputs are all >= 0, so the int8 range used is [0, 127]. That is
// not new: every relu^2 MoE path here quantizes the same non-negative plane.
__global__ void pd_quantize_q8_relu2_kernel(const float* __restrict__ x,
                                            signed char* __restrict__ q,
                                            float* __restrict__ scale, uint32_t n_blocks) {
    PD_PDL_ARM();
    // 8 warps/CTA, same geometry as pd_quantize_q8_kernel above (bit-exact)
    uint32_t b = blockIdx.x * 8u + (threadIdx.x >> 5);
    if (b >= n_blocks) return;
    uint32_t d = threadIdx.x & 31u;           // 0..31, one warp per 32-block
    float r = fmaxf(x[b * 32u + d], 0.0f);
    float v = r * r;
    float a = fabsf(v);
    for (uint32_t s = 16; s > 0; s >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, s));
    float scl = a * (1.0f / 127.0f);
    if (d == 0) scale[b] = scl;
    float inv = scl > 0.0f ? 1.0f / scl : 0.0f;
    int qi = __float2int_rn(v * inv);
    qi = qi < -127 ? -127 : (qi > 127 ? 127 : qi);
    q[b * 32u + d] = (signed char)qi;
}

// Fused variant: also emits the per-16 int8 sums (pd_q8_sums_strided's output)
// from the qi values still in registers - one graph node instead of two. The
// b=1 kq serving tick paid ~143 separate 1.3 us sums launches per token.
// BIT-IDENTICAL outputs: q/scale math unchanged; the sums
// are the same integer totals the strided kernel dp4a's from the stored
// bytes, folded via half-warp shuffles instead.
__global__ void pd_quantize_q8_sums_kernel(const float* __restrict__ x,
                                           signed char* __restrict__ q,
                                           float* __restrict__ scale,
                                           float* __restrict__ sums,
                                           uint32_t n_blocks) {
    // cascade: x is always the immediately-preceding rmsnorm/attention/
    // swiglu output, so the arm sits first (before the defensive b check -
    // every CTA must reach the wait). No-op under plain launches / pre-sm90.
    PD_PDL_ARM();
    uint32_t b = blockIdx.x;
    if (b >= n_blocks) return;
    uint32_t d = threadIdx.x;
    float v = x[b * 32u + d];
    float a = fabsf(v);
    for (uint32_t s = 16; s > 0; s >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, s));
    float scl = a * (1.0f / 127.0f);
    if (d == 0) scale[b] = scl;
    float inv = scl > 0.0f ? 1.0f / scl : 0.0f;
    int qi = __float2int_rn(v * inv);
    qi = qi < -127 ? -127 : (qi > 127 ? 127 : qi);
    q[b * 32u + d] = (signed char)qi;
    // per-16 sums: xor-reduce stays inside each 16-lane half
    int s16 = qi;
    for (uint32_t s = 8; s > 0; s >>= 1) s16 += __shfl_xor_sync(0xffffffffu, s16, s);
    if ((d & 15u) == 0u) sums[b * 2u + (d >> 4u)] = (float)s16;
}

// dp4a Q8_0 GEMV: y[o] = bias[o] + Σ_b wscale[b]·ascale[b]·Σ_k dp4a(w_int8, a_int8).
// Weight is Q8_0 (f16 scale + 32 int8 / 34-byte block); the activation is
// pre-quantized (pd_quantize_q8) so the dot runs as integer __dp4a - 4 int8 MACs
// per instruction, ~10× fewer ops than the f32 dequant path. Learns from the ggml
// vec_dot_q8_0_q8_1 structure; kernel is our own. Not bit-exact to f32 (quantized
// activation) but perplexity-close - validated by an integer CPU reference.
// Warp-per-output shape: in_dim/32 is small (90-128 on the 20b), so a
// block-per-output 256-thread strided loop leaves ~2/3 of every block idle -
// exposed on small out_dims (wk/wv: 512 rows ran at 28% DRAM). One warp owns
// one output row, 8 outputs per 256-thread block, pure shuffle reduce.
__global__ void pd_q8_0_gemv_dp4a_kernel(
    const unsigned char* __restrict__ W, const float* __restrict__ bias,
    const signed char* __restrict__ xq, const float* __restrict__ xs,
    float* __restrict__ y, uint32_t in_dim, uint32_t out_dim) {
    uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    uint32_t o = blockIdx.x * (blockDim.x >> 5) + warp;
    if (o >= out_dim) return;
    uint32_t n_blocks = in_dim >> 5;
    size_t byte_base = ((size_t)o * in_dim / 32) * 34;
    float acc = 0.0f;
    for (uint32_t b = lane; b < n_blocks; b += 32) {
        const unsigned char* bp = W + byte_base + (size_t)b * 34;
        __half h;
        memcpy(&h, bp, sizeof(h));
        float wd = __half2float(h);
        const signed char* aq = xq + (size_t)b * 32;
        int sumi = 0;
#pragma unroll
        for (uint32_t k = 0; k < 8; ++k) {
            const unsigned char* wp = bp + 2 + k * 4;
            int wv = (int)wp[0] | ((int)wp[1] << 8) | ((int)wp[2] << 16) | ((int)wp[3] << 24);
            int av = *(const int*)(aq + k * 4);
            sumi = __dp4a(wv, av, sumi);
        }
        acc += wd * xs[b] * (float)sumi;
    }
    for (uint32_t s = 16; s > 0; s >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, s);
    if (lane == 0) {
        if (bias) acc += bias[o];
        y[o] = acc;
    }
}

// dp4a MXFP4 GEMV for one expert selected by a device index (MoE), against a
// pre-quantized activation. Weight nibbles unpack to int8 in-register (no LUT),
// then integer __dp4a against the Q8_1 activation. y[o] = bias[e][o] +
// Σ_b e8m0_half(e)·ascale[b]·Σ dp4a(w_int8, a_int8). ~10× fewer compute ops than
// the float dequant path on the compute-bound MoE. Perplexity-close, not f32-exact.
__global__ void pd_mxfp4_gemv_indexed_dp4a_kernel(
    const unsigned char* __restrict__ W, const float* __restrict__ bias,
    const unsigned int* __restrict__ idx, uint32_t slot,
    const signed char* __restrict__ xq, const float* __restrict__ xs,
    float* __restrict__ y, uint32_t in_dim, uint32_t out_dim) {
    uint32_t o = blockIdx.x;
    if (o >= out_dim) return;
    uint32_t e = idx[slot];
    uint32_t tid = threadIdx.x, nth = blockDim.x;
    uint32_t n_blocks = in_dim >> 5;
    __shared__ float wsum[32];

    size_t elem_base = (size_t)e * in_dim * out_dim + (size_t)o * in_dim;
    size_t byte_base = (elem_base / 32) * 17;
    float acc = 0.0f;
    for (uint32_t b = tid; b < n_blocks; b += nth) {
        const unsigned char* bp = W + byte_base + (size_t)b * 17;
        float wd = pd_e8m0_half(bp[0]);
        const signed char* aq = xq + (size_t)b * 32;
        int sumi = 0;
#pragma unroll
        for (uint32_t i = 0; i < 4; ++i) {
            uint32_t q4;
            memcpy(&q4, bp + 1 + i * 4, 4);   // nibble bytes (unaligned in 17-byte block)
            int2 v = pd_fp4_unpack8(q4);
            int a_lo = *(const int*)(aq + i * 4);        // elems 4i..4i+3
            int a_hi = *(const int*)(aq + 16 + i * 4);   // elems 16+4i..
            sumi = __dp4a(v.x, a_lo, sumi);
            sumi = __dp4a(v.y, a_hi, sumi);
        }
        acc += wd * xs[b] * (float)sumi;
    }
    for (uint32_t s = 16; s > 0; s >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, s);
    uint32_t warp = tid >> 5, lane = tid & 31u;
    if (lane == 0) wsum[warp] = acc;
    __syncthreads();
    if (tid == 0) {
        float v = 0.0f;
        uint32_t nwarps = (nth + 31u) >> 5;
        for (uint32_t w = 0; w < nwarps; ++w) v += wsum[w];
        if (bias) v += bias[(size_t)e * out_dim + o];
        y[o] = v;
    }
}

// x += w[slot] * y, weight read from a device buffer (expert weighting without
// a host round-trip).
__global__ void pd_scale_add_dev_kernel(float* __restrict__ x, const float* __restrict__ y,
                                        const float* __restrict__ w, uint32_t slot, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += w[slot] * y[i];
}


