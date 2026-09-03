// asr/granite_speech.cuh - Granite Speech conformer tower + Q-Former kernels.
// Textually-included segment of the single pack translation unit.
// Not standalone-compilable: include order is defined by ../../pack.cu.
// ------------------------------------------------------- asr/granite_speech
//
// The granite-speech mmproj is a CONFORMER, not a whisper-shaped
// encoder, so four of its pieces have no counterpart anywhere else in the
// pack: a macaron FFN with a HALF-weighted residual, a sigmoid-GLU on a
// channel-split landing, a CENTERED depthwise conv over time (deltanet's
// `pd_causal_conv1d_silu` is causal - the whole point here is that the
// conformer looks both ways), and blockwise attention carrying **Shaw
// relative position embeddings**, which are an additive term computed from Q
// itself rather than a rotation or a fixed bias table.
//
// References studied (algorithms, no code copied): transformers 5.6
// `modeling_granite_speech.py` (the upstream graph and the correctness
// oracle) and llama.cpp b10330 `tools/mtmd/models/granite-speech.cpp`.
//
// The rest of the tower rides kernels that already exist: `pd_whisper_ln_f16`
// for every pre-norm, `pd_vision_attn_x` for the Q-Former's tiny batched
// self/cross attention (granite-vision's Q-Former generalized that kernel
// already), and cuBLAS for every GEMM.
//
// Numerics: f32 accumulate throughout, f16 only where a landing feeds a GEMM
// (the same class the qwen3-asr and whisper towers run at). Fixed summation
// order - key rows walk in index order per warp, tiles fold in index order.
//
// LAUNCH COUNT is the known open door, not occupancy: at 20 ms per encoder
// frame a 30 s clip is 1500 rows, so every one of these kernels is wide, and
// the tower's cost on the engine thread is the ~18 launches per layer it
// issues (7 cuBLAS + 11 our own - see `move-the-work`: 7.7 us
// and 4.4 us respectively on sm_120a). The fusions below already collapse the
// obvious pairs (bias+activation+cast, residual+norm+cast); the next rungs
// are folding the two macaron FFNs' norms into their predecessors' epilogues
// and giving the conformer attention an f16 K/V tile.

// ---------------------------------------------------------------------------
// 335: macaron FFN epilogue - bias, SiLU, and the f16 cast the down-projection
// eats, in one pass over the 4096-wide landing. SiLU is x*sigmoid(x), the same
// form `pd_swiglu_kernel` applies to its gate half.
// ---------------------------------------------------------------------------
__global__ void pd_gs_bias_silu_f16_kernel(const float* __restrict__ x,
                                           const float* __restrict__ bias,
                                           __half* __restrict__ out, uint32_t n,
                                           uint64_t total) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const float xi = x[i] + bias[i % n];
    out[i] = __float2half(xi / (1.0f + __expf(-xi)));
}

PD_EXPORT
int pd_gs_bias_silu_f16(const void* x, const void* bias, void* out, uint32_t rows, uint32_t n,
                        void* stream) {
    if (rows == 0 || n == 0) return 0;
    const uint64_t total = (uint64_t)rows * n;
    const uint64_t blocks = (total + 255ull) / 256ull;
    pd_gs_bias_silu_f16_kernel<<<(uint32_t)blocks, 256, 0, (cudaStream_t)stream>>>(
        (const float*)x, (const float*)bias, (__half*)out, n, total);
    return pd_launch_status();
}

// ---------------------------------------------------------------------------
// 336: the conv module's gate. The pointwise-up projection lands [rows, 2*d]
// and torch's `nn.GLU(dim=1)` splits it on the CHANNEL axis - with rows
// major and channels contiguous that is exactly the two halves of each row:
// `out[r][c] = (x[r][c] + bias[c]) * sigmoid(x[r][d+c] + bias[d+c])`.
// Stays f32: the depthwise conv below reads 15 neighbours per output and is
// the one place in the tower where an f16 round-trip would be re-read many
// times rather than consumed once.
// ---------------------------------------------------------------------------
__global__ void pd_gs_bias_glu_kernel(const float* __restrict__ x,
                                      const float* __restrict__ bias, float* __restrict__ out,
                                      uint32_t d, uint64_t total) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const uint32_t c = (uint32_t)(i % d);
    const uint64_t row = i / d;
    const float a = x[row * 2ull * d + c] + bias[c];
    const float g = x[row * 2ull * d + d + c] + bias[d + c];
    out[i] = a / (1.0f + __expf(-g));
}

PD_EXPORT
int pd_gs_bias_glu(const void* x, const void* bias, void* out, uint32_t rows, uint32_t d,
                   void* stream) {
    if (rows == 0 || d == 0) return 0;
    const uint64_t total = (uint64_t)rows * d;
    const uint64_t blocks = (total + 255ull) / 256ull;
    pd_gs_bias_glu_kernel<<<(uint32_t)blocks, 256, 0, (cudaStream_t)stream>>>(
        (const float*)x, (const float*)bias, (float*)out, d, total);
    return pd_launch_status();
}

// ---------------------------------------------------------------------------
// 337: depthwise conv over TIME, centered, with the folded BatchNorm affine
// and SiLU riding along, landing at f16 for the pointwise-down GEMM.
//
// `out[t][c] = silu(bnw[c] * sum_j w[j][c] * x[t + j - k/2][c] + bnb[c])`
//
// Two layout decisions make this coalesced: activations are row-major over
// time with the channel contiguous, so a warp covers 32 adjacent channels of
// one time step and every one of the 15 taps is a full-width load; and the
// weight plane is TRANSPOSED at load from the file's `[k][c]`-per-channel
// order to `w[j*d + c]`, so the tap index is the slow axis and each tap read
// is contiguous across the warp too.
//
// Out-of-range time steps contribute zero - the reference pads the sequence
// with `(k/2, k/2 - (k+1)%2)`, which for the odd kernel this model ships (15)
// is symmetric.
// ---------------------------------------------------------------------------
__global__ void pd_gs_dwconv_bn_silu_f16_kernel(const float* __restrict__ x,
                                                const float* __restrict__ w,
                                                const float* __restrict__ bnw,
                                                const float* __restrict__ bnb,
                                                __half* __restrict__ out, uint32_t rows,
                                                uint32_t d, uint32_t k, uint32_t pad,
                                                uint64_t total) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const uint32_t c = (uint32_t)(i % d);
    const int t = (int)(i / d);
    float acc = 0.0f;
    for (uint32_t j = 0; j < k; ++j) {
        const int ti = t + (int)j - (int)pad;
        if (ti >= 0 && ti < (int)rows)
            acc = fmaf(w[(size_t)j * d + c], x[(size_t)ti * d + c], acc);
    }
    const float v = fmaf(acc, bnw[c], bnb[c]);
    out[i] = __float2half(v / (1.0f + __expf(-v)));
}

PD_EXPORT
int pd_gs_dwconv_bn_silu_f16(const void* x, const void* w, const void* bnw, const void* bnb,
                             void* out, uint32_t rows, uint32_t d, uint32_t k, void* stream) {
    if (rows == 0 || d == 0 || k == 0) return 0;
    const uint64_t total = (uint64_t)rows * d;
    const uint64_t blocks = (total + 255ull) / 256ull;
    pd_gs_dwconv_bn_silu_f16_kernel<<<(uint32_t)blocks, 256, 0, (cudaStream_t)stream>>>(
        (const float*)x, (const float*)w, (const float*)bnw, (const float*)bnb, (__half*)out,
        rows, d, k, k / 2u, total);
    return pd_launch_status();
}

// ---------------------------------------------------------------------------
// 338: conformer attention - blockwise, bidirectional, with Shaw's relative
// position embeddings.
//
// The sequence is cut into independent `ctx`-row blocks (200 frames = 4 s;
// blocks never attend across), and within a block the logits carry an extra
// term that a rotation or a static bias table cannot express:
//
//   logits[i][j] = (q_i . k_j + q_i . rel[i - j + max_pos]) * scale
//
// `rel` is a learned [2*max_pos+1, hd] table shared by all heads, so the
// second term is a full hd-dot per (query, key) pair - the same arithmetic
// cost as the QK dot itself. The reference materializes it as a
// [ctx, ctx, hd] gather plus a batched mat-mul; we compute it inline, which
// is why the tile below stages three things instead of two.
//
// Shape follows `pd_vision_attn_kernel` (256 threads, one warp per query,
// lane-per-key dots straight out of shared, flash-style online softmax):
//   sh_kv[32][129]  K, then reused for V          16.5 KB
//   sh_r [40][129]  the rel rows this tile needs  20.6 KB
//   sh_q [8][128]   pre-scaled queries             4.0 KB
//   sh_w [8][32]    tile weights                   1.0 KB
// 42.1 KB total, under the 48 KB static ceiling (static __shared__
// has no opt-in override - only dynamic shared can go past it). That pins one
// block per SM; the lever when this shows up in a profile is f16 K/V/rel
// tiles, which is a numeric-class change and wants its own parity arm.
//
// A tile needs `QT + KT - 1` distinct rel rows because the index depends on
// i - j: query i0+w against key j0+lane reads row `(i0 - j0 - jl + 1 +
// max_pos) + w + (jl - 1 - lane)`, i.e. the tile's rel rows are a contiguous
// run and each lane walks it backwards. `context_size <= max_pos_emb` is a
// config invariant upstream enforces, so the clamp in the reference formula
// can never fire here and the run always lands inside the table.
//
// The last block of a clip is SHORT rather than padded - the reference pads
// to a whole block and masks, which is the same thing: real queries see only
// real keys, and padded rows are dropped before the output projection.
// ---------------------------------------------------------------------------
#define PD_GS_QT 8u
#define PD_GS_KT 32u
#define PD_GS_MAXD 128u
#define PD_GS_RT (PD_GS_QT + PD_GS_KT)

__global__ void __launch_bounds__(256) pd_gs_conf_attn_kernel(
    const float* __restrict__ qkv, __half* __restrict__ out, const float* __restrict__ rel,
    uint32_t rows, uint32_t ctx, uint32_t n_heads, uint32_t hd, uint32_t max_pos, float scale) {
    const uint32_t h = blockIdx.x;
    const uint32_t i0 = blockIdx.y * PD_GS_QT;
    const uint32_t row0 = blockIdx.z * ctx;
    if (row0 >= rows) return;
    const uint32_t len = min(ctx, rows - row0);
    if (i0 >= len) return;

    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const uint32_t il = i0 + warp;
    const bool live = il < len;
    const uint32_t dm = n_heads * hd;
    const size_t stride = (size_t)3 * dm;
    const size_t hoff = (size_t)h * hd;

    __shared__ float sh_kv[PD_GS_KT][PD_GS_MAXD + 1u];
    __shared__ float sh_r[PD_GS_RT][PD_GS_MAXD + 1u];
    __shared__ float sh_q[PD_GS_QT][PD_GS_MAXD];
    __shared__ float sh_w[PD_GS_QT][PD_GS_KT];

    if (live) {
        const float* qr = qkv + (size_t)(row0 + il) * stride + hoff;
        for (uint32_t d = lane; d < hd; d += 32u) sh_q[warp][d] = qr[d] * scale;
    }
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float m = -3.402823e38f, l = 0.0f;
    const uint32_t nd = (hd + 31u) >> 5;

    for (uint32_t j0 = 0; j0 < len; j0 += PD_GS_KT) {
        const uint32_t jl = min(PD_GS_KT, len - j0);
        const uint32_t rcnt = PD_GS_QT + jl - 1u;
        const uint32_t rlo = max_pos + i0 - (j0 + jl - 1u);
        __syncthreads();
        for (uint32_t idx = threadIdx.x; idx < jl * hd; idx += 256u) {
            const uint32_t jj = idx / hd, dd = idx % hd;
            sh_kv[jj][dd] = qkv[(size_t)(row0 + j0 + jj) * stride + dm + hoff + dd];
        }
        for (uint32_t idx = threadIdx.x; idx < rcnt * hd; idx += 256u) {
            const uint32_t rr = idx / hd, dd = idx % hd;
            sh_r[rr][dd] = rel[(size_t)(rlo + rr) * hd + dd];
        }
        __syncthreads();

        float dot = -3.402823e38f;
        if (live && lane < jl) {
            // this lane's rel row inside the staged run
            const uint32_t rr = warp + (jl - 1u - lane);
            float pk = 0.0f, pr = 0.0f;
            for (uint32_t d = 0; d < hd; ++d) {
                const float qd = sh_q[warp][d];
                pk = fmaf(qd, sh_kv[lane][d], pk);
                pr = fmaf(qd, sh_r[rr][d], pr);
            }
            dot = pk + pr;
        }
        float tmax = dot;
#pragma unroll
        for (uint32_t off = 16; off > 0; off >>= 1)
            tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffu, tmax, off));
        const float m_new = fmaxf(m, tmax);
        const float corr = __expf(m - m_new);
        const float wv = (live && lane < jl) ? __expf(dot - m_new) : 0.0f;
        sh_w[warp][lane] = wv;
        float wsum = wv;
#pragma unroll
        for (uint32_t off = 16; off > 0; off >>= 1)
            wsum += __shfl_xor_sync(0xffffffffu, wsum, off);
        __syncwarp();

        // K is done; the same tile now carries V
        __syncthreads();
        for (uint32_t idx = threadIdx.x; idx < jl * hd; idx += 256u) {
            const uint32_t jj = idx / hd, dd = idx % hd;
            sh_kv[jj][dd] = qkv[(size_t)(row0 + j0 + jj) * stride + 2u * dm + hoff + dd];
        }
        __syncthreads();
        if (live) {
#pragma unroll
            for (uint32_t c = 0; c < 4; ++c) {
                const uint32_t d = lane + c * 32u;
                if (c < nd && d < hd) {
                    float a = acc[c] * corr;
                    for (uint32_t j = 0; j < jl; ++j) a = fmaf(sh_w[warp][j], sh_kv[j][d], a);
                    acc[c] = a;
                }
            }
            l = l * corr + wsum;
            m = m_new;
        }
    }
    if (live) {
        __half* orow = out + (size_t)(row0 + il) * dm + hoff;
#pragma unroll
        for (uint32_t c = 0; c < 4; ++c) {
            const uint32_t d = lane + c * 32u;
            if (c < nd && d < hd) orow[d] = __float2half(acc[c] / l);
        }
    }
}

PD_EXPORT
int pd_gs_conf_attn(const void* qkv, void* out, const void* rel, uint32_t rows, uint32_t ctx,
                    uint32_t n_heads, uint32_t hd, uint32_t max_pos, float scale,
                    void* stream) {
    if (rows == 0 || ctx == 0 || n_heads == 0) return 0;
    if (hd > PD_GS_MAXD || ctx > max_pos) return 1;
    const dim3 grid(n_heads, (ctx + PD_GS_QT - 1u) / PD_GS_QT, (rows + ctx - 1u) / ctx);
    pd_gs_conf_attn_kernel<<<grid, 256, 0, (cudaStream_t)stream>>>(
        (const float*)qkv, (__half*)out, (const float*)rel, rows, ctx, n_heads, hd, max_pos,
        scale);
    return pd_launch_status();
}

// ---------------------------------------------------------------------------
// 339: the CTC branch's head - bias, a row softmax over the 348 CTC symbols,
// and the f16 cast the projection back into model width eats. One block per
// row; the reduction is the standard two-pass max/sum through a 32-slot warp
// staging array, the same structure the LayerNorms here use.
// ---------------------------------------------------------------------------
__global__ void pd_gs_bias_softmax_f16_kernel(const float* __restrict__ x,
                                              const float* __restrict__ bias,
                                              __half* __restrict__ out, uint32_t n) {
    const float* xr = x + (size_t)blockIdx.x * n;
    __half* orow = out + (size_t)blockIdx.x * n;
    const uint32_t tid = threadIdx.x, nth = blockDim.x;
    const uint32_t lane = tid & 31u, warp = tid >> 5, nwarps = (nth + 31u) >> 5;
    __shared__ float red[32];
    __shared__ float s_max, s_sum;

    float mx = -3.402823e38f;
    for (uint32_t i = tid; i < n; i += nth) mx = fmaxf(mx, xr[i] + bias[i]);
    for (uint32_t s = 16; s > 0; s >>= 1) mx = fmaxf(mx, __shfl_down_sync(0xffffffffu, mx, s));
    if (lane == 0) red[warp] = mx;
    __syncthreads();
    if (tid == 0) {
        float v = -3.402823e38f;
        for (uint32_t i = 0; i < nwarps; ++i) v = fmaxf(v, red[i]);
        s_max = v;
    }
    __syncthreads();
    const float mval = s_max;
    float acc = 0.0f;
    for (uint32_t i = tid; i < n; i += nth) acc += __expf(xr[i] + bias[i] - mval);
    for (uint32_t s = 16; s > 0; s >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, s);
    if (lane == 0) red[warp] = acc;
    __syncthreads();
    if (tid == 0) {
        float v = 0.0f;
        for (uint32_t i = 0; i < nwarps; ++i) v += red[i];
        s_sum = v;
    }
    __syncthreads();
    const float inv = 1.0f / s_sum;
    for (uint32_t i = tid; i < n; i += nth)
        orow[i] = __float2half(__expf(xr[i] + bias[i] - mval) * inv);
}

PD_EXPORT
int pd_gs_bias_softmax_f16(const void* x, const void* bias, void* out, uint32_t rows,
                           uint32_t n, void* stream) {
    if (rows == 0 || n == 0) return 0;
    pd_gs_bias_softmax_f16_kernel<<<rows, 256, 0, (cudaStream_t)stream>>>(
        (const float*)x, (const float*)bias, (__half*)out, n);
    return pd_launch_status();
}

// ---------------------------------------------------------------------------
// 340/341: the two residual seams, each one launch.
//
//   res_ln  (pre-norm blocks):  x += s*(proj + bias); out = LN(x)  at f16
//   post_ln (post-norm blocks): x  = LN(x + s*(proj + bias));      f32 + f16
//
// `s` is what makes these granite-speech's rather than whisper's: the
// conformer's two macaron FFN halves fold in at **0.5**, and getting that
// scale onto the residual instead of the branch is the difference between a
// conformer and a transformer that merely looks like one.
//
// The Q-Former is post-LN (BLIP-2's contract, eps 1e-12), so its residual
// stream is the normalized value - which is why `post_ln` writes f32 back as
// well as the f16 landing, and why the two forms cannot be one kernel with a
// flag: they disagree about what the next block reads.
//
// `w`/`b`/`out` may be null on `res_ln` (pure residual update); `out` may be
// null on `post_ln` (the conformer's own post-norm feeds another LayerNorm,
// not a GEMM). Both share `pd_gs_ln_row`, whose reduction structure matches
// `pd_layernorm_kernel` term for term.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void pd_gs_ln_row(const float* __restrict__ xr,
                                             const float* __restrict__ w,
                                             const float* __restrict__ b,
                                             float* __restrict__ f32out,
                                             __half* __restrict__ f16out, uint32_t n,
                                             float eps) {
    const uint32_t tid = threadIdx.x, nth = blockDim.x;
    const uint32_t lane = tid & 31u, warp = tid >> 5, nwarps = (nth + 31u) >> 5;
    __shared__ float red[32];
    __shared__ float s_mean, s_inv;

    float acc = 0.0f;
    for (uint32_t i = tid; i < n; i += nth) acc += xr[i];
    for (uint32_t s = 16; s > 0; s >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, s);
    if (lane == 0) red[warp] = acc;
    __syncthreads();
    if (tid == 0) {
        float v = 0.0f;
        for (uint32_t i = 0; i < nwarps; ++i) v += red[i];
        s_mean = v / (float)n;
    }
    __syncthreads();
    const float mean = s_mean;
    float vacc = 0.0f;
    for (uint32_t i = tid; i < n; i += nth) {
        const float dd = xr[i] - mean;
        vacc += dd * dd;
    }
    for (uint32_t s = 16; s > 0; s >>= 1) vacc += __shfl_down_sync(0xffffffffu, vacc, s);
    if (lane == 0) red[warp] = vacc;
    __syncthreads();
    if (tid == 0) {
        float v = 0.0f;
        for (uint32_t i = 0; i < nwarps; ++i) v += red[i];
        s_inv = rsqrtf(v / (float)n + eps);
    }
    __syncthreads();
    const float inv = s_inv;
    for (uint32_t i = tid; i < n; i += nth) {
        const float v = (xr[i] - mean) * inv * w[i] + b[i];
        if (f32out) f32out[i] = v;
        if (f16out) f16out[i] = __float2half(v);
    }
}

__global__ void pd_gs_res_ln_f16_kernel(float* __restrict__ x, const float* __restrict__ proj,
                                        const float* __restrict__ bias,
                                        const float* __restrict__ w,
                                        const float* __restrict__ b, __half* __restrict__ out,
                                        uint32_t n, float s, float eps) {
    const size_t off = (size_t)blockIdx.x * n;
    float* xr = x + off;
    const float* pr = proj + off;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x)
        xr[i] += s * (pr[i] + (bias ? bias[i] : 0.0f));
    if (!w) return;
    __syncthreads();
    pd_gs_ln_row(xr, w, b, nullptr, out ? out + off : nullptr, n, eps);
}

PD_EXPORT
int pd_gs_res_ln_f16(void* x, const void* proj, const void* bias, const void* w, const void* b,
                     void* out, uint32_t rows, uint32_t n, float s, float eps, void* stream) {
    if (rows == 0 || n == 0) return 0;
    pd_gs_res_ln_f16_kernel<<<rows, 256, 0, (cudaStream_t)stream>>>(
        (float*)x, (const float*)proj, (const float*)bias, (const float*)w, (const float*)b,
        (__half*)out, n, s, eps);
    return pd_launch_status();
}

__global__ void pd_gs_post_ln_f16_kernel(float* __restrict__ x, const float* __restrict__ proj,
                                         const float* __restrict__ bias,
                                         const float* __restrict__ w,
                                         const float* __restrict__ b, __half* __restrict__ out,
                                         uint32_t n, float s, float eps) {
    const size_t off = (size_t)blockIdx.x * n;
    float* xr = x + off;
    const float* pr = proj + off;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x)
        xr[i] += s * (pr[i] + (bias ? bias[i] : 0.0f));
    __syncthreads();
    // the normalized value replaces the residual in place - safe because the
    // variance pass is behind `pd_gs_ln_row`'s own barrier before it writes
    pd_gs_ln_row(xr, w, b, xr, out ? out + off : nullptr, n, eps);
}

PD_EXPORT
int pd_gs_post_ln_f16(void* x, const void* proj, const void* bias, const void* w,
                      const void* b, void* out, uint32_t rows, uint32_t n, float s, float eps,
                      void* stream) {
    if (rows == 0 || n == 0) return 0;
    pd_gs_post_ln_f16_kernel<<<rows, 256, 0, (cudaStream_t)stream>>>(
        (float*)x, (const float*)proj, (const float*)bias, (const float*)w, (const float*)b,
        (__half*)out, n, s, eps);
    return pd_launch_status();
}
