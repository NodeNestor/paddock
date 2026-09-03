// qwen4exp.cuh - Qwen3.8-Flash-Next (`qwen4_exp`) new-math kernels.
// Textually-included segment of the single pack translation unit.
// Not standalone-compilable: include order is defined by ../pack.cu.
//
// What is new in this family (nothing else in the pack computes it):
//   * grouped RMSNorm with the Gemma (1+w) FMA affine - the 4-stream
//     hyper-connection state normalizes each stream by its own rms while the
//     weight spans the full 4*hidden width, so no existing rmsnorm shape fits.
//   * the hyper-connection mix/combine pair (the model's residual replacement).
//   * the PLE n-gram per-stream gate (signed-sqrt scaled dot).
//   * causal depthwise conv1d+silu with a DILATION (PLE uses k=4 dilation 3;
//     every existing conv kernel here is dilation-1).
//
// Ground truth for every formula: paddock-kernels reference::qwen4exp
// (source-verified against HF transformers modular_qwen4_exp.py AND the vLLM
// PR#53899 nvidia package, which agree). Correctness-first shapes - shared
// tree reductions at a fixed 256 threads, exact expf/sqrtf, no fast-math
// intrinsics - because this family's whole job is to hold parity
// against that reference. The perf pass (vectorized loads, fusion of the
// mix/norm pair, one launch per layer instead of five) is a follow-up.

// sigmoid, the pack's usual exact form (not __expf - parity first).
__device__ __forceinline__ float pd_q4x_sig(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// ---------------------------------------------------------------- group norm

// Grouped Gemma RMSNorm, (1+w) affine in the FMA form vLLM's kernel uses:
// each of `groups` equal slices of a row is normalized by its own RMS, then
// `y = xn + xn*w` with w spanning the full row width.
// grid.x = groups, grid.y = rows; block must be 256 (power-of-2 tree).
__global__ void pd_q4x_group_norm_1p_kernel(const float* __restrict__ x,
                                            const float* __restrict__ w,
                                            float* __restrict__ out,
                                            uint32_t groups, uint32_t gd, float eps) {
    // Perf pass. The correctness-first shape this replaced used a
    // scalar walk and an 8-deep __syncthreads tree, and measured 9.32 us per
    // launch for 40 KB - three times what the neighbouring streaming kernels
    // cost. Same math, same (1+w) FMA epilogue; what changed is the reduction
    // (warp shuffle + one cross-warp combine) and the access width (float4
    // where the group is 4-aligned, which every shape in this family is).
    // The exact 1/sqrtf is kept, not rsqrtf: this norm feeds the router and
    // the gates, where a last-ulp change can flip a near-tie.
    // Deliberately not PDL-armed: arming pd_f8r_gemv earlier today measured a
    // wash (and the standing note says the same), so the cascade is a separate
    // question from this one and gets its own measurement if it is ever asked.
    __shared__ float wsum[8];
    __shared__ float s_inv;
    const uint32_t g = blockIdx.x, r = blockIdx.y;
    const size_t base = (size_t)r * groups * gd + (size_t)g * gd;
    const float* xb = x + base;
    float* ob = out + base;
    const float* wb = w + (size_t)g * gd;
    const uint32_t tid = threadIdx.x, nth = blockDim.x;
    const uint32_t warp = tid >> 5, lane = tid & 31u;
    const bool vec = (gd & 3u) == 0;

    float acc = 0.0f;
    if (vec) {
        const uint32_t n4 = gd >> 2;
        const float4* x4 = reinterpret_cast<const float4*>(xb);
        for (uint32_t i = tid; i < n4; i += nth) {
            const float4 v = x4[i];
            acc += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }
    } else {
        for (uint32_t i = tid; i < gd; i += nth) acc += xb[i] * xb[i];
    }
    for (uint32_t s = 16; s > 0; s >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, s);
    if (lane == 0) wsum[warp] = acc;
    __syncthreads();
    if (tid == 0) {
        float t = 0.0f;
        const uint32_t nw = nth >> 5;
        for (uint32_t i = 0; i < nw; ++i) t += wsum[i];
        s_inv = 1.0f / sqrtf(t / (float)gd + eps);
    }
    __syncthreads();
    const float inv = s_inv;
    if (vec) {
        const uint32_t n4 = gd >> 2;
        const float4* x4 = reinterpret_cast<const float4*>(xb);
        const float4* w4 = reinterpret_cast<const float4*>(wb);
        float4* o4 = reinterpret_cast<float4*>(ob);
        for (uint32_t i = tid; i < n4; i += nth) {
            const float4 v = x4[i];
            const float4 wv = w4[i];
            float4 o;
            o.x = v.x * inv + (v.x * inv) * wv.x;
            o.y = v.y * inv + (v.y * inv) * wv.y;
            o.z = v.z * inv + (v.z * inv) * wv.z;
            o.w = v.w * inv + (v.w * inv) * wv.w;
            o4[i] = o;
        }
    } else {
        for (uint32_t i = tid; i < gd; i += nth) {
            const float xn = xb[i] * inv;
            ob[i] = xn + xn * wb[i];
        }
    }
}

PD_EXPORT
int pd_q4x_group_norm_1p(const void* x, const void* w, void* out,
                         uint32_t rows, uint32_t groups, uint32_t gd, float eps,
                         void* stream) {
    if (rows == 0 || groups == 0 || gd == 0) return 0;
    dim3 grid(groups, rows);
    // 256 threads = 8 warps, matching the wsum[8] cross-warp slab above
    pd_q4x_group_norm_1p_kernel<<<grid, 256, 0, (cudaStream_t)stream>>>(
        (const float*)x, (const float*)w, (float*)out, groups, gd, eps);
    return pd_launch_status();
}

// ------------------------------------------------------- hyper-connection mix

// block_input[d] = Σ_s sigmoid(gate[s,d]) * xn[s,d] / hc.
// grid.x = ceil(hidden/256), grid.y = rows.
__global__ void pd_q4x_hc_mix_kernel(const float* __restrict__ xn,
                                     const float* __restrict__ gate,
                                     float* __restrict__ out,
                                     uint32_t hc, uint32_t hidden) {
    const uint32_t d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= hidden) return;
    const uint32_t r = blockIdx.y;
    const size_t base = (size_t)r * hc * hidden + d;
    float acc = 0.0f;
    for (uint32_t s = 0; s < hc; ++s) {
        const size_t i = base + (size_t)s * hidden;
        acc += pd_q4x_sig(gate[i]) * xn[i];
    }
    out[(size_t)r * hidden + d] = acc / (float)hc;
}

PD_EXPORT
int pd_q4x_hc_mix(const void* xn, const void* gate, void* out,
                  uint32_t rows, uint32_t hc, uint32_t hidden, void* stream) {
    if (rows == 0 || hc == 0 || hidden == 0) return 0;
    dim3 grid((hidden + 255) / 256, rows);
    pd_q4x_hc_mix_kernel<<<grid, 256, 0, (cudaStream_t)stream>>>(
        (const float*)xn, (const float*)gate, (float*)out, hc, hidden);
    return pd_launch_status();
}

// H[s,:] += block_out * 2*sigmoid(inj[s]/hc)  - the combine half of the
// hyper-connection residual. grid.x = ceil(hidden/256), grid.y = hc, grid.z = rows.
__global__ void pd_q4x_hc_combine_kernel(float* __restrict__ h,
                                         const float* __restrict__ block_out,
                                         const float* __restrict__ inj,
                                         uint32_t hc, uint32_t hidden) {
    const uint32_t d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= hidden) return;
    const uint32_t s = blockIdx.y, r = blockIdx.z;
    const float wgt = 2.0f * pd_q4x_sig(inj[(size_t)r * hc + s] / (float)hc);
    h[(size_t)r * hc * hidden + (size_t)s * hidden + d] +=
        block_out[(size_t)r * hidden + d] * wgt;
}

PD_EXPORT
int pd_q4x_hc_combine(void* h, const void* block_out, const void* inj,
                      uint32_t rows, uint32_t hc, uint32_t hidden, void* stream) {
    if (rows == 0 || hc == 0 || hidden == 0) return 0;
    dim3 grid((hidden + 255) / 256, hc, rows);
    pd_q4x_hc_combine_kernel<<<grid, 256, 0, (cudaStream_t)stream>>>(
        (float*)h, (const float*)block_out, (const float*)inj, hc, hidden);
    return pd_launch_status();
}

// m = silu(m * inv) in place - the low-rank mix's activation, where the
// hyper-connection divides by hc before the nonlinearity.
__global__ void pd_q4x_scale_silu_kernel(float* __restrict__ m, uint32_t n, float inv) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float s = m[i] * inv;
    m[i] = s * pd_q4x_sig(s);
}

PD_EXPORT
int pd_q4x_scale_silu(void* m, uint32_t n, float inv, void* stream) {
    if (n == 0) return 0;
    pd_q4x_scale_silu_kernel<<<(n + 255) / 256, 256, 0, (cudaStream_t)stream>>>(
        (float*)m, n, inv);
    return pd_launch_status();
}

// ------------------------------------------------------------------ PLE gate

// gv[s,:] = sigmoid( signed_sqrt( (K_s·Q_s) / sqrt(hidden) ) ) * V
// with K/Q already group-normalized by the caller (pd_q4x_group_norm_1p).
// grid.x = hc, grid.y = rows; block must be 256.
__global__ void pd_q4x_ple_gate_kernel(const float* __restrict__ kn,
                                       const float* __restrict__ qn,
                                       const float* __restrict__ value,
                                       float* __restrict__ gv,
                                       uint32_t hc, uint32_t hidden) {
    __shared__ float sred[256];
    __shared__ float s_gate;
    const uint32_t s = blockIdx.x, r = blockIdx.y;
    const size_t base = (size_t)r * hc * hidden + (size_t)s * hidden;

    float acc = 0.0f;
    for (uint32_t i = threadIdx.x; i < hidden; i += blockDim.x) {
        acc += kn[base + i] * qn[base + i];
    }
    sred[threadIdx.x] = acc;
    __syncthreads();
    for (uint32_t st = blockDim.x / 2; st > 0; st >>= 1) {
        if (threadIdx.x < st) sred[threadIdx.x] += sred[threadIdx.x + st];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const float raw = sred[0] / sqrtf((float)hidden);
        // signed_sqrt(x) = sign(x)*sqrt(max(|x|, 1e-6)) - copysignf matches the
        // reference's signum on -0.0 (both hand back the negative branch).
        s_gate = pd_q4x_sig(copysignf(sqrtf(fmaxf(fabsf(raw), 1e-6f)), raw));
    }
    __syncthreads();
    const float g = s_gate;
    const float* vb = value + (size_t)r * hidden;
    for (uint32_t i = threadIdx.x; i < hidden; i += blockDim.x) {
        gv[base + i] = g * vb[i];
    }
}

PD_EXPORT
int pd_q4x_ple_gate(const void* kn, const void* qn, const void* value, void* gv,
                    uint32_t rows, uint32_t hc, uint32_t hidden, void* stream) {
    if (rows == 0 || hc == 0 || hidden == 0) return 0;
    dim3 grid(hc, rows);
    pd_q4x_ple_gate_kernel<<<grid, 256, 0, (cudaStream_t)stream>>>(
        (const float*)kn, (const float*)qn, (const float*)value, (float*)gv, hc, hidden);
    return pd_launch_status();
}

// -------------------------------------------------------- dilated causal conv

// Depthwise causal conv1d + silu over a token sequence with DILATION, fresh
// state (positions before the start read zero): the PLE walk's k=4/dilation=3
// nine-token ring. `src` and `out` must be distinct (every output row reads
// earlier rows). w is [dim, k]. grid.x = ceil(dim/256), grid.y = n_tokens.
__global__ void pd_q4x_conv_dil_kernel(const float* __restrict__ src,
                                       const float* __restrict__ w,
                                       float* __restrict__ out,
                                       uint32_t dim, uint32_t k, uint32_t dil) {
    const uint32_t d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= dim) return;
    const uint32_t t = blockIdx.y;
    float acc = 0.0f;
    for (uint32_t j = 0; j < k; ++j) {
        const uint32_t back = (k - 1 - j) * dil;
        if (t >= back) acc += w[(size_t)d * k + j] * src[(size_t)(t - back) * dim + d];
    }
    out[(size_t)t * dim + d] = acc * pd_q4x_sig(acc);
}

PD_EXPORT
int pd_q4x_conv_dil(const void* src, const void* w, void* out,
                    uint32_t n_tokens, uint32_t dim, uint32_t k, uint32_t dil,
                    void* stream) {
    if (n_tokens == 0 || dim == 0 || k == 0) return 0;
    dim3 grid((dim + 255) / 256, n_tokens);
    pd_q4x_conv_dil_kernel<<<grid, 256, 0, (cudaStream_t)stream>>>(
        (const float*)src, (const float*)w, (float*)out, dim, k, dil);
    return pd_launch_status();
}

// One-token twin of the above off a carried window. `win` holds the last
// W = (k-1)*dil pre-conv rows OLDEST-FIRST ([W, dim]); back-offset b reads
// win[W-b] for b > 0 and `x` for b == 0. The window is advanced by the caller
// (a device-to-device row shift), so this kernel is graph-safe and stateless.
__global__ void pd_q4x_conv_dil_step_kernel(const float* __restrict__ x,
                                            const float* __restrict__ win,
                                            const float* __restrict__ w,
                                            float* __restrict__ out,
                                            uint32_t dim, uint32_t k, uint32_t dil,
                                            uint32_t wrows) {
    const uint32_t d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= dim) return;
    float acc = 0.0f;
    for (uint32_t j = 0; j < k; ++j) {
        const uint32_t back = (k - 1 - j) * dil;
        const float v = (back == 0) ? x[d] : win[(size_t)(wrows - back) * dim + d];
        acc += w[(size_t)d * k + j] * v;
    }
    out[d] = acc * pd_q4x_sig(acc);
}

PD_EXPORT
int pd_q4x_conv_dil_step(const void* x, const void* win, const void* w, void* out,
                         uint32_t dim, uint32_t k, uint32_t dil, void* stream) {
    if (dim == 0 || k == 0) return 0;
    const uint32_t wrows = (k - 1) * dil;
    pd_q4x_conv_dil_step_kernel<<<(dim + 255) / 256, 256, 0, (cudaStream_t)stream>>>(
        (const float*)x, (const float*)win, (const float*)w, (float*)out, dim, k, dil, wrows);
    return pd_launch_status();
}

// -------------------------------------------------------------- GDN mixer bits

// GDN output gated norm: per head_dim group, `y = w · rms_norm(x) · sigmoid(z)`
// - PLAIN w (not the 1+w form) and a SIGMOID gate. The pack's existing
// pd_gated_rmsnorm is the qwen3.5 shape (deltanet/core.cuh:883): same norm,
// but a SILU gate - feeding this family through it would multiply by
// z*sigmoid(z) and be silently wrong, so this is its own kernel.
// grid = rows (tokens * v_heads), block must be d (power of two).
__global__ void pd_q4x_gdn_gated_norm_kernel(const float* __restrict__ x,
                                             const float* __restrict__ z,
                                             const float* __restrict__ w,
                                             float* __restrict__ out,
                                             uint32_t d, float eps) {
    extern __shared__ float pd_q4x_gn_sh[];
    const uint32_t r = blockIdx.x, j = threadIdx.x;
    const size_t off = (size_t)r * d;
    const float xj = x[off + j];
    pd_q4x_gn_sh[j] = xj * xj;
    __syncthreads();
    for (uint32_t s = d >> 1; s > 0; s >>= 1) {
        if (j < s) pd_q4x_gn_sh[j] += pd_q4x_gn_sh[j + s];
        __syncthreads();
    }
    const float inv = 1.0f / sqrtf(pd_q4x_gn_sh[0] / (float)d + eps);
    out[off + j] = w[j] * (xj * inv) * pd_q4x_sig(z[off + j]);
}

PD_EXPORT
int pd_q4x_gdn_gated_norm(const void* x, const void* z, const void* w, void* out,
                          uint32_t n_rows, uint32_t d, float eps, void* stream) {
    if (n_rows == 0 || d == 0) return 0;
    if ((d & (d - 1)) != 0 || d > 1024) return cudaErrorInvalidValue;
    pd_q4x_gdn_gated_norm_kernel<<<n_rows, d, d * sizeof(float), (cudaStream_t)stream>>>(
        (const float*)x, (const float*)z, (const float*)w, (float*)out, d, eps);
    return pd_launch_status();
}

// Split the GDN conv output [rows, 2*hk*kd + hv*vd] into q, k (widened from
// hk key heads to hv value heads) and v, RAW - pd_gated_delta_recurrent does
// the L2 norm and the 1/sqrt(D) scale itself (deltanet/core.cuh:217-230).
//
// The widening is REPEAT_INTERLEAVE - key head `vh / (hv/hk)` serves value
// head `vh` (modeling_qwen3_5.py:504). The pack's own split kernels use
// `hk = vh % n_k_heads` (deltanet/core.cuh:317), which is the GGUF lane's
// load-permuted head order; raw safetensors planes need this map instead, and
// no permutation of the key heads can convert one into the other.
// grid.x = v_heads, grid.y = rows.
__global__ void pd_q4x_gdn_split_widen_kernel(const float* __restrict__ conv,
                                              float* __restrict__ q,
                                              float* __restrict__ k,
                                              float* __restrict__ v,
                                              uint32_t hk, uint32_t hv,
                                              uint32_t kd, uint32_t vd) {
    const uint32_t vh = blockIdx.x, r = blockIdx.y;
    const uint32_t kdim = hk * kd;
    const float* row = conv + (size_t)r * (2u * kdim + hv * vd);
    const uint32_t kh = vh / (hv / hk);
    const size_t qo = ((size_t)r * hv + vh) * kd;
    for (uint32_t i = threadIdx.x; i < kd; i += blockDim.x) {
        q[qo + i] = row[(size_t)kh * kd + i];
        k[qo + i] = row[(size_t)kdim + (size_t)kh * kd + i];
    }
    const size_t vo = ((size_t)r * hv + vh) * vd;
    for (uint32_t i = threadIdx.x; i < vd; i += blockDim.x) {
        v[vo + i] = row[(size_t)2 * kdim + (size_t)vh * vd + i];
    }
}

PD_EXPORT
int pd_q4x_gdn_split_widen(const void* conv, void* q, void* k, void* v,
                           uint32_t rows, uint32_t k_heads, uint32_t v_heads,
                           uint32_t k_dim, uint32_t v_dim, void* stream) {
    if (rows == 0 || v_heads == 0 || k_heads == 0) return 0;
    if (v_heads % k_heads != 0) return cudaErrorInvalidValue;
    dim3 grid(v_heads, rows);
    pd_q4x_gdn_split_widen_kernel<<<grid, 256, 0, (cudaStream_t)stream>>>(
        (const float*)conv, (float*)q, (float*)k, (float*)v,
        k_heads, v_heads, k_dim, v_dim);
    return pd_launch_status();
}

// ------------------------------------------------------- shared-expert fold

// y[r,:] += x[r,:] * sigmoid(s[r]) - the MoE shared expert's per-token scalar
// gate. `mul_sigmoid` is elementwise against a same-length gate plane; this one
// broadcasts one gate value across the row, which is what a scalar gate is.
// grid.x = ceil(n/256), grid.y = rows.
__global__ void pd_q4x_add_gated_row_kernel(float* __restrict__ y,
                                            const float* __restrict__ x,
                                            const float* __restrict__ s,
                                            uint32_t n) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint32_t r = blockIdx.y;
    const size_t o = (size_t)r * n + i;
    y[o] += x[o] * pd_q4x_sig(s[r]);
}

PD_EXPORT
int pd_q4x_add_gated_row(void* y, const void* x, const void* s,
                         uint32_t rows, uint32_t n, void* stream) {
    if (rows == 0 || n == 0) return 0;
    dim3 grid((n + 255) / 256, rows);
    pd_q4x_add_gated_row_kernel<<<grid, 256, 0, (cudaStream_t)stream>>>(
        (float*)y, (const float*)x, (const float*)s, n);
    return pd_launch_status();
}

// ---------------------------------------------------- combine + norm, fused

// The hyper-connection combine and the grouped (1+w) norm that always follows
// it, in one launch: `h[s,:] += block_out * 2*sigmoid(inj[s]/hc)`, then the
// normalized `(1+w)` image of the UPDATED h into `xn`.
//
// Every combine in this model is immediately followed by a norm of its own
// output - the next sub-block's mix, or the final mixer's - so the pair was
// 2 launches and two passes over the 4-stream state (write it, read it back).
// Fused it is one launch and one pass: the sum of squares is accumulated on
// the values as they are written. The caller skips the fusion for the one
// combine whose output is modified before the norm (a PLE layer adds to the
// state in between), which is the only ordering this kernel cannot see.
//
// grid = (hc, rows), block 256 (8 warps, matching wsum[8]). `norm_w` is the
// FOLLOWING norm's full-width weight; `inj` is already offset by the caller
// when the inject rides the tail of a folded low-rank output.
__global__ void pd_q4x_combine_norm_kernel(float* __restrict__ h,
                                           const float* __restrict__ block_out,
                                           const float* __restrict__ inj,
                                           const float* __restrict__ norm_w,
                                           float* __restrict__ xn,
                                           uint32_t hc, uint32_t hidden, float eps) {
    __shared__ float wsum[8];
    __shared__ float s_inv;
    const uint32_t s = blockIdx.x, r = blockIdx.y;
    const size_t base = (size_t)r * hc * hidden + (size_t)s * hidden;
    float* hb = h + base;
    float* ob = xn + base;
    const float* bo = block_out + (size_t)r * hidden;
    const float* wb = norm_w + (size_t)s * hidden;
    const uint32_t tid = threadIdx.x, nth = blockDim.x;
    const uint32_t warp = tid >> 5, lane = tid & 31u;
    const float wgt = 2.0f * pd_q4x_sig(inj[(size_t)r * hc + s] / (float)hc);
    const bool vec = (hidden & 3u) == 0;

    // pass 1: combine into h AND accumulate the sum of squares of what lands
    float acc = 0.0f;
    if (vec) {
        const uint32_t n4 = hidden >> 2;
        float4* h4 = reinterpret_cast<float4*>(hb);
        const float4* b4 = reinterpret_cast<const float4*>(bo);
        for (uint32_t i = tid; i < n4; i += nth) {
            float4 v = h4[i];
            const float4 bv = b4[i];
            v.x += bv.x * wgt; v.y += bv.y * wgt;
            v.z += bv.z * wgt; v.w += bv.w * wgt;
            h4[i] = v;
            acc += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }
    } else {
        for (uint32_t i = tid; i < hidden; i += nth) {
            const float v = hb[i] + bo[i] * wgt;
            hb[i] = v;
            acc += v * v;
        }
    }
    for (uint32_t k = 16; k > 0; k >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, k);
    if (lane == 0) wsum[warp] = acc;
    __syncthreads();
    if (tid == 0) {
        float t = 0.0f;
        const uint32_t nw = nth >> 5;
        for (uint32_t i = 0; i < nw; ++i) t += wsum[i];
        s_inv = 1.0f / sqrtf(t / (float)hidden + eps);
    }
    __syncthreads();
    const float inv = s_inv;

    // pass 2: the (1+w) FMA image of the updated state
    if (vec) {
        const uint32_t n4 = hidden >> 2;
        const float4* h4 = reinterpret_cast<const float4*>(hb);
        const float4* w4 = reinterpret_cast<const float4*>(wb);
        float4* o4 = reinterpret_cast<float4*>(ob);
        for (uint32_t i = tid; i < n4; i += nth) {
            const float4 v = h4[i];
            const float4 wv = w4[i];
            float4 o;
            o.x = v.x * inv + (v.x * inv) * wv.x;
            o.y = v.y * inv + (v.y * inv) * wv.y;
            o.z = v.z * inv + (v.z * inv) * wv.z;
            o.w = v.w * inv + (v.w * inv) * wv.w;
            o4[i] = o;
        }
    } else {
        for (uint32_t i = tid; i < hidden; i += nth) {
            const float t = hb[i] * inv;
            ob[i] = t + t * wb[i];
        }
    }
}

PD_EXPORT
int pd_q4x_combine_norm(void* h, const void* block_out, const void* inj,
                        const void* norm_w, void* xn, uint32_t rows,
                        uint32_t hc, uint32_t hidden, float eps, void* stream) {
    if (rows == 0 || hc == 0 || hidden == 0) return 0;
    dim3 grid(hc, rows);
    pd_q4x_combine_norm_kernel<<<grid, 256, 0, (cudaStream_t)stream>>>(
        (float*)h, (const float*)block_out, (const float*)inj,
        (const float*)norm_w, (float*)xn, hc, hidden, eps);
    return pd_launch_status();
}
