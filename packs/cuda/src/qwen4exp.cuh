#include <cuda_bf16.h>
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
                                            __nv_bfloat16* __restrict__ out16,
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
    __shared__ float wsum[32];   // up to 1024 threads = 32 warps
    __shared__ float s_inv;
    const uint32_t g = blockIdx.x, r = blockIdx.y;
    const size_t base = (size_t)r * groups * gd + (size_t)g * gd;
    const float* xb = x + base;
    float* ob = out + base;
    __nv_bfloat16* ob16 = out16 ? out16 + base : nullptr;
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
            if (ob16) {
                // bf16 MIRROR of the same values (slot-547 TGV feed): the
                // per-plane f32->bf16 cast launches were eating half the
                // TGV win, so writers mirror at the store.
                const uint32_t j = i * 4u;
                ob16[j] = __float2bfloat16(o.x);
                ob16[j + 1u] = __float2bfloat16(o.y);
                ob16[j + 2u] = __float2bfloat16(o.z);
                ob16[j + 3u] = __float2bfloat16(o.w);
            }
        }
    } else {
        for (uint32_t i = tid; i < gd; i += nth) {
            const float xn = xb[i] * inv;
            const float o = xn + xn * wb[i];
            ob[i] = o;
            if (ob16) ob16[i] = __float2bfloat16(o);
        }
    }
}

PD_EXPORT
int pd_q4x_group_norm_1p(const void* x, const void* w, void* out, void* out16,
                         uint32_t rows, uint32_t groups, uint32_t gd, float eps,
                         void* stream) {
    if (rows == 0 || groups == 0 || gd == 0) return 0;
    dim3 grid(groups, rows);
    // 256 threads = 8 warps, matching the wsum[8] cross-warp slab above
    pd_q4x_group_norm_1p_kernel<<<grid, 256, 0, (cudaStream_t)stream>>>(
        (const float*)x, (const float*)w, (float*)out, (__nv_bfloat16*)out16,
        groups, gd, eps);
    return pd_launch_status();
}

// ------------------------------------------------------- hyper-connection mix

// block_input[d] = Σ_s sigmoid(gate[s,d]) * xn[s,d] / hc.
// grid.x = ceil(hidden/256), grid.y = rows.
__global__ void pd_q4x_hc_mix_kernel(const float* __restrict__ xn,
                                     const float* __restrict__ gate,
                                     float* __restrict__ out,
                                     __nv_bfloat16* __restrict__ out16,
                                     uint32_t hc, uint32_t hidden) {
    PD_PDL_ARM();
    const uint32_t d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= hidden) return;
    const uint32_t r = blockIdx.y;
    const size_t base = (size_t)r * hc * hidden + d;
    float acc = 0.0f;
    for (uint32_t s = 0; s < hc; ++s) {
        const size_t i = base + (size_t)s * hidden;
        acc += pd_q4x_sig(gate[i]) * xn[i];
    }
    const float v = acc / (float)hc;
    out[(size_t)r * hidden + d] = v;
    // bf16 mirror for the slot-547 TGV feed (see group_norm_1p note)
    if (out16) out16[(size_t)r * hidden + d] = __float2bfloat16(v);
}

PD_EXPORT
int pd_q4x_hc_mix(const void* xn, const void* gate, void* out, void* out16,
                  uint32_t rows, uint32_t hc, uint32_t hidden, void* stream) {
    if (rows == 0 || hc == 0 || hidden == 0) return 0;
    dim3 grid((hidden + 255) / 256, rows);
    pd_pdl_go(pd_q4x_hc_mix_kernel, grid, 256, 0, (cudaStream_t)stream, 
        (const float*)xn, (const float*)gate, (float*)out,
        (__nv_bfloat16*)out16, hc, hidden);
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

// Per-SLOT twin: `rows` sequences, each against its own dilated window.
// Same expression and order as the single-row form.
__global__ void pd_q4x_conv_dil_step_slots_kernel(const float* __restrict__ x,
                                                  const float* __restrict__ win,
                                                  const float* __restrict__ w,
                                                  float* __restrict__ out,
                                                  const uint32_t* __restrict__ slots,
                                                  uint32_t dim, uint32_t k,
                                                  uint32_t dil, uint32_t wrows) {
    const uint32_t d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= dim) return;
    const uint32_t r = blockIdx.y;
    const uint32_t sl = slots ? slots[r] : r;
    const float* wrow = win + (size_t)sl * wrows * dim;
    const float* xrow = x + (size_t)r * dim;
    float acc = 0.0f;
    for (uint32_t j = 0; j < k; ++j) {
        const uint32_t back = (k - 1 - j) * dil;
        const float v = (back == 0) ? xrow[d] : wrow[(size_t)(wrows - back) * dim + d];
        acc += w[(size_t)d * k + j] * v;
    }
    out[(size_t)r * dim + d] = acc * pd_q4x_sig(acc);
}

PD_EXPORT
int pd_q4x_conv_dil_step_slots(const void* x, const void* win, const void* w,
                               void* out, const void* slots, uint32_t dim,
                               uint32_t k, uint32_t dil, uint32_t rows,
                               void* stream) {
    if (dim == 0 || k == 0 || rows == 0) return 0;
    const uint32_t wrows = (k - 1) * dil;
    dim3 grid((dim + 255) / 256, rows);
    pd_q4x_conv_dil_step_slots_kernel<<<grid, 256, 0, (cudaStream_t)stream>>>(
        (const float*)x, (const float*)win, (const float*)w, (float*)out,
        (const uint32_t*)slots, dim, k, dil, wrows);
    return pd_launch_status();
}

// RING twin of the per-slot step (slot 533): the window is a ring indexed by
// the row's POSITION, and the append rides the same kernel.
//
// The shifted form costs 3 launches per ROW per tick - at c32 that is 96
// dependent copies moving 10.5 MB through one shared scratch row, and because
// the offsets are computed on the host from the slot set, the captured decode
// graph is only valid for the slot set it was taken against (any hole in the
// occupied prefix drops the tick to an eager walk).
//
// Ring invariant: the pre-conv row for token position q lives at physical row
// `q % wrows`, so logical row i (0 = oldest) of a request at position p is
// physical `(p + i) % wrows`. That holds for the prefill seed and every decode
// step with no per-slot head state - the position is already staged in d_pos,
// so nothing host-side enters the launch and one capture serves every slot set.
//
// The only window element a thread writes is logical 0 (the row being
// evicted), which is also the only one it could still need: it is read into a
// register before the store, and column `d` is private to this thread, so the
// fusion needs no barrier.
__global__ void pd_q4x_conv_dil_step_ring_kernel(const float* __restrict__ x,
                                                 float* __restrict__ win,
                                                 const float* __restrict__ w,
                                                 float* __restrict__ out,
                                                 const uint32_t* __restrict__ slots,
                                                 const uint32_t* __restrict__ pos,
                                                 uint32_t dim, uint32_t k,
                                                 uint32_t dil, uint32_t wrows) {
    const uint32_t d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= dim) return;
    const uint32_t r = blockIdx.y;
    const uint32_t sl = slots ? slots[r] : r;
    const uint32_t p = pos[r];
    float* wrow = win + (size_t)sl * wrows * dim;
    const float* xrow = x + (size_t)r * dim;
    const float xv = xrow[d];
    float acc = 0.0f;
    for (uint32_t j = 0; j < k; ++j) {
        const uint32_t back = (k - 1 - j) * dil;
        float v;
        if (back == 0) {
            v = xv;
        } else {
            const uint32_t phys = (p + (wrows - back)) % wrows;
            v = wrow[(size_t)phys * dim + d];
        }
        acc += w[(size_t)d * k + j] * v;
    }
    out[(size_t)r * dim + d] = acc * pd_q4x_sig(acc);
    // evict the oldest row and append this token's, in one store
    wrow[(size_t)(p % wrows) * dim + d] = xv;
}

PD_EXPORT
int pd_q4x_conv_dil_step_ring(const void* x, void* win, const void* w,
                              void* out, const void* slots, const void* pos,
                              uint32_t dim, uint32_t k, uint32_t dil,
                              uint32_t rows, void* stream) {
    if (dim == 0 || k == 0 || rows == 0) return 0;
    const uint32_t wrows = (k - 1) * dil;
    dim3 grid((dim + 255) / 256, rows);
    pd_q4x_conv_dil_step_ring_kernel<<<grid, 256, 0, (cudaStream_t)stream>>>(
        (const float*)x, (float*)win, (const float*)w, (float*)out,
        (const uint32_t*)slots, (const uint32_t*)pos, dim, k, dil, wrows);
    return pd_launch_status();
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
                                             __nv_bfloat16* __restrict__ out16,
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
    const float o = w[j] * (xj * inv) * pd_q4x_sig(z[off + j]);
    out[off + j] = o;
    // bf16 mirror for the slot-547 TGV feed (see gemv_silu note)
    if (out16) out16[off + j] = __float2bfloat16(o);
}

PD_EXPORT
int pd_q4x_gdn_gated_norm(const void* x, const void* z, const void* w, void* out,
                          void* out16, uint32_t n_rows, uint32_t d, float eps,
                          void* stream) {
    if (n_rows == 0 || d == 0) return 0;
    if ((d & (d - 1)) != 0 || d > 1024) return cudaErrorInvalidValue;
    pd_q4x_gdn_gated_norm_kernel<<<n_rows, d, d * sizeof(float), (cudaStream_t)stream>>>(
        (const float*)x, (const float*)z, (const float*)w, (float*)out,
        (__nv_bfloat16*)out16, d, eps);
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
// grid.x = v_heads, grid.y = rows. TILED = the GGUF lane's head order
// (llama.cpp's converter tiles the value heads: key head `vh % hk` serves
// value head `vh`), INTERLEAVE = raw safetensors planes.
template <bool TILED>
__global__ void pd_q4x_gdn_split_widen_kernel(const float* __restrict__ conv,
                                              float* __restrict__ q,
                                              float* __restrict__ k,
                                              float* __restrict__ v,
                                              uint32_t hk, uint32_t hv,
                                              uint32_t kd, uint32_t vd) {
    const uint32_t vh = blockIdx.x, r = blockIdx.y;
    const uint32_t kdim = hk * kd;
    const float* row = conv + (size_t)r * (2u * kdim + hv * vd);
    const uint32_t kh = TILED ? (vh % hk) : (vh / (hv / hk));
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
    pd_q4x_gdn_split_widen_kernel<false><<<grid, 256, 0, (cudaStream_t)stream>>>(
        (const float*)conv, (float*)q, (float*)k, (float*)v,
        k_heads, v_heads, k_dim, v_dim);
    return pd_launch_status();
}

// slot 540: the same split with the TILED head map (GGUF lane).
PD_EXPORT
int pd_q4x_gdn_split_widen_tiled(const void* conv, void* q, void* k, void* v,
                           uint32_t rows, uint32_t k_heads, uint32_t v_heads,
                           uint32_t k_dim, uint32_t v_dim, void* stream) {
    if (rows == 0 || v_heads == 0 || k_heads == 0) return 0;
    if (v_heads % k_heads != 0) return cudaErrorInvalidValue;
    dim3 grid(v_heads, rows);
    pd_q4x_gdn_split_widen_kernel<true><<<grid, 256, 0, (cudaStream_t)stream>>>(
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
    PD_PDL_ARM();
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
    pd_pdl_go(pd_q4x_add_gated_row_kernel, grid, 256, 0, (cudaStream_t)stream, 
        (float*)y, (const float*)x, (const float*)s, n);
    return pd_launch_status();
}

// strided twin: gate for row r at s[r*rs] - the folded router plane's own
// [n, ne+1] layout. Same sigmoid on the same value.
__global__ void pd_q4x_add_gated_row_s_kernel(float* __restrict__ y,
                                              const float* __restrict__ x,
                                              const float* __restrict__ s,
                                              uint32_t rs, uint32_t n) {
    PD_PDL_ARM();
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint32_t r = blockIdx.y;
    const size_t o = (size_t)r * n + i;
    y[o] += x[o] * pd_q4x_sig(s[(size_t)r * rs]);
}

PD_EXPORT
int pd_q4x_add_gated_row_s(void* y, const void* x, const void* s, uint32_t rs,
                           uint32_t rows, uint32_t n, void* stream) {
    if (rows == 0 || n == 0) return 0;
    dim3 grid((n + 255) / 256, rows);
    pd_pdl_go(pd_q4x_add_gated_row_s_kernel, grid, 256, 0, (cudaStream_t)stream, 
        (float*)y, (const float*)x, (const float*)s, rs, n);
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
// slot 561: combine_norm with the DSL-lane MoE gather FUSED - instead of
// reading block_out = d_mix (a separate dslfork_combine kernel's output),
// pass 1 gathers directly from the dn gemm's C slab: sum_j topw[j]*s2d[e_j]*
// C_dn[e_j][0][i] (expert-outer, chunk height 256 = the dslfork exports' m,
// k = 10 picks). One block per (stream,row) is pure latency (see the plain
// launcher's note), so the 10 extra bf16 loads per element hide; the win is
// the retired combine launch + the d_mix round-trip.
__global__ __launch_bounds__(1024) void pd_q4x_combine_norm_moe_kernel(
        float* __restrict__ h, const __nv_bfloat16* __restrict__ cdn,
        const unsigned int* __restrict__ ids, const float* __restrict__ topw,
        const float* __restrict__ s2d, const float* __restrict__ shd,
        const float* __restrict__ sgate, const float* __restrict__ inj,
        const float* __restrict__ norm_w, float* __restrict__ xn,
        uint32_t hc, uint32_t hidden, float eps) {
    PD_PDL_ARM();
    __shared__ float wsum[32];
    __shared__ float s_inv;
    __shared__ float sw[10];
    __shared__ const __nv_bfloat162* sc2[10];
    const uint32_t s = blockIdx.x, r = blockIdx.y;
    const size_t base = (size_t)r * hc * hidden + (size_t)s * hidden;
    float* hb = h + base;
    float* ob = xn + base;
    const float* wb = norm_w + (size_t)s * hidden;
    const float* sb = shd + (size_t)r * hidden;
    const uint32_t tid = threadIdx.x, nth = blockDim.x;
    const uint32_t warp = tid >> 5, lane = tid & 31u;
    if (tid < 10u) {
        const unsigned int e = ids[tid];
        sw[tid] = topw[tid] * s2d[e];
        sc2[tid] = reinterpret_cast<const __nv_bfloat162*>(
            cdn + (size_t)e * (256u * (size_t)hidden));
    }
    __syncthreads();
    const float wgt = 2.0f * pd_q4x_sig(inj[(size_t)r * hc + s] / (float)hc);
    // the shared expert's scalar gate, folded in here too: the plain lane
    // wrote it into d_mix with its own add_gated_row launch. Same order
    // (routed sum first, then the gated shared row) => bit-parity.
    const float sg = pd_q4x_sig(sgate[r]);
    const bool vec = (hidden & 3u) == 0;
    float acc = 0.0f;
    if (vec) {
        const uint32_t n4 = hidden >> 2;
        float4* h4 = reinterpret_cast<float4*>(hb);
        const float4* s4 = reinterpret_cast<const float4*>(sb);
        for (uint32_t i = tid; i < n4; i += nth) {
            float4 m4 = make_float4(0.f, 0.f, 0.f, 0.f);
            #pragma unroll
            for (int j = 0; j < 10; ++j) {
                const __nv_bfloat162 a = sc2[j][2u * i];
                const __nv_bfloat162 b = sc2[j][2u * i + 1u];
                const float2 af = __bfloat1622float2(a);
                const float2 bf = __bfloat1622float2(b);
                m4.x += sw[j] * af.x; m4.y += sw[j] * af.y;
                m4.z += sw[j] * bf.x; m4.w += sw[j] * bf.y;
            }
            const float4 sv = s4[i];
            m4.x = fmaf(sv.x, sg, m4.x); m4.y = fmaf(sv.y, sg, m4.y);
            m4.z = fmaf(sv.z, sg, m4.z); m4.w = fmaf(sv.w, sg, m4.w);
            float4 v = h4[i];
            v.x += m4.x * wgt; v.y += m4.y * wgt;
            v.z += m4.z * wgt; v.w += m4.w * wgt;
            h4[i] = v;
            acc += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }
    } else {
        for (uint32_t i = tid; i < hidden; i += nth) {
            const __nv_bfloat16* const* sc1 =
                reinterpret_cast<const __nv_bfloat16* const*>(sc2);
            float mix = 0.0f;
            #pragma unroll
            for (int j = 0; j < 10; ++j) mix += sw[j] * (float)sc1[j][i];
            mix = fmaf(sb[i], sg, mix);
            const float v = hb[i] + mix * wgt;
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

extern "C" int pd_q4x_combine_norm_moe(void* h, const void* cdn, const void* ids,
                                       const void* topw, const void* s2d,
                                       const void* shd, const void* sgate,
                                       const void* inj, const void* norm_w,
                                       void* xn, uint32_t rows, uint32_t hc,
                                       uint32_t hidden, float eps, void* stream) {
    if (rows == 0 || hc == 0 || hidden == 0) return 0;
    dim3 grid(hc, rows);
    // same sizing as the plain launcher: one float4 per thread, capped 1024
    uint32_t nth = 256u;
    if ((hidden & 3u) == 0u) {
        const uint32_t n4 = hidden >> 2;
        nth = ((n4 + 31u) / 32u) * 32u;
        if (nth > 1024u) nth = 1024u;
        if (nth < 64u) nth = 64u;
    } else {
        nth = ((hidden + 31u) / 32u) * 32u;
        if (nth > 1024u) nth = 1024u;
    }
    static int nth_env2 = -1;
    if (nth_env2 < 0) {
        const char* e = getenv("PADDOCK_Q4X_CNM_THREADS");
        nth_env2 = (e && *e) ? atoi(e) : 0;
    }
    if (nth_env2 >= 32 && nth_env2 <= 1024) nth = (uint32_t)nth_env2;
    pd_pdl_go(pd_q4x_combine_norm_moe_kernel, grid, nth, 0, (cudaStream_t)stream,
        (float*)h, (const __nv_bfloat16*)cdn, (const unsigned int*)ids,
        (const float*)topw, (const float*)s2d, (const float*)shd,
        (const float*)sgate, (const float*)inj,
        (const float*)norm_w, (float*)xn, hc, hidden, eps);
    return pd_launch_status();
}

__global__ __launch_bounds__(1024) void pd_q4x_combine_norm_kernel(float* __restrict__ h,
                                           const float* __restrict__ block_out,
                                           const float* __restrict__ inj,
                                           const float* __restrict__ norm_w,
                                           float* __restrict__ xn,
                                           uint32_t hc, uint32_t hidden, float eps,
                                           // bf16 MIRROR of xn written at the store:
                                           // its consumers otherwise cast it per call
                                           // (96 casts of [8, 10240] a tick at c8).
                                           __nv_bfloat16* __restrict__ xn16) {
    PD_PDL_ARM();
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
        __nv_bfloat162* m2 = xn16 ? reinterpret_cast<__nv_bfloat162*>(xn16 + base) : nullptr;
        for (uint32_t i = tid; i < n4; i += nth) {
            const float4 v = h4[i];
            const float4 wv = w4[i];
            float4 o;
            o.x = v.x * inv + (v.x * inv) * wv.x;
            o.y = v.y * inv + (v.y * inv) * wv.y;
            o.z = v.z * inv + (v.z * inv) * wv.z;
            o.w = v.w * inv + (v.w * inv) * wv.w;
            o4[i] = o;
            if (m2) {
                m2[2u * i] = __floats2bfloat162_rn(o.x, o.y);
                m2[2u * i + 1u] = __floats2bfloat162_rn(o.z, o.w);
            }
        }
    } else {
        for (uint32_t i = tid; i < hidden; i += nth) {
            const float t = hb[i] * inv;
            ob[i] = t + t * wb[i];
            if (xn16) xn16[base + i] = __float2bfloat16(ob[i]);
        }
    }
}

PD_EXPORT
int pd_q4x_combine_norm(void* h, const void* block_out, const void* inj,
                        const void* norm_w, void* xn, uint32_t rows,
                        uint32_t hc, uint32_t hidden, float eps, void* xn16,
                        void* stream) {
    if (rows == 0 || hc == 0 || hidden == 0) return 0;
    dim3 grid(hc, rows);
    // One block per (stream, row) is all the parallelism this op has - hc is 4,
    // so the grid is 4 blocks on a 148-SM die and the kernel is pure latency.
    // The only knob left is threads: at 256 each thread walks 2.5 float4s of a
    // 2560-wide stream with dependent loads. Size the block to the vector work
    // (one float4 per thread) so the walk is a single load, capped at 1024.
    uint32_t nth = 256u;
    if ((hidden & 3u) == 0u) {
        const uint32_t n4 = hidden >> 2;
        nth = ((n4 + 31u) / 32u) * 32u;
        if (nth > 1024u) nth = 1024u;
        if (nth < 64u) nth = 64u;
    }
    static int nth_env = -1;
    if (nth_env < 0) {
        const char* e = getenv("PADDOCK_Q4X_CN_THREADS");
        nth_env = (e && *e) ? atoi(e) : 0;
    }
    if (nth_env >= 32 && nth_env <= 1024) nth = (uint32_t)nth_env;
    pd_pdl_go(pd_q4x_combine_norm_kernel, grid, nth, 0, (cudaStream_t)stream, 
        (float*)h, (const float*)block_out, (const float*)inj,
        (const float*)norm_w, (float*)xn, hc, hidden, eps,
        (__nv_bfloat16*)xn16);
    return pd_launch_status();
}

// ---------------------------------------------------------------------------
// 532: PLE n-gram row gather off the DEVICE-RESIDENT 51.2 GB fp8 table.
//
// The host twin of this (mmap + per-row `bytes()` lookup + a scalar
// e4m3->f32 loop) is a random-access read over a 51.2 GB file: 16 rows of
// 160 B per token, uniformly spread over 320M rows. Each 160 B row costs a
// 4 KB page fault, so the page cache can only help once the whole table is
// resident, and until then a 128-token prefill takes ~2000 faults. Measured
// on the serve ladder: prefill ticks of 891-48697 ms and a c8 TTFT p50 of
// 7858 ms. vLLM never had this problem - its `NgramEmbedding` holds the
// table in a `VocabParallelEmbedding`, i.e. device-resident, and gathers
// with an index_select (`longcat_flash_ngram.py`, `embed_batched`).
//
// On device the same access is 2560 B/token of coalesced HBM reads.
//
// `ids` is [rows, heads] GLOBAL row ids (the offsets are already folded in by
// the host hash, which is pure integer arithmetic on the token stream and
// touches no table memory). `out` is [rows, heads*width] f32, so a row's
// heads land contiguously exactly as the host gather laid them out.
__global__ void pd_q4x_ple_gather_kernel(const uint8_t* __restrict__ table,
                                         const uint32_t* __restrict__ ids,
                                         float* __restrict__ out,
                                         float scale, uint32_t heads,
                                         uint32_t width) {
    const uint32_t h = blockIdx.x;
    const uint32_t t = blockIdx.y;
    const size_t slot = (size_t)t * (size_t)heads + (size_t)h;
    // 320M rows x 160 B overruns 32 bits at 51.2e9 - the row id fits u32, the
    // BYTE offset does not.
    const uint8_t* src = table + (size_t)ids[slot] * (size_t)width;
    float* dst = out + slot * (size_t)width;
    for (uint32_t i = threadIdx.x; i < width; i += blockDim.x) {
        dst[i] = (float)reinterpret_cast<const __nv_fp8_e4m3&>(src[i]) * scale;
    }
}

PD_EXPORT
int pd_q4x_ple_gather(const void* table, const void* ids, void* out,
                      float scale, uint32_t rows, uint32_t heads,
                      uint32_t width, void* stream) {
    if (rows == 0 || heads == 0 || width == 0) return 0;
    // one block per (token, head) row: 160 B is a single coalesced burst and
    // the grid is rows*heads, which is already 512 blocks at c32 decode
    uint32_t nth = ((width + 31u) / 32u) * 32u;
    if (nth > 256u) nth = 256u;
    pd_q4x_ple_gather_kernel<<<dim3(heads, rows), nth, 0, (cudaStream_t)stream>>>(
        (const uint8_t*)table, (const uint32_t*)ids, (float*)out, scale, heads,
        width);
    return pd_launch_status();
}
