// BF16 dense weight planes - the "read the tensor in the class the FILE ships
// it in" lane.
//
// Why this exists: UD quant files are MIXED. muse-glimmer's UD-Q8_K_XL keeps
// token_embd / output / attn_k / attn_v at BF16 while attn_q / attn_gate /
// attn_output / ffn_* are Q8_0 - the quantizer deliberately spends bytes on
// the planes it judges sensitive. The engine's rule for that is
// per-TENSOR dispatch, not a per-model switch, and the correctness spine is
// same-weights parity against llama.cpp on the identical GGUF. Down-quantizing
// a bf16 plane to Q8_0 at load would break both: different weights, no exact
// greedy target, and a quality choice silently overridden.
//
// So: keep the bytes, widen in-register. Weights stay bf16 in DRAM (file
// bytes, no 2x f32 inflation - the 202048-row LM head would otherwise cost
// 5.4 GB and 2x the decode read), activations stay f32, accumulation is f32.
// That is the same arithmetic class the Q8_0 lane runs and strictly more
// precise than it, so nothing downstream has to know which class a plane came
// from beyond picking the launcher.
//
// Layout is the same convention as the repacked Q8_0 planes: a GGUF [in, out]
// tensor is out rows of in contiguous elements, row-major, so `w + o*in_dim`
// is output row o. That is what makes these drop-in twins of
// pd_q8_0_gemv_repacked / pd_q8_0_gemm_repacked at the call site.
//
// The prefill arm is the mma.sync one (pd_bf16_gemm_mma below):
// paddleocr-vl serves bf16 as its PRIMARY class (official-GGUF planes verbatim,
// no quant ladder), which is exactly the condition this file's original header
// reserved the tensor-core arm for. It stages the f32 activations to bf16 in
// shared memory - the same class the parity reference computes: llama.cpp's
// batched BF16 path converts src1 to bf16 and runs cublasGemmEx bf16xbf16 with
// CUBLAS_COMPUTE_32F (ggml-cuda.cu batched_mul_mat_traits<GGML_TYPE_BF16>), and
// the greedy gate holds byte-exact against it. Weights are exact in
// bf16, so only the activation cast (~2^-8 relative) separates it from the
// f32-FMA tile; the correctness battery gates the switch. The f32-FMA tile
// GEMM stays as the fallback for older packs and ragged in_dim.

// One block per output row, 16 elements per thread: one 32-byte weight load
// (two int4) plus four float4 activation loads, so a warp issues fully
// coalesced 512-byte transactions. 128 threads, not 256, for the same reason
// pd_q8_0_gemv_repacked uses 128 - this die's maxThreadsPerMultiProcessor is
// 1536, so 128 gets 12 resident blocks/SM against 256's 6, and this kernel is
// bandwidth-bound end to end.
__global__ void pd_bf16_gemv_f32_kernel(
    const __nv_bfloat16* __restrict__ w, const float* __restrict__ bias,
    const float* __restrict__ x, float* __restrict__ y,
    uint32_t in_dim, uint32_t out_dim) {
    uint32_t o = blockIdx.x;
    if (o >= out_dim) return;
    uint32_t tid = threadIdx.x, nth = blockDim.x;
    // dep-free prologue: nothing above touches chain data, so the wait gates
    // only the x reads below. No-op under plain launches.
    PD_PDL_ARM();
    __shared__ float wsum[32];
    const __nv_bfloat16* row = w + (size_t)o * in_dim;
    float acc = 0.0f;
    if ((in_dim & 15u) == 0u) {
        for (uint32_t base = tid * 16u; base < in_dim; base += nth * 16u) {
            int4 w0 = *reinterpret_cast<const int4*>(row + base);
            int4 w1 = *reinterpret_cast<const int4*>(row + base + 8);
            const __nv_bfloat16* wa = reinterpret_cast<const __nv_bfloat16*>(&w0);
            const __nv_bfloat16* wb = reinterpret_cast<const __nv_bfloat16*>(&w1);
            float4 x0 = *reinterpret_cast<const float4*>(x + base);
            float4 x1 = *reinterpret_cast<const float4*>(x + base + 4);
            float4 x2 = *reinterpret_cast<const float4*>(x + base + 8);
            float4 x3 = *reinterpret_cast<const float4*>(x + base + 12);
            acc += __bfloat162float(wa[0]) * x0.x + __bfloat162float(wa[1]) * x0.y
                 + __bfloat162float(wa[2]) * x0.z + __bfloat162float(wa[3]) * x0.w
                 + __bfloat162float(wa[4]) * x1.x + __bfloat162float(wa[5]) * x1.y
                 + __bfloat162float(wa[6]) * x1.z + __bfloat162float(wa[7]) * x1.w
                 + __bfloat162float(wb[0]) * x2.x + __bfloat162float(wb[1]) * x2.y
                 + __bfloat162float(wb[2]) * x2.z + __bfloat162float(wb[3]) * x2.w
                 + __bfloat162float(wb[4]) * x3.x + __bfloat162float(wb[5]) * x3.y
                 + __bfloat162float(wb[6]) * x3.z + __bfloat162float(wb[7]) * x3.w;
        }
    } else {
        // ragged in_dim: scalar walk. No plane in any elected file lands here
        // (every in_dim is a multiple of 128), but a silent wrong answer on
        // one that did would be far worse than the slow path.
        for (uint32_t i = tid; i < in_dim; i += nth)
            acc += __bfloat162float(row[i]) * x[i];
    }
    for (uint32_t s = 16; s > 0; s >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, s);
    uint32_t warp = tid >> 5, lane = tid & 31u;
    if (lane == 0) wsum[warp] = acc;
    __syncthreads();
    if (tid == 0) {
        float v = 0.0f;
        uint32_t nwarps = (nth + 31u) >> 5;
        for (uint32_t w2 = 0; w2 < nwarps; ++w2) v += wsum[w2];
        if (bias) v += bias[o];
        y[o] = v;
    }
}

PD_EXPORT
int pd_bf16_gemv_f32(const void* w, const void* bias, const void* x, void* y,
                     uint32_t in_dim, uint32_t out_dim, void* stream) {
    if (out_dim == 0 || in_dim == 0) return 0;
    pd_pdl_go(pd_bf16_gemv_f32_kernel, out_dim, 128u, 0u, (cudaStream_t)stream,
              (const __nv_bfloat16*)w, (const float*)bias, (const float*)x,
              (float*)y, in_dim, out_dim);
    return pd_launch_status();
}

// Decode-band multi-row twin (2 <= batch <= 8). The tile GEMM below collapses
// at decode widths: its grid is (out/64, batch/64), so batch<=8 leaves gridY=1
// and the whole launch is out_dim/64 blocks - 4 blocks for a 256-wide K/V
// plane, measured at 794us for a 0.5 MB weight read on paddleocr at c4
// (kernel-shape-not-weight-class). Same disease the f16 lane fixed
// with pd_f16_gemv_kernel; this is that shape for bf16 W x f32 X: one warp per
// output row, 8 rows per 256-thread block, the W row streamed once as 16B
// packs while the batch's activation rows ride L1 (all 8 warps of a block
// read the same X), f32 products into acc[NB], butterfly reduce, lane b
// stores output row b. No smem X stage: X here is f32 straight from the batch
// scratch, and at these in_dims (<= a few K) the block-shared L1 window
// carries it - the f16 kernel's swizzled stage exists because its X is f16
// activations it must also transpose.
template <uint32_t NB>
__global__ void __launch_bounds__(256) pd_bf16_gemv_mr_f32_kernel(
    const __nv_bfloat16* __restrict__ w, const float* __restrict__ bias,
    const float* __restrict__ x, float* __restrict__ y,
    uint32_t in_dim, uint32_t out_dim, uint32_t batch) {
    const uint32_t o = blockIdx.x * 8u + (threadIdx.x >> 5);
    const uint32_t lane = threadIdx.x & 31u;
    // arm before the ragged-edge return: the grid over-covers out_dim, and
    // every thread must pass the griddepcontrol.wait
    PD_PDL_ARM();
    if (o >= out_dim) return;
    const __nv_bfloat16* row = w + (size_t)o * in_dim;
    float acc[NB];
#pragma unroll
    for (uint32_t b = 0; b < NB; ++b) acc[b] = 0.0f;
    // 16 weights (two int4) per lane per iter, the single-row kernel's pack
    // shape; the launcher gates in_dim % 16 == 0 (every elected plane is a
    // multiple of 128)
    for (uint32_t base = lane * 16u; base < in_dim; base += 32u * 16u) {
        int4 w0 = *reinterpret_cast<const int4*>(row + base);
        int4 w1 = *reinterpret_cast<const int4*>(row + base + 8);
        const __nv_bfloat16* wa = reinterpret_cast<const __nv_bfloat16*>(&w0);
        const __nv_bfloat16* wb = reinterpret_cast<const __nv_bfloat16*>(&w1);
        float wf[16];
#pragma unroll
        for (uint32_t i = 0; i < 8; ++i) {
            wf[i] = __bfloat162float(wa[i]);
            wf[8u + i] = __bfloat162float(wb[i]);
        }
#pragma unroll
        for (uint32_t b = 0; b < NB; ++b) {
            if (b >= batch) break;  // NB is rounded up; row b>=batch is not ours to read
            const float* xr = x + (size_t)b * in_dim + base;
            float4 x0 = *reinterpret_cast<const float4*>(xr);
            float4 x1 = *reinterpret_cast<const float4*>(xr + 4);
            float4 x2 = *reinterpret_cast<const float4*>(xr + 8);
            float4 x3 = *reinterpret_cast<const float4*>(xr + 12);
            acc[b] += wf[0] * x0.x + wf[1] * x0.y + wf[2] * x0.z + wf[3] * x0.w
                    + wf[4] * x1.x + wf[5] * x1.y + wf[6] * x1.z + wf[7] * x1.w
                    + wf[8] * x2.x + wf[9] * x2.y + wf[10] * x2.z + wf[11] * x2.w
                    + wf[12] * x3.x + wf[13] * x3.y + wf[14] * x3.z + wf[15] * x3.w;
        }
    }
#pragma unroll
    for (uint32_t b = 0; b < NB; ++b)
        for (uint32_t s = 16; s; s >>= 1) acc[b] += __shfl_xor_sync(~0u, acc[b], s);
    if (lane < batch) {
        float v = acc[lane];  // NB-bounded select, pd_f16_gemv_kernel's epilogue
        if (bias) v += bias[o];
        y[(size_t)lane * out_dim + o] = v;
    }
}

PD_EXPORT
int pd_bf16_gemv_mr_f32(const void* w, const void* bias, const void* x, void* y,
                        uint32_t in_dim, uint32_t out_dim, uint32_t batch,
                        void* stream) {
    if (out_dim == 0 || in_dim == 0 || batch == 0) return 0;
    // outside the decode band or a ragged in_dim: not this kernel's shape -
    // the caller keeps the tile GEMM
    if (batch < 2u || batch > 8u || (in_dim & 15u)) return -2;
    uint32_t grid = (out_dim + 7u) / 8u;
    cudaStream_t st = (cudaStream_t)stream;
    const __nv_bfloat16* wp = (const __nv_bfloat16*)w;
    const float* bp = (const float*)bias;
    const float* xp = (const float*)x;
    float* yp = (float*)y;
    if (batch <= 2u)
        pd_pdl_go(pd_bf16_gemv_mr_f32_kernel<2u>, grid, 256u, 0u, st, wp, bp, xp,
                  yp, in_dim, out_dim, batch);
    else if (batch <= 4u)
        pd_pdl_go(pd_bf16_gemv_mr_f32_kernel<4u>, grid, 256u, 0u, st, wp, bp, xp,
                  yp, in_dim, out_dim, batch);
    else
        pd_pdl_go(pd_bf16_gemv_mr_f32_kernel<8u>, grid, 256u, 0u, st, wp, bp, xp,
                  yp, in_dim, out_dim, batch);
    return pd_launch_status();
}

// Register-tiled f32 GEMM over a bf16 weight plane: y[r][o] = sum_k w[o][k] *
// x[r][k]. 64x64 output tile, 16-deep K chunk, 256 threads each holding a 4x4
// register tile. Both shared tiles are stored k-major so the inner product
// walks them without bank conflicts.
#define PD_BF16_BM 64u
#define PD_BF16_BN 64u
#define PD_BF16_BK 16u

__global__ void pd_bf16_gemm_f32_kernel(
    const __nv_bfloat16* __restrict__ w, const float* __restrict__ bias,
    const float* __restrict__ x, float* __restrict__ y,
    uint32_t in_dim, uint32_t out_dim, uint32_t batch) {
    __shared__ float sw[PD_BF16_BK][PD_BF16_BM];
    __shared__ float sx[PD_BF16_BK][PD_BF16_BN];
    uint32_t o0 = blockIdx.x * PD_BF16_BM;
    uint32_t n0 = blockIdx.y * PD_BF16_BN;
    uint32_t tid = threadIdx.x;
    uint32_t ty = tid >> 4, tx = tid & 15u;   // 16x16 thread grid
    float acc[4][4] = {{0.f}};
    PD_PDL_ARM();
    for (uint32_t k0 = 0; k0 < in_dim; k0 += PD_BF16_BK) {
        // stage W[o0..o0+64)[k0..k0+16) transposed into sw[k][o]
        for (uint32_t idx = tid; idx < PD_BF16_BM * PD_BF16_BK; idx += blockDim.x) {
            uint32_t ol = idx >> 4, kk = idx & 15u;
            uint32_t o = o0 + ol, k = k0 + kk;
            sw[kk][ol] = (o < out_dim && k < in_dim)
                ? __bfloat162float(w[(size_t)o * in_dim + k]) : 0.0f;
        }
        // stage X[n0..n0+64)[k0..k0+16) transposed into sx[k][n]
        for (uint32_t idx = tid; idx < PD_BF16_BN * PD_BF16_BK; idx += blockDim.x) {
            uint32_t nl = idx >> 4, kk = idx & 15u;
            uint32_t n = n0 + nl, k = k0 + kk;
            sx[kk][nl] = (n < batch && k < in_dim)
                ? x[(size_t)n * in_dim + k] : 0.0f;
        }
        __syncthreads();
        #pragma unroll
        for (uint32_t kk = 0; kk < PD_BF16_BK; ++kk) {
            float a[4], b[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i) a[i] = sw[kk][ty * 4 + i];
            #pragma unroll
            for (int j = 0; j < 4; ++j) b[j] = sx[kk][tx * 4 + j];
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                #pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        uint32_t o = o0 + ty * 4 + i;
        if (o >= out_dim) continue;
        float bv = bias ? bias[o] : 0.0f;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            uint32_t n = n0 + tx * 4 + j;
            if (n < batch) y[(size_t)n * out_dim + o] = acc[i][j] + bv;
        }
    }
}

PD_EXPORT
int pd_bf16_gemm_f32(const void* w, const void* bias, const void* x, void* y,
                     uint32_t in_dim, uint32_t out_dim, uint32_t batch,
                     void* stream) {
    if (out_dim == 0 || in_dim == 0 || batch == 0) return 0;
    dim3 grid((out_dim + PD_BF16_BM - 1) / PD_BF16_BM,
              (batch + PD_BF16_BN - 1) / PD_BF16_BN);
    pd_pdl_go(pd_bf16_gemm_f32_kernel, grid, dim3(256u), 0u, (cudaStream_t)stream,
              (const __nv_bfloat16*)w, (const float*)bias, (const float*)x,
              (float*)y, in_dim, out_dim, batch);
    return pd_launch_status();
}

// ------------------------------------------------------------- mma prefill arm
// Tensor-core GEMM for the prefill band (batch > 8). The f32-FMA
// tile above runs the paddleocr prefill chunk at ~18 TFLOP/s - several x under
// the f32 roof and ~10x under what a bf16 tensor-core GEMM reaches, and its
// (out/64, rows/64) grid drops to 68 blocks on the 256-wide K/V planes. This
// arm is the elected pd_f16_gemm_mma_kernel ring (same tile census) with two
// deltas: the A plane loads bf16 weights (16-bit lanes - the cp.async staging
// and ldmatrix moves are byte-identical), and the B plane stages f32
// activations through a __float2bfloat16 cast (see the file header for why
// that class is sanctioned). Helpers are self-named pd_bf16m_* because this
// header is included before int8_mma/f16_dense in the pack blob, so their
// equivalents don't exist yet at this point in the translation unit.
//
// Arch note: the guarded-out body (< sm_80) compiles to a silent no-op - the
// pf5 lesson. The validated-arch floor is sm_86 (allowlist), so every arch
// this pack actually serves has the real body; do not ship this slot to a
// pre-Ampere pack without a runtime arch check.
#if defined(__CUDA_ARCH__)
#define PD_BF16MMA_OK (__CUDA_ARCH__ >= 800)
#else
#define PD_BF16MMA_OK 0
#endif

#if PD_BF16MMA_OK
__device__ __forceinline__ void pd_bf16m_ldm_x4(const void* p, uint32_t& r0,
                                                uint32_t& r1, uint32_t& r2,
                                                uint32_t& r3) {
    const unsigned a = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(a));
}
__device__ __forceinline__ void pd_bf16m_ldm_x2(const void* p, uint32_t& r0,
                                                uint32_t& r1) {
    const unsigned a = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                 : "=r"(r0), "=r"(r1)
                 : "r"(a));
}
// predicated 16B cp.async: ok==false issues a size-0 copy that zero-fills the
// shared destination without reading gmem (OOB source never dereferenced).
__device__ __forceinline__ void pd_bf16m_cpa16(void* smem, const void* gmem, bool ok) {
    const unsigned sm = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(sm),
                 "l"(gmem), "r"(ok ? 16u : 0u));
}
__device__ __forceinline__ void pd_bf16m_cpa_commit() {
    asm volatile("cp.async.commit_group;");
}
template <int N>
__device__ __forceinline__ void pd_bf16m_cpa_waitN() {
    asm volatile("cp.async.wait_group %0;" ::"n"(N));
}
// D = A*B + D, m16n8k16 bf16xbf16->f32 - same fragment maps as the f16 twin
// (A a0=(m=g,k=2t) a1=(m+8,2t) a2=(m,8+2t) a3=(m+8,8+2t); B b0=(n=g,k=2t)
// b1=(n,8+2t); D d0=(m=g,n=2t) d1=(m,2t+1) d2=(m+8,2t) d3=(m+8,2t+1)).
__device__ __forceinline__ void pd_bf16m_mma(float d[4], const uint32_t a[4],
                                             const uint32_t b[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
#endif

// Ring/compute/store structure is pd_f16_gemm_mma_kernel's, verbatim where the
// data allows: BM out-rows x BN batch-cols per CTA, K staged KT deep with ST
// cp.async buffers on the WEIGHT plane; the activation plane is staged with
// plain vector loads + cast (cp.async cannot convert), which the end-of-loop
// __syncthreads() orders exactly like the async stages. RG x CG register
// micro-tile per warp so each fragment feeds RG*CG mmas. Zero-padding past
// M/N/K keeps ragged tiles correct; the launcher gates K%16==0 so B's float4
// loads stay 16B-aligned (no ragged-K arm here - that class keeps the tile
// GEMM).
//
// QKV=true is the fused q|k|v decode arm (thin-k/v rung): W is the
// load-time-concatenated [q;k;v] plane (M = OQ + 2*OKV rows), and the store
// routes each out-row to its segment plane (Y=q, Yk, Yv - each [N, seg]
// row-major). Everything before the epilogue is untouched, so per out-row
// the result is BIT-identical to the plain arm on that segment. The point:
// a 256-row k/v plane on its own launch is latency-starved (~40 us floor at
// 1-3% of the DRAM roof, any kernel shape - bf16_thin_probe.cu); riding the
// fused grid streams it at the big plane's rate, and 3 launches become 1.
// KS=true is the decode-band K-split arm (bf16 K-split rung): the
// same tile with grid.z K-slabs, each block accumulating only its slab's
// KT-runs and writing an UNnormalized f32 partial plane into Y-as-partials
// ([z][col*M + row], flat over the fused M when QKV data is served - segment
// routing and bias move to the combine). Same numeric class as the f8row/
// int8 ks paths: f32 partial regroup only, per-slab sub-sums identical to
// the unsplit kernel's same k-range.
template <uint32_t BM, uint32_t BN, uint32_t NWARP, uint32_t ST, uint32_t KT,
          uint32_t RG, uint32_t CG, bool QKV = false, bool KS = false>
__global__ void __launch_bounds__(NWARP * 32) pd_bf16_gemm_mma_kernel(
        const __nv_bfloat16* __restrict__ W, const float* __restrict__ X,
        const float* __restrict__ bias, float* __restrict__ Y,
        uint32_t K, uint32_t M, uint32_t N, float* __restrict__ Yk,
        float* __restrict__ Yv, uint32_t OQ, uint32_t OKV) {
#if PD_BF16MMA_OK
    // decode-graph PDL: joins the tick's launch cascade when launched with
    // the attribute (pd_pdl_go); a plain launch makes this a no-op
    PD_PDL_ARM();
    constexpr uint32_t NTH = NWARP * 32u;
    constexpr uint32_t WM = RG * 16u;      // warp tile rows
    constexpr uint32_t WN = CG * 8u;       // warp tile cols
    constexpr uint32_t WR = BM / WM;       // warp-rows in the CTA tile
    constexpr uint32_t WC = BN / WN;       // warp-cols in the CTA tile
    constexpr uint32_t NSUBK = KT / 16u;   // k16 sub-tiles per stage
    constexpr uint32_t KPAD = KT + 8u;     // padded shared K-stride (bf16s)
    constexpr uint32_t H8PR = KT / 8u;     // 8-element groups per staged row
    static_assert(WR * WC == NWARP, "warp grid");
    static_assert(WM * WR == BM && WN * WC == BN, "tile cover");
    static_assert(KT % 16u == 0u, "KT k16-multiple");
    static_assert(ST >= 2u && ST <= 4u, "stage count (ring only)");

    extern __shared__ __align__(16) __nv_bfloat16 pd_bf16m_dyn[];
    auto sh_a = reinterpret_cast<__nv_bfloat16(*)[BM * KPAD]>(pd_bf16m_dyn);
    auto sh_b = reinterpret_cast<__nv_bfloat16(*)[BN * KPAD]>(pd_bf16m_dyn + ST * BM * KPAD);

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u, warp = tid >> 5;
    const uint32_t g = lane >> 2, t = lane & 3u;
    const uint32_t wr = (warp % WR) * WM;   // warp row base within tile
    const uint32_t wc = (warp / WR) * WN;   // warp col base within tile
    const uint32_t row_base = blockIdx.x * BM;
    const uint32_t col_base = blockIdx.y * BN;
    const __nv_bfloat16 zero = __float2bfloat16(0.0f);

    // KS slab bounds: gridDim.z slabs measured in 512-ELEMENT blocks, not
    // KT-runs - 512 is a multiple of every config's KT (32/64/128), so the
    // slab boundaries are a pure function of K and identical across
    // configs. That is what keeps the fused-qkv export's per-segment
    // bit-identity: the qkv tiers run KT=64 where the plain tiers run
    // KT=128, and a KT-derived slab regroups the two sides differently
    // (caught by gpu_nemotron_kernels' to_bits gate at bt=9, 1-ulp drift).
    // k_eff replaces K in every stage guard, so a slab's ragged tail
    // zero-fills exactly the way the plain kernel's K tail does. The
    // launcher guarantees every slab is non-empty.
    uint32_t k_begin = 0u, k_eff = K;
    if constexpr (KS) {
        const uint32_t nb = (K + 511u) / 512u;
        const uint32_t per = (nb + gridDim.z - 1u) / gridDim.z;
        k_begin = blockIdx.z * per * 512u;
        const uint32_t ke = k_begin + per * 512u;
        k_eff = ke < K ? ke : K;
    }

    // stage kt's A (weights, async) and B (acts, cast) planes into buffer `buf`
    auto stage = [&](uint32_t k0, uint32_t buf) {
        #pragma unroll
        for (uint32_t i = tid; i < BM * H8PR; i += NTH) {
            const uint32_t row = i / H8PR, h8 = (i % H8PR) * 8u, gk = k0 + h8;
            const bool ok = (row_base + row) < M && gk + 8u <= k_eff;
            __nv_bfloat16* dst = &sh_a[buf][row * KPAD + h8];
            const __nv_bfloat16* src = W + (size_t)(row_base + row) * K + gk;
            pd_bf16m_cpa16(dst, src, ok);   // K%16==0: no ragged 8-group exists
        }
        #pragma unroll
        for (uint32_t i = tid; i < BN * H8PR; i += NTH) {
            const uint32_t col = i / H8PR, h8 = (i % H8PR) * 8u, gk = k0 + h8;
            const bool colok = (col_base + col) < N;
            __nv_bfloat16* dst = &sh_b[buf][col * KPAD + h8];
            if (colok && gk + 8u <= k_eff) {
                const float* src = X + (size_t)(col_base + col) * K + gk;
                const float4 v0 = *reinterpret_cast<const float4*>(src);
                const float4 v1 = *reinterpret_cast<const float4*>(src + 4);
                __nv_bfloat16 tmp[8] = {
                    __float2bfloat16(v0.x), __float2bfloat16(v0.y),
                    __float2bfloat16(v0.z), __float2bfloat16(v0.w),
                    __float2bfloat16(v1.x), __float2bfloat16(v1.y),
                    __float2bfloat16(v1.z), __float2bfloat16(v1.w)};
                *reinterpret_cast<int4*>(dst) = *reinterpret_cast<const int4*>(tmp);
            } else {
                #pragma unroll
                for (uint32_t e = 0; e < 8u; ++e)
                    dst[e] = (colok && gk + e < k_eff)
                        ? __float2bfloat16(X[(size_t)(col_base + col) * K + gk + e])
                        : zero;
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
                pd_bf16m_ldm_x4(&sh_a[buf][(wr + rg * 16u + a_roff) * KPAD + ko + a_kof],
                                a[rg][0], a[rg][1], a[rg][2], a[rg][3]);
            uint32_t b[CG][2];
            #pragma unroll
            for (uint32_t cg = 0; cg < CG; ++cg)
                pd_bf16m_ldm_x2(&sh_b[buf][(wc + cg * 8u + l7) * KPAD + ko + b_kof],
                                b[cg][0], b[cg][1]);
            #pragma unroll
            for (uint32_t rg = 0; rg < RG; ++rg)
                #pragma unroll
                for (uint32_t cg = 0; cg < CG; ++cg)
                    pd_bf16m_mma(acc[rg][cg], a[rg], b[cg]);
        }
    };

    // ST-deep ring: buffer p computes while up to ST-1 stages stream in. The
    // B-plane's plain smem stores ride the same barriers: they land in a
    // buffer no compute reads until after a later __syncthreads().
    #pragma unroll
    for (uint32_t s = 0; s < ST - 1u; ++s) {
        const uint32_t k0 = k_begin + s * KT;
        if (k0 < k_eff) stage(k0, s);
        pd_bf16m_cpa_commit();
    }
    uint32_t p = 0;
    for (uint32_t k0 = k_begin; k0 < k_eff; k0 += KT) {
        const uint32_t pre = k0 + (ST - 1u) * KT;
        if (pre < k_eff) stage(pre, (p + ST - 1u) % ST);
        pd_bf16m_cpa_commit();
        pd_bf16m_cpa_waitN<(int)ST - 1>();
        __syncthreads();
        compute(p);
        __syncthreads();
        p = (p + 1u) % ST;
    }

    // store: element (m=out row, n=batch col) -> Y[n*M + m]; bias is per
    // out-row. Each element is written by exactly one thread. The QKV arm
    // routes rows to their segment plane instead (see the header comment).
    auto put = [&](uint32_t r, uint32_t c, float v) {
        if constexpr (KS) {
            // partial plane z, flat [c][r] over the (possibly fused) M -
            // bias and QKV segment routing happen once, in the combine
            Y[(size_t)blockIdx.z * M * N + (size_t)c * M + r] = v;
        } else if constexpr (!QKV) {
            Y[(size_t)c * M + r] = v;
        } else if (r < OQ) {
            Y[(size_t)c * OQ + r] = v;
        } else if (r < OQ + OKV) {
            Yk[(size_t)c * OKV + (r - OQ)] = v;
        } else {
            Yv[(size_t)c * OKV + (r - OQ - OKV)] = v;
        }
    };
    #pragma unroll
    for (uint32_t rg = 0; rg < RG; ++rg) {
        const uint32_t r0 = row_base + wr + rg * 16u + g;
        const uint32_t r8 = r0 + 8u;
        const float b0 = (bias && r0 < M) ? bias[r0] : 0.0f;
        const float b8 = (bias && r8 < M) ? bias[r8] : 0.0f;
        #pragma unroll
        for (uint32_t cg = 0; cg < CG; ++cg) {
            const uint32_t c0 = col_base + wc + cg * 8u + 2u * t;
            const uint32_t c1 = c0 + 1u;
            if (r0 < M) {
                if (c0 < N) put(r0, c0, acc[rg][cg][0] + b0);
                if (c1 < N) put(r0, c1, acc[rg][cg][1] + b0);
            }
            if (r8 < M) {
                if (c0 < N) put(r8, c0, acc[rg][cg][2] + b8);
                if (c1 < N) put(r8, c1, acc[rg][cg][3] + b8);
            }
        }
    }
#else
    (void)W; (void)X; (void)bias; (void)Y; (void)K; (void)M; (void)N;
    (void)Yk; (void)Yv; (void)OQ; (void)OKV;
#endif
}

// ── decode-band K-split (bf16 K-split rung) ──────────────────────
// At c32 the attn projections launch 42 (wo, BM=64) and 72 (fused qkv)
// blocks on a 188-SM die and run at 24-37% of the DRAM roof, while cuBLASLt
// split-K covers the same shapes at ~69%. Careful with the batch-scaling
// "grid-fill ladder" reading that argues K-split off: that ladder scales
// BATCH, and its GB/s-wt metric divides SINGLE-plane bytes by time, so a
// flat curve means time held constant while physical traffic doubled - CTA
// starvation, exactly what a K-split cures at constant bytes.
//
// Scratch: the mma path runs only inside captured decode ticks (nemotron
// step_replay captures the first tick per r), so f8row's lazy
// grow-outside-capture cudaMalloc can never allocate here. A fixed static
// __device__ plane (16 MB, pd_smp_scr precedent: single-engine-stream
// serving contract) is capture-safe by construction; the fit gate below is
// structural - K-split only matters when the 2D grid is small, and a small
// grid bounds nz*M*N.
#define PD_BF16KS_ELEMS (8u * 8192u * 64u)
__device__ float pd_bf16ks_part[PD_BF16KS_ELEMS];

// fixed-order partial combine. Plain arm: y[c*M+r] = sum_z part + bias[r].
// QKV arm (yk != nullptr): the fused row is routed to its segment plane,
// exactly pd_bf16_gemm_mma_kernel's put() - bias is qkv-null upstream.
__global__ void pd_bf16_ks_combine_kernel(
        const float* __restrict__ part, const float* __restrict__ bias,
        float* __restrict__ y, float* __restrict__ yk, float* __restrict__ yv,
        uint32_t n, uint32_t nz, uint32_t m, uint32_t oq, uint32_t okv) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float acc = 0.0f;
    for (uint32_t z = 0; z < nz; ++z) acc += part[(size_t)z * n + i];
    const uint32_t r = i % m, c = i / m;
    if (bias) acc += bias[r];
    if (!yk) {
        y[i] = acc;
    } else if (r < oq) {
        y[(size_t)c * oq + r] = acc;
    } else if (r < oq + okv) {
        yk[(size_t)c * okv + (r - oq)] = acc;
    } else {
        yv[(size_t)c * okv + (r - oq - okv)] = acc;
    }
}

// nz policy. The grid-fill test (blocks2d < 2 CTAs/SM) only decides whether
// to split; the split COUNT is a pure function of K: slabs of 512 elements
// (a multiple of every config's KT), capped at 8 (the scratch's plane
// budget). That purity is what keeps pd_bf16_qkv_gemm_mma's documented
// per-segment bit-identity intact - the fused plane and each segment see
// the same K, hence the same slab boundaries, hence identical regroup
// (gpu_nemotron_kernels.rs asserts to_bits equality; the qkv tiers run a
// different KT than the plain tiers, which is why the unit is elements,
// not KT-runs). The fit clamp can, in principle, diverge nz for M*batch
// beyond ~9k*64 - no shipped bit-compared pairing lives there; every
// nemotron shape fits. 0 = stay unsplit. Kill: PADDOCK_NO_BF16_KSPLIT.
static uint32_t pd_bf16ks_nz(uint32_t blocks2d, uint32_t in_dim,
                             uint32_t out_m, uint32_t batch) {
    static const bool off = pd_env("PADDOCK_NO_BF16_KSPLIT") != nullptr;
    if (off) return 0u;
    static int nsm = 0;
    if (nsm == 0) {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
        if (nsm <= 0) nsm = 128;
    }
    if (blocks2d >= (uint32_t)nsm * 2u) return 0u;
    const uint32_t nb = (in_dim + 511u) / 512u;
    uint32_t nz = nb > 8u ? 8u : nb;
    while (nz >= 2u && (size_t)nz * out_m * batch > (size_t)PD_BF16KS_ELEMS)
        --nz;
    if (nz < 2u) return 0u;
    const uint32_t per = (nb + nz - 1u) / nz;
    return (nb + per - 1u) / per;   // every slab non-empty
}

static float* pd_bf16ks_scr() {
    static float* p = [] {
        void* d = nullptr;
        if (cudaGetSymbolAddress(&d, pd_bf16ks_part) != cudaSuccess) d = nullptr;
        return (float*)d;
    }();
    return p;
}

template <uint32_t BM, uint32_t BN, uint32_t NW, uint32_t ST, uint32_t KT,
          uint32_t RG, uint32_t CG>
static int pd_bf16_mma_cfg(const __nv_bfloat16* w, const float* x,
                           const float* bias, float* y, uint32_t in_dim,
                           uint32_t out_dim, uint32_t batch, cudaStream_t st) {
    constexpr uint32_t KPAD = KT + 8u;
    constexpr uint32_t smem = ST * (BM * KPAD + BN * KPAD) * 2u;
    static bool set = false;
    if (!set) {
        cudaFuncSetAttribute(pd_bf16_gemm_mma_kernel<BM, BN, NW, ST, KT, RG, CG>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        cudaFuncSetAttribute(
            pd_bf16_gemm_mma_kernel<BM, BN, NW, ST, KT, RG, CG, false, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        set = true;
    }
    dim3 grid((out_dim + BM - 1u) / BM, (batch + BN - 1u) / BN);
    // K-split arm first (see the block comment above pd_bf16ks_part): same
    // tile, grid.z K-slabs into the static partials plane, fixed-order
    // combine applies the bias. Self-gates on grid fill / slab depth / fit.
    const uint32_t nz = pd_bf16ks_nz(grid.x * grid.y, in_dim, out_dim, batch);
    float* part = nz >= 2u ? pd_bf16ks_scr() : nullptr;
    if (part) {
        dim3 gz(grid.x, grid.y, nz);
        pd_bf16_gemm_mma_kernel<BM, BN, NW, ST, KT, RG, CG, false, true>
            <<<gz, NW * 32u, smem, st>>>(w, x, nullptr, part, in_dim, out_dim,
                                         batch, nullptr, nullptr, 0u, 0u);
        const uint32_t n = out_dim * batch;
        pd_bf16_ks_combine_kernel<<<(n + 255u) / 256u, 256, 0, st>>>(
            part, bias, y, nullptr, nullptr, n, nz, out_dim, 0u, 0u);
        return (int)cudaGetLastError();
    }
    pd_bf16_gemm_mma_kernel<BM, BN, NW, ST, KT, RG, CG><<<grid, NW * 32u, smem, st>>>(
            w, x, bias, y, in_dim, out_dim, batch, nullptr, nullptr, 0u, 0u);
    return (int)cudaGetLastError();
}

// The QKV=true twin: grid covers the fused row count, launch rides the PDL
// cascade (this arm only runs inside decode ticks, where the launch train
// is armed; pd_pdl_go degrades to a plain launch elsewhere).
template <uint32_t BM, uint32_t BN, uint32_t NW, uint32_t ST, uint32_t KT,
          uint32_t RG, uint32_t CG>
static int pd_bf16_qkv_cfg(const __nv_bfloat16* w, const float* x, float* yq,
                           float* yk, float* yv, uint32_t in_dim, uint32_t oq,
                           uint32_t okv, uint32_t batch, cudaStream_t st) {
    constexpr uint32_t KPAD = KT + 8u;
    constexpr uint32_t smem = ST * (BM * KPAD + BN * KPAD) * 2u;
    static bool set = false;
    if (!set) {
        cudaFuncSetAttribute(
            pd_bf16_gemm_mma_kernel<BM, BN, NW, ST, KT, RG, CG, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        cudaFuncSetAttribute(
            pd_bf16_gemm_mma_kernel<BM, BN, NW, ST, KT, RG, CG, true, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        set = true;
    }
    const uint32_t m = oq + 2u * okv;
    dim3 grid((m + BM - 1u) / BM, (batch + BN - 1u) / BN);
    // K-split arm first (same shape as the plain cfg): partials are flat
    // over the fused m; the combine does the q|k|v segment routing that
    // put() would have done. The partial kernel keeps the PDL arm - it is
    // the tick-cascade member; the combine is stream-ordered behind it.
    const uint32_t nz = pd_bf16ks_nz(grid.x * grid.y, in_dim, m, batch);
    float* part = nz >= 2u ? pd_bf16ks_scr() : nullptr;
    if (part) {
        dim3 gz(grid.x, grid.y, nz);
        pd_pdl_go(pd_bf16_gemm_mma_kernel<BM, BN, NW, ST, KT, RG, CG, true, true>,
                  gz, dim3(NW * 32u), smem, st, w, x, (const float*)nullptr,
                  part, in_dim, m, batch, yk, yv, oq, okv);
        const uint32_t n = m * batch;
        pd_bf16_ks_combine_kernel<<<(n + 255u) / 256u, 256, 0, st>>>(
            part, nullptr, yq, yk, yv, n, nz, m, oq, okv);
        return (int)cudaGetLastError();
    }
    pd_pdl_go(pd_bf16_gemm_mma_kernel<BM, BN, NW, ST, KT, RG, CG, true>, grid,
              dim3(NW * 32u), smem, st, w, x, (const float*)nullptr, yq,
              in_dim, m, batch, yk, yv, oq, okv);
    return (int)cudaGetLastError();
}

// Prefill entry. Declines (-2) on the decode band (the mr kernel's shape) and
// on ragged in_dim (the tile GEMM stays correct there) - the Rust route treats
// nonzero as "keep the fallback". No PDL arm: this band only runs in the eager
// chunked-prefill pass, never inside a captured decode graph. Configs are the
// f16 census picks (narrow tile below 64 cols, fat 128x128 above) - with a
// grid-fill gate on top: the fat tile's grid on the paddleocr-vl
// decoder's square 1024-wide planes is ceil(1024/128)^2 = 64 blocks, a third
// of the PRO 6000's SMs in a single wave, which held the tick's GEMM band at
// ~56 effective TFLOPS. When the fat grid
// cannot fill the device once, the narrow tile quadruples the block count at
// the same per-element K-walk (summation order is config-independent - RG/CG
// move warp tile ownership, not the k sequence), so it wins on occupancy.
PD_EXPORT
int pd_bf16_gemm_mma(const void* w, const void* bias, const void* x, void* y,
                     uint32_t in_dim, uint32_t out_dim, uint32_t batch,
                     void* stream) {
    if (out_dim == 0 || in_dim == 0 || batch == 0) return 0;
    // batch 2..8 is admitted: on big-out planes (lm_head 4096x100352) the
    // b<=16 tier config holds the wall at b=8 (556.8 us cold vs the mr
    // GEMV's 1020.4 on a 4-plane rotation; NB=8 is the mr kernel's
    // issue-bound arm). The ENGINE elects
    // by out width - small-out planes keep the mr class this entry used to
    // refuse into.
    if (batch < 2u || (in_dim & 15u)) return -2;
    cudaStream_t st = (cudaStream_t)stream;
    const __nv_bfloat16* wp = (const __nv_bfloat16*)w;
    const float* xp = (const float*)x;
    const float* bp = (const float*)bias;
    float* yp = (float*)y;
    static int sms = -1;
    if (sms < 0) {
        int dev = 0;
        cudaGetDevice(&dev);
        if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev)
                != cudaSuccess || sms <= 0)
            sms = 1; // query failure: fall through to the batch-only rule
    }
    // Decode-band tier: at batch<=32
    // the BN=32/KT=128/ST=2 2-CTA config streams the probed shapes ~40%
    // faster than the 64x64 tile (q 4096x2688 b32: 33.2 vs 45.5 us; wo
    // 2688x4096: 46.8 vs 61.4) - BN=64 pads half the B tile away at these
    // widths, and the deeper KT holds more bytes in flight per CTA (the
    // latency lever). Same per-element k-walk as every other config
    // (configs move tile ownership, never the k sequence), so the pick is
    // bit-neutral.
    // A later DRAM-honest 12-clone sweep: <64,32,8,ST=4,KT=64,2,1> beats this
    // arm on both served shapes -- wo 2688x4096 b32 40.9 us vs 46.8 -- and it
    // is the same tile with a DEEPER pipeline and a SHALLOWER K tile, so the
    // earlier pick (which swept BM/BN but held ST/KT) could not see it.
    // Configs move tile ownership, never the k sequence, so the pick stays
    // bit-neutral.
    // CORRECTION to that reading: the grid-fill ladder's "538 GB/s flat
    // across b32..b256" looked like saturation, but a GB/s-wt metric divides
    // SINGLE-plane bytes by time while batch doubles the PHYSICAL weight
    // traffic - flat GB/s-wt means time held constant as bytes doubled, i.e.
    // aggregate bandwidth scaled with CTA count. That is starvation, and the
    // K-split it argued against is exactly what pays: wo b32 40.9 -> 22.2 us
    // (538 -> 990 GB/s), kv b16 26.6 -> 10.2, qkv32 28.7 -> 26.5 (already at
    // 55%). The KS arm below the tier ladder is the cure; the row-stride
    // observation stands only as the reason qkv (k=2688) streams faster per
    // CTA than wo (k=4096).
    // BN=32 pads three quarters of the B tile away below b16, which is why
    // the ST=4/KT=64 pick measured +0.46% at c32 and -0.77% at c8 when it
    // was applied to the whole batch<=32 band.
    // Band it at the same 16 boundary the qkv arm already uses: the narrow
    // tile keeps the small widths, the deep-pipeline tile takes 16<b<=32.
    if (batch <= 16u)
        return pd_bf16_mma_cfg<32u, 32u, 4u, 2u, 128u, 1u, 2u>(
                wp, xp, bp, yp, in_dim, out_dim, batch, st);
    if (batch <= 32u)
        return pd_bf16_mma_cfg<64u, 32u, 8u, 4u, 64u, 2u, 1u>(
                wp, xp, bp, yp, in_dim, out_dim, batch, st);
    const uint32_t fat_blocks =
            ((out_dim + 127u) / 128u) * ((batch + 127u) / 128u);
    if (batch <= 64u || fat_blocks < (uint32_t)sms)
        return pd_bf16_mma_cfg<64u, 64u, 8u, 3u, 32u, 2u, 2u>(
                wp, xp, bp, yp, in_dim, out_dim, batch, st);
    return pd_bf16_mma_cfg<128u, 128u, 8u, 3u, 32u, 4u, 4u>(
            wp, xp, bp, yp, in_dim, out_dim, batch, st);
}

// Fused q|k|v decode-band entry (thin-k/v rung): one launch over
// the load-time-concatenated [q;k;v] plane (in_dim x (oq + 2*okv)) against
// the shared x, segmented store into yq/yk/yv ([batch, seg] each). Per
// out-row bit-identical to pd_bf16_gemm_mma on the matching segment. The
// probe's per-band picks: b8 20.8 us / b16 26.6 / b32 34.6 at the nemotron
// fused shape (4608x2688) vs 131 us for the three separate launches at b32
// - the thin k/v rows were latency-starved on their own grids. Declines
// (-2) batch<2 (the serial row keeps its per-segment GEMVs) and ragged
// in_dim.
PD_EXPORT
int pd_bf16_qkv_gemm_mma(const void* w, const void* x, void* yq, void* yk,
                         void* yv, uint32_t in_dim, uint32_t oq, uint32_t okv,
                         uint32_t batch, void* stream) {
    if (in_dim == 0 || (oq == 0 && okv == 0) || batch == 0) return 0;
    if (batch < 2u || (in_dim & 15u)) return -2;
    cudaStream_t st = (cudaStream_t)stream;
    const __nv_bfloat16* wp = (const __nv_bfloat16*)w;
    const float* xp = (const float*)x;
    float* qp = (float*)yq;
    float* kp = (float*)yk;
    float* vp = (float*)yv;
    if (batch <= 8u)
        return pd_bf16_qkv_cfg<32u, 32u, 4u, 2u, 128u, 1u, 2u>(
                wp, xp, qp, kp, vp, in_dim, oq, okv, batch, st);
    if (batch <= 16u)
        return pd_bf16_qkv_cfg<32u, 32u, 4u, 3u, 64u, 1u, 2u>(
                wp, xp, qp, kp, vp, in_dim, oq, okv, batch, st);
    // same sweep: ST=4/KT=64 over ST=2/KT=128 on the fused plane -
    // 4608x2688 b32 28.7 us vs 34.6, bit-neutral (config, not k order).
    return pd_bf16_qkv_cfg<64u, 32u, 8u, 4u, 64u, 2u, 1u>(
            wp, xp, qp, kp, vp, in_dim, oq, okv, batch, st);
}

// bf16 -> f32 widen with the DequantF32Fn shape (src, dst, n_blocks, stream),
// 32 elements per "block" so it slots into the engine's dequant_for table next
// to the real quant types. Feeds dequant_slice, i.e. the single-row embedding
// gather off a bf16 token_embd.
__global__ void pd_bf16_dequant_f32_kernel(const __nv_bfloat16* __restrict__ src,
                                           float* __restrict__ dst, uint64_t n) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = __bfloat162float(src[i]);
}

PD_EXPORT
int pd_bf16_dequant_f32(const void* src, void* dst, uint64_t n_blocks, void* stream) {
    if (n_blocks == 0) return 0;
    uint64_t n = n_blocks * 32ull;
    uint32_t threads = 256;
    uint64_t blocks = (n + threads - 1) / threads;
    pd_bf16_dequant_f32_kernel<<<(uint32_t)blocks, threads, 0, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)src, (float*)dst, n);
    return pd_launch_status();
}

// bf16 twin of pd_embed_gather_q8: gathers device-selected token rows straight
// out of a bf16 embedding table with the fused output scale. Graph-capturable
// (token ids read from device memory), same as the Q8_0 one.
__global__ void pd_embed_gather_bf16_kernel(const __nv_bfloat16* __restrict__ table,
                                            const uint32_t* __restrict__ tokens,
                                            float* __restrict__ out, uint32_t embd,
                                            float scale) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t t = blockIdx.y;
    if (i >= embd) return;
    out[(size_t)t * embd + i] =
        __bfloat162float(table[(size_t)tokens[t] * embd + i]) * scale;
}

PD_EXPORT
int pd_embed_gather_bf16(const void* table, const void* tokens, void* out,
                         uint32_t embd, uint32_t n_tokens, float scale, void* stream) {
    if (embd == 0 || n_tokens == 0) return 0;
    uint32_t threads = 256;
    dim3 grid((embd + threads - 1) / threads, n_tokens);
    pd_embed_gather_bf16_kernel<<<grid, threads, 0, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)table, (const uint32_t*)tokens, (float*)out, embd, scale);
    return pd_launch_status();
}
