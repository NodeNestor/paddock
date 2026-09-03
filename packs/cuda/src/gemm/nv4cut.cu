// nv4cut: the checkpoint-native NVFP4 decode GEMM (task: nvfp4 ITL lever 1).
//
// Why this EXISTS. The qwen3.8-27b NVFP4 checkpoint ships `weight_packed`
// (e2m1, 0.5 B/param) for the MLP and nothing else; every attention/GDN
// projection and the lm_head are FP8. Our decode lane served the MLP from
// e4m3 f8t tile planes, i.e. at 1.0 B/param, so the decode tick read 25.6 GB
// where the checkpoint's own nibbles only cost 17.6. This TU serves those 56
// layers straight from those nibbles.
//
// Same kernel class FlashInfer's NVFP4 linear path uses
// (FlashInferCuteDslNvFp4LinearKernel ->
// Sm100BlockScaledPersistentDenseGemmKernel): CUTLASS's SM100 block-scaled
// mainloop, `KernelTmaWarpSpecialized1SmNvf4Sm100`, e2m1 x e2m1 with e4m3
// scale factors every 16 elements. It runs on sm_100a - see the note in
// quant/nvf4.cuh.
//
// LAYOUTS, all three bit-exact against pd_nvf4_dequant's convention (with a
// row-major-SF negative control that fails as it must):
//   - weights: `Nvf4Plane::data` UNCHANGED - [n][k/2] bytes, element 2j in
//     the low nibble, which is exactly CUTLASS's ColumnMajor (k-major) B.
//     No weight repack, no extra weight bytes.
//   - scales: one e4m3 per 16 elements, but CUTLASS wants them in its
//     blocked SF layout, so `pd_nv4cut_sf_repack` scatters the plane's
//     [n][k/16] vector through
//     `Sm1xxBlockScaledConfig<16>::tile_atom_to_shape_SFB` once at load.
//     The layout is CUTE_HOST_DEVICE, so the scatter cannot drift from the
//     mainloop's view of it.
//   - activations: `pd_nv4cut_quant_a` writes e2m1 nibbles plus SFA in the
//     same blocked layout. Per-16 dynamic scale, no global scale: the
//     checkpoint's `input_global_scale` exists to keep a static scale inside
//     e4m3's range, and a dynamic per-block scale is both free here and
//     strictly finer. Underflow is handled by clamping the scale to e4m3's
//     smallest subnormal, which cannot clip (amax < 6*min_sub => q <= 6).
//
// alpha is the plane's per-tensor `scale2` (= 1/weight_global_scale) and is
// folded in the epilogue, so D comes out already dequantized, bf16.
//
// Compiled only for sm_100a as its own TU, like cutgemm.cu; build.sh links
// it. Guarded by PD_CUTGEMM - without CUTLASS every entry is a
// cudaErrorNotSupported stub and the engine's has_* gate stays false.
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cstdlib>
#include <cstdio>

#ifdef PD_CUTGEMM
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cute/tensor.hpp"

using namespace cute;

namespace pdnv4 {

using ElementA = cutlass::float_e2m1_t;
using ElementB = cutlass::float_e2m1_t;
using SFType   = cutlass::float_ue4m3_t;
using ElementD = cutlass::bfloat16_t;
using ElementAcc = float;

static constexpr int SFVEC = 16;
using BlkCfg = cutlass::detail::Sm1xxBlockScaledConfig<SFVEC>;

// (mn, k) -> the blocked SF layout both SFA and SFB use (they are the same
// function of their own leading dim, which is why one helper serves both).
static CUTE_HOST_DEVICE auto sf_layout(int mn, int k) {
    return BlkCfg::tile_atom_to_shape_SFA(cute::make_shape(mn, 1, k, 1));
}
using SFLayout = decltype(sf_layout(0, 0));

// ---- scale-factor scatter (load time, once per plane) -------------------
__global__ void pd_nv4cut_sf_repack_kernel(const unsigned char* __restrict__ src,
                                           unsigned char* __restrict__ dst,
                                           SFLayout lay, unsigned int nblk,
                                           unsigned int mn) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)mn * nblk) return;
    const unsigned int r = (unsigned int)(i / nblk);
    const unsigned int b = (unsigned int)(i - (size_t)r * nblk);
    dst[lay(cute::make_coord((int)r, (int)(b * SFVEC), 0))] = src[i];
}

// ---- activation quantize (per tick) -------------------------------------
// e2m1 magnitudes {0,.5,1,1.5,2,3,4,6}; round to nearest, ties away from 0.
static __device__ __forceinline__ unsigned int e2m1_enc(float v) {
    const unsigned int s = (v < 0.0f) ? 8u : 0u;
    const float a = fabsf(v);
    unsigned int m = 7u;
    if (a < 0.25f)      m = 0u;
    else if (a < 0.75f) m = 1u;
    else if (a < 1.25f) m = 2u;
    else if (a < 1.75f) m = 3u;
    else if (a < 2.5f)  m = 4u;
    else if (a < 3.5f)  m = 5u;
    else if (a < 5.0f)  m = 6u;
    return s | m;
}

__global__ void pd_nv4cut_quant_a_kernel(const float* __restrict__ x,
                                         unsigned char* __restrict__ q,
                                         unsigned char* __restrict__ sf,
                                         SFLayout lay, unsigned int k,
                                         unsigned int m) {
    const unsigned int nblk = k / SFVEC;
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)m * nblk) return;
    const unsigned int r = (unsigned int)(i / nblk);
    const unsigned int b = (unsigned int)(i - (size_t)r * nblk);
    const float4* src = (const float4*)(x + (size_t)r * k + (size_t)b * SFVEC);
    float4 v0 = src[0], v1 = src[1], v2 = src[2], v3 = src[3];
    float a = 0.0f;
    #pragma unroll
    for (int t = 0; t < 4; ++t) {
        const float4 v = t == 0 ? v0 : (t == 1 ? v1 : (t == 2 ? v2 : v3));
        a = fmaxf(a, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
    }
    // e4m3 smallest subnormal is 2^-9; clamping there cannot clip, because
    // amax < 6 * 2^-9 is exactly the case that produced the underflow.
    const float sfv = fmaxf(a * (1.0f / 6.0f), 1.9531250e-3f);
    const __nv_fp8_e4m3 se = __nv_fp8_e4m3(sfv);
    const float sd = (float)se;
    const float inv = sd > 0.0f ? 1.0f / sd : 0.0f;
    sf[lay(cute::make_coord((int)r, (int)(b * SFVEC), 0))] = se.__x;
    unsigned char* out = q + (size_t)r * (k >> 1) + (size_t)b * (SFVEC >> 1);
    #pragma unroll
    for (int t = 0; t < 4; ++t) {
        const float4 v = t == 0 ? v0 : (t == 1 ? v1 : (t == 2 ? v2 : v3));
        const unsigned int lo0 = e2m1_enc(v.x * inv), hi0 = e2m1_enc(v.y * inv);
        const unsigned int lo1 = e2m1_enc(v.z * inv), hi1 = e2m1_enc(v.w * inv);
        out[2 * t]     = (unsigned char)(lo0 | (hi0 << 4));
        out[2 * t + 1] = (unsigned char)(lo1 | (hi1 << 4));
    }
}

// ---- the GEMM -----------------------------------------------------------
// SWAP = FlashInfer's decode rule, read out of its source
// (flashinfer/gemm/kernels/utils.py:_score_sm100_mm_fp4_tactic):
//     rule_swap = (n % 8 == 0) and (1 <= m <= 32) and n > m
// With the operands swapped the WEIGHT is the M operand -- tiled 128 rows at
// a time, so the grid is out_dim/128 CTAs -- and the <=32 activation rows are
// the whole N tile. Unswapped, m=32 is PADDED to the 128-row MMA tile and 75%
// of the M dimension is padding. D comes out ColumnMajor at (out_dim, batch),
// which is byte-for-byte the same [batch][out_dim] row-major buffer the
// caller already owns, so nothing downstream changes.
//
// Both SF planes are untouched by the swap: this file writes them through
// tile_atom_to_shape_SFA and the shipped (unswapped) kernel reads the weight
// plane as SFB, so the two atom layouts are already known-equal as functions
// of (mn, k) -- that equality is the invariant a test has to gate.
template <int TM, int TN, int TK, bool SWAP = false>
struct Cfg {
    using MmaTileShape = Shape<Int<TM>, Int<TN>, Int<TK>>;
    using ClusterShape = Shape<int, int, _1>;
    using EpiTile = std::conditional_t<TM == 128 && TN == 256 && TK == 256,
                                       Shape<_128, _64>,
                                       cutlass::epilogue::collective::EpilogueTileAuto>;
    using LayoutC = std::conditional_t<SWAP, cutlass::layout::ColumnMajor,
                                             cutlass::layout::RowMajor>;
    using Epi = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, MmaTileShape, ClusterShape,
        EpiTile, ElementAcc, float, void, LayoutC, 8,
        ElementD, LayoutC, 8,
        cutlass::epilogue::TmaWarpSpecialized1Sm,
        cutlass::epilogue::fusion::LinearCombination<ElementD, float, void, float>>::CollectiveOp;
    using Main = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm100, cutlass::arch::OpClassBlockScaledTensorOp,
        cute::tuple<ElementA, SFType>, cutlass::layout::RowMajor, 32,
        cute::tuple<ElementB, SFType>, cutlass::layout::ColumnMajor, 32,
        ElementAcc, MmaTileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename Epi::SharedStorage))>,
        cutlass::gemm::KernelTmaWarpSpecialized1SmNvf4Sm100>::CollectiveOp;
    using Kernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int, int, int, int>, Main, Epi, cutlass::gemm::PersistentScheduler>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;

    // -1 = shape rejected (or wants workspace), try the next arm
    static int run(const void* wq, const void* wsf, const void* aq, const void* asf,
                   float alpha, void* d, int m, int n, int k, cudaStream_t st) {
        Gemm gemm;
        typename Gemm::Arguments args;
        args.mode = cutlass::gemm::GemmUniversalMode::kGemm;
        // swapped: (M, N) = (out_dim, batch), weight is A. The C stride is the
        // ColumnMajor leading dim = M = out_dim, which lands element (row j of
        // the batch, output i) at j*out_dim + i -- the same [batch][out_dim]
        // row-major bytes the unswapped arm writes.
        args.problem_shape = cute::make_shape(SWAP ? n : m, SWAP ? m : n, k, 1);
        args.mainloop.ptr_A = (ElementA const*)(SWAP ? wq : aq);
        args.mainloop.ptr_B = (ElementB const*)(SWAP ? aq : wq);
        args.mainloop.ptr_SFA = (SFType const*)(SWAP ? wsf : asf);
        args.mainloop.ptr_SFB = (SFType const*)(SWAP ? asf : wsf);
        args.mainloop.dA = cute::make_int_tuple_from<typename Kernel::StrideA>(k, 0);
        args.mainloop.dB = cute::make_int_tuple_from<typename Kernel::StrideB>(k, 0);
        args.mainloop.layout_SFA = BlkCfg::tile_atom_to_shape_SFA(args.problem_shape);
        args.mainloop.layout_SFB = BlkCfg::tile_atom_to_shape_SFB(args.problem_shape);
        args.epilogue.ptr_C = nullptr;
        args.epilogue.ptr_D = (ElementD*)d;
        args.epilogue.dC = cute::make_int_tuple_from<typename Kernel::StrideC>(n, 0);
        args.epilogue.dD = args.epilogue.dC;
        args.epilogue.thread.alpha = alpha;
        args.epilogue.thread.beta = 0.0f;
        args.hw_info.cluster_shape = dim3(1, 1, 1);
        args.hw_info.cluster_shape_fallback = dim3(1, 1, 1);
        if (gemm.can_implement(args) != cutlass::Status::kSuccess) return -1;
        // a workspace demand would need a caller-owned, address-stable buffer
        // (the decode graphs bake addresses); refuse instead of allocating
        if (Gemm::get_workspace_size(args) != 0) return -1;
        if (gemm.initialize(args, nullptr, st) != cutlass::Status::kSuccess)
            return (int)cudaErrorInvalidValue;
        if (gemm.run(st) != cutlass::Status::kSuccess) return (int)cudaErrorUnknown;
        return (int)cudaGetLastError();
    }
};

// Elected at the live FFN decode shapes, m=32, L2-honest 4-clone rotation:
//   gate|up n=34816 k=5120 : 128x256x128 26.32 us  (128x256x256 26.85,
//                            128x128x256 32.47, 2SM 256x256x256 34.20)
//   down    n=5120 k=17408 : 128x64x256  21.37 us  (128x192x256 24.74,
//                            128x256x256 54.28 - the wide tiles starve a
//                            40-CTA grid)
// Wide-N shapes take the wide tile, thin-N the narrow one; the crossover is
// the only thing the two rows disagree about.
using WideN = Cfg<128, 256, 128>;
using WideN2 = Cfg<128, 256, 256>;
using ThinN = Cfg<128, 64, 256>;
using Mid = Cfg<128, 128, 256>;

// Re-election with the operand swap, m=32, same L2-honest rotation:
//   gate|up n=34816 k=5120 : SWAP 128x64x256  22.57 us
//                            SWAP 128x128x256 30.87, plain 128x256x128 26.56
//   down    n=5120  k=17408: SWAP 128x64x256  22.55
//                            SWAP 128x64x256 c21 21.90, plain 128x64x256 22.55
// One tile now wins both shapes, so the wide/thin crossover is retired. The
// unswapped ladder stays reachable behind PADDOCK_NV4CUT_NOSWAP so the whole
// election tree is still A/B-able against a shipped binary.
using SwapN = Cfg<128, 64, 256, true>;
using SwapN2 = Cfg<128, 128, 256, true>;

// Env truthy read; this TU is standalone and cannot include the pack blob's
// pd_env_on, so it carries the same contract locally (see cutgemm.cu).
#if defined(_WIN32)
extern "C" __declspec(dllimport) unsigned long __stdcall GetEnvironmentVariableA(
    const char* name, char* buffer, unsigned long size);
static bool nv4_env_on(const char* name) {
    char b[128];
    const unsigned long n = GetEnvironmentVariableA(name, b, 128ul);
    if (n == 0ul || n >= 128ul) return false;
    return b[0] != '\0' && !(b[0] == '0' && b[1] == '\0');
}
#else
static bool nv4_env_on(const char* name) {
    const char* v = std::getenv(name);
    return v != nullptr && v[0] != '\0' && !(v[0] == '0' && v[1] == '\0');
}
#endif

static bool noswap_on() {
    static const bool v = nv4_env_on("PADDOCK_NV4CUT_NOSWAP");
    return v;
}

}  // namespace pdnv4
#endif  // PD_CUTGEMM

// slot 462: SF-plane byte size for the CUTLASS blocked layout at (mn, k).
extern "C" int pd_nv4cut_sf_bytes(unsigned int mn, unsigned int k,
                                  unsigned long long* out) {
#ifdef PD_CUTGEMM
    if (!out || (k & (pdnv4::SFVEC - 1)) != 0) return (int)cudaErrorInvalidValue;
    *out = (unsigned long long)cute::cosize(pdnv4::sf_layout((int)mn, (int)k));
    return 0;
#else
    (void)mn; (void)k; (void)out;
    return (int)cudaErrorNotSupported;
#endif
}

// slot 463: scatter a row-major [mn][k/16] e4m3 scale vector into the
// blocked SF layout. `dst` must be pd_nv4cut_sf_bytes() big and ZEROED.
extern "C" int pd_nv4cut_sf_repack(const void* src, void* dst, unsigned int mn,
                                   unsigned int k, void* stream) {
#ifdef PD_CUTGEMM
    if ((k & (pdnv4::SFVEC - 1)) != 0) return (int)cudaErrorInvalidValue;
    const unsigned int nblk = k / pdnv4::SFVEC;
    const size_t n = (size_t)mn * nblk;
    if (n == 0) return 0;
    const unsigned int thr = 256u;
    const unsigned int blocks = (unsigned int)((n + thr - 1) / thr);
    pdnv4::pd_nv4cut_sf_repack_kernel<<<blocks, thr, 0, (cudaStream_t)stream>>>(
        (const unsigned char*)src, (unsigned char*)dst,
        pdnv4::sf_layout((int)mn, (int)k), nblk, mn);
    return (int)cudaGetLastError();
#else
    (void)src; (void)dst; (void)mn; (void)k; (void)stream;
    return (int)cudaErrorNotSupported;
#endif
}

// slot 464: f32 [m][k] -> e2m1 nibbles + blocked SFA, per-16 dynamic scale.
extern "C" int pd_nv4cut_quant_a(const void* x, void* aq, void* asf,
                                 unsigned int k, unsigned int m, void* stream) {
#ifdef PD_CUTGEMM
    if ((k & (pdnv4::SFVEC - 1)) != 0) return (int)cudaErrorInvalidValue;
    const unsigned int nblk = k / pdnv4::SFVEC;
    const size_t n = (size_t)m * nblk;
    if (n == 0) return 0;
    const unsigned int thr = 256u;
    const unsigned int blocks = (unsigned int)((n + thr - 1) / thr);
    pdnv4::pd_nv4cut_quant_a_kernel<<<blocks, thr, 0, (cudaStream_t)stream>>>(
        (const float*)x, (unsigned char*)aq, (unsigned char*)asf,
        pdnv4::sf_layout((int)m, (int)k), k, m);
    return (int)cudaGetLastError();
#else
    (void)x; (void)aq; (void)asf; (void)k; (void)m; (void)stream;
    return (int)cudaErrorNotSupported;
#endif
}

// slot 465: D[m][n] bf16 = alpha * (A_nvfp4 x B_nvfp4^T), block-scaled.
extern "C" int pd_nv4cut_gemm(const void* wq, const void* wsf, const void* aq,
                              const void* asf, float alpha, void* d,
                              unsigned int in_dim, unsigned int out_dim,
                              unsigned int batch, void* stream) {
#ifdef PD_CUTGEMM
    using namespace pdnv4;
    cudaStream_t st = (cudaStream_t)stream;
    const int m = (int)batch, n = (int)out_dim, k = (int)in_dim;
    if (m == 0 || n == 0 || k == 0) return 0;
    int r;
    // The swap only pays while the batch fits inside one N tile; past that it
    // is the unswapped shape that has the tile-quantization edge, which is the
    // same crossover FlashInfer spells as `1 <= m <= 32`.
    if (!noswap_on() && m <= 64) {
        r = SwapN::run(wq, wsf, aq, asf, alpha, d, m, n, k, st);
        if (r >= 0) return r;
        r = SwapN2::run(wq, wsf, aq, asf, alpha, d, m, n, k, st);
        if (r >= 0) return r;
    }
    if (n >= 16384) {
        r = WideN::run(wq, wsf, aq, asf, alpha, d, m, n, k, st);
        if (r >= 0) return r;
        r = WideN2::run(wq, wsf, aq, asf, alpha, d, m, n, k, st);
        if (r >= 0) return r;
    } else {
        r = ThinN::run(wq, wsf, aq, asf, alpha, d, m, n, k, st);
        if (r >= 0) return r;
    }
    r = Mid::run(wq, wsf, aq, asf, alpha, d, m, n, k, st);
    return r < 0 ? (int)cudaErrorInvalidValue : r;
#else
    (void)wq; (void)wsf; (void)aq; (void)asf; (void)alpha; (void)d;
    (void)in_dim; (void)out_dim; (void)batch; (void)stream;
    return (int)cudaErrorNotSupported;
#endif
}
