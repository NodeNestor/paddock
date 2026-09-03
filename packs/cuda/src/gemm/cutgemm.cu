// cutgemm: vendored CUTLASS sm100 fp8 GEMM.
//
// A CUTLASS fp8 GEMM behind a pack export. Measured at m=32: gu
// 40.2us/5.75TB/s, o 8.2us, qkv 13.3us, i.e. a 4.18 ms/tick decode band
// against tc5's 5.57. The seam:
//   - A = xq  [batch][in_dim] e4m3 row-major (quantize_e4m3_row's layout)
//   - B = flat k-major e4m3 weights [out_dim][in_dim] (pd_f8t_detile builds
//     them once at load from the SW128 tile image; rowwise scale vector is
//     the same wrs the tile plane carries)
//   - D = y [batch][out_dim] f32 row-major - exactly our epilogue layout
//   - EVT epilogue: D = acc * xrs[m] * wrs[n] in one pass
// two tile arms: the 64x64 decode arm (m < wide floor) and a
// 128x128 cluster-(2,1,1) wide arm for the prefill band - the 64-tile
// config spends 173ms of a 372ms burst pass inside wide GEMM, so wide tiles
// are the m>=1024 half of that election.
// The wide arm falls back to the narrow arm at runtime if can_implement
// rejects a shape. PADDOCK_F8CUT_WIDEB overrides the floor (0 disables).
// Compiled only for sm_100a as its own TU (CUTLASS stays out of the 8-arch
// fatbin); build.sh links it into pd-cuda-sm120.so. Guarded by PD_CUTGEMM.
//
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cstdlib>
#include <cstdio>

#ifdef PD_CUTGEMM
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cute/tensor.hpp"

using namespace cute;

namespace pdcut {

using ElementAB = cutlass::float_e4m3_t;
using ElementD = float;
using ElementAcc = float;

template <class Tile, class Cluster, class EltD = float, int AlignD = 4>
struct Cfg {
    // EVT scale fusion: D = acc * xrs[m] * wrs[n] folded into the epilogue
    using ColXrs = cutlass::epilogue::fusion::Sm90ColBroadcast<
        0, Tile, float, float, Stride<_1, _0, int64_t>>;
    using RowWrs = cutlass::epilogue::fusion::Sm90RowBroadcast<
        0, Tile, float, float, Stride<_0, _1, int64_t>>;
    using Mul = cutlass::epilogue::fusion::Sm90Compute<
        cutlass::multiplies, float, float,
        cutlass::FloatRoundStyle::round_to_nearest>;
    // the OUTER node's output element must be the D store type - a float
    // output into a bf16 store is the sm100 epilogue's operator= mismatch
    using MulOut = cutlass::epilogue::fusion::Sm90Compute<
        cutlass::multiplies, EltD, float,
        cutlass::FloatRoundStyle::round_to_nearest>;
    using AccX = cutlass::epilogue::fusion::Sm90EVT<
        Mul, ColXrs, cutlass::epilogue::fusion::Sm90AccFetch>;
    using Fusion = cutlass::epilogue::fusion::Sm90EVT<MulOut, RowWrs, AccX>;

    using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
        Tile, Cluster,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAcc, ElementAcc,
        void, cutlass::layout::RowMajor, 16,
        EltD, cutlass::layout::RowMajor, AlignD,
        cutlass::epilogue::collective::EpilogueScheduleAuto,
        Fusion>::CollectiveOp;

    using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
        ElementAB, cutlass::layout::RowMajor, 16,
        ElementAB, cutlass::layout::ColumnMajor, 16,
        ElementAcc,
        Tile, Cluster,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename Epilogue::SharedStorage))>,
        cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

    using Kernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int, int, int, int>, Mainloop, Epilogue>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;

    // returns cudaSuccess-class int, or -1 for "shape rejected, try another
    // arm" (workspace demand counts as rejection - no caching here)
    static int run(const void* w_flat, const void* wrs, const void* xq,
                   const void* xrs, void* y, int m, int n, int k,
                   cudaStream_t st) {
        Gemm gemm;
        typename Gemm::Arguments args{
            cutlass::gemm::GemmUniversalMode::kGemm,
            {m, n, k, 1},
            {(const ElementAB*)xq,
             cutlass::make_cute_packed_stride(typename Kernel::StrideA{}, {m, k, 1}),
             (const ElementAB*)w_flat,
             cutlass::make_cute_packed_stride(typename Kernel::StrideB{}, {n, k, 1})},
            {{}, nullptr, typename Kernel::StrideC{},
             (EltD*)y,
             cutlass::make_cute_packed_stride(typename Kernel::StrideD{}, {m, n, 1})}};
        // EVT arg tree: {rowb(wrs), {colb(xrs), accfetch{}, mul{}}, mul{}}
        args.epilogue.thread = {
            {(const float*)wrs, float(0), {_0{}, _1{}, int64_t(0)}},
            {{(const float*)xrs, float(0), {_1{}, _0{}, int64_t(0)}}, {}, {}},
            {}};
        if (gemm.can_implement(args) != cutlass::Status::kSuccess) return -1;
        if (Gemm::get_workspace_size(args) != 0) return -1;
        if (gemm.initialize(args, nullptr, st) != cutlass::Status::kSuccess)
            return (int)cudaErrorInvalidValue;
        if (gemm.run(st) != cutlass::Status::kSuccess)
            return (int)cudaErrorUnknown;
        return (int)cudaGetLastError();
    }
};

using Narrow = Cfg<Shape<_64, _64, _128>, Shape<_1, _1, _1>>;

// the narrow arm is only right from m=32 up. `Narrow` was elected
// against a m=32 decode tick; below that the 64-row M tile carries 8-16 live
// rows and the CTA's TMA pipeline runs dry long before the B plane is drained.
// The die then wants FEWER, DEEPER tiles, not more shallow ones: a 256-wide N
// tile with a 2-CTA M cluster reads the same n*k bytes (the cluster
// multicasts B) but keeps ~5x the bytes in flight per CTA.
// L2-honest 4-clone rotation, m=16:
//   gdn_qkvz 16384x5120   Narrow 24.30 -> ThinWide 18.54   (-23.7%)
//   attn_qkv 14336x5120   Narrow 23.58 -> ThinWide 16.48   (-30.1%)
//   out_proj  5120x6144   Narrow 12.39 -> ThinPair 10.36   (-16.4%)
//   ffn_down  5120x17408  Narrow 20.63 -> ThinPair 20.73   (Narrow keeps it)
// At m=32 Narrow is optimal or within 0.2% on all four, which is why this is
// gated at m <= 16 and not simply swapped in.
using ThinWide = Cfg<Shape<_128, _256, _128>, Shape<_2, _1, _1>>;
using ThinPair = Cfg<Shape<_64, _64, _128>, Shape<_1, _2, _1>>;
using Wide = Cfg<Shape<_128, _128, _128>, Shape<_2, _1, _1>>;
// bf16-D wide arm - halves the D write and every downstream
// glue read in the burst pass
using WideB16 = Cfg<Shape<_128, _128, _128>, Shape<_2, _1, _1>,
                    cutlass::bfloat16_t, 8>;
// cluster-(1,4,1) shape - 4 CTAs multicast B, so the win case is
// B-bound shapes (gu n=43008).
// A/B via PADDOCK_F8CUT_C141=1 (replaces the wide arm when set).
using Wide141 = Cfg<Shape<_128, _128, _128>, Shape<_1, _4, _1>>;

// N-wide tiles for the huge-N gu shape (muse n=39936) - more A
// reuse per B byte on the pass's one B-bound GEMM. Elected only at
// n >= 16384 behind PADDOCK_F8CUT_N256; falls back on can_implement
// rejection like every arm here.
using WideN256 = Cfg<Shape<_128, _256, _128>, Shape<_2, _1, _1>>;
using WideB16N256 = Cfg<Shape<_128, _256, _128>, Shape<_2, _1, _1>,
                        cutlass::bfloat16_t, 8>;

// M-tall tiles for the m~6000 single-pass wave - the 2-CTA M256
// collective halves the B re-reads per A byte at wave-class m. Elected at
// m >= 4096 behind PADDOCK_F8CUT_M256 (tried before the N256/wide arms).
using WideM256 = Cfg<Shape<_256, _128, _128>, Shape<_2, _1, _1>>;
using WideB16M256 = Cfg<Shape<_256, _128, _128>, Shape<_2, _1, _1>,
                        cutlass::bfloat16_t, 8>;

// the BOTH-wide 256x256 tile - never in the earlier election space, which
// explored M-tall 256x128 and N-wide 128x256 separately, never together.
// Isolated on the muse wave (EVT bf16-D, m=5984) it beats M256 on every shape:
// per-layer band 1560->1349 us (-13.5%), down -20% (the deep-K shape M256 does
// worst on), gu -11%, at ~4.2 PF / ~93% of B200 fp8 SOL. Fits with bf16 D +
// StageCountAutoCarveout at zero workspace (serve's rejection gate stays clean).
// Tried first at wave-class m behind PADDOCK_F8CUT_BIG; falls back to the
// M256/N256/Wide ladder on can_implement rejection like every arm here.
using WideBig = Cfg<Shape<_256, _256, _128>, Shape<_2, _1, _1>>;
using WideB16Big = Cfg<Shape<_256, _256, _128>, Shape<_2, _1, _1>,
                       cutlass::bfloat16_t, 8>;

// Env truthy read, duplicated from abi.cuh's pd_env_on (this TU is
// standalone and cannot textually include the pack blob). Same contract:
// UCRT-safe on Windows (Rust set_var writes an environment this TU's getenv
// never sees), and "0" means off because the engine fills these three as
// muse tuned defaults - "the env always wins" needs a spelled opt-out.
#if defined(_WIN32)
extern "C" __declspec(dllimport) unsigned long __stdcall GetEnvironmentVariableA(
    const char* name, char* buffer, unsigned long size);
static bool cut_env_on(const char* name) {
    char b[128];
    const unsigned long n = GetEnvironmentVariableA(name, b, 128ul);
    if (n == 0ul || n >= 128ul) return false;
    return b[0] != '\0' && !(b[0] == '0' && b[1] == '\0');
}
#else
static bool cut_env_on(const char* name) {
    const char* v = std::getenv(name);
    return v != nullptr && v[0] != '\0' && !(v[0] == '0' && v[1] == '\0');
}
#endif

static bool c141_on() {
    static const bool v = cut_env_on("PADDOCK_F8CUT_C141");
    return v;
}

static bool n256_on() {
    static const bool v = cut_env_on("PADDOCK_F8CUT_N256");
    return v;
}

static bool m256_on() {
    static const bool v = cut_env_on("PADDOCK_F8CUT_M256");
    return v;
}

static bool big_on() {
    static const bool v = cut_env_on("PADDOCK_F8CUT_BIG");
    return v;
}

static int wide_floor() {
    static int v = [] {
        const char* e = std::getenv("PADDOCK_F8CUT_WIDEB");
        if (!e) return 1024;
        return std::atoi(e);
    }();
    return v;
}

// the ThinWide/ThinPair ceiling. PADDOCK_F8CUT_THIN overrides it;
// 0 disables the thin ladder and restores the earlier Narrow-for-everything
// election (the A/B control).
static int thin_ceiling() {
    static int v = [] {
        const char* e = std::getenv("PADDOCK_F8CUT_THIN");
        if (!e) return 16;
        return std::atoi(e);
    }();
    return v;
}

// ---- gluq: fused geglu + per-fragment pow2 e4m3 quantize epilogue ----
// The gu GEMM emits {act_byte, scale_byte} e4m3 pairs over a gate/up-
// INTERLEAVED weight plane (pd_f8t_detile_gui below); a fixup kernel then
// compacts to [batch][n_ff] at the row-max exponent. The pd_rowq exponent
// formula is monotone in amax, so max-over-fragment-exponents == the row
// exponent the shipped quantize kernels produce (rscale verified bit-equal
// 128/128 and 56/56 on both tile arms). The fused GEMM lands at or below
// the stock arm - the byte-width D store pays for the epilogue math - and
// the chain runs -5.8% at m=128 / -10.6% at m=56 against the shipped pair.
// Scale-byte range: 2^e is exact in e4m3 for e in [-9, 8]; the clamp at 8
// silently saturates rows with amax > 448*2^8 ~ 114k - 4 orders above
// post-norm glu2 activations; the PPL/acceptance gates carry formal safety.
__device__ __forceinline__ float pd_gluq_gelu(float g) {
    // pd_glu_act GELU constants, tanh via the sm_75+ hw approximation.
    // PPL attribution: on the (bit-exact) stream lane ctl=9.89599,
    // arm/approx=9.97727, arm/tanhf=10.05415 - the more exact tanh moved
    // FURTHER, i.e. the deltas are ulp-class chaos from the lane newly
    // riding cutlass acc-order + the finer fragment rounding (rscale
    // exact, diffs <= 1 subnormal ulp), not a tanh bias. approx is faster
    // AND closer; the ship gate for this numerics class is acceptance
    // testing, not ulp equality.
    const float t = 0.79788456080286535587989211986876f * g
                    * (1.0f + 0.044715f * g * g);
    float r;
    asm("tanh.approx.f32 %0, %1;" : "=f"(r) : "f"(t));
    return 0.5f * g * (1.0f + r);
}

// silu twin (qwen35/muse swiglu): the shipped pd_swiglu_fused expression
// verbatim (plain expf) - same numerics class as the classic chain.
__device__ __forceinline__ float pd_gluq_silu(float g) {
    return g / (1.0f + expf(-g));
}

// Shared pair body: the exponent construction is load-bearing (pd_rowq
// formula; monotone in amax so max-over-fragments == the shipped row
// exponent) - keep it in one place across activation twins.
template <class ActF, int N>
CUTLASS_DEVICE cutlass::Array<float, N>
pd_gluq_pair_body(cutlass::Array<float, N> const& v) {
    static_assert(N % 2 == 0, "pairing needs an even fragment");
    float act[N / 2];
    float amax = 0.0f;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
        act[i] = ActF::act(v[2 * i]) * v[2 * i + 1];
        amax = fmaxf(amax, fabsf(act[i]));
    }
    int e = 0;                     // pd_rowq_exp construction (448 bound)
    if (amax > 0.0f) {
        int ex;
        const float fr = frexpf(amax, &ex);
        e = ex - 9 + (fr > 0.875f ? 1 : 0);
    }
    e = e < -9 ? -9 : (e > 8 ? 8 : e);
    const float inv = ldexpf(1.0f, -e), s = ldexpf(1.0f, e);
    cutlass::Array<float, N> out;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
        out[2 * i] = act[i] * inv;
        out[2 * i + 1] = s;        // 2^e, exact in e4m3
    }
    return out;
}
struct GeluActF { static CUTLASS_DEVICE float act(float g) { return pd_gluq_gelu(g); } };
struct SiluActF { static CUTLASS_DEVICE float act(float g) { return pd_gluq_silu(g); } };

template <class T> struct PairQGelu {
    CUTLASS_HOST_DEVICE T operator()(T const& v) const { return v; }
};
template <int N> struct PairQGelu<cutlass::Array<float, N>> {
    CUTLASS_DEVICE cutlass::Array<float, N>
    operator()(cutlass::Array<float, N> const& v) const {
        return pd_gluq_pair_body<GeluActF, N>(v);
    }
};
template <class T> struct PairQSilu {
    CUTLASS_HOST_DEVICE T operator()(T const& v) const { return v; }
};
template <int N> struct PairQSilu<cutlass::Array<float, N>> {
    CUTLASS_DEVICE cutlass::Array<float, N>
    operator()(cutlass::Array<float, N> const& v) const {
        return pd_gluq_pair_body<SiluActF, N>(v);
    }
};

template <class Tile, class Cluster, template <class> class PairFn>
struct CfgQ {
    using ColXrs = cutlass::epilogue::fusion::Sm90ColBroadcast<
        0, Tile, float, float, Stride<_1, _0, int64_t>>;
    using RowWrs = cutlass::epilogue::fusion::Sm90RowBroadcast<
        0, Tile, float, float, Stride<_0, _1, int64_t>>;
    using MulF = cutlass::epilogue::fusion::Sm90Compute<
        cutlass::multiplies, float, float,
        cutlass::FloatRoundStyle::round_to_nearest>;
    using AccX = cutlass::epilogue::fusion::Sm90EVT<
        MulF, ColXrs, cutlass::epilogue::fusion::Sm90AccFetch>;
    using Inner = cutlass::epilogue::fusion::Sm90EVT<MulF, RowWrs, AccX>;
    using Fusion = cutlass::epilogue::fusion::Sm90EVT<
        cutlass::epilogue::fusion::Sm90Compute<
            PairFn, cutlass::float_e4m3_t, float,
            cutlass::FloatRoundStyle::round_to_nearest>,
        Inner>;

    using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
        Tile, Cluster,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAcc, ElementAcc,
        void, cutlass::layout::RowMajor, 16,
        cutlass::float_e4m3_t, cutlass::layout::RowMajor, 16,
        cutlass::epilogue::collective::EpilogueScheduleAuto,
        Fusion>::CollectiveOp;

    using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
        ElementAB, cutlass::layout::RowMajor, 16,
        ElementAB, cutlass::layout::ColumnMajor, 16,
        ElementAcc,
        Tile, Cluster,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename Epilogue::SharedStorage))>,
        cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

    using Kernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int, int, int, int>, Mainloop, Epilogue>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;

    static int run(const void* w_flat, const void* wrs, const void* xq,
                   const void* xrs, void* q2, int m, int n, int k,
                   cudaStream_t st) {
        Gemm gemm;
        typename Gemm::Arguments args{
            cutlass::gemm::GemmUniversalMode::kGemm,
            {m, n, k, 1},
            {(const ElementAB*)xq,
             cutlass::make_cute_packed_stride(typename Kernel::StrideA{}, {m, k, 1}),
             (const ElementAB*)w_flat,
             cutlass::make_cute_packed_stride(typename Kernel::StrideB{}, {n, k, 1})},
            {{}, nullptr, typename Kernel::StrideC{},
             (cutlass::float_e4m3_t*)q2,
             cutlass::make_cute_packed_stride(typename Kernel::StrideD{}, {m, n, 1})}};
        // {inner = {rowb(wrs), {colb(xrs), acc{}, mul{}}, mul{}}, pairq{}}
        args.epilogue.thread = {
            {{(const float*)wrs, float(0), {_0{}, _1{}, int64_t(0)}},
             {{(const float*)xrs, float(0), {_1{}, _0{}, int64_t(0)}}, {}, {}},
             {}},
            {}};
        if (gemm.can_implement(args) != cutlass::Status::kSuccess) return -1;
        if (Gemm::get_workspace_size(args) != 0) return -1;
        if (gemm.initialize(args, nullptr, st) != cutlass::Status::kSuccess)
            return (int)cudaErrorInvalidValue;
        if (gemm.run(st) != cutlass::Status::kSuccess)
            return (int)cudaErrorUnknown;
        return (int)cudaGetLastError();
    }
};
using NarrowQ = CfgQ<Shape<_64, _64, _128>, Shape<_1, _1, _1>, PairQGelu>;
using WideQ = CfgQ<Shape<_128, _128, _128>, Shape<_2, _1, _1>, PairQGelu>;
using NarrowQS = CfgQ<Shape<_64, _64, _128>, Shape<_1, _1, _1>, PairQSilu>;
using WideQS = CfgQ<Shape<_128, _128, _128>, Shape<_2, _1, _1>, PairQSilu>;
// P72 config ladder (silu/qwen only): the decode-m gluq at (1,544) streams
// 178MB at 4.45 TB/s vs the ~7 floor - 3.7 waves of 64-wide n-tiles with a
// quantization tail. Cluster-2 multicasts X; the 128-wide n-tile halves the
// CTA count for longer K-serial streams. Reassociation class per config
// (tile schedule changes the K-sum order) - elected by receipt, gated on
// distinct/coherence + the cell, never token-match. PADDOCK_GLUQ_CFG=0..3.
using NarrowQS_C2 = CfgQ<Shape<_64, _64, _128>, Shape<_2, _1, _1>, PairQSilu>;
using MidQS = CfgQ<Shape<_64, _128, _128>, Shape<_1, _1, _1>, PairQSilu>;
using MidQS_C2 = CfgQ<Shape<_64, _128, _128>, Shape<_2, _1, _1>, PairQSilu>;

// scale byte -> exponent (positive pow2 e4m3 only; subnormals 1/2/4)
__device__ __forceinline__ int pd_gluq_scale_exp(unsigned int b) {
    return b >= 8u ? (int)(b >> 3) - 7 : (b == 1u ? -9 : (b == 2u ? -8 : -7));
}
__device__ __forceinline__ float pd_gluq_e4m3_dec(unsigned char b) {
    const int s = (b >> 7) & 1, e = (b >> 3) & 0xF, mf = b & 7;
    float v;
    if (e == 0) v = ldexpf((float)mf, -9);
    else v = ldexpf(1.0f + (float)mf * 0.125f, e - 7);
    return s ? -v : v;
}
// q2 [batch][2*n_ff] {act, scale} byte pairs -> q [batch][n_ff] + rscale.
// One CTA per row; the row rides registers between the passes; the shift
// itself is a branchless 18x256 smem table (dec/enc roundtrip at k=0 is
// exact, so tab[0][b] == b). n_ff <= 32768 (MAXI * 1024 * 4).
__global__ void __launch_bounds__(1024) pd_gluq_fixup_kernel(
    const uchar2* __restrict__ q2, unsigned char* __restrict__ q,
    float* __restrict__ rscale, uint32_t n_ff) {
    constexpr uint32_t MAXI = 8;
    const uint32_t row = blockIdx.x;
    const uint32_t tid = threadIdx.x, nth = blockDim.x;
    const uint2* src = (const uint2*)(q2 + (size_t)row * n_ff);
    const uint32_t n4 = n_ff >> 2;
    __shared__ unsigned char s_tab[18 * 256];
    for (uint32_t t = tid; t < 18u * 256u; t += nth) {
        const int k = (int)(t >> 8);
        s_tab[t] = __nv_fp8_e4m3(
            pd_gluq_e4m3_dec((unsigned char)(t & 0xFFu)) * ldexpf(1.0f, -k)).__x;
    }
    uint2 vv[MAXI];
    unsigned int smax = 0;
    #pragma unroll
    for (uint32_t j = 0; j < MAXI; ++j) {
        const uint32_t i = tid + j * nth;
        if (i < n4) {
            const uint2 v = src[i];
            vv[j] = v;
            smax = max(smax, (v.x >> 8) & 0xFFu);
            smax = max(smax, v.x >> 24);
            smax = max(smax, (v.y >> 8) & 0xFFu);
            smax = max(smax, v.y >> 24);
        }
    }
    __shared__ unsigned int wmax[32];
    __shared__ int s_er;
    for (uint32_t sh = 16; sh > 0; sh >>= 1)
        smax = max(smax, __shfl_xor_sync(0xffffffffu, smax, sh));
    if ((tid & 31u) == 0) wmax[tid >> 5] = smax;
    __syncthreads();
    if (tid == 0) {
        unsigned int m = 0;
        for (uint32_t w = 0; w < ((nth + 31u) >> 5); ++w) m = max(m, wmax[w]);
        s_er = pd_gluq_scale_exp(m);
        rscale[row] = ldexpf(1.0f, s_er);
    }
    __syncthreads();
    const int er = s_er;
    unsigned char* qr = q + (size_t)row * n_ff;
    #pragma unroll
    for (uint32_t j = 0; j < MAXI; ++j) {
        const uint32_t i = tid + j * nth;
        if (i < n4) {
            const uint2 v = vv[j];
            unsigned int ob = 0;
            #pragma unroll
            for (int h = 0; h < 4; ++h) {
                const unsigned int w = h < 2 ? v.x : v.y;
                const unsigned int b = (w >> (16 * (h & 1))) & 0xFFu;
                const int k = min(er - pd_gluq_scale_exp((w >> (16 * (h & 1) + 8)) & 0xFFu), 17);
                ob |= (unsigned int)s_tab[(k << 8) | b] << (8 * h);
            }
            *(unsigned int*)(qr + (size_t)i * 4u) = ob;
        }
    }
}

// gate/up-interleaved twin of the detiler: dst flat row 2f = src row f
// (gate), dst 2f+1 = src row out/2 + f (up) - the pairing layout CfgQ needs
__global__ void pd_f8t_detile_gui_kernel(const unsigned char* __restrict__ tiles,
                                         unsigned char* __restrict__ flat,
                                         uint32_t in_dim, uint32_t out_dim) {
    const uint32_t nk = in_dim >> 7;
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)in_dim * out_dim) return;
    const uint32_t r = (uint32_t)(i / in_dim);
    const uint32_t k = (uint32_t)(i % in_dim);
    const uint32_t s = (r & 1u) ? (out_dim >> 1) + (r >> 1) : (r >> 1);
    const uint32_t rt = s >> 7, rr = s & 127u;
    const uint32_t kt = k >> 7, kk = k & 127u;
    const size_t tile = ((size_t)rt * nk + kt) << 14;
    const uint32_t off = rr * 128u + ((((kk >> 4) ^ (rr & 7u)) << 4)) + (kk & 15u);
    flat[i] = tiles[tile + off];
}

// flat k-major [out][in] from the SW128 tile image ((row_tile, kt)-major
// 16KB tiles; within a tile: row r, k-byte d at
// r*128 + (((d>>4) ^ (r&7)) << 4) + (d&15) - the tma_kt canonical layout)
__global__ void pd_f8t_detile_kernel(const unsigned char* __restrict__ tiles,
                                     unsigned char* __restrict__ flat,
                                     uint32_t in_dim, uint32_t out_dim) {
    const uint32_t nk = in_dim >> 7;
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)in_dim * out_dim) return;
    const uint32_t row = (uint32_t)(i / in_dim);
    const uint32_t k = (uint32_t)(i % in_dim);
    const uint32_t rt = row >> 7, rr = row & 127u;
    const uint32_t kt = k >> 7, kk = k & 127u;
    const size_t tile = ((size_t)rt * nk + kt) << 14;
    const uint32_t off = rr * 128u + ((((kk >> 4) ^ (rr & 7u)) << 4)) + (kk & 15u);
    flat[i] = tiles[tile + off];
}

}  // namespace pdcut
#endif  // PD_CUTGEMM

extern "C" int pd_f8cut_gemm(const void* w_flat, const void* wrs,
                             const void* xq, const void* xrs, void* y,
                             unsigned int in_dim, unsigned int out_dim,
                             unsigned int batch, void* stream) {
#ifdef PD_CUTGEMM
    using namespace pdcut;
    cudaStream_t st = (cudaStream_t)stream;
    const int m = (int)batch, n = (int)out_dim, k = (int)in_dim;
    const int wf = wide_floor();
    if (wf > 0 && m >= wf) {
        if (n256_on() && n >= 16384) {
            const int r = WideN256::run(w_flat, wrs, xq, xrs, y, m, n, k, st);
            if (r >= 0) return r;
        }
        if (big_on() && m >= 4096) {
            const int r = WideBig::run(w_flat, wrs, xq, xrs, y, m, n, k, st);
            if (r >= 0) return r;
        }
        if (m256_on() && m >= 4096) {
            const int r = WideM256::run(w_flat, wrs, xq, xrs, y, m, n, k, st);
            if (r >= 0) return r;
        }
        const int r = c141_on()
            ? Wide141::run(w_flat, wrs, xq, xrs, y, m, n, k, st)
            : Wide::run(w_flat, wrs, xq, xrs, y, m, n, k, st);
        if (r >= 0) return r;  // -1 = shape rejected, fall through to narrow
        static bool dbg = std::getenv("PADDOCK_F8CUT_DBG") != nullptr;
        if (dbg) fprintf(stderr, "[cutwide-rej] m=%d n=%d k=%d\n", m, n, k);
    }
    // thin-m ladder (see ThinWide/ThinPair). ThinWide's grid is
    // 2 * ceil(n/256) CTAs, so it needs a wide output to stay off a fractional
    // wave - at n=5120 it is 40 CTAs and loses badly (53.30 us on ffn_down).
    // ThinPair takes the shallow-K shapes; a deep K loop (ffn_down, k=17408)
    // already fills the pipeline and Narrow keeps it.
    const int tc = thin_ceiling();
    if (tc > 0 && m <= tc) {
        if (2 * ((n + 255) / 256) >= 96) {
            const int r = ThinWide::run(w_flat, wrs, xq, xrs, y, m, n, k, st);
            if (r >= 0) return r;
        } else if (k <= 8192) {
            const int r = ThinPair::run(w_flat, wrs, xq, xrs, y, m, n, k, st);
            if (r >= 0) return r;
        }
    }
    const int r = Narrow::run(w_flat, wrs, xq, xrs, y, m, n, k, st);
    return r < 0 ? (int)cudaErrorInvalidValue : r;
#else
    (void)w_flat; (void)wrs; (void)xq; (void)xrs; (void)y;
    (void)in_dim; (void)out_dim; (void)batch; (void)stream;
    return (int)cudaErrorNotSupported;
#endif
}

extern "C" int pd_f8cut_gemm_b16(const void* w_flat, const void* wrs,
                                 const void* xq, const void* xrs, void* y,
                                 unsigned int in_dim, unsigned int out_dim,
                                 unsigned int batch, void* stream) {
#ifdef PD_CUTGEMM
    using namespace pdcut;
    if (big_on() && batch >= 4096) {
        const int r = WideB16Big::run(w_flat, wrs, xq, xrs, y, (int)batch,
                                      (int)out_dim, (int)in_dim,
                                      (cudaStream_t)stream);
        if (r >= 0) return r;
    }
    if (m256_on() && batch >= 4096) {
        const int r = WideB16M256::run(w_flat, wrs, xq, xrs, y, (int)batch,
                                       (int)out_dim, (int)in_dim,
                                       (cudaStream_t)stream);
        if (r >= 0) return r;
    }
    if (n256_on() && out_dim >= 16384) {
        const int r = WideB16N256::run(w_flat, wrs, xq, xrs, y, (int)batch,
                                       (int)out_dim, (int)in_dim,
                                       (cudaStream_t)stream);
        if (r >= 0) return r;
    }
    const int r = WideB16::run(w_flat, wrs, xq, xrs, y, (int)batch,
                               (int)out_dim, (int)in_dim, (cudaStream_t)stream);
    return r < 0 ? (int)cudaErrorNotSupported : r;
#else
    (void)w_flat; (void)wrs; (void)xq; (void)xrs; (void)y;
    (void)in_dim; (void)out_dim; (void)batch; (void)stream;
    return (int)cudaErrorNotSupported;
#endif
}

extern "C" int pd_f8t_detile(const void* tiles, void* flat,
                             unsigned int in_dim, unsigned int out_dim,
                             void* stream) {
#ifdef PD_CUTGEMM
    using namespace pdcut;
    const size_t total = (size_t)in_dim * out_dim;
    pd_f8t_detile_kernel<<<(unsigned)((total + 255) / 256), 256, 0,
                           (cudaStream_t)stream>>>(
        (const unsigned char*)tiles, (unsigned char*)flat, in_dim, out_dim);
    return (int)cudaGetLastError();
#else
    (void)tiles; (void)flat; (void)in_dim; (void)out_dim; (void)stream;
    return (int)cudaErrorNotSupported;
#endif
}

extern "C" int pd_f8cut_gemm_gluq(const void* w_flat_gui, const void* wrs_gui,
                                  const void* xq, const void* xrs,
                                  void* q2_scratch, void* q, void* rscale,
                                  unsigned int in_dim, unsigned int n_ff,
                                  unsigned int batch, unsigned int act,
                                  void* stream) {
#ifdef PD_CUTGEMM
    using namespace pdcut;
    if (act > 1u) return -2;               // 0 = gelu, 1 = silu
    if (n_ff > 32768u || (n_ff & 31u) || batch == 0u) return -2;
    cudaStream_t st = (cudaStream_t)stream;
    const int m = (int)batch, n = (int)(2u * n_ff), k = (int)in_dim;
    // per-activation wide/narrow election, same cross-arm retry both rungs
    auto wide = [&] {
        return act ? WideQS::run(w_flat_gui, wrs_gui, xq, xrs, q2_scratch, m, n, k, st)
                   : WideQ::run(w_flat_gui, wrs_gui, xq, xrs, q2_scratch, m, n, k, st);
    };
    auto narrow = [&] {
        if (act) {
            // P72 ladder receipt (gluq_cfg_bench, m=8/16/32 us/call):
            // Narrow c1 49.7/48.6/42.1, Narrow c2 66/63/51 (cluster-2 loses),
            // Mid 64x128 c1 47.2/45.7/45.2, Mid c2 52/49/48. Mid's ~3us
            // standalone win at m<=16 INVERTED in serve (c16 ABBA: itl 9.84
            // vs 9.75 - Narrow keeps the default): the cold-L2 single-stream
            // bench mispredicts sub-10% election deltas that the serve's
            // partial L2 residency + PDL overlap decide. Serve-ABBA is the
            // only judge at that margin. Configs kept as the A/B surface:
            // PADDOCK_GLUQ_CFG=0..3.
            static int cfg = -1;
            if (cfg < 0) {
                const char* v = getenv("PADDOCK_GLUQ_CFG");
                cfg = v ? atoi(v) : 0;
            }
            const int c = cfg;
            switch (c) {
                case 1: return NarrowQS_C2::run(w_flat_gui, wrs_gui, xq, xrs, q2_scratch, m, n, k, st);
                case 2: return MidQS::run(w_flat_gui, wrs_gui, xq, xrs, q2_scratch, m, n, k, st);
                case 3: return MidQS_C2::run(w_flat_gui, wrs_gui, xq, xrs, q2_scratch, m, n, k, st);
                default: return NarrowQS::run(w_flat_gui, wrs_gui, xq, xrs, q2_scratch, m, n, k, st);
            }
        }
        return NarrowQ::run(w_flat_gui, wrs_gui, xq, xrs, q2_scratch, m, n, k, st);
    };
    int r = m >= 65 ? wide() : narrow();
    if (r == -1) r = m >= 65 ? narrow() : wide();
    if (r == -1) return -2;                // both arms rejected: caller falls back
    if (r != 0) {
        // A LAUNCH refusal is a DECLINE for an elective fast path, not a fatal
        // error. The cross-arm retry above can land on a cluster-2 config whose
        // grid cannot satisfy its cluster shape at this m (one CTA in M against
        // a 2-CTA cluster), and the launch answers 801/NotSupported - which
        // can_implement does not catch.
        //
        // This mattered far past a lost fast path. The rc travelled up as
        // GpuError::Launch, the service finished every in-flight sequence, and
        // the runner mapped the error to finish_reason "stop" with nothing
        // logged. So the qwen3.8 nvfp4 spec lane emitted one token per request
        // at >= 20 live slots and it surfaced as a plausible-looking
        // throughput number instead of a failure. The caller's
        // contract already says -2 means "keep the classic chain"; take it.
        static int said = 0;
        if (!said) {
            said = 1;
            fprintf(stderr, "[gluq] declined: launch rc=%d at m=%d n_ff=%u - "
                            "falling back to the classic chain\n", r, m, n_ff);
        }
        cudaGetLastError();               // clear the sticky error for the next call
        return -2;
    }
    pd_gluq_fixup_kernel<<<(unsigned)m, 1024, 0, st>>>(
        (const uchar2*)q2_scratch, (unsigned char*)q, (float*)rscale, n_ff);
    return (int)cudaGetLastError();
#else
    (void)w_flat_gui; (void)wrs_gui; (void)xq; (void)xrs; (void)q2_scratch;
    (void)q; (void)rscale; (void)in_dim; (void)n_ff; (void)batch; (void)act;
    (void)stream;
    return (int)cudaErrorNotSupported;
#endif
}

extern "C" int pd_f8t_detile_gui(const void* tiles, void* flat,
                                 unsigned int in_dim, unsigned int out_dim,
                                 void* stream) {
#ifdef PD_CUTGEMM
    using namespace pdcut;
    if (out_dim & 1u) return (int)cudaErrorInvalidValue;
    const size_t total = (size_t)in_dim * out_dim;
    pd_f8t_detile_gui_kernel<<<(unsigned)((total + 255) / 256), 256, 0,
                               (cudaStream_t)stream>>>(
        (const unsigned char*)tiles, (unsigned char*)flat, in_dim, out_dim);
    return (int)cudaGetLastError();
#else
    (void)tiles; (void)flat; (void)in_dim; (void)out_dim; (void)stream;
    return (int)cudaErrorNotSupported;
#endif
}
