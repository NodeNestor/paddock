// Paddock CUDA kernel pack - Single translation unit, split into ordered
// include segments under src/ (a pure line-range split; SASS verified
// identical to the monolithic file it came from). This LIST is the ORDER:
// later segments use kernels/helpers from earlier ones, and exports.cuh
// closes with the KernelTableV1 + entry points. Segments are named by op
// domain (they used to be numbered 01_..21_, and the numbers had stopped
// matching the real include order).
// Build: build.sh [arches] compiles only this file.
#include <cstdio>  // host-side route witnesses (puts) only
#include "src/abi.cuh"
#include "src/tma_desc.cuh"      // TMA tensor-map builders + tcgen05 sdesc/idesc constructors; leaf-level, consumed by gemm/f8_lin, attn/decode_tc5, moe/f8, quant/nvf4 (moved out of dense_fp4_w8)
#include "src/elementwise.cuh"
#include "src/attn/decode.cuh"
#include "src/attn/decode_spec.cuh"  // spec-verify attention: GQA walk, FA-lite, spec-FA krs (split from decode.cuh; needs its ldm/mma/cpa helpers)
#include "src/attn/decode_fp8.cuh"   // fp8-native decode lane v8q/v9q/v9q2 + vdim sync + laguna sigmoid router (split from decode.cuh)
#include "src/attn/lagd.cuh"        // hd128 v5-class decode partial (needs decode's ldm/mma/cpa helpers; f32_qkv launches it)
#include "src/gemm/f32_qkv.cuh"
#include "src/qwen4exp.cuh"   // qwen4_exp (Qwen3.8-Flash-Next) new math: grouped (1+w) norm, hyper-connection mix/combine, PLE gate, dilated conv, GDN sigmoid gated-norm + repeat-interleave split; plain CUDA, needs f32_qkv's pd_launch_status
#include "src/gemm/bf16_dense.cuh"
#include "src/gemm/exp_lt.cuh"
#include "src/gemm/lowm.cuh"  // bf16 weight planes (mixed UD files); abi.cuh helpers only
#include "src/attn/fmha16.cuh"      // Q16xKv128 tensor-core decode attention, muse hd128/G16 (needs bf16_dense's pd_bf16m_ldm/mma)
#include "src/deltanet/core.cuh"
#include "src/vision.cuh"
#include "src/gemm/int8_mma.cuh"
#include"src/gemm/f16_dense.cuh"   // in-house f16xf16->f32 wmma GEMM (PADDOCK_INHOUSE_F16 cuBLAS-removal); needs int8_mma's PD_MMA_OK
#include "src/deltanet/split.cuh"   // split walk/o-pass + shared tf32 mma helpers (needs int8_mma's PD_DNC_* defines; stage2_sample's launcher dispatches it)
#include "src/deltanet/walk_rs.cuh"  // register-state bf16-operand walk (needs split.cuh's pd_dnc_cpa16; stage2_sample's launcher dispatches it)
#include "src/deltanet/stage2_sample.cuh"
#include "src/deltanet/spec_rs.cuh"   // canonical spec rejection sampling (sampled drafts + full-q verify)
#include "src/mamba/core.cuh"       // Mamba-2 SSD lane (nemotron_h_moe): conv step w/ bias, seq scan, grouped gated norm, f8r GEMV (needs deltanet core's PD_CONV_K_MAX)
#include"src/mamba/ssd.cuh"        // chunked SSD prefill scan: defines pd_mamba2_ssd_run, elected by core.cuh's seq launchers for long segments
#include "src/gemm/mmq.cuh"
#include "src/moe/mmq.cuh"
#include "src/attn/prefill.cuh"
#include "src/attn/prefill_fa2.cuh"  // FA-2 prefill tile, f16 v4, pf7/pf7rp (split from prefill.cuh); also carries the shared CK macro
#include "src/attn/prefill_pf5.cuh"  // tcgen05 prefill pf5/pf6 family + batch f16 WMMA (split from prefill.cuh); all PD_TC5_OK-gated, so verify changes here on sm_100
#include "src/moe/block_scale_quant.cuh"
#include "src/moe/decode_block_scale.cuh"
#include "src/gemm/dense_fp4_w8.cuh"
#include "src/gemm/dense_tc5.cuh"       // tcgen05 dense GEMM families (split from dense_fp4_w8); needs its mma/tmap helpers
#include "src/gemm/dense_f8_decode.cuh" // e4m3 decode/GEMV lane, mma_ks twins, rowwise prefill; must follow dense_tc5 (uses pd_rowq_*/pd_tc5{p,q}_* from it)
#include "src/gemm/f8_lin.cuh"      // tile-linear f8 weight lane (needs int8_mma ldm helpers + dense_fp4_w8's mma helpers; tmap builders now from tma_desc)
#include"src/attn/decode_tc5.cuh"  // tcgen05 decode attention (needs tma_desc's pd_tc5_sdesc + f32_qkv's pd_attn_tmap_kv_f8s - no longer tied to dense_fp4_w8)
#include "src/quant/nvf4.cuh"
#include "src/moe/nvf4_expert.cuh"   // NVFP4 MoE expert consumers + persistent raw-ring + TM/TF plane twins (split from quant/nvf4.cuh); needs its quantizers/mma helpers
#include "src/moe/nvf4_sorted.cuh"   // NVFP4 MoE over the sorted layout + sm_100 sorted-tile arm + decode expert GEMVs; follows nvf4_expert
#include"src/moe/nvf4_st.cuh"      // tiled-layout MoE consumers (skinny-tile pair; needs nvf4's mma/dot4w helpers)
#include "src/moe/q8.cuh"
#include "src/moe/f8.cuh"           // tcgen05 e4m3 grouped MoE (needs attn/decode + moe/block_scale_quant; its tc5 descriptors now come from tma_desc)
#include "src/moe/f8row.cuh"        // flat per-row-scale e4m3 expert GEMM, sm_89+ mma.sync (needs moe/mmq stage_y + int8_mma's PD_MMA_OK)
#include "src/quant/iq_grids.cuh"   // ggml i-quant codebooks (generated from ggml-common.h)
#include "src/quant/iquant.cuh"     // IQ1/IQ2/IQ3/IQ4_NL on the k-quant streams: repack, dequant, window unpack (needs abi.cuh only)
#include "src/quant/kquant.cuh"     // must precede exports.cuh (which closes the table)
#include "src/quant/kquant_w4a8.cuh"  // stage-2 W4A8 (needs kquant layouts + gemm/mmq constants)
#include "src/asr/whisper.cuh"      // whisper decode lane (flash-decoding attn + fused decode epilogues)
#include "src/asr/granite_speech.cuh"  // granite-speech conformer tower (macaron FFN, GLU, centered dwconv, Shaw-RPE attention)
#include "src/moe/kquant.cuh"       // k-quant MoE expert seats (needs the two kquant segments)
#include "src/moe/offload.cuh"      // MoE expert offload: device-managed LRU slot cache over host-mapped expert planes (needs kquant.cuh layouts; plain CUDA)
#include "src/dflash.cuh"        // DFlash2 drafter grouped dynamic conv (abi.cuh helpers only)
#include "src/tier/xfer.cuh"     // KV tier extent gather/scatter (kv-offload 1a.2; abi.cuh helpers only)
#include "src/exports.cuh"
