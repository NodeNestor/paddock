#include "common.cuh"

// ============================================================================
// Error Level Analysis (ELA) - GPU path of the one canonical algorithm shared
// with the CPU reference (crates/paddock-forensics/src/pixel/ela.rs).
//
// All pixel values are kept in [0,255] float space (not normalized to [0,1]) so
// the arithmetic matches the CPU reference bit-for-bit up to f32-vs-f64 rounding
// - which is what makes the two-level parity test meaningful.
//
// The JPEG re-encode/decode stays on the host (there is no GPU JPEG codec);
// these kernels do the embarrassingly-parallel per-pixel and per-block work:
//   1. fela_error_map   - multi-scale per-pixel error map
//   2. fela_block_stats - per-block mean error, luminance complexity, adaptive
//                         hotspot count (fused; one CTA per block)
//   3. fela_global      - global sum / sum-of-squares over the whole map
// The tiny cross-block outlier reduction is done once on the host, shared
// verbatim with the CPU path.
// ============================================================================

/// Per-block statistics. repr matches the Rust `ElaBlockStatsRaw` (16 bytes).
struct FElaBlockStats {
    float mean_error;
    float complexity;        // luminance variance over the block
    unsigned int hotspot_count;
    unsigned int _pad;
};

/// Multi-scale per-pixel ELA error map.
///
/// `orig`    : work image, interleaved RGB u8, length n*3
/// `resaved` : k re-saved images concatenated, interleaved RGB u8, length k*n*3
/// `error_map[i]` = mean over the k scales of (|dR|+|dG|+|dB|)/3 - identical
/// accumulation order to the CPU reference.
extern "C" __global__ void fela_error_map(
    const unsigned char* __restrict__ orig,
    const unsigned char* __restrict__ resaved,
    float* __restrict__ error_map,
    unsigned int n,
    unsigned int k
) {
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += blockDim.x * gridDim.x) {
        unsigned int oi = i * 3u;
        float acc = 0.0f;
        for (unsigned int q = 0; q < k; ++q) {
            unsigned int ri = q * n * 3u + oi;
            int dr = (int)orig[oi + 0] - (int)resaved[ri + 0];
            int dg = (int)orig[oi + 1] - (int)resaved[ri + 1];
            int db = (int)orig[oi + 2] - (int)resaved[ri + 2];
            // sum of |channel diffs| (integer, exact) then /3 - matches CPU.
            int s = abs(dr) + abs(dg) + abs(db);
            acc += (float)s / 3.0f;
        }
        error_map[i] = acc / (float)k;
    }
}

/// Per-block mean error + luminance complexity + adaptive hotspot count.
/// Launch: blockDim = (256,1,1); grid = (blocks_x, blocks_y, 1). One CTA owns
/// one `block_size` x `block_size` tile of the (floor-covered) image region.
extern "C" __global__ void fela_block_stats(
    const float* __restrict__ error_map,
    const unsigned char* __restrict__ orig,   // work image, interleaved RGB u8
    unsigned int width,
    unsigned int height,
    unsigned int block_size,
    unsigned int blocks_x,
    FElaBlockStats* __restrict__ out
) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int x0 = bx * block_size;
    int y0 = by * block_size;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;              // 256 (power of two)
    int ppb = block_size * block_size;

    // ---- pass 1: reduce error sum + luminance sum/sum_sq over the block ----
    float local_err = 0.0f, local_lsum = 0.0f, local_lsq = 0.0f;
    for (int i = tid; i < ppb; i += nthreads) {
        int dx = i % (int)block_size;
        int dy = i / (int)block_size;
        int x = x0 + dx;
        int y = y0 + dy;
        if (x >= (int)width || y >= (int)height) continue;
        int idx = y * (int)width + x;
        local_err += error_map[idx];
        int pi = idx * 3;
        float lum = rgb_to_luma((float)orig[pi], (float)orig[pi + 1], (float)orig[pi + 2]);
        local_lsum += lum;
        local_lsq += lum * lum;
    }

    __shared__ float s_err[256];
    __shared__ float s_lsum[256];
    __shared__ float s_lsq[256];
    s_err[tid] = local_err;
    s_lsum[tid] = local_lsum;
    s_lsq[tid] = local_lsq;
    __syncthreads();
    for (int stride = nthreads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_err[tid] += s_err[tid + stride];
            s_lsum[tid] += s_lsum[tid + stride];
            s_lsq[tid] += s_lsq[tid + stride];
        }
        __syncthreads();
    }

    __shared__ float s_mean;
    __shared__ float s_complexity;
    __shared__ float s_threshold;
    if (tid == 0) {
        float count = (float)ppb;
        float mean = s_err[0] / count;
        float lmean = s_lsum[0] / count;
        // variance = E[x^2] - E[x]^2, clamped at 0 (fp can dip slightly negative;
        // a negative variance is nonsensical and would NaN the sqrt below).
        float complexity = fmaxf(s_lsq[0] / count - lmean * lmean, 0.0f);
        s_mean = mean;
        s_complexity = complexity;
        s_threshold = mean + 2.0f * sqrtf(complexity);
    }
    __syncthreads();

    // ---- pass 2: adaptive hotspot count (error > local threshold AND > 15) --
    unsigned int local_hot = 0;
    float thr = s_threshold;
    for (int i = tid; i < ppb; i += nthreads) {
        int dx = i % (int)block_size;
        int dy = i / (int)block_size;
        int x = x0 + dx;
        int y = y0 + dy;
        if (x >= (int)width || y >= (int)height) continue;
        int idx = y * (int)width + x;
        float e = error_map[idx];
        if (e > thr && e > 15.0f) local_hot++;
    }
    __shared__ unsigned int s_hot[256];
    s_hot[tid] = local_hot;
    __syncthreads();
    for (int stride = nthreads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s_hot[tid] += s_hot[tid + stride];
        __syncthreads();
    }

    if (tid == 0) {
        int block_idx = by * (int)blocks_x + bx;
        out[block_idx].mean_error = s_mean;
        out[block_idx].complexity = s_complexity;
        out[block_idx].hotspot_count = s_hot[0];
        out[block_idx]._pad = 0u;
    }
}

/// Global sum and sum-of-squares over the whole error map.
/// Launch with a single block of 256 threads -> deterministic reduction order.
/// out[0] = sum(error), out[1] = sum(error^2).
extern "C" __global__ void fela_global(
    const float* __restrict__ error_map,
    unsigned int n,
    float* __restrict__ out2
) {
    int tid = threadIdx.x;
    float lsum = 0.0f, lsq = 0.0f;
    for (unsigned int i = tid; i < n; i += blockDim.x) {
        float e = error_map[i];
        lsum += e;
        lsq += e * e;
    }
    __shared__ float s_sum[256];
    __shared__ float s_sq[256];
    s_sum[tid] = lsum;
    s_sq[tid] = lsq;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
            s_sq[tid] += s_sq[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        out2[0] = s_sum[0];
        out2[1] = s_sq[0];
    }
}
