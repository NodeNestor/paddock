#include "common.cuh"

// ============================================================================
// Wavelet noise estimate - per-block median of |Haar HH| coefficients.
//
// The canonical CPU estimator (Donoho-Johnstone MAD) is σ = median(|HH|)/0.6745
// per block. The per-block MEDIAN is the parallel-unfriendly part; this kernel
// computes it exactly: one Haar HH coefficient per thread, then a bitonic sort
// of the block's |HH| values in shared memory, and thread 0 writes the median.
//
// HH = (a - b - c + d)/2 over each 2×2 cell of the block, with a..d in [0,255],
// so every |HH| is a multiple of 0.5 - exactly representable in f32 AND f64.
// The sort is a permutation of exact values, so the GPU median equals the CPU
// median bit-for-bit; the /0.6745 and all cross-block statistics run host-side
// on the identical medians. Exact parity, no tolerance.
//
// block_size ≤ 32 -> HH count = (block_size/2)² ≤ 256. Threads past the count
// pad with +INF (sorts to the top, never the median for count > 0).
// ============================================================================

extern "C" __global__ void noise_block_median(
    const unsigned char* __restrict__ gray,
    float* __restrict__ out,         // f32[blocks_x*blocks_y], per-block median(|HH|)
    unsigned int width,
    unsigned int block_size
) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tid = threadIdx.x;           // 0..255 (blockDim.x == 256)
    unsigned int half = block_size >> 1;
    unsigned int count = half * half;
    int x0 = bx * (int)block_size;
    int y0 = by * (int)block_size;

    __shared__ float s[256];
    float v = INFINITY;
    if ((unsigned int)tid < count) {
        unsigned int hx = (unsigned int)tid % half;
        unsigned int hy = (unsigned int)tid / half;
        unsigned int ax = (unsigned int)x0 + 2u * hx;
        unsigned int ay = (unsigned int)y0 + 2u * hy;
        float a = (float)gray[ay * width + ax];
        float b = (float)gray[ay * width + ax + 1u];
        float c = (float)gray[(ay + 1u) * width + ax];
        float d = (float)gray[(ay + 1u) * width + ax + 1u];
        v = fabsf((a - b - c + d) * 0.5f);
    }
    s[tid] = v;
    __syncthreads();

    // Ascending bitonic sort over the fixed 256-lane array.
    for (unsigned int k = 2u; k <= 256u; k <<= 1) {
        for (unsigned int j = k >> 1; j > 0u; j >>= 1) {
            unsigned int ixj = (unsigned int)tid ^ j;
            if (ixj > (unsigned int)tid) {
                bool up = (((unsigned int)tid & k) == 0u);
                float x = s[tid];
                float y = s[ixj];
                if ((x > y) == up) {
                    s[tid] = y;
                    s[ixj] = x;
                }
            }
            __syncthreads();
        }
    }

    if (tid == 0) {
        out[by * gridDim.x + bx] = s[count >> 1];
    }
}
