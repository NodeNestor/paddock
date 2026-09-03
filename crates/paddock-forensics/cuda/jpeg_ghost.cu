#include "common.cuh"

// ============================================================================
// JPEG Ghost - per-block, per-quality mean-squared-error against a re-save.
//
// The heavy part of the ghost sweep is the per-block MSE for every quality
// level; the JPEG re-encode/decode stays on the host. This kernel computes the
// exact integer sum of squared byte differences per (quality, block) - for a
// 64×64 block that sum is ≤ 64·64·3·255² ≈ 8.0e8, which fits in u32 - so the
// result is bit-identical to the CPU reference (no float rounding). The host
// divides by (block_size²·3) and does the best-quality / histogram reduction.
// ============================================================================

/// `sweep`   : work image, interleaved RGB u8, length n*3
/// `resaved` : k re-saved images concatenated, interleaved RGB u8, length k*n*3
/// `out`     : u32[k * blocks_x * blocks_y], out[q*nblocks + by*blocks_x+bx] =
///             sum over the block of (sweep - resaved_q)^2
/// Launch: blockDim = (256,1,1); grid = (blocks_x, blocks_y, k). One CTA per
/// (block, quality).
extern "C" __global__ void jghost_block_sse(
    const unsigned char* __restrict__ sweep,
    const unsigned char* __restrict__ resaved,
    unsigned int* __restrict__ out,
    unsigned int width,
    unsigned int height,
    unsigned int block_size,
    unsigned int blocks_x
) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int q = blockIdx.z;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    unsigned int n = width * height;
    unsigned int q_off = (unsigned int)q * n * 3u;
    int x0 = bx * (int)block_size;
    int y0 = by * (int)block_size;
    int elems = (int)block_size * (int)block_size * 3; // RGB bytes in the block

    // Each thread accumulates squared diffs over a strided slice of the block's
    // RGB bytes. Integer accumulation -> exact.
    unsigned int local = 0u;
    for (int i = tid; i < elems; i += nthreads) {
        int px = i / 3;           // pixel index within block
        int c = i % 3;            // channel
        int dx = px % (int)block_size;
        int dy = px / (int)block_size;
        int x = x0 + dx;
        int y = y0 + dy;
        if (x >= (int)width || y >= (int)height) continue;
        unsigned int idx = ((unsigned int)y * width + (unsigned int)x) * 3u + (unsigned int)c;
        int a = (int)sweep[idx];
        int b = (int)resaved[q_off + idx];
        int d = a - b;
        local += (unsigned int)(d * d);
    }

    unsigned int total = block_reduce_usum(local);
    if (tid == 0) {
        int nblocks = gridDim.x * gridDim.y;
        out[(unsigned int)q * nblocks + (unsigned int)(by * gridDim.x + bx)] = total;
    }
}
