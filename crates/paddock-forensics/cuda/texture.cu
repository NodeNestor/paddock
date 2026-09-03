#include "common.cuh"

// ============================================================================
// Per-block 256-bin LBP (Local Binary Pattern) histogram. One thread per block
// accumulates integer counts into a private 256-slot array, then writes them as
// f32. Counts are integers well within f32's exact range, and integer counting
// is order-independent, so the histogram equals the CPU reference exactly - no
// --fmad concern here. The host normalizes + does chi-squared neighbor MAD.
//
// LBP code bit order matches the CPU exactly (clockwise from top-left = 128).
// ============================================================================

extern "C" __global__ void lbp_histogram_block(
    const unsigned char* __restrict__ gray,
    float* __restrict__ out_hist,           // f32[blocks_x*blocks_y*256]
    unsigned int width,
    unsigned int height,
    unsigned int block_size,
    unsigned int blocks_x,
    unsigned int blocks_y
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = blocks_x * blocks_y;
    if (tid >= total) return;

    unsigned int bx = tid % blocks_x;
    unsigned int by = tid / blocks_x;
    unsigned int x0 = bx * block_size;
    unsigned int y0 = by * block_size;

    unsigned int hist[256];
    for (int i = 0; i < 256; ++i) hist[i] = 0u;

    for (unsigned int dy = 1; dy + 1 < block_size; ++dy) {
        for (unsigned int dx = 1; dx + 1 < block_size; ++dx) {
            unsigned int x = x0 + dx;
            unsigned int y = y0 + dy;
            if (x >= width - 1u || y >= height - 1u) continue;

            unsigned char center = gray[y * width + x];
#define G(px, py) (gray[(py) * width + (px)])
            unsigned int code = 0u;
            code |= (G(x - 1u, y - 1u) >= center ? 1u : 0u) * 128u;
            code |= (G(x, y - 1u) >= center ? 1u : 0u) * 64u;
            code |= (G(x + 1u, y - 1u) >= center ? 1u : 0u) * 32u;
            code |= (G(x + 1u, y) >= center ? 1u : 0u) * 16u;
            code |= (G(x + 1u, y + 1u) >= center ? 1u : 0u) * 8u;
            code |= (G(x, y + 1u) >= center ? 1u : 0u) * 4u;
            code |= (G(x - 1u, y + 1u) >= center ? 1u : 0u) * 2u;
            code |= (G(x - 1u, y) >= center ? 1u : 0u);
#undef G
            hist[code] += 1u;
        }
    }

    unsigned int base = tid * 256u;
    for (int i = 0; i < 256; ++i) {
        out_hist[base + (unsigned int)i] = (float)hist[i];
    }
}
