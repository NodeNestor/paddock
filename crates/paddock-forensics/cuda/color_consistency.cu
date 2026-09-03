#include "common.cuh"

// ============================================================================
// Per-block quantized RGB histogram (8×8×8 = 512 bins). One thread per block
// accumulates integer counts into a private 512-slot array, written as f32.
// Integer counting is order-independent and within f32's exact range, so the
// histogram equals the CPU reference exactly. Host normalizes + chi-squared MAD.
//
// Bin index matches the CPU exactly: q = min(v*8/256, 7); bin = r*64 + g*8 + b.
// ============================================================================

extern "C" __global__ void color_histogram_block(
    const unsigned char* __restrict__ rgb,
    float* __restrict__ out_hist,           // f32[blocks_x*blocks_y*512]
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

    unsigned int hist[512];
    for (int i = 0; i < 512; ++i) hist[i] = 0u;

    for (unsigned int dy = 0; dy < block_size; ++dy) {
        for (unsigned int dx = 0; dx < block_size; ++dx) {
            unsigned int x = x0 + dx;
            unsigned int y = y0 + dy;
            if (x >= width || y >= height) continue;

            unsigned int idx = (y * width + x) * 3u;
            unsigned int ri = (unsigned int)rgb[idx] * 8u / 256u;
            unsigned int gi = (unsigned int)rgb[idx + 1u] * 8u / 256u;
            unsigned int bi = (unsigned int)rgb[idx + 2u] * 8u / 256u;
            if (ri > 7u) ri = 7u;
            if (gi > 7u) gi = 7u;
            if (bi > 7u) bi = 7u;

            unsigned int bin = ri * 64u + gi * 8u + bi;
            hist[bin] += 1u;
        }
    }

    unsigned int base = tid * 512u;
    for (int i = 0; i < 512; ++i) {
        out_hist[base + (unsigned int)i] = (float)hist[i];
    }
}
