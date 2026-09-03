#include "common.cuh"

// ============================================================================
// Per-block Haar subband energies (LL/LH/HL/HH), each = Σ(coeff²)/count over the
// block's 2×2 cells. One thread per block, f64, exact CPU order (--fmad=false),
// so the four energies match the Rust reference bit-for-bit; the host derives
// hh_ratio / detail_ratio / directional_balance and does the MAD logic.
// ============================================================================

extern "C" __global__ void wavelet_subband_block(
    const unsigned char* __restrict__ gray,
    double* __restrict__ out_ll,
    double* __restrict__ out_lh,
    double* __restrict__ out_hl,
    double* __restrict__ out_hh,
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
    unsigned int half = block_size >> 1;

    double ll_sum = 0.0, lh_sum = 0.0, hl_sum = 0.0, hh_sum = 0.0;
    int count = 0;

    for (unsigned int py = 0; py < half; ++py) {
        for (unsigned int px = 0; px < half; ++px) {
            unsigned int x = x0 + px * 2u;
            unsigned int y = y0 + py * 2u;
            if (x + 1u >= width || y + 1u >= height) continue;

            double a = (double)gray[y * width + x] / 255.0;
            double b = (double)gray[y * width + x + 1u] / 255.0;
            double c = (double)gray[(y + 1u) * width + x] / 255.0;
            double d = (double)gray[(y + 1u) * width + x + 1u] / 255.0;

            double ll = (a + b + c + d) * 0.5;
            double lh = (a + b - c - d) * 0.5;
            double hl = (a - b + c - d) * 0.5;
            double hh = (a - b - c + d) * 0.5;

            ll_sum += ll * ll;
            lh_sum += lh * lh;
            hl_sum += hl * hl;
            hh_sum += hh * hh;
            count += 1;
        }
    }

    double n = (count > 0) ? (double)count : 1.0;
    out_ll[tid] = ll_sum / n;
    out_lh[tid] = lh_sum / n;
    out_hl[tid] = hl_sum / n;
    out_hh[tid] = hh_sum / n;
}
