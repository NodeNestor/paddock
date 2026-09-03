#include "common.cuh"

// ============================================================================
// Edge-sharpness profiling - per-block mean edge width + edge count.
//
// Canonical algorithm (CPU reference): per block, over inner pixels
// dy,dx in [2, bs-3], compute the Sobel gradient magnitude; where mag ≥
// threshold, w = min(mag / |laplacian|, 10) (or 10 when |lap| ≤ 1e-3); the
// block's mean_width = Σw / count. The host then does MAD-outlier detection on
// mean_width across blocks - that decision logic is order-independent, but
// mean_width itself is a float sum, so it must match the CPU BIT-FOR-BIT or a
// borderline block could flip flagged/not.
//
// This kernel therefore runs one thread per image block and accumulates in the
// exact CPU order in f64. build.rs compiles with --fmad=false, so gx*gx+gy*gy,
// the Sobel sums, and Σw carry no FMA contraction -> identical rounding to Rust's
// separate mul+add. sqrt/fabs/fmin are IEEE-correct on both. Result: the same
// per-block mean_width and count, hence identical findings. Exact parity.
// ============================================================================

extern "C" __global__ void edge_sharpness_block(
    const unsigned char* __restrict__ gray,
    double* __restrict__ out_mean_width,   // f64[blocks_x*blocks_y]
    int* __restrict__ out_count,           // i32[blocks_x*blocks_y]
    unsigned int width,
    unsigned int height,
    unsigned int block_size,
    double threshold,
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

    double w_sum = 0.0;
    int count = 0;

    // dy,dx in [2, block_size-3] - matches Rust `2..bs.saturating_sub(2)`.
    for (unsigned int dy = 2; dy + 2 < block_size; ++dy) {
        for (unsigned int dx = 2; dx + 2 < block_size; ++dx) {
            unsigned int x = x0 + dx;
            unsigned int y = y0 + dy;
            if (x >= width - 2 || y >= height - 2) continue;

#define F(px, py) ((double)gray[(py) * width + (px)])
            double gx = -F(x - 1, y - 1) + F(x + 1, y - 1)
                        - 2.0 * F(x - 1, y) + 2.0 * F(x + 1, y)
                        - F(x - 1, y + 1) + F(x + 1, y + 1);
            double gy = -F(x - 1, y - 1) - 2.0 * F(x, y - 1)
                        - F(x + 1, y - 1) + F(x - 1, y + 1)
                        + 2.0 * F(x, y + 1) + F(x + 1, y + 1);

            double mag = sqrt(gx * gx + gy * gy);
            if (mag < threshold) continue;

            double center = F(x, y);
            double lap = -4.0 * center
                         + F(x - 1, y) + F(x + 1, y)
                         + F(x, y - 1) + F(x, y + 1);
#undef F
            double abs_lap = fabs(lap);
            double w = (abs_lap > 1e-3) ? fmin(mag / abs_lap, 10.0) : 10.0;

            w_sum += w;
            count += 1;
        }
    }

    out_mean_width[tid] = (count > 0) ? (w_sum / (double)count) : 0.0;
    out_count[tid] = count;
}
