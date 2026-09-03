#include "common.cuh"

// ============================================================================
// Per-block lighting-direction plane fit: I(dx,dy) = a*dx + b*dy + c via
// least squares (Johnson & Farid 2007). The device part is TRANSCENDENTAL-FREE
// - only the normal-equation sums + a 3×3 Cramer's-rule solve (+,-,*,/), so with
// --fmad=false it is bit-identical to the Rust reference. The kernel emits only
// the raw gradient (a=grad_x, b=grad_y); direction = atan2(b,a) and magnitude =
// sqrt(a²+b²), plus all downstream cos/sin/atan2 neighbour math, run host-side
// in Rust for both paths, so no libm-vs-CUDA transcendental drift can enter.
//
// Edge cases match the CPU exactly: n<4 or |det|<1e-10 -> a=b=0 (-> dir 0, mag 0).
// ============================================================================

extern "C" __global__ void lighting_plane_block(
    const unsigned char* __restrict__ gray,
    double* __restrict__ out_a,   // grad_x
    double* __restrict__ out_b,   // grad_y
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

    double sx = 0.0, sy = 0.0, si = 0.0;
    double sxx = 0.0, syy = 0.0, sxy = 0.0;
    double sxi = 0.0, syi = 0.0;
    double n = 0.0;

    for (unsigned int dy = 0; dy < block_size; ++dy) {
        for (unsigned int dx = 0; dx < block_size; ++dx) {
            unsigned int x = x0 + dx;
            unsigned int y = y0 + dy;
            if (x >= width || y >= height) continue;

            double intensity = (double)gray[y * width + x] / 255.0;
            double fdx = (double)dx;
            double fdy = (double)dy;

            sx += fdx; sy += fdy; si += intensity;
            sxx += fdx * fdx; syy += fdy * fdy; sxy += fdx * fdy;
            sxi += fdx * intensity; syi += fdy * intensity;
            n += 1.0;
        }
    }

    double a = 0.0, b = 0.0;
    if (n >= 4.0) {
        double det = sxx * (syy * n - sy * sy)
                   - sxy * (sxy * n - sy * sx)
                   + sx * (sxy * sy - syy * sx);
        if (fabs(det) >= 1e-10) {
            a = (sxi * (syy * n - sy * sy)
               - sxy * (syi * n - sy * si)
               + sx * (syi * sy - syy * si)) / det;
            b = (sxx * (syi * n - sy * si)
               - sxi * (sxy * n - sy * sx)
               + sx * (sxy * si - syi * sx)) / det;
        }
    }

    out_a[tid] = a;
    out_b[tid] = b;
}
