#include "common.cuh"

// ============================================================================
// Per-block R/G/B Pearson correlation -> min correlation. One thread per block,
// f64, exact CPU accumulation order (build.rs: --fmad=false) so the sums, the
// n*Sxx - Sx*Sx terms, and cov/denom round identically to the Rust reference.
// Only min_corr feeds the downstream neighbor/MAD logic, so that is all we emit.
// ============================================================================

extern "C" __global__ void channel_correlation_block(
    const unsigned char* __restrict__ rgb,
    double* __restrict__ out_min_corr,      // f64[blocks_x*blocks_y]
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

    double sr = 0.0, sg = 0.0, sb = 0.0;
    double srr = 0.0, sgg = 0.0, sbb = 0.0;
    double srg = 0.0, srb = 0.0, sgb = 0.0;
    double n = 0.0;

    for (unsigned int dy = 0; dy < block_size; ++dy) {
        for (unsigned int dx = 0; dx < block_size; ++dx) {
            unsigned int x = x0 + dx;
            unsigned int y = y0 + dy;
            if (x >= width || y >= height) continue;

            unsigned int idx = (y * width + x) * 3u;
            double r = (double)rgb[idx] / 255.0;
            double g = (double)rgb[idx + 1u] / 255.0;
            double b = (double)rgb[idx + 2u] / 255.0;

            sr += r; sg += g; sb += b;
            srr += r * r; sgg += g * g; sbb += b * b;
            srg += r * g; srb += r * b; sgb += g * b;
            n += 1.0;
        }
    }

    double min_corr;
    if (n < 4.0) {
        min_corr = 1.0;
    } else {
        // pearson(sx,sy,sxx,syy,sxy) = (n*sxy - sx*sy) / sqrt(max(varx*vary,1e-10))
        double var_r = n * srr - sr * sr;
        double var_g = n * sgg - sg * sg;
        double var_b = n * sbb - sb * sb;

        double rg = (n * srg - sr * sg) / sqrt(fmax(var_r * var_g, 1e-10));
        double rb = (n * srb - sr * sb) / sqrt(fmax(var_r * var_b, 1e-10));
        double gb = (n * sgb - sg * sb) / sqrt(fmax(var_g * var_b, 1e-10));

        min_corr = fmin(fmin(rg, rb), gb);
    }

    out_min_corr[tid] = min_corr;
}
