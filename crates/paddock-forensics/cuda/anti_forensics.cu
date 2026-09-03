#include "common.cuh"

// ============================================================================
// Per-block anti-forensics features: first-order difference variance, high-pass
// (median-residual) kurtosis, and 32-bin histogram flatness. One thread per
// block, f64, exact CPU order (--fmad=false). The 3×3 median matches the CPU's
// sort-and-take-middle; powi(4)/powi(2) are reproduced as ((r²)²)/(x·x) so the
// rounding is identical. Host does the low/high MAD-outlier + Gaussian-kurtosis
// logic.
// ============================================================================

extern "C" __global__ void anti_forensics_block(
    const unsigned char* __restrict__ gray,
    double* __restrict__ out_diff_var,
    double* __restrict__ out_kurtosis,
    double* __restrict__ out_flatness,
    int* __restrict__ out_pixel_count,
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

    double diff_sum = 0.0, diff_sq = 0.0;
    int diff_count = 0;
    double res_sum = 0.0, res_sq = 0.0, res_4th = 0.0;
    int res_count = 0;
    unsigned int hist[32];
    for (int i = 0; i < 32; ++i) hist[i] = 0u;

    for (unsigned int dy = 1; dy + 1 < block_size; ++dy) {
        for (unsigned int dx = 1; dx + 1 < block_size; ++dx) {
            unsigned int x = x0 + dx;
            unsigned int y = y0 + dy;
            if (x >= width - 1u || y >= height - 1u) continue;

            double center = (double)gray[y * width + x] / 255.0;

            double dh = (double)gray[y * width + x + 1u] / 255.0 - center;
            double dv = (double)gray[(y + 1u) * width + x] / 255.0 - center;
            diff_sum += dh + dv;
            diff_sq += dh * dh + dv * dv;
            diff_count += 2;

            // 3×3 median (raw u8), neighbours in ky -1..1, kx -1..1 order.
            unsigned char v[9];
            int k = 0;
            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    v[k++] = gray[(y + ky) * width + (x + kx)];
                }
            }
            // Insertion sort -> v[4] is the median (order-independent value).
            for (int i = 1; i < 9; ++i) {
                unsigned char key = v[i];
                int j = i - 1;
                while (j >= 0 && v[j] > key) { v[j + 1] = v[j]; --j; }
                v[j + 1] = key;
            }
            double residual = (double)gray[y * width + x] / 255.0 - (double)v[4] / 255.0;
            res_sum += residual;
            res_sq += residual * residual;
            double r2 = residual * residual;
            res_4th += r2 * r2;   // matches Rust powi(4) = (r²)²
            res_count += 1;

            unsigned int bin = (unsigned int)(center * 31.999);
            if (bin > 31u) bin = 31u;
            hist[bin] += 1u;
        }
    }

    double diff_var = 0.0;
    if (diff_count > 1) {
        double mean = diff_sum / (double)diff_count;
        diff_var = diff_sq / (double)diff_count - mean * mean;
    }

    double kurtosis = 0.0;
    if (res_count > 3) {
        double m = res_sum / (double)res_count;
        double var = res_sq / (double)res_count - m * m;
        if (var > 1e-10) {
            kurtosis = (res_4th / (double)res_count) / (var * var);
        }
    }

    unsigned int htotal = 0u;
    unsigned int hmax = 0u;
    for (int i = 0; i < 32; ++i) {
        htotal += hist[i];
        if (hist[i] > hmax) hmax = hist[i];
    }
    double flatness = 0.0;
    if (htotal > 0u && hmax > 0u) {
        flatness = ((double)htotal / 32.0) / (double)hmax;
    }

    out_diff_var[tid] = diff_var;
    out_kurtosis[tid] = kurtosis;
    out_flatness[tid] = flatness;
    out_pixel_count[tid] = res_count;
}
