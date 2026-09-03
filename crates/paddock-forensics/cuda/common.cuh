#ifndef PADDOCK_FORENSICS_COMMON_CUH
#define PADDOCK_FORENSICS_COMMON_CUH

#include <cuda_runtime.h>
#include <math.h>

// ============================================================================
// Warp / block reductions (shuffle + shared memory).
// Deterministic within a launch: fixed thread-tree order, so a given input
// yields a bit-identical result run to run - a property the forensic parity
// test relies on.
// ============================================================================

__device__ inline float warp_reduce_sum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFFu, val, offset);
    }
    return val;
}

__device__ inline float warp_reduce_max(float val) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFFu, val, offset));
    }
    return val;
}

__device__ inline unsigned int warp_reduce_usum(unsigned int val) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFFu, val, offset);
    }
    return val;
}

/// Block-level sum reduction. Result valid on thread 0.
__device__ inline float block_reduce_sum(float val) {
    __shared__ float shared[32]; // one slot per warp (max 1024 threads)
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

    val = warp_reduce_sum(val);
    if (lane == 0) shared[warp_id] = val;
    __syncthreads();

    int num_warps = (blockDim.x + warpSize - 1) / warpSize;
    val = (threadIdx.x < num_warps) ? shared[lane] : 0.0f;
    if (warp_id == 0) val = warp_reduce_sum(val);
    return val;
}

/// Block-level unsigned-sum reduction. Result valid on thread 0.
__device__ inline unsigned int block_reduce_usum(unsigned int val) {
    __shared__ unsigned int shared[32];
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

    val = warp_reduce_usum(val);
    if (lane == 0) shared[warp_id] = val;
    __syncthreads();

    int num_warps = (blockDim.x + warpSize - 1) / warpSize;
    val = (threadIdx.x < num_warps) ? shared[lane] : 0u;
    if (warp_id == 0) val = warp_reduce_usum(val);
    return val;
}

// ============================================================================
// Pixel helpers
// ============================================================================

/// ITU-R BT.601 luminance from 8-bit RGB kept in [0,255] float space - matches
/// the CPU reference exactly (0.299 R + 0.587 G + 0.114 B).
__device__ inline float rgb_to_luma(float r, float g, float b) {
    return 0.299f * r + 0.587f * g + 0.114f * b;
}

#endif // PADDOCK_FORENSICS_COMMON_CUH
