// tier/xfer.cuh - KV tier extent gather/scatter (kv-offload).
// Textually-included segment of the single pack translation unit.
// Not standalone-compilable: include order is defined by ../pack.cu.
//
// The one lesson every serving stack pays for separately, reproduced
// on our own bus by the R1 probe:
// per-page transfers of a paged KV pool run the PCIe leg at ~5% of ceiling
// (8 KiB fragments), while >=2 MiB contiguous extents ride at 97%. So the
// demote path is: this gather kernel rearranges scattered pool blocks into a
// page-first contiguous extent in device staging (all planes of a block-run
// back to back), then one plain cudaMemcpyAsync moves the extent at full
// rate. Restore mirrors it: one H2D into staging, then scatter back into
// pool blocks. Layout transform on-die where bandwidth is ~800 GB/s, bus leg
// contiguous - inspired by the technique Strata/vLLM converged on, written
// in-house per the kernel policy.
//
// Extent record layout (the engine's `kv_tier` payload contract): for block
// record b (0..n_blocks) and plane p (0..n_planes),
//   extent[b * record_stride + dst_off[p] .. + bytes[p]]
//     <-> plane_base[p] + block_ids[b] * src_stride[p] .. + bytes[p]
// `planes` is a device u64 array of 4-tuples {base, stride, bytes, dst_off}.
// All of base/stride/bytes/dst_off/record_stride are multiples of 16 (the
// engine validates before launch) so every copy is uint4-vectorized.

// One CTA per (chunk, plane, block); each CTA strides its plane's bytes in
// 16 B vectors. Grid: x = copy chunks (sized by the launcher from the widest
// plane), y = plane, z = block record.
__global__ void pd_kv_gather_blocks_kernel(
        const unsigned long long* __restrict__ planes,
        const uint32_t* __restrict__ block_ids,
        char* __restrict__ extent, unsigned long long record_stride) {
    const uint32_t p = blockIdx.y;
    const uint32_t b = blockIdx.z;
    const unsigned long long base   = planes[4ull * p + 0];
    const unsigned long long stride = planes[4ull * p + 1];
    const unsigned long long bytes  = planes[4ull * p + 2];
    const unsigned long long doff   = planes[4ull * p + 3];
    const uint4* __restrict__ src = (const uint4*)(base + (unsigned long long)block_ids[b] * stride);
    uint4* __restrict__ dst = (uint4*)(extent + (unsigned long long)b * record_stride + doff);
    const unsigned long long n16 = bytes >> 4;
    for (unsigned long long i =
             (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < n16; i += (unsigned long long)gridDim.x * blockDim.x) {
        dst[i] = src[i];
    }
}

__global__ void pd_kv_scatter_blocks_kernel(
        const unsigned long long* __restrict__ planes,
        const uint32_t* __restrict__ block_ids,
        const char* __restrict__ extent, unsigned long long record_stride) {
    const uint32_t p = blockIdx.y;
    const uint32_t b = blockIdx.z;
    const unsigned long long base   = planes[4ull * p + 0];
    const unsigned long long stride = planes[4ull * p + 1];
    const unsigned long long bytes  = planes[4ull * p + 2];
    const unsigned long long doff   = planes[4ull * p + 3];
    uint4* __restrict__ dst = (uint4*)(base + (unsigned long long)block_ids[b] * stride);
    const uint4* __restrict__ src = (const uint4*)(extent + (unsigned long long)b * record_stride + doff);
    const unsigned long long n16 = bytes >> 4;
    for (unsigned long long i =
             (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < n16; i += (unsigned long long)gridDim.x * blockDim.x) {
        dst[i] = src[i];
    }
}

// Shared launch shape: 256 threads/CTA; x-chunks from the widest plane so
// small planes just exit their loop early. max_plane_bytes is a HOST-side
// value (the planes array lives on device and cannot shape the grid), the
// engine computes it alongside the descriptor upload.
static inline dim3 pd_kv_xfer_grid(unsigned long long max_plane_bytes,
                                   uint32_t n_planes, uint32_t n_blocks) {
    const unsigned long long n16 = max_plane_bytes >> 4;
    uint32_t x = (uint32_t)((n16 + 255) / 256);
    if (x == 0) x = 1;
    if (x > 64) x = 64;  // deep planes loop; 64 CTAs/plane saturates DRAM
    return dim3(x, n_planes, n_blocks);
}

PD_EXPORT
int pd_kv_gather_blocks(const void* planes, const void* block_ids,
                        void* extent, unsigned long long record_stride,
                        unsigned long long max_plane_bytes, uint32_t n_planes,
                        uint32_t n_blocks, void* stream) {
    if (n_planes == 0 || n_blocks == 0) return 0;
    const dim3 grid = pd_kv_xfer_grid(max_plane_bytes, n_planes, n_blocks);
    pd_kv_gather_blocks_kernel<<<grid, 256u, 0, (cudaStream_t)stream>>>(
        (const unsigned long long*)planes, (const uint32_t*)block_ids,
        (char*)extent, record_stride);
    return pd_launch_status();
}

PD_EXPORT
int pd_kv_scatter_blocks(const void* planes, const void* block_ids,
                         const void* extent, unsigned long long record_stride,
                         unsigned long long max_plane_bytes, uint32_t n_planes,
                         uint32_t n_blocks, void* stream) {
    if (n_planes == 0 || n_blocks == 0) return 0;
    const dim3 grid = pd_kv_xfer_grid(max_plane_bytes, n_planes, n_blocks);
    pd_kv_scatter_blocks_kernel<<<grid, 256u, 0, (cudaStream_t)stream>>>(
        (const unsigned long long*)planes, (const uint32_t*)block_ids,
        (const char*)extent, record_stride);
    return pd_launch_status();
}
