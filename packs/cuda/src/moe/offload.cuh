// moe/offload.cuh - MoE expert offload: a device-managed LRU cache of routed
// experts in VRAM, fed from a host-mapped mirror of the full expert planes.
//
// Everything here runs INSIDE the decode/prefill graphs: routing -> resolve
// (expert id -> cache slot, LRU victim on a miss) -> fill (copy the missing
// experts' repacked bytes from the pinned host mirror into their slots) ->
// the unchanged MoE kernels over the slot planes with the remapped ids. No
// host round-trip, no sync, and the cache state lives in device memory, so a
// captured graph keeps making correct decisions on every replay.
//
// Slot planes carry the same repacked k-quant layout as a resident plane
// (moe/kquant.cuh addressing `(slot*ff + o) * n_super * bytes`), which is
// what lets the consumer kernels stay untouched: a slot IS an expert index
// into a plane that happens to hold S experts instead of n_expert.
//
// resolve: one block, thread 0 walks the rows in order. Rows are at most a
// few hundred on the token-batched class this serves (decode: B x top-k), so
// a serial walk is microseconds and keeps the LRU bookkeeping trivially
// race-free. A row whose expert is resident takes its slot; a miss takes the
// least-recently-used slot that no row of THIS tick pinned (so a tick never
// evicts what it is about to read - the caller guarantees rows <= S). Empty
// slots have last_use 0 and are taken first.
//
// fill: grid (chunks, 6 streams, jobs). Every block re-reads the job count
// the resolve wrote, so the launch is shaped for the maximum and idles the
// blocks past the real count - graph-stable grids, live job counts.
// The six streams are gate/up/down x data/scales; sizes come from the
// planes, the kernel only knows byte ranges. uint4 copies, coalesced: this
// is the PCIe-bound path; measured at 12.6 GB/s on a PCIe 4.0 x8 link, the
// link's practical ceiling for this access pattern.

#define PD_MOE_CACHE_NONE 0xFFFFFFFFu

__global__ void pd_moe_cache_resolve_kernel(
    const unsigned int* __restrict__ idx, uint32_t rows, uint32_t n_slots,
    unsigned int* __restrict__ slot_of, unsigned int* __restrict__ expert_in,
    unsigned int* __restrict__ last_use, unsigned int* __restrict__ tick,
    unsigned int* __restrict__ idx_slot, unsigned int* __restrict__ jobs,
    unsigned int* __restrict__ n_jobs, unsigned int* __restrict__ stats) {
    if (threadIdx.x != 0) return;
    const unsigned int t = *tick + 1u;
    *tick = t;
    unsigned int nj = 0;
    for (uint32_t r = 0; r < rows; ++r) {
        const unsigned int e = idx[r];
        unsigned int s = slot_of[e];
        if (s != PD_MOE_CACHE_NONE) {
            idx_slot[r] = s;
            last_use[s] = t;
            continue;
        }
        // miss: LRU victim among slots not pinned by this tick
        unsigned int victim = PD_MOE_CACHE_NONE, best = 0xFFFFFFFFu;
        for (uint32_t c = 0; c < n_slots; ++c) {
            const unsigned int lu = last_use[c];
            if (lu != t && lu < best) { best = lu; victim = c; }
        }
        if (victim == PD_MOE_CACHE_NONE) {
            // cannot happen when rows <= n_slots (caller's contract); make the
            // failure loud rather than silent: point the row at slot 0 and
            // flag it in the job count's high bit
            idx_slot[r] = 0;
            nj |= 0x80000000u;
            continue;
        }
        const unsigned int old = expert_in[victim];
        if (old != PD_MOE_CACHE_NONE) slot_of[old] = PD_MOE_CACHE_NONE;
        expert_in[victim] = e;
        slot_of[e] = victim;
        last_use[victim] = t;
        idx_slot[r] = victim;
        jobs[2u * (nj & 0x7FFFFFFFu)] = victim;
        jobs[2u * (nj & 0x7FFFFFFFu) + 1u] = e;
        ++nj;
    }
    *n_jobs = nj;
    // running counters for the hit-rate readout: [0] rows resolved, [1] misses
    stats[0] += rows;
    stats[1] += nj & 0x7FFFFFFFu;
}

struct PdMoeFillDesc {
    unsigned long long src[6];    // host-mirror device pointers, one per stream
    unsigned long long dst[6];    // slot-plane base pointers
    unsigned long long bytes[6];  // bytes per expert per stream (multiple of 16)
};

__global__ void __launch_bounds__(256) pd_moe_cache_fill_kernel(
    const unsigned int* __restrict__ jobs, const unsigned int* __restrict__ n_jobs,
    const __grid_constant__ PdMoeFillDesc d) {
    const uint32_t j = blockIdx.z;
    if (j >= (*n_jobs & 0x7FFFFFFFu)) return;
    const uint32_t k = blockIdx.y;
    const unsigned int slot = jobs[2u * j], e = jobs[2u * j + 1u];
    const unsigned long long nb = d.bytes[k];
    const uint4* __restrict__ src = (const uint4*)(d.src[k] + (unsigned long long)e * nb);
    uint4* __restrict__ dst = (uint4*)(d.dst[k] + (unsigned long long)slot * nb);
    const uint32_t n16 = (uint32_t)(nb >> 4);
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += gridDim.x * blockDim.x)
        dst[i] = src[i];
}

PD_EXPORT
int pd_moe_cache_resolve(const void* idx, uint32_t rows, uint32_t n_slots,
                         void* slot_of, void* expert_in, void* last_use,
                         void* tick, void* idx_slot, void* jobs, void* n_jobs,
                         void* stats, void* stream) {
    if (rows == 0 || n_slots == 0) return cudaErrorInvalidValue;
    if (rows > n_slots) return cudaErrorInvalidValue;
    pd_moe_cache_resolve_kernel<<<1, 32, 0, (cudaStream_t)stream>>>(
        (const unsigned int*)idx, rows, n_slots, (unsigned int*)slot_of,
        (unsigned int*)expert_in, (unsigned int*)last_use, (unsigned int*)tick,
        (unsigned int*)idx_slot, (unsigned int*)jobs, (unsigned int*)n_jobs,
        (unsigned int*)stats);
    return pd_launch_status();
}

// src/dst/bytes: HOST arrays of 6 u64 each, copied into the launch by value.
PD_EXPORT
int pd_moe_cache_fill(const void* jobs, const void* n_jobs, uint32_t max_jobs,
                      const void* src, const void* dst, const void* bytes,
                      void* stream) {
    if (max_jobs == 0) return 0;
    if (max_jobs > 1024u) return cudaErrorInvalidValue;
    PdMoeFillDesc d;
    for (int k = 0; k < 6; ++k) {
        d.src[k] = ((const unsigned long long*)src)[k];
        d.dst[k] = ((const unsigned long long*)dst)[k];
        d.bytes[k] = ((const unsigned long long*)bytes)[k];
        if (d.bytes[k] & 15ull) return cudaErrorInvalidValue;
    }
    // 16 chunks x 256 threads x 16 B = 64 KB per pass over a ~0.5 MB stream:
    // enough blocks in flight to keep the link busy (measured flat from 4 to
    // 64 chunks), small enough that a 1-job decode fill is not a 1000-block
    // launch
    dim3 grid(16u, 6u, max_jobs);
    pd_moe_cache_fill_kernel<<<grid, 256, 0, (cudaStream_t)stream>>>(
        (const unsigned int*)jobs, (const unsigned int*)n_jobs, d);
    return pd_launch_status();
}
