// Chunked SSD prefill scan (the structural prefill rung).
//
// pd_mamba2_scan_seq walks a segment token-by-token: one launch of
// n_heads/hpb (16-32) CTAs, serial in T, no tensor-scale parallelism - the
// serial walk costs ~55 us per 128-token chunk. vLLM's chunked
// SSD form (5 triton kernels, chunk_size=128 from the checkpoint) is the
// standard answer; this is the same reformulation, in-house:
// the recurrence S_t = a_t S_{t-1} + dt_t x_t B_t^T unrolls per 128-token
// chunk into GEMM-class pieces,
//
//   cum_t   = sum_{u<=t} dt_u A_h                        (K1, per-chunk scan)
//   M[t][s] = C_t . B_s                                  (K2, per-GROUP Gram)
//   dS      = sum_t exp(cum_L - cum_t) dt_t B_t x_t^T    (K3, per-head GEMM)
//   S_in(c) staged, S_run = exp(cum_L) S_run + dS        (K4, chunk chain)
//   y_t     = sum_{s<=t} M[t][s] exp(cum_t - cum_s) dt_s x_s   (intra)
//           + exp(cum_t) C_t . S_in(c)                         (inter)
//           + D_h x_t                                    (K5, per-head GEMM)
//
// so every kernel is parallel over (chunk, head) and only K4 - a cheap
// elementwise pass - walks chunks serially. All arithmetic f32; the f16
// arena rounds once at the segment boundary exactly like the serial walk
// (register-resident there, f32 scratch chain here), so the storage class
// is unchanged. Decays are exp of non-positive cumsums (dt >= 0, A < 0):
// every weight is <= 1, no overflow anywhere. expf (never __expf) - the
// serial kernels' precision class.
//
// Scratch is a per-call cudaMallocAsync blob (~12 MB at 4 chunks),
// stream-ordered and freed at the end of the chain. Not pack-static, on
// purpose: parallel test threads share one exec stream and interleave
// their submission sequences, so a static blob lets caller B's K1 clobber
// caller A's cum between A's K1 and K5 - measured as two flaky boundary
// flips in gpu_nemotron_batch before this landed. Stream-ordered private
// scratch is correct under any submission interleave; serving (one model,
// one tick thread) never even sees the hazard. Prefill is not graph-
// captured, so the async alloc is legal (the no-lazy-malloc law binds the
// captured decode graphs only); if the alloc ever fails the launcher falls
// back to the serial walk.
//
// Numerics class: the y/state values REGROUP vs the serial walk (that is
// the whole point), so the gates are the f64-host-reference class
// (mamba2_scan_seq_ssd_matches_f64_reference, T=200) plus the f32/f16
// twin bit-identity and run-to-run determinism - no atomics, fixed
// summation orders throughout.
//
// Election lives in the pd_mamba2_scan_seq{,_f16} launchers: SSD iff the
// segment is long enough to amortize the 5-launch chain (PD_SSD_MIN_T) on
// the pinned geometry (hd 64 / S 128). PADDOCK_NO_SSD_SEQ pins the serial
// walk (dev A/B; the witness is the pd_ssd_* kernel names in a capture).

#define PD_SSD_L 128u          // chunk length (the checkpoint convention)
#define PD_SSD_NCMAX 4u        // chunks per pass (512 tokens), passes loop
// Serial-vs-SSD crossover, measured on nemotron:
// at 128-token segments the chain still costs ~+30% vs the serial
// walk (fixed per-pass overheads dominate), at 512-token passes it wins
// -34%; interpolated break-even ~T=250. Two full chunks is the structural
// floor for the chain to pay.
#define PD_SSD_MIN_T 256u

// K1: per (chunk, head) - softplus dt, log-decay a, inclusive scan -> cum.
// One CTA = one (c,h); 128 threads = one token each. Tokens at/after the
// pass length (a partial tail chunk) get dt = 0: a = 0 keeps the cumsum
// flat and a zero dt zeroes every weight the pad token could contribute,
// so padding is identity by construction. cum/dtv are [c][h][t] blobs.
__global__ void pd_ssd_prep_kernel(
        const float* __restrict__ dt_raw, uint32_t dt_stride, uint32_t t0,
        const float* __restrict__ A, const float* __restrict__ dt_bias,
        uint32_t n_heads, uint32_t n_tok, float* __restrict__ cum,
        float* __restrict__ dtv) {
    const uint32_t h = blockIdx.x, c = blockIdx.y, t = threadIdx.x;
    const uint32_t gt = c * PD_SSD_L + t;   // pass-local token index
    float dt = 0.0f;
    if (gt < n_tok) {
        float v = dt_raw[(size_t)(t0 + gt) * dt_stride + h] + dt_bias[h];
        dt = (v <= PD_M2_SOFTPLUS_LIM) ? log1pf(expf(v)) : v;
    }
    __shared__ float sa[PD_SSD_L];
    sa[t] = dt * A[h];
    __syncthreads();
    // Hillis-Steele inclusive scan, 7 steps at L=128 - fixed order, so the
    // cumsum is deterministic
    #pragma unroll
    for (uint32_t off = 1u; off < PD_SSD_L; off <<= 1u) {
        const float prev = (t >= off) ? sa[t - off] : 0.0f;
        __syncthreads();
        sa[t] += prev;
        __syncthreads();
    }
    const size_t ch = (size_t)c * n_heads + h;
    dtv[ch * PD_SSD_L + t] = dt;
    cum[ch * PD_SSD_L + t] = sa[t];
}

// K2: per-(chunk, group) Gram M = C . B^T over the S=128 dot. Full L x L
// (the s > t half is masked in K5; computing it costs less than a ragged
// tile). CTA = 64 x 64 output tile, 128 threads x 32 accs, K staged in
// 16-wide slices. grid.x = 4 tiles (2x2), grid.y = c * G + g. m is a
// [c][g][t][s] blob.
template <uint32_t S_>
__global__ void __launch_bounds__(128, 1) pd_ssd_gram_kernel(
        const float* __restrict__ xbc, uint32_t t0, uint32_t conv_dim,
        uint32_t d_inner, uint32_t n_groups, uint32_t n_tok,
        float* __restrict__ m) {
    const uint32_t c = blockIdx.y / n_groups, g = blockIdx.y % n_groups;
    const uint32_t tile_t = (blockIdx.x >> 1) * 64u;   // 0 or 64
    const uint32_t tile_s = (blockIdx.x & 1u) * 64u;
    // thread (tt, ts): 16 x 8 threads, each owns 4(t) x 8(s) outputs
    const uint32_t tt = threadIdx.x / 8u, ts = threadIdx.x % 8u;
    float acc[4][8];
    #pragma unroll
    for (uint32_t a = 0; a < 4; ++a)
        #pragma unroll
        for (uint32_t b = 0; b < 8; ++b) acc[a][b] = 0.0f;

    __shared__ float sc[64][17], sb[64][17];  // [row][k-slice], padded
    const uint32_t base = c * PD_SSD_L;
    for (uint32_t k0 = 0; k0 < S_; k0 += 16u) {
        // stage C rows (t side) and B rows (s side) for this k slice:
        // 128 threads x 8 loads = the 64x16 tiles
        #pragma unroll
        for (uint32_t l = 0; l < 8; ++l) {
            const uint32_t e = threadIdx.x * 8u + l;    // 0..1023
            const uint32_t row = e / 16u, kk = e % 16u;
            const uint32_t trow = base + tile_t + row;
            const uint32_t srow = base + tile_s + row;
            sc[row][kk] = (trow < n_tok)
                ? xbc[(size_t)(t0 + trow) * conv_dim + d_inner
                      + (size_t)(n_groups + g) * S_ + k0 + kk]
                : 0.0f;
            sb[row][kk] = (srow < n_tok)
                ? xbc[(size_t)(t0 + srow) * conv_dim + d_inner
                      + (size_t)g * S_ + k0 + kk]
                : 0.0f;
        }
        __syncthreads();
        #pragma unroll
        for (uint32_t kk = 0; kk < 16u; ++kk)
            #pragma unroll
            for (uint32_t a = 0; a < 4; ++a) {
                const float cv = sc[tt * 4u + a][kk];
                #pragma unroll
                for (uint32_t b = 0; b < 8; ++b)
                    acc[a][b] += cv * sb[ts * 8u + b][kk];
            }
        __syncthreads();
    }
    float* mo = m + ((size_t)c * n_groups + g) * PD_SSD_L * PD_SSD_L;
    #pragma unroll
    for (uint32_t a = 0; a < 4; ++a)
        #pragma unroll
        for (uint32_t b = 0; b < 8; ++b)
            mo[(size_t)(tile_t + tt * 4u + a) * PD_SSD_L + tile_s + ts * 8u
               + b] = acc[a][b];
}

// K3: per (chunk, head) - dS[j][i] = sum_t w_t B_t[j] x_t[i] with
// w_t = exp(cum_L - cum_t) dt_t (<= dt_t, no overflow). GEMM K = L staged
// in 16-wide slices; CTA 256 threads = (32 j-quarters x 8 i-octets), each
// thread 4(j) x 8(i) accs over the 128 x 64 output. ds is a
// [c][h][j=S][i=HD] blob (i-minor, the arena layout). (A j-half grid.z
// split was tried and measured a wash at every nc - this kernel's bound
// is its staging chain, not CTA count; K5's t-half split, which did pay,
// stays.)
template <uint32_t S_, uint32_t HD_>
__global__ void __launch_bounds__(256, 1) pd_ssd_dstate_kernel(
        const float* __restrict__ xbc, uint32_t t0, uint32_t conv_dim,
        uint32_t d_inner, uint32_t n_groups, uint32_t n_heads,
        uint32_t n_tok, const float* __restrict__ cum,
        const float* __restrict__ dtv, float* __restrict__ ds) {
    const uint32_t h = blockIdx.x, c = blockIdx.y;
    const uint32_t g = h / (n_heads / n_groups);
    const uint32_t tj = threadIdx.x / 8u, ti = threadIdx.x % 8u;
    const float* cumr = cum + ((size_t)c * n_heads + h) * PD_SSD_L;
    const float* dtr = dtv + ((size_t)c * n_heads + h) * PD_SSD_L;
    const float cum_l = cumr[PD_SSD_L - 1u];
    float acc[4][8];
    #pragma unroll
    for (uint32_t a = 0; a < 4; ++a)
        #pragma unroll
        for (uint32_t b = 0; b < 8; ++b) acc[a][b] = 0.0f;

    __shared__ float sbw[16][S_ + 1];   // w_t-weighted B_t rows
    __shared__ float sx[16][HD_ + 1];   // x_t rows
    __shared__ float swt[16];           // w_t, one expf per token (the
                                        // first cut recomputed it per j -
                                        // 128x redundant, 49 us)
    const uint32_t base = c * PD_SSD_L;
    for (uint32_t k0 = 0; k0 < PD_SSD_L; k0 += 16u) {
        if (threadIdx.x < 16u) {
            const uint32_t row = base + k0 + threadIdx.x;
            swt[threadIdx.x] = (row < n_tok)
                ? expf(cum_l - cumr[k0 + threadIdx.x]) * dtr[k0 + threadIdx.x]
                : 0.0f;
        }
        __syncthreads();
        // stage 16 tokens: B (128 wide, weighted) and x (64 wide)
        for (uint32_t e = threadIdx.x; e < 16u * S_; e += 256u) {
            const uint32_t tt = e / S_, j = e % S_;
            const uint32_t row = base + k0 + tt;
            float v = 0.0f;
            if (row < n_tok)
                v = swt[tt] * xbc[(size_t)(t0 + row) * conv_dim + d_inner
                                  + (size_t)g * S_ + j];
            sbw[tt][j] = v;
        }
        for (uint32_t e = threadIdx.x; e < 16u * HD_; e += 256u) {
            const uint32_t tt = e / HD_, i = e % HD_;
            const uint32_t row = base + k0 + tt;
            sx[tt][i] = (row < n_tok)
                ? xbc[(size_t)(t0 + row) * conv_dim + (size_t)h * HD_ + i]
                : 0.0f;
        }
        __syncthreads();
        #pragma unroll
        for (uint32_t tt = 0; tt < 16u; ++tt)
            #pragma unroll
            for (uint32_t a = 0; a < 4; ++a) {
                const float bv = sbw[tt][tj * 4u + a];
                #pragma unroll
                for (uint32_t b = 0; b < 8; ++b)
                    acc[a][b] += bv * sx[tt][ti * 8u + b];
            }
        __syncthreads();
    }
    float* dso = ds + ((size_t)c * n_heads + h) * (S_ * HD_);
    #pragma unroll
    for (uint32_t a = 0; a < 4; ++a)
        #pragma unroll
        for (uint32_t b = 0; b < 8; ++b)
            dso[(size_t)(tj * 4u + a) * HD_ + ti * 8u + b] = acc[a][b];
}

// K4: the chunk chain - the one sequential piece, elementwise over the
// state. Per head: stage S_in(c) (into the dS slot, in place), advance
// S_run = exp(cum_L) S_run + dS. Arena IO at the pass edges: load on the
// first pass, store on the last (f16 rounds here and only here - one
// round per segment, the serial walk's class). Per-head state is
// HD*S = 8192 elements on the pinned geometry.
__global__ void pd_ssd_chain_kernel(
        void* __restrict__ state, int state_f16, uint32_t nc,
        uint32_t n_heads, int first_pass, int last_pass,
        const float* __restrict__ cum, float* __restrict__ ds,
        float* __restrict__ run) {
    const uint32_t h = blockIdx.x;
    // grid.y slices the per-head 8192 elements so the launch covers the
    // die (64 CTAs left 2/3 of the SMs idle at 35 us/launch); every slice
    // is thread-private, so slicing changes nothing else
    const uint32_t e0 = blockIdx.y * 2048u, e1 = e0 + 2048u;
    float* runh = run + (size_t)h * 8192u;
    if (first_pass) {
        if (state_f16) {
            const __half* src = (const __half*)state + (size_t)h * 8192u;
            for (uint32_t e = e0 + threadIdx.x; e < e1; e += blockDim.x)
                runh[e] = __half2float(src[e]);
        } else {
            const float* src = (const float*)state + (size_t)h * 8192u;
            for (uint32_t e = e0 + threadIdx.x; e < e1; e += blockDim.x)
                runh[e] = src[e];
        }
    }
    for (uint32_t c = 0; c < nc; ++c) {
        const float decay =
            expf(cum[((size_t)c * n_heads + h) * PD_SSD_L + PD_SSD_L - 1u]);
        float* dsc = ds + ((size_t)c * n_heads + h) * 8192u;
        for (uint32_t e = e0 + threadIdx.x; e < e1; e += blockDim.x) {
            const float old = runh[e];
            runh[e] = decay * old + dsc[e];
            dsc[e] = old;   // stage S_in(c) for K5, in place
        }
    }
    if (last_pass) {
        if (state_f16) {
            __half* dst = (__half*)state + (size_t)h * 8192u;
            for (uint32_t e = e0 + threadIdx.x; e < e1; e += blockDim.x)
                dst[e] = __float2half_rn(runh[e]);
        } else {
            float* dst = (float*)state + (size_t)h * 8192u;
            for (uint32_t e = e0 + threadIdx.x; e < e1; e += blockDim.x)
                dst[e] = runh[e];
        }
    }
}

// K5: per (chunk, head) - the output GEMM pair, in two tiled K-loops.
// Intra: y[t][i] += sum_s W[t][s] x[s][i] with W = M[t][s] exp(cum_t -
// cum_s) dt_s masked to s <= t; the W tile is built COOPERATIVELY into
// shared, one expf per (t,s) entry, then consumed as pure shared-FMA.
// (The first cut computed the expf inside every i-thread's inner loop -
// 64x redundant, and the kernel measured 282 us/launch, 16.7% of the lc8
// GPU time; the weight tile is the whole fix.) Inter: y[t][i] +=
// exp(cum_t) sum_j C_t[j] S_in[j][i], with the S_in slice staged through
// shared so the t-threads stop re-reading it from L2. Thread map is the
// K3 GEMM shape: per (chunk, head, t-half) CTA, (32 t-pairs x 8 i-octets),
// 2x8 accs. Fixed ascending-s / ascending-j orders: deterministic.
template <uint32_t S_, uint32_t HD_>
__global__ void __launch_bounds__(256, 1) pd_ssd_y_kernel(
        const float* __restrict__ xbc, uint32_t t0, uint32_t conv_dim,
        uint32_t d_inner, uint32_t n_groups, uint32_t n_heads,
        const float* __restrict__ D, float* __restrict__ y, uint32_t n_tok,
        const float* __restrict__ cum, const float* __restrict__ dtv,
        const float* __restrict__ m, const float* __restrict__ ds) {
    const uint32_t h = blockIdx.x, c = blockIdx.y;
    const uint32_t th = blockIdx.z * (PD_SSD_L / 2u);  // t-half (see K3's
                                                       // j-half note)
    const uint32_t g = h / (n_heads / n_groups);
    const uint32_t tj = threadIdx.x / 8u, ti = threadIdx.x % 8u;
    const float d_h = D[h];

    __shared__ float sx[PD_SSD_L][HD_];  // whole-chunk x, 32 KB (stride 64
                                         // floats = conflict-free)
    __shared__ float sw[PD_SSD_L / 2u][17];  // W tile, this CTA's t-half x
                                             // one 16-wide s-slice
    __shared__ float sjs[16][HD_ + 1];   // S_in slice for the inter loop
    __shared__ float scum[PD_SSD_L], sdt[PD_SSD_L];
    const uint32_t base = c * PD_SSD_L;
    for (uint32_t e = threadIdx.x; e < PD_SSD_L * HD_; e += 256u) {
        const uint32_t tt = e / HD_, ii = e % HD_;
        const uint32_t row = base + tt;
        sx[tt][ii] = (row < n_tok)
            ? xbc[(size_t)(t0 + row) * conv_dim + (size_t)h * HD_ + ii]
            : 0.0f;
    }
    const float* cumr = cum + ((size_t)c * n_heads + h) * PD_SSD_L;
    const float* dtr = dtv + ((size_t)c * n_heads + h) * PD_SSD_L;
    for (uint32_t e = threadIdx.x; e < PD_SSD_L; e += 256u) {
        scum[e] = cumr[e];
        sdt[e] = dtr[e];
    }
    __syncthreads();

    const float* mg = m + ((size_t)c * n_groups + g) * PD_SSD_L * PD_SSD_L;
    const float* sin = ds + ((size_t)c * n_heads + h) * (S_ * HD_);
    float acc[2][8];
    #pragma unroll
    for (uint32_t a = 0; a < 2; ++a)
        #pragma unroll
        for (uint32_t b = 0; b < 8; ++b) acc[a][b] = 0.0f;

    // ---- intra: W-tile GEMM over s ----------------------------------------
    // the lower t-half sees only s < 64 (everything above is masked), so
    // its s-walk stops early - sound, the skipped weights are all zero
    const uint32_t s_hi = th + PD_SSD_L / 2u;
    for (uint32_t s0 = 0; s0 < s_hi; s0 += 16u) {
        for (uint32_t e = threadIdx.x; e < (PD_SSD_L / 2u) * 16u; e += 256u) {
            const uint32_t tt = e / 16u, ss = e % 16u;
            const uint32_t t = th + tt, s = s0 + ss;
            sw[tt][ss] = (s <= t)
                ? mg[(size_t)t * PD_SSD_L + s] * expf(scum[t] - scum[s])
                      * sdt[s]
                : 0.0f;
        }
        __syncthreads();
        #pragma unroll
        for (uint32_t ss = 0; ss < 16u; ++ss)
            #pragma unroll
            for (uint32_t a = 0; a < 2; ++a) {
                const float w = sw[tj * 2u + a][ss];
                #pragma unroll
                for (uint32_t b = 0; b < 8; ++b)
                    acc[a][b] += w * sx[s0 + ss][ti * 8u + b];
            }
        __syncthreads();
    }

    // ---- inter: C . S_in over j, S_in slice staged ------------------------
    // per-thread inter accumulates separately so the exp(cum_t) decay can
    // scale it once at the end (same value class as the first cut)
    float inter[2][8];
    #pragma unroll
    for (uint32_t a = 0; a < 2; ++a)
        #pragma unroll
        for (uint32_t b = 0; b < 8; ++b) inter[a][b] = 0.0f;
    for (uint32_t j0 = 0; j0 < S_; j0 += 16u) {
        for (uint32_t e = threadIdx.x; e < 16u * HD_; e += 256u) {
            const uint32_t jj = e / HD_, ii = e % HD_;
            sjs[jj][ii] = sin[(size_t)(j0 + jj) * HD_ + ii];
        }
        __syncthreads();
        #pragma unroll
        for (uint32_t a = 0; a < 2; ++a) {
            const uint32_t t = th + tj * 2u + a;
            const uint32_t row = base + t;
            if (row >= n_tok) continue;
            const float* crow = xbc + (size_t)(t0 + row) * conv_dim
                              + d_inner + (size_t)(n_groups + g) * S_ + j0;
            #pragma unroll
            for (uint32_t jj = 0; jj < 16u; ++jj) {
                const float cv = crow[jj];
                #pragma unroll
                for (uint32_t b = 0; b < 8; ++b)
                    inter[a][b] += cv * sjs[jj][ti * 8u + b];
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (uint32_t a = 0; a < 2; ++a) {
        const uint32_t t = th + tj * 2u + a;
        const uint32_t row = base + t;
        if (row >= n_tok) continue;
        const float dec = expf(scum[t]);
        #pragma unroll
        for (uint32_t b = 0; b < 8; ++b) {
            const uint32_t i = ti * 8u + b;
            y[(size_t)(t0 + row) * d_inner + (size_t)h * HD_ + i] =
                acc[a][b] + dec * inter[a][b] + d_h * sx[t][i];
        }
    }
}

// The runner: passes of up to NCMAX chunks; the chunk chain state persists
// in the run slice of the per-call scratch blob between passes. `state`
// points at the segment's slot state (arena [h, S, i]); xbc/dt/y are the
// caller's base pointers, already offset (the serial walk's contract).
// Returns a NEGATIVE value if the stream-ordered alloc fails, so the
// launcher can fall back to the serial walk.
static int pd_mamba2_ssd_run(void* state, int state_f16, const void* xbc,
                             const void* dt_raw, uint32_t dt_stride,
                             const void* A, const void* D,
                             const void* dt_bias, void* y, uint32_t n_tokens,
                             uint32_t n_heads, uint32_t n_groups,
                             void* stream) {
    const cudaStream_t st = (cudaStream_t)stream;
    const uint32_t d_inner = n_heads * 64u;
    const uint32_t conv_dim = d_inner + 2u * n_groups * 128u;
    // one blob, stream-ordered: [cum | dtv | m | ds | run]
    const size_t chl = (size_t)PD_SSD_NCMAX * n_heads * PD_SSD_L;
    const size_t o_dtv = chl;
    const size_t o_m = o_dtv + chl;
    const size_t o_ds = o_m + (size_t)PD_SSD_NCMAX * n_groups * PD_SSD_L
                                * PD_SSD_L;
    const size_t o_run = o_ds + (size_t)PD_SSD_NCMAX * n_heads * 8192u;
    const size_t total = o_run + (size_t)n_heads * 8192u;
    float* blob = nullptr;
    if (cudaMallocAsync((void**)&blob, total * sizeof(float), st) !=
        cudaSuccess) {
        cudaGetLastError();  // eat it; the caller runs the serial walk
        return -1;
    }
    float* cum = blob;
    float* dtv = blob + o_dtv;
    float* m = blob + o_m;
    float* ds = blob + o_ds;
    float* run = blob + o_run;
    const uint32_t pass_cap = PD_SSD_NCMAX * PD_SSD_L;
    const uint32_t n_pass = (n_tokens + pass_cap - 1u) / pass_cap;
    for (uint32_t p = 0; p < n_pass; ++p) {
        const uint32_t t0 = p * pass_cap;
        const uint32_t rem = n_tokens - t0;
        const uint32_t ptok = rem < pass_cap ? rem : pass_cap;
        const uint32_t nc = (ptok + PD_SSD_L - 1u) / PD_SSD_L;
        pd_ssd_prep_kernel<<<dim3(n_heads, nc), PD_SSD_L, 0, st>>>(
            (const float*)dt_raw, dt_stride, t0, (const float*)A,
            (const float*)dt_bias, n_heads, ptok, cum, dtv);
        pd_ssd_gram_kernel<128u><<<dim3(4u, nc * n_groups), 128u, 0, st>>>(
            (const float*)xbc, t0, conv_dim, d_inner, n_groups, ptok, m);
        pd_ssd_dstate_kernel<128u, 64u>
            <<<dim3(n_heads, nc), 256u, 0, st>>>(
                (const float*)xbc, t0, conv_dim, d_inner, n_groups, n_heads,
                ptok, cum, dtv, ds);
        pd_ssd_chain_kernel<<<dim3(n_heads, 4u), 256u, 0, st>>>(
            state, state_f16, nc, n_heads, p == 0, p + 1u == n_pass, cum, ds,
            run);
        pd_ssd_y_kernel<128u, 64u><<<dim3(n_heads, nc, 2u), 256u, 0, st>>>(
            (const float*)xbc, t0, conv_dim, d_inner, n_groups, n_heads,
            (const float*)D, (float*)y, ptok, cum, dtv, m, ds);
    }
    cudaFreeAsync(blob, st);
    return pd_launch_status();
}
