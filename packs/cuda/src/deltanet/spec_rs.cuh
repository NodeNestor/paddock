// spec_rs.cuh - canonical speculative rejection sampling.
// Textually-included segment of the single pack translation unit.
// Not standalone-compilable: include order is defined by ../pack.cu.
//
// The full-q arm of speculative verification (arXiv 2211.17192): drafts are
// SAMPLED from the drafter's softmax q (not argmax'd), verify accepts draft
// x with probability min(1, p(x)/q(x)) against the target's softcapped
// distribution p, and a rejection recovers from the residual max(p-q, 0).
// Lossless by construction - the committed stream is distributed exactly as
// target sampling. Rationale: with greedy drafts both our sample-and-match
// rule and vLLM's NO_DRAFT_PROBS rejection sampler accept with probability
// p(argmax q) - identical schemes. The broad-distribution headroom
// (sum min(p,q) vs p(argmax q)) exists only with sampled drafts + real q,
// which neither engine ships. That's what these two kernels add.
//
// Exactness contract: q is the fp16-rounded row this kernel stores -
// pd_draft_rs samples from the store (not from the f32 logits), so the
// draft draw, the accept test's q(x), and the residual all read one
// distribution. qsum holds the store row's exact f32 sum in ascending-v
// thread-chunk order; the quantile walks re-accumulate in that same order
// (the pd_sample_rows invariant), so u*S never walks off the row except at
// the subtract-vs-add rounding split, which keeps the last-with-mass
// backstop. qsum == 0 marks a greedy (argmax) chain row: the resolver then
// runs the point-mass rule - accept w.p. p(x), recover from p with x masked
// (vLLM's NO_DRAFT_PROBS semantics), which is exactly our classic rule.
//
// Latency note: one 256-thread CTA per row over a 262k vocab, same class as
// the single-block pd_sample_rows path (~40-60us per chain step at 32 rows,
// ~0.1-0.2ms worst per verify resolve at 96 draft rows) - noise against a
// c32 round; split-phase A/B/C treatment is the recorded follow-up if RS
// wins its A/B and the resolve ever shows in a trace.

#define PD_RS_TPB 256u

// ---- chain-step draft sampler -------------------------------------------
// One CTA per chain row. invt[row] <= 0 => greedy argmax (classic chain,
// no store write, qsum = 0 marker). Otherwise: pass 1 block max, pass 2
// exp((x - m) * invt) rounded to fp16 into qstore[(step*rmax + row)*n + v]
// with the exact rounded sum in qsum[step*rmax + row], pass 3 inverse-CDF
// draw at u*S over the STORED halves. step comes from a device counter so
// the rr-keyed chain graphs replay k times with no per-step host patching.
__global__ void pd_draft_rs_kernel(const float* __restrict__ logits,
                                   const float* __restrict__ invt,
                                   const float* __restrict__ uplane,
                                   const unsigned int* __restrict__ step,
                                   __half* __restrict__ qstore,
                                   float* __restrict__ qsum,
                                   unsigned int* __restrict__ tok,
                                   uint32_t rows, uint32_t n, uint32_t rmax) {
    const uint32_t row = blockIdx.x, tid = threadIdx.x;
    const uint32_t s = *step;
    const float* x = logits + (size_t)row * n;
    const float it = invt[row];
    const size_t qoff = ((size_t)s * rmax + row) * n;

    // pass 1: block max + lowest-index argmax (greedy answer, softmax shift,
    // degenerate fallback) - the pd_sample_rows idiom verbatim
    float bv = -3.402823466e+38f;
    uint32_t bi = 0;
    for (uint32_t i = tid; i < n; i += PD_RS_TPB) {
        float v = x[i];
        if (v > bv) { bv = v; bi = i; }
    }
#pragma unroll
    for (uint32_t off = 16; off > 0; off >>= 1) {
        float ov = __shfl_down_sync(0xffffffffu, bv, off);
        uint32_t oi = __shfl_down_sync(0xffffffffu, bi, off);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    __shared__ float sv[PD_RS_TPB / 32u];
    __shared__ uint32_t si[PD_RS_TPB / 32u];
    const uint32_t lane = tid & 31u, warp = tid >> 5;
    if (lane == 0) { sv[warp] = bv; si[warp] = bi; }
    __syncthreads();
    __shared__ float s_max;
    __shared__ uint32_t s_arg;
    if (tid == 0) {
        for (uint32_t w = 1; w < PD_RS_TPB / 32u; ++w)
            if (sv[w] > bv || (sv[w] == bv && si[w] < bi)) { bv = sv[w]; bi = si[w]; }
        s_max = bv;
        s_arg = bi;
    }
    __syncthreads();
    if (it <= 0.0f) { // greedy chain row: no q distribution exists
        if (tid == 0) {
            tok[row] = s_arg;
            qsum[(size_t)s * rmax + row] = 0.0f;
        }
        return;
    }
    const float m = s_max * it;
    if (!isfinite(m)) { // all -inf/nan logits: argmax fallback, greedy marker
        if (tid == 0) {
            tok[row] = s_arg;
            qsum[(size_t)s * rmax + row] = 0.0f;
        }
        return;
    }

    // pass 2: store fp16 exp mass, per-thread CONTIGUOUS chunks; S = the sum
    // of the ROUNDED values in ascending-v order (chunk-major) - the exact
    // distribution the draw and the resolver both use
    const uint32_t chunk = (n + PD_RS_TPB - 1u) / PD_RS_TPB;
    const uint32_t lo = tid * chunk;
    const uint32_t hi = min(lo + chunk, n);
    float csum = 0.0f;
    for (uint32_t i = lo; i < hi; ++i) {
        const __half h = __float2half(expf(x[i] * it - m));
        qstore[qoff + i] = h;
        csum += __half2float(h);
    }
    __shared__ float ssum[PD_RS_TPB];
    ssum[tid] = csum;
    __syncthreads();

    __shared__ float s_rr;
    __shared__ uint32_t s_owner;
    if (tid == 0) {
        float total = 0.0f;
        for (uint32_t t = 0; t < PD_RS_TPB; ++t) total += ssum[t];
        qsum[(size_t)s * rmax + row] = total;
        if (!(total > 0.0f)) { // all mass underflowed even at the max: argmax
            tok[row] = s_arg;
            s_owner = PD_RS_TPB;
        } else {
            const float r = uplane[s * rows + row] * total;
            float pre = 0.0f;
            uint32_t own = PD_RS_TPB;
            float rr = 0.0f;
            for (uint32_t t = 0; t < PD_RS_TPB; ++t) {
                const float c = ssum[t];
                if (c > 0.0f && pre + c >= r) { own = t; rr = r - pre; break; }
                pre += c;
            }
            if (own == PD_RS_TPB) { // fp round-off tail: last chunk with mass
                for (uint32_t t = PD_RS_TPB; t-- > 0u;)
                    if (ssum[t] > 0.0f) { own = t; rr = ssum[t]; break; }
            }
            s_owner = own;
            s_rr = rr;
        }
    }
    __syncthreads();
    if (tid == s_owner) {
        // pass 3: draw from the store - the same rounded values pass 2
        // summed, same order, so the quantile crosses inside the chunk
        float rr = s_rr;
        uint32_t last = 0xFFFFFFFFu;
        for (uint32_t i = lo; i < hi; ++i) {
            const float e = __half2float(qstore[qoff + i]);
            if (e > 0.0f) {
                last = i;
                rr -= e;
                if (rr <= 0.0f) break;
            }
        }
        tok[row] = (last != 0xFFFFFFFFu) ? last : s_arg;
    }
}

// ---- verify resolver -----------------------------------------------------
// One CTA per drafted verify row. par is 8 u32 words per row:
//   {vrow, jstep, srow, invt f32-bits, u1 f32-bits, u2 f32-bits, pad, pad}
// Reads the draft token from the step-major chain plane (drafts[jstep*rr +
// srow] - async-safe, host placeholders never matter), q from the fp16
// store, p from the tick's softcapped logits row. Writes out[vrow] = draft
// on accept / residual pick on reject - downstream (spec_accept walk, host
// replay, service replay) consumes accept-while-match unchanged: accepted
// rows match their draft, a rejected row's residual provably differs (the
// residual mass at the draft is max(p-q,0) with u1 >= p/q => p < q => 0).
__global__ void pd_spec_rs_resolve_kernel(const float* __restrict__ logits,
                                          const unsigned int* __restrict__ drafts,
                                          const __half* __restrict__ qstore,
                                          const float* __restrict__ qsum,
                                          const unsigned int* __restrict__ par,
                                          unsigned int* __restrict__ out,
                                          uint32_t rr, uint32_t n, uint32_t rmax) {
    const uint32_t i = blockIdx.x, tid = threadIdx.x;
    const unsigned int* p8 = par + (size_t)i * 8u;
    const uint32_t vrow = p8[0], jstep = p8[1], srow = p8[2];
    const float it = __uint_as_float(p8[3]);
    const float u1 = __uint_as_float(p8[4]);
    const float u2 = __uint_as_float(p8[5]);
    const uint32_t d = drafts[(size_t)jstep * rr + srow];
    const float* x = logits + (size_t)vrow * n;
    const float qs = qsum[(size_t)jstep * rmax + srow];
    const __half* q = qstore + ((size_t)jstep * rmax + srow) * n;
    const bool point_q = !(qs > 0.0f); // greedy chain row: q = delta at d

    // pass 1: target max (+ argmax for degenerate fallbacks)
    float bv = -3.402823466e+38f;
    uint32_t bi = 0;
    for (uint32_t v = tid; v < n; v += PD_RS_TPB) {
        float xv = x[v];
        if (xv > bv) { bv = xv; bi = v; }
    }
#pragma unroll
    for (uint32_t off = 16; off > 0; off >>= 1) {
        float ov = __shfl_down_sync(0xffffffffu, bv, off);
        uint32_t oi = __shfl_down_sync(0xffffffffu, bi, off);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    __shared__ float sv[PD_RS_TPB / 32u];
    __shared__ uint32_t si[PD_RS_TPB / 32u];
    const uint32_t lane = tid & 31u, warp = tid >> 5;
    if (lane == 0) { sv[warp] = bv; si[warp] = bi; }
    __syncthreads();
    __shared__ float s_max;
    __shared__ uint32_t s_arg;
    if (tid == 0) {
        for (uint32_t w = 1; w < PD_RS_TPB / 32u; ++w)
            if (sv[w] > bv || (sv[w] == bv && si[w] < bi)) { bv = sv[w]; bi = si[w]; }
        s_max = bv;
        s_arg = bi;
    }
    __syncthreads();
    const float m = s_max * it;
    if (it <= 0.0f || !isfinite(m)) { // defensive: no distribution - greedy rule
        if (tid == 0) out[vrow] = (s_arg == d) ? d : s_arg;
        return;
    }

    // pass 2: Z (target normalizer), per-thread contiguous chunks
    const uint32_t chunk = (n + PD_RS_TPB - 1u) / PD_RS_TPB;
    const uint32_t lo = tid * chunk;
    const uint32_t hi = min(lo + chunk, n);
    float zsum = 0.0f;
    for (uint32_t v = lo; v < hi; ++v) zsum += expf(x[v] * it - m);
    __shared__ float ssum[PD_RS_TPB];
    ssum[tid] = zsum;
    __syncthreads();
    __shared__ float s_z;
    __shared__ int s_accept;
    if (tid == 0) {
        float z = 0.0f;
        for (uint32_t t = 0; t < PD_RS_TPB; ++t) z += ssum[t];
        s_z = z;
        // accept test: u1 < p(d)/q(d), div-free. point_q: q(d) = 1.
        const float pd_num = expf(x[d] * it - m); // p(d) * Z
        int acc;
        if (!(z > 0.0f)) {
            acc = (s_arg == d); // degenerate target: greedy-match rule
        } else if (point_q) {
            acc = (u1 * z < pd_num);
        } else {
            const float qd = __half2float(q[d]) / qs;
            acc = (qd > 0.0f) && (u1 * qd * z < pd_num);
        }
        s_accept = acc;
        if (acc) out[vrow] = d;
    }
    __syncthreads();
    if (s_accept) return;
    const float z = s_z;
    if (!(z > 0.0f)) { if (tid == 0) out[vrow] = s_arg; return; }

    // pass 3: residual normalizer Zr = sum max(p - q, 0), same chunk order
    const float inv_z = 1.0f / z;
    float rsum = 0.0f;
    for (uint32_t v = lo; v < hi; ++v) {
        const float pv = expf(x[v] * it - m) * inv_z;
        const float qv = point_q ? (v == d ? 1.0f : 0.0f)
                                 : __half2float(q[v]) / qs;
        const float rv = pv - qv;
        if (rv > 0.0f) rsum += rv;
    }
    ssum[tid] = rsum;
    __syncthreads();
    __shared__ float s_rr;
    __shared__ uint32_t s_owner;
    if (tid == 0) {
        float zr = 0.0f;
        for (uint32_t t = 0; t < PD_RS_TPB; ++t) zr += ssum[t];
        if (!(zr > 0.0f)) {
            // rounding dust (reject with p <= q everywhere): masked argmax -
            // never re-emit d on a reject, the walk would book it as accepted
            out[vrow] = (s_arg != d) ? s_arg : (d + 1u < n ? d + 1u : d - 1u);
            s_owner = PD_RS_TPB;
        } else {
            const float r = u2 * zr;
            float pre = 0.0f;
            uint32_t own = PD_RS_TPB;
            float rrq = 0.0f;
            for (uint32_t t = 0; t < PD_RS_TPB; ++t) {
                const float c = ssum[t];
                if (c > 0.0f && pre + c >= r) { own = t; rrq = r - pre; break; }
                pre += c;
            }
            if (own == PD_RS_TPB) {
                for (uint32_t t = PD_RS_TPB; t-- > 0u;)
                    if (ssum[t] > 0.0f) { own = t; rrq = ssum[t]; break; }
            }
            s_owner = own;
            s_rr = rrq;
        }
    }
    __syncthreads();
    if (tid == s_owner) {
        // walk the owner chunk: same residual sequence as pass 3
        float rrq = s_rr;
        uint32_t last = 0xFFFFFFFFu;
        for (uint32_t v = lo; v < hi; ++v) {
            const float pv = expf(x[v] * it - m) * inv_z;
            const float qv = point_q ? (v == d ? 1.0f : 0.0f)
                                     : __half2float(q[v]) / qs;
            const float rv = pv - qv;
            if (rv > 0.0f) {
                last = v;
                rrq -= rv;
                if (rrq <= 0.0f) break;
            }
        }
        if (last != 0xFFFFFFFFu) {
            out[vrow] = last;
        } else {
            out[vrow] = (s_arg != d) ? s_arg : (d + 1u < n ? d + 1u : d - 1u);
        }
    }
}

// ---- C launchers ---------------------------------------------------------
PD_EXPORT int pd_draft_rs(const void* logits, const void* invt,
                          const void* uplane, const void* step, void* qstore,
                          void* qsum, void* tok, uint32_t rows, uint32_t n,
                          uint32_t rmax, void* stream) {
    if (rows == 0 || n == 0) return 0;
    pd_draft_rs_kernel<<<rows, PD_RS_TPB, 0, (cudaStream_t)stream>>>(
        (const float*)logits, (const float*)invt, (const float*)uplane,
        (const unsigned int*)step, (__half*)qstore, (float*)qsum,
        (unsigned int*)tok, rows, n, rmax);
    return pd_launch_status();
}

PD_EXPORT int pd_spec_rs_resolve(const void* logits, const void* drafts,
                                 const void* qstore, const void* qsum,
                                 const void* par, void* out, uint32_t nrs,
                                 uint32_t rr, uint32_t n, uint32_t rmax,
                                 void* stream) {
    if (nrs == 0 || n == 0) return 0;
    pd_spec_rs_resolve_kernel<<<nrs, PD_RS_TPB, 0, (cudaStream_t)stream>>>(
        (const float*)logits, (const unsigned int*)drafts,
        (const __half*)qstore, (const float*)qsum, (const unsigned int*)par,
        (unsigned int*)out, rr, n, rmax);
    return pd_launch_status();
}
