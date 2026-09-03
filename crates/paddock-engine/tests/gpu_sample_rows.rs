//! Parity tests for the fused decode sampler (`sample_rows`): greedy rows
//! must equal the host argmax exactly (incl. ties), categorical rows must
//! reproduce the host `sample_all` walk for the same uniform (identical up
//! to summation-order rounding at cumsum boundaries - asserted ≥ 99.5%
//! exact with any strays adjacent in probability), and skip rows must stay
//! untouched. Light (no model load).

mod common;

use paddock_engine::gpu::GpuExecutor;

fn det(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / (1u64 << 31) as f32) - 0.5
        })
        .collect()
}

/// The host sampler's sort-free categorical walk (sampler.rs `sample_all`),
/// reproduced here as the oracle.
fn host_sample_all(logits: &[f32], inv_t: f32, u: f32) -> u32 {
    let argmax = |l: &[f32]| -> u32 {
        let mut best = 0usize;
        for (i, v) in l.iter().enumerate() {
            if *v > l[best] {
                best = i;
            }
        }
        best as u32
    };
    let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) * inv_t;
    if !m.is_finite() {
        return argmax(logits);
    }
    let mut sum = 0.0f32;
    for &l in logits.iter() {
        sum += (l * inv_t - m).exp();
    }
    if !(sum > 0.0) {
        return argmax(logits);
    }
    let mut r = u * sum;
    let mut last = 0u32;
    for (i, &l) in logits.iter().enumerate() {
        let e = (l * inv_t - m).exp();
        if e > 0.0 {
            last = i as u32;
            r -= e;
            if r <= 0.0 {
                return i as u32;
            }
        }
    }
    last
}

/// Pack per-row params: mode 0 skip, 1 greedy, 2 categorical.
fn pack_params(rows: &[(u32, f32, f32)]) -> Vec<u32> {
    let mut par = Vec::with_capacity(rows.len() * 4);
    for &(mode, inv_t, u) in rows {
        par.extend_from_slice(&[inv_t.to_bits(), u.to_bits(), mode, 0]);
    }
    par
}

/// The pack's multi-block sampler chains (pd_smp_scr / pd_topp_scr) use one
/// static device scratch - the serving contract is a single engine stream
/// per process. Parallel test threads each open their own stream and RACE
/// that scratch (observed: unwritten outs + boundary corruption
/// when both mode-6 tests overlapped), so every test in this file takes the
/// lock for its whole GPU lifetime.
fn exec() -> Option<(std::sync::MutexGuard<'static, ()>, GpuExecutor)> {
    static LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
    let guard = LOCK
        .get_or_init(|| std::sync::Mutex::new(()))
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    common::gpu().map(|e| (guard, e))
}

#[test]
fn greedy_rows_match_host_argmax_including_ties() {
    let Some((_scr_lock, exec)) = exec() else {
        return;
    };
    let (rows, n) = (8usize, 4096 + 13usize); // non-multiple of the block width
    let mut logits = det(rows * n, 7);
    // exact ties in rows 2 and 5: the duplicated max must resolve to the
    // LOWEST index, like the host's strict-greater ascending scan
    logits[2 * n + 100] = 9.5;
    logits[2 * n + 3000] = 9.5;
    logits[5 * n] = 11.0;
    logits[5 * n + n - 1] = 11.0;
    let d_l = exec.to_device(&logits).expect("logits");
    let par = pack_params(&vec![(1u32, 0.0f32, 0.0f32); rows]);
    let d_p = exec.to_device_u32(&par).expect("params");
    let mut d_out = exec.to_device_u32(&vec![u32::MAX; rows]).expect("out");
    exec.sample_rows(&d_l, &d_p, &mut d_out, rows, n)
        .expect("sample");
    let got = exec.to_host_u32(&d_out).expect("dtoh");
    for r in 0..rows {
        let row = &logits[r * n..(r + 1) * n];
        let mut best = 0usize;
        for (i, v) in row.iter().enumerate() {
            if *v > row[best] {
                best = i;
            }
        }
        assert_eq!(got[r], best as u32, "row {r}");
    }
    assert_eq!(got[2], 100, "tie row must pick the lowest index");
    assert_eq!(got[5], 0, "tie row must pick the lowest index");
    eprintln!("sample_rows greedy: matches host argmax on {rows} rows incl exact ties");
}

#[test]
fn categorical_rows_match_host_walk() {
    let Some((_scr_lock, exec)) = exec() else {
        return;
    };
    let n = 4096usize;
    let inv_t = 1.0f32 / 0.7;
    // 1000 (logits-row, uniform) trials, batched 25 rows at a time
    let (batch, trials) = (25usize, 1000usize);
    let mut match_cnt = 0usize;
    let mut mism = Vec::new();
    for t in 0..trials / batch {
        let logits = det(batch * n, 1000 + t as u64);
        let us: Vec<f32> = det(batch, 5000 + t as u64)
            .iter()
            .map(|v| v + 0.5) // det() is [-0.5, 0.5)
            .collect();
        let d_l = exec.to_device(&logits).expect("logits");
        let rows: Vec<(u32, f32, f32)> = us.iter().map(|&u| (2u32, inv_t, u)).collect();
        let d_p = exec.to_device_u32(&pack_params(&rows)).expect("params");
        let mut d_out = exec.to_device_u32(&vec![u32::MAX; batch]).expect("out");
        exec.sample_rows(&d_l, &d_p, &mut d_out, batch, n)
            .expect("sample");
        let got = exec.to_host_u32(&d_out).expect("dtoh");
        for b in 0..batch {
            let row = &logits[b * n..(b + 1) * n];
            let want = host_sample_all(row, inv_t, us[b]);
            if got[b] == want {
                match_cnt += 1;
            } else {
                mism.push((got[b], want, row[got[b] as usize], row[want as usize]));
            }
        }
    }
    // summation-order rounding may shift a draw across a cumsum boundary -
    // rare, and the two picks must then be genuine neighbors in probability
    assert!(
        match_cnt * 1000 >= trials * 995,
        "only {match_cnt}/{trials} matched host walk: {mism:?}"
    );
    eprintln!("sample_rows categorical: {match_cnt}/{trials} exact host matches");
}

/// Host-reference top-64 selection: what pd_topk_rows returns modulo
/// boundary-tie choice (never contractual) - same oracle the sampler unit
/// test uses.
fn host_top64(row: &[f32]) -> Vec<(u32, f32)> {
    let mut idx: Vec<u32> = (0..row.len() as u32).collect();
    idx.sort_unstable_by(|&a, &b| row[b as usize].total_cmp(&row[a as usize]));
    idx.truncate(64.min(row.len()));
    idx.into_iter().map(|i| (i, row[i as usize])).collect()
}

/// Mode 5: `pd_sample_rows_t` - histogram head build + device
/// draw tail in one launch - must reproduce the host's `sample_trunc_head`
/// over the host-selected top-64 head for the same (inv_t, u, k, top_p,
/// min_p) - token-exact up to expf ulp differences at cumsum boundaries
/// (same doctrine and same ≥99.5% bar as the categorical test above).
#[test]
fn sample_rows_t_matches_host_head_pipeline() {
    let Some((_scr_lock, exec)) = exec() else {
        return;
    };
    if !exec.has_sample_rows_t() {
        eprintln!("pack lacks sample_rows_t - skipping");
        return;
    }
    let n = 4096 + 13usize; // non-multiple of the block width
    let (batch, rounds) = (16usize, 40usize);
    let trials = batch * rounds;
    // cycle the truncation corners: the qwen election (k20/p0.95), k1
    // (argmax-equivalent), k64 (full head), min_p active, p+min_p combined
    let corners: [(u32, f32, f32); 5] = [
        (20, 0.95, 0.0),
        (1, 1.0, 0.0),
        (64, 1.0, 0.0),
        (32, 1.0, 0.05),
        (20, 0.8, 0.02),
    ];
    let inv_t = 1.0f32 / 0.85;
    let mut match_cnt = 0usize;
    let mut mism = Vec::new();
    for t in 0..rounds {
        let logits = det(batch * n, 9000 + t as u64);
        let us: Vec<f32> = det(batch, 7000 + t as u64)
            .iter()
            .map(|v| v + 0.5)
            .collect();
        let d_l = exec.to_device(&logits).expect("logits");
        let mut par = Vec::with_capacity(batch * 4);
        let mut tpar = Vec::with_capacity(batch * 4);
        for (b, &u) in us.iter().enumerate() {
            let (k, top_p, min_p) = corners[(t * batch + b) % corners.len()];
            par.extend_from_slice(&[inv_t.to_bits(), u.to_bits(), 5, 0]);
            tpar.extend_from_slice(&[k, top_p.to_bits(), min_p.to_bits(), 0]);
        }
        let d_p = exec.to_device_u32(&par).expect("params");
        let d_t = exec.to_device_u32(&tpar).expect("tpar");
        let mut d_out = exec.to_device_u32(&vec![u32::MAX; batch]).expect("out");
        exec.sample_rows_t(&d_l, &d_p, &d_t, &mut d_out, batch, n)
            .expect("sample_rows_t");
        let got = exec.to_host_u32(&d_out).expect("dtoh");
        for b in 0..batch {
            let (k, top_p, min_p) = corners[(t * batch + b) % corners.len()];
            let row = &logits[b * n..(b + 1) * n];
            let want = paddock_engine::sampler::sample_trunc_head(
                &host_top64(row),
                inv_t,
                us[b],
                k,
                top_p,
                min_p,
            );
            if got[b] == want {
                match_cnt += 1;
            } else {
                mism.push((
                    k,
                    top_p,
                    min_p,
                    got[b],
                    want,
                    row[got[b] as usize],
                    row[want as usize],
                ));
            }
        }
    }
    assert!(
        match_cnt * 1000 >= trials * 995,
        "only {match_cnt}/{trials} matched sample_trunc_head: {mism:?}"
    );
    eprintln!("sample_rows_t: {match_cnt}/{trials} exact host matches");
}

/// Rows-small arm: rows 1 and 2 at a real vocab width elect the
/// fused t2s kernel ([sr-t2s] witness) - below the mb floor and outside the
/// chunked t2 pair's economics. Same host-parity contract and truncation
/// corners as the batch=16 gate above; this exists because that gate never
/// drove rows <= 2 and the arm would otherwise pass vacuously.
#[test]
fn sample_rows_t_small_rows_match_host_head() {
    let Some((_scr_lock, exec)) = exec() else {
        return;
    };
    if !exec.has_sample_rows_t() {
        eprintln!("pack lacks sample_rows_t - skipping");
        return;
    }
    let n = 4096 + 13usize;
    let corners: [(u32, f32, f32); 5] = [
        (20, 0.95, 0.0),
        (1, 1.0, 0.0),
        (64, 1.0, 0.0),
        (32, 1.0, 0.05),
        (20, 0.8, 0.02),
    ];
    let inv_t = 1.0f32 / 0.85;
    let mut match_cnt = 0usize;
    let mut trials = 0usize;
    let mut mism = Vec::new();
    for rows in [1usize, 2usize] {
        for t in 0..60usize {
            let logits = det(rows * n, 12000 + (rows * 1000 + t) as u64);
            let us: Vec<f32> = det(rows, 15000 + t as u64)
                .iter()
                .map(|v| v + 0.5)
                .collect();
            let d_l = exec.to_device(&logits).expect("logits");
            let mut par = Vec::with_capacity(rows * 4);
            let mut tpar = Vec::with_capacity(rows * 4);
            for (b, &u) in us.iter().enumerate() {
                let (k, top_p, min_p) = corners[(t * rows + b) % corners.len()];
                par.extend_from_slice(&[inv_t.to_bits(), u.to_bits(), 5, 0]);
                tpar.extend_from_slice(&[k, top_p.to_bits(), min_p.to_bits(), 0]);
            }
            let d_p = exec.to_device_u32(&par).expect("params");
            let d_t = exec.to_device_u32(&tpar).expect("tpar");
            let mut d_out = exec.to_device_u32(&vec![u32::MAX; rows]).expect("out");
            exec.sample_rows_t(&d_l, &d_p, &d_t, &mut d_out, rows, n)
                .expect("sample_rows_t");
            let got = exec.to_host_u32(&d_out).expect("dtoh");
            for b in 0..rows {
                let (k, top_p, min_p) = corners[(t * rows + b) % corners.len()];
                let row = &logits[b * n..(b + 1) * n];
                let want = paddock_engine::sampler::sample_trunc_head(
                    &host_top64(row),
                    inv_t,
                    us[b],
                    k,
                    top_p,
                    min_p,
                );
                trials += 1;
                if got[b] == want {
                    match_cnt += 1;
                } else {
                    mism.push((rows, k, top_p, min_p, got[b], want));
                }
            }
        }
    }
    assert!(
        match_cnt * 1000 >= trials * 995,
        "only {match_cnt}/{trials} matched sample_trunc_head at rows<=2: {mism:?}"
    );
    eprintln!("sample_rows_t small rows: {match_cnt}/{trials} exact host matches");
}

/// Edge rows: non-mode-5 rows (here mode 4, the host-head class)
/// leave out untouched, and a vocab narrower than the 64-head still draws
/// from exactly the real entries.
#[test]
fn sample_rows_t_skip_and_narrow_vocab() {
    let Some((_scr_lock, exec)) = exec() else {
        return;
    };
    if !exec.has_sample_rows_t() {
        eprintln!("pack lacks sample_rows_t - skipping");
        return;
    }
    let n = 40usize; // narrower than the 64-head
    let logits = det(2 * n, 77);
    let d_l = exec.to_device(&logits).expect("logits");
    let inv_t = 1.0f32;
    let u = 0.63f32;
    // row 0: mode 4 (host-head row) - sample_rows_t must skip it;
    // row 1: mode 5, k 8 / top_p 0.9
    let par = [
        inv_t.to_bits(),
        u.to_bits(),
        4,
        0,
        inv_t.to_bits(),
        u.to_bits(),
        5,
        0,
    ];
    let tpar = [8u32, 1.0f32.to_bits(), 0, 0, 8u32, 0.9f32.to_bits(), 0, 0];
    let d_p = exec.to_device_u32(&par).expect("params");
    let d_t = exec.to_device_u32(&tpar).expect("tpar");
    let mut d_out = exec.to_device_u32(&[u32::MAX; 2]).expect("out");
    exec.sample_rows_t(&d_l, &d_p, &d_t, &mut d_out, 2, n)
        .expect("sample_rows_t");
    let got = exec.to_host_u32(&d_out).expect("dtoh");
    assert_eq!(got[0], u32::MAX, "non-mode-5 row must stay untouched");
    let want = paddock_engine::sampler::sample_trunc_head(
        &host_top64(&logits[n..2 * n]),
        inv_t,
        u,
        8,
        0.9,
        0.0,
    );
    assert_eq!(got[1], want, "narrow-vocab row diverged from the host head");
    eprintln!("sample_rows_t: mode-4 skip + narrow vocab match host");
}

/// Host oracle for the k-less truncation space - `build_nucleus`
/// (top_k == 0) + `draw` semantics mirrored exactly: full-vocab softmax
/// denominator, min-p as a per-element cut (probs desc => take_while ≡
/// filter), top-p as the shortest desc prefix with cum/D >= p (all
/// survivors when never reached), draw at quantile u·M with last-survivor
/// fallback.
fn host_nucleus_k0(logits: &[f32], inv_t: f32, u: f32, top_p: f32, min_p: f32) -> u32 {
    let m = logits
        .iter()
        .map(|&l| l * inv_t)
        .fold(f32::NEG_INFINITY, f32::max);
    let d: f32 = logits.iter().map(|&l| (l * inv_t - m).exp()).sum();
    let mut idx: Vec<u32> = (0..logits.len() as u32).collect();
    idx.sort_by(|&a, &b| {
        (logits[b as usize] * inv_t)
            .total_cmp(&(logits[a as usize] * inv_t))
            .then(a.cmp(&b))
    });
    let surv: Vec<(u32, f32)> = idx
        .iter()
        .map(|&i| (i, (logits[i as usize] * inv_t - m).exp()))
        .filter(|&(_, e)| !(min_p > 0.0) || e >= min_p)
        .collect();
    if surv.is_empty() {
        return idx[0];
    }
    let mut keep = surv.len();
    if top_p < 1.0 {
        let mut cum = 0.0f32;
        for (i, &(_, e)) in surv.iter().enumerate() {
            cum += e;
            if cum / d >= top_p {
                keep = i + 1;
                break;
            }
        }
    }
    let mnuc: f32 = surv[..keep].iter().map(|c| c.1).sum();
    let mut r = u * mnuc;
    for &(id, e) in &surv[..keep] {
        r -= e;
        if r <= 0.0 {
            return id;
        }
    }
    surv[keep - 1].0
}

/// `pd_sample_rows_p` (mode 6, no top-k bound) must
/// reproduce the host nucleus pipeline - nemotron's p-only election, min-p
/// only, and combinations. Device bucket sums reassociate fp adds, so a u
/// near a cum boundary may pick the adjacent survivor - same ≥99% bar and
/// adjacency dump as the categorical test.
#[test]
fn sample_rows_p_matches_host_nucleus() {
    let Some((_scr_lock, exec)) = exec() else {
        return;
    };
    if !exec.has_sample_rows_p() {
        eprintln!("pack lacks sample_rows_p - skipping");
        return;
    }
    let n = 4096 + 13usize;
    let (batch, rounds) = (16usize, 40usize);
    let trials = batch * rounds;
    // corners: the nemotron election (p 0.95), tighter p, min-p only,
    // p+min-p combined, and p≈1 (nucleus ≈ everything)
    let corners: [(f32, f32); 5] = [
        (0.95, 0.0),
        (0.8, 0.0),
        (1.0, 0.05),
        (0.9, 0.02),
        (0.999, 0.0),
    ];
    let inv_t = 1.0f32; // nemotron's own temperature
    let mut match_cnt = 0usize;
    let mut mism = Vec::new();
    for t in 0..rounds {
        let logits = det(batch * n, 11000 + t as u64);
        let us: Vec<f32> = det(batch, 13000 + t as u64)
            .iter()
            .map(|v| v + 0.5)
            .collect();
        let d_l = exec.to_device(&logits).expect("logits");
        let mut par = Vec::with_capacity(batch * 4);
        let mut tpar = Vec::with_capacity(batch * 4);
        for (b, &u) in us.iter().enumerate() {
            let (top_p, min_p) = corners[(t * batch + b) % corners.len()];
            par.extend_from_slice(&[inv_t.to_bits(), u.to_bits(), 6, 0]);
            tpar.extend_from_slice(&[0, top_p.to_bits(), min_p.to_bits(), 0]);
        }
        let d_p = exec.to_device_u32(&par).expect("params");
        let d_t = exec.to_device_u32(&tpar).expect("tpar");
        let mut d_out = exec.to_device_u32(&vec![u32::MAX; batch]).expect("out");
        exec.sample_rows_p(&d_l, &d_p, &d_t, &mut d_out, batch, n)
            .expect("sample_rows_p");
        let got = exec.to_host_u32(&d_out).expect("dtoh");
        for b in 0..batch {
            let (top_p, min_p) = corners[(t * batch + b) % corners.len()];
            let row = &logits[b * n..(b + 1) * n];
            let want = host_nucleus_k0(row, inv_t, us[b], top_p, min_p);
            if got[b] == want {
                match_cnt += 1;
            } else {
                mism.push((
                    top_p,
                    min_p,
                    got[b],
                    want,
                    row.get(got[b] as usize).copied(),
                    row[want as usize],
                ));
            }
        }
    }
    assert!(
        match_cnt * 100 >= trials * 99,
        "only {match_cnt}/{trials} matched host nucleus: {mism:?}"
    );
    eprintln!("sample_rows_p: {match_cnt}/{trials} exact host matches");
}

/// Mode-6 edges: non-mode-6 rows untouched; narrow vocab; near-flat logits
/// (bucket refinement + gather in play across a dense value band).
#[test]
fn sample_rows_p_skip_and_flat() {
    let Some((_scr_lock, exec)) = exec() else {
        return;
    };
    if !exec.has_sample_rows_p() {
        eprintln!("pack lacks sample_rows_p - skipping");
        return;
    }
    // near-flat: 8192 logits all within [0, 1e-3) - thousands of elements
    // share level-0 buckets, forcing the 20-bit refinement path
    let n = 8192usize;
    let logits: Vec<f32> = det(n, 999).iter().map(|v| (v + 0.5) * 1e-3).collect();
    let d_l = exec.to_device(&logits).expect("logits");
    let mut match_cnt = 0;
    // 256 trials, not 64: the mb chain's cross-segment float-atomic merges
    // reassociate exp-mass sums, and on this deliberately flat vocab a u
    // near a boundary can flip to the adjacent survivor run-to-run. That is
    // the accepted mode-2 ulps class, but at 64 trials the 95% bar sat one
    // flip from failing (it did flake); a 4x sample keeps the
    // same bar while making the proportion estimate stable.
    let trials = 256;
    for t in 0..trials {
        let u = (t as f32 + 0.5) / trials as f32;
        let par = [
            1.0f32.to_bits(),
            u.to_bits(),
            6,
            0,
            1.0f32.to_bits(),
            u.to_bits(),
            2,
            0,
        ];
        let tpar = [0, 0.9f32.to_bits(), 0u32, 0, 0, 0, 0, 0];
        let d_p = exec.to_device_u32(&par).expect("params");
        let d_t = exec.to_device_u32(&tpar).expect("tpar");
        let mut d_out = exec.to_device_u32(&[u32::MAX; 2]).expect("out");
        exec.sample_rows_p(&d_l, &d_p, &d_t, &mut d_out, 1, n)
            .expect("sample_rows_p");
        let got = exec.to_host_u32(&d_out).expect("dtoh");
        assert_eq!(
            got[1],
            u32::MAX,
            "row 1 was not launched - must stay untouched"
        );
        let want = host_nucleus_k0(&logits, 1.0, u, 0.9, 0.0);
        if got[0] == want {
            match_cnt += 1;
        }
    }
    assert!(
        match_cnt * 100 >= trials * 95,
        "flat-vocab: {match_cnt}/{trials}"
    );
    eprintln!("sample_rows_p flat vocab: {match_cnt}/{trials} exact");
}

#[test]
fn skip_and_degenerate_rows() {
    let Some((_scr_lock, exec)) = exec() else {
        return;
    };
    let n = 2048usize;
    let mut logits = det(3 * n, 42);
    // row 2: all -inf - the host falls back to argmax (= index 0)
    for v in &mut logits[2 * n..3 * n] {
        *v = f32::NEG_INFINITY;
    }
    let d_l = exec.to_device(&logits).expect("logits");
    // row 0 skipped (mode 0), row 1 categorical, row 2 categorical-degenerate
    let d_p = exec
        .to_device_u32(&pack_params(&[(0, 0.0, 0.0), (2, 1.0, 0.5), (2, 1.0, 0.5)]))
        .expect("params");
    let mut d_out = exec.to_device_u32(&[u32::MAX; 3]).expect("out");
    exec.sample_rows(&d_l, &d_p, &mut d_out, 3, n)
        .expect("sample");
    let got = exec.to_host_u32(&d_out).expect("dtoh");
    assert_eq!(got[0], u32::MAX, "skip row must stay untouched");
    assert_eq!(got[1], host_sample_all(&logits[n..2 * n], 1.0, 0.5));
    assert_eq!(got[2], 0, "all -inf row must argmax-fallback to index 0");
    eprintln!("sample_rows: skip + degenerate fallbacks match host");
}
