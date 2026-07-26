//! Artifact detection — port of pydynamo `artifacts.py::detect_artifacts`
//! (and MATLAB `toolbox/helper_functions/artifact_detection/detect_artifacts.m`)
//! with default parameters.
//!
//! Two-band detection: broad-band (`bb_pass`=0.1 Hz HP) + high-frequency
//! (`hf_pass`=35 Hz HP). Per band:
//!   1. Cheby-I order-4 IIR highpass (0.2 dB ripple)
//!   2. abs(hilbert(·)) → envelope
//!   3. movmean over `smooth_duration`
//!   4. log
//!   5. movmedian over `detrend_duration` → subtract (detrend)
//!   6. iterative robust z-score (median + MAD) until convergence
//!
//! **Slope test is NOT ported.** The MATLAB / pydynamo `slope_test=True`
//! branch needs a multitaper spectrogram; deferred until Part 5 step 7
//! (multitaper_rs pull-in). `detect_artifacts_default` here sets
//! `slope_test=false`; the caller can OR in a slope mask computed elsewhere.
//!
//! Bit-equivalence target: the band-based path must agree with pydynamo
//! when pydynamo is invoked with `slope_test=False`.

use crate::filter_design::cheby1_sos;
use crate::signal::{hilbert, sosfiltfilt};

/// Options for `detect_artifacts`. Matches pydynamo defaults.
#[derive(Debug, Clone)]
pub struct ArtifactOpts {
    pub hf_pass: f64,
    pub hf_crit: f64,
    pub bb_pass: f64,
    pub bb_crit: f64,
    pub hf_detrend: bool,
    pub bb_detrend: bool,
    pub smooth_duration: f64,
    pub detrend_duration: f64,
    pub buffer_duration: f64,
    /// "robust" (median + MAD, pydynamo default) or "standard" (mean + std).
    pub zscore_method: ZScoreMethod,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZScoreMethod {
    Robust,
    Standard,
}

impl Default for ArtifactOpts {
    fn default() -> Self {
        Self {
            hf_pass: 35.0,
            hf_crit: 5.5,
            bb_pass: 0.1,
            bb_crit: 5.5,
            hf_detrend: true,
            bb_detrend: true,
            smooth_duration: 2.0,
            detrend_duration: 300.0,
            buffer_duration: 0.0,
            zscore_method: ZScoreMethod::Robust,
        }
    }
}

/// True where `data` sits in a run of ≥ `min_run` identical values.
/// Port of pydynamo `_flat_mask`.
pub fn flat_run_mask(data: &[f64], min_run: usize) -> Vec<bool> {
    flat_run_mask_with_tolerance(data, min_run, 0.0)
}

/// Return a mask marking runs of at least `min_run` samples whose full value
/// span (`max - min`) does not exceed `tolerance`.
///
/// Measuring the full span prevents a gradual drift from being joined into
/// one run merely because each adjacent step is small. A non-positive
/// `tolerance` uses strict equality, matching [`flat_run_mask`]. With a
/// positive tolerance, non-finite samples terminate the current run.
pub fn flat_run_mask_with_tolerance(data: &[f64], min_run: usize, tolerance: f64) -> Vec<bool> {
    let n = data.len();
    if n == 0 {
        return vec![];
    }
    let mut mask = vec![false; n];
    let mut run_start = 0usize;
    let mut run_min = data[0];
    let mut run_max = data[0];

    for i in 1..=n {
        let same = if i == n {
            false
        } else if tolerance > 0.0 {
            if data[i].is_finite() && run_min.is_finite() {
                let next_min = run_min.min(data[i]);
                let next_max = run_max.max(data[i]);
                next_max - next_min <= tolerance
            } else {
                false
            }
        } else {
            data[i] == data[run_start]
        };

        if same {
            run_min = run_min.min(data[i]);
            run_max = run_max.max(data[i]);
        } else {
            if i - run_start >= min_run {
                mask[run_start..i].fill(true);
            }
            run_start = i;
            if i < n {
                run_min = data[i];
                run_max = data[i];
            }
        }
    }
    mask
}

/// MATLAB `movmean(x, win)` — centered moving mean with shrinking window
/// at the endpoints (partial means on the edges).
fn movmean(x: &[f64], win: usize) -> Vec<f64> {
    let n = x.len();
    if win <= 1 || n == 0 {
        return x.to_vec();
    }
    let half_l = (win - 1) / 2;
    let half_r = win / 2;
    let mut csum = vec![0.0_f64; n + 1];
    for i in 0..n {
        csum[i + 1] = csum[i] + x[i];
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let a = i.saturating_sub(half_l);
        let b = (i + half_r + 1).min(n);
        out.push((csum[b] - csum[a]) / (b - a) as f64);
    }
    out
}

/// MATLAB `movmedian(x, win)` — centered moving median with shrinking
/// window at the endpoints.
///
/// Implementation: two-heap with lazy deletion (max-heap of the lower
/// half + min-heap of the upper half, rebalanced after every
/// insert/remove). All operations are O(log W) where W is the window
/// size, with a ~300 KB working set for W ≈ 40 000 — fits in L2 cache,
/// which is the dominant performance win vs the previous Fenwick-BIT-
/// over-the-full-n implementation (BIT size N = 5 M ⇒ ~40 MB working
/// set, mostly cache-missing).
///
/// Tested for bit-identity against the Fenwick reference on random
/// inputs (see `movmedian_two_heap_matches_fenwick`).
fn movmedian(x: &[f64], win: usize) -> Vec<f64> {
    movmedian_two_heap(x, win)
}

/// Two-heap sliding median with lazy deletion. O(n log W).
fn movmedian_two_heap(x: &[f64], win: usize) -> Vec<f64> {
    use std::cmp::Reverse;
    use std::collections::{BinaryHeap, HashMap};

    /// Total-ordering wrapper around f64. Uses `partial_cmp` and falls
    /// back to `Equal` for NaN — matches the Fenwick reference impl.
    #[derive(Copy, Clone, Debug)]
    struct Of(f64);
    impl PartialEq for Of { fn eq(&self, o: &Self) -> bool { self.0.to_bits() == o.0.to_bits() } }
    impl Eq for Of {}
    impl PartialOrd for Of {
        fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(o))
        }
    }
    impl Ord for Of {
        fn cmp(&self, o: &Self) -> std::cmp::Ordering {
            self.0.partial_cmp(&o.0).unwrap_or(std::cmp::Ordering::Equal)
        }
    }

    let n = x.len();
    if win <= 1 || n == 0 {
        return x.to_vec();
    }
    let half_l = (win - 1) / 2;
    let half_r = win / 2;

    // lo: max-heap (top is largest of the small half).
    // hi: min-heap (top is smallest of the large half).
    let mut lo: BinaryHeap<Of> = BinaryHeap::new();
    let mut hi: BinaryHeap<Reverse<Of>> = BinaryHeap::new();
    // Lazy-delete counts keyed by f64 bit pattern.
    let mut del_lo: HashMap<u64, u32> = HashMap::new();
    let mut del_hi: HashMap<u64, u32> = HashMap::new();
    // Effective sizes (excluding pending deletions).
    let mut n_lo: usize = 0;
    let mut n_hi: usize = 0;

    // Pop any tops that have been marked for lazy deletion.
    let prune_lo = |lo: &mut BinaryHeap<Of>, del_lo: &mut HashMap<u64, u32>| {
        while let Some(&Of(top)) = lo.peek() {
            let bits = top.to_bits();
            match del_lo.get_mut(&bits) {
                Some(c) if *c > 0 => {
                    *c -= 1;
                    if *c == 0 { del_lo.remove(&bits); }
                    lo.pop();
                }
                _ => break,
            }
        }
    };
    let prune_hi = |hi: &mut BinaryHeap<Reverse<Of>>, del_hi: &mut HashMap<u64, u32>| {
        while let Some(&Reverse(Of(top))) = hi.peek() {
            let bits = top.to_bits();
            match del_hi.get_mut(&bits) {
                Some(c) if *c > 0 => {
                    *c -= 1;
                    if *c == 0 { del_hi.remove(&bits); }
                    hi.pop();
                }
                _ => break,
            }
        }
    };

    let mut out = Vec::with_capacity(n);
    let mut a: usize = 0;
    let mut b: usize = 0;

    for i in 0..n {
        let new_a = i.saturating_sub(half_l);
        let new_b = (i + half_r + 1).min(n);

        // Add newly-arriving samples on the right.
        while b < new_b {
            let v = x[b];
            // Decide which heap on the basis of the current lo-top.
            let goes_lo = match lo.peek() {
                None => true,
                Some(&Of(top)) => Of(v).cmp(&Of(top)) != std::cmp::Ordering::Greater,
            };
            if goes_lo {
                lo.push(Of(v));
                n_lo += 1;
            } else {
                hi.push(Reverse(Of(v)));
                n_hi += 1;
            }
            b += 1;
        }

        // Remove departed samples on the left (lazy).
        while a < new_a {
            let v = x[a];
            // Decide which heap by comparing to lo-top.
            let on_lo = match lo.peek() {
                Some(&Of(top)) => Of(v).cmp(&Of(top)) != std::cmp::Ordering::Greater,
                None => false,
            };
            if on_lo {
                *del_lo.entry(v.to_bits()).or_default() += 1;
                n_lo -= 1;
            } else {
                *del_hi.entry(v.to_bits()).or_default() += 1;
                n_hi -= 1;
            }
            a += 1;
        }

        // Rebalance: keep n_lo ∈ {n_hi, n_hi + 1}. Lazy-delete the tops
        // we're going to move so we never carry stale entries across.
        loop {
            prune_lo(&mut lo, &mut del_lo);
            prune_hi(&mut hi, &mut del_hi);
            if n_lo > n_hi + 1 {
                if let Some(Of(top)) = lo.pop() {
                    hi.push(Reverse(Of(top)));
                    n_lo -= 1;
                    n_hi += 1;
                } else { break; }
            } else if n_hi > n_lo {
                if let Some(Reverse(Of(top))) = hi.pop() {
                    lo.push(Of(top));
                    n_hi -= 1;
                    n_lo += 1;
                } else { break; }
            } else {
                break;
            }
        }
        prune_lo(&mut lo, &mut del_lo);
        prune_hi(&mut hi, &mut del_hi);

        let m = new_b - new_a;
        debug_assert_eq!(n_lo + n_hi, m);
        let med = if m % 2 == 1 {
            // Odd: median is lo.top().
            lo.peek().map(|&Of(v)| v).unwrap_or(f64::NAN)
        } else {
            // Even: avg of lo.top() and hi.top().
            let l = lo.peek().map(|&Of(v)| v).unwrap_or(f64::NAN);
            let h = hi.peek().map(|&Reverse(Of(v))| v).unwrap_or(f64::NAN);
            0.5 * (l + h)
        };
        out.push(med);
    }
    out
}

#[cfg(test)]
#[allow(dead_code)]
/// Original Fenwick-BIT implementation, kept as a regression reference
/// to assert bit-identity against the new two-heap impl. O(n log n)
/// total but ~3× slower on a 5 M-sample 30 s window-detrend benchmark
/// because the BIT is sized N and doesn't fit in cache.
fn movmedian_fenwick(x: &[f64], win: usize) -> Vec<f64> {
    let n = x.len();
    if win <= 1 || n == 0 {
        return x.to_vec();
    }
    let half_l = (win - 1) / 2;
    let half_r = win / 2;

    // 1) Rank-sort x: rank[i] = i's position in sorted order.
    //    sorted_vals[r] gives the r-th smallest value.
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| {
        x[a].partial_cmp(&x[b])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    let mut rank = vec![0usize; n];
    for (r, &i) in idx.iter().enumerate() {
        rank[i] = r;
    }
    let sorted_vals: Vec<f64> = idx.iter().map(|&i| x[i]).collect();

    // 2) Fenwick tree over [0, n] (1-indexed). `bit[k]` counts how many
    //    live indices sit at ranks ≤ k.
    let mut bit = vec![0i32; n + 1];
    fn bit_add(bit: &mut [i32], mut i: usize, delta: i32) {
        while i < bit.len() {
            bit[i] += delta;
            i += i & i.wrapping_neg();
        }
    }
    // `select(k)`: return the smallest 1-based rank whose prefix-sum ≥ k.
    fn bit_select(bit: &[i32], k_target: i32) -> usize {
        let n = bit.len() - 1;
        let mut pos = 0_usize;
        let mut k = k_target;
        let mut step = 1_usize;
        while step * 2 <= n {
            step *= 2;
        }
        while step > 0 {
            let next = pos + step;
            if next <= n && bit[next] < k {
                pos = next;
                k -= bit[next];
            }
            step >>= 1;
        }
        pos + 1
    }

    let mut out = Vec::with_capacity(n);

    // 3) Seed window for i = 0.
    let init_a = 0_usize;
    let init_b = (half_r + 1).min(n);
    for j in init_a..init_b {
        bit_add(&mut bit, rank[j] + 1, 1);
    }
    let mut a = init_a;
    let mut b = init_b;

    for i in 0..n {
        let new_a = i.saturating_sub(half_l);
        let new_b = (i + half_r + 1).min(n);
        // Add newly-arriving samples on the right.
        while b < new_b {
            bit_add(&mut bit, rank[b] + 1, 1);
            b += 1;
        }
        // Remove departed samples on the left.
        while a < new_a {
            bit_add(&mut bit, rank[a] + 1, -1);
            a += 1;
        }
        let m = (new_b - new_a) as i32;
        let med = if m % 2 == 1 {
            sorted_vals[bit_select(&bit, m / 2 + 1) - 1]
        } else {
            let lo = sorted_vals[bit_select(&bit, m / 2) - 1];
            let hi = sorted_vals[bit_select(&bit, m / 2 + 1) - 1];
            0.5 * (lo + hi)
        };
        out.push(med);
    }
    out
}

/// Median of finite values. Returns NaN for all-NaN input.
fn median(xs: &[f64]) -> f64 {
    let mut v: Vec<f64> = xs.iter().filter(|x| x.is_finite()).copied().collect();
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let m = v.len();
    if m % 2 == 1 {
        v[m / 2]
    } else {
        0.5 * (v[m / 2 - 1] + v[m / 2])
    }
}

/// MATLAB default `mad(x)`: mean absolute deviation from the mean.
/// (Not median absolute deviation! pydynamo matches.)
fn mad_meanabs(xs: &[f64]) -> f64 {
    let finite: Vec<f64> = xs.iter().filter(|x| x.is_finite()).copied().collect();
    if finite.is_empty() {
        return f64::NAN;
    }
    let mean_ = finite.iter().sum::<f64>() / finite.len() as f64;
    finite.iter().map(|&v| (v - mean_).abs()).sum::<f64>() / finite.len() as f64
}

fn mean_std(xs: &[f64]) -> (f64, f64) {
    let finite: Vec<f64> = xs.iter().filter(|x| x.is_finite()).copied().collect();
    if finite.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    let n = finite.len() as f64;
    let mean_ = finite.iter().sum::<f64>() / n;
    if finite.len() < 2 {
        return (mean_, f64::NAN);
    }
    let var_ = finite.iter().map(|&v| (v - mean_).powi(2)).sum::<f64>() / (n - 1.0);
    (mean_, var_.sqrt())
}

fn center_scale(xs: &[f64], method: ZScoreMethod) -> (f64, f64) {
    match method {
        ZScoreMethod::Robust => (median(xs), mad_meanabs(xs)),
        ZScoreMethod::Standard => mean_std(xs),
    }
}

/// Iteratively flag samples where |z| > crit, recomputing centering stats on
/// the shrinking good set until convergence. Returns a bad-sample mask
/// including the input `bad`.
///
/// Implementation:
/// * Reuse a single `good_vals` buffer across iterations — saves ~5 MB
///   alloc+free per iteration on a 5 M-sample series.
/// * Use `select_nth_unstable_by` for the median instead of a full sort.
///   O(n) average vs O(n log n). For n ≈ 4.6 M this is the dominant
///   win — saves ~150-250 ms per iteration on the BB band.
/// * For even-n medians, after the partition pass `buf[..k]` contains
///   the k smallest values (unordered); the max of that range is the
///   (k-1)-th order statistic. Saves the second select call.
/// * Bit-identical to the previous "filter → sort → index" path because
///   `select_nth_unstable` returns the same scalar value at index `k`
///   as a stable sort would, given identical inputs and tie-breaking
///   on the values themselves rather than indices (we never reuse the
///   ordering, only the values).
pub fn iterative_robust_zscore(
    y: &[f64],
    bad: &[bool],
    crit: f64,
    method: ZScoreMethod,
) -> Vec<bool> {
    assert_eq!(y.len(), bad.len());
    let mut mask = bad.to_vec();
    let mut buf: Vec<f64> = Vec::with_capacity(y.len());
    loop {
        // Refill the good-values buffer in place.
        buf.clear();
        for (v, &m) in y.iter().zip(mask.iter()) {
            if !m && v.is_finite() {
                buf.push(*v);
            }
        }
        if buf.is_empty() {
            return mask;
        }
        let (mid, scale) = match method {
            ZScoreMethod::Robust => {
                // O(n) median via partition + (for even n) max over the
                // lower half. Matches the prior `median()`'s output bit-
                // for-bit.
                let n = buf.len();
                let k = n / 2;
                let cmp = |a: &f64, b: &f64| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal);
                let mid_val = *buf.select_nth_unstable_by(k, cmp).1;
                let med = if n % 2 == 1 {
                    mid_val
                } else {
                    // buf[..k] holds the k smallest values (unordered).
                    // The (k-1)-th order statistic is their max.
                    let lower_max = buf[..k].iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    0.5 * (lower_max + mid_val)
                };
                // MAD = mean absolute deviation from the mean (not from
                // the median — matches MATLAB default `mad(x)`).
                let mean_ = buf.iter().sum::<f64>() / n as f64;
                let mad = buf.iter().map(|&v| (v - mean_).abs()).sum::<f64>() / n as f64;
                (med, mad)
            }
            ZScoreMethod::Standard => {
                let n = buf.len() as f64;
                let mean_ = buf.iter().sum::<f64>() / n;
                if buf.len() < 2 {
                    (mean_, f64::NAN)
                } else {
                    let var_ = buf.iter().map(|&v| (v - mean_).powi(2)).sum::<f64>() / (n - 1.0);
                    (mean_, var_.sqrt())
                }
            }
        };
        if !scale.is_finite() || scale == 0.0 {
            return mask;
        }
        let mut any_over = false;
        for i in 0..y.len() {
            if !mask[i] && y[i].is_finite() {
                let z = (y[i] - mid) / scale;
                if z.abs() > crit {
                    mask[i] = true;
                    any_over = true;
                }
            }
        }
        if !any_over {
            return mask;
        }
    }
}

/// Flag samples more than `outlier_scalar` SD from the mean of the currently-
/// good samples, returning the union with `bad`.
fn find_outlier_noise(data: &[f64], bad: &[bool], outlier_scalar: f64) -> Vec<bool> {
    let mut out = bad.to_vec();
    let good_vals: Vec<f64> = data
        .iter()
        .zip(bad.iter())
        .filter_map(|(v, &m)| if !m && v.is_finite() { Some(*v) } else { None })
        .collect();
    if good_vals.is_empty() {
        return out;
    }
    let n = good_vals.len() as f64;
    let mean_ = good_vals.iter().sum::<f64>() / n;
    if good_vals.len() < 2 {
        return out;
    }
    let var_ = good_vals.iter().map(|&v| (v - mean_).powi(2)).sum::<f64>() / (n - 1.0);
    let sd = var_.sqrt();
    let lo = mean_ - outlier_scalar * sd;
    let hi = mean_ + outlier_scalar * sd;
    for i in 0..data.len() {
        if data[i] <= lo || data[i] >= hi {
            out[i] = true;
        }
    }
    out
}

/// Linear interp over bad samples using the nearest good values on either side
/// (padded with good-endpoint values). Replaces pydynamo
/// `detect_artifacts` lines 341-350.
fn interp_bad_samples(data: &[f64], bad: &[bool]) -> Vec<f64> {
    let n = data.len();
    let mut fixed = data.to_vec();
    let good_idx: Vec<usize> = (0..n).filter(|&i| !bad[i]).collect();
    if good_idx.is_empty() {
        return fixed;
    }
    let good_vals: Vec<f64> = good_idx.iter().map(|&i| data[i]).collect();
    let xp: Vec<f64> = std::iter::once(-1.0)
        .chain(good_idx.iter().map(|&i| i as f64))
        .chain(std::iter::once(n as f64))
        .collect();
    let fp: Vec<f64> = std::iter::once(*good_vals.first().unwrap())
        .chain(good_vals.iter().copied())
        .chain(std::iter::once(*good_vals.last().unwrap()))
        .collect();

    for i in 0..n {
        if bad[i] {
            let q = i as f64;
            // Binary search: find j s.t. xp[j-1] <= q < xp[j]
            let j = xp.partition_point(|&t| t <= q);
            let t0 = xp[j - 1];
            let t1 = xp[j];
            let y0 = fp[j - 1];
            let y1 = fp[j];
            fixed[i] = y0 + (q - t0) / (t1 - t0) * (y1 - y0);
        }
    }
    fixed
}

/// Detect artifacts in one frequency band.
fn compute_band_artifacts(
    data: &[f64],
    fs: f64,
    passband: f64,
    crit: f64,
    bad: &[bool],
    smooth_duration: f64,
    detrend_duration: f64,
    detrend_on: bool,
    zscore_method: ZScoreMethod,
) -> Vec<bool> {
    // Sub-step timing — log at `info` level so it surfaces in the
    // Tauri stderr stream with default `RUST_LOG=info`. Lets us see
    // where the per-band budget actually goes (filter, Hilbert, smooth,
    // z-score) on a real recording.
    let band_tag = format!("band(pass={:.2}Hz, crit={:.2}, detrend={})", passband, crit, detrend_on);
    let t_band = std::time::Instant::now();

    // Cheby-I order-4 highpass SOS with 0.2 dB ripple.
    let t = std::time::Instant::now();
    let sections = cheby1_sos(4, 0.2, &[passband], "highpass", fs);
    let sos: Vec<[f64; 6]> = sections.into_iter().collect();
    let t_design_ms = t.elapsed().as_secs_f64() * 1e3;

    let t = std::time::Instant::now();
    let filtered = sosfiltfilt(&sos, data);
    let t_filter_ms = t.elapsed().as_secs_f64() * 1e3;

    let t = std::time::Instant::now();
    let (re, im) = hilbert(&filtered);
    // abs(analytic) = sqrt(re^2 + im^2). pydynamo uses np.abs(scipy.signal.hilbert(...)).
    let env: Vec<f64> = re
        .iter()
        .zip(im.iter())
        .map(|(r, i)| (r * r + i * i).sqrt())
        .collect();
    let t_hilbert_ms = t.elapsed().as_secs_f64() * 1e3;

    let t = std::time::Instant::now();
    let win_smooth = (smooth_duration * fs).round().max(1.0) as usize;
    let mut y = movmean(&env, win_smooth);
    for v in y.iter_mut() {
        *v = if *v > 0.0 { v.ln() } else { f64::NAN };
    }
    let t_smooth_ms = t.elapsed().as_secs_f64() * 1e3;

    let t = std::time::Instant::now();
    if detrend_on {
        let win_det = (detrend_duration * fs).round().max(1.0) as usize;
        let med = movmedian(&y, win_det);
        for i in 0..y.len() {
            y[i] -= med[i];
        }
    }
    let t_detrend_ms = t.elapsed().as_secs_f64() * 1e3;

    let t = std::time::Instant::now();
    let out = iterative_robust_zscore(&y, bad, crit, zscore_method);
    let t_zscore_ms = t.elapsed().as_secs_f64() * 1e3;

    let t_total_ms = t_band.elapsed().as_secs_f64() * 1e3;
    // Gated on env var so production runs aren't spammed. Set
    // `DYNAMO_ARTIFACT_TIMING=1` (or `=verbose`) before launching to
    // see per-band breakdowns on stderr. No `tracing` dep on
    // `dynamo_rs` so we use plain `eprintln!`.
    if std::env::var("DYNAMO_ARTIFACT_TIMING").is_ok() {
        eprintln!(
            "[artifact_timing] {} n={} fs={} total={:.1}ms  design={:.2}ms filter={:.1}ms hilbert={:.1}ms smooth={:.1}ms detrend={:.1}ms zscore={:.1}ms",
            band_tag, data.len(), fs, t_total_ms,
            t_design_ms, t_filter_ms, t_hilbert_ms,
            t_smooth_ms, t_detrend_ms, t_zscore_ms,
        );
    }
    out
}

/// Morphological dilation of a bool mask by `k` samples on each side.
fn binary_dilate(mask: &[bool], k: usize) -> Vec<bool> {
    let n = mask.len();
    if n == 0 || k == 0 {
        return mask.to_vec();
    }
    let mut out = vec![false; n];
    // Find runs of True; expand each run by k on both sides.
    let mut i = 0;
    while i < n {
        if mask[i] {
            let start = i.saturating_sub(k);
            let mut j = i;
            while j < n && mask[j] {
                j += 1;
            }
            let end = (j + k).min(n);
            for p in start..end {
                out[p] = true;
            }
            i = j;
        } else {
            i += 1;
        }
    }
    out
}

/// Detect artifacts in an EEG time series. Two-band (BB + HF) detection + flat-
/// run + outlier-noise flagging. **Does NOT run the slope test** — the caller
/// must OR in any slope-based mask computed via multitaper elsewhere.
pub fn detect_artifacts(data: &[f64], fs: f64, opts: &ArtifactOpts) -> Vec<bool> {
    let n = data.len();
    if n == 0 {
        return vec![];
    }

    // Flat runs of ≥ 1 s. Resampling leaves small ringing in genuinely
    // disconnected segments, so match MATLAB's tolerance of 2% of the finite
    // signal's sample standard deviation instead of requiring exact equality.
    let (_, sample_std) = mean_std(data);
    let flat_tol = 0.02 * sample_std;
    let flat = flat_run_mask_with_tolerance(
        data,
        fs.round().max(1.0) as usize,
        flat_tol,
    );
    let mut bad: Vec<bool> = (0..n)
        .map(|i| !data[i].is_finite() || flat[i])
        .collect();
    bad = find_outlier_noise(data, &bad, 10.0);

    // Interp through bad samples so filters don't see NaN / flats.
    let data_fixed = interp_bad_samples(data, &bad);

    // Run the BB and HF bands concurrently — they're independent
    // (different filter design, separate copy of `data_fixed`), and
    // each one is sequential internally, so `rayon::join` halves the
    // wall-clock for this phase on any machine with ≥2 cores. The
    // sequential fallback (when the `parallel` feature is off, e.g.
    // WASM) is bit-identical because each band is a deterministic
    // function of `data_fixed` + `bad`.
    let (hf_art, bb_art) = {
        let band_hf = (opts.hf_pass, opts.hf_crit, opts.hf_detrend);
        let band_bb = (opts.bb_pass, opts.bb_crit, opts.bb_detrend);
        let smooth = opts.smooth_duration;
        let detrend = opts.detrend_duration;
        let zsm = opts.zscore_method;
        let data_ref = &data_fixed;
        let bad_ref = &bad;
        let hf = move || {
            compute_band_artifacts(
                data_ref, fs, band_hf.0, band_hf.1, bad_ref,
                smooth, detrend, band_hf.2, zsm,
            )
        };
        let bb = move || {
            compute_band_artifacts(
                data_ref, fs, band_bb.0, band_bb.1, bad_ref,
                smooth, detrend, band_bb.2, zsm,
            )
        };
        #[cfg(feature = "parallel")]
        { rayon::join(hf, bb) }
        #[cfg(not(feature = "parallel"))]
        { (hf(), bb()) }
    };
    let mut artifacts: Vec<bool> = (0..n)
        .map(|i| hf_art[i] || bb_art[i] || bad[i])
        .collect();

    if opts.buffer_duration > 0.0 {
        let k = (opts.buffer_duration * fs).round() as usize;
        if k > 0 {
            artifacts = binary_dilate(&artifacts, k);
        }
    }
    artifacts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_run_detection() {
        let x = vec![1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 3.0];
        // min_run=3 → the first run (4 × 1.0) and last (3 × 3.0) are flat.
        let m = flat_run_mask(&x, 3);
        assert_eq!(
            m,
            vec![true, true, true, true, false, false, true, true, true]
        );
    }

    #[test]
    fn flat_run_tolerance_uses_full_run_span() {
        // Adjacent differences are within tolerance, but the full span is not.
        // MATLAB get_chunks therefore splits this sequence.
        let data = [0.0, 0.009, 0.018];

        let mask = flat_run_mask_with_tolerance(&data, 3, 0.01);

        assert!(!mask.iter().any(|&value| value));
    }

    #[test]
    fn flat_run_tolerance_recovers_resampled_disconnection() {
        let mut data: Vec<f64> = (0..2_000)
            .map(|i| {
                let x = i as f64;
                (0.37 * x).sin() + 0.5 * (0.11 * x).cos()
            })
            .collect();
        let (_, sample_std) = mean_std(&data);
        let flat_tol = 0.02 * sample_std;
        assert!(
            !flat_run_mask_with_tolerance(&data, 100, flat_tol)
                .iter()
                .any(|&value| value),
            "ordinary varying EEG must not be marked flat"
        );

        for (offset, value) in data[800..950].iter_mut().enumerate() {
            *value = -0.005 + 0.01 * offset as f64 / 149.0;
        }
        let (_, sample_std) = mean_std(&data);
        let mask = flat_run_mask_with_tolerance(&data, 100, 0.02 * sample_std);

        assert!(mask[800..950].iter().all(|&value| value));
        assert!(!mask[..500].iter().any(|&value| value));
        assert!(!mask[1_200..].iter().any(|&value| value));

        let artifacts = detect_artifacts(&data, 100.0, &ArtifactOpts::default());
        assert!(artifacts[800..950].iter().all(|&value| value));
    }

    #[test]
    fn movmean_shrinking_window() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let m = movmean(&x, 3);
        // edges partial: [avg(1,2), avg(1,2,3), avg(2,3,4), avg(3,4,5), avg(4,5)]
        assert!((m[0] - 1.5).abs() < 1e-12);
        assert!((m[1] - 2.0).abs() < 1e-12);
        assert!((m[2] - 3.0).abs() < 1e-12);
        assert!((m[3] - 4.0).abs() < 1e-12);
        assert!((m[4] - 4.5).abs() < 1e-12);
    }

    #[test]
    fn movmedian_two_heap_matches_fenwick() {
        // Bit-identity smoke test across a variety of (n, win) sizes
        // and value distributions, against the kept-in-the-source-tree
        // Fenwick-BIT reference implementation. Uses an inline xorshift
        // PRNG so we don't drag in a `rand` dev-dep.
        let mut state: u64 = 0xDEADBEEF;
        let mut next = || -> f64 {
            state ^= state << 13; state ^= state >> 7; state ^= state << 17;
            (state >> 11) as f64 / ((1u64 << 53) as f64)  // in [0, 1)
        };
        let cases: &[(usize, usize)] = &[
            (1, 1), (1, 5), (2, 5),
            (10, 3), (10, 5), (10, 11),
            (100, 7), (100, 33),
            (1024, 51), (1024, 257),
            (10_000, 301), (10_000, 1001),
        ];
        for &(n, win) in cases {
            // Mix: gaussian-ish + occasional duplicates to stress
            // lazy-delete bookkeeping.
            let x: Vec<f64> = (0..n).map(|_| {
                if next() < 0.10 { 0.0 } else { next() * 100.0 - 50.0 }
            }).collect();
            let new_ = movmedian_two_heap(&x, win);
            let ref_ = movmedian_fenwick(&x, win);
            assert_eq!(new_.len(), ref_.len(), "(n={}, win={}) length mismatch", n, win);
            for (i, (a, b)) in new_.iter().zip(ref_.iter()).enumerate() {
                // Use bit-equality so duplicate-median ties are
                // identical (median is 0.5*(lo + hi) which is exact).
                assert_eq!(
                    a.to_bits(), b.to_bits(),
                    "(n={}, win={}) mismatch at i={}: two_heap={} fenwick={}",
                    n, win, i, a, b
                );
            }
        }
    }

    #[test]
    fn movmedian_shrinking_window() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let m = movmedian(&x, 3);
        // edge: median(1,2)=1.5; center: median(1,2,3)=2, etc.
        assert!((m[0] - 1.5).abs() < 1e-12);
        assert!((m[1] - 2.0).abs() < 1e-12);
        assert!((m[2] - 3.0).abs() < 1e-12);
        assert!((m[3] - 4.0).abs() < 1e-12);
        assert!((m[4] - 4.5).abs() < 1e-12);
    }

    #[test]
    fn parallel_band_dispatch_is_bit_identical_to_sequential() {
        // Build a 5-minute synthetic 100 Hz signal with embedded
        // high-frequency bursts (band-HF artifacts) and a flat run.
        // Run detect_artifacts twice — once via the public API (which
        // uses rayon::join when the `parallel` feature is on), and
        // once via direct sequential band calls. Both must produce
        // bit-identical masks because each band is a pure function of
        // its inputs.
        let fs = 100.0;
        let n = 30_000usize; // 5 min at 100 Hz
        let mut data: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / fs;
                // 10 Hz background + occasional 40 Hz burst spike
                let bg = (2.0 * std::f64::consts::PI * 10.0 * t).sin();
                let burst = if (i / 500) % 7 == 0 { 8.0 * (2.0 * std::f64::consts::PI * 40.0 * t).sin() } else { 0.0 };
                bg + burst
            })
            .collect();
        // Inject a flat run (≥1 s).
        for v in data.iter_mut().skip(15_000).take(120) { *v = 0.5; }
        // Inject one NaN sample.
        data[20_000] = f64::NAN;

        let opts = ArtifactOpts::default();
        let via_api = detect_artifacts(&data, fs, &opts);

        // Sequential reference — mirror the body of detect_artifacts
        // but without rayon::join.
        let (_, sample_std) = mean_std(&data);
        let flat = flat_run_mask_with_tolerance(
            &data,
            fs.round().max(1.0) as usize,
            0.02 * sample_std,
        );
        let mut bad: Vec<bool> = (0..data.len())
            .map(|i| !data[i].is_finite() || flat[i])
            .collect();
        bad = find_outlier_noise(&data, &bad, 10.0);
        let data_fixed = interp_bad_samples(&data, &bad);
        let hf_seq = compute_band_artifacts(
            &data_fixed, fs, opts.hf_pass, opts.hf_crit, &bad,
            opts.smooth_duration, opts.detrend_duration, opts.hf_detrend,
            opts.zscore_method,
        );
        let bb_seq = compute_band_artifacts(
            &data_fixed, fs, opts.bb_pass, opts.bb_crit, &bad,
            opts.smooth_duration, opts.detrend_duration, opts.bb_detrend,
            opts.zscore_method,
        );
        let mut seq: Vec<bool> = (0..n)
            .map(|i| hf_seq[i] || bb_seq[i] || bad[i])
            .collect();
        if opts.buffer_duration > 0.0 {
            let k = (opts.buffer_duration * fs).round() as usize;
            if k > 0 { seq = binary_dilate(&seq, k); }
        }

        assert_eq!(via_api.len(), seq.len(), "length mismatch");
        let diffs = via_api.iter().zip(seq.iter()).filter(|(a, b)| a != b).count();
        assert_eq!(diffs, 0, "parallel vs sequential masks differ at {} samples", diffs);
        // Sanity: SOMETHING was flagged.
        assert!(via_api.iter().any(|&b| b), "expected at least one artifact sample");
    }

    #[test]
    fn iter_robust_zscore_matches_sort_reference() {
        // Compare the optimised select_nth_unstable + buffer-reuse path
        // against a deliberately-naïve sort + alloc-each-iter reference
        // on a stack of synthetic series. Mask must be bit-identical.
        //
        // The reference mimics the prior implementation: for each
        // iteration it collects good values into a fresh Vec, runs the
        // OLD `median()` + `mad_meanabs()` helpers (sort + sum), then
        // does the same threshold pass.
        fn reference(
            y: &[f64], bad: &[bool], crit: f64, method: ZScoreMethod,
        ) -> Vec<bool> {
            let mut mask = bad.to_vec();
            loop {
                let good: Vec<f64> = y.iter().zip(mask.iter())
                    .filter_map(|(v, &m)| if !m && v.is_finite() { Some(*v) } else { None })
                    .collect();
                if good.is_empty() { return mask; }
                let (mid, scale) = match method {
                    ZScoreMethod::Robust => (median(&good), mad_meanabs(&good)),
                    ZScoreMethod::Standard => mean_std(&good),
                };
                if !scale.is_finite() || scale == 0.0 { return mask; }
                let mut any = false;
                for i in 0..y.len() {
                    if !mask[i] && y[i].is_finite() && ((y[i] - mid) / scale).abs() > crit {
                        mask[i] = true; any = true;
                    }
                }
                if !any { return mask; }
            }
        }
        let mut state: u64 = 0xC0FFEE;
        let mut next = || -> f64 {
            state ^= state << 13; state ^= state >> 7; state ^= state << 17;
            (state >> 11) as f64 / ((1u64 << 53) as f64)
        };
        for &n in &[1usize, 2, 7, 64, 1000, 10_000] {
            for &(crit, method) in &[
                (2.0_f64, ZScoreMethod::Robust),
                (5.5, ZScoreMethod::Robust),
                (3.0, ZScoreMethod::Standard),
            ] {
                // Heavy-tail-ish input + duplicates: gaussian-ish bulk
                // and the occasional 10× outlier.
                let y: Vec<f64> = (0..n).map(|_| {
                    let u = next();
                    let base = (u - 0.5) * 4.0;
                    if next() < 0.02 { base * 25.0 } else { base }
                }).collect();
                let bad = vec![false; n];
                let opt = iterative_robust_zscore(&y, &bad, crit, method);
                let ref_ = reference(&y, &bad, crit, method);
                assert_eq!(
                    opt, ref_,
                    "(n={}, crit={}, method={:?}) mask diverged",
                    n, crit, method
                );
            }
        }
    }

    #[test]
    fn iter_robust_zscore_converges() {
        // Outlier at index 5; should be flagged after 1+ iterations.
        let y = vec![1.0, 1.1, 0.9, 1.05, 0.95, 50.0, 1.0, 1.02];
        let bad = vec![false; 8];
        let out = iterative_robust_zscore(&y, &bad, 3.0, ZScoreMethod::Robust);
        assert!(out[5], "outlier at index 5 should be flagged");
        for i in [0, 1, 2, 3, 4, 6, 7] {
            assert!(!out[i], "index {} should not be flagged", i);
        }
    }
}
