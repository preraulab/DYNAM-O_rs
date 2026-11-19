//! Artifact detection. Port of pydynamo `artifacts.py::detect_artifacts`
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
//! **Slope test.** [`slope_test_artifacts`] is MATLAB's `slope_test=true`
//! branch: a multitaper spectrogram on 10 s windows, a log-log line fit per
//! window, and a flag where the spectrum is too flat.
//! [`detect_artifacts_with_slope_test`] is the full MATLAB default. It runs
//! the slope test first and hands the mask to
//! [`detect_artifacts_with_known_bad`], which treats it the way
//! `detect_artifacts.m` treats `bad_slope`: the flagged samples join the
//! flat-run and non-finite samples BEFORE the band tests, so they are
//! interpolated out of the filtered signal and left out of the z-score
//! statistics. OR-ing a slope mask onto the finished band mask is not
//! equivalent, because the band thresholds are then computed from data
//! MATLAB would have excluded.
//!
//! [`detect_artifacts`] is the same detector with no slope test (MATLAB
//! `slope_test=false`).

use crate::filter_design::cheby1_sos;
use crate::signal::{hilbert, movmean, sosfiltfilt};
use multitaper_rs::dpss::dpss_tapers;
use multitaper_rs::{compute_spectrogram, DetrendMode, SpectrogramParams, Weighting};
use ndarray::ArrayView1;

/// Order of the Chebyshev Type-I highpass that isolates each band.
const BAND_HIGHPASS_ORDER: usize = 4;
/// Passband ripple of that highpass, in dB.
const BAND_HIGHPASS_RIPPLE_DB: f64 = 0.2;
/// Samples this many standard deviations from the mean of the good samples
/// are flagged before the band tests.
const OUTLIER_NOISE_SD: f64 = 10.0;

/// Options for `detect_artifacts`. Matches pydynamo defaults.
#[derive(Debug, Clone)]
pub struct ArtifactOpts {
    /// Highpass cutoff of the high-frequency band, in Hz.
    pub hf_pass: f64,
    /// z-score above which a high-frequency-band sample is flagged.
    pub hf_crit: f64,
    /// Highpass cutoff of the broad band, in Hz.
    pub bb_pass: f64,
    /// z-score above which a broad-band sample is flagged.
    pub bb_crit: f64,
    /// Subtract a moving median from the high-frequency band's log envelope
    /// before the z-score.
    pub hf_detrend: bool,
    /// Subtract a moving median from the broad band's log envelope before
    /// the z-score.
    pub bb_detrend: bool,
    /// Length of the moving-mean window that smooths the envelope, in seconds.
    pub smooth_duration: f64,
    /// Length of the moving-median detrend window, in seconds.
    pub detrend_duration: f64,
    /// Time added to each side of every flagged run, in seconds. 0 adds none.
    pub buffer_duration: f64,
    /// How the z-score is centered and scaled.
    pub zscore_method: ZScoreMethod,
}

/// How `iterative_robust_zscore` centers and scales.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZScoreMethod {
    /// Median center, mean-absolute-deviation scale (pydynamo default).
    Robust,
    /// Mean center, standard-deviation scale.
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

/// True where `data` sits in a run of at least `min_run` identical values.
/// The strict form of [`flat_run_mask_with_tolerance`] (tolerance zero),
/// matching MATLAB `get_chunks` called without a tolerance.
pub fn flat_run_mask(data: &[f64], min_run: usize) -> Vec<bool> {
    flat_run_mask_with_tolerance(data, min_run, 0.0)
}

/// Flag runs of near-constant samples of length at least `min_run`. A run
/// extends while its full span (max − min) stays within `tol`, so a slow
/// drift is not joined into one run merely because each step is small.
/// `tol == 0.0` reduces to strict equality. A non-finite sample always ends
/// the run. This is MATLAB `get_chunks(data, min_len, max_len, tol)`.
///
/// A tolerance is needed in practice because artifact detection runs on the
/// resampled signal, and the resampler leaves small ringing in flat regions.
/// A truly disconnected or railed segment is then near-constant but never
/// bit-identical. The tolerance recovers those while staying far below real
/// EEG variation (flat interior about 0.01 µV against about 20 µV for EEG),
/// so quiet sleep is not flagged.
pub fn flat_run_mask_with_tolerance(data: &[f64], min_run: usize, tol: f64) -> Vec<bool> {
    let n = data.len();
    let mut mask = vec![false; n];
    if n == 0 {
        return mask;
    }
    let mut run_start = 0usize;
    let mut run_min = data[0];
    let mut run_max = data[0];
    for i in 1..=n {
        // Would extending the run to sample i keep its span within tol? A
        // non-finite sample always breaks the run (handled as bad elsewhere).
        let mut extend = false;
        if i < n && data[i].is_finite() {
            let min_if_extended = run_min.min(data[i]);
            let max_if_extended = run_max.max(data[i]);
            extend = (max_if_extended - min_if_extended) <= tol;
        }
        if extend {
            run_min = run_min.min(data[i]);
            run_max = run_max.max(data[i]);
        } else {
            if i - run_start >= min_run {
                for m in mask.iter_mut().take(i).skip(run_start) {
                    *m = true;
                }
            }
            if i < n {
                run_start = i;
                run_min = data[i];
                run_max = data[i];
            }
        }
    }
    mask
}

/// Fraction of the signal's standard deviation used as the flat-run
/// tolerance when `DYNAMO_FLAT_TOL_FRAC` is not set.
const DEFAULT_FLAT_TOL_FRAC: f64 = 0.02;

/// Tolerance for [`flat_run_mask_with_tolerance`] as used by
/// [`detect_artifacts`]. It is `frac × std` over the finite samples, where
/// `frac` comes from the `DYNAMO_FLAT_TOL_FRAC` environment variable and
/// defaults to 0.02. That is large enough for resampler ringing inside a
/// flat or disconnected segment (about 0.01 µV) to still read as flat. It
/// is about 1000 times below real EEG variation (about 20 µV), so quiet
/// sleep is never flagged. `frac = 0` gives strict equality.
pub fn flat_tolerance(data: &[f64]) -> f64 {
    // Use the environment value only when it parses as a non-negative number.
    let mut frac = DEFAULT_FLAT_TOL_FRAC;
    if let Ok(text) = std::env::var("DYNAMO_FLAT_TOL_FRAC") {
        if let Ok(value) = text.parse::<f64>() {
            if value >= 0.0 {
                frac = value;
            }
        }
    }
    let mut sum = 0.0f64;
    let mut sum2 = 0.0f64;
    let mut cnt = 0usize;
    for &v in data {
        if v.is_finite() {
            sum += v;
            sum2 += v * v;
            cnt += 1;
        }
    }
    let std_dev = if cnt > 1 {
        ((sum2 - sum * sum / cnt as f64) / (cnt as f64 - 1.0))
            .max(0.0)
            .sqrt()
    } else {
        0.0
    };
    frac * std_dev
}

/// MATLAB `movmedian(x, win)`. Centered moving median with shrinking
/// window at the endpoints.
///
/// Implementation: two heaps with lazy deletion (a max-heap holding the
/// lower half of the window and a min-heap holding the upper half,
/// rebalanced after every insert and remove). Each step costs O(log W)
/// for a window of W samples, and the working set is the window alone
/// (about 300 KB for W ≈ 40 000), so it stays in the CPU cache.
///
/// The test `movmedian_two_heap_matches_fenwick` checks it bit for bit
/// against the Fenwick-tree reference `movmedian_fenwick`.
fn movmedian(x: &[f64], win: usize) -> Vec<f64> {
    movmedian_two_heap(x, win)
}

/// Two-heap sliding median with lazy deletion. O(n log W).
fn movmedian_two_heap(x: &[f64], win: usize) -> Vec<f64> {
    use std::cmp::Reverse;
    use std::collections::{BinaryHeap, HashMap};

    /// Total-ordering wrapper around f64. Uses `partial_cmp` and falls
    /// back to `Equal` for NaN. Matches the Fenwick reference impl.
    #[derive(Copy, Clone, Debug)]
    struct OrderedF64(f64);
    impl PartialEq for OrderedF64 {
        fn eq(&self, o: &Self) -> bool {
            self.0.to_bits() == o.0.to_bits()
        }
    }
    impl Eq for OrderedF64 {}
    impl PartialOrd for OrderedF64 {
        fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(o))
        }
    }
    impl Ord for OrderedF64 {
        fn cmp(&self, o: &Self) -> std::cmp::Ordering {
            self.0.partial_cmp(&o.0).unwrap_or(std::cmp::Ordering::Equal)
        }
    }

    let n = x.len();
    if win <= 1 || n == 0 {
        return x.to_vec();
    }
    let (half_l, half_r) = crate::signal::matlab_window_halves(win);

    // lo: max-heap (top is largest of the small half).
    // hi: min-heap (top is smallest of the large half).
    let mut lo: BinaryHeap<OrderedF64> = BinaryHeap::new();
    let mut hi: BinaryHeap<Reverse<OrderedF64>> = BinaryHeap::new();
    // Lazy-delete counts keyed by f64 bit pattern.
    let mut del_lo: HashMap<u64, u32> = HashMap::new();
    let mut del_hi: HashMap<u64, u32> = HashMap::new();
    // Effective sizes (excluding pending deletions).
    let mut n_lo: usize = 0;
    let mut n_hi: usize = 0;

    // Pop any tops that have been marked for lazy deletion.
    let prune_lo = |lo: &mut BinaryHeap<OrderedF64>, del_lo: &mut HashMap<u64, u32>| {
        while let Some(&OrderedF64(top)) = lo.peek() {
            let bits = top.to_bits();
            match del_lo.get_mut(&bits) {
                Some(c) if *c > 0 => {
                    *c -= 1;
                    if *c == 0 {
                        del_lo.remove(&bits);
                    }
                    lo.pop();
                }
                _ => break,
            }
        }
    };
    let prune_hi = |hi: &mut BinaryHeap<Reverse<OrderedF64>>, del_hi: &mut HashMap<u64, u32>| {
        while let Some(&Reverse(OrderedF64(top))) = hi.peek() {
            let bits = top.to_bits();
            match del_hi.get_mut(&bits) {
                Some(c) if *c > 0 => {
                    *c -= 1;
                    if *c == 0 {
                        del_hi.remove(&bits);
                    }
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
                Some(&OrderedF64(top)) => OrderedF64(v).cmp(&OrderedF64(top)) != std::cmp::Ordering::Greater,
            };
            if goes_lo {
                lo.push(OrderedF64(v));
                n_lo += 1;
            } else {
                hi.push(Reverse(OrderedF64(v)));
                n_hi += 1;
            }
            b += 1;
        }

        // Remove departed samples on the left (lazy).
        while a < new_a {
            let v = x[a];
            // Decide which heap by comparing to lo-top.
            let on_lo = match lo.peek() {
                Some(&OrderedF64(top)) => OrderedF64(v).cmp(&OrderedF64(top)) != std::cmp::Ordering::Greater,
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
                if let Some(OrderedF64(top)) = lo.pop() {
                    hi.push(Reverse(OrderedF64(top)));
                    n_lo -= 1;
                    n_hi += 1;
                } else { break; }
            } else if n_hi > n_lo {
                if let Some(Reverse(OrderedF64(top))) = hi.pop() {
                    lo.push(OrderedF64(top));
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
            lo.peek().map(|&OrderedF64(v)| v).unwrap_or(f64::NAN)
        } else {
            // Even: avg of lo.top() and hi.top().
            let l = lo.peek().map(|&OrderedF64(v)| v).unwrap_or(f64::NAN);
            let h = hi.peek().map(|&Reverse(OrderedF64(v))| v).unwrap_or(f64::NAN);
            0.5 * (l + h)
        };
        out.push(med);
    }
    out
}

/// Fenwick-tree (binary indexed tree) moving median. Test-only reference:
/// `movmedian_two_heap` must match it bit for bit. Its tree spans all n
/// samples rather than one window, so it is slower on long signals.
#[cfg(test)]
fn movmedian_fenwick(x: &[f64], win: usize) -> Vec<f64> {
    let n = x.len();
    if win <= 1 || n == 0 {
        return x.to_vec();
    }
    let (half_l, half_r) = crate::signal::matlab_window_halves(win);

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

/// Median of finite values. Returns NaN for all-NaN input. Used by the
/// test-only reference implementation in `mod tests`.
#[cfg(test)]
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
/// (Not median absolute deviation! pydynamo matches.) Used by the test-only
/// reference implementation in `mod tests`.
#[cfg(test)]
fn mad_meanabs(xs: &[f64]) -> f64 {
    let finite: Vec<f64> = xs.iter().filter(|x| x.is_finite()).copied().collect();
    if finite.is_empty() {
        return f64::NAN;
    }
    let mean = finite.iter().sum::<f64>() / finite.len() as f64;
    finite.iter().map(|&v| (v - mean).abs()).sum::<f64>() / finite.len() as f64
}

/// Mean and sample standard deviation (n - 1) of the finite values. Used by
/// the test-only reference implementation in `mod tests`.
#[cfg(test)]
fn mean_std(xs: &[f64]) -> (f64, f64) {
    let finite: Vec<f64> = xs.iter().filter(|x| x.is_finite()).copied().collect();
    if finite.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    let n = finite.len() as f64;
    let mean = finite.iter().sum::<f64>() / n;
    if finite.len() < 2 {
        return (mean, f64::NAN);
    }
    let variance = finite.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
    (mean, variance.sqrt())
}

/// Iteratively flag samples where |z| > crit, recomputing the center and
/// scale on the remaining good samples until no new sample is flagged.
/// Returns a bad-sample mask that includes the input `bad`.
///
/// `Robust` centers on the median and scales by the mean absolute
/// deviation from the mean (MATLAB default `mad(x)`). `Standard` uses
/// the mean and the sample standard deviation.
///
/// Implementation:
/// * One buffer of good values is refilled on every pass.
/// * The median comes from `select_nth_unstable_by`, which is O(n) on
///   average where a full sort is O(n log n). It returns the same value
///   a full sort would put at index `k`.
/// * For an even count, the partition leaves the `k` smallest values
///   (unordered) in `buf[..k]`. Their max is the other middle value, so
///   no second selection is needed.
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
                // Median by partition, plus (for an even count) the max
                // of the lower half. Same value as sorting and taking the
                // middle.
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
                // the median, matching MATLAB default `mad(x)`).
                let mean = buf.iter().sum::<f64>() / n as f64;
                let mad = buf.iter().map(|&v| (v - mean).abs()).sum::<f64>() / n as f64;
                (med, mad)
            }
            ZScoreMethod::Standard => {
                let n = buf.len() as f64;
                let mean = buf.iter().sum::<f64>() / n;
                if buf.len() < 2 {
                    (mean, f64::NAN)
                } else {
                    let variance = buf.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
                    (mean, variance.sqrt())
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
    let mut good_vals: Vec<f64> = Vec::new();
    for i in 0..data.len() {
        if !bad[i] && data[i].is_finite() {
            good_vals.push(data[i]);
        }
    }
    if good_vals.is_empty() {
        return out;
    }
    let n = good_vals.len() as f64;
    let mean = good_vals.iter().sum::<f64>() / n;
    if good_vals.len() < 2 {
        return out;
    }
    let variance = good_vals.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let std_dev = variance.sqrt();
    let lower_limit = mean - outlier_scalar * std_dev;
    let upper_limit = mean + outlier_scalar * std_dev;
    for i in 0..data.len() {
        if data[i] <= lower_limit || data[i] >= upper_limit {
            out[i] = true;
        }
    }
    out
}

/// Linear interpolation over bad samples from the nearest good sample on
/// each side. Before the first good sample and after the last one, the
/// nearest good value is held. Same as the `np.interp` step in pydynamo
/// `detect_artifacts`.
fn interp_bad_samples(data: &[f64], bad: &[bool]) -> Vec<f64> {
    let n = data.len();
    let mut fixed = data.to_vec();
    let mut good_idx: Vec<usize> = Vec::new();
    for i in 0..n {
        if !bad[i] {
            good_idx.push(i);
        }
    }
    if good_idx.is_empty() {
        return fixed;
    }
    // Interpolation knots: every good sample, plus one knot just outside
    // each end (positions -1 and n) carrying the first and last good
    // value, so bad samples at the edges take the nearest good value.
    let first_good = data[good_idx[0]];
    let last_good = data[good_idx[good_idx.len() - 1]];
    let mut knot_positions: Vec<f64> = Vec::with_capacity(good_idx.len() + 2);
    let mut knot_values: Vec<f64> = Vec::with_capacity(good_idx.len() + 2);
    knot_positions.push(-1.0);
    knot_values.push(first_good);
    for &i in &good_idx {
        knot_positions.push(i as f64);
        knot_values.push(data[i]);
    }
    knot_positions.push(n as f64);
    knot_values.push(last_good);

    for i in 0..n {
        if bad[i] {
            let q = i as f64;
            // Binary search: find j such that
            // knot_positions[j-1] <= q < knot_positions[j].
            let j = knot_positions.partition_point(|&t| t <= q);
            let t0 = knot_positions[j - 1];
            let t1 = knot_positions[j];
            let y0 = knot_values[j - 1];
            let y1 = knot_values[j];
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
    // Time each sub-step (filter design, filter, Hilbert, smooth, detrend,
    // z-score). The timings are printed at the end of this function, and
    // only when the `DYNAMO_ARTIFACT_TIMING` environment variable is set.
    let t_band = std::time::Instant::now();

    // Chebyshev Type-I highpass, as second-order sections.
    let t = std::time::Instant::now();
    let sos: Vec<[f64; 6]> = cheby1_sos(
        BAND_HIGHPASS_ORDER,
        BAND_HIGHPASS_RIPPLE_DB,
        &[passband],
        "highpass",
        fs,
    );
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
    // Log of the smoothed envelope. A non-positive value has no log and
    // becomes NaN, which the z-score step skips.
    for v in y.iter_mut() {
        if *v > 0.0 {
            *v = v.ln();
        } else {
            *v = f64::NAN;
        }
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
        let band_tag = format!("band(pass={:.2}Hz, crit={:.2}, detrend={})", passband, crit, detrend_on);
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
    // Find runs of True. Expand each run by k on both sides.
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
/// run + outlier-noise flagging, with no slope test (MATLAB
/// `detect_artifacts(..., 'slope_test', false)`).
pub fn detect_artifacts(data: &[f64], fs: f64, opts: &ArtifactOpts) -> Vec<bool> {
    detect_artifacts_with_known_bad(data, fs, opts, None)
}

/// Detect artifacts given samples already known to be bad, such as the
/// slope-test mask. Matches `detect_artifacts.m` with `slope_test=true`:
///
/// ```text
/// bad_inds = isnan(data) | isinf(data) | is_flat | bad_slope;
/// bad_inds = find_outlier_noise(data, bad_inds);
/// ```
///
/// The known-bad samples are part of `bad_inds` from the start. They are
/// excluded from the outlier-noise mean and standard deviation, interpolated
/// out of the signal the band filters see, left out of the band z-score
/// statistics, and always present in the returned mask.
///
/// `known_bad`, when given, must have the same length as `data`.
pub fn detect_artifacts_with_known_bad(
    data: &[f64],
    fs: f64,
    opts: &ArtifactOpts,
    known_bad: Option<&[bool]>,
) -> Vec<bool> {
    let n = data.len();
    if n == 0 {
        return vec![];
    }
    if let Some(mask) = known_bad {
        assert_eq!(mask.len(), n, "known_bad must have one entry per sample");
    }

    // Flat runs of at least one second, with a data-scaled tolerance (see
    // flat_tolerance).
    let one_second_samples = fs.round().max(1.0) as usize;
    let flat = flat_run_mask_with_tolerance(data, one_second_samples, flat_tolerance(data));
    let mut bad: Vec<bool> = Vec::with_capacity(n);
    for i in 0..n {
        let already_known_bad = match known_bad {
            Some(mask) => mask[i],
            None => false,
        };
        bad.push(!data[i].is_finite() || flat[i] || already_known_bad);
    }
    bad = find_outlier_noise(data, &bad, OUTLIER_NOISE_SD);

    // Interp through bad samples so filters don't see NaN / flats.
    let data_fixed = interp_bad_samples(data, &bad);

    // Run the BB and HF bands concurrently. They're independent
    // (different filter design, separate copy of `data_fixed`), and
    // each one is sequential internally, so `rayon::join` halves the
    // wall-clock for this phase on any machine with ≥2 cores. The
    // sequential fallback (when the `parallel` feature is off, e.g.
    // WASM) is bit-identical because each band is a deterministic
    // function of `data_fixed` + `bad`.
    let (hf_art, bb_art) = {
        let data_ref = &data_fixed;
        let bad_ref = &bad;
        let run_hf_band = move || {
            compute_band_artifacts(
                data_ref,
                fs,
                opts.hf_pass,
                opts.hf_crit,
                bad_ref,
                opts.smooth_duration,
                opts.detrend_duration,
                opts.hf_detrend,
                opts.zscore_method,
            )
        };
        let run_bb_band = move || {
            compute_band_artifacts(
                data_ref,
                fs,
                opts.bb_pass,
                opts.bb_crit,
                bad_ref,
                opts.smooth_duration,
                opts.detrend_duration,
                opts.bb_detrend,
                opts.zscore_method,
            )
        };
        #[cfg(feature = "parallel")]
        {
            rayon::join(run_hf_band, run_bb_band)
        }
        #[cfg(not(feature = "parallel"))]
        {
            (run_hf_band(), run_bb_band())
        }
    };
    let mut artifacts: Vec<bool> = (0..n)
        .map(|i| hf_art[i] || bb_art[i] || bad[i])
        .collect();

    if opts.buffer_duration > 0.0 {
        let buffer_samples = (opts.buffer_duration * fs).round() as usize;
        if buffer_samples > 0 {
            artifacts = binary_dilate(&artifacts, buffer_samples);
        }
    }
    artifacts
}

/// Slope-test artifact mask. The `slope_test` block of `detect_artifacts.m`.
///
/// ```text
/// [spect, stimes, sfreqs] = multitaper_spectrogram(data, Fs, [1 min(55,Fs/2)], [10,19], [10 5], ...);
/// B(ii,:) = polyfit(log(sfreqs)', log(spect(:,ii)), 1);
/// bad_slope = interp1(stimes, double(B(:,1) > slope_crit)', t, 'nearest')';
/// bad_slope(isnan(bad_slope)) = 1;
/// ```
///
/// A multitaper spectrogram (time-half-bandwidth 10, 19 tapers) is computed
/// on 10 s windows stepped by 5 s over 1 to `min(55, fs/2)` Hz. Each window
/// gets a straight-line fit of log power against log frequency. A window
/// whose slope is above `slope_crit` (MATLAB default −0.5) has a spectrum
/// too flat to be EEG and is flagged. A window with any non-positive or
/// non-finite power is flagged too.
///
/// Each sample takes the verdict of the nearest window center, an exact
/// midpoint going to the later window. Samples before the first or after
/// the last window center are always flagged, as in MATLAB: with the 10 s
/// window that is the first and last 5 s of every recording.
///
/// Returns one flag per sample of `data`. Errors come from the taper or
/// spectrogram computation.
pub fn slope_test_artifacts(data: &[f64], fs: f64, slope_crit: f64) -> Result<Vec<bool>, String> {
    let n_samples = data.len();
    let window_s = 10.0_f64;
    let step_s = 5.0_f64;
    let time_half_bandwidth = 10.0_f64;
    let n_tapers = 19_usize;
    let freq_max = (55.0_f64).min(fs / 2.0);
    let freq_range = (1.0_f64, freq_max);

    let window_samples = (window_s * fs).round() as usize;
    if window_samples < 4 {
        return Ok(vec![false; n_samples]);
    }
    // FFT length for a 0.1 Hz frequency grid, rounded up to a power of two.
    // MATLAB arrives at the same length as the next power of two of the
    // 10 s window (`fs / 0.1` and `10 * fs` are the same number).
    let frequency_spacing_hz = 0.1_f64;
    let nfft = ((fs / frequency_spacing_hz).ceil() as usize).next_power_of_two();

    let tapers = dpss_tapers(window_samples, time_half_bandwidth, n_tapers)
        .map_err(|e| format!("slope test tapers: {}", e))?;
    let spectrogram = compute_spectrogram(
        ArrayView1::from(data),
        tapers.view(),
        None,
        &SpectrogramParams {
            fs,
            frequency_range: freq_range,
            window_params: (window_s, step_s),
            nfft,
            // MATLAB passes [] for the detrend option here, which its
            // multitaper wrapper resolves to 'linear'. A constant detrend
            // would leave drift inside each 10 s window and tilt the fit.
            detrend: DetrendMode::Linear,
            weighting: Weighting::Unity,
        },
    )
    .map_err(|e| format!("slope test spectrogram: {}", e))?;

    let power = &spectrogram.mt_spectrogram; // (frequencies, windows)
    let window_centers = &spectrogram.stimes;
    let frequencies = &spectrogram.sfreqs;
    let n_freqs = frequencies.len();
    let n_windows = window_centers.len();
    if n_freqs == 0 || n_windows == 0 {
        return Ok(vec![false; n_samples]);
    }

    // Least-squares slope of log(power) on log(frequency):
    //   slope = sum((x - mean_x) * (y - mean_y)) / sum((x - mean_x)^2)
    // The x side is the same for every window, so it is computed once.
    let mut log_freq: Vec<f64> = Vec::with_capacity(n_freqs);
    for &frequency in frequencies.iter() {
        // Floor that keeps log() finite. The grid starts at 1 Hz, so it is
        // not reached in practice.
        log_freq.push(frequency.max(1e-30).ln());
    }
    let mean_log_freq: f64 = log_freq.iter().sum::<f64>() / n_freqs as f64;
    let mut sum_squares_x = 0.0_f64;
    for &x in &log_freq {
        sum_squares_x += (x - mean_log_freq).powi(2);
    }
    if sum_squares_x <= 0.0 || !sum_squares_x.is_finite() {
        return Ok(vec![false; n_samples]);
    }

    let mut window_is_bad: Vec<bool> = vec![false; n_windows];
    let mut log_power: Vec<f64> = Vec::with_capacity(n_freqs);
    for window in 0..n_windows {
        log_power.clear();
        let mut has_unusable_power = false;
        for freq_index in 0..n_freqs {
            let value = power[[freq_index, window]];
            if !(value > 0.0 && value.is_finite()) {
                has_unusable_power = true;
                break;
            }
            log_power.push(value.ln());
        }
        if has_unusable_power {
            window_is_bad[window] = true;
            continue;
        }
        let mean_log_power: f64 = log_power.iter().sum::<f64>() / n_freqs as f64;
        let mut sum_cross = 0.0_f64;
        for freq_index in 0..n_freqs {
            sum_cross += (log_freq[freq_index] - mean_log_freq) * (log_power[freq_index] - mean_log_power);
        }
        let slope = sum_cross / sum_squares_x;
        window_is_bad[window] = slope > slope_crit;
    }

    // Spread the per-window verdicts onto the samples.
    let first_center = window_centers[0];
    let last_center = window_centers[n_windows - 1];
    let mut mask: Vec<bool> = vec![false; n_samples];
    for (sample, flag) in mask.iter_mut().enumerate() {
        let time = sample as f64 / fs;
        if time < first_center || time > last_center {
            *flag = true;
            continue;
        }
        // First window center at or after this sample.
        let mut low = 0_usize;
        let mut high = n_windows - 1;
        while low < high {
            let middle = (low + high) / 2;
            if window_centers[middle] < time {
                low = middle + 1;
            } else {
                high = middle;
            }
        }
        // Step back to the previous center when it is strictly closer.
        let mut nearest = low;
        if nearest > 0 {
            let distance_previous = (time - window_centers[nearest - 1]).abs();
            let distance_current = (time - window_centers[nearest]).abs();
            if distance_previous < distance_current {
                nearest -= 1;
            }
        }
        *flag = window_is_bad[nearest];
    }
    Ok(mask)
}

/// The full MATLAB default detector, `detect_artifacts(data, Fs)` with
/// `slope_test = true`: the slope test runs first and its mask is folded
/// into the bad-sample set before the band tests (see
/// [`detect_artifacts_with_known_bad`]).
pub fn detect_artifacts_with_slope_test(
    data: &[f64],
    fs: f64,
    opts: &ArtifactOpts,
    slope_crit: f64,
) -> Result<Vec<bool>, String> {
    let slope_mask = slope_test_artifacts(data, fs, slope_crit)?;
    Ok(detect_artifacts_with_known_bad(data, fs, opts, Some(&slope_mask)))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic noise in (-1, 1) from a 64-bit LCG.
    fn lcg_noise(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }

    /// Ten minutes at 100 Hz: quiet noise, one long loud stretch
    /// (samples 20000..32000), and one short moderate burst
    /// (samples 45000..45300).
    fn signal_with_loud_stretch_and_burst() -> Vec<f64> {
        let mut state = 0x00C0_FFEE_u64;
        let mut data = Vec::with_capacity(60_000);
        for i in 0..60_000 {
            let mut amplitude = 1.0;
            if (20_000..32_000).contains(&i) {
                amplitude = 60.0;
            }
            if (45_000..45_300).contains(&i) {
                amplitude = 12.0;
            }
            data.push(amplitude * lcg_noise(&mut state));
        }
        data
    }

    /// White noise has a flat spectrum (slope near 0, above the −0.5
    /// threshold). Integrated white noise falls off as 1/f² (slope near −2).
    #[test]
    fn slope_test_flags_flat_spectra_and_the_recording_edges() {
        let fs = 100.0;
        let n = 12_000; // 120 s
        let mut state = 0x5EED_u64;
        let mut data = Vec::with_capacity(n);
        let mut random_walk = 0.0;
        for i in 0..n {
            let noise = lcg_noise(&mut state);
            random_walk += noise;
            // 40 s to 80 s is white noise, the rest is the random walk.
            if (4_000..8_000).contains(&i) {
                data.push(noise);
            } else {
                data.push(random_walk);
            }
        }
        let mask = slope_test_artifacts(&data, fs, -0.5).expect("slope test");
        assert_eq!(mask.len(), n);
        // Window centers run from 5 s to 115 s. Samples strictly before the
        // first or strictly after the last center are always flagged.
        assert!(mask[0] && mask[499]);
        assert!(mask[n - 1] && mask[n - 499]);
        // Steep-spectrum stretches pass, the flat-spectrum stretch is flagged.
        assert!(!mask[2_000], "random walk at 20 s should pass");
        assert!(mask[6_000], "white noise at 60 s should be flagged");
        assert!(!mask[10_000], "random walk at 100 s should pass");
    }

    #[test]
    fn detector_with_slope_test_is_the_known_bad_detector_on_the_slope_mask() {
        let data = signal_with_loud_stretch_and_burst();
        let opts = ArtifactOpts::default();
        let slope_mask = slope_test_artifacts(&data, 100.0, -0.5).expect("slope test");
        let expected = detect_artifacts_with_known_bad(&data, 100.0, &opts, Some(&slope_mask));
        let combined = detect_artifacts_with_slope_test(&data, 100.0, &opts, -0.5).expect("detector");
        assert_eq!(combined, expected);
    }

    #[test]
    fn known_bad_none_is_the_plain_detector() {
        let data = signal_with_loud_stretch_and_burst();
        let opts = ArtifactOpts::default();
        let plain = detect_artifacts(&data, 100.0, &opts);
        let with_none = detect_artifacts_with_known_bad(&data, 100.0, &opts, None);
        assert_eq!(plain, with_none);
    }

    #[test]
    fn known_bad_samples_are_always_flagged() {
        let data = signal_with_loud_stretch_and_burst();
        let mut known_bad = vec![false; data.len()];
        for flag in known_bad.iter_mut().take(1_500).skip(1_000) {
            *flag = true; // a quiet stretch no band test would flag
        }
        let mask =
            detect_artifacts_with_known_bad(&data, 100.0, &ArtifactOpts::default(), Some(&known_bad));
        for i in 1_000..1_500 {
            assert!(mask[i], "known-bad sample {} missing from the mask", i);
        }
    }

    #[test]
    fn known_bad_samples_change_the_band_statistics() {
        // Folding a mask in before the band tests is not the same as OR-ing
        // it on afterwards: with the loud stretch excluded up front, the
        // thresholds are computed from the quiet data alone.
        let data = signal_with_loud_stretch_and_burst();
        let opts = ArtifactOpts::default();
        let mut known_bad = vec![false; data.len()];
        for flag in known_bad.iter_mut().take(32_000).skip(20_000) {
            *flag = true;
        }
        let folded_in = detect_artifacts_with_known_bad(&data, 100.0, &opts, Some(&known_bad));
        let plain = detect_artifacts(&data, 100.0, &opts);
        let mut or_afterwards = Vec::with_capacity(data.len());
        for i in 0..data.len() {
            or_afterwards.push(plain[i] || known_bad[i]);
        }
        assert_ne!(folded_in, or_afterwards);
    }

    #[test]
    fn flat_run_detection() {
        let x = vec![1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 3.0];
        // min_run=3 → the first run (4 × 1.0) and last (3 × 3.0) are flat.
        // tol=0 is strict equality.
        let m = flat_run_mask(&x, 3);
        assert_eq!(
            m,
            vec![true, true, true, true, false, false, true, true, true]
        );
    }

    #[test]
    fn even_windows_match_matlab() {
        // MATLAB R2025b:
        //   movmean([1 2 3 4 5 6], 4)   -> 1.5 2 2.5 3.5 4.5 5
        //   movmedian([1 5 2 8 3 9], 4) -> 3 2 3.5 4 5.5 8
        let mean = movmean(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 4);
        assert_eq!(mean, vec![1.5, 2.0, 2.5, 3.5, 4.5, 5.0]);
        let median = movmedian(&[1.0, 5.0, 2.0, 8.0, 3.0, 9.0], 4);
        assert_eq!(median, vec![3.0, 2.0, 3.5, 4.0, 5.5, 8.0]);
        let median_reference = movmedian_fenwick(&[1.0, 5.0, 2.0, 8.0, 3.0, 9.0], 4);
        assert_eq!(median_reference, vec![3.0, 2.0, 3.5, 4.0, 5.5, 8.0]);
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
    fn tolerance_flat_catches_resampled_flat_but_not_eeg() {
        // A genuinely flat/disconnected segment survives resampling as
        // near-constant (sub-unit ringing), not bit-identical. Exact equality
        // misses it. A tolerance catches it, and real EEG is never flagged.
        let min_run = 100usize;
        let mut data = vec![0.0f64; 500];
        // EEG (±20)
        for i in 0..150 {
            data[i] = 20.0 * (i as f64 * 0.5).sin();
        }
        for (i, v) in data.iter_mut().enumerate().take(400).skip(150) {
            *v = -2048.0 + 0.1 * (((i * 7) % 5) as f64 - 2.0); // flat ± ~0.2
        }
        // EEG (±20)
        for i in 400..500 {
            data[i] = 20.0 * (i as f64 * 0.5).sin();
        }

        // Exact equality (tol=0) misses the ringing flat.
        let exact = flat_run_mask(&data, min_run);
        assert!(!exact[250], "exact test wrongly caught (or ringing was zero)");

        // Tolerance above the ±0.2 ringing catches the flat, spares the EEG.
        let m = flat_run_mask_with_tolerance(&data, min_run, 1.0);
        assert!(m[250], "tolerance should catch the near-constant flat");
        assert!(!m[50] && !m[450], "real EEG (±20) must not be flagged with tol=1");
    }

    #[test]
    fn movmedian_two_heap_matches_fenwick() {
        // Bit-identity smoke test across a variety of (n, win) sizes
        // and value distributions, against the Fenwick-tree reference
        // implementation. Uses an inline xorshift PRNG so we don't drag
        // in a `rand` dev-dep.
        let mut state: u64 = 0xDEADBEEF;
        // Uniform value in [0, 1).
        let mut next = || -> f64 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / ((1u64 << 53) as f64)
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
            let two_heap = movmedian_two_heap(&x, win);
            let fenwick = movmedian_fenwick(&x, win);
            assert_eq!(two_heap.len(), fenwick.len(), "(n={}, win={}) length mismatch", n, win);
            for (i, (a, b)) in two_heap.iter().zip(fenwick.iter()).enumerate() {
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
        // Run detect_artifacts twice. Once via the public API (which
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
        for v in data.iter_mut().skip(15_000).take(120) {
            *v = 0.5;
        }
        // Inject one NaN sample.
        data[20_000] = f64::NAN;

        let opts = ArtifactOpts::default();
        let via_api = detect_artifacts(&data, fs, &opts);

        // Sequential reference. Mirror the body of detect_artifacts
        // but without rayon::join.
        let flat = flat_run_mask_with_tolerance(&data, fs.round().max(1.0) as usize, flat_tolerance(&data));
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
            if k > 0 {
                seq = binary_dilate(&seq, k);
            }
        }

        assert_eq!(via_api.len(), seq.len(), "length mismatch");
        let diffs = via_api.iter().zip(seq.iter()).filter(|(a, b)| a != b).count();
        assert_eq!(diffs, 0, "parallel vs sequential masks differ at {} samples", diffs);
        // Sanity: SOMETHING was flagged.
        assert!(via_api.iter().any(|&b| b), "expected at least one artifact sample");
    }

    #[test]
    fn iter_robust_zscore_matches_sort_reference() {
        // Compare the optimized select_nth_unstable + buffer-reuse path
        // against a deliberately-naïve sort + alloc-each-iter reference
        // on a stack of synthetic series. Mask must be bit-identical.
        //
        // The reference collects the good values into a fresh Vec on
        // every pass and uses the sort-based `median()` and
        // `mad_meanabs()` helpers, then applies the same threshold.
        fn reference(
            y: &[f64], bad: &[bool], crit: f64, method: ZScoreMethod,
        ) -> Vec<bool> {
            let mut mask = bad.to_vec();
            loop {
                let good: Vec<f64> = y.iter().zip(mask.iter())
                    .filter_map(|(v, &m)| if !m && v.is_finite() { Some(*v) } else { None })
                    .collect();
                if good.is_empty() {
                    return mask;
                }
                let (mid, scale) = match method {
                    ZScoreMethod::Robust => (median(&good), mad_meanabs(&good)),
                    ZScoreMethod::Standard => mean_std(&good),
                };
                if !scale.is_finite() || scale == 0.0 {
                    return mask;
                }
                let mut any = false;
                for i in 0..y.len() {
                    if !mask[i] && y[i].is_finite() && ((y[i] - mid) / scale).abs() > crit {
                        mask[i] = true;
                        any = true;
                    }
                }
                if !any {
                    return mask;
                }
            }
        }
        let mut state: u64 = 0xC0FFEE;
        let mut next = || -> f64 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
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
