//! Global permutation testing. Port of `gpermtest.m` / `gpermtest2.m`.
//!
//! Where [`super::fdr_grid`] controls the *false discovery rate* (the
//! expected share of flagged bins that are false), this controls the
//! *family-wise error rate* (the probability of any false positive
//! anywhere) by the max-statistic method: one global bound chosen so
//! that at most `alpha` of the permutations have ANY bin exceeding it.
//!
//! The bound is not the `1 - alpha` quantile of the null. MATLAB walks
//! *down* the sorted null one column at a time and, at each candidate,
//! counts the permutations in which any bin exceeds it, stopping once
//! that count reaches `ceil(alpha * iterations)`. The resulting per-bin
//! vector has a *joint* exceedance rate of alpha, which is what makes
//! the control family-wise rather than per-bin.
//!
//! Permutations are drawn from a seeded PRNG so a run is reproducible.
//! The generator differs from MATLAB's, so results are expected to match
//! MATLAB in distribution only, not draw-for-draw.

/// MATLAB default global acceptance level.
/// cbindgen:ignore
pub const DEFAULT_ALPHA: f64 = 0.05;
/// MATLAB default permutation count.
/// cbindgen:ignore
pub const DEFAULT_ITERATIONS: usize = 10_000;

/// MATLAB's offset added to each candidate bound. It keeps the bound
/// strictly above the sorted-null column it came from, so that column does
/// not count itself as an exceedance.
const BOUND_OFFSET: f64 = 1e-5;
/// Largest |requested alpha - achieved alpha| accepted without a warning.
const ALPHA_MISMATCH_TOLERANCE: f64 = 0.01;

/// Outcome of a global permutation test.
#[derive(Debug, Clone)]
pub struct PermTest {
    /// `|true_stat| >= acceptance_bounds`, per bin.
    pub sigbins: Vec<bool>,
    /// Per-bin global bound at the requested alpha.
    pub acceptance_bounds: Vec<f64>,
    /// Observed `statfcn(g1) - statfcn(g2)`.
    pub true_stat: Vec<f64>,
    /// Permutations excluded by the final bound, which is the achieved
    /// alpha numerator. Compare `n_excluded / iterations` against `alpha`.
    pub n_excluded: usize,
    /// Set when the requested alpha could not be resolved on the
    /// permutation grid. The caller should surface it.
    pub warning: Option<String>,
}

/// xoshiro256++ generator. Small, fast, good enough for label shuffling,
/// and vendored here so the kernel gains no dependency for one use. The
/// four fields are the 256-bit state.
struct Rng(u64, u64, u64, u64);

impl Rng {
    fn new(seed: u64) -> Self {
        // SplitMix64 to spread a single seed across the state.
        let mut z = seed;
        let mut next = || {
            z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            x ^ (x >> 31)
        };
        Rng(next(), next(), next(), next())
    }
    /// Next 64 random bits (the reference xoshiro256++ step).
    fn next_u64(&mut self) -> u64 {
        let r = self.0.wrapping_add(self.3).rotate_left(23).wrapping_add(self.0);
        let t = self.1 << 17;
        self.2 ^= self.0;
        self.3 ^= self.1;
        self.1 ^= self.2;
        self.0 ^= self.3;
        self.2 ^= t;
        self.3 = self.3.rotate_left(45);
        r
    }
    /// Uniform in `0..n`, rejection-sampled so the range is unbiased.
    fn below(&mut self, n: u64) -> u64 {
        debug_assert!(n > 0);
        let zone = u64::MAX - (u64::MAX % n);
        loop {
            let v = self.next_u64();
            if v < zone {
                return v % n;
            }
        }
    }
    /// Fisher-Yates shuffle in place.
    fn shuffle(&mut self, idx: &mut [usize]) {
        for i in (1..idx.len()).rev() {
            let j = self.below((i + 1) as u64) as usize;
            idx.swap(i, j);
        }
    }
}

/// NaN-omitting mean across observations, MATLAB's default `statfcn`,
/// `@(x) mean(x, 2, 'omitnan')`.
///
/// `data` is row-major `(n_bins, n_obs)`. The mean of each bin is taken over
/// the observation columns listed in `cols`, skipping non-finite values. A
/// bin with no finite value yields NaN.
fn nanmean_rows(data: &[f64], n_bins: usize, cols: &[usize], n_obs: usize) -> Vec<f64> {
    let mut out = vec![f64::NAN; n_bins];
    for (bin, slot) in out.iter_mut().enumerate() {
        let mut sum = 0.0;
        let mut cnt = 0usize;
        for &c in cols {
            let v = data[bin * n_obs + c];
            if v.is_finite() {
                sum += v;
                cnt += 1;
            }
        }
        if cnt > 0 {
            *slot = sum / cnt as f64;
        }
    }
    out
}

/// Order two values ascending, with every NaN after every number and all
/// NaN equal to each other. This is a total order, and it is the order
/// MATLAB's `sort` produces.
fn compare_nan_last(a: &f64, b: &f64) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    if a.is_nan() && b.is_nan() {
        return Ordering::Equal;
    }
    if a.is_nan() {
        return Ordering::Greater;
    }
    if b.is_nan() {
        return Ordering::Less;
    }
    if a < b {
        Ordering::Less
    } else if a > b {
        Ordering::Greater
    } else {
        Ordering::Equal
    }
}

/// Sort a slice ascending in place, with NaN last.
fn sort_ascending_nan_last(values: &mut [f64]) {
    values.sort_by(compare_nan_last);
}

/// Indices of the observations (columns) that have a finite value in at
/// least one bin. `data` is row-major `(n_bins, n_obs)`. This is MATLAB's
/// `g(:, any(~isnan(g)))`.
fn observed_columns(data: &[f64], n_bins: usize, n_obs: usize) -> Vec<usize> {
    let mut kept = Vec::new();
    for c in 0..n_obs {
        let mut has_finite_value = false;
        for b in 0..n_bins {
            if data[b * n_obs + c].is_finite() {
                has_finite_value = true;
                break;
            }
        }
        if has_finite_value {
            kept.push(c);
        }
    }
    kept
}

/// Global (max-statistic) permutation test over a bin grid.
///
/// `g1` / `g2` are row-major `(n_bins, n_trials)`. Observations that are
/// entirely NaN across bins are dropped first, as `gpermtest.m` does.
/// The statistic is fixed to the NaN-omitting mean. MATLAB accepts an
/// arbitrary `statfcn` handle, which has no clean analog across an FFI
/// boundary and is unused by DYNAM-O.
pub fn gpermtest(
    g1: &[f64],
    g2: &[f64],
    n_bins: usize,
    alpha: f64,
    iterations: usize,
    seed: u64,
) -> Result<PermTest, String> {
    if n_bins == 0 {
        return Err("n_bins must be non-zero".into());
    }
    if g1.len() % n_bins != 0 || g2.len() % n_bins != 0 {
        return Err("group lengths are not multiples of n_bins".into());
    }
    if !(alpha > 0.0 && alpha <= 1.0) {
        return Err(format!("alpha must be in (0, 1], got {}", alpha));
    }
    if iterations == 0 {
        return Err("iterations must be non-zero".into());
    }
    let n1 = g1.len() / n_bins;
    let n2 = g2.len() / n_bins;

    // Drop observations with no finite value in any bin.
    let kept1 = observed_columns(g1, n_bins, n1);
    let kept2 = observed_columns(g2, n_bins, n2);
    if kept1.is_empty() || kept2.is_empty() {
        return Err("both groups need at least one non-empty observation".into());
    }

    // Pool into one (n_bins, kept1 + kept2) block so permuting is an index
    // shuffle.
    let n_pool = kept1.len() + kept2.len();
    let mut pooled = vec![f64::NAN; n_bins * n_pool];
    for b in 0..n_bins {
        for (i, &c) in kept1.iter().enumerate() {
            pooled[b * n_pool + i] = g1[b * n1 + c];
        }
        for (i, &c) in kept2.iter().enumerate() {
            pooled[b * n_pool + kept1.len() + i] = g2[b * n2 + c];
        }
    }

    let stat_a = nanmean_rows(g1, n_bins, &kept1, n1);
    let stat_b = nanmean_rows(g2, n_bins, &kept2, n2);
    let true_stat: Vec<f64> = stat_a
        .iter()
        .zip(stat_b.iter())
        .map(|(a, b)| a - b)
        .collect();

    // Null: |mean(A) - mean(B)| per bin, over shuffled labels. `null_mat`
    // is row-major (n_bins, iterations).
    let mut rng = Rng::new(seed);
    let mut idx: Vec<usize> = (0..n_pool).collect();
    let mut null_mat = vec![0.0f64; n_bins * iterations];
    for it in 0..iterations {
        rng.shuffle(&mut idx);
        let a = nanmean_rows(&pooled, n_bins, &idx[..kept1.len()], n_pool);
        let b = nanmean_rows(&pooled, n_bins, &idx[kept1.len()..], n_pool);
        for bin in 0..n_bins {
            null_mat[bin * iterations + it] = (a[bin] - b[bin]).abs();
        }
    }

    // Per-bin sorted null, ascending, with NaN at the end of the row as
    // MATLAB's `sort` places it. A NaN entry is a shuffle that left one
    // side of that bin with no finite value.
    let mut sorted = null_mat.clone();
    for bin in 0..n_bins {
        let row = &mut sorted[bin * iterations..(bin + 1) * iterations];
        sort_ascending_nan_last(row);
    }

    let out_crit = (alpha * iterations as f64).ceil() as usize;
    let mut warning = None;
    let mut bounds = vec![0.0f64; n_bins];
    let mut numout = 0usize;

    if out_crit >= iterations {
        warning = Some(
            "global bounds would include all iterations; increase iterations or raise alpha"
                .into(),
        );
    } else if out_crit == 0 {
        warning = Some(
            "alpha is below the resolution of the permutation grid; increase iterations"
                .into(),
        );
        for bin in 0..n_bins {
            bounds[bin] = sorted[bin * iterations + iterations - 1];
        }
    } else {
        // Walk down the sorted null, one column per pass, until at least
        // `out_crit` permutations have ANY bin at or above the bound. Here
        // 1 <= out_crit < iterations, so the first pass always runs and
        // sets every bound.
        let mut cutoff = iterations as isize - 1;
        while numout < out_crit && cutoff >= 0 {
            for bin in 0..n_bins {
                bounds[bin] = sorted[bin * iterations + cutoff as usize] + BOUND_OFFSET;
            }
            // Count the permutations in which any bin reaches its bound.
            numout = 0;
            for it in 0..iterations {
                let mut any_bin_exceeds = false;
                for bin in 0..n_bins {
                    let v = null_mat[bin * iterations + it];
                    if v.is_finite() && v >= bounds[bin] {
                        any_bin_exceeds = true;
                        break;
                    }
                }
                if any_bin_exceeds {
                    numout += 1;
                }
            }
            cutoff -= 1;
        }
    }

    let achieved = numout as f64 / iterations as f64;
    if warning.is_none() && (alpha - achieved).abs() > ALPHA_MISMATCH_TOLERANCE {
        warning = Some(format!(
            "achieved alpha {:.4} differs from requested {:.4} by more than 0.01; \
             increase iterations or de-noise the data",
            achieved, alpha
        ));
    }

    let sigbins = true_stat
        .iter()
        .zip(bounds.iter())
        .map(|(t, b)| t.is_finite() && t.abs() >= *b)
        .collect();
    Ok(PermTest {
        sigbins,
        acceptance_bounds: bounds,
        true_stat,
        n_excluded: numout,
        warning,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sort_places_nan_last_like_matlab() {
        let mut values = vec![f64::NAN, 3.0, 1.0, f64::NAN, 2.0, 0.0, f64::NAN, 1.0];
        sort_ascending_nan_last(&mut values);
        assert_eq!(&values[..5], &[0.0, 1.0, 1.0, 2.0, 3.0]);
        assert!(values[5].is_nan());
        assert!(values[6].is_nan());
        assert!(values[7].is_nan());
    }

    /// A bin that is mostly missing gives NaN null entries on the shuffles
    /// that leave one side of it empty. The test must still run, and the
    /// fully observed bin must keep a finite bound.
    #[test]
    fn runs_when_a_bin_has_nan_null_entries() {
        let n_bins = 2;
        let n_obs = 6;
        // Bin 0 is fully observed. Bin 1 has one finite value per group.
        let mut g1 = vec![f64::NAN; n_bins * n_obs];
        let mut g2 = vec![f64::NAN; n_bins * n_obs];
        for obs in 0..n_obs {
            g1[obs] = obs as f64;
            g2[obs] = obs as f64 + 0.5;
        }
        g1[n_obs] = 1.0;
        g2[n_obs] = 2.0;

        let result = gpermtest(&g1, &g2, n_bins, 0.05, 500, 7).unwrap();
        assert!(result.acceptance_bounds[0].is_finite());
        assert!(!result.sigbins[1]);
    }

    fn two_groups(n_bins: usize, n: usize, shift_bin: Option<usize>) -> (Vec<f64>, Vec<f64>) {
        let mut g1 = Vec::new();
        let mut g2 = Vec::new();
        for bin in 0..n_bins {
            for k in 0..n {
                let base = ((bin * 13 + k * 7) % 11) as f64 * 0.1;
                g1.push(base + if Some(bin) == shift_bin { 8.0 } else { 0.0 });
                g2.push(base);
            }
        }
        (g1, g2)
    }

    #[test]
    fn finds_a_planted_effect_and_leaves_the_rest() {
        let n_bins = 6;
        let n = 24;
        let (g1, g2) = two_groups(n_bins, n, Some(3));
        let r = gpermtest(&g1, &g2, n_bins, 0.05, 2000, 42).unwrap();
        assert!(r.sigbins[3], "planted bin must exceed the global bound");
        assert_eq!(r.sigbins.iter().filter(|s| **s).count(), 1);
    }

    #[test]
    fn null_data_yields_no_significant_bins() {
        let n_bins = 6;
        let n = 24;
        let (g1, g2) = two_groups(n_bins, n, None);
        let r = gpermtest(&g1, &g2, n_bins, 0.05, 2000, 7).unwrap();
        assert_eq!(r.sigbins.iter().filter(|s| **s).count(), 0);
    }

    /// The bound controls the FAMILY-wise rate: the share of
    /// permutations with any bin exceeding it should land near alpha.
    #[test]
    fn achieved_alpha_tracks_the_request() {
        let n_bins = 8;
        let n = 20;
        let (g1, g2) = two_groups(n_bins, n, None);
        let iterations = 4000;
        let r = gpermtest(&g1, &g2, n_bins, 0.05, iterations, 11).unwrap();
        let achieved = r.n_excluded as f64 / iterations as f64;
        assert!(
            (achieved - 0.05).abs() < 0.02,
            "achieved {} for alpha 0.05",
            achieved
        );
    }

    #[test]
    fn is_reproducible_for_a_given_seed() {
        let n_bins = 4;
        let n = 16;
        let (g1, g2) = two_groups(n_bins, n, Some(1));
        let a = gpermtest(&g1, &g2, n_bins, 0.05, 500, 99).unwrap();
        let b = gpermtest(&g1, &g2, n_bins, 0.05, 500, 99).unwrap();
        assert_eq!(a.acceptance_bounds, b.acceptance_bounds);
        assert_eq!(a.n_excluded, b.n_excluded);
    }

    #[test]
    fn rejects_degenerate_inputs() {
        let (g1, g2) = two_groups(2, 4, None);
        assert!(gpermtest(&g1, &g2, 0, 0.05, 10, 1).is_err());
        assert!(gpermtest(&g1, &g2, 2, 0.0, 10, 1).is_err());
        assert!(gpermtest(&g1, &g2, 2, 0.05, 0, 1).is_err());
    }
}
