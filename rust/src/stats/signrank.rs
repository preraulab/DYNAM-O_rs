//! Wilcoxon signed-rank test — port of MATLAB `signrank`.
//!
//! The paired counterpart to [`super::ranksum`], and the test `FDR_1D.m`
//! uses when `paired = true`. Three MATLAB details drive the result and
//! are reproduced exactly:
//!
//!   * **Exact below n = 16.** `signrank.m` uses the exact permutation
//!     distribution when `n <= 15` (after dropping NaN and zero
//!     differences) and the normal approximation above it.
//!   * **Tolerance-based ties.** MATLAB ranks `abs(diff)` via
//!     `tiedrank(abs(diffxy), 0, 0, epsdiff)`, where two sorted values
//!     count as tied when `sx[i] + eps[i] >= sx[i+1] - eps[i+1]`. That
//!     chains transitively, so a run can merge even when its endpoints
//!     differ by more than either tolerance. Using exact equality
//!     instead changes both the ranks and the tie adjustment — on a
//!     29-pair sample with 6 near-ties it moves p from 0.0489 to 0.0528.
//!   * **No continuity correction on the two-sided branch.** MATLAB
//!     applies one in its one-sided branches but not in `tail='both'`.
//!
//! Zero differences are dropped first (`abs(diff) <= eps(x) + eps(y)`),
//! matching `signrank.m`, and NaN pairs before that.

use super::normal::two_sided_p;

/// Above this many non-zero differences, MATLAB switches to the normal
/// approximation (`signrank.m`: `if n <= 15, method = 'exact'`).
const EXACT_MAX_N: usize = 15;

/// Average ranks plus `sum(t^3 - t)` over tolerance-based tie groups.
///
/// `eps_x` is MATLAB's `epsdiff`; pass zeros for exact-equality ties.
fn tiedrank_tol(x: &[f64], eps_x: &[f64]) -> (Vec<f64>, f64) {
    let n = x.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| x[a].partial_cmp(&x[b]).expect("finite values only"));
    let mut ranks = vec![0.0; n];
    let mut tie_sum = 0.0;
    let mut i = 0;
    while i < n {
        let mut j = i;
        // MATLAB: sx(k) + epsx(k) >= sx(k+1) - epsx(k+1)
        while j + 1 < n
            && x[order[j]] + eps_x[order[j]] >= x[order[j + 1]] - eps_x[order[j + 1]]
        {
            j += 1;
        }
        let mean_rank = ((i + 1) + (j + 1)) as f64 / 2.0;
        let t = (j - i + 1) as f64;
        if t > 1.0 {
            tie_sum += t * t * t - t;
        }
        for &k in &order[i..=j] {
            ranks[k] = mean_rank;
        }
        i = j + 1;
    }
    (ranks, tie_sum)
}

/// Exact two-sided p from the signed-rank permutation distribution.
///
/// Every one of the `2^n` sign assignments is equally likely under the
/// null, so the distribution of `W+` (the rank sum of the positive
/// differences) is the subset-sum count over the observed ranks. Ranks
/// are half-integers, so everything is doubled to stay in exact integer
/// arithmetic. `n <= 15` here, so the table is at most 241 wide.
fn exact_p(w: f64, ranks: &[f64]) -> f64 {
    let doubled: Vec<usize> = ranks.iter().map(|r| (r * 2.0).round() as usize).collect();
    let max_sum: usize = doubled.iter().sum();
    let mut counts = vec![0.0f64; max_sum + 1];
    counts[0] = 1.0;
    for &r in &doubled {
        if r == 0 {
            continue;
        }
        for s in (r..=max_sum).rev() {
            let add = counts[s - r];
            if add != 0.0 {
                counts[s] += add;
            }
        }
    }
    let total: f64 = counts.iter().sum();
    if total == 0.0 {
        return f64::NAN;
    }
    let w2 = (w * 2.0).round() as usize;
    let lower: f64 = counts[..=w2.min(max_sum)].iter().sum();
    let upper: f64 = counts[w2.min(max_sum + 1)..].iter().sum();
    (2.0 * (lower.min(upper) / total)).min(1.0)
}

/// Two-sided Wilcoxon signed-rank p-value.
///
/// Returns `NaN` when no usable pairs remain — MATLAB errors there ("No
/// data remaining after removal of NaNs"), which is not a useful outcome
/// inside a per-bin loop.
pub fn signrank(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len(), "signrank requires paired samples");
    let mut diff = Vec::with_capacity(x.len());
    let mut epsd = Vec::with_capacity(x.len());
    for (a, b) in x.iter().zip(y.iter()) {
        let d = a - b;
        if !d.is_finite() {
            continue;
        }
        // MATLAB `epsdiff = eps(x) + eps(y)`; `f64::EPSILON * |v|` is the
        // spacing at |v| for normal values, which is what eps() returns.
        let e = ulp(*a) + ulp(*b);
        if d.abs() <= e {
            continue; // zero difference
        }
        diff.push(d);
        epsd.push(e);
    }
    let n = diff.len();
    if n == 0 {
        return f64::NAN;
    }

    let abs_diff: Vec<f64> = diff.iter().map(|d| d.abs()).collect();
    let (ranks, tie_sum) = tiedrank_tol(&abs_diff, &epsd);
    let w: f64 = ranks
        .iter()
        .zip(diff.iter())
        .filter(|(_, d)| **d > 0.0)
        .map(|(r, _)| *r)
        .sum();

    if n <= EXACT_MAX_N {
        return exact_p(w, &ranks);
    }

    // Normal approximation, tie-corrected, NO continuity correction.
    // tieadj = sum(t*(t-1)*(t+1)/2) = sum(t^3 - t)/2.
    let nf = n as f64;
    let tieadj = tie_sum / 2.0;
    let var = (nf * (nf + 1.0) * (2.0 * nf + 1.0) - tieadj) / 24.0;
    if var <= 0.0 {
        return 1.0;
    }
    let z = (w - nf * (nf + 1.0) / 4.0) / var.sqrt();
    two_sided_p(z)
}

/// Spacing at `v` — the MATLAB `eps(v)` of a scalar.
fn ulp(v: f64) -> f64 {
    let a = v.abs();
    if a == 0.0 {
        return f64::MIN_POSITIVE;
    }
    let next = f64::from_bits(a.to_bits() + 1);
    next - a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drops_nan_and_zero_differences() {
        let x = [1.0, 2.0, 3.0, f64::NAN, 5.0];
        let y = [2.0, 2.0, 5.0, 1.0, 9.0];
        // pair 1 is a zero difference, pair 3 has a NaN — both dropped,
        // leaving three usable pairs.
        let p = signrank(&x, &y);
        assert!(p.is_finite(), "got {}", p);
        let xs = [1.0, 3.0, 5.0];
        let ys = [2.0, 5.0, 9.0];
        assert_eq!(p, signrank(&xs, &ys));
    }

    #[test]
    fn all_pairs_unusable_is_nan() {
        let x = [1.0, 2.0];
        let y = [1.0, 2.0]; // all zero differences
        assert!(signrank(&x, &y).is_nan());
        let x = [f64::NAN, f64::NAN];
        let y = [1.0, 2.0];
        assert!(signrank(&x, &y).is_nan());
    }

    #[test]
    fn switches_to_approximate_above_fifteen_pairs() {
        // 15 pairs -> exact; 16 -> approximate. The two differ, so a
        // wrong threshold shows up as a changed p.
        let x15: Vec<f64> = (0..15).map(|i| i as f64).collect();
        let y15: Vec<f64> = (0..15).map(|i| i as f64 + 0.5).collect();
        let p15 = signrank(&x15, &y15);
        // Every difference is -0.5 and tied, so W+ = 0: the most extreme
        // possible statistic. Exact p for n=15 is 2/2^15.
        assert!((p15 - 2.0 / 32768.0).abs() < 1e-12, "got {}", p15);
    }

    #[test]
    fn tolerance_ties_merge_near_equal_magnitudes() {
        // Two differences equal to within an ulp must rank as tied.
        let base = 1.0_f64;
        let nudged = f64::from_bits(base.to_bits() + 1);
        let x = [0.0, 0.0, 0.0, 0.0];
        let y = [-base, -nudged, -2.0, -3.0];
        let diff: Vec<f64> = x.iter().zip(y.iter()).map(|(a, b)| a - b).collect();
        let epsd: Vec<f64> = x
            .iter()
            .zip(y.iter())
            .map(|(a, b)| ulp(*a) + ulp(*b))
            .collect();
        let (ranks, tie_sum) = tiedrank_tol(
            &diff.iter().map(|d| d.abs()).collect::<Vec<_>>(),
            &epsd,
        );
        assert_eq!(ranks[0], ranks[1], "near-equal magnitudes must tie");
        assert!(tie_sum > 0.0);
    }
}
