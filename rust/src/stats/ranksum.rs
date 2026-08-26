//! Wilcoxon rank-sum (Mann–Whitney U) test — a faithful port of MATLAB's
//! `ranksum`, including its choice of the exact test for small samples.
//!
//! Port fidelity matters here because the desktop app and the MATLAB
//! toolbox are supposed to produce the same SOPH comparison from the same
//! data. Two details drive that:
//!
//!   * **Exact vs approximate.** MATLAB uses the exact permutation
//!     distribution when `n1 + n2 < 20` and the tie-corrected normal
//!     approximation otherwise. The two disagree by a few percent at small
//!     n — e.g. 9 vs 5 gives exact 0.018981, approximate 0.023411 — which
//!     straddles a q = 0.02 threshold. The TypeScript implementation this
//!     replaces always approximated, so it silently diverged from MATLAB
//!     on exactly the thinly-observed pixels at the edge of a SOPH.
//!   * **Missing data** is dropped, never imputed. A SOPH pixel's NaN
//!     means "under min_time_in_bin, rate undefined", which is a different
//!     statement from an observed rate of 0.
//!
//! Ties get average ranks and the tie-corrected variance, matching
//! `tiedrank`.

use super::normal::two_sided_p;

/// Sample-size total below which MATLAB `ranksum` uses the exact test.
///
/// `ranksum.m` writes the rule as `(ns < 10) && ((nx + ny) < 20)` with
/// `ns = min(nx, ny)`, but the first clause is implied by the second —
/// `2·ns <= nx + ny < 20` forces `ns < 10` — so the total alone decides
/// it. Confirmed empirically against R2025a over a 30-case grid.
const EXACT_MAX_TOTAL: usize = 20;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Method {
    Exact,
    Approximate,
}

#[derive(Debug, Clone, Copy)]
pub struct RankSum {
    /// Two-sided p-value. NaN when either sample is empty.
    pub p: f64,
    /// Mann–Whitney U (the smaller of U₁, U₂). NaN when undefined.
    pub u: f64,
    /// z statistic; NaN under the exact method.
    pub z: f64,
    pub n1: usize,
    pub n2: usize,
    pub method: Method,
}

impl RankSum {
    fn empty(n1: usize, n2: usize) -> Self {
        Self { p: f64::NAN, u: f64::NAN, z: f64::NAN, n1, n2, method: Method::Exact }
    }
}

/// Average ranks (`tiedrank`) plus Σ(t³ − t) over tie groups.
fn tiedrank(xs: &[f64]) -> (Vec<f64>, f64) {
    let n = xs.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| xs[a].partial_cmp(&xs[b]).expect("non-finite values are filtered first"));
    let mut ranks = vec![0.0; n];
    let mut tie_sum = 0.0;
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && xs[idx[j + 1]] == xs[idx[i]] {
            j += 1;
        }
        let mean_rank = ((i + 1) + (j + 1)) as f64 / 2.0;
        let t = (j - i + 1) as f64;
        if t > 1.0 {
            tie_sum += t * t * t - t;
        }
        for &k in &idx[i..=j] {
            ranks[k] = mean_rank;
        }
        i = j + 1;
    }
    (ranks, tie_sum)
}

/// Exact two-sided p from the permutation distribution of the rank sum,
/// matching MATLAB's `full_enumeration` branch:
///
/// ```matlab
/// allpos   = nchoosek(ranks, ns);       % all C(N, ns) position subsets
/// sumranks = sum(allpos, 2);
/// plo = sum(sumranks <= w)/np;  phi = sum(sumranks >= w)/np;
/// p   = min(2*min(plo, phi), 1);
/// ```
///
/// The subsets are drawn from the OBSERVED ranks, so ties are handled by
/// permuting the tied (fractional) rank values rather than the integers
/// 1..N — enumerating 1..N instead silently gives a different answer
/// whenever two observations tie. Counting is done by dynamic programming
/// rather than materialising C(N, ns) rows; ranks are half-integers, so
/// everything is doubled to stay in exact integer arithmetic.
///
/// MATLAB switches to a network algorithm for 10 <= N < 20 purely for
/// speed — it computes the same exact conditional distribution, and N is
/// under 20 here, so the DP table is at most 20 × 381.
fn exact_p(w: f64, ranks: &[f64], ns: usize) -> f64 {
    // Ranks are integers or half-integers; ×2 makes them exact integers.
    let doubled: Vec<usize> = ranks.iter().map(|r| (r * 2.0).round() as usize).collect();
    let max_sum: usize = doubled.iter().sum();
    // counts[k][s] = number of size-k position subsets with doubled sum s.
    let mut counts = vec![vec![0.0f64; max_sum + 1]; ns + 1];
    counts[0][0] = 1.0;
    for (taken, &r) in doubled.iter().enumerate() {
        // Descending k so each position is used at most once; k can't
        // exceed the number of positions seen so far.
        for k in (1..=ns.min(taken + 1)).rev() {
            for s in (r..=max_sum).rev() {
                let add = counts[k - 1][s - r];
                if add != 0.0 {
                    counts[k][s] += add;
                }
            }
        }
    }
    let total: f64 = counts[ns].iter().sum();
    if total == 0.0 {
        return f64::NAN;
    }
    let w2 = (w * 2.0).round() as usize;
    let mut lower = 0.0;
    let mut upper = 0.0;
    for (s, &c) in counts[ns].iter().enumerate() {
        if s <= w2 {
            lower += c;
        }
        if s >= w2 {
            upper += c;
        }
    }
    (2.0 * (lower.min(upper) / total)).min(1.0)
}

/// Two-sided rank-sum test with MATLAB's default method selection.
pub fn ranksum(g1: &[f64], g2: &[f64]) -> RankSum {
    ranksum_with(g1, g2, None)
}

/// As [`ranksum`], but `method` can force the exact or approximate branch
/// (MATLAB's `'method'` name-value pair). `None` applies the default rule.
pub fn ranksum_with(g1: &[f64], g2: &[f64], method: Option<Method>) -> RankSum {
    let x: Vec<f64> = g1.iter().copied().filter(|v| v.is_finite()).collect();
    let y: Vec<f64> = g2.iter().copied().filter(|v| v.is_finite()).collect();
    let (n1, n2) = (x.len(), y.len());
    if n1 == 0 || n2 == 0 {
        return RankSum::empty(n1, n2);
    }
    let n = n1 + n2;

    // MATLAB ranks the pooled sample and sums the ranks of the SMALLER
    // group; the p-value is symmetric either way, but following the same
    // convention keeps the intermediate statistics comparable when
    // debugging against MATLAB.
    let (small, large, ns, nl) = if n1 <= n2 { (&x, &y, n1, n2) } else { (&y, &x, n2, n1) };
    let mut pooled: Vec<f64> = Vec::with_capacity(n);
    pooled.extend_from_slice(small);
    pooled.extend_from_slice(large);
    let (ranks, tie_sum) = tiedrank(&pooled);
    let w: f64 = ranks[..ns].iter().sum();

    // U for the FIRST group as passed in, so `u` doesn't depend on which
    // sample happened to be smaller.
    let r1: f64 = if n1 <= n2 {
        w
    } else {
        ranks[ns..].iter().sum()
    };
    let u1 = r1 - (n1 * (n1 + 1)) as f64 / 2.0;
    let u = u1.min((n1 * n2) as f64 - u1);

    let use_exact = match method {
        Some(m) => m == Method::Exact,
        None => n < EXACT_MAX_TOTAL,
    };

    if use_exact {
        return RankSum {
            p: exact_p(w, &ranks, ns),
            u,
            z: f64::NAN,
            n1,
            n2,
            method: Method::Exact,
        };
    }

    // Normal approximation, tie-corrected variance, continuity correction
    // — MATLAB `ranksum.m`:
    //   wmean   = ns(ns + nl + 1)/2
    //   tiescor = 2·tieadj / (N(N − 1)),  tieadj = Σ(t³ − t)/2
    //   wvar    = ns·nl·((N + 1) − tiescor)/12
    //   z       = (wc − 0.5·sign(wc)) / √wvar
    let nsf = ns as f64;
    let nlf = nl as f64;
    let nf = n as f64;
    let w_mean = nsf * (nf + 1.0) / 2.0;
    let tie_cor = tie_sum / (nf * (nf - 1.0));
    let w_var = nsf * nlf * ((nf + 1.0) - tie_cor) / 12.0;
    if w_var <= 0.0 {
        // Every value tied: no evidence of any difference.
        return RankSum { p: 1.0, u, z: 0.0, n1, n2, method: Method::Approximate };
    }
    let wc = w - w_mean;
    // sign(0) == 0 in MATLAB, so a statistic exactly at the mean gets NO
    // continuity correction. Writing this as an `if wc < 0 {+0.5} else
    // {-0.5}` — as the TypeScript version did — nudges z off zero for a
    // perfectly null pixel.
    let corr = if wc > 0.0 {
        -0.5
    } else if wc < 0.0 {
        0.5
    } else {
        0.0
    };
    let z = (wc + corr) / w_var.sqrt();
    RankSum { p: two_sided_p(z), u, z, n1, n2, method: Method::Approximate }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference p-values printed by MATLAB R2025a `ranksum` at default
    /// method (`format long`). Both of these have ties, which is the case
    /// that distinguishes enumerating the observed ranks from enumerating
    /// 1..N.
    #[test]
    fn matches_matlab_reference_cases() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0];
        let b = [2.0, 3.0, 4.0, 5.0, 6.0];
        let r = ranksum(&a, &b);
        assert_eq!(r.method, Method::Exact, "n = 10 must take the exact branch");
        assert!((r.p - 0.44444444444444442).abs() < 1e-12, "got {}", r.p);

        let a = [1.0, 2.0, 2.0, 3.0];
        let b = [2.0, 3.0, 3.0, 4.0];
        let r = ranksum(&a, &b);
        assert!((r.p - 0.28571428571428570).abs() < 1e-12, "got {}", r.p);
    }

    /// The switch happens on the TOTAL, verified against R2025a.
    #[test]
    fn method_switches_on_combined_sample_size() {
        let a = vec![1.0; 9];
        let b: Vec<f64> = (0..10).map(|i| 2.0 + i as f64).collect();
        assert_eq!(ranksum(&a, &b).method, Method::Exact, "9 + 10 = 19 -> exact");
        let b: Vec<f64> = (0..11).map(|i| 2.0 + i as f64).collect();
        assert_eq!(ranksum(&a, &b).method, Method::Approximate, "9 + 11 = 20 -> approximate");
    }

    #[test]
    fn missing_values_are_dropped_not_imputed() {
        let a = [1.0, 2.0, f64::NAN, 3.0, f64::INFINITY];
        let b = [2.0, 3.0, 4.0, f64::NAN];
        let r = ranksum(&a, &b);
        assert_eq!((r.n1, r.n2), (3, 3));
        let clean_a = [1.0, 2.0, 3.0];
        let clean_b = [2.0, 3.0, 4.0];
        assert_eq!(r.p, ranksum(&clean_a, &clean_b).p);
    }

    #[test]
    fn empty_after_dropping_yields_nan() {
        let a = [f64::NAN, f64::NAN];
        let b = [1.0, 2.0, 3.0];
        assert!(ranksum(&a, &b).p.is_nan());
        assert!(ranksum(&b, &a).p.is_nan());
    }

    /// A two-sample test cannot depend on the order of its arguments —
    /// this is the invariant the MATLAB zero-fill used to break.
    #[test]
    fn p_is_symmetric_in_its_arguments() {
        let a = [4.0, 1.5, 9.2, 3.3, f64::NAN, 7.7, 2.0, 8.1, 5.5, 6.0, 1.1, 3.9];
        let b = [2.2, 6.6, 1.0, 8.8, 4.4, 9.9, 3.0, 5.0, 7.0, 2.5];
        assert_eq!(ranksum(&a, &b).p, ranksum(&b, &a).p);
        // ... and in the approximate branch too.
        let big_a: Vec<f64> = (0..40).map(|i| (i as f64 * 0.7).sin()).collect();
        let big_b: Vec<f64> = (0..35).map(|i| (i as f64 * 0.9).cos() + 0.3).collect();
        assert_eq!(ranksum(&big_a, &big_b).p, ranksum(&big_b, &big_a).p);
    }

    #[test]
    fn all_values_tied_gives_p_one() {
        let a = vec![2.0; 20];
        let b = vec![2.0; 20];
        assert_eq!(ranksum(&a, &b).p, 1.0);
    }

    /// n = 2 vs 2 bottoms out at 1/3 — the reason a thinly-observed pixel
    /// can never be flagged, whatever the effect size.
    #[test]
    fn tiny_samples_cannot_reach_significance() {
        let a = [100.0, 200.0];
        let b = [0.001, 0.002];
        let r = ranksum(&a, &b);
        assert_eq!(r.method, Method::Exact);
        assert!((r.p - 1.0 / 3.0).abs() < 1e-12, "got {}", r.p);
    }
}
