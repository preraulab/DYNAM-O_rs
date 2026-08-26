//! Per-bin group comparison with FDR control — port of `FDR_1D.m` /
//! `FDR_2D.m`.
//!
//! `FDR_2D` is `FDR_1D` over a flattened grid, so both live here and the
//! caller decides the shape. Data is `(n_bins, n_trials)`, one row per
//! bin, one column per subject.
//!
//! **Missing data is dropped, never imputed.** A SOPH bin is a peak
//! *rate* whose NaN means "under `min_time_in_bin`, rate undefined",
//! while a real `0` means "occupied the bin, no peaks". Filling NaN with
//! 0 collapses the first onto the second, and since the values are
//! non-negative the injected zeros sit at the bottom of every rank
//! ordering and swamp the real observations. A bin with nothing left to
//! compare is untestable and yields NaN; [`super::fdr::adjust`] keeps
//! those out of the family entirely.

use super::fdr;
use super::ranksum::ranksum;
use super::signrank::signrank;

/// Which test to run per bin.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Test {
    /// Wilcoxon rank-sum (MATLAB `nonparam=true, paired=false`).
    RankSum,
    /// Wilcoxon signed-rank (MATLAB `nonparam=true, paired=true`).
    SignRank,
}

impl Test {
    /// Resolve MATLAB's `(nonparam, paired)` pair. The parametric
    /// branches are deliberately not implemented: `FDR_1D`'s default is
    /// `nonparam = true`, DYNAM-O never calls it otherwise, and a
    /// half-used t-test path would be one more thing to keep in parity.
    pub fn from_matlab_flags(nonparam: bool, paired: bool) -> Result<Self, String> {
        if !nonparam {
            return Err(
                "parametric (nonparam=false) FDR is not implemented in the kernel; \
                 use the MATLAB toolbox for t-test comparisons"
                    .into(),
            );
        }
        Ok(if paired { Test::SignRank } else { Test::RankSum })
    }
}

/// Result of an FDR-controlled comparison over a bin grid.
#[derive(Debug, Clone)]
pub struct FdrGrid {
    /// `p_adj < q`, per bin. False wherever the bin was untestable.
    pub sigbins: Vec<bool>,
    /// Adjusted p per bin; NaN where untestable. Not clipped to 1 — BY
    /// can exceed it and MATLAB does not clip either.
    pub p_adj: Vec<f64>,
    /// Raw p per bin; NaN where untestable.
    pub p_values: Vec<f64>,
    /// Largest raw p that passes, or 0.0 when none do.
    pub crit_p: f64,
    /// Family size — testable bins only.
    pub m: usize,
}

/// Is there anything left to compare in this bin?
///
/// Unpaired needs at least one observation in each group; paired needs
/// at least one position where BOTH are observed, because `signrank`
/// has nothing to rank otherwise.
fn testable(a: &[f64], b: &[f64], test: Test) -> bool {
    match test {
        Test::RankSum => {
            a.iter().any(|v| v.is_finite()) && b.iter().any(|v| v.is_finite())
        }
        Test::SignRank => a
            .iter()
            .zip(b.iter())
            .any(|(x, y)| x.is_finite() && y.is_finite()),
    }
}

/// Per-bin comparison of two groups, then FDR control over the grid.
///
/// `g1` / `g2` are row-major `(n_bins, n_trials)`. Inf is treated as
/// missing, matching `FDR_1D.m`'s `group(isinf(group)) = nan` on entry.
pub fn fdr_grid(
    g1: &[f64],
    g2: &[f64],
    n_bins: usize,
    q: f64,
    method: fdr::Method,
    test: Test,
) -> Result<FdrGrid, String> {
    if n_bins == 0 {
        return Err("n_bins must be non-zero".into());
    }
    if g1.len() % n_bins != 0 || g2.len() % n_bins != 0 {
        return Err(format!(
            "group lengths {} / {} are not multiples of n_bins {}",
            g1.len(),
            g2.len(),
            n_bins
        ));
    }
    let n1 = g1.len() / n_bins;
    let n2 = g2.len() / n_bins;
    if test == Test::SignRank && n1 != n2 {
        return Err(format!(
            "a paired test needs the same number of observations in each group ({} vs {})",
            n1, n2
        ));
    }

    let clean = |v: f64| if v.is_finite() { v } else { f64::NAN };
    let mut p_values = Vec::with_capacity(n_bins);
    let mut buf_a: Vec<f64> = Vec::with_capacity(n1);
    let mut buf_b: Vec<f64> = Vec::with_capacity(n2);
    for bin in 0..n_bins {
        buf_a.clear();
        buf_b.clear();
        buf_a.extend(g1[bin * n1..(bin + 1) * n1].iter().map(|v| clean(*v)));
        buf_b.extend(g2[bin * n2..(bin + 1) * n2].iter().map(|v| clean(*v)));
        p_values.push(if testable(&buf_a, &buf_b, test) {
            match test {
                Test::RankSum => ranksum(&buf_a, &buf_b).p,
                Test::SignRank => signrank(&buf_a, &buf_b),
            }
        } else {
            f64::NAN
        });
    }

    let r = fdr::adjust(&p_values, q, method);
    let sigbins = r
        .p_adj
        .iter()
        .map(|p| p.is_finite() && *p < q)
        .collect();
    Ok(FdrGrid {
        sigbins,
        p_adj: r.p_adj,
        p_values,
        crit_p: r.crit_p,
        m: r.m,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn planted_effect_is_found_and_nothing_else_is() {
        // 4 bins x 20 subjects; only bin 2 differs.
        let (n_bins, n) = (4usize, 20usize);
        let mut g1 = Vec::new();
        let mut g2 = Vec::new();
        for bin in 0..n_bins {
            for k in 0..n {
                let base = (k % 7) as f64 * 0.01;
                g1.push(base + if bin == 2 { 10.0 } else { 1.0 });
                g2.push(base + 1.0);
            }
        }
        let r = fdr_grid(&g1, &g2, n_bins, 0.05, fdr::Method::Dep, Test::RankSum).unwrap();
        assert!(r.sigbins[2], "planted bin must survive");
        assert_eq!(r.sigbins.iter().filter(|s| **s).count(), 1);
        assert_eq!(r.m, n_bins);
    }

    #[test]
    fn untestable_bins_leave_the_family() {
        let (n_bins, n) = (3usize, 10usize);
        let mut g1 = Vec::new();
        let mut g2 = Vec::new();
        for bin in 0..n_bins {
            for k in 0..n {
                // bin 1: group 1 entirely unobserved -> untestable
                g1.push(if bin == 1 { f64::NAN } else { k as f64 });
                g2.push(k as f64 + 5.0);
            }
        }
        let r = fdr_grid(&g1, &g2, n_bins, 0.05, fdr::Method::Dep, Test::RankSum).unwrap();
        assert!(r.p_values[1].is_nan(), "got {}", r.p_values[1]);
        assert!(!r.sigbins[1]);
        assert_eq!(r.m, 2, "only the testable bins form the family");
    }

    #[test]
    fn paired_requires_matching_group_sizes() {
        let g1 = vec![1.0; 6];
        let g2 = vec![1.0; 9];
        let err = fdr_grid(&g1, &g2, 3, 0.05, fdr::Method::Dep, Test::SignRank).unwrap_err();
        assert!(err.contains("paired"), "{}", err);
    }

    #[test]
    fn inf_is_treated_as_missing() {
        let n_bins = 1;
        let g1 = vec![1.0, 2.0, f64::INFINITY, 4.0];
        let g2 = vec![5.0, 6.0, 7.0, 8.0];
        let with_inf = fdr_grid(&g1, &g2, n_bins, 0.05, fdr::Method::Dep, Test::RankSum).unwrap();
        let g1b = vec![1.0, 2.0, f64::NAN, 4.0];
        let with_nan = fdr_grid(&g1b, &g2, n_bins, 0.05, fdr::Method::Dep, Test::RankSum).unwrap();
        assert_eq!(with_inf.p_values[0], with_nan.p_values[0]);
    }

    #[test]
    fn parametric_is_refused_rather_than_approximated() {
        let err = Test::from_matlab_flags(false, false).unwrap_err();
        assert!(err.contains("not implemented"), "{}", err);
        assert_eq!(Test::from_matlab_flags(true, false).unwrap(), Test::RankSum);
        assert_eq!(Test::from_matlab_flags(true, true).unwrap(), Test::SignRank);
    }
}
