//! Benjamini–Hochberg and Benjamini–Yekutieli FDR control. Port of the
//! `fdr_bh` nested in DYNAM-O's `FDR_1D.m` (itself David Groppe's
//! `fdr_bh.m`).
//!
//! References:
//!   Benjamini & Hochberg (1995), J. R. Stat. Soc. B 57(1)
//!   Benjamini & Yekutieli (2001), Ann. Stat. 29(4)
//!
//! Non-finite p-values are excluded before the family size `m` is counted
//! and reinstated as NaN in the output. That is load-bearing for SOPH
//! maps, where roughly half the pixels are untestable: counting them would
//! inflate `m` and, under BY especially, cost real detections.

/// Default false-discovery rate, matching `FDR_1D.m` / `FDR_2D.m`.
/// Every caller that doesn't take an explicit `q` from the user uses this,
/// so the app and the toolbox threshold at the same level by default.
/// cbindgen:ignore
pub const DEFAULT_Q: f64 = 0.05;

/// Which dependence structure the correction is valid under.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Method {
    /// Benjamini–Hochberg: independent or positively dependent tests.
    /// MATLAB `'pdep'`, `FDR_2D`'s `'independent'`.
    Pdep,
    /// Benjamini–Yekutieli: any dependence structure. MATLAB `'dep'`,
    /// `FDR_2D`'s `'dependent'` and its default. Neighboring SOPH pixels
    /// are strongly correlated, so this is what the app uses.
    Dep,
}

impl Method {
    /// Parse the MATLAB spellings, both the `fdr_bh` short form and the
    /// `FDR_2D` long form.
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "pdep" | "independent" => Some(Method::Pdep),
            "dep" | "dependent" => Some(Method::Dep),
            _ => None,
        }
    }
}

/// Output of [`adjust`].
#[derive(Debug, Clone)]
pub struct FdrResult {
    /// Adjusted p-values, aligned with the input. NaN wherever the input
    /// was non-finite. Not clipped to 1, because BY can exceed it and
    /// MATLAB doesn't clip either.
    pub p_adj: Vec<f64>,
    /// Largest raw p-value that passes, or 0 when nothing does.
    pub crit_p: f64,
    /// Tests that survive at `q`.
    pub n_sig: usize,
    /// Family size: finite p-values only.
    pub m: usize,
}

/// Apply FDR control at level `q`.
pub fn adjust(pvals: &[f64], q: f64, method: Method) -> FdrResult {
    // (original index, p-value) for every finite p-value, sorted ascending
    // by p-value below.
    let mut order: Vec<(usize, f64)> = Vec::new();
    for (i, p) in pvals.iter().enumerate() {
        if p.is_finite() {
            order.push((i, *p));
        }
    }
    let m = order.len();
    let mut p_adj = vec![f64::NAN; pvals.len()];
    if m == 0 {
        return FdrResult { p_adj, crit_p: 0.0, n_sig: 0, m: 0 };
    }

    order.sort_by(|a, b| a.1.partial_cmp(&b.1).expect("filtered to finite"));
    let p_sorted: Vec<f64> = order.iter().map(|(_, p)| *p).collect();
    let mf = m as f64;

    // Denominator of the step-up thresholds. BH uses m. BY uses
    // m · Σ 1/k for k = 1..m.
    let denom = match method {
        Method::Pdep => mf,
        Method::Dep => {
            let mut harmonic_sum = 0.0_f64;
            for k in 1..=m {
                harmonic_sum += 1.0 / k as f64;
            }
            mf * harmonic_sum
        }
    };

    // For the k-th smallest p-value (k = 1..m):
    //   thresh = k * q / denom   (the rejection threshold)
    //   wtd_p  = denom * p / k   (the p-value scaled to the q axis)
    let mut thresh = Vec::with_capacity(m);
    let mut wtd_p = Vec::with_capacity(m);
    for k in 0..m {
        let rank = (k + 1) as f64;
        thresh.push(rank * q / denom);
        wtd_p.push(denom * p_sorted[k] / rank);
    }

    // Adjusted p in sorted order: the running minimum of `wtd_p` from the
    // top, which makes the sequence monotone non-decreasing. Implemented
    // as the same sort-and-fill Groppe uses so the two agree exactly,
    // including on ties. Each weighted p-value, taken in ascending order,
    // fills every still-empty slot up to and including its own position.
    let mut wtd_order: Vec<(usize, f64)> = wtd_p.iter().copied().enumerate().collect();
    wtd_order.sort_by(|a, b| a.1.partial_cmp(&b.1).expect("finite"));
    let mut adj_sorted = vec![f64::NAN; m];
    let mut next_fill = 0usize;
    for &(target, v) in &wtd_order {
        if target >= next_fill {
            for slot in &mut adj_sorted[next_fill..=target] {
                *slot = v;
            }
            next_fill = target + 1;
            if next_fill >= m {
                break;
            }
        }
    }

    // Largest sorted p-value at or below its threshold, or 0 if none is.
    let mut crit_p = 0.0;
    for k in 0..m {
        if p_sorted[k] <= thresh[k] {
            crit_p = p_sorted[k];
        }
    }

    let mut n_sig = 0usize;
    for (k, (orig_idx, _)) in order.iter().enumerate() {
        p_adj[*orig_idx] = adj_sorted[k];
        // The stored value is unclipped. Only the significance count clips
        // at 1.
        if adj_sorted[k].min(1.0) <= q {
            n_sig += 1;
        }
    }
    FdrResult { p_adj, crit_p, n_sig, m }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference from MATLAB R2025a:
    ///   p = [0.001 0.008 0.039 0.041 0.042 0.06 0.074 0.205 0.212 0.216];
    ///   [~,~,~,adj] = fdr_bh(p, 0.05, 'pdep')
    #[test]
    fn bh_matches_matlab() {
        let p = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216];
        let want = [0.01, 0.04, 0.084, 0.084, 0.084, 0.1, 0.105714285714286,
                    0.216, 0.216, 0.216];
        let r = adjust(&p, 0.05, Method::Pdep);
        for (got, want) in r.p_adj.iter().zip(want.iter()) {
            assert!((got - want).abs() < 1e-12, "got {:?} want {:?}", r.p_adj, want);
        }
        assert_eq!(r.m, 10);
    }

    /// Same p-values under BY: every adjusted value scales by Σ1/k = 2.928968…
    #[test]
    fn by_matches_matlab() {
        let p = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216];
        let r_bh = adjust(&p, 0.05, Method::Pdep);
        let r_by = adjust(&p, 0.05, Method::Dep);
        let harmonic: f64 = (1..=10).map(|k| 1.0 / k as f64).sum();
        for (bh, by) in r_bh.p_adj.iter().zip(r_by.p_adj.iter()) {
            assert!((by - bh * harmonic).abs() < 1e-12);
        }
    }

    /// The behavior SOPH maps depend on: untestable pixels must not
    /// count toward m, or half the map would dilute the other half.
    #[test]
    fn non_finite_p_are_excluded_from_the_family() {
        let with_nan = [0.001, f64::NAN, 0.008, f64::NAN, 0.039];
        let without = [0.001, 0.008, 0.039];
        let a = adjust(&with_nan, 0.05, Method::Dep);
        let b = adjust(&without, 0.05, Method::Dep);
        assert_eq!(a.m, 3);
        assert!(a.p_adj[1].is_nan() && a.p_adj[3].is_nan());
        assert_eq!(
            vec![a.p_adj[0], a.p_adj[2], a.p_adj[4]],
            b.p_adj,
            "the finite p-values must adjust as if the NaNs weren't there"
        );
    }

    #[test]
    fn all_nan_input_is_handled() {
        let r = adjust(&[f64::NAN, f64::NAN], 0.1, Method::Dep);
        assert_eq!((r.m, r.n_sig, r.crit_p), (0, 0, 0.0));
        assert!(r.p_adj.iter().all(|v| v.is_nan()));
    }

    #[test]
    fn method_parses_both_matlab_spellings() {
        assert_eq!(Method::parse("dep"), Some(Method::Dep));
        assert_eq!(Method::parse("dependent"), Some(Method::Dep));
        assert_eq!(Method::parse("PDEP"), Some(Method::Pdep));
        assert_eq!(Method::parse("independent"), Some(Method::Pdep));
        assert_eq!(Method::parse("nonsense"), None);
    }
}
