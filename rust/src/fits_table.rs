//! The per-mode rows of a paramfit table: what each column of a paramfit
//! CSV holds, computed from a fit.
//!
//! The desktop app's CSV writer formats these rows, and the Python wrapper
//! returns them as its table, so both carry the same numbers. Column order
//! and meaning are documented in the app's `documents/OUTPUT_FORMAT.md`.

use ndarray::Array2;

use crate::fits::ParamFitResult;
use crate::mode_peaks::{assign_members, stats_for_members, ModePeakStats, PeakProps};
use crate::paramfit::opts::{PeakAssign, PEAK_ASSIGN_PHASE_FALLBACK};

/// Which SOPH a paramfit table describes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FitAxis {
    Power,
    Phase,
}

/// One mode's row of a paramfit table.
///
/// The first six are the fit's mode parameters as emitted (`Density` is the
/// phase fit's empirical amplitude on the phase axis). The cross-axis
/// columns depend on the axis: `pref_phase` and `coupling` are filled on the
/// power axis and `sopower_mean` on the phase axis; the others are NaN.
#[derive(Debug, Clone)]
pub struct ModeRow {
    pub density: f64,
    pub freq_mean: f64,
    pub freq_std: f64,
    pub so_mean: f64,
    pub so_std: f64,
    pub theta: f64,
    pub volume: f64,
    pub pref_phase: f64,
    pub coupling: f64,
    pub sopower_mean: f64,
    /// The mode's member-peak summary (`Pk*` columns); None when no peak
    /// properties were supplied.
    pub peaks: Option<ModePeakStats>,
}

/// The rows of one axis's paramfit table, one per mode in fit order.
///
/// * `freq_bins`: the SOPH frequency bin centers.
/// * Power axis: `phase_fit` and `phase_bins` (the SO-phase fit and its bin
///   centers) fill `PrefPhase` / `Coupling`; either missing leaves NaN.
/// * Phase axis: `sopower_hist` (`[n_power x n_freq]`) fills `SOpowerMean`;
///   missing leaves NaN.
/// * `mode_peaks`: per-mode member-peak summaries from [`compute_mode_peaks`].
pub fn paramfit_rows(
    fit: &ParamFitResult,
    axis: FitAxis,
    freq_bins: &[f64],
    phase_fit: Option<&ParamFitResult>,
    phase_bins: Option<&[f64]>,
    sopower_hist: Option<&Array2<f64>>,
    mode_peaks: Option<&[ModePeakStats]>,
) -> Vec<ModeRow> {
    let n_modes = fit.params.nrows();
    let mut rows = Vec::with_capacity(n_modes);
    for m in 0..n_modes {
        let density = fit.params[[m, 0]];
        let freq_mean = fit.params[[m, 1]];
        let freq_std = fit.params[[m, 2]];
        let so_mean = fit.params[[m, 3]];
        let so_std = fit.params[[m, 4]];
        let theta = fit.params[[m, 5]];
        let volume = mode_volume(axis, density, freq_std, so_std);

        let mut pref_phase = f64::NAN;
        let mut coupling = f64::NAN;
        let mut sopower_mean = f64::NAN;
        match axis {
            FitAxis::Power => {
                // Model-based preferred phase and coupling at the mode's
                // frequency, as MATLAB `annotatePowerWithPreferredPhase.m`.
                if let (Some(pf), Some(pb)) = (phase_fit, phase_bins) {
                    (pref_phase, coupling) = model_argmax_at_freq(pf, pb, freq_bins, freq_mean);
                }
            }
            FitAxis::Phase => {
                // Cross-axis location: mean of the SO-power histogram column
                // at the mode's frequency (a linear axis, so a plain mean).
                if let Some(h) = sopower_hist {
                    sopower_mean = column_mean_at_freq(h, freq_bins, freq_mean);
                }
            }
        }

        let peaks = mode_peaks.and_then(|stats| stats.get(m)).cloned();
        rows.push(ModeRow {
            density, freq_mean, freq_std, so_mean, so_std, theta, volume,
            pref_phase, coupling, sopower_mean, peaks,
        });
    }
    rows
}

/// The peak assignment the `Pk*` columns were computed under on `axis`. The
/// phase axis always uses a numeric contour, so under the density rules
/// (background / argmax) it uses [`PEAK_ASSIGN_PHASE_FALLBACK`].
pub fn effective_peak_assign(axis: FitAxis, assign: PeakAssign) -> PeakAssign {
    match axis {
        FitAxis::Phase if assign.threshold().is_none() => PEAK_ASSIGN_PHASE_FALLBACK,
        _ => assign,
    }
}

/// Per-mode member-peak summaries (the `Pk*` columns) for a fit.
///
/// The density rules (background / argmax) need every mode and the fitted
/// background at once, so membership is computed for the whole mode set,
/// then summarized one mode at a time. `FreqStd` is a true sigma on both
/// axes, so the fit's rows are used as they are.
pub fn compute_mode_peaks(
    fit: &ParamFitResult,
    is_phase: bool,
    peaks: &PeakProps<'_>,
    assign: PeakAssign,
) -> Vec<ModePeakStats> {
    let n = fit.params.nrows();
    let mut rows: Vec<[f64; 6]> = Vec::with_capacity(n);
    for m in 0..n {
        rows.push([
            fit.params[[m, 0]], fit.params[[m, 1]], fit.params[[m, 2]],
            fit.params[[m, 3]], fit.params[[m, 4]], fit.params[[m, 5]],
        ]);
    }
    let members = assign_members(assign, is_phase, &rows, fit.background, peaks);
    let mut mode_stats = Vec::with_capacity(members.len());
    for mode_members in members.iter() {
        mode_stats.push(stats_for_members(mode_members, peaks));
    }
    mode_stats
}

/// Argmax of the fitted phase-paramfit *model surface* column at the
/// nearest `freq_bins` index to `fmean`. Returns
/// `(phase_at_argmax, magnitude_at_argmax)`, mirroring MATLAB
/// `annotatePowerWithPreferredPhase.m`'s
/// `get_mode_phase_argmax(f, model_SOPH_phase, …)` call. NaN-tuple when
/// the model grid doesn't match `phase_bins` / `freq_bins`.
///
/// Also a NaN-tuple when the phase fit has no model at that frequency.
/// The phase fit covers its own frequency window, and `phase_fit.model`
/// is zero at every phase bin of a frequency outside it. A column that is
/// zero throughout has no preferred phase to report.
///
/// `phase_fit.model` is shape `[n_phase × n_freq]` on the same grid the
/// caller passed — verified by checking `phase_fit.model.dim()` against
/// `(phase_bins.len(), freq_bins.len())`.
fn model_argmax_at_freq(
    phase_fit: &ParamFitResult,
    phase_bins: &[f64],
    freq_bins: &[f64],
    fmean: f64,
) -> (f64, f64) {
    let (n_phase, n_freq) = phase_fit.model.dim();
    if n_phase != phase_bins.len() || n_freq != freq_bins.len() || n_phase == 0 {
        return (f64::NAN, f64::NAN);
    }
    if !fmean.is_finite() {
        return (f64::NAN, f64::NAN);
    }
    let fi = nearest_idx(freq_bins, fmean);
    let mut best_k = 0usize;
    let mut best_v = f64::NEG_INFINITY;
    let mut column_is_all_zero = true;
    for k in 0..n_phase {
        let v = phase_fit.model[[k, fi]];
        if v != 0.0 {
            column_is_all_zero = false;
        }
        if v.is_finite() && v > best_v {
            best_v = v;
            best_k = k;
        }
    }
    if best_v == f64::NEG_INFINITY || column_is_all_zero {
        return (f64::NAN, f64::NAN);
    }
    (phase_bins[best_k], best_v)
}

/// Mean of the cross-axis histogram column at the nearest freq bin
/// to `fmean` (used for the phase paramfit's `SOpowerMean` cross-axis
/// column — linear axis, so a plain mean is appropriate).
fn column_mean_at_freq(
    cross_hist: &Array2<f64>,
    freq_bins: &[f64],
    fmean: f64,
) -> f64 {
    let (n_cross, n_freq) = cross_hist.dim();
    if n_cross == 0 || n_freq != freq_bins.len() || !fmean.is_finite() {
        return f64::NAN;
    }
    let fi = nearest_idx(freq_bins, fmean);
    // Mean over the finite cells of that column, accumulated in row order.
    let mut sum = 0.0_f64;
    let mut n_finite = 0usize;
    for k in 0..n_cross {
        let v = cross_hist[[k, fi]];
        if v.is_finite() {
            sum += v;
            n_finite += 1;
        }
    }
    if n_finite == 0 { f64::NAN } else { sum / (n_finite as f64) }
}

/// Linear search nearest-bin index for a small (≤1k) bins vector.
pub fn nearest_idx(bins: &[f64], v: f64) -> usize {
    let mut best = 0usize;
    let mut best_d = f64::INFINITY;
    for (i, &b) in bins.iter().enumerate() {
        let d = (b - v).abs();
        if d < best_d {
            best_d = d;
            best = i;
        }
    }
    best
}

/// Scaled modified Bessel I0: returns `exp(-x) * I_0(x)`. Polynomial
/// approximations from Abramowitz & Stegun §9.8.1 (|x| ≤ 3.75) and
/// §9.8.2 (|x| > 3.75). Accurate to ~1e-7 across the whole range, which
/// is well below the f64 tolerance we care about for paramfit volumes.
/// Same coefficient set used by NumPy / Cephes / SciPy.
fn bessel_i0_scaled(x: f64) -> f64 {
    let ax = x.abs();
    if ax < 3.75 {
        let t = (x / 3.75).powi(2);
        let p = 1.0
            + t * (3.5156229
                + t * (3.0899424
                    + t * (1.2067492
                        + t * (0.2659732
                            + t * (0.0360768
                                + t * 0.0045813)))));
        // I_0(x); multiply by exp(-|x|) to return the scaled form.
        p * (-ax).exp()
    } else {
        let t = 3.75 / ax;
        let p = 0.39894228
            + t * (0.01328592
                + t * (0.00225319
                    + t * (-0.00157565
                        + t * (0.00916281
                            + t * (-0.02057706
                                + t * (0.02635537
                                    + t * (-0.01647633
                                        + t * 0.00392377)))))));
        p / ax.sqrt()
    }
}

/// Standalone-kernel mode area per `createSOPHparamfitStruct.m`. The
/// Gaussian-width coefficients follow from ∫exp(-½(d/σ)²)dd = σ·√(2π).
pub fn mode_volume(axis: FitAxis, density: f64, fstd: f64, xstd: f64) -> f64 {
    match axis {
        // power (rotGauss): exact full-plane component integral,
        // V = density * 2π * xstd * fstd.
        FitAxis::Power => density * 2.0 * std::f64::consts::PI * xstd * fstd,
        // phase (vmGauss): historical standalone-vmGauss area surrogate,
        // V = density * 2π * besseli(0, k, scaled) * √(2π) · fstd, where
        // k = 1 / xstd², `besseli(_, _, 'scaled')` is exp(-k)·I_0(k) (exact
        // A&S polynomial, valid at the small k≈1 of broad REM modes). The
        // von Mises factor and k consume `recikappa = 1/√k` directly, and
        // density is the empirical no-sinusoid height of the full jointly
        // normalized model, so V is not the exact integral of an isolated
        // normalized component. Approved data change per
        // handoff/DATA_FIDELITY.md.
        FitAxis::Phase => {
            let k = 1.0 / (xstd * xstd);
            let scaled_i0 = bessel_i0_scaled(k);
            let v = density * 2.0 * std::f64::consts::PI * scaled_i0
                * (2.0 * std::f64::consts::PI).sqrt() * fstd;
            if v.is_finite() { v } else { f64::NAN }
        }
    }
}

#[cfg(test)]
mod preferred_phase_tests {
    use super::model_argmax_at_freq;
    use crate::fits::{GofValues, ParamFitResult};
    use ndarray::Array2;

    /// Phase fit on 4 phase bins by 5 frequency bins. The fit window is
    /// frequency bins 1 to 3, so columns 0 and 4 of the model are zero.
    fn phase_fit_with_window() -> ParamFitResult {
        let mut model = Array2::<f64>::zeros((4, 5));
        for freq_bin in 1..=3 {
            model[[0, freq_bin]] = 0.2;
            model[[1, freq_bin]] = 0.7;
            model[[2, freq_bin]] = 0.4;
            model[[3, freq_bin]] = 0.1;
        }
        ParamFitResult {
            params: Array2::zeros((0, 6)),
            background: [0.0, 0.0, 0.0],
            unit_row: f64::NAN,
            model,
            gof: GofValues { sse: 0.0, rsquare: 1.0, adjrsquare: 1.0, rmse: 0.0, dfe: 1.0, dfm: 1.0 },
        }
    }

    #[test]
    fn a_mode_inside_the_phase_window_gets_the_model_argmax() {
        let phase_bins = [-2.0, -1.0, 1.0, 2.0];
        let freq_bins = [8.0, 10.0, 12.0, 14.0, 16.0];
        let (pref_phase, coupling) =
            model_argmax_at_freq(&phase_fit_with_window(), &phase_bins, &freq_bins, 12.3);
        assert_eq!(pref_phase, -1.0);
        assert_eq!(coupling, 0.7);
    }

    #[test]
    fn a_mode_outside_the_phase_window_gets_nan() {
        let phase_bins = [-2.0, -1.0, 1.0, 2.0];
        let freq_bins = [8.0, 10.0, 12.0, 14.0, 16.0];
        for mode_freq in [7.5, 8.4, 15.9, 20.0] {
            let (pref_phase, coupling) =
                model_argmax_at_freq(&phase_fit_with_window(), &phase_bins, &freq_bins, mode_freq);
            assert!(pref_phase.is_nan(), "{} Hz: PrefPhase {}", mode_freq, pref_phase);
            assert!(coupling.is_nan(), "{} Hz: Coupling {}", mode_freq, coupling);
        }
    }
}

#[cfg(test)]
mod volume_tests {
    use super::*;

    /// Pin the scaled-Bessel implementation against reference values
    /// (from MATLAB `besseli(0, k, 1)` / SciPy `i0e(k)`).
    #[test]
    fn bessel_i0_scaled_reference() {
        // (k, exp(-k)*I_0(k)) — reference from standard tables (A&S Table
        // 9.8 / SciPy `i0e`). The A&S 9.8.1/9.8.2 polynomials we use give
        // ~1.9e-7 max relative error against true values.
        let cases = [
            (0.0,    1.0),
            (0.5,    0.6450352706),
            (1.0,    0.4657596077),
            (2.0,    0.3085083226),
            (5.0,    0.1835408126),
            (10.0,   0.1278333371),
            (50.0,   0.0565616),
            (100.0,  0.0399444),
        ];
        for (k, expect) in cases {
            let got = bessel_i0_scaled(k);
            let rel = (got - expect).abs() / expect.abs().max(1e-30);
            assert!(rel < 5e-6,
                "bessel_i0_scaled({k}) = {got}, expected {expect} (rel {rel:.2e})");
        }
    }

    /// Sanity-check the Bessel-based phase volume against its large-k
    /// asymptote. At large k the two should agree to <2%. At small k they
    /// should differ noticeably, which is why the exact Bessel form is used.
    ///
    /// The asymptote reference is `density · 2π · xstd · fstd`, from
    /// `I0e(k) ≈ 1/√(2πk)` with `k = 1/xstd²` folded through the σ-form
    /// volume `density · 2π · I0e(k) · √(2π) · fstd`.
    #[test]
    fn mode_volume_phase_matches_asymptote_at_large_k() {
        let density = 1.0;
        let fstd = 1.0;
        // Sharp mode: k = 1/xstd² = 25 → asymptote within ~1% of truth.
        let xstd_sharp = 0.2;
        let v_new = mode_volume(FitAxis::Phase, density, fstd, xstd_sharp);
        let v_asym = density * xstd_sharp * 2.0 * std::f64::consts::PI * fstd;
        let rel = (v_new - v_asym).abs() / v_asym;
        assert!(rel < 0.02, "sharp-mode asymptote disagreement: {rel:.4}");

        // Broad mode: k = 1/xstd² = 1 → asymptote is ~13% low.
        let xstd_broad = 1.0;
        let v_new2 = mode_volume(FitAxis::Phase, density, fstd, xstd_broad);
        let v_asym2 = density * xstd_broad * 2.0 * std::f64::consts::PI * fstd;
        let rel2 = (v_new2 - v_asym2).abs() / v_asym2;
        assert!(rel2 > 0.08,
            "broad-mode volume should differ from the large-k asymptote by more than 8% when the exact Bessel form is used, got {rel2:.4}");
    }
}
