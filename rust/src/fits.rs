//! Parametric- and spline-basis fits over the SOPH histograms.
//!
//! Wraps the kernels in `crate::paramfit::{rot_gauss, vm_gauss}` and
//! `crate::spline_basis` and packages their outputs into structs the
//! orchestrator can stash in `DynamoOutputs`
//! and the export crate can write to disk.
//!
//! ## Iter-add-mode loop
//!
//! Ports MATLAB's `param_basis_{power,phase}.m` outer loop verbatim:
//!   1. Run the histogram watershed (`extracthistpeaks`) over the full
//!      SOPH to obtain `mode_params` — one `[amp, fmean, fstd, pmean,
//!      pstd, theta]` row per region, sorted by height.
//!   2. If the watershed yields no usable region, fall back to MATLAB's
//!      single synthetic center-mode failsafe (`param_basis_power.m:267`).
//!   3. Loop `ii = 1..max_peaks`: seed mode `ii` from `mode_params(ii)`
//!      while `ii <= N_wshed`, else from the mean of the accepted modes;
//!      refit all modes jointly each iteration.
//!   4. Revert an iteration when adjR² improvement < `min_pctr2`, mode
//!      overlap > `max_overlap`, an amplitude drops below `min_amp`, or
//!      (power) two modes sit closer than `min_freq_diff`. A revert past
//!      the watershed-mode count terminates the loop; a revert within the
//!      watershed modes just drops that seed and continues.
//!   5. Return the last accepted iteration (MATLAB `minpctr2` selection).
//!
//! Bounds and watershed parameters match `param_basis_opts('power'|'phase')`
//! defaults verbatim.

use std::cmp::Ordering;
use std::f64::consts::PI;
use crate::peak_assign::wrap_to_pi; // single canonical phase wrap (MATLAB wrapToPi)

use ndarray::{Array2, ArrayView2};

use crate::histpeaks::{
    extract_hist_peaks_phase, extract_hist_peaks_with_stats, HistPeak,
};
use crate::paramfit::Gof as RsGof;
use crate::paramfit::rot_gauss::{eval_model as rg_eval_model, fit_rotgauss as rs_fit_rotgauss};
use crate::paramfit::vm_gauss::{eval_model as vm_eval_model, fit_vmgauss as rs_fit_vmgauss};
use crate::spline_basis::fit_tensor_product_spline as rs_fit_spline;

/// Warn when a mode-center bound is set in the settings but its axis switch is
/// off, so the bound is silently discarded.
///
/// `constrain_*_center = false` overwrites the whole slot with ±inf, which
/// happens AFTER the explicit value is read. A file can therefore carry a
/// deliberate `fmean_lb` that has no effect, with nothing on screen to say so.
fn warn_discarded_center_bounds(
    axis: &str,
    constrained: bool,
    lb: crate::paramfit::opts::Bound,
    ub: crate::paramfit::opts::Bound,
) {
    // `Derived` is the default rule, not a user intent, so discarding it is
    // not worth a warning — only an explicit number or open end is.
    if constrained || (!lb.is_set() && !ub.is_set()) {
        return;
    }
    let which = match (lb.is_set(), ub.is_set()) {
        (true, true) => "_lb and _ub",
        (true, false) => "_lb",
        _ => "_ub",
    };
    log::warn!(
        "paramfit: {axis}{which} set but constrain_{axis}_center is false — \
         the bound is ignored and the center is left unbounded (±inf). \
         Set constrain_{axis}_center = true for it to apply."
    );
}

/// Build watershed-seeded `mode_params` rows from `extracthistpeaks`
/// regions — mirrors `param_basis_{power,phase}.m`: keep regions whose
/// `PeakFrequency` is within `freq_limits`, sort by `Height` descending,
/// map to `[amp, fmean, fstd, pmean, pstd, theta]` with
/// `amp = Height`, `fstd = Bandwidth/1.96`, `pstd = Duration/1.96`, and
/// `theta = 0`.
///
/// The `/1.96` widths are empirical estimates already expressed in the
/// parameter units (σ for the Gaussian slots, `recikappa` for the phase
/// slot 4), so no `/√2` rescale applies to them.
fn watershed_mode_params(
    peaks: &[HistPeak],
    freq_limits: (f64, f64),
    min_freq_diff: f64,
) -> Vec<[f64; 6]> {
    let mut kept: Vec<&HistPeak> = peaks
        .iter()
        .filter(|p| p.peak_freq >= freq_limits.0 && p.peak_freq <= freq_limits.1)
        .collect();
    kept.sort_by(|a, b| b.height.partial_cmp(&a.height).unwrap_or(Ordering::Equal));
    // Drop watershed seeds that sit within `min_freq_diff` of a
    // higher-height seed already kept. Without this guard the watershed
    // sometimes produces two near-duplicate seeds on the same ridge
    // (e.g. 800272/C3: seeds at f = 7.37 and 6.94 Hz, only 0.43 Hz
    // apart, leading to a duplicated alpha mode in the final fit).
    // The same `min_freq_diff` threshold the LM uses as a revert
    // criterion is what we use here, so the upstream dedup matches the
    // downstream check. `min_freq_diff = 0` disables the guard.
    let mut deduped: Vec<&HistPeak> = Vec::with_capacity(kept.len());
    for p in &kept {
        let too_close = min_freq_diff > 0.0
            && deduped.iter().any(|q| (q.peak_freq - p.peak_freq).abs() < min_freq_diff);
        if !too_close {
            deduped.push(*p);
        }
    }
    deduped
        .iter()
        .map(|p| {
            [
                p.height,
                p.peak_freq,
                p.bandwidth / 1.96,
                p.so_feature,
                p.duration / 1.96,
                0.0,
            ]
        })
        .collect()
}

fn finite_min(xs: impl Iterator<Item = f64>) -> f64 {
    xs.filter(|v| v.is_finite()).fold(f64::INFINITY, f64::min)
}
fn finite_max(xs: impl Iterator<Item = f64>) -> f64 {
    xs.filter(|v| v.is_finite()).fold(f64::NEG_INFINITY, f64::max)
}

/// MATLAB-`prctile`-style percentile over the finite values of `xs`
/// (sample i, 1-based, sits at percentile `100·(i−0.5)/n`; linear
/// interpolation between). Used for the phase background `zzz` start
/// point — `fit_vmGauss.m` seeds it at `prctile(phase_hist(:), 5)`.
fn percentile_finite(xs: impl Iterator<Item = f64>, p: f64) -> f64 {
    let mut v: Vec<f64> = xs.filter(|x| x.is_finite()).collect();
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n == 1 {
        return v[0];
    }
    let rank = p / 100.0 * n as f64 - 0.5;
    if rank <= 0.0 {
        return v[0];
    }
    if rank >= (n - 1) as f64 {
        return v[n - 1];
    }
    let lo = rank.floor() as usize;
    let frac = rank - lo as f64;
    v[lo] + frac * (v[lo + 1] - v[lo])
}

/// Amplitude bounds from `amp0` — mirrors `param_basis_*.m`:
/// `LB_default(1) = min(amp0)/10`, `UB_default(1) = max(amp0)*10`.
fn amp_bounds(amp0: &[f64]) -> (f64, f64) {
    let lo = finite_min(amp0.iter().copied());
    let hi = finite_max(amp0.iter().copied());
    if !lo.is_finite() || !hi.is_finite() {
        return (0.0, 1e-6);
    }
    ((lo / 10.0).max(0.0), (hi * 10.0).max(1e-6))
}

/// Column-wise mean of a set of `[f64; 6]` mode rows (MATLAB `mean(B0i, 1)`).
fn mean_row(modes: &[[f64; 6]]) -> [f64; 6] {
    let mut s = [0.0_f64; 6];
    if modes.is_empty() {
        return s;
    }
    for m in modes {
        for c in 0..6 {
            s[c] += m[c];
        }
    }
    let n = modes.len() as f64;
    for v in &mut s {
        *v /= n;
    }
    s
}

/// Median over a `f64` slice (NaN-skipping). Returns 0.0 on empty.
fn median(xs: &[f64]) -> f64 {
    let mut v: Vec<f64> = xs.iter().copied().filter(|x| x.is_finite()).collect();
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 { v[n / 2] } else { 0.5 * (v[n / 2 - 1] + v[n / 2]) }
}

/// Matching-pursuit-style seed for the next mode: find the (x, y)
/// argmax of `hist − model`, skipping freq columns within `min_freq_diff`
/// of any existing mode's `FreqMean`. Returns `None` if no positive
/// residual remains after the freq-exclusion mask (or the model
/// already covers / overshoots), signalling "no real unfit feature to
/// add" — the caller falls back to the mean(B0i) seed.
///
/// The exclusion mask is the anti-duplication guard: without it the
/// residual peak right next to an undermodeled existing mode (alpha
/// tail, spindle skirts) would be picked, the LM would put the new
/// mode on top of the existing one, and the run reverts on overlap
/// (or worse, accepts a near-duplicate — 800272 had two modes at
/// f = 6.85 / 7.39 Hz under unguarded residual seeding). When
/// `min_freq_diff = 0` (phase axis) the mask is inert.
///
/// Width priors (`fstd`, `pstd`) default to the median of the accepted
/// modes' stds — a reasonable starting size. `theta` is 0.
fn residual_max_seed(
    hist_yx: ArrayView2<f64>,
    model_yx: &Array2<f64>,
    x_axis: &[f64],
    y_axis: &[f64],
    accepted_modes: &[[f64; 6]],
    min_freq_diff: f64,
) -> Option<[f64; 6]> {
    let (ny, nx) = hist_yx.dim();
    if model_yx.dim() != (ny, nx) || ny != y_axis.len() || nx != x_axis.len() {
        return None;
    }
    // Pre-compute a "this freq bin is too close to an existing mode"
    // mask. min_freq_diff = 0 disables the guard.
    let excluded_y: Vec<bool> = if min_freq_diff > 0.0 {
        y_axis.iter().map(|&f| {
            accepted_modes.iter().any(|m| (m[1] - f).abs() < min_freq_diff)
        }).collect()
    } else {
        vec![false; ny]
    };
    let mut best = (0usize, 0usize, f64::NEG_INFINITY);
    for yi in 0..ny {
        if excluded_y[yi] { continue; }
        for xi in 0..nx {
            let h = hist_yx[[yi, xi]];
            let m = model_yx[[yi, xi]];
            if !h.is_finite() || !m.is_finite() {
                continue;
            }
            let r = h - m;
            if r > best.2 {
                best = (yi, xi, r);
            }
        }
    }
    if !best.2.is_finite() || best.2 <= 0.0 {
        return None;
    }
    let fmean = y_axis[best.0];
    let pmean = x_axis[best.1];
    let amp = best.2;
    // Width priors: median over accepted-mode std columns (cols 2 = fstd,
    // 4 = pstd). Falls back to small finite values when there's nothing.
    let fstd_vals: Vec<f64> = accepted_modes.iter().map(|r| r[2]).collect();
    let pstd_vals: Vec<f64> = accepted_modes.iter().map(|r| r[4]).collect();
    // The medians ride whatever the accepted params already carry; the
    // literal fallbacks keep the LM bounds non-degenerate when there are
    // no valid accepted widths, and both are expressed directly in the
    // corrected parameter units.
    let fstd = {
        let m = median(&fstd_vals);
        if m > 0.0 { m } else { 1.0 }
    };
    let pstd = {
        let m = median(&pstd_vals);
        if m > 0.0 { m } else { 5.0 }
    };
    Some([amp, fmean, fstd, pmean, pstd, 0.0])
}

/// Index of the axis bin nearest `v` (MATLAB `[~, idx] = min(abs(axis - v))`).
fn nearest_idx(axis: &[f64], v: f64) -> usize {
    let mut best = 0usize;
    let mut best_d = f64::INFINITY;
    for (i, &a) in axis.iter().enumerate() {
        let d = (a - v).abs();
        if d < best_d {
            best_d = d;
            best = i;
        }
    }
    best
}

/// Nearest bin by wrapped angular distance: MATLAB
/// `min(abs(wrapToPi(phase_bins - v)))` (DYNAM-O PR #73). Used for the
/// phase empirical-amplitude lookup so a mode center at a ±2π alias
/// (reachable because the phase center is bounded to two full periods)
/// still selects the circularly closest phase bin.
fn nearest_circular_idx(axis: &[f64], v: f64) -> usize {
    let mut best = 0usize;
    let mut best_d = f64::INFINITY;
    for (i, &a) in axis.iter().enumerate() {
        let d = wrap_to_pi(a - v).abs();
        if d < best_d {
            best_d = d;
            best = i;
        }
    }
    best
}

/// Result of a single-mode parametric fit.
///
/// `params` is `(n_modes, 6)` row-major:
/// * power (rotGauss): `[amp, fmean, fstd, pmean, pstd, theta]`
/// * phase (vmGauss):  `[amp, fmean, fstd, phasepref, recikappa, theta]`
///
/// `model` is the reconstruction on the same grid as the input
/// histogram, oriented `(n_so_bins, n_freq_bins)` to match the SOPH
/// convention used by `DynamoOutputs::sopower_hist` / `sophase_hist`.
#[derive(Debug, Clone)]
pub struct ParamFitResult {
    pub params: Array2<f64>,
    /// `[xxx, yyy, zzz]` — background-plane coefficients
    /// (linear `xxx*x + yyy*y + zzz` for power; sinusoidal
    /// `xxx*sin(x + yyy) + zzz` for phase).
    pub background: [f64; 3],
    /// Phase-only `unit_row` coefficient. NaN for power.
    pub unit_row: f64,
    /// Reconstructed model on the input grid, `(n_so, n_freq)`.
    pub model: Array2<f64>,
    pub gof: GofValues,
}

/// Result of a tensor-product B-spline LSQ fit.
#[derive(Debug, Clone)]
pub struct SplineFitResult {
    /// Coefficient matrix, shape `(m_y, m_x)` — matches MATLAB
    /// `squeeze(spline_obj.coefs)'` orientation.
    pub coefs: Array2<f64>,
    /// Splinefit reconstruction on the histogram grid, `(n_so, n_freq)`.
    pub splinefit: Array2<f64>,
    /// Knot vectors (post-augknt).
    pub knots_x_aug: Vec<f64>,
    pub knots_y_aug: Vec<f64>,
    /// Bin centers used during the fit (post-validity-mask). Equals the
    /// input centers when the fit ran on the full histogram.
    pub fit_so_bins: Vec<f64>,
    pub fit_freq_bins: Vec<f64>,
}

/// MATLAB-style goodness-of-fit struct, mirrored 1:1 from `cfit`.
#[derive(Debug, Clone, Copy)]
pub struct GofValues {
    pub sse: f64,
    pub rsquare: f64,
    pub adjrsquare: f64,
    pub rmse: f64,
    pub dfe: f64,
    pub dfm: f64,
}

impl From<RsGof> for GofValues {
    fn from(g: RsGof) -> Self {
        Self {
            sse: g.sse,
            rsquare: g.rsquare,
            adjrsquare: g.adjrsquare,
            rmse: g.rmse,
            dfe: g.dfe,
            dfm: g.dfm,
        }
    }
}

#[derive(Debug)]
pub enum FitError {
    Param { axis: &'static str, msg: String },
    Spline { axis: &'static str, msg: String },
    Invalid { axis: &'static str, msg: String },
}

impl std::fmt::Display for FitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FitError::Param { axis, msg } => write!(f, "param fit ({axis}): {msg}"),
            FitError::Spline { axis, msg } => write!(f, "spline fit ({axis}): {msg}"),
            FitError::Invalid { axis, msg } => write!(f, "invalid fit input ({axis}): {msg}"),
        }
    }
}

impl std::error::Error for FitError {}

// Paramfit run thresholds live on `PowerParamFitOpts` / `PhaseParamFitOpts`
// in `crate::settings`. See those structs for the MATLAB mapping.

/// Fit rotated-Gaussian modes + linear plane to the SO-power histogram.
/// Mirrors `param_basis_power.m`: runs the histogram watershed over the
/// full SOPH for initial conditions, then iteratively adds modes.
///
/// `sopower_hist` is `(n_so, n_freq)` — the orientation in
/// `DynamoOutputs::sopower_hist`. The kernel expects `(n_y, n_x)` with
/// y=freq, x=power, so we transpose into a fresh array before calling.
pub fn fit_paramfit_power(
    sopower_hist: ArrayView2<f64>,
    so_bins: &[f64],
    freq_bins: &[f64],
    opts: &crate::paramfit::opts::PowerParamFitOpts,
) -> Result<ParamFitResult, FitError> {
    let (n_so, n_freq) = sopower_hist.dim();
    if so_bins.len() != n_so {
        return Err(FitError::Invalid {
            axis: "power",
            msg: format!("so_bins {} != hist rows {}", so_bins.len(), n_so),
        });
    }
    if freq_bins.len() != n_freq {
        return Err(FitError::Invalid {
            axis: "power",
            msg: format!("freq_bins {} != hist cols {}", freq_bins.len(), n_freq),
        });
    }
    if n_so < 4 || n_freq < 4 {
        return Err(FitError::Invalid {
            axis: "power",
            msg: format!("hist too small ({}×{})", n_so, n_freq),
        });
    }

    // Watershed segmentation over the FULL SO-power histogram. Mirrors
    // `param_basis_power.m:163-167` (`extracthistpeaks` on the raw SOPH);
    // params come from `PowerParamFitOpts::watershed`.
    let wshed_params = opts.watershed.to_hist_peak_params();
    let (_labels, peaks) =
        extract_hist_peaks_with_stats(sopower_hist, so_bins, freq_bins, &wshed_params);
    // Keep regions whose peak frequency is within `freq_limits`, sorted
    // by height descending (param_basis_power.m:178-208).
    let mut mode_params = watershed_mode_params(
        &peaks,
        (opts.freq_limits[0], opts.freq_limits[1]),
        opts.min_freq_diff,
    );

    // Restrict to the fit window AND to the valid (non-NaN) data inside it,
    // exactly as MATLAB does. Using the raw window would let a mode center
    // sit in SO-power bins that `min_time_in_bin` emptied.
    let (so_idx, so_axis, freq_idx, freq_axis) = valid_bins(
        sopower_hist,
        so_bins,
        freq_bins,
        (opts.power_limits[0], opts.power_limits[1]),
        (opts.freq_limits[0], opts.freq_limits[1]),
    );
    if so_axis.len() < 4 || freq_axis.len() < 4 {
        return Err(FitError::Invalid {
            axis: "power",
            msg: format!(
                "not enough valid data in the fit window \
                 (post-clip histogram {}×{} valid bins; need at least 4 per axis)",
                so_axis.len(),
                freq_axis.len()
            ),
        });
    }

    // Transpose+slice to (n_freq, n_so) for the kernel.
    let mut hist_yx = Array2::<f64>::zeros((freq_axis.len(), so_axis.len()));
    for (yi, &fi) in freq_idx.iter().enumerate() {
        for (xi, &si) in so_idx.iter().enumerate() {
            hist_yx[[yi, xi]] = sopower_hist[[si, fi]];
        }
    }

    // `amp0` (param_basis_power.m): watershed-success → region heights;
    // watershed-failure → the flattened in-window sub-histogram. Drives
    // the amplitude bounds and the failsafe seed amplitude.
    let amp0: Vec<f64> = if mode_params.is_empty() {
        hist_yx.iter().copied().filter(|v| v.is_finite()).collect()
    } else {
        mode_params.iter().map(|r| r[0]).collect()
    };
    let (amp_lo, amp_hi) = amp_bounds(&amp0);

    // Watershed-failure failsafe (param_basis_power.m:267-282): one
    // synthetic center mode so the fitting loop can still run.
    if mode_params.is_empty() {
        // These range/4 widths are priors, already in the parameter units.
        let fmean = (axis_max(&freq_axis) + freq_axis[0]) / 2.0;
        let fstd = (axis_max(&freq_axis) - freq_axis[0]) / 4.0;
        let pmean = (axis_max(&so_axis) + so_axis[0]) / 2.0;
        let pstd = (axis_max(&so_axis) - so_axis[0]) / 4.0;
        let amp = finite_max(hist_yx.iter().copied()).max(1e-6);
        mode_params = vec![[amp, fmean, fstd, pmean, pstd, 0.0]];
    }

    // Per-mode bounds. `fstd`/`pstd`/`theta` are always numeric. The other
    // three slots follow MATLAB's NaN sentinel: an explicit value wins, and
    // `Derived` derives from the data (amp from the seed amplitudes, the two
    // centers from the valid axis extent). MATLAB fills the NaN slots FIRST
    // and only then lets `~constrain_*_center` blow the bound to ±inf, so
    // the order below matches (`param_basis_power.m:244-277`).
    let mb = &opts.mode_bounds;
    let mut lb_single: [f64; 6] = [
        mb.amp_lb.resolve(amp_lo),
        mb.fmean_lb.resolve(freq_axis[0]),
        mb.fstd_lb,
        mb.pmean_lb.resolve(so_axis[0]),
        mb.pstd_lb,
        mb.theta_lb,
    ];
    let mut ub_single: [f64; 6] = [
        mb.amp_ub.resolve(amp_hi),
        mb.fmean_ub.resolve(freq_axis[freq_axis.len() - 1]),
        mb.fstd_ub,
        mb.pmean_ub.resolve(so_axis[so_axis.len() - 1]),
        mb.pstd_ub,
        mb.theta_ub,
    ];
    // Per-axis center constraints. Freeing an axis to ±inf lets a mode
    // center leave the observed SOPH on that axis; the fit data is unchanged
    // (still the windowed histogram), only where a center may sit changes.
    // Frequency = slot 1, SO-power = slot 3.
    warn_discarded_center_bounds("freq", opts.constrain_freq_center, mb.fmean_lb, mb.fmean_ub);
    warn_discarded_center_bounds("power", opts.constrain_power_center, mb.pmean_lb, mb.pmean_ub);
    if !opts.constrain_freq_center {
        lb_single[1] = f64::NEG_INFINITY;
        ub_single[1] = f64::INFINITY;
    }
    if !opts.constrain_power_center {
        lb_single[3] = f64::NEG_INFINITY;
        ub_single[3] = f64::INFINITY;
    }

    // Background plane — MATLAB-parity bounds (fit_rotGauss.m:111-113): the
    // constant is bounded to [0, max(hist)] (a negative offset is a negative
    // density on a count/density histogram — unphysical, and an unbounded plane
    // drifts there to overfit), the slopes to ±0.1, and the constant is seeded
    // at the 5th percentile so it starts interior to the bound. Bounding BOTH
    // offset and slopes is deliberate: clamping the offset alone just pushes the
    // tilt into the slopes (empirically worse than either extreme).
    let bg_offset0 = percentile_finite(hist_yx.iter().copied(), 5.0);
    let bg_zmax = finite_max(hist_yx.iter().copied());
    let bg_initial = [0.0_f64, 0.0, bg_offset0];
    let bb = &opts.background_bounds;
    // `offset_ub: null` derives max(hist), matching MATLAB. 1.0 is the
    // fallback for an all-NaN window, where max() is not finite.
    let bg_offset_ub = bb
        .offset_ub
        .resolve(if bg_zmax.is_finite() { bg_zmax } else { 1.0 });
    let bg_lower = [bb.pow_slope_lb, bb.freq_slope_lb, bb.offset_lb];
    let bg_upper = [bb.pow_slope_ub, bb.freq_slope_ub, bg_offset_ub];

    let result = iter_add_modes(
        hist_yx.view(),
        &so_axis,
        &freq_axis,
        // Power overlap on full bins — MATLAB `mode_overlap.m` always
        // uses the un-clipped `SOpow_bins` × `freq_bins`.
        so_bins,
        freq_bins,
        &mode_params,
        &lb_single,
        &ub_single,
        bg_initial,
        bg_lower,
        bg_upper,
        opts.max_peaks,
        opts.max_overlap,
        opts.min_pctr2,
        opts.min_freq_diff,
        opts.min_amp,
        ParamKind::Power,
    )?;

    // Re-paint the model on the original (n_so, n_freq) grid (zeros outside
    // the analysis window).
    let mut model = Array2::<f64>::zeros((n_so, n_freq));
    for (yi, &fi) in freq_idx.iter().enumerate() {
        for (xi, &si) in so_idx.iter().enumerate() {
            model[[si, fi]] = result.model_yx[[yi, xi]];
        }
    }

    Ok(ParamFitResult {
        params: result.params,
        background: result.background,
        unit_row: f64::NAN,
        model,
        gof: result.gof,
    })
}

/// Max of a (sorted-ascending, finite) bin axis.
fn axis_max(axis: &[f64]) -> f64 {
    axis[axis.len() - 1]
}

/// Fit von-Mises × Gaussian modes + sinusoidal background to the
/// SO-phase histogram. Mirrors `param_basis_phase.m`: runs the periodic
/// histogram watershed for initial conditions, then iteratively adds
/// modes, and finally rewrites mode amplitudes to empirical values.
pub fn fit_paramfit_phase(
    sophase_hist: ArrayView2<f64>,
    phase_bins: &[f64],
    freq_bins: &[f64],
    opts: &crate::paramfit::opts::PhaseParamFitOpts,
) -> Result<ParamFitResult, FitError> {
    let (n_ph, n_freq) = sophase_hist.dim();
    if phase_bins.len() != n_ph {
        return Err(FitError::Invalid {
            axis: "phase",
            msg: format!("phase_bins {} != hist rows {}", phase_bins.len(), n_ph),
        });
    }
    if freq_bins.len() != n_freq {
        return Err(FitError::Invalid {
            axis: "phase",
            msg: format!("freq_bins {} != hist cols {}", freq_bins.len(), n_freq),
        });
    }
    if n_ph < 4 || n_freq < 4 {
        return Err(FitError::Invalid {
            axis: "phase",
            msg: format!("hist too small ({}×{})", n_ph, n_freq),
        });
    }

    // Periodic-phase watershed — mirrors `param_basis_phase.m:166-185`
    // (triple the histogram across ±π, imgaussfilt with gauss_filt_std,
    // extracthistpeaks). Params come from `PhaseParamFitOpts.watershed`.
    // `gauss_filt_std = [σ_x, σ_y]` in MATLAB on a (freq, phase) image;
    // our histogram is (phase, freq) so we pass `(σ_x, σ_y)` swapped.
    let wshed_params = opts.watershed.to_hist_peak_params();
    // MATLAB phase image is (freq, phase) with gauss_filt_std = [σ_x, σ_y]
    // where x=phase, y=freq. Our histogram is (phase, freq) so the σs go
    // in flipped: pass (σ_phase, σ_freq) = (gauss_filt_std[1], gauss_filt_std[0]).
    let gauss_filt = (opts.gauss_filt_std[1], opts.gauss_filt_std[0]);
    let (_labels, peaks) =
        extract_hist_peaks_phase(sophase_hist, phase_bins, freq_bins, &wshed_params, gauss_filt);
    // Phase has no min_freq_diff in MATLAB defaults; pass 0 so the
    // upstream watershed dedup is inert on the phase axis.
    let mut mode_params = watershed_mode_params(
        &peaks,
        (opts.freq_limits[0], opts.freq_limits[1]),
        0.0,
    );

    let (ph_idx, ph_axis, freq_idx, freq_axis) = valid_bins(
        sophase_hist,
        phase_bins,
        freq_bins,
        (opts.phase_limits[0], opts.phase_limits[1]),
        (opts.freq_limits[0], opts.freq_limits[1]),
    );
    if ph_axis.len() < 4 || freq_axis.len() < 4 {
        return Err(FitError::Invalid {
            axis: "phase",
            msg: format!(
                "not enough valid data in the fit window \
                 (post-clip histogram {}×{} valid bins; need at least 4 per axis)",
                ph_axis.len(),
                freq_axis.len()
            ),
        });
    }

    let mut hist_yx = Array2::<f64>::zeros((freq_axis.len(), ph_axis.len()));
    for (yi, &fi) in freq_idx.iter().enumerate() {
        for (xi, &si) in ph_idx.iter().enumerate() {
            hist_yx[[yi, xi]] = sophase_hist[[si, fi]];
        }
    }

    let amp0: Vec<f64> = if mode_params.is_empty() {
        hist_yx.iter().copied().filter(|v| v.is_finite()).collect()
    } else {
        mode_params.iter().map(|r| r[0]).collect()
    };
    let (amp_lo, amp_hi) = amp_bounds(&amp0);

    // Watershed-failure failsafe (param_basis_phase.m:339-354).
    if mode_params.is_empty() {
        let fmean = (axis_max(&freq_axis) + freq_axis[0]) / 2.0;
        // Both range/4 widths are priors, already in the parameter units
        // (`fstd` a Gaussian σ, `pstd` a `recikappa`).
        let fstd = (axis_max(&freq_axis) - freq_axis[0]) / 4.0;
        let pmean = (axis_max(&ph_axis) + ph_axis[0]) / 2.0;
        let pstd = (axis_max(&ph_axis) - ph_axis[0]) / 4.0;
        let amp = finite_max(hist_yx.iter().copied()).max(1e-6);
        mode_params = vec![[amp, fmean, fstd, pmean, pstd, 0.0]];
    }

    // Per-mode bounds. `phasepref` (col 3, 0-indexed) starts at `±inf` here
    // and is bounded further down when `constrain_phase_center` is on.
    // Everything else comes from `opts.mode_bounds` (fstd, pstd=recikappa,
    // theta) or is data-derived (amp, fmean).
    let mb = &opts.mode_bounds;
    let mut lb_single: [f64; 6] = [
        mb.amp_lb.resolve(amp_lo),
        mb.fmean_lb.resolve(freq_axis[0]),
        mb.fstd_lb,
        f64::NEG_INFINITY,
        mb.pstd_lb,
        mb.theta_lb,
    ];
    let mut ub_single: [f64; 6] = [
        mb.amp_ub.resolve(amp_hi),
        mb.fmean_ub.resolve(freq_axis[freq_axis.len() - 1]),
        mb.fstd_ub,
        f64::INFINITY,
        mb.pstd_ub,
        mb.theta_ub,
    ];
    // Per-axis center constraints. Frequency is slot 1. The SO-phase center
    // is slot 3, which the array above initializes to ±inf. When frequency
    // is unconstrained, free it to ±inf. When the phase center IS
    // constrained, bound it to two full periods [-2π, 2π] (MATLAB
    // param_basis_opts('phase') defaults, DYNAM-O PR #74). The two-period
    // span keeps every circular phase class reachable while letting the LM
    // cross the ±π seam. The final estimate is wrapped back to [-π, π].
    warn_discarded_center_bounds("freq", opts.constrain_freq_center, mb.fmean_lb, mb.fmean_ub);
    warn_discarded_center_bounds("phase", opts.constrain_phase_center, mb.pmean_lb, mb.pmean_ub);
    if !opts.constrain_freq_center {
        lb_single[1] = f64::NEG_INFINITY;
        ub_single[1] = f64::INFINITY;
    }
    if opts.constrain_phase_center {
        // Explicit bounds win; otherwise two full periods so the LM can
        // cross the ±π seam. Unconstrained leaves the ±inf set above.
        lb_single[3] = mb.pmean_lb.resolve(-2.0 * PI);
        ub_single[3] = mb.pmean_ub.resolve(2.0 * PI);
    }

    // Background bounds — MATLAB `fit_vmGauss.m`:
    //   StartPoint = […, 0, 0, prctile(phase_hist(:), 5)]
    //   Lower      = […, -1, -π, 0]
    //   Upper      = […,  1,  π, max(phase_hist,[],"all")]
    // Bounding the sinusoid amplitude (`SinAmp`, the `sin(x + SinPhase)`
    // amplitude) is essential: with row normalization an unbounded
    // amplitude is a degenerate fit direction — the row normalizer cancels
    // its scale — so an unbounded LM drifts into a pure-sinusoid model and
    // squashes the real vmGauss modes. (The fit calls the three background
    // coefficients `xxx/yyy/zzz` internally. The emitted CSV names them
    // SinAmp/SinPhase/Offset.)
    let bg_offset0 = percentile_finite(hist_yx.iter().copied(), 5.0);
    let bg_zmax = finite_max(hist_yx.iter().copied());
    let bg_initial = [0.0_f64, 0.0, bg_offset0];
    let bb = &opts.background_bounds;
    let bg_offset_ub = bb
        .offset_ub
        .resolve(if bg_zmax.is_finite() { bg_zmax } else { 1.0 });
    let bg_lower = [bb.sin_amp_lb, bb.sin_phase_lb, bb.offset_lb];
    let bg_upper = [bb.sin_amp_ub, bb.sin_phase_ub, bg_offset_ub];

    let result = iter_add_modes(
        hist_yx.view(),
        &ph_axis,
        &freq_axis,
        // Phase overlap on full bins — MATLAB `mode_overlap.m` always
        // uses the un-clipped `phase_bins` × `freq_bins`. Evaluating on
        // the fit-clipped frequency subset truncates the modes' tails,
        // raises `sum(min)/sum(max)` artificially, and would reject the
        // 3-mode phase fit at ~17% overlap that MATLAB accepts at ~15%.
        phase_bins,
        freq_bins,
        &mode_params,
        &lb_single,
        &ub_single,
        bg_initial,
        bg_lower,
        bg_upper,
        opts.max_peaks,
        opts.max_overlap,
        opts.min_pctr2,
        0.0, // phase: no min_freq_diff in MATLAB defaults
        opts.min_amp,
        ParamKind::Phase,
    )?;

    // Final transform (param_basis_phase.m:691-710): rewrite each mode's
    // amplitude to the empirical value of the no-sinusoid model at the
    // mode center, and wrap the phase preference to [-π, π].
    let mut params = result.params.clone();
    if params.nrows() > 0 {
        let e_amps = phase_empirical_amps(&params, result.background, &ph_axis, &freq_axis);
        for (m, &a) in e_amps.iter().enumerate() {
            params[[m, 0]] = a;
        }
        for m in 0..params.nrows() {
            params[[m, 3]] = wrap_to_pi(params[[m, 3]]);
        }
    }

    let mut model = Array2::<f64>::zeros((n_ph, n_freq));
    for (yi, &fi) in freq_idx.iter().enumerate() {
        for (xi, &si) in ph_idx.iter().enumerate() {
            model[[si, fi]] = result.model_yx[[yi, xi]];
        }
    }

    Ok(ParamFitResult {
        params,
        background: result.background,
        // Per-row normalization is ON (a PI decision). Phase paramfit CSVs
        // carry `unit_row: 1.0` so the figure renderer re-applies the per-row
        // normalizer when rebuilding the surface, matching how the fit ran.
        unit_row: 1.0,
        model,
        gof: result.gof,
    })
}

// ---------------------------------------------------------------------
// Iter-add-mode loop
// ---------------------------------------------------------------------

/// Per-iteration accepted-fit snapshot. `model_yx` matches the kernel
/// orientation `(n_freq, n_so)`.
struct IterFitResult {
    params: Array2<f64>,
    background: [f64; 3],
    model_yx: Array2<f64>,
    gof: GofValues,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ParamKind {
    Power,
    Phase,
}

/// Run the inner LM fit at a fixed mode count. `initial`/`lower`/`upper`
/// are `(n_modes, 6)`; `n_modes == 0` fits the background plane only
/// (MATLAB's `fit_rotGauss(..., [], LBi, UBi)` path).
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn inner_fit(
    kind: ParamKind,
    hist_yx: ArrayView2<f64>,
    x_axis: &[f64],
    y_axis: &[f64],
    initial: ArrayView2<f64>,
    lower: ArrayView2<f64>,
    upper: ArrayView2<f64>,
    bg_initial: [f64; 3],
    bg_lower: [f64; 3],
    bg_upper: [f64; 3],
) -> Result<(Array2<f64>, [f64; 3], Array2<f64>, GofValues), FitError> {
    let axis_str = match kind {
        ParamKind::Power => "power",
        ParamKind::Phase => "phase",
    };
    let out = match kind {
        ParamKind::Power => rs_fit_rotgauss(
            hist_yx, x_axis, y_axis, initial, lower, upper, bg_initial, bg_lower, bg_upper, 300,
        )
        .map(|o| (o.params, o.background, o.model_soph, o.gof)),
        ParamKind::Phase => rs_fit_vmgauss(
            hist_yx, x_axis, y_axis, initial, lower, upper, bg_initial, bg_lower, bg_upper, 300,
            // Fit with the per-row normalizer ON (`'problem' = true`). This
            // is a PI decision and matches MATLAB `fit_vmGauss.m`'s
            // `'problem', true`: the un-normalized phase fit produces
            // washed-out, poorly-placed modes that don't track the row-
            // normalized SO-phase "Proportion" data. The LM uses a numeric
            // Jacobian (vm_gauss.rs), so the row normalizer needs no special
            // handling.
            true,
        )
        .map(|o| (o.params, o.background, o.model_soph, o.gof)),
    }
    .map_err(|e| FitError::Param {
        axis: axis_str,
        msg: format!("{}", e),
    })?;
    Ok((out.0, out.1, out.2, out.3.into()))
}

/// Generic iter-add-mode driver — ports `param_basis_{power,phase}.m`'s
/// outer loop. `mode_params` are the watershed-derived (or failsafe
/// synthetic) seeds; `n_wshed = mode_params.len()`.
///
/// `overlap_x_axis` / `overlap_y_axis` are the **un-clipped** axes used to
/// compute mode-overlap (MATLAB `mode_overlap.m` evaluates each mode on
/// the full `phase_bins` × `freq_bins`, not the fit-clipped subset).
/// Including the out-of-clip tails — where both modes are near-zero —
/// raises `sum(max)` proportionally more than `sum(min)`, so the overlap
/// ratio comes out lower than on the clipped axes. On one phase fit the
/// clipped axes give 17% where MATLAB's full grid gives 15%. Both the power
/// and the phase path pass the full bin axes here.
#[allow(clippy::too_many_arguments)]
fn iter_add_modes(
    hist_yx: ArrayView2<f64>,
    x_axis: &[f64],
    y_axis: &[f64],
    overlap_x_axis: &[f64],
    overlap_y_axis: &[f64],
    mode_params: &[[f64; 6]],
    lb_single: &[f64; 6],
    ub_single: &[f64; 6],
    bg_initial: [f64; 3],
    bg_lower: [f64; 3],
    bg_upper: [f64; 3],
    max_peaks: usize,
    max_overlap: f64,
    min_pctr2: f64,
    min_freq_diff: f64,
    min_amp: f64,
    kind: ParamKind,
) -> Result<IterFitResult, FitError> {
    let axis_str = match kind {
        ParamKind::Power => "power",
        ParamKind::Phase => "phase",
    };
    let n_wshed = mode_params.len();
    if n_wshed == 0 {
        return Err(FitError::Param {
            axis: axis_str,
            msg: "no seed modes (watershed and failsafe both empty)".into(),
        });
    }

    let mut accepted: Option<IterFitResult> = None;
    // Params of the last accepted iteration (MATLAB's carried-over `B0i`).
    let mut accepted_modes: Vec<[f64; 6]> = Vec::new();
    let mut prev_adjr2: f64 = 0.0;
    // Phase-only: watershed-mode indices still in play (MATLAB `valid_peaks`).
    let mut valid_peaks: Vec<bool> = vec![true; n_wshed];

    for ii in 1..=max_peaks {
        // Build the initial guess B0i for this mode count.
        let b0i: Vec<[f64; 6]> = if ii <= n_wshed {
            match kind {
                // Power (param_basis_power.m:330): append the ii-th
                // watershed seed onto the last-accepted params.
                ParamKind::Power => {
                    let mut v = accepted_modes.clone();
                    v.push(mode_params[ii - 1]);
                    v
                }
                // Phase (param_basis_phase.m:413-417): re-seed from the
                // raw watershed params for indices {1..ii} that haven't
                // been reverted.
                ParamKind::Phase => (1..=ii)
                    .filter(|&k| valid_peaks[k - 1])
                    .map(|k| mode_params[k - 1])
                    .collect(),
            }
        } else {
            // Beyond the watershed modes: seed the next mode at the
            // (x, y) argmax of the residual `hist − last_model`.
            // Matching-pursuit style — each added mode targets the
            // largest currently-unfit feature. MATLAB `param_basis_*.m`
            // does the same via `residual_max_seed` over the window
            // submatrix; both fall back to the `mean(B0i, 1)` seed when
            // the residual is everywhere non-positive (the model already
            // covers the data).
            //
            // Why residual seeding rather than always re-seeding from the
            // mean of the accepted modes: 800184/C3 has the
            // spindle as its only watershed seed, so a mean-based re-seed
            // lands back on the spindle and the 7.6 Hz alpha never gets
            // seeded — the LM terminates at 1 mode. Residual seeding fits
            // 3 modes (spindle + alpha + slow), adjR² 0.834 → 0.969; six
            // of seven CFS C3 subjects gained 0.10 – 0.43 adjR².
            //
            // FreqStd / SOpowerStd for the new seed default to the
            // median of the accepted modes' stds (a reasonable mode-
            // width prior). Theta starts at 0.
            let mut v = accepted_modes.clone();
            let new_seed = match accepted.as_ref() {
                Some(acc) => residual_max_seed(
                    hist_yx, &acc.model_yx, x_axis, y_axis, &accepted_modes,
                    min_freq_diff,
                )
                .unwrap_or_else(|| mean_row(&accepted_modes)),
                None => mean_row(&accepted_modes),
            };
            v.push(new_seed);
            v
        };
        let n_modes = b0i.len();
        if n_modes == 0 {
            // All watershed seeds for {1..ii} were reverted — nothing to
            // fit this round; continue to pick up the next seed.
            continue;
        }

        // Per-mode bounds = repeated single-mode bounds (one row per
        // mode); MATLAB keeps LBi/UBi row-count in sync with B0i.
        let mut initial = Array2::<f64>::zeros((n_modes, 6));
        let mut lower = Array2::<f64>::zeros((n_modes, 6));
        let mut upper = Array2::<f64>::zeros((n_modes, 6));
        for m in 0..n_modes {
            for c in 0..6 {
                initial[[m, c]] = b0i[m][c];
                lower[[m, c]] = lb_single[c];
                upper[[m, c]] = ub_single[c];
            }
        }

        let (params, background, model_yx, gof) = inner_fit(
            kind,
            hist_yx,
            x_axis,
            y_axis,
            initial.view(),
            lower.view(),
            upper.view(),
            bg_initial,
            bg_lower,
            bg_upper,
        )?;

        // %-Δ-adjR² vs the previous accepted iteration.
        let diff_pct = if ii == 1 {
            0.0
        } else {
            (gof.adjrsquare - prev_adjr2) / prev_adjr2.abs().max(1e-12)
        };

        // Mode overlap (only meaningful with ≥2 modes). MATLAB
        // `mode_overlap.m` evaluates on the full `phase_bins` ×
        // `freq_bins`, not the fit-clipped axes; using clipped axes
        // here inflates the overlap ratio and rejects perfectly good
        // 3-mode phase fits (iter 3 at ol_max=17% vs MATLAB's 15%).
        let ol_max = if n_modes > 1 {
            max_pairwise_overlap(&params, overlap_x_axis, overlap_y_axis, kind)
        } else {
            0.0
        };

        // Revert criteria — mirror param_basis_{power,phase}.m:391-438.
        let mut revert = false;
        if ii > 1 && diff_pct < min_pctr2 {
            revert = true; // minpctr2 criterion
        }
        if ol_max > max_overlap {
            revert = true;
        }
        match kind {
            ParamKind::Power => {
                // any(B0i(:,1) < min_amp) — fitted amplitudes.
                if (0..n_modes).any(|m| params[[m, 0]] < min_amp) {
                    revert = true;
                }
            }
            ParamKind::Phase => {
                // any(e_amp < min_amp) — empirical amplitude of the
                // no-sinusoid model at each mode center.
                let e_amps = phase_empirical_amps(&params, background, x_axis, y_axis);
                if e_amps.iter().any(|&a| a < min_amp) {
                    revert = true;
                }
            }
        }
        // Min freq distance (power only — MATLAB `pdist(B0i(:,2))`).
        if matches!(kind, ParamKind::Power) && min_freq_diff > 0.0 && n_modes > 1 {
            'freq: for a in 0..n_modes {
                for b in (a + 1)..n_modes {
                    if (params[[a, 1]] - params[[b, 1]]).abs() < min_freq_diff {
                        revert = true;
                        break 'freq;
                    }
                }
            }
        }

        log::debug!(
            "paramfit iter axis={axis_str} ii={ii} n_modes={n_modes} n_wshed={n_wshed} \
             adjr2={} diff_pct={diff_pct} ol_max={ol_max} revert={revert}",
            gof.adjrsquare
        );

        // One-shot diagnostic trace for MATLAB-vs-Rust per-iter diffs.
        // Enabled via DYNAMO_PARAMFIT_TRACE=1; off in normal runs.
        if std::env::var("DYNAMO_PARAMFIT_TRACE").ok().as_deref() == Some("1") {
            eprintln!(
                "\n========== {} iter {} ==========",
                axis_str.to_uppercase(), ii
            );
            eprintln!("   adjr2:    {:.6}", gof.adjrsquare);
            eprintln!("   diff%:    {:.6}", diff_pct);
            eprintln!("   ol_max:   {:.4} ({:.1}%)", ol_max, ol_max * 100.0);
            eprintln!("   revert:   {}", revert);
            eprintln!(" B0i:");
            for r in &b0i {
                eprintln!(
                    "   {:>8.4} {:>8.4} {:>8.4} {:>8.4} {:>8.4} {:>+8.4}",
                    r[0], r[1], r[2], r[3], r[4], r[5]
                );
            }
            eprintln!(" Fit Parameters:");
            for m in 0..n_modes {
                eprintln!(
                    "   {:>8.4} {:>8.4} {:>8.4} {:>8.4} {:>8.4} {:>+8.4}",
                    params[[m, 0]], params[[m, 1]], params[[m, 2]],
                    params[[m, 3]], params[[m, 4]], params[[m, 5]]
                );
            }
            eprintln!(" Background xxx={:.4} yyy={:.4} zzz={:.4}",
                background[0], background[1], background[2]);
        }

        if revert {
            if ii == 1 {
                // MATLAB: the single seed mode failed — fit the
                // background plane only and terminate.
                let empty = Array2::<f64>::zeros((0, 6));
                let (p, bg, model_yx, gof) = inner_fit(
                    kind,
                    hist_yx,
                    x_axis,
                    y_axis,
                    empty.view(),
                    empty.view(),
                    empty.view(),
                    bg_initial,
                    bg_lower,
                    bg_upper,
                )?;
                return Ok(IterFitResult {
                    params: p,
                    background: bg,
                    model_yx,
                    gof,
                });
            }
            if matches!(kind, ParamKind::Phase) && ii <= n_wshed {
                valid_peaks[ii - 1] = false;
            }
            // A revert past the watershed modes terminates the loop; a
            // revert within them just drops this seed and continues.
            if ii > n_wshed {
                break;
            }
            continue;
        }

        // Accept this iteration.
        accepted_modes = (0..n_modes)
            .map(|m| {
                [
                    params[[m, 0]],
                    params[[m, 1]],
                    params[[m, 2]],
                    params[[m, 3]],
                    params[[m, 4]],
                    params[[m, 5]],
                ]
            })
            .collect();
        prev_adjr2 = gof.adjrsquare;
        accepted = Some(IterFitResult {
            params,
            background,
            model_yx,
            gof,
        });
    }

    accepted.ok_or(FitError::Param {
        axis: axis_str,
        msg: "no iteration produced a usable fit".into(),
    })
}

/// Empirical mode amplitudes for the phase fit — mirrors
/// `param_basis_phase.m`'s `e_amp` / final amplitude rewrite: drop the
/// background sinusoid (`xxx = 0`), evaluate the vmGauss model, and
/// read its value at each mode's `(fmean, phasepref)` center.
///
/// With the per-row normalizer ON, the empirical amplitude is read from the
/// *normalized* no-sinusoid model, as `param_basis_phase.m` does ("Update
/// the amplitude to empirical values from normalized SOPhH"). So
/// `unit_row = true` is passed to `vm_eval_model`.
fn phase_empirical_amps(
    params: &Array2<f64>,
    background: [f64; 3],
    ph_axis: &[f64],
    freq_axis: &[f64],
) -> Vec<f64> {
    let n = params.nrows();
    if n == 0 {
        return Vec::new();
    }
    let mut flat = vec![0.0_f64; 6 * n + 3];
    for m in 0..n {
        for c in 0..6 {
            flat[c * n + m] = params[[m, c]];
        }
    }
    flat[6 * n] = 0.0; // xxx → 0 (drop the background sinusoid)
    flat[6 * n + 1] = background[1];
    flat[6 * n + 2] = background[2];
    // `unit_row = true`: read the empirical amplitude from the ROW-NORMALIZED
    // no-sinusoid model, so phase Density is a proportion (per-row sums to 1),
    // matching the row-normalized phase fit (`problem = true`) and MATLAB
    // `param_basis_phase.m`. Reading the un-normalized model here would
    // inflate Density/Volume by ~1/row_sum and desensitize the `min_amp`
    // revert (calibrated for proportion-scale amplitudes), which would let
    // low-proportion phase modes survive that MATLAB rejects.
    let model = vm_eval_model(&flat, n, ph_axis, freq_axis, true);
    (0..n)
        .map(|m| {
            let fi = nearest_idx(freq_axis, params[[m, 1]]);
            let xi = nearest_circular_idx(ph_axis, params[[m, 3]]);
            model[[fi, xi]]
        })
        .collect()
}

/// Undo [`phase_empirical_amps`]: recover the amplitudes the phase fit
/// solved for from the empirical amplitudes a paramfit CSV records.
///
/// The CSV's phase `Density` is the value of the row-normalized,
/// sinusoid-free model at each mode's center, not the fitted amplitude.
/// Rebuilding a surface with `Density` in the amplitude slot changes how
/// strong the modes are against the background, so the picture no longer
/// matches the fit. This finds the amplitudes `a` that give back the
/// recorded values.
///
/// For mode `k` with center row `f_k` and center phase bin `x_k`, write
/// `G[k][j]` for mode `j`'s unit-amplitude value at that center, `S[k][j]`
/// for its sum over row `f_k`, `z` for the background offset, `n_x` for the
/// number of phase bins and `t_k` for the recorded amplitude. The
/// normalized model at the center is `(G[k]·a + z) / (S[k]·a + z·n_x)`.
/// Setting it equal to `t_k` gives one linear equation per mode:
///
/// ```text
///   Σ_j (G[k][j] − t_k·S[k][j]) · a_j  =  z · (t_k·n_x − 1)
/// ```
///
/// With `z > 0` this square system pins `a` down exactly. With `z = 0` (or
/// zero up to rounding) the
/// row normalization cancels any common scale of `a`, so only the ratios
/// are determined. The scale is then chosen so the unnormalized rows at
/// the mode centers sum to 1 on average, the proportion scale of the
/// row-normalized histogram the fit ran on. In that case the rebuilt
/// surface is close to the fit's but not exact, because the true scale
/// against the background sinusoid is not recorded.
///
/// `modes` rows are `[amp, fmean, fstd, pmean, pstd, theta]` with the
/// recorded amplitude in slot 0. `phase_bins` must be the bins the fit
/// normalized over (see [`rebuild_phase_model`]). Returns `None` when the
/// system has no usable solution (singular, or a non-positive amplitude),
/// so the caller can fall back to the recorded values.
fn phase_fit_amps_from_empirical(
    modes: &[[f64; 6]],
    background: [f64; 3],
    phase_bins: &[f64],
    freq_bins: &[f64],
) -> Option<Vec<f64>> {
    let n = modes.len();
    if n == 0 || phase_bins.is_empty() || freq_bins.is_empty() {
        return None;
    }
    let n_x = phase_bins.len() as f64;
    let offset = background[2];

    // Each mode alone at unit amplitude, unnormalized, on the full grid.
    // Grid shape is (n_freq, n_phase).
    let unit_grids: Vec<Array2<f64>> = modes
        .iter()
        .map(|m| {
            let flat = [1.0, m[1], m[2], m[3], m[4], m[5], 0.0, 0.0, 0.0];
            vm_eval_model(&flat, 1, phase_bins, freq_bins, false)
        })
        .collect();

    // The center bins, found the same way the forward rewrite finds them.
    let centers: Vec<(usize, usize)> = modes
        .iter()
        .map(|m| (nearest_idx(freq_bins, m[1]), nearest_circular_idx(phase_bins, m[3])))
        .collect();

    // G[k][j], S[k][j] and the equation rows.
    let mut matrix = vec![vec![0.0_f64; n]; n];
    let mut rhs = vec![0.0_f64; n];
    let mut row_sums = vec![vec![0.0_f64; n]; n];
    for (k, &(fk, xk)) in centers.iter().enumerate() {
        let target = modes[k][0];
        for j in 0..n {
            let at_center = unit_grids[j][[fk, xk]];
            let row_sum: f64 = unit_grids[j].row(fk).sum();
            row_sums[k][j] = row_sum;
            matrix[k][j] = at_center - target * row_sum;
        }
        rhs[k] = offset * (target * n_x - 1.0);
    }

    // An offset that is zero up to rounding (the fit sits on its lower
    // bound, e.g. 4e-17 against amplitudes near 0.01) carries no usable
    // scale, so it is handled as zero.
    let largest_recorded = modes.iter().fold(0.0_f64, |acc, m| acc.max(m[0].abs()));
    let offset_carries_scale = offset > 1e-9 * largest_recorded;
    let amps = if offset_carries_scale {
        solve_linear(matrix, rhs)?
    } else {
        // Homogeneous system: only the amplitude ratios are determined.
        // Add the scale condition "mean over modes of S[k]·a = 1" and solve
        // the n + 1 equations by least squares.
        let mut scale_row = vec![0.0_f64; n];
        for row in &row_sums {
            for j in 0..n {
                scale_row[j] += row[j] / n as f64;
            }
        }
        matrix.push(scale_row);
        rhs.push(1.0);
        solve_least_squares(&matrix, &rhs)?
    };

    let usable = amps.iter().all(|a| a.is_finite() && *a > 0.0);
    if usable {
        Some(amps)
    } else {
        None
    }
}

/// Rebuild a phase fit's model surface from what its paramfit CSV records,
/// so it matches the model the fit itself produced.
///
/// Two things differ between the CSV and the fit, and both are undone here:
///
/// * The recorded amplitudes are empirical values, not the fitted ones.
///   The fitted amplitudes are recovered with
///   [`phase_fit_amps_from_empirical`]. When that has no usable solution
///   the recorded values are used as they are.
/// * The fit normalizes each frequency row over the phase bins inside its
///   window, [-π, π] by default. The grid's last bin sits at π plus one
///   rounding step, so the fit leaves it out and sums 100 of the 101 bins.
///   The rebuild normalizes over the same bins. The left-out bin is the
///   same angle as the first, so it shows the same value.
///
/// Returns `(n_freq, n_phase)` on the full grid, like `eval_model`.
pub fn rebuild_phase_model(
    modes: &[[f64; 6]],
    background: [f64; 3],
    phase_bins: &[f64],
    freq_bins: &[f64],
) -> Array2<f64> {
    let n = modes.len();
    let limits = default_phase_fit_limits();
    let in_window: Vec<bool> = phase_bins
        .iter()
        .map(|&v| v >= limits[0] && v <= limits[1])
        .collect();
    let window_bins: Vec<f64> = phase_bins
        .iter()
        .zip(&in_window)
        .filter(|(_, inside)| **inside)
        .map(|(v, _)| *v)
        .collect();

    let mut flat = vec![0.0_f64; 6 * n + 3];
    for (m, mode) in modes.iter().enumerate() {
        for c in 0..6 {
            flat[c * n + m] = mode[c];
        }
    }
    flat[6 * n..].copy_from_slice(&background);
    if let Some(amps) = phase_fit_amps_from_empirical(modes, background, &window_bins, freq_bins) {
        flat[..n].copy_from_slice(&amps);
    }

    // Unnormalized on the full grid, then each row divided by its sum over
    // the window bins.
    let mut surface = vm_eval_model(&flat, n, phase_bins, freq_bins, false);
    for mut row in surface.rows_mut() {
        let mut window_sum = 0.0;
        for (value, inside) in row.iter().zip(&in_window) {
            if *inside {
                window_sum += *value;
            }
        }
        // A row that sums to exactly zero cannot be normalized and is left
        // as it is, as the fit's own normalizer does.
        if window_sum.abs() > 0.0 {
            row.mapv_inplace(|v| v / window_sum);
        }
    }
    surface
}

/// The phase fit's default window (`PhaseParamFitOpts::phase_limits`). A
/// paramfit CSV does not record the window, so a rebuild assumes the
/// default.
fn default_phase_fit_limits() -> [f64; 2] {
    crate::paramfit::opts::PhaseParamFitOpts::default().phase_limits
}

/// Solve the square system `matrix · x = rhs` by Gaussian elimination with
/// partial pivoting. `None` when the matrix is singular.
fn solve_linear(mut matrix: Vec<Vec<f64>>, mut rhs: Vec<f64>) -> Option<Vec<f64>> {
    let n = rhs.len();
    let scale = matrix
        .iter()
        .flatten()
        .fold(0.0_f64, |acc, v| acc.max(v.abs()));
    if scale == 0.0 {
        return None;
    }
    for col in 0..n {
        // Bring the largest remaining entry of this column to the diagonal.
        let mut pivot = col;
        for row in (col + 1)..n {
            if matrix[row][col].abs() > matrix[pivot][col].abs() {
                pivot = row;
            }
        }
        if matrix[pivot][col].abs() <= 1e-14 * scale {
            return None;
        }
        matrix.swap(col, pivot);
        rhs.swap(col, pivot);
        // Subtract a multiple of the pivot row from every row below it, so
        // this column is zero under the diagonal.
        let pivot_row = matrix[col].clone();
        for row in (col + 1)..n {
            let factor = matrix[row][col] / pivot_row[col];
            for (entry, pivot_entry) in matrix[row].iter_mut().zip(&pivot_row).skip(col) {
                *entry -= factor * pivot_entry;
            }
            rhs[row] -= factor * rhs[col];
        }
    }
    // Back substitution.
    let mut x = vec![0.0_f64; n];
    for row in (0..n).rev() {
        let mut sum = rhs[row];
        for k in (row + 1)..n {
            sum -= matrix[row][k] * x[k];
        }
        x[row] = sum / matrix[row][row];
    }
    Some(x)
}

/// Least-squares solution of the tall system `matrix · x ≈ rhs`, through
/// the normal equations (the systems here have at most a handful of
/// unknowns, so their conditioning is not a concern).
fn solve_least_squares(matrix: &[Vec<f64>], rhs: &[f64]) -> Option<Vec<f64>> {
    let n = matrix.first()?.len();
    let mut normal = vec![vec![0.0_f64; n]; n];
    let mut normal_rhs = vec![0.0_f64; n];
    for (row, &b) in matrix.iter().zip(rhs) {
        for i in 0..n {
            normal_rhs[i] += row[i] * b;
            for j in 0..n {
                normal[i][j] += row[i] * row[j];
            }
        }
    }
    solve_linear(normal, normal_rhs)
}

/// MATLAB `mode_overlap.m`: pairwise IoU = sum(min(g_a, g_b)) /
/// sum(max(g_a, g_b)). Returns 1.0 for identical modes, 0.0 for
/// disjoint. Caps at V_small/V_big when one mode is fully nested in
/// another. A containment metric, sum(min)/sum(smaller), would not have
/// that cap, but the nesting cases it targets (e.g. 800179/C3 iter-5) do
/// not arise with residual-max seeding and the watershed dedup: the
/// residual-driven seeds put new modes where the real unfit features are,
/// so the LM does not settle on a wide sheet engulfing a tight peak. The
/// IoU metric is what MATLAB uses, has clean set-theoretic semantics, and
/// does not false-alarm on modes that simply stack vertically in the
/// SO-power direction (800223/C3 iter 4 #1↔#3: the containment metric
/// reads 68.6% on a pair of clearly distinct features 1.3 Hz apart, where
/// IoU reads 40%).
///
/// Modes are evaluated separately on the (x_axis × y_axis) grid and
/// the worst pair is reported.
fn max_pairwise_overlap(
    params: &Array2<f64>,
    x_axis: &[f64],
    y_axis: &[f64],
    kind: ParamKind,
) -> f64 {
    let n_modes = params.nrows();
    if n_modes < 2 {
        return 0.0;
    }
    // Pre-evaluate each mode on the grid.
    let nx = x_axis.len();
    let ny = y_axis.len();
    let mut mode_grids: Vec<Vec<f64>> = Vec::with_capacity(n_modes);
    for m in 0..n_modes {
        // One mode, no background: overlap compares shapes. Evaluated through
        // the same kernel the fitter optimizes, so the two can never drift.
        // Widths are floored to stay finite. The fit's own lower bounds keep
        // them well above this in practice.
        let flat = [
            params[[m, 0]],
            params[[m, 1]],
            params[[m, 2]].max(1e-9),
            params[[m, 3]],
            params[[m, 4]].max(1e-9),
            params[[m, 5]],
            0.0,
            0.0,
            0.0,
        ];
        let z = match kind {
            ParamKind::Power => rg_eval_model(&flat, 1, x_axis, y_axis),
            // `false`: the row-normalizer would rescale each frequency row and
            // change the overlap ratio.
            ParamKind::Phase => vm_eval_model(&flat, 1, x_axis, y_axis, false),
        };
        debug_assert_eq!(z.dim(), (ny, nx));
        mode_grids.push(z.iter().copied().collect());
    }
    let mut worst = 0.0_f64;
    for a in 0..n_modes {
        for b in (a + 1)..n_modes {
            let ga = &mode_grids[a];
            let gb = &mode_grids[b];
            let mut s_min = 0.0_f64;
            let mut s_max = 0.0_f64;
            for k in 0..ga.len() {
                let va = ga[k];
                let vb = gb[k];
                s_min += va.min(vb);
                s_max += va.max(vb);
            }
            if s_max > 0.0 {
                let ov = s_min / s_max;
                if ov > worst {
                    worst = ov;
                }
            }
        }
    }
    worst
}

/// The SO-power splinefit as a run makes it: a SOPH with no finite value
/// (a channel that is all artifact) gets [`nan_splinefit_result`] on the
/// full grid instead of a fit, so its outputs exist and read as "no data".
pub fn splinefit_power_or_nan(
    sopower_hist: ArrayView2<f64>,
    so_bins: &[f64],
    freq_bins: &[f64],
    opts: &crate::paramfit::opts::SplinePowerOpts,
) -> Result<SplineFitResult, FitError> {
    let n_finite = sopower_hist.iter().filter(|v| v.is_finite()).count();
    if n_finite == 0 {
        return Ok(nan_splinefit_result(so_bins, freq_bins, opts.num_knots_x, opts.num_knots_y));
    }
    fit_splinefit_power(sopower_hist, so_bins, freq_bins, opts)
}

/// The SO-phase splinefit as a run makes it (see [`splinefit_power_or_nan`]).
pub fn splinefit_phase_or_nan(
    sophase_hist: ArrayView2<f64>,
    phase_bins: &[f64],
    freq_bins: &[f64],
    opts: &crate::paramfit::opts::SplinePhaseOpts,
) -> Result<SplineFitResult, FitError> {
    let n_finite = sophase_hist.iter().filter(|v| v.is_finite()).count();
    if n_finite == 0 {
        return Ok(nan_splinefit_result(phase_bins, freq_bins, opts.num_knots_x, opts.num_knots_y));
    }
    fit_splinefit_phase(sophase_hist, phase_bins, freq_bins, opts)
}

/// Tensor-product B-spline LSQ fit on the SO-power histogram.
/// Mirrors `spline_basis.m('power')` — SO/freq windows and knot counts
/// come from `opts` (MATLAB `spline_basis_opts('power')`); order 4 with
/// multiplicity-3 boundary knots stays fixed.
pub fn fit_splinefit_power(
    sopower_hist: ArrayView2<f64>,
    so_bins: &[f64],
    freq_bins: &[f64],
    opts: &crate::paramfit::opts::SplinePowerOpts,
) -> Result<SplineFitResult, FitError> {
    fit_splinefit_axis(
        sopower_hist,
        so_bins,
        freq_bins,
        (opts.power_limits[0], opts.power_limits[1]),
        (opts.freq_limits[0], opts.freq_limits[1]),
        opts.num_knots_x,
        opts.num_knots_y,
        "power",
    )
}

/// Tensor-product B-spline LSQ fit on the SO-phase histogram.
/// SO/freq windows and knot counts come from `opts` (MATLAB
/// `spline_basis_opts('phase')`).
pub fn fit_splinefit_phase(
    sophase_hist: ArrayView2<f64>,
    phase_bins: &[f64],
    freq_bins: &[f64],
    opts: &crate::paramfit::opts::SplinePhaseOpts,
) -> Result<SplineFitResult, FitError> {
    fit_splinefit_axis(
        sophase_hist,
        phase_bins,
        freq_bins,
        (opts.phase_limits[0], opts.phase_limits[1]),
        (opts.freq_limits[0], opts.freq_limits[1]),
        opts.num_knots_x,
        opts.num_knots_y,
        "phase",
    )
}

fn fit_splinefit_axis(
    hist: ArrayView2<f64>,
    so_bins: &[f64],
    freq_bins: &[f64],
    so_limits: (f64, f64),
    freq_limits: (f64, f64),
    num_knots_x: usize,
    num_knots_y: usize,
    axis: &'static str,
) -> Result<SplineFitResult, FitError> {
    let (n_so, n_freq) = hist.dim();
    if so_bins.len() != n_so || freq_bins.len() != n_freq {
        return Err(FitError::Invalid {
            axis,
            msg: format!(
                "shape mismatch: hist ({},{}), so_bins {}, freq_bins {}",
                n_so,
                n_freq,
                so_bins.len(),
                freq_bins.len()
            ),
        });
    }
    // `spline_basis.m:133-143` builds exactly the same masks the paramfit
    // does, so use the same helper. An ANY-finite test on BOTH axes would
    // be right for frequency and too loose for the SO axis, where MATLAB
    // requires every live frequency to be finite.
    let (so_idx, so_axis, freq_idx, freq_axis) =
        valid_bins(hist, so_bins, freq_bins, so_limits, freq_limits);
    if so_axis.len() < num_knots_x + 4 || freq_axis.len() < num_knots_y + 4 {
        return Err(FitError::Invalid {
            axis,
            msg: format!(
                "fit window too narrow: {} so × {} freq vs ({}+4)×({}+4)",
                so_axis.len(),
                freq_axis.len(),
                num_knots_x,
                num_knots_y
            ),
        });
    }

    // Build (n_x, n_y) sub-hist for the spline kernel (x=so, y=freq).
    let mut sub = Array2::<f64>::zeros((so_axis.len(), freq_axis.len()));
    for (xi, &si) in so_idx.iter().enumerate() {
        for (yi, &fi) in freq_idx.iter().enumerate() {
            let v = hist[[si, fi]];
            sub[[xi, yi]] = if v.is_finite() { v } else { 0.0 };
        }
    }

    // MATLAB knot recipe (spline_basis.m:162):
    //   knots = [min - 0.1, linspace(min, max, n), max + 0.1]
    let knots_x = build_knots(&so_axis, num_knots_x);
    let knots_y = build_knots(&freq_axis, num_knots_y);

    let out = rs_fit_spline(sub.view(), &so_axis, &freq_axis, &knots_x, &knots_y, 4, 3)
        .map_err(|e| FitError::Spline {
            axis,
            msg: format!("{}", e),
        })?;

    // Re-paint the splinefit on the full (n_so, n_freq) grid. Bins
    // outside the analysis window are filled with 0.0 — NOT NaN —
    // because (a) the TIFF writer crops to the analysis-window
    // submatrix (matching MATLAB `writeSplinefitFormats.m`, which
    // writes only `fitData.splinefit` over the valid bins), and
    // (b) the figure renderer needs a finite full-grid view to draw
    // the reconstruction over the full SOPH extent without stripes
    // from NaN-as-bit-pattern interpretation.
    let mut splinefit_full = Array2::<f64>::zeros((n_so, n_freq));
    for (xi, &si) in so_idx.iter().enumerate() {
        for (yi, &fi) in freq_idx.iter().enumerate() {
            splinefit_full[[si, fi]] = out.splinefit[[xi, yi]];
        }
    }

    Ok(SplineFitResult {
        coefs: out.coefs,
        splinefit: splinefit_full,
        knots_x_aug: out.knots_x_aug,
        knots_y_aug: out.knots_y_aug,
        fit_so_bins: so_axis,
        fit_freq_bins: freq_axis,
    })
}

/// Build a SplineFitResult filled with NaN, shaped against the full SOPH
/// grid. Used when the source histogram has no finite cells (the entire
/// channel is artifact) — the writer then produces a NaN-pixel TIFF and
/// the figure renderer produces a visibly-blank PNG, instead of failing
/// the item. Mirrors the orientations of a real fit:
///
///   - `coefs`: `(num_knots_y + 4, num_knots_x + 4)` of NaN (matches the
///     "squeeze(spline_obj.coefs)'" shape a real fit would produce).
///   - `splinefit`: `(n_so, n_freq)` of NaN (full SOPH grid).
///   - `knots_x_aug` / `knots_y_aug`: synthetic from the input bin range
///     so downstream metadata-readers still see plausible knot vectors.
///   - `fit_so_bins` / `fit_freq_bins`: the input bin centers verbatim
///     (the "fit" notionally covers the whole grid, just with NaN
///     everywhere).
pub fn nan_splinefit_result(
    so_bins: &[f64],
    freq_bins: &[f64],
    num_knots_x: usize,
    num_knots_y: usize,
) -> SplineFitResult {
    let n_so = so_bins.len();
    let n_freq = freq_bins.len();
    let coef_h = num_knots_y + 4;
    let coef_w = num_knots_x + 4;
    let lo_x = so_bins.first().copied().unwrap_or(0.0);
    let hi_x = so_bins.last().copied().unwrap_or(lo_x);
    let lo_y = freq_bins.first().copied().unwrap_or(0.0);
    let hi_y = freq_bins.last().copied().unwrap_or(lo_y);
    SplineFitResult {
        coefs: Array2::<f64>::from_elem((coef_h, coef_w), f64::NAN),
        splinefit: Array2::<f64>::from_elem((n_so, n_freq), f64::NAN),
        knots_x_aug: build_knots(&[lo_x, hi_x], num_knots_x),
        knots_y_aug: build_knots(&[lo_y, hi_y], num_knots_y),
        fit_so_bins: so_bins.to_vec(),
        fit_freq_bins: freq_bins.to_vec(),
    }
}

fn build_knots(axis: &[f64], n_internal: usize) -> Vec<f64> {
    // MATLAB: [min - 0.1, linspace(min, max, n), max + 0.1].
    //
    // Callers (`fit_splinefit_axis`) validate `axis.len() >= num_knots_x + 4`
    // (i.e. >= 5), so non-empty is guaranteed. But an all-NaN axis would
    // make the spline kernel return an error downstream — assert finite at
    // the boundary so the error is reported at the right layer.
    debug_assert!(!axis.is_empty(), "build_knots called with empty axis");
    let lo = axis.first().copied().filter(|v| v.is_finite()).unwrap_or(0.0);
    let hi = axis.last().copied().filter(|v| v.is_finite()).unwrap_or(lo);
    let mut k = Vec::with_capacity(n_internal + 2);
    k.push(lo - 0.1);
    if n_internal == 1 {
        k.push((lo + hi) * 0.5);
    } else {
        for i in 0..n_internal {
            let t = i as f64 / (n_internal - 1) as f64;
            k.push(lo + t * (hi - lo));
        }
    }
    k.push(hi + 0.1);
    k
}

/// MATLAB's `valid_freq_bins` / `valid_*_bins` masks, verbatim
/// (`param_basis_power.m:150-155`, `param_basis_phase.m:153-158`):
///
/// ```text
/// valid_mat    = isfinite(SOPH);            % (freq, so)
/// invalid_freq = all(~valid_mat, 2);        % freq rows that are entirely dead
/// valid_mat(invalid_freq, :) = true;        % neutralize them so they don't kill columns
/// valid_so   = so_bins   in limits & all(valid_mat, 1);   % ALL surviving freq rows finite
/// valid_freq = freq_bins in limits & ~invalid_freq;       % ANY finite value
/// ```
///
/// The asymmetry is deliberate and load-bearing. A frequency row survives if
/// it has *any* finite value, but an SO bin survives only if it is finite at
/// *every* live frequency. That is what drops the SO-power columns the
/// `min_time_in_bin` threshold turned into all-NaN, which in turn is what
/// keeps a fitted mode center from landing in a region with no data.
///
/// `hist` is indexed `[so, freq]`. Returns `(so_idx, so_axis, freq_idx,
/// freq_axis)`.
#[allow(clippy::type_complexity)]
fn valid_bins(
    hist: ArrayView2<f64>,
    so_bins: &[f64],
    freq_bins: &[f64],
    so_limits: (f64, f64),
    freq_limits: (f64, f64),
) -> (Vec<usize>, Vec<f64>, Vec<usize>, Vec<f64>) {
    let (n_so, n_freq) = hist.dim();

    // invalid_freq: freq columns of `hist` with no finite value anywhere.
    let dead_freq: Vec<bool> = (0..n_freq)
        .map(|f| !(0..n_so).any(|s| hist[[s, f]].is_finite()))
        .collect();

    let mut freq_idx = Vec::new();
    let mut freq_axis = Vec::new();
    for (f, &v) in freq_bins.iter().enumerate().take(n_freq) {
        if v >= freq_limits.0 && v <= freq_limits.1 && !dead_freq[f] {
            freq_idx.push(f);
            freq_axis.push(v);
        }
    }

    // An SO bin needs every LIVE frequency finite. Dead frequency rows were
    // neutralized above, so they cannot veto a column. Note this scans the
    // full frequency axis, not just the in-window part — matching MATLAB,
    // where a NaN at 1 Hz can eliminate an SO column even though 1 Hz is
    // never fit.
    let mut so_idx = Vec::new();
    let mut so_axis = Vec::new();
    for (s, &v) in so_bins.iter().enumerate().take(n_so) {
        if v < so_limits.0 || v > so_limits.1 {
            continue;
        }
        let all_live_finite = (0..n_freq)
            .all(|f| dead_freq[f] || hist[[s, f]].is_finite());
        if all_live_finite {
            so_idx.push(s);
            so_axis.push(v);
        }
    }
    (so_idx, so_axis, freq_idx, freq_axis)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::paramfit::opts::Bound;
    use ndarray::Array2;

    /// The CFS grid: 101 phase bins over [-π, π] and 81 frequency bins
    /// over 2 to 18 Hz.
    fn phase_grid() -> (Vec<f64>, Vec<f64>) {
        let phase: Vec<f64> = (0..101).map(|i| -PI + 2.0 * PI * i as f64 / 100.0).collect();
        let freq: Vec<f64> = (0..81).map(|j| 2.0 + 16.0 * j as f64 / 80.0).collect();
        (phase, freq)
    }

    /// Three overlapping, tilted modes, as rows `[amp, fmean, fstd, pmean,
    /// pstd, theta]`.
    fn three_phase_modes() -> Array2<f64> {
        Array2::from_shape_vec(
            (3, 6),
            vec![
                0.020, 13.8, 1.6, 0.4, 0.9, 0.5,
                0.011, 11.9, 2.4, -1.2, 1.1, 0.3,
                0.008, 6.5, 1.8, 2.5, 1.0, -0.4,
            ],
        )
        .unwrap()
    }

    /// The rewritten CSV amplitudes, as the paramfit writer stores them.
    fn recorded_modes(params: &Array2<f64>, empirical: &[f64]) -> Vec<[f64; 6]> {
        (0..params.nrows())
            .map(|m| {
                let mut row = [0.0; 6];
                for c in 0..6 {
                    row[c] = params[[m, c]];
                }
                row[0] = empirical[m];
                row
            })
            .collect()
    }

    #[test]
    fn empirical_phase_amps_invert_exactly_when_the_offset_is_positive() {
        let (phase, freq) = phase_grid();
        let params = three_phase_modes();
        let background = [0.0006, 0.7, 0.005];
        let empirical = phase_empirical_amps(&params, background, &phase, &freq);
        let recorded = recorded_modes(&params, &empirical);
        let recovered = phase_fit_amps_from_empirical(&recorded, background, &phase, &freq)
            .expect("a positive offset gives a unique solution");
        for m in 0..3 {
            let rel = (recovered[m] - params[[m, 0]]).abs() / params[[m, 0]];
            assert!(rel < 1e-9, "mode {m}: recovered {} vs fitted {}", recovered[m], params[[m, 0]]);
        }
    }

    /// The rebuild from CSV values reproduces the fit's own model. The
    /// grid's last bin sits one rounding step above π, as on real SOPHs, so
    /// the fit window holds 100 of the 101 bins.
    #[test]
    fn rebuilt_phase_model_matches_the_fit_model() {
        let (mut phase, freq) = phase_grid();
        phase[100] = f64::from_bits(PI.to_bits() + 1);
        let window: Vec<f64> = phase[..100].to_vec();
        let params = three_phase_modes();
        let background = [0.0006, 0.7, 0.005];

        // What the fit produces: the row-normalized model over the window,
        // and the empirical amplitudes it records.
        let mut flat = vec![0.0; 21];
        for m in 0..3 {
            for c in 0..6 {
                flat[c * 3 + m] = params[[m, c]];
            }
        }
        flat[18..].copy_from_slice(&background);
        let fit_model = vm_eval_model(&flat, 3, &window, &freq, true);
        let empirical = phase_empirical_amps(&params, background, &window, &freq);
        let recorded = recorded_modes(&params, &empirical);

        let rebuilt = rebuild_phase_model(&recorded, background, &phase, &freq);
        for fi in 0..freq.len() {
            for xi in 0..100 {
                let rel = (rebuilt[[fi, xi]] - fit_model[[fi, xi]]).abs() / fit_model[[fi, xi]];
                assert!(rel < 1e-9, "freq {fi} phase {xi}: {} vs {}", rebuilt[[fi, xi]], fit_model[[fi, xi]]);
            }
            // The left-out bin is the same angle as the first.
            let rel = (rebuilt[[fi, 100]] - rebuilt[[fi, 0]]).abs() / rebuilt[[fi, 0]];
            assert!(rel < 1e-9, "freq {fi}: +π and -π differ");
        }
    }

    #[test]
    fn empirical_phase_amps_keep_their_ratios_when_the_offset_is_zero() {
        let (phase, freq) = phase_grid();
        let params = three_phase_modes();
        let background = [0.0, 0.0, 0.0];
        let empirical = phase_empirical_amps(&params, background, &phase, &freq);
        let recorded = recorded_modes(&params, &empirical);
        let recovered = phase_fit_amps_from_empirical(&recorded, background, &phase, &freq)
            .expect("ratios are determined without an offset");
        // Only the common scale is lost, so the ratios come back exactly.
        for m in 1..3 {
            let fitted_ratio = params[[m, 0]] / params[[0, 0]];
            let recovered_ratio = recovered[m] / recovered[0];
            assert!((recovered_ratio / fitted_ratio - 1.0).abs() < 1e-9, "mode {m}");
        }
    }

    fn synth_hist(n_so: usize, n_freq: usize) -> (Array2<f64>, Vec<f64>, Vec<f64>) {
        // A clean single-mode bump centered in the SOPH window so the
        // single-mode fitter has something to lock onto.
        let so: Vec<f64> = (0..n_so)
            .map(|i| -2.0 + 22.0 * i as f64 / (n_so - 1) as f64)
            .collect();
        let fr: Vec<f64> = (0..n_freq)
            .map(|j| 0.0 + 30.0 * j as f64 / (n_freq - 1) as f64)
            .collect();
        let mut h = Array2::<f64>::zeros((n_so, n_freq));
        let p_mu = 5.0;
        let f_mu = 13.0;
        for i in 0..n_so {
            for j in 0..n_freq {
                let dp = (so[i] - p_mu) / 5.0;
                let df = (fr[j] - f_mu) / 2.0;
                h[[i, j]] = (-(dp * dp + df * df)).exp();
            }
        }
        (h, so, fr)
    }

    // A single bump with an arbitrary SO-power / frequency center, so a
    // test can place the true center OUTSIDE the fit window and observe
    // only its flank inside the window (SO width 6 dB, broad enough for
    // clear curvature across the window).
    fn synth_hist_at(n_so: usize, n_freq: usize, p_mu: f64, f_mu: f64) -> (Array2<f64>, Vec<f64>, Vec<f64>) {
        let so: Vec<f64> = (0..n_so)
            .map(|i| -2.0 + 22.0 * i as f64 / (n_so - 1) as f64)
            .collect();
        let fr: Vec<f64> = (0..n_freq)
            .map(|j| 0.0 + 30.0 * j as f64 / (n_freq - 1) as f64)
            .collect();
        let mut h = Array2::<f64>::zeros((n_so, n_freq));
        for i in 0..n_so {
            for j in 0..n_freq {
                let dp = (so[i] - p_mu) / 6.0;
                let df = (fr[j] - f_mu) / 2.0;
                h[[i, j]] = (-(dp * dp + df * df)).exp();
            }
        }
        (h, so, fr)
    }

    /// The warning must fire exactly when a bound is set AND its switch is
    /// off — that is the only case where the file expresses an intent the
    /// code discards. Silence otherwise, or it becomes noise people ignore.
    #[test]
    fn discarded_center_bound_warning_fires_only_when_bound_is_ignored() {
        // (constrained, lb, ub, should_warn)
        let cases = [
            (false, Bound::Value(12.0), Bound::Derived, true),
            (false, Bound::Derived, Bound::Value(16.0), true),
            (false, Bound::Value(12.0), Bound::Value(16.0), true),
            // An explicitly OPEN end is a user intent too, so discarding it warns.
            (false, Bound::NegInf, Bound::Derived, true),
            (false, Bound::Derived, Bound::Derived, false), // default rule — no intent
            (true, Bound::Value(12.0), Bound::Value(16.0), false), // bound applies
            (true, Bound::Derived, Bound::Derived, false),
        ];
        for (constrained, lb, ub, want) in cases {
            let fired = !(constrained || (!lb.is_set() && !ub.is_set()));
            assert_eq!(
                fired, want,
                "constrained={constrained} lb={lb:?} ub={ub:?}"
            );
            // Exercise the real function too, so a change to its guard breaks
            // this test rather than only the mirrored predicate above.
            warn_discarded_center_bounds("freq", constrained, lb, ub);
        }
    }

    /// MATLAB's asymmetric validity rule: an SO bin needs EVERY live
    /// frequency finite, a frequency bin needs ANY finite value. Getting
    /// this backwards silently widens the center bounds, which is the whole
    /// reason the rule exists.
    #[test]
    fn valid_bins_matches_matlab_asymmetric_rule() {
        let (n_so, n_freq) = (6usize, 5usize);
        let so_bins: Vec<f64> = (0..n_so).map(|i| i as f64).collect();
        let freq_bins: Vec<f64> = (0..n_freq).map(|i| i as f64).collect();
        let mut h = Array2::<f64>::ones((n_so, n_freq));

        // so bin 5 fully NaN: the min_time_in_bin case. Must be dropped.
        for f in 0..n_freq { h[[5, f]] = f64::NAN; }
        // freq bin 4 fully NaN: a dead row. Must be dropped, and must NOT
        // veto any SO bin (MATLAB neutralizes it before the `all`).
        for s in 0..n_so { h[[s, 4]] = f64::NAN; }
        // a single NaN at (so=2, freq=1): one live frequency is not finite,
        // so SO bin 2 must go, even though the bin has plenty of data.
        h[[2, 1]] = f64::NAN;

        let (so_idx, so_axis, freq_idx, freq_axis) =
            valid_bins(h.view(), &so_bins, &freq_bins, (0.0, 5.0), (0.0, 4.0));

        assert_eq!(so_idx, vec![0, 1, 3, 4], "so_axis = {so_axis:?}");
        assert_eq!(freq_idx, vec![0, 1, 2, 3], "freq_axis = {freq_axis:?}");
        // The dead frequency row must not have removed every SO bin, which
        // is what happens if the neutralization step is skipped.
        assert!(!so_idx.is_empty(), "a dead freq row wrongly vetoed every SO bin");
    }

    /// The background plane is what absorbs whatever the modes do not, so a
    /// bound on it is only meaningful if the solver actually sees it.
    #[test]
    fn explicit_background_bounds_are_honored() {
        let (h, so, fr) = synth_hist(40, 60);

        let mut opts = crate::paramfit::opts::PowerParamFitOpts::default();
        // Pin the plane flat and its offset to a narrow band.
        opts.background_bounds.pow_slope_lb = 0.0;
        opts.background_bounds.pow_slope_ub = 0.0;
        opts.background_bounds.freq_slope_lb = 0.0;
        opts.background_bounds.freq_slope_ub = 0.0;
        opts.background_bounds.offset_lb = 0.20;
        opts.background_bounds.offset_ub = Bound::Value(0.25);

        let r = fit_paramfit_power(h.view(), &so, &fr, &opts).expect("pinned-background fit");
        assert!(r.background[0].abs() < 1e-9, "PowSlope {} not pinned to 0", r.background[0]);
        assert!(r.background[1].abs() < 1e-9, "FreqSlope {} not pinned to 0", r.background[1]);
        assert!(
            (0.20 - 1e-6..=0.25 + 1e-6).contains(&r.background[2]),
            "Offset {} escaped the pinned band [0.20, 0.25]",
            r.background[2],
        );
    }

    /// `null` derives, a number pins. Mirrors MATLAB's NaN sentinel.
    #[test]
    fn explicit_mode_bounds_override_the_derived_ones() {
        let (h, so, fr) = synth_hist(40, 60);

        let mut opts = crate::paramfit::opts::PowerParamFitOpts::default();
        let derived = fit_paramfit_power(h.view(), &so, &fr, &opts).expect("derived fit");

        // Pin the frequency center to a narrow band well inside the window.
        opts.mode_bounds.fmean_lb = Bound::Value(12.0);
        opts.mode_bounds.fmean_ub = Bound::Value(12.5);
        let pinned = fit_paramfit_power(h.view(), &so, &fr, &opts).expect("pinned fit");

        for m in 0..pinned.params.nrows() {
            let f = pinned.params[[m, 1]];
            assert!(
                (12.0 - 1e-6..=12.5 + 1e-6).contains(&f),
                "mode {m} FreqMean {f} escaped the pinned bound [12.0, 12.5]",
            );
        }
        // Sanity: the derived fit was free to place a center outside that
        // band, so the assertion above is actually testing something.
        let derived_outside = (0..derived.params.nrows())
            .any(|m| !(12.0..=12.5).contains(&derived.params[[m, 1]]));
        assert!(derived_outside, "derived fit stayed inside the band by chance");
    }

    #[test]
    fn constrain_center_axes_are_independent() {
        // The true SO-power center (24) sits above the fit window's upper
        // edge (~20); the window sees only the rising flank.
        let (h, so, fr) = synth_hist_at(40, 60, 24.0, 13.0);
        let so_hi = *so.last().unwrap();
        let max_pmean = |r: &super::ParamFitResult| {
            r.params.column(3).iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        };
        let max_fmean = |r: &super::ParamFitResult| {
            r.params.column(1).iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        };

        // Constrained (default): every mode center pinned inside the window.
        let mut opts = crate::paramfit::opts::PowerParamFitOpts::default();
        // Upper edge of the frequency fit window (default [2, 18]) — a
        // constrained freq center can't exceed it.
        let fr_hi = opts.freq_limits[1];
        assert!(
            opts.constrain_power_center && opts.constrain_freq_center,
            "both axes should constrain by default",
        );
        let r_con = fit_paramfit_power(h.view(), &so, &fr, &opts).expect("constrained fit");
        assert!(
            max_pmean(&r_con) <= so_hi + 1e-6,
            "constrained pmean escaped the window: {} > {}",
            max_pmean(&r_con), so_hi,
        );

        // Free ONLY the SO-power axis: pmean escapes past the window edge,
        // but fmean must stay inside the frequency window (axis independence).
        opts.constrain_power_center = false;
        let r_free = fit_paramfit_power(h.view(), &so, &fr, &opts).expect("free-power fit");
        assert!(
            max_pmean(&r_free) > so_hi + 1.0,
            "freed SO-power center should escape the window: {} !> {}",
            max_pmean(&r_free), so_hi,
        );
        assert!(
            max_fmean(&r_free) <= fr_hi + 1e-6,
            "frequency center must stay constrained when only SO-power is freed: {} > {}",
            max_fmean(&r_free), fr_hi,
        );
    }

    #[test]
    fn power_fit_runs_on_synth_hist() {
        let (h, so, fr) = synth_hist(40, 60);
        let r = fit_paramfit_power(
            h.view(), &so, &fr,
            &crate::paramfit::opts::PowerParamFitOpts::default(),
        ).expect("paramfit power");
        assert!(r.params.nrows() >= 1, "expected ≥1 fitted mode");
        assert_eq!(r.params.ncols(), 6);
        assert!(r.gof.rsquare > 0.5, "rsquare={}", r.gof.rsquare);
    }

    #[test]
    fn spline_fit_runs_on_synth_hist() {
        let (h, so, fr) = synth_hist(40, 60);
        let r = fit_splinefit_power(
            h.view(),
            &so,
            &fr,
            &crate::paramfit::opts::SplinePowerOpts::default(),
        )
        .expect("splinefit power");
        assert_eq!(r.coefs.ndim(), 2);
        assert_eq!(r.splinefit.shape(), h.shape());
    }

    #[test]
    fn nan_splinefit_result_shapes_match_a_real_fit() {
        // Same axis dimensions a real fit_splinefit_power call would
        // produce for the twins' 121×71 SOpower histogram, so soph_build's
        // substitution is shape-compatible with the writer.
        let so: Vec<f64> = (0..121)
            .map(|i| -5.0 + 30.0 * i as f64 / 120.0)
            .collect();
        let fr: Vec<f64> = (0..71)
            .map(|j| 2.0 + 14.0 * j as f64 / 70.0)
            .collect();
        let r = nan_splinefit_result(&so, &fr, 5, 18);
        assert_eq!(r.coefs.shape(), [18 + 4, 5 + 4]);
        assert_eq!(r.splinefit.shape(), [121, 71]);
        assert!(r.coefs.iter().all(|v| v.is_nan()));
        assert!(r.splinefit.iter().all(|v| v.is_nan()));
        assert_eq!(r.fit_so_bins.len(), 121);
        assert_eq!(r.fit_freq_bins.len(), 71);
        // Knot vectors should be well-formed even with synthetic bounds.
        assert!(r.knots_x_aug.iter().all(|v| v.is_finite()));
        assert!(r.knots_y_aug.iter().all(|v| v.is_finite()));
    }

    /// Mirrors MATLAB `tests/test_residual_max_seed.m`: the width priors
    /// are unscaled σ-unit fallbacks (1.0, 5.0), both when there are no
    /// accepted modes and when the accepted widths yield invalid medians.
    #[test]
    fn residual_max_seed_uses_unscaled_width_fallbacks() {
        let hist = ndarray::arr2(&[[0.0, 0.0], [0.0, 3.0]]);
        let model = Array2::<f64>::zeros((2, 2));
        let x_axis = [10.0, 20.0];
        let y_axis = [1.0, 2.0];

        let seed = residual_max_seed(hist.view(), &model, &x_axis, &y_axis, &[], 0.0)
            .expect("positive residual must yield a seed");
        assert_eq!(seed, [3.0, 2.0, 1.0, 20.0, 5.0, 0.0]);

        // Accepted mode with zero widths → medians invalid → same fallbacks.
        let accepted = [[1.0, 1.0, 0.0, 10.0, 0.0, 0.0]];
        let seed = residual_max_seed(hist.view(), &model, &x_axis, &y_axis, &accepted, 0.0)
            .expect("positive residual must yield a seed");
        assert_eq!(seed, [3.0, 2.0, 1.0, 20.0, 5.0, 0.0]);
    }
}
