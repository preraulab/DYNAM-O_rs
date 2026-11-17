//! Assign per-peak metadata: stage, SO-power, SO-phase.
//!
//! Ports pydynamo `pipeline.py` lines ~205–248 (MATLAB `computePeakStage.m`,
//! `computePeakSOpower.m`, `computePeakSOphase.m`). Each function takes the
//! already-computed time-series metric on its own time grid plus the peak
//! times, and returns one value per peak.
//!
//! All inputs are assumed sorted by increasing time. No sorting is done here.
//!
//! Conventions follow the MATLAB functions exactly:
//!
//! * `assign_peak_stage` — previous-neighbor interp (step function) with
//!   fill=0 for peaks outside the staging range; artifact override sets the
//!   stage to 6 when the nearest `artifacts` sample is true
//!   (`interp1(..., 'nearest')`, an exact midpoint goes to the later sample).
//! * `assign_peak_sopower` — MATLAB's padded `interp1`: the series is
//!   extended by one grid step at each end with the endpoint value, so a
//!   query within one step of either end gets the endpoint value and a
//!   query farther out is NaN.
//! * `assign_peak_sophase` — the same padded interp on the *unwrapped*
//!   phase, then `wrapToPi` to the closed interval `[-π, π]`.

/// Previous-neighbor (step) interpolation: each query `q` maps to the value
/// `fp[i]` where `i` is the largest index with `xp[i] <= q`. Queries outside
/// the support `[xp[0], xp[-1]]` get `fill`. Matches
/// `scipy.interpolate.interp1d(kind='previous', bounds_error=False,
/// fill_value=fill)` and MATLAB `interp1(xp, fp, q, 'previous', fill)`
/// (no `'extrap'`).
pub fn interp_previous(xp: &[f64], fp: &[f64], queries: &[f64], fill: f64) -> Vec<f64> {
    debug_assert_eq!(xp.len(), fp.len());
    let mut out = Vec::with_capacity(queries.len());
    if xp.is_empty() {
        return vec![fill; queries.len()];
    }
    let last = *xp.last().unwrap();
    for &q in queries {
        if q < xp[0] || q > last {
            out.push(fill);
            continue;
        }
        // Largest i with xp[i] <= q. Binary search via partition_point.
        let i = xp.partition_point(|&t| t <= q);
        // partition_point returns the first index with t > q, so i-1 is what we want.
        out.push(fp[i - 1]);
    }
    out
}

/// Nearest-neighbor interpolation of a boolean flag series onto `queries`.
/// For each query, finds the sample in `xp` closest in time and returns
/// `flags[idx]`. An exact midpoint goes to the higher index, as in MATLAB
/// `interp1(xp, flags, q, 'nearest')`.
pub fn interp_nearest_bool(xp: &[f64], flags: &[bool], queries: &[f64]) -> Vec<bool> {
    debug_assert_eq!(xp.len(), flags.len());
    let mut out = Vec::with_capacity(queries.len());
    for &q in queries {
        if xp.is_empty() {
            out.push(false);
            continue;
        }
        let j = xp.partition_point(|&t| t <= q);
        let idx = if j == 0 {
            0
        } else if j == xp.len() {
            xp.len() - 1
        } else {
            // Compare distance to xp[j-1] vs xp[j]; tie → j (higher index).
            let left = q - xp[j - 1];
            let right = xp[j] - q;
            if right <= left { j } else { j - 1 }
        };
        out.push(flags[idx]);
    }
    out
}

/// Linear interpolation with flat extrapolation at both ends (one-sample pad
/// to one second outside the grid, matching pydynamo `pipeline.py:231`).
pub fn interp_linear_padded(xp: &[f64], fp: &[f64], queries: &[f64]) -> Vec<f64> {
    debug_assert_eq!(xp.len(), fp.len());
    if xp.is_empty() {
        return vec![0.0; queries.len()];
    }
    let n = xp.len();
    let mut out = Vec::with_capacity(queries.len());
    for &q in queries {
        if q <= xp[0] {
            out.push(fp[0]);
        } else if q >= xp[n - 1] {
            out.push(fp[n - 1]);
        } else {
            let j = xp.partition_point(|&t| t <= q);
            // xp[j-1] <= q < xp[j]
            let t0 = xp[j - 1];
            let t1 = xp[j];
            let y0 = fp[j - 1];
            let y1 = fp[j];
            let a = (q - t0) / (t1 - t0);
            out.push(y0 + a * (y1 - y0));
        }
    }
    out
}

/// Linear interpolation exactly as `computePeakSOpower.m` and
/// `computePeakSOphase.m` call it:
///
/// ```text
/// interp1([t(1)-step, t, t(end)+step], [v(1), v, v(end)], q)
/// ```
///
/// where `step = t(2) - t(1)`. The behavior at each kind of query point:
///
/// * exactly on a grid point: that sample's value, even when a neighboring
///   sample is NaN;
/// * strictly between two samples: the linear blend, or NaN when either
///   sample is NaN;
/// * outside the grid but within one `step` of an end (the padded endpoint
///   included): the end sample's value;
/// * farther out, or a non-finite query: NaN.
///
/// `xp` must be sorted ascending. A series with fewer than two samples has
/// no step, so only an exact hit on its single sample returns a value.
pub fn interp_linear_matlab_padded(xp: &[f64], fp: &[f64], queries: &[f64]) -> Vec<f64> {
    debug_assert_eq!(xp.len(), fp.len());
    let n = xp.len();
    if n == 0 {
        return vec![f64::NAN; queries.len()];
    }
    let first = xp[0];
    let last = xp[n - 1];
    let step = if n >= 2 { xp[1] - xp[0] } else { 0.0 };

    let mut out = Vec::with_capacity(queries.len());
    for &q in queries {
        if !q.is_finite() {
            out.push(f64::NAN);
            continue;
        }
        // Left of the grid: the one-step pad holds the first sample's value.
        if q < first {
            let inside_pad = q >= first - step;
            out.push(if inside_pad { fp[0] } else { f64::NAN });
            continue;
        }
        // Right of the grid: the one-step pad holds the last sample's value.
        if q > last {
            let inside_pad = q <= last + step;
            out.push(if inside_pad { fp[n - 1] } else { f64::NAN });
            continue;
        }
        // Inside the grid. Binary search for the bracketing pair
        // xp[lo] <= q <= xp[hi] with hi == lo + 1 (or lo == hi when n == 1).
        let mut lo = 0usize;
        let mut hi = n - 1;
        while lo + 1 < hi {
            let mid = (lo + hi) / 2;
            if xp[mid] <= q {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        if xp[lo] == q {
            out.push(fp[lo]);
        } else if xp[hi] == q {
            out.push(fp[hi]);
        } else {
            let v0 = fp[lo];
            let v1 = fp[hi];
            if !v0.is_finite() || !v1.is_finite() {
                out.push(f64::NAN);
            } else {
                let t = (q - xp[lo]) / (xp[hi] - xp[lo]);
                out.push(v0 + t * (v1 - v0));
            }
        }
    }
    out
}

/// Wrap an angle to the closed interval `[-π, π]`, matching MATLAB
/// `wrapToPi`: values already in range pass through unchanged (so `+π`
/// stays `+π` and `−π` stays `−π`), and an out-of-range fold that lands on
/// the boundary keeps the input's sign (`+3π → +π`, `−3π → −π`). NaN stays
/// NaN.
pub fn wrap_to_pi(x: f64) -> f64 {
    use std::f64::consts::{PI, TAU};
    if (-PI..=PI).contains(&x) {
        return x;
    }
    let folded = (x + PI).rem_euclid(TAU) - PI;
    if folded == -PI && x > 0.0 {
        PI
    } else {
        folded
    }
}

/// Assign sleep stage to each peak by previous-neighbor interp of staging,
/// then override with stage=6 where the nearest EEG sample is flagged as an
/// artifact. `artifact_times` is the EEG time grid on which `artifacts`
/// lives; typically `t_tr` in the pydynamo pipeline.
pub fn assign_peak_stage(
    peak_times: &[f64],
    stage_times: &[f64],
    stage_vals: &[f64],
    artifact_times: &[f64],
    artifacts: &[bool],
) -> Vec<f64> {
    let mut stages = interp_previous(stage_times, stage_vals, peak_times, 0.0);
    let art_at_peak = interp_nearest_bool(artifact_times, artifacts, peak_times);
    for (s, &a) in stages.iter_mut().zip(art_at_peak.iter()) {
        if a {
            *s = 6.0;
        }
    }
    stages
}

/// Assign SO-power to each peak: MATLAB's one-step-padded linear interp of
/// the normalized SO-power time series (`computePeakSOpower.m`). A peak at
/// an excluded (NaN) stretch of the series gets NaN.
pub fn assign_peak_sopower(
    peak_times: &[f64],
    sopower_times: &[f64],
    sopower_norm: &[f64],
) -> Vec<f64> {
    interp_linear_matlab_padded(sopower_times, sopower_norm, peak_times)
}

/// Assign SO-phase to each peak: MATLAB's one-step-padded linear interp of
/// the *unwrapped* phase, then `wrapToPi` to `[-π, π]`
/// (`computePeakSOphase.m`). Interpolating before wrapping keeps a peak
/// whose two neighboring samples straddle the ±π seam near ±π.
pub fn assign_peak_sophase(
    peak_times: &[f64],
    sophase_times: &[f64],
    sophase_unwrapped: &[f64],
) -> Vec<f64> {
    let unwrapped_at_peaks =
        interp_linear_matlab_padded(sophase_times, sophase_unwrapped, peak_times);
    let mut phases = Vec::with_capacity(unwrapped_at_peaks.len());
    for phase in unwrapped_at_peaks {
        phases.push(wrap_to_pi(phase));
    }
    phases
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn previous_step_basic() {
        let xp = [0.0, 10.0, 20.0, 30.0];
        let fp = [1.0, 2.0, 3.0, 4.0]; // N1 N2 N3 REM ~
        // Fill for q < xp[0] AND q > xp[-1] (scipy/MATLAB no-extrap semantics).
        let q = [-5.0, 0.0, 5.0, 10.0, 29.9, 30.0, 100.0];
        let out = interp_previous(&xp, &fp, &q, 0.0);
        assert_eq!(out, vec![0.0, 1.0, 1.0, 2.0, 3.0, 4.0, 0.0]);
    }

    #[test]
    fn linear_padded_flat_ends() {
        let xp = [0.0, 1.0, 2.0];
        let fp = [10.0, 20.0, 30.0];
        let q = [-1.0, 0.0, 0.5, 1.5, 2.0, 100.0];
        let out = interp_linear_padded(&xp, &fp, &q);
        assert_eq!(out, vec![10.0, 10.0, 15.0, 25.0, 30.0, 30.0]);
    }

    // The expected values in the next five tests are the outputs of
    // MATLAB R2025b for the same inputs (commands quoted in each test).

    #[test]
    fn nearest_bool_exact_midpoint_goes_to_higher_index() {
        // interp1([1 2 3],[10 20 30],[1.5 2.5],'nearest') → 20 30
        let xp = [1.0, 2.0, 3.0];
        let flags = [false, true, false];
        let out = interp_nearest_bool(&xp, &flags, &[1.5, 2.5]);
        assert_eq!(out, vec![true, false]);
    }

    #[test]
    fn nearest_bool_follows_double_precision_distance() {
        // t=(0:4)*0.01; interp1(t,single([0 1 0 1 0]),[0.005 0.015 0.025 0.035],'nearest')
        //   → 1 1 1 0
        // 0.015 is not an exact midpoint in doubles (0.015-0.01 < 0.02-0.015),
        // so it resolves to the lower sample.
        let xp: Vec<f64> = (0..5).map(|i| i as f64 * 0.01).collect();
        let flags = [false, true, false, true, false];
        let out = interp_nearest_bool(&xp, &flags, &[0.005, 0.015, 0.025, 0.035]);
        assert_eq!(out, vec![true, true, true, false]);
    }

    #[test]
    fn matlab_padded_one_step_pad_then_nan() {
        // x=0:3; v=[5 6 7 8];
        // interp1([x(1)-1, x, x(end)+1],[v(1), v, v(end)],[-1.5 -1 -0.5 0 3 3.5 4 4.5])
        //   → NaN 5 5 5 8 8 8 NaN
        let xp = [0.0, 1.0, 2.0, 3.0];
        let fp = [5.0, 6.0, 7.0, 8.0];
        let q = [-1.5, -1.0, -0.5, 0.0, 3.0, 3.5, 4.0, 4.5];
        let out = interp_linear_matlab_padded(&xp, &fp, &q);
        assert!(out[0].is_nan());
        assert_eq!(&out[1..7], &[5.0, 5.0, 5.0, 8.0, 8.0, 8.0]);
        assert!(out[7].is_nan());
    }

    #[test]
    fn matlab_padded_nan_neighbor_rules() {
        // interp1([1 2 3],[10 NaN 30],[1 1.5 2 2.5 3]) → 10 NaN NaN NaN 30
        let out = interp_linear_matlab_padded(
            &[1.0, 2.0, 3.0],
            &[10.0, f64::NAN, 30.0],
            &[1.0, 1.5, 2.0, 2.5, 3.0],
        );
        assert_eq!(out[0], 10.0);
        assert!(out[1].is_nan() && out[2].is_nan() && out[3].is_nan());
        assert_eq!(out[4], 30.0);

        // interp1([1 2 3],[10 20 NaN],[2 2.0000001]) → 20 NaN
        let out = interp_linear_matlab_padded(
            &[1.0, 2.0, 3.0],
            &[10.0, 20.0, f64::NAN],
            &[2.0, 2.0000001],
        );
        assert_eq!(out[0], 20.0);
        assert!(out[1].is_nan());
    }

    #[test]
    fn matlab_padded_interior_and_non_finite_query() {
        let out = interp_linear_matlab_padded(
            &[0.0, 1.0, 2.0],
            &[10.0, 20.0, 30.0],
            &[0.5, 1.5, f64::NAN],
        );
        assert_eq!(out[0], 15.0);
        assert_eq!(out[1], 25.0);
        assert!(out[2].is_nan());
    }

    #[test]
    fn wrap_to_pi_matches_matlab() {
        // wrapToPi([pi -pi 3*pi -3*pi]) → pi -pi pi -pi
        let pi = std::f64::consts::PI;
        assert_eq!(wrap_to_pi(0.0), 0.0);
        assert_eq!(wrap_to_pi(pi), pi);
        assert_eq!(wrap_to_pi(-pi), -pi);
        assert!((wrap_to_pi(3.0 * pi) - pi).abs() < 1e-9);
        assert!((wrap_to_pi(-3.0 * pi) + pi).abs() < 1e-9);
        assert!((wrap_to_pi(1.5 * pi) + 0.5 * pi).abs() < 1e-9);
        // A value just inside the range maps to itself.
        let just_below = pi - 1e-9;
        assert_eq!(wrap_to_pi(just_below), just_below);
        assert!(wrap_to_pi(f64::NAN).is_nan());
    }

    #[test]
    fn peak_stage_applies_artifact_override() {
        let peak_times = [5.0, 15.0];
        let stage_times = [0.0, 10.0];
        let stage_vals = [2.0, 3.0]; // N2, N1
        let eeg_times = [5.0, 15.0];
        let artifacts = [false, true];
        let out = assign_peak_stage(&peak_times, &stage_times, &stage_vals, &eeg_times, &artifacts);
        assert_eq!(out, vec![2.0, 6.0]); // first N2; second overridden to artifact=6
    }

    #[test]
    fn peak_sophase_wraps_to_pi() {
        // The unwrapped phase interpolates to 3π at t=1, and MATLAB
        // `wrapToPi(3*pi)` is +π.
        let peak_times = [1.0];
        let sophase_times = [0.0, 2.0];
        let sophase_unwrapped = [0.0, 6.0 * std::f64::consts::PI];
        let out = assign_peak_sophase(&peak_times, &sophase_times, &sophase_unwrapped);
        assert!((out[0] - std::f64::consts::PI).abs() < 1e-12);
    }

    #[test]
    fn peak_sophase_straddling_the_seam_stays_near_pi() {
        // Two samples either side of the ±π seam: unwrapped 3.0 and 3.3 rad.
        // A peak midway is at 3.15 rad unwrapped, which wraps to just above −π.
        let out = assign_peak_sophase(&[0.5], &[0.0, 1.0], &[3.0, 3.3]);
        assert!(out[0].abs() > std::f64::consts::FRAC_PI_2);
    }
}
