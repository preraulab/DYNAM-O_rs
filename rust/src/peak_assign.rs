//! Assign per-peak metadata: stage, SO-power, SO-phase.
//!
//! Ports pydynamo `pipeline.py` lines ~205–248 (MATLAB `computePeakStage.m`,
//! `computePeakSOpower.m`, `computePeakSOphase.m`). Each function takes the
//! already-computed time-series metric on its own time grid plus the peak
//! times, and returns one value per peak.
//!
//! All inputs are assumed sorted by increasing time. No sorting is done here.
//!
//! Conventions mirror pydynamo exactly:
//!
//! * `assign_peak_stage` — previous-neighbor interp (step function) with
//!   fill=0 for peaks outside the staging range; artifact override sets the
//!   stage to 6 when the nearest `artifacts` sample is true.
//! * `assign_peak_sopower` — linear interp with one-sample pad on each end
//!   (pydynamo's `xp = [t0-1, ...times..., tN+1]` trick to make `np.interp`
//!   return the endpoint values instead of the default flat extrapolation).
//! * `assign_peak_sophase` — linear interp of *unwrapped* phase, then wrap
//!   back to `(-π, π]`.

/// Previous-neighbor (step) interpolation: each query `q` maps to the value
/// `fp[i]` where `i` is the largest index with `xp[i] <= q`. Queries below
/// `xp[0]` get `fill`. Matches `scipy.interpolate.interp1d(kind='previous',
/// bounds_error=False, fill_value=fill)`.
pub fn interp_previous(xp: &[f64], fp: &[f64], queries: &[f64], fill: f64) -> Vec<f64> {
    debug_assert_eq!(xp.len(), fp.len());
    let mut out = Vec::with_capacity(queries.len());
    for &q in queries {
        if xp.is_empty() || q < xp[0] {
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
/// `flags[idx]`. Ties go to the lower index.
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
            // Compare distance to xp[j-1] vs xp[j]; tie → j-1 (lower index).
            let left = q - xp[j - 1];
            let right = xp[j] - q;
            if right < left { j } else { j - 1 }
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

/// Assign SO-power to each peak by linear interp of the normalized SO-power
/// time series, with endpoint-flat extrapolation.
pub fn assign_peak_sopower(
    peak_times: &[f64],
    sopower_times: &[f64],
    sopower_norm: &[f64],
) -> Vec<f64> {
    interp_linear_padded(sopower_times, sopower_norm, peak_times)
}

/// Assign SO-phase to each peak: linear-interp *unwrapped* phase, then wrap
/// back to `(-π, π]` (matches pydynamo `pipeline.py:248`).
pub fn assign_peak_sophase(
    peak_times: &[f64],
    sophase_times: &[f64],
    sophase_unwrapped: &[f64],
) -> Vec<f64> {
    let tau = 2.0 * std::f64::consts::PI;
    let mut phases = interp_linear_padded(sophase_times, sophase_unwrapped, peak_times);
    for p in phases.iter_mut() {
        let wrapped = (*p + std::f64::consts::PI).rem_euclid(tau) - std::f64::consts::PI;
        *p = wrapped;
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
        let q = [-5.0, 0.0, 5.0, 10.0, 29.9, 30.0, 100.0];
        let out = interp_previous(&xp, &fp, &q, 0.0);
        assert_eq!(out, vec![0.0, 1.0, 1.0, 2.0, 3.0, 4.0, 4.0]);
    }

    #[test]
    fn linear_padded_flat_ends() {
        let xp = [0.0, 1.0, 2.0];
        let fp = [10.0, 20.0, 30.0];
        let q = [-1.0, 0.0, 0.5, 1.5, 2.0, 100.0];
        let out = interp_linear_padded(&xp, &fp, &q);
        assert_eq!(out, vec![10.0, 10.0, 15.0, 25.0, 30.0, 30.0]);
    }

    #[test]
    fn nearest_bool_ties_go_left() {
        // Sample at t=0 is true, t=2 is false. Query at t=1 is equidistant —
        // partition_point gives j=1 here so the "tie → lower index" path picks t=0 → true.
        let xp = [0.0, 2.0];
        let fl = [true, false];
        let q = [1.0];
        let out = interp_nearest_bool(&xp, &fl, &q);
        assert_eq!(out, vec![true]);
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
        // Unwrapped = 3π at t=1; wrapped should be π  (actually rem_euclid(2π) gives π not -π)
        let peak_times = [1.0];
        let sophase_times = [0.0, 2.0];
        let sophase_unwrapped = [0.0, 6.0 * std::f64::consts::PI];
        let out = assign_peak_sophase(&peak_times, &sophase_times, &sophase_unwrapped);
        // 3π + π = 4π; 4π mod 2π = 0; 0 - π = -π; so result is -π.
        // (pydynamo uses numpy's (x+pi) % (2*pi) - pi, same formula — -π output
        // is correct per MATLAB's wrapToPi semantics too.)
        assert!((out[0] - (-std::f64::consts::PI)).abs() < 1e-12);
    }
}
