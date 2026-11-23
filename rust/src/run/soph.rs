//! SO-power / SO-phase histogram construction.
//!
//! Per MATLAB convention:
//!   * freq bins: `'partial'` mode from `create_bins.m` (overlapping; first
//!     and last centers fixed at `range`, edges clamped to `range`).
//!   * SO-power bins: `'partial'` mode over the data-derived range.
//!   * SO-phase bins: `'extend'` mode over `[-π, π]` (circular-friendly).
//!
//! See `DYNAM-O/toolbox/helper_functions/binning/create_bins.m` and
//! `SOpowphase_functions/TFPeakHistogram.m`.

use ndarray::{Array1, Array2};

/// MATLAB colon operator `start : step : stop`.
///
/// Element count is derived once (`floor((stop-start)/step + tol) + 1`) and
/// each center is computed by multiplication (`start + i*step`) rather than
/// iterative accumulation, so the values carry no accumulated rounding drift
/// and the count matches MATLAB's colon exactly at near-integer boundaries.
fn colon(start: f64, step: f64, stop: f64) -> Vec<f64> {
    if step <= 0.0 || !(start.is_finite() && step.is_finite() && stop.is_finite()) {
        return vec![start];
    }
    // Relative tolerance mirrors MATLAB's colon boundary handling.
    let tol = 1e-9 * ((start.abs()).max(stop.abs())).max(1.0);
    let n = (((stop - start) + tol) / step).floor().max(0.0) as usize + 1;
    (0..n).map(|i| start + i as f64 * step).collect()
}

/// Per `create_bins.m` 'partial' mode:
///   `centers = range[0] : step : range[1]`
///   `edges   = clamp(centers ± width/2, range)`
/// Returns `(edges (2 × N), centers (N))`.
pub fn create_partial_bins(range: (f64, f64), width: f64, step: f64) -> (Array2<f64>, Vec<f64>) {
    let centers = colon(range.0, step, range.1);
    let n = centers.len();
    let mut edges = Array2::<f64>::zeros((2, n));
    for (i, &center) in centers.iter().enumerate() {
        let lo = (center - width * 0.5).max(range.0).min(range.1);
        let hi = (center + width * 0.5).max(range.0).min(range.1);
        edges[[0, i]] = lo;
        edges[[1, i]] = hi;
    }
    (edges, centers)
}

/// Per `create_bins.m` 'extend' mode:
///   `new_lo = range[0] - floor((width/2)/step)*step`
///   `new_hi = range[1] + floor((width/2)/step)*step`
///   `centers = (new_lo + width/2) : step : (new_hi - width/2)`
///   `edges   = centers ± width/2`
pub fn create_extend_bins(range: (f64, f64), width: f64, step: f64) -> (Array2<f64>, Vec<f64>) {
    let half = width * 0.5;
    let pad = (half / step).floor() * step;
    let new_lo = range.0 - pad;
    let new_hi = range.1 + pad;

    let centers = colon(new_lo + half, step, new_hi - half);
    let n = centers.len();
    let mut edges = Array2::<f64>::zeros((2, n));
    for (i, &center) in centers.iter().enumerate() {
        edges[[0, i]] = center - half;
        edges[[1, i]] = center + half;
    }
    (edges, centers)
}

/// The workspace's single phase wrap: MATLAB `wrapToPi`, closed `[-π, π]`.
/// It lives in the kernel next to the per-peak SO-phase sampling that uses
/// it, and is re-exported here for `fits.rs` and `mode_peaks.rs`.
pub use crate::peak_assign::wrap_to_pi;

/// SO-power range derived from the data: the min and max of the finite
/// values, which is MATLAB's `SOpower_range` default (the whole data range
/// when none is specified). Returns `(0.0, 1.0)` when there is no finite
/// value or the range is empty.
pub fn sopower_range_from_data(so_power: &[f64]) -> (f64, f64) {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for &v in so_power {
        if v.is_finite() {
            if v < lo {
                lo = v;
            }
            if v > hi {
                hi = v;
            }
        }
    }
    if !lo.is_finite() || !hi.is_finite() || lo >= hi {
        return (0.0, 1.0);
    }
    (lo, hi)
}

/// Stage interpolation onto a target time grid (previous-neighbor),
/// returning the per-sample stage code. Outside the staging range → 0.
pub fn stages_at_times(stage_times: &[f64], stage_vals: &[f64], queries: &[f64]) -> Array1<f64> {
    let interp = crate::peak_assign::interp_previous(stage_times, stage_vals, queries, 0.0);
    Array1::from_iter(interp.into_iter().map(|v| if v.is_nan() { 0.0 } else { v }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partial_bins_basic() {
        let (edges, centers) = create_partial_bins((0.0, 30.0), 1.0, 0.2);
        // 0..30 step 0.2 → 151 centers (inclusive)
        assert_eq!(centers.len(), 151);
        assert!((centers[0] - 0.0).abs() < 1e-12);
        assert!((centers[centers.len() - 1] - 30.0).abs() < 1e-9);
        // First bin's left edge clamped to range start.
        assert!((edges[[0, 0]] - 0.0).abs() < 1e-12);
        // Last bin's right edge clamped to range end.
        assert!((edges[[1, centers.len() - 1]] - 30.0).abs() < 1e-9);
    }
}
