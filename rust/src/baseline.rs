//! Per-frequency Nth-percentile baseline over a spectrogram.
//!
//! Port of `computeBaseline` / pydynamo `baseline.compute_baseline`.
//!
//! For each frequency row, compute `prctile` (Hyndman-Fan method #5, aka
//! "hazen") of the valid time-columns. A column is "valid" when:
//!   - `baseline_exclude[i] == false` at the nearest data-sample to that
//!     column's stimes, AND
//!   - `baseline_range[0] ≤ stimes[t] ≤ baseline_range[1]`
//! Zero pixels inside the valid columns are treated as NaN (so the
//! percentile isn't biased down by masked/boundary zeros).
//!
//! MATLAB ties to:
//!   `baseline = prctile(spect(:, valid), ptile, 2)`  % method = H-F 5
//!
//! Returns a column vector `(F, 1)`.

use ndarray::{s, Array2, ArrayView1, ArrayView2};

/// Hyndman-Fan method #5 ("hazen") percentile over a NaN-aware slice.
///
/// MATLAB `prctile` default. numpy `nanpercentile(method="hazen")`.
///
/// Formula: given N finite values sorted ascending, index `h = q*N + 0.5`
/// (1-based continuous index), clamped to `[1, N]`. Linear-interpolate
/// between `floor(h)` and `ceil(h)`.
///
/// NaNs are skipped. Returns NaN if no finite values.
pub fn nanpercentile_hazen(values: &[f64], q_pct: f64) -> f64 {
    // Collect finite values into a scratch buffer.
    let mut scratch: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if scratch.is_empty() {
        return f64::NAN;
    }
    scratch.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    hazen_from_sorted(&scratch, q_pct)
}

/// Hazen percentile given an already-sorted (ascending, finite) slice.
fn hazen_from_sorted(sorted: &[f64], q_pct: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    if n == 1 {
        return sorted[0];
    }
    // 1-based hazen index
    let h = q_pct / 100.0 * n as f64 + 0.5;
    let h = h.clamp(1.0, n as f64);
    let lo = h.floor() as usize; // 1-based
    let hi = h.ceil() as usize; // 1-based
    let frac = h - h.floor();
    // Convert to 0-based
    let a = sorted[lo - 1];
    let b = sorted[hi - 1];
    a + (b - a) * frac
}

/// Map each `stimes[t]` to the nearest index into `t_data`, assuming
/// `t_data` is sorted ascending. Returns a Vec of indices the same length
/// as `stimes`.
pub fn nearest_indices(stimes: ArrayView1<f64>, t_data: ArrayView1<f64>) -> Vec<usize> {
    let n_td = t_data.len();
    let mut out = Vec::with_capacity(stimes.len());
    for &s in stimes.iter() {
        // Binary search: find the first index with t_data[i] >= s.
        let mut lo = 0usize;
        let mut hi = n_td;
        while lo < hi {
            let mid = (lo + hi) / 2;
            if t_data[mid] < s {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        // Candidates: lo (first ≥ s) and lo-1 (last < s).
        let right = lo.min(n_td - 1);
        let left = if lo == 0 { 0 } else { lo - 1 };
        let d_right = (t_data[right] - s).abs();
        let d_left = (t_data[left] - s).abs();
        out.push(if d_left < d_right { left } else { right });
    }
    out
}

/// Compute per-frequency baseline.
pub fn compute_baseline(
    spect: ArrayView2<f64>,
    stimes: ArrayView1<f64>,
    t_data: ArrayView1<f64>,
    baseline_exclude: &[bool],
    baseline_range: (f64, f64),
    baseline_ptile: f64,
) -> Result<Array2<f64>, String> {
    let (n_freq, n_time) = (spect.nrows(), spect.ncols());
    if stimes.len() != n_time {
        return Err(format!(
            "stimes len {} != spect.ncols {}",
            stimes.len(),
            n_time
        ));
    }
    if baseline_exclude.len() != t_data.len() {
        return Err(format!(
            "baseline_exclude len {} != t_data len {}",
            baseline_exclude.len(),
            t_data.len()
        ));
    }

    // Nearest t_data index per stimes column.
    let idx_nearest = nearest_indices(stimes, t_data);

    // Compute the set of valid time columns.
    let (r_lo, r_hi) = baseline_range;
    let mut valid_cols: Vec<usize> = Vec::with_capacity(n_time);
    for t in 0..n_time {
        let excluded = baseline_exclude[idx_nearest[t]];
        let s = stimes[t];
        let in_range = s >= r_lo && s <= r_hi;
        if !excluded && in_range {
            valid_cols.push(t);
        }
    }

    if valid_cols.is_empty() {
        return Err(
            "No valid baseline time bins remain after applying artifacts, \
             stage filtering, and baseline_range."
                .to_string(),
        );
    }

    // Per-frequency hazen percentile over valid columns, treating 0 as NaN.
    let mut baseline = Array2::<f64>::zeros((n_freq, 1));
    // Scratch buffer reused across frequency rows
    let mut scratch: Vec<f64> = Vec::with_capacity(valid_cols.len());
    for f in 0..n_freq {
        scratch.clear();
        for &t in &valid_cols {
            let v = spect[[f, t]];
            if v != 0.0 && v.is_finite() {
                scratch.push(v);
            }
        }
        if scratch.is_empty() {
            baseline[[f, 0]] = f64::NAN;
        } else {
            scratch.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            baseline[[f, 0]] = hazen_from_sorted(&scratch, baseline_ptile);
        }
    }
    Ok(baseline)
}

/// Divide spectrogram by baseline (column broadcast).
pub fn subtract_baseline(
    spect: ArrayView2<f64>,
    baseline: ArrayView2<f64>,
) -> Result<Array2<f64>, String> {
    if baseline.nrows() != spect.nrows() || baseline.ncols() != 1 {
        return Err(format!(
            "baseline must be (F, 1) with F = spect.nrows; got {:?} vs spect {:?}",
            baseline.dim(),
            spect.dim()
        ));
    }
    let mut out = spect.to_owned();
    for f in 0..spect.nrows() {
        let b = baseline[[f, 0]];
        let mut row = out.slice_mut(s![f, ..]);
        row.mapv_inplace(|v| v / b);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn hazen_matches_matlab_simple() {
        // MATLAB: prctile([1,2,3,4,5], 50) = 3
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((nanpercentile_hazen(&v, 50.0) - 3.0).abs() < 1e-12);
        // prctile([1,2,3,4,5], 25) = 1.75 (hazen: h = 0.25*5 + 0.5 = 1.75)
        assert!((nanpercentile_hazen(&v, 25.0) - 1.75).abs() < 1e-12);
        // prctile([1,2,3,4,5], 2) = 1  (h = 0.02*5 + 0.5 = 0.6 → clamp to 1)
        assert!((nanpercentile_hazen(&v, 2.0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn hazen_skips_nan() {
        let v = vec![1.0, f64::NAN, 2.0, 3.0, f64::NAN, 4.0, 5.0];
        assert!((nanpercentile_hazen(&v, 50.0) - 3.0).abs() < 1e-12);
    }

    #[test]
    fn nearest_index_basic() {
        let stimes = array![0.5, 1.5, 2.4];
        let t_data = array![0.0, 1.0, 2.0, 3.0];
        let idx = nearest_indices(stimes.view(), t_data.view());
        // Ties broken toward the RIGHT neighbor (matches pydynamo
        // `use_left = np.abs(left) < np.abs(right)` — strict <, so equal
        // distances fall through to the right-neighbor).
        // 0.5 → d_left=d_right=0.5 → right (1)
        // 1.5 → d_left=d_right=0.5 → right (2)
        // 2.4 → d_left=0.4, d_right=0.6 → left (2)
        assert_eq!(idx, vec![1, 2, 2]);
    }

    #[test]
    fn compute_baseline_simple() {
        // 2×4 spect, 1 artifact col, baseline = 2nd pct per row over kept cols
        let spect = array![[10.0, 20.0, 30.0, 40.0], [1.0, 2.0, 3.0, 4.0]];
        let stimes = array![0.0, 1.0, 2.0, 3.0];
        let t_data = array![0.0, 1.0, 2.0, 3.0];
        let excl = vec![false, false, false, true]; // drop col 3
        let bl = compute_baseline(
            spect.view(),
            stimes.view(),
            t_data.view(),
            &excl,
            (f64::NEG_INFINITY, f64::INFINITY),
            50.0,
        )
        .unwrap();
        // Row 0: finite = [10, 20, 30], median (50th hazen) = 20
        // Row 1: finite = [1, 2, 3], median = 2
        assert_eq!(bl.dim(), (2, 1));
        assert!((bl[[0, 0]] - 20.0).abs() < 1e-12);
        assert!((bl[[1, 0]] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn zero_treated_as_nan() {
        let spect = array![[10.0, 0.0, 30.0, 20.0]];
        let stimes = array![0.0, 1.0, 2.0, 3.0];
        let t_data = array![0.0, 1.0, 2.0, 3.0];
        let excl = vec![false; 4];
        let bl = compute_baseline(
            spect.view(),
            stimes.view(),
            t_data.view(),
            &excl,
            (f64::NEG_INFINITY, f64::INFINITY),
            50.0,
        )
        .unwrap();
        // Finite non-zero: [10, 20, 30], median = 20
        assert!((bl[[0, 0]] - 20.0).abs() < 1e-12);
    }
}
