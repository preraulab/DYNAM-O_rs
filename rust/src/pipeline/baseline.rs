//! Thin wrapper around `crate::baseline` — kernel already in Rust.

use ndarray::{Array2, ArrayView1, ArrayView2};

pub fn compute_baseline(
    spect: ArrayView2<f64>,
    stimes: ArrayView1<f64>,
    t_data: ArrayView1<f64>,
    baseline_exclude: &[bool],
    baseline_range: (f64, f64),
    baseline_ptile: f64,
) -> Result<Array2<f64>, String> {
    crate::baseline::compute_baseline(
        spect,
        stimes,
        t_data,
        baseline_exclude,
        baseline_range,
        baseline_ptile,
    )
}

pub fn subtract_baseline(spect: ArrayView2<f64>, baseline: ArrayView2<f64>) -> Result<Array2<f64>, String> {
    crate::baseline::subtract_baseline(spect, baseline)
}
