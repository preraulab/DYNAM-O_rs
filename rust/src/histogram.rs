//! Core 2D TF-peak × C-metric histogram. Port of pydynamo
//! `soph.histogram.tfpeak_histogram` (itself a port of MATLAB
//! `TFPeakHistogram.m`).
//!
//! Stage convention: DYNAM-O uses 1..=5 (1=N3, 2=N2, 3=N1, 4=REM, 5=Wake).
//! `time_in_bin` returns 5 columns for these 5 stages.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use crate::parallel::*;

/// Number of scored sleep stages (1=N3, 2=N2, 3=N1, 4=REM, 5=Wake).
const NUM_STAGES: usize = 5;

/// Inputs to [`tfpeak_histogram`].
pub struct HistogramInputs<'a> {
    /// C-metric (SO-power or SO-phase) time series.
    pub c_metric: ArrayView1<'a, f64>,
    /// Stage per C-metric sample (1..=5).
    pub c_stages: ArrayView1<'a, f64>,
    /// C-metric sample spacing in seconds.
    pub c_dt: f64,
    /// Samples counted toward per-stage time in bin.
    pub c_valid: &'a [bool],
    /// Samples counted toward the all-stages denominator of `prop_in_bin`.
    pub c_valid_allstages: &'a [bool],
    /// Frequency of each peak.
    pub peak_freqs: ArrayView1<'a, f64>,
    /// C-metric value of each peak.
    pub peak_c: ArrayView1<'a, f64>,
    /// (2, num_fbins): row 0 lower edges, row 1 upper edges. A bin is
    /// [lower, upper).
    pub freq_edges: ArrayView2<'a, f64>,
    /// (2, num_cbins): row 0 lower edges, row 1 upper edges. A bin is
    /// [lower, upper).
    pub c_edges: ArrayView2<'a, f64>,
    /// The C-metric is circular (phase): bins that reach a bound wrap around.
    pub circular: bool,
    /// (low, high) of the circular range.
    pub circular_bounds: (f64, f64),
    /// 0 = none, 1 = normalize each frequency column over C-bins,
    /// 2 = normalize each C-bin row over frequencies.
    pub norm_dim: i32,
    /// Divide counts by minutes in the bin (peaks per minute).
    pub compute_rate: bool,
    /// C-bin rows with less total time (minutes) are NaN.
    pub min_time_in_bin: f64,
    /// Frequency columns with fewer peaks are NaN.
    pub min_peak_at_freq: i32,
}

/// Outputs of [`tfpeak_histogram`].
pub struct HistogramOutputs {
    /// (num_cbins, num_fbins) histogram.
    pub c_mat: Array2<f64>,
    /// (num_cbins, 5) minutes per stage.
    pub time_in_bin: Array2<f64>,
    /// (num_cbins, 5) fraction of all-stages time per stage.
    pub prop_in_bin: Array2<f64>,
    /// (num_fbins,) peaks per frequency bin.
    pub peak_at_freq: Array1<f64>,
}

/// One C-bin's results, assembled into the output arrays afterwards.
struct CBinRow {
    /// One value per frequency bin.
    counts: Vec<f64>,
    /// Minutes per stage.
    time_in_bin: [f64; NUM_STAGES],
    /// Fraction of all-stages time per stage.
    prop_in_bin: [f64; NUM_STAGES],
}

/// Build the TF-peak histogram over C-metric bins (rows) and frequency bins
/// (columns).
pub fn tfpeak_histogram(inp: &HistogramInputs) -> Result<HistogramOutputs, String> {
    let num_fbins = inp.freq_edges.ncols();
    let num_cbins = inp.c_edges.ncols();
    let n_times = inp.c_metric.len();
    let n_peaks = inp.peak_freqs.len();

    if inp.c_stages.len() != n_times
        || inp.c_valid.len() != n_times
        || inp.c_valid_allstages.len() != n_times
    {
        return Err(format!(
            "c_metric/c_stages/c_valid/c_valid_allstages length mismatch: {} / {} / {} / {}",
            n_times,
            inp.c_stages.len(),
            inp.c_valid.len(),
            inp.c_valid_allstages.len()
        ));
    }
    if inp.peak_c.len() != n_peaks {
        return Err(format!(
            "peak_freqs ({}) and peak_c ({}) length mismatch",
            n_peaks,
            inp.peak_c.len()
        ));
    }
    if inp.freq_edges.nrows() != 2 || inp.c_edges.nrows() != 2 {
        return Err("freq_edges and c_edges must be (2, num_bins)".into());
    }

    // Precompute peak → freq-bin membership once.
    // Freq bins may overlap ("partial" mode, see create_bins), so a peak can
    // fall in several bins. `peak_in_fbin[p * num_fbins + f]` is true when
    // peak p lies in freq bin f: a flat row-major (n_peaks, num_fbins) table.
    let peak_in_fbin = {
        let mut membership = vec![false; n_peaks * num_fbins];
        for p in 0..n_peaks {
            let pf = inp.peak_freqs[p];
            let row_start = p * num_fbins;
            for f in 0..num_fbins {
                let lo = inp.freq_edges[[0, f]];
                let hi = inp.freq_edges[[1, f]];
                if pf >= lo && pf < hi {
                    membership[row_start + f] = true;
                }
            }
        }
        membership
    };

    // Precompute per-stage validity per time: (n_times, 5) flattened.
    let stage_valid_mask = {
        let mut valid = vec![false; n_times * NUM_STAGES];
        for t in 0..n_times {
            if !inp.c_valid[t] {
                continue;
            }
            let stage = inp.c_stages[t];
            if stage >= 1.0 && stage <= NUM_STAGES as f64 && stage.fract() == 0.0 {
                let k = stage as usize; // 1..=5
                valid[t * NUM_STAGES + (k - 1)] = true;
            }
        }
        valid
    };

    let compute_tib = inp.compute_rate || inp.min_time_in_bin > 0.0;
    let (circ_lo, circ_hi) = inp.circular_bounds;
    let circ_span = circ_hi - circ_lo;

    // Per c-bin: parallel
    let c_metric_slice = inp
        .c_metric
        .as_slice()
        .ok_or_else(|| "c_metric must be contiguous".to_string())?;
    let peak_c_slice = inp
        .peak_c
        .as_slice()
        .ok_or_else(|| "peak_c must be contiguous".to_string())?;

    // Each c-bin independently writes one row of c_mat and time_in_bin.
    // The rows are collected and assembled afterwards.
    let per_bin: Vec<CBinRow> = (0..num_cbins)
        .into_par_iter()
        .map(|c_bin| {
            let bin_lo = inp.c_edges[[0, c_bin]];
            let bin_hi = inp.c_edges[[1, c_bin]];

            // Which form of the bin test applies. A circular bin that
            // reaches a bound wraps around to the other end of the range.
            enum BinCond {
                Normal { lo: f64, hi: f64 },
                WrapLow { wrap_lo: f64, hi: f64 },
                WrapHigh { wrap_hi: f64, lo: f64 },
            }
            let cond = if inp.circular && bin_lo <= circ_lo {
                BinCond::WrapLow { wrap_lo: bin_lo + circ_span, hi: bin_hi }
            } else if inp.circular && bin_hi >= circ_hi {
                BinCond::WrapHigh { wrap_hi: bin_hi - circ_span, lo: bin_lo }
            } else {
                BinCond::Normal { lo: bin_lo, hi: bin_hi }
            };
            // True when a C-metric value (of a time sample or of a peak)
            // falls in this C-bin.
            let in_c_bin = |v: f64| -> bool {
                match cond {
                    BinCond::Normal { lo, hi } => v >= lo && v < hi,
                    BinCond::WrapLow { wrap_lo, hi } => v >= wrap_lo || v < hi,
                    BinCond::WrapHigh { wrap_hi, lo } => v < wrap_hi || v >= lo,
                }
            };

            // Build per-stage time-in-bin.
            let mut tib_per_stage = [0.0f64; NUM_STAGES];
            let mut tib_allstages_count = 0u64;
            for t in 0..n_times {
                let v = c_metric_slice[t];
                if !in_c_bin(v) {
                    continue;
                }
                if inp.c_valid_allstages[t] {
                    tib_allstages_count += 1;
                }
                let base = t * NUM_STAGES;
                for k in 0..NUM_STAGES {
                    if stage_valid_mask[base + k] {
                        tib_per_stage[k] += 1.0;
                    }
                }
            }
            let minutes = inp.c_dt / 60.0;
            for k in 0..NUM_STAGES {
                tib_per_stage[k] *= minutes;
            }
            let tib_allstages = tib_allstages_count as f64 * minutes;

            let mut prop_per_stage = [0.0f64; NUM_STAGES];
            if tib_allstages > 0.0 {
                for k in 0..NUM_STAGES {
                    prop_per_stage[k] = tib_per_stage[k] / tib_allstages;
                }
            }

            // Guard against min_time_in_bin: if sum < min, row stays NaN.
            let tib_sum: f64 = tib_per_stage.iter().sum();
            if compute_tib && tib_sum < inp.min_time_in_bin {
                // Row all NaN
                return CBinRow {
                    counts: vec![f64::NAN; num_fbins],
                    time_in_bin: tib_per_stage,
                    prop_in_bin: prop_per_stage,
                };
            }

            // Count peaks whose peak_c falls in this c-bin.
            // For each such peak, add 1 to counts[f] for each freq bin that
            // contains its peak_freq.
            let mut counts = vec![0.0f64; num_fbins];
            for p in 0..n_peaks {
                if !in_c_bin(peak_c_slice[p]) {
                    continue;
                }
                let row_start = p * num_fbins;
                for f in 0..num_fbins {
                    if peak_in_fbin[row_start + f] {
                        counts[f] += 1.0;
                    }
                }
            }

            // Rate normalization: divide by sum of stage minutes.
            if inp.compute_rate && tib_sum > 0.0 {
                for f in 0..num_fbins {
                    counts[f] /= tib_sum;
                }
            }

            CBinRow {
                counts,
                time_in_bin: tib_per_stage,
                prop_in_bin: prop_per_stage,
            }
        })
        .collect();

    // Assemble arrays from per-bin rows.
    let mut c_mat = Array2::<f64>::zeros((num_cbins, num_fbins));
    let mut time_in_bin = Array2::<f64>::zeros((num_cbins, NUM_STAGES));
    let mut prop_in_bin = Array2::<f64>::zeros((num_cbins, NUM_STAGES));
    for (c_bin, bin_row) in per_bin.into_iter().enumerate() {
        for f in 0..num_fbins {
            c_mat[[c_bin, f]] = bin_row.counts[f];
        }
        for k in 0..NUM_STAGES {
            time_in_bin[[c_bin, k]] = bin_row.time_in_bin[k];
            prop_in_bin[[c_bin, k]] = bin_row.prop_in_bin[k];
        }
    }

    // Peak_at_freq: total count per freq bin (over all peaks, no c-bin filter).
    let mut peak_at_freq = Array1::<f64>::zeros(num_fbins);
    for p in 0..n_peaks {
        let row_start = p * num_fbins;
        for f in 0..num_fbins {
            if peak_in_fbin[row_start + f] {
                peak_at_freq[f] += 1.0;
            }
        }
    }

    if inp.min_peak_at_freq > 0 {
        let min_peaks = inp.min_peak_at_freq as f64;
        for f in 0..num_fbins {
            if peak_at_freq[f] < min_peaks {
                for c_bin in 0..num_cbins {
                    c_mat[[c_bin, f]] = f64::NAN;
                }
            }
        }
    }

    // Axis normalization.
    if inp.norm_dim == 1 || inp.norm_dim == 2 {
        if inp.norm_dim == 1 {
            // Each frequency column sums to 1 over the C-bins. A column
            // whose finite values sum to zero is left as it is.
            for f in 0..num_fbins {
                let mut column_sum = 0.0f64;
                for c_bin in 0..num_cbins {
                    let v = c_mat[[c_bin, f]];
                    if v.is_finite() {
                        column_sum += v;
                    }
                }
                if column_sum == 0.0 {
                    column_sum = 1.0;
                }
                for c_bin in 0..num_cbins {
                    c_mat[[c_bin, f]] /= column_sum;
                }
            }
        } else {
            // Each C-bin row sums to 1 over the frequency bins.
            for c_bin in 0..num_cbins {
                let mut row_sum = 0.0f64;
                for f in 0..num_fbins {
                    let v = c_mat[[c_bin, f]];
                    if v.is_finite() {
                        row_sum += v;
                    }
                }
                if row_sum == 0.0 {
                    row_sum = 1.0;
                }
                for f in 0..num_fbins {
                    c_mat[[c_bin, f]] /= row_sum;
                }
            }
        }
    }

    Ok(HistogramOutputs {
        c_mat,
        time_in_bin,
        prop_in_bin,
        peak_at_freq,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array2};

    #[test]
    fn trivial_noncircular_counts() {
        // 10 time samples, c_metric uniformly 0..9, stage 1, all valid.
        // peak_c all in [0, 5). Peak_freqs at 2Hz. 2 c-bins: [0,5), [5,10).
        // freq bins: [1, 3).
        let c_metric = Array1::from(vec![0., 1., 2., 3., 4., 5., 6., 7., 8., 9.]);
        let c_stages = Array1::from(vec![1.0; 10]);
        let c_valid = vec![true; 10];
        let c_valid_all = vec![true; 10];
        let peak_freqs = Array1::from(vec![2.0, 2.0, 2.0]);
        let peak_c = Array1::from(vec![0.5, 1.5, 4.5]);
        let freq_edges: Array2<f64> = array![[1.0], [3.0]];
        let c_edges: Array2<f64> = array![[0.0, 5.0], [5.0, 10.0]];
        let inp = HistogramInputs {
            c_metric: c_metric.view(),
            c_stages: c_stages.view(),
            c_dt: 1.0,
            c_valid: &c_valid,
            c_valid_allstages: &c_valid_all,
            peak_freqs: peak_freqs.view(),
            peak_c: peak_c.view(),
            freq_edges: freq_edges.view(),
            c_edges: c_edges.view(),
            circular: false,
            circular_bounds: (0.0, 0.0),
            norm_dim: 0,
            compute_rate: false,
            min_time_in_bin: 0.0,
            min_peak_at_freq: 0,
        };
        let out = tfpeak_histogram(&inp).unwrap();
        // 3 peaks, all in freq bin 0. All 3 fall in c-bin 0 (0.5, 1.5 and 4.5
        // are in [0, 5)), none in c-bin 1.
        assert_eq!(out.c_mat[[0, 0]], 3.0);
        assert_eq!(out.c_mat[[1, 0]], 0.0);
        // peak_at_freq: 3 peaks all at 2Hz → 3
        assert_eq!(out.peak_at_freq[0], 3.0);
        // 5 time samples in each c-bin, stage 1, 1s each = 5/60 min
        assert!((out.time_in_bin[[0, 0]] - 5.0 / 60.0).abs() < 1e-12);
        assert!((out.time_in_bin[[1, 0]] - 5.0 / 60.0).abs() < 1e-12);
    }

    #[test]
    fn circular_wrap_low_edge() {
        // c-bin [-pi, -pi + eps) with circular wrap covers values near +pi too.
        // 2 samples at -pi+0.01, 2 at +pi-0.01, and a single c-bin spanning a tiny
        // wrap.
        let c_metric = Array1::from(vec![
            -std::f64::consts::PI + 0.01,
            -std::f64::consts::PI + 0.01,
            std::f64::consts::PI - 0.01,
            std::f64::consts::PI - 0.01,
        ]);
        let c_stages = Array1::from(vec![1.0; 4]);
        let c_valid = vec![true; 4];
        let c_valid_all = vec![true; 4];
        let peak_freqs = Array1::from(vec![2.0, 2.0]);
        let peak_c = Array1::from(vec![-3.14, 3.14]);
        let freq_edges: Array2<f64> = array![[1.0], [3.0]];
        // One bin [-pi - 0.02, -pi + 0.02) whose left edge wraps.
        let c_edges: Array2<f64> = array![[-std::f64::consts::PI - 0.02], [-std::f64::consts::PI + 0.02]];
        let inp = HistogramInputs {
            c_metric: c_metric.view(),
            c_stages: c_stages.view(),
            c_dt: 1.0,
            c_valid: &c_valid,
            c_valid_allstages: &c_valid_all,
            peak_freqs: peak_freqs.view(),
            peak_c: peak_c.view(),
            freq_edges: freq_edges.view(),
            c_edges: c_edges.view(),
            circular: true,
            circular_bounds: (-std::f64::consts::PI, std::f64::consts::PI),
            norm_dim: 0,
            compute_rate: false,
            min_time_in_bin: 0.0,
            min_peak_at_freq: 0,
        };
        let out = tfpeak_histogram(&inp).unwrap();
        // Both peaks (-3.14 near +pi boundary, 3.14 near -pi boundary) wrap in:
        //   wrap_lo = lo + 2pi ≈ pi - 0.02, hi_e ≈ -pi + 0.02
        //   cond: (v >= wrap_lo) OR (v < hi_e)
        //   peak_c[0] = -3.14: -3.14 >= pi-0.02? NO. -3.14 < -pi+0.02 ≈ -3.12? YES → in bin
        //   peak_c[1] = 3.14:   3.14 >= pi-0.02 ≈ 3.12? YES → in bin
        assert_eq!(out.c_mat[[0, 0]], 2.0);
    }
}
