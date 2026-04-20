//! Core 2D TF-peak × C-metric histogram — port of pydynamo
//! `soph.histogram.tfpeak_histogram` (itself a port of MATLAB
//! `TFPeakHistogram.m`).
//!
//! Stage convention: DYNAM-O uses 1..=5 (1=N3, 2=N2, 3=N1, 4=REM, 5=Wake).
//! `time_in_bin` returns 5 columns for these 5 stages.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rayon::prelude::*;

pub struct HistogramInputs<'a> {
    pub c_metric: ArrayView1<'a, f64>,
    pub c_stages: ArrayView1<'a, f64>,
    pub c_dt: f64,
    pub c_valid: &'a [bool],
    pub c_valid_allstages: &'a [bool],
    pub peak_freqs: ArrayView1<'a, f64>,
    pub peak_c: ArrayView1<'a, f64>,
    pub freq_edges: ArrayView2<'a, f64>, // (2, num_fbins)
    pub c_edges: ArrayView2<'a, f64>,    // (2, num_cbins)
    pub circular: bool,
    pub circular_bounds: (f64, f64),
    pub norm_dim: i32,     // 0 = no, 1 = along axis 0, 2 = along axis 1
    pub compute_rate: bool,
    pub min_time_in_bin: f64, // minutes
    pub min_peak_at_freq: i32,
}

pub struct HistogramOutputs {
    pub c_mat: Array2<f64>,       // (num_cbins, num_fbins)
    pub time_in_bin: Array2<f64>, // (num_cbins, 5) minutes
    pub prop_in_bin: Array2<f64>, // (num_cbins, 5)
    pub peak_at_freq: Array1<f64>, // (num_fbins,)
}

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

    // Precompute peak→freq-bin mapping once.
    // all_infreqbin[p] = bin index (or -1 if none).
    // Freq bins are in "partial" mode (see create_bins), so they may overlap
    // — a peak can land in multiple freq bins. So we need a per-bin count,
    // not a per-peak single assignment. Build a flat (N_peaks, num_fbins) bool
    // bitmap via a packed-bit matrix (saves memory for large N_peaks).
    //
    // For typical segment: N_peaks ≤ 8k, num_fbins = 151 → 8000*151 = 1.2M
    // booleans = 1.2MB. Fine as flat Vec<bool>.
    let peak_in_fbin = {
        let mut v = vec![false; n_peaks * num_fbins];
        for p in 0..n_peaks {
            let pf = inp.peak_freqs[p];
            let row_start = p * num_fbins;
            for f in 0..num_fbins {
                let lo = inp.freq_edges[[0, f]];
                let hi = inp.freq_edges[[1, f]];
                if pf >= lo && pf < hi {
                    v[row_start + f] = true;
                }
            }
        }
        v
    };

    // Precompute per-stage validity per time: (n_times, 5) flattened.
    let stage_valid_mask = {
        let mut v = vec![false; n_times * 5];
        for t in 0..n_times {
            if !inp.c_valid[t] {
                continue;
            }
            let stg = inp.c_stages[t];
            if stg >= 1.0 && stg <= 5.0 && stg.fract() == 0.0 {
                let k = stg as usize; // 1..=5
                v[t * 5 + (k - 1)] = true;
            }
        }
        v
    };

    let compute_tib = inp.compute_rate || inp.min_time_in_bin > 0.0;
    let (low_b, high_b) = inp.circular_bounds;
    let crange = high_b - low_b;

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
    // Collect as Vec<(row_cmat, row_tib, row_prop)> then assemble.
    let per_bin: Vec<(Vec<f64>, [f64; 5], [f64; 5])> = (0..num_cbins)
        .into_par_iter()
        .map(|s| {
            let lo_e = inp.c_edges[[0, s]];
            let hi_e = inp.c_edges[[1, s]];

            // Decide which side of the bin condition applies. Encode the
            // branch as an enum discriminant so the inner hot-loops don't
            // call through a boxed closure (saves a per-bin allocation).
            enum BinCond {
                Normal { lo: f64, hi: f64 },
                WrapLow { wrap_lo: f64, hi: f64 },
                WrapHigh { wrap_hi: f64, lo: f64 },
            }
            let cond = if inp.circular && lo_e <= low_b {
                BinCond::WrapLow { wrap_lo: lo_e + crange, hi: hi_e }
            } else if inp.circular && hi_e >= high_b {
                BinCond::WrapHigh { wrap_hi: hi_e - crange, lo: lo_e }
            } else {
                BinCond::Normal { lo: lo_e, hi: hi_e }
            };
            let tib_fn = |v: f64| -> bool {
                match cond {
                    BinCond::Normal { lo, hi } => v >= lo && v < hi,
                    BinCond::WrapLow { wrap_lo, hi } => v >= wrap_lo || v < hi,
                    BinCond::WrapHigh { wrap_hi, lo } => v < wrap_hi || v >= lo,
                }
            };

            // Build per-stage time-in-bin.
            let mut tib_per_stage = [0.0f64; 5];
            let mut tib_allstages_count = 0u64;
            for t in 0..n_times {
                let v = c_metric_slice[t];
                if !tib_fn(v) {
                    continue;
                }
                if inp.c_valid_allstages[t] {
                    tib_allstages_count += 1;
                }
                let base = t * 5;
                for k in 0..5 {
                    if stage_valid_mask[base + k] {
                        tib_per_stage[k] += 1.0;
                    }
                }
            }
            let minutes = inp.c_dt / 60.0;
            for k in 0..5 {
                tib_per_stage[k] *= minutes;
            }
            let tib_allstages = tib_allstages_count as f64 * minutes;

            let mut prop_per_stage = [0.0f64; 5];
            if tib_allstages > 0.0 {
                for k in 0..5 {
                    prop_per_stage[k] = tib_per_stage[k] / tib_allstages;
                }
            }

            // Guard against min_time_in_bin: if sum < min, row stays NaN.
            let tib_sum: f64 = tib_per_stage.iter().sum();
            if compute_tib && tib_sum < inp.min_time_in_bin {
                // Row all NaN
                return (vec![f64::NAN; num_fbins], tib_per_stage, prop_per_stage);
            }

            // Count peaks whose peak_c falls in this c-bin.
            // For each such peak, add 1 to counts[f] for each freq bin that
            // contains its peak_freq.
            let mut counts = vec![0.0f64; num_fbins];
            for p in 0..n_peaks {
                if !tib_fn(peak_c_slice[p]) {
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

            (counts, tib_per_stage, prop_per_stage)
        })
        .collect();

    // Assemble arrays from per-bin rows.
    let mut c_mat = Array2::<f64>::zeros((num_cbins, num_fbins));
    let mut time_in_bin = Array2::<f64>::zeros((num_cbins, 5));
    let mut prop_in_bin = Array2::<f64>::zeros((num_cbins, 5));
    for (s, (row, tib, prop)) in per_bin.into_iter().enumerate() {
        for f in 0..num_fbins {
            c_mat[[s, f]] = row[f];
        }
        for k in 0..5 {
            time_in_bin[[s, k]] = tib[k];
            prop_in_bin[[s, k]] = prop[k];
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
        let thr = inp.min_peak_at_freq as f64;
        for f in 0..num_fbins {
            if peak_at_freq[f] < thr {
                for s in 0..num_cbins {
                    c_mat[[s, f]] = f64::NAN;
                }
            }
        }
    }

    // Axis normalization.
    if inp.norm_dim == 1 || inp.norm_dim == 2 {
        let axis = inp.norm_dim - 1;
        if axis == 0 {
            // sum along rows (fix f, sum over s)
            for f in 0..num_fbins {
                let mut s_sum = 0.0f64;
                for s in 0..num_cbins {
                    let v = c_mat[[s, f]];
                    if v.is_finite() {
                        s_sum += v;
                    }
                }
                if s_sum == 0.0 {
                    s_sum = 1.0;
                }
                for s in 0..num_cbins {
                    c_mat[[s, f]] /= s_sum;
                }
            }
        } else {
            for s in 0..num_cbins {
                let mut f_sum = 0.0f64;
                for f in 0..num_fbins {
                    let v = c_mat[[s, f]];
                    if v.is_finite() {
                        f_sum += v;
                    }
                }
                if f_sum == 0.0 {
                    f_sum = 1.0;
                }
                for f in 0..num_fbins {
                    c_mat[[s, f]] /= f_sum;
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
        // peak_c all in [0, 5); peak_freqs at 2Hz. 2 c-bins: [0,5), [5,10).
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
        // 3 peaks, all in freq bin 0; 3 in c-bin 0 (0.5, 1.5, 4.5 — wait
        // 4.5 is in [0,5) so yes), 0 in c-bin 1.
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
        // 2 samples at -pi+0.01, 2 at +pi-0.01; single c-bin spanning a tiny
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
        // One bin whose left edge is -pi-0.01 (wraps).
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
        //   wrap_lo = lo + 2pi ≈ pi - 0.02; or hi_e ≈ -pi + 0.02
        //   cond: (v >= wrap_lo) OR (v < hi_e)
        //   peak_c[0] = -3.14: -3.14 >= pi-0.02? NO; -3.14 < -pi+0.02 ≈ -3.12? YES → in bin
        //   peak_c[1] = 3.14:   3.14 >= pi-0.02 ≈ 3.12? YES → in bin
        assert_eq!(out.c_mat[[0, 0]], 2.0);
    }
}
