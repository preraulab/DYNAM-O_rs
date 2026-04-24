//! Artifact detection — port of pydynamo `artifacts.py::detect_artifacts`
//! (and MATLAB `toolbox/helper_functions/artifact_detection/detect_artifacts.m`)
//! with default parameters.
//!
//! Two-band detection: broad-band (`bb_pass`=0.1 Hz HP) + high-frequency
//! (`hf_pass`=35 Hz HP). Per band:
//!   1. Cheby-I order-4 IIR highpass (0.2 dB ripple)
//!   2. abs(hilbert(·)) → envelope
//!   3. movmean over `smooth_duration`
//!   4. log
//!   5. movmedian over `detrend_duration` → subtract (detrend)
//!   6. iterative robust z-score (median + MAD) until convergence
//!
//! **Slope test is NOT ported.** The MATLAB / pydynamo `slope_test=True`
//! branch needs a multitaper spectrogram; deferred until Part 5 step 7
//! (multitaper_rs pull-in). `detect_artifacts_default` here sets
//! `slope_test=false`; the caller can OR in a slope mask computed elsewhere.
//!
//! Bit-equivalence target: the band-based path must agree with pydynamo
//! when pydynamo is invoked with `slope_test=False`.

use crate::filter_design::cheby1_sos;
use crate::signal::{hilbert, sosfiltfilt};

/// Options for `detect_artifacts`. Matches pydynamo defaults.
#[derive(Debug, Clone)]
pub struct ArtifactOpts {
    pub hf_pass: f64,
    pub hf_crit: f64,
    pub bb_pass: f64,
    pub bb_crit: f64,
    pub hf_detrend: bool,
    pub bb_detrend: bool,
    pub smooth_duration: f64,
    pub detrend_duration: f64,
    pub buffer_duration: f64,
    /// "robust" (median + MAD, pydynamo default) or "standard" (mean + std).
    pub zscore_method: ZScoreMethod,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZScoreMethod {
    Robust,
    Standard,
}

impl Default for ArtifactOpts {
    fn default() -> Self {
        Self {
            hf_pass: 35.0,
            hf_crit: 5.5,
            bb_pass: 0.1,
            bb_crit: 5.5,
            hf_detrend: true,
            bb_detrend: true,
            smooth_duration: 2.0,
            detrend_duration: 300.0,
            buffer_duration: 0.0,
            zscore_method: ZScoreMethod::Robust,
        }
    }
}

/// True where `data` sits in a run of ≥ `min_run` identical values.
/// Port of pydynamo `_flat_mask`.
pub fn flat_run_mask(data: &[f64], min_run: usize) -> Vec<bool> {
    let n = data.len();
    if n == 0 {
        return vec![];
    }
    let mut mask = vec![false; n];
    let mut run_start = 0usize;
    for i in 1..=n {
        if i == n || data[i] != data[i - 1] {
            let run_len = i - run_start;
            if run_len >= min_run {
                for j in run_start..i {
                    mask[j] = true;
                }
            }
            run_start = i;
        }
    }
    mask
}

/// MATLAB `movmean(x, win)` — centered moving mean with shrinking window
/// at the endpoints (partial means on the edges).
fn movmean(x: &[f64], win: usize) -> Vec<f64> {
    let n = x.len();
    if win <= 1 || n == 0 {
        return x.to_vec();
    }
    let half_l = (win - 1) / 2;
    let half_r = win / 2;
    let mut csum = vec![0.0_f64; n + 1];
    for i in 0..n {
        csum[i + 1] = csum[i] + x[i];
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let a = i.saturating_sub(half_l);
        let b = (i + half_r + 1).min(n);
        out.push((csum[b] - csum[a]) / (b - a) as f64);
    }
    out
}

/// MATLAB `movmedian(x, win)` — centered moving median with shrinking window.
/// O(n * win log win) naive implementation; good enough for our use (win ~= 30k
/// on a 300 s window at 100 Hz, called twice per artifact run). Matches
/// pydynamo `_movmedian` at every sample.
fn movmedian(x: &[f64], win: usize) -> Vec<f64> {
    let n = x.len();
    if win <= 1 || n == 0 {
        return x.to_vec();
    }
    let half_l = (win - 1) / 2;
    let half_r = win / 2;
    let mut out = Vec::with_capacity(n);
    let mut scratch: Vec<f64> = Vec::with_capacity(win);
    for i in 0..n {
        let a = i.saturating_sub(half_l);
        let b = (i + half_r + 1).min(n);
        scratch.clear();
        scratch.extend_from_slice(&x[a..b]);
        scratch.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let m = scratch.len();
        let med = if m % 2 == 1 {
            scratch[m / 2]
        } else {
            0.5 * (scratch[m / 2 - 1] + scratch[m / 2])
        };
        out.push(med);
    }
    out
}

/// Median of finite values. Returns NaN for all-NaN input.
fn median(xs: &[f64]) -> f64 {
    let mut v: Vec<f64> = xs.iter().filter(|x| x.is_finite()).copied().collect();
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let m = v.len();
    if m % 2 == 1 {
        v[m / 2]
    } else {
        0.5 * (v[m / 2 - 1] + v[m / 2])
    }
}

/// MATLAB default `mad(x)`: mean absolute deviation from the mean.
/// (Not median absolute deviation! pydynamo matches.)
fn mad_meanabs(xs: &[f64]) -> f64 {
    let finite: Vec<f64> = xs.iter().filter(|x| x.is_finite()).copied().collect();
    if finite.is_empty() {
        return f64::NAN;
    }
    let mean_ = finite.iter().sum::<f64>() / finite.len() as f64;
    finite.iter().map(|&v| (v - mean_).abs()).sum::<f64>() / finite.len() as f64
}

fn mean_std(xs: &[f64]) -> (f64, f64) {
    let finite: Vec<f64> = xs.iter().filter(|x| x.is_finite()).copied().collect();
    if finite.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    let n = finite.len() as f64;
    let mean_ = finite.iter().sum::<f64>() / n;
    if finite.len() < 2 {
        return (mean_, f64::NAN);
    }
    let var_ = finite.iter().map(|&v| (v - mean_).powi(2)).sum::<f64>() / (n - 1.0);
    (mean_, var_.sqrt())
}

fn center_scale(xs: &[f64], method: ZScoreMethod) -> (f64, f64) {
    match method {
        ZScoreMethod::Robust => (median(xs), mad_meanabs(xs)),
        ZScoreMethod::Standard => mean_std(xs),
    }
}

/// Iteratively flag samples where |z| > crit, recomputing centering stats on
/// the shrinking good set until convergence. Returns a bad-sample mask
/// including the input `bad`.
pub fn iterative_robust_zscore(
    y: &[f64],
    bad: &[bool],
    crit: f64,
    method: ZScoreMethod,
) -> Vec<bool> {
    assert_eq!(y.len(), bad.len());
    let mut mask = bad.to_vec();
    loop {
        let good_vals: Vec<f64> = y
            .iter()
            .zip(mask.iter())
            .filter_map(|(v, &m)| if !m && v.is_finite() { Some(*v) } else { None })
            .collect();
        if good_vals.is_empty() {
            return mask;
        }
        let (mid, scale) = center_scale(&good_vals, method);
        if !scale.is_finite() || scale == 0.0 {
            return mask;
        }
        let mut any_over = false;
        for i in 0..y.len() {
            if !mask[i] && y[i].is_finite() {
                let z = (y[i] - mid) / scale;
                if z.abs() > crit {
                    mask[i] = true;
                    any_over = true;
                }
            }
        }
        if !any_over {
            return mask;
        }
    }
}

/// Flag samples more than `outlier_scalar` SD from the mean of the currently-
/// good samples, returning the union with `bad`.
fn find_outlier_noise(data: &[f64], bad: &[bool], outlier_scalar: f64) -> Vec<bool> {
    let mut out = bad.to_vec();
    let good_vals: Vec<f64> = data
        .iter()
        .zip(bad.iter())
        .filter_map(|(v, &m)| if !m && v.is_finite() { Some(*v) } else { None })
        .collect();
    if good_vals.is_empty() {
        return out;
    }
    let n = good_vals.len() as f64;
    let mean_ = good_vals.iter().sum::<f64>() / n;
    if good_vals.len() < 2 {
        return out;
    }
    let var_ = good_vals.iter().map(|&v| (v - mean_).powi(2)).sum::<f64>() / (n - 1.0);
    let sd = var_.sqrt();
    let lo = mean_ - outlier_scalar * sd;
    let hi = mean_ + outlier_scalar * sd;
    for i in 0..data.len() {
        if data[i] <= lo || data[i] >= hi {
            out[i] = true;
        }
    }
    out
}

/// Linear interp over bad samples using the nearest good values on either side
/// (padded with good-endpoint values). Replaces pydynamo
/// `detect_artifacts` lines 341-350.
fn interp_bad_samples(data: &[f64], bad: &[bool]) -> Vec<f64> {
    let n = data.len();
    let mut fixed = data.to_vec();
    let good_idx: Vec<usize> = (0..n).filter(|&i| !bad[i]).collect();
    if good_idx.is_empty() {
        return fixed;
    }
    let good_vals: Vec<f64> = good_idx.iter().map(|&i| data[i]).collect();
    let xp: Vec<f64> = std::iter::once(-1.0)
        .chain(good_idx.iter().map(|&i| i as f64))
        .chain(std::iter::once(n as f64))
        .collect();
    let fp: Vec<f64> = std::iter::once(*good_vals.first().unwrap())
        .chain(good_vals.iter().copied())
        .chain(std::iter::once(*good_vals.last().unwrap()))
        .collect();

    for i in 0..n {
        if bad[i] {
            let q = i as f64;
            // Binary search: find j s.t. xp[j-1] <= q < xp[j]
            let j = xp.partition_point(|&t| t <= q);
            let t0 = xp[j - 1];
            let t1 = xp[j];
            let y0 = fp[j - 1];
            let y1 = fp[j];
            fixed[i] = y0 + (q - t0) / (t1 - t0) * (y1 - y0);
        }
    }
    fixed
}

/// Detect artifacts in one frequency band.
fn compute_band_artifacts(
    data: &[f64],
    fs: f64,
    passband: f64,
    crit: f64,
    bad: &[bool],
    smooth_duration: f64,
    detrend_duration: f64,
    detrend_on: bool,
    zscore_method: ZScoreMethod,
) -> Vec<bool> {
    // Cheby-I order-4 highpass SOS with 0.2 dB ripple.
    let sections = cheby1_sos(4, 0.2, &[passband], "highpass", fs);
    let sos: Vec<[f64; 6]> = sections.into_iter().collect();

    let filtered = sosfiltfilt(&sos, data);
    let (re, im) = hilbert(&filtered);
    // abs(analytic) = sqrt(re^2 + im^2). pydynamo uses np.abs(scipy.signal.hilbert(...)).
    let env: Vec<f64> = re
        .iter()
        .zip(im.iter())
        .map(|(r, i)| (r * r + i * i).sqrt())
        .collect();

    let win_smooth = (smooth_duration * fs).round().max(1.0) as usize;
    let mut y = movmean(&env, win_smooth);
    for v in y.iter_mut() {
        *v = if *v > 0.0 { v.ln() } else { f64::NAN };
    }
    if detrend_on {
        let win_det = (detrend_duration * fs).round().max(1.0) as usize;
        let med = movmedian(&y, win_det);
        for i in 0..y.len() {
            y[i] -= med[i];
        }
    }
    iterative_robust_zscore(&y, bad, crit, zscore_method)
}

/// Morphological dilation of a bool mask by `k` samples on each side.
fn binary_dilate(mask: &[bool], k: usize) -> Vec<bool> {
    let n = mask.len();
    if n == 0 || k == 0 {
        return mask.to_vec();
    }
    let mut out = vec![false; n];
    // Find runs of True; expand each run by k on both sides.
    let mut i = 0;
    while i < n {
        if mask[i] {
            let start = i.saturating_sub(k);
            let mut j = i;
            while j < n && mask[j] {
                j += 1;
            }
            let end = (j + k).min(n);
            for p in start..end {
                out[p] = true;
            }
            i = j;
        } else {
            i += 1;
        }
    }
    out
}

/// Detect artifacts in an EEG time series. Two-band (BB + HF) detection + flat-
/// run + outlier-noise flagging. **Does NOT run the slope test** — the caller
/// must OR in any slope-based mask computed via multitaper elsewhere.
pub fn detect_artifacts(data: &[f64], fs: f64, opts: &ArtifactOpts) -> Vec<bool> {
    let n = data.len();
    if n == 0 {
        return vec![];
    }

    // Flat runs of ≥ 1 s.
    let flat = flat_run_mask(data, fs.round().max(1.0) as usize);
    let mut bad: Vec<bool> = (0..n)
        .map(|i| !data[i].is_finite() || flat[i])
        .collect();
    bad = find_outlier_noise(data, &bad, 10.0);

    // Interp through bad samples so filters don't see NaN / flats.
    let data_fixed = interp_bad_samples(data, &bad);

    let hf_art = compute_band_artifacts(
        &data_fixed, fs, opts.hf_pass, opts.hf_crit, &bad,
        opts.smooth_duration, opts.detrend_duration, opts.hf_detrend,
        opts.zscore_method,
    );
    let bb_art = compute_band_artifacts(
        &data_fixed, fs, opts.bb_pass, opts.bb_crit, &bad,
        opts.smooth_duration, opts.detrend_duration, opts.bb_detrend,
        opts.zscore_method,
    );
    let mut artifacts: Vec<bool> = (0..n)
        .map(|i| hf_art[i] || bb_art[i] || bad[i])
        .collect();

    if opts.buffer_duration > 0.0 {
        let k = (opts.buffer_duration * fs).round() as usize;
        if k > 0 {
            artifacts = binary_dilate(&artifacts, k);
        }
    }
    artifacts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_run_detection() {
        let x = vec![1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 3.0];
        // min_run=3 → the first run (4 × 1.0) and last (3 × 3.0) are flat.
        let m = flat_run_mask(&x, 3);
        assert_eq!(
            m,
            vec![true, true, true, true, false, false, true, true, true]
        );
    }

    #[test]
    fn movmean_shrinking_window() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let m = movmean(&x, 3);
        // edges partial: [avg(1,2), avg(1,2,3), avg(2,3,4), avg(3,4,5), avg(4,5)]
        assert!((m[0] - 1.5).abs() < 1e-12);
        assert!((m[1] - 2.0).abs() < 1e-12);
        assert!((m[2] - 3.0).abs() < 1e-12);
        assert!((m[3] - 4.0).abs() < 1e-12);
        assert!((m[4] - 4.5).abs() < 1e-12);
    }

    #[test]
    fn movmedian_shrinking_window() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let m = movmedian(&x, 3);
        // edge: median(1,2)=1.5; center: median(1,2,3)=2, etc.
        assert!((m[0] - 1.5).abs() < 1e-12);
        assert!((m[1] - 2.0).abs() < 1e-12);
        assert!((m[2] - 3.0).abs() < 1e-12);
        assert!((m[3] - 4.0).abs() < 1e-12);
        assert!((m[4] - 4.5).abs() < 1e-12);
    }

    #[test]
    fn iter_robust_zscore_converges() {
        // Outlier at index 5; should be flagged after 1+ iterations.
        let y = vec![1.0, 1.1, 0.9, 1.05, 0.95, 50.0, 1.0, 1.02];
        let bad = vec![false; 8];
        let out = iterative_robust_zscore(&y, &bad, 3.0, ZScoreMethod::Robust);
        assert!(out[5], "outlier at index 5 should be flagged");
        for i in [0, 1, 2, 3, 4, 6, 7] {
            assert!(!out[i], "index {} should not be flagged", i);
        }
    }
}
