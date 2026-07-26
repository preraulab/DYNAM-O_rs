//! SO-power time series (post-spectrogram pipeline).
//!
//! Port of `computeSOpower.m` lines ~115–end and pydynamo
//! `soph/sopower.py::compute_so_power` (steps 3–8). Takes an already-computed
//! multitaper spectrogram over the SO band and produces the normalized
//! SO-power time series.
//!
//! Why the split: the multitaper spectrogram is the dominant cost and already
//! has acceleration options in both ecosystems — MATLAB ships
//! `multitaper_spectrogram_mex`, pydynamo has an optional Rust backend via
//! the `multitaper_rs` crate. Keeping this function spect-in / power-out means
//! we don't re-implement DPSS here, and the same Rust implementation can be
//! wrapped for MATLAB (MEX) and pydynamo (PyO3) to cover the post-spectrogram
//! work (dB conversion, outlier z-score, percentile-shift normalization,
//! upsampling) in all three pipelines.
//!
//! Equivalence with pydynamo `compute_so_power` (and thus `computeSOpower.m`)
//! is by construction: each step below names the reference-line it ports.

use crate::baseline::nanpercentile_hazen;
use crate::peak_assign::{interp_linear_padded, interp_previous};

/// Normalization method — one of:
///   * `Shift { ptile, stages }`  — "p{N}shift{S}" (e.g. p2shift1234).
///     Subtract `ptile`-th percentile of SOpower_db over the given stages.
///   * `Percent` — (x - p1) / (p99 - p1)
///   * `None_` — return SOpower_db unchanged
#[derive(Debug, Clone)]
pub enum NormMethod {
    Shift { ptile: f64, stages: Vec<i32> },
    Percent,
    None_,
}

impl NormMethod {
    /// Parse the MATLAB-style string form ("p2shift1234", "percent", "none",
    /// "absolute", "%", "%SOP"). Returns `None` if unrecognized.
    pub fn parse(s: &str) -> Option<Self> {
        let lower = s.to_ascii_lowercase();
        // "p{N}shift{S}" where N is 0–100 and S is a string of digits 1–5.
        if let Some(rest) = lower.strip_prefix('p') {
            if let Some(idx) = rest.find("shift") {
                let ptile_str = &rest[..idx];
                let stages_str = &rest[idx + 5..];
                if let Ok(ptile) = ptile_str.parse::<f64>() {
                    if (0.0..=100.0).contains(&ptile) && !stages_str.is_empty() {
                        let mut stages: Vec<i32> = stages_str
                            .chars()
                            .filter_map(|c| c.to_digit(10).map(|d| d as i32))
                            .filter(|&d| (1..=5).contains(&d))
                            .collect();
                        stages.sort_unstable();
                        stages.dedup();
                        if !stages.is_empty() {
                            return Some(NormMethod::Shift { ptile, stages });
                        }
                    }
                }
            }
        }
        match lower.as_str() {
            "percent" | "percentile" | "%" | "%sop" => Some(NormMethod::Percent),
            "none" | "absolute" => Some(NormMethod::None_),
            _ => None,
        }
    }
}

/// Output of `so_power_from_spectrogram`.
pub struct SoPowerOut {
    /// Normalized SO-power, length = retain_Fs ? eeg_times.len() : stimes.len().
    pub so_power_norm: Vec<f64>,
    /// Time stamps matching `so_power_norm`.
    pub so_power_times: Vec<f64>,
    /// Stage per output sample (0 where unstaged or excluded).
    pub so_power_stages: Vec<f64>,
    /// The percentile value used (Shift: single p; Percent: (p1, p99)). `None`
    /// if `NormMethod::None_` or if normalization was bypassed (degenerate).
    pub ptile: Option<PtileUsed>,
}

#[derive(Debug, Clone, Copy)]
pub enum PtileUsed {
    Single(f64),
    Pair(f64, f64),
}

/// Compute the SO-power time series from a pre-computed multitaper spectrogram
/// over the SO band.
///
/// Inputs:
/// * `so_spect` — row-major (F, T) spectrogram. NaN entries propagate (nansum).
/// * `stimes`   — window-center times (length T), *relative to start of input*.
///                Absolute times are computed as `stimes + eeg_times[0]`.
/// * `sfreqs`   — frequency bin centers (length F). Only `df = sfreqs[1]-sfreqs[0]`
///                is used here (the band-integration multiplier).
/// * `eeg_times`— the EEG sample time axis (length N). Used for time_range
///                defaults and (if `retain_fs`) upsampling.
/// * `isexcluded` — per-EEG-sample artifact mask (length N). Upsampled output
///                  has NaN wherever `isexcluded[i]`.
/// * `stage_times`, `stage_vals` — sleep staging to interpolate onto the
///                  output time grid (previous-neighbor step). Pass empty
///                  slices to default stages to all-1.
/// * `time_range` — `(t_start, t_end)` used for percentile normalization scope.
/// * `outlier_threshold` — |z| >= threshold → NaN (3.0 in pydynamo).
/// * `norm_method`
/// * `retain_fs`  — if true, output is upsampled back to `eeg_times`.
#[allow(clippy::too_many_arguments)]
pub fn so_power_from_spectrogram(
    so_spect: &[f64],
    n_freqs: usize,
    n_times: usize,
    stimes: &[f64],
    sfreqs: &[f64],
    eeg_times: &[f64],
    isexcluded: &[bool],
    stage_times: &[f64],
    stage_vals: &[f64],
    time_range: (f64, f64),
    outlier_threshold: f64,
    norm_method: &NormMethod,
    retain_fs: bool,
) -> Result<SoPowerOut, String> {
    if so_spect.len() != n_freqs * n_times {
        return Err(format!(
            "spect length {} != n_freqs * n_times = {} * {}",
            so_spect.len(), n_freqs, n_times
        ));
    }
    if stimes.len() != n_times {
        return Err("stimes length != n_times".into());
    }
    if sfreqs.len() != n_freqs {
        return Err("sfreqs length != n_freqs".into());
    }
    if eeg_times.len() != isexcluded.len() {
        return Err("eeg_times and isexcluded length mismatch".into());
    }
    if eeg_times.is_empty() {
        return Err("eeg_times must be non-empty".into());
    }

    // Step: absolute window-center times. pydynamo:89, MATLAB is symmetrical.
    let t0 = eeg_times[0];
    let mut so_power_times: Vec<f64> = stimes.iter().map(|&s| s + t0).collect();

    // Step: nan-sum along freq axis * df. pydynamo:93-94.
    let df = if n_freqs >= 2 { sfreqs[1] - sfreqs[0] } else { 1.0 };
    let mut so_power: Vec<f64> = Vec::with_capacity(n_times);
    for ct in 0..n_times {
        let mut acc = 0.0_f64;
        let mut any = false;
        for cf in 0..n_freqs {
            let v = so_spect[cf * n_times + ct];
            if v.is_finite() {
                acc += v;
                any = true;
            }
        }
        so_power.push(if any { acc * df } else { f64::NAN });
    }

    // Step: 10*log10(x>0, else NaN).  pydynamo:_nan_pow2db.
    let mut so_power_db: Vec<f64> = so_power
        .iter()
        .map(|&v| if v > 0.0 { 10.0 * v.log10() } else { f64::NAN })
        .collect();

    // Step: stage assignment (previous-neighbor).  pydynamo:98-108.
    let mut so_power_stages: Vec<f64> = if !stage_times.is_empty() && !stage_vals.is_empty() {
        let interp = interp_previous(stage_times, stage_vals, &so_power_times, f64::NAN);
        interp.into_iter().map(|v| if v.is_nan() { 0.0 } else { v }).collect()
    } else {
        vec![1.0; so_power_times.len()]
    };

    // Step: outlier z-score exclusion. pydynamo:110-113.
    {
        let (mean_, std_) = nan_mean_std(&so_power_db);
        if std_.is_finite() && std_ > 0.0 {
            for v in so_power_db.iter_mut() {
                if v.is_finite() {
                    let z = (*v - mean_) / std_;
                    if z.abs() >= outlier_threshold {
                        *v = f64::NAN;
                    }
                }
            }
        }
    }

    // Degenerate path: all-NaN after outlier removal. pydynamo:115-117.
    if so_power_db.iter().all(|v| v.is_nan()) {
        return Ok(SoPowerOut {
            so_power_norm: so_power_db,
            so_power_times,
            so_power_stages,
            ptile: None,
        });
    }

    // Step: normalization. pydynamo:119-146 / MATLAB:167-196.
    let in_range: Vec<bool> = so_power_times
        .iter()
        .map(|&t| t >= time_range.0 && t <= time_range.1)
        .collect();
    let (mut so_power_norm, ptile_used) = match norm_method {
        NormMethod::Shift { ptile, stages } => {
            let sel: Vec<f64> = so_power_db
                .iter()
                .zip(so_power_stages.iter())
                .zip(in_range.iter())
                .filter_map(|((&v, &s), &r)| {
                    if r && v.is_finite() && stages.contains(&(s as i32)) {
                        Some(v)
                    } else {
                        None
                    }
                })
                .collect();
            if sel.is_empty() {
                return Err(format!(
                    "No valid stages {:?} found for shift normalization.",
                    stages
                ));
            }
            let p = nanpercentile_hazen(&sel, *ptile);
            let out: Vec<f64> = so_power_db.iter().map(|&v| v - p).collect();
            (out, Some(PtileUsed::Single(p)))
        }
        NormMethod::Percent => {
            let sel: Vec<f64> = so_power_db
                .iter()
                .zip(in_range.iter())
                .filter_map(|(&v, &r)| if r && v.is_finite() { Some(v) } else { None })
                .collect();
            if sel.is_empty() {
                return Err("No valid samples for percent normalization.".into());
            }
            let p1 = nanpercentile_hazen(&sel, 1.0);
            let p99 = nanpercentile_hazen(&sel, 99.0);
            let denom = p99 - p1;
            let out: Vec<f64> = so_power_db
                .iter()
                .map(|&v| (v - p1) / denom)
                .collect();
            (out, Some(PtileUsed::Pair(p1, p99)))
        }
        NormMethod::None_ => (so_power_db.clone(), None),
    };

    // Step: upsample to EEG rate if requested. pydynamo:148-164.
    if retain_fs {
        let valid_vals: Vec<f64> = so_power_norm
            .iter()
            .filter(|v| v.is_finite())
            .copied()
            .collect();
        let valid_times: Vec<f64> = so_power_norm
            .iter()
            .zip(so_power_times.iter())
            .filter_map(|(v, &t)| if v.is_finite() { Some(t) } else { None })
            .collect();

        if valid_vals.is_empty() {
            // Nothing valid → stay at the downsampled grid.
            return Ok(SoPowerOut {
                so_power_norm,
                so_power_times,
                so_power_stages,
                ptile: ptile_used,
            });
        }
        // Pad endpoints with eeg_times[0]/eeg_times[-1] so interp holds flat
        // outside the valid range (matches pydynamo's xp/fp pad trick).
        let tn = *eeg_times.last().unwrap();
        let mut xp = Vec::with_capacity(valid_times.len() + 2);
        let mut fp = Vec::with_capacity(valid_vals.len() + 2);
        xp.push(t0);
        fp.push(valid_vals[0]);
        xp.extend_from_slice(&valid_times);
        fp.extend_from_slice(&valid_vals);
        xp.push(tn);
        fp.push(*valid_vals.last().unwrap());

        let mut up = interp_linear_padded(&xp, &fp, eeg_times);
        // Match MATLAB/Python: interpolate through finite coarse-grid values,
        // then restore NaNs only at the original excluded EEG samples.
        for (v, &ex) in up.iter_mut().zip(isexcluded.iter()) {
            if ex {
                *v = f64::NAN;
            }
        }
        so_power_norm = up;
        so_power_times = eeg_times.to_vec();

        // Re-assign stages onto the EEG grid. pydynamo:160-164.
        if !stage_times.is_empty() && !stage_vals.is_empty() {
            let stages_up = interp_previous(stage_times, stage_vals, &so_power_times, f64::NAN);
            so_power_stages = stages_up
                .into_iter()
                .map(|v| if v.is_nan() { 0.0 } else { v })
                .collect();
        } else {
            so_power_stages = vec![1.0; so_power_times.len()];
        }
    }

    Ok(SoPowerOut {
        so_power_norm,
        so_power_times,
        so_power_stages,
        ptile: ptile_used,
    })
}

/// NaN-aware mean + sample std (ddof=1). Mirrors numpy `np.nanmean` and
/// `np.nanstd(ddof=1)` — same definitions pydynamo uses.
fn nan_mean_std(xs: &[f64]) -> (f64, f64) {
    let mut n = 0usize;
    let mut s = 0.0_f64;
    for &v in xs {
        if v.is_finite() {
            s += v;
            n += 1;
        }
    }
    if n == 0 {
        return (f64::NAN, f64::NAN);
    }
    let mean_ = s / n as f64;
    if n < 2 {
        return (mean_, f64::NAN);
    }
    let mut sq = 0.0_f64;
    for &v in xs {
        if v.is_finite() {
            let d = v - mean_;
            sq += d * d;
        }
    }
    let var_ = sq / (n - 1) as f64;
    (mean_, var_.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_norm_method() {
        let m = NormMethod::parse("p2shift1234").unwrap();
        match m {
            NormMethod::Shift { ptile, stages } => {
                assert_eq!(ptile, 2.0);
                assert_eq!(stages, vec![1, 2, 3, 4]);
            }
            _ => panic!(),
        }
        assert!(matches!(NormMethod::parse("percent"), Some(NormMethod::Percent)));
        assert!(matches!(NormMethod::parse("%SOP"), Some(NormMethod::Percent)));
        assert!(matches!(NormMethod::parse("none"), Some(NormMethod::None_)));
        assert!(NormMethod::parse("nonsense").is_none());
    }

    #[test]
    fn basic_pipeline_runs() {
        // Tiny synthetic: 2 freqs x 4 window centers, df = 1
        let n_freqs = 2;
        let n_times = 4;
        let so_spect: Vec<f64> = vec![
            // freq row 0
            1.0, 2.0, 3.0, 4.0,
            // freq row 1
            1.0, 2.0, 3.0, 4.0,
        ];
        let stimes: Vec<f64> = (0..n_times).map(|i| i as f64).collect();
        let sfreqs = vec![0.5, 1.5];
        let eeg_times: Vec<f64> = (0..100).map(|i| i as f64 * 0.1).collect();
        let isexcluded = vec![false; eeg_times.len()];
        let stage_times: Vec<f64> = vec![];
        let stage_vals: Vec<f64> = vec![];
        let out = so_power_from_spectrogram(
            &so_spect, n_freqs, n_times, &stimes, &sfreqs, &eeg_times,
            &isexcluded, &stage_times, &stage_vals,
            (0.0, 100.0), 3.0, &NormMethod::None_, false,
        ).unwrap();
        // With norm=none and no upsample, output length = n_times.
        assert_eq!(out.so_power_norm.len(), n_times);
        // Monotone increasing input → monotone increasing dB output.
        assert!(out.so_power_norm[0] < out.so_power_norm[3]);
    }

    #[test]
    fn shift_normalization_centers_on_percentile() {
        let n_freqs = 1;
        let n_times = 5;
        // Powers 1..5 → dB 0, 3.01, 4.77, 6.02, 6.99 (10*log10(1..5))
        let so_spect: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stimes: Vec<f64> = (0..n_times).map(|i| i as f64).collect();
        let sfreqs = vec![1.0];
        let eeg_times: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let isexcluded = vec![false; eeg_times.len()];
        // Staging covers the full query range so every SOpower window maps to stage=2.
        let stage_times = vec![0.0, 1000.0];
        let stage_vals = vec![2.0, 2.0];
        let out = so_power_from_spectrogram(
            &so_spect, n_freqs, n_times, &stimes, &sfreqs, &eeg_times,
            &isexcluded, &stage_times, &stage_vals,
            (0.0, 100.0), 10.0,
            &NormMethod::Shift { ptile: 50.0, stages: vec![2] },
            false,
        ).unwrap();
        // Median of 10*log10([1,2,3,4,5]) with df=1 should yield ~4.77 (10*log10(3))
        match out.ptile {
            Some(PtileUsed::Single(p)) => {
                assert!((p - (10.0 * 3.0_f64.log10())).abs() < 1e-10, "got ptile={}", p);
            }
            _ => panic!("expected single ptile"),
        }
        // After shift, index 2 (the median sample) should be ~0.
        assert!(out.so_power_norm[2].abs() < 1e-10);
    }

    #[test]
    fn p2shift1234_excludes_wake_and_unknown_from_percentile() {
        let n_freqs = 1;
        let n_times = 6;
        // Wake and Unknown have the two lowest powers. If either contributes
        // to the shift percentile, the percentile will be below 0 dB.
        let so_spect = vec![0.0001, 0.001, 1.0, 10.0, 100.0, 1000.0];
        let stimes: Vec<f64> = (0..n_times).map(|i| i as f64).collect();
        let sfreqs = vec![1.0];
        let eeg_times = stimes.clone();
        let isexcluded = vec![false; eeg_times.len()];
        let stage_times = stimes.clone();
        let stage_vals = vec![5.0, 0.0, 1.0, 2.0, 3.0, 4.0];
        let norm_method = NormMethod::parse("p2shift1234").unwrap();

        let out = so_power_from_spectrogram(
            &so_spect,
            n_freqs,
            n_times,
            &stimes,
            &sfreqs,
            &eeg_times,
            &isexcluded,
            &stage_times,
            &stage_vals,
            (0.0, 5.0),
            10.0,
            &norm_method,
            false,
        )
        .unwrap();

        assert_eq!(out.so_power_stages, stage_vals);
        match out.ptile {
            Some(PtileUsed::Single(p)) => {
                // Valid stages 1-4 have powers [1, 10, 100, 1000], or
                // [0, 10, 20, 30] dB. Their Hazen 2nd percentile is 0 dB.
                assert!(p.abs() < 1e-12, "got ptile={}", p);
            }
            _ => panic!("expected single ptile"),
        }
    }

    #[test]
    fn upsample_masks_excluded() {
        let n_freqs = 1;
        let n_times = 3;
        let so_spect = vec![1.0, 2.0, 3.0];
        let stimes = vec![0.0, 5.0, 10.0];
        let sfreqs = vec![1.0];
        let eeg_times: Vec<f64> = (0..=10).map(|i| i as f64).collect();
        let mut isexcluded = vec![false; eeg_times.len()];
        isexcluded[5] = true;
        let out = so_power_from_spectrogram(
            &so_spect, n_freqs, n_times, &stimes, &sfreqs, &eeg_times,
            &isexcluded, &[], &[],
            (0.0, 10.0), 10.0,
            &NormMethod::None_, true,
        ).unwrap();
        assert_eq!(out.so_power_norm.len(), eeg_times.len());
        assert!(out.so_power_norm[5].is_nan(), "excluded sample should be NaN");
        assert!(out.so_power_norm[0].is_finite());
    }

    #[test]
    fn upsample_bridges_coarse_nans() {
        let n_freqs = 1;
        let n_times = 3;
        let so_spect = vec![1.0, f64::NAN, 4.0];
        let stimes = vec![0.0, 5.0, 10.0];
        let sfreqs = vec![1.0];
        let eeg_times: Vec<f64> = (0..=10).map(|i| i as f64).collect();
        let isexcluded = vec![false; eeg_times.len()];

        let out = so_power_from_spectrogram(
            &so_spect, n_freqs, n_times, &stimes, &sfreqs, &eeg_times,
            &isexcluded, &[], &[],
            (0.0, 10.0), 10.0,
            &NormMethod::None_, true,
        ).unwrap();

        assert!(out.so_power_norm.iter().all(|v| v.is_finite()));
        let expected_midpoint = 10.0 * 2.0_f64.log10();
        assert!((out.so_power_norm[5] - expected_midpoint).abs() < 1e-12);
    }
}
