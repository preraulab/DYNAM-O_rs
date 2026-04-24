//! SO-phase time series.
//!
//! Port of `computeSOphase.m` (MATLAB) and `pydynamo.soph.sophase.compute_so_phase`.
//! SO-phase is returned **unwrapped** to match MATLAB convention — downstream
//! binning / per-peak assignment applies `wrapToPi` when needed.
//!
//! Pipeline:
//!   1. Band-pass filter EEG with SO-band SOS (typically 0.3–1.5 Hz) via
//!      `sosfiltfilt`.
//!   2. Compute Hilbert analytic signal → (real, imag).
//!   3. `atan2(imag, real)` → phase in [-π, π].
//!   4. Unwrap to remove 2π jumps.
//!   5. Set excluded samples to NaN in both `filtdata` and `so_phase`.
//!   6. Stage assignment via previous-neighbor interp (0 outside staging range).

use crate::peak_assign::interp_previous;
use crate::signal::{hilbert, sosfiltfilt, unwrap};

pub struct SoPhaseOut {
    /// Unwrapped phase (radians), length = eeg.len(). NaN at excluded samples.
    pub so_phase_unwrapped: Vec<f64>,
    /// EEG time grid (echo of input).
    pub so_phase_times: Vec<f64>,
    /// Stage per sample (0 outside staging range). Length = eeg.len().
    pub so_phase_stages: Vec<f64>,
    /// Filtered EEG — useful for downstream visualisation / QA. NaN at excluded samples.
    pub filtdata: Vec<f64>,
}

/// Compute SO-phase from raw EEG. Caller supplies the SOS filter (2D array
/// flattened row-major: shape (n_sections, 6) per scipy layout [b0 b1 b2 a0 a1 a2]).
///
/// To load MATLAB's canonical elliptic SOS for a given (fs, band), use
/// `filter_cache::get_sophase_sos` — same cache pydynamo uses.
pub fn so_phase_from_eeg(
    eeg: &[f64],
    eeg_times: &[f64],
    isexcluded: &[bool],
    sos: &[[f64; 6]],
    stage_times: &[f64],
    stage_vals: &[f64],
) -> Result<SoPhaseOut, String> {
    if eeg.len() != eeg_times.len() {
        return Err(format!(
            "eeg length {} != eeg_times length {}",
            eeg.len(),
            eeg_times.len()
        ));
    }
    if isexcluded.len() != eeg.len() {
        return Err(format!(
            "isexcluded length {} != eeg length {}",
            isexcluded.len(),
            eeg.len()
        ));
    }

    let filt = sosfiltfilt(sos, eeg);
    let (re, im) = hilbert(&filt);
    // atan2(imag, real) of the analytic signal — scipy hilbert returns
    // analytic = filt + i*H{filt}; numpy angle(analytic) = atan2(imag, real).
    let phase_wrapped: Vec<f64> = re.iter().zip(im.iter()).map(|(r, i)| i.atan2(*r)).collect();
    let phase_unwrapped = unwrap(&phase_wrapped, std::f64::consts::PI);

    // NaN mask at excluded samples (applied to both filtdata_out and phase).
    let mut filtdata_out: Vec<f64> = filt;
    let mut so_phase_out: Vec<f64> = phase_unwrapped;
    for (i, &ex) in isexcluded.iter().enumerate() {
        if ex {
            filtdata_out[i] = f64::NAN;
            so_phase_out[i] = f64::NAN;
        }
    }

    let so_phase_stages: Vec<f64> = if !stage_times.is_empty() && !stage_vals.is_empty() {
        interp_previous(stage_times, stage_vals, eeg_times, f64::NAN)
            .into_iter()
            .map(|v| if v.is_nan() { 0.0 } else { v })
            .collect()
    } else {
        vec![1.0; eeg.len()]
    };

    Ok(SoPhaseOut {
        so_phase_unwrapped: so_phase_out,
        so_phase_times: eeg_times.to_vec(),
        so_phase_stages,
        filtdata: filtdata_out,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// Tiny synthetic test — drive a single sine oscillation at a frequency
    /// inside the band, confirm unwrapped phase increases roughly linearly.
    #[test]
    fn pure_sine_gives_linear_phase() {
        let fs = 100.0;
        let f0 = 1.0; // 1 Hz (inside 0.3–1.5)
        let n = 2000usize;
        let eeg: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * f0 * (i as f64) / fs).sin())
            .collect();
        let eeg_times: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();
        let isexcluded = vec![false; n];

        // Use the filter cache (this will fall back to cheby1 if the SO band
        // elliptic isn't on disk in this test env — OK, we're testing the
        // pipeline shape, not the filter coefficients).
        let sos = crate::filter_cache::get_sophase_sos(fs, (0.3, 1.5))
            .expect("cache or cheby1 fallback should succeed");
        let sos_slice: Vec<[f64; 6]> = sos
            .rows()
            .into_iter()
            .map(|r| [r[0], r[1], r[2], r[3], r[4], r[5]])
            .collect();

        let out = so_phase_from_eeg(
            &eeg, &eeg_times, &isexcluded, &sos_slice, &[], &[],
        ).unwrap();

        assert_eq!(out.so_phase_unwrapped.len(), n);
        assert_eq!(out.filtdata.len(), n);
        // Sine at 1 Hz → phase increases by 2π/s → unwrapped phase slope ~2π
        // (sign depends on hilbert convention; check absolute slope).
        // Skip the edges where sosfiltfilt is ramping up.
        let mid = n / 2;
        let dphi = out.so_phase_unwrapped[mid + 100] - out.so_phase_unwrapped[mid];
        let dt = eeg_times[mid + 100] - eeg_times[mid];
        let rad_per_s = dphi / dt;
        // 1 Hz ⇒ 2π rad/s. With the cheby1 fallback SOS the slope is
        // within ~0.2 rad/s of the ideal; the MATLAB elliptic SOS is
        // closer to ideal. Tolerance excludes any adjacent integer Hz.
        assert!(
            (rad_per_s.abs() - 2.0 * PI).abs() < 0.3,
            "expected ~2π rad/s, got {}",
            rad_per_s
        );
    }

    #[test]
    fn excluded_samples_become_nan() {
        let fs = 100.0;
        let n = 200usize;
        let eeg: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
        let eeg_times: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();
        let mut isexcluded = vec![false; n];
        isexcluded[100] = true;

        let sos = crate::filter_cache::get_sophase_sos(fs, (0.3, 1.5)).unwrap();
        let sos_slice: Vec<[f64; 6]> = sos
            .rows()
            .into_iter()
            .map(|r| [r[0], r[1], r[2], r[3], r[4], r[5]])
            .collect();

        let out = so_phase_from_eeg(
            &eeg, &eeg_times, &isexcluded, &sos_slice, &[], &[],
        ).unwrap();
        assert!(out.so_phase_unwrapped[100].is_nan());
        assert!(out.filtdata[100].is_nan());
        assert!(out.so_phase_unwrapped[50].is_finite());
    }
}
