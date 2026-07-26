//! Multitaper spectrogram wrapper — thin re-export of `multitaper_rs` with
//! DYNAM-O defaults + named-return convenience.
//!
//! Delegates to the `multitaper_rs` crate from `preraulab/multitaper_toolbox`
//! (path-dep sibling under DYNAM-O_toolbox). That crate implements the core
//! compute loop — per-window detrend, taper multiply, zero-padded rFFT,
//! weighted power average, one-sided PSD scaling — and is bit-identical
//! (f64 round-off) to the same toolbox's pure-Python implementation.
//!
//! DPSS tapers are **not** generated here; `multitaper_rs` requires them
//! as inputs. Callers compute `scipy.signal.windows.dpss` (Python) or
//! `dpss` (MATLAB) once and pass the tapers + eigenvalues in.
//!
//! Why thin: `multitaper_rs` already has the proper API. No re-wrapping
//! value added except naming consistency with the rest of `dynamo_rs`
//! and pre-loaded defaults.

pub use multitaper_rs::{
    compute_spectrogram, DetrendMode, SpectrogramOutput, SpectrogramParams, Weighting,
};

/// DYNAM-O pass-1 defaults: 1 s window, 50 ms step, freq 0–30 Hz, constant
/// detrend, unity weighting. nfft is **not** defaulted — caller picks
/// (typically next-pow2 of `winsize_samples` or a padded multiple).
pub fn dynamo_pass1_params(fs: f64, nfft: usize) -> SpectrogramParams {
    SpectrogramParams {
        fs,
        frequency_range: (0.0, 30.0),
        window_params: (1.0, 0.05),
        nfft,
        detrend: DetrendMode::Constant,
        weighting: Weighting::Unity,
    }
}

/// DYNAM-O pass-2 defaults: 2 s window, 50 ms step (same freq range / detrend).
pub fn dynamo_pass2_params(fs: f64, nfft: usize) -> SpectrogramParams {
    SpectrogramParams {
        fs,
        frequency_range: (0.0, 30.0),
        window_params: (2.0, 0.05),
        nfft,
        detrend: DetrendMode::Constant,
        weighting: Weighting::Unity,
    }
}

/// SO-power defaults: 5 s window, 0.5 s step, 0.3–1.5 Hz (SO band), linear
/// detrend, unity weighting. Used in `so_power` pipeline.
pub fn dynamo_sopower_params(fs: f64, nfft: usize) -> SpectrogramParams {
    SpectrogramParams {
        fs,
        frequency_range: (0.3, 1.5),
        window_params: (5.0, 0.5),
        nfft,
        detrend: DetrendMode::Linear,
        weighting: Weighting::Unity,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tf_passes_use_constant_detrending() {
        assert!(matches!(
            dynamo_pass1_params(100.0, 1024).detrend,
            DetrendMode::Constant
        ));
        assert!(matches!(
            dynamo_pass2_params(100.0, 1024).detrend,
            DetrendMode::Constant
        ));
    }

    #[test]
    fn so_power_keeps_linear_detrending() {
        assert!(matches!(
            dynamo_sopower_params(100.0, 1024).detrend,
            DetrendMode::Linear
        ));
    }
}
