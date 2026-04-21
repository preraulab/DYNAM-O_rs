//! Multitaper spectrogram wrapper — DYNAM-O-flavored.
//!
//! Port of `pydynamo.spectrogram.mtm_spectrogram`. Loads DPSS tapers from
//! disk cache, then delegates to `multitaper_rs::compute_spectrogram`.

use ndarray::{Array1, Array2, ArrayView1};

use multitaper_rs::{compute_spectrogram, DetrendMode, SpectrogramParams, Weighting};

use super::dpss;

/// Smallest power of 2 >= x.
pub fn next_pow2(x: f64) -> usize {
    1usize << ((x.max(1.0)).log2().ceil() as usize)
}

/// MATLAB-style multitaper spectrogram.
///
/// Parameters match pydynamo's `mtm_spectrogram` defaults:
///   - taper_params = (time_bandwidth, num_tapers)
///   - window_params = (window_s, step_s)
///   - dsfreqs = frequency resolution target (0.1 Hz by default)
pub fn mtm_spectrogram(
    data: ArrayView1<f64>,
    fs: f64,
    freq_range: (f64, f64),
    taper_params: (f64, usize),
    window_params: (f64, f64),
    dsfreqs: f64,
    detrend_opt: &str,
) -> Result<(Array2<f64>, Array1<f64>, Array1<f64>), String> {
    let winsize = (window_params.0 * fs).round() as usize;
    let nfft = next_pow2(fs / dsfreqs);

    let tapers = dpss::get_tapers(winsize, taper_params.0, taper_params.1)?;

    let detrend = DetrendMode::parse(detrend_opt)?;
    let params = SpectrogramParams {
        fs,
        frequency_range: freq_range,
        window_params,
        nfft,
        detrend,
        weighting: Weighting::Unity,
    };

    let out = compute_spectrogram(data, tapers.view(), None, &params)?;
    Ok((out.mt_spectrogram, out.stimes, out.sfreqs))
}
