//! Top-level DYNAM-O pipeline orchestrators (pure Rust).
//!
//! Three entry points, in increasing scope:
//!
//! 1. [`run_extract_from_spectrogram`] — shortest path: given a pre-computed
//!    spectrogram (baseline-normalized or raw), run the watershed-merge-trim
//!    extract pipeline and return `SegmentPeaks` + label image. This is
//!    what the CLI binary wires up today. Multitaper + baseline + refine
//!    are the caller's responsibility.
//!
//! 2. `run_extract_two_pass` (TODO) — chain pass-1 baseline + extract, then
//!    pass-2 with mask_spectrogram from pass-1 labels.
//!
//! 3. `run_dynamo` (TODO) — full pipeline: EDF → artifacts → multitaper ×2 →
//!    baseline ×2 → extract ×2 → refine → peak_stage → SO-power/phase →
//!    histograms. Blocked on DPSS source (Part 5 step 7 note: tapers are
//!    caller-supplied until we embed DPSS fixtures or a Slepian solver).
//!
//! The plumbing for all three is already in separate modules — this file
//! just assembles them.

use crate::extract_pipeline::{extract_tfpeaks, ExtractParams, SegmentPeaks};
use ndarray::{Array2, ArrayView1, ArrayView2};

/// Run extract_tfpeaks on a pre-computed spectrogram (baseline already
/// applied by the caller if desired). Returns (peaks, full-resolution label
/// image).
///
/// This wraps `extract_tfpeaks` with a named-output struct for clarity;
/// the underlying function also takes an optional per-segment progress
/// callback, omitted here.
pub fn run_extract_from_spectrogram(
    spect: ArrayView2<f64>,
    stimes: ArrayView1<f64>,
    sfreqs: ArrayView1<f64>,
    params: &ExtractParams,
) -> Result<(SegmentPeaks, Array2<i64>), String> {
    extract_tfpeaks(spect, stimes, sfreqs, None, params, None)
}

/// Default `ExtractParams` matching `runDYNAMO` defaults:
/// seg_time=30 s, downsample [2,2], merge_thresh=11, trim_vol=0.8,
/// dur_min=0.5 s, dur_max=5 s, bw_min=2 Hz, bw_max=15 Hz, ht_db_min=-inf,
/// expand_labels_distance=0 (MATLAB-paint, default).
pub fn default_extract_params() -> ExtractParams {
    ExtractParams {
        seg_time: 30.0,
        downsample_f: 2,
        downsample_t: 2,
        merge_thresh: 11.0,
        max_merges: f64::INFINITY,
        trim_vol_thresh: 0.8,
        trim_shift_val: f64::NAN, // per-segment min
        dur_min: 0.5,
        dur_max: 5.0,
        bw_min: 2.0,
        bw_max: 15.0,
        freq_min: 0.0,
        freq_max: f64::INFINITY,
        ht_db_min: f64::NEG_INFINITY,
        expand_labels_distance: 0, // MATLAB paint
    }
}

/// Write a `SegmentPeaks` to a CSV file in the same column layout as the
/// MATLAB `stats_table` (headers: PeakTime, PeakFrequency, Duration,
/// Bandwidth, Height, Volume, SegmentNum, Area, Peakiness — where
/// Peakiness = 10*log10(Area·Height/Volume), in dB — bbox_tl_s, bbox_tl_Hz,
/// bbox_width_s, bbox_height_Hz). Variable-length columns (HeightData,
/// Boundaries) are intentionally omitted from CSV.
pub fn write_stats_csv(peaks: &SegmentPeaks, path: &std::path::Path) -> std::io::Result<()> {
    use std::io::Write;
    let f = std::fs::File::create(path)?;
    let mut w = std::io::BufWriter::new(f);
    writeln!(
        w,
        "PeakTime,PeakFrequency,Duration,Bandwidth,Height,Volume,SegmentNum,Area,Peakiness,bbox_tl_s,bbox_tl_Hz,bbox_width_s,bbox_height_Hz"
    )?;
    for i in 0..peaks.len() {
        let b0 = peaks.bbox[i * 4];
        let b1 = peaks.bbox[i * 4 + 1];
        let b2 = peaks.bbox[i * 4 + 2];
        let b3 = peaks.bbox[i * 4 + 3];
        writeln!(
            w,
            "{},{},{},{},{},{},{},{},{},{},{},{},{}",
            peaks.peak_time[i],
            peaks.peak_freq[i],
            peaks.duration[i],
            peaks.bandwidth[i],
            peaks.height[i],
            peaks.volume[i],
            peaks.segment_num[i],
            peaks.area[i],
            peaks.peakiness[i],
            b0, b1, b2, b3,
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn default_params_round_trip() {
        let p = default_extract_params();
        assert_eq!(p.seg_time, 30.0);
        assert_eq!(p.downsample_f, 2);
        assert_eq!(p.expand_labels_distance, 0);
    }

    #[test]
    fn run_extract_on_tiny_spect_is_empty() {
        let spect = Array2::<f64>::zeros((10, 10));
        let stimes = ndarray::Array1::linspace(0.0, 9.0, 10);
        let sfreqs = ndarray::Array1::linspace(0.0, 9.0, 10);
        let p = default_extract_params();
        let (peaks, labels) =
            run_extract_from_spectrogram(spect.view(), stimes.view(), sfreqs.view(), &p).unwrap();
        assert_eq!(peaks.len(), 0);
        assert_eq!(labels.dim(), (10, 10));
    }
}
