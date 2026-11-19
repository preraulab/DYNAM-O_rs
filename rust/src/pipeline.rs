//! Top-level DYNAM-O pipeline helpers (pure Rust).
//!
//! * [`run_extract_from_spectrogram`]: given a pre-computed spectrogram
//!   (baseline-normalized or raw), run the watershed-merge-trim extract
//!   pipeline and return `SegmentPeaks` plus the label image. Multitaper,
//!   baseline and refine are the caller's responsibility.
//! * [`default_extract_params`]: the `runDYNAMO` extraction defaults.
//! * [`write_stats_csv`]: the per-peak stats table writer.

use crate::extract_pipeline::{extract_tfpeaks, ExtractParams, SegmentPeaks};
use ndarray::{Array2, ArrayView1, ArrayView2};

/// Run extract_tfpeaks on a pre-computed spectrogram (baseline already
/// applied by the caller if desired). Returns (peaks, full-resolution label
/// image).
///
/// Thin wrapper over `extract_tfpeaks` with no baseline, no progress
/// callback, and no cancel check.
pub fn run_extract_from_spectrogram(
    spect: ArrayView2<f64>,
    stimes: ArrayView1<f64>,
    sfreqs: ArrayView1<f64>,
    params: &ExtractParams,
) -> Result<(SegmentPeaks, Array2<i64>), String> {
    extract_tfpeaks(spect, stimes, sfreqs, None, params, None, None)
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

/// Significant-figure precision per stats-CSV column, resolved + validated
/// against the run's actual `fs` so the caller can never accidentally
/// serialize a too-coarse value. Build via [`StatsCsvPrecision::resolve`],
/// which honors the `OutputOpts` knobs and bumps any user override below
/// its safe minimum with a `log::warn!` per adjustment.
///
/// Every column uses significant figures (see OUTPUT_FORMAT.md §6.2).
/// For time columns specifically, the floor is `fs`-derived so a long
/// recording at high `fs` still resolves sub-sample.
#[derive(Debug, Clone, Copy)]
pub struct StatsCsvPrecision {
    /// Significant figures for time-class columns: `PeakTime`,
    /// `Duration`, `bbox_tl_s`.
    pub time_sigfigs: u8,
    /// Significant figures for frequency-class columns: `PeakFrequency`,
    /// `Bandwidth`, `bbox_tl_Hz`.
    pub freq_sigfigs: u8,
    /// Significant figures for the `SOphase` column (radians).
    pub phase_sigfigs: u8,
    /// Significant figures for the `Height` column.
    pub height_sigfigs: u8,
    /// Significant figures for the `Volume` column.
    pub volume_sigfigs: u8,
    /// Significant figures for the `Area` column.
    pub area_sigfigs: u8,
    /// Significant figures for the `Peakiness` column.
    pub peakiness_sigfigs: u8,
    /// Significant figures for the `SOpower` column.
    pub sopower_sigfigs: u8,
}

/// Lowest allowed significant figures for the frequency and phase columns.
const MIN_FREQ_PHASE_SIGFIGS: u8 = 4;
/// Lowest allowed significant figures for the variable-magnitude columns
/// (height, volume, area, peakiness, SO-power).
const MIN_MAGNITUDE_SIGFIGS: u8 = 3;

impl StatsCsvPrecision {
    /// Build the effective precision for a run. **Never returns a value
    /// that would lose information**: time has an fs-derived floor
    /// (`time_sigfigs_for_fs(fs)`). Frequency and phase have a flat floor
    /// of 4 sig figs (~5e-5 relative, orders below any physical noise), and
    /// height, volume, area, peakiness and SO-power have a floor of 3.
    /// A user override below its floor is bumped up with a `log::warn!`.
    /// The default path (no overrides at `fs ≤ 100 Hz`) emits no warnings.
    pub fn resolve(
        time_sigfigs: Option<u8>,
        freq_sigfigs: u8,
        phase_sigfigs: u8,
        height_sigfigs: u8,
        volume_sigfigs: u8,
        area_sigfigs: u8,
        peakiness_sigfigs: u8,
        sopower_sigfigs: u8,
        fs: f64,
    ) -> Self {
        let time_min = time_sigfigs_for_fs(fs);
        let time_sigfigs = match time_sigfigs {
            None => time_min,
            Some(n) if n < time_min => {
                log::warn!(
                    "csv_time_sigfigs={} below fs={} minimum {} (would lose sub-sample \
                     precision for a typical recording length); using {} instead",
                    n, fs, time_min, time_min
                );
                time_min
            }
            Some(n) => n,
        };
        let bump = |name: &str, value: u8, min: u8| -> u8 {
            if value < min {
                log::warn!(
                    "{}={} below safe minimum {} (would lose sub-bin / sub-precision \
                     information); using {} instead",
                    name, value, min, min
                );
                min
            } else {
                value
            }
        };
        Self {
            time_sigfigs,
            freq_sigfigs: bump("csv_freq_sigfigs", freq_sigfigs, MIN_FREQ_PHASE_SIGFIGS),
            phase_sigfigs: bump("csv_phase_sigfigs", phase_sigfigs, MIN_FREQ_PHASE_SIGFIGS),
            height_sigfigs: bump("csv_height_sigfigs", height_sigfigs, MIN_MAGNITUDE_SIGFIGS),
            volume_sigfigs: bump("csv_volume_sigfigs", volume_sigfigs, MIN_MAGNITUDE_SIGFIGS),
            area_sigfigs: bump("csv_area_sigfigs", area_sigfigs, MIN_MAGNITUDE_SIGFIGS),
            peakiness_sigfigs: bump("csv_peakiness_sigfigs", peakiness_sigfigs, MIN_MAGNITUDE_SIGFIGS),
            sopower_sigfigs: bump("csv_sopower_sigfigs", sopower_sigfigs, MIN_MAGNITUDE_SIGFIGS),
        }
    }
}

impl Default for StatsCsvPrecision {
    /// Settings-side defaults assuming `fs = 100 Hz` (the same values as
    /// `dynamo_pipeline::OutputOpts`). Used by the
    /// bare CLI bin and tests that don't carry an `OutputOpts` through.
    fn default() -> Self {
        Self {
            time_sigfigs: 8,
            freq_sigfigs: 6,
            phase_sigfigs: 6,
            height_sigfigs: 6,
            volume_sigfigs: 6,
            area_sigfigs: 6,
            peakiness_sigfigs: 6,
            sopower_sigfigs: 6,
        }
    }
}

/// Stats-CSV schema version written by this crate (OUTPUT_FORMAT.md §8.2):
/// format 3 = the 14-column layout below plus the `#` provenance preamble.
/// Format 2 is the same columns with no preamble. Format 1 is the
/// 16-column layout that also carries `bbox_width_s`/`bbox_height_Hz`.
pub const STATS_CSV_FORMAT: u32 = 3;

/// Caller-supplied half of the stats-CSV provenance stamp
/// (OUTPUT_FORMAT.md §8.1). The kernel-supplied half (`format`,
/// `kernel_version`) is filled in by [`write_stats_csv`] itself so it
/// always reflects the compiled kernel and cannot be spoofed by a stale
/// caller.
pub struct StatsCsvProvenance<'a> {
    /// Tool id: `dynamo-app`, `dynamo-cli`, `dynamo-matlab`, `pydynamo`,
    /// or `dynamo-rs-bin`.
    pub writer: &'a str,
    /// The writing tool's build, grammar `<semver>+<sha12>[.dirty]`.
    pub writer_version: &'a str,
    /// Emitted as `# subjectID:` when non-empty.
    pub subject_id: Option<&'a str>,
    /// Computation-semantics version (`# semantics:` line), emitted when
    /// `Some`. Same number across builds ⇒ comparable numbers.
    pub semantics: Option<u32>,
}

/// Write a `SegmentPeaks` to a CSV file, with an optional `#` provenance
/// preamble (see [`StatsCsvProvenance`]). `None` gives the bare format-2
/// layout, with identical data rows either way. Column layout mirrors the
/// MATLAB `stats_table` (PeakTime, PeakFrequency, Duration, Bandwidth,
/// Height, Volume, SegmentNum, Area, Peakiness, bbox_tl_s, bbox_tl_Hz).
/// Peakiness is N/(N−1)·(max − mean)/(max − min) of the region pixels,
/// unitless in [0, 1]. Two parts of MATLAB's set are left out. The bbox
/// extent columns `bbox_width_s` / `bbox_height_Hz` have no column because
/// they are identical to `Duration` / `Bandwidth` (same `n_pixels * bin
/// size`), so only the top-left corner of the bounding box is written. The
/// variable-length columns (HeightData, Boundaries) are not written either.
///
/// Three trailing per-peak columns are appended beyond MATLAB's base
/// `stats_table` so downstream views (the TF-peak scatter) are
/// self-contained. They mirror exactly what MATLAB `runDYNAMO` appends
/// in its "additional peak feature properties" block, with the same
/// column names and order:
///   * `PeakStage`: `computePeakStage`, interp1(stage_times,
///     stage_vals, PeakTime, 'previous'). 0 = Unknown, 6 = Artifact.
///   * `SOpower`: `computePeakSOpower`, SO-power at PeakTime.
///   * `SOphase`: `computePeakSOphase`, SO-phase at PeakTime, rad.
/// Callers without staging (the bare extract CLI) pass empty slices.
/// Those rows then carry stage `0` and `NaN` SO-power/phase.
pub fn write_stats_csv(
    peaks: &SegmentPeaks,
    peak_sopower: &[f64],
    peak_sophase: &[f64],
    peak_stage: &[u8],
    prec: StatsCsvPrecision,
    prov: Option<&StatsCsvProvenance>,
    path: &std::path::Path,
) -> std::io::Result<()> {
    use std::io::Write;
    let file = std::fs::File::create(path)?;
    let mut writer = std::io::BufWriter::new(file);
    // Provenance preamble (format 3). `None` emits the bare format-2 file
    // (identical data rows, no `#` lines) for callers that manage their own
    // provenance or need the format-2 shape.
    if let Some(provenance) = prov {
        writeln!(writer, "# DYNAM-O stats table")?;
        writeln!(writer, "# format: {}", STATS_CSV_FORMAT)?;
        writeln!(writer, "# writer: {}", provenance.writer)?;
        writeln!(writer, "# writer_version: {}", provenance.writer_version)?;
        writeln!(writer, "# kernel_version: {}", crate::build_info::VERSION)?;
        if let Some(semantics) = provenance.semantics {
            writeln!(writer, "# semantics: {}", semantics)?;
        }
        if let Some(subject_id) = provenance.subject_id {
            if !subject_id.is_empty() {
                writeln!(writer, "# subjectID: {}", subject_id)?;
            }
        }
    }
    writeln!(
        writer,
        "PeakTime,PeakFrequency,Duration,Bandwidth,Height,Volume,SegmentNum,Area,Peakiness,bbox_tl_s,bbox_tl_Hz,PeakStage,SOpower,SOphase"
    )?;
    // Every float column rounds to its column-class significant figures.
    // `Display` then picks the shortest round-trippable text so
    // `1.000000000003638 → 1`. Integer columns stay untouched. NaN / ±∞
    // pass through.
    let round_time = |v: f64| round_sig(v, prec.time_sigfigs);
    let round_freq = |v: f64| round_sig(v, prec.freq_sigfigs);
    let round_phase = |v: f64| round_sig(v, prec.phase_sigfigs);
    for i in 0..peaks.len() {
        // Only the bounding-box top-left corner is written. The bbox extent
        // (width_s, height_Hz) equals Duration / Bandwidth (all are
        // n_pixels * bin size, see extract_pipeline), so it has no column
        // of its own.
        let bbox_time_tl = peaks.bbox[i * 4];
        let bbox_freq_tl = peaks.bbox[i * 4 + 1];
        writeln!(
            writer,
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            round_time(peaks.peak_time[i]),
            round_freq(peaks.peak_freq[i]),
            round_time(peaks.duration[i]),
            round_freq(peaks.bandwidth[i]),
            round_sig(peaks.height[i], prec.height_sigfigs),
            round_sig(peaks.volume[i], prec.volume_sigfigs),
            peaks.segment_num[i],
            round_sig(peaks.area[i], prec.area_sigfigs),
            round_sig(peaks.peakiness[i], prec.peakiness_sigfigs),
            round_time(bbox_time_tl),
            round_freq(bbox_freq_tl),
            peak_stage.get(i).copied().unwrap_or(0),
            round_sig(peak_sopower.get(i).copied().unwrap_or(f64::NAN), prec.sopower_sigfigs),
            round_phase(peak_sophase.get(i).copied().unwrap_or(f64::NAN)),
        )?;
    }
    Ok(())
}

/// Minimum significant figures the time columns need at sample rate `fs` so
/// the round-trip absolute precision is still finer than the sample period
/// even for a long recording. Derived as `ceil(log10 fs) + 6` (covers
/// PeakTime up to ~1e5 s ≈ 28 h with ~½-sample headroom), clamped to
/// `[6, 12]`. `fs=10 → 7`, `fs=100 → 8`, `fs=1000 → 9`, `fs=10000 → 10`.
pub fn time_sigfigs_for_fs(fs: f64) -> u8 {
    // Invalid fs: fall back to the value for the common 100 Hz case.
    const FALLBACK_TIME_SIGFIGS: u8 = 8;
    if !fs.is_finite() || fs <= 0.0 {
        return FALLBACK_TIME_SIGFIGS;
    }
    let fs_decades = fs.log10().ceil() as i32;
    (fs_decades + 6).clamp(6, 12) as u8
}

/// Round an `f64` to `n` significant figures (analogue of C `printf "%.Ng"`).
/// `1.000000000003638 → 1` at n=6, `0.357365679560… → 0.357366` at n=6,
/// `4321.39241260… → 4321.39` at n=6. `0` / `NaN` / `±∞` pass through.
fn round_sig(v: f64, n: u8) -> f64 {
    if v == 0.0 || !v.is_finite() {
        return v;
    }
    let leading_exponent = v.abs().log10().floor() as i32;
    // n sig figs keeps (n - 1) digits after the leading digit.
    let scale = 10f64.powi((n as i32 - 1) - leading_exponent);
    (v * scale).round() / scale
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

    fn two_peak_fixture() -> (crate::extract_pipeline::SegmentPeaks, Vec<f64>, Vec<f64>, Vec<u8>) {
        let mut p = crate::extract_pipeline::SegmentPeaks::default();
        for (t, f) in [(12.5, 11.0), (300.0, 13.5)] {
            p.peak_time.push(t);
            p.peak_freq.push(f);
            p.duration.push(1.25);
            p.bandwidth.push(2.0);
            p.height.push(3.5);
            p.volume.push(8.75);
            p.segment_num.push(1.0);
            p.bbox.extend_from_slice(&[t - 0.5, f - 1.0, 1.25, 2.0]);
            p.area.push(2.5);
            p.peakiness.push(0.0);
            p.height_data.push(vec![]);
            p.boundaries_xy.push(vec![]);
        }
        (p, vec![3.0, -1.5], vec![0.5, -2.0], vec![1, 2])
    }

    /// Provenance is preamble-only: `prov: Some` vs `None` on identical
    /// peaks must produce byte-identical files once leading `#` lines are
    /// dropped, and the format-3 preamble must carry exactly the frozen
    /// stamp keys (OUTPUT_FORMAT.md §8.1).
    #[test]
    fn stats_csv_provenance_is_preamble_only() {
        let (peaks, sopow, soph, stage) = two_peak_fixture();
        let dir = std::env::temp_dir().join(format!("dynamo_stats_prov_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let bare = dir.join("bare.csv");
        let stamped = dir.join("stamped.csv");
        let prec = StatsCsvPrecision::default();
        write_stats_csv(&peaks, &sopow, &soph, &stage, prec, None, &bare).unwrap();
        let prov = StatsCsvProvenance {
            writer: "dynamo-rs-bin",
            writer_version: crate::build_info::VERSION,
            subject_id: Some("S001"),
            semantics: Some(4),
        };
        write_stats_csv(&peaks, &sopow, &soph, &stage, prec, Some(&prov), &stamped).unwrap();

        let bare_txt = std::fs::read_to_string(&bare).unwrap();
        let stamped_txt = std::fs::read_to_string(&stamped).unwrap();
        assert!(!bare_txt.starts_with('#'), "bare file must have no preamble");
        let preamble: Vec<&str> = stamped_txt.lines().take_while(|l| l.starts_with('#')).collect();
        let stripped: String = stamped_txt
            .lines()
            .skip_while(|l| l.starts_with('#'))
            .map(|l| format!("{l}\n"))
            .collect();
        assert_eq!(stripped, bare_txt, "data section must be byte-identical");

        assert_eq!(preamble[0], "# DYNAM-O stats table");
        assert_eq!(preamble[1], format!("# format: {STATS_CSV_FORMAT}"));
        assert_eq!(preamble[2], "# writer: dynamo-rs-bin");
        assert_eq!(preamble[3], format!("# writer_version: {}", crate::build_info::VERSION));
        assert_eq!(preamble[4], format!("# kernel_version: {}", crate::build_info::VERSION));
        assert_eq!(preamble[5], "# semantics: 4");
        assert_eq!(preamble[6], "# subjectID: S001");
        assert_eq!(preamble.len(), 7);
        std::fs::remove_dir_all(&dir).ok();
    }
}
