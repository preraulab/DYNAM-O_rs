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

/// Write a `SegmentPeaks` to a CSV file, with an optional `#` provenance
/// preamble (see [`StatsCsvProvenance`]; `None` → bare format-2 layout,
/// data rows identical either way). Column layout mirrors the MATLAB
/// `stats_table` (PeakTime, PeakFrequency, Duration, Bandwidth, Height,
/// Volume, SegmentNum, Area, Peakiness — where Peakiness =
/// N/(N−1)·(max − mean)/(max − min) of region pixels, unitless in
/// [0, 1] — bbox_tl_s, bbox_tl_Hz), with two
/// deliberate omissions from MATLAB's set: the bbox extent columns
/// `bbox_width_s` / `bbox_height_Hz` are dropped because they are identical
/// to `Duration` / `Bandwidth` (same `n_pixels * bin size`), so only the
/// top-left corner of the bounding box is kept. Variable-length columns
/// (HeightData, Boundaries) are also omitted from CSV.
///
/// Three trailing per-peak columns are appended beyond MATLAB's base
/// `stats_table` so downstream views (the TF-peak scatter) are
/// self-contained. They mirror exactly what MATLAB `runDYNAMO` appends
/// in its "additional peak feature properties" block, with the same
/// column names and order:
///   * `PeakStage` — `computePeakStage`: interp1(stage_times,
///     stage_vals, PeakTime, 'previous'); 0 = Unknown, 6 = Artifact.
///   * `SOpower`   — `computePeakSOpower`: SO-power at PeakTime.
///   * `SOphase`   — `computePeakSOphase`: SO-phase at PeakTime, rad.
/// Callers without staging (the bare extract CLI) pass empty slices —
/// those rows then carry stage `0` and `NaN` SO-power/phase.
/// Significant-figure precision per stats-CSV column, resolved + validated
/// against the run's actual `fs` so the caller can never accidentally
/// serialise a too-coarse value. Build via [`StatsCsvPrecision::resolve`],
/// which honours the `OutputOpts` knobs and bumps any user override below
/// its safe minimum with a `log::warn!` per adjustment.
///
/// Every column uses significant figures (decimals-after-the-dot was the
/// previous mix; the unification is documented in OUTPUT_FORMAT.md §6.2).
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
    /// Significant figures for each variable-magnitude "other" column.
    pub height_sigfigs: u8,
    pub volume_sigfigs: u8,
    pub area_sigfigs: u8,
    pub peakiness_sigfigs: u8,
    pub sopower_sigfigs: u8,
}

impl StatsCsvPrecision {
    /// Build the effective precision for a run. **Never returns a value
    /// that would lose information**: time has an fs-derived floor
    /// (`time_sigfigs_for_fs(fs)`); every other knob has a flat floor of
    /// 4 sig figs (~5e-5 relative — orders below any physical noise).
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
            freq_sigfigs: bump("csv_freq_sigfigs", freq_sigfigs, 4),
            phase_sigfigs: bump("csv_phase_sigfigs", phase_sigfigs, 4),
            height_sigfigs: bump("csv_height_sigfigs", height_sigfigs, 3),
            volume_sigfigs: bump("csv_volume_sigfigs", volume_sigfigs, 3),
            area_sigfigs: bump("csv_area_sigfigs", area_sigfigs, 3),
            peakiness_sigfigs: bump("csv_peakiness_sigfigs", peakiness_sigfigs, 3),
            sopower_sigfigs: bump("csv_sopower_sigfigs", sopower_sigfigs, 3),
        }
    }
}

impl Default for StatsCsvPrecision {
    /// Settings-side defaults assuming `fs = 100 Hz` (matches the
    /// matching defaults in `dynamo_pipeline::OutputOpts`). Used by the
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
/// Format 2 was the same columns bare; format 1 was the 16-column layout
/// with the redundant `bbox_width_s`/`bbox_height_Hz`.
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
}

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
    let f = std::fs::File::create(path)?;
    let mut w = std::io::BufWriter::new(f);
    // Provenance preamble (format 3). `None` emits the bare format-2 file —
    // identical data rows, no `#` lines — for callers that manage their own
    // provenance or need the legacy shape.
    if let Some(p) = prov {
        writeln!(w, "# DYNAM-O stats table")?;
        writeln!(w, "# format: {}", STATS_CSV_FORMAT)?;
        writeln!(w, "# writer: {}", p.writer)?;
        writeln!(w, "# writer_version: {}", p.writer_version)?;
        writeln!(w, "# kernel_version: {}", crate::build_info::VERSION)?;
        if let Some(sid) = p.subject_id {
            if !sid.is_empty() {
                writeln!(w, "# subjectID: {}", sid)?;
            }
        }
    }
    writeln!(
        w,
        "PeakTime,PeakFrequency,Duration,Bandwidth,Height,Volume,SegmentNum,Area,Peakiness,bbox_tl_s,bbox_tl_Hz,PeakStage,SOpower,SOphase"
    )?;
    // Every float column rounds to its column-class significant figures.
    // `Display` then picks the shortest round-trippable text so
    // `1.000000000003638 → 1`. Integer columns stay untouched; NaN / ±∞
    // pass through.
    let t  = |v: f64| round_sig(v, prec.time_sigfigs);
    let fz = |v: f64| round_sig(v, prec.freq_sigfigs);
    let ph = |v: f64| round_sig(v, prec.phase_sigfigs);
    for i in 0..peaks.len() {
        // Only the bounding-box top-left corner is emitted. The bbox extent
        // (width_s, height_Hz) is byte-for-byte identical to Duration /
        // Bandwidth (all = n_pixels * bin size; see extract_pipeline), so the
        // two extent columns were pure duplication and are dropped.
        let b0 = peaks.bbox[i * 4];
        let b1 = peaks.bbox[i * 4 + 1];
        writeln!(
            w,
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            t(peaks.peak_time[i]),
            fz(peaks.peak_freq[i]),
            t(peaks.duration[i]),
            fz(peaks.bandwidth[i]),
            round_sig(peaks.height[i], prec.height_sigfigs),
            round_sig(peaks.volume[i], prec.volume_sigfigs),
            peaks.segment_num[i],
            round_sig(peaks.area[i], prec.area_sigfigs),
            round_sig(peaks.peakiness[i], prec.peakiness_sigfigs),
            t(b0), fz(b1),
            peak_stage.get(i).copied().unwrap_or(0),
            round_sig(peak_sopower.get(i).copied().unwrap_or(f64::NAN), prec.sopower_sigfigs),
            ph(peak_sophase.get(i).copied().unwrap_or(f64::NAN)),
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
    if !fs.is_finite() || fs <= 0.0 {
        return 8;
    }
    let log = fs.log10().ceil() as i32;
    (log + 6).clamp(6, 12) as u8
}

/// Round an `f64` to `n` significant figures (analogue of C `printf "%.Ng"`).
/// `1.000000000003638 → 1` at n=6; `0.357365679560… → 0.357366` at n=6;
/// `4321.39241260… → 4321.39` at n=6. `0` / `NaN` / `±∞` pass through.
fn round_sig(v: f64, n: u8) -> f64 {
    if v == 0.0 || !v.is_finite() {
        return v;
    }
    let m = v.abs().log10().floor() as i32;
    // n sig figs → (n-1) trailing digits relative to leading.
    let p = 10f64.powi((n as i32 - 1) - m);
    (v * p).round() / p
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
        assert_eq!(preamble[5], "# subjectID: S001");
        assert_eq!(preamble.len(), 6);
        std::fs::remove_dir_all(&dir).ok();
    }
}
