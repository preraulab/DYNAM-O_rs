//! End-to-end DYNAM-O orchestrators that compose `dynamo_rs` stages.
//!
//! Match against MATLAB `computeTFPeaks.m` (two-pass extract) and
//! `runDYNAMO.m` (full pipeline).

use crate::run::settings::DynamoSettings;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use std::sync::atomic::{AtomicU32, Ordering};

use crate::artifacts::{detect_artifacts_with_slope_test, ArtifactOpts, ZScoreMethod};
use crate::baseline::{build_baseline_exclude, compute_baseline};
use crate::extract_pipeline::{extract_tfpeaks, ExtractParams, SegmentPeaks};
use crate::mask::mask_spectrogram;
use crate::refine::{hann_event_spectra, refine_from_spectra, DetrendOpt};

// -------- Public types --------

/// Multitaper spectrogram parameters used to compute one pass's spectrogram.
/// The orchestrator needs them to derive `dur_min`, `bw_min`, and
/// `ht_db_min` filter thresholds — the spectrogram itself is supplied by the
/// caller. Matches the math in `computeTFPeaks.m::computeSpectrogram`.
#[derive(Debug, Clone, Copy)]
pub struct PassMtmSpec {
    /// DPSS time-half-bandwidth product (typically 2.0).
    pub nw: f64,
    /// Number of tapers (typically 3).
    pub n_tapers: usize,
    /// Window length in seconds (pass-1 = 1.0, pass-2 = 2.0 default).
    pub window_length: f64,
}

impl PassMtmSpec {
    pub fn dynamo_pass1() -> Self { Self { nw: 2.0, n_tapers: 3, window_length: 1.0 } }
    pub fn dynamo_pass2() -> Self { Self { nw: 2.0, n_tapers: 3, window_length: 2.0 } }

    /// Per-pass duration floor: `window_length / 2`.
    fn dur_min(&self) -> f64 { self.window_length / 2.0 }
    /// Per-pass bandwidth floor: `df / 2` where `df = 2·NW / window_length`.
    fn bw_min(&self) -> f64 { (2.0 * self.nw / self.window_length) / 2.0 }
    /// Minimum height (dB) from the lower bound of the multitaper
    /// spectrum's 95% chi² confidence interval:
    ///   `ht_db_min = -2 · pow2db(2K / chi2inv(0.975, 2K))`
    /// for `K` tapers, as in `computeTFPeaks.m` (`computeSpectrogram`).
    fn ht_db_min(&self) -> f64 {
        let degrees_of_freedom = 2.0 * self.n_tapers as f64;
        let upper_quantile = chi2inv_even_dof(0.975, self.n_tapers);
        let ratio_db = 10.0 * (degrees_of_freedom / upper_quantile).log10();
        -2.0 * ratio_db
    }
}

/// Chi² cumulative probability at `x` for `2 * half_dof` degrees of
/// freedom. With an even number of degrees of freedom the distribution
/// function has a closed form:
///   `P(x) = 1 - exp(-y) · Σ_{j=0}^{half_dof-1} y^j / j!`,  `y = x / 2`.
fn chi2cdf_even_dof(x: f64, half_dof: usize) -> f64 {
    let y = x / 2.0;
    let mut term = 1.0_f64; // y^j / j!
    let mut sum = 0.0_f64;
    for j in 0..half_dof {
        if j > 0 {
            term *= y / j as f64;
        }
        sum += term;
    }
    1.0 - (-y).exp() * sum
}

/// Chi² quantile for `2 * half_dof` degrees of freedom: the `x` with
/// cumulative probability `p`. MATLAB `chi2inv(p, 2 * half_dof)`.
///
/// Solved by bisection on the closed-form distribution function until the
/// bracket cannot shrink further in double precision.
fn chi2inv_even_dof(p: f64, half_dof: usize) -> f64 {
    if half_dof == 0 || !(p > 0.0 && p < 1.0) {
        return f64::NAN;
    }
    // Bracket the quantile: grow the upper end until it is past `p`.
    let mut low = 0.0_f64;
    let mut high = 2.0 * half_dof as f64;
    while chi2cdf_even_dof(high, half_dof) < p {
        high *= 2.0;
    }
    loop {
        let middle = 0.5 * (low + high);
        if middle <= low || middle >= high {
            return middle;
        }
        if chi2cdf_even_dof(middle, half_dof) < p {
            low = middle;
        } else {
            high = middle;
        }
    }
}

/// Owned, in-memory mirror of `dynamo_export::aux_h5::SummarySpecCache`.
/// Lives in dynamo-pipeline so it can sit on `DynamoOutputs` without
/// dragging in a dynamo-export dependency. The CLI's cache-rebuild
/// populates it from a freshly-read AuxBundle; the renderer reads it
/// and decodes the PNG bytes directly.
#[derive(Debug, Clone)]
pub struct SummarySpecCacheData {
    pub png_bytes: Vec<u8>,
    pub vmin: f32,
    pub vmax: f32,
    pub t_first_h: f64,
    pub t_last_h: f64,
}

/// Inputs that apply to every stage of the pipeline. Bundled to keep
/// `run_extract_two_pass` / `run_dynamo` from devolving into 15-arg
/// functions.
pub struct DynamoInput<'a> {
    /// EEG samples after time-range truncation (sample[0] at `t_time_range[0]`).
    pub data: ArrayView1<'a, f64>,
    /// Sampling rate (Hz).
    pub fs: f64,
    /// Absolute timestamps (seconds) for each `data` sample.
    pub t_time_range: ArrayView1<'a, f64>,
    /// Sleep-stage scoring breakpoints (seconds).
    pub stage_times: &'a [f64],
    /// Stage values (typically 1..=5; 0 = unscored).
    pub stage_vals: &'a [f64],
    /// Artifact mask aligned with `data` / `t_time_range`.
    pub artifacts: &'a [bool],
}

/// A pre-computed multitaper spectrogram for one pass.
pub struct PassSpectrogram<'a> {
    pub spect: ArrayView2<'a, f64>,
    pub stimes: ArrayView1<'a, f64>,
    pub mtm: PassMtmSpec,
}

/// Phase-1 early-exit selector for [`run_dynamo_with_progress`]. When set,
/// the orchestrator runs through peak detection + SO-power/SO-phase compute +
/// summary-spectrogram compute, then returns a partial `DynamoOutputs`
/// with empty SOPHs / fits. Used by the two-phase batch driver so Phase 2
/// (cache-rebuild at higher concurrency) handles the cheap downstream work.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopAfter {
    /// Stop after writing peaks + summary-spec PNG into aux.h5. Skip SOPH
    /// binning, paramfit, splinefit, and the final figure render.
    Peaks,
}

/// Outputs of the two-pass extract orchestrator.
pub struct TwoPassOutputs {
    /// Final filtered + refined peak table.
    pub stats: SegmentPeaks,
    /// Pass-2 label image (full-resolution, peaks numbered 1..len).
    pub label_image: Array2<i64>,
    /// Per-frequency baseline used to flatten both spectrograms (F,).
    pub baseline: Array1<f64>,
}

/// Boundary value carried from [`prepare_pass2_input`] into
/// [`extract_pass2_and_finalize`]. Owning the masked pass-2 spectrogram
/// here is what lets the caller drop the pass-1 spectrogram (held in
/// the outer caller's scope) once prepare returns — that's the whole
/// point of splitting `run_extract_two_pass_with_progress` into two
/// public phases: pass-1 holds a full-resolution spectrogram, and keeping
/// it alive through the pass-2 watershed roughly doubles peak memory.
#[derive(Debug)]
pub struct Pass2Prep {
    /// Pass-2 spectrogram after masking with pass-1 labels. Owned so the
    /// caller is free to drop the pass-1 spectrogram + label image
    /// before the finalize phase allocates pass-2 watershed buffers.
    pub spect_masked: Array2<f64>,
    /// Per-frequency baseline (single column), shared by both passes.
    pub baseline: Array1<f64>,
    /// Pass-1 `dur_min`, carried as a scalar so the caller can drop
    /// `pass1` (and its `mtm`) entirely. MATLAB convention has pass-2
    /// inherit pass-1's `dur_min` for filterStatsTable.
    pub dur_min_p1: f64,
}

/// Outputs of the full `run_dynamo` pipeline.
pub struct DynamoOutputs {
    pub stats: SegmentPeaks,
    pub baseline: Array1<f64>,
    /// Frequency axis (Hz) of `baseline` — the extraction spectrogram's
    /// rows. Persisted to aux h5 so the cache-rebuild path can re-anchor
    /// peak heights (PkPower) without the EDF; empty when the rebuild's
    /// aux predates baseline persistence.
    pub baseline_freqs: Vec<f64>,
    pub artifacts: Vec<bool>,
    /// Sample rate (Hz) at which the pipeline actually ran — equals
    /// `settings.resample_target_fs` when downsampling kicked in,
    /// otherwise the EDF's native fs. Needed by the aux-data writer
    /// (MATLAB `auxiliary_data.Fs`).
    pub fs: f64,
    /// Absolute timestamps for each EEG sample at `fs` (length matches
    /// the resampled data; equals the input `t_time_range`).
    pub t_time_range: Vec<f64>,
    /// Sleep-stage breakpoint times (s), echoed from the input for
    /// downstream writers (aux h5 carries them).
    pub stage_times: Vec<f64>,
    /// Sleep-stage values aligned with `stage_times`.
    pub stage_vals: Vec<f64>,

    /// SO-power time series, normalized per `settings.sopower.norm_method`.
    /// Length matches `so_power_times`. EEG-rate when `retain_fs=true`.
    pub so_power_norm: Vec<f64>,
    pub so_power_times: Vec<f64>,

    /// SO-power on its NATIVE compute grid (window step), regardless of
    /// `retain_fs`. This is what the compact aux artifact stores — the
    /// EEG-rate `so_power_norm` above is a redundant upsample for per-peak
    /// interpolation. `so_power_native_t_start` is the timestamp of the
    /// first native sample; the uniform step is `sopower.window_params.1`.
    pub so_power_norm_native: Vec<f64>,
    pub so_power_native_t_start: f64,

    /// SO-phase time series, unwrapped (radians). NaN at excluded samples.
    /// Length matches `so_phase_times` (which equals `t_time_range` when
    /// `retain_fs=true`).
    pub so_phase_unwrapped: Vec<f64>,
    pub so_phase_times: Vec<f64>,

    /// Per-peak SO-power from `crate::peak_assign::assign_peak_sopower`
    /// (MATLAB `computePeakSOpower.m`): linear interpolation of the time
    /// series at each peak's `peak_time`. NaN where a neighboring sample is
    /// NaN (e.g., artifact) or the peak is more than one grid step outside
    /// the SO-power time range. Aligned with `stats.peak_time`.
    pub peak_sopower: Vec<f64>,
    /// Per-peak SO-phase from `assign_peak_sophase` (MATLAB
    /// `computePeakSOphase.m`), wrapped to [-π, π]. NaN where unavailable.
    pub peak_sophase: Vec<f64>,

    /// Per-peak sleep stage from `assign_peak_stage` (MATLAB
    /// `computePeakStage.m`):
    /// `interp1(stage_times, stage_vals, PeakTime, 'previous')`, with
    /// peaks whose nearest EEG sample is an artifact overridden to `6`
    /// (Artifact) and peaks outside the scored range set to `0` (Unknown).
    pub peak_stage: Vec<u8>,
    /// Whether each peak passes the SOPH inclusion filter
    /// (MATLAB `hist_peakidx`): `peak_stage ∈ SOPH_stages` AND peak inside
    /// time_range AND `peak_freq` inside `mtm_freq_range`. Use this to
    /// drive the summary scatter plot and any other "what's in the
    /// histogram?" view.
    pub peak_in_hist: Vec<bool>,

    /// 2-D SO-power histogram: (sopower_bins, freq_bins). Empty (0×0) when
    /// the run stopped early with `StopAfter::Peaks`.
    pub sopower_hist: Array2<f64>,
    /// Frequency-bin centers for both histograms (Hz).
    pub hist_freq_centers: Vec<f64>,
    /// SO-power-bin centers matching `sopower_hist`'s first axis.
    pub sopower_bin_centers: Vec<f64>,

    /// 2-D SO-phase histogram: (sophase_bins, freq_bins).
    pub sophase_hist: Array2<f64>,
    /// SO-phase-bin centers (radians).
    pub sophase_bin_centers: Vec<f64>,

    /// Pass-2 multitaper spectrogram (shape: F × T), the matrix that fed
    /// the pass-2 extract. Empty (0×0) unless the run was made with
    /// [`run_dynamo_keeping_spectrogram`]: it is as large as the whole
    /// time-frequency grid and no output is written from it.
    pub spectrogram: Array2<f64>,
    /// Frequency axis for `spectrogram` (Hz).
    pub spect_sfreqs: Vec<f64>,
    /// Time axis for `spectrogram` (seconds, absolute frame).
    pub spect_stimes: Vec<f64>,

    /// Display spectrogram for the summary figure — NW=15, K=29, 30s
    /// window, 15s step, linear detrend, freq range [2, 25] Hz. Matches
    /// `displaySummaryPlot.m` line 220's `multitaper_spectrogram_dynamo`
    /// call. Coarser than the extract spectrogram on purpose; gives a
    /// clean overview without the 2s-window noise.
    pub summary_spectrogram: Array2<f64>,
    pub summary_sfreqs: Vec<f64>,
    pub summary_stimes: Vec<f64>,

    /// Pre-rendered summary-spec panel cache (paletted PNG bytes + clim +
    /// time range). Populated by cache-rebuild from the aux.h5 it
    /// just read; the renderer prefers this over recomputing the
    /// spec from `summary_spectrogram`. Empty on full runs (the
    /// renderer makes a fresh panel and exports the bytes back to be
    /// embedded in the aux.h5 that gets written downstream).
    pub summary_spec_cached: Option<SummarySpecCacheData>,

    /// Parametric (rotated-Gaussian) fit to the SO-power histogram.
    /// `None` when `settings.output.write_paramfits == false` or the
    /// fit failed (the orchestrator logs and continues — paramfit is
    /// best-effort).
    pub paramfit_power: Option<crate::fits::ParamFitResult>,
    /// Parametric (von-Mises × Gaussian) fit to the SO-phase histogram.
    pub paramfit_phase: Option<crate::fits::ParamFitResult>,
    /// Reason the power paramfit declined on degenerate input (e.g. no
    /// valid histogram bins in the fit window). The CSV writer emits a
    /// stamped zero-mode file recording it so the output set is complete.
    pub paramfit_power_skipped: Option<String>,
    /// Same as `paramfit_power_skipped`, for the phase axis.
    pub paramfit_phase_skipped: Option<String>,
    /// Tensor-product B-spline fit to the SO-power histogram.
    pub splinefit_power: Option<crate::fits::SplineFitResult>,
    /// Tensor-product B-spline fit to the SO-phase histogram.
    pub splinefit_phase: Option<crate::fits::SplineFitResult>,
}

#[derive(Debug, thiserror::Error)]
pub enum DynamoError {
    #[error("DPSS solver: {0}")]
    Dpss(#[from] multitaper_rs::dpss::DpssError),
    #[error("EDF I/O: {0}")]
    Io(String),
    #[error("stage {stage} failed: {msg}")]
    Stage { stage: &'static str, msg: String },
    #[error("invalid inputs: {0}")]
    Invalid(String),
    #[error("cancelled by user")]
    Cancelled,
}

impl DynamoError {
    pub(crate) fn stage(stage: &'static str, msg: impl Into<String>) -> Self {
        Self::Stage { stage, msg: msg.into() }
    }
}

// -------- Helpers --------

/// Build the `ExtractParams` we pass into `extract_tfpeaks`. **MATLAB
/// convention**: only the lower bounds (`dur_min`, `bw_min`) flow into the
/// extract algorithm; max bounds and `ht_db_min` are deferred to an
/// external `filterStatsTable` step on the returned peaks. This is the
/// behavior of `runSegmentedData.m` which passes `dur_max=inf`, `bw_max=inf`,
/// `freq_*=±inf`, `ht_db_min=-inf` to `extract_tfpeaks_mex`, keeping the
/// full label image available for the pass-2 mask step.
///
/// Apply `filter_stats_table` to the resulting peaks before exposing them
/// to the user.
fn extract_params(
    settings: &DynamoSettings,
    mtm: &PassMtmSpec,
    bw_min_override: Option<f64>,
    dur_min_override: Option<f64>,
    trim_shift_val: f64,
) -> ExtractParams {
    let det = &settings.detection;
    ExtractParams {
        seg_time: det.seg_time,
        downsample_f: det.downsample_f as usize,
        downsample_t: det.downsample_t as usize,
        merge_thresh: det.merge_thresh,
        max_merges: f64::INFINITY,
        trim_vol_thresh: det.trim_vol_thresh,
        trim_shift_val,
        dur_min: dur_min_override.unwrap_or(mtm.dur_min()),
        // Loose upper bounds — filter externally after extract.
        dur_max: f64::INFINITY,
        bw_min: bw_min_override.unwrap_or(mtm.bw_min()),
        bw_max: f64::INFINITY,
        freq_min: f64::NEG_INFINITY,
        freq_max: f64::INFINITY,
        ht_db_min: f64::NEG_INFINITY,
        expand_labels_distance: 0, // MATLAB-paint (default)
    }
}

/// External post-extract filter matching MATLAB `filterStatsTable.m`:
///   * `dur_min < Duration < dur_max`
///   * `bw_min  < Bandwidth < bw_max`
///   * `freq_min < PeakFrequency < freq_max`
///   * `pow2db(Height) > ht_db_min`  (Height must be positive)
fn filter_stats_table_indices(
    peaks: &SegmentPeaks,
    dur_range: (f64, f64),
    bw_range: (f64, f64),
    freq_range: (f64, f64),
    ht_db_min: f64,
) -> Vec<usize> {
    let mut keep = Vec::with_capacity(peaks.len());
    for i in 0..peaks.len() {
        let dur = peaks.duration[i];
        let bw = peaks.bandwidth[i];
        let pf = peaks.peak_freq[i];
        let h = peaks.height[i];
        let h_db = if h > 0.0 { 10.0 * h.log10() } else { f64::NAN };
        if dur > dur_range.0 && dur < dur_range.1
            && bw > bw_range.0 && bw < bw_range.1
            && pf > freq_range.0 && pf < freq_range.1
            && h_db.is_finite() && h_db > ht_db_min
        {
            keep.push(i);
        }
    }
    keep
}

/// Return a new `SegmentPeaks` containing only the rows at `idx`. Unlike
/// `take_with_refined_freq` below, this preserves `peak_freq` exactly.
fn take_indices(p: &SegmentPeaks, idx: &[usize]) -> SegmentPeaks {
    let mut out = SegmentPeaks::default();
    out.peak_time.reserve(idx.len());
    out.peak_freq.reserve(idx.len());
    out.duration.reserve(idx.len());
    out.bandwidth.reserve(idx.len());
    out.height.reserve(idx.len());
    out.volume.reserve(idx.len());
    out.segment_num.reserve(idx.len());
    out.bbox.reserve(idx.len() * 4);
    out.area.reserve(idx.len());
    out.peakiness.reserve(idx.len());
    out.height_data.reserve(idx.len());
    out.boundaries_xy.reserve(idx.len());
    for &i in idx {
        out.peak_time.push(p.peak_time[i]);
        out.peak_freq.push(p.peak_freq[i]);
        out.duration.push(p.duration[i]);
        out.bandwidth.push(p.bandwidth[i]);
        out.height.push(p.height[i]);
        out.volume.push(p.volume[i]);
        out.segment_num.push(p.segment_num[i]);
        out.bbox.extend_from_slice(&p.bbox[i * 4..i * 4 + 4]);
        out.area.push(p.area[i]);
        out.peakiness.push(p.peakiness[i]);
        out.height_data.push(p.height_data[i].clone());
        out.boundaries_xy.push(p.boundaries_xy[i].clone());
    }
    out
}

fn to_vec(view: ArrayView1<f64>) -> Vec<f64> { view.iter().copied().collect() }

/// Port of `runSegmentedData::compute_global_trim_shift` (MATLAB local fn):
/// global min over (spect / baseline), restricted to rows where baseline is
/// positive and finite. Used as the `trim_shift_val` passed to extract so
/// every segment trims against the same baseline-reference floor — without
/// this, per-segment-min trimming makes us drop peaks at segment edges and
/// drift ~2% off MATLAB's count.
fn compute_global_trim_shift(spect: ArrayView2<f64>, baseline: Option<ArrayView1<f64>>) -> f64 {
    let (f, t) = spect.dim();
    let mut min_val = f64::INFINITY;
    #[allow(clippy::single_match)]
    match baseline {
        Some(bl) => {
            let mut any_good = false;
            for r in 0..f {
                let bv = bl[r];
                if bv > 0.0 && bv.is_finite() {
                    any_good = true;
                    for c in 0..t {
                        let v = spect[[r, c]] / bv;
                        if v < min_val { min_val = v; }
                    }
                }
            }
            if any_good {
                return min_val;
            }
        }
        None => {}
    }
    for v in spect.iter() {
        if *v < min_val { min_val = *v; }
    }
    min_val
}

// -------- run_extract_two_pass --------

/// Compose pass-1 baseline + extract → pass-2 mask + extract → refine, on
/// pre-computed multitaper spectrograms. Mirrors the body of
/// `computeTFPeaks.m` from pass-1 (line ~298) through refinement (~line 430).
///
/// Per-pass `dur_min`/`bw_min`/`ht_db_min` are derived from the multitaper
/// spec on the corresponding pass, but per MATLAB convention pass-2 inherits
/// pass-1's `dur_min` (intentional: keeps shorter-duration peaks alive).
pub fn run_extract_two_pass(
    input: &DynamoInput,
    pass1: &PassSpectrogram,
    pass2: &PassSpectrogram,
    sfreqs: ArrayView1<f64>,
    freq_range: (f64, f64),
    settings: &DynamoSettings,
    cancel: Option<&(dyn Fn() -> bool + Sync)>,
) -> Result<TwoPassOutputs, DynamoError> {
    run_extract_two_pass_with_progress(
        input, pass1, pass2, sfreqs, freq_range, settings, cancel, None, (0.0, 1.0),
    )
}

/// Emit a `TF peaks (pass N) D0%` stage line at each new 10% bucket of a pass's
/// per-segment progress, mapping the within-pass fraction into the overall
/// `[lo, hi]` progress window. Mirrors DYNAM-O `runSegmentedData.m`'s
/// decile console ticks. Thread-safe: `extract_tfpeaks` calls this from its
/// rayon per-segment loop, and `fetch_max` makes exactly one thread emit each
/// decile.
fn decile_report(
    progress: Option<&ProgressFn<'_>>,
    last_bucket: &AtomicU32,
    pass: u8,
    lo: f64,
    hi: f64,
    cur: u32,
    total: u32,
) {
    if total == 0 {
        return;
    }
    let frac_in = (cur as f64 / total as f64).clamp(0.0, 1.0);
    let bucket = ((frac_in * 100.0).floor() as u32 / 10) * 10;
    if bucket == 0 {
        return;
    }
    let prev = last_bucket.fetch_max(bucket, Ordering::SeqCst);
    if bucket > prev {
        if let Some(cb) = progress {
            cb(
                &format!("TF peaks (pass {pass}) {bucket}%"),
                (lo + (hi - lo) * frac_in).clamp(0.0, 1.0),
            );
        }
    }
}

/// Same as [`run_extract_two_pass`] but emits decile progress/log lines for
/// each watershed pass via `progress`, mapped into the `frac_range` window.
///
/// Implemented as a thin wrapper around [`prepare_pass2_input`] +
/// [`extract_pass2_and_finalize`]. The two-stage path is what
/// `run_dynamo_with_progress` uses directly so it can drop the pass-1
/// spectrogram between the two phases; this wrapper preserves the
/// single-call public API for tests and any external callers.
#[allow(clippy::too_many_arguments)]
pub fn run_extract_two_pass_with_progress(
    input: &DynamoInput,
    pass1: &PassSpectrogram,
    pass2: &PassSpectrogram,
    sfreqs: ArrayView1<f64>,
    freq_range: (f64, f64),
    settings: &DynamoSettings,
    cancel: Option<&(dyn Fn() -> bool + Sync)>,
    progress: Option<&ProgressFn<'_>>,
    frac_range: (f64, f64),
) -> Result<TwoPassOutputs, DynamoError> {
    // Split the watershed window so pass 1 (shorter) maps to the first ~40%.
    let (flo, fhi) = frac_range;
    let fmid = flo + (fhi - flo) * 0.4;
    let prep = prepare_pass2_input(
        input, pass1, pass2, sfreqs, settings, cancel, progress, (flo, fmid),
    )?;
    extract_pass2_and_finalize(
        input, prep, pass2, sfreqs, freq_range, settings, cancel, progress, (fmid, fhi),
    )
}

/// Phase A of the two-pass extract: baseline + pass-1 watershed +
/// pass-2 masking. Returns the masked pass-2 spectrogram (owned), the
/// shared baseline, and pass-1's `dur_min` scalar — everything the
/// finalize phase needs from pass-1. After this returns, the caller is
/// free to drop the pass-1 spectrogram + stimes + labels (held in the
/// outer caller's scope) before [`extract_pass2_and_finalize`]
/// allocates pass-2 watershed buffers.
///
/// These are steps 1-4 of the two-pass extract, in MATLAB's operation
/// order.
#[allow(clippy::too_many_arguments)]
pub fn prepare_pass2_input(
    input: &DynamoInput,
    pass1: &PassSpectrogram,
    pass2: &PassSpectrogram,
    sfreqs: ArrayView1<f64>,
    settings: &DynamoSettings,
    cancel: Option<&(dyn Fn() -> bool + Sync)>,
    progress: Option<&ProgressFn<'_>>,
    frac_range: (f64, f64),
) -> Result<Pass2Prep, DynamoError> {
    let (flo, fhi) = frac_range;
    // 1. Build baseline_exclude mask.
    let t_vec = to_vec(input.t_time_range);
    let exclude = build_baseline_exclude(
        &t_vec,
        input.stage_times,
        input.stage_vals,
        &settings.baseline.stages,
        input.artifacts,
        None,
    )
    .map_err(|e| DynamoError::stage("baseline_exclude", e))?;

    // 2. Baseline from pass-1 spectrogram (per-freq percentile over valid time cols).
    // The baseline time range is always unbounded: MATLAB's `baseline_trim`
    // option is not surfaced here, so no app-side override applies.
    let baseline_range = (f64::NEG_INFINITY, f64::INFINITY);
    let baseline_2d = compute_baseline(
        pass1.spect,
        pass1.stimes,
        input.t_time_range,
        &exclude,
        baseline_range,
        settings.baseline.ptile,
    )
    .map_err(|e| DynamoError::stage("compute_baseline", e))?;
    let baseline_1d: Array1<f64> = baseline_2d.column(0).to_owned();

    // 3. Pass-1 extract with LOOSE filter params (dur_max=inf, etc.) so the
    //    full label image is preserved for the pass-2 mask. External
    //    filterStatsTable still gets applied on the peaks below — but the
    //    label image we hand to pass-2 retains every region.
    let trim_shift_1 = compute_global_trim_shift(pass1.spect, Some(baseline_1d.view()));
    let params1 = extract_params(settings, &pass1.mtm, None, None, trim_shift_1);
    if std::env::var("TFPEAKS_DEBUG").is_ok() {
        eprintln!("[tfdbg] === pass 1 watershed ===");
    }
    let p1_last = AtomicU32::new(0);
    let p1cb = |cur: u32, total: u32| decile_report(progress, &p1_last, 1, flo, fhi, cur, total);
    let (_peaks1_unfiltered, labels1) = extract_tfpeaks(
        pass1.spect,
        pass1.stimes,
        sfreqs,
        Some(baseline_1d.view()),
        &params1,
        Some(&p1cb as &(dyn Fn(u32, u32) + Sync)),
        cancel,
    )
    .map_err(|e| DynamoError::stage("extract_pass1", e))?;

    // 4. Mask pass-2 spectrogram with the FULL pass-1 label image.
    let spect_masked = mask_spectrogram(
        pass2.spect,
        pass2.stimes,
        labels1.view(),
        pass1.stimes,
    )
    .map_err(|e| DynamoError::stage("mask_spectrogram", e))?;
    // `labels1` is not used past this point.

    if std::env::var("TFPEAKS_DEBUG").is_ok() {
        eprintln!(
            "[tfdbg] pass1 peaks (unfiltered, → pass-2 mask) = {}",
            _peaks1_unfiltered.len()
        );
    }

    Ok(Pass2Prep {
        spect_masked,
        baseline: baseline_1d,
        dur_min_p1: pass1.mtm.dur_min(),
    })
}

/// Phase B of the two-pass extract: pass-2 watershed + MATLAB-style
/// filterStatsTable + Hann-window peak frequency refinement. Consumes
/// the [`Pass2Prep`] produced by [`prepare_pass2_input`].
///
/// Pass-1's only contributions at this point are the masked pass-2
/// spectrogram (carried by `prep.spect_masked`), the shared baseline
/// (`prep.baseline`), and `prep.dur_min_p1` (MATLAB convention has
/// pass-2 inherit pass-1's `dur_min` for filtering). Everything else
/// pass-1 produced is no longer needed and may already be freed.
///
/// These are steps 5-7 of the two-pass extract, in MATLAB's operation
/// order.
#[allow(clippy::too_many_arguments)]
pub fn extract_pass2_and_finalize(
    input: &DynamoInput,
    prep: Pass2Prep,
    pass2: &PassSpectrogram,
    sfreqs: ArrayView1<f64>,
    freq_range: (f64, f64),
    settings: &DynamoSettings,
    cancel: Option<&(dyn Fn() -> bool + Sync)>,
    progress: Option<&ProgressFn<'_>>,
    frac_range: (f64, f64),
) -> Result<TwoPassOutputs, DynamoError> {
    let (flo, fhi) = frac_range;
    let Pass2Prep { spect_masked, baseline: baseline_1d, dur_min_p1 } = prep;

    // 5. Pass-2 extract — pass-2 mtm spec for bw_min, but inherit pass-1
    //    dur_min (MATLAB convention). Loose upper bounds again.
    let trim_shift_2 = compute_global_trim_shift(spect_masked.view(), Some(baseline_1d.view()));
    let params2 = extract_params(
        settings,
        &pass2.mtm,
        Some(pass2.mtm.bw_min()),
        Some(dur_min_p1),
        trim_shift_2,
    );
    if std::env::var("TFPEAKS_DEBUG").is_ok() {
        eprintln!("[tfdbg] === pass 2 watershed (pass-2 spect masked by pass-1 labels) ===");
    }
    let p2_last = AtomicU32::new(0);
    let p2cb = |cur: u32, total: u32| decile_report(progress, &p2_last, 2, flo, fhi, cur, total);
    let (peaks2_unfiltered, labels2) = extract_tfpeaks(
        spect_masked.view(),
        pass2.stimes,
        sfreqs,
        Some(baseline_1d.view()),
        &params2,
        Some(&p2cb as &(dyn Fn(u32, u32) + Sync)),
        cancel,
    )
    .map_err(|e| DynamoError::stage("extract_pass2", e))?;

    // 6. Apply MATLAB-equivalent filterStatsTable on pass-2 peaks.
    let det = &settings.detection;
    let freq_hi = det.freq_max.resolve(f64::INFINITY);
    let keep2 = filter_stats_table_indices(
        &peaks2_unfiltered,
        (dur_min_p1, det.dur_max),
        (pass2.mtm.bw_min(), det.bw_max),
        (det.freq_min, freq_hi),
        pass2.mtm.ht_db_min(),
    );
    let peaks2_filtered = take_indices(&peaks2_unfiltered, &keep2);

    // 7. Hann-window peak frequency refinement.
    let refined = refine_peaks(&peaks2_filtered, input, freq_range)?;

    if std::env::var("TFPEAKS_DEBUG").is_ok() {
        eprintln!(
            "[tfdbg] pass2 peaks: {} trimmed → {} after filterStatsTable → {} after refine",
            peaks2_unfiltered.len(),
            peaks2_filtered.len(),
            refined.len()
        );
    }

    Ok(TwoPassOutputs {
        stats: refined,
        label_image: labels2,
        baseline: baseline_1d,
    })
}

/// Hann-window peak frequency refinement.
///
/// Matches `refinePeakFrequency.m` defaults:
///   * window_size = 4 s
///   * dsfreqs     = 0.05 Hz → nfft = next_pow2(Fs/dsfreqs)
///   * detrend     = 'constant'
///
/// Events within `window_size/2` of either data edge are kept unchanged
/// (no refinement, no drop). Interior events whose refined frequency lands
/// within 1e-3 of their bbox bounds return NaN and are dropped.
fn refine_peaks(
    peaks: &SegmentPeaks,
    input: &DynamoInput,
    freq_range: (f64, f64),
) -> Result<SegmentPeaks, DynamoError> {
    const WINDOW_SIZE: f64 = 4.0;
    const DSFREQS: f64 = 0.05;
    if peaks.is_empty() {
        return Ok(SegmentPeaks::default());
    }

    let t0 = input.t_time_range[0];
    let t_end = input.t_time_range[input.t_time_range.len() - 1];
    let half = 0.5 * WINDOW_SIZE;
    let n = peaks.len();

    let mut interior_idx: Vec<usize> = Vec::with_capacity(n);
    for i in 0..n {
        let pt = peaks.peak_time[i];
        if pt >= t0 + half && pt <= t_end - half {
            interior_idx.push(i);
        }
    }

    let event_times: Vec<f64> = interior_idx.iter().map(|&i| peaks.peak_time[i]).collect();
    let data_vec: Vec<f64> = input.data.iter().copied().collect();
    let (spect, sfreqs_refine) = hann_event_spectra(
        &data_vec,
        input.fs,
        &event_times,
        t0,
        freq_range,
        WINDOW_SIZE,
        DSFREQS,
        DetrendOpt::Constant,
    );

    // bbox layout = (t_tl, f_tl, width_s, height_Hz) per pydynamo convention.
    let bbox_lo: Vec<f64> = interior_idx.iter().map(|&i| peaks.bbox[i * 4 + 1]).collect();
    let bbox_hi: Vec<f64> = interior_idx
        .iter()
        .map(|&i| peaks.bbox[i * 4 + 1] + peaks.bbox[i * 4 + 3])
        .collect();

    let refined = refine_from_spectra(&spect, &sfreqs_refine, &bbox_lo, &bbox_hi, 1000, true);

    let mut keep_mask = vec![true; n];
    let mut new_peak_freq: Vec<f64> = peaks.peak_freq.clone();
    for (j, &i) in interior_idx.iter().enumerate() {
        let r = refined[j];
        if r.is_finite() {
            new_peak_freq[i] = r;
        } else {
            keep_mask[i] = false;
        }
    }
    let keep_idx: Vec<usize> = (0..n).filter(|&i| keep_mask[i]).collect();
    Ok(take_with_refined_freq(peaks, &keep_idx, &new_peak_freq))
}

fn take_with_refined_freq(p: &SegmentPeaks, idx: &[usize], new_peak_freq: &[f64]) -> SegmentPeaks {
    let mut out = SegmentPeaks::default();
    let n = idx.len();
    out.peak_time.reserve(n);
    out.peak_freq.reserve(n);
    out.duration.reserve(n);
    out.bandwidth.reserve(n);
    out.height.reserve(n);
    out.volume.reserve(n);
    out.segment_num.reserve(n);
    out.bbox.reserve(n * 4);
    out.area.reserve(n);
    out.peakiness.reserve(n);
    out.height_data.reserve(n);
    out.boundaries_xy.reserve(n);
    for &i in idx {
        out.peak_time.push(p.peak_time[i]);
        out.peak_freq.push(new_peak_freq[i]);
        out.duration.push(p.duration[i]);
        out.bandwidth.push(p.bandwidth[i]);
        out.height.push(p.height[i]);
        out.volume.push(p.volume[i]);
        out.segment_num.push(p.segment_num[i]);
        out.bbox.extend_from_slice(&p.bbox[i * 4..i * 4 + 4]);
        out.area.push(p.area[i]);
        out.peakiness.push(p.peakiness[i]);
        out.height_data.push(p.height_data[i].clone());
        out.boundaries_xy.push(p.boundaries_xy[i].clone());
    }
    out
}

// -------- run_dynamo: full EDF-to-SOPHs pipeline --------

use multitaper_rs::{
    compute_spectrogram, DetrendMode, SpectrogramParams, Weighting,
};

/// Smallest power of two ≥ `x.ceil()`. `next_power_of_two` maps 0 to 1, so
/// no separate zero case is needed.
fn next_pow2(x: f64) -> usize {
    (x.ceil() as usize).next_power_of_two()
}

/// First-to-last scored-stage window: the canonical time-range rule
/// shared by the full pipeline and every cache-rebuild path. Scans
/// `(stage_times, stage_vals)` for values strictly inside (0, 6) —
/// NREM/REM/Wake, excluding Undefined=0 and Artifact=6 — and returns
/// `(first, last)` when last > first. Mirrors `runDYNAMO.m:255-258`.
/// Generic over the stage-value type because the orchestrator holds
/// f64 stages while aux.h5 readers hold integer stages.
pub fn scored_window<T: Copy + Into<f64>>(
    stage_times: &[f64],
    stage_vals: &[T],
) -> Option<(f64, f64)> {
    let mut first: Option<f64> = None;
    let mut last: Option<f64> = None;
    for (t, v) in stage_times.iter().zip(stage_vals.iter()) {
        let v: f64 = (*v).into();
        if v > 0.0 && v < 6.0 {
            if first.is_none() {
                first = Some(*t);
            }
            last = Some(*t);
        }
    }
    match (first, last) {
        (Some(a), Some(b)) if b > a => Some((a, b)),
        _ => None,
    }
}

/// Apply a time-range filter to `data` and the implicit time axis.
/// Returns `(data_trim, t_trim)` where `t_trim` are the absolute timestamps
/// (seconds since data[0]). When `time_range` is `None`, returns the full
/// arrays.
fn truncate_time_range(data: &[f64], fs: f64, time_range: Option<(f64, f64)>)
    -> (Vec<f64>, Vec<f64>)
{
    let n_full = data.len();
    match time_range {
        None => {
            let t: Vec<f64> = (0..n_full).map(|i| i as f64 / fs).collect();
            (data.to_vec(), t)
        }
        Some((t_start, t_end)) => {
            let mut dt = Vec::new();
            let mut tt = Vec::new();
            #[allow(clippy::needless_range_loop)]
            for i in 0..n_full {
                let t = i as f64 / fs;
                if t >= t_start && t <= t_end {
                    dt.push(data[i]);
                    tt.push(t);
                }
            }
            (dt, tt)
        }
    }
}

/// Stage progress callback: `(stage_name, fraction_done_0_to_1)`.
/// `fraction` is monotonic across the call but rough — bin it for a
/// progress bar.
pub type ProgressFn<'a> = dyn Fn(&str, f64) + Send + Sync + 'a;

/// Cancellation poll. Return `true` to abort the pipeline at the next
/// checkpoint with `DynamoError::Cancelled`. The orchestrator polls this
/// after every phase boundary and inside the chunked summary spectrogram
/// loop — worst-case latency is one phase, typically a few hundred ms.
pub type CancelFn<'a> = dyn Fn() -> bool + Send + Sync + 'a;

/// Full pipeline from raw EEG + staging through TF-peak stats, SO-power,
/// SO-phase, the SOPH histograms, the summary spectrogram, and the fits.
///
/// Matches `runDYNAMO.m` orchestration (time-range trim → artifact detect →
/// pass-1 spectrogram → pass-2 spectrogram → two-pass extract → refine →
/// SO-power / SO-phase → histograms → fits).
pub fn run_dynamo(
    data: &[f64],
    fs: f64,
    stage_times: &[f64],
    stage_vals: &[f64],
    time_range: Option<(f64, f64)>,
    settings: &DynamoSettings,
) -> Result<DynamoOutputs, DynamoError> {
    run_dynamo_with_progress(
        data,
        fs,
        stage_times,
        stage_vals,
        time_range,
        settings,
        None,
        None,
        None,
    )
}

/// Same as [`run_dynamo`] plus a progress callback. Stages are named
/// in MATLAB's `runDYNAMO.m` style ("computing spectrogram (pass 1)…",
/// "extracting TF peaks…"). `stop_after` (None for full pipeline; see
/// [`StopAfter`]) controls the Phase-1 early-exit used by the two-phase
/// batch driver.
#[allow(clippy::too_many_arguments)]
pub fn run_dynamo_with_progress(
    data: &[f64],
    fs: f64,
    stage_times: &[f64],
    stage_vals: &[f64],
    time_range: Option<(f64, f64)>,
    settings: &DynamoSettings,
    progress: Option<&ProgressFn<'_>>,
    cancel: Option<&CancelFn<'_>>,
    stop_after: Option<StopAfter>,
) -> Result<DynamoOutputs, DynamoError> {
    let keep_pass2_spectrogram = false;
    run_dynamo_impl(
        data,
        fs,
        stage_times,
        stage_vals,
        time_range,
        settings,
        progress,
        cancel,
        stop_after,
        keep_pass2_spectrogram,
    )
}

/// Same as [`run_dynamo`], but also returns the pass-2 spectrogram in
/// [`DynamoOutputs::spectrogram`]. That matrix is as large as the whole
/// time-frequency grid (over a gigabyte for a night at the default
/// settings) and no output is written from it, so the ordinary entry
/// points release it as soon as the peaks are extracted. This one is for
/// tools that inspect the extraction itself.
pub fn run_dynamo_keeping_spectrogram(
    data: &[f64],
    fs: f64,
    stage_times: &[f64],
    stage_vals: &[f64],
    time_range: Option<(f64, f64)>,
    settings: &DynamoSettings,
) -> Result<DynamoOutputs, DynamoError> {
    let keep_pass2_spectrogram = true;
    run_dynamo_impl(
        data,
        fs,
        stage_times,
        stage_vals,
        time_range,
        settings,
        None,
        None,
        None,
        keep_pass2_spectrogram,
    )
}

/// The pipeline behind the `run_dynamo*` entry points.
/// `keep_pass2_spectrogram` decides whether the pass-2 spectrogram is
/// returned or released once the peaks are extracted.
#[allow(clippy::too_many_arguments)]
fn run_dynamo_impl(
    data: &[f64],
    fs: f64,
    stage_times: &[f64],
    stage_vals: &[f64],
    time_range: Option<(f64, f64)>,
    settings: &DynamoSettings,
    progress: Option<&ProgressFn<'_>>,
    cancel: Option<&CancelFn<'_>>,
    stop_after: Option<StopAfter>,
    keep_pass2_spectrogram: bool,
) -> Result<DynamoOutputs, DynamoError> {
    // Progress fractions below are calibrated from observed wall-clock
    // timings across a 12-job sample (3 EDF × 4 channels, ~58 s/job
    // mean). Cumulative-time milestones from that sample:
    //   resampling/artifact-setup  ≈   0.2 s →  0.00–0.005
    //   artifact detection         ≈   8.5 s →  break at 0.15
    //   spec pass 1                ≈   4.6 s →  break at 0.23
    //   spec pass 2 → TF peaks     ≈   4.8 s →  break at 0.31
    //   TF peaks watershed         ≈  34.3 s →  break at 0.90
    //   SO-power                   ≈   0.02 s →  break at 0.90
    //   SO-phase                   ≈   0.85 s →  break at 0.91
    //   build histograms           ≈   1.85 s →  break at 0.94
    //   summary spect              ≈   0.98 s →  break at 0.96
    //   fitting + finalize         ≈   1.14 s →  break at 0.98
    //   (Done state)               ≈   0.9  s →  bar snaps to 1.00 via
    //                                            JobState::Done transition.
    // Total ≈ 58 s; fractions match elapsed-time proportions so the
    // bar moves at a roughly-constant wall-clock rate instead of
    // stalling on the long stages and racing on the short ones.
    let report = |name: &str, frac: f64| {
        if let Some(cb) = progress { cb(name, frac.clamp(0.0, 1.0)); }
    };
    let check_cancel = || -> Result<(), DynamoError> {
        if cancel.is_some_and(|c| c()) { Err(DynamoError::Cancelled) } else { Ok(()) }
    };
    report("starting", 0.0);
    check_cancel()?;

    // 0. Optional EEG resample. Downsample to `settings.resample_target_fs`
    //    when input fs exceeds it. Never upsamples. The 100 Hz default
    //    is an app choice (one rate across a cohort, shorter run time).
    //    MATLAB runs at the native rate unless given `resample_freq`.
    let (data_owned, fs_actual): (Vec<f64>, f64) =
        match settings.resample_target_fs {
            Some(target) if target.is_finite() && target > 0.0 && fs > target => {
                report(&format!("resampling {:.0}→{:.0} Hz", fs, target), 0.005);
                check_cancel()?;
                let resampled = crate::run::resample::resample_downsample(data, fs, target)
                    .map_err(|e| DynamoError::stage("resample", format!("{}", e)))?;
                (resampled, target)
            }
            _ => (data.to_vec(), fs),
        };
    let data: &[f64] = &data_owned;
    let fs = fs_actual;
    if data.is_empty() {
        return Err(DynamoError::Invalid("data is empty".into()));
    }
    if !(fs > 0.0 && fs.is_finite()) {
        return Err(DynamoError::Invalid(format!("invalid fs: {}", fs)));
    }
    if !(settings.mtm.dsfreqs > 0.0 && settings.mtm.dsfreqs.is_finite()) {
        return Err(DynamoError::Invalid(format!(
            "invalid mtm.dsfreqs: {} (must be finite and > 0)",
            settings.mtm.dsfreqs
        )));
    }
    if stage_times.len() != stage_vals.len() {
        return Err(DynamoError::Invalid(format!(
            "stage_times.len {} != stage_vals.len {}",
            stage_times.len(),
            stage_vals.len()
        )));
    }

    // 1. Time-range truncation. MATLAB does this inside computeTFPeaks; we
    //    do it once up front so all downstream stages share the same view.
    //    Default: [first valid stage, last valid stage] over `stage_vals in
    //    [1, 5]` (NREM/REM/Wake — excludes Undefined=0 and Artifact=6).
    //    Mirrors `runDYNAMO.m:255-258`.
    let time_range_resolved = match time_range {
        Some(r) => Some(r),
        None => scored_window(stage_times, stage_vals),
    };
    let (data_trim, t_trim) = truncate_time_range(data, fs, time_range_resolved);
    if data_trim.len() < 4 {
        let rec_dur = data.len() as f64 / fs.max(1e-9);
        let window_desc = match time_range_resolved {
            Some((a, b)) => format!("the scored-sleep window [{:.0}, {:.0}] s", a, b),
            None => "the recording (no scored sleep/wake epochs were found in the staging)"
                .to_string(),
        };
        let stage_span = match (stage_times.first(), stage_times.last()) {
            (Some(a), Some(b)) => format!("[{:.0}, {:.0}] s", a, b),
            _ => "empty".to_string(),
        };
        return Err(DynamoError::Invalid(format!(
            "no valid times remaining for peak detection: {} leaves fewer than 4 samples of \
             the {:.0} s recording. The staging spans {}; if that does not overlap [0, {:.0}] s \
             the staging and EDF are on different time bases (elapsed vs clock time) or the \
             staging file belongs to a different recording.",
            window_desc, rec_dur, stage_span, rec_dur
        )));
    }
    let t_trim_arr = Array1::from(t_trim);

    // 2. Artifact detection.
    report("detecting artifacts", 0.005);
    check_cancel()?;
    let artifacts = detect_artifacts_default(&data_trim, fs)?;
    report("artifacts done", 0.15);
    check_cancel()?;
    if std::env::var("TFPEAKS_DEBUG").is_ok() {
        let n_art = artifacts.iter().filter(|&&a| a).count();
        let n = artifacts.len().max(1);
        eprintln!(
            "[tfdbg] analyzed window [{:.1}, {:.1}] s  ({} samples @ {} Hz)  \
             artifacts {}/{} ({:.3}%)",
            t_trim_arr[0],
            t_trim_arr[t_trim_arr.len() - 1],
            data_trim.len(),
            fs,
            n_art,
            artifacts.len(),
            100.0 * n_art as f64 / n as f64,
        );
    }

    // 3. Compute pass-1 + pass-2 multitaper spectrograms with the pure-Rust
    //    DPSS solver. Detrend mode is `Constant` to match
    //    `computeTFPeaks.m:457`.
    let mtm = &settings.mtm;
    let win1_samples = (mtm.window_length_pass1 * fs).round() as usize;
    let win2_samples = (mtm.window_length_pass2 * fs).round() as usize;
    // Match MATLAB `multitaper_spectrogram_mex.m:218`:
    //   nfft = max(2^nextpow2(winsize), winsize, 2^nextpow2(min_NFFT))
    // where `min_NFFT = 2^nextpow2(Fs/dsfreqs)`. Both passes are masked
    // against each other, so they must share one freq grid; floor at the
    // longer window so zero-padding is never negative (nfft >= winsize).
    let win_max = win1_samples.max(win2_samples);
    let nfft = next_pow2(fs / mtm.dsfreqs)
        .max(next_pow2(win_max as f64))
        .max(win_max);
    let tapers1 = multitaper_rs::dpss::dpss_tapers(win1_samples, mtm.nw, mtm.n_tapers)?;
    let tapers2 = multitaper_rs::dpss::dpss_tapers(win2_samples, mtm.nw, mtm.n_tapers)?;

    // Own the trimmed signal once; both spectrogram passes and the
    // two-pass extract below borrow it as a view.
    let data_arr = Array1::from(data_trim);

    report("computing spectrogram (pass 1)", 0.15);
    check_cancel()?;
    let p1 = compute_spectrogram(
        data_arr.view(),
        tapers1.view(),
        None,
        &SpectrogramParams {
            fs,
            frequency_range: mtm.freq_range,
            window_params: (mtm.window_length_pass1, mtm.window_step),
            nfft,
            detrend: DetrendMode::Constant,
            weighting: Weighting::Unity,
        },
    )
    .map_err(|e| DynamoError::stage("spectrogram_pass1", e))?;
    report("computing spectrogram (pass 2)", 0.23);
    check_cancel()?;
    let p2 = compute_spectrogram(
        data_arr.view(),
        tapers2.view(),
        None,
        &SpectrogramParams {
            fs,
            frequency_range: mtm.freq_range,
            window_params: (mtm.window_length_pass2, mtm.window_step),
            nfft,
            detrend: DetrendMode::Constant,
            weighting: Weighting::Unity,
        },
    )
    .map_err(|e| DynamoError::stage("spectrogram_pass2", e))?;

    // Shift the stimes so they're absolute (match runDYNAMO convention —
    // computeTFPeaks.m:299: `stimes = stimes + t_time_range(1);`).
    let t0 = t_trim_arr[0];
    let stimes1: Array1<f64> = p1.stimes.iter().map(|t| t + t0).collect();
    let stimes2: Array1<f64> = p2.stimes.iter().map(|t| t + t0).collect();
    let sfreqs = p1.sfreqs.clone();

    // 4. Two-pass extract.
    let input = DynamoInput {
        data: data_arr.view(),
        fs,
        t_time_range: t_trim_arr.view(),
        stage_times,
        stage_vals,
        artifacts: &artifacts,
    };
    let pass1 = PassSpectrogram {
        spect: p1.mt_spectrogram.view(),
        stimes: stimes1.view(),
        mtm: PassMtmSpec {
            nw: mtm.nw,
            n_tapers: mtm.n_tapers,
            window_length: mtm.window_length_pass1,
        },
    };
    let pass2 = PassSpectrogram {
        spect: p2.mt_spectrogram.view(),
        stimes: stimes2.view(),
        mtm: PassMtmSpec {
            nw: mtm.nw,
            n_tapers: mtm.n_tapers,
            window_length: mtm.window_length_pass2,
        },
    };
    report("extracting TF peaks (two-pass watershed)", 0.31);
    check_cancel()?;
    // Poll the host cancel flag inside extract_tfpeaks' per-segment loop so
    // Stop is responsive during the ~30 s watershed, not just at this stage
    // boundary.
    let cancel_poll = move || cancel.is_some_and(|c| c());
    // Staged path: prepare (pass-1 watershed + pass-2 mask) returns the
    // masked pass-2 spectrogram and the few scalars finalize needs.
    // After that we drop the pass-1 spectrogram views + storage before
    // the pass-2 watershed allocates its working buffers. Keeping p1
    // alive through the entire pass-2 watershed would cost ~2.5 GB more
    // per worker on a CFS-sized cohort. The split point 0.546 is
    // `fmid = flo + (fhi - flo) * 0.4` for (0.31, 0.90), the same split
    // `run_extract_two_pass_with_progress` uses.
    let prep_res = prepare_pass2_input(
        &input,
        &pass1,
        &pass2,
        sfreqs.view(),
        settings,
        Some(&cancel_poll),
        progress,
        (0.31, 0.546),
    );
    check_cancel()?;
    let prep = prep_res?;
    // pass1 holds views into p1 + stimes1; its borrow ends at its last
    // use above, so we can free the underlying spectrogram storage here.
    drop(p1);
    drop(stimes1);
    let two_pass_res = extract_pass2_and_finalize(
        &input,
        prep,
        &pass2,
        sfreqs.view(),
        mtm.freq_range,
        settings,
        Some(&cancel_poll),
        progress,
        (0.546, 0.90),
    );
    // A cancel mid-extract comes back as a stage error; classify it as
    // Cancelled before the real-error path.
    check_cancel()?;
    let mut two_pass = two_pass_res?;
    report("TF peaks extracted", 0.90);
    check_cancel()?;

    // The extraction leaves two arrays as large as the whole
    // time-frequency grid: the region label image and the pass-2
    // spectrogram. No later stage reads either one, so they are released
    // here and not held through the histograms, the fits and the export.
    two_pass.label_image = Array2::zeros((0, 0));
    let spectrogram_pass2: Array2<f64> = if keep_pass2_spectrogram {
        p2.mt_spectrogram
    } else {
        drop(p2);
        Array2::zeros((0, 0))
    };

    // Per-peak stage + analysis mask — computed up front so the SOPH
    // histogram calls can pre-filter peaks the same way MATLAB
    // `SOpowerHistogram.m:240`/`:276` does (drop peaks outside SOPH_stages
    // / time_range BEFORE binning). Without this, WAKE/REM peaks leak
    // into the NREM-only SOPH and the leftmost SO-power bin lights up.
    let t_time_range_vec_for_stage: Vec<f64> = t_trim_arr.iter().copied().collect();
    // Per-peak stage from the kernel (MATLAB `computePeakStage.m`):
    // `interp1(stage_times, stage_vals, PeakTime, 'previous')` with peaks
    // outside the scored range set to 0 (unknown), then stage 6 for every
    // peak whose NEAREST EEG sample is flagged as an artifact
    // (`interp1(t_artifacts, artifacts, PeakTime, 'nearest')`).
    let peak_stage: Vec<u8> = {
        let stages_at_peaks = crate::peak_assign::assign_peak_stage(
            &two_pass.stats.peak_time,
            stage_times,
            stage_vals,
            &t_time_range_vec_for_stage,
            &artifacts,
        );
        let mut stage_codes: Vec<u8> = Vec::with_capacity(stages_at_peaks.len());
        for stage in stages_at_peaks {
            stage_codes.push((stage as i32).clamp(0, 6) as u8);
        }
        stage_codes
    };
    let soph_stages_u8: Vec<u8> = settings.hist.soph_stages.clone();
    // MATLAB `peak_selection_inds` carries no explicit frequency test; the
    // frequency restriction is implicit in the histogram's freq bin edges
    // (`SOpowerphasehist_opts.freq_range`). A peak therefore appears in the
    // histogram only if it also falls within `hist.freq_range` — NOT the
    // wider spectrogram range (`mtm.freq_range`).
    let (hist_fa, hist_fb) = settings.hist.freq_range;
    let (tr_a, tr_b) = match time_range_resolved {
        Some((a, b)) => (a, b),
        None => (f64::NEG_INFINITY, f64::INFINITY),
    };
    // `peak_in_analysis`: stage ∈ SOPH_stages AND in time_range AND in the
    // histogram freq range — the MATLAB `peak_selection_inds` combined with
    // the histogram freq binning (minus the NaN-metric check, which is added
    // per-histogram below).
    let peak_in_analysis: Vec<bool> = two_pass.stats.peak_time.iter()
        .zip(two_pass.stats.peak_freq.iter())
        .zip(peak_stage.iter())
        .map(|((&pt, &pf), &st)| {
            let stage_ok = if soph_stages_u8.is_empty() { true } else { soph_stages_u8.contains(&st) };
            stage_ok && pt >= tr_a && pt <= tr_b && pf >= hist_fa && pf <= hist_fb
        })
        .collect();

    // 5. SO-power: compute the SO-band multitaper spectrogram, then run
    //    so_power_from_spectrogram. Different (NW, K) than the main pass.
    report("computing SO-power", 0.90);
    check_cancel()?;
    let (sop_nw, sop_n_tapers) = settings.sopower.tapers;
    let (sop_window_len, sop_window_step) = settings.sopower.window_params;
    let sop_band = settings.sopower.band;
    let sop_win_samples = (sop_window_len * fs).round() as usize;
    let tapers_sop = multitaper_rs::dpss::dpss_tapers(sop_win_samples, sop_nw, sop_n_tapers)?;
    // MATLAB `computeSOpower.m:253` calls `multitaper_spectrogram_dynamo`
    // with `'linear'` detrend AND lets the helper pick `nfft` per its own
    // default policy: `max(2^nextpow2(winN), 256)`. For a 5 s × 100 Hz
    // window that's 512, not the main-pipeline nfft (1024). Using a
    // different nfft shifts the freq-bin edges in the SO-power
    // integration band ([0.3, 1.5] Hz) and visibly broadens the SOpower
    // distribution tails. Match MATLAB by computing the SO-specific
    // nfft here.
    let sop_nfft = {
        let mut n = 256usize;
        let win_n = (sop_window_len * fs).round() as usize;
        while n < win_n { n <<= 1; }
        n
    };
    // **MATLAB `computeSOpower.m:116-117` pre-NaN's the EEG at artifact
    // samples BEFORE the multitaper compute.** Any window containing a
    // NaN sample produces NaN spectrogram → NaN SOpower, so artifact
    // windows are excluded from the SOpower distribution. Without this
    // step the artifact regions contribute extreme values to the tails
    // (min/max ≈ ±28 dB vs MATLAB's ±23 dB).
    let data_for_sop: Vec<f64> = data_arr.iter().zip(artifacts.iter())
        .map(|(&v, &is_art)| if is_art { f64::NAN } else { v })
        .collect();
    let p_sop = compute_spectrogram(
        Array1::from(data_for_sop).view(),
        tapers_sop.view(),
        None,
        &SpectrogramParams {
            fs,
            frequency_range: sop_band,
            window_params: (sop_window_len, sop_window_step),
            nfft: sop_nfft,
            detrend: DetrendMode::Linear,
            weighting: Weighting::Unity,
        },
    )
    .map_err(|e| DynamoError::stage("spectrogram_sopower", e))?;

    // so_power_from_spectrogram wants row-major f64 + dimensions. Convert
    // the ndarray. Times stay in spectrogram-local frame (relative to
    // data[0]) — the function shifts internally by `eeg_times[0]`.
    let sop_spect_owned = p_sop.mt_spectrogram.to_owned();
    let (sop_nf, sop_nt) = sop_spect_owned.dim();
    let sop_spect_vec: Vec<f64> = sop_spect_owned.iter().copied().collect();
    let sop_stimes_vec: Vec<f64> = p_sop.stimes.to_vec();
    let sop_sfreqs_vec: Vec<f64> = p_sop.sfreqs.to_vec();
    let eeg_times_vec: Vec<f64> = t_trim_arr.iter().copied().collect();
    let norm = crate::so_power::NormMethod::parse(&settings.sopower.norm_method)
        .ok_or_else(|| {
            DynamoError::Invalid(format!(
                "unknown norm_method {:?}",
                settings.sopower.norm_method
            ))
        })?;
    let time_range_sop = (
        *eeg_times_vec.first().unwrap_or(&0.0),
        *eeg_times_vec.last().unwrap_or(&0.0),
    );
    let so_power_out = crate::so_power::so_power_from_spectrogram(
        &sop_spect_vec,
        sop_nf,
        sop_nt,
        &sop_stimes_vec,
        &sop_sfreqs_vec,
        &eeg_times_vec,
        &artifacts,
        stage_times,
        stage_vals,
        time_range_sop,
        settings.sopower.outlier_threshold,
        &norm,
        settings.sopower.retain_fs,
    )
    .map_err(|e| DynamoError::stage("so_power_from_spectrogram", e))?;

    // Native-grid SO-power for the compact aux artifact. When retain_fs is
    // on, `so_power_out` is upsampled to EEG rate (50× larger, no extra
    // information — pure linear interp). The aux file only needs the native
    // 0.5 s compute grid: lossless vs the computed coarse series and ~50×
    // smaller. Recover it with a second cheap normalization pass at
    // retain_fs=false (the expensive SO-band spectrogram is already done and
    // reused by reference); when retain_fs is already off, the first call
    // *is* the native series, so reuse it. The per-peak interpolation and
    // the in-memory SOPH keep using `so_power_out`, so this extra pass does
    // not change them.
    let (so_power_norm_native, so_power_native_t_start) = if settings.sopower.retain_fs {
        let native = crate::so_power::so_power_from_spectrogram(
            &sop_spect_vec,
            sop_nf,
            sop_nt,
            &sop_stimes_vec,
            &sop_sfreqs_vec,
            &eeg_times_vec,
            &artifacts,
            stage_times,
            stage_vals,
            time_range_sop,
            settings.sopower.outlier_threshold,
            &norm,
            false,
        )
        .map_err(|e| DynamoError::stage("so_power_native", e))?;
        let t0 = native.so_power_times.first().copied().unwrap_or(0.0);
        (native.so_power_norm, t0)
    } else {
        let t0 = so_power_out.so_power_times.first().copied().unwrap_or(0.0);
        (so_power_out.so_power_norm.clone(), t0)
    };

    // 6. SO-phase: load/design SOS filter, then so_phase_from_eeg.
    report("computing SO-phase", 0.91);
    check_cancel()?;
    // SO-phase uses the *same* SO band as SO-power: MATLAB
    // `SOpowerphasehist_opts.SO_freqrange` is a single option, so both read
    // from one field here rather than two that could drift apart.
    let sop_phase_band = settings.sopower.band;
    let sos = crate::filter_cache::get_sophase_sos(fs, sop_phase_band)
        .map_err(|e| DynamoError::stage("sophase_sos", format!("{}", e)))?;
    let sos_rows: Vec<[f64; 6]> = (0..sos.nrows())
        .map(|i| {
            [
                sos[[i, 0]], sos[[i, 1]], sos[[i, 2]],
                sos[[i, 3]], sos[[i, 4]], sos[[i, 5]],
            ]
        })
        .collect();
    // `data_arr` is built with `Array1::from(Vec<f64>)`, so it is always
    // contiguous and `.as_slice()` returns `Some`. A typed error is used
    // instead of `.unwrap()` so that a refactor which breaks that
    // assumption (e.g. taking a view instead of an owned Array1) reports
    // an error instead of crashing the Tauri command thread.
    let data_slice = data_arr.as_slice().ok_or_else(|| {
        DynamoError::Invalid(
            "internal: data_arr expected to be contiguous (Array1::from(Vec))".into(),
        )
    })?;
    let so_phase_out = crate::so_phase::so_phase_from_eeg(
        data_slice,
        &eeg_times_vec,
        &artifacts,
        &sos_rows,
        stage_times,
        stage_vals,
    )
    .map_err(|e| DynamoError::stage("so_phase_from_eeg", e))?;

    // 7. Per-peak SO-power and SO-phase from the kernel, which samples
    // each series at the peak time exactly as MATLAB does
    // (`computePeakSOpower.m`, `computePeakSOphase.m`). The phase is
    // interpolated on the UNWRAPPED series and wrapped afterwards, so a peak
    // between two samples that straddle the ±π seam stays near ±π.
    let peak_times: Vec<f64> = two_pass.stats.peak_time.clone();
    let peak_sopower = crate::peak_assign::assign_peak_sopower(
        &peak_times,
        &so_power_out.so_power_times,
        &so_power_out.so_power_norm,
    );
    let peak_sophase: Vec<f64> = crate::peak_assign::assign_peak_sophase(
        &peak_times,
        &so_phase_out.so_phase_times,
        &so_phase_out.so_phase_unwrapped,
    );
    // The wrapped series itself (on the SO-phase time grid, which may differ
    // from so_power_times) feeds the histogram's time-in-bin metric below.
    let mut so_phase_wrapped_own: Vec<f64> =
        Vec::with_capacity(so_phase_out.so_phase_unwrapped.len());
    for unwrapped_phase in so_phase_out.so_phase_unwrapped.iter() {
        so_phase_wrapped_own.push(crate::run::soph::wrap_to_pi(*unwrapped_phase));
    }

    // ── Phase 1 early-exit (two-phase batch driver) ──
    // Compute the summary spec PNG's source matrix here so it lands in
    // aux.h5, then return with empty SOPHs/fits. Phase 2 rebuilds the
    // downstream stages from the on-disk cache.
    if let Some(StopAfter::Peaks) = stop_after {
        report("phase 1: computing summary spectrogram", 0.95);
        check_cancel()?;
        let (summary_spectrogram, summary_sfreqs, summary_stimes) =
            crate::run::soph_build::compute_summary_spectrogram(
                data_arr.view(),
                fs,
                t0,
                Some(&|m: &str, f: f64| report(m, f)),
            )?;
        report("phase 1 complete", 0.98);
        let t_time_range_out: Vec<f64> = t_trim_arr.iter().copied().collect();
        return Ok(DynamoOutputs {
            stats: two_pass.stats,
            baseline: two_pass.baseline,
            baseline_freqs: sfreqs.to_vec(),
            artifacts,
            fs,
            t_time_range: t_time_range_out,
            stage_times: stage_times.to_vec(),
            stage_vals: stage_vals.to_vec(),
            so_power_norm: so_power_out.so_power_norm,
            so_power_times: so_power_out.so_power_times,
            so_power_norm_native,
            so_power_native_t_start,
            so_phase_unwrapped: so_phase_out.so_phase_unwrapped,
            so_phase_times: so_phase_out.so_phase_times,
            peak_sopower,
            peak_sophase,
            peak_stage,
            peak_in_hist: peak_in_analysis,
            sopower_hist: Array2::zeros((0, 0)),
            hist_freq_centers: Vec::new(),
            sopower_bin_centers: Vec::new(),
            sophase_hist: Array2::zeros((0, 0)),
            sophase_bin_centers: Vec::new(),
            spectrogram: Array2::zeros((0, 0)),
            spect_sfreqs: Vec::new(),
            spect_stimes: Vec::new(),
            summary_spectrogram,
            summary_sfreqs,
            summary_stimes,
            summary_spec_cached: None,
            paramfit_power: None,
            paramfit_phase: None,
            paramfit_power_skipped: None,
            paramfit_phase_skipped: None,
            splinefit_power: None,
            splinefit_phase: None,
        });
    }

    // 8. Build SOPH histograms + summary spectrogram + fits. Shared with the
    //    cache-rebuild path via `soph_build::build_soph_outputs`, so the two
    //    can't drift — only the inputs differ (here: in-memory series).
    report("building SO-power/phase histograms", 0.94);
    check_cancel()?;
    let soph = crate::run::soph_build::build_soph_outputs(
        &crate::run::soph_build::SophBuildInputs {
            peak_freq: &two_pass.stats.peak_freq,
            peak_in_analysis: &peak_in_analysis,
            peak_sopower: &peak_sopower,
            peak_sophase: &peak_sophase,
            so_power_norm: &so_power_out.so_power_norm,
            so_power_times: &so_power_out.so_power_times,
            so_phase_metric: &so_phase_wrapped_own,
            so_phase_times: &so_phase_out.so_phase_times,
            stage_times,
            stage_vals,
            fs,
            t0,
            settings,
        },
        Some(data_arr.view()),
        Some(&|m: &str, f: f64| report(m, f)),
        false, // full pipeline always computes the summary spectrogram
    )?;

    // Per-stage SOPH coverage — useful for diagnosing empty histograms
    // (sparse staging or too-strict `min_time_in_bin`).
    {
        // `time_in_bin` has one row per SO-power bin and one column per
        // stage code 1..5 (N3, N2, N1, REM, Wake), in minutes.
        let tib = &soph.sopower_time_in_bin;
        let mut per_stage = [0.0f64; 5];
        let mut max_tib_sum = 0.0;
        for s in 0..tib.nrows() {
            let mut sum_row = 0.0;
            for k in 0..5 {
                per_stage[k] += tib[[s, k]];
                sum_row += tib[[s, k]];
            }
            if sum_row > max_tib_sum {
                max_tib_sum = sum_row;
            }
        }
        log::info!(
            "SOPH stage coverage n3={} n2={} n1={} rem={} wake={} max_bin_min={} threshold_min={}",
            per_stage[0], per_stage[1], per_stage[2], per_stage[3], per_stage[4],
            max_tib_sum, settings.hist.min_time_in_bin,
        );
    }

    let spect_sfreqs: Vec<f64> = sfreqs.iter().copied().collect();
    let spect_stimes: Vec<f64> = stimes2.iter().copied().collect();

    // `peak_in_hist` re-exports `peak_in_analysis` under the public-API name so
    // summary scatter (and future consumers) can filter peaks.
    let peak_in_hist: Vec<bool> = peak_in_analysis.clone();
    let sopower_hist = soph.sopower_hist;
    let sophase_hist = soph.sophase_hist;
    let hist_freq_centers = soph.hist_freq_centers;
    let sopower_bin_centers = soph.sopower_bin_centers;
    let sophase_bin_centers = soph.sophase_bin_centers;
    let summary_spectrogram = soph.summary_spectrogram;
    let summary_sfreqs = soph.summary_sfreqs;
    let summary_stimes = soph.summary_stimes;
    let paramfit_power = soph.paramfit_power;
    let paramfit_phase = soph.paramfit_phase;
    let paramfit_power_skipped = soph.paramfit_power_skipped;
    let paramfit_phase_skipped = soph.paramfit_phase_skipped;
    let splinefit_power = soph.splinefit_power;
    let splinefit_phase = soph.splinefit_phase;

    report("pipeline complete", 0.98);
    let t_time_range_out: Vec<f64> = t_trim_arr.iter().copied().collect();
    Ok(DynamoOutputs {
        stats: two_pass.stats,
        baseline: two_pass.baseline,
        baseline_freqs: spect_sfreqs.clone(),
        artifacts,
        fs,
        t_time_range: t_time_range_out,
        stage_times: stage_times.to_vec(),
        stage_vals: stage_vals.to_vec(),
        so_power_norm: so_power_out.so_power_norm,
        so_power_times: so_power_out.so_power_times,
        so_power_norm_native,
        so_power_native_t_start,
        so_phase_unwrapped: so_phase_out.so_phase_unwrapped,
        so_phase_times: so_phase_out.so_phase_times,
        peak_sopower,
        peak_sophase,
        peak_stage,
        peak_in_hist,
        sopower_hist,
        hist_freq_centers,
        sopower_bin_centers,
        sophase_hist,
        sophase_bin_centers,
        spectrogram: spectrogram_pass2,
        spect_sfreqs,
        spect_stimes,
        summary_spectrogram,
        summary_sfreqs,
        summary_stimes,
        paramfit_power,
        paramfit_phase,
        paramfit_power_skipped,
        paramfit_phase_skipped,
        splinefit_power,
        splinefit_phase,
        summary_spec_cached: None,
    })
}

/// Run artifact detection with DYNAM-O defaults — convenience wrapper.
///
/// This is MATLAB `detect_artifacts(data, Fs)` with its defaults, which
/// include the slope test. The whole detector lives in the kernel
/// (`crate::artifacts::detect_artifacts_with_slope_test`): the slope
/// test runs first and its mask is folded into the bad-sample set before
/// the high-frequency and broadband tests.
///
/// A detector failure is the subject's error: the mask feeds the baseline,
/// the SO-power series, and the surviving peak set.
pub fn detect_artifacts_default(data: &[f64], fs: f64) -> Result<Vec<bool>, DynamoError> {
    let opts = ArtifactOpts {
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
    };
    // MATLAB `detect_artifacts` default: a window is flagged when the slope
    // of its log-log spectrum is above this value.
    let slope_crit = -0.5;
    detect_artifacts_with_slope_test(data, fs, &opts, slope_crit)
        .map_err(|e| DynamoError::stage("artifact_detection", e))
}

#[cfg(test)]
mod height_floor_tests {
    use super::{chi2inv_even_dof, PassMtmSpec};

    /// `(tapers, chi2inv(0.975, 2 * tapers), height floor in dB)` from
    /// MATLAB R2025b: `chi2inv(0.975, 2*K)` and
    /// `-pow2db(2*K / chi2inv(0.975, 2*K)) * 2`. The digits are MATLAB's
    /// `%.17g` output, kept as printed.
    #[allow(clippy::excessive_precision)]
    const MATLAB_REFERENCE: [(usize, f64, f64); 10] = [
        (1, 7.3777589082278707, 11.337889270063215),
        (2, 11.143286781877794, 8.899066325378385),
        (3, 14.449375335447918, 7.6339564404318025),
        (4, 17.53454613948465, 6.8160908436751697),
        (5, 20.483177350807395, 6.2279465059309587),
        (7, 26.118948045037367, 5.4165529165699127),
        (9, 31.526378440386623, 4.8680315877471685),
        (15, 46.979242243671152, 3.8956950556533911),
        (29, 80.9355918865364, 2.8942310704289866),
        (60, 152.21140272515149, 2.0653188445737669),
    ];

    #[test]
    fn chi2_quantile_matches_matlab_for_every_taper_count() {
        for &(tapers, matlab_quantile, _) in MATLAB_REFERENCE.iter() {
            let quantile = chi2inv_even_dof(0.975, tapers);
            let relative_difference = ((quantile - matlab_quantile) / matlab_quantile).abs();
            // The bisection converges to double precision, so the quantile
            // agrees to a few units in the last place.
            assert!(
                relative_difference < 1e-12,
                "{} tapers: {} vs MATLAB {}",
                tapers, quantile, matlab_quantile
            );
        }
    }

    #[test]
    fn height_floor_matches_matlab_for_every_taper_count() {
        for &(tapers, _, matlab_floor_db) in MATLAB_REFERENCE.iter() {
            let spec = PassMtmSpec { nw: 2.0, n_tapers: tapers, window_length: 2.0 };
            let floor_db = spec.ht_db_min();
            // Absolute tolerance in dB. The floor is a log10 of the quantile
            // ratio, which costs about one more digit than the quantile.
            assert!(
                (floor_db - matlab_floor_db).abs() < 1e-11,
                "{} tapers: {} dB vs MATLAB {} dB",
                tapers, floor_db, matlab_floor_db
            );
        }
    }
}

#[cfg(test)]
mod scored_window_tests {
    use super::scored_window;

    #[test]
    fn picks_first_to_last_scored_excluding_undef_and_artifact() {
        let times = [0.0, 30.0, 60.0, 90.0, 120.0, 150.0];
        let vals = [0.0, 6.0, 2.0, 3.0, 6.0, 0.0];
        assert_eq!(scored_window(&times, &vals), Some((60.0, 90.0)));
    }

    #[test]
    fn none_when_no_scored_or_single_epoch() {
        assert_eq!(scored_window::<f64>(&[0.0, 30.0], &[0.0, 6.0]), None);
        assert_eq!(scored_window(&[42.0], &[2.0f64]), None);
    }

    #[test]
    fn integer_stage_vals_agree_with_f64() {
        let times = [0.0, 30.0, 60.0];
        assert_eq!(
            scored_window(&times, &[6i8, 1, 5]),
            scored_window(&times, &[6.0f64, 1.0, 5.0]),
        );
    }
}
