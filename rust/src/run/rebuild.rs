//! Rebuild SOPHs + parametric/spline fits + the summary figure's display
//! spectrogram from a prior run's on-disk cache (`stats.csv` + `aux.h5`),
//! WITHOUT re-detecting TF-peaks or recomputing SO-power/SO-phase.
//!
//! The histogram → summary-spectrogram → fits glue is **shared with the full
//! run** via [`crate::run::soph_build::build_soph_outputs`] — both paths call it, so
//! they cannot drift. This module only prepares the cache-specific
//! inputs (peaks from `stats.csv`, the native 0.5 s SO-power series, and an
//! EMPTY SO-phase series) and assembles `DynamoOutputs`.
//!
//! The cache stores SO-power on its native window grid. With
//! `sopower.retain_fs` (the default) the rebuild brings it to the EEG sample
//! grid exactly as the full run and MATLAB `computeSOpower.m` do
//! ([`crate::so_power::upsample_to_eeg_rate`]: interpolate through the finite
//! windows, then NaN the artifact samples), so the SO-power time-in-bin, and
//! with it the SOPH, matches a full run. The EEG grid is the scored window
//! ([`cache_time_grid`]) and the artifact mask comes from the cached spans. The
//! SO-phase histogram is built from per-peak values alone with an EMPTY
//! series: `build_soph_outputs` runs the phase axis with `compute_rate=false`
//! / `min_time_in_bin=0`, so its `c_mat` never reads the series. Per-peak
//! SO-power/SO-phase come straight from `stats.csv`.

use ndarray::{Array1, Array2, ArrayView1};

use crate::run::orchestrator::{DynamoError, DynamoOutputs};
use crate::run::settings::DynamoSettings;
use crate::extract_pipeline::SegmentPeaks;

/// Inputs for [`rebuild_from_cache`] — plain borrowed slices so this
/// crate stays free of a dependency on `dynamo-export` (which depends on
/// `dynamo-pipeline`; the reverse would be a cycle). The jobs layer
/// reads `StatsBundle` / `AuxBundle` and fills this in.
pub struct RebuildInputs<'a> {
    // ── Peaks (from stats.csv) ──
    pub peak_time: &'a [f64],
    pub peak_freq: &'a [f64],
    pub duration: &'a [f64],
    pub bandwidth: &'a [f64],
    pub height: &'a [f64],
    pub volume: &'a [f64],
    pub segment_num: &'a [i64],
    pub bbox_tl_s: &'a [f64],
    pub bbox_tl_hz: &'a [f64],
    pub bbox_width_s: &'a [f64],
    pub bbox_height_hz: &'a [f64],
    pub area: &'a [f64],
    pub peakiness: &'a [f64],
    pub peak_stage: &'a [u8],
    pub peak_sopower: &'a [f64],
    pub peak_sophase: &'a [f64],
    // ── Baseline spectrum (from aux.h5; empty for caches written
    //    before it was persisted — PkPower is then NaN) ──
    pub baseline: &'a [f64],
    pub baseline_freqs: &'a [f64],
    // ── SO-power native series (from aux.h5) ──
    pub so_power_norm: &'a [f64],
    pub so_power_t_start: f64,
    pub so_power_step: f64,
    // ── Stages + artifacts (from aux.h5) ──
    pub stage_times: &'a [f64],
    pub stage_vals: &'a [u8],
    pub artifact_spans: &'a [[f64; 2]],
    // ── EEG channel for the summary spectrogram ──
    /// Resampled channel data fed to the summary-spectrogram compute.
    /// `None` for the EDF-free Phase-2 rebuild path: the caller has no
    /// signal in hand, only `aux.h5` + `stats.csv`, and the cached PNG
    /// from `aux.h5` substitutes for re-computing the spec. When `None`,
    /// `skip_summary_spec` MUST be true and `n_samp` MUST be set to the
    /// length the resampled signal would have had.
    pub data_arr: Option<ArrayView1<'a, f64>>,
    pub fs: f64,
    pub t0: f64,
    /// Length of the (would-be) resampled signal, in samples. Used only
    /// when `data_arr` is `None`; ignored otherwise (`data_arr.len()`
    /// wins). Drives the per-sample artifact mask and `t_time_range`
    /// length downstream consumers expect.
    pub n_samp: usize,
    /// When true, skip the multitaper summary-spectrogram compute and
    /// leave `DynamoOutputs.summary_spectrogram` empty. The renderer
    /// then loads a pre-rendered spec PNG from disk if available. Only
    /// the spec compute is gated; the time axis and artifact mask still
    /// land from `fs` / `t0` / `n_samp` (or `data_arr.len()`) as normal.
    pub skip_summary_spec: bool,
    // ── Config ──
    pub settings: &'a DynamoSettings,
}

/// Build a `SegmentPeaks` from the cached stats columns. `height_data`
/// and `boundaries_xy` are left empty — they're not in `stats.csv` and
/// are only read by the stats-CSV writer (off in the cache path) and not
/// by the histogram or summary figure.
pub fn segment_peaks_from_stats(inp: &RebuildInputs) -> SegmentPeaks {
    let n = inp.peak_time.len();
    let mut bbox = Vec::with_capacity(n * 4);
    for i in 0..n {
        bbox.push(inp.bbox_tl_s[i]);
        bbox.push(inp.bbox_tl_hz[i]);
        bbox.push(inp.bbox_width_s[i]);
        bbox.push(inp.bbox_height_hz[i]);
    }
    SegmentPeaks {
        peak_time: inp.peak_time.to_vec(),
        peak_freq: inp.peak_freq.to_vec(),
        duration: inp.duration.to_vec(),
        bandwidth: inp.bandwidth.to_vec(),
        height: inp.height.to_vec(),
        volume: inp.volume.to_vec(),
        segment_num: inp.segment_num.iter().map(|&s| s as f64).collect(),
        bbox,
        area: inp.area.to_vec(),
        peakiness: inp.peakiness.to_vec(),
        height_data: Vec::new(),
        boundaries_xy: Vec::new(),
    }
}

/// Collapse a per-sample artifact bool mask into `[start_s, end_s]`
/// runs. `end_s` is exclusive (the time of the first non-artifact
/// sample after the run), matching how the summary plot draws spans.
pub fn artifact_mask_to_spans(mask: &[bool], fs: f64) -> Vec<[f64; 2]> {
    let mut spans: Vec<[f64; 2]> = Vec::new();
    if fs <= 0.0 {
        return spans;
    }
    let mut run_start: Option<usize> = None;
    for (i, &is_art) in mask.iter().enumerate() {
        match (is_art, run_start) {
            (true, None) => run_start = Some(i),
            (false, Some(start)) => {
                spans.push([start as f64 / fs, i as f64 / fs]);
                run_start = None;
            }
            _ => {}
        }
    }
    // A run still open at the end of the mask closes at the mask length.
    if let Some(start) = run_start {
        spans.push([start as f64 / fs, mask.len() as f64 / fs]);
    }
    spans
}

/// Expand `[start_s, end_s]` artifact runs to a per-sample bool mask of
/// length `n` at sample rate `fs`. Inverse of `artifact_mask_to_spans`
/// in `dynamo-export`. The summary figure reads a per-sample mask.
fn artifact_spans_to_mask(spans: &[[f64; 2]], n: usize, fs: f64) -> Vec<bool> {
    let mut mask = vec![false; n];
    if fs <= 0.0 {
        return mask;
    }
    for s in spans {
        let lo = ((s[0] * fs).round() as isize).max(0) as usize;
        let hi = ((s[1] * fs).round() as isize).max(0) as usize;
        // Flag samples `lo..hi`, with both ends clamped to the mask length.
        for m in mask.iter_mut().take(hi.min(n)).skip(lo.min(n)) {
            *m = true;
        }
    }
    mask
}

/// The EEG sample grid a full run analyzes when no time range is given:
/// `(t0, n_samp)` for the samples `i` with `i / fs` inside the scored window
/// (first to last epoch staged 1-5, both ends included), the same samples
/// the orchestrator's time-range truncation keeps. `(0.0, 0)` when nothing
/// is scored. Callers of [`rebuild_from_cache`] without the signal in hand
/// pass these as `t0` / `n_samp`.
pub fn cache_time_grid(stage_times: &[f64], stage_vals: &[u8], fs: f64) -> (f64, usize) {
    let (window_start, window_end) = match crate::run::orchestrator::scored_window(stage_times, stage_vals) {
        Some(window) => window,
        None => return (0.0, 0),
    };
    if fs <= 0.0 {
        return (0.0, 0);
    }
    // The first and last sample index inside [window_start, window_end],
    // tested with the same `i / fs` comparison the truncation uses.
    let mut first = (window_start * fs).floor().max(0.0) as usize;
    while (first as f64) / fs < window_start {
        first += 1;
    }
    let mut last = (window_end * fs).ceil().max(0.0) as usize;
    while last > first && (last as f64) / fs > window_end {
        last -= 1;
    }
    if (last as f64) / fs > window_end || last < first {
        return (0.0, 0);
    }
    (first as f64 / fs, last - first + 1)
}

pub fn rebuild_from_cache(inp: &RebuildInputs) -> Result<DynamoOutputs, DynamoError> {
    // EDF-free contract: when caller has no signal, the cached PNG must
    // be the figure's spec source; we won't be running the multitaper.
    if inp.data_arr.is_none() && !inp.skip_summary_spec {
        return Err(DynamoError::Invalid(
            "rebuild_from_cache: data_arr=None requires skip_summary_spec=true"
                .into(),
        ));
    }
    let stats = segment_peaks_from_stats(inp);
    let stage_vals_f: Vec<f64> = inp.stage_vals.iter().map(|&v| v as f64).collect();

    // SO-power native window grid: t_start + i*step.
    let so_power_native_times: Vec<f64> = (0..inp.so_power_norm.len())
        .map(|i| inp.so_power_t_start + i as f64 * inp.so_power_step)
        .collect();

    // The EEG sample grid and its artifact mask. `data_arr.len()` when the
    // caller has the signal, the caller's `n_samp` otherwise. Sample times
    // are `(first_index + k) / fs`, as the full run computes them.
    let n_samp = inp.data_arr.map(|a| a.len()).unwrap_or(inp.n_samp);
    let first_index = (inp.t0 * inp.fs).round();
    let t_time_range: Vec<f64> = (0..n_samp)
        .map(|k| (first_index + k as f64) / inp.fs)
        .collect();
    let artifacts = artifact_spans_to_mask(inp.artifact_spans, n_samp, inp.fs);

    // The SO-power series the SOPH and the figure read: on the EEG grid
    // when `retain_fs` (as the full run and MATLAB), else the window grid.
    let mut so_power_norm: Vec<f64> = inp.so_power_norm.to_vec();
    let mut so_power_times: Vec<f64> = so_power_native_times.clone();
    if inp.settings.sopower.retain_fs && n_samp > 0 {
        if let Some(upsampled) = crate::so_power::upsample_to_eeg_rate(
            inp.so_power_norm,
            &so_power_native_times,
            &t_time_range,
            &artifacts,
        ) {
            so_power_norm = upsampled;
            so_power_times = t_time_range.clone();
        }
    }

    // `peak_in_analysis` — stage ∈ SOPH_stages AND freq ∈ mtm.freq_range. Time
    // range is unbounded (the cache path doesn't persist a sub-range; peaks
    // were already detected within the recording).
    let (mtm_fa, mtm_fb) = inp.settings.mtm.freq_range;
    let soph_stages_u8 = &inp.settings.hist.soph_stages;
    let peak_in_analysis: Vec<bool> = inp
        .peak_freq
        .iter()
        .zip(inp.peak_stage.iter())
        .map(|(&pf, &st)| {
            let stage_ok = soph_stages_u8.is_empty() || soph_stages_u8.contains(&st);
            stage_ok && pf >= mtm_fa && pf <= mtm_fb
        })
        .collect();

    // Shared SOPH builder — single source of truth with the full run. The
    // cache path feeds an EMPTY SO-phase series (see the module note).
    // Pass-through: cache-rebuild callers set this to true when a cached
    // spec PNG is available on disk for the renderer to load.
    let skip_summary_spec = inp.skip_summary_spec;
    let soph = crate::run::soph_build::build_soph_outputs(
        &crate::run::soph_build::SophBuildInputs {
            peak_freq: inp.peak_freq,
            peak_in_analysis: &peak_in_analysis,
            peak_sopower: inp.peak_sopower,
            peak_sophase: inp.peak_sophase,
            so_power_norm: &so_power_norm,
            so_power_times: &so_power_times,
            so_phase_metric: &[],
            so_phase_times: &[],
            stage_times: inp.stage_times,
            stage_vals: &stage_vals_f,
            fs: inp.fs,
            t0: inp.t0,
            settings: inp.settings,
        },
        inp.data_arr,
        None,
        skip_summary_spec,
    )?;

    // ── Assemble DynamoOutputs ──

    Ok(DynamoOutputs {
        stats,
        baseline: Array1::from(inp.baseline.to_vec()),
        baseline_freqs: inp.baseline_freqs.to_vec(),
        artifacts,
        fs: inp.fs,
        t_time_range,
        stage_times: inp.stage_times.to_vec(),
        stage_vals: stage_vals_f,
        so_power_norm,
        so_power_times,
        so_power_norm_native: inp.so_power_norm.to_vec(),
        so_power_native_t_start: inp.so_power_t_start,
        so_phase_unwrapped: Vec::new(),
        so_phase_times: Vec::new(),
        peak_sopower: inp.peak_sopower.to_vec(),
        peak_sophase: inp.peak_sophase.to_vec(),
        peak_stage: inp.peak_stage.to_vec(),
        peak_in_hist: peak_in_analysis,
        sopower_hist: soph.sopower_hist,
        hist_freq_centers: soph.hist_freq_centers,
        sopower_bin_centers: soph.sopower_bin_centers,
        sophase_hist: soph.sophase_hist,
        sophase_bin_centers: soph.sophase_bin_centers,
        spectrogram: Array2::zeros((0, 0)),
        spect_sfreqs: Vec::new(),
        spect_stimes: Vec::new(),
        summary_spectrogram: soph.summary_spectrogram,
        summary_sfreqs: soph.summary_sfreqs,
        summary_stimes: soph.summary_stimes,
        paramfit_power: soph.paramfit_power,
        paramfit_phase: soph.paramfit_phase,
        paramfit_power_skipped: soph.paramfit_power_skipped,
        paramfit_phase_skipped: soph.paramfit_phase_skipped,
        splinefit_power: soph.splinefit_power,
        splinefit_phase: soph.splinefit_phase,
        // Cache-rebuild callers in cli/src/main.rs populate this after
        // we return — they have the freshly-read AuxBundle in hand.
        summary_spec_cached: None,
    })
}
