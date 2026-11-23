//! Shared SO-power / SO-phase histogram + summary-spectrogram + fits builder.
//!
//! Single source of truth for the "glue" that turns peaks + the SO-power/
//! SO-phase series + stages into the two SOPH histograms, the summary-figure
//! spectrogram, and the parametric/spline fits. Both the **full run**
//! (`orchestrator.rs`) and the **cache rebuild** (`rebuild.rs`) call this, so
//! the two paths share one implementation and cannot drift.
//!
//! Only the *inputs* differ between callers: the full run feeds in-memory
//! series; the rebuild feeds the cached native-grid SO-power series and an
//! EMPTY SO-phase series. The SO-phase histogram forces `compute_rate=false`
//! / `min_time_in_bin=0`, so its `c_mat` is a function of the per-peak values
//! alone — the series only feeds the unused `time_in_bin` output — which is
//! why the empty-series rebuild matches the full run bit-for-bit.

use ndarray::{Array1, Array2, ArrayView1};

use multitaper_rs::{compute_spectrogram, DetrendMode, SpectrogramParams, Weighting};

// Per-thread last-splinefit-error slot. It lets the dispatcher put the
// rich diagnostic context (hist shape, finite/nonzero counts, bin
// ranges) into the per-item FAIL message without going through tracing.
// A tracing line written to stderr would overwrite the live progress
// bar, so the diagnostic travels through this slot.
//
// Consumed by `take_last_splinefit_error()`. The call site reads and
// clears, so a later item's success does not inherit stale context.
thread_local! {
    static LAST_SPLINEFIT_ERROR: std::cell::RefCell<Option<String>>
        = const { std::cell::RefCell::new(None) };
}

fn set_last_splinefit_error(msg: String) {
    LAST_SPLINEFIT_ERROR.with(|slot| {
        // Concat if a phase-axis failure follows a power-axis failure
        // for the same item: both contexts matter and the dispatcher
        // only reads the slot once at item-end.
        let mut stored_error = slot.borrow_mut();
        match stored_error.as_mut() {
            Some(existing) => {
                existing.push_str("; ");
                existing.push_str(&msg);
            }
            None => *stored_error = Some(msg),
        }
    });
}

/// Read-and-clear the per-thread last-splinefit-error slot. Returns the
/// rich diagnostic string emitted by `build_soph_outputs` since the last
/// call on this thread, or `None` if no splinefit failure was recorded.
/// Called by the CLI dispatcher's post-run check.
pub fn take_last_splinefit_error() -> Option<String> {
    LAST_SPLINEFIT_ERROR.with(|slot| slot.borrow_mut().take())
}

use crate::fits::{ParamFitResult, SplineFitResult};
use crate::run::orchestrator::DynamoError;
use crate::run::settings::DynamoSettings;

/// Borrowed inputs for [`build_soph_outputs`]. `peak_in_analysis` is supplied
/// by the caller because the two paths derive it with slightly different
/// time-range semantics (the rebuild path is unbounded).
pub(crate) struct SophBuildInputs<'a> {
    pub peak_freq: &'a [f64],
    /// Per-peak `stage ∈ SOPH_stages AND freq ∈ mtm.freq_range [AND time ∈
    /// range]` — computed by the caller.
    pub peak_in_analysis: &'a [bool],
    pub peak_sopower: &'a [f64],
    pub peak_sophase: &'a [f64],
    /// SO-power series (normalized) + its uniform time grid.
    pub so_power_norm: &'a [f64],
    pub so_power_times: &'a [f64],
    /// SO-phase series (wrapped) + its time grid. Empty in the cache path —
    /// see the module note on why that's bit-equivalent.
    pub so_phase_metric: &'a [f64],
    pub so_phase_times: &'a [f64],
    /// Hypnogram (stage codes as f64) sampled at `stage_times`.
    pub stage_times: &'a [f64],
    pub stage_vals: &'a [f64],
    pub fs: f64,
    pub t0: f64,
    pub settings: &'a DynamoSettings,
}

/// Everything the SOPH stage produces. `sopower_time_in_bin` is returned so
/// the orchestrator can log its per-stage coverage diagnostic.
pub(crate) struct SophBuildOutputs {
    pub hist_freq_centers: Vec<f64>,
    pub sopower_bin_centers: Vec<f64>,
    pub sophase_bin_centers: Vec<f64>,
    pub sopower_hist: Array2<f64>,
    pub sophase_hist: Array2<f64>,
    pub sopower_time_in_bin: Array2<f64>,
    pub summary_spectrogram: Array2<f64>,
    pub summary_sfreqs: Vec<f64>,
    pub summary_stimes: Vec<f64>,
    pub paramfit_power: Option<ParamFitResult>,
    pub paramfit_phase: Option<ParamFitResult>,
    /// Reason the power paramfit was attempted but declined (degenerate
    /// input, e.g. no valid histogram bins in the fit window). `None` when
    /// the fit ran (or wasn't requested). The writer emits a stamped
    /// zero-mode CSV recording this, so the output set stays complete.
    pub paramfit_power_skipped: Option<String>,
    /// Same as `paramfit_power_skipped`, for the phase axis.
    pub paramfit_phase_skipped: Option<String>,
    pub splinefit_power: Option<SplineFitResult>,
    pub splinefit_phase: Option<SplineFitResult>,
}

/// Summary-figure overview spectrogram (NW=15, K=29, 30 s window, 15 s step,
/// linear detrend, 2–25 Hz). MATLAB `displaySummaryPlot.m:220`. Returned in
/// absolute time (windows shifted by `t0`).
///
/// Returns `(spect, sfreqs, stimes_abs)`. When `data_arr` is shorter than the
/// window length, returns empty arrays — caller should treat this as "no
/// summary spec available" rather than an error.
pub fn compute_summary_spectrogram(
    data_arr: ArrayView1<f64>,
    fs: f64,
    t0: f64,
    progress: Option<&dyn Fn(&str, f64)>,
) -> Result<(Array2<f64>, Vec<f64>, Vec<f64>), DynamoError> {
    let report = |msg: &str, f: f64| {
        if let Some(p) = progress {
            p(msg, f);
        }
    };
    let summary_freq_range: (f64, f64) = (2.0, 25.0);
    let summary_window_len: f64 = 30.0;
    let summary_window_step: f64 = 15.0;
    let summary_nw: f64 = 15.0;
    let summary_k: usize = 29;
    let summary_win_samples = (summary_window_len * fs).round() as usize;
    // FFT length: the next power of two at or above the window length, and
    // at least 256.
    let summary_nfft = {
        let mut n = 256usize;
        while n < summary_win_samples {
            n <<= 1;
        }
        n
    };
    if data_arr.len() < summary_win_samples + 4 {
        report("summary spect: skipped (data too short)", 0.96);
        return Ok((Array2::zeros((0, 0)), Vec::new(), Vec::new()));
    }
    report(
        &format!(
            "summary spect: computing DPSS (N={}, K={})",
            summary_win_samples, summary_k
        ),
        0.95,
    );
    let tapers_sum = multitaper_rs::dpss::dpss_tapers(summary_win_samples, summary_nw, summary_k)?;
    report("summary spect: FFT loop", 0.95);
    let out = compute_spectrogram(
        data_arr,
        tapers_sum.view(),
        None,
        &SpectrogramParams {
            fs,
            frequency_range: summary_freq_range,
            window_params: (summary_window_len, summary_window_step),
            nfft: summary_nfft,
            detrend: DetrendMode::Linear,
            weighting: Weighting::Unity,
        },
    )
    .map_err(|e| DynamoError::stage("summary_spectrogram", e))?;
    report(
        &format!("summary spect: done ({} windows)", out.stimes.len()),
        0.96,
    );
    let stimes_abs: Vec<f64> = out.stimes.iter().map(|t| t + t0).collect();
    Ok((out.mt_spectrogram, out.sfreqs.to_vec(), stimes_abs))
}

/// Build the SOPH histograms, summary spectrogram, and fits. `progress` is an
/// optional `(message, fraction)` sink (the full run wires its job-progress
/// reporter; the rebuild passes `None`).
pub(crate) fn build_soph_outputs(
    inp: &SophBuildInputs,
    data_arr: Option<ArrayView1<f64>>,
    progress: Option<&dyn Fn(&str, f64)>,
    skip_summary_spec: bool,
) -> Result<SophBuildOutputs, DynamoError> {
    let report = |msg: &str, f: f64| {
        if let Some(p) = progress {
            p(msg, f);
        }
    };
    let hist_opts = &inp.settings.hist;

    // ── Frequency bins ──
    // SOPH frequency bins use the histogram's own `freq_range` (MATLAB
    // `SOpowerphasehist_opts.freq_range`, default [0, 30]), NOT the multitaper
    // spectrogram range — the two are independent options.
    let (fb_width, fb_step) = hist_opts.freq_binsizestep;
    let (freq_edges, hist_freq_centers) =
        crate::run::soph::create_partial_bins(hist_opts.freq_range, fb_width, fb_step);

    // ── SO-power range + bins ──
    // Pinned via `hist_opts.sopower_range` when set; else derived from data
    // restricted to SOPH_stages (MATLAB `SOpowerHistogram.m:265`).
    let sop_c_stages =
        crate::run::soph::stages_at_times(inp.stage_times, inp.stage_vals, inp.so_power_times);
    let stage_in_soph: Vec<bool> = if hist_opts.soph_stages.is_empty() {
        vec![true; sop_c_stages.len()]
    } else {
        sop_c_stages
            .iter()
            .map(|s| hist_opts.soph_stages.iter().any(|&t| (t as f64 - *s).abs() < 0.5))
            .collect()
    };
    let sop_range = match hist_opts.sopower_range {
        Some(r) => r,
        None => {
            let for_range: Vec<f64> = inp
                .so_power_norm
                .iter()
                .zip(stage_in_soph.iter())
                .map(|(v, &ok)| if ok { *v } else { f64::NAN })
                .collect();
            crate::run::soph::sopower_range_from_data(&for_range)
        }
    };
    // `sopower_binsizestep` override. When it is unset the bins are derived
    // from the range: width = range / 10, step = range / 100.
    let (sop_width, sop_step) = match hist_opts.sopower_binsizestep {
        Some(ws) => ws,
        None => (
            (sop_range.1 - sop_range.0) / 10.0,
            (sop_range.1 - sop_range.0) / 100.0,
        ),
    };
    let (sop_edges, sopower_bin_centers) =
        crate::run::soph::create_partial_bins(sop_range, sop_width.max(1e-9), sop_step.max(1e-9));

    // ── SO-phase bins (via 'extend') ──
    let (sph_width, sph_step) = hist_opts.sophase_binsizestep;
    let (sph_edges, sophase_bin_centers) =
        crate::run::soph::create_extend_bins(hist_opts.sophase_range, sph_width, sph_step);

    // ── SO-power histogram ──
    // Pre-filter peaks (MATLAB `SOpowerHistogram.m:240/:276`): `peak_stage ∈
    // SOPH_stages AND peak_time ∈ time_range AND !isnan(metric)` before binning.
    let sop_peak_mask: Vec<bool> = inp
        .peak_in_analysis
        .iter()
        .zip(inp.peak_sopower.iter())
        .map(|(&ok, v)| ok && v.is_finite())
        .collect();
    let sop_peak_freqs: Vec<f64> = inp
        .peak_freq
        .iter()
        .zip(sop_peak_mask.iter())
        .filter_map(|(&f, &k)| if k { Some(f) } else { None })
        .collect();
    let sop_peak_c: Vec<f64> = inp
        .peak_sopower
        .iter()
        .zip(sop_peak_mask.iter())
        .filter_map(|(&v, &k)| if k { Some(v) } else { None })
        .collect();
    let sop_peak_freqs_arr = Array1::from(sop_peak_freqs);
    let sop_peak_c_arr = Array1::from(sop_peak_c);

    // `sop_valid` is finite-AND-in-SOPH-stages (MATLAB `SOpower_valid`);
    // `sop_valid_allstages` drops the stage filter (the TIB denominator).
    let sop_valid_allstages: Vec<bool> = inp.so_power_norm.iter().map(|v| v.is_finite()).collect();
    let sop_valid: Vec<bool> = sop_valid_allstages
        .iter()
        .zip(stage_in_soph.iter())
        .map(|(&f, &s)| f && s)
        .collect();
    let sop_c_metric_arr = Array1::from(inp.so_power_norm.to_vec());
    let sop_dt = if inp.so_power_times.len() >= 2 {
        inp.so_power_times[1] - inp.so_power_times[0]
    } else {
        1.0
    };

    let sop_out = crate::histogram::tfpeak_histogram(&crate::histogram::HistogramInputs {
        c_metric: sop_c_metric_arr.view(),
        c_stages: sop_c_stages.view(),
        c_dt: sop_dt,
        c_valid: &sop_valid,
        c_valid_allstages: &sop_valid_allstages,
        peak_freqs: sop_peak_freqs_arr.view(),
        peak_c: sop_peak_c_arr.view(),
        freq_edges: freq_edges.view(),
        c_edges: sop_edges.view(),
        circular: false,
        circular_bounds: (0.0, 0.0),
        norm_dim: 0,
        compute_rate: hist_opts.compute_rate,
        min_time_in_bin: hist_opts.min_time_in_bin,
        min_peak_at_freq: 0,
    })
    .map_err(|e| DynamoError::stage("sopower_histogram", e))?;

    // ── SO-phase histogram ──
    // Inputs derive from the passed series (empty in the cache path). The
    // phase histogram is intentionally a circular probability density rather
    // than a rate, so `compute_rate=false` / `min_time_in_bin=0` regardless of
    // the user-facing `compute_rate` (power-axis only) — which is why an empty
    // series yields the same `c_mat`.
    let sph_c_stages =
        crate::run::soph::stages_at_times(inp.stage_times, inp.stage_vals, inp.so_phase_times);
    let sph_stage_in_soph: Vec<bool> = if hist_opts.soph_stages.is_empty() {
        vec![true; sph_c_stages.len()]
    } else {
        sph_c_stages
            .iter()
            .map(|s| hist_opts.soph_stages.iter().any(|&t| (t as f64 - *s).abs() < 0.5))
            .collect()
    };
    let sph_valid_allstages: Vec<bool> = inp.so_phase_metric.iter().map(|v| v.is_finite()).collect();
    let sph_valid: Vec<bool> = sph_valid_allstages
        .iter()
        .zip(sph_stage_in_soph.iter())
        .map(|(&f, &s)| f && s)
        .collect();
    let sph_c_metric_arr = Array1::from(inp.so_phase_metric.to_vec());
    let sph_dt = if inp.so_phase_times.len() >= 2 {
        inp.so_phase_times[1] - inp.so_phase_times[0]
    } else {
        1.0
    };
    let sph_peak_mask: Vec<bool> = inp
        .peak_in_analysis
        .iter()
        .zip(inp.peak_sophase.iter())
        .map(|(&ok, v)| ok && v.is_finite())
        .collect();
    let sph_peak_freqs: Vec<f64> = inp
        .peak_freq
        .iter()
        .zip(sph_peak_mask.iter())
        .filter_map(|(&f, &k)| if k { Some(f) } else { None })
        .collect();
    let sph_peak_c: Vec<f64> = inp
        .peak_sophase
        .iter()
        .zip(sph_peak_mask.iter())
        .filter_map(|(&v, &k)| if k { Some(v) } else { None })
        .collect();
    let sph_peak_freqs_arr = Array1::from(sph_peak_freqs);
    let sph_peak_c_arr = Array1::from(sph_peak_c);

    let sph_out = crate::histogram::tfpeak_histogram(&crate::histogram::HistogramInputs {
        c_metric: sph_c_metric_arr.view(),
        c_stages: sph_c_stages.view(),
        c_dt: sph_dt,
        c_valid: &sph_valid,
        c_valid_allstages: &sph_valid_allstages,
        peak_freqs: sph_peak_freqs_arr.view(),
        peak_c: sph_peak_c_arr.view(),
        freq_edges: freq_edges.view(),
        c_edges: sph_edges.view(),
        circular: true,
        circular_bounds: hist_opts.sophase_range,
        norm_dim: hist_opts.sophase_norm_dim as i32,
        compute_rate: false,
        min_time_in_bin: 0.0,
        min_peak_at_freq: hist_opts.sophase_min_peak_at_freq as i32,
    })
    .map_err(|e| DynamoError::stage("sophase_histogram", e))?;

    let sopower_hist = sop_out.c_mat;
    let sophase_hist = sph_out.c_mat;

    // ── Summary-figure spectrogram ──
    let (summary_spectrogram, summary_sfreqs, summary_stimes) = if skip_summary_spec {
        report("summary spect: skipped (cached PNG)", 0.96);
        (Array2::zeros((0, 0)), Vec::new(), Vec::new())
    } else {
        match data_arr {
            Some(view) => {
                compute_summary_spectrogram(view, inp.fs, inp.t0, Some(&|m, f| report(m, f)))?
            }
            None => {
                report("summary spect: skipped (no signal)", 0.96);
                (Array2::zeros((0, 0)), Vec::new(), Vec::new())
            }
        }
    };

    // ── Parametric / spline fits (best-effort; a failed axis stays None) ──
    report("fitting parametric / spline bases", 0.96);
    let out_opts = &inp.settings.output;
    let (paramfit_power, paramfit_phase, paramfit_power_skipped, paramfit_phase_skipped) =
        if out_opts.write_paramfits || out_opts.write_param_png {
            let (power_fit, power_skip_reason) = match crate::fits::fit_paramfit_power(
                sopower_hist.view(),
                &sopower_bin_centers,
                &hist_freq_centers,
                &inp.settings.paramfit.power,
            ) {
                Ok(r) => (Some(r), None),
                Err(e) => {
                    log::warn!("no parametric model was fit: {}", e);
                    (None, Some(format!("{}", e)))
                }
            };
            let (phase_fit, phase_skip_reason) = match crate::fits::fit_paramfit_phase(
                sophase_hist.view(),
                &sophase_bin_centers,
                &hist_freq_centers,
                &inp.settings.paramfit.phase,
            ) {
                Ok(r) => (Some(r), None),
                Err(e) => {
                    log::warn!("no parametric model was fit: {}", e);
                    (None, Some(format!("{}", e)))
                }
            };
            (power_fit, phase_fit, power_skip_reason, phase_skip_reason)
        } else {
            (None, None, None, None)
        };
    let (splinefit_power, splinefit_phase) = if out_opts.write_splinefits || out_opts.write_spline_png
    {
        // Pre-check: an all-NaN (or all-zero, no finite values) SOPH means
        // the channel is entirely artifact. The spline LSQ has nothing to
        // fit (`clip_bins_finite` returns 0×0), so it would error. Instead
        // of failing the whole item we produce a NaN-filled SplineFitResult
        // shaped against the full SOPH grid; the TIFF writer then emits a
        // NaN-pixel TIFF and the figure renderer produces a visibly-blank
        // PNG. Downstream readers that respect NaN see "no data" rather
        // than zero (which would be wrong). The user gets a present file
        // they can browse to, instead of a missing-output failure.
        let spline_pw = &inp.settings.splinefit.power;
        let spline_ph = &inp.settings.splinefit.phase;
        let sop_finite = sopower_hist.iter().filter(|v| v.is_finite()).count();
        let spline_power_fit = if sop_finite == 0 {
            Some(crate::fits::nan_splinefit_result(
                &sopower_bin_centers,
                &hist_freq_centers,
                spline_pw.num_knots_x,
                spline_pw.num_knots_y,
            ))
        } else {
            match crate::fits::fit_splinefit_power(
                sopower_hist.view(),
                &sopower_bin_centers,
                &hist_freq_centers,
                spline_pw,
            ) {
                Ok(r) => Some(r),
                Err(e) => {
                    let (n_sop, n_freq_p) = sopower_hist.dim();
                    let sop_nonzero =
                        sopower_hist.iter().filter(|v| v.is_finite() && **v > 0.0).count();
                    set_last_splinefit_error(format!(
                        "splinefit power failed: {} \
                         (hist {}x{}, finite={}, nonzero={}, \
                         so_bins=[{:.3}..{:.3}], freq_bins=[{:.3}..{:.3}])",
                        e, n_sop, n_freq_p, sop_finite, sop_nonzero,
                        sopower_bin_centers.first().copied().unwrap_or(f64::NAN),
                        sopower_bin_centers.last().copied().unwrap_or(f64::NAN),
                        hist_freq_centers.first().copied().unwrap_or(f64::NAN),
                        hist_freq_centers.last().copied().unwrap_or(f64::NAN),
                    ));
                    None
                }
            }
        };
        let sph_finite = sophase_hist.iter().filter(|v| v.is_finite()).count();
        let spline_phase_fit = if sph_finite == 0 {
            Some(crate::fits::nan_splinefit_result(
                &sophase_bin_centers,
                &hist_freq_centers,
                spline_ph.num_knots_x,
                spline_ph.num_knots_y,
            ))
        } else {
            match crate::fits::fit_splinefit_phase(
                sophase_hist.view(),
                &sophase_bin_centers,
                &hist_freq_centers,
                spline_ph,
            ) {
                Ok(r) => Some(r),
                Err(e) => {
                    let (n_sph, n_freq_h) = sophase_hist.dim();
                    let sph_nonzero =
                        sophase_hist.iter().filter(|v| v.is_finite() && **v > 0.0).count();
                    set_last_splinefit_error(format!(
                        "splinefit phase failed: {} \
                         (hist {}x{}, finite={}, nonzero={}, \
                         phase_bins=[{:.3}..{:.3}], freq_bins=[{:.3}..{:.3}])",
                        e, n_sph, n_freq_h, sph_finite, sph_nonzero,
                        sophase_bin_centers.first().copied().unwrap_or(f64::NAN),
                        sophase_bin_centers.last().copied().unwrap_or(f64::NAN),
                        hist_freq_centers.first().copied().unwrap_or(f64::NAN),
                        hist_freq_centers.last().copied().unwrap_or(f64::NAN),
                    ));
                    None
                }
            }
        };
        (spline_power_fit, spline_phase_fit)
    } else {
        (None, None)
    };

    Ok(SophBuildOutputs {
        hist_freq_centers,
        sopower_bin_centers,
        sophase_bin_centers,
        sopower_hist,
        sophase_hist,
        sopower_time_in_bin: sop_out.time_in_bin,
        summary_spectrogram,
        summary_sfreqs,
        summary_stimes,
        paramfit_power,
        paramfit_phase,
        paramfit_power_skipped,
        paramfit_phase_skipped,
        splinefit_power,
        splinefit_phase,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::run::settings::DynamoSettings;

    /// The cache-rebuild path feeds an EMPTY SO-phase series. This locks the
    /// invariant that makes that correct: with the phase axis forced to
    /// `compute_rate=false` / `min_time_in_bin=0`, the SO-phase histogram is a
    /// function of the per-peak values alone — independent of the series — and
    /// the SO-power histogram doesn't depend on the SO-phase series at all.
    #[test]
    fn sophase_hist_independent_of_series() {
        let mut settings = DynamoSettings::default();
        // Skip the heavy fits — we're checking the histograms/bins.
        settings.output.write_paramfits = false;
        settings.output.write_param_png = false;
        settings.output.write_splinefits = false;
        settings.output.write_spline_png = false;

        let peak_freq = vec![5.0, 8.0, 11.0, 13.0];
        let peak_in_analysis = vec![true, true, true, true];
        let peak_sopower = vec![0.0, 5.0, -2.0, 10.0];
        let peak_sophase = vec![0.1, -1.0, 2.0, 1.5];
        let so_power_norm: Vec<f64> = (0..200).map(|i| (i as f64 * 0.1).sin() * 5.0).collect();
        let grid: Vec<f64> = (0..200).map(|i| i as f64 * 0.5).collect();
        let stage_vals = vec![2.0_f64; 200]; // all N2 (∈ SOPH_stages)
        // Short signal → summary spectrogram is skipped (keeps the test fast).
        let data_arr = Array1::<f64>::zeros(8);

        let run = |sph_metric: &[f64], sph_times: &[f64]| {
            build_soph_outputs(
                &SophBuildInputs {
                    peak_freq: &peak_freq,
                    peak_in_analysis: &peak_in_analysis,
                    peak_sopower: &peak_sopower,
                    peak_sophase: &peak_sophase,
                    so_power_norm: &so_power_norm,
                    so_power_times: &grid,
                    so_phase_metric: sph_metric,
                    so_phase_times: sph_times,
                    stage_times: &grid,
                    stage_vals: &stage_vals,
                    fs: 100.0,
                    t0: 0.0,
                    settings: &settings,
                },
                Some(data_arr.view()),
                None,
                false,
            )
            .expect("build_soph_outputs")
        };

        let real_series: Vec<f64> = (0..200).map(|i| (i as f64 * 0.07).sin()).collect();
        let with_series = run(&real_series, &grid);
        let empty = run(&[], &[]);

        // SOPH histograms carry NaN in empty bins, and NaN != NaN — so compare
        // element-wise treating NaN-at-the-same-position as equal.
        let same = |a: &Array2<f64>, b: &Array2<f64>| {
            a.shape() == b.shape()
                && a.iter()
                    .zip(b.iter())
                    .all(|(x, y)| (x.is_nan() && y.is_nan()) || x == y)
        };
        // Sanity: the phase histogram has at least one finite bin (peaks landed),
        // so this isn't a trivial all-NaN match.
        assert!(
            with_series.sophase_hist.iter().any(|v| v.is_finite()),
            "test setup produced an all-NaN SO-phase histogram"
        );
        assert!(
            same(&with_series.sophase_hist, &empty.sophase_hist),
            "SO-phase histogram must not depend on the series (empty-series rebuild invariant)"
        );
        assert!(
            same(&with_series.sopower_hist, &empty.sopower_hist),
            "SO-power histogram must not depend on the SO-phase series"
        );
        assert_eq!(with_series.hist_freq_centers, empty.hist_freq_centers);
        assert_eq!(with_series.sopower_bin_centers, empty.sopower_bin_centers);
        assert_eq!(with_series.sophase_bin_centers, empty.sophase_bin_centers);
    }
}
