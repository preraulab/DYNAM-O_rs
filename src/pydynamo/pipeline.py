"""Top-level pipeline: run_dynamo().

Matches DYNAMO_dev's computeTFPeaks flow:
  1. Multitaper spectrogram pass-1 (1 s window)
  2. Artifacts + baseline subtract → pass-1 spect
  3. Extract pass-1 peaks (watershed + merge + trim), keep labels image
  4. Multitaper spectrogram pass-2 (2 s window) + baseline subtract
  5. Mask pass-2 spect to pass-1 regions
  6. Extract pass-2 peaks on masked spect
  7. Hann-window 1 Hz frequency refinement per peak
  8. Assign per-peak Stage, SO-power, SO-phase
  9. SO-power + SO-phase histograms
 10. Summary plot
"""

from __future__ import annotations

import time as _time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


@contextmanager
def _timer(store: dict, key: str):
    """Context manager that records wallclock in `store[key]`."""
    t0 = _time.time()
    try:
        yield
    finally:
        store[key] = store.get(key, 0.0) + (_time.time() - t0)

from pydynamo.artifacts import detect_artifacts
from pydynamo.baseline import compute_baseline, subtract_baseline
from pydynamo.soph.histogram import so_power_histogram, so_phase_histogram
from pydynamo.soph.sophase import compute_so_phase
from pydynamo.soph.sopower import compute_so_power
from pydynamo.spectrogram import mtm_spectrogram
from pydynamo.tfpeaks.extract import extract_tfpeaks
from pydynamo.tfpeaks.mask import mask_spectrogram
from pydynamo.tfpeaks.refine import refine_peak_frequency


@dataclass
class SOPHsResult:
    SOpower_mat: np.ndarray
    SOphase_mat: np.ndarray
    SOpower_bins: np.ndarray
    SOphase_bins: np.ndarray
    freq_bins: np.ndarray
    SOpower_TIB: np.ndarray
    SOphase_TIB: np.ndarray
    peak_at_freq_SOpower: np.ndarray
    peak_at_freq_SOphase: np.ndarray
    SOpower_norm: np.ndarray
    SOpower_times: np.ndarray
    SOphase: np.ndarray
    SOphase_times: np.ndarray


@dataclass
class DynamoOutput:
    stats_table: pd.DataFrame
    spect: np.ndarray              # pass-2 spectrogram (for plotting)
    stimes: np.ndarray
    sfreqs: np.ndarray
    artifacts: np.ndarray
    SOPHs: SOPHsResult
    fig: Any = None
    timings: dict | None = None    # matches MATLAB `timings` struct fields


def run_dynamo(
    data: np.ndarray,
    fs: float,
    stage_times: np.ndarray,
    stage_vals: np.ndarray,
    *,
    time_range: tuple[float, float] | None = None,
    merge_thresh: float = 11.0,    # MATLAB detection_opts.m 'default' preset
    trim_vol: float = 0.8,
    seg_time: float = 30.0,
    soph_stages: tuple[int, ...] = (1, 2, 3),
    min_time_in_bin: float = 10.0,   # MATLAB SOpowerphasehist_opts default (runExampleData.m:72 overrides to 5 for segment only)
    min_peak_at_freq: int = 0,       # MATLAB SOpowerphasehist_opts default (runExampleData.m:73 overrides to 10 for segment only)
    double_watershed: bool = True,
    refinement: bool = True,
    plot: bool = True,
    verbose: bool = True,
) -> DynamoOutput:
    """Run the pydynamo pipeline.

    Stage convention (DYNAM-O): 1=N3, 2=N2, 3=N1, 4=REM, 5=Wake.
    """
    data = np.ascontiguousarray(np.asarray(data, dtype=np.float64).ravel())
    fs = float(fs)
    stage_times = np.asarray(stage_times, dtype=float).ravel()
    stage_vals = np.asarray(stage_vals, dtype=float).ravel()

    if time_range is None:
        time_range = (0.0, (data.size - 1) / fs)

    i0 = int(round(time_range[0] * fs))
    i1 = int(round(time_range[1] * fs))
    data_tr = data[i0 : i1 + 1]
    t_tr = np.arange(i0, i1 + 1) / fs

    if verbose:
        print(f"[dynamo] {data_tr.size} samples ({data_tr.size/fs:.0f}s) at {fs} Hz")

    timings: dict[str, float] = {}
    t_start = _time.time()

    # ---- Artifacts (once, on raw data) ----
    if verbose: print("[dynamo] artifacts...")
    with _timer(timings, "artifact"):
        artifacts = detect_artifacts(data_tr, fs)

    # Stage mask for baseline exclusion
    stage_at_data = interp1d(
        stage_times, stage_vals, kind="previous",
        bounds_error=False, fill_value=0.0,
    )(t_tr)
    stage_exclude = ~np.isin(stage_at_data, (1, 2, 3, 4, 5))
    baseline_exclude = artifacts | stage_exclude

    # ---- Pass 1: 1 s window ----
    if verbose: print("[dynamo] pass-1 spectrogram (1 s window)...")
    with _timer(timings, "spect_pass1"):
        spect1, stimes1_rel, sfreqs = mtm_spectrogram(
            data_tr, fs,
            freq_range=(0.0, 30.0), taper_params=(2, 3),
            window_params=(1.0, 0.05), dsfreqs=0.1,
        )
    stimes1 = stimes1_rel + t_tr[0]
    with _timer(timings, "baseline_pass1"):
        baseline1 = compute_baseline(spect1, stimes1, t_tr, baseline_exclude,
                                      baseline_ptile=2.0)
        spect1_norm = subtract_baseline(spect1, baseline1)

    if verbose: print("[dynamo] pass-1 extract...")
    with _timer(timings, "extract_pass1"):
        stats1, labels1 = extract_tfpeaks(
            spect1_norm, stimes1, sfreqs,
            seg_time=seg_time, return_labels=True,
            downsample=(2, 2), merge_thresh=merge_thresh, trim_vol=trim_vol,
            dur_min=0.5, dur_max=5.0, bw_min=2.0, bw_max=15.0,
        )
    if verbose: print(f"[dynamo]   pass-1: {len(stats1)} peaks")

    # ---- Pass 2 (optional): 2 s window, masked by pass-1 regions ----
    if double_watershed and not stats1.empty:
        if verbose: print("[dynamo] pass-2 spectrogram (2 s window)...")
        with _timer(timings, "spect_pass2"):
            spect2, stimes2_rel, sfreqs2 = mtm_spectrogram(
                data_tr, fs,
                freq_range=(0.0, 30.0), taper_params=(2, 3),
                window_params=(2.0, 0.05), dsfreqs=0.1,
            )
        stimes2 = stimes2_rel + t_tr[0]
        assert sfreqs2.shape == sfreqs.shape, \
            "pass-1/pass-2 sfreqs differ (nfft mismatch)"
        with _timer(timings, "baseline_pass2"):
            baseline2 = compute_baseline(spect2, stimes2, t_tr, baseline_exclude,
                                          baseline_ptile=2.0)
            spect2_norm = subtract_baseline(spect2, baseline2)
            if verbose: print("[dynamo] masking pass-2 spect with pass-1 regions...")
            spect2_masked = mask_spectrogram(spect2_norm, stimes2, labels1, stimes1)

        if verbose: print("[dynamo] pass-2 extract on masked spectrogram...")
        # MATLAB-exact params (computeTFPeaks.m:342): pass-2 uses dur_min=0.5
        # from pass-1 (line 342 ignores pass-2's own dur_min return), and
        # bw_min = (TW/T) = 2/2 = 1.0 Hz (spectral resolution / 2).
        with _timer(timings, "extract_pass2"):
            stats = extract_tfpeaks(
                spect2_masked, stimes2, sfreqs2,
                seg_time=seg_time,
                downsample=(2, 2), merge_thresh=merge_thresh, trim_vol=trim_vol,
                dur_min=0.5, dur_max=5.0, bw_min=1.0, bw_max=15.0,
            )
        spect_for_plot = spect2
        stimes_for_plot = stimes2
        if verbose: print(f"[dynamo]   pass-2: {len(stats)} peaks")
    else:
        stats = stats1
        spect_for_plot = spect1
        stimes_for_plot = stimes1

    # ---- Hann refinement ----
    if refinement and not stats.empty:
        if verbose: print("[dynamo] Hann frequency refinement...")
        with _timer(timings, "refine"):
            stats = refine_peak_frequency(
                stats, data_tr, fs, t=t_tr,
                freq_range=(0.0, 30.0), window_size=4.0, dsfreqs=0.05,
                refine_method="spline_interp", remove_edge_peaks=True,
            )
        if verbose: print(f"[dynamo]   after refinement: {len(stats)} peaks")

    # ---- Per-peak stage ----
    with _timer(timings, "peak_stage"):
        if not stats.empty:
            stage_interp = interp1d(
                stage_times, stage_vals, kind="previous",
                bounds_error=False, fill_value=0.0,
            )
            stats["PeakStage"] = stage_interp(stats["PeakTime"].to_numpy())
            art_at_peak = interp1d(
                t_tr, artifacts.astype(float), kind="nearest",
                bounds_error=False, fill_value=0.0,
            )(stats["PeakTime"].to_numpy()) >= 0.5
            stats.loc[art_at_peak, "PeakStage"] = 6

    # ---- SO-power / SO-phase timeseries ----
    if verbose: print("[dynamo] SO-power...")
    with _timer(timings, "peak_sopower"):
        SOpower_norm, SOpower_times, SOpower_stages, _, _ = compute_so_power(
            data_tr, fs,
            stage_times=stage_times, stage_vals=stage_vals,
            eeg_times=t_tr, time_range=time_range,
            isexcluded=artifacts,
            SO_freqrange=(0.3, 1.5), tapers=(5, 9), window_params=(5.0, 0.5),
            SOpower_outlier_threshold=3.0, norm_method="p2shift1234",
            retain_Fs=True,
        )
        if not stats.empty:
            xp = np.concatenate(([SOpower_times[0] - 1], SOpower_times,
                                 [SOpower_times[-1] + 1]))
            fp = np.concatenate(([SOpower_norm[0]], SOpower_norm, [SOpower_norm[-1]]))
            stats["SOpower"] = np.interp(stats["PeakTime"].to_numpy(), xp, fp)

    if verbose: print("[dynamo] SO-phase...")
    with _timer(timings, "peak_sophase"):
        SOphase_unwrapped, SOphase_times, SOphase_stages, _ = compute_so_phase(
            data_tr, fs,
            stage_times=stage_times, stage_vals=stage_vals,
            eeg_times=t_tr, isexcluded=artifacts,
            SO_freqrange=(0.3, 1.5),
        )
        if not stats.empty:
            peak_phase_unwrapped = np.interp(
                stats["PeakTime"].to_numpy(), SOphase_times, SOphase_unwrapped,
            )
            stats["SOphase"] = (peak_phase_unwrapped + np.pi) % (2 * np.pi) - np.pi

    # ---- SOPH histograms ----
    if verbose: print("[dynamo] SO-power histogram...")
    pf = stats["PeakFrequency"].to_numpy() if not stats.empty else np.array([])
    pt = stats["PeakTime"].to_numpy() if not stats.empty else np.array([])
    ps = stats["PeakStage"].to_numpy() if not stats.empty else np.array([])
    with _timer(timings, "soph_sopower_hist"):
        sopow = so_power_histogram(
            pf, pt, ps, SOpower_norm, SOpower_times, SOpower_stages,
            time_range=time_range, soph_stages=soph_stages,
            freq_range=(0.0, 30.0), freq_binsizestep=(1.0, 0.2),
            so_range=None, so_binsizestep=None,
            min_time_in_bin=min_time_in_bin, compute_rate=True, norm_dim=0,
        )

    if verbose: print("[dynamo] SO-phase histogram...")
    with _timer(timings, "soph_sophase_hist"):
        sopha = so_phase_histogram(
            pf, pt, ps, SOphase_unwrapped, SOphase_times, SOphase_stages,
            time_range=time_range, soph_stages=soph_stages,
            freq_range=(0.0, 30.0), freq_binsizestep=(1.0, 0.2),
            so_range=(-np.pi, np.pi),
            so_binsizestep=(2 * np.pi / 5, 2 * np.pi / 100),
            min_peak_at_freq=min_peak_at_freq, compute_rate=True, norm_dim=1,
        )

    # NREM-included peaks (SOPH_stages filter — matches MATLAB's hist_peakidx).
    # SO-power and SO-phase selection masks should agree on stage/time/NaN so
    # combine them; peaks excluded by either are not counted in any histogram.
    peak_selection_inds = np.asarray(
        sopow["peak_selection_inds"] & sopha["peak_selection_inds"], dtype=bool
    ) if (not stats.empty) else np.zeros(0, dtype=bool)

    sophs = SOPHsResult(
        SOpower_mat=sopow["c_mat"], SOphase_mat=sopha["c_mat"],
        SOpower_bins=sopow["c_cbins"], SOphase_bins=sopha["c_cbins"],
        freq_bins=sopow["freq_cbins"],
        SOpower_TIB=sopow["time_in_bin"], SOphase_TIB=sopha["time_in_bin"],
        peak_at_freq_SOpower=sopow["peak_at_freq"],
        peak_at_freq_SOphase=sopha["peak_at_freq"],
        SOpower_norm=SOpower_norm, SOpower_times=SOpower_times,
        SOphase=(SOphase_unwrapped + np.pi) % (2 * np.pi) - np.pi,
        SOphase_times=SOphase_times,
    )

    fig = None
    if plot:
        from pydynamo.plot import summary_plot
        if verbose: print("[dynamo] plotting...")
        with _timer(timings, "plot_summary"):
            fig = summary_plot(
                spect_for_plot, stimes_for_plot, sfreqs, artifacts, t_tr,
                stage_times, stage_vals, stats, sophs,
                data=data_tr, fs=fs, time_range=time_range,
                freq_limits=(2.0, 25.0), mtm_freq_range=(2.0, 25.0),
                hist_peakidx=peak_selection_inds,
            )
    timings["total"] = _time.time() - t_start

    if verbose:
        print(f"\n[dynamo] total wallclock: {timings['total']:.1f}s")

    return DynamoOutput(
        stats_table=stats, spect=spect_for_plot, stimes=stimes_for_plot,
        sfreqs=sfreqs, artifacts=artifacts, SOPHs=sophs, fig=fig,
        timings=timings,
    )
