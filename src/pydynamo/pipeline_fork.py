"""Fork-path top-level pipeline: uses vendored pyDYNAM-O detect_tfpeaks for
pass-1 and pass-2, with the new multitaper + symmetric merge + Hann
refinement + two-pass masked watershed layered on top.

This is the parallel "known-good" pipeline we fall back to if my first-try
tfpeaks/ modules don't reproduce MATLAB structure. Same public
run_dynamo_fork() signature as run_dynamo() so the plot/SOPH code stays
identical.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from pydynamo.artifacts import detect_artifacts
from pydynamo.soph.histogram import so_power_histogram, so_phase_histogram
from pydynamo.soph.sophase import compute_so_phase
from pydynamo.soph.sopower import compute_so_power
from pydynamo.tfpeaks.refine import refine_peak_frequency
from pydynamo.tfpeaks_fork import compute_tfpeaks_fork
from pydynamo.pipeline import SOPHsResult, DynamoOutput


def _peak_labels_image(spect_shape, stimes, stats_df, tcol, fcol):
    """Build a labels image from a pass-1 stats DataFrame: label = row+1
    placed at each peak's weighted centroid. Used to mask the pass-2
    spectrogram."""
    # This is a placeholder mask: for the vendored tfpeaks_fork, the peak
    # regions aren't returned directly — only stats. For mask we reconstruct
    # a region mask from each peak's bounding-box.
    F, T = spect_shape
    mask = np.zeros((F, T), dtype=bool)
    if stats_df.empty:
        return mask
    # detect_tfpeaks returns BoundingBox coords as bbox-0..3 (removed in
    # vendored code) — or reconstructed via duration/bandwidth. Use
    # min/max sfreq and min/max time from peak centroid + bw/dur.
    pt = stats_df[tcol].to_numpy()
    pf = stats_df[fcol].to_numpy()
    dur = stats_df["duration"].to_numpy()
    bw = stats_df["bandwidth"].to_numpy()
    dt = float(stimes[1] - stimes[0])
    df_step = 100.0 / 2048 if F == 308 else 1.0  # 0.0977 Hz for Fs=100, nfft=1024
    t0 = stimes[0]
    for i in range(len(stats_df)):
        t_lo = pt[i] - dur[i] / 2
        t_hi = pt[i] + dur[i] / 2
        f_lo = pf[i] - bw[i] / 2
        f_hi = pf[i] + bw[i] / 2
        j0 = max(int(round((t_lo - t0) / dt)), 0)
        j1 = min(int(round((t_hi - t0) / dt)), T - 1)
        i0 = max(int(round(f_lo / df_step)), 0)
        i1 = min(int(round(f_hi / df_step)), F - 1)
        if j1 > j0 and i1 > i0:
            mask[i0:i1 + 1, j0:j1 + 1] = True
    return mask


def run_dynamo_fork(
    data: np.ndarray,
    fs: float,
    stage_times: np.ndarray,
    stage_vals: np.ndarray,
    *,
    time_range: tuple[float, float] | None = None,
    merge_thresh: float = 8.0,
    trim_vol: float = 0.8,
    seg_time: float = 30.0,
    soph_stages: tuple[int, ...] = (1, 2, 3),
    min_time_in_bin: float = 5.0,
    double_watershed: bool = True,
    refinement: bool = True,
    plot: bool = True,
    verbose: bool = True,
    n_jobs: int = -1,
) -> DynamoOutput:
    """Fork-path pipeline. Same output contract as `run_dynamo`."""
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
        print(f"[fork] {data_tr.size} samples ({data_tr.size/fs:.0f}s)")

    # Artifact detection
    if verbose: print("[fork] artifacts...")
    artifacts = detect_artifacts(data_tr, fs)

    # Pass-1 extract (1 s window). compute_tfpeaks_fork takes time_range
    # RELATIVE to the `data` it receives — we've already sliced, so pass None
    # and let it use the full range.
    if verbose: print("[fork] pass-1 extract...")
    stats1, spect1, stimes1_rel, sfreqs = compute_tfpeaks_fork(
        data_tr, fs, time_range=None,
        isexcluded=artifacts, merge_thresh=merge_thresh,
        trim_volume=trim_vol, downsample=(2, 2),
        segment_dur=seg_time, n_jobs=n_jobs, verbose=verbose,
    )
    # Shift peak times + stimes to absolute frame
    if not stats1.empty and "peak_time" in stats1.columns:
        stats1["peak_time"] = stats1["peak_time"] + t_tr[0]
    stimes1 = stimes1_rel + t_tr[0]
    if verbose: print(f"[fork]   pass-1: {len(stats1)} peaks")

    # Pass-2 (2 s window) with mask from pass-1 bounding boxes
    stats = stats1
    spect_for_plot, stimes_for_plot = spect1, stimes1
    if double_watershed and not stats1.empty:
        if verbose: print("[fork] pass-2 spectrogram (2 s window)...")
        from pydynamo.spectrogram import mtm_spectrogram
        spect2, stimes2_rel, sfreqs2 = mtm_spectrogram(
            data_tr, fs, freq_range=(0.0, 30.0), taper_params=(2, 3),
            window_params=(2.0, 0.05), dsfreqs=0.1,
        )
        stimes2 = stimes2_rel + t_tr[0]
        baseline2 = np.percentile(spect2, 2, axis=1, keepdims=True)
        spect2_norm = spect2 / baseline2

        if verbose: print("[fork] masking pass-2 from pass-1 peaks...")
        mask = _peak_labels_image(spect2.shape, stimes2, stats1,
                                  "peak_time", "peak_frequency")
        spect2_masked = np.where(mask, spect2_norm, 0.0)

        if verbose: print("[fork] pass-2 extract...")
        from pydynamo.tfpeaks_fork import compute_tfpeaks_fork as _ctf
        # Re-run the vendored extraction on the masked 2s spect. Since
        # compute_tfpeaks_fork recomputes spectrogram internally, we call
        # detect_tfpeaks directly via a thin path.
        from pydynamo.vendor_pydynam_o.TFpeaks import (
            detect_tfpeaks, process_segments_params,
        )
        window_idxs, start_times = process_segments_params(seg_time, stimes2_rel)
        d_time = float(stimes2_rel[1] - stimes2_rel[0])
        d_freq = float(sfreqs2[1] - sfreqs2[0])
        from scipy.stats import chi2
        prom_min = -(10 * np.log10(6 / chi2.ppf(0.975, 6))) * 2
        from joblib import Parallel, delayed
        def _one(idxs, st):
            return detect_tfpeaks(
                spect2_masked[:, idxs], float(st + t_tr[0]),
                d_time=d_time, d_freq=d_freq,
                merge_thresh=merge_thresh, max_merges=np.inf,
                trim_volume=trim_vol, downsample=[2, 2],
                dur_min=2.0 / 2, dur_max=5.0,
                bw_min=(2 / 2.0 * 2) / 2, bw_max=15.0,
                prom_min=prom_min, plot_on=False, verbose=False,
            )
        if n_jobs == 1:
            tables = [_one(idx, st) for idx, st in zip(window_idxs, start_times)]
        else:
            tables = Parallel(n_jobs=n_jobs, prefer="processes")(
                delayed(_one)(idx, st) for idx, st in zip(window_idxs, start_times)
            )
        tables = [t for t in tables if not t.empty]
        stats = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
        if "label" in stats.columns:
            del stats["label"]
        spect_for_plot, stimes_for_plot = spect2, stimes2
        if verbose: print(f"[fork]   pass-2: {len(stats)} peaks")

    # Convert pyDYNAM-O columns → my schema for downstream compatibility
    if not stats.empty:
        stats = stats.rename(columns={
            "peak_time": "PeakTime",
            "peak_frequency": "PeakFrequency",
            "duration": "Duration",
            "bandwidth": "Bandwidth",
            "volume": "Volume",
            "prominence": "Height",
        }).reset_index(drop=True)
        # Synthesize BoundingBox column expected by refinePeakFrequency
        stats["BoundingBox"] = [
            (float(pt - d / 2), float(pf - bw / 2), float(d), float(bw))
            for pt, pf, d, bw in zip(
                stats["PeakTime"], stats["PeakFrequency"],
                stats["Duration"], stats["Bandwidth"],
            )
        ]

    # Hann refinement
    if refinement and not stats.empty:
        if verbose: print("[fork] Hann refinement...")
        stats = refine_peak_frequency(
            stats, data_tr, fs, t=t_tr,
            freq_range=(0.0, 30.0), window_size=4.0, dsfreqs=0.05,
            refine_method="spline_interp", remove_edge_peaks=True,
        )
        if verbose: print(f"[fork]   after refine: {len(stats)} peaks")

    # Per-peak stage
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

    # SO-power and SO-phase
    if verbose: print("[fork] SO-power...")
    SOpower_norm, SOpower_times, SOpower_stages, _, _ = compute_so_power(
        data_tr, fs, stage_times=stage_times, stage_vals=stage_vals,
        eeg_times=t_tr, time_range=time_range, isexcluded=artifacts,
        SO_freqrange=(0.3, 1.5), tapers=(5, 9), window_params=(5.0, 0.5),
        SOpower_outlier_threshold=3.0, norm_method="p2shift1234",
        retain_Fs=True,
    )
    if not stats.empty:
        xp = np.concatenate(([SOpower_times[0] - 1], SOpower_times,
                             [SOpower_times[-1] + 1]))
        fp = np.concatenate(([SOpower_norm[0]], SOpower_norm, [SOpower_norm[-1]]))
        stats["SOpower"] = np.interp(stats["PeakTime"].to_numpy(), xp, fp)

    if verbose: print("[fork] SO-phase...")
    SOphase_unwrapped, SOphase_times, SOphase_stages, _ = compute_so_phase(
        data_tr, fs, stage_times=stage_times, stage_vals=stage_vals,
        eeg_times=t_tr, isexcluded=artifacts, SO_freqrange=(0.3, 1.5),
    )
    if not stats.empty:
        peak_phase = np.interp(
            stats["PeakTime"].to_numpy(), SOphase_times, SOphase_unwrapped,
        )
        stats["SOphase"] = (peak_phase + np.pi) % (2 * np.pi) - np.pi

    # SOPH histograms
    if verbose: print("[fork] SOPH histograms...")
    pf = stats["PeakFrequency"].to_numpy() if not stats.empty else np.array([])
    pt = stats["PeakTime"].to_numpy() if not stats.empty else np.array([])
    ps = stats["PeakStage"].to_numpy() if not stats.empty else np.array([])
    sopow = so_power_histogram(
        pf, pt, ps, SOpower_norm, SOpower_times, SOpower_stages,
        time_range=time_range, soph_stages=soph_stages,
        freq_range=(0.0, 30.0), freq_binsizestep=(1.0, 0.2),
        so_range=None, so_binsizestep=None,
        min_time_in_bin=min_time_in_bin, compute_rate=True, norm_dim=0,
    )
    sopha = so_phase_histogram(
        pf, pt, ps, SOphase_unwrapped, SOphase_times, SOphase_stages,
        time_range=time_range, soph_stages=soph_stages,
        freq_range=(0.0, 30.0), freq_binsizestep=(1.0, 0.2),
        so_range=(-np.pi, np.pi),
        so_binsizestep=(2 * np.pi / 5, 2 * np.pi / 100),
        min_peak_at_freq=1, compute_rate=True, norm_dim=1,
    )

    peak_selection_inds = (
        np.asarray(sopow["peak_selection_inds"] & sopha["peak_selection_inds"],
                    dtype=bool)
        if not stats.empty else np.zeros(0, dtype=bool)
    )

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
        fig = summary_plot(
            spect_for_plot, stimes_for_plot, sfreqs, artifacts, t_tr,
            stage_times, stage_vals, stats, sophs,
            data=data_tr, fs=fs, time_range=time_range,
            freq_limits=(2.0, 25.0), mtm_freq_range=(2.0, 25.0),
            hist_peakidx=peak_selection_inds,
        )

    return DynamoOutput(
        stats_table=stats, spect=spect_for_plot, stimes=stimes_for_plot,
        sfreqs=sfreqs, artifacts=artifacts, SOPHs=sophs, fig=fig,
    )
