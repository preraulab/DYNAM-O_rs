"""Fork path: thin wrapper around vendored pyDYNAM-O detect_tfpeaks.

Uses:
  - multitaper_toolbox submodule (via pydynamo.spectrogram.mtm_spectrogram)
  - vendor_pydynam_o.TFpeaks.detect_tfpeaks (with symmetric merge fix)
  - pydynamo.tfpeaks.refine.refine_peak_frequency (Hann refinement, vectorized)

Also orchestrates the two-pass masked watershed: pass-1 (1 s window) →
build mask → pass-2 (2 s window, masked) → concatenate.
"""

from __future__ import annotations

import time
from typing import Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import chi2

from pydynamo.spectrogram import mtm_spectrogram
from pydynamo.vendor_pydynam_o.TFpeaks import detect_tfpeaks, process_segments_params


def _min_prominence(num_tapers: int, alpha: float = 0.95) -> float:
    chi2_df = 2 * num_tapers
    val = chi2_df / chi2.ppf(alpha / 2 + 0.5, chi2_df)
    return -(10 * np.log10(val)) * 2


def _run_one_segment(spect_slice, start_time, d_time, d_freq, merge_thresh,
                     trim_volume, downsample, dur_min, dur_max,
                     bw_min, bw_max, prom_min):
    """Call vendored detect_tfpeaks. kwargs-style so joblib pickles cleanly."""
    return detect_tfpeaks(
        spect_slice, start_time=start_time,
        d_time=d_time, d_freq=d_freq,
        merge_thresh=merge_thresh, max_merges=np.inf,
        trim_volume=trim_volume, downsample=downsample,
        dur_min=dur_min, dur_max=dur_max,
        bw_min=bw_min, bw_max=bw_max,
        prom_min=prom_min, plot_on=False, verbose=False,
    )


def compute_tfpeaks_fork(
    data: np.ndarray,
    fs: float,
    *,
    time_range: tuple[float, float] | None = None,
    isexcluded: np.ndarray | None = None,
    merge_thresh: float = 8.0,
    max_merges: float = np.inf,
    trim_volume: float = 0.8,
    downsample: Tuple[int, int] = (2, 2),
    segment_dur: float = 30.0,
    n_jobs: int = -1,
    verbose: bool = True,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Run pass-1 extract (pyDYNAM-O's detect_tfpeaks) on a 1 s-window
    multitaper spectrogram. Returns (stats_table, spect, stimes, sfreqs)
    for the pass-2 step to consume.

    Parameters match pyDYNAM-O's compute_tfpeaks signature but compute
    spect via the multitaper_toolbox submodule (through
    `pydynamo.spectrogram.mtm_spectrogram`).
    """
    data = np.ascontiguousarray(np.asarray(data, dtype=np.float64).ravel())
    fs = float(fs)
    n = data.size
    if time_range is None:
        time_range = (0.0, (n - 1) / fs)

    # Slice to time_range (inclusive — matches MATLAB)
    i0 = int(round(time_range[0] * fs))
    i1 = int(round(time_range[1] * fs))
    sub = data[i0 : i1 + 1]
    t_tr = np.arange(i0, i1 + 1) / fs
    if verbose:
        print(f"[fork] pass-1 spect (1 s window) on {sub.size} samples...")

    # pyDYNAM-O spec: freq_range=[0,30], taper_params=[2,3], window=[1,0.05]
    window_params = (1.0, 0.05)
    taper_params = (2, 3)
    freq_range = (0.0, 30.0)
    spect, stimes_rel, sfreqs = mtm_spectrogram(
        sub, fs, freq_range=freq_range, taper_params=taper_params,
        window_params=window_params, dsfreqs=0.1,
    )
    d_time = float(stimes_rel[1] - stimes_rel[0])
    d_freq = float(sfreqs[1] - sfreqs[0])
    df_mtm = taper_params[0] / window_params[0] * 2     # pyDYNAM-O line 61

    # Remove baseline (2nd pctile per freq, divide)
    baseline = np.percentile(spect, 2, axis=1, keepdims=True)
    spect_baseline = spect / baseline

    # Zero out excluded samples via nearest-neighbour lookup (same as
    # my baseline.compute_baseline).
    if isexcluded is not None:
        idx = np.searchsorted(t_tr, stimes_rel + t_tr[0])
        idx = np.clip(idx, 0, t_tr.size - 1)
        left = np.clip(idx - 1, 0, t_tr.size - 1)
        use_left = np.abs(t_tr[left] - (stimes_rel + t_tr[0])) < \
                   np.abs(t_tr[idx] - (stimes_rel + t_tr[0]))
        nearest = np.where(use_left, left, idx)
        exclude_cols = np.asarray(isexcluded, dtype=bool)[nearest]
        spect_baseline = np.where(exclude_cols[None, :], 0.0, spect_baseline)

    # Per-segment pyDYNAM-O detect_tfpeaks
    window_idxs, start_times = process_segments_params(segment_dur, stimes_rel)
    if verbose:
        print(f"[fork] {len(window_idxs)} segments; running detect_tfpeaks...")

    dur_min = window_params[0] / 2
    bw_min = df_mtm / 2
    dur_max = 5.0
    bw_max = 15.0
    prom_min = _min_prominence(taper_params[1], 0.95)

    tic = time.time()
    seg_args = [
        (spect_baseline[:, idxs].copy(), float(st), d_time, d_freq,
         merge_thresh, trim_volume, list(downsample) if downsample else [],
         dur_min, dur_max, bw_min, bw_max, prom_min)
        for idxs, st in zip(window_idxs, start_times)
    ]
    if n_jobs == 1 or len(seg_args) <= 1:
        tables = [_run_one_segment(*a) for a in seg_args]
    else:
        tables = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(_run_one_segment)(*a) for a in seg_args
        )
    if verbose:
        print(f"[fork]   pass-1 extract took {time.time()-tic:.1f}s")

    tables = [t for t in tables if not t.empty]
    if not tables:
        stats = pd.DataFrame()
    else:
        stats = pd.concat(tables, ignore_index=True)
        if "label" in stats.columns:
            del stats["label"]
    # Return stats in the slice-relative time frame (start_times come from
    # process_segments_params which operates on stimes_rel). Caller shifts
    # to absolute if desired. Same for stimes.
    return stats, spect_baseline, stimes_rel, sfreqs
