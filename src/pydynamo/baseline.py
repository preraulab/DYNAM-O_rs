"""Percentile-baseline subtraction — port of computeTFPeaks.m:computeBaseline.

baseline_exclude is the OR of:
    - artifact mask (from detect_artifacts, interpolated onto stimes)
    - samples in stages NOT in baseline_stages
    - any explicit user-supplied baseline_exclude

The baseline is the Nth-percentile power per frequency bin, computed over
the valid (non-excluded, in-range) spectrogram columns. Zero-valued pixels
in the spectrogram are treated as missing data (NaN) to avoid them biasing
the percentile.

The MATLAB pipeline divides the spectrogram by the broadcast baseline
(equivalent to dB-domain subtraction after 10*log10).

Implementation: the hot path delegates to `dynamo_rs.compute_baseline`
(Rust, Hyndman-Fan method #5). Python fallback preserved for environments
without the Rust extension.
"""

from __future__ import annotations

import numpy as np

try:
    import dynamo_rs as _dynamo_rs
    _HAS_RUST = True
except ImportError:
    _dynamo_rs = None
    _HAS_RUST = False


def compute_baseline(
    spect: np.ndarray,
    stimes: np.ndarray,
    t_data: np.ndarray,
    baseline_exclude: np.ndarray,
    baseline_range: tuple[float, float] = (float("-inf"), float("inf")),
    baseline_ptile: float = 2.0,
) -> np.ndarray:
    """Compute per-frequency baseline spectrum.

    Parameters
    ----------
    spect : (F, T) array — spectrogram
    stimes : (T,) array — spectrogram window-center times (s)
    t_data : (N,) array — data sample timestamps (s), same length as
             baseline_exclude
    baseline_exclude : (N,) bool — samples to exclude from baseline
    baseline_range : (start_s, end_s) — further restriction on stimes
    baseline_ptile : percentile (0-100)

    Returns
    -------
    baseline : (F, 1) array
    """
    spect = np.ascontiguousarray(np.asarray(spect, dtype=np.float64))
    stimes = np.ascontiguousarray(np.asarray(stimes, dtype=np.float64).ravel())
    t_data = np.ascontiguousarray(np.asarray(t_data, dtype=np.float64).ravel())
    baseline_exclude = np.ascontiguousarray(
        np.asarray(baseline_exclude, dtype=bool).ravel()
    )

    if _HAS_RUST:
        return _dynamo_rs.compute_baseline(
            spect, stimes, t_data, baseline_exclude,
            baseline_range=(float(baseline_range[0]), float(baseline_range[1])),
            baseline_ptile=float(baseline_ptile),
        )

    # ---- Python fallback (kept bit-equivalent to Rust for tests) ----
    idx = np.searchsorted(t_data, stimes)
    idx = np.clip(idx, 0, t_data.size - 1)
    left = np.clip(idx - 1, 0, t_data.size - 1)
    use_left = np.abs(t_data[left] - stimes) < np.abs(t_data[idx] - stimes)
    idx = np.where(use_left, left, idx)
    exclude_stimes = baseline_exclude[idx]

    in_range = (stimes >= baseline_range[0]) & (stimes <= baseline_range[1])
    valid = (~exclude_stimes) & in_range
    if not valid.any():
        raise ValueError(
            "No valid baseline time bins remain after applying artifacts, "
            "stage filtering, and baseline_range."
        )

    spect_bl = spect[:, valid].astype(np.float64, copy=True)
    spect_bl[spect_bl == 0] = np.nan
    # MATLAB `prctile` uses Hyndman-Fan method #5 ("hazen").
    return np.nanpercentile(spect_bl, baseline_ptile, axis=1,
                            keepdims=True, method="hazen")


def subtract_baseline(spect: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """Divide-in-linear == subtract-in-dB. Matches MATLAB::

        spect_baseline = spect ./ baseline  % element-wise broadcast
    """
    if _HAS_RUST:
        spect = np.ascontiguousarray(np.asarray(spect, dtype=np.float64))
        baseline = np.ascontiguousarray(np.asarray(baseline, dtype=np.float64))
        return _dynamo_rs.subtract_baseline(spect, baseline)
    return spect / baseline
