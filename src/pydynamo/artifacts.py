"""Artifact detection — port of toolbox/helper_functions/artifact_detection/detect_artifacts.m

DYNAMO calls it as::

    detect_artifacts(data, Fs, 'hpFilt_high', hpf_hi, 'hpFilt_broad', hpf_bb)

with all other parameters at defaults:
    zscore_method = 'robust'       (median / MAD)
    hf_crit = 5.5,  hf_pass = 35
    bb_crit = 5.5,  bb_pass = 0.1
    hf_detrend = bb_detrend = True
    slope_test = True              (disabled in this port — v1)
    smooth_duration = 2 s
    detrend_duration = 300 s (5 min)
    buffer_duration = 0
    hpFilt_* = highpassiir order-4 Chebyshev-I, 0.2 dB passband ripple

Algorithm per frequency band (`_compute_band_artifacts`):
    1. filtfilt with the Chebyshev-I highpass
    2. abs(hilbert(·)) → envelope
    3. movmean over `smooth_duration` s
    4. log
    5. movmedian over `detrend_duration` s → subtract (detrend)
    6. iterative robust z-score: until no |z| > crit, mark those samples bad,
       recompute median/MAD on the remaining good samples.

Masks from HF and BB bands are OR'd with `bad_inds` (flat runs + outlier noise).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import cheby1, sosfiltfilt, hilbert
from scipy.ndimage import median_filter


def _flat_mask(data: np.ndarray, min_run: int) -> np.ndarray:
    """True where a sample sits inside a run of ≥ `min_run` identical values."""
    n = data.size
    if n == 0:
        return np.zeros(0, dtype=bool)
    # Boundaries between equal-value runs
    change = np.empty(n, dtype=bool)
    change[0] = True
    change[1:] = data[1:] != data[:-1]
    run_starts = np.flatnonzero(change)
    run_lengths = np.diff(np.append(run_starts, n))
    mask = np.zeros(n, dtype=bool)
    for start, length in zip(run_starts, run_lengths):
        if length >= min_run:
            mask[start : start + length] = True
    return mask


def _find_outlier_noise(data: np.ndarray, bad: np.ndarray,
                        outlier_scalar: float = 10.0) -> np.ndarray:
    """Extend `bad` with samples more than `outlier_scalar` SD from the mean
    (computed on currently-good samples)."""
    good = ~bad
    if not good.any():
        return bad
    mu = float(np.mean(data[good]))
    sd = float(np.std(data[good], ddof=1))  # MATLAB std uses N-1
    lo = mu - outlier_scalar * sd
    hi = mu + outlier_scalar * sd
    return bad | (data <= lo) | (data >= hi)


def _movmean(x: np.ndarray, win: int) -> np.ndarray:
    """Match MATLAB `movmean(x, win)`: centered moving average with
    shrinking window at the endpoints (partial means on the edges)."""
    n = x.size
    if win <= 1 or n == 0:
        return x.astype(float, copy=True)
    half_l = (win - 1) // 2
    half_r = win // 2
    # cumulative sum trick: sum(x[a..b]) = csum[b+1] - csum[a]
    csum = np.concatenate(([0.0], np.cumsum(x, dtype=np.float64)))
    idx = np.arange(n)
    a = np.maximum(0, idx - half_l)
    b = np.minimum(n, idx + half_r + 1)
    return (csum[b] - csum[a]) / (b - a)


def _movmedian(x: np.ndarray, win: int) -> np.ndarray:
    """Match MATLAB `movmedian(x, win)`: centered moving median with
    shrinking window at the endpoints.

    pandas.Series.rolling(...).median() is cython-backed and ~400x faster
    than scipy.ndimage.median_filter for long signals + wide windows.
    min_periods=1 gives MATLAB's shrinking-window behaviour at the edges.
    """
    import pandas as pd
    n = x.size
    if win <= 1 or n == 0:
        return x.astype(float, copy=True)
    return pd.Series(x).rolling(win, center=True, min_periods=1).median().to_numpy()


def _mad(x: np.ndarray) -> float:
    """MATLAB default `mad(x)`: mean of absolute deviation from the mean."""
    m = float(np.mean(x))
    return float(np.mean(np.abs(x - m)))


def _robust_zscore_iter(
    y: np.ndarray,
    bad_inds: np.ndarray,
    crit: float,
    zscore_method: str = "robust",
) -> np.ndarray:
    """Iteratively flag samples where |z| > crit, recomputing the centering
    stats on the shrinking good set until convergence.

    Returns the updated artifact mask (includes input bad_inds).
    """
    mask = bad_inds.copy()

    def _center_scale(vals: np.ndarray) -> tuple[float, float]:
        if zscore_method == "standard":
            return float(np.mean(vals)), float(np.std(vals, ddof=1))
        # robust: median / MAD (MATLAB `mad` = mean abs dev from mean)
        return float(np.median(vals)), _mad(vals)

    good = ~mask
    if not good.any():
        return mask
    mid, scale = _center_scale(y[good])
    if scale == 0:
        return mask

    z = (y - mid) / scale
    over = (np.abs(z) > crit) & good

    while over.any():
        mask |= over
        good = ~mask
        if not good.any():
            break
        mid, scale = _center_scale(y[good])
        if scale == 0:
            break
        z = (y - mid) / scale
        over = (np.abs(z) > crit) & good

    return mask


def _compute_band_artifacts(
    data: np.ndarray,
    fs: float,
    passband: float,
    crit: float,
    bad_inds: np.ndarray,
    smooth_duration: float = 2.0,
    detrend_duration: float = 300.0,
    detrend_on: bool = True,
    zscore_method: str = "robust",
) -> np.ndarray:
    """Detect artifacts in one frequency band. Mirrors the MATLAB
    `compute_artifacts` nested function (detect_artifacts.m:221-355)."""
    # Cheby-I IIR highpass at order 4 with 0.2 dB passband ripple (MATLAB
    # designfilt('highpassiir', 'FilterOrder', 4, 'PassbandFrequency', pb,
    #            'PassbandRipple', 0.2, 'SampleRate', Fs)).
    sos = cheby1(4, 0.2, passband, btype="highpass", fs=fs, output="sos")
    y = sosfiltfilt(sos, data)
    y = np.abs(hilbert(y))
    y = _movmean(y, int(round(smooth_duration * fs)))
    # log (MATLAB: log = natural log)
    with np.errstate(divide="ignore", invalid="ignore"):
        y = np.log(y)
    # detrend via rolling median
    if detrend_on:
        y = y - _movmedian(y, int(round(detrend_duration * fs)))
    return _robust_zscore_iter(y, bad_inds, crit, zscore_method=zscore_method)


def _slope_test(data: np.ndarray, fs: float, slope_crit: float = -0.5) -> np.ndarray:
    """Port of the slope-test block in detect_artifacts.m:142-152.

    Compute a coarse multitaper spectrogram (TW=10, K=19, window 10 s,
    step 5 s, freq range [1, min(55, fs/2)]), fit a line in log-log space
    per window, and flag windows where `slope > slope_crit` as bad.
    Interpolate nearest-neighbour to data rate. Samples outside the coarse
    grid (e.g. at the very ends) are marked bad.

    Returns boolean mask same length as `data`.
    """
    n = data.size
    fs = float(fs)
    fmax = min(55.0, fs / 2.0)
    if fmax <= 1.0:
        # Can't do slope test on very low-rate data
        return np.zeros(n, dtype=bool)
    # Lazy-import mtm to avoid circular import (spectrogram depends on us).
    from pydynamo.spectrogram import _mts
    spect, stimes_rel, sfreqs = _mts(
        np.ascontiguousarray(data, dtype=np.float64), fs,
        [1.0, fmax],
        10.0,    # time_bandwidth (TW)
        19,      # num_tapers (K = 2*TW - 1)
        [10.0, 5.0],         # window size, step (s)
        min_nfft=0,          # let MTS pick
        detrend_opt="linear",
        multiprocess=True, n_jobs=None, weighting="unity",
        plot_on=False, verbose=False, xyflip=False,
    )
    spect = np.ascontiguousarray(spect, dtype=np.float64)
    sfreqs = np.asarray(sfreqs, dtype=np.float64).ravel()
    stimes_rel = np.asarray(stimes_rel, dtype=np.float64).ravel()

    # Drop any freq bins where values are non-positive across the whole record
    # (log of non-positive is undefined); keep the rest for polyfit.
    log_f = np.log(sfreqs)
    # polyfit per column: for each window t, fit log(spect[:, t]) ~ m*log_f + b.
    # Use least-squares solution vectorized: m = cov(log_f, log_s) / var(log_f)
    # Faster than np.polyfit in a loop on 1000+ windows.
    with np.errstate(divide="ignore", invalid="ignore"):
        log_s = np.log(spect)
    # Guard: replace -inf (from 0 entries) with NaN; skip windows with any NaN.
    log_s = np.where(np.isfinite(log_s), log_s, np.nan)

    # Per-window slope via least squares: slope = sum((x-xbar)*(y-ybar)) / sum((x-xbar)^2)
    # where x = log_f, y = log_s[:, t].
    x = log_f
    xbar = x.mean()
    x_c = x - xbar
    denom = float((x_c * x_c).sum())
    # y-center per column
    col_valid = np.isfinite(log_s).all(axis=0)
    ybar = np.where(col_valid, np.nanmean(log_s, axis=0), np.nan)
    log_s_centered = log_s - ybar[None, :]
    slopes = (x_c[:, None] * log_s_centered).sum(axis=0) / denom
    bad_stimes = slopes > slope_crit       # bool per window
    # Windows with any NaN data → mark bad
    bad_stimes |= ~col_valid

    # Interpolate nearest-neighbor from stimes grid → sample grid
    t_data = np.arange(n) / fs
    # Nearest-neighbor lookup
    idx = np.searchsorted(stimes_rel, t_data)
    idx = np.clip(idx, 0, stimes_rel.size - 1)
    left = np.clip(idx - 1, 0, stimes_rel.size - 1)
    use_left = np.abs(stimes_rel[left] - t_data) < np.abs(stimes_rel[idx] - t_data)
    nearest = np.where(use_left, left, idx)
    bad_samples = bad_stimes[nearest]
    # Samples OUTSIDE the coarse grid (t before stimes[0] or after stimes[-1])
    # are always bad — MATLAB's `bad_slope(isnan(bad_slope)) = 1` does that.
    outside = (t_data < stimes_rel[0]) | (t_data > stimes_rel[-1])
    bad_samples |= outside
    return bad_samples


def detect_artifacts(
    data: np.ndarray,
    fs: float,
    *,
    hf_pass: float = 35.0,
    hf_crit: float = 5.5,
    bb_pass: float = 0.1,
    bb_crit: float = 5.5,
    hf_detrend: bool = True,
    bb_detrend: bool = True,
    zscore_method: str = "robust",
    smooth_duration: float = 2.0,
    detrend_duration: float = 300.0,
    buffer_duration: float = 0.0,
    slope_test: bool = True,
    slope_crit: float = -0.5,
    isexcluded: np.ndarray | None = None,
) -> np.ndarray:
    """Detect artifacts in an EEG time series.

    Matches MATLAB `detect_artifacts(data, Fs, 'hpFilt_high', ..., 'hpFilt_broad', ...)`
    with default arguments (slope_test ON by default, matching MATLAB).

    Returns a boolean 1-D mask the same length as `data`, True at artifact
    samples.
    """
    data = np.ascontiguousarray(np.asarray(data, dtype=np.float64).ravel())
    fs = float(fs)
    n = data.size
    if isexcluded is None:
        isexcluded = np.zeros(n, dtype=bool)
    else:
        isexcluded = np.asarray(isexcluded, dtype=bool).ravel()
        assert isexcluded.size == n, "isexcluded must match len(data)"

    # Flat runs + outlier noise
    flat = _flat_mask(data, int(round(fs)))  # ≥ 1 s of identical samples
    bad = np.isnan(data) | np.isinf(data) | flat
    # Slope test (MATLAB default ON) — flag 1/f-violating windows.
    if slope_test:
        bad |= _slope_test(data, fs, slope_crit=slope_crit)
    bad = _find_outlier_noise(data, bad)

    # Interpolate through bad samples so the filters don't see NaN / flats
    data_fixed = data.copy()
    if bad.any():
        idx = np.arange(n)
        good_idx = idx[~bad]
        good_vals = data[~bad]
        # pad endpoints so interp1d is fully within-range
        xp = np.concatenate(([-1.0], good_idx, [n]))
        fp = np.concatenate(([good_vals[0]], good_vals, [good_vals[-1]]))
        data_fixed[bad] = np.interp(idx[bad], xp, fp)

    hf_art = _compute_band_artifacts(
        data_fixed, fs, hf_pass, hf_crit, bad,
        smooth_duration=smooth_duration,
        detrend_duration=detrend_duration,
        detrend_on=hf_detrend,
        zscore_method=zscore_method,
    )
    bb_art = _compute_band_artifacts(
        data_fixed, fs, bb_pass, bb_crit, bad,
        smooth_duration=smooth_duration,
        detrend_duration=detrend_duration,
        detrend_on=bb_detrend,
        zscore_method=zscore_method,
    )
    artifacts = hf_art | bb_art | bad

    if buffer_duration > 0:
        # Dilate each run of True by buffer_duration * fs samples on both sides
        k = int(round(buffer_duration * fs))
        if k > 0 and artifacts.any():
            from scipy.ndimage import binary_dilation
            artifacts = binary_dilation(artifacts, iterations=k)

    return artifacts
