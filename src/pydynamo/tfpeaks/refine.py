"""Hann-window peak frequency refinement.

Port of toolbox/TFpeak_functions/refinePeakFrequency.m + hann_event_spectra.m.

One vectorized Hann FFT over all event windows at once (not per-peak — MATLAB
does tens of thousands of individual FFTs via parfor; we do a single
(n_events × nfft) rfft batch). Then per-peak spline interpolation within
the bounding-box frequency range to get sub-bin frequency.

Default params mirror MATLAB:
    window_size = 4.0 s
    dsfreqs = 0.05 Hz  → nfft = 2**nextpow2(Fs/dsfreqs)
    detrend_opt = 'constant'
    refine_method = 'spline_interp'  (1000-point linspace + np.interp cubic)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.fft import rfft
from scipy.interpolate import CubicSpline
from scipy.signal.windows import hann


def _next_pow2(x: float) -> int:
    return 1 << int(np.ceil(np.log2(max(x, 1))))


def _hann_event_spectra(
    data: np.ndarray,
    fs: float,
    event_times: np.ndarray,
    t: np.ndarray,
    *,
    freq_range: tuple[float, float] = (0.0, 30.0),
    window_size: float = 4.0,
    dsfreqs: float = 0.05,
    detrend_opt: str = "constant",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a Hann-tapered one-sided PSD at each event center in one
    vectorized FFT batch. Returns (spect (F, N_events), sfreqs)."""
    data = np.ascontiguousarray(np.asarray(data, dtype=np.float64).ravel())
    t = np.asarray(t, dtype=np.float64).ravel()
    event_times = np.asarray(event_times, dtype=np.float64).ravel()
    fs = float(fs)

    win_samples = int(round(window_size * fs))
    half = win_samples // 2
    nfft = _next_pow2(fs / dsfreqs)

    # Start sample for each event's window (floor, 0-based)
    start_idx = np.floor((event_times - t[0] - window_size / 2.0) * fs).astype(np.int64)
    assert (start_idx >= 0).all() and (start_idx + win_samples <= data.size).all(), \
        "event windows must fall within the data range"

    # (N_events, win_samples) slice matrix
    N = event_times.size
    idx = start_idx[:, None] + np.arange(win_samples)[None, :]
    segments = data[idx]  # (N, win_samples)

    # Detrend
    if detrend_opt == "constant":
        segments = segments - segments.mean(axis=1, keepdims=True)
    elif detrend_opt == "linear":
        # linear detrend per row: fit y = a*x + b and subtract
        x = np.arange(win_samples)
        xm = x.mean()
        xv = ((x - xm) ** 2).sum()
        ym = segments.mean(axis=1, keepdims=True)
        slope = ((segments - ym) * (x - xm)).sum(axis=1, keepdims=True) / xv
        segments = segments - (slope * (x - xm) + ym)
    # "off" → no detrend

    # Hann taper, normalized (matches MATLAB: hann_taper / sqrt(sum(h^2)))
    taper = hann(win_samples, sym=True)
    taper = taper / np.sqrt((taper ** 2).sum())
    segments = segments * taper[None, :]

    # FFT (N, nfft // 2 + 1 bins for real FFT)
    fft_full = rfft(segments, n=nfft, axis=1)
    power = fft_full.real ** 2 + fft_full.imag ** 2
    # Convert to one-sided PSD (MATLAB: [DC; 2*mid; Nyquist] / Fs)
    power_1s = np.empty_like(power)
    power_1s[:, 0] = power[:, 0] / fs
    power_1s[:, -1] = power[:, -1] / fs
    power_1s[:, 1:-1] = 2.0 * power[:, 1:-1] / fs

    # Build sfreqs and slice to freq_range
    sfreqs_full = np.arange(nfft // 2 + 1) * (fs / nfft)
    mask = (sfreqs_full >= freq_range[0]) & (sfreqs_full <= freq_range[1])
    sfreqs = sfreqs_full[mask]
    spect = power_1s[:, mask].T.copy()  # (F, N)
    return spect, sfreqs


def refine_peak_frequency(
    stats_table: pd.DataFrame,
    data: np.ndarray,
    fs: float,
    t: np.ndarray | None = None,
    *,
    freq_range: tuple[float, float] = (0.0, 30.0),
    window_size: float = 4.0,
    dsfreqs: float = 0.05,
    refine_method: str = "spline_interp",
    remove_edge_peaks: bool = True,
    n_spline_pts: int = 1000,
) -> pd.DataFrame:
    """Refine `PeakFrequency` in-place (returns copy) using a Hann-window
    sub-bin estimate per peak."""
    if stats_table.empty:
        return stats_table
    data = np.ascontiguousarray(np.asarray(data, dtype=np.float64).ravel())
    fs = float(fs)
    if t is None:
        t = np.arange(data.size) / fs

    stats = stats_table.copy()
    event_times = stats["PeakTime"].to_numpy()
    # BoundingBox = (time_tl, freq_tl, width_s, height_Hz) per peak
    bbox = np.asarray(list(stats["BoundingBox"]), dtype=float)
    bbox_lo = bbox[:, 1]
    bbox_h = bbox[:, 3]

    # Keep only events whose ±window/2 fits inside the data
    half = window_size / 2.0
    keep = (event_times > t[0] + half) & (event_times < t[-1] - half)

    if not keep.any():
        stats["PeakFrequency"] = np.nan
        return stats

    spect, sfreqs = _hann_event_spectra(
        data, fs, event_times[keep], t,
        freq_range=freq_range, window_size=window_size,
        dsfreqs=dsfreqs, detrend_opt="constant",
    )

    refined = np.full(event_times.size, np.nan, dtype=float)

    for i, idx in enumerate(np.flatnonzero(keep)):
        start_freq = float(bbox_lo[idx])
        end_freq = float(bbox_lo[idx] + bbox_h[idx])
        if end_freq <= start_freq:
            continue
        # Frequency indices inside [start, end]
        range_inds = (sfreqs >= start_freq) & (sfreqs <= end_freq)
        if range_inds.sum() < 2:
            continue

        curr = spect[:, i]
        if refine_method == "spline_interp":
            # Fit spline on the full spectrum, then evaluate on dense grid
            # inside the bounding-box range and find the argmax. MATLAB uses
            # interp1(..., 'spline') which is a cubic spline.
            try:
                cs = CubicSpline(sfreqs, curr, extrapolate=False)
            except ValueError:
                continue
            grid = np.linspace(start_freq, end_freq, n_spline_pts)
            vals = cs(grid)
            refined[idx] = float(grid[np.nanargmax(vals)])
        elif refine_method == "spect_max":
            refined[idx] = float(sfreqs[range_inds][np.argmax(curr[range_inds])])
        else:
            raise ValueError(f"unknown refine_method {refine_method!r}")

        # Edge-peak rejection (MATLAB removes peaks within 1e-3 of the
        # bounding-box freq boundaries, as those are likely artifacts)
        if remove_edge_peaks:
            if (abs(refined[idx] - start_freq) < 1e-3 or
                abs(refined[idx] - end_freq) < 1e-3 or
                refined[idx] > end_freq or end_freq < start_freq):
                refined[idx] = np.nan

    stats["PeakFrequency"] = refined
    return stats.dropna(subset=["PeakFrequency"]).reset_index(drop=True)
