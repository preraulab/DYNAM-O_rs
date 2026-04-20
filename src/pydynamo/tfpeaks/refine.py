"""Hann-window peak frequency refinement.

Port of toolbox/TFpeak_functions/refinePeakFrequency.m + hann_event_spectra.m.

Hot path delegates to `dynamo_rs.hann_event_spectra` + `dynamo_rs.refine_from_spectra`
(FFT via realfft, rayon-parallel cubic-spline argmax per event). Python
fallback retained for environments without the Rust extension.

Default params mirror MATLAB:
    window_size = 4.0 s
    dsfreqs = 0.05 Hz  → nfft = 2**nextpow2(Fs/dsfreqs)
    detrend_opt = 'constant'
    refine_method = 'spline_interp'  (1000-point grid + natural cubic spline)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.fft import rfft
from scipy.interpolate import CubicSpline
from scipy.signal.windows import hann

try:
    import dynamo_rs as _dynamo_rs
    _HAS_RUST = hasattr(_dynamo_rs, "hann_event_spectra") and hasattr(
        _dynamo_rs, "refine_from_spectra"
    )
except ImportError:
    _dynamo_rs = None
    _HAS_RUST = False


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
    event_times = np.ascontiguousarray(np.asarray(event_times, dtype=np.float64).ravel())
    fs = float(fs)

    if _HAS_RUST:
        spect, sfreqs = _dynamo_rs.hann_event_spectra(
            data, fs, event_times, float(t[0]),
            (float(freq_range[0]), float(freq_range[1])),
            float(window_size), float(dsfreqs), detrend_opt,
        )
        return spect, sfreqs

    # ---- Python fallback ----
    win_samples = int(round(window_size * fs))
    nfft = _next_pow2(fs / dsfreqs)

    start_idx = np.floor((event_times - t[0] - window_size / 2.0) * fs).astype(np.int64)
    assert (start_idx >= 0).all() and (start_idx + win_samples <= data.size).all(), \
        "event windows must fall within the data range"

    N = event_times.size
    idx = start_idx[:, None] + np.arange(win_samples)[None, :]
    segments = data[idx]

    if detrend_opt == "constant":
        segments = segments - segments.mean(axis=1, keepdims=True)
    elif detrend_opt == "linear":
        x = np.arange(win_samples)
        xm = x.mean()
        xv = ((x - xm) ** 2).sum()
        ym = segments.mean(axis=1, keepdims=True)
        slope = ((segments - ym) * (x - xm)).sum(axis=1, keepdims=True) / xv
        segments = segments - (slope * (x - xm) + ym)

    taper = hann(win_samples, sym=True)
    taper = taper / np.sqrt((taper ** 2).sum())
    segments = segments * taper[None, :]

    fft_full = rfft(segments, n=nfft, axis=1)
    power = fft_full.real ** 2 + fft_full.imag ** 2
    power_1s = np.empty_like(power)
    power_1s[:, 0] = power[:, 0] / fs
    power_1s[:, -1] = power[:, -1] / fs
    power_1s[:, 1:-1] = 2.0 * power[:, 1:-1] / fs

    sfreqs_full = np.arange(nfft // 2 + 1) * (fs / nfft)
    mask = (sfreqs_full >= freq_range[0]) & (sfreqs_full <= freq_range[1])
    sfreqs = sfreqs_full[mask]
    spect = power_1s[:, mask].T.copy()
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
    """Refine `PeakFrequency` (returns a copy) using a Hann-window sub-bin
    estimate per peak."""
    if stats_table.empty:
        return stats_table
    data = np.ascontiguousarray(np.asarray(data, dtype=np.float64).ravel())
    fs = float(fs)
    if t is None:
        t = np.arange(data.size) / fs

    stats = stats_table.copy()
    event_times = stats["PeakTime"].to_numpy()
    bbox = np.asarray(list(stats["BoundingBox"]), dtype=float)
    bbox_lo_all = bbox[:, 1]
    bbox_h_all = bbox[:, 3]

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

    if _HAS_RUST and refine_method == "spline_interp":
        # Run the per-event spline+argmax loop in Rust (rayon-parallel).
        kept_lo = np.ascontiguousarray(bbox_lo_all[keep], dtype=np.float64)
        kept_hi = np.ascontiguousarray(bbox_lo_all[keep] + bbox_h_all[keep], dtype=np.float64)
        out = _dynamo_rs.refine_from_spectra(
            np.ascontiguousarray(spect, dtype=np.float64),
            np.ascontiguousarray(sfreqs, dtype=np.float64),
            kept_lo, kept_hi,
            int(n_spline_pts), bool(remove_edge_peaks),
        )
        kept_idx = np.flatnonzero(keep)
        refined[kept_idx] = out
        stats["PeakFrequency"] = refined
        return stats.dropna(subset=["PeakFrequency"]).reset_index(drop=True)

    # ---- Python fallback ----
    for i, idx in enumerate(np.flatnonzero(keep)):
        start_freq = float(bbox_lo_all[idx])
        end_freq = float(bbox_lo_all[idx] + bbox_h_all[idx])
        if end_freq <= start_freq:
            continue
        range_inds = (sfreqs >= start_freq) & (sfreqs <= end_freq)
        if range_inds.sum() < 2:
            continue
        curr = spect[:, i]
        if refine_method == "spline_interp":
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
        if remove_edge_peaks:
            if (abs(refined[idx] - start_freq) < 1e-3 or
                abs(refined[idx] - end_freq) < 1e-3 or
                refined[idx] > end_freq or end_freq < start_freq):
                refined[idx] = np.nan

    stats["PeakFrequency"] = refined
    return stats.dropna(subset=["PeakFrequency"]).reset_index(drop=True)
