"""SO-power time series — port of computeSOpower.m.

MATLAB pipeline:
    1. Zero-out excluded samples to NaN
    2. Multitaper spectrogram with tapers=(5,9), window=(5, 0.5)s,
       freq_range=(0.3, 1.5), detrend='linear'
    3. Integrate power across the freq band → total (nan-summed, * df)
    4. Convert to dB: 10*log10(.)
    5. Interpolate stages (previous) onto window-center times
    6. Outlier: |z| >= 3 → NaN
    7. Normalize (default 'p2shift1234'): subtract 2nd percentile of SO-power
       over stages {1,2,3,4} (NREM+REM)
    8. Upsample to EEG rate via linear interp, restore NaN at excluded times
"""

from __future__ import annotations

import re
import numpy as np
from scipy.interpolate import interp1d

from pydynamo.spectrogram import _mts, _next_pow2


def _nan_pow2db(x: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return 10.0 * np.log10(np.where(x > 0, x, np.nan))


def _nan_zscore(x: np.ndarray) -> np.ndarray:
    mu = np.nanmean(x)
    sd = np.nanstd(x, ddof=1)
    if sd == 0 or not np.isfinite(sd):
        return np.zeros_like(x)
    return (x - mu) / sd


def compute_so_power(
    eeg: np.ndarray,
    fs: float,
    *,
    stage_times: np.ndarray | None = None,
    stage_vals: np.ndarray | None = None,
    eeg_times: np.ndarray | None = None,
    time_range: tuple[float, float] | None = None,
    isexcluded: np.ndarray | None = None,
    SO_freqrange: tuple[float, float] = (0.3, 1.5),
    tapers: tuple[float, int] = (5, 9),
    window_params: tuple[float, float] = (5.0, 0.5),
    SOpower_outlier_threshold: float = 3.0,
    norm_method: str = "p2shift1234",
    retain_Fs: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, float | None]:
    """Return (SOpower_norm, SOpower_times, SOpower_stages, norm_method, ptile)."""
    eeg = np.ascontiguousarray(np.asarray(eeg, dtype=np.float64).ravel())
    fs = float(fs)
    n = eeg.size
    if eeg_times is None:
        eeg_times = np.arange(n) / fs
    else:
        eeg_times = np.asarray(eeg_times, dtype=np.float64).ravel()
        assert eeg_times.size == n

    if time_range is None:
        time_range = (float(eeg_times[0]), float(eeg_times[-1]))

    if isexcluded is None:
        isexcluded = np.zeros(n, dtype=bool)
    else:
        isexcluded = np.asarray(isexcluded, dtype=bool).ravel()

    nan_eeg = eeg.copy()
    nan_eeg[isexcluded] = np.nan

    # Multitaper spectrogram over SO band. MATLAB uses detrend='linear' for
    # SO-power (not 'constant' like the main TF spectrogram), and lets the
    # multitaper fn choose nfft when passed []. We mirror that by calling
    # _mts directly.
    so_spect, stimes, sfreqs = _mts(
        nan_eeg, fs, list(SO_freqrange),
        float(tapers[0]), int(tapers[1]), list(window_params),
        min_nfft=0, detrend_opt="linear",
        multiprocess=True, n_jobs=None, weighting="unity",
        plot_on=False, verbose=False, xyflip=False,
    )
    so_spect = np.ascontiguousarray(so_spect, dtype=np.float64)

    # Shift stimes to absolute time (MATLAB: SOpower_times += EEG_times(1))
    SOpower_times = stimes + float(eeg_times[0])

    # Total power across freq band * df (MATLAB sums with implicit NaN
    # handling via multitaper's own NaN handling; do a nan-safe sum).
    df = float(sfreqs[1] - sfreqs[0])
    SOpower = np.nansum(so_spect, axis=0) * df
    SOpower_db = _nan_pow2db(SOpower)

    # Stage assignment ('previous' interp)
    if stage_times is not None and stage_vals is not None and len(stage_times):
        stage_vals = np.asarray(stage_vals, dtype=float).ravel()
        stage_times = np.asarray(stage_times, dtype=float).ravel()
        stage_interp = interp1d(
            stage_times, stage_vals, kind="previous",
            bounds_error=False, fill_value=np.nan, assume_sorted=True,
        )
        SOpower_stages = stage_interp(SOpower_times)
        SOpower_stages = np.where(np.isnan(SOpower_stages), 0.0, SOpower_stages)
    else:
        SOpower_stages = np.ones_like(SOpower_times, dtype=float)

    # Outlier exclusion
    z = _nan_zscore(SOpower_db)
    SOpower_db_clean = SOpower_db.copy()
    SOpower_db_clean[np.abs(z) >= SOpower_outlier_threshold] = np.nan

    if np.all(np.isnan(SOpower_db_clean)):
        # Degenerate — keep MATLAB behaviour of warning + returning.
        return SOpower_db_clean, SOpower_times, SOpower_stages, "nan", None

    # Normalization
    m = re.match(r"^p(0*[0-9]|[1-9][0-9]|100)shift([1-5]+)$", norm_method)
    ptile = None
    in_range = (SOpower_times >= time_range[0]) & (SOpower_times <= time_range[1])
    if m:
        shift_ptile = float(m.group(1))
        shift_stages = sorted({int(c) for c in m.group(2)}) or list(range(1, 5))
        valid_stage = np.isin(SOpower_stages, shift_stages)
        sel = in_range & valid_stage
        if not sel.any():
            raise ValueError(
                f"No valid stages {shift_stages} found for shift normalization."
            )
        # MATLAB `prctile` uses Hyndman-Fan method #5 ("hazen"). numpy's
        # default is method #7 ("linear"). Use hazen to match MATLAB bit-ish.
        ptile = float(np.nanpercentile(
            SOpower_db_clean[sel], shift_ptile, method="hazen"))
        SOpower_norm = SOpower_db_clean - ptile
    elif norm_method in ("percent", "percentile", "%", "%SOP"):
        ptile_lo, ptile_hi = np.nanpercentile(
            SOpower_db_clean[in_range], [1, 99], method="hazen"
        )
        SOpower_norm = (SOpower_db_clean - ptile_lo) / (ptile_hi - ptile_lo)
        ptile = (float(ptile_lo), float(ptile_hi))
    elif norm_method in ("none", "absolute"):
        SOpower_norm = SOpower_db_clean
    else:
        raise ValueError(f"Normalization method {norm_method!r} not recognized.")

    # Upsample back to EEG rate
    if retain_Fs:
        valid = ~np.isnan(SOpower_norm)
        if not valid.any():
            return SOpower_norm, SOpower_times, SOpower_stages, norm_method, ptile
        vals = SOpower_norm[valid]
        xp = np.concatenate(([eeg_times[0]], SOpower_times[valid], [eeg_times[-1]]))
        fp = np.concatenate(([vals[0]], vals, [vals[-1]]))
        up = np.interp(eeg_times, xp, fp)
        up[isexcluded] = np.nan
        SOpower_norm = up
        SOpower_times = eeg_times
        if stage_times is not None and stage_vals is not None and len(stage_times):
            SOpower_stages = stage_interp(SOpower_times)
            SOpower_stages = np.where(np.isnan(SOpower_stages), 0.0, SOpower_stages)
        else:
            SOpower_stages = np.ones_like(SOpower_times, dtype=float)

    return SOpower_norm, SOpower_times, SOpower_stages, norm_method, ptile
