"""SO-phase time series — port of computeSOphase.m.

MATLAB uses a precomputed elliptic bandpass filter from
`SOphase_filters.mat`, designed as:

    d = designfilt('bandpassiir', 'StopbandFrequency1', 0.2,
                   'PassbandFrequency1', 0.3, 'PassbandFrequency2', 1.5,
                   'StopbandFrequency2', 1.6, 'StopbandAttenuation1', 60,
                   'PassbandRipple', 1, 'StopbandAttenuation2', 60,
                   'DesignMethod', 'ellip', 'MatchExactly', 'passband',
                   'SampleRate', Fs)

`MatchExactly='passband'` is the load-bearing bit — scipy.signal.ellip and
scipy.signal.iirdesign cannot reproduce it (they pick a different passband
edge), so the coefficients differ by up to ~2.0 per SOS section and the
filtered-timeseries cosine-similarity stalls at 0.93 vs MATLAB.

We therefore SHIP MATLAB's exact SOS for the common (Fs, SO_freqrange) pairs
as .npy files under `data_matlab_filters/`. When compute_so_phase is called
with a matching (fs, SO_freqrange), the precomputed SOS is loaded; otherwise
we fall back to scipy iirdesign and warn.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
from scipy.signal import hilbert, iirdesign, sosfiltfilt
from scipy.interpolate import interp1d


_FILTER_DIR = Path(__file__).parent.parent / "data_matlab_filters"


def _matlab_sos_filename(fs: float, so_range: tuple[float, float]) -> str:
    """Filename convention: sophase_sos_Fs{fs}_{lo}_{hi}.npy"""
    lo, hi = so_range
    return f"sophase_sos_Fs{int(round(fs))}_{lo}_{hi}.npy"


def _load_matlab_sos(fs: float, so_range: tuple[float, float]) -> np.ndarray | None:
    path = _FILTER_DIR / _matlab_sos_filename(fs, so_range)
    if path.exists():
        return np.ascontiguousarray(np.load(path), dtype=np.float64)
    return None


def _design_so_bandpass_scipy(fs: float, so_range: tuple[float, float] = (0.3, 1.5)):
    """scipy fallback — does NOT match MATLAB's `MatchExactly='passband'`.
    Produces SOphase timeseries cosine-sim ~0.93 with MATLAB.
    """
    lo, hi = so_range
    stop_lo, stop_hi = lo - 0.1, hi + 0.1
    nyq = fs / 2.0
    wp = [lo / nyq, hi / nyq]
    ws = [stop_lo / nyq, stop_hi / nyq]
    return iirdesign(wp, ws, gpass=1.0, gstop=60.0, ftype="ellip", output="sos")


def _get_sos(fs: float, so_range: tuple[float, float]) -> np.ndarray:
    """Return the MATLAB-exact SOS if shipped for this (fs, so_range);
    otherwise design via scipy and emit a warning."""
    sos = _load_matlab_sos(fs, so_range)
    if sos is not None:
        return sos
    warnings.warn(
        f"No MATLAB-exact SOphase filter shipped for Fs={fs} SO_freqrange={so_range}; "
        f"falling back to scipy iirdesign (SOphase cos ~0.93 vs MATLAB). "
        f"To fix: export MATLAB's filter SOS via scripts/export_bisect_intermediates.m "
        f"and save to {_FILTER_DIR / _matlab_sos_filename(fs, so_range)}.",
        RuntimeWarning, stacklevel=3,
    )
    return _design_so_bandpass_scipy(fs, so_range)


# Backward-compat alias for callers that imported the old name.
_design_so_bandpass = _design_so_bandpass_scipy


def compute_so_phase(
    eeg: np.ndarray,
    fs: float,
    *,
    stage_times: np.ndarray | None = None,
    stage_vals: np.ndarray | None = None,
    eeg_times: np.ndarray | None = None,
    isexcluded: np.ndarray | None = None,
    SO_freqrange: tuple[float, float] = (0.3, 1.5),
    sos_filter: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (SOphase_unwrapped, SOphase_times, SOphase_stages, filtdata).

    SOphase is returned **unwrapped** to match MATLAB — downstream code
    applies wrapToPi when binning / assigning per-peak phase.

    By default loads MATLAB's exact SOS from data_matlab_filters/ if
    available for this (fs, SO_freqrange); falls back to scipy iirdesign
    otherwise (with a RuntimeWarning).
    """
    eeg = np.ascontiguousarray(np.asarray(eeg, dtype=np.float64).ravel())
    fs = float(fs)
    n = eeg.size
    if eeg_times is None:
        eeg_times = np.arange(n) / fs
    else:
        eeg_times = np.asarray(eeg_times, dtype=np.float64).ravel()
        assert eeg_times.size == n

    if isexcluded is None:
        isexcluded = np.zeros(n, dtype=bool)
    else:
        isexcluded = np.asarray(isexcluded, dtype=bool).ravel()

    if sos_filter is not None:
        sos = np.ascontiguousarray(sos_filter, dtype=np.float64)
    else:
        sos = _get_sos(fs, SO_freqrange)

    filtdata = sosfiltfilt(sos, eeg)
    analytic = hilbert(filtdata)
    SOphase = np.unwrap(np.angle(analytic))

    filtdata_out = filtdata.copy()
    filtdata_out[isexcluded] = np.nan
    SOphase[isexcluded] = np.nan

    if stage_times is not None and stage_vals is not None and len(stage_times):
        stage_vals = np.asarray(stage_vals, dtype=float).ravel()
        stage_times_a = np.asarray(stage_times, dtype=float).ravel()
        SOphase_stages = interp1d(
            stage_times_a, stage_vals, kind="previous",
            bounds_error=False, fill_value=np.nan, assume_sorted=True,
        )(eeg_times)
        SOphase_stages = np.where(np.isnan(SOphase_stages), 0.0, SOphase_stages)
    else:
        SOphase_stages = np.ones_like(eeg_times, dtype=float)

    return SOphase, eeg_times, SOphase_stages, filtdata_out
