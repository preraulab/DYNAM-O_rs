"""Multitaper spectrogram wrapper — delegates to the DYNAMO_dev submodule.

Rather than reimplement the wrapper around multitaper_rs, we import the
canonical Prerau-lab `multitaper_spectrogram` function already vetted for
MATLAB equivalence. It lives as a submodule of DYNAM-O at

    ~/code/toolboxes/DYNAMO_dev/toolbox/helper_functions/multitaper_spectrogram/python/

and internally dispatches to the Rust kernel (multitaper_rs) when available.

DYNAM-O's MATLAB call (computeTFPeaks.m:432-435):
    multitaper_spectrogram(data, Fs, freq_range, taper_params,
                           time_window_params, nfft, 'constant', 'unity',
                           false, false)
    nfft = 2^nextpow2(Fs / dsfreqs)      # dsfreqs = 0.1 → nfft = 1024 at Fs=100
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_SUBMODULE_PATH = Path(
    "/Users/Mike/code/toolboxes/DYNAMO_dev/toolbox/helper_functions/multitaper_spectrogram/python"
)
if str(_SUBMODULE_PATH) not in sys.path:
    sys.path.insert(0, str(_SUBMODULE_PATH))

from multitaper_spectrogram_python import multitaper_spectrogram as _mts  # noqa: E402


def _next_pow2(x: float) -> int:
    return 1 << int(np.ceil(np.log2(x)))


def mtm_spectrogram(
    data: np.ndarray,
    fs: float,
    freq_range: tuple[float, float] = (0.0, 30.0),
    taper_params: tuple[float, int] = (2, 3),
    window_params: tuple[float, float] = (1.0, 0.05),
    dsfreqs: float = 0.1,
    multiprocess: bool = True,
    n_jobs: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a DYNAM-O-compatible multitaper spectrogram.

    Returns
    -------
    spect : (num_freqs, num_windows) float64 — PSD-scaled
    stimes : (num_windows,) float64 — window centers (s)
    sfreqs : (num_freqs,) float64 — frequency bins (Hz)
    """
    data = np.ascontiguousarray(np.asarray(data, dtype=np.float64).ravel())
    nfft = _next_pow2(float(fs) / float(dsfreqs))

    spect, stimes, sfreqs = _mts(
        data,
        float(fs),
        list(freq_range),
        float(taper_params[0]),      # time_bandwidth
        int(taper_params[1]),         # num_tapers
        list(window_params),
        min_nfft=int(nfft),
        detrend_opt="constant",
        multiprocess=multiprocess,
        n_jobs=n_jobs,
        weighting="unity",
        plot_on=False,
        verbose=False,
        xyflip=False,
    )
    # multitaper_spectrogram returns (F, T) with the rust backend; no flip needed.
    return np.ascontiguousarray(spect, dtype=np.float64), \
           np.ascontiguousarray(stimes, dtype=np.float64), \
           np.ascontiguousarray(sfreqs, dtype=np.float64)
