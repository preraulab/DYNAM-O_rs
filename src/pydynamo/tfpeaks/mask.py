"""Mask a pass-2 spectrogram using pass-1 region pixels.

Port of computeTFPeaks.m:maskSpectrogram:

    % STEP 1: paint region pixels (interior + border, from Ldata2graph's
    %         union(rgn, Lborders))
    spect_masked(region_inds) = spect(region_inds);
    % STEP 2: zero the 1-pixel perimeter of each trimmed region
    spect_masked(border_inds) = 0;

Implementation: hot path delegates to `dynamo_rs.mask_spectrogram` (Rust).
Python fallback kept for environments without the Rust extension.
"""

from __future__ import annotations

import numpy as np
from skimage.segmentation import find_boundaries

try:
    import dynamo_rs as _dynamo_rs
    _HAS_RUST = True
except ImportError:
    _dynamo_rs = None
    _HAS_RUST = False


def mask_spectrogram(
    spect_2s: np.ndarray,           # (F, T_2)
    stimes_2s: np.ndarray,
    labels_1s: np.ndarray,          # (F, T_1) from pass-1
    stimes_1s: np.ndarray,
) -> np.ndarray:
    """Return a copy of spect_2s with all pixels outside any pass-1 region
    zeroed out, AND with the 1-pixel perimeter of each region zeroed out
    (matching MATLAB's `maskSpectrogram` STEP 2).
    """
    assert spect_2s.shape[0] == labels_1s.shape[0], \
        "pass-1 and pass-2 spectrograms must share freq axis"

    if _HAS_RUST:
        spect_2s = np.ascontiguousarray(np.asarray(spect_2s, dtype=np.float64))
        stimes_2s = np.ascontiguousarray(np.asarray(stimes_2s, dtype=np.float64).ravel())
        labels_1s = np.ascontiguousarray(np.asarray(labels_1s, dtype=np.int64))
        stimes_1s = np.ascontiguousarray(np.asarray(stimes_1s, dtype=np.float64).ravel())
        return _dynamo_rs.mask_spectrogram(spect_2s, stimes_2s, labels_1s, stimes_1s)

    # ---- Python fallback ----
    idx = np.searchsorted(stimes_1s, stimes_2s)
    idx = np.clip(idx, 0, labels_1s.shape[1] - 1)
    left = np.clip(idx - 1, 0, labels_1s.shape[1] - 1)
    use_left = np.abs(stimes_1s[left] - stimes_2s) < np.abs(stimes_1s[idx] - stimes_2s)
    nearest = np.where(use_left, left, idx)
    labels_on_2s = labels_1s[:, nearest]
    perimeter = find_boundaries(labels_on_2s, mode="inner", connectivity=2)
    masked = np.where(labels_on_2s > 0, spect_2s, 0.0)
    masked[perimeter] = 0.0
    return masked
