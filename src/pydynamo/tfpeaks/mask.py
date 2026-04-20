"""Mask a pass-2 spectrogram using pass-1 region pixels.

Port of computeTFPeaks.m:maskSpectrogram:

    % STEP 1: paint region pixels (interior + border, from Ldata2graph's
    %         union(rgn, Lborders))
    spect_masked(region_inds) = spect(region_inds);
    % STEP 2: zero the 1-pixel perimeter of each trimmed region
    spect_masked(border_inds) = 0;

`borders` there is `trim_borders` from trimWshedRegions — the per-region
perimeter obtained via bwboundaries on each region's binary mask (8-conn
default). Net mask = strict interior of each trimmed region (perimeter
pixels zeroed so pass-2 watershed doesn't flow across region edges).
"""

from __future__ import annotations

import numpy as np
from skimage.segmentation import find_boundaries


def mask_spectrogram(
    spect_2s: np.ndarray,           # (F, T_2)
    stimes_2s: np.ndarray,
    labels_1s: np.ndarray,          # (F, T_1) from pass-1
    stimes_1s: np.ndarray,
) -> np.ndarray:
    """Return a copy of spect_2s with all pixels outside any pass-1 region
    zeroed out, AND with the 1-pixel perimeter of each region zeroed out
    (matching MATLAB's `maskSpectrogram` STEP 2 `spect_masked(border_inds)=0`).

    Both spectrograms must share the freq axis (nfft). stimes may differ.
    """
    assert spect_2s.shape[0] == labels_1s.shape[0], \
        "pass-1 and pass-2 spectrograms must share freq axis"

    # Map each pass-2 time column to the nearest pass-1 column.
    idx = np.searchsorted(stimes_1s, stimes_2s)
    idx = np.clip(idx, 0, labels_1s.shape[1] - 1)
    left = np.clip(idx - 1, 0, labels_1s.shape[1] - 1)
    use_left = np.abs(stimes_1s[left] - stimes_2s) < np.abs(stimes_1s[idx] - stimes_2s)
    nearest = np.where(use_left, left, idx)

    # Take the corresponding pass-1 columns → (F, T_2) label map
    labels_on_2s = labels_1s[:, nearest]

    # Compute 1-pixel inner boundary of each region (matches MATLAB
    # trim_borders from bwboundaries). `find_boundaries(mode='inner')`
    # marks a pixel True if it belongs to a region and has a neighbor
    # (in 8-conn) with a different label. Zeroing these matches
    # `spect_masked(border_inds) = 0`.
    perimeter = find_boundaries(labels_on_2s, mode="inner", connectivity=2)

    # Paint region pixels, then zero perimeter pixels.
    masked = np.where(labels_on_2s > 0, spect_2s, 0.0)
    masked[perimeter] = 0.0
    return masked
