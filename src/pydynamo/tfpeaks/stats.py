"""Compute per-region stats (Area, BoundingBox, WeightedCentroid, etc.)
and pack into a pandas DataFrame — port of computePeakStatsTable.m.

MATLAB uses regionprops + pandas-like table. We use scikit-image's
regionprops_table with intensity image.

NOTE: DYNAM-O's MATLAB data is (F, T) with rows = freq, cols = time.
- BoundingBox(:, 1) = top-left time  → stimes axis (cols, aka x)
- BoundingBox(:, 2) = top-left freq  → sfreqs axis (rows, aka y)
- WeightedCentroid(:, 1) = time centroid (cols)
- WeightedCentroid(:, 2) = freq centroid (rows)

scikit-image's regionprops returns centroids as (row, col); remember to
swap when mapping to (time, freq).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from skimage.measure import regionprops


def compute_peak_stats(
    labels: np.ndarray,
    data: np.ndarray,
    stimes: np.ndarray,
    sfreqs: np.ndarray,
    segment_num: int = 1,
) -> pd.DataFrame:
    """Build a stats DataFrame with one row per labelled region.

    Columns (subset of DYNAM-O):
        PeakTime, PeakFrequency, Height, Area, Duration, Bandwidth, Volume,
        BoundingBox (4-tuple: (time_tl, freq_tl, width_s, height_Hz)),
        SegmentNum
    """
    labels = np.asarray(labels, dtype=np.int64)
    data = np.asarray(data, dtype=np.float64)
    stimes = np.asarray(stimes, dtype=np.float64)
    sfreqs = np.asarray(sfreqs, dtype=np.float64)
    assert labels.shape == data.shape

    dt = float(stimes[1] - stimes[0])
    df = float(sfreqs[1] - sfreqs[0])
    seg_startx = float(stimes[0])
    seg_starty = float(sfreqs[0])

    # skimage regionprops with intensity image. Relabel to contiguous 1..K
    # so regionprops doesn't silently skip labels.
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels > 0]
    if unique_labels.size == 0:
        return pd.DataFrame(
            columns=["PeakTime", "PeakFrequency", "Height", "Area",
                     "Duration", "Bandwidth", "Volume", "BoundingBox",
                     "SegmentNum"]
        )

    # Treat NaN pixels as out-of-region. MATLAB does this explicitly.
    data_clean = np.where(np.isnan(data), 0.0, data)
    safe_labels = labels.copy()
    safe_labels[np.isnan(data)] = 0

    props = regionprops(safe_labels, intensity_image=data_clean)

    rows = []
    for p in props:
        # BoundingBox (skimage): (min_row, min_col, max_row, max_col)
        r0, c0, r1, c1 = p.bbox
        bb_time = (c0) * dt + seg_startx
        bb_freq = (r0) * df + seg_starty
        bb_w = (c1 - c0) * dt
        bb_h = (r1 - r0) * df

        # WeightedCentroid (skimage): (row, col) in pixel coords, zero-based.
        # MATLAB's WeightedCentroid is 1-based so subtracting 0.5 aligns; we
        # use 0-based but offset by the segment start identically.
        wc_r, wc_c = p.weighted_centroid
        peak_time = wc_c * dt + seg_startx
        peak_freq = wc_r * df + seg_starty

        vals = p.image_intensity[p.image]
        area = p.area * dt * df
        volume = float(vals.sum()) * dt * df
        height = float(vals.max() - vals.min())

        rows.append({
            "Label": p.label,
            "PeakTime": peak_time,
            "PeakFrequency": peak_freq,
            "Height": height,
            "Area": area,
            "Duration": bb_w,
            "Bandwidth": bb_h,
            "Volume": volume,
            "BoundingBox": (bb_time, bb_freq, bb_w, bb_h),
            "SegmentNum": int(segment_num),
        })

    return pd.DataFrame(rows)
