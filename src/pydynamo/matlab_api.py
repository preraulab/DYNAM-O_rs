"""MATLAB-facing shims over pydynamo's Rust-accelerated internals.

These functions are designed to be called from MATLAB via `py.*`:

    from MATLAB:
        out = py.pydynamo.matlab_api.extract_tfpeaks_rs(spect_norm, stimes, sfreqs, ...)

Outputs are dicts of numpy arrays / scalars so MATLAB can trivially
unpack via `struct(out)`. Label images are returned as int64 2-D arrays
that map directly to MATLAB `uint32` matrices.

Drop-in MATLAB replacements covered:
    - runSegmentedData  →  extract_tfpeaks_rs        (single pass)
    - maskSpectrogram   →  mask_spectrogram_rs       (perimeter-aware)
    - refinePeakFrequency → refine_peaks_rs          (Hann + spline argmax)

Everything else (computeSpectrogram, computeBaseline, filterStatsTable,
computePeakSOpower, etc.) stays native MATLAB; you swap only the
expensive bits.
"""
from __future__ import annotations

import numpy as np

from pydynamo.tfpeaks.extract import extract_tfpeaks
from pydynamo.tfpeaks.mask import mask_spectrogram as _mask_spect
from pydynamo.tfpeaks.refine import refine_peak_frequency


def stats_table_to_dict(df):
    """Flatten a pandas DataFrame `stats_table` into a dict of numpy arrays
    MATLAB can trivially unpack (via py.getattr on the dict). BoundingBox
    (column of 4-tuples) is converted to Nx4 float64."""
    import pandas as pd
    out = {}
    n = len(df)
    for col in df.columns:
        v = df[col]
        if col == "BoundingBox":
            if n == 0:
                out[col] = np.zeros((0, 4), dtype=np.float64)
            else:
                out[col] = np.asarray(
                    [list(t) for t in v], dtype=np.float64
                ).reshape(n, 4)
        else:
            try:
                out[col] = np.asarray(v.values, dtype=np.float64)
            except (TypeError, ValueError):
                # object-dtype column → list of strings
                out[col] = [str(x) for x in v]
    out["_n_rows"] = int(n)
    return out


def extract_tfpeaks_rs(
    spect,
    stimes,
    sfreqs,
    *,
    baseline=None,
    seg_time=30.0,
    downsample=(2, 2),
    merge_thresh=11.0,
    trim_vol=0.8,
    dur_min=0.5,
    dur_max=5.0,
    bw_min=1.0,
    bw_max=15.0,
):
    """MATLAB-facing wrapper over `extract_tfpeaks` — one pass of the
    double-watershed with all kernels (watershed, merge, trim, regionprops)
    in Rust.

    Parameters
    ----------
    spect : (F, T) float64
        Spectrogram (raw power).
    stimes, sfreqs : 1-D float64
        Time / frequency axes.
    baseline : (F, 1) or (F,) float64 or None
        If given, `spect` is divided by `baseline` before extraction. Pass
        `None` if `spect` is already baseline-normalized.
    All other kwargs mirror `extract_tfpeaks`.

    Returns
    -------
    dict with keys:
        PeakTime, PeakFrequency, Duration, Bandwidth, Height, Volume,
        SegmentNum, BoundingBox  -- each a 1-D numpy array (Volume, Bounding-
                                    Box are N×1 / N×4 respectively)
        labels    -- (F, T) int64 label image suitable for passing back
                     to mask_spectrogram_rs on a pass-2 spectrogram.
        n_peaks   -- scalar
    """
    spect = np.ascontiguousarray(np.asarray(spect, dtype=np.float64))
    stimes = np.ascontiguousarray(np.asarray(stimes, dtype=np.float64).ravel())
    sfreqs = np.ascontiguousarray(np.asarray(sfreqs, dtype=np.float64).ravel())

    if baseline is not None:
        baseline = np.ascontiguousarray(np.asarray(baseline, dtype=np.float64))
        if baseline.ndim == 1:
            baseline = baseline[:, None]
        spect = spect / baseline

    stats, labels = extract_tfpeaks(
        spect, stimes, sfreqs,
        seg_time=float(seg_time),
        return_labels=True,
        downsample=tuple(int(v) for v in downsample),
        merge_thresh=float(merge_thresh),
        trim_vol=float(trim_vol),
        dur_min=float(dur_min), dur_max=float(dur_max),
        bw_min=float(bw_min), bw_max=float(bw_max),
    )

    out = {"labels": np.ascontiguousarray(labels, dtype=np.int64),
           "n_peaks": int(len(stats))}
    for col in ("PeakTime", "PeakFrequency", "Duration", "Bandwidth",
                "Height", "Volume", "SegmentNum"):
        out[col] = np.ascontiguousarray(
            stats[col].to_numpy(), dtype=np.float64
        ) if col in stats.columns else np.zeros(0)
    # BoundingBox is a column of 4-tuples → (N, 4) array
    if "BoundingBox" in stats.columns and len(stats):
        bb = np.asarray(list(stats["BoundingBox"]), dtype=np.float64)
    else:
        bb = np.zeros((0, 4), dtype=np.float64)
    out["BoundingBox"] = np.ascontiguousarray(bb)
    return out


def mask_spectrogram_rs(spect_2s, stimes_2s, labels_1s, stimes_1s):
    """Rust mask (perimeter-aware). Mirrors MATLAB's maskSpectrogram.

    Returns a (F, T_2) float64 array where pass-2 pixels outside pass-1
    regions OR on the 1-pixel perimeter of each region are zeroed.
    """
    return _mask_spect(
        np.ascontiguousarray(np.asarray(spect_2s, dtype=np.float64)),
        np.ascontiguousarray(np.asarray(stimes_2s, dtype=np.float64).ravel()),
        np.ascontiguousarray(np.asarray(labels_1s, dtype=np.int64)),
        np.ascontiguousarray(np.asarray(stimes_1s, dtype=np.float64).ravel()),
    )


def refine_peaks_rs(
    peak_times, peak_frequencies, bounding_boxes,
    data, fs, t=None,
    *,
    freq_range=(0.0, 30.0),
    window_size=4.0,
    dsfreqs=0.05,
    refine_method="spline_interp",
    remove_edge_peaks=True,
    n_spline_pts=1000,
):
    """Rust Hann-refinement. MATLAB drop-in for refinePeakFrequency.

    Parameters
    ----------
    peak_times, peak_frequencies : 1-D
    bounding_boxes : (N, 4) float64 [time_tl, freq_tl, width_s, height_Hz]
    data, fs, t : EEG + sampling rate + time axis (t defaults to arange(n)/fs)
    """
    import pandas as pd
    n = int(len(peak_times))
    stats = pd.DataFrame({
        "PeakTime": np.asarray(peak_times, dtype=float).ravel(),
        "PeakFrequency": np.asarray(peak_frequencies, dtype=float).ravel(),
        "BoundingBox": [tuple(r) for r in np.asarray(bounding_boxes, dtype=float)],
    })
    refined = refine_peak_frequency(
        stats, np.asarray(data, dtype=np.float64).ravel(), float(fs), t=t,
        freq_range=tuple(freq_range),
        window_size=float(window_size),
        dsfreqs=float(dsfreqs),
        refine_method=str(refine_method),
        remove_edge_peaks=bool(remove_edge_peaks),
        n_spline_pts=int(n_spline_pts),
    )
    # refined drops NaN rows; return a bool keep_mask aligned to the input.
    keep = np.isin(np.arange(n), refined.index.to_numpy()) \
           if len(refined) else np.zeros(n, dtype=bool)
    return {
        "PeakFrequency": np.ascontiguousarray(
            refined["PeakFrequency"].to_numpy(), dtype=np.float64
        ) if len(refined) else np.zeros(0),
        "PeakTime": np.ascontiguousarray(
            refined["PeakTime"].to_numpy(), dtype=np.float64
        ) if len(refined) else np.zeros(0),
        "keep_mask": keep,
        "n_kept": int(len(refined)),
    }
