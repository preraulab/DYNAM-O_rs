"""Per-segment TF-peak extraction — follows pyDYNAM-O's `detect_tfpeaks`.

Flow per segment:
    1. (optional) downsample by (freq_factor, time_factor)
    2. watershed on -spect with watershed_line=True
    3. build RAG via expand_labels(distance=5) + skimage.graph.RAG
    4. merge_segment: iterative max-weight edge merge until w < merge_thresh
    5. expand merged labels (fill the 0-line) + (optional) resize up
    6. filter regions by bandwidth/duration/prominence BEFORE trimming (so
       we don't pay for trimming peaks we'd throw away)
    7. per-region trim to `trim_volume` fraction — keep largest subregion
    8. regionprops stats table → pandas
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import chi2
from skimage import measure, morphology
from skimage.segmentation import expand_labels
from skimage.segmentation import watershed as _sk_watershed
from skimage.transform import resize

# Use MATLAB-matching watershed from dynamo_rs when available.
try:
    import dynamo_rs as _dynamo_rs
    _HAS_RUST_WS = True
except ImportError:
    _HAS_RUST_WS = False
    _dynamo_rs = None


def watershed(img_neg, connectivity=2, watershed_line=True):
    """Drop-in replacement matching the signature used in this module.

    Prefers dynamo_rs.matlab_watershed (pure-Rust port of MATLAB's IPT
    watershed with column-major indexing + FIFO-within-priority) for
    better agreement with MATLAB. Falls back to skimage when not installed.
    """
    if _HAS_RUST_WS:
        arr = np.ascontiguousarray(img_neg, dtype=np.float64)
        return _dynamo_rs.matlab_watershed(arr).astype(np.int64)
    return _sk_watershed(img_neg, connectivity=connectivity, watershed_line=watershed_line)

from pydynamo.tfpeaks.merge import merge_segment


def _pow2db(y: np.ndarray | float) -> np.ndarray | float:
    """Power → dB, 0 → NaN (pyDYNAM-O compatibility)."""
    if np.isscalar(y):
        if y <= 0:
            return np.nan
        return 10 * np.log10(y)
    arr = np.asarray(y, dtype=float)
    out = np.where(arr > 0, 10 * np.log10(np.where(arr > 0, arr, 1.0)), np.nan)
    return out


def min_prominence(num_tapers: int, alpha: float = 0.95) -> float:
    """Chi-squared-based minimum peak prominence (dB). Matches MATLAB
    `ht_db_min` and pyDYNAM-O's min_prominence."""
    chi2_df = 2 * num_tapers
    return -_pow2db(chi2_df / chi2.ppf(alpha / 2 + 0.5, chi2_df)) * 2


def _trim_region(
    labels_merged: np.ndarray,
    graph_data: np.ndarray,
    region_num: int,
    trim_volume: float,
) -> np.ndarray:
    """Port of pyDYNAM-O's trim_region. Returns a mask image with
    `region_num` at retained pixels, 0 elsewhere."""
    reg_idx = np.where(labels_merged == region_num)
    reg_vals = np.sort(graph_data[reg_idx])
    total = float(reg_vals.sum())
    if total <= 0 or reg_vals.size == 0:
        return np.zeros_like(graph_data)

    percent_vol = np.cumsum(reg_vals) / total
    # First index where cumulative fraction exceeds (1 - trim_volume)
    above = np.flatnonzero(percent_vol > (1.0 - trim_volume))
    if above.size == 0:
        return np.zeros_like(graph_data)
    trim_idx = int(above[0])
    trim_level = reg_vals[trim_idx]

    rplot = np.zeros(graph_data.shape, dtype=bool)
    rplot[reg_idx] = True

    img = rplot & (graph_data > trim_level)
    filled = morphology.remove_small_holes(img)
    label_img = measure.label(filled, connectivity=2)

    if label_img.max() > 1:
        props = measure.regionprops(label_img)
        max_area_label = props[int(np.argmax([p.area for p in props]))].label
        label_img = label_img == max_area_label

    return (label_img > 0).astype(np.int64) * region_num


def extract_tfpeaks_segment(
    spect: np.ndarray,                 # (F, T) baseline-subtracted segment
    stimes: np.ndarray,
    sfreqs: np.ndarray,
    segment_num: int = 1,
    downsample: Tuple[int, int] | None = (2, 2),
    merge_thresh: float = 8.0,
    max_merges: float = float("inf"),
    trim_vol: float = 0.8,
    dur_min: float = 0.5,              # default to half a window (~1s/2)
    dur_max: float = 5.0,
    bw_min: float = 2.0,               # ~ 2 * (TW / windowlen)
    bw_max: float = 15.0,
    prom_min: float | None = None,     # chi-squared-based if None
    num_tapers_for_prom: int = 3,
    return_labels: bool = False,
) -> pd.DataFrame | Tuple[pd.DataFrame, np.ndarray]:
    """Extract TF-peaks from one spectrogram segment.

    When `return_labels=True`, also returns the trimmed (F, T) label image,
    useful as a pass-1 mask for the second watershed pass.
    """
    spect = np.ascontiguousarray(np.asarray(spect, dtype=np.float64))
    F, T = spect.shape
    d_time = float(stimes[1] - stimes[0])
    d_freq = float(sfreqs[1] - sfreqs[0])

    if prom_min is None:
        prom_min = min_prominence(num_tapers_for_prom, 0.95)

    # 1) Downsample (pyDYNAM-O: simple stride slicing)
    if downsample is not None and (downsample[0] > 1 or downsample[1] > 1):
        f_f, t_f = int(downsample[0]), int(downsample[1])
        seg_LR = spect[::f_f, ::t_f]
    else:
        f_f = t_f = 1
        seg_LR = spect

    # 2) Watershed on negated spectrogram (+ watershed_line)
    labels = watershed(-seg_LR, connectivity=2, watershed_line=True)

    # 3 + 4) Build RAG and merge
    labels_merged = merge_segment(
        labels, seg_LR,
        merge_thresh=merge_thresh, max_merges=max_merges,
    )

    # 5) Fill the remaining 0-line (former borders) with surrounding region ids.
    # MATLAB's extractTFPeaks.m:272 paints interior+border into Ldata
    # (`Ldata(ii_pixels)=ii` where ii_pixels = rgn{ii} = union(interior,
    # border) via Ldata2graph.m:233). expand_labels is an approximation of
    # that painting: it fills 0-line pixels with the nearest region label.
    # A fully MATLAB-exact border paint (painting each region's claimed
    # border pixels in label order) produced a cleaner per-segment match
    # but regressed the pipeline pass-2 count by -22% — likely because the
    # border Vec after symdiff omits pixels that MATLAB's full-union
    # approach keeps. Stick with expand_labels for now.
    labels_merged = expand_labels(labels_merged, distance=5).astype(np.int64)

    # Resize merged labels back up to the full-segment shape if we downsampled.
    if downsample is not None and (f_f > 1 or t_f > 1):
        # order=0 to preserve label values
        labels_full = resize(
            labels_merged, spect.shape, order=0,
            preserve_range=True, anti_aliasing=False,
        ).astype(np.int64)
    else:
        labels_full = labels_merged

    # 6a) Pre-trim dur/bw filter (MATLAB extractTFPeaks.m:300-307). MATLAB
    # drops ~57% of merged regions here before trim runs, using
    # (max_idx - min_idx)*dt > dur_min — i.e. (N_pixels - 1)*dt.
    # Dropping small regions before trim both changes the final peak count
    # (because trim can create spurious sub-regions from jagged bboxes)
    # AND skips trim's per-region work on regions we'd reject anyway.
    if dur_min > 0 or bw_min > 0:
        pre_props = measure.regionprops(labels_full)
        drop_labels = []
        for p in pre_props:
            minr, minc, maxr, maxc = p.bbox
            pre_dur = (maxc - minc - 1) * d_time
            pre_bw = (maxr - minr - 1) * d_freq
            if not (pre_dur > dur_min and pre_bw > bw_min):
                drop_labels.append(p.label)
        if drop_labels:
            drop_mask = np.isin(labels_full, drop_labels)
            labels_full = np.where(drop_mask, 0, labels_full).astype(np.int64)

    # 6b) Trim all regions in one Rust call (was a per-region Python loop that
    # dominated wallclock — ~800ms/segment vs ~12ms in Rust). Post-filter
    # by duration/bandwidth/prominence happens after stats computation below.
    from pydynamo.tfpeaks.trim import trim_regions as _trim_all
    trim_labels, _ = _trim_all(
        labels_full.astype(np.int64, copy=False),
        spect, vol_thresh=trim_vol, shift_val=None,
    )
    # Cast to int64 so downstream regionprops_table / indexing stays consistent.
    trim_labels = trim_labels.astype(np.int64, copy=False)

    if trim_labels.max() == 0:
        empty = pd.DataFrame()
        if return_labels:
            return empty, trim_labels
        return empty

    # 7) Stats table
    props = pd.DataFrame(
        measure.regionprops_table(
            trim_labels, spect,
            properties=("label", "centroid_weighted", "bbox",
                        "intensity_min", "intensity_max"),
        )
    )
    if props.empty:
        return props

    # Prominence in dB: MATLAB / pyDYNAM-O compute pow2db(max - min) on
    # the baseline-subtracted (linear) spect; reject peaks below prom_min.
    # This was the pre-trim filter; now enforced as a post-stats gate.
    raw_prom_linear = props["intensity_max"] - props["intensity_min"]
    props["Prominence_dB"] = _pow2db(raw_prom_linear.to_numpy())
    minr = props["bbox-0"]
    minc = props["bbox-1"]
    maxr = props["bbox-2"]
    maxc = props["bbox-3"]
    # Reported stats match MATLAB's computePeakStatsTable: bb-width * dx.
    # skimage bbox is (min, max+1), so (maxc - minc) already = N_pixels.
    props["Duration"] = (maxc - minc) * d_time
    props["Bandwidth"] = (maxr - minr) * d_freq

    # Bounding box in (time_tl, freq_tl, width_s, height_Hz)
    props["BoundingBox"] = [
        (float(c) * d_time + float(stimes[0]),
         float(r) * d_freq + float(sfreqs[0]),
         float(mc - c) * d_time,
         float(mr - r) * d_freq)
        for r, c, mr, mc in zip(minr, minc, maxr, maxc)
    ]

    props["PeakTime"] = props["centroid_weighted-1"] * d_time + float(stimes[0])
    props["PeakFrequency"] = props["centroid_weighted-0"] * d_freq + float(sfreqs[0])
    props["Height"] = props["intensity_max"] - props["intensity_min"]
    props["SegmentNum"] = int(segment_num)

    # Volume: sum(spect at label pixels) * dt * df
    def _vol(lab):
        return float(spect[trim_labels == lab].sum()) * d_time * d_freq
    props["Volume"] = [_vol(int(lab)) for lab in props["label"]]

    # MATLAB extractTFPeaks.m:304/336/355 filter uses
    #   (max(x) - min(x)) * dt > dur_min
    # where (max-min) = N_pixels - 1. Reported Duration = N_pixels * dt.
    # So the filter value is (Duration - d_time). Same for Bandwidth.
    # filterStatsTable.m:78-81 uses Duration > dur_min (less strict by 1
    # pixel), but the internal extractTFPeaks filter is the governing one.
    prom_ok = np.isnan(props["Prominence_dB"].to_numpy()) | \
              (props["Prominence_dB"].to_numpy() > prom_min)
    filter_dur = props["Duration"].to_numpy() - d_time
    filter_bw = props["Bandwidth"].to_numpy() - d_freq
    props = props[
        (filter_dur > dur_min) & (props["Duration"] < dur_max) &
        (filter_bw > bw_min) & (props["Bandwidth"] < bw_max) &
        prom_ok
    ].reset_index(drop=True)

    # Drop regionprops raw columns we no longer need
    for c in ["centroid_weighted-0", "centroid_weighted-1",
              "bbox-0", "bbox-1", "bbox-2", "bbox-3",
              "intensity_min", "intensity_max", "Prominence_dB"]:
        if c in props.columns:
            del props[c]
    if return_labels:
        return props, trim_labels
    return props


def extract_tfpeaks(
    spect: np.ndarray,
    stimes: np.ndarray,
    sfreqs: np.ndarray,
    seg_time: float = 30.0,
    n_jobs: int = -1,
    return_labels: bool = False,
    **segment_kwargs,
) -> pd.DataFrame | Tuple[pd.DataFrame, np.ndarray]:
    """Split the spectrogram into `seg_time`-second segments and extract
    TF-peaks per segment, concatenating results. Uses joblib for parallelism.

    When `return_labels=True`, also returns a stitched (F, T) label image
    covering the full spectrogram (each peak carries a unique int label).
    """
    dt = float(stimes[1] - stimes[0])
    F, T = spect.shape

    # Match MATLAB segmentData.m:85-107 exactly: ceil-divide to get n_segs,
    # then ceil-divide again to get even-sized segments (last one may be
    # truncated). Stride-N with start+N misaligns every segment after the
    # first by 1 sample vs MATLAB, so per-segment watershed + merge produce
    # slightly different regions.
    import math
    max_dx = int(math.floor(seg_time / dt))
    n_segs = int(math.ceil(T / max_dx))
    new_dx = int(math.ceil(T / n_segs))
    segments = []
    for ii in range(n_segs):
        start = ii * new_dx
        end = min(start + new_dx, T)
        if end - start < 2:
            continue
        segments.append((ii + 1, start, end))

    def _run(si, start, end):
        return extract_tfpeaks_segment(
            spect[:, start:end], stimes[start:end], sfreqs,
            segment_num=si, return_labels=return_labels, **segment_kwargs,
        )

    if n_jobs == 1 or len(segments) <= 1:
        results = [_run(*s) for s in segments]
    else:
        results = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(_run)(*s) for s in segments
        )

    if not results:
        empty = pd.DataFrame()
        return (empty, np.zeros((F, T), dtype=np.int64)) if return_labels else empty

    if return_labels:
        full_labels = np.zeros((F, T), dtype=np.int64)
        tables = []
        offset = 0
        for (si, start, end), (tbl, seg_labels) in zip(segments, results):
            if not tbl.empty:
                tbl = tbl.copy()
                # Keep only labels that passed the post-stats filter. The raw
                # seg_labels image contains every label output by trim, so if
                # we paint it as-is we'd include peaks that failed the dur/
                # bw/height filter — inflating the downstream mask area.
                kept_labels = set(int(x) for x in tbl["label"].to_numpy())
                filtered_seg = np.where(
                    np.isin(seg_labels, list(kept_labels)), seg_labels, 0
                )
                tbl["label"] = tbl["label"] + offset
                shifted = np.where(filtered_seg > 0, filtered_seg + offset, 0)
                full_labels[:, start:end] = shifted
                offset += int(seg_labels.max())
                tables.append(tbl)
        if not tables:
            return pd.DataFrame(), full_labels
        return pd.concat(tables, ignore_index=True), full_labels

    tables = [t for t in results if not t.empty]
    if not tables:
        return pd.DataFrame()
    return pd.concat(tables, ignore_index=True)
