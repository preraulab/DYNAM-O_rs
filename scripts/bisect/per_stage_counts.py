"""Track pydynamo peak counts at every stage of the pipeline and compare
to MATLAB's stage counts from pass1/pass2_diagnostics mat files.

Stages tracked (per pass, summed across segments):
    post_watershed   — raw watershed region count (all basins)
    post_merge       — after iterative merge
    post_dur_bw_pre  — after pre-trim dur/bw filter
    post_trim        — after trim_regions (shrunk regions)
    post_filter      — after final dur/bw/prominence filter (post-trim)

Plus whole-pipeline: after_refine (final stats_table count).

Writes side-by-side py/mat per stage.
"""
from __future__ import annotations
import math
import sys
from pathlib import Path

import h5py
import numpy as np

from pydynamo.io_compat import load_example_data
from pydynamo.spectrogram import mtm_spectrogram
from pydynamo.baseline import compute_baseline, subtract_baseline
from pydynamo.tfpeaks.extract import watershed, min_prominence
from pydynamo.tfpeaks.merge import merge_segment
from pydynamo.tfpeaks.trim import trim_regions as _trim_all
from pydynamo.tfpeaks.mask import mask_spectrogram
from pydynamo.tfpeaks.refine import refine_peak_frequency
from scipy.interpolate import interp1d
from skimage import measure
from skimage.segmentation import expand_labels
from skimage.transform import resize

DC = Path(__file__).parent.parent.parent / "data_cache"


def _sq(x):
    a = np.asarray(x)
    if a.ndim >= 2:
        a = a.T
    return np.squeeze(a)


def _count_unique_labels(labels: np.ndarray) -> int:
    return int(np.unique(labels[labels > 0]).size)


def walk_pass(
    spect: np.ndarray,
    stimes: np.ndarray,
    sfreqs: np.ndarray,
    *,
    merge_thresh: float,
    trim_vol: float,
    dur_min: float,
    bw_min: float,
    dur_max: float = 5.0,
    bw_max: float = 15.0,
    prom_min: float | None = None,
    downsample: tuple[int, int] = (2, 2),
    seg_time: float = 30.0,
    return_labels: bool = False,
) -> tuple[dict, np.ndarray | None]:
    """Replicates extract_tfpeaks but captures per-stage counts."""
    dt = float(stimes[1] - stimes[0])
    F, T = spect.shape
    max_dx = int(math.floor(seg_time / dt))
    n_segs = int(math.ceil(T / max_dx))
    new_dx = int(math.ceil(T / n_segs))

    if prom_min is None:
        prom_min = min_prominence(3, 0.95)

    counts = {
        "post_watershed": 0,
        "post_merge": 0,
        "post_dur_bw_pre": 0,
        "post_trim": 0,
        "post_filter": 0,
    }
    rejected = {
        "dropped_dur_bw_pretrim": 0,
        "dropped_by_trim": 0,
        "dropped_post_filter_duration": 0,
        "dropped_post_filter_bandwidth": 0,
        "dropped_post_filter_prominence": 0,
    }

    full_labels = np.zeros((F, T), dtype=np.int64) if return_labels else None
    label_offset = 0

    for ii in range(n_segs):
        start = ii * new_dx
        end = min(start + new_dx, T)
        if end - start < 2:
            continue
        seg = spect[:, start:end]
        x = stimes[start:end]
        d_time = float(x[1] - x[0])
        d_freq = float(sfreqs[1] - sfreqs[0])
        if np.all(seg == 0) or np.all(np.isnan(seg)):
            continue
        seg_LR = seg[::downsample[0], ::downsample[1]]

        labels = watershed(-seg_LR, connectivity=2, watershed_line=True)
        counts["post_watershed"] += int(labels.max())

        merged = merge_segment(labels, seg_LR, merge_thresh=merge_thresh,
                               max_merges=float("inf"))
        counts["post_merge"] += _count_unique_labels(merged)

        merged_exp = expand_labels(merged, distance=5).astype(np.int64)
        labels_full = resize(merged_exp, seg.shape, order=0,
                              preserve_range=True, anti_aliasing=False).astype(np.int64)

        # Pre-trim dur/bw filter (MATLAB-style, (N-1)*dt)
        props_pre = measure.regionprops(labels_full)
        drop = []
        for p in props_pre:
            minr, minc, maxr, maxc = p.bbox
            pre_dur = (maxc - minc - 1) * d_time
            pre_bw = (maxr - minr - 1) * d_freq
            if not (pre_dur > dur_min and pre_bw > bw_min):
                drop.append(p.label)
        n_pre = len(props_pre)
        if drop:
            labels_full = np.where(np.isin(labels_full, drop), 0, labels_full).astype(np.int64)
        counts["post_dur_bw_pre"] += _count_unique_labels(labels_full)
        rejected["dropped_dur_bw_pretrim"] += len(drop)

        # Trim
        trim_labels, _ = _trim_all(labels_full, seg, vol_thresh=trim_vol, shift_val=None)
        trim_labels = trim_labels.astype(np.int64)
        n_trim = _count_unique_labels(trim_labels)
        counts["post_trim"] += n_trim
        rejected["dropped_by_trim"] += max(_count_unique_labels(labels_full) - n_trim, 0)

        # Post-trim filter (dur / bw / prominence), matching MATLAB strict >
        if n_trim == 0:
            continue
        props = measure.regionprops(trim_labels, seg)
        kept_labels = []
        for p in props:
            minr, minc, maxr, maxc = p.bbox
            dur = (maxc - minc) * d_time
            bw = (maxr - minr) * d_freq
            if not ((dur - d_time) > dur_min):
                rejected["dropped_post_filter_duration"] += 1
                continue
            if dur >= dur_max:
                rejected["dropped_post_filter_duration"] += 1
                continue
            if not ((bw - d_freq) > bw_min):
                rejected["dropped_post_filter_bandwidth"] += 1
                continue
            if bw >= bw_max:
                rejected["dropped_post_filter_bandwidth"] += 1
                continue
            prom = 10.0 * np.log10(max(p.intensity_max - p.intensity_min, 1e-300))
            if not (np.isnan(prom) or prom > prom_min):
                rejected["dropped_post_filter_prominence"] += 1
                continue
            kept_labels.append(p.label)
        counts["post_filter"] += len(kept_labels)

        if return_labels and kept_labels:
            mask_keep = np.isin(trim_labels, kept_labels)
            kept = np.where(mask_keep, trim_labels, 0).astype(np.int64)
            shifted = np.where(kept > 0, kept + label_offset, 0)
            full_labels[:, start:end] = shifted
            label_offset += int(kept.max())

    return counts, rejected, full_labels


def main() -> int:
    # ---- Load artifacts (as pipeline does) ----
    with h5py.File(DC / "bisect_intermediates_segment.mat", "r") as f:
        mat_art = _sq(f["artifacts"][...]).astype(bool)
    # ---- Load MATLAB pass-1 and pass-2 counts ----
    with h5py.File(DC / "pass1_diagnostics_segment.mat", "r") as f:
        p1_mat = {k: int(np.asarray(f[f"seg_counts/n_regions_{k}"][...]).ravel().sum())
                  for k in ["postwshed", "postmerge", "postdurbw", "posttrim", "postfilter"]}
    with h5py.File(DC / "merge_diagnostics_segment.mat", "r") as f:
        p2_mat = {k: int(np.asarray(f[f"seg_counts/n_regions_{k}"][...]).ravel().sum())
                  for k in ["postwshed", "postmerge", "postdurbw", "posttrim", "postfilter"]}

    # ---- pydynamo pass-1 ----
    ed = load_example_data()
    fs = float(ed["Fs"])
    i0 = int(round(8420 * fs)); i1 = int(round(13446 * fs))
    data_tr = ed["data"].ravel()[i0:i1+1].astype(np.float64)
    t_tr = np.arange(i0, i1+1) / fs
    stage_at_data = interp1d(ed["stage_times"].ravel(), ed["stage_vals"].ravel(),
                             kind="previous", bounds_error=False, fill_value=0.0)(t_tr)
    baseline_exclude = mat_art | ~np.isin(stage_at_data, (1, 2, 3, 4, 5))

    spect1, stimes1_rel, sfreqs = mtm_spectrogram(
        data_tr, fs, freq_range=(0, 30), taper_params=(2, 3),
        window_params=(1.0, 0.05), dsfreqs=0.1,
    )
    stimes1 = stimes1_rel + t_tr[0]
    baseline1 = compute_baseline(spect1, stimes1, t_tr, baseline_exclude, baseline_ptile=2.0)
    spect1_norm = subtract_baseline(spect1, baseline1)

    print("=== PASS-1 ===")
    p1_py, p1_rej, labels1 = walk_pass(
        spect1_norm, stimes1, sfreqs,
        merge_thresh=11.0, trim_vol=0.8,
        dur_min=0.5, bw_min=2.0, return_labels=True,
    )
    print(f"{'stage':24s}  {'py':>8s}  {'mat':>8s}  {'diff':>8s}  {'diff %':>7s}")
    for k in ["post_watershed", "post_merge", "post_dur_bw_pre", "post_trim", "post_filter"]:
        mk = {"post_watershed": "postwshed", "post_merge": "postmerge",
              "post_dur_bw_pre": "postdurbw", "post_trim": "posttrim",
              "post_filter": "postfilter"}[k]
        py = p1_py[k]; mat = p1_mat[mk]
        pct = 100 * (py - mat) / max(mat, 1)
        print(f"  {k:22s}  {py:>8d}  {mat:>8d}  {py-mat:>+8d}  {pct:>+6.1f}%")
    print(f"\n  rejection details (py):")
    for k, v in p1_rej.items():
        print(f"    {k:38s} {v:>8d}")

    # ---- pydynamo pass-2 ----
    spect2, stimes2_rel, sfreqs2 = mtm_spectrogram(
        data_tr, fs, freq_range=(0, 30), taper_params=(2, 3),
        window_params=(2.0, 0.05), dsfreqs=0.1,
    )
    stimes2 = stimes2_rel + t_tr[0]
    baseline2 = compute_baseline(spect2, stimes2, t_tr, baseline_exclude, baseline_ptile=2.0)
    spect2_norm = subtract_baseline(spect2, baseline2)
    spect2_masked = mask_spectrogram(spect2_norm, stimes2, labels1, stimes1)

    print("\n=== PASS-2 ===")
    p2_py, p2_rej, labels2 = walk_pass(
        spect2_masked, stimes2, sfreqs2,
        merge_thresh=11.0, trim_vol=0.8,
        dur_min=0.5, bw_min=1.0, return_labels=True,
    )
    for k in ["post_watershed", "post_merge", "post_dur_bw_pre", "post_trim", "post_filter"]:
        mk = {"post_watershed": "postwshed", "post_merge": "postmerge",
              "post_dur_bw_pre": "postdurbw", "post_trim": "posttrim",
              "post_filter": "postfilter"}[k]
        py = p2_py[k]; mat = p2_mat[mk]
        pct = 100 * (py - mat) / max(mat, 1)
        print(f"  {k:22s}  {py:>8d}  {mat:>8d}  {py-mat:>+8d}  {pct:>+6.1f}%")
    print(f"\n  rejection details (py):")
    for k, v in p2_rej.items():
        print(f"    {k:38s} {v:>8d}")

    # MATLAB doesn't save pass-2 dur_min=0.5 specifically (their export used
    # 1.0). Comparison for pass-2 is apples-to-oranges for that reason.
    print(f"\n  NOTE: MATLAB merge_diagnostics_segment was exported with dur_min=1.0,\n"
          f"  pydynamo uses dur_min=0.5 (MATLAB's actual pipeline value). Direct compare\n"
          f"  of post-filter is not apples-to-apples at pass-2. post_merge is.")

    # ---- End-to-end: run_dynamo final ----
    from pydynamo import run_dynamo
    out = run_dynamo(
        ed["data"].ravel(), fs, ed["stage_times"].ravel(), ed["stage_vals"].ravel(),
        time_range=(8420.0, 13446.0), merge_thresh=11.0, trim_vol=0.8, seg_time=30.0,
        min_time_in_bin=5.0, double_watershed=True, refinement=True, plot=False, verbose=False,
    )
    import pandas as pd
    mat_final = pd.read_csv(DC / "segment_stats.csv")

    print("\n=== END-TO-END ===")
    print(f"  py pass-2 post_filter + before refine  : ~{p2_py['post_filter']}")
    print(f"  py final (after refinement)            : {len(out.stats_table)}")
    print(f"  mat final                               : {len(mat_final)}")
    print(f"  diff                                    : {len(out.stats_table)-len(mat_final):+d} "
          f"({100*(len(out.stats_table)/len(mat_final)-1):+.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
