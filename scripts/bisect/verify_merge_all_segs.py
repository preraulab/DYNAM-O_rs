"""Measure merge exactness across ALL segments.

For each segment in the segment dataset, run pydynamo's watershed→merge
and compare region count to MATLAB's (from merge_diagnostics). If
watershed+merge is bit-exact across all 168 segments, the full-pipeline
17% divergence is elsewhere. If several segments differ, localize which.
"""
from __future__ import annotations
import sys
from pathlib import Path

import h5py
import numpy as np

import dynamo_rs
from pydynamo.tfpeaks.extract import watershed

DC = Path(__file__).parent.parent.parent / "data_cache"


def _sq(x):
    a = np.asarray(x)
    if a.ndim >= 2:
        a = a.T
    return np.squeeze(a)


def main() -> int:
    # Load MATLAB per-segment counts
    with h5py.File(DC / "merge_diagnostics_segment.mat", "r") as f:
        mat_postmerge = np.asarray(
            f["seg_counts/n_regions_postmerge"][...]
        ).ravel().astype(int)
        mat_postwshed = np.asarray(
            f["seg_counts/n_regions_postwshed"][...]
        ).ravel().astype(int)

    # Load spect2_masked (same input MATLAB used for pass-2)
    with h5py.File(DC / "bisect_intermediates_segment.mat", "r") as f:
        spect2_masked = _sq(f["spect2_masked"][...]).astype(np.float64)
        stimes2 = _sq(f["stimes2"][...]).ravel()

    dt = float(stimes2[1] - stimes2[0])
    import math
    max_dx = int(math.floor(30.0 / dt))
    T = spect2_masked.shape[1]
    n_segs = int(math.ceil(T / max_dx))
    new_dx = int(math.ceil(T / n_segs))

    diffs_wshed = []
    diffs_merge = []
    for ii in range(n_segs):
        start = ii * new_dx
        end = min(start + new_dx, T)
        if end - start < 2:
            diffs_wshed.append(0)
            diffs_merge.append(0)
            continue
        spect = spect2_masked[:, start:end]
        if np.all(spect == 0) or np.all(np.isnan(spect)):
            diffs_wshed.append(0)
            diffs_merge.append(0)
            continue
        seg_LR = spect[::2, ::2]
        labels = watershed(-seg_LR, connectivity=2, watershed_line=True)
        py_wshed = int(labels.max())
        merged = dynamo_rs.merge_segment(
            np.ascontiguousarray(labels, dtype=np.int64),
            np.ascontiguousarray(seg_LR, dtype=np.float64),
            11.0, float("inf"),
        )
        py_merge = int(np.unique(merged[merged > 0]).size)
        diffs_wshed.append(py_wshed - mat_postwshed[ii])
        diffs_merge.append(py_merge - mat_postmerge[ii])

    diffs_wshed = np.array(diffs_wshed)
    diffs_merge = np.array(diffs_merge)
    print(f"watershed diff (py - mat) per segment:")
    print(f"  all zero: {(diffs_wshed == 0).sum()}/{n_segs} segments bit-exact on watershed")
    print(f"  total diff: {diffs_wshed.sum():+d}")
    if (diffs_wshed != 0).any():
        for i, d in enumerate(diffs_wshed):
            if d != 0:
                print(f"    segment {i+1}: diff={d:+d}  (py={diffs_wshed[i]+mat_postwshed[i]}, mat={mat_postwshed[i]})")

    print(f"\nmerge diff (py - mat) per segment:")
    print(f"  all zero: {(diffs_merge == 0).sum()}/{n_segs} segments bit-exact on count")
    print(f"  total diff: {diffs_merge.sum():+d}  (sum py: {diffs_merge.sum()+mat_postmerge.sum()}, sum mat: {mat_postmerge.sum()})")
    print(f"  median per-seg |diff|: {int(np.median(np.abs(diffs_merge)))}")
    print(f"  max per-seg |diff|: {int(np.max(np.abs(diffs_merge)))}")
    if (diffs_merge != 0).any():
        nz = np.flatnonzero(diffs_merge != 0)
        print(f"  {len(nz)} segments differ (first 10): "
              f"{[(int(i+1), int(diffs_merge[i])) for i in nz[:10]]}")

    return 0 if (diffs_merge == 0).all() else 2


if __name__ == "__main__":
    sys.exit(main())
