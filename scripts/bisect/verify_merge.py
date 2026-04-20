"""Merge-replay test harness.

Loads segment 40's rep_dump from merge_diagnostics_segment.mat (MATLAB's
exact pre-merge input state), runs pydynamo's merge, and compares the
output region-by-region to MATLAB's post-merge state.

Passes if pydynamo produces the same labeled image as MATLAB.
"""
from __future__ import annotations
import sys
from pathlib import Path

import h5py
import numpy as np

# dynamo_rs is the Rust kernel we want to validate
import dynamo_rs


DC = Path(__file__).parent.parent.parent / "data_cache"


def _sq(x):
    a = np.asarray(x)
    if a.ndim >= 2:
        a = a.T
    return np.squeeze(a)


def _deref(f, ref_array):
    """Dereference an HDF5 cell-of-references into a list of arrays
    of Python int64 (linear indices, 1-based column-major from MATLAB)."""
    out = []
    for r in ref_array.ravel():
        obj = f[r]
        a = np.asarray(obj[...])
        if a.ndim >= 2:
            a = a.T
        out.append(np.squeeze(a).astype(np.int64))
    return out


def matlab_to_rc(inds_1based: np.ndarray, F: int) -> tuple[np.ndarray, np.ndarray]:
    """MATLAB column-major 1-based linear index → (row, col) 0-based Python."""
    i0 = inds_1based.astype(np.int64) - 1
    c = i0 // F
    r = i0 % F
    return r, c


def build_label_image_from_regions(
    regions: list[np.ndarray],
    shape: tuple[int, int],
    labels_for_each: list[int],
    one_based: bool = True,
) -> np.ndarray:
    """Reconstruct a label image from MATLAB's region cell array."""
    F, T = shape
    out = np.zeros(shape, dtype=np.int64)
    for lbl, inds in zip(labels_for_each, regions):
        if inds.size == 0:
            continue
        if one_based:
            r, c = matlab_to_rc(inds, F)
        else:
            r = inds // T
            c = inds % T
        out[r, c] = lbl
    return out


def main() -> int:
    mat_path = DC / "merge_diagnostics_segment.mat"
    if not mat_path.exists():
        print(f"missing {mat_path}; run export_merge_diagnostics.m first")
        return 1

    print(f"=== loading rep_dump from {mat_path.name} ===")
    with h5py.File(mat_path, "r") as f:
        rd = f["rep_dump"]
        img_LR = _sq(rd["img_LR"][...]).astype(np.float64)
        Ldata_mat = _sq(rd["Ldata_wshed"][...]).astype(np.int64)
        regions_pre = _deref(f, rd["regions_pre"][...])
        borders_pre = _deref(f, rd["borders_pre"][...])
        adj_list = _sq(rd["adj_list_pre"][...]).astype(np.int64)
        region_lbls = _sq(rd["region_lbls_pre"][...]).astype(np.int64)
        regions_postmerge = _deref(f, rd["regions_postmerge_LR"][...])
        borders_postmerge = _deref(f, rd["borders_postmerge_LR"][...])
        seg_num = int(_sq(rd["seg_num"][...]))

    H, W = img_LR.shape
    print(f"image {img_LR.shape}, {len(regions_pre)} regions pre-merge, "
          f"{adj_list.shape[0]} adjacencies, "
          f"{sum(1 for r in regions_postmerge if r.size > 0)} regions post-merge")

    # ---- Reconstruct pre-merge label image from MATLAB regions ----
    # MATLAB rep_dump regions include border pixels (Ldata2graph line 233
    # unions them back). For the watershed label input we want 0 at border
    # pixels + label at interior. Ldata_mat is already that.
    #
    # For feeding our Rust merge, we need a label image. Ldata_mat IS the
    # watershed output with 0-line borders — that's the input to merge.
    print()
    print(f"Ldata_mat unique labels: {len(np.unique(Ldata_mat))} (max={Ldata_mat.max()})")

    # ---- Run current pydynamo merge on this labeled image ----
    py_merged = dynamo_rs.merge_segment(
        np.ascontiguousarray(Ldata_mat, dtype=np.int64),
        np.ascontiguousarray(img_LR, dtype=np.float64),
        11.0,  # merge_thresh
        float("inf"),  # max_merges
    )
    py_merged = py_merged.astype(np.int64)
    n_py = len(np.unique(py_merged[py_merged > 0]))
    n_mat = sum(1 for r in regions_postmerge if r.size > 0)
    print(f"\npost-merge: py={n_py} mat={n_mat}  "
          f"(diff {n_py - n_mat:+d})")

    # ---- Build MATLAB post-merge label image ----
    # The post-merge labels correspond to the LR image size. MATLAB's
    # regions_postmerge_LR cells have linear indices into Ldata_wshed.
    mat_lbls_postmerge = [
        int(region_lbls[i]) for i in range(len(regions_postmerge))
    ]
    mat_merged = build_label_image_from_regions(
        regions_postmerge, img_LR.shape, mat_lbls_postmerge, one_based=True
    )

    # ---- Pixel-level comparison ----
    # Labels are arbitrary IDs — what we care about is the PARTITION: do
    # pixels that share a MATLAB label also share a pydynamo label?
    # Use mutual-information / Rand-index-style check: pair each pixel with
    # its MATLAB label and its pydynamo label, count agreement on "same
    # region vs different region" between all pairs.
    # Simpler: build a remap from pydynamo label → most common MATLAB label
    # in its pixels, and vice versa, and measure agreement.
    both_labeled = (py_merged > 0) & (mat_merged > 0)
    nz = both_labeled.sum()
    print(f"\npixels labeled in both: {nz} / {mat_merged.size} "
          f"({nz / mat_merged.size * 100:.1f}%)")

    # Build confusion matrix: mat_label × py_label
    mat_u = np.unique(mat_merged[both_labeled])
    py_u = np.unique(py_merged[both_labeled])
    print(f"unique mat labels: {len(mat_u)}, unique py labels: {len(py_u)}")

    # For each mat label, find the py label that best overlaps; check if
    # that py label maps 1:1 (exclusive) to this mat label.
    from collections import Counter
    mat_pairs = list(zip(
        mat_merged[both_labeled].tolist(),
        py_merged[both_labeled].tolist(),
    ))
    by_mat: dict[int, Counter] = {}
    by_py: dict[int, Counter] = {}
    for m, p in mat_pairs:
        by_mat.setdefault(m, Counter())[p] += 1
        by_py.setdefault(p, Counter())[m] += 1

    # Exact matches: mat_label whose pixels are all in one py_label
    # and vice versa (1:1 bijection on the pixel level)
    perfect_mat = 0
    for m, c in by_mat.items():
        if len(c) == 1:
            (p_only,) = list(c.keys())
            if len(by_py[p_only]) == 1:
                perfect_mat += 1
    print(f"exact 1:1 mat↔py regions: {perfect_mat} / {len(mat_u)}")

    # Per-pixel partition agreement: map each mat_label to its dominant
    # py_label, then count pixels where py_merged equals that mapping.
    mat_to_py_mode = {int(m): int(c.most_common(1)[0][0]) for m, c in by_mat.items()}
    agreement_count = 0
    for m, c in by_mat.items():
        expected_py = mat_to_py_mode[int(m)]
        agreement_count += c.get(expected_py, 0)
    print(f"per-pixel partition agreement (inside both-labeled mask): "
          f"{agreement_count / nz * 100:.2f}%")

    print(f"\n=== SUMMARY ===")
    print(f"region count: py={n_py}, mat={n_mat}")
    print(f"exact 1:1 matches: {perfect_mat}/{len(mat_u)}")
    return 0 if perfect_mat == len(mat_u) else 2


if __name__ == "__main__":
    sys.exit(main())
