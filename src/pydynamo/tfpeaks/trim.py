"""Trim watershed regions to retain `vol_thresh` fraction of peak volume.

Port of toolbox/TFpeak_functions/trimWshedRegions.m. MATLAB MEX fast path
is replaced by scipy.ndimage operations.

Algorithm per region:
    1. Subtract `shift_val` (default min(data)) so values are ≥ 0
    2. Sort region pixels by value ascending
    3. Find cutoff j where cumsum(vals)/total >= (1 - vol_thresh)
    4. Keep pixels with value >= vals[j]
    5. Fill holes (scipy.ndimage.binary_fill_holes with 4-connectivity)
    6. Keep only the largest connected component (by sum of values)
    7. Extract 4-neighbor boundary
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

try:
    import dynamo_rs as _dynamo_rs
    _HAS_RUST_TRIM = True
except ImportError:
    _dynamo_rs = None
    _HAS_RUST_TRIM = False


def trim_regions(
    labels: np.ndarray,
    data: np.ndarray,
    vol_thresh: float = 0.8,
    conn: int = 8,
    shift_val: float | None = None,
    use_rust: bool | None = None,
    compute_borders: bool = False,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Trim every region in `labels` to retain `vol_thresh` of its volume.

    Returns
    -------
    trimmed_labels : (F, T) int array — same shape as labels, with pixels
                     outside the trimmed regions set to 0.
    borders        : dict mapping label → flat indices of 4-neighbor boundary pixels
    """
    labels = np.ascontiguousarray(np.asarray(labels, dtype=np.int64))
    data = np.ascontiguousarray(np.asarray(data, dtype=np.float64))
    assert labels.shape == data.shape
    H, W = labels.shape

    if shift_val is None:
        shift_val = float(data.min())

    # Rust fast path (~70x faster).
    if use_rust is None:
        use_rust = _HAS_RUST_TRIM
    if use_rust:
        if not _HAS_RUST_TRIM:
            raise RuntimeError("use_rust=True but dynamo_rs not installed")
        labels_i32 = np.ascontiguousarray(labels, dtype=np.int32)
        trimmed = _dynamo_rs.trim_regions(
            labels_i32, data, float(vol_thresh), float(shift_val)
        ).astype(np.int64, copy=False)
        # borders dict is expensive (O(R) Python loop + dilation per region);
        # only build if the caller asked. extract.py currently doesn't use
        # it — skip by default.
        borders: dict[int, np.ndarray] = {}
        if compute_borders:
            for lab in np.unique(trimmed):
                if lab == 0:
                    continue
                flat = np.flatnonzero(trimmed.ravel() == lab)
                if flat.size == 0:
                    continue
                borders[int(lab)] = _boundary_indices(trimmed.shape, flat)
        return trimmed, borders
    shifted = np.where(data > shift_val, data - shift_val, 0.0)

    out_labels = np.zeros_like(labels)
    borders: dict[int, np.ndarray] = {}

    structure = np.ones((3, 3), dtype=bool) if conn == 8 else ndi.generate_binary_structure(2, 1)

    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels > 0]
    flat_shift = shifted.ravel()

    for lab in unique_labels:
        # Gather pixel indices for this label
        pixel_idx = np.flatnonzero(labels.ravel() == lab)
        if pixel_idx.size == 0:
            continue
        vals = flat_shift[pixel_idx]
        vmax, vmin = vals.max(), vals.min()
        if vmax == vmin or pixel_idx.size == 1:
            # Constant region — keep as-is (MATLAB has a special path; we
            # simplify to keep the whole region).
            out_labels.ravel()[pixel_idx] = lab
            borders[int(lab)] = _boundary_indices(labels.shape, pixel_idx)
            continue

        # Sort ascending by value
        order = np.argsort(vals, kind="stable")
        vals_sorted = vals[order]
        total = float(vals_sorted.sum())
        if total <= 0:
            out_labels.ravel()[pixel_idx] = lab
            borders[int(lab)] = _boundary_indices(labels.shape, pixel_idx)
            continue

        cum = np.cumsum(vals_sorted)
        # Find first j where cum[j]/total >= (1 - vol_thresh)
        cutoff_idx = np.searchsorted(cum, (1.0 - vol_thresh) * total, side="left")
        if cutoff_idx >= vals_sorted.size:
            # degenerate — keep as-is
            out_labels.ravel()[pixel_idx] = lab
            borders[int(lab)] = _boundary_indices(labels.shape, pixel_idx)
            continue
        level = vals_sorted[cutoff_idx]

        # Build a binary mask in a bounding-box sub-image for efficiency.
        rows = pixel_idx // W
        cols = pixel_idx % W
        r0, r1 = max(rows.min() - 1, 0), min(rows.max() + 1, H - 1)
        c0, c1 = max(cols.min() - 1, 0), min(cols.max() + 1, W - 1)
        sub_shape = (r1 - r0 + 1, c1 - c0 + 1)
        sub_mask = np.zeros(sub_shape, dtype=bool)
        sub_vals = shifted[r0:r1 + 1, c0:c1 + 1]
        # Only pixels originally in this region AND ≥ level
        sub_label = labels[r0:r1 + 1, c0:c1 + 1] == lab
        sub_mask[sub_label & (sub_vals >= level)] = True

        # Fill holes (4-connectivity per MATLAB).
        four_conn = ndi.generate_binary_structure(2, 1)
        sub_mask = ndi.binary_fill_holes(sub_mask, structure=four_conn)

        # Connected components; pick the one with largest summed value.
        cc_labels, n_cc = ndi.label(sub_mask, structure=structure)
        if n_cc == 0:
            continue
        if n_cc > 1:
            vols = ndi.sum(sub_vals, cc_labels, index=np.arange(1, n_cc + 1))
            best = int(np.argmax(vols)) + 1
            sub_mask = cc_labels == best

        # Write back: pixel indices in full image
        sub_rows, sub_cols = np.nonzero(sub_mask)
        full_rows = sub_rows + r0
        full_cols = sub_cols + c0
        flat_idx = full_rows * W + full_cols
        out_labels.ravel()[flat_idx] = lab
        borders[int(lab)] = _boundary_indices(labels.shape, flat_idx)

    return out_labels, borders


def _boundary_indices(shape: tuple, flat_idx: np.ndarray) -> np.ndarray:
    """4-neighbor boundary of a region given its flat indices."""
    H, W = shape
    mask = np.zeros((H, W), dtype=bool)
    mask.ravel()[flat_idx] = True
    padded = np.zeros((H + 2, W + 2), dtype=bool)
    padded[1:-1, 1:-1] = mask
    # Boundary = mask pixels where any 4-neighbor is False (or outside)
    interior = (
        padded[:-2, 1:-1] & padded[2:, 1:-1] &
        padded[1:-1, :-2] & padded[1:-1, 2:]
    )
    bnd_mask = mask & ~interior
    bi, bj = np.nonzero(bnd_mask)
    return bi * W + bj
