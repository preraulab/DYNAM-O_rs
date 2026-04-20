"""Region adjacency graph + iterative merge loop.

Follows pyDYNAM-O's implementation (dynam_o.TFpeaks.detect_tfpeaks) which
reproduces MATLAB DYNAM-O closely. Key points:

    - Watershed on -spect with watershed_line=True leaves 0 between regions.
    - `expand_labels(labels, distance=5)` fills the 0-line so skimage's RAG
      detects adjacency between regions.
    - Per-region `border` = outer ring pixels that fall on the zero-line
      (NOT inner-edge pixels). This is what makes two adjacent regions
      share common border pixels in flat-index space.
    - Edge weight = max(w_ij, w_ji) with
          w_ij = 2*A_ij_max - B_i_min - j_max
          w_ji = 2*A_ij_max - B_j_min - i_max
    - Merge: regions union, borders symmetric-difference (setxor), recompute
      weights on all neighbours of the merged node.
    - Stop when max weight < merge_thresh (or NaN if no shared border pixels).

Border / region are stored as flat row-major indices into the
downsampled-spectrogram shape (H, W) so `data.ravel()[idx]` gives values.
"""

from __future__ import annotations

import numpy as np
from skimage import morphology
from skimage.graph import RAG
from skimage.segmentation import expand_labels, watershed

# Rust fast-path. dynamo_rs.merge_segment(labels, data, merge_thresh,
# max_merges) returns the merged label image (I32 same shape). Falls back
# to the pure-Python implementation below if the extension isn't installed.
try:
    import dynamo_rs as _dynamo_rs
    _HAS_RUST_MERGE = True
except ImportError:
    _dynamo_rs = None
    _HAS_RUST_MERGE = False


def _compute_edge_weight(
    rag: "RAG",
    u: int, v: int,
    data_flat: np.ndarray,
) -> float:
    """Directed edge weight between regions u and v (port of edge_weight)."""
    i_border = rag.nodes[u]["border"]
    i_region = rag.nodes[u]["region"]
    j_border = rag.nodes[v]["border"]
    j_region = rag.nodes[v]["region"]

    A_ij = np.intersect1d(i_border, j_border)
    if A_ij.size == 0:
        return np.nan

    A_ij_max = float(data_flat[A_ij].max())

    i_border_vals = data_flat[i_border]
    j_border_vals = data_flat[j_border]

    B_i_min = float(i_border_vals.min())
    B_j_min = float(j_border_vals.min())

    i_region_vals = data_flat[i_region]
    j_region_vals = data_flat[j_region]

    i_max = float(np.concatenate([i_border_vals, i_region_vals]).max())
    j_max = float(np.concatenate([j_border_vals, j_region_vals]).max())

    w_ij = 2 * A_ij_max - B_i_min - j_max
    w_ji = 2 * A_ij_max - B_j_min - i_max
    return max(w_ij, w_ji)


def _merge_region_borders(rag: "RAG", src: int, dst: int) -> None:
    """Absorb src's region/border into dst (matches merge_regions in pyDYNAM-O)."""
    rag.nodes[dst]["region"] = np.union1d(rag.nodes[dst]["region"],
                                           rag.nodes[src]["region"])
    rag.nodes[dst]["border"] = np.setxor1d(rag.nodes[dst]["border"],
                                            rag.nodes[src]["border"])


def _merge_nodes(rag: "RAG", src: int, dst: int,
                 data_flat: np.ndarray) -> None:
    """Move src's edges to dst, recompute weights for dst's neighbours,
    remove src (port of pyDYNAM-O's merge_nodes)."""
    src_nbrs = set(rag.neighbors(src))
    dst_nbrs = set(rag.neighbors(dst))
    neighbours = (src_nbrs | dst_nbrs) - {src, dst}

    for nb in neighbours:
        w = _compute_edge_weight(rag, dst, nb, data_flat)
        if np.isnan(w):
            if rag.has_edge(dst, nb):
                rag.remove_edge(dst, nb)
        else:
            rag.add_edge(dst, nb, weight=w)

    rag.remove_node(src)


def build_rag_from_watershed(labels: np.ndarray) -> "RAG":
    """Build a Region Adjacency Graph with interior/border pixels per node.

    labels : (H, W) watershed output with 0 = border line, >0 = region id
    """
    labels = np.ascontiguousarray(labels, dtype=np.int64)
    H, W = labels.shape

    # Expand by 5 so regions previously separated by the watershed 0-line
    # become spatially adjacent; skimage's RAG then finds the neighbour list.
    expanded = expand_labels(labels, distance=5)
    rag = RAG(expanded, connectivity=2)

    zero_mask = labels == 0
    struct_3x3 = np.ones((3, 3), dtype=bool)

    for n in list(rag.nodes):
        curr = labels == n
        # Border = watershed 0-line pixels adjacent to this region.
        # Interior and border are flat row-major indices.
        border_mask = morphology.dilation(curr, struct_3x3) & zero_mask
        bx, by = np.nonzero(border_mask)
        rag.nodes[n]["border"] = (bx * W + by).astype(np.int64)
        rx, ry = np.nonzero(curr)
        rag.nodes[n]["region"] = (rx * W + ry).astype(np.int64)

    return rag


def merge_segment(
    labels: np.ndarray,
    data: np.ndarray,
    merge_thresh: float = 8.0,
    max_merges: int | float = float("inf"),
    use_rust: bool | None = None,
) -> np.ndarray:
    """Iteratively merge adjacent regions until max edge weight drops
    below `merge_thresh` (or `max_merges` reached).

    Returns a (H, W) int label image with the merged region ids; pixels on
    the (now-thinner) watershed border stay 0 if they belong to a surviving
    node's border set, else get the surrounding region id.

    `use_rust`: None → use Rust kernel when available; True/False → force.
    """
    labels = np.ascontiguousarray(np.asarray(labels, dtype=np.int64))
    data = np.ascontiguousarray(np.asarray(data, dtype=np.float64))
    assert labels.shape == data.shape
    H, W = labels.shape

    # Rust fast path — ~100× faster than the Python implementation below.
    if use_rust is None:
        use_rust = _HAS_RUST_MERGE
    if use_rust:
        if not _HAS_RUST_MERGE:
            raise RuntimeError("use_rust=True but dynamo_rs is not installed.")
        max_m = float(max_merges) if np.isfinite(max_merges) else float("inf")
        merged = _dynamo_rs.merge_segment(
            labels, data, float(merge_thresh), max_m
        )
        return merged.astype(np.int64, copy=False)

    rag = build_rag_from_watershed(labels)
    data_flat = data.ravel()

    # Initial weights; drop edges whose border intersection is empty.
    for (u, v) in list(rag.edges):
        w = _compute_edge_weight(rag, u, v, data_flat)
        if np.isnan(w):
            rag.remove_edge(u, v)
        else:
            rag[u][v]["weight"] = w

    n_merges = 0
    while rag.number_of_edges() > 0:
        if n_merges >= max_merges:
            break
        # Pick max-weight edge
        u, v, w = max(
            ((uu, vv, d["weight"]) for uu, vv, d in rag.edges(data=True)),
            key=lambda x: x[2],
        )
        if w < merge_thresh:
            break
        _merge_region_borders(rag, u, v)
        _merge_nodes(rag, u, v, data_flat)
        n_merges += 1

    # Rebuild labels image. Paint region pixels with the node id.
    out = np.zeros_like(labels)
    flat = out.ravel()
    for n in rag.nodes:
        flat[rag.nodes[n]["region"]] = n
    # Zero out border pixels (downstream resize re-expands them)
    for n in rag.nodes:
        flat[rag.nodes[n]["border"]] = 0

    return out
