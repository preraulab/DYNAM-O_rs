"""Per-swap parity + timing tests.

For each Rust swap, run the Rust path AND the Python fallback on identical
real-data inputs and assert byte-level (or near-byte-level) equality AND
record wall-clock time for both, appending to `data_cache/swap_timings.csv`
(one row per swap, per pytest run).

These are the gating tests the user asked for: "systematically swap out
each module to show no (or minimal) change in output" + "track change in
timing".

Data source: `data_cache/bisect_intermediates_segment.mat` (symlinked from
the sibling DYNAM-O_py repo; regenerate via its `scripts/export_*.m`).
"""
from __future__ import annotations

import csv
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

DC = Path(__file__).parent.parent / "data_cache"
BISECT = DC / "bisect_intermediates_segment.mat"
TIMINGS_CSV = DC / "swap_timings.csv"
N_REPEATS = int(os.environ.get("SWAP_PARITY_REPEATS", "5"))


def _log_timing(swap: str, py_ms: float, rust_ms: float, max_diff: float):
    """Append a row to data_cache/swap_timings.csv."""
    TIMINGS_CSV.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "swap": swap,
        "py_ms_median": f"{py_ms:.3f}",
        "rust_ms_median": f"{rust_ms:.3f}",
        "speedup_x": f"{py_ms / rust_ms:.2f}" if rust_ms > 0 else "inf",
        "max_abs_diff": f"{max_diff:.3e}",
    }
    write_header = not TIMINGS_CSV.exists()
    with TIMINGS_CSV.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=row.keys())
        if write_header:
            w.writeheader()
        w.writerow(row)
    # Also print to the pytest captured output
    print(f"\n[{swap}] py={py_ms:.1f}ms  rust={rust_ms:.1f}ms  "
          f"speedup={py_ms / rust_ms:.2f}x  max_diff={max_diff:.3e}")


def _bench(fn, *args, repeats: int = N_REPEATS):
    """Return (median ms, first-call result). Warm up once, then time."""
    _ = fn(*args)  # warmup
    timings = []
    result = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn(*args)
        timings.append((time.perf_counter() - t0) * 1000)
    return float(np.median(timings)), result


@pytest.fixture(scope="module")
def bisect_segment():
    if not BISECT.exists():
        pytest.skip(f"missing {BISECT}; run export_bisect_intermediates.m")
    import h5py
    def _sq(x):
        a = np.asarray(x)
        if a.ndim >= 2: a = a.T
        return np.squeeze(a)
    with h5py.File(BISECT, "r") as f:
        return {
            "spect1":   _sq(f["spect1"][...]).astype(np.float64),
            "spect2":   _sq(f["spect2"][...]).astype(np.float64),
            "stimes1":  _sq(f["stimes1"][...]).ravel().astype(np.float64),
            "stimes2":  _sq(f["stimes2"][...]).ravel().astype(np.float64),
            "sfreqs":   _sq(f["sfreqs"][...]).ravel().astype(np.float64),
            "artifacts": _sq(f["artifacts"][...]).astype(bool).ravel(),
        }


# ---------------------------------------------------------------------------
# swap #1 — baseline
# ---------------------------------------------------------------------------

def _python_baseline(spect, stimes, t_data, excl, rng, ptile):
    """Pure-Python reference implementation used before the Rust swap."""
    idx = np.searchsorted(t_data, stimes)
    idx = np.clip(idx, 0, t_data.size - 1)
    left = np.clip(idx - 1, 0, t_data.size - 1)
    use_left = np.abs(t_data[left] - stimes) < np.abs(t_data[idx] - stimes)
    idx = np.where(use_left, left, idx)
    exclude_stimes = excl[idx]
    in_range = (stimes >= rng[0]) & (stimes <= rng[1])
    valid = (~exclude_stimes) & in_range
    spect_bl = spect[:, valid].astype(np.float64, copy=True)
    spect_bl[spect_bl == 0] = np.nan
    return np.nanpercentile(spect_bl, ptile, axis=1, keepdims=True, method="hazen")


def test_swap1_baseline_parity(bisect_segment):
    d = bisect_segment
    fs = 100.0
    t_data = np.arange(d["artifacts"].size) / fs + d["stimes2"][0]

    py_ms, py = _bench(
        _python_baseline,
        d["spect2"], d["stimes2"], t_data,
        d["artifacts"], (-np.inf, np.inf), 2.0,
    )

    from pydynamo.baseline import compute_baseline
    rust_ms, rs = _bench(
        compute_baseline,
        d["spect2"], d["stimes2"], t_data,
        d["artifacts"], (-np.inf, np.inf), 2.0,
    )

    assert py.shape == rs.shape
    diff = float(np.abs(rs - py).max())
    _log_timing("baseline", py_ms, rust_ms, diff)
    # Bit-identity bar (hazen sort+interp on identical inputs must match)
    assert diff == 0.0, f"baseline swap introduced diff {diff}"


# ---------------------------------------------------------------------------
# swap #2 — mask_spectrogram
# ---------------------------------------------------------------------------

def _python_mask(spect_2s, stimes_2s, labels_1s, stimes_1s):
    """Pre-swap reference implementation."""
    from skimage.segmentation import find_boundaries
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


def test_swap2_mask_parity(bisect_segment):
    d = bisect_segment
    # Deterministic synthetic pass-1 label image (we only need something
    # with regions & watershed gaps so find_boundaries has work to do).
    F, T1 = d["spect1"].shape
    rng = np.random.default_rng(0)
    labels_1s = np.zeros((F, T1), dtype=np.int64)
    for i in range(1, 401):
        r = rng.integers(0, F - 10)
        c = rng.integers(0, T1 - 20)
        labels_1s[r:r+5, c:c+15] = i

    py_ms, py = _bench(
        _python_mask, d["spect2"], d["stimes2"], labels_1s, d["stimes1"]
    )
    from pydynamo.tfpeaks.mask import mask_spectrogram
    rust_ms, rs = _bench(
        mask_spectrogram, d["spect2"], d["stimes2"], labels_1s, d["stimes1"]
    )

    assert py.shape == rs.shape
    diff = float(np.abs(rs - py).max())
    _log_timing("mask_spectrogram", py_ms, rust_ms, diff)
    assert diff == 0.0, f"mask swap introduced diff {diff}"
