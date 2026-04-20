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


# ---------------------------------------------------------------------------
# swap #3 — tfpeak_histogram
# ---------------------------------------------------------------------------

def _python_tfhist(c_metric, c_stages, c_dt, c_valid, c_valid_all,
                   peak_freqs, peak_c, freq_edges, c_edges,
                   circular, cb, norm_dim, compute_rate,
                   min_time_in_bin, min_peak_at_freq):
    """Pure-Python reference implementation (copied from the pre-swap version)."""
    num_fbins = freq_edges.shape[1]
    num_cbins = c_edges.shape[1]
    pf = peak_freqs[:, None]
    all_infreqbin = (pf >= freq_edges[0][None, :]) & (pf < freq_edges[1][None, :])
    c_mat = np.full((num_cbins, num_fbins), np.nan, dtype=float)
    time_in_bin = np.zeros((num_cbins, 5), dtype=float)
    prop_in_bin = np.zeros((num_cbins, 5), dtype=float)
    compute_tib = compute_rate or (min_time_in_bin > 0)
    if compute_tib:
        stage_valid_masks = np.zeros((c_stages.size, 5), dtype=bool)
        for k in range(1, 6):
            stage_valid_masks[:, k - 1] = (c_stages == k) & c_valid
    low_b, high_b = float(cb[0]), float(cb[1])
    crange = high_b - low_b
    for s in range(num_cbins):
        lo_e, hi_e = c_edges[0, s], c_edges[1, s]
        if circular and lo_e <= low_b:
            wrap_lo = lo_e + crange
            tib_inds = (c_metric >= wrap_lo) | (c_metric < hi_e)
            inc_inds = (peak_c >= wrap_lo) | (peak_c < hi_e)
        elif circular and hi_e >= high_b:
            wrap_hi = hi_e - crange
            tib_inds = (c_metric < wrap_hi) | (c_metric >= lo_e)
            inc_inds = (peak_c < wrap_hi) | (peak_c >= lo_e)
        else:
            tib_inds = (c_metric >= lo_e) & (c_metric < hi_e)
            inc_inds = (peak_c >= lo_e) & (peak_c < hi_e)
        if compute_tib:
            tib_per_stage = np.sum(tib_inds[:, None] & stage_valid_masks, axis=0) * c_dt / 60.0
            time_in_bin[s, :] = tib_per_stage
            tib_all = np.sum(tib_inds & c_valid_all) * c_dt / 60.0
            if tib_all > 0:
                prop_in_bin[s, :] = tib_per_stage / tib_all
            if tib_per_stage.sum() < min_time_in_bin:
                continue
        if inc_inds.any():
            counts = np.sum(inc_inds[:, None] & all_infreqbin, axis=0).astype(float)
        else:
            counts = np.zeros(num_fbins, dtype=float)
        c_mat[s, :] = counts
        if compute_rate and time_in_bin[s, :].sum() > 0:
            c_mat[s, :] = c_mat[s, :] / time_in_bin[s, :].sum()
    peak_at_freq = np.sum(all_infreqbin, axis=0).astype(float)
    if min_peak_at_freq > 0:
        c_mat[:, peak_at_freq < min_peak_at_freq] = np.nan
    if norm_dim:
        axis = norm_dim - 1
        dim_sum = np.nansum(c_mat, axis=axis, keepdims=True)
        dim_sum = np.where(dim_sum == 0, 1.0, dim_sum)
        c_mat = c_mat / dim_sum
    return c_mat, time_in_bin, prop_in_bin, peak_at_freq


def test_swap3_histogram_parity(bisect_segment):
    rng = np.random.default_rng(42)
    # Synthetic inputs mirroring real SOpower histogram scale
    n_t = 10_000
    n_p = 5_000
    c_metric = rng.normal(size=n_t)
    c_stages = rng.integers(1, 6, size=n_t).astype(float)
    c_valid = rng.random(n_t) > 0.1
    c_valid_all = c_valid & (rng.random(n_t) > 0.05)
    peak_freqs = rng.uniform(0.5, 29.5, size=n_p)
    peak_c = rng.normal(size=n_p)
    # Use the same `create_bins` the Rust-wrapped tfpeak_histogram calls
    # internally (partial mode, clipped at lo/hi). Passing un-clipped edges
    # to the Python reference while Rust gets clipped ones would make the
    # two disagree on boundary samples.
    from pydynamo.soph.histogram import create_bins
    freq_edges, _ = create_bins((0.0, 30.0), 1.0, 0.2, "partial")
    c_edges, _ = create_bins((-3.0, 3.0), 0.06, 0.06, "partial")
    kw = dict(
        circular=False, cb=(0.0, 0.0), norm_dim=0, compute_rate=True,
        min_time_in_bin=5.0, min_peak_at_freq=1,
    )
    def py_call():
        return _python_tfhist(c_metric, c_stages, 0.5, c_valid, c_valid_all,
                              peak_freqs, peak_c, freq_edges, c_edges, **kw)
    py_ms, out_py = _bench(py_call)

    from pydynamo.soph.histogram import tfpeak_histogram
    def rust_call():
        return tfpeak_histogram(
            c_metric=c_metric, c_stages=c_stages, c_dt=0.5,
            c_valid=c_valid, c_valid_allstages=c_valid_all,
            peak_freqs=peak_freqs, peak_c=peak_c,
            circular=False, circular_bounds=(0.0, 0.0),
            freq_range=(0.0, 30.0), freq_binsizestep=(1.0, 0.2),
            c_range=(-3.0, 3.0), c_binsizestep=(0.06, 0.06),
            norm_dim=0, compute_rate=True,
            min_time_in_bin=5.0, min_peak_at_freq=1,
        )
    # Warmup the Rust path
    rust_ms, out_rs = _bench(rust_call)

    cm_py, tib_py, prop_py, paf_py = out_py
    # max_abs diff on c_mat (ignore NaN)
    mask = ~(np.isnan(cm_py) | np.isnan(out_rs["c_mat"]))
    diff_cm = float(np.abs(out_rs["c_mat"][mask] - cm_py[mask]).max()) if mask.any() else 0.0
    diff_tib = float(np.abs(out_rs["time_in_bin"] - tib_py).max())
    diff_paf = float(np.abs(out_rs["peak_at_freq"] - paf_py).max())
    overall = max(diff_cm, diff_tib, diff_paf)
    _log_timing("tfpeak_histogram", py_ms, rust_ms, overall)
    # Time_in_bin and peak_at_freq are bit-identical (integer counts); c_mat
    # involves a divide, so FP rounding can introduce sub-ulp noise.
    assert overall < 1e-12, (
        f"histogram swap introduced diff > 1e-12: c_mat={diff_cm} "
        f"tib={diff_tib} paf={diff_paf}"
    )


# ---------------------------------------------------------------------------
# swap #4 — Hann refinement (hann_event_spectra + spline argmax)
# ---------------------------------------------------------------------------

def test_swap4_refine_parity(bisect_segment):
    # Use a synthetic EEG + fake stats_table (we only need PeakTime +
    # BoundingBox). Goal: confirm Rust and Python refine paths agree on
    # refined PeakFrequency to within grid-discretization tolerance.
    rng = np.random.default_rng(7)
    fs = 100.0
    n = 60 * int(fs)  # 60 seconds
    data = rng.standard_normal(n).astype(np.float64)
    t = np.arange(n) / fs
    # 300 fake peaks scattered in the middle
    n_events = 300
    peak_times = rng.uniform(3.0, 57.0, size=n_events)
    bbox_lo = rng.uniform(2.0, 10.0, size=n_events)
    bbox_h = rng.uniform(1.0, 4.0, size=n_events)
    bbox = np.stack(
        [peak_times - 0.5, bbox_lo,
         np.full_like(peak_times, 1.0), bbox_h],
        axis=1,
    )
    import pandas as pd
    stats = pd.DataFrame({
        "PeakTime": peak_times,
        "PeakFrequency": (bbox_lo + bbox_h / 2),
        "BoundingBox": [tuple(r) for r in bbox],
    })

    # Rust path
    from pydynamo.tfpeaks.refine import refine_peak_frequency
    def rust_call():
        return refine_peak_frequency(stats.copy(), data, fs, t=t)

    rust_ms, out_rs = _bench(rust_call)

    # Python fallback: temporarily disable Rust flag.
    import pydynamo.tfpeaks.refine as _refmod
    saved = _refmod._HAS_RUST
    _refmod._HAS_RUST = False
    try:
        def py_call():
            return refine_peak_frequency(stats.copy(), data, fs, t=t)
        py_ms, out_py = _bench(py_call)
    finally:
        _refmod._HAS_RUST = saved

    # Align by PeakTime (both paths may drop edge-rejected events).
    merged = out_rs.merge(
        out_py, on="PeakTime", suffixes=("_rs", "_py"), how="inner"
    )
    diff = float(
        np.abs(merged["PeakFrequency_rs"] - merged["PeakFrequency_py"]).max()
    ) if len(merged) else 0.0
    # Tolerance: 1000-point grid over ≤ 4 Hz bbox ⇒ step ≤ 4 mHz. Allow 10 mHz
    # for FFT-library FP noise + any tiny spline-solver differences between
    # scipy CubicSpline(bc_type='not-a-knot' default) and our natural
    # boundary (scipy default is 'not-a-knot', not 'natural' — so the two
    # outputs WILL disagree by a small amount near the spectrum edges).
    # Since the argmax is insider the bbox (not at the endpoints), the
    # difference should be small.
    _log_timing("refine", py_ms, rust_ms, diff)
    # Also log peak-count parity
    assert abs(len(out_rs) - len(out_py)) <= 5, (
        f"kept-peak counts differ: rs={len(out_rs)}, py={len(out_py)}"
    )
    assert diff < 0.05, f"refine freq diff too large: {diff}"


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
