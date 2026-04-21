"""End-to-end night pipeline: with vs without Rust unwrap+movmean flips.

Compares wall clock and SOpower/SOphase cos-similarity vs MATLAB baseline.

Run:
    .venv-matlab/bin/python scripts/bench_night_flips.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import scipy.io as sio

DC = Path(__file__).parent.parent / "data_cache"


def _sq(x):
    a = np.asarray(x)
    if a.ndim >= 2:
        a = a.T
    return np.squeeze(a)


def _read_mat_field(path, field_path):
    m = sio.loadmat(path, simplify_cells=True)
    v = m
    for k in field_path:
        v = v[k] if isinstance(v, dict) else getattr(v, k)
    return np.asarray(v).squeeze()


def _cos(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.dot(a[m], b[m]) / (np.linalg.norm(a[m]) * np.linalg.norm(b[m]) + 1e-20))


def _set_flips(use_rust_extras: bool):
    """Toggle the newly-introduced Rust extras (unwrap, movmean)."""
    import pydynamo.artifacts as A
    import pydynamo.soph.sophase as S
    A._USE_RUST_MOVMEAN = bool(use_rust_extras and A._HAS_RUST_MODULE)
    S._USE_RUST_UNWRAP = bool(use_rust_extras and S._HAS_RUST_MODULE)


def run_once(tag, use_rust_extras, ed, fs, time_range):
    _set_flips(use_rust_extras)
    from pydynamo import run_dynamo
    t0 = time.time()
    out = run_dynamo(
        ed["data"].ravel(), fs,
        ed["stage_times"].ravel(), ed["stage_vals"].ravel(),
        time_range=time_range,
        merge_thresh=11.0, trim_vol=0.8, seg_time=30.0,
        min_time_in_bin=10.0, min_peak_at_freq=0,
        double_watershed=True, refinement=True,
        plot=False, verbose=False,
    )
    dt = time.time() - t0
    return out, dt


def main():
    from pydynamo.io_compat import load_example_data
    ed = load_example_data()
    fs = float(ed["Fs"])
    sv = ed["stage_vals"].ravel(); st = ed["stage_times"].ravel()
    nw = np.flatnonzero((sv < 5) & (sv > 0))
    night_tr = (float(st[nw[0]]) - 300, float(st[nw[-1]]) + 300)
    print(f"night time_range: {night_tr[0]:.1f} - {night_tr[1]:.1f} s  ({(night_tr[1]-night_tr[0])/3600:.2f} h)")

    # MATLAB reference
    compat = DC / "night_out.mat"
    # Detect root key
    root = None
    for r in ("SOPHs_flat", "SOPHs"):
        try:
            _read_mat_field(compat, [r, "SOpower_mat"]); root = r; break
        except Exception:
            pass
    assert root is not None, "no SOPHs root found in night_out.mat"
    mat_sop = _read_mat_field(compat, [root, "SOpower_mat"]).astype(float)
    mat_ph = _read_mat_field(compat, [root, "SOphase_mat"]).astype(float)

    # Warmup (first run suffers from cold FFT planners / spectral caches)
    print("\n-- warmup --")
    run_once("warm", False, ed, fs, night_tr)

    # Alternating trials to average out ordering effects
    trials = [("off", False), ("on", True), ("on", True), ("off", False), ("off", False), ("on", True)]
    results = {"off": [], "on": []}
    last = {"off": None, "on": None}
    for tag, use in trials:
        print(f"-- run with rust extras {tag.upper()} --")
        out, dt = run_once(tag, use, ed, fs, night_tr)
        sop = np.asarray(out.SOPHs.SOpower_mat, float)
        ph = np.asarray(out.SOPHs.SOphase_mat, float)
        cos_sop = _cos(sop, mat_sop)
        cos_ph = _cos(ph, mat_ph)
        print(f"   wallclock = {dt:.2f}s  cos(SOpower)={cos_sop:.6f}  cos(SOphase)={cos_ph:.6f}")
        results[tag].append(dt)
        last[tag] = (sop, ph, cos_sop, cos_ph)

    def _med(xs): return float(np.median(xs))
    dt_off = _med(results["off"])
    dt_on = _med(results["on"])
    sop_off, ph_off, cos_sp_off, cos_ph_off = last["off"]
    sop_on, ph_on, cos_sp_on, cos_ph_on = last["on"]

    print(f"\n==== SUMMARY (median of {len(results['off'])} off, {len(results['on'])} on) ====")
    print(f"off trials: {[f'{x:.2f}' for x in results['off']]}")
    print(f"on  trials: {[f'{x:.2f}' for x in results['on']]}")
    print(f"wallclock: off={dt_off:.2f}s  on={dt_on:.2f}s  delta={dt_off - dt_on:+.2f}s ({(dt_off-dt_on)/dt_off*100:+.1f}%)")
    print(f"cos(SOpower): off={cos_sp_off:.6f}  on={cos_sp_on:.6f}  delta={cos_sp_on - cos_sp_off:+.2e}")
    print(f"cos(SOphase): off={cos_ph_off:.6f}  on={cos_ph_on:.6f}  delta={cos_ph_on - cos_ph_off:+.2e}")
    # Diff on finite values only
    m = np.isfinite(sop_on) & np.isfinite(sop_off)
    print(f"max-abs-diff (on vs off, finite-only) SOpower: {np.max(np.abs(sop_on[m] - sop_off[m])):.3e}")
    m = np.isfinite(ph_on) & np.isfinite(ph_off)
    print(f"max-abs-diff (on vs off, finite-only) SOphase: {np.max(np.abs(ph_on[m] - ph_off[m])):.3e}")


if __name__ == "__main__":
    sys.exit(main() or 0)
