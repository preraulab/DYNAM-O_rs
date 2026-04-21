"""Side-by-side table: MATLAB | Python | Rust.

Reports:
  - Wall-clock time per stage (artifact, spect*, baseline*, extract*, refine,
    peak_*, soph_*)
  - Peak counts at each step (post-pass-1, post-pass-2, post-refine = final)
  - SOPH similarity vs MATLAB (cos) for Python and Rust

Usage:  python scripts/compare_table.py [segment|night|both]
"""
from __future__ import annotations

import io
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.io as sio

DC = Path(__file__).parent.parent / "data_cache"


def _sq(x):
    a = np.asarray(x)
    if a.ndim >= 2:
        a = a.T
    return np.squeeze(a)


# Stage display order + friendly labels
STAGES = [
    ("artifact",           "artifacts"),
    ("spect_pass1",        "spect pass-1"),
    ("baseline_pass1",     "baseline pass-1"),
    ("extract_pass1",      "extract pass-1"),
    ("spect_pass2",        "spect pass-2"),
    ("baseline_pass2",     "baseline pass-2"),
    ("extract_pass2",      "extract pass-2"),
    ("refine",             "Hann refinement"),
    ("peak_sopower",       "assign SO-power"),
    ("peak_sophase",       "assign SO-phase"),
    ("soph_sopower_hist",  "SO-power histogram"),
    ("soph_sophase_hist",  "SO-phase histogram"),
    ("total",              "TOTAL"),
]


def _force_python_only(on: bool):
    """Toggle every Rust fast-path flag in the currently-loaded pydynamo."""
    modnames = [
        "pydynamo.baseline",
        "pydynamo.tfpeaks.mask",
        "pydynamo.soph.histogram",
        "pydynamo.tfpeaks.refine",
        "pydynamo.soph.sophase",
        "pydynamo.artifacts",
    ]
    flips = []
    for m in modnames:
        mod = sys.modules.get(m)
        if mod is None:
            continue
        for attr in ("_HAS_RUST", "_HAS_RUST_SIGNAL"):
            if hasattr(mod, attr):
                prev = getattr(mod, attr)
                setattr(mod, attr, not on if prev else False)
                # simpler: just force True if on, False if off
                setattr(mod, attr, False if on else prev)
                flips.append((mod, attr, prev))
    try:
        import pydynamo.tfpeaks.extract as _ex
        if hasattr(_ex, "_HAS_RUST_WS"):
            prev = _ex._HAS_RUST_WS
            _ex._HAS_RUST_WS = False if on else prev
            flips.append((_ex, "_HAS_RUST_WS", prev))
    except ImportError:
        pass
    try:
        import pydynamo.tfpeaks.merge as _mg
        if hasattr(_mg, "_HAS_RUST_MERGE"):
            prev = _mg._HAS_RUST_MERGE
            _mg._HAS_RUST_MERGE = False if on else prev
            flips.append((_mg, "_HAS_RUST_MERGE", prev))
    except ImportError:
        pass
    return flips


def _restore(flips):
    for mod, attr, prev in flips:
        setattr(mod, attr, prev)


def run_backend(backend: str, ed, fs, time_range, min_tib, min_paf):
    """Run the pipeline in the given backend. Capture timings, peak counts
    per stage, and the final SOPH output."""
    from pydynamo import run_dynamo

    flips = _force_python_only(backend == "python")
    try:
        # We want the verbose peak counts; capture stdout.
        buf = io.StringIO()
        with redirect_stdout(buf):
            t0 = time.time()
            out = run_dynamo(
                ed["data"].ravel(), fs,
                ed["stage_times"].ravel(), ed["stage_vals"].ravel(),
                time_range=time_range,
                merge_thresh=11.0, trim_vol=0.8, seg_time=30.0,
                min_time_in_bin=min_tib, min_peak_at_freq=min_paf,
                double_watershed=True, refinement=True,
                plot=False, verbose=True,
            )
            wall = time.time() - t0
        text = buf.getvalue()
    finally:
        _restore(flips)

    # Parse peak counts from verbose output
    pass1 = pass2 = final_cnt = None
    for line in text.splitlines():
        if "pass-1:" in line and "peaks" in line:
            pass1 = int(line.rsplit(":", 1)[1].strip().split()[0])
        elif "pass-2:" in line and "peaks" in line:
            pass2 = int(line.rsplit(":", 1)[1].strip().split()[0])
        elif "after refinement:" in line:
            final_cnt = int(line.split(":")[-1].strip().split()[0])
    if final_cnt is None:
        final_cnt = len(out.stats_table)

    return {
        "timings": out.timings,
        "peaks_pass1": pass1,
        "peaks_pass2": pass2,
        "peaks_final": final_cnt,
        "out": out,
        "wall": wall,
    }


def load_matlab(path: Path):
    """Load MATLAB timings dict + SOPHs + stats CSV peak count."""
    try:
        m = sio.loadmat(path, simplify_cells=True)
    except Exception:
        m = {}
    timings = m.get("timings") if isinstance(m.get("timings"), dict) else {}

    # v7.3 fallback
    if not timings:
        try:
            with h5py.File(path, "r") as f:
                if "timings" in f:
                    g = f["timings"]
                    timings = {}
                    for k in g.keys():
                        v = np.asarray(g[k][...]).squeeze()
                        if v.size == 1:
                            timings[k] = float(v)
        except OSError:
            pass

    # SOPHs
    sop_mat = soph_mat = None
    for root in ("SOPHs_flat", "SOPHs"):
        try:
            if root in m and isinstance(m[root], dict):
                sop_mat = np.asarray(m[root]["SOpower_mat"])
                soph_mat = np.asarray(m[root]["SOphase_mat"])
                break
            with h5py.File(path, "r") as f:
                if root in f:
                    sop_mat = _sq(f[f"{root}/SOpower_mat"][...])
                    soph_mat = _sq(f[f"{root}/SOphase_mat"][...])
                    break
        except Exception:
            pass
    return timings, sop_mat, soph_mat


def cos_sim(a, b):
    if a is None or b is None:
        return float("nan")
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if not m.any():
        return float("nan")
    return float(np.dot(a[m], b[m]) / (np.linalg.norm(a[m]) * np.linalg.norm(b[m]) + 1e-20))


def _fmt(v):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def build_table(name, time_range, compat_mat, stats_csv, min_tib, min_paf):
    from pydynamo.io_compat import load_example_data
    ed = load_example_data()
    fs = float(ed["Fs"])

    print(f"\n=== {name} ({time_range[0]}s – {time_range[1]}s) ===\n")

    print("running RUST...")
    rs = run_backend("rust", ed, fs, time_range, min_tib, min_paf)
    print("running PYTHON-only...")
    py = run_backend("python", ed, fs, time_range, min_tib, min_paf)

    print("loading MATLAB...")
    mat_timings, mat_sop, mat_ph = load_matlab(compat_mat)
    mat_final = len(pd.read_csv(stats_csv)) if stats_csv.exists() else None

    # Build table rows
    rows = []
    rows.append(("stage", "MATLAB (s)", "Python (s)", "Rust (s)"))
    rows.append(("---", "---:", "---:", "---:"))
    for key, label in STAGES:
        m = mat_timings.get(key)
        p = py["timings"].get(key)
        r = rs["timings"].get(key)
        rows.append((label, _fmt(m), _fmt(p), _fmt(r)))

    rows.append(("", "", "", ""))
    rows.append(("peaks", "MATLAB", "Python", "Rust"))
    rows.append(("---", "---:", "---:", "---:"))
    rows.append(("post pass-1", "—",
                 _fmt(py["peaks_pass1"]), _fmt(rs["peaks_pass1"])))
    rows.append(("post pass-2 (pre-refine)", "—",
                 _fmt(py["peaks_pass2"]), _fmt(rs["peaks_pass2"])))
    rows.append(("final (post-refine)", _fmt(mat_final),
                 _fmt(py["peaks_final"]), _fmt(rs["peaks_final"])))

    rows.append(("", "", "", ""))
    rows.append(("SOPH cos vs MATLAB", "MATLAB (ref)", "Python", "Rust"))
    rows.append(("---", "---:", "---:", "---:"))
    sp_py = cos_sim(py["out"].SOPHs.SOpower_mat, mat_sop)
    sp_rs = cos_sim(rs["out"].SOPHs.SOpower_mat, mat_sop)
    ph_py = cos_sim(py["out"].SOPHs.SOphase_mat, mat_ph)
    ph_rs = cos_sim(rs["out"].SOPHs.SOphase_mat, mat_ph)
    rows.append(("SOpower cos", "1.0000", f"{sp_py:.4f}", f"{sp_rs:.4f}"))
    rows.append(("SOphase cos", "1.0000", f"{ph_py:.4f}", f"{ph_rs:.4f}"))

    # Speedup summary
    rows.append(("", "", "", ""))
    t_mat = mat_timings.get("total")
    t_py = py["timings"].get("total")
    t_rs = rs["timings"].get("total")
    if t_mat and t_py and t_rs:
        rows.append(("speedup vs MATLAB", "1.00×",
                     f"{t_mat / t_py:.2f}×", f"{t_mat / t_rs:.2f}×"))
        rows.append(("speedup vs Python", f"{t_py / t_mat:.2f}×",
                     "1.00×", f"{t_py / t_rs:.2f}×"))

    return rows


def print_table(rows):
    # Column widths
    widths = [
        max(len(str(r[i])) for r in rows)
        for i in range(len(rows[0]))
    ]
    for r in rows:
        line = " | ".join(str(r[i]).ljust(widths[i]) for i in range(len(r)))
        print(line)


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    datasets = []
    if which in ("segment", "both"):
        datasets.append(("segment", (8420.0, 13446.0),
                          DC / "segment_out_compat.mat",
                          DC / "segment_stats.csv", 5.0, 10))
    if which in ("night", "both"):
        from pydynamo.io_compat import load_example_data
        ed = load_example_data()
        sv = ed["stage_vals"].ravel(); st = ed["stage_times"].ravel()
        nw = np.flatnonzero((sv < 5) & (sv > 0))
        tr = (float(st[nw[0]]) - 300, float(st[nw[-1]]) + 300)
        datasets.append(("night", tr,
                          DC / "night_out.mat",
                          DC / "night_stats.csv", 10.0, 0))

    for args in datasets:
        rows = build_table(*args)
        print_table(rows)
        print()


if __name__ == "__main__":
    main()
