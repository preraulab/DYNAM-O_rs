"""Bisection Step 9: Hann-window frequency refinement.

Given MATLAB's stats_pre_refine (peaks pre-refinement), run pydynamo's
refine_peak_frequency and compare refined PeakFrequencies to MATLAB's
final post-refine values from segment_stats.csv.

If refinement is clean: matching pydynamo pre-refine peaks will end up
at nearly the same refined freq as MATLAB.
If not: source is scipy CubicSpline vs MATLAB interp1(..., 'spline'),
or Hann FFT noise.
"""
import sys
from pathlib import Path
import h5py
import numpy as np
import pandas as pd

from pydynamo.io_compat import load_example_data
from pydynamo.tfpeaks.refine import refine_peak_frequency


DC = Path(__file__).parent.parent.parent / "data_cache"
BISECT_MAT = DC / "bisect_intermediates_segment.mat"


def main() -> int:
    if not BISECT_MAT.exists():
        print(f"[step9] MISSING {BISECT_MAT}")
        return 2

    # Load MATLAB pre-refine stats
    with h5py.File(BISECT_MAT, "r") as f:
        g = f["stats_pre_refine"]
        mat_pt = np.asarray(g["PeakTime"][...]).ravel().astype(float)
        mat_pf_pre = np.asarray(g["PeakFrequency"][...]).ravel().astype(float)
        mat_dur = np.asarray(g["Duration"][...]).ravel().astype(float)
        mat_bw = np.asarray(g["Bandwidth"][...]).ravel().astype(float)

    # Build a pydynamo-compatible stats DataFrame from MATLAB pre-refine rows
    n = mat_pt.size
    stats = pd.DataFrame({
        "PeakTime": mat_pt,
        "PeakFrequency": mat_pf_pre,
        "Duration": mat_dur,
        "Bandwidth": mat_bw,
        # BoundingBox = (time_tl, freq_tl, width, height) per-peak
        "BoundingBox": [
            (float(pt - d / 2), float(pf - bw / 2), float(d), float(bw))
            for pt, pf, d, bw in zip(mat_pt, mat_pf_pre, mat_dur, mat_bw)
        ],
    })

    ed = load_example_data()
    fs = float(ed["Fs"])
    T0, T1 = 8420, 13446
    i0 = int(round(T0 * fs))
    i1 = int(round(T1 * fs))
    data_tr = ed["data"].ravel()[i0 : i1 + 1].astype(np.float64)
    t_tr = np.arange(i0, i1 + 1) / fs

    refined = refine_peak_frequency(
        stats, data_tr, fs, t=t_tr,
        freq_range=(0.0, 30.0), window_size=4.0, dsfreqs=0.05,
        refine_method="spline_interp", remove_edge_peaks=True,
    )
    print(f"[step9] pre-refine  count: {n}")
    print(f"[step9] post-refine count: {len(refined)}  (ratio: {len(refined)/n:.3f})")

    # Find MATLAB post-refine peaks that started from a pre-refine peak we have
    # — we match pre-refine peaks by (PeakTime, pre-refine PeakFrequency) and
    # compare the refined frequency.
    mat_final = pd.read_csv(DC / "segment_stats.csv")
    mat_final_pt = mat_final["PeakTime"].to_numpy()
    mat_final_pf = mat_final["PeakFrequency"].to_numpy()

    # For each refined pydynamo peak, find nearest MATLAB post-refine peak
    # by PeakTime (they share peak IDs if we match by time since time isn't
    # changed by refinement).
    r_pt = refined["PeakTime"].to_numpy()
    r_pf = refined["PeakFrequency"].to_numpy()
    # Vectorized nearest-time match
    from scipy.spatial import cKDTree
    tree = cKDTree(mat_final_pt[:, None])
    dist, idx = tree.query(r_pt[:, None], k=1)
    # Match only peaks within 0.01s of a MATLAB peak
    ok = dist < 0.01
    print(f"[step9] peaks with a MATLAB time-match (<0.01s): {ok.sum()}/{len(r_pt)}")
    if ok.any():
        diff_pf = r_pf[ok] - mat_final_pf[idx[ok]]
        print(f"[step9] refined PeakFrequency diff vs MATLAB:")
        print(f"         median={np.median(diff_pf):+.5f} Hz")
        print(f"         p50_abs={np.median(np.abs(diff_pf)):.5f} Hz")
        print(f"         p95_abs={np.quantile(np.abs(diff_pf), 0.95):.5f} Hz")
        print(f"         max_abs={np.abs(diff_pf).max():.5f} Hz")

    return 0


if __name__ == "__main__":
    sys.exit(main())
