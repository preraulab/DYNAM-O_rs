"""Sweep merge_thresh ∈ {6, 8, 10, 12} and measure SOPH MSE vs MATLAB.

Metric: mean squared error between my SOpower_mat / SOphase_mat and MATLAB's
SOpower_mat / SOphase_mat from segment_out.mat. Only the freq_range subset
that displaySummaryPlot.m actually shows (2-25 Hz) counts — out-of-range
bins are masked. NaN bins are pair-dropped.

For each thresh: run full pydynamo pipeline on the segment data, save
SOPHs, compute MSE. Parabolic fit → interpolated minimum.
"""

import json, time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from pydynamo import run_dynamo
from pydynamo.io_compat import load_example_data


DATA_CACHE = Path(__file__).parent.parent / "data_cache"
THRESHOLDS = [6.0, 8.0, 10.0, 12.0]
TIME_RANGE = (8420.0, 13446.0)     # segment from runExampleData.m
FREQ_LIMITS = (2.0, 25.0)          # restrict comparison to MATLAB display range


def load_matlab_ref():
    with h5py.File(DATA_CACHE / "segment_out_compat.mat", "r") as f:
        g = {}
        for k in ["SOpower_mat", "SOphase_mat",
                  "SOpower_bins", "SOphase_bins", "freq_bins"]:
            arr = np.asarray(f[f"SOPHs_flat/{k}"][...])
            if arr.ndim >= 2:
                arr = arr.T
            g[k] = np.squeeze(arr).astype(float)
    return g


def soph_mse(py_mat, mat_mat, freq_bins, freq_lo, freq_hi):
    """Pair-wise MSE over finite cells within freq_lo..freq_hi.
    Returns (mse, n_cells_compared, cosine_similarity, l2_rel_diff).
    """
    f_sel = (freq_bins >= freq_lo) & (freq_bins <= freq_hi)
    py = py_mat[:, f_sel]
    mt = mat_mat[:, f_sel]
    if py.shape != mt.shape:
        raise ValueError(f"shape mismatch py={py.shape} mat={mt.shape}")
    both = np.isfinite(py) & np.isfinite(mt)
    if not both.any():
        return float("inf"), 0, 0.0, float("inf")
    a = py[both].astype(float)
    b = mt[both].astype(float)
    mse = float(np.mean((a - b) ** 2))
    cos_sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    l2 = float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))
    return mse, int(both.sum()), cos_sim, l2


def main():
    ed = load_example_data()
    data = ed["data"].ravel()
    fs = float(ed["Fs"])
    stage_times = ed["stage_times"].ravel()
    stage_vals = ed["stage_vals"].ravel()
    ref = load_matlab_ref()

    results = []
    for thresh in THRESHOLDS:
        print(f"\n===== merge_thresh = {thresh} =====")
        tic = time.time()
        out = run_dynamo(
            data, fs, stage_times, stage_vals,
            time_range=TIME_RANGE, merge_thresh=thresh,
            trim_vol=0.8, seg_time=30.0,
            min_time_in_bin=5.0,                 # runExampleData override
            double_watershed=True, refinement=True,
            plot=False, verbose=False,
        )
        dt = time.time() - tic
        n_peaks = len(out.stats_table)
        pow_mse, pow_n, pow_cos, pow_l2 = soph_mse(
            out.SOPHs.SOpower_mat, ref["SOpower_mat"],
            out.SOPHs.freq_bins, *FREQ_LIMITS,
        )
        ph_mse, ph_n, ph_cos, ph_l2 = soph_mse(
            out.SOPHs.SOphase_mat, ref["SOphase_mat"],
            out.SOPHs.freq_bins, *FREQ_LIMITS,
        )
        rec = {
            "merge_thresh": thresh, "wallclock_s": dt, "n_peaks": n_peaks,
            "SOpower_mse": pow_mse, "SOpower_cos": pow_cos, "SOpower_l2rel": pow_l2,
            "SOphase_mse": ph_mse, "SOphase_cos": ph_cos, "SOphase_l2rel": ph_l2,
        }
        print(json.dumps(rec, indent=2))
        results.append(rec)
        # Save per-threshold SOPHs for diagnostics
        np.savez_compressed(
            DATA_CACHE / f"sweep_thresh_{thresh:g}.npz",
            SOpower_mat=out.SOPHs.SOpower_mat,
            SOphase_mat=out.SOPHs.SOphase_mat,
            SOpower_bins=out.SOPHs.SOpower_bins,
            SOphase_bins=out.SOPHs.SOphase_bins,
            freq_bins=out.SOPHs.freq_bins,
        )

    # Parabolic fit → minimum (use combined metric: average normalized L2)
    thrs = np.array([r["merge_thresh"] for r in results])
    combined = np.array([(r["SOpower_l2rel"] + r["SOphase_l2rel"]) / 2
                          for r in results])
    coeffs = np.polyfit(thrs, combined, 2)
    thr_min = -coeffs[1] / (2 * coeffs[0]) if coeffs[0] > 0 else thrs[np.argmin(combined)]
    thr_min = float(np.clip(thr_min, thrs.min() - 1, thrs.max() + 1))
    print("\n===== SWEEP SUMMARY =====")
    print(f"{'thresh':>8} {'n_peaks':>8} {'wall_s':>8} "
           f"{'pow_mse':>10} {'pow_cos':>8} {'pow_l2':>8} "
           f"{'ph_mse':>10} {'ph_cos':>8} {'ph_l2':>8}")
    for r in results:
        print(f"{r['merge_thresh']:>8.1f} {r['n_peaks']:>8} {r['wallclock_s']:>8.1f} "
              f"{r['SOpower_mse']:>10.4f} {r['SOpower_cos']:>8.4f} {r['SOpower_l2rel']:>8.4f} "
              f"{r['SOphase_mse']:>10.6f} {r['SOphase_cos']:>8.4f} {r['SOphase_l2rel']:>8.4f}")
    print(f"\nParabola-fit minimum combined-L2 at merge_thresh ≈ {thr_min:.2f}")
    with open(DATA_CACHE / "sweep_results.json", "w") as f:
        json.dump({"results": results, "parabola_min_thresh": thr_min}, f, indent=2)
    print(f"wrote sweep_results.json")


if __name__ == "__main__":
    main()
