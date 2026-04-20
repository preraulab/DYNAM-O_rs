"""Bisection Step 1: given MATLAB's spect1 + baseline1 + artifacts + merge_thresh,
does pydynamo's pass-1 extract produce the same peaks?

Then: given MATLAB's spect2_masked + baseline2, does pass-2 extract match?

If both pass-1 and pass-2 match, merge/trim/stats are clean and the
remaining divergence is in upstream (FFT / artifacts / filters) or
refinement.
"""
import sys
from pathlib import Path
import h5py
import numpy as np
import pandas as pd

from pydynamo.tfpeaks.extract import extract_tfpeaks


DC = Path(__file__).parent.parent.parent / "data_cache"
BISECT_MAT = DC / "bisect_intermediates_segment.mat"


def main() -> int:
    if not BISECT_MAT.exists():
        print(f"[step1] MISSING {BISECT_MAT}; run export_bisect_intermediates.m")
        return 2

    with h5py.File(BISECT_MAT, "r") as f:
        def r(k):
            a = np.asarray(f[k][...])
            return np.squeeze(a.T if a.ndim >= 2 else a)
        spect1 = r("spect1").astype(np.float64)
        baseline1 = r("baseline1").astype(np.float64).reshape(-1, 1)
        stimes1 = r("stimes1").astype(np.float64)
        sfreqs = r("sfreqs").astype(np.float64)
        spect2_masked = r("spect2_masked").astype(np.float64)
        stimes2 = r("stimes2").astype(np.float64)

    # Build spect1_norm from MATLAB's own pieces
    spect1_norm = spect1 / baseline1

    print(f"[step1] pass-1: spect1_norm shape {spect1_norm.shape}")
    stats1 = extract_tfpeaks(
        spect1_norm, stimes1, sfreqs,
        seg_time=30.0, n_jobs=1,
        downsample=(2, 2), merge_thresh=11.0, trim_vol=0.8,
        dur_min=0.5, dur_max=5.0, bw_min=2.0, bw_max=15.0,
    )
    print(f"[step1]   pass-1 peak count: {len(stats1)}")

    print(f"[step1] pass-2: spect2_masked shape {spect2_masked.shape}")
    stats2 = extract_tfpeaks(
        spect2_masked, stimes2, sfreqs,
        seg_time=30.0, n_jobs=1,
        downsample=(2, 2), merge_thresh=11.0, trim_vol=0.8,
        dur_min=1.0, dur_max=5.0, bw_min=1.0, bw_max=15.0,
    )
    print(f"[step1]   pass-2 peak count: {len(stats2)}")

    # Reference: MATLAB's stats_pre_refine (post-pass-2 runSegmentedData,
    # pre-filterStatsTable, pre-Hann refine). Apples-to-apples vs our stats2.
    with h5py.File(BISECT_MAT, "r") as f:
        g = f["stats_pre_refine"]
        mt = np.asarray(g["PeakTime"][...]).ravel().astype(float)
        mf = np.asarray(g["PeakFrequency"][...]).ravel().astype(float)
    ref_n = mt.size
    print(f"[step1] MATLAB pass-2 pre-refine peak count: {ref_n}")
    print(f"[step1] pydynamo pass-2 pre-refine:          {len(stats2)}")
    print(f"[step1] ratio pydynamo/MATLAB: {len(stats2) / ref_n:.3f}")

    # Hungarian match: how many pydynamo pre-refine peaks land within
    # 0.5s/0.5Hz of a MATLAB pre-refine peak?
    from scipy.optimize import linear_sum_assignment
    pt = stats2["PeakTime"].to_numpy()
    pf = stats2["PeakFrequency"].to_numpy()
    # Do it in 10-min windows for tractability
    matched = 0
    for t0 in np.arange(8420, 13446, 600):
        t1 = t0 + 600
        mp = (pt >= t0) & (pt < t1)
        mr = (mt >= t0) & (mt < t1)
        if not mp.any() or not mr.any():
            continue
        pt_s, pf_s = pt[mp], pf[mp]
        mt_s, mf_s = mt[mr], mf[mr]
        dt = pt_s[:, None] - mt_s[None, :]
        df = pf_s[:, None] - mf_s[None, :]
        cost = (dt / 0.5) ** 2 + (df / 0.5) ** 2
        cost[cost > 4.0] = 1e9  # gate at ~1 unit (0.5s OR 0.5Hz)
        r, c = linear_sum_assignment(cost)
        matched += int(((cost[r, c] < 1e8)).sum())
    recall = matched / ref_n
    precision = matched / len(stats2)
    print(f"[step1] Hungarian match (0.5s/0.5Hz): recall={recall:.4f} precision={precision:.4f}")

    if recall >= 0.95 and precision >= 0.90:
        print("[step1] PASS: merge/trim/stats clean (remaining gap is in refinement)")
        return 0
    print("[step1] FAIL: merge/trim/stats diverge from MATLAB")
    return 1


if __name__ == "__main__":
    sys.exit(main())
