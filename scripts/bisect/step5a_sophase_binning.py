"""Bisection Step 5a: SOPH phase binning must be bit-identical to MATLAB
when given MATLAB's stats_table and MATLAB's SOphase timeseries.

Requires `data_cache/bisect_intermediates_segment.mat` with the
`SOphase_norm` (wrapped) + `SOphase_times` + `SOphase_stages` fields.
"""
import sys
from pathlib import Path
import h5py
import numpy as np
import pandas as pd

from pydynamo.io_compat import load_example_data
from pydynamo.soph.histogram import so_phase_histogram


DC = Path(__file__).parent.parent.parent / "data_cache"
BISECT_MAT = DC / "bisect_intermediates_segment.mat"


def main() -> int:
    if not BISECT_MAT.exists():
        print(f"[step5a] MISSING {BISECT_MAT}")
        print("        Run scripts/export_bisect_intermediates.m in MATLAB first.")
        return 2

    # Load MATLAB SOphase timeseries (wrapped)
    with h5py.File(BISECT_MAT, "r") as f:
        def r(k):
            a = np.asarray(f[k][...])
            return np.squeeze(a.T if a.ndim >= 2 else a)
        SOphase = r("SOphase_norm").astype(float)
        SOphase_times = r("SOphase_times").astype(float)
        SOphase_stages = r("SOphase_stages").astype(float)

    # MATLAB reference SOPHs
    with h5py.File(DC / "segment_out_compat.mat", "r") as f:
        ref_mat = np.squeeze(np.asarray(f["SOPHs_flat/SOphase_mat"][...]).T).astype(float)
        ref_fb = np.squeeze(np.asarray(f["SOPHs_flat/freq_bins"][...])).astype(float)

    # MATLAB stats_table
    stats = pd.read_csv(DC / "segment_stats.csv")
    pf = stats["PeakFrequency"].to_numpy(float)
    pt = stats["PeakTime"].to_numpy(float)
    ps = stats["PeakStage"].to_numpy(float)

    out = so_phase_histogram(
        pf, pt, ps,
        SOphase, SOphase_times, SOphase_stages,
        time_range=(8420.0, 13446.0),
        soph_stages=(1, 2, 3),
        freq_range=(0.0, 30.0), freq_binsizestep=(1.0, 0.2),
        so_range=(-np.pi, np.pi),
        so_binsizestep=(2 * np.pi / 5, 2 * np.pi / 100),
        min_peak_at_freq=1, compute_rate=True, norm_dim=1,
    )
    py_mat = out["c_mat"]
    if py_mat.shape != ref_mat.shape:
        print(f"[step5a] FAIL: shape mismatch py={py_mat.shape} ref={ref_mat.shape}")
        return 1
    both = np.isfinite(py_mat) & np.isfinite(ref_mat)
    a, b = py_mat[both], ref_mat[both]
    diff = a - b
    max_abs = float(np.abs(diff).max()) if a.size else float("nan")
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    print(f"[step5a] peaks={len(stats)}  shape={py_mat.shape}")
    print(f"[step5a] SOphase_mat max_abs={max_abs:.3e}  cos={cos:.6f}")
    if max_abs < 1e-12:
        print("[step5a] PASS: bit-identical")
        return 0
    print("[step5a] NOTE: non-bit-identical; investigate wrap convention or TIB")
    return 1


if __name__ == "__main__":
    sys.exit(main())
