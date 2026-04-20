"""Bisection Step 5b: does swapping in MATLAB's elliptic SOS filter close
the SOphase cos gap?

If YES: the scipy `iirdesign(..., ftype='ellip')` coefficients are the
problem — we should permanently load MATLAB's SOS.
If NO: the remaining gap is Hilbert-FFT + unwrap/rewrap + interp.

Runs twice: once with our current scipy filter, once with MATLAB's SOS
loaded from `bisect_intermediates_segment.mat:SOphase_filter_sos`.
"""
import sys
from pathlib import Path
import h5py
import numpy as np
from scipy.signal import hilbert, sosfiltfilt
from scipy.interpolate import interp1d

from pydynamo.io_compat import load_example_data
from pydynamo.soph.sophase import compute_so_phase, _design_so_bandpass


DC = Path(__file__).parent.parent.parent / "data_cache"
BISECT_MAT = DC / "bisect_intermediates_segment.mat"


def cos_sim(py, mt):
    both = np.isfinite(py) & np.isfinite(mt)
    a, b = py[both], mt[both]
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main() -> int:
    if not BISECT_MAT.exists():
        print(f"[step5b] MISSING {BISECT_MAT}")
        return 2

    ed = load_example_data()
    fs = float(ed["Fs"])
    T0, T1 = 8420, 13446
    i0 = int(round(T0 * fs))
    i1 = int(round(T1 * fs))
    data_tr = ed["data"].ravel()[i0 : i1 + 1].astype(np.float64)
    t_tr = np.arange(i0, i1 + 1) / fs
    stage_times = np.asarray(ed["stage_times"]).ravel()
    stage_vals = np.asarray(ed["stage_vals"]).ravel()

    with h5py.File(BISECT_MAT, "r") as f:
        ref_phase = np.squeeze(np.asarray(f["SOphase_norm"][...]).T).astype(float)
        # MATLAB v7.3 is col-major → h5py reads as (6, n_sections). Transpose
        # to scipy's (n_sections, 6) and force contiguous.
        sos_raw = np.asarray(f["SOphase_filter_sos"][...]).astype(np.float64)
        if sos_raw.shape[0] == 6:
            sos_raw = sos_raw.T
        sos_mat = np.ascontiguousarray(sos_raw, dtype=np.float64)
        # MATLAB's tf2sos returns (sos, g) with g separate. Fold g into the
        # first section's b0,b1,b2 so the concatenated filter has the right gain.
        try:
            g = float(np.asarray(f["SOphase_filter_g"][...]).item())
        except Exception:
            g = 1.0
        if g != 1.0:
            sos_mat = sos_mat.copy()
            sos_mat[0, :3] *= g

    from pydynamo.io_compat import load_segment_out
    artifacts = np.asarray(load_segment_out()["artifacts"]).ravel().astype(bool)

    print(f"MATLAB SOS shape: {sos_mat.shape}")

    # 5b-a: our default (scipy-designed) filter
    out_default, _, _, _ = compute_so_phase(
        data_tr, fs, stage_times=stage_times, stage_vals=stage_vals,
        eeg_times=t_tr, isexcluded=artifacts, SO_freqrange=(0.3, 1.5),
        sos_filter=None,
    )
    wrap_default = (out_default + np.pi) % (2 * np.pi) - np.pi

    # 5b-b: MATLAB's exact SOS
    out_mat, _, _, _ = compute_so_phase(
        data_tr, fs, stage_times=stage_times, stage_vals=stage_vals,
        eeg_times=t_tr, isexcluded=artifacts, SO_freqrange=(0.3, 1.5),
        sos_filter=sos_mat,
    )
    wrap_mat = (out_mat + np.pi) % (2 * np.pi) - np.pi

    # Compare timeseries
    cos_default = cos_sim(wrap_default, ref_phase)
    cos_matsos = cos_sim(wrap_mat, ref_phase)
    print(f"[step5b] SOphase timeseries vs MATLAB:")
    print(f"         scipy-designed filter: cos={cos_default:.6f}")
    print(f"         MATLAB SOS loaded:     cos={cos_matsos:.6f}")

    if cos_matsos > cos_default + 0.01:
        print("[step5b] FINDING: MATLAB SOS materially improves SOphase. Swap it in.")
    else:
        print("[step5b] FINDING: Filter is NOT the dominant source; look at Hilbert/unwrap.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
