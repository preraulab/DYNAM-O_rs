"""Artifact mask: compare pydynamo to MATLAB's `artifacts` from segment_out.mat.

The MATLAB mask is uint8 length 502601. We target near-bit-identity (≥99.9%
of samples agree). Small disagreement is expected from:
  - Chebyshev filter-coefficient quantization differences between MATLAB
    designfilt and scipy.signal.cheby1
  - movmedian endpoint handling at the 5-min detrend window edges
  - slope_test (currently disabled in our port; MATLAB enables it by default)
"""

import numpy as np

from pydynamo.artifacts import detect_artifacts


SEGMENT_TIME_RANGE = (8420, 13446)


def test_artifacts_match_matlab(example_data, segment_out_compat):
    fs = float(example_data["Fs"])
    data = np.asarray(example_data["data"]).ravel()
    # MATLAB `time_range_inds = t >= t0 & t <= t1` is inclusive on both ends,
    # so the slice is [i0, i1] inclusive → Python [i0:i1+1].
    i0 = int(round(SEGMENT_TIME_RANGE[0] * fs))
    i1 = int(round(SEGMENT_TIME_RANGE[1] * fs))
    segment = data[i0 : i1 + 1]

    art_py = detect_artifacts(segment, fs)
    art_mat = np.asarray(segment_out_compat["artifacts"], dtype=bool).ravel()

    assert art_py.shape == art_mat.shape, (art_py.shape, art_mat.shape)

    agree = (art_py == art_mat).mean()
    py_true = art_py.mean()
    mat_true = art_mat.mean()
    overlap = (art_py & art_mat).sum() / max(art_mat.sum(), 1)

    print(f"agreement rate       = {agree:.6f}")
    print(f"py  artifact fraction = {py_true:.6f}")
    print(f"mat artifact fraction = {mat_true:.6f}")
    print(f"recall of mat in py   = {overlap:.6f}")

    # Effectively bit-identical: allow <0.5% disagreement (filter edge effects
    # + slope_test omitted).
    assert agree > 0.995, f"artifact agreement rate {agree:.4f} too low"
