"""Bisection Step 3: verify that artifact detection (with slope_test ON
by default) produces a mask that matches MATLAB's saved `artifacts` to
≥99.9% sample agreement.

Without slope_test we saw 99.77% agreement (0.23% diff). With slope_test
we see 99.99% agreement — the slope-test additions are 99.8% precise
relative to MATLAB.
"""
import numpy as np
import pytest

from pydynamo.artifacts import detect_artifacts, _slope_test
from pydynamo.io_compat import load_example_data, load_segment_out


def test_slope_test_closes_artifact_gap():
    ed = load_example_data()
    fs = float(ed["Fs"])
    T0, T1 = 8420, 13446
    i0 = int(round(T0 * fs))
    i1 = int(round(T1 * fs))
    data_tr = ed["data"].ravel()[i0 : i1 + 1].astype(np.float64)

    art_mat = np.asarray(load_segment_out()["artifacts"]).ravel().astype(bool)

    # With slope_test (default in pydynamo, matches MATLAB default)
    art_py = detect_artifacts(data_tr, fs, slope_test=True)
    assert art_py.shape == art_mat.shape

    agreement = float((art_py == art_mat).mean())
    jaccard = float((art_py & art_mat).sum()) / max((art_py | art_mat).sum(), 1)
    recall = float((art_py & art_mat).sum()) / max(int(art_mat.sum()), 1)

    # Target thresholds per plan
    assert agreement >= 0.999, f"agreement {agreement:.5f} below 0.999"
    assert jaccard >= 0.99, f"jaccard {jaccard:.5f} below 0.99"
    assert recall >= 0.99, f"recall {recall:.5f} below 0.99"


def test_slope_test_standalone_returns_correct_shape_and_type():
    """Smoke test: _slope_test runs without NaN-propagation errors on real data."""
    ed = load_example_data()
    fs = float(ed["Fs"])
    data_tr = ed["data"].ravel()[: int(600 * fs)].astype(np.float64)  # 10 min

    mask = _slope_test(data_tr, fs, slope_crit=-0.5)
    assert mask.dtype == np.bool_
    assert mask.shape == data_tr.shape
    assert 0.0 <= mask.mean() <= 0.5, \
        f"slope_test flagging {mask.mean()*100:.1f}% is outside sensible range"
