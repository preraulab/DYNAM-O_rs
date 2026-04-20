"""Verify the Rust MATLAB-watershed port against MATLAB ground truth."""

from pathlib import Path
import numpy as np
import pytest
import scipy.io as sio


MAT_PATH = Path("/Users/Mike/code/toolboxes/DYNAMO_dev/watershed analysis/watershed_matlab.mat")


@pytest.mark.skipif(not MAT_PATH.exists(), reason=f"{MAT_PATH} not found")
def test_rust_watershed_matches_matlab():
    """dynamo_rs.matlab_watershed must produce the same partition as
    MATLAB's `watershed(data, 8)` on the provided ground-truth test image."""
    dynamo_rs = pytest.importorskip("dynamo_rs")
    m = sio.loadmat(str(MAT_PATH), simplify_cells=True)
    data = np.ascontiguousarray(m["data"], dtype=np.float64)
    L_mat = np.asarray(m["L"])

    L_rs = dynamo_rs.matlab_watershed(data)

    # Shape and region count
    assert L_rs.shape == L_mat.shape
    assert L_rs.max() == L_mat.max(), (
        f"region-count mismatch: rust={L_rs.max()}, matlab={L_mat.max()}"
    )

    # Zero-pixel placement must match exactly
    diff_zero = ((L_rs == 0) != (L_mat == 0)).sum()
    assert diff_zero == 0, f"{diff_zero} pixels differ on zero placement"

    # Partition agreement (label-invariant) via ARI
    try:
        from sklearn.metrics import adjusted_rand_score
        both = (L_rs > 0) & (L_mat > 0)
        ari = adjusted_rand_score(L_rs[both].ravel(), L_mat[both].ravel())
        assert ari == 1.0, f"partition ARI = {ari}, expected 1.0"
    except ImportError:
        # sklearn not required; zero-placement is the strong signal
        pass
