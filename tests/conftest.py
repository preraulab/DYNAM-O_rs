"""Shared pytest fixtures for pydynamo equivalence tests."""

from pathlib import Path

import pytest

DATA_CACHE = Path(__file__).parent.parent / "data_cache"


@pytest.fixture(scope="session")
def segment_out_compat():
    """segment_out_compat.mat produced by scripts/export_matlab_ground_truth.m."""
    import h5py  # v7.3 MAT = HDF5
    path = DATA_CACHE / "segment_out_compat.mat"
    if not path.exists():
        pytest.skip(
            f"{path} not found. Run scripts/export_matlab_ground_truth.m in MATLAB first."
        )
    # Cache as a dict of numpy arrays, transposing from MATLAB (column-major)
    # to NumPy (row-major) conventions.
    import numpy as np
    data = {}
    def _load(obj):
        arr = np.asarray(obj[...])
        if arr.ndim >= 2:
            arr = arr.T
        # MATLAB row/column vectors arrive as (1, N) or (N, 1); squeeze them.
        arr = np.squeeze(arr)
        return arr

    with h5py.File(path, "r") as f:
        for key in f.keys():
            if key.startswith("#"):
                continue
            obj = f[key]
            if isinstance(obj, h5py.Dataset):
                data[key] = _load(obj)
            else:  # group (struct) — SOPHs_flat
                data[key] = {subk: _load(obj[subk]) for subk in obj.keys()}
    return data


@pytest.fixture(scope="session")
def stats_table_matlab():
    """MATLAB-exported stats_table CSV as a pandas DataFrame."""
    import pandas as pd
    path = DATA_CACHE / "segment_stats.csv"
    if not path.exists():
        pytest.skip(
            f"{path} not found. Run scripts/export_matlab_ground_truth.m in MATLAB first."
        )
    return pd.read_csv(path)


@pytest.fixture(scope="session")
def example_data():
    """Load DYNAMO_dev/example_data/example_data.mat — the raw EEG input."""
    import scipy.io as sio
    path = Path("/Users/Mike/code/toolboxes/DYNAMO_dev/example_data/example_data.mat")
    if not path.exists():
        pytest.skip(f"{path} not found.")
    return sio.loadmat(str(path), simplify_cells=True)
