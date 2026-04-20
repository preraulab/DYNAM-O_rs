"""Helpers for loading MATLAB ground-truth data for validation.

`segment_out.mat` top-level contains:
    spect              (308, 100481) float32
    stimes             (100481,)     float64
    sfreqs             (308,)        float64
    artifacts          (502601,)     uint8
    data_time_range    (502601,)     float32
    t_time_range       (502601,)     float64
    SOPHs              struct  (see below)
    timings            struct
    stats_table        MATLAB table  -- OPAQUE to scipy.io; must be exported
                                        to CSV first via scripts/export_matlab_ground_truth.m

SOPHs fields:
    SOpower_mat (101,151) float64
    SOphase_mat (101,151) float64
    SOpower_bins (101,)   float64
    SOphase_bins (101,)   float64
    freq_bins    (151,)   float64
    num_peaks_at_freq (151,) uint16
    SOpower_TIB (101,5)   float64
    SOphase_TIB (101,5)   float64
    SOpower_norm  (502601,) float32
    SOpower_times (502601,) float64
    ... (paramfit / splinefit skipped)
"""

from pathlib import Path
from typing import Any

import numpy as np
import scipy.io as sio

DEFAULT_SEGMENT_OUT = Path(
    "/Users/Mike/code/toolboxes/DYNAMO_dev/segment_out.mat"
)
DEFAULT_EXAMPLE_DATA = Path(
    "/Users/Mike/code/toolboxes/DYNAMO_dev/example_data/example_data.mat"
)


def load_segment_out(path: Path = DEFAULT_SEGMENT_OUT) -> dict[str, Any]:
    """Load segment_out.mat, dropping the opaque stats_table entry."""
    raw = sio.loadmat(str(path), simplify_cells=True)
    out = {k: v for k, v in raw.items() if not k.startswith("__")}
    # stats_table comes through as MatlabOpaque with the key being a literal
    # string "None" under simplify_cells — drop it.
    out.pop("None", None)
    return out


def load_example_data(path: Path = DEFAULT_EXAMPLE_DATA) -> dict[str, Any]:
    """Load the bundled example EEG (data, Fs, stage_times, stage_vals)."""
    raw = sio.loadmat(str(path), simplify_cells=True)
    return {k: v for k, v in raw.items() if not k.startswith("__")}


def load_stats_csv(path: Path) -> "Any":
    """Load the MATLAB-exported stats_table CSV as a pandas DataFrame."""
    import pandas as pd  # lazy import so tests can skip if pandas absent
    return pd.read_csv(path)
