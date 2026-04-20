"""Render summary figures for the best merge_thresh on the ORIGINAL pipeline
and compare side-by-side to MATLAB. Uses the cached SOPH from the sweep."""

from pathlib import Path
import h5py
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

from pydynamo import run_dynamo
from pydynamo.io_compat import load_example_data

DATA_CACHE = Path(__file__).parent.parent / "data_cache"
BEST_THRESH = 8.0   # from sweep_results.json

ed = load_example_data()
out = run_dynamo(
    ed["data"].ravel(), float(ed["Fs"]),
    ed["stage_times"].ravel(), ed["stage_vals"].ravel(),
    time_range=(8420, 13446), merge_thresh=BEST_THRESH,
    trim_vol=0.8, seg_time=30.0, min_time_in_bin=5.0,
    double_watershed=True, refinement=True,
    plot=True, verbose=True,
)
out.fig.savefig(DATA_CACHE / f"pydynamo_summary_best_thresh_{BEST_THRESH:g}.png", dpi=150)
print(f"wrote pydynamo_summary_best_thresh_{BEST_THRESH:g}.png")
print(f"n_peaks={len(out.stats_table)}, shape={out.SOPHs.SOpower_mat.shape}")
