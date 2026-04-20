"""Full pipeline smoke test on the bundled DYNAM-O segment data.

Runs run_dynamo on the `runDYNAMO('segment')` slice, checks outputs exist
and are plausibly shaped, writes the summary figure as PNG so it can be
eyeballed against the MATLAB output.
"""

from pathlib import Path

import numpy as np
import pytest

from pydynamo import run_dynamo


SEGMENT_TIME_RANGE = (8420, 13446)


def test_end_to_end_segment(example_data, tmp_path):
    fs = float(example_data["Fs"])
    data = np.asarray(example_data["data"]).ravel()
    stage_times = np.asarray(example_data["stage_times"]).ravel()
    stage_vals = np.asarray(example_data["stage_vals"]).ravel()

    out = run_dynamo(
        data, fs, stage_times, stage_vals,
        time_range=SEGMENT_TIME_RANGE,
        merge_thresh=0.0,
        plot=True,
        verbose=False,
    )

    # Outputs
    assert out.stats_table is not None
    assert out.SOPHs.SOpower_mat.shape == (101, 151) or \
           out.SOPHs.SOpower_mat.shape[1] == 151, \
           f"unexpected SOpower_mat shape {out.SOPHs.SOpower_mat.shape}"
    assert out.SOPHs.SOphase_mat.shape[1] == 151
    assert out.SOPHs.SOphase_bins.size == 101
    assert out.SOPHs.freq_bins.size == 151

    # Peak count — MATLAB has 5738 in this segment; ours will differ but
    # should be in a reasonable range.
    print(f"\npeak count: {len(out.stats_table)}  (MATLAB: ~5738)")
    assert 500 < len(out.stats_table) < 20000, \
        f"peak count {len(out.stats_table)} out of ballpark"

    # Columns
    required = {"PeakTime", "PeakFrequency", "PeakStage", "SOpower",
                "SOphase", "Duration", "Bandwidth", "Volume"}
    assert required.issubset(set(out.stats_table.columns))

    # Save the figure so a human can visually compare to MATLAB's output.
    png_path = tmp_path / "pydynamo_summary.png"
    out.fig.savefig(png_path, dpi=120)
    print(f"summary figure written to {png_path}")
    assert png_path.exists()
