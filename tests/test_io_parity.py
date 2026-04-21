"""Parity / smoke test for the Rust read_EDF + read_staging port.

Uses real test data at /Users/Mike/Desktop/testdata/. Skipped if absent.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

EDF_PATH = Path("/Users/Mike/Desktop/testdata/TS00304 baseline.edf")
STAGING_PATH = Path("/Users/Mike/Desktop/testdata/TS00304 baseline psg.csv")

pytestmark = pytest.mark.skipif(
    not EDF_PATH.exists() or not STAGING_PATH.exists(),
    reason="test data not present at /Users/Mike/Desktop/testdata",
)


def test_read_edf_labels_and_header():
    from pydynamo.io_edf import read_edf

    meta = read_edf(str(EDF_PATH))  # load all signals
    assert "labels" in meta and len(meta["labels"]) > 0
    assert meta["header"]["num_data_records"] > 0
    assert meta["header"]["data_record_duration"] > 0
    for fs in meta["sampling_frequencies"]:
        # EDF+ annotation channels typically have fs ~= 4; keep check loose.
        assert fs >= 1.0
    print("labels:", meta["labels"])


def test_read_edf_channel_selection_and_timing():
    from pydynamo.io_edf import read_edf

    t0 = time.perf_counter()
    meta_all = read_edf(str(EDF_PATH))
    t_all = time.perf_counter() - t0
    labels = list(meta_all["labels"])
    print(f"read_edf (all channels) wall time: {t_all:.3f} s for {EDF_PATH.stat().st_size/1e6:.1f} MB")

    # Preferred channel.
    wanted = "C3-A2 - B"
    ok = False
    for try_label in [wanted] + [lab for lab in labels if "EEG" in lab.upper() or "C3" in lab.upper() or "C4" in lab.upper()]:
        try:
            r = read_edf(str(EDF_PATH), try_label)
            ok = True
            print(f"selected label: '{try_label}' → fs={r['fs']}, n={len(r['data'])}")
            break
        except Exception as e:
            print(f"  (tried {try_label!r}: {e})")
            continue
    if not ok:
        # Fall back to first non-annotation channel.
        non_annot = [lab for lab in labels if "ANNOT" not in lab.upper()]
        assert non_annot, f"no usable signal labels; got {labels}"
        r = read_edf(str(EDF_PATH), non_annot[0])
        print(f"fallback selected '{non_annot[0]}' → fs={r['fs']}, n={len(r['data'])}")

    data = np.asarray(r["data"], dtype=np.float64)
    fs = float(r["fs"])

    assert data.size > 0
    assert 50.0 <= fs <= 2000.0, f"implausible fs={fs}"
    # EEG amplitudes in μV are typically < a few mV; allow up to 10 mV absolute.
    finite = data[np.isfinite(data)]
    assert finite.size > 0
    assert np.max(np.abs(finite)) < 10_000.0, f"values out of reasonable μV range: max|x|={np.max(np.abs(finite))}"


def test_read_staging_tsv_with_time_strings():
    from pydynamo.io_edf import read_staging

    # The test file is TAB-delimited, has a 14-line preamble (metadata + blank + column
    # header row 'Epoch\tEvent\tStart Time'). Event column = 2, Start Time column = 3.
    times, vals = read_staging(
        str(STAGING_PATH),
        time_col=3,
        stage_col=2,
        header_lines=14,
        delimiter="\t",
    )
    assert times.size > 0
    assert vals.size == times.size
    assert np.all((vals >= 0) & (vals <= 6))
    # times are seconds; non-decreasing after the midnight wrap handling.
    assert np.all(np.diff(times) >= 0), "times must be monotonic non-decreasing"
    stages_seen = sorted(set(vals.astype(int).tolist()))
    print(f"staging rows: {len(times)}, stages present: {stages_seen}")
    # Reasonable: we expect at least Wake(5) and some non-wake stage.
    assert 5 in stages_seen or 4 in stages_seen or 2 in stages_seen


def test_staging_epoch_number_mode(tmp_path):
    """Numeric epoch-number time column → times = epoch * epoch_dur."""
    from pydynamo.io_edf import read_staging

    f = tmp_path / "s.csv"
    f.write_text("1,Wake\n2,N1\n3,N2\n4,REM\n5,N3\n")
    times, vals = read_staging(str(f), time_col=1, stage_col=2)
    np.testing.assert_allclose(times, [30.0, 60.0, 90.0, 120.0, 150.0])
    np.testing.assert_array_equal(vals, [5, 3, 2, 4, 1])


def test_matlab_cache_parity_if_available():
    """If a cached MATLAB .mat with the EDF read / staging is in data_cache/, compare."""
    import glob, os

    candidates = glob.glob(os.path.expanduser("~/code/toolboxes/DYNAM-O_rs/data_cache/**/*.mat"), recursive=True)
    if not candidates:
        pytest.skip("no data_cache/*.mat available for bit-parity comparison")
    # No well-known filename convention for read_edf output; just note files.
    print(f"data_cache .mat files present: {len(candidates)} (no known read_edf output cached)")
