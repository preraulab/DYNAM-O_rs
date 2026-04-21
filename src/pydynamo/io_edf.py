"""Thin wrappers around the Rust port of MATLAB's `read_EDF` and `read_staging`.

These mirror the behavior of
  toolbox/helper_functions/read_EDF/read_EDF_mex.c
and
  toolbox/helper_functions/sleep/read_staging.m
"""

from __future__ import annotations

from typing import Any

import numpy as np

import dynamo_rs


def read_edf(path: str, label: str | None = None) -> dict[str, Any]:
    """Read an EDF/EDF+ file.

    Parameters
    ----------
    path
        Absolute file path.
    label
        Optional channel label. May be a direct signal label (case-insensitive,
        whitespace-trimmed) or an 'A-B' rereference request.

    Returns
    -------
    dict
        Keys common to both modes: ``header``, ``labels``, ``sampling_frequencies``.
        If ``label`` was given: ``data`` (1-D float64 array of physical units),
        ``fs`` (Hz), ``label``, ``signal_header``.
        Otherwise: ``signals`` (list of per-signal dicts each with ``data`` plus
        per-signal header fields).
    """
    return dynamo_rs.read_edf(path, label)


def read_staging(
    path: str,
    time_col: int = 1,
    stage_col: int = 2,
    header_lines: int = 0,
    delimiter: str = ",",
    epoch_dur: float = 30.0,
    start_time: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Read a staging CSV/TSV.

    Columns are 1-based to match the MATLAB API. Stage-value codes:

    ====  ==========================================
    code  strings
    ====  ==========================================
    6     art / artifact / A / '6'
    5     wake / W / '5'
    4     REM / R / '4'
    3     N1 / 'Stage 1' / '1'
    2     N2 / 'Stage 2' / '2'
    1     N3 / 'Stage 3' / '3'
    0     Unk / U / Unknown / '0'
    ====  ==========================================

    Time-column heuristics:
      - Consecutive integer epoch numbers → times = values * ``epoch_dur``.
      - Numeric seconds (other) → times used directly.
      - Time strings (with or without AM/PM) → seconds-of-day, wrapping past
        midnight; if ``start_time`` is provided it anchors the zero point.

    Returns
    -------
    (times, vals) : (ndarray, ndarray) of float64
    """
    times, vals = dynamo_rs.read_staging(
        path,
        time_col,
        stage_col,
        header_lines,
        delimiter,
        float(epoch_dur),
        start_time,
    )
    return np.asarray(times, dtype=np.float64), np.asarray(vals, dtype=np.float64)
