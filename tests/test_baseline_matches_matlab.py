"""Bisection Step 2: pydynamo compute_baseline must match MATLAB's saved
baseline_segment.mat vector to within FFT-noise tolerance.

MATLAB's saved `baseline_segment.mat:baseline` is the pass-1 (1s window)
baseline — verified by comparing against both pydynamo pass-1 and pass-2
spectrograms (pass-2 has ~2x values, pass-1 matches).

Tolerance budget: max_rel ≤ 1e-3 (spectrogram differs from MATLAB at
~5.5e-5 rel due to RustFFT vs MATLAB FFT; 2nd-percentile picking can
amplify that slightly when consecutive ranks have close values).
"""
from pathlib import Path
import numpy as np
import pytest
import scipy.io as sio
from scipy.interpolate import interp1d

from pydynamo.baseline import compute_baseline
from pydynamo.io_compat import load_example_data, load_segment_out
from pydynamo.spectrogram import mtm_spectrogram


BASELINE_MAT = Path(__file__).parent.parent / "data_cache" / "baseline_segment.mat"


@pytest.mark.skipif(not BASELINE_MAT.exists(), reason="baseline_segment.mat not staged")
def test_pass1_baseline_matches_matlab():
    ed = load_example_data()
    fs = float(ed["Fs"])
    T0, T1 = 8420, 13446
    i0 = int(round(T0 * fs))
    i1 = int(round(T1 * fs))
    data_tr = ed["data"].ravel()[i0 : i1 + 1].astype(np.float64)
    t_tr = np.arange(i0, i1 + 1) / fs

    # Pass-1 spectrogram (1-s window, matches what MATLAB used for the saved baseline)
    spect1, stimes1_rel, _ = mtm_spectrogram(
        data_tr, fs,
        freq_range=(0, 30), taper_params=(2, 3),
        window_params=(1.0, 0.05), dsfreqs=0.1,
    )
    stimes1 = stimes1_rel + t_tr[0]

    # Match MATLAB computeTFPeaks.m:294-297 exclusion:
    #   baseline_exclude = artifacts | stages not in baseline_stages | user_exclude
    d = load_segment_out()
    artifacts = np.asarray(d["artifacts"]).ravel().astype(bool)
    stage_at_data = interp1d(
        ed["stage_times"].ravel(), ed["stage_vals"].ravel(),
        kind="previous", bounds_error=False, fill_value=0.0,
    )(t_tr)
    stage_exclude = ~np.isin(stage_at_data, (1, 2, 3, 4, 5))
    baseline_exclude = artifacts | stage_exclude

    bl_py = compute_baseline(
        spect1, stimes1, t_tr, baseline_exclude, baseline_ptile=2.0
    )
    bl_py = np.asarray(bl_py).ravel()

    bl_mat = np.asarray(
        sio.loadmat(str(BASELINE_MAT), simplify_cells=True)["baseline"]
    ).ravel().astype(np.float64)

    assert bl_py.shape == bl_mat.shape, (bl_py.shape, bl_mat.shape)

    diff = np.abs(bl_py - bl_mat)
    rel = diff / np.maximum(np.abs(bl_mat), 1e-20)
    max_rel = float(rel.max())
    p99_rel = float(np.quantile(rel, 0.99))

    # argmin/argmax of baseline should pick the same freq bin
    assert bl_py.argmax() == bl_mat.argmax()
    assert bl_py.argmin() == bl_mat.argmin()

    # FFT-noise-limited tolerance (our spect diverges from MATLAB at ~5.5e-5
    # rel, percentile on ranked samples amplifies to ~5e-4 here).
    assert max_rel < 1e-3, f"max_rel={max_rel:.3e}"
    assert p99_rel < 1e-3, f"p99_rel={p99_rel:.3e}"
