"""SO-power timeseries: compare our computeSOpower port to MATLAB's
saved SOpower_norm.

Can't be bit-identical due to the same FFT-library floor that affects
the multitaper spectrogram (see test_spectrogram.py), but we expect
relative diffs similar to the spectrogram (< 1e-3 for 99th pct) once
the pipeline matches.
"""

import numpy as np
import pytest

from pydynamo.soph.sopower import compute_so_power


SEGMENT_TIME_RANGE = (8420, 13446)


def test_sopower_timeseries_close_to_matlab(example_data, segment_out_compat):
    fs = float(example_data["Fs"])
    data = np.asarray(example_data["data"]).ravel()
    i0 = int(round(SEGMENT_TIME_RANGE[0] * fs))
    i1 = int(round(SEGMENT_TIME_RANGE[1] * fs))
    segment = data[i0 : i1 + 1]
    eeg_times = np.arange(i0, i1 + 1) / fs

    stage_times = np.asarray(example_data["stage_times"]).ravel()
    stage_vals = np.asarray(example_data["stage_vals"]).ravel()

    # Artifacts: use MATLAB's artifact mask for an apples-to-apples compare,
    # isolating the SO-power computation from the artifact-detection port.
    artifacts_mat = np.asarray(segment_out_compat["artifacts"], dtype=bool).ravel()

    so_power_py, so_times_py, _, _, _ = compute_so_power(
        segment, fs,
        stage_times=stage_times, stage_vals=stage_vals,
        eeg_times=eeg_times,
        time_range=SEGMENT_TIME_RANGE,
        isexcluded=artifacts_mat,
        SO_freqrange=(0.3, 1.5),
        tapers=(5, 9),
        window_params=(5.0, 0.5),
        SOpower_outlier_threshold=3.0,
        norm_method="p2shift1234",
        retain_Fs=True,
    )
    so_power_mat = np.asarray(
        segment_out_compat["SOPHs_flat"]["SOpower_norm"], dtype=float
    ).ravel()
    assert so_power_py.shape == so_power_mat.shape

    # NaN agreement
    nan_match = (np.isnan(so_power_py) == np.isnan(so_power_mat)).mean()
    print(f"NaN agreement: {nan_match:.6f}")

    both_finite = np.isfinite(so_power_py) & np.isfinite(so_power_mat)
    if both_finite.any():
        diff = np.abs(so_power_py[both_finite] - so_power_mat[both_finite])
        denom = np.maximum(np.abs(so_power_mat[both_finite]), 1.0)
        p99 = float(np.quantile(diff / denom, 0.99))
        mx = float(diff.max())
        mx_rel = float((diff / denom).max())
        print(f"max abs diff  = {mx:.3e}")
        print(f"max rel diff  = {mx_rel:.3e}")
        print(f"99th-pct rel  = {p99:.3e}")
        # SO-power is in dB after scaling; diffs of ~0.01 dB are expected.
        assert mx < 0.5, f"max abs diff too large: {mx:.3e}"
        assert p99 < 0.1, f"99th-pct rel diff too large: {p99:.3e}"
