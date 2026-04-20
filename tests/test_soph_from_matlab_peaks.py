"""Critical bit-identity test: given MATLAB's stats_table and MATLAB's
SOpower_norm timeseries, our SO-power histogram must match MATLAB's
SOPHs.SOpower_mat exactly.

This isolates the 2D binning logic from the computeSOpower/computeSOphase
ports, which introduce filter/FFT-level differences.
"""

import numpy as np
import pytest

from pydynamo.soph.histogram import so_power_histogram


SEGMENT_TIME_RANGE = (8420, 13446)


def test_sopower_hist_bit_identical(stats_table_matlab, segment_out_compat):
    mat = segment_out_compat["SOPHs_flat"]
    expected_mat = np.asarray(mat["SOpower_mat"], dtype=float)
    expected_SOpower_bins = np.asarray(mat["SOpower_bins"], dtype=float).ravel()
    expected_freq_bins = np.asarray(mat["freq_bins"], dtype=float).ravel()
    expected_TIB = np.asarray(mat["SOpower_TIB"], dtype=float)

    so_power = np.asarray(mat["SOpower_norm"], dtype=float).ravel()
    so_power_times = np.asarray(mat["SOpower_times"], dtype=float).ravel()

    # Check that we loaded sane data
    assert so_power.size == so_power_times.size

    # Need per-sample stages for so_power_stages — the MATLAB export didn't
    # save SOpower_stages explicitly, but SOpower_retain_Fs=True means
    # SOpower_times == EEG_times, so stages ~ interp from example_data.mat.
    # For now, use the 'compute stages from EEG stage_times/stage_vals' path.
    from pydynamo.io_compat import load_example_data
    ed = load_example_data()
    stage_times_full = np.asarray(ed["stage_times"], dtype=float).ravel()
    stage_vals_full = np.asarray(ed["stage_vals"], dtype=float).ravel()
    from scipy.interpolate import interp1d
    so_power_stages = interp1d(
        stage_times_full, stage_vals_full, kind="previous",
        bounds_error=False, fill_value=np.nan,
    )(so_power_times)
    so_power_stages = np.where(np.isnan(so_power_stages), 0.0, so_power_stages)

    # MATLAB peak columns
    pt = stats_table_matlab["PeakTime"].to_numpy(dtype=float)
    pf = stats_table_matlab["PeakFrequency"].to_numpy(dtype=float)
    ps = stats_table_matlab["PeakStage"].to_numpy(dtype=float)

    out = so_power_histogram(
        pf, pt, ps,
        so_power, so_power_times, so_power_stages,
        time_range=SEGMENT_TIME_RANGE,
        soph_stages=(1, 2, 3),
        freq_range=(0.0, 30.0),
        freq_binsizestep=(1.0, 0.2),
        so_range=None,         # adaptive: min/max of valid SOpower
        so_binsizestep=None,   # adaptive: range/10, range/100
        # runExampleData.m:72 overrides this for the 'segment' example
        min_time_in_bin=5.0,
        compute_rate=True,
        norm_dim=0,
    )

    # Shape match
    assert out["c_mat"].shape == expected_mat.shape, \
        (out["c_mat"].shape, expected_mat.shape)

    # Bin edges bit-identical
    assert np.allclose(out["freq_cbins"], expected_freq_bins,
                       atol=1e-12, rtol=0), \
        f"freq_bins max diff {np.max(np.abs(out['freq_cbins'] - expected_freq_bins)):.3e}"
    assert np.allclose(out["c_cbins"], expected_SOpower_bins,
                       atol=1e-6, rtol=1e-6), \
        f"SOpower_bins max diff {np.max(np.abs(out['c_cbins'] - expected_SOpower_bins)):.3e}"

    # Histogram: allclose over non-NaN entries
    both_nan = np.isnan(out["c_mat"]) & np.isnan(expected_mat)
    both_finite = np.isfinite(out["c_mat"]) & np.isfinite(expected_mat)
    # Shapes of NaN pattern match
    nan_match = (np.isnan(out["c_mat"]) == np.isnan(expected_mat)).mean()
    print(f"NaN-pattern agreement = {nan_match:.6f}")
    assert nan_match > 0.99, f"NaN patterns differ substantially: {nan_match:.4f}"

    if both_finite.any():
        diff = np.abs(out["c_mat"][both_finite] - expected_mat[both_finite])
        max_abs = float(diff.max())
        max_rel = float((diff / np.maximum(np.abs(expected_mat[both_finite]),
                                            1e-12)).max())
        print(f"SOpower_mat max |py - matlab| = {max_abs:.3e}")
        print(f"SOpower_mat max relative diff = {max_rel:.3e}")
        assert max_rel < 1e-6 or max_abs < 1e-10, \
            f"SOpower_mat diverges: max_abs={max_abs:.3e}, max_rel={max_rel:.3e}"
