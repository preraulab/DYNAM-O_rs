"""Baseline: compare pydynamo baseline to MATLAB.

We don't have the raw MATLAB baseline vector saved, but we can verify the
round-trip: given the MATLAB spectrogram + MATLAB artifacts, computing our
baseline and subtracting should produce a non-negative, finite result with
the same shape.

The actual DYNAM-O pass-2 baseline is reflected in the `spect` saved in
segment_out.mat being already baseline-subtracted? — check: MATLAB saves the
post-baseline-subtracted spect when running through runDYNAMO. Inspect the
min: if all-positive it's been divided by baseline already.
"""

import numpy as np

from pydynamo.baseline import compute_baseline, subtract_baseline


def test_baseline_percentile_compute(segment_out_compat):
    spect_mat = np.asarray(segment_out_compat["spect"], dtype=np.float64)
    stimes = np.asarray(segment_out_compat["stimes"], dtype=np.float64)
    artifacts = np.asarray(segment_out_compat["artifacts"], dtype=bool).ravel()
    t_data = np.asarray(segment_out_compat["t_time_range"], dtype=np.float64).ravel()

    baseline = compute_baseline(
        spect_mat, stimes, t_data, baseline_exclude=artifacts,
        baseline_range=(float("-inf"), float("inf")),
        baseline_ptile=2.0,
    )
    assert baseline.shape == (spect_mat.shape[0], 1)
    # Baselines must be > 0 for positive-power spectrogram
    assert np.all(baseline > 0), "baseline should be strictly positive"
    assert np.all(np.isfinite(baseline))


def test_subtract_baseline_roundtrip(segment_out_compat):
    """Subtracting the 2nd-percentile baseline from its own spectrogram should
    give a spectrogram whose per-frequency 2nd percentile ≈ 1 (since we
    divided out the 2nd percentile)."""
    spect_mat = np.asarray(segment_out_compat["spect"], dtype=np.float64)
    stimes = np.asarray(segment_out_compat["stimes"], dtype=np.float64)
    artifacts = np.asarray(segment_out_compat["artifacts"], dtype=bool).ravel()
    t_data = np.asarray(segment_out_compat["t_time_range"], dtype=np.float64).ravel()

    baseline = compute_baseline(
        spect_mat, stimes, t_data, artifacts, baseline_ptile=2.0
    )
    spect_norm = subtract_baseline(spect_mat, baseline)
    # Drop excluded columns for the check
    idx = np.searchsorted(t_data, stimes)
    idx = np.clip(idx, 0, t_data.size - 1)
    valid = ~artifacts[idx]
    ptile = np.nanpercentile(np.where(spect_norm[:, valid] == 0, np.nan,
                                      spect_norm[:, valid]),
                             2.0, axis=1)
    # Every row's 2nd percentile should be very near 1.0 after this.
    assert np.allclose(ptile, 1.0, rtol=1e-6, atol=1e-6), \
        f"per-row ptile off, max diff {np.max(np.abs(ptile - 1.0))}"
