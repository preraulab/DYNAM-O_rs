"""Smoke test: package imports and ground-truth fixtures load."""

import pytest


def test_pydynamo_imports():
    import pydynamo
    assert pydynamo.__version__ == "0.1.0"
    from pydynamo import run_dynamo, DetectionOpts, BaselineOpts, SOPHOpts
    assert callable(run_dynamo)
    # DetectionOpts() should construct with defaults matching MATLAB 'default' preset.
    det = DetectionOpts()
    assert det.merge_thresh == 11.0
    assert det.trim_vol == 0.8
    assert det.seg_time == 30.0
    soph = SOPHOpts()
    assert soph.SOPH_stages == (1, 2, 3)
    assert soph.SO_freqrange == (0.3, 1.5)
    base = BaselineOpts()
    assert base.baseline_ptile == 2.0


def test_segment_out_fixture_shape(segment_out_compat):
    """Verify the MATLAB ground-truth export has expected shapes."""
    assert segment_out_compat["spect"].shape == (308, 100481)
    assert segment_out_compat["artifacts"].shape == (502601,)
    sophs = segment_out_compat["SOPHs_flat"]
    assert sophs["SOpower_mat"].shape == (101, 151)
    assert sophs["SOphase_mat"].shape == (101, 151)


def test_stats_table_fixture(stats_table_matlab):
    """Verify exported stats_table CSV has the expected columns."""
    required = {"PeakTime", "PeakFrequency", "PeakStage", "SOpower", "SOphase"}
    missing = required - set(stats_table_matlab.columns)
    assert not missing, f"stats_table CSV missing columns: {missing}"
    assert len(stats_table_matlab) > 0
