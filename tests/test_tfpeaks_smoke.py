"""Smoke test: TF-peak extraction on a small segment of the real
baseline-subtracted MATLAB spectrogram. Not a bit-identity test; just
checks that the pipeline runs and produces plausible peaks."""

import numpy as np

from pydynamo.baseline import compute_baseline, subtract_baseline
from pydynamo.tfpeaks.extract import extract_tfpeaks_segment


def test_extract_single_segment(segment_out_compat):
    spect = np.asarray(segment_out_compat["spect"], dtype=np.float64)
    stimes = np.asarray(segment_out_compat["stimes"], dtype=np.float64)
    sfreqs = np.asarray(segment_out_compat["sfreqs"], dtype=np.float64)
    artifacts = np.asarray(segment_out_compat["artifacts"], dtype=bool).ravel()
    t_data = np.asarray(segment_out_compat["t_time_range"], dtype=np.float64).ravel()

    baseline = compute_baseline(spect, stimes, t_data, artifacts, baseline_ptile=2.0)
    spect_norm = subtract_baseline(spect, baseline)

    # One 30-second segment
    dt = float(stimes[1] - stimes[0])
    n = int(round(30.0 / dt))
    seg_spect = spect_norm[:, :n]
    seg_stimes = stimes[:n]

    peaks = extract_tfpeaks_segment(
        seg_spect, seg_stimes, sfreqs,
        segment_num=1,
        downsample=(2, 2),
        merge_thresh=8.0,    # pyDYNAM-O default — reproduces MATLAB behaviour
        trim_vol=0.8,
        dur_max=5.0,
        bw_max=15.0,
    )
    print(f"peaks in first 30s: {len(peaks)}")
    # MATLAB's first segment has ~33 peaks; expect within ~2x.
    assert 5 < len(peaks) < 100, f"unexpected peak count {len(peaks)}"

    # Required columns present
    expected = {"PeakTime", "PeakFrequency", "Height", "Duration", "Bandwidth",
                "Volume", "SegmentNum"}
    missing = expected - set(peaks.columns)
    assert not missing, f"missing columns: {missing}"

    # Values in plausible ranges
    assert peaks["PeakTime"].between(seg_stimes[0], seg_stimes[-1]).all()
    assert peaks["PeakFrequency"].between(sfreqs.min(), sfreqs.max()).all()
    assert (peaks["Duration"] <= 5.0).all()
    assert (peaks["Bandwidth"] <= 15.0).all()
