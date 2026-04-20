"""Profile the pipeline at merge_thresh=8.0 to find hotspots.

Wraps run_dynamo with manual tic/toc around each stage. Reports a MATLAB-style
timings struct.
"""

import time
import numpy as np

from pydynamo.io_compat import load_example_data
from pydynamo.spectrogram import mtm_spectrogram
from pydynamo.baseline import compute_baseline, subtract_baseline
from pydynamo.artifacts import detect_artifacts
from pydynamo.tfpeaks.extract import extract_tfpeaks
from pydynamo.tfpeaks.mask import mask_spectrogram
from pydynamo.tfpeaks.refine import refine_peak_frequency
from pydynamo.soph.histogram import so_power_histogram, so_phase_histogram
from pydynamo.soph.sopower import compute_so_power
from pydynamo.soph.sophase import compute_so_phase
from scipy.interpolate import interp1d


def tic(): return time.time()


def main():
    ed = load_example_data()
    data = ed["data"].ravel()
    fs = float(ed["Fs"])
    stage_times = ed["stage_times"].ravel()
    stage_vals = ed["stage_vals"].ravel()
    T0, T1 = 8420, 13446
    i0 = int(round(T0 * fs)); i1 = int(round(T1 * fs))
    data_tr = data[i0:i1 + 1]
    t_tr = np.arange(i0, i1 + 1) / fs

    timings = {}

    t = tic()
    artifacts = detect_artifacts(data_tr, fs)
    timings["artifact"] = time.time() - t

    t = tic()
    spect1, stimes1_rel, sfreqs = mtm_spectrogram(
        data_tr, fs, freq_range=(0, 30), taper_params=(2, 3),
        window_params=(1.0, 0.05), dsfreqs=0.1,
    )
    timings["spect_pass1"] = time.time() - t
    stimes1 = stimes1_rel + t_tr[0]

    t = tic()
    stage_at_data = interp1d(
        stage_times, stage_vals, kind="previous",
        bounds_error=False, fill_value=0.0)(t_tr)
    stage_exclude = ~np.isin(stage_at_data, (1, 2, 3, 4, 5))
    baseline_exclude = artifacts | stage_exclude
    baseline1 = compute_baseline(spect1, stimes1, t_tr, baseline_exclude,
                                  baseline_ptile=2.0)
    spect1_norm = subtract_baseline(spect1, baseline1)
    timings["baseline_pass1"] = time.time() - t

    t = tic()
    stats1, labels1 = extract_tfpeaks(
        spect1_norm, stimes1, sfreqs,
        seg_time=30.0, return_labels=True,
        downsample=(2, 2), merge_thresh=8.0, trim_vol=0.8,
        dur_min=0.5, dur_max=5.0, bw_min=2.0, bw_max=15.0,
    )
    timings["extract_pass1"] = time.time() - t

    t = tic()
    spect2, stimes2_rel, sfreqs2 = mtm_spectrogram(
        data_tr, fs, freq_range=(0, 30), taper_params=(2, 3),
        window_params=(2.0, 0.05), dsfreqs=0.1,
    )
    timings["spect_pass2"] = time.time() - t
    stimes2 = stimes2_rel + t_tr[0]

    t = tic()
    baseline2 = compute_baseline(spect2, stimes2, t_tr, baseline_exclude,
                                  baseline_ptile=2.0)
    spect2_norm = subtract_baseline(spect2, baseline2)
    spect2_masked = mask_spectrogram(spect2_norm, stimes2, labels1, stimes1)
    timings["baseline_pass2"] = time.time() - t

    t = tic()
    stats = extract_tfpeaks(
        spect2_masked, stimes2, sfreqs2, seg_time=30.0,
        downsample=(2, 2), merge_thresh=8.0, trim_vol=0.8,
        dur_min=1.0, dur_max=5.0, bw_min=1.0, bw_max=15.0,
    )
    timings["extract_pass2"] = time.time() - t

    t = tic()
    stats = refine_peak_frequency(
        stats, data_tr, fs, t=t_tr,
        freq_range=(0.0, 30.0), window_size=4.0, dsfreqs=0.05,
        refine_method="spline_interp", remove_edge_peaks=True,
    )
    timings["refine"] = time.time() - t

    t = tic()
    stats["PeakStage"] = interp1d(
        stage_times, stage_vals, kind="previous",
        bounds_error=False, fill_value=0.0)(stats["PeakTime"])
    timings["peak_stage"] = time.time() - t

    t = tic()
    SOpower_norm, SOpower_times, SOpower_stages, _, _ = compute_so_power(
        data_tr, fs, stage_times=stage_times, stage_vals=stage_vals,
        eeg_times=t_tr, time_range=(T0, T1), isexcluded=artifacts,
        SO_freqrange=(0.3, 1.5), tapers=(5, 9), window_params=(5.0, 0.5),
        SOpower_outlier_threshold=3.0, norm_method="p2shift1234",
        retain_Fs=True,
    )
    timings["peak_sopower"] = time.time() - t

    t = tic()
    SOphase_unwrapped, SOphase_times, SOphase_stages, _ = compute_so_phase(
        data_tr, fs, stage_times=stage_times, stage_vals=stage_vals,
        eeg_times=t_tr, isexcluded=artifacts, SO_freqrange=(0.3, 1.5),
    )
    timings["peak_sophase"] = time.time() - t

    t = tic()
    xp = np.concatenate(([SOpower_times[0] - 1], SOpower_times,
                         [SOpower_times[-1] + 1]))
    fp = np.concatenate(([SOpower_norm[0]], SOpower_norm, [SOpower_norm[-1]]))
    stats["SOpower"] = np.interp(stats["PeakTime"].to_numpy(), xp, fp)
    peak_phase = np.interp(stats["PeakTime"].to_numpy(),
                            SOphase_times, SOphase_unwrapped)
    stats["SOphase"] = (peak_phase + np.pi) % (2 * np.pi) - np.pi
    timings["peak_assign"] = time.time() - t

    t = tic()
    pf = stats["PeakFrequency"].to_numpy()
    pt = stats["PeakTime"].to_numpy()
    ps = stats["PeakStage"].to_numpy()
    sopow = so_power_histogram(
        pf, pt, ps, SOpower_norm, SOpower_times, SOpower_stages,
        time_range=(T0, T1), soph_stages=(1, 2, 3),
        freq_range=(0.0, 30.0), freq_binsizestep=(1.0, 0.2),
        so_range=None, so_binsizestep=None,
        min_time_in_bin=5.0, compute_rate=True, norm_dim=0,
    )
    timings["soph_sopower_hist"] = time.time() - t

    t = tic()
    sopha = so_phase_histogram(
        pf, pt, ps, SOphase_unwrapped, SOphase_times, SOphase_stages,
        time_range=(T0, T1), soph_stages=(1, 2, 3),
        freq_range=(0.0, 30.0), freq_binsizestep=(1.0, 0.2),
        so_range=(-np.pi, np.pi),
        so_binsizestep=(2 * np.pi / 5, 2 * np.pi / 100),
        min_peak_at_freq=1, compute_rate=True, norm_dim=1,
    )
    timings["soph_sophase_hist"] = time.time() - t

    total = sum(timings.values())
    print(f"{'stage':<22} {'sec':>8}   {'pct':>5}")
    print("-" * 40)
    for k, v in sorted(timings.items(), key=lambda kv: -kv[1]):
        print(f"{k:<22} {v:>8.2f}   {100*v/total:>5.1f}%")
    print("-" * 40)
    print(f"{'TOTAL':<22} {total:>8.2f}   100.0%")


if __name__ == "__main__":
    main()
