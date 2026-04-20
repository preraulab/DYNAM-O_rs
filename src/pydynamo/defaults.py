"""Frozen default option sets mirroring the DYNAM-O 'default' preset.

Sourced from:
- toolbox/TFpeak_functions/option_sets/detection_opts.m
- toolbox/TFpeak_functions/option_sets/baseline_opts.m
- toolbox/SOpowphase_functions/SOpowerphasehist_opts.m

Only the 'default' preset is modelled here. No 'stokes_2023' or 'precision'.
"""

from dataclasses import dataclass, field
from typing import Sequence


@dataclass(frozen=True)
class DetectionOpts:
    # Multitaper spectrogram
    mtm_freq_range: tuple = (0.0, 30.0)
    mtm_taper_params: tuple = (2, 3)           # (time-BW, num_tapers)
    mtm_window_length_1: float = 1.0            # pass 1 (temporal resolution)
    mtm_window_length_2: float = 2.0            # pass 2 (spectral resolution)
    mtm_window_stepsize: float = 0.05
    mtm_dsfreqs: float = 0.1

    # Watershed + merge + trim
    double_watershed: bool = True
    downsample_spect: tuple = (2, 2)            # (time_factor, freq_factor)
    seg_time: float = 30.0                      # segment duration (s)
    merge_thresh: float = 11.0
    max_merges: float = float("inf")
    trim_vol: float = 0.8

    # Peak filters
    dur_max: float = 5.0
    bw_max: float = 15.0

    # Refinement
    refinement: bool = True                     # 1 Hz Hann window peak-freq refine


@dataclass(frozen=True)
class BaselineOpts:
    baseline_stages: tuple = (1, 2, 3, 4, 5)
    baseline_exclude: Sequence[float] = field(default_factory=tuple)
    baseline_ptile: float = 2.0
    baseline_trim: tuple = (float("-inf"), float("inf"))


@dataclass(frozen=True)
class SOPHOpts:
    # Frequency axis (applies to both SO-power and SO-phase histograms)
    freq_range: tuple = (0.0, 30.0)
    freq_binsizestep: tuple = (1.0, 0.2)        # (bin size, step) in Hz

    # Slow-oscillation band
    SO_freqrange: tuple = (0.3, 1.5)

    # SO-power computation
    SOpower_norm_method: str = "p2shift1234"
    SOpower_binsizestep: tuple = (0.2, 0.01)    # (bin size, step); MATLAB adaptive default
    SOpower_min_time_in_bin: float = 10.0       # minutes
    SOpower_outlier_threshold: float = 3.0
    SOpower_retain_Fs: bool = True
    SOpower_tapers: tuple = (5, 9)
    SOpower_window_params: tuple = (5.0, 0.5)   # (window length s, step s)

    # SO-phase histogram
    SOphase_binsizestep: tuple = (1.2566370614359172, 0.06283185307179587)  # (2pi/5, 2pi/100)
    SOphase_min_peak_at_freq: int = 1
    SOphase_norm_dim: int = 1

    # Stages included in histograms (DYNAM-O convention: 1=N3 ... 5=Wake)
    SOPH_stages: tuple = (1, 2, 3)              # NREM only

    compute_rate: bool = True                   # peaks/min (not raw count)
