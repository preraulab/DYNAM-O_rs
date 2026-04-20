"""Side-by-side: pyDYNAM-O's detect_tfpeaks vs my pydynamo pass-1 extract
on the same 300-s segment. Share the exact same baseline-subtracted
spectrogram so the only variable is the per-segment peak algorithm.
"""

import sys, time, warnings
import skimage.graph, skimage.future
sys.modules['skimage.future.graph'] = skimage.graph
skimage.future.graph = skimage.graph
sys.path.insert(0, '/Users/Mike/code/toolboxes/pyDYNAM-O')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pydynamo.io_compat import load_example_data
from pydynamo.spectrogram import mtm_spectrogram
from pydynamo.tfpeaks.extract import extract_tfpeaks
from pydynamo.plot import _rainbow4

from dynam_o.TFpeaks import detect_tfpeaks, process_segments_params
from dynam_o.utils import min_prominence

warnings.filterwarnings("ignore")

ed = load_example_data()
fs = float(ed['Fs'])
data = ed['data'].ravel()

# 300-s slice starting at 8420 s
T0 = 8420
dur_s = 300
i0 = int(round(T0 * fs))
i1 = i0 + int(dur_s * fs)
sub = data[i0:i1 + 1].astype(np.float64)
print(f"slice: {sub.size} samples ({dur_s}s)")

# Shared spectrogram (1 s window, pyDYNAM-O convention)
spect, stimes_rel, sfreqs = mtm_spectrogram(
    sub, fs, freq_range=(0, 30), taper_params=(2, 3),
    window_params=(1.0, 0.05), dsfreqs=0.1,
)
d_time = float(stimes_rel[1] - stimes_rel[0])
d_freq = float(sfreqs[1] - sfreqs[0])
df_mtm = 2 / 1.0 * 2            # 2*TW/windowlen — pyDYNAM-O formula (line 61)

# Shared baseline
baseline = np.percentile(spect, 2, axis=1, keepdims=True)
spect_norm = spect / baseline

# ----- pyDYNAM-O: segment the spectrogram and run detect_tfpeaks on each -----
window_idxs, start_times = process_segments_params(30.0, stimes_rel)
print(f"\npyDYNAM-O: {len(window_idxs)} segments")

dp_params = dict(
    d_time=d_time, d_freq=d_freq,
    merge_thresh=8, max_merges=np.inf, trim_volume=0.8,
    downsample=[],
    dur_min=1.0 / 2,        # window_params[0] / 2
    dur_max=5.0,
    bw_min=df_mtm / 2,
    bw_max=15.0,
    prom_min=min_prominence(3, 0.95),
    plot_on=False, verbose=False,
)

tic = time.time()
tables_py = []
for idxs, start_t in zip(window_idxs, start_times):
    tbl = detect_tfpeaks(spect_norm[:, idxs], float(start_t), **dp_params)
    tables_py.append(tbl)
stats_py = pd.concat(tables_py, ignore_index=True) if tables_py else pd.DataFrame()
print(f"  {len(stats_py)} peaks in {time.time()-tic:.1f}s")

# ----- pydynamo pass-1 on the same spect_norm -----
tic = time.time()
stats_me = extract_tfpeaks(
    spect_norm, stimes_rel, sfreqs,
    seg_time=30.0, n_jobs=1,
    downsample=(2, 2), merge_thresh=8.0, trim_vol=0.8,
    dur_min=0.5, dur_max=5.0, bw_min=2.0, bw_max=15.0,
)
print(f"pydynamo pass-1: {len(stats_me)} peaks in {time.time()-tic:.1f}s")

# ----- Plot -----
with np.errstate(divide="ignore", invalid="ignore"):
    spect_db = 10 * np.log10(np.where(spect_norm > 0, spect_norm, np.nan))
vmin = np.nanpercentile(spect_db, 5)
vmax = np.nanpercentile(spect_db, 98)
extent = (stimes_rel[0], stimes_rel[-1], sfreqs[0], sfreqs[-1])

fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True, sharey=True,
                          constrained_layout=True)
for ax, title, stats, tcol, fcol in [
    (axes[0], f"pyDYNAM-O ({len(stats_py)} peaks)", stats_py, 'peak_time', 'peak_frequency'),
    (axes[1], f"pydynamo pass-1 ({len(stats_me)} peaks)", stats_me, 'PeakTime', 'PeakFrequency'),
]:
    ax.imshow(spect_db, origin="lower", aspect="auto", extent=extent,
              cmap=_rainbow4, vmin=vmin, vmax=vmax)
    if len(stats):
        ax.scatter(stats[tcol], stats[fcol], s=10, c="white",
                    edgecolors="k", linewidths=0.4, alpha=0.9)
    ax.set_title(title, fontsize=13)
    ax.set_ylabel("Frequency (Hz)")
    ax.set_ylim(0, 30)
axes[-1].set_xlabel("Time (s) from slice start")
fig.suptitle(f"{dur_s} s slice of bundled example, same baseline-subtracted spectrogram",
             fontsize=11)
out_path = "data_cache/compare_pydynam_o_vs_pydynamo.png"
fig.savefig(out_path, dpi=130)
print(f"\nwrote {out_path}")

print("\nPer-frequency-band peak counts:")
print(f"  {'band':>12} | {'pyDYNAM-O':>10} | {'pydynamo':>10}")
for lo, hi in [(0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 30)]:
    n_py = ((stats_py['peak_frequency'] >= lo) & (stats_py['peak_frequency'] < hi)).sum() if len(stats_py) else 0
    n_me = ((stats_me['PeakFrequency'] >= lo) & (stats_me['PeakFrequency'] < hi)).sum() if len(stats_me) else 0
    print(f"  [{lo:>2}-{hi:>2} Hz]   | {n_py:>10} | {n_me:>10}")
