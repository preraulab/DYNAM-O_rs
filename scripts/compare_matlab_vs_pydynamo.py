from __future__ import annotations
"""Render MATLAB's output and pydynamo's output side-by-side using the
IDENTICAL Python plot code, so any remaining visual difference is purely
data difference (not color scaling or layout).

Writes three files to data_cache/:
  - compare_segment_matlab.png   — MATLAB data rendered via summary_plot
  - compare_segment_pydynamo.png — pydynamo data rendered via summary_plot
  - compare_segment_sidebyside.png — both stitched horizontally
And the same three for `_night` when night data is available.

Both figures use the SAME climscale rules (MATLAB-matching hazen prctile),
so if the colormaps look different it's because the underlying histograms
differ.
"""
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
from PIL import Image

from pydynamo import run_dynamo
from pydynamo.io_compat import load_example_data
from pydynamo.pipeline import SOPHsResult
from pydynamo.plot import summary_plot

DC = Path(__file__).parent.parent / "data_cache"


def _squeeze(x):
    a = np.asarray(x)
    if a.ndim >= 2:
        a = a.T
    return np.squeeze(a)


def _read_mat_field(path: Path, field_path: list[str]):
    """Read a nested field from a .mat file, v5 (scipy) or v7.3 (h5py)."""
    try:
        # v7.3 → HDF5
        with h5py.File(path, "r") as f:
            d = f
            for k in field_path:
                d = d[k]
            return _squeeze(np.asarray(d[...]))
    except (OSError, KeyError):
        # v5 → scipy
        m = sio.loadmat(path, simplify_cells=True)
        v = m
        for k in field_path:
            v = v[k] if isinstance(v, dict) else getattr(v, k)
        return np.asarray(v).squeeze()


def load_matlab_sophs(compat_mat: Path, bisect_mat: Path | None) -> SOPHsResult:
    """Build a SOPHsResult from MATLAB's saved outputs.

    Handles BOTH layouts:
      - segment_out_compat.mat (v7.3, fields under 'SOPHs_flat')
      - night_out.mat (v5/v7.3, fields under 'SOPHs')
    """
    # Detect which root key exists
    root_candidates = ["SOPHs_flat", "SOPHs"]
    found_root = None
    for root in root_candidates:
        try:
            _read_mat_field(compat_mat, [root, "SOpower_mat"])
            found_root = root
            break
        except Exception:
            continue
    if found_root is None:
        raise RuntimeError(f"Could not find SOPHs root in {compat_mat}")
    r = found_root

    SOpower_mat = _read_mat_field(compat_mat, [r, "SOpower_mat"]).astype(float)
    SOphase_mat = _read_mat_field(compat_mat, [r, "SOphase_mat"]).astype(float)
    SOpower_bins = _read_mat_field(compat_mat, [r, "SOpower_bins"]).astype(float)
    SOphase_bins = _read_mat_field(compat_mat, [r, "SOphase_bins"]).astype(float)
    freq_bins = _read_mat_field(compat_mat, [r, "freq_bins"]).astype(float)
    SOpower_TIB = _read_mat_field(compat_mat, [r, "SOpower_TIB"]).astype(float)
    SOphase_TIB = _read_mat_field(compat_mat, [r, "SOphase_TIB"]).astype(float)
    peak_at_freq = _read_mat_field(compat_mat, [r, "num_peaks_at_freq"]).astype(float)
    SOpower_norm = _read_mat_field(compat_mat, [r, "SOpower_norm"]).astype(float)
    SOpower_times = _read_mat_field(compat_mat, [r, "SOpower_times"]).astype(float)

    # SOphase timeseries — in bisect_intermediates_*.mat only
    if bisect_mat is not None and bisect_mat.exists():
        with h5py.File(bisect_mat, "r") as f:
            SOphase = _squeeze(f["SOphase_norm"][...]).astype(float)
            SOphase_times = _squeeze(f["SOphase_times"][...]).astype(float)
    else:
        SOphase = np.zeros_like(SOpower_times)
        SOphase_times = SOpower_times

    return SOPHsResult(
        SOpower_mat=SOpower_mat,
        SOphase_mat=SOphase_mat,
        SOpower_bins=SOpower_bins,
        SOphase_bins=SOphase_bins,
        freq_bins=freq_bins,
        SOpower_TIB=SOpower_TIB,
        SOphase_TIB=SOphase_TIB,
        peak_at_freq_SOpower=peak_at_freq,
        peak_at_freq_SOphase=peak_at_freq,
        SOpower_norm=SOpower_norm,
        SOpower_times=SOpower_times,
        SOphase=SOphase,
        SOphase_times=SOphase_times,
    )


def render_one(tag: str,
               dataset: str,
               time_range: tuple[float, float],
               compat_mat: Path,
               bisect_mat: Path | None,
               stats_csv: Path,
               pydynamo_out,
               data_tr: np.ndarray,
               fs: float,
               artifacts: np.ndarray,
               t_tr: np.ndarray,
               stage_times: np.ndarray,
               stage_vals: np.ndarray):
    """Render one side (either MATLAB or pydynamo) via summary_plot."""
    if tag == "matlab":
        sophs = load_matlab_sophs(compat_mat, bisect_mat)
        if stats_csv.exists():
            stats = pd.read_csv(stats_csv)
            hist_mask = np.isin(stats["PeakStage"].to_numpy(), [1, 2, 3])
        else:
            print(f"  NOTE: {stats_csv} missing; MATLAB scatter will be empty.")
            stats = pd.DataFrame(columns=["PeakTime", "PeakFrequency",
                                          "Volume", "SOphase", "PeakStage"])
            hist_mask = np.zeros(0, dtype=bool)
        spect = pydynamo_out.spect   # use pydynamo's spect for the background
        stimes = pydynamo_out.stimes
        sfreqs = pydynamo_out.sfreqs
    elif tag == "pydynamo":
        sophs = pydynamo_out.SOPHs
        stats = pydynamo_out.stats_table
        # Use pydynamo's peak_selection_inds equivalent: NREM only
        hist_mask = np.isin(stats["PeakStage"].to_numpy(), [1, 2, 3]) \
                    if len(stats) else np.zeros(0, dtype=bool)
        spect = pydynamo_out.spect
        stimes = pydynamo_out.stimes
        sfreqs = pydynamo_out.sfreqs
    else:
        raise ValueError(tag)

    fig = summary_plot(
        spect, stimes, sfreqs, artifacts, t_tr,
        stage_times, stage_vals, stats, sophs,
        data=data_tr, fs=fs, time_range=time_range,
        freq_limits=(2.0, 25.0), mtm_freq_range=(2.0, 25.0),
        hist_peakidx=hist_mask,
    )
    out_path = DC / f"compare_{dataset}_{tag}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Add a banner ABOVE the figure so the existing "EEG Spectrogram" title
    # on the hypnogram axis isn't clobbered.
    img = Image.open(out_path)
    banner_h = 60
    banner = Image.new("RGB", (img.width, img.height + banner_h), "white")
    banner.paste(img, (0, banner_h))
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(banner)
    label = f"{dataset.upper()} — {tag.upper()}"
    try:
        font = ImageFont.truetype(
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf", 32)
    except OSError:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), label, font=font)
    tw = bbox[2] - bbox[0]
    draw.text(((img.width - tw) // 2, 10), label, fill="black", font=font)
    banner.save(out_path, dpi=(150, 150))
    return out_path


def side_by_side(a_path: Path, b_path: Path, out_path: Path):
    """Stitch two PNGs horizontally."""
    a = Image.open(a_path); b = Image.open(b_path)
    # Resize to same height
    h = max(a.height, b.height)
    if a.height != h:
        a = a.resize((int(a.width * h / a.height), h))
    if b.height != h:
        b = b.resize((int(b.width * h / b.height), h))
    w = a.width + b.width
    combined = Image.new("RGB", (w, h), "white")
    combined.paste(a, (0, 0))
    combined.paste(b, (a.width, 0))
    combined.save(out_path, dpi=(150, 150))
    print(f"  wrote {out_path}")


def run_dataset(name: str, time_range: tuple[float, float],
                compat_mat: Path, stats_csv: Path,
                bisect_mat: Path | None,
                min_time_in_bin: float,
                min_peak_at_freq: int):
    if not compat_mat.exists():
        print(f"skipping {name}: {compat_mat} not found"); return
    print(f"\n=== {name} ({time_range[0]}s – {time_range[1]}s) ===")
    ed = load_example_data()
    fs = float(ed["Fs"])
    stage_times = ed["stage_times"].ravel()
    stage_vals = ed["stage_vals"].ravel()

    # Run pydynamo once
    print("running pydynamo...")
    out = run_dynamo(
        ed["data"].ravel(), fs, stage_times, stage_vals,
        time_range=time_range, merge_thresh=11.0, trim_vol=0.8,
        seg_time=30.0, min_time_in_bin=min_time_in_bin,
        min_peak_at_freq=min_peak_at_freq,
        double_watershed=True, refinement=True,
        plot=False, verbose=False,
    )
    # Build data slice for plotting (spectrogram panel uses this)
    i0 = int(round(time_range[0] * fs))
    i1 = int(round(time_range[1] * fs))
    data_tr = ed["data"].ravel()[i0 : i1 + 1].astype(np.float64)
    t_tr = np.arange(i0, i1 + 1) / fs

    # Render both sides
    print("rendering MATLAB side...")
    mat_path = render_one(
        "matlab", name, time_range, compat_mat, bisect_mat, stats_csv,
        out, data_tr, fs, out.artifacts, t_tr, stage_times, stage_vals,
    )
    print("rendering pydynamo side...")
    py_path = render_one(
        "pydynamo", name, time_range, compat_mat, bisect_mat, stats_csv,
        out, data_tr, fs, out.artifacts, t_tr, stage_times, stage_vals,
    )
    combined = DC / f"compare_{name}_sidebyside.png"
    side_by_side(mat_path, py_path, combined)


def main() -> int:
    # Segment: runExampleData.m 'segment' → [8420, 13446]
    run_dataset(
        "segment", (8420.0, 13446.0),
        DC / "segment_out_compat.mat",
        DC / "segment_stats.csv",
        DC / "bisect_intermediates_segment.mat",
        min_time_in_bin=5.0,     # runExampleData.m:72 segment override
        min_peak_at_freq=10,     # runExampleData.m:73 segment override
    )
    # Night: 5-min buffer around first/last non-wake stage
    ed = load_example_data()
    stage_times = ed["stage_times"].ravel()
    stage_vals = ed["stage_vals"].ravel()
    nw = np.flatnonzero((stage_vals < 5) & (stage_vals > 0))
    t0 = float(stage_times[nw[0]]) - 300
    t1 = float(stage_times[nw[-1]]) + 300
    night_compat = DC / "night_out.mat"
    night_stats_csv = DC / "night_stats.csv"
    # night_out.mat has stats_table opaque — need a separate CSV export.
    # If it doesn't exist, fall back to MATLAB computed night data from the
    # scipy-loadable night_out.mat directly.
    if not night_stats_csv.exists():
        # Extract stats from night_out.mat if possible via scipy (stats_table
        # is opaque; this won't give us per-peak). Skip if unavailable —
        # the MATLAB side render will have empty scatter.
        print(f"NOTE: {night_stats_csv} not present; night scatter will be empty.")
    run_dataset(
        "night", (t0, t1),
        night_compat, night_stats_csv,
        DC / "bisect_intermediates_night.mat",
        min_time_in_bin=10.0,   # MATLAB SOpowerphasehist_opts default
        min_peak_at_freq=0,     # MATLAB SOpowerphasehist_opts default (no segment override)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
