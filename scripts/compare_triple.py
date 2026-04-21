"""Triple-panel comparison figure: Rust vs Python vs MATLAB.

For a given dataset (segment / night), renders three SOpower-histogram
summary plots side-by-side through the IDENTICAL summary_plot code so any
visual difference is pure data difference. Titles include the wall-clock
time each backend took.

  - RUST:   run_dynamo with the Rust-accelerated hot paths (default build)
  - PYTHON: run_dynamo with every Rust fast path disabled at import time
  - MATLAB: loaded from data_cache/{name}_out_compat.mat + {name}_stats.csv

Usage:
    python scripts/compare_triple.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import h5py
import scipy.io as sio
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

DC = Path(__file__).parent.parent / "data_cache"


def _sq(x):
    a = np.asarray(x)
    if a.ndim >= 2:
        a = a.T
    return np.squeeze(a)


def _read_mat_field(path: Path, field_path: list[str]):
    try:
        with h5py.File(path, "r") as f:
            d = f
            for k in field_path:
                d = d[k]
            return _sq(np.asarray(d[...]))
    except (OSError, KeyError):
        m = sio.loadmat(path, simplify_cells=True)
        v = m
        for k in field_path:
            v = v[k] if isinstance(v, dict) else getattr(v, k)
        return np.asarray(v).squeeze()


def _force_python_only():
    """Disable every Rust fast path in the currently-loaded pydynamo."""
    flips = []
    for modname in (
        "pydynamo.baseline",
        "pydynamo.tfpeaks.mask",
        "pydynamo.soph.histogram",
        "pydynamo.tfpeaks.refine",
        "pydynamo.soph.sophase",
        "pydynamo.artifacts",
    ):
        mod = sys.modules.get(modname)
        if mod is None:
            continue
        for attr in (
            "_HAS_RUST",
            "_HAS_RUST_SIGNAL",
            # per-function gates introduced after the rustfft-NEON bench
            "_USE_RUST_MOVMEAN",
            "_USE_RUST_UNWRAP",
            "_USE_RUST_SOSFILTFILT",
            "_USE_RUST_HILBERT",
        ):
            if hasattr(mod, attr):
                prev = getattr(mod, attr)
                setattr(mod, attr, False)
                flips.append((mod, attr, prev))
    # Additionally, disable the dynamo_rs-based merge/trim/watershed inside
    # tfpeaks.extract. extract.py imports dynamo_rs at top — we can monkey-
    # patch those to use the sk-image fallback.
    try:
        import pydynamo.tfpeaks.extract as _ex
        if getattr(_ex, "_HAS_RUST_WS", False):
            _ex._HAS_RUST_WS = False
            flips.append((_ex, "_HAS_RUST_WS", True))
    except ImportError:
        pass
    try:
        import pydynamo.tfpeaks.merge as _mg
        if getattr(_mg, "_HAS_RUST_MERGE", False):
            _mg._HAS_RUST_MERGE = False
            flips.append((_mg, "_HAS_RUST_MERGE", True))
    except ImportError:
        pass
    return flips


def _restore(flips):
    for mod, attr, prev in flips:
        setattr(mod, attr, prev)


def _run(backend: str, ed, fs, time_range, min_tib, min_paf):
    # Fresh import each time so caches are consistent.
    from pydynamo import run_dynamo
    if backend == "python":
        flips = _force_python_only()
    else:
        flips = []
    try:
        t0 = time.time()
        out = run_dynamo(
            ed["data"].ravel(), fs,
            ed["stage_times"].ravel(), ed["stage_vals"].ravel(),
            time_range=time_range,
            merge_thresh=11.0, trim_vol=0.8, seg_time=30.0,
            min_time_in_bin=min_tib, min_peak_at_freq=min_paf,
            double_watershed=True, refinement=True,
            plot=False, verbose=False,
        )
        dt = time.time() - t0
    finally:
        _restore(flips)
    return out, dt


def _build_matlab_view(name: str, compat_mat: Path, stats_csv: Path, py_out):
    """Synthesize a summary_plot-ready view of MATLAB's outputs, reusing
    pydynamo's own spectrogram / artifacts / stages for the background
    panels so the only difference is the peak scatter + histograms."""
    from pydynamo.pipeline import SOPHsResult

    # Detect root key
    root = None
    for r in ("SOPHs_flat", "SOPHs"):
        try:
            _read_mat_field(compat_mat, [r, "SOpower_mat"])
            root = r
            break
        except Exception:
            continue
    if root is None:
        raise RuntimeError(f"no SOPHs root in {compat_mat}")

    sophs = SOPHsResult(
        SOpower_mat=_read_mat_field(compat_mat, [root, "SOpower_mat"]).astype(float),
        SOphase_mat=_read_mat_field(compat_mat, [root, "SOphase_mat"]).astype(float),
        SOpower_bins=_read_mat_field(compat_mat, [root, "SOpower_bins"]).astype(float),
        SOphase_bins=_read_mat_field(compat_mat, [root, "SOphase_bins"]).astype(float),
        freq_bins=_read_mat_field(compat_mat, [root, "freq_bins"]).astype(float),
        SOpower_TIB=_read_mat_field(compat_mat, [root, "SOpower_TIB"]).astype(float),
        SOphase_TIB=_read_mat_field(compat_mat, [root, "SOphase_TIB"]).astype(float),
        peak_at_freq_SOpower=_read_mat_field(compat_mat, [root, "num_peaks_at_freq"]).astype(float),
        peak_at_freq_SOphase=_read_mat_field(compat_mat, [root, "num_peaks_at_freq"]).astype(float),
        SOpower_norm=_read_mat_field(compat_mat, [root, "SOpower_norm"]).astype(float),
        SOpower_times=_read_mat_field(compat_mat, [root, "SOpower_times"]).astype(float),
        SOphase=np.zeros_like(_read_mat_field(compat_mat, [root, "SOpower_times"])),
        SOphase_times=_read_mat_field(compat_mat, [root, "SOpower_times"]).astype(float),
    )

    if stats_csv.exists():
        stats = pd.read_csv(stats_csv)
        hist_mask = np.isin(stats["PeakStage"].to_numpy(), [1, 2, 3])
    else:
        stats = pd.DataFrame(columns=["PeakTime", "PeakFrequency", "Volume",
                                      "SOphase", "PeakStage"])
        hist_mask = np.zeros(0, dtype=bool)

    return sophs, stats, hist_mask


def _render_backend(tag, dataset, time_range, out, stats, sophs, hist_mask,
                    ed, fs, dt_sec, cos_vs_mat=None):
    from pydynamo.plot import summary_plot
    i0 = int(round(time_range[0] * fs))
    i1 = int(round(time_range[1] * fs))
    data_tr = ed["data"].ravel()[i0:i1+1].astype(np.float64)
    t_tr = np.arange(i0, i1+1) / fs
    fig = summary_plot(
        out.spect if out is not None else None,
        out.stimes if out is not None else None,
        out.sfreqs if out is not None else None,
        out.artifacts if out is not None else None,
        t_tr,
        ed["stage_times"].ravel(), ed["stage_vals"].ravel(),
        stats, sophs,
        data=data_tr, fs=fs, time_range=time_range,
        freq_limits=(2.0, 25.0), mtm_freq_range=(2.0, 25.0),
        hist_peakidx=hist_mask,
    )
    png_path = DC / f"triple_{dataset}_{tag}.png"
    fig.savefig(png_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    # Add a minimal banner: "<TAG> <time>s"
    img = Image.open(png_path)
    banner_h = 36
    banner = Image.new("RGB", (img.width, img.height + banner_h), "white")
    banner.paste(img, (0, banner_h))
    draw = ImageDraw.Draw(banner)
    # tag formatting: capitalize first word; drop " (baseline)" annotation if any
    display_tag = tag.split(" ")[0].capitalize() if " " in tag else tag.capitalize()
    label = f"{display_tag} {dt_sec:.1f}s"
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 20)
    except OSError:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), label, font=font)
    tw = bbox[2] - bbox[0]
    draw.text(((img.width - tw) // 2, 6), label, fill="black", font=font)
    banner.save(png_path, dpi=(120, 120))
    return png_path


def triple_stitch(png_paths, out_path):
    imgs = [Image.open(p) for p in png_paths]
    h = max(im.height for im in imgs)
    imgs = [im.resize((int(im.width * h / im.height), h)) if im.height != h else im
            for im in imgs]
    w = sum(im.width for im in imgs)
    combined = Image.new("RGB", (w, h), "white")
    x = 0
    for im in imgs:
        combined.paste(im, (x, 0))
        x += im.width
    combined.save(out_path, dpi=(120, 120))
    print(f"  wrote {out_path}")


def run_dataset(name, time_range, compat_mat, stats_csv,
                min_time_in_bin, min_peak_at_freq):
    if not compat_mat.exists():
        print(f"skipping {name}: missing {compat_mat}")
        return
    print(f"\n=== {name} ({time_range[0]}s – {time_range[1]}s) ===")
    from pydynamo.io_compat import load_example_data
    ed = load_example_data()
    fs = float(ed["Fs"])

    print("  running RUST pipeline...")
    out_rs, dt_rs = _run("rust", ed, fs, time_range, min_time_in_bin, min_peak_at_freq)

    print("  running PYTHON-only pipeline...")
    out_py, dt_py = _run("python", ed, fs, time_range, min_time_in_bin, min_peak_at_freq)

    print("  loading MATLAB outputs...")
    mat_sophs, mat_stats, mat_hist_mask = _build_matlab_view(
        name, compat_mat, stats_csv, out_rs
    )
    # Use the pydynamo-rust spectrogram + artifacts + stages as the background
    # panels of the MATLAB figure (same convention as compare_matlab_vs_pydynamo.py).
    dt_mat = _read_matlab_total_runtime(compat_mat)

    # Cos-similarity vs MATLAB for each backend (SOpower + SOphase).
    def _cos(a, b):
        m = np.isfinite(a) & np.isfinite(b)
        return float(np.dot(a[m], b[m]) / (np.linalg.norm(a[m]) * np.linalg.norm(b[m]) + 1e-20))

    mat_sop = np.asarray(mat_sophs.SOpower_mat, dtype=float)
    mat_ph = np.asarray(mat_sophs.SOphase_mat, dtype=float)
    cos_rs = (_cos(out_rs.SOPHs.SOpower_mat, mat_sop),
              _cos(out_rs.SOPHs.SOphase_mat, mat_ph))
    cos_py = (_cos(out_py.SOPHs.SOpower_mat, mat_sop),
              _cos(out_py.SOPHs.SOphase_mat, mat_ph))

    print("  rendering 3 panels...")
    p_rs = _render_backend(
        "rust", name, time_range, out_rs, out_rs.stats_table, out_rs.SOPHs,
        np.isin(out_rs.stats_table["PeakStage"].to_numpy(), [1, 2, 3]) if len(out_rs.stats_table) else np.zeros(0, bool),
        ed, fs, dt_rs, cos_vs_mat=cos_rs,
    )
    p_py = _render_backend(
        "python", name, time_range, out_py, out_py.stats_table, out_py.SOPHs,
        np.isin(out_py.stats_table["PeakStage"].to_numpy(), [1, 2, 3]) if len(out_py.stats_table) else np.zeros(0, bool),
        ed, fs, dt_py, cos_vs_mat=cos_py,
    )
    p_mat = _render_backend(
        "matlab (baseline)", name, time_range, out_rs,  # reuse rust's spect/artifacts for bg
        mat_stats, mat_sophs, mat_hist_mask, ed, fs, dt_mat,
    )

    combined = DC / f"triple_{name}.png"
    triple_stitch([p_mat, p_py, p_rs], combined)


def _read_matlab_total_runtime(compat_mat):
    """Return MATLAB's total runtime in seconds if timings struct is saved,
    else NaN."""
    try:
        m = sio.loadmat(compat_mat, simplify_cells=True)
        if "timings" in m and isinstance(m["timings"], dict):
            return float(m["timings"].get("total", float("nan")))
    except Exception:
        pass
    try:
        with h5py.File(compat_mat, "r") as f:
            if "timings/total" in f:
                return float(np.asarray(f["timings/total"][...]).squeeze())
    except Exception:
        pass
    return float("nan")


def main() -> int:
    run_dataset(
        "segment", (8420.0, 13446.0),
        DC / "segment_out_compat.mat",
        DC / "segment_stats.csv",
        min_time_in_bin=5.0, min_peak_at_freq=10,
    )
    run_dataset(
        "night", None,
        DC / "night_out.mat",
        DC / "night_stats.csv",
        min_time_in_bin=10.0, min_peak_at_freq=0,
    )
    return 0


if __name__ == "__main__":
    # night time_range is computed inside run_dataset; patch it here.
    # Actually we need to compute it once; simpler to just call main and let
    # the night runner handle None → compute in-place.
    # Tweak: rewrite run_dataset to compute None time_range:
    from pydynamo.io_compat import load_example_data
    ed_quick = load_example_data()
    sv = ed_quick["stage_vals"].ravel(); st = ed_quick["stage_times"].ravel()
    nw = np.flatnonzero((sv < 5) & (sv > 0))
    night_tr = (float(st[nw[0]]) - 300, float(st[nw[-1]]) + 300)
    # Run both datasets explicitly
    run_dataset("segment", (8420.0, 13446.0),
                DC / "segment_out_compat.mat", DC / "segment_stats.csv",
                min_time_in_bin=5.0, min_peak_at_freq=10)
    run_dataset("night", night_tr,
                DC / "night_out.mat", DC / "night_stats.csv",
                min_time_in_bin=10.0, min_peak_at_freq=0)
