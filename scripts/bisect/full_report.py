"""Exhaustive stage-by-stage bisect of pydynamo vs MATLAB.

Walks every transitional step of the pipeline, injecting MATLAB intermediates
as inputs wherever we have them, and reports divergence at each boundary.

Output: data_cache/bisect_full_report.txt (human-readable), bisect_full_report.json
        (machine-readable), and prints to stdout.

Stages:
  A. data → artifacts                      (slope_test submask, full mask)
  B. data → spect1 (pass-1, 1s window)
  C. data → spect2 (pass-2, 2s window)
  D. spect1 + exclude → baseline1
  E. spect2 + exclude → baseline2
  F. spect / baseline → spect_norm        (division; should be bit-identical given inputs)
  G. spect1_norm → pass-1 segmentation bounds
  H. MATLAB spect1_norm → pydynamo pass-1 peaks
  I. MATLAB spect2_norm + mask → pydynamo spect2_masked
  J. MATLAB spect2_masked → pydynamo pass-2 peaks
  K. MATLAB stats_pre_refine → Hann-refined
  L. data → SOpower timeseries
  M. data → SOphase timeseries (with MATLAB SOS loaded)
  N. MATLAB stats + SOpower → SOpower_mat (already known bit-identical)
  O. MATLAB stats + SOphase → SOphase_mat (already known cos=0.9998)
  P. Parameter audit (what defaults differ)

Tolerance budget tracked per-stage in the report.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import sosfiltfilt

from pydynamo.artifacts import _slope_test, detect_artifacts
from pydynamo.baseline import compute_baseline, subtract_baseline
from pydynamo.io_compat import load_example_data, load_segment_out
from pydynamo.soph.histogram import so_power_histogram, so_phase_histogram
from pydynamo.soph.sophase import compute_so_phase, _get_sos
from pydynamo.soph.sopower import compute_so_power
from pydynamo.spectrogram import mtm_spectrogram
from pydynamo.tfpeaks.extract import extract_tfpeaks
from pydynamo.tfpeaks.mask import mask_spectrogram
from pydynamo.tfpeaks.refine import refine_peak_frequency


DC = Path(__file__).parent.parent.parent / "data_cache"
BISECT_MAT = DC / "bisect_intermediates_segment.mat"
COMPAT_MAT = DC / "segment_out_compat.mat"
BASELINE_MAT = DC / "baseline_segment.mat"

T_RANGE = (8420.0, 13446.0)


def squeeze(x):
    a = np.asarray(x)
    if a.ndim >= 2:
        a = a.T
    return np.squeeze(a)


def load_matlab_intermediates():
    """Load everything from the bisect + compat mat files into a dict."""
    out = {}
    with h5py.File(BISECT_MAT, "r") as f:
        for k in ["spect1", "spect2", "spect2_masked", "baseline1", "baseline2",
                  "stimes1", "stimes2", "sfreqs", "artifacts",
                  "slope_test_mask", "SOphase_norm", "SOphase_times",
                  "SOphase_stages", "SOphase_filter_sos",
                  "SOphase_impulse_response"]:
            out[k] = squeeze(f[k][...])
        g = f["stats_pre_refine"]
        out["stats_pre_refine"] = {
            k: squeeze(g[k][...]) for k in g.keys() if k != "Boundaries"
        }
    with h5py.File(COMPAT_MAT, "r") as f:
        out["SOpower_norm"] = squeeze(f["SOPHs_flat/SOpower_norm"][...])
        out["SOpower_times"] = squeeze(f["SOPHs_flat/SOpower_times"][...])
        out["SOpower_mat"] = squeeze(f["SOPHs_flat/SOpower_mat"][...])
        out["SOphase_mat"] = squeeze(f["SOPHs_flat/SOphase_mat"][...])
        out["freq_bins"] = squeeze(f["SOPHs_flat/freq_bins"][...])
        out["SOpower_bins"] = squeeze(f["SOPHs_flat/SOpower_bins"][...])
        out["SOphase_bins"] = squeeze(f["SOPHs_flat/SOphase_bins"][...])
        out["SOpower_TIB"] = squeeze(f["SOPHs_flat/SOpower_TIB"][...])
        out["SOphase_TIB"] = squeeze(f["SOPHs_flat/SOphase_TIB"][...])
    # The standalone pass-1 baseline vector shipped earlier
    import scipy.io as sio
    out["baseline1_standalone"] = np.asarray(
        sio.loadmat(str(BASELINE_MAT), simplify_cells=True)["baseline"]
    ).ravel()
    return out


def rel_stats(py: np.ndarray, mat: np.ndarray) -> dict:
    """Compute max_abs, max_rel, p99_rel for continuous arrays."""
    both = np.isfinite(py) & np.isfinite(mat)
    if not both.any():
        return {"max_abs": float("nan"), "max_rel": float("nan"),
                "p99_rel": float("nan"), "n": 0}
    a, b = py[both].astype(np.float64), mat[both].astype(np.float64)
    diff = np.abs(a - b)
    denom = np.maximum(np.abs(b), 1e-20)
    rel = diff / denom
    return {
        "max_abs": float(diff.max()),
        "max_rel": float(rel.max()),
        "p99_rel": float(np.quantile(rel, 0.99)),
        "n": int(both.sum()),
        "cos": float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)),
    }


def bool_stats(py: np.ndarray, mat: np.ndarray) -> dict:
    """Compute boolean-mask agreement metrics."""
    py = np.asarray(py, dtype=bool).ravel()
    mat = np.asarray(mat, dtype=bool).ravel()
    n = min(py.size, mat.size)
    py = py[:n]; mat = mat[:n]
    agree = float((py == mat).mean()) if n else 0.0
    both = int((py & mat).sum())
    either = int((py | mat).sum())
    return {
        "agreement": agree,
        "jaccard": both / max(either, 1),
        "recall": both / max(int(mat.sum()), 1),
        "precision": both / max(int(py.sum()), 1),
        "py_true_frac": float(py.mean()),
        "mat_true_frac": float(mat.mean()),
        "n": n,
    }


REPORT: list[dict] = []


def add(stage: str, status: str, metrics: dict, notes: str = ""):
    rec = {"stage": stage, "status": status, **metrics}
    if notes: rec["notes"] = notes
    REPORT.append(rec)
    # Format one-line console summary
    marker = {"pass": "✓", "close": "≈", "fail": "✗", "info": "·"}[status]
    m = []
    for k in ["max_abs", "max_rel", "p99_rel", "cos",
              "agreement", "jaccard", "recall", "precision",
              "py_shape", "mat_shape", "py", "mat", "ratio",
              "py_n_seg", "mat_n_seg", "n_peaks",
              "n_matched", "p50_abs_Hz", "p95_abs_Hz", "max_abs_Hz"]:
        if k in metrics:
            v = metrics[k]
            if isinstance(v, float):
                m.append(f"{k}={v:.4g}")
            elif isinstance(v, (int, str)):
                m.append(f"{k}={v}")
    if "n" in metrics: m.append(f"n={metrics['n']}")
    line = f"[{marker}] {stage:<50} {' '.join(m)}"
    if notes: line += f"\n      {notes}"
    print(line)


def main() -> int:
    print("=" * 90)
    print("EXHAUSTIVE BISECT REPORT: pydynamo vs MATLAB (segment dataset)")
    print("=" * 90)

    mat = load_matlab_intermediates()
    ed = load_example_data()
    fs = float(ed["Fs"])
    i0 = int(round(T_RANGE[0] * fs))
    i1 = int(round(T_RANGE[1] * fs))
    data_tr = ed["data"].ravel()[i0 : i1 + 1].astype(np.float64)
    t_tr = np.arange(i0, i1 + 1) / fs
    stage_times = ed["stage_times"].ravel().astype(float)
    stage_vals = ed["stage_vals"].ravel().astype(float)

    # =============================================================
    # STAGE A. Artifacts
    # =============================================================
    print("\n" + "-" * 90)
    print("A. DATA → ARTIFACTS")
    print("-" * 90)

    # A1. slope_test submask
    py_slope = _slope_test(data_tr, fs, slope_crit=-0.5)
    mat_slope = np.asarray(mat["slope_test_mask"]).ravel().astype(bool)
    add("A1. slope_test submask", "pass" if bool_stats(py_slope, mat_slope)["agreement"] > 0.999 else "close",
        bool_stats(py_slope, mat_slope),
        "MATLAB reference saved via export_bisect_intermediates.m")

    # A2. Full artifact mask (with slope_test on)
    py_art = detect_artifacts(data_tr, fs, slope_test=True)
    mat_art = np.asarray(mat["artifacts"]).ravel().astype(bool)
    add("A2. artifacts (slope_test=True)",
        "pass" if bool_stats(py_art, mat_art)["agreement"] > 0.999 else "close",
        bool_stats(py_art, mat_art))

    # =============================================================
    # STAGE B/C. Spectrograms
    # =============================================================
    print("\n" + "-" * 90)
    print("B. DATA → SPECT1 (pass-1, 1-s window)")
    print("-" * 90)
    spect1_py, stimes1_py, sfreqs_py = mtm_spectrogram(
        data_tr, fs, freq_range=(0, 30), taper_params=(2, 3),
        window_params=(1.0, 0.05), dsfreqs=0.1,
    )
    # Match shapes: MATLAB stored spect1 as (T, F); transpose to (F, T)
    mat_spect1 = np.asarray(mat["spect1"]).astype(np.float64)  # (F, T) from squeeze()
    add("B1. spect1 shape match",
        "pass" if spect1_py.shape == mat_spect1.shape else "fail",
        {"py_shape": str(spect1_py.shape), "mat_shape": str(mat_spect1.shape)})
    if spect1_py.shape == mat_spect1.shape:
        add("B2. spect1 values (FFT-noise bound)",
            "pass" if rel_stats(spect1_py, mat_spect1)["max_rel"] < 1e-3 else "close",
            rel_stats(spect1_py, mat_spect1))

    print("\n" + "-" * 90)
    print("C. DATA → SPECT2 (pass-2, 2-s window)")
    print("-" * 90)
    spect2_py, stimes2_py, _ = mtm_spectrogram(
        data_tr, fs, freq_range=(0, 30), taper_params=(2, 3),
        window_params=(2.0, 0.05), dsfreqs=0.1,
    )
    mat_spect2 = np.asarray(mat["spect2"]).astype(np.float64)  # (F, T) from squeeze()
    add("C1. spect2 shape match",
        "pass" if spect2_py.shape == mat_spect2.shape else "fail",
        {"py_shape": str(spect2_py.shape), "mat_shape": str(mat_spect2.shape)})
    if spect2_py.shape == mat_spect2.shape:
        add("C2. spect2 values (FFT-noise bound)",
            "pass" if rel_stats(spect2_py, mat_spect2)["max_rel"] < 1e-3 else "close",
            rel_stats(spect2_py, mat_spect2))

    # =============================================================
    # STAGE D/E. Baselines
    # =============================================================
    print("\n" + "-" * 90)
    print("D. SPECT1 + EXCLUDE → BASELINE1")
    print("-" * 90)

    # Build MATLAB-matching baseline_exclude
    stage_at_data = interp1d(stage_times, stage_vals, kind="previous",
                              bounds_error=False, fill_value=0.0)(t_tr)
    stage_exclude = ~np.isin(stage_at_data, (1, 2, 3, 4, 5))
    baseline_exclude = mat_art | stage_exclude

    # D1. pydynamo baseline on MATLAB's pass-1 spect (→ should match MATLAB baseline1 bit-perfectly)
    bl1_from_mat_spect = compute_baseline(
        mat_spect1, np.asarray(mat["stimes1"]).ravel(), t_tr,
        baseline_exclude, baseline_ptile=2.0,
    ).ravel()
    bl1_mat = np.asarray(mat["baseline1"]).ravel().astype(np.float64)
    add("D1. baseline1 from MATLAB spect1 (should be bit-identical)",
        "pass" if rel_stats(bl1_from_mat_spect, bl1_mat)["max_rel"] < 1e-6 else "close",
        rel_stats(bl1_from_mat_spect, bl1_mat))

    # D2. pydynamo baseline on pydynamo spect1 (FFT-noise limited)
    # pydynamo stimes are relative to data start (0); shift to absolute time.
    bl1_from_py_spect = compute_baseline(
        spect1_py, stimes1_py + T_RANGE[0], t_tr, baseline_exclude,
        baseline_ptile=2.0,
    ).ravel()
    add("D2. baseline1 from pydynamo spect1 (FFT-noise bound)",
        "pass" if rel_stats(bl1_from_py_spect, bl1_mat)["max_rel"] < 1e-3 else "close",
        rel_stats(bl1_from_py_spect, bl1_mat))

    print("\n" + "-" * 90)
    print("E. SPECT2 + EXCLUDE → BASELINE2")
    print("-" * 90)
    bl2_from_mat_spect = compute_baseline(
        mat_spect2, np.asarray(mat["stimes2"]).ravel(), t_tr,
        baseline_exclude, baseline_ptile=2.0,
    ).ravel()
    bl2_mat = np.asarray(mat["baseline2"]).ravel().astype(np.float64)
    add("E1. baseline2 from MATLAB spect2 (should be bit-identical)",
        "pass" if rel_stats(bl2_from_mat_spect, bl2_mat)["max_rel"] < 1e-6 else "close",
        rel_stats(bl2_from_mat_spect, bl2_mat))

    # =============================================================
    # STAGE F. Division: spect/baseline → spect_norm
    # =============================================================
    print("\n" + "-" * 90)
    print("F. DIVISION: spect / baseline → spect_norm")
    print("-" * 90)
    # MATLAB: spect_norm_from_mat = mat_spect1 / bl1_mat[:,None]
    mat_spect1_norm = mat_spect1 / bl1_mat[:, None]
    py_spect1_norm = subtract_baseline(mat_spect1, bl1_mat[:, None])
    add("F1. spect1_norm bit-identity",
        "pass" if np.allclose(py_spect1_norm, mat_spect1_norm, atol=0, rtol=0) else "close",
        rel_stats(py_spect1_norm, mat_spect1_norm))

    # =============================================================
    # STAGE G. Segmentation bounds
    # =============================================================
    print("\n" + "-" * 90)
    print("G. SEGMENTATION BOUNDS (30-s windows over stimes)")
    print("-" * 90)
    # Check pydynamo's windowing in extract_tfpeaks matches MATLAB process_segments_params
    # MATLAB: win_samples = round(seg_time / dt); window_start = 0:win_samples:len-1
    dt1 = float(np.asarray(mat["stimes1"]).ravel()[1] -
                 np.asarray(mat["stimes1"]).ravel()[0])
    win_samples = int(round(30.0 / dt1))
    n_stimes = np.asarray(mat["stimes1"]).size
    expected_starts = np.arange(0, n_stimes - 1, win_samples)
    add("G1. pass-1 seg_samples, n_segments",
        "info",
        {"seg_samples": win_samples, "n_seg": len(expected_starts),
         "dt1": dt1, "n_stimes": n_stimes})
    # pydynamo's extract_tfpeaks uses: start in range(0, T, samples_per_seg)
    T = n_stimes
    py_starts = list(range(0, T, win_samples))
    py_n_seg = sum(1 for s in py_starts if T - s >= 2)
    add("G2. pydynamo seg start alignment",
        "pass" if py_n_seg == len(expected_starts) else "close",
        {"py_n_seg": py_n_seg, "mat_n_seg": len(expected_starts)})

    # =============================================================
    # STAGE H. pass-1 extract on MATLAB spect1_norm
    # =============================================================
    print("\n" + "-" * 90)
    print("H. MATLAB SPECT1_NORM → pydynamo pass-1 peaks")
    print("-" * 90)
    pass1 = extract_tfpeaks(
        mat_spect1_norm, np.asarray(mat["stimes1"]).ravel(),
        np.asarray(mat["sfreqs"]).ravel(),
        seg_time=30.0, n_jobs=1,
        downsample=(2, 2), merge_thresh=11.0, trim_vol=0.8,
        dur_min=0.5, dur_max=5.0, bw_min=2.0, bw_max=15.0,
    )
    add("H1. pass-1 peaks on MATLAB spect1_norm", "info",
        {"n_peaks": len(pass1)},
        "MATLAB pass-1 count not separately saved; this is informational only")

    # =============================================================
    # STAGE I. Masking: spect2_norm + pass-1 labels → spect2_masked
    # =============================================================
    print("\n" + "-" * 90)
    print("I. MASKING: spect2_norm + pass-1 regions → spect2_masked")
    print("-" * 90)
    mat_spect2_masked = np.asarray(mat["spect2_masked"]).astype(np.float64)
    # The MATLAB mask sets zero outside pass-1 regions. We can reconstruct
    # an "approximate pass-1 mask" by checking where spect2_masked is nonzero
    # vs spect2_norm.
    mat_spect2_norm = mat_spect2 / bl2_mat[:, None]
    # mask_equivalent = mat_spect2_masked > 0  (MATLAB sets 0 outside regions)
    # Now we can apply pydynamo's mask_spectrogram with labels1 (from stats)
    # but we don't have labels1 directly. So compare mask element-wise to
    # the "nonzero" set.
    mat_mask = mat_spect2_masked != 0  # True inside regions
    # pydynamo's mask implementation: builds labels image, masks via
    # mask_spectrogram. We'd need labels to test that directly. Instead we
    # check that pydynamo's pass-2 extract on MATLAB spect2_masked gives
    # MATLAB-like peak counts (done separately).
    py_mask_from_labels = None
    add("I1. MATLAB spect2_masked mask fraction",
        "info",
        {"mat_mask_nonzero_frac": float(mat_mask.mean())})

    # Test I2: given ideal pass-1 label image (reconstructed from mat_mask),
    # pydynamo mask_spectrogram should produce identical output.
    labels_fake = mat_mask.astype(np.int64) * 1  # label 1 everywhere in region
    pyd_masked = mask_spectrogram(
        mat_spect2_norm,
        np.asarray(mat["stimes2"]).ravel(),
        labels_fake,
        np.asarray(mat["stimes2"]).ravel(),  # same stimes (no time shift)
    )
    # Compare to MATLAB: pydynamo output should be zero where mat_mask==False,
    # and == mat_spect2_norm where mat_mask==True. Since MATLAB also sets
    # watershed pixels to zero (small additional pixels), expect close match.
    add("I2. pydynamo mask_spectrogram on reconstructed mask",
        "pass" if rel_stats(pyd_masked[mat_mask], mat_spect2_masked[mat_mask])["max_rel"] < 1e-6 else "close",
        rel_stats(pyd_masked[mat_mask], mat_spect2_masked[mat_mask]),
        "Compared inside the mask only; reconstruction from mat_spect2_masked>0 loses watershed 0-line pixels")

    # =============================================================
    # STAGE J. pass-2 extract on MATLAB spect2_masked
    # =============================================================
    print("\n" + "-" * 90)
    print("J. MATLAB SPECT2_MASKED → pydynamo pass-2 peaks (pre-refine)")
    print("-" * 90)
    pass2 = extract_tfpeaks(
        mat_spect2_masked, np.asarray(mat["stimes2"]).ravel(),
        np.asarray(mat["sfreqs"]).ravel(),
        seg_time=30.0, n_jobs=1,
        downsample=(2, 2), merge_thresh=11.0, trim_vol=0.8,
        # Match REAL pipeline: pass-2 uses dur_min=0.5 (from pass-1, see
        # computeTFPeaks.m:342) and bw_min=1.0 (df/2 = TW/T for pass-2).
        dur_min=0.5, dur_max=5.0, bw_min=1.0, bw_max=15.0,
    )
    n_py = len(pass2)
    mat_pre_pt = np.asarray(mat["stats_pre_refine"]["PeakTime"]).ravel()
    mat_pre_pf = np.asarray(mat["stats_pre_refine"]["PeakFrequency"]).ravel()
    n_mat = mat_pre_pt.size
    # Hungarian-style matching in 10-min windows
    from scipy.optimize import linear_sum_assignment
    matched = 0
    pt = pass2["PeakTime"].to_numpy()
    pf = pass2["PeakFrequency"].to_numpy()
    for t0 in np.arange(T_RANGE[0], T_RANGE[1], 600):
        t1_w = t0 + 600
        mp = (pt >= t0) & (pt < t1_w); mr = (mat_pre_pt >= t0) & (mat_pre_pt < t1_w)
        if not mp.any() or not mr.any(): continue
        a_t, a_f = pt[mp], pf[mp]
        b_t, b_f = mat_pre_pt[mr], mat_pre_pf[mr]
        dt = a_t[:, None] - b_t[None, :]
        df = a_f[:, None] - b_f[None, :]
        cost = (dt / 0.5) ** 2 + (df / 0.5) ** 2
        cost[cost > 4.0] = 1e9
        r, c = linear_sum_assignment(cost)
        matched += int(((cost[r, c] < 1e8)).sum())
    add("J1. pass-2 peaks vs MATLAB pre-refine",
        "close",
        {"py": n_py, "mat": n_mat,
         "recall": matched / max(n_mat, 1),
         "precision": matched / max(n_py, 1),
         "ratio": n_py / n_mat},
        "Hungarian match at 0.5s/0.5Hz gate. recall<100% means pydynamo is missing some MATLAB peaks; precision<100% means pydynamo has spurious peaks.")

    # =============================================================
    # STAGE K. Hann refinement
    # =============================================================
    print("\n" + "-" * 90)
    print("K. MATLAB STATS_PRE_REFINE → Hann refinement")
    print("-" * 90)
    pre = pd.DataFrame({
        "PeakTime": mat_pre_pt,
        "PeakFrequency": mat_pre_pf,
        "Duration": mat["stats_pre_refine"]["Duration"].ravel(),
        "Bandwidth": mat["stats_pre_refine"]["Bandwidth"].ravel(),
        "BoundingBox": [
            (float(pt_i - d/2), float(pf_i - bw/2), float(d), float(bw))
            for pt_i, pf_i, d, bw in zip(
                mat_pre_pt, mat_pre_pf,
                mat["stats_pre_refine"]["Duration"].ravel(),
                mat["stats_pre_refine"]["Bandwidth"].ravel())
        ],
    })
    refined = refine_peak_frequency(
        pre, data_tr, fs, t=t_tr,
        freq_range=(0.0, 30.0), window_size=4.0, dsfreqs=0.05,
        refine_method="spline_interp", remove_edge_peaks=True,
    )
    # Compare to MATLAB final stats
    mat_final = pd.read_csv(DC / "segment_stats.csv")
    from scipy.spatial import cKDTree
    tree = cKDTree(mat_final["PeakTime"].to_numpy()[:, None])
    dist, idx = tree.query(refined["PeakTime"].to_numpy()[:, None], k=1)
    ok = dist < 0.01
    diff_pf = refined["PeakFrequency"].to_numpy()[ok] - mat_final["PeakFrequency"].to_numpy()[idx[ok]]
    add("K1. refined freq vs MATLAB (time-matched)",
        "close",
        {"n_matched": int(ok.sum()),
         "p50_abs_Hz": float(np.median(np.abs(diff_pf))),
         "p95_abs_Hz": float(np.quantile(np.abs(diff_pf), 0.95)),
         "max_abs_Hz": float(np.abs(diff_pf).max())},
        "p50 << 0.1 Hz means refinement is essentially exact for median peak.")

    # =============================================================
    # STAGE L. SO-power timeseries
    # =============================================================
    print("\n" + "-" * 90)
    print("L. DATA → SO-POWER TIMESERIES")
    print("-" * 90)
    sop_py, sop_t_py, _, _, _ = compute_so_power(
        data_tr, fs, stage_times=stage_times, stage_vals=stage_vals,
        eeg_times=t_tr, time_range=T_RANGE, isexcluded=mat_art,
        SO_freqrange=(0.3, 1.5), tapers=(5, 9), window_params=(5.0, 0.5),
        SOpower_outlier_threshold=3.0, norm_method="p2shift1234",
        retain_Fs=True,
    )
    add("L1. SOpower_norm vs MATLAB", "close",
        rel_stats(sop_py, np.asarray(mat["SOpower_norm"]).astype(float)))

    # =============================================================
    # STAGE M. SO-phase timeseries (with MATLAB SOS loaded)
    # =============================================================
    print("\n" + "-" * 90)
    print("M. DATA → SO-PHASE TIMESERIES (MATLAB SOS)")
    print("-" * 90)
    sop_py, _, _, _ = compute_so_phase(
        data_tr, fs, stage_times=stage_times, stage_vals=stage_vals,
        eeg_times=t_tr, isexcluded=mat_art, SO_freqrange=(0.3, 1.5),
    )
    # Wrap to match MATLAB's SOphase_norm
    soph_py_wrap = (sop_py + np.pi) % (2 * np.pi) - np.pi
    mat_soph = np.asarray(mat["SOphase_norm"]).astype(float)
    add("M1. SOphase_norm vs MATLAB (with MATLAB SOS)", "close",
        rel_stats(soph_py_wrap, mat_soph))
    # Circular-distance stats: wrap difference to (-π, π] then take |.|
    d = soph_py_wrap - mat_soph
    d = np.mod(d + np.pi, 2 * np.pi) - np.pi
    d_abs = np.abs(d)
    finite = np.isfinite(d_abs) & ~mat_art
    if finite.any():
        add("M2. SOphase circular error (excl artifacts)",
            "pass" if np.median(d_abs[finite]) < 0.01 else "close",
            {"median_rad": float(np.median(d_abs[finite])),
             "p95_rad": float(np.quantile(d_abs[finite], 0.95)),
             "p99_rad": float(np.quantile(d_abs[finite], 0.99)),
             "max_rad": float(d_abs[finite].max()),
             "n": int(finite.sum())},
            "Circular distance = |wrap(py - mat, 2π)|. If median > 0.01 rad, filter outputs diverge beyond edge transients.")
        # Split by region: first 5% / middle 90% / last 5%
        n_total = finite.size
        edges = [(0, int(0.05*n_total), "first_5pct"),
                 (int(0.05*n_total), int(0.95*n_total), "middle_90pct"),
                 (int(0.95*n_total), n_total, "last_5pct")]
        for i0_e, i1_e, lbl in edges:
            seg = finite[i0_e:i1_e] & np.isfinite(d_abs[i0_e:i1_e])
            if seg.any():
                sl = d_abs[i0_e:i1_e][seg]
                add(f"M3. SOphase circular error [{lbl}]", "info",
                    {"median_rad": float(np.median(sl)),
                     "p95_rad": float(np.quantile(sl, 0.95)),
                     "n": int(seg.sum())})

    # =============================================================
    # STAGE N, O. SOPH binning
    # =============================================================
    print("\n" + "-" * 90)
    print("N, O. SOPH BINNING (MATLAB inputs only)")
    print("-" * 90)
    stats = mat_final
    # Compute SOpower_stages the way pydynamo does
    sop_stages = interp1d(stage_times, stage_vals, kind="previous",
                           bounds_error=False, fill_value=0.0)(
        np.asarray(mat["SOpower_times"]).astype(float))
    sop_stages = np.where(np.isnan(sop_stages), 0.0, sop_stages)
    out_pow = so_power_histogram(
        stats["PeakFrequency"].to_numpy(), stats["PeakTime"].to_numpy(),
        stats["PeakStage"].to_numpy(),
        np.asarray(mat["SOpower_norm"]).astype(float),
        np.asarray(mat["SOpower_times"]).astype(float),
        sop_stages,
        time_range=T_RANGE, soph_stages=(1, 2, 3),
        freq_range=(0.0, 30.0), freq_binsizestep=(1.0, 0.2),
        so_range=None, so_binsizestep=None,
        min_time_in_bin=5.0, compute_rate=True, norm_dim=0,
    )
    mat_pow = np.asarray(mat["SOpower_mat"])  # (nfreq, nbins) from squeeze()
    add("N1. SOpower_mat bit-identity", "pass" if np.allclose(
            out_pow["c_mat"], mat_pow, atol=0, rtol=0, equal_nan=True) else "close",
        rel_stats(out_pow["c_mat"], mat_pow))

    soph_stages = np.asarray(mat["SOphase_stages"]).astype(float)
    out_ph = so_phase_histogram(
        stats["PeakFrequency"].to_numpy(), stats["PeakTime"].to_numpy(),
        stats["PeakStage"].to_numpy(),
        mat_soph, np.asarray(mat["SOphase_times"]).astype(float), soph_stages,
        time_range=T_RANGE, soph_stages=(1, 2, 3),
        freq_range=(0.0, 30.0), freq_binsizestep=(1.0, 0.2),
        so_range=(-np.pi, np.pi), so_binsizestep=(2*np.pi/5, 2*np.pi/100),
        min_peak_at_freq=1, compute_rate=True, norm_dim=1,
    )
    mat_ph = np.asarray(mat["SOphase_mat"])
    add("O1. SOphase_mat", "close", rel_stats(out_ph["c_mat"], mat_ph))

    # =============================================================
    # Emit report
    # =============================================================
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    for rec in REPORT:
        marker = {"pass": "✓", "close": "≈", "fail": "✗", "info": "·"}[rec["status"]]
        print(f"[{marker}] {rec['stage']}")

    # Write JSON + text report
    with open(DC / "bisect_full_report.json", "w") as f:
        json.dump(REPORT, f, indent=2, default=str)
    with open(DC / "bisect_full_report.txt", "w") as f:
        for rec in REPORT:
            marker = {"pass": "PASS", "close": "CLOSE", "fail": "FAIL", "info": "INFO"}[rec["status"]]
            f.write(f"[{marker}] {rec['stage']}\n")
            for k, v in rec.items():
                if k in ("stage", "status"): continue
                f.write(f"    {k}: {v}\n")
            f.write("\n")
    print(f"\nwrote {DC/'bisect_full_report.json'} and .txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
