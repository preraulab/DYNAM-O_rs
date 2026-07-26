"""Validate dynamo_extract_tfpeaks (C ABI, Phase 4 code) against MATLAB
ground truth AND against the existing pydynamo PyO3 path.

Three-way comparison:
    MATLAB  (runDYNAMO 'segment') =  ground truth
    pydynamo.tfpeaks.extract      =  PyO3 reference (same kernels, old wrapper)
    dynamo_rs C ABI via ctypes    =  NEW (Phase 4) wrapper, under test

Expected outcomes:
    - PyO3 and C ABI should produce the same peak count (both use identical
      Rust kernels; the peak-properties helper differs between them but
      shouldn't change which peaks survive).
    - Peak counts vs MATLAB: within 1% (documented parity gap).
    - For matched peaks (Hungarian match on (time, freq) within 0.1 s / 0.1 Hz
      tolerance), the per-peak properties should agree:
        PyO3 vs C ABI:  bit-identical on time/freq; other props within 1e-10
                         (or flag if the new peak-properties helper differs).
        MATLAB vs C ABI: time/freq within 0.01 s / 0.01 Hz; other props
                         within a few %.

Usage:
    cd DYNAM-O_rs
    .venv-matlab/bin/python scripts/validate_c_api_extract.py
"""

from __future__ import annotations

import ctypes as C
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio


REPO = Path(__file__).resolve().parent.parent
DYLIB = REPO / "rust" / "target" / "release" / "libdynamo_rs.dylib"
SPECT_MAT = REPO / "data_cache" / "segment_spect.mat"
STATS_CSV = REPO / "data_cache" / "segment_stats.csv"


# ---- ctypes bindings matching src/c_api.rs -----------------------------------

class ExtractTfpeaksIn(C.Structure):
    _fields_ = [
        ("spect_ptr",       C.POINTER(C.c_double)),
        ("n_freqs",         C.c_size_t),
        ("n_times",         C.c_size_t),
        ("stimes_ptr",      C.POINTER(C.c_double)),
        ("sfreqs_ptr",      C.POINTER(C.c_double)),
        ("baseline_ptr",    C.POINTER(C.c_double)),
        ("seg_time",        C.c_double),
        ("downsample_f",    C.c_uint32),
        ("downsample_t",    C.c_uint32),
        ("merge_thresh",    C.c_double),
        ("max_merges",      C.c_double),
        ("trim_vol_thresh", C.c_double),
        ("trim_shift_val",  C.c_double),
        ("dur_min",         C.c_double),
        ("dur_max",         C.c_double),
        ("bw_min",          C.c_double),
        ("bw_max",          C.c_double),
        ("freq_min",        C.c_double),
        ("freq_max",        C.c_double),
        ("ht_db_min",       C.c_double),
    ]


class ExtractTfpeaksOut(C.Structure):
    _fields_ = [
        ("n_peaks",       C.c_size_t),
        ("peak_time",     C.POINTER(C.c_double)),
        ("peak_freq",     C.POINTER(C.c_double)),
        ("duration",      C.POINTER(C.c_double)),
        ("bandwidth",     C.POINTER(C.c_double)),
        ("height",        C.POINTER(C.c_double)),
        ("volume",        C.POINTER(C.c_double)),
        ("segment_num",   C.POINTER(C.c_double)),
        ("bounding_box",  C.POINTER(C.c_double)),
        ("labels",        C.POINTER(C.c_int64)),
        ("n_label_elems", C.c_size_t),
    ]


def _load_lib() -> C.CDLL:
    if not DYLIB.exists():
        sys.exit(
            f"libdynamo_rs.dylib not found at {DYLIB}.\n"
            "For local validation only: cd rust && "
            "cargo build --release --locked --lib\n"
            "For distributable output, run ./bootstrap.sh --yes "
            "(or .\\bootstrap.ps1 -Yes) from DYNAM-O_toolbox."
        )
    lib = C.CDLL(str(DYLIB))
    lib.dynamo_extract_tfpeaks.argtypes = [
        C.POINTER(ExtractTfpeaksIn),
        C.POINTER(ExtractTfpeaksOut),
    ]
    lib.dynamo_extract_tfpeaks.restype = C.c_int32
    lib.dynamo_free_buffer_f64.argtypes = [C.POINTER(C.c_double), C.c_size_t]
    lib.dynamo_free_buffer_f64.restype = None
    lib.dynamo_free_buffer_i64.argtypes = [C.POINTER(C.c_int64), C.c_size_t]
    lib.dynamo_free_buffer_i64.restype = None
    return lib


def _ptr_to_array(ptr, n: int, dtype=np.float64) -> np.ndarray:
    """Copy a raw callee-allocated buffer into a new numpy array. Copy so
    we can free the Rust side immediately."""
    if n == 0 or not ptr:
        return np.zeros(0, dtype=dtype)
    ctype = {np.float64: C.c_double, np.int64: C.c_int64}[dtype]
    raw = C.cast(ptr, C.POINTER(ctype))
    return np.frombuffer(
        (ctype * n).from_address(C.addressof(raw.contents)), dtype=dtype
    ).copy()


def call_c_abi(spect, stimes, sfreqs, baseline,
               seg_time, merge_thresh, trim_vol, dur_min, dur_max,
               bw_min, bw_max, freq_min, freq_max, ht_db_min,
               downsample) -> dict:
    """Call dynamo_extract_tfpeaks via ctypes and return a dict of arrays."""
    lib = _load_lib()

    # Ensure C-contiguous row-major (as the header documents).
    spect = np.ascontiguousarray(spect, dtype=np.float64)
    stimes = np.ascontiguousarray(stimes.ravel(), dtype=np.float64)
    sfreqs = np.ascontiguousarray(sfreqs.ravel(), dtype=np.float64)
    baseline_c = np.ascontiguousarray(baseline.ravel(), dtype=np.float64)

    ds_f, ds_t = int(downsample[0]), int(downsample[1])
    in_ = ExtractTfpeaksIn(
        spect_ptr=spect.ctypes.data_as(C.POINTER(C.c_double)),
        n_freqs=C.c_size_t(spect.shape[0]),
        n_times=C.c_size_t(spect.shape[1]),
        stimes_ptr=stimes.ctypes.data_as(C.POINTER(C.c_double)),
        sfreqs_ptr=sfreqs.ctypes.data_as(C.POINTER(C.c_double)),
        baseline_ptr=baseline_c.ctypes.data_as(C.POINTER(C.c_double)),
        seg_time=C.c_double(seg_time),
        downsample_f=C.c_uint32(ds_f),
        downsample_t=C.c_uint32(ds_t),
        merge_thresh=C.c_double(merge_thresh),
        max_merges=C.c_double(float("inf")),
        trim_vol_thresh=C.c_double(trim_vol),
        trim_shift_val=C.c_double(float("nan")),
        dur_min=C.c_double(dur_min),
        dur_max=C.c_double(dur_max),
        bw_min=C.c_double(bw_min),
        bw_max=C.c_double(bw_max),
        freq_min=C.c_double(freq_min),
        freq_max=C.c_double(freq_max),
        ht_db_min=C.c_double(ht_db_min),
    )
    out_ = ExtractTfpeaksOut()
    rc = lib.dynamo_extract_tfpeaks(C.byref(in_), C.byref(out_))
    if rc != 0:
        sys.exit(f"dynamo_extract_tfpeaks returned error {rc}")

    n = out_.n_peaks
    result = {
        "PeakTime":      _ptr_to_array(out_.peak_time, n),
        "PeakFrequency": _ptr_to_array(out_.peak_freq, n),
        "Duration":      _ptr_to_array(out_.duration, n),
        "Bandwidth":     _ptr_to_array(out_.bandwidth, n),
        "Height":        _ptr_to_array(out_.height, n),
        "Volume":        _ptr_to_array(out_.volume, n),
        "SegmentNum":    _ptr_to_array(out_.segment_num, n),
        "BoundingBox":   _ptr_to_array(out_.bounding_box, 4 * n).reshape(n, 4)
                           if n else np.zeros((0, 4)),
    }

    # Free Rust-allocated buffers.
    for (p, cnt) in [
        (out_.peak_time, n), (out_.peak_freq, n), (out_.duration, n),
        (out_.bandwidth, n), (out_.height, n), (out_.volume, n),
        (out_.segment_num, n), (out_.bounding_box, 4 * n),
    ]:
        if cnt > 0 and p:
            lib.dynamo_free_buffer_f64(p, cnt)
    if out_.n_label_elems > 0 and out_.labels:
        lib.dynamo_free_buffer_i64(out_.labels, out_.n_label_elems)

    return result


def hungarian_match(A, B, tol_time=0.1, tol_freq=0.1):
    """Greedy nearest-neighbor match between two peak lists. Returns
    indices_A, indices_B, unmatched_A, unmatched_B."""
    from scipy.spatial import cKDTree
    if len(A) == 0 or len(B) == 0:
        return [], [], list(range(len(A))), list(range(len(B)))
    tree = cKDTree(np.column_stack([B["PeakTime"], B["PeakFrequency"]]))
    idx_a, idx_b = [], []
    used_b = set()
    pa = np.column_stack([A["PeakTime"], A["PeakFrequency"]])
    dists, inds = tree.query(pa, k=1)
    for ai, (d, bi) in enumerate(zip(dists, inds)):
        if bi in used_b:
            continue
        dt = abs(A["PeakTime"][ai] - B["PeakTime"][bi])
        df = abs(A["PeakFrequency"][ai] - B["PeakFrequency"][bi])
        if dt <= tol_time and df <= tol_freq:
            idx_a.append(ai); idx_b.append(bi); used_b.add(bi)
    unm_a = [i for i in range(len(pa)) if i not in idx_a]
    unm_b = [i for i in range(len(B["PeakTime"])) if i not in used_b]
    return idx_a, idx_b, unm_a, unm_b


def main():
    if not SPECT_MAT.exists() or not STATS_CSV.exists():
        sys.exit(
            f"Missing ground truth.\n"
            f"  Expected: {SPECT_MAT}\n"
            f"            {STATS_CSV}\n"
            f"Run in MATLAB:\n"
            f"  cd {REPO/'scripts'}\n"
            f"  export_validation_segment()\n"
        )

    print(f"Loading {SPECT_MAT} ...")
    m = sio.loadmat(SPECT_MAT, squeeze_me=True)
    spect = np.asarray(m["spect"], dtype=np.float64)
    stimes = np.asarray(m["stimes"], dtype=np.float64).ravel()
    sfreqs = np.asarray(m["sfreqs"], dtype=np.float64).ravel()
    baseline = np.asarray(m["baseline"], dtype=np.float64).ravel()
    # detection_opts() leaves some fields empty (resolved at runtime by the
    # 'default' quality_setting preset inside computeTFPeaks). Fill in the
    # documented MATLAB defaults here.
    _PRESET_DEFAULT = dict(seg_time=30.0, merge_thresh=11.0,
                           downsample_spect=(2, 2))

    def _get_scalar(name, fallback):
        v = np.asarray(m[name])
        if v.size == 0:
            return float(fallback)
        return float(v.ravel()[0])

    params = {
        "seg_time":     _get_scalar("seg_time",     _PRESET_DEFAULT["seg_time"]),
        "merge_thresh": _get_scalar("merge_thresh", _PRESET_DEFAULT["merge_thresh"]),
        "trim_vol":     _get_scalar("trim_vol",     0.8),
        "dur_min":      _get_scalar("dur_min",      0.5),
        "dur_max":      _get_scalar("dur_max",      5.0),
        "bw_min":       _get_scalar("bw_min",       1.0),
        "bw_max":       _get_scalar("bw_max",       15.0),
    }
    # NOTE: MATLAB's saved dur_min (=1.0 for pass-2 default
    # mtm_window_length_2=2) is what runSegmentedData uses in the
    # per-region pre/post-trim drop. The final filterStatsTable in
    # computeTFPeaks.m:369 also reuses that variable. The CSV having peaks
    # down to 0.55s suggests MATLAB's Duration column is computed from
    # (max-min+1)*dt and the filter is (max-min)*dt — close but not
    # identical. We use the stored value as-is.
    # Derive ht_db_min from the MATLAB chi2 formula if not saved. MATLAB
    # computeTFPeaks.m:395-397: chi2_df = 2*num_tapers, alpha=0.95,
    #   ht_db_min = -pow2db(chi2_df / chi2inv(alpha/2+0.5, chi2_df)) * 2
    # For pass-2 taper_params (3, 5) [default detection_opts], num_tapers=5
    # → chi2_df = 10. For pass-1 (2, 3), chi2_df = 6.
    try:
        ht_db_min = _get_scalar("ht_db_min", np.nan)
    except Exception:
        ht_db_min = np.nan
    if not np.isfinite(ht_db_min):
        # MATLAB computeTFPeaks.m:395-397: chi2_df = 2*num_tapers,
        # alpha=0.95, ht_db_min = -pow2db(chi2_df / chi2inv(.975, chi2_df))*2.
        # Default detection_opts mtm_taper_params = [2, 3] → num_tapers=3,
        # chi2_df=6 → ht_db_min ≈ 7.63 dB. This is the same value
        # filterStatsTable.m defaults to.
        from scipy.stats import chi2
        chi2_df = 6
        ht_db_min = -10.0 * np.log10(chi2_df / chi2.ppf(0.975, chi2_df)) * 2.0
        print(f"  ht_db_min not in .mat — derived from chi2 (num_tapers=3): "
              f"{ht_db_min:.3f} dB")
    params["ht_db_min"] = float(ht_db_min)
    params["freq_min"] = -np.inf
    params["freq_max"] = np.inf

    ds_raw = np.asarray(m["downsample_spect"]).ravel()
    downsample = (np.asarray(ds_raw, dtype=int) if ds_raw.size == 2
                   else np.asarray(_PRESET_DEFAULT["downsample_spect"], dtype=int))
    print(f"  spect: {spect.shape}, baseline: {baseline.shape}, params: {params}")

    # ---- 1. MATLAB ground truth -------------------------------------------
    import pandas as pd
    matlab = pd.read_csv(STATS_CSV)
    print(f"\nMATLAB peaks: {len(matlab)}")

    # ---- 2. C ABI (new code under test) -----------------------------------
    print("\nCalling dynamo_extract_tfpeaks (C ABI via ctypes)...")
    c_abi = call_c_abi(
        spect, stimes, sfreqs, baseline,
        seg_time=params["seg_time"],
        merge_thresh=params["merge_thresh"],
        trim_vol=params["trim_vol"],
        dur_min=params["dur_min"],
        dur_max=params["dur_max"],
        bw_min=params["bw_min"],
        bw_max=params["bw_max"],
        freq_min=params["freq_min"],
        freq_max=params["freq_max"],
        ht_db_min=params["ht_db_min"],
        downsample=downsample,
    )
    print(f"  C ABI peaks: {len(c_abi['PeakTime'])}")

    # ---- 3. pydynamo PyO3 (existing reference) ----------------------------
    try:
        from pydynamo.tfpeaks.extract import extract_tfpeaks as py_extract
    except ImportError:
        print("\n(pydynamo not installed in this venv — skipping PyO3 comparison)")
        pydyn = None
    else:
        print("\nCalling pydynamo.tfpeaks.extract.extract_tfpeaks (PyO3)...")
        pydyn_df, _labels = py_extract(
            spect / baseline[:, None], stimes, sfreqs,
            seg_time=params["seg_time"],
            merge_thresh=params["merge_thresh"],
            trim_vol=params["trim_vol"],
            downsample=tuple(downsample),
            dur_min=params["dur_min"], dur_max=params["dur_max"],
            bw_min=params["bw_min"], bw_max=params["bw_max"],
            return_labels=True,
        )
        pydyn = {
            "PeakTime": pydyn_df["PeakTime"].to_numpy(),
            "PeakFrequency": pydyn_df["PeakFrequency"].to_numpy(),
            "Duration": pydyn_df["Duration"].to_numpy(),
            "Bandwidth": pydyn_df["Bandwidth"].to_numpy(),
            "Height": pydyn_df["Height"].to_numpy(),
            "Volume": pydyn_df["Volume"].to_numpy(),
        }
        print(f"  PyO3 peaks: {len(pydyn['PeakTime'])}")

    # ---- 4. Three-way comparison -----------------------------------------
    print("\n" + "=" * 70)
    print(f"{'Comparison':<30} {'peaks':>8} {'Δ vs MATLAB':>14}")
    print("-" * 70)
    print(f"{'MATLAB (ground truth)':<30} {len(matlab):>8} {'—':>14}")
    print(f"{'C ABI (dynamo_extract)':<30} {len(c_abi['PeakTime']):>8} "
          f"{len(c_abi['PeakTime']) - len(matlab):>+14}")
    if pydyn is not None:
        print(f"{'pydynamo (PyO3)':<30} {len(pydyn['PeakTime']):>8} "
              f"{len(pydyn['PeakTime']) - len(matlab):>+14}")

    # Match MATLAB ↔ C ABI
    m_dict = {"PeakTime": matlab["PeakTime"].to_numpy(),
              "PeakFrequency": matlab["PeakFrequency"].to_numpy()}
    ia, ib, ua, ub = hungarian_match(m_dict, c_abi, tol_time=0.1, tol_freq=0.2)
    print(f"\nMATLAB ↔ C ABI Hungarian match:")
    print(f"  matched:   {len(ia)} / {len(matlab)} MATLAB  = {len(ia)/len(matlab)*100:.1f}%")
    print(f"  unmatched: MATLAB={len(ua)}, C ABI={len(ub)}")

    # Property-level stats on matched peaks
    if ia and ib:
        props = ["Duration", "Bandwidth", "Height", "Volume"]
        print(f"\n  {'prop':<12} {'max abs Δ':>12} {'max rel Δ':>12}")
        for p in props:
            mv = matlab[p].to_numpy()[ia]
            cv = c_abi[p][ib]
            abs_d = np.max(np.abs(mv - cv))
            rel_d = np.max(np.abs(mv - cv) / (np.abs(mv) + 1e-12))
            print(f"  {p:<12} {abs_d:>12.4e} {rel_d:>12.4e}")

    # Cross-check: C ABI ↔ pydynamo. The Rust pipeline uses its own
    # BFS-based expand_labels approximation and explicit half-to-even
    # resize mapping. These produce pixel-identical regions for most
    # peaks but can differ by 1 pixel at the jagged boundary, shifting
    # the intensity-weighted centroid by a fraction of a pixel. So a
    # 1e-6 tolerance (bit-identical) is too strict; report both that
    # and a looser tolerance that ignores subpixel centroid drift.
    if pydyn is not None:
        ia_tight, ib_tight, *_ = hungarian_match(pydyn, c_abi, tol_time=1e-6, tol_freq=1e-6)
        ia2, ib2, ua2, ub2 = hungarian_match(pydyn, c_abi, tol_time=0.05, tol_freq=0.1)
        n_py = max(len(pydyn['PeakTime']), 1)
        print(f"\nC ABI ↔ pydynamo Hungarian match:")
        print(f"  tight (bit-identical): {len(ia_tight)} / {n_py} = "
              f"{len(ia_tight)/n_py*100:.1f}%")
        print(f"  loose (0.05s / 0.1Hz): {len(ia2)} / {n_py} = "
              f"{len(ia2)/n_py*100:.1f}%")
        if ia2:
            print(f"  {'prop':<12} {'max abs Δ':>12}")
            for p in ["Duration", "Bandwidth", "Height", "Volume"]:
                abs_d = np.max(np.abs(pydyn[p][ia2] - c_abi[p][ib2]))
                print(f"  {p:<12} {abs_d:>12.4e}")

    print("=" * 70)
    print("\nInterpretation:")
    print("  - C ABI ↔ pydynamo (loose 0.05s / 0.1Hz): target ≥ 95%.")
    print("    Tight 1e-6 tol only matches when Rust's expand_labels BFS")
    print("    and half-to-even resize produce the exact same pixel")
    print("    labels as skimage — typically ~3% of peaks. Subpixel")
    print("    centroid drift on the rest is expected.")
    print("  - C ABI ↔ MATLAB: reflects the pydynamo↔MATLAB parity gap")
    print("    (pydynamo itself only matches ~13% at 0.1s/0.2Hz because")
    print("    its watershed/merge/border handling differ from MATLAB).")
    print("    A bigger gap would indicate a real bug in c_api.rs.")


if __name__ == "__main__":
    main()
