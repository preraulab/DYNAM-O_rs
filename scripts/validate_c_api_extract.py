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
        ("spect_ptr",      C.POINTER(C.c_double)),
        ("n_freqs",        C.c_size_t),
        ("n_times",        C.c_size_t),
        ("stimes_ptr",     C.POINTER(C.c_double)),
        ("sfreqs_ptr",     C.POINTER(C.c_double)),
        ("baseline_ptr",   C.POINTER(C.c_double)),
        ("merge_thresh",   C.c_double),
        ("max_merges",     C.c_double),
        ("trim_vol_thresh", C.c_double),
        ("trim_shift_val", C.c_double),
        ("segment_num",    C.c_double),
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
        sys.exit(f"libdynamo_rs.dylib not found at {DYLIB}. "
                 f"Run: cd rust && cargo build --release")
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
               bw_min, bw_max, downsample) -> dict:
    """Call dynamo_extract_tfpeaks via ctypes and return a dict of arrays."""
    lib = _load_lib()

    # Ensure C-contiguous row-major (as the header documents).
    spect = np.ascontiguousarray(spect, dtype=np.float64)
    stimes = np.ascontiguousarray(stimes.ravel(), dtype=np.float64)
    sfreqs = np.ascontiguousarray(sfreqs.ravel(), dtype=np.float64)
    baseline_c = np.ascontiguousarray(baseline.ravel(), dtype=np.float64)

    in_ = ExtractTfpeaksIn(
        spect_ptr=spect.ctypes.data_as(C.POINTER(C.c_double)),
        n_freqs=C.c_size_t(spect.shape[0]),
        n_times=C.c_size_t(spect.shape[1]),
        stimes_ptr=stimes.ctypes.data_as(C.POINTER(C.c_double)),
        sfreqs_ptr=sfreqs.ctypes.data_as(C.POINTER(C.c_double)),
        baseline_ptr=baseline_c.ctypes.data_as(C.POINTER(C.c_double)),
        merge_thresh=C.c_double(merge_thresh),
        max_merges=C.c_double(float("inf")),
        trim_vol_thresh=C.c_double(trim_vol),
        trim_shift_val=C.c_double(0.0),
        segment_num=C.c_double(0.0),
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
    params = {k: float(m[k]) for k in [
        "seg_time", "merge_thresh", "trim_vol",
        "dur_min", "dur_max", "bw_min", "bw_max",
    ]}
    downsample = np.asarray(m["downsample_spect"]).ravel().astype(int)
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

    # Cross-check: C ABI ↔ pydynamo
    if pydyn is not None:
        ia2, ib2, ua2, ub2 = hungarian_match(pydyn, c_abi, tol_time=1e-6, tol_freq=1e-6)
        print(f"\nC ABI ↔ pydynamo Hungarian match (should be near-100%):")
        print(f"  matched:   {len(ia2)} / {len(pydyn['PeakTime'])} pydynamo = "
              f"{len(ia2)/max(len(pydyn['PeakTime']),1)*100:.1f}%")
        if ia2:
            print(f"  {'prop':<12} {'max abs Δ':>12}")
            for p in ["Duration", "Bandwidth", "Height", "Volume"]:
                abs_d = np.max(np.abs(pydyn[p][ia2] - c_abi[p][ib2]))
                print(f"  {p:<12} {abs_d:>12.4e}")

    print("=" * 70)
    print("\nInterpretation:")
    print("  - C ABI ↔ pydynamo should match 100% (same kernels); if not,")
    print("    the bug is in c_api.rs's peak-properties helper.")
    print("  - C ABI ↔ MATLAB: expect ~99% peak overlap + ~0.4% count gap.")
    print("    Larger gaps suggest a real bug in the new c_api extract flow.")


if __name__ == "__main__":
    main()
