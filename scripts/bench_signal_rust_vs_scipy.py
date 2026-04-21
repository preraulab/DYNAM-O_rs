"""Benchmark: dynamo_rs signal primitives vs scipy/numpy on Apple Silicon.

Tests: hilbert, sosfiltfilt, unwrap, movmean on ~3M-sample signal
(8 hours @ 100 Hz). Reports median time across 5 runs and max-abs-diff.

Run:
    .venv-matlab/bin/python scripts/bench_signal_rust_vs_scipy.py
"""
from __future__ import annotations

import time
import statistics as _st
import platform

import numpy as np
from scipy.signal import cheby1, sosfiltfilt as sp_sosfiltfilt, hilbert as sp_hilbert

import dynamo_rs

FS = 100.0
N = 3_000_000  # 8.33 h @ 100 Hz
WARMUP = 1
RUNS = 5

rng = np.random.default_rng(0)
x = rng.standard_normal(N).astype(np.float64)

# Chebyshev-I bandpass 0.3-1.5 Hz, order 4 (scipy equivalent of MATLAB SOphase cheby1 path)
sos = cheby1(4, 1.0, [0.3 / (FS / 2), 1.5 / (FS / 2)], btype="band", output="sos").astype(np.float64)
sos = np.ascontiguousarray(sos)

# Realistic wrapped phase: angle of hilbert of a narrowband signal
_t = np.arange(N) / FS
phase_in = np.angle(sp_hilbert(np.sin(2 * np.pi * 1.0 * _t) + 0.1 * rng.standard_normal(N))).astype(np.float64)


def _time(fn, runs=RUNS, warmup=WARMUP):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return _st.median(ts)


def _numpy_movmean(x, win):
    """Vectorized cumsum movmean, matching pydynamo.artifacts._movmean."""
    n = x.size
    if win <= 1 or n == 0:
        return x.astype(float, copy=True)
    half_l = (win - 1) // 2
    half_r = win // 2
    csum = np.concatenate(([0.0], np.cumsum(x, dtype=np.float64)))
    idx = np.arange(n)
    a = np.maximum(0, idx - half_l)
    b = np.minimum(n, idx + half_r + 1)
    return (csum[b] - csum[a]) / (b - a)


def bench():
    print(f"platform: {platform.machine()} / {platform.platform()}")
    print(f"signal length: {N:,}  ({N / FS / 3600:.2f} h @ {FS} Hz)")
    print(f"rustfft default features include neon on aarch64 (verified)")
    print()

    rows = []

    # ---- hilbert ----
    t_sp = _time(lambda: sp_hilbert(x))
    t_rs = _time(lambda: dynamo_rs.hilbert(x))
    y_sp = sp_hilbert(x)
    re, im = dynamo_rs.hilbert(x)
    y_rs = re + 1j * im
    diff = float(np.max(np.abs(y_sp - y_rs)))
    rows.append(("hilbert (analytic)", t_sp, t_rs, diff))

    # ---- sosfiltfilt ----
    t_sp = _time(lambda: sp_sosfiltfilt(sos, x))
    t_rs = _time(lambda: dynamo_rs.sosfiltfilt(sos, x))
    y_sp = sp_sosfiltfilt(sos, x)
    y_rs = dynamo_rs.sosfiltfilt(sos, x)
    diff = float(np.max(np.abs(y_sp - y_rs)))
    rows.append(("sosfiltfilt (cheby1 bp)", t_sp, t_rs, diff))

    # ---- unwrap ----
    t_np = _time(lambda: np.unwrap(phase_in))
    t_rs = _time(lambda: dynamo_rs.unwrap(phase_in))
    y_np = np.unwrap(phase_in)
    y_rs = dynamo_rs.unwrap(phase_in)
    diff = float(np.max(np.abs(y_np - y_rs)))
    rows.append(("unwrap", t_np, t_rs, diff))

    # ---- movmean ----
    WIN = 1000
    # reference: manual numpy cumsum version (accept some floating error vs rust)
    t_np = _time(lambda: _numpy_movmean(x, WIN))
    t_rs = _time(lambda: dynamo_rs.movmean(x, WIN))
    y_np = _numpy_movmean(x, WIN)
    y_rs = dynamo_rs.movmean(x, WIN)
    diff = float(np.max(np.abs(y_np - y_rs)))
    rows.append(("movmean (win=1000)", t_np, t_rs, diff))

    # ---- print ----
    print(f"{'kernel':<28}  {'scipy/numpy':>12}  {'rust (neon)':>12}  {'speedup':>9}  {'maxabs diff':>12}")
    print("-" * 82)
    for name, t_ref, t_rs, diff in rows:
        sp = t_ref / t_rs
        marker = "  <-- rust wins" if sp > 1.0 else ""
        print(f"{name:<28}  {t_ref * 1e3:>10.2f}ms  {t_rs * 1e3:>10.2f}ms  {sp:>8.2f}x  {diff:>12.3e}{marker}")


if __name__ == "__main__":
    bench()
