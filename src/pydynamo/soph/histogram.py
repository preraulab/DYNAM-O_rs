"""2D TF-peak × C-metric histogram — port of TFPeakHistogram.m + create_bins.m.

Used by both SOpowerHistogram (C = SO-power, non-circular) and
SOphaseHistogram (C = SO-phase, circular with wrap at ±π).

Bit-identical to the MATLAB reference when given the same peak positions,
C-metric timeseries, and stage/validity masks. Verified in
tests/test_soph_from_matlab_peaks.py.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

try:
    import dynamo_rs as _dynamo_rs
    _HAS_RUST = hasattr(_dynamo_rs, "tfpeak_histogram")
except ImportError:
    _dynamo_rs = None
    _HAS_RUST = False


BinMethod = Literal["full", "partial", "extend", "full_extend", "full extend"]


def create_bins(
    bin_range: tuple[float, float],
    bin_width: float,
    bin_step: float,
    bin_method: BinMethod = "full",
) -> tuple[np.ndarray, np.ndarray]:
    """Port of create_bins.m.

    Returns (bin_edges, bin_centers) where bin_edges has shape (2, N):
        edges[0, :] = left edges
        edges[1, :] = right edges
    """
    lo, hi = float(bin_range[0]), float(bin_range[1])
    w, s = float(bin_width), float(bin_step)

    method = bin_method.lower().replace(" ", "_")

    if method == "full":
        new_lo = lo + w / 2
        new_hi = hi - w / 2
        centers = np.arange(new_lo, new_hi + 0.5 * s, s)
        # Drop any center that would exceed new_hi due to FP
        centers = centers[centers <= new_hi + 1e-12]
        edges = np.vstack([centers - w / 2, centers + w / 2])

    elif method == "partial":
        centers = np.arange(lo, hi + 0.5 * s, s)
        centers = centers[centers <= hi + 1e-12]
        edges = np.vstack([centers - w / 2, centers + w / 2])
        edges = np.clip(edges, lo, hi)

    elif method in ("full_extend", "extend"):
        shift = np.floor((w / 2) / s) * s
        new_lo = lo - shift
        new_hi = hi + shift
        centers = np.arange(new_lo + w / 2, new_hi - w / 2 + 0.5 * s, s)
        centers = centers[centers <= new_hi - w / 2 + 1e-12]
        edges = np.vstack([centers - w / 2, centers + w / 2])
    else:
        raise ValueError(f"Unknown bin_method {bin_method!r}")

    return edges, centers


def tfpeak_histogram(
    c_metric: np.ndarray,
    c_stages: np.ndarray,
    c_dt: float,
    c_valid: np.ndarray,
    c_valid_allstages: np.ndarray,
    peak_freqs: np.ndarray,
    peak_c: np.ndarray,
    *,
    circular: bool = False,
    circular_bounds: tuple[float, float] = (-np.pi, np.pi),
    freq_range: tuple[float, float] = (0.0, 30.0),
    freq_binsizestep: tuple[float, float] = (1.0, 0.2),
    c_range: tuple[float, float] | None = None,
    c_binsizestep: tuple[float, float] | None = None,
    norm_dim: int = 0,
    compute_rate: bool = True,
    min_time_in_bin: float = 0.0,      # minutes
    min_peak_at_freq: int = 0,
) -> dict:
    """Build the 2D C-bin × freq-bin histogram.

    Returns a dict with:
        c_mat            (num_Cbins, num_freqbins)
        freq_cbins       (num_freqbins,)
        c_cbins          (num_Cbins,)
        time_in_bin      (num_Cbins, 5)   minutes per stage 1..5
        prop_in_bin      (num_Cbins, 5)
        peak_at_freq     (num_freqbins,)
    """
    c_metric = np.asarray(c_metric, dtype=float).ravel()
    c_stages = np.asarray(c_stages).ravel()
    c_valid = np.asarray(c_valid, dtype=bool).ravel()
    c_valid_allstages = np.asarray(c_valid_allstages, dtype=bool).ravel()
    peak_freqs = np.asarray(peak_freqs, dtype=float).ravel()
    peak_c = np.asarray(peak_c, dtype=float).ravel()

    # Bin edges
    freq_edges, freq_cbins = create_bins(freq_range, freq_binsizestep[0],
                                          freq_binsizestep[1], "partial")
    if circular:
        c_edges, c_cbins = create_bins(c_range, c_binsizestep[0],
                                        c_binsizestep[1], "extend")
    else:
        c_edges, c_cbins = create_bins(c_range, c_binsizestep[0],
                                        c_binsizestep[1], "partial")

    num_fbins = freq_cbins.size
    num_cbins = c_cbins.size

    # ---- Rust fast path ---------------------------------------------------
    if _HAS_RUST:
        cm = np.ascontiguousarray(c_metric, dtype=np.float64)
        cs = np.ascontiguousarray(np.asarray(c_stages, dtype=float), dtype=np.float64)
        cv = np.ascontiguousarray(c_valid, dtype=bool)
        cva = np.ascontiguousarray(c_valid_allstages, dtype=bool)
        pf = np.ascontiguousarray(peak_freqs, dtype=np.float64)
        pc = np.ascontiguousarray(peak_c, dtype=np.float64)
        fe = np.ascontiguousarray(freq_edges, dtype=np.float64)
        ce = np.ascontiguousarray(c_edges, dtype=np.float64)
        out = _dynamo_rs.tfpeak_histogram(
            cm, cs, float(c_dt), cv, cva, pf, pc, fe, ce,
            bool(circular),
            (float(circular_bounds[0]), float(circular_bounds[1])),
            int(norm_dim), bool(compute_rate),
            float(min_time_in_bin), int(min_peak_at_freq),
        )
        return {
            "c_mat": out["c_mat"],
            "freq_cbins": freq_cbins,
            "c_cbins": c_cbins,
            "time_in_bin": out["time_in_bin"],
            "prop_in_bin": out["prop_in_bin"],
            "peak_at_freq": out["peak_at_freq"],
        }
    # ---- Python fallback --------------------------------------------------

    # [N_peaks × num_fbins] mask: peak p in freq bin f?
    pf = peak_freqs[:, None]
    all_infreqbin = (pf >= freq_edges[0][None, :]) & (pf < freq_edges[1][None, :])

    c_mat = np.full((num_cbins, num_fbins), np.nan, dtype=float)
    time_in_bin = np.zeros((num_cbins, 5), dtype=float)
    prop_in_bin = np.zeros((num_cbins, 5), dtype=float)

    compute_tib = compute_rate or (min_time_in_bin > 0)

    # Precompute per-stage validity: (N_times, 5) logical.
    # Stages are labelled 1..5 per DYNAM-O convention.
    if compute_tib:
        stage_valid_masks = np.zeros((c_stages.size, 5), dtype=bool)
        for k in range(1, 6):
            stage_valid_masks[:, k - 1] = (c_stages == k) & c_valid

    low_b, high_b = float(circular_bounds[0]), float(circular_bounds[1])
    crange = high_b - low_b

    for s in range(num_cbins):
        lo_e, hi_e = c_edges[0, s], c_edges[1, s]

        if circular and lo_e <= low_b:
            wrap_lo = lo_e + crange
            tib_inds = (c_metric >= wrap_lo) | (c_metric < hi_e)
            inc_inds = (peak_c >= wrap_lo) | (peak_c < hi_e)
        elif circular and hi_e >= high_b:
            wrap_hi = hi_e - crange
            tib_inds = (c_metric < wrap_hi) | (c_metric >= lo_e)
            inc_inds = (peak_c < wrap_hi) | (peak_c >= lo_e)
        else:
            tib_inds = (c_metric >= lo_e) & (c_metric < hi_e)
            inc_inds = (peak_c >= lo_e) & (peak_c < hi_e)

        if compute_tib:
            # minutes per stage
            # (N,5) & (N,1) → (N,5); sum along axis=0; * dt / 60
            tib_per_stage = np.sum(tib_inds[:, None] & stage_valid_masks, axis=0) * c_dt / 60.0
            time_in_bin[s, :] = tib_per_stage
            tib_allstages = np.sum(tib_inds & c_valid_allstages) * c_dt / 60.0
            if tib_allstages > 0:
                prop_in_bin[s, :] = tib_per_stage / tib_allstages
            if tib_per_stage.sum() < min_time_in_bin:
                continue

        if inc_inds.any():
            counts = np.sum(inc_inds[:, None] & all_infreqbin, axis=0).astype(float)
        else:
            counts = np.zeros(num_fbins, dtype=float)

        c_mat[s, :] = counts
        if compute_rate and time_in_bin[s, :].sum() > 0:
            c_mat[s, :] = c_mat[s, :] / time_in_bin[s, :].sum()

    peak_at_freq = np.sum(all_infreqbin, axis=0).astype(float)
    if min_peak_at_freq > 0:
        c_mat[:, peak_at_freq < min_peak_at_freq] = np.nan

    if norm_dim:
        axis = norm_dim - 1  # MATLAB 1-based
        dim_sum = np.nansum(c_mat, axis=axis, keepdims=True)
        dim_sum = np.where(dim_sum == 0, 1.0, dim_sum)
        c_mat = c_mat / dim_sum

    return {
        "c_mat": c_mat,
        "freq_cbins": freq_cbins,
        "c_cbins": c_cbins,
        "time_in_bin": time_in_bin,
        "prop_in_bin": prop_in_bin,
        "peak_at_freq": peak_at_freq,
    }


def so_power_histogram(
    peak_freqs: np.ndarray,
    peak_times: np.ndarray,
    peak_stages: np.ndarray,
    so_power: np.ndarray,
    so_power_times: np.ndarray,
    so_power_stages: np.ndarray,
    *,
    time_range: tuple[float, float],
    soph_stages: tuple[int, ...] = (1, 2, 3),
    freq_range: tuple[float, float] = (0.0, 30.0),
    freq_binsizestep: tuple[float, float] = (1.0, 0.2),
    so_range: tuple[float, float] | None = None,
    so_binsizestep: tuple[float, float] | None = None,
    min_time_in_bin: float = 10.0,
    compute_rate: bool = True,
    norm_dim: int = 0,
) -> dict:
    """High-level SO-power histogram wrapper. Mirrors SOpowerHistogram.m."""
    so_power = np.asarray(so_power, dtype=float).ravel()
    so_power_times = np.asarray(so_power_times, dtype=float).ravel()
    so_power_stages = np.asarray(so_power_stages).ravel()
    peak_freqs = np.asarray(peak_freqs, dtype=float).ravel()
    peak_times = np.asarray(peak_times, dtype=float).ravel()
    peak_stages = np.asarray(peak_stages).ravel()

    dt = float(so_power_times[1] - so_power_times[0])

    # Interpolate SOpower onto each peak time (linear, edge-padded by one dt)
    xp = np.concatenate(([so_power_times[0] - dt], so_power_times, [so_power_times[-1] + dt]))
    fp = np.concatenate(([so_power[0]], so_power, [so_power[-1]]))
    # NaN handling in np.interp: np.interp propagates through the xp sequence
    # without nan-awareness; we want NaN where the peak lands inside a NaN run.
    peak_so = np.interp(peak_times, xp, fp, left=np.nan, right=np.nan)
    # Also set NaN where the nearest sample in so_power is NaN.
    idx = np.clip(np.searchsorted(so_power_times, peak_times), 0, so_power.size - 1)
    left = np.clip(idx - 1, 0, so_power.size - 1)
    use_left = np.abs(so_power_times[left] - peak_times) < np.abs(so_power_times[idx] - peak_times)
    nearest = np.where(use_left, left, idx)
    peak_so = np.where(np.isnan(so_power[nearest]), np.nan, peak_so)

    # Valid peak selection
    stage_ok = np.isin(peak_stages, soph_stages)
    time_ok = (peak_times >= time_range[0]) & (peak_times <= time_range[1])
    sop_ok = ~np.isnan(peak_so)
    peak_sel = stage_ok & time_ok & sop_ok

    # Valid SOpower columns
    so_stage_ok = np.isin(so_power_stages, soph_stages)
    so_nan_ok = ~np.isnan(so_power)
    so_time_ok = (so_power_times >= time_range[0]) & (so_power_times <= time_range[1])
    so_valid = so_stage_ok & so_nan_ok & so_time_ok
    so_valid_all = so_nan_ok & so_time_ok

    assert so_valid.any(), "No valid SO-power overlapping data."

    if so_range is None:
        so_range = (float(so_power[so_valid].min()), float(so_power[so_valid].max()))
    if so_binsizestep is None:
        so_binsizestep = ((so_range[1] - so_range[0]) / 10,
                          (so_range[1] - so_range[0]) / 100)

    out = tfpeak_histogram(
        so_power, so_power_stages, dt, so_valid, so_valid_all,
        peak_freqs[peak_sel], peak_so[peak_sel],
        circular=False,
        freq_range=freq_range, freq_binsizestep=freq_binsizestep,
        c_range=so_range, c_binsizestep=so_binsizestep,
        norm_dim=norm_dim, compute_rate=compute_rate,
        min_time_in_bin=min_time_in_bin,
    )
    out["peak_so"] = peak_so
    out["peak_selection_inds"] = peak_sel
    return out


def so_phase_histogram(
    peak_freqs: np.ndarray,
    peak_times: np.ndarray,
    peak_stages: np.ndarray,
    so_phase: np.ndarray,            # UNWRAPPED; we wrap internally
    so_phase_times: np.ndarray,
    so_phase_stages: np.ndarray,
    *,
    time_range: tuple[float, float],
    soph_stages: tuple[int, ...] = (1, 2, 3),
    freq_range: tuple[float, float] = (0.0, 30.0),
    freq_binsizestep: tuple[float, float] = (1.0, 0.2),
    so_range: tuple[float, float] = (-np.pi, np.pi),
    so_binsizestep: tuple[float, float] = (2 * np.pi / 5, 2 * np.pi / 100),
    min_peak_at_freq: int = 0,     # MATLAB SOpowerphasehist_opts default
    compute_rate: bool = True,
    norm_dim: int = 1,
) -> dict:
    """High-level SO-phase histogram wrapper. Mirrors SOphaseHistogram.m."""
    so_phase = np.asarray(so_phase, dtype=float).ravel()
    so_phase_times = np.asarray(so_phase_times, dtype=float).ravel()
    so_phase_stages = np.asarray(so_phase_stages).ravel()
    peak_freqs = np.asarray(peak_freqs, dtype=float).ravel()
    peak_times = np.asarray(peak_times, dtype=float).ravel()
    peak_stages = np.asarray(peak_stages).ravel()

    dt = float(so_phase_times[1] - so_phase_times[0])

    xp = np.concatenate(([so_phase_times[0] - dt], so_phase_times,
                         [so_phase_times[-1] + dt]))
    fp = np.concatenate(([so_phase[0]], so_phase, [so_phase[-1]]))
    peak_phase = np.interp(peak_times, xp, fp, left=np.nan, right=np.nan)

    # Mark NaN where nearest SOphase sample is NaN (excluded times)
    idx = np.clip(np.searchsorted(so_phase_times, peak_times), 0, so_phase.size - 1)
    left = np.clip(idx - 1, 0, so_phase.size - 1)
    use_left = np.abs(so_phase_times[left] - peak_times) < np.abs(so_phase_times[idx] - peak_times)
    nearest = np.where(use_left, left, idx)
    peak_phase = np.where(np.isnan(so_phase[nearest]), np.nan, peak_phase)

    # Wrap to [-π, π]
    peak_phase = _wrap_to_pi(peak_phase)
    so_phase_wrapped = _wrap_to_pi(so_phase)

    stage_ok = np.isin(peak_stages, soph_stages)
    time_ok = (peak_times >= time_range[0]) & (peak_times <= time_range[1])
    ph_ok = ~np.isnan(peak_phase)
    peak_sel = stage_ok & time_ok & ph_ok

    so_stage_ok = np.isin(so_phase_stages, soph_stages)
    so_nan_ok = ~np.isnan(so_phase_wrapped)
    so_time_ok = (so_phase_times >= time_range[0]) & (so_phase_times <= time_range[1])
    so_valid = so_stage_ok & so_nan_ok & so_time_ok
    so_valid_all = so_nan_ok & so_time_ok

    out = tfpeak_histogram(
        so_phase_wrapped, so_phase_stages, dt, so_valid, so_valid_all,
        peak_freqs[peak_sel], peak_phase[peak_sel],
        circular=True, circular_bounds=so_range,
        freq_range=freq_range, freq_binsizestep=freq_binsizestep,
        c_range=so_range, c_binsizestep=so_binsizestep,
        norm_dim=norm_dim, compute_rate=compute_rate,
        min_peak_at_freq=min_peak_at_freq,
    )
    out["peak_so"] = peak_phase
    out["peak_selection_inds"] = peak_sel
    return out


def _wrap_to_pi(x: np.ndarray) -> np.ndarray:
    """MATLAB wrapToPi: map to (-π, π]. NaN preserved."""
    # MATLAB: y = mod(x + pi, 2*pi) - pi; then y(y == -pi & x > 0) = pi
    x = np.asarray(x, dtype=float)
    y = np.mod(x + np.pi, 2 * np.pi) - np.pi
    # MATLAB's +pi boundary: when y == -pi and x > 0, force to +pi
    y = np.where((y == -np.pi) & (x > 0), np.pi, y)
    return y
