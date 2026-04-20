"""Summary plot — port of displaySummaryPlot.m + hypnoplot.m.

Matches MATLAB reference in:
  - layout (portrait 8.5 × 11, exact axes positions)
  - panels: stacked hypnogram + spectrogram + SO-power trace (x-axis shared),
    TF-peak scatter (color=SOphase, size=Volume, cyclic HSV colormap
    rotated by −650), SO-power histogram (gouldian), SO-phase histogram
    (magma)
  - hypnogram: stage-colored filled bands + black step line + artifact row
    with black tick marks
  - colormaps: exact 256-color gouldian and rainbow4 LUTs extracted from
    DYNAM-O's MATLAB files; matplotlib parula for SO-power if requested
  - color scaling: 5th/98th percentile (climscale-equivalent) on the
    spectrogram and per-histogram; non-zero percentile for SO-phase
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.patches import Rectangle

from pydynamo.spectrogram import _mts, _next_pow2


# ---------------------------------------------------------------------------
# MATLAB-compatible colormaps
# ---------------------------------------------------------------------------

_HERE = Path(__file__).parent


def _load_matlab_cmap(name: str) -> ListedColormap:
    lut = np.load(_HERE / f"_cmap_{name}.npy")
    cmap = ListedColormap(lut, name=name)
    if name not in plt.colormaps():
        plt.colormaps.register(cmap=cmap, name=name)
    return cmap


_gouldian = _load_matlab_cmap("gouldian")
_rainbow4 = _load_matlab_cmap("rainbow4")
# Make rainbow4 also handle NaN → gray background for SOPH
_rainbow4_nan = ListedColormap(_rainbow4.colors, name="rainbow4_nan")
_rainbow4_nan.set_bad(color=(0.2, 0.2, 0.2, 1.0))
_gouldian_nan = ListedColormap(_gouldian.colors, name="gouldian_nan")
_gouldian_nan.set_bad(color=(0.2, 0.2, 0.2, 1.0))


# Cyclic HSV — MATLAB `circshift(hsv(4096), -650)`.
def _matlab_cyclic_hsv(n: int = 4096, shift: int = -650) -> ListedColormap:
    import matplotlib.colors as mc
    hues = np.linspace(0.0, 1.0, n, endpoint=False)
    rgb = np.stack([mc.hsv_to_rgb([(h, 1.0, 1.0)]) for h in hues]).reshape(n, 3)
    rgb = np.roll(rgb, shift, axis=0)
    return ListedColormap(rgb, name="matlab_hsv_rot")


_CMAP_PHASE = _matlab_cyclic_hsv()


# Hypnogram palette matching MATLAB hypnoplot.m default_colors (lines 80–86):
#   0 Undef, 1 N3, 2 N2, 3 N1, 4 REM, 5 Wake, 6 Art
_HYP_COLORS = {
    0: (0.9, 0.9, 0.9),
    1: (0.6, 0.6, 1.0),
    2: (0.8, 0.8, 1.0),
    3: (0.8, 1.0, 1.0),
    4: (0.7, 1.0, 0.7),
    5: (1.0, 0.7, 0.7),
    6: (0.6, 0.6, 0.6),
}


# ---------------------------------------------------------------------------
# climscale — match MATLAB's default (ptiles=(5,98), outlier-robust)
# ---------------------------------------------------------------------------

def _matlab_prctile(x: np.ndarray, q):
    """MATLAB-compatible prctile using Hyndman-Fan method #5 (aka 'hazen').
    numpy default is method='linear' (H&F #7) — slightly different at the
    tails (5th / 95th percentile can shift by ~1 rank position)."""
    return np.percentile(x, q, method="hazen")


def _climscale(image: np.ndarray, ptiles=(5.0, 98.0), drop_outliers: bool = True):
    """MATLAB climscale.m port — drop NaN/Inf, optionally drop isoutlier's
    median±3*MAD outliers, then prctile at `ptiles`. Used on spectrogram
    (default drop_outliers=True). Histograms call `_matlab_prctile` directly.
    """
    vals = image[np.isfinite(image)]
    if vals.size == 0:
        return 0.0, 1.0
    if drop_outliers:
        # MATLAB `isoutlier(A)` default = median method:
        # outlier iff |A - median(A)| > 3 * (1.4826 * median(|A - median(A)|)).
        med = float(np.median(vals))
        mad = float(np.median(np.abs(vals - med)))
        if mad > 0:
            keep = np.abs(vals - med) <= 3.0 * 1.4826 * mad
            if keep.any():
                vals = vals[keep]
    lo, hi = _matlab_prctile(vals, list(ptiles))
    if lo == hi:
        hi = lo + 1.0
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# Hypnogram panel — port of hypnoplot.m (colored bands + step + artifact row)
# ---------------------------------------------------------------------------

def _plot_hypnogram(ax, stage_times_hr, stage_vals,
                    artifacts_hr=None, artifacts_mask=None,
                    group_nrem: bool = True):
    """MATLAB-style hypnogram: colored background bands per stage (grouped
    NREM by default), black step line on top, and an artifact row below.

    stage_times/vals: same coord frame as ax x-axis (hours)
    """
    st = np.asarray(stage_times_hr, dtype=float)
    sv = np.asarray(stage_vals, dtype=float).astype(int)
    epoch = 30.0 / 3600.0
    # Close the last epoch
    st_c = np.concatenate([st, [st[-1] + epoch]])
    sv_c = np.concatenate([sv, [sv[-1]]])

    y_min = min(int(sv_c.min()), 1)
    y_max = max(int(sv_c.max()), 5)
    art_y = y_min - 1  # artifact row

    # Draw per-epoch colored bands behind the stepline
    for i in range(len(st_c) - 1):
        s = sv_c[i]
        if s < 0:
            continue
        if group_nrem:
            # NREM (N1/N2/N3) share N3's color (darker purple); REM green; Wake pink.
            color = {
                0: _HYP_COLORS[0],
                1: _HYP_COLORS[1], 2: _HYP_COLORS[1], 3: _HYP_COLORS[1],
                4: _HYP_COLORS[4],
                5: _HYP_COLORS[5],
            }.get(s, _HYP_COLORS.get(s, (1, 1, 1)))
        else:
            color = _HYP_COLORS.get(s, (1, 1, 1))
        ax.add_patch(Rectangle(
            (st_c[i], y_min - 0.3), st_c[i + 1] - st_c[i],
            (y_max + 0.3) - (y_min - 0.3),
            facecolor=color, edgecolor="none", zorder=0,
        ))

    # Simplify the step vector before plotting (MATLAB does this too)
    change = np.concatenate(([True], np.diff(sv_c) != 0))
    st_s = st_c[change]
    sv_s = sv_c[change]
    ax.step(np.concatenate([st_s, [st_c[-1]]]),
            np.concatenate([sv_s, [sv_s[-1]]]),
            where="post", color="k", linewidth=1.0, zorder=3)

    # Artifact row (black tick marks)
    if artifacts_hr is not None and artifacts_mask is not None:
        art = np.asarray(artifacts_mask, dtype=bool).ravel()
        if art.any():
            diff = np.diff(np.concatenate(([0], art.astype(int), [0])))
            starts = np.flatnonzero(diff == 1)
            ends = np.flatnonzero(diff == -1)
            for s, e in zip(starts, ends):
                x0 = artifacts_hr[min(s, len(artifacts_hr) - 1)]
                x1 = artifacts_hr[min(e, len(artifacts_hr) - 1)]
                ax.add_patch(Rectangle(
                    (x0, art_y - 0.2), max(x1 - x0, epoch * 0.1), 0.4,
                    facecolor="k", edgecolor="none", zorder=4,
                ))

    # y-ticks (bottom-to-top on axis = highest y value at top: Wake at top)
    labels_all = {0: "Undef", 1: "N3", 2: "N2", 3: "N1", 4: "REM", 5: "Wake"}
    ticks = list(range(y_min, y_max + 1))
    ticklabels = [labels_all.get(t, "") for t in ticks]
    if artifacts_mask is not None:
        ticks = [art_y] + ticks
        ticklabels = ["Art"] + ticklabels
    ax.set_yticks(ticks)
    ax.set_yticklabels(ticklabels, fontsize=8)
    ax.set_ylim(art_y - 0.5, y_max + 0.3)
    ax.tick_params(axis="x", which="both", labelbottom=False)
    # Clean up spines / grid for a cleaner MATLAB look
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    ax.set_xlim(ax.get_xlim())   # lock in limits for patches


# ---------------------------------------------------------------------------
# Main summary plot
# ---------------------------------------------------------------------------

def summary_plot(
    spect, stimes, sfreqs, artifacts, t_data,
    stage_times, stage_vals, stats, sophs,
    *,
    data=None, fs=None,
    freq_limits: tuple[float, float] = (2.0, 25.0),
    mtm_freq_range: tuple[float, float] = (2.0, 25.0),
    time_range: tuple[float, float] | None = None,
    peak_size_prctiles: tuple[float, float] = (5.0, 95.0),
    soph_clim_prctiles: tuple[float, float] = (5.0, 98.0),
    SOpower_norm_method: str = "p2shift1234",
    hist_peakidx: np.ndarray | None = None,
):
    """Build the DYNAM-O summary figure. Returns a matplotlib Figure."""
    fig = plt.figure(figsize=(8.5, 11), facecolor="white")

    # Axes positions match displaySummaryPlot.m:172-192 (normalized)
    ax_hyp = fig.add_axes([0.10, 0.893, 0.79, 0.076])
    ax_spec = fig.add_axes([0.10, 0.726, 0.79, 0.167])
    ax_sop = fig.add_axes([0.10, 0.670, 0.79, 0.056])
    ax_scat = fig.add_axes([0.10, 0.420, 0.79, 0.200])
    ax_pow = fig.add_axes([0.07, 0.050, 0.36, 0.300])
    ax_ph = fig.add_axes([0.555, 0.050, 0.36, 0.300])

    if time_range is None:
        time_range = (float(stimes[0]), float(stimes[-1]))
    t0_hr = time_range[0] / 3600.0
    t1_hr = time_range[1] / 3600.0

    # ---- Hypnogram ----
    ax_hyp.set_xlim(t0_hr, t1_hr)
    _plot_hypnogram(
        ax_hyp, np.asarray(stage_times) / 3600.0, stage_vals,
        artifacts_hr=np.asarray(t_data) / 3600.0 if artifacts is not None else None,
        artifacts_mask=artifacts,
    )
    ax_hyp.set_title("EEG Spectrogram", fontsize=15, pad=6, fontweight="bold")

    # ---- Spectrogram (display quality: 30 s window, TW=15, K=29) ----
    if data is not None and fs is not None:
        spect_disp, stimes_disp, sfreqs_disp = _mts(
            np.asarray(data, dtype=np.float64).ravel(), float(fs),
            list(mtm_freq_range), 15.0, 29, [30.0, 15.0],
            min_nfft=_next_pow2(float(fs) / 0.1),
            detrend_opt="linear", multiprocess=True, n_jobs=None,
            weighting="unity", plot_on=False, verbose=False, xyflip=False,
        )
        stimes_disp = stimes_disp + (t_data[0] if len(t_data) else 0)
        sfreqs_disp = np.asarray(sfreqs_disp)
    else:
        spect_disp, stimes_disp, sfreqs_disp = spect, stimes, sfreqs

    with np.errstate(divide="ignore", invalid="ignore"):
        spect_db = 10 * np.log10(np.where(spect_disp > 0, spect_disp, np.nan))
    sel_t = (stimes_disp >= time_range[0]) & (stimes_disp <= time_range[1])
    sel_f = (sfreqs_disp >= mtm_freq_range[0]) & (sfreqs_disp <= mtm_freq_range[1])
    vmin, vmax = _climscale(spect_db[np.ix_(sel_f, sel_t)])
    im_s = ax_spec.imshow(
        spect_db[np.ix_(sel_f, sel_t)], origin="lower", aspect="auto",
        extent=(stimes_disp[sel_t][0] / 3600.0,
                stimes_disp[sel_t][-1] / 3600.0,
                sfreqs_disp[sel_f][0], sfreqs_disp[sel_f][-1]),
        cmap=_rainbow4, vmin=vmin, vmax=vmax,
    )
    ax_spec.set_ylim(freq_limits)
    ax_spec.set_ylabel("Frequency (Hz)")
    ax_spec.tick_params(axis="x", labelbottom=False)
    cbar = fig.colorbar(im_s, ax=ax_spec, pad=0.01, fraction=0.03, aspect=15)
    cbar.set_label("PSD (dB)", rotation=-90, va="bottom", fontsize=9)

    # ---- SO-power trace ----
    sop = np.asarray(sophs.SOpower_norm, dtype=float)
    sop_t = np.asarray(sophs.SOpower_times, dtype=float) / 3600.0
    ax_sop.plot(sop_t, sop, color="tab:blue", linewidth=0.8)
    finite = sop[np.isfinite(sop)]
    if finite.size:
        lo, hi = float(finite.min()), float(finite.max())
        pad = 0.1 * max(abs(lo), abs(hi), 1.0)
        ax_sop.set_ylim(lo - pad, hi + pad)
        ax_sop.set_yticks([round(lo, 2), round((lo + hi) / 2, 2), round(hi, 2)])
    ax_sop.set_ylabel({
        "percent": "%SOP", "proportion": "SO Prop.",
    }.get(SOpower_norm_method, "SOP (dB)"), fontsize=9)
    ax_sop.tick_params(axis="x", labelbottom=False)
    ax_sop.tick_params(axis="y", labelsize=8)

    # ---- TF-peak scatter (NREM-only peaks, matching SOPH selection) ----
    # Mirror MATLAB displaySummaryPlot.m:269: scatter the same subset that
    # contributes to the SOPH histograms (hist_peakidx = SOPH_stages filter).
    if stats is not None and not stats.empty and "SOphase" in stats.columns:
        if hist_peakidx is not None:
            scatter_stats = stats[np.asarray(hist_peakidx, dtype=bool)].reset_index(drop=True)
        else:
            scatter_stats = stats
        vol = scatter_stats["Volume"].to_numpy()
        phase = scatter_stats["SOphase"].to_numpy()
        # MATLAB displaySummaryPlot.m:273-276 —
        #   pmin = prctile(..., peak_size_prctiles(1));
        #   pmax = prctile(..., peak_size_prctiles(2));
        # Use MATLAB-matching hazen method.
        vol_finite = vol[np.isfinite(vol)]
        if vol_finite.size:
            pmin = _matlab_prctile(vol_finite, float(peak_size_prctiles[0]))
            pmax = _matlab_prctile(vol_finite, float(peak_size_prctiles[1]))
        else:
            pmin = pmax = 1.0
        size = vol / max(pmin, 1e-9) * 0.5
        size[vol > pmax] = np.nan
        keep = np.isfinite(size) & np.isfinite(phase)
        ax_scat.scatter(
            scatter_stats["PeakTime"].to_numpy()[keep] / 3600.0,
            scatter_stats["PeakFrequency"].to_numpy()[keep],
            s=np.clip(size[keep], 0.1, 60),
            c=phase[keep], cmap=_CMAP_PHASE,
            vmin=-np.pi, vmax=np.pi, edgecolors="none",
        )
        sm = plt.cm.ScalarMappable(
            cmap=_CMAP_PHASE,
            norm=plt.Normalize(vmin=-np.pi, vmax=np.pi),
        )
        cbar = fig.colorbar(sm, ax=ax_scat, pad=0.01, fraction=0.03, aspect=15)
        cbar.set_label("Phase (rad)", rotation=-90, va="bottom", fontsize=9)
        cbar.set_ticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
        cbar.set_ticklabels(["-π", "-π/2", "0", "π/2", "π"])
    ax_scat.set_ylim(freq_limits)
    ax_scat.set_ylabel("Frequency (Hz)")
    ax_scat.set_xlabel("Time (hr)")
    ax_scat.set_title("Extracted Time-Frequency Peaks", fontsize=13, pad=4)

    # Link x on all time-domain panels
    for ax in (ax_hyp, ax_spec, ax_sop, ax_scat):
        ax.set_xlim(t0_hr, t1_hr)

    # ---- SO-power histogram ----
    pow_mat = np.asarray(sophs.SOpower_mat, dtype=float)   # (B, F)
    pow_bins = np.asarray(sophs.SOpower_bins, dtype=float)
    f_bins = np.asarray(sophs.freq_bins, dtype=float)
    f_sel = (f_bins >= freq_limits[0]) & (f_bins <= freq_limits[1])
    # MATLAB displaySummaryPlot.m:307 —
    #   tmp_mat = SOpower_mat(:, freq_in_range);
    #   c_ptiles = prctile(tmp_mat(:), SOPH_clim_prctiles);
    # No outlier removal, no zero filter.
    sel_mat = pow_mat[:, f_sel]
    finite_vals = sel_mat[np.isfinite(sel_mat)]
    if finite_vals.size:
        lo_p, hi_p = _matlab_prctile(finite_vals, list(soph_clim_prctiles))
    else:
        lo_p, hi_p = 0.0, 1.0
    # Dark background so NaN bins render gray (matches MATLAB's axes bg)
    ax_pow.set_facecolor((0.2, 0.2, 0.2))
    im_p = ax_pow.imshow(
        pow_mat.T, origin="lower", aspect="auto",
        extent=(pow_bins[0], pow_bins[-1], f_bins[0], f_bins[-1]),
        cmap=_gouldian_nan, vmin=lo_p, vmax=hi_p,
    )
    ax_pow.set_ylim(freq_limits)
    ax_pow.set_xlabel({
        "percent": "% SO-Power", "proportion": "SO-Power Proportion",
    }.get(SOpower_norm_method, "SO-Power (dB)"))
    ax_pow.set_ylabel("Frequency (Hz)")
    ax_pow.set_title("SO-Power Histogram", fontsize=13, pad=4)
    cbar = fig.colorbar(im_p, ax=ax_pow, pad=0.01, fraction=0.04, aspect=20)
    cbar.set_label("Density\n(peaks/min in bin)", rotation=-90, va="bottom",
                   fontsize=9)

    # ---- SO-phase histogram ----
    ph_mat = np.asarray(sophs.SOphase_mat, dtype=float)
    ph_bins = np.asarray(sophs.SOphase_bins, dtype=float)
    # MATLAB displaySummaryPlot.m:342 —
    #   tmp_mat = SOphase_mat(:, freq_in_range);
    #   c_ptiles = prctile(tmp_mat(tmp_mat(:)~=0), SOPH_clim_prctiles);
    # Excludes zero entries (unmodeled bins) but keeps NaN outside finite
    # filter (matches MATLAB's prctile which ignores NaN).
    sel_ph = ph_mat[:, f_sel]
    nonzero = sel_ph[np.isfinite(sel_ph) & (sel_ph != 0)]
    if nonzero.size:
        lo_ph, hi_ph = _matlab_prctile(nonzero, list(soph_clim_prctiles))
    else:
        lo_ph, hi_ph = 0.0, 1.0
    im_ph = ax_ph.imshow(
        ph_mat.T, origin="lower", aspect="auto",
        extent=(ph_bins[0], ph_bins[-1], f_bins[0], f_bins[-1]),
        cmap="magma", vmin=lo_ph, vmax=hi_ph,
    )
    ax_ph.set_ylim(freq_limits)
    ax_ph.set_xlabel("SO-Phase (rad)")
    ax_ph.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    ax_ph.set_xticklabels(["-π", "-π/2", "0", "π/2", "π"])
    ax_ph.set_title("SO-Phase Histogram", fontsize=13, pad=4)
    cbar = fig.colorbar(im_ph, ax=ax_ph, pad=0.01, fraction=0.04, aspect=20)
    cbar.set_label("Proportion", rotation=-90, va="bottom", fontsize=9)

    return fig
