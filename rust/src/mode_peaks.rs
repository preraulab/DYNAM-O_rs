//! Associate TF-peaks with parametric-fit modes and summarize their
//! properties.
//!
//! A mode is the single-mode envelope of the parametric fit — a rotated
//! Gaussian over (frequency × SO-power) for power, or a Gaussian (freq) ×
//! von Mises (phase) for phase. A peak "belongs" to a mode when its
//! `(freq, SO-feature)` location falls inside the mode's assignment
//! contour (default p = 0.95): a `p`-mass ellipse on the power axis, a
//! `(1 - p)` relative-height contour with κ-dependent enclosed mass on
//! the phase axis. From the member peaks we compute per-mode
//! medians of the TF-peak properties (the "peak amplitude/duration/…" the
//! Mode Scatter surfaces).
//!
//! # Containment math (derived from the verbatim model source —
//! `rotGauss.rs` / `vm_gauss.rs`, mirrored in MATLAB `rotGauss.m` /
//! `vmGauss.m`)
//!
//! The mode envelope is evaluated **normalized to its own peak**
//! (`model/amp`, background excluded). A peak at `(F, s)` is inside the
//! `p`-region iff `model/amp ≥ 1 − p`, i.e. `Q ≤ −ln(1 − p)`.
//!
//! * Power (rotGauss, ½ form), `θ = Theta`:
//!   ```text
//!   u =  (F-FreqMean)·cosθ + (s-SOpowerMean)·sinθ
//!   v = -(F-FreqMean)·sinθ + (s-SOpowerMean)·cosθ
//!   Q = ½·[ (u/FreqStd)² + (v/SOpowerStd)² ]
//!   ```
//!   `Q ≤ −ln(1−p)` is exactly the ellipse enclosing fraction `p` of the
//!   fitted Gaussian's volume (`−ln(1−p) = χ²₂(p)/2`; p=0.95 → ln 20 ≈ 2.996).
//!
//! * Phase (Gaussian × von Mises), `k = 1/SOphaseStd²`:
//!   ```text
//!   Δ = wrap_to_pi( s - SOphaseMean + (F-FreqMean)·sinθ )
//!   Q = ½·((F-FreqMean)/FreqStd)² + k·(1 − cosΔ)
//!   ```
//!   `Q ≤ −ln(1−p)` is a `(1−p)` relative-height contour of the envelope,
//!   not generally a `p`-mass region: the enclosed mass is κ-dependent
//!   (exact on the Gaussian freq axis; → p mass as κ grows on the
//!   circular axis).
//!
//! The ½ sits on the GAUSSIAN terms only, never on `k·(1−cosΔ)`. The von
//! Mises factor is its own small-angle Gaussian and already carries the
//! half intrinsically, so halving it too would double the effective phase
//! concentration. Equivalently: do NOT "simplify" this by dropping the ½
//! and doubling the threshold to `−2·ln(1−p)` — that is right on the power
//! axis and wrong on the phase axis.
//!
//! The invariant that keeps this self-checking is `Q = −ln(model/amp)` on
//! both axes, which stays true through any future kernel change.
//!
//! Widths on both axes are true standard deviations of the ½-form kernels.
//! The ½ in the kernels and the ½ in `Q` must stay together: a kernel that
//! carries the ½ paired with a `Q` that omits it drops containment from 95%
//! to ~77.7% with no error raised anywhere.
//!
//! # Assignment rules (`paramfit.peak_assign`)
//!
//! [`assign_members`] selects each mode's member peaks under one of four
//! forms of the `peak_assign` setting (see `settings::PeakAssign`):
//! a probability-mass contour `Q ≤ −ln(1−p)`, a σ-radius contour
//! `Q ≤ n²/2`, the **background** rule (member iff the mode's density
//! `amp·e^{−Q}` at the peak is at least the fitted background density
//! there — the tilted plane `xxx·P + yyy·F + zzz` evaluated per peak),
//! or **argmax** (exclusive: each peak goes to the densest component
//! among {background, modes}). Densities use the fit-internal `amp` and
//! background — same LM fit of the same histogram, so units cancel. Both
//! density rules operate within each mode's footprint ellipse (default
//! 1.5σ, spellable as `"argmax <n> sigma"`): outside it a mode neither
//! claims a peak nor blocks another mode from claiming it.
//!
//! The phase axis always uses a numeric contour (the configured value,
//! or the 0.95-p fallback under background/argmax): phase `Density` is
//! an empirical row-normalized model height that already includes the
//! background offset, while the stored phase background is
//! pre-normalization — a density comparison there is not well-defined
//! without re-deriving the per-row normalizers.

/// Per-mode `[amp/Density, FreqMean, FreqStd, SO*Mean, SO*Std, Theta]` —
/// the emitted 6-param layout. `amp` (index 0) is unused for containment.
pub const MODE_FMEAN: usize = 1;
pub const MODE_FSTD: usize = 2;
pub const MODE_SOMEAN: usize = 3;
pub const MODE_SOSTD: usize = 4;
pub const MODE_THETA: usize = 5;

use crate::paramfit::opts::{PeakAssign, PEAK_ASSIGN_PHASE_FALLBACK};
use crate::peak_assign::wrap_to_pi;

/// Mahalanobis-style quadratic `Q` for a peak at `(f, so_feat)` relative to
/// a mode. Returns `f64::INFINITY` for degenerate modes / non-finite inputs
/// so the caller's `Q <= threshold` test fails safely.
#[inline]
pub fn mode_quadratic(
    is_phase: bool,
    fmean: f64,
    fstd: f64,
    so_mean: f64,
    so_std: f64,
    theta: f64,
    f: f64,
    so_feat: f64,
) -> f64 {
    let valid = f.is_finite()
        && so_feat.is_finite()
        && fstd.is_finite()
        && fstd > 0.0
        && so_std.is_finite()
        && so_std > 0.0
        && fmean.is_finite()
        && so_mean.is_finite()
        && theta.is_finite();
    if !valid {
        return f64::INFINITY;
    }
    let df = f - fmean;
    if is_phase {
        // ½ on the Gaussian frequency term only. `k*(1-cosΔ)` is already the
        // von Mises negative-log-ratio and must not be halved.
        let k = 1.0 / (so_std * so_std);
        let delta = wrap_to_pi(so_feat - so_mean + df * theta.sin());
        0.5 * (df / fstd).powi(2) + k * (1.0 - delta.cos())
    } else {
        let (s, c) = theta.sin_cos();
        let dp = so_feat - so_mean;
        let u = df * c + dp * s;
        let v = -df * s + dp * c;
        0.5 * ((u / fstd).powi(2) + (v / so_std).powi(2))
    }
}

/// `-ln(1 - prob)` — the `Q` threshold for the `prob`-confidence region.
/// Clamped to a sane open interval; returns `NaN` for invalid `prob` so
/// containment fails closed.
#[inline]
pub fn q_threshold(prob: f64) -> f64 {
    if !(prob > 0.0 && prob < 1.0) {
        return f64::NAN;
    }
    -(1.0 - prob).ln()
}

/// Whether a peak at `(f, so_feat)` lies inside the mode's `prob`-region.
/// `so_feat` is the peak's SO-power (power axis) or SO-phase (phase axis).
#[inline]
pub fn mode_contains(
    is_phase: bool,
    mode: &[f64],
    prob: f64,
    f: f64,
    so_feat: f64,
) -> bool {
    contains_thr(is_phase, mode, q_threshold(prob), f, so_feat)
}

/// Contour membership against an explicit `Q` threshold (`thr` from
/// [`PeakAssign::threshold`]); non-finite `thr` fails closed.
#[inline]
fn contains_thr(is_phase: bool, mode: &[f64], thr: f64, f: f64, so_feat: f64) -> bool {
    if !thr.is_finite() || mode.len() < 6 {
        return false;
    }
    let q = mode_quadratic(
        is_phase,
        mode[MODE_FMEAN],
        mode[MODE_FSTD],
        mode[MODE_SOMEAN],
        mode[MODE_SOSTD],
        mode[MODE_THETA],
        f,
        so_feat,
    );
    q <= thr
}

/// The fitted power background density at a peak location: the tilted
/// plane `xxx·P + yyy·F + zzz` (`background = [xxx, yyy, zzz]`; x is
/// SO-power, y is frequency — same convention as the rotGauss model).
#[inline]
pub fn power_background_at(background: [f64; 3], f: f64, so_pow: f64) -> f64 {
    background[0] * so_pow + background[1] * f + background[2]
}

/// The mode's own fitted density at a peak location, `amp·exp(−Q)` in the
/// fit's density units. `0.0` for degenerate modes / non-finite inputs so
/// density comparisons fail closed.
#[inline]
pub fn mode_density_at(is_phase: bool, mode: &[f64], f: f64, so_feat: f64) -> f64 {
    if mode.len() < 6 {
        return 0.0;
    }
    let amp = mode[0];
    if !amp.is_finite() || amp <= 0.0 {
        return 0.0;
    }
    let q = mode_quadratic(
        is_phase,
        mode[MODE_FMEAN],
        mode[MODE_FSTD],
        mode[MODE_SOMEAN],
        mode[MODE_SOSTD],
        mode[MODE_THETA],
        f,
        so_feat,
    );
    if q.is_finite() {
        amp * (-q).exp()
    } else {
        0.0
    }
}

/// [`mode_density_at`] gated by the rule's footprint (`q_cap = fp²/2`):
/// zero outside the mode's footprint ellipse, so a capped-out mode can
/// neither pass the background rule nor win (or block) an argmax.
///
/// The footprint exists because the density comparison misbehaves in its
/// far tails wherever the background plane approaches zero: any positive
/// mode density — 10⁻⁹ tails included — beats a ~zero background,
/// silently admitting far-flung peaks. Numeric contours (`Prob`/`Sigma`)
/// carry no footprint — their radius IS the rule.
fn assign_density_at(is_phase: bool, mode: &[f64], q_cap: f64, f: f64, so_feat: f64) -> f64 {
    if mode.len() < 6 {
        return 0.0;
    }
    let q = mode_quadratic(
        is_phase,
        mode[MODE_FMEAN],
        mode[MODE_FSTD],
        mode[MODE_SOMEAN],
        mode[MODE_SOSTD],
        mode[MODE_THETA],
        f,
        so_feat,
    );
    if !q.is_finite() || q > q_cap {
        return 0.0;
    }
    let amp = mode[0];
    if amp.is_finite() && amp > 0.0 {
        amp * (-q).exp()
    } else {
        0.0
    }
}

/// Per-peak property slices used to summarize a mode's member peaks. All
/// slices must have the same length (one entry per peak); `in_hist` marks
/// the peaks that fed the SOPH (the population the mode was fit to).
pub struct PeakProps<'a> {
    pub freq: &'a [f64],
    pub duration: &'a [f64],
    pub bandwidth: &'a [f64],
    pub height: &'a [f64],
    pub volume: &'a [f64],
    pub area: &'a [f64],
    pub peakiness: &'a [f64],
    pub sopower: &'a [f64],
    pub sophase: &'a [f64],
    /// Per-peak absolute prominence power in dB — see [`peak_power_db`].
    /// All-NaN when the recording's baseline is unavailable (caches
    /// written before the baseline was persisted).
    pub power: &'a [f64],
    pub in_hist: &'a [bool],
}

/// Per-peak absolute prominence power in dB:
/// `pow2db(Height × baseline(PeakFrequency))`.
///
/// `Height` is the peak's prominence on the baseline-divided LINEAR
/// spectrogram (unitless ratio), so multiplying by the per-frequency
/// baseline (linear power) re-anchors it to absolute spectral power —
/// the quantity that is comparable across recordings, unlike the ratio,
/// which is hostage to each recording's own floor. `baseline` /
/// `baseline_freqs` are the persisted 2nd-percentile floor and its
/// frequency axis; empty (legacy caches) → all-NaN.
pub fn peak_power_db(
    freq: &[f64],
    height: &[f64],
    baseline: &[f64],
    baseline_freqs: &[f64],
) -> Vec<f64> {
    let n = freq.len().min(height.len());
    if baseline.is_empty() || baseline.len() != baseline_freqs.len() {
        return vec![f64::NAN; n];
    }
    (0..n)
        .map(|i| {
            let (f, h) = (freq[i], height[i]);
            if !f.is_finite() || !h.is_finite() || h <= 0.0 {
                return f64::NAN;
            }
            let b = interp_clamped(baseline_freqs, baseline, f);
            if !b.is_finite() || b <= 0.0 {
                return f64::NAN;
            }
            10.0 * (h * b).log10()
        })
        .collect()
}

/// Linear interpolation of `(xs, ys)` at `x`, clamped to the endpoints.
/// `xs` must be ascending (spectrogram frequency axes are).
fn interp_clamped(xs: &[f64], ys: &[f64], x: f64) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    if x <= xs[0] {
        return ys[0];
    }
    if x >= xs[xs.len() - 1] {
        return ys[ys.len() - 1];
    }
    let j = xs.partition_point(|&v| v < x).max(1);
    let (x0, x1) = (xs[j - 1], xs[j]);
    let (y0, y1) = (ys[j - 1], ys[j]);
    if x1 == x0 {
        return y0;
    }
    y0 + (y1 - y0) * (x - x0) / (x1 - x0)
}

/// Per-mode summary: count of member peaks + the MEDIAN of each tracked
/// property over them (`sophase` is a circular mean — the median is not
/// the natural robust statistic on a circle; `count` is the number of
/// member peaks regardless of per-property finiteness). Medians, not
/// means: the per-peak size properties (Height, Volume, Area, Duration)
/// are heavy-tailed, and a mean is dominated by a night's few giant
/// tail events — on the NSS two-night cohort that costs ALL of the
/// night-to-night subject reliability (mean PkVolume r ≈ 0.0 vs median
/// r ≈ 0.8). Values are `NaN` when no member peak has a finite value
/// for that property.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModePeakStats {
    pub count: usize,
    pub freq: f64,
    pub duration: f64,
    pub bandwidth: f64,
    pub height: f64,
    pub volume: f64,
    pub area: f64,
    pub peakiness: f64,
    /// Median per-peak absolute prominence power, dB ([`peak_power_db`]).
    pub power: f64,
    pub sopower: f64,
    pub sophase: f64,
}

impl ModePeakStats {
    /// All-NaN, zero-count — for modes with no member peaks.
    pub fn empty() -> Self {
        Self {
            count: 0,
            freq: f64::NAN,
            duration: f64::NAN,
            bandwidth: f64::NAN,
            height: f64::NAN,
            volume: f64::NAN,
            area: f64::NAN,
            peakiness: f64::NAN,
            power: f64::NAN,
            sopower: f64::NAN,
            sophase: f64::NAN,
        }
    }
}

/// Median of the finite entries of `vals` at the given member indices;
/// `NaN` if none are finite. Even counts average the two middle values
/// (the same convention as MATLAB `median` and `numpy.median`, so the
/// three implementations agree exactly).
fn median_at(vals: &[f64], members: &[usize]) -> f64 {
    let mut v: Vec<f64> = members
        .iter()
        .filter_map(|&i| vals.get(i).copied().filter(|x| x.is_finite()))
        .collect();
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

/// Circular mean (radians, in (-π, π]) of the finite entries at the given
/// member indices; `NaN` if none are finite or the resultant is ~0.
fn circ_mean_at(vals: &[f64], members: &[usize]) -> f64 {
    let mut sx = 0.0;
    let mut sy = 0.0;
    let mut n = 0usize;
    for &i in members {
        let v = vals.get(i).copied().unwrap_or(f64::NAN);
        if v.is_finite() {
            sx += v.cos();
            sy += v.sin();
            n += 1;
        }
    }
    if n == 0 || (sx == 0.0 && sy == 0.0) {
        f64::NAN
    } else {
        sy.atan2(sx)
    }
}

/// Indices of the peaks that belong to `mode` (in `in_hist` and inside the
/// `prob`-region). `is_phase` selects the envelope and which SO-feature
/// (`sopower` vs `sophase`) is tested.
pub fn mode_member_indices(
    is_phase: bool,
    mode: &[f64],
    prob: f64,
    p: &PeakProps,
) -> Vec<usize> {
    let so_feat = if is_phase { p.sophase } else { p.sopower };
    let n = p.freq.len();
    let mut out = Vec::new();
    for i in 0..n {
        if !p.in_hist.get(i).copied().unwrap_or(false) {
            continue;
        }
        if mode_contains(is_phase, mode, prob, p.freq[i], so_feat.get(i).copied().unwrap_or(f64::NAN)) {
            out.push(i);
        }
    }
    out
}

/// Per-mode property medians over an explicit member-index set (e.g. one
/// entry of [`assign_members`]).
pub fn stats_for_members(members: &[usize], p: &PeakProps) -> ModePeakStats {
    if members.is_empty() {
        return ModePeakStats::empty();
    }
    ModePeakStats {
        count: members.len(),
        freq: median_at(p.freq, members),
        duration: median_at(p.duration, members),
        bandwidth: median_at(p.bandwidth, members),
        height: median_at(p.height, members),
        volume: median_at(p.volume, members),
        area: median_at(p.area, members),
        peakiness: median_at(p.peakiness, members),
        power: median_at(p.power, members),
        sopower: median_at(p.sopower, members),
        sophase: circ_mean_at(p.sophase, members),
    }
}

/// Per-mode property medians over the mode's `prob`-contour member peaks.
pub fn mode_peak_stats(is_phase: bool, mode: &[f64], prob: f64, p: &PeakProps) -> ModePeakStats {
    stats_for_members(&mode_member_indices(is_phase, mode, prob, p), p)
}

/// Member-index sets for every mode under one assignment rule — the
/// entry point for the `peak_assign` setting.
///
/// * `Prob` / `Sigma`: per-mode independent contour at the rule's `Q`
///   threshold (may overlap between modes).
/// * `Background` (power axis): member iff the mode's density at the
///   peak is at least the fitted background density there — both finite,
///   mode density > 0 (degenerate modes stay empty). Non-exclusive.
/// * `Argmax` (power axis): exclusive — each peak goes to the first mode
///   attaining the maximal density (strict `>` scan, so ties keep the
///   earlier mode, matching MATLAB `max` and `np.argmax`), and only when
///   that density beats the background there (ties with background →
///   background; non-finite background → unassigned).
///
/// On the phase axis `Background`/`Argmax` fall back to the
/// [`PEAK_ASSIGN_PHASE_FALLBACK`] contour: phase mode heights are
/// row-normalized empirical values while the stored phase background is
/// pre-normalization, so the density comparison is not well-defined.
pub fn assign_members(
    assign: PeakAssign,
    is_phase: bool,
    modes: &[[f64; 6]],
    background: [f64; 3],
    p: &PeakProps,
) -> Vec<Vec<usize>> {
    let effective = match assign {
        PeakAssign::Background(_) | PeakAssign::Argmax(_) if is_phase => {
            PEAK_ASSIGN_PHASE_FALLBACK
        }
        other => other,
    };
    let so_feat = if is_phase { p.sophase } else { p.sopower };
    let n = p.freq.len();
    let mut out: Vec<Vec<usize>> = vec![Vec::new(); modes.len()];
    let in_hist = |i: usize| p.in_hist.get(i).copied().unwrap_or(false);
    let at = |v: &[f64], i: usize| v.get(i).copied().unwrap_or(f64::NAN);

    match effective {
        PeakAssign::Prob(_) | PeakAssign::Sigma(_) => {
            let thr = effective.threshold().unwrap_or(f64::NAN);
            for (m, mode) in modes.iter().enumerate() {
                for i in 0..n {
                    if in_hist(i) && contains_thr(is_phase, mode, thr, p.freq[i], at(so_feat, i)) {
                        out[m].push(i);
                    }
                }
            }
        }
        PeakAssign::Background(_) => {
            let q_cap = effective.footprint_q_cap().unwrap_or(f64::NAN);
            for i in 0..n {
                if !in_hist(i) {
                    continue;
                }
                let (f, s) = (p.freq[i], at(so_feat, i));
                let bg = power_background_at(background, f, s);
                if !bg.is_finite() {
                    continue; // fail closed
                }
                for (m, mode) in modes.iter().enumerate() {
                    let d = assign_density_at(is_phase, mode, q_cap, f, s);
                    if d > 0.0 && d >= bg {
                        out[m].push(i);
                    }
                }
            }
        }
        PeakAssign::Argmax(_) => {
            let q_cap = effective.footprint_q_cap().unwrap_or(f64::NAN);
            for i in 0..n {
                if !in_hist(i) {
                    continue;
                }
                let (f, s) = (p.freq[i], at(so_feat, i));
                let bg = power_background_at(background, f, s);
                let mut best = 0usize;
                let mut best_d = 0.0f64;
                for (m, mode) in modes.iter().enumerate() {
                    let d = assign_density_at(is_phase, mode, q_cap, f, s);
                    if d > best_d {
                        best_d = d;
                        best = m;
                    }
                }
                // `best_d > bg` is false for NaN bg → unassigned; a tie
                // with the background also goes to the background.
                if best_d > 0.0 && best_d > bg {
                    out[best].push(i);
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    // p=0.95 threshold
    fn thr() -> f64 {
        q_threshold(0.95)
    }

    #[test]
    fn threshold_is_ln20() {
        assert!((thr() - 20f64.ln()).abs() < 1e-12);
        assert!((thr() - 2.99573227).abs() < 1e-6);
        // equals chi2_2(0.95)/2
        assert!((thr() - 5.991464547 / 2.0).abs() < 1e-6);
    }

    #[test]
    fn power_axis_aligned_ellipse() {
        // mode: fmean=13, fstd=1, pmean=5, pstd=10, theta=0
        let mode = [0.0, 13.0, 1.0, 5.0, 10.0, 0.0];
        // center is inside
        assert!(mode_contains(false, &mode, 0.95, 13.0, 5.0));
        // On the freq semi-axis. Q = ½(u/fstd)², so the boundary Q = thr
        // sits at u/fstd = sqrt(2·thr).
        let r = (2.0 * thr()).sqrt();
        assert!(mode_contains(false, &mode, 0.95, 13.0 + r * 1.0 - 1e-9, 5.0));
        assert!(!mode_contains(false, &mode, 0.95, 13.0 + r * 1.0 + 1e-3, 5.0));
        // power semi-axis
        assert!(mode_contains(false, &mode, 0.95, 13.0, 5.0 + r * 10.0 - 1e-9));
        assert!(!mode_contains(false, &mode, 0.95, 13.0, 5.0 + r * 10.0 + 1e-2));
        // far away outside
        assert!(!mode_contains(false, &mode, 0.95, 2.0, -4.0));
    }

    #[test]
    fn power_rotation_matches_rotgauss() {
        // theta rotates the ellipse: a point along the rotated major axis
        // should be inside where the unrotated test would put it.
        let theta = 0.3_f64;
        let mode = [0.0, 13.0, 1.0, 5.0, 10.0, theta];
        let r = ((2.0 * thr()).sqrt()) * 0.99;
        // place a point at rotated-u = r*fstd, v = 0 ->
        // df = u*cos? invert: (df, dp) = R^-1 * (u, v) with
        // u =  df*cos + dp*sin ; v = -df*sin + dp*cos
        let (s, c) = theta.sin_cos();
        let u = r * 1.0;
        let v = 0.0;
        let df = u * c - v * s;
        let dp = u * s + v * c;
        assert!(mode_contains(false, &mode, 0.95, 13.0 + df, 5.0 + dp));
    }

    #[test]
    fn phase_axis_aligned() {
        // mode: fmean=13, fstd=1, phasemean=0, sostd=0.5 (k=4), theta=0
        let mode = [0.0, 13.0, 1.0, 0.0, 0.5, 0.0];
        assert!(mode_contains(true, &mode, 0.95, 13.0, 0.0)); // center
        // opposite phase is far outside (k*(1-cos pi)=2k=8 > thr)
        assert!(!mode_contains(true, &mode, 0.95, 13.0, PI));
        // wrap: phase just past -pi equals just past +pi region
        assert!(!mode_contains(true, &mode, 0.95, 13.0, -PI + 1e-6));
        // small phase offset within region
        assert!(mode_contains(true, &mode, 0.95, 13.0, 0.2));
    }

    #[test]
    fn phase_freq_axis_uses_sigma() {
        // FreqStd is a sigma; freq term = ½(dF/FreqStd)^2 <= thr, so the
        // boundary sits at dF/FreqStd = sqrt(2·thr). The von Mises term is
        // zero on this axis, so this isolates the Gaussian half.
        let mode = [0.0, 13.0, 2.0, 0.0, 0.5, 0.0];
        let r = (2.0 * thr()).sqrt();
        assert!(mode_contains(true, &mode, 0.95, 13.0 + r * 2.0 - 1e-6, 0.0));
        assert!(!mode_contains(true, &mode, 0.95, 13.0 + r * 2.0 + 1e-2, 0.0));
    }

    /// The pairing of `mode_quadratic` with `q_threshold` must actually
    /// enclose `prob` of the fitted Gaussian's mass. A kernel that carries
    /// the ½ paired with a `Q` that omits it leaves every other test green
    /// while silently dropping containment from 95% to ~77.7%, which would
    /// shift every Pk* column with no error raised anywhere.
    ///
    /// Deterministic grid integration of the true bivariate normal that the
    /// kernel represents (σ_f = fstd, σ_p = pstd), no RNG.
    #[test]
    fn containment_encloses_the_requested_probability() {
        let (fmean, fstd, pmean, pstd) = (13.0_f64, 1.3_f64, 5.0_f64, 7.0_f64);
        let mode = [0.0, fmean, fstd, pmean, pstd, 0.0];

        for &prob in &[0.5_f64, 0.8, 0.95, 0.99] {
            // Integrate out to ±8σ on each axis, well past any region.
            let (n, span) = (1201usize, 8.0_f64);
            let (df, dp) = (2.0 * span * fstd / n as f64, 2.0 * span * pstd / n as f64);
            let mut inside = 0.0_f64;
            let mut total = 0.0_f64;
            for i in 0..n {
                let f = fmean - span * fstd + (i as f64 + 0.5) * df;
                for j in 0..n {
                    let p = pmean - span * pstd + (j as f64 + 0.5) * dp;
                    let zf = (f - fmean) / fstd;
                    let zp = (p - pmean) / pstd;
                    let dens = (-0.5 * (zf * zf + zp * zp)).exp();
                    total += dens;
                    if mode_contains(false, &mode, prob, f, p) {
                        inside += dens;
                    }
                }
            }
            let captured = inside / total;
            assert!(
                (captured - prob).abs() < 2e-3,
                "prob={prob}: captured {captured:.5}, expected {prob:.5} \
                 (a ~0.777 at prob=0.95 means Q lost its ½)",
            );
        }
    }

    #[test]
    fn two_member_medians_exclude_the_outside_peak() {
        // Two peaks inside a power mode with known heights/durations, one
        // peak far outside that must be excluded.
        let mode = [0.0, 13.0, 1.0, 5.0, 10.0, 0.0];
        let freq = [13.0, 13.2, 2.0];
        let sopower = [5.0, 6.0, -4.0];
        let sophase = [0.0, 0.1, 0.0];
        let height = [10.0, 20.0, 999.0];
        let duration = [1.0, 3.0, 999.0];
        let in_hist = [true, true, true];
        let z = [0.0, 0.0, 0.0];
        let p = PeakProps {
            freq: &freq,
            duration: &duration,
            bandwidth: &z,
            height: &height,
            volume: &z,
            area: &z,
            peakiness: &z,
            sopower: &sopower,
            sophase: &sophase,
            power: &z,
            in_hist: &in_hist,
        };
        let m = mode_peak_stats(false, &mode, 0.95, &p);
        assert_eq!(m.count, 2);
        assert!((m.height - 15.0).abs() < 1e-9);
        assert!((m.duration - 2.0).abs() < 1e-9);
    }

    #[test]
    fn medians_resist_a_tail_event() {
        // Three member peaks. The third is a giant tail event. The median
        // must sit at the middle value, where a mean would be dragged to
        // 343. This is why the summary uses medians (NSS two-night cohort:
        // mean PkVolume r ~ 0, median r ~ 0.8).
        let mode = [0.0, 13.0, 1.0, 5.0, 10.0, 0.0];
        let freq = [13.0, 13.1, 13.2];
        let sopower = [5.0, 5.5, 6.0];
        let sophase = [0.0, 0.0, 0.0];
        let height = [10.0, 20.0, 999.0];
        let in_hist = [true, true, true];
        let z = [0.0, 0.0, 0.0];
        let p = PeakProps {
            freq: &freq,
            duration: &z,
            bandwidth: &z,
            height: &height,
            volume: &z,
            area: &z,
            peakiness: &z,
            sopower: &sopower,
            sophase: &sophase,
            power: &z,
            in_hist: &in_hist,
        };
        let m = mode_peak_stats(false, &mode, 0.95, &p);
        assert_eq!(m.count, 3);
        assert!((m.height - 20.0).abs() < 1e-12, "median height, got {}", m.height);
        assert!((m.sopower - 5.5).abs() < 1e-12);
        // Even count averages the two middle values (MATLAB/numpy parity).
        let m2 = stats_for_members(&[0, 1], &p);
        assert!((m2.height - 15.0).abs() < 1e-12);
    }

    #[test]
    fn peak_power_reanchors_to_the_baseline() {
        // Height is a LINEAR ratio over the per-recording floor; PkPower
        // re-anchors it: pow2db(height × baseline(freq)). Baseline grid
        // [10, 20] Hz with values [0.1, 0.4]: a peak at 15 Hz interpolates
        // baseline 0.25, so height 40 → pow2db(40 × 0.25) = 10 dB exactly.
        let power = peak_power_db(&[15.0, 10.0, 25.0], &[40.0, 10.0, 10.0], &[0.1, 0.4], &[10.0, 20.0]);
        assert!((power[0] - 10.0).abs() < 1e-12, "got {}", power[0]);
        assert!((power[1] - 0.0).abs() < 1e-12, "clamped to the low edge, got {}", power[1]);
        assert!((power[2] - 10.0 * (10.0_f64 * 0.4).log10()).abs() < 1e-12, "clamped high");
        // No baseline (legacy cache) → all NaN; nonpositive height → NaN.
        assert!(peak_power_db(&[15.0], &[40.0], &[], &[]).iter().all(|v| v.is_nan()));
        assert!(peak_power_db(&[15.0], &[0.0], &[0.1, 0.4], &[10.0, 20.0])[0].is_nan());
    }

    #[test]
    fn in_hist_filter_excludes() {
        let mode = [0.0, 13.0, 1.0, 5.0, 10.0, 0.0];
        let freq = [13.0, 13.0];
        let sopower = [5.0, 5.0];
        let sophase = [0.0, 0.0];
        let height = [10.0, 20.0];
        let in_hist = [true, false]; // second excluded despite being at center
        let z = [0.0, 0.0];
        let p = PeakProps {
            freq: &freq,
            duration: &z,
            bandwidth: &z,
            height: &height,
            volume: &z,
            area: &z,
            peakiness: &z,
            sopower: &sopower,
            sophase: &sophase,
            power: &z,
            in_hist: &in_hist,
        };
        let m = mode_peak_stats(false, &mode, 0.95, &p);
        assert_eq!(m.count, 1);
        assert!((m.height - 10.0).abs() < 1e-9);
    }

    #[test]
    fn circular_mean_wraps() {
        // phases near ±pi average to ~pi, not ~0
        let mode = [0.0, 13.0, 100.0, PI, 5.0, 0.0]; // wide, centered at pi
        let freq = [13.0, 13.0];
        let sopower = [0.0, 0.0];
        let sophase = [PI - 0.1, -PI + 0.1]; // both near pi
        let in_hist = [true, true];
        let z = [0.0, 0.0];
        let p = PeakProps {
            freq: &freq,
            duration: &z,
            bandwidth: &z,
            height: &z,
            volume: &z,
            area: &z,
            peakiness: &z,
            sopower: &sopower,
            sophase: &sophase,
            power: &z,
            in_hist: &in_hist,
        };
        let m = mode_peak_stats(true, &mode, 0.95, &p);
        assert_eq!(m.count, 2);
        assert!(m.sophase.abs() > 3.0, "circular mean near pi, got {}", m.sophase);
    }

    #[test]
    fn degenerate_mode_contains_nothing() {
        let mode = [0.0, 13.0, 0.0, 5.0, 10.0, 0.0]; // fstd=0
        assert!(!mode_contains(false, &mode, 0.95, 13.0, 5.0));
        let bad_prob = [0.0, 13.0, 1.0, 5.0, 10.0, 0.0];
        assert!(!mode_contains(false, &bad_prob, 1.5, 13.0, 5.0)); // prob>1
    }

    // ── assignment rules (shared analytic expectations E1–E9; the same
    //    fixtures are pinned in the MATLAB and Python mirrors) ─────────
    //
    // Mode A: amp 10 at (13 Hz, 5 dB), σf=1, σp=10.
    // Mode B: amp  5 at (13 Hz, 25 dB), σf=1, σp=10.
    const MODE_A: [f64; 6] = [10.0, 13.0, 1.0, 5.0, 10.0, 0.0];
    const MODE_B: [f64; 6] = [5.0, 13.0, 1.0, 25.0, 10.0, 0.0];

    /// PeakProps over (freq, sopower) pairs, everything in_hist.
    fn props(pts: &[(f64, f64)]) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<bool>) {
        let f: Vec<f64> = pts.iter().map(|p| p.0).collect();
        let s: Vec<f64> = pts.iter().map(|p| p.1).collect();
        let z = vec![0.0; pts.len()];
        let ih = vec![true; pts.len()];
        (f, s, z, ih)
    }
    fn peak_props<'a>(
        f: &'a [f64],
        s: &'a [f64],
        z: &'a [f64],
        ih: &'a [bool],
    ) -> PeakProps<'a> {
        PeakProps {
            freq: f,
            duration: z,
            bandwidth: z,
            height: z,
            volume: z,
            area: z,
            peakiness: z,
            sopower: s,
            sophase: z,
            power: z,
            in_hist: ih,
        }
    }

    #[test]
    fn e1_background_rule_equals_pstar_contour() {
        // Constant bg 0.5, amp 10 → p* = 0.95: freq boundary at
        // 13 + sqrt(2·ln 20)·σf, same numbers as the prob test above.
        let r = (2.0 * thr()).sqrt();
        let pts = [(13.0 + r - 1e-9, 5.0), (13.0 + r + 1e-3, 5.0)];
        let (f, s, z, ih) = props(&pts);
        let p = peak_props(&f, &s, &z, &ih);
        let m = assign_members(PeakAssign::Background(3.0), false, &[MODE_A], [0.0, 0.0, 0.5], &p);
        assert_eq!(m[0], vec![0]);
    }

    #[test]
    fn e2_background_rule_adapts_to_amp() {
        // amp 5 vs bg 0.5 → p* = 0.9: power boundary at
        // 25 + sqrt(2·ln 10)·10 ≈ 46.4597.
        let b = 25.0 + (2.0 * 10f64.ln()).sqrt() * 10.0;
        let pts = [(13.0, b - 0.01), (13.0, b + 0.01)];
        let (f, s, z, ih) = props(&pts);
        let p = peak_props(&f, &s, &z, &ih);
        let m = assign_members(PeakAssign::Background(3.0), false, &[MODE_B], [0.0, 0.0, 0.5], &p);
        assert_eq!(m[0], vec![0]);
    }

    #[test]
    fn e3_background_evaluated_at_the_peak() {
        // Sloped bg [0.1, 0, 0]: at (13, 25) the plane is 2.5 while
        // d_A = 10·e⁻² ≈ 1.3534 → not a member; a constant bg 0.5 admits it.
        let pts = [(13.0, 25.0)];
        let (f, s, z, ih) = props(&pts);
        let p = peak_props(&f, &s, &z, &ih);
        let sloped = assign_members(PeakAssign::Background(3.0), false, &[MODE_A], [0.1, 0.0, 0.0], &p);
        assert!(sloped[0].is_empty());
        let flat = assign_members(PeakAssign::Background(3.0), false, &[MODE_A], [0.0, 0.0, 0.5], &p);
        assert_eq!(flat[0], vec![0]);
    }

    #[test]
    fn e4_argmax_is_exclusive() {
        // (13, 15) sits inside both 95% ellipses; d_A ≈ 6.065 > d_B ≈ 3.033
        // → argmax gives it to A only.
        let pts = [(13.0, 15.0)];
        let (f, s, z, ih) = props(&pts);
        let p = peak_props(&f, &s, &z, &ih);
        let both = assign_members(PeakAssign::Prob(0.95), false, &[MODE_A, MODE_B], [0.0; 3], &p);
        assert_eq!((both[0].len(), both[1].len()), (1, 1));
        let ex = assign_members(PeakAssign::Argmax(3.0), false, &[MODE_A, MODE_B], [0.0, 0.0, 0.5], &p);
        assert_eq!(ex[0], vec![0]);
        assert!(ex[1].is_empty());
    }

    #[test]
    fn e5_background_wins_leaves_unassigned() {
        // (13, 60): d_A ≈ 2.7e-6, d_B ≈ 0.0109, both < bg 0.5.
        let pts = [(13.0, 60.0)];
        let (f, s, z, ih) = props(&pts);
        let p = peak_props(&f, &s, &z, &ih);
        let ex = assign_members(PeakAssign::Argmax(3.0), false, &[MODE_A, MODE_B], [0.0, 0.0, 0.5], &p);
        assert!(ex[0].is_empty() && ex[1].is_empty());
        let bgm = assign_members(PeakAssign::Background(3.0), false, &[MODE_A, MODE_B], [0.0, 0.0, 0.5], &p);
        assert!(bgm[0].is_empty() && bgm[1].is_empty());
    }

    #[test]
    fn e6_argmax_tie_is_first_wins() {
        let pts = [(13.0, 5.0)];
        let (f, s, z, ih) = props(&pts);
        let p = peak_props(&f, &s, &z, &ih);
        let ex = assign_members(PeakAssign::Argmax(3.0), false, &[MODE_A, MODE_A], [0.0, 0.0, 0.5], &p);
        assert_eq!(ex[0], vec![0]);
        assert!(ex[1].is_empty());
    }

    #[test]
    fn e7_phase_axis_falls_back_to_prob() {
        let mode = [0.5, 13.0, 1.0, 0.0, 0.5, 0.0];
        let pts = [(13.0, 0.0), (13.0, 0.2), (13.0, PI)];
        let f: Vec<f64> = pts.iter().map(|p| p.0).collect();
        let ph: Vec<f64> = pts.iter().map(|p| p.1).collect();
        let z = vec![0.0; pts.len()];
        let ih = vec![true; pts.len()];
        let p = PeakProps {
            freq: &f, duration: &z, bandwidth: &z, height: &z, volume: &z,
            area: &z, peakiness: &z, sopower: &z, sophase: &ph, power: &z,
            in_hist: &ih,
        };
        let fallback = assign_members(PEAK_ASSIGN_PHASE_FALLBACK, true, &[mode], [0.0; 3], &p);
        for rule in [PeakAssign::Background(1.5), PeakAssign::Argmax(1.5)] {
            assert_eq!(assign_members(rule, true, &[mode], [7.0, 7.0, 7.0], &p), fallback);
        }
    }

    #[test]
    fn e8_prob_rule_matches_mode_member_indices() {
        let pts = [(13.0, 5.0), (13.0, 15.0), (2.0, -4.0), (13.4, 28.0)];
        let (f, s, z, ih) = props(&pts);
        let p = peak_props(&f, &s, &z, &ih);
        let via_rule = assign_members(PeakAssign::Prob(0.95), false, &[MODE_A, MODE_B], [0.0; 3], &p);
        assert_eq!(via_rule[0], mode_member_indices(false, &MODE_A, 0.95, &p));
        assert_eq!(via_rule[1], mode_member_indices(false, &MODE_B, 0.95, &p));
    }

    #[test]
    fn e11_density_rules_respect_the_footprint_cap() {
        // Near-zero background: without the cap, ANY positive tail density
        // wins. A peak at 3.5σ (Q = 6.125 > 4.5) must be excluded under
        // both density rules; one at 2.9σ (Q = 4.205 ≤ 4.5) stays in.
        let bg = [0.0, 0.0, 1e-6];
        let pts = [(13.0, 5.0 + 2.9 * 10.0), (13.0, 5.0 + 3.5 * 10.0)];
        let (f, s, z, ih) = props(&pts);
        let p = peak_props(&f, &s, &z, &ih);
        for rule in [PeakAssign::Background(3.0), PeakAssign::Argmax(3.0)] {
            let m = assign_members(rule, false, &[MODE_A], bg, &p);
            assert_eq!(m[0], vec![0], "{rule:?}: cap must drop the 3.5σ peak only");
        }
        // The default footprint is 1.5σ: a 1.4σ peak is in, a 1.6σ peak out.
        let pts = [(13.0, 5.0 + 1.4 * 10.0), (13.0, 5.0 + 1.6 * 10.0)];
        let (f, s, z, ih) = props(&pts);
        let p = peak_props(&f, &s, &z, &ih);
        for rule in [PeakAssign::default(), PeakAssign::Background(1.5)] {
            let m = assign_members(rule, false, &[MODE_A], bg, &p);
            assert_eq!(m[0], vec![0], "{rule:?}: default footprint is 1.5σ");
        }
    }

    #[test]
    fn e12_capped_out_mode_does_not_block_argmax() {
        // A peak far from strong Mode A (capped out) but at weak Mode C's
        // center: C wins the argmax — A's extrapolated tail neither claims
        // nor blocks it.
        let mode_c = [0.01, 13.0, 1.0, 45.0, 10.0, 0.0];
        let pts = [(13.0, 45.0)]; // 4σ from A, center of C
        let (f, s, z, ih) = props(&pts);
        let p = peak_props(&f, &s, &z, &ih);
        let m = assign_members(PeakAssign::Argmax(3.0), false, &[MODE_A, mode_c], [0.0, 0.0, 1e-6], &p);
        assert!(m[0].is_empty(), "capped-out A must not claim the peak");
        assert_eq!(m[1], vec![0], "local weak mode C wins");
    }

    #[test]
    fn e9_sigma_form_is_an_exact_sigma_radius() {
        // "1 sigma": thr = 0.5 → freq boundary at 13 + 1·σf exactly.
        let pts = [(14.0 - 1e-9, 5.0), (14.0 + 1e-3, 5.0)];
        let (f, s, z, ih) = props(&pts);
        let p = peak_props(&f, &s, &z, &ih);
        let m = assign_members(PeakAssign::Sigma(1.0), false, &[MODE_A], [0.0; 3], &p);
        assert_eq!(m[0], vec![0]);
        // Sigma(n) ≡ Prob(1 − e^{−n²/2}) on the power axis.
        let peq = 1.0 - (-0.5f64).exp();
        let via_p = assign_members(PeakAssign::Prob(peq), false, &[MODE_A], [0.0; 3], &p);
        assert_eq!(m, via_p);
    }
}
