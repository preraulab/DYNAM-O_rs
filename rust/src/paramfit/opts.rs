//! Options of the parametric-basis and spline fits, as a settings file
//! spells them.
//!
//! These types are the fit options of every caller: the desktop app and
//! command-line tool (which embed them in their settings file), and the
//! Python and MATLAB wrappers. Their serde forms are the settings-file
//! schema, so a field rename here is a settings-format change.

use serde::{Deserialize, Serialize};

fn default_true() -> bool { true }

/// How TF-peaks are associated with a parametric mode for the per-mode
/// `Pk*` columns. One flexible settings value, written as a bare JSON
/// number or a string (case-insensitive):
///
/// * `0.3` / `"0.3"` / `"0.3 p"` — probability-mass contour `p`:
///   member iff the peak lies inside the mode's `p`-mass ellipse
///   (power axis) / `(1-p)` relative-height contour (phase axis).
/// * `"1.3 s"` / `"1.3 sigma"` — σ-radius contour `n`: the same
///   machinery with semi-axes `n·σ` (on the power axis this equals
///   `p = 1 − exp(−n²/2)`).
/// * `"background"` — member iff the mode's fitted density at the peak
///   is at least the fitted background density there (power axis only).
/// * `"argmax"` — exclusive: each peak belongs to the densest component
///   among {background, modes} (power axis only). **The default.**
///
/// The density rules operate within each mode's footprint ellipse —
/// `"argmax"` / `"background"` alone mean a 1.5σ footprint
/// ([`DEFAULT_DENSITY_FOOTPRINT_SIGMA`]); append a radius to widen or
/// tighten it: `"argmax 2 s"`, `"background 1.75 sigma"`.
///
/// The phase axis always uses a numeric contour: the configured value
/// under Prob/Sigma, else [`PEAK_ASSIGN_PHASE_FALLBACK`] — phase mode
/// heights are row-normalized empirical values while the stored phase
/// background is pre-normalization, so a density comparison is not
/// well-defined there. Exact rule semantics:
/// `dynamo_rs::mode_peaks`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PeakAssign {
    /// Probability-mass contour, `0 < p < 1`.
    Prob(f64),
    /// σ-radius contour, `n > 0`.
    Sigma(f64),
    /// Mode density ≥ fitted background density at the peak, within the
    /// given σ footprint.
    Background(f64),
    /// Exclusive assignment to the densest component (incl. background),
    /// each mode restricted to the given σ footprint.
    Argmax(f64),
}

/// Numeric contour the phase axis falls back to under `Background` /
/// `Argmax`: the 0.95 probability contour.
pub const PEAK_ASSIGN_PHASE_FALLBACK: PeakAssign = PeakAssign::Prob(0.95);

/// Footprint radius the density rules use when none is spelled out.
/// 1.5σ hugs the visually distinct mode core (~67.5% of the mode's
/// mass), and the NSS footprint sweep showed the Pk property means sit
/// on a plateau around it (1.3–1.7σ moves them by only a few percent —
/// the footprint choice mainly scales PkCount).
pub const DEFAULT_DENSITY_FOOTPRINT_SIGMA: f64 = 1.5;

impl Default for PeakAssign {
    fn default() -> Self {
        Self::Argmax(DEFAULT_DENSITY_FOOTPRINT_SIGMA)
    }
}

impl PeakAssign {
    /// Parse the settings form. Accepts `"argmax"`, `"background"`
    /// (optionally with a σ footprint: `"argmax 2 s"`,
    /// `"background 1.75 sigma"`), `"<x>"`, `"<x> p"`, `"<n> s"`,
    /// `"<n> sigma"` (trimmed, case-insensitive).
    pub fn parse(s: &str) -> Result<Self, String> {
        let lowered = s.trim().to_ascii_lowercase();
        let tokens: Vec<&str> = lowered.split_whitespace().collect();
        match tokens.as_slice() {
            ["argmax"] => return Ok(Self::Argmax(DEFAULT_DENSITY_FOOTPRINT_SIGMA)),
            ["background"] => return Ok(Self::Background(DEFAULT_DENSITY_FOOTPRINT_SIGMA)),
            [rule @ ("argmax" | "background"), n, unit @ ("s" | "sigma")] => {
                // `unit` only restricts the pattern to a valid spelling.
                // Its value is not needed.
                let _ = unit;
                let fp: f64 = n.parse().map_err(|_| {
                    format!("peak_assign: `{s}` — footprint `{n}` is not a number")
                })?;
                if !(fp.is_finite() && fp > 0.0) {
                    return Err(format!("peak_assign: footprint must be > 0, got {fp}"));
                }
                return Ok(if *rule == "argmax" {
                    Self::Argmax(fp)
                } else {
                    Self::Background(fp)
                });
            }
            _ => {}
        }
        // A bare number has no unit and is read as a probability (`p`).
        let (num, unit) = match lowered.rsplit_once(char::is_whitespace) {
            Some((n, u)) => (n.trim(), u.trim()),
            None => (lowered.as_str(), "p"),
        };
        let v: f64 = num
            .parse()
            .map_err(|_| format!("peak_assign: `{s}` is not a number, `background`, or `argmax`"))?;
        match unit {
            "p" => Self::prob(v),
            "s" | "sigma" => Self::sigma(v),
            _ => Err(format!(
                "peak_assign: unknown unit `{unit}` in `{s}` (use `p`, `s`, or `sigma`)"
            )),
        }
    }

    fn prob(p: f64) -> Result<Self, String> {
        if p.is_finite() && p > 0.0 && p < 1.0 {
            Ok(Self::Prob(p))
        } else {
            Err(format!("peak_assign: probability must be in (0, 1), got {p}"))
        }
    }

    fn sigma(n: f64) -> Result<Self, String> {
        if n.is_finite() && n > 0.0 {
            Ok(Self::Sigma(n))
        } else {
            Err(format!("peak_assign: sigma radius must be > 0, got {n}"))
        }
    }

    /// The `Q` threshold of a numeric contour (`Prob(p)` → `-ln(1-p)`,
    /// `Sigma(n)` → `n²/2`); `None` for the density rules.
    pub fn threshold(&self) -> Option<f64> {
        match *self {
            Self::Prob(p) => Some(-(1.0 - p).ln()),
            Self::Sigma(n) => Some(0.5 * n * n),
            Self::Background(_) | Self::Argmax(_) => None,
        }
    }

    /// The density rules' footprint as a `Q` cap (`fp²/2`); `None` for
    /// the numeric contours (their radius IS the rule).
    pub fn footprint_q_cap(&self) -> Option<f64> {
        match *self {
            Self::Background(fp) | Self::Argmax(fp) => Some(0.5 * fp * fp),
            Self::Prob(_) | Self::Sigma(_) => None,
        }
    }
}

impl std::fmt::Display for PeakAssign {
    /// Canonical settings/CSV form: `argmax 1.5 sigma`,
    /// `background 1.5 sigma`, `0.3 p`, `1.3 sigma` — density rules
    /// always spell their footprint so provenance is explicit.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Self::Prob(p) => write!(f, "{p} p"),
            Self::Sigma(n) => write!(f, "{n} sigma"),
            Self::Background(fp) => write!(f, "background {fp} sigma"),
            Self::Argmax(fp) => write!(f, "argmax {fp} sigma"),
        }
    }
}

impl Serialize for PeakAssign {
    fn serialize<S: serde::Serializer>(&self, ser: S) -> Result<S::Ok, S::Error> {
        ser.serialize_str(&self.to_string())
    }
}

impl<'de> Deserialize<'de> for PeakAssign {
    fn deserialize<D: serde::Deserializer<'de>>(de: D) -> Result<Self, D::Error> {
        struct V;
        impl serde::de::Visitor<'_> for V {
            type Value = PeakAssign;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a probability in (0,1), or a string like \"0.3 p\", \"1.3 sigma\", \"background\", \"argmax\"")
            }
            fn visit_f64<E: serde::de::Error>(self, v: f64) -> Result<Self::Value, E> {
                PeakAssign::prob(v).map_err(E::custom)
            }
            fn visit_u64<E: serde::de::Error>(self, v: u64) -> Result<Self::Value, E> {
                PeakAssign::prob(v as f64).map_err(E::custom)
            }
            fn visit_i64<E: serde::de::Error>(self, v: i64) -> Result<Self::Value, E> {
                PeakAssign::prob(v as f64).map_err(E::custom)
            }
            fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<Self::Value, E> {
                PeakAssign::parse(v).map_err(E::custom)
            }
        }
        de.deserialize_any(V)
    }
}

/// Paramfit bounds + thresholds — the analogue of MATLAB's
/// `param_basis_opts('power')` / `('phase')`. Each axis has its own
/// substruct so the schema can grow asymmetrically (e.g. phase has
/// `gauss_filt_std`, power has `min_freq_diff`) without forcing
/// otherwise-meaningless fields onto the other axis.
///
/// MATLAB-parity reference: `toolbox/SOPH_dim_reduction/parametric_basis/
/// param_basis_opts.m`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ParamFitOpts {
    #[serde(default)]
    pub power: PowerParamFitOpts,
    #[serde(default)]
    pub phase: PhaseParamFitOpts,
    /// TF-peak → mode assignment for the per-mode `Pk*` columns — one
    /// flexible value: a probability-mass contour (`0.3` or `"0.3 p"`),
    /// a σ-radius contour (`"1.3 sigma"`), `"background"` (member iff
    /// the mode's density at the peak is at least the fitted background
    /// density there), or `"argmax"` (exclusive: each peak to the
    /// densest component, background included). The density rules apply
    /// within each mode's footprint ellipse — 1.5σ by default
    /// ([`DEFAULT_DENSITY_FOOTPRINT_SIGMA`]), overridable as
    /// `"argmax 2 sigma"` — so near-zero background regions can't admit
    /// far tails. The phase
    /// axis always uses a numeric contour (the configured value, else
    /// `0.95 p`).
    /// A settings file from before semantics version 4 carries
    /// `peak_assign_prob` instead, and its value migrates to `"<x> p"`.
    /// Default "argmax". See [`PeakAssign`] and
    /// `dynamo_rs::mode_peaks`.
    #[serde(default)]
    pub peak_assign: PeakAssign,
}

/// Spline-basis fit windows + knot counts — the analogue of MATLAB's
/// `spline_basis_opts('power')` / `('phase')`. Exposes the same surface
/// MATLAB does: the SO and frequency fit windows plus the per-axis
/// interior-knot counts. Spline order (cubic, order 4) and the ±0.1
/// boundary-knot padding stay hard-coded in `fits.rs` to match MATLAB,
/// which does not expose them.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct SplineFitOpts {
    #[serde(default)]
    pub power: SplinePowerOpts,
    #[serde(default)]
    pub phase: SplinePhaseOpts,
}

/// Power-axis spline windows + knot counts. MATLAB `spline_basis_opts('power')`:
/// `power_limits=[-2,20]`, `freq_limits=[2,18]`, `num_knots_x=5`, `num_knots_y=18`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SplinePowerOpts {
    /// The slow-oscillation power range to fit over, in dB. Data outside is
    /// ignored. Also sets the SO-power x-axis of the splinefit diagnostic
    /// figure (MATLAB `plot_SOPH_splinefits` `xlim`).
    #[serde(default = "default_spline_power_limits")]
    pub power_limits: [f64; 2],
    /// The frequency range to fit over, in Hz. Data outside is ignored.
    /// Also sets the frequency y-axis of the splinefit diagnostic figure
    /// (MATLAB `plot_SOPH_splinefits` `ylim`).
    #[serde(default = "default_spline_freq_limits")]
    pub freq_limits: [f64; 2],
    /// How many spline knots along the SO axis. More knots follow the data more
    /// closely and risk fitting noise.
    #[serde(default = "default_spline_power_knots_x")]
    pub num_knots_x: usize,
    /// How many spline knots along the frequency axis. More knots follow the
    /// data more closely and risk fitting noise.
    #[serde(default = "default_spline_power_knots_y")]
    pub num_knots_y: usize,
}

impl Default for SplinePowerOpts {
    fn default() -> Self {
        Self {
            power_limits: default_spline_power_limits(),
            freq_limits: default_spline_freq_limits(),
            num_knots_x: default_spline_power_knots_x(),
            num_knots_y: default_spline_power_knots_y(),
        }
    }
}

/// Phase-axis spline windows + knot counts. MATLAB `spline_basis_opts('phase')`:
/// `phase_limits=[-pi,pi]`, `freq_limits=[2,18]`, `num_knots_x=5`, `num_knots_y=9`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SplinePhaseOpts {
    /// The slow-oscillation phase range to fit over, in radians. Also sets
    /// the SO-phase x-axis of the splinefit diagnostic figure (MATLAB
    /// `plot_SOPH_splinefits` `xlim`).
    #[serde(default = "default_spline_phase_limits")]
    pub phase_limits: [f64; 2],
    /// The frequency range to fit over, in Hz. Data outside is ignored.
    /// Also sets the frequency y-axis of the splinefit diagnostic figure
    /// (MATLAB `plot_SOPH_splinefits` `ylim`).
    #[serde(default = "default_spline_freq_limits")]
    pub freq_limits: [f64; 2],
    /// How many spline knots along the SO axis. More knots follow the data more
    /// closely and risk fitting noise.
    #[serde(default = "default_spline_phase_knots_x")]
    pub num_knots_x: usize,
    /// How many spline knots along the frequency axis. More knots follow the
    /// data more closely and risk fitting noise.
    #[serde(default = "default_spline_phase_knots_y")]
    pub num_knots_y: usize,
}

impl Default for SplinePhaseOpts {
    fn default() -> Self {
        Self {
            phase_limits: default_spline_phase_limits(),
            freq_limits: default_spline_freq_limits(),
            num_knots_x: default_spline_phase_knots_x(),
            num_knots_y: default_spline_phase_knots_y(),
        }
    }
}

fn default_spline_power_limits() -> [f64; 2] { [-2.0, 20.0] }
fn default_spline_phase_limits() -> [f64; 2] {
    [-std::f64::consts::PI, std::f64::consts::PI]
}
/// Matches MATLAB `spline_basis_opts.m:57,65`, which sets `[2, 18]` for both
/// axes. That file's docstring says `[2, 16]`, which disagrees with its own
/// code. This default follows the code.
fn default_spline_freq_limits() -> [f64; 2] { [2.0, 18.0] }
fn default_spline_power_knots_x() -> usize { 5 }
fn default_spline_power_knots_y() -> usize { 18 }
fn default_spline_phase_knots_x() -> usize { 5 }
fn default_spline_phase_knots_y() -> usize { 9 }

/// Watershed seed-extraction params for the histogram peak detector
/// (`extracthistpeaks`). Mirrors MATLAB's
/// `watershed_params = [merge_thresh, dur_min, bw_min, height_min, trim_vol]`.
///
/// `merge_thresh = "derived"` means "auto". The extractor receives the
/// MATLAB `NaN` sentinel and merges down to the 95th percentile of the
/// initial region-edge weights (MATLAB `mergeWshedSegment.m`).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct WatershedParams {
    /// Merge two regions if their lower neighbor pixel is within this
    /// many height units of the lower peak. `"derived"` = auto, passed to
    /// the extractor as the `NaN` sentinel, which it replaces with the 95th
    /// percentile of the initial region-edge weights. `null` still parses,
    /// from older files.
    #[serde(default)]
    pub merge_thresh: Bound,
    /// Minimum region duration (SO-power dB / SO-phase rad units).
    pub dur_min: f64,
    /// Minimum region bandwidth (Hz).
    pub bw_min: f64,
    /// Minimum region peak height (events/min/bin or proportion/bin).
    pub height_min: f64,
    /// Volume-trim threshold: keep cells above `trim_vol · max_in_region`.
    pub trim_vol: f64,
}

impl WatershedParams {
    /// The watershed options in the form the kernel's histogram-peak
    /// extractor takes. A derived `merge_thresh` is passed as the `NaN`
    /// sentinel, which the extractor replaces with the 95th percentile of
    /// the initial region-edge weights. Every caller that runs the
    /// watershed builds its parameters here, so the fit, the exported
    /// figure and the in-app panel cannot disagree.
    pub fn to_hist_peak_params(&self) -> crate::histpeaks::HistPeakParams {
        crate::histpeaks::HistPeakParams {
            merge_thresh: self.merge_thresh.resolve(f64::NAN),
            dur_min: self.dur_min,
            bw_min: self.bw_min,
            height_min: self.height_min,
            trim_vol: self.trim_vol,
            ..crate::histpeaks::HistPeakParams::default()
        }
    }
}

impl Default for WatershedParams {
    // Backs the struct-level `#[serde(default)]` (fills a field missing from a
    // present `watershed` block). Uses the power-fit defaults; the power/phase
    // variants differ only in field values, and a partial watershed block is a
    // hand-edit edge case (writers always emit the full block).
    fn default() -> Self { default_power_watershed() }
}

/// Deserialize a [`Bound`] whose "no value" case is an open end rather than a
/// data-derived one.
///
/// `detection.freq_max` has no derive-from-data behavior — its absent state is
/// simply "unbounded". Older files spell that `null`, which would otherwise
/// land on [`Bound::Derived`] and serialize back as `"derived"`, leaving a
/// field whose only derivation is infinity claiming to be computed from the
/// data. Normalizing to `"inf"` on read keeps the written file honest.
pub fn bound_open_when_absent<'de, D>(de: D) -> Result<Bound, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Ok(match Bound::deserialize(de)? {
        Bound::Derived => Bound::PosInf,
        other => other,
    })
}

/// A fit bound that can be a number, an open end, or a rule.
///
/// MATLAB's `LB_default`/`UB_default` carry three distinct states: `NaN`
/// means "derive from the data" (`param_basis_power.m:246-265` branches on
/// `isnan`), `±Inf` means an open end, and anything else is used verbatim.
/// JSON has no infinity literal, so the two non-numeric states are spelled
/// as strings:
///
/// | JSON | meaning |
/// |---|---|
/// | `"derived"` | compute from the valid-data extent (MATLAB `NaN`) |
/// | `"-inf"` / `"inf"` | open end (MATLAB `±Inf`) |
/// | `12.0` | explicit finite bound |
/// | `null` or absent | unset, so the default — which is `Derived` |
///
/// `null` is accepted on read and always will be, because older settings
/// files use it. It is never written: "absent" is a poor way to say "compute
/// this from the data" when every state has a name.
///
/// The named spellings also keep an open end from being lost. JSON cannot
/// hold a non-finite number, so an infinite bound written as a number would
/// serialize as `null` and read back as "derive".
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum Bound {
    /// Derive from the valid-data extent for this axis.
    #[default]
    Derived,
    /// Open upper end (`+inf`).
    PosInf,
    /// Open lower end (`-inf`).
    NegInf,
    /// Explicit finite bound.
    Value(f64),
}

impl Bound {
    /// Resolve to the number the fitter uses. `derived` is the value this
    /// axis computes from its data, evaluated by the caller.
    pub fn resolve(self, derived: f64) -> f64 {
        match self {
            Bound::Derived => derived,
            Bound::PosInf => f64::INFINITY,
            Bound::NegInf => f64::NEG_INFINITY,
            Bound::Value(v) => v,
        }
    }

    /// Whether the user pinned something explicit — an actual number or an
    /// explicitly open end. `Derived` is not "set"; it is the default rule.
    pub fn is_set(self) -> bool {
        !matches!(self, Bound::Derived)
    }
}

impl Serialize for Bound {
    fn serialize<S: serde::Serializer>(&self, ser: S) -> Result<S::Ok, S::Error> {
        match self {
            Bound::Derived => ser.serialize_str("derived"),
            Bound::PosInf => ser.serialize_str("inf"),
            Bound::NegInf => ser.serialize_str("-inf"),
            // A non-finite slipping in here would serialize as `null` and read
            // back as Derived, so map it onto the sentinel it actually means.
            Bound::Value(v) if v.is_infinite() => {
                ser.serialize_str(if *v > 0.0 { "inf" } else { "-inf" })
            }
            Bound::Value(v) if v.is_nan() => ser.serialize_str("derived"),
            Bound::Value(v) => ser.serialize_f64(*v),
        }
    }
}

impl<'de> Deserialize<'de> for Bound {
    fn deserialize<D: serde::Deserializer<'de>>(de: D) -> Result<Self, D::Error> {
        use serde::de::Error;
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Raw { Null, Num(f64), Str(String) }
        match Raw::deserialize(de)? {
            Raw::Null => Ok(Bound::Derived),
            Raw::Num(v) if v.is_nan() => Ok(Bound::Derived),
            Raw::Num(v) if v.is_infinite() => {
                Ok(if v > 0.0 { Bound::PosInf } else { Bound::NegInf })
            }
            Raw::Num(v) => Ok(Bound::Value(v)),
            Raw::Str(t) => match t.trim().to_ascii_lowercase().as_str() {
                "derived" | "derive" | "auto" | "nan" => Ok(Bound::Derived),
                "inf" | "+inf" | "infinity" | "+infinity" => Ok(Bound::PosInf),
                "-inf" | "-infinity" => Ok(Bound::NegInf),
                other => Err(D::Error::custom(format!(
                    "bound must be a number, \"derived\", \"inf\" or \"-inf\"; got \"{other}\""
                ))),
            },
        }
    }
}

/// Per-mode parameter bounds. Full analogue of MATLAB's
/// `LB_default = [amp, fmean, fstd, pmean, pstd, theta]` / `UB_default`.
///
/// MATLAB uses `NaN` in a slot to mean "derive this from the data" and a
/// number to mean "use exactly this". The six [`Bound`] fields below carry
/// the same choice: `"derived"` (or `null`) derives, a number pins, and
/// `"inf"` / `"-inf"` leave that end open.
///
/// The three always-numeric slots (`fstd`, `pstd`, `theta`) have no derived
/// form in MATLAB either, so they stay plain `f64`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ModeParamBounds {
    /// FreqStd (Hz σ, both axes — kernels use `exp(-½(d/σ)²)`).
    pub fstd_lb: f64,
    pub fstd_ub: f64,
    /// SOpowerStd (dB) for power; recikappa (rad) for phase.
    pub pstd_lb: f64,
    pub pstd_ub: f64,
    /// Rotation angle (rad). MATLAB always sets this symmetric.
    pub theta_lb: f64,
    pub theta_ub: f64,

    // ── MATLAB's NaN slots. `"derived"` (or `null`) derives from the data. ──
    /// Density/amplitude bounds. `null` derives them from the seed
    /// amplitudes as `min/10` and `max·10` (MATLAB
    /// `param_basis_power.m:246-251`).
    pub amp_lb: Bound,
    pub amp_ub: Bound,
    /// FreqMean (Hz) bounds. `null` derives them from the frequency extent
    /// of the valid data inside the fit window. Ignored entirely when
    /// `constrain_freq_center` is false, which forces ±inf.
    ///
    /// **`null` is not "unbounded".** Three states exist: derive-from-data
    /// (`"derived"` or `null`), an explicit finite bound (a number), and an
    /// open end (`"inf"` or `"-inf"`). Turning the axis's
    /// `constrain_*_center` off opens BOTH ends at once. JSON has no infinity
    /// literal, which is why the open end is spelled as a string. See
    /// [`Bound`].
    pub fmean_lb: Bound,
    pub fmean_ub: Bound,
    /// SOpowerMean (dB) for power; SOphaseMean (rad) for phase. `null`
    /// derives from the valid SO extent inside the fit window on the power
    /// axis, and is `±2π` on the phase axis (two periods, so the LM can
    /// cross the ±π seam). Ignored when the axis's `constrain_*_center` is
    /// false.
    pub pmean_lb: Bound,
    pub pmean_ub: Bound,
}

impl Default for ModeParamBounds {
    // Backs the struct-level `#[serde(default)]`. Uses the power-fit defaults;
    // see WatershedParams::default for the power/phase caveat.
    fn default() -> Self { default_power_mode_bounds() }
}

/// Bounds on the power fit's background, a tilted plane
/// `PowSlope·x + FreqSlope·y + Offset` (MATLAB `fit_rotGauss.m:111-113`).
/// Field names match the emitted CSV keys.
///
/// Bounding BOTH the slopes and the offset is deliberate. Clamping the
/// offset alone just pushes the tilt into the slopes, which is empirically
/// worse than either extreme.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct PowerBackgroundBounds {
    /// Slope along the SO-power axis (per dB).
    pub pow_slope_lb: f64,
    pub pow_slope_ub: f64,
    /// Slope along the frequency axis (per Hz).
    pub freq_slope_lb: f64,
    pub freq_slope_ub: f64,
    /// Constant offset. The lower bound is 0 because a negative offset is a
    /// negative density on a count histogram, and an unbounded plane drifts
    /// there to overfit.
    pub offset_lb: f64,
    /// `null` derives the upper bound from the data as `max(hist)`, which is
    /// what MATLAB does. A number pins it.
    pub offset_ub: Bound,
}

/// Bounds on the phase fit's background, a sinusoid
/// `SinAmp·sin(x + SinPhase) + Offset` (MATLAB `fit_vmGauss.m:102-104`).
///
/// `SinAmp` must stay bounded. Under row normalization an unbounded
/// amplitude is a degenerate fit direction, since the normalizer cancels its
/// scale, so the LM drifts into a pure-sinusoid model and squashes the real
/// vmGauss modes.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct PhaseBackgroundBounds {
    /// Sinusoid amplitude.
    pub sin_amp_lb: f64,
    pub sin_amp_ub: f64,
    /// Sinusoid phase offset (rad).
    pub sin_phase_lb: f64,
    pub sin_phase_ub: f64,
    /// Constant offset. Same non-negativity reasoning as the power plane.
    pub offset_lb: f64,
    /// `null` derives from the data as `max(hist)`.
    pub offset_ub: Bound,
}

impl Default for PowerBackgroundBounds {
    fn default() -> Self {
        Self {
            pow_slope_lb: -0.1, pow_slope_ub: 0.1,
            freq_slope_lb: -0.1, freq_slope_ub: 0.1,
            offset_lb: 0.0, offset_ub: Bound::Derived,
        }
    }
}

impl Default for PhaseBackgroundBounds {
    fn default() -> Self {
        let pi = std::f64::consts::PI;
        Self {
            sin_amp_lb: -1.0, sin_amp_ub: 1.0,
            sin_phase_lb: -pi, sin_phase_ub: pi,
            offset_lb: 0.0, offset_ub: Bound::Derived,
        }
    }
}

/// Power-axis paramfit options — full MATLAB `param_basis_opts('power')`
/// surface. Where Rust defaults diverge from MATLAB they're called out
/// in the field docs.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct PowerParamFitOpts {
    /// Fit window on the SO-power axis (dB). MATLAB default `[-2, 20]`.
    /// Also sets the SO-power x-axis of the paramfit diagnostic figure,
    /// mirroring MATLAB `plot_SOPH_paramfits` (which takes its `xlim`
    /// from this same option).
    #[serde(default = "default_power_limits")]
    pub power_limits: [f64; 2],
    /// Fit window on the frequency axis (Hz). Default `[2, 18]`, matching
    /// MATLAB `param_basis_opts.m`. The 2 Hz of headroom above the
    /// analysis band lets the paramfit park a junk-absorber Gaussian
    /// rather than pinning one to the upper bound. Also sets the frequency
    /// y-axis of the paramfit diagnostic figure (MATLAB
    /// `plot_SOPH_paramfits` `ylim`).
    #[serde(default = "default_freq_limits")]
    pub freq_limits: [f64; 2],
    /// Watershed seed-extraction params.
    #[serde(default = "default_power_watershed")]
    pub watershed: WatershedParams,
    /// Maximum number of modes to attempt before terminating.
    /// MATLAB default 6.
    #[serde(default = "default_max_peaks")]
    pub max_peaks: usize,
    /// Maximum pairwise mode overlap (IoU) before revert. Matches
    /// MATLAB `mode_overlap.m` semantics:
    /// sum(min(g_a, g_b)) / sum(max(g_a, g_b)). Default `0.25`, matching
    /// MATLAB `param_basis_opts.m`. At this threshold an iteration that
    /// adds a well-separated slow band is accepted rather than reverted.
    #[serde(default = "default_power_max_overlap")]
    pub max_overlap: f64,
    /// Minimum fitted-amplitude floor (revert if any mode's amp drops
    /// below). MATLAB ties this to `watershed.height_min = 0`.
    #[serde(default = "default_power_min_amp")]
    pub min_amp: f64,
    /// Minimum allowed pairwise difference between mode FreqMean values
    /// (Hz). MATLAB default 0.5; power-only.
    #[serde(default = "default_power_min_freq_diff")]
    pub min_freq_diff: f64,
    /// %ΔadjR² floor that ends the add-modes loop. MATLAB default 0.01.
    /// This is the only stopping rule implemented; MATLAB's other three
    /// (`max`, `mindr2`, `kneedle`) are not, so they are not offered.
    #[serde(default = "default_power_min_pctr2")]
    pub min_pctr2: f64,
    /// Per-mode parameter bounds. The power axis uses tight caps
    /// (DYNAM-O `param_basis_opts('power')` matches):
    ///   - **`fstd_ub = 2.5/√2 Hz`** (a true σ). A wider cap gets pinned
    ///     only by plateau/sheet modes.
    ///   - **`theta_ub = +0.03 rad`** / `theta_lb = -0.03` — a wider cap
    ///     is reached by nearly every multi-iter fit (basis-stretch
    ///     escape, not real rotation).
    #[serde(default = "default_power_mode_bounds")]
    pub mode_bounds: ModeParamBounds,
    /// Bounds on the background plane. See [`PowerBackgroundBounds`].
    #[serde(default)]
    pub background_bounds: PowerBackgroundBounds,
    /// Constrain each fitted mode's FREQUENCY center to the fit window.
    /// Default `true` keeps `fmean ∈ [freq_axis]`. Set
    /// `false` to free it to ±inf. Independent of the SO-power axis.
    #[serde(default = "default_true")]
    pub constrain_freq_center: bool,
    /// Constrain each fitted mode's SO-POWER center to the fit window.
    /// Default `true` keeps `pmean ∈ [so_axis]`, so a mode
    /// stays inside the observed SOPH. Set `false` to free it to ±inf,
    /// letting a mode center lie outside the observed SO-power range (test
    /// for structure the SOPH window can't contain). The fit still runs on
    /// the same windowed histogram; only the center bound changes.
    /// A paramfit option that the Rust code defines first and MATLAB mirrors.
    #[serde(default = "default_true")]
    pub constrain_power_center: bool,
}

/// Phase-axis paramfit options — full MATLAB `param_basis_opts('phase')`
/// surface. Defaults mirror MATLAB.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct PhaseParamFitOpts {
    /// Fit window on the SO-phase axis (rad). MATLAB default `[-π, π]`.
    /// Also sets the SO-phase x-axis of the paramfit diagnostic figure
    /// (MATLAB `plot_SOPH_paramfits` `xlim`).
    #[serde(default = "default_phase_limits")]
    pub phase_limits: [f64; 2],
    /// Fit window on the frequency axis (Hz). Default `[2, 18]`, matching
    /// MATLAB `param_basis_opts.m`. The 2 Hz of headroom above the
    /// analysis band lets the paramfit park a junk-absorber Gaussian
    /// rather than pinning one to the upper bound.
    #[serde(default = "default_freq_limits")]
    pub freq_limits: [f64; 2],
    /// Watershed seed-extraction params (periodic-phase variant).
    #[serde(default = "default_phase_watershed")]
    pub watershed: WatershedParams,
    /// Gaussian pre-filter σ for the periodic-phase watershed, in
    /// MATLAB's `[σ_phase, σ_freq]` order. MATLAB default `[10, 5]`.
    /// Honored — `fit_paramfit_phase` feeds it (axis-swapped to match
    /// our (phase, freq) histogram) into the phase peak extractor.
    #[serde(default = "default_phase_gauss_filt_std")]
    pub gauss_filt_std: [f64; 2],
    /// Maximum number of modes to fit. The fit stops early if extra modes stop
    /// improving it.
    #[serde(default = "default_phase_max_peaks")]
    pub max_peaks: usize,
    /// Maximum pairwise mode overlap (IoU) before revert. MATLAB
    /// default 0.15.
    #[serde(default = "default_phase_max_overlap")]
    pub max_overlap: f64,
    /// Minimum empirical-amplitude floor. MATLAB ties this to
    /// `watershed.height_min = 1e-4`.
    #[serde(default = "default_phase_min_amp")]
    pub min_amp: f64,
    /// MATLAB default 0.025.
    #[serde(default = "default_phase_min_pctr2")]
    pub min_pctr2: f64,
    /// Per-mode parameter bounds. Defaults match MATLAB
    /// `param_basis_opts('phase')`:
    ///   - `fstd ∈ [1/√2, √7.5]` (Hz, a true σ: vmGauss uses
    ///     `exp(-½(dy/fstd)²)`).
    ///   - `pstd ∈ [π/5, 2π]` (recikappa, rad).
    ///   - `theta ∈ [-π/3, π/3]`.
    #[serde(default = "default_phase_mode_bounds")]
    pub mode_bounds: ModeParamBounds,
    /// Bounds on the background sinusoid. See [`PhaseBackgroundBounds`].
    #[serde(default)]
    pub background_bounds: PhaseBackgroundBounds,
    /// Constrain each fitted mode's FREQUENCY center to the fit window.
    /// Default `true` keeps `fmean ∈ [freq_axis]`. Set
    /// `false` to free it to ±inf. Independent of the phase axis.
    #[serde(default = "default_true")]
    pub constrain_freq_center: bool,
    /// Constrain each fitted mode's SO-PHASE center. Default `true` bounds
    /// it to two full periods `[-2π, 2π]`, as MATLAB does: every circular
    /// phase class stays reachable while the LM can cross the ±π seam, and
    /// the reported center is wrapped back to `[-π, π]`. Set `false` to free
    /// the center to ±inf.
    #[serde(default = "default_true")]
    pub constrain_phase_center: bool,
}

// ─────────────────────────── default helpers ─────────────────────────
//
// Each helper is a free function (serde requires that). Kept compact so
// the struct definitions stay readable.
fn default_power_limits() -> [f64; 2] { [-2.0, 20.0] }
fn default_phase_limits() -> [f64; 2] { [-std::f64::consts::PI, std::f64::consts::PI] }
fn default_freq_limits()  -> [f64; 2] { [2.0, 18.0] }

fn default_power_watershed() -> WatershedParams {
    WatershedParams {
        merge_thresh: Bound::Derived, dur_min: 4.0, bw_min: 0.25, height_min: 0.0, trim_vol: 0.7,
    }
}
fn default_phase_watershed() -> WatershedParams {
    WatershedParams {
        merge_thresh: Bound::Derived,
        dur_min: std::f64::consts::FRAC_PI_6,
        bw_min: 2.0, height_min: 1e-4, trim_vol: 0.4,
    }
}
fn default_phase_gauss_filt_std() -> [f64; 2] { [10.0, 5.0] }

fn default_max_peaks() -> usize { 6 }
fn default_phase_max_peaks() -> usize { 3 }
fn default_power_max_overlap() -> f64 { 0.25 }
fn default_phase_max_overlap() -> f64 { 0.15 }
fn default_power_min_amp() -> f64 { 0.0 }
fn default_phase_min_amp() -> f64 { 1e-4 }
fn default_power_min_freq_diff() -> f64 { 0.5 }
fn default_power_min_pctr2() -> f64 { 0.01 }
fn default_phase_min_pctr2() -> f64 { 0.025 }

// The shape-parameter defaults (fstd, pstd, theta) come from the kernel's
// `paramfit::defaults`, the one copy every toolbox reads.
fn default_power_mode_bounds() -> ModeParamBounds {
    let shape = crate::paramfit::defaults::default_power_mode_shape_bounds();
    ModeParamBounds {
        fstd_lb: shape.fstd_lower, fstd_ub: shape.fstd_upper,
        pstd_lb: shape.pstd_lower, pstd_ub: shape.pstd_upper,
        theta_lb: shape.theta_lower, theta_ub: shape.theta_upper,
        // MATLAB's NaN slots: derived from the data.
        amp_lb: Bound::Derived, amp_ub: Bound::Derived,
        fmean_lb: Bound::Derived, fmean_ub: Bound::Derived,
        pmean_lb: Bound::Derived, pmean_ub: Bound::Derived,
    }
}
fn default_phase_mode_bounds() -> ModeParamBounds {
    let shape = crate::paramfit::defaults::default_phase_mode_shape_bounds();
    ModeParamBounds {
        // fstd is a σ in Hz. pstd is recikappa = 1/√κ in radians.
        fstd_lb: shape.fstd_lower, fstd_ub: shape.fstd_upper,
        pstd_lb: shape.pstd_lower, pstd_ub: shape.pstd_upper,
        theta_lb: shape.theta_lower, theta_ub: shape.theta_upper,
        amp_lb: Bound::Derived, amp_ub: Bound::Derived,
        fmean_lb: Bound::Derived, fmean_ub: Bound::Derived,
        // null here means the ±2π two-period default, not a data extent.
        pmean_lb: Bound::Derived, pmean_ub: Bound::Derived,
    }
}

impl Default for PowerParamFitOpts {
    fn default() -> Self {
        Self {
            power_limits: default_power_limits(),
            freq_limits: default_freq_limits(),
            watershed: default_power_watershed(),
            max_peaks: default_max_peaks(),
            max_overlap: default_power_max_overlap(),
            min_amp: default_power_min_amp(),
            min_freq_diff: default_power_min_freq_diff(),
            min_pctr2: default_power_min_pctr2(),
            mode_bounds: default_power_mode_bounds(),
            background_bounds: PowerBackgroundBounds::default(),
            constrain_freq_center: true,
            constrain_power_center: true,
        }
    }
}

impl Default for PhaseParamFitOpts {
    fn default() -> Self {
        Self {
            phase_limits: default_phase_limits(),
            freq_limits: default_freq_limits(),
            watershed: default_phase_watershed(),
            gauss_filt_std: default_phase_gauss_filt_std(),
            max_peaks: default_phase_max_peaks(),
            max_overlap: default_phase_max_overlap(),
            min_amp: default_phase_min_amp(),
            min_pctr2: default_phase_min_pctr2(),
            mode_bounds: default_phase_mode_bounds(),
            background_bounds: PhaseBackgroundBounds::default(),
            constrain_freq_center: true,
            constrain_phase_center: true,
        }
    }
}
