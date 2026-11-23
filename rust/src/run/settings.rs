//! Resolved settings for a DYNAM-O run.
//!
//! Mirrors the MATLAB option structs (`detection_opts`, `baseline_opts`,
//! `SOpowerphasehist_opts`) so a user can round-trip a JSON dump and reload
//! it byte-identical.

use serde::{Deserialize, Serialize};

/// Top-level resolved settings for a single run. Serializes to the
/// `settings.json` written next to each run's outputs.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DynamoSettings {
    pub mtm: MtmOpts,
    pub detection: DetectionOpts,
    pub baseline: BaselineOpts,
    pub sopower: SoPowerOpts,
    pub hist: HistogramOpts,
    pub output: OutputOpts,
    /// Parametric-basis fit options (analogue of MATLAB's
    /// `param_basis_opts`): the fit windows, seed extraction, stopping
    /// thresholds and parameter bounds for each axis, plus the rule that
    /// assigns TF-peaks to fitted modes.
    #[serde(default)]
    pub paramfit: ParamFitOpts,
    /// Spline-basis fit knots and windows (analogue of MATLAB's
    /// `spline_basis_opts`). Exposes the same surface MATLAB does — SO
    /// and frequency windows plus the per-axis knot counts. Spline order
    /// (cubic / order 4) and the boundary-knot padding stay hard-coded in
    /// `fits.rs`, matching MATLAB (which doesn't expose them either).
    #[serde(default)]
    pub splinefit: SplineFitOpts,
    /// EEG resample target (Hz). When `Some(target)` and input fs >
    /// target, the data is downsampled to `target` before any
    /// spectrogram is computed. `None` disables resampling.
    /// Default `Some(100.0)`. This is an app default, not a MATLAB one:
    /// the MATLAB toolbox leaves `resample_freq` empty and runs at the
    /// recording's native rate, so set `resample_freq = 100` there (or
    /// disable resampling here) when the two must produce the same
    /// numbers. A fixed target is the default for two reasons. First,
    /// recordings that will be compared should all be analyzed at one
    /// sampling rate, so every subject shares the same time grid,
    /// filters, and frequency bins no matter what hardware recorded
    /// them. Second, run time grows quickly with sampling rate: every
    /// filtering, artifact, and SO-phase step scales with the sample
    /// count, and the spectrogram FFT length is `fs / dsfreqs` rounded
    /// UP to the next power of two, so it jumps in doublings (1024
    /// points at 100 Hz, 2048 at 128 Hz, 4096 at 256 Hz, 8192 at
    /// 512 Hz with the default 0.1 Hz spacing). The spectral analysis
    /// ends at 30 Hz, well inside the 50 Hz Nyquist limit of a 100 Hz
    /// signal.
    #[serde(default = "default_resample_target_fs")]
    pub resample_target_fs: Option<f64>,
    /// Which pipeline steps to (re)compute on a run. Steps NOT in this
    /// set are reused from existing on-disk artifacts when available.
    /// Default is "compute everything" (matches first-run intent).
    ///
    /// Set by the CLI's `--recompute` flag and the wizard's re-run
    /// checkbox prompt. Separate from `output.write_csv`, which is the
    /// "save the stats CSV" toggle and NOT a cache-reuse signal.
    ///
    /// A per-run directive, NOT an analysis setting — `#[serde(skip)]` keeps it
    /// out of the persisted batch_settings.json and ignores any value in older
    /// files. The source of truth for "what's already done" is the on-disk
    /// artifacts, which the CLI probes directly; persisting the last run's
    /// selection would only cause spurious settings-immutability diffs.
    #[serde(skip)]
    pub recompute: RecomputeSet,
}

fn default_resample_target_fs() -> Option<f64> { Some(100.0) }

impl Default for DynamoSettings {
    fn default() -> Self {
        Self {
            mtm: MtmOpts::default(),
            detection: DetectionOpts::default(),
            baseline: BaselineOpts::default(),
            sopower: SoPowerOpts::default(),
            hist: HistogramOpts::default(),
            output: OutputOpts::default(),
            paramfit: ParamFitOpts::default(),
            splinefit: SplineFitOpts::default(),
            resample_target_fs: default_resample_target_fs(),
            recompute: RecomputeSet::default(),
        }
    }
}

/// Per-step recompute toggle. Mirrors the wizard / GUI output-toggle
/// granularity (`apply_output_toggles` in `cli/src/main.rs`) so the
/// re-run prompt and the wizard expose the same set of choices.
///
/// Seven atomic categories: four pipeline data steps (peaks, sophs,
/// paramfit, splinefit) + three figure outputs (summary, paramfit
/// diagnostic, splinefit diagnostic). `true` means recompute, `false`
/// means reuse the cached artifact from disk when present.
///
/// **The single `figures` key:** a manifest may carry one
/// `figures: true/false` in place of the three per-figure fields. Each
/// per-figure field that is absent takes that value. The custom
/// [`Deserialize`] impl handles this.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct RecomputeSet {
    pub peaks: bool,
    pub sophs: bool,
    pub paramfit: bool,
    pub splinefit: bool,
    /// Summary figure — the big multi-panel PNG. Independent of
    /// paramfit/splinefit; depends on peaks + sophs upstream.
    pub summary_figure: bool,
    /// Paramfit diagnostic figure (vmGauss 2×3 grid).
    pub paramfit_figure: bool,
    /// Splinefit diagnostic figure (smoothing-spline 2×3 grid).
    pub splinefit_figure: bool,
}

impl Default for RecomputeSet {
    fn default() -> Self {
        Self {
            peaks: true,
            sophs: true,
            paramfit: true,
            splinefit: true,
            summary_figure: true,
            paramfit_figure: true,
            splinefit_figure: true,
        }
    }
}

impl RecomputeSet {
    /// Convenience: cache-rebuild mode. Peak detection is reused from
    /// disk and every step downstream of it (SOPHs, both fits, all three
    /// figures) is recomputed.
    pub fn cache_rebuild() -> Self {
        Self {
            peaks: false,
            sophs: true,
            paramfit: true,
            splinefit: true,
            summary_figure: true,
            paramfit_figure: true,
            splinefit_figure: true,
        }
    }

    /// Whether any step at all is ticked.
    pub fn any_set(&self) -> bool {
        self.peaks
            || self.sophs
            || self.paramfit
            || self.splinefit
            || self.summary_figure
            || self.paramfit_figure
            || self.splinefit_figure
    }

    /// Ticked step names in canonical render order (peaks → data → figures).
    fn step_names(&self) -> Vec<&'static str> {
        let mut out = Vec::with_capacity(7);
        if self.peaks { out.push("peaks"); }
        if self.sophs { out.push("sophs"); }
        if self.paramfit { out.push("paramfit"); }
        if self.splinefit { out.push("splinefit"); }
        if self.summary_figure { out.push("summary figure"); }
        if self.paramfit_figure { out.push("paramfit figure"); }
        if self.splinefit_figure { out.push("splinefit figure"); }
        out
    }

    /// One-line human label of the active recompute scope.
    ///
    /// Enumerates the ticked steps in a fixed order so the user sees in one
    /// glance what the run will actually touch. Three formatting rules:
    ///
    /// * If every step is ticked, the label collapses to `full pipeline`.
    /// * If exactly one step is ticked, the label is `<name> only`.
    /// * Otherwise, ticked step names are joined with ` + `.
    ///
    /// An empty set returns `no recompute` defensively — we don't expect to
    /// render it at any user-facing site, but it keeps callers honest if
    /// they do.
    ///
    /// Lives here rather than in the CLI so every front end that announces a
    /// run says the same thing; a label stranded in a binary crate is one the
    /// desktop app cannot reach.
    pub fn scope_label(&self) -> String {
        if !self.any_set() {
            return "no recompute".to_string();
        }
        if self.peaks
            && self.sophs
            && self.paramfit
            && self.splinefit
            && self.summary_figure
            && self.paramfit_figure
            && self.splinefit_figure
        {
            return "full pipeline".to_string();
        }
        let parts = self.step_names();
        if parts.len() == 1 {
            format!("{} only", parts[0])
        } else {
            parts.join(" + ")
        }
    }

    /// Same as [`RecomputeSet::scope_label`] but constrained to the steps
    /// owned by the second pass (everything except `peaks`).
    ///
    /// Used by the announcement / banner / per-item worker bar so the message
    /// never names `peaks` (which that pass never touches) and never names a
    /// step that isn't actually being recomputed.
    pub fn phase2_scope_label(&self) -> String {
        let mut p2 = self.clone();
        p2.peaks = false;
        if !p2.any_set() {
            return "no recompute".to_string();
        }
        if p2.sophs
            && p2.paramfit
            && p2.splinefit
            && p2.summary_figure
            && p2.paramfit_figure
            && p2.splinefit_figure
        {
            return "sophs + fits + figures".to_string();
        }
        let parts = p2.step_names();
        if parts.len() == 1 {
            format!("{} only", parts[0])
        } else {
            parts.join(" + ")
        }
    }
}

// Custom Deserialize: accept both the current schema (with the three
// per-figure fields) and the legacy schema (with a single `figures`
// field that fans out to all three). Missing fields default to `true`
// so a partial manifest still produces a sensible default-on set.
impl<'de> Deserialize<'de> for RecomputeSet {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Raw {
            #[serde(default = "default_true")]
            peaks: bool,
            #[serde(default = "default_true")]
            sophs: bool,
            #[serde(default = "default_true")]
            paramfit: bool,
            #[serde(default = "default_true")]
            splinefit: bool,
            // Legacy single-figure field. When present and no per-figure
            // fields are, it sets all three new slots.
            figures: Option<bool>,
            summary_figure: Option<bool>,
            paramfit_figure: Option<bool>,
            splinefit_figure: Option<bool>,
        }
        let r = Raw::deserialize(deserializer)?;
        let legacy = r.figures.unwrap_or(true);
        Ok(RecomputeSet {
            peaks: r.peaks,
            sophs: r.sophs,
            paramfit: r.paramfit,
            splinefit: r.splinefit,
            summary_figure: r.summary_figure.unwrap_or(legacy),
            paramfit_figure: r.paramfit_figure.unwrap_or(legacy),
            splinefit_figure: r.splinefit_figure.unwrap_or(legacy),
        })
    }
}

/// Multitaper spectrogram options shared across pass-1, pass-2, and the
/// SO-band spectrogram. Matches the `mtm_*` fields under MATLAB
/// `detection_opts`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct MtmOpts {
    /// DPSS time-half-bandwidth product (typically 2.0).
    pub nw: f64,
    /// Number of tapers (typically 3).
    pub n_tapers: usize,
    /// Pass-1 window length (s). Default 1.0.
    pub window_length_pass1: f64,
    /// Pass-2 window length (s). Default 2.0.
    pub window_length_pass2: f64,
    /// Window step (s). Default 0.05.
    pub window_step: f64,
    /// Frequency-bin resolution target; nfft = next_pow2(fs / dsfreqs).
    pub dsfreqs: f64,
    /// Frequency range (Hz).
    pub freq_range: (f64, f64),
}

impl Default for MtmOpts {
    fn default() -> Self {
        Self {
            nw: 2.0,
            n_tapers: 3,
            window_length_pass1: 1.0,
            window_length_pass2: 2.0,
            window_step: 0.05,
            dsfreqs: 0.1,
            freq_range: (0.0, 30.0),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DetectionOpts {
    /// Length of each analysis segment, in seconds. The recording is cut into
    /// segments this long and peaks are found in each.
    pub seg_time: f64,
    /// Shrink the spectrogram by this factor along frequency before finding
    /// peaks. Higher means faster and coarser.
    pub downsample_f: u32,
    /// Shrink the spectrogram by this factor along time before finding peaks.
    /// Higher means faster and coarser.
    pub downsample_t: u32,
    /// Merge two neighboring peaks when the dip between them is shallower than
    /// this, in power units. Higher merges more aggressively.
    pub merge_thresh: f64,
    /// How much of each peak to keep when trimming its edges, as a fraction of
    /// total volume. 0.8 keeps the densest 80%.
    pub trim_vol_thresh: f64,
    // `dur_min` / `bw_min` are NOT settings. They are derived per pass from
    // the multitaper window length and taper bandwidth (see
    // `orchestrator::PassMtmSpec`), matching MATLAB, which does not expose
    // them either.
    /// Discard peaks longer than this many seconds. Very long blobs are usually
    /// artifact, not oscillation.
    pub dur_max: f64,
    /// Discard peaks wider than this many Hz. Very wide blobs are usually
    /// broadband noise.
    pub bw_max: f64,
    /// Ignore peaks below this frequency, in Hz.
    pub freq_min: f64,
    /// `"inf"` = no upper bound, which is the default and matches MATLAB
    /// `freq_max = Inf`. A number pins the upper edge. `null` still parses,
    /// from files written before the spelling existed.
    #[serde(deserialize_with = "bound_open_when_absent")]
    pub freq_max: Bound,
}

impl Default for DetectionOpts {
    fn default() -> Self {
        Self {
            seg_time: 30.0,
            downsample_f: 2,
            downsample_t: 2,
            merge_thresh: 11.0,
            trim_vol_thresh: 0.8,
            dur_max: 5.0,
            bw_max: 15.0,
            freq_min: 0.0,
            freq_max: Bound::PosInf,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct BaselineOpts {
    /// Percentile of the power distribution used as the baseline to subtract. 2
    /// means the 2nd percentile, i.e. near the quiet floor.
    pub ptile: f64,
    /// Which sleep stages contribute to the baseline. Codes are 1=N3, 2=N2,
    /// 3=N1, 4=REM, 5=wake.
    pub stages: Vec<f64>,
}

impl Default for BaselineOpts {
    fn default() -> Self {
        // MATLAB `baseline_opts.m`: baseline_stages = [1,2,3,4,5],
        // baseline_ptile = 2. `baseline_trim` is not a setting here. The
        // orchestrator always uses an unbounded `[-Inf, Inf]` baseline
        // time range.
        Self {
            ptile: 2.0,
            stages: vec![1.0, 2.0, 3.0, 4.0, 5.0],
        }
    }
}

/// SO-band multitaper spectrogram + normalization options. The `band`
/// here is the **single** SO frequency range used by BOTH the SO-power
/// integration and the SO-phase bandpass (MATLAB
/// `SOpowerphasehist_opts.SO_freqrange`, default `[0.3, 1.5]` Hz).
/// There is only ever one slow-oscillation band per recording.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SoPowerOpts {
    /// The slow-oscillation frequency band, in Hz. Used both to measure SO
    /// power and to extract SO phase.
    pub band: (f64, f64),
    /// How SO power is normalized before binning, so recordings with different
    /// overall amplitude can be compared.
    pub norm_method: String,
    /// Drop slow-oscillation power values more than this many standard
    /// deviations from the mean before building histograms.
    pub outlier_threshold: f64,
    /// MATLAB `SOpower_tapers` `[NW, K]` — controls the DPSS basis for
    /// the SO-band spectrogram. Default `(5.0, 9)`.
    #[serde(default = "default_sopower_tapers")]
    pub tapers: (f64, usize),
    /// MATLAB `SOpower_window_params` `[window_s, step_s]` — the SO-
    /// band spectrogram window. Default `(5.0, 0.5)`.
    #[serde(default = "default_sopower_window_params")]
    pub window_params: (f64, f64),
    /// MATLAB `SOpower_retain_Fs` — when `true`, the normalized SO-
    /// power trace is upsampled back to the EEG sample rate so per-
    /// peak interpolation is over the dense grid. Default `true`.
    #[serde(default = "default_sopower_retain_fs")]
    pub retain_fs: bool,
}

fn default_sopower_tapers() -> (f64, usize) { (5.0, 9) }
fn default_sopower_window_params() -> (f64, f64) { (5.0, 0.5) }
fn default_sopower_retain_fs() -> bool { true }

impl Default for SoPowerOpts {
    fn default() -> Self {
        Self {
            band: (0.3, 1.5),
            norm_method: "p2shift1234".to_string(),
            outlier_threshold: 3.0,
            tapers: default_sopower_tapers(),
            window_params: default_sopower_window_params(),
            retain_fs: default_sopower_retain_fs(),
        }
    }
}


// The fit options live in the kernel so every caller (this app, the CLI,
// the Python and MATLAB wrappers) shares one definition. Re-exported here
// under their settings names.
pub use crate::paramfit::opts::{
    bound_open_when_absent, Bound, ModeParamBounds, ParamFitOpts, PeakAssign,
    PhaseBackgroundBounds, PhaseParamFitOpts, PowerBackgroundBounds, PowerParamFitOpts,
    SplineFitOpts, SplinePhaseOpts, SplinePowerOpts, WatershedParams,
    DEFAULT_DENSITY_FOOTPRINT_SIGMA, PEAK_ASSIGN_PHASE_FALLBACK,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct HistogramOpts {
    /// A histogram bin needs at least this many minutes of data to be used.
    /// Thinner bins are left empty rather than trusted.
    pub min_time_in_bin: f64,
    /// TF-peak frequency range (Hz) included in the SO-power/phase
    /// histograms. Default `[2, 18]`, matching MATLAB
    /// `SOpowerphasehist_opts.m:94`. The upper edge at 18 Hz (vs the
    /// analysis band's 16 Hz) leaves room for the paramfit to park a
    /// junk-absorber Gaussian above the analysis window, which otherwise
    /// pins flush against the upper bound.
    /// **Independent** of the multitaper spectrogram's frequency range
    /// (`MtmOpts::freq_range`) — deriving the SOPH bins from the spectrogram
    /// range silently produces a wrong frequency axis whenever a user
    /// narrows the spectrogram range.
    #[serde(default = "default_soph_freq_range")]
    pub freq_range: (f64, f64),
    /// MATLAB `freq_binsizestep` — `(bin_width, bin_step)` for the
    /// frequency axis of every SOPH. Default `(1.0, 0.2)` Hz.
    #[serde(default = "default_freq_binsizestep")]
    pub freq_binsizestep: (f64, f64),
    /// MATLAB `SOpower_range` — `Some((lo, hi))` pins the SO-power
    /// axis across runs (the only way to make cross-subject SOPH
    /// averages meaningful); `None` falls back to the per-run
    /// min/max-of-data range. Default `Some((-5, 25))` dB, matching MATLAB
    /// `SOpowerphasehist_opts.m:114` — pinned so cross-subject means are
    /// comparable out of the box, with the adaptive path still reachable
    /// by setting the field to `null`.
    #[serde(default = "default_sopower_range")]
    pub sopower_range: Option<(f64, f64)>,
    /// MATLAB `SOpower_binsizestep` — `(width, step)` for SO-power
    /// bins. `None` derives `(range / 10, range / 100)` at run time
    /// from whichever SO-power range is in effect. Default
    /// `Some((2.5, 0.25))` — 2.5 dB-wide bins stepping 0.25 dB
    /// across the default `[-5, 25]` dB range.
    #[serde(default = "default_sopower_binsizestep")]
    pub sopower_binsizestep: Option<(f64, f64)>,
    /// MATLAB `SOphase_range` — SO-phase axis range. Default
    /// `(-π, π)`.
    #[serde(default = "default_sophase_range")]
    pub sophase_range: (f64, f64),
    /// MATLAB `SOphase_binsizestep` — `(width, step)` for the SO-
    /// phase axis. Default `(2π/5, 2π/100)`.
    #[serde(default = "default_sophase_binsizestep")]
    pub sophase_binsizestep: (f64, f64),
    /// MATLAB `compute_rate` — when `true`, the SO-power histogram
    /// values are peak counts divided by time-in-bin (rate / Hz·min);
    /// `false` yields raw counts. Default `true`. **Power axis
    /// only**: the SO-phase histogram is intentionally kept as
    /// freq-axis-normalized density (`soph_build` forces
    /// `compute_rate = false` for it) because the phase axis is circular
    /// probability rather than a rate.
    #[serde(default = "default_compute_rate")]
    pub compute_rate: bool,
    /// MATLAB `SOphase_norm_dim` — which axis the phase histogram is
    /// density-normalized over (1 = freq axis, 0 = SO-phase axis).
    /// Default `1`.
    #[serde(default = "default_sophase_norm_dim")]
    pub sophase_norm_dim: u8,
    /// MATLAB `SOphase_min_peak_at_freq` — minimum peak count at a
    /// given frequency below which that frequency row is dropped
    /// from the phase histogram. Default `0` (disabled).
    #[serde(default)]
    pub sophase_min_peak_at_freq: u32,
    /// Stage codes to include in SOPHs. Matches MATLAB
    /// `SOPH_stages`. Default `[1, 2, 3]` (N3, N2, N1 — NREM only).
    /// Empty disables filtering (all stages included).
    #[serde(default = "default_soph_stages")]
    pub soph_stages: Vec<u8>,
}

fn default_soph_stages() -> Vec<u8> { vec![1, 2, 3] }
fn default_soph_freq_range() -> (f64, f64) { (2.0, 18.0) }
fn default_freq_binsizestep() -> (f64, f64) { (1.0, 0.2) }
fn default_sopower_range() -> Option<(f64, f64)> { Some((-5.0, 25.0)) }
fn default_sopower_binsizestep() -> Option<(f64, f64)> { Some((2.5, 0.25)) }
fn default_sophase_range() -> (f64, f64) {
    (-std::f64::consts::PI, std::f64::consts::PI)
}
fn default_sophase_binsizestep() -> (f64, f64) {
    (
        2.0 * std::f64::consts::PI / 5.0,
        2.0 * std::f64::consts::PI / 100.0,
    )
}
fn default_compute_rate() -> bool { true }
fn default_sophase_norm_dim() -> u8 { 1 }

impl Default for HistogramOpts {
    fn default() -> Self {
        // Defaults match MATLAB SOpowerphasehist_opts.m.
        Self {
            min_time_in_bin: 10.0,
            freq_range: default_soph_freq_range(),
            freq_binsizestep: default_freq_binsizestep(),
            sopower_range: default_sopower_range(),
            sopower_binsizestep: default_sopower_binsizestep(),
            sophase_range: default_sophase_range(),
            sophase_binsizestep: default_sophase_binsizestep(),
            compute_rate: default_compute_rate(),
            sophase_norm_dim: default_sophase_norm_dim(),
            sophase_min_peak_at_freq: 0,
            soph_stages: default_soph_stages(),
        }
    }
}

/// Bitmap format for the saved figures (`figures/summary/*`, `figures/
/// param_basis/*`, `figures/spline_basis/*`). The renderer always produces
/// a 300-DPI PNG first; `finalize_figure` then Lanczos-resizes to
/// `figure_dpi` and transcodes to this format. Only bitmap formats — SVG
/// is on-demand via the Results Browser Export button (the cluster CLI
/// has no browser engine).
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum FigureFormat {
    /// Lossless, smallest for our colormap-heavy renders (default).
    #[default]
    Png,
    /// Lossy (~q92) — review-only / Slack thumbnails.
    Jpg,
    /// Lossless + widely archival; ~1.5–2× larger than PNG.
    Tiff,
}

impl FigureFormat {
    /// File extension to use for figures written in this format.
    pub fn extension(&self) -> &'static str {
        match self {
            FigureFormat::Png  => "png",
            FigureFormat::Jpg  => "jpg",
            FigureFormat::Tiff => "tiff",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct OutputOpts {
    /// Peak-stats table CSV (`TFpeaks/<subj>_stats_table_<chan>.csv`).
    #[serde(default = "default_true")]
    pub write_csv: bool,
    /// SOPH histogram TIFFs (`SOPHs/<subj>_SOPHs_{power,phase}_<chan>.tiff`).
    #[serde(default = "default_true")]
    pub write_sophs: bool,
    /// Auxiliary-data HDF5 (`auxiliary_data/<subj>_auxiliary_data_<chan>.h5`).
    #[serde(default = "default_true")]
    pub write_aux: bool,
    /// Save the histograms as multi-page TIFF images.
    #[serde(default = "default_true")]
    pub write_tiff: bool,
    /// Spectrogram tile-pyramid output under `<channel>/tiles/<z>/<x>_<y>.png`,
    /// consumed by the desktop UI's interactive Spectrogram panel
    /// (OpenLayers-style pan/zoom). Off by default because it adds
    /// ~3–5 MB of PNG tiles per run.
    ///
    /// The UI flips this true when the Spectrogram panel is enabled.
    #[serde(default)]
    pub write_tiles: bool,
    /// Parametric (rotated-Gaussian for SO-power, von-Mises × Gaussian
    /// for SO-phase) basis fits over the SOPHs. When `true`, the
    /// orchestrator runs `fit_paramfit_*` and the export layer writes
    /// `<chan>/param_basis/<subj>_SO{power,phase}_paramfit_<chan>.csv`.
    /// Defaults to `true`.
    #[serde(default = "default_true")]
    pub write_paramfits: bool,
    /// Tensor-product B-spline basis fits over the SOPHs. When `true`,
    /// the orchestrator runs `fit_splinefit_*` and the export layer
    /// writes `<chan>/spline_basis/<subj>_SO{power,phase}_splinefit_<chan>.tiff`.
    #[serde(default = "default_true")]
    pub write_splinefits: bool,
    /// Output figure DPI (pixel resolution). Internally the figure
    /// renderer always works at 300 DPI (the canonical MATLAB resolution);
    /// the result is then downsampled to this DPI with a Lanczos filter
    /// and encoded in `figure_format`. 200 (default)
    /// → ~55% smaller files at near-identical on-screen sharpness; 300
    /// keeps the canonical print-grade output (no downsample); 100/150
    /// for quick browsing.
    #[serde(default = "default_figure_dpi")]
    pub figure_dpi: u32,
    /// Bitmap format the figure is encoded in. `Png` (default) is lossless +
    /// smallest for our colormap-heavy renders. `Tiff` is lossless +
    /// widely archival but ~1.5–2× larger. `Jpg` is lossy (q≈92) for
    /// review-only / Slack-thumbnail use cases. Vector output (SVG) is
    /// available on-demand via the Results Browser Export button; it is
    /// not a batch-time option (the cluster CLI ships without a browser
    /// engine and the existing per-figure Export already covers it).
    #[serde(default)]
    pub figure_format: FigureFormat,
    /// 5-panel summary dashboard PNG
    /// (`figures/summary/<subj>_summary_figure_<chan>.png`).
    #[serde(default = "default_true")]
    pub write_summary_png: bool,
    /// Parametric-fit diagnostic PNG (the 2×3 grid mirroring MATLAB's
    /// `plot_SOPH_paramfits.m`). Rendered only when `write_paramfits`
    /// also produced a fit.
    #[serde(default = "default_true")]
    pub write_param_png: bool,
    /// Spline-fit diagnostic PNG (the 2×3 grid mirroring MATLAB's
    /// `plot_SOPH_splinefits.m`). Rendered only when `write_splinefits`
    /// also produced a fit.
    #[serde(default = "default_true")]
    pub write_spline_png: bool,

    // ── Stats-CSV per-column precision ─────────────────────────────────
    // All fields are storage-only knobs — they round the column before
    // text-encoding. The pipeline never reads them back; downstream
    // analysis sees the rounded values. `StatsCsvPrecision::resolve` in
    // the kernel's `crate::pipeline` walks these together with the run's `fs`
    // and warns + bumps any user override that would lose sub-sample /
    // sub-bin precision (the standing "do not corrupt scientific
    // findings" rule). Defaults are safe for any realistic recording.

    /// Time-column significant figures (`PeakTime`, `Duration`,
    /// `bbox_tl_s`). `None` (default) = derive from `fs`
    /// via `time_sigfigs_for_fs(fs)`; sub-sample resolved everywhere up
    /// to 24 h at 1 kHz. An explicit value is validated against the auto
    /// floor at run start and bumped up if too coarse.
    #[serde(default)]
    pub csv_time_sigfigs: Option<u8>,
    /// Frequency-column significant figures (`PeakFrequency`,
    /// `Bandwidth`, `bbox_tl_Hz`). Default 6 → rel.
    /// 5e-7, finer than any spectrogram bin or centroid noise.
    #[serde(default = "default_csv_sigfigs")]
    pub csv_freq_sigfigs: u8,
    /// `SOphase` column significant figures (radians). Default 6 — same
    /// rel. 5e-7 across the bounded (-π, π] range = ~1.5e-6 rad.
    #[serde(default = "default_csv_sigfigs")]
    pub csv_phase_sigfigs: u8,
    /// Significant figures for the variable-magnitude "other" floats —
    /// per-column so a user can keep `Volume` short while leaving
    /// `Peakiness` at higher precision, etc. Default 6 everywhere (relative
    /// round-trip ~5e-7, orders of magnitude below any physical noise).
    #[serde(default = "default_csv_sigfigs")]
    pub csv_height_sigfigs: u8,
    /// Significant figures for the volume column in the peak CSV.
    #[serde(default = "default_csv_sigfigs")]
    pub csv_volume_sigfigs: u8,
    /// Significant figures for the area column in the peak CSV.
    #[serde(default = "default_csv_sigfigs")]
    pub csv_area_sigfigs: u8,
    /// Significant figures for the peakiness column in the peak CSV.
    #[serde(default = "default_csv_sigfigs")]
    pub csv_peakiness_sigfigs: u8,
    /// Significant figures for the SO-power column in the peak CSV.
    #[serde(default = "default_csv_sigfigs")]
    pub csv_sopower_sigfigs: u8,
}

fn default_true() -> bool { true }
fn default_figure_dpi() -> u32 { 200 }
fn default_csv_sigfigs() -> u8 { 6 }

impl Default for OutputOpts {
    fn default() -> Self {
        Self {
            write_csv: true,
            write_sophs: true,
            write_aux: true,
            write_tiff: true,
            write_tiles: false,
            write_paramfits: true,
            write_splinefits: true,
            figure_dpi: 200,
            figure_format: FigureFormat::Png,
            write_summary_png: true,
            write_param_png: true,
            write_spline_png: true,
            csv_time_sigfigs: None,
            csv_freq_sigfigs: 6,
            csv_phase_sigfigs: 6,
            csv_height_sigfigs: 6,
            csv_volume_sigfigs: 6,
            csv_area_sigfigs: 6,
            csv_peakiness_sigfigs: 6,
            csv_sopower_sigfigs: 6,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The extractor parameters carry every watershed option across, and a
    /// derived merge threshold becomes the extractor's `NaN` sentinel.
    #[test]
    fn watershed_params_convert_to_extractor_params() {
        let derived = WatershedParams {
            merge_thresh: Bound::Derived,
            dur_min: 4.0,
            bw_min: 0.25,
            height_min: 0.5,
            trim_vol: 0.7,
        };
        let params = derived.to_hist_peak_params();
        assert!(params.merge_thresh.is_nan());
        assert_eq!(params.dur_min, 4.0);
        assert_eq!(params.bw_min, 0.25);
        assert_eq!(params.height_min, 0.5);
        assert_eq!(params.trim_vol, 0.7);

        let explicit = WatershedParams { merge_thresh: Bound::Value(8.0), ..derived };
        assert_eq!(explicit.to_hist_peak_params().merge_thresh, 8.0);
    }

    #[test]
    fn sparse_json_fills_defaults_at_every_depth() {
        // Backward compat: a settings file that predates fields (missing whole
        // blocks, missing nested fields) must load with each absent value
        // filled from its default — never error. Struct-level `#[serde(default)]`
        // is what guarantees this without a per-field annotation.
        let def = DynamoSettings::default();

        // Empty object → all defaults.
        let empty: DynamoSettings = serde_json::from_str("{}").expect("empty object");
        assert_eq!(empty.mtm.n_tapers, def.mtm.n_tapers);
        assert_eq!(empty.paramfit.power.power_limits, def.paramfit.power.power_limits);

        // A present block missing most fields; a set field wins, the rest default.
        let sparse: DynamoSettings = serde_json::from_str(
            r#"{ "mtm": { "nw": 5.0 },
                 "paramfit": { "power": { "power_limits": [-30.0, 50.0] } } }"#,
        )
        .expect("sparse object");
        assert_eq!(sparse.mtm.nw, 5.0, "set field wins");
        assert_eq!(sparse.mtm.n_tapers, def.mtm.n_tapers, "sibling field defaults");
        assert_eq!(sparse.paramfit.power.power_limits, [-30.0, 50.0], "nested set field wins");
        // A brand-new nested flag absent from the file still defaults on.
        assert!(
            sparse.paramfit.power.constrain_power_center,
            "missing new nested field defaults to true",
        );
        assert_eq!(sparse.hist.sopower_range, def.hist.sopower_range, "absent block defaults");
    }

    #[test]
    fn settings_round_trip_json() {
        let s = DynamoSettings::default();
        let j = serde_json::to_string(&s).unwrap();
        let s2: DynamoSettings = serde_json::from_str(&j).unwrap();
        assert_eq!(s.detection.seg_time, s2.detection.seg_time);
        assert_eq!(s.baseline.ptile, s2.baseline.ptile);
    }

    fn empty_recompute() -> RecomputeSet {
        RecomputeSet {
            peaks: false,
            sophs: false,
            paramfit: false,
            splinefit: false,
            summary_figure: false,
            paramfit_figure: false,
            splinefit_figure: false,
        }
    }

    #[test]
    fn scope_label_empty() {
        assert_eq!(empty_recompute().scope_label(), "no recompute");
        assert_eq!(empty_recompute().phase2_scope_label(), "no recompute");
    }

    #[test]
    fn scope_label_full() {
        let r = RecomputeSet::default();
        assert_eq!(r.scope_label(), "full pipeline");
        assert_eq!(r.phase2_scope_label(), "sophs + fits + figures");
    }

    #[test]
    fn scope_label_peaks_only() {
        let mut r = empty_recompute();
        r.peaks = true;
        assert_eq!(r.scope_label(), "peaks only");
        assert_eq!(r.phase2_scope_label(), "no recompute");
    }

    #[test]
    fn scope_label_splinefit_only() {
        let mut r = empty_recompute();
        r.splinefit = true;
        assert_eq!(r.scope_label(), "splinefit only");
        assert_eq!(r.phase2_scope_label(), "splinefit only");
    }

    #[test]
    fn scope_label_sophs_paramfit() {
        let mut r = empty_recompute();
        r.sophs = true;
        r.paramfit = true;
        assert_eq!(r.scope_label(), "sophs + paramfit");
        assert_eq!(r.phase2_scope_label(), "sophs + paramfit");
    }

    /// The second pass never touches peaks, so naming it would misreport
    /// the run.
    #[test]
    fn scope_label_strips_peaks_from_phase2() {
        let mut r = empty_recompute();
        r.peaks = true;
        r.splinefit = true;
        assert_eq!(r.scope_label(), "peaks + splinefit");
        assert_eq!(r.phase2_scope_label(), "splinefit only");
    }

    /// The three states must survive a round trip and stay distinct. If an
    /// open end serialized to `null`, it would silently read back as "derive
    /// from data".
    #[test]
    fn bound_states_round_trip_and_stay_distinct() {
        let cases = [
            (Bound::Derived, "\"derived\""),
            (Bound::PosInf, "\"inf\""),
            (Bound::NegInf, "\"-inf\""),
            (Bound::Value(12.5), "12.5"),
        ];
        for (b, want_json) in cases {
            let j = serde_json::to_string(&b).unwrap();
            assert_eq!(j, want_json, "serialize {b:?}");
            let back: Bound = serde_json::from_str(&j).unwrap();
            assert_eq!(back, b, "round trip {b:?}");
        }
        // Every state resolves to something different.
        assert_eq!(Bound::Derived.resolve(7.0), 7.0);
        assert!(Bound::PosInf.resolve(7.0).is_infinite());
        assert!(Bound::NegInf.resolve(7.0) < 0.0);
        assert_eq!(Bound::Value(12.5).resolve(7.0), 12.5);
    }

    /// Files written before this type used `null`, and there are a few
    /// hundred of them. They must keep reading as "derive", forever.
    #[test]
    fn bound_accepts_legacy_null_and_matlab_spellings() {
        for txt in ["null", "\"derived\"", "\"auto\"", "\"NaN\""] {
            assert_eq!(serde_json::from_str::<Bound>(txt).unwrap(), Bound::Derived, "{txt}");
        }
        for txt in ["\"inf\"", "\"+inf\"", "\"Infinity\""] {
            assert_eq!(serde_json::from_str::<Bound>(txt).unwrap(), Bound::PosInf, "{txt}");
        }
        assert_eq!(serde_json::from_str::<Bound>("\"-inf\"").unwrap(), Bound::NegInf);
        // A typo must fail loudly rather than silently becoming a default.
        assert!(serde_json::from_str::<Bound>("\"derivedd\"").is_err());
    }

    /// A non-finite that slipped in programmatically must not serialize as
    /// `null` and read back as Derived, which would silently turn an open
    /// bound into a data-derived one.
    #[test]
    fn non_finite_value_does_not_collapse_to_derived() {
        let j = serde_json::to_string(&Bound::Value(f64::INFINITY)).unwrap();
        assert_eq!(j, "\"inf\"");
        assert_eq!(serde_json::from_str::<Bound>(&j).unwrap(), Bound::PosInf);
    }

    /// The two non-fit bounds must normalize to the concept they actually
    /// encode, not to whichever variant `null` happens to land on.
    /// `freq_max` has no data-derived state — its empty case is "unbounded" —
    /// while `merge_thresh` genuinely is a derive rule (the extractor reads
    /// its NaN as a volume-percentile autosearch).
    #[test]
    fn legacy_nulls_normalize_to_the_right_concept() {
        let d: DetectionOpts =
            serde_json::from_str(r#"{"freq_max": null}"#).expect("legacy null");
        assert_eq!(d.freq_max, Bound::PosInf, "freq_max null means unbounded");
        assert_eq!(
            serde_json::to_value(&d).unwrap()["freq_max"],
            serde_json::json!("inf"),
            "and must write as inf, not derived"
        );
        assert!(d.freq_max.resolve(f64::INFINITY).is_infinite());

        let w: WatershedParams =
            serde_json::from_str(r#"{"merge_thresh": null}"#).expect("legacy null");
        assert_eq!(w.merge_thresh, Bound::Derived, "merge_thresh null means auto");
        // The extractor's sentinel for autosearch is NaN, so that is what the
        // resolve must hand back — not a computed extent.
        assert!(w.merge_thresh.resolve(f64::NAN).is_nan());

        // An explicit number is untouched in both.
        let d: DetectionOpts = serde_json::from_str(r#"{"freq_max": 25.0}"#).unwrap();
        assert_eq!(d.freq_max, Bound::Value(25.0));
    }

    // E9 (parser half): all accepted spellings of `peak_assign`, the
    // rejects, and canonical round-trip. The membership half lives in
    // `mode_peaks::tests`.
    #[test]
    fn peak_assign_parses_every_form() {
        let de = |j: &str| serde_json::from_str::<PeakAssign>(j).unwrap();
        assert_eq!(de("0.3"), PeakAssign::Prob(0.3));
        assert_eq!(de(r#""0.3""#), PeakAssign::Prob(0.3));
        assert_eq!(de(r#""0.3 P""#), PeakAssign::Prob(0.3));
        assert_eq!(de(r#"" 0.3 p ""#), PeakAssign::Prob(0.3));
        assert_eq!(de(r#""1.3 s""#), PeakAssign::Sigma(1.3));
        assert_eq!(de(r#""1.3 SIGMA""#), PeakAssign::Sigma(1.3));
        assert_eq!(de(r#""background""#), PeakAssign::Background(1.5));
        assert_eq!(de(r#""Argmax""#), PeakAssign::Argmax(1.5));
        assert_eq!(de(r#""argmax 2 s""#), PeakAssign::Argmax(2.0));
        assert_eq!(de(r#""BACKGROUND 1.75 SIGMA""#), PeakAssign::Background(1.75));
        for bad in [r#""1.5 p""#, "1.5", r#""0 sigma""#, r#""-1 s""#, r#""nonsense""#, r#""3 q""#, r#""argmax 0 s""#, r#""argmax 2 p""#, r#""argmax 2""#] {
            assert!(
                serde_json::from_str::<PeakAssign>(bad).is_err(),
                "{bad} must be rejected"
            );
        }
    }

    #[test]
    fn peak_assign_canonical_round_trip() {
        for v in [
            PeakAssign::Argmax(1.5),
            PeakAssign::Argmax(2.0),
            PeakAssign::Background(1.5),
            PeakAssign::Prob(0.3),
            PeakAssign::Sigma(1.3),
        ] {
            let s = serde_json::to_string(&v).unwrap();
            assert_eq!(serde_json::from_str::<PeakAssign>(&s).unwrap(), v, "via {s}");
        }
        assert_eq!(serde_json::to_string(&PeakAssign::Prob(0.3)).unwrap(), r#""0.3 p""#);
        assert_eq!(serde_json::to_string(&PeakAssign::Sigma(1.3)).unwrap(), r#""1.3 sigma""#);
        assert_eq!(
            serde_json::to_string(&PeakAssign::default()).unwrap(),
            r#""argmax 1.5 sigma""#,
            "density rules always spell their footprint"
        );
        assert_eq!(PeakAssign::default(), PeakAssign::Argmax(1.5));
    }

    #[test]
    fn peak_assign_thresholds() {
        // Prob(p) → -ln(1-p); Sigma(n) → n²/2; density rules have none.
        let t = PeakAssign::Prob(0.95).threshold().unwrap();
        assert!((t - 20f64.ln()).abs() < 1e-12);
        assert_eq!(PeakAssign::Sigma(1.0).threshold().unwrap(), 0.5);
        // Sigma(n) equals Prob(1 - e^{-n²/2}) as a threshold.
        let n = 1.3f64;
        let p_eq = 1.0 - (-0.5 * n * n).exp();
        let dt = PeakAssign::Sigma(n).threshold().unwrap()
            - PeakAssign::Prob(p_eq).threshold().unwrap();
        assert!(dt.abs() < 1e-12);
        assert!(PeakAssign::Background(1.5).threshold().is_none());
        assert!(PeakAssign::Argmax(1.5).threshold().is_none());
        assert_eq!(PeakAssign::Argmax(1.5).footprint_q_cap(), Some(1.125));
        assert!(PeakAssign::Sigma(1.5).footprint_q_cap().is_none());
    }
}
