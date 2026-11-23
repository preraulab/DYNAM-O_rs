//! End-to-end DYNAM-O runs: EEG + staging in, every computed output out.
//!
//! The DYNAM-O desktop app, its command-line tool and the Python wrapper
//! all run a recording through [`run_dynamo`], so their numbers are the
//! same by construction. Bridges the atomic stages (watershed, merge,
//! trim, refine, mask, baseline, artifacts, SO-power/phase, histograms,
//! fits) into two pipelines:
//!
//! - [`run_extract_two_pass`]: pass-1 baseline + extract on a 1 s
//!   spectrogram, pass-2 multitaper at a 2 s window, mask via the pass-1
//!   labels, extract, refine.
//! - [`run_dynamo`]: EEG + staging, spectrograms, artifacts, baseline,
//!   two-pass extraction, refinement, SO-power/phase, SOPHs, fits.
//!
//! [`settings::DynamoSettings`] holds every option, and
//! [`settings_schema`] the settings-file semantics version and migrations.

pub mod orchestrator;
pub mod rebuild;
pub mod resample;
pub mod settings;
pub mod settings_schema;
pub mod soph;
mod soph_build;
pub use soph_build::take_last_splinefit_error;

pub use rebuild::{rebuild_from_cache, RebuildInputs};
pub use orchestrator::{
    detect_artifacts_default, run_dynamo, run_dynamo_keeping_spectrogram,
    run_dynamo_with_progress, run_extract_two_pass,
    CancelFn, DynamoError, DynamoInput, DynamoOutputs, PassMtmSpec, PassSpectrogram, ProgressFn,
    StopAfter, SummarySpecCacheData, TwoPassOutputs, scored_window};
pub use settings::{
    BaselineOpts, DetectionOpts, DynamoSettings, FigureFormat, HistogramOpts, MtmOpts,
    OutputOpts, PeakAssign, RecomputeSet, SoPowerOpts, WatershedParams,
    PEAK_ASSIGN_PHASE_FALLBACK,
};
