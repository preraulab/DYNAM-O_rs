//! Parametric-basis fits for SOPH dimensionality reduction.
//!
//! Two kernels, mirroring `DYNAM-O/toolbox/SOPH_dim_reduction/parametric_basis/`:
//! * [`rot_gauss`] — sum of N rotated 2-D Gaussians + linear background plane
//!   (`xxx*x + yyy*y + zzz`). Used for SO-power histograms.
//! * [`vm_gauss`] — sum of N von-Mises × Gaussian peaks + sinusoidal baseline
//!   (`xxx*sin(x + yyy) + zzz`). Used for SO-phase histograms.
//!
//! Each kernel exposes a `fit_*` driver that runs a bound-constrained
//! Levenberg–Marquardt solver (via the [`levenberg_marquardt`] crate) on a
//! flat parameter vector laid out exactly as MATLAB's Curve-Fitting-Toolbox
//! `fittype` orders its alphabetized coefficients:
//!
//!   `[amp_1..amp_N, fmean_1..fmean_N, fstd_1..fstd_N, ...,
//!     theta_1..theta_N, xxx, yyy, zzz]`
//!
//! The MATLAB-side `fit_rotGauss.m` / `fit_vmGauss.m` `reshape(B0, 1, [])`
//! flattening (column-major over an Nx6 matrix) produces this exact layout.
//!
//! The `levenberg-marquardt` crate implements the classical MINPACK-style
//! unbounded Levenberg–Marquardt. MATLAB's `lsqcurvefit` defaults to a
//! trust-region-reflective bounded solver. We get bounded behavior here
//! by projecting parameters into the feasible box after each step
//! (active-set / projected-gradient style — see `bounded_lm`).

pub mod rot_gauss;
pub mod vm_gauss;
pub mod bounded_lm;

use ndarray::Array2;

/// Goodness-of-fit summary mirroring the MATLAB `gof` struct that
/// `fit(...)` returns. All quantities computed on the residuals
/// `r = z_data - z_model`:
///
/// * `sse`        — `sum(r.^2)`
/// * `dfe`        — `numel(z) - n_params`
/// * `dfm`        — `n_params`  (for completeness; MATLAB reports `df`/`dfm`)
/// * `rmse`       — `sqrt(sse / dfe)`
/// * `rsquare`    — `1 - sse / sst`, `sst = sum((z - mean(z)).^2)`
/// * `adjrsquare` — `1 - (sse / dfe) / (sst / (numel(z) - 1))`
#[derive(Clone, Copy, Debug, Default)]
pub struct Gof {
    pub sse: f64,
    pub rsquare: f64,
    pub adjrsquare: f64,
    pub rmse: f64,
    pub dfe: f64,
    pub dfm: f64,
}

impl Gof {
    /// Compute from data + model.
    pub fn from_data(z_data: &[f64], z_model: &[f64], n_params: usize) -> Self {
        let n = z_data.len() as f64;
        let mean = z_data.iter().sum::<f64>() / n;
        let mut sse = 0.0_f64;
        let mut sst = 0.0_f64;
        for (zd, zm) in z_data.iter().zip(z_model.iter()) {
            let r = zd - zm;
            sse += r * r;
            let dm = zd - mean;
            sst += dm * dm;
        }
        let dfe = n - n_params as f64;
        let rsq = if sst > 0.0 { 1.0 - sse / sst } else { f64::NAN };
        let adj = if sst > 0.0 && dfe > 0.0 {
            1.0 - (sse / dfe) / (sst / (n - 1.0))
        } else { f64::NAN };
        let rmse = if dfe > 0.0 { (sse / dfe).sqrt() } else { f64::NAN };
        Gof { sse, rsquare: rsq, adjrsquare: adj, rmse, dfe, dfm: n_params as f64 }
    }
}

/// Final state of a parametric fit.
pub struct ParamFitOut {
    /// Per-mode parameters, shape `(N, 6)` row-major.
    pub params: Array2<f64>,
    /// `[xxx, yyy, zzz]` background-plane coefficients.
    pub background: [f64; 3],
    /// Model reconstruction on the input grid, shape `(n_y, n_x)` —
    /// matches the orientation of the input `SOPH` argument.
    pub model_soph: Array2<f64>,
    pub gof: Gof,
    pub iters_used: u32,
}

#[derive(Debug)]
pub enum ParamFitError {
    InvalidArgument(String),
    SolverFailed(String),
}

impl std::fmt::Display for ParamFitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParamFitError::InvalidArgument(s) => write!(f, "invalid argument: {}", s),
            ParamFitError::SolverFailed(s)    => write!(f, "LM solver failed: {}", s),
        }
    }
}

impl std::error::Error for ParamFitError {}
