//! von-Mises × Gaussian mixture model + LM fit driver.
//!
//! Mirrors MATLAB
//! `DYNAM-O/toolbox/SOPH_dim_reduction/parametric_basis/basis_functions/{vmGauss,normalized_vmGauss}.m`:
//!
//! ```text
//!   z(x, y) = xxx * sin(x + yyy) + zzz
//!             + sum_{n=1..N} vmGauss(x, y, amp_n, fmean_n, fstd_n,
//!                                          phasepref_n, recikappa_n, theta_n)
//! ```
//!
//! where
//!
//! ```text
//!   vmGauss(x, y, A, ym, ys, xm, xs, t)
//!     = A * exp(-0.5 * ((y-ym) / ys)^2)            -- ys is a true σ (Hz)
//!         * exp( κ * (cos(x - xm + (y-ym)*sin(t)) - 1) )
//!     κ = 1 / xs^2
//! ```
//!
//! Both widths are genuine standard deviations, but they get there by
//! different routes, so they are NOT symmetric under rescaling:
//!
//! * `ys` (`fstd`) is a σ because of the explicit `-0.5` above. The
//!   squared denominator alone would make the width `sqrt(2)·σ`. The one
//!   half makes it σ.
//! * `xs` (`recikappa`) is ALREADY a σ with no half needed, because the
//!   von Mises factor is its own small-angle Gaussian:
//!   `exp(κ(cos Δ - 1)) → exp(-Δ²/(2·xs²))`. It carries the half
//!   intrinsically. Never rescale it alongside `ys`.
//!
//! The reported `FreqStd`/`SOphaseStd` columns, the mode-volume integral,
//! and the peak-containment geometry all consume these directly as σ.
//! There is no variance⇄σ transform at any boundary.
//!
//! Parameter layout (flat vector, length `6N + 3`):
//!
//! ```text
//!   [ amp_1..amp_N,
//!     fmean_1..fmean_N,
//!     fstd_1..fstd_N,
//!     phasepref_1..phasepref_N,
//!     recikappa_1..recikappa_N,
//!     theta_1..theta_N,
//!     xxx, yyy, zzz ]
//! ```
//!
//! ## Row normalization
//!
//! MATLAB `fit_vmGauss.m` passes `'problem', true` to
//! `normalized_vmGauss`, which row-normalizes the assembled surface
//! (each frequency row divided by its own sum so every row sums to 1).
//! Production phase paths mirror this: they pass `unit_row=true` to
//! [`eval_model`] and phase paramfit CSVs are written with
//! `unit_row: 1.0`. An un-normalized fit (`'problem', false`) washes the
//! modes out, so it is not used.
//!
//! The `unit_row=false` branch is kept for the parity kernel tests. The
//! numerical Jacobian is used in both branches. Row coupling makes the
//! analytic Jacobian expensive to write and the LM is solver-bound
//! anyway, so it isn't worth the added complexity.

use ndarray::{Array2, ArrayView2};
use nalgebra::DMatrix;

use super::bounded_lm::{run_bounded_lm, ResidualsAndJacobian};
use super::{Gof, ParamFitError, ParamFitOut};

/// Parameters per mode: amp, fmean, fstd, phasepref, recikappa, theta.
const PARAMS_PER_MODE: usize = 6;
/// Background parameters: xxx, yyy, zzz.
const N_BACKGROUND_PARAMS: usize = 3;

/// Evaluate the model surface on a grid. `params` is the flat vector laid
/// out as in the module docs (length `6 * n_modes + 3`), `x_grid` is SO
/// phase and `y_grid` is frequency. Returns `z[[iy, ix]]`, one row per
/// frequency. With `unit_row`, each row is then divided by its own sum
/// (a row summing to exactly zero is left as it is).
pub fn eval_model(
    params: &[f64],
    n_modes: usize,
    x_grid: &[f64],
    y_grid: &[f64],
    unit_row: bool,
) -> Array2<f64> {
    let nx = x_grid.len();
    let ny = y_grid.len();
    let mut z = Array2::<f64>::zeros((ny, nx));
    let background_start = PARAMS_PER_MODE * n_modes;
    let xxx = params[background_start];
    let yyy = params[background_start + 1];
    let zzz = params[background_start + 2];
    for iy in 0..ny {
        let y = y_grid[iy];
        for ix in 0..nx {
            let x = x_grid[ix];
            let mut value = xxx * (x + yyy).sin() + zzz;
            for n in 0..n_modes {
                let amp = params[n];
                let fmean = params[n_modes + n];
                let fstd = params[2 * n_modes + n];
                let phasepref = params[3 * n_modes + n];
                let recikappa = params[4 * n_modes + n];
                let theta = params[5 * n_modes + n];
                let dy = y - fmean;
                let kappa = 1.0 / (recikappa * recikappa);
                let arg_cos = x - phasepref + dy * theta.sin();
                value += amp * (-0.5 * (dy * dy) / (fstd * fstd)).exp()
                    * (kappa * (arg_cos.cos() - 1.0)).exp();
            }
            z[[iy, ix]] = value;
        }
    }
    if unit_row {
        for iy in 0..ny {
            let row_sum: f64 = (0..nx).map(|ix| z[[iy, ix]]).sum();
            // A row that sums to exactly zero cannot be normalized and is
            // left as it is.
            if row_sum.abs() > 0.0 {
                let inv = 1.0 / row_sum;
                for ix in 0..nx {
                    z[[iy, ix]] *= inv;
                }
            }
        }
    }
    z
}

/// The fit problem handed to the bounded LM solver.
struct VmGaussProblem<'a> {
    n_modes: usize,
    x_grid: &'a [f64],
    y_grid: &'a [f64],
    z_data: &'a [f64],
    /// False where the data is not finite. The residual is zero there
    /// (MATLAB's `prepareSurfaceData` drops those cells).
    finite_mask: &'a [bool],
    unit_row: bool,
}

impl<'a> ResidualsAndJacobian for VmGaussProblem<'a> {
    fn residuals(&self, params: &[f64], out: &mut [f64]) {
        let model = eval_model(params, self.n_modes, self.x_grid, self.y_grid, self.unit_row);
        for (k, model_value) in model.iter().enumerate() {
            if self.finite_mask[k] {
                out[k] = *model_value - self.z_data[k];
            } else {
                out[k] = 0.0;
            }
        }
    }

    fn jacobian(&self, params: &[f64], out: &mut DMatrix<f64>) {
        // Numerical Jacobian by central differences (the module docs say
        // why it is not analytic).
        let n_p = self.n_params();
        let n_r = self.n_residuals();
        let mut r_plus  = vec![0.0_f64; n_r];
        let mut r_minus = vec![0.0_f64; n_r];
        let mut p = params.to_vec();
        for j in 0..n_p {
            let v = params[j];
            // Step: 1e-6 of the parameter's magnitude, and never below 1e-6.
            let h = 1e-6_f64.max(v.abs() * 1e-6);
            p[j] = v + h;
            self.residuals(&p, &mut r_plus);
            p[j] = v - h;
            self.residuals(&p, &mut r_minus);
            p[j] = v;
            let inv_2h = 0.5 / h;
            for i in 0..n_r {
                out[(i, j)] = (r_plus[i] - r_minus[i]) * inv_2h;
            }
        }
    }

    fn n_params(&self) -> usize {
        PARAMS_PER_MODE * self.n_modes + N_BACKGROUND_PARAMS
    }

    fn n_residuals(&self) -> usize {
        self.x_grid.len() * self.y_grid.len()
    }
}

/// Fit the model to a SOPH by bounded Levenberg-Marquardt.
///
/// `soph` is `(y_grid.len(), x_grid.len())`. `initial`, `lower` and
/// `upper` are `(n_modes, 6)` with columns `[amp, fmean, fstd, phasepref,
/// recikappa, theta]`. `bg_*` are the background terms `[xxx, yyy, zzz]`.
/// Non-finite cells of `soph` are left out of the fit and of the
/// goodness-of-fit. Returns the fitted parameters in the same layout,
/// the model surface, the goodness-of-fit, and the iteration count.
pub fn fit_vmgauss(
    soph: ArrayView2<f64>,
    x_grid: &[f64],
    y_grid: &[f64],
    initial: ArrayView2<f64>,
    lower: ArrayView2<f64>,
    upper: ArrayView2<f64>,
    bg_initial: [f64; 3],
    bg_lower: [f64; 3],
    bg_upper: [f64; 3],
    max_iters: u32,
    unit_row: bool,
) -> Result<ParamFitOut, ParamFitError> {
    let (ny, nx) = soph.dim();
    if y_grid.len() != ny || x_grid.len() != nx {
        return Err(ParamFitError::InvalidArgument(format!(
            "soph dim ({},{}) doesn't match grids ({},{})",
            ny, nx, y_grid.len(), x_grid.len()
        )));
    }
    if initial.ncols() != PARAMS_PER_MODE
        || lower.ncols() != PARAMS_PER_MODE
        || upper.ncols() != PARAMS_PER_MODE
    {
        return Err(ParamFitError::InvalidArgument(
            "initial/lower/upper must each have 6 columns".into()
        ));
    }
    let n_modes = initial.nrows();
    if lower.nrows() != n_modes || upper.nrows() != n_modes {
        return Err(ParamFitError::InvalidArgument(
            "initial/lower/upper row counts must match".into()
        ));
    }
    let n_params = PARAMS_PER_MODE * n_modes + N_BACKGROUND_PARAMS;
    // The background terms sit after all the per-mode parameters.
    let background_start = PARAMS_PER_MODE * n_modes;

    let mut p0 = vec![0.0_f64; n_params];
    let mut p_lo = vec![0.0_f64; n_params];
    let mut p_hi = vec![0.0_f64; n_params];
    for col in 0..PARAMS_PER_MODE {
        for m in 0..n_modes {
            let i = col * n_modes + m;
            p0[i] = initial[[m, col]];
            p_lo[i] = lower[[m, col]];
            p_hi[i] = upper[[m, col]];
        }
    }
    for k in 0..N_BACKGROUND_PARAMS {
        p0[background_start + k] = bg_initial[k];
        p_lo[background_start + k] = bg_lower[k];
        p_hi[background_start + k] = bg_upper[k];
    }

    let z_data: Vec<f64> = soph.iter().copied().collect();
    let finite_mask: Vec<bool> = z_data.iter().map(|v| v.is_finite()).collect();
    let problem = VmGaussProblem {
        n_modes, x_grid, y_grid,
        z_data: &z_data,
        finite_mask: &finite_mask,
        unit_row,
    };

    let report = run_bounded_lm(&problem, &p0, &p_lo, &p_hi, max_iters)
        .map_err(ParamFitError::SolverFailed)?;

    let mut params_out = Array2::<f64>::zeros((n_modes, PARAMS_PER_MODE));
    for col in 0..PARAMS_PER_MODE {
        for m in 0..n_modes {
            params_out[[m, col]] = report.params[col * n_modes + m];
        }
    }
    let bg = [
        report.params[background_start],
        report.params[background_start + 1],
        report.params[background_start + 2],
    ];
    let model_soph = eval_model(&report.params, n_modes, x_grid, y_grid, unit_row);
    // Only count finite cells in GoF (matches MATLAB's prepareSurfaceData drop).
    let mut z_keep_data: Vec<f64> = Vec::with_capacity(z_data.len());
    let mut z_keep_model: Vec<f64> = Vec::with_capacity(z_data.len());
    for (k, model_value) in model_soph.iter().enumerate() {
        if finite_mask[k] {
            z_keep_data.push(z_data[k]);
            z_keep_model.push(*model_value);
        }
    }
    let gof = Gof::from_data(&z_keep_data, &z_keep_model, n_params);

    Ok(ParamFitOut {
        params: params_out,
        background: bg,
        model_soph,
        gof,
        iters_used: report.iters_used,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn eval_matches_matlab_vmgauss_at_known_point() {
        // amp=1, fmean=12, fstd=2.5, phasepref=π/3, recikappa=1.0, theta=0.1.
        let amp: f64 = 1.0;
        let fmean: f64 = 12.0;
        let fstd: f64 = 2.5;
        let phasepref: f64 = std::f64::consts::PI / 3.0;
        let recikappa: f64 = 1.0;
        let theta: f64 = 0.1;
        let x: f64 = 0.5;
        let y: f64 = 11.0;
        let dy = y - fmean;
        let kappa = 1.0 / (recikappa * recikappa);
        // The ½ is on the Gaussian factor only. The von Mises factor is
        // already its own half-form and is written here unchanged.
        let expected = amp * (-0.5 * (dy * dy) / (fstd * fstd)).exp()
            * (kappa * ((x - phasepref + dy * theta.sin()).cos() - 1.0)).exp();
        let p = vec![amp, fmean, fstd, phasepref, recikappa, theta, 0.0, 0.0, 0.0];
        let z = eval_model(&p, 1, &[x], &[y], false);
        assert!((z[[0, 0]] - expected).abs() < 1e-14);
    }

    #[test]
    fn frequency_width_is_standard_deviation() {
        let amp = 2.0;
        let fmean = 10.0;
        let fstd = 2.5;
        let phasepref = 0.0;
        let recikappa = 1.0;
        let theta = 0.0;
        let p = [
            amp, fmean, fstd, phasepref, recikappa, theta,
            0.0, 0.0, 0.0,
        ];

        let at_center = eval_model(&p, 1, &[phasepref], &[fmean], false)[[0, 0]];
        let at_one_std =
            eval_model(&p, 1, &[phasepref], &[fmean + fstd], false)[[0, 0]];

        assert!((at_one_std / at_center - (-0.5_f64).exp()).abs() < 1e-12);
    }

    #[test]
    fn recovers_known_single_mode_phase() {
        use std::f64::consts::PI;

        let nx = 40usize;
        let ny = 40usize;
        let xg: Vec<f64> = (0..nx)
            .map(|i| -PI + 2.0 * PI * i as f64 / (nx - 1) as f64)
            .collect();
        let yg: Vec<f64> = (0..ny)
            .map(|j| 0.5 + 30.0 * j as f64 / (ny - 1) as f64)
            .collect();
        let true_p = [0.08_f64, 11.0, 2.5, std::f64::consts::PI/3.0, 1.0, 0.0,
                       0.0, 0.0, 0.005];
        let z_clean = eval_model(&true_p, 1, &xg, &yg, false);
        let mut z = z_clean.clone();
        for iy in 0..ny {
            for ix in 0..nx {
                z[[iy, ix]] += ((iy as f64 * 13.0 + ix as f64 * 7.0).cos() * 0.001).abs();
            }
        }

        let initial = Array2::from_shape_vec((1, 6),
            vec![0.05, 10.0, 3.0, 1.0, 1.5, 0.0]).unwrap();
        let lower = Array2::from_shape_vec((1, 6),
            vec![0.0, 1.0, 1.0, -std::f64::consts::PI, 0.5, -std::f64::consts::PI/3.0]).unwrap();
        let upper = Array2::from_shape_vec((1, 6),
            vec![1.0, 30.0, 15.0, std::f64::consts::PI, 5.0, std::f64::consts::PI/3.0]).unwrap();

        let out = fit_vmgauss(
            z.view(), &xg, &yg,
            initial.view(), lower.view(), upper.view(),
            [0.0, 0.0, 0.005], [-1.0, -std::f64::consts::PI, 0.0], [1.0, std::f64::consts::PI, 1.0],
            500, false,
        ).unwrap();

        assert!(out.gof.rsquare > 0.95,
            "Phase recovery R² too low: {} (sse={}, dfe={})",
            out.gof.rsquare, out.gof.sse, out.gof.dfe);
        assert!(((out.params[[0, 2]] - true_p[2]) / true_p[2]).abs() < 0.12,
            "Recovered FreqStd {} differs from planted σ {}",
            out.params[[0, 2]], true_p[2]);
    }

    #[test]
    fn phase_center_bounds_allow_crossing_pi_seam() {
        use std::f64::consts::PI;

        let nx = 81usize;
        let ny = 65usize;
        let xg: Vec<f64> = (0..nx)
            .map(|i| -PI + 2.0 * PI * i as f64 / (nx - 1) as f64)
            .collect();
        let yg: Vec<f64> = (0..ny)
            .map(|i| 2.0 + 16.0 * i as f64 / (ny - 1) as f64)
            .collect();
        let planted = [
            0.07, 10.5, 1.4, -PI + 0.12, 0.9, 0.42,
            0.012, 0.35, 0.003,
        ];
        let soph = eval_model(&planted, 1, &xg, &yg, true);
        let initial = Array2::from_shape_vec(
            (1, 6),
            vec![0.06, 10.2, 1.2, PI - 0.04, 1.0, 0.30],
        ).unwrap();
        let historical_fstd_lower = 1.0 / 2.0_f64.sqrt();
        let historical_fstd_upper = 7.5_f64.sqrt();
        for (label, phase_lower, phase_upper) in [
            ("two-period", -2.0 * PI, 2.0 * PI),
            ("unbounded", f64::NEG_INFINITY, f64::INFINITY),
        ] {
            let lower = Array2::from_shape_vec(
                (1, 6),
                vec![0.001, 2.0, historical_fstd_lower, phase_lower, PI / 5.0, -PI / 3.0],
            ).unwrap();
            let upper = Array2::from_shape_vec(
                (1, 6),
                vec![1.0, 18.0, historical_fstd_upper, phase_upper, 2.0 * PI, PI / 3.0],
            ).unwrap();

            let out = fit_vmgauss(
                soph.view(), &xg, &yg,
                initial.view(), lower.view(), upper.view(),
                [0.012, 0.35, 0.003], [-1.0, -PI, 0.0], [1.0, PI, 1.0],
                500, true,
            ).unwrap();
            let raw_phase = out.params[[0, 3]];
            let phase_delta = raw_phase - planted[3];
            let circular_error = phase_delta.sin().atan2(phase_delta.cos()).abs();

            assert!(raw_phase > PI,
                "{label}: expected the raw phase center to cross +π, got {raw_phase}");
            assert!(circular_error < 1e-4,
                "{label}: recovered phase {raw_phase} is not circularly close to {}",
                planted[3]);
            assert!(out.gof.adjrsquare > 0.9999,
                "{label}: seam-crossing adjusted R² too low: {}", out.gof.adjrsquare);
        }
    }
}
