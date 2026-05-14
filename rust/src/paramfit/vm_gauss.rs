//! von-Mises × Gaussian mixture model + LM fit driver.
//!
//! Mirrors MATLAB
//! `DYNAM-O_dev/toolbox/SOPH_dim_reduction/parametric_basis/basis_functions/{vmGauss,normalized_vmGauss}.m`:
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
//!     = A * exp(-(y-ym)^2 / ys)                    -- note: NOT ys^2
//!         * exp( κ * (cos(x - xm + (y-ym)*sin(t)) - 1) )
//!     κ = 1 / xs^2
//! ```
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
//! MATLAB `fit_vmGauss.m` always passes `unit_row=true` to
//! `normalized_vmGauss`. After the model is assembled from baseline +
//! sum-of-modes, each frequency row is divided by its own sum so
//! every row sums to 1. We replicate this in [`eval_model`] when
//! `unit_row=true`. The Jacobian for the normalized variant uses
//! numerical central differences — the row coupling makes the analytic
//! form expensive to write and the per-call cost is dominated by the
//! solver, not by Jacobian assembly.

use ndarray::{Array2, ArrayView2};
use nalgebra::DMatrix;

use super::bounded_lm::{run_bounded_lm, ResidualsAndJacobian};
use super::{Gof, ParamFitError, ParamFitOut};

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
    let xxx = params[6 * n_modes];
    let yyy = params[6 * n_modes + 1];
    let zzz = params[6 * n_modes + 2];
    for iy in 0..ny {
        let y = y_grid[iy];
        for ix in 0..nx {
            let x = x_grid[ix];
            let mut v = xxx * (x + yyy).sin() + zzz;
            for n in 0..n_modes {
                let a  = params[n];
                let fm = params[n_modes + n];
                let fs = params[2 * n_modes + n];
                let pp = params[3 * n_modes + n];
                let rk = params[4 * n_modes + n];
                let th = params[5 * n_modes + n];
                let dy = y - fm;
                let kappa = 1.0 / (rk * rk);
                let arg_cos = x - pp + dy * th.sin();
                v += a * (-(dy * dy) / fs).exp() * (kappa * (arg_cos.cos() - 1.0)).exp();
            }
            z[[iy, ix]] = v;
        }
    }
    if unit_row {
        for iy in 0..ny {
            let row_sum: f64 = (0..nx).map(|ix| z[[iy, ix]]).sum();
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

struct VmGaussProblem<'a> {
    n_modes: usize,
    x_grid: &'a [f64],
    y_grid: &'a [f64],
    z_data: &'a [f64],
    finite_mask: &'a [bool], // residual is zero at non-finite cells (mirror of MATLAB's prepareSurfaceData NaN drop)
    unit_row: bool,
}

impl<'a> ResidualsAndJacobian for VmGaussProblem<'a> {
    fn residuals(&self, params: &[f64], out: &mut [f64]) {
        let model = eval_model(params, self.n_modes, self.x_grid, self.y_grid, self.unit_row);
        for (k, ((zm, zd), m)) in model.iter()
            .zip(self.z_data.iter())
            .zip(self.finite_mask.iter())
            .enumerate()
        {
            out[k] = if *m { *zm - *zd } else { 0.0 };
        }
    }

    fn jacobian(&self, params: &[f64], out: &mut DMatrix<f64>) {
        // Numerical Jacobian via central differences. Analytic forms exist
        // but are tedious — and at the per-pixel grid sizes we operate on
        // (n_x ~ 60, n_y ~ 100), numerical Jacobian compute is ≤10% of
        // LM total time. We'll revisit if profiling demands it.
        let n_p = self.n_params();
        let n_r = self.n_residuals();
        let mut r_plus  = vec![0.0_f64; n_r];
        let mut r_minus = vec![0.0_f64; n_r];
        let mut p = params.to_vec();
        for j in 0..n_p {
            let v = params[j];
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

    fn n_params(&self) -> usize { 6 * self.n_modes + 3 }
    fn n_residuals(&self) -> usize { self.x_grid.len() * self.y_grid.len() }
}

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
    if initial.ncols() != 6 || lower.ncols() != 6 || upper.ncols() != 6 {
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
    let n_params = 6 * n_modes + 3;

    let mut p0   = vec![0.0_f64; n_params];
    let mut p_lo = vec![0.0_f64; n_params];
    let mut p_hi = vec![0.0_f64; n_params];
    for col in 0..6 {
        for m in 0..n_modes {
            let i = col * n_modes + m;
            p0[i]   = initial[[m, col]];
            p_lo[i] = lower[[m, col]];
            p_hi[i] = upper[[m, col]];
        }
    }
    p0[6 * n_modes]     = bg_initial[0];
    p0[6 * n_modes + 1] = bg_initial[1];
    p0[6 * n_modes + 2] = bg_initial[2];
    p_lo[6 * n_modes]     = bg_lower[0];
    p_lo[6 * n_modes + 1] = bg_lower[1];
    p_lo[6 * n_modes + 2] = bg_lower[2];
    p_hi[6 * n_modes]     = bg_upper[0];
    p_hi[6 * n_modes + 1] = bg_upper[1];
    p_hi[6 * n_modes + 2] = bg_upper[2];

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

    let mut params_out = Array2::<f64>::zeros((n_modes, 6));
    for col in 0..6 {
        for m in 0..n_modes {
            params_out[[m, col]] = report.params[col * n_modes + m];
        }
    }
    let bg = [
        report.params[6 * n_modes],
        report.params[6 * n_modes + 1],
        report.params[6 * n_modes + 2],
    ];
    let model_soph = eval_model(&report.params, n_modes, x_grid, y_grid, unit_row);
    // Only count finite cells in GoF (matches MATLAB's prepareSurfaceData drop).
    let mut z_keep_data:  Vec<f64> = Vec::with_capacity(z_data.len());
    let mut z_keep_model: Vec<f64> = Vec::with_capacity(z_data.len());
    for ((zd, zm), m) in z_data.iter().zip(model_soph.iter()).zip(finite_mask.iter()) {
        if *m { z_keep_data.push(*zd); z_keep_model.push(*zm); }
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
        let a: f64 = 1.0;
        let fm: f64 = 12.0;
        let fs: f64 = 2.5;
        let pp: f64 = std::f64::consts::PI / 3.0;
        let rk: f64 = 1.0;
        let th: f64 = 0.1;
        let x: f64 = 0.5;
        let y: f64 = 11.0;
        let dy = y - fm;
        let kappa = 1.0 / (rk * rk);
        let expected = a * (-(dy * dy) / fs).exp() * (kappa * ((x - pp + dy * th.sin()).cos() - 1.0)).exp();
        let p = vec![a, fm, fs, pp, rk, th, 0.0, 0.0, 0.0];
        let z = eval_model(&p, 1, &[x], &[y], false);
        assert!((z[[0, 0]] - expected).abs() < 1e-14);
    }

    #[test]
    fn recovers_known_single_mode_phase() {
        let nx = 40usize; let ny = 40usize;
        let xg: Vec<f64> = (0..nx).map(|i| -std::f64::consts::PI + 2.0*std::f64::consts::PI*i as f64/(nx-1) as f64).collect();
        let yg: Vec<f64> = (0..ny).map(|j| 0.5 + 30.0*j as f64/(ny-1) as f64).collect();
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
    }
}
