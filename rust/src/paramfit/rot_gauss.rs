//! Rotated 2-D Gaussian mixture model + LM fit driver.
//!
//! Mirrors MATLAB
//! `DYNAM-O/toolbox/SOPH_dim_reduction/parametric_basis/basis_functions/rotGauss.m`
//! and the linear background-plane sum from `fit_rotGauss.m:105`:
//!
//! ```text
//!   z(x, y) = sum_{n=1..N} rotGauss(x, y, amp_n, fmean_n, fstd_n,
//!                                          pmean_n, pstd_n, theta_n)
//!             + xxx*x + yyy*y + zzz
//! ```
//!
//! where the single-mode kernel is
//!
//! ```text
//!   rotGauss(x, y, A, ym, ys, xm, xs, t)
//!     = A * exp( -(((y-ym)*cos(t) + (x-xm)*sin(t)) / ys)^2
//!                 -((-(y-ym)*sin(t) + (x-xm)*cos(t)) / xs)^2 )
//! ```
//!
//! Parameter layout (flat vector, length `6N + 3`):
//!
//! ```text
//!   [ amp_1..amp_N,
//!     fmean_1..fmean_N,
//!     fstd_1..fstd_N,
//!     pmean_1..pmean_N,
//!     pstd_1..pstd_N,
//!     theta_1..theta_N,
//!     xxx, yyy, zzz ]
//! ```
//!
//! This is the column-major flattening of MATLAB's
//! `reshape(B0, 1, numel(B0))` where `B0` is the canonical Nx6 matrix
//! `[amp fmean fstd pmean pstd theta]`. The CFTool alphabetizes the
//! coefficient names so the order is `amp, fmean, fstd, pmean, pstd,
//! theta, xxx, yyy, zzz` — same order.

use ndarray::{Array2, ArrayView2};
use nalgebra::DMatrix;

use super::bounded_lm::{run_bounded_lm, ResidualsAndJacobian};
use super::{Gof, ParamFitError, ParamFitOut};

/// Evaluate the rotated-Gaussian-mixture model on a grid.
///
/// `x_grid`, `y_grid` are the bin centers; output `(n_y, n_x)` row-major
/// matches the MATLAB SOPH orientation `(n_freqs, n_features)`.
pub fn eval_model(
    params: &[f64],
    n_modes: usize,
    x_grid: &[f64],
    y_grid: &[f64],
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
            let mut v = xxx * x + yyy * y + zzz;
            for n in 0..n_modes {
                let a  = params[n];
                let fm = params[n_modes + n];
                let fs = params[2 * n_modes + n];
                let pm = params[3 * n_modes + n];
                let ps = params[4 * n_modes + n];
                let th = params[5 * n_modes + n];
                let dy = y - fm;
                let dx = x - pm;
                let (s, c) = th.sin_cos();
                let u = (dy * c + dx * s) / fs;
                let w = (-dy * s + dx * c) / ps;
                v += a * (-(u * u) - w * w).exp();
            }
            z[[iy, ix]] = v;
        }
    }
    z
}

struct RotGaussProblem<'a> {
    n_modes: usize,
    x_grid: &'a [f64],
    y_grid: &'a [f64],
    z_data: &'a [f64], // length nx * ny, row-major (n_y, n_x) so iy is the outer loop
    finite_mask: &'a [bool],
}

impl<'a> ResidualsAndJacobian for RotGaussProblem<'a> {
    fn residuals(&self, params: &[f64], out: &mut [f64]) {
        let model = eval_model(params, self.n_modes, self.x_grid, self.y_grid);
        for (k, ((zm, zd), m)) in model.iter()
            .zip(self.z_data.iter())
            .zip(self.finite_mask.iter())
            .enumerate()
        {
            out[k] = if *m { *zm - *zd } else { 0.0 };
        }
    }

    fn jacobian(&self, params: &[f64], out: &mut DMatrix<f64>) {
        // Analytic Jacobian. nalgebra DMatrix is column-major, so each
        // column j occupies `out.as_mut_slice()[j*n_r..(j+1)*n_r]`. We
        // index that slice directly to avoid per-element bounds-check
        // overhead and stride computation in Index/IndexMut. To get
        // contiguous writes within each column, we iterate columns in
        // the outer loop and pixels in the inner loop.
        let n = self.n_modes;
        let nx = self.x_grid.len();
        let ny = self.y_grid.len();
        let n_r = nx * ny;
        let n_p = ResidualsAndJacobian::n_params(self);
        debug_assert_eq!(out.nrows(), n_r);
        debug_assert_eq!(out.ncols(), n_p);

        let buf: &mut [f64] = out.as_mut_slice();
        // Zero entire Jacobian first (one big memset → fast).
        for v in buf.iter_mut() { *v = 0.0; }

        // Background plane columns (xxx, yyy, zzz). Each is contiguous.
        let xxx_idx = 6 * n;
        let yyy_idx = 6 * n + 1;
        let zzz_idx = 6 * n + 2;
        {
            let (col_x, col_y, col_z) = unsafe {
                let p = buf.as_mut_ptr();
                (
                    std::slice::from_raw_parts_mut(p.add(xxx_idx * n_r), n_r),
                    std::slice::from_raw_parts_mut(p.add(yyy_idx * n_r), n_r),
                    std::slice::from_raw_parts_mut(p.add(zzz_idx * n_r), n_r),
                )
            };
            for iy in 0..ny {
                let y = self.y_grid[iy];
                for ix in 0..nx {
                    let row = iy * nx + ix;
                    if !self.finite_mask[row] { continue; }
                    col_x[row] = self.x_grid[ix];
                    col_y[row] = y;
                    col_z[row] = 1.0;
                }
            }
        }

        // Per-mode kernel-derivative columns. Get 6 separate &mut slices
        // (one per derivative column) so each write is a direct offset
        // into a contiguous f64 buffer.
        for m in 0..n {
            let a  = params[m];
            let fm = params[n + m];
            let fs = params[2 * n + m];
            let pm = params[3 * n + m];
            let ps = params[4 * n + m];
            let th = params[5 * n + m];
            let (sin_t, cos_t) = th.sin_cos();
            let inv_fs = 1.0 / fs;
            let inv_ps = 1.0 / ps;
            let two_inv_fs = 2.0 * inv_fs;
            let two_inv_ps = 2.0 * inv_ps;
            let fs_over_ps_minus = fs * inv_ps - ps * inv_fs;

            let (col_a, col_fm, col_fs, col_pm, col_ps, col_t) = unsafe {
                let p = buf.as_mut_ptr();
                (
                    std::slice::from_raw_parts_mut(p.add( m            * n_r), n_r),
                    std::slice::from_raw_parts_mut(p.add((n + m)        * n_r), n_r),
                    std::slice::from_raw_parts_mut(p.add((2 * n + m)    * n_r), n_r),
                    std::slice::from_raw_parts_mut(p.add((3 * n + m)    * n_r), n_r),
                    std::slice::from_raw_parts_mut(p.add((4 * n + m)    * n_r), n_r),
                    std::slice::from_raw_parts_mut(p.add((5 * n + m)    * n_r), n_r),
                )
            };

            for iy in 0..ny {
                let y = self.y_grid[iy];
                let dy = y - fm;
                let dy_c = dy * cos_t;
                let dy_s = dy * sin_t;
                let row_base = iy * nx;
                for ix in 0..nx {
                    let row = row_base + ix;
                    if !self.finite_mask[row] { continue; }
                    let dx = self.x_grid[ix] - pm;
                    let u = (dy_c + dx * sin_t) * inv_fs;
                    let w = (-dy_s + dx * cos_t) * inv_ps;
                    let exp_arg = -(u * u) - w * w;
                    let kern = exp_arg.exp();
                    let g = a * kern;
                    col_a[row]  = if a != 0.0 { g / a } else { kern };
                    col_fm[row] = g * (two_inv_fs * u * cos_t - two_inv_ps * w * sin_t);
                    col_fs[row] = g * two_inv_fs * u * u;
                    col_pm[row] = g * (two_inv_fs * u * sin_t + two_inv_ps * w * cos_t);
                    col_ps[row] = g * two_inv_ps * w * w;
                    col_t[row]  = 2.0 * g * u * w * fs_over_ps_minus;
                }
            }
        }
    }

    fn n_params(&self) -> usize { 6 * self.n_modes + 3 }
    fn n_residuals(&self) -> usize { self.x_grid.len() * self.y_grid.len() }
}

/// Fit a rotated-Gaussian mixture + linear plane to a SOPH histogram.
///
/// * `soph` — shape `(n_y, n_x)` = `(n_freqs, n_features)`, row-major.
/// * `x_grid` — feature/power bin centers, length `n_x`.
/// * `y_grid` — frequency bin centers,    length `n_y`.
/// * `initial` — Nx6 matrix of seed params `[amp, fmean, fstd, pmean, pstd, theta]`.
/// * `lower`, `upper` — bound matrices, same shape as `initial`.
/// * `bg_lower`, `bg_upper` — bounds on `[xxx, yyy, zzz]`.
/// * `bg_initial` — seed for `[xxx, yyy, zzz]`.
/// * `max_iters` — LM evaluation cap.
pub fn fit_rotgauss(
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

    // Flatten to MATLAB alphabetical layout: all amps, then all fmeans, ...
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

    // The model evaluator produces (n_y, n_x). soph is stored as (n_y, n_x)
    // row-major already (the caller hands us this orientation).
    let z_data: Vec<f64> = soph.iter().copied().collect();
    let finite_mask: Vec<bool> = z_data.iter().map(|v| v.is_finite()).collect();
    let problem = RotGaussProblem {
        n_modes, x_grid, y_grid,
        z_data: &z_data,
        finite_mask: &finite_mask,
    };

    let report = run_bounded_lm(&problem, &p0, &p_lo, &p_hi, max_iters)
        .map_err(ParamFitError::SolverFailed)?;

    // Pack output.
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
    let model_soph = eval_model(&report.params, n_modes, x_grid, y_grid);
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

    fn make_grid(nx: usize, ny: usize) -> (Vec<f64>, Vec<f64>) {
        let x: Vec<f64> = (0..nx).map(|i| -50.0 + 100.0 * i as f64 / (nx - 1) as f64).collect();
        let y: Vec<f64> = (0..ny).map(|j| 0.5 + 30.0 * j as f64 / (ny - 1) as f64).collect();
        (x, y)
    }

    #[test]
    fn eval_matches_matlab_rotgauss_at_known_point() {
        // Single-mode reference: amp=1, fmean=12, fstd=2.5, pmean=10,
        // pstd=15, theta=pi/8. Hand-evaluate at (x=15, y=14).
        let a = 1.0;
        let fm = 12.0;
        let fs = 2.5;
        let pm = 10.0;
        let ps = 15.0;
        let th: f64 = std::f64::consts::PI / 8.0;
        let x = 15.0;
        let y = 14.0;
        let dy = y - fm;
        let dx = x - pm;
        let (s, c) = th.sin_cos();
        let u = (dy * c + dx * s) / fs;
        let w = (-dy * s + dx * c) / ps;
        let expected = a * (-(u * u) - w * w).exp();

        // Single-mode params + zero background.
        let p = vec![a, fm, fs, pm, ps, th, 0.0, 0.0, 0.0];
        let z = eval_model(&p, 1, &[x], &[y]);
        assert!((z[[0, 0]] - expected).abs() < 1e-14, "z = {}, expected = {}", z[[0, 0]], expected);
    }

    #[test]
    fn recovers_known_single_mode_from_noisy_data() {
        // Synthesize a single-mode rotGauss + small noise, then recover.
        let (xg, yg) = make_grid(40, 40);
        let true_p = [3.0_f64, 12.0, 2.5, 10.0, 15.0, std::f64::consts::PI / 12.0, 0.0, 0.0, 0.5];
        let n_modes = 1;
        let z_clean = eval_model(&true_p, n_modes, &xg, &yg);
        // Deterministic pseudo-noise (no RNG dep).
        let mut z_noisy = z_clean.clone();
        let nx = xg.len();
        let ny = yg.len();
        for iy in 0..ny {
            for ix in 0..nx {
                let n = ((iy as f64 * 31.0 + ix as f64 * 17.0).sin() * 0.05).abs();
                z_noisy[[iy, ix]] += n;
            }
        }

        let initial = Array2::from_shape_vec((1, 6),
            vec![2.5, 13.0, 3.0, 11.0, 16.0, 0.0]).unwrap();
        let lower = Array2::from_shape_vec((1, 6),
            vec![0.1, 0.5, 0.1, -50.0, 2.5, -std::f64::consts::PI / 4.0]).unwrap();
        let upper = Array2::from_shape_vec((1, 6),
            vec![30.0, 30.0, 5.0, 50.0, 30.0, std::f64::consts::PI / 4.0]).unwrap();

        let out = fit_rotgauss(
            z_noisy.view(), &xg, &yg,
            initial.view(), lower.view(), upper.view(),
            [0.0, 0.0, 0.1], [-0.1, -0.1, 0.0], [0.1, 0.1, 5.0],
            500,
        ).unwrap();

        assert!(out.gof.rsquare > 0.99,
            "Recovery R² too low: {} (sse={}, dfe={})", out.gof.rsquare, out.gof.sse, out.gof.dfe);
        // Loose parameter check — LM minima drift on noisy data.
        assert!((out.params[[0, 0]] - 3.0).abs() < 0.5, "amp drift: {}", out.params[[0, 0]]);
        assert!((out.params[[0, 1]] - 12.0).abs() < 0.5, "fmean drift: {}", out.params[[0, 1]]);
    }
}
