//! Bivariate cubic B-spline least-squares fit on a regular grid.
//!
//! Port of MATLAB Curve Fitting Toolbox's
//! `spap2({augknt(knots_x,3), augknt(knots_y,3)}, [4 4], {x, y}, SOPH')`
//! as called from
//! `DYNAM-O/toolbox/SOPH_dim_reduction/spline_basis/spline_basis.m`.
//!
//! # Algorithm
//!
//! The fit is a separable bivariate LSQ — on a regular grid the normal
//! equations factor into two 1-D problems (Boor 1978, *A Practical Guide
//! to Splines*, §XVII). Given the 1-D basis matrices
//!
//! ```text
//! B_x[i, a] = N_a(x_i)   shape (n_x, m_x)
//! B_y[j, b] = N_b(y_j)   shape (n_y, m_y)
//! ```
//!
//! and data Z of shape `(n_x, n_y)`, the LSQ-optimal coefficients are
//!
//! ```text
//! C = (B_x^T B_x)^-1  B_x^T  Z  B_y  (B_y^T B_y)^-1     shape (m_x, m_y)
//! ```
//!
//! Both Gram matrices are small SPD (tens of rows) so we solve via
//! `nalgebra::Cholesky`.
//!
//! The 1-D basis matrices are built via Cox–de Boor recursion on the
//! augmented knot vector produced by [`augknt`].
//!
//! # Output orientation
//!
//! Matches `spline_basis.m`'s output convention:
//! * `coefs.shape == (m_y, m_x)`  (the `squeeze(spline_obj.coefs)'` step)
//! * `splinefit.shape == (n_x, n_y)` (reshape of `fnval` on `[X(:)'; Y(:)']`,
//!   which `spline_basis.m` reshapes to `size(SOPH') = (n_features, n_freqs)`).
//!
//! The MATLAB-side wrapper transposes from MATLAB `(n_freqs, n_features)`
//! column-major to `(n_features, n_freqs)` row-major before the FFI call;
//! `feature_bins` is the x-axis, `freq_bins` is the y-axis.

use nalgebra::{Cholesky, DMatrix};
use ndarray::{Array2, ArrayView2};

#[derive(Debug)]
pub enum SplineError {
    InvalidArgument(String),
    Singular,
}

impl std::fmt::Display for SplineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SplineError::InvalidArgument(s) => write!(f, "invalid argument: {}", s),
            SplineError::Singular => write!(f, "Cholesky decomposition failed (singular Gram matrix)"),
        }
    }
}

impl std::error::Error for SplineError {}

/// Output of [`fit_tensor_product_spline`].
pub struct SplineBasisOut {
    /// Coefficient matrix, shape `(m_y, m_x)` — matches MATLAB
    /// `squeeze(spline_obj.coefs)'`.
    pub coefs: Array2<f64>,
    /// Splinefit reconstruction on the input grid, shape `(n_x, n_y)`.
    pub splinefit: Array2<f64>,
    /// Augmented knot vectors (the output of `augknt(internal, order-1)`).
    pub knots_x_aug: Vec<f64>,
    pub knots_y_aug: Vec<f64>,
}

/// Fit a tensor-product B-spline on a regular grid via separable normal-equations LSQ.
///
/// * `soph` — input data, shape `(n_x, n_y)` where x = feature axis,
///   y = freq axis. This is the orientation that MATLAB's `spline_basis.m`
///   feeds to `spap2` (i.e. `SOPH'` of the originally-`(n_freqs,
///   n_features)`-laid-out histogram).
/// * `x_eval`, `y_eval` — grid points on each axis, length `n_x` / `n_y`.
/// * `internal_knots_x`, `internal_knots_y` — pre-augmented knot
///   vectors (the spline_basis.m `knots_x` / `knots_y` locals at line 162).
/// * `order` — spline order (4 = cubic; MATLAB uses 4 for both axes).
/// * `boundary_multiplicity` — `augknt(.,k)` second argument; 3 for the
///   DYNAM-O spline (under-clamped: order-4 spline with multiplicity-3
///   boundary knots).
pub fn fit_tensor_product_spline(
    soph: ArrayView2<f64>,
    x_eval: &[f64],
    y_eval: &[f64],
    internal_knots_x: &[f64],
    internal_knots_y: &[f64],
    order: usize,
    boundary_multiplicity: usize,
) -> Result<SplineBasisOut, SplineError> {
    let (n_x, n_y) = soph.dim();
    if n_x != x_eval.len() {
        return Err(SplineError::InvalidArgument(format!(
            "soph rows {} != x_eval len {}",
            n_x,
            x_eval.len()
        )));
    }
    if n_y != y_eval.len() {
        return Err(SplineError::InvalidArgument(format!(
            "soph cols {} != y_eval len {}",
            n_y,
            y_eval.len()
        )));
    }
    if order < 2 || order > 6 {
        return Err(SplineError::InvalidArgument(format!(
            "unsupported order {}",
            order
        )));
    }
    if boundary_multiplicity == 0 || boundary_multiplicity > order {
        return Err(SplineError::InvalidArgument(format!(
            "bad boundary_multiplicity {}",
            boundary_multiplicity
        )));
    }
    // find_span / Cox-de Boor require n_basis >= order. Since
    // n_basis = augmented_knots.len() - order, the augmented vector must
    // contain at least 2 * order knots.
    let min_internal_knots = 2 * order - 2 * (boundary_multiplicity - 1);
    for (name, knots) in [
        ("internal_knots_x", internal_knots_x),
        ("internal_knots_y", internal_knots_y),
    ] {
        if knots.len() < min_internal_knots {
            return Err(SplineError::InvalidArgument(format!(
                "{} length {} is too short for order {} and boundary_multiplicity {}; need at least {}",
                name,
                knots.len(),
                order,
                boundary_multiplicity,
                min_internal_knots
            )));
        }
    }

    let knots_x_aug = augknt(internal_knots_x, boundary_multiplicity);
    let knots_y_aug = augknt(internal_knots_y, boundary_multiplicity);

    // 1-D basis matrices.
    let bx = bspline_basis_matrix(&knots_x_aug, order, x_eval); // (n_x, m_x)
    let by = bspline_basis_matrix(&knots_y_aug, order, y_eval); // (n_y, m_y)
    let m_x = bx.ncols();
    let m_y = by.ncols();

    // Build dense f64 nalgebra matrices.
    let bx_n = ndarray_to_nalgebra(&bx);
    let by_n = ndarray_to_nalgebra(&by);
    let soph_n = ndarray_to_nalgebra(&soph.to_owned());

    // Gram matrices.
    let bxtbx = bx_n.transpose() * &bx_n;
    let bytby = by_n.transpose() * &by_n;

    // Solve (Bx^T Bx)^-1 Bx^T Z = M_left  via Cholesky.
    let chol_x = Cholesky::new(bxtbx).ok_or(SplineError::Singular)?;
    let m_left = chol_x.solve(&(bx_n.transpose() * &soph_n)); // (m_x, n_y)

    // Compute m_left * By, then solve (By^T By)^-1 * (m_left * By)^T = C^T.
    // i.e. coefs = (m_left * By) * (By^T By)^-1, shape (m_x, m_y).
    let m_left_by = &m_left * &by_n; // (m_x, m_y)
    let chol_y = Cholesky::new(bytby).ok_or(SplineError::Singular)?;
    let coefs_mxmy = chol_y.solve(&m_left_by.transpose()).transpose(); // (m_x, m_y)

    // Match MATLAB layout: coefs is (m_y, m_x) after `squeeze(...)'`.
    let mut coefs_arr = Array2::<f64>::zeros((m_y, m_x));
    for a in 0..m_x {
        for b in 0..m_y {
            coefs_arr[[b, a]] = coefs_mxmy[(a, b)];
        }
    }

    // Reconstruction on the grid: splinefit = Bx * coefs_mxmy * By^T.
    let recon_n = &bx_n * &coefs_mxmy * by_n.transpose(); // (n_x, n_y)
    let mut splinefit = Array2::<f64>::zeros((n_x, n_y));
    for i in 0..n_x {
        for j in 0..n_y {
            splinefit[[i, j]] = recon_n[(i, j)];
        }
    }

    Ok(SplineBasisOut {
        coefs: coefs_arr,
        splinefit,
        knots_x_aug,
        knots_y_aug,
    })
}

/// MATLAB `augknt(knots, k)`: prepend k-1 copies of `knots[0]` and append
/// k-1 copies of `knots[-1]`. Returns length `knots.len() + 2*(k-1)`.
pub fn augknt(knots: &[f64], k: usize) -> Vec<f64> {
    assert!(!knots.is_empty(), "augknt: empty knot vector");
    assert!(k >= 1, "augknt: k must be >= 1");
    let pad = k - 1;
    let mut out = Vec::with_capacity(knots.len() + 2 * pad);
    for _ in 0..pad {
        out.push(knots[0]);
    }
    out.extend_from_slice(knots);
    for _ in 0..pad {
        out.push(*knots.last().unwrap());
    }
    out
}

/// Compute the 1-D B-spline basis matrix `B[i, a] = N_a^{order}(x_eval[i])`
/// on the (already-augmented) knot vector `t`.
///
/// `n_basis = t.len() - order`.
///
/// Implementation: Cox–de Boor recursion. For each evaluation point we
/// locate the knot span and recurse upward; matches `scipy.interpolate.BSpline.design_matrix`
/// to f64 round-off.
///
/// **Boundary convention** matches MATLAB Curve Fitting Toolbox `spcol` /
/// `spap2`: evaluating *exactly* at the right boundary knot returns the
/// limit-from-the-left (so the last basis function evaluates to 1.0 at
/// `t[t.len()-1]` rather than to 0.0). We implement this by clamping
/// the eval point's "active" span at `n_basis - 1` if `x == t[-1]`.
pub fn bspline_basis_matrix(t: &[f64], order: usize, x_eval: &[f64]) -> Array2<f64> {
    let degree = order - 1;
    let n_basis = t.len().saturating_sub(order);
    let n_eval = x_eval.len();
    let mut out = Array2::<f64>::zeros((n_eval, n_basis));

    for (i, &x) in x_eval.iter().enumerate() {
        // Locate span k such that t[k] <= x < t[k+1], with a right-boundary
        // clamp so x == t[n_basis + degree] still hits the last span.
        let k = find_span(t, n_basis, degree, x);
        // Compute the `order` non-zero basis functions at x: N[k-degree..=k].
        let basis = cox_de_boor(t, degree, k, x);
        for j in 0..order {
            let col = k + j - degree;
            if col < n_basis {
                out[[i, col]] = basis[j];
            }
        }
    }
    out
}

fn find_span(t: &[f64], n_basis: usize, degree: usize, x: f64) -> usize {
    // Span index k means the non-zero basis at x ∈ [t[k], t[k+1]] are
    // N_{k-degree}..=N_k. Valid k range: [degree, n_basis - 1].
    let high = n_basis - 1;
    // Right boundary: MATLAB spap2 / spcol's "extend-from-left" convention —
    // x == t[n_basis] still hits the last span.
    if x >= t[n_basis] {
        return high;
    }
    if x <= t[degree] {
        return degree;
    }
    // Binary search: find k with t[k] <= x < t[k+1].
    let mut lo = degree;
    let mut hi = n_basis;
    let mut mid = (lo + hi) / 2;
    while x < t[mid] || x >= t[mid + 1] {
        if x < t[mid] {
            hi = mid;
        } else {
            lo = mid;
        }
        mid = (lo + hi) / 2;
    }
    mid
}

fn cox_de_boor(t: &[f64], degree: usize, k: usize, x: f64) -> Vec<f64> {
    // Returns the `degree + 1` non-zero basis-function values at x for the
    // knot span starting at index k. Standard de-Boor recursion (e.g.
    // Piegl & Tiller, *The NURBS Book*, Algorithm A2.2).
    let p = degree;
    let mut left = vec![0.0_f64; p + 1];
    let mut right = vec![0.0_f64; p + 1];
    let mut n = vec![0.0_f64; p + 1];
    n[0] = 1.0;
    for j in 1..=p {
        left[j] = x - t[k + 1 - j];
        right[j] = t[k + j] - x;
        let mut saved = 0.0_f64;
        for r in 0..j {
            let denom = right[r + 1] + left[j - r];
            if denom == 0.0 {
                n[r] = saved;
                saved = 0.0;
                continue;
            }
            let temp = n[r] / denom;
            n[r] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }
        n[j] = saved;
    }
    n
}

fn ndarray_to_nalgebra(a: &Array2<f64>) -> DMatrix<f64> {
    let (n_rows, n_cols) = a.dim();
    DMatrix::from_fn(n_rows, n_cols, |i, j| a[[i, j]])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_invalid_knot_error(
        result: Result<SplineBasisOut, SplineError>,
        axis_name: &str,
    ) {
        match result {
            Err(SplineError::InvalidArgument(message)) => {
                assert!(message.contains(axis_name), "unexpected error: {}", message);
            }
            Err(error) => panic!("expected InvalidArgument, got {}", error),
            Ok(_) => panic!("invalid knot vector was accepted"),
        }
    }

    #[test]
    fn fit_rejects_empty_knot_vector() {
        let soph = Array2::<f64>::zeros((2, 2));
        let eval = [0.0, 1.0];
        let valid_knots = [0.0, 0.25, 0.75, 1.0];
        let result = fit_tensor_product_spline(
            soph.view(),
            &eval,
            &eval,
            &[],
            &valid_knots,
            4,
            3,
        );
        assert_invalid_knot_error(result, "internal_knots_x");
    }

    #[test]
    fn fit_rejects_too_few_augmented_knots() {
        let soph = Array2::<f64>::zeros((2, 2));
        let eval = [0.0, 1.0];
        let valid_knots = [0.0, 0.25, 0.75, 1.0];
        let too_short = [0.0, 0.5, 1.0];
        let result = fit_tensor_product_spline(
            soph.view(),
            &eval,
            &eval,
            &valid_knots,
            &too_short,
            4,
            3,
        );
        assert_invalid_knot_error(result, "internal_knots_y");
    }

    #[test]
    fn augknt_basic() {
        let t = vec![0.0, 1.0, 2.0, 3.0];
        assert_eq!(augknt(&t, 1), t);
        assert_eq!(augknt(&t, 2), vec![0.0, 0.0, 1.0, 2.0, 3.0, 3.0]);
        assert_eq!(
            augknt(&t, 3),
            vec![0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0]
        );
        assert_eq!(
            augknt(&t, 4),
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0]
        );
    }

    #[test]
    fn basis_partition_of_unity() {
        // For any x in [t[degree], t[n_basis]], the cubic B-spline basis
        // functions sum to 1. (Partition-of-unity property.)
        let internal = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        let t = augknt(&internal, 4); // multiplicity-4 boundary -> order-4 cubic
        let order = 4;
        let xs: Vec<f64> = (0..=100).map(|i| i as f64 / 100.0).collect();
        let b = bspline_basis_matrix(&t, order, &xs);
        for i in 0..xs.len() {
            let s: f64 = b.row(i).sum();
            assert!(
                (s - 1.0).abs() < 1e-10,
                "row {}: sum = {} (x = {})",
                i,
                s,
                xs[i]
            );
        }
    }

    #[test]
    fn fit_quadratic_exactly_with_cubic_basis() {
        // A cubic B-spline can reproduce any quadratic exactly. Fit
        // f(x, y) = (x+1)*(y-2)^2 on a regular grid and confirm the
        // reconstruction is bit-near the input.
        let nx = 31;
        let ny = 41;
        // Eval range must span the full internal-knot range so every basis
        // function has data support — otherwise the Gram is singular.
        let x_eval: Vec<f64> = (0..nx).map(|i| -2.0 + 6.0 * i as f64 / (nx - 1) as f64).collect();
        let y_eval: Vec<f64> = (0..ny).map(|j| -1.0 + 6.0 * j as f64 / (ny - 1) as f64).collect();
        let mut z = Array2::<f64>::zeros((nx, ny));
        for i in 0..nx {
            for j in 0..ny {
                z[[i, j]] = (x_eval[i] + 1.0) * (y_eval[j] - 2.0).powi(2);
            }
        }
        // Knots span the eval range with no over-extension (otherwise the
        // outer basis functions have no eval support and the Gram is singular).
        let internal_x = vec![-2.0, -0.5, 1.0, 2.5, 4.0];
        let internal_y = vec![-1.0, 0.5, 2.0, 3.5, 5.0];
        let out = fit_tensor_product_spline(z.view(), &x_eval, &y_eval, &internal_x, &internal_y, 4, 4).unwrap();

        let mut max_err = 0.0_f64;
        for i in 0..nx {
            for j in 0..ny {
                max_err = max_err.max((out.splinefit[[i, j]] - z[[i, j]]).abs());
            }
        }
        // f64-noise bound on a 31x41 cubic LSQ.
        assert!(max_err < 1e-9, "max reconstruction error = {} (>1e-9)", max_err);
    }
}
