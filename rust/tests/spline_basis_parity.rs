//! Parity test: Rust `fit_tensor_product_spline` vs MATLAB `spap2` on
//! synthesized SOPH fixtures (`/tmp/generate_spline_fixtures.m`).
//!
//! Budgets:
//!   * `coefs`        ≤ 1e-9   elementwise (separable normal-equations f64 floor)
//!   * `splinefit`    ≤ 1e-8   elementwise
//!   * `knots_x_aug`, `knots_y_aug`  bit-exact match to `spline_obj.knots{}`
//!
//! Fixtures live in `tests/fixtures/spline/sim_{power,phase}_*.npy`.

use ndarray::{Array1, Array2};
use ndarray_npy::read_npy;
use std::path::PathBuf;

use dynamo_rs::spline_basis::fit_tensor_product_spline;

fn fix_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/spline")
}

fn load_vec(name: &str) -> Vec<f64> {
    let p = fix_dir().join(name);
    let a: Array1<f64> = read_npy(&p).unwrap_or_else(|e| panic!("read {}: {}", p.display(), e));
    a.to_vec()
}

fn load_mat(name: &str) -> Array2<f64> {
    let p = fix_dir().join(name);
    read_npy(&p).unwrap_or_else(|e| panic!("read {}: {}", p.display(), e))
}

fn max_abs_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    assert_eq!(a.dim(), b.dim(), "shape mismatch");
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

fn check_one(typ: &str, coef_tol: f64, fit_tol: f64) {
    let soph_input = load_mat(&format!("sim_{}_SOPH_input.npy", typ));      // (n_feat, n_freq)
    let feat_bins = load_vec(&format!("sim_{}_feat_bins.npy", typ));
    let freq_bins = load_vec(&format!("sim_{}_freq_bins.npy", typ));
    let internal_x = load_vec(&format!("sim_{}_internal_knots_x.npy", typ));
    let internal_y = load_vec(&format!("sim_{}_internal_knots_y.npy", typ));
    let mat_coefs = load_mat(&format!("sim_{}_coefs.npy", typ));            // (m_y, m_x)
    let mat_fit = load_mat(&format!("sim_{}_fit.npy", typ));                // (n_feat, n_freq)
    let mat_kx_aug = load_vec(&format!("sim_{}_knots_x_aug.npy", typ));
    let mat_ky_aug = load_vec(&format!("sim_{}_knots_y_aug.npy", typ));

    let out = fit_tensor_product_spline(
        soph_input.view(),
        &feat_bins,
        &freq_bins,
        &internal_x,
        &internal_y,
        4,
        3, // DYNAM-O uses augknt(.,3) — multiplicity-3 boundary knots
    )
    .expect("fit_tensor_product_spline");

    assert_eq!(
        out.knots_x_aug, mat_kx_aug,
        "[{}] knots_x_aug not bit-exact", typ
    );
    assert_eq!(
        out.knots_y_aug, mat_ky_aug,
        "[{}] knots_y_aug not bit-exact", typ
    );

    let dc = max_abs_diff(&out.coefs, &mat_coefs);
    let df = max_abs_diff(&out.splinefit, &mat_fit);
    println!("[{}] coefs max abs diff = {:.3e}, fit max abs diff = {:.3e}", typ, dc, df);
    assert!(dc < coef_tol, "[{}] coefs: max abs diff {:.3e} >= {:.3e}", typ, dc, coef_tol);
    assert!(df < fit_tol,  "[{}] fit:   max abs diff {:.3e} >= {:.3e}", typ, df, fit_tol);
}

#[test]
fn spline_basis_matches_matlab_power() {
    // Observed: coefs ~2e-14, fit ~8e-15 — essentially f64 round-off.
    check_one("power", 1e-12, 1e-12);
}

#[test]
fn spline_basis_matches_matlab_phase() {
    // Observed: coefs ~5e-16, fit ~2e-16 — essentially f64 round-off.
    check_one("phase", 1e-12, 1e-12);
}
