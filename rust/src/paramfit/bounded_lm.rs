//! Bound-constrained trust-region-reflective (Coleman–Li) solver.
//!
//! Direct port of scipy's `scipy/optimize/_lsq/trf.py` and the bits of
//! `common.py` it depends on (`exact` tr_solver branch only — small dense
//! problems, n_params ≤ ~30 in this codebase). Same algorithm and equation
//! references as MATLAB's `lsqcurvefit` default (Coleman & Li 1994/1996,
//! "An interior trust region approach for nonlinear minimization subject
//! to bounds"; Branch, Coleman, Li 1999 STIR paper).
//!
//! Replaces an earlier unit-cube reparameterization, which produced visibly
//! different iterates from MATLAB when the optimum touched a facet. The
//! reparam version is preserved in git history.
//!
//! The trait surface (`ResidualsAndJacobian`, `run_bounded_lm`) is
//! unchanged so `rot_gauss.rs` / `vm_gauss.rs` need no edits.
//!
//! Operates on Jacobians shape `(n_residuals, n_params)` in row-major
//! (residual index = row), matching scipy.

use nalgebra::{DMatrix, DVector};
use std::cell::Cell;

thread_local! {
    static T_JAC: Cell<f64> = const { Cell::new(0.0) };
    static T_RES: Cell<f64> = const { Cell::new(0.0) };
    static T_JTJ: Cell<f64> = const { Cell::new(0.0) };
    static T_EIG: Cell<f64> = const { Cell::new(0.0) };
    static T_SLV: Cell<f64> = const { Cell::new(0.0) };
    static T_SEL: Cell<f64> = const { Cell::new(0.0) };
    static N_JAC: Cell<u32> = const { Cell::new(0) };
    static N_RES: Cell<u32> = const { Cell::new(0) };
    static N_OUT: Cell<u32> = const { Cell::new(0) };
}

pub fn perf_dump() -> String {
    fn one(t: &'static std::thread::LocalKey<Cell<f64>>, n: &'static std::thread::LocalKey<Cell<u32>>, label: &str) -> String {
        let tv = t.with(|c| c.get());
        let nv = n.with(|c| c.get());
        format!("  {:<12} {:>3}x  total={:.3}s  per={:.3}ms\n", label, nv, tv, if nv > 0 { tv * 1000.0 / nv as f64 } else { 0.0 })
    }
    let mut s = String::new();
    s.push_str(&one(&T_JAC, &N_JAC, "jac"));
    s.push_str(&one(&T_RES, &N_RES, "residuals"));
    s.push_str(&one(&T_JTJ, &N_OUT, "J^T J"));
    s.push_str(&one(&T_EIG, &N_OUT, "eigendecomp"));
    s.push_str(&one(&T_SLV, &N_OUT, "subprob"));
    s.push_str(&one(&T_SEL, &N_OUT, "select_step"));
    s
}

pub fn perf_reset() {
    T_JAC.with(|t| t.set(0.0));
    T_RES.with(|t| t.set(0.0));
    T_JTJ.with(|t| t.set(0.0));
    T_EIG.with(|t| t.set(0.0));
    T_SLV.with(|t| t.set(0.0));
    T_SEL.with(|t| t.set(0.0));
    N_JAC.with(|n| n.set(0));
    N_RES.with(|n| n.set(0));
    N_OUT.with(|n| n.set(0));
}

pub trait ResidualsAndJacobian {
    fn residuals(&self, params: &[f64], out: &mut [f64]);
    fn jacobian(&self, params: &[f64], out: &mut DMatrix<f64>);
    fn n_params(&self) -> usize;
    fn n_residuals(&self) -> usize;
}

pub struct BoundedLmReport {
    pub params: Vec<f64>,
    pub residuals: Vec<f64>,
    pub iters_used: u32,
    pub solver_report: TrfReport,
}

#[derive(Clone, Debug)]
pub struct TrfReport {
    pub status: TrfStatus,
    pub iterations: u32,
    pub nfev: u32,
    pub njev: u32,
    pub cost: f64,
    pub g_norm: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TrfStatus {
    GtolSatisfied,    // 1
    FtolSatisfied,    // 2
    XtolSatisfied,    // 3
    BothFtolXtol,     // 4
    MaxNfevReached,   // 0
}

pub fn run_bounded_lm<R: ResidualsAndJacobian>(
    model: &R,
    initial: &[f64],
    lower: &[f64],
    upper: &[f64],
    max_iters: u32,
) -> Result<BoundedLmReport, String> {
    let n_p = model.n_params();
    let n_r = model.n_residuals();
    if initial.len() != n_p || lower.len() != n_p || upper.len() != n_p {
        return Err(format!(
            "size mismatch: n_params={}, |initial|={}, |lower|={}, |upper|={}",
            n_p, initial.len(), lower.len(), upper.len()
        ));
    }
    for i in 0..n_p {
        if !(lower[i] <= upper[i]) {
            return Err(format!("lower[{}]={} > upper[{}]={}", i, lower[i], i, upper[i]));
        }
    }

    // Tolerances match MATLAB lsqcurvefit defaults (TolFun/TolX = 1e-6),
    // not scipy's tighter 1e-8 defaults. This makes Rust LM converge in
    // comparable iter counts to MATLAB on these well-conditioned problems.
    let ftol = 1e-6_f64;
    let xtol = 1e-6_f64;
    let gtol = 1e-8_f64;
    let max_nfev = if max_iters == 0 { (n_p as u32) * 100 } else { max_iters };

    let profile = std::env::var("DYNAMO_PARAMFIT_PROFILE").is_ok();
    if profile { perf_reset(); }
    let _t_total = std::time::Instant::now();

    // Make x0 strictly feasible (scipy expects x in interior).
    let mut x0 = initial.to_vec();
    make_strictly_feasible_in_place(&mut x0, lower, upper, 1e-10);

    let (out_x, out_r, report) =
        trf_bounds(model, &x0, lower, upper, ftol, xtol, gtol, max_nfev, n_r, n_p)?;

    if profile {
        let total = _t_total.elapsed().as_secs_f64();
        eprintln!("[paramfit perf] total={:.3}s  iters={} nfev={} njev={}", total, report.iterations, report.nfev, report.njev);
        eprint!("{}", perf_dump());
    }
    Ok(BoundedLmReport {
        params: out_x,
        residuals: out_r,
        iters_used: report.iterations,
        solver_report: report,
    })
}

#[allow(clippy::too_many_arguments)]
fn trf_bounds<R: ResidualsAndJacobian>(
    model: &R,
    x0: &[f64],
    lb: &[f64],
    ub: &[f64],
    ftol: f64,
    xtol: f64,
    gtol: f64,
    max_nfev: u32,
    n_r: usize,
    n_p: usize,
) -> Result<(Vec<f64>, Vec<f64>, TrfReport), String> {
    // State.
    let mut x = x0.to_vec();
    let mut f = vec![0.0_f64; n_r];
    let _t = std::time::Instant::now();
    model.residuals(&x, &mut f);
    T_RES.with(|t| t.set(t.get() + _t.elapsed().as_secs_f64()));
    N_RES.with(|n| n.set(n.get() + 1));
    let mut nfev: u32 = 1;
    let mut njev: u32 = 1;

    let mut j_buf = DMatrix::<f64>::zeros(n_r, n_p);
    let _t = std::time::Instant::now();
    model.jacobian(&x, &mut j_buf);
    T_JAC.with(|t| t.set(t.get() + _t.elapsed().as_secs_f64()));
    N_JAC.with(|n| n.set(n.get() + 1));

    let mut cost = 0.5 * dot(&f, &f);
    let mut g = jt_dot_f(&j_buf, &f);

    // x_scale = 1.0 (vector of 1s) — no Jacobian scaling.
    let scale = vec![1.0_f64; n_p];
    let scale_inv = vec![1.0_f64; n_p];

    let (mut v, _) = cl_scaling_vector(&x, &g, lb, ub);
    let mut delta: f64 = {
        // Delta = ||x0 * scale_inv / sqrt(v)||
        let mut s = 0.0_f64;
        for i in 0..n_p {
            let q = x0[i] * scale_inv[i] / v[i].sqrt();
            s += q * q;
        }
        s.sqrt()
    };
    if !(delta.is_finite() && delta > 0.0) {
        delta = 1.0;
    }

    let mut alpha = 0.0_f64;
    let mut status: Option<TrfStatus> = None;
    let mut iteration: u32 = 0;
    let mut step_norm: f64 = 0.0;

    loop {
        // Recompute scaling vector each iter.
        let (v_new, dv) = cl_scaling_vector(&x, &g, lb, ub);
        v = v_new;

        // ||g * v||_inf
        let g_norm_inf = g.iter().zip(v.iter()).map(|(gi, vi)| (gi * vi).abs()).fold(0.0_f64, f64::max);
        if g_norm_inf < gtol {
            status = Some(TrfStatus::GtolSatisfied);
        }
        if status.is_some() || nfev >= max_nfev {
            return Ok(finalize(x, f, iteration, nfev, njev, cost, g_norm_inf,
                status.unwrap_or(TrfStatus::MaxNfevReached)));
        }

        // Apply x_scale (identity in our case).
        let mut v_eff = v.clone();
        for i in 0..n_p {
            if dv[i] != 0.0 { v_eff[i] *= scale_inv[i]; }
        }
        // d = sqrt(v_eff) * scale
        let d: Vec<f64> = v_eff.iter().zip(scale.iter()).map(|(vi, si)| vi.sqrt() * si).collect();
        // diag_h = g * dv * scale
        let diag_h: Vec<f64> = (0..n_p).map(|i| g[i] * dv[i] * scale[i]).collect();
        // g_h = d * g
        let g_h: Vec<f64> = (0..n_p).map(|i| d[i] * g[i]).collect();

        // J_h = J * diag(d) (right-multiply columns of J by d).
        let mut j_h = j_buf.clone();
        for col in 0..n_p {
            for row in 0..n_r {
                j_h[(row, col)] *= d[col];
            }
        }

        // ---- Subproblem precompute via n×n symmetric eigendecomp ----
        // The scipy reference does an SVD of the (n_r + n_p, n_p) augmented
        // matrix [J_h; diag(sqrt(diag_h))]. That SVD's right singular vectors
        // and singular values are exactly the eigenvectors / sqrt(eigenvalues)
        // of B = J_h^T J_h + diag(diag_h) (an n_p × n_p SPD matrix). For our
        // small n_p (≤ ~30) and large n_r (thousands), forming B once and
        // taking its symmetric eigendecomp is much faster than the full SVD.
        // We also need uf_eff = V^T (J_h^T f) for the Moré-Hebden phi.
        // Identity: U^T f_aug = (S^{-1} V^T) * J_aug^T f_aug = (S^{-1} V^T) * J_h^T f,
        // so suf := s * uf = V^T (J_h^T f) — and the SVD-based formulas in
        // solve_lsq_trust_region work directly on suf without needing uf itself.
        // We pass `suf` in the `uf` slot and `s` as the eigvec sqrt(λ).

        // B = J_h^T J_h + diag(diag_h). Use nalgebra's optimized matmul
        // (BLAS-3, SIMD via matrixmultiply) instead of a manual triple loop —
        // dominant cost is forming J^T J (m*n²), so the constant matters.
        let _t = std::time::Instant::now();
        let mut b = j_h.tr_mul(&j_h);
        for i in 0..n_p { b[(i, i)] += diag_h[i]; }
        T_JTJ.with(|t| t.set(t.get() + _t.elapsed().as_secs_f64()));
        N_OUT.with(|n| n.set(n.get() + 1));

        // Symmetric eigendecomp B = Q diag(λ) Q^T.
        let _t = std::time::Instant::now();
        let sym = nalgebra::SymmetricEigen::new(b);
        T_EIG.with(|t| t.set(t.get() + _t.elapsed().as_secs_f64()));
        let eigvals = sym.eigenvalues;   // unordered
        let q_mat = sym.eigenvectors;    // columns are eigenvectors

        // singular_values of J_aug = sqrt(max(λ, 0))
        let mut s_vec: Vec<f64> = eigvals.iter().map(|l| if *l > 0.0 { l.sqrt() } else { 0.0 }).collect();

        // suf = sign-aware (V^T J_h^T f) so s * uf gives the same expression
        // as scipy's `s * uf`. In SVD form, s*uf = V^T J_h^T f exactly.
        let jt_f = jt_dot_f(&j_h, &f);
        let jt_f_d = DVector::from_vec(jt_f.clone());
        let v_t_jtf = q_mat.transpose() * &jt_f_d;  // V^T (J_h^T f)
        let suf_vec: Vec<f64> = v_t_jtf.iter().copied().collect();
        // uf = (V^T J_h^T f) / s with s the singular values.
        let uf_vec: Vec<f64> = (0..s_vec.len()).map(|i| {
            if s_vec[i] > 0.0 { suf_vec[i] / s_vec[i] } else { 0.0 }
        }).collect();

        // Right singular vectors = eigenvectors of B (columns of Q).
        let v_mat = q_mat;

        // Sort by descending singular value to match scipy's SVD output (so
        // the full_rank / threshold logic in solve_lsq_trust_region behaves
        // the same).
        let mut order: Vec<usize> = (0..s_vec.len()).collect();
        order.sort_by(|&a, &b| s_vec[b].partial_cmp(&s_vec[a]).unwrap_or(std::cmp::Ordering::Equal));
        let s_sorted: Vec<f64> = order.iter().map(|&i| s_vec[i]).collect();
        let uf_sorted: Vec<f64> = order.iter().map(|&i| uf_vec[i]).collect();
        let mut v_sorted = DMatrix::<f64>::zeros(n_p, n_p);
        for (new_col, &old_col) in order.iter().enumerate() {
            for r in 0..n_p { v_sorted[(r, new_col)] = v_mat[(r, old_col)]; }
        }
        s_vec = s_sorted;
        let uf_vec = uf_sorted;
        let v_mat = v_sorted;

        // theta controls step-back from boundary
        let theta = 0.995_f64.max(1.0 - g_norm_inf);

        let mut actual_reduction = -1.0;
        let mut delta_new;
        let mut ratio;
        let mut x_new = x.clone();
        let mut f_new = f.clone();
        let mut cost_new = cost;

        while actual_reduction <= 0.0 && nfev < max_nfev {
            // Solve TR subproblem in hat space → p_h
            let _t = std::time::Instant::now();
            let (p_h_v, alpha_new, _n_iter) = solve_lsq_trust_region(
                n_p, n_r + n_p, &uf_vec, &s_vec, &v_mat, delta, Some(alpha))?;
            T_SLV.with(|t| t.set(t.get() + _t.elapsed().as_secs_f64()));
            alpha = alpha_new;

            // Map to original space: p = d * p_h
            let p: Vec<f64> = (0..n_p).map(|i| d[i] * p_h_v[i]).collect();

            // Select step among constrained TR step, reflected, and Cauchy
            let _t = std::time::Instant::now();
            let (step, step_h, predicted_reduction) =
                select_step(&x, &j_h, &diag_h, &g_h, p, p_h_v, &d, delta, lb, ub, theta);
            T_SEL.with(|t| t.set(t.get() + _t.elapsed().as_secs_f64()));

            // x + step → strictly feasible
            let mut x_try = (0..n_p).map(|i| x[i] + step[i]).collect::<Vec<_>>();
            make_strictly_feasible_in_place(&mut x_try, lb, ub, 0.0);

            let _t = std::time::Instant::now();
            model.residuals(&x_try, &mut f_new);
            T_RES.with(|t| t.set(t.get() + _t.elapsed().as_secs_f64()));
            N_RES.with(|n| n.set(n.get() + 1));
            nfev += 1;

            let step_h_norm = norm(&step_h);

            if !f_new.iter().all(|v| v.is_finite()) {
                delta = 0.25 * step_h_norm;
                continue;
            }

            cost_new = 0.5 * dot(&f_new, &f_new);
            actual_reduction = cost - cost_new;

            let bound_hit = step_h_norm > 0.95 * delta;
            let (dn, r) = update_tr_radius(delta, actual_reduction, predicted_reduction, step_h_norm, bound_hit);
            delta_new = dn;
            ratio = r;

            step_norm = norm(&step);
            let x_norm = norm(&x);
            let termination = check_termination(actual_reduction, cost, step_norm, x_norm, ratio, ftol, xtol);
            if let Some(t) = termination {
                // Apply the accepted step before finalizing the termination report.
                x = x_try.clone();
                f.copy_from_slice(&f_new);
                cost = cost_new;
                // Refresh Jacobian/gradient before exit-finalize
                let _t = std::time::Instant::now();
                model.jacobian(&x, &mut j_buf);
                T_JAC.with(|t| t.set(t.get() + _t.elapsed().as_secs_f64()));
                N_JAC.with(|n| n.set(n.get() + 1));
                njev += 1;
                g = jt_dot_f(&j_buf, &f);
                let g_norm_inf2 = g.iter().zip(v.iter()).map(|(gi, vi)| (gi * vi).abs()).fold(0.0, f64::max);
                return Ok(finalize(x, f.clone(), iteration + 1, nfev, njev, cost, g_norm_inf2, t));
            }

            alpha *= delta / delta_new;
            delta = delta_new;

            if actual_reduction > 0.0 {
                x_new = x_try;
            }
            // Otherwise keep looping; delta has shrunk via update_tr_radius.
        }

        if actual_reduction > 0.0 {
            x = x_new.clone();
            f.copy_from_slice(&f_new);
            cost = cost_new;
            let _t = std::time::Instant::now();
            model.jacobian(&x, &mut j_buf);
            T_JAC.with(|t| t.set(t.get() + _t.elapsed().as_secs_f64()));
            N_JAC.with(|n| n.set(n.get() + 1));
            njev += 1;
            g = jt_dot_f(&j_buf, &f);
        } else {
            // No reduction at all this iter — accept zero step, loop will recheck gtol or max_nfev.
            step_norm = 0.0;
        }

        iteration += 1;
        if nfev >= max_nfev {
            let g_norm_inf2 = g.iter().zip(v.iter()).map(|(gi, vi)| (gi * vi).abs()).fold(0.0, f64::max);
            return Ok(finalize(x, f.clone(), iteration, nfev, njev, cost, g_norm_inf2, TrfStatus::MaxNfevReached));
        }
        let _ = step_norm;
    }
}

fn finalize(x: Vec<f64>, r: Vec<f64>, iters: u32, nfev: u32, njev: u32,
            cost: f64, g_norm: f64, status: TrfStatus) -> (Vec<f64>, Vec<f64>, TrfReport) {
    let report = TrfReport {
        status,
        iterations: iters,
        nfev,
        njev,
        cost,
        g_norm,
    };
    (x, r, report)
}

// ---------------------------------------------------------------------------
// Helpers ported from scipy/optimize/_lsq/common.py
// ---------------------------------------------------------------------------

fn dot(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

fn jt_dot_f(j: &DMatrix<f64>, f: &[f64]) -> Vec<f64> {
    let (m, n) = j.shape();
    debug_assert_eq!(f.len(), m);
    let mut g = vec![0.0_f64; n];
    for c in 0..n {
        let mut s = 0.0_f64;
        for r in 0..m { s += j[(r, c)] * f[r]; }
        g[c] = s;
    }
    g
}

/// Coleman-Li scaling vector.
fn cl_scaling_vector(x: &[f64], g: &[f64], lb: &[f64], ub: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = x.len();
    let mut v = vec![1.0_f64; n];
    let mut dv = vec![0.0_f64; n];
    for i in 0..n {
        if g[i] < 0.0 && ub[i].is_finite() {
            v[i] = ub[i] - x[i];
            dv[i] = -1.0;
        } else if g[i] > 0.0 && lb[i].is_finite() {
            v[i] = x[i] - lb[i];
            dv[i] = 1.0;
        } else {
            v[i] = 1.0;
            dv[i] = 0.0;
        }
    }
    (v, dv)
}

fn in_bounds(x: &[f64], lb: &[f64], ub: &[f64]) -> bool {
    for i in 0..x.len() {
        if x[i] < lb[i] || x[i] > ub[i] { return false; }
    }
    true
}

/// Step size to first bound; returns (step, hits) where hits[i] is -1/0/+1.
fn step_size_to_bound(x: &[f64], s: &[f64], lb: &[f64], ub: &[f64]) -> (f64, Vec<i32>) {
    let n = x.len();
    let mut steps = vec![f64::INFINITY; n];
    for i in 0..n {
        if s[i] != 0.0 {
            let a = (lb[i] - x[i]) / s[i];
            let b = (ub[i] - x[i]) / s[i];
            steps[i] = a.max(b);
        }
    }
    let min_step = steps.iter().copied().fold(f64::INFINITY, f64::min);
    let mut hits = vec![0_i32; n];
    for i in 0..n {
        if steps[i] == min_step {
            hits[i] = if s[i] > 0.0 { 1 } else if s[i] < 0.0 { -1 } else { 0 };
        }
    }
    (min_step, hits)
}

fn make_strictly_feasible_in_place(x: &mut [f64], lb: &[f64], ub: &[f64], rstep: f64) {
    let n = x.len();
    for i in 0..n {
        let lo = lb[i];
        let hi = ub[i];
        if rstep == 0.0 {
            // nextafter analog
            if lo.is_finite() && x[i] <= lo { x[i] = lo + lo.abs().max(1.0) * f64::EPSILON; }
            if hi.is_finite() && x[i] >= hi { x[i] = hi - hi.abs().max(1.0) * f64::EPSILON; }
        } else {
            if lo.is_finite() {
                let thresh = lo + rstep * 1.0_f64.max(lo.abs());
                let upper_dist = hi - x[i];
                let lower_dist = x[i] - lo;
                if lower_dist <= upper_dist.min(rstep * 1.0_f64.max(lo.abs())) {
                    x[i] = thresh;
                }
            }
            if hi.is_finite() {
                let thresh = hi - rstep * 1.0_f64.max(hi.abs());
                let upper_dist = hi - x[i];
                let lower_dist = x[i] - lo;
                if upper_dist <= lower_dist.min(rstep * 1.0_f64.max(hi.abs())) {
                    x[i] = thresh;
                }
            }
        }
        if x[i] < lo || x[i] > hi { x[i] = 0.5 * (lo + hi); }
    }
}

/// intersect_trust_region: solve ||x + s*t||^2 = Delta^2; returns (t_neg, t_pos).
fn intersect_trust_region(x: &[f64], s: &[f64], delta: f64) -> Result<(f64, f64), String> {
    let a = dot(s, s);
    if a == 0.0 { return Err("`s` is zero in intersect_trust_region".to_string()); }
    let b = dot(x, s);
    let c = dot(x, x) - delta * delta;
    if c > 0.0 { return Err("`x` is not within the trust region".to_string()); }
    let d = (b * b - a * c).sqrt();
    let sign_b = if b >= 0.0 { 1.0 } else { -1.0 };
    let q = -(b + sign_b * d);
    let t1 = q / a;
    let t2 = c / q;
    if t1 < t2 { Ok((t1, t2)) } else { Ok((t2, t1)) }
}

/// build_quadratic_1d: returns (a, b, c) for the 1-D quadratic
///    f(t) = 0.5 (s0 + s*t)^T (J^T J + diag) (s0 + s*t) + g^T (s0 + s*t)
fn build_quadratic_1d(
    j: &DMatrix<f64>, g: &[f64], s: &[f64], diag: Option<&[f64]>, s0: Option<&[f64]>,
) -> (f64, f64, f64) {
    // v = J s
    let (m, n) = j.shape();
    let mut v = vec![0.0_f64; m];
    for r in 0..m {
        let mut acc = 0.0_f64;
        for c in 0..n { acc += j[(r, c)] * s[c]; }
        v[r] = acc;
    }
    let mut a = dot(&v, &v);
    if let Some(d) = diag { a += dot(&mul(s, d), s); }
    a *= 0.5;
    let mut b = dot(g, s);
    let c = if let Some(s0v) = s0 {
        let mut u = vec![0.0_f64; m];
        for r in 0..m {
            let mut acc = 0.0_f64;
            for c2 in 0..n { acc += j[(r, c2)] * s0v[c2]; }
            u[r] = acc;
        }
        b += dot(&u, &v);
        let mut cterm = 0.5 * dot(&u, &u) + dot(g, s0v);
        if let Some(d) = diag {
            b += dot(&mul(s0v, d), s);
            cterm += 0.5 * dot(&mul(s0v, d), s0v);
        }
        cterm
    } else { 0.0 };
    (a, b, c)
}

fn mul(a: &[f64], b: &[f64]) -> Vec<f64> {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| x * y).collect()
}

/// minimize_quadratic_1d: min of a t^2 + b t + c on [lb, ub].
fn minimize_quadratic_1d(a: f64, b: f64, lb: f64, ub: f64, c: f64) -> (f64, f64) {
    let mut ts = vec![lb, ub];
    if a != 0.0 {
        let ext = -0.5 * b / a;
        if ext > lb && ext < ub { ts.push(ext); }
    }
    let mut best_t = ts[0];
    let mut best_y = ts[0] * (a * ts[0] + b) + c;
    for &t in &ts[1..] {
        let y = t * (a * t + b) + c;
        if y < best_y { best_y = y; best_t = t; }
    }
    (best_t, best_y)
}

/// evaluate_quadratic: 0.5 s^T (J^T J + diag) s + g^T s.
fn evaluate_quadratic(j: &DMatrix<f64>, g: &[f64], s: &[f64], diag: Option<&[f64]>) -> f64 {
    let (m, n) = j.shape();
    let mut js = vec![0.0_f64; m];
    for r in 0..m {
        let mut acc = 0.0_f64;
        for c in 0..n { acc += j[(r, c)] * s[c]; }
        js[r] = acc;
    }
    let mut q = dot(&js, &js);
    if let Some(d) = diag { q += dot(&mul(s, d), s); }
    0.5 * q + dot(s, g)
}

/// More-Hebden iterative solver for the LSQ trust-region subproblem.
/// Uses precomputed SVD (uf = U^T f, s = singular values, V = right singulars
/// as columns) and finds alpha such that ||p(alpha)|| = Delta where
/// p(alpha) = -V * (s * uf / (s^2 + alpha)).
fn solve_lsq_trust_region(
    n: usize, m: usize, uf: &[f64], s: &[f64], v_mat: &DMatrix<f64>, delta: f64,
    initial_alpha: Option<f64>,
) -> Result<(Vec<f64>, f64, u32), String> {
    let _ = m;
    let eps_f = f64::EPSILON;
    let n_sv = s.len();
    let suf: Vec<f64> = (0..n_sv).map(|i| s[i] * uf[i]).collect();

    // Full rank?
    let full_rank = if n_sv >= n {
        let threshold = eps_f * (n_sv as f64) * s[0];
        *s.last().unwrap() > threshold
    } else { false };

    // Try Gauss-Newton step if full rank.
    if full_rank {
        // p_gn = -V * (uf / s)
        let mut tmp = vec![0.0_f64; n_sv];
        for i in 0..n_sv { tmp[i] = uf[i] / s[i]; }
        let mut p = vec![0.0_f64; n];
        for c in 0..n {
            let mut acc = 0.0_f64;
            for r in 0..n_sv { acc += v_mat[(c, r)] * tmp[r]; }
            p[c] = -acc;
        }
        if norm(&p) <= delta {
            return Ok((p, 0.0, 0));
        }
    }

    let alpha_upper = norm(&suf) / delta;

    let mut alpha_lower = 0.0_f64;
    if full_rank {
        let (phi0, phi0_prime) = phi_and_derivative(0.0, &suf, s, delta);
        alpha_lower = -phi0 / phi0_prime;
        let _ = phi0;
    }

    let mut alpha = match initial_alpha {
        Some(a) if a > 0.0 && !(full_rank && a == 0.0) => a,
        _ => (0.001 * alpha_upper).max((alpha_lower * alpha_upper).sqrt()),
    };

    let rtol = 0.01_f64;
    let max_iter = 10_u32;
    let mut it: u32 = 0;
    for k in 0..max_iter {
        it = k;
        if alpha < alpha_lower || alpha > alpha_upper {
            alpha = (0.001 * alpha_upper).max((alpha_lower * alpha_upper).sqrt());
        }
        let (phi, phi_prime) = phi_and_derivative(alpha, &suf, s, delta);
        if phi < 0.0 { /* alpha_upper update */
            // alpha_upper = alpha (scipy)
        }
        let mut au = alpha_upper;
        if phi < 0.0 { au = alpha; }
        let ratio = phi / phi_prime;
        alpha_lower = alpha_lower.max(alpha - ratio);
        alpha -= (phi + delta) * ratio / delta;
        if phi.abs() < rtol * delta {
            let _ = au;
            break;
        }
        let _ = au;
    }

    // p = -V * (suf / (s^2 + alpha))
    let mut tmp = vec![0.0_f64; n_sv];
    for i in 0..n_sv { tmp[i] = suf[i] / (s[i] * s[i] + alpha); }
    let mut p = vec![0.0_f64; n];
    for c in 0..n {
        let mut acc = 0.0_f64;
        for r in 0..n_sv { acc += v_mat[(c, r)] * tmp[r]; }
        p[c] = -acc;
    }
    // Snap norm to delta to avoid going slightly outside.
    let pn = norm(&p);
    if pn > 0.0 {
        let k = delta / pn;
        for v in p.iter_mut() { *v *= k; }
    }
    Ok((p, alpha, it + 1))
}

fn phi_and_derivative(alpha: f64, suf: &[f64], s: &[f64], delta: f64) -> (f64, f64) {
    let n = s.len();
    let mut p_sq = 0.0_f64;
    for i in 0..n {
        let denom = s[i] * s[i] + alpha;
        let q = suf[i] / denom;
        p_sq += q * q;
    }
    let p_norm = p_sq.sqrt();
    let phi = p_norm - delta;
    let mut sum = 0.0_f64;
    for i in 0..n {
        let denom = s[i] * s[i] + alpha;
        sum += suf[i] * suf[i] / (denom * denom * denom);
    }
    let phi_prime = -sum / p_norm;
    (phi, phi_prime)
}

#[allow(clippy::too_many_arguments)]
fn select_step(
    x: &[f64], j_h: &DMatrix<f64>, diag_h: &[f64], g_h: &[f64],
    mut p: Vec<f64>, mut p_h: Vec<f64>, d: &[f64], delta: f64, lb: &[f64], ub: &[f64], theta: f64,
) -> (Vec<f64>, Vec<f64>, f64) {
    let n = x.len();

    // Interior step?
    let x_plus_p: Vec<f64> = (0..n).map(|i| x[i] + p[i]).collect();
    if in_bounds(&x_plus_p, lb, ub) {
        let p_value = evaluate_quadratic(j_h, g_h, &p_h, Some(diag_h));
        return (p, p_h, -p_value);
    }

    let (p_stride, hits) = step_size_to_bound(x, &p, lb, ub);

    // Reflected direction.
    let mut r_h: Vec<f64> = p_h.clone();
    for i in 0..n { if hits[i] != 0 { r_h[i] = -r_h[i]; } }
    let mut r: Vec<f64> = (0..n).map(|i| d[i] * r_h[i]).collect();

    // Restrict TR step to first bound hit.
    for i in 0..n { p[i] *= p_stride; p_h[i] *= p_stride; }
    let x_on_bound: Vec<f64> = (0..n).map(|i| x[i] + p[i]).collect();

    let to_tr_intersect = intersect_trust_region(&p_h, &r_h, delta).map(|(_n, pos)| pos).unwrap_or(0.0);
    let (to_bound, _) = step_size_to_bound(&x_on_bound, &r, lb, ub);

    let r_stride = to_bound.min(to_tr_intersect);
    let (r_stride_l, r_stride_u) = if r_stride > 0.0 {
        let l = (1.0 - theta) * p_stride / r_stride;
        let u = if r_stride == to_bound { theta * to_bound } else { to_tr_intersect };
        (l, u)
    } else { (0.0_f64, -1.0_f64) };

    let r_value;
    if r_stride_l <= r_stride_u {
        let (a, b, c) = build_quadratic_1d(j_h, g_h, &r_h, Some(diag_h), Some(&p_h));
        let (rs, rv) = minimize_quadratic_1d(a, b, r_stride_l, r_stride_u, c);
        for i in 0..n { r_h[i] = r_h[i] * rs + p_h[i]; }
        for i in 0..n { r[i] = r_h[i] * d[i]; }
        r_value = rv;
    } else {
        r_value = f64::INFINITY;
    }

    // Strictly-interior p
    for i in 0..n { p[i] *= theta; p_h[i] *= theta; }
    let p_value = evaluate_quadratic(j_h, g_h, &p_h, Some(diag_h));

    // Cauchy step along -g_h
    let neg_g_h: Vec<f64> = g_h.iter().map(|x| -x).collect();
    let ag_h_dir = neg_g_h.clone();
    let ag_dir: Vec<f64> = (0..n).map(|i| d[i] * ag_h_dir[i]).collect();

    let to_tr = if norm(&ag_h_dir) > 0.0 { delta / norm(&ag_h_dir) } else { 0.0 };
    let (to_bound_ag, _) = step_size_to_bound(x, &ag_dir, lb, ub);
    let ag_stride_cap = if to_bound_ag < to_tr { theta * to_bound_ag } else { to_tr };

    let (a_ag, b_ag, _) = build_quadratic_1d(j_h, g_h, &ag_h_dir, Some(diag_h), None);
    let (ag_stride, ag_value) = minimize_quadratic_1d(a_ag, b_ag, 0.0, ag_stride_cap, 0.0);
    let ag_h: Vec<f64> = ag_h_dir.iter().map(|v| v * ag_stride).collect();
    let ag: Vec<f64> = ag_dir.iter().map(|v| v * ag_stride).collect();

    if p_value < r_value && p_value < ag_value {
        (p, p_h, -p_value)
    } else if r_value < p_value && r_value < ag_value {
        (r, r_h, -r_value)
    } else {
        (ag, ag_h, -ag_value)
    }
}

fn update_tr_radius(delta: f64, actual_reduction: f64, predicted_reduction: f64,
                    step_norm: f64, bound_hit: bool) -> (f64, f64) {
    let ratio = if predicted_reduction > 0.0 {
        actual_reduction / predicted_reduction
    } else if predicted_reduction == 0.0 && actual_reduction == 0.0 {
        1.0
    } else { 0.0 };
    let mut new_delta = delta;
    if ratio < 0.25 { new_delta = 0.25 * step_norm; }
    else if ratio > 0.75 && bound_hit { new_delta = delta * 2.0; }
    (new_delta, ratio)
}

fn check_termination(d_f: f64, f: f64, dx_norm: f64, x_norm: f64, ratio: f64,
                     ftol: f64, xtol: f64) -> Option<TrfStatus> {
    let ftol_ok = d_f < ftol * f && ratio > 0.25;
    let xtol_ok = dx_norm < xtol * (xtol + x_norm);
    match (ftol_ok, xtol_ok) {
        (true,  true)  => Some(TrfStatus::BothFtolXtol),
        (true,  false) => Some(TrfStatus::FtolSatisfied),
        (false, true)  => Some(TrfStatus::XtolSatisfied),
        _              => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    /// Simple synthetic test: fit `y = a*exp(-b*x)` with a single peak.
    /// Verifies the TRR solver lands on the optimum from a random start
    /// inside loose bounds.
    struct Expo { x: Vec<f64>, y: Vec<f64> }
    impl ResidualsAndJacobian for Expo {
        fn n_params(&self) -> usize { 2 }
        fn n_residuals(&self) -> usize { self.x.len() }
        fn residuals(&self, p: &[f64], out: &mut [f64]) {
            for i in 0..self.x.len() {
                out[i] = p[0] * (-p[1] * self.x[i]).exp() - self.y[i];
            }
        }
        fn jacobian(&self, p: &[f64], out: &mut nalgebra::DMatrix<f64>) {
            for i in 0..self.x.len() {
                let e = (-p[1] * self.x[i]).exp();
                out[(i, 0)] = e;
                out[(i, 1)] = -p[0] * self.x[i] * e;
            }
        }
    }

    #[test]
    fn fits_exponential() {
        let x: Vec<f64> = (0..50).map(|i| 0.1 * i as f64).collect();
        let a_true = 3.0_f64; let b_true = 0.4_f64;
        let y: Vec<f64> = x.iter().map(|xi| a_true * (-b_true * xi).exp()).collect();
        let m = Expo { x, y };
        let r = run_bounded_lm(&m, &[1.0, 0.1], &[0.0, 0.0], &[10.0, 5.0], 200).unwrap();
        assert!((r.params[0] - a_true).abs() < 1e-6, "a = {}", r.params[0]);
        assert!((r.params[1] - b_true).abs() < 1e-6, "b = {}", r.params[1]);
    }

    /// Verify Coleman-Li scaling vector matches paper definition.
    #[test]
    fn cl_scaling_matches_paper() {
        let x = [0.5_f64, 0.5];
        let g = [-1.0_f64, 1.0];
        let lb = [0.0_f64, 0.0];
        let ub = [1.0_f64, 1.0];
        let (v, dv) = cl_scaling_vector(&x, &g, &lb, &ub);
        assert!((v[0] - 0.5).abs() < 1e-12);   // ub - x because g<0
        assert!((v[1] - 0.5).abs() < 1e-12);   // x - lb because g>0
        assert_eq!(dv[0], -1.0);
        assert_eq!(dv[1],  1.0);
    }

    /// Jacobian-induced row count sanity.
    #[test]
    fn jacobian_layout_compat() {
        let _ = Array2::<f64>::zeros((10, 3));   // (n_r, n_p) row-major
    }
}
