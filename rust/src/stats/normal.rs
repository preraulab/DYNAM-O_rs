//! Standard-normal tail probability, accurate enough to compare against
//! MATLAB's `normcdf` in the deep tail.
//!
//! The obvious shortcut — an Abramowitz & Stegun polynomial for Φ — is
//! accurate to ~7.5e-8 in ABSOLUTE terms, which is fine for a threshold
//! test at q = 0.1 and useless for anything else: a p-value of 2e-16 comes
//! back as noise around 1e-8. Since these p-values are compared against
//! MATLAB's for parity, the tail has to be right in RELATIVE terms too, so
//! `erfc` is computed by series below 2 and by continued fraction above it,
//! both to near machine precision.

/// erf(x) by its Maclaurin series. Converges quickly for |x| <= 2; the
/// term ratio is x²/n, so ~30 terms suffice at the top of that range.
fn erf_series(x: f64) -> f64 {
    let mut term = x;
    let mut sum = x;
    let x2 = x * x;
    for n in 1..200 {
        term *= -x2 / n as f64;
        let add = term / (2 * n + 1) as f64;
        sum += add;
        if add.abs() <= f64::EPSILON * sum.abs() {
            break;
        }
    }
    sum * 2.0 / std::f64::consts::PI.sqrt()
}

/// erfc(x) for x > 2 by the classical continued fraction
///
///   erfc(x) = exp(-x²)/√π · 1/(x + (1/2)/(x + 1/(x + (3/2)/(x + ...))))
///
/// evaluated with the modified Lentz algorithm.
fn erfc_cf(x: f64) -> f64 {
    const TINY: f64 = 1e-300;
    let mut f = TINY;
    let mut c = f;
    let mut d = 0.0f64;
    for i in 0..300 {
        // a₁ = 1, aᵢ = (i-1)/2 for i >= 2; every bᵢ = x.
        let a = if i == 0 { 1.0 } else { i as f64 / 2.0 };
        d = x + a * d;
        if d.abs() < TINY {
            d = TINY;
        }
        c = x + a / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        let delta = c * d;
        f *= delta;
        if (delta - 1.0).abs() < f64::EPSILON {
            break;
        }
    }
    (-x * x).exp() / std::f64::consts::PI.sqrt() * f
}

/// Complementary error function, full double precision.
pub fn erfc(x: f64) -> f64 {
    if x < 0.0 {
        return 2.0 - erfc(-x);
    }
    if x < 2.0 {
        1.0 - erf_series(x)
    } else {
        erfc_cf(x)
    }
}

/// Two-sided standard-normal tail: P(|Z| >= |z|) = erfc(|z|/√2).
///
/// Written as a single erfc rather than `2·(1 − Φ(|z|))` so the deep tail
/// never goes through a subtraction from 1.
pub fn two_sided_p(z: f64) -> f64 {
    if !z.is_finite() {
        return f64::NAN;
    }
    erfc(z.abs() / std::f64::consts::SQRT_2).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference values from MATLAB R2025a `erfc` (format long).
    #[test]
    fn erfc_matches_matlab_across_the_range() {
        let cases: [(f64, f64); 9] = [
            (0.0, 1.00000000000000000e0),
            (0.5, 4.79500122186953481e-1),
            (1.0, 1.57299207050285134e-1),
            // Straddles the series/continued-fraction handover at x = 2.
            (1.9999, 4.67980209297060804e-3),
            (2.0, 4.67773498104726623e-3),
            (3.0, 2.20904969985854378e-5),
            (5.0, 1.53745979442803514e-12),
            (8.0, 1.12242971729829264e-29),
            (-1.5, 1.96610514647531076e0),
        ];
        for (x, want) in cases {
            let got = erfc(x);
            let rel = (got - want).abs() / want.abs();
            assert!(rel < 1e-13, "erfc({}) = {:e}, want {:e} (rel {:e})", x, got, want, rel);
        }
    }

    /// The whole point of not using an A&S polynomial: relative accuracy
    /// where the p-value is small.
    #[test]
    fn deep_tail_stays_relatively_accurate() {
        // MATLAB R2025a: 2*normcdf(-8)
        let p = two_sided_p(8.0);
        let want = 1.24419211485436387e-15;
        assert!((p - want).abs() / want < 1e-12, "got {:e}, want {:e}", p, want);
    }

    #[test]
    fn two_sided_is_symmetric_and_bounded() {
        assert_eq!(two_sided_p(1.3), two_sided_p(-1.3));
        assert!((two_sided_p(0.0) - 1.0).abs() < 1e-15);
        assert!(two_sided_p(f64::NAN).is_nan());
    }
}
