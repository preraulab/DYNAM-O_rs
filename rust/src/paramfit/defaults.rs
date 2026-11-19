//! Default bounds on the shape parameters of a fitted mode.
//!
//! These are the bounds the parametric-basis fits use when the caller does
//! not set its own. They live here so that every caller (the desktop app,
//! the command-line tool, and the MATLAB and Python wrappers) reads the same
//! numbers. MATLAB's `param_basis_opts.m` returns the same values in its
//! `LB_default` and `UB_default` rows, and
//! `tests/paramfit_defaults_matlab_parity.rs` holds the two to each other
//! through a MATLAB-generated fixture.
//!
//! Only the three shape parameters have fixed defaults. The bounds on the
//! amplitude and on the two centers depend on the histogram being fitted,
//! so the caller derives them from the data.

use std::f64::consts::{PI, SQRT_2};

/// Lower and upper bounds on the three shape parameters of one mode.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ModeShapeBounds {
    /// Lower bound on the frequency width `fstd`, a standard deviation in Hz.
    pub fstd_lower: f64,
    /// Upper bound on the frequency width `fstd`, a standard deviation in Hz.
    pub fstd_upper: f64,
    /// Lower bound on the width along the second axis `pstd`. For the
    /// SO-power fit it is a standard deviation in SO-power units. For the
    /// SO-phase fit it is `recikappa = 1 / sqrt(kappa)` in radians.
    pub pstd_lower: f64,
    /// Upper bound on `pstd`, in the same units as `pstd_lower`.
    pub pstd_upper: f64,
    /// Lower bound on the rotation or skew parameter `theta`.
    pub theta_lower: f64,
    /// Upper bound on the rotation or skew parameter `theta`.
    pub theta_upper: f64,
}

/// Default shape bounds for the SO-power fit (rotated Gaussian modes).
///
/// The widths are standard deviations. MATLAB writes them as a width divided
/// by sqrt(2), and the same form is kept here so the values agree to the
/// last bit.
pub fn default_power_mode_shape_bounds() -> ModeShapeBounds {
    ModeShapeBounds {
        fstd_lower: 0.1 / SQRT_2,
        fstd_upper: 2.5 / SQRT_2,
        pstd_lower: 2.5 / SQRT_2,
        pstd_upper: 30.0 / SQRT_2,
        theta_lower: -0.03,
        theta_upper: 0.03,
    }
}

/// Default shape bounds for the SO-phase fit (von Mises by Gaussian modes).
///
/// The frequency width is a standard deviation between 1/sqrt(2) and
/// sqrt(7.5) Hz. The phase width is `recikappa`, between pi/5 and 2*pi rad.
pub fn default_phase_mode_shape_bounds() -> ModeShapeBounds {
    ModeShapeBounds {
        fstd_lower: 1.0 / SQRT_2,
        fstd_upper: 7.5_f64.sqrt(),
        pstd_lower: PI / 5.0,
        pstd_upper: 2.0 * PI,
        theta_lower: -PI / 3.0,
        theta_upper: PI / 3.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every default range is non-empty, so a fit can start inside it.
    #[test]
    fn default_ranges_are_ordered() {
        let both = [
            default_power_mode_shape_bounds(),
            default_phase_mode_shape_bounds(),
        ];
        for bounds in both {
            assert!(bounds.fstd_lower < bounds.fstd_upper);
            assert!(bounds.pstd_lower < bounds.pstd_upper);
            assert!(bounds.theta_lower < bounds.theta_upper);
        }
    }

    /// The values MATLAB's `param_basis_opts.m` lists, written the way that
    /// file writes them.
    #[test]
    fn defaults_match_the_matlab_table() {
        let power = default_power_mode_shape_bounds();
        assert_eq!(power.fstd_lower, 0.1 / 2.0_f64.sqrt());
        assert_eq!(power.fstd_upper, 2.5 / 2.0_f64.sqrt());
        assert_eq!(power.pstd_lower, 2.5 / 2.0_f64.sqrt());
        assert_eq!(power.pstd_upper, 30.0 / 2.0_f64.sqrt());
        assert_eq!(power.theta_lower, -0.03);
        assert_eq!(power.theta_upper, 0.03);

        let phase = default_phase_mode_shape_bounds();
        assert_eq!(phase.fstd_lower, 1.0 / 2.0_f64.sqrt());
        assert_eq!(phase.fstd_upper, 7.5_f64.sqrt());
        assert_eq!(phase.pstd_lower, PI / 5.0);
        assert_eq!(phase.pstd_upper, 2.0 * PI);
        assert_eq!(phase.theta_lower, -PI / 3.0);
        assert_eq!(phase.theta_upper, PI / 3.0);
    }
}
