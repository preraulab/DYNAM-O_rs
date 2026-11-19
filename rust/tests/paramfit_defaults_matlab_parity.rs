//! The kernel's default fit bounds against MATLAB's.
//!
//! `tests/fixtures/parity_paramfit_default_bounds.csv` holds the default
//! bounds on the mode shape parameters as MATLAB's `param_basis_opts`
//! returns them. The kernel's `paramfit::defaults` must hold exactly the
//! same doubles, so the comparison is bit for bit.
//!
//! Regenerate the fixture with `scripts/gen_matlab_paramfit_defaults.m`
//! when the MATLAB defaults change on purpose. A diff in that file is a
//! decision to review.

use dynamo_rs::paramfit::defaults::{
    default_phase_mode_shape_bounds, default_power_mode_shape_bounds, ModeShapeBounds,
};

/// The six bounds in the column order of the fixture.
fn as_fixture_row(bounds: &ModeShapeBounds) -> [f64; 6] {
    [
        bounds.fstd_lower,
        bounds.fstd_upper,
        bounds.pstd_lower,
        bounds.pstd_upper,
        bounds.theta_lower,
        bounds.theta_upper,
    ]
}

#[test]
fn default_shape_bounds_match_matlab() {
    let fixture = include_str!("fixtures/parity_paramfit_default_bounds.csv");
    let column_names = ["fstd_lower", "fstd_upper", "pstd_lower", "pstd_upper", "theta_lower", "theta_upper"];

    let mut fits_checked = 0;
    for line in fixture.lines().skip(1) {
        let fields: Vec<&str> = line.split(',').collect();
        assert_eq!(fields.len(), 7, "unexpected fixture row: {line}");
        let fit = fields[0];
        let kernel_bounds = match fit {
            "power" => default_power_mode_shape_bounds(),
            "phase" => default_phase_mode_shape_bounds(),
            other => panic!("unknown fit in fixture: {other}"),
        };
        let kernel_row = as_fixture_row(&kernel_bounds);
        for column in 0..6 {
            let matlab_value: f64 = fields[column + 1].parse().unwrap();
            assert_eq!(
                kernel_row[column].to_bits(),
                matlab_value.to_bits(),
                "{fit} {}: kernel {} differs from MATLAB {}",
                column_names[column],
                kernel_row[column],
                matlab_value,
            );
        }
        fits_checked += 1;
    }
    assert_eq!(fits_checked, 2, "the fixture must cover the power and phase fits");
}
