//! Standalone test that replays the same synthesized SOPH as
//! `validate_rotgauss_fit_mex.m`, runs the Rust fit, and prints diagnostics
//! independent of MATLAB.

use dynamo_rs::paramfit::rot_gauss::fit_rotgauss;
use ndarray::Array2;

#[test]
fn two_mode_rotgauss_recovers_synthesized_truth() {
    let nx = 60usize;       // SOpower_bins
    let ny = 151usize;      // freq_bins
    let x: Vec<f64> = (0..nx).map(|i| -3.9 + (23.1 - (-3.9)) * i as f64 / (nx - 1) as f64).collect();
    let y: Vec<f64> = (0..ny).map(|j| 0.2 * j as f64).collect();

    // Synthesize: 4.5*exp(-((X-10)/8)² - ((Y-12)/2.5)²)
    //           + 3.0*exp(-((X-6)/6)² - ((Y-2.5)/1.3)²)
    //           + 0.05*Y + 0.1 + tiny noise
    let mut soph = Array2::<f64>::zeros((ny, nx));
    for iy in 0..ny {
        for ix in 0..nx {
            let xv = x[ix];
            let yv = y[iy];
            let p1 = 4.5 * ((-((xv-10.0)/8.0).powi(2) - ((yv-12.0)/2.5).powi(2)).exp());
            let p2 = 3.0 * ((-((xv-6.0)/6.0).powi(2)  - ((yv-2.5)/1.3).powi(2)).exp());
            let n = ((iy as f64 * 31.0 + ix as f64 * 17.0).sin() * 0.05).abs();
            soph[[iy, ix]] = p1 + p2 + 0.05 * yv + 0.1 + n;
        }
    }

    // Seeds as in validate_rotgauss_fit_mex.m
    let b0 = Array2::from_shape_vec((2, 6), vec![
        4.0, 11.0, 2.0, 10.0, 8.0, 0.0,
        3.0,  3.0, 1.5,  6.0, 6.0, 0.0,
    ]).unwrap();
    let lb = Array2::from_shape_vec((2, 6), vec![
        0.1,  0.0, 0.1, -5.0, 2.5, -std::f64::consts::PI/20.0,
        0.1,  0.0, 0.1, -5.0, 2.5, -std::f64::consts::PI/20.0,
    ]).unwrap();
    let ub = Array2::from_shape_vec((2, 6), vec![
        45.0, 25.0, 5.0, 25.0, 30.0, std::f64::consts::PI/20.0,
        45.0, 25.0, 5.0, 25.0, 30.0, std::f64::consts::PI/20.0,
    ]).unwrap();

    // Background seed: prctile(SOPH(:), 5) for zzz, 0 for slope coefs.
    let mut all: Vec<f64> = soph.iter().copied().collect();
    all.sort_by(|a,b| a.partial_cmp(b).unwrap());
    let idx = ((all.len()-1) as f64 * 0.05).floor() as usize;
    let bg0 = [0.0, 0.0, all[idx]];
    let mxv = all.last().copied().unwrap();
    let bg_lo = [-0.1, -0.1, 0.0];
    let bg_hi = [0.1, 0.1, mxv];

    let out = fit_rotgauss(
        soph.view(), &x, &y, b0.view(), lb.view(), ub.view(),
        bg0, bg_lo, bg_hi, 5000,
    ).expect("fit");

    eprintln!("---- two_mode_rotgauss recovery ----");
    eprintln!("seed zzz = {:.4}, max = {:.4}", bg0[2], mxv);
    for m in 0..2 {
        eprintln!("mode {}: amp={:.4}  fmean={:.4}  fstd={:.4}  pmean={:.4}  pstd={:.4}  theta={:.4}",
            m+1,
            out.params[[m,0]], out.params[[m,1]], out.params[[m,2]],
            out.params[[m,3]], out.params[[m,4]], out.params[[m,5]]);
    }
    eprintln!("background: xxx={:.5e}  yyy={:.5e}  zzz={:.5e}", out.background[0], out.background[1], out.background[2]);
    eprintln!("gof: sse={:.4} rsq={:.6} adjrsq={:.6} rmse={:.6} dfe={} iters={}",
        out.gof.sse, out.gof.rsquare, out.gof.adjrsquare, out.gof.rmse, out.gof.dfe, out.iters_used);
    eprintln!("expected truth: mode1=(amp=4.5, fmean=12, fstd=2.5, pmean=10, pstd=8, theta=0)");
    eprintln!("                mode2=(amp=3.0, fmean=2.5, fstd=1.3, pmean=6, pstd=6, theta=0)");

    assert!(out.gof.rsquare > 0.99, "rsq = {} - LM failed to converge", out.gof.rsquare);
}
