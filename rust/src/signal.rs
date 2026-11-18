//! Signal-processing primitives: sosfiltfilt, hilbert, unwrap, movmean.
//!
//! Targets:
//!   - MATLAB filtfilt(sos, 1, x) (section-by-section forward-backward)
//!   - scipy.signal.hilbert(x, axis=-1)
//!   - numpy.unwrap(p, discont=π)
//!   - numpy running-window mean (centered, partial-window at edges, matches
//!     MATLAB `movmean`).

use realfft::num_complex::Complex;
use realfft::RealFftPlanner;

// ---------------------------------------------------------------------------
// sosfiltfilt
// ---------------------------------------------------------------------------

/// Apply a Direct-Form-II-Transposed biquad section in place on `x`, using
/// initial state `z[0..2]`. Returns the final state.
///
/// Section layout: `sec = [b0, b1, b2, a0, a1, a2]`, with a0 typically 1.
/// We normalize by a0 to match scipy.
#[inline]
fn sosfilt_single_section(sec: &[f64; 6], x: &mut [f64], mut z0: f64, mut z1: f64) -> (f64, f64) {
    let a0 = sec[3];
    let b0 = sec[0] / a0;
    let b1 = sec[1] / a0;
    let b2 = sec[2] / a0;
    let a1 = sec[4] / a0;
    let a2 = sec[5] / a0;
    for v in x.iter_mut() {
        let xn = *v;
        let yn = b0 * xn + z0;
        z0 = b1 * xn - a1 * yn + z1;
        z1 = b2 * xn - a2 * yn;
        *v = yn;
    }
    (z0, z1)
}

/// Steady-state initial conditions for one section, for a unit step input
/// (MATLAB `filtfilt.m`, `getCoeffsAndInitialConditions`, the same values
/// as `scipy.signal.lfilter_zi`). Multiplying them by the first input
/// sample starts the filter as if that value had been held forever, which
/// removes the start-up transient for a constant offset.
fn section_initial_conditions(sec: &[f64; 6]) -> [f64; 2] {
    let a0 = sec[3];
    let b0 = sec[0] / a0;
    let b1 = sec[1] / a0;
    let b2 = sec[2] / a0;
    let a1 = sec[4] / a0;
    let a2 = sec[5] / a0;
    // Solve
    //   (1 + a1)*z0 - z1 = b1 - a1*b0
    //        a2*z0 + z1 = b2 - a2*b0
    // by adding the two rows.
    let rhs0 = b1 - a1 * b0;
    let rhs1 = b2 - a2 * b0;
    let z0 = (rhs0 + rhs1) / (1.0 + a1 + a2);
    let z1 = rhs1 - a2 * z0;
    [z0, z1]
}

/// Order of one section as MATLAB `filtfilt.m` `findFilterOrder` counts it:
/// the position of the last nonzero coefficient, in the numerator or the
/// denominator whichever is longer, minus one.
fn section_order(sec: &[f64; 6]) -> usize {
    let numerator = [sec[0], sec[1], sec[2]];
    let denominator = [sec[3], sec[4], sec[5]];
    let mut numerator_length = 0usize;
    let mut denominator_length = 0usize;
    for k in 0..3 {
        if numerator[k] != 0.0 {
            numerator_length = k + 1;
        }
        if denominator[k] != 0.0 {
            denominator_length = k + 1;
        }
    }
    let longest = numerator_length.max(denominator_length);
    longest.saturating_sub(1)
}

/// Length of the reflected extension at each end, MATLAB `filtfilt.m`:
/// three times the total filter order (6 samples per full biquad), at
/// least 1.
fn edge_transient_length(sos: &[[f64; 6]]) -> usize {
    let mut total_order = 0usize;
    for sec in sos {
        total_order += section_order(sec);
    }
    (3 * total_order).max(1)
}

/// Zero-phase filtering with second-order sections, following MATLAB
/// `filtfilt(sos, 1, x)` step for step.
///
/// The sections are applied one at a time, and each one is run forward
/// and backward before the next starts:
///   1. Extend the current signal at both ends by odd reflection,
///      `2*y[0] - y[L..1]` in front and `2*y[n-1] - y[n-2..n-1-L]` behind,
///      with `L = edge_transient_length(sos)` (the same for every section).
///   2. Filter forward, starting from the section's initial conditions
///      times the first extended sample.
///   3. Reverse, filter again from the initial conditions times the first
///      sample of the reversed signal, reverse back.
///   4. Drop the extensions. The result is the input to the next section.
///
/// Running all sections forward and then all backward on a single
/// extension gives the same answer far from the ends but a different one
/// near them, where each procedure's start-up transient lives. For a
/// narrow low-frequency band those transients last tens of seconds.
///
/// MATLAB refuses a signal no longer than `L`. Here the extension is
/// shortened to `n - 1` samples in that case.
pub fn sosfiltfilt(sos: &[[f64; 6]], x: &[f64]) -> Vec<f64> {
    let n = x.len();
    if n == 0 {
        return Vec::new();
    }
    let mut pad = edge_transient_length(sos);
    if pad >= n {
        pad = n - 1;
    }

    let mut current: Vec<f64> = x.to_vec();
    let mut extended: Vec<f64> = Vec::with_capacity(n + 2 * pad);
    for sec in sos {
        let zi = section_initial_conditions(sec);

        // 1. Odd reflection at both ends.
        extended.clear();
        for i in (1..=pad).rev() {
            extended.push(2.0 * current[0] - current[i]);
        }
        extended.extend_from_slice(&current);
        for i in 1..=pad {
            extended.push(2.0 * current[n - 1] - current[n - 1 - i]);
        }

        // 2. Forward pass.
        let first = extended[0];
        sosfilt_single_section(sec, &mut extended, zi[0] * first, zi[1] * first);

        // 3. Backward pass.
        extended.reverse();
        let first_reversed = extended[0];
        sosfilt_single_section(sec, &mut extended, zi[0] * first_reversed, zi[1] * first_reversed);
        extended.reverse();

        // 4. Keep the central part.
        current.copy_from_slice(&extended[pad..pad + n]);
    }
    current
}

// ---------------------------------------------------------------------------
// hilbert  (analytic signal)
// ---------------------------------------------------------------------------

/// FFT-based analytic signal. Returns (real, imag) parts in two separate
/// Vecs so the pyo3 boundary doesn't need complex types.
pub fn hilbert(x: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = x.len();
    if n == 0 {
        return (Vec::new(), Vec::new());
    }
    // Round up to power-of-two FFT for speed? scipy uses n directly by default.
    // Match scipy: use length = n.
    let mut planner = RealFftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(n);
    let ifft = planner.plan_fft_inverse(n);

    let mut in_buf: Vec<f64> = x.to_vec();
    let mut spec: Vec<Complex<f64>> = fft.make_output_vec();
    let mut scratch: Vec<Complex<f64>> = fft.make_scratch_vec();
    fft.process_with_scratch(&mut in_buf, &mut spec, &mut scratch).unwrap();

    // One-sided → two-sided analytic:
    // Build full-length complex spectrum with scaled positive freqs.
    let mut full: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); n];
    full[0] = spec[0]; // DC
    let nyq = if n % 2 == 0 { Some(n / 2) } else { None };
    for k in 1..spec.len() {
        if Some(k) == nyq {
            full[k] = spec[k]; // Nyquist bin unscaled
        } else {
            full[k] = spec[k] * 2.0; // positive freqs ×2
        }
    }
    // Negative freqs are already zero → analytic signal construction.

    // Inverse FFT: need a Complex→Real? No, analytic is complex. Use full
    // complex FFT. realfft's inverse returns real only, so we need a
    // complex FFT here. Use rustfft directly.
    use rustfft::FftPlanner;
    let mut cplanner = FftPlanner::<f64>::new();
    let cfft_inv = cplanner.plan_fft_inverse(n);
    // We'll do an in-place inverse.
    let mut buf = full.clone();
    cfft_inv.process(&mut buf);
    // Normalize (rustfft is unnormalized).
    let inv_n = 1.0 / n as f64;
    let mut re = Vec::with_capacity(n);
    let mut im = Vec::with_capacity(n);
    for c in &buf {
        re.push(c.re * inv_n);
        im.push(c.im * inv_n);
    }
    let _ = ifft; // unused; we used rustfft's complex IFFT for the analytic signal
    (re, im)
}

// ---------------------------------------------------------------------------
// unwrap (numpy.unwrap)
// ---------------------------------------------------------------------------

/// numpy.unwrap with default discont = π.
pub fn unwrap(p: &[f64], discont: f64) -> Vec<f64> {
    let n = p.len();
    let mut out = Vec::with_capacity(n);
    if n == 0 {
        return out;
    }
    let period = 2.0 * std::f64::consts::PI;
    // Running cumulative offset (multiples of period added to subsequent samples).
    let mut offset = 0.0f64;
    out.push(p[0]);
    for i in 1..n {
        // delta uses ORIGINAL-adjacent samples (p[i] - p[i-1]). The jump
        // location is determined by raw input, consistent with numpy.unwrap.
        let delta = p[i] - p[i - 1];
        if delta > discont {
            let cycles = ((delta + std::f64::consts::PI) / period).floor();
            offset -= cycles * period;
        } else if delta < -discont {
            let cycles = ((-delta + std::f64::consts::PI) / period).floor();
            offset += cycles * period;
        }
        out.push(p[i] + offset);
    }
    out
}

// ---------------------------------------------------------------------------
// movmean  (centered window, partial at edges, MATLAB `movmean`)
// ---------------------------------------------------------------------------

/// How many samples a moving window of length `win` reaches to the left and
/// to the right of its center sample, as MATLAB `movmean` and `movmedian`
/// place it. An even-length window sits one sample toward the past, covering
/// samples `[i - win/2, i + win/2 - 1]`. An odd-length window is symmetric.
pub(crate) fn matlab_window_halves(win: usize) -> (usize, usize) {
    let half_left = win / 2;
    let half_right = (win - 1) / 2;
    (half_left, half_right)
}

/// Running mean with centered window of length `win`. Edges use partial
/// windows (as MATLAB `movmean(x, win)`).
pub fn movmean(x: &[f64], win: usize) -> Vec<f64> {
    let n = x.len();
    if win <= 1 || n == 0 {
        return x.to_vec();
    }
    let (half_l, half_r) = matlab_window_halves(win);
    // Prefix sum for O(N) window query.
    let mut csum = vec![0.0f64; n + 1];
    for i in 0..n {
        csum[i + 1] = csum[i] + x[i];
    }
    let mut out = vec![0.0f64; n];
    for i in 0..n {
        let a = i.saturating_sub(half_l);
        let b = (i + half_r + 1).min(n);
        let sum = csum[b] - csum[a];
        out[i] = sum / (b - a) as f64;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unwrap_simple() {
        // Jumps at indices 2, 5
        let p = vec![0.0, 1.0, -3.0, -2.5, -2.0, 3.0, 3.5];
        let out = unwrap(&p, std::f64::consts::PI);
        // Diffs: 1,-4,0.5,0.5,5,0.5 → wraps at 2 (add 2π) and 5 (subtract 2π)
        // Expected unwrap: 0, 1, -3+2π, -2.5+2π, -2+2π, 3, 3.5 (adjusted)
        // Just sanity: diff(unwrap) magnitudes all ≤ π.
        for i in 1..out.len() {
            assert!((out[i] - out[i - 1]).abs() <= std::f64::consts::PI + 1e-12);
        }
    }

    #[test]
    fn movmean_even_window_matches_matlab() {
        // MATLAB R2025b: movmean([1 2 3 4 5 6], 4) -> 1.5 2 2.5 3.5 4.5 5
        let out = movmean(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 4);
        assert_eq!(out, vec![1.5, 2.0, 2.5, 3.5, 4.5, 5.0]);
    }

    #[test]
    fn movmean_uniform() {
        let x = vec![1.0; 10];
        let out = movmean(&x, 3);
        for &v in &out {
            assert!((v - 1.0).abs() < 1e-12);
        }
    }

    /// Twenty samples used by the MATLAB-referenced filter tests.
    fn filter_test_signal() -> Vec<f64> {
        vec![
            1.0, 3.0, -2.0, 4.0, 0.5, -1.0, 2.0, 6.0, -3.0, 1.5, 0.0, 2.5, -4.0, 1.0, 3.0, -0.5,
            2.0, 1.0, -1.0, 0.25,
        ]
    }

    #[test]
    fn sosfiltfilt_matches_matlab_for_two_sections() {
        // MATLAB R2025b: filtfilt(sos, 1, x) for these two sections. The
        // signal is short, so every sample is inside the edge transient
        // and the per-section forward-backward order shows in each value.
        let sos = [
            [0.2, 0.3, 0.1, 1.0, -0.5, 0.25],
            [0.5, -0.4, 0.3, 1.0, 0.2, 0.1],
        ];
        let expected = [
            0.060615437759993715, 0.063128012819028773, 0.039762767348181891,
            0.11004233829414317, 0.08006004681626884, 0.0054990525849684668,
            0.11609159617429311, 0.14709455538778837, 0.0052953103829598211,
            -0.0026329053627278526, 0.085746897291231694, 0.03638896700719628,
            -0.049659639998364216, 0.028800304687914183, 0.082676527515276182,
            0.025948909084606522, 0.049749356818104731, 0.059157810678188164,
            0.0017809135577381355, 0.015171566031306094,
        ];
        assert_eq!(edge_transient_length(&sos), 12);
        let y = sosfiltfilt(&sos, &filter_test_signal());
        for i in 0..expected.len() {
            assert!((y[i] - expected[i]).abs() < 1e-13, "sample {}: {} vs {}", i, y[i], expected[i]);
        }
    }

    #[test]
    fn sosfiltfilt_counts_a_first_order_section_as_order_one() {
        // A full biquad then a first-order section (trailing zeros in both
        // numerator and denominator): total order 3, so MATLAB extends by
        // 9 samples, not 12. Values from MATLAB R2025b filtfilt(sos, 1, x).
        let sos = [
            [0.2, 0.3, 0.1, 1.0, -0.5, 0.25],
            [0.3, 0.2, 0.0, 1.0, -0.6, 0.0],
        ];
        let expected = [
            0.99916061642573784, 1.0433005311514916, 1.0739746286132892, 1.0908942291621668,
            1.1206120181854424, 1.1959112366784157, 1.2673704087192197, 1.2096887862005048,
            0.99124705720011153, 0.72012699822331894, 0.49238591942932608, 0.34369776312763795,
            0.32595877488488612, 0.46240264566135808, 0.65336216826290183, 0.75227897726200643,
            0.7021333489271242, 0.55140599751589181, 0.38705556014010317, 0.25306284049141103,
        ];
        assert_eq!(edge_transient_length(&sos), 9);
        let y = sosfiltfilt(&sos, &filter_test_signal());
        for i in 0..expected.len() {
            assert!((y[i] - expected[i]).abs() < 1e-13, "sample {}: {} vs {}", i, y[i], expected[i]);
        }
    }

    #[test]
    fn sosfiltfilt_passes_constant() {
        // Single-section identity filter: b=[1,0,0], a=[1,0,0]. sosfiltfilt
        // on a constant signal should return the same constant.
        let sos: Vec<[f64; 6]> = vec![[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]];
        let x = vec![3.5; 50];
        let y = sosfiltfilt(&sos, &x);
        for &v in &y {
            assert!((v - 3.5).abs() < 1e-10);
        }
    }

    #[test]
    fn hilbert_shape() {
        let n = 128;
        let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();
        let (re, im) = hilbert(&x);
        assert_eq!(re.len(), n);
        assert_eq!(im.len(), n);
        // Real part should approximately equal the input.
        for i in 0..n {
            assert!((re[i] - x[i]).abs() < 1e-8);
        }
    }
}
