//! Signal-processing primitives: sosfiltfilt, hilbert, unwrap, movmean.
//!
//! Targets byte-level parity with:
//!   - scipy.signal.sosfiltfilt(sos, x, padtype='odd', padlen=None)
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

/// Forward lfilter across all sections. Mutates `x` in place. Initial state
/// per section defaults to zeros unless `zi` is given.
#[allow(dead_code)]
fn sosfilt(sos: &[[f64; 6]], x: &mut [f64], zi: Option<&[[f64; 2]]>) {
    for (i, sec) in sos.iter().enumerate() {
        let (z0, z1) = zi.map(|z| (z[i][0], z[i][1])).unwrap_or((0.0, 0.0));
        sosfilt_single_section(sec, x, z0, z1);
    }
}

/// Compute initial conditions scipy.signal.sosfilt_zi.
///
/// For each section, solves (I - A) * zi = B * sum(b)/sum(a) with
///   A = [[-a1, 1], [-a2, 0]],
///   B = [[b1 - a1*b0], [b2 - a2*b0]]
///
/// The per-section state is scaled by the cumulative DC gain of preceding
/// sections so that a steady-state input produces zero transient.
fn sosfilt_zi(sos: &[[f64; 6]]) -> Vec<[f64; 2]> {
    let mut out = Vec::with_capacity(sos.len());
    let mut scale = 1.0f64;
    for sec in sos {
        let a0 = sec[3];
        let b0 = sec[0] / a0;
        let b1 = sec[1] / a0;
        let b2 = sec[2] / a0;
        let a1 = sec[4] / a0;
        let a2 = sec[5] / a0;
        // I - A = [[1+a1, -1], [a2, 1]]
        // solve:
        //   (1+a1)*z0 - z1 = (b1 - a1*b0)
        //   a2*z0 + z1      = (b2 - a2*b0)
        // add: (1+a1+a2)*z0 = (b1 + b2 - a1*b0 - a2*b0)
        let det = 1.0 + a1 + a2;
        let rhs0 = b1 - a1 * b0;
        let rhs1 = b2 - a2 * b0;
        let z0 = (rhs0 + rhs1) / det;
        // row 2: a2*z0 + z1 = rhs1 → z1 = rhs1 - a2*z0
        let z1 = rhs1 - a2 * z0;
        out.push([scale * z0, scale * z1]);
        // DC gain of this section: (b0+b1+b2)/(1+a1+a2)
        let dc_gain = (b0 + b1 + b2) / det;
        scale *= dc_gain;
    }
    out
}

/// Default padlen chosen by scipy.signal.sosfiltfilt for SOS:
///   edge = max(0, ord_est - 1) * 2, where ord_est is approximated from
///   the number of sections. scipy uses ntaps = 2*len(sos) + 1; padlen = 3
///   * ntaps. We replicate that.
fn default_padlen_sos(n_sections: usize) -> usize {
    let ntaps = 2 * n_sections + 1;
    3 * ntaps
}

/// Apply scipy.signal.sosfiltfilt (`padtype='odd'`, default `padlen`).
///
/// Implementation follows scipy:
///   1. Build the odd-extension padded signal:
///        ext = [2*x[0] - x[padlen..0:-1], x, 2*x[-1] - x[-2:-padlen-1:-1]]
///   2. Compute zi = sosfilt_zi(sos).
///   3. Forward lfilter with initial state = zi * ext[0].
///   4. Reverse, forward with initial state = zi * reversed_ext[0].
///   5. Reverse again, trim padding.
pub fn sosfiltfilt(sos: &[[f64; 6]], x: &[f64]) -> Vec<f64> {
    let n = x.len();
    if n == 0 {
        return Vec::new();
    }
    let mut padlen = default_padlen_sos(sos.len());
    if padlen >= n {
        padlen = if n == 0 { 0 } else { n - 1 };
    }

    // Build odd-extension pad: y[-i] = 2*x[0] - x[i]; y[n+i] = 2*x[n-1] - x[n-1-i]
    let mut ext = Vec::with_capacity(n + 2 * padlen);
    for i in (1..=padlen).rev() {
        ext.push(2.0 * x[0] - x[i]);
    }
    ext.extend_from_slice(x);
    for i in 1..=padlen {
        ext.push(2.0 * x[n - 1] - x[n - 1 - i]);
    }

    let zi = sosfilt_zi(sos);

    // Forward pass scaled by ext[0].
    let mut y = ext.clone();
    let zi0_scaled: Vec<[f64; 2]> = zi.iter().map(|s| [s[0] * ext[0], s[1] * ext[0]]).collect();
    let zi0_flat: Vec<f64> = zi0_scaled.iter().flat_map(|s| [s[0], s[1]]).collect();
    apply_sosfilt_with_state(sos, &mut y, &zi0_flat);

    // Reverse, forward pass scaled by y_rev[0] = y[last].
    y.reverse();
    let scale_rev = y[0];
    let zi_scaled: Vec<f64> = zi
        .iter()
        .flat_map(|s| [s[0] * scale_rev, s[1] * scale_rev])
        .collect();
    apply_sosfilt_with_state(sos, &mut y, &zi_scaled);
    y.reverse();

    // Trim padding.
    y[padlen..padlen + n].to_vec()
}

/// Apply forward lfilter across all sections with the given initial states
/// (flat, per-section pairs).
fn apply_sosfilt_with_state(sos: &[[f64; 6]], x: &mut [f64], zi_flat: &[f64]) {
    for (i, sec) in sos.iter().enumerate() {
        let z0 = zi_flat[2 * i];
        let z1 = zi_flat[2 * i + 1];
        sosfilt_single_section(sec, x, z0, z1);
    }
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
        // delta uses ORIGINAL-adjacent samples (p[i] - p[i-1]); the jump
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
// movmean  (centered window, partial at edges — MATLAB `movmean`)
// ---------------------------------------------------------------------------

/// Running mean with centered window of length `win`. Edges use partial
/// windows (as MATLAB `movmean(x, win)`).
pub fn movmean(x: &[f64], win: usize) -> Vec<f64> {
    let n = x.len();
    if win <= 1 || n == 0 {
        return x.to_vec();
    }
    let half_l = (win - 1) / 2;
    let half_r = win / 2;
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
    fn movmean_uniform() {
        let x = vec![1.0; 10];
        let out = movmean(&x, 3);
        for &v in &out {
            assert!((v - 1.0).abs() < 1e-12);
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
