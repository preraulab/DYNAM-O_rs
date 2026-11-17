//! Hann-window peak-frequency refinement — port of pydynamo
//! `tfpeaks.refine.refine_peak_frequency`.
//!
//! Two stages:
//!   1. `hann_event_spectra`: batched FFT of windowed+detrended+tapered EEG
//!      segments, one per event. Returns one-sided PSD per event (F, N).
//!   2. `refine_from_spectra`: per-event, fit a not-a-knot cubic spline
//!      (MATLAB `interp1(..., 'spline')`) to the full PSD, evaluate on a
//!      dense grid inside the peak's bbox frequency range, take argmax →
//!      refined frequency. Reject peaks that sit within 1e-3 Hz of the
//!      bbox boundaries.

use ndarray::Array2;
use crate::parallel::*;
use realfft::RealFftPlanner;
use realfft::num_complex::Complex;

fn next_pow2(x: f64) -> usize {
    if x <= 1.0 {
        return 1;
    }
    1usize << (x.log2().ceil() as u32)
}

/// Batched Hann-windowed FFT of EEG data slices. For each `event_time[i]`,
/// take a `window_size`-second window centered on it, demean, apply
/// normalized Hann taper, zero-pad to `nfft = next_pow2(Fs / dsfreqs)`, run
/// rfft, convert to one-sided PSD.
///
/// Returns (spect (F_mask, N_events), sfreqs (F_mask,)) where F_mask is
/// the subset of one-sided freqs inside `freq_range`.
pub fn hann_event_spectra(
    data: &[f64],
    fs: f64,
    event_times: &[f64],
    t0: f64,
    freq_range: (f64, f64),
    window_size: f64,
    dsfreqs: f64,
    detrend: DetrendOpt,
) -> (Array2<f64>, Vec<f64>) {
    let win_samples = (window_size * fs).round() as usize;
    let nfft = next_pow2(fs / dsfreqs);
    let n_rfft = nfft / 2 + 1;

    // Precompute normalized Hann taper: h[n] = 0.5*(1-cos(2pi n/(N-1)))
    // then divide by sqrt(sum(h^2)).
    let mut taper: Vec<f64> = (0..win_samples)
        .map(|n| 0.5 * (1.0 - ((2.0 * std::f64::consts::PI * n as f64)
            / (win_samples as f64 - 1.0)).cos()))
        .collect();
    let taper_norm: f64 = taper.iter().map(|v| v * v).sum::<f64>().sqrt();
    for v in taper.iter_mut() {
        *v /= taper_norm;
    }

    // Build frequency mask once.
    let sfreqs_full: Vec<f64> = (0..n_rfft).map(|i| i as f64 * fs / nfft as f64).collect();
    let keep_idx: Vec<usize> = sfreqs_full
        .iter()
        .enumerate()
        .filter_map(|(i, &f)| {
            if f >= freq_range.0 && f <= freq_range.1 {
                Some(i)
            } else {
                None
            }
        })
        .collect();
    let sfreqs: Vec<f64> = keep_idx.iter().map(|&i| sfreqs_full[i]).collect();
    let f_masked = sfreqs.len();

    let n_events = event_times.len();
    let mut spect = Array2::<f64>::zeros((f_masked, n_events));
    if n_events == 0 {
        return (spect, sfreqs);
    }

    // Thread-local FFT planner + scratch buffers (realfft FFT is not Sync).
    // Use rayon par_iter over event indices.
    let event_out: Vec<Vec<f64>> = (0..n_events)
        .into_par_iter()
        .map_init(
            || {
                let mut planner = RealFftPlanner::<f64>::new();
                let fft = planner.plan_fft_forward(nfft);
                let in_buf = vec![0.0f64; nfft];
                let out_buf: Vec<Complex<f64>> = fft.make_output_vec();
                let scratch: Vec<Complex<f64>> = fft.make_scratch_vec();
                (fft, in_buf, out_buf, scratch)
            },
            |(fft, in_buf, out_buf, scratch), e_idx| {
                let center = event_times[e_idx];
                let start_idx =
                    ((center - t0 - window_size * 0.5) * fs).floor() as isize;
                // Fill in_buf[..win_samples] with the windowed segment.
                // Clamp defensively — caller is expected to have pre-filtered.
                in_buf.fill(0.0);
                for n in 0..win_samples {
                    let idx = start_idx + n as isize;
                    if idx < 0 || (idx as usize) >= data.len() {
                        continue;
                    }
                    in_buf[n] = data[idx as usize];
                }
                // Detrend.
                match detrend {
                    DetrendOpt::Constant => {
                        let mean: f64 =
                            in_buf[..win_samples].iter().sum::<f64>() / win_samples as f64;
                        for v in in_buf[..win_samples].iter_mut() {
                            *v -= mean;
                        }
                    }
                    DetrendOpt::Linear => {
                        // y = a*x + b; subtract.
                        let n = win_samples as f64;
                        let xm = (n - 1.0) * 0.5;
                        let ym: f64 = in_buf[..win_samples].iter().sum::<f64>() / n;
                        let mut xy = 0.0f64;
                        let mut xx = 0.0f64;
                        for i in 0..win_samples {
                            let xi = i as f64 - xm;
                            xy += xi * (in_buf[i] - ym);
                            xx += xi * xi;
                        }
                        let slope = xy / xx;
                        for i in 0..win_samples {
                            let xi = i as f64 - xm;
                            in_buf[i] -= slope * xi + ym;
                        }
                    }
                    DetrendOpt::None => {}
                }
                // Hann taper.
                for i in 0..win_samples {
                    in_buf[i] *= taper[i];
                }
                // Zero out the pad-region (in case some earlier iteration
                // left junk; we already fill(0.0) above so this is a no-op,
                // but keep explicit).
                // FFT.
                fft.process_with_scratch(in_buf, out_buf, scratch).unwrap();
                // Convert to one-sided PSD and keep only the freq_range bins.
                let mut row = Vec::with_capacity(f_masked);
                for (mi, &fi) in keep_idx.iter().enumerate() {
                    let c = out_buf[fi];
                    let p = (c.re * c.re + c.im * c.im) / fs;
                    let v = if fi == 0 || fi == n_rfft - 1 { p } else { 2.0 * p };
                    let _ = mi;
                    row.push(v);
                }
                row
            },
        )
        .collect();

    for (e_idx, row) in event_out.into_iter().enumerate() {
        for f in 0..f_masked {
            spect[[f, e_idx]] = row[f];
        }
    }
    (spect, sfreqs)
}

#[derive(Copy, Clone, Debug)]
pub enum DetrendOpt {
    None,
    Constant,
    Linear,
}

/// Not-a-knot cubic spline coefficients for y(x), as MATLAB `spline` and
/// `interp1(..., 'spline')` compute them. Returns (a, b, c, d), each of
/// length N-1, so that for x in [xs[i], xs[i+1]]
/// y(x) = a[i] + b[i]*(x-xs[i]) + c[i]*(x-xs[i])^2 + d[i]*(x-xs[i])^3.
///
/// Not-a-knot end condition: the third derivative is continuous across
/// the second and the second-to-last points, so the first two pieces are
/// one cubic and so are the last two. With three points that leaves a
/// single parabola through them, and with two points a straight line,
/// which is what MATLAB returns in those cases.
fn not_a_knot_cubic_spline_coeffs(
    xs: &[f64],
    ys: &[f64],
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = xs.len();
    assert!(n >= 2);
    let n_pieces = n - 1;

    // h[i] = width of piece i, slope[i] = its secant slope.
    let mut h = vec![0.0f64; n_pieces];
    let mut slope = vec![0.0f64; n_pieces];
    for i in 0..n_pieces {
        h[i] = xs[i + 1] - xs[i];
        slope[i] = (ys[i + 1] - ys[i]) / h[i];
    }

    // c_node[i] is the quadratic coefficient at point i (half the second
    // derivative there).
    let mut c_node = vec![0.0f64; n];

    if n == 3 {
        // One parabola through the three points: constant second derivative.
        let curvature = (slope[1] - slope[0]) / (h[0] + h[1]);
        c_node[0] = curvature;
        c_node[1] = curvature;
        c_node[2] = curvature;
    } else if n >= 4 {
        // Interior continuity equations, one per interior point i = 1..n-2:
        //   h[i-1]*c[i-1] + 2*(h[i-1]+h[i])*c[i] + h[i]*c[i+1] = 3*(slope[i]-slope[i-1])
        // The two not-a-knot conditions express the end values through
        // their neighbors,
        //   c[0]   = ((h[0]+h[1])*c[1] - h[0]*c[2]) / h[1]
        //   c[n-1] = ((hA+hB)*c[n-2] - hB*c[n-3]) / hA,  hA = h[n-3], hB = h[n-2]
        // and substituting them into the first and last interior equations
        // leaves a tridiagonal system in c[1..n-2].
        let n_unknowns = n - 2;
        let mut lower = vec![0.0f64; n_unknowns];
        let mut diagonal = vec![0.0f64; n_unknowns];
        let mut upper = vec![0.0f64; n_unknowns];
        let mut rhs = vec![0.0f64; n_unknowns];
        for k in 0..n_unknowns {
            let i = k + 1;
            lower[k] = h[i - 1];
            diagonal[k] = 2.0 * (h[i - 1] + h[i]);
            upper[k] = h[i];
            rhs[k] = 3.0 * (slope[i] - slope[i - 1]);
        }
        // First interior equation with c[0] substituted.
        let h0 = h[0];
        let h1 = h[1];
        diagonal[0] = (h0 + h1) * (h0 + 2.0 * h1) / h1;
        upper[0] = (h1 * h1 - h0 * h0) / h1;
        lower[0] = 0.0;
        // Last interior equation with c[n-1] substituted.
        let h_a = h[n - 3];
        let h_b = h[n - 2];
        let last = n_unknowns - 1;
        diagonal[last] = (h_a + h_b) * (h_b + 2.0 * h_a) / h_a;
        lower[last] = (h_a * h_a - h_b * h_b) / h_a;
        upper[last] = 0.0;

        // Thomas algorithm: forward elimination, then back substitution.
        for k in 1..n_unknowns {
            let factor = lower[k] / diagonal[k - 1];
            diagonal[k] -= factor * upper[k - 1];
            rhs[k] -= factor * rhs[k - 1];
        }
        let mut solution = vec![0.0f64; n_unknowns];
        solution[last] = rhs[last] / diagonal[last];
        for k in (0..last).rev() {
            solution[k] = (rhs[k] - upper[k] * solution[k + 1]) / diagonal[k];
        }
        for k in 0..n_unknowns {
            c_node[k + 1] = solution[k];
        }
        c_node[0] = ((h0 + h1) * c_node[1] - h0 * c_node[2]) / h1;
        c_node[n - 1] = ((h_a + h_b) * c_node[n - 2] - h_b * c_node[n - 3]) / h_a;
    }
    // n == 2: c_node stays zero, which is a straight line.

    let mut b = vec![0.0f64; n_pieces];
    let mut d = vec![0.0f64; n_pieces];
    for j in 0..n_pieces {
        b[j] = slope[j] - h[j] * (c_node[j + 1] + 2.0 * c_node[j]) / 3.0;
        d[j] = (c_node[j + 1] - c_node[j]) / (3.0 * h[j]);
    }
    let a: Vec<f64> = ys[..n_pieces].to_vec();
    let c: Vec<f64> = c_node[..n_pieces].to_vec();
    (a, b, c, d)
}

/// Evaluate the cubic spline at a single x-point. Outside the data range
/// the end piece is extended, as `interp1(..., 'spline')` extrapolates.
#[inline]
fn spline_eval(xs: &[f64], coeffs: &(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>), x: f64) -> f64 {
    // Find piece index via binary search (xs sorted ascending).
    let (a, b, c, d) = coeffs;
    if x <= xs[0] {
        let dx = x - xs[0];
        return a[0] + b[0] * dx + c[0] * dx * dx + d[0] * dx * dx * dx;
    }
    let last = xs.len() - 1;
    if x >= xs[last] {
        let i = last - 1;
        let dx = x - xs[i];
        return a[i] + b[i] * dx + c[i] * dx * dx + d[i] * dx * dx * dx;
    }
    // binary search for i s.t. xs[i] <= x < xs[i+1]
    let mut lo = 0usize;
    let mut hi = last;
    while lo + 1 < hi {
        let mid = (lo + hi) / 2;
        if xs[mid] <= x {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let dx = x - xs[lo];
    a[lo] + b[lo] * dx + c[lo] * dx * dx + d[lo] * dx * dx * dx
}

/// Given a Hann-event spectrum matrix (F, N) + sfreqs, refine each event's
/// frequency by fitting a cubic spline on the row and taking the argmax on
/// a dense grid inside `[bbox_lo, bbox_hi]`. Returns refined freq or NaN
/// if the peak sits within 1e-3 of the bbox boundaries.
pub fn refine_from_spectra(
    spect: &Array2<f64>,
    sfreqs: &[f64],
    bbox_lo: &[f64],   // length N_events (pre-keep filter applied by caller)
    bbox_hi: &[f64],
    n_grid: usize,
    remove_edge_peaks: bool,
) -> Vec<f64> {
    let n_events = spect.ncols();
    let f_mask = spect.nrows();
    assert_eq!(bbox_lo.len(), n_events);
    assert_eq!(bbox_hi.len(), n_events);

    (0..n_events)
        .into_par_iter()
        .map(|e| {
            let start_freq = bbox_lo[e];
            let end_freq = bbox_hi[e];
            if end_freq <= start_freq {
                return f64::NAN;
            }
            // Extract column (event's full PSD).
            let mut curr = Vec::with_capacity(f_mask);
            for f in 0..f_mask {
                curr.push(spect[[f, e]]);
            }
            if curr.len() < 2 {
                return f64::NAN;
            }
            let coeffs = not_a_knot_cubic_spline_coeffs(sfreqs, &curr);
            let mut best_x = start_freq;
            let mut best_y = f64::NEG_INFINITY;
            // MATLAB `linspace(start_freq, end_freq, n_grid)`:
            // start + i*(end - start)/(n - 1), with the last point set to
            // `end_freq` exactly.
            let n_steps = (n_grid - 1) as f64;
            for i in 0..n_grid {
                let x = if i == n_grid - 1 {
                    end_freq
                } else {
                    start_freq + (i as f64) * (end_freq - start_freq) / n_steps
                };
                let y = spline_eval(sfreqs, &coeffs, x);
                if y > best_y {
                    best_y = y;
                    best_x = x;
                }
            }
            if remove_edge_peaks {
                if (best_x - start_freq).abs() < 1e-3
                    || (best_x - end_freq).abs() < 1e-3
                    || best_x > end_freq
                    || end_freq < start_freq
                {
                    return f64::NAN;
                }
            }
            best_x
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spline_exact_on_linear() {
        // Linear function: spline should exactly reproduce.
        let xs = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let ys = vec![0.0, 2.0, 4.0, 6.0, 8.0];
        let coeffs = not_a_knot_cubic_spline_coeffs(&xs, &ys);
        for &x in &[0.5, 1.5, 2.75, 3.9] {
            let y = spline_eval(&xs, &coeffs, x);
            let expected = 2.0 * x;
            assert!((y - expected).abs() < 1e-10, "x={x} y={y} expected={expected}");
        }
    }

    /// Evaluate the spline through (xs, ys) at each query.
    fn spline_at(xs: &[f64], ys: &[f64], queries: &[f64]) -> Vec<f64> {
        let coeffs = not_a_knot_cubic_spline_coeffs(xs, ys);
        let mut values = Vec::new();
        for &query in queries {
            values.push(spline_eval(xs, &coeffs, query));
        }
        values
    }

    fn assert_all_close(got: &[f64], expected: &[f64]) {
        assert_eq!(got.len(), expected.len());
        for i in 0..got.len() {
            assert!(
                (got[i] - expected[i]).abs() < 1e-12,
                "index {}: got {} expected {}",
                i, got[i], expected[i]
            );
        }
    }

    // Expected values are MATLAB R2025b `interp1(x, y, xq, 'spline')`
    // outputs for the same inputs, including the extrapolated queries.

    #[test]
    fn spline_matches_matlab_on_a_uniform_grid() {
        // interp1([0 1 2 3 4 5],[1 3 2 5 4 6],[0.5 1.5 4.5 -0.5 5.5],'spline')
        let got = spline_at(
            &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            &[1.0, 3.0, 2.0, 5.0, 4.0, 6.0],
            &[0.5, 1.5, 4.5, -0.5, 5.5],
        );
        assert_all_close(&got, &[3.125, 2.125, 3.875, -4.875, 11.875]);
    }

    #[test]
    fn spline_matches_matlab_on_a_nonuniform_grid() {
        // interp1([0 0.5 2 3 4.5 5],[1 3 2 5 4 6],[0.25 1 4.75 -0.5 5.5],'spline')
        let got = spline_at(
            &[0.0, 0.5, 2.0, 3.0, 4.5, 5.0],
            &[1.0, 3.0, 2.0, 5.0, 4.0, 6.0],
            &[0.25, 1.0, 4.75, -0.5, 5.5],
        );
        assert_all_close(&got, &[2.354375, 2.88, 4.645625, -4.55, 11.55]);
    }

    #[test]
    fn spline_matches_matlab_with_few_points() {
        // interp1([0 1 2 3],[1 3 2 5],[0.5 1.5 2.5],'spline')
        let four = spline_at(&[0.0, 1.0, 2.0, 3.0], &[1.0, 3.0, 2.0, 5.0], &[0.5, 1.5, 2.5]);
        assert_all_close(&four, &[2.8125, 2.4375, 2.5625]);
        // interp1([0 1 2],[1 3 2],[0.5 1.5 2.5],'spline'): one parabola
        let three = spline_at(&[0.0, 1.0, 2.0], &[1.0, 3.0, 2.0], &[0.5, 1.5, 2.5]);
        assert_all_close(&three, &[2.375, 2.875, 0.375]);
        // interp1([0 1],[1 3],[0.5 1.5],'spline'): a straight line
        let two = spline_at(&[0.0, 1.0], &[1.0, 3.0], &[0.5, 1.5]);
        assert_all_close(&two, &[2.0, 4.0]);
    }

    #[test]
    fn refine_picks_interior_peak() {
        // 2 events, both with Gaussian-ish PSD centered at 5 Hz.
        // bbox = [3, 7] — expect ~5 Hz.
        let sfreqs: Vec<f64> = (0..11).map(|i| i as f64).collect();
        let spect_vals: Vec<f64> = sfreqs
            .iter()
            .map(|f| (-((f - 5.0).powi(2)) / 1.0).exp())
            .collect();
        let mut spect = Array2::<f64>::zeros((11, 2));
        for i in 0..11 {
            spect[[i, 0]] = spect_vals[i];
            spect[[i, 1]] = spect_vals[i];
        }
        let lo = vec![3.0, 3.0];
        let hi = vec![7.0, 7.0];
        let out = refine_from_spectra(&spect, &sfreqs, &lo, &hi, 1000, true);
        // Grid resolution is (7-3)/999 ≈ 4e-3; allow that tolerance.
        for &v in &out {
            assert!((v - 5.0).abs() < 5e-3, "{v}");
        }
    }
}
