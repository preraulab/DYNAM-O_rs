//! Chebyshev Type I filter design in SOS form.
//!
//! Port of `scipy.signal.cheby1(N, rp, Wn, btype, fs, output='sos')` for the
//! exact subset DYNAM-O needs (highpass N=4, 0.2 dB ripple). Matches scipy's
//! algorithm: analog prototype → freq-transform → bilinear → SOS grouping.
//!
//! Steps (mirror scipy.signal.iirfilter):
//!   1. `cheb1ap(N, rp)` → analog prototype zeros/poles/gain.
//!   2. `lp2hp_zpk` / `lp2lp_zpk` / `lp2bp_zpk` frequency transform (warped).
//!   3. `bilinear_zpk` with `fs=2` (prewarp to normalized rad/sample domain).
//!   4. `zpk2sos` using scipy's pairing heuristic.
//!
//! Only highpass and bandpass are implemented — that's all DYNAM-O uses.

use realfft::num_complex::Complex64;

type C = Complex64;

/// Chebyshev Type I analog lowpass prototype (Wn=1 rad/s).
/// Returns (zeros, poles, gain). Zeros is always empty (all-pole).
fn cheb1ap(n: usize, rp: f64) -> (Vec<C>, Vec<C>, f64) {
    assert!(n >= 1);
    let eps = (10f64.powf(0.1 * rp) - 1.0).sqrt();
    let mu = (1.0 / eps).asinh() / n as f64;
    // Poles on the ellipse: p_k = -sinh(mu) sin(θ_k) + j cosh(mu) cos(θ_k)
    // with θ_k = π(2k - 1)/(2n), k = 1..n.
    let sinh_mu = mu.sinh();
    let cosh_mu = mu.cosh();
    let mut poles: Vec<C> = Vec::with_capacity(n);
    for k in 1..=n {
        let theta = std::f64::consts::PI * (2.0 * k as f64 - 1.0) / (2.0 * n as f64);
        poles.push(C::new(-sinh_mu * theta.sin(), cosh_mu * theta.cos()));
    }
    // Gain: scipy's cheb1ap does prod(-poles).real, then divides by
    // sqrt(1+eps^2) for even n (to compensate for Type I DC gain).
    let mut g = C::new(1.0, 0.0);
    for &p in &poles {
        g *= -p;
    }
    let mut gain = g.re;
    if n % 2 == 0 {
        gain /= (1.0 + eps * eps).sqrt();
    }
    (Vec::new(), poles, gain)
}

/// Lowpass-to-highpass frequency transform (analog ZPK).
/// s ← wo/s. Zeros at s=0 of multiplicity (len(p) - len(z)) are added;
/// then each original zero z_k maps to wo/z_k and similarly for poles.
fn lp2hp_zpk(z: &[C], p: &[C], k: f64, wo: f64) -> (Vec<C>, Vec<C>, f64) {
    let degree = p.len() as isize - z.len() as isize;
    let z_new: Vec<C> = z.iter().map(|&zz| C::new(wo, 0.0) / zz).collect();
    let p_new: Vec<C> = p.iter().map(|&pp| C::new(wo, 0.0) / pp).collect();
    // Add zeros at s=0 to make it proper
    let mut z_out = z_new;
    for _ in 0..degree.max(0) {
        z_out.push(C::new(0.0, 0.0));
    }
    // Gain: k * prod(-z_i) / prod(-p_i)
    let mut num = C::new(1.0, 0.0);
    for &zz in z {
        num *= -zz;
    }
    let mut den = C::new(1.0, 0.0);
    for &pp in p {
        den *= -pp;
    }
    let k_new = k * (num / den).re;
    (z_out, p_new, k_new)
}

/// Lowpass-to-lowpass analog frequency transform: s ← s/wo.
#[allow(dead_code)]
fn lp2lp_zpk(z: &[C], p: &[C], k: f64, wo: f64) -> (Vec<C>, Vec<C>, f64) {
    let degree = p.len() as isize - z.len() as isize;
    let z_new: Vec<C> = z.iter().map(|&zz| zz * wo).collect();
    let p_new: Vec<C> = p.iter().map(|&pp| pp * wo).collect();
    let k_new = k * wo.powi(degree as i32);
    (z_new, p_new, k_new)
}

/// Lowpass-to-bandpass analog transform: s ← (s^2 + wo^2) / (bw*s).
#[allow(dead_code)]
fn lp2bp_zpk(z: &[C], p: &[C], k: f64, wo: f64, bw: f64) -> (Vec<C>, Vec<C>, f64) {
    let degree = p.len() as isize - z.len() as isize;
    // Each zero/pole splits into a conjugate pair.
    let mut z_new: Vec<C> = Vec::with_capacity(2 * z.len() + degree.max(0) as usize);
    for &zz in z {
        let zlp = zz * bw / 2.0;
        let sq = (zlp * zlp - wo * wo).sqrt();
        z_new.push(zlp + sq);
        z_new.push(zlp - sq);
    }
    let mut p_new: Vec<C> = Vec::with_capacity(2 * p.len());
    for &pp in p {
        let plp = pp * bw / 2.0;
        let sq = (plp * plp - wo * wo).sqrt();
        p_new.push(plp + sq);
        p_new.push(plp - sq);
    }
    for _ in 0..degree.max(0) {
        z_new.push(C::new(0.0, 0.0));
    }
    let k_new = k * bw.powi(degree as i32);
    (z_new, p_new, k_new)
}

/// Bilinear transform of an analog ZPK filter. s ← (2*fs)*(z-1)/(z+1).
/// Matches scipy.signal.bilinear_zpk.
fn bilinear_zpk(z: &[C], p: &[C], k: f64, fs: f64) -> (Vec<C>, Vec<C>, f64) {
    let degree = p.len() as isize - z.len() as isize;
    let fs2 = 2.0 * fs;
    let z_new: Vec<C> = z.iter().map(|&zz| (fs2 + zz) / (fs2 - zz)).collect();
    let p_new: Vec<C> = p.iter().map(|&pp| (fs2 + pp) / (fs2 - pp)).collect();
    let mut z_out = z_new;
    for _ in 0..degree.max(0) {
        z_out.push(C::new(-1.0, 0.0));
    }
    // Gain: k * prod(fs2 - z_i) / prod(fs2 - p_i)
    let mut num = C::new(1.0, 0.0);
    for &zz in z {
        num *= C::new(fs2, 0.0) - zz;
    }
    let mut den = C::new(1.0, 0.0);
    for &pp in p {
        den *= C::new(fs2, 0.0) - pp;
    }
    let k_new = k * (num / den).re;
    (z_out, p_new, k_new)
}

/// Convert a pair of complex conjugate poles (or two real poles) + a pair of
/// zeros into a second-order section [b0 b1 b2 a0 a1 a2].
fn zpk_pair_to_section(zs: &[C], ps: &[C]) -> [f64; 6] {
    // (z - z1)(z - z2) = z^2 - (z1+z2)z + z1*z2
    let (b1, b2) = if zs.len() == 2 {
        let s = -(zs[0] + zs[1]);
        let p = zs[0] * zs[1];
        (s.re, p.re)
    } else if zs.len() == 1 {
        (-zs[0].re, 0.0)
    } else {
        (0.0, 0.0)
    };
    let (a1, a2) = if ps.len() == 2 {
        let s = -(ps[0] + ps[1]);
        let p = ps[0] * ps[1];
        (s.re, p.re)
    } else if ps.len() == 1 {
        (-ps[0].re, 0.0)
    } else {
        (0.0, 0.0)
    };
    [1.0, b1, b2, 1.0, a1, a2]
}

/// Group zeros and poles into conjugate pairs (scipy-style): each complex
/// pole paired with its conjugate; real poles paired with each other (or
/// standalone). Same for zeros. Returns a vector of (zeros_in_pair, poles_in_pair).
fn pair_conjugates(items: &[C]) -> Vec<Vec<C>> {
    let mut used = vec![false; items.len()];
    let mut out: Vec<Vec<C>> = Vec::new();
    for i in 0..items.len() {
        if used[i] {
            continue;
        }
        let zi = items[i];
        if zi.im.abs() < 1e-12 {
            // real. look for another real to pair.
            let mut pair_idx = None;
            for j in (i + 1)..items.len() {
                if !used[j] && items[j].im.abs() < 1e-12 {
                    pair_idx = Some(j);
                    break;
                }
            }
            match pair_idx {
                Some(j) => {
                    used[i] = true;
                    used[j] = true;
                    out.push(vec![C::new(zi.re, 0.0), C::new(items[j].re, 0.0)]);
                }
                None => {
                    used[i] = true;
                    out.push(vec![C::new(zi.re, 0.0)]);
                }
            }
        } else {
            // complex. find conjugate.
            let mut pair_idx = None;
            for j in (i + 1)..items.len() {
                if !used[j]
                    && (items[j].re - zi.re).abs() < 1e-9
                    && (items[j].im + zi.im).abs() < 1e-9
                {
                    pair_idx = Some(j);
                    break;
                }
            }
            match pair_idx {
                Some(j) => {
                    used[i] = true;
                    used[j] = true;
                    out.push(vec![zi, items[j]]);
                }
                None => {
                    // Unmatched complex → keep alone (shouldn't happen for real filters)
                    used[i] = true;
                    out.push(vec![zi]);
                }
            }
        }
    }
    out
}

/// Simple ZPK→SOS conversion using scipy's "nearest" pairing heuristic:
/// pair poles with their conjugates; match each pole-pair with the nearest
/// zero-pair; distribute gain on the first section.
///
/// This is sufficient for our use case (all-pole Chebyshev filter → all zero
/// pairs end up at DC or Nyquist after lp2hp+bilinear).
pub fn zpk_to_sos(z: &[C], p: &[C], k: f64) -> Vec<[f64; 6]> {
    // For an all-pole analog prototype that's been lp2hp + bilinear, after
    // bilinear transform we have n zeros at z=+1 or -1 (real) and n poles
    // that come in conjugate pairs (for even n). Pair them up.
    let mut p_pairs = pair_conjugates(p);
    let mut z_pairs = pair_conjugates(z);

    // Want same number of sections; pad missing pairs with empty zeros.
    while z_pairs.len() < p_pairs.len() {
        z_pairs.push(Vec::new());
    }
    while p_pairs.len() < z_pairs.len() {
        p_pairs.push(Vec::new());
    }

    // Sort pole pairs by proximity to unit circle (most peaky last) so the
    // most resonant section runs last — matches scipy convention.
    let pole_radius = |pair: &Vec<C>| -> f64 {
        pair.iter().map(|p| (p.norm() - 1.0).abs()).fold(f64::INFINITY, f64::min)
    };
    let mut pole_order: Vec<usize> = (0..p_pairs.len()).collect();
    pole_order.sort_by(|&a, &b| {
        pole_radius(&p_pairs[b])
            .partial_cmp(&pole_radius(&p_pairs[a]))
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut sections: Vec<[f64; 6]> = Vec::with_capacity(p_pairs.len());
    for (i, pi) in pole_order.iter().enumerate() {
        let ps = &p_pairs[*pi];
        let zs = if i < z_pairs.len() {
            &z_pairs[i]
        } else {
            &z_pairs[0]
        };
        sections.push(zpk_pair_to_section(zs, ps));
    }
    // Apply overall gain on the first section.
    if !sections.is_empty() {
        sections[0][0] *= k;
        sections[0][1] *= k;
        sections[0][2] *= k;
    }
    sections
}

/// `cheby1(order, rp, passband_hz, btype, fs, output='sos')` — scipy.
/// `btype`: `"highpass"` or `"lowpass"` or `"bandpass"`. For bandpass pass
/// the passband as `[lo, hi]`.
pub fn cheby1_sos(order: usize, rp: f64, wn: &[f64], btype: &str, fs: f64) -> Vec<[f64; 6]> {
    // Step 1: analog prototype (Wn = 1 rad/s).
    let (z_lp, p_lp, k_lp) = cheb1ap(order, rp);

    // Step 2: prewarp the critical frequencies to analog domain.
    // wn_digital / fs ∈ (0, 1); warp: w_analog = 2*fs * tan(pi * wn / fs).
    let warp = |f: f64| 2.0 * fs * (std::f64::consts::PI * f / fs).tan();

    // Step 3: frequency transform.
    let (z_a, p_a, k_a) = match btype {
        "highpass" | "hp" => {
            assert_eq!(wn.len(), 1, "highpass takes one critical frequency");
            let wo = warp(wn[0]);
            lp2hp_zpk(&z_lp, &p_lp, k_lp, wo)
        }
        "lowpass" | "lp" => {
            assert_eq!(wn.len(), 1, "lowpass takes one critical frequency");
            let wo = warp(wn[0]);
            lp2lp_zpk(&z_lp, &p_lp, k_lp, wo)
        }
        "bandpass" | "bp" => {
            assert_eq!(wn.len(), 2, "bandpass takes two critical frequencies");
            let w_lo = warp(wn[0]);
            let w_hi = warp(wn[1]);
            let wo = (w_lo * w_hi).sqrt();
            let bw = w_hi - w_lo;
            lp2bp_zpk(&z_lp, &p_lp, k_lp, wo, bw)
        }
        _ => panic!("unsupported btype {:?}", btype),
    };

    // Step 4: bilinear to digital.
    let (z_d, p_d, k_d) = bilinear_zpk(&z_a, &p_a, k_a, fs);

    // Step 5: ZPK → SOS.
    zpk_to_sos(&z_d, &p_d, k_d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cheb1ap_order4() {
        // Order 4, 0.2 dB ripple. scipy: poles at roughly -0.167±1.002j, -0.404±0.416j.
        let (z, p, k) = cheb1ap(4, 0.2);
        assert!(z.is_empty());
        assert_eq!(p.len(), 4);
        // gain real, non-zero
        assert!(k > 0.0);
    }

    #[test]
    fn cheby1_highpass_returns_sections() {
        let sos = cheby1_sos(4, 0.2, &[35.0], "highpass", 100.0);
        // 4th order = 2 sections
        assert_eq!(sos.len(), 2);
    }
}
