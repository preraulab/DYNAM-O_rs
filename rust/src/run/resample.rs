//! EEG resampler — single-channel f64 wrapper around rubato's
//! `FftFixedIn`. Agrees with the `EDF_toolbox/rust/src/resample.rs`
//! reference port to <1% RMS. It is not MATLAB's method: MATLAB
//! `smartresample.m` uses a polyphase FIR (`resample(x, n, d, 500)`).
//!
//! Measured on three 128 Hz recordings resampled to 100 Hz: the two
//! outputs agree to 0.002-0.003% RMS below 35 Hz, and 99.5% or more of
//! their difference lies in 48-50 Hz, where MATLAB's filter (cutoff at
//! the new Nyquist) keeps 7-10 times more power. The analyzed band is
//! therefore the same, but results are not identical: the artifact
//! detector reads that edge band, so the artifact mask and the baseline
//! shift slightly (baseline about 1.5%), and 2-5% of the weakest peaks
//! appear or vanish. SOPH modes do not move (whole-histogram
//! correlation 0.976-0.999). To compare the toolboxes exactly, give both
//! the same already-resampled samples.
//!
//! Only downsamples (or passes through). If `target_rate >= orig_rate`,
//! returns the input untouched — we never upsample EEG.

use rubato::{FftFixedIn, Resampler};

#[derive(Debug, thiserror::Error)]
pub enum ResampleError {
    #[error("rubato: {0}")]
    Rubato(String),
    #[error("invalid rate: orig={orig} target={target}")]
    Rate { orig: f64, target: f64 },
}

/// Resample `data` from `orig_rate` Hz to `target_rate` Hz. Returns
/// the resampled vector. No-op when target >= orig.
pub fn resample_downsample(
    data: &[f64],
    orig_rate: f64,
    target_rate: f64,
) -> Result<Vec<f64>, ResampleError> {
    if !(orig_rate.is_finite() && target_rate.is_finite() && orig_rate > 0.0 && target_rate > 0.0) {
        return Err(ResampleError::Rate { orig: orig_rate, target: target_rate });
    }
    if target_rate >= orig_rate {
        return Ok(data.to_vec());
    }
    let in_rate = orig_rate.round() as usize;
    let out_rate = target_rate.round() as usize;
    if in_rate == 0 || out_rate == 0 {
        return Err(ResampleError::Rate { orig: orig_rate, target: target_rate });
    }

    let n_in = data.len();
    let chunk_size_in = 1024;
    let sub_chunks = 2;
    let mut resampler =
        FftFixedIn::<f64>::new(in_rate, out_rate, chunk_size_in, sub_chunks, 1)
            .map_err(|e| ResampleError::Rubato(format!("{}", e)))?;

    let leading_delay = resampler.output_delay();
    let expected_out = ((n_in as f64) * (out_rate as f64) / (in_rate as f64)).round() as usize;
    let target_collect = leading_delay + expected_out;
    let needed = resampler.input_frames_next();
    let mut output: Vec<f64> = Vec::with_capacity(target_collect + needed);
    let mut inbuf: Vec<Vec<f64>> = vec![vec![0.0f64; needed]; 1];

    // Feed the resampler one fixed-size chunk at a time until enough output
    // has been collected. Past the end of the data the chunk is padded with
    // zeros, which pushes the last real samples through the filter delay.
    let mut pos = 0usize;
    loop {
        if output.len() >= target_collect {
            break;
        }
        let need = resampler.input_frames_next();
        if need != needed {
            inbuf[0].resize(need, 0.0);
        }
        let real_in = if pos < n_in { (n_in - pos).min(need) } else { 0 };
        if real_in > 0 {
            inbuf[0][..real_in].copy_from_slice(&data[pos..pos + real_in]);
        }
        if real_in < need {
            inbuf[0][real_in..].fill(0.0);
        }
        let out = resampler
            .process(&inbuf, None)
            .map_err(|e| ResampleError::Rubato(format!("{}", e)))?;
        output.extend_from_slice(&out[0]);
        pos += real_in;
        if real_in == 0 && out[0].is_empty() {
            break;
        }
    }

    // Trim leading group delay, truncate to expected length.
    if output.len() > leading_delay {
        output.drain(..leading_delay);
    } else {
        output.clear();
    }
    if output.len() > expected_out {
        output.truncate(expected_out);
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn passes_through_when_target_higher() {
        let x: Vec<f64> = (0..1000).map(|i| i as f64).collect();
        let y = resample_downsample(&x, 100.0, 200.0).unwrap();
        assert_eq!(x, y);
    }

    #[test]
    fn downsample_length_matches_matlab_convention() {
        // 1 s of 128 Hz → 100 Hz → exactly 100 output samples.
        let n_in = 128 * 4; // 4 s for stable filter behavior
        let x: Vec<f64> = (0..n_in).map(|i| (i as f64 * 0.1).sin()).collect();
        let y = resample_downsample(&x, 128.0, 100.0).unwrap();
        let expected = ((n_in as f64) * 100.0 / 128.0).round() as usize;
        assert_eq!(y.len(), expected);
    }

    /// How long must a native flat (constant) region be for a *strictly-equal*
    /// flat run (≥ min_run) to survive 128→100 resampling? Run with:
    ///   cargo test -p dynamo-pipeline flat_run_survival -- --nocapture
    #[test]
    fn flat_run_survival_after_resample() {
        let (orig, target) = (128.0_f64, 100.0_f64);
        let min_run = target.round() as usize; // 100 = 1 s at the target rate
        let pad = 512usize;
        println!(
            "\nnative_flat: samples / seconds  →  max exactly-equal run in output  |  in-flat max run  |  survives(≥{min_run})"
        );
        for &flat_len in &[13usize, 64, 128, 640, 1280, 12_800, 128_000, 640_000] {
            let mut x = Vec::with_capacity(2 * pad + flat_len);
            for i in 0..pad {
                x.push((i as f64 * 0.3).sin() * 100.0);
            }
            let flat_start = x.len();
            x.extend(std::iter::repeat(-2048.0).take(flat_len));
            let flat_end = x.len();
            for i in 0..pad {
                x.push((i as f64 * 0.37).sin() * 100.0);
            }
            let y = resample_downsample(&x, orig, target).unwrap();
            // longest exactly-equal run anywhere
            let mut best = 1usize;
            let mut cur = 1usize;
            // longest exactly-equal run restricted to the resampled flat interior
            let (rs, re) = (
                (flat_start as f64 * target / orig).ceil() as usize,
                (flat_end as f64 * target / orig).floor() as usize,
            );
            let mut in_best = 0usize;
            let mut in_cur = 1usize;
            for i in 1..y.len() {
                if y[i] == y[i - 1] {
                    cur += 1;
                    best = best.max(cur);
                    if i >= rs && i < re {
                        in_cur += 1;
                        in_best = in_best.max(in_cur);
                    }
                } else {
                    cur = 1;
                    in_cur = 1;
                }
            }
            println!(
                "  {:>7} / {:>7.2}s  →  {:>8}  |  {:>8}  |  {}",
                flat_len,
                flat_len as f64 / orig,
                best,
                in_best,
                best >= min_run
            );
        }
    }
}
