//! A cache rebuild reproduces the full run's SOPHs.
//!
//! `rebuild_from_cache` rebuilds SOPHs (and fits) from what a full run
//! stores: the per-peak table, the native-grid SO-power series, the
//! staging, and the artifact spans. Given the full run's own per-peak
//! values (not rounded through a CSV), its SOPHs must equal the full run's
//! exactly. In particular the SO-power time-in-bin has to be counted on the
//! same EEG-rate series the full run and MATLAB `computeSOpower.m` use, not
//! on the stored window series.
//!
//! Fixture-free: the recording is generated here from a seeded generator.

use dynamo_rs::run::rebuild::{artifact_mask_to_spans, cache_time_grid};
use dynamo_rs::run::{rebuild_from_cache, run_dynamo, DynamoSettings, RebuildInputs};

const FS: f64 = 100.0;
const DURATION_S: f64 = 900.0;

/// Deterministic uniform generator on (0, 1) (64-bit LCG, top 53 bits).
struct Lcg(u64);

impl Lcg {
    fn next_uniform(&mut self) -> f64 {
        // Knuth's MMIX multiplier and increment.
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let top_bits = self.0 >> 11;
        (top_bits as f64 + 0.5) / (1u64 << 53) as f64
    }

    /// Standard normal sample (Box-Muller).
    fn next_normal(&mut self) -> f64 {
        let u1 = self.next_uniform();
        let u2 = self.next_uniform();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

/// Fifteen minutes of synthetic sleep EEG: a slow oscillation whose
/// amplitude drifts (so SO-power spans many bins), spindle bursts, noise,
/// and three stretches the artifact detector must flag.
fn synthetic_eeg() -> Vec<f64> {
    let n_samples = (DURATION_S * FS) as usize;
    let mut rng = Lcg(0x5EED_2026_1122_0017);
    let mut data = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        let t = i as f64 / FS;

        let so_amplitude = 20.0 + 25.0 * (std::f64::consts::TAU * t / 300.0).sin().abs();
        let slow_oscillation = so_amplitude * (std::f64::consts::TAU * 0.9 * t).sin();

        let time_in_cycle = t % 5.0;
        let envelope = (-0.5 * ((time_in_cycle - 2.5) / 0.4).powi(2)).exp();
        let spindle = 12.0 * envelope * (std::f64::consts::TAU * 13.0 * t).sin();

        let noise = 3.0 * rng.next_normal();
        let mut sample = slow_oscillation + spindle + noise;

        if (200.0..202.0).contains(&t) {
            sample += 400.0 * (std::f64::consts::TAU * 40.0 * t).sin();
        }
        if (430.0..433.0).contains(&t) {
            sample = 0.0;
        }
        if (700.0..701.5).contains(&t) {
            sample += 500.0 * (std::f64::consts::TAU * 35.0 * t).sin();
        }
        data.push(sample);
    }
    data
}

/// 30 s epochs. The first epoch is unscored, so the scored window (and the
/// analyzed sample grid) starts at 30 s rather than at the first sample.
fn synthetic_staging() -> (Vec<f64>, Vec<f64>) {
    let cycle = [2.0, 1.0, 2.0, 3.0, 1.0, 2.0];
    let n_epochs = (DURATION_S / 30.0) as usize;
    let mut stage_times = Vec::with_capacity(n_epochs);
    let mut stage_vals = Vec::with_capacity(n_epochs);
    for epoch in 0..n_epochs {
        stage_times.push(epoch as f64 * 30.0);
        if epoch == 0 {
            stage_vals.push(0.0);
        } else {
            stage_vals.push(cycle[epoch % cycle.len()]);
        }
    }
    (stage_times, stage_vals)
}

#[test]
fn cache_time_grid_is_the_full_runs_sample_grid() {
    let data = synthetic_eeg();
    let (stage_times, stage_vals) = synthetic_staging();
    let settings = DynamoSettings::default();
    let out = run_dynamo(&data, FS, &stage_times, &stage_vals, None, &settings)
        .expect("run_dynamo failed on the synthetic recording");

    let stage_codes: Vec<u8> = stage_vals.iter().map(|&v| v as u8).collect();
    let (t0, n_samp) = cache_time_grid(&stage_times, &stage_codes, FS);
    assert_eq!(n_samp, out.t_time_range.len(), "sample count of the scored window");
    assert_eq!(t0, out.t_time_range[0]);
}

#[test]
fn rebuilt_sophs_equal_the_full_runs() {
    let data = synthetic_eeg();
    let (stage_times, stage_vals) = synthetic_staging();
    let mut settings = DynamoSettings::default();
    assert!(settings.sopower.retain_fs, "the default upsamples SO-power to the EEG grid");
    // The default 10-minute floor would leave every SO-power bin of a
    // 15-minute recording empty.
    settings.hist.min_time_in_bin = 0.25;
    let out = run_dynamo(&data, FS, &stage_times, &stage_vals, None, &settings)
        .expect("run_dynamo failed on the synthetic recording");
    assert!(out.artifacts.iter().any(|&a| a), "the synthetic artifacts were not detected");

    // What a results folder stores, at full precision.
    let stats = &out.stats;
    let n_peaks = stats.peak_time.len();
    assert!(n_peaks > 100, "expected a few hundred peaks, got {}", n_peaks);
    let mut bbox_tl_s = Vec::with_capacity(n_peaks);
    let mut bbox_tl_hz = Vec::with_capacity(n_peaks);
    let mut bbox_width_s = Vec::with_capacity(n_peaks);
    let mut bbox_height_hz = Vec::with_capacity(n_peaks);
    for i in 0..n_peaks {
        bbox_tl_s.push(stats.bbox[4 * i]);
        bbox_tl_hz.push(stats.bbox[4 * i + 1]);
        bbox_width_s.push(stats.bbox[4 * i + 2]);
        bbox_height_hz.push(stats.bbox[4 * i + 3]);
    }
    let segment_num: Vec<i64> = stats.segment_num.iter().map(|&s| s as i64).collect();
    let stage_codes: Vec<u8> = out.stage_vals.iter().map(|&v| v as u8).collect();
    let baseline: Vec<f64> = out.baseline.to_vec();
    let artifact_spans = artifact_mask_to_spans(&out.artifacts, out.fs);
    let (t0, n_samp) = cache_time_grid(&out.stage_times, &stage_codes, out.fs);

    let mut rebuild_settings = settings.clone();
    // The rebuild has no signal, so it cannot compute the display spectrogram.
    rebuild_settings.output.write_summary_png = false;
    let rebuilt = rebuild_from_cache(&RebuildInputs {
        peak_time: &stats.peak_time,
        peak_freq: &stats.peak_freq,
        duration: &stats.duration,
        bandwidth: &stats.bandwidth,
        height: &stats.height,
        volume: &stats.volume,
        segment_num: &segment_num,
        bbox_tl_s: &bbox_tl_s,
        bbox_tl_hz: &bbox_tl_hz,
        bbox_width_s: &bbox_width_s,
        bbox_height_hz: &bbox_height_hz,
        area: &stats.area,
        peakiness: &stats.peakiness,
        peak_stage: &out.peak_stage,
        peak_sopower: &out.peak_sopower,
        peak_sophase: &out.peak_sophase,
        baseline: &baseline,
        baseline_freqs: &out.baseline_freqs,
        so_power_norm: &out.so_power_norm_native,
        so_power_t_start: out.so_power_native_t_start,
        so_power_step: settings.sopower.window_params.1,
        stage_times: &out.stage_times,
        stage_vals: &stage_codes,
        artifact_spans: &artifact_spans,
        data_arr: None,
        fs: out.fs,
        t0,
        n_samp,
        skip_summary_spec: true,
        settings: &rebuild_settings,
    })
    .expect("rebuild_from_cache failed");

    // The rebuilt artifact mask and sample grid are the full run's.
    assert_eq!(rebuilt.artifacts, out.artifacts, "artifact mask");
    assert_eq!(rebuilt.t_time_range, out.t_time_range, "sample grid");

    // SO-power SOPH: same bins, same rates (and the same empty cells).
    assert_eq!(rebuilt.sopower_bin_centers, out.sopower_bin_centers);
    assert_eq!(rebuilt.hist_freq_centers, out.hist_freq_centers);
    assert_eq!(rebuilt.sopower_hist.dim(), out.sopower_hist.dim());
    let n_finite_cells = out.sopower_hist.iter().filter(|v| v.is_finite()).count();
    assert!(n_finite_cells > 100, "too few non-empty SO-power cells ({}) to compare", n_finite_cells);
    let mut largest_relative_difference: f64 = 0.0;
    for (rebuilt_value, full_value) in rebuilt.sopower_hist.iter().zip(out.sopower_hist.iter()) {
        assert_eq!(rebuilt_value.is_nan(), full_value.is_nan(), "empty-cell pattern");
        if full_value.is_finite() && *full_value != 0.0 {
            let relative = ((rebuilt_value - full_value) / full_value).abs();
            largest_relative_difference = largest_relative_difference.max(relative);
        }
    }
    // The stored window times are rebuilt as `t_start + i * step`, which can
    // differ from the full run's by a few ulps; nothing beyond that.
    assert!(
        largest_relative_difference < 1e-9,
        "SO-power SOPH differs from the full run's by up to {:e} (relative)",
        largest_relative_difference
    );

    // SO-phase SOPH: built from the per-peak values alone.
    assert_eq!(rebuilt.sophase_hist.dim(), out.sophase_hist.dim());
    for (rebuilt_value, full_value) in rebuilt.sophase_hist.iter().zip(out.sophase_hist.iter()) {
        assert!(
            rebuilt_value.to_bits() == full_value.to_bits()
                || (rebuilt_value.is_nan() && full_value.is_nan()),
            "SO-phase SOPH differs: {} vs {}",
            rebuilt_value,
            full_value
        );
    }
}
