//! Conformance of the kernel with the pure-MATLAB pipeline, stage by stage.
//!
//! `scripts/gen_matlab_pipeline_fixtures.m` runs MATLAB (`backend =
//! 'matlab'`) on a ten-minute synthetic recording and saves the input and
//! the MATLAB output of each stage under `tests/fixtures/matlab_pipeline/`.
//! Each test here gives the kernel the same input MATLAB had and compares
//! the outputs, so one stage's error cannot hide behind another's.
//!
//! Every tolerance below is stated next to the MATLAB file it answers to.
//! A failure means the kernel and MATLAB have drifted apart. Regenerate
//! the fixtures only when the MATLAB reference changed on purpose.

// Indexed loops keep the comparisons readable next to the MATLAB code.
#![allow(clippy::needless_range_loop)]

use std::fs;
use std::path::PathBuf;

use dynamo_rs::artifacts::{detect_artifacts_with_slope_test, ArtifactOpts};
use dynamo_rs::extract_pipeline::{extract_tfpeaks_segment, ExtractParams};
use dynamo_rs::filter_cache::get_sophase_sos;
use dynamo_rs::peak_assign::{assign_peak_sophase, assign_peak_sopower, assign_peak_stage};
use dynamo_rs::so_phase::so_phase_from_eeg;
use ndarray::{Array1, Array2};

/// Read one fixture file: a flat list of little-endian doubles.
fn read_f64(name: &str) -> Vec<f64> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/matlab_pipeline")
        .join(format!("{}.f64", name));
    let bytes = fs::read(&path).unwrap_or_else(|e| panic!("read {}: {}", path.display(), e));
    assert_eq!(bytes.len() % 8, 0, "{} is not a whole number of doubles", name);
    let mut values = Vec::with_capacity(bytes.len() / 8);
    for chunk in bytes.chunks_exact(8) {
        let mut raw = [0u8; 8];
        raw.copy_from_slice(chunk);
        values.push(f64::from_le_bytes(raw));
    }
    values
}

fn sample_times(n: usize, fs: f64) -> Vec<f64> {
    let mut times = Vec::with_capacity(n);
    for i in 0..n {
        times.push(i as f64 / fs);
    }
    times
}

/// `detect_artifacts.m` with its defaults (slope test on).
///
/// Tolerance: the two masks may differ on at most 2 samples. The decision
/// at each sample is a threshold on a moving statistic, and a value that
/// lands on the threshold to within rounding can fall either way.
#[test]
fn artifact_mask_matches_matlab() {
    let fs = read_f64("fs")[0];
    let signal = read_f64("signal");
    let matlab_mask = read_f64("artifacts");

    let slope_crit = -0.5;
    let kernel_mask =
        detect_artifacts_with_slope_test(&signal, fs, &ArtifactOpts::default(), slope_crit)
            .expect("artifact detection failed");

    assert_eq!(kernel_mask.len(), matlab_mask.len());
    let mut n_matlab = 0usize;
    let mut n_differ = 0usize;
    for i in 0..kernel_mask.len() {
        let matlab_bad = matlab_mask[i] != 0.0;
        if matlab_bad {
            n_matlab += 1;
        }
        if matlab_bad != kernel_mask[i] {
            n_differ += 1;
        }
    }
    assert!(n_matlab > 0, "the fixture has no artifacts, so it tests nothing");
    assert!(
        n_differ <= 2,
        "artifact mask differs from MATLAB on {} of {} samples",
        n_differ,
        kernel_mask.len()
    );
}

/// `computeSOphase.m`: band-pass with the canonical filter through
/// `filtfilt`, then the Hilbert phase.
///
/// Tolerance: filtered signal within 1e-9 of its RMS, phase within 1e-6
/// rad (compared on the circle, since the unwrapped phase may start from a
/// different multiple of 2π).
#[test]
fn so_phase_matches_matlab() {
    let fs = read_f64("fs")[0];
    let signal = read_f64("signal");
    let band = read_f64("so_band");
    let matlab_phase = read_f64("so_phase");
    let matlab_filtered = read_f64("so_filtered");

    let sos = get_sophase_sos(fs, (band[0], band[1])).expect("canonical SO filter");
    let mut sos_rows: Vec<[f64; 6]> = Vec::with_capacity(sos.nrows());
    for section in 0..sos.nrows() {
        let mut row = [0.0f64; 6];
        for k in 0..6 {
            row[k] = sos[[section, k]];
        }
        sos_rows.push(row);
    }

    let times = sample_times(signal.len(), fs);
    let excluded = vec![false; signal.len()];
    let out = so_phase_from_eeg(&signal, &times, &excluded, &sos_rows, &[], &[])
        .expect("so_phase_from_eeg failed");

    let mut sum_squares = 0.0;
    for value in &matlab_filtered {
        sum_squares += value * value;
    }
    let rms = (sum_squares / matlab_filtered.len() as f64).sqrt();

    let mut worst_filtered = 0.0f64;
    let mut worst_phase = 0.0f64;
    for i in 0..signal.len() {
        let filtered_difference = (out.filtdata[i] - matlab_filtered[i]).abs() / rms;
        worst_filtered = worst_filtered.max(filtered_difference);

        // Distance on the circle.
        let raw = out.so_phase_unwrapped[i] - matlab_phase[i];
        let phase_difference = raw.sin().atan2(raw.cos()).abs();
        worst_phase = worst_phase.max(phase_difference);
    }
    assert!(worst_filtered <= 1e-9, "filtered signal differs by {:e} of RMS", worst_filtered);
    assert!(worst_phase <= 1e-6, "SO-phase differs by {:e} rad", worst_phase);
}

/// `extractTFPeaks.m` on single baseline-divided segments: watershed,
/// merge, trim, and the peak properties.
///
/// Tolerance: the same peaks (time and frequency within 1e-6 relative),
/// and every property within 1e-9 relative.
#[test]
fn extracted_peaks_match_matlab() {
    let extract = read_f64("extract_params");
    let merge_thresh = extract[0];
    let downsample_time = extract[1] as usize;
    let downsample_freq = extract[2] as usize;
    let dur_min = extract[3];
    let bw_min = extract[4];
    let trim_vol = extract[5];
    let trim_shift = extract[6];
    let sfreqs = Array1::from(read_f64("sfreqs"));
    let n_segments = read_f64("n_segments")[0] as usize;

    let params = ExtractParams {
        seg_time: 30.0,
        downsample_f: downsample_freq,
        downsample_t: downsample_time,
        merge_thresh,
        max_merges: f64::INFINITY,
        trim_vol_thresh: trim_vol,
        trim_shift_val: trim_shift,
        dur_min,
        dur_max: f64::INFINITY,
        bw_min,
        bw_max: f64::INFINITY,
        freq_min: f64::NEG_INFINITY,
        freq_max: f64::INFINITY,
        ht_db_min: f64::NEG_INFINITY,
        expand_labels_distance: 0,
    };

    let mut n_peaks_checked = 0usize;
    for segment in 0..n_segments {
        let prefix = format!("seg_{}", segment);
        let dims = read_f64(&format!("{}_dims", prefix));
        let n_rows = dims[0] as usize;
        let n_cols = dims[1] as usize;
        let values = read_f64(&format!("{}_values", prefix));
        let image = Array2::from_shape_vec((n_rows, n_cols), values).expect("segment shape");
        let stimes = Array1::from(read_f64(&format!("{}_stimes", prefix)));
        let matlab_flat = read_f64(&format!("{}_peaks", prefix));
        let n_matlab = matlab_flat.len() / 10;

        let (peaks, _labels) = extract_tfpeaks_segment(
            image.view(),
            stimes.view(),
            sfreqs.view(),
            (segment + 1) as f64,
            &params,
        )
        .expect("extract_tfpeaks_segment failed");

        // One kernel row per peak, in the fixture's column order. MATLAB
        // drops peaks at or below the duration and bandwidth floors.
        let mut kernel_rows: Vec<[f64; 10]> = Vec::new();
        for i in 0..peaks.len() {
            if peaks.duration[i] <= dur_min || peaks.bandwidth[i] <= bw_min {
                continue;
            }
            kernel_rows.push([
                peaks.peak_time[i],
                peaks.peak_freq[i],
                peaks.height[i],
                peaks.volume[i],
                peaks.duration[i],
                peaks.bandwidth[i],
                peaks.bbox[i * 4],
                peaks.bbox[i * 4 + 1],
                peaks.bbox[i * 4 + 2],
                peaks.bbox[i * 4 + 3],
            ]);
        }
        assert_eq!(
            kernel_rows.len(),
            n_matlab,
            "segment {}: kernel found {} peaks, MATLAB {}",
            segment,
            kernel_rows.len(),
            n_matlab
        );

        let mut used = vec![false; kernel_rows.len()];
        for m in 0..n_matlab {
            let matlab_row = &matlab_flat[m * 10..(m + 1) * 10];

            // Pair this MATLAB peak with the unused kernel peak nearest in
            // time and frequency.
            let mut best_index = usize::MAX;
            let mut best_distance = f64::INFINITY;
            for k in 0..kernel_rows.len() {
                if used[k] {
                    continue;
                }
                let time_part = (kernel_rows[k][0] - matlab_row[0]).abs() / matlab_row[0].abs().max(1.0);
                let freq_part = (kernel_rows[k][1] - matlab_row[1]).abs() / matlab_row[1].abs().max(1.0);
                let distance = time_part + freq_part;
                if distance < best_distance {
                    best_distance = distance;
                    best_index = k;
                }
            }
            assert!(
                best_distance <= 1e-6,
                "segment {}: MATLAB peak at t={} f={} has no kernel partner",
                segment,
                matlab_row[0],
                matlab_row[1]
            );
            used[best_index] = true;

            let names = [
                "Height", "Volume", "Duration", "Bandwidth",
                "BoundingBox time corner", "BoundingBox frequency corner",
                "BoundingBox width", "BoundingBox height",
            ];
            for property in 0..names.len() {
                let kernel_value = kernel_rows[best_index][2 + property];
                let matlab_value = matlab_row[2 + property];
                let relative = (kernel_value - matlab_value).abs() / matlab_value.abs().max(f64::EPSILON);
                assert!(
                    relative <= 1e-9,
                    "segment {} peak at t={}: {} is {} in the kernel, {} in MATLAB",
                    segment,
                    matlab_row[0],
                    names[property],
                    kernel_value,
                    matlab_value
                );
            }
            n_peaks_checked += 1;
        }
    }
    assert!(n_peaks_checked >= 20, "only {} peaks in the fixture", n_peaks_checked);
}

/// `computePeakStage.m`, `computePeakSOpower.m`, `computePeakSOphase.m`.
/// The queries include samples on the grid, exact midpoints, and points
/// around both ends of each grid, where the rules are easiest to get
/// wrong.
///
/// Tolerance: stage exact. SO-power and SO-phase within 1e-9, and NaN in
/// the same places.
#[test]
fn per_peak_assignment_matches_matlab() {
    let fs = read_f64("fs")[0];
    let stage_times = read_f64("stage_times");
    let stage_vals = read_f64("stage_vals");
    let artifacts_raw = read_f64("artifacts");
    let so_power = read_f64("so_power");
    let so_power_times = read_f64("so_power_times");
    let so_phase = read_f64("so_phase");
    let query_times = read_f64("query_times");
    let matlab_stage = read_f64("query_stage");
    let matlab_so_power = read_f64("query_so_power");
    let matlab_so_phase = read_f64("query_so_phase");

    let eeg_times = sample_times(artifacts_raw.len(), fs);
    let mut artifacts = Vec::with_capacity(artifacts_raw.len());
    for value in &artifacts_raw {
        artifacts.push(*value != 0.0);
    }

    let kernel_stage = assign_peak_stage(&query_times, &stage_times, &stage_vals, &eeg_times, &artifacts);
    let kernel_so_power = assign_peak_sopower(&query_times, &so_power_times, &so_power);
    let kernel_so_phase = assign_peak_sophase(&query_times, &eeg_times, &so_phase);

    for i in 0..query_times.len() {
        assert_eq!(
            kernel_stage[i], matlab_stage[i],
            "stage at t={}: kernel {}, MATLAB {}",
            query_times[i], kernel_stage[i], matlab_stage[i]
        );
        assert_same_or_both_nan("SO-power", query_times[i], kernel_so_power[i], matlab_so_power[i]);
        assert_same_or_both_nan("SO-phase", query_times[i], kernel_so_phase[i], matlab_so_phase[i]);
    }
}

fn assert_same_or_both_nan(name: &str, time: f64, kernel_value: f64, matlab_value: f64) {
    if kernel_value.is_nan() || matlab_value.is_nan() {
        assert!(
            kernel_value.is_nan() && matlab_value.is_nan(),
            "{} at t={}: kernel {}, MATLAB {}",
            name, time, kernel_value, matlab_value
        );
        return;
    }
    assert!(
        (kernel_value - matlab_value).abs() <= 1e-9,
        "{} at t={}: kernel {}, MATLAB {}",
        name, time, kernel_value, matlab_value
    );
}
