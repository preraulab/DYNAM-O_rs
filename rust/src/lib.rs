//! dynamo_rs: the Rust numerical kernel for DYNAM-O.
//!
//! The crate covers the whole per-recording computation: EDF and staging
//! readers (`io`), artifact detection (`artifacts`), the multitaper
//! spectrogram presets (`mts`), baseline (`baseline`), TF-peak extraction
//! (`matlab_watershed`, `adjacency`, `merge`, `trim`, `extract_pipeline`),
//! peak refinement (`refine`), SO-power and SO-phase (`so_power`,
//! `so_phase`), per-peak assignment (`peak_assign`), histograms
//! (`histogram`, `histpeaks`), the SOPH fits (`paramfit`, `spline_basis`),
//! and group statistics (`stats`).
//!
//! Three front ends expose it: the C ABI in `c_api` (MATLAB MEX), the PyO3
//! module below (behind the `python` feature), and the `dynamo` binary.

pub mod adjacency;
pub mod artifacts;
pub mod baseline;
pub mod build_info;
pub mod c_api;
pub mod extract_pipeline;
pub mod filter_cache;
pub mod filter_design;
pub mod fits;
pub mod fits_table;
pub mod histogram;
pub mod histpeaks;
pub mod io;
pub mod mask;
pub mod matlab_watershed;
pub mod mts;
pub mod merge;
pub mod mode_peaks;
pub mod parallel;
pub mod peak_assign;
pub mod pipeline;
pub mod refine;
pub mod run;
pub mod signal;
pub mod paramfit;
pub mod so_phase;
pub mod so_power;
pub mod spline_basis;
pub mod stats;
pub mod trim;

#[cfg(feature = "python")]
mod python {
    use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
    use pyo3::prelude::*;

    /// Copy an `(n_sections, 6)` SOS matrix into one `[b0 b1 b2 a0 a1 a2]`
    /// row per section. The caller checks that there are 6 columns.
    fn sos_rows(sos: numpy::ndarray::ArrayView2<'_, f64>) -> Vec<[f64; 6]> {
        let mut rows: Vec<[f64; 6]> = Vec::with_capacity(sos.nrows());
        for section in 0..sos.nrows() {
            let mut coeffs = [0.0_f64; 6];
            for (col, coeff) in coeffs.iter_mut().enumerate() {
                *coeff = sos[[section, col]];
            }
            rows.push(coeffs);
        }
        rows
    }

    /// Store `values` in `dict` under `name` as a 1-D numpy array.
    fn set_vec_item<'py, T: numpy::Element>(
        py: Python<'py>,
        dict: &Bound<'py, pyo3::types::PyDict>,
        name: &str,
        values: Vec<T>,
    ) -> PyResult<()> {
        let array = ndarray::Array1::from(values).into_pyarray_bound(py);
        dict.set_item(name, array)
    }

    /// merge_segment(label_img, data, merge_thresh=8.0, max_merges=inf)
    #[pyfunction]
    #[pyo3(signature = (label_img, data, merge_thresh=8.0, max_merges=f64::INFINITY))]
    fn merge_segment<'py>(
        py: Python<'py>,
        label_img: PyReadonlyArray2<'py, i64>,
        data: PyReadonlyArray2<'py, f64>,
        merge_thresh: f64,
        max_merges: f64,
    ) -> PyResult<Bound<'py, PyArray2<i32>>> {
        let labels = label_img.as_array();
        let values = data.as_array();
        if labels.shape() != values.shape() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "labels and data must have the same shape",
            ));
        }
        let out = super::merge::run(labels, values, merge_thresh, max_merges)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(out.into_pyarray_bound(py))
    }

    /// merge_segment_with_borders(label_img, data, merge_thresh, max_merges)
    /// → (interior_only, with_borders)
    ///
    /// Returns two label images after merge: interior_only (0 on all watershed
    /// lines, for masking) and with_borders (each region's interior + its
    /// claimed border pixels painted, for dur/bw filter bbox computation).
    /// Matches MATLAB's dual semantics.
    #[pyfunction]
    #[pyo3(signature = (label_img, data, merge_thresh=8.0, max_merges=f64::INFINITY))]
    fn merge_segment_with_borders<'py>(
        py: Python<'py>,
        label_img: PyReadonlyArray2<'py, i64>,
        data: PyReadonlyArray2<'py, f64>,
        merge_thresh: f64,
        max_merges: f64,
    ) -> PyResult<(Bound<'py, PyArray2<i32>>, Bound<'py, PyArray2<i32>>)> {
        let labels = label_img.as_array();
        let values = data.as_array();
        if labels.shape() != values.shape() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "labels and data must have the same shape",
            ));
        }
        let (interior, with_borders) = super::merge::run_with_borders(
            labels, values, merge_thresh, max_merges,
        ).map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok((interior.into_pyarray_bound(py), with_borders.into_pyarray_bound(py)))
    }

    /// trim_regions(labels, data, vol_thresh=0.8, shift_val=None) -> int32 labels
    ///
    /// Trim each watershed region to `vol_thresh` of its peak volume.
    /// shift_val=None → use min(data).
    #[pyfunction]
    #[pyo3(signature = (labels, data, vol_thresh=0.8, shift_val=None))]
    fn trim_regions<'py>(
        py: Python<'py>,
        labels: PyReadonlyArray2<'py, i32>,
        data: PyReadonlyArray2<'py, f64>,
        vol_thresh: f64,
        shift_val: Option<f64>,
    ) -> PyResult<Bound<'py, PyArray2<i32>>> {
        let labels_view = labels.as_array();
        let data_view = data.as_array();
        if labels_view.shape() != data_view.shape() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "labels and data shape mismatch",
            ));
        }
        let shift = match shift_val {
            Some(value) => value,
            None => {
                // Default shift: the minimum of the data.
                let data_slice = data_view.as_slice().ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err("data must be C-contiguous")
                })?;
                let mut data_min = f64::INFINITY;
                for &value in data_slice {
                    data_min = f64::min(data_min, value);
                }
                data_min
            }
        };
        let out = super::trim::trim_all_regions(labels_view, data_view, vol_thresh, shift)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(out.into_pyarray_bound(py))
    }

    /// matlab_watershed(data) -> uint16 labels (same shape as data)
    ///
    /// Calls the MATLAB-Coder-generated watershed (Vincent-Soille via
    /// FifoPriorityQueue, 8-connectivity). Bit-identical to MATLAB's IPT
    /// `watershed()`. Input must be 2D float64 C-contiguous.
    #[pyfunction]
    fn matlab_watershed<'py>(
        py: Python<'py>,
        data: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, numpy::PyArray2<u16>>> {
        let arr = data.as_array();
        let out = super::matlab_watershed::matlab_watershed_2d(arr);
        Ok(out.into_pyarray_bound(py))
    }

    /// matlab_paint_labels(labels) -> int64 labels (same shape)
    ///
    /// Port of MATLAB `extractTFPeaks.m:272` + `Ldata2graph.m:233`: for each
    /// label in ascending order, 8-connectivity 1-pixel dilate its pixels
    /// and paint the *dense* 1..N cell index into the output. Higher labels
    /// overwrite lower on overlap (MATLAB-exact). Zero stays background.
    ///
    /// This is the default border-fill step. It keeps pydynamo's pass-2
    /// peak count within about 0.8% of MATLAB, where
    /// `skimage.segmentation.expand_labels(distance=5)` gives about +2%.
    /// Input must be 2D int64 C-contiguous. Output is int64, same shape.
    #[pyfunction]
    fn matlab_paint_labels<'py>(
        py: Python<'py>,
        labels: PyReadonlyArray2<'py, i64>,
    ) -> PyResult<Bound<'py, PyArray2<i64>>> {
        let arr = labels.as_array();
        let out = super::extract_pipeline::matlab_paint_labels_in_order(arr);
        Ok(out.into_pyarray_bound(py))
    }

    /// so_power_from_spectrogram(so_spect, stimes, sfreqs, eeg_times, isexcluded,
    ///                           stage_times, stage_vals, time_range,
    ///                           outlier_threshold, norm_method, retain_fs)
    ///   → (so_power_norm, so_power_times, so_power_stages, ptile)
    ///
    /// Post-spectrogram SO-power pipeline, a port of computeSOpower.m steps
    /// 3–8 and pydynamo soph/sopower.py lines ~93–164. Takes the
    /// already-computed multitaper spectrogram over the SO band, emits the
    /// normalized SO-power time series on either the window-center grid
    /// (`retain_fs=False`) or upsampled back to `eeg_times` (`retain_fs=True`).
    ///
    /// `ptile` is:
    ///   * None for norm_method='none'/'absolute' or the degenerate all-NaN path
    ///   * float for p{N}shift{S} (the ptile-th percentile used as the shift)
    ///   * (float, float) for 'percent' (p1, p99)
    #[pyfunction]
    #[pyo3(signature = (
        so_spect, stimes, sfreqs, eeg_times, isexcluded,
        stage_times, stage_vals, time_range,
        outlier_threshold=3.0, norm_method="p2shift1234",
        retain_fs=true,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn so_power_from_spectrogram<'py>(
        py: Python<'py>,
        so_spect: PyReadonlyArray2<'py, f64>,
        stimes: PyReadonlyArray1<'py, f64>,
        sfreqs: PyReadonlyArray1<'py, f64>,
        eeg_times: PyReadonlyArray1<'py, f64>,
        isexcluded: PyReadonlyArray1<'py, bool>,
        stage_times: PyReadonlyArray1<'py, f64>,
        stage_vals: PyReadonlyArray1<'py, f64>,
        time_range: (f64, f64),
        outlier_threshold: f64,
        norm_method: &str,
        retain_fs: bool,
    ) -> PyResult<Py<pyo3::types::PyTuple>> {
        use super::so_power::{so_power_from_spectrogram as rs_fn, NormMethod, PtileUsed};

        let spect_arr = so_spect.as_array();
        let (n_freqs, n_times) = spect_arr.dim();
        let spect_contig = spect_arr.to_owned();
        let spect_slice = spect_contig.as_slice().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("so_spect must be C-contiguous")
        })?;

        let stimes_slice = stimes.as_array().to_owned().into_raw_vec_and_offset().0;
        let sfreqs_slice = sfreqs.as_array().to_owned().into_raw_vec_and_offset().0;
        let eeg_times_slice = eeg_times.as_array().to_owned().into_raw_vec_and_offset().0;
        let isexcluded_slice: Vec<bool> = isexcluded.as_array().iter().copied().collect();
        let stage_times_slice = stage_times.as_array().to_owned().into_raw_vec_and_offset().0;
        let stage_vals_slice = stage_vals.as_array().to_owned().into_raw_vec_and_offset().0;

        let norm = NormMethod::parse(norm_method).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "unrecognized norm_method {:?}",
                norm_method
            ))
        })?;

        let out = rs_fn(
            spect_slice, n_freqs, n_times,
            &stimes_slice, &sfreqs_slice,
            &eeg_times_slice, &isexcluded_slice,
            &stage_times_slice, &stage_vals_slice,
            time_range, outlier_threshold, &norm, retain_fs,
        )
        .map_err(pyo3::exceptions::PyValueError::new_err)?;

        let arr_norm = numpy::ndarray::Array1::from(out.so_power_norm).into_pyarray_bound(py);
        let arr_times = numpy::ndarray::Array1::from(out.so_power_times).into_pyarray_bound(py);
        let arr_stages = numpy::ndarray::Array1::from(out.so_power_stages).into_pyarray_bound(py);
        let ptile_obj: PyObject = match out.ptile {
            None => py.None(),
            Some(PtileUsed::Single(p)) => p.into_py(py),
            Some(PtileUsed::Pair(a, b)) => (a, b).into_py(py),
        };
        let tup = pyo3::types::PyTuple::new_bound(py, [
            arr_norm.into_any(), arr_times.into_any(), arr_stages.into_any(),
            ptile_obj.into_bound(py),
        ]);
        Ok(tup.unbind())
    }

    /// so_phase_from_eeg(eeg, eeg_times, isexcluded, sos, stage_times, stage_vals)
    ///   → (so_phase_unwrapped, so_phase_times, so_phase_stages, filtdata)
    ///
    /// Port of computeSOphase.m / pydynamo compute_so_phase. Band-pass SOS
    /// filter via sosfiltfilt → Hilbert analytic → atan2 → unwrap → NaN mask
    /// at excluded samples → stage assign (previous-interp). Returns phase
    /// **unwrapped** (wrapToPi applied downstream during binning).
    ///
    /// `sos` is shape (n_sections, 6) float64 in scipy order [b0 b1 b2 a0 a1 a2].
    /// Use `dynamo_rs.get_sophase_sos(fs, band)` (or scipy's iirdesign) to
    /// obtain it.
    #[pyfunction]
    fn so_phase_from_eeg<'py>(
        py: Python<'py>,
        eeg: PyReadonlyArray1<'py, f64>,
        eeg_times: PyReadonlyArray1<'py, f64>,
        isexcluded: PyReadonlyArray1<'py, bool>,
        sos: PyReadonlyArray2<'py, f64>,
        stage_times: PyReadonlyArray1<'py, f64>,
        stage_vals: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Py<pyo3::types::PyTuple>> {
        let eeg_vec = eeg.as_array().to_owned().into_raw_vec_and_offset().0;
        let eeg_times_vec = eeg_times.as_array().to_owned().into_raw_vec_and_offset().0;
        let isexcluded_vec: Vec<bool> = isexcluded.as_array().iter().copied().collect();
        let stage_times_vec = stage_times.as_array().to_owned().into_raw_vec_and_offset().0;
        let stage_vals_vec = stage_vals.as_array().to_owned().into_raw_vec_and_offset().0;

        let sos_arr = sos.as_array();
        if sos_arr.ncols() != 6 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "sos must have 6 columns (scipy SOS format: [b0 b1 b2 a0 a1 a2]).",
            ));
        }
        let sos_slice = sos_rows(sos_arr);

        let out = super::so_phase::so_phase_from_eeg(
            &eeg_vec, &eeg_times_vec, &isexcluded_vec, &sos_slice,
            &stage_times_vec, &stage_vals_vec,
        )
        .map_err(pyo3::exceptions::PyValueError::new_err)?;

        let a_phase = numpy::ndarray::Array1::from(out.so_phase_unwrapped).into_pyarray_bound(py);
        let a_times = numpy::ndarray::Array1::from(out.so_phase_times).into_pyarray_bound(py);
        let a_stages = numpy::ndarray::Array1::from(out.so_phase_stages).into_pyarray_bound(py);
        let a_filt = numpy::ndarray::Array1::from(out.filtdata).into_pyarray_bound(py);
        let tup = pyo3::types::PyTuple::new_bound(py, [
            a_phase.into_any(), a_times.into_any(), a_stages.into_any(), a_filt.into_any(),
        ]);
        Ok(tup.unbind())
    }

    /// detect_artifacts(data, fs, hf_pass, hf_crit, bb_pass, bb_crit,
    ///                   hf_detrend, bb_detrend, zscore_method,
    ///                   smooth_duration, detrend_duration, buffer_duration)
    ///   → (N,) bool mask
    ///
    /// Port of pydynamo `detect_artifacts(..., slope_test=False)` and the
    /// corresponding MATLAB detect_artifacts.m (slope-test branch disabled).
    /// Default params match pydynamo's defaults.
    #[pyfunction]
    #[pyo3(signature = (
        data, fs,
        hf_pass=35.0, hf_crit=5.5, bb_pass=0.1, bb_crit=5.5,
        hf_detrend=true, bb_detrend=true, zscore_method="robust",
        smooth_duration=2.0, detrend_duration=300.0, buffer_duration=0.0,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn detect_artifacts<'py>(
        py: Python<'py>,
        data: PyReadonlyArray1<'py, f64>,
        fs: f64,
        hf_pass: f64, hf_crit: f64, bb_pass: f64, bb_crit: f64,
        hf_detrend: bool, bb_detrend: bool,
        zscore_method: &str,
        smooth_duration: f64, detrend_duration: f64, buffer_duration: f64,
    ) -> PyResult<Bound<'py, numpy::PyArray1<bool>>> {
        use super::artifacts::{detect_artifacts as rs_fn, ArtifactOpts, ZScoreMethod};
        let zscore = match zscore_method.to_ascii_lowercase().as_str() {
            "robust" => ZScoreMethod::Robust,
            "standard" => ZScoreMethod::Standard,
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "zscore_method must be 'robust' or 'standard', got {:?}",
                    other
                )));
            }
        };
        let opts = ArtifactOpts {
            hf_pass, hf_crit, bb_pass, bb_crit,
            hf_detrend, bb_detrend,
            smooth_duration, detrend_duration, buffer_duration,
            zscore_method: zscore,
        };
        let data_vec = data.as_array().to_owned().into_raw_vec_and_offset().0;
        let out = rs_fn(&data_vec, fs, &opts);
        Ok(numpy::ndarray::Array1::from(out).into_pyarray_bound(py))
    }

    /// build_baseline_exclude(t_data, stage_times, stage_vals, baseline_stages,
    ///                        artifacts, user_exclude=None) -> (N,) bool
    ///
    /// ORs together three masks: `user_exclude` (when given), samples whose
    /// stage is not in `baseline_stages`, and `artifacts`.
    /// Matches pydynamo pipeline.py:123-129.
    #[pyfunction]
    #[pyo3(signature = (
        t_data, stage_times, stage_vals, baseline_stages, artifacts,
        user_exclude=None,
    ))]
    fn build_baseline_exclude<'py>(
        py: Python<'py>,
        t_data: PyReadonlyArray1<'py, f64>,
        stage_times: PyReadonlyArray1<'py, f64>,
        stage_vals: PyReadonlyArray1<'py, f64>,
        baseline_stages: PyReadonlyArray1<'py, f64>,
        artifacts: PyReadonlyArray1<'py, bool>,
        user_exclude: Option<PyReadonlyArray1<'py, bool>>,
    ) -> PyResult<Bound<'py, numpy::PyArray1<bool>>> {
        let t_data_vec = t_data.as_array().to_owned().into_raw_vec_and_offset().0;
        let stage_times_vec = stage_times.as_array().to_owned().into_raw_vec_and_offset().0;
        let stage_vals_vec = stage_vals.as_array().to_owned().into_raw_vec_and_offset().0;
        let baseline_stages_vec =
            baseline_stages.as_array().to_owned().into_raw_vec_and_offset().0;
        let artifacts_vec: Vec<bool> = artifacts.as_array().iter().copied().collect();
        let user_exclude_vec: Option<Vec<bool>> = user_exclude
            .as_ref()
            .map(|mask| mask.as_array().iter().copied().collect());
        let out = super::baseline::build_baseline_exclude(
            &t_data_vec,
            &stage_times_vec,
            &stage_vals_vec,
            &baseline_stages_vec,
            &artifacts_vec,
            user_exclude_vec.as_deref(),
        )
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(numpy::ndarray::Array1::from(out).into_pyarray_bound(py))
    }

    /// compute_baseline(spect, stimes, t_data, baseline_exclude, baseline_range, baseline_ptile)
    /// → (F, 1) float64 baseline
    ///
    /// Port of pydynamo `baseline.compute_baseline`. Hyndman-Fan #5 percentile
    /// per frequency row over the valid (non-excluded, in-range) time columns,
    /// treating 0 pixels as NaN.
    #[pyfunction]
    #[pyo3(signature = (
        spect, stimes, t_data, baseline_exclude,
        baseline_range=(f64::NEG_INFINITY, f64::INFINITY),
        baseline_ptile=2.0,
    ))]
    fn compute_baseline<'py>(
        py: Python<'py>,
        spect: PyReadonlyArray2<'py, f64>,
        stimes: PyReadonlyArray1<'py, f64>,
        t_data: PyReadonlyArray1<'py, f64>,
        baseline_exclude: PyReadonlyArray1<'py, bool>,
        baseline_range: (f64, f64),
        baseline_ptile: f64,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let excl = baseline_exclude.as_array();
        let excl_vec: Vec<bool> = excl.iter().copied().collect();
        let out = super::baseline::compute_baseline(
            spect.as_array(),
            stimes.as_array(),
            t_data.as_array(),
            &excl_vec,
            baseline_range,
            baseline_ptile,
        )
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(out.into_pyarray_bound(py))
    }

    /// mask_spectrogram(spect_2s, stimes_2s, labels_1s, stimes_1s) → masked (F, T2) f64.
    ///
    /// Port of pydynamo `mask_spectrogram`: nearest-stime lookup from pass-2 cols
    /// to pass-1 cols, paint spect where labels>0, zero the 1-pixel inner
    /// perimeter (8-conn) of each label.
    #[pyfunction]
    fn mask_spectrogram<'py>(
        py: Python<'py>,
        spect_2s: PyReadonlyArray2<'py, f64>,
        stimes_2s: PyReadonlyArray1<'py, f64>,
        labels_1s: PyReadonlyArray2<'py, i64>,
        stimes_1s: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let out = super::mask::mask_spectrogram(
            spect_2s.as_array(),
            stimes_2s.as_array(),
            labels_1s.as_array(),
            stimes_1s.as_array(),
        )
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(out.into_pyarray_bound(py))
    }

    /// subtract_baseline(spect, baseline) → spect / baseline (column broadcast).
    #[pyfunction]
    fn subtract_baseline<'py>(
        py: Python<'py>,
        spect: PyReadonlyArray2<'py, f64>,
        baseline: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let out = super::baseline::subtract_baseline(spect.as_array(), baseline.as_array())
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(out.into_pyarray_bound(py))
    }

    /// tfpeak_histogram(...) → dict with c_mat, time_in_bin, prop_in_bin, peak_at_freq.
    ///
    /// Port of pydynamo `soph.histogram.tfpeak_histogram`. Stage labels 1..=5.
    #[pyfunction]
    #[pyo3(signature = (
        c_metric, c_stages, c_dt,
        c_valid, c_valid_allstages,
        peak_freqs, peak_c,
        freq_edges, c_edges,
        circular, circular_bounds,
        norm_dim=0,
        compute_rate=true,
        min_time_in_bin=0.0,
        min_peak_at_freq=0,
    ))]
    fn tfpeak_histogram<'py>(
        py: Python<'py>,
        c_metric: PyReadonlyArray1<'py, f64>,
        c_stages: PyReadonlyArray1<'py, f64>,
        c_dt: f64,
        c_valid: PyReadonlyArray1<'py, bool>,
        c_valid_allstages: PyReadonlyArray1<'py, bool>,
        peak_freqs: PyReadonlyArray1<'py, f64>,
        peak_c: PyReadonlyArray1<'py, f64>,
        freq_edges: PyReadonlyArray2<'py, f64>,
        c_edges: PyReadonlyArray2<'py, f64>,
        circular: bool,
        circular_bounds: (f64, f64),
        norm_dim: i32,
        compute_rate: bool,
        min_time_in_bin: f64,
        min_peak_at_freq: i32,
    ) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let c_valid_vec: Vec<bool> = c_valid.as_array().iter().copied().collect();
        let c_valid_allstages_vec: Vec<bool> =
            c_valid_allstages.as_array().iter().copied().collect();
        let inp = super::histogram::HistogramInputs {
            c_metric: c_metric.as_array(),
            c_stages: c_stages.as_array(),
            c_dt,
            c_valid: &c_valid_vec,
            c_valid_allstages: &c_valid_allstages_vec,
            peak_freqs: peak_freqs.as_array(),
            peak_c: peak_c.as_array(),
            freq_edges: freq_edges.as_array(),
            c_edges: c_edges.as_array(),
            circular,
            circular_bounds,
            norm_dim,
            compute_rate,
            min_time_in_bin,
            min_peak_at_freq,
        };
        let out = super::histogram::tfpeak_histogram(&inp)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        let dict = pyo3::types::PyDict::new_bound(py);
        dict.set_item("c_mat", out.c_mat.into_pyarray_bound(py))?;
        dict.set_item("time_in_bin", out.time_in_bin.into_pyarray_bound(py))?;
        dict.set_item("prop_in_bin", out.prop_in_bin.into_pyarray_bound(py))?;
        dict.set_item("peak_at_freq", out.peak_at_freq.into_pyarray_bound(py))?;
        Ok(dict)
    }

    /// hann_event_spectra(data, fs, event_times, t0, freq_range, window_size, dsfreqs, detrend_opt)
    /// → (spect (F, N), sfreqs (F,))
    ///
    /// One Hann-windowed spectrum per event, centered on each event time.
    /// Column `i` of `spect` is the spectrum for `event_times[i]`.
    #[pyfunction]
    #[pyo3(signature = (
        data, fs, event_times, t0,
        freq_range=(0.0, 30.0), window_size=4.0, dsfreqs=0.05,
        detrend_opt="constant",
    ))]
    fn hann_event_spectra<'py>(
        py: Python<'py>,
        data: PyReadonlyArray1<'py, f64>,
        fs: f64,
        event_times: PyReadonlyArray1<'py, f64>,
        t0: f64,
        freq_range: (f64, f64),
        window_size: f64,
        dsfreqs: f64,
        detrend_opt: &str,
    ) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, numpy::PyArray1<f64>>)> {
        let data_vec: Vec<f64> = data.as_array().iter().copied().collect();
        let event_vec: Vec<f64> = event_times.as_array().iter().copied().collect();
        let detrend = match detrend_opt {
            "constant" => super::refine::DetrendOpt::Constant,
            "linear" => super::refine::DetrendOpt::Linear,
            "off" | "none" => super::refine::DetrendOpt::None,
            other => return Err(pyo3::exceptions::PyValueError::new_err(
                format!("unknown detrend_opt {:?}", other)
            )),
        };
        let (spect, sfreqs) = super::refine::hann_event_spectra(
            &data_vec, fs, &event_vec, t0, freq_range, window_size, dsfreqs, detrend,
        );
        let sfreqs_arr = ndarray::Array1::from(sfreqs);
        Ok((spect.into_pyarray_bound(py), sfreqs_arr.into_pyarray_bound(py)))
    }

    /// refine_from_spectra(spect, sfreqs, bbox_lo, bbox_hi, n_grid=1000, remove_edge_peaks=true)
    /// → refined_freqs (N,)
    ///
    /// Refine each peak frequency by locating the spectral maximum inside
    /// its bounding-box frequency range `[bbox_lo[i], bbox_hi[i]]`, on an
    /// `n_grid`-point interpolation grid.
    #[pyfunction]
    #[pyo3(signature = (spect, sfreqs, bbox_lo, bbox_hi, n_grid=1000, remove_edge_peaks=true))]
    fn refine_from_spectra<'py>(
        py: Python<'py>,
        spect: PyReadonlyArray2<'py, f64>,
        sfreqs: PyReadonlyArray1<'py, f64>,
        bbox_lo: PyReadonlyArray1<'py, f64>,
        bbox_hi: PyReadonlyArray1<'py, f64>,
        n_grid: usize,
        remove_edge_peaks: bool,
    ) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
        let sfreqs_vec: Vec<f64> = sfreqs.as_array().iter().copied().collect();
        let lo_vec: Vec<f64> = bbox_lo.as_array().iter().copied().collect();
        let hi_vec: Vec<f64> = bbox_hi.as_array().iter().copied().collect();
        let spect_owned = spect.as_array().to_owned();
        let out = super::refine::refine_from_spectra(
            &spect_owned, &sfreqs_vec, &lo_vec, &hi_vec, n_grid, remove_edge_peaks,
        );
        Ok(ndarray::Array1::from(out).into_pyarray_bound(py))
    }

    /// sosfiltfilt(sos (M, 6), x (N,)) → (N,) zero-phase filtered signal.
    /// Follows MATLAB filtfilt(sos, 1, x): each section is run forward and
    /// backward on its own odd-reflected extension. Differs from
    /// scipy.signal.sosfiltfilt near the ends of the signal.
    #[pyfunction]
    fn sosfiltfilt<'py>(
        py: Python<'py>,
        sos: PyReadonlyArray2<'py, f64>,
        x: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
        let sos_arr = sos.as_array();
        if sos_arr.ncols() != 6 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "sos must have 6 columns (b0 b1 b2 a0 a1 a2)",
            ));
        }
        let sos_vec = sos_rows(sos_arr);
        let x_vec: Vec<f64> = x.as_array().iter().copied().collect();
        let y = super::signal::sosfiltfilt(&sos_vec, &x_vec);
        Ok(ndarray::Array1::from(y).into_pyarray_bound(py))
    }

    /// hilbert(x) → (re, im) analytic signal real/imag parts.
    #[pyfunction]
    fn hilbert<'py>(
        py: Python<'py>,
        x: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<(Bound<'py, numpy::PyArray1<f64>>, Bound<'py, numpy::PyArray1<f64>>)> {
        let x_vec: Vec<f64> = x.as_array().iter().copied().collect();
        let (re, im) = super::signal::hilbert(&x_vec);
        Ok((
            ndarray::Array1::from(re).into_pyarray_bound(py),
            ndarray::Array1::from(im).into_pyarray_bound(py),
        ))
    }

    /// unwrap(p, discont=π), matching numpy.unwrap.
    #[pyfunction]
    #[pyo3(signature = (p, discont=std::f64::consts::PI))]
    fn unwrap<'py>(
        py: Python<'py>,
        p: PyReadonlyArray1<'py, f64>,
        discont: f64,
    ) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
        let phase_vec: Vec<f64> = p.as_array().iter().copied().collect();
        let out = super::signal::unwrap(&phase_vec, discont);
        Ok(ndarray::Array1::from(out).into_pyarray_bound(py))
    }

    /// movmean(x, win): MATLAB-style centered moving mean with partial edges.
    #[pyfunction]
    fn movmean<'py>(
        py: Python<'py>,
        x: PyReadonlyArray1<'py, f64>,
        win: usize,
    ) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
        let x_vec: Vec<f64> = x.as_array().iter().copied().collect();
        let out = super::signal::movmean(&x_vec, win);
        Ok(ndarray::Array1::from(out).into_pyarray_bound(py))
    }

    /// read_edf(path, channel=None) → dict
    ///
    /// The dict always has `header`, `labels` and `sampling_frequencies`.
    /// If `channel` is None it also has `signals`, a list with one dict per
    /// signal (header fields plus `data`). Otherwise it has `data`, `fs`,
    /// `label` and `signal_header` for the single selected channel (with
    /// A–B rereference support). Port of `read_EDF_mex.c`.
    #[pyfunction]
    #[pyo3(signature = (path, channel=None))]
    fn read_edf<'py>(
        py: Python<'py>,
        path: &str,
        channel: Option<&str>,
    ) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        use pyo3::types::{PyDict, PyList};
        let edf = super::io::edf::read_edf_all(path)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;

        let out = PyDict::new_bound(py);

        // Header
        let header_dict = PyDict::new_bound(py);
        header_dict.set_item("edf_ver", &edf.header.edf_ver)?;
        header_dict.set_item("patient_id", &edf.header.patient_id)?;
        header_dict.set_item("local_rec_id", &edf.header.local_rec_id)?;
        header_dict.set_item("recording_startdate", &edf.header.recording_startdate)?;
        header_dict.set_item("recording_starttime", &edf.header.recording_starttime)?;
        header_dict.set_item("num_header_bytes", edf.header.num_header_bytes)?;
        header_dict.set_item("num_data_records", edf.header.num_data_records)?;
        header_dict.set_item("data_record_duration", edf.header.data_record_duration)?;
        header_dict.set_item("num_signals", edf.header.num_signals)?;
        out.set_item("header", header_dict)?;

        // All signal labels (useful even when selecting one channel).
        let labels: Vec<&str> = edf.signals.iter().map(|signal| signal.signal_labels.as_str()).collect();
        out.set_item("labels", labels)?;
        let fs_all: Vec<f64> = edf.signals.iter().map(|signal| signal.sampling_frequency).collect();
        out.set_item("sampling_frequencies", fs_all)?;

        if let Some(channel_expr) = channel {
            let (signal_header, values) = super::io::edf::select_channel(&edf, channel_expr)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;
            let arr = ndarray::Array1::from(values);
            out.set_item("data", arr.into_pyarray_bound(py))?;
            out.set_item("fs", signal_header.sampling_frequency)?;
            out.set_item("label", signal_header.signal_labels.clone())?;
            let signal_info = PyDict::new_bound(py);
            signal_info.set_item("signal_labels", signal_header.signal_labels)?;
            signal_info.set_item("transducer_type", signal_header.transducer_type)?;
            signal_info.set_item("physical_dimension", signal_header.physical_dimension)?;
            signal_info.set_item("physical_min", signal_header.physical_min)?;
            signal_info.set_item("physical_max", signal_header.physical_max)?;
            signal_info.set_item("digital_min", signal_header.digital_min)?;
            signal_info.set_item("digital_max", signal_header.digital_max)?;
            signal_info.set_item("prefiltering", signal_header.prefiltering)?;
            signal_info.set_item("samples_in_record", signal_header.samples_in_record)?;
            signal_info.set_item("sampling_frequency", signal_header.sampling_frequency)?;
            out.set_item("signal_header", signal_info)?;
        } else {
            let signal_list = PyList::empty_bound(py);
            for (signal, samples) in edf.signals.iter().zip(edf.data.iter()) {
                let signal_dict = PyDict::new_bound(py);
                signal_dict.set_item("signal_labels", &signal.signal_labels)?;
                signal_dict.set_item("transducer_type", &signal.transducer_type)?;
                signal_dict.set_item("physical_dimension", &signal.physical_dimension)?;
                signal_dict.set_item("physical_min", signal.physical_min)?;
                signal_dict.set_item("physical_max", signal.physical_max)?;
                signal_dict.set_item("digital_min", signal.digital_min)?;
                signal_dict.set_item("digital_max", signal.digital_max)?;
                signal_dict.set_item("prefiltering", &signal.prefiltering)?;
                signal_dict.set_item("samples_in_record", signal.samples_in_record)?;
                signal_dict.set_item("sampling_frequency", signal.sampling_frequency)?;
                let arr = ndarray::Array1::from(samples.clone());
                signal_dict.set_item("data", arr.into_pyarray_bound(py))?;
                signal_list.append(signal_dict)?;
            }
            out.set_item("signals", signal_list)?;
        }

        Ok(out)
    }

    /// read_staging(path, time_col=1, stage_col=2, header_lines=0,
    ///              delimiter=",", epoch_dur=30.0, start_time=None)
    /// → (times (N,), vals (N,)) numpy arrays.
    ///
    /// Read a delimited-text sleep-staging file. `time_col` and `stage_col`
    /// are 1-based column numbers. Port of `read_staging.m`.
    #[pyfunction]
    #[pyo3(signature = (
        path, time_col=1, stage_col=2, header_lines=0,
        delimiter=",", epoch_dur=30.0, start_time=None,
    ))]
    fn read_staging<'py>(
        py: Python<'py>,
        path: &str,
        time_col: usize,
        stage_col: usize,
        header_lines: usize,
        delimiter: &str,
        epoch_dur: f64,
        start_time: Option<&str>,
    ) -> PyResult<(
        Bound<'py, numpy::PyArray1<f64>>,
        Bound<'py, numpy::PyArray1<f64>>,
    )> {
        let delim = delimiter.chars().next().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("delimiter must be at least one char")
        })?;
        let opts = super::io::staging::StagingOpts {
            time_col,
            stage_col,
            header_lines,
            delimiter: delim,
            epoch_dur,
            start_time: start_time.map(|s| s.to_string()),
            recording_duration: None,
            stage_vals: Vec::new(),
        };
        let out = super::io::staging::read_staging(path, &opts)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;
        let times = ndarray::Array1::from(out.times);
        let vals = ndarray::Array1::from(out.vals);
        Ok((times.into_pyarray_bound(py), vals.into_pyarray_bound(py)))
    }

    /// Unpack a length-3 background-coefficient array into `[f64; 3]`.
    fn bg3(name: &str, arr: PyReadonlyArray1<'_, f64>) -> PyResult<[f64; 3]> {
        let coeffs = arr.as_array();
        if coeffs.len() != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "{} must have 3 elements, got {}",
                name,
                coeffs.len()
            )));
        }
        Ok([coeffs[0], coeffs[1], coeffs[2]])
    }

    /// Pack a `ParamFitOut` into the dict both paramfit kernels return. The
    /// `gof` fields are flattened to top-level keys so the caller can build a
    /// MATLAB-style gof struct without a nested lookup.
    fn paramfit_dict<'py>(
        py: Python<'py>,
        out: super::paramfit::ParamFitOut,
    ) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let dict = pyo3::types::PyDict::new_bound(py);
        dict.set_item("params", out.params.into_pyarray_bound(py))?;
        dict.set_item(
            "background",
            ndarray::Array1::from(out.background.to_vec()).into_pyarray_bound(py),
        )?;
        dict.set_item("model_soph", out.model_soph.into_pyarray_bound(py))?;
        dict.set_item("sse", out.gof.sse)?;
        dict.set_item("rsquare", out.gof.rsquare)?;
        dict.set_item("adjrsquare", out.gof.adjrsquare)?;
        dict.set_item("rmse", out.gof.rmse)?;
        dict.set_item("dfe", out.gof.dfe)?;
        dict.set_item("dfm", out.gof.dfm)?;
        dict.set_item("iters_used", out.iters_used)?;
        Ok(dict)
    }

    /// fit_rotgauss(soph, x_grid, y_grid, initial, lower, upper,
    ///              bg_initial, bg_lower, bg_upper, max_iters=0)
    /// → dict{params (N,6), background (3,), model_soph (n_y,n_x),
    ///        sse, rsquare, adjrsquare, rmse, dfe, dfm, iters_used}
    ///
    /// Rotated-Gaussian mixture + linear background plane, mirroring MATLAB
    /// `fit_rotGauss.m`. `soph` is `(n_y, n_x)` = `(n_freqs, n_features)`;
    /// `initial`/`lower`/`upper` are `(N, 6)` with columns
    /// `[amp, fmean, fstd, pmean, pstd, theta]`. `max_iters=0` selects the
    /// scipy-equivalent default of `100 * n_params`.
    ///
    /// The kernel is
    /// `amp * exp(-0.5*(u/fstd)^2 - 0.5*(v/pstd)^2)`, so `fstd` and `pstd`
    /// are standard deviations, not variances. Widths stored by versions
    /// below 0.2.0 are `sqrt(2)` times larger. To reuse such widths as a
    /// modeled surface or bound window, divide columns 2 and 4 by `sqrt(2)`.
    /// Seeds and priors already expressed as standard deviations keep their
    /// numeric values. This function applies no conversion at its boundary.
    #[pyfunction]
    #[pyo3(signature = (soph, x_grid, y_grid, initial, lower, upper,
                        bg_initial, bg_lower, bg_upper, max_iters=0))]
    #[allow(clippy::too_many_arguments)]
    fn fit_rotgauss<'py>(
        py: Python<'py>,
        soph: PyReadonlyArray2<'py, f64>,
        x_grid: PyReadonlyArray1<'py, f64>,
        y_grid: PyReadonlyArray1<'py, f64>,
        initial: PyReadonlyArray2<'py, f64>,
        lower: PyReadonlyArray2<'py, f64>,
        upper: PyReadonlyArray2<'py, f64>,
        bg_initial: PyReadonlyArray1<'py, f64>,
        bg_lower: PyReadonlyArray1<'py, f64>,
        bg_upper: PyReadonlyArray1<'py, f64>,
        max_iters: u32,
    ) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let x_grid_vec: Vec<f64> = x_grid.as_array().iter().copied().collect();
        let y_grid_vec: Vec<f64> = y_grid.as_array().iter().copied().collect();
        let out = super::paramfit::rot_gauss::fit_rotgauss(
            soph.as_array(),
            &x_grid_vec,
            &y_grid_vec,
            initial.as_array(),
            lower.as_array(),
            upper.as_array(),
            bg3("bg_initial", bg_initial)?,
            bg3("bg_lower", bg_lower)?,
            bg3("bg_upper", bg_upper)?,
            max_iters,
        )
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;
        paramfit_dict(py, out)
    }

    /// fit_vmgauss(soph, x_grid, y_grid, initial, lower, upper,
    ///             bg_initial, bg_lower, bg_upper, max_iters=0, unit_row=True)
    /// → dict{params (N,6), background (3,), model_soph (n_y,n_x),
    ///        sse, rsquare, adjrsquare, rmse, dfe, dfm, iters_used}
    ///
    /// von-Mises × Gaussian mixture + sinusoidal baseline, mirroring MATLAB
    /// `fit_vmGauss.m`. Column layout of `initial`/`lower`/`upper` is
    /// `[amp, fmean, fstd, phasepref, recikappa, theta]`. `unit_row=True`
    /// replicates `normalized_vmGauss.m`'s per-frequency-row normalization
    /// (MATLAB passes `problem=true` for the phase fit).
    ///
    /// The kernel is `amp * exp(-0.5*(dy/fstd)^2) *
    /// exp(k*(cos(dx) - 1))`, `k = 1/recikappa^2`, so both widths are
    /// standard deviations. `fstd` values stored by versions below 0.2.0 are
    /// `sqrt(2)` times larger. To reuse them as a modeled surface or bound
    /// window, divide by `sqrt(2)`. Seeds and priors already expressed as
    /// standard deviations keep their numeric values. `recikappa` (column 4)
    /// must NOT be rescaled, because the von Mises factor is its own
    /// small-angle Gaussian and carries the half intrinsically. Column 4 is
    /// `pstd` for `fit_rotgauss` and `recikappa` here, so any conversion
    /// driven by column index instead of by kernel corrupts every phase fit.
    /// This function applies no conversion at its boundary.
    #[pyfunction]
    #[pyo3(signature = (soph, x_grid, y_grid, initial, lower, upper,
                        bg_initial, bg_lower, bg_upper, max_iters=0, unit_row=true))]
    #[allow(clippy::too_many_arguments)]
    fn fit_vmgauss<'py>(
        py: Python<'py>,
        soph: PyReadonlyArray2<'py, f64>,
        x_grid: PyReadonlyArray1<'py, f64>,
        y_grid: PyReadonlyArray1<'py, f64>,
        initial: PyReadonlyArray2<'py, f64>,
        lower: PyReadonlyArray2<'py, f64>,
        upper: PyReadonlyArray2<'py, f64>,
        bg_initial: PyReadonlyArray1<'py, f64>,
        bg_lower: PyReadonlyArray1<'py, f64>,
        bg_upper: PyReadonlyArray1<'py, f64>,
        max_iters: u32,
        unit_row: bool,
    ) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let x_grid_vec: Vec<f64> = x_grid.as_array().iter().copied().collect();
        let y_grid_vec: Vec<f64> = y_grid.as_array().iter().copied().collect();
        let out = super::paramfit::vm_gauss::fit_vmgauss(
            soph.as_array(),
            &x_grid_vec,
            &y_grid_vec,
            initial.as_array(),
            lower.as_array(),
            upper.as_array(),
            bg3("bg_initial", bg_initial)?,
            bg3("bg_lower", bg_lower)?,
            bg3("bg_upper", bg_upper)?,
            max_iters,
            unit_row,
        )
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;
        paramfit_dict(py, out)
    }

    /// fit_tensor_product_spline(soph, x_eval, y_eval, internal_knots_x,
    ///                           internal_knots_y, order=4,
    ///                           boundary_multiplicity=3)
    /// → dict{coefs (m_y,m_x), splinefit (n_x,n_y), knots_x_aug, knots_y_aug}
    ///
    /// Tensor-product cubic B-spline least squares, equivalent to MATLAB
    /// `spap2`. `soph` is `(n_x, n_y)`, the transpose of the canonical
    /// `(n_freqs, n_features)` layout, matching what `spline_basis.m` feeds
    /// `spap2`.
    #[pyfunction]
    #[pyo3(signature = (soph, x_eval, y_eval, internal_knots_x, internal_knots_y,
                        order=4, boundary_multiplicity=3))]
    fn fit_tensor_product_spline<'py>(
        py: Python<'py>,
        soph: PyReadonlyArray2<'py, f64>,
        x_eval: PyReadonlyArray1<'py, f64>,
        y_eval: PyReadonlyArray1<'py, f64>,
        internal_knots_x: PyReadonlyArray1<'py, f64>,
        internal_knots_y: PyReadonlyArray1<'py, f64>,
        order: usize,
        boundary_multiplicity: usize,
    ) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let x_eval_vec: Vec<f64> = x_eval.as_array().iter().copied().collect();
        let y_eval_vec: Vec<f64> = y_eval.as_array().iter().copied().collect();
        let knots_x_vec: Vec<f64> = internal_knots_x.as_array().iter().copied().collect();
        let knots_y_vec: Vec<f64> = internal_knots_y.as_array().iter().copied().collect();
        let out = super::spline_basis::fit_tensor_product_spline(
            soph.as_array(),
            &x_eval_vec,
            &y_eval_vec,
            &knots_x_vec,
            &knots_y_vec,
            order,
            boundary_multiplicity,
        )
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;
        let dict = pyo3::types::PyDict::new_bound(py);
        dict.set_item("coefs", out.coefs.into_pyarray_bound(py))?;
        dict.set_item("splinefit", out.splinefit.into_pyarray_bound(py))?;
        dict.set_item(
            "knots_x_aug",
            ndarray::Array1::from(out.knots_x_aug).into_pyarray_bound(py),
        )?;
        dict.set_item(
            "knots_y_aug",
            ndarray::Array1::from(out.knots_y_aug).into_pyarray_bound(py),
        )?;
        Ok(dict)
    }

    /// extract_tfpeaks(spect, stimes, sfreqs, baseline=None, ...)
    /// → dict{peak_time, peak_freq, duration, bandwidth, height, volume,
    ///        segment_num, area, peakiness, bbox (n,4), labels (F,T) i64,
    ///        height_data?, boundaries?}
    ///
    /// The whole TF-peak extraction in one call: segment, downsample,
    /// watershed, merge, paint labels, resize, trim, regionprops, then the
    /// `filterStatsTable` cuts. This is the same fused path MATLAB reaches
    /// through `dynamo_extract_tfpeaks`, so Python and the MATLAB rust
    /// backend run identical code rather than Python re-assembling the
    /// stages itself.
    ///
    /// `trim_shift_val` defaults to NaN, meaning "per-segment min(spect)".
    /// `height_data` / `boundaries` are ragged and cost an allocation per
    /// peak, so they are only built when asked for.
    #[pyfunction]
    #[pyo3(signature = (spect, stimes, sfreqs, baseline=None, seg_time=30.0,
                        downsample_f=2, downsample_t=2, merge_thresh=11.0,
                        max_merges=f64::INFINITY, trim_vol_thresh=0.8,
                        trim_shift_val=f64::NAN, dur_min=0.5, dur_max=5.0,
                        bw_min=2.0, bw_max=15.0, freq_min=0.0,
                        freq_max=f64::INFINITY, ht_db_min=f64::NEG_INFINITY,
                        expand_labels_distance=0, with_height_data=false,
                        with_boundaries=false))]
    #[allow(clippy::too_many_arguments)]
    fn extract_tfpeaks<'py>(
        py: Python<'py>,
        spect: PyReadonlyArray2<'py, f64>,
        stimes: PyReadonlyArray1<'py, f64>,
        sfreqs: PyReadonlyArray1<'py, f64>,
        baseline: Option<PyReadonlyArray1<'py, f64>>,
        seg_time: f64,
        downsample_f: usize,
        downsample_t: usize,
        merge_thresh: f64,
        max_merges: f64,
        trim_vol_thresh: f64,
        trim_shift_val: f64,
        dur_min: f64,
        dur_max: f64,
        bw_min: f64,
        bw_max: f64,
        freq_min: f64,
        freq_max: f64,
        ht_db_min: f64,
        expand_labels_distance: u32,
        with_height_data: bool,
        with_boundaries: bool,
    ) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let params = super::extract_pipeline::ExtractParams {
            seg_time,
            downsample_f,
            downsample_t,
            merge_thresh,
            max_merges,
            trim_vol_thresh,
            trim_shift_val,
            dur_min,
            dur_max,
            bw_min,
            bw_max,
            freq_min,
            freq_max,
            ht_db_min,
            expand_labels_distance,
        };
        let baseline_view = baseline.as_ref().map(|values| values.as_array());
        let (peaks, labels) = super::extract_pipeline::extract_tfpeaks(
            spect.as_array(),
            stimes.as_array(),
            sfreqs.as_array(),
            baseline_view,
            &params,
            None,
            None,
        )
        .map_err(pyo3::exceptions::PyValueError::new_err)?;

        let n_peaks = peaks.len();
        let dict = pyo3::types::PyDict::new_bound(py);
        set_vec_item(py, &dict, "peak_time", peaks.peak_time)?;
        set_vec_item(py, &dict, "peak_freq", peaks.peak_freq)?;
        set_vec_item(py, &dict, "duration", peaks.duration)?;
        set_vec_item(py, &dict, "bandwidth", peaks.bandwidth)?;
        set_vec_item(py, &dict, "height", peaks.height)?;
        set_vec_item(py, &dict, "volume", peaks.volume)?;
        set_vec_item(py, &dict, "segment_num", peaks.segment_num)?;
        set_vec_item(py, &dict, "area", peaks.area)?;
        set_vec_item(py, &dict, "peakiness", peaks.peakiness)?;

        let bbox = ndarray::Array2::from_shape_vec((n_peaks, 4), peaks.bbox)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(
                format!("bbox shape: {}", e)))?;
        dict.set_item("bbox", bbox.into_pyarray_bound(py))?;
        dict.set_item("labels", labels.into_pyarray_bound(py))?;

        if with_height_data {
            let items: Vec<Bound<'py, numpy::PyArray1<f64>>> = peaks
                .height_data
                .into_iter()
                .map(|pixel_values| ndarray::Array1::from(pixel_values).into_pyarray_bound(py))
                .collect();
            dict.set_item("height_data", items)?;
        }
        if with_boundaries {
            let items: Vec<Bound<'py, numpy::PyArray2<f64>>> = peaks
                .boundaries_xy
                .into_iter()
                .map(|xy_flat| {
                    let n_points = xy_flat.len() / 2;
                    ndarray::Array2::from_shape_vec((n_points, 2), xy_flat)
                        .unwrap_or_else(|_| ndarray::Array2::zeros((0, 2)))
                        .into_pyarray_bound(py)
                })
                .collect();
            dict.set_item("boundaries", items)?;
        }
        Ok(dict)
    }

    // ───────────────────────────────── group-comparison statistics
    //
    // Thin wrappers over `crate::stats`, so pydynamo runs the same code
    // the desktop app and CLI do. Shapes are flat `(n_bins * n_trials)`
    // row-major with `n_bins` given separately. The Python layer owns
    // reshaping, which keeps 1-D and 2-D (SOPH) callers on one binding.

    /// fdr_grid(g1, g2, n_bins, q=0.05, method="dep", paired=False)
    ///
    /// Returns `(p_values, p_adj, sigbins, crit_p, m)`. Port of
    /// `FDR_1D.m` / `FDR_2D.m`. Missing data is dropped, never imputed,
    /// and untestable bins return NaN and stay out of the FDR family.
    #[pyfunction]
    #[pyo3(signature = (g1, g2, n_bins, q=0.05, method="dep", paired=false))]
    fn fdr_grid<'py>(
        py: Python<'py>,
        g1: PyReadonlyArray1<'py, f64>,
        g2: PyReadonlyArray1<'py, f64>,
        n_bins: usize,
        q: f64,
        method: &str,
        paired: bool,
    ) -> PyResult<(
        Bound<'py, numpy::PyArray1<f64>>,
        Bound<'py, numpy::PyArray1<f64>>,
        Bound<'py, numpy::PyArray1<bool>>,
        f64,
        usize,
    )> {
        use crate::stats::{fdr, fdr_grid as fg};
        let fdr_method = fdr::Method::parse(method).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "unknown FDR method '{}' (want dep/dependent or pdep/independent)",
                method
            ))
        })?;
        // The kernel implements only the nonparametric tests.
        let nonparam = true;
        let test = fg::Test::from_matlab_flags(nonparam, paired)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        let result = fg::fdr_grid(
            g1.as_slice()?,
            g2.as_slice()?,
            n_bins,
            q,
            fdr_method,
            test,
        )
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok((
            result.p_values.into_pyarray_bound(py),
            result.p_adj.into_pyarray_bound(py),
            ndarray::Array1::from(result.sigbins).into_pyarray_bound(py),
            result.crit_p,
            result.m,
        ))
    }

    /// fdr_adjust(pvals, q=0.05, method="dep") -> (p_adj, crit_p, m)
    ///
    /// The bare Benjamini–Hochberg / Benjamini–Yekutieli step, for
    /// callers that already have p-values.
    #[pyfunction]
    #[pyo3(signature = (pvals, q=0.05, method="dep"))]
    fn fdr_adjust<'py>(
        py: Python<'py>,
        pvals: PyReadonlyArray1<'py, f64>,
        q: f64,
        method: &str,
    ) -> PyResult<(Bound<'py, numpy::PyArray1<f64>>, f64, usize)> {
        use crate::stats::fdr;
        let fdr_method = fdr::Method::parse(method).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!("unknown FDR method '{}'", method))
        })?;
        let result = fdr::adjust(pvals.as_slice()?, q, fdr_method);
        Ok((result.p_adj.into_pyarray_bound(py), result.crit_p, result.m))
    }

    /// gpermtest(g1, g2, n_bins, alpha=0.05, iterations=10000, seed=0)
    ///
    /// Returns `(sigbins, acceptance_bounds, true_stat, n_excluded,
    /// warning)`. Port of `gpermtest.m` / `gpermtest2.m`: family-wise
    /// control by the max statistic. `warning` is None or a string the
    /// caller should surface.
    #[pyfunction]
    #[pyo3(signature = (g1, g2, n_bins, alpha=0.05, iterations=10000, seed=0))]
    fn gpermtest<'py>(
        py: Python<'py>,
        g1: PyReadonlyArray1<'py, f64>,
        g2: PyReadonlyArray1<'py, f64>,
        n_bins: usize,
        alpha: f64,
        iterations: usize,
        seed: u64,
    ) -> PyResult<(
        Bound<'py, numpy::PyArray1<bool>>,
        Bound<'py, numpy::PyArray1<f64>>,
        Bound<'py, numpy::PyArray1<f64>>,
        usize,
        Option<String>,
    )> {
        let result = crate::stats::permtest::gpermtest(
            g1.as_slice()?,
            g2.as_slice()?,
            n_bins,
            alpha,
            iterations,
            seed,
        )
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok((
            ndarray::Array1::from(result.sigbins).into_pyarray_bound(py),
            result.acceptance_bounds.into_pyarray_bound(py),
            result.true_stat.into_pyarray_bound(py),
            result.n_excluded,
            result.warning,
        ))
    }


    /// One axis of [`fit_paramfits`], as a dict.
    #[allow(clippy::too_many_arguments)]
    fn paramfit_axis_dict<'py>(
        py: Python<'py>,
        fit: Result<&super::fits::ParamFitResult, String>,
        axis: super::fits_table::FitAxis,
        freq_bins: &[f64],
        phase_fit: Option<&super::fits::ParamFitResult>,
        phase_bins: Option<&[f64]>,
        sopower_hist: Option<&ndarray::Array2<f64>>,
        peaks: Option<&super::mode_peaks::PeakProps<'_>>,
        assign: super::paramfit::opts::PeakAssign,
    ) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        use super::fits_table::{
            compute_mode_peaks, effective_peak_assign, paramfit_rows, row_values, table_columns,
            FitAxis,
        };
        let dict = pyo3::types::PyDict::new_bound(py);
        let columns = table_columns(axis);
        dict.set_item("columns", columns.clone())?;
        let effective = effective_peak_assign(axis, assign);
        dict.set_item("peak_assign", effective.to_string())?;
        let fit = match fit {
            Ok(fit) => fit,
            Err(reason) => {
                // The fit declined (degenerate input) or was not run: no
                // modes, and why.
                dict.set_item("skipped", reason)?;
                dict.set_item("table", ndarray::Array2::<f64>::zeros((0, columns.len())).into_pyarray_bound(py))?;
                return Ok(dict);
            }
        };
        dict.set_item("skipped", py.None())?;
        dict.set_item("params", fit.params.clone().into_pyarray_bound(py))?;
        dict.set_item("background", ndarray::Array1::from(fit.background.to_vec()).into_pyarray_bound(py))?;
        dict.set_item("unit_row", fit.unit_row)?;
        dict.set_item("model", fit.model.clone().into_pyarray_bound(py))?;
        let gof = pyo3::types::PyDict::new_bound(py);
        gof.set_item("sse", fit.gof.sse)?;
        gof.set_item("rsquare", fit.gof.rsquare)?;
        gof.set_item("adjrsquare", fit.gof.adjrsquare)?;
        gof.set_item("rmse", fit.gof.rmse)?;
        gof.set_item("dfe", fit.gof.dfe)?;
        gof.set_item("dfm", fit.gof.dfm)?;
        dict.set_item("gof", gof)?;

        let is_phase = axis == FitAxis::Phase;
        let mode_peaks = peaks.map(|p| compute_mode_peaks(fit, is_phase, p, assign));
        let rows = paramfit_rows(
            fit, axis, freq_bins, phase_fit, phase_bins, sopower_hist, mode_peaks.as_deref(),
        );
        let mut table = ndarray::Array2::<f64>::zeros((rows.len(), columns.len()));
        for (r, row) in rows.iter().enumerate() {
            for (c, value) in row_values(row, axis).into_iter().enumerate() {
                table[[r, c]] = value;
            }
        }
        dict.set_item("table", table.into_pyarray_bound(py))?;
        Ok(dict)
    }

    /// One float column of the `peaks` dict.
    fn peak_column(peaks: &Bound<'_, pyo3::types::PyDict>, key: &str) -> PyResult<Vec<f64>> {
        let item = peaks
            .get_item(key)?
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(format!("peaks is missing '{key}'")))?;
        let array: PyReadonlyArray1<'_, f64> = item.extract()?;
        Ok(array.as_array().iter().copied().collect())
    }

    /// fit_paramfits(sopower_hist, sophase_hist, sopower_bins, sophase_bins,
    ///               freq_bins, opts_json=None, peaks=None,
    ///               baseline=None, baseline_freqs=None)
    /// → dict{"power": axis, "phase": axis}
    ///
    /// Both parametric-basis fits of a recording's SOPHs and their paramfit
    /// tables, exactly as the DesktopApp and CLI compute them. The
    /// histograms are `(n_so, n_freq)`. `opts_json` is the `paramfit`
    /// section of a settings file (`{"power": {...}, "phase": {...},
    /// "peak_assign": ...}`); omitted keys take their defaults.
    ///
    /// `peaks`, when given, fills the `Pk*` columns: a dict of 1-D arrays
    /// over the TF-peaks, keys `freq, duration, bandwidth, height, volume,
    /// area, peakiness, sopower, sophase` (floats) and `in_hist` (bools,
    /// the peaks the SOPHs counted). `baseline` / `baseline_freqs` (the
    /// extraction baseline, linear power) give `PkPower`; without them it
    /// is NaN.
    ///
    /// Each axis dict: `columns` (names), `table` (n_modes x columns),
    /// `peak_assign` (rule as used on that axis), `skipped` (None, or why
    /// the fit declined), and for a fit: `params` (n_modes x 6),
    /// `background` (3,), `unit_row`, `model` (n_so x n_freq), `gof`.
    #[pyfunction]
    #[pyo3(signature = (sopower_hist, sophase_hist, sopower_bins, sophase_bins, freq_bins,
                        opts_json=None, peaks=None, baseline=None, baseline_freqs=None))]
    #[allow(clippy::too_many_arguments)]
    fn fit_paramfits<'py>(
        py: Python<'py>,
        sopower_hist: PyReadonlyArray2<'py, f64>,
        sophase_hist: PyReadonlyArray2<'py, f64>,
        sopower_bins: PyReadonlyArray1<'py, f64>,
        sophase_bins: PyReadonlyArray1<'py, f64>,
        freq_bins: PyReadonlyArray1<'py, f64>,
        opts_json: Option<&str>,
        peaks: Option<Bound<'py, pyo3::types::PyDict>>,
        baseline: Option<PyReadonlyArray1<'py, f64>>,
        baseline_freqs: Option<PyReadonlyArray1<'py, f64>>,
    ) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        use super::fits_table::FitAxis;
        let opts: super::paramfit::opts::ParamFitOpts = match opts_json {
            Some(text) => serde_json::from_str(text)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("opts_json: {e}")))?,
            None => super::paramfit::opts::ParamFitOpts::default(),
        };
        let sopower_bins: Vec<f64> = sopower_bins.as_array().iter().copied().collect();
        let sophase_bins: Vec<f64> = sophase_bins.as_array().iter().copied().collect();
        let freq_bins: Vec<f64> = freq_bins.as_array().iter().copied().collect();
        let sopower_hist = sopower_hist.as_array().to_owned();
        let sophase_hist = sophase_hist.as_array().to_owned();

        let power_fit = super::fits::fit_paramfit_power(
            sopower_hist.view(), &sopower_bins, &freq_bins, &opts.power,
        );
        let phase_fit = super::fits::fit_paramfit_phase(
            sophase_hist.view(), &sophase_bins, &freq_bins, &opts.phase,
        );

        // Peak properties for the Pk* columns, owned here and borrowed below.
        let mut peak_columns: Option<(Vec<Vec<f64>>, Vec<bool>, Vec<f64>)> = None;
        if let Some(p) = &peaks {
            let names = [
                "freq", "duration", "bandwidth", "height", "volume", "area", "peakiness",
                "sopower", "sophase",
            ];
            let mut columns = Vec::with_capacity(names.len());
            for name in names {
                columns.push(peak_column(p, name)?);
            }
            let in_hist_item = p
                .get_item("in_hist")?
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("peaks is missing 'in_hist'"))?;
            let in_hist_array: PyReadonlyArray1<'_, bool> = in_hist_item.extract()?;
            let in_hist: Vec<bool> = in_hist_array.as_array().iter().copied().collect();
            let power = match (&baseline, &baseline_freqs) {
                (Some(b), Some(bf)) => {
                    let b: Vec<f64> = b.as_array().iter().copied().collect();
                    let bf: Vec<f64> = bf.as_array().iter().copied().collect();
                    super::mode_peaks::peak_power_db(&columns[0], &columns[3], &b, &bf)
                }
                _ => vec![f64::NAN; columns[0].len()],
            };
            peak_columns = Some((columns, in_hist, power));
        }
        let peak_props = peak_columns.as_ref().map(|(c, in_hist, power)| super::mode_peaks::PeakProps {
            freq: &c[0],
            duration: &c[1],
            bandwidth: &c[2],
            height: &c[3],
            volume: &c[4],
            area: &c[5],
            peakiness: &c[6],
            sopower: &c[7],
            sophase: &c[8],
            power,
            in_hist,
        });

        let out = pyo3::types::PyDict::new_bound(py);
        let power_dict = paramfit_axis_dict(
            py, power_fit.as_ref().map_err(|e| e.to_string()), FitAxis::Power, &freq_bins,
            phase_fit.as_ref().ok(), Some(&sophase_bins), None,
            peak_props.as_ref(), opts.peak_assign,
        )?;
        let phase_dict = paramfit_axis_dict(
            py, phase_fit.as_ref().map_err(|e| e.to_string()), FitAxis::Phase, &freq_bins,
            None, Some(&sopower_bins), Some(&sopower_hist),
            peak_props.as_ref(), opts.peak_assign,
        )?;
        out.set_item("power", power_dict)?;
        out.set_item("phase", phase_dict)?;
        Ok(out)
    }

    /// rebuild_phase_model(modes, background, phase_bins, freq_bins)
    /// → (n_freq, n_phase) array
    ///
    /// Rebuild a phase fit's model surface from its paramfit CSV values
    /// (`modes` n x 6 with the recorded Density in column 0), matching the
    /// fit's own model. See `fits::rebuild_phase_model`.
    #[pyfunction]
    fn rebuild_phase_model<'py>(
        py: Python<'py>,
        modes: PyReadonlyArray2<'py, f64>,
        background: PyReadonlyArray1<'py, f64>,
        phase_bins: PyReadonlyArray1<'py, f64>,
        freq_bins: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let modes = modes.as_array();
        if modes.ncols() != 6 {
            return Err(pyo3::exceptions::PyValueError::new_err("modes must be n x 6"));
        }
        let mut rows: Vec<[f64; 6]> = Vec::with_capacity(modes.nrows());
        for r in 0..modes.nrows() {
            let mut row = [0.0; 6];
            for c in 0..6 {
                row[c] = modes[[r, c]];
            }
            rows.push(row);
        }
        let phase_bins: Vec<f64> = phase_bins.as_array().iter().copied().collect();
        let freq_bins: Vec<f64> = freq_bins.as_array().iter().copied().collect();
        let surface = super::fits::rebuild_phase_model(
            &rows, bg3("background", background)?, &phase_bins, &freq_bins,
        );
        Ok(surface.into_pyarray_bound(py))
    }


    /// Convert a run error for Python: invalid inputs raise
    /// ValueError, anything else RuntimeError.
    fn run_error(e: super::run::DynamoError) -> PyErr {
        match e {
            super::run::DynamoError::Invalid(_) => {
                pyo3::exceptions::PyValueError::new_err(e.to_string())
            }
            _ => pyo3::exceptions::PyRuntimeError::new_err(e.to_string()),
        }
    }

    /// The `settings` section of a settings file, parsed with defaults for
    /// every key it omits (as the DYNAM-O app parses it).
    fn parse_settings(settings_json: Option<&str>) -> PyResult<super::run::DynamoSettings> {
        match settings_json {
            Some(text) => serde_json::from_str(text).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("settings_json: {e}"))
            }),
            None => Ok(super::run::DynamoSettings::default()),
        }
    }

    /// default_settings() → str
    ///
    /// The default `settings` section of a DYNAM-O settings file, as JSON.
    #[pyfunction]
    fn default_settings() -> PyResult<String> {
        serde_json::to_string(&super::run::DynamoSettings::default())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// migrate_settings(settings_json, from_version) → (settings_json, log)
    ///
    /// Bring the `settings` section of a settings file written under an
    /// older semantics version up to this kernel's
    /// (`SEMANTICS_VERSION`), with one log line per step. The same
    /// migrations `dynamo-cli migrate-settings` applies.
    #[pyfunction]
    fn migrate_settings(settings_json: &str, from_version: u32) -> PyResult<(String, Vec<String>)> {
        let mut value: serde_json::Value = serde_json::from_str(settings_json)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("settings_json: {e}")))?;
        let log = super::run::settings_schema::migrate(&mut value, from_version);
        let text = serde_json::to_string(&value)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok((text, log))
    }

    /// load_recording(edf_path, channel, staging_path=None, references=None,
    ///                time_col=1, stage_col=2, header_lines=0,
    ///                delimiter=",", epoch_dur=30.0, stage_labels=None)
    /// → dict{data, fs, stage_times, stage_vals}
    ///
    /// One channel of an EDF plus its staging, exactly as the DYNAM-O app
    /// loads a recording: channel expressions and references
    /// (`mean(C3, C4)`, `C3-A2`, `REF = ...`), staging clock times aligned
    /// to the EDF recording start, and an all-N2 hypnogram when there is no
    /// staging file. `stage_labels` is seven alias lists in the order
    /// `[artifact, wake, REM, N1, N2, N3, unknown]` (the settings file's
    /// `stage_labels`); empty uses the reader's built-in table.
    #[pyfunction]
    #[pyo3(signature = (edf_path, channel, staging_path=None, references=None, time_col=1,
                        stage_col=2, header_lines=0, delimiter=",", epoch_dur=30.0,
                        stage_labels=None))]
    #[allow(clippy::too_many_arguments)]
    fn load_recording<'py>(
        py: Python<'py>,
        edf_path: &str,
        channel: &str,
        staging_path: Option<&str>,
        references: Option<Vec<String>>,
        time_col: usize,
        stage_col: usize,
        header_lines: usize,
        delimiter: &str,
        epoch_dur: f64,
        stage_labels: Option<Vec<Vec<String>>>,
    ) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let references = references.unwrap_or_default();
        let stage_labels = stage_labels.unwrap_or_default();
        let staging = staging_path.map(std::path::Path::new);
        let (data, fs, stage_times, stage_vals) = super::run::load::load_edf_bundle(
            std::path::Path::new(edf_path), staging, channel, &references, time_col, stage_col,
            header_lines, delimiter, epoch_dur, &stage_labels,
        )
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.0))?;
        let dict = pyo3::types::PyDict::new_bound(py);
        dict.set_item("data", ndarray::Array1::from(data).into_pyarray_bound(py))?;
        dict.set_item("fs", fs)?;
        dict.set_item("stage_times", ndarray::Array1::from(stage_times).into_pyarray_bound(py))?;
        dict.set_item("stage_vals", ndarray::Array1::from(stage_vals).into_pyarray_bound(py))?;
        Ok(dict)
    }

    /// A 1-D float array from a Vec.
    fn arr1<'py>(py: Python<'py>, values: &[f64]) -> Bound<'py, numpy::PyArray1<f64>> {
        ndarray::Array1::from(values.to_vec()).into_pyarray_bound(py)
    }

    /// One axis's splinefit as a dict, or None.
    fn splinefit_dict<'py>(
        py: Python<'py>,
        fit: Option<&super::fits::SplineFitResult>,
    ) -> PyResult<PyObject> {
        let Some(fit) = fit else {
            return Ok(py.None());
        };
        let dict = pyo3::types::PyDict::new_bound(py);
        dict.set_item("coefs", fit.coefs.clone().into_pyarray_bound(py))?;
        dict.set_item("splinefit", fit.splinefit.clone().into_pyarray_bound(py))?;
        dict.set_item("knots_x_aug", arr1(py, &fit.knots_x_aug))?;
        dict.set_item("knots_y_aug", arr1(py, &fit.knots_y_aug))?;
        dict.set_item("fit_so_bins", arr1(py, &fit.fit_so_bins))?;
        dict.set_item("fit_freq_bins", arr1(py, &fit.fit_freq_bins))?;
        Ok(dict.into_any().unbind())
    }

    /// run_dynamo(data, fs, stage_times, stage_vals, settings_json=None,
    ///            time_range=None, keep_spectrogram=False, progress=None)
    /// → dict
    ///
    /// The whole DYNAM-O run of one recording, the code the DYNAM-O app
    /// and CLI run (`run::run_dynamo`), so the numbers are the same.
    /// `settings_json` is the `settings` section of a settings file
    /// (omitted keys take their defaults). `progress`, when given, is
    /// called as `progress(stage, fraction)`.
    ///
    /// Returns: `fs` (the rate the run used, after any resampling),
    /// `t_time_range`, `artifacts`, `stage_times`, `stage_vals`, `peaks`
    /// (dict of per-peak columns: PeakTime, PeakFrequency, Duration,
    /// Bandwidth, Height, Volume, SegmentNum, Area, Peakiness, bbox_tl_s,
    /// bbox_tl_Hz, PeakStage, SOpower, SOphase), `peak_in_hist`,
    /// `baseline`, `baseline_freqs`, `so_power_norm`, `so_power_times`,
    /// `so_power_norm_native`, `so_power_native_t_start`,
    /// `so_phase_unwrapped`, `so_phase_times`, `sopower_hist` and
    /// `sophase_hist` (n_so x n_freq), `freq_bins`, `sopower_bins`,
    /// `sophase_bins`, `summary_spectrogram` (+ `summary_sfreqs`,
    /// `summary_stimes`), `spectrogram` (+ `spect_sfreqs`,
    /// `spect_stimes`; empty unless `keep_spectrogram`), `paramfit`
    /// (`{"power": axis, "phase": axis}` as [`fit_paramfits`] returns),
    /// `splinefit` (`{"power": dict|None, "phase": dict|None}`),
    /// `semantics` (the computation-semantics version).
    #[pyfunction]
    #[pyo3(signature = (data, fs, stage_times, stage_vals, settings_json=None, time_range=None,
                        keep_spectrogram=false, progress=None))]
    #[allow(clippy::too_many_arguments)]
    fn run_dynamo<'py>(
        py: Python<'py>,
        data: PyReadonlyArray1<'py, f64>,
        fs: f64,
        stage_times: PyReadonlyArray1<'py, f64>,
        stage_vals: PyReadonlyArray1<'py, f64>,
        settings_json: Option<&str>,
        time_range: Option<(f64, f64)>,
        keep_spectrogram: bool,
        progress: Option<PyObject>,
    ) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        use super::fits_table::FitAxis;
        let settings = parse_settings(settings_json)?;
        let data: Vec<f64> = data.as_array().iter().copied().collect();
        let stage_times: Vec<f64> = stage_times.as_array().iter().copied().collect();
        let stage_vals: Vec<f64> = stage_vals.as_array().iter().copied().collect();

        let out = if keep_spectrogram {
            super::run::run_dynamo_keeping_spectrogram(
                &data, fs, &stage_times, &stage_vals, time_range, &settings,
            )
        } else {
            // The progress callback runs on the run's own thread; it takes
            // the GIL to call back into Python and ignores any error there.
            let report = |stage: &str, fraction: f64| {
                if let Some(callback) = &progress {
                    Python::with_gil(|py| {
                        let _ = callback.call1(py, (stage, fraction));
                    });
                }
            };
            let progress_fn: Option<&super::run::ProgressFn<'_>> = match progress {
                Some(_) => Some(&report),
                None => None,
            };
            super::run::run_dynamo_with_progress(
                &data, fs, &stage_times, &stage_vals, time_range, &settings, progress_fn, None,
                None,
            )
        }
        .map_err(run_error)?;

        let dict = pyo3::types::PyDict::new_bound(py);
        dict.set_item("fs", out.fs)?;
        dict.set_item("t_time_range", arr1(py, &out.t_time_range))?;
        dict.set_item(
            "artifacts",
            ndarray::Array1::from(out.artifacts.clone()).into_pyarray_bound(py),
        )?;
        dict.set_item("stage_times", arr1(py, &out.stage_times))?;
        dict.set_item("stage_vals", arr1(py, &out.stage_vals))?;

        // Per-peak columns, named as the stats CSV names them.
        let peaks = pyo3::types::PyDict::new_bound(py);
        let stats = &out.stats;
        let n_peaks = stats.len();
        let mut bbox_tl_s = Vec::with_capacity(n_peaks);
        let mut bbox_tl_hz = Vec::with_capacity(n_peaks);
        for i in 0..n_peaks {
            bbox_tl_s.push(stats.bbox[4 * i]);
            bbox_tl_hz.push(stats.bbox[4 * i + 1]);
        }
        let mut peak_stage = Vec::with_capacity(n_peaks);
        for &stage in out.peak_stage.iter() {
            peak_stage.push(stage as f64);
        }
        peaks.set_item("PeakTime", arr1(py, &stats.peak_time))?;
        peaks.set_item("PeakFrequency", arr1(py, &stats.peak_freq))?;
        peaks.set_item("Duration", arr1(py, &stats.duration))?;
        peaks.set_item("Bandwidth", arr1(py, &stats.bandwidth))?;
        peaks.set_item("Height", arr1(py, &stats.height))?;
        peaks.set_item("Volume", arr1(py, &stats.volume))?;
        peaks.set_item("SegmentNum", arr1(py, &stats.segment_num))?;
        peaks.set_item("Area", arr1(py, &stats.area))?;
        peaks.set_item("Peakiness", arr1(py, &stats.peakiness))?;
        peaks.set_item("bbox_tl_s", arr1(py, &bbox_tl_s))?;
        peaks.set_item("bbox_tl_Hz", arr1(py, &bbox_tl_hz))?;
        peaks.set_item("PeakStage", arr1(py, &peak_stage))?;
        peaks.set_item("SOpower", arr1(py, &out.peak_sopower))?;
        peaks.set_item("SOphase", arr1(py, &out.peak_sophase))?;
        dict.set_item("peaks", peaks)?;
        dict.set_item(
            "peak_in_hist",
            ndarray::Array1::from(out.peak_in_hist.clone()).into_pyarray_bound(py),
        )?;

        dict.set_item("baseline", out.baseline.clone().into_pyarray_bound(py))?;
        dict.set_item("baseline_freqs", arr1(py, &out.baseline_freqs))?;
        dict.set_item("so_power_norm", arr1(py, &out.so_power_norm))?;
        dict.set_item("so_power_times", arr1(py, &out.so_power_times))?;
        dict.set_item("so_power_norm_native", arr1(py, &out.so_power_norm_native))?;
        dict.set_item("so_power_native_t_start", out.so_power_native_t_start)?;
        dict.set_item("so_phase_unwrapped", arr1(py, &out.so_phase_unwrapped))?;
        dict.set_item("so_phase_times", arr1(py, &out.so_phase_times))?;
        dict.set_item("sopower_hist", out.sopower_hist.clone().into_pyarray_bound(py))?;
        dict.set_item("sophase_hist", out.sophase_hist.clone().into_pyarray_bound(py))?;
        dict.set_item("freq_bins", arr1(py, &out.hist_freq_centers))?;
        dict.set_item("sopower_bins", arr1(py, &out.sopower_bin_centers))?;
        dict.set_item("sophase_bins", arr1(py, &out.sophase_bin_centers))?;
        dict.set_item("summary_spectrogram", out.summary_spectrogram.clone().into_pyarray_bound(py))?;
        dict.set_item("summary_sfreqs", arr1(py, &out.summary_sfreqs))?;
        dict.set_item("summary_stimes", arr1(py, &out.summary_stimes))?;
        dict.set_item("spectrogram", out.spectrogram.clone().into_pyarray_bound(py))?;
        dict.set_item("spect_sfreqs", arr1(py, &out.spect_sfreqs))?;
        dict.set_item("spect_stimes", arr1(py, &out.spect_stimes))?;

        // Paramfit tables, with the Pk* columns over the peaks the SOPHs
        // counted, as the app's CSV writer computes them.
        let peak_power = super::mode_peaks::peak_power_db(
            &stats.peak_freq,
            &stats.height,
            out.baseline.as_slice().unwrap_or(&[]),
            &out.baseline_freqs,
        );
        let peak_props = super::mode_peaks::PeakProps {
            freq: &stats.peak_freq,
            duration: &stats.duration,
            bandwidth: &stats.bandwidth,
            height: &stats.height,
            volume: &stats.volume,
            area: &stats.area,
            peakiness: &stats.peakiness,
            sopower: &out.peak_sopower,
            sophase: &out.peak_sophase,
            power: &peak_power,
            in_hist: &out.peak_in_hist,
        };
        let not_run = "not computed (paramfits are turned off)".to_string();
        let power_fit = match (&out.paramfit_power, &out.paramfit_power_skipped) {
            (Some(fit), _) => Ok(fit),
            (None, Some(reason)) => Err(reason.clone()),
            (None, None) => Err(not_run.clone()),
        };
        let phase_fit = match (&out.paramfit_phase, &out.paramfit_phase_skipped) {
            (Some(fit), _) => Ok(fit),
            (None, Some(reason)) => Err(reason.clone()),
            (None, None) => Err(not_run),
        };
        let paramfit = pyo3::types::PyDict::new_bound(py);
        paramfit.set_item(
            "power",
            paramfit_axis_dict(
                py, power_fit, FitAxis::Power, &out.hist_freq_centers,
                out.paramfit_phase.as_ref(), Some(&out.sophase_bin_centers), None,
                Some(&peak_props), settings.paramfit.peak_assign,
            )?,
        )?;
        paramfit.set_item(
            "phase",
            paramfit_axis_dict(
                py, phase_fit, FitAxis::Phase, &out.hist_freq_centers,
                None, Some(&out.sopower_bin_centers), Some(&out.sopower_hist),
                Some(&peak_props), settings.paramfit.peak_assign,
            )?,
        )?;
        dict.set_item("paramfit", paramfit)?;

        let splinefit = pyo3::types::PyDict::new_bound(py);
        splinefit.set_item("power", splinefit_dict(py, out.splinefit_power.as_ref())?)?;
        splinefit.set_item("phase", splinefit_dict(py, out.splinefit_phase.as_ref())?)?;
        dict.set_item("splinefit", splinefit)?;
        dict.set_item("semantics", super::run::settings_schema::SEMANTICS_VERSION)?;
        Ok(dict)
    }

    /// write_stats_csv(path, peaks, fs, settings_json=None, writer="pydynamo",
    ///                 writer_version="", subject_id="")
    ///
    /// Write the stats table of a run (the `peaks` dict `run_dynamo`
    /// returns, with the run's `fs`) with the kernel's writer, the one the
    /// DYNAM-O app uses: same columns, precision rules and preamble,
    /// including the `# semantics:` line.
    #[pyfunction]
    #[pyo3(signature = (path, peaks, fs, settings_json=None, writer="pydynamo",
                        writer_version="", subject_id=""))]
    #[allow(clippy::too_many_arguments)]
    fn write_stats_csv(
        path: &str,
        peaks: Bound<'_, pyo3::types::PyDict>,
        fs: f64,
        settings_json: Option<&str>,
        writer: &str,
        writer_version: &str,
        subject_id: &str,
    ) -> PyResult<()> {
        let settings = parse_settings(settings_json)?;
        let column = |name: &str| peak_column(&peaks, name);
        let peak_time = column("PeakTime")?;
        let duration = column("Duration")?;
        let bandwidth = column("Bandwidth")?;
        let bbox_tl_s = column("bbox_tl_s")?;
        let bbox_tl_hz = column("bbox_tl_Hz")?;
        let mut bbox = Vec::with_capacity(4 * peak_time.len());
        for i in 0..peak_time.len() {
            bbox.extend_from_slice(&[bbox_tl_s[i], bbox_tl_hz[i], duration[i], bandwidth[i]]);
        }
        let mut peak_stage = Vec::with_capacity(peak_time.len());
        for stage in column("PeakStage")? {
            peak_stage.push(stage as u8);
        }
        let table = super::extract_pipeline::SegmentPeaks {
            peak_time,
            peak_freq: column("PeakFrequency")?,
            duration,
            bandwidth,
            height: column("Height")?,
            volume: column("Volume")?,
            segment_num: column("SegmentNum")?,
            bbox,
            area: column("Area")?,
            peakiness: column("Peakiness")?,
            ..Default::default()
        };
        let o = &settings.output;
        let precision = super::pipeline::StatsCsvPrecision::resolve(
            o.csv_time_sigfigs,
            o.csv_freq_sigfigs,
            o.csv_phase_sigfigs,
            o.csv_height_sigfigs,
            o.csv_volume_sigfigs,
            o.csv_area_sigfigs,
            o.csv_peakiness_sigfigs,
            o.csv_sopower_sigfigs,
            fs,
        );
        let provenance = super::pipeline::StatsCsvProvenance {
            writer,
            writer_version,
            subject_id: Some(subject_id),
            semantics: Some(super::run::settings_schema::SEMANTICS_VERSION),
        };
        super::pipeline::write_stats_csv(
            &table,
            &column("SOpower")?,
            &column("SOphase")?,
            &peak_stage,
            precision,
            Some(&provenance),
            std::path::Path::new(path),
        )
        .map_err(|e| pyo3::exceptions::PyOSError::new_err(format!("write_stats_csv: {e}")))
    }

    #[pymodule]
    fn dynamo_rs(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(merge_segment, m)?)?;
        m.add_function(wrap_pyfunction!(merge_segment_with_borders, m)?)?;
        m.add_function(wrap_pyfunction!(trim_regions, m)?)?;
        m.add_function(wrap_pyfunction!(matlab_watershed, m)?)?;
        m.add_function(wrap_pyfunction!(matlab_paint_labels, m)?)?;
        m.add_function(wrap_pyfunction!(so_power_from_spectrogram, m)?)?;
        m.add_function(wrap_pyfunction!(so_phase_from_eeg, m)?)?;
        m.add_function(wrap_pyfunction!(build_baseline_exclude, m)?)?;
        m.add_function(wrap_pyfunction!(detect_artifacts, m)?)?;
        m.add_function(wrap_pyfunction!(compute_baseline, m)?)?;
        m.add_function(wrap_pyfunction!(subtract_baseline, m)?)?;
        m.add_function(wrap_pyfunction!(mask_spectrogram, m)?)?;
        m.add_function(wrap_pyfunction!(tfpeak_histogram, m)?)?;
        m.add_function(wrap_pyfunction!(hann_event_spectra, m)?)?;
        m.add_function(wrap_pyfunction!(refine_from_spectra, m)?)?;
        m.add_function(wrap_pyfunction!(sosfiltfilt, m)?)?;
        m.add_function(wrap_pyfunction!(hilbert, m)?)?;
        m.add_function(wrap_pyfunction!(unwrap, m)?)?;
        m.add_function(wrap_pyfunction!(movmean, m)?)?;
        m.add_function(wrap_pyfunction!(read_edf, m)?)?;
        m.add_function(wrap_pyfunction!(read_staging, m)?)?;
        m.add_function(wrap_pyfunction!(extract_tfpeaks, m)?)?;
        m.add_function(wrap_pyfunction!(fit_rotgauss, m)?)?;
        m.add_function(wrap_pyfunction!(fit_vmgauss, m)?)?;
        m.add_function(wrap_pyfunction!(fit_tensor_product_spline, m)?)?;
        m.add_function(wrap_pyfunction!(fit_paramfits, m)?)?;
        m.add_function(wrap_pyfunction!(rebuild_phase_model, m)?)?;
        m.add_function(wrap_pyfunction!(default_settings, m)?)?;
        m.add_function(wrap_pyfunction!(migrate_settings, m)?)?;
        m.add_function(wrap_pyfunction!(load_recording, m)?)?;
        m.add_function(wrap_pyfunction!(run_dynamo, m)?)?;
        m.add_function(wrap_pyfunction!(write_stats_csv, m)?)?;
        m.add("SEMANTICS_VERSION", super::run::settings_schema::SEMANTICS_VERSION)?;
        m.add_function(wrap_pyfunction!(build_info_py, m)?)?;
        m.add_function(wrap_pyfunction!(fdr_grid, m)?)?;
        m.add_function(wrap_pyfunction!(fdr_adjust, m)?)?;
        m.add_function(wrap_pyfunction!(gpermtest, m)?)?;
        m.add("__version__", env!("CARGO_PKG_VERSION"))?;
        Ok(())
    }

    /// Build identity of the compiled kernel, for provenance stamping
    /// (`kernel_version`, DesktopApp OUTPUT_FORMAT.md §8.1).
    ///
    /// Returns `{"version": semver, "git_sha": sha12[.dirty]|unknown,
    /// "dirty": bool, "build_info": "<semver>+<sha>[.dirty]"}`. Unlike the
    /// dist metadata (`importlib.metadata.version`), this reflects the
    /// compiled extension module itself.
    #[pyfunction]
    #[pyo3(name = "build_info")]
    fn build_info_py(py: Python<'_>) -> PyResult<Bound<'_, pyo3::types::PyDict>> {
        let dict = pyo3::types::PyDict::new_bound(py);
        let sha = env!("DYNAMO_GIT_SHA");
        dict.set_item("version", env!("CARGO_PKG_VERSION"))?;
        dict.set_item("git_sha", sha)?;
        dict.set_item("dirty", sha.ends_with(".dirty"))?;
        dict.set_item("build_info", crate::build_info::VERSION)?;
        Ok(dict)
    }
}
