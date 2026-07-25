//! dynamo_rs — Rust kernel for the DYNAM-O merge-loop hotspot.
//!
//! Exposes `merge_segment(labels, data, merge_thresh, max_merges)` which
//! takes a watershed-labeled 2D image and the spectrogram values, builds
//! the region adjacency graph (region pixels + watershed-0-line borders +
//! neighbors via 2-pixel dilation), then runs the iterative max-weight
//! merge until no edge exceeds `merge_thresh`.
//!
//! Output: a (F, T) int32 label image where every surviving region's
//! pixels carry its label; watershed 0-line pixels stay 0.
//!
//! Merge rule (symmetric, matches current MATLAB `edgeWeightEqual.m`):
//!   A_ij_max = max(data at border intersection of i and j)
//!   w_ij = 2·A_ij_max − min_bnds_i − max_over(i_region ∪ i_border)
//!   w_ji = 2·A_ij_max − min_bnds_j − max_over(j_region ∪ j_border)
//!   weight = max(w_ij, w_ji)

pub mod adjacency;
pub mod artifacts;
pub mod baseline;
pub mod c_api;
pub mod extract_pipeline;
pub mod filter_cache;
pub mod filter_design;
pub mod histogram;
pub mod io;
pub mod mask;
pub mod matlab_watershed;
pub mod mts;
pub mod merge;
pub mod parallel;
pub mod peak_assign;
pub mod pipeline;
pub mod refine;
pub mod signal;
pub mod paramfit;
pub mod so_phase;
pub mod so_power;
pub mod spline_basis;
pub mod trim;

#[cfg(feature = "python")]
mod python {
    use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
    use pyo3::prelude::*;

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
        let lab = labels.as_array();
        let dat = data.as_array();
        if lab.shape() != dat.shape() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "labels and data shape mismatch",
            ));
        }
        let sv = match shift_val {
            Some(v) => v,
            None => {
                let slice = dat.as_slice().ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err("data must be C-contiguous")
                })?;
                slice.iter().cloned().fold(f64::INFINITY, f64::min)
            }
        };
        let out = super::trim::trim_all_regions(lab, dat, vol_thresh, sv)
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
    /// This replaces `skimage.segmentation.expand_labels(distance=5)` as
    /// the border-fill step so that pydynamo's pass-2 count matches
    /// MATLAB's ~0.8% peak-drift target (instead of skimage's ~+2%).
    /// Input must be 2D int64 C-contiguous; output is int64 same shape.
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
    /// Post-spectrogram SO-power pipeline — port of computeSOpower.m steps
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
        let (nf, nt) = spect_arr.dim();
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

        let nm = NormMethod::parse(norm_method).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "unrecognized norm_method {:?}",
                norm_method
            ))
        })?;

        let out = rs_fn(
            spect_slice, nf, nt,
            &stimes_slice, &sfreqs_slice,
            &eeg_times_slice, &isexcluded_slice,
            &stage_times_slice, &stage_vals_slice,
            time_range, outlier_threshold, &nm, retain_fs,
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
        let (nsec, ncols) = sos_arr.dim();
        if ncols != 6 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "sos must have 6 columns (scipy SOS format: [b0 b1 b2 a0 a1 a2]).",
            ));
        }
        let sos_slice: Vec<[f64; 6]> = (0..nsec)
            .map(|i| {
                [
                    sos_arr[[i, 0]], sos_arr[[i, 1]], sos_arr[[i, 2]],
                    sos_arr[[i, 3]], sos_arr[[i, 4]], sos_arr[[i, 5]],
                ]
            })
            .collect();

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
        let zm = match zscore_method.to_ascii_lowercase().as_str() {
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
            zscore_method: zm,
        };
        let vec = data.as_array().to_owned().into_raw_vec_and_offset().0;
        let out = rs_fn(&vec, fs, &opts);
        Ok(numpy::ndarray::Array1::from(out).into_pyarray_bound(py))
    }

    /// build_baseline_exclude(t_data, stage_times, stage_vals, baseline_stages,
    ///                        artifacts, user_exclude=None) -> (N,) bool
    ///
    /// OR together (explicit_exclude, stage_not_in_baseline_stages, artifacts).
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
        let t = t_data.as_array().to_owned().into_raw_vec_and_offset().0;
        let st = stage_times.as_array().to_owned().into_raw_vec_and_offset().0;
        let sv = stage_vals.as_array().to_owned().into_raw_vec_and_offset().0;
        let bs = baseline_stages.as_array().to_owned().into_raw_vec_and_offset().0;
        let art: Vec<bool> = artifacts.as_array().iter().copied().collect();
        let ue_vec: Option<Vec<bool>> = user_exclude
            .as_ref()
            .map(|ue| ue.as_array().iter().copied().collect());
        let out = super::baseline::build_baseline_exclude(
            &t, &st, &sv, &bs, &art, ue_vec.as_deref(),
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
        let v = c_valid.as_array().to_owned();
        let va = c_valid_allstages.as_array().to_owned();
        let v_vec: Vec<bool> = v.iter().copied().collect();
        let va_vec: Vec<bool> = va.iter().copied().collect();
        let inp = super::histogram::HistogramInputs {
            c_metric: c_metric.as_array(),
            c_stages: c_stages.as_array(),
            c_dt,
            c_valid: &v_vec,
            c_valid_allstages: &va_vec,
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
    /// Matches scipy.signal.sosfiltfilt with padtype='odd', default padlen.
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
        let sos_vec: Vec<[f64; 6]> = (0..sos_arr.nrows())
            .map(|r| [
                sos_arr[[r, 0]], sos_arr[[r, 1]], sos_arr[[r, 2]],
                sos_arr[[r, 3]], sos_arr[[r, 4]], sos_arr[[r, 5]],
            ])
            .collect();
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

    /// unwrap(p, discont=π) — matches numpy.unwrap.
    #[pyfunction]
    #[pyo3(signature = (p, discont=std::f64::consts::PI))]
    fn unwrap<'py>(
        py: Python<'py>,
        p: PyReadonlyArray1<'py, f64>,
        discont: f64,
    ) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
        let v: Vec<f64> = p.as_array().iter().copied().collect();
        let out = super::signal::unwrap(&v, discont);
        Ok(ndarray::Array1::from(out).into_pyarray_bound(py))
    }

    /// movmean(x, win) — MATLAB-style centered moving mean with partial edges.
    #[pyfunction]
    fn movmean<'py>(
        py: Python<'py>,
        x: PyReadonlyArray1<'py, f64>,
        win: usize,
    ) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
        let v: Vec<f64> = x.as_array().iter().copied().collect();
        let out = super::signal::movmean(&v, win);
        Ok(ndarray::Array1::from(out).into_pyarray_bound(py))
    }

    /// read_edf(path, channel=None) → dict with header, signals, data, fs, channel.
    ///
    /// If `channel` is None, returns all signals in a list. Otherwise returns
    /// (data, fs, label) for the single selected channel (with A–B rereference
    /// support). Port of `read_EDF_mex.c`.
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
        let hdr = PyDict::new_bound(py);
        hdr.set_item("edf_ver", &edf.header.edf_ver)?;
        hdr.set_item("patient_id", &edf.header.patient_id)?;
        hdr.set_item("local_rec_id", &edf.header.local_rec_id)?;
        hdr.set_item("recording_startdate", &edf.header.recording_startdate)?;
        hdr.set_item("recording_starttime", &edf.header.recording_starttime)?;
        hdr.set_item("num_header_bytes", edf.header.num_header_bytes)?;
        hdr.set_item("num_data_records", edf.header.num_data_records)?;
        hdr.set_item("data_record_duration", edf.header.data_record_duration)?;
        hdr.set_item("num_signals", edf.header.num_signals)?;
        out.set_item("header", hdr)?;

        // All signal labels (useful even when selecting one channel).
        let labels: Vec<&str> = edf.signals.iter().map(|s| s.signal_labels.as_str()).collect();
        out.set_item("labels", labels)?;
        let fs_all: Vec<f64> = edf.signals.iter().map(|s| s.sampling_frequency).collect();
        out.set_item("sampling_frequencies", fs_all)?;

        if let Some(ch) = channel {
            let (sh, values) = super::io::edf::select_channel(&edf, ch)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;
            let arr = ndarray::Array1::from(values);
            out.set_item("data", arr.into_pyarray_bound(py))?;
            out.set_item("fs", sh.sampling_frequency)?;
            out.set_item("label", sh.signal_labels.clone())?;
            let sinfo = PyDict::new_bound(py);
            sinfo.set_item("signal_labels", sh.signal_labels)?;
            sinfo.set_item("transducer_type", sh.transducer_type)?;
            sinfo.set_item("physical_dimension", sh.physical_dimension)?;
            sinfo.set_item("physical_min", sh.physical_min)?;
            sinfo.set_item("physical_max", sh.physical_max)?;
            sinfo.set_item("digital_min", sh.digital_min)?;
            sinfo.set_item("digital_max", sh.digital_max)?;
            sinfo.set_item("prefiltering", sh.prefiltering)?;
            sinfo.set_item("samples_in_record", sh.samples_in_record)?;
            sinfo.set_item("sampling_frequency", sh.sampling_frequency)?;
            out.set_item("signal_header", sinfo)?;
        } else {
            let lst = PyList::empty_bound(py);
            for (s, v) in edf.signals.iter().zip(edf.data.iter()) {
                let d = PyDict::new_bound(py);
                d.set_item("signal_labels", &s.signal_labels)?;
                d.set_item("transducer_type", &s.transducer_type)?;
                d.set_item("physical_dimension", &s.physical_dimension)?;
                d.set_item("physical_min", s.physical_min)?;
                d.set_item("physical_max", s.physical_max)?;
                d.set_item("digital_min", s.digital_min)?;
                d.set_item("digital_max", s.digital_max)?;
                d.set_item("prefiltering", &s.prefiltering)?;
                d.set_item("samples_in_record", s.samples_in_record)?;
                d.set_item("sampling_frequency", s.sampling_frequency)?;
                let arr = ndarray::Array1::from(v.clone());
                d.set_item("data", arr.into_pyarray_bound(py))?;
                lst.append(d)?;
            }
            out.set_item("signals", lst)?;
        }

        Ok(out)
    }

    /// read_staging(path, time_col=1, stage_col=2, header_lines=0,
    ///              delimiter=",", epoch_dur=30.0, start_time=None)
    /// → (times (N,), vals (N,)) numpy arrays.
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
        };
        let out = super::io::staging::read_staging(path, &opts)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;
        let t = ndarray::Array1::from(out.times);
        let v = ndarray::Array1::from(out.vals);
        Ok((t.into_pyarray_bound(py), v.into_pyarray_bound(py)))
    }

    /// Unpack a length-3 background-coefficient array into `[f64; 3]`.
    fn bg3(name: &str, arr: PyReadonlyArray1<'_, f64>) -> PyResult<[f64; 3]> {
        let v = arr.as_array();
        if v.len() != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "{} must have 3 elements, got {}",
                name,
                v.len()
            )));
        }
        Ok([v[0], v[1], v[2]])
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
    /// `[amp, fmean, fstd, pmean, pstd, theta]`. Note `fstd` is a standard
    /// deviation, not a variance. `max_iters=0` selects the scipy-equivalent
    /// default of `100 * n_params`.
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
        let xg: Vec<f64> = x_grid.as_array().iter().copied().collect();
        let yg: Vec<f64> = y_grid.as_array().iter().copied().collect();
        let out = super::paramfit::rot_gauss::fit_rotgauss(
            soph.as_array(),
            &xg,
            &yg,
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
    /// `[amp, fmean, fstd, phasepref, recikappa, theta]`, where `fstd` is a
    /// frequency standard deviation in Hz. `unit_row=True` replicates
    /// `normalized_vmGauss.m`'s per-frequency-row normalization (MATLAB
    /// passes `problem=true` for the phase fit).
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
        let xg: Vec<f64> = x_grid.as_array().iter().copied().collect();
        let yg: Vec<f64> = y_grid.as_array().iter().copied().collect();
        let out = super::paramfit::vm_gauss::fit_vmgauss(
            soph.as_array(),
            &xg,
            &yg,
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
    /// `spap2`. `soph` is `(n_x, n_y)` — the transpose of the canonical
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
        let xe: Vec<f64> = x_eval.as_array().iter().copied().collect();
        let ye: Vec<f64> = y_eval.as_array().iter().copied().collect();
        let kx: Vec<f64> = internal_knots_x.as_array().iter().copied().collect();
        let ky: Vec<f64> = internal_knots_y.as_array().iter().copied().collect();
        let out = super::spline_basis::fit_tensor_product_spline(
            soph.as_array(),
            &xe,
            &ye,
            &kx,
            &ky,
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
        let bl = baseline.as_ref().map(|b| b.as_array());
        let (peaks, labels) = super::extract_pipeline::extract_tfpeaks(
            spect.as_array(),
            stimes.as_array(),
            sfreqs.as_array(),
            bl,
            &params,
            None,
        )
        .map_err(pyo3::exceptions::PyValueError::new_err)?;

        let n = peaks.len();
        let dict = pyo3::types::PyDict::new_bound(py);
        macro_rules! put {
            ($name:expr, $v:expr) => {
                dict.set_item($name,
                    ndarray::Array1::from($v).into_pyarray_bound(py))?;
            };
        }
        put!("peak_time", peaks.peak_time);
        put!("peak_freq", peaks.peak_freq);
        put!("duration", peaks.duration);
        put!("bandwidth", peaks.bandwidth);
        put!("height", peaks.height);
        put!("volume", peaks.volume);
        put!("segment_num", peaks.segment_num);
        put!("area", peaks.area);
        put!("peakiness", peaks.peakiness);

        let bbox = ndarray::Array2::from_shape_vec((n, 4), peaks.bbox)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(
                format!("bbox shape: {}", e)))?;
        dict.set_item("bbox", bbox.into_pyarray_bound(py))?;
        dict.set_item("labels", labels.into_pyarray_bound(py))?;

        if with_height_data {
            let items: Vec<Bound<'py, numpy::PyArray1<f64>>> = peaks
                .height_data
                .into_iter()
                .map(|v| ndarray::Array1::from(v).into_pyarray_bound(py))
                .collect();
            dict.set_item("height_data", items)?;
        }
        if with_boundaries {
            let items: Vec<Bound<'py, numpy::PyArray2<f64>>> = peaks
                .boundaries_xy
                .into_iter()
                .map(|v| {
                    let m = v.len() / 2;
                    ndarray::Array2::from_shape_vec((m, 2), v)
                        .unwrap_or_else(|_| ndarray::Array2::zeros((0, 2)))
                        .into_pyarray_bound(py)
                })
                .collect();
            dict.set_item("boundaries", items)?;
        }
        Ok(dict)
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
        Ok(())
    }
}
