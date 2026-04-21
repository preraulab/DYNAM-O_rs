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
pub mod baseline;
pub mod c_api;
pub mod filter_cache;
pub mod histogram;
pub mod io;
pub mod mask;
pub mod matlab_watershed;
pub mod merge;
pub mod refine;
pub mod signal;
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

    #[pymodule]
    fn dynamo_rs(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(merge_segment, m)?)?;
        m.add_function(wrap_pyfunction!(merge_segment_with_borders, m)?)?;
        m.add_function(wrap_pyfunction!(trim_regions, m)?)?;
        m.add_function(wrap_pyfunction!(matlab_watershed, m)?)?;
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
        Ok(())
    }
}
