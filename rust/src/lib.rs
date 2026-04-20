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
pub mod matlab_watershed;
pub mod merge;
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

    #[pymodule]
    fn dynamo_rs(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(merge_segment, m)?)?;
        m.add_function(wrap_pyfunction!(merge_segment_with_borders, m)?)?;
        m.add_function(wrap_pyfunction!(trim_regions, m)?)?;
        m.add_function(wrap_pyfunction!(matlab_watershed, m)?)?;
        m.add_function(wrap_pyfunction!(compute_baseline, m)?)?;
        m.add_function(wrap_pyfunction!(subtract_baseline, m)?)?;
        Ok(())
    }
}
