//! C ABI surface for MEX wrappers (Phase 5) and any non-Rust consumer.
//!
//! Three public functions expose the heavy kernels already present in this
//! crate:
//!   * [`dynamo_extract_tfpeaks`]  — watershed → merge → trim → peak props
//!   * [`dynamo_refine_peaks`]     — hann_event_spectra + refine_from_spectra
//!   * [`dynamo_tfpeak_histogram`] — port of `histogram::tfpeak_histogram`
//!
//! Plus three matching buffer-free helpers so callers can release the
//! callee-allocated output arrays:
//!   * [`dynamo_free_buffer_f64`]
//!   * [`dynamo_free_buffer_i64`]
//!   * [`dynamo_free_buffer_u8`]
//!
//! ## Memory model
//!
//! Every output pointer is allocated with `Box::leak(vec.into_boxed_slice())`.
//! The callee also writes the element count into the `*Out` struct alongside
//! the pointer so the caller can reconstruct the slice when freeing. The
//! caller then:
//!   1. Copies bytes into its own managed array (e.g. `mxArray`).
//!   2. Calls the matching `dynamo_free_buffer_*(ptr, len)`.
//!
//! ## Panics
//!
//! None. Every FFI entry point wraps its body in `std::panic::catch_unwind`
//! and converts panics to `ErrorCode::Panic`.
//!
//! ## Error codes
//!
//! See [`ErrorCode`]. Zero = success; negative integers are well-defined
//! errors. We never return positive codes.

use ndarray::{Array2, ArrayView1, ArrayView2};
use std::os::raw::c_int;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr::NonNull;
use std::slice;

/// Status/error codes for the C ABI. All non-zero values are negative.
#[repr(i32)]
#[derive(Clone, Copy, Debug)]
pub enum ErrorCode {
    Ok = 0,
    NullPointer = -1,
    ShapeMismatch = -2,
    KernelError = -3,
    Panic = -4,
    InvalidArgument = -5,
}

impl ErrorCode {
    #[inline]
    fn code(self) -> c_int {
        self as c_int
    }
}

// -------------------------------------------------------------------------
// extract_tfpeaks: in/out structs
// -------------------------------------------------------------------------

/// Input descriptor for [`dynamo_extract_tfpeaks`].
///
/// All array pointers point to column-major-equivalent data: the spectrogram
/// is (F, T) with F varying fastest in memory (row-major C layout matching
/// `ndarray::Array2::from_shape_vec((F, T), vec)` where vec was built
/// row-by-row). `stimes.len() == T`, `sfreqs.len() == F`.
///
/// If `baseline_ptr` is null the spectrogram is used as-is. Otherwise the
/// spectrogram is divided column-wise by `baseline` (length F).
#[repr(C)]
pub struct ExtractTfpeaksIn {
    pub spect_ptr: *const f64,
    pub n_freqs: usize,
    pub n_times: usize,
    pub stimes_ptr: *const f64,
    pub sfreqs_ptr: *const f64,
    pub baseline_ptr: *const f64, // may be null
    pub merge_thresh: f64,
    pub max_merges: f64,
    pub trim_vol_thresh: f64,
    pub trim_shift_val: f64,     // NaN → use min(spect)
    pub segment_num: f64,        // tagged onto every peak
}

/// Output descriptor for [`dynamo_extract_tfpeaks`].
///
/// All `*mut` pointers are callee-allocated; caller must free each with the
/// matching `dynamo_free_buffer_*` call. `bounding_box` is `n_peaks × 4`
/// row-major ([f_lo, f_hi, t_lo, t_hi] per peak). `labels` is `F × T`
/// row-major (same layout as `spect_ptr`).
#[repr(C)]
pub struct ExtractTfpeaksOut {
    pub n_peaks: usize,
    pub peak_time: *mut f64,
    pub peak_freq: *mut f64,
    pub duration: *mut f64,
    pub bandwidth: *mut f64,
    pub height: *mut f64,
    pub volume: *mut f64,
    pub segment_num: *mut f64,
    pub bounding_box: *mut f64, // n_peaks * 4
    pub labels: *mut i64,       // n_freqs * n_times
    pub n_label_elems: usize,
}

// -------------------------------------------------------------------------
// helpers
// -------------------------------------------------------------------------

/// Allocate a heap slice, leak it, return a raw pointer. The caller must
/// eventually hand the pointer back to `dynamo_free_buffer_*` with the
/// same length.
#[inline]
fn leak_vec_f64(v: Vec<f64>) -> *mut f64 {
    let boxed: Box<[f64]> = v.into_boxed_slice();
    Box::into_raw(boxed) as *mut f64
}

#[inline]
fn leak_vec_i64(v: Vec<i64>) -> *mut i64 {
    let boxed: Box<[i64]> = v.into_boxed_slice();
    Box::into_raw(boxed) as *mut i64
}

#[inline]
#[allow(dead_code)] // used by tests / future u8 outputs; kept alongside i64/f64 variants
fn leak_vec_u8(v: Vec<u8>) -> *mut u8 {
    let boxed: Box<[u8]> = v.into_boxed_slice();
    Box::into_raw(boxed) as *mut u8
}

/// SAFETY: `ptr` must have been produced by `leak_vec_f64` with this `len`,
/// and must not have been freed already.
unsafe fn drop_leaked_f64(ptr: *mut f64, len: usize) {
    if ptr.is_null() {
        return;
    }
    let _ = Box::from_raw(slice::from_raw_parts_mut(ptr, len));
}

unsafe fn drop_leaked_i64(ptr: *mut i64, len: usize) {
    if ptr.is_null() {
        return;
    }
    let _ = Box::from_raw(slice::from_raw_parts_mut(ptr, len));
}

unsafe fn drop_leaked_u8(ptr: *mut u8, len: usize) {
    if ptr.is_null() {
        return;
    }
    let _ = Box::from_raw(slice::from_raw_parts_mut(ptr, len));
}

/// Defensive pointer-to-slice helper. Returns `None` if `ptr` is null and
/// `len != 0`.
unsafe fn ptr_as_slice<'a, T>(ptr: *const T, len: usize) -> Option<&'a [T]> {
    if len == 0 {
        return Some(&[]);
    }
    NonNull::new(ptr as *mut T).map(|nn| slice::from_raw_parts(nn.as_ptr(), len))
}

// -------------------------------------------------------------------------
// 1. dynamo_extract_tfpeaks
// -------------------------------------------------------------------------

/// Run the full tf-peak extraction pipeline on a single segment spectrogram.
///
/// # Safety
/// `in_` must point to a valid `ExtractTfpeaksIn`. `out` must point to a
/// valid (uninitialized or zeroed) `ExtractTfpeaksOut`. Array pointers
/// inside `in_` must be non-null and live for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn dynamo_extract_tfpeaks(
    in_: *const ExtractTfpeaksIn,
    out: *mut ExtractTfpeaksOut,
) -> c_int {
    let result = catch_unwind(AssertUnwindSafe(|| {
        let input = match NonNull::new(in_ as *mut ExtractTfpeaksIn) {
            Some(p) => &*p.as_ptr(),
            None => return ErrorCode::NullPointer.code(),
        };
        let output = match NonNull::new(out) {
            Some(p) => &mut *p.as_ptr(),
            None => return ErrorCode::NullPointer.code(),
        };

        let f = input.n_freqs;
        let t = input.n_times;
        if f == 0 || t == 0 {
            *output = empty_extract_out();
            return ErrorCode::Ok.code();
        }
        let n = f * t;
        let spect_slice = match ptr_as_slice::<f64>(input.spect_ptr, n) {
            Some(s) => s,
            None => return ErrorCode::NullPointer.code(),
        };
        let stimes_slice = match ptr_as_slice::<f64>(input.stimes_ptr, t) {
            Some(s) => s,
            None => return ErrorCode::NullPointer.code(),
        };
        let sfreqs_slice = match ptr_as_slice::<f64>(input.sfreqs_ptr, f) {
            Some(s) => s,
            None => return ErrorCode::NullPointer.code(),
        };

        let baseline_vec: Option<Vec<f64>> = if input.baseline_ptr.is_null() {
            None
        } else {
            let b = slice::from_raw_parts(input.baseline_ptr, f);
            Some(b.to_vec())
        };

        // Build ndarray view (F, T) row-major.
        let spect_view = match ArrayView2::from_shape((f, t), spect_slice) {
            Ok(v) => v,
            Err(_) => return ErrorCode::ShapeMismatch.code(),
        };

        // Optionally divide by baseline (broadcast along columns).
        let spect_owned: Array2<f64> = if let Some(bl) = baseline_vec.as_ref() {
            let mut m = spect_view.to_owned();
            for row in 0..f {
                let denom = bl[row];
                if denom == 0.0 || !denom.is_finite() {
                    continue;
                }
                for col in 0..t {
                    m[[row, col]] /= denom;
                }
            }
            m
        } else {
            spect_view.to_owned()
        };

        // Step 1: watershed on (negated) spectrogram — same as MATLAB which
        // calls watershed() on -spect to get basins-at-peaks behaviour.
        let neg: Array2<f64> = spect_owned.mapv(|v| -v);
        let ws_u16 = crate::matlab_watershed::matlab_watershed_2d(neg.view());
        let ws_i64: Array2<i64> = ws_u16.mapv(|v| v as i64);

        // Step 2: merge (with borders) on positive spectrogram.
        let (interior_labels, with_borders) = match crate::merge::run_with_borders(
            ws_i64.view(),
            spect_owned.view(),
            input.merge_thresh,
            input.max_merges,
        ) {
            Ok(r) => r,
            Err(_) => return ErrorCode::KernelError.code(),
        };

        // Step 3: trim.
        let shift_val = if input.trim_shift_val.is_nan() {
            spect_owned
                .iter()
                .cloned()
                .fold(f64::INFINITY, f64::min)
        } else {
            input.trim_shift_val
        };
        let trimmed = match crate::trim::trim_all_regions(
            interior_labels.view(),
            spect_owned.view(),
            input.trim_vol_thresh,
            shift_val,
        ) {
            Ok(r) => r,
            Err(_) => return ErrorCode::KernelError.code(),
        };

        // Step 4: compute peak properties. Enumerate labels present in
        // `trimmed`, group pixels by label, compute centroid + bbox over
        // the trimmed interior; compute bbox using with_borders if present
        // (MATLAB uses interior+border for bbox), fall back to interior.
        let props = compute_peak_properties(
            trimmed.view(),
            with_borders.view(),
            spect_owned.view(),
            stimes_slice,
            sfreqs_slice,
            input.segment_num,
        );

        // Fill out struct.
        let np = props.n_peaks;
        output.n_peaks = np;
        output.peak_time = leak_vec_f64(props.peak_time);
        output.peak_freq = leak_vec_f64(props.peak_freq);
        output.duration = leak_vec_f64(props.duration);
        output.bandwidth = leak_vec_f64(props.bandwidth);
        output.height = leak_vec_f64(props.height);
        output.volume = leak_vec_f64(props.volume);
        output.segment_num = leak_vec_f64(props.segment_num);
        output.bounding_box = leak_vec_f64(props.bounding_box);

        // Labels: always the trimmed F×T image as i64.
        let labels_i64: Vec<i64> = trimmed.iter().map(|&v| v as i64).collect();
        output.n_label_elems = labels_i64.len();
        output.labels = leak_vec_i64(labels_i64);

        ErrorCode::Ok.code()
    }));

    match result {
        Ok(code) => code,
        Err(_) => ErrorCode::Panic.code(),
    }
}

fn empty_extract_out() -> ExtractTfpeaksOut {
    ExtractTfpeaksOut {
        n_peaks: 0,
        peak_time: std::ptr::null_mut(),
        peak_freq: std::ptr::null_mut(),
        duration: std::ptr::null_mut(),
        bandwidth: std::ptr::null_mut(),
        height: std::ptr::null_mut(),
        volume: std::ptr::null_mut(),
        segment_num: std::ptr::null_mut(),
        bounding_box: std::ptr::null_mut(),
        labels: std::ptr::null_mut(),
        n_label_elems: 0,
    }
}

struct PeakProps {
    n_peaks: usize,
    peak_time: Vec<f64>,
    peak_freq: Vec<f64>,
    duration: Vec<f64>,
    bandwidth: Vec<f64>,
    height: Vec<f64>,
    volume: Vec<f64>,
    segment_num: Vec<f64>,
    bounding_box: Vec<f64>,
}

/// For each label present in `trimmed_interior`, compute:
///   * peak_time, peak_freq: intensity-weighted centroid over interior.
///   * height: max spect value at interior pixels.
///   * volume: sum of spect values at interior pixels.
///   * bbox: [f_lo, f_hi, t_lo, t_hi] using interior+border pixels.
///   * duration = t_hi - t_lo, bandwidth = f_hi - f_lo.
///
/// MATLAB's extractTFPeaks uses `Ldata` (interior+border) for the bbox;
/// `with_borders` matches that. Height/volume/centroid use interior only
/// (same as MATLAB's peak-pixel set `rgn{ii}`-interior filter).
fn compute_peak_properties(
    trimmed_interior: ArrayView2<i32>,
    with_borders: ArrayView2<i32>,
    spect: ArrayView2<f64>,
    stimes: &[f64],
    sfreqs: &[f64],
    segment_num_val: f64,
) -> PeakProps {
    use std::collections::HashMap;

    let (f, t) = trimmed_interior.dim();
    debug_assert_eq!(with_borders.dim(), (f, t));
    debug_assert_eq!(spect.dim(), (f, t));
    debug_assert_eq!(sfreqs.len(), f);
    debug_assert_eq!(stimes.len(), t);

    struct Accum {
        sum_v: f64,
        sum_v_f: f64,
        sum_v_t: f64,
        max_v: f64,
        f_min: usize,
        f_max: usize,
        t_min: usize,
        t_max: usize,
    }
    let mut interior_map: HashMap<i32, Accum> = HashMap::new();

    // First pass: interior pixels — height, volume, centroid.
    for row in 0..f {
        for col in 0..t {
            let lbl = trimmed_interior[[row, col]];
            if lbl <= 0 {
                continue;
            }
            let v = spect[[row, col]];
            let sf = sfreqs[row];
            let st = stimes[col];
            let entry = interior_map.entry(lbl).or_insert(Accum {
                sum_v: 0.0,
                sum_v_f: 0.0,
                sum_v_t: 0.0,
                max_v: f64::NEG_INFINITY,
                f_min: row,
                f_max: row,
                t_min: col,
                t_max: col,
            });
            entry.sum_v += v;
            entry.sum_v_f += v * sf;
            entry.sum_v_t += v * st;
            if v > entry.max_v {
                entry.max_v = v;
            }
            if row < entry.f_min { entry.f_min = row; }
            if row > entry.f_max { entry.f_max = row; }
            if col < entry.t_min { entry.t_min = col; }
            if col > entry.t_max { entry.t_max = col; }
        }
    }

    // Second pass: expand bbox to include border-painted pixels.
    for row in 0..f {
        for col in 0..t {
            let lbl = with_borders[[row, col]];
            if lbl <= 0 {
                continue;
            }
            if let Some(entry) = interior_map.get_mut(&lbl) {
                if row < entry.f_min { entry.f_min = row; }
                if row > entry.f_max { entry.f_max = row; }
                if col < entry.t_min { entry.t_min = col; }
                if col > entry.t_max { entry.t_max = col; }
            }
        }
    }

    // Sort labels ascending for deterministic output order.
    let mut keys: Vec<i32> = interior_map.keys().copied().collect();
    keys.sort();

    let np = keys.len();
    let mut peak_time = Vec::with_capacity(np);
    let mut peak_freq = Vec::with_capacity(np);
    let mut duration = Vec::with_capacity(np);
    let mut bandwidth = Vec::with_capacity(np);
    let mut height = Vec::with_capacity(np);
    let mut volume = Vec::with_capacity(np);
    let mut segment_out = Vec::with_capacity(np);
    let mut bbox = Vec::with_capacity(np * 4);

    for k in &keys {
        let a = &interior_map[k];
        let cf = if a.sum_v > 0.0 { a.sum_v_f / a.sum_v } else { sfreqs[a.f_min] };
        let ct = if a.sum_v > 0.0 { a.sum_v_t / a.sum_v } else { stimes[a.t_min] };
        let f_lo = sfreqs[a.f_min];
        let f_hi = sfreqs[a.f_max];
        let t_lo = stimes[a.t_min];
        let t_hi = stimes[a.t_max];
        peak_time.push(ct);
        peak_freq.push(cf);
        duration.push(t_hi - t_lo);
        bandwidth.push(f_hi - f_lo);
        height.push(a.max_v);
        volume.push(a.sum_v);
        segment_out.push(segment_num_val);
        bbox.push(f_lo);
        bbox.push(f_hi);
        bbox.push(t_lo);
        bbox.push(t_hi);
    }

    PeakProps {
        n_peaks: np,
        peak_time,
        peak_freq,
        duration,
        bandwidth,
        height,
        volume,
        segment_num: segment_out,
        bounding_box: bbox,
    }
}

// -------------------------------------------------------------------------
// 2. dynamo_refine_peaks
// -------------------------------------------------------------------------

/// Run Hann-event spectra + spline refinement for a batch of TF peaks.
///
/// Inputs:
///   * `peak_time`, `peak_freq`, `bbox` (n × 4, [f_lo, f_hi, t_lo, t_hi]), `n`.
///   * `data`, `n_data`, `fs` — raw 1-D signal + sampling rate.
///   * `freq_lo`, `freq_hi`, `window_size` (s), `dsfreqs` (Hz).
///
/// Caller provides preallocated output buffers:
///   * `out_peak_freq[i]` — refined frequency (NaN if removed).
///   * `out_keep[i]` — 1 if peak passed edge filter, 0 otherwise.
///
/// # Safety
/// All pointers must be valid for at least `n` elements (`n_data` for `data`),
/// and output buffers must be writable for `n` elements.
#[no_mangle]
pub unsafe extern "C" fn dynamo_refine_peaks(
    peak_time: *const f64,
    peak_freq: *const f64,
    bbox: *const f64,
    n: usize,
    data: *const f64,
    n_data: usize,
    fs: f64,
    freq_lo: f64,
    freq_hi: f64,
    window_size: f64,
    dsfreqs: f64,
    out_peak_freq: *mut f64,
    out_keep: *mut u8,
) -> c_int {
    let result = catch_unwind(AssertUnwindSafe(|| {
        if n == 0 {
            return ErrorCode::Ok.code();
        }
        let pt = match ptr_as_slice::<f64>(peak_time, n) {
            Some(s) => s,
            None => return ErrorCode::NullPointer.code(),
        };
        let _pf = match ptr_as_slice::<f64>(peak_freq, n) {
            Some(s) => s,
            None => return ErrorCode::NullPointer.code(),
        };
        let bb = match ptr_as_slice::<f64>(bbox, n * 4) {
            Some(s) => s,
            None => return ErrorCode::NullPointer.code(),
        };
        let dat = match ptr_as_slice::<f64>(data, n_data) {
            Some(s) => s,
            None => return ErrorCode::NullPointer.code(),
        };
        let out_freq = match NonNull::new(out_peak_freq) {
            Some(p) => slice::from_raw_parts_mut(p.as_ptr(), n),
            None => return ErrorCode::NullPointer.code(),
        };
        let out_k = match NonNull::new(out_keep) {
            Some(p) => slice::from_raw_parts_mut(p.as_ptr(), n),
            None => return ErrorCode::NullPointer.code(),
        };

        // Unpack bbox (n, 4) row-major → bbox_lo, bbox_hi (freq band per peak).
        let mut bbox_lo = Vec::with_capacity(n);
        let mut bbox_hi = Vec::with_capacity(n);
        for i in 0..n {
            bbox_lo.push(bb[i * 4]);     // f_lo
            bbox_hi.push(bb[i * 4 + 1]); // f_hi
        }

        let event_times: Vec<f64> = pt.to_vec();
        let data_vec: Vec<f64> = dat.to_vec();
        // t0 = 0.0 — caller is expected to have pre-aligned event_times to data.
        let (spect, sfreqs) = crate::refine::hann_event_spectra(
            &data_vec,
            fs,
            &event_times,
            0.0,
            (freq_lo, freq_hi),
            window_size,
            dsfreqs,
            crate::refine::DetrendOpt::Constant,
        );
        let refined = crate::refine::refine_from_spectra(
            &spect,
            &sfreqs,
            &bbox_lo,
            &bbox_hi,
            1000,
            true,
        );
        for i in 0..n {
            let v = refined[i];
            out_freq[i] = v;
            out_k[i] = if v.is_finite() { 1 } else { 0 };
        }
        ErrorCode::Ok.code()
    }));
    match result {
        Ok(c) => c,
        Err(_) => ErrorCode::Panic.code(),
    }
}

// -------------------------------------------------------------------------
// 3. dynamo_tfpeak_histogram
// -------------------------------------------------------------------------

/// FFI mirror of `histogram::tfpeak_histogram`.
///
/// Inputs are raw pointers with explicit lengths. Edge arrays are `(2, nbins)`
/// row-major: `[0 * nbins + i]` is the low edge of bin i, `[1 * nbins + i]`
/// is the high edge.
///
/// Caller-provided output buffers (must all be non-null):
///   * `out_c_mat`          — `num_cbins * num_fbins` row-major.
///   * `out_time_in_bin`    — `num_cbins * 5` row-major (minutes).
///   * `out_prop_in_bin`    — `num_cbins * 5` row-major.
///   * `out_peak_at_freq`   — `num_fbins`.
///
/// # Safety
/// All pointers must be valid for the lengths derived from `n_times`,
/// `n_peaks`, `num_fbins`, `num_cbins`.
#[no_mangle]
pub unsafe extern "C" fn dynamo_tfpeak_histogram(
    c_metric: *const f64,
    c_stages: *const f64,
    c_dt: f64,
    c_valid: *const u8,
    c_valid_allstages: *const u8,
    n_times: usize,
    peak_freqs: *const f64,
    peak_c: *const f64,
    n_peaks: usize,
    freq_edges: *const f64, // 2 * num_fbins
    num_fbins: usize,
    c_edges: *const f64, // 2 * num_cbins
    num_cbins: usize,
    circular: u8,
    circular_lo: f64,
    circular_hi: f64,
    norm_dim: i32,
    compute_rate: u8,
    min_time_in_bin: f64,
    min_peak_at_freq: i32,
    out_c_mat: *mut f64,        // num_cbins * num_fbins
    out_time_in_bin: *mut f64,  // num_cbins * 5
    out_prop_in_bin: *mut f64,  // num_cbins * 5
    out_peak_at_freq: *mut f64, // num_fbins
) -> c_int {
    let result = catch_unwind(AssertUnwindSafe(|| {
        let cm = match ptr_as_slice::<f64>(c_metric, n_times) {
            Some(s) => s,
            None => return ErrorCode::NullPointer.code(),
        };
        let cs = match ptr_as_slice::<f64>(c_stages, n_times) {
            Some(s) => s,
            None => return ErrorCode::NullPointer.code(),
        };
        let cv_raw = match ptr_as_slice::<u8>(c_valid, n_times) {
            Some(s) => s,
            None => return ErrorCode::NullPointer.code(),
        };
        let cva_raw = match ptr_as_slice::<u8>(c_valid_allstages, n_times) {
            Some(s) => s,
            None => return ErrorCode::NullPointer.code(),
        };
        let pf = match ptr_as_slice::<f64>(peak_freqs, n_peaks) {
            Some(s) => s,
            None => return ErrorCode::NullPointer.code(),
        };
        let pc = match ptr_as_slice::<f64>(peak_c, n_peaks) {
            Some(s) => s,
            None => return ErrorCode::NullPointer.code(),
        };
        let fe = match ptr_as_slice::<f64>(freq_edges, 2 * num_fbins) {
            Some(s) => s,
            None => return ErrorCode::NullPointer.code(),
        };
        let ce = match ptr_as_slice::<f64>(c_edges, 2 * num_cbins) {
            Some(s) => s,
            None => return ErrorCode::NullPointer.code(),
        };

        let cv: Vec<bool> = cv_raw.iter().map(|&b| b != 0).collect();
        let cva: Vec<bool> = cva_raw.iter().map(|&b| b != 0).collect();

        let cm_arr = match ArrayView1::from_shape(n_times, cm) {
            Ok(v) => v,
            Err(_) => return ErrorCode::ShapeMismatch.code(),
        };
        let cs_arr = match ArrayView1::from_shape(n_times, cs) {
            Ok(v) => v,
            Err(_) => return ErrorCode::ShapeMismatch.code(),
        };
        let pf_arr = match ArrayView1::from_shape(n_peaks, pf) {
            Ok(v) => v,
            Err(_) => return ErrorCode::ShapeMismatch.code(),
        };
        let pc_arr = match ArrayView1::from_shape(n_peaks, pc) {
            Ok(v) => v,
            Err(_) => return ErrorCode::ShapeMismatch.code(),
        };
        let fe_arr = match ArrayView2::from_shape((2, num_fbins), fe) {
            Ok(v) => v,
            Err(_) => return ErrorCode::ShapeMismatch.code(),
        };
        let ce_arr = match ArrayView2::from_shape((2, num_cbins), ce) {
            Ok(v) => v,
            Err(_) => return ErrorCode::ShapeMismatch.code(),
        };

        let inp = crate::histogram::HistogramInputs {
            c_metric: cm_arr,
            c_stages: cs_arr,
            c_dt,
            c_valid: &cv,
            c_valid_allstages: &cva,
            peak_freqs: pf_arr,
            peak_c: pc_arr,
            freq_edges: fe_arr,
            c_edges: ce_arr,
            circular: circular != 0,
            circular_bounds: (circular_lo, circular_hi),
            norm_dim,
            compute_rate: compute_rate != 0,
            min_time_in_bin,
            min_peak_at_freq,
        };
        let out = match crate::histogram::tfpeak_histogram(&inp) {
            Ok(o) => o,
            Err(_) => return ErrorCode::KernelError.code(),
        };

        // Copy into caller-allocated buffers.
        let cmat_len = num_cbins * num_fbins;
        let cmat_ptr = match NonNull::new(out_c_mat) {
            Some(p) => slice::from_raw_parts_mut(p.as_ptr(), cmat_len),
            None => return ErrorCode::NullPointer.code(),
        };
        let tib_ptr = match NonNull::new(out_time_in_bin) {
            Some(p) => slice::from_raw_parts_mut(p.as_ptr(), num_cbins * 5),
            None => return ErrorCode::NullPointer.code(),
        };
        let pib_ptr = match NonNull::new(out_prop_in_bin) {
            Some(p) => slice::from_raw_parts_mut(p.as_ptr(), num_cbins * 5),
            None => return ErrorCode::NullPointer.code(),
        };
        let paf_ptr = match NonNull::new(out_peak_at_freq) {
            Some(p) => slice::from_raw_parts_mut(p.as_ptr(), num_fbins),
            None => return ErrorCode::NullPointer.code(),
        };

        for (i, &v) in out.c_mat.iter().enumerate() {
            cmat_ptr[i] = v;
        }
        for (i, &v) in out.time_in_bin.iter().enumerate() {
            tib_ptr[i] = v;
        }
        for (i, &v) in out.prop_in_bin.iter().enumerate() {
            pib_ptr[i] = v;
        }
        for (i, &v) in out.peak_at_freq.iter().enumerate() {
            paf_ptr[i] = v;
        }
        ErrorCode::Ok.code()
    }));
    match result {
        Ok(c) => c,
        Err(_) => ErrorCode::Panic.code(),
    }
}

// -------------------------------------------------------------------------
// 4. buffer-free helpers
// -------------------------------------------------------------------------

/// Free a buffer previously returned via one of the callee-allocated output
/// pointers of `ExtractTfpeaksOut`.
///
/// `len` MUST match the original allocation length (the matching `n_peaks`
/// or `n_label_elems` field for array-of-peaks or array-of-labels buffers).
///
/// # Safety
/// See [`drop_leaked_f64`].
#[no_mangle]
pub unsafe extern "C" fn dynamo_free_buffer_f64(ptr: *mut f64, len: usize) {
    drop_leaked_f64(ptr, len);
}

/// See [`dynamo_free_buffer_f64`].
#[no_mangle]
pub unsafe extern "C" fn dynamo_free_buffer_i64(ptr: *mut i64, len: usize) {
    drop_leaked_i64(ptr, len);
}

/// See [`dynamo_free_buffer_f64`].
#[no_mangle]
pub unsafe extern "C" fn dynamo_free_buffer_u8(ptr: *mut u8, len: usize) {
    drop_leaked_u8(ptr, len);
}

// -------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------
//
// Example FFI lifecycle (commented out as unit test because it runs the full
// pipeline — exercised in integration tests instead):
//
// ```ignore
// use dynamo_rs::c_api::*;
// let spect = vec![0.0_f64; 10 * 20]; // (F=10, T=20)
// let stimes: Vec<f64> = (0..20).map(|i| i as f64 * 0.1).collect();
// let sfreqs: Vec<f64> = (0..10).map(|i| i as f64).collect();
// let in_ = ExtractTfpeaksIn {
//     spect_ptr: spect.as_ptr(),
//     n_freqs: 10, n_times: 20,
//     stimes_ptr: stimes.as_ptr(),
//     sfreqs_ptr: sfreqs.as_ptr(),
//     baseline_ptr: std::ptr::null(),
//     merge_thresh: 8.0,
//     max_merges: f64::INFINITY,
//     trim_vol_thresh: 0.8,
//     trim_shift_val: f64::NAN,
//     segment_num: 1.0,
// };
// let mut out = ExtractTfpeaksOut {
//     n_peaks: 0,
//     peak_time: std::ptr::null_mut(),
//     peak_freq: std::ptr::null_mut(),
//     duration: std::ptr::null_mut(),
//     bandwidth: std::ptr::null_mut(),
//     height: std::ptr::null_mut(),
//     volume: std::ptr::null_mut(),
//     segment_num: std::ptr::null_mut(),
//     bounding_box: std::ptr::null_mut(),
//     labels: std::ptr::null_mut(),
//     n_label_elems: 0,
// };
// unsafe {
//     let rc = dynamo_extract_tfpeaks(&in_, &mut out);
//     assert_eq!(rc, 0);
//     // ... copy out.peak_time[0..out.n_peaks] etc into caller storage ...
//     dynamo_free_buffer_f64(out.peak_time,    out.n_peaks);
//     dynamo_free_buffer_f64(out.peak_freq,    out.n_peaks);
//     dynamo_free_buffer_f64(out.duration,     out.n_peaks);
//     dynamo_free_buffer_f64(out.bandwidth,    out.n_peaks);
//     dynamo_free_buffer_f64(out.height,       out.n_peaks);
//     dynamo_free_buffer_f64(out.volume,       out.n_peaks);
//     dynamo_free_buffer_f64(out.segment_num,  out.n_peaks);
//     dynamo_free_buffer_f64(out.bounding_box, out.n_peaks * 4);
//     dynamo_free_buffer_i64(out.labels,       out.n_label_elems);
// }
// ```

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn free_buffer_lifecycle_f64() {
        let v = vec![1.0_f64, 2.0, 3.0, 4.0];
        let ptr = leak_vec_f64(v);
        unsafe {
            // Read one value back.
            assert_eq!(*ptr.add(2), 3.0);
            dynamo_free_buffer_f64(ptr, 4);
            // Freeing null is a no-op.
            dynamo_free_buffer_f64(std::ptr::null_mut(), 0);
        }
    }

    #[test]
    fn free_buffer_lifecycle_i64() {
        let v: Vec<i64> = (0..8).collect();
        let ptr = leak_vec_i64(v);
        unsafe {
            dynamo_free_buffer_i64(ptr, 8);
            dynamo_free_buffer_i64(std::ptr::null_mut(), 0);
        }
    }

    #[test]
    fn free_buffer_lifecycle_u8() {
        let v: Vec<u8> = vec![1, 2, 3];
        let ptr = leak_vec_u8(v);
        unsafe {
            dynamo_free_buffer_u8(ptr, 3);
            dynamo_free_buffer_u8(std::ptr::null_mut(), 0);
        }
    }

    #[test]
    fn extract_tfpeaks_null_input_returns_error() {
        let mut out = empty_extract_out();
        unsafe {
            let rc = dynamo_extract_tfpeaks(std::ptr::null(), &mut out);
            assert_eq!(rc, ErrorCode::NullPointer.code());
        }
    }

    #[test]
    fn extract_tfpeaks_empty_spect_ok() {
        // F=0 or T=0 → returns Ok with zero peaks.
        let stimes: Vec<f64> = vec![];
        let sfreqs: Vec<f64> = vec![];
        let spect: Vec<f64> = vec![];
        let in_ = ExtractTfpeaksIn {
            spect_ptr: spect.as_ptr(),
            n_freqs: 0,
            n_times: 0,
            stimes_ptr: stimes.as_ptr(),
            sfreqs_ptr: sfreqs.as_ptr(),
            baseline_ptr: std::ptr::null(),
            merge_thresh: 8.0,
            max_merges: f64::INFINITY,
            trim_vol_thresh: 0.8,
            trim_shift_val: f64::NAN,
            segment_num: 1.0,
        };
        let mut out = empty_extract_out();
        unsafe {
            let rc = dynamo_extract_tfpeaks(&in_, &mut out);
            assert_eq!(rc, ErrorCode::Ok.code());
            assert_eq!(out.n_peaks, 0);
        }
    }
}
