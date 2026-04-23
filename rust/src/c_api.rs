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
/// All array pointers point to row-major C layout: the spectrogram is
/// (F, T) stored as `spect[row * T + col]`. `stimes.len() == T`,
/// `sfreqs.len() == F`.
///
/// If `baseline_ptr` is null the spectrogram is used as-is. Otherwise the
/// spectrogram is divided row-wise by `baseline` (length F) before
/// segmentation.
///
/// The pipeline mirrors pydynamo's `extract_tfpeaks` end-to-end:
///   1. Split the full (F, T) spect into `seg_time`-second segments
///      (MATLAB `segmentData.m` semantics: floor + ceil).
///   2. For each segment in parallel: stride-downsample by
///      (`downsample_f`, `downsample_t`), watershed(-spect), merge,
///      expand_labels(distance=5), resize labels up, trim, regionprops.
///   3. Concatenate per-segment peaks.
///   4. Apply filterStatsTable: keep peaks where
///        dur_min < Duration < dur_max,
///        bw_min  < Bandwidth < bw_max,
///        freq_min < PeakFrequency < freq_max,
///        pow2db(Height) > ht_db_min.
///
/// Parameter defaults when 0.0 is passed:
///   * `seg_time` → 30.0
///   * `downsample_f` / `downsample_t` → 1
///   * `max_merges` → +∞ if 0
///   * `freq_max` → +∞ if 0
///   * `dur_max`, `bw_max`, `ht_db_min` → no-op if you want everything
///     through, pass +∞ / +∞ / -∞ respectively.
#[repr(C)]
pub struct ExtractTfpeaksIn {
    pub spect_ptr: *const f64,
    pub n_freqs: usize,
    pub n_times: usize,
    pub stimes_ptr: *const f64,
    pub sfreqs_ptr: *const f64,
    pub baseline_ptr: *const f64, // may be null
    pub seg_time: f64,
    pub downsample_f: u32,
    pub downsample_t: u32,
    pub merge_thresh: f64,
    pub max_merges: f64,
    pub trim_vol_thresh: f64,
    pub trim_shift_val: f64, // NaN → use min(spect) per segment
    pub dur_min: f64,
    pub dur_max: f64,
    pub bw_min: f64,
    pub bw_max: f64,
    pub freq_min: f64,
    pub freq_max: f64,
    pub ht_db_min: f64,
    /// Distance for expand_labels(). 0 = skip (MATLAB-native; labels
    /// keep 0 on watershed lines). 5 = pydynamo default (skimage-style
    /// fill so regions touch directly).
    pub expand_labels_distance: u32,
    /// Optional progress callback invoked once per completed segment.
    /// Signature: `fn(segments_done: u32, segments_total: u32)`. Pass
    /// `NULL` / `None` to skip. Called from rayon worker threads, but
    /// dynamo_extract_tfpeaks serializes calls with an internal mutex
    /// so the C callee may assume it is never invoked concurrently.
    pub progress_cb: Option<extern "C" fn(u32, u32)>,
}

/// Output descriptor for [`dynamo_extract_tfpeaks`].
///
/// All `*mut` pointers are callee-allocated; caller must free each with the
/// matching `dynamo_free_buffer_*` call. `bounding_box` is `n_peaks × 4`
/// row-major `[t_tl, f_tl, width_s, height_Hz]` per peak (pydynamo format).
///
/// `labels` is a row-major `(n_freqs, n_times)` i64 buffer of length
/// `n_label_elems = n_freqs * n_times`. Non-zero pixels carry a 1-based
/// peak index: a pixel with value `k` belongs to the k-th peak in the
/// returned arrays (so label `k` maps to row `k - 1` in `peak_time` etc).
/// Zero = background. Per-segment label images are stitched column-wise
/// with a running offset and then renumbered after the post-filter so
/// that surviving labels span `1..=n_peaks` densely.
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
    pub labels: *mut i64,       // (n_freqs, n_times) row-major; 1-based peak indices
    pub n_label_elems: usize,   // = n_freqs * n_times (0 if spect was empty)
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

        // Build ndarray views.
        let spect_view = match ArrayView2::from_shape((f, t), spect_slice) {
            Ok(v) => v,
            Err(_) => return ErrorCode::ShapeMismatch.code(),
        };
        let stimes_view = ArrayView1::from(stimes_slice);
        let sfreqs_view = ArrayView1::from(sfreqs_slice);
        let baseline_view = baseline_vec
            .as_ref()
            .map(|v| ArrayView1::from(v.as_slice()));

        // Normalize parameter defaults for values sentinel'd as 0 / inf.
        let seg_time = if input.seg_time > 0.0 { input.seg_time } else { 30.0 };
        let df = if input.downsample_f == 0 { 1 } else { input.downsample_f as usize };
        let dt_ds = if input.downsample_t == 0 { 1 } else { input.downsample_t as usize };
        let max_merges = if input.max_merges == 0.0 { f64::INFINITY } else { input.max_merges };
        let freq_max = if input.freq_max == 0.0 { f64::INFINITY } else { input.freq_max };

        let params = crate::extract_pipeline::ExtractParams {
            seg_time,
            downsample_f: df,
            downsample_t: dt_ds,
            merge_thresh: input.merge_thresh,
            max_merges,
            trim_vol_thresh: input.trim_vol_thresh,
            trim_shift_val: input.trim_shift_val,
            dur_min: input.dur_min,
            dur_max: input.dur_max,
            bw_min: input.bw_min,
            bw_max: input.bw_max,
            freq_min: input.freq_min,
            freq_max,
            ht_db_min: input.ht_db_min,
            expand_labels_distance: input.expand_labels_distance,
        };

        // Wrap the optional C callback in a Sync closure guarded by a
        // mutex. Rayon workers may call it concurrently; the mutex
        // ensures only one call reaches the C side at a time (MATLAB's
        // mexPrintf is not thread-safe).
        let progress_boxed: Option<Box<dyn Fn(u32, u32) + Sync + Send>> =
            input.progress_cb.map(|cb| {
                let m = std::sync::Mutex::new(());
                Box::new(move |d, t| {
                    let _g = m.lock().unwrap();
                    cb(d, t);
                }) as Box<dyn Fn(u32, u32) + Sync + Send>
            });
        let progress_ref: Option<&(dyn Fn(u32, u32) + Sync)> =
            progress_boxed.as_deref().map(|b| b as &(dyn Fn(u32, u32) + Sync));

        let (peaks, labels) = match crate::extract_pipeline::extract_tfpeaks(
            spect_view, stimes_view, sfreqs_view, baseline_view, &params, progress_ref,
        ) {
            Ok(p) => p,
            Err(_) => return ErrorCode::KernelError.code(),
        };

        // Fill out struct.
        let np = peaks.len();
        output.n_peaks = np;
        output.peak_time = leak_vec_f64(peaks.peak_time);
        output.peak_freq = leak_vec_f64(peaks.peak_freq);
        output.duration = leak_vec_f64(peaks.duration);
        output.bandwidth = leak_vec_f64(peaks.bandwidth);
        output.height = leak_vec_f64(peaks.height);
        output.volume = leak_vec_f64(peaks.volume);
        output.segment_num = leak_vec_f64(peaks.segment_num);
        output.bounding_box = leak_vec_f64(peaks.bbox);

        // Concatenated (F, T) row-major label image, 1-based peak indices.
        let labels_vec: Vec<i64> = labels.into_raw_vec();
        let n_label = labels_vec.len();
        output.labels = leak_vec_i64(labels_vec);
        output.n_label_elems = n_label;

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
// 3b. dynamo_mask_spectrogram
// -------------------------------------------------------------------------

/// Perimeter-aware mask of pass-2 spectrogram using pass-1 labels.
/// Thin FFI wrapper over [`crate::mask::mask_spectrogram`].
///
/// All 2-D arrays are row-major: `spect_2s[f * n_times_2 + t]`, etc.
/// `labels_1s` is (n_freqs, n_times_1) int64; `spect_2s` and `out_masked`
/// are both (n_freqs, n_times_2) float64.
///
/// Caller allocates `out_masked` of size `n_freqs * n_times_2`.
///
/// Returns 0 on success, negative on error.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn dynamo_mask_spectrogram(
    spect_2s: *const f64,
    stimes_2s: *const f64,
    labels_1s: *const i64,
    stimes_1s: *const f64,
    n_freqs: usize,
    n_times_2: usize,
    n_times_1: usize,
    out_masked: *mut f64,
) -> c_int {
    let result = catch_unwind(AssertUnwindSafe(|| {
        let ss = match ptr_as_slice::<f64>(spect_2s, n_freqs * n_times_2) {
            Some(s) => s, None => return ErrorCode::NullPointer.code(),
        };
        let st2 = match ptr_as_slice::<f64>(stimes_2s, n_times_2) {
            Some(s) => s, None => return ErrorCode::NullPointer.code(),
        };
        let ll = match ptr_as_slice::<i64>(labels_1s, n_freqs * n_times_1) {
            Some(s) => s, None => return ErrorCode::NullPointer.code(),
        };
        let st1 = match ptr_as_slice::<f64>(stimes_1s, n_times_1) {
            Some(s) => s, None => return ErrorCode::NullPointer.code(),
        };
        let out = match NonNull::new(out_masked) {
            Some(p) => std::slice::from_raw_parts_mut(p.as_ptr(), n_freqs * n_times_2),
            None => return ErrorCode::NullPointer.code(),
        };
        let spect_view = match ArrayView2::from_shape((n_freqs, n_times_2), ss) {
            Ok(v) => v, Err(_) => return ErrorCode::ShapeMismatch.code(),
        };
        let lbl_view = match ArrayView2::from_shape((n_freqs, n_times_1), ll) {
            Ok(v) => v, Err(_) => return ErrorCode::ShapeMismatch.code(),
        };
        let st2_view = ArrayView1::from(st2);
        let st1_view = ArrayView1::from(st1);

        let masked = match crate::mask::mask_spectrogram(
            spect_view, st2_view, lbl_view, st1_view,
        ) {
            Ok(m) => m,
            Err(_) => return ErrorCode::KernelError.code(),
        };
        // Copy into caller buffer (row-major).
        for (i, v) in masked.iter().enumerate() {
            out[i] = *v;
        }
        ErrorCode::Ok.code()
    }));
    match result {
        Ok(code) => code,
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
//     seg_time: 30.0,
//     downsample_f: 2, downsample_t: 2,
//     merge_thresh: 8.0,
//     max_merges: f64::INFINITY,
//     trim_vol_thresh: 0.8,
//     trim_shift_val: f64::NAN,
//     dur_min: 0.5, dur_max: 5.0,
//     bw_min: 2.0, bw_max: 15.0,
//     freq_min: 0.0, freq_max: 40.0,
//     ht_db_min: 7.63,
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
            seg_time: 30.0,
            downsample_f: 1,
            downsample_t: 1,
            merge_thresh: 8.0,
            max_merges: f64::INFINITY,
            trim_vol_thresh: 0.8,
            trim_shift_val: f64::NAN,
            dur_min: 0.5,
            dur_max: 5.0,
            bw_min: 2.0,
            bw_max: 15.0,
            freq_min: 0.0,
            freq_max: 40.0,
            ht_db_min: 7.63,
            expand_labels_distance: 5,
        };
        let mut out = empty_extract_out();
        unsafe {
            let rc = dynamo_extract_tfpeaks(&in_, &mut out);
            assert_eq!(rc, ErrorCode::Ok.code());
            assert_eq!(out.n_peaks, 0);
        }
    }
}
