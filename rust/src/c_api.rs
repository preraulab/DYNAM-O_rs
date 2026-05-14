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

use ndarray::{ArrayView1, ArrayView2};
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
///
/// Variable-length per-peak data (`height_data`, `boundaries_xy`) is
/// flattened with a CSR-style offset array of length `n_peaks + 1`:
///   * `height_data[height_data_offsets[i] .. height_data_offsets[i+1]]`
///     are the spectrogram values inside peak i (one per pixel).
///   * `boundaries_xy[2*boundary_offsets[i] .. 2*boundary_offsets[i+1]]`
///     are interleaved (time, freq) pairs along peak i's perimeter.
/// Both offset arrays have length `n_peaks + 1` (always; even when n_peaks
/// is 0 the array is `[0]`).
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
    /// Time-frequency area per peak (sec*Hz). Length n_peaks.
    pub area: *mut f64,
    /// Peakiness = 10*log10(Area * Height / Volume) per peak, in dB. Length n_peaks.
    pub peakiness: *mut f64,
    /// Flattened per-peak pixel values; length `n_height_data_elems`.
    pub height_data: *mut f64,
    pub n_height_data_elems: usize,
    /// CSR offsets into `height_data`, length `n_peaks + 1`.
    pub height_data_offsets: *mut u64,
    /// Flattened per-peak boundary coords, interleaved (t, f). Length
    /// `2 * n_boundary_pixels`.
    pub boundaries_xy: *mut f64,
    pub n_boundary_pixels: usize,
    /// CSR offsets into `boundaries_xy` in PIXEL units (each pixel is two
    /// f64s). Length `n_peaks + 1`.
    pub boundary_offsets: *mut u64,
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
fn leak_vec_u64(v: Vec<u64>) -> *mut u64 {
    let boxed: Box<[u64]> = v.into_boxed_slice();
    Box::into_raw(boxed) as *mut u64
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

unsafe fn drop_leaked_u64(ptr: *mut u64, len: usize) {
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
        let labels_vec: Vec<i64> = labels.into_raw_vec_and_offset().0;
        let n_label = labels_vec.len();
        output.labels = leak_vec_i64(labels_vec);
        output.n_label_elems = n_label;

        // Area, Peakiness — scalar per peak.
        output.area = leak_vec_f64(peaks.area);
        output.peakiness = leak_vec_f64(peaks.peakiness);

        // HeightData: flatten Vec<Vec<f64>> into one buffer + CSR offsets.
        let mut hd_offsets: Vec<u64> = Vec::with_capacity(np + 1);
        hd_offsets.push(0);
        let mut hd_total: usize = 0;
        for v in &peaks.height_data {
            hd_total += v.len();
        }
        let mut hd_flat: Vec<f64> = Vec::with_capacity(hd_total);
        for v in peaks.height_data.into_iter() {
            hd_flat.extend_from_slice(&v);
            hd_offsets.push(hd_flat.len() as u64);
        }
        output.n_height_data_elems = hd_flat.len();
        output.height_data = leak_vec_f64(hd_flat);
        output.height_data_offsets = leak_vec_u64(hd_offsets);

        // Boundaries: each Vec<f64> already interleaves (t, f) pairs. Offset
        // unit is one *pair*, i.e., divide flat-len by 2.
        let mut b_offsets: Vec<u64> = Vec::with_capacity(np + 1);
        b_offsets.push(0);
        let mut b_total: usize = 0;
        for v in &peaks.boundaries_xy {
            b_total += v.len();
        }
        let mut b_flat: Vec<f64> = Vec::with_capacity(b_total);
        for v in peaks.boundaries_xy.into_iter() {
            b_flat.extend_from_slice(&v);
            b_offsets.push((b_flat.len() / 2) as u64);
        }
        output.n_boundary_pixels = b_flat.len() / 2;
        output.boundaries_xy = leak_vec_f64(b_flat);
        output.boundary_offsets = leak_vec_u64(b_offsets);

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
        area: std::ptr::null_mut(),
        peakiness: std::ptr::null_mut(),
        height_data: std::ptr::null_mut(),
        n_height_data_elems: 0,
        height_data_offsets: std::ptr::null_mut(),
        boundaries_xy: std::ptr::null_mut(),
        n_boundary_pixels: 0,
        boundary_offsets: std::ptr::null_mut(),
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
        let pf = match ptr_as_slice::<f64>(peak_freq, n) {
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
        // Edge-peak exclusion — match MATLAB refinePeakFrequency.m:141
        // behavior. MATLAB skips peaks whose center is within window_size/2
        // of data[0] or data[end] (a partial window would need zero-padding
        // and produce a distorted Hann-PSD → biased refined frequency).
        // Those peaks are kept at their original (pass-2 bbox-centroid)
        // PeakFrequency. We mirror that here rather than returning NaN or
        // the zero-padded Hann refine output.
        let data_t_end = if n_data > 0 { (n_data - 1) as f64 / fs } else { 0.0 };
        let edge_margin = window_size * 0.5;
        let t_lo = edge_margin;                 // t0 = 0.0 (caller pre-shifts)
        let t_hi = data_t_end - edge_margin;
        for i in 0..n {
            let is_edge = !(pt[i] > t_lo && pt[i] < t_hi);
            if is_edge {
                out_freq[i] = pf[i];            // keep original, unrefined
                out_k[i] = 1;                   // retained (not dropped)
            } else {
                let v = refined[i];
                out_freq[i] = v;
                out_k[i] = if v.is_finite() { 1 } else { 0 };
            }
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

        // MATLAB MEX allocates all 2-D outputs as COLUMN-major
        // (mxCreateDoubleMatrix). ndarray's Array2 default is row-major,
        // and `.iter()` walks row-major, so copy via explicit index
        // transpose. This bug caused striped histograms in TFPeakHistogram.m
        // (2026-04-24): row-major writes were being read column-major,
        // scrambling the (num_cbins, num_fbins) grid at period num_fbins.
        for s in 0..num_cbins {
            for f in 0..num_fbins {
                cmat_ptr[f * num_cbins + s] = out.c_mat[[s, f]];
            }
        }
        // time_in_bin, prop_in_bin: shape (num_cbins, 5) — same fix.
        for s in 0..num_cbins {
            for k in 0..5 {
                tib_ptr[k * num_cbins + s] = out.time_in_bin[[s, k]];
                pib_ptr[k * num_cbins + s] = out.prop_in_bin[[s, k]];
            }
        }
        // peak_at_freq: 1-D, no layout ambiguity.
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
// 5. dynamo_multitaper_spectrogram: in/out structs
// -------------------------------------------------------------------------

/// Input descriptor for [`dynamo_multitaper_spectrogram`].
///
/// `data_ptr` is a length-`n_data` f64 vector. `tapers_ptr` is the
/// pre-computed DPSS taper bank (caller supplies; we don't reimplement
/// `dpss` in Rust), row-major (n_tapers × winsize) so element (k, i) is
/// at `tapers_ptr[k * winsize + i]`. Caller MUST guarantee
/// `winsize == round(window_size_s * fs)` — the underlying compute
/// validates and returns ShapeMismatch otherwise.
///
/// `eigen_ptr` is required iff `weighting == 1` (Eigen). Pass null
/// (and `weighting = 0`) for unity weighting (the DYNAM-O default).
#[repr(C)]
pub struct MtsIn {
    pub data_ptr:        *const f64,
    pub n_data:          usize,
    pub fs:              f64,

    pub tapers_ptr:      *const f64,
    pub n_tapers:        usize,
    pub winsize:         usize,
    pub eigen_ptr:       *const f64, // null if unity
    pub eigen_len:       usize,      // 0 if unity

    pub freq_min:        f64,
    pub freq_max:        f64,
    pub window_size_s:   f64,
    pub window_step_s:   f64,
    pub nfft:            usize,
    pub detrend:         u32,        // 0=none, 1=linear, 2=constant
    pub weighting:       u32,        // 0=unity, 1=eigen
}

/// Output descriptor for [`dynamo_multitaper_spectrogram`].
///
/// `spect_ptr` points to a row-major (n_freqs_out × n_windows) f64 buffer.
/// All three pointers come from `Box::leak` and must be released with
/// `dynamo_free_buffer_f64(ptr, len)` where `len` is the matching
/// `n_freqs_out`, `n_windows`, or `n_freqs_out * n_windows` count.
#[repr(C)]
pub struct MtsOut {
    pub spect_ptr:    *mut f64,
    pub n_freqs_out:  usize,
    pub n_windows:    usize,
    pub stimes_ptr:   *mut f64,
    pub sfreqs_ptr:   *mut f64,
}

fn empty_mts_out() -> MtsOut {
    MtsOut {
        spect_ptr:   std::ptr::null_mut(),
        n_freqs_out: 0,
        n_windows:   0,
        stimes_ptr:  std::ptr::null_mut(),
        sfreqs_ptr:  std::ptr::null_mut(),
    }
}

/// Compute a multitaper spectrogram via the `multitaper_rs` crate.
///
/// Wraps `multitaper_rs::compute_spectrogram` with a C ABI matching the
/// shape of the other dynamo_rs ABI entries. f64 throughout — the MATLAB
/// Coder MEX uses f32, so callers swapping over should expect f32-roundoff
/// (~1e-7 relative) drift in output.
///
/// # Safety
/// `in_` must point to a valid `MtsIn`. `out` must point to a valid
/// (uninitialized or zeroed) `MtsOut`. Array pointers must be non-null
/// for the lengths declared (data_ptr, tapers_ptr, eigen_ptr if used).
#[no_mangle]
pub unsafe extern "C" fn dynamo_multitaper_spectrogram(
    in_: *const MtsIn,
    out: *mut MtsOut,
) -> c_int {
    let result = catch_unwind(AssertUnwindSafe(|| {
        let input = match NonNull::new(in_ as *mut MtsIn) {
            Some(p) => &*p.as_ptr(),
            None => return ErrorCode::NullPointer.code(),
        };
        let output = match NonNull::new(out) {
            Some(p) => &mut *p.as_ptr(),
            None => return ErrorCode::NullPointer.code(),
        };

        if input.n_data == 0 || input.n_tapers == 0 || input.winsize == 0 || input.nfft == 0 {
            *output = empty_mts_out();
            return ErrorCode::InvalidArgument.code();
        }

        let data_slice = match ptr_as_slice::<f64>(input.data_ptr, input.n_data) {
            Some(s) => s,
            None => return ErrorCode::NullPointer.code(),
        };
        let tapers_slice = match ptr_as_slice::<f64>(
            input.tapers_ptr,
            input.n_tapers * input.winsize,
        ) {
            Some(s) => s,
            None => return ErrorCode::NullPointer.code(),
        };

        let data_view = ArrayView1::from(data_slice);
        let tapers_view = match ArrayView2::from_shape((input.n_tapers, input.winsize), tapers_slice) {
            Ok(v) => v,
            Err(_) => return ErrorCode::ShapeMismatch.code(),
        };

        // Eigen weights are optional — only consumed when weighting=Eigen.
        let eigen_storage: Option<Vec<f64>> = if input.weighting == 1 {
            if input.eigen_ptr.is_null() || input.eigen_len != input.n_tapers {
                return ErrorCode::InvalidArgument.code();
            }
            Some(slice::from_raw_parts(input.eigen_ptr, input.eigen_len).to_vec())
        } else {
            None
        };
        let eigen_view = eigen_storage.as_ref().map(|v| ArrayView1::from(v.as_slice()));

        let detrend_mode = match input.detrend {
            0 => crate::mts::DetrendMode::Off,
            1 => crate::mts::DetrendMode::Linear,
            2 => crate::mts::DetrendMode::Constant,
            _ => return ErrorCode::InvalidArgument.code(),
        };
        let weighting = match input.weighting {
            0 => crate::mts::Weighting::Unity,
            1 => crate::mts::Weighting::Eigen,
            _ => return ErrorCode::InvalidArgument.code(),
        };

        let params = crate::mts::SpectrogramParams {
            fs: input.fs,
            frequency_range: (input.freq_min, input.freq_max),
            window_params: (input.window_size_s, input.window_step_s),
            nfft: input.nfft,
            detrend: detrend_mode,
            weighting,
        };

        let mts_out = match crate::mts::compute_spectrogram(data_view, tapers_view, eigen_view, &params) {
            Ok(o) => o,
            Err(_) => return ErrorCode::KernelError.code(),
        };

        let n_freqs_out = mts_out.sfreqs.len();
        let n_windows = mts_out.stimes.len();

        // mt_spectrogram comes back as Array2 with shape (n_freqs_out, n_windows).
        // Box::leak via a row-major Vec<f64> produced by .into_raw_vec_and_offset().
        // ndarray's into_raw_vec preserves layout; .as_standard_layout() ensures
        // the elements are contiguous row-major before we pull out the Vec.
        let spect_std = mts_out.mt_spectrogram.as_standard_layout().to_owned();
        let spect_vec: Vec<f64> = spect_std.into_raw_vec_and_offset().0;
        let stimes_vec: Vec<f64> = mts_out.stimes.to_vec();
        let sfreqs_vec: Vec<f64> = mts_out.sfreqs.to_vec();

        output.spect_ptr   = leak_vec_f64(spect_vec);
        output.n_freqs_out = n_freqs_out;
        output.n_windows   = n_windows;
        output.stimes_ptr  = leak_vec_f64(stimes_vec);
        output.sfreqs_ptr  = leak_vec_f64(sfreqs_vec);

        ErrorCode::Ok.code()
    }));
    result.unwrap_or(ErrorCode::Panic.code())
}

// -------------------------------------------------------------------------
// dynamo_dpss — DPSS (Slepian) taper generation
// -------------------------------------------------------------------------

/// Generate K DPSS tapers of length N with time-half-bandwidth NW.
///
/// Caller-allocated outputs:
///   `tapers_out`  — row-major (K, N) double, total K*N elements.
///                   Row k is the k-th Slepian sequence, unit L²-norm.
///   `ratios_out`  — length-K double, concentration ratios in (0, 1].
///
/// Matches MATLAB R2025a `dpss(N, NW, K)` and
/// `scipy.signal.windows.dpss(N, NW, K, return_ratios=True)` after the
/// scipy/MATLAB sign convention (even-index tapers have positive sum;
/// odd-index tapers have positive central derivative). Validated to
/// ≤1e-8 elementwise on (N=128, NW=2, K=3) and (N=1024, NW=4, K=7).
///
/// # Safety
/// `tapers_out` must point to at least `k * n` writable f64s.
/// `ratios_out` must point to at least `k` writable f64s.
#[no_mangle]
pub unsafe extern "C" fn dynamo_dpss(
    n: usize,
    nw: f64,
    k: usize,
    tapers_out: *mut f64,
    ratios_out: *mut f64,
) -> c_int {
    let result = catch_unwind(AssertUnwindSafe(|| {
        if tapers_out.is_null() || ratios_out.is_null() {
            return ErrorCode::NullPointer.code();
        }
        if n == 0 || k == 0 || k > n || !nw.is_finite() || nw <= 0.0 {
            return ErrorCode::InvalidArgument.code();
        }

        let (tapers, ratios) = match multitaper_rs::dpss(n, nw, k) {
            Ok(p) => p,
            Err(_) => return ErrorCode::KernelError.code(),
        };

        // Tapers come back as (K, N); ensure standard (row-major) layout
        // before copying.
        let tapers_std = tapers.as_standard_layout().to_owned();
        let tapers_slice = tapers_std.as_slice().unwrap_or(&[]);
        if tapers_slice.len() != k * n {
            return ErrorCode::ShapeMismatch.code();
        }
        let ratios_slice = match ratios.as_slice() {
            Some(s) => s,
            None => return ErrorCode::ShapeMismatch.code(),
        };
        if ratios_slice.len() != k {
            return ErrorCode::ShapeMismatch.code();
        }

        std::ptr::copy_nonoverlapping(tapers_slice.as_ptr(), tapers_out, k * n);
        std::ptr::copy_nonoverlapping(ratios_slice.as_ptr(), ratios_out, k);

        ErrorCode::Ok.code()
    }));
    result.unwrap_or(ErrorCode::Panic.code())
}

// -------------------------------------------------------------------------
// dynamo_detect_artifacts — band-detection iter-zscore artifact mask
// -------------------------------------------------------------------------

/// Plain-C mirror of `crate::artifacts::ArtifactOpts`. Mirrors the field
/// order of the Rust struct so `cbindgen` lays it out identically.
#[repr(C)]
pub struct ArtifactOptsFFI {
    pub hf_pass:          f64,
    pub hf_crit:          f64,
    pub bb_pass:          f64,
    pub bb_crit:          f64,
    pub hf_detrend:       u8,      // 0/1
    pub bb_detrend:       u8,      // 0/1
    pub smooth_duration:  f64,
    pub detrend_duration: f64,
    pub buffer_duration:  f64,
    pub zscore_method:    u32,     // 0=Robust, 1=Standard
}

impl ArtifactOptsFFI {
    fn to_rust(&self) -> Option<crate::artifacts::ArtifactOpts> {
        let zm = match self.zscore_method {
            0 => crate::artifacts::ZScoreMethod::Robust,
            1 => crate::artifacts::ZScoreMethod::Standard,
            _ => return None,
        };
        Some(crate::artifacts::ArtifactOpts {
            hf_pass: self.hf_pass,
            hf_crit: self.hf_crit,
            bb_pass: self.bb_pass,
            bb_crit: self.bb_crit,
            hf_detrend: self.hf_detrend != 0,
            bb_detrend: self.bb_detrend != 0,
            smooth_duration: self.smooth_duration,
            detrend_duration: self.detrend_duration,
            buffer_duration: self.buffer_duration,
            zscore_method: zm,
        })
    }
}

/// Band-detection artifact mask. **Does NOT include the slope-test
/// branch** — that needs the multitaper spectrogram and is left to the
/// caller (the MATLAB `detect_artifacts.m` ORs the slope mask in
/// separately). The Rust path here matches `detect_artifacts.m` with
/// `'slope_test', false` to within ≤30 samples on the synthetic-EEG
/// fixture battery; validated end-to-end in the upstream
/// `preraulab/artifact_detection` PR #1 and now MEX-bridged here.
///
/// Caller-allocated output `mask_out` is length `n`; 1 = artifact.
///
/// # Safety
/// `data` must point to `n` readable f64s; `opts` must point to a valid
/// `ArtifactOptsFFI`; `mask_out` must point to `n` writable u8s.
#[no_mangle]
pub unsafe extern "C" fn dynamo_detect_artifacts(
    data: *const f64,
    n: usize,
    fs: f64,
    opts: *const ArtifactOptsFFI,
    mask_out: *mut u8,
) -> c_int {
    let result = catch_unwind(AssertUnwindSafe(|| {
        if mask_out.is_null() || opts.is_null() {
            return ErrorCode::NullPointer.code();
        }
        if n == 0 || !fs.is_finite() || fs <= 0.0 {
            return ErrorCode::InvalidArgument.code();
        }
        let data_slice = match ptr_as_slice::<f64>(data, n) {
            Some(s) => s,
            None => return ErrorCode::NullPointer.code(),
        };
        let opts_ref = &*opts;
        let opts_rust = match opts_ref.to_rust() {
            Some(o) => o,
            None => return ErrorCode::InvalidArgument.code(),
        };

        let mask = crate::artifacts::detect_artifacts(data_slice, fs, &opts_rust);
        if mask.len() != n {
            return ErrorCode::ShapeMismatch.code();
        }
        let mask_u8: Vec<u8> = mask.iter().map(|&b| b as u8).collect();
        std::ptr::copy_nonoverlapping(mask_u8.as_ptr(), mask_out, n);
        ErrorCode::Ok.code()
    }));
    result.unwrap_or(ErrorCode::Panic.code())
}

// -------------------------------------------------------------------------
// dynamo_compute_baseline — per-frequency percentile baseline
// -------------------------------------------------------------------------

/// Compute the per-frequency baseline used in TFpeak extraction.
///
/// Inputs (all row-major):
///   `spect`           — shape (n_freqs, n_times) double; baseline is the
///                       `baseline_ptile`-th percentile over the valid
///                       columns of each row (zeros treated as NaN).
///   `stimes`          — length n_times; window-center times.
///   `t_data`          — length n_data; the EEG time grid that the
///                       `baseline_exclude` mask is defined on.
///   `baseline_exclude` — length n_data u8 (1 = exclude); nearest-neighbor
///                       interpolated onto `stimes` columns.
///   `baseline_range_lo` / `baseline_range_hi` — time-range trimming
///                       applied to `stimes`.
///   `baseline_ptile`  — percentile in [0, 100] (Hyndman-Fan #5, matching
///                       MATLAB `prctile(..., baseline_ptile, 2)`).
///
/// Caller-allocated output `baseline_out` is length `n_freqs`.
///
/// # Safety
/// All input pointers must point to readable buffers of the declared
/// lengths. `baseline_out` must be writable for `n_freqs` f64s.
#[no_mangle]
pub unsafe extern "C" fn dynamo_compute_baseline(
    spect: *const f64,
    n_freqs: usize,
    n_times: usize,
    stimes: *const f64,
    t_data: *const f64,
    n_data: usize,
    baseline_exclude: *const u8,
    baseline_range_lo: f64,
    baseline_range_hi: f64,
    baseline_ptile: f64,
    baseline_out: *mut f64,
) -> c_int {
    let result = catch_unwind(AssertUnwindSafe(|| {
        if baseline_out.is_null() {
            return ErrorCode::NullPointer.code();
        }
        if n_freqs == 0 || n_times == 0 || n_data == 0 {
            return ErrorCode::InvalidArgument.code();
        }
        let spect_slice = match ptr_as_slice::<f64>(spect, n_freqs * n_times) {
            Some(s) => s,
            None => return ErrorCode::NullPointer.code(),
        };
        let stimes_slice = match ptr_as_slice::<f64>(stimes, n_times) {
            Some(s) => s,
            None => return ErrorCode::NullPointer.code(),
        };
        let t_data_slice = match ptr_as_slice::<f64>(t_data, n_data) {
            Some(s) => s,
            None => return ErrorCode::NullPointer.code(),
        };
        let excl_slice = match ptr_as_slice::<u8>(baseline_exclude, n_data) {
            Some(s) => s,
            None => return ErrorCode::NullPointer.code(),
        };

        let spect_view = match ArrayView2::from_shape((n_freqs, n_times), spect_slice) {
            Ok(v) => v,
            Err(_) => return ErrorCode::ShapeMismatch.code(),
        };
        let stimes_view = ArrayView1::from(stimes_slice);
        let t_data_view = ArrayView1::from(t_data_slice);
        let excl_vec: Vec<bool> = excl_slice.iter().map(|&b| b != 0).collect();

        let baseline = match crate::baseline::compute_baseline(
            spect_view, stimes_view, t_data_view,
            &excl_vec, (baseline_range_lo, baseline_range_hi), baseline_ptile,
        ) {
            Ok(b) => b,
            Err(_) => return ErrorCode::KernelError.code(),
        };

        // Shape is (F, 1); flatten and copy.
        if baseline.nrows() != n_freqs || baseline.ncols() != 1 {
            return ErrorCode::ShapeMismatch.code();
        }
        let std_baseline = baseline.as_standard_layout().to_owned();
        let baseline_slice = std_baseline.as_slice().unwrap_or(&[]);
        if baseline_slice.len() != n_freqs {
            return ErrorCode::ShapeMismatch.code();
        }
        std::ptr::copy_nonoverlapping(baseline_slice.as_ptr(), baseline_out, n_freqs);
        ErrorCode::Ok.code()
    }));
    result.unwrap_or(ErrorCode::Panic.code())
}

// -------------------------------------------------------------------------
// dynamo_so_power — SO-power time-series from a pre-computed spectrogram
// -------------------------------------------------------------------------

#[repr(C)]
pub struct SoPowerIn {
    pub spect_ptr:          *const f64,
    pub n_freqs:            usize,
    pub n_times:            usize,
    pub stimes_ptr:         *const f64,    // len n_times
    pub sfreqs_ptr:         *const f64,    // len n_freqs
    pub eeg_times_ptr:      *const f64,    // len n_data
    pub n_data:             usize,
    pub isexcluded_ptr:     *const u8,     // len n_data (0/1)
    pub stage_times_ptr:    *const f64,    // len n_stages (may be 0)
    pub stage_vals_ptr:     *const f64,    // len n_stages
    pub n_stages:           usize,
    pub time_range_lo:      f64,
    pub time_range_hi:      f64,
    pub outlier_threshold:  f64,
    pub retain_fs:          u8,            // 0/1
    /// Null-terminated ASCII norm method spec: e.g. "p2shift1234",
    /// "percent", "none". Same parser as `NormMethod::parse`.
    pub norm_method_ptr:    *const u8,
    pub norm_method_len:    usize,
}

#[repr(C)]
pub struct SoPowerOut {
    pub so_power_norm_ptr:    *mut f64,
    pub so_power_times_ptr:   *mut f64,
    pub so_power_stages_ptr:  *mut f64,
    /// Output length (matches all three arrays above).
    pub n_out:                usize,
    /// ptile result: 0 = none, 1 = single (`ptile_value[0]`),
    /// 2 = pair (`ptile_value[0]`, `ptile_value[1]`).
    pub ptile_kind:           u32,
    pub ptile_value:          [f64; 2],
}

fn empty_so_power_out() -> SoPowerOut {
    SoPowerOut {
        so_power_norm_ptr:   std::ptr::null_mut(),
        so_power_times_ptr:  std::ptr::null_mut(),
        so_power_stages_ptr: std::ptr::null_mut(),
        n_out: 0,
        ptile_kind: 0,
        ptile_value: [0.0, 0.0],
    }
}

/// Compute the SO-power time series and stage / outlier-masked
/// normalization. Mirrors `crate::so_power::so_power_from_spectrogram`
/// (and pydynamo `compute_so_power`); MATLAB analog is `computeSOpower.m`.
///
/// Output pointers are `Box::leak`-allocated; caller frees with
/// `dynamo_free_buffer_f64(ptr, n_out)` for each of the three arrays.
///
/// # Safety
/// `in_` and `out` must be valid. All array pointers must back the
/// declared lengths.
#[no_mangle]
pub unsafe extern "C" fn dynamo_so_power(
    in_: *const SoPowerIn,
    out: *mut SoPowerOut,
) -> c_int {
    let result = catch_unwind(AssertUnwindSafe(|| {
        let input = match NonNull::new(in_ as *mut SoPowerIn) {
            Some(p) => &*p.as_ptr(),
            None => return ErrorCode::NullPointer.code(),
        };
        let output = match NonNull::new(out) {
            Some(p) => &mut *p.as_ptr(),
            None => return ErrorCode::NullPointer.code(),
        };
        *output = empty_so_power_out();

        if input.n_freqs == 0 || input.n_times == 0 || input.n_data == 0 {
            return ErrorCode::InvalidArgument.code();
        }
        let spect_slice = match ptr_as_slice::<f64>(input.spect_ptr, input.n_freqs * input.n_times) {
            Some(s) => s,
            None => return ErrorCode::NullPointer.code(),
        };
        let stimes = match ptr_as_slice::<f64>(input.stimes_ptr, input.n_times) {
            Some(s) => s, None => return ErrorCode::NullPointer.code(),
        };
        let sfreqs = match ptr_as_slice::<f64>(input.sfreqs_ptr, input.n_freqs) {
            Some(s) => s, None => return ErrorCode::NullPointer.code(),
        };
        let eeg_times = match ptr_as_slice::<f64>(input.eeg_times_ptr, input.n_data) {
            Some(s) => s, None => return ErrorCode::NullPointer.code(),
        };
        let excl_u8 = match ptr_as_slice::<u8>(input.isexcluded_ptr, input.n_data) {
            Some(s) => s, None => return ErrorCode::NullPointer.code(),
        };
        let isexcluded: Vec<bool> = excl_u8.iter().map(|&b| b != 0).collect();
        let stage_times = match ptr_as_slice::<f64>(input.stage_times_ptr, input.n_stages) {
            Some(s) => s, None => return ErrorCode::NullPointer.code(),
        };
        let stage_vals = match ptr_as_slice::<f64>(input.stage_vals_ptr, input.n_stages) {
            Some(s) => s, None => return ErrorCode::NullPointer.code(),
        };

        let nm_bytes = match ptr_as_slice::<u8>(input.norm_method_ptr, input.norm_method_len) {
            Some(s) => s, None => return ErrorCode::NullPointer.code(),
        };
        let nm_str = match std::str::from_utf8(nm_bytes) {
            Ok(s) => s, Err(_) => return ErrorCode::InvalidArgument.code(),
        };
        let nm = match crate::so_power::NormMethod::parse(nm_str) {
            Some(n) => n, None => return ErrorCode::InvalidArgument.code(),
        };

        let kernel_out = match crate::so_power::so_power_from_spectrogram(
            spect_slice, input.n_freqs, input.n_times,
            stimes, sfreqs, eeg_times, &isexcluded,
            stage_times, stage_vals,
            (input.time_range_lo, input.time_range_hi),
            input.outlier_threshold,
            &nm,
            input.retain_fs != 0,
        ) {
            Ok(o) => o,
            Err(_) => return ErrorCode::KernelError.code(),
        };

        output.n_out = kernel_out.so_power_norm.len();
        if kernel_out.so_power_times.len() != output.n_out
            || kernel_out.so_power_stages.len() != output.n_out
        {
            return ErrorCode::ShapeMismatch.code();
        }
        output.so_power_norm_ptr   = leak_vec_f64(kernel_out.so_power_norm);
        output.so_power_times_ptr  = leak_vec_f64(kernel_out.so_power_times);
        output.so_power_stages_ptr = leak_vec_f64(kernel_out.so_power_stages);
        match kernel_out.ptile {
            None => {
                output.ptile_kind = 0;
            }
            Some(crate::so_power::PtileUsed::Single(p)) => {
                output.ptile_kind = 1;
                output.ptile_value[0] = p;
            }
            Some(crate::so_power::PtileUsed::Pair(a, b)) => {
                output.ptile_kind = 2;
                output.ptile_value[0] = a;
                output.ptile_value[1] = b;
            }
        }
        ErrorCode::Ok.code()
    }));
    result.unwrap_or(ErrorCode::Panic.code())
}

// -------------------------------------------------------------------------
// dynamo_so_phase — SO-phase time-series from raw EEG
// -------------------------------------------------------------------------

#[repr(C)]
pub struct SoPhaseIn {
    pub eeg_ptr:           *const f64,
    pub eeg_times_ptr:     *const f64,
    pub isexcluded_ptr:    *const u8,
    pub n_data:            usize,
    /// SOS filter coefficients, scipy layout: (n_sections, 6) flattened
    /// row-major [b0 b1 b2 a0 a1 a2] per section.
    pub sos_ptr:           *const f64,
    pub n_sections:        usize,
    pub stage_times_ptr:   *const f64,
    pub stage_vals_ptr:    *const f64,
    pub n_stages:          usize,
}

#[repr(C)]
pub struct SoPhaseOut {
    pub so_phase_ptr:        *mut f64,    // length n_data; unwrapped, NaN at excluded
    pub so_phase_times_ptr:  *mut f64,    // length n_data (echo of eeg_times)
    pub so_phase_stages_ptr: *mut f64,    // length n_data
    pub filtdata_ptr:        *mut f64,    // length n_data; filtered EEG with NaN at excluded
    pub n_out:               usize,
}

fn empty_so_phase_out() -> SoPhaseOut {
    SoPhaseOut {
        so_phase_ptr:        std::ptr::null_mut(),
        so_phase_times_ptr:  std::ptr::null_mut(),
        so_phase_stages_ptr: std::ptr::null_mut(),
        filtdata_ptr:        std::ptr::null_mut(),
        n_out: 0,
    }
}

/// Compute unwrapped SO-phase from raw EEG via SOS bandpass + Hilbert +
/// atan2 + unwrap + NaN-at-excluded + stage interp. Mirrors pydynamo
/// `compute_so_phase`; MATLAB analog is `computeSOphase.m`.
///
/// All four output arrays are length `n_data` and `Box::leak`-allocated.
/// Caller frees with `dynamo_free_buffer_f64(ptr, n_data)` each.
///
/// `sos_ptr` must point to `n_sections * 6` f64s in scipy layout
/// `[b0 b1 b2 a0 a1 a2]` per section.
///
/// # Safety
/// All declared array pointers must back the corresponding lengths.
#[no_mangle]
pub unsafe extern "C" fn dynamo_so_phase(
    in_: *const SoPhaseIn,
    out: *mut SoPhaseOut,
) -> c_int {
    let result = catch_unwind(AssertUnwindSafe(|| {
        let input = match NonNull::new(in_ as *mut SoPhaseIn) {
            Some(p) => &*p.as_ptr(),
            None => return ErrorCode::NullPointer.code(),
        };
        let output = match NonNull::new(out) {
            Some(p) => &mut *p.as_ptr(),
            None => return ErrorCode::NullPointer.code(),
        };
        *output = empty_so_phase_out();

        if input.n_data == 0 || input.n_sections == 0 {
            return ErrorCode::InvalidArgument.code();
        }
        let eeg = match ptr_as_slice::<f64>(input.eeg_ptr, input.n_data) {
            Some(s) => s, None => return ErrorCode::NullPointer.code(),
        };
        let eeg_times = match ptr_as_slice::<f64>(input.eeg_times_ptr, input.n_data) {
            Some(s) => s, None => return ErrorCode::NullPointer.code(),
        };
        let excl_u8 = match ptr_as_slice::<u8>(input.isexcluded_ptr, input.n_data) {
            Some(s) => s, None => return ErrorCode::NullPointer.code(),
        };
        let isexcluded: Vec<bool> = excl_u8.iter().map(|&b| b != 0).collect();
        let sos_flat = match ptr_as_slice::<f64>(input.sos_ptr, input.n_sections * 6) {
            Some(s) => s, None => return ErrorCode::NullPointer.code(),
        };
        let sos: Vec<[f64; 6]> = (0..input.n_sections)
            .map(|i| {
                let b = i * 6;
                [sos_flat[b], sos_flat[b+1], sos_flat[b+2], sos_flat[b+3], sos_flat[b+4], sos_flat[b+5]]
            })
            .collect();
        let stage_times = match ptr_as_slice::<f64>(input.stage_times_ptr, input.n_stages) {
            Some(s) => s, None => return ErrorCode::NullPointer.code(),
        };
        let stage_vals = match ptr_as_slice::<f64>(input.stage_vals_ptr, input.n_stages) {
            Some(s) => s, None => return ErrorCode::NullPointer.code(),
        };

        let kernel_out = match crate::so_phase::so_phase_from_eeg(
            eeg, eeg_times, &isexcluded, &sos, stage_times, stage_vals,
        ) {
            Ok(o) => o,
            Err(_) => return ErrorCode::KernelError.code(),
        };

        if kernel_out.so_phase_unwrapped.len() != input.n_data
            || kernel_out.so_phase_times.len()  != input.n_data
            || kernel_out.so_phase_stages.len() != input.n_data
            || kernel_out.filtdata.len()        != input.n_data
        {
            return ErrorCode::ShapeMismatch.code();
        }

        output.n_out               = input.n_data;
        output.so_phase_ptr        = leak_vec_f64(kernel_out.so_phase_unwrapped);
        output.so_phase_times_ptr  = leak_vec_f64(kernel_out.so_phase_times);
        output.so_phase_stages_ptr = leak_vec_f64(kernel_out.so_phase_stages);
        output.filtdata_ptr        = leak_vec_f64(kernel_out.filtdata);
        ErrorCode::Ok.code()
    }));
    result.unwrap_or(ErrorCode::Panic.code())
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

/// See [`dynamo_free_buffer_f64`]. Used for the CSR offset arrays
/// (`height_data_offsets`, `boundary_offsets`) returned in `ExtractTfpeaksOut`.
#[no_mangle]
pub unsafe extern "C" fn dynamo_free_buffer_u64(ptr: *mut u64, len: usize) {
    drop_leaked_u64(ptr, len);
}

// -------------------------------------------------------------------------
// spline_basis FFI
// -------------------------------------------------------------------------

/// Input descriptor for [`dynamo_spline_basis_fit`].
///
/// Layout mirrors MATLAB `spline_basis.m` semantics:
///   * `soph` is `(n_x, n_y)` row-major, where `n_x = feat_bins.len()` and
///     `n_y = freq_bins.len()`. This is the **transposed** orientation of
///     the canonical SOPH (the MATLAB-side wrapper does the transpose).
///   * `internal_knots_x`/`y` are the pre-`augknt` knot vectors.
///   * `order = 4` and `boundary_multiplicity = 3` reproduce DYNAM-O.
#[repr(C)]
pub struct SplineBasisIn {
    pub soph_ptr:               *const f64,
    pub n_x:                    usize,
    pub n_y:                    usize,
    pub feat_bins_ptr:          *const f64,
    pub freq_bins_ptr:          *const f64,
    pub internal_knots_x_ptr:   *const f64,
    pub n_internal_knots_x:     usize,
    pub internal_knots_y_ptr:   *const f64,
    pub n_internal_knots_y:     usize,
    pub order:                  u32,
    pub boundary_multiplicity:  u32,
}

#[repr(C)]
pub struct SplineBasisOut {
    /// `(m_y, m_x)` row-major. Matches MATLAB `squeeze(spline_obj.coefs)'`.
    pub coefs_ptr:        *mut f64,
    pub m_y:              usize,
    pub m_x:              usize,
    /// `(n_x, n_y)` row-major. Matches MATLAB `splinefit` (= `size(SOPH')`).
    pub splinefit_ptr:    *mut f64,
    /// Augmented knot vectors.
    pub knots_x_aug_ptr:  *mut f64,
    pub n_knots_x_aug:    usize,
    pub knots_y_aug_ptr:  *mut f64,
    pub n_knots_y_aug:    usize,
}

fn empty_spline_basis_out() -> SplineBasisOut {
    SplineBasisOut {
        coefs_ptr:       std::ptr::null_mut(),
        m_y: 0, m_x: 0,
        splinefit_ptr:   std::ptr::null_mut(),
        knots_x_aug_ptr: std::ptr::null_mut(),
        n_knots_x_aug:   0,
        knots_y_aug_ptr: std::ptr::null_mut(),
        n_knots_y_aug:   0,
    }
}

/// Fit a bivariate tensor-product B-spline on a regular grid.
///
/// Parity-tested against MATLAB `spap2` to f64 round-off (see
/// `tests/spline_basis_parity.rs`).
///
/// All four output arrays are `Box::leak`-allocated. Caller frees with
/// `dynamo_free_buffer_f64`:
///   * `coefs_ptr`       length `m_y * m_x`
///   * `splinefit_ptr`   length `n_x * n_y`
///   * `knots_x_aug_ptr` length `n_knots_x_aug`
///   * `knots_y_aug_ptr` length `n_knots_y_aug`
///
/// # Safety
/// All declared array pointers must back the corresponding lengths.
#[no_mangle]
pub unsafe extern "C" fn dynamo_spline_basis_fit(
    in_: *const SplineBasisIn,
    out: *mut SplineBasisOut,
) -> c_int {
    let result = catch_unwind(AssertUnwindSafe(|| {
        let input = match NonNull::new(in_ as *mut SplineBasisIn) {
            Some(p) => &*p.as_ptr(),
            None => return ErrorCode::NullPointer.code(),
        };
        let output = match NonNull::new(out) {
            Some(p) => &mut *p.as_ptr(),
            None => return ErrorCode::NullPointer.code(),
        };
        *output = empty_spline_basis_out();

        if input.n_x == 0 || input.n_y == 0
            || input.n_internal_knots_x == 0 || input.n_internal_knots_y == 0
        {
            return ErrorCode::InvalidArgument.code();
        }

        let soph_flat = match ptr_as_slice::<f64>(input.soph_ptr, input.n_x * input.n_y) {
            Some(s) => s, None => return ErrorCode::NullPointer.code(),
        };
        let feat_bins = match ptr_as_slice::<f64>(input.feat_bins_ptr, input.n_x) {
            Some(s) => s, None => return ErrorCode::NullPointer.code(),
        };
        let freq_bins = match ptr_as_slice::<f64>(input.freq_bins_ptr, input.n_y) {
            Some(s) => s, None => return ErrorCode::NullPointer.code(),
        };
        let ikx = match ptr_as_slice::<f64>(input.internal_knots_x_ptr, input.n_internal_knots_x) {
            Some(s) => s, None => return ErrorCode::NullPointer.code(),
        };
        let iky = match ptr_as_slice::<f64>(input.internal_knots_y_ptr, input.n_internal_knots_y) {
            Some(s) => s, None => return ErrorCode::NullPointer.code(),
        };

        let soph = match ArrayView2::from_shape((input.n_x, input.n_y), soph_flat) {
            Ok(a) => a,
            Err(_) => return ErrorCode::ShapeMismatch.code(),
        };

        let kernel_out = match crate::spline_basis::fit_tensor_product_spline(
            soph,
            feat_bins,
            freq_bins,
            ikx,
            iky,
            input.order as usize,
            input.boundary_multiplicity as usize,
        ) {
            Ok(o) => o,
            Err(_) => return ErrorCode::KernelError.code(),
        };

        let (my, mx) = kernel_out.coefs.dim();
        let (nx, ny) = kernel_out.splinefit.dim();
        if nx != input.n_x || ny != input.n_y {
            return ErrorCode::ShapeMismatch.code();
        }

        let coefs_vec: Vec<f64> = kernel_out.coefs.iter().copied().collect();
        let splinefit_vec: Vec<f64> = kernel_out.splinefit.iter().copied().collect();

        output.m_y = my;
        output.m_x = mx;
        output.coefs_ptr       = leak_vec_f64(coefs_vec);
        output.splinefit_ptr   = leak_vec_f64(splinefit_vec);
        output.n_knots_x_aug   = kernel_out.knots_x_aug.len();
        output.n_knots_y_aug   = kernel_out.knots_y_aug.len();
        output.knots_x_aug_ptr = leak_vec_f64(kernel_out.knots_x_aug);
        output.knots_y_aug_ptr = leak_vec_f64(kernel_out.knots_y_aug);
        ErrorCode::Ok.code()
    }));
    result.unwrap_or(ErrorCode::Panic.code())
}

// -------------------------------------------------------------------------
// paramfit FFI: rotgauss_fit + vmgauss_fit
// -------------------------------------------------------------------------

/// Input descriptor for the two paramfit kernels (`dynamo_rotgauss_fit`
/// and `dynamo_vmgauss_fit`).
///
/// SOPH is `(n_y, n_x)` row-major where `n_y = freq_bins.len()` and
/// `n_x = feat_bins.len()`. This matches the canonical MATLAB SOPH
/// `(n_freqs, n_features)` column-major byte-for-byte — pass `mxGetPr`
/// directly without copying.
///
/// Initial/lower/upper are `(n_modes, 6)` row-major:
///   * power: `[amp, fmean, fstd, pmean,    pstd,      theta]`
///   * phase: `[amp, fmean, fstd, phasepref, recikappa, theta]`
///
/// `bg_initial`, `bg_lower`, `bg_upper` are 3-vectors `[xxx, yyy, zzz]`.
#[repr(C)]
pub struct ParamFitIn {
    pub soph_ptr:        *const f64,
    pub n_y:             usize,
    pub n_x:             usize,
    pub feat_bins_ptr:   *const f64,
    pub freq_bins_ptr:   *const f64,
    pub initial_ptr:     *const f64,    // (n_modes * 6)
    pub lower_ptr:       *const f64,
    pub upper_ptr:       *const f64,
    pub n_modes:         usize,
    pub bg_initial:      [f64; 3],
    pub bg_lower:        [f64; 3],
    pub bg_upper:        [f64; 3],
    pub max_iters:       u32,
    /// vmGauss-only: nonzero = row-normalize the assembled model (matches
    /// MATLAB `fit_vmGauss.m`'s `unit_row=true`). Ignored by rotgauss_fit.
    pub unit_row:        u32,
}

#[repr(C)]
pub struct ParamFitOutFFI {
    /// Final parameters, `(n_modes, 6)` row-major. Allocated, length `n_modes * 6`.
    pub params_ptr:      *mut f64,
    /// Background-plane coefficients `[xxx, yyy, zzz]`.
    pub background:      [f64; 3],
    /// Model reconstruction on the input grid, `(n_y, n_x)` row-major.
    pub model_ptr:       *mut f64,
    pub n_y:             usize,
    pub n_x:             usize,
    /// gof.sse / rsquare / adjrsquare / rmse / dfe / dfm.
    pub gof_sse:         f64,
    pub gof_rsquare:     f64,
    pub gof_adjrsquare:  f64,
    pub gof_rmse:        f64,
    pub gof_dfe:         f64,
    pub gof_dfm:         f64,
    pub iters_used:      u32,
}

fn empty_paramfit_out() -> ParamFitOutFFI {
    ParamFitOutFFI {
        params_ptr:     std::ptr::null_mut(),
        background:     [0.0; 3],
        model_ptr:      std::ptr::null_mut(),
        n_y: 0, n_x: 0,
        gof_sse: 0.0, gof_rsquare: 0.0, gof_adjrsquare: 0.0,
        gof_rmse: 0.0, gof_dfe: 0.0, gof_dfm: 0.0,
        iters_used: 0,
    }
}

unsafe fn paramfit_common_setup(
    input: &ParamFitIn,
) -> Result<(ArrayView2<'static, f64>, &'static [f64], &'static [f64],
             ArrayView2<'static, f64>, ArrayView2<'static, f64>, ArrayView2<'static, f64>),
            c_int>
{
    if input.n_y == 0 || input.n_x == 0 || input.n_modes == 0 {
        return Err(ErrorCode::InvalidArgument.code());
    }
    let soph_flat = match ptr_as_slice::<f64>(input.soph_ptr, input.n_y * input.n_x) {
        Some(s) => s, None => return Err(ErrorCode::NullPointer.code()),
    };
    let feat_bins = match ptr_as_slice::<f64>(input.feat_bins_ptr, input.n_x) {
        Some(s) => s, None => return Err(ErrorCode::NullPointer.code()),
    };
    let freq_bins = match ptr_as_slice::<f64>(input.freq_bins_ptr, input.n_y) {
        Some(s) => s, None => return Err(ErrorCode::NullPointer.code()),
    };
    let init = match ptr_as_slice::<f64>(input.initial_ptr, input.n_modes * 6) {
        Some(s) => s, None => return Err(ErrorCode::NullPointer.code()),
    };
    let lo = match ptr_as_slice::<f64>(input.lower_ptr, input.n_modes * 6) {
        Some(s) => s, None => return Err(ErrorCode::NullPointer.code()),
    };
    let hi = match ptr_as_slice::<f64>(input.upper_ptr, input.n_modes * 6) {
        Some(s) => s, None => return Err(ErrorCode::NullPointer.code()),
    };
    let soph = ArrayView2::from_shape((input.n_y, input.n_x), soph_flat)
        .map_err(|_| ErrorCode::ShapeMismatch.code())?;
    let initial = ArrayView2::from_shape((input.n_modes, 6), init)
        .map_err(|_| ErrorCode::ShapeMismatch.code())?;
    let lower = ArrayView2::from_shape((input.n_modes, 6), lo)
        .map_err(|_| ErrorCode::ShapeMismatch.code())?;
    let upper = ArrayView2::from_shape((input.n_modes, 6), hi)
        .map_err(|_| ErrorCode::ShapeMismatch.code())?;
    Ok((soph, feat_bins, freq_bins, initial, lower, upper))
}

fn fill_paramfit_out(out: &mut ParamFitOutFFI, kernel: crate::paramfit::ParamFitOut) {
    let (n_modes, _) = kernel.params.dim();
    let (ny, nx) = kernel.model_soph.dim();
    out.background = kernel.background;
    out.n_y = ny;
    out.n_x = nx;
    out.gof_sse        = kernel.gof.sse;
    out.gof_rsquare    = kernel.gof.rsquare;
    out.gof_adjrsquare = kernel.gof.adjrsquare;
    out.gof_rmse       = kernel.gof.rmse;
    out.gof_dfe        = kernel.gof.dfe;
    out.gof_dfm        = kernel.gof.dfm;
    out.iters_used     = kernel.iters_used;
    let params_vec: Vec<f64> = kernel.params.iter().copied().collect();
    let model_vec: Vec<f64> = kernel.model_soph.iter().copied().collect();
    out.params_ptr = leak_vec_f64(params_vec);
    out.model_ptr  = leak_vec_f64(model_vec);
    let _ = n_modes;
}

/// Fit a rotated-Gaussian-mixture model + linear background plane to a SOPH
/// histogram. Mirrors MATLAB `fit_rotGauss` (without `prepareSurfaceData`'s
/// NaN drop — caller must hand us a finite-valued grid).
///
/// # Safety
/// All pointers must back the declared lengths.
#[no_mangle]
pub unsafe extern "C" fn dynamo_rotgauss_fit(
    in_: *const ParamFitIn,
    out: *mut ParamFitOutFFI,
) -> c_int {
    let result = catch_unwind(AssertUnwindSafe(|| {
        let input = match NonNull::new(in_ as *mut ParamFitIn) {
            Some(p) => &*p.as_ptr(),
            None => return ErrorCode::NullPointer.code(),
        };
        let output = match NonNull::new(out) {
            Some(p) => &mut *p.as_ptr(),
            None => return ErrorCode::NullPointer.code(),
        };
        *output = empty_paramfit_out();

        let (soph, feat_bins, freq_bins, initial, lower, upper) =
            match paramfit_common_setup(input) {
                Ok(t) => t,
                Err(e) => return e,
            };

        let kernel_out = match crate::paramfit::rot_gauss::fit_rotgauss(
            soph, feat_bins, freq_bins,
            initial, lower, upper,
            input.bg_initial, input.bg_lower, input.bg_upper,
            input.max_iters,
        ) {
            Ok(o) => o,
            Err(_) => return ErrorCode::KernelError.code(),
        };
        fill_paramfit_out(output, kernel_out);
        ErrorCode::Ok.code()
    }));
    result.unwrap_or(ErrorCode::Panic.code())
}

/// Fit a von-Mises × Gaussian mixture + sinusoidal background to a SOPH
/// histogram. Mirrors MATLAB `fit_vmGauss` **without** the per-row
/// normalization step in `normalized_vmGauss.m` (see
/// `src/paramfit/vm_gauss.rs` module docs).
///
/// # Safety
/// All pointers must back the declared lengths.
#[no_mangle]
pub unsafe extern "C" fn dynamo_vmgauss_fit(
    in_: *const ParamFitIn,
    out: *mut ParamFitOutFFI,
) -> c_int {
    let result = catch_unwind(AssertUnwindSafe(|| {
        let input = match NonNull::new(in_ as *mut ParamFitIn) {
            Some(p) => &*p.as_ptr(),
            None => return ErrorCode::NullPointer.code(),
        };
        let output = match NonNull::new(out) {
            Some(p) => &mut *p.as_ptr(),
            None => return ErrorCode::NullPointer.code(),
        };
        *output = empty_paramfit_out();

        let (soph, feat_bins, freq_bins, initial, lower, upper) =
            match paramfit_common_setup(input) {
                Ok(t) => t,
                Err(e) => return e,
            };

        let kernel_out = match crate::paramfit::vm_gauss::fit_vmgauss(
            soph, feat_bins, freq_bins,
            initial, lower, upper,
            input.bg_initial, input.bg_lower, input.bg_upper,
            input.max_iters,
            input.unit_row != 0,
        ) {
            Ok(o) => o,
            Err(_) => return ErrorCode::KernelError.code(),
        };
        fill_paramfit_out(output, kernel_out);
        ErrorCode::Ok.code()
    }));
    result.unwrap_or(ErrorCode::Panic.code())
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
//     area: std::ptr::null_mut(),
//     peakiness: std::ptr::null_mut(),
//     height_data: std::ptr::null_mut(),
//     n_height_data_elems: 0,
//     height_data_offsets: std::ptr::null_mut(),
//     boundaries_xy: std::ptr::null_mut(),
//     n_boundary_pixels: 0,
//     boundary_offsets: std::ptr::null_mut(),
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
//     dynamo_free_buffer_f64(out.area,         out.n_peaks);
//     dynamo_free_buffer_f64(out.peakiness,    out.n_peaks);
//     dynamo_free_buffer_f64(out.height_data,  out.n_height_data_elems);
//     dynamo_free_buffer_u64(out.height_data_offsets, out.n_peaks + 1);
//     dynamo_free_buffer_f64(out.boundaries_xy, out.n_boundary_pixels * 2);
//     dynamo_free_buffer_u64(out.boundary_offsets, out.n_peaks + 1);
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
            progress_cb: None,
        };
        let mut out = empty_extract_out();
        unsafe {
            let rc = dynamo_extract_tfpeaks(&in_, &mut out);
            assert_eq!(rc, ErrorCode::Ok.code());
            assert_eq!(out.n_peaks, 0);
        }
    }
}
