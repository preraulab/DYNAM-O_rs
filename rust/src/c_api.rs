//! C ABI surface for the MATLAB MEX wrappers and any non-Rust consumer.
//!
//! Entry points:
//!   * [`dynamo_c_abi_version`], [`dynamo_version`]: build identity
//!   * [`dynamo_extract_tfpeaks`]: watershed, merge, trim, peak properties
//!   * [`dynamo_refine_peaks`]: hann_event_spectra + refine_from_spectra
//!   * [`dynamo_tfpeak_histogram`]: SO-power / SO-phase histogram
//!   * [`dynamo_mask_spectrogram`]: pass-2 mask from pass-1 labels
//!   * [`dynamo_multitaper_spectrogram`], [`dynamo_dpss`]
//!   * [`dynamo_detect_artifacts`], [`dynamo_compute_baseline`]
//!   * [`dynamo_so_power`], [`dynamo_so_phase`]
//!   * [`dynamo_spline_basis_fit`], [`dynamo_rotgauss_fit`], [`dynamo_vmgauss_fit`]
//!
//! Four helpers release the callee-allocated output arrays:
//! [`dynamo_free_buffer_f64`], [`dynamo_free_buffer_i64`],
//! [`dynamo_free_buffer_u8`], [`dynamo_free_buffer_u64`].
//!
//! ## Memory model
//!
//! Every output pointer is handed out via `Box::into_raw(vec.into_boxed_slice())`.
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
//! See [`ErrorCode`]. Zero = success. Negative integers are well-defined
//! errors. We never return positive codes.

use ndarray::{ArrayView1, ArrayView2};
use std::os::raw::{c_char, c_int};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr::NonNull;
use std::slice;

/// C ABI version. The ABI contract is **additive-only**: new symbols and
/// new `#define`s may appear, but existing signatures, struct layouts,
/// and enum values never change under the same value. Bump this on any
/// breaking change. Numeric-semantics changes that keep the ABI stable
/// (e.g. a width reparameterization) do NOT bump this. They
/// bump the crate semver, which callers observe via [`dynamo_version`].
pub const DYNAMO_C_ABI_VERSION: u32 = 1;

/// Runtime accessor for [`DYNAMO_C_ABI_VERSION`], for `dlopen` /
/// `loadlibrary` consumers that cannot see the compile-time constant.
#[no_mangle]
pub extern "C" fn dynamo_c_abi_version() -> u32 {
    DYNAMO_C_ABI_VERSION
}

/// The kernel build identity as a NUL-terminated static string in the
/// DYNAM-O provenance grammar `<semver>+<sha12>[.dirty]` (or
/// `<semver>+unknown`). This is the `kernel_version` every consumer
/// (MEX bridge, PyO3, app/CLI) should record next to its outputs. The
/// pointer is `'static`. Callers must NOT free it.
#[no_mangle]
pub extern "C" fn dynamo_version() -> *const c_char {
    static VERSION_CSTR: &str =
        concat!(env!("CARGO_PKG_VERSION"), "+", env!("DYNAMO_GIT_SHA"), "\0");
    VERSION_CSTR.as_ptr() as *const c_char
}

/// Status/error codes for the C ABI. All non-zero values are negative.
#[repr(i32)]
#[derive(Clone, Copy, Debug)]
pub enum ErrorCode {
    /// Success.
    Ok = 0,
    /// A required pointer was null.
    NullPointer = -1,
    /// Array lengths are inconsistent with the declared dimensions.
    ShapeMismatch = -2,
    /// The numeric kernel returned an error.
    KernelError = -3,
    /// The kernel panicked. The panic was caught at the FFI boundary.
    Panic = -4,
    /// A scalar argument or enum code is out of range.
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
///      border fill (see `expand_labels_distance`), resize labels up, trim,
///      regionprops.
///   3. Concatenate per-segment peaks.
///   4. Apply filterStatsTable: keep peaks where
///        dur_min < Duration < dur_max,
///        bw_min  < Bandwidth < bw_max,
///        freq_min < PeakFrequency < freq_max,
///        pow2db(Height) > ht_db_min.
///
/// Sentinel values that select a default:
///   * `seg_time` <= 0 → 30.0 s
///   * `downsample_f` / `downsample_t` = 0 → 1 (no downsampling)
///   * `max_merges` = 0 → +∞
///   * `freq_max` = 0 → +∞
/// `dur_max`, `bw_max` and `ht_db_min` have no sentinel. Pass +∞, +∞ and
/// -∞ respectively to disable those cuts.
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
    /// Border-fill mode applied after the merge. 0 = MATLAB paint in label
    /// order (8-connected 1-pixel dilation, higher labels overwrite lower),
    /// matching extractTFPeaks.m. N > 0 = skimage-style expand_labels
    /// (breadth-first fill out to N pixels, lower label wins ties).
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
/// All `*mut` pointers are callee-allocated. The caller must free each with the
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
/// Both offset arrays have length `n_peaks + 1` (always, even when n_peaks
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
    /// Peakiness = N/(N-1) * (max - mean) / (max - min) of the region's N
    /// pixels per peak, unitless in [0, 1] (0 = flat plateau, 1 = spike;
    /// small-sample corrected). Length n_peaks.
    pub peakiness: *mut f64,
    /// Flattened per-peak pixel values, length `n_height_data_elems`.
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

#[cfg(test)]
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
// dynamo_extract_tfpeaks
// -------------------------------------------------------------------------

/// Segment length in seconds used when the caller passes `seg_time` <= 0.
const DEFAULT_SEG_TIME_S: f64 = 30.0;

/// Run the full TF-peak extraction pipeline on a whole spectrogram, which is
/// split internally into `seg_time`-second segments.
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

        let n_freqs = input.n_freqs;
        let n_times = input.n_times;
        if n_freqs == 0 || n_times == 0 {
            *output = empty_extract_out();
            return ErrorCode::Ok.code();
        }
        let n_pixels = n_freqs * n_times;
        let spect_slice = match ptr_as_slice::<f64>(input.spect_ptr, n_pixels) {
            Some(s) => s,
            None => return ErrorCode::NullPointer.code(),
        };
        let stimes_slice = match ptr_as_slice::<f64>(input.stimes_ptr, n_times) {
            Some(s) => s,
            None => return ErrorCode::NullPointer.code(),
        };
        let sfreqs_slice = match ptr_as_slice::<f64>(input.sfreqs_ptr, n_freqs) {
            Some(s) => s,
            None => return ErrorCode::NullPointer.code(),
        };

        let baseline_vec: Option<Vec<f64>> = if input.baseline_ptr.is_null() {
            None
        } else {
            let baseline_slice = slice::from_raw_parts(input.baseline_ptr, n_freqs);
            Some(baseline_slice.to_vec())
        };

        // Build ndarray views.
        let spect_view = match ArrayView2::from_shape((n_freqs, n_times), spect_slice) {
            Ok(v) => v,
            Err(_) => return ErrorCode::ShapeMismatch.code(),
        };
        let stimes_view = ArrayView1::from(stimes_slice);
        let sfreqs_view = ArrayView1::from(sfreqs_slice);
        let baseline_view = baseline_vec
            .as_ref()
            .map(|v| ArrayView1::from(v.as_slice()));

        // Replace sentinel values (0 or non-positive) with their defaults.
        let seg_time = if input.seg_time > 0.0 { input.seg_time } else { DEFAULT_SEG_TIME_S };
        let downsample_f = if input.downsample_f == 0 { 1 } else { input.downsample_f as usize };
        let downsample_t = if input.downsample_t == 0 { 1 } else { input.downsample_t as usize };
        let max_merges = if input.max_merges == 0.0 { f64::INFINITY } else { input.max_merges };
        let freq_max = if input.freq_max == 0.0 { f64::INFINITY } else { input.freq_max };

        let params = crate::extract_pipeline::ExtractParams {
            seg_time,
            downsample_f,
            downsample_t,
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
        // mutex. Rayon workers may call it concurrently. The mutex
        // ensures only one call reaches the C side at a time (MATLAB's
        // mexPrintf is not thread-safe).
        let progress_boxed: Option<Box<dyn Fn(u32, u32) + Sync + Send>> =
            input.progress_cb.map(|cb| {
                let callback_lock = std::sync::Mutex::new(());
                Box::new(move |segments_done, segments_total| {
                    let _guard = callback_lock.lock().unwrap();
                    cb(segments_done, segments_total);
                }) as Box<dyn Fn(u32, u32) + Sync + Send>
            });
        let progress_ref: Option<&(dyn Fn(u32, u32) + Sync)> =
            progress_boxed.as_deref().map(|boxed| boxed as &(dyn Fn(u32, u32) + Sync));

        let (peaks, labels) = match crate::extract_pipeline::extract_tfpeaks(
            spect_view, stimes_view, sfreqs_view, baseline_view, &params, progress_ref, None,
        ) {
            Ok(p) => p,
            Err(_) => return ErrorCode::KernelError.code(),
        };

        // Fill out struct.
        let n_peaks = peaks.len();
        output.n_peaks = n_peaks;
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

        // Area and Peakiness: one scalar per peak.
        output.area = leak_vec_f64(peaks.area);
        output.peakiness = leak_vec_f64(peaks.peakiness);

        // HeightData: flatten Vec<Vec<f64>> into one buffer + CSR offsets.
        let mut height_offsets: Vec<u64> = Vec::with_capacity(n_peaks + 1);
        height_offsets.push(0);
        let mut height_total: usize = 0;
        for peak_values in &peaks.height_data {
            height_total += peak_values.len();
        }
        let mut height_flat: Vec<f64> = Vec::with_capacity(height_total);
        for peak_values in peaks.height_data.into_iter() {
            height_flat.extend_from_slice(&peak_values);
            height_offsets.push(height_flat.len() as u64);
        }
        output.n_height_data_elems = height_flat.len();
        output.height_data = leak_vec_f64(height_flat);
        output.height_data_offsets = leak_vec_u64(height_offsets);

        // Boundaries: each Vec<f64> already interleaves (t, f) pairs. Offset
        // unit is one *pair*, i.e., divide flat-len by 2.
        let mut boundary_offsets: Vec<u64> = Vec::with_capacity(n_peaks + 1);
        boundary_offsets.push(0);
        let mut boundary_total: usize = 0;
        for peak_boundary in &peaks.boundaries_xy {
            boundary_total += peak_boundary.len();
        }
        let mut boundary_flat: Vec<f64> = Vec::with_capacity(boundary_total);
        for peak_boundary in peaks.boundaries_xy.into_iter() {
            boundary_flat.extend_from_slice(&peak_boundary);
            boundary_offsets.push((boundary_flat.len() / 2) as u64);
        }
        output.n_boundary_pixels = boundary_flat.len() / 2;
        output.boundaries_xy = leak_vec_f64(boundary_flat);
        output.boundary_offsets = leak_vec_u64(boundary_offsets);

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
// dynamo_refine_peaks
// -------------------------------------------------------------------------

/// Run Hann-event spectra + spline refinement for a batch of TF peaks.
///
/// Inputs:
///   * `peak_time`, `peak_freq`, `bbox` (n × 4, [f_lo, f_hi, t_lo, t_hi]), `n`.
///   * `data`, `n_data`, `fs`: raw 1-D signal and sampling rate.
///   * `freq_lo`, `freq_hi`, `window_size` (s), `dsfreqs` (Hz).
///
/// Caller provides preallocated output buffers:
///   * `out_peak_freq[i]`: refined frequency. Peaks within `window_size/2`
///     of either end of `data` keep their input `peak_freq[i]`. NaN when
///     the refined maximum sits on the edge of the peak's frequency band.
///   * `out_keep[i]`: 1 for a peak near either end of `data` (it keeps its
///     input frequency). For every other peak, 1 when the refined frequency
///     is finite and 0 when it is NaN.
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
        let peak_times = match ptr_as_slice::<f64>(peak_time, n) {
            Some(s) => s,
            None => return ErrorCode::NullPointer.code(),
        };
        let peak_freqs_in = match ptr_as_slice::<f64>(peak_freq, n) {
            Some(s) => s,
            None => return ErrorCode::NullPointer.code(),
        };
        let bbox_flat = match ptr_as_slice::<f64>(bbox, n * 4) {
            Some(s) => s,
            None => return ErrorCode::NullPointer.code(),
        };
        let data_slice = match ptr_as_slice::<f64>(data, n_data) {
            Some(s) => s,
            None => return ErrorCode::NullPointer.code(),
        };
        let out_freq = match NonNull::new(out_peak_freq) {
            Some(p) => slice::from_raw_parts_mut(p.as_ptr(), n),
            None => return ErrorCode::NullPointer.code(),
        };
        let out_keep_slice = match NonNull::new(out_keep) {
            Some(p) => slice::from_raw_parts_mut(p.as_ptr(), n),
            None => return ErrorCode::NullPointer.code(),
        };

        // Unpack bbox (n, 4) row-major → bbox_lo, bbox_hi (freq band per peak).
        let mut bbox_lo = Vec::with_capacity(n);
        let mut bbox_hi = Vec::with_capacity(n);
        for i in 0..n {
            let f_lo = bbox_flat[i * 4];
            let f_hi = bbox_flat[i * 4 + 1];
            bbox_lo.push(f_lo);
            bbox_hi.push(f_hi);
        }

        // The caller aligns the peak times to the start of `data`, so the
        // time of the first sample is 0.
        let data_start_time = 0.0;
        let (spect, sfreqs) = crate::refine::hann_event_spectra(
            data_slice,
            fs,
            peak_times,
            data_start_time,
            (freq_lo, freq_hi),
            window_size,
            dsfreqs,
            crate::refine::DetrendOpt::Constant,
        );
        // Number of points on the dense frequency grid searched per peak.
        const REFINE_GRID_POINTS: usize = 1000;
        // Refined maxima on the edge of the peak's frequency band become NaN.
        let remove_edge_peaks = true;
        let refined = crate::refine::refine_from_spectra(
            &spect,
            &sfreqs,
            &bbox_lo,
            &bbox_hi,
            REFINE_GRID_POINTS,
            remove_edge_peaks,
        );
        // Edge-peak exclusion, as in MATLAB refinePeakFrequency.m:141.
        // MATLAB skips peaks whose center is within window_size/2
        // of data[0] or data[end] (a partial window would need zero-padding
        // and produce a distorted Hann-PSD and a biased refined frequency).
        // Those peaks are kept at their original (pass-2 bbox-centroid)
        // PeakFrequency. The same happens here, in place of returning NaN or
        // the zero-padded Hann refine output.
        let data_t_end = if n_data > 0 { (n_data - 1) as f64 / fs } else { 0.0 };
        let edge_margin = window_size * 0.5;
        let t_lo = data_start_time + edge_margin;
        let t_hi = data_t_end - edge_margin;
        for i in 0..n {
            let is_edge = !(peak_times[i] > t_lo && peak_times[i] < t_hi);
            if is_edge {
                // Keep the original, unrefined frequency and retain the peak.
                out_freq[i] = peak_freqs_in[i];
                out_keep_slice[i] = 1;
            } else {
                let refined_freq = refined[i];
                out_freq[i] = refined_freq;
                out_keep_slice[i] = if refined_freq.is_finite() { 1 } else { 0 };
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
// dynamo_tfpeak_histogram
// -------------------------------------------------------------------------

/// Number of sleep-stage columns in `time_in_bin` / `prop_in_bin` (stage
/// labels 1..=5).
const N_STAGE_COLS: usize = 5;

/// FFI mirror of `histogram::tfpeak_histogram`.
///
/// Inputs are raw pointers with explicit lengths. Edge arrays are `(2, nbins)`
/// row-major: `[0 * nbins + i]` is the low edge of bin i, `[1 * nbins + i]`
/// is the high edge.
///
/// Caller-provided output buffers (must all be non-null). The 2-D outputs
/// are written column-major (MATLAB layout):
///   * `out_c_mat`: `(num_cbins, num_fbins)`, element (c, f) at
///     `f * num_cbins + c`.
///   * `out_time_in_bin`: `(num_cbins, 5)` column-major (minutes).
///   * `out_prop_in_bin`: `(num_cbins, 5)` column-major.
///   * `out_peak_at_freq`: `num_fbins`.
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
        let c_metric_slice = match ptr_as_slice::<f64>(c_metric, n_times) {
            Some(s) => s,
            None => return ErrorCode::NullPointer.code(),
        };
        let c_stages_slice = match ptr_as_slice::<f64>(c_stages, n_times) {
            Some(s) => s,
            None => return ErrorCode::NullPointer.code(),
        };
        let c_valid_raw = match ptr_as_slice::<u8>(c_valid, n_times) {
            Some(s) => s,
            None => return ErrorCode::NullPointer.code(),
        };
        let c_valid_allstages_raw = match ptr_as_slice::<u8>(c_valid_allstages, n_times) {
            Some(s) => s,
            None => return ErrorCode::NullPointer.code(),
        };
        let peak_freqs_slice = match ptr_as_slice::<f64>(peak_freqs, n_peaks) {
            Some(s) => s,
            None => return ErrorCode::NullPointer.code(),
        };
        let peak_c_slice = match ptr_as_slice::<f64>(peak_c, n_peaks) {
            Some(s) => s,
            None => return ErrorCode::NullPointer.code(),
        };
        let freq_edges_slice = match ptr_as_slice::<f64>(freq_edges, 2 * num_fbins) {
            Some(s) => s,
            None => return ErrorCode::NullPointer.code(),
        };
        let c_edges_slice = match ptr_as_slice::<f64>(c_edges, 2 * num_cbins) {
            Some(s) => s,
            None => return ErrorCode::NullPointer.code(),
        };

        let c_valid_flags: Vec<bool> = c_valid_raw.iter().map(|&b| b != 0).collect();
        let c_valid_allstages_flags: Vec<bool> = c_valid_allstages_raw.iter().map(|&b| b != 0).collect();

        let c_metric_view = match ArrayView1::from_shape(n_times, c_metric_slice) {
            Ok(v) => v,
            Err(_) => return ErrorCode::ShapeMismatch.code(),
        };
        let c_stages_view = match ArrayView1::from_shape(n_times, c_stages_slice) {
            Ok(v) => v,
            Err(_) => return ErrorCode::ShapeMismatch.code(),
        };
        let peak_freqs_view = match ArrayView1::from_shape(n_peaks, peak_freqs_slice) {
            Ok(v) => v,
            Err(_) => return ErrorCode::ShapeMismatch.code(),
        };
        let peak_c_view = match ArrayView1::from_shape(n_peaks, peak_c_slice) {
            Ok(v) => v,
            Err(_) => return ErrorCode::ShapeMismatch.code(),
        };
        let freq_edges_view = match ArrayView2::from_shape((2, num_fbins), freq_edges_slice) {
            Ok(v) => v,
            Err(_) => return ErrorCode::ShapeMismatch.code(),
        };
        let c_edges_view = match ArrayView2::from_shape((2, num_cbins), c_edges_slice) {
            Ok(v) => v,
            Err(_) => return ErrorCode::ShapeMismatch.code(),
        };

        let inp = crate::histogram::HistogramInputs {
            c_metric: c_metric_view,
            c_stages: c_stages_view,
            c_dt,
            c_valid: &c_valid_flags,
            c_valid_allstages: &c_valid_allstages_flags,
            peak_freqs: peak_freqs_view,
            peak_c: peak_c_view,
            freq_edges: freq_edges_view,
            c_edges: c_edges_view,
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
        let c_mat_out = match NonNull::new(out_c_mat) {
            Some(p) => slice::from_raw_parts_mut(p.as_ptr(), cmat_len),
            None => return ErrorCode::NullPointer.code(),
        };
        let time_in_bin_out = match NonNull::new(out_time_in_bin) {
            Some(p) => slice::from_raw_parts_mut(p.as_ptr(), num_cbins * N_STAGE_COLS),
            None => return ErrorCode::NullPointer.code(),
        };
        let prop_in_bin_out = match NonNull::new(out_prop_in_bin) {
            Some(p) => slice::from_raw_parts_mut(p.as_ptr(), num_cbins * N_STAGE_COLS),
            None => return ErrorCode::NullPointer.code(),
        };
        let peak_at_freq_out = match NonNull::new(out_peak_at_freq) {
            Some(p) => slice::from_raw_parts_mut(p.as_ptr(), num_fbins),
            None => return ErrorCode::NullPointer.code(),
        };

        // The MEX caller allocates every 2-D output column-major
        // (mxCreateDoubleMatrix), while ndarray indexes row-major. Write each
        // element at its column-major offset: element (c, f) of a
        // (num_cbins, num_fbins) matrix goes to index f * num_cbins + c.
        for c_bin in 0..num_cbins {
            for f_bin in 0..num_fbins {
                c_mat_out[f_bin * num_cbins + c_bin] = out.c_mat[[c_bin, f_bin]];
            }
        }
        // time_in_bin, prop_in_bin: shape (num_cbins, 5), also written
        // column-major.
        for c_bin in 0..num_cbins {
            for stage in 0..N_STAGE_COLS {
                time_in_bin_out[stage * num_cbins + c_bin] = out.time_in_bin[[c_bin, stage]];
                prop_in_bin_out[stage * num_cbins + c_bin] = out.prop_in_bin[[c_bin, stage]];
            }
        }
        // peak_at_freq: 1-D, no layout ambiguity.
        for (i, &v) in out.peak_at_freq.iter().enumerate() {
            peak_at_freq_out[i] = v;
        }
        ErrorCode::Ok.code()
    }));
    match result {
        Ok(c) => c,
        Err(_) => ErrorCode::Panic.code(),
    }
}

// -------------------------------------------------------------------------
// dynamo_mask_spectrogram
// -------------------------------------------------------------------------

/// Perimeter-aware mask of pass-2 spectrogram using pass-1 labels.
/// Thin FFI wrapper over [`crate::mask::mask_spectrogram`].
///
/// All 2-D arrays are row-major: `spect_2s[f * n_times_2 + t]`, etc.
/// `labels_1s` is (n_freqs, n_times_1) int64. `spect_2s` and `out_masked`
/// are both (n_freqs, n_times_2) float64.
///
/// Caller allocates `out_masked` of size `n_freqs * n_times_2`.
///
/// Returns 0 on success, negative on error.
///
/// # Safety
/// All pointers must be valid for the lengths derived from `n_freqs`,
/// `n_times_1`, `n_times_2`. `out_masked` must be writable.
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
        let spect_slice = match ptr_as_slice::<f64>(spect_2s, n_freqs * n_times_2) {
            Some(s) => s,
            None => return ErrorCode::NullPointer.code(),
        };
        let stimes_2s_slice = match ptr_as_slice::<f64>(stimes_2s, n_times_2) {
            Some(s) => s,
            None => return ErrorCode::NullPointer.code(),
        };
        let labels_slice = match ptr_as_slice::<i64>(labels_1s, n_freqs * n_times_1) {
            Some(s) => s,
            None => return ErrorCode::NullPointer.code(),
        };
        let stimes_1s_slice = match ptr_as_slice::<f64>(stimes_1s, n_times_1) {
            Some(s) => s,
            None => return ErrorCode::NullPointer.code(),
        };
        let out = match NonNull::new(out_masked) {
            Some(p) => std::slice::from_raw_parts_mut(p.as_ptr(), n_freqs * n_times_2),
            None => return ErrorCode::NullPointer.code(),
        };
        let spect_view = match ArrayView2::from_shape((n_freqs, n_times_2), spect_slice) {
            Ok(v) => v,
            Err(_) => return ErrorCode::ShapeMismatch.code(),
        };
        let labels_view = match ArrayView2::from_shape((n_freqs, n_times_1), labels_slice) {
            Ok(v) => v,
            Err(_) => return ErrorCode::ShapeMismatch.code(),
        };
        let stimes_2s_view = ArrayView1::from(stimes_2s_slice);
        let stimes_1s_view = ArrayView1::from(stimes_1s_slice);

        let masked = match crate::mask::mask_spectrogram(
            spect_view, stimes_2s_view, labels_view, stimes_1s_view,
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
// dynamo_multitaper_spectrogram: in/out structs
// -------------------------------------------------------------------------

/// Input descriptor for [`dynamo_multitaper_spectrogram`].
///
/// `data_ptr` is a length-`n_data` f64 vector. `tapers_ptr` is the
/// pre-computed DPSS taper bank (caller supplies it, for example from
/// [`dynamo_dpss`]), row-major (n_tapers × winsize) so element (k, i) is
/// at `tapers_ptr[k * winsize + i]`. Caller MUST guarantee
/// `winsize == round(window_size_s * fs)`. The underlying compute
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
/// All three pointers are callee-allocated and must be released with
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
/// shape of the other dynamo_rs ABI entries. f64 throughout. The MATLAB
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

        // Eigen weights are optional. They are only consumed when weighting=Eigen.
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

        // The spectrogram is (n_freqs_out, n_windows). Force row-major
        // layout first so the flat Vec handed to the caller is in the
        // documented order.
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
// dynamo_dpss: DPSS (Slepian) taper generation
// -------------------------------------------------------------------------

/// Generate K DPSS tapers of length N with time-half-bandwidth NW.
///
/// Caller-allocated outputs:
///   `tapers_out`: row-major (K, N) double, total K*N elements.
///                   Row k is the k-th Slepian sequence, unit L²-norm.
///   `ratios_out`: length-K double, concentration ratios in (0, 1].
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

        // Tapers come back as (K, N). Ensure standard (row-major) layout
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
// dynamo_detect_artifacts: band-detection iter-zscore artifact mask
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
/// branch**. That needs the multitaper spectrogram and is left to the
/// caller (the MATLAB `detect_artifacts.m` ORs the slope mask in
/// separately). The Rust path here matches `detect_artifacts.m` with
/// `'slope_test', false` to within ≤30 samples on the synthetic-EEG
/// fixture battery.
///
/// Caller-allocated output `mask_out` is length `n`; 1 = artifact.
///
/// # Safety
/// `data` must point to `n` readable f64s, `opts` must point to a valid
/// `ArtifactOptsFFI`, and `mask_out` must point to `n` writable u8s.
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
// dynamo_compute_baseline: per-frequency percentile baseline
// -------------------------------------------------------------------------

/// Compute the per-frequency baseline used in TFpeak extraction.
///
/// Inputs (all row-major):
///   `spect`: shape (n_freqs, n_times) double. The baseline is the
///                       `baseline_ptile`-th percentile over the valid
///                       columns of each row (zeros treated as NaN).
///   `stimes`: length n_times, window-center times.
///   `t_data`: length n_data, the EEG time grid that the
///                       `baseline_exclude` mask is defined on.
///   `baseline_exclude`: length n_data u8 (1 = exclude), nearest-neighbor
///                       interpolated onto `stimes` columns.
///   `baseline_range_lo` / `baseline_range_hi`: time-range trimming
///                       applied to `stimes`.
///   `baseline_ptile`: percentile in [0, 100] (Hyndman-Fan #5, matching
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

        // Shape is (F, 1). Flatten and copy.
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
// dynamo_so_power: SO-power time-series from a pre-computed spectrogram
// -------------------------------------------------------------------------

/// Input descriptor for [`dynamo_so_power`]. `spect_ptr` is a row-major
/// (n_freqs, n_times) SO-band spectrogram.
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
    /// ASCII norm method spec, e.g. "p2shift1234", "percent", "none".
    /// Exactly `norm_method_len` bytes are read, so no NUL terminator is
    /// needed and none should be counted in the length. Parsed by
    /// `NormMethod::parse`.
    pub norm_method_ptr:    *const u8,
    pub norm_method_len:    usize,
}

/// Output descriptor for [`dynamo_so_power`]. The three arrays are
/// callee-allocated with length `n_out`.
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
/// (and pydynamo `compute_so_power`). The MATLAB analog is `computeSOpower.m`.
///
/// Output pointers are callee-allocated. Caller frees with
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
            Some(s) => s,
            None => return ErrorCode::NullPointer.code(),
        };
        let sfreqs = match ptr_as_slice::<f64>(input.sfreqs_ptr, input.n_freqs) {
            Some(s) => s,
            None => return ErrorCode::NullPointer.code(),
        };
        let eeg_times = match ptr_as_slice::<f64>(input.eeg_times_ptr, input.n_data) {
            Some(s) => s,
            None => return ErrorCode::NullPointer.code(),
        };
        let excl_u8 = match ptr_as_slice::<u8>(input.isexcluded_ptr, input.n_data) {
            Some(s) => s,
            None => return ErrorCode::NullPointer.code(),
        };
        let isexcluded: Vec<bool> = excl_u8.iter().map(|&b| b != 0).collect();
        let stage_times = match ptr_as_slice::<f64>(input.stage_times_ptr, input.n_stages) {
            Some(s) => s,
            None => return ErrorCode::NullPointer.code(),
        };
        let stage_vals = match ptr_as_slice::<f64>(input.stage_vals_ptr, input.n_stages) {
            Some(s) => s,
            None => return ErrorCode::NullPointer.code(),
        };

        let nm_bytes = match ptr_as_slice::<u8>(input.norm_method_ptr, input.norm_method_len) {
            Some(s) => s,
            None => return ErrorCode::NullPointer.code(),
        };
        let nm_str = match std::str::from_utf8(nm_bytes) {
            Ok(s) => s,
            Err(_) => return ErrorCode::InvalidArgument.code(),
        };
        let nm = match crate::so_power::NormMethod::parse(nm_str) {
            Some(n) => n,
            None => return ErrorCode::InvalidArgument.code(),
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
// dynamo_so_phase: SO-phase time-series from raw EEG
// -------------------------------------------------------------------------

/// Input descriptor for [`dynamo_so_phase`].
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

/// Output descriptor for [`dynamo_so_phase`]. All four arrays are
/// callee-allocated with length `n_out`.
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
/// `compute_so_phase`. The MATLAB analog is `computeSOphase.m`.
///
/// All four output arrays are length `n_data` and callee-allocated.
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
            Some(s) => s,
            None => return ErrorCode::NullPointer.code(),
        };
        let eeg_times = match ptr_as_slice::<f64>(input.eeg_times_ptr, input.n_data) {
            Some(s) => s,
            None => return ErrorCode::NullPointer.code(),
        };
        let excl_u8 = match ptr_as_slice::<u8>(input.isexcluded_ptr, input.n_data) {
            Some(s) => s,
            None => return ErrorCode::NullPointer.code(),
        };
        let isexcluded: Vec<bool> = excl_u8.iter().map(|&b| b != 0).collect();
        // Number of coefficients per second-order section: b0 b1 b2 a0 a1 a2.
        const SOS_COLS: usize = 6;
        let sos_flat = match ptr_as_slice::<f64>(input.sos_ptr, input.n_sections * SOS_COLS) {
            Some(s) => s,
            None => return ErrorCode::NullPointer.code(),
        };
        // Regroup the flat coefficient buffer into one row per section.
        let mut sos: Vec<[f64; SOS_COLS]> = Vec::with_capacity(input.n_sections);
        for section in sos_flat.chunks_exact(SOS_COLS) {
            let mut coeffs = [0.0_f64; SOS_COLS];
            coeffs.copy_from_slice(section);
            sos.push(coeffs);
        }
        let stage_times = match ptr_as_slice::<f64>(input.stage_times_ptr, input.n_stages) {
            Some(s) => s,
            None => return ErrorCode::NullPointer.code(),
        };
        let stage_vals = match ptr_as_slice::<f64>(input.stage_vals_ptr, input.n_stages) {
            Some(s) => s,
            None => return ErrorCode::NullPointer.code(),
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
// buffer-free helpers
// -------------------------------------------------------------------------

/// Free an f64 buffer returned through any callee-allocated output
/// pointer in this module (`ExtractTfpeaksOut`, `MtsOut`, `SoPowerOut`,
/// `SoPhaseOut`, `SplineBasisOut`, `ParamFitOutFFI`).
///
/// `len` MUST equal the element count of the original allocation, as
/// documented on the struct field that returned the pointer.
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

/// Output descriptor for [`dynamo_spline_basis_fit`].
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
        m_y:             0,
        m_x:             0,
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
/// All four output arrays are callee-allocated. Caller frees with
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
            Some(s) => s,
            None => return ErrorCode::NullPointer.code(),
        };
        let feat_bins = match ptr_as_slice::<f64>(input.feat_bins_ptr, input.n_x) {
            Some(s) => s,
            None => return ErrorCode::NullPointer.code(),
        };
        let freq_bins = match ptr_as_slice::<f64>(input.freq_bins_ptr, input.n_y) {
            Some(s) => s,
            None => return ErrorCode::NullPointer.code(),
        };
        let internal_knots_x = match ptr_as_slice::<f64>(input.internal_knots_x_ptr, input.n_internal_knots_x) {
            Some(s) => s,
            None => return ErrorCode::NullPointer.code(),
        };
        let internal_knots_y = match ptr_as_slice::<f64>(input.internal_knots_y_ptr, input.n_internal_knots_y) {
            Some(s) => s,
            None => return ErrorCode::NullPointer.code(),
        };

        let soph = match ArrayView2::from_shape((input.n_x, input.n_y), soph_flat) {
            Ok(a) => a,
            Err(_) => return ErrorCode::ShapeMismatch.code(),
        };

        let kernel_out = match crate::spline_basis::fit_tensor_product_spline(
            soph,
            feat_bins,
            freq_bins,
            internal_knots_x,
            internal_knots_y,
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
/// `(n_freqs, n_features)` column-major byte-for-byte, so `mxGetPr` can be
/// passed directly without copying.
///
/// Initial/lower/upper are `(n_modes, 6)` row-major:
///   * power: `[amp, fmean, fstd, pmean,    pstd,      theta]`
///   * phase: `[amp, fmean, fstd, phasepref, recikappa, theta]`
/// `fstd` is a frequency standard deviation in Hz for both fit types.
///
/// `bg_initial`, `bg_lower`, `bg_upper` are 3-vectors `[xxx, yyy, zzz]`.
///
/// # Width convention
///
/// The Gaussian widths are standard deviations. Both kernels carry an
/// explicit factor of one half in the exponent:
///
/// ```text
///   power: amp * exp(-0.5*(u/fstd)^2 - 0.5*(v/pstd)^2)
///   phase: amp * exp(-0.5*(dy/fstd)^2) * exp(k*(cos(dx) - 1)), k = 1/recikappa^2
/// ```
///
/// Widths written by kernel versions below 0.2.0 are `sqrt(2)` times
/// larger than these standard deviations, although the struct layout, the
/// field types and the function signatures are identical. Values cross
/// this ABI unchanged, so divide stored widths from those versions by
/// `sqrt(2)` before reusing them as a modeled surface or bound window, and
/// do not pool widths across the two conventions. Seeds and priors already
/// expressed as standard deviations keep their numeric values.
///
/// Column 4 must NOT be converted by index. A stored `pstd` from a version
/// below 0.2.0 for `dynamo_rotgauss_fit` rescales by `1/sqrt(2)`, while
/// `recikappa` for `dynamo_vmgauss_fit` needs no conversion, because the
/// von Mises factor is its own small-angle Gaussian and already carries
/// the half.
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

/// Output descriptor for [`dynamo_rotgauss_fit`] and [`dynamo_vmgauss_fit`].
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
        n_y:            0,
        n_x:            0,
        gof_sse:        0.0,
        gof_rsquare:    0.0,
        gof_adjrsquare: 0.0,
        gof_rmse:       0.0,
        gof_dfe:        0.0,
        gof_dfm:        0.0,
        iters_used:     0,
    }
}

/// Number of parameters per mode in the paramfit matrices.
const PARAMFIT_COLS: usize = 6;

/// Borrowed views over the caller's `ParamFitIn` buffers, shared by the
/// two paramfit entry points.
struct ParamFitViews {
    soph: ArrayView2<'static, f64>,
    feat_bins: &'static [f64],
    freq_bins: &'static [f64],
    initial: ArrayView2<'static, f64>,
    lower: ArrayView2<'static, f64>,
    upper: ArrayView2<'static, f64>,
}

/// Validate the sizes in `input` and wrap its raw pointers as views.
///
/// SAFETY: every pointer in `input` must back its declared length for as
/// long as the returned views are used.
unsafe fn paramfit_common_setup(input: &ParamFitIn) -> Result<ParamFitViews, c_int> {
    if input.n_y == 0 || input.n_x == 0 || input.n_modes == 0 {
        return Err(ErrorCode::InvalidArgument.code());
    }
    let soph_flat = match ptr_as_slice::<f64>(input.soph_ptr, input.n_y * input.n_x) {
        Some(s) => s,
        None => return Err(ErrorCode::NullPointer.code()),
    };
    let feat_bins = match ptr_as_slice::<f64>(input.feat_bins_ptr, input.n_x) {
        Some(s) => s,
        None => return Err(ErrorCode::NullPointer.code()),
    };
    let freq_bins = match ptr_as_slice::<f64>(input.freq_bins_ptr, input.n_y) {
        Some(s) => s,
        None => return Err(ErrorCode::NullPointer.code()),
    };
    let n_param_values = input.n_modes * PARAMFIT_COLS;
    let initial_flat = match ptr_as_slice::<f64>(input.initial_ptr, n_param_values) {
        Some(s) => s,
        None => return Err(ErrorCode::NullPointer.code()),
    };
    let lower_flat = match ptr_as_slice::<f64>(input.lower_ptr, n_param_values) {
        Some(s) => s,
        None => return Err(ErrorCode::NullPointer.code()),
    };
    let upper_flat = match ptr_as_slice::<f64>(input.upper_ptr, n_param_values) {
        Some(s) => s,
        None => return Err(ErrorCode::NullPointer.code()),
    };
    let param_shape = (input.n_modes, PARAMFIT_COLS);
    let soph = ArrayView2::from_shape((input.n_y, input.n_x), soph_flat)
        .map_err(|_| ErrorCode::ShapeMismatch.code())?;
    let initial = ArrayView2::from_shape(param_shape, initial_flat)
        .map_err(|_| ErrorCode::ShapeMismatch.code())?;
    let lower = ArrayView2::from_shape(param_shape, lower_flat)
        .map_err(|_| ErrorCode::ShapeMismatch.code())?;
    let upper = ArrayView2::from_shape(param_shape, upper_flat)
        .map_err(|_| ErrorCode::ShapeMismatch.code())?;
    Ok(ParamFitViews { soph, feat_bins, freq_bins, initial, lower, upper })
}

/// Copy the kernel's fit result into the C output struct. The parameter
/// and model arrays are handed out as callee-allocated buffers.
fn fill_paramfit_out(out: &mut ParamFitOutFFI, kernel: crate::paramfit::ParamFitOut) {
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
}

/// Fit a rotated-Gaussian-mixture model + linear background plane to a SOPH
/// histogram. Mirrors MATLAB `fit_rotGauss` (without `prepareSurfaceData`'s
/// NaN drop. The caller must pass a finite-valued grid).
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

        let views = match paramfit_common_setup(input) {
            Ok(views) => views,
            Err(code) => return code,
        };

        let kernel_out = match crate::paramfit::rot_gauss::fit_rotgauss(
            views.soph, views.feat_bins, views.freq_bins,
            views.initial, views.lower, views.upper,
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
/// histogram. `ParamFitIn::unit_row` controls per-row normalization;
/// nonzero matches MATLAB `fit_vmGauss`.
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

        let views = match paramfit_common_setup(input) {
            Ok(views) => views,
            Err(code) => return code,
        };

        let kernel_out = match crate::paramfit::vm_gauss::fit_vmgauss(
            views.soph, views.feat_bins, views.freq_bins,
            views.initial, views.lower, views.upper,
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

#[cfg(test)]
mod tests {
    use super::*;

    /// `dynamo_version()` must return a NUL-terminated static matching
    /// `build_info::VERSION` exactly, and the ABI accessor must agree
    /// with the compile-time constant.
    #[test]
    fn version_symbols_are_consistent() {
        assert_eq!(dynamo_c_abi_version(), DYNAMO_C_ABI_VERSION);
        let ptr = dynamo_version();
        assert!(!ptr.is_null());
        let s = unsafe { std::ffi::CStr::from_ptr(ptr) }.to_str().unwrap();
        assert_eq!(s, crate::build_info::VERSION);
    }

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
