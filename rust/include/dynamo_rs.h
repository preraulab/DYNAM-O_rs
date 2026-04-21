/* Auto-generated C ABI header for dynamo_rs. Do not edit. */
/* Generated from src/c_api.rs by build.rs via cbindgen. */

#ifndef DYNAMO_RS_H
#define DYNAMO_RS_H

#include <stddef.h>
#include <stdint.h>

/**
 * Input descriptor for [`dynamo_extract_tfpeaks`].
 *
 * All array pointers point to column-major-equivalent data: the spectrogram
 * is (F, T) with F varying fastest in memory (row-major C layout matching
 * `ndarray::Array2::from_shape_vec((F, T), vec)` where vec was built
 * row-by-row). `stimes.len() == T`, `sfreqs.len() == F`.
 *
 * If `baseline_ptr` is null the spectrogram is used as-is. Otherwise the
 * spectrogram is divided column-wise by `baseline` (length F).
 */
typedef struct ExtractTfpeaksIn {
  const double *spect_ptr;
  uintptr_t n_freqs;
  uintptr_t n_times;
  const double *stimes_ptr;
  const double *sfreqs_ptr;
  const double *baseline_ptr;
  double merge_thresh;
  double max_merges;
  double trim_vol_thresh;
  double trim_shift_val;
  double segment_num;
} ExtractTfpeaksIn;

/**
 * Output descriptor for [`dynamo_extract_tfpeaks`].
 *
 * All `*mut` pointers are callee-allocated; caller must free each with the
 * matching `dynamo_free_buffer_*` call. `bounding_box` is `n_peaks × 4`
 * row-major ([f_lo, f_hi, t_lo, t_hi] per peak). `labels` is `F × T`
 * row-major (same layout as `spect_ptr`).
 */
typedef struct ExtractTfpeaksOut {
  uintptr_t n_peaks;
  double *peak_time;
  double *peak_freq;
  double *duration;
  double *bandwidth;
  double *height;
  double *volume;
  double *segment_num;
  double *bounding_box;
  int64_t *labels;
  uintptr_t n_label_elems;
} ExtractTfpeaksOut;

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/**
 * Run the full tf-peak extraction pipeline on a single segment spectrogram.
 *
 * # Safety
 * `in_` must point to a valid `ExtractTfpeaksIn`. `out` must point to a
 * valid (uninitialized or zeroed) `ExtractTfpeaksOut`. Array pointers
 * inside `in_` must be non-null and live for the duration of the call.
 */
int dynamo_extract_tfpeaks(const struct ExtractTfpeaksIn *in_, struct ExtractTfpeaksOut *out);

/**
 * Run Hann-event spectra + spline refinement for a batch of TF peaks.
 *
 * Inputs:
 *   * `peak_time`, `peak_freq`, `bbox` (n × 4, [f_lo, f_hi, t_lo, t_hi]), `n`.
 *   * `data`, `n_data`, `fs` — raw 1-D signal + sampling rate.
 *   * `freq_lo`, `freq_hi`, `window_size` (s), `dsfreqs` (Hz).
 *
 * Caller provides preallocated output buffers:
 *   * `out_peak_freq[i]` — refined frequency (NaN if removed).
 *   * `out_keep[i]` — 1 if peak passed edge filter, 0 otherwise.
 *
 * # Safety
 * All pointers must be valid for at least `n` elements (`n_data` for `data`),
 * and output buffers must be writable for `n` elements.
 */
int dynamo_refine_peaks(const double *peak_time,
                        const double *peak_freq,
                        const double *bbox,
                        uintptr_t n,
                        const double *data,
                        uintptr_t n_data,
                        double fs,
                        double freq_lo,
                        double freq_hi,
                        double window_size,
                        double dsfreqs,
                        double *out_peak_freq,
                        uint8_t *out_keep);

/**
 * FFI mirror of `histogram::tfpeak_histogram`.
 *
 * Inputs are raw pointers with explicit lengths. Edge arrays are `(2, nbins)`
 * row-major: `[0 * nbins + i]` is the low edge of bin i, `[1 * nbins + i]`
 * is the high edge.
 *
 * Caller-provided output buffers (must all be non-null):
 *   * `out_c_mat`          — `num_cbins * num_fbins` row-major.
 *   * `out_time_in_bin`    — `num_cbins * 5` row-major (minutes).
 *   * `out_prop_in_bin`    — `num_cbins * 5` row-major.
 *   * `out_peak_at_freq`   — `num_fbins`.
 *
 * # Safety
 * All pointers must be valid for the lengths derived from `n_times`,
 * `n_peaks`, `num_fbins`, `num_cbins`.
 */
int dynamo_tfpeak_histogram(const double *c_metric,
                            const double *c_stages,
                            double c_dt,
                            const uint8_t *c_valid,
                            const uint8_t *c_valid_allstages,
                            uintptr_t n_times,
                            const double *peak_freqs,
                            const double *peak_c,
                            uintptr_t n_peaks,
                            const double *freq_edges,
                            uintptr_t num_fbins,
                            const double *c_edges,
                            uintptr_t num_cbins,
                            uint8_t circular,
                            double circular_lo,
                            double circular_hi,
                            int32_t norm_dim,
                            uint8_t compute_rate,
                            double min_time_in_bin,
                            int32_t min_peak_at_freq,
                            double *out_c_mat,
                            double *out_time_in_bin,
                            double *out_prop_in_bin,
                            double *out_peak_at_freq);

/**
 * Free a buffer previously returned via one of the callee-allocated output
 * pointers of `ExtractTfpeaksOut`.
 *
 * `len` MUST match the original allocation length (the matching `n_peaks`
 * or `n_label_elems` field for array-of-peaks or array-of-labels buffers).
 *
 * # Safety
 * See [`drop_leaked_f64`].
 */
void dynamo_free_buffer_f64(double *ptr, uintptr_t len);

/**
 * See [`dynamo_free_buffer_f64`].
 */
void dynamo_free_buffer_i64(int64_t *ptr, uintptr_t len);

/**
 * See [`dynamo_free_buffer_f64`].
 */
void dynamo_free_buffer_u8(uint8_t *ptr, uintptr_t len);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  /* DYNAMO_RS_H */
