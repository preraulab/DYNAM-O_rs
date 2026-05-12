/* Auto-generated C ABI header for dynamo_rs. Do not edit. */
/* Generated from src/c_api.rs by build.rs via cbindgen. */

#ifndef DYNAMO_RS_H
#define DYNAMO_RS_H

#include <stddef.h>
#include <stdint.h>

/**
 * Input descriptor for [`dynamo_extract_tfpeaks`].
 *
 * All array pointers point to row-major C layout: the spectrogram is
 * (F, T) stored as `spect[row * T + col]`. `stimes.len() == T`,
 * `sfreqs.len() == F`.
 *
 * If `baseline_ptr` is null the spectrogram is used as-is. Otherwise the
 * spectrogram is divided row-wise by `baseline` (length F) before
 * segmentation.
 *
 * The pipeline mirrors pydynamo's `extract_tfpeaks` end-to-end:
 *   1. Split the full (F, T) spect into `seg_time`-second segments
 *      (MATLAB `segmentData.m` semantics: floor + ceil).
 *   2. For each segment in parallel: stride-downsample by
 *      (`downsample_f`, `downsample_t`), watershed(-spect), merge,
 *      expand_labels(distance=5), resize labels up, trim, regionprops.
 *   3. Concatenate per-segment peaks.
 *   4. Apply filterStatsTable: keep peaks where
 *        dur_min < Duration < dur_max,
 *        bw_min  < Bandwidth < bw_max,
 *        freq_min < PeakFrequency < freq_max,
 *        pow2db(Height) > ht_db_min.
 *
 * Parameter defaults when 0.0 is passed:
 *   * `seg_time` → 30.0
 *   * `downsample_f` / `downsample_t` → 1
 *   * `max_merges` → +∞ if 0
 *   * `freq_max` → +∞ if 0
 *   * `dur_max`, `bw_max`, `ht_db_min` → no-op if you want everything
 *     through, pass +∞ / +∞ / -∞ respectively.
 */
typedef struct ExtractTfpeaksIn {
  const double *spect_ptr;
  uintptr_t n_freqs;
  uintptr_t n_times;
  const double *stimes_ptr;
  const double *sfreqs_ptr;
  const double *baseline_ptr;
  double seg_time;
  uint32_t downsample_f;
  uint32_t downsample_t;
  double merge_thresh;
  double max_merges;
  double trim_vol_thresh;
  double trim_shift_val;
  double dur_min;
  double dur_max;
  double bw_min;
  double bw_max;
  double freq_min;
  double freq_max;
  double ht_db_min;
  /**
   * Distance for expand_labels(). 0 = skip (MATLAB-native; labels
   * keep 0 on watershed lines). 5 = pydynamo default (skimage-style
   * fill so regions touch directly).
   */
  uint32_t expand_labels_distance;
  /**
   * Optional progress callback invoked once per completed segment.
   * Signature: `fn(segments_done: u32, segments_total: u32)`. Pass
   * `NULL` / `None` to skip. Called from rayon worker threads, but
   * dynamo_extract_tfpeaks serializes calls with an internal mutex
   * so the C callee may assume it is never invoked concurrently.
   */
  void (*progress_cb)(uint32_t, uint32_t);
} ExtractTfpeaksIn;

/**
 * Output descriptor for [`dynamo_extract_tfpeaks`].
 *
 * All `*mut` pointers are callee-allocated; caller must free each with the
 * matching `dynamo_free_buffer_*` call. `bounding_box` is `n_peaks × 4`
 * row-major `[t_tl, f_tl, width_s, height_Hz]` per peak (pydynamo format).
 *
 * `labels` is a row-major `(n_freqs, n_times)` i64 buffer of length
 * `n_label_elems = n_freqs * n_times`. Non-zero pixels carry a 1-based
 * peak index: a pixel with value `k` belongs to the k-th peak in the
 * returned arrays (so label `k` maps to row `k - 1` in `peak_time` etc).
 * Zero = background. Per-segment label images are stitched column-wise
 * with a running offset and then renumbered after the post-filter so
 * that surviving labels span `1..=n_peaks` densely.
 *
 * Variable-length per-peak data (`height_data`, `boundaries_xy`) is
 * flattened with a CSR-style offset array of length `n_peaks + 1`:
 *   * `height_data[height_data_offsets[i] .. height_data_offsets[i+1]]`
 *     are the spectrogram values inside peak i (one per pixel).
 *   * `boundaries_xy[2*boundary_offsets[i] .. 2*boundary_offsets[i+1]]`
 *     are interleaved (time, freq) pairs along peak i's perimeter.
 * Both offset arrays have length `n_peaks + 1` (always; even when n_peaks
 * is 0 the array is `[0]`).
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
  /**
   * Time-frequency area per peak (sec*Hz). Length n_peaks.
   */
  double *area;
  /**
   * Peakiness = 10*log10(Area * Height / Volume) per peak, in dB. Length n_peaks.
   */
  double *peakiness;
  /**
   * Flattened per-peak pixel values; length `n_height_data_elems`.
   */
  double *height_data;
  uintptr_t n_height_data_elems;
  /**
   * CSR offsets into `height_data`, length `n_peaks + 1`.
   */
  uint64_t *height_data_offsets;
  /**
   * Flattened per-peak boundary coords, interleaved (t, f). Length
   * `2 * n_boundary_pixels`.
   */
  double *boundaries_xy;
  uintptr_t n_boundary_pixels;
  /**
   * CSR offsets into `boundaries_xy` in PIXEL units (each pixel is two
   * f64s). Length `n_peaks + 1`.
   */
  uint64_t *boundary_offsets;
} ExtractTfpeaksOut;

/**
 * Input descriptor for [`dynamo_multitaper_spectrogram`].
 *
 * `data_ptr` is a length-`n_data` f64 vector. `tapers_ptr` is the
 * pre-computed DPSS taper bank (caller supplies; we don't reimplement
 * `dpss` in Rust), row-major (n_tapers × winsize) so element (k, i) is
 * at `tapers_ptr[k * winsize + i]`. Caller MUST guarantee
 * `winsize == round(window_size_s * fs)` — the underlying compute
 * validates and returns ShapeMismatch otherwise.
 *
 * `eigen_ptr` is required iff `weighting == 1` (Eigen). Pass null
 * (and `weighting = 0`) for unity weighting (the DYNAM-O default).
 */
typedef struct MtsIn {
  const double *data_ptr;
  uintptr_t n_data;
  double fs;
  const double *tapers_ptr;
  uintptr_t n_tapers;
  uintptr_t winsize;
  const double *eigen_ptr;
  uintptr_t eigen_len;
  double freq_min;
  double freq_max;
  double window_size_s;
  double window_step_s;
  uintptr_t nfft;
  uint32_t detrend;
  uint32_t weighting;
} MtsIn;

/**
 * Output descriptor for [`dynamo_multitaper_spectrogram`].
 *
 * `spect_ptr` points to a row-major (n_freqs_out × n_windows) f64 buffer.
 * All three pointers come from `Box::leak` and must be released with
 * `dynamo_free_buffer_f64(ptr, len)` where `len` is the matching
 * `n_freqs_out`, `n_windows`, or `n_freqs_out * n_windows` count.
 */
typedef struct MtsOut {
  double *spect_ptr;
  uintptr_t n_freqs_out;
  uintptr_t n_windows;
  double *stimes_ptr;
  double *sfreqs_ptr;
} MtsOut;

/**
 * Plain-C mirror of `crate::artifacts::ArtifactOpts`. Mirrors the field
 * order of the Rust struct so `cbindgen` lays it out identically.
 */
typedef struct ArtifactOptsFFI {
  double hf_pass;
  double hf_crit;
  double bb_pass;
  double bb_crit;
  uint8_t hf_detrend;
  uint8_t bb_detrend;
  double smooth_duration;
  double detrend_duration;
  double buffer_duration;
  uint32_t zscore_method;
} ArtifactOptsFFI;

typedef struct SoPowerIn {
  const double *spect_ptr;
  uintptr_t n_freqs;
  uintptr_t n_times;
  const double *stimes_ptr;
  const double *sfreqs_ptr;
  const double *eeg_times_ptr;
  uintptr_t n_data;
  const uint8_t *isexcluded_ptr;
  const double *stage_times_ptr;
  const double *stage_vals_ptr;
  uintptr_t n_stages;
  double time_range_lo;
  double time_range_hi;
  double outlier_threshold;
  uint8_t retain_fs;
  /**
   * Null-terminated ASCII norm method spec: e.g. "p2shift1234",
   * "percent", "none". Same parser as `NormMethod::parse`.
   */
  const uint8_t *norm_method_ptr;
  uintptr_t norm_method_len;
} SoPowerIn;

typedef struct SoPowerOut {
  double *so_power_norm_ptr;
  double *so_power_times_ptr;
  double *so_power_stages_ptr;
  /**
   * Output length (matches all three arrays above).
   */
  uintptr_t n_out;
  /**
   * ptile result: 0 = none, 1 = single (`ptile_value[0]`),
   * 2 = pair (`ptile_value[0]`, `ptile_value[1]`).
   */
  uint32_t ptile_kind;
  double ptile_value[2];
} SoPowerOut;

typedef struct SoPhaseIn {
  const double *eeg_ptr;
  const double *eeg_times_ptr;
  const uint8_t *isexcluded_ptr;
  uintptr_t n_data;
  /**
   * SOS filter coefficients, scipy layout: (n_sections, 6) flattened
   * row-major [b0 b1 b2 a0 a1 a2] per section.
   */
  const double *sos_ptr;
  uintptr_t n_sections;
  const double *stage_times_ptr;
  const double *stage_vals_ptr;
  uintptr_t n_stages;
} SoPhaseIn;

typedef struct SoPhaseOut {
  double *so_phase_ptr;
  double *so_phase_times_ptr;
  double *so_phase_stages_ptr;
  double *filtdata_ptr;
  uintptr_t n_out;
} SoPhaseOut;

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
 * Perimeter-aware mask of pass-2 spectrogram using pass-1 labels.
 * Thin FFI wrapper over [`crate::mask::mask_spectrogram`].
 *
 * All 2-D arrays are row-major: `spect_2s[f * n_times_2 + t]`, etc.
 * `labels_1s` is (n_freqs, n_times_1) int64; `spect_2s` and `out_masked`
 * are both (n_freqs, n_times_2) float64.
 *
 * Caller allocates `out_masked` of size `n_freqs * n_times_2`.
 *
 * Returns 0 on success, negative on error.
 */
int dynamo_mask_spectrogram(const double *spect_2s,
                            const double *stimes_2s,
                            const int64_t *labels_1s,
                            const double *stimes_1s,
                            uintptr_t n_freqs,
                            uintptr_t n_times_2,
                            uintptr_t n_times_1,
                            double *out_masked);

/**
 * Compute a multitaper spectrogram via the `multitaper_rs` crate.
 *
 * Wraps `multitaper_rs::compute_spectrogram` with a C ABI matching the
 * shape of the other dynamo_rs ABI entries. f64 throughout — the MATLAB
 * Coder MEX uses f32, so callers swapping over should expect f32-roundoff
 * (~1e-7 relative) drift in output.
 *
 * # Safety
 * `in_` must point to a valid `MtsIn`. `out` must point to a valid
 * (uninitialized or zeroed) `MtsOut`. Array pointers must be non-null
 * for the lengths declared (data_ptr, tapers_ptr, eigen_ptr if used).
 */
int dynamo_multitaper_spectrogram(const struct MtsIn *in_, struct MtsOut *out);

/**
 * Generate K DPSS tapers of length N with time-half-bandwidth NW.
 *
 * Caller-allocated outputs:
 *   `tapers_out`  — row-major (K, N) double, total K*N elements.
 *                   Row k is the k-th Slepian sequence, unit L²-norm.
 *   `ratios_out`  — length-K double, concentration ratios in (0, 1].
 *
 * Matches MATLAB R2025a `dpss(N, NW, K)` and
 * `scipy.signal.windows.dpss(N, NW, K, return_ratios=True)` after the
 * scipy/MATLAB sign convention (even-index tapers have positive sum;
 * odd-index tapers have positive central derivative). Validated to
 * ≤1e-8 elementwise on (N=128, NW=2, K=3) and (N=1024, NW=4, K=7).
 *
 * # Safety
 * `tapers_out` must point to at least `k * n` writable f64s.
 * `ratios_out` must point to at least `k` writable f64s.
 */
int dynamo_dpss(uintptr_t n, double nw, uintptr_t k, double *tapers_out, double *ratios_out);

/**
 * Band-detection artifact mask. **Does NOT include the slope-test
 * branch** — that needs the multitaper spectrogram and is left to the
 * caller (the MATLAB `detect_artifacts.m` ORs the slope mask in
 * separately). The Rust path here matches `detect_artifacts.m` with
 * `'slope_test', false` to within ≤30 samples on the synthetic-EEG
 * fixture battery; validated end-to-end in the upstream
 * `preraulab/artifact_detection` PR #1 and now MEX-bridged here.
 *
 * Caller-allocated output `mask_out` is length `n`; 1 = artifact.
 *
 * # Safety
 * `data` must point to `n` readable f64s; `opts` must point to a valid
 * `ArtifactOptsFFI`; `mask_out` must point to `n` writable u8s.
 */
int dynamo_detect_artifacts(const double *data,
                            uintptr_t n,
                            double fs,
                            const struct ArtifactOptsFFI *opts,
                            uint8_t *mask_out);

/**
 * Compute the per-frequency baseline used in TFpeak extraction.
 *
 * Inputs (all row-major):
 *   `spect`           — shape (n_freqs, n_times) double; baseline is the
 *                       `baseline_ptile`-th percentile over the valid
 *                       columns of each row (zeros treated as NaN).
 *   `stimes`          — length n_times; window-center times.
 *   `t_data`          — length n_data; the EEG time grid that the
 *                       `baseline_exclude` mask is defined on.
 *   `baseline_exclude` — length n_data u8 (1 = exclude); nearest-neighbor
 *                       interpolated onto `stimes` columns.
 *   `baseline_range_lo` / `baseline_range_hi` — time-range trimming
 *                       applied to `stimes`.
 *   `baseline_ptile`  — percentile in [0, 100] (Hyndman-Fan #5, matching
 *                       MATLAB `prctile(..., baseline_ptile, 2)`).
 *
 * Caller-allocated output `baseline_out` is length `n_freqs`.
 *
 * # Safety
 * All input pointers must point to readable buffers of the declared
 * lengths. `baseline_out` must be writable for `n_freqs` f64s.
 */
int dynamo_compute_baseline(const double *spect,
                            uintptr_t n_freqs,
                            uintptr_t n_times,
                            const double *stimes,
                            const double *t_data,
                            uintptr_t n_data,
                            const uint8_t *baseline_exclude,
                            double baseline_range_lo,
                            double baseline_range_hi,
                            double baseline_ptile,
                            double *baseline_out);

/**
 * Compute the SO-power time series and stage / outlier-masked
 * normalization. Mirrors `crate::so_power::so_power_from_spectrogram`
 * (and pydynamo `compute_so_power`); MATLAB analog is `computeSOpower.m`.
 *
 * Output pointers are `Box::leak`-allocated; caller frees with
 * `dynamo_free_buffer_f64(ptr, n_out)` for each of the three arrays.
 *
 * # Safety
 * `in_` and `out` must be valid. All array pointers must back the
 * declared lengths.
 */
int dynamo_so_power(const struct SoPowerIn *in_, struct SoPowerOut *out);

/**
 * Compute unwrapped SO-phase from raw EEG via SOS bandpass + Hilbert +
 * atan2 + unwrap + NaN-at-excluded + stage interp. Mirrors pydynamo
 * `compute_so_phase`; MATLAB analog is `computeSOphase.m`.
 *
 * All four output arrays are length `n_data` and `Box::leak`-allocated.
 * Caller frees with `dynamo_free_buffer_f64(ptr, n_data)` each.
 *
 * `sos_ptr` must point to `n_sections * 6` f64s in scipy layout
 * `[b0 b1 b2 a0 a1 a2]` per section.
 *
 * # Safety
 * All declared array pointers must back the corresponding lengths.
 */
int dynamo_so_phase(const struct SoPhaseIn *in_, struct SoPhaseOut *out);

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

/**
 * See [`dynamo_free_buffer_f64`]. Used for the CSR offset arrays
 * (`height_data_offsets`, `boundary_offsets`) returned in `ExtractTfpeaksOut`.
 */
void dynamo_free_buffer_u64(uint64_t *ptr, uintptr_t len);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  /* DYNAMO_RS_H */
