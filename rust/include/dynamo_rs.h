/* Auto-generated C ABI header for dynamo_rs. Do not edit. */
/* Generated from src/c_api.rs by build.rs via cbindgen. */

#ifndef DYNAMO_RS_H
#define DYNAMO_RS_H

#include <stddef.h>
#include <stdint.h>

/**
 * C ABI version. The ABI contract is **additive-only**: new symbols and
 * new `#define`s may appear, but existing signatures, struct layouts,
 * and enum values never change under the same value. Bump this on any
 * breaking change. Numeric-semantics changes that keep the ABI stable
 * (e.g. a width reparameterization) do NOT bump this. They
 * bump the crate semver, which callers observe via [`dynamo_version`].
 */
#define DYNAMO_C_ABI_VERSION 1

/**
 * Per-mode `[amp/Density, FreqMean, FreqStd, SO*Mean, SO*Std, Theta]` —
 * the emitted 6-param layout. `amp` (index 0) is unused for containment.
 */
#define MODE_FMEAN 1

#define MODE_FSTD 2

#define MODE_SOMEAN 3

#define MODE_SOSTD 4

#define MODE_THETA 5

/**
 * Stats-CSV schema version written by this crate (OUTPUT_FORMAT.md §8.2):
 * format 3 = the 14-column layout below plus the `#` provenance preamble.
 * Format 2 is the same columns with no preamble. Format 1 is the
 * 16-column layout that also carries `bbox_width_s`/`bbox_height_Hz`.
 */
#define STATS_CSV_FORMAT 3

/**
 * The computation-semantics version. Bumped when either:
 *
 * 1. a settings field keeps its name, type and position but changes
 *    what its value means (the original trigger — these entries carry
 *    a real migration), or
 * 2. **any change alters the numbers the pipeline computes** from the
 *    same inputs and settings — a numeric bugfix, a changed constant,
 *    a reworked algorithm. These get a NO-OP migration (precedent:
 *    v2→v3 is log-only) whose `summary` line IS the numeric-changelog
 *    entry.
 *
 * Because every artifact stamps this number (`# semantics:`,
 * dynamo-export `stamp.rs`), it is what lets `check-results` /
 * `validate` say whether an existing tree's numbers still match
 * current code: builds whose shas differ but whose semantics match are
 * numerically comparable (the commits between are UI or tooling).
 * **A numeric change that ships without a bump silently breaks that
 * verdict** — when in doubt, bump.
 *
 * Shape changes are caught by [`fingerprint`] and alone must NOT bump
 * this.
 *
 * | value | date       | what changed |
 * |-------|------------|--------------|
 * | 1     | (pre-2026-08-03) | original convention |
 * | 2     | 2026-08-03 | paramfit Gaussian widths became true standard deviations. `paramfit.{power,phase}.mode_bounds.fstd_*` and `paramfit.power.mode_bounds.pstd_*` are now σ, previously `√2·σ`. `paramfit.phase.mode_bounds.pstd_*` (recikappa) is unchanged. A v1 file's width bounds are `√2` too wide if read as v2. |
 * | 3     | 2026-08-03 | `constrain_freq_center` / `constrain_power_center` now constrain to the VALID (non-NaN) data inside the fit window, not to the window itself, matching MATLAB `param_basis_*.m`. The field names, types and values are unchanged, but the region they describe is narrower wherever `hist.min_time_in_bin` emptied SO-power bins, so the same v2 file yields different fits under v3. Also affects the derived (`null`) value of the new `mode_bounds.{fmean,pmean}_*` slots. |
 * | 4     | 2026-09-11 | `paramfit.peak_assign_prob` was absorbed into the flexible `paramfit.peak_assign`, whose default is argmax within a 1.5σ footprint (serialized `"argmax 1.5 sigma"`) — so an ABSENT key changed meaning (it used to mean a 0.95 probability contour). Migration pins a pre-v4 file to its old meaning: the stored `peak_assign_prob` (or 0.95 when absent) becomes `peak_assign: "<x> p"` and the retired key is removed. (The footprint default was fixed at 1.5σ on 2026-09-12, the day after v4 landed, before any tree outside local test data ran under bare argmax.) |
 * | 5     | 2026-10-13 | The per-mode `Pk*` peak-property columns aggregate by MEDIAN instead of mean (`PkSOphase` stays a circular mean, `PkCount` a count): the per-peak size properties are heavy-tailed and a mean is dominated by a few giant tail events — on the NSS two-night cohort the mean costs all night-to-night subject reliability (PkVolume r ≈ 0.0 mean vs ≈ 0.8 median). Also adds `PkPower` (median per-peak `pow2db(Height × baseline(PeakFrequency))`, absolute power re-anchored to the extraction baseline, which phase-1 now persists in aux h5). No settings value changes shape or meaning; the migration is a changelog no-op. |
 * | 6     | 2026-11-16 | Per-peak `SOphase` is interpolated from the UNWRAPPED SO-phase series and wrapped afterwards (MATLAB `computePeakSOphase.m`). It was interpolated from the wrapped series, so a peak whose two neighboring EEG samples straddle ±π received an arbitrary phase in (−π, π) instead of one near ±π: ~0.7% of peaks on the Twin test cohort, which carved a ~3% trough into every SO-phase histogram bin covering ±π. Changes the stats-table `SOphase` column for those peaks, the SO-phase SOPH, and everything fit to it. No settings value changes shape or meaning; the migration is a changelog no-op. |
 * | 7     | 2026-11-17 | Per-peak `PeakStage`, `SOpower`, and `SOphase` come from the kernel's `assign_peak_*`, which follow MATLAB `computePeakStage.m` / `computePeakSOpower.m` / `computePeakSOphase.m` exactly. The artifact override (stage 6) now tests the EEG sample NEAREST the peak time, as MATLAB's `interp1(..., 'nearest')` does. It tested the sample at or before the peak time, which shifted every artifact span half a sample late, so a peak in the half-sample before an artifact onset was not labeled 6 and a peak in the half-sample after an artifact end was. The SO-power and SO-phase lookups gain MATLAB's one-grid-step endpoint pad (a peak up to one step outside the series takes the end value instead of NaN), which no peak reaches while `sopower.retain_fs` is true. Changes `PeakStage` for peaks within half a sample of an artifact-span edge, and with it the SOPH membership of those peaks. No settings value changes shape or meaning; the migration is a changelog no-op. |
 * | 8     | 2026-11-17 | Artifact detection matches MATLAB `detect_artifacts.m` in two places where it did not. (1) The slope-test mask is folded into the bad-sample set BEFORE the high-frequency and broadband tests, so slope-flagged samples are interpolated out of the filtered signal and left out of the z-score statistics. It was OR-ed onto the finished band mask, which computed the band thresholds from data MATLAB excludes. (2) The 2 s envelope smoothing and 300 s detrend use MATLAB's alignment for even-length `movmean` / `movmedian` windows (samples `i - k/2` to `i + k/2 - 1`). They sat one sample later, which shifted every band-detected artifact span by one sample. On the Twin test cohort the artifact mask differed from MATLAB's at 0.07% to 1.6% of each recording and now differs at 0 or 1 sample. The mask feeds the baseline, SO-power, SO-phase, `PeakStage`, and the surviving peak set, so every downstream number can change. No settings value changes shape or meaning; the migration is a changelog no-op. |
 * | 9     | 2026-11-17 | TF-peak extraction follows the pure-MATLAB code (`runWatershed.m`, `mergeWshedSegment.m`, `mergeRegions.m`) in four places where it did not. (1) After a merge, a region whose only remaining neighbor is the merged region is absorbed into it. (2) Merging continues only while the heaviest edge is strictly above `detection.merge_thresh`. An edge exactly at the threshold used to merge. (3) Watershed regions are renumbered by `bwlabel` order (first pixel going down the columns) before the merge, and each merge keeps the label MATLAB keeps, which decides the order equal-weight edges merge in and which region owns a shared border pixel. (4) Equal-weight edges merge in MATLAB's edge-table order. On the Twin test cohort the per-segment extraction went from 27% of MATLAB's peaks reproduced exactly to all of them. A recording keeps about 93% of its peaks within 0.1 s and 0.5 Hz, loses 1.5% to 1.9% of its peak count, and most surviving peaks move slightly in every property. The SOPH mode-peak finder that seeds the parametric fit keeps its own merge procedure and is unchanged. No settings value changes shape or meaning; the migration is a changelog no-op. |
 * | 10    | 2026-11-17 | Peak-frequency refinement interpolates the event spectrum with a not-a-knot cubic spline, as MATLAB `interp1(..., 'spline')` in `refinePeakFrequency.m` does. It used a natural spline (second derivative forced to zero at both ends of the frequency axis). The two agree away from the ends, so only peaks whose bounding box reaches the bottom or top of the analyzed band move: about 0.1% of peaks on the Twin test cohort, by up to 0.23 Hz, and a handful of peaks per recording change sides of the edge-peak removal. Given the same samples, event times, and bounding boxes, the refined frequency now equals MATLAB's for every event. No settings value changes shape or meaning; the migration is a changelog no-op. |
 * | 11    | 2026-11-17 | The peak BoundingBox is the exact extent of the region's pixels: its corner sits half a bin before the center of the first pixel on each axis (it sat on that center), so the box ends half a bin after the center of the last pixel. `bbox_tl_s` moves half a time bin earlier and `bbox_tl_Hz` half a frequency bin lower for every peak, and width and height are unchanged. The box is the frequency window the refinement searches, so the window moves down by half a bin (0.049 Hz at the defaults): about 1% more peaks survive the edge-peak removal on the Twin test cohort, and a few change refined frequency. MATLAB's corner was a full bin high and is fixed to the same extent in the MATLAB toolbox. No settings value changes shape or meaning; the migration is a changelog no-op. |
 * | 12    | 2026-11-17 | The pass-2 mask zeros each pass-1 region's traced contour, as MATLAB `maskSpectrogram` does with the `bwtraceboundary` borders from `trimWshedRegions.m`. It zeroed every region pixel with a differently labeled neighbor in any of the 8 directions, which also removed pixels touching the outside only at a corner: about 5% of all region pixels on the Twin test cohort. With those pixels kept, pass 2 finds more and slightly larger peaks: the final peak count rises 2.4% to 3.0%, about 92% to 93% of peaks keep a partner within 0.1 s and 0.5 Hz, and the SOPHs correlate with their previous versions at 0.997 (SO-power) and 0.962 to 0.976 (SO-phase). The masked spectrogram now equals MATLAB's pixel for pixel. No settings value changes shape or meaning; the migration is a changelog no-op. |
 * | 13    | 2026-11-17 | Zero-phase filtering follows MATLAB `filtfilt`: each second-order section is run forward and backward on its own odd-reflected extension of `3 x filter order` samples, before the next section starts. It ran all sections forward and then all backward on one extension of `6 x sections + 3` samples (the scipy `sosfiltfilt` procedure). The two agree away from the ends of a recording and differ inside the filter's start-up transient, which for the narrow slow-oscillation band lasts minutes: the filtered slow-oscillation signal differed from MATLAB's by several times its RMS in the first and last 10 s, and SO-phase differed by more than 0.01 rad at 1% to 5% of samples. It now equals MATLAB's to 1e-10 of RMS, and SO-phase to 1e-7 rad. Effect on outputs is small because the ends of a night are mostly wake: SO-phase SOPHs correlate with their previous versions at 0.999 or better, and peaks, artifact masks, and SO-power are unchanged or change by a handful of samples. No settings value changes shape or meaning; the migration is a changelog no-op. |
 * | 14    | 2026-11-17 | The minimum TF-peak height is computed from the chi² quantile for any number of tapers, as MATLAB `computeTFPeaks.m` does: `-2 x pow2db(2K / chi2inv(0.975, 2K))`. It was a stored constant for 3 tapers and no floor at all for any other count, so a run with `mtm.n_tapers` other than 3 kept every low peak that MATLAB removes. The stored 3-taper quantile was also off in the 8th digit (14.449374448 against MATLAB's 14.449375335), which moves the default floor by 5e-7 dB: no peak changes on the three Twin test nights. Runs at the default 3 tapers are unchanged. Runs at another taper count now lose the peaks below the floor. No settings value changes shape or meaning; the migration is a changelog no-op. |
 * | 15    | 2026-11-17 | `PrefPhase` and `Coupling` in the power paramfit CSV are NaN for a mode whose `FreqMean` lies outside the phase fit's frequency window. The phase model is zero there, so the argmax returned the first SO-phase bin with a coupling of 0, a value that reads as real. Only reachable when the power and phase fits use different frequency windows, or a frequency row is invalid for the phase fit only. Every other value is unchanged. No settings value changes shape or meaning; the migration is a changelog no-op. |
 * | 16    | 2026-11-19 | The TF-peak region labels found on the downsampled segment are resized to full resolution exactly as MATLAB `imresize(..., 'nearest')` does (kernel `resize_labels_nn`, DYNAM-O_rs #39). A full-resolution column that falls halfway between two downsampled columns was rounded half to even. `imresize` settles it by its own floating-point arithmetic, so for a 571-column segment (30 s at the default 0.05 s step) column 286 read a different downsampled column than in MATLAB, and regions touching it entered the trim with a few more or fewer pixels. On the three Twin test nights 2 of 136,238 peaks move (by at most 0.028 s and 0.007 Hz), no peak is added or removed, and every paramfit value is unchanged. No settings value changes shape or meaning; the migration is a changelog no-op. |
 * | 17    | 2026-11-22 | SOPHs rebuilt from a results folder's cache (the second phase of `dynamo-cli batch`, and app re-runs that rebuild SOPHs or fits from `stats.csv` + `aux.h5`) count SO-power time-in-bin on the same series as a full run and MATLAB `computeSOpower.m` with `retain_Fs`: the cached window series interpolated through its finite windows onto the EEG sample grid, then NaN at the artifact samples only. The rebuild used the window series as stored, where every window overlapping an artifact and every outlier window is NaN, so it counted about 6% less valid NREM time on the Twin test cohort and its SO-power SOPH rates came out 2% to 8% high, by a different factor in each SO-power row (relative L1 2.7% to 5.4% against a full run; on one night a column crossed `hist.min_time_in_bin`). It now matches a full run to about 1e-5, the remainder being peaks on a bin edge whose per-peak values the stats CSV stores rounded. The rebuild's sample grid also includes both ends of the scored window, as a full run's does: it was one sample short, which moved the end of a recording's last artifact span by one sample in a rebuilt aux file. Full runs are unchanged. No settings value changes shape or meaning; the migration is a changelog no-op. |
 *
 * A value changing meaning without changing its name or type is a rare
 * and deliberate act; entries land here only when the same file would
 * yield different outputs under new code. The read path stays
 * regardless: a few hundred existing files are v1, and the conversion
 * from `√2·σ` to `σ` has to remain available or those files become
 * unmigratable. Freezing the table is not a reason to delete the field.
 */
#define SEMANTICS_VERSION 17

/**
 * Footprint radius the density rules use when none is spelled out.
 * 1.5σ hugs the visually distinct mode core (~67.5% of the mode's
 * mass), and the NSS footprint sweep showed the Pk property means sit
 * on a plateau around it (1.3–1.7σ moves them by only a few percent —
 * the footprint choice mainly scales PkCount).
 */
#define DEFAULT_DENSITY_FOOTPRINT_SIGMA 1.5

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
 *      border fill (see `expand_labels_distance`), resize labels up, trim,
 *      regionprops.
 *   3. Concatenate per-segment peaks.
 *   4. Apply filterStatsTable: keep peaks where
 *        dur_min < Duration < dur_max,
 *        bw_min  < Bandwidth < bw_max,
 *        freq_min < PeakFrequency < freq_max,
 *        pow2db(Height) > ht_db_min.
 *
 * Sentinel values that select a default:
 *   * `seg_time` <= 0 → 30.0 s
 *   * `downsample_f` / `downsample_t` = 0 → 1 (no downsampling)
 *   * `max_merges` = 0 → +∞
 *   * `freq_max` = 0 → +∞
 * `dur_max`, `bw_max` and `ht_db_min` have no sentinel. Pass +∞, +∞ and
 * -∞ respectively to disable those cuts.
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
   * Border-fill mode applied after the merge. 0 = MATLAB paint in label
   * order (8-connected 1-pixel dilation, higher labels overwrite lower),
   * matching extractTFPeaks.m. N > 0 = skimage-style expand_labels
   * (breadth-first fill out to N pixels, lower label wins ties).
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
 * All `*mut` pointers are callee-allocated. The caller must free each with the
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
 * Both offset arrays have length `n_peaks + 1` (always, even when n_peaks
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
   * Peakiness = N/(N-1) * (max - mean) / (max - min) of the region's N
   * pixels per peak, unitless in [0, 1] (0 = flat plateau, 1 = spike;
   * small-sample corrected). Length n_peaks.
   */
  double *peakiness;
  /**
   * Flattened per-peak pixel values, length `n_height_data_elems`.
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
 * pre-computed DPSS taper bank (caller supplies it, for example from
 * [`dynamo_dpss`]), row-major (n_tapers × winsize) so element (k, i) is
 * at `tapers_ptr[k * winsize + i]`. Caller MUST guarantee
 * `winsize == round(window_size_s * fs)`. The underlying compute
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
 * All three pointers are callee-allocated and must be released with
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

/**
 * Input descriptor for [`dynamo_so_power`]. `spect_ptr` is a row-major
 * (n_freqs, n_times) SO-band spectrogram.
 */
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
   * ASCII norm method spec, e.g. "p2shift1234", "percent", "none".
   * Exactly `norm_method_len` bytes are read, so no NUL terminator is
   * needed and none should be counted in the length. Parsed by
   * `NormMethod::parse`.
   */
  const uint8_t *norm_method_ptr;
  uintptr_t norm_method_len;
} SoPowerIn;

/**
 * Output descriptor for [`dynamo_so_power`]. The three arrays are
 * callee-allocated with length `n_out`.
 */
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

/**
 * Input descriptor for [`dynamo_so_phase`].
 */
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

/**
 * Output descriptor for [`dynamo_so_phase`]. All four arrays are
 * callee-allocated with length `n_out`.
 */
typedef struct SoPhaseOut {
  double *so_phase_ptr;
  double *so_phase_times_ptr;
  double *so_phase_stages_ptr;
  double *filtdata_ptr;
  uintptr_t n_out;
} SoPhaseOut;

/**
 * Input descriptor for [`dynamo_spline_basis_fit`].
 *
 * Layout mirrors MATLAB `spline_basis.m` semantics:
 *   * `soph` is `(n_x, n_y)` row-major, where `n_x = feat_bins.len()` and
 *     `n_y = freq_bins.len()`. This is the **transposed** orientation of
 *     the canonical SOPH (the MATLAB-side wrapper does the transpose).
 *   * `internal_knots_x`/`y` are the pre-`augknt` knot vectors.
 *   * `order = 4` and `boundary_multiplicity = 3` reproduce DYNAM-O.
 */
typedef struct SplineBasisIn {
  const double *soph_ptr;
  uintptr_t n_x;
  uintptr_t n_y;
  const double *feat_bins_ptr;
  const double *freq_bins_ptr;
  const double *internal_knots_x_ptr;
  uintptr_t n_internal_knots_x;
  const double *internal_knots_y_ptr;
  uintptr_t n_internal_knots_y;
  uint32_t order;
  uint32_t boundary_multiplicity;
} SplineBasisIn;

/**
 * Output descriptor for [`dynamo_spline_basis_fit`].
 */
typedef struct SplineBasisOut {
  /**
   * `(m_y, m_x)` row-major. Matches MATLAB `squeeze(spline_obj.coefs)'`.
   */
  double *coefs_ptr;
  uintptr_t m_y;
  uintptr_t m_x;
  /**
   * `(n_x, n_y)` row-major. Matches MATLAB `splinefit` (= `size(SOPH')`).
   */
  double *splinefit_ptr;
  /**
   * Augmented knot vectors.
   */
  double *knots_x_aug_ptr;
  uintptr_t n_knots_x_aug;
  double *knots_y_aug_ptr;
  uintptr_t n_knots_y_aug;
} SplineBasisOut;

/**
 * Input descriptor for the two paramfit kernels (`dynamo_rotgauss_fit`
 * and `dynamo_vmgauss_fit`).
 *
 * SOPH is `(n_y, n_x)` row-major where `n_y = freq_bins.len()` and
 * `n_x = feat_bins.len()`. This matches the canonical MATLAB SOPH
 * `(n_freqs, n_features)` column-major byte-for-byte, so `mxGetPr` can be
 * passed directly without copying.
 *
 * Initial/lower/upper are `(n_modes, 6)` row-major:
 *   * power: `[amp, fmean, fstd, pmean,    pstd,      theta]`
 *   * phase: `[amp, fmean, fstd, phasepref, recikappa, theta]`
 * `fstd` is a frequency standard deviation in Hz for both fit types.
 *
 * `bg_initial`, `bg_lower`, `bg_upper` are 3-vectors `[xxx, yyy, zzz]`.
 *
 * # Width convention
 *
 * The Gaussian widths are standard deviations. Both kernels carry an
 * explicit factor of one half in the exponent:
 *
 * ```text
 *   power: amp * exp(-0.5*(u/fstd)^2 - 0.5*(v/pstd)^2)
 *   phase: amp * exp(-0.5*(dy/fstd)^2) * exp(k*(cos(dx) - 1)), k = 1/recikappa^2
 * ```
 *
 * Widths written by kernel versions below 0.2.0 are `sqrt(2)` times
 * larger than these standard deviations, although the struct layout, the
 * field types and the function signatures are identical. Values cross
 * this ABI unchanged, so divide stored widths from those versions by
 * `sqrt(2)` before reusing them as a modeled surface or bound window, and
 * do not pool widths across the two conventions. Seeds and priors already
 * expressed as standard deviations keep their numeric values.
 *
 * Column 4 must NOT be converted by index. A stored `pstd` from a version
 * below 0.2.0 for `dynamo_rotgauss_fit` rescales by `1/sqrt(2)`, while
 * `recikappa` for `dynamo_vmgauss_fit` needs no conversion, because the
 * von Mises factor is its own small-angle Gaussian and already carries
 * the half.
 */
typedef struct ParamFitIn {
  const double *soph_ptr;
  uintptr_t n_y;
  uintptr_t n_x;
  const double *feat_bins_ptr;
  const double *freq_bins_ptr;
  const double *initial_ptr;
  const double *lower_ptr;
  const double *upper_ptr;
  uintptr_t n_modes;
  double bg_initial[3];
  double bg_lower[3];
  double bg_upper[3];
  uint32_t max_iters;
  /**
   * vmGauss-only: nonzero = row-normalize the assembled model (matches
   * MATLAB `fit_vmGauss.m`'s `unit_row=true`). Ignored by rotgauss_fit.
   */
  uint32_t unit_row;
} ParamFitIn;

/**
 * Output descriptor for [`dynamo_rotgauss_fit`] and [`dynamo_vmgauss_fit`].
 */
typedef struct ParamFitOutFFI {
  /**
   * Final parameters, `(n_modes, 6)` row-major. Allocated, length `n_modes * 6`.
   */
  double *params_ptr;
  /**
   * Background-plane coefficients `[xxx, yyy, zzz]`.
   */
  double background[3];
  /**
   * Model reconstruction on the input grid, `(n_y, n_x)` row-major.
   */
  double *model_ptr;
  uintptr_t n_y;
  uintptr_t n_x;
  /**
   * gof.sse / rsquare / adjrsquare / rmse / dfe / dfm.
   */
  double gof_sse;
  double gof_rsquare;
  double gof_adjrsquare;
  double gof_rmse;
  double gof_dfe;
  double gof_dfm;
  uint32_t iters_used;
} ParamFitOutFFI;

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/**
 * Runtime accessor for [`DYNAMO_C_ABI_VERSION`], for `dlopen` /
 * `loadlibrary` consumers that cannot see the compile-time constant.
 */
uint32_t dynamo_c_abi_version(void);

/**
 * The kernel build identity as a NUL-terminated static string in the
 * DYNAM-O provenance grammar `<semver>+<sha12>[.dirty]` (or
 * `<semver>+unknown`). This is the `kernel_version` every consumer
 * (MEX bridge, PyO3, app/CLI) should record next to its outputs. The
 * pointer is `'static`. Callers must NOT free it.
 */
const char *dynamo_version(void);

/**
 * Run the full TF-peak extraction pipeline on a whole spectrogram, which is
 * split internally into `seg_time`-second segments.
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
 *   * `data`, `n_data`, `fs`: raw 1-D signal and sampling rate.
 *   * `freq_lo`, `freq_hi`, `window_size` (s), `dsfreqs` (Hz).
 *
 * Caller provides preallocated output buffers:
 *   * `out_peak_freq[i]`: refined frequency. Peaks within `window_size/2`
 *     of either end of `data` keep their input `peak_freq[i]`. NaN when
 *     the refined maximum sits on the edge of the peak's frequency band.
 *   * `out_keep[i]`: 1 for a peak near either end of `data` (it keeps its
 *     input frequency). For every other peak, 1 when the refined frequency
 *     is finite and 0 when it is NaN.
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
 * Caller-provided output buffers (must all be non-null). The 2-D outputs
 * are written column-major (MATLAB layout):
 *   * `out_c_mat`: `(num_cbins, num_fbins)`, element (c, f) at
 *     `f * num_cbins + c`.
 *   * `out_time_in_bin`: `(num_cbins, 5)` column-major (minutes).
 *   * `out_prop_in_bin`: `(num_cbins, 5)` column-major.
 *   * `out_peak_at_freq`: `num_fbins`.
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
 * `labels_1s` is (n_freqs, n_times_1) int64. `spect_2s` and `out_masked`
 * are both (n_freqs, n_times_2) float64.
 *
 * Caller allocates `out_masked` of size `n_freqs * n_times_2`.
 *
 * Returns 0 on success, negative on error.
 *
 * # Safety
 * All pointers must be valid for the lengths derived from `n_freqs`,
 * `n_times_1`, `n_times_2`. `out_masked` must be writable.
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
 * shape of the other dynamo_rs ABI entries. f64 throughout. The MATLAB
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
 *   `tapers_out`: row-major (K, N) double, total K*N elements.
 *                   Row k is the k-th Slepian sequence, unit L²-norm.
 *   `ratios_out`: length-K double, concentration ratios in (0, 1].
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
 * branch**. That needs the multitaper spectrogram and is left to the
 * caller (the MATLAB `detect_artifacts.m` ORs the slope mask in
 * separately). The Rust path here matches `detect_artifacts.m` with
 * `'slope_test', false` to within ≤30 samples on the synthetic-EEG
 * fixture battery.
 *
 * Caller-allocated output `mask_out` is length `n`; 1 = artifact.
 *
 * # Safety
 * `data` must point to `n` readable f64s, `opts` must point to a valid
 * `ArtifactOptsFFI`, and `mask_out` must point to `n` writable u8s.
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
 *   `spect`: shape (n_freqs, n_times) double. The baseline is the
 *                       `baseline_ptile`-th percentile over the valid
 *                       columns of each row (zeros treated as NaN).
 *   `stimes`: length n_times, window-center times.
 *   `t_data`: length n_data, the EEG time grid that the
 *                       `baseline_exclude` mask is defined on.
 *   `baseline_exclude`: length n_data u8 (1 = exclude), nearest-neighbor
 *                       interpolated onto `stimes` columns.
 *   `baseline_range_lo` / `baseline_range_hi`: time-range trimming
 *                       applied to `stimes`.
 *   `baseline_ptile`: percentile in [0, 100] (Hyndman-Fan #5, matching
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
 * (and pydynamo `compute_so_power`). The MATLAB analog is `computeSOpower.m`.
 *
 * Output pointers are callee-allocated. Caller frees with
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
 * `compute_so_phase`. The MATLAB analog is `computeSOphase.m`.
 *
 * All four output arrays are length `n_data` and callee-allocated.
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
 * Free an f64 buffer returned through any callee-allocated output
 * pointer in this module (`ExtractTfpeaksOut`, `MtsOut`, `SoPowerOut`,
 * `SoPhaseOut`, `SplineBasisOut`, `ParamFitOutFFI`).
 *
 * `len` MUST equal the element count of the original allocation, as
 * documented on the struct field that returned the pointer.
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

/**
 * Fit a bivariate tensor-product B-spline on a regular grid.
 *
 * Parity-tested against MATLAB `spap2` to f64 round-off (see
 * `tests/spline_basis_parity.rs`).
 *
 * All four output arrays are callee-allocated. Caller frees with
 * `dynamo_free_buffer_f64`:
 *   * `coefs_ptr`       length `m_y * m_x`
 *   * `splinefit_ptr`   length `n_x * n_y`
 *   * `knots_x_aug_ptr` length `n_knots_x_aug`
 *   * `knots_y_aug_ptr` length `n_knots_y_aug`
 *
 * # Safety
 * All declared array pointers must back the corresponding lengths.
 */
int dynamo_spline_basis_fit(const struct SplineBasisIn *in_, struct SplineBasisOut *out);

/**
 * Fit a rotated-Gaussian-mixture model + linear background plane to a SOPH
 * histogram. Mirrors MATLAB `fit_rotGauss` (without `prepareSurfaceData`'s
 * NaN drop. The caller must pass a finite-valued grid).
 *
 * # Safety
 * All pointers must back the declared lengths.
 */
int dynamo_rotgauss_fit(const struct ParamFitIn *in_, struct ParamFitOutFFI *out);

/**
 * Fit a von-Mises × Gaussian mixture + sinusoidal background to a SOPH
 * histogram. `ParamFitIn::unit_row` controls per-row normalization;
 * nonzero matches MATLAB `fit_vmGauss`.
 *
 * # Safety
 * All pointers must back the declared lengths.
 */
int dynamo_vmgauss_fit(const struct ParamFitIn *in_, struct ParamFitOutFFI *out);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  /* DYNAMO_RS_H */
