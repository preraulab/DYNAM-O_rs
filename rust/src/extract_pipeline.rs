//! Full `extract_tfpeaks` pipeline: segmentation + per-segment downsampling +
//! watershed + merge + expand-labels + resize + trim + peak properties +
//! final filterStatsTable. Port of
//! `pydynamo.tfpeaks.extract.extract_tfpeaks` and MATLAB's
//! `runSegmentedData` + `filterStatsTable`.
//!
//! Design notes
//! ------------
//! *Segmentation* follows MATLAB `segmentData.m:85-107`:
//!     max_dx = floor(seg_time / dt)
//!     n_segs = ceil(T / max_dx)
//!     new_dx = ceil(T / n_segs)
//! Each segment covers columns `[(ii-1)*new_dx+1 .. min(ii*new_dx, T)]` in
//! MATLAB 1-based, which is `[i*new_dx .. min((i+1)*new_dx, T)]` in Rust.
//!
//! *Downsampling* is simple stride slicing `spect[::f_f, ::t_f]`, matching
//! pydynamo (pyDYNAM-O compatibility). After merge, labels are resized back
//! up to the full-resolution segment shape using nearest-neighbor.
//!
//! *Peak properties* match pydynamo's `regionprops_table` exactly:
//!   - `PeakTime  = sum(v*t) / sum(v)` over label pixels (intensity-weighted centroid)
//!   - `PeakFrequency = sum(v*f) / sum(v)`
//!   - `Duration  = (t_max - t_min + 1) * d_time`   (N_pixels * dt)
//!   - `Bandwidth = (f_max - f_min + 1) * d_freq`
//!   - `Height    = max(v) - min(v)`  over label pixels (linear, pow2db'ed in filter)
//!   - `Volume    = sum(v) * d_time * d_freq`

use ndarray::{Array2, ArrayView1, ArrayView2};
use crate::parallel::*;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};

/// `TFPEAKS_DEBUG` per-stage region tallies, summed across all segments of
/// one `extract_tfpeaks` call (reset at the start of each call). Lets the
/// caller print a watershed → merge → dur/bw-cut → trim breakdown.
pub static DBG_WS: AtomicU64 = AtomicU64::new(0);
pub static DBG_MERGE: AtomicU64 = AtomicU64::new(0);
pub static DBG_CUT: AtomicU64 = AtomicU64::new(0);
pub static DBG_TRIM: AtomicU64 = AtomicU64::new(0);

fn tfpeaks_debug() -> bool {
    std::env::var("TFPEAKS_DEBUG").is_ok()
}

/// Count distinct positive labels in a label image.
fn count_labels(a: &Array2<i64>) -> u64 {
    let mut seen = std::collections::HashSet::new();
    for &v in a.iter() {
        if v > 0 {
            seen.insert(v);
        }
    }
    seen.len() as u64
}

/// Per-segment peak table (one row per surviving label).
///
/// `area`, `height_data`, `boundaries_xy`, `peakiness` mirror the MATLAB
/// `computePeakStatsTable` columns of the same names. `boundaries_xy[i]`
/// is interleaved `[t1, f1, t2, f2, ...]` with one (t, f) pair per
/// boundary pixel of peak i (8-connectivity perimeter, in physical units).
/// `height_data[i]` is the spectrogram values at every interior pixel of
/// peak i, in MATLAB regionprops column-major order (bit-comparable across
/// backends). `peakiness = N/(N-1) * (max - mean) / (max - min)` of the
/// region's N pixels, unitless in [0, 1] (0 = flat plateau, 1 = spike;
/// the N/(N-1) small-sample factor makes a spike score exactly 1).
///
/// `boundaries_xy[i]` is the perimeter in **clockwise traversal order**
/// (Moore-neighbor trace). This deliberately diverges from MATLAB's pure
/// path, which stores perimeter pixels in column-major scan order and so
/// can't be plotted as a polyline without a separate trace pass. See
/// trimWshedRegions.m:229 (`find(bnd_mask)`). Storing them traversal-
/// ordered means downstream plot() of `Boundaries{i}(:, 1:2)` draws a
/// closed perimeter directly.
#[derive(Default)]
pub struct SegmentPeaks {
    /// Intensity-weighted centroid time of each peak, in seconds.
    pub peak_time: Vec<f64>,
    /// Intensity-weighted centroid frequency of each peak, in Hz.
    pub peak_freq: Vec<f64>,
    /// Bounding-box extent along time, in seconds.
    pub duration: Vec<f64>,
    /// Bounding-box extent along frequency, in Hz.
    pub bandwidth: Vec<f64>,
    /// max - min of the peak's pixel values (linear power).
    pub height: Vec<f64>,
    /// Sum of the pixel values times d_time * d_freq.
    pub volume: Vec<f64>,
    /// 1-based number of the segment each peak came from.
    pub segment_num: Vec<f64>,
    /// Four values per peak (t_tl, f_tl, width_s, height_Hz). The box is the
    /// exact pixel extent, with the corner half a bin before the first pixel
    /// center.
    pub bbox: Vec<f64>,
    /// Area per peak in s*Hz (pixel_count * d_time * d_freq).
    pub area: Vec<f64>,
    /// N/(N-1) * (max - mean) / (max - min) of the region's pixels, in [0, 1].
    pub peakiness: Vec<f64>,
    /// Pixel values per peak (interior, NaN excluded).
    pub height_data: Vec<Vec<f64>>,
    /// Interleaved (t, f) pairs per peak.
    pub boundaries_xy: Vec<Vec<f64>>,
}

impl SegmentPeaks {
    /// Number of peaks in the table.
    pub fn len(&self) -> usize {
        self.peak_time.len()
    }

    /// True when the table holds no peaks.
    pub fn is_empty(&self) -> bool {
        self.peak_time.is_empty()
    }

    /// Extend self with `other` in order.
    pub fn extend(&mut self, mut other: SegmentPeaks) {
        self.peak_time.append(&mut other.peak_time);
        self.peak_freq.append(&mut other.peak_freq);
        self.duration.append(&mut other.duration);
        self.bandwidth.append(&mut other.bandwidth);
        self.height.append(&mut other.height);
        self.volume.append(&mut other.volume);
        self.segment_num.append(&mut other.segment_num);
        self.bbox.append(&mut other.bbox);
        self.area.append(&mut other.area);
        self.peakiness.append(&mut other.peakiness);
        self.height_data.append(&mut other.height_data);
        self.boundaries_xy.append(&mut other.boundaries_xy);
    }
}

/// Segment length used when `seg_time` is not positive, in seconds.
const DEFAULT_SEG_TIME_S: f64 = 30.0;

/// Parameters for `extract_tfpeaks`.
#[derive(Clone, Copy)]
pub struct ExtractParams {
    /// Segment length in seconds. The spectrogram is split into pieces about
    /// this long. A value <= 0 means 30 s.
    pub seg_time: f64,
    /// Keep every n-th frequency bin for the watershed and merge. 1 keeps all.
    pub downsample_f: usize,
    /// Keep every n-th time bin for the watershed and merge. 1 keeps all.
    pub downsample_t: usize,
    /// Regions keep merging while the heaviest edge weight is above this.
    pub merge_thresh: f64,
    /// Cap on the number of merges per segment. Infinity means no cap.
    pub max_merges: f64,
    /// Fraction of each region's volume the trim keeps.
    pub trim_vol_thresh: f64,
    /// Floor from which trim volume is measured. NaN means the segment's
    /// minimum.
    pub trim_shift_val: f64,
    /// A peak is kept when its Duration in seconds is strictly between
    /// `dur_min` and `dur_max`.
    pub dur_min: f64,
    pub dur_max: f64,
    /// A peak is kept when its Bandwidth in Hz is strictly between `bw_min`
    /// and `bw_max`.
    pub bw_min: f64,
    pub bw_max: f64,
    /// A peak is kept when its PeakFrequency in Hz is strictly between
    /// `freq_min` and `freq_max`.
    pub freq_min: f64,
    pub freq_max: f64,
    /// A peak is kept when its Height in dB (10*log10) is above this.
    pub ht_db_min: f64,
    /// Border-fill mode applied after the merge. 0 = MATLAB paint in label
    /// order (8-connected 1-pixel dilation, higher labels overwrite lower),
    /// matching extractTFPeaks.m. N > 0 = skimage-style expand_labels
    /// (breadth-first fill out to N pixels, lower label wins ties), which is
    /// what pydynamo does (its default is 5).
    pub expand_labels_distance: u32,
}

/// Stride-downsample a 2-D array: `out = arr[::f_stride, ::t_stride]`.
fn stride_downsample(arr: ArrayView2<f64>, f_stride: usize, t_stride: usize) -> Array2<f64> {
    if f_stride <= 1 && t_stride <= 1 {
        return arr.to_owned();
    }
    let f_stride = f_stride.max(1);
    let t_stride = t_stride.max(1);
    let (h, w) = arr.dim();
    let out_rows = (h + f_stride - 1) / f_stride;
    let out_cols = (w + t_stride - 1) / t_stride;
    let mut out = Array2::<f64>::zeros((out_rows, out_cols));
    for i in 0..out_rows {
        let src_row = i * f_stride;
        for j in 0..out_cols {
            let src_col = j * t_stride;
            out[[i, j]] = arr[[src_row, src_col]];
        }
    }
    out
}

/// Resize a label image to `(out_rows, out_cols)` by nearest neighbor,
/// matching MATLAB `imresize(labels, [out_rows out_cols], 'nearest')` as
/// extractTFPeaks.m calls it.
///
/// For an integer upsampling factor this is `j / factor`. When the output
/// length is not a multiple of the input length, some output positions fall
/// exactly halfway between two input positions. `imresize` settles those by
/// its own floating-point arithmetic, so the expression below is written in
/// the same order as `imresize`'s and must stay that way.
fn resize_labels_nn(labels: ArrayView2<i64>, out_rows: usize, out_cols: usize) -> Array2<i64> {
    /// Input index that output index `out_index` reads along one axis.
    fn nearest_source_index(out_index: usize, out_len: usize, in_len: usize) -> usize {
        if out_len == 0 || in_len == 0 {
            return 0;
        }
        // imresize works in 1-based coordinates.
        let scale = out_len as f64 / in_len as f64;
        let out_position = (out_index + 1) as f64;
        let source_position = out_position / scale + 0.5 * (1.0 - 1.0 / scale);
        let source_one_based = (source_position + 0.5).floor() as i64;
        let source_clamped = source_one_based.clamp(1, in_len as i64);
        (source_clamped - 1) as usize
    }

    let (in_rows, in_cols) = labels.dim();
    if in_rows == out_rows && in_cols == out_cols {
        return labels.to_owned();
    }
    let mut out = Array2::<i64>::zeros((out_rows, out_cols));
    for i in 0..out_rows {
        let src_row = nearest_source_index(i, out_rows, in_rows);
        for j in 0..out_cols {
            let src_col = nearest_source_index(j, out_cols, in_cols);
            out[[i, j]] = labels[[src_row, src_col]];
        }
    }
    out
}

/// MATLAB extractTFPeaks.m paint-in-label-order semantics. For each cell
/// index i=1..N in ascending order (N = number of distinct non-zero labels
/// in the input, compacted densely):
///   1. Compute 8-conn 1-pixel dilation of the cell-i mask.
///   2. Set output[dilated_mask] = i.
///
/// Higher cell indices overwrite lower on shared borders. Output has NO
/// watershed 0-lines (every pixel adjacent to any region is labeled).
/// Exactly matches MATLAB's `Ldata(rgn{ii}) = ii` for `ii=1..N` where
/// `rgn{ii}` is `union(interior, 8-conn-dilation-border)` from
/// Ldata2graph.m.
///
/// The output uses DENSE 1..N cell indices, not the input's label values,
/// because MATLAB paints by cell index. With the sparse original labels,
/// regionprops counts regions differently and the peak count does not
/// match MATLAB.
pub fn matlab_paint_labels_in_order(labels: ArrayView2<i64>) -> Array2<i64> {
    let (h, w) = labels.dim();
    let mut out = Array2::<i64>::zeros((h, w));
    if h == 0 || w == 0 {
        return out;
    }
    // Bucket pixels by label, then paint in ascending label order with dense
    // cell indices (matching MATLAB's 1..N paint).
    let mut by_label: BTreeMap<i64, Vec<(usize, usize)>> = BTreeMap::new();
    for r in 0..h {
        for c in 0..w {
            let v = labels[[r, c]];
            if v > 0 {
                by_label.entry(v).or_default().push((r, c));
            }
        }
    }
    if by_label.is_empty() {
        return out;
    }
    for (idx, (_orig_label, pixels)) in by_label.iter().enumerate() {
        let cell_idx = (idx + 1) as i64;
        for &(r, c) in pixels {
            for dr in [-1i32, 0, 1] {
                for dc in [-1i32, 0, 1] {
                    let nr = r as i32 + dr;
                    let nc = c as i32 + dc;
                    if nr < 0 || nr >= h as i32 || nc < 0 || nc >= w as i32 {
                        continue;
                    }
                    out[[nr as usize, nc as usize]] = cell_idx;
                }
            }
        }
    }
    out
}

/// `skimage.segmentation.expand_labels(labels, distance=d)`: for every
/// pixel with label 0 within Euclidean distance `d` of some labeled pixel,
/// fill with the nearest label. Ties are broken by the lowest label id.
///
/// Implemented via multi-source BFS (4-connectivity, stops after `d` steps).
/// This approximates Euclidean distance but matches the behavior needed
/// to plug the watershed 0-line. For distance=5 it is effectively
/// equivalent (skimage itself computes a distance transform which gives
/// the same result on thin separators).
fn expand_labels_bfs(labels: ArrayView2<i64>, distance: usize) -> Array2<i64> {
    let (h, w) = labels.dim();
    let mut out = labels.to_owned();
    if distance == 0 || h == 0 || w == 0 {
        return out;
    }
    // dist: steps to nearest labeled source; u32::MAX = unreached.
    let mut dist = vec![u32::MAX; h * w];
    let mut queue: VecDeque<(usize, usize)> = VecDeque::new();
    for r in 0..h {
        for c in 0..w {
            if out[[r, c]] != 0 {
                dist[r * w + c] = 0;
                queue.push_back((r, c));
            }
        }
    }
    let max_steps = distance as u32;
    while let Some((r, c)) = queue.pop_front() {
        let steps_here = dist[r * w + c];
        if steps_here >= max_steps {
            continue;
        }
        let source_label = out[[r, c]];
        let steps_next = steps_here + 1;
        // wrapping_sub turns 0 - 1 into usize::MAX, which the bounds check
        // below rejects.
        let neighbors = [
            (r.wrapping_sub(1), c),
            (r + 1, c),
            (r, c.wrapping_sub(1)),
            (r, c + 1),
        ];
        for (nr, nc) in neighbors {
            if nr >= h || nc >= w {
                continue;
            }
            let idx = nr * w + nc;
            if dist[idx] > steps_next {
                // First visit (or a closer source was found).
                if out[[nr, nc]] == 0 {
                    dist[idx] = steps_next;
                    out[[nr, nc]] = source_label;
                    queue.push_back((nr, nc));
                }
            } else if dist[idx] == steps_next {
                // Tie → prefer lower label (stable, matches "first reached")
                let existing = out[[nr, nc]];
                if existing != 0 && source_label != 0 && source_label < existing {
                    out[[nr, nc]] = source_label;
                }
            }
        }
    }
    out
}

/// Intensity-weighted centroid + bbox + max/min intensity + volume per
/// label. Matches pydynamo's `regionprops_table` columns
/// `centroid_weighted`, `bbox`, `intensity_max`, `intensity_min` on the
/// supplied `spect` image.
struct Accum {
    sum_v: f64,
    sum_v_r: f64, // Σ v*r (pixel row index)
    sum_v_c: f64,
    min_v: f64,
    max_v: f64,
    r_min: usize,
    r_max: usize,
    c_min: usize,
    c_max: usize,
    pixel_count: usize,
    pixel_values: Vec<f64>,    // matches MATLAB regionprops PixelValues
    /// First column-major-encountered in-region pixel (smallest col, then
    /// smallest row within that col). Guaranteed boundary pixel and
    /// guaranteed to have West/N/NW as out-of-region neighbors. So it's a
    /// safe Moore-neighbor trace start with initial backtrack = West.
    start_r: usize,
    start_c: usize,
}

// Moore-neighbor 8-conn offsets, clockwise starting from West.
// Index: 0=W, 1=NW, 2=N, 3=NE, 4=E, 5=SE, 6=S, 7=SW.
const MOORE_DIRS: [(i32, i32); 8] = [
    ( 0, -1), (-1, -1), (-1, 0), (-1,  1),
    ( 0,  1), ( 1,  1), ( 1, 0), ( 1, -1),
];

// After moving in direction `move_idx`, the new backtrack ("where we came
// from") is at this MOORE_DIRS index from the new position. Derived once
// algebraically: the out-of-region (OOR) pixel just before `move_idx` in the clockwise
// scan is at MOORE_DIRS[(move_idx+7)%8] from the OLD position. Its
// position relative to the NEW position is MOORE_DIRS[(move_idx+7)%8]
// minus MOORE_DIRS[move_idx], which always lands on one of {N, E, S, W}.
const MOORE_NEW_C_DIR: [usize; 8] = [6, 6, 0, 0, 2, 2, 4, 4];

/// Moore-neighbor boundary trace (Pavlidis-style, 8-conn, clockwise).
///
/// Walks the perimeter of the connected region with label `lbl` starting
/// at `(start_r, start_c)` (must be a boundary pixel, conventionally the
/// column-major-first in-region pixel, which guarantees W is out of the
/// region (OOR) so we can use West as the initial backtrack).
///
/// Returns pixel coords in clockwise traversal order. Consecutive entries
/// are 8-adjacent, so plot()-as-polyline draws a closed perimeter rather
/// than the sawtooth that column-major scan order produces. The start
/// pixel is repeated as the last entry, which closes the loop.
///
/// Stopping rule (Pavlidis / Jacob's, "about to repeat first move"):
/// after entering the loop we record the very first move taken from start;
/// thereafter, whenever we're back at start AND the next move would equal
/// the first-recorded move, we've completed a full circuit. This handles
/// concave shapes (L, U, …) correctly. A naive "same backtrack on revisit"
/// check fails on those because the trace can revisit start with a
/// different backtrack direction mid-circuit, then run away. A `max_steps` cap is kept
/// as a defense-in-depth net.
pub(crate) fn moore_trace(
    labels: ArrayView2<i64>,
    lbl: i64,
    start_r: usize,
    start_c: usize,
) -> Vec<(usize, usize)> {
    let (h, w) = labels.dim();
    let in_region = |r: i32, c: i32| -> bool {
        r >= 0 && c >= 0 && r < h as i32 && c < w as i32
            && labels[[r as usize, c as usize]] == lbl
    };
    let start = (start_r as i32, start_c as i32);
    let mut out: Vec<(usize, usize)> = vec![(start_r, start_c)];

    // Single-isolated-pixel region: nothing to trace.
    if !MOORE_DIRS.iter().any(|&(dr, dc)| in_region(start.0 + dr, start.1 + dc)) {
        return out;
    }

    let mut current = start;
    // West (guaranteed OOR for a column-major-first start).
    let mut backtrack_dir: usize = 0;

    // Find the first in-region neighbor scanning clockwise from
    // backtrack_dir + 1.
    let scan = |position: (i32, i32), backtrack_dir: usize| -> Option<usize> {
        for k in 1..=8 {
            let idx = (backtrack_dir + k) % 8;
            let (dr, dc) = MOORE_DIRS[idx];
            if in_region(position.0 + dr, position.1 + dc) {
                return Some(idx);
            }
        }
        None
    };

    let first_move = match scan(current, backtrack_dir) {
        Some(i) => i,
        None => return out,
    };
    // Take the first move outside the main loop. Otherwise the Pavlidis
    // stop ("at start AND scan would equal first_move") fires immediately
    // on iteration 0 and we never leave the start pixel.
    let (move_dr, move_dc) = MOORE_DIRS[first_move];
    current = (current.0 + move_dr, current.1 + move_dc);
    backtrack_dir = MOORE_NEW_C_DIR[first_move];
    out.push((current.0 as usize, current.1 as usize));

    // Step bound: each unique (pixel, backtrack_dir) state is visited at
    // most once before the algorithm closes, so 8·h·w is a safe upper
    // bound. Fine-tuning the bound doesn't matter. We should always exit
    // via the natural Pavlidis stop in correct runs.
    let max_steps = h.saturating_mul(w).saturating_mul(8) + 16;
    for _ in 0..max_steps {
        let next_move = match scan(current, backtrack_dir) {
            Some(i) => i,
            None => break,
        };
        // Stop: at start AND about to repeat the first move.
        if current == start && next_move == first_move {
            break;
        }
        let (move_dr, move_dc) = MOORE_DIRS[next_move];
        current = (current.0 + move_dr, current.1 + move_dc);
        backtrack_dir = MOORE_NEW_C_DIR[next_move];
        out.push((current.0 as usize, current.1 as usize));
    }
    out
}

/// Peak table for one segment from its trimmed label image: one row per
/// label, in ascending label order. `spect` is the full-resolution segment
/// and `stimes` / `sfreqs` are its axes.
fn compute_peak_props_from_trim(
    trim_labels: ArrayView2<i64>,
    spect: ArrayView2<f64>,
    stimes: ArrayView1<f64>,
    sfreqs: ArrayView1<f64>,
    segment_num: f64,
) -> SegmentPeaks {
    let (h, w) = trim_labels.dim();
    debug_assert_eq!(spect.dim(), (h, w));
    debug_assert_eq!(stimes.len(), w);
    debug_assert_eq!(sfreqs.len(), h);
    if stimes.len() < 2 || sfreqs.len() < 2 {
        return SegmentPeaks::default();
    }
    let d_time = stimes[1] - stimes[0];
    let d_freq = sfreqs[1] - sfreqs[0];
    let t0 = stimes[0];
    let f0 = sfreqs[0];

    // Column-major iteration to match MATLAB's regionprops scan order, so
    // pixel_values comes out in the same order MATLAB writes PixelValues
    // (column-major linear scan).
    //
    // The first in-region pixel encountered for a given label is its
    // column-major-first pixel (smallest col, then smallest row within
    // that col). That pixel is recorded as `start_r`/`start_c` and used
    // post-loop as the Moore-trace start: West is guaranteed OOR, so the
    // trace is well-formed with West as the initial backtrack.
    //
    // Boundary pixels themselves are NOT collected during this pass.
    // Unordered perimeter pixels can't be plotted as a polyline (column-
    // major scan would produce sawtooth crossing lines). They're emitted
    // post-loop in clockwise traversal order via `moore_trace`.
    let mut map: HashMap<i64, Accum> = HashMap::new();
    for c in 0..w {
        for r in 0..h {
            let lbl = trim_labels[[r, c]];
            if lbl <= 0 {
                continue;
            }
            let v = spect[[r, c]];
            // Skip NaN pixels. Matches MATLAB `Ldata(regions{ii}(~isnan(data(...))))`.
            if v.is_nan() {
                continue;
            }
            let e = map.entry(lbl).or_insert(Accum {
                sum_v: 0.0,
                sum_v_r: 0.0,
                sum_v_c: 0.0,
                min_v: f64::INFINITY,
                max_v: f64::NEG_INFINITY,
                r_min: r,
                r_max: r,
                c_min: c,
                c_max: c,
                pixel_count: 0,
                pixel_values: Vec::new(),
                start_r: r,
                start_c: c,
            });
            e.sum_v += v;
            e.sum_v_r += v * (r as f64);
            e.sum_v_c += v * (c as f64);
            if v < e.min_v {
                e.min_v = v;
            }
            if v > e.max_v {
                e.max_v = v;
            }
            if r < e.r_min {
                e.r_min = r;
            }
            if r > e.r_max {
                e.r_max = r;
            }
            if c < e.c_min {
                e.c_min = c;
            }
            if c > e.c_max {
                e.c_max = c;
            }
            e.pixel_count += 1;
            e.pixel_values.push(v);
        }
    }
    // Sort by label id for determinism.
    let mut keys: Vec<i64> = map.keys().copied().collect();
    keys.sort();

    let mut out = SegmentPeaks::default();
    for k in keys {
        let a = &map[&k];
        // skimage centroid_weighted = Σ v·idx / Σ v in pixel-index space.
        // A region whose values do not sum above zero has no weighted
        // centroid and takes its bounding-box corner.
        let centroid_r = if a.sum_v > 0.0 { a.sum_v_r / a.sum_v } else { a.r_min as f64 };
        let centroid_c = if a.sum_v > 0.0 { a.sum_v_c / a.sum_v } else { a.c_min as f64 };
        let peak_time = centroid_c * d_time + t0;
        let peak_freq = centroid_r * d_freq + f0;
        // N_pixels along each axis = (max - min + 1). Pydynamo uses
        // skimage bbox which is (min, max+1) → (maxc - minc) = N_pixels.
        let n_t = (a.c_max - a.c_min + 1) as f64;
        let n_f = (a.r_max - a.r_min + 1) as f64;
        let duration = n_t * d_time;
        let bandwidth = n_f * d_freq;
        let height = a.max_v - a.min_v;
        let volume = a.sum_v * d_time * d_freq;
        let area = (a.pixel_count as f64) * d_time * d_freq;
        // peakiness = N/(N-1) * (max - mean) / (max - min) over the
        // region's N pixels. A single-pixel spike on a flat floor has
        // mean = min + (max - min)/N, so the raw ratio tops out at
        // 1 - 1/N. The N/(N-1) factor corrects that, and a spike scores
        // exactly 1 at every region size. The smallest possible value is
        // 1/(N-1), which approaches 0 for a large flat plateau. Gain and
        // additive offset both cancel. NaN pixels are excluded above. A
        // flat region (max == min) gives 0/0 = NaN, and so does N = 1.
        // The NaN is passed on for the caller to handle.
        let n_px = a.pixel_count as f64;
        let mean_v = a.sum_v / n_px;
        let peakiness =
            (n_px / (n_px - 1.0)) * (a.max_v - mean_v) / (a.max_v - a.min_v);
        // BoundingBox: (time_tl, freq_tl, width_s, height_Hz), the exact
        // extent of the region's pixels. A pixel at index c is centered on
        // `c * d_time + t0` and covers half a bin to each side, so the
        // box starts half a bin before the center of the first pixel and,
        // with width `n_t * d_time`, ends half a bin after the center of
        // the last one. The same holds along frequency.
        let t_tl = (a.c_min as f64 - 0.5) * d_time + t0;
        let f_tl = (a.r_min as f64 - 0.5) * d_freq + f0;
        let w_s = n_t * d_time;
        let h_hz = n_f * d_freq;

        // Moore-neighbor trace from the column-major-first in-region pixel
        // that produces clockwise-ordered perimeter coords suitable for direct
        // plot()-as-polyline (interleaved t, f).
        let trace = moore_trace(trim_labels, k, a.start_r, a.start_c);
        let mut bxy: Vec<f64> = Vec::with_capacity(trace.len() * 2);
        for (tr, tc) in trace {
            bxy.push(tc as f64 * d_time + t0);
            bxy.push(tr as f64 * d_freq + f0);
        }

        out.peak_time.push(peak_time);
        out.peak_freq.push(peak_freq);
        out.duration.push(duration);
        out.bandwidth.push(bandwidth);
        out.height.push(height);
        out.volume.push(volume);
        out.segment_num.push(segment_num);
        out.bbox.push(t_tl);
        out.bbox.push(f_tl);
        out.bbox.push(w_s);
        out.bbox.push(h_hz);
        out.area.push(area);
        out.peakiness.push(peakiness);
        out.height_data.push(a.pixel_values.clone());
        out.boundaries_xy.push(bxy);
    }
    out
}

/// Run the full pipeline on one (F, T) spectrogram segment with its own
/// stimes. Returns peak stats (no filtering yet) AND the trimmed label
/// image aligned with those peaks: row k of the returned `SegmentPeaks` is
/// label `k + 1` in the label image, and zero pixels are background.
///
/// `segment_num` is a 1-based tag assigned per segment (matches MATLAB).
pub fn extract_tfpeaks_segment(
    spect: ArrayView2<f64>,
    stimes: ArrayView1<f64>,
    sfreqs: ArrayView1<f64>,
    segment_num: f64,
    params: &ExtractParams,
) -> Result<(SegmentPeaks, Array2<i64>), String> {
    let (f, t) = spect.dim();
    if f < 2 || t < 2 {
        return Ok((SegmentPeaks::default(), Array2::<i64>::zeros((f, t))));
    }
    // 1) Downsample.
    let freq_stride = params.downsample_f.max(1);
    let time_stride = params.downsample_t.max(1);
    let seg_lr = stride_downsample(spect, freq_stride, time_stride);
    let (f_lr, t_lr) = seg_lr.dim();
    if f_lr < 2 || t_lr < 2 {
        return Ok((SegmentPeaks::default(), Array2::<i64>::zeros((f, t))));
    }

    // 2) Watershed on -seg_LR, numbered as MATLAB `runWatershed.m` does.
    let neg: Array2<f64> = seg_lr.mapv(|v| -v);
    let ws_i64: Array2<i64> = crate::matlab_watershed::matlab_run_watershed_2d(neg.view());
    if tfpeaks_debug() {
        DBG_WS.fetch_add(count_labels(&ws_i64), Ordering::Relaxed);
    }

    // 3) Merge (interior-only labels).
    let merged_i32 = crate::merge::run(
        ws_i64.view(),
        seg_lr.view(),
        params.merge_thresh,
        params.max_merges,
    )?;
    let merged_i64: Array2<i64> = merged_i32.mapv(|v| v as i64);
    if tfpeaks_debug() {
        DBG_MERGE.fetch_add(count_labels(&merged_i64), Ordering::Relaxed);
    }

    // 4) Border handling, by `expand_labels_distance`:
    //      0   => MATLAB paint-in-label-order (8-connected 1-pixel
    //             dilation, higher labels overwrite lower), as
    //             extractTFPeaks.m + Ldata2graph.m do.
    //      N>0 => skimage-style expand_labels (BFS to distance N, lower
    //             label wins ties). The pydynamo behavior (its default is 5).
    let expanded_lr = if params.expand_labels_distance == 0 {
        matlab_paint_labels_in_order(merged_i64.view())
    } else {
        expand_labels_bfs(merged_i64.view(), params.expand_labels_distance as usize)
    };

    // 5) Resize up to full segment shape (nearest-neighbor).
    let labels_full = if freq_stride > 1 || time_stride > 1 {
        resize_labels_nn(expanded_lr.view(), f, t)
    } else {
        expanded_lr
    };

    // 5b) Pre-trim duration / bandwidth cut (MATLAB `extractTFPeaks.m`).
    //     Drop regions whose bounding-box span, (max_idx - min_idx + 1) * dx,
    //     fails the dur_min / bw_min cut BEFORE trim runs. This both
    //     changes the final peak count (trim creates spurious sub-regions
    //     from jagged bboxes on rejected peaks) and skips unneeded trim work.
    let mut labels_full = labels_full;
    if params.dur_min > 0.0 || params.bw_min > 0.0 {
        /// Inclusive bounding box of one label, in pixel indices.
        struct LabelBounds {
            row_min: usize,
            row_max: usize,
            col_min: usize,
            col_max: usize,
        }
        let d_time = stimes[1] - stimes[0];
        let d_freq = sfreqs[1] - sfreqs[0];
        let (h, w) = labels_full.dim();
        let mut bounds_by_label: HashMap<i64, LabelBounds> = HashMap::new();
        for r in 0..h {
            for c in 0..w {
                let lbl = labels_full[[r, c]];
                if lbl <= 0 {
                    continue;
                }
                let bounds = bounds_by_label.entry(lbl).or_insert(LabelBounds {
                    row_min: r,
                    row_max: r,
                    col_min: c,
                    col_max: c,
                });
                if r < bounds.row_min {
                    bounds.row_min = r;
                }
                if r > bounds.row_max {
                    bounds.row_max = r;
                }
                if c < bounds.col_min {
                    bounds.col_min = c;
                }
                if c > bounds.col_max {
                    bounds.col_max = c;
                }
            }
        }
        let mut labels_to_drop: std::collections::HashSet<i64> = std::collections::HashSet::new();
        for (lbl, bounds) in &bounds_by_label {
            // Span = N_pixels * dx = (max - min + 1) * dx, with inclusive
            // row_max / col_max. Same form as the MATLAB `extractTFPeaks.m`
            // cut.
            let pre_dur = (bounds.col_max as f64 - bounds.col_min as f64 + 1.0) * d_time;
            let pre_bw = (bounds.row_max as f64 - bounds.row_min as f64 + 1.0) * d_freq;
            if !(pre_dur > params.dur_min && pre_bw > params.bw_min) {
                labels_to_drop.insert(*lbl);
            }
        }
        if !labels_to_drop.is_empty() {
            for r in 0..h {
                for c in 0..w {
                    let lbl = labels_full[[r, c]];
                    if labels_to_drop.contains(&lbl) {
                        labels_full[[r, c]] = 0;
                    }
                }
            }
        }
    }

    if tfpeaks_debug() {
        DBG_CUT.fetch_add(count_labels(&labels_full), Ordering::Relaxed);
    }

    // 6) Trim on the full-resolution spect.
    let labels_full_i32: Array2<i32> = labels_full.mapv(|v| v as i32);
    let shift_val = if params.trim_shift_val.is_nan() {
        // Smallest value in the segment. `f64::min` skips NaN.
        spect.iter().cloned().fold(f64::INFINITY, f64::min)
    } else {
        params.trim_shift_val
    };
    let trimmed = crate::trim::trim_all_regions(
        labels_full_i32.view(), spect, params.trim_vol_thresh, shift_val,
    )?;
    let trimmed_i64: Array2<i64> = trimmed.mapv(|v| v as i64);
    if tfpeaks_debug() {
        DBG_TRIM.fetch_add(count_labels(&trimmed_i64), Ordering::Relaxed);
    }

    // 7) Peak properties.
    let peaks = compute_peak_props_from_trim(
        trimmed_i64.view(), spect, stimes, sfreqs, segment_num,
    );

    // 8) Build the per-segment label image whose labels align with the
    //    returned `peaks` rows. `compute_peak_props_from_trim` iterates
    //    over unique label IDs in sorted order, so the k-th peak in the
    //    output corresponds to the k-th smallest label in `trimmed_i64`.
    //    Build a remap old_id -> new_id = rank+1 (1-based), zero stays 0.
    // Distinct positive labels in ascending order.
    let mut seen = std::collections::BTreeSet::new();
    for &v in trimmed_i64.iter() {
        if v > 0 {
            seen.insert(v);
        }
    }
    let ids: Vec<i64> = seen.into_iter().collect();
    debug_assert_eq!(ids.len(), peaks.len());
    let mut remap: HashMap<i64, i64> = HashMap::with_capacity(ids.len());
    for (k, old) in ids.iter().enumerate() {
        remap.insert(*old, (k as i64) + 1);
    }
    let labels_out = trimmed_i64.mapv(|v| {
        if v <= 0 {
            0
        } else {
            *remap.get(&v).unwrap_or(&0)
        }
    });

    Ok((peaks, labels_out))
}

/// Full top-level: split spect into seg_time segments, extract peaks from
/// each (in parallel), concatenate, then apply filterStatsTable (dur / bw /
/// peak_freq / height-dB cuts).
///
/// `cancel`, when supplied, is polled at the start of each segment task: if
/// it returns `true` the call early-returns `Err` without running the
/// remaining segments' watershed. This keeps a host's Stop control
/// responsive. A canceled call returns within about one segment instead of
/// the whole multi-segment extract. With `cancel: None` every segment runs.
///
/// `baseline`, when given, has one value per frequency and each spectrogram
/// row is divided by its value first (a zero or non-finite value leaves the
/// row as it is). `progress_cb(done, total)` is called after each segment.
/// Returns the filtered peak table and a full-size label image in which
/// peak row k carries label k + 1.
pub fn extract_tfpeaks(
    spect: ArrayView2<f64>,
    stimes: ArrayView1<f64>,
    sfreqs: ArrayView1<f64>,
    baseline: Option<ArrayView1<f64>>,
    params: &ExtractParams,
    progress_cb: Option<&(dyn Fn(u32, u32) + Sync)>,
    cancel: Option<&(dyn Fn() -> bool + Sync)>,
) -> Result<(SegmentPeaks, Array2<i64>), String> {
    let (f, t) = spect.dim();
    if stimes.len() != t {
        return Err(format!(
            "spect time dimension {} does not match stimes length {}",
            t,
            stimes.len()
        ));
    }
    if sfreqs.len() != f {
        return Err(format!(
            "spect frequency dimension {} does not match sfreqs length {}",
            f,
            sfreqs.len()
        ));
    }
    if f == 0 || t == 0 || stimes.len() < 2 {
        return Ok((SegmentPeaks::default(), Array2::<i64>::zeros((f, t))));
    }
    // Apply baseline once up front (broadcast divide along rows).
    let spect_owned: Array2<f64> = if let Some(bl) = baseline {
        if bl.len() != f {
            return Err(format!(
                "baseline length {} does not match n_freqs {}",
                bl.len(), f
            ));
        }
        let mut m = spect.to_owned();
        for r in 0..f {
            let denom = bl[r];
            if !denom.is_finite() || denom == 0.0 {
                continue;
            }
            for c in 0..t {
                m[[r, c]] /= denom;
            }
        }
        m
    } else {
        spect.to_owned()
    };

    // MATLAB-compatible segmentation.
    let dt = stimes[1] - stimes[0];
    let seg_time = if params.seg_time > 0.0 {
        params.seg_time
    } else {
        DEFAULT_SEG_TIME_S
    };
    let max_dx = (seg_time / dt).floor() as usize;
    if max_dx < 2 {
        return Err("seg_time too small relative to dt (max_dx < 2)".to_string());
    }
    let n_segs = ((t as f64) / (max_dx as f64)).ceil() as usize;
    let new_dx = (((t as f64) / (n_segs as f64)).ceil() as usize).max(1);

    /// One segment: its 1-based number and its column range `[start, end)`.
    #[derive(Clone, Copy)]
    struct SegmentBounds {
        segment_num: usize,
        start: usize,
        end: usize,
    }
    let mut seg_bounds: Vec<SegmentBounds> = Vec::with_capacity(n_segs);
    for ii in 0..n_segs {
        let start = ii * new_dx;
        let end = (start + new_dx).min(t);
        // A segment narrower than 2 columns cannot hold a peak and is skipped.
        if end.saturating_sub(start) < 2 {
            continue;
        }
        seg_bounds.push(SegmentBounds {
            segment_num: ii + 1,
            start,
            end,
        });
    }

    // Parallel extract. Collect per-segment results preserving order.
    let n_segs_total = seg_bounds.len() as u32;
    if tfpeaks_debug() {
        DBG_WS.store(0, Ordering::Relaxed);
        DBG_MERGE.store(0, Ordering::Relaxed);
        DBG_CUT.store(0, Ordering::Relaxed);
        DBG_TRIM.store(0, Ordering::Relaxed);
    }
    let done = std::sync::atomic::AtomicU32::new(0);
    let results: Vec<Result<(SegmentPeaks, Array2<i64>), String>> = seg_bounds
        .par_iter()
        .map(|segment| {
            // Bail before a segment's watershed if the run was canceled. This
            // keeps a host's Stop responsive within ~one segment instead of
            // the whole multi-segment extract.
            if let Some(is_canceled) = cancel {
                if is_canceled() {
                    return Err("cancelled".to_string());
                }
            }
            // Make the segment spect contiguous. Trim / merge require it.
            let sub_spect = spect_owned
                .slice(ndarray::s![.., segment.start..segment.end])
                .to_owned();
            let sub_times = stimes
                .slice(ndarray::s![segment.start..segment.end])
                .to_owned();
            let segment_result = extract_tfpeaks_segment(
                sub_spect.view(),
                sub_times.view(),
                sfreqs,
                segment.segment_num as f64,
                params,
            );
            if let Some(report_progress) = progress_cb {
                let n_done = done.fetch_add(1, Ordering::SeqCst) + 1;
                report_progress(n_done, n_segs_total);
            }
            segment_result
        })
        .collect();

    if tfpeaks_debug() {
        eprintln!(
            "[tfdbg]   extract_tfpeaks: {} segs  watershed={} -> merge={} -> dur/bw-cut={} -> trim={}  (regions, summed over segments)",
            n_segs_total,
            DBG_WS.load(Ordering::Relaxed),
            DBG_MERGE.load(Ordering::Relaxed),
            DBG_CUT.load(Ordering::Relaxed),
            DBG_TRIM.load(Ordering::Relaxed),
        );
    }

    // Stitch per-segment labels into a full (F, T) image with running
    // label offset so that peak row k in the concatenated SegmentPeaks
    // maps to label k+1 in the global image.
    let mut all_peaks = SegmentPeaks::default();
    let mut labels_full = Array2::<i64>::zeros((f, t));
    let mut offset: i64 = 0;
    for (segment_result, segment) in results.into_iter().zip(seg_bounds.iter()) {
        match segment_result {
            Ok((segment_peaks, segment_labels)) => {
                let n_here = segment_peaks.len() as i64;
                // Write with offset into the global image.
                let (n_rows, n_cols) = segment_labels.dim();
                debug_assert_eq!(n_rows, f);
                debug_assert_eq!(n_cols, segment.end - segment.start);
                for r in 0..n_rows {
                    for c in 0..n_cols {
                        let v = segment_labels[[r, c]];
                        if v > 0 {
                            labels_full[[r, segment.start + c]] = v + offset;
                        }
                    }
                }
                all_peaks.extend(segment_peaks);
                offset += n_here;
            }
            Err(e) => return Err(e),
        }
    }

    // filterStatsTable: Duration ∈ (dur_min, dur_max), Bandwidth ∈ (bw_min, bw_max),
    // PeakFrequency ∈ (freq_min, freq_max), pow2db(Height) > ht_db_min.
    let keep = filter_indices(&all_peaks, params);

    // Build a remap from old global label id -> new label id (1..n_kept).
    // Peaks not in `keep` -> 0. Indexed by the old id (1-based).
    let mut remap_filter: Vec<i64> = vec![0; all_peaks.len() + 1];
    for (new_idx, &old_idx) in keep.iter().enumerate() {
        // old label id = old_idx + 1 (1-based global id)
        remap_filter[old_idx + 1] = (new_idx as i64) + 1;
    }
    for v in labels_full.iter_mut() {
        if *v > 0 {
            let old = *v as usize;
            if old < remap_filter.len() {
                *v = remap_filter[old];
            } else {
                *v = 0;
            }
        }
    }

    Ok((take_indices(&all_peaks, &keep), labels_full))
}

/// Row indices of the peaks that pass MATLAB `filterStatsTable`: Duration,
/// Bandwidth and PeakFrequency strictly inside their limits, and Height in
/// dB above `ht_db_min`.
fn filter_indices(
    p: &SegmentPeaks,
    params: &ExtractParams,
) -> Vec<usize> {
    let mut out = Vec::new();
    for i in 0..p.len() {
        let dur = p.duration[i];
        let bw = p.bandwidth[i];
        let pf = p.peak_freq[i];
        let h = p.height[i];
        let h_db = if h > 0.0 { 10.0 * h.log10() } else { f64::NAN };
        // Duration and Bandwidth are inclusive bounding-box spans
        // (N_pixels * dx). MATLAB `extractTFPeaks.m` and
        // `filterStatsTable.m` compare them to the limits directly, and
        // so does this. pydynamo's `extract.py` uses the (N - 1) * dx
        // form, which is stricter by one bin.
        let dur_ok = dur > params.dur_min && dur < params.dur_max;
        let bw_ok = bw > params.bw_min && bw < params.bw_max;
        let pf_ok = pf > params.freq_min && pf < params.freq_max;
        let ht_ok = h_db.is_finite() && h_db > params.ht_db_min;
        if dur_ok && bw_ok && pf_ok && ht_ok {
            out.push(i);
        }
    }
    out
}

/// Copy of the peak table holding only the rows in `idx`, in that order.
fn take_indices(p: &SegmentPeaks, idx: &[usize]) -> SegmentPeaks {
    let mut out = SegmentPeaks::default();
    out.peak_time.reserve(idx.len());
    out.peak_freq.reserve(idx.len());
    out.duration.reserve(idx.len());
    out.bandwidth.reserve(idx.len());
    out.height.reserve(idx.len());
    out.volume.reserve(idx.len());
    out.segment_num.reserve(idx.len());
    out.bbox.reserve(idx.len() * 4);
    out.area.reserve(idx.len());
    out.peakiness.reserve(idx.len());
    out.height_data.reserve(idx.len());
    out.boundaries_xy.reserve(idx.len());
    for &i in idx {
        out.peak_time.push(p.peak_time[i]);
        out.peak_freq.push(p.peak_freq[i]);
        out.duration.push(p.duration[i]);
        out.bandwidth.push(p.bandwidth[i]);
        out.height.push(p.height[i]);
        out.volume.push(p.volume[i]);
        out.segment_num.push(p.segment_num[i]);
        out.bbox.push(p.bbox[i * 4]);
        out.bbox.push(p.bbox[i * 4 + 1]);
        out.bbox.push(p.bbox[i * 4 + 2]);
        out.bbox.push(p.bbox[i * 4 + 3]);
        out.area.push(p.area[i]);
        out.peakiness.push(p.peakiness[i]);
        out.height_data.push(p.height_data[i].clone());
        out.boundaries_xy.push(p.boundaries_xy[i].clone());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn stride_downsample_basic() {
        let a = Array2::from_shape_vec((4, 4), (0..16).map(|x| x as f64).collect()).unwrap();
        let out = stride_downsample(a.view(), 2, 2);
        assert_eq!(out.dim(), (2, 2));
        assert_eq!(out[[0, 0]], 0.0);
        assert_eq!(out[[0, 1]], 2.0);
        assert_eq!(out[[1, 0]], 8.0);
        assert_eq!(out[[1, 1]], 10.0);
    }

    #[test]
    fn resize_labels_nn_integer_factor() {
        let a = Array2::from_shape_vec((2, 2), vec![1_i64, 2, 3, 4]).unwrap();
        let out = resize_labels_nn(a.view(), 4, 4);
        assert_eq!(out.dim(), (4, 4));
        // Nearest neighbor with integer factor 2 → each input pixel fills a 2×2 block.
        assert_eq!(out[[0, 0]], 1);
        assert_eq!(out[[0, 3]], 2);
        assert_eq!(out[[3, 0]], 3);
        assert_eq!(out[[3, 3]], 4);
    }

    /// Source index (1-based, as MATLAB prints it) that each output column
    /// reads when a single row `1..=in_len` is resized to `out_len` columns.
    fn resized_row(in_len: usize, out_len: usize) -> Vec<i64> {
        let row: Vec<i64> = (1..=in_len as i64).collect();
        let labels = Array2::from_shape_vec((1, in_len), row).unwrap();
        let out = resize_labels_nn(labels.view(), 1, out_len);
        out.row(0).to_vec()
    }

    /// Expected values are the output of MATLAB R2025b
    /// `imresize(1:in_len, [1 out_len], 'nearest')`.
    #[test]
    fn resize_labels_nn_matches_imresize_on_odd_lengths() {
        assert_eq!(resized_row(4, 7), vec![1, 1, 2, 3, 3, 4, 4]);
        assert_eq!(resized_row(6, 11), vec![1, 1, 2, 2, 3, 4, 4, 5, 5, 6, 6]);

        // A 30 s segment of the default spectrogram is 571 columns wide and
        // its 2x downsampled image is 286. Column 286 falls halfway between
        // input columns 143 and 144, and imresize reads 144.
        let segment = resized_row(286, 571);
        assert_eq!(&segment[283..288], &[142, 143, 144, 144, 145]);
        assert_eq!(segment.iter().sum::<i64>(), 81939);

        // Here the halfway column goes the other way: imresize reads the
        // lower input column.
        assert_eq!(resized_row(12, 23)[11], 6);
        assert_eq!(resized_row(34, 67)[33], 17);
        assert_eq!(resized_row(40, 79)[39], 20);
        assert_eq!(resized_row(42, 83)[41], 21);
    }

    #[test]
    fn expand_labels_fills_zero_gap() {
        let mut a = Array2::<i64>::zeros((3, 5));
        a[[1, 0]] = 1;
        a[[1, 4]] = 2;
        let out = expand_labels_bfs(a.view(), 2);
        // Pixels within distance 2 of labeled sources get filled.
        assert_eq!(out[[1, 0]], 1);
        assert_eq!(out[[1, 1]], 1);
        assert_eq!(out[[1, 3]], 2);
        assert_eq!(out[[1, 4]], 2);
    }

    #[test]
    fn moore_trace_3x3_square() {
        // 3x3 solid square at rows 1..=3, cols 1..=3 in a 5x5 image.
        // Perimeter: 8 pixels clockwise from top-left, then a closing
        // repeat of (1,1) so plot()-as-polyline draws the closing edge
        // for free.
        let mut lab = Array2::<i64>::zeros((5, 5));
        for r in 1..=3 {
            for c in 1..=3 {
                lab[[r, c]] = 7;
            }
        }
        let t = moore_trace(lab.view(), 7, 1, 1);
        assert_eq!(t,
            vec![(1,1),(1,2),(1,3),(2,3),(3,3),(3,2),(3,1),(2,1),(1,1)],
            "Moore-trace of 3x3 square should be 8 clockwise + closing repeat"
        );
        // Successive pixels are 8-adjacent.
        for w in t.windows(2) {
            let a = w[0];
            let b = w[1];
            let dr = (a.0 as i32 - b.0 as i32).abs();
            let dc = (a.1 as i32 - b.1 as i32).abs();
            assert!(dr <= 1 && dc <= 1 && (dr + dc) >= 1,
                "non-adjacent step {:?} -> {:?}", a, b);
        }
    }

    #[test]
    fn moore_trace_single_pixel() {
        let mut lab = Array2::<i64>::zeros((3, 3));
        lab[[1, 1]] = 1;
        let t = moore_trace(lab.view(), 1, 1, 1);
        assert_eq!(t, vec![(1, 1)]);
    }

    #[test]
    fn moore_trace_l_shape_terminates() {
        // L-shape (5 pixels, concave). Naive same-backtrack-on-revisit
        // stop diverges here; Pavlidis-first-move stop closes the trace.
        //   . X X
        //   . X .
        //   X X .
        let mut lab = Array2::<i64>::zeros((3, 3));
        lab[[0, 1]] = 9;
        lab[[0, 2]] = 9;
        lab[[1, 1]] = 9;
        lab[[2, 0]] = 9;
        lab[[2, 1]] = 9;
        // Column-major first pixel: scan column 0 → (2, 0).
        let t = moore_trace(lab.view(), 9, 2, 0);
        // Trace must be small and end at start (closing edge for plot).
        assert!(
            t.len() <= 16,
            "L-shape trace exploded ({} pixels, expected <16)",
            t.len()
        );
        assert_eq!(t.first().copied(), Some((2, 0)));
        // Each step must be 8-adjacent.
        for w in t.windows(2) {
            let a = w[0];
            let b = w[1];
            let dr = (a.0 as i32 - b.0 as i32).abs();
            let dc = (a.1 as i32 - b.1 as i32).abs();
            assert!(
                dr <= 1 && dc <= 1 && (dr + dc) >= 1,
                "non-adjacent step {:?} -> {:?} in trace {:?}",
                a, b, t
            );
        }
    }

    #[test]
    fn moore_trace_u_shape_terminates() {
        // U-shape (concave with a bigger notch).
        //   X . X
        //   X . X
        //   X X X
        let mut lab = Array2::<i64>::zeros((3, 3));
        for r in 0..3 {
            lab[[r, 0]] = 5;
            lab[[r, 2]] = 5;
        }
        for c in 0..3 {
            lab[[2, c]] = 5;
        }
        // Column-major first: column 0 has rows 0,1,2 in region. First = (0,0).
        let t = moore_trace(lab.view(), 5, 0, 0);
        assert!(
            t.len() <= 32,
            "U-shape trace exploded ({} pixels, expected <32)",
            t.len()
        );
        for w in t.windows(2) {
            let a = w[0];
            let b = w[1];
            let dr = (a.0 as i32 - b.0 as i32).abs();
            let dc = (a.1 as i32 - b.1 as i32).abs();
            assert!(dr <= 1 && dc <= 1 && (dr + dc) >= 1);
        }
    }

    #[test]
    fn filter_basic() {
        let mut p = SegmentPeaks::default();
        p.peak_time.push(1.0);
        p.peak_freq.push(10.0);
        p.duration.push(1.0);
        p.bandwidth.push(5.0);
        p.height.push(100.0);
        p.volume.push(1.0);
        p.segment_num.push(1.0);
        p.bbox.extend_from_slice(&[0.0, 0.0, 0.0, 0.0]);
        p.area.push(5.0);
        // N/(N-1) * (max - mean) / (max - min), unitless.
        p.peakiness.push(0.5);
        p.height_data.push(vec![1.0, 2.0]);
        p.boundaries_xy.push(vec![0.0, 10.0]);
        let params = ExtractParams {
            seg_time: 30.0, downsample_f: 1, downsample_t: 1,
            merge_thresh: 8.0, max_merges: f64::INFINITY, trim_vol_thresh: 0.8,
            trim_shift_val: f64::NAN,
            dur_min: 0.5, dur_max: 5.0,
            bw_min: 2.0, bw_max: 15.0,
            freq_min: 0.0, freq_max: 40.0,
            ht_db_min: 7.0, expand_labels_distance: 5,
        };
        let kept = filter_indices(&p, &params);
        // Height=100 → pow2db = 20 dB > 7 dB ✓; dur 1.0 > dur_min 0.5 ✓;
        // bw 5 ∈ (2, 15) ✓.
        assert_eq!(kept, vec![0usize]);

        // Out-of-range duration rejected.
        p.duration[0] = 10.0;
        let kept = filter_indices(&p, &params);
        assert!(kept.is_empty());
    }

    #[test]
    fn extract_tfpeaks_rejects_mismatched_axis_lengths() {
        let spect = Array2::<f64>::zeros((3, 4));
        let stimes = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0]);
        let sfreqs = Array1::from_vec(vec![0.0, 1.0, 2.0]);
        let params = ExtractParams {
            seg_time: 30.0,
            downsample_f: 1, downsample_t: 1,
            merge_thresh: 8.0, max_merges: f64::INFINITY,
            trim_vol_thresh: 0.8, trim_shift_val: f64::NAN,
            dur_min: 0.0, dur_max: f64::INFINITY,
            bw_min: 0.0, bw_max: f64::INFINITY,
            freq_min: 0.0, freq_max: f64::INFINITY,
            ht_db_min: f64::NEG_INFINITY,
            expand_labels_distance: 5,
        };

        let time_err = match extract_tfpeaks(
            spect.view(),
            stimes.slice(ndarray::s![..3]),
            sfreqs.view(),
            None,
            &params,
            None,
            None,
        ) {
            Err(err) => err,
            Ok(_) => panic!("mismatched stimes length was accepted"),
        };
        assert!(time_err.contains("stimes"));

        let freq_err = match extract_tfpeaks(
            spect.view(),
            stimes.view(),
            sfreqs.slice(ndarray::s![..1]),
            None,
            &params,
            None,
            None,
        ) {
            Err(err) => err,
            Ok(_) => panic!("mismatched sfreqs length was accepted"),
        };
        assert!(freq_err.contains("sfreqs"));
    }

    /// Synthetic 2-segment extract: verify labels shape matches spect,
    /// that non-zero labels are 1..n_peaks contiguously, that each non-zero
    /// pixel points at a valid peak row, and that post-filter surviving
    /// labels remain contiguous 1..n_kept.
    #[test]
    fn extract_tfpeaks_labels_shape_and_stitching() {
        // Construct a 2-segment spect: 32 freq bins, 600 time bins at
        // dt=0.1 s → T*dt = 60 s. With seg_time = 30 s the pipeline
        // produces two segments.
        let n_f = 32usize;
        let n_t = 600usize;
        let dt = 0.1_f64;
        let df = 0.5_f64;
        let mut spect = Array2::<f64>::zeros((n_f, n_t));
        // Two Gaussian blobs: one in segment 1 (t around col 100), one in
        // segment 2 (t around col 450). Both high enough in freq and wide
        // enough to survive duration / bandwidth cuts.
        let blobs = [(16usize, 100usize), (16usize, 450usize)];
        for (fc, tc) in &blobs {
            for r in 0..n_f {
                for c in 0..n_t {
                    let dr = r as f64 - *fc as f64;
                    let dc = c as f64 - *tc as f64;
                    let g = (-(dr * dr) / (2.0 * 3.0 * 3.0)
                        - (dc * dc) / (2.0 * 8.0 * 8.0))
                        .exp();
                    spect[[r, c]] += 100.0 * g;
                }
            }
        }
        // Small positive floor so pow2db is finite everywhere.
        for v in spect.iter_mut() {
            *v += 0.1;
        }

        let stimes: Array1<f64> = Array1::from_iter((0..n_t).map(|i| i as f64 * dt));
        let sfreqs: Array1<f64> = Array1::from_iter((0..n_f).map(|i| i as f64 * df));

        let params = ExtractParams {
            seg_time: 30.0,
            downsample_f: 1, downsample_t: 1,
            merge_thresh: 8.0, max_merges: f64::INFINITY,
            trim_vol_thresh: 0.8, trim_shift_val: f64::NAN,
            dur_min: 0.0, dur_max: f64::INFINITY,
            bw_min: 0.0, bw_max: f64::INFINITY,
            freq_min: 0.0, freq_max: f64::INFINITY,
            ht_db_min: f64::NEG_INFINITY,
            expand_labels_distance: 5,
        };

        let (peaks, labels) = extract_tfpeaks(
            spect.view(), stimes.view(), sfreqs.view(), None, &params, None, None,
        ).expect("extract_tfpeaks failed");

        // Shape matches the input spect.
        assert_eq!(labels.dim(), (n_f, n_t));
        // At least the two planted peaks should survive.
        assert!(peaks.len() >= 2,
            "expected >= 2 peaks, got {}", peaks.len());

        // Non-zero labels form a 1..n_peaks contiguous set.
        let mut seen: std::collections::BTreeSet<i64> =
            std::collections::BTreeSet::new();
        for &v in labels.iter() {
            if v > 0 {
                assert!(v as usize <= peaks.len(),
                    "label {} out of range (n_peaks = {})", v, peaks.len());
                seen.insert(v);
            }
        }
        // After filtering, surviving labels are dense 1..=n_peaks.
        let expected: std::collections::BTreeSet<i64> =
            (1..=peaks.len() as i64).collect();
        assert_eq!(seen, expected,
            "label set {:?} != expected {:?}", seen, expected);

        // Each planted blob should be associated with some non-zero label
        // at its center, with the peak time near the planted column.
        for (fc, tc) in &blobs {
            let lbl = labels[[*fc, *tc]];
            assert!(lbl > 0, "blob at ({},{}) lost its label", fc, tc);
            let pt = peaks.peak_time[(lbl - 1) as usize];
            let expected_t = (*tc as f64) * dt;
            assert!((pt - expected_t).abs() < 2.0,
                "peak_time {} far from planted {}", pt, expected_t);
        }

        // Stitching sanity: pixels in columns [0..n_t/2) with a label
        // should point at peaks whose segment_num == 1.0, and the mirrored
        // half at segment_num == 2.0. This also verifies the running
        // label-id offset across segments.
        for r in 0..n_f {
            for c in 0..n_t {
                let v = labels[[r, c]];
                if v <= 0 {
                    continue;
                }
                let idx = (v - 1) as usize;
                let seg = peaks.segment_num[idx];
                let boundary = n_t / 2;
                if c < boundary - 1 {
                    assert_eq!(seg, 1.0,
                        "label {} in seg-1 col {} has segment_num {}",
                        v, c, seg);
                } else if c > boundary {
                    assert_eq!(seg, 2.0,
                        "label {} in seg-2 col {} has segment_num {}",
                        v, c, seg);
                }
                // Near the exact boundary, either segment is acceptable
                // (resize-nearest may straddle the cut).
            }
        }
    }

    /// A cancel callback that always fires makes `extract_tfpeaks` return
    /// `Err` instead of running every segment's watershed. This is what
    /// keeps a host's Stop control responsive mid-extract.
    #[test]
    fn extract_tfpeaks_cancels() {
        // dt = 0.1 s, T = 60 s → two segments at seg_time = 30 s.
        let n_f = 16usize;
        let n_t = 600usize;
        let spect = Array2::<f64>::from_elem((n_f, n_t), 1.0);
        let stimes: Array1<f64> = Array1::from_iter((0..n_t).map(|i| i as f64 * 0.1));
        let sfreqs: Array1<f64> = Array1::from_iter((0..n_f).map(|i| i as f64 * 0.5));
        let params = ExtractParams {
            seg_time: 30.0,
            downsample_f: 1, downsample_t: 1,
            merge_thresh: 8.0, max_merges: f64::INFINITY,
            trim_vol_thresh: 0.8, trim_shift_val: f64::NAN,
            dur_min: 0.0, dur_max: f64::INFINITY,
            bw_min: 0.0, bw_max: f64::INFINITY,
            freq_min: 0.0, freq_max: f64::INFINITY,
            ht_db_min: f64::NEG_INFINITY,
            expand_labels_distance: 5,
        };
        let always_cancel: &(dyn Fn() -> bool + Sync) = &|| true;
        let r = extract_tfpeaks(
            spect.view(), stimes.view(), sfreqs.view(), None, &params,
            None, Some(always_cancel),
        );
        assert!(r.is_err(), "a canceled extract_tfpeaks must return Err");
    }
}
