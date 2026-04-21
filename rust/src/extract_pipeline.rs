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
//!   - `Height    = max(v) - min(v)`  over label pixels (linear; pow2db'ed in filter)
//!   - `Volume    = sum(v) * d_time * d_freq`

use ndarray::{Array2, ArrayView1, ArrayView2};
use rayon::prelude::*;
use std::collections::{HashMap, VecDeque};

/// Per-segment peak table (one row per surviving label).
#[derive(Default)]
pub struct SegmentPeaks {
    pub peak_time: Vec<f64>,
    pub peak_freq: Vec<f64>,
    pub duration: Vec<f64>,
    pub bandwidth: Vec<f64>,
    pub height: Vec<f64>,
    pub volume: Vec<f64>,
    pub segment_num: Vec<f64>,
    pub bbox: Vec<f64>, // 4 * n (t_tl, f_tl, width_s, height_Hz) in pydynamo format
}

impl SegmentPeaks {
    pub fn len(&self) -> usize { self.peak_time.len() }
    pub fn is_empty(&self) -> bool { self.peak_time.is_empty() }

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
    }
}

/// Parameters for `extract_tfpeaks`.
#[derive(Clone, Copy)]
pub struct ExtractParams {
    pub seg_time: f64,
    pub downsample_f: usize,
    pub downsample_t: usize,
    pub merge_thresh: f64,
    pub max_merges: f64,
    pub trim_vol_thresh: f64,
    pub trim_shift_val: f64, // NaN → min(spect) per segment
    pub dur_min: f64,
    pub dur_max: f64,
    pub bw_min: f64,
    pub bw_max: f64,
    pub freq_min: f64,
    pub freq_max: f64,
    pub ht_db_min: f64,
    /// Distance for expand_labels(). 0 = skip (MATLAB-native behavior,
    /// labels keep 0 on watershed lines). 5 = pydynamo default
    /// (skimage-style fill so regions touch directly).
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
    let oh = (h + f_stride - 1) / f_stride;
    let ow = (w + t_stride - 1) / t_stride;
    let mut out = Array2::<f64>::zeros((oh, ow));
    for i in 0..oh {
        let si = i * f_stride;
        for j in 0..ow {
            let sj = j * t_stride;
            out[[i, j]] = arr[[si, sj]];
        }
    }
    out
}

/// Resize a label image up by nearest-neighbor, matching
/// `skimage.transform.resize(..., order=0, preserve_range=True)`.
/// `output_shape = (oh, ow)`. skimage uses
///     src_y = floor((y + 0.5) * ih / oh - 0.5 + eps)  [rounded]
/// but for integer factors this reduces to `y // factor_y`, which is what
/// we need when `oh = ih * f_f` or `oh = ih * f_f + k` for small k.
/// We implement the exact skimage mapping for correctness on non-integer
/// ratios.
fn resize_labels_nn(labels: ArrayView2<i64>, oh: usize, ow: usize) -> Array2<i64> {
    let (ih, iw) = labels.dim();
    if ih == oh && iw == ow {
        return labels.to_owned();
    }
    let mut out = Array2::<i64>::zeros((oh, ow));
    // skimage resize order=0: nearest neighbor with the sampling grid
    // col = round((j + 0.5) * iw / ow - 0.5).
    let map_coord = |j: usize, out_n: usize, in_n: usize| -> usize {
        if out_n == 0 || in_n == 0 { return 0; }
        let src = (j as f64 + 0.5) * (in_n as f64) / (out_n as f64) - 0.5;
        // skimage.transform.resize(order=0) uses numpy-style "round half
        // to even" (banker's rounding). Matches the pydynamo reference.
        let src = src.round_ties_even() as i64;
        src.clamp(0, in_n as i64 - 1) as usize
    };
    for i in 0..oh {
        let si = map_coord(i, oh, ih);
        for j in 0..ow {
            let sj = map_coord(j, ow, iw);
            out[[i, j]] = labels[[si, sj]];
        }
    }
    out
}

/// `skimage.segmentation.expand_labels(labels, distance=d)`: for every
/// pixel with label 0 within Euclidean distance `d` of some labeled pixel,
/// fill with the nearest label. Ties are broken by the lowest label id.
///
/// Implemented via multi-source BFS (4-connectivity, stops after `d` steps).
/// This approximates Euclidean distance but matches the behaviour needed
/// to plug the watershed 0-line — for distance=5 it is effectively
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
    let mut q: VecDeque<(usize, usize)> = VecDeque::new();
    for r in 0..h {
        for c in 0..w {
            if out[[r, c]] != 0 {
                dist[r * w + c] = 0;
                q.push_back((r, c));
            }
        }
    }
    let dmax = distance as u32;
    while let Some((r, c)) = q.pop_front() {
        let d_here = dist[r * w + c];
        if d_here >= dmax { continue; }
        let my_lbl = out[[r, c]];
        let d_new = d_here + 1;
        let neigh = [
            (r.wrapping_sub(1), c),
            (r + 1, c),
            (r, c.wrapping_sub(1)),
            (r, c + 1),
        ];
        for (nr, nc) in neigh {
            if nr >= h || nc >= w { continue; }
            let idx = nr * w + nc;
            if dist[idx] > d_new {
                // first visit (or we've found a closer source)
                if out[[nr, nc]] == 0 {
                    dist[idx] = d_new;
                    out[[nr, nc]] = my_lbl;
                    q.push_back((nr, nc));
                }
            } else if dist[idx] == d_new {
                // Tie → prefer lower label (stable, matches "first reached")
                let existing = out[[nr, nc]];
                if existing != 0 && my_lbl != 0 && my_lbl < existing {
                    out[[nr, nc]] = my_lbl;
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
}

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

    let mut map: HashMap<i64, Accum> = HashMap::new();
    for r in 0..h {
        for c in 0..w {
            let lbl = trim_labels[[r, c]];
            if lbl <= 0 { continue; }
            let v = spect[[r, c]];
            let e = map.entry(lbl).or_insert(Accum {
                sum_v: 0.0, sum_v_r: 0.0, sum_v_c: 0.0,
                min_v: f64::INFINITY, max_v: f64::NEG_INFINITY,
                r_min: r, r_max: r, c_min: c, c_max: c,
            });
            e.sum_v += v;
            e.sum_v_r += v * (r as f64);
            e.sum_v_c += v * (c as f64);
            if v < e.min_v { e.min_v = v; }
            if v > e.max_v { e.max_v = v; }
            if r < e.r_min { e.r_min = r; }
            if r > e.r_max { e.r_max = r; }
            if c < e.c_min { e.c_min = c; }
            if c > e.c_max { e.c_max = c; }
        }
    }
    // Sort by label id for determinism.
    let mut keys: Vec<i64> = map.keys().copied().collect();
    keys.sort();

    let mut out = SegmentPeaks::default();
    for k in keys {
        let a = &map[&k];
        // skimage centroid_weighted = Σ v·idx / Σ v in pixel-index space.
        let centroid_r = if a.sum_v > 0.0 { a.sum_v_r / a.sum_v } else { a.r_min as f64 };
        let centroid_c = if a.sum_v > 0.0 { a.sum_v_c / a.sum_v } else { a.c_min as f64 };
        let peak_time = centroid_c * d_time + t0;
        let peak_freq = centroid_r * d_freq + f0;
        // N_pixels along each axis = (max - min + 1); pydynamo uses
        // skimage bbox which is (min, max+1) → (maxc - minc) = N_pixels.
        let n_t = (a.c_max - a.c_min + 1) as f64;
        let n_f = (a.r_max - a.r_min + 1) as f64;
        let duration = n_t * d_time;
        let bandwidth = n_f * d_freq;
        let height = a.max_v - a.min_v;
        let volume = a.sum_v * d_time * d_freq;
        // BoundingBox in pydynamo format: (time_tl, freq_tl, width_s, height_Hz).
        let t_tl = (a.c_min as f64) * d_time + t0;
        let f_tl = (a.r_min as f64) * d_freq + f0;
        let w_s = n_t * d_time;
        let h_hz = n_f * d_freq;

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
    }
    out
}

/// Run the full pipeline on one (F, T) spectrogram segment with its own
/// stimes. Returns peak stats (no filtering yet) AND the trimmed label
/// image aligned with those peaks (row k in the returned `SegmentPeaks`
/// corresponds to label `k + 1` in the label image; zero pixels are
/// background).
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
    let f_s = params.downsample_f.max(1);
    let t_s = params.downsample_t.max(1);
    let seg_lr = stride_downsample(spect, f_s, t_s);
    let (f_lr, t_lr) = seg_lr.dim();
    if f_lr < 2 || t_lr < 2 {
        return Ok((SegmentPeaks::default(), Array2::<i64>::zeros((f, t))));
    }

    // 2) Watershed on -seg_LR.
    let neg: Array2<f64> = seg_lr.mapv(|v| -v);
    let ws_u16 = crate::matlab_watershed::matlab_watershed_2d(neg.view());
    let ws_i64: Array2<i64> = ws_u16.mapv(|v| v as i64);

    // 3) Merge (interior-only labels).
    let merged_i32 = crate::merge::run(
        ws_i64.view(),
        seg_lr.view(),
        params.merge_thresh,
        params.max_merges,
    )?;
    let merged_i64: Array2<i64> = merged_i32.mapv(|v| v as i64);

    // 4) expand_labels(distance=5) — fills watershed 0-line with nearest
    //    label. pydynamo does this for skimage semantic match; MATLAB
    //    extractTFPeaks.m does NOT. Controlled by ExtractParams.expand_labels
    //    (default true for pydynamo parity; set false to get MATLAB-native
    //    behavior without the 1-pixel-wider regions).
    let expanded_lr = if params.expand_labels_distance > 0 {
        expand_labels_bfs(merged_i64.view(), params.expand_labels_distance as usize)
    } else {
        merged_i64.clone()
    };

    // 5) Resize up to full segment shape (nearest-neighbor).
    let labels_full = if f_s > 1 || t_s > 1 {
        resize_labels_nn(expanded_lr.view(), f, t)
    } else {
        expanded_lr
    };

    // 5b) Pre-trim dur/bw filter — pydynamo extract.py:182-193 /
    //     MATLAB extractTFPeaks.m:300-307. Drop regions whose
    //     (max_idx - min_idx - 1) * dx fails the dur_min / bw_min cut
    //     BEFORE trim runs. This both changes the final peak count
    //     (trim creates spurious sub-regions from jagged bboxes on
    //     rejected peaks) and skips unnecessary trim work.
    let mut labels_full = labels_full;
    if params.dur_min > 0.0 || params.bw_min > 0.0 {
        let d_time = stimes[1] - stimes[0];
        let d_freq = sfreqs[1] - sfreqs[0];
        let (h, w) = labels_full.dim();
        let mut bb: HashMap<i64, (usize, usize, usize, usize)> = HashMap::new();
        for r in 0..h {
            for c in 0..w {
                let lbl = labels_full[[r, c]];
                if lbl <= 0 { continue; }
                let e = bb.entry(lbl).or_insert((r, r, c, c));
                if r < e.0 { e.0 = r; }
                if r > e.1 { e.1 = r; }
                if c < e.2 { e.2 = c; }
                if c > e.3 { e.3 = c; }
            }
        }
        let mut drop: std::collections::HashSet<i64> = std::collections::HashSet::new();
        for (lbl, (r_min, r_max, c_min, c_max)) in &bb {
            // MATLAB/pydynamo use (N - 1) * dx = (max - min) * dx. Note
            // `p.bbox` in skimage is (min, max+1) so (maxc - minc - 1) =
            // (N_pixels - 1). Our r_max/c_max are inclusive → use
            // (max - min) directly.
            let pre_dur = (*c_max as f64 - *c_min as f64) * d_time;
            let pre_bw = (*r_max as f64 - *r_min as f64) * d_freq;
            if !(pre_dur > params.dur_min && pre_bw > params.bw_min) {
                drop.insert(*lbl);
            }
        }
        if !drop.is_empty() {
            for r in 0..h {
                for c in 0..w {
                    let lbl = labels_full[[r, c]];
                    if drop.contains(&lbl) {
                        labels_full[[r, c]] = 0;
                    }
                }
            }
        }
    }

    // 6) Trim on the full-resolution spect.
    let labels_full_i32: Array2<i32> = labels_full.mapv(|v| v as i32);
    let shift_val = if params.trim_shift_val.is_nan() {
        spect.iter().cloned().fold(f64::INFINITY, f64::min)
    } else {
        params.trim_shift_val
    };
    let trimmed = crate::trim::trim_all_regions(
        labels_full_i32.view(), spect, params.trim_vol_thresh, shift_val,
    )?;
    let trimmed_i64: Array2<i64> = trimmed.mapv(|v| v as i64);

    // 7) Peak properties.
    let peaks = compute_peak_props_from_trim(
        trimmed_i64.view(), spect, stimes, sfreqs, segment_num,
    );

    // 8) Build the per-segment label image whose labels align with the
    //    returned `peaks` rows. `compute_peak_props_from_trim` iterates
    //    over unique label IDs in sorted order, so the k-th peak in the
    //    output corresponds to the k-th smallest label in `trimmed_i64`.
    //    Build a remap old_id -> new_id = rank+1 (1-based), zero stays 0.
    let mut ids: Vec<i64> = Vec::new();
    {
        let mut seen = std::collections::BTreeSet::new();
        for &v in trimmed_i64.iter() {
            if v > 0 { seen.insert(v); }
        }
        ids.extend(seen);
    }
    debug_assert_eq!(ids.len(), peaks.len());
    // Use a HashMap for the remap; small N.
    let mut remap: HashMap<i64, i64> = HashMap::with_capacity(ids.len());
    for (k, old) in ids.iter().enumerate() {
        remap.insert(*old, (k as i64) + 1);
    }
    let labels_out = trimmed_i64.mapv(|v| {
        if v <= 0 { 0 } else { *remap.get(&v).unwrap_or(&0) }
    });

    Ok((peaks, labels_out))
}

/// Full top-level: split spect into seg_time segments, extract peaks from
/// each (in parallel), concatenate, then apply filterStatsTable (dur / bw /
/// peak_freq / height-dB cuts).
pub fn extract_tfpeaks(
    spect: ArrayView2<f64>,
    stimes: ArrayView1<f64>,
    sfreqs: ArrayView1<f64>,
    baseline: Option<ArrayView1<f64>>,
    params: &ExtractParams,
) -> Result<(SegmentPeaks, Array2<i64>), String> {
    let (f, t) = spect.dim();
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
            if !denom.is_finite() || denom == 0.0 { continue; }
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
    let seg_time = if params.seg_time > 0.0 { params.seg_time } else { 30.0 };
    let max_dx = (seg_time / dt).floor() as usize;
    if max_dx < 2 {
        return Err("seg_time too small relative to dt (max_dx < 2)".to_string());
    }
    let n_segs = ((t as f64) / (max_dx as f64)).ceil() as usize;
    let new_dx = (((t as f64) / (n_segs as f64)).ceil() as usize).max(1);

    let mut seg_bounds: Vec<(usize, usize, usize)> = Vec::with_capacity(n_segs);
    for ii in 0..n_segs {
        let start = ii * new_dx;
        let end = (start + new_dx).min(t);
        if end.saturating_sub(start) < 2 {
            continue;
        }
        seg_bounds.push((ii + 1, start, end));
    }

    // Parallel extract; collect per-segment results preserving order.
    let results: Vec<Result<(SegmentPeaks, Array2<i64>), String>> = seg_bounds
        .par_iter()
        .map(|&(si, start, end)| {
            // Make the segment spect contiguous — trim / merge require it.
            let sub_spect = spect_owned
                .slice(ndarray::s![.., start..end])
                .to_owned();
            let sub_times = stimes.slice(ndarray::s![start..end]).to_owned();
            extract_tfpeaks_segment(
                sub_spect.view(), sub_times.view(), sfreqs, si as f64, params,
            )
        })
        .collect();

    // Stitch per-segment labels into a full (F, T) image with running
    // label offset so that peak row k in the concatenated SegmentPeaks
    // maps to label k+1 in the global image.
    let mut all = SegmentPeaks::default();
    let mut labels_full = Array2::<i64>::zeros((f, t));
    let mut offset: i64 = 0;
    let mut per_seg: Vec<(usize, usize, Array2<i64>)> = Vec::with_capacity(seg_bounds.len());
    for (r, &(_si, start, end)) in results.into_iter().zip(seg_bounds.iter()) {
        match r {
            Ok((p, lbl)) => {
                let n_here = p.len() as i64;
                // Write with offset into the global image.
                let (fh, tw) = lbl.dim();
                debug_assert_eq!(fh, f);
                debug_assert_eq!(tw, end - start);
                for r in 0..fh {
                    for c in 0..tw {
                        let v = lbl[[r, c]];
                        if v > 0 {
                            labels_full[[r, start + c]] = v + offset;
                        }
                    }
                }
                all.extend(p);
                offset += n_here;
                per_seg.push((start, end, lbl));
            }
            Err(e) => return Err(e),
        }
    }
    // `per_seg` is kept so the debug_assert above can be cheap; drop it.
    drop(per_seg);

    // filterStatsTable: Duration ∈ (dur_min, dur_max), Bandwidth ∈ (bw_min, bw_max),
    // PeakFrequency ∈ (freq_min, freq_max), pow2db(Height) > ht_db_min.
    let d_time = stimes[1] - stimes[0];
    let d_freq = if sfreqs.len() > 1 { sfreqs[1] - sfreqs[0] } else { 0.0 };
    let keep = filter_indices(&all, params, d_time, d_freq);

    // Build a remap from old global label id -> new label id (1..n_kept).
    // Peaks not in `keep` -> 0.
    let mut remap_filter: Vec<i64> = vec![0; all.len() + 1]; // index by old id (1-based)
    for (new_idx, &old_idx) in keep.iter().enumerate() {
        // old label id = old_idx + 1 (1-based global id)
        remap_filter[old_idx + 1] = (new_idx as i64) + 1;
    }
    for v in labels_full.iter_mut() {
        if *v > 0 {
            let old = *v as usize;
            *v = if old < remap_filter.len() { remap_filter[old] } else { 0 };
        }
    }

    Ok((take_indices(&all, &keep), labels_full))
}

fn filter_indices(
    p: &SegmentPeaks,
    params: &ExtractParams,
    d_time: f64,
    d_freq: f64,
) -> Vec<usize> {
    let mut out = Vec::new();
    for i in 0..p.len() {
        let dur = p.duration[i];
        let bw = p.bandwidth[i];
        let pf = p.peak_freq[i];
        let h = p.height[i];
        let h_db = if h > 0.0 { 10.0 * h.log10() } else { f64::NAN };
        // MATLAB extractTFPeaks.m post-trim filter uses (max-min)*dx >
        // dur_min, where Duration = N_pixels*dx → filter-value = Duration
        // - dx. pydynamo replicates this (extract.py:262-264). We apply
        // both it AND MATLAB's filterStatsTable.m `Duration > dur_min`
        // (line 78), which is the strictly looser of the two; the stricter
        // one is the binding constraint.
        let filter_dur = dur - d_time;
        let filter_bw = bw - d_freq;
        let dur_ok = filter_dur > params.dur_min && dur < params.dur_max;
        let bw_ok = filter_bw > params.bw_min && bw < params.bw_max;
        let pf_ok = pf > params.freq_min && pf < params.freq_max;
        let ht_ok = h_db.is_finite() && h_db > params.ht_db_min;
        if dur_ok && bw_ok && pf_ok && ht_ok {
            out.push(i);
        }
    }
    out
}

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
    fn filter_basic() {
        let mut p = SegmentPeaks::default();
        p.peak_time.push(1.0); p.peak_freq.push(10.0);
        p.duration.push(1.0);  p.bandwidth.push(5.0);
        p.height.push(100.0);  p.volume.push(1.0);
        p.segment_num.push(1.0);
        p.bbox.extend_from_slice(&[0.0, 0.0, 0.0, 0.0]);
        let params = ExtractParams {
            seg_time: 30.0, downsample_f: 1, downsample_t: 1,
            merge_thresh: 8.0, max_merges: f64::INFINITY, trim_vol_thresh: 0.8,
            trim_shift_val: f64::NAN,
            dur_min: 0.5, dur_max: 5.0,
            bw_min: 2.0, bw_max: 15.0,
            freq_min: 0.0, freq_max: 40.0,
            ht_db_min: 7.0, expand_labels_distance: 5,
        };
        let kept = filter_indices(&p, &params, 0.05, 0.1);
        // Height=100 → pow2db = 20 dB > 7 dB ✓; filter_dur = 1.0 - 0.05
        // = 0.95 > dur_min 0.5 ✓; bw 5 ∈ (2, 15) ✓ (filter_bw = 4.9 > 2).
        assert_eq!(kept, vec![0usize]);

        // Out-of-range duration rejected.
        p.duration[0] = 10.0;
        let kept = filter_indices(&p, &params, 0.05, 0.1);
        assert!(kept.is_empty());
    }

    fn _unused(_a: Array1<f64>) {}

    /// Synthetic 2-segment extract: verify labels shape matches spect,
    /// that non-zero labels are 1..n_peaks contiguously, that each non-zero
    /// pixel points at a valid peak row, and that post-filter surviving
    /// labels remain contiguous 1..n_kept.
    #[test]
    fn extract_tfpeaks_labels_shape_and_stitching() {
        // Construct a 2-segment spect: 32 freq bins, 600 time bins at
        // dt=0.1 s → T*dt = 60 s; with seg_time = 30 s the pipeline
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
        for v in spect.iter_mut() { *v += 0.1; }

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
            spect.view(), stimes.view(), sfreqs.view(), None, &params,
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
        // at its centre, with the peak time near the planted column.
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
                if v <= 0 { continue; }
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
}
