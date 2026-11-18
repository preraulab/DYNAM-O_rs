//! Port of MATLAB `extracthistpeaks.m` — watershed segmentation of a 2-D
//! SO-power / SO-phase histogram.
//!
//! `extracthistpeaks.m` runs, on a small 2-D histogram `img` (rows = SO
//! feature axis, cols = frequency axis):
//!   1. `watershed(-img, 8)` — invert so histogram *peaks* become catchment
//!      basins, then flood-label (MATLAB `runWatershed`).
//!   2. Build a region-adjacency graph (`Ldata2graph`).
//!   3. Iteratively merge neighboring regions whose edge weight exceeds
//!      `merge_thresh` (`mergeWshedSegment`).
//!   4. Drop regions failing a pre-trim duration / bandwidth cut.
//!   5. Trim each surviving region to `trim_vol` of its peak volume
//!      (`trimWshedRegions`).
//!   6. Compute per-region stats and apply a final duration / bandwidth /
//!      height filter (`filterStatsTable`).
//!   7. Paint a label matrix `Ldata` (1..N), which MATLAB renders to RGB
//!      via `label2rgb(Ldata, 'jet', 'k', 'shuffle')`.
//!
//! This is exactly the spectrogram `extract_tfpeaks` pipeline already in
//! `extract_pipeline.rs`, minus segmentation / downsampling / resize (a
//! SOPH is a single small array). We therefore reuse the same building
//! blocks:
//!   * `matlab_watershed::matlab_watershed_2d`  — step 1
//!   * `merge::run`                             — step 3 (via the adjacency
//!                                                graph it builds internally)
//!   * `trim::trim_all_regions`                 — step 5
//!
//! Axis convention. MATLAB calls `extracthistpeaks(SOPH, x, y)` with
//! `x = SO_bins` (the columns of MATLAB's `(freq, SO)` image) and
//! `y = freq_bins` (the rows). So `dt = x(2)-x(1)` is the SO spacing and
//! `df = y(2)-y(1)` is the frequency spacing; the `dur_min` filter cuts
//! on the SO span and `bw_min` cuts on the frequency span.
//!
//! This port takes `hist` as `(rows = SO bins, cols = freq bins)` — the
//! transpose of MATLAB's image, matching how the DYNAM-O SOPH TIFFs are
//! stored. So here `dur` (= `dur_min`) is the ROW span and `bw`
//! (= `bw_min`) is the COLUMN span.

use ndarray::{Array2, Array3, ArrayView2};
use std::collections::{BTreeSet, HashMap, HashSet};

/// Parameters for `extract_hist_peaks`. Defaults match `extracthistpeaks.m`
/// (the values it falls back to when the corresponding argument is empty).
#[derive(Clone, Copy, Debug)]
pub struct HistPeakParams {
    /// Edge-weight threshold for stopping the region merge. MATLAB default
    /// (`extracthistpeaks.m:131-133`): `merge_thresh = 0.1`.
    pub merge_thresh: f64,
    /// Maximum number of merges. MATLAB default: `inf`.
    pub max_merges: f64,
    /// Fraction of peak volume each region is trimmed to. MATLAB default
    /// (`extracthistpeaks.m:139-141`): `trim_vol = 0.95`.
    pub trim_vol: f64,
    /// Floor level from which trim volume is measured. NaN → `min(img)`,
    /// matching MATLAB `trim_shift = min(img(:))`.
    pub trim_shift: f64,
    /// Minimum allowed span along the SO-feature (row) axis. MATLAB
    /// default (`extracthistpeaks.m:69-72`): `dur_min = 5`.
    pub dur_min: f64,
    /// Minimum allowed span along the frequency (column) axis. MATLAB
    /// default (`extracthistpeaks.m:74-77`): `bw_min = 2`.
    pub bw_min: f64,
    /// Minimum allowed peak height (max-min over region pixels, linear).
    /// MATLAB default (`extracthistpeaks.m:79-82`): `height_min = 2`.
    /// `filterStatsTable` compares `pow2db(height) > pow2db(height_min)`,
    /// which is monotone, so we compare the linear height directly.
    pub height_min: f64,
}

impl Default for HistPeakParams {
    fn default() -> Self {
        HistPeakParams {
            merge_thresh: 0.1,
            max_merges: f64::INFINITY,
            trim_vol: 0.95,
            trim_shift: f64::NAN,
            dur_min: 5.0,
            bw_min: 2.0,
            height_min: 2.0,
        }
    }
}

/// Per-region statistics, mirroring the columns MATLAB
/// `computePeakStatsTable` produces and `param_basis_{power,phase}.m`
/// consumes to seed the fit (`mode_params`).
#[derive(Clone, Copy, Debug)]
pub struct HistPeak {
    /// Densely-numbered region label (1-based).
    pub label: i64,
    /// `max(pixels) - min(pixels)` over the region.
    pub height: f64,
    /// Intensity-weighted centroid on the frequency axis.
    pub peak_freq: f64,
    /// Intensity-weighted centroid on the SO axis (MATLAB `SOFeature`).
    pub so_feature: f64,
    /// Bounding-box extent along the SO axis (MATLAB `Duration`).
    pub duration: f64,
    /// Bounding-box extent along the frequency axis (MATLAB `Bandwidth`).
    pub bandwidth: f64,
}

/// Separable 2-D Gaussian blur (`sigma` in bins; non-finite → 0). Filter
/// radius `ceil(2σ)` matches MATLAB `imgaussfilt`'s default FilterSize.
pub fn gaussian_blur(hist: ArrayView2<f64>, sigma_row: f64, sigma_col: f64) -> Array2<f64> {
    fn kernel(sigma: f64) -> Vec<f64> {
        if sigma <= 0.0 {
            return vec![1.0];
        }
        let r = (2.0 * sigma).ceil() as isize;
        let mut k: Vec<f64> = (-r..=r)
            .map(|i| (-(i as f64).powi(2) / (2.0 * sigma * sigma)).exp())
            .collect();
        let s: f64 = k.iter().sum();
        for v in &mut k {
            *v /= s;
        }
        k
    }
    let (h, w) = hist.dim();
    let kr = kernel(sigma_row);
    let kc = kernel(sigma_col);
    let rr = (kr.len() / 2) as isize;
    let rc = (kc.len() / 2) as isize;

    let mut src = Array2::<f64>::zeros((h, w));
    for r in 0..h {
        for c in 0..w {
            let v = hist[[r, c]];
            src[[r, c]] = if v.is_finite() { v } else { 0.0 };
        }
    }
    let mut mid = Array2::<f64>::zeros((h, w));
    for r in 0..h {
        for c in 0..w {
            let mut acc = 0.0;
            for (ki, &kv) in kr.iter().enumerate() {
                let r2 = (r as isize + ki as isize - rr).clamp(0, h as isize - 1) as usize;
                acc += kv * src[[r2, c]];
            }
            mid[[r, c]] = acc;
        }
    }
    let mut out = Array2::<f64>::zeros((h, w));
    for r in 0..h {
        for c in 0..w {
            let mut acc = 0.0;
            for (ki, &kv) in kc.iter().enumerate() {
                let c2 = (c as isize + ki as isize - rc).clamp(0, w as isize - 1) as usize;
                acc += kv * mid[[r, c2]];
            }
            out[[r, c]] = acc;
        }
    }
    out
}

/// Per-region bounding box / intensity accumulator over a label image.
struct RegionAccum {
    r_min: usize,
    r_max: usize,
    c_min: usize,
    c_max: usize,
    min_v: f64,
    max_v: f64,
    count: usize,
}

/// Collect, for every non-zero label in `labels`, its inclusive bounding
/// box plus min/max of `img` over that label's pixels (NaN pixels skipped,
/// matching MATLAB's `~isnan` exclusion in `computePeakStatsTable`).
fn accumulate_regions(
    labels: ArrayView2<i64>,
    img: ArrayView2<f64>,
) -> HashMap<i64, RegionAccum> {
    let (h, w) = labels.dim();
    let mut map: HashMap<i64, RegionAccum> = HashMap::new();
    for r in 0..h {
        for c in 0..w {
            let lbl = labels[[r, c]];
            if lbl <= 0 {
                continue;
            }
            let v = img[[r, c]];
            let e = map.entry(lbl).or_insert(RegionAccum {
                r_min: r,
                r_max: r,
                c_min: c,
                c_max: c,
                min_v: f64::INFINITY,
                max_v: f64::NEG_INFINITY,
                count: 0,
            });
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
            if !v.is_nan() {
                if v < e.min_v {
                    e.min_v = v;
                }
                if v > e.max_v {
                    e.max_v = v;
                }
            }
            e.count += 1;
        }
    }
    map
}

/// Extract peak regions from a 2-D SO-power / SO-phase histogram, label
/// image only — see [`extract_hist_peaks_with_stats`] for the per-region
/// statistics.
pub fn extract_hist_peaks(
    hist: ArrayView2<f64>,
    so_bins: &[f64],
    freq_bins: &[f64],
    params: &HistPeakParams,
) -> Array2<i64> {
    extract_hist_peaks_with_stats(hist, so_bins, freq_bins, params).0
}

/// 8-connected connected-component labeling of a boolean mask — the
/// MATLAB `bwlabel(_, 8)` equivalent. Components are numbered in
/// column-major first-encounter order (MATLAB's scan order), so the
/// label ids match MATLAB's for downstream label-ordered tie-breaks.
fn bwlabel_8(mask: &Array2<bool>) -> Array2<i64> {
    let (h, w) = mask.dim();
    let mut labels = Array2::<i64>::zeros((h, w));
    let mut next: i64 = 1;
    let mut stack: Vec<(usize, usize)> = Vec::new();
    for c in 0..w {
        for r in 0..h {
            if !mask[[r, c]] || labels[[r, c]] != 0 {
                continue;
            }
            let lab = next;
            next += 1;
            labels[[r, c]] = lab;
            stack.clear();
            stack.push((r, c));
            while let Some((kr, kc)) = stack.pop() {
                for dr in -1i32..=1 {
                    for dc in -1i32..=1 {
                        if dr == 0 && dc == 0 {
                            continue;
                        }
                        let nr = kr as i32 + dr;
                        let nc = kc as i32 + dc;
                        if nr < 0 || nc < 0 || nr >= h as i32 || nc >= w as i32 {
                            continue;
                        }
                        let (nr, nc) = (nr as usize, nc as usize);
                        if mask[[nr, nc]] && labels[[nr, nc]] == 0 {
                            labels[[nr, nc]] = lab;
                            stack.push((nr, nc));
                        }
                    }
                }
            }
        }
    }
    labels
}

/// Extract peak regions from a 2-D SO-power / SO-phase histogram.
///
/// `hist` is the histogram with `rows = so_bins.len()` and
/// `cols = freq_bins.len()`. `so_bins` / `freq_bins` are the bin-center
/// axes — their spacing drives the dur/bw filters and their first value
/// anchors the per-region centroids.
///
/// Returns `(label_image, peaks)`. The label image is the same shape as
/// `hist`: `0` = background (watershed lines + rejected pixels), `1..=N`
/// = the surviving peak regions, densely numbered. `peaks[k]` carries
/// region `k+1`'s stats (`computePeakStatsTable` columns).
///
/// Pipeline = watershed(`-hist`) → merge → pre-trim dur/bw cut → trim →
/// final dur/bw/height filter, mirroring `extracthistpeaks.m`.
pub fn extract_hist_peaks_with_stats(
    hist: ArrayView2<f64>,
    so_bins: &[f64],
    freq_bins: &[f64],
    params: &HistPeakParams,
) -> (Array2<i64>, Vec<HistPeak>) {
    let (h, w) = hist.dim();
    if h < 2 || w < 2 {
        return (Array2::<i64>::zeros((h, w)), Vec::new());
    }
    if std::env::var("HISTPEAKS_DEBUG").is_ok() {
        eprintln!(
            "[params] merge_thresh={} dur_min={} bw_min={} height_min={} trim_vol={} trim_shift={} max_merges={}",
            params.merge_thresh, params.dur_min, params.bw_min, params.height_min,
            params.trim_vol, params.trim_shift, params.max_merges
        );
    }

    // Bin spacings. `hist` is `(rows = SO, cols = freq)`. In MATLAB
    // `extracthistpeaks(SOPH, x = SO_bins, y = freq_bins)` the duration
    // filter uses `dt = x(2)-x(1)` (the SO axis) and the bandwidth filter
    // uses `df = y(2)-y(1)` (the frequency axis) — so `dur` is an SO span
    // and `bw` is a frequency span.
    let d_row = if so_bins.len() >= 2 {
        so_bins[1] - so_bins[0]
    } else {
        1.0
    };
    let d_col = if freq_bins.len() >= 2 {
        freq_bins[1] - freq_bins[0]
    } else {
        1.0
    };

    // Make a contiguous owned copy: watershed / merge / trim all require
    // C-contiguous slices, and a histogram may be a non-contiguous view
    // (e.g. sliced from a TIFF buffer).
    let img: Array2<f64> = hist.to_owned();

    // ---- 1) Watershed on the inverted histogram. -------------------------
    // MATLAB `runWatershed` runs `watershed(-data, 8)`: inverting turns
    // histogram peaks into catchment basins. NaN handling: MATLAB zeroes
    // NaNs before watershed and zeroes their labels after; we substitute
    // 0.0 for NaN in the watershed input (label cleanup happens implicitly
    // because trim/stats skip NaN pixels).
    let neg: Array2<f64> = img.mapv(|v| if v.is_nan() { 0.0 } else { -v });
    let ws_u16 = crate::matlab_watershed::matlab_watershed_2d(neg.view());
    // `runWatershed.m`: after `watershed(-data)`, zero the NaN pixels'
    // labels (`Ldata(nan_idx)=0`) and re-label the connected components
    // (`Ldata=bwlabel(Ldata)`). Zeroing NaN pixels can split a basin, so
    // this is not a no-op — it must run before the graph is built.
    let mut ws_mask = Array2::<bool>::from_elem((h, w), false);
    for r in 0..h {
        for c in 0..w {
            ws_mask[[r, c]] = ws_u16[[r, c]] != 0 && !img[[r, c]].is_nan();
        }
    }
    let ws_i64: Array2<i64> = bwlabel_8(&ws_mask);
    if std::env::var("HISTPEAKS_DEBUG").is_ok() {
        let n: std::collections::HashSet<i64> =
            ws_i64.iter().copied().filter(|&v| v > 0).collect();
        eprintln!("[histpeaks] after watershed: {} basins", n.len());
    }

    // ---- 2 & 3) Build adjacency graph + iterative merge. -----------------
    // `merge::run` builds the region-adjacency graph internally (the
    // `Ldata2graph` step) and runs the max-weight merge loop
    // (`mergeWshedSegment`). We take the `all_px` label image: MATLAB's
    // `extracthistpeaks` carries each region as `regions{}` = `interior ∪
    // border` union-merged through the dur/bw cut, trim, and
    // `computePeakStatsTable`, so every span/centroid/height is measured
    // over the full (solid, no internal 0-line holes) region.
    let merged = match crate::merge::run_all_px(
        ws_i64.view(),
        img.view(),
        params.merge_thresh,
        params.max_merges,
    ) {
        Ok(m) => m,
        Err(_) => return (Array2::<i64>::zeros((h, w)), Vec::new()),
    };
    let mut labels: Array2<i64> = merged.mapv(|v| v as i64);
    if std::env::var("HISTPEAKS_DEBUG").is_ok() {
        let n: std::collections::HashSet<i64> =
            labels.iter().copied().filter(|&v| v > 0).collect();
        eprintln!("[histpeaks] after merge: {} regions", n.len());
    }

    // ---- 4) Pre-trim duration / bandwidth cut. ---------------------------
    // MATLAB `extracthistpeaks.m`: drops regions whose SO span (`*dt`,
    // the column axis in MATLAB's `(freq, SO)` image — here the ROW axis)
    // fails `dur_min`, or whose frequency span (`*df` — here the COLUMN
    // axis) fails `bw_min`, BEFORE trimming. Uses `(max-min)*Δ`.
    if params.dur_min > 0.0 || params.bw_min > 0.0 {
        let accum = accumulate_regions(labels.view(), img.view());
        let mut drop: HashSet<i64> = HashSet::new();
        let dbg = std::env::var("HISTPEAKS_DEBUG").is_ok();
        for (lbl, a) in &accum {
            let so_span = (a.r_max - a.r_min) as f64 * d_row;
            let freq_span = (a.c_max - a.c_min) as f64 * d_col;
            let keep = so_span > params.dur_min && freq_span > params.bw_min;
            if !keep {
                drop.insert(*lbl);
            } else if dbg {
                eprintln!(
                    "[cut] keep lbl={} so_span={:.3} freq_span={:.3}",
                    lbl, so_span, freq_span
                );
            }
        }
        if !drop.is_empty() {
            for v in labels.iter_mut() {
                if drop.contains(v) {
                    *v = 0;
                }
            }
        }
    }
    if std::env::var("HISTPEAKS_DEBUG").is_ok() {
        let n: std::collections::HashSet<i64> =
            labels.iter().copied().filter(|&v| v > 0).collect();
        eprintln!("[histpeaks] after pre-trim dur/bw cut: {} regions", n.len());
    }

    // ---- 5) Trim each region to `trim_vol` of its peak volume. -----------
    let labels_i32: Array2<i32> = labels.mapv(|v| v as i32);
    let shift_val = if params.trim_shift.is_nan() {
        // Smallest non-NaN value of the image (MATLAB `min(img(:))`).
        let mut image_min = f64::INFINITY;
        for &value in img.iter() {
            if !value.is_nan() && value < image_min {
                image_min = value;
            }
        }
        image_min
    } else {
        params.trim_shift
    };
    let trimmed = if params.trim_vol < 1.0 {
        match crate::trim::trim_all_regions(
            labels_i32.view(),
            img.view(),
            params.trim_vol,
            shift_val,
        ) {
            Ok(t) => t.mapv(|v| v as i64),
            Err(_) => return (Array2::<i64>::zeros((h, w)), Vec::new()),
        }
    } else {
        labels
    };
    if std::env::var("HISTPEAKS_DEBUG").is_ok() {
        let n: std::collections::HashSet<i64> =
            trimmed.iter().copied().filter(|&v| v > 0).collect();
        eprintln!("[histpeaks] after trim: {} regions", n.len());
    }

    // ---- 6) Final duration / bandwidth / height filter. ------------------
    // MATLAB `filterStatsTable` keeps regions with Duration > dur_min,
    // Bandwidth > bw_min, pow2db(Height) > pow2db(height_min). pow2db is
    // monotone, so comparing linear height to `height_min` is equivalent.
    // `computePeakStatsTable` sets `Duration = BoundingBox width` (the SO
    // axis = ROW span here) and `Bandwidth = BoundingBox height` (the
    // frequency axis = COLUMN span), each as `(max-min+1)*Δ`.
    let accum = accumulate_regions(trimmed.view(), img.view());
    let mut surviving: BTreeSet<i64> = BTreeSet::new();
    for (lbl, a) in &accum {
        let duration = (a.r_max - a.r_min + 1) as f64 * d_row;
        let bandwidth = (a.c_max - a.c_min + 1) as f64 * d_col;
        let height = if a.max_v.is_finite() && a.min_v.is_finite() {
            a.max_v - a.min_v
        } else {
            0.0
        };
        let dur_ok = duration > params.dur_min;
        let bw_ok = bandwidth > params.bw_min;
        let ht_ok = height > params.height_min;
        if std::env::var("HISTPEAKS_DEBUG").is_ok() {
            eprintln!(
                "[filter] lbl={} dur={:.3}({}) bw={:.3}({}) ht={:.3}({})",
                lbl, duration, dur_ok, bandwidth, bw_ok, height, ht_ok
            );
        }
        if dur_ok && bw_ok && ht_ok {
            surviving.insert(*lbl);
        }
    }

    // ---- 7) Densely renumber survivors 1..=N (deterministic order). ------
    // MATLAB paints `Ldata(regions{ii}) = ii` for ii = 1..N in table order;
    // the surviving watershed labels are processed in ascending order, so
    // numbering by sorted label id is the deterministic equivalent.
    let mut remap: HashMap<i64, i64> = HashMap::with_capacity(surviving.len());
    for (k, lbl) in surviving.iter().enumerate() {
        remap.insert(*lbl, (k as i64) + 1);
    }
    // Survivors take their new number. Every other pixel becomes background.
    let mut out = Array2::<i64>::zeros(trimmed.dim());
    for ((row, col), &old_label) in trimmed.indexed_iter() {
        if old_label > 0 {
            if let Some(&new_label) = remap.get(&old_label) {
                out[[row, col]] = new_label;
            }
        }
    }

    // ---- 8) Per-region stats (`computePeakStatsTable`). ------------------
    let stats = compute_peak_stats(
        out.view(),
        img.view(),
        surviving.len(),
        so_bins,
        freq_bins,
        d_row,
        d_col,
    );
    (out, stats)
}

/// Compute `computePeakStatsTable`-equivalent stats for each densely-
/// numbered region `1..=n` in `labels`. `regionprops` on MATLAB's
/// `Ldata` (NaN pixels excluded) gives: WeightedCentroid → `PeakFrequency`
/// / `SOFeature`, BoundingBox → `Duration` / `Bandwidth`, PixelValues →
/// `Height = max-min`. Centroid → axis value: `first_bin + centroid·Δ`.
fn compute_peak_stats(
    labels: ArrayView2<i64>,
    img: ArrayView2<f64>,
    n: usize,
    so_bins: &[f64],
    freq_bins: &[f64],
    d_row: f64,
    d_col: f64,
) -> Vec<HistPeak> {
    /// Running totals for one region over its finite pixels.
    struct RegionTotals {
        /// Sum of the pixel values.
        sum_v: f64,
        /// Sum of row index times value, for the value-weighted centroid.
        sum_rv: f64,
        /// Sum of column index times value, for the value-weighted centroid.
        sum_cv: f64,
        min_v: f64,
        max_v: f64,
        row_min: usize,
        row_max: usize,
        col_min: usize,
        col_max: usize,
        has_pixels: bool,
    }
    let mut acc: Vec<RegionTotals> = Vec::with_capacity(n);
    for _ in 0..n {
        acc.push(RegionTotals {
            sum_v: 0.0,
            sum_rv: 0.0,
            sum_cv: 0.0,
            min_v: f64::INFINITY,
            max_v: f64::NEG_INFINITY,
            row_min: usize::MAX,
            row_max: 0,
            col_min: usize::MAX,
            col_max: 0,
            has_pixels: false,
        });
    }
    let (h, w) = labels.dim();
    for r in 0..h {
        for c in 0..w {
            let lbl = labels[[r, c]];
            if lbl < 1 || (lbl as usize) > n {
                continue;
            }
            let v = img[[r, c]];
            if !v.is_finite() {
                continue; // MATLAB `Ldata` excludes NaN pixels
            }
            let a = &mut acc[(lbl - 1) as usize];
            a.has_pixels = true;
            a.sum_v += v;
            a.sum_rv += r as f64 * v;
            a.sum_cv += c as f64 * v;
            a.min_v = a.min_v.min(v);
            a.max_v = a.max_v.max(v);
            a.row_min = a.row_min.min(r);
            a.row_max = a.row_max.max(r);
            a.col_min = a.col_min.min(c);
            a.col_max = a.col_max.max(c);
        }
    }
    let so0 = so_bins.first().copied().unwrap_or(0.0);
    let fq0 = freq_bins.first().copied().unwrap_or(0.0);
    let mut peaks: Vec<HistPeak> = Vec::new();
    for (i, a) in acc.iter().enumerate() {
        if !a.has_pixels {
            continue;
        }
        // Value-weighted centroid. A region whose values sum to zero has no
        // weighted centroid, so it takes the middle of its extent.
        let centroid_row;
        let centroid_col;
        if a.sum_v.abs() > 0.0 {
            centroid_row = a.sum_rv / a.sum_v;
            centroid_col = a.sum_cv / a.sum_v;
        } else {
            centroid_row = (a.row_min + a.row_max) as f64 / 2.0;
            centroid_col = (a.col_min + a.col_max) as f64 / 2.0;
        }
        peaks.push(HistPeak {
            label: i as i64 + 1,
            height: a.max_v - a.min_v,
            so_feature: so0 + centroid_row * d_row,
            peak_freq: fq0 + centroid_col * d_col,
            duration: (a.row_max - a.row_min + 1) as f64 * d_row,
            bandwidth: (a.col_max - a.col_min + 1) as f64 * d_col,
        });
    }
    peaks
}

/// Periodic-phase watershed — mirrors `param_basis_phase.m`'s SO-phase
/// pre-processing: triple the histogram along the phase axis
/// (`[SOPhH SOPhH(:,2:end-1) SOPhH]`) so regions can wrap across ±π,
/// blur with `imgaussfilt(_, gauss_filt_std)`, watershed the tripled
/// grid, then return the central period.
///
/// `hist` is `(rows = phase, cols = freq)`. `gauss_filt_std` is
/// `(sigma_phase, sigma_freq)` in bins (MATLAB `[10,5]` on the
/// `(freq, phase)` image → `(5, 10)` here). Returns the central-period
/// label image (same shape as `hist`, densely renumbered) and the
/// per-region stats whose SO-phase centroid lands in `[-π, π]`.
///
/// Keeping only the central-period regions is the net effect of MATLAB's
/// `uniquetol` wraparound de-dup — each real region appears once per
/// period in the tripled grid, and exactly one copy has its centroid in
/// `[-π, π]`.
pub fn extract_hist_peaks_phase(
    hist: ArrayView2<f64>,
    phase_bins: &[f64],
    freq_bins: &[f64],
    params: &HistPeakParams,
    gauss_filt_std: (f64, f64),
) -> (Array2<i64>, Vec<HistPeak>) {
    let (n, cols) = hist.dim();
    if n < 4 || cols < 2 || phase_bins.len() != n {
        return (Array2::<i64>::zeros((n, cols)), Vec::new());
    }
    // Triple along the phase (row) axis: [block1 | block2 minus ends | block3].
    let trip = 3 * n - 2;
    let mut tripled = Array2::<f64>::zeros((trip, cols));
    for c in 0..cols {
        for r in 0..n {
            tripled[[r, c]] = hist[[r, c]];
        }
        for r in 1..n - 1 {
            tripled[[n + r - 1, c]] = hist[[r, c]];
        }
        for r in 0..n {
            tripled[[2 * n - 2 + r, c]] = hist[[r, c]];
        }
    }
    let tripled = if gauss_filt_std.0 > 0.0 || gauss_filt_std.1 > 0.0 {
        gaussian_blur(tripled.view(), gauss_filt_std.0, gauss_filt_std.1)
    } else {
        tripled
    };

    // Real tripled phase axis: [phase-2π, phase(2:end-1), phase+2π].
    let two_pi = std::f64::consts::TAU;
    let mut trip_phase: Vec<f64> = Vec::with_capacity(trip);
    trip_phase.extend(phase_bins.iter().map(|&p| p - two_pi));
    trip_phase.extend(phase_bins[1..n - 1].iter().copied());
    trip_phase.extend(phase_bins.iter().map(|&p| p + two_pi));

    let (trip_labels, trip_stats) =
        extract_hist_peaks_with_stats(tripled.view(), &trip_phase, freq_bins, params);

    // Central period [-π, π] = tripled rows [n-1 .. 2n-2].
    let mut central = Array2::<i64>::zeros((n, cols));
    for r in 0..n {
        for c in 0..cols {
            central[[r, c]] = trip_labels[[n - 1 + r, c]];
        }
    }
    let mut uniq: Vec<i64> = central.iter().copied().filter(|&v| v > 0).collect();
    uniq.sort_unstable();
    uniq.dedup();
    // Renumber the labels present in the central period 1..=N in
    // ascending order.
    let mut remap: HashMap<i64, i64> = HashMap::with_capacity(uniq.len());
    for (k, lbl) in uniq.iter().enumerate() {
        remap.insert(*lbl, (k as i64) + 1);
    }
    for v in central.iter_mut() {
        if *v > 0 {
            *v = remap[v];
        } else {
            *v = 0;
        }
    }

    let pi = std::f64::consts::PI;
    let central_stats: Vec<HistPeak> = trip_stats
        .into_iter()
        .filter(|p| p.so_feature >= -pi && p.so_feature <= pi)
        .collect();
    (central, central_stats)
}

/// MATLAB's 256-entry `jet` colormap, sampled at `t ∈ [0, 1]`.
///
/// `jet` is piecewise-linear in each RGB channel. This evaluates the
/// standard MATLAB jet ramp directly (the same breakpoints `jet.m` uses),
/// returning 8-bit RGB. `t` is clamped to `[0, 1]`.
fn jet_color(t: f64) -> [u8; 3] {
    let t = t.clamp(0.0, 1.0);
    // MATLAB jet: each channel is a clamped triangle/trapezoid.
    //   red   peaks over t ∈ [3/8, 5/8] .. 1
    //   green peaks over t ∈ [1/8, 3/8] .. [5/8, 7/8]
    //   blue  peaks over t ∈ 0 .. [3/8, 5/8]
    // Implemented via the canonical jet interpolant.
    let chan = |t: f64, center: f64| -> f64 {
        // Triangle of half-width 1/4 centered at `center`, clamped to [0,1].
        (1.5 - (4.0 * (t - center)).abs()).clamp(0.0, 1.0)
    };
    let r = chan(t, 0.75);
    let g = chan(t, 0.5);
    let b = chan(t, 0.25);
    [
        (r * 255.0).round() as u8,
        (g * 255.0).round() as u8,
        (b * 255.0).round() as u8,
    ]
}

/// Deterministic label-index scramble, standing in for the shuffle of
/// MATLAB `label2rgb(..., 'shuffle')`. It serves the same purpose, which is
/// to give visually adjacent labels well-separated colors. It is not
/// MATLAB's permutation, so the colors differ from MATLAB's, and two labels
/// can land on the same color. Only the rendered image is affected. No
/// number in the peak table depends on it.
fn shuffled_index(label: i64, n: usize) -> usize {
    if n == 0 {
        return 0;
    }
    // Multiplicative hash that spreads consecutive labels far apart in color
    // space. 2654435761 is Knuth's multiplicative-hash constant.
    let h = (label as u64).wrapping_mul(2_654_435_761);
    (h % (n as u64)) as usize
}

/// Render a label image to an RGB array, MATLAB-equivalent to
/// `label2rgb(labels, 'jet', 'k', 'shuffle')`.
///
/// * Background (label 0) → black `[0, 0, 0]` (MATLAB `'k'`).
/// * Each non-zero label `1..=N` is mapped, via a deterministic shuffle, to
///   a distinct sample of the `jet` colormap.
///
/// Output shape is `(rows, cols, 3)`, `u8` per channel.
///
/// MATLAB `extracthistpeaks.m:281-283` then repaints background pixels gray
/// `[100,100,100]`. This function stops at `label2rgb` and leaves them
/// black. A caller that wants the gray background overwrites the zero-label
/// pixels.
pub fn label_to_rgb_jet(labels: ArrayView2<i64>) -> Array3<u8> {
    let (h, w) = labels.dim();
    let mut out = Array3::<u8>::zeros((h, w, 3));
    // Number of distinct non-zero labels.
    let mut uniq: BTreeSet<i64> = BTreeSet::new();
    for &v in labels.iter() {
        if v > 0 {
            uniq.insert(v);
        }
    }
    let n = uniq.len();
    if n == 0 {
        return out;
    }
    // Map each label id to a dense rank 0..n-1, then shuffle that rank and
    // sample jet. Dense ranking makes the colors independent of any gaps
    // in the label numbering.
    let mut rank: HashMap<i64, usize> = HashMap::with_capacity(uniq.len());
    for (k, lbl) in uniq.iter().enumerate() {
        rank.insert(*lbl, k);
    }
    for r in 0..h {
        for c in 0..w {
            let v = labels[[r, c]];
            if v <= 0 {
                continue; // background stays black
            }
            let dense = rank[&v];
            let shuf = shuffled_index(dense as i64, n);
            // Sample jet at `shuf / (n - 1)`, so the ranks span the whole
            // colormap. A single region takes the middle.
            let t = if n == 1 {
                0.5
            } else {
                (shuf as f64) / ((n - 1) as f64)
            };
            let rgb = jet_color(t);
            out[[r, c, 0]] = rgb[0];
            out[[r, c, 1]] = rgb[1];
            out[[r, c, 2]] = rgb[2];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a synthetic SOPH-like histogram: a uniform grid with three
    /// well-separated Gaussian blobs of differing amplitude / width. The
    /// blobs are spaced far enough apart (and the inter-blob valleys reach
    /// near zero) that even the aggressive `merge_thresh = 0.1` MATLAB
    /// default keeps them as three distinct watershed regions. Amplitudes
    /// (8..15) are in the magnitude range of real SO-power SOPH TIFFs.
    fn synthetic_hist() -> (Array2<f64>, Vec<f64>, Vec<f64>) {
        let n_so = 80usize; // rows
        let n_freq = 120usize; // cols
        let d_so = 0.4_f64;
        let d_freq = 0.25_f64;
        let so_bins: Vec<f64> = (0..n_so).map(|i| -5.0 + i as f64 * d_so).collect();
        let freq_bins: Vec<f64> = (0..n_freq).map(|i| i as f64 * d_freq).collect();

        let mut img = Array2::<f64>::zeros((n_so, n_freq));
        // (row_center, col_center, amplitude, sigma_row, sigma_col)
        let blobs = [
            (20.0, 25.0, 15.0, 4.0, 5.0),
            (58.0, 90.0, 11.0, 4.5, 5.0),
            (40.0, 60.0, 8.0, 3.5, 4.0),
        ];
        for (rc, cc, amp, sr, sc) in blobs {
            for r in 0..n_so {
                for c in 0..n_freq {
                    let dr = r as f64 - rc;
                    let dc = c as f64 - cc;
                    let g = (-(dr * dr) / (2.0 * sr * sr)
                        - (dc * dc) / (2.0 * sc * sc))
                        .exp();
                    img[[r, c]] += amp * g;
                }
            }
        }
        // Small positive floor so trim has a finite shift baseline.
        for v in img.iter_mut() {
            *v += 0.01;
        }
        (img, so_bins, freq_bins)
    }

    #[test]
    fn extract_hist_peaks_finds_blobs_deterministically() {
        let (img, so_bins, freq_bins) = synthetic_hist();
        // Generous thresholds so all three planted blobs survive.
        let params = HistPeakParams {
            merge_thresh: 0.1,
            max_merges: f64::INFINITY,
            trim_vol: 0.95,
            trim_shift: f64::NAN,
            dur_min: 0.0,
            bw_min: 0.0,
            height_min: 1.0,
        };

        let labels = extract_hist_peaks(
            img.view(),
            &so_bins,
            &freq_bins,
            &params,
        );
        assert_eq!(labels.dim(), img.dim());

        let n_regions = {
            let mut s: BTreeSet<i64> = BTreeSet::new();
            for &v in labels.iter() {
                if v > 0 {
                    s.insert(v);
                }
            }
            s
        };
        // Three planted blobs → at least three regions, all non-zero.
        assert!(
            n_regions.len() >= 3,
            "expected >= 3 regions, got {}",
            n_regions.len()
        );
        // Labels are dense 1..=N.
        let expected: BTreeSet<i64> = (1..=n_regions.len() as i64).collect();
        assert_eq!(n_regions, expected, "labels must be dense 1..=N");

        // Determinism: repeated runs produce a byte-identical label image.
        for _ in 0..5 {
            let again = extract_hist_peaks(
                img.view(),
                &so_bins,
                &freq_bins,
                &params,
            );
            assert_eq!(again, labels, "extract_hist_peaks is not deterministic");
        }

        // Each planted blob center should land inside some region.
        for (rc, cc) in [(20usize, 25usize), (58, 90), (40, 60)] {
            assert!(
                labels[[rc, cc]] > 0,
                "blob center ({}, {}) lost its label",
                rc,
                cc
            );
        }
    }

    #[test]
    fn extract_hist_peaks_filters_by_height() {
        let (img, so_bins, freq_bins) = synthetic_hist();
        // height_min above the weakest blob's amplitude (8) → at most two
        // regions survive.
        let params = HistPeakParams {
            height_min: 9.0,
            dur_min: 0.0,
            bw_min: 0.0,
            ..HistPeakParams::default()
        };
        let labels =
            extract_hist_peaks(img.view(), &so_bins, &freq_bins, &params);
        let n = labels.iter().copied().collect::<BTreeSet<_>>();
        let n_regions = n.iter().filter(|&&v| v > 0).count();
        assert!(
            n_regions <= 2,
            "height filter should drop the weak blob, got {} regions",
            n_regions
        );
    }

    #[test]
    fn extract_hist_peaks_empty_on_flat_input() {
        // A flat histogram has no peaks → no regions survive the height cut.
        let img = Array2::<f64>::from_elem((30, 40), 1.0);
        let so_bins: Vec<f64> = (0..30).map(|i| i as f64).collect();
        let freq_bins: Vec<f64> = (0..40).map(|i| i as f64).collect();
        let labels = extract_hist_peaks(
            img.view(),
            &so_bins,
            &freq_bins,
            &HistPeakParams::default(),
        );
        let n_regions = labels.iter().filter(|&&v| v > 0).count();
        assert_eq!(n_regions, 0, "flat input should yield no peak regions");
    }

    #[test]
    fn label_to_rgb_jet_background_and_shape() {
        let mut labels = Array2::<i64>::zeros((4, 5));
        labels[[1, 1]] = 1;
        labels[[1, 2]] = 1;
        labels[[2, 3]] = 2;
        let rgb = label_to_rgb_jet(labels.view());
        assert_eq!(rgb.dim(), (4, 5, 3));
        // Background pixel → black.
        assert_eq!([rgb[[0, 0, 0]], rgb[[0, 0, 1]], rgb[[0, 0, 2]]], [0, 0, 0]);
        // Labeled pixels → non-black.
        let p1 = [rgb[[1, 1, 0]], rgb[[1, 1, 1]], rgb[[1, 1, 2]]];
        assert_ne!(p1, [0, 0, 0], "label 1 should not be black");
        // Same label → same color.
        let p1b = [rgb[[1, 2, 0]], rgb[[1, 2, 1]], rgb[[1, 2, 2]]];
        assert_eq!(p1, p1b, "same label must map to same color");
        // Distinct labels → distinct colors.
        let p2 = [rgb[[2, 3, 0]], rgb[[2, 3, 1]], rgb[[2, 3, 2]]];
        assert_ne!(p1, p2, "distinct labels should differ in color");
    }

    #[test]
    fn jet_color_endpoints() {
        // jet(0) is blue-ish, jet(1) is red-ish.
        let lo = jet_color(0.0);
        let hi = jet_color(1.0);
        assert!(lo[2] > lo[0], "jet at 0 should be blue-dominant");
        assert!(hi[0] > hi[2], "jet at 1 should be red-dominant");
    }
}
