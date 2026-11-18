//! Iterative max-weight merge loop (MATLAB `mergeWshedSegment.m`).
//!
//! Symmetric edge-weight rule matching current MATLAB edgeWeightEqual.m.
//! Stops when the max remaining weight ≤ merge_thresh, or `max_merges`
//! iterations reached.
//!
//! Each merge also absorbs encircled regions, as MATLAB `mergeRegions.m`
//! does: after region `b` joins region `a`, any former neighbor of `b`
//! whose only remaining neighbor is `a` joins `a` too, whatever its own
//! merge weight. That is usually a region surrounded by `a`, and can also
//! be one at the segment edge that touches nothing else.

use crate::adjacency::{Graph, Region};
use ndarray::{Array2, ArrayView2};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

/// Heap entry for the max-weight merge loop. We pop highest weight first;
/// on ties, lower insertion_order wins. `insertion_order` is the edge's row
/// position in MATLAB's edge table `ematr`, where `max` returns the first
/// row among equal weights. Stale entries
/// (generation != edge_meta[edge].generation) are discarded on pop.
#[derive(Clone, Copy)]
struct HeapEntry {
    weight: f64,
    insertion_order: u64,
    generation: u64,
    edge: (i32, i32),
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.weight == other.weight && self.insertion_order == other.insertion_order
    }
}
impl Eq for HeapEntry {}
impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
}
impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Primary key: higher weight first (BinaryHeap is max-heap → Greater).
        // NaN-safe: partial_cmp of f64 returns None for NaN; treat as Equal
        // and fall back to tie-break. Real weights are always finite here.
        let w = self.weight.partial_cmp(&other.weight).unwrap_or(Ordering::Equal);
        if w != Ordering::Equal {
            return w;
        }
        // Tie-break: earlier insertion_order wins. Invert so BinaryHeap pops it first.
        other.insertion_order.cmp(&self.insertion_order)
    }
}

/// One row of MATLAB's `ematr` as the de-duplication step sees it: where
/// the row sits in the table and which way it points.
#[derive(Clone, Copy)]
struct TableRow {
    /// Row position in `ematr`. A smaller position is earlier in the table.
    position: u64,
    /// The row as MATLAB stores it, `[merge_to, merge_from]`.
    oriented: (i32, i32),
}

/// Bookkeeping for one edge, mirroring its row in MATLAB's `ematr`.
#[derive(Clone, Copy)]
struct EdgeMeta {
    /// Row position in `ematr`. After every merge MATLAB moves all rows of
    /// the surviving region to the end of the table, sorted by label pair,
    /// so those edges get fresh, larger positions.
    insertion_order: u64,
    generation: u64,
    /// The row as MATLAB stores it, `[merge_to, merge_from]`. Rows start as
    /// `[lower, higher]`, but relabeling an absorbed region in place can
    /// leave the higher label first, and that row then merges the lower
    /// label INTO the higher one.
    oriented: (i32, i32),
}

/// Two-pointer intersection of two sorted `Vec<u32>`. Returns `None` when
/// the intersection is empty (regions not actually adjacent). Otherwise
/// `Some(max)` of `values` over the intersecting indices. NaN values are
/// ignored (MATLAB `max` is omitnan). An all-NaN intersection → `Some(NaN)`.
fn sorted_intersect_max(a: &[u32], b: &[u32], values: &[f64]) -> Option<f64> {
    let mut i = 0usize;
    let mut j = 0usize;
    let mut found_any = false;
    let mut mx = f64::NEG_INFINITY;
    let mut any_finite = false;
    while i < a.len() && j < b.len() {
        let ai = a[i];
        let bj = b[j];
        if ai == bj {
            found_any = true;
            let v = values[ai as usize];
            if !v.is_nan() {
                any_finite = true;
                if v > mx {
                    mx = v;
                }
            }
            i += 1;
            j += 1;
        } else if ai < bj {
            i += 1;
        } else {
            j += 1;
        }
    }
    if !found_any {
        None
    } else if any_finite {
        Some(mx)
    } else {
        Some(f64::NAN)
    }
}

/// Min over `values` at the given indices, ignoring NaN (MATLAB `min`
/// omitnan). All-NaN (or empty) → NaN.
fn sorted_min(a: &[u32], values: &[f64]) -> f64 {
    let mut m = f64::INFINITY;
    let mut any = false;
    for &idx in a {
        let v = values[idx as usize];
        if !v.is_nan() {
            any = true;
            if v < m {
                m = v;
            }
        }
    }
    if any {
        m
    } else {
        f64::NAN
    }
}

/// Max over `values` at the union of two index sets, ignoring NaN (MATLAB
/// `max` omitnan). All-NaN (or empty) → NaN.
fn sorted_max_over_two(a: &[u32], b: &[u32], values: &[f64]) -> f64 {
    let mut m = f64::NEG_INFINITY;
    let mut any = false;
    for &idx in a.iter().chain(b.iter()) {
        let v = values[idx as usize];
        if !v.is_nan() {
            any = true;
            if v > m {
                m = v;
            }
        }
    }
    if any {
        m
    } else {
        f64::NAN
    }
}

/// Symmetric merge weight (matches pyDYNAM-O's edge_weight after bug fix).
/// Returns None when the two regions' border intersection is empty (= not
/// actually adjacent, the "diagonal neighbor" case).
fn edge_weight(
    ra: &Region,
    rb: &Region,
    values: &[f64],
) -> Option<f64> {
    let a_ij_max = sorted_intersect_max(&ra.border, &rb.border, values)?;
    let min_bi = sorted_min(&ra.border, values);
    let min_bj = sorted_min(&rb.border, values);
    let i_max = sorted_max_over_two(&ra.interior, &ra.border, values);
    let j_max = sorted_max_over_two(&rb.interior, &rb.border, values);
    let w_ij = -min_bi - j_max;
    let w_ji = -min_bj - i_max;
    let w_max = 2.0 * a_ij_max + w_ij.max(w_ji);
    Some(w_max)
}

/// Merge region `src` into `dst` (by label). Borders: symmetric difference
/// on sorted Vec<u32> via merge-based walk.
fn merge_regions(dst: &mut Region, src: Region) {
    dst.interior = sorted_union(&dst.interior, &src.interior);
    dst.border = sorted_symdiff(&dst.border, &src.border);
    // `all_px` (MATLAB `regions{a}=unique([regions{a};regions{b}])`):
    // pure union. Keeps the now-interior 0-lines the symdiff border drops.
    dst.all_px = sorted_union(&dst.all_px, &src.all_px);
    // neighbors: union, excluding the absorbed src label
    for n in src.neighbors {
        dst.neighbors.insert(n);
    }
    dst.neighbors.remove(&src.label);
    dst.neighbors.remove(&dst.label);
}

/// Union of two sorted lists, sorted, with each shared value kept once.
fn sorted_union(a: &[u32], b: &[u32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(a.len() + b.len());
    let mut i = 0usize;
    let mut j = 0usize;
    while i < a.len() && j < b.len() {
        if a[i] < b[j] {
            out.push(a[i]);
            i += 1;
        } else if a[i] > b[j] {
            out.push(b[j]);
            j += 1;
        } else {
            out.push(a[i]);
            i += 1;
            j += 1;
        }
    }
    out.extend_from_slice(&a[i..]);
    out.extend_from_slice(&b[j..]);
    out
}

/// Values that are in exactly one of two sorted lists, sorted.
fn sorted_symdiff(a: &[u32], b: &[u32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(a.len() + b.len());
    let mut i = 0usize;
    let mut j = 0usize;
    while i < a.len() && j < b.len() {
        if a[i] < b[j] {
            out.push(a[i]);
            i += 1;
        } else if a[i] > b[j] {
            out.push(b[j]);
            j += 1;
        } else {
            i += 1;
            j += 1;
        }
    }
    out.extend_from_slice(&a[i..]);
    out.extend_from_slice(&b[j..]);
    out
}

/// Run merge and return BOTH the interior-only label image (for masking)
/// and the interior+border label image (for dur/bw filter bbox computation).
/// Matches MATLAB's dual semantics: extractTFPeaks.m paints interior+border
/// for bbox (`Ldata(ii_pixels)=ii` with ii_pixels=interior∪border), but the
/// maskSpectrogram inlining in export_bisect_intermediates.m zeros borders
/// back out (`spect2_masked(border_inds)=0`).
pub fn run_with_borders(
    labels: ArrayView2<i64>,
    values: ArrayView2<f64>,
    merge_thresh: f64,
    max_merges: f64,
) -> Result<(Array2<i32>, Array2<i32>), String> {
    let (interior_only, graph) =
        run_inner(labels, values, merge_thresh, max_merges, MergeRules::MatlabReference)?;
    let h = interior_only.nrows();
    let w = interior_only.ncols();
    // Build with-borders variant. Iterate regions sorted by label (1..N),
    // painting interior first then border, so shared border pixels claimed
    // by two surviving regions get the higher-label one's paint last. Matches
    // MATLAB's `for ii=1..N; Ldata(rgn{ii})=ii; end` iteration order.
    let mut with_borders = Array2::<i32>::zeros((h, w));
    {
        let wb = with_borders.as_slice_mut().unwrap();
        // Every live region as (label, slot), to be painted in label order.
        let mut ordered: Vec<(i32, usize)> = Vec::new();
        for slot in 0..graph.regions.len() {
            if let Some(region) = &graph.regions[slot] {
                ordered.push((region.label, slot));
            }
        }
        ordered.sort_by_key(|&(lbl, _)| lbl);
        for (_, slot) in ordered {
            let r = graph.regions[slot].as_ref().unwrap();
            for &p in &r.interior {
                wb[p as usize] = r.label;
            }
            for &p in &r.border {
                wb[p as usize] = r.label;
            }
        }
    }
    Ok((interior_only, with_borders))
}

pub fn run(
    labels: ArrayView2<i64>,
    values: ArrayView2<f64>,
    merge_thresh: f64,
    max_merges: f64,
) -> Result<Array2<i32>, String> {
    Ok(run_inner(labels, values, merge_thresh, max_merges, MergeRules::MatlabReference)?.0)
}

/// Run merge and return a label image painted from each region's
/// `all_px` (= MATLAB `regions{}` = `interior ∪ border`, union-merged).
/// This is the pixel set `extracthistpeaks.m` carries into the dur/bw
/// cut, `trimWshedRegions`, and `computePeakStatsTable`. Regions are
/// painted in ascending-label order so a pixel shared by two regions'
/// `all_px` lists takes the higher label, matching MATLAB's
/// `for ii=1:N; Ldata(regions{ii})=ii; end`.
///
/// This entry point is the SOPH mode-peak finder's and merges under
/// [`MergeRules::PairwiseOnly`]. Its regions seed the parametric fit, so
/// its procedure is held fixed separately from TF-peak extraction.
pub fn run_all_px(
    labels: ArrayView2<i64>,
    values: ArrayView2<f64>,
    merge_thresh: f64,
    max_merges: f64,
) -> Result<Array2<i32>, String> {
    let (interior_only, graph) =
        run_inner(labels, values, merge_thresh, max_merges, MergeRules::PairwiseOnly)?;
    let (h, w) = interior_only.dim();
    let mut out = Array2::<i32>::zeros((h, w));
    {
        let flat = out.as_slice_mut().unwrap();
        // Every live region as (label, slot), to be painted in label order.
        let mut ordered: Vec<(i32, usize)> = Vec::new();
        for slot in 0..graph.regions.len() {
            if let Some(region) = &graph.regions[slot] {
                ordered.push((region.label, slot));
            }
        }
        ordered.sort_by_key(|&(lbl, _)| lbl);
        for (_, slot) in ordered {
            let r = graph.regions[slot].as_ref().unwrap();
            for &p in &r.all_px {
                flat[p as usize] = r.label;
            }
        }
    }
    Ok(out)
}

/// Which merge procedure `run_inner` follows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MergeRules {
    /// MATLAB `mergeWshedSegment.m` / `mergeRegions.m` step for step:
    /// merge while the heaviest edge is strictly above the threshold, take
    /// the first row of the edge table among equal weights, merge the
    /// row's second region into its first, and absorb regions whose only
    /// neighbor is the merged region. This is TF-peak extraction.
    MatlabReference,
    /// The SOPH mode-peak finder's merge (`histpeaks.rs`), which seeds the
    /// parametric fit: merge while the heaviest edge is at or above the
    /// threshold, the higher label always merges into the lower, an edge
    /// keeps its original position for tie-breaking, and nothing is
    /// absorbed beyond the two regions of the edge.
    PairwiseOnly,
}

/// One region left after merging, as pixel lists (flat row-major indices).
#[derive(Debug, Clone)]
pub struct MergedRegion {
    /// Label the region carries after merging (the label of the region it
    /// merged into).
    pub label: i32,
    /// MATLAB `regions{}`: every pixel of the region, interior and border.
    pub all_px: Vec<u32>,
    /// MATLAB `borders{}`: the region's current border pixels.
    pub border: Vec<u32>,
}

/// Run merge and return the surviving regions themselves, in ascending
/// label order. A painted label image cannot represent the border pixels
/// that two regions share, so this is the form to compare against MATLAB's
/// `regions` / `borders` cell arrays after `mergeWshedSegment`.
pub fn run_regions(
    labels: ArrayView2<i64>,
    values: ArrayView2<f64>,
    merge_thresh: f64,
    max_merges: f64,
) -> Result<Vec<MergedRegion>, String> {
    let (_, graph) =
        run_inner(labels, values, merge_thresh, max_merges, MergeRules::MatlabReference)?;
    let mut merged: Vec<MergedRegion> = Vec::new();
    for slot in 0..graph.regions.len() {
        if let Some(region) = &graph.regions[slot] {
            merged.push(MergedRegion {
                label: region.label,
                all_px: region.all_px.clone(),
                border: region.border.clone(),
            });
        }
    }
    merged.sort_by_key(|region| region.label);
    Ok(merged)
}

/// MATLAB `prctile(x, p)` for a vector. The default "exact" method.
/// The i-th (0-based) sorted value sits at percentile `100*(i+0.5)/n`;
/// targets below/above the first/last sample clamp, others interpolate
/// linearly. NaNs are ignored, matching MATLAB. Empty input → 0.0.
fn matlab_prctile(values: impl Iterator<Item = f64>, p: f64) -> f64 {
    let mut xs: Vec<f64> = values.filter(|v| v.is_finite()).collect();
    if xs.is_empty() {
        return 0.0;
    }
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = xs.len();
    if n == 1 {
        return xs[0];
    }
    /// Percentile that sorted sample `i` of `n` sits at: the midpoint of
    /// its 1/n share, which is MATLAB `prctile`'s convention.
    fn percentile_of_sample(i: usize, n: usize) -> f64 {
        100.0 * (i as f64 + 0.5) / n as f64
    }
    if p <= percentile_of_sample(0, n) {
        return xs[0];
    }
    if p >= percentile_of_sample(n - 1, n) {
        return xs[n - 1];
    }
    // Find the bracketing samples.
    let mut i = 0;
    while i + 1 < n && percentile_of_sample(i + 1, n) < p {
        i += 1;
    }
    let percentile_below = percentile_of_sample(i, n);
    let percentile_above = percentile_of_sample(i + 1, n);
    let t = (p - percentile_below) / (percentile_above - percentile_below);
    xs[i] + t * (xs[i + 1] - xs[i])
}

fn run_inner(
    labels: ArrayView2<i64>,
    values: ArrayView2<f64>,
    merge_thresh: f64,
    max_merges: f64,
    rules: MergeRules,
) -> Result<(Array2<i32>, crate::adjacency::Graph), String> {
    let h = labels.nrows();
    let w = labels.ncols();
    let values_slice = values
        .as_slice()
        .ok_or("values must be C-contiguous")?;

    let mut graph = Graph::from_labels(labels);

    // Heap-based max-weight merge loop (2026-04-24 refactor from O(|E|)
    // linear scan + O(|E|) retain per iteration). Structure:
    //   edge_weights : live weight per edge (source of truth)
    //   edge_meta    : insertion_order + generation per edge (for heap tie-break
    //                  and lazy-deletion staleness check)
    //   heap         : max-heap of HeapEntry, may contain stale entries;
    //                  we pop-and-discard any whose generation mismatches meta
    // When an edge is removed or reweighted we bump its generation and push
    // a new heap entry (old entries become stale and are skipped on pop).
    let mut edge_weights: HashMap<(i32, i32), f64> = HashMap::new();
    let mut edge_meta:    HashMap<(i32, i32), EdgeMeta> = HashMap::new();
    let mut heap:         BinaryHeap<HeapEntry> = BinaryHeap::new();
    let mut next_order: u64 = 0;

    // Helper closures would need to borrow mutably. Use local macros-via-fn instead.
    // We inline the insert/update logic at the three places it's needed.

    // --- Seed: for each edge (ra_label, n) with n > ra_label, compute weight. ---
    // Iterate slots in ascending label order so the initial insertion_order
    // reflects label order (matches MATLAB's edge enumeration, giving stable
    // first-index tie-break on identical weights).
    let mut ordered_slots: Vec<usize> = (0..graph.regions.len())
        .filter(|&s| graph.regions[s].is_some())
        .collect();
    ordered_slots.sort_by_key(|&s| graph.regions[s].as_ref().unwrap().label);
    let mut n_pairs = 0u64;
    for slot in &ordered_slots {
        let ra_label = graph.regions[*slot].as_ref().unwrap().label;
        let mut nbrs: Vec<i32> = graph.regions[*slot].as_ref().unwrap()
            .neighbors.iter().copied().collect();
        nbrs.sort();
        for n in nbrs {
            if n <= ra_label { continue; }
            let sb = match graph.label_to_slot.get(&n).copied() {
                Some(s) => s, None => continue,
            };
            n_pairs += 1;
            let ra = graph.regions[*slot].as_ref().unwrap();
            let rb = graph.regions[sb].as_ref().unwrap();
            if let Some(w) = edge_weight(ra, rb, values_slice) {
                let key = (ra_label, n);
                let meta = EdgeMeta {
                    insertion_order: next_order,
                    generation: 0,
                    oriented: (ra_label, n),
                };
                next_order += 1;
                edge_weights.insert(key, w);
                edge_meta.insert(key, meta);
                heap.push(HeapEntry {
                    weight: w, insertion_order: meta.insertion_order,
                    generation: meta.generation, edge: key,
                });
            }
        }
    }

    // MATLAB `mergeWshedSegment.m:184-185`: a NaN `merge_thresh` means
    // "merge down to the 95th percentile of the initial edge weights".
    // `extracthistpeaks` callers (`param_basis_opts.m`) pass exactly that
    // value, `watershed_params(1) = nan`.
    let merge_thresh = if merge_thresh.is_nan() {
        matlab_prctile(edge_weights.values().copied(), 95.0)
    } else {
        merge_thresh
    };
    if std::env::var("HISTPEAKS_DEBUG").is_ok() {
        let n_finite = edge_weights.values().filter(|v| v.is_finite()).count();
        eprintln!(
            "[merge] {} neighbor-pairs, {} edges ({} finite), merge_thresh={:.6}",
            n_pairs,
            edge_weights.len(),
            n_finite,
            merge_thresh
        );
    }

    let mut n_merges = 0u64;
    let max_merges_u = if max_merges.is_finite() { max_merges as u64 } else { u64::MAX };

    loop {
        if n_merges >= max_merges_u { break; }

        // Pop the heap until we find a live entry.
        let popped = loop {
            match heap.pop() {
                None => break None,
                Some(e) => match edge_meta.get(&e.edge) {
                    Some(m) if m.generation == e.generation
                        && edge_weights.get(&e.edge).copied() == Some(e.weight) => {
                        break Some(e);
                    }
                    _ => continue,  // stale, discard
                },
            }
        };
        let e = match popped {
            Some(e) => e,
            None => break,  // no more live edges
        };
        let dst_label: i32;
        let src_label: i32;
        match rules {
            MergeRules::MatlabReference => {
                // MATLAB `mergeWshedSegment.m` loops `while max_wt >
                // merge_thresh`, so an edge exactly at the threshold does
                // not merge.
                if e.weight <= merge_thresh { break; }
                // MATLAB sets `mrg_to = ematr(row,1)` and `mrg_from = ematr(row,2)`.
                let oriented = edge_meta[&e.edge].oriented;
                dst_label = oriented.0;
                src_label = oriented.1;
            }
            MergeRules::PairwiseOnly => {
                // An edge exactly at the threshold still merges, and the
                // lower label is the destination.
                if e.weight < merge_thresh { break; }
                dst_label = e.edge.0.min(e.edge.1);
                src_label = e.edge.0.max(e.edge.1);
            }
        }
        let dst_slot = graph.label_to_slot[&dst_label];
        let src_slot = graph.label_to_slot[&src_label];
        let src_region = graph.take(src_slot).unwrap();
        let absorbed_neighbors: Vec<i32> = src_region.neighbors.iter().copied().collect();

        // Rows of src that are live right now, as (row position, row with
        // src relabeled to dst). MATLAB relabels these rows in place, and
        // one of them survives the de-duplication below when it sits
        // earlier in the table than dst's own row to the same neighbor.
        let mut src_rows: HashMap<i32, TableRow> = HashMap::new();
        for nb in &absorbed_neighbors {
            if *nb == dst_label {
                continue;
            }
            let key = (src_label.min(*nb), src_label.max(*nb));
            if edge_weights.contains_key(&key) {
                let meta = edge_meta[&key];
                let (row_to, row_from) = meta.oriented;
                let relabeled_to = if row_to == src_label { dst_label } else { row_to };
                let relabeled_from = if row_from == src_label { dst_label } else { row_from };
                let relabeled_row = TableRow {
                    position: meta.insertion_order,
                    oriented: (relabeled_to, relabeled_from),
                };
                src_rows.insert(*nb, relabeled_row);
            }
        }
        {
            let dst_region = graph.regions[dst_slot].as_mut().unwrap();
            merge_regions(dst_region, src_region);
        }
        graph.label_to_slot.remove(&src_label);

        // Remove every edge touching src_label. Iterate src_label's neighbors
        // (known from absorbed_neighbors). This replaces the old O(|E|) retain.
        for nb in &absorbed_neighbors {
            let key = (src_label.min(*nb), src_label.max(*nb));
            if edge_weights.remove(&key).is_some() {
                if let Some(m) = edge_meta.get_mut(&key) {
                    m.generation += 1;  // invalidate any stale heap entries
                }
            }
        }

        // Reconnect src's former neighbors to dst (MATLAB `mergeRegions.m`:
        // "Connect b's neighbors to a"), so every neighbor set names only
        // regions that still exist.
        for nb in &absorbed_neighbors {
            if *nb == dst_label {
                continue;
            }
            if let Some(nb_slot) = graph.label_to_slot.get(nb).copied() {
                let neighbor_region = graph.regions[nb_slot].as_mut().unwrap();
                neighbor_region.neighbors.remove(&src_label);
                neighbor_region.neighbors.insert(dst_label);
            }
        }

        // Absorb regions the merge has encircled (MATLAB `mergeRegions.m`,
        // "Look for regions encircled by the merge"). The candidates are
        // src's former neighbors other than dst, visited in ascending label
        // order. One whose only neighbor is now dst joins it: pixels by
        // union, border by symmetric difference.
        let mut encircled_candidates: Vec<i32> = Vec::new();
        if rules == MergeRules::MatlabReference {
            for nb in &absorbed_neighbors {
                if *nb != dst_label && *nb != src_label {
                    encircled_candidates.push(*nb);
                }
            }
        }
        encircled_candidates.sort_unstable();
        for candidate_label in encircled_candidates {
            let candidate_slot = match graph.label_to_slot.get(&candidate_label).copied() {
                Some(slot) => slot,
                None => continue,
            };
            let only_neighbor_is_dst = {
                let candidate = graph.regions[candidate_slot].as_ref().unwrap();
                candidate.neighbors.len() == 1 && candidate.neighbors.contains(&dst_label)
            };
            if !only_neighbor_is_dst {
                continue;
            }
            let encircled_region = graph.take(candidate_slot).unwrap();
            graph.label_to_slot.remove(&candidate_label);
            {
                let dst_region = graph.regions[dst_slot].as_mut().unwrap();
                merge_regions(dst_region, encircled_region);
            }
            // Its one edge (to dst) is gone.
            let key = (dst_label.min(candidate_label), dst_label.max(candidate_label));
            if edge_weights.remove(&key).is_some() {
                if let Some(m) = edge_meta.get_mut(&key) {
                    m.generation += 1;
                }
            }
        }

        // Recompute weights for every edge touching dst (dst's border changed
        // via symdiff). Walk dst's current neighbor set (which already has
        // src's former neighbors folded in by merge_regions).
        let dst_nbrs_now: Vec<i32> = graph.regions[dst_slot].as_ref().unwrap()
            .neighbors.iter().copied().collect();
        for nb in dst_nbrs_now {
            if nb == dst_label || nb == src_label { continue; }
            let key = (dst_label.min(nb), dst_label.max(nb));
            let nb_slot = match graph.label_to_slot.get(&nb).copied() {
                Some(s) => s, None => continue,
            };
            let ra = graph.regions[dst_slot].as_ref().unwrap();
            let rb = graph.regions[nb_slot].as_ref().unwrap();
            match edge_weight(ra, rb, values_slice) {
                Some(w) => {
                    // Which row survives MATLAB's de-duplication of
                    // `[dst nb]` / `[nb dst]`: the one earlier in the table,
                    // between dst's own row and src's relabeled row. It
                    // keeps its orientation.
                    let dst_row: Option<TableRow> = if edge_weights.contains_key(&key) {
                        let existing = edge_meta[&key];
                        Some(TableRow {
                            position: existing.insertion_order,
                            oriented: existing.oriented,
                        })
                    } else {
                        None
                    };
                    let src_row: Option<TableRow> = src_rows.get(&nb).copied();
                    let oriented = match (dst_row, src_row) {
                        (Some(dst), Some(src)) => {
                            if dst.position < src.position {
                                dst.oriented
                            } else {
                                src.oriented
                            }
                        }
                        (Some(dst), None) => dst.oriented,
                        (None, Some(src)) => src.oriented,
                        // A new edge with no row yet starts as `[lower, higher]`.
                        (None, None) => key,
                    };
                    let existing_meta: Option<EdgeMeta> = edge_meta.get(&key).copied();
                    let previous_generation = match existing_meta {
                        Some(existing) => existing.generation,
                        None => 0,
                    };
                    let new_position: u64;
                    match (rules, existing_meta) {
                        (MergeRules::PairwiseOnly, Some(existing)) => {
                            // A known edge keeps its position.
                            new_position = existing.insertion_order;
                        }
                        _ => {
                            // MATLAB moves dst's rows to the end of the
                            // table in label-pair order, which is the
                            // ascending-neighbor order of this loop, so
                            // each takes the next position. An edge new to
                            // the table does the same under either rule.
                            new_position = next_order;
                            next_order += 1;
                        }
                    }
                    let meta = edge_meta.entry(key).or_insert(EdgeMeta {
                        insertion_order: new_position,
                        generation: 0,
                        oriented,
                    });
                    meta.insertion_order = new_position;
                    meta.oriented = oriented;
                    meta.generation = previous_generation + 1;
                    edge_weights.insert(key, w);
                    heap.push(HeapEntry {
                        weight: w, insertion_order: meta.insertion_order,
                        generation: meta.generation, edge: key,
                    });
                    // Maintain neighbor sets (in case this was a new adjacency
                    // exposed by the merge of borders).
                    graph.regions[nb_slot].as_mut().unwrap().neighbors.insert(dst_label);
                    graph.regions[dst_slot].as_mut().unwrap().neighbors.insert(nb);
                }
                None => {
                    edge_weights.remove(&key);
                    if let Some(m) = edge_meta.get_mut(&key) {
                        m.generation += 1;
                    }
                    graph.regions[nb_slot].as_mut().unwrap().neighbors.remove(&dst_label);
                    graph.regions[dst_slot].as_mut().unwrap().neighbors.remove(&nb);
                }
            }
        }

        n_merges += 1;
    }
    if std::env::var("HISTPEAKS_DEBUG").is_ok() {
        eprintln!("[merge] n_merges={}", n_merges);
    }

    // Rebuild output label image: interior pixels only.
    // Note: MATLAB paints interior+border for dur/bw filter bbox computation
    // (via `Ldata(ii_pixels)=ii` where ii_pixels = rgn{ii} = interior∪border),
    // but zeros borders back out for masking
    // (`spect2_masked(border_inds)=0`). Pydynamo returns interior-only here
    // and uses `merge_segment_with_borders` to get the with-borders variant
    // for the filter path.
    let mut out = Array2::<i32>::zeros((h, w));
    {
        let out_flat = out.as_slice_mut().unwrap();
        for slot in 0..graph.regions.len() {
            if let Some(r) = &graph.regions[slot] {
                for &p in &r.interior {
                    out_flat[p as usize] = r.label;
                }
            }
        }
    }
    Ok((out, graph))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn no_edges_no_change() {
        let labels = array![[1i64, 1, 0, 2, 2], [1, 1, 0, 2, 2]];
        let data = array![[1.0f64, 2.0, 3.0, 4.0, 5.0], [1.0, 2.0, 3.0, 4.0, 5.0]];
        let out = run(labels.view(), data.view(), 100.0, f64::INFINITY).unwrap();
        // thresh 100 → no merging
        assert_eq!(out.iter().max().unwrap(), &2);
    }

    /// One ring and its center under the mode-peak finder's rules: the
    /// same single merge leaves the center alone, because nothing is
    /// absorbed beyond the two regions of the merged edge.
    #[test]
    fn pairwise_rules_do_not_absorb() {
        let (_, graph) = run_inner(
            ring_labels().view(),
            ring_values().view(),
            0.0,
            1.0,
            MergeRules::PairwiseOnly,
        )
        .unwrap();
        let mut surviving: Vec<i32> = Vec::new();
        for region in graph.regions.iter().flatten() {
            surviving.push(region.label);
        }
        surviving.sort_unstable();
        assert_eq!(surviving, vec![1, 3, 4, 5]);
    }

    #[test]
    fn an_edge_exactly_at_the_threshold_follows_each_rule() {
        // Two flat regions at 10 with a ridge at 5 between them: the edge
        // weight is 2*5 - 5 - 10 = -5. With the threshold at exactly -5,
        // MATLAB's strict `>` leaves both regions and the mode-peak
        // finder's rule merges them.
        let labels = array![[1i64, 1, 0, 2, 2], [1, 1, 0, 2, 2]];
        let data = array![[10.0f64, 10.0, 5.0, 10.0, 10.0], [10.0, 10.0, 5.0, 10.0, 10.0]];
        let matlab = run_regions(labels.view(), data.view(), -5.0, f64::INFINITY).unwrap();
        assert_eq!(matlab.len(), 2);
        let (_, graph) = run_inner(
            labels.view(),
            data.view(),
            -5.0,
            f64::INFINITY,
            MergeRules::PairwiseOnly,
        )
        .unwrap();
        assert_eq!(graph.regions.iter().flatten().count(), 1);
    }

    /// Two halves of a ring (regions 1 and 2) around a center (region 3),
    /// with watershed 0-lines between them. To the right, region 4 borders
    /// the ring and region 5 borders only region 4.
    fn ring_labels() -> Array2<i64> {
        array![
            [1i64, 1, 1, 0, 2, 2, 2, 0, 4, 4, 0, 5, 5],
            [1, 1, 0, 0, 0, 2, 2, 0, 4, 4, 0, 5, 5],
            [1, 0, 0, 3, 0, 0, 2, 0, 4, 4, 0, 5, 5],
            [1, 1, 0, 0, 0, 2, 2, 0, 4, 4, 0, 5, 5],
            [1, 1, 1, 0, 2, 2, 2, 0, 4, 4, 0, 5, 5],
        ]
    }

    /// Values that make the 1-2 edge the heaviest: the ring halves are high
    /// (10) with a high ridge between them at the top and bottom (9). The
    /// center is low (1) behind a low rim (0.5), and the walls in front of
    /// regions 4 and 5 are low (0.2).
    fn ring_values() -> Array2<f64> {
        array![
            [10.0f64, 10.0, 10.0, 9.0, 10.0, 10.0, 10.0, 0.2, 10.0, 10.0, 0.2, 10.0, 10.0],
            [10.0, 10.0, 0.5, 0.5, 0.5, 10.0, 10.0, 0.2, 10.0, 10.0, 0.2, 10.0, 10.0],
            [10.0, 0.5, 0.5, 1.0, 0.5, 0.5, 10.0, 0.2, 10.0, 10.0, 0.2, 10.0, 10.0],
            [10.0, 10.0, 0.5, 0.5, 0.5, 10.0, 10.0, 0.2, 10.0, 10.0, 0.2, 10.0, 10.0],
            [10.0, 10.0, 10.0, 9.0, 10.0, 10.0, 10.0, 0.2, 10.0, 10.0, 0.2, 10.0, 10.0],
        ]
    }

    #[test]
    fn merge_absorbs_a_region_whose_only_neighbor_is_the_merged_region() {
        // max_merges = 1: exactly one weight-driven merge happens (1 with
        // 2, the heaviest edge). Region 3 then has the merged ring as its
        // only neighbor and is absorbed in the same step. Region 4 also
        // neighbors region 5, so it survives, and so does region 5.
        let regions = run_regions(ring_labels().view(), ring_values().view(), 0.0, 1.0).unwrap();
        let labels: Vec<i32> = regions.iter().map(|r| r.label).collect();
        assert_eq!(labels, vec![1, 4, 5]);
        // The ring now owns the center pixel (row 2, column 3 → index 2*13+3).
        let ring = &regions[0];
        assert!(ring.all_px.contains(&29));
    }

    #[test]
    fn absorption_does_not_need_geometric_enclosure() {
        // MATLAB's test is "the only neighbor is a", not "surrounded by a".
        // Without region 5, region 4 sits at the image edge with the ring
        // as its only neighbor, so the same single merge absorbs it too.
        let labels = ring_labels().slice(ndarray::s![.., 0..10]).to_owned();
        let values = ring_values().slice(ndarray::s![.., 0..10]).to_owned();
        let regions = run_regions(labels.view(), values.view(), 0.0, 1.0).unwrap();
        let surviving: Vec<i32> = regions.iter().map(|r| r.label).collect();
        assert_eq!(surviving, vec![1]);
    }

    #[test]
    fn trivially_merges() {
        // Two regions separated by a 0-line. Large thresh prevents merging.
        // Small (negative) thresh forces all merges.
        let labels = array![[1i64, 1, 0, 2, 2], [1, 1, 0, 2, 2]];
        let data = array![[10.0f64, 10.0, 5.0, 10.0, 10.0], [10.0, 10.0, 5.0, 10.0, 10.0]];
        let out = run(labels.view(), data.view(), -1000.0, f64::INFINITY).unwrap();
        // After merging, one label remains
        let uniq: std::collections::BTreeSet<i32> =
            out.iter().copied().filter(|&v| v > 0).collect();
        assert_eq!(uniq.len(), 1);
    }
}
