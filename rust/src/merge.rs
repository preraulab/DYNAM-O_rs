//! Iterative max-weight merge loop.
//!
//! Symmetric edge-weight rule matching current MATLAB edgeWeightEqual.m.
//! Stops when the max remaining weight ≤ merge_thresh, or `max_merges`
//! iterations reached.

use crate::adjacency::{Graph, Region};
use ndarray::{Array2, ArrayView2};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

/// Heap entry for the max-weight merge loop. We pop highest weight first;
/// on ties, lower insertion_order wins (matches MATLAB's edge-index
/// tie-break where earlier edges are selected first). Stale entries
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
        // Tie-break: earlier insertion_order wins — invert so BinaryHeap pops it first.
        other.insertion_order.cmp(&self.insertion_order)
    }
}

#[derive(Clone, Copy)]
struct EdgeMeta {
    insertion_order: u64,
    generation: u64,
}

/// Two-pointer intersection of two sorted Vec<u32>. Returns (count, max_value_at_intersect)
/// where max_value_at_intersect = max of `values` at the intersecting indices.
fn sorted_intersect_max(a: &[u32], b: &[u32], values: &[f64]) -> Option<f64> {
    let (mut i, mut j) = (0usize, 0usize);
    let mut mx: Option<f64> = None;
    while i < a.len() && j < b.len() {
        let (ai, bj) = (a[i], b[j]);
        if ai == bj {
            let v = values[ai as usize];
            mx = Some(match mx {
                Some(m) if m >= v => m,
                _ => v,
            });
            i += 1;
            j += 1;
        } else if ai < bj {
            i += 1;
        } else {
            j += 1;
        }
    }
    mx
}

fn sorted_min(a: &[u32], values: &[f64]) -> f64 {
    let mut m = f64::INFINITY;
    for &idx in a {
        let v = values[idx as usize];
        if v < m {
            m = v;
        }
    }
    m
}

fn sorted_max_over_two(a: &[u32], b: &[u32], values: &[f64]) -> f64 {
    let mut m = f64::NEG_INFINITY;
    for &idx in a {
        let v = values[idx as usize];
        if v > m {
            m = v;
        }
    }
    for &idx in b {
        let v = values[idx as usize];
        if v > m {
            m = v;
        }
    }
    m
}

/// Symmetric merge weight (matches pyDYNAM-O's edge_weight after bug fix).
/// Returns None when the two regions' border intersection is empty (= not
/// actually adjacent — "diagonal neighbor" case).
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
    // neighbors: union, excluding the absorbed src label
    for n in src.neighbors {
        dst.neighbors.insert(n);
    }
    dst.neighbors.remove(&src.label);
    dst.neighbors.remove(&dst.label);
}

fn sorted_union(a: &[u32], b: &[u32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(a.len() + b.len());
    let (mut i, mut j) = (0usize, 0usize);
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

fn sorted_symdiff(a: &[u32], b: &[u32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(a.len() + b.len());
    let (mut i, mut j) = (0usize, 0usize);
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
    let (interior_only, graph) = run_inner(labels, values, merge_thresh, max_merges)?;
    let h = interior_only.nrows();
    let w = interior_only.ncols();
    // Build with-borders variant. Iterate regions sorted by label (1..N),
    // painting interior first then border, so shared border pixels claimed
    // by two surviving regions get the higher-label one's paint last — matches
    // MATLAB's `for ii=1..N; Ldata(rgn{ii})=ii; end` iteration order.
    let mut with_borders = Array2::<i32>::zeros((h, w));
    {
        let wb = with_borders.as_slice_mut().unwrap();
        let mut ordered: Vec<(i32, usize)> = (0..graph.regions.len())
            .filter_map(|slot| graph.regions[slot].as_ref().map(|r| (r.label, slot)))
            .collect();
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
    Ok(run_inner(labels, values, merge_thresh, max_merges)?.0)
}

fn run_inner(
    labels: ArrayView2<i64>,
    values: ArrayView2<f64>,
    merge_thresh: f64,
    max_merges: f64,
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

    // Helper closures would need to borrow mutably; use local macros-via-fn instead.
    // We inline the insert/update logic at the three places it's needed.

    // --- Seed: for each edge (ra_label, n) with n > ra_label, compute weight. ---
    // Iterate slots in ascending label order so the initial insertion_order
    // reflects label order (matches MATLAB's edge enumeration, giving stable
    // first-index tie-break on identical weights).
    let mut ordered_slots: Vec<usize> = (0..graph.regions.len())
        .filter(|&s| graph.regions[s].is_some())
        .collect();
    ordered_slots.sort_by_key(|&s| graph.regions[s].as_ref().unwrap().label);
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
            let ra = graph.regions[*slot].as_ref().unwrap();
            let rb = graph.regions[sb].as_ref().unwrap();
            if let Some(w) = edge_weight(ra, rb, values_slice) {
                let key = (ra_label, n);
                let meta = EdgeMeta { insertion_order: next_order, generation: 0 };
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
        if e.weight < merge_thresh { break; }

        let (a_label, b_label) = e.edge;
        // Lower label becomes the destination (arbitrary convention, preserved
        // from original code so bisection fixtures still match).
        let (dst_label, src_label) =
            if a_label < b_label { (a_label, b_label) } else { (b_label, a_label) };
        let dst_slot = graph.label_to_slot[&dst_label];
        let src_slot = graph.label_to_slot[&src_label];
        let src_region = graph.take(src_slot).unwrap();
        let absorbed_neighbors: Vec<i32> = src_region.neighbors.iter().copied().collect();
        {
            let dst_region = graph.regions[dst_slot].as_mut().unwrap();
            merge_regions(dst_region, src_region);
        }
        graph.label_to_slot.remove(&src_label);

        // Remove every edge touching src_label. Iterate src_label's neighbors
        // (known from absorbed_neighbors) — this replaces the old O(|E|) retain.
        for nb in &absorbed_neighbors {
            let key = (src_label.min(*nb), src_label.max(*nb));
            if edge_weights.remove(&key).is_some() {
                if let Some(m) = edge_meta.get_mut(&key) {
                    m.generation += 1;  // invalidate any stale heap entries
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
                    let meta = edge_meta.entry(key).or_insert(EdgeMeta {
                        insertion_order: {
                            let i = next_order; next_order += 1; i
                        },
                        generation: 0,
                    });
                    meta.generation += 1;
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

    #[test]
    fn trivially_merges() {
        // Two regions separated by a 0-line; large thresh prevents merging.
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
