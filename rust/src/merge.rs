//! Iterative max-weight merge loop.
//!
//! Symmetric edge-weight rule matching current MATLAB edgeWeightEqual.m.
//! Stops when the max remaining weight ≤ merge_thresh, or `max_merges`
//! iterations reached.

use crate::adjacency::{Graph, Region};
use ndarray::{Array2, ArrayView2};
use std::collections::BTreeSet;

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
    // Compute initial weights — store in a HashMap keyed by sorted (a, b).
    // Find max by linear scan each iteration (simple + correct; can replace
    // with a heap later if this becomes the bottleneck).
    let mut weights: std::collections::HashMap<(i32, i32), f64> =
        std::collections::HashMap::new();

    // seed weights: for each region's neighbors
    for slot in 0..graph.regions.len() {
        let ra_label = match &graph.regions[slot] {
            Some(r) => r.label,
            None => continue,
        };
        let nbrs: Vec<i32> = graph.regions[slot].as_ref().unwrap()
            .neighbors.iter().copied().collect();
        for n in nbrs {
            if n <= ra_label {
                continue; // each edge once
            }
            let sb = match graph.label_to_slot.get(&n).copied() {
                Some(s) => s,
                None => continue,
            };
            let ra = graph.regions[slot].as_ref().unwrap();
            let rb = graph.regions[sb].as_ref().unwrap();
            if let Some(w) = edge_weight(ra, rb, values_slice) {
                weights.insert((ra_label.min(n), ra_label.max(n)), w);
            }
        }
    }

    let mut n_merges = 0u64;
    let max_merges_u = if max_merges.is_finite() { max_merges as u64 } else { u64::MAX };

    loop {
        if n_merges >= max_merges_u { break; }
        // find max weight edge
        let mut best: Option<(i32, i32, f64)> = None;
        for (&(a, b), &w) in weights.iter() {
            if let Some((_, _, bw)) = best {
                if w > bw {
                    best = Some((a, b, w));
                }
            } else {
                best = Some((a, b, w));
            }
        }
        let (a_label, b_label, w) = match best {
            Some(t) => t,
            None => break,
        };
        if w < merge_thresh {
            break;
        }

        // Merge b into a (lower label wins as destination, arbitrary choice).
        let (dst_label, src_label) = if a_label < b_label { (a_label, b_label) } else { (b_label, a_label) };
        let dst_slot = graph.label_to_slot[&dst_label];
        let src_slot = graph.label_to_slot[&src_label];
        let src_region = graph.take(src_slot).unwrap();
        let absorbed_neighbors: Vec<i32> = src_region.neighbors.iter().copied().collect();
        {
            let dst_region = graph.regions[dst_slot].as_mut().unwrap();
            merge_regions(dst_region, src_region);
        }
        // Remove src's presence in label_to_slot
        graph.label_to_slot.remove(&src_label);
        // Remove edges involving src_label, and stale self-loops
        weights.retain(|&(a, b), _| a != src_label && b != src_label);
        // Remove edges where both endpoints are src (shouldn't exist, defensive)
        // For each neighbour of src that's not dst, recompute weight vs dst.
        for nb in &absorbed_neighbors {
            if *nb == dst_label { continue; }
            // Remove stale edge between dst and nb (will re-add with new weight)
            let key_old = (dst_label.min(*nb), dst_label.max(*nb));
            weights.remove(&key_old);
            // Is nb still alive?
            let nb_slot = match graph.label_to_slot.get(nb).copied() {
                Some(s) => s,
                None => continue,
            };
            let ra = graph.regions[dst_slot].as_ref().unwrap();
            let rb = graph.regions[nb_slot].as_ref().unwrap();
            if let Some(w) = edge_weight(ra, rb, values_slice) {
                weights.insert(key_old, w);
                // Also record dst in nb's neighbors and vice versa
                graph.regions[nb_slot].as_mut().unwrap().neighbors.insert(dst_label);
                graph.regions[dst_slot].as_mut().unwrap().neighbors.insert(*nb);
            } else {
                // No intersect → drop adjacency
                graph.regions[nb_slot].as_mut().unwrap().neighbors.remove(&dst_label);
                graph.regions[dst_slot].as_mut().unwrap().neighbors.remove(nb);
            }
        }
        // Also recompute weights for all other neighbors of dst (not just src's),
        // because dst's border changed via symdiff.
        let other_nbrs: Vec<i32> = graph.regions[dst_slot].as_ref().unwrap()
            .neighbors.iter().copied().collect();
        for nb in other_nbrs {
            if nb == dst_label { continue; }
            let key = (dst_label.min(nb), dst_label.max(nb));
            let nb_slot = match graph.label_to_slot.get(&nb).copied() {
                Some(s) => s, None => continue,
            };
            let ra = graph.regions[dst_slot].as_ref().unwrap();
            let rb = graph.regions[nb_slot].as_ref().unwrap();
            if let Some(w) = edge_weight(ra, rb, values_slice) {
                weights.insert(key, w);
            } else {
                weights.remove(&key);
                graph.regions[nb_slot].as_mut().unwrap().neighbors.remove(&dst_label);
                graph.regions[dst_slot].as_mut().unwrap().neighbors.remove(&nb);
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
