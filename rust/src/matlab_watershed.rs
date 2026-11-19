//! Pure-Rust port of MATLAB's Image Processing Toolbox `watershed`.
//!
//! The algorithm is Vincent-Soille / Meyer flooding: find regional minima
//! as markers (`imregionalmin`), 8-connected-label the markers, then flood
//! the image from each marker in a FIFO priority queue. A popped pixel
//! that touches >1 distinct basin becomes a watershed line (label 0).
//!
//! Matching MATLAB's exact result requires three implementation choices:
//!   (a) column-major pixel indexing throughout (rows-first enumeration)
//!   (b) column-major neighbor scan order within the 3x3 footprint
//!   (c) FIFO tie-breaking within equal priorities (insertion order)

use ndarray::{Array2, ArrayView2};
use std::collections::BinaryHeap;

/// The 8 neighbors of a pixel as (row offset, column offset), in
/// column-major scan order (down each column, columns left to right).
const NEIGHBOR_OFFSETS: [(i32, i32); 8] = [
    (-1, -1), (0, -1), (1, -1),
    (-1,  0),          (1,  0),
    (-1,  1), (0,  1), (1,  1),
];

/// Regional minima of a 2D image, 8-connectivity.
///
/// A regional minimum is a connected component of equal-value pixels
/// whose external boundary has all values strictly greater than the
/// component value. Output: true at regional-min pixels, false elsewhere.
///
/// `img` is a column-major flat image with `h` rows and `w` columns.
fn imregionalmin_8conn(img: &[f64], h: usize, w: usize) -> Vec<bool> {
    let n = h * w;
    let mut visited = vec![false; n];
    let mut out = vec![false; n];
    // Stack of flat (column-major) indices still to visit in the flood fill.
    let mut stack: Vec<u32> = Vec::with_capacity(64);

    for start in 0..n {
        if visited[start] {
            continue;
        }
        let v = img[start];
        if v.is_nan() {
            visited[start] = true;
            continue;
        }

        // Flood fill over the plateau of equal-value pixels, noting any strictly lower neighbor.
        visited[start] = true;
        let mut plateau: Vec<u32> = vec![start as u32];
        let mut has_lower = false;
        stack.clear();
        stack.push(start as u32);
        while let Some(idx) = stack.pop() {
            let idx = idx as usize;
            let r = idx % h; // column-major
            let c = idx / h;
            for &(dr, dc) in &NEIGHBOR_OFFSETS {
                let nr = r as i32 + dr;
                let nc = c as i32 + dc;
                if nr < 0 || nc < 0 || nr >= h as i32 || nc >= w as i32 {
                    continue;
                }
                let nidx = (nc as usize) * h + (nr as usize);
                let nv = img[nidx];
                if nv < v {
                    has_lower = true;
                } else if nv == v {
                    if !visited[nidx] {
                        visited[nidx] = true;
                        plateau.push(nidx as u32);
                        stack.push(nidx as u32);
                    }
                }
            }
        }
        if !has_lower {
            for p in &plateau {
                out[*p as usize] = true;
            }
        }
    }
    out
}

/// 8-connected connected-component labeling on a boolean mask.
/// Labels start at 1 (0 = background). Iteration order matches MATLAB
/// (column-major scan, so labels are assigned column-by-column).
///
/// `mask` is column-major with `h` rows and `w` columns. Returns the
/// label image (same layout) and the number of labels.
fn bwlabel_8conn(mask: &[bool], h: usize, w: usize) -> (Vec<u32>, u32) {
    let n = h * w;
    let mut labels = vec![0u32; n];
    let mut next_label: u32 = 1;
    let mut stack: Vec<u32> = Vec::with_capacity(64);
    // Column-major scan: for each column, for each row.
    for c in 0..w {
        for r in 0..h {
            let idx = c * h + r;
            if !mask[idx] || labels[idx] != 0 {
                continue;
            }
            let label = next_label;
            next_label += 1;
            labels[idx] = label;
            stack.clear();
            stack.push(idx as u32);
            while let Some(k) = stack.pop() {
                let kr = (k as usize) % h;
                let kc = (k as usize) / h;
                for &(dr, dc) in &NEIGHBOR_OFFSETS {
                    let nr = kr as i32 + dr;
                    let nc = kc as i32 + dc;
                    if nr < 0 || nc < 0 || nr >= h as i32 || nc >= w as i32 {
                        continue;
                    }
                    let nidx = (nc as usize) * h + (nr as usize);
                    if mask[nidx] && labels[nidx] == 0 {
                        labels[nidx] = label;
                        stack.push(nidx as u32);
                    }
                }
            }
        }
    }
    (labels, next_label - 1)
}

/// One pixel waiting in the flooding queue.
#[derive(Clone, Copy, Debug, PartialEq)]
struct QueueItem {
    /// Flood level at which the pixel is processed (lowest first).
    priority: f64,
    /// Insertion counter. Among equal priorities the earliest goes first.
    order: u64,
    /// Column-major flat index of the pixel.
    idx: u32,
}

impl Eq for QueueItem {}

impl Ord for QueueItem {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // BinaryHeap pops the greatest item, so both comparisons are
        // reversed: lowest priority first, then earliest insertion.
        // A NaN priority compares as equal and falls through to the order.
        let by_priority = other
            .priority
            .partial_cmp(&self.priority)
            .unwrap_or(std::cmp::Ordering::Equal);
        by_priority.then(other.order.cmp(&self.order))
    }
}
impl PartialOrd for QueueItem {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Watershed transform of a 2D image with 8-connectivity. Returns a label
/// image of the same shape: 0 = watershed line, 1..N = basin labels.
/// Matches MATLAB's `watershed(A, 8)` output.
pub fn matlab_watershed_2d(img: ArrayView2<f64>) -> Array2<u16> {
    let (h, w) = img.dim();
    if h == 0 || w == 0 {
        return Array2::zeros((h, w));
    }

    // Column-major linear buffer
    let n = h * w;
    let mut img_cm: Vec<f64> = Vec::with_capacity(n);
    for c in 0..w {
        for r in 0..h {
            img_cm.push(img[[r, c]]);
        }
    }

    // 1) Regional minima → binary mask
    let min_mask = imregionalmin_8conn(&img_cm, h, w);

    // 2) Label minima (8-connectivity, column-major scan)
    let (labels_u32, _n_labels) = bwlabel_8conn(&min_mask, h, w);
    // Narrow to u16. A label above u16::MAX saturates at u16::MAX.
    let mut labels: Vec<u16> = Vec::with_capacity(n);
    for &label in labels_u32.iter() {
        if label > u16::MAX as u32 {
            labels.push(u16::MAX);
        } else {
            labels.push(label as u16);
        }
    }

    // 3) Seed the FIFO priority queue with all neighbors of labeled pixels
    let mut seen = vec![false; n];
    let mut heap: BinaryHeap<QueueItem> = BinaryHeap::with_capacity(n);
    let mut order_counter: u64 = 0;

    for idx in 0..n {
        if labels[idx] != 0 {
            seen[idx] = true;
            let r = idx % h;
            let c = idx / h;
            for &(dr, dc) in &NEIGHBOR_OFFSETS {
                let nr = r as i32 + dr;
                let nc = c as i32 + dc;
                if nr < 0 || nc < 0 || nr >= h as i32 || nc >= w as i32 {
                    continue;
                }
                let nidx = (nc as usize) * h + (nr as usize);
                if !seen[nidx] && labels[nidx] == 0 {
                    seen[nidx] = true;
                    order_counter += 1;
                    heap.push(QueueItem {
                        priority: img_cm[nidx],
                        order: order_counter,
                        idx: nidx as u32,
                    });
                }
            }
        }
    }

    // 4) Flooding loop: pop lowest priority and assign the label if 1 distinct,
    //    else stay 0 (watershed line). New neighbors enqueued with
    //    priority = max(neighbor_value, popped_priority).
    while let Some(item) = heap.pop() {
        let idx = item.idx as usize;
        let popped_priority = item.priority;
        let r = idx % h;
        let c = idx / h;

        // Collect 8-neighbors and check labels
        let mut basin_label: u16 = 0;
        let mut touches_two_basins = false;
        let mut neighbor_idx_list: [u32; 8] = [u32::MAX; 8];
        let mut n_neighbors = 0;
        for &(dr, dc) in &NEIGHBOR_OFFSETS {
            let nr = r as i32 + dr;
            let nc = c as i32 + dc;
            if nr < 0 || nc < 0 || nr >= h as i32 || nc >= w as i32 {
                continue;
            }
            let nidx = (nc as usize) * h + (nr as usize);
            neighbor_idx_list[n_neighbors] = nidx as u32;
            n_neighbors += 1;
            if touches_two_basins {
                continue;
            }
            let neighbor_label = labels[nidx];
            if neighbor_label != 0 {
                if basin_label != 0 && neighbor_label != basin_label {
                    touches_two_basins = true;
                } else {
                    basin_label = neighbor_label;
                }
            }
        }

        if !touches_two_basins {
            labels[idx] = basin_label;
            for k in 0..n_neighbors {
                let nidx = neighbor_idx_list[k] as usize;
                if !seen[nidx] {
                    seen[nidx] = true;
                    let neighbor_value = img_cm[nidx];
                    // Larger of the two. Written as a comparison (not
                    // f64::max) so that a NaN on either side yields the
                    // popped priority. f64::max would return the neighbor
                    // value when the popped priority is NaN.
                    let neighbor_priority = if neighbor_value > popped_priority {
                        neighbor_value
                    } else {
                        popped_priority
                    };
                    order_counter += 1;
                    heap.push(QueueItem {
                        priority: neighbor_priority,
                        order: order_counter,
                        idx: nidx as u32,
                    });
                }
            }
        }
    }

    // 5) Rebuild as (h, w) row-major output
    let mut out = Array2::<u16>::zeros((h, w));
    for c in 0..w {
        for r in 0..h {
            out[[r, c]] = labels[c * h + r];
        }
    }
    out
}

/// MATLAB `runWatershed.m`: `watershed(-data, 8)` followed by
/// `bwlabel(Ldata)`.
///
/// The `bwlabel` pass renumbers the regions: each 8-connected component of
/// nonzero pixels gets the next number in column-major scan order, so a
/// region is numbered by its first pixel going down the columns, where
/// `watershed` alone numbers it by its regional minimum. The numbering
/// matters downstream because the merge visits edges in label order. Two
/// basins that touch with no watershed line between them become one
/// region, as they do in MATLAB.
///
/// `negated` is the already negated image (`-data`).
pub fn matlab_run_watershed_2d(negated: ArrayView2<f64>) -> Array2<i64> {
    let basins = matlab_watershed_2d(negated);
    let (h, w) = basins.dim();

    // Nonzero mask in column-major order, the layout `bwlabel_8conn` scans.
    let mut nonzero_mask: Vec<bool> = Vec::with_capacity(h * w);
    for c in 0..w {
        for r in 0..h {
            nonzero_mask.push(basins[[r, c]] != 0);
        }
    }
    let (component_labels, _n_components) = bwlabel_8conn(&nonzero_mask, h, w);

    let mut relabeled = Array2::<i64>::zeros((h, w));
    for c in 0..w {
        for r in 0..h {
            relabeled[[r, c]] = component_labels[c * h + r] as i64;
        }
    }
    relabeled
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn simple_two_basin_image() {
        let img = array![
            [3.0, 2.0, 3.0, 4.0, 3.0, 2.0, 3.0],
            [2.0, 1.0, 2.0, 4.0, 2.0, 1.0, 2.0],
            [3.0, 2.0, 3.0, 4.0, 3.0, 2.0, 3.0],
        ];
        let labels = matlab_watershed_2d(img.view());
        let uniq: std::collections::BTreeSet<u16> =
            labels.iter().copied().filter(|&v| v > 0).collect();
        assert_eq!(uniq.len(), 2, "expected 2 basins, got {:?}", uniq);
    }

    #[test]
    fn run_watershed_numbers_regions_by_first_pixel_down_the_columns() {
        // Two basins side by side, split by a ridge in column 2. The left
        // basin's minimum is at the bottom of column 1, the right basin's
        // minimum is at the top of column 3. `watershed` numbers by minimum
        // position and `bwlabel` by first pixel, and both scans go down the
        // columns, so the left basin is 1 either way. Moving the right
        // basin's minimum does not change that.
        let negated = array![
            [-1.0f64, -1.0, 0.0, -9.0, -1.0],
            [-1.0, -1.0, 0.0, -1.0, -1.0],
            [-1.0, -9.0, 0.0, -1.0, -1.0],
        ];
        let relabeled = matlab_run_watershed_2d(negated.view());
        assert_eq!(relabeled[[0, 0]], 1);
        assert_eq!(relabeled[[0, 4]], 2);
        for r in 0..3 {
            assert_eq!(relabeled[[r, 2]], 0, "ridge column stays a watershed line");
        }
    }

    #[test]
    fn run_watershed_renumbers_when_the_minimum_order_differs_from_the_pixel_order() {
        // The top-left basin reaches column 0 but has its minimum far to
        // the right (column 4). The bottom basin has its minimum in column
        // 0. `watershed` alone numbers the bottom basin first (its minimum
        // comes first going down the columns). `bwlabel` numbers the top
        // basin first because its pixel at row 0, column 0 is scanned first.
        let negated = array![
            [-1.0f64, -1.0, -1.0, -1.0, -9.0],
            [-1.0, -1.0, -1.0, -1.0, -1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [-1.0, -1.0, -1.0, -1.0, -1.0],
            [-9.0, -1.0, -1.0, -1.0, -1.0],
        ];
        let plain = matlab_watershed_2d(negated.view());
        let relabeled = matlab_run_watershed_2d(negated.view());
        assert_eq!(plain[[4, 0]], 1, "watershed numbers the bottom basin first");
        assert_eq!(plain[[0, 0]], 2);
        assert_eq!(relabeled[[0, 0]], 1, "bwlabel numbers the top basin first");
        assert_eq!(relabeled[[4, 0]], 2);
    }
}
