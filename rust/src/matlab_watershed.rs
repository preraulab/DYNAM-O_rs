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
//!
//! Reference: MATLAB Coder-generated sources at
//!   ~/code/toolboxes/DYNAMO_dev/codegen/lib/matlab_watershed/
//! (watershed.c, FifoPriorityQueue.c, NeighborhoodProcessor.c, bwlabel.c).

use ndarray::{Array2, ArrayView2};
use std::cmp::Reverse;
use std::collections::BinaryHeap;

/// Regional minima of a 2D image, 8-connectivity.
///
/// A regional minimum is a connected component of equal-value pixels
/// whose external boundary has all values strictly greater than the
/// component value. Output: true at regional-min pixels, false elsewhere.
fn imregionalmin_8conn(img: &[f64], h: usize, w: usize) -> Vec<bool> {
    let n = h * w;
    let mut visited = vec![false; n];
    let mut out = vec![false; n];
    // BFS queue of flat (column-major) indices
    let mut stack: Vec<u32> = Vec::with_capacity(64);
    // Column-major neighbor offsets in (drow, dcol)
    const DRC: [(i32, i32); 8] = [
        (-1, -1), (0, -1), (1, -1),
        (-1,  0),          (1,  0),
        (-1,  1), (0,  1), (1,  1),
    ];

    for start in 0..n {
        if visited[start] { continue; }
        let v = img[start];
        if v.is_nan() { visited[start] = true; continue; }

        // BFS over plateau of equal-value pixels; detect any strictly-lower neighbor.
        visited[start] = true;
        let mut plateau: Vec<u32> = vec![start as u32];
        let mut has_lower = false;
        stack.clear();
        stack.push(start as u32);
        while let Some(idx) = stack.pop() {
            let idx = idx as usize;
            let r = idx % h;   // column-major
            let c = idx / h;
            for &(dr, dc) in &DRC {
                let nr = r as i32 + dr;
                let nc = c as i32 + dc;
                if nr < 0 || nc < 0 || nr >= h as i32 || nc >= w as i32 { continue; }
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
fn bwlabel_8conn(mask: &[bool], h: usize, w: usize) -> (Vec<u32>, u32) {
    let n = h * w;
    let mut labels = vec![0u32; n];
    let mut next_label: u32 = 1;
    let mut stack: Vec<u32> = Vec::with_capacity(64);
    const DRC: [(i32, i32); 8] = [
        (-1, -1), (0, -1), (1, -1),
        (-1,  0),          (1,  0),
        (-1,  1), (0,  1), (1,  1),
    ];
    // Column-major scan: for each column, for each row.
    for c in 0..w {
        for r in 0..h {
            let idx = c * h + r;
            if !mask[idx] || labels[idx] != 0 { continue; }
            let lab = next_label;
            next_label += 1;
            labels[idx] = lab;
            stack.clear();
            stack.push(idx as u32);
            while let Some(k) = stack.pop() {
                let kr = (k as usize) % h;
                let kc = (k as usize) / h;
                for &(dr, dc) in &DRC {
                    let nr = kr as i32 + dr;
                    let nc = kc as i32 + dc;
                    if nr < 0 || nc < 0 || nr >= h as i32 || nc >= w as i32 { continue; }
                    let nidx = (nc as usize) * h + (nr as usize);
                    if mask[nidx] && labels[nidx] == 0 {
                        labels[nidx] = lab;
                        stack.push(nidx as u32);
                    }
                }
            }
        }
    }
    (labels, next_label - 1)
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct QueueItem {
    priority: f64,
    order: u64,
    idx: u32,
}

impl Eq for QueueItem {}

impl Ord for QueueItem {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Min-heap by priority, then FIFO by order.
        other.priority.partial_cmp(&self.priority).unwrap_or(std::cmp::Ordering::Equal)
            .then(other.order.cmp(&self.order))
    }
}
impl PartialOrd for QueueItem {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Public entry point. Returns (h, w) u16 label image. 0 = watershed line,
/// 1..N = basin labels. Matches MATLAB's `watershed(A, 8)` output.
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
    let (labels_u32, n_labels) = bwlabel_8conn(&min_mask, h, w);
    let mut labels: Vec<u16> = labels_u32.iter()
        .map(|&v| {
            if v > u16::MAX as u32 { u16::MAX } else { v as u16 }
        })
        .collect();
    let _ = n_labels;

    // 3) Seed the FIFO priority queue with all neighbors of labelled pixels
    let mut seen = vec![false; n];
    let mut heap: BinaryHeap<QueueItem> = BinaryHeap::with_capacity(n);
    let mut order_counter: u64 = 0;

    const DRC: [(i32, i32); 8] = [
        (-1, -1), (0, -1), (1, -1),
        (-1,  0),          (1,  0),
        (-1,  1), (0,  1), (1,  1),
    ];

    for idx in 0..n {
        if labels[idx] != 0 {
            seen[idx] = true;
            let r = idx % h;
            let c = idx / h;
            for &(dr, dc) in &DRC {
                let nr = r as i32 + dr;
                let nc = c as i32 + dc;
                if nr < 0 || nc < 0 || nr >= h as i32 || nc >= w as i32 { continue; }
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

    // 4) Flooding loop: pop lowest priority; assign label if 1 distinct,
    //    else stay 0 (watershed line). New neighbors enqueued with
    //    priority = max(neighbor_value, popped_priority).
    while let Some(it) = heap.pop() {
        let idx = it.idx as usize;
        let p = it.priority;
        let r = idx % h;
        let c = idx / h;

        // Collect 8-neighbors and check labels
        let mut found_lab: u16 = 0;
        let mut watershed_state = false;
        let mut nbr_list: [u32; 8] = [u32::MAX; 8];
        let mut n_nbrs = 0;
        for &(dr, dc) in &DRC {
            let nr = r as i32 + dr;
            let nc = c as i32 + dc;
            if nr < 0 || nc < 0 || nr >= h as i32 || nc >= w as i32 { continue; }
            let nidx = (nc as usize) * h + (nr as usize);
            nbr_list[n_nbrs] = nidx as u32;
            n_nbrs += 1;
            if watershed_state { continue; }
            let l = labels[nidx];
            if l != 0 {
                if found_lab != 0 && l != found_lab {
                    watershed_state = true;
                } else {
                    found_lab = l;
                }
            }
        }

        if !watershed_state {
            labels[idx] = found_lab;
            for k in 0..n_nbrs {
                let nidx = nbr_list[k] as usize;
                if !seen[nidx] {
                    seen[nidx] = true;
                    let nv = img_cm[nidx];
                    let push_p = if nv > p { nv } else { p };
                    order_counter += 1;
                    heap.push(QueueItem {
                        priority: push_p,
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
}
