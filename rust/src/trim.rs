//! Trim watershed regions to `vol_thresh` fraction of peak volume.
//!
//! Port of pyDYNAM-O's `trim_region` / MATLAB `trimWshedRegions.m`:
//!   for each region:
//!     1. sort pixels by shifted-data value ascending
//!     2. find cutoff j where cumsum(vals)/total > (1-vol_thresh)
//!     3. keep pixels with value >= vals[j]
//!     4. fill holes (4-connectivity)
//!     5. pick largest connected component (by total shifted value)
//!     6. return trimmed labels (same shape as input)

use ndarray::{Array2, ArrayView2};
use std::collections::VecDeque;

/// Fill holes in a binary subimage, in place. A `false` pixel reachable
/// from the image edge through `false` pixels (4-connectivity) is
/// background. Any other `false` pixel is a hole and is set to `true`.
fn fill_holes_4conn(mask: &mut [bool], h: usize, w: usize) {
    // Mark every edge-connected `false` pixel as "outside" by breadth-first
    // search from the image edge. The remaining `false` pixels are holes.
    let mut outside = vec![false; h * w];
    let mut queue: VecDeque<(usize, usize)> = VecDeque::new();
    // Seed: all non-mask border pixels
    for x in 0..w {
        if !mask[x] {
            outside[x] = true;
            queue.push_back((0, x));
        }
        let bottom_idx = (h - 1) * w + x;
        if !mask[bottom_idx] {
            outside[bottom_idx] = true;
            queue.push_back((h - 1, x));
        }
    }
    for y in 0..h {
        let left_idx = y * w;
        let right_idx = y * w + (w - 1);
        if !mask[left_idx] {
            outside[left_idx] = true;
            queue.push_back((y, 0));
        }
        if !mask[right_idx] {
            outside[right_idx] = true;
            queue.push_back((y, w - 1));
        }
    }
    while let Some((y, x)) = queue.pop_front() {
        // wrapping_sub turns 0 - 1 into usize::MAX, which the bounds check
        // below rejects.
        let neighbors = [
            (y.wrapping_sub(1), x),
            (y + 1, x),
            (y, x.wrapping_sub(1)),
            (y, x + 1),
        ];
        for (ny, nx) in neighbors {
            if ny >= h || nx >= w {
                continue;
            }
            let nidx = ny * w + nx;
            if !mask[nidx] && !outside[nidx] {
                outside[nidx] = true;
                queue.push_back((ny, nx));
            }
        }
    }
    // Interior `false` pixels are holes. Fill them.
    for i in 0..(h * w) {
        if !mask[i] && !outside[i] {
            mask[i] = true;
        }
    }
}

/// 8-connected component labeling. Returns the component number of every
/// pixel (0 = not in the mask, components numbered from 1) and the number
/// of components.
fn connected_components_8(mask: &[bool], h: usize, w: usize) -> (Vec<u32>, usize) {
    let mut labels = vec![0u32; h * w];
    let mut next_label: u32 = 1;
    let mut q: VecDeque<(usize, usize)> = VecDeque::new();
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            if !mask[idx] || labels[idx] != 0 {
                continue;
            }
            let my_label = next_label;
            next_label += 1;
            labels[idx] = my_label;
            q.push_back((y, x));
            while let Some((cy, cx)) = q.pop_front() {
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        if dx == 0 && dy == 0 {
                            continue;
                        }
                        let ny = cy as i32 + dy;
                        let nx = cx as i32 + dx;
                        if ny < 0 || nx < 0 || ny >= h as i32 || nx >= w as i32 {
                            continue;
                        }
                        let ni = (ny as usize) * w + (nx as usize);
                        if mask[ni] && labels[ni] == 0 {
                            labels[ni] = my_label;
                            q.push_back((ny as usize, nx as usize));
                        }
                    }
                }
            }
        }
    }
    (labels, (next_label - 1) as usize)
}

/// Trim every region of `labels` to `vol_thresh` of its volume and return
/// the trimmed label image (same shape, 0 = background).
///
/// `data` is the image the labels were drawn on. Volume is measured above
/// `shift_val`: each pixel counts `max(data - shift_val, 0)`. For each
/// region the lowest pixels are removed up to `1 - vol_thresh` of the
/// total volume, holes in what is left are filled, and the connected
/// piece with the most volume is kept.
///
/// A region of fewer than 2 pixels is dropped. A region with constant
/// values, or with no volume above `shift_val`, is kept whole.
pub fn trim_all_regions(
    labels: ArrayView2<i32>,
    data: ArrayView2<f64>,
    vol_thresh: f64,
    shift_val: f64,
) -> Result<Array2<i32>, String> {
    let h = labels.nrows();
    let w = labels.ncols();
    let labels_slice = labels.as_slice().ok_or("labels not contiguous")?;
    let data_slice = data.as_slice().ok_or("data not contiguous")?;

    // Shifted data: max(data - shift_val, 0)
    let shifted: Vec<f64> = data_slice
        .iter()
        .map(|&v| (v - shift_val).max(0.0))
        .collect();

    // Group pixels by label in one linear pass.
    let mut max_label = 0i32;
    for &v in labels_slice {
        if v > max_label {
            max_label = v;
        }
    }
    if max_label <= 0 {
        return Ok(Array2::zeros((h, w)));
    }

    // Count per label first to preallocate.
    let mut counts = vec![0u32; (max_label as usize) + 1];
    for &v in labels_slice {
        if v > 0 {
            counts[v as usize] += 1;
        }
    }
    let mut by_label: Vec<Vec<u32>> = counts
        .iter()
        .map(|&c| Vec::with_capacity(c as usize))
        .collect();
    for (idx, &v) in labels_slice.iter().enumerate() {
        if v > 0 {
            by_label[v as usize].push(idx as u32);
        }
    }

    let mut out = Array2::<i32>::zeros((h, w));
    let out_slice = out.as_slice_mut().unwrap();

    for label in 1..=max_label {
        let pixels = &by_label[label as usize];
        if pixels.len() < 2 {
            continue;
        }

        // Collect values, compute min/max
        let mut vals: Vec<f64> = pixels.iter().map(|&p| shifted[p as usize]).collect();
        let vmin = vals.iter().cloned().fold(f64::INFINITY, f64::min);
        let vmax = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        if vmin == vmax {
            // Constant region. Keep as-is
            for &p in pixels {
                out_slice[p as usize] = label;
            }
            continue;
        }

        // Sort values ascending (matching scipy/numpy sort semantics)
        vals.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let total: f64 = vals.iter().sum();
        if total <= 0.0 {
            for &p in pixels {
                out_slice[p as usize] = label;
            }
            continue;
        }

        // Find cutoff: first j where cumsum[j] > (1 - vol_thresh) * total
        let target = (1.0 - vol_thresh) * total;
        let mut cumulative_volume = 0.0;
        let mut cutoff_idx = vals.len();
        for (j, v) in vals.iter().enumerate() {
            cumulative_volume += v;
            if cumulative_volume > target {
                cutoff_idx = j;
                break;
            }
        }
        if cutoff_idx >= vals.len() {
            for &p in pixels {
                out_slice[p as usize] = label;
            }
            continue;
        }
        let level = vals[cutoff_idx];

        // Bounding box of this region's pixels
        let mut rmin = h;
        let mut rmax = 0usize;
        let mut cmin = w;
        let mut cmax = 0usize;
        for &p in pixels {
            let r = (p as usize) / w;
            let c = (p as usize) % w;
            if r < rmin {
                rmin = r;
            }
            if r > rmax {
                rmax = r;
            }
            if c < cmin {
                cmin = c;
            }
            if c > cmax {
                cmax = c;
            }
        }
        // Pad by 1 for fill
        let rmin_p = rmin.saturating_sub(1);
        let rmax_p = (rmax + 1).min(h - 1);
        let cmin_p = cmin.saturating_sub(1);
        let cmax_p = (cmax + 1).min(w - 1);
        let sub_h = rmax_p - rmin_p + 1;
        let sub_w = cmax_p - cmin_p + 1;
        let sub_n = sub_h * sub_w;

        // Build sub_mask: True where label matches AND shift >= level
        let mut sub_mask = vec![false; sub_n];
        for sub_row in 0..sub_h {
            for sub_col in 0..sub_w {
                let r = rmin_p + sub_row;
                let c = cmin_p + sub_col;
                let full_idx = r * w + c;
                if labels_slice[full_idx] == label && shifted[full_idx] >= level {
                    sub_mask[sub_row * sub_w + sub_col] = true;
                }
            }
        }

        // Fill holes (4-connectivity)
        fill_holes_4conn(&mut sub_mask, sub_h, sub_w);

        // 8-CC + pick largest by volume
        let (component_labels, n_components) = connected_components_8(&sub_mask, sub_h, sub_w);
        if n_components == 0 {
            continue;
        }
        let mut component_volumes = vec![0.0f64; n_components + 1];
        for (i, &component) in component_labels.iter().enumerate() {
            if component == 0 {
                continue;
            }
            let sub_row = i / sub_w;
            let sub_col = i % sub_w;
            let full_idx = (rmin_p + sub_row) * w + (cmin_p + sub_col);
            component_volumes[component as usize] += shifted[full_idx];
        }
        let mut best_component = 1usize;
        // Volumes are never negative, so starting at -1 lets the first
        // component win even at volume 0.
        let mut best_volume = -1.0;
        for k in 1..=n_components {
            if component_volumes[k] > best_volume {
                best_volume = component_volumes[k];
                best_component = k;
            }
        }

        // Paint the kept component into the output.
        let mut n_written = 0u32;
        for sub_row in 0..sub_h {
            for sub_col in 0..sub_w {
                let sub_idx = sub_row * sub_w + sub_col;
                if component_labels[sub_idx] as usize == best_component {
                    let full_idx = (rmin_p + sub_row) * w + (cmin_p + sub_col);
                    out_slice[full_idx] = label;
                    n_written += 1;
                }
            }
        }
        if std::env::var("HISTPEAKS_DEBUG").is_ok() {
            let n_ge = sub_mask.iter().filter(|&&b| b).count();
            eprintln!(
                "[trim] lbl={} npix={} total={:.2} target={:.2} cutoff_idx={} level={:.4} n_ge_level(filled)={} n_largest_cc={}",
                label, pixels.len(), total, target, cutoff_idx, level, n_ge, n_written
            );
        }
    }

    Ok(out)
}
