//! Pass-2 spectrogram masking.
//!
//! Follows MATLAB `computeTFPeaks.m:maskSpectrogram`:
//!
//! 1. For each pass-2 time column, find the nearest pass-1 column and
//!    look up its label vector.
//! 2. Keep spect2[f,t] wherever the pass-1 label is > 0; zero elsewhere.
//! 3. Zero each region's contour, MATLAB's `spect_masked(border_inds) = 0`.
//!    `border_inds` are the `trimmed_borders` of `trimWshedRegions.m`:
//!    the 8-connected outer boundary that `bwtraceboundary` walks from the
//!    region's first pixel. A region pixel that touches the outside only
//!    at a corner is not on that walk and is kept.

use ndarray::{Array2, ArrayView1, ArrayView2};
use std::collections::HashMap;

use crate::extract_pipeline::moore_trace;

/// Nearest-neighbor mapping: for each stimes_2s value, the index into
/// stimes_1s of the closest entry. Ties broken toward the right (matches
/// pydynamo `use_left = abs(left) < abs(right)` strict-less).
fn nearest_indices(stimes_2s: ArrayView1<f64>, stimes_1s: ArrayView1<f64>) -> Vec<usize> {
    let n = stimes_1s.len();
    let mut out = Vec::with_capacity(stimes_2s.len());
    for &s in stimes_2s.iter() {
        let mut lo = 0usize;
        let mut hi = n;
        while lo < hi {
            let mid = (lo + hi) / 2;
            if stimes_1s[mid] < s {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        let right = lo.min(n - 1);
        let left = if lo == 0 { 0 } else { lo - 1 };
        let d_right = (stimes_1s[right] - s).abs();
        let d_left = (stimes_1s[left] - s).abs();
        out.push(if d_left < d_right { left } else { right });
    }
    out
}

/// Contour pixels of every labeled region, on the label image's own grid.
///
/// For each label the walk starts at the region's first pixel in
/// column-major order (top of its leftmost column), as `trimWshedRegions.m`
/// seeds `bwtraceboundary(cc_mask, seed, 'N', 8)`, and follows the
/// 8-connected outer boundary all the way around. Every pixel the walk
/// visits is marked.
fn region_contour_mask(labels: ArrayView2<i64>) -> Array2<bool> {
    let (n_rows, n_cols) = labels.dim();

    // First pixel of each label, scanning down each column, left to right.
    let mut first_pixel: HashMap<i64, (usize, usize)> = HashMap::new();
    for c in 0..n_cols {
        for r in 0..n_rows {
            let label = labels[[r, c]];
            if label > 0 && !first_pixel.contains_key(&label) {
                first_pixel.insert(label, (r, c));
            }
        }
    }

    let mut on_contour = Array2::<bool>::from_elem((n_rows, n_cols), false);
    for (&label, &(start_r, start_c)) in first_pixel.iter() {
        let walk = moore_trace(labels, label, start_r, start_c);
        for (r, c) in walk {
            on_contour[[r, c]] = true;
        }
    }
    on_contour
}

/// Build the pass-2 masked spectrogram.
pub fn mask_spectrogram(
    spect_2s: ArrayView2<f64>,
    stimes_2s: ArrayView1<f64>,
    labels_1s: ArrayView2<i64>,
    stimes_1s: ArrayView1<f64>,
) -> Result<Array2<f64>, String> {
    let (f_count, t_count_2) = spect_2s.dim();
    if labels_1s.nrows() != f_count {
        return Err(format!(
            "labels_1s.nrows {} != spect_2s.nrows {}",
            labels_1s.nrows(),
            f_count
        ));
    }
    if stimes_2s.len() != t_count_2 {
        return Err(format!(
            "stimes_2s.len {} != spect_2s.ncols {}",
            stimes_2s.len(),
            t_count_2
        ));
    }
    if stimes_1s.len() != labels_1s.ncols() {
        return Err(format!(
            "stimes_1s.len {} != labels_1s.ncols {}",
            stimes_1s.len(),
            labels_1s.ncols()
        ));
    }

    // Map each pass-2 column to the nearest pass-1 column.
    let nearest = nearest_indices(stimes_2s, stimes_1s);

    // Region contours are traced on the pass-1 grid, where the regions
    // were found, and carried to the pass-2 grid with the labels. MATLAB
    // does the same: borders are pass-1 pixel indices shifted in time.
    let on_contour_1s = region_contour_mask(labels_1s);

    // Keep a pass-2 pixel when its pass-1 pixel is inside a region and
    // not on that region's contour.
    let mut masked = Array2::<f64>::zeros((f_count, t_count_2));
    for t in 0..t_count_2 {
        let src = nearest[t];
        for f in 0..f_count {
            let in_region = labels_1s[[f, src]] > 0;
            if in_region && !on_contour_1s[[f, src]] {
                masked[[f, t]] = spect_2s[[f, t]];
            }
        }
    }

    Ok(masked)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn mask_interior_keeps_strict_interior() {
        // 3x3 region of label 1 surrounded by 0.
        // Inner perimeter = the 8 outer pixels of the 3x3; only center
        // survives.
        let labels = array![
            [0i64, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ];
        let spect = array![
            [1.0f64, 2.0, 3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0, 9.0, 10.0],
            [11.0, 12.0, 13.0, 14.0, 15.0],
            [16.0, 17.0, 18.0, 19.0, 20.0],
            [21.0, 22.0, 23.0, 24.0, 25.0],
        ];
        let stimes = array![0.0, 1.0, 2.0, 3.0, 4.0];
        let masked = mask_spectrogram(spect.view(), stimes.view(), labels.view(), stimes.view())
            .unwrap();
        // Only the center pixel (2,2) should be non-zero.
        let nz: Vec<(usize, usize)> = (0..masked.nrows())
            .flat_map(|r| (0..masked.ncols()).map(move |c| (r, c)))
            .filter(|&(r, c)| masked[[r, c]] != 0.0)
            .collect();
        assert_eq!(nz, vec![(2, 2)]);
        assert_eq!(masked[[2, 2]], 13.0);
    }

    #[test]
    fn a_pixel_touching_outside_only_at_a_corner_is_kept() {
        // The region's top-right corner is cut off, so pixel (2,3) has the
        // outside pixel (1,4) as a diagonal neighbor and region pixels on
        // all four sides. The contour walk goes (1,3) -> (2,4) and steps
        // past it, as MATLAB's bwtraceboundary does, so it is kept.
        let labels = array![
            [0i64, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0],
        ];
        let spect = Array2::<f64>::from_elem((6, 6), 7.0);
        let stimes = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let masked = mask_spectrogram(spect.view(), stimes.view(), labels.view(), stimes.view())
            .unwrap();
        let mut kept: Vec<(usize, usize)> = Vec::new();
        for r in 0..6 {
            for c in 0..6 {
                if masked[[r, c]] != 0.0 {
                    kept.push((r, c));
                }
            }
        }
        assert_eq!(kept, vec![(2, 2), (2, 3), (3, 2), (3, 3)]);
    }

    #[test]
    fn touching_regions_each_lose_their_own_contour() {
        // Two regions side by side with no gap. Each contour is traced on
        // its own region, so both facing columns are zeroed and only the
        // two interior pixels survive.
        let labels = array![
            [1i64, 1, 1, 2, 2, 2],
            [1, 1, 1, 2, 2, 2],
            [1, 1, 1, 2, 2, 2],
        ];
        let spect = Array2::<f64>::from_elem((3, 6), 5.0);
        let stimes = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let masked = mask_spectrogram(spect.view(), stimes.view(), labels.view(), stimes.view())
            .unwrap();
        let mut kept: Vec<(usize, usize)> = Vec::new();
        for r in 0..3 {
            for c in 0..6 {
                if masked[[r, c]] != 0.0 {
                    kept.push((r, c));
                }
            }
        }
        assert_eq!(kept, vec![(1, 1), (1, 4)]);
    }

    #[test]
    fn nearest_mapping_different_stimes() {
        // 2×2 label at time 1; pass-2 stimes land at 0.5 and 1.5 (tie → right)
        let labels = array![[1i64, 2], [1, 2]];
        let stimes_1s = array![1.0, 2.0];
        let spect_2s = array![[10.0f64, 20.0, 30.0], [40.0, 50.0, 60.0]];
        let stimes_2s = array![0.5, 1.5, 2.0];
        let masked = mask_spectrogram(
            spect_2s.view(),
            stimes_2s.view(),
            labels.view(),
            stimes_1s.view(),
        )
        .unwrap();
        assert_eq!(masked.dim(), (2, 3));
    }
}
