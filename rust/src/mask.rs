//! Pass-2 spectrogram masking.
//!
//! Port of pydynamo `tfpeaks.mask.mask_spectrogram` which itself matches
//! MATLAB `computeTFPeaks.m:maskSpectrogram`:
//!
//! 1. For each pass-2 time column, find the nearest pass-1 column and
//!    look up its label vector.
//! 2. Paint spect2[f,t] wherever labels>0; zero elsewhere.
//! 3. Zero the 1-pixel inner perimeter of each labeled region (8-conn),
//!    matching MATLAB's `spect_masked(border_inds) = 0` where
//!    border_inds = trim_borders from bwboundaries.

use ndarray::{Array2, ArrayView1, ArrayView2};

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

/// True if pixel (r, c) is an "inner boundary" pixel of its label under
/// 8-connectivity — i.e. at least one of its 8 neighbors has a different
/// label (including 0 / out-of-bounds). Matches
/// `skimage.segmentation.find_boundaries(mode="inner", connectivity=2)`.
#[inline]
fn is_inner_boundary(labels: &Array2<i64>, r: usize, c: usize) -> bool {
    let h = labels.nrows();
    let w = labels.ncols();
    let my_label = labels[[r, c]];
    if my_label == 0 {
        return false;
    }
    for dr in -1i32..=1 {
        for dc in -1i32..=1 {
            if dr == 0 && dc == 0 {
                continue;
            }
            let nr = r as i32 + dr;
            let nc = c as i32 + dc;
            if nr < 0 || nc < 0 || nr >= h as i32 || nc >= w as i32 {
                // OOB counts as different label → inner-boundary
                return true;
            }
            if labels[[nr as usize, nc as usize]] != my_label {
                return true;
            }
        }
    }
    false
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

    // Construct labels_on_2s and apply mask in one pass.
    // Easier to materialize labels_on_2s so we can do the boundary pass
    // on it directly.
    let mut labels_on_2s = Array2::<i64>::zeros((f_count, t_count_2));
    for t in 0..t_count_2 {
        let src = nearest[t];
        for f in 0..f_count {
            labels_on_2s[[f, t]] = labels_1s[[f, src]];
        }
    }

    // Initial mask: paint spect wherever labels > 0, else 0.
    let mut masked = Array2::<f64>::zeros((f_count, t_count_2));
    for t in 0..t_count_2 {
        for f in 0..f_count {
            if labels_on_2s[[f, t]] > 0 {
                masked[[f, t]] = spect_2s[[f, t]];
            }
        }
    }

    // Zero 1-pixel inner perimeter of each labeled region.
    // (skimage find_boundaries mode='inner', connectivity=2)
    for t in 0..t_count_2 {
        for f in 0..f_count {
            if labels_on_2s[[f, t]] > 0 && is_inner_boundary(&labels_on_2s, f, t) {
                masked[[f, t]] = 0.0;
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
