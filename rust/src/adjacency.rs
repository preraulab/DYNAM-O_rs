//! Region adjacency graph from a watershed label image.
//!
//! For each region id (label > 0):
//!   - `interior`  = flat indices (row-major) of pixels IN the region
//!   - `border`    = flat indices of every pixel 8-adjacent to the region
//!                   that is not in it (watershed-0-line pixels and pixels
//!                   of other regions)
//!   - `all_px`    = interior ∪ border
//!   - `neighbors` = set of region ids reachable via 2-pixel dilation
//!                   (crossing the watershed line to adjacent regions)
//!
//! Matches pyDYNAM-O's layout: borders are stored as SORTED Vec<u32> so
//! intersection is linear-time two-pointer. All internal indices are u32
//! because segment sizes fit (≤ 600×300 ≈ 180k pixels).

use ndarray::ArrayView2;
use std::collections::BTreeSet;

/// One watershed region and its pixel lists (flat row-major indices, each
/// list sorted ascending).
#[derive(Debug, Default, Clone)]
pub struct Region {
    /// The region's watershed label (> 0).
    pub label: i32,
    /// Pixels carrying this label.
    pub interior: Vec<u32>,
    /// Pixels 8-adjacent to the region that are not in it. After merges,
    /// the symmetric difference of the merged regions' borders.
    pub border: Vec<u32>,
    /// MATLAB `Ldata2graph`'s `rgn{}` = `interior ∪ border`, carried
    /// through merges by *union* (`mergeRegions`: `regions{a}=unique([…])`).
    /// Unlike `border` (kept as a symmetric difference for edge weights),
    /// `all_px` accumulates every pixel, including the now-interior
    /// 0-lines between absorbed sub-regions, so the cut / trim / stats
    /// see a solid region, exactly as `extracthistpeaks.m` does.
    pub all_px: Vec<u32>,
    /// Labels of the regions within 2 pixels (Chebyshev distance) of this
    /// one.
    pub neighbors: BTreeSet<i32>,
}

/// Region adjacency graph of a label image.
#[derive(Debug, Default)]
pub struct Graph {
    /// One slot per region, in ascending label order. A slot becomes `None`
    /// when its region is taken out (merged away).
    pub regions: Vec<Option<Region>>,
    /// Label → slot index into `regions`. Filled by `from_labels`. A label
    /// is removed when its region is merged away.
    pub label_to_slot: std::collections::HashMap<i32, usize>,
    /// Image rows.
    pub height: usize,
    /// Image columns.
    pub width: usize,
}

impl Graph {
    /// Build the graph from a watershed label image (0 = watershed line,
    /// > 0 = region). The image must be C-contiguous.
    pub fn from_labels(labels: ArrayView2<i64>) -> Self {
        let h = labels.nrows();
        let w = labels.ncols();
        assert!(labels.is_standard_layout(), "labels must be C-contiguous");
        let raw = labels
            .as_slice()
            .expect("labels must be contiguous");

        // Largest label, to size the per-label count table.
        let mut max_label: i32 = 0;
        for &raw_value in raw.iter() {
            let label_value = raw_value as i32;
            if label_value > max_label {
                max_label = label_value;
            }
        }
        // Pixels per label, to size each region's interior list up front.
        let mut counts: Vec<u32> = vec![0; (max_label as usize) + 1];
        for &v in raw.iter() {
            if v > 0 {
                counts[v as usize] += 1;
            }
        }
        let n_regions_nonzero = counts.iter().skip(1).filter(|&&c| c > 0).count();

        let mut label_to_slot = std::collections::HashMap::with_capacity(n_regions_nonzero);
        let mut regions: Vec<Option<Region>> = Vec::with_capacity(n_regions_nonzero);
        for label in 1..=max_label {
            let pixel_count = counts[label as usize];
            if pixel_count > 0 {
                let slot = regions.len();
                regions.push(Some(Region {
                    label,
                    interior: Vec::with_capacity(pixel_count as usize),
                    border: Vec::new(),
                    all_px: Vec::new(),
                    neighbors: BTreeSet::new(),
                }));
                label_to_slot.insert(label, slot);
            }
        }

        // Second pass: fill interiors in row-major order (so the Vecs are sorted).
        for (idx, &v) in raw.iter().enumerate() {
            if v > 0 {
                let slot = label_to_slot[&(v as i32)];
                regions[slot].as_mut().unwrap().interior.push(idx as u32);
            }
        }

        let mut graph = Graph {
            regions,
            label_to_slot,
            height: h,
            width: w,
        };
        graph.fill_borders_and_neighbors(raw, h, w);
        graph
    }

    /// Per region (mirrors MATLAB `Ldata2graph.m`):
    /// - `border` = every pixel 8-adjacent to the region whose label differs
    ///   from the region's own label. I.e. the `conv2(sub_img,[1 1 1;1 -8
    ///   1;1 1 1])>0` rim, which is *all* non-self pixels around the region
    ///   (watershed-0-line pixels AND pixels of other regions). MATLAB then
    ///   redefines `rgn{} = interior ∪ border`.
    /// - `neighbors` = every other region label appearing within a 2-pixel
    ///   (Chebyshev) dilation of the region. MATLAB's 5×5 `H_dilate`,
    ///   excluding 0 and the region itself.
    fn fill_borders_and_neighbors(&mut self, raw: &[i64], h: usize, w: usize) {
        for y in 0..h {
            for x in 0..w {
                let v = raw[y * w + x] as i32;
                if v <= 0 {
                    continue;
                }
                let slot = match self.label_to_slot.get(&v).copied() {
                    Some(s) => s,
                    None => continue,
                };
                // 8-neighbor rim → border pixels of region v (any label ≠ v).
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        if dx == 0 && dy == 0 {
                            continue;
                        }
                        let ny = y as i32 + dy;
                        let nx = x as i32 + dx;
                        if ny < 0 || nx < 0 || ny >= h as i32 || nx >= w as i32 {
                            continue;
                        }
                        let nidx = (ny as usize) * w + (nx as usize);
                        if raw[nidx] as i32 != v {
                            self.regions[slot].as_mut().unwrap().border.push(nidx as u32);
                        }
                    }
                }
                // 5×5 (2-pixel Chebyshev) dilation → neighbor region labels.
                for dy in -2i32..=2 {
                    for dx in -2i32..=2 {
                        if dx == 0 && dy == 0 {
                            continue;
                        }
                        let ny = y as i32 + dy;
                        let nx = x as i32 + dx;
                        if ny < 0 || nx < 0 || ny >= h as i32 || nx >= w as i32 {
                            continue;
                        }
                        let nv = raw[(ny as usize) * w + (nx as usize)] as i32;
                        if nv > 0 && nv != v {
                            self.regions[slot].as_mut().unwrap().neighbors.insert(nv);
                        }
                    }
                }
            }
        }

        // Dedupe / sort borders (a rim pixel is reached from several region
        // pixels). They're pushed in row-major order so largely sorted.
        // Then build `all_px = interior ∪ border` (MATLAB `rgn{}`).
        for slot in 0..self.regions.len() {
            if let Some(region) = self.regions[slot].as_mut() {
                region.border.sort_unstable();
                region.border.dedup();
                // interior is already sorted (row-major fill). Merge with border.
                let mut merged = Vec::with_capacity(region.interior.len() + region.border.len());
                let mut interior_pos = 0usize;
                let mut border_pos = 0usize;
                while interior_pos < region.interior.len() && border_pos < region.border.len() {
                    match region.interior[interior_pos].cmp(&region.border[border_pos]) {
                        std::cmp::Ordering::Less => {
                            merged.push(region.interior[interior_pos]);
                            interior_pos += 1;
                        }
                        std::cmp::Ordering::Greater => {
                            merged.push(region.border[border_pos]);
                            border_pos += 1;
                        }
                        // A pixel in both lists is kept once.
                        std::cmp::Ordering::Equal => {
                            merged.push(region.interior[interior_pos]);
                            interior_pos += 1;
                            border_pos += 1;
                        }
                    }
                }
                merged.extend_from_slice(&region.interior[interior_pos..]);
                merged.extend_from_slice(&region.border[border_pos..]);
                region.all_px = merged;
            }
        }
    }

    /// Slot of the region with this label, or `None` if no such region is
    /// live.
    pub fn slot_of(&self, label: i32) -> Option<usize> {
        self.label_to_slot.get(&label).copied()
    }

    /// Remove and return the region in `slot`, leaving the slot empty.
    pub fn take(&mut self, slot: usize) -> Option<Region> {
        self.regions[slot].take()
    }
}
