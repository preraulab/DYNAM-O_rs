//! Region adjacency graph from a watershed label image.
//!
//! For each region id (label > 0):
//!   - `interior`  = flat indices (row-major) of pixels IN the region
//!   - `border`    = flat indices of watershed-0-line pixels adjacent to
//!                   the region (8-connectivity, 1-pixel dilation ring)
//!   - `neighbors` = set of region ids reachable via 2-pixel dilation
//!                   (crossing the watershed line to adjacent regions)
//!
//! Matches pyDYNAM-O's layout: borders are stored as SORTED Vec<u32> so
//! intersection is linear-time two-pointer. All internal indices are u32
//! because segment sizes fit (≤ 600×300 ≈ 180k pixels).

use ndarray::ArrayView2;
use std::collections::BTreeSet;

#[derive(Debug, Default, Clone)]
pub struct Region {
    pub label: i32,
    pub interior: Vec<u32>,
    pub border: Vec<u32>,
    /// MATLAB `Ldata2graph`'s `rgn{}` = `interior ∪ border`, carried
    /// through merges by *union* (`mergeRegions`: `regions{a}=unique([…])`).
    /// Unlike `border` (kept as a symmetric difference for edge weights),
    /// `all_px` accumulates every pixel — including the now-interior
    /// 0-lines between absorbed sub-regions — so the cut / trim / stats
    /// see a solid region, exactly as `extracthistpeaks.m` does.
    pub all_px: Vec<u32>,
    pub neighbors: BTreeSet<i32>,
}

#[derive(Debug, Default)]
pub struct Graph {
    /// `regions[i]` — the i-th region (1:1 with dense slot indices).
    pub regions: Vec<Option<Region>>,
    /// label → slot index. Filled lazily; unused labels have sentinel -1.
    pub label_to_slot: std::collections::HashMap<i32, usize>,
    pub height: usize,
    pub width: usize,
}

impl Graph {
    pub fn from_labels(labels: ArrayView2<i64>) -> Self {
        let (h, w) = (labels.nrows(), labels.ncols());
        assert!(labels.is_standard_layout(), "labels must be C-contiguous");
        let raw = labels
            .as_slice()
            .expect("labels must be contiguous");

        // First pass: count pixels per label → pre-allocate interior Vec sizes.
        let mut max_label: i32 = 0;
        for &v in raw.iter() {
            let v = v as i32;
            if v > max_label {
                max_label = v;
            }
        }
        // Grouped counts and region list. We iterate linearly once, placing
        // interior pixel indices into each region's Vec.
        let mut counts: Vec<u32> = vec![0; (max_label as usize) + 1];
        for &v in raw.iter() {
            if v > 0 {
                counts[v as usize] += 1;
            }
        }
        let n_regions_nonzero = counts.iter().skip(1).filter(|&&c| c > 0).count();

        let mut label_to_slot = std::collections::HashMap::with_capacity(n_regions_nonzero);
        let mut regions: Vec<Option<Region>> = Vec::with_capacity(n_regions_nonzero);
        for lab in 1..=max_label {
            let c = counts[lab as usize];
            if c > 0 {
                let slot = regions.len();
                regions.push(Some(Region {
                    label: lab,
                    interior: Vec::with_capacity(c as usize),
                    border: Vec::new(),
                    all_px: Vec::new(),
                    neighbors: BTreeSet::new(),
                }));
                label_to_slot.insert(lab, slot);
            }
        }

        // Second pass: fill interiors in row-major order (so the Vecs are sorted).
        for (idx, &v) in raw.iter().enumerate() {
            if v > 0 {
                let slot = label_to_slot[&(v as i32)];
                regions[slot].as_mut().unwrap().interior.push(idx as u32);
            }
        }

        let mut g = Graph {
            regions,
            label_to_slot,
            height: h,
            width: w,
        };
        g.fill_borders_and_neighbors(raw, h, w);
        g
    }

    /// Per region (mirrors MATLAB `Ldata2graph.m`):
    /// - `border` = every pixel 8-adjacent to the region whose label differs
    ///   from the region's own label — i.e. the `conv2(sub_img,[1 1 1;1 -8
    ///   1;1 1 1])>0` rim, which is *all* non-self pixels around the region
    ///   (watershed-0-line pixels AND pixels of other regions). MATLAB then
    ///   redefines `rgn{} = interior ∪ border`.
    /// - `neighbors` = every other region label appearing within a 2-pixel
    ///   (Chebyshev) dilation of the region — MATLAB's 5×5 `H_dilate`,
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
                // 8-neighbour rim → border pixels of region v (any label ≠ v).
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
                // 5×5 (2-pixel Chebyshev) dilation → neighbour region labels.
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
            if let Some(r) = self.regions[slot].as_mut() {
                r.border.sort_unstable();
                r.border.dedup();
                // interior is already sorted (row-major fill); merge with border.
                let mut all = Vec::with_capacity(r.interior.len() + r.border.len());
                let (mut i, mut j) = (0usize, 0usize);
                while i < r.interior.len() && j < r.border.len() {
                    match r.interior[i].cmp(&r.border[j]) {
                        std::cmp::Ordering::Less => { all.push(r.interior[i]); i += 1; }
                        std::cmp::Ordering::Greater => { all.push(r.border[j]); j += 1; }
                        std::cmp::Ordering::Equal => { all.push(r.interior[i]); i += 1; j += 1; }
                    }
                }
                all.extend_from_slice(&r.interior[i..]);
                all.extend_from_slice(&r.border[j..]);
                r.all_px = all;
            }
        }
    }

    pub fn slot_of(&self, label: i32) -> Option<usize> {
        self.label_to_slot.get(&label).copied()
    }

    pub fn take(&mut self, slot: usize) -> Option<Region> {
        self.regions[slot].take()
    }
}
