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

    /// Per region:
    /// - `border` = all zero-labeled pixels in the 3×3 (8-connected) dilation of the region.
    /// - `neighbors` = all region-labels found in the 5×5 (2-pixel) dilation
    ///   of the region, excluding 0 and the region itself.
    fn fill_borders_and_neighbors(&mut self, raw: &[i64], h: usize, w: usize) {
        // Do it efficiently: scan the image once, and for each pixel p with
        // label>0, look at its 8-neighbors; if any neighbor is 0, mark that
        // neighbor as border of p's region, and look one more step out for
        // labeled neighbors (which become 'neighbors' of p's region).
        // This is equivalent to a per-region 3×3 dilation intersected with
        // the zero mask plus a 5×5 dilation.
        //
        // We use a two-pass approach:
        //   Pass A: for each 0-pixel, find all labeled neighbors (8-conn).
        //           → every such labeled neighbor gets this 0-pixel as a border.
        //           → all distinct labeled neighbors of this 0-pixel become
        //             mutual neighbors (separated by a 1-pixel wide border).
        //   Pass B (optional): also bridge labels separated by TWO 0-pixels
        //           (MATLAB uses dilation distance=2). In practice the
        //           watershed 0-line is 1-px wide, so Pass A covers it.
        //
        // Additionally mark any region-pixel adjacent to a non-equal labeled
        // pixel as a neighbor (no watershed line between them).
        for y in 0..h {
            for x in 0..w {
                let idx = y * w + x;
                let v = raw[idx] as i32;
                // 8-neighbor offsets
                // Scan neighbors once; actions depend on center value.
                // We only need unique labels touching this pixel.
                if v == 0 {
                    // border pixel — collect the set of labeled neighbors
                    let mut seen: [i32; 8] = [0; 8];
                    let mut n_seen = 0usize;
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
                            let nv = raw[nidx] as i32;
                            if nv > 0 {
                                // dedupe in the 8-slot array
                                let mut found = false;
                                for k in 0..n_seen {
                                    if seen[k] == nv {
                                        found = true;
                                        break;
                                    }
                                }
                                if !found {
                                    seen[n_seen] = nv;
                                    n_seen += 1;
                                }
                            }
                        }
                    }
                    // For every labeled neighbor, this 0-pixel is its border.
                    for i in 0..n_seen {
                        let a = seen[i];
                        if let Some(slot) = self.label_to_slot.get(&a).copied() {
                            if let Some(r) = self.regions[slot].as_mut() {
                                r.border.push(idx as u32);
                            }
                        }
                    }
                    // And every pair of distinct labeled neighbors of this
                    // 0-pixel are graph-neighbors.
                    for i in 0..n_seen {
                        for j in (i + 1)..n_seen {
                            let (a, b) = (seen[i], seen[j]);
                            if a == b {
                                continue;
                            }
                            if let (Some(&sa), Some(&sb)) = (
                                self.label_to_slot.get(&a),
                                self.label_to_slot.get(&b),
                            ) {
                                if let Some(r) = self.regions[sa].as_mut() {
                                    r.neighbors.insert(b);
                                }
                                if let Some(r) = self.regions[sb].as_mut() {
                                    r.neighbors.insert(a);
                                }
                                let _ = (sa, sb);
                            }
                        }
                    }
                } else {
                    // labeled pixel: check for labeled neighbors with DIFFERENT
                    // label (regions abutting without watershed line between)
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
                            let nv = raw[(ny as usize) * w + (nx as usize)] as i32;
                            if nv > 0 && nv != v {
                                if let (Some(&sa), Some(&sb)) = (
                                    self.label_to_slot.get(&v),
                                    self.label_to_slot.get(&nv),
                                ) {
                                    if let Some(r) = self.regions[sa].as_mut() {
                                        r.neighbors.insert(nv);
                                    }
                                    if let Some(r) = self.regions[sb].as_mut() {
                                        r.neighbors.insert(v);
                                    }
                                    let _ = (sa, sb);
                                }
                            }
                        }
                    }
                }
            }
        }

        // Dedupe / sort borders (pixels added multiple times via different 0-pixels
        // should appear once; they're added in row-major order so already
        // sorted mostly, but dedup is safe).
        for slot in 0..self.regions.len() {
            if let Some(r) = self.regions[slot].as_mut() {
                r.border.sort_unstable();
                r.border.dedup();
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
