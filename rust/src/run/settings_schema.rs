//! Settings-file schema identity, drift reporting, and migrations.
//!
//! A settings JSON is the only record of how a run was configured, and it
//! outlives the code that wrote it. Three things can silently change what
//! it means, and they need three different mechanisms:
//!
//! 1. **A field is absent**, so it takes today's default. `#[serde(default)]`
//!    fills it with no complaint, and defaults do move (`sopower_range` went
//!    `[-5,20]` → `[-5,25]`, `freq_range` `[0,30]` → `[2,18]`).
//! 2. **A field was removed or renamed**, so the file asks for something the
//!    code no longer offers. serde drops unknown keys silently.
//! 3. **A field keeps its name, type and position but changes MEANING.**
//!
//! [`fingerprint`] catches 1 and 2 automatically, by hashing the *shape* of
//! `DynamoSettings::default()` — every key path and its type, values
//! ignored. It needs no maintenance, is immune to refactors and comment
//! edits, and because [`diff`] keeps the key sets rather than only the hash,
//! it can name exactly which settings appeared or vanished.
//!
//! It cannot catch 3. The σ reparameterization (2026-08-03) changed
//! `FreqStd`/`SOpowerStd` from `√2·σ` to `σ` without touching a single key
//! name or type, so the fingerprint before and after is byte-identical.
//! Nothing derived from the code's structure or its git history can see
//! that. Only a human can, which is what [`SEMANTICS_VERSION`] is for.
//!
//! The manual part is deliberately tiny: bump `SEMANTICS_VERSION` only when
//! the shape stays the same and the meaning changes. `schema_fingerprint_is_pinned`
//! fails whenever the shape moves, which is the moment to ask whether any
//! existing field also changed meaning — so the question gets *asked* rather
//! than *remembered*.

use serde_json::Value;

use crate::run::settings::DynamoSettings;

/// The computation-semantics version. Bumped when either:
///
/// 1. a settings field keeps its name, type and position but changes
///    what its value means (the original trigger — these entries carry
///    a real migration), or
/// 2. **any change alters the numbers the pipeline computes** from the
///    same inputs and settings — a numeric bugfix, a changed constant,
///    a reworked algorithm. These get a NO-OP migration (precedent:
///    v2→v3 is log-only) whose `summary` line IS the numeric-changelog
///    entry.
///
/// Because every artifact stamps this number (`# semantics:`,
/// dynamo-export `stamp.rs`), it is what lets `check-results` /
/// `validate` say whether an existing tree's numbers still match
/// current code: builds whose shas differ but whose semantics match are
/// numerically comparable (the commits between are UI or tooling).
/// **A numeric change that ships without a bump silently breaks that
/// verdict** — when in doubt, bump.
///
/// Shape changes are caught by [`fingerprint`] and alone must NOT bump
/// this.
///
/// | value | date       | what changed |
/// |-------|------------|--------------|
/// | 1     | (pre-2026-08-03) | original convention |
/// | 2     | 2026-08-03 | paramfit Gaussian widths became true standard deviations. `paramfit.{power,phase}.mode_bounds.fstd_*` and `paramfit.power.mode_bounds.pstd_*` are now σ, previously `√2·σ`. `paramfit.phase.mode_bounds.pstd_*` (recikappa) is unchanged. A v1 file's width bounds are `√2` too wide if read as v2. |
/// | 3     | 2026-08-03 | `constrain_freq_center` / `constrain_power_center` now constrain to the VALID (non-NaN) data inside the fit window, not to the window itself, matching MATLAB `param_basis_*.m`. The field names, types and values are unchanged, but the region they describe is narrower wherever `hist.min_time_in_bin` emptied SO-power bins, so the same v2 file yields different fits under v3. Also affects the derived (`null`) value of the new `mode_bounds.{fmean,pmean}_*` slots. |
/// | 4     | 2026-09-11 | `paramfit.peak_assign_prob` was absorbed into the flexible `paramfit.peak_assign`, whose default is argmax within a 1.5σ footprint (serialized `"argmax 1.5 sigma"`) — so an ABSENT key changed meaning (it used to mean a 0.95 probability contour). Migration pins a pre-v4 file to its old meaning: the stored `peak_assign_prob` (or 0.95 when absent) becomes `peak_assign: "<x> p"` and the retired key is removed. (The footprint default was fixed at 1.5σ on 2026-09-12, the day after v4 landed, before any tree outside local test data ran under bare argmax.) |
/// | 5     | 2026-10-13 | The per-mode `Pk*` peak-property columns aggregate by MEDIAN instead of mean (`PkSOphase` stays a circular mean, `PkCount` a count): the per-peak size properties are heavy-tailed and a mean is dominated by a few giant tail events — on the NSS two-night cohort the mean costs all night-to-night subject reliability (PkVolume r ≈ 0.0 mean vs ≈ 0.8 median). Also adds `PkPower` (median per-peak `pow2db(Height × baseline(PeakFrequency))`, absolute power re-anchored to the extraction baseline, which phase-1 now persists in aux h5). No settings value changes shape or meaning; the migration is a changelog no-op. |
/// | 6     | 2026-11-16 | Per-peak `SOphase` is interpolated from the UNWRAPPED SO-phase series and wrapped afterwards (MATLAB `computePeakSOphase.m`). It was interpolated from the wrapped series, so a peak whose two neighboring EEG samples straddle ±π received an arbitrary phase in (−π, π) instead of one near ±π: ~0.7% of peaks on the Twin test cohort, which carved a ~3% trough into every SO-phase histogram bin covering ±π. Changes the stats-table `SOphase` column for those peaks, the SO-phase SOPH, and everything fit to it. No settings value changes shape or meaning; the migration is a changelog no-op. |
/// | 7     | 2026-11-17 | Per-peak `PeakStage`, `SOpower`, and `SOphase` come from the kernel's `assign_peak_*`, which follow MATLAB `computePeakStage.m` / `computePeakSOpower.m` / `computePeakSOphase.m` exactly. The artifact override (stage 6) now tests the EEG sample NEAREST the peak time, as MATLAB's `interp1(..., 'nearest')` does. It tested the sample at or before the peak time, which shifted every artifact span half a sample late, so a peak in the half-sample before an artifact onset was not labeled 6 and a peak in the half-sample after an artifact end was. The SO-power and SO-phase lookups gain MATLAB's one-grid-step endpoint pad (a peak up to one step outside the series takes the end value instead of NaN), which no peak reaches while `sopower.retain_fs` is true. Changes `PeakStage` for peaks within half a sample of an artifact-span edge, and with it the SOPH membership of those peaks. No settings value changes shape or meaning; the migration is a changelog no-op. |
/// | 8     | 2026-11-17 | Artifact detection matches MATLAB `detect_artifacts.m` in two places where it did not. (1) The slope-test mask is folded into the bad-sample set BEFORE the high-frequency and broadband tests, so slope-flagged samples are interpolated out of the filtered signal and left out of the z-score statistics. It was OR-ed onto the finished band mask, which computed the band thresholds from data MATLAB excludes. (2) The 2 s envelope smoothing and 300 s detrend use MATLAB's alignment for even-length `movmean` / `movmedian` windows (samples `i - k/2` to `i + k/2 - 1`). They sat one sample later, which shifted every band-detected artifact span by one sample. On the Twin test cohort the artifact mask differed from MATLAB's at 0.07% to 1.6% of each recording and now differs at 0 or 1 sample. The mask feeds the baseline, SO-power, SO-phase, `PeakStage`, and the surviving peak set, so every downstream number can change. No settings value changes shape or meaning; the migration is a changelog no-op. |
/// | 9     | 2026-11-17 | TF-peak extraction follows the pure-MATLAB code (`runWatershed.m`, `mergeWshedSegment.m`, `mergeRegions.m`) in four places where it did not. (1) After a merge, a region whose only remaining neighbor is the merged region is absorbed into it. (2) Merging continues only while the heaviest edge is strictly above `detection.merge_thresh`. An edge exactly at the threshold used to merge. (3) Watershed regions are renumbered by `bwlabel` order (first pixel going down the columns) before the merge, and each merge keeps the label MATLAB keeps, which decides the order equal-weight edges merge in and which region owns a shared border pixel. (4) Equal-weight edges merge in MATLAB's edge-table order. On the Twin test cohort the per-segment extraction went from 27% of MATLAB's peaks reproduced exactly to all of them. A recording keeps about 93% of its peaks within 0.1 s and 0.5 Hz, loses 1.5% to 1.9% of its peak count, and most surviving peaks move slightly in every property. The SOPH mode-peak finder that seeds the parametric fit keeps its own merge procedure and is unchanged. No settings value changes shape or meaning; the migration is a changelog no-op. |
/// | 10    | 2026-11-17 | Peak-frequency refinement interpolates the event spectrum with a not-a-knot cubic spline, as MATLAB `interp1(..., 'spline')` in `refinePeakFrequency.m` does. It used a natural spline (second derivative forced to zero at both ends of the frequency axis). The two agree away from the ends, so only peaks whose bounding box reaches the bottom or top of the analyzed band move: about 0.1% of peaks on the Twin test cohort, by up to 0.23 Hz, and a handful of peaks per recording change sides of the edge-peak removal. Given the same samples, event times, and bounding boxes, the refined frequency now equals MATLAB's for every event. No settings value changes shape or meaning; the migration is a changelog no-op. |
/// | 11    | 2026-11-17 | The peak BoundingBox is the exact extent of the region's pixels: its corner sits half a bin before the center of the first pixel on each axis (it sat on that center), so the box ends half a bin after the center of the last pixel. `bbox_tl_s` moves half a time bin earlier and `bbox_tl_Hz` half a frequency bin lower for every peak, and width and height are unchanged. The box is the frequency window the refinement searches, so the window moves down by half a bin (0.049 Hz at the defaults): about 1% more peaks survive the edge-peak removal on the Twin test cohort, and a few change refined frequency. MATLAB's corner was a full bin high and is fixed to the same extent in the MATLAB toolbox. No settings value changes shape or meaning; the migration is a changelog no-op. |
/// | 12    | 2026-11-17 | The pass-2 mask zeros each pass-1 region's traced contour, as MATLAB `maskSpectrogram` does with the `bwtraceboundary` borders from `trimWshedRegions.m`. It zeroed every region pixel with a differently labeled neighbor in any of the 8 directions, which also removed pixels touching the outside only at a corner: about 5% of all region pixels on the Twin test cohort. With those pixels kept, pass 2 finds more and slightly larger peaks: the final peak count rises 2.4% to 3.0%, about 92% to 93% of peaks keep a partner within 0.1 s and 0.5 Hz, and the SOPHs correlate with their previous versions at 0.997 (SO-power) and 0.962 to 0.976 (SO-phase). The masked spectrogram now equals MATLAB's pixel for pixel. No settings value changes shape or meaning; the migration is a changelog no-op. |
/// | 13    | 2026-11-17 | Zero-phase filtering follows MATLAB `filtfilt`: each second-order section is run forward and backward on its own odd-reflected extension of `3 x filter order` samples, before the next section starts. It ran all sections forward and then all backward on one extension of `6 x sections + 3` samples (the scipy `sosfiltfilt` procedure). The two agree away from the ends of a recording and differ inside the filter's start-up transient, which for the narrow slow-oscillation band lasts minutes: the filtered slow-oscillation signal differed from MATLAB's by several times its RMS in the first and last 10 s, and SO-phase differed by more than 0.01 rad at 1% to 5% of samples. It now equals MATLAB's to 1e-10 of RMS, and SO-phase to 1e-7 rad. Effect on outputs is small because the ends of a night are mostly wake: SO-phase SOPHs correlate with their previous versions at 0.999 or better, and peaks, artifact masks, and SO-power are unchanged or change by a handful of samples. No settings value changes shape or meaning; the migration is a changelog no-op. |
/// | 14    | 2026-11-17 | The minimum TF-peak height is computed from the chi² quantile for any number of tapers, as MATLAB `computeTFPeaks.m` does: `-2 x pow2db(2K / chi2inv(0.975, 2K))`. It was a stored constant for 3 tapers and no floor at all for any other count, so a run with `mtm.n_tapers` other than 3 kept every low peak that MATLAB removes. The stored 3-taper quantile was also off in the 8th digit (14.449374448 against MATLAB's 14.449375335), which moves the default floor by 5e-7 dB: no peak changes on the three Twin test nights. Runs at the default 3 tapers are unchanged. Runs at another taper count now lose the peaks below the floor. No settings value changes shape or meaning; the migration is a changelog no-op. |
/// | 15    | 2026-11-17 | `PrefPhase` and `Coupling` in the power paramfit CSV are NaN for a mode whose `FreqMean` lies outside the phase fit's frequency window. The phase model is zero there, so the argmax returned the first SO-phase bin with a coupling of 0, a value that reads as real. Only reachable when the power and phase fits use different frequency windows, or a frequency row is invalid for the phase fit only. Every other value is unchanged. No settings value changes shape or meaning; the migration is a changelog no-op. |
/// | 16    | 2026-11-19 | The TF-peak region labels found on the downsampled segment are resized to full resolution exactly as MATLAB `imresize(..., 'nearest')` does (kernel `resize_labels_nn`, DYNAM-O_rs #39). A full-resolution column that falls halfway between two downsampled columns was rounded half to even. `imresize` settles it by its own floating-point arithmetic, so for a 571-column segment (30 s at the default 0.05 s step) column 286 read a different downsampled column than in MATLAB, and regions touching it entered the trim with a few more or fewer pixels. On the three Twin test nights 2 of 136,238 peaks move (by at most 0.028 s and 0.007 Hz), no peak is added or removed, and every paramfit value is unchanged. No settings value changes shape or meaning; the migration is a changelog no-op. |
///
/// A value changing meaning without changing its name or type is a rare
/// and deliberate act; entries land here only when the same file would
/// yield different outputs under new code. The read path stays
/// regardless: a few hundred existing files are v1, and the conversion
/// from `√2·σ` to `σ` has to remain available or those files become
/// unmigratable. Freezing the table is not a reason to delete the field.
pub const SEMANTICS_VERSION: u32 = 16;

/// One migration step between adjacent [`SEMANTICS_VERSION`] values.
///
/// Only meaning changes need a step. A field added, removed or renamed needs
/// nothing: an absent key already takes the current default, and an extra
/// key is already ignored.
pub struct Migration {
    /// Upgrades a file at this semantics version to `from + 1`.
    pub from: u32,
    pub summary: &'static str,
    /// Rewrites `settings` in place, returning one line per change made.
    /// A step that changes no values still returns lines, because a
    /// behavior change the user cannot see in a diff is exactly what they
    /// most need told.
    pub apply: fn(&mut Value) -> Vec<String>,
}

/// Divide a numeric leaf by `sqrt(2)`, logging it. Silent when the path is
/// absent, since an absent key takes the (already correct) default.
fn rescale_sigma(settings: &mut Value, dotted: &str, log: &mut Vec<String>) {
    let segs: Vec<&str> = dotted.split('.').collect();
    let mut cur = settings;
    for s in &segs[..segs.len() - 1] {
        match cur.get_mut(*s) {
            Some(next) => cur = next,
            None => return,
        }
    }
    let last = segs[segs.len() - 1];
    // Only a numeric leaf is rescaled. An absent key or a non-numeric value
    // is left alone and logs nothing.
    let Some(leaf) = cur.get_mut(last) else {
        return;
    };
    let Some(original) = leaf.as_f64() else {
        return;
    };
    let scaled = original / std::f64::consts::SQRT_2;
    *leaf = serde_json::json!(scaled);
    log.push(format!(
        "    {dotted}: {} -> {}",
        last_num(original),
        last_num(scaled)
    ));
}

fn last_num(v: f64) -> String {
    format!("{v:.6}")
}

fn migrate_v1_to_v2(settings: &mut Value) -> Vec<String> {
    let mut log = vec![
        "  widths become true standard deviations (divide by sqrt(2)):".to_string(),
    ];
    // Power: both widths are rotGauss sigmas.
    for k in ["fstd_lb", "fstd_ub", "pstd_lb", "pstd_ub"] {
        rescale_sigma(settings, &format!("paramfit.power.mode_bounds.{k}"), &mut log);
    }
    // Phase: ONLY the frequency width. `pstd_*` is recikappa, which the von
    // Mises factor already made a true sigma, so rescaling it would be wrong.
    for k in ["fstd_lb", "fstd_ub"] {
        rescale_sigma(settings, &format!("paramfit.phase.mode_bounds.{k}"), &mut log);
    }
    log.push(
        "    paramfit.phase.mode_bounds.pstd_* left alone (recikappa was always a sigma)"
            .to_string(),
    );
    log
}

fn migrate_v2_to_v3(_settings: &mut Value) -> Vec<String> {
    vec![
        "  no values change. constrain_freq_center / constrain_power_center now".to_string(),
        "  constrain to the VALID (non-NaN) data inside the fit window rather than".to_string(),
        "  to the window itself, so fits will differ wherever hist.min_time_in_bin".to_string(),
        "  emptied SO-power bins. Nothing to edit; this is recorded so the shift is".to_string(),
        "  not a surprise.".to_string(),
    ]
}

fn migrate_v3_to_v4(settings: &mut Value) -> Vec<String> {
    // peak_assign_prob is absorbed into the flexible peak_assign, whose
    // fresh default is "argmax". A pre-v4 file keeps its exact old meaning:
    // the stored probability (0.95 when the key was absent) becomes the
    // "<x> p" form and the retired key is removed.
    let mut log = Vec::new();
    let old = settings
        .get_mut("paramfit")
        .and_then(|p| p.as_object_mut())
        .map(|p| (p.remove("peak_assign_prob"), p));
    if let Some((removed, paramfit)) = old {
        let prob = removed.as_ref().and_then(|v| v.as_f64()).unwrap_or(0.95);
        let form = format!("{prob} p");
        paramfit.insert("peak_assign".to_string(), serde_json::json!(form));
        match removed {
            Some(v) => log.push(format!(
                "    paramfit.peak_assign_prob: {v} -> paramfit.peak_assign: \"{form}\""
            )),
            None => log.push(format!(
                "    paramfit.peak_assign: \"{form}\" (pinning the old default; the fresh default is now \"argmax\")"
            )),
        }
    } else {
        // No paramfit block at all — the absent key took the old 0.95
        // default, so pin it explicitly rather than inherit "argmax".
        settings["paramfit"] = serde_json::json!({ "peak_assign": "0.95 p" });
        log.push(
            "    paramfit.peak_assign: \"0.95 p\" (pinning the old default; the fresh default is now \"argmax\")"
                .to_string(),
        );
    }
    log.push("  new runs written from defaults use peak_assign: \"argmax\"".to_string());
    log
}

/// Ordered migration steps. Index `i` upgrades version `i + 1` to `i + 2`.
pub fn migrations() -> Vec<Migration> {
    vec![
        Migration {
            from: 1,
            summary: "paramfit widths become true standard deviations",
            apply: migrate_v1_to_v2,
        },
        Migration {
            from: 2,
            summary: "mode centers bound by valid data, not the raw fit window",
            apply: migrate_v2_to_v3,
        },
        Migration {
            from: 3,
            summary: "peak_assign_prob absorbed into peak_assign (new default: argmax)",
            apply: migrate_v3_to_v4,
        },
        Migration {
            from: 4,
            summary:
                "Pk* columns aggregate by median (was mean); new PkPower column \
                 re-anchored to the persisted extraction baseline",
            apply: migrate_v4_to_v5,
        },
        Migration {
            from: 5,
            summary:
                "per-peak SOphase interpolated from the unwrapped series, then wrapped \
                 (peaks straddling the ±π seam were getting an arbitrary phase)",
            apply: migrate_v5_to_v6,
        },
        Migration {
            from: 6,
            summary:
                "per-peak PeakStage / SOpower / SOphase come from the kernel's \
                 MATLAB-exact assign_peak_* (artifact override tests the nearest EEG \
                 sample, not the previous one)",
            apply: migrate_v6_to_v7,
        },
        Migration {
            from: 7,
            summary:
                "artifact detection folds the slope-test mask in before the band tests \
                 and aligns even-length smoothing windows as MATLAB does",
            apply: migrate_v7_to_v8,
        },
        Migration {
            from: 8,
            summary:
                "TF-peak extraction follows the pure-MATLAB watershed numbering and \
                 merge (encircled-region absorption, strict merge threshold, MATLAB \
                 merge order)",
            apply: migrate_v8_to_v9,
        },
        Migration {
            from: 9,
            summary:
                "peak-frequency refinement uses MATLAB's not-a-knot spline instead of \
                 a natural spline",
            apply: migrate_v9_to_v10,
        },
        Migration {
            from: 10,
            summary:
                "the peak BoundingBox is the exact pixel extent (corner half a bin \
                 earlier and lower), which is also the refinement's frequency window",
            apply: migrate_v10_to_v11,
        },
        Migration {
            from: 11,
            summary:
                "the pass-2 mask zeros each region's traced contour as MATLAB does, \
                 keeping pixels that touch the outside only at a corner",
            apply: migrate_v11_to_v12,
        },
        Migration {
            from: 12,
            summary:
                "zero-phase filtering runs each section forward and backward in turn, \
                 as MATLAB filtfilt does",
            apply: migrate_v12_to_v13,
        },
        Migration {
            from: 13,
            summary:
                "the minimum TF-peak height follows the chi-squared rule for any \
                 number of tapers",
            apply: migrate_v13_to_v14,
        },
        Migration {
            from: 14,
            summary:
                "PrefPhase and Coupling are NaN for a power mode outside the phase \
                 fit's frequency window",
            apply: migrate_v14_to_v15,
        },
        Migration {
            from: 15,
            summary: "region labels are upsampled to full resolution the way MATLAB imresize does",
            apply: migrate_v15_to_v16,
        },
    ]
}

/// v15 → v16: a computation-semantics change with NO settings rewrite — one time column per segment of the upsampled TF-peak region labels follows MATLAB `imresize`, every settings value keeps its shape and meaning.
/// Log-only, like the earlier computation-only versions.
fn migrate_v15_to_v16(_settings: &mut Value) -> Vec<String> {
    vec!["  no settings values change; TF-peaks computed under v16 upsample the downsampled region labels as MATLAB imresize does, which moves a small number of peaks slightly".to_string()]
}

/// v14 → v15: computation semantics only, with no settings rewrite.
/// `PrefPhase` and `Coupling` are NaN for a power mode outside the phase
/// fit's frequency window. Every settings value keeps its shape and meaning,
/// so this step only logs.
fn migrate_v14_to_v15(_settings: &mut Value) -> Vec<String> {
    vec![
        "  no settings values change. Outputs computed under v15 write NaN for \
         PrefPhase and Coupling when a power mode lies outside the phase fit's \
         frequency window, where they held the first phase bin and 0"
            .to_string(),
    ]
}

/// v13 → v14: computation semantics only, with no settings rewrite. The
/// minimum TF-peak height is computed for every taper count. Every settings
/// value keeps its shape and meaning, so this step only logs.
fn migrate_v13_to_v14(_settings: &mut Value) -> Vec<String> {
    vec![
        "  no settings values change. Outputs computed under v14 apply MATLAB's \
         minimum peak height for any mtm.n_tapers. Runs at the default 3 tapers are \
         unchanged, runs at another count lose the peaks below the floor"
            .to_string(),
    ]
}

/// v12 → v13: computation semantics only, with no settings rewrite. The zero-
/// phase filter treats the ends of a recording as MATLAB `filtfilt` does.
/// Every settings value keeps its shape and meaning, so this step only logs.
fn migrate_v12_to_v13(_settings: &mut Value) -> Vec<String> {
    vec![
        "  no settings values change. Outputs computed under v13 filter the \
         slow-oscillation and artifact bands the way MATLAB filtfilt does. SO-phase \
         changes in the first and last minutes of a recording, everything else is \
         unchanged or nearly so"
            .to_string(),
    ]
}

/// v11 → v12: computation semantics only, with no settings rewrite. The
/// pass-2 mask zeros each region's traced contour, as MATLAB does. Every
/// settings value keeps its shape and meaning, so this step only logs.
fn migrate_v11_to_v12(_settings: &mut Value) -> Vec<String> {
    vec![
        "  no settings values change. Outputs computed under v12 mask the pass-2 \
         spectrogram with each region's traced contour, as MATLAB does. About 5% more \
         of each region's pixels are kept, the peak count rises by about 2.5%, and \
         most peaks change slightly"
            .to_string(),
    ]
}

/// v10 → v11: computation semantics only, with no settings rewrite. The peak
/// BoundingBox is the exact extent of the region's pixels. Every settings
/// value keeps its shape and meaning, so this step only logs.
fn migrate_v10_to_v11(_settings: &mut Value) -> Vec<String> {
    vec![
        "  no settings values change. Outputs computed under v11 report each peak's \
         BoundingBox as the exact extent of its pixels (bbox_tl_s half a time bin \
         earlier, bbox_tl_Hz half a frequency bin lower). The refinement searches that \
         window, so about 1% more peaks survive the edge-peak removal"
            .to_string(),
    ]
}

/// v9 → v10: computation semantics only, with no settings rewrite. The
/// refinement spline has MATLAB's not-a-knot end condition. Every settings
/// value keeps its shape and meaning, so this step only logs.
fn migrate_v9_to_v10(_settings: &mut Value) -> Vec<String> {
    vec![
        "  no settings values change. Outputs computed under v10 refine PeakFrequency \
         with a not-a-knot spline, as MATLAB does. About 0.1% of peaks, those whose \
         bounding box reaches the edge of the analyzed band, move by up to a few \
         tenths of a Hz"
            .to_string(),
    ]
}

/// v8 → v9: computation semantics only, with no settings rewrite. The TF-peak
/// watershed numbering and merge follow the pure-MATLAB code. Every settings
/// value keeps its shape and meaning, so this step only logs.
fn migrate_v8_to_v9(_settings: &mut Value) -> Vec<String> {
    vec![
        "  no settings values change. Outputs computed under v9 extract TF peaks with \
         MATLAB's merge: a region left with the merged region as its only neighbor is \
         absorbed, an edge exactly at merge_thresh no longer merges, and regions are \
         numbered and merged in MATLAB's order. Most peaks move slightly and the peak \
         count drops by about 2%, so earlier outputs differ throughout"
            .to_string(),
    ]
}

/// v7 → v8: computation semantics only, with no settings rewrite. The
/// artifact detector follows MATLAB `detect_artifacts.m`. Every settings
/// value keeps its shape and meaning, so this step only logs.
fn migrate_v7_to_v8(_settings: &mut Value) -> Vec<String> {
    vec![
        "  no settings values change. Outputs computed under v8 use an artifact mask \
         that matches MATLAB's: slope-flagged samples are excluded before the \
         high-frequency and broadband tests, and band-detected spans sit one sample \
         earlier. The mask feeds the baseline and every per-peak value, so earlier \
         outputs can differ throughout"
            .to_string(),
    ]
}

/// v6 → v7: computation semantics only, with no settings rewrite. The per-
/// peak stage, SO-power, and SO-phase lookups are the kernel's MATLAB-exact
/// functions. Every settings value keeps its shape and meaning, so this step
/// only logs.
fn migrate_v6_to_v7(_settings: &mut Value) -> Vec<String> {
    vec![
        "  no settings values change. Outputs computed under v7 label a peak as an \
         artifact (PeakStage 6) when the EEG sample NEAREST its time is flagged, as \
         MATLAB does. Earlier outputs tested the sample at or before the peak time, so \
         PeakStage can differ for peaks within half a sample of an artifact-span edge"
            .to_string(),
    ]
}

/// v5 → v6: computation semantics only, with no settings rewrite. Per-peak
/// SO-phase is interpolated from the unwrapped series and wrapped afterwards.
/// Every settings value keeps its shape and meaning, so this step only logs.
fn migrate_v5_to_v6(_settings: &mut Value) -> Vec<String> {
    vec![
        "  no settings values change. Outputs computed under v6 sample each peak's \
         SOphase from the unwrapped SO-phase series and wrap afterwards, so peaks \
         between two EEG samples that straddle ±π keep a phase near ±π (they were \
         spread across (−π, π))"
            .to_string(),
    ]
}

/// v4 → v5: computation semantics only, with no settings rewrite. The Pk*
/// columns aggregate by median, and PkPower exists. Every settings value
/// keeps its shape and meaning, so this step only logs.
fn migrate_v4_to_v5(_settings: &mut Value) -> Vec<String> {
    vec![
        "  no settings values change. Outputs computed under v5 aggregate the Pk* \
         peak-property columns by median (was mean) and add PkPower"
            .to_string(),
    ]
}

/// Apply every step from `from` up to [`SEMANTICS_VERSION`].
///
/// `from = 0` means the file carries no `semantics_version`. Every build that
/// wrote an unstamped file predates the sigma work, so an unstamped file is
/// treated as v1.
pub fn migrate(settings: &mut Value, from: u32) -> Vec<String> {
    let start = if from == 0 { 1 } else { from };
    let mut log = Vec::new();
    for m in migrations() {
        if m.from >= start && m.from < SEMANTICS_VERSION {
            log.push(format!("v{} -> v{}: {}", m.from, m.from + 1, m.summary));
            log.extend((m.apply)(settings));
        }
    }
    log
}

/// Values that were the default when a file was written and have since been
/// corrected. Not a semantics change, so [`migrate`] does not touch them: an
/// explicitly pinned value might be a deliberate choice. Reported so the
/// reader can decide.
///
/// `(dotted path, superseded value, current value, why)`. The two values are
/// JSON literals, compared NUMERICALLY by [`json_num_eq`] rather than as
/// text: a real file writes `[2, 16]` where this table says `[2.0, 16.0]`,
/// and a string comparison silently never matches.
pub const SUPERSEDED_DEFAULTS: &[(&str, &str, &str, &str)] = &[
    (
        "splinefit.power.freq_limits",
        "[2.0, 16.0]",
        "[2.0, 18.0]",
        "MATLAB spline_basis_opts.m sets [2,18]; [2,16] came from that file's stale docstring",
    ),
    (
        "splinefit.phase.freq_limits",
        "[2.0, 16.0]",
        "[2.0, 18.0]",
        "MATLAB spline_basis_opts.m sets [2,18]; [2,16] came from that file's stale docstring",
    ),
    (
        "paramfit.power.max_overlap",
        "0.6",
        "0.25",
        "calibrated for the containment metric sum(min)/sum(smaller); the metric is now MATLAB's \
         IoU sum(min)/sum(max), on which 0.6 is so permissive the revert guard never fires",
    ),
    (
        "paramfit.phase.max_overlap",
        "0.55",
        "0.15",
        "calibrated for the containment metric sum(min)/sum(smaller); the metric is now MATLAB's \
         IoU sum(min)/sum(max), on which 0.55 is so permissive the revert guard never fires",
    ),
];

/// Compare two JSON values, treating numbers as equal when they are
/// numerically equal regardless of how they were written. `2` and `2.0` are
/// the same setting; serde_json keeps them as distinct tokens.
pub fn json_num_eq(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => match (x.as_f64(), y.as_f64()) {
            (Some(p), Some(q)) => (p - q).abs() <= 1e-12 * p.abs().max(q.abs()).max(1.0),
            _ => x == y,
        },
        (Value::Array(x), Value::Array(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(p, q)| json_num_eq(p, q))
        }
        (Value::Object(x), Value::Object(y)) => {
            x.len() == y.len()
                && x.iter().all(|(k, v)| y.get(k).is_some_and(|w| json_num_eq(v, w)))
        }
        _ => a == b,
    }
}

/// One `(path, type)` pair per leaf of the settings tree.
fn shape(v: &Value, path: &str, out: &mut Vec<(String, &'static str)>) {
    match v {
        Value::Object(map) => {
            // BTreeMap-backed unless `preserve_order` is on; sort anyway so
            // the fingerprint never depends on serde_json's feature flags.
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            for k in keys {
                shape(&map[k], &format!("{path}.{k}"), out);
            }
        }
        Value::Array(items) => {
            // Length is a value, not a shape. A 2-element range and a
            // 3-element one are the same schema.
            out.push((path.to_string(), "array"));
            if let Some(first) = items.first() {
                shape(first, &format!("{path}[]"), out);
            }
        }
        // int vs float is a JSON serialization accident (`5` and `5.0` are
        // the same f64 in Rust), so both are "number" or the fingerprint
        // would flip on a value edit.
        Value::Number(_) => out.push((path.to_string(), "number")),
        Value::Bool(_) => out.push((path.to_string(), "bool")),
        // A `Bound` at a sentinel ("derived" / "inf" / "-inf"). The same slot
        // legitimately holds a number, so give it its own kind rather than
        // "string" — otherwise an explicit numeric bound reads as a type change.
        Value::String(t)
            if matches!(t.as_str(), "derived" | "inf" | "-inf") =>
        {
            out.push((path.to_string(), "bound"))
        }
        Value::String(_) => out.push((path.to_string(), "string")),
        // A null leaf is an `Option` at its None value. Its type is
        // unknowable from the instance, so record it as its own kind rather
        // than guess — otherwise the fingerprint would depend on whether a
        // default happened to be Some or None.
        Value::Null => out.push((path.to_string(), "nullable")),
    }
}

fn shape_of_defaults() -> Vec<(String, &'static str)> {
    let v = serde_json::to_value(DynamoSettings::default())
        .expect("DynamoSettings must serialize");
    let mut out = Vec::new();
    shape(&v, "", &mut out);
    out.sort();
    out
}

/// Stable identity of the settings *shape*. Changes when a field is added,
/// removed, renamed, or retyped. Does NOT change when a default value
/// changes, when a comment is edited, or when code is refactored.
pub fn fingerprint() -> String {
    let mut hasher = Fnv::new();
    for (p, t) in shape_of_defaults() {
        hasher.write(p.as_bytes());
        hasher.write(b":");
        hasher.write(t.as_bytes());
        hasher.write(b"\n");
    }
    format!("{:016x}", hasher.finish())
}

/// What a loaded settings file differs from the current schema by.
#[derive(Debug, Default, Clone)]
pub struct SettingsReport {
    /// In the schema, absent from the file. These took today's default,
    /// which may not be the default in force when the file was written.
    pub defaulted: Vec<String>,
    /// In the file, not in the schema. Silently ignored by serde — the file
    /// asked for something this build no longer offers.
    pub unknown: Vec<String>,
    /// Present in both but with a different type, e.g. a number that became
    /// a string. serde would reject the file outright, but naming the field
    /// beats a bare parse error.
    pub retyped: Vec<String>,
}

impl SettingsReport {
    pub fn is_clean(&self) -> bool {
        self.defaulted.is_empty() && self.unknown.is_empty() && self.retyped.is_empty()
    }

    /// Human-readable lines for the run log. Empty when clean.
    pub fn lines(&self) -> Vec<String> {
        let mut out = Vec::new();
        if !self.unknown.is_empty() {
            out.push(format!(
                "settings: {} key(s) in the file are not part of this build's schema and were IGNORED:",
                self.unknown.len()
            ));
            for k in &self.unknown {
                out.push(format!("    {k}"));
            }
        }
        if !self.retyped.is_empty() {
            out.push(format!(
                "settings: {} key(s) have a different type than the schema expects:",
                self.retyped.len()
            ));
            for k in &self.retyped {
                out.push(format!("    {k}"));
            }
        }
        if !self.defaulted.is_empty() {
            out.push(format!(
                "settings: {} key(s) were absent and took this build's defaults:",
                self.defaulted.len()
            ));
            for k in &self.defaulted {
                out.push(format!("    {k}"));
            }
        }
        out
    }
}

/// Compare a raw settings JSON against the current schema.
///
/// `raw` is the `settings` object as it appeared in the file, before serde
/// filled anything in. Comparing the parsed struct would be useless, since
/// by then every absent key already holds a default.
pub fn diff(raw: &Value) -> SettingsReport {
    let mut file_shape = Vec::new();
    shape(raw, "", &mut file_shape);
    // Types are compared as well as paths, so a field that changes type
    // without changing name is reported. It gets its own category instead
    // of being folded into the key (which would leak `[type]` into every
    // message).
    use std::collections::BTreeMap;
    let file_map: BTreeMap<String, &'static str> = file_shape.into_iter().collect();
    let default_map: BTreeMap<String, &'static str> = shape_of_defaults().into_iter().collect();

    let report = SettingsReport {
        defaulted: default_map.keys().filter(|k| !file_map.contains_key(*k)).cloned().collect(),
        unknown: file_map.keys().filter(|k| !default_map.contains_key(*k)).cloned().collect(),
        retyped: default_map
            .iter()
            .filter_map(|(k, want)| {
                let got = file_map.get(k)?;
                // "nullable" is an Option at its None value, and an Option
                // legitimately has two shapes: `null` or the inner type. A
                // default of None against a file that sets the field is
                // normal, not a type change.
                if *want == "nullable" || **got == *"nullable" {
                    return None;
                }
                // A `Bound` slot holds a number OR a sentinel string
                // ("derived" / "inf" / "-inf"), so neither direction is a
                // type change — an explicit numeric bound against a
                // defaulted "derived" is the ordinary case.
                let bound_ok = |a: &str, b: &str| {
                    a == "bound" && (b == "number" || b == "string")
                };
                if bound_ok(want, got) || bound_ok(got, want) {
                    return None;
                }
                (got != want).then(|| format!("{k}: file has {got}, schema expects {want}"))
            })
            .collect(),
    };

    report
}

/// Compare only the TOP-LEVEL key set of two objects.
///
/// For the manifest envelope, where a schema instance cannot describe nested
/// shape: `Manifest::default()` has empty vectors and `None` options, so a
/// deep walk reports every nested key as unknown. Comparing immediate keys
/// catches a top-level key appearing or vanishing.
///
/// `skip` names keys to leave out, so the envelope can exclude `settings`
/// and avoid reporting the same fields twice.
pub fn diff_top_level(raw: &Value, schema: &Value, skip: &[&str]) -> SettingsReport {
    let keys = |v: &Value| -> std::collections::BTreeSet<String> {
        v.as_object()
            .map(|o| {
                o.keys()
                    .filter(|k| !skip.contains(&k.as_str()))
                    .cloned()
                    .collect()
            })
            .unwrap_or_default()
    };
    let file = keys(raw);
    let want = keys(schema);
    SettingsReport {
        defaulted: want.difference(&file).cloned().collect(),
        unknown: file.difference(&want).cloned().collect(),
        retyped: Vec::new(),
    }
}

/// Resolve a leading-dot dotted path (`.paramfit.power.freq_limits`).
/// Used only by the SUPERSEDED_DEFAULTS test.
#[cfg(test)]
fn pointer<'a>(v: &'a Value, dotted: &str) -> Option<&'a Value> {
    let mut cur = v;
    for seg in dotted.trim_start_matches('.').split('.') {
        cur = cur.get(seg)?;
    }
    Some(cur)
}

/// FNV-1a. A hash here only needs to be stable across builds and platforms,
/// which rules out `DefaultHasher` (explicitly not stable) but does not
/// justify pulling in a crypto dependency.
struct Fnv(u64);
impl Fnv {
    fn new() -> Self {
        Fnv(0xcbf2_9ce4_8422_2325)
    }
    fn write(&mut self, bytes: &[u8]) {
        for b in bytes {
            self.0 ^= *b as u64;
            self.0 = self.0.wrapping_mul(0x1000_0000_01b3);
        }
    }
    fn finish(&self) -> u64 {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// E10: the v3→v4 migration pins a pre-v4 file to its old meaning
    /// while fresh defaults move to "argmax".
    #[test]
    fn peak_assign_migration_pins_the_old_meaning() {
        // Stored prob → "<x> p", retired key removed.
        let mut v = serde_json::json!({"paramfit": {"peak_assign_prob": 0.8}});
        let log = migrate(&mut v, 3);
        assert!(v["paramfit"].get("peak_assign_prob").is_none());
        assert_eq!(v["paramfit"]["peak_assign"], "0.8 p");
        assert!(log.iter().any(|l| l.contains("0.8 p")), "log names the pinned value: {log:?}");
        // Absent key → the old 0.95 default is pinned, not inherited.
        let mut v = serde_json::json!({"paramfit": {}});
        migrate(&mut v, 3);
        assert_eq!(v["paramfit"]["peak_assign"], "0.95 p");
        // No paramfit block at all — same pin.
        let mut v = serde_json::json!({});
        migrate(&mut v, 3);
        assert_eq!(v["paramfit"]["peak_assign"], "0.95 p");
        // A fresh default file serializes the new default (footprint
        // spelled explicitly).
        let d = serde_json::to_value(DynamoSettings::default()).unwrap();
        assert_eq!(d["paramfit"]["peak_assign"], "argmax 1.5 sigma");
        // The migrated form round-trips through the settings loader.
        let s: DynamoSettings =
            serde_json::from_value(serde_json::json!({"paramfit": {"peak_assign": "0.8 p"}}))
                .unwrap();
        assert_eq!(s.paramfit.peak_assign, crate::run::settings::PeakAssign::Prob(0.8));
    }

    /// Pins the settings shape. When this fails, the schema changed.
    ///
    /// Update the constant below, and then answer the question this test
    /// exists to force: **did any EXISTING field also change meaning?** If
    /// so, bump `SEMANTICS_VERSION` and add a row to its table. A pure
    /// add/remove/rename does not need a semantics bump.
    ///
    /// (Separately — no test can force it — a change that alters computed
    /// NUMBERS without touching any setting also bumps SEMANTICS_VERSION,
    /// with a no-op migration as its changelog entry. See the constant's
    /// doc.)
    #[test]
    fn schema_fingerprint_is_pinned() {
        const PINNED: &str = "762c897dc96551a3";
        let actual = fingerprint();
        assert_eq!(
            actual, PINNED,
            "\n\nThe settings schema changed.\n\
             Update PINNED to {actual}, then ask: did an EXISTING field also \
             change MEANING?\n  - shape only (field added/removed/renamed) -> no semantics bump\n  \
             - an existing field now means something different -> bump SEMANTICS_VERSION \
             and add a row to its table\n\
             Also: any change that alters computed NUMBERS bumps SEMANTICS_VERSION \
             with a no-op migration as its changelog entry (artifact stamps key the \
             is-this-tree-current verdict on it).\n\n"
        );
    }

    #[test]
    fn fingerprint_ignores_values_but_tracks_shape() {
        let base = serde_json::to_value(DynamoSettings::default()).unwrap();
        let fp = |v: &Value| {
            let mut s = Vec::new();
            shape(v, "", &mut s);
            s.sort();
            s
        };
        let baseline = fp(&base);

        // a value edit must NOT move the shape
        let mut v = base.clone();
        v["hist"]["min_time_in_bin"] = serde_json::json!(999.0);
        assert_eq!(fp(&v), baseline, "a value change must not move the fingerprint");

        // an added field must
        let mut v = base.clone();
        v["detection"]["brand_new_knob"] = serde_json::json!(1.0);
        assert_ne!(fp(&v), baseline, "an added field must move the fingerprint");

        // a removed field must
        let mut v = base.clone();
        v["detection"].as_object_mut().unwrap().remove("merge_thresh");
        assert_ne!(fp(&v), baseline, "a removed field must move the fingerprint");
    }

    #[test]
    fn diff_names_absent_and_unknown_keys() {
        let mut v = serde_json::to_value(DynamoSettings::default()).unwrap();
        v["detection"].as_object_mut().unwrap().remove("merge_thresh");
        v["detection"]["retired_knob"] = serde_json::json!(7);

        let r = diff(&v);
        assert!(r.defaulted.iter().any(|k| k == ".detection.merge_thresh"),
            "absent key not reported: {:?}", r.defaulted);
        assert!(r.unknown.iter().any(|k| k == ".detection.retired_knob"),
            "unknown key not reported: {:?}", r.unknown);
        assert!(!r.is_clean());
    }

    #[test]
    fn diff_is_clean_for_a_freshly_written_file() {
        let v = serde_json::to_value(DynamoSettings::default()).unwrap();
        let r = diff(&v);
        assert!(r.is_clean(), "defaults must round-trip clean: {:?}", r);
    }



    /// The v1 -> v2 step must rescale exactly the six Gaussian widths and
    /// leave `recikappa` alone. Rescaling it too is the single most likely
    /// way to get this migration wrong, and it would be invisible in the
    /// output.
    #[test]
    fn v1_to_v2_rescales_only_the_gaussian_widths() {
        let mut v = serde_json::to_value(DynamoSettings::default()).unwrap();
        // Put a v1-convention file together: widths sqrt(2) larger.
        let s2 = std::f64::consts::SQRT_2;
        for (ax, keys) in [
            ("power", vec!["fstd_lb", "fstd_ub", "pstd_lb", "pstd_ub"]),
            ("phase", vec!["fstd_lb", "fstd_ub"]),
        ] {
            for k in keys {
                let cur = v["paramfit"][ax]["mode_bounds"][k].as_f64().unwrap();
                v["paramfit"][ax]["mode_bounds"][k] = serde_json::json!(cur * s2);
            }
        }
        let phase_pstd_before = v["paramfit"]["phase"]["mode_bounds"]["pstd_ub"].as_f64().unwrap();
        let theta_before = v["paramfit"]["power"]["mode_bounds"]["theta_ub"].as_f64().unwrap();

        let log = migrate(&mut v, 1);
        assert!(!log.is_empty(), "migration produced no log");

        // The six widths land back on the current defaults.
        let d = serde_json::to_value(DynamoSettings::default()).unwrap();
        for (ax, keys) in [
            ("power", vec!["fstd_lb", "fstd_ub", "pstd_lb", "pstd_ub"]),
            ("phase", vec!["fstd_lb", "fstd_ub"]),
        ] {
            for k in keys {
                let got = v["paramfit"][ax]["mode_bounds"][k].as_f64().unwrap();
                let want = d["paramfit"][ax]["mode_bounds"][k].as_f64().unwrap();
                assert!((got - want).abs() < 1e-12, "{ax}.{k}: {got} != {want}");
            }
        }
        // recikappa and theta must be untouched.
        assert_eq!(
            v["paramfit"]["phase"]["mode_bounds"]["pstd_ub"].as_f64().unwrap(),
            phase_pstd_before,
            "phase pstd_ub (recikappa) was wrongly rescaled",
        );
        assert_eq!(
            v["paramfit"]["power"]["mode_bounds"]["theta_ub"].as_f64().unwrap(),
            theta_before,
            "theta_ub was wrongly rescaled",
        );
    }

    /// An unstamped file is v1, and migrating an already-current file is a
    /// no-op rather than a second rescale.
    #[test]
    fn migrate_is_idempotent_and_treats_unstamped_as_v1() {
        let d = serde_json::to_value(DynamoSettings::default()).unwrap();

        let mut current = d.clone();
        let log = migrate(&mut current, SEMANTICS_VERSION);
        assert!(log.is_empty(), "a current file should need no migration: {log:?}");
        assert_eq!(current, d, "migrating a current file changed values");

        // from = 0 (no stamp) must behave exactly like from = 1.
        let mut a = d.clone();
        let mut b = d.clone();
        assert_eq!(migrate(&mut a, 0).len(), migrate(&mut b, 1).len());
        assert_eq!(a, b);
    }

    /// An Option field has two legitimate shapes: `null`, or the inner
    /// type. A default of None against a file that sets the field is normal
    /// and must not be reported as a type change. The case here is
    /// `output.csv_time_sigfigs` (Option<u8>, default None) set to 8.
    #[test]
    fn retyped_ignores_option_null_versus_value() {
        let mut v = serde_json::to_value(DynamoSettings::default()).unwrap();
        assert!(v["output"]["csv_time_sigfigs"].is_null(), "test assumes this defaults to null");
        v["output"]["csv_time_sigfigs"] = serde_json::json!(8);
        let r = diff(&v);
        assert!(
            r.retyped.is_empty(),
            "setting an Option field was reported as a type change: {:?}",
            r.retyped,
        );

        // A genuine type change is still caught.
        let mut v = serde_json::to_value(DynamoSettings::default()).unwrap();
        v["detection"]["merge_thresh"] = serde_json::json!("eleven");
        assert!(
            diff(&v).retyped.iter().any(|s| s.starts_with(".detection.merge_thresh")),
            "a number becoming a string must be reported",
        );
    }

    /// A real settings file writes `[2, 16]` where SUPERSEDED_DEFAULTS says
    /// `[2.0, 16.0]`. Comparing as text silently never matches, so the
    /// splinefit entry would never fire on a real file.
    #[test]
    fn json_num_eq_ignores_int_float_spelling() {
        let file: Value = serde_json::from_str("[2, 16]").unwrap();
        let table: Value = serde_json::from_str("[2.0, 16.0]").unwrap();
        assert!(json_num_eq(&file, &table), "int/float spelling must compare equal");
        assert_ne!(file.to_string(), table.to_string(), "…and text compare must not");

        // Still discriminating.
        let other: Value = serde_json::from_str("[2, 18]").unwrap();
        assert!(!json_num_eq(&file, &other));
        // Nested and mixed.
        let a: Value = serde_json::from_str(r#"{"k":[1,{"j":2}]}"#).unwrap();
        let b: Value = serde_json::from_str(r#"{"k":[1.0,{"j":2.0}]}"#).unwrap();
        assert!(json_num_eq(&a, &b));
    }

    /// Every SUPERSEDED_DEFAULTS entry must parse and must actually differ
    /// from the current default, or it is dead weight that can never fire.
    #[test]
    fn superseded_defaults_are_parseable_and_actually_superseded() {
        let d = serde_json::to_value(DynamoSettings::default()).unwrap();
        for (path, old, new, _) in SUPERSEDED_DEFAULTS {
            let old_v: Value = serde_json::from_str(old)
                .unwrap_or_else(|e| panic!("{path}: old value {old} is not JSON: {e}"));
            let new_v: Value = serde_json::from_str(new)
                .unwrap_or_else(|e| panic!("{path}: new value {new} is not JSON: {e}"));
            assert!(!json_num_eq(&old_v, &new_v), "{path}: old and new are the same value");
            let cur = pointer(&d, &format!(".{path}"))
                .unwrap_or_else(|| panic!("{path} is not in the schema"));
            assert!(
                json_num_eq(cur, &new_v),
                "{path}: table says the current default is {new}, schema says {cur}",
            );
        }
    }

    /// SEMANTICS_VERSION is really an index into the migration table, so it
    /// is fully determined by that table: N steps means version N+1. Keeping
    /// them in sync by hand is the one part a reader could get wrong
    /// silently, and a file stamped with a version that has no matching
    /// migration path is unmigratable.
    #[test]
    fn semantics_version_matches_the_migration_table() {
        let steps = migrations();
        assert_eq!(
            SEMANTICS_VERSION as usize,
            steps.len() + 1,
            "SEMANTICS_VERSION is {} but there are {} migration step(s). \
             Bumping one without the other leaves a version nothing can migrate \
             to or from.",
            SEMANTICS_VERSION,
            steps.len(),
        );
        // And the steps must form an unbroken 1..N chain, or migrate() would
        // skip a version silently.
        for (i, m) in steps.iter().enumerate() {
            assert_eq!(
                m.from as usize,
                i + 1,
                "migration {i} upgrades from v{}, expected v{}. The table must be \
                 a contiguous chain starting at 1.",
                m.from,
                i + 1,
            );
        }
    }

}
