# DYNAM-O_rs

Pure Rust implementation of the DYNAM-O pipeline — TF-peak extraction (double
watershed + merge + MATLAB-paint border + trim + Hann refinement), SO-power
and SO-phase time-series + 2D histograms, two-band artifact detection,
baseline subtraction, peak-stage/SO assignment, and EDF / staging I/O. Ships
as a library, a C ABI for MATLAB MEX, a PyO3 extension for pydynamo, and a
standalone `dynamo` CLI binary.

This crate is the Rust core shared by:

- **[DYNAM-O](https://github.com/preraulab/DYNAM-O)** — MATLAB toolbox. The `backend='rust'` path calls `dynamo_rs` via MEX wrappers (`DYNAMO_dev/rust_bridge/`).
- **[pyDYNAM-O](https://github.com/preraulab/DYNAM-O_py)** — Python port. Uses `dynamo_rs` via PyO3 bindings.
- **[DYNAM-O_toolbox](https://github.com/preraulab/DYNAM-O_toolbox)** — parent meta-repo that pins all three as git submodules.
- **Standalone `dynamo` CLI** — native binary, no MATLAB or Python dependency at runtime. See *CLI usage* below.

---

## Accuracy vs MATLAB reference

Measured head-to-head on the bundled night recording via
`benchmark_runDYNAMO` (warm, 3-trial median per backend), 2026-04-24
on an M3 8-core, MATLAB R2025b, Rust release build (fat LTO +
codegen-units = 1). Committed JSON at
[`rust_bridge/benchmarks/runs/2026-04-24-140745__DIMHDXTD6TQFV__darwin-arm64.json`](../../../../DYNAM-O_dev/blob/rust-bridge/rust_bridge/benchmarks/runs/2026-04-24-140745__DIMHDXTD6TQFV__darwin-arm64.json).

**Backend contract:** `backend='matlab'` is a pure-MATLAB reference
implementation (except for the bundled `multitaper_spectrogram_mex`, which
predates the rust_bridge work). All four Rust-backed MEX wrappers
(`extract_tfpeaks_mex`, `mask_spectrogram_mex`, `refine_peaks_mex`,
`tfpeak_histogram_mex`) are gated behind `backend='rust'`.

| Stage | `backend='matlab'` (pure MATLAB) | `backend='rust'` (MEX) | Speedup |
|---|---:|---:|---:|
| **Total `runDYNAMO('night')`** | **163.5 s** | **34.8 s** | **4.70×** |
| Combined Rust extract (pass 1 + 2) | 144.9 s | 16.1 s | **9.0×** |
| Extract pass 1 | 100.7 s | 10.2 s | 9.8× |
| Extract pass 2 | 44.2 s | 5.9 s | 7.5× |
| Peak refinement | 0.9 s (warm parfor) | 0.3 s | 2.8× |
| Histogram binning (SO-power + SO-phase) | ~3.5 s (pure-MATLAB loop) | 0.23 s (MEX) | ~15× |

| Peak count | `backend='matlab'` | `backend='rust'` | Δ |
|---|---:|---:|---:|
| Pass 1 (raw) | 65 829 | 70 115 | Rust +6.5% |
| **Pass 2 (final, post-rejection)** | **34 788** | **34 579** | **Rust −0.60%** |

The final **−0.60 % peak-count gap** is tighter than the historical
~−0.8 %, reflecting the edge-peak refine + `trim_shift` global-min fixes
landed 2026-04-24 (both pull Rust toward MATLAB's retention behaviour).
The remaining gap is watershed border tie-breaking between MATLAB's IPT
implementation and `matlab_watershed.rs` — pixel sets of painted regions
match 100 %, but a handful of peaks land on opposite sides of the
bandwidth/duration filter cutoffs. Pass 1 diverges more (Rust +6.5 %)
because pass-1 sees raw watershed output; the pass-2 mask absorbs most
of that, and the two paths converge.

### Recent refinements (2026-04-24)

Four correctness + perf changes tightened parity and shaved extract time:

1. **`trim_shift` parity** — `runSegmentedData.m` now passes the MATLAB
   global `min(spect/baseline, [], 'all')` instead of NaN (which routed Rust
   to its per-segment-min fallback). Both backends see the same shift.
2. **Edge-peak refine** — peaks within `window_size/2` of data start/end
   keep their pass-2 bbox-centroid `PeakFrequency` unchanged instead of
   running the Hann refine on a zero-padded partial window (matches
   `refinePeakFrequency.m:141`).
3. **Heap-based merge loop** — `merge.rs` replaces two O(|E|) linear scans
   per iteration (max-find + retain-filter-on-src) with a `BinaryHeap` +
   lazy deletion via per-edge generation counters. Tie-break preserved via
   insertion order (earliest-label-pair wins ties, matches MATLAB's
   edge-index convention). ~3 % off extract on night.
4. **`lto = "fat"` + `codegen-units = 1`** in the release profile — ~5 %
   off extract from full cross-unit inlining + dead-code elim. Dylib
   shrinks from 1.21 MB → 1.16 MB. Build time 14 s → 28 s.

Net wall-clock on the bundled night fixture: total `runDYNAMO('night')`
drops from 37.8 s to 36.5 s (−3.4 %); combined Rust extract (passes 1 + 2)
drops from 15.5 s to 14.3 s (−7.9 %). Peak-count parity vs MATLAB reference
is unchanged (still ~−0.8 %, dominated by merge tie-breaking not
`trim_shift`).

A separate fix unrelated to extract: `dynamo_tfpeak_histogram`'s C ABI
copy-out was writing row-major into MATLAB-column-major buffers, producing
~1 Hz striped SO-power / SO-phase histograms that crashed downstream
`fitParamBasis`. Now bit-identical to the pure-MATLAB binning loop
(`TFPeakHistogram.m`); warm MEX is 21 × faster than the MATLAB loop on
5 k-peak / 101 c-bin / 151 f-bin fixtures.

---

## Crate layout

Triple-target `cdylib` / `staticlib` / `rlib` with an optional `python`
feature for PyO3 bindings:

```toml
[lib]
crate-type = ["cdylib", "staticlib", "rlib"]

[features]
default = []
python  = ["dep:pyo3", "dep:numpy"]
```

```
rust/
  Cargo.toml
  src/
    lib.rs                # Rust API + PyO3 wrappers (behind `python` feature)
    c_api.rs              # extern "C" surface for MEX wrappers + cbindgen
    pipeline.rs           # top-level run_extract_from_spectrogram + write_stats_csv
    extract_pipeline.rs   # full extract (watershed+merge+paint+trim+stats)
    matlab_watershed.rs   # IPT-compatible watershed port (Vincent-Soille)
    merge.rs              # region adjacency graph + iterative merge
    trim.rs               # volume-based region trimming
    mask.rs               # pass-2 spectrogram masking
    refine.rs             # Hann-window peak-frequency refinement
    histogram.rs          # SO-power / SO-phase 2D histogram accumulator
    baseline.rs           # percentile-based baseline + build_baseline_exclude helper
    so_power.rs           # SO-power time-series pipeline (post-MTS)
    so_phase.rs           # SO-phase time-series (filter+hilbert+unwrap)
    peak_assign.rs        # per-peak stage / SO-power / SO-phase interpolation
    artifacts.rs          # two-band artifact detection (HF + BB, robust z-score)
    mts.rs                # thin wrapper around `multitaper_rs` crate
    signal.rs             # sosfiltfilt, hilbert, unwrap, movmean
    filter_cache.rs       # SOphase SOS cache (.npy) + cheby1 fallback
    filter_design.rs      # cheby1_sos (ported from scipy.signal.cheby1)
    adjacency.rs          # region adjacency utilities
    io/                   # edf, staging
    bin/dynamo.rs         # CLI entry point (`cargo build --bin dynamo`)
  include/
    dynamo_rs.h           # cbindgen-generated C header
data_matlab_filters/      # 43 pre-computed SOphase SOS filters (.npy)
```

---

## Build

### As a Rust library / C library (for MEX consumers)

```bash
cd rust
cargo build --release
```

Produces:
- `target/release/libdynamo_rs.{dylib,so,a}` (macOS / Linux; `.dll` + `.dll.lib` on Windows).
- `include/dynamo_rs.h` — regenerated on each build via `build.rs` + `cbindgen`.

MATLAB MEX wrappers live in `DYNAMO_dev/rust_bridge/` and link against these
artifacts. See
[`rust_bridge/README.md`](https://github.com/preraulab/DYNAM-O/blob/main/rust_bridge/README.md)
in the MATLAB repo for the end-to-end build recipe.

### As a Python extension (for pydynamo)

```bash
cd rust
maturin build --release --features python
# or, for in-place development:
maturin develop --release --features python
```

Produces `dynamo_rs*.whl`. Pydynamo's optional `import dynamo_rs` gate picks
it up automatically when available.

### As a standalone CLI (no MATLAB or Python needed)

```bash
cd rust
cargo build --release --bin dynamo
./target/release/dynamo extract \
    --spect  spect.npy  \
    --stimes stimes.npy \
    --sfreqs sfreqs.npy \
    --out    stats.csv
```

Currently covers the "from-spectrogram" slice: given a pre-computed
multitaper spectrogram as three `.npy` files, run the watershed / merge /
trim / region-props / filter pipeline and write a CSV with the same columns
as MATLAB's `stats_table`. Full EDF-to-CSV (multitaper + baseline + refine +
histograms) is follow-up work; the library primitives are all in place, the
CLI just needs stitching.

Defaults match `runDYNAMO`: `seg_time=30`, `downsample=(2,2)`,
`merge_thresh=11`, `trim_vol=0.8`, `dur_min=0.5`, `bw_min=2`, etc. All
overridable via flags.

### Regenerate the C header manually

The `build.rs` script invokes `cbindgen` on every `cargo build`. If you need
to regenerate manually:

```bash
cargo run --bin cbindgen -- --output include/dynamo_rs.h
```

---

## Consumers at a glance

| Client | How it links | Entry points |
|---|---|---|
| **MATLAB MEX** (`DYNAMO_dev/rust_bridge/`) | Classic-C MEX `.c` files link `-ldynamo_rs` at build, load the dylib at runtime via `dlopen` (macOS embeds rpath) | `dynamo_extract_tfpeaks`, `dynamo_mask_spectrogram`, `dynamo_refine_peaks`, `dynamo_tfpeak_histogram` — in `src/c_api.rs` |
| **Python** (`pydynamo`) | PyO3 extension (`maturin build --features python`) | `matlab_watershed`, `matlab_paint_labels`, `merge_segment`, `trim_regions`, `mask_spectrogram`, `compute_baseline`, `build_baseline_exclude`, `subtract_baseline`, `so_power_from_spectrogram`, `so_phase_from_eeg`, `detect_artifacts`, `hann_event_spectra`, `refine_from_spectra`, `tfpeak_histogram`, `hilbert`, `sosfiltfilt`, `movmean`, `unwrap`, … — in `src/lib.rs` under `#[pyfunction]` |
| **Rust** | `Cargo.toml` path or git dep | Public Rust items in `src/lib.rs` |
| **Standalone CLI** | `cargo build --release --bin dynamo` | `dynamo extract --spect ... --out stats.csv` — in `src/bin/dynamo.rs` |

---

## Border-handling: `matlab_paint_labels_in_order` (default)

The Rust extract pipeline supports two modes for filling watershed 0-lines
between merged regions. Controlled by `ExtractParams.expand_labels_distance`:

- **`0` (default)** → `matlab_paint_labels_in_order` — matches MATLAB's
  `Ldata(rgn{ii}) = ii` paint-in-label-order semantics (8-conn 1-px dilation
  per label, higher cell indices overwrite lower on shared borders). This is
  the mode the MATLAB backend expects and is what closes the previously
  +1.96 % peak-count gap to −0.8 %.

- **`N > 0`** → `expand_labels_bfs(distance=N)` — skimage-style distance-based
  BFS fill with lower-label-wins ties. Historical pydynamo default (5). Kept
  for backward compatibility; produces ~1–2 % more peaks than MATLAB.

See `src/extract_pipeline.rs::extract_tfpeaks_segment` for the call site.

---

## Branches

- `rust-bridge` — active development line for the three-repo restructure. MATLAB-paint default, matlab_watershed, c_api with `expand_labels_distance`.
- `main` — historical hybrid layout (Python + Rust + MATLAB shims together). Use only for archaeology.

Tag releases as `v0.x.y-<feature>` when cutting consumer-pinnable snapshots.

---

## License

BSD 3-Clause.
