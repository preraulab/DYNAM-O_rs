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

Measured end-to-end against `runDYNAMO('night', 'backend', 'matlab')` on the
bundled example recording:

| Metric | Rust (`dynamo_rs`) | MATLAB (reference) | Δ |
|---|---:|---:|---:|
| Full-night peak count | 34 511 | 34 788 | **−0.80 %** |
| Full-night wallclock (M3, 8-core) | ~30 s | ~125 s | **4.2× speedup** |
| SO-power histogram cosine similarity | — | — | **0.999** |
| SO-phase histogram cosine similarity | — | — | **0.996** |

The remaining −0.8 % peak-count gap is a subtle label-assignment-order
difference in the merge step (pixel sets of painted regions match 100 %; it
only shifts ~270 peaks across the bandwidth/duration filter cutoffs).

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
