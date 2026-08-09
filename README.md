# DYNAM-O_rs

Pure Rust implementation of the DYNAM-O pipeline — TF-peak extraction (double
watershed + merge + MATLAB-paint border + trim + Hann refinement), SO-power
and SO-phase time-series + 2D histograms, two-band artifact detection,
baseline subtraction, peak-stage/SO assignment, and EDF / staging I/O. Ships
as a library, a C ABI for MATLAB MEX, and a PyO3 extension for `pydynamo`. A
small `dynamo` binary exercises one slice of the pipeline directly from the
shell — see *The `dynamo` developer binary* below for what it does and does
not cover.

This crate is the Rust core shared by:

- **[DYNAM-O](https://github.com/preraulab/DYNAM-O)** — MATLAB toolbox. The `backend='rust'` path calls `dynamo_rs` via MEX wrappers (`DYNAM-O/rust_bridge/`).
- **[DYNAM-O_py](https://github.com/preraulab/DYNAM-O_py)** — Python port. Uses `dynamo_rs` via PyO3 bindings.
- **[DYNAM-O_toolbox](https://github.com/preraulab/DYNAM-O_toolbox)** — parent meta-repo that bootstraps all three as sibling repositories.
- **The `dynamo` binary** — a development utility for driving the extraction kernel from the shell, with no MATLAB or Python in the loop. It is not a general-purpose DYNAM-O command-line tool; see below.

---

## Accuracy vs MATLAB reference

Measured head-to-head on the bundled night recording via
`benchmark_runDYNAMO` (warm, 3-trial median per backend), 2026-04-24
on an M3 8-core, MATLAB R2025b, Rust release build (fat LTO +
codegen-units = 1).

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

### Recommended sampling frequency: 100 Hz

DYNAM-O analyzes 0–30 Hz, so 100 Hz Nyquist covers everything the
pipeline cares about. The multitaper-spectrogram NFFT is
`2^nextpow2(Fs / mtm_dsfreqs)` (default `mtm_dsfreqs = 0.1`), so
**Fs > 102.4 Hz** doubles NFFT and typically pushes the spectrogram
past CPU L3 cache — every downstream stage (extract / baseline /
mask / watershed / refine) takes a 2–3× memory-bandwidth hit on top
of the doubled FFT cost.

| Native Fs | NFFT | Spec stage cost vs 100 Hz |
|---:|---:|---:|
| ≤ 100 | 1024 | 1× |
| 128, 200 | 2048 | ~2.2× |
| 256 | 4096 | ~4.6× |
| 500, 512 | 8192 | ~9.3× |
| 1000 | 16384 | ~18× |

Resample to 100 Hz before feeding `dynamo extract` / `dynamo_extract_tfpeaks`
for ~2× end-to-end speedup with zero analytical loss for sleep oscillations.
Empirical: 10.5 h × 128 Hz EDF goes from ~41 s → ~22 s on a 32-core
Threadripper (Rust backend, full pipeline). The MATLAB FileManager has
this enabled by default; CLI / pydynamo callers should pass already-resampled
data.

---

## Crate layout

Dual-target `cdylib` / `rlib` with an optional `python`
feature for PyO3 bindings:

```toml
[lib]
crate-type = ["cdylib", "rlib"]

[features]
default = ["parallel"]
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
    bin/dynamo.rs         # standalone CLI entry point
  include/
    dynamo_rs.h           # cbindgen-generated C header
data_matlab_filters/      # 42 pre-computed SOphase SOS filters (.npy)
```

---

## Build

Build distributable artifacts from the parent `DYNAM-O_toolbox` checkout with
the controlled bootstrap entrypoint:

macOS or Linux:

```bash
cd <workspace>/DYNAM-O_toolbox
./bootstrap.sh --yes
```

Windows PowerShell:

```powershell
cd <workspace>\DYNAM-O_toolbox
.\bootstrap.ps1 -Yes
```

These entrypoints synchronize the repositories, establish source-path
remapping, record provenance, and run the mandatory privacy gate.

### As a Rust library / C library (for MEX consumers)

The direct Cargo command below is for local development only. Do not distribute
its outputs: it does not establish the controlled source-path remapping
environment.

```bash
cd rust
cargo build --release --locked --lib
```

Produces:
- `target/release/libdynamo_rs.dylib` on macOS, `libdynamo_rs.so` on Linux,
  or `dynamo_rs.dll` plus its import library on Windows.
- `target/release/libdynamo_rs.rlib`, an internal Cargo artifact for Rust
  consumers. It is not shipped.
- `include/dynamo_rs.h` — tracked and refreshed by `build.rs` + `cbindgen`
  when relevant Rust inputs change.

MATLAB MEX wrappers live in `DYNAM-O/rust_bridge/` and link against the
platform shared library. See
[`rust_bridge/README.md`](https://github.com/preraulab/DYNAM-O/blob/master/rust_bridge/README.md)
in the MATLAB repo for the end-to-end build recipe.

### As a Python extension (for pydynamo)

Use the controlled toolbox bootstrap above. It installs the pinned Maturin
version internally, builds and installs `dynamo_rs` into
`DYNAM-O_py/.venv` under the path-remapped environment, includes the 42
canonical SO-phase filters, sanitizes the installed SBOM, and runs the final
privacy gate.

The controlled workflow does not currently produce a standalone `dynamo_rs`
wheel for distribution. Native extensions or wheels produced by direct
Maturin, pip, or other PEP 517 commands are not controlled release artifacts
and must not be published as such.

### The `dynamo` developer binary

`src/bin/dynamo.rs` is a thin shell wrapper over one slice of the kernel. It
exists so the extraction path can be driven and profiled without MATLAB or
Python in the loop — useful when bisecting a parity difference or timing a
change. It is not a general-purpose DYNAM-O command-line tool, and is not
what you want for running a study.

**Scope: one subcommand, `extract`.** It takes a *pre-computed* multitaper
spectrogram as three `.npy` files and writes a peak stats CSV with the same
columns as MATLAB's `stats_table`:

```bash
cd rust
cargo build --release --locked --bin dynamo
./target/release/dynamo extract \
    --spect  spect.npy  \
    --stimes stimes.npy \
    --sfreqs sfreqs.npy \
    --out    stats.csv
```

**It does not read EDFs**, compute the spectrogram, subtract a baseline,
refine peaks, or build SO-power / SO-phase histograms. Producing the three
input arrays is the caller's problem. Everything upstream and downstream of
the watershed belongs to whichever front end you are driving the kernel
from — the MATLAB toolbox or `pydynamo` — and those remain the supported
ways to run the pipeline end to end.

The direct build above is for local development only. Use the controlled
toolbox bootstrap before distributing the executable.

Defaults match `runDYNAMO`: `seg_time=30`, `downsample=(2,2)`,
`merge_thresh=11`, `trim_vol=0.8`, `dur_min=0.5`, `bw_min=2`, etc. All
overridable via flags.

### C header

The generated `rust/include/dynamo_rs.h` is tracked. Cargo's `build.rs`
refreshes it through `cbindgen` when relevant Rust inputs change. Use the
controlled toolbox bootstrap when distributing native artifacts built against
an updated header.

---

## Consumers at a glance

| Client | How it links | Entry points |
|---|---|---|
| **MATLAB MEX** (`DYNAM-O/rust_bridge/`) | Classic-C MEX `.c` files link `-ldynamo_rs` at build, load the dylib at runtime via `dlopen` (macOS embeds rpath) | `dynamo_extract_tfpeaks`, `dynamo_mask_spectrogram`, `dynamo_refine_peaks`, `dynamo_tfpeak_histogram` — in `src/c_api.rs` |
| **Python** (`pydynamo`) | PyO3 extension installed by the controlled toolbox bootstrap | `matlab_watershed`, `matlab_paint_labels`, `merge_segment`, `trim_regions`, `mask_spectrogram`, `compute_baseline`, `build_baseline_exclude`, `subtract_baseline`, `so_power_from_spectrogram`, `so_phase_from_eeg`, `detect_artifacts`, `hann_event_spectra`, `refine_from_spectra`, `tfpeak_histogram`, `hilbert`, `sosfiltfilt`, `movmean`, `unwrap`, … — in `src/lib.rs` under `#[pyfunction]` |
| **Rust** | `Cargo.toml` path or git dep | Public Rust items in `src/lib.rs` |
| **`dynamo` binary** (development utility, extraction slice only) | Local Cargo build; controlled toolbox bootstrap for distribution | `dynamo extract --spect ... --out stats.csv` — in `src/bin/dynamo.rs` |

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

## License

BSD 3-Clause.
