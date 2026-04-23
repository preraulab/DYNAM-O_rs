# DYNAM-O_rs

Pure Rust implementation of the hot kernels for [DYNAM-O](https://github.com/preraulab/DYNAM-O) —
TF-peak extraction (double watershed + merge + MATLAB-paint border + trim +
Hann refinement), SO-power / SO-phase histograms, artifact-detection
primitives, and EDF / staging I/O.

This crate is the Rust core shared by:

- **[DYNAM-O](https://github.com/preraulab/DYNAM-O)** — MATLAB toolbox. The `backend='rust'` path calls `dynamo_rs` via MEX wrappers (`DYNAMO_dev/rust_bridge/`).
- **[pyDYNAM-O](https://github.com/preraulab/DYNAM-O_py)** — Python port. Uses `dynamo_rs` via PyO3 bindings.
- **[DYNAM-O_toolbox](https://github.com/preraulab/DYNAM-O_toolbox)** — parent meta-repo that pins all three as git submodules.

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
    extract_pipeline.rs   # full extract (watershed+merge+paint+trim+stats)
    matlab_watershed.rs   # IPT-compatible watershed port (Vincent-Soille)
    merge.rs              # region adjacency graph + iterative merge
    trim.rs               # volume-based region trimming
    mask.rs               # pass-2 spectrogram masking
    refine.rs             # Hann-window peak-frequency refinement
    histogram.rs          # SO-power / SO-phase 2D histogram accumulator
    baseline.rs           # percentile-based baseline subtraction
    signal.rs             # sosfiltfilt, hilbert, unwrap, movmean
    filter_cache.rs       # SOphase SOS cache (.npy) + sci-rs fallback
    adjacency.rs          # region adjacency utilities
    io/                   # edf, staging
    pipeline/             # artifacts, baseline, dpss, spectrogram (WIP)
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
| **Python** (`pydynamo`) | PyO3 extension (`maturin build --features python`) | `matlab_watershed`, `merge_segment`, `trim_regions`, `mask_spectrogram`, `subtract_baseline`, `hann_event_spectra`, `refine_from_spectra`, `tfpeak_histogram`, … — in `src/lib.rs` under `#[pyfunction]` |
| **Rust** | `Cargo.toml` path or git dep | Public Rust items in `src/lib.rs` |

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
