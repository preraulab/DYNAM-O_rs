# DYNAM-O_rs

Pure Rust implementation of the hot kernels for [DYNAM-O](https://github.com/preraulab/DYNAM-O): TF-peak extraction (double watershed + merge + trim + Hann refinement), SO-power / SO-phase histograms, artifact-detection primitives, and EDF / staging I/O.

This repo is the Rust-only sibling of:
- [**DYNAM-O**](https://github.com/preraulab/DYNAM-O) — MATLAB toolbox, source-of-truth algorithm, GUI (DYNAMOFileManager)
- [**DYNAM-O_py**](https://github.com/preraulab/DYNAM-O_py) — Python port (optionally uses this crate for speed)

## Crate targets

`dynamo_rs` is a triple-target Cargo crate:

- `rlib` / `cdylib` / `staticlib` — consumed from Rust, from C (via the C ABI in `src/c_api.rs`), or from MATLAB via MEX wrappers.
- `python` feature — builds a PyO3 extension module (used by `DYNAM-O_py` when the optional speed path is active).

```toml
[lib]
crate-type = ["cdylib", "staticlib", "rlib"]

[features]
default = []
python  = ["dep:pyo3", "dep:numpy"]
```

## Layout

```
rust/
  Cargo.toml
  src/
    lib.rs                # public Rust API
    c_api.rs              # extern "C" surface for MEX wrappers + cbindgen
    filter_cache.rs       # SOphase SOS cache (.npy) + sci-rs fallback
    kernels/              # watershed, merge, trim, baseline, mask, refine, histogram
    io/                   # edf, staging
    pipeline/             # artifacts, baseline, dpss, filter_design, spectrogram (WIP)
    signal.rs             # sosfiltfilt, hilbert, unwrap, movmean
  include/
    dynamo_rs.h           # cbindgen-generated C header
data_matlab_filters/      # 43 pre-computed SOphase SOS filters (.npy)
```

## Consumers

**MATLAB** (ship path): `DYNAM-O_dev/rust_bridge/*.cpp` MEX wrappers link against `libdynamo_rs.{dylib,dll,so}` + `include/dynamo_rs.h`. Bundled into standalone `.app` via `mcc`.

**Python** (optional speed): `DYNAM-O_py` installs this crate as a pip package (`maturin build --release --features python`). If present, its hot paths delegate to Rust; otherwise falls back to scipy/numpy.

**Rust directly**: add as a path or git dep in `Cargo.toml`, use the public Rust API in `lib.rs`.

## Build

```bash
# lib only (rlib + cdylib + staticlib), no Python
cargo build --release

# with Python bindings, via maturin (needs a venv with python ≥3.8)
maturin build --release --features python
```

## Accuracy vs MATLAB reference

Validated end-to-end against `runDYNAMO` on the bundled example EEG, matching MATLAB pipeline defaults.

| Dataset | Rust peaks | MATLAB peaks | Δ | **SOpower cos** | **SOphase cos** |
|---|---:|---:|---:|---:|---:|
| segment (~84 min) | 5 811 | 5 738 | +1.3% | **0.997** | **0.982** |
| full night (~8.4 h) | 34 926 | 34 788 | +0.4% | **0.999** | **0.996** |

## Status

This repo's `rust-bridge` branch is the active development line for the three-repo restructure. See the `main` branch for the prior hybrid layout (Python + Rust + MATLAB shims in one tree).
