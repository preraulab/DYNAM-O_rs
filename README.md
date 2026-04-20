# DYNAM-O_py

Python + Rust port of [DYNAM-O](https://github.com/preraulab/DYNAM-O): TF-peak
extraction (double watershed + merge + trim + Hann refinement), SO-power /
SO-phase histograms, and a MATLAB-style summary figure.

The Rust crate `dynamo_rs` (in `rust/`) accelerates the hot paths:
- `matlab_watershed` — bit-identical to MATLAB IPT `watershed` (Vincent-Soille
  + FIFO priority)
- `merge_segment` — port of `mergeWshedSegment` with the symmetric
  `edgeWeightEqual` rule
- `trim_regions` — port of `trimWshedRegions`

Multitaper spectrogram delegates to the existing
[`multitaper_rs`](https://github.com/preraulab/multitaper_toolbox) crate.

## Accuracy vs MATLAB reference

Validated end-to-end against `runDYNAMO` on the bundled example EEG, using the
real MATLAB pipeline defaults (`detection_opts()`, `baseline_opts()`,
`SOpowerphasehist_opts()`). Per-stage intermediates are compared via
`scripts/bisect/*.py` against ground-truth `.mat` files exported by
`scripts/export_*.m`.

| dataset | pydynamo peaks | MATLAB peaks | diff | **SOpower cos** | **SOphase cos** |
|---|---:|---:|---:|---:|---:|
| segment (~84 min) | 5 785 | 5 738 | **+0.8 %** | **0.9958** | **0.9822** |
| full night (~8.4 h) | 34 911 | 34 788 | **+0.4 %** | **0.9973** | **0.9960** |

Stage-by-stage verification:

| stage | agreement vs MATLAB |
|---|---|
| multitaper spectrogram | max_rel ≈ 5e-5 (FFT-library floor) |
| artifacts | 99.99 % sample agreement |
| baseline (hazen prctile) | max_rel ≈ 1e-7 (bit-identical) |
| spect / baseline division | exact |
| watershed | bit-identical to MATLAB IPT |
| merge (post-merge region counts) | +0.1 % |
| pass-2 mask (interior − 1-px perimeter) | matches `maskSpectrogram` exactly |
| SOpower timeseries | cos = 1.0000 |
| SOphase timeseries (MATLAB-exact SOS filter) | cos ≈ 0.987 (edge transient); mid-90 % bit-identical (0.0006 rad median) |
| SOpower / SOphase histogram binning | bit-identical given same stats table |

## Runtime (full night, 8.4 h recording, macOS x86_64)

| stage | pydynamo | MATLAB | speedup |
|---|---:|---:|---:|
| **total end-to-end** | **101 s** | **146 s** | **1.45×** |
| extract pass-1 | 20 s | 84 s | 4.2× |
| extract pass-2 | 21 s | 32 s | 1.5× |

## Install from scratch

### Prerequisites

- **Python ≥ 3.9**
- **Rust toolchain** (`cargo`, `rustc`) — install via [rustup](https://rustup.rs/):
  ```bash
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  ```
- **maturin** — builds the Rust extensions against CPython:
  ```bash
  pip install maturin
  ```
- **MATLAB is NOT required** to run the pipeline. It is only needed to
  regenerate ground-truth `.mat` intermediates under `data_cache/` (the
  `scripts/export_*.m` files). Accuracy benchmarking against MATLAB uses
  those exported files, which can be produced once and reused.

### Install

```bash
git clone git@github.com:preraulab/DYNAM-O_py.git
cd DYNAM-O_py

# (recommended) fresh virtualenv
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip maturin

# Build + install the Rust merge / trim / watershed extension (dynamo_rs)
maturin develop --release -m rust/pyproject.toml

# Build + install the multitaper spectrogram Rust extension (multitaper_rs)
# from its own repo — it's a sibling dependency, not on PyPI yet:
git clone git@github.com:preraulab/multitaper_toolbox.git
maturin develop --release -m multitaper_toolbox/src/python/rust/pyproject.toml

# Install the Python package
pip install -e .

# (optional) test dependencies
pip install -e '.[test]'
```

Python runtime deps (installed automatically by `pip install -e .`):
numpy ≥ 1.24, scipy ≥ 1.11, scikit-image ≥ 0.22, matplotlib ≥ 3.7, pandas ≥ 2.0,
joblib ≥ 1.3, tqdm ≥ 4.65, colorcet ≥ 3.0.

## Usage

```python
import scipy.io as sio
from pydynamo import run_dynamo

m = sio.loadmat("example_data.mat", simplify_cells=True)
out = run_dynamo(
    m["data"].ravel(),
    float(m["Fs"]),
    m["stage_times"],
    m["stage_vals"],
    # MATLAB runExampleData.m 'segment' overrides (omit for full-night defaults)
    time_range=(8420.0, 13446.0),
    min_time_in_bin=5,
    min_peak_at_freq=10,
)
print(out.stats_table.head())       # per-peak stats
print(out.SOPHs.SOpower_mat.shape)  # (freq_bins, SOpower_bins)
```

## Stage convention

DYNAM-O uses `1=N3, 2=N2, 3=N1, 4=REM, 5=Wake` — reversed from most EDF
stagers. Pass stages in this convention or histograms will be empty or
inverted.

## Side-by-side figures

`scripts/compare_matlab_vs_pydynamo.py` renders MATLAB's output and
pydynamo's output through the identical `summary_plot` code, so any visual
difference is pure data difference (not color scaling or layout).

![segment comparison](data_cache/compare_segment_sidebyside.png)

![night comparison](data_cache/compare_night_sidebyside.png)

## Validation data

All ground-truth comparison uses `runDYNAMO` output from the DYNAM-O repo's
bundled example EEG (`example_data.mat`). Regenerate MATLAB intermediates
once from the DYNAM-O repo root:

```matlab
run('<path-to-DYNAM-O_py>/scripts/export_bisect_intermediates.m')
run('<path-to-DYNAM-O_py>/scripts/export_merge_diagnostics.m')
run('<path-to-DYNAM-O_py>/scripts/export_pass1_diagnostics.m')
```

These populate `data_cache/` with `.mat` / `.csv` files (not version-controlled).

## Tests

```bash
pytest tests/
```

Smoke + unit coverage for spectrogram, artifacts, baseline, SOpower timeseries,
SOpower histogram binning from MATLAB peaks, slope_test, watershed-vs-MATLAB
equivalence, and an end-to-end run.
