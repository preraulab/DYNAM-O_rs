# MATLAB interface to DYNAM-O_rs

Call the Rust-accelerated pipeline from MATLAB as if it were `runDYNAMO`,
using MATLAB's built-in Python interface under the hood.

## Why `py.*` and not a MEX file?

- **Robust.** MATLAB's `py` interface is stable across MATLAB releases; no
  MEX ABI pinning, no per-MATLAB-version recompilation.
- **Fast enough.** Conversion overhead is ~10 ms on an 8-hour recording.
  Rust still delivers a ~2.4× speedup over `runDYNAMO`.
- **Zero new Rust glue.** The existing `dynamo_rs` PyO3 extensions handle
  the hot paths directly; MATLAB just calls `py.pydynamo.run_dynamo`.

A direct Rust MEX via [`rustmex`](https://docs.rs/rustmex) is a valid path
for bare-metal performance, but adds ~500 lines of marshalling glue and
per-platform build infrastructure for a 1–2 % speed gain. Not worth it for
this pipeline.

## One-time setup

1. Install the package in a Python venv:
   ```bash
   cd /path/to/DYNAM-O_rs
   python -m venv .venv && source .venv/bin/activate
   pip install --upgrade pip maturin
   maturin develop --release -m rust/Cargo.toml
   pip install -e .
   ```
2. Point MATLAB's `pyenv` at that venv (once per MATLAB session; MATLAB
   remembers between sessions if you save via `savepath` + `pyenv`):
   ```matlab
   addpath('/path/to/DYNAM-O_rs/matlab')
   setup_dynamo_py('/path/to/DYNAM-O_rs/.venv/bin/python')
   ```

## Usage

```matlab
load('example_data.mat')    % data, Fs, stage_times, stage_vals

[stats_table, spect, stimes, sfreqs, data_time_range, t_time_range, ...
 artifacts, SOPHs, timings] = ...
    runDYNAMO_rs(data, Fs, stage_times, stage_vals, [8420 13446]);

% Any name-value kwargs of pydynamo.run_dynamo are forwarded through:
[stats_table, ~, ~, ~, ~, ~, ~, SOPHs] = runDYNAMO_rs( ...
    data, Fs, stage_times, stage_vals, [8420 13446], ...
    'merge_thresh', 11, 'trim_vol', 0.8, ...
    'min_time_in_bin', 5, 'min_peak_at_freq', 10);
```

Output shapes and field names match `runDYNAMO` exactly except:
- `stats_table` is converted from a pandas DataFrame to a MATLAB `table`
  (columns preserved; `BoundingBox` becomes an `Nx4` numeric matrix).
- `SOPHs` is a struct with the same fields as MATLAB `SOPHs` but in
  pydynamo's naming (`SOpower_mat`, `freq_bins`, etc.).
- `timings` includes both pydynamo's per-stage breakdown and a
  `matlab_wall` field with the wall-clock time observed from MATLAB.

## Drop-in swap for existing scripts

### Whole-pipeline replacement (~2.4× speedup, easiest)

```matlab
% old:
% [st, sp, t, f, dtr, ttr, art, SOPHs, tim] = runDYNAMO(data, Fs, stage_times, stage_vals, [t0 t1]);

% new (same signature):
[st, sp, t, f, dtr, ttr, art, SOPHs, tim] = runDYNAMO_rs(data, Fs, stage_times, stage_vals, [t0 t1]);
```

### Just the extraction step (~3× speedup, keep the rest in MATLAB)

If you want to keep MATLAB's spectrogram / baseline / SOPH / plot code but
swap out the slow runSegmentedData calls (which dominate runDYNAMO's
runtime), use these per-stage shims:

```matlab
% Original:
%   [stats1, regions1, borders1] = runSegmentedData(spect1, stimes1, sfreqs, baseline1, ...);
%   spect2_masked = maskSpectrogram(spect2, stimes1, stimes2, regions1, borders1);
%   [stats2, ~, ~] = runSegmentedData(spect2_masked, stimes2, sfreqs, ones(size(baseline2)), ...);
%   stats_final = refinePeakFrequency(data, Fs, stats2, ...);
%
% Rust-accelerated equivalents:
[stats1, labels1] = extract_tfpeaks_rs(spect1, stimes1, sfreqs, ...
    'baseline', baseline1, 'merge_thresh', 11, 'trim_vol', 0.8, ...
    'dur_min', 0.5, 'bw_min', 2);

spect2_masked = mask_spectrogram_rs(spect2 ./ baseline2, stimes2, labels1, stimes1);

[stats2, ~] = extract_tfpeaks_rs(spect2_masked, stimes2, sfreqs, ...
    'merge_thresh', 11, 'trim_vol', 0.8, 'dur_min', 0.5, 'bw_min', 1);

stats_final = refine_peaks_rs(stats2, data, Fs);
```

This path keeps your MATLAB plumbing intact and avoids the wholesale
swap to a different output struct format.

## Troubleshooting

| symptom | fix |
|---|---|
| `ImportError: No module named pydynamo` | Run `pip install -e .` inside your venv, then `pyenv('Version', ...)` again. |
| `ImportError: No module named dynamo_rs` | `maturin develop --release -m rust/Cargo.toml` in the same venv. |
| Slow first call | MATLAB caches the Python interpreter after the first call; subsequent calls avoid startup cost. |
| Different numerical results than MATLAB runDYNAMO | Expected: SOpower/SOphase cos ≥ 0.996 vs MATLAB. See `scripts/compare_table.py` in the repo root. |
