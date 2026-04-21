//! DPSS (Slepian) taper provider.
//!
//! scipy's dpss computes K largest eigenvectors of a symmetric tridiagonal
//! matrix. Porting the eigensolver to Rust is ~500 LOC of careful numerical
//! code. To keep this port focused, we cache DPSS tapers on disk as .npy
//! files generated once by scipy (see scripts/export_dpss_cache.py).
//!
//! Filename convention:
//!   dpss_N{N}_NW{nw}_K{k}.npy   shape (K, N), float64, row-major
//!
//! If a cache miss occurs, we emit a clear error message with the scipy
//! command to generate it.

use ndarray::{Array2, Axis};
use ndarray_npy::ReadNpyExt;
use std::fs::File;
use std::path::{Path, PathBuf};

/// Default cache dir. Can be overridden via `DYNAMO_DPSS_CACHE_DIR`.
fn default_cache_dir() -> PathBuf {
    if let Ok(v) = std::env::var("DYNAMO_DPSS_CACHE_DIR") {
        return PathBuf::from(v);
    }
    // Repo-relative fallback.
    let repo_root = env!("CARGO_MANIFEST_DIR");
    PathBuf::from(repo_root)
        .parent()
        .unwrap()
        .join("src/pydynamo/data_matlab_filters/dpss_cache")
}

pub fn cache_filename(n: usize, nw: f64, k: usize) -> String {
    // Match scripts/export_dpss_cache.py: NW printed with up to 4 decimals, trailing zeros stripped.
    let nw_str = format!("{:.4}", nw);
    let nw_trim = nw_str.trim_end_matches('0').trim_end_matches('.');
    format!("dpss_N{}_NW{}_K{}.npy", n, nw_trim, k)
}

pub fn load_dpss(n: usize, nw: f64, k: usize) -> Result<Array2<f64>, String> {
    let dir = default_cache_dir();
    let name = cache_filename(n, nw, k);
    let path = dir.join(&name);
    load_dpss_from(&path)
}

pub fn load_dpss_from(path: &Path) -> Result<Array2<f64>, String> {
    let f = File::open(path).map_err(|e| {
        format!(
            "DPSS cache miss: could not open {}: {}. \
             Run scripts/export_dpss_cache.py to generate.",
            path.display(),
            e
        )
    })?;
    let arr = Array2::<f64>::read_npy(f)
        .map_err(|e| format!("failed to read {}: {}", path.display(), e))?;
    Ok(arr)
}

/// Convenience: return tapers shaped (K, winsize) for the DYNAM-O multitaper
/// calls used in the pipeline.
pub fn get_tapers(winsize: usize, time_bandwidth: f64, num_tapers: usize) -> Result<Array2<f64>, String> {
    let arr = load_dpss(winsize, time_bandwidth, num_tapers)?;
    if arr.shape()[0] != num_tapers || arr.shape()[1] != winsize {
        // Some scipy versions output (N, K); handle either layout.
        if arr.shape()[0] == winsize && arr.shape()[1] == num_tapers {
            return Ok(arr.reversed_axes().to_owned());
        }
        return Err(format!(
            "DPSS cache shape mismatch for N={} NW={} K={}: got {:?}",
            winsize, time_bandwidth, num_tapers, arr.shape()
        ));
    }
    // Make sure it's row-major contiguous
    let _ = Axis(0);
    Ok(arr)
}
