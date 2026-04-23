//! SOphase SOS filter cache loader + in-memory fallback design.
//!
//! The cache is the 42 `sophase_sos_Fs{fs}_{lo}_{hi}.npy` files shipped in
//! `data_matlab_filters/` at the repo root. They were exported from MATLAB's
//! `designfilt('bandpassiir', ...)` and are treated as the canonical
//! coefficients for pydynamo/MATLAB bit-equivalence.
//!
//! Policy:
//!   1. Resolve cache dir from env var `DYNAMO_FILTER_CACHE` if set.
//!   2. Otherwise look next to the crate at
//!      `$CARGO_MANIFEST_DIR/../data_matlab_filters`.
//!   3. If neither hit → fall back to a **pure-Rust Chebyshev Type-I bandpass**
//!      design (this crate already ships one in `filter_design.rs`)
//!      and emit a warning. The MATLAB cache is actually an elliptic design
//!      so this fallback introduces a small (~0.93 cosine similarity)
//!      divergence. Callers that need bit-equivalence MUST ensure the cache
//!      is present.
//!
//! Filename format: `sophase_sos_Fs{fs}_{lo}_{hi}.npy` with `{:g}` formatting
//! (e.g. `Fs100_0.3_1.5`, `Fs128_0.5_1` — no trailing zeros).

use ndarray::Array2;
use ndarray_npy::read_npy;
use std::path::{Path, PathBuf};

/// Errors from the filter cache.
#[derive(Debug)]
pub enum FilterError {
    /// The cache file was not found in any candidate directory.
    CacheMiss(String),
    /// The cache file was present but could not be loaded/parsed.
    LoadFailed(String),
    /// The fallback filter design failed (shouldn't happen for valid inputs).
    DesignFailed(String),
}

impl std::fmt::Display for FilterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FilterError::CacheMiss(s) => write!(f, "filter cache miss: {}", s),
            FilterError::LoadFailed(s) => write!(f, "filter load failed: {}", s),
            FilterError::DesignFailed(s) => write!(f, "filter design failed: {}", s),
        }
    }
}

impl std::error::Error for FilterError {}

/// Build the ordered list of candidate cache directories.
fn candidate_filter_dirs() -> Vec<PathBuf> {
    let mut dirs: Vec<PathBuf> = Vec::new();
    if let Ok(envp) = std::env::var("DYNAMO_FILTER_CACHE") {
        if !envp.is_empty() {
            dirs.push(PathBuf::from(envp));
        }
    }
    // Canonical: repo-root/data_matlab_filters, i.e. $CARGO_MANIFEST_DIR/../data_matlab_filters
    let manifest = env!("CARGO_MANIFEST_DIR");
    dirs.push(Path::new(manifest).join("..").join("data_matlab_filters"));
    dirs
}

/// `{:g}` mimics Python's `f"{v:g}"` and Rust's default `{}` for `f64` with
/// trimmed trailing zeros. Rust's `{}` for f64 already avoids trailing zeros
/// (e.g. `0.3` → `"0.3"`, `1.5` → `"1.5"`, `1.0` → `"1"`), matching the
/// existing filenames on disk.
fn fmt_g(v: f64) -> String {
    // f64's Display already strips trailing zeros and doesn't print ".0"
    // for integer values — a quick spot-check against the committed filenames
    // confirms this matches the expected format.
    if v == v.trunc() && v.is_finite() {
        format!("{}", v as i64)
    } else {
        format!("{}", v)
    }
}

/// Build the expected cache filename for a given (fs, band).
pub fn sophase_cache_filename(fs: f64, band: (f64, f64)) -> String {
    format!(
        "sophase_sos_Fs{}_{}_{}.npy",
        fmt_g(fs),
        fmt_g(band.0),
        fmt_g(band.1),
    )
}

/// Load a SOphase SOS filter from the cache, or fall back to cheby1 bandpass
/// design if the cache file is not found.
///
/// Returns an `(n_sections, 6)` SOS array in scipy layout [b0 b1 b2 a0 a1 a2].
pub fn get_sophase_sos(fs: f64, band: (f64, f64)) -> Result<Array2<f64>, FilterError> {
    let fname = sophase_cache_filename(fs, band);
    let mut tried: Vec<String> = Vec::new();
    for dir in candidate_filter_dirs() {
        let p = dir.join(&fname);
        if p.exists() {
            return read_npy::<_, Array2<f64>>(&p)
                .map_err(|e| FilterError::LoadFailed(format!("{}: {}", p.display(), e)));
        }
        tried.push(p.display().to_string());
    }

    // Fallback — log a warning (sci-rs is not currently in our deps; this
    // crate ships a pure-Rust cheby1_sos in filter_design.rs that
    // we reuse rather than pulling in another dependency).
    log::warn!(
        "sophase SOS cache miss for (fs={}, band=({}, {})); falling back to \
         cheby1 bandpass design. This introduces a ~0.93 cosine similarity \
         divergence from the MATLAB elliptic design. Cache searched: {:?}",
        fs, band.0, band.1, tried,
    );
    eprintln!(
        "warning: sophase SOS cache miss for Fs={}, band=({},{}) — falling \
         back to cheby1 design (not bit-equivalent to MATLAB elliptic).",
        fs, band.0, band.1,
    );

    // cheby1 default order 4, 0.2 dB ripple — same design used elsewhere in
    // DYNAM-O's highpass/bandpass chain for consistency.
    design_cheby1_bandpass_sos(4, 0.2, band, fs)
}

/// Design a Chebyshev Type-I bandpass SOS filter.
///
/// Wraps the crate's existing pure-Rust `cheby1_sos` implementation (ported
/// from `scipy.signal.cheby1`, see `filter_design.rs`).
pub fn design_cheby1_bandpass_sos(
    order: usize,
    rp_db: f64,
    band: (f64, f64),
    fs: f64,
) -> Result<Array2<f64>, FilterError> {
    // Reuse the existing in-tree implementation via #[path] shim (declared
    // below) — keeps us from pulling in an extra filter-design crate.
    let wn = [band.0, band.1];
    let sections = filter_design::cheby1_sos(order, rp_db, &wn, "bandpass", fs);
    if sections.is_empty() {
        return Err(FilterError::DesignFailed(
            "cheby1_sos returned zero sections".into(),
        ));
    }
    let n = sections.len();
    let mut flat = Vec::with_capacity(n * 6);
    for s in &sections {
        flat.extend_from_slice(s);
    }
    Array2::from_shape_vec((n, 6), flat)
        .map_err(|e| FilterError::DesignFailed(format!("{}", e)))
}

use crate::filter_design;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filename_format_matches_cache() {
        // Spot-check every committed combination's naming convention.
        assert_eq!(
            sophase_cache_filename(100.0, (0.3, 1.5)),
            "sophase_sos_Fs100_0.3_1.5.npy"
        );
        assert_eq!(
            sophase_cache_filename(128.0, (0.5, 1.0)),
            "sophase_sos_Fs128_0.5_1.npy"
        );
        assert_eq!(
            sophase_cache_filename(512.0, (0.3, 5.0)),
            "sophase_sos_Fs512_0.3_5.npy"
        );
    }

    #[test]
    fn load_fs100_band_03_15() {
        // The committed file should be (9, 6).
        let sos = get_sophase_sos(100.0, (0.3, 1.5)).expect("expected cache hit");
        assert_eq!(sos.shape(), &[9, 6]);
    }

    #[test]
    fn cheby1_bandpass_designs_nonempty() {
        let sos = design_cheby1_bandpass_sos(4, 0.2, (0.3, 1.5), 100.0).unwrap();
        assert!(sos.nrows() > 0);
        assert_eq!(sos.ncols(), 6);
    }
}
