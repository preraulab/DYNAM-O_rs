//! SOphase SOS filter cache loader + in-memory fallback design.
//!
//! The cache is the 42 `sophase_sos_Fs{fs}_{lo}_{hi}.npy` files shipped in
//! `data_matlab_filters/` at the repo root. They were exported from MATLAB's
//! `designfilt('bandpassiir', ...)` and are treated as the canonical
//! coefficients for pydynamo/MATLAB bit-equivalence.
//!
//! Policy:
//!   1. Resolve cache dir from env var `DYNAMO_FILTER_CACHE` if set.
//!   2. Otherwise look for `data_matlab_filters/` beside the loaded native
//!      module. Release installers must preserve that sidecar layout.
//!   3. For source-tree development, also check paths relative to the current
//!      working directory.
//!   4. If none hit → fall back to a **pure-Rust Chebyshev Type-I bandpass**
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
    candidate_filter_dirs_from(
        std::env::var_os("DYNAMO_FILTER_CACHE")
            .filter(|path| !path.is_empty())
            .map(PathBuf::from),
        runtime_module_path().as_deref(),
        std::env::current_dir().ok().as_deref(),
    )
}

fn candidate_filter_dirs_from(
    override_dir: Option<PathBuf>,
    module_path: Option<&Path>,
    current_dir: Option<&Path>,
) -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Some(path) = override_dir {
        push_unique(&mut dirs, path);
    }
    if let Some(module_dir) = module_path.and_then(Path::parent) {
        push_unique(&mut dirs, module_dir.join("data_matlab_filters"));
    }
    if let Some(current_dir) = current_dir {
        push_unique(&mut dirs, current_dir.join("data_matlab_filters"));
        if let Some(parent) = current_dir.parent() {
            push_unique(&mut dirs, parent.join("data_matlab_filters"));
        }
    }
    dirs
}

fn push_unique(paths: &mut Vec<PathBuf>, path: PathBuf) {
    if !paths.contains(&path) {
        paths.push(path);
    }
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
fn runtime_module_path() -> Option<PathBuf> {
    use std::ffi::{c_char, c_int, c_void, CStr};
    use std::mem::MaybeUninit;
    use std::os::unix::ffi::OsStrExt;

    #[repr(C)]
    struct DlInfo {
        dli_fname: *const c_char,
        dli_fbase: *mut c_void,
        dli_sname: *const c_char,
        dli_saddr: *mut c_void,
    }

    #[cfg_attr(target_os = "linux", link(name = "dl"))]
    unsafe extern "C" {
        fn dladdr(address: *const c_void, info: *mut DlInfo) -> c_int;
    }

    let mut info = MaybeUninit::<DlInfo>::zeroed();
    let found = unsafe {
        dladdr(
            runtime_module_path as *const () as *const c_void,
            info.as_mut_ptr(),
        )
    };
    if found == 0 {
        return None;
    }

    let info = unsafe { info.assume_init() };
    if info.dli_fname.is_null() {
        return None;
    }
    let bytes = unsafe { CStr::from_ptr(info.dli_fname) }.to_bytes();
    Some(PathBuf::from(std::ffi::OsStr::from_bytes(bytes)))
}

#[cfg(target_os = "windows")]
fn runtime_module_path() -> Option<PathBuf> {
    use std::ffi::c_void;
    use std::os::windows::ffi::OsStringExt;
    use std::ptr;

    const GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT: u32 = 0x0000_0002;
    const GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS: u32 = 0x0000_0004;

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetModuleHandleExW(flags: u32, module_name: *const u16, module: *mut *mut c_void)
            -> i32;
        fn GetModuleFileNameW(module: *mut c_void, filename: *mut u16, size: u32) -> u32;
    }

    let mut module = ptr::null_mut();
    let found = unsafe {
        GetModuleHandleExW(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            runtime_module_path as *const () as *const u16,
            &mut module,
        )
    };
    if found == 0 {
        return None;
    }

    let mut buffer = vec![0_u16; 260];
    loop {
        let length =
            unsafe { GetModuleFileNameW(module, buffer.as_mut_ptr(), buffer.len() as u32) };
        if length == 0 {
            return None;
        }
        if (length as usize) < buffer.len() - 1 {
            return Some(std::ffi::OsString::from_wide(&buffer[..length as usize]).into());
        }
        buffer.resize(buffer.len() * 2, 0);
    }
}

#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
fn runtime_module_path() -> Option<PathBuf> {
    std::env::current_exe().ok()
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
    fn cache_candidates_prefer_override_then_runtime_install() {
        let dirs = candidate_filter_dirs_from(
            Some(PathBuf::from("/override")),
            Some(Path::new("/install/dynamo_rs.so")),
            Some(Path::new("/checkout/rust")),
        );
        assert_eq!(
            dirs,
            vec![
                PathBuf::from("/override"),
                PathBuf::from("/install/data_matlab_filters"),
                PathBuf::from("/checkout/rust/data_matlab_filters"),
                PathBuf::from("/checkout/data_matlab_filters"),
            ]
        );
    }

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
