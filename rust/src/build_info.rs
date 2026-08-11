//! Build identity of the compiled kernel.
//!
//! `VERSION` follows the DYNAM-O provenance grammar
//! (`documents/OUTPUT_FORMAT.md §8.1`):
//!
//! ```text
//! version-string := <semver> "+" <sha12> [ ".dirty" ] | "unknown"
//! ```
//!
//! The sha comes from `build.rs` at compile time (`DYNAMO_GIT_SHA`,
//! fail-soft `unknown`), so this string identifies the exact source
//! state the numeric kernels were built from — it is the
//! `kernel_version` value every DYNAM-O output artifact records. The
//! semver alone is not enough: 0.2.0 changed fitted-width semantics
//! over a byte-identical ABI, which is exactly the class of change the
//! sha makes visible.

/// The kernel build identity, e.g. `0.2.1+ab12cd34ef56.dirty`.
pub const VERSION: &str = concat!(env!("CARGO_PKG_VERSION"), "+", env!("DYNAMO_GIT_SHA"));

/// Runtime accessor for [`VERSION`] — the `kernel_version` stamp value.
pub fn build_info() -> &'static str {
    VERSION
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pins the provenance grammar: `<semver> "+" <sha12-hex> [".dirty"]`,
    /// with `+unknown` as the fail-soft sha.
    #[test]
    fn version_matches_provenance_grammar() {
        let (semver, rest) = VERSION.split_once('+').expect("must contain one '+'");
        let mut parts = semver.split('.');
        for _ in 0..3 {
            let p = parts.next().expect("semver has three components");
            assert!(!p.is_empty() && p.chars().all(|c| c.is_ascii_digit()),
                "semver component {p:?} not numeric in {VERSION:?}");
        }
        assert!(parts.next().is_none(), "semver has exactly three components");
        let sha = rest.strip_suffix(".dirty").unwrap_or(rest);
        assert!(
            sha == "unknown"
                || (sha.len() == 12 && sha.chars().all(|c| c.is_ascii_hexdigit())),
            "sha part {sha:?} is neither 12-hex nor 'unknown' in {VERSION:?}"
        );
    }
}
