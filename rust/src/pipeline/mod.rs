//! Pure-Rust DYNAM-O pipeline orchestrators.
//!
//! These are Rust ports of `src/pydynamo/` orchestration glue. They call
//! the existing numeric kernels in the parent crate (merge, trim, baseline,
//! mask, refine, signal, histogram, matlab_watershed) as well as the
//! `multitaper_rs` path-dependency for spectrograms.

pub mod artifacts;
pub mod baseline;
pub mod dpss;
pub mod extract;
pub mod filter_design;
pub mod run;
pub mod soph_histogram;
pub mod sophase;
pub mod sopower;
pub mod spectrogram;
