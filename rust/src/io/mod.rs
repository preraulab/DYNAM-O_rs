//! I/O ports: EDF reader and staging CSV parser.
//!
//! `edf` is a direct port of `read_EDF_mex.c` — main header, per-signal
//! headers, int16 record data, digital→physical conversion, channel selection
//! with A–B rereferencing, EDF+ annotation skipping.
//!
//! `staging` ports `read_staging.m`'s delimited-text sleep-stage parser.

pub mod edf;
pub mod expr;
pub mod staging;
