//! I/O ports: EDF reader, channel-expression parser, and staging CSV parser.
//!
//! `edf` is a direct port of `read_EDF_mex.c`: main header, per-signal
//! headers, int16 record data, digital→physical conversion, channel selection
//! with A–B rereferencing, EDF+ annotation skipping.
//!
//! `staging` ports `read_staging.m`'s delimited-text sleep-stage parser.
//!
//! `expr` parses the channel-derivation expressions (`"C3-A1"`,
//! `"mean(A1, A2)"`, linear combinations) that `edf` evaluates when
//! selecting a channel.

pub mod edf;
pub mod expr;
pub mod staging;
