//! Group-comparison statistics, ported from the MATLAB toolbox.
//!
//! MATLAB is canonical for computation in DYNAM-O, so these are ports of
//! `toolbox/helper_functions/statistical_tests` rather than independent
//! implementations, checked against MATLAB-generated fixtures in
//! `tests/stats_matlab_parity.rs`.
//!
//! They live in the kernel crate rather than in a consumer so that every
//! front end runs the *same* code: the desktop app and CLI link it
//! directly, and pydynamo reaches it through the PyO3 bindings. That is
//! the point — a second implementation is a second thing to keep in
//! parity, and the SO-power bug this crate carried for three months is
//! what that costs.
//!
//! Two families, controlling different things:
//!
//!   * [`fdr_grid`] — per-bin rank tests with Benjamini–Hochberg /
//!     Benjamini–Yekutieli control of the false discovery rate
//!     (`FDR_1D.m` / `FDR_2D.m`). More sensitive; answers per bin.
//!   * [`permtest`] — max-statistic permutation testing with
//!     family-wise control (`gpermtest.m` / `gpermtest2.m`). Stricter;
//!     answers with one global bound over the whole map, assuming only
//!     that the group labels are exchangeable.

pub mod fdr;
pub mod fdr_grid;
pub mod normal;
pub mod permtest;
pub mod ranksum;
pub mod signrank;
