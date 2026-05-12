//! Conditional parallelism wrapper.
//!
//! With the `parallel` feature (on by default) this module re-exports
//! `rayon::prelude::*` so call sites get true multi-threaded iterators.
//! With `--no-default-features` (the WASM build mode, since WASM has no
//! native threads) we provide sequential drop-in shims with the same
//! method names — `into_par_iter`, `par_iter`, `map_init` — so the
//! consumer code in `extract_pipeline.rs`, `refine.rs`, `histogram.rs`
//! and `baseline.rs` stays unchanged.

#[cfg(feature = "parallel")]
pub use rayon::prelude::*;

#[cfg(not(feature = "parallel"))]
pub use seq::*;

#[cfg(not(feature = "parallel"))]
mod seq {
    use std::ops::Range;

    /// Sequential drop-in for `rayon::iter::IntoParallelIterator`.
    pub trait IntoParallelIterator {
        type Iter: Iterator;
        fn into_par_iter(self) -> Self::Iter;
    }
    impl<T> IntoParallelIterator for Vec<T> {
        type Iter = std::vec::IntoIter<T>;
        fn into_par_iter(self) -> Self::Iter {
            self.into_iter()
        }
    }
    impl IntoParallelIterator for Range<usize> {
        type Iter = Range<usize>;
        fn into_par_iter(self) -> Self::Iter {
            self
        }
    }

    /// Sequential drop-in for `rayon::iter::IntoParallelRefIterator`.
    pub trait IntoParallelRefIterator<'a> {
        type Iter: Iterator + 'a;
        fn par_iter(&'a self) -> Self::Iter;
    }
    impl<'a, T: 'a> IntoParallelRefIterator<'a> for [T] {
        type Iter = std::slice::Iter<'a, T>;
        fn par_iter(&'a self) -> Self::Iter {
            self.iter()
        }
    }
    impl<'a, T: 'a> IntoParallelRefIterator<'a> for Vec<T> {
        type Iter = std::slice::Iter<'a, T>;
        fn par_iter(&'a self) -> Self::Iter {
            (**self).iter()
        }
    }

    /// Adds `map_init` to any iterator. State is created once and
    /// threaded through every call to `f` — semantically equivalent to
    /// rayon's `map_init` running on a single worker.
    pub trait ParallelIteratorShim: Iterator + Sized {
        fn map_init<S, INIT, F, R>(self, init: INIT, f: F) -> MapInit<Self, S, F>
        where
            INIT: FnOnce() -> S,
            F: FnMut(&mut S, Self::Item) -> R,
        {
            MapInit {
                iter: self,
                state: init(),
                f,
            }
        }
    }
    impl<I: Iterator> ParallelIteratorShim for I {}

    pub struct MapInit<I, S, F> {
        iter: I,
        state: S,
        f: F,
    }
    impl<I, S, F, R> Iterator for MapInit<I, S, F>
    where
        I: Iterator,
        F: FnMut(&mut S, I::Item) -> R,
    {
        type Item = R;
        fn next(&mut self) -> Option<R> {
            self.iter.next().map(|x| (self.f)(&mut self.state, x))
        }
    }
}
