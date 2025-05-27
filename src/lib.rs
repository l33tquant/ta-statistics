#![doc = include_str!("../README.md")]
#![cfg_attr(not(test), no_std)]
#![deny(
    unsafe_code,
    unused_imports,
    unused_variables,
    unused_must_use,
    missing_docs,
    clippy::all,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::dbg_macro,
    clippy::todo,
    clippy::unimplemented
)]
#![allow(clippy::just_underscores_and_digits, clippy::len_without_is_empty)]

#[macro_use]
extern crate alloc;

pub(crate) type Kbn<T> = compensated_summation::KahanBabuskaNeumaier<T>;

mod utils;
pub(crate) use utils::{RingBuffer, Window, helper};

mod rolling_moments;
pub use rolling_moments::RollingMoments;

mod single_statistics;
pub use single_statistics::SingleStatistics;

mod paired_statistics;
pub use paired_statistics::PairedStatistics;
