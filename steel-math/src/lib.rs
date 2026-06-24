//! all the math of steel

#![feature(portable_simd)]
/// Math utilities used by vanilla world generation noise.
mod noise_math;
/// SIMD-based utility functions for matrix transpositions and vector manipulations.
mod simd_utils;
pub mod trig;

pub use crate::noise_math::*;
