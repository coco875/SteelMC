//! World generation noise, density functions, and surface rule runtime support.

#![feature(portable_simd)]

#[cfg(not(feature = "f32_gen"))]
pub type FloatGen = f64;

#[cfg(feature = "f32_gen")]
pub type FloatGen = f32;

#[cfg(feature = "f32_gen")]
use std::simd::f32x4;
#[cfg(not(feature = "f32_gen"))]
use std::simd::f64x4;

#[cfg(not(feature = "f32_gen"))]
pub type Vec4Gen = f64x4;
#[cfg(feature = "f32_gen")]
pub type Vec4Gen = f32x4;

#[cfg(not(feature = "f32_gen"))]
use std::f64::consts::PI;

#[cfg(feature = "f32_gen")]
use std::f32::consts::PI;

pub const PI_GEN: FloatGen = PI;

extern crate self as steel_worldgen;

pub use steel_utils::{BlockStateId, random};

/// Density function system for world generation.
pub mod density;
/// Math utilities used by vanilla world generation noise.
pub mod math {
    pub use crate::noise_math::{
        bias_towards_extreme, clamp, clamp_i32, clamped_lerp, cube, floor, inverse_lerp, lerp,
        lerp2, lerp3, lfloor, map, map_clamped, smoothstep, smoothstep_derivative, square,
    };
}
/// Noise generation utilities for world generation.
pub mod noise;
mod noise_math;
/// Surface rule context types for generated code.
pub mod surface;

#[expect(warnings)]
#[rustfmt::skip]
#[path = "generated/vanilla_multi_noise.rs"]
pub mod multi_noise;

#[expect(warnings)]
#[rustfmt::skip]
#[path = "generated/vanilla_noise_parameters.rs"]
pub mod noise_parameters;

#[expect(warnings)]
#[rustfmt::skip]
#[path = "generated/vanilla_density_functions/mod.rs"]
pub mod density_functions;
