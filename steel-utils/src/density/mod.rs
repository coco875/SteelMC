//! Density function system for world generation.
//!
//! This module provides the density function evaluation system used for terrain generation
//! in vanilla Minecraft. Density functions form a tree structure that is evaluated at
//! each position to determine terrain shape, biome climate values, and other world features.
//!
//! # Key Types
//!
//! - [`DensityFunction`] - The density function enum with all operation types
//! - [`DensityEvaluator`] - Evaluates density functions with noise generators
//! - [`DensityContext`] - The position context for evaluation
//! - [`NoiseRouter`] - Collection of all density functions for world generation
//! - [`CubicSpline`] - Cubic spline interpolation for smooth terrain transitions

mod evaluator;
mod types;

pub use evaluator::{DensityEvaluator, EvalCache, NoiseParameters};
pub use types::{
    CubicSpline, DensityContext, DensityFunction, NoiseRouter, RarityValueMapper, SplinePoint,
    SplineValue,
};
