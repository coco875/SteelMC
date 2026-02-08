//! Density function evaluator matching vanilla Minecraft's density function evaluation.
//!
//! This evaluates the density function tree at given positions to produce terrain values.

use std::sync::Arc;

use rustc_hash::FxHashMap;

use crate::density::types::{
    CubicSpline, DensityContext, DensityFunction, SplinePoint, SplineValue,
};
use crate::math::{clamp, map_clamped};
use crate::noise::NormalNoise;
use crate::random::RandomSplitter;

/// Cache for density function evaluation, matching vanilla's per-node caching.
///
/// Create one per chunk or per evaluation batch and pass to [`DensityEvaluator::evaluate_cached`].
/// Cache keys are derived from stable `Arc` pointer addresses in the density function tree.
pub struct EvalCache {
    /// `CacheOnce`: caches last (x, y, z) → value per node.
    once: FxHashMap<usize, (i32, i32, i32, f64)>,
    /// `FlatCache`/`Cache2D`: caches (x, z) → value per node, keyed by (`node_key`, x, z).
    flat: FxHashMap<(usize, i32, i32), f64>,
}

impl EvalCache {
    /// Create a new empty evaluation cache.
    #[must_use]
    pub fn new() -> Self {
        Self {
            once: FxHashMap::default(),
            flat: FxHashMap::default(),
        }
    }
}

impl Default for EvalCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Evaluator for density functions.
///
/// Holds the instantiated noise generators and evaluates density function trees.
pub struct DensityEvaluator {
    /// Noise generators keyed by noise ID
    noises: FxHashMap<String, NormalNoise>,
    /// Density function registry for resolving references
    registry: FxHashMap<String, Arc<DensityFunction>>,
}

impl DensityEvaluator {
    /// Create a new evaluator with noises generated from the given random splitter.
    ///
    /// The `noise_params` should contain parameters for each noise type needed.
    #[must_use]
    pub fn new(
        splitter: &RandomSplitter,
        noise_params: &FxHashMap<String, NoiseParameters>,
    ) -> Self {
        let mut noises = FxHashMap::default();

        for (id, params) in noise_params {
            let noise = NormalNoise::create(splitter, id, params.first_octave, &params.amplitudes);
            noises.insert(id.clone(), noise);
        }

        Self {
            noises,
            registry: FxHashMap::default(),
        }
    }

    /// Create an evaluator with pre-built noises.
    #[must_use]
    pub fn with_noises(noises: FxHashMap<String, NormalNoise>) -> Self {
        Self {
            noises,
            registry: FxHashMap::default(),
        }
    }

    /// Create an evaluator with noises and a density function registry.
    #[must_use]
    pub const fn with_noises_and_registry(
        noises: FxHashMap<String, NormalNoise>,
        registry: FxHashMap<String, Arc<DensityFunction>>,
    ) -> Self {
        Self { noises, registry }
    }

    /// Add a density function to the registry.
    pub fn add_to_registry(&mut self, id: String, func: Arc<DensityFunction>) {
        self.registry.insert(id, func);
    }

    /// Evaluate a density function at the given context.
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn evaluate(&self, func: &DensityFunction, ctx: &DensityContext) -> f64 {
        match func {
            DensityFunction::Constant(v) => *v,

            DensityFunction::Reference(id) => {
                if let Some(referenced) = self.registry.get(id) {
                    self.evaluate(referenced, ctx)
                } else {
                    // Reference not found, return 0.0 as fallback
                    0.0
                }
            }

            DensityFunction::YClampedGradient {
                from_y,
                to_y,
                from_value,
                to_value,
            } => map_clamped(
                f64::from(ctx.y),
                f64::from(*from_y),
                f64::from(*to_y),
                *from_value,
                *to_value,
            ),

            DensityFunction::Noise {
                noise_id,
                xz_scale,
                y_scale,
            } => {
                if let Some(noise) = self.noises.get(noise_id) {
                    noise.get_value(
                        f64::from(ctx.x) * xz_scale,
                        f64::from(ctx.y) * y_scale,
                        f64::from(ctx.z) * xz_scale,
                    )
                } else {
                    0.0
                }
            }

            DensityFunction::ShiftedNoise {
                shift_x,
                shift_z,
                xz_scale,
                y_scale,
                noise_id,
            } => {
                // Vanilla: x = blockX * xzScale + shiftX.compute(ctx)
                // Scale first, then add shift (shift is NOT scaled by xz_scale)
                let dx = self.evaluate(shift_x, ctx);
                let dz = self.evaluate(shift_z, ctx);
                if let Some(noise) = self.noises.get(noise_id) {
                    noise.get_value(
                        f64::from(ctx.x) * xz_scale + dx,
                        f64::from(ctx.y) * y_scale,
                        f64::from(ctx.z) * xz_scale + dz,
                    )
                } else {
                    0.0
                }
            }

            DensityFunction::ShiftA { noise_id } => {
                if let Some(noise) = self.noises.get(noise_id) {
                    // ShiftA: compute(blockX, 0.0, blockZ) -> noise(x*0.25, 0, z*0.25) * 4
                    noise.get_value(f64::from(ctx.x) * 0.25, 0.0, f64::from(ctx.z) * 0.25) * 4.0
                } else {
                    0.0
                }
            }

            DensityFunction::ShiftB { noise_id } => {
                if let Some(noise) = self.noises.get(noise_id) {
                    // ShiftB: compute(blockZ, blockX, 0.0) -> noise(z*0.25, x*0.25, 0) * 4
                    noise.get_value(f64::from(ctx.z) * 0.25, f64::from(ctx.x) * 0.25, 0.0) * 4.0
                } else {
                    0.0
                }
            }

            DensityFunction::Shift { noise_id } => {
                if let Some(noise) = self.noises.get(noise_id) {
                    // Shift: compute(blockX, blockY, blockZ) -> noise(x*0.25, y*0.25, z*0.25) * 4
                    noise.get_value(
                        f64::from(ctx.x) * 0.25,
                        f64::from(ctx.y) * 0.25,
                        f64::from(ctx.z) * 0.25,
                    ) * 4.0
                } else {
                    0.0
                }
            }

            DensityFunction::Add(a, b) => self.evaluate(a, ctx) + self.evaluate(b, ctx),

            DensityFunction::Mul(a, b) => self.evaluate(a, ctx) * self.evaluate(b, ctx),

            DensityFunction::Min(a, b) => self.evaluate(a, ctx).min(self.evaluate(b, ctx)),

            DensityFunction::Max(a, b) => self.evaluate(a, ctx).max(self.evaluate(b, ctx)),

            DensityFunction::Abs(f) => self.evaluate(f, ctx).abs(),

            DensityFunction::Square(f) => {
                let v = self.evaluate(f, ctx);
                v * v
            }

            DensityFunction::Cube(f) => {
                let v = self.evaluate(f, ctx);
                v * v * v
            }

            DensityFunction::HalfNegative(f) => {
                let v = self.evaluate(f, ctx);
                if v > 0.0 { v } else { v * 0.5 }
            }

            DensityFunction::QuarterNegative(f) => {
                let v = self.evaluate(f, ctx);
                if v > 0.0 { v } else { v * 0.25 }
            }

            DensityFunction::Squeeze(f) => {
                let v = self.evaluate(f, ctx);
                let c = clamp(v, -1.0, 1.0);
                c / 2.0 - c * c * c / 24.0
            }

            DensityFunction::Clamp { input, min, max } => {
                clamp(self.evaluate(input, ctx), *min, *max)
            }

            DensityFunction::RangeChoice {
                input,
                min_inclusive,
                max_exclusive,
                when_in_range,
                when_out_of_range,
            } => {
                let v = self.evaluate(input, ctx);
                if v >= *min_inclusive && v < *max_exclusive {
                    self.evaluate(when_in_range, ctx)
                } else {
                    self.evaluate(when_out_of_range, ctx)
                }
            }

            DensityFunction::Spline(spline) => self.evaluate_spline(spline, ctx),

            // TODO: Implement vanilla BlendedNoise (requires 3 PerlinNoise generators:
            // minLimitNoise, maxLimitNoise, mainNoise with factor-based blending).
            // Current approximation samples from "offset" noise with scaled coordinates.
            DensityFunction::BlendedNoise {
                xz_scale,
                y_scale,
                xz_factor,
                y_factor,
                smear_scale_multiplier,
            } => {
                if let Some(noise) = self.noises.get("offset") {
                    let scaled_xz = xz_scale / xz_factor;
                    let scaled_y = y_scale / y_factor;
                    noise.get_value(
                        f64::from(ctx.x) * scaled_xz,
                        f64::from(ctx.y) * scaled_y,
                        f64::from(ctx.z) * scaled_xz,
                    ) * smear_scale_multiplier
                } else {
                    0.0
                }
            }

            DensityFunction::WeirdScaledSampler {
                input,
                noise_id,
                rarity_value_mapper,
            } => {
                let rarity = self.evaluate(input, ctx);
                let scale = rarity_value_mapper.get_values(rarity);
                if let Some(noise) = self.noises.get(noise_id) {
                    // Vanilla: rarity * Math.abs(noise.getValue(x / rarity, y / rarity, z / rarity))
                    scale
                        * noise
                            .get_value(
                                f64::from(ctx.x) / scale,
                                f64::from(ctx.y) / scale,
                                f64::from(ctx.z) / scale,
                            )
                            .abs()
                } else {
                    0.0
                }
            }

            // BlendAlpha returns 1.0 (no blending). Vanilla also returns 1.0
            // unless an active Blender overrides it in NoiseChunk.
            DensityFunction::BlendAlpha => 1.0,

            // BlendOffset returns 0.0 (no offset). Vanilla also returns 0.0
            // unless an active Blender overrides it in NoiseChunk.
            // TODO: Implement EndIslands (requires SimplexNoise + island generation algorithm).
            DensityFunction::BlendOffset | DensityFunction::EndIslands => 0.0,

            // TODO: Implement Blender integration for chunk-boundary blending.
            // Pass-through is correct when no Blender is active.
            // Cache markers are also transparent in single-point evaluation.
            DensityFunction::BlendDensity(f)
            | DensityFunction::CacheOnce(f)
            | DensityFunction::Cache2D(f)
            | DensityFunction::CacheAllInCell(f)
            | DensityFunction::FlatCache(f)
            | DensityFunction::Interpolated(f) => self.evaluate(f, ctx),
        }
    }

    /// Evaluate a density function with caching support.
    ///
    /// Like [`evaluate`](Self::evaluate) but uses an [`EvalCache`] to avoid redundant evaluations.
    /// `CacheOnce` caches by last (x, y, z), `FlatCache`/`Cache2D` cache by (x, z).
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn evaluate_cached(
        &self,
        func: &DensityFunction,
        ctx: &DensityContext,
        cache: &mut EvalCache,
    ) -> f64 {
        match func {
            DensityFunction::Constant(v) => *v,

            DensityFunction::Reference(id) => {
                if let Some(referenced) = self.registry.get(id) {
                    self.evaluate_cached(referenced, ctx, cache)
                } else {
                    0.0
                }
            }

            DensityFunction::YClampedGradient {
                from_y,
                to_y,
                from_value,
                to_value,
            } => map_clamped(
                f64::from(ctx.y),
                f64::from(*from_y),
                f64::from(*to_y),
                *from_value,
                *to_value,
            ),

            DensityFunction::Noise {
                noise_id,
                xz_scale,
                y_scale,
            } => {
                if let Some(noise) = self.noises.get(noise_id) {
                    noise.get_value(
                        f64::from(ctx.x) * xz_scale,
                        f64::from(ctx.y) * y_scale,
                        f64::from(ctx.z) * xz_scale,
                    )
                } else {
                    0.0
                }
            }

            DensityFunction::ShiftedNoise {
                shift_x,
                shift_z,
                xz_scale,
                y_scale,
                noise_id,
            } => {
                let dx = self.evaluate_cached(shift_x, ctx, cache);
                let dz = self.evaluate_cached(shift_z, ctx, cache);
                if let Some(noise) = self.noises.get(noise_id) {
                    noise.get_value(
                        f64::from(ctx.x) * xz_scale + dx,
                        f64::from(ctx.y) * y_scale,
                        f64::from(ctx.z) * xz_scale + dz,
                    )
                } else {
                    0.0
                }
            }

            DensityFunction::ShiftA { noise_id } => {
                if let Some(noise) = self.noises.get(noise_id) {
                    noise.get_value(f64::from(ctx.x) * 0.25, 0.0, f64::from(ctx.z) * 0.25) * 4.0
                } else {
                    0.0
                }
            }

            DensityFunction::ShiftB { noise_id } => {
                if let Some(noise) = self.noises.get(noise_id) {
                    noise.get_value(f64::from(ctx.z) * 0.25, f64::from(ctx.x) * 0.25, 0.0) * 4.0
                } else {
                    0.0
                }
            }

            DensityFunction::Shift { noise_id } => {
                if let Some(noise) = self.noises.get(noise_id) {
                    noise.get_value(
                        f64::from(ctx.x) * 0.25,
                        f64::from(ctx.y) * 0.25,
                        f64::from(ctx.z) * 0.25,
                    ) * 4.0
                } else {
                    0.0
                }
            }

            DensityFunction::Add(a, b) => {
                self.evaluate_cached(a, ctx, cache) + self.evaluate_cached(b, ctx, cache)
            }

            DensityFunction::Mul(a, b) => {
                self.evaluate_cached(a, ctx, cache) * self.evaluate_cached(b, ctx, cache)
            }

            DensityFunction::Min(a, b) => self
                .evaluate_cached(a, ctx, cache)
                .min(self.evaluate_cached(b, ctx, cache)),

            DensityFunction::Max(a, b) => self
                .evaluate_cached(a, ctx, cache)
                .max(self.evaluate_cached(b, ctx, cache)),

            DensityFunction::Abs(f) => self.evaluate_cached(f, ctx, cache).abs(),

            DensityFunction::Square(f) => {
                let v = self.evaluate_cached(f, ctx, cache);
                v * v
            }

            DensityFunction::Cube(f) => {
                let v = self.evaluate_cached(f, ctx, cache);
                v * v * v
            }

            DensityFunction::HalfNegative(f) => {
                let v = self.evaluate_cached(f, ctx, cache);
                if v > 0.0 { v } else { v * 0.5 }
            }

            DensityFunction::QuarterNegative(f) => {
                let v = self.evaluate_cached(f, ctx, cache);
                if v > 0.0 { v } else { v * 0.25 }
            }

            DensityFunction::Squeeze(f) => {
                let v = self.evaluate_cached(f, ctx, cache);
                let c = clamp(v, -1.0, 1.0);
                c / 2.0 - c * c * c / 24.0
            }

            DensityFunction::Clamp { input, min, max } => {
                clamp(self.evaluate_cached(input, ctx, cache), *min, *max)
            }

            DensityFunction::RangeChoice {
                input,
                min_inclusive,
                max_exclusive,
                when_in_range,
                when_out_of_range,
            } => {
                let v = self.evaluate_cached(input, ctx, cache);
                if v >= *min_inclusive && v < *max_exclusive {
                    self.evaluate_cached(when_in_range, ctx, cache)
                } else {
                    self.evaluate_cached(when_out_of_range, ctx, cache)
                }
            }

            DensityFunction::Spline(spline) => self.evaluate_spline_cached(spline, ctx, cache),

            DensityFunction::BlendedNoise {
                xz_scale,
                y_scale,
                xz_factor,
                y_factor,
                smear_scale_multiplier,
            } => {
                if let Some(noise) = self.noises.get("offset") {
                    let scaled_xz = xz_scale / xz_factor;
                    let scaled_y = y_scale / y_factor;
                    noise.get_value(
                        f64::from(ctx.x) * scaled_xz,
                        f64::from(ctx.y) * scaled_y,
                        f64::from(ctx.z) * scaled_xz,
                    ) * smear_scale_multiplier
                } else {
                    0.0
                }
            }

            DensityFunction::WeirdScaledSampler {
                input,
                noise_id,
                rarity_value_mapper,
            } => {
                let rarity = self.evaluate_cached(input, ctx, cache);
                let scale = rarity_value_mapper.get_values(rarity);
                if let Some(noise) = self.noises.get(noise_id) {
                    scale
                        * noise
                            .get_value(
                                f64::from(ctx.x) / scale,
                                f64::from(ctx.y) / scale,
                                f64::from(ctx.z) / scale,
                            )
                            .abs()
                } else {
                    0.0
                }
            }

            DensityFunction::BlendAlpha => 1.0,
            DensityFunction::BlendOffset | DensityFunction::EndIslands => 0.0,
            DensityFunction::BlendDensity(f) => self.evaluate_cached(f, ctx, cache),

            // CacheOnce: cache the last (x, y, z) → value per node.
            // Avoids re-evaluating shared subtrees when multiple climate parameters
            // are sampled at the same position.
            DensityFunction::CacheOnce(f) => {
                let key = Arc::as_ptr(f) as usize;
                if let Some(&(cx, cy, cz, val)) = cache.once.get(&key)
                    && cx == ctx.x
                    && cy == ctx.y
                    && cz == ctx.z
                {
                    return val;
                }
                let val = self.evaluate_cached(f, ctx, cache);
                cache.once.insert(key, (ctx.x, ctx.y, ctx.z, val));
                val
            }

            // FlatCache/Cache2D: cache by (x, z) per node, ignoring y.
            // Huge win for climate parameters (temperature, humidity, continentalness,
            // erosion) which only depend on horizontal position.
            DensityFunction::FlatCache(f) | DensityFunction::Cache2D(f) => {
                let key = Arc::as_ptr(f) as usize;
                let flat_key = (key, ctx.x, ctx.z);
                if let Some(&val) = cache.flat.get(&flat_key) {
                    return val;
                }
                let val = self.evaluate_cached(f, ctx, cache);
                cache.flat.insert(flat_key, val);
                val
            }

            // CacheAllInCell/Interpolated: pass-through for now (used by terrain, not biomes).
            DensityFunction::CacheAllInCell(f) | DensityFunction::Interpolated(f) => {
                self.evaluate_cached(f, ctx, cache)
            }
        }
    }

    /// Evaluate a cubic spline.
    fn evaluate_spline(&self, spline: &CubicSpline, ctx: &DensityContext) -> f64 {
        // Get the input value from the coordinate function
        let input = self.evaluate(&spline.coordinate, ctx) as f32;
        f64::from(self.evaluate_spline_at(spline, input, ctx))
    }

    /// Evaluate a cubic spline at a given input value.
    ///
    /// Matches vanilla's `CubicSpline.Multipoint.apply()`:
    /// - If input is before the first point, linear extrapolation using first derivative
    /// - If input is after the last point, linear extrapolation using last derivative
    /// - Otherwise, Hermite cubic interpolation between adjacent points
    fn evaluate_spline_at(&self, spline: &CubicSpline, input: f32, ctx: &DensityContext) -> f32 {
        if spline.points.is_empty() {
            return 0.0;
        }

        let last_index = spline.points.len() - 1;

        // Find interval start: largest i where locations[i] <= input, or -1 if input < all locations
        // Matches vanilla's findIntervalStart using binarySearch
        let start = {
            let mut lo = 0i32;
            let mut hi = spline.points.len() as i32;
            while lo < hi {
                let mid = i32::midpoint(lo, hi);
                if input < spline.points[mid as usize].location {
                    hi = mid;
                } else {
                    lo = mid + 1;
                }
            }
            lo - 1
        };

        if start < 0 {
            // Before first point: linear extrapolation
            let p = &spline.points[0];
            let value = self.get_spline_point_value(p, ctx);
            return value + p.derivative * (input - p.location);
        }

        let start = start as usize;
        if start == last_index {
            // After last point: linear extrapolation
            let p = &spline.points[last_index];
            let value = self.get_spline_point_value(p, ctx);
            return value + p.derivative * (input - p.location);
        }

        // Hermite cubic interpolation between points[start] and points[start+1]
        let p0 = &spline.points[start];
        let p1 = &spline.points[start + 1];
        let x1 = p0.location;
        let x2 = p1.location;
        let t = (input - x1) / (x2 - x1);
        let y1 = self.get_spline_point_value(p0, ctx);
        let y2 = self.get_spline_point_value(p1, ctx);
        let d1 = p0.derivative;
        let d2 = p1.derivative;

        // Vanilla formula: lerp(t, y1, y2) + t * (1 - t) * lerp(t, a, b)
        let h = x2 - x1;
        let a = d1 * h - (y2 - y1);
        let b = -d2 * h + (y2 - y1);
        let lerp_y = y1 + t * (y2 - y1);
        let lerp_ab = a + t * (b - a);
        lerp_y + t * (1.0 - t) * lerp_ab
    }

    /// Get the value at a spline point.
    fn get_spline_point_value(&self, point: &SplinePoint, ctx: &DensityContext) -> f32 {
        match &point.value {
            SplineValue::Constant(v) => *v,
            SplineValue::Spline(nested) => {
                let nested_input = self.evaluate(&nested.coordinate, ctx) as f32;
                self.evaluate_spline_at(nested, nested_input, ctx)
            }
        }
    }

    /// Evaluate a cubic spline with caching.
    fn evaluate_spline_cached(
        &self,
        spline: &CubicSpline,
        ctx: &DensityContext,
        cache: &mut EvalCache,
    ) -> f64 {
        let input = self.evaluate_cached(&spline.coordinate, ctx, cache) as f32;
        f64::from(self.evaluate_spline_at_cached(spline, input, ctx, cache))
    }

    /// Evaluate a cubic spline at a given input value with caching.
    fn evaluate_spline_at_cached(
        &self,
        spline: &CubicSpline,
        input: f32,
        ctx: &DensityContext,
        cache: &mut EvalCache,
    ) -> f32 {
        if spline.points.is_empty() {
            return 0.0;
        }

        let last_index = spline.points.len() - 1;

        let start = {
            let mut lo = 0i32;
            let mut hi = spline.points.len() as i32;
            while lo < hi {
                let mid = i32::midpoint(lo, hi);
                if input < spline.points[mid as usize].location {
                    hi = mid;
                } else {
                    lo = mid + 1;
                }
            }
            lo - 1
        };

        if start < 0 {
            let p = &spline.points[0];
            let value = self.get_spline_point_value_cached(p, ctx, cache);
            return value + p.derivative * (input - p.location);
        }

        let start = start as usize;
        if start == last_index {
            let p = &spline.points[last_index];
            let value = self.get_spline_point_value_cached(p, ctx, cache);
            return value + p.derivative * (input - p.location);
        }

        let p0 = &spline.points[start];
        let p1 = &spline.points[start + 1];
        let x1 = p0.location;
        let x2 = p1.location;
        let t = (input - x1) / (x2 - x1);
        let y1 = self.get_spline_point_value_cached(p0, ctx, cache);
        let y2 = self.get_spline_point_value_cached(p1, ctx, cache);
        let d1 = p0.derivative;
        let d2 = p1.derivative;

        let h = x2 - x1;
        let a = d1 * h - (y2 - y1);
        let b = -d2 * h + (y2 - y1);
        let lerp_y = y1 + t * (y2 - y1);
        let lerp_ab = a + t * (b - a);
        lerp_y + t * (1.0 - t) * lerp_ab
    }

    /// Get the value at a spline point with caching.
    fn get_spline_point_value_cached(
        &self,
        point: &SplinePoint,
        ctx: &DensityContext,
        cache: &mut EvalCache,
    ) -> f32 {
        match &point.value {
            SplineValue::Constant(v) => *v,
            SplineValue::Spline(nested) => {
                let nested_input = self.evaluate_cached(&nested.coordinate, ctx, cache) as f32;
                self.evaluate_spline_at_cached(nested, nested_input, ctx, cache)
            }
        }
    }
}

/// Parameters for creating a noise generator.
#[derive(Debug, Clone)]
pub struct NoiseParameters {
    /// The first octave level.
    pub first_octave: i32,
    /// Amplitude multipliers for each octave.
    pub amplitudes: Vec<f64>,
}

impl NoiseParameters {
    /// Create new noise parameters.
    #[must_use]
    pub const fn new(first_octave: i32, amplitudes: Vec<f64>) -> Self {
        Self {
            first_octave,
            amplitudes,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random::Random;
    use crate::random::xoroshiro::Xoroshiro;
    use std::sync::Arc;

    fn make_test_evaluator() -> DensityEvaluator {
        let mut rng = Xoroshiro::from_seed(12345);
        let splitter = rng.next_positional();

        let mut noise_params = FxHashMap::default();
        noise_params.insert(
            "test_noise".to_string(),
            NoiseParameters::new(-4, vec![1.0, 1.0, 1.0, 1.0]),
        );

        DensityEvaluator::new(&splitter, &noise_params)
    }

    #[test]
    fn test_constant() {
        let evaluator = make_test_evaluator();
        let func = DensityFunction::Constant(42.0);
        let ctx = DensityContext::new(0, 64, 0);
        assert!((evaluator.evaluate(&func, &ctx) - 42.0).abs() < 1e-10);
    }

    #[test]
    fn test_y_clamped_gradient() {
        let evaluator = make_test_evaluator();
        let func = DensityFunction::YClampedGradient {
            from_y: -64,
            to_y: 320,
            from_value: 1.0,
            to_value: -1.0,
        };

        // At from_y, should get from_value
        let ctx = DensityContext::new(0, -64, 0);
        assert!((evaluator.evaluate(&func, &ctx) - 1.0).abs() < 1e-10);

        // At to_y, should get to_value
        let ctx = DensityContext::new(0, 320, 0);
        assert!((evaluator.evaluate(&func, &ctx) - (-1.0)).abs() < 1e-10);

        // At midpoint (128), should get 0.0
        let ctx = DensityContext::new(0, 128, 0);
        assert!(evaluator.evaluate(&func, &ctx).abs() < 0.1);
    }

    #[test]
    fn test_add() {
        let evaluator = make_test_evaluator();
        let a = Arc::new(DensityFunction::Constant(10.0));
        let b = Arc::new(DensityFunction::Constant(5.0));
        let func = DensityFunction::Add(a, b);
        let ctx = DensityContext::new(0, 64, 0);
        assert!((evaluator.evaluate(&func, &ctx) - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_mul() {
        let evaluator = make_test_evaluator();
        let a = Arc::new(DensityFunction::Constant(3.0));
        let b = Arc::new(DensityFunction::Constant(4.0));
        let func = DensityFunction::Mul(a, b);
        let ctx = DensityContext::new(0, 64, 0);
        assert!((evaluator.evaluate(&func, &ctx) - 12.0).abs() < 1e-10);
    }

    #[test]
    fn test_clamp() {
        let evaluator = make_test_evaluator();
        let func = DensityFunction::Clamp {
            input: Arc::new(DensityFunction::Constant(10.0)),
            min: -1.0,
            max: 1.0,
        };
        let ctx = DensityContext::new(0, 64, 0);
        assert!((evaluator.evaluate(&func, &ctx) - 1.0).abs() < 1e-10);

        let func = DensityFunction::Clamp {
            input: Arc::new(DensityFunction::Constant(-10.0)),
            min: -1.0,
            max: 1.0,
        };
        assert!((evaluator.evaluate(&func, &ctx) - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_half_negative() {
        let evaluator = make_test_evaluator();
        let ctx = DensityContext::new(0, 64, 0);

        // Positive should pass through
        let func = DensityFunction::HalfNegative(Arc::new(DensityFunction::Constant(2.0)));
        assert!((evaluator.evaluate(&func, &ctx) - 2.0).abs() < 1e-10);

        // Negative should be halved
        let func = DensityFunction::HalfNegative(Arc::new(DensityFunction::Constant(-2.0)));
        assert!((evaluator.evaluate(&func, &ctx) - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_squeeze() {
        let evaluator = make_test_evaluator();
        let ctx = DensityContext::new(0, 64, 0);

        // Test that squeeze clamps and applies the formula
        let func = DensityFunction::Squeeze(Arc::new(DensityFunction::Constant(0.0)));
        assert!(evaluator.evaluate(&func, &ctx).abs() < 1e-10);

        // At 1.0: c/2 - c³/24 = 0.5 - 1/24 ≈ 0.458
        let func = DensityFunction::Squeeze(Arc::new(DensityFunction::Constant(1.0)));
        let v = evaluator.evaluate(&func, &ctx);
        assert!((v - (0.5 - 1.0 / 24.0)).abs() < 1e-10);
    }
}
