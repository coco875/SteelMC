//! Climate sampler for vanilla world generation.
//!
//! This module bridges the extracted density functions from steel-registry
//! with the climate sampling system in steel-utils.

use std::sync::Arc;

use rustc_hash::FxHashMap;

use steel_registry::density_functions::{
    self, CubicSpline as GenCubicSpline, DensityFunction as GenDensityFunction,
    OVERWORLD_NOISE_ROUTER, RarityValueMapper as GenRarityValueMapper,
};
use steel_utils::climate::{TargetPoint, quantize_coord};
use steel_utils::density::{
    CubicSpline, DensityContext, DensityEvaluator, DensityFunction, EvalCache, RarityValueMapper,
    SplinePoint, SplineValue,
};
use steel_registry::noise_parameters::get_noise_parameters;
use steel_utils::random::{Random, xoroshiro::Xoroshiro};

/// Climate sampler that uses the extracted vanilla density functions.
pub struct VanillaClimateSampler {
    /// The density evaluator with noise generators
    evaluator: DensityEvaluator,
    /// Temperature density function
    temperature: Arc<DensityFunction>,
    /// Humidity/vegetation density function
    humidity: Arc<DensityFunction>,
    /// Continentalness density function
    continentalness: Arc<DensityFunction>,
    /// Erosion density function
    erosion: Arc<DensityFunction>,
    /// Depth density function
    depth: Arc<DensityFunction>,
    /// Weirdness/ridges density function
    weirdness: Arc<DensityFunction>,
}

impl VanillaClimateSampler {
    /// Create a new vanilla climate sampler with the given seed.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        // Create random number generator from seed
        let mut rng = Xoroshiro::from_seed(seed);
        let splitter = rng.next_positional();

        // Build noise parameters from vanilla datapack
        let noise_params = get_noise_parameters();

        // Create the density evaluator with noises
        let mut evaluator = DensityEvaluator::new(&splitter, &noise_params);

        // Build density function registry from generated code
        let registry = build_density_function_registry();
        for (id, func) in registry {
            evaluator.add_to_registry(id, func);
        }

        // Get the overworld noise router and convert to steel-utils types
        let router = &*OVERWORLD_NOISE_ROUTER;

        Self {
            evaluator,
            temperature: convert_density_function(&router.temperature),
            humidity: convert_density_function(&router.vegetation),
            continentalness: convert_density_function(&router.continents),
            erosion: convert_density_function(&router.erosion),
            depth: convert_density_function(&router.depth),
            weirdness: convert_density_function(&router.ridges),
        }
    }

    /// Sample climate at a quart position using a reusable cache.
    ///
    /// The cache should persist across multiple calls (e.g. for an entire chunk)
    /// to benefit from `FlatCache` (caches by x,z) and `CacheOnce` (caches last position).
    #[must_use]
    pub fn sample(
        &self,
        quart_x: i32,
        quart_y: i32,
        quart_z: i32,
        cache: &mut EvalCache,
    ) -> TargetPoint {
        let block_x = quart_x << 2;
        let block_y = quart_y << 2;
        let block_z = quart_z << 2;

        let ctx = DensityContext::new(block_x, block_y, block_z);

        let temp = self
            .evaluator
            .evaluate_cached(&self.temperature, &ctx, cache) as f32;
        let humidity = self.evaluator.evaluate_cached(&self.humidity, &ctx, cache) as f32;
        let cont = self
            .evaluator
            .evaluate_cached(&self.continentalness, &ctx, cache) as f32;
        let erosion = self.evaluator.evaluate_cached(&self.erosion, &ctx, cache) as f32;
        let depth = self.evaluator.evaluate_cached(&self.depth, &ctx, cache) as f32;
        let weirdness = self.evaluator.evaluate_cached(&self.weirdness, &ctx, cache) as f32;

        TargetPoint::new(
            quantize_coord(f64::from(temp)),
            quantize_coord(f64::from(humidity)),
            quantize_coord(f64::from(cont)),
            quantize_coord(f64::from(erosion)),
            quantize_coord(f64::from(depth)),
            quantize_coord(f64::from(weirdness)),
        )
    }
}

/// Build the density function registry from generated code.
fn build_density_function_registry() -> FxHashMap<String, Arc<DensityFunction>> {
    let mut registry = FxHashMap::default();

    // Get all density functions from the generated registry
    let ids = [
        "minecraft:overworld/continents",
        "minecraft:overworld/erosion",
        "minecraft:overworld/depth",
        "minecraft:overworld/ridges",
        "minecraft:overworld/ridges_folded",
        "minecraft:overworld/offset",
        "minecraft:overworld/factor",
        "minecraft:overworld/jaggedness",
        "minecraft:overworld/sloped_cheese",
        "minecraft:overworld/base_3d_noise",
        "minecraft:overworld/caves/entrances",
        "minecraft:overworld/caves/spaghetti_2d",
        "minecraft:overworld/caves/spaghetti_2d_thickness_modulator",
        "minecraft:overworld/caves/spaghetti_roughness_function",
        "minecraft:overworld/caves/pillars",
        "minecraft:overworld/caves/noodle",
        "minecraft:shift_x",
        "minecraft:shift_z",
        "minecraft:y",
        "minecraft:zero",
    ];

    for id in ids {
        if let Some(gen_func) = density_functions::get_density_function(id) {
            registry.insert(id.to_string(), convert_density_function(&gen_func));
        }
    }

    registry
}

/// Convert a generated `DensityFunction` to the steel-utils `DensityFunction` type.
#[allow(clippy::too_many_lines)]
fn convert_density_function(df: &GenDensityFunction) -> Arc<DensityFunction> {
    Arc::new(match df {
        GenDensityFunction::Constant(v) => DensityFunction::Constant(*v),

        GenDensityFunction::Reference(id) => DensityFunction::Reference(id.to_string()),

        GenDensityFunction::YClampedGradient {
            from_y,
            to_y,
            from_value,
            to_value,
        } => DensityFunction::YClampedGradient {
            from_y: *from_y,
            to_y: *to_y,
            from_value: *from_value,
            to_value: *to_value,
        },

        GenDensityFunction::Noise {
            noise_id,
            xz_scale,
            y_scale,
        } => DensityFunction::Noise {
            noise_id: noise_id.to_string(),
            xz_scale: *xz_scale,
            y_scale: *y_scale,
        },

        GenDensityFunction::ShiftedNoise {
            shift_x,
            shift_y: _, // Vanilla ShiftedNoise ignores shift_y (always uses 0)
            shift_z,
            xz_scale,
            y_scale,
            noise_id,
        } => DensityFunction::ShiftedNoise {
            shift_x: convert_density_function(shift_x),
            shift_z: convert_density_function(shift_z),
            xz_scale: *xz_scale,
            y_scale: *y_scale,
            noise_id: noise_id.to_string(),
        },

        GenDensityFunction::ShiftA { noise_id } => DensityFunction::ShiftA {
            noise_id: noise_id.to_string(),
        },

        GenDensityFunction::ShiftB { noise_id } => DensityFunction::ShiftB {
            noise_id: noise_id.to_string(),
        },

        GenDensityFunction::Shift { noise_id } => DensityFunction::Shift {
            noise_id: noise_id.to_string(),
        },

        GenDensityFunction::Clamp { input, min, max } => DensityFunction::Clamp {
            input: convert_density_function(input),
            min: *min,
            max: *max,
        },

        GenDensityFunction::Abs(f) => DensityFunction::Abs(convert_density_function(f)),
        GenDensityFunction::Square(f) => DensityFunction::Square(convert_density_function(f)),
        GenDensityFunction::Cube(f) => DensityFunction::Cube(convert_density_function(f)),
        GenDensityFunction::HalfNegative(f) => {
            DensityFunction::HalfNegative(convert_density_function(f))
        }
        GenDensityFunction::QuarterNegative(f) => {
            DensityFunction::QuarterNegative(convert_density_function(f))
        }
        GenDensityFunction::Squeeze(f) => DensityFunction::Squeeze(convert_density_function(f)),

        GenDensityFunction::Add(a, b) => {
            DensityFunction::Add(convert_density_function(a), convert_density_function(b))
        }
        GenDensityFunction::Mul(a, b) => {
            DensityFunction::Mul(convert_density_function(a), convert_density_function(b))
        }
        GenDensityFunction::Min(a, b) => {
            DensityFunction::Min(convert_density_function(a), convert_density_function(b))
        }
        GenDensityFunction::Max(a, b) => {
            DensityFunction::Max(convert_density_function(a), convert_density_function(b))
        }

        GenDensityFunction::Spline(spline) => {
            DensityFunction::Spline(Arc::new(convert_spline(spline)))
        }

        GenDensityFunction::RangeChoice {
            input,
            min_inclusive,
            max_exclusive,
            when_in_range,
            when_out_of_range,
        } => DensityFunction::RangeChoice {
            input: convert_density_function(input),
            min_inclusive: *min_inclusive,
            max_exclusive: *max_exclusive,
            when_in_range: convert_density_function(when_in_range),
            when_out_of_range: convert_density_function(when_out_of_range),
        },

        GenDensityFunction::Interpolated(f) => {
            DensityFunction::Interpolated(convert_density_function(f))
        }
        GenDensityFunction::FlatCache(f) => DensityFunction::FlatCache(convert_density_function(f)),
        GenDensityFunction::CacheOnce(f) => DensityFunction::CacheOnce(convert_density_function(f)),
        GenDensityFunction::Cache2d(f) => DensityFunction::Cache2D(convert_density_function(f)),
        GenDensityFunction::CacheAllInCell(f) => {
            DensityFunction::CacheAllInCell(convert_density_function(f))
        }

        GenDensityFunction::BlendOffset => DensityFunction::BlendOffset,
        GenDensityFunction::BlendAlpha => DensityFunction::BlendAlpha,
        GenDensityFunction::BlendDensity(f) => {
            DensityFunction::BlendDensity(convert_density_function(f))
        }

        // TODO: Implement Beardifier for structure terrain adaptation.
        // Constant(0.0) is correct when structures are not yet generated.
        GenDensityFunction::Beardifier => DensityFunction::Constant(0.0),
        GenDensityFunction::EndIslands => DensityFunction::EndIslands,

        GenDensityFunction::WeirdScaledSampler {
            input,
            noise_id,
            rarity_value_mapper,
        } => DensityFunction::WeirdScaledSampler {
            input: convert_density_function(input),
            noise_id: noise_id.to_string(),
            rarity_value_mapper: match rarity_value_mapper {
                GenRarityValueMapper::Tunnels => RarityValueMapper::Tunnels,
                GenRarityValueMapper::Caves => RarityValueMapper::Caves,
            },
        },

        GenDensityFunction::OldBlendedNoise {
            xz_scale,
            y_scale,
            xz_factor,
            y_factor,
            smear_scale_multiplier,
        } => DensityFunction::BlendedNoise {
            xz_scale: *xz_scale,
            y_scale: *y_scale,
            xz_factor: *xz_factor,
            y_factor: *y_factor,
            smear_scale_multiplier: *smear_scale_multiplier,
        },
    })
}

/// Convert a generated `CubicSpline` to the steel-utils `CubicSpline` type.
fn convert_spline(spline: &GenCubicSpline) -> CubicSpline {
    match spline {
        GenCubicSpline::Constant(v) => CubicSpline {
            coordinate: Arc::new(DensityFunction::Constant(0.0)),
            points: vec![SplinePoint {
                location: 0.0,
                value: SplineValue::Constant(*v),
                derivative: 0.0,
            }],
        },
        GenCubicSpline::Multipoint { coordinate, points } => CubicSpline {
            coordinate: convert_density_function(coordinate),
            points: points
                .iter()
                .map(|p| SplinePoint {
                    location: p.location,
                    value: convert_spline_value(&p.value),
                    derivative: p.derivative,
                })
                .collect(),
        },
    }
}

/// Convert a generated spline value.
fn convert_spline_value(spline: &GenCubicSpline) -> SplineValue {
    match spline {
        GenCubicSpline::Constant(v) => SplineValue::Constant(*v),
        GenCubicSpline::Multipoint { .. } => SplineValue::Spline(Arc::new(convert_spline(spline))),
    }
}
