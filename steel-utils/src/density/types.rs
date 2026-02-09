//! Density function types matching vanilla Minecraft's DensityFunctions.java
//!
//! These types represent the data-driven density function tree used for world generation.

use std::sync::Arc;

/// A density function that can be evaluated at a position to get a density value.
///
/// Density functions form a tree structure where complex functions are composed
/// from simpler ones. This matches vanilla's `DensityFunction` interface.
#[derive(Debug, Clone)]
pub enum DensityFunction {
    /// A constant value.
    Constant(f64),

    /// A reference to another density function by ID.
    /// This must be resolved before evaluation.
    Reference(String),

    /// A Y-axis clamped gradient.
    ///
    /// Returns `from_value` at Y = `from_y`, `to_value` at Y = `to_y`,
    /// linearly interpolated between, clamped outside the range.
    YClampedGradient {
        /// Starting Y coordinate
        from_y: i32,
        /// Ending Y coordinate
        to_y: i32,
        /// Value at `from_y`
        from_value: f64,
        /// Value at `to_y`
        to_value: f64,
    },

    /// Sample from a noise generator.
    Noise {
        /// Noise identifier
        noise_id: String,
        /// XZ scale factor
        xz_scale: f64,
        /// Y scale factor
        y_scale: f64,
    },

    /// Sample from a shifted noise generator.
    ShiftedNoise {
        /// X coordinate shift
        shift_x: Arc<DensityFunction>,
        /// Y coordinate shift
        shift_y: Arc<DensityFunction>,
        /// Z coordinate shift
        shift_z: Arc<DensityFunction>,
        /// XZ scale factor
        xz_scale: f64,
        /// Y scale factor
        y_scale: f64,
        /// Noise identifier
        noise_id: String,
    },

    /// Shift noise generator A for coordinate offsetting.
    ShiftA {
        /// Noise identifier
        noise_id: String,
    },
    /// Shift noise generator B for coordinate offsetting.
    ShiftB {
        /// Noise identifier
        noise_id: String,
    },
    /// Generic shift noise generator for coordinate offsetting.
    Shift {
        /// Noise identifier
        noise_id: String,
    },

    /// Add two density functions.
    Add(Arc<DensityFunction>, Arc<DensityFunction>),

    /// Multiply two density functions.
    Mul(Arc<DensityFunction>, Arc<DensityFunction>),

    /// Take the minimum of two density functions.
    Min(Arc<DensityFunction>, Arc<DensityFunction>),

    /// Take the maximum of two density functions.
    Max(Arc<DensityFunction>, Arc<DensityFunction>),

    /// Absolute value.
    Abs(Arc<DensityFunction>),

    /// Square the value.
    Square(Arc<DensityFunction>),

    /// Cube the value.
    Cube(Arc<DensityFunction>),

    /// Half negative: if v > 0 then v else v * 0.5
    HalfNegative(Arc<DensityFunction>),

    /// Quarter negative: if v > 0 then v else v * 0.25
    QuarterNegative(Arc<DensityFunction>),

    /// Squeeze: clamp(-1, 1) then apply c/2 - c³/24
    Squeeze(Arc<DensityFunction>),

    /// Clamp the value to a range.
    Clamp {
        /// Input density function
        input: Arc<DensityFunction>,
        /// Minimum value
        min: f64,
        /// Maximum value
        max: f64,
    },

    /// Choose between two functions based on input range.
    RangeChoice {
        /// Input density function
        input: Arc<DensityFunction>,
        /// Minimum inclusive bound
        min_inclusive: f64,
        /// Maximum exclusive bound
        max_exclusive: f64,
        /// Function to use when in range
        when_in_range: Arc<DensityFunction>,
        /// Function to use when out of range
        when_out_of_range: Arc<DensityFunction>,
    },

    /// Cubic spline evaluation.
    Spline(Arc<CubicSpline>),

    /// Blended (interpolated) 3D noise.
    BlendedNoise {
        /// XZ scale factor
        xz_scale: f64,
        /// Y scale factor
        y_scale: f64,
        /// XZ interpolation factor
        xz_factor: f64,
        /// Y interpolation factor
        y_factor: f64,
        /// Smear scale multiplier
        smear_scale_multiplier: f64,
    },

    /// Weird scaled sampler (for cave generation).
    WeirdScaledSampler {
        /// Input density function
        input: Arc<DensityFunction>,
        /// Noise identifier
        noise_id: String,
        /// Rarity value mapper
        rarity_value_mapper: RarityValueMapper,
    },

    /// End islands density function.
    EndIslands,

    /// Blend alpha (returns 1.0, placeholder for blending).
    BlendAlpha,

    /// Blend offset (returns 0.0, placeholder for blending).
    BlendOffset,

    /// Blend density (for chunk blending).
    BlendDensity(Arc<DensityFunction>),

    // === Cache markers (optimization hints) ===
    /// Cache the result for one evaluation.
    CacheOnce(Arc<DensityFunction>),

    /// Cache for 2D (XZ) positions.
    Cache2D(Arc<DensityFunction>),

    /// Cache all values in a cell.
    CacheAllInCell(Arc<DensityFunction>),

    /// Flat cache (no caching, just a marker).
    FlatCache(Arc<DensityFunction>),

    /// Interpolated (cell-based interpolation).
    Interpolated(Arc<DensityFunction>),
}

/// Rarity value mapper for cave generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RarityValueMapper {
    /// Mapper type `"type_1"` for tunnels.
    Tunnels,
    /// Mapper type `"type_2"` for caves.
    Caves,
}

impl RarityValueMapper {
    /// Get the scaling factor for this mapper based on rarity value.
    ///
    /// From vanilla NoiseRouterData.QuantizedSpaghettiRarity.
    #[must_use]
    pub fn get_values(&self, rarity: f64) -> f64 {
        match self {
            // Type 1: getSpaghettiRarity3D (tunnels)
            Self::Tunnels => {
                if rarity < -0.5 {
                    0.75
                } else if rarity < 0.0 {
                    1.0
                } else if rarity < 0.5 {
                    1.5
                } else {
                    2.0
                }
            }
            // Type 2: getSpaghettiRarity2D (caves)
            Self::Caves => {
                if rarity < -0.75 {
                    0.5
                } else if rarity < -0.5 {
                    0.75
                } else if rarity < 0.5 {
                    1.0
                } else if rarity < 0.75 {
                    2.0
                } else {
                    3.0
                }
            }
        }
    }
}

/// A cubic spline for density function interpolation.
#[derive(Debug, Clone)]
pub struct CubicSpline {
    /// The coordinate extractor (which density function to use as input)
    pub coordinate: Arc<DensityFunction>,
    /// The spline points
    pub points: Vec<SplinePoint>,
}

/// A point in a cubic spline.
#[derive(Debug, Clone)]
pub struct SplinePoint {
    /// The location (input value) of this point.
    pub location: f32,
    /// The value or nested spline at this point.
    pub value: SplineValue,
    /// The derivative at this point.
    pub derivative: f32,
}

/// A spline point value can be either a constant or a nested spline.
#[derive(Debug, Clone)]
pub enum SplineValue {
    /// A constant value.
    Constant(f32),
    /// A nested spline.
    Spline(Arc<CubicSpline>),
}

impl CubicSpline {
    /// Create a new cubic spline.
    #[must_use]
    pub const fn new(coordinate: Arc<DensityFunction>, points: Vec<SplinePoint>) -> Self {
        Self { coordinate, points }
    }
}

/// Context for evaluating density functions at a position.
#[derive(Debug, Clone, Copy)]
pub struct DensityContext {
    /// X coordinate (block position)
    pub x: i32,
    /// Y coordinate (block position)
    pub y: i32,
    /// Z coordinate (block position)
    pub z: i32,
}

impl DensityContext {
    /// Create a new density context.
    #[must_use]
    pub const fn new(x: i32, y: i32, z: i32) -> Self {
        Self { x, y, z }
    }
}

/// A noise router containing all the density functions for world generation.
#[derive(Debug, Clone)]
pub struct NoiseRouter {
    /// Barrier noise for aquifers
    pub barrier_noise: Arc<DensityFunction>,
    /// Fluid level floodedness
    pub fluid_level_floodedness: Arc<DensityFunction>,
    /// Fluid level spread
    pub fluid_level_spread: Arc<DensityFunction>,
    /// Lava noise
    pub lava: Arc<DensityFunction>,
    /// Temperature (for biome selection)
    pub temperature: Arc<DensityFunction>,
    /// Vegetation/humidity (for biome selection)
    pub vegetation: Arc<DensityFunction>,
    /// Continentalness (for biome selection)
    pub continentalness: Arc<DensityFunction>,
    /// Erosion (for biome selection)
    pub erosion: Arc<DensityFunction>,
    /// Depth (for biome selection)
    pub depth: Arc<DensityFunction>,
    /// Ridges/weirdness (for biome selection)
    pub ridges: Arc<DensityFunction>,
    /// Preliminary surface level (for aquifers and surface rules)
    pub preliminary_surface_level: Arc<DensityFunction>,
    /// Final density (for terrain generation)
    pub final_density: Arc<DensityFunction>,
    /// Vein toggle
    pub vein_toggle: Arc<DensityFunction>,
    /// Vein ridged
    pub vein_ridged: Arc<DensityFunction>,
    /// Vein gap
    pub vein_gap: Arc<DensityFunction>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant() {
        let func = DensityFunction::Constant(42.0);
        if let DensityFunction::Constant(v) = func {
            assert!((v - 42.0).abs() < 1e-10);
        } else {
            panic!("Expected Constant variant");
        }
    }

    #[test]
    fn test_rarity_value_mapper_tunnels() {
        // getSpaghettiRarity3D from vanilla
        let mapper = RarityValueMapper::Tunnels;
        assert!((mapper.get_values(-0.6) - 0.75).abs() < 0.01); // < -0.5
        assert!((mapper.get_values(-0.3) - 1.0).abs() < 0.01); // >= -0.5, < 0.0
        assert!((mapper.get_values(0.0) - 1.5).abs() < 0.01); // >= 0.0, < 0.5
        assert!((mapper.get_values(0.3) - 1.5).abs() < 0.01); // >= 0.0, < 0.5
        assert!((mapper.get_values(0.6) - 2.0).abs() < 0.01); // >= 0.5
    }

    #[test]
    fn test_rarity_value_mapper_caves() {
        // getSpaghettiRarity2D from vanilla
        let mapper = RarityValueMapper::Caves;
        assert!((mapper.get_values(-0.8) - 0.5).abs() < 0.01); // < -0.75
        assert!((mapper.get_values(-0.6) - 0.75).abs() < 0.01); // >= -0.75, < -0.5
        assert!((mapper.get_values(0.0) - 1.0).abs() < 0.01); // >= -0.5, < 0.5
        assert!((mapper.get_values(0.6) - 2.0).abs() < 0.01); // >= 0.5, < 0.75
        assert!((mapper.get_values(0.8) - 3.0).abs() < 0.01); // >= 0.75
    }
}
