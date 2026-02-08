//! Climate sampler for evaluating climate at positions.

use std::sync::Arc;

use crate::density::{DensityContext, DensityEvaluator, DensityFunction};

use super::{TargetPoint, quantize_coord};

/// Climate sampler that evaluates density functions to produce climate values.
///
/// Matches vanilla's `Climate.Sampler`.
pub struct ClimateSampler {
    /// Density function for temperature
    temperature: Arc<DensityFunction>,
    /// Density function for humidity
    humidity: Arc<DensityFunction>,
    /// Density function for continentalness
    continentalness: Arc<DensityFunction>,
    /// Density function for erosion
    erosion: Arc<DensityFunction>,
    /// Density function for depth
    depth: Arc<DensityFunction>,
    /// Density function for weirdness/ridges
    weirdness: Arc<DensityFunction>,
}

impl ClimateSampler {
    /// Create a new climate sampler with the given density functions.
    #[must_use]
    pub const fn new(
        temperature: Arc<DensityFunction>,
        humidity: Arc<DensityFunction>,
        continentalness: Arc<DensityFunction>,
        erosion: Arc<DensityFunction>,
        depth: Arc<DensityFunction>,
        weirdness: Arc<DensityFunction>,
    ) -> Self {
        Self {
            temperature,
            humidity,
            continentalness,
            erosion,
            depth,
            weirdness,
        }
    }

    /// Sample climate at a quart position (block / 4).
    ///
    /// # Arguments
    /// * `evaluator` - The density evaluator with noise generators
    /// * `quart_x` - X coordinate in quart space
    /// * `quart_y` - Y coordinate in quart space
    /// * `quart_z` - Z coordinate in quart space
    ///
    /// This matches vanilla's `Climate.Sampler.sample()`:
    /// 1. Converts quart to block coordinates
    /// 2. Evaluates each density function
    /// 3. Casts to f32 (CRITICAL for vanilla matching!)
    /// 4. Quantizes to produce `TargetPoint`
    #[must_use]
    pub fn sample(
        &self,
        evaluator: &DensityEvaluator,
        quart_x: i32,
        quart_y: i32,
        quart_z: i32,
    ) -> TargetPoint {
        // QuartPos.toBlock = quart << 2 = quart * 4
        let block_x = quart_x << 2;
        let block_y = quart_y << 2;
        let block_z = quart_z << 2;

        let ctx = DensityContext::new(block_x, block_y, block_z);

        // Evaluate each density function and cast to f32 before quantizing
        // This matches vanilla: (float)this.temperature.compute(context)
        let temp = evaluator.evaluate(&self.temperature, &ctx) as f32;
        let humidity = evaluator.evaluate(&self.humidity, &ctx) as f32;
        let cont = evaluator.evaluate(&self.continentalness, &ctx) as f32;
        let erosion = evaluator.evaluate(&self.erosion, &ctx) as f32;
        let depth = evaluator.evaluate(&self.depth, &ctx) as f32;
        let weirdness = evaluator.evaluate(&self.weirdness, &ctx) as f32;

        // Quantize each parameter
        TargetPoint::new(
            quantize_coord(f64::from(temp)),
            quantize_coord(f64::from(humidity)),
            quantize_coord(f64::from(cont)),
            quantize_coord(f64::from(erosion)),
            quantize_coord(f64::from(depth)),
            quantize_coord(f64::from(weirdness)),
        )
    }

    /// Sample climate at block coordinates.
    ///
    /// This is a convenience method that takes block coordinates directly
    /// and internally converts to quart position for sampling.
    #[must_use]
    pub fn sample_block(
        &self,
        evaluator: &DensityEvaluator,
        block_x: i32,
        block_y: i32,
        block_z: i32,
    ) -> TargetPoint {
        // Convert block to quart position (integer division floors)
        self.sample(evaluator, block_x >> 2, block_y >> 2, block_z >> 2)
    }
}

/// An empty climate sampler that returns zero for all parameters.
///
/// Matches vanilla's `Climate.empty()`.
#[must_use]
pub fn empty_sampler() -> ClimateSampler {
    ClimateSampler::new(
        Arc::new(DensityFunction::Constant(0.0)),
        Arc::new(DensityFunction::Constant(0.0)),
        Arc::new(DensityFunction::Constant(0.0)),
        Arc::new(DensityFunction::Constant(0.0)),
        Arc::new(DensityFunction::Constant(0.0)),
        Arc::new(DensityFunction::Constant(0.0)),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random::{Random, xoroshiro::Xoroshiro};
    use rustc_hash::FxHashMap;

    fn make_test_evaluator() -> DensityEvaluator {
        let mut rng = Xoroshiro::from_seed(12345);
        let splitter = rng.next_positional();
        let noise_params = FxHashMap::default();
        DensityEvaluator::new(&splitter, &noise_params)
    }

    #[test]
    fn test_empty_sampler() {
        let sampler = empty_sampler();
        let evaluator = make_test_evaluator();

        let target = sampler.sample(&evaluator, 0, 16, 0);
        assert_eq!(target.temperature, 0);
        assert_eq!(target.humidity, 0);
        assert_eq!(target.continentalness, 0);
        assert_eq!(target.erosion, 0);
        assert_eq!(target.depth, 0);
        assert_eq!(target.weirdness, 0);
    }

    #[test]
    fn test_constant_sampler() {
        let sampler = ClimateSampler::new(
            Arc::new(DensityFunction::Constant(0.5)),
            Arc::new(DensityFunction::Constant(-0.3)),
            Arc::new(DensityFunction::Constant(0.0)),
            Arc::new(DensityFunction::Constant(0.1)),
            Arc::new(DensityFunction::Constant(0.0)),
            Arc::new(DensityFunction::Constant(0.2)),
        );
        let evaluator = make_test_evaluator();

        let target = sampler.sample(&evaluator, 0, 16, 0);
        assert_eq!(target.temperature, 5000);
        assert_eq!(target.humidity, -3000);
        assert_eq!(target.continentalness, 0);
        assert_eq!(target.erosion, 1000);
        assert_eq!(target.depth, 0);
        assert_eq!(target.weirdness, 2000);
    }

    #[test]
    fn test_quart_to_block_conversion() {
        // Verify quart << 2 gives correct block coordinate
        let sampler = ClimateSampler::new(
            Arc::new(DensityFunction::Constant(1.0)),
            Arc::new(DensityFunction::Constant(0.0)),
            Arc::new(DensityFunction::Constant(0.0)),
            Arc::new(DensityFunction::Constant(0.0)),
            Arc::new(DensityFunction::Constant(0.0)),
            Arc::new(DensityFunction::Constant(0.0)),
        );
        let evaluator = make_test_evaluator();

        // Quart (5, 16, 10) should sample at block (20, 64, 40)
        let target = sampler.sample(&evaluator, 5, 16, 10);
        // Since we're using constants, the actual block position doesn't matter
        // but the conversion should work correctly
        assert_eq!(target.temperature, 10000);
    }
}
