//! Multi-octave simplex noise matching vanilla's `PerlinSimplexNoise`.
//!
//! Used for biome temperature calculations. Not to be confused with
//! `PerlinNoise` which uses improved (gradient) noise octaves.

use crate::noise::SimplexNoise;
use crate::random::RandomSource;

/// Multi-octave simplex noise generator.
///
/// Matches vanilla's `net.minecraft.world.level.levelgen.synth.PerlinSimplexNoise`.
/// Created from a set of octave levels; each octave uses a separate `SimplexNoise`.
pub struct PerlinSimplexNoise {
    noise_levels: Vec<Option<SimplexNoise>>,
    lowest_freq_input_factor: f64,
    lowest_freq_value_factor: f64,
}

impl PerlinSimplexNoise {
    /// Create from a random source and a list of octave levels.
    ///
    /// Matches vanilla's constructor: octave levels are sorted, and a
    /// `SimplexNoise` is created for each present level. Missing levels
    /// in the range are `None` (the random source is still consumed).
    #[must_use]
    pub fn new(random: &mut RandomSource, octaves: &[i32]) -> Self {
        let mut sorted = octaves.to_vec();
        sorted.sort_unstable();
        let first_octave = sorted[0];
        let last_octave = sorted[sorted.len() - 1];
        let range = (last_octave - first_octave) as usize;
        let length = range + 1;

        let mut noise_levels: Vec<Option<SimplexNoise>> = vec![None; length];

        for &octave in octaves {
            let index = (octave - first_octave) as usize;
            noise_levels[index] = Some(SimplexNoise::new(random));
        }

        Self {
            noise_levels,
            lowest_freq_input_factor: 2.0f64.powi(-last_octave),
            lowest_freq_value_factor: 2.0f64.powi(range as i32)
                / (2.0f64.powi(length as i32) - 1.0),
        }
    }

    /// Sample the 2D noise at the given coordinates.
    ///
    /// Matches vanilla's `getValue(x, z, false)` path (no Y offset).
    #[must_use]
    pub fn get_value(&self, x: f64, z: f64) -> f64 {
        let mut sum = 0.0;
        let mut frequency = self.lowest_freq_input_factor;
        let mut amplitude = self.lowest_freq_value_factor;

        for noise in &self.noise_levels {
            if let Some(n) = noise {
                sum += amplitude * n.get_value_2d(x * frequency, z * frequency);
            }
            frequency *= 2.0;
            amplitude /= 2.0;
        }

        sum
    }
}
