//! Improved Perlin noise implementation matching vanilla Minecraft's ImprovedNoise.java
//!
//! This is the base noise generator used by `PerlinNoise` for octave-based noise.

use std::simd::cmp::SimdPartialOrd;
use std::simd::num::SimdFloat;
use std::simd::{Select, StdFloat, f64x8, simd_swizzle};
use std::simd::{f64x2, f64x4, i32x4};

use crate::random::Random;
use glam::DVec3;
use steel_math::simd_utils::{concat_4x, splat_4x, transpose, transpose_2x};
use steel_math::{
    GRADIENT, GRADIENT_4, floor, grad_dot, grad_dot_4x, grad_dot_8x, lerp2, lerp2_3x, lerp3,
    lerp3_3x_simd, lerp3_4x, lerp3_simd, lerp3_simd_4x, smoothstep, smoothstep_3x, smoothstep_4x,
    smoothstep_derivative, smoothstep_derivative_3x,
};

/// Improved Perlin noise generator.
///
/// This implements the improved Perlin noise algorithm as used in Minecraft.
/// Each instance has a permutation table and offset values initialized from
/// a random source.
#[derive(Debug, Clone)]
pub struct ImprovedNoise {
    /// Permutation table (256 bytes)
    p: [u8; 256],
    // offset for the noise coordinates
    pub offset: DVec3,
}

impl ImprovedNoise {
    /// Creates a new `ImprovedNoise` from a random source.
    ///
    /// Initializes the permutation table using Fisher-Yates shuffle
    /// and sets random offsets.
    pub fn new<R: Random>(random: &mut R) -> Self {
        let xo = random.next_f64() * 256.0;
        let yo = random.next_f64() * 256.0;
        let zo = random.next_f64() * 256.0;
        let offset = DVec3::new(xo, yo, zo);

        let mut p = [0u8; 256];
        #[expect(
            clippy::needless_range_loop,
            reason = "index is used as the initial permutation value"
        )]
        for i in 0..256 {
            p[i] = i as u8;
        }

        // Fisher-Yates shuffle matching vanilla's implementation
        for i in 0..256 {
            let offset = random.next_i32_bounded((256 - i) as i32) as usize;
            p.swap(i, i + offset);
        }

        Self { p, offset }
    }

    /// Sample noise at the given coordinates.
    ///
    /// This is the standard 3D Perlin noise sampling without Y scaling.
    #[inline]
    #[must_use]
    pub fn noise(&self, pos: DVec3) -> f64 {
        let pos = pos + self.offset;
        let posf = pos.floor();
        let r = pos - posf;

        let pos = posf.as_ivec3();
        self.sample_and_lerp(pos.x, pos.y, pos.z, r.x, r.y, r.z, r.y)
    }

    /// Sample noise at the given coordinates, accumulating partial derivatives.
    ///
    /// Returns the noise value and adds the partial derivatives (dx, dy, dz)
    /// into `derivative_out`. Used by `BlendedNoise` for terrain generation.
    #[must_use]
    pub fn noise_with_derivative(&self, pos: DVec3, derivative_out: &mut [f64; 3]) -> f64 {
        let pos = pos + self.offset;
        let posf = pos.floor();
        let r = pos - posf;

        let pos = posf.as_ivec3();
        self.sample_with_derivative(pos.x, pos.y, pos.z, r, derivative_out)
    }

    /// Sample noise with Y scale and fudge parameters.
    ///
    /// The `y_scale` and `y_fudge` parameters are used for terrain generation
    /// where vertical noise needs special handling.
    ///
    /// # Arguments
    /// * `x`, `y`, `z` - The coordinates to sample
    /// * `y_scale` - Y scaling factor (0.0 to disable)
    /// * `y_fudge` - Y fudge factor for floor snapping
    #[must_use]
    #[expect(
        clippy::similar_names,
        reason = "yr_fudge and y_fudge match vanilla naming"
    )]
    pub fn noise_with_y_scale(&self, pos: DVec3, y_scale: f64, y_fudge: f64) -> f64 {
        let pos = pos + self.offset;
        let posf = pos.floor();
        let r = pos - posf;

        // Calculate Y fudge for terrain generation
        #[expect(
            clippy::if_not_else,
            reason = "matches vanilla's conditional structure"
        )]
        let yr_fudge = if y_scale != 0.0 {
            let fudge_limit = if y_fudge >= 0.0 && y_fudge < r.y {
                y_fudge
            } else {
                r.y
            };
            // SHIFT_UP_EPSILON = 1.0E-7F in Java (float literal promoted to double)
            (fudge_limit / y_scale + f64::from(1.0e-7_f32)).floor() * y_scale
        } else {
            0.0
        };
        let posf = posf.as_ivec3();
        self.sample_and_lerp(posf.x, posf.y, posf.z, r.x, r.y - yr_fudge, r.z, r.y)
    }

    /// Look up the permutation value at index x.
    #[inline]
    const fn p(&self, x: i32) -> usize {
        self.p[(x & 255) as usize] as usize
    }

    /// Sample noise at grid point and interpolate.
    #[expect(clippy::too_many_arguments, reason = "matches vanilla signature")]
    fn sample_and_lerp(
        &self,
        x: i32,
        y: i32,
        z: i32,
        xr: f64,
        yr: f64,
        zr: f64,
        yr_original: f64,
    ) -> f64 {
        // Get permutation indices for the 8 corners
        let x0 = self.p(x);
        let x1 = self.p(x + 1);
        let xy = [
            self.p(x0 as i32 + y),     // 0 0
            self.p(x0 as i32 + y + 1), // 0 1
            self.p(x1 as i32 + y),     // 1 0
            self.p(x1 as i32 + y + 1), // 1 1
        ];

        // Calculate gradient dot products at each corner
        let x4 = f64x4::from_array([xr, xr - 1., xr, xr - 1.]);
        let y4 = f64x4::from_array([yr, yr, yr - 1., yr - 1.]);
        let z4 = f64x4::splat(zr);

        let d0 = grad_dot_4x(xy.map(|x| self.p(x as i32 + z)), x4, y4, z4);

        let z4 = z4 - f64x4::splat(1.);
        let d1 = grad_dot_4x(xy.map(|x| self.p(x as i32 + z + 1)), x4, y4, z4);

        // Apply smoothstep interpolation
        let x_alpha = smoothstep(xr);
        let y_alpha = smoothstep(yr_original);
        let z_alpha = smoothstep(zr);

        lerp3_simd(x_alpha, y_alpha, z_alpha, d0, d1)
    }

    // -----------------------------------------------------------------------
    // SIMD: process 4 Y values sharing the same (x, z)
    // -----------------------------------------------------------------------

    /// Sample noise for 4 points that share the same x/z but differ in y.
    ///
    /// This is the SIMD counterpart of [`noise_with_y_scale`]. The x and z
    /// coordinate work (offset, floor, permutation) is done once and reused
    /// across all 4 lanes, while the y-dependent math is vectorized.
    #[must_use]
    pub fn noise_with_y_scale_4x(
        &self,
        x: f64,
        ys: f64x4,
        z: f64,
        y_scale: f64,
        y_fudges: f64x4,
    ) -> f64x4 {
        // Shared x/z offset and floor
        let xz = f64x2::from_array([x, z]);
        let xzo = f64x2::from_array([self.offset.x, self.offset.z]);
        let xz = xz + xzo;
        let xzf = xz.floor();
        let xzr = xz - xzf;

        // Per-lane y offset and floor
        let ys = ys + f64x4::splat(self.offset.y);
        let ys_floor = ys.floor();
        let yrs = ys - ys_floor;

        // Y fudge (per-lane)
        let yr_fudge = if y_scale == 0.0 {
            f64x4::splat(0.0)
        } else {
            let y_scale_v = f64x4::splat(y_scale);
            let zero = f64x4::splat(0.0);
            let mask = y_fudges.simd_ge(zero) & y_fudges.simd_lt(yrs);
            let fudge_limits = mask.select(y_fudges, yrs);
            let epsilon = f64x4::splat(f64::from(1.0e-7_f32));
            ((fudge_limits / y_scale_v) + epsilon).floor() * y_scale_v
        };

        let yrs_adjusted = yrs - yr_fudge;

        let xzf = xzf.cast::<i32>();
        self.sample_and_lerp_4x(xzf[0], xzf[1], xzr[0], xzr[1], ys_floor, yrs_adjusted, yrs)
    }

    /// Vectorized sample-and-lerp for 4 Y values sharing x/z grid position.
    ///
    /// `ys_floor` contains the floored y coordinates (as f64 for extraction),
    /// `yrs` are the adjusted fractional y parts, `yrs_original` are the
    /// un-fudged fractional parts (used for smoothstep).
    #[expect(
        clippy::too_many_arguments,
        reason = "mirrors scalar sample_and_lerp with 4x SIMD y-batching"
    )]
    fn sample_and_lerp_4x(
        &self,
        xf: i32,
        zf: i32,
        xr: f64,
        zr: f64,
        ys_floor: f64x4,
        yrs: f64x4,
        yrs_original: f64x4,
    ) -> f64x4 {
        // Shared x permutation lookups (2 instead of 2×4)
        let x0 = self.p(xf);
        let x1 = self.p(xf + 1);

        // Per-lane y-dependent permutation lookups
        let yf = [
            ys_floor[0] as i32,
            ys_floor[1] as i32,
            ys_floor[2] as i32,
            ys_floor[3] as i32,
        ];

        let mut h000 = [0usize; 4];
        let mut h100 = [0usize; 4];
        let mut h010 = [0usize; 4];
        let mut h110 = [0usize; 4];
        let mut h001 = [0usize; 4];
        let mut h101 = [0usize; 4];
        let mut h011 = [0usize; 4];
        let mut h111 = [0usize; 4];

        for i in 0..4 {
            let y = yf[i];
            let xy00 = self.p(x0 as i32 + y);
            let xy01 = self.p(x0 as i32 + y + 1);
            let xy10 = self.p(x1 as i32 + y);
            let xy11 = self.p(x1 as i32 + y + 1);
            h000[i] = self.p(xy00 as i32 + zf);
            h100[i] = self.p(xy10 as i32 + zf);
            h010[i] = self.p(xy01 as i32 + zf);
            h110[i] = self.p(xy11 as i32 + zf);
            h001[i] = self.p(xy00 as i32 + zf + 1);
            h101[i] = self.p(xy10 as i32 + zf + 1);
            h011[i] = self.p(xy01 as i32 + zf + 1);
            h111[i] = self.p(xy11 as i32 + zf + 1);
        }

        let xr_v0 = f64x4::splat(xr);
        let xr_v1 = f64x4::splat(xr - 1.0);
        let zr_v0 = f64x4::splat(zr);
        let zr_v1 = f64x4::splat(zr - 1.0);

        let yr_v0 = yrs;
        let yr_v1 = yrs - f64x4::splat(1.0);

        // Pair the hashes exactly as the scalar code does, which has index 1 and 2 swapped
        let d000 = grad_dot_4x(h000, xr_v0, yr_v0, zr_v0);
        let d100 = grad_dot_4x(h010, xr_v1, yr_v0, zr_v0);
        let d010 = grad_dot_4x(h100, xr_v0, yr_v1, zr_v0);
        let d110 = grad_dot_4x(h110, xr_v1, yr_v1, zr_v0);
        let d001 = grad_dot_4x(h001, xr_v0, yr_v0, zr_v1);
        let d101 = grad_dot_4x(h011, xr_v1, yr_v0, zr_v1);
        let d011 = grad_dot_4x(h101, xr_v0, yr_v1, zr_v1);
        let d111 = grad_dot_4x(h111, xr_v1, yr_v1, zr_v1);

        // Smoothstep — x and z are shared across lanes
        let x_alpha = f64x4::splat(smoothstep(xr));
        let y_alpha = smoothstep_4x(yrs_original);
        let z_alpha = f64x4::splat(smoothstep(zr));

        lerp3_4x(
            x_alpha, y_alpha, z_alpha, d000, d100, d010, d110, d001, d101, d011, d111,
        )
    }

    /// Sample noise at grid point, interpolate, and accumulate derivatives.
    #[expect(clippy::too_many_arguments, reason = "matches vanilla signature")]
    fn sample_with_derivative(
        &self,
        x: i32,
        y: i32,
        z: i32,
        r: DVec3,
        derivative_out: &mut [f64; 3],
    ) -> f64 {
        let x0 = self.p(x);
        let x1 = self.p(x + 1);
        let xy = [
            self.p(x0 as i32 + y),     // 0 0
            self.p(x0 as i32 + y + 1), // 0 1
            self.p(x1 as i32 + y),     // 1 0
            self.p(x1 as i32 + y + 1), // 1 1
        ];

        // Get hashes and gradient vectors for all 8 corners
        let h0 = xy.map(|xy| self.p(xy as i32 + z));
        let h1 = xy.map(|xy| self.p(xy as i32 + z + 1));

        let g000 = f64x4::from_array(GRADIENT_4[h0[0b00] & 15]);
        let g100 = f64x4::from_array(GRADIENT_4[h0[0b10] & 15]);
        let g010 = f64x4::from_array(GRADIENT_4[h0[0b01] & 15]);
        let g110 = f64x4::from_array(GRADIENT_4[h0[0b11] & 15]);
        let g001 = f64x4::from_array(GRADIENT_4[h1[0b00] & 15]);
        let g101 = f64x4::from_array(GRADIENT_4[h1[0b10] & 15]);
        let g011 = f64x4::from_array(GRADIENT_4[h1[0b01] & 15]);
        let g111 = f64x4::from_array(GRADIENT_4[h1[0b11] & 15]);

        let (g00, g01, g02, _g03) = transpose(g000, g100, g010, g110);
        let (g10, g11, g12, _g13) = transpose(g001, g101, g011, g111);

        let x4 = f64x4::from_array([r.x, r.x - 1., r.x, r.x - 1.]);
        let y4 = f64x4::from_array([r.y, r.y, r.y - 1., r.y - 1.]);
        let z4 = f64x4::splat(r.z);

        // Gradient dot products at each corner
        let d0 = grad_dot_4x(h0, x4, y4, z4);

        let z4 = z4 - f64x4::splat(1.);
        let d1 = grad_dot_4x(h1, x4, y4, z4);

        let alpha = smoothstep_3x(r);

        // Interpolate gradient components for direct derivative contribution
        let d1_v = lerp3_3x_simd(alpha.x, alpha.y, alpha.z, g00, g10, g01, g11, g02, g12);

        // Smoothstep correction terms via differences
        let a1 = DVec3::new(alpha.y, alpha.z, alpha.x);
        let a2 = DVec3::new(alpha.z, alpha.x, alpha.y);

        let ax1 = simd_swizzle!(d0, d1, [0b10, 0b11, 0b10 + 4, 0b11 + 4]);
        let ax0 = simd_swizzle!(d0, d1, [0b00, 0b01, 0b00 + 4, 0b01 + 4]);
        let ax = ax1 - ax0;

        let bx1 = simd_swizzle!(d0, d1, [0b01, 0b01 + 4, 0b11, 0b11 + 4]);
        let bx0 = simd_swizzle!(d0, d1, [0b00, 0b00 + 4, 0b10, 0b10 + 4]);
        let bx = bx1 - bx0;

        let cx1 = d1;
        let cx0 = d0;
        let cx = cx1 - cx0;

        let x00 = DVec3::new(bx[0b00], ax[0b00], cx[0b00]);
        let x10 = DVec3::new(bx[0b10], ax[0b10], cx[0b10]);
        let x01 = DVec3::new(bx[0b01], ax[0b01], cx[0b01]);
        let x11 = DVec3::new(bx[0b11], ax[0b11], cx[0b11]);

        let d2_v = lerp2_3x(a1, a2, x00, x10, x01, x11);

        let sd = smoothstep_derivative_3x(r);

        // Accumulate derivatives (vanilla uses +=)
        let mut d = DVec3::from_array(*derivative_out);
        d += d1_v + sd * d2_v;
        derivative_out[0] = d.x;
        derivative_out[1] = d.y;
        derivative_out[2] = d.z;

        lerp3_simd(alpha.x, alpha.y, alpha.z, d0, d1)
    }
}

// ---------------------------------------------------------------------------
// SIMD helpers (4-wide f64)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random::xoroshiro::Xoroshiro;

    #[test]
    fn test_noise_with_y_scale_4x_matches_scalar() {
        let mut rng = Xoroshiro::from_seed(42);
        let noise = ImprovedNoise::new(&mut rng);

        // Test various coordinate combinations
        let test_x_zs: &[(f64, f64)] = &[
            (0.0, 0.0),
            (1.5, 3.7),
            (-5.2, 100.3),
            (0.001, -0.001),
            (1000.0, -500.0),
        ];
        let test_ys: &[[f64; 4]] = &[
            [0.0, 1.0, 2.0, 3.0],
            [64.0, 64.5, 65.0, 65.5],
            [-5.0, -2.5, 0.0, 2.5],
            [0.25, 0.5, 0.75, 1.0],
            [-100.0, -50.0, 50.0, 100.0],
        ];
        let y_scales = [0.0, 1.0, 8.0];

        for &(x, z) in test_x_zs {
            for ys in test_ys {
                for &y_scale in &y_scales {
                    let y_fudges: [f64; 4] = if y_scale == 0.0 {
                        [0.0; 4]
                    } else {
                        *ys // use ys as fudge values (matching BlendedNoise usage)
                    };

                    let simd_result = noise.noise_with_y_scale_4x(
                        x,
                        f64x4::from_array(*ys),
                        z,
                        y_scale,
                        f64x4::from_array(y_fudges),
                    );

                    for i in 0..4 {
                        let scalar =
                            noise.noise_with_y_scale(DVec3::new(x, ys[i], z), y_scale, y_fudges[i]);
                        let simd_val = simd_result[i];
                        assert!(
                            (scalar - simd_val).abs() < 1e-14,
                            "Mismatch at x={x}, y={}, z={z}, y_scale={y_scale}: \
                             scalar={scalar}, simd={simd_val}, diff={}",
                            ys[i],
                            (scalar - simd_val).abs(),
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_improved_noise_deterministic() {
        let mut rng1 = Xoroshiro::from_seed(12345);
        let mut rng2 = Xoroshiro::from_seed(12345);

        let noise1 = ImprovedNoise::new(&mut rng1);
        let noise2 = ImprovedNoise::new(&mut rng2);

        // Same seed should produce same noise
        #[expect(
            clippy::float_cmp,
            reason = "determinism test: identical seeds must produce bit-identical offsets"
        )]
        {
            assert_eq!(noise1.offset, noise2.offset);
        }
        assert_eq!(noise1.p, noise2.p);

        // Same coordinates should produce same values
        let v1 = noise1.noise(DVec3::new(100.0, 64.0, 100.0));
        let v2 = noise2.noise(DVec3::new(100.0, 64.0, 100.0));
        assert!((v1 - v2).abs() < 1e-15);
    }

    #[test]
    fn test_noise_matches_zero_y_scale_path() {
        let mut rng = Xoroshiro::from_seed(42);
        let noise = ImprovedNoise::new(&mut rng);

        for pos in [
            DVec3::new(0.0, 0.0, 0.0),
            DVec3::new(1.25, 64.5, -30.75),
            DVec3::new(-1000.0, -20.25, 4096.5),
        ] {
            assert!((noise.noise(pos) - noise.noise_with_y_scale(pos, 0.0, 0.0)).abs() < 1e-15);
        }
    }

    #[test]
    fn test_improved_noise_range() {
        let mut rng = Xoroshiro::from_seed(42);
        let noise = ImprovedNoise::new(&mut rng);

        // Sample at various points and verify output is in reasonable range
        for x in -10..10 {
            for z in -10..10 {
                let v = noise.noise(DVec3::new(f64::from(x) * 10.0, 64.0, f64::from(z) * 10.0));
                // Perlin noise should be in [-1, 1] range roughly
                assert!(
                    (-1.5..=1.5).contains(&v),
                    "Noise value {v} at ({x}, {z}) out of expected range",
                );
            }
        }
    }

    #[test]
    fn test_improved_noise_spatial_variation() {
        let mut rng = Xoroshiro::from_seed(42);
        let noise = ImprovedNoise::new(&mut rng);

        // Noise at different positions should generally be different
        let v1 = noise.noise(DVec3::new(0.0, 0.0, 0.0));
        let v2 = noise.noise(DVec3::new(100.0, 0.0, 0.0));
        let v3 = noise.noise(DVec3::new(0.0, 100.0, 0.0));
        let v4 = noise.noise(DVec3::new(0.0, 0.0, 100.0));

        // At least some should be different (statistically almost certain)
        #[expect(
            clippy::float_cmp,
            reason = "intentional exact equality check to detect degenerate constant noise"
        )]
        let all_same = v1 == v2 && v2 == v3 && v3 == v4;
        assert!(!all_same, "All noise values are the same - unexpected");
    }

    #[test]
    fn test_noise_with_derivative_matches_noise() {
        let mut rng = Xoroshiro::from_seed(42);
        let noise = ImprovedNoise::new(&mut rng);

        // noise_with_derivative should return the same value as noise()
        // (when no y_scale/y_fudge is used)
        for &pos in &[
            DVec3::new(0.0, 0.0, 0.0),
            DVec3::new(1.5, 2.3, 3.7),
            DVec3::new(-5.2, 64.0, 100.3),
            DVec3::new(0.25, 0.25, 0.25),
        ] {
            let v1 = noise.noise(pos);
            let mut deriv = [0.0; 3];
            let v2 = noise.noise_with_derivative(pos, &mut deriv);
            assert!(
                (v1 - v2).abs() < 1e-12,
                "Value mismatch at ({pos}): {v1} vs {v2}",
            );
        }
    }

    #[test]
    fn test_noise_with_derivative_produces_derivatives() {
        let mut rng = Xoroshiro::from_seed(42);
        let noise = ImprovedNoise::new(&mut rng);

        let mut deriv = [0.0; 3];
        let _ = noise.noise_with_derivative(DVec3::new(1.5, 2.3, 3.7), &mut deriv);

        // At a non-grid point, at least some derivatives should be nonzero
        let any_nonzero = deriv.iter().any(|&d| d.abs() > 1e-15);
        assert!(any_nonzero, "All derivatives are zero: {deriv:?}");
    }

    #[test]
    fn test_noise_with_derivative_accumulates() {
        let mut rng = Xoroshiro::from_seed(42);
        let noise = ImprovedNoise::new(&mut rng);

        // First call
        let mut deriv = [0.0; 3];
        let _ = noise.noise_with_derivative(DVec3::new(1.5, 2.3, 3.7), &mut deriv);
        let first = deriv;

        // Second call should accumulate (+=)
        let _ = noise.noise_with_derivative(DVec3::new(4.1, 5.2, 6.3), &mut deriv);
        let mut deriv2 = [0.0; 3];
        let _ = noise.noise_with_derivative(DVec3::new(4.1, 5.2, 6.3), &mut deriv2);

        for i in 0..3 {
            let expected = first[i] + deriv2[i];
            assert!(
                (deriv[i] - expected).abs() < 1e-12,
                "Derivative[{i}] not accumulated: {0} vs expected {expected}",
                deriv[i],
            );
        }
    }
}
