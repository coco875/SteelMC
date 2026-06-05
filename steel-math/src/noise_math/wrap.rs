use std::{
    ops,
    simd::{Simd, SimdCast, SimdElement, StdFloat, num::SimdFloat},
};

/// Round-off constant for coordinate wrapping to prevent precision loss.
/// This is 2^25 = 33554432.
const ROUND_OFF: f64 = 33_554_432.0;
const HALF_ROUND_OFF: f64 = ROUND_OFF / 2.0;

/// Wrap a coordinate to prevent precision loss at large values.
///
/// This wraps the coordinate to the range `[-ROUND_OFF/2, ROUND_OFF/2]` to
/// maintain numerical precision for coordinates far from the origin.
///
/// Public because `BlendedNoise` calls this directly on per-octave coordinates.
#[inline]
#[must_use]
pub fn wrap(x: f64) -> f64 {
    if (-HALF_ROUND_OFF..HALF_ROUND_OFF).contains(&x) {
        return x;
    }

    x - (x / ROUND_OFF + 0.5).floor() * ROUND_OFF
}

/// Wrap 4 coordinates to prevent precision loss (SIMD version of [`wrap`]).
#[inline]
#[must_use]
pub fn wrap_simd<F, const N: usize>(x: Simd<F, N>) -> Simd<F, N>
where
    F: SimdElement + SimdCast,
    Simd<F, N>: ops::Div<Output = Simd<F, N>>
        + ops::Add<Output = Simd<F, N>>
        + ops::Mul<Output = Simd<F, N>>
        + ops::Sub<Output = Simd<F, N>>
        + StdFloat,
{
    let round_off = Simd::splat(ROUND_OFF).cast::<F>();
    x - (x / round_off + Simd::splat(0.5).cast::<F>()).floor() * round_off
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_wrap() {
        fn wrap_reference(x: f64) -> f64 {
            x - (x / ROUND_OFF + 0.5).floor() * ROUND_OFF
        }

        // Small values should be unchanged
        assert!((wrap(100.0) - 100.0).abs() < 1e-10);
        assert!((wrap(-100.0) - (-100.0)).abs() < 1e-10);

        // Very large values should be wrapped
        let large = 100_000_000.0;
        let wrapped = wrap(large);
        assert!(wrapped.abs() < ROUND_OFF);

        for x in [
            -HALF_ROUND_OFF,
            -HALF_ROUND_OFF + 1.0,
            0.0,
            HALF_ROUND_OFF - 1.0,
            HALF_ROUND_OFF,
            ROUND_OFF,
            -ROUND_OFF,
            100_000_000.0,
            -100_000_000.0,
        ] {
            assert!((wrap(x) - wrap_reference(x)).abs() < 1e-15);
        }
    }
}
