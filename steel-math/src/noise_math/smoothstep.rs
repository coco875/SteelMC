use core::simd::f64x4;
use std::{
    ops,
    simd::{Simd, SimdCast, SimdElement, num::SimdFloat},
};

use glam::DVec3;

/// Smoothstep - quintic Hermite interpolation (NOT cubic!)
///
/// Formula: 6x^5 - 15x^4 + 10x^3
///
/// This is the standard smoothstep used in Perlin noise for smooth transitions.
/// Java reference: `Mth.smoothstep(double)`
#[expect(clippy::inline_always, reason = "hot-path noise primitive")]
#[inline(always)]
#[must_use]
pub fn smoothstep(x: f64) -> f64 {
    x * x * x * (x * (x * 6.0 - 15.0) + 10.0)
}

#[expect(clippy::inline_always, reason = "hot-path noise primitive")]
#[inline(always)]
#[must_use]
pub fn smoothstep_simd<F, const N: usize>(x: Simd<F, N>) -> Simd<F, N>
where
    F: SimdElement + SimdCast,
    Simd<F, N>: ops::Mul<Output = Simd<F, N>>
        + ops::Sub<Output = Simd<F, N>>
        + ops::Add<Output = Simd<F, N>>,
{
    x * x
        * x
        * (x * (x * Simd::splat(6.0).cast::<F>() - Simd::splat(15.0).cast::<F>())
            + Simd::splat(10.0).cast::<F>())
}

/// Quintic Hermite interpolation for 3-dimensional double vectors (`DVec3`).
#[expect(clippy::inline_always, reason = "hot-path noise primitive")]
#[inline(always)]
#[must_use]
pub fn smoothstep_3x(x: DVec3) -> DVec3 {
    x * x * x * (x * (x * 6.0 - 15.0) + 10.0)
}

/// Smoothstep derivative for noise with derivatives.
///
/// Formula: 30x^2(x-1)^2
///
/// Java reference: `Mth.smoothstepDerivative(double)`
#[inline]
#[must_use]
pub fn smoothstep_derivative(x: f64) -> f64 {
    30.0 * x * x * (x - 1.0) * (x - 1.0)
}

/// Smoothstep derivative for 3-dimensional double vectors (`DVec3`).
#[inline]
#[must_use]
pub fn smoothstep_derivative_3x(x: DVec3) -> DVec3 {
    30.0 * x * x * (x - 1.0) * (x - 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_smoothstep() {
        // At boundaries
        assert!((smoothstep(0.0) - 0.0).abs() < 1e-10);
        assert!((smoothstep(1.0) - 1.0).abs() < 1e-10);
        // At midpoint
        assert!((smoothstep(0.5) - 0.5).abs() < 1e-10);
    }
}
