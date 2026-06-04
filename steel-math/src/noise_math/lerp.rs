use std::{
    ops,
    simd::{Simd, SimdElement},
};

use glam::DVec3;

/// Linear interpolation.
///
/// Formula: a + alpha * (b - a)
///
/// Java reference: `Mth.lerp(double, double, double)`
#[expect(clippy::inline_always, reason = "hot-path noise primitive")]
#[inline(always)]
#[must_use]
pub fn lerp(alpha: f64, a: f64, b: f64) -> f64 {
    a + alpha * (b - a)
}

/// SIMD linear interpolation.
#[expect(clippy::inline_always, reason = "hot-path noise primitive")]
#[inline(always)]
#[must_use]
pub fn lerp_simd<F, const N: usize>(alpha: Simd<F, N>, a: Simd<F, N>, b: Simd<F, N>) -> Simd<F, N>
where
    F: SimdElement,
    Simd<F, N>: ops::Mul<Output = Simd<F, N>>
        + ops::Add<Output = Simd<F, N>>
        + ops::Sub<Output = Simd<F, N>>,
{
    a + alpha * (b - a)
}

/// Bilinear interpolation.
///
/// Interpolates between 4 values in a 2D grid.
///
/// Java reference: `Mth.lerp2(double, double, double, double, double, double)`
#[expect(clippy::inline_always, reason = "hot-path noise primitive")]
#[inline(always)]
#[must_use]
pub fn lerp2(a1: f64, a2: f64, x00: f64, x10: f64, x01: f64, x11: f64) -> f64 {
    lerp(a2, lerp(a1, x00, x10), lerp(a1, x01, x11))
}

/// SIMD bilinear interpolation.
#[expect(clippy::inline_always, reason = "hot-path noise primitive")]
#[inline(always)]
#[must_use]
pub fn lerp2_simd<F, const N: usize>(
    a1: Simd<F, N>,
    a2: Simd<F, N>,
    x00: Simd<F, N>,
    x10: Simd<F, N>,
    x01: Simd<F, N>,
    x11: Simd<F, N>,
) -> Simd<F, N>
where
    F: SimdElement,
    Simd<F, N>: ops::Mul<Output = Simd<F, N>>
        + ops::Add<Output = Simd<F, N>>
        + ops::Sub<Output = Simd<F, N>>,
{
    lerp_simd(a2, lerp_simd(a1, x00, x10), lerp_simd(a1, x01, x11))
}

/// Bilinear interpolation for 3-dimensional double vectors (`DVec3`).
#[expect(clippy::inline_always, reason = "hot-path noise primitive")]
#[inline(always)]
#[must_use]
pub fn lerp2_3x(a1: DVec3, a2: DVec3, x00: DVec3, x10: DVec3, x01: DVec3, x11: DVec3) -> DVec3 {
    lerp_3x(a2, lerp_3x(a1, x00, x10), lerp_3x(a1, x01, x11))
}

/// Trilinear interpolation.
///
/// Interpolates between 8 values in a 3D grid.
///
/// Java reference: `Mth.lerp3(...)`
#[expect(clippy::inline_always, reason = "hot-path noise primitive")]
#[inline(always)]
#[must_use]
#[expect(
    clippy::too_many_arguments,
    reason = "matches vanilla's Mth.lerp3 signature with 8 grid corner values"
)]
pub fn lerp3(
    a1: f64,
    a2: f64,
    a3: f64,
    x000: f64,
    x100: f64,
    x010: f64,
    x110: f64,
    x001: f64,
    x101: f64,
    x011: f64,
    x111: f64,
) -> f64 {
    lerp(
        a3,
        lerp2(a1, a2, x000, x100, x010, x110),
        lerp2(a1, a2, x001, x101, x011, x111),
    )
}

/// SIMD trilinear interpolation.
#[expect(clippy::inline_always, reason = "hot-path noise primitive")]
#[inline(always)]
#[must_use]
#[expect(
    clippy::too_many_arguments,
    reason = "matches vanilla's Mth.lerp3 signature with 8 grid corner values"
)]
pub fn lerp3_simd<F, const N: usize>(
    a1: Simd<F, N>,
    a2: Simd<F, N>,
    a3: Simd<F, N>,
    x000: Simd<F, N>,
    x100: Simd<F, N>,
    x010: Simd<F, N>,
    x110: Simd<F, N>,
    x001: Simd<F, N>,
    x101: Simd<F, N>,
    x011: Simd<F, N>,
    x111: Simd<F, N>,
) -> Simd<F, N>
where
    F: SimdElement,
    Simd<F, N>: ops::Mul<Output = Simd<F, N>>
        + ops::Add<Output = Simd<F, N>>
        + ops::Sub<Output = Simd<F, N>>,
{
    lerp_simd(
        a3,
        lerp2_simd(a1, a2, x000, x100, x010, x110),
        lerp2_simd(a1, a2, x001, x101, x011, x111),
    )
}

/// Trilinear interpolation for three separate coordinate dimensions simultaneously using 4-lane SIMD.
#[expect(clippy::inline_always, reason = "hot-path noise primitive")]
#[inline(always)]
#[must_use]
#[expect(
    clippy::too_many_arguments,
    reason = "matches vanilla's Mth.lerp3 signature with 8 grid corner values"
)]
pub fn lerp3_3x(
    a1: f64,
    a2: f64,
    a3: f64,
    x000: DVec3,
    x100: DVec3,
    x010: DVec3,
    x110: DVec3,
    x001: DVec3,
    x101: DVec3,
    x011: DVec3,
    x111: DVec3,
) -> DVec3 {
    let a1 = DVec3::splat(a1);
    let a2 = DVec3::splat(a2);
    let a3 = DVec3::splat(a3);
    lerp_3x(
        a3,
        lerp2_3x(a1, a2, x000, x100, x010, x110),
        lerp2_3x(a1, a2, x001, x101, x011, x111),
    )
}

/// Linear interpolation for 3-dimensional double vectors (`DVec3`).
#[expect(clippy::inline_always, reason = "hot-path noise primitive")]
#[inline(always)]
#[must_use]
pub fn lerp_3x(alpha: DVec3, a: DVec3, b: DVec3) -> DVec3 {
    a + alpha * (b - a)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lerp() {
        assert!((lerp(0.0, 10.0, 20.0) - 10.0).abs() < 1e-10);
        assert!((lerp(1.0, 10.0, 20.0) - 20.0).abs() < 1e-10);
        assert!((lerp(0.5, 10.0, 20.0) - 15.0).abs() < 1e-10);
    }
}
