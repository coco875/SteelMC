use core::simd::f64x4;
use std::simd::{f64x2, f64x8, simd_swizzle};

use glam::DVec3;

use crate::simd_utils::splat_4x;

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

/// Trilinear interpolation using 4-lane SIMD vectors (`f64x4`) for intermediate calculations.
#[expect(clippy::inline_always, reason = "hot-path noise primitive")]
#[inline(always)]
#[must_use]
pub fn lerp3_simd(a1: f64, a2: f64, a3: f64, x0: f64x4, x1: f64x4) -> f64 {
    let a1 = f64x4::splat(a1);
    let a2 = f64x2::splat(a2);

    let res = lerp2_simd(a1, a2, x0, x1);
    lerp(a3, res[0], res[1])
}

/// Trilinear interpolation for three separate coordinate dimensions simultaneously using 4-lane SIMD.
#[expect(clippy::inline_always, reason = "hot-path noise primitive")]
#[inline(always)]
#[must_use]
pub fn lerp3_3x_simd(
    a1: f64,
    a2: f64,
    a3: f64,
    ax0: f64x4,
    ax1: f64x4,
    bx0: f64x4,
    bx1: f64x4,
    cx0: f64x4,
    cx1: f64x4,
) -> DVec3 {
    let a1 = f64x4::splat(a1);
    let a2 = f64x2::splat(a2);
    let res_a = lerp2_simd(a1, a2, ax0, ax1);
    let res_b = lerp2_simd(a1, a2, bx0, bx1);
    let res_c = lerp2_simd(a1, a2, cx0, cx1);
    let a = DVec3::new(res_a[0], res_b[0], res_c[0]);
    let b = DVec3::new(res_a[1], res_b[1], res_c[1]);
    lerp_3x(DVec3::splat(a3), a, b)
}

fn lerp2_simd(a1: f64x4, a2: f64x2, x0: f64x4, x1: f64x4) -> f64x2 {
    let (a, b) = x0.deinterleave(x1);
    let res = lerp_4x(a1, a, b);

    let a = simd_swizzle!(res, [0, 2]); // a * (x0 x1)
    let b = simd_swizzle!(res, [1, 3]); // b * (x0 x1)

    lerp_2x(a2, a, b)
}

/// Linear interpolation for 2-lane SIMD vectors (`f64x2`).
#[expect(clippy::inline_always, reason = "hot-path noise primitive")]
#[inline(always)]
#[must_use]
pub fn lerp_2x(alpha: f64x2, a: f64x2, b: f64x2) -> f64x2 {
    a + alpha * (b - a)
}

/// Linear interpolation for 3-dimensional double vectors (`DVec3`).
#[expect(clippy::inline_always, reason = "hot-path noise primitive")]
#[inline(always)]
#[must_use]
pub fn lerp_3x(alpha: DVec3, a: DVec3, b: DVec3) -> DVec3 {
    a + alpha * (b - a)
}

/// Linear interpolation for 4 lanes. see lerp.
#[inline]
#[must_use]
pub fn lerp_4x(alpha: f64x4, a: f64x4, b: f64x4) -> f64x4 {
    a + alpha * (b - a)
}

#[inline]
#[must_use]
pub fn lerp_8x(alpha: f64x8, a: f64x8, b: f64x8) -> f64x8 {
    a + alpha * (b - a)
}

/// Bilinear interpolation for 4 lanes. see lerp2.
#[inline]
#[must_use]
pub fn lerp2_4x(a1: f64x4, a2: f64x4, x00: f64x4, x10: f64x4, x01: f64x4, x11: f64x4) -> f64x4 {
    lerp_4x(a2, lerp_4x(a1, x00, x10), lerp_4x(a1, x01, x11))
}

#[inline]
#[must_use]
pub fn lerp2_8x(a1: f64x8, a2: f64x8, x00: f64x8, x10: f64x8, x01: f64x8, x11: f64x8) -> f64x8 {
    lerp_8x(a2, lerp_8x(a1, x00, x10), lerp_8x(a1, x01, x11))
}

/// Trilinear interpolation for 4 lanes. see lerp3.
#[inline]
#[expect(clippy::too_many_arguments, reason = "mirrors lerp3 with SIMD vectors")]
#[must_use]
pub fn lerp3_4x(
    a1: f64x4,
    a2: f64x4,
    a3: f64x4,
    x000: f64x4,
    x100: f64x4,
    x010: f64x4,
    x110: f64x4,
    x001: f64x4,
    x101: f64x4,
    x011: f64x4,
    x111: f64x4,
) -> f64x4 {
    lerp_4x(
        a3,
        lerp2_4x(a1, a2, x000, x100, x010, x110),
        lerp2_4x(a1, a2, x001, x101, x011, x111),
    )
}

/// Trilinear interpolation for 4 lanes, optimized using 8-lane SIMD vectors (`f64x8`) for bilinear steps.
#[inline]
#[expect(clippy::too_many_arguments, reason = "mirrors lerp3 with SIMD vectors")]
#[must_use]
pub fn lerp3_simd_4x(
    a1: f64x4,
    a2: f64x4,
    a3: f64x4,
    x00: f64x8,
    x10: f64x8,
    x01: f64x8,
    x11: f64x8,
) -> f64x4 {
    let a1 = splat_4x(a1);
    let a2 = splat_4x(a2);
    let res = lerp2_8x(a1, a2, x00, x10, x01, x11);
    let lower_half: f64x4 = simd_swizzle!(res, [0, 1, 2, 3]);
    let upper_half = simd_swizzle!(res, [4, 5, 6, 7]);
    lerp_4x(a3, lower_half, upper_half)
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

    #[test]
    fn test_lerp3_simd() {
        let a1: f64 = 1.0;
        let a2: f64 = 2.0;
        let a3: f64 = 3.0;
        let x000: f64 = 4.0;
        let x100: f64 = 5.0;
        let x010: f64 = 6.0;
        let x110: f64 = 7.0;
        let x001: f64 = 8.0;
        let x101: f64 = 9.0;
        let x011: f64 = 10.0;
        let x111: f64 = 11.0;
        assert_eq!(
            lerp3(a1, a2, a3, x000, x100, x010, x110, x001, x101, x011, x111),
            lerp3_simd(
                a1,
                a2,
                a3,
                f64x4::from_array([x000, x100, x010, x110]),
                f64x4::from_array([x001, x101, x011, x111])
            )
        );
    }
}
