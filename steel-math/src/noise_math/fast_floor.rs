use std::simd::{
    Select, Simd,
    cmp::SimdPartialOrd,
    f64x2, i32x2,
    num::{SimdFloat, SimdInt},
};

use glam::{DVec3, IVec3};

/// Floor function that matches Java behavior.
///
/// In Java, `(int)v` truncates toward zero, but we need floor behavior.
/// For negative values, we need to subtract 1 if there's a fractional part.
///
/// Fast Floor from Stefan Gustavson's in "Simplex Noise Demystified" 2005 paper
///
/// Java reference: `Mth.floor(double)`
#[expect(clippy::inline_always, reason = "hot-path noise primitive")]
#[inline(always)]
#[must_use]
pub fn fast_floor(v: f64) -> i32 {
    let i = v as i32;
    if v < f64::from(i) { i - 1 } else { i }
}

#[expect(clippy::inline_always, reason = "hot-path noise primitive")]
#[inline(always)]
#[must_use]
pub fn fast_floor_2x(v: f64x2) -> i32x2 {
    let i = v.cast::<i32>();
    let b = v.simd_lt(i.cast::<f64>());
    b.select(i - Simd::splat(1), i)
}

#[expect(clippy::inline_always, reason = "hot-path noise primitive")]
#[inline(always)]
#[must_use]
pub fn fast_floor_3x(v: DVec3) -> IVec3 {
    let i = v.as_ivec3();
    let b = v.cmplt(i.as_dvec3());
    IVec3::select(b, i - 1, i)
}

/// Long floor function matching Java behavior.
///
/// Java reference: `Mth.lfloor(double)`
#[inline]
#[must_use]
pub fn fast_lfloor(v: f64) -> i64 {
    let i = v as i64;
    if v < i as f64 { i - 1 } else { i }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_floor() {
        assert_eq!(fast_floor(1.5), 1);
        assert_eq!(fast_floor(1.0), 1);
        assert_eq!(fast_floor(0.5), 0);
        assert_eq!(fast_floor(0.0), 0);
        assert_eq!(fast_floor(-0.5), -1);
        assert_eq!(fast_floor(-1.0), -1);
        assert_eq!(fast_floor(-1.5), -2);
    }
}
