use std::{
    ops,
    simd::{
        Mask, Simd, SimdCast, SimdElement, StdFloat,
        cmp::{SimdPartialEq, SimdPartialOrd},
        f64x4, f64x8,
        num::SimdFloat,
        simd_swizzle,
    },
};

/// Transposes a 4x4 matrix of 64-bit floats represented by four SIMD vectors (`f64x4`).
#[inline]
#[must_use]
pub fn transpose(
    r0: f64x4, // [a0, a1, a2, a3]
    r1: f64x4, // [b0, b1, b2, b3]
    r2: f64x4, // [c0, c1, c2, c3]
    r3: f64x4, // [d0, d1, d2, d3]
) -> (f64x4, f64x4, f64x4, f64x4) {
    let (t0, t1) = r0.deinterleave(r1); // t0 = [a0, a2, b0, b2], t1 = [a1, a3, b1, b3]
    let (t2, t3) = r2.deinterleave(r3); // t2 = [c0, c2, d0, d2], t3 = [c1, c3, d1, d3]

    let (col0, col2) = t0.deinterleave(t2); // col0 = [a0, b0, c0, d0], col2 = [a2, b2, c2, d2]
    let (col1, col3) = t1.deinterleave(t3); // col1 = [a1, b1, c1, d1], col3 = [a3, b3, c3, d3]

    (col0, col1, col2, col3)
}

/// Transposes two 4x4 matrices of 64-bit floats simultaneously using four 8-lane SIMD vectors (`f64x8`).
#[inline]
#[must_use]
pub fn transpose_2x(
    r0: f64x8, // [a0, a1, a2, a3] [e0, e1, e2, e3]
    r1: f64x8, // [b0, b1, b2, b3] [f0, f1, f2, f3]
    r2: f64x8, // [c0, c1, c2, c3] [g0, g1, g2, g3]
    r3: f64x8, // [d0, d1, d2, d3] [h0, h1, h2, h3]
) -> (f64x8, f64x8, f64x8, f64x8) {
    let r01_lo = simd_swizzle!(r0, r1, [0, 8, 1, 9, 4, 12, 5, 13]);
    // Result: [a0, b0, a1, b1,  e0, f0, e1, f1]
    let r01_hi = simd_swizzle!(r0, r1, [2, 10, 3, 11, 6, 14, 7, 15]);
    // Result: [a2, b2, a3, b3,  e2, f2, e3, f3]

    let r23_lo = simd_swizzle!(r2, r3, [0, 8, 1, 9, 4, 12, 5, 13]);
    // Result: [c0, d0, c1, d1,  g0, h0, g1, h1]    let (col1, col3) = t1.deinterleave(t3); // col1 = [a1, b1, c1, d1], col3 = [a3, b3, c3, d3]
    let r23_hi = simd_swizzle!(r2, r3, [2, 10, 3, 11, 6, 14, 7, 15]);
    // Result: [c2, d2, c3, d3,  g2, h2, g3, h3]

    let col0 = simd_swizzle!(r01_lo, r23_lo, [0, 1, 8, 9, 4, 5, 12, 13]);
    // Result: [a0, b0, c0, d0,  e0, f0, g0, h0]
    let col1 = simd_swizzle!(r01_lo, r23_lo, [2, 3, 10, 11, 6, 7, 14, 15]);
    // Result: [a1, b1, c1, d1,  e1, f1, g1, h1]

    let col2 = simd_swizzle!(r01_hi, r23_hi, [0, 1, 8, 9, 4, 5, 12, 13]);
    // Result: [a2, b2, c2, d2,  e2, f2, g2, h2]
    let col3 = simd_swizzle!(r01_hi, r23_hi, [2, 3, 10, 11, 6, 7, 14, 15]);
    // Result: [a3, b3, c3, d3,  e3, f3, g3, h3]

    (col0, col1, col2, col3)
}

/// Concatenates two 4-lane SIMD vectors (`f64x4`) into a single 8-lane SIMD vector (`f64x8`).
#[inline]
#[must_use]
pub fn concat_4x(a: f64x4, b: f64x4) -> f64x8 {
    simd_swizzle!(a, b, [0, 1, 2, 3, 4, 5, 6, 7])
}

/// Replicates a 4-lane SIMD vector (`f64x4`) twice into a single 8-lane SIMD vector (`f64x8`).
#[inline]
#[must_use]
pub fn splat_4x(v: f64x4) -> f64x8 {
    simd_swizzle!(v, [0, 1, 2, 3, 0, 1, 2, 3])
}

pub trait SimpleSimdElement: SimdCast + SimdElement {}
pub trait SimpleSimd<F, const N: usize>:
    SimdFloat<Cast<i32> = Simd<i32, N>>
    + SimdPartialOrd
    + SimdPartialEq<Mask = Mask<<F as SimdElement>::Mask, N>>
    + ops::Add<Output = Simd<F, N>>
    + ops::Sub<Output = Simd<F, N>>
    + ops::Mul<Output = Simd<F, N>>
    + ops::Div<Output = Simd<F, N>>
    + StdFloat
where
    F: SimpleSimdElement,
{
}
