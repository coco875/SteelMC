use core::simd::f64x4;
use std::{
    ops,
    simd::{Simd, SimdCast, SimdElement, num::SimdFloat},
};

use crate::{GRADIENT, GRADIENT_4, simd_utils::transpose};

/// Calculate the dot product of a gradient vector and the position vector.
#[expect(clippy::inline_always, reason = "hot-path noise primitive")]
#[inline(always)]
#[must_use]
pub fn grad_dot(hash: usize, x: f64, y: f64, z: f64) -> f64 {
    let g = &GRADIENT[hash & 15];
    g[0] * x + g[1] * y + g[2] * z
}

/// Calculate the dot product of a SIMD gradient vector and position vectors.
#[expect(clippy::inline_always, reason = "hot-path noise primitive")]
#[inline(always)]
#[must_use]
pub fn grad_dot_simd<F, const N: usize>(
    hash: [usize; N],
    x: Simd<F, N>,
    y: Simd<F, N>,
    z: Simd<F, N>,
) -> Simd<F, N>
where
    F: SimdElement + SimdCast,
    Simd<F, N>: ops::Mul<Output = Simd<F, N>> + ops::Add<Output = Simd<F, N>>,
{
    let mut gx = [0.; N];
    let mut gy = [0.; N];
    let mut gz = [0.; N];

    for i in 0..N {
        let g = &GRADIENT[hash[i] & 15];
        gx[i] = g[0];
        gy[i] = g[1];
        gz[i] = g[2];
    }
    let gx = Simd::from_array(gx).cast::<F>();
    let gy = Simd::from_array(gy).cast::<F>();
    let gz = Simd::from_array(gz).cast::<F>();
    gx * x + gy * y + gz * z
}

/// Gather gradient components for 4 hashes into separate x/y/z SIMD vectors,
/// then compute the dot product with the given position vectors.
#[expect(clippy::inline_always, reason = "hot-path noise primitive")]
#[inline(always)]
#[must_use]
pub fn grad_dot_4x(hashes: [usize; 4], x: f64x4, y: f64x4, z: f64x4) -> f64x4 {
    let h0 = f64x4::from_array(GRADIENT_4[hashes[0] & 15]);
    let h1 = f64x4::from_array(GRADIENT_4[hashes[1] & 15]);
    let h2 = f64x4::from_array(GRADIENT_4[hashes[2] & 15]);
    let h3 = f64x4::from_array(GRADIENT_4[hashes[3] & 15]);

    let (gx, gy, gz, _gw) = transpose(h0, h1, h2, h3);

    gx * x + gy * y + gz * z
}
