use core::simd::f64x4;
use std::simd::{f64x8, f64x16, simd_swizzle};

use crate::{
    GRADIENT, GRADIENT_4,
    simd_utils::{concat_4x, transpose, transpose_2x},
};

/// Calculate the dot product of a gradient vector and the position vector.
#[expect(clippy::inline_always, reason = "hot-path noise primitive")]
#[inline(always)]
#[must_use]
pub fn grad_dot(hash: usize, x: f64, y: f64, z: f64) -> f64 {
    let g = &GRADIENT[hash & 15];
    g[0] * x + g[1] * y + g[2] * z
}

/// Gather gradient components for 4 hashes into separate x/y/z SIMD vectors,
/// then compute the dot product with the given position vectors.
#[inline]
#[must_use]
pub fn grad_dot_4x(hashes: [usize; 4], x: f64x4, y: f64x4, z: f64x4) -> f64x4 {
    let h0 = f64x4::from_array(GRADIENT_4[hashes[0] & 15]);
    let h1 = f64x4::from_array(GRADIENT_4[hashes[1] & 15]);
    let h2 = f64x4::from_array(GRADIENT_4[hashes[2] & 15]);
    let h3 = f64x4::from_array(GRADIENT_4[hashes[3] & 15]);

    let (gx, gy, gz, _gw) = transpose(h0, h1, h2, h3);

    gx * x + gy * y + gz * z
}

pub fn grad_dot_8x(hashes: [usize; 8], x: f64x8, y: f64x8, z: f64x8) -> f64x8 {
    let h0 = f64x4::from_array(GRADIENT_4[hashes[0] & 15]);
    let h1 = f64x4::from_array(GRADIENT_4[hashes[1] & 15]);
    let h2 = f64x4::from_array(GRADIENT_4[hashes[2] & 15]);
    let h3 = f64x4::from_array(GRADIENT_4[hashes[3] & 15]);
    let h4 = f64x4::from_array(GRADIENT_4[hashes[4] & 15]);
    let h5 = f64x4::from_array(GRADIENT_4[hashes[5] & 15]);
    let h6 = f64x4::from_array(GRADIENT_4[hashes[6] & 15]);
    let h7 = f64x4::from_array(GRADIENT_4[hashes[7] & 15]);

    let h0 = concat_4x(h0, h4);
    let h1 = concat_4x(h1, h5);
    let h2 = concat_4x(h2, h6);
    let h3 = concat_4x(h3, h7);

    let (gx, gy, gz, _gw) = transpose_2x(h0, h1, h2, h3);

    gx * x + gy * y + gz * z
}
