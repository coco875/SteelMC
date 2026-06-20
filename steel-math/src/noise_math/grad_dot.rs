use core::simd::f64x4;
use std::{
    mem::transmute_copy,
    ops,
    simd::{Simd, SimdCast, SimdElement, num::SimdFloat},
};

use crate::{GRADIENT, GRADIENT_4, simd_utils::transpose};

/// Calculate 4 gradient dot products.
///
/// Baseline builds use table assembly because it is faster without AVX-512
/// masks; native AVX-512 builds use the branchless hash formula.
#[inline]
#[must_use]
pub fn grad_dot_4x(hashes: [usize; 4], x: f64x4, y: f64x4, z: f64x4) -> f64x4 {
    #[cfg(target_feature = "avx512f")]
    {
        grad_dot_simd(hashes, x, y, z)
    }

    #[cfg(not(target_feature = "avx512f"))]
    {
        let h0 = f64x4::from_array(GRADIENT_4[hashes[0] & 15]);
        let h1 = f64x4::from_array(GRADIENT_4[hashes[1] & 15]);
        let h2 = f64x4::from_array(GRADIENT_4[hashes[2] & 15]);
        let h3 = f64x4::from_array(GRADIENT_4[hashes[3] & 15]);

        let (gx, gy, gz, _gw) = transpose(h0, h1, h2, h3);

        gx * x + gy * y + gz * z
    }
}

/// Generic N-lane gradient dot product.
///
/// Evaluates Minecraft's 16-entry `GRADIENT` table branchlessly from the hash
/// bits using the public-domain reference formula from Ken Perlin's improved
/// noise implementation. For `hash & 15`, that formula is value-identical to
/// indexing Minecraft's `GRADIENT` table for all 16 entries.
///
/// Wider SIMD uses this path because the branchless formula avoids per-lane
/// component assembly.
#[inline]
#[must_use]
#[expect(
    clippy::similar_names,
    reason = "gx_4, gy_4, gz_4 match Cartesian coordinate components and are distinct"
)]
pub fn grad_dot_simd<F, const N: usize>(
    hashes: [usize; N],
    x: Simd<F, N>,
    y: Simd<F, N>,
    z: Simd<F, N>,
) -> Simd<F, N>
where
    F: SimdElement + SimdCast,
    Simd<F, N>: ops::Mul<Output = Simd<F, N>> + ops::Add<Output = Simd<F, N>>,
{
    #[cfg(target_feature = "avx512f")]
    {
        let hash_lanes = Simd::<i64, N>::from_array(hashes.map(|value| (value & 15) as i64));
        // u = h < 8 ? x : y
        let u_component = hash_lanes.simd_lt(Simd::splat(8)).select(x, y);
        // v = h < 4 ? y : (h == 12 || h == 14 ? x : z)
        let v_component = hash_lanes.simd_lt(Simd::splat(4)).select(
            y,
            (hash_lanes.simd_eq(Simd::splat(12)) | hash_lanes.simd_eq(Simd::splat(14)))
                .select(x, z),
        );
        // grad·pos = ((h & 1) == 0 ? u : -u) + ((h & 2) == 0 ? v : -v)
        let signed_u = (hash_lanes & Simd::splat(1))
            .simd_eq(Simd::splat(0))
            .select(u_component, -u_component);
        let signed_v = (hash_lanes & Simd::splat(2))
            .simd_eq(Simd::splat(0))
            .select(v_component, -v_component);
        signed_u + signed_v
    }

    #[cfg(not(target_feature = "avx512f"))]
    if N == 4 {
        let h0 = Simd::from_array(GRADIENT_4[hashes[0] & 15]).cast::<F>();
        let h1 = Simd::from_array(GRADIENT_4[hashes[1] & 15]).cast::<F>();
        let h2 = Simd::from_array(GRADIENT_4[hashes[2] & 15]).cast::<F>();
        let h3 = Simd::from_array(GRADIENT_4[hashes[3] & 15]).cast::<F>();

        let (gx_4, gy_4, gz_4, _gw) = transpose(h0, h1, h2, h3);

        // SAFETY: N is checked to be 4 in the branch condition, making Simd<F, 4> and Simd<F, N> layout compatible.
        let gx: Simd<F, N> = unsafe { transmute_copy(&gx_4) };
        // SAFETY: N is checked to be 4 in the branch condition, making Simd<F, 4> and Simd<F, N> layout compatible.
        let gy: Simd<F, N> = unsafe { transmute_copy(&gy_4) };
        // SAFETY: N is checked to be 4 in the branch condition, making Simd<F, 4> and Simd<F, N> layout compatible.
        let gz: Simd<F, N> = unsafe { transmute_copy(&gz_4) };

        gx * x + gy * y + gz * z
    } else {
        let mut gx = [0.; N];
        let mut gy = [0.; N];
        let mut gz = [0.; N];

        for i in 0..N {
            let g = &GRADIENT[hashes[i] & 15];
            gx[i] = g[0];
            gy[i] = g[1];
            gz[i] = g[2];
        }
        let gx = Simd::from_array(gx).cast();
        let gy = Simd::from_array(gy).cast();
        let gz = Simd::from_array(gz).cast();
        gx * x + gy * y + gz * z
    }
}

/// Calculate the dot product of a gradient vector and the position vector.
#[expect(clippy::inline_always, reason = "hot-path noise primitive")]
#[inline(always)]
#[must_use]
pub fn grad_dot(hash: usize, x: f64, y: f64, z: f64) -> f64 {
    let g = &GRADIENT[hash & 15];
    g[0] * x + g[1] * y + g[2] * z
}
