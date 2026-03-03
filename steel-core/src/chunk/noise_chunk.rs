//! NoiseChunk: cell-based terrain density evaluation with trilinear interpolation.
//!
//! Matches vanilla's `NoiseChunk` + `NoiseBasedChunkGenerator.doFill()` flow.
//! Samples the `final_density` function at cell corners, then interpolates
//! between corners for each individual block position.
//!
//! Cell dimensions depend on the dimension's noise settings.

use std::marker::PhantomData;
use std::mem;

use steel_utils::density::{ColumnCache, DimensionNoises, NoiseSettings};
use steel_utils::math::lerp;

/// Stores density values at cell corners for a single chunk and provides
/// trilinear interpolation between corners for block-level resolution.
pub struct NoiseChunk<N: DimensionNoises> {
    /// Density values at cell corners.
    /// Layout: `[cell_z_corner][cell_y_corner]` for the current and next X slices.
    slice0: Vec<Vec<f64>>,
    slice1: Vec<Vec<f64>>,

    /// First cell X/Z in world coordinates (cell index, not block).
    first_cell_x: i32,
    first_cell_z: i32,
    /// Minimum cell Y index.
    cell_min_y: i32,
    /// Number of cells in Y direction.
    cell_count_y: usize,
    /// Number of cells per chunk in XZ.
    cell_count_xz: usize,

    _phantom: PhantomData<N>,
}

impl<N: DimensionNoises> NoiseChunk<N> {
    /// Create a new `NoiseChunk` for the given chunk position.
    ///
    /// `chunk_min_block_x` and `chunk_min_block_z` are the world-space block
    /// coordinates of the chunk's northwest corner.
    #[must_use]
    pub fn new(chunk_min_block_x: i32, chunk_min_block_z: i32) -> Self {
        let cell_width = N::Settings::CELL_WIDTH;
        let cell_height = N::Settings::CELL_HEIGHT;
        let min_y = N::Settings::MIN_Y;
        let height = N::Settings::HEIGHT;

        let first_cell_x = chunk_min_block_x.div_euclid(cell_width);
        let first_cell_z = chunk_min_block_z.div_euclid(cell_width);
        let cell_min_y = min_y.div_euclid(cell_height);

        let cell_count_xz = (16 / cell_width) as usize;
        let cell_count_y = (height / cell_height) as usize;
        let corners_y = cell_count_y + 1;
        let z_corners = cell_count_xz + 1;

        Self {
            slice0: vec![vec![0.0; corners_y]; z_corners],
            slice1: vec![vec![0.0; corners_y]; z_corners],
            first_cell_x,
            first_cell_z,
            cell_min_y,
            cell_count_y,
            cell_count_xz,
            _phantom: PhantomData,
        }
    }

    /// Fill a density slice at the given cell X coordinate.
    #[allow(clippy::needless_range_loop)]
    fn fill_slice(
        &mut self,
        use_slice0: bool,
        cell_x: i32,
        noises: &N,
        cache: &mut N::ColumnCache,
    ) {
        let cell_width = N::Settings::CELL_WIDTH;
        let cell_height = N::Settings::CELL_HEIGHT;
        let corners_y = self.cell_count_y + 1;

        let block_x = cell_x * cell_width;
        let slice = if use_slice0 {
            &mut self.slice0
        } else {
            &mut self.slice1
        };

        for cz in 0..=self.cell_count_xz {
            let cell_z = self.first_cell_z + cz as i32;
            let block_z = cell_z * cell_width;

            // Ensure column cache for this (x, z)
            cache.ensure(block_x, block_z, noises);

            for cy in 0..corners_y {
                let block_y = (cy as i32 + self.cell_min_y) * cell_height;
                let density = noises.router_final_density(cache, block_x, block_y, block_z);
                slice[cz][cy] = density;
            }
        }
    }

    /// Fill the chunk with terrain blocks using trilinear interpolation.
    ///
    /// Calls `place_block` for each block position with the interpolated density.
    /// The callback receives `(local_x, world_y, local_z, density)` where
    /// density > 0 means solid, <= 0 means air/fluid.
    pub fn fill<F>(&mut self, noises: &N, cache: &mut N::ColumnCache, mut place_block: F)
    where
        F: FnMut(usize, i32, usize, f64),
    {
        let cell_width = N::Settings::CELL_WIDTH;
        let cell_height = N::Settings::CELL_HEIGHT;
        let sea_level = N::Settings::SEA_LEVEL;
        let cell_count_xz = self.cell_count_xz;
        let cell_count_y = self.cell_count_y;

        // Fill initial X slice (slice0)
        self.fill_slice(true, self.first_cell_x, noises, cache);

        for cell_x_idx in 0..cell_count_xz {
            // Fill next X slice (slice1)
            self.fill_slice(
                false,
                self.first_cell_x + cell_x_idx as i32 + 1,
                noises,
                cache,
            );

            for cell_z_idx in 0..cell_count_xz {
                for cell_y_idx in (0..cell_count_y).rev() {
                    // Get 8 corner values for this cell
                    let n000 = self.slice0[cell_z_idx][cell_y_idx];
                    let n001 = self.slice0[cell_z_idx + 1][cell_y_idx];
                    let n100 = self.slice1[cell_z_idx][cell_y_idx];
                    let n101 = self.slice1[cell_z_idx + 1][cell_y_idx];
                    let n010 = self.slice0[cell_z_idx][cell_y_idx + 1];
                    let n011 = self.slice0[cell_z_idx + 1][cell_y_idx + 1];
                    let n110 = self.slice1[cell_z_idx][cell_y_idx + 1];
                    let n111 = self.slice1[cell_z_idx + 1][cell_y_idx + 1];

                    for y_in_cell in (0..cell_height).rev() {
                        let factor_y = f64::from(y_in_cell) / f64::from(cell_height);
                        // Lerp in Y for 4 XZ corners
                        let d00 = lerp(factor_y, n000, n010);
                        let d10 = lerp(factor_y, n100, n110);
                        let d01 = lerp(factor_y, n001, n011);
                        let d11 = lerp(factor_y, n101, n111);

                        for x_in_cell in 0..cell_width {
                            let factor_x = f64::from(x_in_cell) / f64::from(cell_width);
                            // Lerp in X
                            let d0 = lerp(factor_x, d00, d10);
                            let d1 = lerp(factor_x, d01, d11);

                            for z_in_cell in 0..cell_width {
                                let factor_z = f64::from(z_in_cell) / f64::from(cell_width);
                                // Lerp in Z
                                let density = lerp(factor_z, d0, d1);

                                let world_y =
                                    (self.cell_min_y + cell_y_idx as i32) * cell_height + y_in_cell;
                                let local_x = (cell_x_idx as i32 * cell_width + x_in_cell) as usize;
                                let local_z = (cell_z_idx as i32 * cell_width + z_in_cell) as usize;

                                if density > 0.0 || world_y < sea_level {
                                    place_block(local_x, world_y, local_z, density);
                                }
                                // else: air above sea level (don't place anything)
                            }
                        }
                    }
                }
            }

            // Swap slices: current next becomes current for the next iteration
            mem::swap(&mut self.slice0, &mut self.slice1);
        }
    }
}
