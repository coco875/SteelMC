//! NoiseChunk: cell-based terrain density evaluation with trilinear interpolation.
//!
//! Matches vanilla's `NoiseChunk` + `NoiseBasedChunkGenerator.doFill()` flow.
//! Samples the `final_density` function at cell corners, then interpolates
//! between corners for each individual block position.
//!
//! Cell dimensions (overworld):
//! - `cellWidth` = 4 blocks (XZ)
//! - `cellHeight` = 8 blocks (Y)
//! - A 16×384×16 chunk = 4×48×4 cells = 5×49×5 corner grid

use steel_registry::density_functions::{self, OverworldColumnCache, OverworldNoises};
use steel_utils::math::lerp;

/// Overworld noise settings.
const CELL_WIDTH: i32 = 4;
const CELL_HEIGHT: i32 = 8;
const MIN_Y: i32 = -64;
const WORLD_HEIGHT: i32 = 384;
const SEA_LEVEL: i32 = 63;

/// Number of cells per chunk in XZ.
const CELL_COUNT_XZ: usize = (16 / CELL_WIDTH) as usize; // 4
/// Number of cells in Y.
const CELL_COUNT_Y: usize = (WORLD_HEIGHT / CELL_HEIGHT) as usize; // 48
/// Number of cell corners in Y.
const CORNERS_Y: usize = CELL_COUNT_Y + 1; // 49

/// Stores density values at cell corners for a single chunk and provides
/// trilinear interpolation between corners for block-level resolution.
pub struct NoiseChunk {
    /// Density values at cell corners.
    /// Layout: `[cell_z_corner][cell_y_corner]` for the current and next X slices.
    /// Each slice is `(CELL_COUNT_XZ + 1)` Z columns, each with `CORNERS_Y` Y values.
    slice0: Vec<[f64; CORNERS_Y]>,
    slice1: Vec<[f64; CORNERS_Y]>,

    /// First cell X/Z in world coordinates (cell index, not block).
    first_cell_x: i32,
    first_cell_z: i32,
    /// Minimum cell Y index.
    cell_min_y: i32,
}

impl NoiseChunk {
    /// Create a new `NoiseChunk` for the given chunk position.
    ///
    /// `chunk_min_block_x` and `chunk_min_block_z` are the world-space block
    /// coordinates of the chunk's northwest corner.
    #[must_use]
    pub fn new(chunk_min_block_x: i32, chunk_min_block_z: i32) -> Self {
        let first_cell_x = chunk_min_block_x.div_euclid(CELL_WIDTH);
        let first_cell_z = chunk_min_block_z.div_euclid(CELL_WIDTH);
        let cell_min_y = MIN_Y.div_euclid(CELL_HEIGHT);

        let z_corners = CELL_COUNT_XZ + 1;
        Self {
            slice0: vec![[0.0; CORNERS_Y]; z_corners],
            slice1: vec![[0.0; CORNERS_Y]; z_corners],
            first_cell_x,
            first_cell_z,
            cell_min_y,
        }
    }

    /// Fill a density slice at the given cell X coordinate.
    fn fill_slice(
        &mut self,
        use_slice0: bool,
        cell_x: i32,
        noises: &OverworldNoises,
        cache: &mut OverworldColumnCache,
    ) {
        let block_x = cell_x * CELL_WIDTH;
        let slice = if use_slice0 {
            &mut self.slice0
        } else {
            &mut self.slice1
        };

        for cz in 0..=CELL_COUNT_XZ {
            let cell_z = self.first_cell_z + cz as i32;
            let block_z = cell_z * CELL_WIDTH;

            // Ensure column cache for this (x, z)
            cache.ensure(block_x, block_z, noises);

            for cy in 0..CORNERS_Y {
                let block_y = (cy as i32 + self.cell_min_y) * CELL_HEIGHT;
                let density = density_functions::router_final_density(
                    noises, cache, block_x, block_y, block_z,
                );
                slice[cz][cy] = density;
            }
        }
    }

    /// Fill the chunk with terrain blocks using trilinear interpolation.
    ///
    /// Calls `place_block` for each block position with the interpolated density.
    /// The callback receives `(local_x, world_y, local_z, density)` where
    /// density > 0 means solid, <= 0 means air/fluid.
    pub fn fill<F>(
        &mut self,
        noises: &OverworldNoises,
        cache: &mut OverworldColumnCache,
        mut place_block: F,
    ) where
        F: FnMut(usize, i32, usize, f64),
    {
        // Fill initial X slice (slice0)
        self.fill_slice(true, self.first_cell_x, noises, cache);

        for cell_x_idx in 0..CELL_COUNT_XZ {
            // Fill next X slice (slice1)
            self.fill_slice(
                false,
                self.first_cell_x + cell_x_idx as i32 + 1,
                noises,
                cache,
            );

            for cell_z_idx in 0..CELL_COUNT_XZ {
                for cell_y_idx in (0..CELL_COUNT_Y).rev() {
                    // Get 8 corner values for this cell
                    let n000 = self.slice0[cell_z_idx][cell_y_idx];
                    let n001 = self.slice0[cell_z_idx + 1][cell_y_idx];
                    let n100 = self.slice1[cell_z_idx][cell_y_idx];
                    let n101 = self.slice1[cell_z_idx + 1][cell_y_idx];
                    let n010 = self.slice0[cell_z_idx][cell_y_idx + 1];
                    let n011 = self.slice0[cell_z_idx + 1][cell_y_idx + 1];
                    let n110 = self.slice1[cell_z_idx][cell_y_idx + 1];
                    let n111 = self.slice1[cell_z_idx + 1][cell_y_idx + 1];

                    for y_in_cell in (0..CELL_HEIGHT).rev() {
                        let factor_y = f64::from(y_in_cell) / f64::from(CELL_HEIGHT);
                        // Lerp in Y for 4 XZ corners
                        let d00 = lerp(factor_y, n000, n010);
                        let d10 = lerp(factor_y, n100, n110);
                        let d01 = lerp(factor_y, n001, n011);
                        let d11 = lerp(factor_y, n101, n111);

                        for x_in_cell in 0..CELL_WIDTH {
                            let factor_x = f64::from(x_in_cell) / f64::from(CELL_WIDTH);
                            // Lerp in X
                            let d0 = lerp(factor_x, d00, d10);
                            let d1 = lerp(factor_x, d01, d11);

                            for z_in_cell in 0..CELL_WIDTH {
                                let factor_z = f64::from(z_in_cell) / f64::from(CELL_WIDTH);
                                // Lerp in Z
                                let density = lerp(factor_z, d0, d1);

                                let world_y =
                                    (self.cell_min_y + cell_y_idx as i32) * CELL_HEIGHT + y_in_cell;
                                let local_x = (cell_x_idx as i32 * CELL_WIDTH + x_in_cell) as usize;
                                let local_z = (cell_z_idx as i32 * CELL_WIDTH + z_in_cell) as usize;

                                if density > 0.0 || world_y < SEA_LEVEL {
                                    place_block(local_x, world_y, local_z, density);
                                }
                                // else: air above sea level (don't place anything)
                            }
                        }
                    }
                }
            }

            // Swap slices: current next becomes current for the next iteration
            std::mem::swap(&mut self.slice0, &mut self.slice1);
        }
    }
}
