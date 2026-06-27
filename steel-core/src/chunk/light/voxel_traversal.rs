use steel_utils::{BlockPos, Direction};

/// Iterates every adjacent block offset except `(0, 0, 0)`.
pub fn for_each_adjacent_block_offset(mut f: impl FnMut(i32, i32, i32)) {
    for dx in -1..=1 {
        for dy in -1..=1 {
            for dz in -1..=1 {
                if dx == 0 && dy == 0 && dz == 0 {
                    continue;
                }
                f(dx, dy, dz);
            }
        }
    }
}

/// Returns true when the offset is diagonal (two or more axes non-zero).
#[must_use]
pub const fn is_diagonal_block_offset(dx: i32, dy: i32, dz: i32) -> bool {
    let non_zero = (dx != 0) as u8 + (dy != 0) as u8 + (dz != 0) as u8;
    non_zero >= 2
}

/// Returns the next block voxel stepped from `from` toward `to`, excluding `from`.
#[must_use]
pub fn next_voxel_toward(from: BlockPos, to: BlockPos) -> Option<BlockPos> {
    if from == to {
        return None;
    }

    let mut traversal = VoxelLineTraversal::new(from, to);
    traversal.next()
}

/// Returns the next block one DDA step past `from`, continuing the line from
/// `through_from` through `from`.
#[must_use]
pub fn next_voxel_past(from: BlockPos, through_from: BlockPos) -> Option<BlockPos> {
    if from == through_from {
        return None;
    }

    let past_target = BlockPos::new(
        from.x() + (from.x() - through_from.x()),
        from.y() + (from.y() - through_from.y()),
        from.z() + (from.z() - through_from.z()),
    );
    next_voxel_toward(from, past_target)
}

/// Returns true when `through` lies on the integer grid line from `from` to `to`.
#[must_use]
pub fn ray_passes_through(from: BlockPos, to: BlockPos, through: BlockPos) -> bool {
    if from == through || to == through {
        return true;
    }

    VoxelLineTraversal::new(from, to).any(|pos| pos == through)
}

/// Returns the dominant axis direction from `from` toward `to`.
#[must_use]
pub fn dominant_direction(from: BlockPos, to: BlockPos) -> Direction {
    let dx = to.x() - from.x();
    let dy = to.y() - from.y();
    let dz = to.z() - from.z();
    let abs_x = dx.abs();
    let abs_y = dy.abs();
    let abs_z = dz.abs();

    if abs_x >= abs_y && abs_x >= abs_z {
        if dx >= 0 {
            Direction::East
        } else {
            Direction::West
        }
    } else if abs_y >= abs_z {
        if dy >= 0 {
            Direction::Up
        } else {
            Direction::Down
        }
    } else if dz >= 0 {
        Direction::South
    } else {
        Direction::North
    }
}

/// Fast voxel traversal (Amanatides & Woo) between two block positions.
struct VoxelLineTraversal {
    end: BlockPos,
    x: i32,
    y: i32,
    z: i32,
    step_x: i32,
    step_y: i32,
    step_z: i32,
    t_max_x: f64,
    t_max_y: f64,
    t_max_z: f64,
    t_delta_x: f64,
    t_delta_y: f64,
    t_delta_z: f64,
    finished: bool,
}

impl VoxelLineTraversal {
    fn new(from: BlockPos, to: BlockPos) -> Self {
        let start = (
            f64::from(from.x()) + 0.5,
            f64::from(from.y()) + 0.5,
            f64::from(from.z()) + 0.5,
        );
        let end = (
            f64::from(to.x()) + 0.5,
            f64::from(to.y()) + 0.5,
            f64::from(to.z()) + 0.5,
        );

        let dir = (end.0 - start.0, end.1 - start.1, end.2 - start.2);
        let step_x = dir.0.signum() as i32;
        let step_y = dir.1.signum() as i32;
        let step_z = dir.2.signum() as i32;

        let t_delta_x = if dir.0 == 0.0 {
            f64::INFINITY
        } else {
            (1.0 / dir.0).abs()
        };
        let t_delta_y = if dir.1 == 0.0 {
            f64::INFINITY
        } else {
            (1.0 / dir.1).abs()
        };
        let t_delta_z = if dir.2 == 0.0 {
            f64::INFINITY
        } else {
            (1.0 / dir.2).abs()
        };

        let mut t_max_x = if dir.0 == 0.0 {
            f64::INFINITY
        } else {
            let boundary = if step_x > 0 {
                f64::from(from.x()) + 1.0
            } else {
                f64::from(from.x())
            };
            (boundary - start.0) / dir.0
        };
        let mut t_max_y = if dir.1 == 0.0 {
            f64::INFINITY
        } else {
            let boundary = if step_y > 0 {
                f64::from(from.y()) + 1.0
            } else {
                f64::from(from.y())
            };
            (boundary - start.1) / dir.1
        };
        let mut t_max_z = if dir.2 == 0.0 {
            f64::INFINITY
        } else {
            let boundary = if step_z > 0 {
                f64::from(from.z()) + 1.0
            } else {
                f64::from(from.z())
            };
            (boundary - start.2) / dir.2
        };

        // Advance once so the first yielded cell is the first step away from `from`.
        let (x, y, z) = advance_voxel(
            from.x(),
            from.y(),
            from.z(),
            step_x,
            step_y,
            step_z,
            &mut t_max_x,
            &mut t_max_y,
            &mut t_max_z,
            t_delta_x,
            t_delta_y,
            t_delta_z,
        );

        Self {
            end: to,
            x,
            y,
            z,
            step_x,
            step_y,
            step_z,
            t_max_x,
            t_max_y,
            t_max_z,
            t_delta_x,
            t_delta_y,
            t_delta_z,
            finished: false,
        }
    }
}

impl Iterator for VoxelLineTraversal {
    type Item = BlockPos;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }

        let current = BlockPos::new(self.x, self.y, self.z);
        if current == self.end {
            self.finished = true;
            return Some(current);
        }

        let (x, y, z) = advance_voxel(
            self.x,
            self.y,
            self.z,
            self.step_x,
            self.step_y,
            self.step_z,
            &mut self.t_max_x,
            &mut self.t_max_y,
            &mut self.t_max_z,
            self.t_delta_x,
            self.t_delta_y,
            self.t_delta_z,
        );
        self.x = x;
        self.y = y;
        self.z = z;

        Some(current)
    }
}

#[expect(clippy::too_many_arguments, reason = "Amanatides & Woo voxel step state")]
fn advance_voxel(
    x: i32,
    y: i32,
    z: i32,
    step_x: i32,
    step_y: i32,
    step_z: i32,
    t_max_x: &mut f64,
    t_max_y: &mut f64,
    t_max_z: &mut f64,
    t_delta_x: f64,
    t_delta_y: f64,
    t_delta_z: f64,
) -> (i32, i32, i32) {
    if *t_max_x < *t_max_y {
        if *t_max_x < *t_max_z {
            *t_max_x += t_delta_x;
            (x + step_x, y, z)
        } else {
            *t_max_z += t_delta_z;
            (x, y, z + step_z)
        }
    } else if *t_max_y < *t_max_z {
        *t_max_y += t_delta_y;
        (x, y + step_y, z)
    } else {
        *t_max_z += t_delta_z;
        (x, y, z + step_z)
    }
}

/// Yields every block voxel on the line from `from` through `to`, including both ends.
pub fn line_voxels_between(from: BlockPos, to: BlockPos) -> impl Iterator<Item = BlockPos> {
    LineVoxelsBetween {
        from,
        to,
        inner: None,
        finished: from == to,
    }
}

struct LineVoxelsBetween {
    from: BlockPos,
    to: BlockPos,
    inner: Option<VoxelLineTraversal>,
    finished: bool,
}

impl Iterator for LineVoxelsBetween {
    type Item = BlockPos;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }

        if self.inner.is_none() {
            self.inner = Some(VoxelLineTraversal::new(self.from, self.to));
            return Some(self.from);
        }

        let Some(inner) = self.inner.as_mut() else {
            return None;
        };
        if let Some(step) = inner.next() {
            if step == self.to {
                self.finished = true;
            }
            return Some(step);
        }

        if self.from != self.to {
            self.finished = true;
            return Some(self.to);
        }

        self.finished = true;
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn collect_line(from: BlockPos, to: BlockPos) -> Vec<BlockPos> {
        VoxelLineTraversal::new(from, to).collect()
    }

    #[test]
    fn traversal_steps_along_z_axis() {
        let from = BlockPos::new(11, 70, 21);
        let to = BlockPos::new(11, 70, 14);
        assert_eq!(
            collect_line(from, to),
            vec![
                BlockPos::new(11, 70, 20),
                BlockPos::new(11, 70, 19),
                BlockPos::new(11, 70, 18),
                BlockPos::new(11, 70, 17),
                BlockPos::new(11, 70, 16),
                BlockPos::new(11, 70, 15),
                BlockPos::new(11, 70, 14),
            ]
        );
    }

    #[test]
    fn next_voxel_past_continues_diagonal_line() {
        let emitter = BlockPos::new(0, 0, 0);
        let first = BlockPos::new(1, 0, -1);
        let second = next_voxel_past(first, emitter);
        assert_eq!(second, Some(BlockPos::new(1, 0, -2)));
    }

    #[test]
    fn line_voxels_between_includes_endpoints_without_duplicates() {
        let from = BlockPos::new(1, 1, 1);
        let to = BlockPos::new(3, 1, 1);
        assert_eq!(
            line_voxels_between(from, to).collect::<Vec<_>>(),
            vec![
                BlockPos::new(1, 1, 1),
                BlockPos::new(2, 1, 1),
                BlockPos::new(3, 1, 1),
            ]
        );
    }

    #[test]
    fn ray_passes_through_collinear_torch_line() {
        assert!(ray_passes_through(
            BlockPos::new(14, 1, 1),
            BlockPos::new(16, 1, 1),
            BlockPos::new(15, 1, 1),
        ));
        assert_eq!(
            next_voxel_past(BlockPos::new(15, 1, 1), BlockPos::new(16, 1, 1)),
            Some(BlockPos::new(14, 1, 1))
        );
    }

    #[test]
    fn ray_passes_through_intermediate_voxel() {
        let from = BlockPos::new(11, 70, 22);
        let to = BlockPos::new(11, 70, 14);
        let through = BlockPos::new(11, 70, 18);
        assert!(ray_passes_through(from, to, through));
    }
}
