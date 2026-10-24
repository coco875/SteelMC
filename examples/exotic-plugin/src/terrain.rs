use steel_plugin_api::stabby;
use steel_plugin_api::{BlockChange, BlockPos, Terrain};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Material {
    Air,
    Stone,
    Shell,
    Pillar,
    Light,
}

struct Shape {
    x_offset: i64,
    z_offset: i64,
    height_offset: i64,
    twist_offset: i64,
}

impl Shape {
    fn new(samples: [u64; 4]) -> Self {
        Self {
            x_offset: (samples[0] % 160) as i64,
            z_offset: (samples[1] % 160) as i64,
            height_offset: (samples[2] % 32) as i64,
            twist_offset: (samples[3] % 16) as i64,
        }
    }

    // Integer geometry keeps threshold decisions independent of floating-point math.
    fn material(&self, x: i32, y: i32, z: i32) -> Material {
        let x = i64::from(x) + self.x_offset;
        let z = i64::from(z) + self.z_offset;
        let y = i64::from(y);
        let dx = x.rem_euclid(80) - 40;
        let dz = z.rem_euclid(80) - 40;
        let center_y =
            80 + self.height_offset + (x.div_euclid(80) + z.div_euclid(80)).rem_euclid(3) * 16;
        let radial = dx * dx + dz * dz;
        let thickness = (900 - radial).max(0) / 55;
        if thickness > 0 && (y - center_y).abs() <= thickness {
            return Material::Stone;
        }

        let sx = x.rem_euclid(160) - 80;
        let sz = z.rem_euclid(160) - 80;
        let sy = y - 64 - self.height_offset;
        let radius = sx * sx + sy * sy + sz * sz;
        if (900..=1024).contains(&radius) {
            return Material::Shell;
        }

        const RING: [(i64, i64); 16] = [
            (6, 0),
            (6, 2),
            (4, 4),
            (2, 6),
            (0, 6),
            (-2, 6),
            (-4, 4),
            (-6, 2),
            (-6, 0),
            (-6, -2),
            (-4, -4),
            (-2, -6),
            (0, -6),
            (2, -6),
            (4, -4),
            (6, -2),
        ];
        let (twist_x, twist_z) =
            RING[(y.div_euclid(4) + self.twist_offset).rem_euclid(16) as usize];
        let px = x.rem_euclid(64) - 32 - twist_x;
        let pz = z.rem_euclid(64) - 32 - twist_z;
        if (24..=160).contains(&y) && px * px + pz * pz <= 6 {
            return if y.rem_euclid(9) == 0 {
                Material::Light
            } else {
                Material::Pillar
            };
        }
        Material::Air
    }
}

#[steel_plugin_api::stabby::export(callback)]
pub(crate) fn generate(terrain: Terrain<'_>) {
    let origin = BlockPos { x: 0, y: 0, z: 0 };
    let shape = Shape::new([
        terrain.random_at(0x6578_6f74_6963_0000, origin),
        terrain.random_at(0x6578_6f74_6963_0001, origin),
        terrain.random_at(0x6578_6f74_6963_0002, origin),
        terrain.random_at(0x6578_6f74_6963_0003, origin),
    ]);
    let materials = &crate::settings().materials;
    let mut changes = Vec::new();
    let min_x = terrain.min_x();
    let min_z = terrain.min_z();
    for x in min_x..min_x + 16 {
        for z in min_z..min_z + 16 {
            for y in terrain.min_y()..terrain.min_y() + terrain.height() {
                let state = match shape.material(x, y, z) {
                    Material::Air => continue,
                    Material::Stone => materials.stone,
                    Material::Shell => materials.shell,
                    Material::Pillar => materials.pillar,
                    Material::Light => materials.light,
                };
                changes.push(BlockChange {
                    pos: BlockPos { x, y, z },
                    state,
                });
            }
        }
    }
    if let Err(error) = terrain
        .write_blocks(changes.as_slice().into())
        .match_owned(Ok, Err)
    {
        panic!("strange-world generation produced an invalid terrain batch: {error}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hollow_shell_has_an_air_interior() {
        let shape = Shape::new([0; 4]);
        assert_eq!(shape.material(80, 64, 80), Material::Air);
        assert_eq!(shape.material(80, 95, 80), Material::Shell);
    }

    #[test]
    fn changing_seed_samples_changes_geometry() {
        let left = Shape::new([0; 4]);
        let right = Shape::new([27, 13, 19, 8]);
        let differences = (-64..64)
            .flat_map(|x| (-64..64).map(move |z| (x, z)))
            .filter(|(x, z)| left.material(*x, 90, *z) != right.material(*x, 90, *z))
            .count();
        assert!(differences > 100);
    }
}
