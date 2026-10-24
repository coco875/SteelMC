//! Authored maps, built by terrain callbacks without template files or disk saves.
use crate::api;
use std::sync::OnceLock;
use steel_plugin_api::{BlockChange, BlockPos, BlockState, Terrain, default_block_state, stabby};
pub const FLOOR: i32 = 64;
pub const RADIUS: i32 = 16;
struct Materials {
    snow: BlockState,
    quartz: BlockState,
    glass: BlockState,
    bedrock: BlockState,
    light: BlockState,
}
static MATERIALS: OnceLock<Materials> = OnceLock::new();
pub fn initialize() -> Result<(), String> {
    MATERIALS
        .set(Materials {
            snow: api!(default_block_state("minecraft:snow_block".into()))?,
            quartz: api!(default_block_state("minecraft:smooth_quartz".into()))?,
            glass: api!(default_block_state("minecraft:glass".into()))?,
            bedrock: api!(default_block_state("minecraft:bedrock".into()))?,
            light: api!(default_block_state("minecraft:sea_lantern".into()))?,
        })
        .map_err(|_| "Spleef materials initialized twice".into())
}
pub fn snow() -> Result<BlockState, String> {
    Ok(MATERIALS.get().ok_or("Spleef materials unavailable")?.snow)
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Material {
    Snow,
    Quartz,
    Glass,
    Bedrock,
    Light,
}
fn at(arena: bool, x: i32, y: i32, z: i32) -> Option<Material> {
    let radius = if arena { RADIUS } else { 12 };
    if x.abs() > radius || z.abs() > radius {
        return None;
    }
    let edge = x.abs() == radius || z.abs() == radius;
    if y == FLOOR {
        return Some(if edge && arena {
            Material::Bedrock
        } else if !arena && x.abs() == 8 && z.abs() == 8 {
            Material::Light
        } else if arena {
            Material::Snow
        } else {
            Material::Quartz
        });
    }
    if edge && (FLOOR + 1..=FLOOR + 3).contains(&y) {
        return Some(if arena {
            Material::Bedrock
        } else {
            Material::Glass
        });
    }
    None
}
fn generate(terrain: Terrain<'_>, arena: bool) {
    let Some(m) = MATERIALS.get() else {
        panic!("Spleef generator ran before init");
    };
    let mut changes = Vec::new();
    for x in terrain.min_x()..terrain.min_x() + 16 {
        for z in terrain.min_z()..terrain.min_z() + 16 {
            for y in FLOOR..=FLOOR + 3 {
                let Some(material) = at(arena, x, y, z) else {
                    continue;
                };
                let state = match material {
                    Material::Snow => m.snow,
                    Material::Quartz => m.quartz,
                    Material::Glass => m.glass,
                    Material::Bedrock => m.bedrock,
                    Material::Light => m.light,
                };
                changes.push(BlockChange {
                    pos: BlockPos { x, y, z },
                    state,
                });
            }
        }
    }
    if let Err(error) = api!(terrain.write_blocks(changes.as_slice().into())) {
        panic!("invalid Spleef terrain batch: {error}");
    }
}
#[stabby::export(callback)]
pub fn lobby(terrain: Terrain<'_>) {
    generate(terrain, false);
}
#[stabby::export(callback)]
pub fn arena(terrain: Terrain<'_>) {
    generate(terrain, true);
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn arena_has_breakable_floor_a_void_below_and_unbreakable_walls() {
        assert_eq!(at(true, 0, FLOOR, 0), Some(Material::Snow));
        assert_eq!(at(true, 0, FLOOR - 1, 0), None);
        assert_eq!(at(true, RADIUS, FLOOR + 2, 0), Some(Material::Bedrock));
        assert_eq!(at(true, RADIUS + 1, FLOOR, 0), None);
        assert_eq!(at(false, 12, FLOOR + 1, 0), Some(Material::Glass));
    }
}
