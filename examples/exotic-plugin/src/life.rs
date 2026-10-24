use serde::Deserialize;
use steel_plugin_api::stabby;
use steel_plugin_api::{BlockPos, World, stabby::string::String as ApiString};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct LifeConfig {
    pub world: String,
    pub x: i32,
    pub y: i32,
    pub z: i32,
    pub width: usize,
    pub depth: usize,
    pub interval: u64,
}

impl LifeConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.world.is_empty() {
            return Err("life.world must identify a loaded world".into());
        }
        // This example limits its synchronous work to at most 16,384 cells.
        if !(1..=128).contains(&self.width) || !(1..=128).contains(&self.depth) {
            return Err("life.width and life.depth must be between 1 and 128".into());
        }
        if self.interval == 0 {
            return Err("life.interval must be positive".into());
        }
        if self.x.checked_add(self.width as i32 - 1).is_none()
            || self.z.checked_add(self.depth as i32 - 1).is_none()
        {
            return Err("life board coordinates overflow".into());
        }
        Ok(())
    }

    fn position(&self, x: usize, z: usize) -> BlockPos {
        BlockPos {
            x: self.x + x as i32,
            y: self.y,
            z: self.z + z as i32,
        }
    }

    fn all_chunks_loaded(&self, mut is_loaded: impl FnMut(i32, i32) -> bool) -> bool {
        let max_x = self.x + self.width as i32 - 1;
        let max_z = self.z + self.depth as i32 - 1;
        for x in self.x.div_euclid(16)..=max_x.div_euclid(16) {
            for z in self.z.div_euclid(16)..=max_z.div_euclid(16) {
                if !is_loaded(x, z) {
                    return false;
                }
            }
        }
        true
    }
}

#[steel_plugin_api::stabby::export(callback)]
pub(crate) fn tick(world: World<'_>, tick: u64) {
    let settings = crate::settings();
    let config = &settings.life;
    if world.key().as_str() != config.world
        || !tick.is_multiple_of(config.interval)
        || !config.all_chunks_loaded(|x, z| world.is_chunk_loaded(x, z))
    {
        return;
    }
    let result = (|| -> Result<(), ApiString> {
        let mut current = Vec::with_capacity(config.width * config.depth);
        for z in 0..config.depth {
            for x in 0..config.width {
                let state = world.block(config.position(x, z)).match_owned(Ok, Err)?;
                current.push(state == settings.materials.alive);
            }
        }
        let next = step(&current, config.width, config.depth);
        for z in 0..config.depth {
            for x in 0..config.width {
                let index = z * config.width + x;
                if current[index] == next[index] {
                    continue;
                }
                let state = if next[index] {
                    settings.materials.alive
                } else {
                    settings.materials.dead
                };
                world
                    .set_block(config.position(x, z), state)
                    .match_owned(Ok, Err)?;
            }
        }
        Ok(())
    })();
    if let Err(error) = result {
        eprintln!("Game of Life update failed: {error}");
    }
}

fn step(current: &[bool], width: usize, depth: usize) -> Vec<bool> {
    let mut next = vec![false; current.len()];
    for z in 0..depth {
        for x in 0..width {
            let mut neighbors = 0;
            for dz in -1..=1isize {
                for dx in -1..=1isize {
                    if dx == 0 && dz == 0 {
                        continue;
                    }
                    let Some(nx) = x.checked_add_signed(dx).filter(|nx| *nx < width) else {
                        continue;
                    };
                    let Some(nz) = z.checked_add_signed(dz).filter(|nz| *nz < depth) else {
                        continue;
                    };
                    neighbors += u8::from(current[nz * width + nx]);
                }
            }
            let index = z * width + x;
            next[index] = neighbors == 3 || (current[index] && neighbors == 2);
        }
    }
    next
}

#[cfg(test)]
mod tests {
    use super::{LifeConfig, step};

    #[test]
    fn blinker_generations_are_simultaneous() {
        let mut horizontal = vec![false; 25];
        for x in 1..=3 {
            horizontal[2 * 5 + x] = true;
        }
        let mut vertical = vec![false; 25];
        for z in 1..=3 {
            vertical[z * 5 + 2] = true;
        }
        assert_eq!(step(&horizontal, 5, 5), vertical);
        assert_eq!(step(&vertical, 5, 5), horizontal);
    }

    #[test]
    fn opposite_edges_do_not_wrap_into_neighbors() {
        let mut cells = vec![false; 25];
        cells[0] = true;
        cells[4] = true;
        cells[20] = true;
        assert!(step(&cells, 5, 5).iter().all(|alive| !alive));
    }

    #[test]
    fn a_block_is_stable_against_the_dead_boundary() {
        let cells = vec![true, true, false, true, true, false, false, false, false];
        assert_eq!(step(&cells, 3, 3), cells);
    }

    #[test]
    fn board_readiness_includes_chunks_across_negative_boundaries() {
        let config = LifeConfig {
            world: "test:world".into(),
            x: -1,
            y: 64,
            z: -1,
            width: 2,
            depth: 2,
            interval: 10,
        };
        let mut visited = Vec::new();
        assert!(config.all_chunks_loaded(|x, z| {
            visited.push((x, z));
            true
        }));
        assert_eq!(visited, [(-1, -1), (-1, 0), (0, -1), (0, 0)]);
        assert!(!config.all_chunks_loaded(|x, z| (x, z) != (0, -1)));
    }
}
