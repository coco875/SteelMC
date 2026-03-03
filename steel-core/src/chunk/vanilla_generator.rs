use steel_registry::density_functions::{OverworldColumnCache, OverworldNoises};
use steel_registry::noise_parameters::get_noise_parameters;
use steel_registry::{REGISTRY, vanilla_blocks};
use steel_utils::BlockStateId;
use steel_utils::random::{Random, RandomSplitter, xoroshiro::Xoroshiro};

use crate::chunk::aquifer::{Aquifer, AquiferResult};
use crate::chunk::chunk_access::ChunkAccess;
use crate::chunk::chunk_generator::ChunkGenerator;
use crate::chunk::noise_chunk::NoiseChunk;
use crate::chunk::ore_veinifier::OreVeinifier;
use crate::worldgen::BiomeSourceKind;

/// Overworld minimum Y coordinate.
const MIN_Y: i32 = -64;

/// A chunk generator for vanilla (normal) world generation.
///
/// Matches vanilla's `NoiseBasedChunkGenerator`. The biome source is pluggable
/// per-dimension — overworld, nether, and end each provide a different
/// [`BiomeSourceKind`] variant.
pub struct VanillaGenerator {
    /// Biome source for this dimension. Determines biomes at each quart position.
    biome_source: BiomeSourceKind,
    /// Noise generators for the overworld density functions.
    /// Boxed because `OverworldNoises` is ~5600 bytes.
    noises: Box<OverworldNoises>,
    /// Seed positional splitter for per-chunk construction of aquifers.
    splitter: RandomSplitter,
    /// Ore vein generator for replacing stone with ore blocks.
    ore_veinifier: OreVeinifier,
    /// Block state ID for stone, cached at construction time.
    stone_id: BlockStateId,
}

impl VanillaGenerator {
    /// Creates a new `VanillaGenerator` with the given biome source and seed.
    #[must_use]
    pub fn new(biome_source: BiomeSourceKind, seed: u64) -> Self {
        let mut rng = Xoroshiro::from_seed(seed);
        let splitter = rng.next_positional();
        let noise_params = get_noise_parameters();
        let noises = OverworldNoises::create(&splitter, &noise_params);

        let ore_veinifier = OreVeinifier::new(&splitter);

        Self {
            biome_source,
            noises: Box::new(noises),
            splitter,
            ore_veinifier,
            stone_id: REGISTRY.blocks.get_default_state_id(vanilla_blocks::STONE),
        }
    }
}

impl ChunkGenerator for VanillaGenerator {
    fn create_structures(&self, _chunk: &ChunkAccess) {}

    fn create_biomes(&self, chunk: &ChunkAccess) {
        let pos = chunk.pos();
        let min_y = chunk.min_y();
        let section_count = chunk.sections().sections.len();

        let chunk_x = pos.0.x;
        let chunk_z = pos.0.y;

        let mut sampler = self.biome_source.chunk_sampler();

        // Column-major iteration: sample all Y values for each (X, Z) column
        // before moving to the next column. This keeps the column cache effective —
        // column-level density functions (continents, erosion, ridges, etc.) are
        // computed once per column instead of once per sample.
        for local_quart_x in 0..4i32 {
            for local_quart_z in 0..4i32 {
                let quart_x = chunk_x * 4 + local_quart_x;
                let quart_z = chunk_z * 4 + local_quart_z;

                for section_index in 0..section_count {
                    let section_y = (min_y / 16) + section_index as i32;
                    let section = &chunk.sections().sections[section_index];
                    let mut section_guard = section.write();

                    for local_quart_y in 0..4i32 {
                        let quart_y = section_y * 4 + local_quart_y;

                        let biome = sampler.sample(quart_x, quart_y, quart_z);
                        let biome_id = *REGISTRY.biomes.get_id(biome) as u16;

                        section_guard.biomes.set(
                            local_quart_x as usize,
                            local_quart_y as usize,
                            local_quart_z as usize,
                            biome_id,
                        );
                    }
                }
            }
        }

        chunk.mark_dirty();
    }

    fn fill_from_noise(&self, chunk: &ChunkAccess) {
        let pos = chunk.pos();
        let chunk_min_x = pos.0.x * 16;
        let chunk_min_z = pos.0.y * 16;

        let mut noise_chunk = NoiseChunk::new(chunk_min_x, chunk_min_z);
        let mut column_cache = OverworldColumnCache::new();

        let noises = &*self.noises;
        let stone_id = self.stone_id;
        let ore_veinifier = &self.ore_veinifier;
        let mut ore_cache = OverworldColumnCache::new();
        let mut aquifer = Aquifer::new(
            chunk_min_x,
            chunk_min_z,
            MIN_Y,
            384, // overworld height
            &self.splitter,
            noises,
        );

        noise_chunk.fill(
            noises,
            &mut column_cache,
            |local_x, world_y, local_z, density| {
                let relative_y = (world_y - MIN_Y) as usize;
                let world_x = chunk_min_x + local_x as i32;
                let world_z = chunk_min_z + local_z as i32;

                match aquifer.compute_substance(noises, world_x, world_y, world_z, density) {
                    AquiferResult::Solid => {
                        let block = ore_veinifier
                            .compute(noises, &mut ore_cache, world_x, world_y, world_z)
                            .unwrap_or(stone_id);
                        chunk.set_relative_block(local_x, relative_y, local_z, block);
                    }
                    AquiferResult::Fluid(id) => {
                        chunk.set_relative_block(local_x, relative_y, local_z, id);
                    }
                    AquiferResult::Air => {}
                }
            },
        );
    }

    fn build_surface(&self, _chunk: &ChunkAccess) {}

    fn apply_carvers(&self, _chunk: &ChunkAccess) {}

    fn apply_biome_decorations(&self, _chunk: &ChunkAccess) {}
}
