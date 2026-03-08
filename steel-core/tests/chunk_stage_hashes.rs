//! Chunk generation stage regression test.
//!
//! Verifies that Steel's chunk generation matches vanilla Minecraft at each stage
//! by comparing MD5 hashes of block data. Enable stages one at a time as they
//! are implemented.

use std::fmt::Write;

use rustc_hash::FxHashMap;
use serde::Deserialize;
use steel_core::chunk::section::Sections;

#[derive(Deserialize, Debug)]
struct ChunkStageEntry {
    x: i32,
    z: i32,
    stages: FxHashMap<String, String>,
}

#[derive(Deserialize, Debug)]
struct ChunkStageHashesJson {
    seed: u64,
    chunks: Vec<ChunkStageEntry>,
    #[allow(dead_code)]
    chunk_count: usize,
}

/// Stages to verify. Uncomment as each stage is implemented.
const STAGES: &[&str] = &[
    "minecraft:noise",
    "minecraft:surface",
    // "minecraft:carvers",
    // "minecraft:features",
];

fn load_expected_hashes() -> ChunkStageHashesJson {
    let json_str = include_str!("../test_assets/chunk_stage_hashes.json");
    serde_json::from_str(json_str).expect("Failed to parse chunk_stage_hashes.json")
}

fn compute_block_hash(sections: &Sections) -> String {
    let mut ctx = md5::Context::new();

    for section_holder in &sections.sections {
        let section = section_holder.read();
        if section.states.has_only_air() {
            ctx.consume([0u8]);
        } else {
            for y in 0..16 {
                for z in 0..16 {
                    for x in 0..16 {
                        let state = section.states.get(x, y, z);
                        let state_id = u32::from(state.0);
                        ctx.consume([(state_id >> 24) as u8]);
                        ctx.consume([(state_id >> 16) as u8]);
                        ctx.consume([(state_id >> 8) as u8]);
                        ctx.consume([state_id as u8]);
                    }
                }
            }
        }
    }

    format!("{:x}", ctx.finalize())
}

#[test]
fn chunk_stage_hashes() {
    use std::thread;

    // Run on a thread with a larger stack to avoid overflow in debug builds,
    // since pre-generating biome data for neighbor lookups increases stack usage.
    thread::Builder::new()
        .stack_size(16 * 1024 * 1024)
        .spawn(chunk_stage_hashes_inner)
        .expect("Failed to spawn test thread")
        .join()
        .expect("Test thread panicked");
}

#[allow(clippy::too_many_lines, clippy::similar_names)]
fn chunk_stage_hashes_inner() {
    use steel_core::chunk::chunk_access::ChunkAccess;
    use steel_core::chunk::chunk_generator::ChunkGenerator;
    use steel_core::chunk::proto_chunk::ProtoChunk;
    use steel_core::chunk::section::ChunkSection;
    use steel_core::chunk::world_gen_context::OverworldGenerator;
    use steel_core::worldgen::BiomeSourceKind;
    use steel_registry::{REGISTRY, Registry};
    use steel_utils::ChunkPos;

    let mut registry = Registry::new_vanilla();
    registry.freeze();
    let _ = REGISTRY.init(registry);

    let expected = load_expected_hashes();
    let seed = expected.seed;
    assert_eq!(seed, 13579, "Expected seed 13579");

    let source = BiomeSourceKind::overworld(seed);
    let generator = OverworldGenerator::new(source, seed);

    let section_count = 24;
    let min_y = -64;
    let height = 384;
    let min_qy = min_y >> 2;
    let total_quarts_y = section_count * 4;

    for &stage in STAGES {
        let stage_chunks: Vec<_> = expected
            .chunks
            .iter()
            .filter_map(|c| c.stages.get(stage).map(|hash| (c.x, c.z, hash.clone())))
            .collect();

        // Pre-generate biomes for neighbor lookups (needed for surface and later stages).
        // Vanilla reads out-of-chunk biomes from neighbor chunk palettes via WorldGenRegion;
        // we replicate this by pre-populating biome data for the extended grid (±1 chunk).
        let biome_chunks: FxHashMap<(i32, i32), ChunkAccess> = if stage == "minecraft:noise" {
            FxHashMap::default()
        } else {
            let (min_cx, max_cx, min_cz, max_cz) = stage_chunks.iter().fold(
                (i32::MAX, i32::MIN, i32::MAX, i32::MIN),
                |(mnx, mxx, mnz, mxz), (x, z, _)| {
                    (mnx.min(*x), mxx.max(*x), mnz.min(*z), mxz.max(*z))
                },
            );
            let mut map = FxHashMap::default();
            for x in (min_cx - 1)..=(max_cx + 1) {
                for z in (min_cz - 1)..=(max_cz + 1) {
                    let sections: Box<[ChunkSection]> = (0..section_count)
                        .map(|_| ChunkSection::new_empty())
                        .collect::<Vec<_>>()
                        .into_boxed_slice();
                    let proto = ProtoChunk::new(
                        Sections::from_owned(sections),
                        ChunkPos::new(x, z),
                        min_y,
                        height,
                    );
                    let chunk = ChunkAccess::Proto(proto);
                    generator.create_biomes(&chunk);
                    map.insert((x, z), chunk);
                }
            }
            map
        };

        let mut mismatches = Vec::new();

        for (chunk_x, chunk_z, expected_hash) in &stage_chunks {
            let sections: Box<[ChunkSection]> = (0..section_count)
                .map(|_| ChunkSection::new_empty())
                .collect::<Vec<_>>()
                .into_boxed_slice();

            let proto = ProtoChunk::new(
                Sections::from_owned(sections),
                ChunkPos::new(*chunk_x, *chunk_z),
                min_y,
                height,
            );

            let chunk = ChunkAccess::Proto(proto);

            // Apply prerequisite stages up to the target
            generator.create_biomes(&chunk);
            generator.fill_from_noise(&chunk);
            if stage != "minecraft:noise" {
                let neighbor_biomes = |qx: i32, qy: i32, qz: i32| -> u16 {
                    let cx = qx >> 2;
                    let cz = qz >> 2;
                    let neighbor = &biome_chunks[&(cx, cz)];
                    let sections = neighbor.sections();
                    let local_qx = (qx - cx * 4) as usize;
                    let local_qz = (qz - cz * 4) as usize;
                    let qy_clamped = (qy - min_qy).clamp(0, total_quarts_y - 1) as usize;
                    let section_idx = qy_clamped / 4;
                    let local_qy = qy_clamped % 4;
                    sections.sections[section_idx]
                        .read()
                        .biomes
                        .get(local_qx, local_qy, local_qz)
                };
                generator.build_surface(&chunk, &neighbor_biomes);
            }

            let actual_hash = compute_block_hash(chunk.sections());

            if actual_hash != *expected_hash {
                mismatches.push((*chunk_x, *chunk_z, expected_hash.clone(), actual_hash));
            }
        }

        if mismatches.is_empty() {
            continue;
        }

        let total = stage_chunks.len();
        let failed = mismatches.len();
        let mut msg = format!("{stage}: {failed}/{total} chunks do not match vanilla\n");
        for (x, z, expected, actual) in mismatches.iter().take(5) {
            let _ = writeln!(msg, "  ({x:3},{z:3}): expected {expected}, got {actual}");
        }
        if failed > 5 {
            let _ = writeln!(msg, "  ... and {} more", failed - 5);
        }

        panic!("{msg}");
    }
}
