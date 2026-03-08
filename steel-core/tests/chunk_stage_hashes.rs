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

    for &stage in STAGES {
        let stage_chunks: Vec<_> = expected
            .chunks
            .iter()
            .filter_map(|c| c.stages.get(stage).map(|hash| (c.x, c.z, hash.clone())))
            .collect();

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
                generator.build_surface(&chunk);
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
