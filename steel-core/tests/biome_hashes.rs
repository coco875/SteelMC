//! Biome regression test.
//!
//! Verifies that Steel's biome generation matches vanilla Minecraft (seed 13579)
//! by comparing per-chunk MD5 hashes of biome names across all 24 sections.
//!
//! Hashes were extracted from a vanilla 1.21.1 world using world-gen-comp
//! `--extract-biome-hashes 5`.

use std::fmt::Write;
use std::thread;

use rustc_hash::FxHashMap;
use steel_core::worldgen::{EvalCache, VanillaClimateSampler};
use steel_registry::multi_noise::get_overworld_biome_cached;

const SEED: u64 = 13579;

/// Compute a biome MD5 hash for a chunk.
///
/// Samples biomes using vanilla's generation iteration order (X,Y,Z) with cache
/// so that tie-breaking matches actual world generation, then hashes in
/// deterministic Y,Z,X order with `section_y` markers for a stable digest.
fn chunk_biome_hash(sampler: &VanillaClimateSampler, chunk_x: i32, chunk_z: i32) -> String {
    // Step 1: Sample biomes in generation order (X outer, Y middle, Z inner)
    // so the cache tie-breaking matches vanilla/Steel world generation.
    let mut biomes = FxHashMap::default();
    let mut biome_cache: Option<usize> = None;
    let mut eval_cache = EvalCache::new();

    for section_y in -4i32..20 {
        for x in 0..4i32 {
            for y in 0..4i32 {
                for z in 0..4i32 {
                    let quart_x = chunk_x * 4 + x;
                    let quart_y = section_y * 4 + y;
                    let quart_z = chunk_z * 4 + z;

                    let target = sampler.sample(quart_x, quart_y, quart_z, &mut eval_cache);
                    let biome = get_overworld_biome_cached(&target, &mut biome_cache);
                    biomes.insert((section_y, x, y, z), biome);
                }
            }
        }
    }

    // Step 2: Hash in deterministic Y,Z,X order with section markers.
    let mut ctx = md5::Context::new();
    for section_y in -4i32..20 {
        ctx.consume([section_y as u8]);
        for y in 0..4i32 {
            for z in 0..4i32 {
                for x in 0..4i32 {
                    let biome = biomes[&(section_y, x, y, z)];
                    let name = biome.strip_prefix("minecraft:").unwrap_or(biome);
                    ctx.consume(name.as_bytes());
                }
            }
        }
    }

    format!("{:x}", ctx.finalize())
}

/// Expected vanilla biome hashes for seed 13579, radius 5 (-5..=5).
/// Extracted from vanilla 1.21.1 world via world-gen-comp --extract-biome-hashes.
const EXPECTED: &[(i32, i32, &str)] = &[
    (-5, -5, "bc342290df7b4487de3b0ce520913faf"),
    (-5, -4, "f29721a301db729af96c1aa1b7f4fc2f"),
    (-5, -3, "64c48792ebebadd20866cbf879b35c8d"),
    (-5, -2, "64c48792ebebadd20866cbf879b35c8d"),
    (-5, -1, "64c48792ebebadd20866cbf879b35c8d"),
    (-5, 0, "64c48792ebebadd20866cbf879b35c8d"),
    (-5, 1, "64c48792ebebadd20866cbf879b35c8d"),
    (-5, 2, "64c48792ebebadd20866cbf879b35c8d"),
    (-5, 3, "64c48792ebebadd20866cbf879b35c8d"),
    (-5, 4, "64c48792ebebadd20866cbf879b35c8d"),
    (-5, 5, "64c48792ebebadd20866cbf879b35c8d"),
    (-4, -5, "553b9ccdb26f930fb5932786391dbb9b"),
    (-4, -4, "17e2e46c53f580c6ae419869687ae92b"),
    (-4, -3, "5b3afb7e31be724e273956229534d05b"),
    (-4, -2, "64c48792ebebadd20866cbf879b35c8d"),
    (-4, -1, "64c48792ebebadd20866cbf879b35c8d"),
    (-4, 0, "64c48792ebebadd20866cbf879b35c8d"),
    (-4, 1, "64c48792ebebadd20866cbf879b35c8d"),
    (-4, 2, "64c48792ebebadd20866cbf879b35c8d"),
    (-4, 3, "64c48792ebebadd20866cbf879b35c8d"),
    (-4, 4, "64c48792ebebadd20866cbf879b35c8d"),
    (-4, 5, "64c48792ebebadd20866cbf879b35c8d"),
    (-3, -5, "28c03b73fc06b0f99b477a0216a83de2"),
    (-3, -4, "dfd2f61653d14ff842dfd84b4122d926"),
    (-3, -3, "17e2e46c53f580c6ae419869687ae92b"),
    (-3, -2, "019481970f80ccb40e938c3110c530de"),
    (-3, -1, "3839b894cc46445a596e503cb0b00809"),
    (-3, 0, "598ea3e1fa46a101f83bf4297c7beda1"),
    (-3, 1, "64c48792ebebadd20866cbf879b35c8d"),
    (-3, 2, "64c48792ebebadd20866cbf879b35c8d"),
    (-3, 3, "64c48792ebebadd20866cbf879b35c8d"),
    (-3, 4, "64c48792ebebadd20866cbf879b35c8d"),
    (-3, 5, "64c48792ebebadd20866cbf879b35c8d"),
    (-2, -5, "64c48792ebebadd20866cbf879b35c8d"),
    (-2, -4, "7ba04c81a2e1ca15f57c2ec3a95b611a"),
    (-2, -3, "dfd2f61653d14ff842dfd84b4122d926"),
    (-2, -2, "dfd2f61653d14ff842dfd84b4122d926"),
    (-2, -1, "dfd2f61653d14ff842dfd84b4122d926"),
    (-2, 0, "dfd2f61653d14ff842dfd84b4122d926"),
    (-2, 1, "23fec073a5c386fb5d4617effc08698d"),
    (-2, 2, "bc6ebc6bdf640acf036ad587e6eeac5d"),
    (-2, 3, "64c48792ebebadd20866cbf879b35c8d"),
    (-2, 4, "64c48792ebebadd20866cbf879b35c8d"),
    (-2, 5, "64c48792ebebadd20866cbf879b35c8d"),
    (-1, -5, "64c48792ebebadd20866cbf879b35c8d"),
    (-1, -4, "64c48792ebebadd20866cbf879b35c8d"),
    (-1, -3, "389ee9b65c60724d0cc605d6ca928333"),
    (-1, -2, "70dfd012f8ab60713ad32097c22e953d"),
    (-1, -1, "c7261ffee096927d0b75bbded8f06316"),
    (-1, 0, "dfd2f61653d14ff842dfd84b4122d926"),
    (-1, 1, "dfd2f61653d14ff842dfd84b4122d926"),
    (-1, 2, "dfd2f61653d14ff842dfd84b4122d926"),
    (-1, 3, "23fec073a5c386fb5d4617effc08698d"),
    (-1, 4, "64c48792ebebadd20866cbf879b35c8d"),
    (-1, 5, "64c48792ebebadd20866cbf879b35c8d"),
    (0, -5, "64c48792ebebadd20866cbf879b35c8d"),
    (0, -4, "64c48792ebebadd20866cbf879b35c8d"),
    (0, -3, "64c48792ebebadd20866cbf879b35c8d"),
    (0, -2, "64c48792ebebadd20866cbf879b35c8d"),
    (0, -1, "64c48792ebebadd20866cbf879b35c8d"),
    (0, 0, "c564a2513acdbc3ccd03f7a3afc2b73b"),
    (0, 1, "dfd2f61653d14ff842dfd84b4122d926"),
    (0, 2, "dfd2f61653d14ff842dfd84b4122d926"),
    (0, 3, "dfd2f61653d14ff842dfd84b4122d926"),
    (0, 4, "4fa25db3fa2639e6eda43fb7037a9150"),
    (0, 5, "8ab92218f82372136f95bd1b983cb5f7"),
    (1, -5, "64c48792ebebadd20866cbf879b35c8d"),
    (1, -4, "64c48792ebebadd20866cbf879b35c8d"),
    (1, -3, "64c48792ebebadd20866cbf879b35c8d"),
    (1, -2, "64c48792ebebadd20866cbf879b35c8d"),
    (1, -1, "64c48792ebebadd20866cbf879b35c8d"),
    (1, 0, "64c48792ebebadd20866cbf879b35c8d"),
    (1, 1, "bd231a17b27d430d50a63939e2ae4efc"),
    (1, 2, "dfd2f61653d14ff842dfd84b4122d926"),
    (1, 3, "9d94d310de5986e270e518dc56e29e2e"),
    (1, 4, "dfd2f61653d14ff842dfd84b4122d926"),
    (1, 5, "04825546fa002fbe3877759798e16bd8"),
    (2, -5, "64c48792ebebadd20866cbf879b35c8d"),
    (2, -4, "64c48792ebebadd20866cbf879b35c8d"),
    (2, -3, "601c6cc2892ffa99b5626253e3d8cabd"),
    (2, -2, "34b023c6d77ede9159ec373da146ee51"),
    (2, -1, "64c48792ebebadd20866cbf879b35c8d"),
    (2, 0, "64c48792ebebadd20866cbf879b35c8d"),
    (2, 1, "64c48792ebebadd20866cbf879b35c8d"),
    (2, 2, "f7bd07738b3e3827a2f0778ee8372f7e"),
    (2, 3, "64c48792ebebadd20866cbf879b35c8d"),
    (2, 4, "ac4fa1f8e714d9404a09dac447a3a045"),
    (2, 5, "06901394c8fed9b7811c6c41e41a7d95"),
    (3, -5, "64c48792ebebadd20866cbf879b35c8d"),
    (3, -4, "64c48792ebebadd20866cbf879b35c8d"),
    (3, -3, "a7ff7b5be6e5e2ac870a35ad865daf1a"),
    (3, -2, "56fdca71a830306d2677cd2c93bb2301"),
    (3, -1, "014b1c087e32ae0e9db0bb3b8e5a74fb"),
    (3, 0, "0a58a4b4e98494d9c17e9b964ad0545b"),
    (3, 1, "93df8ed1b2a1e1023e7bd5d3de6ce2f8"),
    (3, 2, "0df5b2be9fdb27f70e33d43ef1a4e3c5"),
    (3, 3, "64c48792ebebadd20866cbf879b35c8d"),
    (3, 4, "f7bd07738b3e3827a2f0778ee8372f7e"),
    (3, 5, "197b5ed167800687bcee93cd1df06115"),
    (4, -5, "64c48792ebebadd20866cbf879b35c8d"),
    (4, -4, "64c48792ebebadd20866cbf879b35c8d"),
    (4, -3, "7172a3e3e70e07c94e9f27ea89434a53"),
    (4, -2, "56fdca71a830306d2677cd2c93bb2301"),
    (4, -1, "56fdca71a830306d2677cd2c93bb2301"),
    (4, 0, "56fdca71a830306d2677cd2c93bb2301"),
    (4, 1, "56fdca71a830306d2677cd2c93bb2301"),
    (4, 2, "7afcac113b3432cdc02ad3a18a47eb68"),
    (4, 3, "ea24535689b91f3574e6f6a43b75c3a1"),
    (4, 4, "601c6cc2892ffa99b5626253e3d8cabd"),
    (4, 5, "8f60e92ef50cd40928f44fdb572c62cb"),
    (5, -5, "64c48792ebebadd20866cbf879b35c8d"),
    (5, -4, "64c48792ebebadd20866cbf879b35c8d"),
    (5, -3, "75505653670c616109ae5a08067b8732"),
    (5, -2, "56fdca71a830306d2677cd2c93bb2301"),
    (5, -1, "56fdca71a830306d2677cd2c93bb2301"),
    (5, 0, "56fdca71a830306d2677cd2c93bb2301"),
    (5, 1, "56fdca71a830306d2677cd2c93bb2301"),
    (5, 2, "56fdca71a830306d2677cd2c93bb2301"),
    (5, 3, "0a58a4b4e98494d9c17e9b964ad0545b"),
    (5, 4, "371d6ecb1f7d04a4a905cd249e1421c7"),
    (5, 5, "56fdca71a830306d2677cd2c93bb2301"),
];

#[test]
fn biome_hashes_match_vanilla() {
    // VanillaClimateSampler initialization has deep recursion; needs a large stack.
    let builder = thread::Builder::new().stack_size(16 * 1024 * 1024);
    let handle = builder
        .spawn(|| {
            let sampler = VanillaClimateSampler::new(SEED);

            let mut mismatches = Vec::new();

            for &(chunk_x, chunk_z, expected_hash) in EXPECTED {
                let actual_hash = chunk_biome_hash(&sampler, chunk_x, chunk_z);
                if actual_hash != expected_hash {
                    mismatches.push((chunk_x, chunk_z, expected_hash.to_string(), actual_hash));
                }
            }

            if !mismatches.is_empty() {
                let total = EXPECTED.len();
                let failed = mismatches.len();
                let mut msg = format!("{failed}/{total} chunks MISMATCHED:\n");
                for (x, z, expected, actual) in &mismatches {
                    let _ = writeln!(msg, "  ({x:3},{z:3}): expected {expected} got {actual}");
                }
                panic!("{msg}");
            }
        })
        .expect("failed to spawn test thread");

    handle.join().expect("test thread panicked");
}
