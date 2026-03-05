//! Dumps density values at specific positions for comparison with vanilla.
//!
//! Run with: cargo test -p steel-core --test density_debug -- --nocapture

use steel_registry::density_functions::OverworldNoises;
use steel_registry::noise_parameters::get_noise_parameters;
use steel_registry::{REGISTRY, Registry};
use steel_utils::density::{DimensionNoises, NoiseSettings};
use steel_utils::random::{Random, RandomSplitter, xoroshiro::Xoroshiro};

fn init() -> (OverworldNoises, RandomSplitter) {
    let mut registry = Registry::new_vanilla();
    registry.freeze();
    let _ = REGISTRY.init(registry);

    let seed: u64 = 13579;
    let mut rng = Xoroshiro::from_seed(seed);
    let splitter = rng.next_positional();
    let noise_params = get_noise_parameters();
    let noises = OverworldNoises::create(seed, &splitter, &noise_params);
    (noises, splitter)
}

type OWCache = <OverworldNoises as DimensionNoises>::ColumnCache;
type OWSettings = <OverworldNoises as DimensionNoises>::Settings;

#[test]
fn dump_density_at_positions() {
    let (noises, _splitter) = init();
    let mut cache = OWCache::default();

    let cell_width = OWSettings::CELL_WIDTH;
    let cell_height = OWSettings::CELL_HEIGHT;

    println!("\n=== Overworld Noise Settings ===");
    println!("  cell_width={cell_width}, cell_height={cell_height}");
    println!("  min_y={}, height={}, sea_level={}", OWSettings::MIN_Y, OWSettings::HEIGHT, OWSettings::SEA_LEVEL);

    // Dump density at cell corners for chunk (3,4) — the worst matching chunk
    let chunk_x: i32 = 3;
    let chunk_z: i32 = 4;
    let chunk_block_x = chunk_x * 16;
    let chunk_block_z = chunk_z * 16;

    println!("\n=== Cell corner densities for chunk ({chunk_x},{chunk_z}) ===");
    println!("  Block origin: ({chunk_block_x}, {chunk_block_z})");

    let first_cell_x = chunk_block_x.div_euclid(cell_width);
    let first_cell_z = chunk_block_z.div_euclid(cell_width);
    let cell_count_xz = (16 / cell_width) as usize;

    // Sample a few Y levels around the surface (Y=45-65) and underground (Y=-20 to 0)
    let y_levels: Vec<i32> = (-24..=72).step_by(cell_height as usize).collect();

    for cx in 0..=cell_count_xz {
        for cz in 0..=cell_count_xz {
            let block_x = (first_cell_x + cx as i32) * cell_width;
            let block_z = (first_cell_z + cz as i32) * cell_width;

            cache.ensure(block_x, block_z, &noises);

            println!("\n  Column ({block_x}, {block_z}):");
            for &y in &y_levels {
                let density = noises.router_final_density(&mut cache, block_x, y, block_z);
                let marker = if density.abs() < 0.1 { " <-- near boundary" } else { "" };
                println!("    y={y:4}: {density:>20.15}{marker}");
            }
        }
    }
}

#[test]
fn dump_intermediate_values() {
    let (noises, _splitter) = init();
    let mut cache = OWCache::default();

    // Pick a single position in the worst chunk to compare intermediate density functions
    let positions = [
        (48, 50, 64),  // chunk (3,4), near surface
        (48, 0, 64),   // chunk (3,4), underground
        (48, -20, 64), // chunk (3,4), deep underground
        (0, 50, 0),    // origin, near surface
        (0, 0, 0),     // origin
    ];

    println!("\n=== Intermediate density function values ===");
    for (x, y, z) in positions {
        cache.ensure(x, z, &noises);

        let final_d = noises.router_final_density(&mut cache, x, y, z);
        let depth = noises.router_depth(&mut cache, x, y, z);
        let erosion = noises.router_erosion(&mut cache, x, y, z);
        let continentalness = noises.router_continentalness(&mut cache, x, y, z);
        let ridges = noises.router_ridges(&mut cache, x, y, z);
        let temperature = noises.router_temperature(&mut cache, x, y, z);
        let vegetation = noises.router_vegetation(&mut cache, x, y, z);
        let prelim_surface = noises.router_preliminary_surface_level(&mut cache, x, y, z);

        println!("\n  Position ({x}, {y}, {z}):");
        println!("    final_density:    {final_d:>22.17}");
        println!("    depth:            {depth:>22.17}");
        println!("    erosion:          {erosion:>22.17}");
        println!("    continentalness:  {continentalness:>22.17}");
        println!("    ridges:           {ridges:>22.17}");
        println!("    temperature:      {temperature:>22.17}");
        println!("    vegetation:       {vegetation:>22.17}");
        println!("    prelim_surface:   {prelim_surface:>22.17}");
    }
}

/// Verify that at cell corners, combine_interpolated(fill_cell_corner_densities(pos))
/// equals router_final_density(pos). This must be exact — no tolerance.
#[test]
fn verify_interpolation_split_identity() {
    let (noises, _splitter) = init();
    let mut cache = OWCache::default();

    let cell_width = OWSettings::CELL_WIDTH;
    let cell_height = OWSettings::CELL_HEIGHT;
    let interp_count = OverworldNoises::interpolated_count();

    println!("\n=== Interpolation split identity test ===");
    println!("  INTERPOLATED_COUNT = {interp_count}");

    let chunk_x: i32 = 3;
    let chunk_z: i32 = 4;
    let chunk_block_x = chunk_x * 16;
    let chunk_block_z = chunk_z * 16;

    let first_cell_x = chunk_block_x.div_euclid(cell_width);
    let first_cell_z = chunk_block_z.div_euclid(cell_width);
    let cell_count_xz = (16 / cell_width) as usize;

    let y_levels: Vec<i32> = (-64..=320).step_by(cell_height as usize).collect();

    let mut max_error = 0.0f64;
    let mut total_tests = 0u64;
    let mut failures = 0u64;

    for cx in 0..=cell_count_xz {
        for cz in 0..=cell_count_xz {
            let block_x = (first_cell_x + cx as i32) * cell_width;
            let block_z = (first_cell_z + cz as i32) * cell_width;

            cache.ensure(block_x, block_z, &noises);

            for &y in &y_levels {
                total_tests += 1;

                // Method 1: Direct router_final_density
                let direct = noises.router_final_density(&mut cache, block_x, y, block_z);

                // Method 2: Split — fill inner values, then combine
                let mut inner_values = vec![0.0f64; interp_count];
                noises.fill_cell_corner_densities(&mut cache, block_x, y, block_z, &mut inner_values);
                let combined = noises.combine_interpolated(&mut cache, &inner_values, block_x, y, block_z);

                let error = (direct - combined).abs();
                max_error = max_error.max(error);

                if error > 1e-15 {
                    if failures < 10 {
                        println!("  MISMATCH at ({block_x}, {y}, {block_z}): direct={direct:.17}, combined={combined:.17}, error={error:.2e}");
                        println!("    inner_values: {inner_values:?}");
                    }
                    failures += 1;
                }
            }
        }
    }

    println!("\n  Results: {total_tests} positions tested, {failures} mismatches, max_error={max_error:.2e}");
    assert_eq!(failures, 0, "Interpolation split is not identity-preserving at cell corners");
}
