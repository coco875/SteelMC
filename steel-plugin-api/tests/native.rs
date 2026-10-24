//! Integration against an independently built native example library.
#![cfg(feature = "host")]

use rayon::ThreadPoolBuilder;
use std::{
    env::consts::{DLL_PREFIX, DLL_SUFFIX},
    path::PathBuf,
    ptr,
    sync::Arc,
};
use steel_core::behavior::{BLOCK_BEHAVIORS, BlockBehaviorRegistry};
use steel_core::bootstrap::init_globals;
use steel_core::{extension::ExtensionHooks, worldgen::WorldGeneratorRegistry};
use steel_plugin_api::host;
use steel_registry::{REGISTRY, vanilla_blocks};
use steel_utils::locks::SyncMutex;
use toml::{Value, map::Map};

/// Build the example first, then run with `--ignored` to test real native loading.
#[test]
#[ignore = "requires cargo build --manifest-path examples/exotic-plugin/Cargo.toml"]
fn loads_separately_compiled_example_and_resolves_its_generator() -> Result<(), String> {
    let file = format!("{DLL_PREFIX}steel_exotic_plugin{DLL_SUFFIX}");
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../examples/exotic-plugin/target/debug")
        .join(file);
    let config = "[life]\nworld = 'minecraft:overworld'\nx = 0\ny = 96\nz = 0\nwidth = 5\ndepth = 5\ninterval = 10\n";
    let captured = Arc::new(SyncMutex::new(None));
    let capture = Arc::clone(&captured);
    BlockBehaviorRegistry::on_registration(move |registry| {
        *capture.lock() = Some(registry.get_behavior_shared(&vanilla_blocks::WHITE_WOOL));
    })?;
    let mut hooks = ExtensionHooks::default();
    // SAFETY: the example supplies the checked v2 entrypoints; load pins the library.
    let plugin = unsafe { host::load(&path, config, &mut hooks) }?;
    assert_eq!(plugin.name, "exotic-example");
    assert!(!REGISTRY.is_initialized());
    assert!(BLOCK_BEHAVIORS.get().is_none());
    assert!(
        steel_plugin_api::default_block_state("minecraft:stone".into())
            .match_owned(|_| false, |_| true)
    );
    init_globals();
    let registry = WorldGeneratorRegistry::new_with_builtins()?;
    assert_eq!(host::initialize_all(vec![plugin])?, ["exotic-example"]);
    let original = captured
        .lock()
        .take()
        .ok_or("block registration did not run")?;
    // SAFETY: reloading the same trusted module is rejected before pre-init repeats.
    assert!(unsafe { host::load(&path, config, &mut hooks) }.is_err());
    let replacement = BLOCK_BEHAVIORS.get_behavior(&vanilla_blocks::WHITE_WOOL);
    assert!(!ptr::eq(original.as_ref(), replacement));
    assert_eq!(replacement.is_bed(), original.is_bed());
    assert_eq!(
        replacement.fits_inside_container_items(),
        original.fits_inside_container_items()
    );
    let key = "example:strange".parse().map_err(str::to_owned)?;
    let empty = Value::Table(Map::new());
    let config = registry.validate_config(&key, &empty)?;
    let pool = Arc::new(
        ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .map_err(|e| e.to_string())?,
    );
    registry.create(None, &config, 12345, pool)?;
    let unexpected = toml::from_str("unexpected = true").map_err(|e| e.to_string())?;
    assert!(registry.validate_config(&key, &unexpected).is_err());
    Ok(())
}
