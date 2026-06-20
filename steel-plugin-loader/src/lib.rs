//! Dynamic plugin loader for `SteelMC` using `stabby` for stable FFI.

use libloading::Library;
use rustc_hash::FxHashMap;
use stabby::libloading::{StabbyLibrary, Symbol};
use std::env::consts::{ARCH, OS};
use std::fs;
use std::fs::File;
use std::io::{Read, copy};
use std::path::Path;
use steel_plugin_api::{InitPluginFn, InitResult, PluginInitContext};
use zip::ZipArchive;

/// Host-side hook registry implementation for WordPress-style actions and filters.
pub mod hook;

/// Structure representing `mods.toml` config within a plugin zip.
#[derive(serde::Deserialize, Debug, Clone)]
pub struct ModsToml {
    /// The package config.
    pub package: PackageConfig,
    /// The targets mapping.
    pub targets: FxHashMap<String, String>,
}

/// Package details from `mods.toml`.
#[derive(serde::Deserialize, Debug, Clone)]
pub struct PackageConfig {
    /// The name of the package.
    pub name: String,
    /// The version of the package.
    pub version: String,
}

/// A loaded plugin along with its dynamic library handle.
///
/// Declaring `plugin` before `library` ensures that the plugin instance is
/// dropped *before* the library is unloaded, preventing undefined behavior
/// from calling code or dropping objects from an unloaded library.
pub struct LoadedPlugin {
    /// The FFI-stable plugin instance.
    pub properties: ModsToml,
    /// The handle to the loaded dynamic library, wrapped in Arc to share and persist.
    pub library: Library,
}

/// Scans `./plugins`, loads all compatible zip plugins, and returns the loaded plugins.
pub fn load_plugins(registry_api: steel_plugin_api::hook::PluginRegistryApiVtableRef) -> Vec<LoadedPlugin> {
    let mut loaded = Vec::new();
    let plugins_dir = Path::new("plugins");
    let temp_dir = Path::new(".tmp/plugins");

    let _ = fs::create_dir_all(plugins_dir);
    let _ = fs::remove_dir_all(temp_dir);
    if let Err(e) = fs::create_dir_all(temp_dir) {
        log::error!("Failed to create temporary plugins directory: {e}");
        return loaded;
    }

    let Ok(entries) = fs::read_dir(plugins_dir) else {
        log::error!("Failed to read plugins directory");
        return loaded;
    };

    log::info!("Scanning `./plugins` for *.zip plugins... ");

    for entry in entries.filter_map(Result::ok) {
        let path = entry.path();
        if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("zip") {
            let name = path.file_name().unwrap_or_default().to_string_lossy();
            match load_zip_plugin(&path, temp_dir, registry_api) {
                Ok(plugin) => {
                    log::info!("Successfully loaded plugin from {name}");
                    loaded.push(plugin);
                }
                Err(e) => {
                    log::error!("Failed to load plugin {name}: {e}");
                }
            }
        }
    }

    loaded
}

/// Extracts and loads a single zip plugin.
fn load_zip_plugin(
    zip_path: &Path,
    temp_dir: &Path,
    registry_api: steel_plugin_api::hook::PluginRegistryApiVtableRef,
) -> Result<LoadedPlugin, String> {
    let file = File::open(zip_path).map_err(|e| format!("failed to open plugin zip: {e}"))?;
    let mut archive =
        ZipArchive::new(file).map_err(|e| format!("failed to read zip archive: {e}"))?;

    // Read mods.toml
    let mut mods_toml_file = archive
        .by_name("mods.toml")
        .map_err(|_| "missing mods.toml at root")?;

    let mut mods_toml_content = String::new();
    let read_res = mods_toml_file.read_to_string(&mut mods_toml_content);
    drop(mods_toml_file);
    read_res.map_err(|e| format!("failed to read mods.toml: {e}"))?;

    // Parse mods.toml
    let mods_config: ModsToml = toml::from_str(&mods_toml_content)
        .map_err(|e| format!("failed to parse mods.toml: {e}"))?;

    // Determine target
    let target_key = format!("{OS}-{ARCH}");
    let lib_path_in_zip = mods_config
        .targets
        .get(&target_key)
        .ok_or_else(|| format!("unsupported target {target_key}"))?;

    // Extract library
    let mut lib_file_in_zip = archive
        .by_name(lib_path_in_zip)
        .map_err(|_| format!("library file '{lib_path_in_zip}' not found in zip"))?;

    let dest_dir = temp_dir.join(&mods_config.package.name);
    fs::create_dir_all(&dest_dir).map_err(|e| format!("failed to create extract dir: {e}"))?;

    let lib_filename = Path::new(lib_path_in_zip)
        .file_name()
        .ok_or("invalid library filename in zip")?;
    let dest_path = dest_dir.join(lib_filename);

    let mut dest_file =
        File::create(&dest_path).map_err(|e| format!("failed to create temporary file: {e}"))?;
    copy(&mut lib_file_in_zip, &mut dest_file)
        .map_err(|e| format!("failed to extract library: {e}"))?;

    // Load plugin
    log::info!("Loading dynamic library from {}", dest_path.display());
    
    let plugin_id = steel_plugin_api::AbiString::from(mods_config.package.name.clone());
    let registry = hook::get_host_registry();
    let host_api = hook::HookApiVtableRef(registry.into());
    let init_ctx = PluginInitContext {
        hook_api: steel_plugin_api::hook::HookApi {
            host_api,
            plugin_id,
            registry: steel_plugin_api::hook::PluginRegistryApi {
                api: registry_api,
            },
        },
    };

    // SAFETY: Loading dynamic libraries is unsafe, but we trust the plugin zip.
    let lib = unsafe { load_plugin(&dest_path, init_ctx) }?;
    Ok(LoadedPlugin {
        properties: mods_config,
        library: lib,
    })
}

/// Loads a single dynamic library and constructs the plugin using stabby's safe signature check.
///
/// # Safety
/// Loading raw dynamic libraries is inherently unsafe. `stabby` mitigates this
/// by performing runtime type compatibility checks on the symbol signature.
pub unsafe fn load_plugin(path: &Path, ctx: PluginInitContext) -> Result<Library, String> {
    // SAFETY: Loading a raw dynamic library is unsafe, but required for plugin loader.
    let library =
        unsafe { Library::new(path).map_err(|e| format!("failed to load dynamic library: {e}"))? };

    // Load the FFI-stable constructor function.
    // We use stabby::libloading::Symbol to match get_stabbied's return type.
    // SAFETY: Signature checking at runtime ensures the constructor function matches the expected signature.
    let constructor: Symbol<'_, InitPluginFn> = unsafe {
        library
            .get_stabbied(b"init_plugin")
            .map_err(|e| format!("failed to load or verify 'create_plugin' symbol ABI: {e:?}"))?
    };

    let res = constructor(ctx);

    match res {
        InitResult::Ok => Ok(library),
        InitResult::Panic => Err("The mod don't have init correctly".to_string()),
    }
}
