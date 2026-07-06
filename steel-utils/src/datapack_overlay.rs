//! Build-time logical merge of vanilla datapack files with optional zip overlays.
//!
//! Layers (bottom to top):
//! 1. Extracted vanilla tree under [`VANILLA_DATAPACK_BASE`]
//! 2. Each `.zip` in the datapacks directory ([`DEFAULT_DATAPACKS_DIR`] or [`DATAPACKS_DIR_ENV`]),
//!    ordered by [`DATAPACKS_CONFIG_FILE`] when present, otherwise lexicographic filename order
//!
//! Within a namespace, later layers override the same logical path. Zip namespaces are never
//! merged into each other: `data/nova_structures/worldgen/structure/foo.json` is stored as
//! `nova_structures/worldgen/structure/foo.json` and registers as `nova_structures:foo`.
//!
//! [`load_minecraft`] overlays zip `minecraft` entries only.
//! [`load_minecraft_with_zip_namespaces`] loads vanilla `minecraft` plus every zip namespace.

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::io::Read;
use std::path::{Component, Path, PathBuf};

use serde::Deserialize;

/// Root of the extracted vanilla datapack tree (contains `minecraft/`).
pub const VANILLA_DATAPACK_BASE: &str = "../steel-utils/build_assets/builtin_datapacks";

/// Default repo-local folder of `.zip` datapacks when [`DATAPACKS_DIR_ENV`] is unset.
///
/// Relative to build-script working directory (`steel-registry/`, `steel-worldgen/`).
pub const DEFAULT_DATAPACKS_DIR: &str = "../datapacks";

/// Environment variable pointing at a folder of `.zip` datapacks to overlay.
///
/// Unset → use [`DEFAULT_DATAPACKS_DIR`] when that directory exists.
/// Set to empty string → disable zip overlays.
pub const DATAPACKS_DIR_ENV: &str = "STEEL_DATAPACKS_DIR";

/// TOML file in the datapacks directory that defines zip overlay order.
pub const DATAPACKS_CONFIG_FILE: &str = "datapacks.toml";

#[derive(Debug, Deserialize)]
struct DatapacksConfig {
    order: Vec<String>,
}

/// Datapack `pack_format` for the targeted Minecraft version (`mc26.2`).
pub const DATAPACK_FORMAT: u32 = 107;

#[derive(Debug, Deserialize)]
struct PackMeta {
    overlays: Option<PackOverlays>,
}

#[derive(Debug, Deserialize)]
struct PackOverlays {
    entries: Vec<PackOverlayEntry>,
}

#[derive(Debug, Deserialize)]
struct PackOverlayEntry {
    directory: String,
    #[serde(default)]
    formats: Option<[PackFormat; 2]>,
    #[serde(default)]
    min_format: Option<PackFormat>,
    #[serde(default)]
    max_format: Option<PackFormat>,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct PackFormat {
    major: u32,
    minor: u32,
}

impl PackFormat {
    const fn major(major: u32) -> Self {
        Self { major, minor: 0 }
    }
}

impl<'de> Deserialize<'de> for PackFormat {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum FormatJson {
            Major(u32),
            Version([u32; 2]),
        }

        match FormatJson::deserialize(deserializer)? {
            FormatJson::Major(major) => Ok(Self::major(major)),
            FormatJson::Version([major, minor]) => Ok(Self { major, minor }),
        }
    }
}

/// Logical view over vanilla plus optional zip datapack overlays.
pub struct DatapackOverlay {
    files: BTreeMap<String, Vec<DatapackFile>>,
    rerun_paths: Vec<PathBuf>,
}

#[derive(Clone, Debug)]
struct DatapackFile {
    content: Vec<u8>,
    source: DatapackFileSource,
}

/// Final source for a datapack file after overlay resolution.
#[derive(Clone, Debug)]
pub enum DatapackFileSource {
    /// File came from an unpacked datapack directory.
    Path(PathBuf),
    /// File came from an entry inside a datapack zip.
    Zip {
        /// Source zip path.
        zip_path: PathBuf,
        /// Entry path inside the zip archive.
        entry_name: String,
    },
}

/// Structure template NBT bytes plus the final overlay source that supplied them.
#[derive(Clone, Debug)]
pub struct StructureTemplateNbt {
    /// Compressed NBT bytes.
    pub bytes: Vec<u8>,
    /// Source that supplied the final overlaid file.
    pub source: DatapackFileSource,
}

impl DatapackOverlay {
    fn insert_file(&mut self, logical_path: String, file: DatapackFile) {
        self.files.entry(logical_path).or_default().push(file);
    }

    /// Load vanilla `minecraft` namespace with optional zip overlays.
    ///
    /// Emits `cargo:rerun-if-changed` for overlay inputs when not compiled for unit tests.
    #[must_use]
    pub fn load_minecraft() -> Self {
        Self::load_namespace("minecraft")
    }

    /// Load vanilla `minecraft` plus every zip namespace (each kept under its own namespace).
    ///
    /// Emits `cargo:rerun-if-changed` for overlay inputs when not compiled for unit tests.
    #[must_use]
    pub fn load_minecraft_with_zip_namespaces() -> Self {
        let mut overlay = Self {
            files: BTreeMap::new(),
            rerun_paths: Vec::new(),
        };

        let vanilla_root = Path::new(VANILLA_DATAPACK_BASE).join("minecraft");
        overlay.rerun_paths.push(vanilla_root.clone());
        overlay.load_directory(&vanilla_root, "minecraft");
        overlay.load_zip_directory_namespaces();
        overlay.finish_loading()
    }

    /// Load vanilla namespace data with optional zip overlays.
    ///
    /// Emits `cargo:rerun-if-changed` for overlay inputs when not compiled for unit tests.
    #[must_use]
    pub fn load_namespace(namespace: &str) -> Self {
        let mut overlay = Self {
            files: BTreeMap::new(),
            rerun_paths: Vec::new(),
        };

        let vanilla_root = Path::new(VANILLA_DATAPACK_BASE).join(namespace);
        overlay.rerun_paths.push(vanilla_root.clone());
        overlay.load_directory(&vanilla_root, namespace);

        if let Some(datapacks_dir) = Self::resolve_datapacks_dir() {
            overlay.load_zip_overlays(&datapacks_dir, namespace);
        }

        overlay.finish_loading()
    }

    /// # Panics
    ///
    /// Panics if [`DATAPACKS_DIR_ENV`] points to a missing directory, or is not valid UTF-8.
    #[must_use]
    pub fn resolve_datapacks_dir() -> Option<PathBuf> {
        match env::var(DATAPACKS_DIR_ENV) {
            Ok(value) if value.is_empty() => None,
            Ok(value) => {
                let path = PathBuf::from(value);
                assert!(
                    path.is_dir(),
                    "{} points at missing directory: {}",
                    DATAPACKS_DIR_ENV,
                    path.display()
                );
                Some(path)
            }
            Err(env::VarError::NotPresent) => {
                let default = Path::new(DEFAULT_DATAPACKS_DIR);
                if default.is_dir() {
                    Some(default.to_path_buf())
                } else {
                    None
                }
            }
            Err(env::VarError::NotUnicode(_)) => {
                panic!("{DATAPACKS_DIR_ENV} is not valid UTF-8");
            }
        }
    }

    fn load_zip_directory_namespaces(&mut self) {
        let Some(datapacks_dir) = Self::resolve_datapacks_dir() else {
            return;
        };
        self.rerun_paths.push(datapacks_dir.clone());
        let config_path = datapacks_dir.join(DATAPACKS_CONFIG_FILE);
        if config_path.is_file() {
            self.rerun_paths.push(config_path);
        }

        for zip_path in Self::ordered_zip_paths(&datapacks_dir) {
            self.rerun_paths.push(zip_path.clone());
            self.load_zip_file_all_namespaces(&zip_path);
        }
    }

    /// Number of logical datapack files currently loaded.
    #[must_use]
    pub fn file_count(&self) -> usize {
        self.files.len()
    }

    /// Emit `cargo:rerun-if-changed` for vanilla roots and overlay inputs.
    ///
    /// Called automatically by [`Self::load_namespace`] and
    /// [`Self::load_minecraft_with_zip_namespaces`] in non-test builds.
    pub fn emit_rerun_if_changed(&self) {
        for path in &self.rerun_paths {
            println!("cargo:rerun-if-changed={}", path.display());
        }
    }

    #[expect(clippy::allow_attributes, reason = "missing_const_for_fn is conditional on test config")]
    #[allow(clippy::missing_const_for_fn, reason = "missing_const_for_fn is conditional on test config")]
    fn finish_loading(self) -> Self {
        #[cfg(not(test))]
        self.emit_rerun_if_changed();
        self
    }

    /// Read a datapack file by logical path (e.g. `minecraft/worldgen/biome/plains.json`).
    #[must_use]
    pub fn read(&self, path: &str) -> Option<&[u8]> {
        self.files.get(path).and_then(|layers| layers.last().map(|file| file.content.as_slice()))
    }

    /// # Panics
    ///
    /// Panics if the read bytes are not valid UTF-8.
    #[must_use]
    pub fn read_string(&self, path: &str) -> Option<String> {
        self.read(path).map(|bytes| {
            String::from_utf8(bytes.to_vec())
                .unwrap_or_else(|error| panic!("datapack file {path} is not valid UTF-8: {error}"))
        })
    }

    /// # Panics
    ///
    /// Panics if the path is missing, or contains invalid UTF-8.
    #[must_use]
    pub fn read_string_required(&self, path: &str) -> String {
        self.read_string(path)
            .unwrap_or_else(|| panic!("missing datapack file: {path}"))
    }

    /// Read UTF-8 text directly from the extracted vanilla datapack, before overlays are applied.
    #[must_use]
    pub fn read_vanilla_string(path: &str) -> Option<String> {
        let path = Path::new(VANILLA_DATAPACK_BASE).join(path.trim_start_matches('/'));
        fs::read_to_string(path).ok()
    }

    /// List structure template NBT files keyed by registry id (`namespace:path/to/template`).
    #[must_use]
    pub fn list_structure_template_nbt(&self) -> BTreeMap<String, Vec<u8>> {
        self.list_structure_template_nbt_with_sources()
            .into_iter()
            .map(|(id, template)| (id, template.bytes))
            .collect()
    }

    /// # Panics
    ///
    /// Panics if internal prefix matching layouts are violated.
    #[must_use]
    pub fn list_structure_template_nbt_with_sources(
        &self,
    ) -> BTreeMap<String, StructureTemplateNbt> {
        const STRUCTURE_PREFIX: &str = "structure/";
        let mut entries = BTreeMap::new();

        for (path, layers) in &self.files {
            let Some(file) = layers.last() else {
                continue;
            };
            let ext_ok = Path::new(path)
                .extension()
                .and_then(|ext| ext.to_str())
                .is_some_and(|ext| ext.eq_ignore_ascii_case("nbt"));
            if !ext_ok {
                continue;
            }

            let Some((namespace, rest)) = path.split_once('/') else {
                continue;
            };
            let Some(relative) = rest.strip_prefix(STRUCTURE_PREFIX) else {
                continue;
            };
            let id = format!(
                "{namespace}:{}",
                relative
                    .strip_suffix(".nbt")
                    .unwrap_or(relative)
                    .replace('\\', "/")
            );

            entries.insert(
                id,
                StructureTemplateNbt {
                    bytes: file.content.clone(),
                    source: file.source.clone(),
                },
            );
        }

        entries
    }

    /// List `.json` files under `dir_prefix`, keyed by path relative to that directory without extension.
    ///
    /// Example: prefix `minecraft/worldgen/biome` -> key `plains` for `.../plains.json`.
    /// # Panics
    ///
    /// Panics if a file contains invalid UTF-8.
    #[must_use]
    pub fn list_json_relative(&self, dir_prefix: &str) -> BTreeMap<String, String> {
        let prefix = dir_prefix.trim_end_matches('/');
        let search_prefix = format!("{prefix}/");
        let mut entries = BTreeMap::new();

        for (path, layers) in &self.files {
            let Some(file) = layers.last() else {
                continue;
            };
            let ext_ok = Path::new(path)
                .extension()
                .and_then(|ext| ext.to_str())
                .is_some_and(|ext| ext.eq_ignore_ascii_case("json"));
            if !path.starts_with(&search_prefix) || !ext_ok {
                continue;
            }

            let relative = path
                .strip_prefix(&search_prefix)
                .unwrap_or_else(|| panic!("internal datapack path prefix mismatch: {path}"));
            let key = relative
                .strip_suffix(".json")
                .unwrap_or(relative)
                .replace('\\', "/");

            entries.insert(
                key,
                String::from_utf8(file.content.clone()).unwrap_or_else(|error| {
                    panic!("datapack file {path} is not valid UTF-8: {error}")
                }),
            );
        }

        entries
    }

    /// List `.json` files under any loaded namespace whose path ends with `path_suffix`.
    ///
    /// Keys are registry ids (`namespace:relative/path` without `.json`).
    /// # Panics
    ///
    /// Panics if any file does not match internal prefix layout.
    #[must_use]
    pub fn list_json_registry_ids_with_suffix(
        &self,
        path_suffix: &str,
    ) -> BTreeMap<String, String> {
        let suffix = path_suffix.trim_matches('/');
        let prefix = format!("{suffix}/");
        let mut entries = BTreeMap::new();

        for (path, layers) in &self.files {
            let Some(file) = layers.last() else {
                continue;
            };
            let ext_ok = Path::new(path)
                .extension()
                .and_then(|ext| ext.to_str())
                .is_some_and(|ext| ext.eq_ignore_ascii_case("json"));
            if !ext_ok {
                continue;
            }

            let Some((namespace, rest)) = path.split_once('/') else {
                continue;
            };
            if !rest.starts_with(&prefix) {
                continue;
            }

            let relative = rest
                .strip_prefix(&prefix)
                .unwrap_or_else(|| panic!("internal datapack path prefix mismatch: {path}"));
            let id = format!(
                "{namespace}:{}",
                relative
                    .strip_suffix(".json")
                    .unwrap_or(relative)
                    .replace('\\', "/")
            );

            entries.insert(
                id,
                String::from_utf8(file.content.clone()).unwrap_or_else(|error| {
                    panic!("datapack file {path} is not valid UTF-8: {error}")
                }),
            );
        }

        entries
    }

    /// List `.json` files under any loaded namespace whose path ends with `path_suffix`,
    /// returning the UTF-8 content of all layers in priority order (bottom to top).
    ///
    /// Keys are registry ids (`namespace:relative/path` without `.json`).
    /// # Panics
    ///
    /// Panics if any file is not valid UTF-8.
    #[must_use]
    pub fn list_json_registry_ids_with_suffix_all_layers(
        &self,
        path_suffix: &str,
    ) -> BTreeMap<String, Vec<String>> {
        let suffix = path_suffix.trim_matches('/');
        let prefix = format!("{suffix}/");
        let mut entries = BTreeMap::new();

        for (path, layers) in &self.files {
            let ext_ok = Path::new(path)
                .extension()
                .and_then(|ext| ext.to_str())
                .is_some_and(|ext| ext.eq_ignore_ascii_case("json"));
            if !ext_ok {
                continue;
            }

            let Some((namespace, rest)) = path.split_once('/') else {
                continue;
            };
            if !rest.starts_with(&prefix) {
                continue;
            }

            let relative = rest
                .strip_prefix(&prefix)
                .unwrap_or_else(|| panic!("internal datapack path prefix mismatch: {path}"));
            let id = format!(
                "{namespace}:{}",
                relative
                    .strip_suffix(".json")
                    .unwrap_or(relative)
                    .replace('\\', "/")
            );

            let contents: Vec<String> = layers
                .iter()
                .map(|file| {
                    String::from_utf8(file.content.clone()).unwrap_or_else(|error| {
                        panic!("datapack file {path} is not valid UTF-8: {error}")
                    })
                })
                .collect();

            entries.insert(id, contents);
        }

        entries
    }

    fn load_directory(&mut self, root: &Path, namespace: &str) {
        if !root.is_dir() {
            return;
        }

        self.walk_directory(root, root, namespace);
    }

    fn walk_directory(&mut self, root: &Path, current: &Path, namespace: &str) {
        let entries = fs::read_dir(current).unwrap_or_else(|error| {
            panic!(
                "failed to read datapack directory {}: {error}",
                current.display()
            )
        });

        for entry in entries {
            let entry = entry.unwrap_or_else(|error| {
                panic!(
                    "failed to read datapack directory entry in {}: {error}",
                    current.display()
                )
            });
            let path = entry.path();

            if path.is_dir() {
                self.walk_directory(root, &path, namespace);
                continue;
            }

            if path.extension().is_none_or(|ext| {
                ext.to_str()
                    .is_none_or(|ext| !Self::is_datapack_extension(ext))
            }) {
                continue;
            }

            let relative = path.strip_prefix(root).unwrap_or_else(|error| {
                panic!(
                    "datapack path {} is not under {}: {error}",
                    path.display(),
                    root.display()
                )
            });
            let logical_path = format!(
                "{namespace}/{}",
                relative
                    .to_str()
                    .unwrap_or_else(|| { panic!("non-UTF-8 datapack path: {}", path.display()) })
                    .replace('\\', "/")
            );

            let content = fs::read(&path).unwrap_or_else(|error| {
                panic!("failed to read datapack file {}: {error}", path.display())
            });
            self.insert_file(
                logical_path,
                DatapackFile {
                    content,
                    source: DatapackFileSource::Path(path),
                },
            );
        }
    }

    fn ordered_zip_paths(datapacks_dir: &Path) -> Vec<PathBuf> {
        let config_path = datapacks_dir.join(DATAPACKS_CONFIG_FILE);
        let available = Self::discover_zip_paths(datapacks_dir);

        let mut ordered = if config_path.is_file() {
            Self::ordered_zip_paths_from_config(datapacks_dir, &config_path, &available)
        } else {
            let mut paths: Vec<_> = available.into_values().collect();
            paths.sort();
            paths
        };

        ordered.dedup();
        ordered
    }

    fn discover_zip_paths(datapacks_dir: &Path) -> BTreeMap<String, PathBuf> {
        let mut zips = BTreeMap::new();

        for entry in fs::read_dir(datapacks_dir)
            .unwrap_or_else(|error| {
                panic!(
                    "failed to read datapacks directory {}: {error}",
                    datapacks_dir.display()
                )
            })
            .filter_map(Result::ok)
        {
            let path = entry.path();
            if path.extension().is_some_and(|ext| ext == "zip") {
                let file_name = path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or_else(|| {
                        panic!("non-UTF-8 zip filename in {}", datapacks_dir.display())
                    })
                    .to_string();
                zips.insert(file_name, path);
            }
        }

        zips
    }

    fn ordered_zip_paths_from_config(
        datapacks_dir: &Path,
        config_path: &Path,
        available: &BTreeMap<String, PathBuf>,
    ) -> Vec<PathBuf> {
        let config_text = fs::read_to_string(config_path).unwrap_or_else(|error| {
            panic!(
                "failed to read datapack config {}: {error}",
                config_path.display()
            )
        });
        let config: DatapacksConfig = toml::from_str(&config_text).unwrap_or_else(|error| {
            panic!(
                "failed to parse datapack config {}: {error}",
                config_path.display()
            )
        });

        let mut ordered = Vec::new();
        let mut seen = rustc_hash::FxHashSet::default();

        for entry in config.order {
            assert!(
                seen.insert(entry.clone()),
                "duplicate datapack entry in {}: {entry}",
                config_path.display()
            );

            let zip_path = Self::resolve_config_zip_path(datapacks_dir, config_path, &entry);
            assert!(
                zip_path.is_file(),
                "datapack listed in {} does not exist: {}",
                config_path.display(),
                zip_path.display()
            );
            ordered.push(zip_path);
        }

        let mut remaining: Vec<_> = available
            .iter()
            .filter(|(name, _)| !seen.contains(name.as_str()))
            .map(|(_, path)| path.clone())
            .collect();
        remaining.sort();

        ordered.extend(remaining);
        ordered
    }

    fn resolve_config_zip_path(datapacks_dir: &Path, config_path: &Path, entry: &str) -> PathBuf {
        let path = Path::new(entry);
        if path
            .components()
            .any(|component| matches!(component, Component::ParentDir | Component::RootDir))
        {
            panic!(
                "datapack order entry in {} must be a filename, not a path: {entry}",
                config_path.display()
            );
        }

        datapacks_dir.join(path)
    }

    fn load_zip_overlays(&mut self, datapacks_dir: &Path, namespace: &str) {
        self.rerun_paths.push(datapacks_dir.to_path_buf());
        let config_path = datapacks_dir.join(DATAPACKS_CONFIG_FILE);
        if config_path.is_file() {
            self.rerun_paths.push(config_path);
        }

        for zip_path in Self::ordered_zip_paths(datapacks_dir) {
            self.rerun_paths.push(zip_path.clone());
            self.load_zip_file(&zip_path, namespace);
        }
    }

    fn load_zip_file(&mut self, zip_path: &Path, namespace: &str) {
        self.load_zip_file_filtered(zip_path, |entry_namespace| entry_namespace == namespace);
    }

    fn load_zip_file_all_namespaces(&mut self, zip_path: &Path) {
        self.load_zip_file_filtered(zip_path, |_| true);
    }

    fn load_zip_file_filtered(
        &mut self,
        zip_path: &Path,
        include_namespace: impl Fn(&str) -> bool,
    ) {
        let file = fs::File::open(zip_path).unwrap_or_else(|error| {
            panic!(
                "failed to open datapack zip {}: {error}",
                zip_path.display()
            )
        });
        let mut archive = zip::ZipArchive::new(file).unwrap_or_else(|error| {
            panic!(
                "failed to read datapack zip {}: {error}",
                zip_path.display()
            )
        });

        let mut layers = vec![Self::collect_zip_data_layer(
            &mut archive,
            zip_path,
            "",
            &include_namespace,
        )];

        for overlay_dir in Self::overlay_directories_for_zip(&mut archive) {
            layers.push(Self::collect_zip_data_layer(
                &mut archive,
                zip_path,
                &format!("{overlay_dir}/"),
                &include_namespace,
            ));
        }

        for layer in layers {
            for (path, file) in layer {
                self.insert_file(path, file);
            }
        }
    }

    fn overlay_directories_for_zip(archive: &mut zip::ZipArchive<fs::File>) -> Vec<String> {
        let Some(pack_meta) = Self::read_zip_text_entry(archive, "pack.mcmeta") else {
            return Vec::new();
        };
        let pack_meta: PackMeta = serde_json::from_str(&pack_meta).unwrap_or_else(|error| {
            panic!("failed to parse pack.mcmeta in datapack zip: {error}");
        });
        let Some(overlays) = pack_meta.overlays else {
            return Vec::new();
        };

        overlays
            .entries
            .into_iter()
            .filter(|entry| Self::overlay_applies(entry, DATAPACK_FORMAT))
            .map(|entry| entry.directory)
            .collect()
    }

    fn overlay_applies(entry: &PackOverlayEntry, format: u32) -> bool {
        let format = PackFormat::major(format);
        if let Some([min, max]) = entry.formats {
            return format >= min && format <= max;
        }

        let min = entry.min_format.unwrap_or(PackFormat::major(0));
        let max = entry.max_format.unwrap_or(PackFormat::major(u32::MAX));
        format >= min && format <= max
    }

    fn read_zip_text_entry(archive: &mut zip::ZipArchive<fs::File>, name: &str) -> Option<String> {
        let mut entry = archive.by_name(name).ok()?;
        if entry.is_dir() {
            return None;
        }

        let mut content = String::new();
        entry
            .read_to_string(&mut content)
            .unwrap_or_else(|error| panic!("failed to read {name} from datapack zip: {error}"));
        Some(content)
    }

    fn collect_zip_data_layer(
        archive: &mut zip::ZipArchive<fs::File>,
        zip_path: &Path,
        zip_prefix: &str,
        include_namespace: &impl Fn(&str) -> bool,
    ) -> BTreeMap<String, DatapackFile> {
        let data_prefix = format!("{zip_prefix}data/");
        let mut layer = BTreeMap::new();

        for index in 0..archive.len() {
            let mut entry = archive.by_index(index).unwrap_or_else(|error| {
                panic!("failed to read zip entry in datapack archive: {error}")
            });
            if entry.is_dir() || entry.name().ends_with('/') {
                continue;
            }

            let name = entry.name().to_string();
            if !name.starts_with(&data_prefix) {
                continue;
            }

            let relative = name
                .strip_prefix(&data_prefix)
                .unwrap_or_else(|| panic!("zip entry missing data/ prefix: {name}"));
            let Some(entry_namespace) = relative.split('/').next() else {
                continue;
            };
            if !include_namespace(entry_namespace) {
                continue;
            }
            let ext_ok = Path::new(relative)
                .extension()
                .and_then(|ext| ext.to_str())
                .is_some_and(Self::is_datapack_extension);
            if !ext_ok {
                continue;
            }

            let mut content = Vec::new();
            entry
                .read_to_end(&mut content)
                .unwrap_or_else(|error| panic!("failed to read {name} from datapack zip: {error}"));
            layer.insert(
                relative.replace('\\', "/"),
                DatapackFile {
                    content,
                    source: DatapackFileSource::Zip {
                        zip_path: zip_path.to_path_buf(),
                        entry_name: name,
                    },
                },
            );
        }

        layer
    }

    const fn is_datapack_extension(ext: &str) -> bool {
        ext.eq_ignore_ascii_case("json") || ext.eq_ignore_ascii_case("nbt")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lists_vanilla_density_functions() {
        let overlay = DatapackOverlay::load_minecraft_with_zip_namespaces();
        assert!(
            overlay.file_count() > 0,
            "overlay loaded no files; vanilla root missing?"
        );
        let entries = overlay.list_json_registry_ids_with_suffix("worldgen/density_function");
        assert!(
            entries.len() > 30,
            "expected vanilla density functions, got {} keys: {:?}",
            entries.len(),
            entries.keys().take(5).collect::<Vec<_>>()
        );
    }

    #[test]
    fn datapacks_toml_controls_order_before_unlisted_zips() {
        let Some(datapacks_dir) = DatapackOverlay::resolve_datapacks_dir() else {
            return;
        };

        let config_path = datapacks_dir.join(DATAPACKS_CONFIG_FILE);
        if !config_path.is_file() {
            return;
        }

        let ordered = DatapackOverlay::ordered_zip_paths(&datapacks_dir);
        let config: DatapacksConfig =
            toml::from_str(&fs::read_to_string(&config_path).expect("read datapacks.toml"))
                .expect("parse datapacks.toml");

        for (index, entry) in config.order.iter().enumerate() {
            assert_eq!(
                ordered[index].file_name().and_then(|name| name.to_str()),
                Some(entry.as_str()),
                "datapacks.toml order mismatch at index {index}"
            );
        }
    }
}
