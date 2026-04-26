//! Runtime registry for world generator factories.

use rustc_hash::FxHashMap;
use serde::Deserialize;
use steel_registry::dimension_type::DimensionTypeRef;
use steel_registry::vanilla_dimension_types::{OVERWORLD, THE_END, THE_NETHER};
use steel_registry::{REGISTRY, RegistryExt};
use steel_utils::Identifier;

use crate::worldgen::{
    BiomeSourceKind, ChunkGeneratorType, EmptyChunkGenerator, FlatChunkGenerator, VanillaGenerator,
};

/// Fully constructed generator metadata for a world.
pub struct GeneratorOutput {
    /// Dimension rules used by this world.
    pub dimension: DimensionTypeRef,
    /// Chunk generator instance.
    pub generator: ChunkGeneratorType,
    /// Whether the client should treat this as a flat world.
    pub is_flat: bool,
    /// Sea level sent in login/respawn packets.
    pub sea_level: i32,
}

struct WorldGeneratorFactory {
    validate: fn(&toml::Value) -> Result<(), String>,
    create: fn(&toml::Value, i64) -> Result<GeneratorOutput, String>,
}

/// Registry of server-side world generator factories.
pub struct WorldGeneratorRegistry {
    factories: FxHashMap<Identifier, WorldGeneratorFactory>,
}

impl WorldGeneratorRegistry {
    /// Creates a registry containing Steel's built-in generator factories.
    pub fn new_with_builtins() -> Result<Self, String> {
        let mut registry = Self {
            factories: FxHashMap::default(),
        };

        registry.register(
            Identifier::vanilla_static("overworld"),
            WorldGeneratorFactory {
                validate: validate_empty_config,
                create: create_overworld,
            },
        )?;
        registry.register(
            Identifier::vanilla_static("the_nether"),
            WorldGeneratorFactory {
                validate: validate_empty_config,
                create: create_nether,
            },
        )?;
        registry.register(
            Identifier::vanilla_static("the_end"),
            WorldGeneratorFactory {
                validate: validate_empty_config,
                create: create_end,
            },
        )?;
        registry.register(
            Identifier::vanilla_static("flat"),
            WorldGeneratorFactory {
                validate: validate_flat_config,
                create: create_flat,
            },
        )?;
        registry.register(
            Identifier::new("steel", "empty"),
            WorldGeneratorFactory {
                validate: validate_empty_world_config,
                create: create_empty,
            },
        )?;

        Ok(registry)
    }

    fn register(&mut self, key: Identifier, factory: WorldGeneratorFactory) -> Result<(), String> {
        if self.factories.insert(key.clone(), factory).is_some() {
            return Err(format!("duplicate world generator registration {key}"));
        }
        Ok(())
    }

    /// Validates config for a generator ID.
    pub fn validate_config(&self, key: &Identifier, config: &toml::Value) -> Result<(), String> {
        let factory = self
            .factories
            .get(key)
            .ok_or_else(|| format!("unknown world generator {key}"))?;
        (factory.validate)(config)
    }

    /// Creates a generator from a validated generator ID and config.
    pub fn create(
        &self,
        key: &Identifier,
        config: &toml::Value,
        seed: i64,
    ) -> Result<GeneratorOutput, String> {
        let factory = self
            .factories
            .get(key)
            .ok_or_else(|| format!("unknown world generator {key}"))?;
        (factory.create)(config, seed)
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DimensionOnlyConfig {
    dimension: Identifier,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FlatGeneratorConfig {
    #[serde(default = "default_flat_dimension")]
    dimension: Identifier,
    #[serde(default = "default_flat_layers")]
    layers: Vec<FlatLayerConfig>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FlatLayerConfig {
    block: Identifier,
    height: usize,
}

fn default_flat_dimension() -> Identifier {
    Identifier::vanilla_static("overworld")
}

fn default_flat_layers() -> Vec<FlatLayerConfig> {
    vec![
        FlatLayerConfig {
            block: Identifier::vanilla_static("bedrock"),
            height: 1,
        },
        FlatLayerConfig {
            block: Identifier::vanilla_static("dirt"),
            height: 2,
        },
        FlatLayerConfig {
            block: Identifier::vanilla_static("grass_block"),
            height: 1,
        },
    ]
}

fn validate_empty_config(config: &toml::Value) -> Result<(), String> {
    let Some(table) = config.as_table() else {
        return Err("generator config must be a table".to_owned());
    };
    if !table.is_empty() {
        return Err("this generator does not accept config".to_owned());
    }
    Ok(())
}

fn validate_empty_world_config(config: &toml::Value) -> Result<(), String> {
    let parsed: DimensionOnlyConfig = config
        .clone()
        .try_into()
        .map_err(|e| format!("invalid steel:empty config: {e}"))?;
    dimension_by_key(&parsed.dimension).map(|_| ())
}

fn validate_flat_config(config: &toml::Value) -> Result<(), String> {
    let parsed = parse_flat_config(config)?;
    if parsed.layers.is_empty() {
        return Err("minecraft:flat requires at least one layer".to_owned());
    }
    dimension_by_key(&parsed.dimension)?;
    for layer in &parsed.layers {
        if layer.height == 0 {
            return Err("minecraft:flat layer height must be greater than zero".to_owned());
        }
        if REGISTRY.blocks.by_key(&layer.block).is_none() {
            return Err(format!(
                "unknown block {} in minecraft:flat layer",
                layer.block
            ));
        }
    }
    Ok(())
}

fn parse_flat_config(config: &toml::Value) -> Result<FlatGeneratorConfig, String> {
    config
        .clone()
        .try_into()
        .map_err(|e| format!("invalid minecraft:flat config: {e}"))
}

fn create_overworld(config: &toml::Value, seed: i64) -> Result<GeneratorOutput, String> {
    validate_empty_config(config)?;
    let seed = seed as u64;
    Ok(GeneratorOutput {
        dimension: &OVERWORLD,
        generator: ChunkGeneratorType::Overworld(VanillaGenerator::new(
            BiomeSourceKind::overworld(seed),
            seed,
        )),
        is_flat: false,
        sea_level: sea_level_for_dimension(&OVERWORLD),
    })
}

fn create_nether(config: &toml::Value, seed: i64) -> Result<GeneratorOutput, String> {
    validate_empty_config(config)?;
    let seed = seed as u64;
    Ok(GeneratorOutput {
        dimension: &THE_NETHER,
        generator: ChunkGeneratorType::Nether(VanillaGenerator::new(
            BiomeSourceKind::nether(seed),
            seed,
        )),
        is_flat: false,
        sea_level: sea_level_for_dimension(&THE_NETHER),
    })
}

fn create_end(config: &toml::Value, seed: i64) -> Result<GeneratorOutput, String> {
    validate_empty_config(config)?;
    let seed = seed as u64;
    Ok(GeneratorOutput {
        dimension: &THE_END,
        generator: ChunkGeneratorType::End(VanillaGenerator::new(BiomeSourceKind::end(seed), seed)),
        is_flat: false,
        sea_level: sea_level_for_dimension(&THE_END),
    })
}

fn create_flat(config: &toml::Value, _seed: i64) -> Result<GeneratorOutput, String> {
    let parsed = parse_flat_config(config)?;
    validate_flat_config(config)?;
    let dimension = dimension_by_key(&parsed.dimension)?;
    let mut layers = Vec::new();
    for layer in parsed.layers {
        let block = REGISTRY
            .blocks
            .by_key(&layer.block)
            .ok_or_else(|| format!("unknown block {} in minecraft:flat layer", layer.block))?;
        let state = REGISTRY.blocks.get_default_state_id(block);
        layers.extend(std::iter::repeat_n(state, layer.height));
    }

    Ok(GeneratorOutput {
        dimension,
        generator: ChunkGeneratorType::Flat(FlatChunkGenerator::new_layers(layers)),
        is_flat: true,
        sea_level: sea_level_for_dimension(dimension),
    })
}

fn create_empty(config: &toml::Value, _seed: i64) -> Result<GeneratorOutput, String> {
    let parsed: DimensionOnlyConfig = config
        .clone()
        .try_into()
        .map_err(|e| format!("invalid steel:empty config: {e}"))?;
    let dimension = dimension_by_key(&parsed.dimension)?;
    Ok(GeneratorOutput {
        dimension,
        generator: ChunkGeneratorType::Empty(EmptyChunkGenerator::new()),
        is_flat: false,
        sea_level: sea_level_for_dimension(dimension),
    })
}

fn dimension_by_key(key: &Identifier) -> Result<DimensionTypeRef, String> {
    REGISTRY
        .dimension_types
        .by_key(key)
        .ok_or_else(|| format!("unknown dimension type {key}"))
}

fn sea_level_for_dimension(dimension: DimensionTypeRef) -> i32 {
    if dimension == &THE_NETHER {
        32
    } else if dimension == &THE_END {
        0
    } else {
        63
    }
}
