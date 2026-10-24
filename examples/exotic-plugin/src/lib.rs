//! Wool shearing, Conway's Game of Life, and a deterministic strange-world generator.
#![recursion_limit = "512"]

mod life;
mod terrain;

use std::sync::OnceLock;

use serde::Deserialize;
use steel_plugin_api::{
    ApiResult, BlockPos, BlockState, GeneratorRegistry, Hooks, Registry, World,
    block_behavior::{
        BlockBehavior, InteractionResult, OriginalBlockBehavior,
        types::{BlockHit, InteractionHand, Inventory, ItemStack, Player},
    },
    blocks_in_tag, default_block_state,
    stabby::{self, str::Str, string::String as ApiString},
};

struct Materials {
    alive: BlockState,
    dead: BlockState,
    stone: BlockState,
    shell: BlockState,
    pillar: BlockState,
    light: BlockState,
}

struct Settings {
    materials: Materials,
    life: life::LifeConfig,
}

static SETTINGS: OnceLock<Settings> = OnceLock::new();

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    life: life::LifeConfig,
}

fn pre_init(mut hooks: Hooks<'_>) -> ApiResult<()> {
    let result = (|| -> Result<(), ApiString> {
        hooks
            .register_block_registration(register_blocks)
            .match_owned(Ok, Err)?;
        hooks
            .register_generator_registration(register_generators)
            .match_owned(Ok, Err)?;
        hooks.register_world_tick(life::tick);
        Ok(())
    })();
    result.into()
}

#[stabby::export(callback)]
fn register_blocks(mut registry: Registry<'_>) -> ApiResult<()> {
    let result = (|| -> Result<(), ApiString> {
        let wood = default_block_state("minecraft:oak_wood".into()).match_owned(Ok, Err)?;
        for block in blocks_in_tag("minecraft:wool".into())
            .match_owned(Ok, Err)?
            .iter()
        {
            let original = OriginalBlockBehavior::get(&registry, block.as_str().into())
                .match_owned(Ok, Err)?;
            registry
                .replace_block_behavior(
                    block.as_str().into(),
                    stabby::boxed::Box::new(ShearedWool { original, wood }).into(),
                )
                .match_owned(Ok, Err)?;
        }
        Ok(())
    })();
    result.into()
}

#[stabby::export(callback)]
fn register_generators(mut registry: GeneratorRegistry<'_>) -> ApiResult<()> {
    registry.register_terrain_generator(
        "example:strange".into(),
        "minecraft:overworld".into(),
        "minecraft:the_end".into(),
        0,
        terrain::generate,
    )
}

fn init(config: Str<'_>) -> ApiResult<()> {
    let result = (|| -> Result<(), ApiString> {
        let config: Config =
            toml::from_str(config.as_str()).map_err(|error| ApiString::from(error.to_string()))?;
        config.life.validate().map_err(ApiString::from)?;
        let materials = Materials {
            alive: default_block_state("minecraft:black_concrete".into()).match_owned(Ok, Err)?,
            dead: default_block_state("minecraft:white_concrete".into()).match_owned(Ok, Err)?,
            stone: default_block_state("minecraft:stone".into()).match_owned(Ok, Err)?,
            shell: default_block_state("minecraft:end_stone".into()).match_owned(Ok, Err)?,
            pillar: default_block_state("minecraft:purpur_block".into()).match_owned(Ok, Err)?,
            light: default_block_state("minecraft:glowstone".into()).match_owned(Ok, Err)?,
        };
        SETTINGS
            .set(Settings {
                materials,
                life: config.life,
            })
            .map_err(|_| ApiString::from("example plugin was initialized twice"))?;
        Ok(())
    })();
    result.into()
}

fn settings() -> &'static Settings {
    match SETTINGS.get() {
        Some(settings) => settings,
        None => panic!("example plugin callback invoked before successful initialization"),
    }
}

struct ShearedWool {
    original: OriginalBlockBehavior,
    wood: BlockState,
}

#[stabby::export(callback)]
impl BlockBehavior for ShearedWool {
    fn original(&self) -> OriginalBlockBehavior {
        self.original
    }

    fn use_item_on(
        &self,
        state: BlockState,
        world: World<'_>,
        pos: BlockPos,
        player: Player<'_>,
        hand: InteractionHand,
        hit_result: BlockHit<'_>,
        mut inv: Inventory<'_>,
    ) -> InteractionResult {
        if !inv.held_item_is("minecraft:shears".into()) {
            return self
                .original
                .use_item_on(state, world, pos, player, hand, hit_result, inv);
        }
        let wool = match ItemStack::from_block(state).match_owned(Ok, Err) {
            Ok(wool) => wool,
            Err(error) => {
                eprintln!("Wool interaction could not create its drop: {error}");
                return self
                    .original
                    .use_item_on(state, world, pos, player, hand, hit_result, inv);
            }
        };
        if let Err(error) = world.set_block(pos, self.wood).match_owned(Ok, Err) {
            eprintln!("Wool interaction could not replace the block: {error}");
            return self
                .original
                .use_item_on(state, world, pos, player, hand, hit_result, inv);
        }
        if let Err(error) = inv
            .damage_item(1, player.has_infinite_materials())
            .match_owned(Ok, Err)
        {
            panic!("valid shears damage failed after replacing wool: {error}");
        }
        if let Err(error) = world.pop_resource(pos, wool).match_owned(Ok, Err) {
            eprintln!("Wool interaction could not drop its item: {error}");
        }
        InteractionResult::Success
    }
}

steel_plugin_api::export_plugin!("exotic-example", pre_init, init);
