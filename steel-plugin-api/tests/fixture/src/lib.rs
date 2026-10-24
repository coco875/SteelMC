//! Independently compiled fixture covering complete behavior and entity delegation.
#![recursion_limit = "512"]
use std::sync::atomic::{AtomicI32, Ordering};
use steel_plugin_api::{
    ApiResult, BlockPos, BlockState, Hooks, Registry,
    block_behavior::{
        BlockBehavior, OriginalBlockBehavior,
        block_entity::BlockEntityBehavior,
        capabilities::{RailBehavior, RegisteredRailBehavior},
        types::{
            BlockEntity, BlockEntityCreation, BonemealableRef, BorrowedNbt, NbtMut, RailRef,
            WeakWorld,
        },
    },
    stabby::{self, boxed::Box, option::Option as StableOption, string::String as ApiString},
};

struct Inherit(OriginalBlockBehavior);
#[stabby::export(callback)]
impl BlockBehavior for Inherit {
    fn original(&self) -> OriginalBlockBehavior {
        self.0
    }
}
struct DisableBonemeal(OriginalBlockBehavior);
#[stabby::export(callback)]
impl BlockBehavior for DisableBonemeal {
    fn original(&self) -> OriginalBlockBehavior {
        self.0
    }
    fn as_bonemealable(&self) -> StableOption<BonemealableRef> {
        StableOption::None()
    }
}
struct StraightRail;
#[stabby::export(callback)]
impl RailBehavior for StraightRail {
    fn is_straight(&self) -> bool {
        true
    }
}
struct Rail {
    original: OriginalBlockBehavior,
    rail: RegisteredRailBehavior,
}
#[stabby::export(callback)]
impl BlockBehavior for Rail {
    fn original(&self) -> OriginalBlockBehavior {
        self.original
    }
    fn as_rail(&self) -> StableOption<RailRef> {
        Some(self.rail.borrow()).into()
    }
}
struct CounterBlock(OriginalBlockBehavior);
#[stabby::export(callback)]
impl BlockBehavior for CounterBlock {
    fn original(&self) -> OriginalBlockBehavior {
        self.0
    }
    fn new_block_entity(
        &self,
        level: WeakWorld,
        pos: BlockPos,
        state: BlockState,
    ) -> BlockEntityCreation {
        let original = self
            .0
            .new_block_entity(level, pos, state)
            .into_created()
            .match_owned(
                |entity| entity,
                || panic!("comparator fixture requires its original entity"),
            );
        let data = Box::new(Counter {
            original: original.share(),
            count: AtomicI32::new(0),
        })
        .into();
        BlockEntityCreation::created(original.with_behavior(data))
    }
}
struct Counter {
    original: BlockEntity,
    count: AtomicI32,
}
#[stabby::export(callback)]
impl BlockEntityBehavior for Counter {
    fn load_additional(&self, nbt: BorrowedNbt<'_>) {
        if let Some(count) = nbt.int("plugin_count".into()).as_ref() {
            self.count.store(*count, Ordering::Relaxed);
        }
        self.original.borrow().load_additional(nbt);
    }
    fn save_additional(&self, mut nbt: NbtMut<'_>) {
        nbt.put_int("plugin_count".into(), self.count.load(Ordering::Relaxed));
        self.original.borrow().save_additional(nbt);
    }
    fn trigger_event(&self, kind: i32, value: i32) -> bool {
        if kind == 1234 {
            self.count.store(value, Ordering::Relaxed);
            return true;
        }
        self.original.borrow().trigger_event(kind, value)
    }
}

#[stabby::export(callback)]
fn register_blocks(mut registry: Registry<'_>) -> ApiResult<()> {
    let result = (|| -> Result<(), ApiString> {
        let carrots = OriginalBlockBehavior::get(&registry, "minecraft:carrots".into())
            .match_owned(Ok, Err)?;
        if carrots.as_bonemealable().is_none() {
            return Err("original carrot capability is unavailable during registration".into());
        }
        registry
            .replace_block_behavior(
                "minecraft:carrots".into(),
                Box::new(Inherit(carrots)).into(),
            )
            .match_owned(Ok, Err)?;
        if registry.config().as_str() == "fail_after_replacement" {
            return Err("fixture failed after replacing carrots".into());
        }
        let potatoes = OriginalBlockBehavior::get(&registry, "minecraft:potatoes".into())
            .match_owned(Ok, Err)?;
        registry
            .replace_block_behavior(
                "minecraft:potatoes".into(),
                Box::new(DisableBonemeal(potatoes)).into(),
            )
            .match_owned(Ok, Err)?;
        let rail =
            OriginalBlockBehavior::get(&registry, "minecraft:rail".into()).match_owned(Ok, Err)?;
        let implementation = registry.create_rail(Box::new(StraightRail).into());
        registry
            .replace_block_behavior(
                "minecraft:rail".into(),
                Box::new(Rail {
                    original: rail,
                    rail: implementation,
                })
                .into(),
            )
            .match_owned(Ok, Err)?;
        let comparator = OriginalBlockBehavior::get(&registry, "minecraft:comparator".into())
            .match_owned(Ok, Err)?;
        registry
            .replace_block_behavior(
                "minecraft:comparator".into(),
                Box::new(CounterBlock(comparator)).into(),
            )
            .match_owned(Ok, Err)?;
        Ok(())
    })();
    result.into()
}
fn pre_init(mut hooks: Hooks<'_>) -> ApiResult<()> {
    hooks.register_block_registration(register_blocks)
}
fn init(config: stabby::str::Str<'_>) -> ApiResult<()> {
    if config.as_str() == "fail_after_replacement" {
        return Err(ApiString::from("init ran after failed registration")).into();
    }
    steel_plugin_api::default_block_state("minecraft:stone".into()).map(|_| ())
}
steel_plugin_api::export_plugin!("behavior-boundary-fixture", pre_init, init);
