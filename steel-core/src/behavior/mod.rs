//! Block and item behavior system.
//!
//! This module contains the behavior traits and registries that define how
//! blocks and items behave dynamically. This is separate from the static data
//! in steel-registry to maintain a clean separation between constant data and
//! functional/dynamic behavior.
//!
//! # Architecture
//!
//! After the main registry (`steel-registry`) is frozen, behavior registries
//! are created:
//! - `BlockBehaviorRegistry` - assigns default or custom behaviors to each block
//! - `ItemBehaviorRegistry` - assigns default or custom behaviors to each item
//!
//! # Usage
//!
//! ```ignore
//! use steel_core::behavior::{init_behaviors, BLOCK_BEHAVIORS, ITEM_BEHAVIORS};
//!
//! // After registry is frozen, call once at startup:
//! init_behaviors();
//!
//! // Then access behaviors via the global registries:
//! let behavior = BLOCK_BEHAVIORS.get_behavior(block);
//! ```

mod block;
pub mod blocks;
mod context;
pub mod fluid;
mod item;
pub mod items;

#[expect(warnings)]
#[rustfmt::skip]
#[path = "generated/blocks.rs"]
pub mod block_behaviors;

#[expect(warnings)]
#[rustfmt::skip]
#[path = "generated/candle_cakes.rs"]
pub mod candle_cakes;

#[allow(warnings)]
#[rustfmt::skip]
#[path = "generated/items.rs"]
pub mod item_behaviors;

#[expect(warnings)]
#[rustfmt::skip]
#[path = "generated/strippables.rs"]
pub mod strippables;

#[expect(warnings)]
#[rustfmt::skip]
#[path = "generated/waxables.rs"]
pub mod waxables;

#[expect(warnings)]
#[rustfmt::skip]
#[path = "generated/weathering.rs"]
pub mod weathering;

pub use block::{
    BlockBehavior, BlockBehaviorRegistry, BlockCollisionContext, DefaultBlockBehavior,
    EntityFallDamage, EntityFallOnContext, EntityFallOnFacts, EntityLandingContext,
    PluginBlockBehaviorWrapper,
};
use block_behaviors::register_block_behaviors;
pub use context::{
    BlockHitResult, BlockPlaceContext, InteractionResult, InventoryAccess, UseItemContext,
    UseOnContext,
};
pub use fluid::{FLUID_BEHAVIORS, FluidBehaviorRegistry};
pub use item::{ItemBehavior, ItemBehaviorRegistry};
use item_behaviors::register_item_behaviors;
pub use items::{
    BlockItem, BucketItem, DefaultItemBehavior, DoubleHighBlockItem, EnderEyeItem, HangingSignItem,
    ShovelItem, SignItem, StandingAndWallBlockItem,
};
use std::ops::Deref;
use std::sync::OnceLock;
use steel_registry::blocks::block_state_ext::BlockStateExt;
use steel_registry::RegistryExt;
use steel_registry::fluid::FluidState;
use steel_registry::vanilla_fluids;
use steel_utils::BlockStateId;

use crate::fluid::{FluidBehavior, LavaFluid, WaterFluid};

/// Wrapper for the global block behavior registry that implements `Deref`.
pub struct BlockBehaviorLock(OnceLock<BlockBehaviorRegistry>);

impl Deref for BlockBehaviorLock {
    type Target = BlockBehaviorRegistry;

    fn deref(&self) -> &Self::Target {
        self.0.get().expect("Block behaviors not initialized")
    }
}

/// Wrapper for the global item behavior registry that implements `Deref`.
pub struct ItemBehaviorLock(OnceLock<ItemBehaviorRegistry>);

impl Deref for ItemBehaviorLock {
    type Target = ItemBehaviorRegistry;

    fn deref(&self) -> &Self::Target {
        self.0.get().expect("Item behaviors not initialized")
    }
}

/// Extension trait for `BlockStateId` that provides access to behavior-dependent methods.
///
/// This is separate from `BlockStateExt` (in steel-registry) because these methods
/// require access to the behavior registry which lives in steel-core.
pub trait BlockStateBehaviorExt {
    /// Returns the fluid state for this block state.
    ///
    /// Delegates to the block's `BlockBehavior::get_fluid_state` implementation.
    fn get_fluid_state(&self) -> FluidState;
}

impl BlockStateBehaviorExt for BlockStateId {
    fn get_fluid_state(&self) -> FluidState {
        let block = self.get_block();
        let behavior = BLOCK_BEHAVIORS.get_behavior(block);
        behavior.get_fluid_state(*self)
    }
}

/// Global block behavior registry.
///
/// Access behaviors directly via deref: `BLOCK_BEHAVIORS.get_behavior(block)`
pub static BLOCK_BEHAVIORS: BlockBehaviorLock = BlockBehaviorLock(OnceLock::new());

/// Global item behavior registry.
///
/// Access behaviors directly via deref: `ITEM_BEHAVIORS.get_behavior(item)`
pub static ITEM_BEHAVIORS: ItemBehaviorLock = ItemBehaviorLock(OnceLock::new());
/// Host-side wrapper that implements the plugin API's `PluginBehaviorRegistry` trait.
pub struct HostBehaviorRegistry {
    /// Opaque pointer to the underlying block behavior registry.
    pub block_registry: *mut BlockBehaviorRegistry,
    /// Opaque pointer to the underlying item behavior registry.
    pub item_registry: *mut ItemBehaviorRegistry,
}

// SAFETY: HostBehaviorRegistry is only used synchronously during initialization on a single thread.
unsafe impl Send for HostBehaviorRegistry {}
// SAFETY: HostBehaviorRegistry is only used synchronously during initialization on a single thread.
unsafe impl Sync for HostBehaviorRegistry {}

struct DummyPluginBlockBehavior;
impl steel_plugin_api::hook::PluginBlockBehavior for DummyPluginBlockBehavior {
    extern "C" fn get_original(&self) -> steel_plugin_api::hook::PluginBlockBehaviorRef {
        panic!("dummy behavior has no original behavior")
    }
}

struct DummyPluginItemBehavior;
impl steel_plugin_api::hook::PluginItemBehavior for DummyPluginItemBehavior {
    extern "C" fn get_original(&self) -> steel_plugin_api::hook::PluginItemBehaviorRef {
        panic!("dummy behavior has no original behavior")
    }
}

impl steel_plugin_api::hook::PluginBehaviorRegistry for HostBehaviorRegistry {
    extern "C" fn register_block_behavior(
        &self,
        namespace: steel_plugin_api::AbiStr<'_>,
        path: steel_plugin_api::AbiStr<'_>,
        behavior: extern "C" fn(steel_plugin_api::hook::PluginBlockBehaviorRef) -> steel_plugin_api::hook::PluginBlockBehaviorRef,
    ) -> steel_plugin_api::hook::PluginBlockBehaviorRef {
        // SAFETY: The host guarantees that `block_registry` points to a valid `BlockBehaviorRegistry` during the hook execution.
        let registry = unsafe { &mut *self.block_registry };
        let identifier = steel_utils::Identifier::new(namespace.as_str().to_string(), path.as_str().to_string());
        if let Some(block) = steel_registry::REGISTRY.blocks.by_key(&identifier) {
            let mut prev_ref_opt = None;
            registry.replace_behavior_with(block, |original| {
                let original_ref: &'static dyn BlockBehavior = Box::leak(original);
                let wrapper = block::BlockBehaviorWrapper { inner: original_ref };
                let leaked_prev = stabby::sync::Arc::new(wrapper);
                let prev_ref = steel_plugin_api::hook::PluginBlockBehaviorRef(leaked_prev.into());
                prev_ref_opt = Some(prev_ref.clone());

                let new_behavior_ref = behavior(prev_ref);

                Box::new(PluginBlockBehaviorWrapper {
                    inner: new_behavior_ref,
                    original: original_ref,
                })
            });
            prev_ref_opt.expect("block exists, behavior closure must be executed")
        } else {
            let dummy_arc = stabby::sync::Arc::new(DummyPluginBlockBehavior);
            steel_plugin_api::hook::PluginBlockBehaviorRef(dummy_arc.into())
        }
    }

    extern "C" fn register_item_behavior(
        &self,
        namespace: steel_plugin_api::AbiStr<'_>,
        path: steel_plugin_api::AbiStr<'_>,
        behavior: extern "C" fn(steel_plugin_api::hook::PluginItemBehaviorRef) -> steel_plugin_api::hook::PluginItemBehaviorRef,
    ) -> steel_plugin_api::hook::PluginItemBehaviorRef {
        // SAFETY: The host guarantees that `item_registry` points to a valid `ItemBehaviorRegistry` during the hook execution.
        let registry = unsafe { &mut *self.item_registry };
        let identifier = steel_utils::Identifier::new(namespace.as_str().to_string(), path.as_str().to_string());
        if let Some(item) = steel_registry::REGISTRY.items.by_key(&identifier) {
            let mut prev_ref_opt = None;
            registry.replace_behavior_with(item, |original| {
                let original_ref: &'static dyn ItemBehavior = Box::leak(original);
                let wrapper = item::ItemBehaviorWrapper { inner: original_ref };
                let leaked_prev = stabby::sync::Arc::new(wrapper);
                let prev_ref = steel_plugin_api::hook::PluginItemBehaviorRef(leaked_prev.into());
                prev_ref_opt = Some(prev_ref.clone());

                let new_behavior_ref = behavior(prev_ref);

                Box::new(item::PluginItemBehaviorWrapper {
                    inner: new_behavior_ref,
                    original: original_ref,
                })
            });
            prev_ref_opt.expect("item exists, behavior closure must be executed")
        } else {
            let dummy_arc = stabby::sync::Arc::new(DummyPluginItemBehavior);
            steel_plugin_api::hook::PluginItemBehaviorRef(dummy_arc.into())
        }
    }
}

/// Initializes the global behavior registries.
///
/// This should be called after the main registry is frozen. Repeated calls are a no-op.
pub fn init_behaviors() {
    let mut block_behaviors = BlockBehaviorRegistry::new();
    register_block_behaviors(&mut block_behaviors);

    let mut item_behaviors = ItemBehaviorRegistry::new();
    register_item_behaviors(&mut item_behaviors);

    let host_behavior_registry: &'static HostBehaviorRegistry = Box::leak(Box::new(HostBehaviorRegistry {
        block_registry: &raw mut block_behaviors,
        item_registry: &raw mut item_behaviors,
    }));
    let registry_dynptr = host_behavior_registry.into();

    // Fire behavior initialization hook
    let hook_registry = steel_plugin_loader::hook::get_host_registry();
    let action_args = steel_plugin_api::hook::BehaviorInitAction {
        registry: registry_dynptr,
    };
    hook_registry.do_action_typed(&action_args);

    BLOCK_BEHAVIORS.0.get_or_init(|| block_behaviors);

    FLUID_BEHAVIORS.0.get_or_init(|| {
        let mut fluid_behaviors = FluidBehaviorRegistry::new();

        // Water: WaterFluid implements FluidBehavior directly
        let water_behavior: Box<dyn FluidBehavior> = Box::new(WaterFluid);
        // Both WATER and FLOWING_WATER share the same behavior
        fluid_behaviors.set_behavior(&vanilla_fluids::WATER, water_behavior);
        fluid_behaviors.set_behavior(&vanilla_fluids::FLOWING_WATER, Box::new(WaterFluid));

        // Lava: LavaFluid implements FluidBehavior directly
        let lava_behavior: Box<dyn FluidBehavior> = Box::new(LavaFluid);
        fluid_behaviors.set_behavior(&vanilla_fluids::LAVA, lava_behavior);
        fluid_behaviors.set_behavior(&vanilla_fluids::FLOWING_LAVA, Box::new(LavaFluid));

        fluid_behaviors
    });

    ITEM_BEHAVIORS.0.get_or_init(|| item_behaviors);
}
