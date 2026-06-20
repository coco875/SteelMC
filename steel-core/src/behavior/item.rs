//! Item behavior trait and registry.

use steel_registry::items::ItemRef;
use steel_registry::{REGISTRY, RegistryEntry, RegistryExt};

use crate::behavior::items::DefaultItemBehavior;
use crate::behavior::{InteractionResult, UseItemContext, UseOnContext};
use crate::entity::Entity;
use steel_plugin_api::hook::{PluginItemBehavior, PluginWorld};

/// Trait defining the behavior of an item.
///
/// This trait handles dynamic/functional aspects of items:
/// - Use on blocks (placing, interacting)
/// - Use in air
/// - etc.
pub trait ItemBehavior: Send + Sync {
    /// Returns the Rust type name of the concrete behavior implementation.
    #[cfg(feature = "flint")]
    #[must_use]
    #[expect(clippy::absolute_paths, reason = "easier for features")]
    fn type_name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }

    /// Called when this item is used on a block.
    fn use_on(&self, _context: &mut UseOnContext) -> InteractionResult {
        InteractionResult::Pass
    }

    /// Called when this item is used (e.g. right click in air).
    fn use_item(&self, _context: &mut UseItemContext) -> InteractionResult {
        InteractionResult::Pass
    }
}

/// Registry for item behaviors.
///
/// Created after the main registry is frozen. Block items get `BlockItemBehavior`,
/// other items get `DefaultItemBehavior`. Custom behaviors can be registered.
pub struct ItemBehaviorRegistry {
    behaviors: Vec<Box<dyn ItemBehavior>>,
}

impl ItemBehaviorRegistry {
    /// Creates a new behavior registry with default behaviors for all items.
    ///
    /// Call `register_item_behaviors()` after this to set up proper behaviors.
    #[must_use]
    pub fn new() -> Self {
        let item_count = REGISTRY.items.len();
        let behaviors = (0..item_count)
            .map(|_| Box::new(DefaultItemBehavior) as Box<dyn ItemBehavior>)
            .collect();

        Self { behaviors }
    }

    /// Sets a custom behavior for an item.
    pub fn set_behavior(&mut self, item: ItemRef, behavior: Box<dyn ItemBehavior>) {
        let id = item.id();
        self.behaviors[id] = behavior;
    }

    /// Replaces the behavior for an item, passing the original behavior to a closure to construct the new one.
    pub fn replace_behavior_with<F>(&mut self, item: ItemRef, f: F)
    where
        F: FnOnce(Box<dyn ItemBehavior>) -> Box<dyn ItemBehavior>,
    {
        let id = item.id();
        let dummy = Box::new(DefaultItemBehavior);
        let original = std::mem::replace(&mut self.behaviors[id], dummy);
        let new_behavior = f(original);
        self.behaviors[id] = new_behavior;
    }

    /// Gets the behavior for an item.
    #[must_use]
    pub fn get_behavior(&self, item: ItemRef) -> &dyn ItemBehavior {
        let id = item.id();
        self.behaviors[id].as_ref()
    }

    /// Gets the behavior for an item by its ID.
    #[must_use]
    pub fn get_behavior_by_id(&self, id: usize) -> Option<&dyn ItemBehavior> {
        self.behaviors.get(id).map(AsRef::as_ref)
    }

    /// Get all behaviors.
    #[cfg(feature = "flint")]
    #[must_use]
    pub fn get_behaviors(&self) -> &[Box<dyn ItemBehavior>] {
        &self.behaviors
    }
}

impl Default for ItemBehaviorRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Host-side wrapper that sits in the `ItemBehaviorRegistry` for plugin-overridden items.
pub struct PluginItemBehaviorWrapper {
    pub(super) inner: steel_plugin_api::hook::PluginItemBehaviorRef,
    #[allow(dead_code)]
    pub(super) original: &'static dyn ItemBehavior,
}

impl ItemBehavior for PluginItemBehaviorWrapper {
    #[cfg(feature = "flint")]
    fn type_name(&self) -> &'static str {
        self.original.type_name()
    }

    fn use_on(&self, context: &mut UseOnContext) -> InteractionResult {
        let host_world = crate::world::HostWorld {
            world: context.world.clone(),
        };
        // SAFETY: HostWorld lives on the stack for the duration of this call;
        // the plugin ABI call is synchronous and does not retain the reference.
        let host_world_static: &'static crate::world::HostWorld =
            unsafe { std::mem::transmute(&host_world) };

        let face_u8 = context.hit_result.direction as u8;

        let res = self.inner.use_on(
            host_world_static.into(),
            context.player.id(),
            context.hand,
            context.hit_result.block_pos.x(),
            context.hit_result.block_pos.y(),
            context.hit_result.block_pos.z(),
            face_u8,
            context.hit_result.location.x,
            context.hit_result.location.y,
            context.hit_result.location.z,
        );

        match res {
            steel_plugin_api::hook::InteractionResult::Success => InteractionResult::Success,
            steel_plugin_api::hook::InteractionResult::Fail => InteractionResult::Fail,
            steel_plugin_api::hook::InteractionResult::Pass => InteractionResult::Pass,
            steel_plugin_api::hook::InteractionResult::TryEmptyHandInteraction => {
                InteractionResult::TryEmptyHandInteraction
            }
        }
    }

    fn use_item(&self, context: &mut UseItemContext) -> InteractionResult {
        let host_world = crate::world::HostWorld {
            world: context.world.clone(),
        };
        // SAFETY: HostWorld lives on the stack for the duration of this call;
        // the plugin ABI call is synchronous and does not retain the reference.
        let host_world_static: &'static crate::world::HostWorld =
            unsafe { std::mem::transmute(&host_world) };

        let res = self
            .inner
            .use_item(host_world_static.into(), context.player.id(), context.hand);

        match res {
            steel_plugin_api::hook::InteractionResult::Success => InteractionResult::Success,
            steel_plugin_api::hook::InteractionResult::Fail => InteractionResult::Fail,
            steel_plugin_api::hook::InteractionResult::Pass => InteractionResult::Pass,
            steel_plugin_api::hook::InteractionResult::TryEmptyHandInteraction => {
                InteractionResult::TryEmptyHandInteraction
            }
        }
    }
}

pub struct ItemBehaviorWrapper {
    pub(super) inner: &'static dyn ItemBehavior,
}

impl steel_plugin_api::hook::PluginItemBehavior for ItemBehaviorWrapper {
    extern "C" fn get_original(&self) -> steel_plugin_api::hook::PluginItemBehaviorRef {
        panic!("ItemBehaviorWrapper has no original behavior")
    }

    extern "C" fn use_on(
        &self,
        world: steel_plugin_api::hook::PluginWorldRef,
        player_id: i32,
        hand: steel_utils::types::InteractionHand,
        x: i32,
        y: i32,
        z: i32,
        face: u8,
        hit_x: f64,
        hit_y: f64,
        hit_z: f64,
    ) -> steel_plugin_api::hook::InteractionResult {
        let world_ptr = world.get_raw_world_ptr();
        if world_ptr.is_null() {
            return steel_plugin_api::hook::InteractionResult::Pass;
        }
        // SAFETY: The pointer returned by HostWorld is a valid *const Arc<World>.
        let world_arc = unsafe { &*world_ptr.cast::<std::sync::Arc<crate::world::World>>() };
        let Some(player) = world_arc.players.get_by_entity_id(player_id) else {
            return steel_plugin_api::hook::InteractionResult::Pass;
        };

        let pos = steel_utils::BlockPos::new(x, y, z);
        let direction = match face {
            0 => steel_utils::Direction::Down,
            1 => steel_utils::Direction::Up,
            2 => steel_utils::Direction::North,
            3 => steel_utils::Direction::South,
            4 => steel_utils::Direction::West,
            5 => steel_utils::Direction::East,
            _ => steel_utils::Direction::Up,
        };
        let hit_result = crate::behavior::BlockHitResult {
            block_pos: pos,
            direction,
            location: glam::DVec3::new(hit_x, hit_y, hit_z),
            inside: false,
            world_border_hit: false,
            miss: false,
        };
        let mut context = UseOnContext::new(
            &player,
            hand,
            hit_result,
            world_arc,
            player.inventory.clone(),
        );

        let res = self.inner.use_on(&mut context);
        match res {
            InteractionResult::Success => steel_plugin_api::hook::InteractionResult::Success,
            InteractionResult::Fail => steel_plugin_api::hook::InteractionResult::Fail,
            InteractionResult::Pass => steel_plugin_api::hook::InteractionResult::Pass,
            InteractionResult::TryEmptyHandInteraction => {
                steel_plugin_api::hook::InteractionResult::TryEmptyHandInteraction
            }
        }
    }

    extern "C" fn use_item(
        &self,
        world: steel_plugin_api::hook::PluginWorldRef,
        player_id: i32,
        hand: steel_utils::types::InteractionHand,
    ) -> steel_plugin_api::hook::InteractionResult {
        let world_ptr = world.get_raw_world_ptr();
        if world_ptr.is_null() {
            return steel_plugin_api::hook::InteractionResult::Pass;
        }
        // SAFETY: The pointer returned by HostWorld is a valid *const Arc<World>.
        let world_arc = unsafe { &*world_ptr.cast::<std::sync::Arc<crate::world::World>>() };
        let Some(player) = world_arc.players.get_by_entity_id(player_id) else {
            return steel_plugin_api::hook::InteractionResult::Pass;
        };

        let mut context = UseItemContext::new(&player, hand, world_arc, player.inventory.clone());

        let res = self.inner.use_item(&mut context);
        match res {
            InteractionResult::Success => steel_plugin_api::hook::InteractionResult::Success,
            InteractionResult::Fail => steel_plugin_api::hook::InteractionResult::Fail,
            InteractionResult::Pass => steel_plugin_api::hook::InteractionResult::Pass,
            InteractionResult::TryEmptyHandInteraction => {
                steel_plugin_api::hook::InteractionResult::TryEmptyHandInteraction
            }
        }
    }
}
