use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::{
    behavior::{BlockBehavior, BlockBehaviorRegistry, BlockPlaceContext, DefaultBlockBehavior},
    bootstrap::init_globals,
    test_support::fresh_test_world,
};
use steel_registry::vanilla_blocks;
use steel_utils::BlockStateId;

struct ReplacedBehavior {
    original: Arc<dyn BlockBehavior>,
}

impl BlockBehavior for ReplacedBehavior {
    fn get_state_for_placement(&self, context: &BlockPlaceContext<'_>) -> Option<BlockStateId> {
        self.original.get_state_for_placement(context)
    }

    fn is_bed(&self) -> bool {
        true
    }
}

#[test]
fn complete_replacement_updates_the_vector_and_keeps_the_original_available() {
    init_globals();
    let mut registry = BlockBehaviorRegistry::new();
    let block = &vanilla_blocks::WHITE_WOOL;
    registry.set_behavior(block, Box::new(DefaultBlockBehavior::new(block)));
    let original = registry.get_behavior_shared(block);
    registry.set_behavior(
        block,
        Box::new(ReplacedBehavior {
            original: Arc::clone(&original),
        }),
    );
    assert!(registry.get_behavior(block).is_bed());
    assert!(!original.is_bed());
    assert_eq!(
        registry
            .get_behavior_for_state(block.default_state())
            .map(BlockBehavior::is_bed),
        Some(true),
    );
    registry.set_behavior(block, Box::new(DefaultBlockBehavior::new(block)));
    assert!(!registry.get_behavior(block).is_bed());
    assert!(!original.is_bed());
}

struct TickExtension(Arc<AtomicUsize>);
impl WorldTickHandler for TickExtension {
    fn tick(&self, _world: &Arc<World>, _tick_count: u64) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }
}

#[test]
fn frozen_world_ticks_do_not_invoke_gameplay_extensions() {
    init_globals();
    let fixture = fresh_test_world("extension_frozen_tick");
    let calls = Arc::new(AtomicUsize::new(0));
    let mut extensions = Extensions::default();
    extensions
        .hooks
        .register_world_tick(Box::new(TickExtension(Arc::clone(&calls))));
    assert!(fixture.world.extensions.set(Arc::new(extensions)).is_ok());
    fixture.world.tick_game(1, false);
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    fixture.world.tick_game(2, true);
    assert_eq!(calls.load(Ordering::Relaxed), 1);
}
