//! Fence gate block behavior implementation.
//!
//! Fence gates can be opened/closed by right-clicking or redstone signals.
//! They have a `IN_WALL` property that changes their appearance when between walls.

use steel_registry::REGISTRY;
use steel_registry::blocks::BlockRef;
use steel_registry::blocks::block_state_ext::BlockStateExt;
use steel_registry::blocks::properties::{
    Axis, BlockStateProperties, BoolProperty, Direction, EnumProperty,
};
use steel_utils::{BlockPos, BlockStateId, Identifier, types::UpdateFlags};

use crate::behavior::block::BlockBehaviour;
use crate::behavior::context::{BlockHitResult, BlockPlaceContext, InteractionResult};
use crate::player::Player;
use crate::world::World;

/// Behavior for fence gate blocks.
///
/// Fence gates have 4 properties:
/// - `facing` (Direction): horizontal direction the gate faces
/// - `open` (bool): whether the gate is open
/// - `powered` (bool): whether receiving redstone signal
/// - `in_wall` (bool): whether between walls (changes height)
pub struct FenceGateBlock {
    block: BlockRef,
}

impl FenceGateBlock {
    /// Facing direction property.
    pub const FACING: EnumProperty<Direction> = BlockStateProperties::HORIZONTAL_FACING;
    /// Open property.
    pub const OPEN: BoolProperty = BlockStateProperties::OPEN;
    /// Powered property.
    pub const POWERED: BoolProperty = BlockStateProperties::POWERED;
    /// In wall property.
    pub const IN_WALL: BoolProperty = BlockStateProperties::IN_WALL;

    /// Creates a new fence gate block behavior.
    #[must_use]
    pub const fn new(block: BlockRef) -> Self {
        Self { block }
    }

    /// Checks if a block state is a wall.
    fn is_wall(state: BlockStateId) -> bool {
        let block = state.get_block();
        let walls_tag = Identifier::vanilla_static("walls");
        REGISTRY.blocks.is_in_tag(block, &walls_tag)
    }

    /// Returns the perpendicular axis for a horizontal direction.
    #[allow(clippy::match_same_arms)]
    fn perpendicular_axis(facing: Direction) -> Axis {
        match facing.get_axis() {
            Axis::X => Axis::Z,
            Axis::Z => Axis::X,
            Axis::Y => Axis::X, // shouldn't happen for horizontal
        }
    }

    /// Checks if fence gate should be in wall mode based on neighbors.
    fn should_be_in_wall(world: &World, pos: &BlockPos, facing: Direction) -> bool {
        let perp_axis = Self::perpendicular_axis(facing);
        match perp_axis {
            // Gate facing E/W (X axis), check N/S neighbors for walls
            Axis::Z => {
                let north = world.get_block_state(&Direction::North.relative(pos));
                let south = world.get_block_state(&Direction::South.relative(pos));
                Self::is_wall(north) || Self::is_wall(south)
            }
            // Gate facing N/S (Z axis), check E/W neighbors for walls
            Axis::X => {
                let west = world.get_block_state(&Direction::West.relative(pos));
                let east = world.get_block_state(&Direction::East.relative(pos));
                Self::is_wall(west) || Self::is_wall(east)
            }
            Axis::Y => false,
        }
    }
}

impl BlockBehaviour for FenceGateBlock {
    fn get_state_for_placement(&self, context: &BlockPlaceContext<'_>) -> Option<BlockStateId> {
        let facing = context.horizontal_direction;
        let in_wall = Self::should_be_in_wall(context.world, &context.relative_pos, facing);

        // TODO: Check for redstone signal to set OPEN/POWERED
        let state = self
            .block
            .default_state()
            .set_value(&Self::FACING, facing)
            .set_value(&Self::OPEN, false)
            .set_value(&Self::POWERED, false)
            .set_value(&Self::IN_WALL, in_wall);

        Some(state)
    }

    fn update_shape(
        &self,
        state: BlockStateId,
        world: &World,
        pos: BlockPos,
        direction: Direction,
        _neighbor_pos: BlockPos,
        neighbor_state: BlockStateId,
    ) -> BlockStateId {
        let facing: Direction = state.get_value(&Self::FACING);
        let perp_axis = Self::perpendicular_axis(facing);

        // Only update IN_WALL for perpendicular neighbors
        if direction.get_axis() != perp_axis {
            return state;
        }

        // Check if either perpendicular neighbor is a wall
        let opposite_pos = direction.opposite().relative(&pos);
        let opposite_state = world.get_block_state(&opposite_pos);
        let in_wall = Self::is_wall(neighbor_state) || Self::is_wall(opposite_state);

        state.set_value(&Self::IN_WALL, in_wall)
    }

    fn use_without_item(
        &self,
        state: BlockStateId,
        world: &World,
        pos: BlockPos,
        _player: &Player,
        _hit_result: &BlockHitResult,
    ) -> InteractionResult {
        let is_open: bool = state.get_value(&Self::OPEN);

        // Toggle the gate open/closed
        let new_state = state.set_value(&Self::OPEN, !is_open);

        world.set_block(pos, new_state, UpdateFlags::UPDATE_ALL);

        // TODO: Play sound
        log::debug!(
            "Fence gate at {:?} {}",
            pos,
            if is_open { "closed" } else { "opened" }
        );

        InteractionResult::Success
    }
}
