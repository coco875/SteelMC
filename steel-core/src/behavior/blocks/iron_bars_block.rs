//! Iron bars block behavior implementation.
//!
//! Iron bars connect to adjacent iron bars, glass panes, walls, and solid blocks.
//! They use boolean properties for each horizontal direction.

use steel_registry::REGISTRY;
use steel_registry::blocks::BlockRef;
use steel_registry::blocks::block_state_ext::{BlockStateExt, is_exception_for_connection};
use steel_registry::blocks::properties::{BlockStateProperties, BoolProperty, Direction};
use steel_utils::{BlockPos, BlockStateId, Identifier};

use crate::behavior::block::BlockBehaviour;
use crate::behavior::context::BlockPlaceContext;
use crate::world::World;

/// Behavior for iron bars blocks.
///
/// Iron bars have 5 properties:
/// - `north`/`east`/`south`/`west` (bool): whether connected in that direction
/// - `waterlogged` (bool): whether filled with water
pub struct IronBarsBlock {
    block: BlockRef,
}

impl IronBarsBlock {
    /// North connection.
    pub const NORTH: BoolProperty = BlockStateProperties::NORTH;
    /// East connection.
    pub const EAST: BoolProperty = BlockStateProperties::EAST;
    /// South connection.
    pub const SOUTH: BoolProperty = BlockStateProperties::SOUTH;
    /// West connection.
    pub const WEST: BoolProperty = BlockStateProperties::WEST;
    /// Waterlogged property.
    pub const WATERLOGGED: BoolProperty = BlockStateProperties::WATERLOGGED;

    /// Creates a new iron bars block behavior.
    #[must_use]
    pub const fn new(block: BlockRef) -> Self {
        Self { block }
    }

    /// Checks if this block attaches to the given neighbor.
    /// Attaches to: other iron bars/glass panes, walls, and sturdy faces (except exceptions).
    fn attaches_to(neighbor_state: BlockStateId, face_sturdy: bool) -> bool {
        let neighbor_block = neighbor_state.get_block();

        // Connect to other iron bars / glass panes (CrossCollisionBlock in vanilla)
        // Check if in the "bars" tag or is an iron bars block
        let bars_tag = Identifier::vanilla_static("bars");
        if REGISTRY.blocks.is_in_tag(neighbor_block, &bars_tag) {
            return true;
        }

        // Connect to walls
        let walls_tag = Identifier::vanilla_static("walls");
        if REGISTRY.blocks.is_in_tag(neighbor_block, &walls_tag) {
            return true;
        }

        // Connect to sturdy faces (solid blocks) unless exception
        !is_exception_for_connection(neighbor_state) && face_sturdy
    }

    /// Gets the connection state for a position by checking all 4 horizontal neighbors.
    fn get_connection_state(&self, world: &World, pos: &BlockPos) -> BlockStateId {
        let mut state = self.block.default_state();

        let north_pos = Direction::North.relative(pos);
        let north_state = world.get_block_state(&north_pos);
        let north_sturdy = north_state.is_face_sturdy(Direction::South);
        state = state.set_value(&Self::NORTH, Self::attaches_to(north_state, north_sturdy));

        let east_pos = Direction::East.relative(pos);
        let east_state = world.get_block_state(&east_pos);
        let east_sturdy = east_state.is_face_sturdy(Direction::West);
        state = state.set_value(&Self::EAST, Self::attaches_to(east_state, east_sturdy));

        let south_pos = Direction::South.relative(pos);
        let south_state = world.get_block_state(&south_pos);
        let south_sturdy = south_state.is_face_sturdy(Direction::North);
        state = state.set_value(&Self::SOUTH, Self::attaches_to(south_state, south_sturdy));

        let west_pos = Direction::West.relative(pos);
        let west_state = world.get_block_state(&west_pos);
        let west_sturdy = west_state.is_face_sturdy(Direction::East);
        state = state.set_value(&Self::WEST, Self::attaches_to(west_state, west_sturdy));

        state
    }

    /// Updates a single side connection.
    fn update_side(
        state: BlockStateId,
        neighbor_state: BlockStateId,
        direction: Direction,
    ) -> BlockStateId {
        let opposite = direction.opposite();
        let face_sturdy = neighbor_state.is_face_sturdy(opposite);
        let connects = Self::attaches_to(neighbor_state, face_sturdy);

        match direction {
            Direction::North => state.set_value(&Self::NORTH, connects),
            Direction::East => state.set_value(&Self::EAST, connects),
            Direction::South => state.set_value(&Self::SOUTH, connects),
            Direction::West => state.set_value(&Self::WEST, connects),
            _ => state,
        }
    }
}

impl BlockBehaviour for IronBarsBlock {
    fn get_state_for_placement(&self, context: &BlockPlaceContext<'_>) -> Option<BlockStateId> {
        Some(self.get_connection_state(context.world, &context.relative_pos))
    }

    fn update_shape(
        &self,
        state: BlockStateId,
        _world: &World,
        _pos: BlockPos,
        direction: Direction,
        _neighbor_pos: BlockPos,
        neighbor_state: BlockStateId,
    ) -> BlockStateId {
        // Only update for horizontal directions
        match direction {
            Direction::North | Direction::East | Direction::South | Direction::West => {
                Self::update_side(state, neighbor_state, direction)
            }
            Direction::Up | Direction::Down => state,
        }
    }
}
