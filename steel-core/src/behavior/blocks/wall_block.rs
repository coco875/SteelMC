//! Wall block behavior implementation.
//!
//! Walls connect to adjacent walls, iron bars, and solid blocks.
//! Unlike fences, walls have three connection states: None, Low, and Tall.

use steel_registry::REGISTRY;
use steel_registry::blocks::BlockRef;
use steel_registry::blocks::block_state_ext::{BlockStateExt, is_exception_for_connection};
use steel_registry::blocks::properties::{
    BlockStateProperties, BoolProperty, Direction, EnumProperty, WallSide,
};
use steel_utils::{BlockPos, BlockStateId, Identifier};

use crate::behavior::block::BlockBehaviour;
use crate::behavior::context::BlockPlaceContext;
use crate::world::World;

/// Behavior for wall blocks.
///
/// Walls have 5 main properties:
/// - `up` (bool): whether to render the center post
/// - `north`/`east`/`south`/`west` (WallSide): None, Low, or Tall for each direction
pub struct WallBlock {
    block: BlockRef,
}

impl WallBlock {
    /// Up property (center post).
    pub const UP: BoolProperty = BlockStateProperties::UP;
    /// North wall side.
    pub const NORTH: EnumProperty<WallSide> = BlockStateProperties::NORTH_WALL;
    /// East wall side.
    pub const EAST: EnumProperty<WallSide> = BlockStateProperties::EAST_WALL;
    /// South wall side.
    pub const SOUTH: EnumProperty<WallSide> = BlockStateProperties::SOUTH_WALL;
    /// West wall side.
    pub const WEST: EnumProperty<WallSide> = BlockStateProperties::WEST_WALL;
    /// Waterlogged property.
    pub const WATERLOGGED: BoolProperty = BlockStateProperties::WATERLOGGED;

    /// Creates a new wall block behavior.
    #[must_use]
    pub const fn new(block: BlockRef) -> Self {
        Self { block }
    }

    /// Checks if this wall should connect to the given neighbor state.
    fn connects_to(neighbor_state: BlockStateId, face_sturdy: bool, direction: Direction) -> bool {
        let neighbor_block = neighbor_state.get_block();

        // Connect to other walls
        let walls_tag = Identifier::vanilla_static("walls");
        if REGISTRY.blocks.is_in_tag(neighbor_block, &walls_tag) {
            return true;
        }

        // Connect to iron bars / glass panes
        // In vanilla: block instanceof IronBarsBlock
        // We check for the tag instead
        // TODO: Add proper iron_bars check when tag is available

        // Connect to fence gates (perpendicular to their facing)
        let fence_gates_tag = Identifier::vanilla_static("fence_gates");
        if REGISTRY.blocks.is_in_tag(neighbor_block, &fence_gates_tag) {
            // Fence gates connect perpendicular to their facing direction
            // A gate facing north/south connects to fences to its east/west
            // A gate facing east/west connects to fences to its north/south
            if let Some(facing_str) = neighbor_state.get_property_str("facing") {
                let gate_facing = match facing_str.as_str() {
                    "north" => Some(Direction::North),
                    "south" => Some(Direction::South),
                    "east" => Some(Direction::East),
                    "west" => Some(Direction::West),
                    _ => None,
                };

                if let Some(gate_facing) = gate_facing {
                    // Gate connects perpendicular to its facing
                    let connects = match (gate_facing, direction) {
                        // Gate facing N/S connects to blocks on E/W sides,
                        // Gate facing E/W connects to blocks on N/S sides
                        (
                            Direction::North | Direction::South,
                            Direction::East | Direction::West,
                        )
                        | (
                            Direction::East | Direction::West,
                            Direction::North | Direction::South,
                        ) => true,
                        _ => false,
                    };
                    if connects {
                        return true;
                    }
                }
            }
        }

        // Connect to sturdy faces (solid blocks) unless exception
        !is_exception_for_connection(neighbor_state) && face_sturdy
    }

    /// Gets the connection state for each direction.
    fn get_connection_state(&self, world: &World, pos: &BlockPos) -> BlockStateId {
        let mut state = self.block.default_state();

        // Check each horizontal direction
        let north_pos = Direction::North.relative(pos);
        let north_state = world.get_block_state(&north_pos);
        let north_sturdy = north_state.is_face_sturdy(Direction::South);
        let north_connects = Self::connects_to(north_state, north_sturdy, Direction::South);

        let east_pos = Direction::East.relative(pos);
        let east_state = world.get_block_state(&east_pos);
        let east_sturdy = east_state.is_face_sturdy(Direction::West);
        let east_connects = Self::connects_to(east_state, east_sturdy, Direction::West);

        let south_pos = Direction::South.relative(pos);
        let south_state = world.get_block_state(&south_pos);
        let south_sturdy = south_state.is_face_sturdy(Direction::North);
        let south_connects = Self::connects_to(south_state, south_sturdy, Direction::North);

        let west_pos = Direction::West.relative(pos);
        let west_state = world.get_block_state(&west_pos);
        let west_sturdy = west_state.is_face_sturdy(Direction::East);
        let west_connects = Self::connects_to(west_state, west_sturdy, Direction::East);

        // Set wall sides (simplified: use Low when connected, None otherwise)
        // TODO: Implement Tall logic based on block above
        state = state.set_value(
            &Self::NORTH,
            if north_connects {
                WallSide::Low
            } else {
                WallSide::None
            },
        );
        state = state.set_value(
            &Self::EAST,
            if east_connects {
                WallSide::Low
            } else {
                WallSide::None
            },
        );
        state = state.set_value(
            &Self::SOUTH,
            if south_connects {
                WallSide::Low
            } else {
                WallSide::None
            },
        );
        state = state.set_value(
            &Self::WEST,
            if west_connects {
                WallSide::Low
            } else {
                WallSide::None
            },
        );

        // Determine if we need the center post
        let up =
            Self::should_raise_post(north_connects, east_connects, south_connects, west_connects);
        state = state.set_value(&Self::UP, up);

        state
    }

    /// Determines if the center post should be raised.
    /// Post appears when: all sides disconnected, or asymmetric connections, or corners.
    #[allow(clippy::fn_params_excessive_bools)]
    fn should_raise_post(north: bool, east: bool, south: bool, west: bool) -> bool {
        // No connections at all -> post
        if !north && !east && !south && !west {
            return true;
        }

        // Asymmetric N/S or E/W -> post (corner or T-junction)
        if north != south || east != west {
            return true;
        }

        // Straight line (N-S or E-W) -> no post
        false
    }

    /// Updates a single side connection.
    fn update_side(
        state: BlockStateId,
        neighbor_state: BlockStateId,
        direction: Direction,
    ) -> BlockStateId {
        let opposite = direction.opposite();
        let face_sturdy = neighbor_state.is_face_sturdy(opposite);
        let connects = Self::connects_to(neighbor_state, face_sturdy, opposite);
        let wall_side = if connects {
            WallSide::Low
        } else {
            WallSide::None
        };

        match direction {
            Direction::North => state.set_value(&Self::NORTH, wall_side),
            Direction::East => state.set_value(&Self::EAST, wall_side),
            Direction::South => state.set_value(&Self::SOUTH, wall_side),
            Direction::West => state.set_value(&Self::WEST, wall_side),
            _ => state,
        }
    }

    /// Recalculates the UP property based on current connections.
    fn update_post(state: BlockStateId) -> BlockStateId {
        let north: WallSide = state.get_value(&Self::NORTH);
        let east: WallSide = state.get_value(&Self::EAST);
        let south: WallSide = state.get_value(&Self::SOUTH);
        let west: WallSide = state.get_value(&Self::WEST);

        let north_connected = north != WallSide::None;
        let east_connected = east != WallSide::None;
        let south_connected = south != WallSide::None;
        let west_connected = west != WallSide::None;

        let up = Self::should_raise_post(
            north_connected,
            east_connected,
            south_connected,
            west_connected,
        );
        state.set_value(&Self::UP, up)
    }
}

impl BlockBehaviour for WallBlock {
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
        // TODO: Waterlogged

        if direction == Direction::Down {
            return state;
        }
        if direction == Direction::Up {
            return Self::update_post(state);
        }
        Self::update_side(state, neighbor_state, direction)
    }
}
