use steel_utils::{BlockPos, SectionPos};

use super::{
    DATA_LAYER_BLOCK_COUNT, DATA_LAYER_EDGE, LightSectionRange,
};

/// Signed axis offset clamped to the packed block-light vector range.
pub const BLOCK_LIGHT_VECTOR_AXIS_MIN: i8 = -15;
/// Signed axis offset clamped to the packed block-light vector range.
pub const BLOCK_LIGHT_VECTOR_AXIS_MAX: i8 = 15;

const AXIS_BITS: u32 = 5;
const AXIS_MASK: u8 = (1 << AXIS_BITS) - 1;
const AXIS_SIGN_BIT: u8 = 1 << (AXIS_BITS - 1);

/// Offset from an illuminated block toward its dominant block-light source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BlockLightVector {
    /// X offset toward the source block.
    pub dx: i8,
    /// Y offset toward the source block.
    pub dy: i8,
    /// Z offset toward the source block.
    pub dz: i8,
}

impl BlockLightVector {
    /// Zero offset: the block is its own block-light source.
    pub const ZERO: Self = Self {
        dx: 0,
        dy: 0,
        dz: 0,
    };

    /// Creates a clamped offset vector from integer deltas.
    #[must_use]
    pub fn from_offset(dx: i32, dy: i32, dz: i32) -> Self {
        Self {
            dx: clamp_axis(dx),
            dy: clamp_axis(dy),
            dz: clamp_axis(dz),
        }
    }

    /// Creates a vector pointing from `from` toward `to`.
    #[must_use]
    pub fn from_positions(from: BlockPos, to: BlockPos) -> Self {
        Self::from_offset(
            to.x() - from.x(),
            to.y() - from.y(),
            to.z() - from.z(),
        )
    }

    /// Chains this vector (stored at `intermediate`) through one more hop to `from`.
    ///
    /// Used when block light spreads from an indirect block rather than the emitter.
    #[must_use]
    pub fn chained_through(self, from: BlockPos, intermediate: BlockPos) -> Self {
        let step = Self::from_positions(from, intermediate);
        Self::from_offset(
            i32::from(self.dx) + i32::from(step.dx),
            i32::from(self.dy) + i32::from(step.dy),
            i32::from(self.dz) + i32::from(step.dz),
        )
    }

    /// Returns the world position of the recorded source block.
    #[must_use]
    pub fn source_position(self, from: BlockPos) -> BlockPos {
        from.offset(self.dx as i32, self.dy as i32, self.dz as i32)
    }

    /// Returns true when every axis offset is zero.
    #[must_use]
    pub const fn is_zero(self) -> bool {
        self.dx == 0 && self.dy == 0 && self.dz == 0
    }
}

/// Packed per-block block-light source vectors for one 16x16x16 section.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BlockLightVectorSectionData {
    /// One vector applies to the whole section.
    Homogeneous(BlockLightVector),
    /// Per-block packed vectors.
    Packed(Box<[u16; DATA_LAYER_BLOCK_COUNT]>),
}

impl BlockLightVectorSectionData {
    /// Creates a homogeneous section filled with one vector.
    #[must_use]
    pub const fn homogeneous(vector: BlockLightVector) -> Self {
        Self::Homogeneous(vector)
    }

    /// Returns the vector at local section coordinates.
    #[must_use]
    pub fn get(&self, x: usize, y: usize, z: usize) -> BlockLightVector {
        debug_assert!(x < DATA_LAYER_EDGE);
        debug_assert!(y < DATA_LAYER_EDGE);
        debug_assert!(z < DATA_LAYER_EDGE);

        match self {
            Self::Homogeneous(vector) => *vector,
            Self::Packed(data) => unpack_vector(data[Self::index(x, y, z)]),
        }
    }

    /// Sets the vector at local section coordinates.
    pub fn set(&mut self, x: usize, y: usize, z: usize, vector: BlockLightVector) {
        debug_assert!(x < DATA_LAYER_EDGE);
        debug_assert!(y < DATA_LAYER_EDGE);
        debug_assert!(z < DATA_LAYER_EDGE);

        let index = Self::index(x, y, z);
        match self {
            Self::Homogeneous(default_vector) => {
                let mut data = Box::new([pack_vector(*default_vector); DATA_LAYER_BLOCK_COUNT]);
                data[index] = pack_vector(vector);
                *self = Self::Packed(data);
            }
            Self::Packed(data) => data[index] = pack_vector(vector),
        }
    }

    /// Fills the whole section with one vector.
    pub fn fill(&mut self, vector: BlockLightVector) {
        *self = Self::homogeneous(vector);
    }

    /// Returns true when every stored vector is zero.
    #[must_use]
    pub fn is_all_zero(&self) -> bool {
        match self {
            Self::Homogeneous(vector) => vector.is_zero(),
            Self::Packed(data) => data.iter().all(|packed| unpack_vector(*packed).is_zero()),
        }
    }

    /// Returns packed `u16` values for persistence.
    #[must_use]
    pub fn to_packed_u16s(&self) -> Box<[u16; DATA_LAYER_BLOCK_COUNT]> {
        match self {
            Self::Homogeneous(vector) => Box::new([pack_vector(*vector); DATA_LAYER_BLOCK_COUNT]),
            Self::Packed(data) => Box::new(**data),
        }
    }

    const fn index(x: usize, y: usize, z: usize) -> usize {
        y << 8 | z << 4 | x
    }
}

/// Chunk-owned block-light source vector sections, indexed like block-light layers.
#[derive(Debug)]
pub struct BlockLightVectorStorage {
    range: LightSectionRange,
    sections: Box<[BlockLightVectorSection]>,
}

impl BlockLightVectorStorage {
    /// Creates missing vector sections for every block-light section in a chunk.
    #[must_use]
    pub fn new(range: LightSectionRange) -> Self {
        let sections = (0..range.section_count())
            .map(|_| BlockLightVectorSection::Missing)
            .collect();
        Self { range, sections }
    }

    /// Returns the vertical light-section range.
    #[must_use]
    pub const fn range(&self) -> LightSectionRange {
        self.range
    }

    /// Returns all vector sections.
    #[must_use]
    pub fn sections(&self) -> &[BlockLightVectorSection] {
        &self.sections
    }

    /// Returns all vector sections mutably.
    #[must_use]
    pub fn sections_mut(&mut self) -> &mut [BlockLightVectorSection] {
        &mut self.sections
    }

    /// Returns the vector for one block position.
    #[must_use]
    pub fn get(&self, block_pos: BlockPos) -> BlockLightVector {
        let section_y = SectionPos::block_to_section_coord(block_pos.y());
        let section_index = self.range.section_index(section_y);
        let section = section_index.and_then(|index| self.sections.get(index));
        let data = section.and_then(|section| section.present_data());
        if let Some(data) = data {
            let local_x = section_relative_coord(block_pos.x());
            let local_y = section_relative_coord(block_pos.y());
            let local_z = section_relative_coord(block_pos.z());
            return data.get(local_x, local_y, local_z);
        }

        BlockLightVector::ZERO
    }
}

/// One chunk-owned block-light vector section.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BlockLightVectorSection {
    /// No vector data exists for this section.
    Missing,
    /// Vector data exists for this section.
    Present(BlockLightVectorSectionData),
}

impl BlockLightVectorSection {
    /// Creates a missing vector section.
    #[must_use]
    pub const fn missing() -> Self {
        Self::Missing
    }

    /// Returns present section data.
    #[must_use]
    pub const fn present_data(&self) -> Option<&BlockLightVectorSectionData> {
        match self {
            Self::Missing => None,
            Self::Present(data) => Some(data),
        }
    }
}

const fn encode_axis(axis: i8) -> u8 {
    (axis as u8) & AXIS_MASK
}

const fn decode_axis(field: u8) -> i8 {
    let field = field & AXIS_MASK;
    if field & AXIS_SIGN_BIT != 0 {
        (field as i8) - (1 << AXIS_BITS)
    } else {
        field as i8
    }
}

fn clamp_axis(value: i32) -> i8 {
    if value < -15 {
        -15
    } else if value > 15 {
        15
    } else {
        value as i8
    }
}

fn pack_vector(vector: BlockLightVector) -> u16 {
    u16::from(encode_axis(vector.dx))
        | (u16::from(encode_axis(vector.dy)) << AXIS_BITS)
        | (u16::from(encode_axis(vector.dz)) << (AXIS_BITS * 2))
}

fn unpack_vector(packed: u16) -> BlockLightVector {
    BlockLightVector {
        dx: decode_axis((packed & u16::from(AXIS_MASK)) as u8),
        dy: decode_axis(((packed >> AXIS_BITS) & u16::from(AXIS_MASK)) as u8),
        dz: decode_axis(((packed >> (AXIS_BITS * 2)) & u16::from(AXIS_MASK)) as u8),
    }
}

const fn section_relative_coord(block_coord: i32) -> usize {
    (block_coord & 15) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_round_trips_signed_offsets() {
        for axis in -15..=15 {
            let vector = BlockLightVector {
                dx: axis,
                dy: -axis,
                dz: 0,
            };
            assert_eq!(unpack_vector(pack_vector(vector)), vector);
        }
    }

    #[test]
    fn pack_round_trips_max_negative_offset() {
        let vector = BlockLightVector {
            dx: 0,
            dy: 0,
            dz: -15,
        };
        assert_eq!(unpack_vector(pack_vector(vector)), vector);
    }

    #[test]
    fn chained_through_accumulates_hops_toward_emitter() {
        let emitter = BlockPos::new(1, 1, 1);
        let intermediate = BlockPos::new(2, 1, 1);
        let target = BlockPos::new(3, 1, 1);

        let at_intermediate = BlockLightVector::from_positions(intermediate, emitter);
        assert_eq!(
            at_intermediate.chained_through(target, intermediate),
            BlockLightVector {
                dx: -2,
                dy: 0,
                dz: 0,
            }
        );
    }

    #[test]
    fn from_positions_clamps_large_offsets() {
        let from = BlockPos::new(0, 0, 0);
        let to = BlockPos::new(20, -20, 3);
        let vector = BlockLightVector::from_positions(from, to);
        assert_eq!(vector.dx, 15);
        assert_eq!(vector.dy, -15);
        assert_eq!(vector.dz, 3);
    }

    #[test]
    fn section_data_set_promotes_homogeneous_to_packed() {
        let mut data = BlockLightVectorSectionData::homogeneous(BlockLightVector::ZERO);
        data.set(1, 2, 3, BlockLightVector {
            dx: 1,
            dy: -2,
            dz: 3,
        });
        assert!(matches!(data, BlockLightVectorSectionData::Packed(_)));
        assert_eq!(
            data.get(1, 2, 3),
            BlockLightVector {
                dx: 1,
                dy: -2,
                dz: 3,
            }
        );
        assert_eq!(data.get(0, 0, 0), BlockLightVector::ZERO);
    }
}
