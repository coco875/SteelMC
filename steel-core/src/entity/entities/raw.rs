//! NBT-preserving fallback entity.

use std::sync::Weak;

use glam::DVec3;
use simdnbt::borrow::NbtCompound as BorrowedNbtCompoundView;
use simdnbt::owned::NbtCompound;
use steel_registry::entity_type::EntityTypeRef;
use steel_utils::{DowncastType, DowncastTypeKey, UuidExt, locks::SyncMutex};
use uuid::Uuid;

use crate::entity::{Entity, EntityBase, EntityBaseLoad};
use crate::world::World;

/// Steel-specific fallback for entity types whose runtime behavior is not implemented yet.
///
/// Vanilla has concrete classes for every entity type. Steel uses this only to preserve
/// worldgen and disk NBT until the corresponding typed implementation is added.
pub struct RawEntity {
    base: EntityBase,
    entity_type: EntityTypeRef,
    data: SyncMutex<NbtCompound>,
}

// SAFETY: This key identifies the Steel fallback implementation, independently
// of the Minecraft entity registry entry stored inside it.
unsafe impl DowncastType for RawEntity {
    const TYPE_KEY: DowncastTypeKey = DowncastTypeKey::new("steel:entity/raw");
}

impl RawEntity {
    /// Creates a fresh raw entity for an entity type Steel cannot behaviorally model yet.
    #[must_use]
    pub fn new(id: i32, position: DVec3, world: Weak<World>, entity_type: EntityTypeRef) -> Self {
        Self {
            base: EntityBase::new(id, position, entity_type.dimensions, world),
            entity_type,
            data: SyncMutex::new(NbtCompound::new()),
        }
    }

    /// Creates a raw entity from base entity data.
    #[must_use]
    pub fn from_saved(load: EntityBaseLoad, entity_type: EntityTypeRef) -> Self {
        Self {
            base: EntityBase::from_load(load, entity_type.dimensions),
            entity_type,
            data: SyncMutex::new(NbtCompound::new()),
        }
    }

    /// Marks a raw mob as persistent when vanilla structure generation would do so.
    pub fn set_persistence_required(&self) {
        self.data.lock().insert("PersistenceRequired", 1_i8);
    }
}

impl Entity for RawEntity {
    fn base(&self) -> &EntityBase {
        &self.base
    }

    fn entity_type(&self) -> EntityTypeRef {
        self.entity_type
    }

    fn spawn_data(&self) -> i32 {
        let is_painting = self.entity_type.key.namespace == "minecraft"
            && self.entity_type.key.path == "painting";
        let is_glow_item_frame = self.entity_type.key.namespace == "minecraft"
            && self.entity_type.key.path == "glow_item_frame";

        if is_painting {
            let data = self.data.lock();
            let facing_2d = data
                .byte("facing")
                .map(i32::from)
                .or_else(|| data.int("facing"));

            if let Some(facing) = facing_2d {
                match facing {
                    1 => 4, // West
                    2 => 2, // North
                    3 => 5, // East
                    _ => 3, // South (and default fallback)
                }
            } else {
                3 // Default to South
            }
        } else if is_glow_item_frame {
            let data = self.data.lock();
            let facing_3d = data
                .byte("Facing")
                .map(i32::from)
                .or_else(|| data.int("Facing"));
            facing_3d.unwrap_or(3) // Default to South
        } else {
            0
        }
    }

    fn tick(&self) {
        // TODO: Replace raw entity ticking with full vanilla behavior for this entity type.
    }

    fn attackable(&self) -> bool {
        false
    }

    fn projectile_owner_uuid(&self) -> Option<Uuid> {
        if !self.entity_type.is_projectile {
            return None;
        }

        self.projectile_owner_uuid_from_nbt()
    }

    fn load_additional(&self, nbt: BorrowedNbtCompoundView<'_, '_>) {
        *self.data.lock() = nbt.to_owned();
    }

    fn save_additional(&self, nbt: &mut NbtCompound) {
        *nbt = self.data.lock().clone();
    }
}

impl RawEntity {
    fn projectile_owner_uuid_from_nbt(&self) -> Option<Uuid> {
        let data = self.data.lock();
        let owner = data.int_array("Owner")?;
        Uuid::from_int_array(owner)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Weak;

    use glam::DVec3;
    use simdnbt::owned::NbtTag;
    use steel_registry::vanilla_entities;
    use steel_utils::UuidExt;
    use uuid::Uuid;

    use crate::entity::Entity;

    use super::RawEntity;

    #[test]
    fn raw_projectile_reads_vanilla_owner_uuid() {
        let owner = Uuid::from_u128(42);
        let entity = RawEntity::new(1, DVec3::ZERO, Weak::new(), &vanilla_entities::ENDER_PEARL);
        entity
            .data
            .lock()
            .insert("Owner", NbtTag::IntArray(owner.to_int_array().to_vec()));

        assert_eq!(entity.projectile_owner_uuid(), Some(owner));
    }

    #[test]
    fn test_painting_spawn_data() {
        let painting = RawEntity::new(1, DVec3::ZERO, Weak::new(), &vanilla_entities::PAINTING);

        assert_eq!(painting.spawn_data(), 3);

        {
            let mut data = painting.data.lock();
            data.insert("facing", NbtTag::Byte(1));
        }
        assert_eq!(painting.spawn_data(), 4);
    }

    #[test]
    fn test_glow_item_frame_spawn_data() {
        let glow_item_frame = RawEntity::new(
            1,
            DVec3::ZERO,
            Weak::new(),
            &vanilla_entities::GLOW_ITEM_FRAME,
        );

        assert_eq!(glow_item_frame.spawn_data(), 3);

        {
            let mut data = glow_item_frame.data.lock();
            data.insert("Facing", NbtTag::Byte(2));
        }
        assert_eq!(glow_item_frame.spawn_data(), 2);
    }
}
